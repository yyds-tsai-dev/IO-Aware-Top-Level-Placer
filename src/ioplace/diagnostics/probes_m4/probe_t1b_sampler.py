"""M4 T1b (e): sampler precision probe (design draft `docs/superpowers/
specs/2026-08-13-m4-scale-up-design-draft.md` sec 6.1/1.4 T1b row (e)):
`profile.DeviceMemSampler`'s default 0.5s polling of `mem_get_info()` can
miss any usage spike narrower than its sampling interval -- this probe
manufactures artificial short GPU-memory spikes of known duration
(allocate a fixed-size tensor, hold it for `spike_s` seconds, free it) and
measures how often a 0.5s sampler actually catches one, at spike widths of
0.05s/0.1s/0.5s, 20 repeats each (module docstring's own acceptance: "漏檢
率有數").

A trial is "detected" iff the sampler's HWM rose by at least
`detect_frac` (default 50%) of the spike's own intended allocation size
above its pre-spike baseline (`run_one_trial`) -- a partial catch (the
background thread happened to sample mid-ramp) still counts, an
essentially-zero delta (the spike started and ended entirely between two
samples) does not. `torch.cuda.empty_cache()` immediately after freeing
each spike's tensor is deliberate: PyTorch's caching allocator would
otherwise keep the block in its own reserved pool rather than returning it
to the driver, so `mem_get_info()` (whole-device, driver-level) would
never see the baseline actually drop back down between repeats.

`run_one_trial`'s sampler argument only needs `.reset_hwm()` and a
`.device_used_gb` property (optionally `._sample_once()`, called right
after the spike ends to catch a same-thread reading the way `DeviceMemSampler.
stop()` already does) -- the real run uses `profile.DeviceMemSampler`
itself; CPU tests inject a scripted fake with no CUDA dependency at all
(see `tests/test_probe_t1b_sampler.py`).

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_t1b_sampler \\
        --interval-s 0.5 --size-gb 0.25 --n-repeats 20 \\
        --out results/m4/profile/t1b_sampler.json
"""
import argparse
import json
import os
import socket
import sys
import time
import uuid

import torch

from ioplace.profile import env_metadata

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
DEFAULT_OUT = os.path.join(REPO, "results", "m4", "profile", "t1b_sampler.json")

# design draft T1b (e)'s three probe widths.
SPIKE_DURATIONS_S = (0.05, 0.1, 0.5)
N_REPEATS = 20
SPIKE_SIZE_GB = 0.25
DETECT_FRAC = 0.5


def _alloc_and_free_spike(size_gb, spike_s):
    """Real (CUDA-only) spike source: allocate `size_gb` GiB, hold it for
    `spike_s` seconds, free it and force the caching allocator to hand the
    block back to the driver (`empty_cache()` -- see module docstring)."""
    n_bytes = int(size_gb * 2**30)
    x = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    time.sleep(spike_s)
    del x
    torch.cuda.empty_cache()


def run_one_trial(sampler, spike_fn, size_gb, detect_frac=DETECT_FRAC):
    """Runs one allocate/hold/free trial against `sampler` (`.reset_hwm()`
    + `.device_used_gb`, optionally `._sample_once()`). Returns
    `{"baseline_before_gb", "peak_after_gb", "delta_gb", "detected"}`.
    `spike_fn()` is called with no arguments and must block for the
    spike's own duration (real usage: `_alloc_and_free_spike` bound via a
    closure/`functools.partial`; tests: a fake that just advances the
    fake sampler's scripted clock)."""
    sampler.reset_hwm()
    baseline_before_gb = sampler.device_used_gb
    spike_fn()
    sample_once = getattr(sampler, "_sample_once", None)
    if sample_once is not None:
        # Mirrors DeviceMemSampler.stop()'s "catch any peak between last
        # sample and stop()" -- without this, a spike that ends between
        # two background-thread samples could be missed even though it
        # was, in principle, long enough for the *next* sample to catch.
        sample_once()
    peak_after_gb = sampler.device_used_gb
    delta_gb = peak_after_gb - baseline_before_gb
    detected = delta_gb >= size_gb * detect_frac
    return {
        "baseline_before_gb": baseline_before_gb,
        "peak_after_gb": peak_after_gb,
        "delta_gb": delta_gb,
        "detected": bool(detected),
    }


def run_sampler_probe(spike_durations=SPIKE_DURATIONS_S, n_repeats=N_REPEATS,
                      size_gb=SPIKE_SIZE_GB, interval_s=0.5, detect_frac=DETECT_FRAC,
                      sampler_cls=None, spike_fn_factory=None):
    """Runs `n_repeats` trials at each of `spike_durations`, against one
    `sampler_cls(interval_s=interval_s)` instance shared across every
    trial (default: `profile.DeviceMemSampler` -- injectable for CPU
    tests). `spike_fn_factory(size_gb, spike_s) -> spike_fn` defaults to
    `_alloc_and_free_spike` bound via a closure; tests inject a fake that
    advances a scripted clock instead of touching CUDA.

    Returns `{str(spike_s): {"spike_s", "n_repeats", "n_detected",
    "detection_rate", "miss_rate", "trials": [...]}, ...}`.
    """
    if sampler_cls is None:
        from ioplace.profile import DeviceMemSampler
        sampler_cls = DeviceMemSampler
    if spike_fn_factory is None:
        spike_fn_factory = lambda sz, dur: (lambda: _alloc_and_free_spike(sz, dur))  # noqa: E731

    sampler = sampler_cls(interval_s=interval_s)
    sampler.start()
    try:
        results = {}
        for spike_s in spike_durations:
            spike_fn = spike_fn_factory(size_gb, spike_s)
            trials = [run_one_trial(sampler, spike_fn, size_gb, detect_frac)
                     for _ in range(n_repeats)]
            n_detected = sum(1 for t in trials if t["detected"])
            results[str(spike_s)] = {
                "spike_s": spike_s, "n_repeats": n_repeats, "n_detected": n_detected,
                "detection_rate": n_detected / n_repeats if n_repeats else 0.0,
                "miss_rate": (n_repeats - n_detected) / n_repeats if n_repeats else 0.0,
                "trials": trials,
            }
    finally:
        sampler.stop()
    return results


def run(*, interval_s=0.5, size_gb=SPIKE_SIZE_GB, n_repeats=N_REPEATS,
       spike_durations=SPIKE_DURATIONS_S, out_json=None):
    """Runs the full T1b (e) probe and writes its RESULT-GATE-adjacent
    record to `out_json` (default `results/m4/profile/t1b_sampler.json`).
    Returns the record dict (also written to disk)."""
    out_json = out_json or DEFAULT_OUT
    results = run_sampler_probe(spike_durations=spike_durations, n_repeats=n_repeats,
                                size_gb=size_gb, interval_s=interval_s)
    env = env_metadata(REPO, DP, input_paths=())
    record = {
        "record_kind": "t1b_sampler",
        "interval_s": interval_s, "size_gb": size_gb, "n_repeats": n_repeats,
        "spike_durations_s": list(spike_durations),
        "results": results,
        "run_id": str(uuid.uuid4()), "repo_commit": env["ioplace_commit"],
        "dp_commit": env["dp_commit"], "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    summary = {k: v["detection_rate"] for k, v in results.items()}
    print(f"[probe_t1b_sampler] detection_rate={summary} wrote {out_json}")
    return record


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval-s", type=float, default=0.5)
    ap.add_argument("--size-gb", type=float, default=SPIKE_SIZE_GB)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--spike-durations-s", type=float, nargs="+",
                    default=list(SPIKE_DURATIONS_S))
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(interval_s=args.interval_s, size_gb=args.size_gb, n_repeats=args.n_repeats,
       spike_durations=tuple(args.spike_durations_s), out_json=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
