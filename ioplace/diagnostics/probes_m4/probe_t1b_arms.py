"""M4 T1b (a): four-arm A/B probe (design draft `docs/superpowers/specs/
2026-08-13-m4-scale-up-design-draft.md` sec 1.4 B1 / T1b row (a)):
runs the SAME small GP-only DREAMPlace workload as two "arms" under four
different inter-arm protocols, to tell apart B1's two live hypotheses for
the M2 `peak_mem_mb` fixed-step contamination:

  - "cumulative high-water mark": `torch.cuda.max_memory_allocated()`'s
    peak-tracking COUNTER is never reset between arms in the same process,
    so arm2's peak reading is just the process-lifetime max over both
    arms. This hypothesis predicts `reset_only` (calling ONLY
    `torch.cuda.reset_peak_memory_stats()` between arms -- exactly what
    T1's per-phase reset already does) is a complete fix.
  - "retained tensors" (v2's adjudicated cause, sec 1.4 B1: "更像是每臂
    結束後仍存活的張量... 單靠 reset 不會清掉仍存活的 allocation"): arm1
    leaves live CUDA ALLOCATIONS behind (reference cycles, a cached
    buffer, missing teardown) that `reset_peak_memory_stats()` cannot
    free -- it only zeroes the HWM counter, not any tensor. This
    hypothesis predicts `reset_only` does NOT bring arm2's run-start
    `memory_allocated()`/`memory_reserved()` baseline back down, while
    `teardown_gc` (explicit `del` of every object the arm created +
    `gc.collect()` + `torch.cuda.empty_cache()`) does.

Four protocols, each running the identical 2-arm workload (same config,
same iteration count, same seed for both arms):
  - `fresh_subprocess`: each arm is its OWN subprocess (`--single-arm`) --
    the golden reference every other protocol is compared against; arm2
    can only "see" whatever a brand-new process starts with.
  - `same_process`: both arms in one process, literally nothing done
    between them -- the unfixed M2 driver behavior (`run_placement.py:129`,
    `run_placement_io.py:202`) that produced B1's fixed-step artifacts.
  - `reset_only`: same process, ONLY `torch.cuda.reset_peak_memory_stats()`
    called between arms -- "T1 already landed" per-phase reset, tested
    here in isolation to see whether it alone is sufficient.
  - `teardown_gc`: same process, explicit `del` of every live object the
    arm created + `gc.collect()` + `torch.cuda.empty_cache()` between
    arms. Deliberately does NOT also call `reset_peak_memory_stats()` --
    this is a disclosed ablation of teardown ALONE, so its raw
    `peak_alloc_gb`/`peak_reserved_gb` numbers remain process-lifetime HWM
    (>= arm1's own peak) even when teardown fully frees arm1's tensors;
    the diagnostic signal for THIS protocol is `run_start_allocated_gb`/
    `run_start_reserved_gb`, not the peak fields (see `judgment`'s
    `arm2_baseline_delta_*` below).

To keep the four protocols from contaminating EACH OTHER (running
same_process/reset_only/teardown_gc back-to-back in one big process would
just reproduce the same "no isolation between measurements" bug this probe
exists to diagnose), the top-level orchestrator (`run`) launches every
protocol as its own subprocess too -- `fresh_subprocess` launches two
`--single-arm` subprocesses directly (its 2 arms ARE the isolation unit);
`same_process`/`reset_only`/`teardown_gc` each launch one `--run-protocol`
subprocess, inside which both of that protocol's arms run.

Every arm, in every protocol, records at its own first/last line:
  - `run_start_allocated_gb`/`run_start_reserved_gb`: `torch.cuda.
    memory_allocated()`/`memory_reserved()` -- LIVE occupancy, taken
    before the arm does anything of its own. This is the primary B1
    diagnostic: retained tensors show up here as an elevated arm2
    baseline relative to arm1's, independent of whether the peak counter
    was ever reset.
  - `run_end_allocated_gb`/`run_end_reserved_gb`: the same two calls at
    the arm's last line.
  - `peak_alloc_gb`/`peak_reserved_gb`: `torch.cuda.max_memory_allocated()`/
    `max_memory_reserved()`, read ONCE at the arm's end, deliberately
    NEVER reset by `run_one_arm` itself (whether/when a reset happens is
    entirely the calling protocol's own between-arm hook -- see the
    protocol table above; that dependence on protocol IS the experiment).

`judgment` (assembled by `build_judgment`) reduces the raw per-protocol
data to: each protocol's `arm2_baseline_delta_gb`/`_pct` (arm2's run-start
`memory_allocated_gb` vs arm1's), `reset_only_fixes_baseline` (does
`reset_only` alone bring that delta within +-2%?), and
`teardown_within_2pct_tolerance` (does `teardown_gc`?) -- sec 1.4 B1's
"正式 ablation 改每臂一個 subprocess,除非 A/B 證明 teardown 後 baseline
回到 ±2% 容忍" exemption test.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_t1b_arms \\
        --config /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/test/ispd2005/adaptec1.json \\
        --iterations 100 --out results/m4/profile/t1b_arms.json
"""
import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import tempfile
import uuid

import torch

from ioplace.profile import env_metadata

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
DEFAULT_PYTHON = os.path.join(DP, ".venv312", "bin", "python")
# design draft T1b row (a)'s own "同一小 case" -- this repo has no adaptec1
# config under benchmarks/ (only the ispd2005/ispd2015/ispd25 M4 corpus
# configs); adaptec1's existing, already-verified-small config is the one
# every other M4 probe/test in this repo already points at (`probe_gp_
# memory.py`'s own usage docstring, `test_reweight.py`, `test_evaluator_
# gpu.py`).
DEFAULT_CONFIG = os.path.join(DP, "install", "test", "ispd2005", "adaptec1.json")
DEFAULT_OUT = os.path.join(REPO, "results", "m4", "profile", "t1b_arms.json")
DEFAULT_ITERATIONS = 100

ARM_FIELDS = ("run_start_allocated_gb", "run_start_reserved_gb",
             "run_end_allocated_gb", "run_end_reserved_gb",
             "peak_alloc_gb", "peak_reserved_gb")

PROTOCOLS = ("fresh_subprocess", "same_process", "reset_only", "teardown_gc")
IN_PROCESS_PROTOCOLS = ("same_process", "reset_only", "teardown_gc")

BASELINE_DELTA_TOLERANCE_PCT = 2.0


def _mem_gb():
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.memory_reserved() / 2**30,
    }


def _prepared_config(config_json, iterations):
    """Copies `config_json` to a fresh temp file with
    `global_place_stages[0].iteration` overwritten to `iterations` (same
    technique as `probe_lifetime_gate._prepared_config` -- this probe only
    needs each arm to run a small, fixed number of GP iterations, not
    converge). Caller is responsible for deleting the returned path."""
    with open(config_json) as f:
        cfg = json.load(f)
    cfg["global_place_stages"][0]["iteration"] = iterations
    fd, path = tempfile.mkstemp(suffix=".json", prefix="t1b_arm_cfg_")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


def run_one_arm(config_json, iterations, seed=0):
    """Runs one GP-only arm in THIS process (`_load_dreamplace` +
    `_place`, the same entry points `probe_gp_memory.py`/the real drivers
    use). Returns `(record, live_objects)`:
      - `record`: dict with `ARM_FIELDS` -- see module docstring for exact
        semantics.
      - `live_objects`: `{"params", "placedb", "placer", "metrics"}`, this
        arm's own top-level references -- a caller running the
        `teardown_gc` protocol `del`s these (plus drops this dict itself)
        before `gc.collect()`/`empty_cache()`.

    Deliberately does NOT call `torch.cuda.reset_peak_memory_stats()`
    anywhere -- see module docstring's `peak_alloc_gb`/`peak_reserved_gb`
    note; whether/when that reset happens is the calling protocol's own
    between-arm hook, not this function's concern.
    """
    from ioplace.drivers.run_placement import _load_dreamplace, _place

    tmp_config = _prepared_config(config_json, iterations)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = _mem_gb()
        params, placedb = _load_dreamplace(tmp_config)
        params.random_seed = seed
        params.deterministic_flag = 1
        placedb.initialize(params)
        placer, metrics = _place(params, placedb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        peak_alloc_gb = (torch.cuda.max_memory_allocated() / 2**30
                         if torch.cuda.is_available() else 0.0)
        peak_reserved_gb = (torch.cuda.max_memory_reserved() / 2**30
                            if torch.cuda.is_available() else 0.0)
        end = _mem_gb()
    finally:
        os.remove(tmp_config)

    record = {
        "run_start_allocated_gb": start["allocated_gb"],
        "run_start_reserved_gb": start["reserved_gb"],
        "run_end_allocated_gb": end["allocated_gb"],
        "run_end_reserved_gb": end["reserved_gb"],
        "peak_alloc_gb": peak_alloc_gb, "peak_reserved_gb": peak_reserved_gb,
    }
    live_objects = {"params": params, "placedb": placedb,
                    "placer": placer, "metrics": metrics}
    return record, live_objects


# ---------------------------------------------------------------------------
# protocol between-arm hooks
# ---------------------------------------------------------------------------

def _between_same_process(live_objects):
    """Literally nothing -- the unfixed M2 driver behavior."""


def _between_reset_only(live_objects):
    """ONLY `reset_peak_memory_stats()` -- "T1 already landed" per-phase
    reset, in isolation."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _between_teardown_gc(live_objects):
    """Explicit `del` of every object this arm created + `gc.collect()` +
    `torch.cuda.empty_cache()` -- deliberately no `reset_peak_memory_
    stats()` call (see module docstring's `teardown_gc` row)."""
    for k in list(live_objects.keys()):
        del live_objects[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_BETWEEN_ARM_HOOKS = {
    "same_process": _between_same_process,
    "reset_only": _between_reset_only,
    "teardown_gc": _between_teardown_gc,
}


def run_protocol_in_process(protocol, config_json, iterations, seed):
    """Runs both arms of `protocol` (one of `IN_PROCESS_PROTOCOLS`) in
    THIS process, applying `protocol`'s between-arm hook after arm1.
    Returns `{"arm1": record, "arm2": record}`."""
    hook = _BETWEEN_ARM_HOOKS[protocol]
    arm1, live1 = run_one_arm(config_json, iterations, seed)
    hook(live1)
    arm2, _live2 = run_one_arm(config_json, iterations, seed)
    return {"arm1": arm1, "arm2": arm2}


# ---------------------------------------------------------------------------
# subprocess launches (fresh_subprocess's 2 arms; every protocol's own
# top-level isolation from the OTHER protocols)
# ---------------------------------------------------------------------------

def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run_arm_subprocess(python, config_json, iterations, seed, out_path,
                        timeout_s=600.0):
    """Launches `--single-arm` as its own subprocess; returns the one-arm
    record it wrote to `out_path`."""
    cmd = [python, "-m", "ioplace.diagnostics.probes_m4.probe_t1b_arms",
          "--single-arm", "--config", config_json, "--iterations", str(iterations),
          "--seed", str(seed), "--out", out_path]
    subprocess.run(cmd, cwd=REPO, env=_subprocess_env(), check=True,
                   timeout=timeout_s, capture_output=True, text=True)
    with open(out_path) as f:
        return json.load(f)


def _run_protocol_subprocess(python, protocol, config_json, iterations, seed,
                             out_path, timeout_s=600.0):
    """Launches `--run-protocol <protocol>` as its own subprocess (keeps
    this protocol's 2-arm measurement isolated from the OTHER 3
    protocols' own process state); returns `{"arm1":..., "arm2":...}`."""
    cmd = [python, "-m", "ioplace.diagnostics.probes_m4.probe_t1b_arms",
          "--run-protocol", protocol, "--config", config_json,
          "--iterations", str(iterations), "--seed", str(seed), "--out", out_path]
    subprocess.run(cmd, cwd=REPO, env=_subprocess_env(), check=True,
                   timeout=timeout_s, capture_output=True, text=True)
    with open(out_path) as f:
        return json.load(f)


def _collect_protocol(protocol, config_json, iterations, seed, python, tmp_dir):
    if protocol == "fresh_subprocess":
        arm1 = _run_arm_subprocess(python, config_json, iterations, seed,
                                   os.path.join(tmp_dir, "fresh_arm1.json"))
        arm2 = _run_arm_subprocess(python, config_json, iterations, seed,
                                   os.path.join(tmp_dir, "fresh_arm2.json"))
        return {"arm1": arm1, "arm2": arm2}
    return _run_protocol_subprocess(python, protocol, config_json, iterations, seed,
                                    os.path.join(tmp_dir, f"{protocol}.json"))


# ---------------------------------------------------------------------------
# judgment
# ---------------------------------------------------------------------------

def _pct_delta(baseline_gb, other_gb):
    """`None` (not a divide-by-zero) when `baseline_gb` is ~0 -- an
    absolute-GB-only comparison is still available via
    `arm2_baseline_delta_gb`."""
    if baseline_gb is not None and abs(baseline_gb) > 1e-9:
        return (other_gb - baseline_gb) / baseline_gb * 100.0
    return None


def build_judgment(protocols):
    """Sec 1.4 B1's cause adjudication, purely from each protocol's
    `arm1`/`arm2` `run_start_allocated_gb` (see module docstring). Returns
    a dict keyed by protocol name (each `{"arm2_baseline_delta_gb",
    "arm2_baseline_delta_pct"}`) plus three top-level verdict fields:
      - `reset_only_fixes_baseline`: does `reset_only`'s arm2 baseline sit
        within `BASELINE_DELTA_TOLERANCE_PCT` of arm1's? (cumulative-HWM
        hypothesis's prediction)
      - `teardown_within_2pct_tolerance`: same check for `teardown_gc`
        (retained-tensors hypothesis's prediction, and sec 1.4 B1's own
        "正式 ablation 改每臂一個 subprocess,除非...teardown 後 baseline
        回到 ±2% 容忍" exemption test).
      - `cause`: `"cumulative_hwm"` if `reset_only` alone already fixes
        it, else `"retained_tensors"` if `teardown_gc` fixes it (but
        `reset_only` didn't), else `"undetermined"` (neither protocol's
        arm2 baseline came back within tolerance -- some other mechanism
        is at play, disclosed rather than forced into one of the two
        named hypotheses).
    """
    judgment = {}
    for name, data in protocols.items():
        a1 = data["arm1"]["run_start_allocated_gb"]
        a2 = data["arm2"]["run_start_allocated_gb"]
        judgment[name] = {
            "arm2_baseline_delta_gb": a2 - a1,
            "arm2_baseline_delta_pct": _pct_delta(a1, a2),
        }

    def _within_tolerance(name):
        pct = judgment.get(name, {}).get("arm2_baseline_delta_pct")
        if pct is not None:
            return abs(pct) <= BASELINE_DELTA_TOLERANCE_PCT
        # baseline ~0 GB -- fall back to the absolute delta, same tolerance
        # magnitude interpreted in GB instead of percent (an all-zero
        # baseline with a near-zero delta is still "within tolerance").
        delta_gb = judgment.get(name, {}).get("arm2_baseline_delta_gb")
        return delta_gb is not None and abs(delta_gb) <= 0.02

    reset_ok = _within_tolerance("reset_only")
    teardown_ok = _within_tolerance("teardown_gc")
    judgment["reset_only_fixes_baseline"] = reset_ok
    judgment["teardown_within_2pct_tolerance"] = teardown_ok
    if reset_ok:
        judgment["cause"] = "cumulative_hwm"
    elif teardown_ok:
        judgment["cause"] = "retained_tensors"
    else:
        judgment["cause"] = "undetermined"
    return judgment


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------

def run(config_json=DEFAULT_CONFIG, iterations=DEFAULT_ITERATIONS, seed=0,
       python=DEFAULT_PYTHON, out_json=None):
    """Runs all four protocols (each as its own subprocess -- see module
    docstring) and writes the assembled T1b (a) record to `out_json`
    (default `results/m4/profile/t1b_arms.json`). Returns the record dict
    (also written to disk)."""
    out_json = out_json or DEFAULT_OUT
    with tempfile.TemporaryDirectory(prefix="t1b_arms_") as tmp_dir:
        protocols = {
            name: _collect_protocol(name, config_json, iterations, seed, python, tmp_dir)
            for name in PROTOCOLS
        }

    judgment = build_judgment(protocols)
    env = env_metadata(REPO, DP, input_paths=(config_json,))
    record = {
        "record_kind": "t1b_arms",
        "config": config_json, "iterations": iterations, "seed": seed,
        "protocol_order": list(PROTOCOLS),
        "protocols": protocols,
        "judgment": judgment,
        "run_id": str(uuid.uuid4()), "repo_commit": env["ioplace_commit"],
        "dp_commit": env["dp_commit"], "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "input_sha256": env["input_sha256"],
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    print(f"[probe_t1b_arms] cause={judgment['cause']} "
         f"reset_only_fixes_baseline={judgment['reset_only_fixes_baseline']} "
         f"teardown_within_2pct_tolerance={judgment['teardown_within_2pct_tolerance']} "
         f"wrote {out_json}")
    return record


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--out", default=None)
    ap.add_argument("--single-arm", action="store_true",
                    help="internal: run exactly one arm in this process, "
                         "write its record to --out")
    ap.add_argument("--run-protocol", choices=IN_PROCESS_PROTOCOLS, default=None,
                    help="internal: run both arms of this protocol in this "
                         "process, write {arm1,arm2} to --out")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.single_arm:
        record, _live = run_one_arm(args.config, args.iterations, args.seed)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)
        return 0

    if args.run_protocol:
        result = run_protocol_in_process(args.run_protocol, args.config,
                                         args.iterations, args.seed)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        return 0

    run(args.config, args.iterations, args.seed, args.python, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
