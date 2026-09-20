"""M4 T1 profiling infrastructure (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 6.1/6.2):
a per-phase GPU-peak/host-RSS timer, a paired CUDA-event iteration timer, a
background `mem_get_info()` device-memory sampler, and the shared JSON
schema every M4 profile record is built from.

Zero new dependencies (sec 6.1): `torch.cuda.mem_get_info()` stands in for
NVML/pynvml and `resource.getrusage` for psutil -- neither is installed and
the host has no network access to add them.

This module only provides the measurement primitives; wiring them into
`ioplace/drivers/run_placement*.py` to actually emit
`results/m4/profile/*.json` is a later M4 task (T8), not part of T1.
"""
import hashlib
import os
import resource
import subprocess
import threading
import time

import torch

# sec 6.1's field groups, flattened into one schema (sec 6.2): every key
# below is present in a build_profile_record(...) output, even when the
# caller has nothing to report for it (None/[]/0.0) -- see
# test_profile_schema_has_all_fields.
PROFILE_SCHEMA_FIELDS = (
    "case", "K", "rtype", "arm",
    "t_read", "t_initialize", "t_gp", "t_lg",
    "t_eval_total", "t_op_total", "t_diag_total", "t_total",
    "iter_ms", "iter_ms_p50", "iter_ms_p90", "iter_ms_p99", "iter_ms_max",
    "op_fwd_ms", "op_bwd_ms", "eval_ms",
    "phases", "device_used_gb", "host_peak_rss_gb",
    "n_movable", "n_physical", "n_filler", "n_nets", "n_pins",
    "lattice", "n_bins", "k_chunk", "n_active",
    "bytes_per_pin_gp", "ns_per_pin_op", "gb_per_mnet_eval",
    "env",
)


def host_rss_gb() -> float:
    """`ru_maxrss` is a **process-lifetime monotonic high-water mark** (in
    KiB on Linux), not a snapshot of current usage and not something that
    can be reset mid-process -- there is no `reset_peak_rss()` analogue to
    `torch.cuda.reset_peak_memory_stats()`. Calling this after phase B can
    only report a value >= what it reported after phase A, even if phase B
    itself freed memory. Sec 6.1's `host_peak_rss_gb` (the run's overall
    peak) is a correct use of that monotonic property; treating a
    same-process call as phase B's *own* peak is not (see
    `_Phase`/`host_rss_hwm_at_phase_end`)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def percentiles_ms(values):
    """p50/p90/p99/max of a list of per-iteration ms samples. Nearest-rank
    (no interpolation) so this has no numpy dependency; empty input ->
    zeros rather than raising (a run with no recorded iterations is a valid,
    if uninteresting, profile)."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)

    def rank(p):
        idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
        return s[idx]

    return {"p50": rank(50), "p90": rank(90), "p99": rank(99), "max": s[-1]}


class EventTimer:
    """Paired CUDA-event timer (sec 6.1): `start()`/`stop()` only call
    `Event.record()` -- safe inside a hot loop, no synchronize(). Every
    pair's elapsed time is computed once, in batch, by `elapsed_ms()` after
    the loop -- the only synchronize() this class ever performs."""

    def __init__(self):
        self._starts = []
        self._stops = []

    def start(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._starts.append(e)
        return self

    def stop(self):
        if len(self._stops) != len(self._starts) - 1:
            raise RuntimeError("EventTimer.stop() without a matching start()")
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._stops.append(e)
        return self

    def elapsed_ms(self):
        """Batch-query every paired (start, stop): one synchronize(), then
        `Event.elapsed_time()` per pair. Raises if a start() is still
        unpaired -- querying mid-pair would silently drop it."""
        if len(self._starts) != len(self._stops):
            raise RuntimeError(
                f"{len(self._starts)} start() vs {len(self._stops)} stop() calls "
                "-- every start() must be paired with stop() before elapsed_ms()")
        if not self._starts:
            return []
        torch.cuda.synchronize()
        return [s.elapsed_time(t) for s, t in zip(self._starts, self._stops)]

    def __len__(self):
        return len(self._stops)


class DeviceMemSampler:
    """Background thread sampling `torch.cuda.mem_get_info()` every
    `interval_s` seconds (sec 6.1's `device_used_gb`) -- catches
    device-level usage (CUDA context + allocator fragmentation) that
    `max_memory_reserved()` cannot see, with no NVML/pynvml dependency.

    Two caveats this class cannot correct for, only report honestly:
      - `total - free` is **whole-GPU** usage, not this process's -- on a
        shared/multi-tenant device, another process's allocations
        contaminate the reading. `device_used_gb` is only a faithful proxy
        for *our* usage when we know we have the device to ourselves (true
        of every case in this repo's benchmark runs, but not a general
        guarantee -- hence the name has no "process" in it).
      - sampling every `interval_s` (default 0.5s) means any usage spike
        narrower than that interval can be missed entirely. `device_used_gb`
        is a lower bound on the true peak, not an exact one.
    """

    def __init__(self, interval_s=0.5, device=None):
        self.interval_s = interval_s
        self.device = device
        self._used_bytes_hwm = 0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread = None

    def _sample_once(self):
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info(self.device)
        used = total - free
        with self._lock:
            self._used_bytes_hwm = max(self._used_bytes_hwm, used)

    def start(self):
        self._sample_once()
        if self._thread is None and self.interval_s is not None and self.interval_s > 0:
            self._stop_evt.clear()

            def _loop():
                while not self._stop_evt.wait(self.interval_s):
                    self._sample_once()

            self._thread = threading.Thread(target=_loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join()
            self._thread = None
        self._sample_once()               # catch any peak between last sample and stop()
        return self

    @property
    def device_used_gb(self) -> float:
        """Running maximum, over every sample taken so far, of whole-device
        `mem_get_info()` usage -- see the class docstring's two caveats."""
        with self._lock:
            return self._used_bytes_hwm / 2**30

    def reset_hwm(self):
        """M4 T2b (`profile_lifetime.LifetimeRecorder.phase_begin`):
        zeroes the running high-water mark and immediately resamples --
        the sampler-thread analogue of `torch.cuda.reset_peak_memory_stats()`,
        so a `LifetimeRecorder` phase's `device_used_peak_gb` reflects only
        that phase's own window, not whatever an earlier phase already
        pushed the HWM to."""
        with self._lock:
            self._used_bytes_hwm = 0
        self._sample_once()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
        return False


class PhaseTimer:
    """Records wall time + a fresh-reset GPU peak (sec 1.4 B1's fix, as a
    reusable primitive outside the drivers -- see the reset caveat below)
    + a host-RSS snapshot for each of sec 6.1's named phases (`read`,
    `initialize`, `gp`, `lg`, ...). `phases[name]` holds `{t_s,
    peak_alloc_gb, peak_reserved_gb, baseline_alloc_gb,
    peak_above_baseline_gb, host_rss_hwm_at_phase_end}` once the
    `with timer.phase(name):` block exits.

    GPU vs host isolation is NOT symmetric here:
      - `peak_alloc_gb`/`peak_reserved_gb` are genuinely phase-scoped:
        `reset_peak_memory_stats()` at phase entry zeroes the allocator's
        high-water-mark *counters*, so these reflect only what happened
        inside this phase's block -- **with one caveat**: the reset clears
        the counters, not any tensor that is still alive and resident from
        an earlier phase (e.g. a cached buffer kept across phases). It is
        necessary but not sufficient for true isolation between arms/runs
        that share a process; a correct per-arm ablation still needs each
        arm in its own subprocess (the pattern `run_ablation_m2.py` does
        not currently follow either -- this is the same gap as driver B1).
        `baseline_alloc_gb` (`torch.cuda.memory_allocated()` at phase entry)
        and `peak_above_baseline_gb` (`peak_alloc_gb - baseline_alloc_gb`)
        make that caveat legible instead of silent: in the main flow, the
        soft phase's own positions are exactly this kind of live-at-entry
        residency that `peak_alloc_gb` alone charges to whichever phase
        opens next.
      - `host_rss_hwm_at_phase_end` is NOT phase-scoped -- see
        `host_rss_gb()`'s docstring. It is the process's cumulative peak
        RSS *as of* this phase's end, not this phase's own contribution.
    """

    def __init__(self, reset_peak=True):
        """`reset_peak=False` (M4 T2b): skip this timer's own
        `reset_peak_memory_stats()`/peak reads entirely -- for driver runs
        where a `profile_lifetime.LifetimeRecorder` is active and has
        taken over GPU-peak accounting via its own `phase_begin`/
        `phase_end` (sec 2.2). Two trackers both calling
        `reset_peak_memory_stats()` on the same process-wide CUDA counter
        would each silently clobber the other's window; `reset_peak=False`
        makes `PhaseTimer` step aside instead."""
        self.phases = {}
        self.reset_peak = reset_peak

    def phase(self, name):
        return _Phase(self, name)


class _Phase:
    def __init__(self, timer, name):
        self._timer = timer
        self._name = name
        self._t0 = None

    def __enter__(self):
        if self._timer.reset_peak and torch.cuda.is_available():
            # Necessary but not sufficient (see PhaseTimer docstring): this
            # zeroes the allocator's peak-tracking counters, not any tensor
            # still resident from a previous phase in the same process.
            torch.cuda.reset_peak_memory_stats()
        # reset_peak_memory_stats() only zeroes the high-water-mark counter,
        # not what's actually allocated -- peak_alloc_gb below still gets
        # charged for whatever is live at entry (e.g. the soft phase's own
        # positions), so record what that is here and let __exit__ report it
        # alongside peak_alloc_gb.
        self._baseline_alloc_gb = (torch.cuda.memory_allocated() / 2**30
                                   if torch.cuda.is_available() else None)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        if not self._timer.reset_peak:
            # M4 T2b: this timer isn't reading/resetting the allocator's
            # peak counters at all here -- a LifetimeRecorder owns that
            # window instead (see PhaseTimer.__init__'s docstring).
            # peak_alloc_gb/peak_reserved_gb are explicitly None (not 0.0,
            # which would misreport as "measured and found empty") so a
            # consumer can tell "not measured here" apart from "measured,
            # zero" -- see run_placement._phase_summary's None-skipping fix.
            self._timer.phases[self._name] = {
                "t_s": dt,
                "peak_alloc_gb": None,
                "peak_reserved_gb": None,
                "peak_semantics": "disabled_owned_by_lifetime_recorder",
                "baseline_alloc_gb": self._baseline_alloc_gb,
                # Not measured here either -- peak_alloc_gb is None on this
                # path, so there is nothing to subtract the baseline from.
                "peak_above_baseline_gb": None,
                "host_rss_hwm_at_phase_end": host_rss_gb(),
            }
            return False
        peak_alloc_gb = (torch.cuda.max_memory_allocated() / 2**30
                         if torch.cuda.is_available() else 0.0)
        peak_reserved_gb = (torch.cuda.max_memory_reserved() / 2**30
                            if torch.cuda.is_available() else 0.0)
        baseline_alloc_gb = self._baseline_alloc_gb if self._baseline_alloc_gb is not None else 0.0
        self._timer.phases[self._name] = {
            "t_s": dt,
            "peak_alloc_gb": peak_alloc_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "baseline_alloc_gb": baseline_alloc_gb,
            # reset_peak_memory_stats() at phase entry zeroes the HWM
            # counter but not what was already live (this phase's docstring,
            # "necessary but not sufficient") -- this is peak_alloc_gb minus
            # that live baseline, i.e. what THIS phase itself added.
            "peak_above_baseline_gb": peak_alloc_gb - baseline_alloc_gb,
            "host_rss_hwm_at_phase_end": host_rss_gb(),
        }
        return False


def release_cuda_scratch():
    """Release ATen's per-(handle, stream) cuBLAS workspace: a fixed
    32 MiB (33,554,432 B) one-time retention diagnosed on the fence-bearing
    drivers, held in a C++ static map by
    `at::cuda::getCurrentCUDABlasHandle()` -- not a Python object, so `gc`
    cannot see it and `torch.cuda.empty_cache()` alone cannot reclaim it (the
    workspace is active, not cached). It is triggered by
    `$DREAMPLACE_ROOT/dreamplace/PlaceObj.py:321`'s
    `self.wirelength + self.density_weight.dot(self.density)`, taken only
    when `len(self.placedb.regions) > 0` (the fence branch).

    Safety: `torch._C._cuda_clearCublasWorkspaces()` is unsafe only if
    another thread is concurrently issuing cuBLAS calls on this process.
    Nothing here does -- `DeviceMemSampler`'s background thread only calls
    `mem_get_info()` -- so calling this after the sampler has stopped is
    safe.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(root):
    if not root:
        return None
    try:
        out = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def env_metadata(repo_root, dp_root=None, input_paths=()):
    """Same shape as the per-probe `_env_metadata` helper M2/M3 probes each
    hand-roll (e.g. `ioplace/diagnostics/probes_m3/probe_free_area_util.py`)
    -- centralized here since profile.py is shared infra, not a one-off
    script. `input_paths` entries may be repo-relative (resolved against
    `repo_root`) or absolute."""
    import numpy as np
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "ioplace_commit": _git_head(repo_root),
        "dp_commit": _git_head(dp_root),
        "input_sha256": {
            p: _sha256(p if os.path.isabs(p) else os.path.join(repo_root, p))
            for p in input_paths
        },
    }


def bytes_per_pin_gp(peak_alloc_bytes, n_pins):
    """sec 1's GP memory model coefficient (empirically ~87-160 B/element
    depending on which term dominates); `None` when n_pins is 0/unknown
    rather than raising, since this is a reporting convenience, not a
    correctness-critical path."""
    return (peak_alloc_bytes / n_pins) if n_pins else None


def ns_per_pin_op(fwd_bwd_ms, n_pins):
    """sec 1.3's IoTerm runtime coefficient (~84 B/pin's time analogue)."""
    return (fwd_bwd_ms * 1e6 / n_pins) if n_pins else None


def gb_per_mnet_eval(peak_gb, n_nets):
    """sec 1.2's evaluator memory coefficient, per million nets."""
    return (peak_gb / (n_nets / 1e6)) if n_nets else None


def build_profile_record(*, case, K, rtype, arm, phase_timer: PhaseTimer,
                         iter_ms=(), op_fwd_ms=(), op_bwd_ms=(), eval_ms=(),
                         device_used_gb=0.0, scale_meta=None,
                         derived=None, env=None):
    """Assembles the sec 6.2 JSON schema (`PROFILE_SCHEMA_FIELDS`, all
    present). `phase_timer.phases` supplies the named-phase wall-time/GPU
    rows (`t_read`/`t_initialize`/`t_gp`/`t_lg` come from phases named
    "read"/"initialize"/"gp"/"lg"; anything else recorded under
    `phase_timer` is preserved verbatim in `phases` but has no matching
    top-level `t_*` alias). `device_used_gb` should come from a
    `DeviceMemSampler.device_used_gb` covering the run (see that class's
    whole-GPU/undersampling caveats -- they apply transitively here).
    `scale_meta` supplies sec 6.1's "規模元資料" row; `derived` optionally
    overrides the auto-computed coefficients (default: None, since deriving
    them needs scale info this function doesn't require callers to
    supply)."""
    scale_meta = dict(scale_meta or {})
    phases = phase_timer.phases

    def phase_t(name):
        return phases.get(name, {}).get("t_s", 0.0)

    t_eval_total = float(sum(eval_ms)) / 1000.0
    t_op_total = float(sum(op_fwd_ms) + sum(op_bwd_ms)) / 1000.0
    t_diag_total = phase_t("diag")
    t_total = sum(p["t_s"] for p in phases.values())

    ip = percentiles_ms(list(iter_ms))
    # host_rss_hwm_at_phase_end is a monotonic process-lifetime sequence
    # (see host_rss_gb()'s docstring), so its true max is just its last
    # recorded value -- folding in a fresh host_rss_gb() call here only
    # guards against phase_timer having recorded no phases at all.
    host_peaks = [p.get("host_rss_hwm_at_phase_end", 0.0) for p in phases.values()] + [host_rss_gb()]

    record = {
        "case": case, "K": K, "rtype": rtype, "arm": arm,
        "t_read": phase_t("read"), "t_initialize": phase_t("initialize"),
        "t_gp": phase_t("gp"), "t_lg": phase_t("lg"),
        "t_eval_total": t_eval_total, "t_op_total": t_op_total,
        "t_diag_total": t_diag_total, "t_total": t_total,
        "iter_ms": list(iter_ms),
        "iter_ms_p50": ip["p50"], "iter_ms_p90": ip["p90"],
        "iter_ms_p99": ip["p99"], "iter_ms_max": ip["max"],
        "op_fwd_ms": list(op_fwd_ms), "op_bwd_ms": list(op_bwd_ms),
        "eval_ms": list(eval_ms),
        "phases": phases,
        "device_used_gb": device_used_gb,
        "host_peak_rss_gb": max(host_peaks),
        "bytes_per_pin_gp": None, "ns_per_pin_op": None, "gb_per_mnet_eval": None,
        "env": env or {},
    }
    for f in ("n_movable", "n_physical", "n_filler", "n_nets", "n_pins",
             "lattice", "n_bins", "k_chunk", "n_active"):
        record[f] = scale_meta.get(f)
    if derived:
        record.update(derived)

    missing = [f for f in PROFILE_SCHEMA_FIELDS if f not in record]
    assert not missing, f"build_profile_record: missing schema fields {missing}"
    return record
