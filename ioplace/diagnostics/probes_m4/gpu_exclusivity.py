"""M4 T1b (d) shared exclusive-GPU protocol (design draft `docs/superpowers/
specs/2026-08-13-m4-scale-up-design-draft.md` sec 1.4 D9 / T1b row (d)):
the nvidia-smi-only preflight/postflight primitives every "formal memory
run needs exclusive GPU" check is built from.

Extracted from `ioplace/bench/spike_30m.py`'s parent process (that
module's own docstring: "this module must NEVER build a CUDA context" --
`nvidia-smi` subprocess calls are the only window it has into device
memory) so `spike_30m.py` and this task's T1b probes share ONE
implementation of "how do we ask nvidia-smi about baseline usage / other
compute processes" instead of two copies -- `spike_30m.py` now imports
these names rather than redefining them.

sec 1.4 D9 (Codex finding, fixing v2's tautology: comparing `device_used -
baseline < 0.5 GB` is always true because `baseline` already contains
whatever contamination is present):
  - preflight: `device_baseline_gb` must be < `CONTAMINATION_BASELINE_GIB`
    in **absolute** terms (not a delta), AND `nvidia-smi
    --query-compute-apps=pid,used_memory` must list no compute-active PID
    other than our own workload's.
  - that compute-apps query needs a self-test before it can be trusted as
    *evidence of absence*: `self_test_own_pid_visible` checks whether the
    query can see our own PID while our own workload is actually
    CUDA-active. This host has been observed to return an empty
    compute-apps list even while a real workload is running (design draft
    sec 6.1's recorded note) -- if the self-test can't see us, an empty
    list from someone else's PID is not trustworthy either, so evidence
    downgrades to `"baseline_only"` rather than
    `"compute_apps_pid_confirmed"`.
  - postflight: `device_used_gb` must return to the preflight baseline
    within `POSTFLIGHT_TOLERANCE_GIB`, and no external compute-apps PID
    may still be present.
  - any preflight/postflight failure is the caller's cue to record
    `experiment_status="contaminated"` (sec 1.4 D9 / T1b (d)) -- this
    module reports pass/fail + reasons, it does not itself write artifacts
    or decide `experiment_status` (that stays with each caller, e.g.
    `spike_30m.py`'s three-state state machine).
"""
import subprocess
import threading
import os

# sec 1.4 D9: baseline must be < 0.5 GiB in absolute terms.
CONTAMINATION_BASELINE_GIB = 0.5
# sec 1.4 T1b (d): postflight device_used must return to baseline +/- 0.2 GiB.
POSTFLIGHT_TOLERANCE_GIB = 0.2


def _nvsmi(query, fmt="csv,noheader,nounits", timeout=5.0):
    """Runs one `nvidia-smi --query-gpu=<query>` (or `--query-compute-apps`
    if `query` starts with 'compute:') call; returns raw stdout text, or
    None if nvidia-smi itself failed/timed out (a provenance-level
    "evidence unavailable" signal, not silently treated as "0 usage")."""
    try:
        if query.startswith("compute:"):
            args = ["nvidia-smi", f"--query-compute-apps={query[8:]}", f"--format={fmt}"]
        else:
            args = ["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"]
        # nvidia-smi ignores CUDA_VISIBLE_DEVICES. Match the device used by
        # CUDA rather than silently sampling physical GPU 0 on multi-GPU hosts.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if visible and visible != "-1":
            args += ["--id", visible]
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, timeout=timeout)
        return out.decode()
    except Exception:
        return None


def gpu_name():
    text = _nvsmi("name", fmt="csv,noheader")
    return text.strip().splitlines()[0] if text else None


def device_used_gb():
    text = _nvsmi("memory.used")
    if not text:
        return None
    try:
        return float(text.strip().splitlines()[0]) / 1024.0   # MiB -> GiB
    except (ValueError, IndexError):
        return None


def compute_app_pids():
    """[(pid:int, used_mib:float), ...], or None if the query itself
    failed/is unavailable on this host (distinct from "empty but the query
    worked" -- see `self_test_own_pid_visible` for the "empty even though
    our own PID should be visible" degrade path)."""
    text = _nvsmi("compute:pid,used_memory")
    if text is None:
        return None
    pids = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            pids.append((int(parts[0]), float(parts[1].split()[0])))
        except (ValueError, IndexError):
            continue
    return pids


def self_test_own_pid_visible(pid):
    """T1b (d)'s compute-apps self-test: call this WHILE `pid` (typically
    `os.getpid()` of a process that is currently CUDA-active, or a child
    workload's pid) should be visible to `nvidia-smi
    --query-compute-apps`. Returns True/False if the query succeeded (did
    it list `pid`?), or None if the query itself failed (a separate
    "evidence unavailable" case from "queried fine, pid absent")."""
    pids = compute_app_pids()
    if pids is None:
        return None
    return any(p == pid for p, _ in pids)


class NvsmiPoller:
    """Background thread sampling `nvidia-smi --query-gpu=memory.used`
    every `interval_s` -- for a caller (like `spike_30m.py`'s parent) that
    never touches CUDA itself and has no other window into device memory."""

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self._peak_gb = 0.0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread = None

    def _sample_once(self):
        used = device_used_gb()
        if used is not None:
            with self._lock:
                self._peak_gb = max(self._peak_gb, used)

    def start(self):
        self._sample_once()
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
        self._sample_once()
        return self

    @property
    def peak_gb(self):
        with self._lock:
            return self._peak_gb


def preflight_exclusivity(*, self_test_pid=None):
    """T1b (d)'s full preflight check. `self_test_pid`, if given, is a PID
    that is expected to be CUDA-active *right now* (this function's own
    caller is responsible for that timing -- e.g. calling this while a
    child workload it just launched is mid-run) -- used to self-test the
    compute-apps query per sec 1.4 D9 before trusting an "empty" reading
    as proof no one else is compute-active. Without `self_test_pid`, the
    self-test is skipped and evidence is capped at `"baseline_only"`.

    Returns a dict:
      `baseline_device_used_gb`, `compute_app_pids`,
      `self_test_own_pid_visible`, `exclusivity_evidence` (
      `"compute_apps_pid_confirmed"` or `"baseline_only"`), `ok` (bool),
      `reasons` (list[str], empty iff `ok`).
    """
    baseline = device_used_gb()
    pids = compute_app_pids()
    self_test = self_test_own_pid_visible(self_test_pid) if self_test_pid is not None else None

    reasons = []
    if baseline is None:
        reasons.append("device_used_gb unavailable (nvidia-smi query failed)")
    elif baseline >= CONTAMINATION_BASELINE_GIB:
        reasons.append(
            f"baseline_device_used_gb={baseline} >= {CONTAMINATION_BASELINE_GIB} GiB")

    other_pids = [(p, m) for p, m in (pids or []) if p != self_test_pid]
    if pids and other_pids:
        reasons.append(f"other compute-apps PIDs present: {other_pids}")

    # sec 1.4 D9's self-test: only trust an empty/other-free compute-apps
    # reading as real evidence of exclusivity if we've confirmed the query
    # can see a PID we KNOW is CUDA-active (our own). No self-test
    # requested, self-test failed, or the query itself unavailable -> the
    # weaker "baseline_only" evidence tier.
    if pids is None:
        exclusivity_evidence = "baseline_only"
    elif self_test_pid is not None and self_test is not True:
        exclusivity_evidence = "baseline_only"
    else:
        exclusivity_evidence = "compute_apps_pid_confirmed"

    return {
        "baseline_device_used_gb": baseline,
        "compute_app_pids": pids,
        "self_test_own_pid_visible": self_test,
        "exclusivity_evidence": exclusivity_evidence,
        "ok": not reasons,
        "reasons": reasons,
    }


def postflight_exclusivity(baseline_before_gb):
    """T1b (d)'s postflight check: `device_used_gb` must have returned to
    `baseline_before_gb` within `POSTFLIGHT_TOLERANCE_GIB`, and no compute
    -apps PID may still be lingering. Returns a dict: `device_used_gb_after`,
    `compute_app_pids`, `ok` (bool), `reasons` (list[str])."""
    used_after = device_used_gb()
    pids = compute_app_pids()
    reasons = []
    if used_after is None or baseline_before_gb is None:
        reasons.append(
            f"device_used_gb unavailable (after={used_after}, baseline={baseline_before_gb})")
    elif abs(used_after - baseline_before_gb) > POSTFLIGHT_TOLERANCE_GIB:
        reasons.append(
            f"device_used_gb after={used_after} vs baseline={baseline_before_gb} "
            f"exceeds +/-{POSTFLIGHT_TOLERANCE_GIB} GiB")
    if pids:
        reasons.append(f"lingering compute-apps PIDs: {pids}")
    return {
        "device_used_gb_after": used_after,
        "compute_app_pids": pids,
        "ok": not reasons,
        "reasons": reasons,
    }
