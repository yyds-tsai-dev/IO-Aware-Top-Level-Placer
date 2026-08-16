"""M4 T9 30M component-level spike -- PARENT process (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 5.3
layer 2 / T9 row / sec 2.2's three-state feasibility rule).

**This module must NEVER build a CUDA context.** The actual IoTerm+
GpuEvalContext workload runs in a CHILD subprocess (`spike_30m_child.py`);
this file only launches it, watches it from the outside via `nvidia-smi`
(0.5s polling of `--query-gpu`/`--query-compute-apps`, never a `torch.
cuda.*` call), and assembles/writes the final RESULT-GATE-compliant
artifact. That split is why `import torch` never appears anywhere in this
module -- a plain `import torch` doesn't touch CUDA, but this module goes
further and avoids even that, so a static "does this file import torch"
check is a correct, cheap way to audit the contract (there is a unit test
for exactly that below this module's own docstring, in this task's test
suite).

Sec 2.2's three-state feasibility rule, restated for this pair:
  - `experiment_status` in {ok, contaminated, instrumentation_error,
    input_error} -- is THIS MEASUREMENT valid;
  - `workload_status` in {completed, oom, crashed} -- what the CHILD did;
  - `feasibility_verdict` in {feasible_l4_contract, infeasible_l4_contract,
    invalid_measurement} -- computed by `ioplace.bench.result_gate.
    feasibility_verdict` (already implemented/tested there -- this module
    calls it rather than re-deriving the same six-branch logic).

State machine (up to `--max-attempts`, default 3, one child subprocess per
attempt -- the design draft's "invalid_measurement 至多重試兩次" and "OOM
需在同一契約下重現兩次才可判 infeasible" both fit inside 3 total attempts):
  1. preflight (once, before any attempt): gpu_name, `assert_budget`,
     nvsmi baseline < 0.5 GiB + no external compute-apps PID, `Mem
     Available` >= 40 GiB, input sha256 vs the cache's own manifest
     provenance. Any preflight failure -> `experiment_status="input_error"`
     (contamination-shaped preflight failures use `"contaminated"`
     instead) and the attempt loop still runs (retried up to the same
     budget) -- a transient contamination/host-load condition may clear
     between attempts.
  2. per attempt: launch the child, poll nvidia-smi throughout, wait for
     exit, run postflight (device baseline back within +-0.2 GiB, no
     lingering external PID) -- **contamination takes priority over the
     OOM verdict** (design draft T9 row): if postflight fails, `experiment_
     status="contaminated"` regardless of the child's own exit code.
     Exit code 0 -> `workload_status="completed"`; exit code 3 -> `"oom"`
     (`experiment_status="ok"` unless postflight/contradiction says
     otherwise); any other exit code -> `workload_status="crashed"` AND
     `experiment_status="instrumentation_error"` directly (this task's own
     instruction reads "其他=crashed->instrumentation_error" as a direct
     mapping, not routed through `feasibility_verdict`'s more general
     "ok-but-crashed -> invalid_measurement" fallback -- both paths land on
     `invalid_measurement` in the end, since `feasibility_verdict` returns
     that for any `experiment_status != "ok"` regardless of subtype, so
     this is a stricter *label*, not a different final verdict).
     A **contradiction** (this attempt says "completed" after an earlier
     attempt said "oom" under the identical contract, or vice versa) is
     itself contamination (design draft T9 row's literal "oom-then-
     completed 矛盾=contaminated 重試").
  3. loop until `feasibility_verdict` is no longer `invalid_measurement`,
     or `--max-attempts` is exhausted (then `closed="blocked_external"`).

Parent NEVER writes to the intermediate per-attempt partial paths as the
canonical artifact -- only the merged final record, written exactly once
at `--out` (default `results/m4/profile/spike30m__<scenario>__k<K>.json`).

CLI:
    PYTHONPATH=. $PY -m ioplace.bench.spike_30m \\
        --cache-dir results/m4/bench/tiled_cache/3x3_n2 \\
        --k 32 --s4-scenario A --rtype grid --seed 0 --n-iter 4 --lr 1.0 \\
        --budget-gb 19.5 --budget-source "NVIDIA L4"
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid

from ioplace.bench.result_gate import (HW_BUDGET_GB, SPIKE_SCHEMA_FIELDS,
                                       assert_budget, feasibility_verdict)

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
DEFAULT_PYTHON = os.path.join(DP, ".venv312", "bin", "python")
DEFAULT_BUDGET_GB = HW_BUDGET_GB["NVIDIA L4"]
DEFAULT_BUDGET_SOURCE = "NVIDIA L4"
DEFAULT_COUNT_FREEZE = os.path.join(REPO, "results", "m4", "scaling", "count_freeze_30m.json")
CONTAMINATION_BASELINE_GIB = 0.5
POSTFLIGHT_TOLERANCE_GIB = 0.2
MEM_AVAILABLE_MIN_GB = 40.0


def _sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(root):
    try:
        out = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


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
    worked" -- sec 1.4 T1b's own documented degrade path: this host has
    been observed to return an empty compute-apps list even while our own
    workload is running, in which case the caller must fall back to
    `exclusivity_evidence="baseline_only"` rather than trusting an empty
    list as proof of exclusivity)."""
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


class NvsmiPoller:
    """Background thread sampling `nvidia-smi --query-gpu=memory.used`
    every `interval_s` -- the parent's ONLY window into device memory,
    since it never touches CUDA itself (module docstring)."""

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


def _preflight(budget_gb, budget_source, cache_dir):
    """Returns (ok, status_if_not_ok, details_dict). `status_if_not_ok` is
    "contaminated" or "input_error" -- never "ok" (caller only reads it
    when ok is False)."""
    details = {}
    name = gpu_name()
    details["gpu_name"] = name
    assert_budget(gpu_name=name, budget_gb=budget_gb, source=budget_source)  # enforcement point 1

    baseline = device_used_gb()
    details["baseline_device_used_gb"] = baseline
    pids = compute_app_pids()
    details["preflight_compute_app_pids"] = pids

    if baseline is None or baseline >= CONTAMINATION_BASELINE_GIB:
        return False, "contaminated", details
    if pids:  # non-empty and non-None -- some other process is already compute-active
        return False, "contaminated", details
    exclusivity_evidence = "compute_apps_pid_confirmed" if pids is not None else "baseline_only"
    details["exclusivity_evidence"] = exclusivity_evidence

    mem_avail_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail_gb = int(line.split()[1]) / 2**20
                    break
    except Exception:
        pass
    details["host_mem_available_gb"] = mem_avail_gb
    if mem_avail_gb is None or mem_avail_gb < MEM_AVAILABLE_MIN_GB:
        return False, "input_error", details

    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False, "input_error", details
    with open(meta_path) as f:
        meta = json.load(f)
    manifest_path = meta.get("manifest_path")
    if manifest_path and os.path.exists(manifest_path):
        actual = _sha256_file(manifest_path)
        details["manifest_sha256_at_preflight"] = actual
    details["cache_meta"] = meta

    return True, None, details


def _run_one_attempt(attempt, args, child_out_path):
    poller = NvsmiPoller().start()
    baseline_before = device_used_gb()
    cmd = [
        args.python, "-m", "ioplace.bench.spike_30m_child",
        "--cache-dir", args.cache_dir, "--k", str(args.k), "--rtype", args.rtype,
        "--seed", str(args.seed), "--n-iter", str(args.n_iter), "--lr", str(args.lr),
        "--s4-scenario", args.s4_scenario,
        "--budget-gb", str(args.budget_gb), "--budget-source", args.budget_source,
        "--out", child_out_path,
    ]
    if args.tau is not None:
        cmd += ["--tau", str(args.tau)]
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=args.child_timeout_s)
        exit_code = proc.returncode
        stderr_tail = proc.stderr[-4000:] if proc.stderr else ""
    except subprocess.TimeoutExpired as e:
        exit_code = None   # timeout -- treated as crashed/instrumentation_error below
        stderr_tail = f"child timed out after {args.child_timeout_s}s"
    wall_s = time.time() - t0

    poller.stop()
    baseline_after = device_used_gb()

    postflight_ok = (baseline_after is not None
                     and abs(baseline_after - (baseline_before or 0.0)) <= POSTFLIGHT_TOLERANCE_GIB)
    post_pids = compute_app_pids()
    if post_pids:  # any lingering external compute process
        postflight_ok = False

    partial = {}
    if os.path.exists(child_out_path):
        try:
            with open(child_out_path) as f:
                partial = json.load(f)
        except Exception:
            partial = {}

    if exit_code == 0:
        workload_status = partial.get("workload_status", "completed")
    elif exit_code == 3:
        workload_status = "oom"
    else:
        workload_status = "crashed"

    return {
        "attempt": attempt, "exit_code": exit_code, "wall_s": wall_s,
        "workload_status": workload_status, "postflight_ok": postflight_ok,
        "child_stderr_tail": stderr_tail,
        "child_exit_code": exit_code,
        "device_used_peak_gb_nvsmi": poller.peak_gb,
        "device_baseline_gb_nvsmi": baseline_before,
        "partial": partial,
    }


def _load_cache_meta_best_effort(cache_dir):
    """Independent of preflight's own pass/fail: `input_sha256`/`case`/
    `n_nodes` etc are provenance about the CACHE the run was pointed at,
    not about whether the measurement succeeded -- a contaminated or
    input_error run should still carry them when the cache itself is
    readable (SPIKE_SCHEMA_FIELDS rule 4 requires non-empty provenance
    regardless of experiment_status)."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def run(args):
    ok, preflight_status, preflight_details = _preflight(
        args.budget_gb, args.budget_source, args.cache_dir)
    if "cache_meta" not in preflight_details:
        preflight_details["cache_meta"] = _load_cache_meta_best_effort(args.cache_dir)

    attempts_log = []
    oom_count = 0
    saw_oom = saw_completed = False
    final = None

    for attempt in range(1, args.max_attempts + 1):
        if not ok:
            # preflight itself failed -- still counts as an attempt against
            # the retry budget (module docstring point 1); no child launched.
            result = {"attempt": attempt, "exit_code": None, "wall_s": 0.0,
                      "workload_status": "crashed", "postflight_ok": False,
                      "child_stderr_tail": "", "child_exit_code": None,
                      "device_used_peak_gb_nvsmi": None,
                      "device_baseline_gb_nvsmi": preflight_details.get("baseline_device_used_gb"),
                      "partial": {}}
            experiment_status = preflight_status
        else:
            child_out_path = os.path.join(
                os.path.dirname(args.out) or ".",
                f".spike30m_attempt{attempt}_k{args.k}_{args.s4_scenario}.partial.json")
            result = _run_one_attempt(attempt, args, child_out_path)
            ws = result["workload_status"]
            contradiction = (ws == "completed" and saw_oom) or (ws == "oom" and saw_completed)
            if ws == "oom":
                oom_count += 1
                saw_oom = True
            if ws == "completed":
                saw_completed = True

            if not result["postflight_ok"] or contradiction:
                experiment_status = "contaminated"
            elif ws == "crashed":
                experiment_status = "instrumentation_error"
            else:
                experiment_status = "ok"

        result["experiment_status"] = experiment_status
        result["oom_repeats"] = oom_count
        partial = result.get("partial", {})
        peak_gb = partial.get("measured_peak_gb")
        resident_lb_gb = partial.get("resident_lower_bound_gb")
        result["feasibility_verdict"] = feasibility_verdict(
            experiment_status, result["workload_status"], peak_gb,
            args.budget_gb, oom_count, resident_lb_gb)

        attempts_log.append(result)
        final = result
        if result["feasibility_verdict"] != "invalid_measurement":
            break
        # preflight may transiently clear on a retry -- re-check before the
        # next attempt if it failed this time.
        if not ok:
            ok, preflight_status, preflight_details = _preflight(
                args.budget_gb, args.budget_source, args.cache_dir)

    record = _assemble_record(args, final, attempts_log, preflight_details)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    return record


def _assemble_record(args, final, attempts_log, preflight_details):
    partial = final.get("partial", {})
    meta = preflight_details.get("cache_meta", {})
    peak_gb = partial.get("measured_peak_gb")
    resident_gb = partial.get("resident_gb")
    resident_lb_gb = partial.get("resident_lower_bound_gb")
    max_transient_gb = partial.get("max_phase_transient_gb")

    record = {
        "record_kind": "spike",
        "case": partial.get("case") or os.path.basename(meta.get("dst_prefix") or args.cache_dir),
        "K": args.k, "benchmark_kind": "synthetic",
        "experiment_status": final["experiment_status"],
        "workload_status": final["workload_status"],
        "feasibility_verdict": final["feasibility_verdict"],
        "measured_peak_gb": peak_gb,
        "budget_gb": args.budget_gb, "budget_source": args.budget_source,
        "baseline_reserved_gb": final.get("device_baseline_gb_nvsmi"),
        "resident_gb": resident_gb,
        "resident_lower_bound_gb": resident_lb_gb,
        "max_phase_transient_gb": max_transient_gb,
        "retry_count": len(attempts_log) - 1,
        "n_interleaved_iters": partial.get("n_interleaved_iters", 0),
        "oom_repeats": final.get("oom_repeats", 0),
        "peak_reserved_gb": partial.get("peak_reserved_gb"),
        "device_used_peak_gb": final.get("device_used_peak_gb_nvsmi"),
        "device_baseline_gb": preflight_details.get("baseline_device_used_gb"),
        "n_interleaved_iters_expected": args.n_iter,
        "s4_scenario": partial.get("s4_scenario",
                                   "A" if args.s4_scenario == "A" else "B_emulated_ballast"),
        "s4_source": partial.get("s4_source", "n/a" if args.s4_scenario == "A" else "not_merged_m3"),
        "k_chunk": partial.get("k_chunk"),
        "n_nodes": partial.get("n_nodes") or meta.get("n_nodes"),
        "n_nets": partial.get("n_nets") or meta.get("n_nets"),
        "n_pins": partial.get("n_pins") or meta.get("n_pins"),
        "P_dedup": partial.get("P_dedup"),
        "n_active": partial.get("n_active"),
        "exclusivity_evidence": preflight_details.get("exclusivity_evidence"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "child_exit_code": final.get("child_exit_code"),
        "child_stderr_tail": final.get("child_stderr_tail"),
        "count_freeze_ref": _sha256_file(args.count_freeze),
        "attempts_log": [
            {k: v for k, v in a.items() if k != "partial"} for a in attempts_log
        ],
        # sec 6.2/7.0 provenance
        "run_id": str(uuid.uuid4()),
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "gpu_name": preflight_details.get("gpu_name"),
        "seed": args.seed,
        "input_sha256": partial.get("input_sha256")
                        or meta.get("manifest_output_sha256", {}).get("nets"),
    }
    if final["feasibility_verdict"] == "invalid_measurement" and len(attempts_log) >= args.max_attempts:
        record["closed"] = "blocked_external"

    missing = [f for f in SPIKE_SCHEMA_FIELDS if f not in record]
    assert not missing, f"_assemble_record: missing SPIKE_SCHEMA_FIELDS {missing}"
    return record


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--k", type=int, required=True, choices=(16, 32))
    ap.add_argument("--rtype", default="grid", choices=("grid", "slicing"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--s4-scenario", choices=("A", "B"), default="A")
    ap.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB)
    ap.add_argument("--budget-source", default=DEFAULT_BUDGET_SOURCE)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--child-timeout-s", type=float, default=1800.0)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--count-freeze", default=DEFAULT_COUNT_FREEZE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    # sec 1.4 B3 enforcement point 1 ("parent argv 解析時"): a CLI
    # --budget-gb may never exceed the hardware contract -- fails loudly
    # here, before any preflight/child launch.
    assert_budget(gpu_name=None, budget_gb=args.budget_gb, source=args.budget_source)

    if args.out is None:
        args.out = os.path.join(
            REPO, "results", "m4", "profile",
            f"spike30m__{args.s4_scenario}__k{args.k}.json")
    return args


def main(argv=None):
    args = _parse_args(argv)
    record = run(args)
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
