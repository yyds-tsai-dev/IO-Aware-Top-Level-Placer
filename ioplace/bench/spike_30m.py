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
import math
import threading
import os
import socket
import subprocess
import sys
import time
import uuid

from ioplace.bench.result_gate import (HW_BUDGET_GB, SPIKE_SCHEMA_FIELDS,
                                       assert_budget, feasibility_verdict)
# M4 T1b (d): the nvidia-smi-only primitives below (gpu_name/device_used_gb/
# compute_app_pids/NvsmiPoller/the two tolerance constants) used to be
# defined here; they now live in `gpu_exclusivity.py` so this module and
# T1b's own exclusivity probe share one implementation instead of two --
# re-imported under their original names so every existing call site
# (`gpu_name()`, `sp.CONTAMINATION_BASELINE_GIB`, etc.) is unchanged.
from ioplace.diagnostics.probes_m4.gpu_exclusivity import (  # noqa: E402  (re-exported)
    CONTAMINATION_BASELINE_GIB, POSTFLIGHT_TOLERANCE_GIB, NvsmiPoller,
    compute_app_pids, device_used_gb, gpu_name, _nvsmi)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DP = os.environ.get("DREAMPLACE_ROOT", os.path.join(os.path.dirname(REPO), "DREAMPlace"))
DEFAULT_PYTHON = os.path.join(DP, ".venv312", "bin", "python")
DEFAULT_BUDGET_GB = HW_BUDGET_GB["NVIDIA L4"]
DEFAULT_BUDGET_SOURCE = "NVIDIA L4"
DEFAULT_COUNT_FREEZE = os.path.join(REPO, "results", "m4", "scaling", "count_freeze_30m.json")
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


def device_snapshot():
    raw = _nvsmi("uuid,name,memory.total,memory.used,memory.free,utilization.gpu")
    if not raw:
        raise RuntimeError("selected GPU snapshot unavailable")
    rows = raw.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    uuid_, name, total, used, free, utilization = [v.strip() for v in rows[0].split(",")]
    values = list(map(float, (total, used, free, utilization)))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("nonfinite or negative device snapshot")
    return dict(time=time.time(), uuid=uuid_, name=name, total_gib=values[0]/1024,
                used_gib=values[1]/1024, free_gib=values[2]/1024,
                utilization_pct=values[3], compute_apps=compute_app_pids())


def shared_verdict(experiment_status, workload_status, peak, budget, iterations, expected):
    if experiment_status != "ok":
        return "invalid_measurement"
    if workload_status == "oom":
        return "shared_oom_unattributed"
    if (workload_status != "completed" or iterations < expected or peak is None
            or not math.isfinite(peak) or peak < 0):
        return "invalid_measurement"
    return ("completed_within_registered_process_budget" if peak <= budget
            else "exceeds_registered_process_budget")


def _preflight(budget_gb, budget_source, cache_dir, measurement_mode="exclusive", shared_margin_gb=4.0):
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

    if measurement_mode == "shared":
        details["measurement_mode"] = "shared"
        details["shared_user_authorized"] = True
        details["shared_margin_gb"] = shared_margin_gb
        details["shared_free_required_gb"] = budget_gb + shared_margin_gb
        try:
            snapshot = device_snapshot()
            details["device_snapshot"] = snapshot
            free = snapshot["free_gib"]
            details["shared_free_gb"] = free
            if free < budget_gb + shared_margin_gb:
                return False, "input_error", details
        except Exception as e:
            details["nvidia_smi_error"] = str(e)
            return False, "input_error", details
    elif baseline is None or baseline >= CONTAMINATION_BASELINE_GIB:
        return False, "contaminated", details
    if measurement_mode != "shared" and pids:  # external process invalidates exclusive run
        return False, "contaminated", details
    exclusivity_evidence = ("shared_user_authorized" if measurement_mode == "shared" else
                            "compute_apps_pid_confirmed" if pids is not None else "baseline_only")
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
    shared = getattr(args, "measurement_mode", "exclusive") == "shared"
    samples, stop = [], threading.Event()
    def sample_loop():
        while True:
            try:
                samples.append(device_snapshot())
            except Exception as error:
                samples.append(dict(time=time.time(), error=str(error)))
            if stop.wait(2):
                break
    thread = threading.Thread(target=sample_loop, daemon=True) if shared else None
    if thread:
        thread.start()
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
    if thread:
        stop.set()
        thread.join()
    baseline_after = device_used_gb()

    postflight_ok = (baseline_after is not None
                     and abs(baseline_after - (baseline_before or 0.0)) <= POSTFLIGHT_TOLERANCE_GIB)
    post_pids = compute_app_pids()
    if post_pids and not shared:  # any lingering external compute process
        postflight_ok = False
    if shared:
        postflight_ok = True

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
        "postflight_compute_app_pids": post_pids,
        "device_baseline_after_gb": baseline_after,
        "device_samples": samples,
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
    args.measurement_mode = getattr(args, "measurement_mode", "exclusive")
    args.shared_margin_gb = getattr(args, "shared_margin_gb", 4.0)
    ok, preflight_status, preflight_details = _preflight(
        args.budget_gb, args.budget_source, args.cache_dir, args.measurement_mode, args.shared_margin_gb)
    if "cache_meta" not in preflight_details:
        preflight_details["cache_meta"] = _load_cache_meta_best_effort(args.cache_dir)
    preflight_details["source_sha256"] = {
        path: _sha256_file(os.path.join(REPO, path)) for path in (
            "ioplace/bench/spike_30m.py", "ioplace/bench/spike_30m_child.py",
            "ioplace/bench/bookshelf_netlist.py", "ioplace/ops/io_term.py",
            "ioplace/evaluator_gpu.py", "ioplace/diagnostics/probes_m4/gpu_exclusivity.py")}

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
        if args.measurement_mode == "shared":
            result["feasibility_verdict"] = shared_verdict(experiment_status,
                result["workload_status"], peak_gb, args.budget_gb,
                partial.get("n_interleaved_iters", 0), args.n_iter)
        else:
            result["feasibility_verdict"] = feasibility_verdict(experiment_status, result["workload_status"], peak_gb, args.budget_gb, oom_count, resident_lb_gb)

        attempts_log.append(result)
        final = result
        if args.measurement_mode == "shared" or result["feasibility_verdict"] != "invalid_measurement":
            break
        # preflight may transiently clear on a retry -- re-check before the
        # next attempt if it failed this time.
        if not ok:
            ok, preflight_status, preflight_details = _preflight(args.budget_gb, args.budget_source, args.cache_dir, args.measurement_mode, args.shared_margin_gb)

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
    record["measurement_mode"] = getattr(args, "measurement_mode", "exclusive")
    record["device_uuid"] = preflight_details.get("device_snapshot", {}).get("uuid")
    record["preflight"] = preflight_details
    record["n_pins_raw"] = meta.get("n_pins_raw")
    record["n_pins_canonical"] = meta.get("n_pins")
    for key in ("phase_timings", "component_build_s", "host_peak_rss_gb", "transient_accounting_note"):
        record[key] = partial.get(key)
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
    ap.add_argument("--measurement-mode", choices=("exclusive", "shared"), default="exclusive")
    ap.add_argument("--shared-margin-gb", type=float, default=4.0)
    args = ap.parse_args(argv)
    if not math.isfinite(args.shared_margin_gb) or args.shared_margin_gb < 0:
        raise ValueError("shared margin must be finite and nonnegative")

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
