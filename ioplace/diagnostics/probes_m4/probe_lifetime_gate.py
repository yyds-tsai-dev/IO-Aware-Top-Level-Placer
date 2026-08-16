"""M4 T2b integrated memory gate runner (design draft `docs/superpowers/
specs/2026-08-13-m4-scale-up-design-draft.md` sec 2.2/7.1 T2b): drives
`run_placement_io.run_io` with `lifetime_out` set, and wraps its
`profile_lifetime.LifetimeRecorder` output into the sec 7.0 RESULT
GATE-compliant `results/m4/profile/lifetime_<case>__k<K>__<arm>.json`
record (`ioplace.bench.result_gate.LIFETIME_SCHEMA_FIELDS`).

The config's own `global_place_stages[0].iteration` is overwritten with
`--iterations`: memory *shape* is independent of iteration count (sec 2.2
-- the gate is about which buffers exist and their peak sizes, not about
running GP to convergence), so this probe only needs enough iterations for
the IO term to activate and interleave a handful of callbacks. `--of-on`
defaults to 2.0 -- an overflow threshold no real run would use (overflow
never exceeds ~1.0) -- deliberately so the IO term is guaranteed to
activate (`state.active`) from iteration 0 regardless of the case's actual
initial overflow; this bias is written into the artifact (`of_on` field)
so it's never silently mistaken for a realistic schedule setting.

Artifact postcondition (design draft T2b's own "n_callbacks_with_active >=
3 且 n_evals_while_active >= 3, 否則 experiment_status=
'instrumentation_error'"): if the IO term never actually turned on/
evaluated at least 3 times, the run measured GPU memory before the very op
it's trying to characterize was active -- not a valid T2b measurement,
regardless of whether DREAMPlace itself completed cleanly.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_lifetime_gate \\
        --config <config.json> --case mempool_group --k 32 --rtype grid \\
        --seed 0 --rho-max 0.1 --of-on 2.0 --iterations 5 --diag-every 1 \\
        --out results/m4/profile/lifetime_mempool_group__k32__ours_M2.json
"""
import argparse
import json
import os
import socket
import sys
import tempfile
import uuid

import torch

from ioplace.bench.result_gate import HW_BUDGET_GB, assert_budget, feasibility_verdict
from ioplace.drivers.run_placement_io import run_io
from ioplace.profile import env_metadata

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"


def _host_mem_available_gb():
    """`/proc/meminfo`'s `MemAvailable` (Linux-only, same convention as
    `profile.py`'s `resource.getrusage` choice over psutil -- zero new
    dependencies). `None` (not 0.0/raise) if unreadable, since this is a
    provenance field, not a correctness-critical one."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        return None
    return None


def _gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _device_baseline_gb():
    if not torch.cuda.is_available():
        return 0.0
    free0, total0 = torch.cuda.mem_get_info()
    return (total0 - free0) / 2**30


def _prepared_config(config_json, iterations):
    """Copies `config_json` to a fresh temp file with
    `global_place_stages[0].iteration` overwritten to `iterations` -- see
    module docstring for why iteration count doesn't affect memory shape.
    Caller is responsible for deleting the returned path."""
    with open(config_json) as f:
        cfg = json.load(f)
    cfg["global_place_stages"][0]["iteration"] = iterations
    fd, path = tempfile.mkstemp(suffix=".json", prefix="lifetime_gate_cfg_")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


def default_out_json(case, k, arm):
    return f"results/m4/profile/lifetime_{case}__k{k}__{arm.replace('@', '_')}.json"


def run(config_json, case, k, rtype, seed, *, rho_max=0.1, of_on=2.0,
       iterations=5, diag_every=1, out_json=None, benchmark_kind="real",
       budget_gb=None, budget_source=None):
    """Runs the T2b integrated lifetime probe once and writes the RESULT
    GATE record to `out_json` (default: sec 7.1 T2b's naming convention).
    Returns the record dict (also written to disk)."""
    arm = "flat" if rho_max == 0.0 else "ours@M2"
    if out_json is None:
        out_json = default_out_json(case, k, arm)

    gpu_name = _gpu_name()
    if budget_source is None:
        budget_source = gpu_name if gpu_name in HW_BUDGET_GB else "_m2_10m_contract"
    if budget_gb is None:
        budget_gb = HW_BUDGET_GB[budget_source]
    # sec 1.4 B3: a CLI flag can request a smaller budget than the
    # contract, never a larger one -- assert_budget enforces the ceiling.
    assert_budget(gpu_name, budget_gb, budget_source)

    device_baseline_gb = _device_baseline_gb()
    tmp_config = _prepared_config(config_json, iterations)
    lifetime_tmp = out_json + ".lifetime_raw.json"

    experiment_status, workload_status = "ok", "crashed"
    driver_result = None
    try:
        driver_result = run_io(
            tmp_config, k, rtype, seed,
            out_json=out_json + ".driver_result.json",
            rho_max=rho_max, of_on=of_on, every=1, diag_every=diag_every,
            benchmark_kind=benchmark_kind, lifetime_out=lifetime_tmp)
        workload_status = "completed"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # sec 2.2's three-state rule: an OOM under the exclusive-GPU
            # contract is a legitimate, informative terminal state, not a
            # measurement failure -- experiment_status stays "ok".
            workload_status = "oom"
        else:
            experiment_status = "crashed"
            workload_status = "crashed"
    finally:
        os.remove(tmp_config)

    if os.path.exists(lifetime_tmp):
        with open(lifetime_tmp) as f:
            lifetime = json.load(f)
        os.remove(lifetime_tmp)
    else:
        lifetime = {"buffers": [], "phases": [], "checkpoints": [],
                    "device_baseline_gb": device_baseline_gb, "totals": {}}

    n_callbacks_with_active = (driver_result or {}).get("n_callbacks_with_active", 0) or 0
    n_evals_while_active = (driver_result or {}).get("n_evals_while_active", 0) or 0
    if experiment_status == "ok" and (n_callbacks_with_active < 3 or n_evals_while_active < 3):
        # module docstring's artifact postcondition: the IO term never
        # actually turned on/evaluated enough to be a valid T2b measurement.
        experiment_status = "instrumentation_error"

    totals = lifetime.get("totals") or {}
    peak_gb = totals.get("peak_alloc_gb_run")
    resident_lower_bound_gb = totals.get("resident_max_gb")
    verdict = feasibility_verdict(
        experiment_status, workload_status, peak_gb, budget_gb,
        oom_repeats=(1 if workload_status == "oom" else 0),
        resident_lower_bound_gb=resident_lower_bound_gb)

    env = env_metadata(REPO, DP, input_paths=(config_json,))
    record = {
        "record_kind": "lifetime",   # T11's m4_report_lint.py excludes this from quality/scaling tables
        "case": case, "K": k, "rtype": rtype, "arm": arm, "benchmark_kind": benchmark_kind,
        "budget_gb": budget_gb, "budget_source": budget_source,
        "experiment_status": experiment_status, "workload_status": workload_status,
        "feasibility_verdict": verdict,
        "device_baseline_gb": device_baseline_gb,
        # M4 T1b (f): sec 1.4 B3's `baseline_reserved_gb` field, carried
        # here too so both spike_30m.py's SPIKE_SCHEMA_FIELDS artifacts and
        # this probe's lifetime artifacts expose the same pre-run device
        # baseline under the same name -- `device_baseline_gb` above is the
        # authoritative field for LIFETIME_SCHEMA_FIELDS (result_gate.py);
        # this is a same-value alias for cross-artifact-kind consistency,
        # not a second measurement.
        "baseline_reserved_gb": device_baseline_gb,
        # sec 1.4 D9 (T1b's exclusivity protocol): this probe doesn't run
        # the `nvidia-smi --query-compute-apps` self-test T1b specifies --
        # deliberately conservative, always the lower evidence tier.
        "exclusivity_evidence": "baseline_only",
        "host_mem_available_gb_at_start": _host_mem_available_gb(),
        "buffers": lifetime.get("buffers", []), "phases": lifetime.get("phases", []),
        "totals": totals,
        "n_callbacks_with_active": n_callbacks_with_active,
        "n_evals_while_active": n_evals_while_active,
        "of_on": of_on,
        "gp_iterations_run": (driver_result or {}).get("gp_iterations_run"),
        "run_id": str(uuid.uuid4()), "repo_commit": env["ioplace_commit"],
        "dp_commit": env["dp_commit"], "command": " ".join(sys.argv),
        "hostname": socket.gethostname(), "gpu_name": gpu_name, "seed": seed,
        "input_sha256": env["input_sha256"],
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[probe_lifetime_gate] case={case} k={k} arm={arm} "
          f"experiment_status={experiment_status} workload_status={workload_status} "
          f"feasibility_verdict={verdict} wrote {out_json}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho-max", type=float, default=0.1)
    ap.add_argument("--of-on", type=float, default=2.0)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--diag-every", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmark-kind", default="real", choices=["real", "synthetic"])
    args = ap.parse_args(argv)
    run(args.config, args.case, args.k, args.rtype, args.seed,
       rho_max=args.rho_max, of_on=args.of_on, iterations=args.iterations,
       diag_every=args.diag_every, out_json=args.out,
       benchmark_kind=args.benchmark_kind)


if __name__ == "__main__":
    main()
