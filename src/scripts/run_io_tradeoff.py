"""Run matched DREAMPlace GP+LG arms and select IO under a HPWL budget.

Each arm has a fresh process. The output directory must be new: historical
results are never silently reused. selected.json references the actual saved
placement, including the flat baseline when all candidates fail the gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ioplace.bench.tradeoff import select


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.1, 0.2, 0.4])
    parser.add_argument("--hpwl-budget", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--benchmark-kind", choices=["real", "synthetic"], default="real")
    parser.add_argument("--emit-def", action="store_true")
    args = parser.parse_args()
    import math
    if (not math.isfinite(args.hpwl_budget) or args.hpwl_budget < 0
            or any(not math.isfinite(r) or r <= 0 for r in args.rhos)
            or len(set(args.rhos)) != len(args.rhos)
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("require finite nonnegative budget, unique positive rhos and unique seeds")
    repo = Path(__file__).resolve().parents[1]
    repo = repo.parent if repo.name == "src" else repo
    config = Path(args.config).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_files = sorted((Path(__file__).resolve().parents[1] / "ioplace").rglob("*.py")) + [Path(__file__).resolve()]
    protocol = {"argv": sys.argv, "config": str(config), "config_sha256": digest(config),
                "source_sha256": {str(p.relative_to(repo)): digest(p) for p in source_files},
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "mtkahypar_threads": os.environ.get("IOPLACE_MTKAHYPAR_THREADS"),
                "rhos": [0., *args.rhos], "seeds": args.seeds,
                "hpwl_budget": args.hpwl_budget, "routing_verified": False,
                "started_unix": time.time()}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    summary = {"protocol": str(out / "protocol.json"), "seeds": {}}
    for seed in args.seeds:
        results, paths = [], []
        for rho in [0., *args.rhos]:
            arm = out / f"seed{seed}_rho{rho:g}"
            arm.mkdir()
            result_path = arm / "metrics.json"
            command = [sys.executable, "-m", "ioplace.drivers.run_placement", "--mode", "io",
                       "--config", str(config), "--out", str(result_path), "--k", str(args.k),
                       "--seed", "0", "--dp-seed", str(seed), "--deterministic", "1",
                       "--rho-max", str(rho), "--callback-order", "atomic", "--no-diag",
                       "--benchmark-kind", args.benchmark_kind]
            if args.emit_def:
                command += ["--emit-def", str(arm)]
            print(f"START seed={seed} rho={rho} {arm}", flush=True)
            started = time.time()
            with (arm / "run.log").open("w") as log:
                proc = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            receipt = {"command": command, "returncode": proc.returncode,
                       "wall_s": time.time() - started, "log_sha256": digest(arm / "run.log")}
            (arm / "receipt.json").write_text(json.dumps(receipt, indent=2))
            if proc.returncode:
                raise RuntimeError(f"arm failed ({proc.returncode}); see {arm / 'run.log'}")
            result = json.loads(result_path.read_text())
            results.append(result)
            paths.append(str(result_path))
            print(f"DONE IO={result['io_count']} HPWL={result['hpwl']} "
                  f"legal={result['legalization_status']} {receipt['wall_s']:.1f}s", flush=True)
        budgets = sorted(set([0., .02, .05, .10, args.hpwl_budget]))
        selections = {str(b): select(results[0], results[1:], hpwl_budget=b) for b in budgets}
        chosen = selections[str(args.hpwl_budget)]["selected_index"]
        index = 0 if chosen is None else chosen + 1
        summary["seeds"][str(seed)] = {"metrics": paths, "selections": selections,
            "selected_metrics": paths[index], "selected_placement": paths[index] + ".npz",
            "selected_placement_sha256": digest(paths[index] + ".npz"),
            "improved": chosen is not None}
        (out / "selected.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
