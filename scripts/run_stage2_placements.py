"""Generate the 3-design, 2-K, 2-arm calibration cohort in fresh processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    configs = root / "configs"
    manifest = []
    for design in ("mgc_fft_1", "des_perf_1", "mempool_tile_wrap"):
        config = configs / f"{design}.json"
        for k in (16, 32):
            for mode, rho in (("flat", 0.), ("ours", .2)):
                arm = f"{mode}_k{k}"
                manifest.append(dict(design=design, arm=arm, k=k, rho=rho, seed=2000,
                    config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    plan = root / "cohort.json"
    if plan.exists() and json.loads(plan.read_text()) != manifest:
        raise ValueError("cohort configuration changed; use a new root")
    plan.write_text(json.dumps(manifest, indent=2))
    for item in manifest:
        run = root / (item["design"] + "__" + item["arm"])
        run.mkdir(exist_ok=True)
        out = run / "metrics.json"
        if out.exists():
            result = json.loads(out.read_text())
            if result.get("workload_status") == "completed" and (run / "out.def").exists() and (run / f"evaluator_k{item['k']}.npz").exists():
                print("completed", run.name, flush=True)
                continue
            raise ValueError(f"partial existing placement requires review: {run}")
        command = [sys.executable, "-m", "ioplace.drivers.run_placement", "--config", item["config"],
            "--mode", "io", "--k", str(item["k"]), "--rho-max", str(item["rho"]),
            "--seed", "0", "--dp-seed", str(item["seed"]), "--deterministic", "1",
            "--no-diag", "--out", str(out), "--emit-def", str(run)]
        print("start", run.name, flush=True)
        with open(run / "placement.log", "w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            (run / "execution.json").write_text(json.dumps(dict(command=command, pid=process.pid,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), status="running"), indent=2))
            rc = process.wait()
        (run / "execution.json").write_text(json.dumps(dict(command=command, pid=process.pid,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), returncode=rc,
            status="completed" if rc == 0 else "failed"), indent=2))
        print("finish", run.name, rc, flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
