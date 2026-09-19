"""Execute preregistered FT arms in isolated processes from one source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot():
    args = ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader"]
    selection = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
    if selection:
        args += ["--id", selection]
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase", choices=["levers", "noise", "pilot", "confirmation"], default="levers")
    args = parser.parse_args()
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    if not source.exists():
        source.mkdir()
        shutil.copytree(Path(__file__).resolve().parents[1] / "ioplace", source / "ioplace",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen_hashes = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*.py"))}
    dp = Path(os.environ["DREAMPLACE_ROOT"])
    config = root / "adaptec1.json"
    if not config.exists():
        data = json.loads((dp / "install/test/ispd2005/adaptec1.json").read_text())
        data.update(detailed_place_flag=0, plot_flag=0, num_threads=8)
        config.write_text(json.dumps(data, indent=2))
    registration = REPO / "docs/experiments/2026-09-06-ft-preregistration.md"
    manifest_path = root / "manifest.json"
    expected = dict(registration_sha256=sha(registration), source_sha256=frozen_hashes,
                    config_sha256=sha(config), sharing="user_authorized_shared_gpu")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != expected:
            raise ValueError("registered inputs or source snapshot changed; use a new run directory")
    else:
        manifest_path.write_text(json.dumps(expected, indent=2))
    common = [sys.executable, "-m", "ioplace.drivers.run_placement", "--config", str(config),
              "--mode", "io", "--k", "16", "--rtype", "grid", "--seed", "0",
              "--deterministic", "1", "--every", "50", "--rho-max", ".40",
              "--tau-hi", ".30", "--tau-lo", ".03", "--callback-order", "atomic", "--no-diag"]
    lever_arms = {
        "A0": [],
        "A1": ["--wl-reweight", "crossings", "--alpha-wl", ".2", "--wl-cap", "10"],
        "A2": ["--wl-reweight", "ft_rg", "--alpha-wl", ".2", "--wl-cap", "10"],
        "A3": ["--alpha-io", ".2", "--cap", "10"],
        "A4": ["--ft-reweight", "on", "--alpha-ft", ".2", "--cap", "10"],
        "A5": ["--rho-max", "0", "--wl-reweight", "crossings", "--alpha-wl", ".2", "--wl-cap", "10"],
    }
    if args.phase == "levers":
        jobs = [(name, 1000, extra) for name, extra in lever_arms.items()]
    elif args.phase == "noise":
        jobs = [("flat", seed, ["--rho-max", "0"]) for seed in range(1000, 1005)]
    elif args.phase == "confirmation":
        jobs = [(name, seed, lever_arms[name]) for seed in (1001, 1002, 1003) for name in ("A0", "A2")]
    else:
        jobs = [(f"P{i}", 1000, ["--f-ft-max", str(value), "--topology-diagnostics",
                 "--home-period", "50", "--ft-ramp-mode", "window", "--tau-start", ".12", "--tau-full", ".05"])
                for i, value in enumerate((0., .1, .25, .5))]
    for arm, seed, extra in jobs:
        out = root / f"{arm}_seed{seed}.json"
        record_path = out.with_suffix(".execution.json")
        if out.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("returncode") == 0 and record.get("output_sha256") == sha(out):
                print(f"verified completed {out.name}", flush=True)
                continue
            raise ValueError(f"incomplete existing attempt requires review: {out}")
        command = common + ["--dp-seed", str(seed), "--out", str(out)] + extra
        record = dict(command=command, before_gpu=gpu_snapshot(), started_at=time.time(),
                      source_manifest_sha256=sha(manifest_path), status="running")
        record_path.write_text(json.dumps(record, indent=2))
        print(f"start {arm} seed={seed}", flush=True)
        env = dict(os.environ, PYTHONPATH=str(source))
        with open(out.with_suffix(".log"), "w") as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            record_path.write_text(json.dumps(record, indent=2))
            rc = process.wait()
        record.update(returncode=rc, finished_at=time.time(), after_gpu=gpu_snapshot(),
                      status="completed" if rc == 0 else "failed",
                      output_sha256=sha(out) if out.exists() else None)
        record_path.write_text(json.dumps(record, indent=2))
        print(f"finish {arm} seed={seed} rc={rc}", flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
