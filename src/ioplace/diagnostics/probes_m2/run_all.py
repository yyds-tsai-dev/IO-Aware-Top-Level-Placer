import hashlib, json, os, subprocess, sys
import numpy as np, torch
from ioplace.diagnostics.probes_m2 import (probe1_numerics, probe2_scale, probe3_grad,
    probe4_loo_grad, probe5_naive_vs_stable, probe6_mem, probe7_gap, probe8_secant)

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
from ioplace.paths import REPO_ROOT
REPO = str(REPO_ROOT)
NPZ = ["results/m1/adaptec1_reweight_k8_grid.json.npz",
       "results/m1/adaptec1_reweight_k16_grid.json.npz",
       "results/m1/adaptec1_reweight_k32_grid.json.npz",
       "results/m1/bigblue4_reweight_k16_grid.json.npz"]

def _git(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def main(out_dir="results/m2/probes"):
    os.makedirs(out_dir, exist_ok=True)
    env = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "numpy_version": np.__version__, "dp_commit": _git(DP), "ioplace_commit": _git(REPO),
           "input_sha256": {p: _sha(os.path.join(REPO, p)) for p in NPZ}}
    json.dump(env, open(os.path.join(out_dir, "env.json"), "w"), indent=1)
    jobs = [("probe1", probe1_numerics.run, {}), ("probe2", probe2_scale.run, {}),
            ("probe3", probe3_grad.run, {}), ("probe4", probe4_loo_grad.run, {}),
            ("probe5", probe5_naive_vs_stable.run, {}), ("probe6", probe6_mem.run, {"case": "bigblue4"}),
            ("probe7", probe7_gap.run, {}), ("probe8", probe8_secant.run, {})]
    for name, fn, kw in jobs:
        json.dump(fn(**kw), open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
        print(f"[probes_m2] wrote {name}.json")

if __name__ == "__main__":
    main(*sys.argv[1:])
