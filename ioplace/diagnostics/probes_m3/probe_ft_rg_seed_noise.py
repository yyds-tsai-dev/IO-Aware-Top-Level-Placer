"""M3 Phase B step 1 (docs/results/2026-08-14-m3-s4-adjudication.md sec C item
4, the "cheap" sigma_seed(ft_rg) fill-in): re-evaluate the 5 existing
seed-noise placements with the current (T1) GPU evaluator and report
mean/sigma/3*sigma for `ft_rg`, alongside `ft_mst` (`ft_count`) for
cross-check against the already-recorded values.

No new placement is run -- this loads the 5 final positions already on disk
(results/m2/noise/flat_seed100{0..4}.json.npz, adaptec1 k16 grid, region
seed=0, det=1: the same config ioplace/diagnostics/measure_noise_floor.py used
for its seed sweep, dp_seed in {1000..1004}) and re-runs evaluate_gpu on them.
`ft_mst`/`io_mst` are expected to reproduce the already-recorded `ft_count`/
`io_count` fields in each flat_seed*.json exactly (evaluator-CPU vs GPU
crossings/ft/lambda fields are bit-exact by construction per
evaluator_gpu.py's module docstring) -- this is checked and any mismatch is
surfaced in the output rather than silently ignored. `ft_rg`/`io_rg` are new
(T1, commit 0a2e32c) and have no prior recorded value to check against.

sigma here is the sample stdev (statistics.stdev, ddof=1), matching
measure_noise_floor.py's `_stats()` and the design draft's already-published
`ft_mst`: mean 2,538.0 / sigma 20.98 / 3*sigma 62.9 (design draft
docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:581).

Usage: PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_ft_rg_seed_noise
Writes results/m3/probes/ft_rg_seed_noise.json (sealed schema, pattern copied
from probe_p0b.py's _env_metadata/_sha256/_git_head).
"""
import datetime
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"
K, RTYPE, REGION_SEED, DET = 16, "grid", 0, 1
DP_SEEDS = (1000, 1001, 1002, 1003, 1004)
OUT_JSON = os.path.join(REPO, "results/m3/probes/ft_rg_seed_noise.json")


# ---------------------------------------------------------------------------
# provenance (pattern copied from probe_p0b.py)
# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(argv, input_relpaths):
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": f"{sys.executable} -m ioplace.diagnostics.probes_m3.probe_ft_rg_seed_noise",
        "argv": list(argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in sorted(set(input_relpaths))
                         if os.path.exists(os.path.join(REPO, p))},
    }


def _stats(values):
    mean = statistics.mean(values)
    sigma = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "values": list(values), "mean": mean, "sigma": sigma,
            "three_sigma": 3.0 * sigma}


def main():
    params, placedb = _load_dreamplace(CFG)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, K, RTYPE, REGION_SEED))
    ctx = GpuEvalContext(nl, rg, device="cuda")

    per_seed = []
    input_relpaths = []
    for dp_seed in DP_SEEDS:
        npz_rel = f"results/m2/noise/flat_seed{dp_seed}.json.npz"
        json_rel = f"results/m2/noise/flat_seed{dp_seed}.json"
        input_relpaths += [npz_rel, json_rel]
        d = np.load(os.path.join(REPO, npz_rel))
        res = ctx.evaluate(d["node_x"], d["node_y"])
        recorded = json.load(open(os.path.join(REPO, json_rel)))
        per_seed.append({
            "dp_seed": dp_seed,
            "ft_rg": int(res.ft_rg), "ft_mst": int(res.ft_count),
            "io_rg": int(res.io_rg), "io_mst": int(res.io_count),
            "recorded_ft_mst": recorded["ft_count"], "recorded_io_mst": recorded["io_count"],
        })

    mismatched = [r["dp_seed"] for r in per_seed
                 if r["ft_mst"] != r["recorded_ft_mst"] or r["io_mst"] != r["recorded_io_mst"]]

    result = {
        "probe": "ft_rg_seed_noise",
        "config": {
            "config": CFG, "k": K, "rtype": RTYPE, "region_seed": REGION_SEED,
            "dp_seeds": list(DP_SEEDS), "det": DET,
            "source": "results/m2/noise/flat_seed100{0..4}.json.npz "
                      "(re-evaluated with the current evaluator, no new placement run)",
        },
        "env": _env_metadata(sys.argv, input_relpaths),
        "per_seed": per_seed,
        "ft_mst_io_mst_vs_recorded_mismatched_seeds": mismatched,
        "ft_rg": _stats([r["ft_rg"] for r in per_seed]),
        "ft_mst": _stats([r["ft_mst"] for r in per_seed]),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
