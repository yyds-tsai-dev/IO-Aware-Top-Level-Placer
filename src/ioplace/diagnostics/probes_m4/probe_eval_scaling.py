"""M4 T0 probe (design draft `docs/superpowers/specs/2026-08-13-m4-scale-
up-design-draft.md` sec 1.2 / sec 4.1): sweeps `evaluator_gpu.GpuEvalContext`
over synthetic (nets, K) points to fit the pre-T2 memory model `mem ~=
(0.85 + 0.061*K) GB per M-net` and supply sec 4.1's per-tensor breakdown
(the `(P,K)` pin one-hot at `evaluator_gpu.py:351` dominates).

Migrated from `/tmp/probe_m4_eval.py` (design draft sec 11): sealed
(repo-relative paths only, no /tmp dependency) and now writes its JSON
directly to `results/m4/probes/` instead of printing one line per row (the
original's `print("ROW " + ...)` streaming-progress convention is kept as a
side-channel stderr-safe stdout log, but the authoritative output is now
the JSON file, env metadata + input sha256 included per T0's sealing
requirement).

Purely synthetic (bigblue4 degree-bucket-weighted resample via
`ioplace.diagnostics.spike_10m._synthesize_netlist`, same as that module) --
no DREAMPlace config/case needed, so T0's "verify on a small case" applies
directly: the sweep's smallest point (0.6M nets) already IS the small-case
verification; the full 5-point sweep (largest: 4.8M nets/K=32, ~13.5GB
peak, fits in L4's 21.7GiB budget) runs in ~10s total, so it is run in
full here rather than truncated.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_eval_scaling
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch

from ioplace.diagnostics.spike_10m import _synthesize_netlist
from ioplace.drivers.run_placement import get_regions_for
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
DEGREE_DIST_JSON = "results/m2/probes/degree_distribution.json"   # _synthesize_netlist's input

# (n_nodes, n_nets, K) -- design draft sec 1.2's exact five points.
SWEEP_POINTS = [
    (500_000, 600_000, 32),
    (1_000_000, 1_200_000, 32),
    (2_000_000, 2_400_000, 32),
    (2_000_000, 2_400_000, 16),
    (4_000_000, 4_800_000, 32),
]

DIE = (0.0, 0.0, 100_000.0, 100_000.0)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_relpaths):
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "dp_commit": _git_head(DP),
        "ioplace_commit": _git_head(REPO),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
    }


def run(sweep_points=SWEEP_POINTS) -> dict:
    rows = []
    for n_nodes, n_nets, K in sweep_points:
        nl, n_pins = _synthesize_netlist(n_nodes, n_nets, DIE)
        rg = RegionGrid(get_regions_for(DIE, K, "grid", 0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        row = dict(n_nodes=n_nodes, n_nets=n_nets, n_pins=n_pins, K=K)
        ctx = res = None
        try:
            t0 = time.time()
            ctx = GpuEvalContext(nl, rg, device="cuda")
            t1 = time.time()
            res = ctx.evaluate(nl.node_x, nl.node_y)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.time()
            row.update(ctx_s=t1 - t0, eval_s=t2 - t1,
                      peak_gb=(torch.cuda.max_memory_allocated() / 2**30
                               if torch.cuda.is_available() else 0.0),
                      io=res.io_count, ok=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            row.update(ok=False, err=str(e)[:150],
                      peak_gb=(torch.cuda.max_memory_allocated() / 2**30
                               if torch.cuda.is_available() else 0.0))
        del nl, rg, ctx, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows.append(row)
        print(f"[probe_eval_scaling] {row}")
    return {"env": _env_metadata([DEGREE_DIST_JSON]), "rows": rows}


if __name__ == "__main__":
    result = run()
    out_path = os.path.join(REPO, "results", "m4", "probes", "probe_eval_scaling.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_eval_scaling] wrote {out_path}")
