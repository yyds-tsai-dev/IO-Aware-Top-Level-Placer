"""M3 T0 probe P6 (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 4.1, L7): per-region area utilisation with the denominator swapped from
total region area to *free* area (region area minus the region's overlap with
fixed macros).

`ioplace/diagnostics/probes_m3/probe_m3_util.py` (already in-repo) computes
the total-area version (`movable_area / region_area`) and is the data source
this probe reuses (same cell-center region assignment, same npz/PlaceDB
inputs) -- only the denominator changes here. Fixed-macro geometry comes from
the PlaceDB-derived `Netlist`'s non-movable node slice (`node_x/y/size_x/size_y
[num_movable:]`), matching probe_m3_util.py's own data source (no LEF/DEF in
ISPD2005 bookshelf, so "macro" here means any fixed/terminal physical node).

Runs: adaptec1 k16 grid A0 (flat) vs A2 (M2 best), bigblue4 k16 grid A0 vs A2.
Per run: per-region movable_area/free_area max/mean and max/median. Judgement
hook: if A2's max/mean worsens by more than 10% relative to A0 (same case),
`l7_triggered` is set (design draft sec 4.1's "no area penalty term" call
needs re-review).

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_free_area_util

Writes results/m3/probes/probe_free_area_util.json directly (not via stdout
redirection): DREAMPlace's PlaceDB loader writes its own INFO/WARNING lines to
stdout, which would otherwise interleave with (and corrupt) a `> out.json`
redirect -- see probe_m3_rg.py's same direct-file-write convention.
"""
import hashlib
import json
import os
import subprocess

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"

CASES = [
    dict(case="adaptec1", cfg=f"{DP}/install/test/ispd2005/adaptec1.json",
         runs=[("A0_flat", "results/m2/ablation/adaptec1_A0_k16_grid.json.npz"),
               ("A2_best", "results/m2/ablation/adaptec1_A2_k16_grid.json.npz")]),
    dict(case="bigblue4", cfg=f"{DP}/install/test/ispd2005/bigblue4.json",
         runs=[("A0_flat", "results/m2/ablation/bigblue4_A0_k16_grid.json.npz"),
               ("A2_best", "results/m2/ablation/bigblue4_A2_k16_grid.json.npz")]),
]
K = 16
RTYPE = "grid"
L7_WORSEN_THRESHOLD = 0.10


def region_free_area(rs, nl):
    """Per-region (region_area, fixed_macro_overlap_area, free_area). Fixed
    macros = the non-movable tail of the Netlist's physical nodes
    (node_x/y/size_x/size_y[num_movable:]); overlap is the plain rectangle
    intersection area with each of the region's (possibly multiple, for
    slicing partitions) rects."""
    nm = nl.num_movable
    fx0 = nl.node_x[nm:]
    fy0 = nl.node_y[nm:]
    fx1 = fx0 + nl.node_size_x[nm:]
    fy1 = fy0 + nl.node_size_y[nm:]

    n_regions = len(rs.regions)
    region_area = np.zeros(n_regions)
    overlap_area = np.zeros(n_regions)
    for rid, reg in enumerate(rs.regions):
        for (rxl, ryl, rxh, ryh) in reg.rects:
            region_area[rid] += (rxh - rxl) * (ryh - ryl)
            ow = np.clip(np.minimum(fx1, rxh) - np.maximum(fx0, rxl), 0, None)
            oh = np.clip(np.minimum(fy1, ryh) - np.maximum(fy0, ryl), 0, None)
            overlap_area[rid] += float((ow * oh).sum())
    free_area = region_area - overlap_area
    return region_area, overlap_area, free_area


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


def util(tag, npz_relpath, k, rtype, nl, placedb, region_area, overlap_area, free_area):
    d = np.load(os.path.join(REPO, npz_relpath))
    nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, 0))
    nm = nl.num_movable
    cx = nx_[:nm] + nl.node_size_x[:nm] / 2.0
    cy = ny_[:nm] + nl.node_size_y[:nm] / 2.0
    rid = rg.region_of_points(cx, cy)
    area = nl.node_size_x[:nm] * nl.node_size_y[:nm]
    acc = np.zeros(k)
    np.add.at(acc, rid, area)

    u_free = acc / np.maximum(free_area, 1e-9)
    u_total = acc / region_area   # cross-reference to probe_m3_util.json's total-area version

    return dict(
        tag=tag, k=k, rtype=rtype, npz=npz_relpath,
        movable_area=[round(float(v), 2) for v in acc],
        region_area=[round(float(v), 2) for v in region_area],
        fixed_overlap_area=[round(float(v), 2) for v in overlap_area],
        free_area=[round(float(v), 2) for v in free_area],
        n_nonpositive_free_area_regions=int((free_area <= 0).sum()),
        util_free=[round(float(v), 4) for v in u_free],
        mean=float(u_free.mean()), min=float(u_free.min()), max=float(u_free.max()),
        median=float(np.median(u_free)),
        max_over_mean=float(u_free.max() / u_free.mean()),
        max_over_median=float(u_free.max() / np.median(u_free)),
        total_area_max_over_mean=float(u_total.max() / u_total.mean()),
        total_area_max_over_median=float(u_total.max() / np.median(u_total)),
    )


def run() -> dict:
    all_npz = []
    case_results = []
    for c in CASES:
        params, placedb = _load_dreamplace(c["cfg"])
        placedb.initialize(params)
        nl = netlist_from_placedb(placedb)
        die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
        rs = get_regions_for(die, K, RTYPE, 0)
        region_area, overlap_area, free_area = region_free_area(rs, nl)
        fixed_area_sum = float((nl.node_size_x[nl.num_movable:]
                                * nl.node_size_y[nl.num_movable:]).sum())

        run_out = []
        for tag, npz_relpath in c["runs"]:
            all_npz.append(npz_relpath)
            run_out.append(util(tag, npz_relpath, K, RTYPE, nl, placedb,
                                region_area, overlap_area, free_area))

        a0, a2 = run_out[0], run_out[1]
        rel_change_max_over_mean = (a2["max_over_mean"] - a0["max_over_mean"]) / a0["max_over_mean"]
        l7_triggered = bool(rel_change_max_over_mean > L7_WORSEN_THRESHOLD)

        case_results.append({
            "case": c["case"], "k": K, "rtype": RTYPE,
            "fixed_area_sum": fixed_area_sum,   # cross-check vs DP's own "fixed area total" log line
            "runs": run_out,
            "rel_change_max_over_mean_A2_vs_A0": rel_change_max_over_mean,
            "l7_triggered": l7_triggered,
        })

    return {
        "env": _env_metadata(all_npz),
        "l7_worsen_threshold": L7_WORSEN_THRESHOLD,
        "l7_triggered": any(cr["l7_triggered"] for cr in case_results),
        "cases": case_results,
    }


OUT_RELPATH = "results/m3/probes/probe_free_area_util.json"

if __name__ == "__main__":
    result = run()
    out_path = os.path.join(REPO, OUT_RELPATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_free_area_util] wrote {out_path}")
