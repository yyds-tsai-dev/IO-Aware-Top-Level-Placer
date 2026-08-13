"""M3 T0 probe P1 (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 2.4, L3): how sensitive is `ft_mst` to the MST tree-edge L-shape walk
direction convention?

`ioplace/evaluator_ref.py:68-72` walks every tree edge horizontal-segment-first
(at the edge's first endpoint's row, y0) then vertical-segment-second (at the
second endpoint's column, x1) -- an arbitrary convention. This probe recomputes
`ft_count` under (a) that convention -- which must reproduce the saved
ablation/sweep JSON's `ft_count` bit-for-bit as a sanity check (evaluate() is
literally how those JSONs were produced) -- and under (b) the reversed
convention, vertical-segment-first (at x0) then horizontal-segment-second (at
y1). `net_mst_edges`/`_walk_segment` are imported unmodified from
evaluator_ref (order-independent); only the segment *order* is swapped, via a
probe-local copy of `evaluate`/`edge_regions_and_crossings` -- the production
evaluator at evaluator_ref.py is untouched.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_l_convention

Writes results/m3/probes/probe_l_convention.json directly (not via stdout
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
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import EvalResult, evaluate, net_mst_edges, _walk_segment

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"

RUNS = [
    dict(tag="adaptec1_A0_k16_grid", cfg=f"{DP}/install/test/ispd2005/adaptec1.json",
         saved_json="results/m2/ablation/adaptec1_A0_k16_grid.json",
         npz="results/m2/ablation/adaptec1_A0_k16_grid.json.npz", k=16, rtype="grid"),
    dict(tag="adaptec1_k16_rho0.40_annealed", cfg=f"{DP}/install/test/ispd2005/adaptec1.json",
         saved_json="results/m2/sweep/adaptec1_k16_rho0.40_annealed.json",
         npz="results/m2/sweep/adaptec1_k16_rho0.40_annealed.json.npz", k=16, rtype="grid"),
    dict(tag="bigblue4_A0_k16_grid", cfg=f"{DP}/install/test/ispd2005/bigblue4.json",
         saved_json="results/m2/ablation/bigblue4_A0_k16_grid.json",
         npz="results/m2/ablation/bigblue4_A0_k16_grid.json.npz", k=16, rtype="grid"),
    dict(tag="bigblue4_A2_k16_grid", cfg=f"{DP}/install/test/ispd2005/bigblue4.json",
         saved_json="results/m2/ablation/bigblue4_A2_k16_grid.json",
         npz="results/m2/ablation/bigblue4_A2_k16_grid.json.npz", k=16, rtype="grid"),
]

REL_DIFF_LO = 0.02   # judgement hook: below this, "arbitrary convention" wording needs softening
REL_DIFF_HI = 0.10   # judgement hook: above this, sec 2.3's MST-veto reasoning gets a hard data point


def edge_regions_and_crossings_v(rg, x0, y0, x1, y1):
    """Reversed L-walk convention: vertical segment first (at x0), then
    horizontal segment second (at y1) -- the mirror image of
    evaluator_ref.edge_regions_and_crossings' horizontal-then-vertical (h,v)
    convention (corner at (x0,y1) instead of (x1,y0))."""
    regions, pairs = set(), []
    n1 = _walk_segment(rg, x0, y0, x0, y1, regions, pairs)   # vertical
    n2 = _walk_segment(rg, x0, y1, x1, y1, regions, pairs)   # horizontal
    return regions, n1 + n2, pairs


def evaluate_v(nl, node_x, node_y, rg, max_degree=256):
    """Probe-only copy of evaluator_ref.evaluate with the reversed
    (vertical-first) walk direction from edge_regions_and_crossings_v above.
    Everything else (MST edges, pin regions, hpwl, tree_wl bookkeeping) is
    identical to evaluate() -- only which regions/pairs a tree edge is
    credited with passing through can differ."""
    px, py = pin_positions(nl, node_x, node_y)
    start = nl.flat_net2pin_start
    n_nets = nl.num_nets
    per_net_crossings = np.zeros(n_nets, dtype=np.int32)
    per_net_ft = np.zeros(n_nets, dtype=np.int32)
    per_net_lambda = np.zeros(n_nets, dtype=np.int32)
    pair_demand = {}
    tree_wl = 0.0
    hpwl = 0.0
    large_lb = 0
    pin_rid_all = rg.region_of_points(px, py)
    for net in range(n_nets):
        s, e = start[net], start[net + 1]
        d = e - s
        if d <= 1:
            continue
        pin_idx = nl.flat_net2pin[s:e]
        nx_, ny_ = px[pin_idx], py[pin_idx]
        hpwl += (nx_.max() - nx_.min()) + (ny_.max() - ny_.min())
        pin_regions = set(pin_rid_all[pin_idx].tolist())
        per_net_lambda[net] = len(pin_regions)
        if d > max_degree:
            lb = len(pin_regions) - 1
            per_net_crossings[net] = lb
            large_lb += lb
            continue
        edges = net_mst_edges(nx_, ny_)
        passed = set()
        ncross = 0
        for (a, b) in edges:
            regs, nc, pairs = edge_regions_and_crossings_v(
                rg, nx_[a], ny_[a], nx_[b], ny_[b])
            passed |= regs
            ncross += nc
            for pr in pairs:
                pair_demand[pr] = pair_demand.get(pr, 0) + 1
            tree_wl += abs(nx_[a] - nx_[b]) + abs(ny_[a] - ny_[b])
        per_net_crossings[net] = ncross
        per_net_ft[net] = len(passed - pin_regions)
    return EvalResult(
        io_count=int(per_net_crossings.sum()),
        ft_count=int(per_net_ft.sum()),
        tree_wl=float(tree_wl), hpwl=float(hpwl),
        per_net_crossings=per_net_crossings, per_net_ft=per_net_ft,
        boundary_pair_demand=pair_demand, large_net_lb=int(large_lb),
        hard_lambda_sum=int(np.maximum(per_net_lambda - 1, 0).sum()),
        per_net_lambda=per_net_lambda)


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


def run() -> dict:
    results = []
    for r in RUNS:
        params, placedb = _load_dreamplace(r["cfg"])
        placedb.initialize(params)
        nl = netlist_from_placedb(placedb)
        die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
        rg = RegionGrid(get_regions_for(die, r["k"], r["rtype"], 0))

        npz_path = os.path.join(REPO, r["npz"])
        d = np.load(npz_path)
        node_x, node_y = d["node_x"], d["node_y"]

        saved = json.load(open(os.path.join(REPO, r["saved_json"])))
        saved_ft = int(saved["ft_count"])

        res_h = evaluate(nl, node_x, node_y, rg)
        if res_h.ft_count != saved_ft:
            raise RuntimeError(
                f"P1 sanity check FAILED for {r['tag']}: evaluate() ft_count="
                f"{res_h.ft_count} != saved ft_count={saved_ft} ({r['saved_json']}). "
                "Stopping -- see task instructions (do not proceed on a failed sanity check).")

        res_v = evaluate_v(nl, node_x, node_y, rg)

        ft_h, ft_v = res_h.ft_count, res_v.ft_count
        rel_diff = abs(ft_v - ft_h) / ft_h if ft_h > 0 else float("nan")

        results.append({
            "tag": r["tag"], "k": r["k"], "rtype": r["rtype"],
            "saved_json": r["saved_json"], "npz": r["npz"],
            "saved_ft_count": saved_ft, "sanity_check_passed": True,
            "ft_h": ft_h, "ft_v": ft_v,
            "io_h": res_h.io_count, "io_v": res_v.io_count,
            "rel_diff": rel_diff,
            "flag_rel_diff_gt_10pct": bool(rel_diff > REL_DIFF_HI),
            "flag_rel_diff_lt_2pct": bool(rel_diff < REL_DIFF_LO),
        })

    return {
        "env": _env_metadata([r["npz"] for r in RUNS] + [r["saved_json"] for r in RUNS]),
        "convention_h": "horizontal segment first (at y0), then vertical segment (at x1) "
                         "-- evaluator_ref.py:68-72, the production convention",
        "convention_v": "vertical segment first (at x0), then horizontal segment (at y1) "
                         "-- reversed, probe-only",
        "rel_diff_lo_threshold": REL_DIFF_LO,
        "rel_diff_hi_threshold": REL_DIFF_HI,
        "runs": results,
    }


OUT_RELPATH = "results/m3/probes/probe_l_convention.json"

if __name__ == "__main__":
    result = run()
    out_path = os.path.join(REPO, OUT_RELPATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_l_convention] wrote {out_path}")
