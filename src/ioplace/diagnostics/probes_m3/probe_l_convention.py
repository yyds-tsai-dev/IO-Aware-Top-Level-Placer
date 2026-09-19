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

T0-b (design draft sec 8): reissued hermetic -- `REPO` derived from
`__file__` (overridable with `--repo-root`), atomic write, `exactness` added
to the unified `env` provenance schema. This probe takes ~35 minutes for all
4 runs (P1's `RUNS`), so it also gains a chunked-execution mode: `--tag TAG`
runs a single entry of `RUNS` and writes a per-tag checkpoint file, and
`--assemble` combines the 4 checkpoints (in `RUNS` order) into the final
unified-schema JSON. A plain no-argument invocation still runs all 4 in one
process, unchanged. Measured per-tag wall time (this evaluate() path is a
pure-Python per-net loop, so it scales with net count, not die size):
adaptec1 (221,142 nets) ~1m45s each, comfortably under a 590s timeout;
bigblue4 (2,229,886 nets, ~10x adaptec1's net count) ~16 minutes each --
past 590s, so the two bigblue4 tags need a longer per-invocation budget
(e.g. `timeout 1800`) even though they're still run one tag at a time.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_l_convention
    # -- or, chunked (adaptec1 tags fit under a 590s timeout; bigblue4 tags need longer, e.g. 1800s):
    PYTHONPATH=src timeout 590 $PY -m ioplace.diagnostics.probes_m3.probe_l_convention --tag adaptec1_A0_k16_grid
    PYTHONPATH=src timeout 590 $PY -m ioplace.diagnostics.probes_m3.probe_l_convention --tag adaptec1_k16_rho0.40_annealed
    PYTHONPATH=src timeout 1800 $PY -m ioplace.diagnostics.probes_m3.probe_l_convention --tag bigblue4_A0_k16_grid
    PYTHONPATH=src timeout 1800 $PY -m ioplace.diagnostics.probes_m3.probe_l_convention --tag bigblue4_A2_k16_grid
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_l_convention --assemble

Writes results/m3/probes/probe_l_convention.json directly (not via stdout
redirection): DREAMPlace's PlaceDB loader writes its own INFO/WARNING lines to
stdout, which would otherwise interleave with (and corrupt) a `> out.json`
redirect -- see probe_m3_rg.py's same direct-file-write convention. Per-tag
chunked runs write results/m3/probes/probe_l_convention.ckpt_<tag>.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import EvalResult, evaluate, net_mst_edges, _walk_segment

from ioplace.paths import REPO_ROOT
REPO = str(REPO_ROOT)
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
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "dp_commit": _git_head(DP),
        "repo_commit": _git_head(REPO),
        "command": " ".join([sys.executable, "-m", "ioplace.diagnostics.probes_m3.probe_l_convention"]
                            + sys.argv[1:]),
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": "exact (both L-walk conventions are literal geometric walks over the "
                    "region lattice, not approximations of each other)",
    }


def _atomic_write_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, out_path)


def _run_one(r: dict) -> dict:
    """One RUNS entry (one placement): sanity check against the saved JSON's
    ft_count, then the horizontal-first (production) vs vertical-first
    (probe-only) L-walk comparison. Self-contained (own PlaceDB load) so it
    can run as its own chunked CLI invocation -- see module docstring."""
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

    return {
        "tag": r["tag"], "k": r["k"], "rtype": r["rtype"],
        "saved_json": r["saved_json"], "npz": r["npz"],
        "saved_ft_count": saved_ft, "sanity_check_passed": True,
        "ft_h": ft_h, "ft_v": ft_v,
        "io_h": res_h.io_count, "io_v": res_v.io_count,
        "rel_diff": rel_diff,
        "flag_rel_diff_gt_10pct": bool(rel_diff > REL_DIFF_HI),
        "flag_rel_diff_lt_2pct": bool(rel_diff < REL_DIFF_LO),
    }


def _finalize(results: list) -> dict:
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


def run() -> dict:
    return _finalize([_run_one(r) for r in RUNS])


def _ckpt_path(tag):
    return os.path.join(REPO, f"results/m3/probes/probe_l_convention.ckpt_{tag}.json")


def run_tag(tag: str) -> None:
    r = next(x for x in RUNS if x["tag"] == tag)
    result = _run_one(r)
    _atomic_write_json(result, _ckpt_path(tag))
    print(f"[probe_l_convention] wrote checkpoint {_ckpt_path(tag)}")


def assemble() -> dict:
    results = []
    for r in RUNS:
        path = _ckpt_path(r["tag"])
        if not os.path.exists(path):
            raise RuntimeError(f"missing checkpoint for tag={r['tag']!r}: {path} -- run "
                              f"`--tag {r['tag']}` first")
        results.append(json.load(open(path)))
    return _finalize(results)


OUT_RELPATH = "results/m3/probes/probe_l_convention.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None,
                    help="override auto-detected repo root (default: derived from __file__)")
    ap.add_argument("--tag", default=None,
                    help="run only this RUNS entry, writing a checkpoint file (chunked mode)")
    ap.add_argument("--assemble", action="store_true",
                    help="combine the 4 per-tag checkpoints into the final JSON")
    args = ap.parse_args()
    if args.repo_root:
        REPO = os.path.abspath(args.repo_root)

    if args.tag:
        run_tag(args.tag)
        sys.exit(0)

    result = assemble() if args.assemble else run()
    out_path = os.path.join(REPO, OUT_RELPATH)
    _atomic_write_json(result, out_path)
    print(f"[probe_l_convention] wrote {out_path}")
