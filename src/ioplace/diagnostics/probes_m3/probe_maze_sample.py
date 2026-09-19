"""M3 T11: maze sampling validator -- RG-Steiner geometric optimism.

Design draft `docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`
sec 2.4 L1 / sec 7 G4 / sec 8 T11 row (this is the "T11 節" the task refers
to -- there is no separate deep-dive subsection in v3.1; the operative text
is the T11 task-table row plus L1's threshold and G4's "以 T11 的 maze 結果
判定" pointer): `ft_rg`/`io_rg` (== `ST_e` per net, `EvalResult.per_net_steiner`)
is computed purely on the region-adjacency graph `G_R` (region ids as nodes,
hop distance as edge weight) -- it never touches actual pin coordinates, so
its Steiner-tree cost could be *optimistic* (too low) relative to a route a
real router would have to take through the physical 2D die. T11 quantifies
that gap directly: for a sample of active (`Lambda_e >= 2`) nets, this probe
runs Dijkstra on the *fine* region-id lattice -- literally `RegionGrid.grid`,
the same (512, 512) array `RegionGrid.region_of_points` /
`RegionGrid.pin_region_bitmask` index into (`ioplace/region_grid.py`,
`RegionSet.lattice` defaults to 512 everywhere in this repo -- see
`ioplace/regions.py`), so "same geometry as evaluator" is exact, not
approximate -- from the net's actual pin positions, with edge
cost = 1.0 * is_crossing + 1e-6 * 1 (crossing=1, length=eps=1e-6, per the
task instruction). `EPS_LENGTH` only breaks ties toward the physically
shortest route *among* min-crossing routes: the lattice is 512x512, so any
simple path has at most ~1022 steps, and 1022 * 1e-6 ~= 1e-3 << 1 -- far
below the smallest possible crossing-count difference (1), so it can never
flip a crossing-count comparison. The resulting crossing count is compared,
per net, against `per_net_steiner` (T1's `EvalResult.per_net_steiner`, taken
as-is -- this probe does not recompute ST_e; "RG 模型的 per_net_steiner
(T1 evaluator 直接給)").

Method -- the task instruction's explicit either/or ("兩兩 terminal 或 star
近似"): this probe uses a single-source Dijkstra *star* rooted at one
terminal cell (the net's lowest-(iy,ix) terminal, a deterministic tie-break).
`Lambda_e == 2` nets (the overwhelming majority of active nets: design draft
sec 2.2 measures 78.7-97.5% share) get an *exact* 2-point shortest path this
way (the root already is one of the 2 terminals, so this degenerates to
plain point-to-point Dijkstra). `Lambda_e >= 3` nets get the union of the
root's shortest paths to every other terminal, deduplicated via Dijkstra's
own shortest-path-tree predecessor structure (walking each terminal's
predecessor chain back to the root, stopping the moment an already-visited
node is hit) -- a feasible upper bound on the true optimal geometric Steiner
tree, not necessarily exact, exactly as the task's "or star approximation"
option licenses. Because the true geometric minimum-crossing Steiner tree
must induce a connected sub-structure of `G_R` spanning every touched region,
its crossing count is always >= `ST_e` (`G_R`'s own minimum Steiner cost) --
so `maze_crossings - st_e >= 0` is an invariant of *any* feasible geometric
tree, star heuristic included; `n_negative_deviation` in the summary is a
correctness sanity check on this probe's own code, not an expected finding.

Each net's terminal cells are one representative point *per touched region*
(that region's pin-centroid for this net, mapped to its fine-grid cell) --
not one point per pin -- matching `ST_e`'s own region-granularity terminal
set (RG's Steiner tree only ever sees "region X is touched", never *which*
pin inside it). A rectangular region is convex, so a centroid of that
region's pins always maps back into the same region's fine cell.

Search is restricted to the bounding box of a net's terminal cells + a
padding margin (auto-doubled and retried if the margin turns out
insufficient) rather than the full 512x512 lattice, purely for speed: on a
*grid*-type `RegionSet` (this probe's only target, adaptec1 k16 grid) each
region is a full-die-spanning rectangle, so the minimum-crossing route
between any two cells is always achievable within their bounding rectangle
(leaving it can only add length, never remove a required crossing) -- the
padding/retry loop is safety headroom for that assumption, not load-bearing.

Usage (CPU-only; a single invocation covers both A0 and A2 -- "分批" here
means capping wall-clock via --budget-seconds per tag, honestly recording
however many of the 10,000-net sample target were reached):
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_maze_sample
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_maze_sample \\
        --n-samples 2000 --budget-seconds 600   # smoke-test-sized run

Writes results/m3/probes/probe_maze_sample.json.
"""
import argparse
import datetime
import hashlib
import heapq
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate as evaluate_ref

from ioplace.paths import REPO_ROOT
REPO = str(REPO_ROOT)
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"

K = 16
RTYPE = "grid"
REGION_SEED = 0
LAMBDA_MIN = 2
EPS_LENGTH = 1e-6
BOUNDED_DETOUR_THRESHOLD = 0.20   # design draft sec 2.4 L1 / sec 8 T11 row: median > 20%

PLACEMENTS = (
    dict(tag="A0_flat", npz="results/m2/ablation/adaptec1_A0_k16_grid.json.npz"),
    dict(tag="A2_best", npz="results/m2/ablation/adaptec1_A2_k16_grid.json.npz"),
)


# ---------------------------------------------------------------------------
# provenance (pattern established by probe_p0b.py / probe_ft_surrogate_soft.py)
# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_relpaths, exactness):
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": " ".join([sys.executable, "-m",
                             "ioplace.diagnostics.probes_m3.probe_maze_sample"] + sys.argv[1:]),
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": exactness,
    }


def _atomic_write_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# fine-lattice maze Dijkstra (star approximation)
# ---------------------------------------------------------------------------
def _net_terminal_cells(rg, px, py, pin_rid, pin_idx):
    """One (ix, iy) fine-grid cell per touched region: that region's
    pin-centroid for this net. Returns a list, len == number of distinct
    regions touched (== Lambda_e)."""
    rids = pin_rid[pin_idx]
    uniq = np.unique(rids)
    cells = []
    for r in uniq:
        m = rids == r
        cx = float(px[pin_idx[m]].mean())
        cy = float(py[pin_idx[m]].mean())
        ix, iy = rg.to_idx(np.array([cx]), np.array([cy]))
        cells.append((int(ix[0]), int(iy[0])))
    return cells


def _dijkstra_star_crossings_once(grid, terminals, pad):
    """One attempt at the star-Dijkstra crossing count, restricted to
    terminals' bounding box + `pad`. Returns None if the search exhausts the
    padded box before reaching every terminal (caller retries with more
    pad) -- see module docstring for why pad is safety headroom, not
    load-bearing, on this probe's grid-partition target."""
    nx_full, ny_full = grid.shape[1], grid.shape[0]
    xs = [c[0] for c in terminals]
    ys = [c[1] for c in terminals]
    x0 = max(0, min(xs) - pad)
    x1 = min(nx_full - 1, max(xs) + pad)
    y0 = max(0, min(ys) - pad)
    y1 = min(ny_full - 1, max(ys) + pad)
    sub = grid[y0:y1 + 1, x0:x1 + 1]
    H, W = sub.shape

    root = (terminals[0][0] - x0, terminals[0][1] - y0)
    targets = set((c[0] - x0, c[1] - y0) for c in terminals)

    INF = float("inf")
    dist = np.full((H, W), INF, dtype=np.float64)
    pred_x = np.full((H, W), -1, dtype=np.int32)
    pred_y = np.full((H, W), -1, dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)

    dist[root[1], root[0]] = 0.0
    heap = [(0.0, root[0], root[1])]
    remaining = set(targets)
    remaining.discard(root)

    while heap and remaining:
        d, cx, cy = heapq.heappop(heap)
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        remaining.discard((cx, cy))
        rid0 = sub[cy, cx]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < W and 0 <= ny_ < H and not visited[ny_, nx_]:
                step_cost = EPS_LENGTH + (1.0 if sub[ny_, nx_] != rid0 else 0.0)
                nd = d + step_cost
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd
                    pred_x[ny_, nx_] = cx
                    pred_y[ny_, nx_] = cy
                    heapq.heappush(heap, (nd, nx_, ny_))

    if remaining:
        return None

    crossing_visited = np.zeros((H, W), dtype=bool)
    crossing_visited[root[1], root[0]] = True
    total_crossings = 0
    for (tx, ty) in targets:
        if (tx, ty) == root:
            continue
        cx, cy = tx, ty
        while not crossing_visited[cy, cx]:
            px_, py_ = int(pred_x[cy, cx]), int(pred_y[cy, cx])
            if sub[cy, cx] != sub[py_, px_]:
                total_crossings += 1
            crossing_visited[cy, cx] = True
            cx, cy = px_, py_
    return total_crossings, H * W


def dijkstra_star_crossings(grid, terminals, pad0):
    """Star-Dijkstra min-crossing tree over `terminals` (>=2 distinct
    (ix,iy) cells), auto-doubling `pad` (capped at the full grid extent) if
    the initial padded box is too small to reach every terminal."""
    terminals = sorted(set(terminals), key=lambda c: (c[1], c[0]))
    pad = pad0
    max_dim = max(grid.shape)
    while True:
        result = _dijkstra_star_crossings_once(grid, terminals, pad)
        if result is not None:
            return result + (pad,)
        if pad >= max_dim:
            raise RuntimeError(
                f"maze search exhausted the full grid ({max_dim}x{max_dim}) "
                f"before reaching all terminals {terminals} -- this should be "
                f"unreachable (the grid is fully covered/connected)")
        pad = min(pad * 2, max_dim)


# ---------------------------------------------------------------------------
# per-tag run
# ---------------------------------------------------------------------------
def _sample_net_ids(per_net_lambda, n_target, seed):
    candidates = np.nonzero(per_net_lambda >= LAMBDA_MIN)[0]
    rng = np.random.default_rng(seed)
    n = min(n_target, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False)
    chosen.sort()
    return chosen, int(len(candidates))


def _run_tag(tag, npz_relpath, nl, rg, budget_seconds, n_target, sample_seed, pad0):
    npz_path = os.path.join(REPO, npz_relpath)
    d = np.load(npz_path)
    node_x, node_y = d["node_x"], d["node_y"]

    t_eval0 = time.time()
    res = evaluate_ref(nl, node_x, node_y, rg)
    eval_ref_seconds = time.time() - t_eval0

    px, py = pin_positions(nl, node_x, node_y)
    pin_rid = rg.region_of_points(px, py).astype(np.int64)
    start = nl.flat_net2pin_start
    flat_pin = nl.flat_net2pin

    net_ids, n_candidates = _sample_net_ids(res.per_net_lambda, n_target, sample_seed)

    rows = []
    n_terminal_mismatch = 0
    stopped_reason = "completed"
    t0 = time.time()
    for i, net in enumerate(net_ids):
        if time.time() - t0 > budget_seconds:
            stopped_reason = "budget_seconds_exceeded"
            break
        s, e = int(start[net]), int(start[net + 1])
        pin_idx = flat_pin[s:e]
        lam = int(res.per_net_lambda[net])
        cells = _net_terminal_cells(rg, px, py, pin_rid, pin_idx)
        if len(cells) != lam:
            # Should not happen (per_net_lambda IS len(set(touched regions)));
            # guard rather than silently mis-measure this net.
            n_terminal_mismatch += 1
            continue
        crossings, bbox_area, pad_used = dijkstra_star_crossings(rg.grid, cells, pad0)
        st_e = int(res.per_net_steiner[net])
        rows.append(dict(net=int(net), lam=lam, st_e=st_e, maze_crossings=int(crossings),
                          deviation=int(crossings - st_e), bbox_area=int(bbox_area),
                          pad_used=int(pad_used)))
    maze_seconds = time.time() - t0

    return dict(
        tag=tag, npz=npz_relpath, npz_sha256=_sha256(npz_path),
        n_active_nets_total=int(nl.num_nets),
        n_candidates_lambda_ge2=n_candidates,
        n_sample_target=n_target, n_sampled=len(rows), sample_seed=sample_seed,
        n_terminal_mismatch_skipped=n_terminal_mismatch,
        eval_ref_seconds=eval_ref_seconds, maze_seconds=maze_seconds,
        stopped_reason=stopped_reason,
        summary=_summarize(rows),
        rows=rows,
    )


def _summarize(rows):
    if not rows:
        return dict(n=0)
    dev = np.array([r["deviation"] for r in rows], dtype=np.float64)
    st = np.array([r["st_e"] for r in rows], dtype=np.float64)
    lam = np.array([r["lam"] for r in rows], dtype=np.int64)
    rel = dev / np.maximum(st, 1.0)

    vals, counts = np.unique(dev.astype(np.int64), return_counts=True)
    deviation_histogram = {str(int(v)): int(c) for v, c in zip(vals, counts)}

    out = dict(
        n=len(rows),
        deviation_mean=float(dev.mean()),
        deviation_median=float(np.median(dev)),
        relative_deviation_median=float(np.median(rel)),
        relative_deviation_p90=float(np.percentile(rel, 90)),
        n_negative_deviation=int((dev < 0).sum()),
        deviation_histogram=deviation_histogram,
        bounded_detour_threshold=BOUNDED_DETOUR_THRESHOLD,
        bounded_detour_needed=bool(np.median(rel) > BOUNDED_DETOUR_THRESHOLD),
    )

    by_lambda = {}
    for label, lo, hi in (("2", 2, 2), ("3", 3, 3), ("4+", 4, 10**9)):
        m = (lam >= lo) & (lam <= hi)
        if not m.any():
            continue
        by_lambda[label] = dict(
            n=int(m.sum()),
            relative_deviation_median=float(np.median(rel[m])),
            relative_deviation_p90=float(np.percentile(rel[m], 90)),
            bounded_detour_needed=bool(np.median(rel[m]) > BOUNDED_DETOUR_THRESHOLD),
        )
    out["by_lambda_bucket"] = by_lambda
    return out


def run(n_target, budget_seconds, sample_seed, pad0):
    params, placedb = _load_dreamplace(CFG)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, K, RTYPE, REGION_SEED)
    rg = RegionGrid(rs)

    tag_results = []
    for spec in PLACEMENTS:
        tag_results.append(_run_tag(spec["tag"], spec["npz"], nl, rg,
                                    budget_seconds, n_target, sample_seed, pad0))

    all_rows = [r for tr in tag_results for r in tr["rows"]]
    combined_summary = _summarize(all_rows)

    exactness = {
        "lambda2_maze": "exact (2-point Dijkstra shortest path)",
        "lambda_ge3_maze": "upper bound (single-source star heuristic; see module docstring)",
        "per_net_steiner_source": "T1 evaluator_ref.EvalResult.per_net_steiner, taken as-is "
                                  "(exact for Lambda<=8 via Dreyfus-Wagner, upper bound for "
                                  "Lambda>8 via metric-closure MST -- ioplace/region_graph.py)",
        "region_id_lattice": "RegionGrid.grid, the literal 512x512 array evaluator geometry "
                             "queries via region_of_points -- not a separate/approximate grid",
    }

    return {
        "probe": "probe_maze_sample",
        "env": _env_metadata([spec["npz"] for spec in PLACEMENTS], exactness),
        "config": dict(
            k=K, rtype=RTYPE, region_seed=REGION_SEED, lambda_min=LAMBDA_MIN,
            eps_length=EPS_LENGTH, bbox_pad0=pad0,
            n_sample_target=n_target, budget_seconds_per_tag=budget_seconds,
            sample_seed=sample_seed,
            bounded_detour_threshold=BOUNDED_DETOUR_THRESHOLD,
        ),
        "tags": tag_results,
        "summary": combined_summary,
    }


OUT_RELPATH = "results/m3/probes/probe_maze_sample.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None,
                    help="override auto-detected repo root (default: derived from __file__)")
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--budget-seconds", type=float, default=3600.0,
                    help="wall-clock cap per tag (A0/A2 each get their own budget)")
    ap.add_argument("--sample-seed", type=int, default=1000)
    ap.add_argument("--pad", type=int, default=8, help="initial bbox padding, in fine-grid cells")
    args = ap.parse_args()
    if args.repo_root:
        REPO = os.path.abspath(args.repo_root)

    result = run(args.n_samples, args.budget_seconds, args.sample_seed, args.pad)
    out_path = os.path.join(REPO, OUT_RELPATH)
    _atomic_write_json(result, out_path)
    print(f"[probe_maze_sample] wrote {out_path}")
    for tr in result["tags"]:
        s = tr["summary"]
        print(f"  {tr['tag']}: n={tr['n_sampled']} median_rel_dev={s.get('relative_deviation_median')} "
             f"p90_rel_dev={s.get('relative_deviation_p90')} bounded_detour_needed={s.get('bounded_detour_needed')}")
