"""Straddling diagnostics (design v2 sec 7, diagnostics 1-3).

This is the numpy reference. `evaluator_ref.evaluate` calls it;
`evaluator_gpu.GpuEvalContext` mirrors it in torch and
`tests/test_evaluator_gpu.py` pins the two together -- every integer field
bit-exact, the float fields at rel <= 1e-12, across batch sizes and chunk
budgets (spec sec 9).

Five conventions, fixed here because a convention that lives in two places
drifts:

1. Closed box, four corners. A movable cell occupies [x, x+w] x [y, y+h] and is
   judged by the region ids of its four corners, as sec 7 prescribes. The box is
   closed, so a cell whose right edge lands exactly on a region boundary counts
   as straddling -- the conservative direction, and the only convention
   RegionGrid.to_idx's truncate-and-clamp reproduces identically in numpy and
   torch without an epsilon.
2. Owner = the cell centre's region. That is the freeze membership (sec 3 phase
   2), the soft-assign anchor (sec 7) and whole-cell fence ownership, all three
   agreeing -- which is the point of P-F.
3. Movable cells only (i < nl.num_movable). per_node_straddle is still shaped
   (num_physical,), zero on the terminal tail, so it aligns with
   evaluation.npz's node_region.
4. Quadrant area split. The box is cut by at most one vertical line -- the
   lattice column boundary immediately to the right of the column containing
   x, i.e. xl + (ix0+1)*cell_w where ix0 = to_idx(x)'s column, clipped to x+w
   if the box does not reach that far -- and at most one horizontal line
   defined the same way from y. Together they split the box into up to four
   rectangles: lower-left owned by r00's region, lower-right by r10's,
   upper-left by r01's, upper-right by r11's. Out-of-owner area sums exactly
   the rectangles whose own corner's region differs from the owner's. The four
   rectangles' areas sum to w*h in real-number arithmetic; float64 addition is
   not associative, so that sum is pinned bit-exact only for the
   integer-valued geometries this module's own tests use, not claimed for
   arbitrary floats (the same caveat the rel<=1e-12 float-field contract
   already carries for tree_wl/hpwl). Exact for a cell spanning at most two
   lattice cells per axis (every standard cell on a 512-lattice); wider cells
   are approximated and counted in straddle_wide_cells, so the approximation
   is measured, not silent.
5. Pin re-attribution by distinct-region count. per_net_lambda is the only
   crossing measure that is a function of pin->region attribution alone: the
   MST-geometry per_net_crossings depends on coordinates, which re-attribution
   does not change, and per_net_steiner would need the whole Lambda>=4 Steiner
   pass re-run for no extra information. lambda_e - 1 is exactly the per-net
   crossing lower bound realised after fence LG, when no cell straddles.
   per_net_pin_split is signed: re-attribution can also raise a net's lambda.
"""
from dataclasses import dataclass

import numpy as np

from ioplace.netlist import pin_positions

STRADDLE_SCALARS = ("straddle_cells", "straddle_area_fraction",
                    "straddle_pin_split_nets", "straddle_out_area",
                    "straddle_movable_area", "straddle_wide_cells")
STRADDLE_ARRAYS = ("per_node_straddle", "per_net_pin_split")

# (net_id, region_id) are packed into one int64 as net*_REGION_STRIDE + region,
# the same packing evaluator_gpu.py's evaluate() uses for pin_bm via its own
# net_region_key local (a bare 64, independent of this constant by design --
# see GpuEvalContext._distinct_regions_per_net, which imports this constant
# by name instead) -- keep the two equal so the CPU and GPU unique/bincount
# passes are the same arithmetic. Named by symbol, not line number: line
# numbers in evaluator_gpu.py drift across edits.
_REGION_STRIDE = np.int64(64)


@dataclass
class StraddleStats:
    straddle_cells: int
    straddle_area_fraction: float
    straddle_pin_split_nets: int
    straddle_out_area: float
    straddle_movable_area: float
    straddle_wide_cells: int
    per_node_straddle: np.ndarray        # (num_physical,) uint8
    per_net_pin_split: np.ndarray        # (num_nets,) int32, signed


def straddle_geometry(rg, x, y, w, h):
    """(M,) lower-left positions and sizes -> (straddle, owner, out_area, wide).

    straddle: bool, the four corners do not all agree.
    owner:    int64, the region of the cell centre.
    out_area: float64, quadrant area not belonging to the owner.
    wide:     bool, the box spans more than two lattice cells in some axis, so
              out_area is approximate (convention 4).

    See the module docstring's convention 4 for the exact definition of the
    two candidate cut lines (xm, ym below) and which rectangle each of the
    four corners owns.
    """
    xr = x + w
    yt = y + h
    r00 = rg.region_of_points(x, y).astype(np.int64)
    r10 = rg.region_of_points(xr, y).astype(np.int64)
    r01 = rg.region_of_points(x, yt).astype(np.int64)
    r11 = rg.region_of_points(xr, yt).astype(np.int64)
    owner = rg.region_of_points(x + 0.5 * w, y + 0.5 * h).astype(np.int64)
    straddle = (r10 != r00) | (r01 != r00) | (r11 != r00)

    ix0, iy0 = rg.to_idx(x, y)
    ix1, iy1 = rg.to_idx(xr, yt)
    # xm/ym: the single vertical/horizontal cut line each axis gets (the
    # lattice line just past x's column / y's row), clipped to the box's own
    # far edge when the box does not reach it.
    xm = np.minimum(xr, rg.die[0] + (ix0 + 1) * rg.cell_w)
    ym = np.minimum(yt, rg.die[1] + (iy0 + 1) * rg.cell_h)
    lw = np.maximum(xm - x, 0.0)
    rw = np.maximum(xr - xm, 0.0)
    bh = np.maximum(ym - y, 0.0)
    th = np.maximum(yt - ym, 0.0)
    out_area = (lw * bh * (r00 != owner) + rw * bh * (r10 != owner)
                + lw * th * (r01 != owner) + rw * th * (r11 != owner))
    wide = ((ix1 - ix0) > 1) | ((iy1 - iy0) > 1)
    return straddle, owner, out_area, wide


def distinct_regions_per_net(nl, pin_rid):
    """(num_nets,) int64 count of distinct regions each net's pins touch, 0 for
    degree < 2 nets -- the same convention EvalResult.per_net_lambda uses."""
    net_of_pin = np.asarray(nl.pin2net, dtype=np.int64)
    key = np.unique(net_of_pin * _REGION_STRIDE
                    + np.asarray(pin_rid, dtype=np.int64))
    counts = np.bincount(key // _REGION_STRIDE,
                         minlength=nl.num_nets).astype(np.int64)
    return np.where(np.asarray(nl.net_degrees, dtype=np.int64) >= 2, counts, 0)


def straddle_diagnostics(nl, node_x, node_y, rg, *, pin_rid=None):
    """Design v2 sec 7 diagnostics 1-3. `pin_rid` lets a caller that has already
    computed the per-pin region ids (evaluator_ref.evaluate has) pass them in
    rather than paying for a second region_of_points over every pin."""
    if rg.k > int(_REGION_STRIDE):
        raise ValueError("straddle diagnostics pack (net, region) into one "
                         "int64 with stride %d; got k=%d"
                         % (int(_REGION_STRIDE), rg.k))
    m = int(nl.num_movable)
    n_physical = int(nl.num_physical)
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    w = np.asarray(nl.node_size_x, dtype=np.float64)[:m]
    h = np.asarray(nl.node_size_y, dtype=np.float64)[:m]
    straddle, owner, out_area, wide = straddle_geometry(
        rg, node_x[:m], node_y[:m], w, h)

    total_area = float((w * h).sum())
    out_total = float(out_area.sum())
    per_node = np.zeros(n_physical, dtype=np.uint8)
    per_node[:m] = straddle.astype(np.uint8)

    if pin_rid is None:
        px, py = pin_positions(nl, node_x, node_y)
        pin_rid = rg.region_of_points(px, py)
    pin_rid = np.asarray(pin_rid, dtype=np.int64)
    node_of_pin = np.asarray(nl.pin2node, dtype=np.int64)
    owner_full = np.zeros(n_physical, dtype=np.int64)
    owner_full[:m] = owner
    pin_rid_re = np.where(per_node[node_of_pin].astype(bool),
                          owner_full[node_of_pin], pin_rid)

    split = (distinct_regions_per_net(nl, pin_rid)
             - distinct_regions_per_net(nl, pin_rid_re)).astype(np.int32)
    return StraddleStats(
        straddle_cells=int(straddle.sum()),
        straddle_area_fraction=(out_total / total_area) if total_area > 0.0 else 0.0,
        straddle_pin_split_nets=int((split > 0).sum()),
        straddle_out_area=out_total,
        straddle_movable_area=total_area,
        straddle_wide_cells=int(wide.sum()),
        per_node_straddle=per_node,
        per_net_pin_split=split)
