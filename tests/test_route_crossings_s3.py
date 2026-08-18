"""Stage 2 S3 tests (`docs/superpowers/specs/2026-08-13-stage2-innovus-
calibration-plan.md` sec 7 / sec 10 S3 row) for
`ioplace.route_eval.route_crossings`.

All three S3 acceptance criteria live here:
    1. synthetic straight line across k grid regions -> exactly k-1
       (`test_synthetic_straight_line_*`).
    2. the raw >= dw(delta) >= lambda_route-1 >= 0 identity, 1,000 random
       trials (`test_identity_holds_random_1000`).
    3. evaluator_ref's own MST edges fed back in as synthetic WIRE rows
       must bit-exactly reproduce per_net_crossings / per_net_ft /
       boundary_pair_demand (`test_mst_edges_as_fake_wire_*`) -- "最強的口徑
       對齊證明,必做".

Plus delta-parametrization behaviour and sec 7.3's coordinate mapping (not
hardcoded to 1.0). No torch/DREAMPlace needed -- everything here is plain
numpy, same as evaluator_ref.py/region_grid.py/regions.py.
"""
import json

import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate, net_mst_edges
from ioplace.netlist import pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from ioplace.route_eval.route_crossings import (
    CoordMap,
    align_net_indices,
    evaluate_route,
)
from ioplace.route_eval.segments import KIND_WIRE, Segments

from tests.test_evaluator_gpu import _random_case
from tests.test_netlist import make_tiny_netlist

DIE = (0.0, 0.0, 100.0, 100.0)


# ---------------------------------------------------------------------------
# test helper: build a Segments object directly (no npz round-trip needed --
# route_crossings only reads the WIRE-kind columns + net_names/net_has_wire)
# ---------------------------------------------------------------------------

def _segments_from_rows(rows, net_names, coord_dtype=np.int64):
    """rows: list of (net_idx, x0, y0, x1, y1) WIRE rows, DBU-like
    coordinates (int64 by default, matching the real dump_segments.py
    contract). net_idx indexes into `net_names` and becomes the Segments
    net_id (odb order) -- tests that want to exercise sec 7.2's name-based
    realignment instead pass a `net_names` order that differs from the
    caller's placedb-index order (see test_align_net_indices_*).

    `coord_dtype=float64` is used by the acceptance-3 (MST-edges-as-fake-
    wire) tests below: those need the *exact* float pin positions
    evaluate_ref.evaluate() used, not an int64-DBU round-trip, since
    rounding to the nearest integer DBU can nudge a coordinate across a
    lattice-cell boundary and change which region it lands in -- a
    perfectly real DBU-quantization effect for actual routed DEF data, but
    not what acceptance criterion 3 ("逐位元重現") is testing.
    """
    n = len(rows)
    return Segments(
        seg_net_id=np.array([r[0] for r in rows], dtype=np.int32),
        seg_kind=np.full(n, KIND_WIRE, dtype=np.uint8),
        seg_layer=np.zeros(n, dtype=np.int16),
        seg_x0=np.array([r[1] for r in rows], dtype=coord_dtype),
        seg_y0=np.array([r[2] for r in rows], dtype=coord_dtype),
        seg_x1=np.array([r[3] for r in rows], dtype=coord_dtype),
        seg_y1=np.array([r[4] for r in rows], dtype=coord_dtype),
        seg_width=np.zeros(n, dtype=np.int64),
        seg_via_id=np.full(n, -1, dtype=np.int32),
        net_names=np.array(net_names),
        net_has_wire=np.ones(len(net_names), dtype=bool),
        layer_names=np.array(["metal1"]),
        via_names=np.array([], dtype="<U1"),
        meta={},
    )


# ---------------------------------------------------------------------------
# acceptance 1 -- synthetic straight line across k regions -> exactly k-1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_synthetic_straight_line_horizontal_exact_k_minus_1(k):
    rg = RegionGrid(make_grid_regions(DIE, k, 1, lattice=120))
    segs = _segments_from_rows([(0, 0, 50, 100, 50)], ["n0"])
    res = evaluate_route(segs, rg, ["n0"], delta=2)
    assert res.route_cross_raw[0] == k - 1
    assert res.route_cross_dw[0] == k - 1
    assert res.lambda_route[0] == k


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_synthetic_straight_line_vertical_exact_k_minus_1(k):
    rg = RegionGrid(make_grid_regions(DIE, 1, k, lattice=120))
    segs = _segments_from_rows([(0, 50, 0, 50, 100)], ["n0"])
    res = evaluate_route(segs, rg, ["n0"], delta=2)
    assert res.route_cross_raw[0] == k - 1
    assert res.route_cross_dw[0] == k - 1
    assert res.lambda_route[0] == k


@pytest.mark.parametrize("delta", [0, 1, 2, 4])
def test_synthetic_straight_line_delta_invariant_when_runs_wide(delta):
    # region cells are 24 lattice cells wide (120/5) -- much wider than any
    # delta in the sec 7.4 scan set, so no run-length filtering should kick
    # in at all: raw == dw(delta) == k-1 for every delta tested.
    k = 5
    rg = RegionGrid(make_grid_regions(DIE, k, 1, lattice=120))
    segs = _segments_from_rows([(0, 0, 50, 100, 50)], ["n0"])
    res = evaluate_route(segs, rg, ["n0"], delta=delta)
    assert res.route_cross_raw[0] == k - 1
    assert res.route_cross_dw[0] == k - 1
    assert res.lambda_route[0] == k


# ---------------------------------------------------------------------------
# acceptance 2 -- raw >= dw(delta) >= lambda_route-1 >= 0, 1,000 random trials
# ---------------------------------------------------------------------------

def _random_walk_segments(rng, die, max_segs=6):
    """A connected random Manhattan walk: each leg is either a horizontal or
    vertical move to a new random coordinate, always starting where the
    previous leg ended -- the same topology real routed-net wire has
    (segments meeting at shared endpoints), which is what keeps the
    raw>=dw(delta)>=lambda_route-1>=0 identity provable (see
    route_crossings.py's module docstring: "Lambda_route and the identity").
    """
    xl, yl, xh, yh = die
    n_segs = int(rng.integers(1, max_segs + 1))
    x, y = rng.uniform(xl, xh), rng.uniform(yl, yh)
    rows = []
    for _ in range(n_segs):
        if bool(rng.integers(0, 2)):
            nx_ = rng.uniform(xl, xh)
            rows.append((x, y, nx_, y))
            x = nx_
        else:
            ny_ = rng.uniform(yl, yh)
            rows.append((x, y, x, ny_))
            y = ny_
    return rows


def test_identity_holds_random_1000():
    rng = np.random.default_rng(20260816)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=40))
    violations = []
    for trial in range(1000):
        legs = _random_walk_segments(rng, DIE)
        seg_rows = [(0, int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
                    for (x0, y0, x1, y1) in legs]
        segs = _segments_from_rows(seg_rows, ["n0"])
        delta = int(rng.choice([0, 1, 2, 4]))
        res = evaluate_route(segs, rg, ["n0"], delta=delta)
        raw, dw = int(res.route_cross_raw[0]), int(res.route_cross_dw[0])
        lam = int(res.lambda_route[0])
        if not (raw >= dw >= lam - 1 >= 0):
            violations.append((trial, delta, raw, dw, lam, seg_rows))
    assert not violations, f"{len(violations)} identity violations, first 5: {violations[:5]}"


# ---------------------------------------------------------------------------
# acceptance 3 -- MST edges as fake wire reproduce evaluator_ref bit-exactly
# ---------------------------------------------------------------------------

def _mst_edges_as_fake_wire_rows(nl, node_x, node_y, rg):
    """Build (net, x0,y0,x1,y1) WIRE rows from evaluator_ref's own MST
    edges, L-shaped exactly the way `edge_regions_and_crossings` walks them
    (horizontal leg first, then vertical) -- so route_crossings, walking
    this "fake wire" with the *same* `_walk_segment`, must retrace the
    identical lattice cells `evaluate()` did. Also returns each net's pin
    region set (for route_ft), computed the same way `evaluate()`'s local
    `pin_regions` is.
    """
    px, py = pin_positions(nl, node_x, node_y)
    start = nl.flat_net2pin_start
    rows, pin_regions = [], {}
    for net in range(nl.num_nets):
        s, e = start[net], start[net + 1]
        pin_idx = nl.flat_net2pin[s:e]
        nx_, ny_ = px[pin_idx], py[pin_idx]
        pin_regions[net] = set(rg.region_of_points(nx_, ny_).tolist())
        if len(nx_) <= 1:
            continue
        for (a, b) in net_mst_edges(nx_, ny_):
            rows.append((net, nx_[a], ny_[a], nx_[b], ny_[a]))   # horizontal leg
            rows.append((net, nx_[b], ny_[a], nx_[b], ny_[b]))   # vertical leg
    return rows, pin_regions


def _assert_route_matches_evaluator(nl, node_x, node_y, rg):
    ref = evaluate(nl, node_x, node_y, rg)
    seg_rows, pin_regions = _mst_edges_as_fake_wire_rows(nl, node_x, node_y, rg)
    net_names = [f"n{i}" for i in range(nl.num_nets)]
    segs = _segments_from_rows(seg_rows, net_names, coord_dtype=np.float64)
    res = evaluate_route(segs, rg, net_names, pin_regions=pin_regions, delta=2)
    assert res.route_cross_raw.tolist() == ref.per_net_crossings.tolist()
    assert res.route_ft.tolist() == ref.per_net_ft.tolist()
    assert res.route_pair_demand == ref.boundary_pair_demand
    return res, ref


def test_mst_edges_as_fake_wire_reproduce_tiny_netlist():
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    nl = make_tiny_netlist()
    _assert_route_matches_evaluator(nl, nl.node_x, nl.node_y, rg)


def test_mst_edges_as_fake_wire_reproduce_pure_feedthrough():
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    nl = make_tiny_netlist()
    node_x = np.array([10., 90., 50., 90.])
    node_y = np.array([10., 90., 40., 90.])
    res, ref = _assert_route_matches_evaluator(nl, node_x, node_y, rg)
    assert ref.ft_count == 1  # sanity: this fixture actually exercises FT>0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_mst_edges_as_fake_wire_reproduce_random_netlists(seed):
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(np.random.default_rng(seed))
    _assert_route_matches_evaluator(nl, nl.node_x, nl.node_y, rg)


# ---------------------------------------------------------------------------
# delta parametrization behaviour (sec 7.4)
# ---------------------------------------------------------------------------

def test_delta_parametrization_zero_equals_raw_and_filters_narrow_runs():
    # 8 regions, 1 lattice cell per region -- every crossing run is exactly
    # 1 cell wide, an adversarial case chosen to show what
    # route_cross_dw(delta) actually does: delta in {0,1} keeps every
    # transition (no run is ever shorter than 1 cell), delta=2 collapses
    # every 1-cell run into its neighbour -- discounting all of them as
    # boundary noise, all the way down to a single surviving run.
    rg = RegionGrid(make_grid_regions(DIE, 8, 1, lattice=8))
    segs = _segments_from_rows([(0, 0, 50, 100, 50)], ["n0"])
    for delta in (0, 1):
        res = evaluate_route(segs, rg, ["n0"], delta=delta)
        assert res.route_cross_dw[0] == res.route_cross_raw[0] == 7

    res2 = evaluate_route(segs, rg, ["n0"], delta=2)
    assert res2.route_cross_raw[0] == 7
    assert res2.route_cross_dw[0] == 0
    assert res2.route_cross_dw[0] < res2.route_cross_raw[0]
    assert res2.lambda_route[0] == 1
    # identity still holds in this heavily-filtered corner case
    assert res2.route_cross_raw[0] >= res2.route_cross_dw[0] >= res2.lambda_route[0] - 1 >= 0


# ---------------------------------------------------------------------------
# sec 7.3 -- coordinate mapping must not be hardcoded to 1.0
# ---------------------------------------------------------------------------

def test_coord_map_shift_scale_applied_not_hardcoded_identity():
    rg = RegionGrid(make_grid_regions(DIE, 4, 1, lattice=40))
    shift = (1000.0, 2000.0)
    scale = 0.5

    def to_def(xi, yi):  # inverse of CoordMap.to_internal
        return xi / scale + shift[0], yi / scale + shift[1]

    x0d, y0d = to_def(0.0, 50.0)
    x1d, y1d = to_def(100.0, 50.0)
    segs = _segments_from_rows(
        [(0, int(round(x0d)), int(round(y0d)), int(round(x1d)), int(round(y1d)))], ["n0"])
    coord_map = CoordMap(shift_factor=shift, scale_factor=scale)
    res = evaluate_route(segs, rg, ["n0"], coord_map=coord_map, delta=2)
    assert res.route_cross_raw[0] == 3    # 4 regions -> 3 crossings
    assert res.lambda_route[0] == 4


def test_coord_map_identity_is_default_when_omitted():
    rg = RegionGrid(make_grid_regions(DIE, 4, 1, lattice=40))
    segs = _segments_from_rows([(0, 0, 50, 100, 50)], ["n0"])
    res_default = evaluate_route(segs, rg, ["n0"], delta=2)
    res_identity = evaluate_route(segs, rg, ["n0"], coord_map=CoordMap.identity(), delta=2)
    assert res_default.route_cross_raw.tolist() == res_identity.route_cross_raw.tolist()


def test_coord_map_from_json_round_trip(tmp_path):
    p = tmp_path / "coord.json"
    p.write_text(json.dumps({
        "shift_factor": [123.0, 456.0], "scale_factor": 0.25,
        "def_units_per_micron": 2000, "xl": 0.0, "yl": 0.0, "xh": 100.0, "yh": 100.0,
    }))
    cm = CoordMap.from_json(p)
    assert cm.shift_factor == (123.0, 456.0)
    assert cm.scale_factor == 0.25
    x, y = cm.to_internal(123.0 + 4 / 0.25, 456.0 + 8 / 0.25)
    assert float(x) == pytest.approx(4.0)
    assert float(y) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# sec 7.2 -- key alignment
# ---------------------------------------------------------------------------

def test_align_net_indices_matches_by_name_not_position():
    # segments net order ("odb order") deliberately differs from the
    # placedb net_index order passed as net_names -- alignment must go by
    # name, never by presumed positional equality (sec 7.2 point 2).
    segs = _segments_from_rows(
        [(0, 0, 0, 10, 0), (1, 0, 0, 10, 0)], ["netB", "netA"])
    placedb_order = ["netA", "netB", "netC"]
    seg_id_of, unmatched = align_net_indices(segs, placedb_order)
    assert seg_id_of.tolist() == [1, 0, -1]
    assert unmatched == [2]


def test_evaluate_route_unmatched_net_gets_zero_and_is_reported():
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    segs = _segments_from_rows([(0, 0, 0, 90, 0)], ["netA"])
    res = evaluate_route(segs, rg, ["netA", "netB"], delta=2)
    assert res.unmatched_net_indices == [1]
    assert res.route_cross_raw[1] == 0
    assert res.route_wl[1] == 0


# ---------------------------------------------------------------------------
# route_wl (sec 7.4: WIRE-only, matches Segments.wire_length()'s convention)
# ---------------------------------------------------------------------------

def test_route_wl_matches_segment_manhattan_length():
    segs = _segments_from_rows([(0, 0, 0, 100, 0), (0, 100, 0, 100, 40)], ["n0"])
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    res = evaluate_route(segs, rg, ["n0"], delta=2)
    assert res.route_wl[0] == 140
    assert res.total_route_wl == 140
