import dataclasses

import numpy as np
import pytest

from ioplace.region_grid import RegionGrid
from ioplace.region_graph import region_graph
from ioplace.region_segments import (ORIENT_H, ORIENT_V, enumerate_segments,
                                     segments_digest)
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions

DIE = (0., 0., 4., 4.)


def _rs(*specs):
    """A 4x4-lattice RegionSet over DIE from (name, rects) pairs."""
    return RegionSet(die=DIE, lattice=4,
                     regions=[RegionSpec(n, np.asarray(r, dtype=np.float64))
                              for n, r in specs])


def l_shape():
    """Region 0 is a genuine L (left column + bottom row); region 1 is the
    3x3 block it wraps. The pair (0,1) is therefore split into two disjoint
    segments -- one vertical, one horizontal."""
    return _rs(("L", [[0., 0., 1., 4.], [1., 0., 4., 1.]]),
               ("B", [[1., 1., 4., 4.]]))


def plug():
    """Region 2 plugs the middle of the 0|1 boundary, so the pair (0,1) is
    split into two disjoint segments *on the same boundary line* -- the case
    that actually exercises the run-length encoding."""
    return _rs(("A", [[0., 0., 2., 4.]]),
               ("B", [[2., 0., 4., 1.], [2., 3., 4., 4.]]),
               ("C", [[2., 1., 4., 3.]]))


def notch():
    """A single-bin notch: region 0 pokes one lattice cell into region 1."""
    return _rs(("A", [[0., 0., 2., 4.], [2., 1., 3., 2.]]),
               ("B", [[2., 0., 4., 1.], [3., 1., 4., 2.], [2., 2., 4., 4.]]))


def single_region():
    """k=1: the whole die is one region. No boundary can exist at all --
    the degenerate case where the empty-table invariants matter (dtypes,
    an all-zero CSR, a digest that stays well-defined on empty arrays)."""
    return _rs(("ONLY", [[0., 0., 4., 4.]]))


def enclosed():
    """Region 1 is a 2x2 block strictly inside region 0's 4x4 lattice,
    touching no die edge on any side -- unlike l_shape/plug/notch/
    make_grid_regions, which all touch a die edge somewhere. This is the
    fixture that would expose a bug in die-edge exclusion, since every one
    of region 1's four boundaries here is a real region/region boundary,
    not a region/outside-the-die edge."""
    return _rs(("A", [[0., 0., 4., 1.], [0., 3., 4., 4.],
                       [0., 1., 1., 3.], [3., 1., 4., 3.]]),
               ("B", [[1., 1., 3., 3.]]))


def _as_tuples(table):
    return [(int(table.orient[s]), int(table.line[s]), int(table.lo[s]),
             int(table.hi[s]), int(table.pair_a[s]), int(table.pair_b[s]))
            for s in range(table.num_segments)]


def test_grid_k4_has_exactly_four_segments():
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=4))
    table = enumerate_segments(rg)
    assert table.num_segments == 4
    # V first (line, lo), then H (line, lo). make_grid_regions numbers
    # regions row-major in y: 0=SW, 1=SE, 2=NW, 3=NE.
    assert _as_tuples(table) == [
        (ORIENT_V, 1, 0, 2, 0, 1),
        (ORIENT_V, 1, 2, 4, 2, 3),
        (ORIENT_H, 1, 0, 2, 0, 2),
        (ORIENT_H, 1, 2, 4, 1, 3),
    ]
    assert np.array_equal(table.length_units, [2, 2, 2, 2])
    # cell_w == cell_h == 1.0 here, so physical length equals the unit count.
    np.testing.assert_allclose(table.length, [2., 2., 2., 2.])


def test_l_shaped_region_splits_one_pair_into_two_segments():
    table = enumerate_segments(RegionGrid(l_shape()))
    assert _as_tuples(table) == [(ORIENT_V, 0, 1, 4, 0, 1),
                                 (ORIENT_H, 0, 1, 4, 0, 1)]
    assert table.num_segments == 2


def test_a_plug_splits_one_pair_on_the_same_boundary_line():
    table = enumerate_segments(RegionGrid(plug()))
    assert _as_tuples(table) == [
        (ORIENT_V, 1, 0, 1, 0, 1),
        (ORIENT_V, 1, 1, 3, 0, 2),
        (ORIENT_V, 1, 3, 4, 0, 1),
        (ORIENT_H, 0, 2, 4, 1, 2),
        (ORIENT_H, 2, 2, 4, 1, 2),
    ]
    zero_one = [s for s in range(table.num_segments)
                if (table.pair_a[s], table.pair_b[s]) == (0, 1)]
    assert len(zero_one) == 2
    assert {int(table.line[s]) for s in zero_one} == {1}   # same boundary line
    assert sorted((int(table.lo[s]), int(table.hi[s])) for s in zero_one) == \
        [(0, 1), (3, 4)]


def test_a_one_bin_notch_produces_unit_length_segments():
    table = enumerate_segments(RegionGrid(notch()))
    assert _as_tuples(table) == [
        (ORIENT_V, 1, 0, 1, 0, 1),
        (ORIENT_V, 1, 2, 4, 0, 1),
        (ORIENT_V, 2, 1, 2, 0, 1),
        (ORIENT_H, 0, 2, 3, 0, 1),
        (ORIENT_H, 1, 2, 3, 0, 1),
    ]
    # Four, not three: the notch's own three edges (its right-hand V edge on
    # line 2, and its top/bottom H edges) are unit-length, but splitting the
    # main A|B boundary at line 1 around the notch also leaves an incidental
    # fourth unit-length fragment, rows [0,1) -- easy to miss if you count
    # "segments touching the notch cell" instead of reading length_units off
    # the tuple list above (which pins this exactly and is itself checked
    # against region_graph's ell aggregate, the raster round-trip and the CSR
    # elsewhere in this file).
    assert int((table.length_units == 1).sum()) == 4


def test_a_single_region_grid_has_zero_segments():
    """k=1: no boundary edge exists anywhere, so every derived structure must
    degenerate cleanly rather than raising or producing garbage shapes."""
    rg = RegionGrid(single_region())
    table = enumerate_segments(rg)
    ny, nx = rg.grid.shape

    assert table.num_segments == 0
    assert table.orient.shape == (0,) and table.orient.dtype == np.int8
    assert table.line.shape == (0,) and table.line.dtype == np.int32
    assert table.lo.shape == (0,) and table.lo.dtype == np.int32
    assert table.hi.shape == (0,) and table.hi.dtype == np.int32
    assert table.pair_a.shape == (0,) and table.pair_a.dtype == np.int16
    assert table.pair_b.shape == (0,) and table.pair_b.dtype == np.int16
    assert table.length_units.shape == (0,) and table.length_units.dtype == np.int32
    assert table.length.shape == (0,) and table.length.dtype == np.float64
    assert table.box.shape == (0, 4) and table.box.dtype == np.float64

    assert table.edge_seg_v.shape == (ny, nx - 1)
    assert table.edge_seg_h.shape == (ny - 1, nx)
    assert (table.edge_seg_v == -1).all()
    assert (table.edge_seg_h == -1).all()

    assert table.row_ptr.shape == (ny + 1,)
    assert table.col_ptr.shape == (nx + 1,)
    np.testing.assert_array_equal(table.row_ptr, np.zeros(ny + 1, dtype=np.int64))
    np.testing.assert_array_equal(table.col_ptr, np.zeros(nx + 1, dtype=np.int64))
    assert table.row_col.shape == (0,) and table.row_seg.shape == (0,)
    assert table.col_row.shape == (0,) and table.col_seg.shape == (0,)

    digest = segments_digest(table)
    assert isinstance(digest, str) and len(digest) == 64


def test_a_fully_enclosed_region_pins_all_four_boundaries():
    """Every other fixture touches a die edge somewhere; this one does not,
    so it's the only fixture where a die-edge-exclusion bug (e.g. treating
    the die boundary itself as a region boundary, or off-by-one at the last
    row/column) would actually show up as a wrong tuple here."""
    table = enumerate_segments(RegionGrid(enclosed()))
    assert _as_tuples(table) == [
        (ORIENT_V, 0, 1, 3, 0, 1),
        (ORIENT_V, 2, 1, 3, 0, 1),
        (ORIENT_H, 0, 1, 3, 0, 1),
        (ORIENT_H, 2, 1, 3, 0, 1),
    ]
    assert table.num_segments == 4
    np.testing.assert_array_equal(table.length_units, [2, 2, 2, 2])


@pytest.mark.parametrize("factory", [
    lambda: make_grid_regions(DIE, 2, 2, lattice=4),
    l_shape, plug, notch,
    lambda: make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20),
])
def test_segment_lengths_sum_to_the_pair_aggregate_ell(factory):
    """spec sec 5: `region_graph`'s ell[a,b] is the sum of a pair's segment
    lengths and remains the pair-level aggregate."""
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    _adj, _D, ell = region_graph(rg)
    totals = np.zeros((rg.k, rg.k), dtype=np.int64)
    for s in range(table.num_segments):
        a, b = int(table.pair_a[s]), int(table.pair_b[s])
        totals[a, b] += int(table.length_units[s])
        totals[b, a] += int(table.length_units[s])
    np.testing.assert_array_equal(totals, ell)
    assert (table.pair_a < table.pair_b).all()


@pytest.mark.parametrize("factory", [l_shape, plug, notch,
                                     lambda: make_grid_regions(DIE, 2, 2, lattice=4)])
def test_raster_round_trips_against_brute_force(factory):
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    grid = rg.grid.astype(np.int64)
    ny, nx = grid.shape
    for i in range(ny):
        for j in range(nx - 1):
            sid = int(table.edge_seg_v[i, j])
            if grid[i, j] == grid[i, j + 1]:
                assert sid == -1
                continue
            assert sid >= 0 and table.orient[sid] == ORIENT_V
            assert int(table.line[sid]) == j
            assert int(table.lo[sid]) <= i < int(table.hi[sid])
            assert (int(table.pair_a[sid]), int(table.pair_b[sid])) == \
                (min(grid[i, j], grid[i, j + 1]), max(grid[i, j], grid[i, j + 1]))
    for i in range(ny - 1):
        for j in range(nx):
            sid = int(table.edge_seg_h[i, j])
            if grid[i, j] == grid[i + 1, j]:
                assert sid == -1
                continue
            assert sid >= 0 and table.orient[sid] == ORIENT_H
            assert int(table.line[sid]) == i
            assert int(table.lo[sid]) <= j < int(table.hi[sid])


def test_csr_matches_the_raster_and_is_sorted():
    rg = RegionGrid(plug())
    table = enumerate_segments(rg)
    ny, nx = rg.grid.shape
    for i in range(ny):
        s0, s1 = int(table.row_ptr[i]), int(table.row_ptr[i + 1])
        cols = table.row_col[s0:s1]
        assert list(cols) == sorted(set(cols))
        expected = np.nonzero(table.edge_seg_v[i] >= 0)[0]
        np.testing.assert_array_equal(cols, expected)
        np.testing.assert_array_equal(table.row_seg[s0:s1],
                                      table.edge_seg_v[i][expected])
    for j in range(nx):
        s0, s1 = int(table.col_ptr[j]), int(table.col_ptr[j + 1])
        rows = table.col_row[s0:s1]
        assert list(rows) == sorted(set(rows))
        expected = np.nonzero(table.edge_seg_h[:, j] >= 0)[0]
        np.testing.assert_array_equal(rows, expected)
        np.testing.assert_array_equal(table.col_seg[s0:s1],
                                      table.edge_seg_h[:, j][expected])


def test_enumeration_is_deterministic_and_digested():
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    a, b = enumerate_segments(rg), enumerate_segments(rg)
    assert segments_digest(a) == segments_digest(b)
    other = enumerate_segments(RegionGrid(plug()))
    assert segments_digest(other) != segments_digest(a)


def test_digest_is_sensitive_to_every_hashed_field():
    """segments_digest is written into capacity.npz/evaluation.npz as a
    provenance key so a capacity file can never be silently paired with a
    different geometry; that guarantee only holds if it actually moves when
    *any* field it reads changes, not just for the one plug-vs-grid instance
    above. Mutate each field segments_digest hashes, one at a time, and
    check the digest moves."""
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    base = enumerate_segments(rg)
    assert base.num_segments > 0  # so every array field below is non-empty
    baseline = segments_digest(base)

    for field in ("orient", "line", "lo", "hi", "pair_a", "pair_b"):
        arr = getattr(base, field).copy()
        arr[0] = arr[0] + 1
        mutated = dataclasses.replace(base, **{field: arr})
        assert segments_digest(mutated) != baseline, \
            f"digest did not move when '{field}' changed"

    assert segments_digest(dataclasses.replace(base, k=base.k + 1)) != baseline
    assert segments_digest(dataclasses.replace(base, lattice=base.lattice + 1)) != baseline
    assert segments_digest(dataclasses.replace(
        base, die=(base.die[0] + 1.0,) + base.die[1:])) != baseline


def test_a_512_lattice_stays_cheap():
    """spec sec 5: O(L^2) ~ 2.6e5, pure numpy. Guard against a python
    per-cell loop creeping back in."""
    import time
    rg = RegionGrid(make_grid_regions((0., 0., 1000., 1000.), 4, 4, lattice=512))
    started = time.perf_counter()
    table = enumerate_segments(rg)
    assert time.perf_counter() - started < 2.0
    # 4x4 regions: 3 internal vertical lines x 4 row bands = 12 V segments,
    # symmetrically 12 H segments.
    assert table.num_segments == 24
    assert table.edge_seg_v.shape == (512, 511)
    assert table.edge_seg_h.shape == (511, 512)


from ioplace.region_segments import (Candidates, edge_segment_ids,
                                     segment_ids_on_col, segment_ids_on_row,
                                     segment_utilisation, select_candidates,
                                     CAPACITY_SCALARS)


def _brute_row(table, row, lo, hi):
    """Every vertical unit edge a horizontal leg in `row` from column `lo` to
    column `hi` (inclusive) crosses, straight off the raster."""
    a, b = min(lo, hi), max(lo, hi)
    ids = table.edge_seg_v[row, a:b]
    return ids[ids >= 0]


def _brute_col(table, col, lo, hi):
    a, b = min(lo, hi), max(lo, hi)
    ids = table.edge_seg_h[a:b, col]
    return ids[ids >= 0]


@pytest.mark.parametrize("factory", [l_shape, plug, notch])
def test_csr_leg_lookup_matches_the_raster_everywhere(factory):
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    ny, nx = rg.grid.shape
    for row in range(ny):
        for lo in range(nx):
            for hi in range(nx):
                np.testing.assert_array_equal(
                    segment_ids_on_row(table, row, lo, hi),
                    _brute_row(table, row, lo, hi))
    for col in range(nx):
        for lo in range(ny):
            for hi in range(ny):
                np.testing.assert_array_equal(
                    segment_ids_on_col(table, col, lo, hi),
                    _brute_col(table, col, lo, hi))


def test_leg_lookup_length_equals_the_prefix_sum_crossing_count():
    """The Ph/Pv identity sec 5 calls a 'free assertion on slice lengths'."""
    rg = RegionGrid(plug())
    table = enumerate_segments(rg)
    grid = rg.grid
    ny, nx = grid.shape
    hdiff = (grid[:, :-1] != grid[:, 1:]).astype(np.int64)
    ph = np.zeros((ny, nx), dtype=np.int64)
    ph[:, 1:] = np.cumsum(hdiff, axis=1)
    for row in range(ny):
        for lo in range(nx):
            for hi in range(lo, nx):
                assert len(segment_ids_on_row(table, row, lo, hi)) == \
                    int(ph[row, hi] - ph[row, lo])


def test_edge_segment_ids_walks_an_l_route_in_coordinates():
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=4))
    table = enumerate_segments(rg)
    # (10,10) is in region 0 (SW); (90,90) is in region 3 (NE).
    ids = edge_segment_ids(rg, table, 10., 10., 90., 90.)
    # The L-route goes horizontally at y=10 (crossing the 0|1 boundary) then
    # vertically at x=90 (crossing the 1|3 boundary).
    assert len(ids) == 2
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
    assert pairs == [(0, 1), (1, 3)]


def test_segment_utilisation_reports_the_capacity_scalars():
    demand = np.array([0, 5, 12, 3, 1], dtype=np.int64)
    capacity = np.array([10., 10., 10., 0., 0.], dtype=np.float64)
    util, scalars = segment_utilisation(demand, capacity)
    np.testing.assert_allclose(util[:3], [0., 0.5, 1.2])
    assert util[3] == np.inf and util[4] == np.inf
    assert tuple(scalars) == CAPACITY_SCALARS
    assert scalars["num_segments"] == 5
    assert scalars["segment_demand_total"] == 21
    # 12 > 10, and both zero-capacity segments carry demand
    assert scalars["num_over_capacity"] == 3
    assert scalars["num_zero_capacity_segments"] == 2
    assert scalars["zero_capacity_demand"] == 4
    assert scalars["max_util"] == pytest.approx(1.2)
    assert scalars["p99_util"] == pytest.approx(np.percentile([0., 0.5, 1.2], 99))


def test_segment_utilisation_never_divides_by_a_zero_capacity():
    """Unit rule: zero-capacity segments stay blocked with a finite penalty and
    no epsilon is substituted -- so no NaN, no inf/inf, no 1e-9 anywhere."""
    util, scalars = segment_utilisation(np.zeros(3, dtype=np.int64),
                                        np.zeros(3, dtype=np.float64))
    assert np.array_equal(util, np.zeros(3))
    assert scalars["num_over_capacity"] == 0
    assert scalars["max_util"] == 0. and scalars["p99_util"] == 0.


def test_select_candidates_caps_groups_and_segments():
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    table = enumerate_segments(rg)
    # Six distinct (u, v, pair(seg)) groups for net 0, one segment each,
    # with descending counts; m_pairs=4 must keep the four heaviest.
    segs, seen = [], set()
    for s in range(table.num_segments):
        key = (int(table.pair_a[s]), int(table.pair_b[s]))
        if key not in seen:
            seen.add(key)
            segs.append(s)
        if len(segs) == 6:
            break
    net = np.zeros(6, dtype=np.int64)
    u = np.array([int(table.pair_a[s]) for s in segs], dtype=np.int64)
    v = np.array([int(table.pair_b[s]) for s in segs], dtype=np.int64)
    seg = np.asarray(segs, dtype=np.int64)
    count = np.array([60, 50, 40, 30, 20, 10], dtype=np.int64)
    cand = select_candidates(net, u, v, seg, count, table, m_pairs=4, m_seg=2)
    assert cand.n_groups == 4
    assert sorted(cand.count.tolist()) == [30, 40, 50, 60]


def test_select_candidates_keeps_two_alternatives_on_one_boundary_pair():
    """Three segments on the same boundary pair for the same net: one group,
    the two heaviest kept (m_seg=2), ties broken by segment id."""
    table = enumerate_segments(RegionGrid(notch()))
    same = [s for s in range(table.num_segments)
            if (int(table.pair_a[s]), int(table.pair_b[s])) == (0, 1)
            and int(table.orient[s]) == ORIENT_V]
    assert len(same) == 3
    net = np.zeros(3, dtype=np.int64)
    u = np.zeros(3, dtype=np.int64)
    v = np.ones(3, dtype=np.int64)
    seg = np.asarray(same, dtype=np.int64)
    count = np.array([1, 1, 5], dtype=np.int64)
    cand = select_candidates(net, u, v, seg, count, table)
    assert cand.n_groups == 1
    # the 5-count one, plus the lower-id of the two tied 1-counts
    assert sorted(cand.seg.tolist()) == sorted([same[2], min(same[0], same[1])])
    assert np.array_equal(cand.group, np.zeros(2, dtype=np.int64))


def test_select_candidates_is_order_independent():
    table = enumerate_segments(RegionGrid(plug()))
    net = np.array([0, 0, 1, 1], dtype=np.int64)
    u = np.array([0, 0, 1, 0], dtype=np.int64)
    v = np.array([1, 2, 2, 2], dtype=np.int64)
    seg = np.array([0, 1, 3, 1], dtype=np.int64)
    count = np.array([3, 7, 2, 9], dtype=np.int64)
    a = select_candidates(net, u, v, seg, count, table)
    order = np.array([2, 0, 3, 1])
    b = select_candidates(net[order], u[order], v[order], seg[order],
                          count[order], table)
    for name in ("net", "u", "v", "seg", "group", "count"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.n_groups == b.n_groups


def test_remap_nets_drops_inactive_nets_and_recompacts_groups():
    table = enumerate_segments(RegionGrid(plug()))
    cand = select_candidates(np.array([0, 5, 5], dtype=np.int64),
                             np.array([0, 0, 1], dtype=np.int64),
                             np.array([1, 2, 2], dtype=np.int64),
                             np.array([0, 1, 3], dtype=np.int64),
                             np.array([1, 1, 1], dtype=np.int64), table)
    mapping = np.full(6, -1, dtype=np.int64)
    mapping[5] = 0                       # only global net 5 is active
    remapped = cand.remap_nets(mapping)
    assert np.array_equal(remapped.net, np.zeros(2, dtype=np.int64))
    assert remapped.n_groups == 2
    assert np.array_equal(remapped.group, np.array([0, 1]))
    assert select_candidates(np.zeros(0, dtype=np.int64), *(
        np.zeros(0, dtype=np.int64) for _ in range(4)), table).is_empty()


# ---------------------------------------------------------------------------
# Fix round 1 (review of commit 4455233): edge_segment_ids edge cases, a
# select_candidates group-tie-break regression test, and a parity check
# against evaluator_ref's independently-implemented crossing walk.
# ---------------------------------------------------------------------------

from ioplace import evaluator_ref


def _brute_edge_segments(rg, table, x0, y0, x1, y1):
    """Brute-force cross-check for edge_segment_ids: the same L-route, but
    built from the raster-backed _brute_row/_brute_col helpers instead of the
    CSR, so a transposition bug in edge_segment_ids' axis composition (row vs
    col, x vs y) would show up here even though both paths ultimately read
    the same table."""
    ax, ay = rg.to_idx(np.asarray([x0], dtype=np.float64),
                       np.asarray([y0], dtype=np.float64))
    bx, by = rg.to_idx(np.asarray([x1], dtype=np.float64),
                       np.asarray([y1], dtype=np.float64))
    horizontal = _brute_row(table, int(ay[0]), int(ax[0]), int(bx[0]))
    vertical = _brute_col(table, int(bx[0]), int(ay[0]), int(by[0]))
    return np.concatenate([horizontal, vertical])


def test_edge_segment_ids_handles_degenerate_and_axis_only_legs():
    """The one existing edge_segment_ids test is a two-crossing diagonal
    happy path; none of same-region, coincident, axis-only, or an elbow
    landing exactly on a multi-region junction were covered, so an
    axis-transposition regression here would have gone uncaught."""
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=4))
    table = enumerate_segments(rg)

    # Same-region edge: both endpoints deep inside region 0 (SW). No boundary
    # is crossed at all.
    ids = edge_segment_ids(rg, table, 10., 10., 20., 20.)
    assert ids.shape == (0,)
    np.testing.assert_array_equal(ids, _brute_edge_segments(rg, table, 10., 10., 20., 20.))

    # Coincident endpoints: a zero-length "edge" degenerates to two
    # zero-length legs.
    ids = edge_segment_ids(rg, table, 10., 10., 10., 10.)
    assert ids.shape == (0,)
    np.testing.assert_array_equal(ids, _brute_edge_segments(rg, table, 10., 10., 10., 10.))

    # Pure horizontal leg (y0 == y1): the vertical leg is zero-length, so only
    # the row lookup can contribute. Crosses the single 0|1 boundary.
    ids = edge_segment_ids(rg, table, 10., 10., 90., 10.)
    np.testing.assert_array_equal(ids, _brute_edge_segments(rg, table, 10., 10., 90., 10.))
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
    assert pairs == [(0, 1)]

    # Pure vertical leg (x0 == x1): the horizontal leg is zero-length, so only
    # the column lookup can contribute. Crosses the single 0|2 boundary.
    ids = edge_segment_ids(rg, table, 10., 10., 10., 90.)
    np.testing.assert_array_equal(ids, _brute_edge_segments(rg, table, 10., 10., 10., 90.))
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
    assert pairs == [(0, 2)]

    # Elbow landing exactly on the 4-way junction at (50, 50): (10,50) is in
    # region 2 (NW), the elbow (50,50) itself resolves unambiguously to
    # region 3 (NE) (RegionGrid.to_idx floors, so the junction coordinate
    # belongs to the cell it is the *lower* corner of), and (50,10) is in
    # region 1 (SE). This is exactly the coordinate where a row/col mixup in
    # edge_segment_ids' axis composition would surface.
    ids = edge_segment_ids(rg, table, 10., 50., 50., 10.)
    np.testing.assert_array_equal(ids, _brute_edge_segments(rg, table, 10., 50., 50., 10.))
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
    assert pairs == [(2, 3), (1, 3)]


def test_select_candidates_breaks_group_ties_by_group_key():
    """Plan wording: 'ties by group key'. Three groups tied on total count
    (pairs (4,5), (8,9), (1,2), each count 40) compete for the last two of
    four m_pairs slots once (0,1)=60 and (5,6)=50 are seated; the tie-break
    must keep the two with the smaller ascending (net, u, v, boundary-pair)
    key -- here (1,2) and (4,5) -- and drop (8,9), never a choice driven by
    input array position or segment id."""
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    table = enumerate_segments(rg)
    segs, seen = [], set()
    for s in range(table.num_segments):
        key = (int(table.pair_a[s]), int(table.pair_b[s]))
        if key not in seen:
            seen.add(key)
            segs.append(s)
        if len(segs) == 6:
            break
    assert [(int(table.pair_a[s]), int(table.pair_b[s])) for s in segs] == \
        [(0, 1), (4, 5), (8, 9), (12, 13), (1, 2), (5, 6)]
    net = np.zeros(6, dtype=np.int64)
    u = np.array([int(table.pair_a[s]) for s in segs], dtype=np.int64)
    v = np.array([int(table.pair_b[s]) for s in segs], dtype=np.int64)
    seg = np.asarray(segs, dtype=np.int64)
    count = np.array([60, 40, 40, 10, 40, 50], dtype=np.int64)
    cand = select_candidates(net, u, v, seg, count, table, m_pairs=4, m_seg=2)
    assert cand.n_groups == 4
    kept_pairs = sorted({(int(a), int(b))
                         for a, b in zip(cand.u.tolist(), cand.v.tolist())})
    assert kept_pairs == [(0, 1), (1, 2), (4, 5), (5, 6)]
    assert sorted(cand.count.tolist()) == [40, 40, 50, 60]


@pytest.mark.parametrize("factory", [l_shape, plug, notch])
def test_edge_segment_ids_agrees_with_evaluator_ref_crossings(factory):
    """edge_segment_ids' docstring claims it walks the identical L-route
    geometry evaluator_ref.edge_regions_and_crossings does. That claim is
    load-bearing: if the capacity term's notion of 'which segments an edge
    crosses' ever disagreed with the evaluator's notion of 'crossings', the
    capacity term would optimise against a quantity the evaluator does not
    measure, and every capacity number in this subproject would be quietly
    meaningless while every other test here kept passing. Exhaustively drive
    both independently-implemented functions over every cell-center-to-
    cell-center edge on a small lattice and require exact agreement on both
    the crossing count and the ordered sequence of boundary pairs crossed."""
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    ny, nx = rg.grid.shape
    xl, yl, _xh, _yh = rg.die
    centers = [(xl + (i + 0.5) * rg.cell_w, yl + (j + 0.5) * rg.cell_h)
               for i in range(nx) for j in range(ny)]
    for x0, y0 in centers:
        for x1, y1 in centers:
            ids = edge_segment_ids(rg, table, x0, y0, x1, y1)
            ours = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
            _regs, ncross, pairs_ref = evaluator_ref.edge_regions_and_crossings(
                rg, x0, y0, x1, y1)
            assert len(ids) == ncross
            assert ours == pairs_ref


def test_edge_segment_ids_agrees_with_evaluator_ref_on_a_larger_lattice():
    """Same cross-check, sampled (deterministic seed) on a bigger lattice so
    at least one case exercises multiple crossings per leg on genuinely
    diagonal, non-cell-center routes."""
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    table = enumerate_segments(rg)
    rng = np.random.default_rng(0)
    xl, yl, xh, yh = rg.die
    pts = rng.uniform([xl, yl], [xh, yh], size=(200, 2))
    for (x0, y0), (x1, y1) in zip(pts[::2], pts[1::2]):
        ids = edge_segment_ids(rg, table, x0, y0, x1, y1)
        ours = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
        _regs, ncross, pairs_ref = evaluator_ref.edge_regions_and_crossings(
            rg, x0, y0, x1, y1)
        assert len(ids) == ncross
        assert ours == pairs_ref
