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
