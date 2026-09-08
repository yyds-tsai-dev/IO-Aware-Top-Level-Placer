import numpy as np
import pytest

from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from ioplace.route_eval.boundary_distance import wire_boundary_distance
from ioplace.route_eval.route_crossings import CoordMap
from ioplace.route_eval.segments import Segments, KIND_VIA, KIND_WIRE


def grid():
    return RegionGrid(make_grid_regions((0, 0, 8, 4), 2, 2, lattice=4))


def seg(rows, nets=1):
    rows = np.asarray(rows, dtype=float).reshape(-1, 4)
    n = len(rows)
    return Segments(np.zeros(n, np.int32), np.full(n, KIND_WIRE, np.uint8),
                    np.zeros(n, np.int16), rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3],
                    np.zeros(n, np.int64), np.full(n, -1, np.int32),
                    np.asarray([f"n{i}" for i in range(nets)]), np.ones(nets, bool),
                    np.asarray(["M1"]), np.asarray([], dtype="<U1"), {})


def run(s, distances=(.25, .5, 1.0), cm=None, mask=None, chunk=250000):
    return wire_boundary_distance(s, grid(), cm or CoordMap.identity(), mask, distances, chunk)


def test_horizontal_vertical_distance_cdf_and_parallel():
    h = run(seg([(0, .5, 8, .5)]))
    assert h["any_boundary_cdf"] == pytest.approx([.125, .25, .5])
    assert h["parallel_boundary_cdf"] == pytest.approx([0, 0, 0])
    h2 = run(seg([(0, .5, 8, .5)]), distances=(1.0, 2.0))
    assert h2["parallel_boundary_cdf"] == pytest.approx([0, 1])
    v = run(seg([(1, 0, 1, 4)]))
    assert v["any_boundary_cdf"] == pytest.approx([.125, .25, .5])
    assert v["parallel_boundary_cdf"] == pytest.approx([0, 0, 0])
    assert run(seg([(1, 0, 1, 4)]), distances=(1.5,))["parallel_boundary_cdf"] == [1]


def test_boundary_collinear_weight_clipping_and_union():
    assert run(seg([(0, 2, 8, 2)]), distances=(0, .5, 1))["any_boundary_cdf"] == [1, 1, 1]
    s = seg([(0, 2, 2, 2), (0, 0, 8, 0)])
    assert run(s, distances=(0,))["any_boundary_cdf"] == pytest.approx([.2])
    # Radius 2 DBU around cuts 2,4,6 merges to cover the full 0..8 span.
    rg = RegionGrid(make_grid_regions((0, 0, 8, 4), 4, 1, lattice=4))
    assert wire_boundary_distance(seg([(0, 1, 8, 1)]), rg, CoordMap.identity(), distances=(1,))["any_boundary_cdf"] == [1.0]
    # Partial union must be 7/8: summing overlapping bands would exceed one
    # and output clipping could otherwise conceal that bug.
    rg = RegionGrid(make_grid_regions((0, 0, 8, 4), 4, 1, lattice=8))
    result = wire_boundary_distance(seg([(0, 1, 8, 1)]), rg, CoordMap.identity(), distances=(1.5,))
    assert result["any_boundary_cdf"] == [.875]
    assert result["parallel_boundary_cdf"] == [0.]


def test_outside_reversed_split_and_chunk_invariance():
    outside = run(seg([(-4, .5, -1, .5)]), distances=(1,))
    assert outside["available"] is False and outside["any_boundary_cdf"] == [None]
    clipped = run(seg([(-4, .5, 12, .5)]), distances=(.25,))
    assert clipped["total_wire_length_dbu"] == 16 and clipped["inside_die_wire_length_dbu"] == 8
    assert clipped["outside_die_wire_length_dbu"] == 8 and clipped["any_boundary_cdf"] == pytest.approx([.125])
    a = run(seg([(0, .5, 8, .5)]), distances=(.25, 1, 2), chunk=1)
    b = run(seg([(8, .5, 4, .5), (4, .5, 0, .5)]), distances=(.25, 1, 2), chunk=250)
    assert a["any_boundary_cdf"] == pytest.approx(b["any_boundary_cdf"])
    assert a["parallel_boundary_cdf"] == pytest.approx(b["parallel_boundary_cdf"])


def test_mask_ignores_zero_wire_and_via():
    s = seg([(0, .5, 8, .5), (0, 0, 0, 0)])
    s.seg_kind[1] = KIND_VIA
    s.seg_net_id[1] = 0
    out = run(s, distances=(0,))
    assert out["positive_wire_segments"] == 1 and out["zero_length_wire_rows"] == 0
    z = seg([(0, .5, 8, .5), (0, .5, 0, .5)])
    assert run(z, distances=(0,))["zero_length_wire_rows"] == 1
    z.net_names = np.asarray(["a", "b"]); z.net_has_wire = np.asarray([True, True]); z.seg_net_id = np.array([0, 1])
    masked = run(z, distances=(.25,), mask=np.array([True, False]))
    assert masked["total_wire_length_dbu"] == 8 and masked["inside_die_wire_length_dbu"] == 8


def test_mask_changes_cdf_for_two_positive_wires():
    z = seg([(0, .5, 8, .5), (0, 0, 3, 0)])
    z.net_names = np.asarray(["boundary", "far"]); z.net_has_wire = np.asarray([True, True]); z.seg_net_id = np.array([0, 1])
    assert run(z, distances=(.25,), mask=np.array([True, True]))["any_boundary_cdf"] == pytest.approx([1/11])
    assert run(z, distances=(.25,), mask=np.array([False, True]))["any_boundary_cdf"] == [0.0]


@pytest.mark.parametrize("dist", [(1, 0), (-1, 1), (np.nan, 1)])
def test_invalid_thresholds(dist):
    with pytest.raises(ValueError): run(seg([(0, 0, 1, 0)]), distances=dist)


def test_non_manhattan_and_partial_grid_rejected():
    with pytest.raises(ValueError): run(seg([(0, 0, 1, 1)]))
    rg = grid(); rg.grid[0, 0] = rg.grid[0, 2]
    with pytest.raises(ValueError, match="full rectangular"):
        wire_boundary_distance(seg([(0, 0, 8, 0)]), rg, CoordMap.identity())


def test_coord_map_scale_shift_and_cut_count():
    base = run(seg([(0, .5, 8, .5)]), distances=(.25, .5, 1.0))
    out = run(seg([(3, 4.25, 7, 4.25)]), distances=(.25, .5, 1.0), cm=CoordMap((3, 4), 2))
    assert out["any_boundary_cdf"] == pytest.approx(base["any_boundary_cdf"])
    assert out["inside_die_wire_length_dbu"] == 4
    assert out["vertical_cut_count"] == 1 and out["horizontal_cut_count"] == 1
