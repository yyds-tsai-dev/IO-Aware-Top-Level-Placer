import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate
from ioplace.netlist import Netlist
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from ioplace.straddle import (STRADDLE_ARRAYS, STRADDLE_SCALARS,
                              distinct_regions_per_net, straddle_diagnostics,
                              straddle_geometry)

DIE = (0., 0., 100., 100.)


def _grid():
    """2x2 grid on a 10x10 lattice: cells are 10x10, region boundaries at
    x=50 and y=50, regions numbered row-major (P0 lower-left, P1 lower-right,
    P2 upper-left, P3 upper-right)."""
    return RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))


def _corner_case():
    """Four movable-plus-one-fixed layout with every interesting case present.

      node 0 (10,10) 4x4  -- wholly inside region 0
      node 1 (48,10) 4x4  -- crosses x=50 only; centre (50,12) -> owner 1
      node 2 (48,48) 4x4  -- crosses both lines; corners hit all four regions,
                             centre (50,50) -> owner 3
      node 3 (90,90) 4x4  -- FIXED (num_movable = 3), must be ignored

    Nets, all zero pin offsets, so pin positions are the cells' lower-left
    corners and every pin of nodes 1 and 2 sits in region 0 before
    re-attribution:
      net 0 {0,1}: {0,0} -> lambda 1;  re {0,1} -> lambda 2   (split rises)
      net 1 {1,3}: {0,3} -> lambda 2;  re {1,3} -> lambda 2   (no change)
      net 2 {2,3}: {0,3} -> lambda 2;  re {3,3} -> lambda 1   (split drops)
      net 3 {0,2}: {0,0} -> lambda 1;  re {0,3} -> lambda 2   (split rises)
    """
    pin2node = np.array([0, 1, 1, 3, 2, 3, 0, 2], np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
    return Netlist(node_x=np.array([10., 48., 48., 90.]),
                   node_y=np.array([10., 10., 48., 90.]),
                   node_size_x=np.full(4, 4.), node_size_y=np.full(4, 4.),
                   num_movable=3, num_terminals=1, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(8), pin_offset_y=np.zeros(8),
                   pin2node=pin2node, pin2net=pin2net,
                   flat_net2pin=np.arange(8, dtype=np.int32),
                   flat_net2pin_start=np.array([0, 2, 4, 6, 8], np.int32),
                   xl=0., yl=0., xh=100., yh=100.)


def test_field_name_contracts():
    assert STRADDLE_SCALARS == ("straddle_cells", "straddle_area_fraction",
                                "straddle_pin_split_nets", "straddle_out_area",
                                "straddle_movable_area", "straddle_wide_cells")
    assert STRADDLE_ARRAYS == ("per_node_straddle", "per_net_pin_split")


def test_geometry_reads_the_four_corners_and_the_centre_owner():
    rg = _grid()
    x = np.array([10., 48., 48.])
    y = np.array([10., 10., 48.])
    w = np.full(3, 4.)
    h = np.full(3, 4.)
    straddle, owner, out_area, wide = straddle_geometry(rg, x, y, w, h)
    assert straddle.tolist() == [False, True, True]
    assert owner.tolist() == [0, 1, 3]
    # node 1: the 2x4 strip left of x=50 is region 0, the owner is 1 -> 8.
    # node 2: three of the four 2x2 quadrants are not region 3 -> 12.
    assert out_area.tolist() == [0., 8., 12.]
    assert wide.tolist() == [False, False, False]


def test_quadrant_areas_always_sum_to_the_cell_area():
    rg = _grid()
    rng = np.random.default_rng(0)
    x = rng.uniform(0., 95.,  400)
    y = rng.uniform(0., 95.,  400)
    w = rng.uniform(0.5, 4.5, 400)
    h = rng.uniform(0.5, 4.5, 400)
    _, owner, out_area, _ = straddle_geometry(rg, x, y, w, h)
    # out_area is a sub-sum of the same four quadrants, so it can never exceed
    # the cell area and is zero exactly when every quadrant belongs to the owner
    assert np.all(out_area <= w * h + 1e-9)
    assert np.all(out_area >= 0.)


def test_wide_cells_are_flagged_and_approximated_not_silently_wrong():
    """A cell spanning four lattice columns: the quadrant split attributes the
    whole 25-wide right part to the right corner's region, so it reports 100
    where the exact out-of-owner area is 60. The point of straddle_wide_cells
    is that this is counted, not hidden."""
    rg = _grid()
    straddle, owner, out_area, wide = straddle_geometry(
        rg, np.array([30.]), np.array([10.]), np.array([35.]), np.array([4.]))
    assert straddle.tolist() == [True] and owner.tolist() == [0]
    assert wide.tolist() == [True]
    assert out_area.tolist() == [100.]


def test_cell_touching_a_boundary_exactly_counts_as_straddling():
    """Closed-box convention: the right edge landing exactly on x=50 puts the
    (x+w) corner in region 1."""
    rg = _grid()
    straddle, owner, out_area, _ = straddle_geometry(
        rg, np.array([46.]), np.array([10.]), np.array([4.]), np.array([4.]))
    assert straddle.tolist() == [True] and owner.tolist() == [0]
    assert out_area.tolist() == [0.]     # every quadrant with area is region 0


def test_distinct_regions_per_net_is_zero_for_degree_one_nets():
    nl = _corner_case()
    rg = _grid()
    px, py = nl.node_x[nl.pin2node], nl.node_y[nl.pin2node]
    lam = distinct_regions_per_net(nl, rg.region_of_points(px, py))
    assert lam.tolist() == [1, 2, 2, 1]


def test_diagnostics_on_the_corner_case():
    nl = _corner_case()
    st = straddle_diagnostics(nl, nl.node_x, nl.node_y, _grid())
    assert st.straddle_cells == 2
    assert st.straddle_out_area == pytest.approx(20.0)
    assert st.straddle_movable_area == pytest.approx(48.0)      # 3 movable 4x4 cells
    assert st.straddle_area_fraction == pytest.approx(20.0 / 48.0)
    assert st.straddle_wide_cells == 0
    assert st.per_node_straddle.tolist() == [0, 1, 1, 0]         # the terminal is excluded
    assert st.per_net_pin_split.tolist() == [-1, 0, 1, -1]
    assert st.straddle_pin_split_nets == 1


def test_diagnostics_are_all_zero_when_nothing_straddles():
    nl = _corner_case()
    nl.node_x = np.array([10., 10., 60., 90.])
    nl.node_y = np.array([10., 20., 60., 90.])
    st = straddle_diagnostics(nl, nl.node_x, nl.node_y, _grid())
    assert st.straddle_cells == 0
    assert st.straddle_area_fraction == 0.0
    assert st.straddle_pin_split_nets == 0
    assert st.per_net_pin_split.tolist() == [0, 0, 0, 0]


def test_pin_offsets_move_the_attribution_not_the_geometry():
    """A cell wholly inside region 0 whose pin offset puts its pin in region 1
    does not straddle, so its pins are not re-attributed."""
    nl = _corner_case()
    nl.pin_offset_x = np.array([0., 0., 0., 0., 0., 0., 40., 0.])   # pin 6 (node 0)
    st = straddle_diagnostics(nl, nl.node_x, nl.node_y, _grid())
    assert st.per_node_straddle[0] == 0
    assert st.straddle_cells == 2


def test_evaluate_reports_the_diagnostics_and_leaves_the_legacy_fields_alone():
    nl = _corner_case()
    rg = _grid()
    off = evaluate(nl, nl.node_x, nl.node_y, rg, straddle=False)
    on = evaluate(nl, nl.node_x, nl.node_y, rg)
    for field in ("io_count", "ft_count", "hard_lambda_sum", "io_rg", "ft_rg",
                  "large_net_lb", "tree_wl", "hpwl", "boundary_pair_demand"):
        assert getattr(on, field) == getattr(off, field)
    assert np.array_equal(on.per_net_lambda, off.per_net_lambda)
    assert off.straddle_cells == 0 and off.per_node_straddle is None
    assert on.straddle_cells == 2
    assert on.straddle_pin_split_nets == 1
    assert on.straddle_area_fraction == pytest.approx(20.0 / 48.0)
    assert on.per_node_straddle.dtype == np.uint8
    assert on.per_net_pin_split.dtype == np.int32


def test_lambda_matches_the_evaluators_own_per_net_lambda():
    """distinct_regions_per_net must agree with the loop evaluator_ref already
    runs, or per_net_pin_split would be measured against a different baseline."""
    nl = _corner_case()
    rg = _grid()
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    px, py = nl.node_x[nl.pin2node], nl.node_y[nl.pin2node]
    assert distinct_regions_per_net(nl, rg.region_of_points(px, py)).tolist() \
        == res.per_net_lambda.tolist()
