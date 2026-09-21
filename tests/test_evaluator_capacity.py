import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate
from ioplace.region_grid import RegionGrid
from ioplace.region_segments import (edge_segment_ids, enumerate_segments,
                                     segment_utilisation)
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions
from tests.test_io_term import _nl

DIE = (0., 0., 90., 30.)


def _strip_case():
    """Regions 0|1|2 left to right; one 2-pin net from region 0 to region 2,
    which feeds through region 1 and therefore crosses both boundaries."""
    rs = make_grid_regions(DIE, 3, 1, lattice=9)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    nl = _nl([(15., 15.), (75., 15.)], [[0, 1]])
    return nl, rg, table


def test_segment_demand_counts_every_crossing_once():
    nl, rg, table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table)
    np.testing.assert_array_equal(res.segment_demand, [1, 1])
    assert int(res.segment_demand.sum()) == res.io_count
    assert res.io_count == 2


def test_segment_demand_matches_a_brute_force_walk():
    rng = np.random.default_rng(3)
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 40
    xy = list(zip(rng.uniform(1., 99., n_cells), rng.uniform(1., 99., n_cells)))
    nets = [sorted(rng.choice(n_cells, int(rng.integers(2, 6)),
                              replace=False).tolist()) for _ in range(30)]
    nl = _nl(xy, nets)
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table)
    # independent recount straight off the raster, via the reference MST
    from ioplace.evaluator_ref import net_mst_edges
    from ioplace.netlist import pin_positions
    px, py = pin_positions(nl, nl.node_x, nl.node_y)
    expected = np.zeros(table.num_segments, dtype=np.int64)
    start = nl.flat_net2pin_start
    for net in range(nl.num_nets):
        s, e = start[net], start[net + 1]
        if e - s <= 1:
            continue
        idx = nl.flat_net2pin[s:e]
        nx_, ny_ = px[idx], py[idx]
        for a, b in net_mst_edges(nx_, ny_):
            for sid in edge_segment_ids(rg, table, nx_[a], ny_[a],
                                        nx_[b], ny_[b]):
                expected[sid] += 1
    np.testing.assert_array_equal(res.segment_demand, expected)
    assert int(res.segment_demand.sum()) == res.io_count - res.large_net_lb


def test_segment_demand_sums_to_the_pair_demand_per_pair():
    """The two accountings must agree: sec 5 keeps ell[a,b] as the pair-level
    aggregate and segment demand as its refinement."""
    rng = np.random.default_rng(11)
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 30
    xy = list(zip(rng.uniform(1., 99., n_cells), rng.uniform(1., 99., n_cells)))
    nets = [sorted(rng.choice(n_cells, 3, replace=False).tolist())
            for _ in range(25)]
    nl = _nl(xy, nets)
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table)
    rolled = {}
    for s in range(table.num_segments):
        key = (int(table.pair_a[s]), int(table.pair_b[s]))
        rolled[key] = rolled.get(key, 0) + int(res.segment_demand[s])
    assert {k: v for k, v in rolled.items() if v} == \
        {k: v for k, v in res.boundary_pair_demand.items() if v}


def test_capacity_fields_come_from_the_shared_helper():
    nl, rg, table = _strip_case()
    capacity = np.array([0.5, 4.0])
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   segment_capacity=capacity)
    util, scalars = segment_utilisation(res.segment_demand, capacity)
    np.testing.assert_array_equal(res.segment_util, util)
    np.testing.assert_array_equal(res.segment_capacity, capacity)
    assert res.num_over_capacity == scalars["num_over_capacity"] == 1
    assert res.max_util == scalars["max_util"] == pytest.approx(2.0)
    assert res.p99_util == scalars["p99_util"]
    assert res.num_zero_capacity_segments == 0
    assert res.zero_capacity_demand == 0


def test_a_zero_capacity_segment_counts_as_over_capacity():
    nl, rg, table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   segment_capacity=np.array([0.0, 4.0]))
    assert res.num_over_capacity == 1
    assert res.num_zero_capacity_segments == 1
    assert res.zero_capacity_demand == 1
    assert res.segment_util[0] == np.inf
    assert res.max_util == pytest.approx(0.25)     # zero-capacity excluded


def test_candidates_carry_the_terminal_pair_for_a_feed_through():
    """Interpretation D-1: the demand pair is the MST-edge endpoints' regions
    (0 and 2), not the segments' own boundary pairs."""
    nl, rg, table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   capacity_candidates=True)
    np.testing.assert_array_equal(res.cand_net, [0, 0])
    np.testing.assert_array_equal(res.cand_u, [0, 0])
    np.testing.assert_array_equal(res.cand_v, [2, 2])
    np.testing.assert_array_equal(res.cand_seg, [0, 1])
    np.testing.assert_array_equal(res.cand_count, [1, 1])
    assert res.cand_dropped == 0


def test_candidates_are_sorted_by_the_composite_key():
    rng = np.random.default_rng(5)
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 30
    xy = list(zip(rng.uniform(1., 99., n_cells), rng.uniform(1., 99., n_cells)))
    nets = [sorted(rng.choice(n_cells, 3, replace=False).tolist())
            for _ in range(20)]
    nl = _nl(xy, nets)
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   capacity_candidates=True)
    k, size = rg.k, table.num_segments
    key = ((res.cand_net * k + res.cand_u) * k + res.cand_v) * size + res.cand_seg
    assert (np.diff(key) > 0).all()
    assert (res.cand_u < res.cand_v).all()
    assert int(res.cand_count.sum()) + res.cand_dropped == \
        int(res.segment_demand.sum())


def test_same_region_legs_are_dropped_and_counted():
    """An L-route between two pins of the same region can still detour through
    a neighbour. Such crossings carry no q-expressible signal (u == v), so they
    are dropped from the candidate list -- and the drop is reported."""
    rs = RegionSet(die=DIE, lattice=9, regions=[
        RegionSpec("A", np.array([[0., 0., 30., 30.], [60., 0., 90., 30.]])),
        RegionSpec("B", np.array([[30., 0., 60., 30.]]))])
    rs.validate()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    nl = _nl([(15., 15.), (75., 15.)], [[0, 1]])     # both pins in region 0
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   capacity_candidates=True)
    assert int(res.segment_demand.sum()) == 2
    assert res.cand_net.size == 0
    assert res.cand_dropped == 2


def test_segments_from_a_different_grid_are_rejected():
    nl, rg, _table = _strip_case()
    other = enumerate_segments(RegionGrid(make_grid_regions(DIE, 2, 1, lattice=9)))
    with pytest.raises(ValueError, match="segment table"):
        evaluate(nl, nl.node_x, nl.node_y, rg, segments=other)


def test_capacity_fields_stay_none_when_no_segments_are_supplied():
    nl, rg, _table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    for name in ("segment_demand", "segment_capacity", "segment_util",
                 "num_over_capacity", "max_util", "p99_util", "cand_net"):
        assert getattr(res, name) is None
