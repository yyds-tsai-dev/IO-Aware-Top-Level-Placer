"""Behavior contracts for shared-net resource use and transactional routing."""
import numpy as np
import pytest
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def state(capacity=1., **kwargs):
    from ioplace.route_eval.joint import ResourceGrid, JointRoutingState
    region = RegionGrid(make_grid_regions((0, 0, 5, 3), 1, 1, lattice=10))
    resources = ResourceGrid(np.arange(6), np.arange(4))
    return JointRoutingState(region, resources, np.full(resources.edge_count, capacity), **kwargs)


def test_shared_reversed_branches_consume_one_unit_per_net_but_other_nets_add():
    # Bug caught: treating branch multiplicity as demand or deduping across nets.
    s = state()
    s.install_segments(7, [[[.5,.5],[3.5,.5]], [[4.5,.5],[1.5,.5]]])
    np.testing.assert_array_equal(s.demand[:4], [1,1,1,1])
    assert s.metrics()['wirelength'] == 4
    s.install_segments(8, [[[2.5,.5],[4.5,.5]]])
    np.testing.assert_array_equal(s.demand[:4], [1,1,2,2])
    assert s.metrics()['overflow'] == 2
    s.remove(7)
    np.testing.assert_array_equal(s.demand[:4], [0,0,1,1])
    assert s.metrics()['overflow'] == 0


def test_fork_replacement_and_failed_routing_leave_original_unchanged():
    # Bug caught: stale old demand after replacement, or mutating a parent trial.
    s = state()
    s.install_segments(7, [[[.5,.5],[4.5,.5]]])
    trial = s.fork()
    trial.install_segments(7, [[[.5,1.5],[4.5,1.5]]])
    np.testing.assert_array_equal(s.demand[:8], [1,1,1,1,0,0,0,0])
    np.testing.assert_array_equal(trial.demand[:8], [0,0,0,0,1,1,1,1])
    blocked = state(0.)
    before = blocked.demand.copy()
    with pytest.raises(ValueError, match='feasible'):
        blocked.replace(2, [[.5,.5],[4.5,.5]])
    np.testing.assert_array_equal(blocked.demand, before)
    assert not blocked.routes


def test_real_flute_multi_pin_tree_enters_joint_cost_and_demand():
    # Bug caught: using MST length or routing only two-pin nets.
    from ioplace.route_eval.joint import ResourceGrid, JointRoutingState
    rg = RegionGrid(make_grid_regions((0,0,11,11),1,1,lattice=11))
    grid = ResourceGrid(np.arange(12), np.arange(12))
    s = JointRoutingState(rg,grid,np.full(grid.edge_count,10.))
    route = s.replace(3, [[.5,.5],[10.5,.5],[5.5,10.5]])
    assert route.topology == 'flute'
    assert route.wirelength == pytest.approx(20.)  # MST would be 25.
    assert s.metrics()['multi_pin_nets'] == 1
    assert s.demand.sum() == 20
    np.testing.assert_array_equal(s.demand,s.recompute_demand())


def test_second_net_routes_against_first_net_shared_capacity():
    # Bug caught: independent frozen routing sends both nets onto the same path.
    s = state(congestion_weight=10., wirelength_budget=.6)
    s.install_segments(1, [[[.5,.5],[4.5,.5]]])
    route = s.replace(2, [[.5,.5],[4.5,.5]])
    assert route.wirelength == pytest.approx(6.)
    assert s.metrics()['overflow'] == 0
    assert (s.demand <= 1).all()


def test_joint_objective_delta_matches_full_recomputation_after_replace():
    s = state(congestion_weight=2.)
    s.replace(1, [[.5,.5],[4.5,.5],[2.5,2.5]])
    s.replace(2, [[.5,.5],[4.5,.5]])
    s.replace(1, [[.5,1.5],[4.5,1.5],[2.5,2.5]])
    np.testing.assert_array_equal(s.demand,s.recompute_demand())
    before = s.metrics()
    fresh = state(congestion_weight=2.)
    for net,route in s.routes.items():fresh.install_segments(net,route.segments,pins=route.pins)
    assert before['objective'] == pytest.approx(fresh.metrics()['objective'])
    assert before['crossings'] == fresh.metrics()['crossings']
