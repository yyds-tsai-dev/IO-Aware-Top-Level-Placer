import numpy as np
import pytest
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions, make_slicing_regions
from ioplace.region_grid import RegionGrid
from ioplace.route_eval.budgeted import BudgetedRouter, route_net
from ioplace.evaluator_ref import _walk_segment


def dense_regions():
    # Ten narrow regions inside a connected outer region; all on the lattice.
    outer = [[0, 0, 100, 49], [0, 51, 100, 100], [0, 49, 20, 51], [80, 49, 100, 51]]
    regs = [RegionSpec("outer", np.array(outer))]
    regs += [RegionSpec(f"dense{i}", np.array([[20+6*i, 49, 26+6*i, 51]])) for i in range(10)]
    return RegionGrid(RegionSet((0, 0, 100, 100), 100, regs))


def recount(rg, points):
    regs, pairs = set(), []
    return sum(_walk_segment(rg, *p, *q, regs, pairs) for p, q in zip(points[:-1], points[1:]))


def test_detour_avoids_dense_regions_with_fixed_hpwl():
    router = BudgetedRouter(dense_regions())
    flat = route_net(router, [5, 95], [50, 50], wirelength_budget=0)
    detour = route_net(router, [5, 95], [50, 50], wirelength_budget=.05)
    assert flat["crossings"].tolist() == [11]
    assert detour["crossings"].tolist() == [0]
    assert detour["wirelength"][0] == 93
    assert detour["hpwl"] == flat["hpwl"] == 90
    assert recount(router.rg, detour["points"][0]) == 0


def test_budget_never_violated_and_geometry_matches_independent_walk():
    rg = RegionGrid(make_slicing_regions((0, 0, 100, 100), 16, seed=9, lattice=64))
    router = BudgetedRouter(rg)
    rng = np.random.default_rng(7)
    a, b = rng.uniform(0, 100, (2, 150, 2))
    previous = None
    for budget in [0., .02, .05, .10]:
        result = router.route_edges(a, b, wirelength_budget=budget)
        points = result["points"]
        np.testing.assert_array_equal(points[:, 0], a)
        np.testing.assert_array_equal(points[:, -1], b)
        assert (np.min(np.abs(np.diff(points, axis=1)), axis=2) == 0).all()
        np.testing.assert_allclose(np.abs(np.diff(points, axis=1)).sum((1, 2)), result["wirelength"])
        assert (result["wirelength"] <= result["baseline_wirelength"] * (1 + budget)).all()
        np.testing.assert_array_equal([recount(rg, p) for p in points], result["crossings"])
        assert (result["crossings"] <= result["baseline_crossings"]).all()
        if previous is not None:
            assert (result["crossings"] <= previous).all()
        previous = result["crossings"]


def test_uniform_grid_is_a_negative_control_and_chunk_invariant():
    router = BudgetedRouter(RegionGrid(make_grid_regions((0, 0, 100, 100), 4, 4, lattice=64)))
    a = np.array([[0, 0], [20, 80], [50, 50], [100, 100]], dtype=float)
    b = np.array([[100, 100], [80, 20], [50, 50], [0, 0]], dtype=float)
    result = router.route_edges(a, b, wirelength_budget=.5)
    np.testing.assert_array_equal(result["crossings"], result["baseline_crossings"])
    for i in range(len(a)):
        one = router.route_edges(a[i:i+1], b[i:i+1], wirelength_budget=.5)
        np.testing.assert_array_equal(one["points"][0], result["points"][i])


def test_empty_and_invalid_endpoints():
    router = BudgetedRouter(dense_regions())
    assert router.route_edges(np.empty((0, 2)), np.empty((0, 2)))["points"].shape == (0, 4, 2)
    with pytest.raises(ValueError):
        router.route_edges([[-1, 0]], [[1, 0]])
    with pytest.raises(ValueError):
        router.route_edges([[0, 0]], [[1, 0]], wirelength_budget=float("nan"))


def test_evaluator_recomputes_io_ft_and_boundary_demand_from_detour():
    from ioplace.netlist import Netlist
    from ioplace.evaluator_ref import evaluate
    x, y = np.array([5., 95., 95.]), np.array([50., 50., 52.])
    nl = Netlist(x, y, np.ones(3), np.ones(3), 3, 0, 0,
                 np.zeros(3), np.zeros(3), np.arange(3), np.zeros(3, dtype=int),
                 np.arange(3), np.array([0, 3]), 0., 0., 100., 100.)
    rg = dense_regions()
    flat = evaluate(nl, x, y, rg)
    detour = evaluate(nl, x, y, rg, route_wirelength_budget=.05)
    assert flat.io_count == 11 and flat.ft_count == 10
    assert detour.io_count == detour.ft_count == 0
    assert sum(detour.boundary_pair_demand.values()) == detour.io_count
    assert detour.hpwl == flat.hpwl == 92.
    assert flat.tree_wl < detour.tree_wl <= flat.tree_wl * 1.05
    np.testing.assert_array_equal(detour.per_net_lambda, flat.per_net_lambda)
    # The adjacency-only region graph diagnostic is separate and unchanged.
    assert detour.io_rg == flat.io_rg


def test_resource_blockage_rejects_impossible_detour_and_routes_zero_io_edges():
    from ioplace.route_eval.resources import RoutingResources
    rg = RegionGrid(make_grid_regions((0,0,10,10),1,1,lattice=10))
    h,v=np.ones((10,9)),np.ones((9,10))
    h[5,4]=0
    router=BudgetedRouter(rg,resources=RoutingResources(rg,h,v,congestion_weight=0))
    result=router.route_edges([[1,5.5]],[[9,5.5]],wirelength_budget=.5)
    assert result['wirelength'][0]>8
    assert router.resources.evaluate(result['points'])[1].all()
    with pytest.raises(ValueError,match='no resource-feasible'):
        router.route_edges([[1,5.5]],[[9,5.5]],wirelength_budget=0)
    h[:,4]=0
    router=BudgetedRouter(rg,resources=RoutingResources(rg,h,v))
    with pytest.raises(ValueError,match='no resource-feasible'):
        router.route_edges([[1,5.5]],[[9,5.5]],wirelength_budget=2)


def test_frozen_congestion_cost_can_change_route_without_crossings():
    from ioplace.route_eval.resources import RoutingResources
    rg=RegionGrid(make_grid_regions((0,0,10,10),1,1,lattice=10))
    h,v=np.ones((10,9)),np.ones((9,10));d=np.zeros_like(h);d[5,:]=100
    resource=RoutingResources(rg,h,v,horizontal_demand=d)
    router=BudgetedRouter(rg,resources=resource)
    result=router.route_edges([[1,5.5]],[[9,5.5]],wirelength_budget=.5)
    assert result['wirelength'][0]>8
