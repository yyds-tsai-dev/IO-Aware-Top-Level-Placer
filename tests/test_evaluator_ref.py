import pytest
import numpy as np
from ioplace.evaluator_ref import net_mst_edges, net_mst_length
from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate, edge_regions_and_crossings
from tests.test_netlist import make_tiny_netlist

def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges.tolist()}

def test_mst_two_pins():
    e = net_mst_edges(np.array([0., 10.]), np.array([0., 0.]))
    assert _edge_set(e) == {(0, 1)}

def test_mst_three_pins_line():
    # (0,0),(5,0),(20,0): MST = {0-1, 1-2},非 {0-2}
    e = net_mst_edges(np.array([0., 5., 20.]), np.array([0., 0., 0.]))
    assert _edge_set(e) == {(0, 1), (1, 2)}
    assert net_mst_length(np.array([0., 5., 20.]), np.array([0., 0., 0.])) == 20.

def test_mst_length_matches_bruteforce_random():
    rng = np.random.default_rng(3)
    for _ in range(20):
        d = rng.integers(2, 8)
        px, py = rng.uniform(0, 100, d), rng.uniform(0, 100, d)
        got = net_mst_length(px, py)
        # brute force: Prim 的另一實作 —— 逐步取最短跨集合邊
        import itertools
        dist = np.abs(px[:, None]-px[None, :]) + np.abs(py[:, None]-py[None, :])
        in_tree = {0}; total = 0.
        while len(in_tree) < d:
            cands = [(dist[i, j], j) for i in in_tree for j in range(d) if j not in in_tree]
            w, j = min(cands)
            total += w; in_tree.add(j)
        assert got == pytest.approx(total)

# Tests for crossing, feedthrough, and evaluate

DIE = (0., 0., 100., 100.)

def _rg22():
    return RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))

def test_edge_crossing_simple():
    rg = _rg22()
    regions, ncross, pairs = edge_regions_and_crossings(rg, 10., 10., 90., 10.)
    assert regions == {0, 1} and ncross == 1 and pairs == [(0, 1)]

def test_edge_crossing_L_through_third_region():
    rg = _rg22()
    # (10,10)→(90,90) L: 水平段經 P0→P1,垂直段經 P1→P3 → 2 crossings,經過 {0,1,3}
    regions, ncross, pairs = edge_regions_and_crossings(rg, 10., 10., 90., 90.)
    assert regions == {0, 1, 3} and ncross == 2
    assert pairs == [(0, 1), (1, 3)]

def test_evaluate_tiny_netlist():
    rg = _rg22()
    nl = make_tiny_netlist()
    # cells: c0(10,10)P0 c1(30,10)P0 c2(50,40)P1 f3(90,90)P3
    # n0={c0,c1}: 全在 P0 → 0 crossing, 0 FT
    # n1={c1,c2,f3}: MST edges = (c1,c2),(c2,f3)
    #   (30,10)-(50,40): L → (50,10) 經 P0→P1 = 1 crossing;再 (50,10)-(50,40) 留 P1
    #   (50,40)-(90,90): L → (90,40) 留 P1;再 (90,40)-(90,90) P1→P3 = 1 crossing
    #   → n1 crossings=2;經過 {0,1,3} 全有 pin → FT=0
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    assert res.io_count == 2
    assert list(res.per_net_crossings) == [0, 2]
    assert res.ft_count == 0
    assert res.boundary_pair_demand == {(0, 1): 1, (1, 3): 1}
    assert res.hpwl == pytest.approx(160.0)     # net0: 20+0;net1: 60+80
    assert res.tree_wl == pytest.approx(160.0)  # net0: 20;net1: 50+90

def test_evaluate_pure_feedthrough():
    rg = _rg22()
    nl = make_tiny_netlist()
    # 搬位:c0(10,10)P0, c1(90,90)P3, c2(50,40)P1, f3(90,90)P3
    # n0={c0,c1}: L (10,10)→(90,10)→(90,90) 經 {P0,P1,P3},pins 只在 {P0,P3}
    #   → P1 是 pure feed-through;crossings = 2
    # n1={c1,c2,f3}: c1 與 f3 同點(邊長 0);(50,40)-(90,90) 的 L 經 {P1,P3} 皆有 pin
    #   → FT=0,crossing=1
    node_x = np.array([10., 90., 50., 90.]); node_y = np.array([10., 90., 40., 90.])
    res = evaluate(nl, node_x, node_y, rg)
    assert list(res.per_net_ft) == [1, 0]
    assert res.ft_count == 1
    assert list(res.per_net_crossings) == [2, 1]
    assert res.io_count == 3

def test_evaluate_bruteforce_random():
    rng = np.random.default_rng(11)
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rg = RegionGrid(rs)
    for _ in range(10):
        n_cells, n_nets = 12, 6
        nx_, ny_ = rng.uniform(1, 99, n_cells), rng.uniform(1, 99, n_cells)
        pins, p2n = [], []
        for net in range(n_nets):
            d = int(rng.integers(2, 5))
            pins += list(rng.integers(0, n_cells, d)); p2n += [net]*d
        pins, p2n = np.array(pins, np.int32), np.array(p2n, np.int32)
        order = np.argsort(p2n, kind="stable")
        flat = pins[order]
        start = np.searchsorted(p2n[order], np.arange(n_nets + 1))
        from ioplace.netlist import Netlist
        nl = Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                     node_size_y=np.ones(n_cells), num_movable=n_cells,
                     num_terminals=0, num_terminal_NIs=0,
                     pin_offset_x=np.zeros(len(flat)), pin_offset_y=np.zeros(len(flat)),
                     pin2node=flat, pin2net=p2n[order].astype(np.int32),
                     flat_net2pin=np.arange(len(flat), dtype=np.int32),
                     flat_net2pin_start=start.astype(np.int32),
                     xl=0., yl=0., xh=100., yh=100.)
        res = evaluate(nl, nl.node_x, nl.node_y, rg)
        # brute force:逐 lattice 小步 walk 每條 MST 邊,數 id 變化
        from ioplace.evaluator_ref import net_mst_edges
        from ioplace.netlist import pin_positions
        px, py = pin_positions(nl)
        io_bf = 0
        for net in range(n_nets):
            s, e = start[net], start[net + 1]
            for (a, b) in net_mst_edges(px[s:e], py[s:e]):
                pa, pb = (px[s+a], py[s+a]), (px[s+b], py[s+b])
                for seg in (((pa[0], pa[1]), (pb[0], pa[1])), ((pb[0], pa[1]), (pb[0], pb[1]))):
                    (x0, y0), (x1, y1) = seg
                    steps = 400
                    xs = np.linspace(x0, x1, steps); ys = np.linspace(y0, y1, steps)
                    ids = rg.region_of_points(xs, ys)
                    io_bf += int(np.count_nonzero(np.diff(ids)))
        assert res.io_count == io_bf

def test_evaluate_large_net_lower_bound():
    # degree > max_degree 的 net 不建樹:presence 下界 = distinct regions - 1,
    # 計入 io_count 與 large_net_lb;不進 tree_wl / boundary_pair_demand;hpwl 照算。
    rg = _rg22()
    n_pins = 5
    node_x = np.array([10., 20., 60., 70., 90.])
    node_y = np.array([10., 20., 10., 20., 90.])   # P0,P0,P1,P1,P3 → 3 distinct regions
    from ioplace.netlist import Netlist
    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=np.ones(n_pins), node_size_y=np.ones(n_pins),
                 num_movable=n_pins, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(n_pins), pin_offset_y=np.zeros(n_pins),
                 pin2node=np.arange(n_pins, dtype=np.int32),
                 pin2net=np.zeros(n_pins, dtype=np.int32),
                 flat_net2pin=np.arange(n_pins, dtype=np.int32),
                 flat_net2pin_start=np.array([0, n_pins], dtype=np.int32),
                 xl=0., yl=0., xh=100., yh=100.)
    res = evaluate(nl, node_x, node_y, rg, max_degree=2)   # 5 > 2 → 強制走大 net 分支
    assert list(res.per_net_crossings) == [2]      # 3 regions - 1
    assert res.large_net_lb == 2
    assert res.io_count == 2
    assert list(res.per_net_ft) == [0]
    assert res.tree_wl == 0.0
    assert res.boundary_pair_demand == {}
    assert res.hpwl == pytest.approx(160.0)        # (90-10)+(90-10)

def test_hard_lambda_fields_on_tiny_netlist():
    rg = _rg22()
    nl = make_tiny_netlist()
    # n0={c0,c1} both in P0 -> lambda=1 ; n1={c1,c2,f3} in P0,P1,P3 -> lambda=3
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    assert list(res.per_net_lambda) == [1, 3]
    assert res.hard_lambda_sum == 2

def test_hard_lambda_sum_never_exceeds_io_count():
    rng = np.random.default_rng(21)
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rg = RegionGrid(rs)
    from tests.test_evaluator_gpu import _random_case
    for s in range(5):
        nl = _random_case(np.random.default_rng(s))
        res = evaluate(nl, nl.node_x, nl.node_y, rg)
        assert res.hard_lambda_sum <= res.io_count
        assert len(res.per_net_lambda) == nl.num_nets
