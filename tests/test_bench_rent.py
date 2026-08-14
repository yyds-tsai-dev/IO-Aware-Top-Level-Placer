import numpy as np
import pytest

from ioplace.bench import rent


def _netlist(node_x, node_y, pins, p2n, n_nodes):
    from ioplace.netlist import Netlist
    pins = np.asarray(pins, dtype=np.int32)
    p2n = np.asarray(p2n, dtype=np.int32)
    n_nets = int(p2n[-1]) + 1 if len(p2n) else 0
    start = np.searchsorted(p2n, np.arange(n_nets + 1)).astype(np.int32)
    return Netlist(node_x=np.asarray(node_x, dtype=np.float64), node_y=np.asarray(node_y, dtype=np.float64),
                    node_size_x=np.ones(n_nodes), node_size_y=np.ones(n_nodes),
                    num_movable=n_nodes, num_terminals=0, num_terminal_NIs=0,
                    pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                    pin2node=pins, pin2net=p2n, flat_net2pin=np.arange(len(pins), dtype=np.int32),
                    flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)


def _toy_two_cliques_netlist():
    """Two 4-cliques bridged by 1 net -- the same 8-node smoke case
    `docs/dev-env.md`'s Task 8 section and `tests/test_hgr.py`'s
    `test_partition_two_clusters` use to confirm the installed
    mtkahypar==1.6.2 API works end to end."""
    nets = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [3, 4]]
    pins, p2n = [], []
    for i, ns in enumerate(nets):
        pins += ns
        p2n += [i] * len(ns)
    return _netlist(np.zeros(8), np.zeros(8), pins, p2n, 8)


def _grid_torus_netlist(G):
    """G x G grid with periodic (torus) wraparound in both axes -- see the
    module docstring's reasoning: a *non*-periodic grid's recursive
    bisection is dominated for many levels by a die-boundary finite-size
    correction (blocks near the top of the hierarchy still touch the outer
    edge on 1-2 sides, not 4), which biases the fitted exponent well below
    the textbook p=0.5 unless the grid is enormous. A torus has no edge --
    every bisection immediately produces blocks whose every non-adjacent
    side already borders another block -- so p=0.5 is recoverable at toy
    scale."""
    n = G * G
    node_x = np.array([i % G for i in range(n)], dtype=np.float64)
    node_y = np.array([i // G for i in range(n)], dtype=np.float64)
    pins, p2n = [], []
    net = 0
    for gy in range(G):
        for gx in range(G):
            idx = gy * G + gx
            xnbr = gy * G + (gx + 1) % G
            ynbr = ((gy + 1) % G) * G + gx
            pins += [idx, xnbr]
            p2n += [net, net]
            net += 1
            pins += [idx, ynbr]
            p2n += [net, net]
            net += 1
    return _netlist(node_x, node_y, pins, p2n, n)


def _fully_random_netlist(n, avg_deg, seed):
    """n nodes, n*avg_deg/2 two-pin nets with both endpoints drawn
    uniformly at random (no spatial/hierarchical locality at all) -- the
    spec's "completely random" p~1.0 synthetic case."""
    rng = np.random.default_rng(seed)
    n_nets = (n * avg_deg) // 2
    node_x = rng.uniform(0, 100, n)
    node_y = rng.uniform(0, 100, n)
    a = rng.integers(0, n, size=n_nets)
    b = (a + rng.integers(1, n, size=n_nets)) % n  # a != b guaranteed
    pins = np.empty(2 * n_nets, dtype=np.int32)
    pins[0::2] = a
    pins[1::2] = b
    p2n = np.repeat(np.arange(n_nets), 2)
    return _netlist(node_x, node_y, pins, p2n, n)


# ---------------------------------------------------------------------------
# mtkahypar availability (task's explicit ask: import succeeds + an 8-node
# toy bisection actually runs through ioplace.bench.rent's own call path)
# ---------------------------------------------------------------------------

def test_mtkahypar_importable_and_bisects_8_node_toy_case():
    pytest.importorskip("mtkahypar")
    nl = _toy_two_cliques_netlist()
    node_ids = np.arange(8, dtype=np.int64)
    left, right = rent.bisect_mtkahypar(node_ids, nl, seed=0)
    assert len(left) == 4 and len(right) == 4
    assert set(left.tolist()) in ({0, 1, 2, 3}, {4, 5, 6, 7})
    assert set(right.tolist()) in ({0, 1, 2, 3}, {4, 5, 6, 7})
    assert set(left.tolist()) != set(right.tolist())


def test_block_terminal_counts_on_two_cliques():
    nl = _toy_two_cliques_netlist()
    blocks = [np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64)]
    terms = rent.block_terminal_counts(blocks, nl)
    # exactly 1 net (the bridge) crosses the two blocks -> both see 1 external terminal
    assert list(terms) == [1, 1]


# ---------------------------------------------------------------------------
# Rent exponent on known-p synthetic cases (design draft sec 7.1 T5
# acceptance: regular mesh p~0.5, fully random p~1.0, both +/-0.05)
# ---------------------------------------------------------------------------

def test_regular_mesh_rent_exponent_near_0_5():
    pytest.importorskip("mtkahypar")
    nl = _grid_torus_netlist(16)
    res = rent.measure_rent(nl, b_lo=4, b_hi=64, seed=0, backend="mtkahypar", n_bootstrap=20)
    assert res.backend == "mtkahypar"
    assert abs(res.p - 0.5) <= 0.05, f"p={res.p} not within 0.05 of the textbook grid Rent exponent 0.5"
    assert res.p_ci_lo <= res.p <= res.p_ci_hi


def test_fully_random_rent_exponent_near_1_0():
    pytest.importorskip("mtkahypar")
    nl = _fully_random_netlist(n=1024, avg_deg=64, seed=0)
    res = rent.measure_rent(nl, b_lo=1, b_hi=8, seed=0, backend="mtkahypar", n_bootstrap=20)
    assert res.backend == "mtkahypar"
    assert abs(res.p - 1.0) <= 0.05, f"p={res.p} not within 0.05 of the textbook random-graph Rent exponent 1.0"
    assert res.p_ci_lo <= res.p <= res.p_ci_hi


def test_measure_rent_is_deterministic_for_a_fixed_seed():
    pytest.importorskip("mtkahypar")
    nl = _grid_torus_netlist(16)
    r1 = rent.measure_rent(nl, b_lo=4, b_hi=64, seed=0, backend="mtkahypar", n_bootstrap=5)
    r2 = rent.measure_rent(nl, b_lo=4, b_hi=64, seed=0, backend="mtkahypar", n_bootstrap=5)
    assert r1.p == r2.p
    assert r1.levels_used == r2.levels_used


def test_geometric_fallback_backend_is_recorded_and_runs():
    """sec 3.3's named fallback when mtkahypar is unavailable -- exercised
    directly here (not gated on mtkahypar's absence) so it's covered
    regardless of what's installed in the test environment."""
    nl = _grid_torus_netlist(16)
    res = rent.measure_rent(nl, b_lo=4, b_hi=64, seed=0, backend="geometric", n_bootstrap=5)
    assert res.backend == "geometric"
    assert not np.isnan(res.p)


def test_auto_backend_prefers_mtkahypar_when_available():
    pytest.importorskip("mtkahypar")
    nl = _grid_torus_netlist(8)
    res = rent.measure_rent(nl, b_lo=2, b_hi=16, seed=0, backend="auto", n_bootstrap=5)
    assert res.backend == "mtkahypar"


def test_too_few_usable_levels_reports_nan_not_a_crash():
    nl = _toy_two_cliques_netlist()
    # b window that no level's average block size can land in
    res = rent.measure_rent(nl, max_level=1, b_lo=1e6, b_hi=1e7, backend="geometric")
    assert res.p != res.p  # NaN
    assert res.levels_used == []
