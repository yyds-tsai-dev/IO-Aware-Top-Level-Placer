import pytest
import numpy as np
from ioplace.evaluator_ref import net_mst_edges, net_mst_length

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
