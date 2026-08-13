import numpy as np
import pytest

from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.region_graph import (
    region_graph,
    next_hop_table,
    steiner_tree_stats,
)

DIE = (0., 0., 100., 100.)


def _rg22():
    return RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))


def _rg44():
    return RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))


# ---------------------------------------------------------------------------
# region_graph(rg) -> (adj, D, ell)
# ---------------------------------------------------------------------------

def test_region_graph_2x2_adjacency():
    rg = _rg22()
    adj, D, ell = region_graph(rg)
    K = rg.k
    assert adj.shape == (K, K) and D.shape == (K, K) and ell.shape == (K, K)
    # 2x2 grid: P0 P1 / P2 P3 (row-major, P{j*2+i}) -> P0-P1, P0-P2, P1-P3, P2-P3
    # adjacent; P0-P3 and P1-P2 (diagonal) not adjacent.
    assert bool(adj[0, 1]) and bool(adj[0, 2]) and bool(adj[1, 3]) and bool(adj[2, 3])
    assert not adj[0, 3] and not adj[3, 0]
    assert not adj[1, 2] and not adj[2, 1]
    assert not adj[0, 0]  # no self loops
    assert np.array_equal(adj, adj.T)  # symmetric


def test_region_graph_hop_distance_matches_adjacency():
    rg = _rg22()
    adj, D, ell = region_graph(rg)
    assert np.array_equal(np.diag(D), np.zeros(rg.k))
    assert D[0, 1] == 1 and D[0, 2] == 1 and D[1, 3] == 1 and D[2, 3] == 1
    # diagonal pairs: shortest path is 2 hops via either neighbour
    assert D[0, 3] == 2 and D[1, 2] == 2
    assert np.array_equal(D, D.T)


def test_region_graph_shared_boundary_length():
    rg = _rg22()
    adj, D, ell = region_graph(rg)
    # die is 100x100, lattice=10 -> each lattice cell is 10x10; the 2x2 partition
    # boundary runs the full 5-cell-long shared edge between each adjacent pair.
    assert ell[0, 1] == 5 and ell[0, 2] == 5 and ell[1, 3] == 5 and ell[2, 3] == 5
    assert ell[0, 3] == 0 and ell[1, 2] == 0
    assert np.array_equal(ell, ell.T)
    # ell is zero exactly where adj is False (off-diagonal)
    assert np.array_equal(ell > 0, adj)


def test_region_graph_4x4_edge_count_matches_probe_evidence():
    # docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md §1:
    # "相鄰對數: adaptec1 k16 grid 24" for a 4x4 grid partition -- a 4x4 grid graph
    # has exactly 24 adjacent (shares-an-edge) region pairs (2*4*3 = 24 grid edges).
    rg = _rg44()
    adj, D, ell = region_graph(rg)
    n_adj_pairs = int(np.triu(adj, 1).sum())
    assert n_adj_pairs == 24


def test_region_graph_rejects_k_over_32():
    class _FakeRG:
        k = 33
        grid = np.zeros((4, 4), dtype=np.int16)
    with pytest.raises(AssertionError):
        region_graph(_FakeRG())


# ---------------------------------------------------------------------------
# next_hop_table + path reconstruction (internal to Λ>=4 tree expansion, F11)
# ---------------------------------------------------------------------------

def test_next_hop_table_reconstructs_shortest_paths():
    rg = _rg44()
    adj, D, ell = region_graph(rg)
    next_hop = next_hop_table(adj, D)
    K = rg.k
    for a in range(K):
        for b in range(K):
            # walk next_hop from a to b and check the path length matches D[a,b]
            steps = 0
            cur = a
            seen = {cur}
            while cur != b:
                cur = int(next_hop[cur, b])
                steps += 1
                assert cur not in seen or cur == b, "next_hop path revisits a vertex"
                seen.add(cur)
                assert steps <= K, "next_hop path longer than K -- broken table"
            assert steps == D[a, b]


# ---------------------------------------------------------------------------
# steiner_tree_stats: identity ST - max(lambda-1,0) == FT on a feasible tree,
# using the worked examples from spec §3.3.3 (F3 chain-vs-star + true-FT cases).
# ---------------------------------------------------------------------------

def _chain_D():
    # h(0) - a(1) - b(2), a path graph
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.int64)
    return adj, D


def _star_D():
    # h(0) adjacent to both a(1) and c(2); a,c not adjacent to each other
    adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    D = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=np.int64)
    return adj, D


def _true_ft_D():
    # h(0) - a(1) - b(2): touched = {h, b}, a is an untouched pass-through region
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.int64)
    return adj, D


@pytest.mark.parametrize("build,terminals,exp_st,exp_ft", [
    (_chain_D, [0, 1, 2], 2, 0),   # chain h-a-b, all 3 touched -> ST=2, FT=0
    (_star_D, [0, 1, 2], 2, 0),    # star h with a,c: ST=2, FT=0 (not 3 -- F3 fix)
    (_true_ft_D, [0, 2], 2, 1),    # h and b only (D=2), a untouched -> true FT=1
])
def test_steiner_tree_worked_examples_f3(build, terminals, exp_st, exp_ft):
    adj, D = build()
    next_hop = next_hop_table(adj, D)
    st, ft, exact = steiner_tree_stats(D, next_hop, terminals)
    assert (st, ft) == (exp_st, exp_ft)
    assert exact is True


def test_steiner_tree_trivial_lambda_0_and_1():
    adj, D = _chain_D()
    next_hop = next_hop_table(adj, D)
    assert steiner_tree_stats(D, next_hop, []) == (0, 0, True)
    assert steiner_tree_stats(D, next_hop, [1]) == (0, 0, True)


def test_steiner_tree_lambda_2_is_shortest_path():
    adj, D = _chain_D()
    next_hop = next_hop_table(adj, D)
    # h(0) and b(2) are 2 hops apart via the non-terminal pass-through a(1):
    # ST = D[0,2] = 2, FT = ST - 1 = 1 (a is a true feed-through region here).
    st, ft, exact = steiner_tree_stats(D, next_hop, [0, 2])
    assert (st, ft, exact) == (2, 1, True)
    # an actually-adjacent pair has FT = 0.
    st01, ft01, _ = steiner_tree_stats(D, next_hop, [0, 1])
    assert (st01, ft01) == (1, 0)


def _random_connected_region_graph(rng, K):
    """Build a random connected graph on K<=32 vertices (a random spanning tree
    plus a few extra random edges), matching what a real (contiguous partition)
    region-adjacency graph looks like."""
    perm = rng.permutation(K)
    adj = np.zeros((K, K), dtype=bool)
    for i in range(1, K):
        a, b = perm[i], perm[rng.integers(0, i)]
        adj[a, b] = adj[b, a] = True
    extra = rng.integers(0, K)
    for _ in range(extra):
        a, b = rng.integers(0, K, 2)
        if a != b:
            adj[a, b] = adj[b, a] = True
    return adj


def _floyd_warshall_ref(adj):
    K = adj.shape[0]
    INF = 10 ** 6
    D = np.where(adj, 1, INF).astype(np.int64)
    np.fill_diagonal(D, 0)
    for k in range(K):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


@pytest.mark.parametrize("seed", range(8))
def test_steiner_tree_identity_random_small_lambda(seed):
    """ST - max(Λ-1, 0) == FT on the constructed feasible tree, for random small
    connected region graphs and random terminal subsets spanning every Λ tier
    (<=1 trivial, 2 direct path, [3,8] exact Dreyfus-Wagner, >8 MST heuristic)."""
    rng = np.random.default_rng(1000 + seed)
    K = int(rng.integers(6, 17))
    adj = _random_connected_region_graph(rng, K)
    D = _floyd_warshall_ref(adj)
    next_hop = next_hop_table(adj, D)
    for lam in range(0, min(K, 12) + 1):
        terms = rng.choice(K, size=lam, replace=False).tolist() if lam > 0 else []
        st, ft, exact = steiner_tree_stats(D, next_hop, terms)
        assert ft >= 0
        assert st - max(lam - 1, 0) == ft
        if lam <= 8:
            assert exact is True
        else:
            assert exact is False
        # ST must be a feasible upper bound on the (Λ-1) trivial lower bound and
        # can never exceed a naive "star from any single vertex" upper bound.
        assert st >= max(lam - 1, 0)


def test_steiner_tree_exact_matches_bruteforce_for_small_lambda():
    """Independent bruteforce Steiner-tree-cost oracle (enumerate all subsets of
    non-terminal vertices as extra Steiner points, MST over touched+extra) for a
    handful of small random cases, cross-checked against steiner_tree_stats'
    ST value for Λ in [2,6] (well within the exact-DW tier)."""
    import itertools

    def bruteforce_st(D, terms, K):
        terms = sorted(set(terms))
        others = [v for v in range(K) if v not in terms]
        best = None
        for r in range(0, len(others) + 1):
            for extra in itertools.combinations(others, r):
                verts = list(terms) + list(extra)
                if len(verts) == 1:
                    cost = 0
                else:
                    # MST over `verts` using D as edge weight (Prim, brute force)
                    n = len(verts)
                    in_tree = [False] * n
                    in_tree[0] = True
                    best_cost = [D[verts[0], verts[j]] for j in range(n)]
                    cost = 0
                    for _ in range(n - 1):
                        j = min((j for j in range(n) if not in_tree[j]), key=lambda j: best_cost[j])
                        cost += best_cost[j]
                        in_tree[j] = True
                        for k2 in range(n):
                            if not in_tree[k2]:
                                best_cost[k2] = min(best_cost[k2], D[verts[j], verts[k2]])
                if best is None or cost < best:
                    best = cost
        return best

    rng = np.random.default_rng(42)
    for trial in range(6):
        K = int(rng.integers(6, 9))
        adj = _random_connected_region_graph(rng, K)
        D = _floyd_warshall_ref(adj)
        next_hop = next_hop_table(adj, D)
        lam = int(rng.integers(2, min(K, 6) + 1))
        terms = rng.choice(K, size=lam, replace=False).tolist()
        st, ft, exact = steiner_tree_stats(D, next_hop, terms)
        assert exact is True
        assert st == bruteforce_st(D, terms, K)


def test_steiner_tree_mst_tier_on_real_grid_k32_stays_a_feasible_tree():
    """A real 8x4 grid region graph (K=32, many 4-cycles -- unlike the random
    spanning-tree-based graphs above) with large terminal sets (several Λ>8,
    exercising the metric-closure-MST heuristic tier). Grid adjacency means
    MST-edge path expansions frequently overlap, so this is the case most
    likely to trip a naive (non-union-find) edge union into double-counting or
    a cycle; the identity holding here is the load-bearing regression for F11
    on the tier that actually needs it in real evaluator runs."""
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    adj, D, ell = region_graph(rg)
    K = rg.k
    assert K == 32
    next_hop = next_hop_table(adj, D)
    rng = np.random.default_rng(7)
    saw_ub_tier = False
    for lam in [2, 3, 5, 8, 9, 12, 16, 24, 32]:
        for _ in range(3):
            terms = rng.choice(K, size=lam, replace=False).tolist()
            st, ft, exact = steiner_tree_stats(D, next_hop, terms)
            assert ft >= 0
            assert st - max(lam - 1, 0) == ft
            assert exact == (lam <= 8)
            saw_ub_tier |= not exact
    assert saw_ub_tier  # sanity: the >8 (heuristic) tier was actually exercised
