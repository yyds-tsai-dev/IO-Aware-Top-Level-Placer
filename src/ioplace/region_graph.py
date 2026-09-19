"""Region-adjacency graph G_R and Steiner-tree machinery for M3 T1 (evaluator
region-graph extension).

See docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md,
T1 (§2.5) and §2.1:

    route(e)     := G_R 上連接 touched(e) 的最小 Steiner tree T_e(邊權 1 = 一次 boundary crossing)
    crossings(e) := |E(T_e)| = ST_e
    FT(e)        := |V(T_e) \\ touched(e)| = (ST_e + 1) - Λ_e        # 對一棵實際的樹成立

`region_graph(rg)` builds G_R (K<=32 constant tables: adjacency, all-pairs hop
distance via Floyd-Warshall, shared-boundary length). `next_hop_table` and
`steiner_tree_stats` are additional public helpers (beyond the literal 3-tuple
`region_graph` deliverable) needed to satisfy F11 of the Opus adversarial review
(docs/reviews/2026-08-13-m3-draft-v1-adversarial-opus.md):

    Λ>=4 的解必須展開成 G_R 上的實際子樹,ST 取邊數、ft_rg 取非 terminal 頂點數
    (與 evaluator_ref.py 的 distinct 語意同構),使 FT = ST + 1 - Λ 每條 net 精確成立。

`steiner_tree_stats` always derives ST/FT by literally constructing a tree (via
union-find deduplication of expanded shortest-path edges) and counting it --
never by trusting a declared DP/MST weight -- so the identity holds exactly on
a feasible tree regardless of tie-breaks in path reconstruction or subset-DP
backtracking. This is also what guarantees evaluator_ref (CPU, per-net loop)
and evaluator_gpu (bulk closed-form for Λ<=3, this module's shared routine for
the rare Λ>=4 nets) agree bit-for-bit on the new fields: both tiers ultimately
route the Λ>=4 nets through this exact same function.

Tiering (see T1 §2.5 and the M3 evidence in §2.2, Λ>=4 is <=0.24% of active
nets on adaptec1 k16 -- a Python-loop reconstruction cost for that tier is
negligible relative to the bulk Λ<=3 closed-form path):
  - Λ<=1: trivial, ST=FT=0 (no tree needed).
  - Λ=2:  ST = D[t0,t1] (a single shortest path IS the minimum Steiner tree).
  - Λ in [3,8]: exact Dreyfus-Wagner subset-DP over the metric closure (D),
    with full backtracking to reconstruct an actual optimal tree.
  - Λ>8: metric-closure MST heuristic (2-approximation), also expanded into an
    actual tree. `exact=False` for this tier (T1 hard requirement #1: report
    st_ub/n_nets_ub-style exactness alongside the value).
"""
import numpy as np


def region_graph(rg):
    """Build the region-adjacency graph G_R from a RegionGrid.

    Returns (adj, D, ell):
      adj: (K,K) bool, symmetric, adj[a,b] True iff regions a,b share >=1
           lattice-grid edge (G_R's edge set).
      D:   (K,K) uint8, all-pairs hop distance on G_R (unit edge weight), via
           Floyd-Warshall. D[a,a] = 0.
      ell: (K,K) int64, symmetric, shared-boundary length = count of shared
           lattice-grid edges between each region pair (0 where not adjacent).

    K = rg.k must be <= 32, matching the M3 experiment matrix (K in {8,16,32})
    and the K<=32 contract the paired-int8 dtype work in evaluator_gpu.py
    depends on (see docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md
    §4.2: `_pow2_k` uses signed int64, so K=64 is out of contract).
    """
    K = rg.k
    assert K <= 32, f"region_graph requires rg.k <= 32, got {K}"
    g = rg.grid.astype(np.int64)
    ell = np.zeros((K, K), dtype=np.int64)
    a, b = g[:, :-1], g[:, 1:]
    m = a != b
    np.add.at(ell, (a[m], b[m]), 1)
    np.add.at(ell, (b[m], a[m]), 1)
    a, b = g[:-1, :], g[1:, :]
    m = a != b
    np.add.at(ell, (a[m], b[m]), 1)
    np.add.at(ell, (b[m], a[m]), 1)
    adj = ell > 0
    D = _floyd_warshall(adj)
    assert D.max() < 255, "region graph diameter too large for a uint8 D table"
    return adj, D.astype(np.uint8), ell


def _floyd_warshall(adj):
    """adj: (K,K) bool/int 0/1 symmetric adjacency -> D (K,K) int64 hop distances."""
    K = adj.shape[0]
    INF = np.iinfo(np.int64).max // 4
    D = np.where(adj, 1, INF).astype(np.int64)
    np.fill_diagonal(D, 0)
    for k in range(K):
        np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :], out=D)
    return D


def next_hop_table(adj, D):
    """(K,K) int64 next-hop table for canonical shortest-path reconstruction on
    G_R: next_hop[a,b] is the neighbour of `a` on a deterministic shortest
    a->b path (next_hop[a,a] = a). Deterministic tie-break: for each (a,b),
    scan a's neighbours in increasing index order and take the first that lies
    on a shortest path.

    This is a separate, simpler tie-break than the S4e `path_mask` convention
    in spec §3.3.3 (lexicographically-smallest region-id sequence) -- that rule
    only matters for S4e's (Phase B, out of scope here) gradient properties.
    All this needs is *a* fixed, deterministic rule so CPU (evaluator_ref) and
    GPU (evaluator_gpu) reconstruct identical paths for the Λ>=4 tree
    expansion (F11), which any deterministic rule provides.
    """
    K = adj.shape[0]
    nh = -np.ones((K, K), dtype=np.int64)
    for v in range(K):
        nh[v, v] = v
    neighbours = [np.nonzero(adj[a])[0].tolist() for a in range(K)]
    Di = D.astype(np.int64)
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            for c in neighbours[a]:
                if Di[c, b] == Di[a, b] - 1:
                    nh[a, b] = c
                    break
    return nh


def _path_edges(next_hop, a, b):
    """Canonical simple-path edges from a to b (inclusive), as (lo,hi) pairs."""
    edges = []
    cur = a
    while cur != b:
        nxt = int(next_hop[cur, b])
        edges.append((min(cur, nxt), max(cur, nxt)))
        cur = nxt
    return edges


class _UnionFind:
    """Minimal union-find over int vertex ids, used to keep tree expansion
    cycle-safe: expanded shortest-path segments for different Steiner/MST
    edges can share sub-paths, so a naive union of edge sets is not guaranteed
    to be a tree. Adding an edge only when it connects two different
    components guarantees the result is a forest (here: a single tree, since
    every expansion step is reachable from the growing component)."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def _grow_tree(next_hop, primitive_pairs):
    """Expand each (a,b) in `primitive_pairs` into its canonical shortest-path
    edges and add them to a shared union-find forest, in iteration order,
    skipping any expanded edge whose endpoints are already connected (F11:
    this is what guarantees the result is an actual feasible tree, not just a
    connected subgraph with redundant edges). Returns (edges, verts) of the
    resulting tree."""
    uf = _UnionFind()
    edges = set()
    verts = set()
    for (a, b) in primitive_pairs:
        verts.add(a)
        verts.add(b)
        for (u, v) in _path_edges(next_hop, a, b):
            verts.add(u)
            verts.add(v)
            if uf.union(u, v):
                edges.add((u, v))
    return edges, verts


def _mst_pairs_over(D, terms):
    """Prim's algorithm MST over the complete graph on `terms` weighted by D
    (the metric closure). Returns the list of (terms[i], terms[j]) MST edges
    as region-id pairs. Deterministic tie-break: lowest candidate index wins,
    mirroring the first-occurrence-wins argmin convention used elsewhere in
    this codebase (evaluator_ref.net_mst_edges / evaluator_gpu._prim_batch)."""
    n = len(terms)
    if n < 2:
        return []
    in_tree = [False] * n
    in_tree[0] = True
    best_cost = [int(D[terms[0], terms[j]]) for j in range(n)]
    best_from = [0] * n
    pairs = []
    for _ in range(n - 1):
        j = min((j for j in range(n) if not in_tree[j]), key=lambda j: (best_cost[j], j))
        pairs.append((terms[best_from[j]], terms[j]))
        in_tree[j] = True
        for k in range(n):
            if not in_tree[k]:
                c = int(D[terms[j], terms[k]])
                if c < best_cost[k]:
                    best_cost[k] = c
                    best_from[k] = j
    return pairs


def _dw_pairs(D, K, terms):
    """Exact Dreyfus-Wagner Steiner-tree DP over terminal subsets (len(terms)
    in [3,8]), with full backtracking. Returns the sequence of primitive
    (a,b) region-id pairs whose canonical shortest-path expansions, unioned
    via _grow_tree, reconstruct an optimal Steiner tree.

    Standard two-phase recurrence on dp[mask][v] (mask: subset of local
    terminal indices 0..lam-1, v: any of the K region ids):
      base:  dp[{i}][v]  = D[terms[i], v]
      merge: dp[mask][v] = min over proper nonempty submasks s of mask of
                            dp[s][v] + dp[mask\\s][v]
      relax: dp[mask][v] = min(dp[mask][v], min_u dp[mask][u] + D[u,v])
    A single relax pass suffices because D already satisfies the triangle
    inequality (it is itself all-pairs-shortest-path): iterating again cannot
    improve any value (standard fixed-point argument), and for the u*
    achieving the one-pass relaxed[v], relaxed[u*] + D[u*,v] == relaxed[v]
    exactly (both directions of the triangle-inequality argument), so
    backtracking with the *post-relax* dp table still finds a valid witness.
    """
    lam = len(terms)
    full = (1 << lam) - 1
    INF = 1 << 30
    dp = [[INF] * K for _ in range(1 << lam)]
    for i in range(lam):
        row = dp[1 << i]
        for v in range(K):
            row[v] = int(D[terms[i], v])
    for mask in range(1, 1 << lam):
        if bin(mask).count("1") < 2:
            continue
        row = dp[mask]
        sub = (mask - 1) & mask
        while sub > 0:
            other = mask ^ sub
            if sub >= other:  # each unordered {sub,other} split visited once
                d_sub, d_other = dp[sub], dp[other]
                for v in range(K):
                    c = d_sub[v] + d_other[v]
                    if c < row[v]:
                        row[v] = c
            sub = (sub - 1) & mask
        relaxed = row[:]
        for v in range(K):
            best = relaxed[v]
            for u in range(K):
                if u == v:
                    continue
                c = row[u] + int(D[u, v])
                if c < best:
                    best = c
            relaxed[v] = best
        dp[mask] = relaxed
    best_v = min(range(K), key=lambda v: dp[full][v])

    pairs = []

    def backtrack(mask, v):
        if bin(mask).count("1") == 1:
            i = mask.bit_length() - 1
            pairs.append((terms[i], v))
            return
        cur = dp[mask][v]
        sub = (mask - 1) & mask
        while sub > 0:
            other = mask ^ sub
            if dp[sub][v] + dp[other][v] == cur:
                backtrack(sub, v)
                backtrack(other, v)
                return
            sub = (sub - 1) & mask
        for u in range(K):
            if u != v and dp[mask][u] + int(D[u, v]) == cur:
                pairs.append((u, v))
                backtrack(mask, u)
                return
        # cur was already the merge-phase value with no strict relax
        # improvement over some other u; unreachable given the checks above
        # always find a witness (see module docstring), but guard defensively.
        raise AssertionError("Dreyfus-Wagner backtrack found no witness split")

    backtrack(full, best_v)
    return pairs


def steiner_tree_stats(D, next_hop, terminals):
    """Exact/heuristic Steiner tree on G_R (metric D, canonical shortest paths
    next_hop) spanning `terminals` (an iterable of region ids; duplicates
    ignored). Returns (ST, FT, exact):
      ST: edge count of an actually-constructed feasible tree.
      FT: that tree's non-terminal vertex count (== ST + 1 - Λ by
          construction, for Λ >= 1; both are 0 for Λ == 0).
      exact: whether ST is provably the true minimum Steiner-tree cost on G_R
             (Λ<=8, Dreyfus-Wagner-exact) or only a feasible upper bound
             (Λ>8, metric-closure MST heuristic).
    """
    terms = sorted(set(int(t) for t in terminals))
    lam = len(terms)
    K = D.shape[0]
    if lam <= 1:
        return 0, 0, True
    if lam == 2:
        st = int(D[terms[0], terms[1]])
        return st, st - 1, True
    if lam <= 8:
        pairs = _dw_pairs(D, K, terms)
        exact = True
    else:
        pairs = _mst_pairs_over(D, terms)
        exact = False
    edges, verts = _grow_tree(next_hop, pairs)
    st = len(edges)
    ft = len(verts) - lam
    return st, ft, exact
