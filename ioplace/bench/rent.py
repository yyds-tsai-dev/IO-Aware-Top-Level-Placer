"""M4 T5 Rent measurement (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 3.3):
Landman-Russo recursive bisection. At level 0 the whole netlist is one
block; each level bisects every current block into two (mtkahypar k=2,
KM1 objective), recording the level's average block size B and average
*external terminal* count T (nets with >=1 pin inside the block and >=1
pin outside it -- i.e. the block's boundary pin count once it is
contracted to a supercell). `log T = log t + p * log B` is then fit by OLS
over the levels whose B falls in `[b_lo, b_hi]` (design draft default
`1e3 <= B <= 1e6`; overridable -- this module's own toy tests need a much
smaller window). `p` is the Rent exponent.

Partitioning backend: `mtkahypar==1.6.2` (see `docs/dev-env.md`'s "Task 8"
section for the confirmed install + API). If the import fails, this module
falls back to geometric median bisection (alternating x/y axis by level,
using `nl.node_x`/`nl.node_y`) -- sec 3.3's explicitly named fallback, with
lower confidence (a naive geometric cut is not a min-cut, so the resulting
Rent exponent is an upper bound on the "true" min-cut Rent exponent, not an
estimate of it). The backend actually used is always recorded in the
output so a caller/report cannot silently treat a geometric-fallback run as
an mtkahypar one.
"""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class RentLevel:
    level: int
    n_blocks: int
    block_sizes: list
    terminals: list

    @property
    def avg_block_size(self):
        return float(np.mean(self.block_sizes))

    @property
    def avg_terminals(self):
        return float(np.mean(self.terminals))


@dataclass
class RentResult:
    p: float
    p_ci_lo: float
    p_ci_hi: float
    log_t: float
    backend: str
    levels_used: list  # level indices that fed the regression
    levels: list  # all RentLevel objects, sc.z. debugging/reporting
    n_bootstrap: int


def node_to_nets_csr(nl):
    """Reverse (node -> incident net ids) index, CSR-style, built once and
    reused across an entire recursive bisection instead of re-scanning
    every net in the netlist on every one of the O(2**level) calls (the
    obvious-but-quadratic alternative -- a block deep in the recursion
    only touches a tiny fraction of nl.num_nets, and this index lets
    `_induced_hyperedges` only look at that fraction)."""
    order = np.argsort(nl.pin2node, kind="stable")
    sorted_nodes = nl.pin2node[order]
    sorted_nets = nl.pin2net[order]
    counts = np.bincount(sorted_nodes, minlength=nl.num_physical)
    starts = np.zeros(nl.num_physical + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    return starts, sorted_nets


def _induced_hyperedges(node_ids, nl, n2n_starts, n2n_nets):
    """Local (0..n-1), deduplicated hyperedges for the induced sub-hypergraph
    on `node_ids` (a numpy array of global node indices). Nets with fewer
    than 2 pins inside the block after restriction are dropped (they carry
    no cut information for *this* bisection). Only nets touching at least
    one node of `node_ids` are ever examined (via `n2n_starts`/`n2n_nets`
    from `node_to_nets_csr`), not the whole netlist."""
    local_of = {int(v): i for i, v in enumerate(node_ids)}
    cand_nets = set()
    for v in node_ids:
        s, e = n2n_starts[v], n2n_starts[v + 1]
        cand_nets.update(n2n_nets[s:e].tolist())
    edges = []
    for net in cand_nets:
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        pins = nl.pin2node[nl.flat_net2pin[s:e]]
        local = sorted({local_of[int(v)] for v in pins if int(v) in local_of})
        if len(local) >= 2:
            edges.append(local)
    return edges


def bisect_mtkahypar(node_ids, nl, seed, threads=4, epsilon=0.03, n2n_csr=None):
    """n2n_csr: optional precomputed `node_to_nets_csr(nl)` result (pass it
    when calling this repeatedly, e.g. from `recursive_bisection`, to avoid
    rebuilding the reverse index every call)."""
    import mtkahypar
    if len(node_ids) < 2:
        return node_ids, np.array([], dtype=node_ids.dtype)
    n2n_starts, n2n_nets = n2n_csr if n2n_csr is not None else node_to_nets_csr(nl)
    edges = _induced_hyperedges(node_ids, nl, n2n_starts, n2n_nets)
    if not edges:
        half = len(node_ids) // 2
        return node_ids[:half], node_ids[half:]
    mtk = mtkahypar.initialize(threads, print_warnings=False)
    ctx = mtk.context_from_preset(mtkahypar.PresetType.DEFAULT)
    ctx.set_partitioning_parameters(2, epsilon, mtkahypar.Objective.KM1)
    mtkahypar.set_seed(seed)
    hg = mtk.create_hypergraph(ctx, len(node_ids), len(edges), edges)
    part = hg.partition(ctx)
    assign = np.array([part.block_id(v) for v in range(len(node_ids))])
    return node_ids[assign == 0], node_ids[assign == 1]


def bisect_geometric(node_ids, nl, axis):
    """Median split on nl.node_x (axis=0) or nl.node_y (axis=1). Not a
    min-cut -- see module docstring."""
    if len(node_ids) < 2:
        return node_ids, np.array([], dtype=node_ids.dtype)
    coord = (nl.node_x if axis == 0 else nl.node_y)[node_ids]
    order = np.argsort(coord, kind="stable")
    half = len(node_ids) // 2
    sorted_ids = node_ids[order]
    return sorted_ids[:half], sorted_ids[half:]


def _resolve_backend(backend):
    if backend == "auto":
        try:
            import mtkahypar  # noqa: F401
            return "mtkahypar"
        except ImportError:
            return "geometric"
    if backend not in ("mtkahypar", "geometric"):
        raise ValueError(f"unknown backend {backend!r}")
    return backend


def recursive_bisection(nl, max_level, seed=0, backend="auto", threads=4, root_node_ids=None):
    """Returns (levels: list[list[np.ndarray node ids]], backend_used).
    levels[0] is [root_node_ids] (the whole netlist, 1 block);
    levels[l] has 2**l blocks."""
    backend = _resolve_backend(backend)
    root = np.arange(nl.num_physical, dtype=np.int64) if root_node_ids is None else np.asarray(root_node_ids)
    n2n_csr = node_to_nets_csr(nl) if backend == "mtkahypar" else None
    levels = [[root]]
    for level in range(1, max_level + 1):
        prev = levels[-1]
        cur = []
        for b_idx, block in enumerate(prev):
            if backend == "mtkahypar":
                left, right = bisect_mtkahypar(block, nl, seed=seed + level * 1000 + b_idx, threads=threads,
                                                n2n_csr=n2n_csr)
            else:
                left, right = bisect_geometric(block, nl, axis=level % 2)
            cur.append(left)
            cur.append(right)
        levels.append(cur)
    return levels, backend


def block_terminal_counts(blocks, nl):
    """External terminal count per block: a net contributes 1 to every
    block it has >=1 pin in, but only for nets that touch more than one
    block (a net entirely inside one block is not a boundary terminal)."""
    node_to_block = np.full(nl.num_physical, -1, dtype=np.int64)
    for b_idx, ids in enumerate(blocks):
        node_to_block[ids] = b_idx
    terms = np.zeros(len(blocks), dtype=np.int64)
    for net in range(nl.num_nets):
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        pins = nl.pin2node[nl.flat_net2pin[s:e]]
        touched = np.unique(node_to_block[pins])
        touched = touched[touched >= 0]
        if len(touched) > 1:
            for b in touched:
                terms[b] += 1
    return terms


def _fit_p(levels_bt):
    """OLS fit of log(T) = log(t) + p*log(B) over [(B, T), ...] level
    averages. Returns (p, log_t)."""
    log_b = np.log(np.array([b for b, _ in levels_bt]))
    log_t = np.log(np.array([t for _, t in levels_bt]))
    p, c = np.polyfit(log_b, log_t, 1)
    return float(p), float(c)


def measure_rent(nl, max_level=None, b_lo=1e3, b_hi=1e6, seed=0, backend="auto",
                  threads=4, n_bootstrap=200, root_node_ids=None, rng=None):
    """Landman-Russo recursive-bisection Rent exponent. `max_level`
    defaults to enough levels that the smallest block averages < b_lo (so
    the window naturally has content); pass it explicitly for small/toy
    netlists where the default would recurse past single-node blocks."""
    n_root = nl.num_physical if root_node_ids is None else len(root_node_ids)
    if max_level is None:
        max_level = max(1, math.ceil(math.log2(max(1, n_root / b_lo))))

    levels, backend_used = recursive_bisection(
        nl, max_level, seed=seed, backend=backend, threads=threads, root_node_ids=root_node_ids)

    rent_levels = []
    for l, blocks in enumerate(levels):
        sizes = [len(b) for b in blocks]
        terms = block_terminal_counts(blocks, nl) if l > 0 else np.array([0])
        # level 0 (the whole netlist as one block) has, by definition, 0
        # external terminals -- it has no "outside".
        rent_levels.append(RentLevel(level=l, n_blocks=len(blocks), block_sizes=sizes,
                                      terminals=list(map(int, terms))))

    usable = [(rl.level, rl.avg_block_size, rl.avg_terminals) for rl in rent_levels
              if rl.level >= 1 and rl.avg_terminals > 0 and b_lo <= rl.avg_block_size <= b_hi]
    if len(usable) < 2:
        return RentResult(p=float("nan"), p_ci_lo=float("nan"), p_ci_hi=float("nan"),
                           log_t=float("nan"), backend=backend_used,
                           levels_used=[u[0] for u in usable], levels=rent_levels, n_bootstrap=0)

    levels_used = [u[0] for u in usable]
    p, log_t = _fit_p([(u[1], u[2]) for u in usable])

    rng = np.random.default_rng(seed) if rng is None else rng
    boot_ps = []
    for _ in range(n_bootstrap):
        resampled = []
        for l in levels_used:
            rl = rent_levels[l]
            n = len(rl.block_sizes)
            idx = rng.integers(0, n, size=n)
            sizes = np.array(rl.block_sizes)[idx]
            terms = np.array(rl.terminals)[idx]
            if terms.sum() == 0:
                resampled = None
                break
            resampled.append((float(sizes.mean()), float(max(terms.mean(), 1e-9))))
        if resampled is None:
            continue
        try:
            bp, _ = _fit_p(resampled)
            boot_ps.append(bp)
        except Exception:
            continue
    if boot_ps:
        p_lo, p_hi = np.quantile(boot_ps, [0.025, 0.975])
    else:
        p_lo = p_hi = p

    return RentResult(p=p, p_ci_lo=float(p_lo), p_ci_hi=float(p_hi), log_t=log_t,
                       backend=backend_used, levels_used=levels_used, levels=rent_levels,
                       n_bootstrap=len(boot_ps))
