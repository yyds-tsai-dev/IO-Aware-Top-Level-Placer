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

**M4 T6 finding (2026-08-15):** `ioplace.netlist.load_netlist` (DREAMPlace's
own `PlaceDB.read`) cannot load this project's Bookshelf exports at all --
independent of `export_bookshelf.py`'s backslash-escape/zero-degree-net/aux
fixes, `PlaceDB.read_pl` (`$DP/install/dreamplace/PlaceDB.py:1052`) matches
node names with a hardcoded `re.search(r"(\\w+)\\s+...", line)` -- `\\w` is
`[A-Za-z0-9_]` only, so it cannot span this project's hierarchical names'
`.`/`/`/`[`/`]` characters; `re.search` (not `re.match`) then silently
matches just the trailing word-run after the name's last non-word
character, and looks that (wrong, truncated) name up in `node_name2id_map`
-> `KeyError`. Unlike the escaping issue, this is not fixable by a lossless
rewrite of the exported files (stripping `.`/`/`/`[`/`]` from names is not
injective -- distinct instances could collide onto the same stripped name).
`load_bookshelf_netlist` below reads `.nodes`/`.pl`/`.nets` directly with a
pure-Python parser instead, producing the same `ioplace.netlist.Netlist`
shape `measure_rent`'s bisection functions need (`pin2node`/`pin2net`/
`flat_net2pin(_start)`/`node_x`/`node_y`/`num_physical` -- verified none of
`bisect_mtkahypar`/`bisect_geometric`/`block_terminal_counts`/
`node_to_nets_csr` touch anything else on `nl`), so H1/H6's Rent
measurement does not depend on `PlaceDB.read` working at all. `V1`
("DREAMPlace 可讀") is a diagnostic-only metric (adjudication doc sec 4:
"診斷不判定:H5、H5′、V0/V1") precisely because it tests exactly this path
-- it is expected to (and does) surface this incompatibility, not gate T6.
"""
from dataclasses import dataclass, field
import math
import time

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
    runtime: dict = None


def load_bookshelf_netlist(prefix):
    """Pure-Python `.nodes`/`.pl`/`.nets` reader -> `ioplace.netlist.Netlist`
    (see module docstring's M4 T6 finding for why this exists instead of
    `ioplace.netlist.load_netlist`). Three single streaming passes (each
    file read once, front to back); `node_size_x/y`, `pin_offset_x/y`,
    `num_terminal_NIs` are left as zeros (unused by anything in this
    module -- `bisect_mtkahypar`/`bisect_geometric`/`block_terminal_counts`/
    `node_to_nets_csr` never touch them); `num_movable`/`num_terminals` are
    filled from the `.nodes` body's own `terminal`/`terminal_NI` markers for
    informational completeness only."""
    from ioplace.bench import tile_bookshelf as tb
    from ioplace.netlist import Netlist

    paths = tb.read_aux(prefix + ".aux")

    name2id = {}
    n_terminals = 0
    with open(paths["nodes"]) as f:
        started = False
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumTerminals"):
                    started = True
                continue
            if not s:
                continue
            parts = s.split()
            name2id[parts[0]] = len(name2id)
            if len(parts) >= 4 and parts[3] in ("terminal", "terminal_NI"):
                n_terminals += 1
    n_physical = len(name2id)

    node_x = np.zeros(n_physical, dtype=np.float64)
    node_y = np.zeros(n_physical, dtype=np.float64)
    with open(paths["pl"]) as f:
        next(f)
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            nid = name2id[parts[0]]
            node_x[nid] = float(parts[1])
            node_y[nid] = float(parts[2])

    pin2node_list, pin2net_list = [], []
    net2pin_start = [0]
    net_id = -1
    with open(paths["nets"]) as f:
        started = False
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumPins"):
                    started = True
                continue
            if not s:
                continue
            if s.startswith("NetDegree"):
                net_id += 1
                net2pin_start.append(net2pin_start[-1])
                continue
            parts = s.split()
            nid = name2id[parts[0]]
            pin2node_list.append(nid)
            pin2net_list.append(net_id)
            net2pin_start[-1] += 1

    pin2node = np.asarray(pin2node_list, dtype=np.int32)
    pin2net = np.asarray(pin2net_list, dtype=np.int32)
    n_pins = len(pin2node)
    flat_net2pin = np.arange(n_pins, dtype=np.int64)  # pins are already net-grouped, in file order
    flat_net2pin_start = np.asarray(net2pin_start, dtype=np.int64)

    return Netlist(
        node_x=node_x, node_y=node_y,
        node_size_x=np.zeros(n_physical), node_size_y=np.zeros(n_physical),
        num_movable=n_physical - n_terminals, num_terminals=n_terminals, num_terminal_NIs=0,
        pin_offset_x=np.zeros(n_pins), pin_offset_y=np.zeros(n_pins),
        pin2node=pin2node, pin2net=pin2net,
        flat_net2pin=flat_net2pin, flat_net2pin_start=flat_net2pin_start,
        xl=float(node_x.min()) if n_physical else 0.0, yl=float(node_y.min()) if n_physical else 0.0,
        xh=float(node_x.max()) if n_physical else 0.0, yh=float(node_y.max()) if n_physical else 0.0,
    )


def load_def_netlist(def_path):
    """Streams `def_path`'s `COMPONENTS`/`NETS` sections directly into an
    `ioplace.netlist.Netlist` -- the DEF-side counterpart to
    `load_bookshelf_netlist`, for measuring Rent on the real cluster
    (`mempool_cluster.def`, 11.31M cells / 12.7M nets / 43.9M pins) without
    a `PlaceDB`-based Bookshelf export+reread round trip (which would cost
    comparable host RAM/time to `export_bookshelf.py`'s `mempool_group` run,
    just ~3.7x larger, for a file this function never needs to touch a
    second time). Two streaming passes over the same file (COMPONENTS then
    NETS, in one open) -- same `NETS`-record state machine as
    `hierarchy_gate.scan_def`'s Bug-B fix (stop scanning pin tuples at a
    net record's first `+`; exclude `( PIN ... )`/`( * ... )` tuples),
    independently written here as a further cross-check. Names are kept
    exactly as written in the DEF (backslash-escaped) -- irrelevant here
    since this never goes through Bookshelf's grammar, only a Python dict
    lookup."""
    from array import array
    from ioplace.netlist import Netlist

    name2id = {}
    geom_x = array("d")
    geom_y = array("d")
    section = "pre"
    t0 = time.time()
    with open(def_path, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("COMPONENTS "):
                    section = "components"
                continue
            if section == "components":
                if line.startswith("END COMPONENTS"):
                    section = "between"
                    continue
                if not line.startswith("- "):
                    continue
                toks = line.split()
                name = toks[1]
                nid = name2id.get(name)
                if nid is None:
                    nid = name2id[name] = len(name2id)
                x = y = 0.0
                for i in range(3, len(toks)):
                    if toks[i] in ("FIXED", "COVER", "PLACED", "UNPLACED"):
                        if i + 3 < len(toks) and toks[i + 1] == "(":
                            x, y = float(toks[i + 2]), float(toks[i + 3])
                        break
                geom_x.append(x)
                geom_y.append(y)
                continue
            if section == "between":
                if line.startswith("NETS "):
                    section = "nets"
                continue
            if line.startswith("END NETS"):
                break
            # section == "nets", handled in the second pass below
            continue
    t_components = time.time() - t0
    n_physical = len(name2id)
    node_x = np.frombuffer(geom_x, dtype=np.float64)
    node_y = np.frombuffer(geom_y, dtype=np.float64)

    pin2node_list, pin2net_list = [], []
    net2pin_start = [0]
    net_id = -1
    cur = None
    in_pins = True
    section = "pre"
    t1 = time.time()
    with open(def_path, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("NETS "):
                    section = "nets"
                continue
            if line.startswith("END NETS"):
                break
            if line.startswith("- "):
                net_id += 1
                net2pin_start.append(net2pin_start[-1])
                in_pins = True
            if not in_pins:
                continue
            if "(" not in line and "+" not in line:
                continue
            toks = line.split()
            i, n = 0, len(toks)
            while i < n:
                t = toks[i]
                if t == "+":
                    in_pins = False
                    break
                if t == "(" and i + 3 < n and toks[i + 3] == ")":
                    inst = toks[i + 1]
                    if inst not in ("PIN", "*"):
                        nid = name2id.get(inst)
                        if nid is not None:
                            pin2node_list.append(nid)
                            pin2net_list.append(net_id)
                            net2pin_start[-1] += 1
                    i += 4
                else:
                    i += 1
    t_nets = time.time() - t1

    pin2node = np.asarray(pin2node_list, dtype=np.int32)
    pin2net = np.asarray(pin2net_list, dtype=np.int32)
    n_pins = len(pin2node)
    flat_net2pin_start = np.asarray(net2pin_start, dtype=np.int64)

    nl = Netlist(
        node_x=node_x, node_y=node_y,
        node_size_x=np.zeros(n_physical), node_size_y=np.zeros(n_physical),
        num_movable=n_physical, num_terminals=0, num_terminal_NIs=0,
        pin_offset_x=np.zeros(n_pins), pin_offset_y=np.zeros(n_pins),
        pin2node=pin2node, pin2net=pin2net,
        flat_net2pin=np.arange(n_pins, dtype=np.int64), flat_net2pin_start=flat_net2pin_start,
        xl=float(node_x.min()) if n_physical else 0.0, yl=float(node_y.min()) if n_physical else 0.0,
        xh=float(node_x.max()) if n_physical else 0.0, yh=float(node_y.max()) if n_physical else 0.0,
    )
    return nl, {"n_physical": n_physical, "n_nets": net_id + 1, "n_pins": n_pins,
                "t_components_s": t_components, "t_nets_s": t_nets}


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


def _induced_hyperedges(node_ids, nl, n2n_starts, n2n_nets, max_net_degree=None):
    """Local (0..n-1), deduplicated hyperedges for the induced sub-hypergraph
    on `node_ids` (a numpy array of global node indices). Nets with fewer
    than 2 pins inside the block after restriction are dropped (they carry
    no cut information for *this* bisection). Only nets touching at least
    one node of `node_ids` are ever examined (via `n2n_starts`/`n2n_nets`
    from `node_to_nets_csr`), not the whole netlist.

    `max_net_degree` (M4 T6 finding, 2026-08-15): nets whose *global* degree
    (`nl.net_degrees[net]`, not the degree restricted to this block) exceeds
    this are skipped entirely. Real cluster-level designs carry a handful of
    pathological global nets (`mempool_cluster`: 2 nets alone carry 986,903
    of 43.9M pins -- power/ground/global-clock-style nets, not local
    connectivity) that make every mtkahypar bisection call's local-search
    refinement (gain updates touch every block a giant hyperedge spans)
    orders of magnitude slower without adding real scaling signal -- a
    single first-level bisection of `mempool_cluster`'s full 12.7M-net
    hypergraph did not complete in 24+ minutes with these included (vs.
    16.6s for `mempool_group`'s 3.5M-net hypergraph, which has no such
    nets, at the same node count ratio that would predict ~60s). Filtering
    is the same discipline already used throughout this project's
    DREAMPlace configs (`ignore_net_degree`, typically 100) -- `measure_rent`
    threads a default of 100 through for consistency, applied identically
    on every side of a comparison (the adjudication doc's H1 "同報告同估計
    器" rule)."""
    local_of = {int(v): i for i, v in enumerate(node_ids)}
    cand_nets = set()
    for v in node_ids:
        s, e = n2n_starts[v], n2n_starts[v + 1]
        cand_nets.update(n2n_nets[s:e].tolist())
    edges = []
    for net in cand_nets:
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        if max_net_degree is not None and (e - s) > max_net_degree:
            continue
        pins = nl.pin2node[nl.flat_net2pin[s:e]]
        local = sorted({local_of[int(v)] for v in pins if int(v) in local_of})
        if len(local) >= 2:
            edges.append(local)
    return edges


def bisect_mtkahypar(node_ids, nl, seed, threads=4, epsilon=0.03, n2n_csr=None, max_net_degree=None):
    """n2n_csr: optional precomputed `node_to_nets_csr(nl)` result (pass it
    when calling this repeatedly, e.g. from `recursive_bisection`, to avoid
    rebuilding the reverse index every call). `max_net_degree`: see
    `_induced_hyperedges`."""
    import mtkahypar
    from ioplace.partition import mtkahypar_runtime
    if len(node_ids) < 2:
        return node_ids, np.array([], dtype=node_ids.dtype)
    n2n_starts, n2n_nets = n2n_csr if n2n_csr is not None else node_to_nets_csr(nl)
    edges = _induced_hyperedges(node_ids, nl, n2n_starts, n2n_nets, max_net_degree=max_net_degree)
    if not edges:
        half = len(node_ids) // 2
        return node_ids[:half], node_ids[half:]
    mtk = mtkahypar_runtime.initialize(mtkahypar, threads)
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


def recursive_bisection(nl, max_level, seed=0, backend="auto", threads=4, root_node_ids=None,
                         max_net_degree=None):
    """Returns (levels: list[list[np.ndarray node ids]], backend_used).
    levels[0] is [root_node_ids] (the whole netlist, 1 block);
    levels[l] has 2**l blocks. `max_net_degree`: see `_induced_hyperedges`."""
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
                                                n2n_csr=n2n_csr, max_net_degree=max_net_degree)
            else:
                left, right = bisect_geometric(block, nl, axis=level % 2)
            cur.append(left)
            cur.append(right)
        levels.append(cur)
    return levels, backend


def block_terminal_counts(blocks, nl, max_net_degree=None):
    """External terminal count per block: a net contributes 1 to every
    block it has >=1 pin in, but only for nets that touch more than one
    block (a net entirely inside one block is not a boundary terminal).
    `max_net_degree` (see `_induced_hyperedges`): nets above this degree are
    excluded here too -- consistency with the partitioning hypergraph
    matters because a giant global net, left in, would inflate *every*
    block's terminal count by ~1 regardless of the partition actually
    chosen, swamping the level-to-level B^p scaling signal this function
    feeds into.

    **M4 T6 performance fix (2026-08-15):** this used to be a Python `for
    net in range(nl.num_nets)` loop with a `np.unique` call per net --
    `measure_rent` calls this once per recursion level (~14 levels for a
    ~12M-node netlist), and at `mempool_cluster` scale (12.7M nets) this
    was the actual bottleneck, not the mtkahypar partitioning it follows:
    a live run was still short of finishing its first level's worth of
    `block_terminal_counts` calls after 13+ minutes of CPU time, versus the
    59s the *partitioning itself* (`bisect_mtkahypar`) took at the same
    scale in isolation. Rewritten as one vectorized pass: bucket every pin
    into `net_id * (n_blocks+1) + block_id`, `np.unique` the whole key
    array once (dedupes same-net/same-block repeats -- the same thing the
    per-net `np.unique` was doing, just batched), then two `bincount`s
    (distinct-block-count per net, to find crossing nets; terminal count
    per block, restricted to those nets). Semantically identical to the
    loop version (verified against it on this module's existing toy
    fixtures), no `for` over nets or pins."""
    node_to_block = np.full(nl.num_physical, -1, dtype=np.int64)
    for b_idx, ids in enumerate(blocks):
        node_to_block[ids] = b_idx
    n_blocks = len(blocks)

    pin_nodes = nl.pin2node[nl.flat_net2pin]  # (P_used,) node id, net-grouped pin order
    pin_block = node_to_block[pin_nodes]  # (P_used,) block id per pin, -1 if node not covered
    net_id = np.repeat(np.arange(nl.num_nets, dtype=np.int64),
                        np.diff(nl.flat_net2pin_start))  # (P_used,) net id, same order as pin_block

    valid = pin_block >= 0
    if max_net_degree is not None:
        keep_net = nl.net_degrees <= max_net_degree
        valid = valid & keep_net[net_id]
    pin_block = pin_block[valid]
    net_id = net_id[valid]

    terms = np.zeros(n_blocks, dtype=np.int64)
    if len(net_id) == 0:
        return terms

    key = net_id * np.int64(n_blocks + 1) + pin_block
    uniq_key = np.unique(key)
    uniq_net = uniq_key // (n_blocks + 1)
    uniq_block = uniq_key % (n_blocks + 1)

    net_distinct_count = np.bincount(uniq_net, minlength=nl.num_nets)
    crossing = net_distinct_count[uniq_net] > 1
    block_counts = np.bincount(uniq_block[crossing], minlength=n_blocks)
    terms[:] = block_counts[:n_blocks]
    return terms


def _fit_p(levels_bt):
    """OLS fit of log(T) = log(t) + p*log(B) over [(B, T), ...] level
    averages. Returns (p, log_t)."""
    log_b = np.log(np.array([b for b, _ in levels_bt]))
    log_t = np.log(np.array([t for _, t in levels_bt]))
    p, c = np.polyfit(log_b, log_t, 1)
    return float(p), float(c)


def measure_rent(nl, max_level=None, b_lo=1e3, b_hi=1e6, seed=0, backend="auto",
                  threads=4, n_bootstrap=200, root_node_ids=None, rng=None, max_net_degree=None):
    """Landman-Russo recursive-bisection Rent exponent. `max_level`
    defaults to enough levels that the smallest block averages < b_lo (so
    the window naturally has content); pass it explicitly for small/toy
    netlists where the default would recurse past single-node blocks.
    `max_net_degree` (see `_induced_hyperedges`; `None` by default, kept
    backward compatible for existing callers/tests): pass e.g. 100 -- this
    project's usual `ignore_net_degree` -- when measuring Rent on a design
    with pathological global nets (confirmed necessary for
    `mempool_cluster`, see `_induced_hyperedges`'s docstring); the same
    value must be used on every side of an H1/H6 comparison."""
    n_root = nl.num_physical if root_node_ids is None else len(root_node_ids)
    if max_level is None:
        max_level = max(1, math.ceil(math.log2(max(1, n_root / b_lo))))

    levels, backend_used = recursive_bisection(
        nl, max_level, seed=seed, backend=backend, threads=threads, root_node_ids=root_node_ids,
        max_net_degree=max_net_degree)

    rent_levels = []
    for l, blocks in enumerate(levels):
        sizes = [len(b) for b in blocks]
        terms = block_terminal_counts(blocks, nl, max_net_degree=max_net_degree) if l > 0 else np.array([0])
        # level 0 (the whole netlist as one block) has, by definition, 0
        # external terminals -- it has no "outside".
        rent_levels.append(RentLevel(level=l, n_blocks=len(blocks), block_sizes=sizes,
                                      terminals=list(map(int, terms))))

    usable = [(rl.level, rl.avg_block_size, rl.avg_terminals) for rl in rent_levels
              if rl.level >= 1 and rl.avg_terminals > 0 and b_lo <= rl.avg_block_size <= b_hi]
    if len(usable) < 2:
        runtime = None
        if backend_used == "mtkahypar":
            from ioplace.partition.mtkahypar_runtime import metadata
            runtime = metadata()
        return RentResult(p=float("nan"), p_ci_lo=float("nan"), p_ci_hi=float("nan"),
                           log_t=float("nan"), backend=backend_used,
                           levels_used=[u[0] for u in usable], levels=rent_levels, n_bootstrap=0,
                           runtime=runtime)

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

    runtime = None
    if backend_used == "mtkahypar":
        from ioplace.partition.mtkahypar_runtime import metadata
        runtime = metadata()
    return RentResult(p=p, p_ci_lo=float(p_lo), p_ci_hi=float(p_hi), log_t=log_t,
                       backend=backend_used, levels_used=levels_used, levels=rent_levels,
                       n_bootstrap=len(boot_ps), runtime=runtime)
