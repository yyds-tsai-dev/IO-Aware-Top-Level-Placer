import json, os, time
import numpy as np
from ioplace.drivers.run_placement import (_load_dreamplace, _place,
    extract_final_positions, _evaluate_and_pack, get_regions_for)
from ioplace.fence_inject import inject_fence_regions
from ioplace.netlist import netlist_from_placedb
from ioplace.partition.mtkahypar_runner import partition_netlist
from ioplace.region_grid import RegionGrid

def _pick_escape_cell(node2fence_region_map, parts, node_size_x, node_size_y, k):
    """選一顆 cell 解除 fence(繞 DREAMPlace 隱含 no-fence bucket 為空的 crash)。
    回傳 cell index。優先挑非 singleton block 中面積最小的 movable cell;
    全部 block 都是 singleton 時 fallback 到全體中面積最小者。

    `node2fence_region_map` is accepted for interface symmetry with the
    caller's `node2fence_region_map[idx] = k` mutation step that follows
    this call, and is intentionally unused here -- this function only
    *selects* an index, it never mutates.
    """
    m = len(parts)
    area = node_size_x[:m] * node_size_y[:m]
    safe = np.bincount(parts, minlength=k)[parts] >= 2
    pool = np.where(safe)[0] if safe.any() else np.arange(m)
    return int(pool[np.argmin(area[pool])])

def assign_blocks_to_regions(parts, nl, rs):
    """把 mtkahypar partition block 指派到幾何 region,最小化
    Σ_{a<b} C[a,b] * D[pi(a),pi(b)](pi: block id -> region id)。
    C = block-pair connectivity(每 net 對其 touched-block set 的每一對 +1),
    D = region 中心間 Manhattan 距離。greedy 初始 + 2-opt pairwise swap 至收斂。
    回傳 remapped parts(cell -> region id)。確定性(無隨機)。

    Context (M0 sanity-check finding; see task-10-report.md): `parts` from
    `partition_netlist` is a *hypergraph* block id -- mtkahypar only knows
    "which cells are cheap to cut apart" and has zero notion of which block
    ends up physically near which die region. The pre-fix driver fed `parts`
    straight into `inject_fence_regions` as if block id == region id, so two
    blocks mtkahypar judged cheap to separate (lightly connected) could just
    as easily land on physically adjacent regions as on opposite die
    corners -- inflating hpwl/io_count/ft_count for reasons unrelated to the
    fence constraint itself. This function computes the block<->region
    permutation that keeps heavily-connected blocks geometrically close, so
    the fence constraint is at least directionally aligned with the
    netlist's own connectivity.

    C[a,b]: for every net (skipping degree>256 nets -- same spirit as
    evaluator_ref's large-net lb branch and `initial_cut_io_lb`'s own
    per-net loop: huge nets touch most blocks trivially and would swamp the
    pairwise signal with an almost-complete clique), the set of distinct
    blocks its *movable* pins touch (terminals are geometrically fixed
    already -- see fence_inject.py's separate `rg.region_of_points` path for
    them -- so they don't participate in this remap) contributes +1 to every
    touched-block pair.

    D[r1,r2]: Manhattan distance between region r1's and r2's area-weighted
    rect centers (a RegionSpec can in general hold multiple rects; grid/
    slicing regions from ioplace.regions happen to always be single-rect).

    Optimization: greedy insertion (start from the globally heaviest block
    in the most central region -- the one minimizing summed distance to
    every other region; then repeatedly place the unassigned block most
    connected to what's already placed into the empty region minimizing its
    connectivity-weighted distance to already-placed neighbors), followed by
    2-opt pairwise region swaps (repeated full passes over all K*(K-1)/2
    region pairs, applying any strictly-improving swap immediately, until
    one full pass finds none). K <= 32 (largest GRID_SHAPES entry) always,
    so both stages are effectively instant.

    Fully deterministic: argmax/argmin break ties on the lowest index and
    there is no RNG anywhere in this function.

    Args:
        parts: (num_movable,) int array, block id per movable cell (as
            returned by `partition_netlist(...)[:num_movable_nodes]`).
        nl: Netlist used for connectivity (flat_net2pin/flat_net2pin_start/
            pin2node/num_nets/num_movable).
        rs: RegionSet used for geometry (`.regions[i].rects` and `.k`).

    Returns:
        (num_movable,) int32 array: `parts` remapped cell -> region id, same
        shape/indexing as the input `parts` -- ready to feed straight into
        `inject_fence_regions`.
    """
    K = rs.k
    parts = np.asarray(parts, dtype=np.int64)
    m = len(parts)
    assert m == nl.num_movable, "parts length must equal nl.num_movable"

    # ---- C: block-pair connectivity, from movable-only touched blocks ----
    pin_block = np.full(len(nl.pin2node), -1, dtype=np.int64)
    movable_pin = nl.pin2node < m
    pin_block[movable_pin] = parts[nl.pin2node[movable_pin]]

    C = np.zeros((K, K), dtype=np.float64)
    start, flat = nl.flat_net2pin_start, nl.flat_net2pin
    for net in range(nl.num_nets):
        s, e = start[net], start[net + 1]
        d = e - s
        if d <= 1 or d > 256:
            continue
        blocks = np.unique(pin_block[flat[s:e]])
        blocks = blocks[blocks >= 0]
        t = len(blocks)
        for i in range(t):
            a = int(blocks[i])
            for j in range(i + 1, t):
                b = int(blocks[j])
                C[a, b] += 1.0
                C[b, a] += 1.0

    # ---- D: region-center Manhattan distances ----
    centers = np.zeros((K, 2), dtype=np.float64)
    for rid, r in enumerate(rs.regions):
        rects = np.asarray(r.rects, dtype=np.float64).reshape(-1, 4)
        area = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        cx = (rects[:, 0] + rects[:, 2]) * 0.5
        cy = (rects[:, 1] + rects[:, 3]) * 0.5
        tot = area.sum()
        centers[rid] = (np.sum(cx * area) / tot, np.sum(cy * area) / tot)
    D = (np.abs(centers[:, None, 0] - centers[None, :, 0]) +
         np.abs(centers[:, None, 1] - centers[None, :, 1]))

    # ---- greedy initial assignment ----
    pi = np.full(K, -1, dtype=np.int64)    # block id -> region id
    inv = np.full(K, -1, dtype=np.int64)   # region id -> block id
    assigned = np.zeros(K, dtype=bool)
    region_used = np.zeros(K, dtype=bool)

    first_block = int(np.argmax(C.sum(axis=1)))
    first_region = int(np.argmin(D.sum(axis=1)))
    pi[first_block] = first_region
    inv[first_region] = first_block
    assigned[first_block] = True
    region_used[first_region] = True

    for _ in range(K - 1):
        scores = C[:, assigned].sum(axis=1)
        scores[assigned] = -np.inf
        b = int(np.argmax(scores))
        assigned_blocks = np.nonzero(assigned)[0]
        weights = C[b, assigned_blocks]
        assigned_regions = pi[assigned_blocks]
        avail_regions = np.nonzero(~region_used)[0]
        cost = (D[np.ix_(avail_regions, assigned_regions)]
                * weights[None, :]).sum(axis=1)
        r = int(avail_regions[int(np.argmin(cost))])
        pi[b] = r
        inv[r] = b
        assigned[b] = True
        region_used[r] = True

    # ---- 2-opt: pairwise region swaps until a full pass improves nothing ----
    def total_cost(p):
        return float(np.sum(C * D[np.ix_(p, p)]))

    cur = total_cost(pi)
    improved = True
    while improved:
        improved = False
        for r1 in range(K):
            for r2 in range(r1 + 1, K):
                a, b = int(inv[r1]), int(inv[r2])
                pi[a], pi[b] = r2, r1
                nxt = total_cost(pi)
                if nxt < cur:
                    inv[r1], inv[r2] = b, a
                    cur = nxt
                    improved = True
                else:
                    pi[a], pi[b] = r1, r2

    return pi[parts].astype(np.int32)

def run_two_stage(config_json, k, rtype, seed, out_json):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    assert params.enable_fillers == 1, "fence mode requires enable_fillers"
    nl0 = netlist_from_placedb(placedb)          # read 後的原始座標系
    parts = partition_netlist(nl0, k, seed=seed)[:placedb.num_movable_nodes]
    die0 = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs0 = get_regions_for(die0, k, rtype, seed)
    assert rs0.k == k
    # M0 fix (task-10-report.md): mtkahypar's block id carries no geometric
    # meaning on its own -- remap block -> region so heavily-connected
    # blocks land in adjacent regions before anything downstream (fence
    # injection, the IO lower bound, fence_compliance) treats `parts` as a
    # region id.
    parts = assign_blocks_to_regions(parts, nl0, rs0)
    inject_fence_regions(placedb, rs0, parts)

    # --- Workaround for a DREAMPlace fence-region bug (investigated in
    # Task 9; see task-9-report.md for the full repro) ---
    # PlaceDB.calc_num_filler_for_fence_region() is called unconditionally
    # from initialize() whenever len(placedb.regions) > 0, once per
    # region_id in range(len(placedb.regions) + 1): ids 0..k-1 are our real
    # regions, and id == k is an *implicit* "no fence assigned" bucket that
    # DREAMPlace always allocates one slot for. Our regions always tile the
    # full die (spec D2/D3: 全 die 分割), so every movable node's
    # node2fence_region_map value is in [0, k) and that implicit bucket's
    # cell mask is *always* empty -- np.percentile() of an empty array
    # (PlaceDB.py ~L687) silently yields NaN, and the later
    # int(round(nan)) (PlaceDB.py ~L728) raises ValueError, crashing
    # initialize() before global placement even starts. DREAMPlace source
    # is off-limits (Global Constraints), so we route around it here in our
    # own driver: reassign the single smallest-area movable cell to that
    # implicit bucket so its mask is non-empty. `parts` (used below for
    # initial_cut_io_lb/fence_compliance) and `rs0` are left untouched, so
    # this only affects DREAMPlace's internal filler bookkeeping, not our
    # reported partition semantics -- worst-case impact on fence_compliance
    # is 1/num_movable_nodes (negligible for real benchmarks; confirmed
    # exactly 1/210904 == the entire gap from 1.0 on adaptec1).
    #
    # Unrelated to, and still needed after, the separate shapely 2.x
    # compatibility patch on $DREAMPLACE_ROOT's `io-aware` branch (see
    # ioplace/dp_patch/shapely2-compat.patch) -- that patch fixes NaN
    # region bounds in slice_non_fence_region; this fixes the always-empty
    # filler bucket in calc_num_filler_for_fence_region. Different
    # functions, different files, both needed.
    #
    # The escaped cell must not be the *sole* member of its own partition
    # block, or emptying that block's mask just moves the identical crash
    # from region_id == k to that now-empty real region_id (observed on the
    # 8-cell `simple` benchmark, where mtkahypar can assign a region as few
    # as 1 cell). Real benchmarks have thousands of cells per partition, so
    # this only matters for tiny/synthetic cases, but the guard is cheap and
    # keeps the driver correct at any scale. Selection logic lives in the
    # module-level _pick_escape_cell() helper (unit-tested directly in
    # tests/test_fence_inject.py).
    m = placedb.num_movable_nodes
    escape_idx = _pick_escape_cell(placedb.node2fence_region_map, parts,
                                    placedb.node_size_x, placedb.node_size_y, k)
    placedb.node2fence_region_map[escape_idx] = k

    # assignment 固有 IO 下界:Σ_e (touched parts − 1)
    assert placedb.num_terminal_NIs == 0, (
        "initial_cut_io_lb assumes no terminal_NI (IO pad) nodes; "
        "extend node_part to num_physical with geometric assignment before using "
        "benchmarks that have IO pads (e.g. LEF/DEF mempool cases)."
    )
    node_part = np.concatenate([parts, placedb.node2fence_region_map[
        placedb.num_movable_nodes:]]).astype(np.int64)
    lam = 0
    for net in range(nl0.num_nets):
        s, e = nl0.flat_net2pin_start[net], nl0.flat_net2pin_start[net + 1]
        lam += len(np.unique(node_part[nl0.pin2node[nl0.flat_net2pin[s:e]]])) - 1
    placedb.initialize(params)
    placer, _ = _place(params, placedb)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    # fence compliance:LG 後 movable cells 落點 region == 指派 region 的比例
    die1 = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg1 = RegionGrid(get_regions_for(die1, k, rtype, seed))
    landed = rg1.region_of_points(node_x[:m], node_y[:m])
    compliance = float((landed == parts).mean())
    result = {"mode": "two_stage", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
              if torch.cuda.is_available() else 0.0,
              "initial_cut_io_lb": int(lam), "fence_compliance": compliance,
              **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result
