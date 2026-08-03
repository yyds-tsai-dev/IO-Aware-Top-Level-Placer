import json, os, time
import numpy as np
from ioplace.drivers.run_placement import (_load_dreamplace, _place,
    extract_final_positions, _evaluate_and_pack, get_regions_for)
from ioplace.fence_inject import inject_fence_regions
from ioplace.netlist import netlist_from_placedb
from ioplace.partition.mtkahypar_runner import partition_netlist
from ioplace.region_grid import RegionGrid

def run_two_stage(config_json, k, rtype, seed, out_json):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    assert params.enable_fillers == 1, "fence mode requires enable_fillers"
    nl0 = netlist_from_placedb(placedb)          # read 後的原始座標系
    parts = partition_netlist(nl0, k, seed=seed)[:placedb.num_movable_nodes]
    die0 = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs0 = get_regions_for(die0, k, rtype, seed)
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
    # is 1/num_movable_nodes (negligible for real benchmarks).
    #
    # The escaped cell must not be the *sole* member of its own partition
    # block, or emptying that block's mask just moves the identical crash
    # from region_id == k to that now-empty real region_id (observed on the
    # 8-cell `simple` benchmark, where mtkahypar can assign a region as few
    # as 1 cell). Real benchmarks have thousands of cells per partition, so
    # this only matters for tiny/synthetic cases, but the guard is cheap and
    # keeps the driver correct at any scale.
    m = placedb.num_movable_nodes
    area = placedb.node_size_x[:m] * placedb.node_size_y[:m]
    safe = np.bincount(parts, minlength=k)[parts] >= 2
    pool = np.where(safe)[0] if safe.any() else np.arange(m)
    escape_idx = int(pool[np.argmin(area[pool])])
    placedb.node2fence_region_map[escape_idx] = k

    # assignment 固有 IO 下界:Σ_e (touched parts − 1)
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
