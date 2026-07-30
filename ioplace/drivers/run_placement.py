import argparse, json, os, time
import numpy as np
from ioplace.dreamplace_env import setup_dreamplace
from ioplace.regions import make_grid_regions, make_slicing_regions
from ioplace.region_grid import RegionGrid
from ioplace.netlist import netlist_from_placedb
from ioplace.evaluator_ref import evaluate

GRID_SHAPES = {4: (2, 2), 8: (4, 2), 16: (4, 4), 32: (8, 4)}

def get_regions_for(die, k, rtype, seed, lattice=512):
    if rtype == "grid":
        nx, ny = GRID_SHAPES[k]
        return make_grid_regions(die, nx, ny, lattice=lattice)
    if rtype == "slicing":
        return make_slicing_regions(die, k, seed=seed, lattice=lattice)
    raise ValueError(rtype)

def _load_dreamplace(config_json):
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))   # config 內是相對路徑
    try:
        params.load(config_json)
        # M0/M1 protocol: GP+LG only (spec §8), DP off regardless of what the
        # input config says.
        #
        # `detailed_place_engine`/`detailed_place_command` (params.json: "external
        # detailed placement engine to be called after placement") only matter to
        # dreamplace/Placer.py's top-level place() driver, which invokes an *external*
        # DP tool (e.g. ntuplace) as a subprocess after the whole flow finishes. Our
        # driver never calls Placer.py -- it drives NonLinearPlace directly -- so that
        # field is a no-op for us either way; kept for documentation/defensiveness.
        #
        # The field NonLinearPlace.__call__ actually branches on (source: $DP/install/
        # dreamplace/NonLinearPlace.py:920, `if params.detailed_place_flag:` guarding
        # the internal GPU detailed-placement op) is `detailed_place_flag`. Some configs
        # (e.g. install/test/ispd2005/adaptec1.json) set it to 1, so it must be forced
        # off explicitly -- setting only detailed_place_engine would silently leave
        # internal DP running for those configs.
        params.detailed_place_engine = ""
        params.detailed_place_flag = 0
        params.plot_flag = 0
        placedb = PlaceDB.PlaceDB()
        placedb.read(params)
        return params, placedb
    finally:
        os.chdir(cwd)

def _place(params, placedb):
    import NonLinearPlace
    lr = params.global_place_stages[0]["learning_rate"]
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb, lr)
    return placer, metrics

def extract_final_positions(placer, placedb):
    """Return the final (node_x, node_y) after GP+LG, len == num_physical_nodes,
    in placedb's internal (shifted+scaled) coordinate system.

    Investigation (Task 7 Step 1; see $DP/install/dreamplace/NonLinearPlace.py:930-937
    and PlaceDB.py:1121-1133 `apply()`): NonLinearPlace.__call__ ends with

        cur_pos = self.pos[0].data.clone().cpu().numpy()
        placedb.apply(params, cur_pos[:num_movable_nodes],
                      cur_pos[num_nodes:num_nodes + num_movable_nodes])

    and `PlaceDB.apply()` writes that back into `self.node_x[:num_movable_nodes]` /
    `self.node_y[:num_movable_nodes]` *before* unscaling (the unscale inside apply()
    only feeds the separate rawdb/C++ mirror used for file export). So by the time
    `placer(params, placedb, lr)` returns, `placedb.node_x`/`node_y` already hold the
    final placement for the movable segment, in the same scaled coordinate system as
    `placedb.xl/yl/xh/yh`. The fixed/terminal segment (indices
    num_movable_nodes:num_physical_nodes) is never touched by apply() because it's
    never touched by the optimizer either -- it was already correct (and already in
    that same scaled system) since `placedb.initialize()`'s `scale()` call.

    This is candidate A from the brief, confirmed empirically: after a full `simple`
    GP+LG run, `placedb.node_x/node_y` and `placer.pos[0]` (candidate B, sliced per
    the brief's `pos[:n_phys]` / `pos[n_all:n_all+n_phys]` scheme) were bit-for-bit
    identical over all num_physical_nodes entries (movable and fixed alike; see
    task-7-report.md for the comparison run). That equality is structural, not a
    coincidence of the `simple` benchmark: `apply()` is always the last data-mutating
    step of `__call__`, and it always assigns from a fresh clone of the same
    `self.pos[0]` tensor that candidate B reads. Candidate A is used here because it
    is simpler (no reliance on `pos[0]`'s internal `[x_all | y_all]` layout) and
    matches how the rest of ioplace (`netlist_from_placedb`, Task 2) already reads
    positions off placedb. `placer` is accepted for interface symmetry with future
    drivers (Tasks 9/12) and is intentionally unused here.
    """
    n_phys = placedb.num_physical_nodes
    node_x = np.array(placedb.node_x[:n_phys], dtype=np.float64)
    node_y = np.array(placedb.node_y[:n_phys], dtype=np.float64)
    return node_x, node_y

def _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed):
    nl = netlist_from_placedb(placedb)
    nl.node_x, nl.node_y = node_x, node_y
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    res = evaluate(nl, node_x, node_y, rg)
    return rg, {"io_count": res.io_count, "ft_count": res.ft_count,
                "tree_wl": res.tree_wl, "hpwl": res.hpwl,
                "large_net_lb": res.large_net_lb}

def run_flat(config_json, k, rtype, seed, out_json):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    placedb.initialize(params)
    placer, _ = _place(params, placedb)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    result = {"mode": "flat", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
              if torch.cuda.is_available() else 0.0, **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["flat", "two_stage", "reweight"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reweight-every", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    if args.mode == "flat":
        run_flat(args.config, args.k, args.rtype, args.seed, args.out)
    elif args.mode == "two_stage":
        from ioplace.drivers.run_placement_two_stage import run_two_stage  # Task 9
        run_two_stage(args.config, args.k, args.rtype, args.seed, args.out)
    else:
        from ioplace.drivers.run_placement_reweight import run_reweight    # Task 12
        run_reweight(args.config, args.k, args.rtype, args.seed, args.out,
                     every=args.reweight_every, alpha=args.alpha)

if __name__ == "__main__":
    main()
