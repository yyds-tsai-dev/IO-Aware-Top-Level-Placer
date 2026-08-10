import json, os, time
import numpy as np
from ioplace.drivers.run_placement import (_load_dreamplace, extract_final_positions,
    _evaluate_and_pack, get_regions_for)
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.reweight import update_net_weights

def run_reweight(config_json, k, rtype, seed, out_json, every=100, alpha=0.5):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    # NonLinearPlace is a bare top-level module inside $DREAMPLACE_ROOT/install
    # (see Global Constraints: DREAMPlace mixes `import dreamplace.ops.*` and
    # bare `import Params`/`import NonLinearPlace` styles) -- it only becomes
    # importable once `_load_dreamplace` has called `setup_dreamplace()` and
    # pushed install/ + install/dreamplace/ onto sys.path, so this import must
    # stay here, after that call, not hoisted to the top of the function.
    import NonLinearPlace
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)           # initialize 後(scale 後)座標系
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    ctx = GpuEvalContext(nl, rg, device="cuda")
    lr = params.global_place_stages[0]["learning_rate"]
    # Same init_pos determinism guard as run_placement._place(): BasicPlace
    # draws centre-noise/filler init from numpy's global RNG, seeded only by
    # Placer.py's flow which we bypass.
    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
    state = {"count": 0}

    def cb(iteration, pos):
        if iteration == 0 or iteration % every != 0:
            return
        node_x = pos.data[:n_phys]
        node_y = pos.data[n_all:n_all + n_phys]
        res = ctx.evaluate(node_x, node_y)
        update_net_weights(placer.data_collections.net_weights,
                           res.per_net_crossings, alpha=alpha)
        state["count"] += 1

    placer.iteration_callback = cb
    placer(params, placedb, lr)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    # Task 12 review follow-up: placer is a local var (not returned), so this
    # is the only way for tests to see the live net_weights tensor's range.
    nw = placer.data_collections.net_weights.detach().cpu().numpy()
    result = {"mode": "reweight", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "reweight_every": every, "alpha": alpha,
              "num_reweights": state["count"], "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
              if torch.cuda.is_available() else 0.0,
              "net_weights_min": float(nw.min()), "net_weights_max": float(nw.max()),
              **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result
