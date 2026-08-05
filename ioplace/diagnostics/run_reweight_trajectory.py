"""Task 13 diagnostic: io_count/ft_count trajectory across reweight callbacks.

Committed per Task 13 review Finding 1: docs/results/m1-reweight-report.md's
"io/ft 軌跡診斷" table -- including its alpha=0 no-op control run, the key
evidence against the "reweight itself perturbs the late-stage trajectory"
hypothesis -- was originally produced by this script from an ephemeral
scratchpad and never committed. Relocated here unchanged in algorithm/output
(only CLI plumbing and paths were tidied for life inside the repo); the
`results/m1/diagnostics/*_trajectory.json` outputs alongside this script are
the original raw runs, not a re-run.

This is a copy of ioplace.drivers.run_placement_reweight.run_reweight that
additionally logs an io_count/ft_count trajectory at every reweight
callback. The committed driver intentionally isn't touched (Task 13's
mandate only touched make_report.py), so this stays a standalone diagnostic
script. The extra logging reuses the GpuEvalContext.evaluate() result the
real loop already computes (res.io_count/res.ft_count are free -- no extra
evaluator calls added, so this does not change runtime/algorithm behavior
relative to the committed run_reweight).

Usage (both invocations used for the M1 report's io/ft trajectory table):
    $PY -m ioplace.diagnostics.run_reweight_trajectory \\
        $DREAMPLACE_ROOT/install/test/ispd2005/adaptec1.json 16 grid 0 \\
        results/m1/diagnostics/adaptec1_reweight_k16_trajectory.json \\
        --every 100 --alpha 0.2

    $PY -m ioplace.diagnostics.run_reweight_trajectory \\
        $DREAMPLACE_ROOT/install/test/ispd2005/adaptec1.json 16 grid 0 \\
        results/m1/diagnostics/adaptec1_control_alpha0_k16_trajectory.json \\
        --every 100 --alpha 0.0
"""
import argparse
import json
import os
import time

from ioplace.drivers.run_placement import (_load_dreamplace, extract_final_positions,
    _evaluate_and_pack, get_regions_for)
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.reweight import update_net_weights


def run_reweight_with_trajectory(config_json, k, rtype, seed, out_json, every=100, alpha=0.5):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    import NonLinearPlace
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    ctx = GpuEvalContext(nl, rg, device="cuda")
    lr = params.global_place_stages[0]["learning_rate"]
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
    state = {"count": 0}
    trajectory = []

    def cb(iteration, pos):
        if iteration == 0 or iteration % every != 0:
            return
        node_x = pos.data[:n_phys]
        node_y = pos.data[n_all:n_all + n_phys]
        res = ctx.evaluate(node_x, node_y)
        nw_before = placer.data_collections.net_weights.detach().cpu().numpy()
        trajectory.append({
            "iteration": int(iteration),
            "io_count": res.io_count, "ft_count": res.ft_count,
            "net_weights_min_before": float(nw_before.min()),
            "net_weights_max_before": float(nw_before.max()),
        })
        update_net_weights(placer.data_collections.net_weights,
                           res.per_net_crossings, alpha=alpha)
        state["count"] += 1

    placer.iteration_callback = cb
    placer(params, placedb, lr)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    nw = placer.data_collections.net_weights.detach().cpu().numpy()
    result = {"mode": "reweight", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "reweight_every": every, "alpha": alpha,
              "num_reweights": state["count"], "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
              if torch.cuda.is_available() else 0.0,
              "net_weights_min": float(nw.min()), "net_weights_max": float(nw.max()),
              "trajectory": trajectory, **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("k", type=int)
    ap.add_argument("rtype", choices=["grid", "slicing"])
    ap.add_argument("seed", type=int)
    ap.add_argument("out_json")
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    res = run_reweight_with_trajectory(args.config, args.k, args.rtype, args.seed,
                                        args.out_json, every=args.every, alpha=args.alpha)
    print("final io_count:", res["io_count"], "ft_count:", res["ft_count"])
    print("trajectory:")
    for t in res["trajectory"]:
        print(" ", t)


if __name__ == "__main__":
    main()
