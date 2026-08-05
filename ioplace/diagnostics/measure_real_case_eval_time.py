"""Task 13 diagnostic: real-case GpuEvalContext.evaluate() timing.

Committed per Task 13 review Finding 1: docs/results/m1-reweight-report.md's
"GPU evaluator runtime" section's real-case (adaptec1/bigblue4) numbers were
originally produced by this script from an ephemeral scratchpad, and its
stdout was never captured to disk -- so the report's headline numbers had no
committed evidence. This copy is unchanged in algorithm (same ctor + 1 cold +
N warm evaluate() methodology as Task 11 Section 3/8.1); it adds an optional
--out-json so the measurement is reproducible and its raw output archivable
under results/m1/diagnostics/, and drops the scratchpad-only sys.path hack
(ioplace is importable directly once the repo's venv is active).

Measures GpuEvalContext.evaluate() per-call time on adaptec1/bigblue4's
actual netlist, using the final placed positions already saved by the M1
reweight runs' .npz side files. Per Task 11's own "Task 13 should re-run
this once flat-placement results exist" follow-up note.

Usage (both invocations used for the M1 report's real-case timing table):
    $PY -m ioplace.diagnostics.measure_real_case_eval_time \\
        $DREAMPLACE_ROOT/install/test/ispd2005/adaptec1.json 16 grid 0 \\
        results/m1/adaptec1_reweight_k16_grid.json.npz \\
        --out-json results/m1/diagnostics/adaptec1_eval_time.json

    $PY -m ioplace.diagnostics.measure_real_case_eval_time \\
        $DREAMPLACE_ROOT/install/test/ispd2005/bigblue4.json 16 grid 0 \\
        results/m1/bigblue4_reweight_k16_grid.json.npz \\
        --out-json results/m1/diagnostics/bigblue4_eval_time.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext


def measure(config_json, k, rtype, seed, npz_path, n_warm=10, out_json=None):
    params, placedb = _load_dreamplace(config_json)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))

    npz = np.load(npz_path)
    node_x, node_y = npz["node_x"], npz["node_y"]

    print(f"case={os.path.basename(config_json)} k={k} rtype={rtype} "
          f"n_nets={nl.num_nets} n_pins={len(nl.pin2net)} n_nodes={len(node_x)}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ctx = GpuEvalContext(nl, rg, device="cuda")
    torch.cuda.synchronize()
    ctor_s = time.time() - t0
    print(f"ctor: {ctor_s:.4f}s")

    t0 = time.time()
    res = ctx.evaluate(node_x, node_y)
    torch.cuda.synchronize()
    cold_s = time.time() - t0
    cold_io_count, cold_ft_count = res.io_count, res.ft_count
    print(f"cold evaluate(): {cold_s:.4f}s  io_count={cold_io_count} ft_count={cold_ft_count}")

    warm_times = []
    for _ in range(n_warm):
        t0 = time.time()
        res = ctx.evaluate(node_x, node_y)
        torch.cuda.synchronize()
        warm_times.append(time.time() - t0)
    warm_times.sort()
    median_warm = warm_times[len(warm_times) // 2]
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    print(f"warm evaluate() times: {[f'{t:.4f}' for t in warm_times]}")
    print(f"median warm: {median_warm:.4f}s")
    print(f"peak_gpu_mem_mb: {peak_mb:.1f}")
    print(f"last warm io_count={res.io_count} ft_count={res.ft_count} "
          f"(determinism check vs cold: "
          f"{'match' if (res.io_count, res.ft_count) == (cold_io_count, cold_ft_count) else 'DIFFERS'})")

    result = {
        "config": config_json, "k": k, "rtype": rtype, "seed": seed, "npz_path": npz_path,
        "n_nets": nl.num_nets, "n_pins": len(nl.pin2net), "n_nodes": int(len(node_x)),
        "ctor_s": ctor_s, "cold_evaluate_s": cold_s,
        "cold_io_count": cold_io_count, "cold_ft_count": cold_ft_count,
        "warm_times_s": warm_times, "median_warm_evaluate_s": median_warm,
        "last_warm_io_count": res.io_count, "last_warm_ft_count": res.ft_count,
        "peak_gpu_mem_mb": peak_mb,
    }
    if out_json:
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
    ap.add_argument("npz_path")
    ap.add_argument("n_warm", type=int, nargs="?", default=10)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    measure(args.config, args.k, args.rtype, args.seed, args.npz_path,
            n_warm=args.n_warm, out_json=args.out_json)


if __name__ == "__main__":
    main()
