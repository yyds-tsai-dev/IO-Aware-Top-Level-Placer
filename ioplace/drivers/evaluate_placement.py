"""M4 T8a overflow-diagnosis follow-up: evaluate-only driver.

Builds PlaceDB + the evaluator from an *existing* driver-produced `.npz`
(`node_x`/`node_y`, the format `run_placement.run_flat`/
`run_placement_io.run_io` both save alongside their JSON output) + the same
DREAMPlace config, WITHOUT running GP or LG, and reports the same
evaluator-field schema those drivers' own final eval block does (io_count,
io_gp, ft_count, hard_lambda_sum, tree_wl, hpwl, io_rg, ft_rg).

Why: some M4 experiments (e.g. mempool_cluster/mempool_group's flat
baseline) only need one real GP+LG run; every other K is a re-evaluation of
that *same* final placement under a different region grid. Re-running GP+LG
per K would burn GPU time on work whose answer (the placement) doesn't
change with K -- only the region assignment does.

Bit-identical to the source run's own fields (for the same K/rtype/seed):
this reuses `run_placement._evaluate_and_pack` (evaluator_ref, gives
io_count/ft_count/tree_wl/hpwl) and `evaluator_gpu.GpuEvalContext.evaluate`
(gives hard_lambda_sum/io_rg/ft_rg) verbatim -- the exact same two calls
`run_placement_io.run_io`'s own final `with timer.phase("eval"):` block
makes -- rather than reimplementing either evaluator's call convention.
`io_gp` has no meaning here (it is run_io's *mid-GP* io_count, measured on a
position snapshot this driver never has -- only the final post-GP+LG
position survives into the saved npz) so it is reported as `None`, not
fabricated from `io_count`.
"""
import argparse, hashlib, json, os, time
import numpy as np

from ioplace.drivers.run_placement import (_load_dreamplace, _evaluate_and_pack,
    get_regions_for, _t8a_provenance)
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def evaluate_placement(npz_path, config_json, k, rtype, out_json, *, seed=0):
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    placedb.initialize(params)
    n_phys = placedb.num_physical_nodes

    npz = np.load(npz_path)
    node_x = np.asarray(npz["node_x"], dtype=np.float64)
    node_y = np.asarray(npz["node_y"], dtype=np.float64)
    if node_x.shape[0] != n_phys or node_y.shape[0] != n_phys:
        raise ValueError(
            f"{npz_path}: node_x/node_y length {node_x.shape[0]}/{node_y.shape[0]} "
            f"!= placedb.num_physical_nodes {n_phys} for config {config_json} -- "
            "npz and config look mismatched")

    # Same two evaluator calls, same order, as run_placement_io.run_io's own
    # final eval block (run_placement_io.py's "with timer.phase(\"eval\"):").
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)

    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    ctx = GpuEvalContext(nl, rg, device="cuda")
    res = ctx.evaluate(node_x, node_y)

    result = {
        "mode": "evaluate_only", "npz": npz_path, "config": config_json,
        "k": k, "rtype": rtype, "seed": seed,
        "io_count": metrics["io_count"],
        # No mid-GP position snapshot survives into the saved npz -- see the
        # module docstring for why this is None rather than an alias of
        # io_count.
        "io_gp": None,
        "ft_count": metrics["ft_count"], "hard_lambda_sum": res.hard_lambda_sum,
        "tree_wl": metrics["tree_wl"], "hpwl": metrics["hpwl"],
        "io_rg": res.io_rg, "ft_rg": res.ft_rg,
        "runtime_s": time.time() - t0,
        "evaluate_only": True,
        "source_npz": npz_path, "source_npz_sha256": _sha256(npz_path),
        **_t8a_provenance(config_json),
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    ap.add_argument("--seed", type=int, default=0,
                    help="region-generation seed (--rtype slicing only; "
                         "ignored by --rtype grid, default 0 matches the "
                         "other drivers' default)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    evaluate_placement(args.npz, args.config, args.k, args.rtype, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
