"""Stream all degree-two nets of a verified native cache through detour routing.

This isolates routing with fixed placement; higher-degree nets are explicitly
outside this benchmark. The route_net API supports their MST branches too.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import resource
import numpy as np
from ioplace.bench.bookshelf_netlist import load_tiled_netlist
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions, make_slicing_regions
from ioplace.route_eval.budgeted import BudgetedRouter
from ioplace.evaluator_ref import _walk_segment


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--placement", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--shift", nargs=2, type=float, required=True)
    ap.add_argument("--budgets", nargs="+", type=float, default=[0., .02, .05, .10])
    ap.add_argument("--region-kind", choices=["grid", "slicing"], default="slicing")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--region-seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=131072)
    args = ap.parse_args()
    if args.scale <= 0 or not np.isfinite(args.scale) or args.chunk_size <= 0:
        ap.error("positive scale and chunk size required")
    if any(b < 0 or not np.isfinite(b) for b in args.budgets):
        ap.error("finite nonnegative budgets required")
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=False)
    verify = json.loads((cache / "verification.json").read_text())["verify"]
    nl, meta = load_tiled_netlist(cache)
    if meta["schema_version"] != 4 or not verify["ok"] or not verify.get("native_pin_order_verified"):
        raise ValueError("requires verified schema4 native ordering")
    with np.load(args.placement) as z:
        x, y = z["node_x"], z["node_y"]
    if x.shape != (nl.num_physical,) or y.shape != x.shape:
        raise ValueError("placement/cache size mismatch")
    for coord, raw, shift in ((x, nl.node_x, args.shift[0]), (y, nl.node_y, args.shift[1])):
        fixed = np.array(raw[nl.num_movable:], dtype=np.float32)
        fixed -= shift
        fixed *= args.scale
        if not np.array_equal(coord[nl.num_movable:], fixed.astype(np.float64)):
            raise ValueError("fixed node identity/coordinate transform mismatch")
    die = (0., 0., (nl.xh-args.shift[0])*args.scale, (nl.yh-args.shift[1])*args.scale)
    if args.region_kind == "slicing":
        rs = make_slicing_regions(die, args.k, seed=args.region_seed)
    else:
        nx = 2 ** int(np.floor(np.log2(args.k) / 2))
        if args.k % nx:
            raise ValueError("grid K must factor into the selected rectangular grid")
        rs = make_grid_regions(die, nx, args.k // nx)
    rs.to_json(out / "regions.json")
    router = BudgetedRouter(RegionGrid(rs))
    protocol = {"args": vars(args), "scope": "all degree-two nets with both pins inside die",
        "full_netlist_nodes": nl.num_physical, "full_netlist_nets": nl.num_nets,
        "placement_sha256": sha(args.placement), "cache_meta_sha256": sha(cache / "meta.json"),
        "cache_verification_sha256": sha(cache / "verification.json"),
        "regions_sha256": sha(out / "regions.json"),
        "router_sha256": sha(Path(__file__).resolve().parents[1] / "ioplace/route_eval/budgeted.py"),
        "fixed_tail_exact": True, "routing_model": "obstacle-free three-segment proposals",
        "signoff_routed": False, "units": "DREAMPlace internal coordinates"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = {str(b): {"nets": 0, "baseline_io": 0, "io": 0, "hpwl": 0.,
                     "wirelength": 0., "improved_nets": 0, "extra_length_improved_nets": 0,
                     "checked_paths": 0, "max_edge_wirelength_ratio": 1.} for b in args.budgets}
    examples = {str(b): [] for b in args.budgets}
    omitted_degree, outside = 0, 0
    started = time.perf_counter()
    for lo in range(0, nl.num_nets, args.chunk_size):
        hi = min(lo + args.chunk_size, nl.num_nets)
        start = nl.flat_net2pin_start[lo:hi+1]
        ids = np.flatnonzero(np.diff(start) == 2)
        omitted_degree += hi-lo-len(ids)
        p = nl.flat_net2pin[start[ids]]
        q = nl.flat_net2pin[start[ids]+1]
        def pins(indices):
            # Match native float32 in-place scale before widening, as in the
            # independently verified 27.7M evaluator parity procedure.
            ox = np.array(nl.pin_offset_x[indices], dtype=np.float32)
            oy = np.array(nl.pin_offset_y[indices], dtype=np.float32)
            ox *= args.scale
            oy *= args.scale
            nodes = nl.pin2node[indices]
            return np.column_stack([x[nodes] + ox.astype(np.float64), y[nodes] + oy.astype(np.float64)])
        a, b = pins(p), pins(q)
        valid = (np.isfinite(a).all(1) & np.isfinite(b).all(1)
                 & (a >= die[:2]).all(1) & (a <= die[2:]).all(1)
                 & (b >= die[:2]).all(1) & (b <= die[2:]).all(1))
        outside += int((~valid).sum())
        a, b, ids = a[valid], b[valid], ids[valid]+lo
        for budget in args.budgets:
            value = router.route_edges(a, b, wirelength_budget=budget)
            row = rows[str(budget)]
            base_io, io = value["baseline_crossings"], value["crossings"]
            base_wl, wl = value["baseline_wirelength"], value["wirelength"]
            assert (io <= base_io).all() and (wl <= base_wl*(1+budget)).all()
            row["nets"] += len(ids)
            row["baseline_io"] += int(base_io.sum())
            row["io"] += int(io.sum())
            row["hpwl"] += float(base_wl.sum())
            row["wirelength"] += float(wl.sum())
            improved = np.flatnonzero(io < base_io)
            row["improved_nets"] += len(improved)
            row["extra_length_improved_nets"] += int(((io < base_io) & (wl > base_wl)).sum())
            row["max_edge_wirelength_ratio"] = max(row["max_edge_wirelength_ratio"],
                float(np.max(np.divide(wl, base_wl, out=np.ones_like(wl), where=base_wl>0), initial=1)))
            # First 128 improvements per budget, independently recounted and
            # retained as explicit geometry; no best-example cherry picking.
            for j in improved[:max(0, 128-len(examples[str(budget)]))]:
                points = value["points"][j]
                regs, pairs = set(), []
                actual = sum(_walk_segment(router.rg, *s, *t, regs, pairs) for s, t in zip(points[:-1], points[1:]))
                assert actual == io[j]
                row["checked_paths"] += 1
                examples[str(budget)].append({"net_id": int(ids[j]), "points": points.tolist(),
                    "baseline_io": int(base_io[j]), "io": int(io[j]),
                    "baseline_wirelength": float(base_wl[j]), "wirelength": float(wl[j])})
        if (lo // args.chunk_size) % 32 == 0:
            print(f"processed {hi}/{nl.num_nets} nets in {time.perf_counter()-started:.1f}s", flush=True)
    for row in rows.values():
        row["io_delta_pct"] = 100 * (row["io"] / row["baseline_io"] - 1) if row["baseline_io"] else 0.
        row["wirelength_delta_pct"] = 100 * (row["wirelength"] / row["hpwl"] - 1) if row["hpwl"] else 0.
        row["hpwl_delta_pct"] = 0.
    result = {"rows": rows, "excluded_non_degree_two_nets": omitted_degree,
              "excluded_outside_die_nets": outside, "routing_s": time.perf_counter()-started,
              "host_hwm_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "examples.json").write_text(json.dumps(examples, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
