"""Stage 2 S8 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
plan.md` sec 10 S8 row): thin CLI wrapper around `ioplace.route_eval.
route_crossings.evaluate_route_from_files` (S3's real interface -- that
module has no `__main__` of its own, it is a library consumed from Python;
this script is the CLI shape `stage2_s8_extract.sh` calls).

Runs under `$DP/.venv312` (numpy-only, no odb/torch needed -- see
`route_crossings.py`'s own module docstring/imports), unlike
`dump_segments.py`/`verify_routed_def.py` which need OpenROAD's embedded
Python.

sec 10 S8 row's "K16 與 K32 各抽取一次(K 只影響 region 格)": the region
grid (`--regions`) is a pure function of (case, K, rtype) -- die geometry +
lattice, independent of which placement arm produced the routed wire being
measured (`ioplace/export/def_export.py`'s regions.json docstring: "die +
lattice + the K region rects used for this run", no placement-dependent
data). So `stage2_s8_extract.sh` calls this script once per (arm's routed
segments, K) pair, passing --regions from whichever arm directory actually
ran with that K in this case (not necessarily the same arm the segments
came from) while always passing --netmap/--coord from the *segments'own*
arm directory (those two are net-index-order / coordinate-mapping data,
which must match the specific PlaceDB.read() that produced the segments'
net indices -- safe to assume identical across arms of the same case in
principle, but this script does not rely on that assumption).

Usage:
    $DP/.venv312/bin/python scripts/stage2_s8_extract_crossings.py \\
        --segments .../segments.npz \\
        --regions  .../regions.json \\
        --netmap   .../netmap.json \\
        --coord    .../coord.json \\
        --delta 2 \\
        --out crossings_k16.json
"""
import argparse
import hashlib
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ioplace.route_eval.route_crossings import evaluate_route_from_files  # noqa: E402
from ioplace.route_eval.route_crossings import load_net_order, region_grid_from_json
from ioplace.export.evaluation import load_evaluation, pin_regions_from_evaluation
from ioplace.route_eval.segments import load_segments


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_route_evidence(evidence, routed_def, segments, provenance_path, coord=None, netmap=None):
    """Bind the decoded wires and pin memberships to the same actual DEF."""
    if not routed_def or not provenance_path:
        raise ValueError("paired evaluator requires --routed-def and --segments-provenance")
    with open(provenance_path) as stream:
        provenance = json.load(stream)
    expected = sha256_file(routed_def)
    ev = evidence["metadata"]["provenance"]
    if ev.get("placement_stage") != "router_def" or ev.get("router_def_sha256") != expected:
        raise ValueError("evaluator was not recomputed from this routed DEF")
    for path,key in ((coord,"coord_sha256"),(netmap,"netmap_sha256")):
        if path is None or sha256_file(path) != ev.get(key):
            raise ValueError(f"evaluator {key} does not match route extraction")
    if (provenance.get("routed_def_sha256") != expected
            or provenance.get("segments_sha256") != sha256_file(segments)
            or not provenance.get("verified")):
        raise ValueError("decoded segments provenance does not match verified routed DEF")
    return expected


def routing_population(segments, net_names, evidence):
    lookup={str(name):i for i,name in enumerate(segments.net_names)}
    wire=np.zeros(len(segments.net_names),dtype=bool)
    positive=segments.wire_mask() & (segments.segment_length()>0)
    wire[segments.seg_net_id[positive]]=True
    routed=np.asarray([wire[lookup[name]] if name in lookup else False for name in net_names])
    excluded=set(evidence["metadata"]["provenance"].get("pin_geometry_metadata",{}).get("excluded_special_net_names",[]))
    signal=np.asarray([name not in excluded for name in net_names])
    degrees=evidence["net_degrees"]
    requires_wire=signal & (degrees>=2)
    eligible=requires_wire & routed & (degrees<=evidence["metadata"]["max_degree"])
    missing=requires_wire & ~routed
    return dict(per_net_has_routed_wire=routed.tolist(),
        excluded_special_net_indices=np.flatnonzero(~signal).tolist(),
        unrouted_net_indices=np.flatnonzero(missing).tolist(),
        per_net_calibration_eligible=eligible.tolist(),
        signal_nets_requiring_wire=int(requires_wire.sum()),
        unrouted_signal_fraction=float(missing.sum()/requires_wire.sum()) if requires_wire.any() else 0.,
        unrouted_signal_gate_pass=bool(missing.sum()<=.02*requires_wire.sum()))


def result_to_json(result, *, segments_npz, regions_json, netmap_json, coord_json):
    return dict(
        segments=segments_npz, regions=regions_json, netmap=netmap_json, coord=coord_json,
        delta=result.delta,
        total_route_cross_raw=result.total_route_cross_raw,
        total_route_cross_dw=result.total_route_cross_dw,
        total_route_wl=result.total_route_wl,
        num_nets=int(len(result.route_cross_raw)),
        num_unmatched_nets=len(result.unmatched_net_indices),
        unmatched_net_indices=[int(i) for i in result.unmatched_net_indices],
        skipped_non_manhattan=int(result.skipped_non_manhattan),
        # per-net arrays, aligned to placedb net_index (sec 7.2) -- kept as
        # plain lists (not numpy) so this JSON has no numpy-specific
        # decoding requirement downstream (S9's calibration analysis, or a
        # human).
        per_net_route_cross_raw=[int(v) for v in result.route_cross_raw],
        per_net_route_cross_dw=[int(v) for v in result.route_cross_dw],
        per_net_lambda_route=[int(v) for v in result.lambda_route],
        per_net_route_ft=[int(v) for v in result.route_ft],
        per_net_route_wl=[int(v) for v in result.route_wl],
        # dict keys must be strings for JSON -- "a,b" (a<b, matches
        # RouteEvalResult.route_pair_demand's (min,max) tuple convention).
        route_pair_demand={f"{a},{b}": int(c)
                            for (a, b), c in result.route_pair_demand.items()},
        # sec 7.4 identity check (S3's own random-test invariant, re-stated
        # here at the total level as a cheap sanity gate any consumer of
        # this JSON can re-verify without re-running the extraction):
        # raw >= dw >= 0 at the total level.
        identity_raw_ge_dw=bool(result.total_route_cross_raw >= result.total_route_cross_dw),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True, help="segments.npz (S2 dump_segments.py output)")
    ap.add_argument("--regions", required=True, help="regions.json (S1 --emit-def sidecar)")
    ap.add_argument("--netmap", required=True, help="netmap.json (S1 --emit-def sidecar)")
    ap.add_argument("--coord", required=True, help="coord.json (S1 --emit-def sidecar)")
    ap.add_argument("--delta", type=int, default=2,
                     help="route_cross_dw/lambda_route run-length filter threshold "
                          "(spec sec 7.4 primary-report value; default 2)")
    ap.add_argument("--evaluator", help="aligned evaluator NPZ for route FT and calibration")
    ap.add_argument("--routed-def")
    ap.add_argument("--segments-provenance")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    evidence = None
    if args.evaluator:
        evidence = load_evaluation(args.evaluator, net_names=load_net_order(args.netmap),
                                   rg=region_grid_from_json(args.regions))
        routed_sha = validate_route_evidence(evidence, args.routed_def, args.segments,
                                             args.segments_provenance,args.coord,args.netmap)
    result = evaluate_route_from_files(
        args.segments, args.regions, args.netmap, args.coord, delta=args.delta,
        pin_regions=pin_regions_from_evaluation(evidence) if evidence is not None else None)

    out = result_to_json(result, segments_npz=args.segments, regions_json=args.regions,
                          netmap_json=args.netmap, coord_json=args.coord)
    if evidence is not None:
        population=routing_population(load_segments(args.segments),load_net_order(args.netmap),evidence)
        out.update(population)
        for i,routed in enumerate(population["per_net_has_routed_wire"]):
            if not routed:
                out["per_net_route_ft"][i]=-1
        out.update(evaluator_file=os.path.abspath(args.evaluator),
                   evaluator_sha256=sha256_file(args.evaluator),
                   routed_def_sha256=routed_sha,
                   segments_sha256=sha256_file(args.segments),
                   segments_provenance_sha256=sha256_file(args.segments_provenance),
                   evaluator_region_sha256=evidence["metadata"]["region_sha256"],
                   evaluator_placement_sha256=evidence["metadata"]["placement_sha256"],
                   evaluator_net_order_sha256=evidence["metadata"]["net_order_sha256"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"[extract_crossings] {args.out}: "
          f"total_route_cross_raw={out['total_route_cross_raw']} "
          f"total_route_cross_dw={out['total_route_cross_dw']} "
          f"(delta={args.delta}) num_nets={out['num_nets']} "
          f"num_unmatched={out['num_unmatched_nets']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
