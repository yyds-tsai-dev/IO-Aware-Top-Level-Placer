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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ioplace.route_eval.route_crossings import evaluate_route_from_files  # noqa: E402


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    result = evaluate_route_from_files(
        args.segments, args.regions, args.netmap, args.coord, delta=args.delta)

    out = result_to_json(result, segments_npz=args.segments, regions_json=args.regions,
                          netmap_json=args.netmap, coord_json=args.coord)

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
