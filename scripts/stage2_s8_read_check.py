"""Stage 2 S8 prerequisite (`docs/superpowers/specs/2026-08-13-stage2-
innovus-calibration-plan.md` sec 10 S8 row): CPU-only `PlaceDB.read()` sanity
check for the three new L1 configs added alongside this script
(`benchmarks/ispd2015_{des_perf_1,matrix_mult_1,superblue19}_m4.json`).

**GPU is off-limits** while M4 is running on the shared GPU (this task's
explicit instruction), so this script forces CPU-only *before* any
torch/DREAMPlace import: `CUDA_VISIBLE_DEVICES=""` in the environment (belt)
plus `params.gpu = 0` after `Params.load()` (suspenders) -- neither
`PlaceDB.read()` nor `PlaceDB.initialize()` should ever try to touch a CUDA
device with both of those set, but this repo has no prior CUDA_VISIBLE_
DEVICES precedent to point at (checked: no `ioplace/`/`tests/`/`scripts/`
file uses it), so both knobs are applied directly here rather than assumed
from elsewhere.

Deliberately narrower than `ioplace/diagnostics/probes_m4/
probe_corpus_stats.py` (which this borrows its area_util/free_area formulas
from): no RSS/sha256/target_density-suggestion bookkeeping, no `probes_m4`
edits (out of scope for this task -- another agent owns that directory).
Just num_nodes/num_nets/num_pins/die/read_s per config, written as one
combined JSON keyed by case name.

Usage:
    CUDA_VISIBLE_DEVICES= /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/.venv312/bin/python \\
        scripts/stage2_s8_read_check.py \\
        --config benchmarks/ispd2015_des_perf_1_m4.json --case des_perf_1 \\
        --config benchmarks/ispd2015_matrix_mult_1_m4.json --case matrix_mult_1 \\
        --config benchmarks/ispd2015_superblue19_m4.json --case superblue19 \\
        --out results/stage2/s8/read_check.json

(the script also sets CUDA_VISIBLE_DEVICES="" itself at import time, so the
env var prefix above is belt-and-suspenders, not strictly required)
"""
import argparse
import json
import os
import sys
import time

# Must happen before any torch/DREAMPlace import (belt; params.gpu=0 below
# is the suspenders).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ioplace.dreamplace_env import setup_dreamplace  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"


def read_check_one(config_json, case):
    root = setup_dreamplace()
    import Params
    import PlaceDB
    params = Params.Params()
    # Same chdir bracket as probe_corpus_stats.py/run_placement.py's
    # _load_dreamplace: Params.load()/PlaceDB.read() resolve *relative*
    # paths against cwd; this repo's own configs use absolute paths (see
    # this task's benchmarks/ispd2015_*_m4.json), but DREAMPlace's own
    # relative-path convention elsewhere means the chdir bracket is kept
    # for consistency/safety regardless.
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))
    try:
        params.load(config_json)
        # Force CPU-only (suspenders -- see module docstring).
        params.gpu = 0

        t0 = time.time()
        db = PlaceDB.PlaceDB()
        db.read(params)
        read_s = time.time() - t0
    finally:
        os.chdir(cwd)

    return dict(
        case=case,
        config=config_json,
        gpu=int(params.gpu),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        read_s=read_s,
        num_physical_nodes=int(db.num_physical_nodes),
        num_movable_nodes=int(db.num_movable_nodes),
        num_terminals=int(db.num_terminals),
        num_terminal_NIs=int(db.num_terminal_NIs),
        num_nets=int(db.num_nets),
        num_pins=int(len(db.pin2node_map)),
        die=[float(db.xl), float(db.yl), float(db.xh), float(db.yh)],
        row_height=float(db.row_height),
        site_width=float(db.site_width),
        target_density_configured=float(params.target_density),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True,
                     help="DREAMPlace params JSON, repeatable; paired "
                          "positionally with --case.")
    ap.add_argument("--case", action="append", required=True,
                     help="case name, repeatable; same count/order as --config.")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "stage2",
                                                    "s8", "read_check.json"))
    args = ap.parse_args(argv)

    if len(args.config) != len(args.case):
        raise SystemExit(f"--config given {len(args.config)} times but "
                          f"--case given {len(args.case)} times; must match 1:1")

    results = {}
    for config_json, case in zip(args.config, args.case):
        config_json = os.path.abspath(config_json)
        print(f"[read_check] {case}: reading {config_json} ...")
        r = read_check_one(config_json, case)
        results[case] = r
        print(f"[read_check] {case}: num_movable={r['num_movable_nodes']} "
              f"num_nets={r['num_nets']} num_pins={r['num_pins']} "
              f"read_s={r['read_s']:.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f"[read_check] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
