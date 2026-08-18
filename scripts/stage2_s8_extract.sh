#!/bin/bash
# Stage 2 S8 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
# plan.md` sec 7 / sec 10 S8 row): for every `<case>__<arm>` run
# `stage2_s8_route.sh` produced a `or_run/routed.def` for, extract the
# routed-wire crossing statistics under both K=16 and K=32 region grids.
#
# Per run (one `results/stage2/s8/<case>__<arm>` directory with an
# `or_run/routed.def`):
#   1. `dump_segments.py --selfcheck` (S2, runs under `openroad -python` --
#      NOT this repo's venv, see that script's own docstring) decodes
#      `or_run/routed.def`'s wire geometry into `or_run/segments.npz` +
#      `or_run/segments.json`.
#   2. `verify_routed_def.py --json-out` (same OpenROAD-python requirement)
#      cross-checks (a) Sigma(route_wl)+RECT-reconciliation vs
#      `dbWire::getLength()` and (b) a def_text_parser sample, writing
#      `or_run/verify_s2.json` -- this is S2's own acceptance check, run
#      here as a standing sanity gate on every S8 routed DEF, not just the
#      one design S4's rehearsal already covered.
#   3. `stage2_s8_extract_crossings.py` (S3's `route_crossings.
#      evaluate_route_from_files`, runs under `$DP/.venv312` -- numpy-only,
#      no odb) extracts crossings TWICE per run, once per K, because K only
#      affects the region *grid* overlaid on the wire, not the wire itself
#      (sec 10 S8 row: "K 只影響 region 格"; region_grid.json is a pure
#      function of (case, K, rtype) -- see that script's own docstring for
#      the full argument): `<run>/crossings_k16.json` and
#      `<run>/crossings_k32.json`, both keyed to THIS run's own
#      netmap.json/coord.json (net-index order / coordinate mapping, which
#      must match the PlaceDB.read() that actually produced this run's
#      segments) but with `--regions` taken from whichever arm in the SAME
#      case actually ran at that K (K16 canonical source: the `flat` arm's
#      regions.json; K32 canonical source: the `ours_k32` arm's
#      regions.json -- both are placement-independent given the same
#      case+K+rtype, per def_export.py's regions.json docstring).
#
# Idempotent: a run whose crossings_k16.json AND crossings_k32.json both
# already exist is skipped entirely (dump_segments/verify are also skipped
# individually if their own output already exists). Non-fatal per-run
# failures do not abort the batch; exits non-zero at the end if anything
# failed.
#
# Usage:
#   scripts/stage2_s8_extract.sh                # process every or_run/routed.def found
#   scripts/stage2_s8_extract.sh --dry-run       # print what would run
set -uo pipefail

REPO="${IOPLACE_REPO:-/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer}"
DP="${DREAMPLACE_ROOT:-/nashome/NVL4/vdalab/yyds-dev/DREAMPlace}"
DP_PY="$DP/.venv312/bin/python"
OPENROAD="${OPENROAD_BIN:-openroad}"
S8_ROOT="$REPO/results/stage2/s8"

DUMP_SEGMENTS_PY="$REPO/ioplace/route_eval/or_scripts/dump_segments.py"
VERIFY_PY="$REPO/ioplace/route_eval/or_scripts/verify_routed_def.py"
EXTRACT_PY="$REPO/scripts/stage2_s8_extract_crossings.py"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "stage2_s8_extract.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

config_for_case() {
    case "$1" in
        des_perf_1) echo "$REPO/benchmarks/ispd2015_des_perf_1_m4.json" ;;
        matrix_mult_1) echo "$REPO/benchmarks/ispd2015_matrix_mult_1_m4.json" ;;
        superblue19) echo "$REPO/benchmarks/ispd2015_superblue19_m4.json" ;;
        superblue12) echo "$REPO/benchmarks/ispd2015_superblue12_m4.json" ;;
        mempool_tile_wrap) echo "$REPO/benchmarks/ispd25/mempool_tile_wrap.json" ;;
        *) return 1 ;;
    esac
}

# Which arm in a case is the canonical regions.json source for a given K
# (sec 10 S8 row: flat and ours_k16 both place at K=16, ours_k32 at K=32 --
# flat is the K16 source, ours_k32 the K32 source; arbitrary but fixed/
# documented choice between flat/ours_k16 for K16, see script docstring).
regions_arm_for_k() {
    case "$1" in
        16) echo "flat" ;;
        32) echo "ours_k32" ;;
        *) return 1 ;;
    esac
}

extract_one() {
    run_dir="$1"           # results/stage2/s8/<case>__<arm>
    base="$(basename "$run_dir")"
    case_name="${base%%__*}"
    or_run="$run_dir/or_run"
    routed_def="$or_run/routed.def"
    segments_npz="$or_run/segments.npz"
    verify_json="$or_run/verify_s2.json"

    config="$(config_for_case "$case_name")" || {
        echo "[FAIL] $base: unknown case '$case_name'" >&2
        return 1
    }
    if [ ! -f "$config" ]; then
        echo "[FAIL] $base: config not found: $config" >&2
        return 1
    fi
    mapfile -t lefs < <(jq -r '.lef_input[]' "$config")
    if [ "${#lefs[@]}" -eq 0 ]; then
        echo "[FAIL] $base: could not read lef_input from $config" >&2
        return 1
    fi

    have_k16="$run_dir/crossings_k16.json"
    have_k32="$run_dir/crossings_k32.json"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $base: dump_segments+verify($routed_def) then extract crossings k16,k32 -> $have_k16, $have_k32"
        return 0
    fi

    if [ ! -f "$routed_def" ]; then
        echo "[FAIL] $base: no routed.def (route not run yet?): $routed_def" >&2
        return 1
    fi

    if [ -f "$have_k16" ] && [ -f "$have_k32" ]; then
        echo "[skip] $base: crossings_k16.json and crossings_k32.json already exist"
        return 0
    fi

    lef_args=()
    for l in "${lefs[@]}"; do lef_args+=(--lef "$l"); done

    # --- step 1: dump_segments.py --selfcheck (openroad -python) -----------
    if [ ! -f "$segments_npz" ]; then
        dump_log="$or_run/dump_segments.log"
        echo "[run]  $base: dump_segments.py --selfcheck -> $segments_npz (log: $dump_log)"
        "$OPENROAD" -python "$DUMP_SEGMENTS_PY" "${lef_args[@]}" \
            --def "$routed_def" --out "$segments_npz" --selfcheck \
            --design-name "$base" >"$dump_log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$segments_npz" ]; then
            echo "[FAIL] $base: dump_segments.py exited $rc or produced no $segments_npz -- see $dump_log" >&2
            return 1
        fi
    else
        echo "[skip] $base: $segments_npz already exists"
    fi

    # --- step 2: verify_routed_def.py --json-out (openroad -python) --------
    if [ ! -f "$verify_json" ]; then
        verify_log="$or_run/verify_s2.log"
        echo "[run]  $base: verify_routed_def.py -> $verify_json (log: $verify_log)"
        "$OPENROAD" -python "$VERIFY_PY" "${lef_args[@]}" \
            --def "$routed_def" --report-out "$or_run/verify_wire_length.rpt" \
            --json-out "$verify_json" >"$verify_log" 2>&1
        rc=$?
        # `openroad -python` prints a SystemExit traceback and exits non-zero
        # even on sys.exit(0) (measured 2026-08-18); the JSON artifact is the
        # authoritative success signal, not the exit code.
        if [ ! -f "$verify_json" ]; then
            echo "[FAIL] $base: verify_routed_def.py (rc=$rc) produced no $verify_json -- see $verify_log" >&2
            return 1
        fi
        [ $rc -ne 0 ] && echo "[note] $base: verify rc=$rc ignored ($verify_json present; openroad-python SystemExit quirk)"
    else
        echo "[skip] $base: $verify_json already exists"
    fi

    # --- step 3: crossing extraction, once per K ($DP/.venv312) -------------
    netmap="$run_dir/netmap.json"
    coord="$run_dir/coord.json"
    if [ ! -f "$netmap" ] || [ ! -f "$coord" ]; then
        echo "[FAIL] $base: missing S1 sidecar(s) (netmap.json/coord.json) in $run_dir -- was this run's placement --emit-def complete?" >&2
        return 1
    fi

    step3_failed=0
    for k in 16 32; do
        out_json="$run_dir/crossings_k${k}.json"
        if [ -f "$out_json" ]; then
            echo "[skip] $base: $out_json already exists"
            continue
        fi
        regions_arm="$(regions_arm_for_k "$k")"
        regions_json="$S8_ROOT/${case_name}__${regions_arm}/regions.json"
        if [ ! -f "$regions_json" ]; then
            echo "[FAIL] $base: k=$k canonical regions.json not found: $regions_json (needs ${case_name}__${regions_arm}'s placement to have completed first)" >&2
            step3_failed=1
            continue
        fi
        extract_log="$or_run/extract_crossings_k${k}.log"
        echo "[run]  $base: extract crossings k=$k (regions from ${case_name}__${regions_arm}) -> $out_json (log: $extract_log)"
        "$DP_PY" "$EXTRACT_PY" --segments "$segments_npz" --regions "$regions_json" \
            --netmap "$netmap" --coord "$coord" --delta 2 --out "$out_json" \
            >"$extract_log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$out_json" ]; then
            echo "[FAIL] $base: extract crossings k=$k exited $rc or produced no $out_json -- see $extract_log" >&2
            step3_failed=1
        fi
    done

    if [ "$step3_failed" -ne 0 ]; then
        return 1
    fi
    echo "[done] $base"
    return 0
}

if [ "$DRY_RUN" -eq 0 ]; then
    for bin in "$OPENROAD" "$DP_PY" jq; do
        if ! command -v "$bin" >/dev/null 2>&1; then
            echo "FATAL: required binary not found on PATH: $bin" >&2
            exit 1
        fi
    done
fi

mapfile -t run_dirs < <(find "$S8_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*__*' 2>/dev/null | sort)

if [ "${#run_dirs[@]}" -eq 0 ]; then
    echo "stage2_s8_extract.sh: no results/stage2/s8/<case>__<arm> directories found -- nothing to extract (run stage2_s8_place.sh / stage2_s8_route.sh first)."
    exit 0
fi

echo "== stage2_s8_extract.sh: ${#run_dirs[@]} run directories found =="

FAILED=""
for run_dir in "${run_dirs[@]}"; do
    if ! extract_one "$run_dir"; then
        FAILED="$FAILED $(basename "$run_dir")"
    fi
done

echo ""
if [ -n "$FAILED" ]; then
    echo "== stage2_s8_extract.sh: DONE WITH FAILURES:$FAILED ==" >&2
    exit 1
fi
echo "== stage2_s8_extract.sh: all done =="
