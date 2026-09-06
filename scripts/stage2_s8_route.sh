#!/bin/bash
# Stage 2 S8 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
# plan.md` sec 6.1 F-OR skeleton / sec 10 S8 row): route every `out.def`
# produced by `stage2_s8_place.sh` (`results/stage2/s8/<case>__<arm>/
# out.def`) through OpenROAD global+detailed route, smallest file first.
#
# Per run (one `<case>__<arm>` directory):
#   1. VIA-dup fix (pure text, always safe to run): `scripts/
#      stage2_fix_def_vias.py` strips any `out.def` VIAS block whose name
#      is also defined in the design's own tech.lef (the LEF-vs-DEF
#      duplicate that blocks TritonRoute with DRT-0338 on this benchmark
#      family -- see that script's docstring). Writes `or_run/fixed.def` +
#      `or_run/fixed.def.diff`. If the DEF has no VIAS section at all (not
#      every design has DEF-side VIA redefinitions), the fix is a
#      documented no-op and routing proceeds on the *original* `out.def`
#      directly -- this is not an error.
#   2. `or_run/route.tcl`: read LEF (order taken verbatim from this case's
#      DREAMPlace config's own `lef_input`, tech first, matching whatever
#      order S8's placement config already uses) + the fixed/original DEF,
#      record the V1/V2 pre-route instance/net counts, `set_thread_count 8`,
#      `global_route -allow_congestion` (required on this benchmark family,
#      spec sec 6.1 note) + `detailed_route` (uncapped -- no
#      `-droute_end_iter`, unlike the S4 rehearsal's time-boxed run; the
#      12h `timeout` below is what bounds wall time here), `write_def
#      routed.def`.
#   3. `or_run/postprocess.tcl`: re-read the *input* DEF and `routed.def`
#      in a fresh session, report V1/V2 (#insts/#nets, exact match
#      required) and `report_wire_length` -> `or_run/wirelength.rpt`.
#
# Runs are limited to two concurrent CPU routing processes by the batch loop.
#
# Idempotent: a run whose `or_run/routed.def` already exists is skipped.
# Each run's OpenROAD stdout/stderr goes to its own log
# (`or_run/openroad_route.log` / `or_run/openroad_postprocess.log`); this
# script's own stdout is a one-line-per-run progress trace. A failed run
# (non-zero openroad exit, or a `timeout` kill) does not abort the batch;
# the script exits non-zero at the end if anything failed.
#
# Usage:
#   scripts/stage2_s8_route.sh                # route everything found under
#                                              #   results/stage2/s8/*/out.def
#   scripts/stage2_s8_route.sh --dry-run       # print the run order + what
#                                              #   would execute, run nothing
set -uo pipefail

REPO="${IOPLACE_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
S8_ROOT="${STAGE2_S8_ROOT:-$REPO/results/stage2/s8}"
OPENROAD="${OPENROAD_BIN:-openroad}"
PYTHON="${IOPLACE_PYTHON:-$(dirname -- "$REPO")/DREAMPlace/.venv312/bin/python}"
FIX_VIAS_PY="${STAGE2_FIX_VIAS_PY:-$PYTHON}"
# 4h + capped DR iterations: des_perf_1 measurement (2026-08-17) showed DRT
# plateaus at ~172k violations for hours on high-utilization ISPD2015 designs
# and a timeout kill leaves NO routed.def at all. Stage2's crossing ground
# truth needs complete wires, not DRC-clean routing (mgc_fft_1 precedent:
# 57.7k violations, 0 unrouted nets, S2/S3 acceptance green) - so cap DR at
# 5 optimization iterations, let it write the DEF, and disclose the final
# violation count in S9/the report instead of chasing convergence.
ROUTE_TIMEOUT_S="${STAGE2_ROUTE_TIMEOUT_S:-14400}"
THREADS="${STAGE2_ROUTE_THREADS:-8}"

DRY_RUN=0
SHARD_GLOB="*"
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        --shard=*) SHARD_GLOB="${arg#--shard=}" ;;
        *) echo "stage2_s8_route.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# case_name -> DREAMPlace config path (same table stage2_s8_place.sh uses;
# kept here too rather than sourcing that script, so each script stays
# independently invocable/testable).
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

route_one() {
    run_dir="$1"          # results/stage2/s8/<case>__<arm>
    base="$(basename "$run_dir")"
    case_name="${base%%__*}"
    out_def="$run_dir/out.def"
    or_run="$run_dir/or_run"
    routed_def="$or_run/routed.def"
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_pid="${BASHPID:-$$}"

    config=""
    if [ -f "$run_dir/metrics.json" ]; then
        config="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("config", ""))' "$run_dir/metrics.json")"
    fi
    if [ ! -f "$config" ]; then
      config="$(config_for_case "$case_name")" || {
        echo "[FAIL] $base: unknown case '$case_name' (no config_for_case entry)" >&2
        return 1
      }
    fi
    if [ ! -f "$config" ]; then
        echo "[FAIL] $base: config not found: $config" >&2
        return 1
    fi

    # LEF list, in the config's own order (tech.lef first by convention --
    # see stage2_s8_place.sh/the benchmark configs themselves).
    mapfile -t lefs < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["lef_input"]))' "$config")
    if [ "${#lefs[@]}" -eq 0 ]; then
        echo "[FAIL] $base: could not read lef_input from $config" >&2
        return 1
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $base: would route $out_def with $(IFS=,; echo "${lefs[*]}") -> $routed_def"
        return 0
    fi

    if [ -f "$routed_def" ]; then
        echo "[skip] $base: $routed_def already exists"
        return 0
    fi
    if [ ! -f "$out_def" ]; then
        echo "[FAIL] $base: out.def not found (placement not run yet?): $out_def" >&2
        return 1
    fi

    mkdir -p "$or_run"
    # Preserve artifacts from any earlier attempt before regenerating them.
    backup_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    for prior in openroad_route.log openroad_postprocess.log route.tcl postprocess.tcl routed.def route.guide congestion.rpt drc.rpt wirelength.rpt fixed.def fixed.def.diff fix_def_vias.log execution.json; do
        if [ -e "$or_run/$prior" ]; then
            mv "$or_run/$prior" "$or_run/$prior.previous.$backup_stamp"
        fi
    done

    # --- step 1: VIA-dup fix (pure text, always safe) ----------------------
    fixed_def="$or_run/fixed.def"
    fix_log="$or_run/fix_def_vias.log"
    lef_args=()
    for l in "${lefs[@]}"; do lef_args+=(--lef "$l"); done
    "$FIX_VIAS_PY" "$REPO/scripts/stage2_fix_def_vias.py" "${lef_args[@]}" \
        --def "$out_def" --out "$fixed_def" --diff "$or_run/fixed.def.diff" \
        >"$fix_log" 2>&1
    fix_rc=$?
    if [ $fix_rc -eq 0 ]; then
        def_for_route="$fixed_def"
    else
        echo "[note] $base: stage2_fix_def_vias.py exited $fix_rc (likely: no VIAS section in this DEF) -- routing original out.def unmodified. See $fix_log."
        def_for_route="$out_def"
    fi

    # --- step 2: route.tcl (skeleton: results/stage2/rehearsal/mgc_fft_1/or_run/route.tcl) ---
    route_tcl="$or_run/route.tcl"
    {
        echo "# S8 route: GR+DR $base to produce a routed DEF for crossing extraction."
        echo "# Skeleton: results/stage2/rehearsal/mgc_fft_1/or_run/route.tcl / spec sec 6.1."
        for l in "${lefs[@]}"; do echo "read_lef $l"; done
        echo "read_def $def_for_route"
        echo 'set block [[[ord::get_db] getChip] getBlock]'
        echo 'puts "COUNT_IN insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"'
        echo "set_thread_count $THREADS"
        echo "global_route -allow_congestion -congestion_report_file $or_run/congestion.rpt -guide_file $or_run/route.guide"
        echo "detailed_route -droute_end_iter 5 -output_drc $or_run/drc.rpt -verbose 1"
        echo "write_def $routed_def"
        echo 'puts "DONE_ROUTE"'
        echo "exit"
    } > "$route_tcl"

    route_log="$or_run/openroad_route.log"
    echo "[run]  $base: routing $def_for_route (timeout ${ROUTE_TIMEOUT_S}s, ${#lefs[@]} LEF(s)) -> $routed_def (log: $route_log)"
    timeout "$ROUTE_TIMEOUT_S" "$OPENROAD" -no_init "$route_tcl" >"$route_log" 2>&1
    route_rc=$?
    if [ "$route_rc" -eq 124 ]; then
        echo "[FAIL] $base: route.tcl timed out after ${ROUTE_TIMEOUT_S}s -- see $route_log" >&2
        return 1
    fi
    if [ "$route_rc" -ne 0 ] || [ ! -f "$routed_def" ]; then
        echo "[FAIL] $base: route.tcl exited $route_rc or did not produce $routed_def -- see $route_log" >&2
        return 1
    fi

    # --- step 3: independent OpenDB instance/net/endpoint identity --------
    lef_args=()
    for l in "${lefs[@]}"; do lef_args+=(--lef "$l"); done
    dump_pin="$REPO/ioplace/route_eval/or_scripts/dump_pin_geometry.py"
    "$OPENROAD" -python "$dump_pin" "${lef_args[@]}" --def "$def_for_route" \
        --out "$or_run/pin_geometry_input.npz" > "$or_run/input_identity.log" 2>&1 || return 1
    "$OPENROAD" -python "$dump_pin" "${lef_args[@]}" --def "$routed_def" \
        --out "$or_run/pin_geometry.npz" > "$or_run/output_identity.log" 2>&1 || return 1
    "$PYTHON" "$REPO/scripts/stage2_verify_identity.py" \
        --before "$or_run/pin_geometry_input.npz" --after "$or_run/pin_geometry.npz" \
        --out "$or_run/verify_identity.json" > "$or_run/openroad_postprocess.log" 2>&1 || return 1

    echo "[done] $base"
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$PYTHON" - "$or_run/execution.json" "$run_dir" "$config" "$started_at" "$finished_at" "$run_pid" "$OPENROAD" "$REPO" <<'PY'
import hashlib, json, os, sys
out, run_dir, config, started, finished, pid, binary, repo = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None
or_run = os.path.join(run_dir, 'or_run')
data = {
    'run_dir': run_dir, 'config': config, 'started_at_utc': started,
    'finished_at_utc': finished, 'pid': int(pid), 'status': 'completed',
    'tool_binary': binary, 'tool_binary_sha256': sha(binary),
    'source_repo': repo, 'source_head': None,
    'input_def_sha256': sha(os.path.join(or_run, 'fixed.def')) or sha(os.path.join(run_dir, 'out.def')),
    'artifacts': {name: sha(os.path.join(or_run, name)) for name in
                  ('route.guide', 'routed.def', 'congestion.rpt', 'drc.rpt', 'verify_identity.json')},
}
try:
    import subprocess
    data['source_head'] = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
except Exception:
    pass
with open(out, 'w') as f: json.dump(data, f, indent=2); f.write('\n')
PY
    return 0
}

if [ "$DRY_RUN" -eq 0 ]; then
    if ! command -v "$OPENROAD" >/dev/null 2>&1; then
        echo "FATAL: openroad not found on PATH (\$OPENROAD_BIN=$OPENROAD)" >&2
        exit 1
    fi
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo "FATAL: Python not found ($PYTHON)" >&2
        exit 1
    fi
fi

# Smallest out.def first (task instruction). `du -k` sorts numerically ascending.
# --shard=GLOB: only route arm dirs whose basename matches the glob (e.g.
# --shard='superblue19__*'), so several instances can run disjoint shards
# concurrently. Default: everything.
mapfile -t ordered_defs < <(
    find "$S8_ROOT" -mindepth 2 -maxdepth 2 -type f -name out.def -path "$S8_ROOT/$SHARD_GLOB/out.def" 2>/dev/null \
        -exec du -k {} \; | sort -n -k1,1 | cut -f2-
)

if [ "${#ordered_defs[@]}" -eq 0 ]; then
    echo "stage2_s8_route.sh: no out.def found under $S8_ROOT/*/out.def -- nothing to route (run stage2_s8_place.sh first)."
    exit 0
fi

echo "== stage2_s8_route.sh: ${#ordered_defs[@]} out.def found, routing smallest-first, concurrency=2 =="

FAILED=""
running=0
pids=()
names=()
for out_def in "${ordered_defs[@]}"; do
    run_dir="$(dirname "$out_def")"
    route_one "$run_dir" &
    pids+=("$!")
    names+=("$(basename "$run_dir")")
    running=$((running + 1))
    if [ "$running" -ge 2 ]; then
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then FAILED="$FAILED ${names[$i]}"; fi
        done
        pids=(); names=(); running=0
    fi
done
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then FAILED="$FAILED ${names[$i]}"; fi
done

echo ""
if [ -n "$FAILED" ]; then
    echo "== stage2_s8_route.sh: DONE WITH FAILURES:$FAILED ==" >&2
    exit 1
fi
echo "== stage2_s8_route.sh: all done =="
