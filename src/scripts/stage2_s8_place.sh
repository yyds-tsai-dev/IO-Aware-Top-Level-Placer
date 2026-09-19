#!/bin/bash
# Stage 2 S8 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
# plan.md` sec 10 S8 row / sec 4.1 L1 corpus table): run the 5-case x
# 3-arm placement matrix that feeds the S8 calibration corpus --
#
#   cases: des_perf_1, matrix_mult_1, superblue19, superblue12 (L1),
#          mempool_tile_wrap (L2 first case)
#   arms:  flat        (--mode io --rho-max 0.0,           k=16)
#          ours_k16     (--mode io --rho-max 0.20,          k=16)
#          ours_k32     (--mode io --rho-max 0.20,          k=32)
#
# i.e. 15 `ioplace.drivers.run_placement` invocations. Each writes
# `results/stage2/s8/<case>__<arm>/metrics.json` (the run's own metrics
# JSON, --out) and `results/stage2/s8/<case>__<arm>/out.def` +
# `regions.json`/`netmap.json`/`coord.json` (Stage 2 S1 sidecar, --emit-def
# pointed at the same directory) -- the route script (stage2_s8_route.sh)
# consumes `out.def` next.
#
# **This script only runs the CPU+GPU global placement + legalization
# (`--mode io`, no router involved) -- it does not route anything.**
#
# GPU gate (this task's explicit instruction: M4 owns the GPU right now):
# before touching the GPU at all, this script blocks until
#   (a) `results/m4/synth_seedb_chain.log` contains the literal string
#       `SYNTH_SEEDB_DONE`, AND
#   (b) `nvidia-smi --query-gpu=memory.used` reports < 1500 MiB
# polling every $POLL_INTERVAL_S seconds (default 60) and re-checking both
# conditions each time (a transient dip under 1500 MiB while M4 is still
# mid-run does not satisfy the gate -- (a) must ALSO hold). Pass
# --gate-once to check both conditions exactly once and exit non-zero
# instead of blocking (useful to dry-test the gate itself without waiting).
#
# Idempotent: a case/arm whose `metrics.json` already exists is skipped
# (not re-run, not overwritten). Per-run stdout/stderr goes to its own log
# file under results/stage2/s8/logs/; this script's own stdout is a
# one-line-per-run progress trace. A failed run does not abort the batch --
# the remaining case/arm combinations still run, and the script exits
# non-zero at the end if anything failed (summary printed).
#
# Usage:
#   scripts/stage2_s8_place.sh                 # gate, then run all 15
#   scripts/stage2_s8_place.sh --gate-once      # just check the gate once
#   scripts/stage2_s8_place.sh --dry-run        # print the 15 commands, run nothing
set -uo pipefail

REPO="${IOPLACE_REPO:-/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer}"
SOURCE="$REPO/src"
if [ ! -d "$SOURCE/ioplace" ]; then SOURCE="$REPO"; fi
DP="${DREAMPLACE_ROOT:-/nashome/NVL4/vdalab/yyds-dev/DREAMPlace}"
PY="$DP/.venv312/bin/python"

OUT_ROOT="$REPO/results/stage2/s8"
LOG_DIR="$OUT_ROOT/logs"
GATE_LOG="$REPO/results/m4/synth_seedb_chain.log"
GPU_MEM_MAX_MIB=1500
POLL_INTERVAL_S=60

SEED=2000
DETERMINISTIC=1
RTYPE=grid
BENCHMARK_KIND=real

GATE_ONCE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --gate-once) GATE_ONCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "stage2_s8_place.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# case_name|config_path
CASE_LIST="des_perf_1|$REPO/benchmarks/ispd2015_des_perf_1_m4.json
matrix_mult_1|$REPO/benchmarks/ispd2015_matrix_mult_1_m4.json
superblue19|$REPO/benchmarks/ispd2015_superblue19_m4.json
superblue12|$REPO/benchmarks/ispd2015_superblue12_m4.json
mempool_tile_wrap|$REPO/benchmarks/ispd25/mempool_tile_wrap.json"

# arm_name|k|rho_max
ARM_LIST="flat|16|0.0
ours_k16|16|0.20
ours_k32|32|0.20"

gate_check_once() {
    # Prints a status line; returns 0 iff both conditions currently hold.
    if [ ! -f "$GATE_LOG" ]; then
        echo "[gate] $GATE_LOG does not exist yet"
        return 1
    fi
    if ! grep -q "SYNTH_SEEDB_DONE" "$GATE_LOG"; then
        echo "[gate] SYNTH_SEEDB_DONE not yet in $GATE_LOG (M4's synthetic-seedB chain still running)"
        return 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gate] nvidia-smi not found -- cannot verify GPU is free" >&2
        return 1
    fi
    USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "${USED_MIB:-}" ]; then
        echo "[gate] could not parse nvidia-smi memory.used output"
        return 1
    fi
    if [ "$USED_MIB" -ge "$GPU_MEM_MAX_MIB" ]; then
        echo "[gate] SYNTH_SEEDB_DONE present but GPU memory.used=${USED_MIB} MiB >= ${GPU_MEM_MAX_MIB} MiB"
        return 1
    fi
    echo "[gate] PASS: SYNTH_SEEDB_DONE present, GPU memory.used=${USED_MIB} MiB < ${GPU_MEM_MAX_MIB} MiB"
    return 0
}

gate_wait() {
    echo "== gate: waiting for SYNTH_SEEDB_DONE in $GATE_LOG AND GPU used < ${GPU_MEM_MAX_MIB} MiB =="
    while ! gate_check_once; do
        sleep "$POLL_INTERVAL_S"
    done
}

run_one() {
    case_name="$1"; config="$2"; arm="$3"; k="$4"; rho="$5"

    run_dir="$OUT_ROOT/${case_name}__${arm}"
    out_json="$run_dir/metrics.json"
    log_file="$LOG_DIR/${case_name}__${arm}.log"

    cmd=("$PY" "$SOURCE/ioplace/drivers/run_placement.py"
         --config "$config" --mode io --k "$k" --rtype "$RTYPE"
         --seed "$SEED" --dp-seed "$SEED" --deterministic "$DETERMINISTIC"
         --rho-max "$rho" --out "$out_json" --emit-def "$run_dir"
         --benchmark-kind "$BENCHMARK_KIND")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] ${case_name}/${arm}: PYTHONPATH=$SOURCE ${cmd[*]}"
        return 0
    fi

    if [ -f "$out_json" ]; then
        echo "[skip] ${case_name}/${arm}: $out_json already exists"
        return 0
    fi
    if [ ! -f "$config" ]; then
        echo "[FAIL] ${case_name}/${arm}: config not found: $config" >&2
        return 1
    fi

    mkdir -p "$run_dir" "$LOG_DIR"
    echo "[run]  ${case_name}/${arm}: k=$k rho-max=$rho -> $out_json (log: $log_file)"
    PYTHONPATH="$SOURCE" "${cmd[@]}" >"$log_file" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] ${case_name}/${arm}: exit $rc -- see $log_file" >&2
        return $rc
    fi
    echo "[done] ${case_name}/${arm}"
    return 0
}

mkdir -p "$OUT_ROOT" "$LOG_DIR"

if [ "$GATE_ONCE" -eq 1 ]; then
    gate_check_once
    exit $?
fi

if [ "$DRY_RUN" -eq 0 ]; then
    gate_wait
fi

FAILED=""
while IFS='|' read -r case_name config; do
    [ -z "$case_name" ] && continue
    while IFS='|' read -r arm k rho; do
        [ -z "$arm" ] && continue
        if ! run_one "$case_name" "$config" "$arm" "$k" "$rho"; then
            FAILED="$FAILED ${case_name}/${arm}"
        fi
    done <<< "$ARM_LIST"
done <<< "$CASE_LIST"

echo ""
if [ -n "$FAILED" ]; then
    echo "== stage2_s8_place.sh: DONE WITH FAILURES:$FAILED ==" >&2
    exit 1
fi
echo "== stage2_s8_place.sh: all done =="
