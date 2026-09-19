#!/bin/sh
# M4 T10 H100 handover runbook (design draft
# `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 6.4
# "H100 交接契約" / sec 7.1 T10 row: "單一指令:scripts/m4_run.sh <case> <K>
# <rtype> <arm>"). This is that single instruction, generalized into flags
# (config path / K / rho / seed / output) so the same script also serves
# the L4-side --dry-run rehearsal, not just the literal H100 invocation.
#
# What this script does:
#   1. environment check -- venv python present, torch importable + CUDA
#      available, GPU name/memory reported, host RAM >= 128 GB (design
#      draft T12 row's trigger threshold) or a warning pointing at T12's
#      numpy-only PlaceDB shim fallback;
#   2. prints (and, unless --dry-run, runs) the full `ioplace.drivers.
#      run_placement` command for a single "io" driver invocation (schema
#      v3 -- `--mode io` with `--rho-max` as the flat(0.0)/ours@M2(>0.0)
#      switch, matching the established T8b convention: see
#      `results/m4/t8b/cluster__k16__grid__flat.json`'s own "mode":"io",
#      "rho_max":0.0 fields -- "flat" has never been a distinct driver mode
#      here, it is io mode with the IO term weighted to zero);
#   3. on a non-zero exit from that command, prints the sec 2.2/7.0
#      three-state recording guidance instead of silently treating a crash
#      or OOM as just a bug -- an OOM can be a legitimate, informative
#      result at this scale (sec 2.2's "三態規則").
#
# Design principle this script follows literally (sec 6.4): "H100 上不做
# 任何設計決策" -- every parameter here has a default sourced from the L4
# side of the project; this script only executes and reports, it does not
# choose K/rho/seed on its own.
#
# POSIX sh (no bashisms: no arrays, no [[ ]], no <<<, no `local`) --
# runs under dash as well as bash, matching the shell this repo's tests
# invoke it with (`tests/test_m4_handoff.py`'s dry-run smoke test).

set -eu

# ---------------------------------------------------------------------------
# defaults (all overridable by flag; env vars only change repo/DP roots)
# ---------------------------------------------------------------------------
REPO="${IOPLACE_REPO:-/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer}"
SOURCE="$REPO/src"
if [ ! -d "$SOURCE/ioplace" ]; then SOURCE="$REPO"; fi
DP="${DREAMPLACE_ROOT:-/nashome/NVL4/vdalab/yyds-dev/DREAMPlace}"
PY="$DP/.venv312/bin/python"

# design draft T7 row: the 3x3 (27.7M) synthetic array's DREAMPlace config
# is expected at this conventional path, sibling to the already-generated
# synthetic_1x2_n2.json / synthetic_2x2_n2.json (`benchmarks/ispd25/`);
# T7's Bookshelf files themselves must be regenerated on the H100 node
# (sec 6.4: "30M Bookshelf 在目標機重新生成,不搬 5.2 GB"), not copied, so
# this default may not exist until that regeneration step has run there --
# override with --config to point at whatever was actually generated.
CONFIG="$REPO/benchmarks/ispd25/synthetic_3x3_n2.json"
K=16
RHO=0.0
SEED=1000
DP_SEED=""
RTYPE=grid
DETERMINISTIC=1
BENCHMARK_KIND=synthetic
OUT="$REPO/results/m4/t10/h100_3x3__k${K}__grid.json"
HOST_RAM_MIN_GB=128
DRY_RUN=0

# design draft T7 row's SKU literal (sec 6.3 E5's frozen "登錄內容"
# table) -- used only for an informational match/mismatch note, never a
# hard gate (this same script also runs the L4 --dry-run rehearsal).
EXPECTED_GPU_SUBSTR="H100"

usage() {
    cat <<'EOF'
Usage: m4_run.sh [options]

  --config PATH       DREAMPlace params JSON (default: benchmarks/ispd25/synthetic_3x3_n2.json)
  --k K                region count (default: 16)
  --rho RHO            IO term weight (--rho-max passthrough); 0.0 = flat arm,
                        >0.0 = ours@M2 arm (default: 0.0)
  --seed SEED           placement seed (default: 1000)
  --dp-seed SEED        DREAMPlace RNG seed (default: same as --seed)
  --rtype RTYPE          grid|slicing (default: grid)
  --deterministic 0|1     (default: 1)
  --benchmark-kind KIND   real|synthetic (default: synthetic -- the 27.7M
                           3x3 array is always synthetic per T7's provenance)
  --out PATH             output profile JSON path
  --dry-run               print the env check + the command that would run,
                           but do not execute the placement itself
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --rho|--rho-max) RHO="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dp-seed) DP_SEED="$2"; shift 2 ;;
        --rtype) RTYPE="$2"; shift 2 ;;
        --deterministic) DETERMINISTIC="$2"; shift 2 ;;
        --benchmark-kind) BENCHMARK_KIND="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "m4_run.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$DP_SEED" ] || DP_SEED="$SEED"

echo "== M4 T10 environment check =="

# -- 1. venv ------------------------------------------------------------
if [ ! -x "$PY" ]; then
    echo "FATAL: DREAMPlace venv python not found/executable at $PY" >&2
    echo "       (see docs/dev-env.md for the build/venv procedure)" >&2
    exit 1
fi
echo "venv python: $PY"

# -- 2. torch + CUDA + GPU name/memory -----------------------------------
# Single python one-liner, '|'-delimited so a POSIX `set --`/IFS split
# below can pull the fields apart without arrays or bash-only `read -a`.
if ! ENV_LINE=$("$PY" - <<'PYEOF'
import torch
avail = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if avail else "NONE"
mem_gb = (torch.cuda.get_device_properties(0).total_memory / 2**30) if avail else 0.0
print("%s|%s|%s|%.1f" % (torch.__version__, avail, name, mem_gb))
PYEOF
); then
    echo "FATAL: could not import torch / query a CUDA device in $PY" >&2
    exit 1
fi
OLD_IFS="$IFS"
IFS='|'
set -- $ENV_LINE
IFS="$OLD_IFS"
TORCH_VERSION="$1"
CUDA_AVAILABLE="$2"
GPU_NAME="$3"
GPU_MEM_GB="$4"

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "FATAL: torch.cuda.is_available() == False -- no usable GPU" >&2
    exit 1
fi
echo "torch: $TORCH_VERSION"
echo "GPU: $GPU_NAME (${GPU_MEM_GB} GB)"
case "$GPU_NAME" in
    *"$EXPECTED_GPU_SUBSTR"*) : ;;
    *)
        echo "WARNING: GPU name '$GPU_NAME' does not match the frozen SKU" >&2
        echo "         contract ('$EXPECTED_GPU_SUBSTR...', design draft sec 6.3 E5" >&2
        echo "         登錄內容 table) -- any comparison against results/m4/forecast/" >&2
        echo "         h100_prediction.json is invalid on this hardware; re-register" >&2
        echo "         the prediction rather than reusing it (spec: 預測無效,須重登錄)." >&2
        ;;
esac

# -- 3. host RAM (T12 trigger threshold) ---------------------------------
HOST_RAM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
HOST_RAM_GB=$((HOST_RAM_KB / 1024 / 1024))
echo "host RAM: ${HOST_RAM_GB} GB"
if [ "$HOST_RAM_GB" -lt "$HOST_RAM_MIN_GB" ]; then
    echo "WARNING: host RAM (${HOST_RAM_GB} GB) < ${HOST_RAM_MIN_GB} GB." >&2
    echo "         design draft T12 row triggers on this condition: a" >&2
    echo "         numpy-only PlaceDB shim is required before PlaceDB.read" >&2
    echo "         can be trusted not to OOM the host on the 27.7M case." >&2
    echo "         See T12 (ioplace/, not this script) before proceeding." >&2
fi

# ---------------------------------------------------------------------------
# the single ioplace.drivers.run_placement invocation (schema v3)
# ---------------------------------------------------------------------------
# --no-diag: matches the phase model `scripts/m4_forecast.py` assumes
# (design draft sec 6.1: "diagnostics 關閉" for the phases the E5 wall-time
# forecast is built from) -- a timing run for the H100 handover should not
# pay io_term.diagnostics()'s extra per-callback cost (sec 1.4 B2).
CMD="$PY $SOURCE/ioplace/drivers/run_placement.py \
--config $CONFIG --mode io --k $K --rtype $RTYPE \
--seed $SEED --dp-seed $DP_SEED --deterministic $DETERMINISTIC \
--rho-max $RHO --out $OUT --benchmark-kind $BENCHMARK_KIND --no-diag"

echo ""
echo "== command =="
echo "PYTHONPATH=$SOURCE $CMD"

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "== DRY RUN: not executing =="
    exit 0
fi

if [ ! -f "$CONFIG" ]; then
    echo "FATAL: --config file does not exist: $CONFIG" >&2
    echo "       (the 27.7M Bookshelf array + its DREAMPlace config must be" >&2
    echo "       regenerated on this node first -- design draft sec 6.4: \"30M" >&2
    echo "       Bookshelf 在目標機重新生成,不搬 5.2 GB\")" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"

set +e
PYTHONPATH="$SOURCE" $CMD
RC=$?
set -e

if [ "$RC" -ne 0 ]; then
    echo "" >&2
    echo "== RUN FAILED (exit $RC) ==" >&2
    echo "Do not treat this as just a bug -- at 27.7M scale a crash or OOM" >&2
    echo "can be a legitimate, informative result (design draft sec 2.2 /" >&2
    echo "sec 7.0 RESULT GATE). Before filing this as a blocker, classify" >&2
    echo "and record it under the three-state feasibility rule" >&2
    echo "(ioplace/bench/result_gate.py: feasibility_verdict() / assert_budget()):" >&2
    echo "" >&2
    echo "  feasible_l4_contract    -- completed, measured peak <= this" >&2
    echo "                             SKU's HW_BUDGET_GB contract" >&2
    echo "  infeasible_l4_contract  -- OOM reproduced twice under the same" >&2
    echo "                             contract, OR completed but over" >&2
    echo "                             budget, OR the analytic resident" >&2
    echo "                             lower bound alone exceeds device" >&2
    echo "                             memory" >&2
    echo "  invalid_measurement     -- crash / contaminated run / an" >&2
    echo "                             as-yet-unreproduced OOM (retry up to" >&2
    echo "                             2x before calling it blocked_external)" >&2
    echo "" >&2
    echo "Keep the stderr above alongside a hand-recorded experiment_status/" >&2
    echo "workload_status note -- an unclassified failure is not a valid" >&2
    echo "artifact, but a classified one (even a crash) is." >&2
    exit "$RC"
fi

echo ""
echo "== done: $OUT =="
