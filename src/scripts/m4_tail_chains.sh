#!/bin/bash
# M4 tail: after S8 place finishes, run synthetic seed-B matrix, then the
# measurement probes (T1b arms/sampler, T2b lifetime). Idempotent; every
# stage skips outputs that already exist. Gate markers:
#   results/stage2/s8/place_chain.log : "stage2_s8_place.sh: all done" | "DONE WITH FAILURES"
#   this log                          : SYNTH_SEEDB_DONE_V3, MEASUREMENT_TAIL_DONE
DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace
REPO=/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer
PY=$DP/.venv312/bin/python
PROF=$REPO/results/m4/profile

echo "m4-tail queued $(date +%F' '%T); gating on stage2_s8_place completion"
while ! grep -q "stage2_s8_place.sh: all done\|DONE WITH FAILURES" $REPO/results/stage2/s8/place_chain.log 2>/dev/null; do sleep 300; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 60; done
echo "gate open $(date +%F' '%T)"

run_gp() {
  cfg=$1; k=$2; rho=$3; out=$4
  [ -s "$out" ] && { echo "skip $(basename $out) (exists)"; return; }
  echo "=== $(basename $out) start $(date +%F' '%T) ==="
  $PY -m ioplace.drivers.run_placement --config $cfg --mode io --k $k --rtype grid --seed 2000 --dp-seed 2000 --deterministic 1 --rho-max $rho --out $out > /dev/null 2>&1 && echo "OK $(basename $out) $(date +%T)" || echo "FAILED $(basename $out) $(date +%T)"
}
run_eval32() {
  cfg=$1; npz=$2; out=$3
  [ -s "$out" ] && { echo "skip $(basename $out) (exists)"; return; }
  echo "=== $(basename $out) eval-only start $(date +%T) ==="
  $PY -m ioplace.drivers.evaluate_placement --npz $npz --config $cfg --k 32 --rtype grid --seed 2000 --out $out > /dev/null 2>&1 && echo "OK $(basename $out) $(date +%T)" || echo "FAILED $(basename $out) $(date +%T)"
}

for spec in "synthetic_1x2_n2 $REPO/benchmarks/ispd25/synthetic_1x2_n2.json" "synthetic_2x2_n2 $REPO/benchmarks/ispd25/synthetic_2x2_n2.json"; do
  set -- $spec; case=$1; cfg=$2
  run_gp $cfg 16 0.0  $PROF/${case}__k16__grid__flat.json
  run_eval32 $cfg $PROF/${case}__k16__grid__flat.json.npz $PROF/${case}__k32__grid__flat.json
  run_gp $cfg 16 0.20 $PROF/${case}__k16__grid__oursM2.json
  run_gp $cfg 32 0.20 $PROF/${case}__k32__grid__oursM2.json
done
echo "SYNTH_SEEDB_DONE_V3 $(date +%F' '%T)"

cd $REPO
if [ ! -s results/m4/profile/t1b_arms.json ]; then
  echo "=== t1b arms start $(date +%T) ==="
  PYTHONPATH=src timeout 3600 $PY -m ioplace.diagnostics.probes_m4.probe_t1b_arms --config $DP/install/test/ispd2005/adaptec1.json --iterations 100 --out results/m4/profile/t1b_arms.json && echo "OK t1b_arms $(date +%T)" || echo "FAILED t1b_arms rc=$?"
fi
if [ ! -s results/m4/profile/t1b_sampler.json ]; then
  echo "=== t1b sampler start $(date +%T) ==="
  PYTHONPATH=src timeout 1800 $PY -m ioplace.diagnostics.probes_m4.probe_t1b_sampler --interval-s 0.5 --size-gb 0.25 --n-repeats 20 --out results/m4/profile/t1b_sampler.json && echo "OK t1b_sampler $(date +%T)" || echo "FAILED t1b_sampler rc=$?"
fi
if [ ! -s results/m4/profile/lifetime_group__k32__ours_m2.json ]; then
  echo "=== t2b lifetime group start $(date +%T) ==="
  PYTHONPATH=src timeout 7200 $PY -m ioplace.diagnostics.probes_m4.probe_lifetime_gate --config benchmarks/ispd25/mempool_group.json --case group --k 32 --rtype grid --seed 1000 --rho-max 0.05 --of-on 2.0 --iterations 300 --diag-every 2 --out results/m4/profile/lifetime_group__k32__ours_m2.json && echo "OK lifetime_group $(date +%T)" || echo "FAILED lifetime_group rc=$?"
fi
if [ ! -s results/m4/profile/lifetime_cluster__k32__ours_m2.json ]; then
  echo "=== t2b lifetime cluster start $(date +%T) ==="
  PYTHONPATH=src timeout 14400 $PY -m ioplace.diagnostics.probes_m4.probe_lifetime_gate --config benchmarks/ispd25/mempool_cluster.json --case cluster --k 32 --rtype grid --seed 1000 --rho-max 0.05 --of-on 2.0 --iterations 300 --diag-every 2 --out results/m4/profile/lifetime_cluster__k32__ours_m2.json && echo "OK lifetime_cluster $(date +%T)" || echo "FAILED lifetime_cluster rc=$?"
fi
echo "MEASUREMENT_TAIL_DONE $(date +%F' '%T)"
