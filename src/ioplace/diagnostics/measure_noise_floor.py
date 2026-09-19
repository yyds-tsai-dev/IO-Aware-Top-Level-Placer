"""design v2 sec 7.2: measure sigma_rep in both deterministic regimes, pick one.

The flat baseline is produced by run_io in *observer mode* (rho_max=0, rho_margin=0),
which T5's test_rho_zero_is_observer_mode_and_bit_identical_to_flat proves is
bit-identical to run_flat while additionally emitting io_gp / lg_loss / hard_lambda_sum
-- T7's F5 gate needs lg_loss_flat.
"""
import json, os, statistics
from ioplace.drivers.run_placement_io import run_io

CFG = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/test/ispd2005/adaptec1.json"
OUT = "results/m2/noise"


def _flat(out, dp_seed, det):
    return run_io(CFG, 16, "grid", 0, out, rho_max=0.0, rho_margin=0.0,
                  every=50, dp_seed=dp_seed, deterministic=det)


def _stats(rs):
    io = [r["io_count"] for r in rs]
    return (statistics.mean(io),
            statistics.stdev(io) if len(io) > 1 else 0.0,
            statistics.mean(r["runtime_s"] for r in rs))


def main():
    os.makedirs(OUT, exist_ok=True)
    reps = {det: [_flat(f"{OUT}/flat_det{det}_rep{i}.json", 1000, det) for i in range(5)]
            for det in (0, 1)}
    m0, s0, t0 = _stats(reps[0])
    m1, s1, t1 = _stats(reps[1])
    ok = (s1 / m1 < 0.002) and ((t1 - t0) / t0 < 0.20)
    regime = 1 if ok else 0
    # the seed sweep runs in the regime we just picked, so sigma_seed is comparable
    seeds = [_flat(f"{OUT}/flat_seed{sd}.json", sd, regime) for sd in range(1000, 1005)]
    ms, ss, _ = _stats(seeds)
    base = reps[regime]
    summary = {"sigma_rep_det0": s0, "mean_det0": m0, "runtime_det0": t0,
               "sigma_rep_det1": s1, "mean_det1": m1, "runtime_det1": t1,
               "sigma_seed": ss, "mean_seed": ms,
               "sigma_rep": s1 if regime else s0,
               "flat_io": m1 if regime else m0,
               "flat_hpwl": statistics.mean(r["hpwl"] for r in base),
               "lg_loss_flat": statistics.mean(r["lg_loss"] for r in base),
               "regime": regime,
               "reason": ("det=1: sigma_rep/mean < 0.2% and runtime penalty < 20%" if ok
                          else "det=0: det=1 failed the sigma or the runtime gate")}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
