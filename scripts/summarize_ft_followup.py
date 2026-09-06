"""Audit registered FT artifacts and report paired effects and P0c gates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def read_run(root, name):
    path = root / f"{name}.json"
    execution = json.loads(path.with_suffix(".execution.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if execution.get("returncode") != 0 or execution.get("output_sha256") != digest:
        raise ValueError(f"unverified execution: {name}")
    data = json.loads(path.read_text())
    if data.get("workload_status") != "completed":
        raise ValueError(f"incomplete workload: {name}")
    return data


def summarize(root):
    levers = {f"A{i}": read_run(root, f"A{i}_seed1000") for i in range(6)}
    flat = [read_run(root, f"flat_seed{seed}") for seed in range(1000, 1005)]
    metrics = ("io_count", "ft_count", "hpwl")
    noise = {key: dict(values=[v[key] for v in flat], mean=float(np.mean([v[key] for v in flat])),
                        std_seed=float(np.std([v[key] for v in flat], ddof=1))) for key in metrics}
    paired = {}
    for key in metrics:
        delta = np.asarray([read_run(root, f"A2_seed{s}")[key] - read_run(root, f"A0_seed{s}")[key]
                            for s in (1001, 1002, 1003)], dtype=float)
        upper = delta.mean() + t.ppf(.95, 2)*delta.std(ddof=1)/np.sqrt(3)
        paired[key] = dict(differences=delta.tolist(), mean=float(delta.mean()),
                           one_sided_95_upper=float(upper), n=3, seeds=[1001,1002,1003])
    p0 = read_run(root, "P0_seed1000")
    pilot = {}
    for i in range(4):
        name = f"P{i}"
        data = read_run(root, f"{name}_seed1000")
        cancellation = [e["cancellation_ratio"] for e in data["trajectory"]
                        if e.get("applied_ft_force_l1", 0) > 0]
        log = (root / f"{name}_seed1000.log").read_text(errors="replace")
        pg1 = data["ft_count"] <= p0["ft_count"] - 62.9
        pg2 = data["io_count"] <= p0["io_count"] + 492 and data["hpwl"] <= 1.005*p0["hpwl"]
        pg4 = (data["legalization_status"] == "success" and data["num_unplaced_cells"] == 0
               and data["stop_overflow_reached"] and data["backtrack_median"] < 5
               and data["hpwl"] <= 1.1*p0["hpwl"]
               and not any(e.get("kappa_clamped", False) for e in data["trajectory"])
               and "DIVERGENCE detected" not in log
               and (min(cancellation) >= .3 if cancellation else i == 0))
        pilot[name] = dict(metrics={key:data[key] for key in metrics},
            PG1=pg1 if i else None, PG2=pg2, PG4_observed_checks=pg4,
            PG4_hpwl_scope="final relative HPWL (G8 delta metric); no per-iteration HPWL trace claimed",
            min_active_cancellation=min(cancellation) if cancellation else None,
            net_chain_to_star=data["net_chain_to_star"],
            net_chain_to_star_minus_P0=data["net_chain_to_star"]-p0["net_chain_to_star"],
            eligible_for_confirmation=bool(i and pg1 and pg2 and pg4))
    return dict(scope="adaptec1 K16, current DREAMPlace build, shared H100 NVL",
        source_manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        historical_M3_verdict="FAIL (unchanged)",
        levers={name:{key:data[key] for key in (*metrics,"runtime_s","legalization_status")}
                for name,data in levers.items()},
        flat_seed_noise=noise, paired_A2_minus_A0=paired, pilot=pilot,
        pilot_confirmation_required=any(v["eligible_for_confirmation"] for v in pilot.values()),
        caveats=["WL reweight improvement is relative to IO-only; FT remains above the flat baseline.",
                 "Three confirmation seeds and one design do not establish cross-design generalization.",
                 "P0c FT reductions violate the registered IO guard; no joint-quality pass.",
                 "Shared-device runtimes are not exclusive GPU performance measurements."])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    result = summarize(root)
    out = root / "summary.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(out)


if __name__ == "__main__":
    main()
