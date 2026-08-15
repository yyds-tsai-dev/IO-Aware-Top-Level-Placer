"""Assembles verify_group2x2_{n1,n2}.json + t6_summary.json from the
already-computed pieces: cluster_stats.json, rent_cluster_result.json,
rent_synthetic_result.json, h4_result.json, kernel_sensitivity.json,
build_summary.json."""
import json
import os
import sys

import numpy as np

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
sys.path.insert(0, REPO)
from ioplace.bench import verify_bench as vb  # noqa: E402

with open("/tmp/rent_cluster_result.json") as f:
    cluster = json.load(f)
with open("/tmp/rent_synthetic_result.json") as f:
    synth = json.load(f)
with open("/tmp/h4_result.json") as f:
    h4 = json.load(f)
with open("/tmp/kernel_sensitivity.json") as f:
    kernel_sens = json.load(f)
with open(os.path.join(REPO, "results/m4/bench/cluster_stats.json")) as f:
    cluster_stats = json.load(f)
with open(os.path.join(REPO, "results/m4/bench/arrays/build_summary.json")) as f:
    build_summary = json.load(f)

cluster_degrees = np.load("/tmp/rent_cluster_net_degrees.npy")


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def h1(norm):
    p_syn = synth[f"2x2_{norm}"]["p"]
    p_real = cluster["p"]
    return vb.check_h1_rent_shape(p_synthetic=p_syn, p_real=p_real)


def h2(norm):
    syn_levels = {l["level"]: l["avg_terminals"] for l in synth[f"2x2_{norm}"]["levels"]
                  if 1e3 <= l["avg_block_size"] <= 1e6}
    real_levels = {l["level"]: l["avg_terminals"] for l in cluster["levels"]
                   if 1e3 <= l["avg_block_size"] <= 1e6}
    return vb.check_h2_cut_histogram(synthetic_levels=syn_levels, real_levels=real_levels)


def h3(norm):
    syn_degrees = np.load(f"/tmp/rent_synth_2x2_{norm}_net_degrees.npy")
    return vb.check_h3_degree_ks(synthetic_degrees=syn_degrees, real_degrees=cluster_degrees,
                                  ks_threshold=0.05, bucket_rel_tol=0.20)


def h4_gate(shape, norm):
    d = h4[f"{shape}_{norm}"]
    ks_threshold = 0.10
    ok = d["ks"] is not None and d["ks"] <= ks_threshold
    return {"status": "ok" if ok else "fail", "metric": "H4_self_consistency",
            "ks": d["ks"], "threshold": ks_threshold, "detail": d}


def h6(norm):
    p1 = synth["1x1_source"]["p"]
    p2 = synth[f"1x2_{norm}"]["p"]
    p3 = synth[f"2x2_{norm}"]["p"]
    threshold = 0.03
    pairs = {"1x1_vs_1x2": abs(p1 - p2), "1x1_vs_2x2": abs(p1 - p3), "1x2_vs_2x2": abs(p2 - p3)}
    ok = all(v <= threshold for v in pairs.values())
    return {"status": "ok" if ok else "fail", "metric": "H6_rent_invariance",
            "p_1x1_source": p1, f"p_1x2_{norm}": p2, f"p_2x2_{norm}": p3,
            "pairwise_abs_diff": pairs, "threshold": threshold}


def diagnostics(shape, norm):
    """V0/H5' -- diagnostic only, never gate (adjudication doc: '診斷不判定:
    H5、H5′、V0/V1'). V1 is not run here at all (DREAMPlace's own PlaceDB
    reader cannot load this project's hierarchical-name Bookshelf exports
    at all -- see rent.py module docstring's M4 T6 finding -- so V1 is
    recorded as infrastructure_blocked rather than executed)."""
    prefix = os.path.join(REPO, "results/m4/bench/arrays", f"{shape}_{norm}", f"{shape}_{norm}")
    with open(prefix + ".manifest.json") as f:
        manifest = json.load(f)
    source_prefix = os.path.join(REPO, "results/m4/bench/mempool_group_export/mempool_group")
    v0 = vb.check_v0_structural(prefix, manifest=manifest, source_prefix=source_prefix)
    h5p = vb.check_h5prime_self_consistency(manifest)
    return {"V0_structural": v0, "H5prime_self_consistency": h5p,
            "V1_dreamplace_readable": {
                "status": "infrastructure_blocked",
                "reason": "DREAMPlace's own PlaceDB.read_pl uses a hardcoded \\w+ node-name "
                          "regex that cannot match this project's hierarchical (./[]/-"
                          "containing) instance names -- confirmed on mempool_group itself, "
                          "independent of any T6-specific construction; see rent.py module "
                          "docstring's M4 T6 finding. Diagnostic-only per adjudication doc "
                          "sec 4 (does not gate T6/T7).",
            }}


verdicts = {}
for norm in ("n1", "n2"):
    gates = {
        "H1_rent_shape": h1(norm),
        "H2_cut_histogram": h2(norm),
        "H3_degree_ks": h3(norm),
        "H4_self_consistency": h4_gate("2x2", norm),
        "H6_rent_invariance": h6(norm),
    }
    all_ok = all(g["status"] == "ok" for g in gates.values())
    out = {
        "normalization": norm,
        "fit_metrics": {
            "lambda_0": cluster_stats["mom"]["lambda_0"],
            "lambda_0_tile": cluster_stats["lambda_0_tile"]["value"],
            "alpha": cluster_stats["mom"]["alpha"],
            "alpha_95ci": cluster_stats.get("net_level_bootstrap", {}).get("alpha_ci"),
            "lambda_0_95ci": cluster_stats.get("net_level_bootstrap", {}).get("lambda_0_ci"),
            "overdispersion_var_over_mean": cluster_stats["overdispersion_var_over_mean"],
            "overdispersion_unreliable": cluster_stats["overdispersion_unreliable"],
            "six_pair_raw_counts": cluster_stats["six_pair_raw_counts"],
            "ngroups_hist_real": cluster_stats["ngroups_hist"],
            "glue_degree_note": "synthetic glue nets are degree-2-only (known simplification, "
                                 "see build_t6_arrays.py); real cross-group nets are 96.2% "
                                 "degree-2 (touch exactly 2 groups), 3.8% touch 3-4 -- not "
                                 "reproduced here (fit metric only, not gating)",
        },
        "rent_p": {
            "1x1_source": synth["1x1_source"]["p"],
            f"1x2_{norm}": synth[f"1x2_{norm}"]["p"],
            f"2x2_{norm}": synth[f"2x2_{norm}"]["p"],
            "real_cluster": cluster["p"],
            "real_cluster_ci": [cluster["p_ci_lo"], cluster["p_ci_hi"]],
        },
        "holdout_gates": gates,
        "verdict": "PASS" if all_ok else "FAIL",
        "diagnostics_not_gating": diagnostics("2x2", norm),
        "build_manifest": build_summary.get("2x2", {}).get(norm),
    }
    verdicts[norm] = out
    out_path = os.path.join(REPO, f"results/m4/bench/verify_group2x2_{norm}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=_json_default)
    print(f"[{norm}] verdict={out['verdict']} gates={[k+':'+v['status'] for k,v in gates.items()]}")

selected = None
if verdicts["n2"]["verdict"] == "PASS":
    selected = "n2"
elif verdicts["n1"]["verdict"] == "PASS":
    selected = "n1"

summary = {
    "shape": "2x2",
    "n1_verdict": verdicts["n1"]["verdict"],
    "n2_verdict": verdicts["n2"]["verdict"],
    "n1_gates": {k: v["status"] for k, v in verdicts["n1"]["holdout_gates"].items()},
    "n2_gates": {k: v["status"] for k, v in verdicts["n2"]["holdout_gates"].items()},
    "selected_normalization": selected,
    "t7_gate": "PASS" if selected else "FAIL",
    "rent_p": {
        "1x1_source": synth["1x1_source"]["p"],
        "1x2_n1": synth["1x2_n1"]["p"], "1x2_n2": synth["1x2_n2"]["p"],
        "2x2_n1": synth["2x2_n1"]["p"], "2x2_n2": synth["2x2_n2"]["p"],
        "real_cluster": cluster["p"], "real_cluster_ci": [cluster["p_ci_lo"], cluster["p_ci_hi"]],
    },
    "kernel_sensitivity_glue_counts": kernel_sens,
    "mtkahypar_feasibility": {
        "mempool_group_3.5M_scale": "feasible: full recursive bisection to 3 levels in 42.7s, "
                                     "peak RSS 3.33GB (no net-degree cap needed)",
        "mempool_cluster_12.7M_scale": "feasible WITH max_net_degree=100 cap: level-1 bisection "
                                        "59.0s (vs >24min stuck/unresolved without the cap, due "
                                        "to 2 pathological global nets with 986,903 combined "
                                        "pins); full recursive bisection completed",
        "cap_applied_both_sides": True, "max_net_degree": 100,
    },
    "known_simplifications": [
        "glue nets are degree-2-only (real cross-group nets: 96.2% degree-2, 3.8% touch 3-4 "
        "groups -- not reproduced)",
        "candidate interface-cell sampling is uniform random per tile, not boundary-biased or "
        "calibrated against a real iface_dist extraction from the cluster",
        "H4 is therefore a self-consistency check (does actual glue-endpoint sampling match "
        "the uniform-random construction it was built to be), not a validation against real "
        "physical interface-cell placement",
    ],
}
with open(os.path.join(REPO, "results/m4/bench/t6_summary.json"), "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True, default=_json_default)
print("t6_summary.json:", json.dumps({k: summary[k] for k in
      ("n1_verdict", "n2_verdict", "selected_normalization", "t7_gate")}, indent=1))
