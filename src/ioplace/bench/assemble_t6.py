"""M4 T6 verdict assembler (2026-08-15 T6 holdout adjudication doc
`docs/results/2026-08-15-m4-t6-holdout-adjudication.md` sec 8 item 1):
assembles `verify_group2x2_{n1,n2}.json` + `t6_summary.json` from the
already-computed pieces -- `cluster_stats.json`, the Rent measurements
(`ioplace.diagnostics.probes_m4.probe_rent_arrays` output, or the rescued
combined `rent_{cluster,synthetic}_result.json`), `h4_result.json`,
`kernel_sensitivity.json`, `build_summary.json` -- by calling
`ioplace.bench.verify_bench`'s H1/H2/H3/H4/H6 checks on them.

Migrated from `/tmp/assemble_t6.py` (rescued to `results/m4/bench/rescue/`
2026-08-15, after nearly being lost -- it was never checked into the
repo): the only change from the rescued version's *logic* is (a) no more
`sys.path.insert` before `from ioplace.bench import verify_bench` --
this module now lives inside the `ioplace.bench` package itself, so the
normal `PYTHONPATH=src` convention every other module in this repo uses is
enough; (b) every input defaults to a repo-relative path instead of
`/tmp/...`; (c) the per-norm gate-computation logic (`h1`/`h2`/`h3`/
`h4_gate`/`h6`/`diagnostics`) is now free functions taking explicit
arguments instead of closures over module-level globals, so it can be
unit-tested without needing every input file to exist at once.

**Known gap (not fixed by this migration, out of its scope per the
2026-08-15 adjudication doc sec 8 item 1's task list):** `h4_result.json`,
`kernel_sensitivity.json`, and the per-array `*_net_degrees.npy` files
(needed by `h3`/`h4_gate`) were *not* among the 5 files rescued to
`results/m4/bench/rescue/` -- only `rent_{cluster,synthetic}_result.json`
and the three `.py` scripts were. As of this migration those files still
exist at their original `/tmp/` paths (`/tmp/h4_result.json`,
`/tmp/kernel_sensitivity.json`, `/tmp/rent_cluster_net_degrees.npy`,
`/tmp/rent_synth_<case>_net_degrees.npy`) but are at the same data-loss
risk the other 5 files were before their rescue. This module's CLI
defaults point at repo-relative `results/m4/bench/rescue/` paths for them
(not yet present) rather than reintroducing a `/tmp` dependency into
checked-in code; running `main()` end to end therefore currently requires
passing `--h4`/`--kernel-sensitivity`/`--cluster-degrees-npy`/
`--synth-degrees-npy-template` explicitly until those files are rescued
too.

Usage:
    PYTHONPATH=src $PY -m ioplace.bench.assemble_t6 \\
        --h4 /tmp/h4_result.json --kernel-sensitivity /tmp/kernel_sensitivity.json \\
        --cluster-degrees-npy /tmp/rent_cluster_net_degrees.npy \\
        --synth-degrees-npy-template "/tmp/rent_synth_{name}_net_degrees.npy"
"""
import argparse
import json
import os
import sys

import numpy as np

from ioplace.bench import verify_bench as vb

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"

DEFAULT_RENT_CLUSTER = os.path.join(REPO, "results/m4/bench/rescue/rent_cluster_result.json")
DEFAULT_RENT_SYNTHETIC = os.path.join(REPO, "results/m4/bench/rescue/rent_synthetic_result.json")
DEFAULT_H4 = os.path.join(REPO, "results/m4/bench/rescue/h4_result.json")
DEFAULT_KERNEL_SENSITIVITY = os.path.join(REPO, "results/m4/bench/rescue/kernel_sensitivity.json")
DEFAULT_CLUSTER_STATS = os.path.join(REPO, "results/m4/bench/cluster_stats.json")
DEFAULT_BUILD_SUMMARY = os.path.join(REPO, "results/m4/bench/arrays/build_summary.json")
DEFAULT_CLUSTER_DEGREES_NPY = os.path.join(REPO, "results/m4/bench/rescue/rent_cluster_net_degrees.npy")
DEFAULT_SYNTH_DEGREES_NPY_TEMPLATE = os.path.join(REPO, "results/m4/bench/rescue/rent_synth_{name}_net_degrees.npy")
DEFAULT_OUT_DIR = os.path.join(REPO, "results/m4/bench")
DEFAULT_SOURCE_PREFIX = os.path.join(REPO, "results/m4/bench/mempool_group_export/mempool_group")


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


def h1(norm, synth, cluster):
    p_syn = synth[f"2x2_{norm}"]["p"]
    p_real = cluster["p"]
    return vb.check_h1_rent_shape(p_synthetic=p_syn, p_real=p_real)


def h2(norm, synth, cluster):
    syn_levels = {l["level"]: l["avg_terminals"] for l in synth[f"2x2_{norm}"]["levels"]
                  if 1e3 <= l["avg_block_size"] <= 1e6}
    real_levels = {l["level"]: l["avg_terminals"] for l in cluster["levels"]
                   if 1e3 <= l["avg_block_size"] <= 1e6}
    return vb.check_h2_cut_histogram(synthetic_levels=syn_levels, real_levels=real_levels)


def h3(norm, synth_degrees, cluster_degrees):
    return vb.check_h3_degree_ks(synthetic_degrees=synth_degrees, real_degrees=cluster_degrees,
                                  ks_threshold=0.05, bucket_rel_tol=0.20)


def h4_gate(shape, norm, h4):
    d = h4[f"{shape}_{norm}"]
    ks_threshold = 0.10
    ok = d["ks"] is not None and d["ks"] <= ks_threshold
    return {"status": "ok" if ok else "fail", "metric": "H4_self_consistency",
            "ks": d["ks"], "threshold": ks_threshold, "detail": d}


def h6(norm, synth):
    p1 = synth["1x1_source"]["p"]
    p2 = synth[f"1x2_{norm}"]["p"]
    p3 = synth[f"2x2_{norm}"]["p"]
    threshold = 0.03
    pairs = {"1x1_vs_1x2": abs(p1 - p2), "1x1_vs_2x2": abs(p1 - p3), "1x2_vs_2x2": abs(p2 - p3)}
    ok = all(v <= threshold for v in pairs.values())
    return {"status": "ok" if ok else "fail", "metric": "H6_rent_invariance",
            "p_1x1_source": p1, f"p_1x2_{norm}": p2, f"p_2x2_{norm}": p3,
            "pairwise_abs_diff": pairs, "threshold": threshold}


def diagnostics(shape, norm, arrays_root, source_prefix=DEFAULT_SOURCE_PREFIX):
    """V0/H5' -- diagnostic only, never gate (adjudication doc: '診斷不判定:
    H5、H5′、V0/V1'). V1 is not run here at all (DREAMPlace's own PlaceDB
    reader cannot load this project's hierarchical-name Bookshelf exports
    at all -- see rent.py module docstring's M4 T6 finding -- so V1 is
    recorded as infrastructure_blocked rather than executed)."""
    prefix = os.path.join(arrays_root, f"{shape}_{norm}", f"{shape}_{norm}")
    with open(prefix + ".manifest.json") as f:
        manifest = json.load(f)
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


def assemble_norm(norm, synth, cluster, h4, kernel_sens, cluster_stats, build_summary,
                   synth_degrees, cluster_degrees, arrays_root, source_prefix=DEFAULT_SOURCE_PREFIX):
    """Builds one norm's (`n1`/`n2`) full `verify_group2x2_<norm>.json`
    payload -- same shape the 2026-08-14/15 adjudication docs' cited
    `verify_group2x2_*.json` already has."""
    gates = {
        "H1_rent_shape": h1(norm, synth, cluster),
        "H2_cut_histogram": h2(norm, synth, cluster),
        "H3_degree_ks": h3(norm, synth_degrees[f"2x2_{norm}"], cluster_degrees),
        "H4_self_consistency": h4_gate("2x2", norm, h4),
        "H6_rent_invariance": h6(norm, synth),
    }
    all_ok = all(g["status"] == "ok" for g in gates.values())
    return {
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
        "diagnostics_not_gating": diagnostics("2x2", norm, arrays_root, source_prefix=source_prefix),
        "build_manifest": build_summary.get("2x2", {}).get(norm),
    }


def assemble_t6_summary(verdicts, synth, cluster, kernel_sens):
    selected = None
    if verdicts["n2"]["verdict"] == "PASS":
        selected = "n2"
    elif verdicts["n1"]["verdict"] == "PASS":
        selected = "n1"
    return {
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rent-cluster", default=DEFAULT_RENT_CLUSTER)
    ap.add_argument("--rent-synthetic", default=DEFAULT_RENT_SYNTHETIC)
    ap.add_argument("--h4", default=DEFAULT_H4)
    ap.add_argument("--kernel-sensitivity", default=DEFAULT_KERNEL_SENSITIVITY)
    ap.add_argument("--cluster-stats", default=DEFAULT_CLUSTER_STATS)
    ap.add_argument("--build-summary", default=DEFAULT_BUILD_SUMMARY)
    ap.add_argument("--cluster-degrees-npy", default=DEFAULT_CLUSTER_DEGREES_NPY)
    ap.add_argument("--synth-degrees-npy-template", default=DEFAULT_SYNTH_DEGREES_NPY_TEMPLATE,
                     help="format string with a {name} placeholder, e.g. .../rent_synth_{name}_net_degrees.npy")
    ap.add_argument("--arrays-root", default=os.path.join(REPO, "results/m4/bench/arrays"))
    ap.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    with open(args.rent_cluster) as f:
        cluster = json.load(f)
    with open(args.rent_synthetic) as f:
        synth = json.load(f)
    with open(args.h4) as f:
        h4 = json.load(f)
    with open(args.kernel_sensitivity) as f:
        kernel_sens = json.load(f)
    with open(args.cluster_stats) as f:
        cluster_stats = json.load(f)
    with open(args.build_summary) as f:
        build_summary = json.load(f)

    cluster_degrees = np.load(args.cluster_degrees_npy)
    synth_degrees = {name: np.load(args.synth_degrees_npy_template.format(name=name))
                      for name in ("2x2_n1", "2x2_n2")}

    verdicts = {}
    for norm in ("n1", "n2"):
        out = assemble_norm(norm, synth, cluster, h4, kernel_sens, cluster_stats, build_summary,
                             synth_degrees, cluster_degrees, args.arrays_root,
                             source_prefix=args.source_prefix)
        verdicts[norm] = out
        out_path = os.path.join(args.out_dir, f"verify_group2x2_{norm}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True, default=_json_default)
        print(f"[{norm}] verdict={out['verdict']} "
              f"gates={[k + ':' + v['status'] for k, v in out['holdout_gates'].items()]}")

    summary = assemble_t6_summary(verdicts, synth, cluster, kernel_sens)
    summary_path = os.path.join(args.out_dir, "t6_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=_json_default)
    print("t6_summary.json:", json.dumps({k: summary[k] for k in
          ("n1_verdict", "n2_verdict", "selected_normalization", "t7_gate")}, indent=1))
    return verdicts, summary


if __name__ == "__main__":
    main()
