"""M4 T6B verdict assembler (2026-08-15 T6 holdout adjudication doc
`docs/results/2026-08-15-m4-t6-holdout-adjudication.md` sec 4.1's B-0..B-6
table, sec 4.0's re-scoping: T6's already-evaluated H1/H2/H3/H4/H6 at
1x2/2x2 are frozen -- "already_evaluated_T6", never re-run -- and T6B only
evaluates (i) H5 (never run before) and (ii) metrics on the *new* 3x3/
zero-glue artifacts).

Assembles `results/m4/bench/verify_group3x3_t6b.json` from the 3x3 arrays
`build_t6_arrays.py` already produced (`results/m4/bench/arrays/3x3_n2`,
`3x3_noglue`) plus the source `mempool_group` Bookshelf export, following
the doc's sec 4.1 table row by row:

  B-0  V0' structural integrity (streaming `verify_bench.check_v0_
       structural`, "not worse than source" semantics) on both 3x3_n2 and
       3x3_noglue.
  B-1  Rent shape-invariance DiD -- needs `rent.measure_rent` runs on 4
       arrays (~45-60 min each, sec 4.3) this module does not run;
       recorded `pending_measurement`.
  B-2  H5 matched-resolution K-grid ratio -- needs `evaluator_gpu`/
       `evaluator_ref` runs (T8/T9 products) this module does not run;
       recorded `pending_measurement`.
  B-3  H3' full-net-degree KS vs source `mempool_group`, degree arrays
       streamed straight from each `.nets` file (no `rent.
       load_bookshelf_netlist`/mtkahypar needed -- just the `NetDegree`
       header counts).
  B-4  H4' interface-cell self-consistency -- same method as T6's 2x2 H4
       (`/tmp/compute_h4.py`, rescued to `results/m4/bench/rescue/
       h4_result.json`), applied fresh to 3x3's `t0_0` tile.
  B-5  H5' arithmetic self-check (`verify_bench.check_h5prime_self_
       consistency`, unmodified) on the built 3x3_n2 array plus three
       kernel/alpha variants that exist only as `n2_pair_counts_sinkhorn`
       recipes (sec 5.3's kernel-sensitivity table), not materialized
       arrays.
  B-6  Per-tile glue-endpoint budget identity (every tile's Sinkhorn row
       sum == B, to 1e-9 relative error) -- independently recomputed here
       (not just re-reading the manifest's own `n2_max_rel_dev`), for the
       four kernels sec 4.1's B-6 row and sec 5.2/5.3 name: K1 main
       (alpha=0.10230633302323285), K3 truncated, K1 alpha=0 (flat), K1
       alpha=1.915 (optimistic) -- the same four the summary table's row c
       and sec 5.3's variant table single out as either materialized or
       recipe-only (K2 exp is not among sec 4.1's "四個核", so it is not
       included here).

H1/H2 (`not_applicable`, no same-scale real reference), H1/H2/H3/H4/H6 at
1x2/2x2 (`already_evaluated_T6`, frozen), and V1 (`infrastructure_blocked`,
same DREAMPlace `PlaceDB.read_pl` regex incompatibility `rent.py`'s module
docstring already documents) are recorded as fixed-status entries, matching
sec 4.1's table exactly -- none of them re-run any measurement.

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.assemble_t6b
"""
import argparse
import json
import os
import subprocess
import sys
import time
from array import array

import numpy as np

from ioplace.bench import glue_gen
from ioplace.bench import verify_bench as vb

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DP = os.environ.get("DREAMPLACE_ROOT", os.path.join(os.path.dirname(REPO), "DREAMPlace"))
DEFAULT_ARRAYS_ROOT = os.path.join(REPO, "results/m4/bench/arrays")
DEFAULT_SOURCE_PREFIX = os.path.join(REPO, "results/m4/bench/mempool_group_export/mempool_group")
DEFAULT_OUT = os.path.join(REPO, "results/m4/bench/verify_group3x3_t6b.json")
DEFAULT_DEGREES_DIR = os.path.join(REPO, "results/m4/bench/degrees")

ADJUDICATION_DOC = "docs/results/2026-08-15-m4-t6-holdout-adjudication.md"
BUDGET_PAIRS = 3.0

# sec 4.1 B-6's "3x3 全部四個核" / sec 5.2/5.3's kernel-sensitivity table --
# K2 (exp kernel) is not one of the four this row names, so it is not
# included in either B-5's "三個變體" or B-6's "四個核" here (see module
# docstring).
_MAIN_ALPHA = 0.10230633302323285  # cluster_stats.json's mom.alpha, frozen sec 4.1
_KERNEL_VARIANTS_NON_MAIN = [
    {"name": "K3_truncated", "kernel": "truncated", "alpha": _MAIN_ALPHA},
    {"name": "K1_alpha0_flat", "kernel": "power_law", "alpha": 0.0},
    {"name": "K1_alpha1915_optimistic", "kernel": "power_law", "alpha": 1.915},
]


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


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


# ---------------------------------------------------------------------------
# B-0
# ---------------------------------------------------------------------------

def check_b0(arrays_root=DEFAULT_ARRAYS_ROOT, source_prefix=DEFAULT_SOURCE_PREFIX):
    """V0' on 3x3_n2 and 3x3_noglue, both against the source group (sec
    4.1 B-0 row: "3×3 + 3×3-noglue")."""
    out = {}
    for name in ("3x3_n2", "3x3_noglue"):
        prefix = os.path.join(arrays_root, name, name)
        with open(prefix + ".manifest.json") as f:
            manifest = json.load(f)
        t0 = time.time()
        res = vb.check_v0_structural(prefix, manifest=manifest, source_prefix=source_prefix)
        res["elapsed_s"] = time.time() - t0
        out[name] = res
    status = "ok" if all(v["status"] == "ok" for v in out.values()) else "fail"
    return {"status": status, "metric": "B0_v0_structural", "arrays": out}


# ---------------------------------------------------------------------------
# B-3 / B-4 shared streaming scan (one pass over 3x3_n2.nets for both)
# ---------------------------------------------------------------------------

def scan_degrees_and_tile_glue_endpoints(nets_path, tile_name=None):
    """Single streaming pass over `nets_path`'s net blocks. Returns
    `(degrees: np.ndarray[int32], glue_local_names: list[str] | None)`.

    Degrees come straight from each block's `NetDegree : <count> <name>`
    header -- no need to touch a single pin line to get B-3's full-net-
    degree array (`rent.load_bookshelf_netlist`'s full pin2node/pin2net
    parse, built for mtkahypar partitioning, is unneeded overhead here).

    If `tile_name` is given (e.g. "t0_0"), also collects the local node
    names used as glue-net endpoints within that tile (B-4's interface-
    cell sample) -- same method as `/tmp/compute_h4.py`'s
    `glue_endpoints_for_tile` (rescued to `results/m4/bench/rescue/
    h4_result.json`'s producer), generalized from the "t0_0"-only 1x2/2x2
    convention to (still) "t0_0" on 3x3. Folded into this same pass so a
    16GB `.nets` file is streamed once for both B-3 and B-4 instead of
    twice."""
    degs = array("i")
    glue_names = [] if tile_name is not None else None
    prefix = (tile_name + "/") if tile_name is not None else None
    with open(nets_path) as f:
        started = False
        in_glue_net = False
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumPins"):
                    started = True
                continue
            if not s:
                continue
            if s.startswith("NetDegree"):
                _, rest = s.split(":", 1)
                deg_str, name = rest.split()
                degs.append(int(deg_str))
                if tile_name is not None:
                    in_glue_net = "glue" in name
                continue
            if tile_name is not None and in_glue_net and s.startswith(prefix):
                glue_names.append(s.split()[0][len(prefix):])
    return np.frombuffer(degs, dtype=np.int32), glue_names


def check_b3(arrays_root=DEFAULT_ARRAYS_ROOT, source_prefix=DEFAULT_SOURCE_PREFIX,
             degrees_dir=DEFAULT_DEGREES_DIR, synth_degrees=None):
    """H3' full-net-degree KS, 3x3_n2 vs source `mempool_group` (sec 4.1
    B-3 row). `synth_degrees` lets the caller reuse a degree array already
    scanned by `check_b4` (both need one pass over 3x3_n2.nets; `main()`
    below scans it once and passes the result to both)."""
    os.makedirs(degrees_dir, exist_ok=True)
    if synth_degrees is None:
        synth_path = os.path.join(arrays_root, "3x3_n2", "3x3_n2")
        t0 = time.time()
        synth_degrees, _ = scan_degrees_and_tile_glue_endpoints(synth_path + ".nets")
        t_synth = time.time() - t0
    else:
        t_synth = None
    np.save(os.path.join(degrees_dir, "3x3_n2.net_degrees.npy"), synth_degrees)

    t0 = time.time()
    src_degrees, _ = scan_degrees_and_tile_glue_endpoints(source_prefix + ".nets")
    t_src = time.time() - t0
    np.save(os.path.join(degrees_dir, "mempool_group_source.net_degrees.npy"), src_degrees)

    result = vb.check_h3_degree_ks(synthetic_degrees=synth_degrees, real_degrees=src_degrees,
                                    ks_threshold=0.05, bucket_rel_tol=0.20)
    result["scan_elapsed_s"] = {"3x3_n2": t_synth, "mempool_group_source": t_src}
    result["n_nets"] = {"3x3_n2": int(len(synth_degrees)), "mempool_group_source": int(len(src_degrees))}
    return result


def _load_pl_positions(pl_path):
    pos = {}
    with open(pl_path) as f:
        next(f)
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            pos[parts[0]] = (float(parts[1]), float(parts[2]))
    return pos


def _dist_to_boundary(xs, ys, w, h):
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    d = np.minimum.reduce([xs, w - xs, ys, h - ys])
    return d / max(w, h)


def check_b4(arrays_root=DEFAULT_ARRAYS_ROOT, source_prefix=DEFAULT_SOURCE_PREFIX, seed=0,
             glue_local_names=None):
    """H4' interface-cell self-consistency on 3x3_n2's `t0_0` tile -- same
    method `/tmp/compute_h4.py` used for T6's 2x2 H4 (rescued result:
    `results/m4/bench/rescue/h4_result.json`), applied fresh to 3x3 (sec
    4.1 B-4 row: "照 T6 對 2×2 的 H4 同法搬到 3×3"). `glue_local_names`
    lets the caller pass in an already-scanned endpoint list (see
    `check_b3`'s docstring)."""
    prefix = os.path.join(arrays_root, "3x3_n2", "3x3_n2")
    with open(prefix + ".manifest.json") as f:
        manifest = json.load(f)
    w, h = manifest["tile_width"], manifest["tile_height"]
    tile_name = "t0_0"

    if glue_local_names is None:
        _, glue_local_names = scan_degrees_and_tile_glue_endpoints(prefix + ".nets", tile_name=tile_name)

    source_pos = _load_pl_positions(source_prefix + ".pl")
    xs, ys, missing = [], [], 0
    for n in glue_local_names:
        p = source_pos.get(n)
        if p is None:
            missing += 1
            continue
        xs.append(p[0])
        ys.append(p[1])
    d_glue = _dist_to_boundary(xs, ys, w, h) if xs else np.array([])

    rng = np.random.default_rng(seed)
    all_names = list(source_pos.keys())
    sample_size = min(len(all_names), max(len(d_glue) * 20, 5000))
    idx = rng.choice(len(all_names), size=sample_size, replace=False)
    sxs = [source_pos[all_names[i]][0] for i in idx]
    sys_ = [source_pos[all_names[i]][1] for i in idx]
    d_ref = _dist_to_boundary(sxs, sys_, w, h)

    result = vb.check_h4_interface_dist_ks(
        synthetic_dist=d_glue if len(d_glue) else None, real_dist=d_ref, ks_threshold=0.10)
    result.update({
        "tile": tile_name, "n_glue_endpoints": len(d_glue), "n_missing_lookup": missing,
        "n_reference_sample": len(d_ref),
        "glue_mean": float(np.mean(d_glue)) if len(d_glue) else None,
        "reference_mean": float(np.mean(d_ref)),
        "seed": seed,
        "method": "same as T6 2x2 H4 (/tmp/compute_h4.py, rescued to results/m4/bench/rescue/"
                  "h4_result.json), applied fresh to 3x3 tile t0_0 with a new rng(seed=0) "
                  "(sec 4.1 B-4 row)",
    })
    return result


# ---------------------------------------------------------------------------
# B-5 (2026-08-15 T6 holdout adjudication doc Appendix A.3: B-5a/B-5b/B-5c
# three-way split, replacing the single `check_h5prime_self_consistency`
# call this used to make -- see that function's and `verify_bench.
# check_b5a_recipe_arithmetic`'s docstrings for why)
# ---------------------------------------------------------------------------

def _b5_leg_status(b5a_status, b5b_status=None):
    """B-5a gates decisively; B-5b (only meaningful for a materialized
    array -- variants have no actual sampled net count) gates exactly;
    B-5c never gates (Appendix A.3: "只入輸出不入 status", not passed here
    at all). `normalization_mismatch` propagates as its own status rather
    than collapsing into "fail" -- it means the *inputs* were built with
    the wrong formula, not that this leg's own recipe arithmetic is
    broken (see `check_b5a_recipe_arithmetic`'s docstring)."""
    if b5a_status == "normalization_mismatch":
        return "normalization_mismatch"
    if b5b_status is None:
        return b5a_status
    return "ok" if (b5a_status == "ok" and b5b_status == "ok") else "fail"


def check_b5(arrays_root=DEFAULT_ARRAYS_ROOT, lambda_0_tile=None, alpha_main=_MAIN_ALPHA,
             r=3, c=3, budget_pairs=BUDGET_PAIRS, supersede_with=None):
    """B-5' arithmetic self-check (Appendix A.3's B-5a/B-5b/B-5c split) on
    the built 3x3_n2 array ("main" leg -- all three sub-checks apply) plus
    the three non-main kernel/alpha variants from sec 5.3's kernel-
    sensitivity table that exist only as `n2_pair_counts_sinkhorn` recipes,
    not materialized arrays ("三個變體" leg, sec 4.1 B-5 row) -- only B-5a
    applies to those (no actual sampled net count exists to run B-5b/B-5c
    against; each variant's `pair_counts` *is* B-5a's `expected_pair_
    counts`, recomputed fresh here, same as before this split).

    `supersede_with` (Appendix A.3: "重算不是第二次機會...原 fail 值逐字保留
    在 superseded_by_recompute 欄"): if given, the caller's *previous*
    `checks.B5_h5prime_arithmetic_self_check` block (the pre-recompute
    single-check result, e.g. `{"status": "fail", "rel_err": 0.6076, ...}`)
    is nested verbatim under the returned dict's `superseded_by_recompute`
    key -- this recompute does not erase the record of what the old
    (buggy-formula) check reported, it only stops gating on it."""
    prefix = os.path.join(arrays_root, "3x3_n2", "3x3_n2")
    with open(prefix + ".manifest.json") as f:
        manifest = json.load(f)
    if lambda_0_tile is None:
        lambda_0_tile = manifest["glue"]["lambda_0"]

    main_b5a = vb.check_b5a_recipe_arithmetic(
        expected_pair_counts=manifest["t6"]["expected_pair_counts"], normalization="n2",
        R=r, C=c, lambda_0=lambda_0_tile, alpha=alpha_main, budget_pairs=budget_pairs)
    main_b5b = vb.check_b5b_manifest_fidelity(manifest)
    main_b5c = vb.check_b5c_sampling_noise(manifest)
    main_leg = {
        "status": _b5_leg_status(main_b5a["status"], main_b5b["status"]),
        "variant": "3x3_n2 (K1 main, built array, alpha=%r)" % alpha_main,
        "B5a_recipe_arithmetic": main_b5a, "B5b_manifest_fidelity": main_b5b,
        "B5c_sampling_noise": main_b5c,
    }

    variant_results = {}
    for v in _KERNEL_VARIANTS_NON_MAIN:
        pair_counts, diag = glue_gen.n2_pair_counts_sinkhorn(
            lambda_0_tile, v["alpha"], r, c, budget_pairs=budget_pairs, kernel=v["kernel"])
        b5a = vb.check_b5a_recipe_arithmetic(
            expected_pair_counts=pair_counts, normalization="n2", R=r, C=c,
            lambda_0=lambda_0_tile, alpha=v["alpha"], budget_pairs=budget_pairs)
        variant_results[v["name"]] = {
            "status": _b5_leg_status(b5a["status"]), "variant": v["name"], "kernel": v["kernel"],
            "sinkhorn_diagnostics": diag, "B5a_recipe_arithmetic": b5a,
            "note": "recipe-only (Appendix A.3/5.3), not a materialized array -- only B-5a "
                    "applies (no actual sampled net count exists for B-5b/B-5c to compare "
                    "against)",
        }

    all_status = [main_leg["status"]] + [r["status"] for r in variant_results.values()]
    overall = "ok" if all(s == "ok" for s in all_status) else "fail"
    result = {"status": overall, "metric": "B5_h5prime_arithmetic_self_check",
              "main_3x3_n2": main_leg, "variants": variant_results}
    if supersede_with is not None:
        result["superseded_by_recompute"] = supersede_with
    return result


# ---------------------------------------------------------------------------
# B-6
# ---------------------------------------------------------------------------

def check_b6(lambda_0_tile, alpha_main=_MAIN_ALPHA, r=3, c=3, budget_pairs=BUDGET_PAIRS,
             rel_err_threshold=1e-9):
    """Per-tile glue-endpoint budget identity (sec 4.1 B-6 row): for each
    of the four kernels (K1 main, K3 truncated, K1 alpha=0, K1
    alpha=1.915), independently recomputes `n2_pair_counts_sinkhorn` and
    checks *every* tile's own endpoint total (not just the manifest's
    already-recorded `n2_max_rel_dev`, which is this same quantity's `max`
    over tiles for the K1-main array only -- this is a fresh, independent
    recompute per sec 4.1's "正式重算一次") against `B = budget_pairs *
    lambda_0_tile` to `rel_err_threshold` relative error."""
    b = budget_pairs * lambda_0_tile
    variants = [{"name": "K1_main", "kernel": "power_law", "alpha": alpha_main}] + _KERNEL_VARIANTS_NON_MAIN

    per_kernel = {}
    all_ok = True
    for v in variants:
        pair_counts, diag = glue_gen.n2_pair_counts_sinkhorn(
            lambda_0_tile, v["alpha"], r, c, budget_pairs=budget_pairs, kernel=v["kernel"])
        tiles = sorted({t for pair in pair_counts for t in pair})
        per_tile_sum = {t: 0.0 for t in tiles}
        for (ta, tb_), cnt in pair_counts.items():
            per_tile_sum[ta] += cnt
            per_tile_sum[tb_] += cnt
        max_rel_err = max(abs(s - b) / b for s in per_tile_sum.values())
        ok = max_rel_err <= rel_err_threshold
        all_ok = all_ok and ok
        per_kernel[v["name"]] = {
            "kernel": v["kernel"], "alpha": v["alpha"], "sinkhorn_diagnostics": diag,
            "max_rel_err_vs_B": max_rel_err, "status": "ok" if ok else "fail",
            "per_tile_sum": {f"{t[0]}_{t[1]}": s for t, s in per_tile_sum.items()},
        }
    return {"status": "ok" if all_ok else "fail", "metric": "B6_glue_budget_identity",
            "B": b, "budget_pairs": budget_pairs, "lambda_0_tile": lambda_0_tile,
            "threshold_rel_err": rel_err_threshold, "kernels": per_kernel}


# ---------------------------------------------------------------------------
# Fixed-status entries (sec 4.1 table, no measurement run)
# ---------------------------------------------------------------------------

def _pending_measurement(metric, reason):
    return {"status": "pending_measurement", "metric": metric, "reason": reason}


def _fixed_status_entries():
    return {
        "B1_rent_invariance_did": _pending_measurement(
            "B1_rent_invariance_did",
            "needs rent.measure_rent runs on 3x3_n2, 3x3_noglue, and 2x2_noglue (2x2_n2's p is "
            "already on file, 0.5262620) -- sec 4.3's pre-registered wall-time estimate is "
            "45-60 min/run, ~3h for the full set; not run in this task's scope (T6B execution "
            "was pre-registered as GPU-clear/CPU-only, Rent runs were left out of this task's "
            "instructed set: B-0/B-3/B-4/B-5/B-6 only)"),
        "B2_h5_matched_resolution": _pending_measurement(
            "B2_h5_matched_resolution",
            "needs evaluator_gpu/evaluator_ref hard_lambda_sum runs at matched per-tile "
            "resolution (K=16 on the 2x2 array vs K=4 on the source group, sec 4.4) -- a "
            "T8/T9 product, not produced in this task's scope"),
        "H1_rent_shape": {
            "status": "not_applicable", "metric": "H1_rent_shape",
            "reason": "27.7M has no same-scale real reference to band against (sec 4.1 table's "
                      "H1/H2 row)"},
        "H2_cut_histogram": {
            "status": "not_applicable", "metric": "H2_cut_histogram",
            "reason": "same as H1 -- no same-scale real reference (sec 4.1 table's H1/H2 row)"},
        "H1_H2_H3_H4_H6_at_1x2_2x2": {
            "status": "already_evaluated_T6",
            "reason": "T6 already evaluated H1/H2/H3/H4/H6 on 1x2/2x2 -- one pre-registered "
                      "metric per artifact, evaluated exactly once (sec 4.0): frozen, not "
                      "re-run here",
            "reference_files": ["results/m4/bench/verify_group2x2_n1.json",
                                 "results/m4/bench/verify_group2x2_n2.json",
                                 "results/m4/bench/t6_summary.json"]},
        "V1_dreamplace_readable": {
            "status": "infrastructure_blocked", "metric": "V1_dreamplace_readable",
            "reason": "DREAMPlace's own PlaceDB.read_pl uses a hardcoded \\w+ node-name regex "
                      "that cannot match this project's hierarchical (./[]/-containing) "
                      "instance names -- confirmed on mempool_group itself, independent of any "
                      "tiler/T6B-specific construction; see rent.py module docstring's M4 T6 "
                      "finding. Diagnostic-only per adjudication doc sec 4 (does not gate "
                      "T6/T6B/T7)."},
    }


def assemble(arrays_root=DEFAULT_ARRAYS_ROOT, source_prefix=DEFAULT_SOURCE_PREFIX,
             degrees_dir=DEFAULT_DEGREES_DIR, seed=0, supersede_b5_with=None):
    prefix_n2 = os.path.join(arrays_root, "3x3_n2", "3x3_n2")
    with open(prefix_n2 + ".manifest.json") as f:
        manifest_n2 = json.load(f)
    lambda_0_tile = manifest_n2["glue"]["lambda_0"]
    alpha_main = manifest_n2["glue"]["alpha"]

    b0 = check_b0(arrays_root=arrays_root, source_prefix=source_prefix)

    # B-3 and B-4 share one streaming pass over 3x3_n2.nets (see
    # scan_degrees_and_tile_glue_endpoints's docstring).
    t0 = time.time()
    synth_degrees, glue_local_names = scan_degrees_and_tile_glue_endpoints(
        prefix_n2 + ".nets", tile_name="t0_0")
    t_scan = time.time() - t0
    b3 = check_b3(arrays_root=arrays_root, source_prefix=source_prefix, degrees_dir=degrees_dir,
                   synth_degrees=synth_degrees)
    b3["scan_elapsed_s"]["3x3_n2"] = t_scan
    b4 = check_b4(arrays_root=arrays_root, source_prefix=source_prefix, seed=seed,
                   glue_local_names=glue_local_names)

    b5 = check_b5(arrays_root=arrays_root, lambda_0_tile=lambda_0_tile, alpha_main=alpha_main,
                   supersede_with=supersede_b5_with)
    b6 = check_b6(lambda_0_tile, alpha_main=alpha_main)

    checks = {"B0_v0_structural": b0, "B3_h3prime_degree_ks": b3,
              "B4_h4prime_interface_self_consistency": b4,
              "B5_h5prime_arithmetic_self_check": b5, "B6_glue_budget_identity": b6}
    checks.update(_fixed_status_entries())

    gating = ["B0_v0_structural", "B3_h3prime_degree_ks", "B4_h4prime_interface_self_consistency",
              "B5_h5prime_arithmetic_self_check", "B6_glue_budget_identity"]
    gate_statuses = {k: checks[k]["status"] for k in gating}
    scaling_usable = all(v == "ok" for v in gate_statuses.values())

    return {
        "shape": "3x3", "normalization": "n2",
        "provenance": {
            "repo_commit": _git_head(REPO), "date": "2026-08-15",
            "basis_doc": f"{ADJUDICATION_DOC} sec 4.1 (B-0..B-6 table), sec 6.1 (count freeze)",
            "n2_rule": "sinkhorn", "n2_budget_pairs": BUDGET_PAIRS,
            "lambda_0_tile": lambda_0_tile, "alpha_main": alpha_main,
            "source_prefix": source_prefix, "arrays_root": arrays_root,
        },
        "checks": checks,
        "gating_check_status": gate_statuses,
        "scaling_usable": scaling_usable,
        "scaling_usable_note": "sec 4.1: \"B-0 與 B-3…B-6 全綠 ⇒ 3×3 bench 可用於 scaling"
                                "(且永久不得用於品質宣稱)。B-1、B-2 的結果一律發表,其判定寫入報告"
                                "但不阻擋 scaling 用途\"",
        "quality_claims_prohibited": True,
    }


def patch_b5(existing, arrays_root=DEFAULT_ARRAYS_ROOT):
    """Recomputes *only* B-5 (Appendix A.3's B-5a/B-5b/B-5c split) against
    an already-assembled `existing` result dict (e.g. a prior `assemble()`
    output loaded from `verify_group3x3_t6b.json`) and splices it back in,
    recomputing `gating_check_status`/`scaling_usable` accordingly --
    B-0/B-3/B-4/B-6 are copied through unchanged.

    This is the "重算" entry point the 2026-08-15 adjudication Appendix
    A.3's B-5 recompute actually needs to run through: B-0/B-3/B-4 each
    stream the 3x3 array's multi-GB `.nets` file (`assemble()`'s ~5 min
    combined B-0+B-3/B-4 scan cost), which this task's scope explicitly
    excludes ("不掃大檔") -- B-5's own inputs are all manifest-JSON-level
    (`t6.expected_pair_counts`/`t6.sampled_pair_counts`, already on disk;
    the three kernel variants are pure in-memory Sinkhorn recomputes), so
    re-running the whole `assemble()` pipeline to get a new B-5 is
    unnecessary I/O this function avoids."""
    prefix_n2 = os.path.join(arrays_root, "3x3_n2", "3x3_n2")
    with open(prefix_n2 + ".manifest.json") as f:
        manifest_n2 = json.load(f)
    lambda_0_tile = manifest_n2["glue"]["lambda_0"]
    alpha_main = manifest_n2["glue"]["alpha"]

    old_b5 = existing["checks"]["B5_h5prime_arithmetic_self_check"]
    new_b5 = check_b5(arrays_root=arrays_root, lambda_0_tile=lambda_0_tile, alpha_main=alpha_main,
                       supersede_with=old_b5)

    out = dict(existing)
    out["checks"] = dict(existing["checks"])
    out["checks"]["B5_h5prime_arithmetic_self_check"] = new_b5
    gate_statuses = dict(existing["gating_check_status"])
    gate_statuses["B5_h5prime_arithmetic_self_check"] = new_b5["status"]
    out["gating_check_status"] = gate_statuses
    out["scaling_usable"] = all(v == "ok" for v in gate_statuses.values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", default=DEFAULT_ARRAYS_ROOT)
    ap.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    ap.add_argument("--degrees-dir", default=DEFAULT_DEGREES_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--supersede-b5-from", default=None,
                     help="path to a previous verify_group3x3_t6b.json (2026-08-15 adjudication "
                          "Appendix A.3: '重算不是第二次機會...原 fail 值逐字保留在 "
                          "superseded_by_recompute 欄') -- its checks.B5_h5prime_arithmetic_self_"
                          "check block is nested verbatim into the new B-5 result's "
                          "superseded_by_recompute key, not overwritten")
    ap.add_argument("--patch-b5-only", default=None, metavar="EXISTING_JSON",
                     help="skip the full B-0/B-3/B-4/B-6 rebuild (which streams the 3x3 array's "
                          "multi-GB .nets file) and only recompute B-5 against EXISTING_JSON's "
                          "already-assembled checks (see patch_b5's docstring); EXISTING_JSON's "
                          "own current B5_h5prime_arithmetic_self_check block becomes the new "
                          "result's superseded_by_recompute")
    args = ap.parse_args(argv)

    if args.patch_b5_only:
        with open(args.patch_b5_only) as f:
            existing = json.load(f)
        out = patch_b5(existing, arrays_root=args.arrays_root)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True, default=_json_default)
        print(f"[assemble_t6b] patched B-5 only, wrote {args.out}")
        print(f"[assemble_t6b] gating_check_status={out['gating_check_status']} "
              f"scaling_usable={out['scaling_usable']}")
        return out

    supersede_b5_with = None
    if args.supersede_b5_from:
        with open(args.supersede_b5_from) as f:
            supersede_b5_with = json.load(f)["checks"]["B5_h5prime_arithmetic_self_check"]

    out = assemble(arrays_root=args.arrays_root, source_prefix=args.source_prefix,
                    degrees_dir=args.degrees_dir, seed=args.seed,
                    supersede_b5_with=supersede_b5_with)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=_json_default)
    print(f"[assemble_t6b] wrote {args.out}")
    print(f"[assemble_t6b] gating_check_status={out['gating_check_status']} "
          f"scaling_usable={out['scaling_usable']}")
    return out


if __name__ == "__main__":
    main()
