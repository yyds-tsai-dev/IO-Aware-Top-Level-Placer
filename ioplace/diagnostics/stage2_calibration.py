"""Stage 2 S9 -- calibration analysis (`docs/superpowers/specs/2026-08-13-
stage2-innovus-calibration-plan.md` sec 8 (8.2's seven tables, 8.3's
regression, 8.4's C1-C5 rules) and sec 10's S9 row).

Pure-CPU aggregation of S8's already-routed samples
(`results/stage2/s8/<design>__<arm>/`) -- **no GPU evaluator recompute, no
re-routing**. Every number in this module's output is either read verbatim
from an on-disk JSON/log or derived from those numbers with plain
numpy/scipy. Two facts discovered while wiring this up bound how far the
seven tables can go, and are surfaced as `not_evaluable` entries rather than
worked around:

1. **No per-net evaluator array is persisted anywhere in S8's output.**
   `metrics.json` only has scalar totals (`io_count`, `ft_count`,
   `hard_lambda_sum`, `io_rg`, `ft_rg`, `tree_wl`, `hpwl`, ...) and
   `metrics.json.npz` only has `node_x`/`node_y` (`ioplace/drivers/
   run_placement.py:408`) -- the driver never calls `evaluator_ref.evaluate`/
   `evaluator_gpu` and saves the resulting `EvalResult.per_net_*` arrays.
   Recomputing them now would mean calling `evaluate_gpu` on the post-route
   `routed.def` COMPONENTS, which needs a GPU and is explicitly out of scope
   for S9. Consequence: every *per-net* evaluator-vs-route comparison (table
   2's Spearman/Pearson, table 3's per-degree bucket, table 7's boundary-pair
   correlation, and C1's rho comparison) is `not_evaluable`; only *total*-
   level ratios are computable (table 1, table 5, table 6).
2. **`per_net_route_ft` is the `-1` "not computed" sentinel for every net in
   every S8 sample** (`ioplace/route_eval/route_crossings.py:373`: a net
   missing from the `pin_regions` argument gets `route_ft=-1`; S8's
   extraction call never supplied it). So table 6's `route_ft` column is
   `not_evaluable` too, even though `ft_rg`/`ft_mst` (evaluator totals) are
   fine.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.stage2_calibration \\
        --s8-dir results/stage2/s8 \\
        --out-dir results/stage2/calibration
"""
import argparse
import json
import os
import re

import numpy as np
from scipy.optimize import nnls

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
S8_DIR_DEFAULT = os.path.join(REPO, "results", "stage2", "s8")
OUT_DIR_DEFAULT = os.path.join(REPO, "results", "stage2", "calibration")

# The 6 valid (design, arm) samples S8 actually produced a routed.def +
# crossings_k*.json for (spec sec 9.1 E1's "instance/net set unchanged"
# V1-V5 checks passed -- see each sample's or_run/verify_s2.json).
VALID_SAMPLES = [
    ("des_perf_1", "flat"),
    ("des_perf_1", "ours_k16"),
    ("des_perf_1", "ours_k32"),
    ("mempool_tile_wrap", "flat"),
    ("mempool_tile_wrap", "ours_k16"),
    ("mempool_tile_wrap", "ours_k32"),
]

# The 9 (design, arm) combinations S8 attempted but could not route to
# completion within the batch's wall-clock budget -- spec sec 8.2 table 5's
# "不可評樣本表". failure_mode / evidence are hand-classified from each
# sample's or_run/openroad_route.log + results/stage2/s8/route_chain*.log +
# results/stage2/s8/extract_chain.log (see this module's docstring for the
# classification method; re-verified against those logs, not guessed).
UNEVALUABLE_SAMPLES = [
    ("matrix_mult_1", "flat"), ("matrix_mult_1", "ours_k16"), ("matrix_mult_1", "ours_k32"),
    ("superblue19", "flat"), ("superblue19", "ours_k16"), ("superblue19", "ours_k32"),
    ("superblue12", "flat"), ("superblue12", "ours_k16"), ("superblue12", "ours_k32"),
]

# Lambda-route bucket edges mirroring M3 draft sec 2.2's Lambda bucket
# convention (spec sec 8.2 item 4): 1 / 2 / 3 / 4-8 / >8.
LAMBDA_BUCKET_EDGES = [(1, 1), (2, 2), (3, 3), (4, 8), (9, None)]
LAMBDA_BUCKET_LABELS = ["1", "2", "3", "4-8", ">8"]

_ELAPSED_RE = re.compile(r"cpu time = (\S+), elapsed time = (\S+),")
_VIOL_RE = re.compile(r"Number of violations = (\d+)")


def _case_dir(s8_dir, design, arm):
    return os.path.join(s8_dir, f"{design}__{arm}")


def _hhmmss_to_s(s):
    parts = [int(p) for p in s.split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-sample loading
# ---------------------------------------------------------------------------

def load_metrics(s8_dir, design, arm):
    return _read_json(os.path.join(_case_dir(s8_dir, design, arm), "metrics.json"))


def load_crossings(s8_dir, design, arm, k):
    path = os.path.join(_case_dir(s8_dir, design, arm), f"crossings_k{k}.json")
    if not os.path.exists(path):
        return None
    return _read_json(path)


def parse_route_log(log_path):
    """Best-effort parse of `or_run/openroad_route.log`: the last `cpu
    time=.../elapsed time=...` line (the grand total the whole
    global_route+detailed_route call reports right before `DONE_ROUTE`) and
    the last `Number of violations = N` line (DRT-0199, the final DR
    optimization-iteration violation count). Returns
    `{elapsed_s, cpu_s, final_dr_violations, done_route}`, with `None` for
    anything not found (e.g. a killed run with no DRT-0267 line yet)."""
    if not os.path.exists(log_path):
        return dict(elapsed_s=None, cpu_s=None, final_dr_violations=None, done_route=False)
    with open(log_path, errors="replace") as f:
        text = f.read()
    elapsed_matches = _ELAPSED_RE.findall(text)
    viol_matches = _VIOL_RE.findall(text)
    cpu_s = elapsed_s = None
    if elapsed_matches:
        cpu_s = _hhmmss_to_s(elapsed_matches[-1][0])
        elapsed_s = _hhmmss_to_s(elapsed_matches[-1][1])
    final_dr_violations = int(viol_matches[-1]) if viol_matches else None
    return dict(elapsed_s=elapsed_s, cpu_s=cpu_s, final_dr_violations=final_dr_violations,
                done_route="DONE_ROUTE" in text)


def parse_drc_rpt(rpt_path):
    if not os.path.exists(rpt_path):
        return None
    n = 0
    with open(rpt_path, errors="replace") as f:
        for line in f:
            if line.startswith("violation type:"):
                n += 1
    return n


def build_sample(s8_dir, design, arm):
    """Assembles one VALID_SAMPLES row: evaluator totals (metrics.json) +
    route totals at the *matched* K (the K the placement/evaluator actually
    used, `metrics.json['k']`) + the *other* K's crossing totals as a
    supplementary cross-K sensitivity point (sec 7.4/L-Q4-a) -- the other-K
    file uses a region grid the evaluator totals were never computed on, so
    it cannot be paired with io_mst/io_rg/lam_minus_1 for a ratio, only
    reported standalone in the sample-list table."""
    case = _case_dir(s8_dir, design, arm)
    metrics = load_metrics(s8_dir, design, arm)
    k = metrics["k"]
    other_k = 32 if k == 16 else 16

    matched = load_crossings(s8_dir, design, arm, k)
    other = load_crossings(s8_dir, design, arm, other_k)
    if matched is None:
        raise FileNotFoundError(f"{design}__{arm}: crossings_k{k}.json (matched K) missing")

    route_ft_arr = np.asarray(matched["per_net_route_ft"], dtype=np.int64)
    route_ft_all_sentinel = bool(np.all(route_ft_arr == -1))
    lam_route_arr = np.asarray(matched["per_net_lambda_route"], dtype=np.int64)

    verify_s2_path = os.path.join(case, "or_run", "verify_s2.json")
    verify_s2 = _read_json(verify_s2_path) if os.path.exists(verify_s2_path) else None

    log_path = os.path.join(case, "or_run", "openroad_route.log")
    log_info = parse_route_log(log_path)
    drc_rpt_path = os.path.join(case, "or_run", "drc.rpt")
    drc_rpt_count = parse_drc_rpt(drc_rpt_path)

    lam_minus_1 = int(metrics["hard_lambda_sum"])
    io_rg = int(metrics["io_rg"])
    ft_rg = int(metrics["ft_rg"])
    io_mst = int(metrics["io_count"])
    ft_mst = int(metrics["ft_count"])
    route_cross_raw = int(matched["total_route_cross_raw"])
    route_cross_dw = int(matched["total_route_cross_dw"])
    route_wl = int(matched["total_route_wl"])
    delta = matched["delta"]

    return dict(
        sample_id=f"{design}__{arm}",
        design=design, arm=arm, k_placement=k, router="OR",
        source_paths=dict(
            metrics=os.path.join(case, "metrics.json"),
            crossings_matched=os.path.join(case, f"crossings_k{k}.json"),
            crossings_other=(os.path.join(case, f"crossings_k{other_k}.json")
                              if other is not None else None),
            verify_s2=verify_s2_path if verify_s2 is not None else None,
            openroad_route_log=log_path if os.path.exists(log_path) else None,
            drc_rpt=drc_rpt_path if os.path.exists(drc_rpt_path) else None,
        ),
        num_nets=int(matched["num_nets"]),
        num_unmatched_nets=int(matched["num_unmatched_nets"]),
        delta=delta,
        # evaluator totals (matched K)
        lam_minus_1=lam_minus_1, io_rg=io_rg, ft_rg=ft_rg, io_mst=io_mst, ft_mst=ft_mst,
        tree_wl=float(metrics["tree_wl"]), hpwl=float(metrics["hpwl"]),
        # route totals (matched K, delta=2 primary report value per sec 7.4)
        route_cross_raw=route_cross_raw, route_cross_dw=route_cross_dw, route_wl=route_wl,
        route_ft_evaluable=not route_ft_all_sentinel,
        route_ft_total=(int(np.sum(route_ft_arr[route_ft_arr >= 0]))
                         if not route_ft_all_sentinel else None),
        route_pair_demand=matched["route_pair_demand"],
        # route-side-only Lambda_route distribution (table 4)
        lambda_route_stats=dict(
            min=int(lam_route_arr.min()), max=int(lam_route_arr.max()),
            mean=float(lam_route_arr.mean()), median=float(np.median(lam_route_arr)),
        ),
        per_net_lambda_route=lam_route_arr,
        per_net_route_cross_dw=np.asarray(matched["per_net_route_cross_dw"], dtype=np.int64),
        per_net_route_wl=np.asarray(matched["per_net_route_wl"], dtype=np.int64),
        # derived three-value-decomposition quantities (sec 1.2 / 8.2-5)
        mst_excess=io_mst - io_rg,
        route_minus_io_rg=route_cross_dw - io_rg,
        io_mst_minus_route=io_mst - route_cross_dw,
        # other-K supplementary point (sample-list table only)
        other_k=other_k,
        other_k_route_cross_dw=(int(other["total_route_cross_dw"]) if other is not None else None),
        other_k_route_cross_raw=(int(other["total_route_cross_raw"]) if other is not None else None),
        # route-run bookkeeping
        drc_final_violations_log=log_info["final_dr_violations"],
        drc_rpt_violation_lines=drc_rpt_count,
        route_elapsed_s=log_info["elapsed_s"],
        route_done_route=log_info["done_route"],
        verify_s2_overall_pass=(verify_s2["overall_pass"] if verify_s2 else None),
    )


# Batch driver logs under results/stage2/s8/ that record route-attempt
# outcomes across all cases (fixed filenames, not derived from design name).
_BATCH_LOG_NAMES = ["route_chain.log", "route_chain_sb12.log", "route_chain_sb19.log",
                     "route_chain_tilewrap.log", "extract_chain.log"]


def build_unevaluable_sample(s8_dir, design, arm):
    case = _case_dir(s8_dir, design, arm)
    log_path = os.path.join(case, "or_run", "openroad_route.log")
    info = parse_route_log(log_path) if os.path.exists(log_path) else None
    sample_id = f"{design}__{arm}"
    route_chain_logs = []
    for name in _BATCH_LOG_NAMES:
        p = os.path.join(s8_dir, name)
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                if sample_id in f.read():
                    route_chain_logs.append(p)
    return dict(
        sample_id=sample_id, design=design, arm=arm,
        out_def_path=os.path.join(case, "out.def"),
        openroad_route_log=log_path if os.path.exists(log_path) else None,
        route_chain_logs=route_chain_logs,
        parsed_log=info,
    )


# ---------------------------------------------------------------------------
# Table 1: total-level ratio (evaluator vs route), per-net marked not_evaluable
# ---------------------------------------------------------------------------

def table1_total_ratio(samples):
    rows = []
    for s in samples:
        def ratio(num, den):
            return (s["route_cross_dw"] / den) if den else None
        rows.append(dict(
            sample_id=s["sample_id"],
            R_lam_minus_1=ratio(None, s["lam_minus_1"]),
            R_io_rg=ratio(None, s["io_rg"]),
            R_io_mst=ratio(None, s["io_mst"]),
            route_cross_dw=s["route_cross_dw"], route_cross_raw=s["route_cross_raw"],
            lam_minus_1=s["lam_minus_1"], io_rg=s["io_rg"], io_mst=s["io_mst"],
        ))
    pooled_route = sum(s["route_cross_dw"] for s in samples)
    pooled = dict(
        R_lam_minus_1=pooled_route / sum(s["lam_minus_1"] for s in samples),
        R_io_rg=pooled_route / sum(s["io_rg"] for s in samples),
        R_io_mst=pooled_route / sum(s["io_mst"] for s in samples),
    )
    return dict(
        rows=rows, pooled=pooled,
        per_net=dict(
            not_evaluable=True,
            reason=("no per-net evaluator array (per_net_crossings/per_net_steiner) is "
                    "persisted in metrics.json or metrics.json.npz for any S8 sample; "
                    "recomputing it needs evaluate_gpu on the post-route placement, which "
                    "needs a GPU (out of scope for S9's CPU-only analysis)"),
        ),
    )


# ---------------------------------------------------------------------------
# Table 2: per-net correlation -- not evaluable, same root cause as table 1
# ---------------------------------------------------------------------------

def table2_per_net_correlation():
    return dict(
        not_evaluable=True,
        reason=("per-net evaluator arrays (per_net_crossings, per_net_steiner) are not "
                "persisted anywhere in S8's output -- see module docstring point 1. Pearson/"
                "Spearman between per_net_crossings and per_net_route_cross_dw cannot be "
                "computed without a GPU evaluator rerun on each routed.def, which is out of "
                "scope for S9."),
    )


# ---------------------------------------------------------------------------
# Table 3: per-degree bucket -- not evaluable, degree not persisted
# ---------------------------------------------------------------------------

def table3_degree_bucket():
    return dict(
        not_evaluable=True,
        reason=("net degree (pin count) is not persisted in any S8 artifact -- netmap.json "
                "only has {net_index: net_name}, crossings_k*.json's per-net arrays carry no "
                "degree field. Recovering it needs reloading placedb via DREAMPlace's C++ "
                "place_io reader, which this module does not do (S9 is read-existing-"
                "artifacts-only, no re-invocation of the placement/DP pipeline)."),
    )


# ---------------------------------------------------------------------------
# Table 4: per-Lambda_route bucket (route-side only self-bucket)
# ---------------------------------------------------------------------------

def _bucket_index(lam):
    for i, (lo, hi) in enumerate(LAMBDA_BUCKET_EDGES):
        if hi is None:
            if lam >= lo:
                return i
        elif lo <= lam <= hi:
            return i
    return None


def table4_lambda_route_bucket(samples):
    rows = []
    for s in samples:
        lam = s["per_net_lambda_route"]
        dw = s["per_net_route_cross_dw"]
        wl = s["per_net_route_wl"]
        idx = np.array([_bucket_index(int(x)) for x in lam])
        for b, label in enumerate(LAMBDA_BUCKET_LABELS):
            mask = idx == b
            n = int(mask.sum())
            rows.append(dict(
                sample_id=s["sample_id"], lambda_route_bucket=label,
                n_nets=n,
                sum_route_cross_dw=int(dw[mask].sum()) if n else 0,
                mean_route_cross_dw=float(dw[mask].mean()) if n else None,
                sum_route_wl=int(wl[mask].sum()) if n else 0,
            ))
    return dict(
        rows=rows,
        degraded=True,
        degraded_reason=("bucketed on route-side Lambda_route only (per_net_lambda_route in "
                          "crossings_k*.json); no evaluator-side per-net Lambda is persisted "
                          "(same cause as table 2), so this is NOT a route-vs-evaluator "
                          "matched bucket table -- it only shows how route-side crossing "
                          "counts distribute across route-observed Lambda_route."),
    )


# ---------------------------------------------------------------------------
# Table 5: three-value decomposition (per spec sec 8.2 item 5)
# ---------------------------------------------------------------------------

def table5_three_value_decomposition(samples):
    rows = []
    for s in samples:
        rows.append(dict(
            design=s["design"], K=s["k_placement"], arm=s["arm"], router=s["router"],
            sum_lam_minus_1=s["lam_minus_1"], sum_io_rg=s["io_rg"],
            sum_route=s["route_cross_dw"], sum_io_mst=s["io_mst"],
            mst_excess=s["mst_excess"], route_minus_io_rg=s["route_minus_io_rg"],
            io_mst_minus_route=s["io_mst_minus_route"],
        ))
    return dict(rows=rows)


# ---------------------------------------------------------------------------
# Table 6: FT three-value table -- route_ft not evaluable (sentinel)
# ---------------------------------------------------------------------------

def table6_ft_three_value(samples):
    rows = []
    any_route_ft = False
    for s in samples:
        any_route_ft = any_route_ft or s["route_ft_evaluable"]
        rows.append(dict(
            sample_id=s["sample_id"], ft_rg=s["ft_rg"], ft_mst=s["ft_mst"],
            route_ft=s["route_ft_total"] if s["route_ft_evaluable"] else None,
        ))
    return dict(
        rows=rows,
        route_ft_not_evaluable=not any_route_ft,
        route_ft_not_evaluable_reason=(
            None if any_route_ft else
            "per_net_route_ft is the -1 'not computed' sentinel for every net in every S8 "
            "sample (ioplace/route_eval/route_crossings.py:373: a net missing from the "
            "pin_regions argument to evaluate_route gets route_ft=-1); S8's extraction call "
            "never supplied pin_regions, so route_ft was never actually computed for this "
            "corpus."
        ),
    )


# ---------------------------------------------------------------------------
# Table 7: boundary-pair demand correlation -- not evaluable
# ---------------------------------------------------------------------------

def table7_boundary_pair_demand():
    return dict(
        not_evaluable=True,
        reason=("route_pair_demand is persisted per sample (crossings_k*.json), but "
                "EvalResult.boundary_pair_demand (the evaluator-side counterpart) is never "
                "serialized anywhere in S8's output -- metrics.json has no such field. "
                "Recomputing it needs the same GPU evaluator rerun table 2 needs."),
    )


# ---------------------------------------------------------------------------
# Sample-list table (spec sec 8.2 item 4) and unevaluable-sample table (item 5)
# ---------------------------------------------------------------------------

def sample_list_table(samples):
    rows = []
    for s in samples:
        rows.append(dict(
            sample_id=s["sample_id"], design=s["design"], arm=s["arm"],
            k_placement=s["k_placement"],
            route_cross_dw_matched_k=s["route_cross_dw"],
            other_k=s["other_k"], route_cross_dw_other_k=s["other_k_route_cross_dw"],
            drc_final_violations=s["drc_final_violations_log"],
            drc_rpt_violation_lines=s["drc_rpt_violation_lines"],
            route_elapsed_s=s["route_elapsed_s"],
            route_done=s["route_done_route"],
            verify_s2_overall_pass=s["verify_s2_overall_pass"],
        ))
    return dict(rows=rows)


def unevaluable_sample_table(s8_dir):
    # Hand-classified from or_run/openroad_route.log +
    # results/stage2/s8/route_chain*.log / extract_chain.log (2026-08-18
    # read of the actual on-disk logs, not guessed).
    classification = {
        ("matrix_mult_1", "flat"): dict(
            failure_mode="congestion_plateau_timeout",
            note="detailed_route capped at -droute_end_iter 5; violations plateaued at "
                 "~544k across the 5 capped iterations and the uncapped continuation "
                 "iteration, never dropping meaningfully; route.tcl was killed by the "
                 "14400s (4h) wall-clock timeout before write_def ran (last recorded: "
                 "544,931 violations at 40% of the post-cap iteration, elapsed 26:47).",
        ),
        ("matrix_mult_1", "ours_k16"): dict(
            failure_mode="congestion_plateau_timeout",
            note="same pattern as flat; last recorded 528,897 violations at 50% "
                 "completion, elapsed 30:59, before the 4h timeout kill.",
        ),
        ("matrix_mult_1", "ours_k32"): dict(
            failure_mode="congestion_plateau_timeout",
            note="same pattern; last recorded 532,973 violations at 50% completion, "
                 "elapsed 30:17, before the 4h timeout kill.",
        ),
        ("superblue19", "flat"): dict(
            failure_mode="congestion_plateau_timeout",
            note="0th optimization iteration alone ran past 3h with violations "
                 "escalating (124,751 at 10% -> 2,199,855 at 50%, elapsed 3:06:13); "
                 "killed by the 4h wall-clock timeout mid-iteration.",
        ),
        ("superblue19", "ours_k16"): dict(
            failure_mode="congestion_plateau_timeout",
            note="same pattern; 2,317,033 violations at 50% completion of the 0th "
                 "iteration, elapsed 3:08:34, before the timeout kill.",
        ),
        ("superblue19", "ours_k32"): dict(
            failure_mode="dr_0th_iteration_timeout",
            note="log ends immediately at 'Start 0th optimization iteration' with zero "
                 "'Completing X%' progress lines logged at all -- the 4h budget was "
                 "consumed before the first progress checkpoint of detailed route.",
        ),
        ("superblue12", "ours_k16"): dict(
            failure_mode="dr_0th_iteration_timeout",
            note="same as superblue19 ours_k32: log ends at 'Start 0th optimization "
                 "iteration', no progress lines at all before the timeout kill.",
        ),
        ("superblue12", "ours_k32"): dict(
            failure_mode="killed_during_global_route_setup",
            note="log (2.8KB) ends mid-setup, right after the 'Timing is not available' "
                 "GRT-0300 warning, well before any global_route progress or "
                 "detailed_route start -- weaker evidence than the other rows (no "
                 "explicit [FAIL] line was ever written for this arm in "
                 "route_chain_sb12.log, consistent with the outer batch script itself "
                 "being interrupted mid-attempt rather than route.tcl completing and "
                 "failing cleanly).",
        ),
        ("superblue12", "flat"): dict(
            failure_mode="route_not_attempted_queue_exhausted",
            note="no or_run/ directory exists at all for this arm. superblue12__flat's "
                 "out.def (169,029,018 bytes) is the single largest file in the whole S8 "
                 "corpus, and stage2_s8_route.sh routes smallest-file-first with "
                 "concurrency=1 -- results/stage2/s8/extract_chain.log confirms: "
                 "'[FAIL] superblue12__flat: no routed.def (route not run yet?)'. This is "
                 "not a route failure; the batch simply never reached this arm within its "
                 "wall-clock budget.",
        ),
    }
    rows = []
    for design, arm in UNEVALUABLE_SAMPLES:
        info = build_unevaluable_sample(s8_dir, design, arm)
        cls = classification[(design, arm)]
        rows.append(dict(**info, **cls))
    return dict(rows=rows)


# ---------------------------------------------------------------------------
# Sec 8.3: three-component NNLS regression, pooled across the 6 samples
# ---------------------------------------------------------------------------

def fit_regression(samples, n_boot=2000, seed=0):
    """route[e] ~ alpha*(lam-1) + beta*(ST-(lam-1)) + gamma*(io_mst-ST), fit
    at the TOTAL (per-sample-sum) level -- sec 8.3's per-net form is not
    fittable here (table 2's per-net-array gap), so each of the 6 VALID
    samples contributes one (x1,x2,x3,y) row: x1=lam_minus_1, x2=ft_rg
    (== io_rg-lam_minus_1 == ST-(lam-1) summed), x3=mst_excess
    (== io_mst-io_rg summed), y=route_cross_dw (matched-K, delta=2). Fit by
    non-negative least squares (no intercept, matching spec sec 8.3's
    formulation), report R^2, per-sample residuals, and a bootstrap 95% CI
    over the 6 rows -- explicitly low-reliability given n=6, 3 free
    coefficients (3 residual dof)."""
    X = np.array([[s["lam_minus_1"], s["ft_rg"], s["mst_excess"]] for s in samples],
                 dtype=np.float64)
    y = np.array([s["route_cross_dw"] for s in samples], dtype=np.float64)
    ids = [s["sample_id"] for s in samples]
    n, p = X.shape
    dof = n - p

    coef, resid_norm = nnls(X, y)
    y_pred = X @ coef
    resid = y - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    per_sample = [
        dict(sample_id=ids[i], y_actual=float(y[i]), y_pred=float(y_pred[i]),
             residual=float(resid[i]),
             relative_residual=float(resid[i] / y[i]) if y[i] else None)
        for i in range(n)
    ]

    rng = np.random.default_rng(seed)
    boot_coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        if np.linalg.matrix_rank(Xb) < p:
            continue
        cb, _ = nnls(Xb, yb)
        boot_coefs.append(cb)
    boot_coefs = np.array(boot_coefs) if boot_coefs else np.zeros((0, p))
    if len(boot_coefs):
        ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
        ci_hi = np.percentile(boot_coefs, 97.5, axis=0)
    else:
        ci_lo = ci_hi = np.full(p, np.nan)

    names = ["alpha", "beta", "gamma"]
    coefficients = {
        name: dict(value=float(coef[i]), ci95_lo=float(ci_lo[i]), ci95_hi=float(ci_hi[i]))
        for i, name in enumerate(names)
    }

    return dict(
        n_samples=n, n_regressors=p, dof=dof,
        n_bootstrap_successful=int(len(boot_coefs)), n_bootstrap_requested=n_boot,
        regressor_definitions=dict(
            x1="lam_minus_1 = hard_lambda_sum = sum(lambda_e - 1)",
            x2="ft_rg = io_rg - hard_lambda_sum = sum(ST_e - (lambda_e - 1))",
            x3="mst_excess = io_mst - io_rg = sum(io_mst_e - ST_e)",
            y="route_cross_dw(delta=2), matched K",
        ),
        coefficients=coefficients, r_squared=r2, residual_sum_squares=ss_res,
        per_sample=per_sample,
        reliability_note=(
            f"n={n} samples, {p} free (non-negative) coefficients, {dof} residual dof -- "
            "this is a small-n fit at the per-sample-TOTAL level (not per-net, see module "
            "docstring), and the 95% CIs above are bootstrap-over-6-rows, which is itself "
            "weak with only 6 rows to resample. Report per spec sec 8.3's instruction to "
            "disclose the n=6 limitation honestly rather than overstate precision."
        ),
    )


def fit_per_design_regression(samples):
    """Informational only (feeds C5's discussion, not a pass/fail gate):
    the same NNLS fit but run separately per design (3 samples, 3
    coefficients -> 0 residual dof, an exact fit with zero information about
    stability). Spec sec 8.4 C5 needs >= 3 design to compute a coefficient
    of variation; this corpus has 2 (des_perf_1, mempool_tile_wrap), so C5
    itself stays not_evaluable regardless of what these numbers say."""
    by_design = {}
    for s in samples:
        by_design.setdefault(s["design"], []).append(s)
    out = {}
    for design, rows in by_design.items():
        X = np.array([[s["lam_minus_1"], s["ft_rg"], s["mst_excess"]] for s in rows],
                     dtype=np.float64)
        y = np.array([s["route_cross_dw"] for s in rows], dtype=np.float64)
        coef, _ = nnls(X, y)
        out[design] = dict(alpha=float(coef[0]), beta=float(coef[1]), gamma=float(coef[2]),
                            n_samples=len(rows), dof=len(rows) - 3,
                            note="exact fit (0 residual dof) -- purely informational")
    return out


# ---------------------------------------------------------------------------
# Sec 8.4: C1-C5 verdicts
# ---------------------------------------------------------------------------

def judge_c1(table1):
    r_rg = table1["pooled"]["R_io_rg"]
    r_mst = table1["pooled"]["R_io_mst"]
    closer = "io_rg" if abs(r_rg - 1) < abs(r_mst - 1) else "io_mst"
    return dict(
        verdict="not_evaluable",
        reason=("C1 needs rho(io_rg, route) vs rho(io_mst, route) at the per-net level "
                "(table 2), which is not_evaluable (no per-net evaluator array persisted). "
                "The total-ratio half of the rule can be computed as supplementary "
                "evidence only -- it alone cannot satisfy C1's AND condition."),
        supplementary_ratio_evidence=dict(
            R_io_rg=r_rg, R_io_mst=r_mst,
            closer_to_1=closer,
            note=f"|R_io_rg-1|={abs(r_rg-1):.4f} vs |R_io_mst-1|={abs(r_mst-1):.4f}; "
                 f"pooled total ratio alone favors the {closer} model, but this is not a "
                 "C1 verdict (per-net rho is required and unavailable).",
        ),
    )


def judge_c2_c3(samples, table1, regression):
    r_lam = table1["pooled"]["R_lam_minus_1"]
    r_rg = table1["pooled"]["R_io_rg"]
    r_mst = table1["pooled"]["R_io_mst"]
    e3_pass = all(0.80 <= r <= 1.25 for r in (r_lam, r_rg, r_mst))
    c2 = dict(e3_pass=e3_pass, pooled_ratios=dict(R_lam_minus_1=r_lam, R_io_rg=r_rg, R_io_mst=r_mst))
    if not e3_pass:
        alpha = regression["coefficients"]["alpha"]["value"]
        beta = regression["coefficients"]["beta"]["value"]
        gamma = regression["coefficients"]["gamma"]["value"]
        io_calibrated = []
        for s in samples:
            val = alpha * s["lam_minus_1"] + beta * s["ft_rg"] + gamma * s["mst_excess"]
            io_calibrated.append(dict(
                sample_id=s["sample_id"], io_calibrated=val,
                route_actual=s["route_cross_dw"],
                residual=s["route_cross_dw"] - val,
            ))
        c2["action"] = "E3 failed -- io_calibrated column added"
        c2["io_calibrated"] = io_calibrated
    else:
        c2["action"] = "E3 passed -- no io_calibrated column needed"
    alpha = regression["coefficients"]["alpha"]["value"]
    beta = regression["coefficients"]["beta"]["value"]
    kappa_ft = (beta / alpha) if alpha != 0 else None
    c3 = dict(alpha_hat=alpha, beta_hat=beta, kappa_ft=kappa_ft,
              note="kappa_ft <- beta_hat/alpha_hat per spec sec 8.4 C3 -- "
                   "the real-router feed-through-to-necessary-crossing cost ratio, "
                   "to replace M3's kappa_ft=1.0 default.")
    return c2, c3


def judge_c4(samples):
    """Sign-invariance check (spec sec 8.4 C4): does flat -> ours_k16 -> ours_k32
    move the same direction pre- and post-route, per design? Uses
    hard_lambda_sum (pre-route evaluator metric, M2's reporting metric) and
    route_cross_dw (post-route ground truth) -- both totals, both
    available for all 6 samples."""
    by_design = {}
    for s in samples:
        by_design.setdefault(s["design"], {})[s["arm"]] = s
    rows = []
    for design, arms in by_design.items():
        if not all(a in arms for a in ("flat", "ours_k16", "ours_k32")):
            continue
        for pair in [("flat", "ours_k16"), ("ours_k16", "ours_k32")]:
            a, b = pair
            pre_delta = arms[b]["lam_minus_1"] - arms[a]["lam_minus_1"]
            post_delta = arms[b]["route_cross_dw"] - arms[a]["route_cross_dw"]
            sign_flip = (pre_delta * post_delta) < 0
            rows.append(dict(
                design=design, from_arm=a, to_arm=b,
                pre_route_metric="hard_lambda_sum", pre_route_delta=pre_delta,
                post_route_metric="route_cross_dw", post_route_delta=post_delta,
                sign_flip=bool(sign_flip),
            ))
    any_flip = any(r["sign_flip"] for r in rows)
    return dict(rows=rows, any_sign_flip=any_flip,
                 verdict=("sign_flip_detected" if any_flip else "sign_invariant"))


def judge_c5(samples):
    designs = sorted({s["design"] for s in samples})
    return dict(
        verdict="not_evaluable",
        reason=(f"C5 needs >= 3 design with valid routed samples to compute a coefficient "
                f"of variation across their (alpha_hat, beta_hat, gamma_hat); this corpus "
                f"has {len(designs)} ({', '.join(designs)}). Per-design fits are reported "
                "as informational only (see per_design_regression), not a CV verdict."),
        n_designs_available=len(designs), n_designs_required=3,
    )


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def build_calibration(s8_dir=S8_DIR_DEFAULT):
    samples = [build_sample(s8_dir, design, arm) for design, arm in VALID_SAMPLES]

    t1 = table1_total_ratio(samples)
    t2 = table2_per_net_correlation()
    t3 = table3_degree_bucket()
    t4 = table4_lambda_route_bucket(samples)
    t5 = table5_three_value_decomposition(samples)
    t6 = table6_ft_three_value(samples)
    t7 = table7_boundary_pair_demand()
    tsample = sample_list_table(samples)
    tuneval = unevaluable_sample_table(s8_dir)

    regression = fit_regression(samples)
    per_design_regression = fit_per_design_regression(samples)

    c1 = judge_c1(t1)
    c2, c3 = judge_c2_c3(samples, t1, regression)
    c4 = judge_c4(samples)
    c5 = judge_c5(samples)

    # raw vs dw(2) drift check (G4, spec sec 9.2)
    raw_total = sum(s["route_cross_raw"] for s in samples)
    dw_total = sum(s["route_cross_dw"] for s in samples)
    g4_drift_pct = (raw_total - dw_total) / dw_total * 100 if dw_total else None
    g4 = dict(
        route_cross_raw_total=raw_total, route_cross_dw2_total=dw_total,
        drift_pct=g4_drift_pct, triggered=(g4_drift_pct is not None and g4_drift_pct > 15),
        note=("only delta in {0(raw),2} was extracted for S8 -- the spec's full delta scan "
              "{0,1,2,4} (sec 7.4/L-Q4-a) was not run; re-deriving delta=1/4 from "
              "segments.npz would need re-invoking route_eval.route_crossings, out of scope "
              "for this read-existing-artifacts pass."),
    )

    for s in samples:
        del s["per_net_lambda_route"], s["per_net_route_cross_dw"], s["per_net_route_wl"]

    return dict(
        spec="docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md sec 8/10 S9",
        s8_dir=s8_dir,
        valid_samples=samples,
        table1_total_ratio=t1,
        table2_per_net_correlation=t2,
        table3_degree_bucket=t3,
        table4_lambda_route_bucket=t4,
        table5_three_value_decomposition=t5,
        table6_ft_three_value=t6,
        table7_boundary_pair_demand=t7,
        sample_list_table=tsample,
        unevaluable_sample_table=tuneval,
        regression=regression,
        per_design_regression_informational=per_design_regression,
        c1_model_selection=c1,
        c2_regression_calibration=c2,
        c3_kappa_ft=c3,
        c4_sign_invariance=c4,
        c5_transfer_cv=c5,
        g4_raw_vs_dw2_drift=g4,
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s8-dir", default=S8_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    result = build_calibration(s8_dir=args.s8_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    tables_path = os.path.join(args.out_dir, "tables.json")
    with open(tables_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=_json_default)
    print(f"[stage2_calibration] {len(result['valid_samples'])} valid sample(s), "
          f"{len(result['unevaluable_sample_table']['rows'])} unevaluable sample(s) -> "
          f"{tables_path}")


if __name__ == "__main__":
    main()
