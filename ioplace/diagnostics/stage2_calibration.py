"""Stage 2 calibration from paired, persisted evaluator and routed-wire evidence.

Historical scalar-only artifacts retain an explicit degraded path. New cohorts
carry per-net evidence recomputed on the actual routed DEF, exact fingerprints,
and a delta scan. This module performs CPU analysis only.
"""
import argparse
import hashlib
from pathlib import Path
import json
import os
import re

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from ioplace.export.evaluation import load_evaluation

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_delta_scan(case, k, primary):
    path = os.path.join(case, f"delta_scan_k{k}.json")
    if not os.path.exists(path):
        return None
    scan = _read_json(path)
    for key in ("routed_def_sha256", "segments_sha256", "evaluator_sha256"):
        if scan[key] != primary.get(key):
            raise ValueError(f"delta scan {key} mismatch")
    if sorted(row["delta"] for row in scan["rows"]) != [0, 1, 2, 4]:
        raise ValueError("incomplete or duplicate delta scan")
    for row in scan["rows"]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("delta scan artifact digest mismatch")
        value = _read_json(row["path"])
        if (value["delta"] != row["delta"] or value["total_route_cross_dw"] != row["route_cross_dw"]
                or value["total_route_cross_raw"] != row["route_cross_raw"]
                or any(value.get(key) != scan[key] for key in
                       ("routed_def_sha256", "segments_sha256", "evaluator_sha256"))):
            raise ValueError("delta scan row does not describe its referenced artifact")
    return scan


def load_metrics(s8_dir, design, arm):
    return _read_json(os.path.join(_case_dir(s8_dir, design, arm), "metrics.json"))


def validate_case_receipt(case, k, matched):
    """Require independently checked identities and a complete evidence chain."""
    case = Path(case).resolve()
    receipt = _read_json(case / "evidence.execution.json")
    if receipt.get("status") != "completed":
        raise ValueError("paired evidence pipeline incomplete")
    required = [case / "or_run" / name for name in (
        "verify_identity.json", "verify_s2.json", "pin_geometry.npz",
        "pin_geometry.npz.json", "pin_geometry_input.npz", "pin_geometry_input.npz.json",
        "segments.npz", "segments.provenance.json")]
    required += [Path(matched["evaluator_file"]).resolve(), case / f"crossings_k{k}.json",
                 case / f"delta_scan_k{k}.json", case / f"boundary_evidence_k{k}.json"]
    outputs = receipt.get("outputs", {})
    if any(str(path) not in outputs for path in required):
        raise ValueError("paired receipt omits required identity or evaluator evidence")
    inputs = receipt.get("inputs", {})
    for path in (case / "or_run/fixed.def", case / "or_run/routed.def",
                 case / f"evaluator_k{k}.npz", case / "metrics.json"):
        if str(path) not in inputs:
            raise ValueError("paired receipt omits required placement input")
    for label, entries in (("input", inputs), ("output", outputs)):
        for path, digest in entries.items():
            if sha256_file(path) != digest:
                raise ValueError(f"paired evidence {label} changed: {path}")
    run = case / "or_run"
    identity = _read_json(run / "verify_identity.json")
    if not identity.get("overall_pass"):
        raise ValueError("routing identity verification failed")
    for key, path in (("input_def_sha256", run / "fixed.def"),
                      ("routed_def_sha256", run / "routed.def"),
                      ("before_geometry_sha256", run / "pin_geometry_input.npz"),
                      ("after_geometry_sha256", run / "pin_geometry.npz")):
        if identity.get(key) != sha256_file(path):
            raise ValueError(f"identity evidence mismatch: {key}")
    if not _read_json(run / "verify_s2.json").get("overall_pass"):
        raise ValueError("routed wire verification failed")
    return receipt


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
    if matched["delta"] != 2:
        raise ValueError("primary calibration must use delta2")

    evidence = None
    ev_path = matched.get("evaluator_file")
    if ev_path:
        validate_case_receipt(case, k, matched)
        if sha256_file(ev_path) != matched.get("evaluator_sha256"):
            raise ValueError(f"{design}__{arm}: evaluator file digest mismatch")
        ev = load_evaluation(ev_path)
        meta = ev["metadata"]
        for key in ("region", "placement", "net_order"):
            if meta[key+"_sha256"] != matched.get("evaluator_"+key+"_sha256"):
                raise ValueError(f"{design}__{arm}: evaluator {key} mismatch")
        if (meta["k"] != k or meta["num_nets"] != matched["num_nets"]
                or meta["provenance"].get("placement_stage") != "router_def"
                or meta["provenance"].get("router_def_sha256") != matched.get("routed_def_sha256")):
            raise ValueError(f"{design}__{arm}: evaluator was not paired to the routed geometry")
        evidence = ev

    route_ft_arr = np.asarray(matched["per_net_route_ft"], dtype=np.int64)
    route_ft_complete = bool(len(route_ft_arr) and np.all(route_ft_arr >= 0))
    lam_route_arr = np.asarray(matched["per_net_lambda_route"], dtype=np.int64)

    verify_s2_path = os.path.join(case, "or_run", "verify_s2.json")
    verify_s2 = _read_json(verify_s2_path) if os.path.exists(verify_s2_path) else None

    log_path = os.path.join(case, "or_run", "openroad_route.log")
    log_info = parse_route_log(log_path)
    drc_rpt_path = os.path.join(case, "or_run", "drc.rpt")
    drc_rpt_count = parse_drc_rpt(drc_rpt_path)

    totals = evidence["metadata"]["totals"] if evidence is not None else metrics
    lam_minus_1 = int(totals["hard_lambda_sum"])
    io_rg = int(totals["io_rg"])
    ft_rg = int(totals["ft_rg"])
    io_mst = int(totals["io_count"])
    ft_mst = int(totals["ft_count"])
    route_cross_raw = int(matched["total_route_cross_raw"])
    route_cross_dw = int(matched["total_route_cross_dw"])
    route_wl = int(matched["total_route_wl"])
    delta = matched["delta"]

    sample = dict(
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
        unmatched_net_indices=matched.get("unmatched_net_indices", []),
        per_net_calibration_eligible=matched.get("per_net_calibration_eligible"),
        per_net_has_routed_wire=matched.get("per_net_has_routed_wire"),
        unrouted_signal_fraction=matched.get("unrouted_signal_fraction"),
        unrouted_signal_gate_pass=matched.get("unrouted_signal_gate_pass"),
        pre_route_lam_minus_1=int(metrics["hard_lambda_sum"]),
        evaluator_stage="router_def" if evidence is not None else "legacy_gp_metrics",
        router_geometry_delta=({key:totals[key]-metrics[key] for key in
            ("hard_lambda_sum", "io_rg", "ft_rg", "io_count", "ft_count", "hpwl")}
            if evidence is not None else None),
        delta_scan=load_delta_scan(case, k, matched),
        delta=delta,
        # evaluator totals (matched K)
        lam_minus_1=lam_minus_1, io_rg=io_rg, ft_rg=ft_rg, io_mst=io_mst, ft_mst=ft_mst,
        tree_wl=float(totals["tree_wl"]), hpwl=float(totals["hpwl"]),
        # route totals (matched K, delta=2 primary report value per sec 7.4)
        route_cross_raw=route_cross_raw, route_cross_dw=route_cross_dw, route_wl=route_wl,
        route_ft_evaluable=route_ft_complete,
        route_ft_coverage=float(np.mean(route_ft_arr >= 0)) if len(route_ft_arr) else 0.,
        route_ft_known_sum=int(route_ft_arr[route_ft_arr >= 0].sum()),
        route_ft_total=(int(np.sum(route_ft_arr[route_ft_arr >= 0]))
                         if route_ft_complete else None),
        route_pair_demand=matched["route_pair_demand"],
        # route-side-only Lambda_route distribution (table 4)
        lambda_route_stats=dict(
            min=int(lam_route_arr.min()), max=int(lam_route_arr.max()),
            mean=float(lam_route_arr.mean()), median=float(np.median(lam_route_arr)),
        ),
        per_net_lambda_route=lam_route_arr,
        per_net_route_cross_dw=np.asarray(matched["per_net_route_cross_dw"], dtype=np.int64),
        per_net_route_wl=np.asarray(matched["per_net_route_wl"], dtype=np.int64),
        evaluator=evidence,
        evaluator_match_reason=("matched" if evidence is not None else
                                "missing or mismatched evaluator provenance"),
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
    if evidence is not None:
        gp = load_evaluation(os.path.join(case, f"evaluator_k{k}.npz"), net_names=evidence["net_names"])
        if (gp["metadata"]["region_sha256"] != evidence["metadata"]["region_sha256"]
                or gp["metadata"]["k"] != k
                or gp["metadata"]["totals"]["hard_lambda_sum"] != metrics["hard_lambda_sum"]):
            raise ValueError("pre-route evaluator identity/region/scalar mismatch")
        sample["pre_route_evaluator"] = gp
        route, models, _, _, formal = _paired_arrays(sample)
        sample["whole_design_totals"] = {key:sample[key] for key in (
            "lam_minus_1", "io_rg", "ft_rg", "io_mst", "ft_mst", "route_cross_dw", "route_cross_raw")}
        sample["calibration_population"] = "matched routed signal nets with degree2..max_degree; sample unrouted<=2%"
        sample["calibration_nets"] = int(formal.sum())
        for key in ("lam_minus_1", "io_rg", "io_mst"):
            sample[key] = int(models[key][formal].sum())
        sample["ft_rg"] = sample["io_rg"]-sample["lam_minus_1"]
        sample["ft_mst"] = int(evidence["per_net_ft"][formal].sum())
        sample["route_cross_dw"] = int(route[formal].sum())
        sample["route_cross_raw"] = int(np.asarray(matched["per_net_route_cross_raw"])[formal].sum())
        if sample["delta_scan"] is not None:
            scan = sample["delta_scan"]
            formal_rows = []
            for row in scan["rows"]:
                value = _read_json(row["path"])
                dw = np.asarray(value["per_net_route_cross_dw"], dtype=np.int64)
                raw = np.asarray(value["per_net_route_cross_raw"], dtype=np.int64)
                if dw.shape != formal.shape or raw.shape != formal.shape:
                    raise ValueError("delta scan per-net population shape mismatch")
                formal_rows.append(dict(row, route_cross_dw=int(dw[formal].sum()),
                    route_cross_raw=int(raw[formal].sum()), population="primary formal eligible mask"))
            primary = next(row for row in formal_rows if row["delta"] == 2)
            if (primary["route_cross_dw"] != sample["route_cross_dw"]
                    or primary["route_cross_raw"] != sample["route_cross_raw"]):
                raise ValueError("formal delta2 totals differ from primary calibration")
            sample["delta_scan"] = dict(scan, rows=formal_rows,
                whole_design_supplementary=scan["rows"], n_nets=int(formal.sum()),
                population="same primary formal eligible mask for every delta")
        sample["route_ft_evaluable"] = bool(formal.any() and np.all(route_ft_arr[formal]>=0))
        sample["route_ft_total"] = int(route_ft_arr[formal].sum()) if sample["route_ft_evaluable"] else None
        sample["route_ft_known_sum"] = int(route_ft_arr[formal & (route_ft_arr>=0)].sum())
        sample["route_ft_coverage"] = float(np.mean(route_ft_arr[formal]>=0)) if formal.any() else 0.
        sample.update(mst_excess=sample["io_mst"]-sample["io_rg"],
            route_minus_io_rg=sample["route_cross_dw"]-sample["io_rg"],
            io_mst_minus_route=sample["io_mst"]-sample["route_cross_dw"])
    boundary_path=os.path.join(case,f"boundary_evidence_k{k}.json")
    if os.path.exists(boundary_path):
        boundary=_read_json(boundary_path)
        if (boundary["evaluator_sha256"]!=matched.get("evaluator_sha256")
                or boundary["crossings_sha256"]!=sha256_file(sample["source_paths"]["crossings_matched"])):
            raise ValueError("boundary evidence population/artifact mismatch")
        sample["boundary_evidence"]=boundary
    return sample


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
    paired = [s for s in samples if s.get("evaluator") is not None]
    supplementary = [s["sample_id"] for s in samples if s.get("evaluator") is None] if paired else []
    if paired:
        samples = paired
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
    pooled = {name: pooled_route / den if den else None for name, den in (
        ("R_lam_minus_1", sum(s["lam_minus_1"] for s in samples)),
        ("R_io_rg", sum(s["io_rg"] for s in samples)),
        ("R_io_mst", sum(s["io_mst"] for s in samples)))}
    paired = sum(s.get("evaluator") is not None for s in samples)
    return dict(rows=rows, pooled=pooled, legacy_supplementary_samples=supplementary,
                per_net=dict(not_evaluable=paired == 0, n_paired_samples=paired,
                             reason=None if paired else "no paired per-net evaluator evidence"))


# ---------------------------------------------------------------------------
# Table 2: per-net correlation -- not evaluable, same root cause as table 1
# ---------------------------------------------------------------------------

def _corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("correlation arrays have unequal lengths")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(pearson=None, spearman=None, n=len(x))
    return dict(pearson=float(pearsonr(x, y).statistic),
                spearman=float(spearmanr(x, y).statistic), n=len(x))


def _paired_arrays(sample):
    ev = sample["evaluator"]
    route = np.asarray(sample["per_net_route_cross_dw"], dtype=np.float64)
    models = dict(io_rg=np.asarray(ev["per_net_steiner"], dtype=np.float64),
                  io_mst=np.asarray(ev["per_net_crossings"], dtype=np.float64),
                  lam_minus_1=np.maximum(np.asarray(ev["per_net_lambda"], dtype=np.float64)-1, 0))
    degree = np.asarray(ev["net_degrees"])
    if any(a.shape != route.shape for a in [*models.values(), degree]) or route.ndim != 1:
        raise ValueError(f"{sample['sample_id']}: evaluator and route arrays have unequal lengths")
    matched = np.ones(len(route), dtype=bool)
    unmatched = np.asarray(sample.get("unmatched_net_indices", []), dtype=np.int64)
    if np.any((unmatched < 0) | (unmatched >= len(route))):
        raise ValueError("unmatched net index outside netlist")
    matched[unmatched] = False
    formal = matched & (degree >= 2) & (degree <= ev["metadata"].get("max_degree", 256))
    routed = sample.get("per_net_calibration_eligible")
    if routed is not None:
        routed = np.asarray(routed,dtype=bool)
        if routed.shape != route.shape:
            raise ValueError("routed population mask shape mismatch")
        formal &= routed
    if sample.get("unrouted_signal_gate_pass") is False:
        formal[:] = False
    return route, models, degree, matched, formal


def table2_per_net_correlation(samples=None):
    rows = []
    pooled_values = {name: ([], []) for name in ("io_rg", "io_mst", "lam_minus_1")}
    for sample in samples or []:
        if sample.get("evaluator") is None:
            continue
        route, models, degree, matched, formal = _paired_arrays(sample)
        active = (route > 0) | np.logical_or.reduce([value>0 for value in models.values()])
        for name, value in models.items():
            mask = formal & active
            rows.append(dict(sample_id=sample["sample_id"], model=name,
                             design=sample.get("design"),k=sample.get("k_placement"),
                             **_corr(value[mask], route[mask]),
                             excluded_unmatched=int((~matched).sum()),
                             excluded_large_net=int((matched & (degree>sample["evaluator"]["metadata"].get("max_degree",256))).sum()),
                             excluded_ineligible=int((matched & ~formal).sum())))
            pooled_values[name][0].append(value[mask])
            pooled_values[name][1].append(route[mask])
    pooled = []
    for name, (xs, ys) in pooled_values.items():
        pooled.append(dict(model=name, **_corr(np.concatenate(xs) if xs else np.array([]),
                                               np.concatenate(ys) if ys else np.array([]))))
    return dict(not_evaluable=not any(row["n"] for row in rows), rows=rows, pooled=pooled,
                reason=None if rows else "no paired per-net evaluator evidence",
                mask_definition="common eligible population & (route>0 OR any_model>0)")


def table3_degree_bucket(samples=None):
    from ioplace.ops.io_term import DEG_BUCKET_EDGES, DEG_BUCKET_LABELS
    rows = []
    bounds = list(zip(DEG_BUCKET_LABELS, DEG_BUCKET_EDGES[:-1], DEG_BUCKET_EDGES[1:]))
    bounds += [("100-256", 100, 257), (">256 (lower-bound)", 257, float("inf"))]
    for sample in samples or []:
        if sample.get("evaluator") is None:
            continue
        route, models, degree, matched, formal = _paired_arrays(sample)
        routed=sample.get("per_net_has_routed_wire")
        eligible=matched & (np.asarray(routed,dtype=bool) if routed is not None else True)
        if sample.get("unrouted_signal_gate_pass") is False:
            eligible[:]=False
        for label, low, high in bounds:
            mask = eligible & (degree >= low) & (degree < high)
            row = dict(sample_id=sample["sample_id"], bucket=label, n=int(mask.sum()), models={})
            for name, values in models.items():
                den = float(values[mask].sum())
                ratio = float(route[mask].sum()) / den if den else None
                corr_mask = mask & formal & ((route > 0) | (values > 0))
                row["models"][name] = dict(R_X=ratio, **_corr(values[corr_mask], route[corr_mask]))
                # Historical table consumers use these ratio field names.
                row[{"io_mst":"R_X", "io_rg":"R_RG", "lam_minus_1":"R_lambda"}[name]] = ratio
            rows.append(row)
    available = any(row["n"] for row in rows)
    return dict(not_evaluable=not available, rows=rows,
                reason=None if available else "no eligible paired degree/evaluator evidence")


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
    degraded = []
    for sample in samples:
        ev = sample.get("evaluator")
        if ev is not None:
            route, models, _, matched, formal = _paired_arrays(sample)
            buckets = (("evaluator_lambda", np.asarray(ev["per_net_lambda"])),
                       ("route_lambda", sample["per_net_lambda_route"]))
        else:
            degraded.append(sample["sample_id"])
            route = sample["per_net_route_cross_dw"]
            matched = np.ones(len(route), dtype=bool)
            buckets = (("route_lambda", sample["per_net_lambda_route"]),)
        for basis, lam in buckets:
            for label, (low, high) in zip(LAMBDA_BUCKET_LABELS, LAMBDA_BUCKET_EDGES):
                mask = (formal if ev is not None else matched) & (lam >= low) & (lam <= high if high else True)
                n = int(mask.sum())
                row = dict(sample_id=sample["sample_id"], bucket_basis=basis,
                    lambda_route_bucket=label, n_nets=n,
                    sum_route_cross_dw=int(route[mask].sum()),
                    mean_route_cross_dw=float(route[mask].mean()) if n else None,
                    sum_route_wl=int(sample["per_net_route_wl"][mask].sum()))
                if ev is not None:
                    row["models"] = {}
                    for name, value in models.items():
                        den = float(value[mask].sum())
                        corr_mask = mask & formal & ((route > 0) | (value > 0))
                        row["models"][name] = dict(R_X=float(route[mask].sum())/den if den else None,
                                                  **_corr(value[corr_mask], route[corr_mask]))
                rows.append(row)
    return dict(rows=rows, degraded=bool(degraded), degraded_samples=degraded,
                degraded_reason="no paired evaluator evidence" if degraded else None)


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
            route_ft_coverage=s.get("route_ft_coverage"), route_ft_known_sum=s.get("route_ft_known_sum"),
        ))
    return dict(
        rows=rows,
        route_ft_not_evaluable=not any_route_ft,
        route_ft_not_evaluable_reason=(
            None if any_route_ft else
            "No sample has a nonempty eligible population with complete route FT; "
            "unknown values remain excluded from totals. See coverage and known sums."
        ),
    )


# ---------------------------------------------------------------------------
# Table 7: boundary-pair demand correlation -- not evaluable
# ---------------------------------------------------------------------------

def table7_boundary_pair_demand(samples=None):
    if samples:
        rows=[]
        for s in samples:
            boundary=s.get("boundary_evidence")
            if boundary is not None:
                if boundary.get("not_evaluable"):
                    rows.append(dict(sample_id=s["sample_id"],**boundary))
                    continue
                pairs=boundary["pairs"]
                x=np.array([row["evaluator"] for row in pairs],dtype=float)
                y=np.array([row["route"] for row in pairs],dtype=float)
                length=np.array([row["length"] for row in pairs],dtype=float)
                rows.append(dict(sample_id=s["sample_id"],**_corr(x,y),pairs=pairs,
                    population=boundary["population"],n_nets=boundary["n_nets"],
                    per_length_correlation=_corr(x/length,y/length),
                    evaluator_statistics=boundary["evaluator_statistics"],
                    route_statistics=boundary["route_statistics"]))
                continue
            ev=s.get("evaluator")
            if ev is None or "boundary_pairs" not in ev: continue
            route={tuple(map(int,k.split(","))):v for k,v in s.get("route_pair_demand",{}).items()}
            pairs={tuple(x):int(v) for x,v in zip(np.asarray(ev["boundary_pairs"]), ev["boundary_demand"])}
            keys=sorted(set(route)|set(pairs)); x=np.array([pairs.get(k,0) for k in keys]); y=np.array([route.get(k,0) for k in keys])
            c=_corr(x,y) or dict(pearson=None,spearman=None,n=0); rows.append(dict(sample_id=s["sample_id"], **c,
                not_evaluable=True, reason="unfiltered boundary populations; supplementary only",
                pairs=[dict(pair=list(k), evaluator=int(pairs.get(k,0)), route=int(route.get(k,0))) for k in keys]))
        return dict(not_evaluable=not any(not r.get("not_evaluable") and r.get("n",0)>0 for r in rows), rows=rows)
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
    """NNLS on paired per-net rows, or explicitly degraded legacy totals.

    Net rows within a design are dependent. We do not attach a misleading
    IID-net bootstrap interval to the per-net fit. Legacy total-row bootstrap
    results retain their historical, explicitly limited interpretation.
    """
    blocks, targets, groups = [], [], []
    paired = [sample for sample in samples if sample.get("evaluator") is not None]
    if paired:
        for sample in paired:
            route, models, _, _, formal = _paired_arrays(sample)
            lam, rg, mst = (models[key] for key in ("lam_minus_1", "io_rg", "io_mst"))
            mask = formal & ((route > 0) | (mst > 0) | (rg > 0) | (lam > 0))
            blocks.append(np.column_stack((lam[mask], (rg-lam)[mask], (mst-rg)[mask])))
            targets.append(route[mask])
            groups.append((sample["sample_id"], int(mask.sum())))
        X, y = np.concatenate(blocks), np.concatenate(targets)
        level = "per_net"
    else:
        X = np.asarray([[v["lam_minus_1"], v["ft_rg"], v["mst_excess"]] for v in samples], dtype=float).reshape(-1, 3)
        y = np.asarray([v["route_cross_dw"] for v in samples], dtype=float)
        groups = [(v["sample_id"], 1) for v in samples]
        level = "sample_total"
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X)) if n else 0
    scales = np.linalg.norm(X, axis=0) if n else np.zeros(p)
    condition = float(np.linalg.cond(X / scales)) if rank == p else None
    coef = nnls(X, y)[0] if n else np.zeros(p)
    prediction = X @ coef
    residual = y - prediction
    ss_res = float(residual @ residual)
    ss_tot = float(np.sum((y - y.mean())**2)) if n else 0.
    grouped = []
    offset = 0
    for name, count in groups:
        section = slice(offset, offset+count)
        actual, estimate = float(y[section].sum()), float(prediction[section].sum())
        grouped.append(dict(sample_id=name, n_rows=count, y_actual=actual, y_pred=estimate,
                            residual=actual-estimate,
                            residual_sum_squares=float(np.sum(residual[section]**2)),
                            relative_residual=(actual-estimate)/actual if actual else None))
        offset += count
    requested = n_boot if level == "sample_total" else 0
    bootstrap = []
    rng = np.random.default_rng(seed)
    if rank == p:
        for _ in range(requested):
            indices = rng.integers(0, n, size=n)
            if np.linalg.matrix_rank(X[indices]) == p:
                bootstrap.append(nnls(X[indices], y[indices])[0])
    ci = np.percentile(bootstrap, [2.5, 97.5], axis=0) if bootstrap else None
    coefficients = {name: dict(value=float(coef[i]),
        ci95_lo=float(ci[0,i]) if ci is not None else None,
        ci95_hi=float(ci[1,i]) if ci is not None else None)
        for i,name in enumerate(("alpha", "beta", "gamma"))}
    return dict(n_samples=n, n_regressors=p, rank=rank, dof=n-rank, level=level,
        identifiable=rank == p, normalized_condition_number=condition,
        n_bootstrap_requested=requested, n_bootstrap_successful=len(bootstrap),
        coefficients=coefficients, r_squared=1-ss_res/ss_tot if ss_tot else None,
        residual_sum_squares=ss_res, per_sample=grouped,
        regressor_definitions=dict(x1="max(lambda-1,0)", x2="ST-max(lambda-1,0)",
                                    x3="io_mst-ST", y="route_cross_dw(2)"),
        reliability_note=(f"n={n}, rank={rank}, residual dof={n-rank}; " +
            ("paired per-net fit; clustered-design uncertainty not estimated; no IID-net CI claimed"
             if paired else "legacy total-row bootstrap; few independent designs limit inference")))


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
        fit = fit_regression(rows, n_boot=0)
        c = fit["coefficients"]
        out[design] = dict(alpha=c["alpha"]["value"], beta=c["beta"]["value"], gamma=c["gamma"]["value"],
                            n_samples=fit["n_samples"], rank=fit["rank"], dof=fit["dof"], identifiable=fit["identifiable"])
    return out


# ---------------------------------------------------------------------------
# Sec 8.4: C1-C5 verdicts
# ---------------------------------------------------------------------------

def judge_c1(table1, table2=None):
    if any(table1["pooled"].get(key) is None for key in ("R_io_rg","R_io_mst")):
        return dict(verdict="not_evaluable", reason="undefined paired total ratio")
    if table2 and not table2.get("not_evaluable"):
        correlations = {}
        for row in table2.get("rows", []):
            correlations.setdefault(row["sample_id"], {})[row["model"]] = row["spearman"]
        comparisons = []
        for row in table1.get("rows", []):
            corr = correlations.get(row["sample_id"], {})
            rg, mst = corr.get("io_rg"), corr.get("io_mst")
            ratios = (row["R_io_rg"], row["R_io_mst"])
            valid = rg is not None and mst is not None and all(v is not None for v in ratios)
            verdict = ("rg_selected" if rg > mst + .05 and abs(ratios[0]-1)<abs(ratios[1]-1)
                       else "reconsider_rg_model") if valid else "not_evaluable"
            comparisons.append(dict(sample_id=row["sample_id"], rho_rg=rg, rho_mst=mst,
                                    R_io_rg=ratios[0], R_io_mst=ratios[1], verdict=verdict))
        # The original threshold is applied independently to each sample. No
        # post-hoc majority/pooled weighting rule is introduced for this cohort.
        return dict(verdict="per_sample_only" if comparisons else "not_evaluable", comparisons=comparisons,
                    scope="C1 thresholds evaluated independently; no registered cross-sample aggregation",
                    pooled_supplementary=table2.get("pooled", []))
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
    if any(value is None for value in (r_lam,r_rg,r_mst)):
        unavailable=dict(verdict="not_evaluable",reason="undefined paired total ratio")
        return unavailable,dict(unavailable)
    e3_pass = all(0.80 <= r <= 1.25 for r in (r_lam, r_rg, r_mst))
    c2 = dict(e3_pass=e3_pass, pooled_ratios=dict(R_lam_minus_1=r_lam, R_io_rg=r_rg, R_io_mst=r_mst))
    if not e3_pass:
        alpha = regression["coefficients"]["alpha"]["value"]
        beta = regression["coefficients"]["beta"]["value"]
        gamma = regression["coefficients"]["gamma"]["value"]
        io_calibrated = []
        for s in samples:
            if regression.get("level") == "per_net" and s.get("evaluator") is None:
                continue
            val = alpha * s["lam_minus_1"] + beta * s["ft_rg"] + gamma * s["mst_excess"]
            io_calibrated.append(dict(
                sample_id=s["sample_id"], io_calibrated=val,
                route_actual=s["route_cross_dw"],
                residual=s["route_cross_dw"] - val,
            ))
        if regression.get("identifiable", True):
            c2["action"] = "E3 failed -- io_calibrated column added"
            c2["io_calibrated"] = io_calibrated
        else:
            c2["action"] = "E3 failed; calibration not identifiable"
            c2["supplementary_in_sample_fit"] = io_calibrated
    else:
        c2["action"] = "E3 passed -- no io_calibrated column needed"
    alpha = regression["coefficients"]["alpha"]["value"]
    beta = regression["coefficients"]["beta"]["value"]
    kappa_ft = (beta / alpha) if alpha != 0 else None
    if not regression.get("identifiable",True) or kappa_ft is None:
        return c2,dict(verdict="not_evaluable",reason="rank-deficient fit or zero alpha",kappa_ft=None)
    c3 = dict(alpha_hat=alpha, beta_hat=beta, kappa_ft=kappa_ft,
              uncertainty="clustered-design interval unavailable; estimate is not a validated objective weight",
              note="kappa_ft <- beta_hat/alpha_hat per spec sec 8.4 C3 -- "
                   "estimated real-router feed-through-to-necessary-crossing cost ratio; "
                   "no placement objective or historical M3 weight is changed by this analysis.")
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
        pairs = [(f"flat_k{k}", f"ours_k{k}") for k in (16,32)
                 if f"flat_k{k}" in arms and f"ours_k{k}" in arms]
        if not pairs and all(a in arms for a in ("flat", "ours_k16", "ours_k32")):
            pairs = [("flat", "ours_k16"), ("ours_k16", "ours_k32")]
        legacy = "flat" in arms and not any(a.startswith("flat_k") for a in arms)
        for pair in pairs:
            a, b = pair
            left, right = arms[a], arms[b]
            count = None
            if not legacy:
                unavailable = dict(design=design, from_arm=a, to_arm=b,
                    comparison_scope="matched_K_unavailable", not_evaluable=True,
                    sign_flip=False, reason="missing common eligible GP/route population")
                if any(s.get("pre_route_evaluator") is None or s.get("evaluator") is None
                       or s.get("unrouted_signal_gate_pass") is not True for s in (left,right)):
                    rows.append(unavailable)
                    continue
                lmask, rmask = _paired_arrays(left)[4], _paired_arrays(right)[4]
                li = {str(n):i for i,n in enumerate(left["evaluator"]["net_names"]) if lmask[i]}
                ri = {str(n):i for i,n in enumerate(right["evaluator"]["net_names"]) if rmask[i]}
                common = sorted(li.keys() & ri.keys())
                if not common:
                    rows.append(unavailable)
                    continue
                lidx, ridx = np.array([li[n] for n in common]), np.array([ri[n] for n in common])
                lgp, rgp = left["pre_route_evaluator"], right["pre_route_evaluator"]
                pre_delta = int(np.maximum(rgp["per_net_lambda"][ridx]-1,0).sum()
                                -np.maximum(lgp["per_net_lambda"][lidx]-1,0).sum())
                post_delta = int(right["per_net_route_cross_dw"][ridx].sum()
                                 -left["per_net_route_cross_dw"][lidx].sum())
                count = len(common)
            else:
                pre_delta = right.get("pre_route_lam_minus_1", right["lam_minus_1"]) - left.get("pre_route_lam_minus_1", left["lam_minus_1"])
                post_delta = right["route_cross_dw"] - left["route_cross_dw"]
            sign_flip = (pre_delta * post_delta) < 0
            rows.append(dict(
                design=design, from_arm=a, to_arm=b,
                pre_route_metric="hard_lambda_sum", pre_route_delta=pre_delta,
                post_route_metric="route_cross_dw", post_route_delta=post_delta,
                sign_flip=bool(sign_flip),
                common_eligible_nets=count,
                comparison_scope="legacy_cross_K_supplementary" if legacy else "matched_K",
            ))
    any_flip = any(r["sign_flip"] for r in rows)
    return dict(rows=rows, any_sign_flip=any_flip,
                 formal_rows=[r for r in rows if r["comparison_scope"]=="matched_K"],
                 formal_verdict=("not_evaluable" if not any(r["comparison_scope"]=="matched_K" for r in rows)
                    else "sign_flip_detected" if any(r["sign_flip"] for r in rows if r["comparison_scope"]=="matched_K")
                    else "sign_invariant"),
                 verdict=("not_evaluable" if not any(not r.get("not_evaluable") for r in rows)
                          else "sign_flip_detected" if any_flip else "sign_invariant"))


def judge_c5(samples):
    designs = sorted({s.get("design") for s in samples})
    if len(designs) < 3:
        return dict(verdict="not_evaluable", n_designs_available=len(designs), n_designs_required=3,
                    reason="fewer than 3 identifiable design fits")
    fits = fit_per_design_regression(samples)
    valid = [v for v in fits.values() if v["identifiable"]]
    if len(valid) < 3:
        return dict(verdict="not_evaluable", n_designs_available=len(valid), n_designs_required=3,
                    reason="fewer than 3 identifiable design fits")
    vals = np.array([[v["alpha"], v["beta"], v["gamma"]] for v in valid])
    mean = vals.mean(axis=0); sd = vals.std(axis=0, ddof=1)
    cv = np.divide(sd, np.abs(mean), out=np.full(3, np.nan), where=np.abs(mean) > 0)
    if np.any(~np.isfinite(cv)):
        return dict(verdict="not_evaluable", n_designs_available=len(valid),
                    reason="zero-mean coefficient has undefined relative variation")
    return dict(verdict="fail" if np.any(cv > .25) else "pass", n_designs_available=len(valid),
                cv=dict(zip(("alpha", "beta", "gamma"), cv.tolist())))


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def build_calibration(s8_dir=S8_DIR_DEFAULT, sample_pairs=None):
    cohort_path = Path(s8_dir) / "cohort.json"
    if sample_pairs is not None:
        pairs = sample_pairs
    elif cohort_path.exists():
        pairs = [(row["design"], row["arm"]) for row in _read_json(cohort_path)]
    else:
        pairs = VALID_SAMPLES
    samples, missing = [], []
    for design, arm in pairs:
        try:
            samples.append(build_sample(s8_dir, design, arm))
        except FileNotFoundError as error:
            missing.append(dict(sample_id=f"{design}__{arm}", reason=str(error)))
    if not samples:
        raise ValueError("no evaluable routed samples in requested cohort")

    t1 = table1_total_ratio(samples)
    t2 = table2_per_net_correlation(samples)
    t3 = table3_degree_bucket(samples)
    t4 = table4_lambda_route_bucket(samples)
    t5 = table5_three_value_decomposition(samples)
    t6 = table6_ft_three_value(samples)
    t7 = table7_boundary_pair_demand(samples)
    tsample = sample_list_table(samples)
    tuneval = dict(rows=missing) if cohort_path.exists() or sample_pairs is not None else unevaluable_sample_table(s8_dir)

    regression = fit_regression(samples)
    per_design_regression = fit_per_design_regression(samples)

    c1 = judge_c1(t1, t2)
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
        delta_scans=[dict(sample_id=s["sample_id"], scan=s.get("delta_scan")) for s in samples],
        complete=all(s.get("delta_scan") is not None for s in samples),
    )

    clean_samples = []
    for sample in samples:
        clean = {key:value for key,value in sample.items()
                 if not key.startswith("per_net_") and key not in ("evaluator", "pre_route_evaluator")}
        clean["evaluator_metadata"] = sample["evaluator"]["metadata"] if sample.get("evaluator") else None
        clean_samples.append(clean)

    return dict(
        spec="docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md sec 8/10 S9",
        s8_dir=s8_dir,
        requested_sample_pairs=pairs,
        requested_scope="explicit subset" if sample_pairs is not None else "registered cohort" if cohort_path.exists() else "historical cohort",
        complete_requested_scope=not missing,
        registered_cohort_size=len(_read_json(cohort_path)) if cohort_path.exists() else None,
        complete_registered_cohort=(not missing and cohort_path.exists() and
            {tuple(pair) for pair in pairs}=={(row["design"],row["arm"]) for row in _read_json(cohort_path)}),
        valid_samples=clean_samples,
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
    ap.add_argument("--samples", type=json.loads, default=None)
    args = ap.parse_args()

    result = build_calibration(s8_dir=args.s8_dir, sample_pairs=args.samples)
    os.makedirs(args.out_dir, exist_ok=True)
    tables_path = os.path.join(args.out_dir, "tables.json")
    with open(tables_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=_json_default)
    print(f"[stage2_calibration] {len(result['valid_samples'])} valid sample(s), "
          f"{len(result['unevaluable_sample_table']['rows'])} unevaluable sample(s) -> "
          f"{tables_path}")


if __name__ == "__main__":
    main()
