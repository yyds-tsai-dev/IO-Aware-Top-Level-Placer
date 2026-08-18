import json
import os

import numpy as np
import pytest

from ioplace.diagnostics import stage2_calibration as sc


# ---------------------------------------------------------------------------
# parse_route_log / parse_drc_rpt: log-scraping helpers
# ---------------------------------------------------------------------------

def test_parse_route_log_takes_last_elapsed_and_violation(tmp_path):
    log = tmp_path / "openroad_route.log"
    log.write_text(
        "[INFO DRT-0199]   Number of violations = 500.\n"
        "[INFO DRT-0267] cpu time = 00:10:00, elapsed time = 00:05:00, memory = 1.0 (MB)\n"
        "[INFO DRT-0199]   Number of violations = 100.\n"
        "[INFO DRT-0267] cpu time = 00:20:00, elapsed time = 00:15:30, memory = 1.0 (MB)\n"
        "DONE_ROUTE\n"
    )
    info = sc.parse_route_log(str(log))
    assert info["final_dr_violations"] == 100
    assert info["elapsed_s"] == 15 * 60 + 30
    assert info["cpu_s"] == 20 * 60
    assert info["done_route"] is True


def test_parse_route_log_missing_file_returns_nones():
    info = sc.parse_route_log("/nonexistent/path/openroad_route.log")
    assert info == dict(elapsed_s=None, cpu_s=None, final_dr_violations=None, done_route=False)


def test_parse_route_log_no_progress_lines_at_all(tmp_path):
    log = tmp_path / "openroad_route.log"
    log.write_text("[INFO DRT-0195] Start 0th optimization iteration.\n")
    info = sc.parse_route_log(str(log))
    assert info["elapsed_s"] is None
    assert info["final_dr_violations"] is None
    assert info["done_route"] is False


def test_parse_drc_rpt_counts_violation_type_lines(tmp_path):
    rpt = tmp_path / "drc.rpt"
    rpt.write_text(
        "violation type: Short\n\tsrcs: net:a net:b\n"
        "violation type: Short\n\tsrcs: net:c net:d\n"
        "violation type: Min Hole\n\tsrcs: net:e\n"
    )
    assert sc.parse_drc_rpt(str(rpt)) == 3


def test_parse_drc_rpt_missing_file_returns_none():
    assert sc.parse_drc_rpt("/nonexistent/drc.rpt") is None


def test_hhmmss_to_s():
    assert sc._hhmmss_to_s("00:00:06") == 6
    assert sc._hhmmss_to_s("02:39:15") == 2 * 3600 + 39 * 60 + 15


# ---------------------------------------------------------------------------
# build_sample: alignment/derivation on a synthetic S8-shaped case directory
# ---------------------------------------------------------------------------

def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _make_case(root, design, arm, k, num_nets=4, route_ft_sentinel=True):
    case = os.path.join(root, f"{design}__{arm}")
    or_run = os.path.join(case, "or_run")
    os.makedirs(or_run, exist_ok=True)

    metrics = dict(
        k=k, hard_lambda_sum=10, io_rg=15, ft_rg=5, io_count=20, ft_count=8,
        tree_wl=123.0, hpwl=100.0,
    )
    _write_json(os.path.join(case, "metrics.json"), metrics)

    def crossings(kk, dw_total, raw_total):
        per_net_ft = [-1] * num_nets if route_ft_sentinel else [1, 2, 0, 3][:num_nets]
        return dict(
            delta=2, total_route_cross_raw=raw_total, total_route_cross_dw=dw_total,
            total_route_wl=999, num_nets=num_nets, num_unmatched_nets=0,
            per_net_route_cross_raw=[raw_total // num_nets] * num_nets,
            per_net_route_cross_dw=[dw_total // num_nets] * num_nets,
            per_net_lambda_route=[1, 2, 3, 9][:num_nets],
            per_net_route_ft=per_net_ft,
            per_net_route_wl=[10, 20, 30, 40][:num_nets],
            route_pair_demand={"0,1": 3},
        )

    _write_json(os.path.join(case, f"crossings_k{k}.json"), crossings(k, dw_total=12, raw_total=18))
    other_k = 32 if k == 16 else 16
    _write_json(os.path.join(case, f"crossings_k{other_k}.json"),
                crossings(other_k, dw_total=99, raw_total=140))

    _write_json(os.path.join(or_run, "verify_s2.json"), dict(overall_pass=True))
    with open(os.path.join(or_run, "openroad_route.log"), "w") as f:
        f.write("[INFO DRT-0199]   Number of violations = 42.\n"
                "[INFO DRT-0267] cpu time = 00:01:00, elapsed time = 00:00:30, memory = 1 (MB)\n"
                "DONE_ROUTE\n")
    with open(os.path.join(or_run, "drc.rpt"), "w") as f:
        f.write("violation type: Short\n\tsrcs: net:a net:b\n")
    return case


def test_build_sample_derives_three_value_decomposition(tmp_path):
    _make_case(str(tmp_path), "toy", "flat", k=16)
    s = sc.build_sample(str(tmp_path), "toy", "flat")

    assert s["k_placement"] == 16
    assert s["lam_minus_1"] == 10
    assert s["io_rg"] == 15
    assert s["ft_rg"] == 5
    assert s["io_mst"] == 20
    assert s["ft_mst"] == 8
    assert s["route_cross_dw"] == 12
    assert s["route_cross_raw"] == 18
    # sec 1.2 identities
    assert s["mst_excess"] == s["io_mst"] - s["io_rg"] == 5
    assert s["route_minus_io_rg"] == s["route_cross_dw"] - s["io_rg"] == -3
    assert s["io_mst_minus_route"] == s["io_mst"] - s["route_cross_dw"] == 8
    # other-K supplementary point uses the *other* file, not the matched one
    assert s["other_k"] == 32
    assert s["other_k_route_cross_dw"] == 99
    # route_ft sentinel detection
    assert s["route_ft_evaluable"] is False
    assert s["route_ft_total"] is None
    # log/rpt scraping wired through
    assert s["drc_final_violations_log"] == 42
    assert s["drc_rpt_violation_lines"] == 1
    assert s["route_elapsed_s"] == 30
    assert s["verify_s2_overall_pass"] is True


def test_build_sample_route_ft_evaluable_when_not_all_sentinel(tmp_path):
    _make_case(str(tmp_path), "toy", "flat", k=16, route_ft_sentinel=False)
    s = sc.build_sample(str(tmp_path), "toy", "flat")
    assert s["route_ft_evaluable"] is True
    assert s["route_ft_total"] == 1 + 2 + 0 + 3


def test_build_sample_missing_matched_crossings_raises(tmp_path):
    case = os.path.join(str(tmp_path), "toy__flat")
    os.makedirs(case)
    _write_json(os.path.join(case, "metrics.json"), dict(
        k=16, hard_lambda_sum=1, io_rg=1, ft_rg=0, io_count=1, ft_count=0,
        tree_wl=0.0, hpwl=0.0))
    with pytest.raises(FileNotFoundError):
        sc.build_sample(str(tmp_path), "toy", "flat")


# ---------------------------------------------------------------------------
# table1_total_ratio: ratio computation + per-net not_evaluable disclosure
# ---------------------------------------------------------------------------

def _toy_samples(tmp_path, n=2):
    out = []
    for i in range(n):
        design, arm = f"case{i}", "flat"
        _make_case(str(tmp_path), design, arm, k=16)
        out.append(sc.build_sample(str(tmp_path), design, arm))
    return out


def test_table1_total_ratio_values_and_pooling(tmp_path):
    samples = _toy_samples(tmp_path, n=2)
    t1 = sc.table1_total_ratio(samples)
    for row in t1["rows"]:
        assert row["R_lam_minus_1"] == pytest.approx(12 / 10)
        assert row["R_io_rg"] == pytest.approx(12 / 15)
        assert row["R_io_mst"] == pytest.approx(12 / 20)
    # pooled = sum(route)/sum(X) across both toy samples (identical here)
    assert t1["pooled"]["R_lam_minus_1"] == pytest.approx(24 / 20)
    assert t1["per_net"]["not_evaluable"] is True


def test_table2_and_table3_and_table7_are_not_evaluable():
    assert sc.table2_per_net_correlation()["not_evaluable"] is True
    assert sc.table3_degree_bucket()["not_evaluable"] is True
    assert sc.table7_boundary_pair_demand()["not_evaluable"] is True


# ---------------------------------------------------------------------------
# table4: Lambda_route bucketing
# ---------------------------------------------------------------------------

def test_bucket_index_edges():
    assert sc._bucket_index(1) == 0
    assert sc._bucket_index(2) == 1
    assert sc._bucket_index(3) == 2
    assert sc._bucket_index(4) == 3
    assert sc._bucket_index(8) == 3
    assert sc._bucket_index(9) == 4
    assert sc._bucket_index(1000) == 4


def test_table4_lambda_route_bucket_counts_and_sums(tmp_path):
    samples = _toy_samples(tmp_path, n=1)
    t4 = sc.table4_lambda_route_bucket(samples)
    assert t4["degraded"] is True
    # 4-net toy case has per_net_lambda_route = [1, 2, 3, 9] -> one net in
    # each of buckets "1", "2", "3", ">8"; bucket "4-8" is empty.
    counts = {r["lambda_route_bucket"]: r["n_nets"] for r in t4["rows"]}
    assert counts == {"1": 1, "2": 1, "3": 1, "4-8": 0, ">8": 1}
    empty_row = next(r for r in t4["rows"] if r["lambda_route_bucket"] == "4-8")
    assert empty_row["mean_route_cross_dw"] is None
    assert empty_row["sum_route_wl"] == 0


# ---------------------------------------------------------------------------
# table5 / table6
# ---------------------------------------------------------------------------

def test_table5_three_value_decomposition_rows(tmp_path):
    samples = _toy_samples(tmp_path, n=1)
    t5 = sc.table5_three_value_decomposition(samples)
    row = t5["rows"][0]
    assert row["sum_lam_minus_1"] == 10
    assert row["sum_io_rg"] == 15
    assert row["sum_route"] == 12
    assert row["sum_io_mst"] == 20
    assert row["mst_excess"] == 5
    assert row["route_minus_io_rg"] == -3
    assert row["io_mst_minus_route"] == 8


def test_table6_ft_three_value_all_sentinel(tmp_path):
    samples = _toy_samples(tmp_path, n=1)
    t6 = sc.table6_ft_three_value(samples)
    assert t6["route_ft_not_evaluable"] is True
    assert t6["rows"][0]["route_ft"] is None
    assert t6["rows"][0]["ft_rg"] == 5
    assert t6["rows"][0]["ft_mst"] == 8


def test_table6_ft_three_value_reports_when_available(tmp_path):
    design, arm = "toy", "flat"
    _make_case(str(tmp_path), design, arm, k=16, route_ft_sentinel=False)
    s = sc.build_sample(str(tmp_path), design, arm)
    t6 = sc.table6_ft_three_value([s])
    assert t6["route_ft_not_evaluable"] is False
    assert t6["rows"][0]["route_ft"] == 6


# ---------------------------------------------------------------------------
# fit_regression: NNLS recovers exact coefficients on a noiseless synthetic set
# ---------------------------------------------------------------------------

def _synthetic_regression_samples(coeffs, n=6, seed=0):
    rng = np.random.default_rng(seed)
    alpha, beta, gamma = coeffs
    samples = []
    for i in range(n):
        x1 = float(rng.integers(50, 500))
        x2 = float(rng.integers(10, 200))
        x3 = float(rng.integers(0, 100))
        y = alpha * x1 + beta * x2 + gamma * x3
        samples.append(dict(sample_id=f"s{i}", lam_minus_1=x1, ft_rg=x2, mst_excess=x3,
                             route_cross_dw=y))
    return samples


def test_fit_regression_recovers_exact_coefficients_noiseless():
    samples = _synthetic_regression_samples((0.8, 1.2, 0.3))
    fit = sc.fit_regression(samples, n_boot=200, seed=1)
    assert fit["coefficients"]["alpha"]["value"] == pytest.approx(0.8, abs=1e-6)
    assert fit["coefficients"]["beta"]["value"] == pytest.approx(1.2, abs=1e-6)
    assert fit["coefficients"]["gamma"]["value"] == pytest.approx(0.3, abs=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)
    for row in fit["per_sample"]:
        assert row["residual"] == pytest.approx(0.0, abs=1e-6)


def test_fit_regression_nonneg_gamma_clips_to_zero_when_unneeded():
    # y depends only on x1, x2 -- NNLS should not invent a positive gamma.
    samples = _synthetic_regression_samples((1.0, 0.5, 0.0))
    fit = sc.fit_regression(samples, n_boot=50, seed=2)
    assert fit["coefficients"]["gamma"]["value"] == pytest.approx(0.0, abs=1e-6)


def test_fit_regression_reports_n_and_dof():
    samples = _synthetic_regression_samples((1.0, 1.0, 1.0), n=6)
    fit = sc.fit_regression(samples, n_boot=10, seed=3)
    assert fit["n_samples"] == 6
    assert fit["n_regressors"] == 3
    assert fit["dof"] == 3
    assert "n=6" in fit["reliability_note"]


# ---------------------------------------------------------------------------
# C1 / C4 / C5 judgement logic
# ---------------------------------------------------------------------------

def test_judge_c1_not_evaluable_but_reports_closer_ratio():
    t1 = dict(pooled=dict(R_io_rg=0.9, R_io_mst=0.5))
    c1 = sc.judge_c1(t1)
    assert c1["verdict"] == "not_evaluable"
    assert c1["supplementary_ratio_evidence"]["closer_to_1"] == "io_rg"


def test_judge_c4_detects_no_flip_when_directions_match():
    samples = [
        dict(design="d", arm="flat", lam_minus_1=100, route_cross_dw=80),
        dict(design="d", arm="ours_k16", lam_minus_1=90, route_cross_dw=70),
        dict(design="d", arm="ours_k32", lam_minus_1=120, route_cross_dw=95),
    ]
    c4 = sc.judge_c4(samples)
    assert c4["verdict"] == "sign_invariant"
    assert c4["any_sign_flip"] is False


def test_judge_c4_detects_sign_flip():
    samples = [
        dict(design="d", arm="flat", lam_minus_1=100, route_cross_dw=80),
        # pre-route says better (lower lambda), post-route says worse (higher route)
        dict(design="d", arm="ours_k16", lam_minus_1=90, route_cross_dw=95),
        dict(design="d", arm="ours_k32", lam_minus_1=95, route_cross_dw=90),
    ]
    c4 = sc.judge_c4(samples)
    assert c4["verdict"] == "sign_flip_detected"
    assert c4["any_sign_flip"] is True


def test_judge_c5_not_evaluable_with_fewer_than_3_designs():
    samples = [dict(design="a"), dict(design="a"), dict(design="b")]
    c5 = sc.judge_c5(samples)
    assert c5["verdict"] == "not_evaluable"
    assert c5["n_designs_available"] == 2
    assert c5["n_designs_required"] == 3
