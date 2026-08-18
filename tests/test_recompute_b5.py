"""Tests for `ioplace.bench.recompute_b5` (2026-08-15 T6 holdout adjudication
doc Appendix A.3/A.4/A.5's B-5 recompute). All toy-scale, deterministic --
no GPU/placement, no dependence on the real `results/m4/bench/arrays`
manifests (those are covered separately by running the module's `main()`
against the real arrays root, done manually as part of this task, not here)."""
import json
import math
import os

import pytest

from ioplace.bench import glue_gen
from ioplace.bench import recompute_b5 as rb


def _toy_manifest(R, C, lambda_0, alpha, normalization, budget_pairs=3.0,
                   sampled_pair_counts=None, shape="toy"):
    """Minimal manifest with just the fields `recompute_b5`'s functions
    read (`R`/`C` at top level, `t6.{normalization,lambda_0_tile,alpha,
    expected_pair_counts,sampled_pair_counts}`, `glue.{n_nets,n_pins}`).
    `sampled_pair_counts` defaults to the deterministic rounded expectation
    (mode="expected") -- Poisson-noise-free, so B-5c's z is ~0 unless the
    caller overrides it to inject noise/breakage."""
    if normalization == "n2":
        expected, _diag = glue_gen.n2_pair_counts_sinkhorn(lambda_0, alpha, R, C,
                                                             budget_pairs=budget_pairs)
    elif normalization == "n1":
        dists = glue_gen.tile_pair_distances(R, C)
        expected = {pair: lambda_0 * glue_gen.phi(d, alpha) for pair, d in dists.items()}
    else:
        raise ValueError(normalization)

    if sampled_pair_counts is None:
        sampled_pair_counts = {pair: round(v) for pair, v in expected.items()}

    n_nets = sum(sampled_pair_counts.values())
    return {
        "R": R, "C": C,
        "base": {"source_n_nets": 1000},  # only `check_h5prime_self_consistency`'s
                                            # `implied_cross_tile_share` diagnostic reads this
        "glue": {"n_nets": n_nets, "n_pins": 2 * n_nets, "kernel": "power_law",
                  "lambda_0": lambda_0, "alpha": alpha, "seed": 0},
        "t6": {
            "shape": shape, "normalization": normalization, "primary_seed": 0,
            "lambda_0_tile": lambda_0, "alpha": alpha,
            "expected_pair_counts": {f"{a}|{b}": v for (a, b), v in expected.items()},
            "sampled_pair_counts": {f"{a}|{b}": v for (a, b), v in sampled_pair_counts.items()},
            "n_glue_nets": n_nets, "n_glue_pins": 2 * n_nets,
        },
    }


# ---------------------------------------------------------------------------
# B-5a/b/c per array (Appendix A.3)
# ---------------------------------------------------------------------------

def test_b5_for_glue_array_ok_on_a_correctly_built_n2_toy_manifest():
    manifest = _toy_manifest(R=2, C=2, lambda_0=1000.0, alpha=0.5, normalization="n2")
    res = rb._b5_for_glue_array(manifest)
    assert res["b5a_recipe_arithmetic"]["status"] == "ok"
    assert res["b5a_recipe_arithmetic"]["rel_err"] <= 1e-3
    assert res["b5b_manifest_fidelity"]["status"] == "ok"
    # deterministic (mode="expected") sampled counts -> zero sampling noise
    assert res["b5c_sampling_noise"]["z"] == pytest.approx(0.0, abs=1e-6)
    assert res["b5c_sampling_noise"]["rel_err"] == pytest.approx(0.0, abs=1e-6)
    assert res["formula"] == "sinkhorn"


def test_b5_for_glue_array_ok_on_a_correctly_built_n1_toy_manifest():
    manifest = _toy_manifest(R=1, C=2, lambda_0=500.0, alpha=0.3, normalization="n1")
    res = rb._b5_for_glue_array(manifest)
    assert res["b5a_recipe_arithmetic"]["status"] == "ok"
    assert res["b5b_manifest_fidelity"]["status"] == "ok"
    assert res["formula"] == "sum_lambda0_phi"


def test_b5_for_glue_array_b5a_detects_normalization_mismatch():
    """The Appendix A.2 bug, reproduced on a toy manifest: an N1-formula
    `expected_pair_counts` dict recorded under a declared `n2` array."""
    R, C, lambda_0, alpha = 2, 2, 1000.0, 0.5
    dists = glue_gen.tile_pair_distances(R, C)
    n1_expected = {pair: lambda_0 * glue_gen.phi(d, alpha) for pair, d in dists.items()}
    manifest = {
        "R": R, "C": C,
        "glue": {"n_nets": round(sum(n1_expected.values())), "n_pins": 0},
        "t6": {"shape": "toy", "normalization": "n2", "lambda_0_tile": lambda_0, "alpha": alpha,
               "expected_pair_counts": {f"{a}|{b}": v for (a, b), v in n1_expected.items()},
               "sampled_pair_counts": {f"{a}|{b}": round(v) for (a, b), v in n1_expected.items()}},
    }
    manifest["glue"]["n_nets"] = sum(manifest["t6"]["sampled_pair_counts"].values())
    manifest["glue"]["n_pins"] = 2 * manifest["glue"]["n_nets"]
    res = rb._b5_for_glue_array(manifest)
    assert res["b5a_recipe_arithmetic"]["status"] == "normalization_mismatch"
    assert res["b5a_recipe_arithmetic"]["other_normalization"] == "n1"


def test_b5_for_glue_array_b5b_fails_on_bookkeeping_break():
    manifest = _toy_manifest(R=2, C=2, lambda_0=1000.0, alpha=0.5, normalization="n2")
    manifest["glue"]["n_nets"] += 1  # now disagrees with sum(sampled_pair_counts)
    res = rb._b5_for_glue_array(manifest)
    assert res["b5b_manifest_fidelity"]["status"] == "fail"


def test_b5c_rel_err_matches_z_times_sqrt_expected():
    manifest = _toy_manifest(R=2, C=2, lambda_0=1000.0, alpha=0.5, normalization="n2")
    # inject a materialized count that differs from the deterministic recipe
    manifest["glue"]["n_nets"] = manifest["t6"]["n_glue_nets"] + 50
    res = rb._b5_for_glue_array(manifest)
    b5c = res["b5c_sampling_noise"]
    # rel_err = |actual-expected|/expected, z = (actual-expected)/sqrt(expected)
    # => rel_err = |z|/sqrt(expected).
    assert b5c["rel_err"] == pytest.approx(abs(b5c["z"]) / math.sqrt(b5c["expected_total"]))
    assert b5c["status"] == "reported"  # disclosure only, never gates


# ---------------------------------------------------------------------------
# 3x3 kernel sensitivity / pre-registered budget (Appendix A.3's "四核 x
# 五陣列", reproduced here on the actual production constants so a
# regression breaks this test, not silently drifts)
# ---------------------------------------------------------------------------

def test_kernel_variant_b5a_ref_total_is_kernel_independent():
    lambda_0, R, C = 4387.31377772961, 3, 3
    ref_totals = set()
    for alpha, kernel in [(0.1023, "power_law"), (0.0, "power_law"),
                           (1.915, "power_law"), (0.1023, "truncated")]:
        res = rb._kernel_variant_b5a(lambda_0, R, C, alpha, kernel)
        ref_totals.add(round(res["ref_total"], 6))
    assert len(ref_totals) == 1, "N2's per-pair total must not depend on kernel/alpha (sec 5.2)"


def test_3x3_kernel_sensitivity_hits_the_preregistered_budget():
    """Locks in Appendix A.3's pre-registered decisive rel_err values
    against the real production (lambda_0_tile, alpha) constants -- this is
    the exact computation `recompute_b5.main()` runs against the real
    `3x3_n2` manifest's own (lambda_0_tile, alpha)."""
    manifest = {
        "R": 3, "C": 3,
        "t6": {"lambda_0_tile": 4387.31377772961, "alpha": 0.10230633302323285},
    }
    sens = rb._3x3_kernel_sensitivity(manifest)
    for name, check in sens["budget_check"].items():
        assert check["hit"], (name, check)
        assert not check["order_of_magnitude_miss"], (name, check)
    assert sens["mainline"]["status"] == "ok"
    for variant in sens["variants"].values():
        assert variant["status"] == "ok"


# ---------------------------------------------------------------------------
# A.5 per-pair sign test / goodness of fit (disclosure only)
# ---------------------------------------------------------------------------

def test_sign_test_pvalue_is_1_for_a_perfectly_balanced_split():
    assert rb._sign_test_pvalue(k=5, n=10) == pytest.approx(1.0)


def test_sign_test_pvalue_is_small_for_a_lopsided_split():
    assert rb._sign_test_pvalue(k=19, n=20) < 0.001


def test_poisson_goodness_of_fit_is_zero_when_sampled_equals_expected():
    gof = rb._poisson_goodness_of_fit(sampled=[100, 200, 50], expected=[100.0, 200.0, 50.0])
    assert gof["chi2_stat"] == pytest.approx(0.0)
    assert gof["df"] == 3
    assert gof["chi2_p_value"] == pytest.approx(1.0)


def test_poisson_goodness_of_fit_stat_matches_pearson_formula():
    sampled, expected = [110, 90, 205], [100.0, 100.0, 200.0]
    gof = rb._poisson_goodness_of_fit(sampled, expected)
    expected_stat = sum((s - e) ** 2 / e for s, e in zip(sampled, expected))
    assert gof["chi2_stat"] == pytest.approx(expected_stat)


def test_a5_per_pair_signs_filters_by_min_expected(tmp_path):
    """Builds a full toy `arrays_root` (the 5 glue arrays `_a5_per_pair_
    signs` reads by name) and checks the n>=min_expected filter and sign
    counts on a hand-constructed pair set."""
    arrays_root = str(tmp_path / "arrays")
    for name in rb.GLUE_ARRAYS:
        os.makedirs(os.path.join(arrays_root, name), exist_ok=True)
        expected = {"(0, 0)|(0, 1)": 100.0, "(0, 0)|(1, 0)": 10.0}  # second pair below n=42
        sampled = {"(0, 0)|(0, 1)": 110, "(0, 0)|(1, 0)": 12}  # both sampled > expected
        manifest = {"t6": {"expected_pair_counts": expected, "sampled_pair_counts": sampled}}
        with open(os.path.join(arrays_root, name, f"{name}.manifest.json"), "w") as f:
            json.dump(manifest, f)

    out = rb._a5_per_pair_signs(arrays_root, min_expected=42)
    for name in rb.GLUE_ARRAYS:
        entry = out[name]
        assert entry["n_pairs_total"] == 2
        assert entry["n_pairs_filtered_ge_min_expected"] == 1  # only the n=100 pair
        assert entry["n_positive"] == 1
        assert entry["n_negative"] == 0
        assert entry["goodness_of_fit"]["df"] == 1


# ---------------------------------------------------------------------------
# End-to-end `recompute()` over a toy arrays_root
# ---------------------------------------------------------------------------

def _write_toy_arrays_root(tmp_path):
    arrays_root = str(tmp_path / "arrays")
    lambda_0, alpha = 1000.0, 0.4
    shapes = {"1x2_n1": (1, 2, "n1"), "1x2_n2": (1, 2, "n2"), "2x2_n1": (2, 2, "n1"),
              "2x2_n2": (2, 2, "n2"), "3x3_n2": (3, 3, "n2")}
    for name, (R, C, norm) in shapes.items():
        manifest = _toy_manifest(R, C, lambda_0, alpha, norm, shape=name.split("_")[0])
        d = os.path.join(arrays_root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.manifest.json"), "w") as f:
            json.dump(manifest, f)
    for name in rb.NOGLUE_ARRAYS:
        d = os.path.join(arrays_root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.manifest.json"), "w") as f:
            json.dump({"R": 2, "C": 2, "glue": None, "no_glue": True}, f)
    return arrays_root


def test_recompute_end_to_end_over_a_toy_arrays_root(tmp_path):
    arrays_root = _write_toy_arrays_root(tmp_path)
    result = rb.recompute(arrays_root)

    for name in rb.GLUE_ARRAYS:
        assert result["arrays"][name]["b5a_recipe_arithmetic"]["status"] == "ok"
        assert result["arrays"][name]["b5b_manifest_fidelity"]["status"] == "ok"
    assert "kernel_sensitivity" in result["arrays"]["3x3_n2"]

    for name in rb.NOGLUE_ARRAYS:
        assert result["arrays"][name]["status"] == "not_applicable"

    assert set(result["a4_verification"]) == {
        "2x2_n2_buggy_n1_formula", "2x2_n1_correct_formula", "1x2_n2_correct_formula_poisson_draw"}
    assert set(result["a5_per_pair_signs"]) == set(rb.GLUE_ARRAYS)


def test_recompute_is_deterministic_across_repeated_runs(tmp_path):
    arrays_root = _write_toy_arrays_root(tmp_path)
    r1 = rb.recompute(arrays_root)
    r2 = rb.recompute(arrays_root)
    assert r1 == r2


# ---------------------------------------------------------------------------
# glue_poisson_gof_report -- pooled summary (this task's separate
# results/m4/bench/glue_poisson_gof.json output)
# ---------------------------------------------------------------------------

def test_pooled_a5_summary_matches_the_sum_of_the_per_array_breakdown(tmp_path):
    arrays_root = _write_toy_arrays_root(tmp_path)
    per_array = rb._a5_per_pair_signs(arrays_root)
    pooled = rb._pooled_a5_summary(arrays_root)
    assert pooled["n_pairs"] == sum(a["n_pairs_filtered_ge_min_expected"] for a in per_array.values())
    assert pooled["n_positive"] == sum(a["n_positive"] for a in per_array.values())
    assert pooled["n_negative"] == sum(a["n_negative"] for a in per_array.values())
    assert pooled["goodness_of_fit"]["df"] == pooled["n_pairs"]
    assert pooled["goodness_of_fit"]["chi2_stat"] == pytest.approx(
        sum(a["goodness_of_fit"]["chi2_stat"] for a in per_array.values()))


def test_glue_poisson_gof_report_never_gates_and_has_a_conclusion_string(tmp_path):
    arrays_root = _write_toy_arrays_root(tmp_path)
    report = rb.glue_poisson_gof_report(arrays_root)
    assert set(report) == {"provenance", "per_array", "pooled", "conclusion"}
    assert "status" not in report  # disclosure only -- no pass/fail verdict at this level
    assert isinstance(report["conclusion"], str) and len(report["conclusion"]) > 0
    assert str(report["pooled"]["n_pairs"]) in report["conclusion"]
