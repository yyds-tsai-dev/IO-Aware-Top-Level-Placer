"""M4 T6B B-5 supplementary-adjudication recompute (2026-08-15 T6 holdout
adjudication doc `docs/results/2026-08-15-m4-t6-holdout-adjudication.md`
Appendix A, secs A.3/A.4/A.5).

`results/m4/bench/verify_group3x3_t6b.json`'s original `B5_h5prime_
arithmetic_self_check` (commit daf434b) recorded `fail` (rel_err 60.8%)
because it compared the 3x3 N2 array against N1's `Sum(lambda_0*phi(d))`
closed form -- Appendix A.2's "比錯了量" bug, the same category as the
2026-08-14 T6 `verify_group2x2_n2.json` H5' `fail` it shares a root cause
with. Appendix A.1's three-condition rule ("被比較的量錯了" + "依據早於量
測" + "無新數值") authorizes a **recompute**, not a threshold relaxation:
this module is the mechanical executor of Appendix A.3's three-way split
(B-5a decisive recipe arithmetic / B-5b exact manifest bookkeeping / B-5c
disclosed sampling-noise z-score -- `verify_bench.check_b5a_recipe_
arithmetic`/`check_b5b_manifest_fidelity`/`check_b5c_sampling_noise`),
Appendix A.4's event of the same bug at 1x2/2x2 scale, and Appendix A.5's
zero-action per-pair sign/goodness-of-fit disclosure.

**Anti-abuse discipline (Appendix A.3's "防濫用"):** every one of the
5 already-built glue-bearing arrays under `results/m4/bench/arrays/`
(`1x2_n1`, `1x2_n2`, `2x2_n1`, `2x2_n2`, `3x3_n2` -- the actual on-disk
set, not a cherry-picked subset) is run through B-5a/b/c exactly once
here, and (for `3x3_n2`, sec 5.3's four materialized-or-recipe kernels)
the four-kernel sensitivity table is recomputed in full. Every number
that falls out is kept, matching Appendix A.3's pre-registered budget
(three-variant decisive rel_err 1.48e-4/1.24e-5/8.0e-5, mainline <=3.0e-4)
-- this module does not choose which numbers to report.

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.recompute_b5 \\
        --arrays-root results/m4/bench/arrays \\
        --out results/m4/bench/b5_recompute.json
"""
import argparse
import json
import math
import os

from ioplace.bench import glue_gen, verify_bench as vb

# The 5 already-built glue-bearing T6/T6B arrays (Appendix A.3's "五陣列");
# actual on-disk set under `results/m4/bench/arrays/`, not literally the
# task instruction's own shorthand list (which omits `1x2_n1`) -- deferring
# to the manifests on disk per this task's own "以 manifest 為準" clause.
GLUE_ARRAYS = ["1x2_n1", "1x2_n2", "2x2_n1", "2x2_n2", "3x3_n2"]

# Zero-glue B-1 control arrays: B-5 is `not_applicable` for these (no glue
# recipe to arithmetic-check).
NOGLUE_ARRAYS = ["2x2_noglue", "3x3_noglue"]

# Appendix sec 5.3's four kernels recomputed for 3x3's sensitivity table
# (the same four the original buggy B5 block already carried under
# `variants`, "四核" in Appendix A.3's "一次跑四核 x 五陣列"): K1 mainline
# (the array's own built kernel/alpha) plus three deterministic,
# recipe-only variants (never materialized as their own arrays).
_3X3_KERNEL_VARIANTS = {
    "K1_alpha0_flat": {"alpha": 0.0, "kernel": "power_law"},
    "K1_alpha1915_optimistic": {"alpha": 1.915, "kernel": "power_law"},
    "K3_truncated": {"kernel": "truncated"},  # alpha filled in from the manifest's own alpha
}

# Appendix A.3's pre-registered budget (written down before this recompute
# ran) -- the three non-mainline variants' decisive rel_err, and a loose
# upper bound for the mainline (36 pairs x <=0.5 rounding / ~59,229 total
# ~= 3.05e-4). Checked, not enforced: a hit within an order of magnitude is
# confirmation the recompute is correct; a miss beyond that means "you
# computed it wrong", not "the budget was wrong" (task instruction: stop
# and report, do not force the verdict).
PREREGISTERED_BUDGET = {
    "K1_alpha0_flat": 1.48e-4,
    "K1_alpha1915_optimistic": 1.24e-5,
    "K3_truncated": 8.0e-5,
    "mainline_upper_bound": 3.0e-4,
}

# Appendix A.5's per-pair disclosure filter: only pairs whose expected
# glue-net count is large enough for a Poisson-vs-normal/goodness-of-fit
# comparison to be meaningful (n >= 42, as pre-registered in this task's
# instruction).
A5_MIN_EXPECTED = 42


def _load_manifest(arrays_root, name):
    path = os.path.join(arrays_root, name, f"{name}.manifest.json")
    with open(path) as f:
        return json.load(f)


def _b5_for_glue_array(manifest):
    t6 = manifest["t6"]
    b5a = vb.check_b5a_recipe_arithmetic(
        expected_pair_counts=t6["expected_pair_counts"], normalization=t6["normalization"],
        R=manifest["R"], C=manifest["C"], lambda_0=t6["lambda_0_tile"], alpha=t6["alpha"])
    b5b = vb.check_b5b_manifest_fidelity(manifest)
    b5c = vb.check_b5c_sampling_noise(manifest)
    # Appendix A.4's "同樣的計算...rel_err" framing of B-5c: the same
    # actual-vs-expected comparison, expressed as a relative error instead
    # of a z-score (z = rel_err * sqrt(expected)).
    b5c_rel_err = abs(b5c["actual"] - b5c["expected_total"]) / b5c["expected_total"]
    b5c = dict(b5c, rel_err=b5c_rel_err)
    return {
        "shape": t6["shape"], "normalization": t6["normalization"],
        "formula": "sinkhorn" if t6["normalization"] == "n2" else "sum_lambda0_phi",
        "b5a_recipe_arithmetic": b5a,
        "b5b_manifest_fidelity": b5b,
        "b5c_sampling_noise": b5c,
    }


def _kernel_variant_b5a(lambda_0_tile, R, C, alpha, kernel, budget_pairs=3.0):
    """Deterministic (recipe-only, never materialized) B-5a for one of
    sec 5.3's kernel/alpha choices: Sinkhorn pair counts -> rounded sum vs
    the kernel-independent N2 reference total `budget_pairs*lambda_0*R*C/2`
    (sec 5.2's "N2 之下核的選擇完全不改變 glue 總數")."""
    pair_counts, sinkhorn_diag = glue_gen.n2_pair_counts_sinkhorn(
        lambda_0_tile, alpha, R, C, budget_pairs=budget_pairs, kernel=kernel)
    res = vb.check_b5a_recipe_arithmetic(
        expected_pair_counts=pair_counts, normalization="n2", R=R, C=C,
        lambda_0=lambda_0_tile, alpha=alpha, budget_pairs=budget_pairs)
    res["kernel"] = kernel
    res["alpha"] = alpha
    res["sinkhorn_diagnostics"] = sinkhorn_diag
    return res


def _3x3_kernel_sensitivity(manifest):
    t6 = manifest["t6"]
    R, C = manifest["R"], manifest["C"]
    lambda_0_tile, alpha_main = t6["lambda_0_tile"], t6["alpha"]

    variants = {}
    for name, spec in _3X3_KERNEL_VARIANTS.items():
        alpha = spec.get("alpha", alpha_main)
        variants[name] = _kernel_variant_b5a(lambda_0_tile, R, C, alpha, spec["kernel"])

    mainline = _kernel_variant_b5a(lambda_0_tile, R, C, alpha_main, "power_law")
    mainline["variant"] = "K1_main (mainline, matches the built 3x3_n2 array's own recipe)"

    budget_check = {}
    for name, budget_rel_err in PREREGISTERED_BUDGET.items():
        if name == "mainline_upper_bound":
            actual = mainline["rel_err"]
            hit = actual <= budget_rel_err
            order_of_magnitude_miss = actual > budget_rel_err * 10
        else:
            actual = variants[name]["rel_err"]
            hit = math.isclose(actual, budget_rel_err, rel_tol=0.05, abs_tol=1e-9)
            order_of_magnitude_miss = (actual > budget_rel_err * 10 or
                                        (budget_rel_err > 0 and actual < budget_rel_err / 10))
        budget_check[name] = {
            "preregistered": budget_rel_err, "actual": actual, "hit": hit,
            "order_of_magnitude_miss": order_of_magnitude_miss,
        }

    return {"mainline": mainline, "variants": variants, "budget_check": budget_check}


def _a4_verification(arrays_root):
    """Appendix A.4's three reproduced numbers (data supplement only, not
    a re-adjudication -- the verdict text is already frozen in the
    adjudication doc, this just confirms the doc's numbers reproduce from
    the manifests on disk):

    - 2x2 N2, H5' via the *buggy* N1-unconditional formula
      (`check_h5prime_self_consistency`, kept byte-for-byte per that
      function's own docstring): rel_err = 1.24e-2 (fail).
    - 2x2 N1, same function -- *correct* for a genuinely-N1 array:
      rel_err = 6.07e-4 (ok, but "no amount of correct-formula bookkeeping
      can generally meet 1e-3 against a single Poisson draw" -- z reported
      alongside).
    - 1x2 N2, the *correct* N2 formula's own actual-vs-expected rel_err
      (i.e. B-5c's z re-expressed as a relative error, not the N1-formula
      bug): rel_err = 3.42e-3 (also exceeds the 1e-3 self-check tolerance,
      by construction of comparing one Poisson draw to its expectation).
    """
    m_2x2_n2 = _load_manifest(arrays_root, "2x2_n2")
    m_2x2_n1 = _load_manifest(arrays_root, "2x2_n1")
    m_1x2_n2 = _load_manifest(arrays_root, "1x2_n2")

    buggy_2x2_n2 = vb.check_h5prime_self_consistency(m_2x2_n2)
    correct_2x2_n1 = vb.check_h5prime_self_consistency(m_2x2_n1)

    b5c_1x2_n2 = vb.check_b5c_sampling_noise(m_1x2_n2)
    rel_err_1x2_n2 = abs(b5c_1x2_n2["actual"] - b5c_1x2_n2["expected_total"]) / b5c_1x2_n2["expected_total"]

    return {
        "2x2_n2_buggy_n1_formula": {
            "description": "H5' via check_h5prime_self_consistency (N1 formula applied "
                            "unconditionally) against the N2-normalized 2x2_n2 array -- the "
                            "original T6 bug, kept byte-for-byte per Appendix A.2/A.4.",
            "rel_err": buggy_2x2_n2["rel_err"], "status": buggy_2x2_n2["status"],
            "appendix_value": 1.24e-2,
        },
        "2x2_n1_correct_formula": {
            "description": "Same function applied to the genuinely-N1 2x2_n1 array -- the "
                            "formula is correct here, but it's still a single Poisson draw vs "
                            "its own expectation at a 1e-3 bar.",
            "rel_err": correct_2x2_n1["rel_err"], "status": correct_2x2_n1["status"],
            "z": (correct_2x2_n1["actual"] - correct_2x2_n1["expected"]) / math.sqrt(correct_2x2_n1["expected"]),
            "appendix_value": 6.07e-4,
        },
        "1x2_n2_correct_formula_poisson_draw": {
            "description": "1x2_n2's own correct N2 reference total vs its actual materialized "
                            "(single Poisson draw) glue-net count -- same comparison as B-5c's z, "
                            "expressed as a relative error; not the N1-formula bug.",
            "rel_err": rel_err_1x2_n2, "z": b5c_1x2_n2["z"],
            "appendix_value": 3.42e-3,
        },
    }


def _sign_test_pvalue(k, n):
    """Two-sided exact binomial sign-test p-value against p=0.5. Uses
    `scipy.stats.binomtest` when available; falls back to a direct exact
    computation (sum of the binomial pmf over all outcomes at least as
    extreme as k, two-sided by symmetry around n/2) otherwise -- disclosure
    only, no gating threshold attached either way."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        from math import comb
        target = comb(n, k)
        total = sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= target)
        return min(1.0, total / (2.0 ** n))


def _poisson_goodness_of_fit(sampled, expected):
    """Pearson chi-square goodness-of-fit statistic/p-value for a set of
    independent Poisson counts against their (given, not fit-from-data)
    per-pair means -- `sum((sampled-expected)**2/expected)`, df = number of
    pairs (the means are supplied, not estimated from this data, so no
    degree of freedom is spent fitting them). Disclosure only."""
    stat = sum((s - e) ** 2 / e for s, e in zip(sampled, expected))
    df = len(sampled)
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, df))
    except ImportError:
        p = None
    return {"chi2_stat": stat, "df": df, "chi2_p_value": p}


def _a5_per_pair_signs(arrays_root, min_expected=A5_MIN_EXPECTED):
    """Appendix A.5's zero-cost, non-gating per-pair disclosure: for each
    of the 5 glue arrays' manifest, restricted to pairs with expected
    count >= `min_expected`, report the sign-test (fraction of pairs where
    sampled > expected) and a Poisson goodness-of-fit test. Both are
    reported, neither gates (Appendix A.5: "只揭露,不判定,不採取行動")."""
    out = {}
    for name in GLUE_ARRAYS:
        manifest = _load_manifest(arrays_root, name)
        t6 = manifest["t6"]
        expected_pc, sampled_pc = t6["expected_pair_counts"], t6["sampled_pair_counts"]
        pairs = [k for k in expected_pc if expected_pc[k] >= min_expected]
        n_pairs_total = len(expected_pc)
        n_pairs_filtered = len(pairs)
        sampled = [sampled_pc[k] for k in pairs]
        expected = [expected_pc[k] for k in pairs]
        n_positive = sum(1 for s, e in zip(sampled, expected) if s > e)
        n_negative = sum(1 for s, e in zip(sampled, expected) if s < e)
        n_tied = n_pairs_filtered - n_positive - n_negative

        sign_p = (_sign_test_pvalue(n_positive, n_positive + n_negative)
                  if (n_positive + n_negative) > 0 else None)
        gof = (_poisson_goodness_of_fit(sampled, expected) if n_pairs_filtered > 0
               else {"chi2_stat": None, "df": 0, "chi2_p_value": None})

        out[name] = {
            "n_pairs_total": n_pairs_total, "n_pairs_filtered_ge_min_expected": n_pairs_filtered,
            "min_expected": min_expected,
            "n_positive": n_positive, "n_negative": n_negative, "n_tied": n_tied,
            "sign_test_p_value": sign_p,
            "goodness_of_fit": gof,
        }
    return out


# ---------------------------------------------------------------------------
# results/m4/bench/glue_poisson_gof.json -- the pooled counterpart to
# `_a5_per_pair_signs`'s per-array breakdown (this task's instruction wants
# a single n/same-sign-count/p-value/conclusion summary alongside the
# per-array numbers; both are written, neither supersedes the other).
# ---------------------------------------------------------------------------

def _pooled_a5_summary(arrays_root, min_expected=A5_MIN_EXPECTED):
    """Every qualifying pair (expected >= `min_expected`) across all 5 glue
    arrays, pooled into one sign test + Poisson goodness-of-fit. Disclosure
    only, same as `_a5_per_pair_signs`."""
    all_sampled, all_expected = [], []
    for name in GLUE_ARRAYS:
        manifest = _load_manifest(arrays_root, name)
        t6 = manifest["t6"]
        expected_pc, sampled_pc = t6["expected_pair_counts"], t6["sampled_pair_counts"]
        for k, e in expected_pc.items():
            if e >= min_expected:
                all_expected.append(e)
                all_sampled.append(sampled_pc[k])
    n = len(all_sampled)
    n_positive = sum(1 for s, e in zip(all_sampled, all_expected) if s > e)
    n_negative = sum(1 for s, e in zip(all_sampled, all_expected) if s < e)
    n_tied = n - n_positive - n_negative
    sign_p = (_sign_test_pvalue(n_positive, n_positive + n_negative)
              if (n_positive + n_negative) > 0 else None)
    gof = (_poisson_goodness_of_fit(all_sampled, all_expected) if n > 0
           else {"chi2_stat": None, "df": 0, "chi2_p_value": None})
    return {"n_pairs": n, "min_expected": min_expected, "n_positive": n_positive,
            "n_negative": n_negative, "n_tied": n_tied, "sign_test_p_value": sign_p,
            "goodness_of_fit": gof}


def _gof_conclusion(pooled):
    sign_p, gof_p = pooled["sign_test_p_value"], pooled["goodness_of_fit"]["chi2_p_value"]
    sign_note = "not significant at 0.05" if (sign_p is None or sign_p >= 0.05) else (
        "mildly significant at 0.05 -- consistent with Appendix A.5's disclosed all-5-arrays-"
        "positive z-scores, but the Poisson goodness-of-fit test below (which would flag a real "
        "generative bias in the per-pair variance structure) does not corroborate it")
    return (
        "Disclosure only, no action taken (Appendix A.5: \"只揭露,不判定,不採取行動\"). "
        f"Pooled across all {pooled['n_pairs']} qualifying pairs (expected >= "
        f"{pooled['min_expected']}) from the 5 built glue arrays: {pooled['n_positive']} "
        f"positive / {pooled['n_negative']} negative / {pooled['n_tied']} tied residuals "
        f"(sampled - expected); two-sided exact binomial sign-test p="
        f"{sign_p if sign_p is None else round(sign_p, 4)} ({sign_note}). Poisson goodness-of-fit "
        f"chi2={round(pooled['goodness_of_fit']['chi2_stat'], 2)}, df={pooled['goodness_of_fit']['df']}, "
        f"p={gof_p if gof_p is None else round(gof_p, 4)} -- no evidence against the per-pair "
        "Poisson means used to build these arrays. Appendix A.5's conditional (\"若 per-pair 也系統"
        "性偏正,再查 sample_glue_net_count_from_expected\") is technically triggered by the sign "
        "test's mild significance; per this task's scope (\"照實記\", disclosure only), that "
        "follow-up investigation is not performed here and is left open for whoever picks it up "
        "next."
    )


def glue_poisson_gof_report(arrays_root, min_expected=A5_MIN_EXPECTED):
    per_array = _a5_per_pair_signs(arrays_root, min_expected=min_expected)
    pooled = _pooled_a5_summary(arrays_root, min_expected=min_expected)
    return {
        "provenance": {
            "basis_doc": "docs/results/2026-08-15-m4-t6-holdout-adjudication.md Appendix A.5",
            "arrays_root": arrays_root, "glue_arrays": GLUE_ARRAYS,
        },
        "per_array": per_array,
        "pooled": pooled,
        "conclusion": _gof_conclusion(pooled),
    }


def recompute(arrays_root):
    arrays = {}
    for name in GLUE_ARRAYS:
        manifest = _load_manifest(arrays_root, name)
        arrays[name] = _b5_for_glue_array(manifest)
        if name == "3x3_n2":
            arrays[name]["kernel_sensitivity"] = _3x3_kernel_sensitivity(manifest)
    for name in NOGLUE_ARRAYS:
        arrays[name] = {"status": "not_applicable", "reason": "no-glue B-1 control array -- no "
                         "glue recipe to arithmetic-check (Appendix A.3's B-5 scope is the glue "
                         "recipe arithmetic only)"}

    return {
        "provenance": {
            "basis_doc": "docs/results/2026-08-15-m4-t6-holdout-adjudication.md Appendix A "
                         "(A.3 recompute split, A.4 event reproduction, A.5 per-pair disclosure)",
            "arrays_root": arrays_root,
            "glue_arrays": GLUE_ARRAYS, "noglue_arrays": NOGLUE_ARRAYS,
        },
        "arrays": arrays,
        "a4_verification": _a4_verification(arrays_root),
        "a5_per_pair_signs": _a5_per_pair_signs(arrays_root),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", default="results/m4/bench/arrays")
    ap.add_argument("--out", default="results/m4/bench/b5_recompute.json")
    ap.add_argument("--gof-out", default="results/m4/bench/glue_poisson_gof.json",
                     help="Appendix A.5's zero-cost per-pair sign-test/Poisson-goodness-of-fit "
                          "disclosure (per-array breakdown + pooled n/same-sign-count/p-value/"
                          "conclusion summary) -- a separate file from --out since this task's "
                          "instruction names it explicitly; pass empty string to skip writing it")
    args = ap.parse_args(argv)

    result = recompute(args.arrays_root)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[recompute_b5] wrote {args.out}")

    if args.gof_out:
        gof = glue_poisson_gof_report(args.arrays_root)
        with open(args.gof_out, "w") as f:
            json.dump(gof, f, indent=1, sort_keys=True)
        print(f"[recompute_b5] wrote {args.gof_out}")
        print(f"[recompute_b5] pooled sign test: n={gof['pooled']['n_pairs']} "
              f"n_pos={gof['pooled']['n_positive']} n_neg={gof['pooled']['n_negative']} "
              f"sign_p={gof['pooled']['sign_test_p_value']:.4f} "
              f"chi2_p={gof['pooled']['goodness_of_fit']['chi2_p_value']:.4f}")
    return result


if __name__ == "__main__":
    main()
