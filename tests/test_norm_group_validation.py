"""P-H acceptance (design sec 9, "Done per subproject" H) at group scale.

Gates set by controller ruling F2' (progress.md, fix wave round 2), replacing
both the original "λ within 2× at matched iterations" criterion and the first
redefinition. Two things changed the question:

1. λ parity was never the right target. Legacy's ρ is overflow-driven
   (`rho_from_overflow`), policy A's `wt` is an iteration-count step ramp
   (`grandplan_weight`); matching `--norm-wt-max` to `--rho-max` matches the
   ceiling, not the trajectory. What the normalisation module owns is the
   *measured* ratio ‖∇WL‖_p / ‖∇T_t‖_p.
2. The r2 rerun showed an arm can satisfy every coefficient-shaped criterion
   while stalling global placement, and that comparing *absolute* iteration
   numbers across arms of different length is meaningless. Hence gate (c) and
   the fraction-of-own-run form of gate (b).

Gates:

* **(a) measured normalisation parity** — grandplan's `terms.io.ratio_ema`
  within 2× of legacy's `trajectory[*].ratio_ema` at
  `max(15, 0.8 × the shorter arm's active callbacks)` matched iterations. The
  denominators are not identical (legacy measures the *merged*
  ‖∇IO + κ_FT·∇FT‖, policy A the *isolated* ‖∇IO‖), which is why the bound is
  2× and not something tight.
* **(b) no premature collapse** — each arm's last `λ_io > 0` iteration as a
  fraction of its *own* `gp_iterations_run`; grandplan's fraction ≥ 0.9 ×
  legacy's.
* **(c) placement quality** — grandplan's final overflow ≤ 1.5 × legacy's and
  final HPWL ≤ 1.2 × legacy's. A normalisation policy that stalls GP fails
  P-H however well-behaved its coefficients look.

Informational (printed, never gating): worst λ ratio and where, each arm's FT
force share at three iterations, and `io_count`/`ft_count`/`hpwl` per arm --
including the adaptive arm, which no gate covers.

Opt-in: produce the runs with the commands in
`.superpowers/sdd/2026-09-19-v2-p-h-normalisation/task-9-brief.md` Steps 3-6
(or the plan's Task 9), then point `IOPLACE_NORM_VALIDATION_DIR` at the
directory holding them. The test skips deterministically when they are absent.
"""
import json
import math
import os
import pathlib

import pytest

from ioplace.norm_trace import read_norm_trace


def _load():
    directory = os.environ.get("IOPLACE_NORM_VALIDATION_DIR")
    if not directory:
        pytest.skip("set IOPLACE_NORM_VALIDATION_DIR to the validation run directory")
    root = pathlib.Path(directory)
    if not (root / "legacy.json").exists() or not (root / "grandplan.json").exists():
        pytest.skip("no legacy.json/grandplan.json in %s" % (root,))
    legacy = json.loads((root / "legacy.json").read_text())
    grandplan = json.loads((root / "grandplan.json").read_text())
    assert legacy["norm_policy"] == "legacy"
    assert grandplan["norm_policy"] == "grandplan"
    assert grandplan["norm_wt_max"] == legacy["rho_max"], \
        "compare matched weight ceilings, not different ramps"
    rows = read_norm_trace(grandplan["norm_trace"])
    assert rows, "grandplan wrote no normalisation rows"
    adaptive = None
    if (root / "adaptive.json").exists():
        adaptive = json.loads((root / "adaptive.json").read_text())
    return legacy, grandplan, rows, adaptive


def _legacy_series(result):
    """`(ratio_ema, lambda_io)` per iteration from a legacy trajectory."""
    ratio, lam = {}, {}
    for event in result["trajectory"]:
        iteration = int(event["iteration"])
        value = event.get("ratio_ema")
        if value is not None and float(value) > 0.:
            ratio[iteration] = float(value)
        if float(event.get("lambda_io", 0.) or 0.) > 0.:
            lam[iteration] = float(event["lambda_io"])
    return ratio, lam


def _trace_series(rows):
    """The same two series from a non-legacy `norm_trace.jsonl`."""
    ratio, lam = {}, {}
    for row in rows:
        iteration = int(row["iteration"])
        term = row["terms"]["io"]
        value = term.get("ratio_ema")
        if value is not None and float(value) > 0.:
            ratio[iteration] = float(value)
        if float(term["lam"]) > 0.:
            lam[iteration] = float(term["lam"])
    return ratio, lam


def _legacy_ft_share(event):
    """λ_FT·‖∇FT‖ / (‖∇WL‖ + Σ_t λ_t‖∇T_t‖), the quantity the trace's
    `terms.ft.share` reports for the new arms."""
    lam_ft = float(event.get("lambda_ft", 0.) or 0.)
    lam_io = float(event.get("lambda_io", 0.) or 0.)
    g_wl = float(event.get("grad_l1_wl", 0.) or 0.)
    g_io = float(event.get("grad_l1_io", 0.) or 0.)
    g_ft = float(event.get("grad_l1_ft", 0.) or 0.)
    denominator = g_wl + lam_io * g_io + lam_ft * g_ft
    return (lam_ft * g_ft / denominator) if denominator > 0. else 0.


def _active_fraction(lam, result):
    """Last iteration with λ_io > 0, as a fraction of this arm's own GP
    length -- the only cross-arm-comparable form when the two arms run for
    different numbers of iterations."""
    length = float(result["gp_iterations_run"])
    return (max(lam) / length) if lam and length > 0 else 0.


@pytest.mark.slow
def test_grandplan_matches_the_retired_path_and_does_not_degrade_placement():
    legacy, grandplan, rows, adaptive = _load()
    legacy_ratio, legacy_lambda = _legacy_series(legacy)
    new_ratio, new_lambda = _trace_series(rows)

    # -- informational -----------------------------------------------------
    for name, result in (("legacy", legacy), ("grandplan", grandplan),
                         ("adaptive", adaptive)):
        if result is None:
            continue
        print("[info] %-9s io_count=%d ft_count=%d hpwl=%.6g final_overflow=%.4f "
              "gp_iterations_run=%d" % (name, result["io_count"], result["ft_count"],
                                        result["hpwl"], result["final_overflow"],
                                        result["gp_iterations_run"]))
    matched_lambda = sorted(set(legacy_lambda) & set(new_lambda))
    if matched_lambda:
        worst_it = max(matched_lambda,
                       key=lambda i: max(new_lambda[i] / legacy_lambda[i],
                                         legacy_lambda[i] / new_lambda[i]))
        print("[info] worst lambda ratio %.3fx at iteration %d "
              "(legacy %.6g, grandplan %.6g) over %d matched active iterations"
              % (max(new_lambda[worst_it] / legacy_lambda[worst_it],
                     legacy_lambda[worst_it] / new_lambda[worst_it]),
                 worst_it, legacy_lambda[worst_it], new_lambda[worst_it],
                 len(matched_lambda)))
    legacy_by_iter = dict((int(e["iteration"]), e) for e in legacy["trajectory"])
    rows_by_iter = dict((int(r["iteration"]), r) for r in rows)
    probes = sorted(set(legacy_by_iter) & set(rows_by_iter))
    for iteration in (probes[:1] + probes[len(probes) // 2:len(probes) // 2 + 1]
                      + probes[-1:]):
        print("[info] iteration %d: FT force share legacy %.5f, grandplan %.5f"
              % (iteration, _legacy_ft_share(legacy_by_iter[iteration]),
                 float(rows_by_iter[iteration]["terms"].get("ft", {})
                       .get("share", 0.) or 0.)))
    if adaptive is not None and adaptive.get("norm_trace"):
        a_ratio, a_lambda = _trace_series(read_norm_trace(adaptive["norm_trace"]))
        a_matched = sorted(set(legacy_ratio) & set(a_ratio))
        if a_matched:
            worst = max(max(a_ratio[i] / legacy_ratio[i], legacy_ratio[i] / a_ratio[i])
                        for i in a_matched)
            print("[info] adaptive: worst ratio_ema ratio %.3fx over %d matched, "
                  "active fraction %.3f (legacy %.3f), overflow %.3fx, hpwl %.3fx"
                  % (worst, len(a_matched), _active_fraction(a_lambda, adaptive),
                     _active_fraction(legacy_lambda, legacy),
                     adaptive["final_overflow"] / legacy["final_overflow"],
                     adaptive["hpwl"] / legacy["hpwl"]))

    # -- gate (a): measured normalisation parity ---------------------------
    matched = sorted(set(legacy_ratio) & set(new_ratio))
    required = max(15, int(math.ceil(0.8 * min(len(legacy_ratio), len(new_ratio)))))
    assert len(matched) >= required, (
        "only %d matched iterations with a ratio_ema on both arms, need %d "
        "(legacy %d active callbacks, grandplan %d)"
        % (len(matched), required, len(legacy_ratio), len(new_ratio)))
    worst_iteration = max(matched, key=lambda i: max(new_ratio[i] / legacy_ratio[i],
                                                     legacy_ratio[i] / new_ratio[i]))
    worst = max(new_ratio[worst_iteration] / legacy_ratio[worst_iteration],
                legacy_ratio[worst_iteration] / new_ratio[worst_iteration])
    assert worst <= 2.0, (
        "ratio_ema ratio %.3f at iteration %d (legacy %.6g, grandplan %.6g) "
        "over %d matched iterations" % (worst, worst_iteration,
                                        legacy_ratio[worst_iteration],
                                        new_ratio[worst_iteration], len(matched)))

    # -- gate (b): no premature collapse, measured per arm's own length ----
    assert legacy_lambda, "legacy never activated lambda_io"
    assert new_lambda, "grandplan never activated lambda_io"
    legacy_fraction = _active_fraction(legacy_lambda, legacy)
    new_fraction = _active_fraction(new_lambda, grandplan)
    assert new_fraction >= 0.9 * legacy_fraction, (
        "grandplan's last active lambda_io is at iteration %d of %d (%.3f of "
        "its own run); legacy's is %d of %d (%.3f) -- need >= %.3f"
        % (max(new_lambda), grandplan["gp_iterations_run"], new_fraction,
           max(legacy_lambda), legacy["gp_iterations_run"], legacy_fraction,
           0.9 * legacy_fraction))

    # -- gate (c): placement quality ---------------------------------------
    overflow_ratio = grandplan["final_overflow"] / legacy["final_overflow"]
    hpwl_ratio = grandplan["hpwl"] / legacy["hpwl"]
    assert overflow_ratio <= 1.5, (
        "grandplan final overflow %.4f is %.3fx legacy's %.4f (limit 1.5x) -- "
        "a normalisation policy that stalls GP fails P-H"
        % (grandplan["final_overflow"], overflow_ratio, legacy["final_overflow"]))
    assert hpwl_ratio <= 1.2, (
        "grandplan final hpwl %.6g is %.3fx legacy's %.6g (limit 1.2x)"
        % (grandplan["hpwl"], hpwl_ratio, legacy["hpwl"]))
