"""P-H acceptance (design sec 9, "Done per subproject" H) at group scale.

Redefined by controller ruling F2 (progress.md) after the first attempt failed
for reasons that were never the acceptance's target. The original criterion --
grandplan's lambda within 2x of legacy's at matched iterations -- compared two
schedules that differ *by design*: legacy's rho is driven by overflow
(`rho_from_overflow`), policy A's wt is an iteration-count step ramp
(`grandplan_weight`). Matching `--norm-wt-max` to `--rho-max` matches the
ceiling, not the trajectory, so a lambda ratio was never evidence about
normalisation.

What the normalisation module actually owns is the *measured* ratio
`||grad WL||_p / ||grad T_t||_p` and the discipline that keeps a coefficient
alive and bounded. So:

* **Gate (a) -- measured normalisation parity.** Grandplan's `ratio_ema`
  within 2x of legacy's at >= 20 matched active iterations. The two
  denominators are not identical (legacy measures the *merged*
  `||grad IO + kappa_FT grad FT||`, policy A the *isolated* `||grad IO||`),
  which is exactly why the bound is 2x rather than something tight.
* **Gate (b) -- no premature collapse.** The last iteration with `lambda_io >
  0` under grandplan must reach 0.9x legacy's. This is the regression the
  first attempt exposed: the pre-F1 tiny-gradient rule zeroed lambda_io around
  iteration 800-1000, where legacy instead let `ratio_inst` explode and sat at
  the Lipschitz cap.
* **Informational (not gating):** the worst lambda ratio and where it occurs,
  and each arm's realised FT force share at three iterations.

Opt-in: produce the runs with the commands in
`.superpowers/sdd/2026-09-19-v2-p-h-normalisation/task-9-brief.md` Steps 3-6
(or the plan's Task 9), then point `IOPLACE_NORM_VALIDATION_DIR` at the
directory holding them. The test skips deterministically when they are absent.
"""
import json
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
    return legacy, grandplan, rows


def _legacy_ft_share(event):
    """lambda_FT * ||grad FT|| / (||grad WL|| + sum_t lambda_t ||grad T_t||),
    the same quantity the trace's `terms.ft.share` reports for the new arms."""
    lam_ft = float(event.get("lambda_ft", 0.) or 0.)
    lam_io = float(event.get("lambda_io", 0.) or 0.)
    g_wl = float(event.get("grad_l1_wl", 0.) or 0.)
    g_io = float(event.get("grad_l1_io", 0.) or 0.)
    g_ft = float(event.get("grad_l1_ft", 0.) or 0.)
    denominator = g_wl + lam_io * g_io + lam_ft * g_ft
    return (lam_ft * g_ft / denominator) if denominator > 0. else 0.


@pytest.mark.slow
def test_grandplan_matches_the_retired_paths_measured_ratio_and_stays_alive():
    legacy, grandplan, rows = _load()

    legacy_ratio, legacy_lambda = {}, {}
    for event in legacy["trajectory"]:
        iteration = int(event["iteration"])
        ratio = event.get("ratio_ema")
        if ratio is not None and float(ratio) > 0.:
            legacy_ratio[iteration] = float(ratio)
        if float(event.get("lambda_io", 0.) or 0.) > 0.:
            legacy_lambda[iteration] = float(event["lambda_io"])

    new_ratio, new_lambda = {}, {}
    for row in rows:
        iteration = int(row["iteration"])
        term = row["terms"]["io"]
        ratio = term.get("ratio_ema")
        if ratio is not None and float(ratio) > 0.:
            new_ratio[iteration] = float(ratio)
        if float(term["lam"]) > 0.:
            new_lambda[iteration] = float(term["lam"])

    # -- informational -----------------------------------------------------
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

    # -- gate (a): measured normalisation parity ---------------------------
    matched = sorted(set(legacy_ratio) & set(new_ratio))
    assert len(matched) >= 20, (
        "only %d matched iterations with a ratio_ema on both arms" % (len(matched),))
    worst_iteration = max(matched, key=lambda i: max(new_ratio[i] / legacy_ratio[i],
                                                     legacy_ratio[i] / new_ratio[i]))
    worst = max(new_ratio[worst_iteration] / legacy_ratio[worst_iteration],
                legacy_ratio[worst_iteration] / new_ratio[worst_iteration])
    assert worst <= 2.0, (
        "ratio_ema ratio %.3f at iteration %d (legacy %.6g, grandplan %.6g) "
        "over %d matched iterations" % (worst, worst_iteration,
                                        legacy_ratio[worst_iteration],
                                        new_ratio[worst_iteration], len(matched)))

    # -- gate (b): no premature collapse of the IO coefficient -------------
    assert legacy_lambda, "legacy never activated lambda_io"
    assert new_lambda, "grandplan never activated lambda_io"
    legacy_last, new_last = max(legacy_lambda), max(new_lambda)
    assert new_last >= 0.9 * legacy_last, (
        "grandplan's lambda_io dies at iteration %d, legacy's survives to %d "
        "(need >= %.1f)" % (new_last, legacy_last, 0.9 * legacy_last))
