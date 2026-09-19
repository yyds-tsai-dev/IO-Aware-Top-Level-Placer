"""P-H acceptance (design sec 9, "Done per subproject" H): a group-scale run
whose lambda is within 2x of the retired path's lambda at matched iterations.

Opt-in: produce the two runs with the commands in
docs/superpowers/plans/2026-09-19-v2-p-h-normalisation.md Task 9, then point
IOPLACE_NORM_VALIDATION_DIR at the directory holding them.
"""
import json
import os
import pathlib

import pytest

from ioplace.norm_trace import read_norm_trace


@pytest.mark.slow
def test_grandplan_lambda_is_within_2x_of_the_retired_path():
    directory = os.environ.get("IOPLACE_NORM_VALIDATION_DIR")
    if not directory:
        pytest.skip("set IOPLACE_NORM_VALIDATION_DIR to the validation run directory")
    root = pathlib.Path(directory)
    legacy = json.loads((root / "legacy.json").read_text())
    assert legacy["norm_policy"] == "legacy"
    legacy_lambda = dict((int(event["iteration"]), float(event["lambda_io"]))
                         for event in legacy["trajectory"]
                         if float(event.get("lambda_io", 0.)) > 0.)

    grandplan = json.loads((root / "grandplan.json").read_text())
    assert grandplan["norm_policy"] == "grandplan"
    assert grandplan["norm_wt_max"] == legacy["rho_max"], \
        "compare matched weight ceilings, not different ramps"
    rows = read_norm_trace(grandplan["norm_trace"])
    new_lambda = dict((int(row["iteration"]), float(row["terms"]["io"]["lam"]))
                      for row in rows if float(row["terms"]["io"]["lam"]) > 0.)

    matched = sorted(set(legacy_lambda) & set(new_lambda))
    assert len(matched) >= 20, "only %d matched active iterations" % (len(matched),)
    tail = matched[len(matched) // 2:]          # after wt has finished ramping
    worst_iteration = max(tail, key=lambda i: max(new_lambda[i] / legacy_lambda[i],
                                                  legacy_lambda[i] / new_lambda[i]))
    worst = max(new_lambda[worst_iteration] / legacy_lambda[worst_iteration],
                legacy_lambda[worst_iteration] / new_lambda[worst_iteration])
    assert worst <= 2.0, (
        "lambda ratio %.3f at iteration %d (legacy %.6g, grandplan %.6g) over %d "
        "matched iterations" % (worst, worst_iteration,
                                legacy_lambda[worst_iteration],
                                new_lambda[worst_iteration], len(tail)))
