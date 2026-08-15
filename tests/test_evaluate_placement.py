"""M4 T8a overflow-diagnosis follow-up: `ioplace.drivers.evaluate_placement`
-- an evaluate-only driver that re-scores an existing run's saved `.npz`
placement (node_x/node_y) under a K/rtype, without re-running GP/LG.

Two things this file locks down:
  1. For the *same* K/rtype/seed the source run used, evaluate_placement's
     evaluator fields (io_count, ft_count, hard_lambda_sum, tree_wl, hpwl,
     io_rg, ft_rg) are bit-identical to the original run_io JSON's own
     fields -- it reuses the exact same two evaluator calls
     (`run_placement._evaluate_and_pack` + `GpuEvalContext.evaluate`) on the
     same saved node_x/node_y, not a reimplementation.
  2. A npz/config mismatch (wrong node count) fails loudly with a clear
     ValueError rather than silently evaluating garbage -- exercised
     without a real placement run (fails before either evaluator call).
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers.evaluate_placement import evaluate_placement

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")
ADAPTEC1 = os.path.join(DP, "install", "test", "ispd2005", "adaptec1.json")


def test_evaluate_placement_rejects_mismatched_npz(tmp_path):
    bad_npz = str(tmp_path / "bad.npz")
    np.savez_compressed(bad_npz, node_x=np.zeros(3), node_y=np.zeros(3))
    with pytest.raises(ValueError, match="mismatched"):
        evaluate_placement(bad_npz, SIMPLE, 4, "grid", str(tmp_path / "out.json"))


@pytest.mark.slow
def test_evaluate_placement_matches_source_run_bit_for_bit(tmp_path):
    from ioplace.drivers.run_placement_io import run_io

    run_out = str(tmp_path / "run_io.json")
    # rho_max=0.0 (observer_mode, same bit-identical-to-flat path
    # tests/test_driver_io.py's test_rho_zero_is_observer_mode... locks
    # down): only the GP+LG placement itself matters here, not the io-term
    # objective, so the lighter/faster path is used.
    run_res = run_io(ADAPTEC1, 4, "grid", 0, run_out, rho_max=0.0)

    eval_out = str(tmp_path / "eval.json")
    eval_res = evaluate_placement(run_out + ".npz", ADAPTEC1, 4, "grid", eval_out)

    for f in ("io_count", "ft_count", "hard_lambda_sum", "tree_wl", "hpwl",
             "io_rg", "ft_rg"):
        assert eval_res[f] == run_res[f], f

    assert eval_res["evaluate_only"] is True
    assert eval_res["mode"] == "evaluate_only"
    assert eval_res["source_npz"] == run_out + ".npz"
    assert len(eval_res["source_npz_sha256"]) == 64          # sha256 hex digest
    assert eval_res["io_gp"] is None                          # no mid-GP snapshot available

    on_disk = json.load(open(eval_out))
    for f in ("io_count", "ft_count", "hard_lambda_sum", "tree_wl", "hpwl",
             "io_rg", "ft_rg", "source_npz", "source_npz_sha256", "evaluate_only",
             "run_id", "status", "repo_commit", "input_sha256", "env"):
        assert f in on_disk
