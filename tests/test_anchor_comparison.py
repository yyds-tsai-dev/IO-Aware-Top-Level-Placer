import json
import os

import numpy as np
import pytest

from scripts.run_anchor_comparison import (ANCHOR_TABLE_COLUMNS, build_parser,
                                           compare, render_markdown)


def _rows():
    return [{"anchor": "lower_left", "l_io_soft": 900.0, "lambda_sum_soft": 5100.0,
             "n_active": 4200, "tau": 0.05},
            {"anchor": "center", "l_io_soft": 1010.0, "lambda_sum_soft": 5210.0,
             "n_active": 4200, "tau": 0.05},
            {"anchor": "pin", "l_io_soft": 1180.0, "lambda_sum_soft": 5380.0,
             "n_active": 4200, "tau": 0.05}]


def test_parser_requires_a_config_and_an_out_dir():
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.k == 16 and args.rtype == "grid" and args.degraded is False
    assert args.anchors == ["lower_left", "center", "pin"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--out-dir", "o"])


def test_compare_scores_every_anchor_against_the_post_lg_lower_bound():
    truth = {"hard_lambda_sum": 1000, "io_count": 1640}
    rows = compare(_rows(), truth)
    assert [r["anchor"] for r in rows] == ["lower_left", "center", "pin"]
    assert [r["abs_err"] for r in rows] == [-100.0, 10.0, 180.0]
    assert rows[1]["rel_err"] == pytest.approx(0.01)
    assert all(r["io_lb_final"] == 1000 for r in rows)
    assert all(r["io_count_final"] == 1640 for r in rows)
    assert [r["closest"] for r in rows] == [False, True, False]


def test_compare_breaks_a_tie_toward_the_earlier_anchor_deterministically():
    truth = {"hard_lambda_sum": 1000, "io_count": 1640}
    rows = _rows()
    rows[0]["l_io_soft"] = 1010.0        # same |error| as center
    out = compare(rows, truth)
    assert [r["closest"] for r in out] == [True, False, False]


def test_compare_refuses_a_zero_lower_bound_rather_than_dividing_by_it():
    with pytest.raises(ValueError, match="hard_lambda_sum"):
        compare(_rows(), {"hard_lambda_sum": 0, "io_count": 12})


def test_markdown_uses_the_documented_column_order():
    assert ANCHOR_TABLE_COLUMNS == ("anchor", "l_io_soft", "lambda_sum_soft",
                                    "io_lb_final", "abs_err", "rel_err",
                                    "io_count_final", "closest")
    text = render_markdown(compare(_rows(), {"hard_lambda_sum": 1000,
                                             "io_count": 1640}), degraded=False)
    header = text.splitlines()[0]
    assert header == "| " + " | ".join(ANCHOR_TABLE_COLUMNS) + " |"
    assert "lower_left" in text and "center" in text and "pin" in text


def test_markdown_shouts_when_the_run_was_degraded():
    text = render_markdown(compare(_rows(), {"hard_lambda_sum": 1000,
                                             "io_count": 1640}), degraded=True,
                           reason="soft snapshot and fence run use different memberships")
    assert "**DEGRADED**" in text
    assert "different memberships" in text


@pytest.mark.slow
@pytest.mark.gpu
def test_anchor_comparison_runs_end_to_end_on_gcd(tmp_path):
    """Smoke test of the real path on the smallest case available. The physical
    claim is made on mempool_tile_wrap by the campaign in
    docs/results/2026-09-19-p-f-anchor-comparison.md, not here -- GCD is 508
    movable cells and its errors are not meaningful."""
    pytest.importorskip("torch")
    pytest.importorskip("ioplace.drivers.run_main_flow",
                        reason="P-B has not landed; the end-to-end path needs "
                               "run_main_flow --phase soft/fence")
    from scripts.run_anchor_comparison import main
    out_dir = str(tmp_path / "gcd")
    record = main(["--config", os.path.abspath("results/route_feedback_20260914/gcd.json"),
                   "--out-dir", out_dir, "--k", "4", "--rtype", "grid",
                   "--dp-seed", "1000", "--run-flow"])
    assert [r["anchor"] for r in record["rows"]] == ["lower_left", "center", "pin"]
    assert all(np.isfinite(r["l_io_soft"]) for r in record["rows"])
    assert record["degraded"] is False
    assert sum(r["closest"] for r in record["rows"]) == 1
    with open(os.path.join(out_dir, "anchor_comparison.json")) as handle:
        assert json.load(handle)["rows"] == record["rows"]
    assert os.path.exists(os.path.join(out_dir, "anchor_comparison.md"))
