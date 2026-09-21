import json
import os
import types

import numpy as np
import pytest

from scripts.run_anchor_comparison import (ANCHOR_TABLE_COLUMNS, _io_term_lam,
                                           build_parser, compare, net_l1_report,
                                           render_markdown, surrogate_io)


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


def test_io_term_lam_matches_production_diagnostics():
    """Fix round 3: net_l1_report's per-net rescoring is only trustworthy if
    _io_term_lam's replicated FWD-1/FWD-2 loop (io_term.py's _IoFn.forward)
    reproduces exactly what IoTerm.diagnostics() computes internally but
    never returns per-net -- this pins that equivalence as a test rather
    than the one-off manual bit-exactness check done by hand on the real
    mempool_tile_wrap artefacts (fix round 3 review)."""
    torch = pytest.importorskip("torch")
    from ioplace.ops.io_term import IoTerm, build_net_node_csr
    from ioplace.ops.soft_assign import rect_table
    from ioplace.regions import make_grid_regions
    from tests.test_io_term import DEV, DIE, _nl, _pos

    rng = np.random.default_rng(0)
    n_nodes, n_nets = 60, 30
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, n_nodes),
                                               rng.uniform(2, 98, n_nodes))]
    nets = [sorted(rng.choice(n_nodes, int(rng.integers(2, 8)), replace=False).tolist())
            for _ in range(n_nets)]
    nl = _nl(xy, nets)
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    tau = 11.0
    for anchor in ("lower_left", "center"):
        term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=16,
                      num_movable=nl.num_movable, num_physical=nl.num_physical,
                      num_nodes=nl.num_physical, device=DEV, w_mode="unit",
                      node_anchor=anchor,
                      node_size_x=(nl.node_size_x if anchor == "center" else None),
                      node_size_y=(nl.node_size_y if anchor == "center" else None))
        pos = _pos(nl, device=DEV)
        lam = _io_term_lam(term, pos, tau)
        diag = term.diagnostics(pos, tau)
        assert float(lam.sum()) == pytest.approx(diag["soft_lambda_sum"], rel=1e-12)
        l_io = float((term.w * (lam - 1.0).clamp(min=0)).sum())
        assert l_io == pytest.approx(diag["l_io"], rel=1e-12)


def _tiny_evaluation_npz(tmp_path):
    """A small, CPU-only evaluation.npz -- net_l1_report's own contract
    (per-net pairing + net-order identity), not IoTerm's, so this needs no
    GPU and no real placement."""
    from ioplace.evaluator_ref import evaluate
    from ioplace.export.evaluation import save_evaluation
    from ioplace.region_grid import RegionGrid
    from ioplace.regions import make_grid_regions
    from tests.test_netlist import make_tiny_netlist

    nl = make_tiny_netlist()
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10))
    result = evaluate(nl, nl.node_x, nl.node_y, rg)
    save_evaluation(tmp_path / "evaluation.npz", nl, rg, result, nl.node_x, nl.node_y,
                    ["n0", "n1"])
    return nl, result


def test_net_l1_report_scores_per_net_against_evaluation_npz(tmp_path):
    nl, result = _tiny_evaluation_npz(tmp_path)
    assert list(result.per_net_lambda) == [1, 3]        # the pin-anchored hard truth
    assert list(nl.net_degrees) == [2, 3]                # both nets land in [2, 100)
    placedb = types.SimpleNamespace(net_names=["n0", "n1"])
    per_net = {
        "lower_left": (np.array([0, 1]), np.array([1.0, 2.0])),   # abs_err [0, 1.0]
        "center":     (np.array([0, 1]), np.array([1.0, 2.5])),   # abs_err [0, 0.5]
        "pin":        (np.array([0, 1]), np.array([2.0, 2.0])),   # abs_err [1.0, 1.0]
    }
    report = net_l1_report(nl, placedb, str(tmp_path), per_net)
    assert report["band_nets"] == 2
    assert report["l1"] == pytest.approx({"lower_left": 1.0, "center": 0.5, "pin": 2.0})
    boot = report["bootstrap_center_vs_lower_left"]
    assert boot is not None and boot["mean_d"] < 0      # center's error is smaller here
    assert boot["n_boot"] == 10000


def test_net_l1_report_band_excludes_degrees_at_or_above_deg_hi(tmp_path):
    nl, _ = _tiny_evaluation_npz(tmp_path)
    placedb = types.SimpleNamespace(net_names=["n0", "n1"])
    per_net = {"center": (np.array([0, 1]), np.array([1.0, 2.5]))}
    report = net_l1_report(nl, placedb, str(tmp_path), per_net, deg_hi=3)
    assert report["band_nets"] == 1                     # net 1 (degree 3) excluded


def test_net_l1_report_returns_none_without_an_evaluation_npz(tmp_path):
    from tests.test_netlist import make_tiny_netlist
    nl = make_tiny_netlist()
    placedb = types.SimpleNamespace(net_names=["n0", "n1"])
    assert net_l1_report(nl, placedb, str(tmp_path), {}) is None


def test_net_l1_report_aborts_on_net_order_mismatch(tmp_path):
    """The controller's fix-round-2 finding: the per-net pairing is
    meaningless without this check, so it must abort, not assume."""
    nl, _ = _tiny_evaluation_npz(tmp_path)
    placedb = types.SimpleNamespace(net_names=["wrong", "order"])
    with pytest.raises(ValueError, match="net-name orders"):
        net_l1_report(nl, placedb, str(tmp_path), {})


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
    # Fix round 3: this is the ONLY end-to-end exercise of net_l1_report --
    # everything else about it is unit-tested on synthetic data above, but
    # only a real run_main_flow --phase fence produces an evaluation.npz to
    # rescore against. GCD's own L1 numbers aren't a claim (508 movable
    # cells, see the docstring above); this just checks the plumbing runs.
    assert record["l1"] is not None
    assert set(record["l1"]["l1"]) == {"lower_left", "center", "pin"}
    assert record["l1"]["bootstrap_center_vs_lower_left"] is not None
    with open(os.path.join(out_dir, "anchor_comparison.json")) as handle:
        on_disk = json.load(handle)
        assert on_disk["rows"] == record["rows"]
        assert on_disk["l1"] == record["l1"]
    assert os.path.exists(os.path.join(out_dir, "anchor_comparison.md"))
