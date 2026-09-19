import json
import os

import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers import run_placement
from ioplace.drivers.run_placement_io import RESULT_FIELDS, run_io
from ioplace.norm_trace import read_norm_trace

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")


def test_norm_fields_are_in_the_result_contract():
    for field in ("norm_policy", "norm_p", "norm_ramp_period", "norm_wt_max",
                  "norm_probe_every", "norm_target_share", "norm_trace",
                  "lambda_ft_final"):
        assert field in RESULT_FIELDS


def test_norm_flag_defaults_match_the_design():
    args = run_placement.build_parser().parse_args(
        ["--config", "c.json", "--mode", "io", "--out", "o.json"])
    assert args.norm_policy == "legacy"
    assert args.norm_p == 1
    assert args.norm_ramp_period == 100
    assert args.norm_wt_max == 1.0
    assert args.norm_probe_every == 50
    assert args.norm_target_share is None
    assert args.norm_trace is None


def test_norm_flags_appear_in_the_driver_help():
    text = run_placement.build_parser().format_help()
    for flag in ("--norm-policy", "--norm-p", "--norm-ramp-period",
                 "--norm-wt-max", "--norm-probe-every", "--norm-target-share",
                 "--norm-trace"):
        assert flag in text


def test_norm_flags_are_forwarded_to_run_io(monkeypatch):
    captured = {}

    def fake_run_io(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("ioplace.drivers.run_placement_io.run_io", fake_run_io)
    monkeypatch.setattr("sys.argv", [
        "run_placement", "--config", "c.json", "--mode", "io", "--out", "o.json",
        "--callback-order", "atomic", "--norm-policy", "adaptive", "--norm-p", "2",
        "--norm-ramp-period", "250", "--norm-wt-max", "0.4",
        "--norm-probe-every", "100", "--norm-target-share", "io=0.3,ft=0.1",
        "--norm-trace", "t.jsonl"])
    run_placement.main()
    assert captured["norm_policy"] == "adaptive"
    assert captured["norm_p"] == 2
    assert captured["norm_ramp_period"] == 250
    assert captured["norm_wt_max"] == 0.4
    assert captured["norm_probe_every"] == 100
    assert captured["norm_target_share"] == "io=0.3,ft=0.1"
    assert captured["norm_trace"] == "t.jsonl"


@pytest.mark.parametrize("kwargs,message", [
    (dict(norm_policy="bogus"), "norm_policy"),
    (dict(norm_p=3), "norm_p"),
    (dict(norm_probe_every=0), "norm_probe_every"),
    (dict(norm_probe_every=75), "norm_probe_every"),        # not a multiple of every=50
    (dict(norm_policy="grandplan"), "atomic"),              # callback_order defaults legacy
])
def test_run_io_rejects_invalid_norm_configuration(kwargs, message, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        run_io("missing.json", 4, "grid", 0, str(tmp_path / "o.json"), **kwargs)
    assert message in str(excinfo.value)


def _small_config(tmp_path):
    cfg = json.load(open(SIMPLE))
    cfg.update(num_threads=4, plot_flag=0, num_bins_x=16, num_bins_y=16,
               global_place_stages=[dict(num_bins_x=16, num_bins_y=16, iteration=40,
                                         learning_rate=.01,
                                         wirelength="weighted_average",
                                         optimizer="nesterov")])
    path = tmp_path / "simple_norm.json"
    path.write_text(json.dumps(cfg))
    return str(path)


@pytest.mark.slow
def test_grandplan_policy_runs_and_writes_a_norm_trace(tmp_path):
    out = str(tmp_path / "grandplan.json")
    result = run_io(_small_config(tmp_path), 4, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, check_invariant=True,
                    norm_policy="grandplan", norm_probe_every=5,
                    norm_ramp_period=10, norm_wt_max=1.0)
    assert result["norm_policy"] == "grandplan"
    assert result["norm_trace"] == os.path.abspath(out + ".norm_trace.jsonl")
    rows = read_norm_trace(result["norm_trace"])
    assert rows, "no normalisation rows were emitted"
    assert [r["obj_version"] for r in rows] == sorted(set(r["obj_version"] for r in rows))
    assert all(r["refreshed_version"] == r["obj_version"] for r in rows)
    assert all(set(r["terms"]) == {"io", "ft"} for r in rows)
    assert any(r["terms"]["io"]["lam"] > 0.0 for r in rows)
    assert result["lambda_io_final"] > 0.0


@pytest.mark.slow
def test_adaptive_policy_runs_and_respects_the_requested_share(tmp_path):
    out = str(tmp_path / "adaptive.json")
    result = run_io(_small_config(tmp_path), 4, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, norm_policy="adaptive", norm_probe_every=5,
                    norm_target_share="io=0.3,ft=0.1")
    rows = read_norm_trace(result["norm_trace"])
    assert rows
    assert all(r["policy"] == "adaptive" for r in rows)
    assert all(r["terms"]["io"]["target_share"] == 0.3 for r in rows)
    assert all(r["terms"]["ft"]["target_share"] == 0.1 for r in rows)


@pytest.mark.slow
def test_legacy_policy_is_the_default_and_emits_no_trace(tmp_path):
    out = str(tmp_path / "legacy.json")
    result = run_io(_small_config(tmp_path), 4, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, check_invariant=True)
    assert result["norm_policy"] == "legacy"
    assert result["norm_trace"] is None
    assert not os.path.exists(out + ".norm_trace.jsonl")
    events = result["trajectory"]
    assert events and any(event["grad_l1_ft"] > 0 for event in events)
    assert all(event["obj_version"] == event["refreshed_version"] for event in events)
