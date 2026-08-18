"""M4 T10 H100 handover package tests (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 6.3
E5 / sec 6.4 / sec 7.1 T10 row): `scripts/m4_forecast.py`,
`scripts/m4_check_prediction.py`, and `scripts/m4_run.sh --dry-run`.

Everything here runs against small hand-written fixtures (round numbers,
easy to hand-verify) rather than the real `results/m4/**` artifacts --
same convention as `test_bench_count_freeze.py`/`test_m4_report_lint.py`.
The forecast fixture's numbers are chosen so every ratio/band/formula in
`m4_forecast.py` produces an exact, hand-checkable result:

  count_freeze: base_n_nodes=20, total_n_nets=30, total_n_pins=40
  probe_gp:     num_physical=10, num_nets=10, num_pins=10
    -> node_ratio=2.0, net_ratio=3.0, pin_ratio=4.0

  cluster_flat.phases:
    read: t_s=100                                  -> node_ratio -> 200
    gp:   t_s=50,  peak_alloc_gb=1.0, peak_reserved_gb=1.5  -> node_ratio -> 100
    lg:   t_s=20                                    -> node_ratio -> 40
    eval: t_s=30,  peak_alloc_gb=2.0, peak_reserved_gb=2.2  -> pin_ratio  -> 120
  host_peak_rss_gb=10.0, device_baseline_gb=0.2

  wall-time: read fast=slow=200; gp fast=100/5=20, slow=100/2.5=40;
             lg fast=slow=40; eval fast=120/5=24, slow=120/2.5=48
             total_fast=284, total_slow=328
  gpu memory: gp scaled_alloc=2.0/scaled_reserved=4.0;
              eval scaled_alloc=8.0/scaled_reserved=8.8
              dominant=eval -> lower=8.0, transient(gp)=2.0,
              allocator_reserve=max(4.0-2.0, 8.8-8.0)=2.0, cuda_context=0.2
              upper=8.0+2.0+2.0+0.2=12.2
  host rss: point_estimate = 10.0*2.0 = 20.0
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # matches other tests/test_m4_*.py's `from scripts...` imports

from scripts import m4_check_prediction as chk
from scripts import m4_forecast as fc

M4_RUN_SH = REPO / "scripts" / "m4_run.sh"


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _count_freeze_fixture():
    return {"base_n_nodes": 20, "total_n_nets": 30, "total_n_pins": 40}


def _probe_gp_fixture():
    return {
        "num_physical": 10, "num_nets": 10, "num_pins": 10,
        "gp_peak_alloc_mb": 1024.0, "gp_peak_reserved_mb": 2048.0,
    }


def _cluster_flat_fixture():
    return {
        "phases": {
            "read": {"t_s": 100.0, "peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0,
                      "host_rss_hwm_at_phase_end": 9.0},
            "gp": {"t_s": 50.0, "peak_alloc_gb": 1.0, "peak_reserved_gb": 1.5,
                   "host_rss_hwm_at_phase_end": 9.5},
            "lg": {"t_s": 20.0, "peak_alloc_gb": 0.5, "peak_reserved_gb": 0.6,
                   "host_rss_hwm_at_phase_end": 9.5},
            "eval": {"t_s": 30.0, "peak_alloc_gb": 2.0, "peak_reserved_gb": 2.2,
                     "host_rss_hwm_at_phase_end": 10.0},
        },
        "host_peak_rss_gb": 10.0,
        "device_baseline_gb": 0.2,
        "device_used_gb": 3.0,
        "t_read": 100.0, "t_gp": 50.0, "t_lg": 20.0, "t_eval": 30.0,
        "env": {"gpu_name": "NVIDIA L4"},
    }


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# m4_forecast.py: pure-function unit tests
# ---------------------------------------------------------------------------

def test_compute_scale_ratios():
    ratios = fc.compute_scale_ratios(_count_freeze_fixture(), _probe_gp_fixture())
    assert ratios["node_ratio"] == pytest.approx(2.0)
    assert ratios["net_ratio"] == pytest.approx(3.0)
    assert ratios["pin_ratio"] == pytest.approx(4.0)
    assert ratios["unvalidated"] is True


def test_build_wall_time_forecast_values_and_band_ordering():
    ratios = fc.compute_scale_ratios(_count_freeze_fixture(), _probe_gp_fixture())
    wtf = fc.build_wall_time_forecast(_cluster_flat_fixture(), ratios)

    assert wtf["phases"]["read"]["t_h100_fast_s"] == pytest.approx(200.0)
    assert wtf["phases"]["read"]["t_h100_slow_s"] == pytest.approx(200.0)
    assert wtf["phases"]["gp"]["t_h100_fast_s"] == pytest.approx(20.0)
    assert wtf["phases"]["gp"]["t_h100_slow_s"] == pytest.approx(40.0)
    assert wtf["phases"]["lg"]["t_h100_fast_s"] == pytest.approx(40.0)
    assert wtf["phases"]["lg"]["t_h100_slow_s"] == pytest.approx(40.0)
    assert wtf["phases"]["eval"]["t_h100_fast_s"] == pytest.approx(24.0)
    assert wtf["phases"]["eval"]["t_h100_slow_s"] == pytest.approx(48.0)

    assert wtf["total_fast_s"] == pytest.approx(284.0)
    assert wtf["total_slow_s"] == pytest.approx(328.0)
    assert wtf["unvalidated"] is True

    # "區間判定兩側": both bounds are internally consistent for every phase
    # and for the total, regardless of fixture values (fast side can never
    # exceed the slow side -- s_p_hi >= s_p_lo by construction).
    for name, p in wtf["phases"].items():
        assert p["t_h100_fast_s"] <= p["t_h100_slow_s"], name
    assert wtf["total_fast_s"] <= wtf["total_slow_s"]


def test_build_gpu_memory_forecast_formula():
    ratios = fc.compute_scale_ratios(_count_freeze_fixture(), _probe_gp_fixture())
    gmf = fc.build_gpu_memory_forecast(_cluster_flat_fixture(), _probe_gp_fixture(), ratios)

    assert gmf["components"]["gp"]["scaled_alloc_gb"] == pytest.approx(2.0)
    assert gmf["components"]["gp"]["scaled_reserved_gb"] == pytest.approx(4.0)
    assert gmf["components"]["eval"]["scaled_alloc_gb"] == pytest.approx(8.0)
    assert gmf["components"]["eval"]["scaled_reserved_gb"] == pytest.approx(8.8)

    assert gmf["dominant_component"] == "eval"
    assert gmf["lower_bound_gb"] == pytest.approx(8.0)
    assert gmf["max_phase_transient_gb"] == pytest.approx(2.0)
    assert gmf["allocator_reserve_gb"] == pytest.approx(2.0)
    assert gmf["cuda_context_gb"] == pytest.approx(0.2)
    assert gmf["upper_bound_gb"] == pytest.approx(12.2)
    assert gmf["lower_bound_gb"] <= gmf["upper_bound_gb"]
    assert gmf["unvalidated"] is True


def test_build_host_rss_forecast_point_estimate_no_interval():
    ratios = fc.compute_scale_ratios(_count_freeze_fixture(), _probe_gp_fixture())
    hrf = fc.build_host_rss_forecast(_cluster_flat_fixture(), ratios)
    assert hrf["point_estimate_gb"] == pytest.approx(20.0)
    assert hrf["interval"] is None
    assert "unfit" in hrf["model_status"]
    assert hrf["unvalidated"] is True


# ---------------------------------------------------------------------------
# m4_forecast.py: CLI / schema / freeze-rejection
# ---------------------------------------------------------------------------

def _write_forecast_fixtures(tmp_path):
    cluster_flat = _write_json(tmp_path / "cluster_flat.json", _cluster_flat_fixture())
    count_freeze = _write_json(tmp_path / "count_freeze_30m.json", _count_freeze_fixture())
    probe_gp = _write_json(tmp_path / "probe_gp_memory.json", _probe_gp_fixture())
    return cluster_flat, count_freeze, probe_gp


def test_forecast_cli_writes_expected_schema(tmp_path):
    cluster_flat, count_freeze, probe_gp = _write_forecast_fixtures(tmp_path)
    out = tmp_path / "h100_prediction.draft.json"

    rc = fc.main([
        "--cluster-flat", str(cluster_flat),
        "--count-freeze", str(count_freeze),
        "--probe-gp-memory", str(probe_gp),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()

    data = json.loads(out.read_text())
    for key in ("schema_version", "task", "status", "generated_at", "sku",
                "l4_reference", "inputs", "scale_ratios", "wall_time_forecast",
                "gpu_memory_forecast", "host_rss_forecast", "t14_check_protocol",
                "refit_on_miss_note", "provenance"):
        assert key in data, key

    assert data["task"] == "T10"
    assert data["status"] == "draft"
    assert data["sku"]["name"] == "H100 SXM5 80GB HBM3"

    for name, path in (("cluster_flat_l4", cluster_flat),
                        ("count_freeze_30m", count_freeze),
                        ("probe_gp_memory", probe_gp)):
        assert data["inputs"][name]["sha256"] == fc.sha256_file(path)

    assert data["wall_time_forecast"]["total_fast_s"] == pytest.approx(284.0)
    assert data["gpu_memory_forecast"]["upper_bound_gb"] == pytest.approx(12.2)


def test_forecast_cli_freeze_status_flag(tmp_path):
    cluster_flat, count_freeze, probe_gp = _write_forecast_fixtures(tmp_path)
    out = tmp_path / "h100_prediction.json"
    rc = fc.main([
        "--cluster-flat", str(cluster_flat), "--count-freeze", str(count_freeze),
        "--probe-gp-memory", str(probe_gp), "--out", str(out), "--freeze",
    ])
    assert rc == 0
    assert json.loads(out.read_text())["status"] == "frozen"


def test_forecast_cli_refuses_overwrite_without_force(tmp_path):
    cluster_flat, count_freeze, probe_gp = _write_forecast_fixtures(tmp_path)
    out = tmp_path / "h100_prediction.json"
    args = [
        "--cluster-flat", str(cluster_flat), "--count-freeze", str(count_freeze),
        "--probe-gp-memory", str(probe_gp), "--out", str(out),
    ]

    assert fc.main(args) == 0
    first_generated_at = json.loads(out.read_text())["generated_at"]

    rc = fc.main(args)  # no --force
    assert rc == 2
    # file must be untouched by the rejected second call
    assert json.loads(out.read_text())["generated_at"] == first_generated_at

    rc = fc.main(args + ["--force"])
    assert rc == 0  # --force allows the overwrite


# ---------------------------------------------------------------------------
# m4_check_prediction.py: in/out-of-band both sides
# ---------------------------------------------------------------------------

def _prediction_fixture():
    return {
        "wall_time_forecast": {"total_fast_s": 200.0, "total_slow_s": 400.0},
        "gpu_memory_forecast": {"lower_bound_gb": 10.0, "upper_bound_gb": 20.0},
        "host_rss_forecast": {"interval": None, "point_estimate_gb": 15.0},
    }


def test_check_wall_time_in_band():
    actual = {"t_total": 300.0}
    result = chk.check_wall_time(actual, _prediction_fixture())
    assert result["verdict"] == "in"
    assert result["in_band"] is True


def test_check_wall_time_out_of_band():
    actual = {"t_total": 500.0}
    result = chk.check_wall_time(actual, _prediction_fixture())
    assert result["verdict"] == "out"
    assert result["in_band"] is False


def test_check_wall_time_falls_back_to_phase_sum_then_named_fields():
    pred = _prediction_fixture()
    assert chk.check_wall_time({"phases": {"a": {"t_s": 100.0}, "b": {"t_s": 200.0}}}, pred)["actual"] == 300.0
    assert chk.check_wall_time({"t_read": 100.0, "t_gp": 100.0, "t_lg": 50.0, "t_eval": 50.0}, pred)["actual"] == 300.0


def test_check_gpu_peak_in_and_out_of_band():
    pred = _prediction_fixture()
    assert chk.check_gpu_peak({"device_used_gb": 15.0}, pred)["verdict"] == "in"
    assert chk.check_gpu_peak({"device_used_gb": 25.0}, pred)["verdict"] == "out"


def test_check_host_rss_not_applicable_when_no_interval():
    result = chk.check_host_rss({"host_peak_rss_gb": 999.0}, _prediction_fixture())
    assert result["verdict"] == "not_applicable"
    assert result["in_band"] is None


def test_check_host_rss_graded_when_interval_present():
    pred = _prediction_fixture()
    pred["host_rss_forecast"]["interval"] = {"low": 10.0, "high": 20.0}
    assert chk.check_host_rss({"host_peak_rss_gb": 15.0}, pred)["verdict"] == "in"
    assert chk.check_host_rss({"host_peak_rss_gb": 5.0}, pred)["verdict"] == "out"


def test_check_prediction_all_in_band_no_disclosure():
    actual = {"t_total": 300.0, "device_used_gb": 15.0, "host_peak_rss_gb": 15.0}
    result = chk.check_prediction(actual, _prediction_fixture())
    assert result["any_out_of_band"] is False
    assert "mandatory_disclosure" not in result


def test_check_prediction_any_out_of_band_carries_mandatory_disclosure():
    actual = {"t_total": 999.0, "device_used_gb": 15.0, "host_peak_rss_gb": 15.0}
    result = chk.check_prediction(actual, _prediction_fixture())
    assert result["any_out_of_band"] is True
    assert result["mandatory_disclosure"] == chk.MANDATORY_REFIT_DISCLOSURE_SENTENCE
    assert result["items"]["wall_time"]["verdict"] == "out"


# ---------------------------------------------------------------------------
# m4_check_prediction.py: per-phase s_p^obs mainline hypothesis (sec 6.3
# T14 row / h100_prediction.json's own t14_check_protocol field)
# ---------------------------------------------------------------------------
#
# Reuses the module's own `_cluster_flat_fixture`/`_count_freeze_fixture`/
# `_probe_gp_fixture` through the real `fc.build_wall_time_forecast` /
# `fc.build_gpu_memory_forecast` code path (same numbers as this file's
# top-of-module docstring: node_ratio=2.0, pin_ratio=4.0):
#
#   phase   t_l4_extrapolated_27m_s   band          fast/slow (s_hi/s_lo)
#   read    200.0                     [1.0, 1.0]    200 / 200
#   gp      100.0                     [2.5, 5.0]    20  / 40
#   lg      40.0                      [1.0, 1.0]    40  / 40
#   eval    120.0                     [2.5, 5.0]    24  / 48
#   total_fast_s=284.0, total_slow_s=328.0
#   gpu_memory_forecast: lower_bound_gb=8.0, upper_bound_gb=12.2

def _full_prediction_fixture():
    ratios = fc.compute_scale_ratios(_count_freeze_fixture(), _probe_gp_fixture())
    cluster_flat = _cluster_flat_fixture()
    return {
        "wall_time_forecast": fc.build_wall_time_forecast(cluster_flat, ratios),
        "gpu_memory_forecast": fc.build_gpu_memory_forecast(cluster_flat, _probe_gp_fixture(), ratios),
        "host_rss_forecast": fc.build_host_rss_forecast(cluster_flat, ratios),
    }


def test_check_phase_s_p_in_band_and_out_of_band():
    pred = _full_prediction_fixture()
    gp_phase = pred["wall_time_forecast"]["phases"]["gp"]  # t_extrap=100, band [2.5, 5.0]

    in_band = chk.check_phase_s_p("gp", gp_phase, actual_t_s=25.0)  # s_p_obs=4.0
    assert in_band["s_p_obs"] == pytest.approx(4.0)
    assert in_band["relative_error"] == pytest.approx(0.0)
    assert in_band["same_order"] is True
    assert in_band["verdict"] == "in"

    out_of_band = chk.check_phase_s_p("gp", gp_phase, actual_t_s=5.0)  # s_p_obs=20.0
    assert out_of_band["s_p_obs"] == pytest.approx(20.0)
    assert out_of_band["relative_error"] == pytest.approx(3.0)  # (20-5)/5
    assert out_of_band["same_order"] is False
    assert out_of_band["verdict"] == "out"


def test_check_phase_s_p_within_tolerance_just_outside_band():
    pred = _full_prediction_fixture()
    eval_phase = pred["wall_time_forecast"]["phases"]["eval"]  # t_extrap=120, band [2.5, 5.0]
    # actual_t_s=50 -> s_p_obs=2.4, just below lo=2.5, relative_error=0.04 <= 0.5
    result = chk.check_phase_s_p("eval", eval_phase, actual_t_s=50.0)
    assert result["s_p_obs"] == pytest.approx(2.4)
    assert result["relative_error"] == pytest.approx(0.04)
    assert result["same_order"] is True


def test_check_all_phases_unavailable_without_phases_block():
    pred = _full_prediction_fixture()
    result = chk.check_all_phases({"t_total": 300.0}, pred)
    assert result["available"] is False
    assert result["phases"] == {}


def _phases_actual(read=200.0, gp=25.0, lg=40.0, eval_=30.0):
    return {"read": {"t_s": read}, "gp": {"t_s": gp}, "lg": {"t_s": lg}, "eval": {"t_s": eval_}}


def test_scenario_all_pass_mainline_confirmed():
    """全過: every phase's s_p_obs in-band, total in [284, 328], memory in
    [8.0, 12.2] -> mainline confirmed True, any_out_of_band False."""
    pred = _full_prediction_fixture()
    actual = {"phases": _phases_actual(), "device_used_gb": 10.0, "host_peak_rss_gb": 15.0}
    # sanity: total = 200+25+40+30 = 295, inside [284, 328]
    assert sum(p["t_s"] for p in actual["phases"].values()) == pytest.approx(295.0)

    result = chk.check_prediction(actual, pred)
    assert result["mainline_hypothesis"]["confirmed"] is True
    assert result["mainline_hypothesis"]["phases_same_order"] is True
    assert result["mainline_hypothesis"]["total_time_in_band"] is True
    assert result["items"]["gpu_peak"]["verdict"] == "in"
    assert result["any_out_of_band"] is False
    assert "mandatory_disclosure" not in result
    assert result["no_new_prediction_interval_note"] == chk.FIRST_RUN_NO_INTERVAL_NOTE


def test_scenario_one_phase_out_of_order_disconfirms_mainline():
    """一 phase 亂序: gp's s_p_obs=20.0 (band [2.5,5.0], >50% relative error)
    while total (295) stays inside [284, 328] and memory stays in-band --
    isolates the per-phase failure from the total-time/GPU-peak items,
    which individually still read "in"."""
    pred = _full_prediction_fixture()
    actual = {"phases": _phases_actual(gp=5.0, eval_=50.0),
              "device_used_gb": 10.0, "host_peak_rss_gb": 15.0}
    assert sum(p["t_s"] for p in actual["phases"].values()) == pytest.approx(295.0)

    result = chk.check_prediction(actual, pred)
    assert result["items"]["wall_time"]["verdict"] == "in"       # total alone still in-band
    assert result["items"]["gpu_peak"]["verdict"] == "in"
    assert result["phase_checks"]["phases"]["gp"]["same_order"] is False
    assert result["mainline_hypothesis"]["confirmed"] is False
    assert result["mainline_hypothesis"]["phases_same_order"] is False
    assert result["mainline_hypothesis"]["total_time_in_band"] is True
    assert "gp" in result["mainline_hypothesis"]["reason"]
    assert result["any_out_of_band"] is True
    assert result["mandatory_disclosure"] == chk.MANDATORY_REFIT_DISCLOSURE_SENTENCE


def test_scenario_total_wall_time_out_of_band_disconfirms_mainline():
    """總時間落外: every individual phase stays same_order (within 50%
    tolerance of its declared band) but the sum (470) falls outside
    [284, 328]."""
    pred = _full_prediction_fixture()
    actual = {"phases": _phases_actual(read=300.0, gp=50.0, lg=60.0, eval_=60.0),
              "device_used_gb": 10.0, "host_peak_rss_gb": 15.0}
    total = sum(p["t_s"] for p in actual["phases"].values())
    assert total == pytest.approx(470.0)
    assert not (284.0 <= total <= 328.0)

    result = chk.check_prediction(actual, pred)
    phase_verdicts = result["phase_checks"]["phases"]
    assert all(p["same_order"] for p in phase_verdicts.values()), phase_verdicts
    assert result["items"]["wall_time"]["verdict"] == "out"
    assert result["mainline_hypothesis"]["phases_same_order"] is True
    assert result["mainline_hypothesis"]["total_time_in_band"] is False
    assert result["mainline_hypothesis"]["confirmed"] is False
    assert "total wall-time" in result["mainline_hypothesis"]["reason"]
    assert result["any_out_of_band"] is True
    assert result["mandatory_disclosure"] == chk.MANDATORY_REFIT_DISCLOSURE_SENTENCE


def test_scenario_gpu_memory_out_of_band_independent_of_mainline():
    """記憶體落外: phases/total all pass (mainline confirmed True) but
    device_used_gb (50.0) is outside [lower_bound_gb=8.0,
    upper_bound_gb=12.2] -- memory is judged as its own sentence (spec sec
    6.3), not folded into the wall-time mainline hypothesis."""
    pred = _full_prediction_fixture()
    actual = {"phases": _phases_actual(), "device_used_gb": 50.0, "host_peak_rss_gb": 15.0}

    result = chk.check_prediction(actual, pred)
    assert result["mainline_hypothesis"]["confirmed"] is True  # wall-time side is clean
    assert result["items"]["gpu_peak"]["verdict"] == "out"
    assert result["any_out_of_band"] is True
    assert result["mandatory_disclosure"] == chk.MANDATORY_REFIT_DISCLOSURE_SENTENCE


def test_scenario_missing_phase_actual_is_indeterminate_not_a_pass():
    pred = _full_prediction_fixture()
    # t_total given explicitly (295, in-band) so the wall_time item's own
    # phases-sum fallback (which would otherwise silently undercount from
    # the same missing "eval" entry) does not conflate with the per-phase
    # mainline check this test isolates.
    actual = {"t_total": 295.0,
              "phases": {"read": {"t_s": 200.0}, "gp": {"t_s": 25.0}, "lg": {"t_s": 40.0}},
              "device_used_gb": 10.0, "host_peak_rss_gb": 15.0}  # eval missing

    result = chk.check_prediction(actual, pred)
    assert result["mainline_hypothesis"]["confirmed"] is None
    assert "eval" in result["mainline_hypothesis"]["reason"]
    # an indeterminate mainline verdict does not itself force any_out_of_band
    # (it is neither a pass nor a disconfirmed hypothesis)
    assert result["any_out_of_band"] is False


def test_check_prediction_cli_exit_codes(tmp_path, capsys):
    pred_path = _write_json(tmp_path / "h100_prediction.json", _prediction_fixture())

    in_band = _write_json(tmp_path / "actual_in.json",
                           {"t_total": 300.0, "device_used_gb": 15.0, "host_peak_rss_gb": 15.0})
    rc = chk.main([str(in_band), "--prediction", str(pred_path)])
    assert rc == 0

    out_of_band = _write_json(tmp_path / "actual_out.json",
                               {"t_total": 999.0, "device_used_gb": 15.0, "host_peak_rss_gb": 15.0})
    rc = chk.main([str(out_of_band), "--prediction", str(pred_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert chk.MANDATORY_REFIT_DISCLOSURE_SENTENCE in captured.err


def test_check_prediction_cli_writes_out_file(tmp_path):
    pred_path = _write_json(tmp_path / "h100_prediction.json", _prediction_fixture())
    actual_path = _write_json(tmp_path / "actual.json",
                               {"t_total": 300.0, "device_used_gb": 15.0, "host_peak_rss_gb": 15.0})
    out_path = tmp_path / "judgement.json"
    rc = chk.main([str(actual_path), "--prediction", str(pred_path), "--out", str(out_path)])
    assert rc == 0
    assert json.loads(out_path.read_text())["any_out_of_band"] is False


# ---------------------------------------------------------------------------
# m4_run.sh --dry-run smoke test (subprocess, POSIX sh)
# ---------------------------------------------------------------------------

def test_m4_run_sh_help_exits_0_without_gpu():
    result = subprocess.run(["sh", str(M4_RUN_SH), "--help"],
                             capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "Usage: m4_run.sh" in result.stdout


def test_m4_run_sh_unknown_flag_exits_2():
    result = subprocess.run(["sh", str(M4_RUN_SH), "--not-a-real-flag"],
                             capture_output=True, text=True, timeout=30)
    assert result.returncode == 2


def test_m4_run_sh_dry_run_smoke(tmp_path):
    # a small, already-existing DREAMPlace config -- the 27.7M default
    # config need not exist on this dev host (design draft sec 6.4: the
    # 27.7M Bookshelf array is regenerated on the H100 node itself).
    config = REPO / "benchmarks" / "ispd25" / "synthetic_2x2_n2.json"
    assert config.exists(), "fixture config missing -- update to any small existing config"
    out = tmp_path / "dry_run_out.json"

    result = subprocess.run(
        ["sh", str(M4_RUN_SH), "--dry-run", "--config", str(config), "--out", str(out),
         "--k", "16", "--rho", "0.0", "--seed", "1000"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "== M4 T10 environment check ==" in result.stdout
    assert "torch:" in result.stdout
    assert "GPU:" in result.stdout
    assert "host RAM:" in result.stdout
    assert "run_placement.py" in result.stdout
    assert "--rho-max 0.0" in result.stdout
    assert "--no-diag" in result.stdout
    assert "== DRY RUN: not executing ==" in result.stdout
    assert not out.exists()  # dry-run never invokes the real placement
