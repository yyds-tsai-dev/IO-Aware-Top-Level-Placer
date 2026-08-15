"""M4 T6b/T7b (2026-08-15 T6 holdout adjudication doc `docs/results/2026-
08-15-m4-t6-holdout-adjudication.md` sec 6.1): tests for `ioplace/bench/
count_freeze.py`.

Everything here runs against small hand-written manifest fixtures (never
the real multi-GB arrays under `results/m4/bench/arrays/`) -- schema/
provenance-tag checks for both output builders, the int32/int64 headroom
formula (including a reproduction of the design draft sec 2.1's own worked
6.40% example), and a regression check against the specific frozen numbers
this task's instructions cite (1x2/2x2/3x3 actual g/gp, the 27.7M total),
using them as hardcoded fixture inputs rather than reading the real
manifests -- keeps the test fast and independent of the real arrays still
being present on disk.
"""
import json
import os

import pytest

from ioplace.bench import count_freeze as cf


# ---------------------------------------------------------------------------
# int/composite-key headroom formula
# ---------------------------------------------------------------------------

def test_headroom_frac_is_relative_to_count_not_capacity():
    # capacity 120, count 100 -> (120-100)/100 = 20%, not (120-100)/120 = 16.67%
    assert cf._headroom_frac(100, 120) == pytest.approx(0.20)


def test_headroom_frac_none_when_count_not_positive():
    assert cf._headroom_frac(0, 100) is None
    assert cf._headroom_frac(None, 100) is None


def test_int_headroom_report_reproduces_spec_sec_2_1_worked_example():
    """design draft sec 2.1: "3×3 base 已用掉 31,535,928,只剩 2,018,503
    (6.40%)給 glue" -- composite key capacity floor((2**31-1)/64) =
    33,554,431; headroom = 2,018,503 / 31,535,928 = 6.401%."""
    n_nets = 31_535_928
    report = cf.int_headroom_report(n_nets, n_pins=1)  # pins irrelevant to this leg
    composite = report["composite_key_net_times_64_plus_rid"]["int32"]
    assert composite["capacity"] == 33_554_431
    assert composite["count"] == n_nets
    assert composite["headroom_pct"] == pytest.approx(6.40, abs=0.01)
    assert composite["fits"] is True
    assert composite["warning_headroom_lt_20pct"] is True  # 6.40% < 20%


def test_int_headroom_report_flags_lt_20pct_warning_and_int64_never_warns():
    # count deliberately close to int32 capacity -> headroom < 20%
    near_cap = int(cf.INT32_MAX * 0.9)
    report = cf.int_headroom_report(n_nets=near_cap, n_pins=near_cap)
    assert report["nets"]["int32"]["warning_headroom_lt_20pct"] is True
    assert report["nets"]["int32"]["fits"] is True
    # int64 has enormous headroom at this scale -> never warns
    assert report["nets"]["int64"]["warning_headroom_lt_20pct"] is False
    assert report["pins"]["int64"]["warning_headroom_lt_20pct"] is False


def test_int_headroom_report_small_counts_have_ample_headroom():
    report = cf.int_headroom_report(n_nets=7_021_183, n_pins=24_078_796)
    assert report["nets"]["int32"]["warning_headroom_lt_20pct"] is False
    assert report["pins"]["int32"]["warning_headroom_lt_20pct"] is False
    assert report["composite_key_net_times_64_plus_rid"]["int32"]["warning_headroom_lt_20pct"] is False


# ---------------------------------------------------------------------------
# build_count_freeze_small / build_count_freeze_30m -- schema, tiny fixtures
# ---------------------------------------------------------------------------

def _write_manifest(path, base_nets, base_pins, glue_nets, glue_pins, base_nodes=None, extra=None):
    base = {"n_nets": base_nets, "n_pins": base_pins}
    if base_nodes is not None:
        base["n_nodes"] = base_nodes
    m = {"base": base, "glue": {"n_nets": glue_nets, "n_pins": glue_pins}}
    if extra:
        m.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f)


def test_build_count_freeze_small_schema(tmp_path):
    arrays_root = str(tmp_path / "arrays")
    _write_manifest(os.path.join(arrays_root, "1x2_n2", "1x2_n2.manifest.json"), 1000, 3000, 100, 200)
    _write_manifest(os.path.join(arrays_root, "1x2_n1", "1x2_n1.manifest.json"), 1000, 3000, 40, 80)
    _write_manifest(os.path.join(arrays_root, "2x2_n2", "2x2_n2.manifest.json"), 2000, 6000, 150, 300)
    _write_manifest(os.path.join(arrays_root, "2x2_n1", "2x2_n1.manifest.json"), 2000, 6000, 90, 180)

    out = cf.build_count_freeze_small(arrays_root=arrays_root)
    assert out["task"] == "T6b"
    assert set(out["rows"]) == {"1x2", "2x2"}
    row = out["rows"]["1x2"]
    assert row["g_n2"] == 100 and row["gp_n2"] == 200
    assert row["total_n_nets"] == 1100 and row["total_n_pins"] == 3200
    assert "int_headroom" in row
    assert out["rejected_alternative"]["1x2"]["g_n1"] == 40
    assert out["rejected_alternative"]["2x2"]["g_n1"] == 90
    assert out["host_rss_three_coefficient_model"]["status"] == "pending_t0b"
    for tag, value in cf.PROVENANCE_TAGS.items():
        assert out["provenance"][tag] == value


def test_build_count_freeze_30m_schema(tmp_path):
    arrays_root = str(tmp_path / "arrays")
    manifest_path = os.path.join(arrays_root, "3x3_n2", "3x3_n2.manifest.json")
    _write_manifest(manifest_path, 30000, 90000, 500, 1000, base_nodes=27000,
                     extra={"t6": {"expected_pair_counts": {"a|b": 250.0, "a|c": 250.0}}})

    out = cf.build_count_freeze_30m(arrays_root=arrays_root)
    assert out["task"] == "T7b"
    assert out["g9"] == 500 and out["gp9"] == 1000
    assert out["total_n_nets"] == 30500 and out["total_n_pins"] == 91000
    assert out["g9_expected_pre_measurement"] == pytest.approx(500.0)
    assert "int_headroom" in out
    assert out["citation_declaration"] == "T9 與 E5 的所有數字一律引用本檔 (count_freeze_30m.json)"
    for tag, value in cf.PROVENANCE_TAGS.items():
        assert out["provenance"][tag] == value


def test_provenance_tags_match_adjudication_sec_5_4():
    assert cf.PROVENANCE_TAGS["benchmark_kind"] == "synthetic"
    assert cf.PROVENANCE_TAGS["glue_provenance"] == "recipe_A_prime_uncertified"
    assert cf.PROVENANCE_TAGS["generator_verified"] is False
    assert cf.PROVENANCE_TAGS["t6_verdict"] == "FAIL"
    assert cf.PROVENANCE_TAGS["normalization"] == "n2"
    assert cf.PROVENANCE_TAGS["n2_rule"] == "sinkhorn"
    assert cf.PROVENANCE_TAGS["quality_claims_prohibited"] is True


# ---------------------------------------------------------------------------
# Regression against this task's cited frozen numbers (hardcoded fixture
# inputs, not the real multi-GB manifests -- keeps the test independent of
# whether results/m4/bench/arrays/ is present on disk).
# ---------------------------------------------------------------------------

def test_regression_1x2_2x2_n2_actual_counts_match_task_citation(tmp_path):
    arrays_root = str(tmp_path / "arrays")
    _write_manifest(os.path.join(arrays_root, "1x2_n2", "1x2_n2.manifest.json"),
                     7_007_976, 24_052_382, 13_207, 26_414)
    _write_manifest(os.path.join(arrays_root, "1x2_n1", "1x2_n1.manifest.json"),
                     7_007_976, 24_052_382, 4_413, 8_826)
    _write_manifest(os.path.join(arrays_root, "2x2_n2", "2x2_n2.manifest.json"),
                     14_015_952, 48_104_764, 26_341, 52_682)
    _write_manifest(os.path.join(arrays_root, "2x2_n1", "2x2_n1.manifest.json"),
                     14_015_952, 48_104_764, 26_034, 52_068)

    out = cf.build_count_freeze_small(arrays_root=arrays_root)
    assert out["rows"]["1x2"]["g_n2"] == 13_207
    assert out["rows"]["1x2"]["gp_n2"] == 26_414
    assert out["rows"]["2x2"]["g_n2"] == 26_341
    assert out["rows"]["2x2"]["gp_n2"] == 52_682
    assert out["rejected_alternative"]["1x2"]["g_n1"] == 4_413
    assert out["rejected_alternative"]["2x2"]["g_n1"] == 26_034


def test_regression_3x3_n2_actual_totals_match_task_citation(tmp_path):
    arrays_root = str(tmp_path / "arrays")
    _write_manifest(os.path.join(arrays_root, "3x3_n2", "3x3_n2.manifest.json"),
                     31_535_892, 108_235_719, 59_360, 118_720, base_nodes=27_804_699)
    out = cf.build_count_freeze_30m(arrays_root=arrays_root)
    assert out["g9"] == 59_360
    assert out["gp9"] == 118_720
    assert out["total_n_nets"] == 31_595_252
    assert out["total_n_pins"] == 108_354_439
