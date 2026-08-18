import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ioplace.diagnostics import m4_model_fit as mf


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# _normalize_record: both probe schemas this aggregator has to read.
# ---------------------------------------------------------------------------

def test_normalize_record_schema_a_probe_gp_memory():
    rec = dict(ok=True, config="/x/adaptec1.json", num_physical=210904, num_filler=160067,
               num_pins=919701, num_bins_x=512, num_bins_y=512,
               gp_peak_alloc_mb=140.8, gp_s=14.4, gp_iterations=873)
    norm = mf._normalize_record(rec, "/results/probe_gp_memory__adaptec1.json")
    assert norm["case"] == "adaptec1"
    assert norm["n_total"] == 210904 + 160067
    assert norm["n_pins"] == 919701
    assert norm["n_bins"] == 512 * 512
    assert norm["gp_peak_bytes"] == pytest.approx(140.8 * 2**20)
    assert norm["t_gp_s"] == 14.4
    assert norm["n_iter"] == 873


def test_normalize_record_schema_b_run_t0b_matrix_with_explicit_n_total():
    rec = dict(ok=True, tag="group__bins2048__td0.714", n_physical=3077669, n_filler=633128,
               n_total=3710797, n_pins=12026191, n_bins=2048 * 2048,
               gp_peak_alloc_gb=1.75, t_gp=92.0, gp_iterations_run=1600)
    norm = mf._normalize_record(rec, "/results/t0b/group__bins2048__td0.714.json")
    assert norm["case"] == "group__bins2048__td0.714"
    assert norm["n_total"] == 3710797
    assert norm["n_pins"] == 12026191
    assert norm["n_bins"] == 2048 * 2048
    assert norm["gp_peak_bytes"] == pytest.approx(1.75 * 2**30)
    assert norm["t_gp_s"] == 92.0
    assert norm["n_iter"] == 1600


def test_normalize_record_schema_b_derives_n_total_when_absent():
    rec = dict(ok=True, tag="x", n_physical=100, n_filler=25, n_pins=50, n_bins=16,
               gp_peak_alloc_gb=0.1, t_gp=1.0, gp_iterations_run=10)
    norm = mf._normalize_record(rec, "/p.json")
    assert norm["n_total"] == 125


def test_normalize_record_not_ok_is_skipped():
    rec = dict(ok=False, tag="x", n_physical=100, n_filler=0, n_pins=50, n_bins=16)
    assert mf._normalize_record(rec, "/p.json") is None


def test_normalize_record_unrecognized_schema_raises():
    with pytest.raises(ValueError, match="unrecognized"):
        mf._normalize_record(dict(ok=True, something_else=1), "/p.json")


# ---------------------------------------------------------------------------
# fit_ols_with_gate: both sides of the pre-registered kappa/CI rejection gate.
# ---------------------------------------------------------------------------

def test_fit_passes_when_well_conditioned_and_tight_ci():
    # Full 2^3 factorial (orthogonal design -> kappa(X_std) ~= 1), replicated
    # twice with small noise -> tight per-coefficient CIs.
    rng = np.random.default_rng(0)
    levels = [(100.0, 200.0), (50.0, 150.0), (16.0, 64.0)]
    combos = [(a, b, c) for a in levels[0] for b in levels[1] for c in levels[2]]
    rows = combos * 2
    X = np.array(rows)
    noise = rng.normal(0, 1.0, size=len(rows))
    y = 1000 + 2 * X[:, 0] + 3 * X[:, 1] + 5 * X[:, 2] + noise
    cases = [f"c{i}" for i in range(len(rows))]

    fit = mf.fit_ols_with_gate(X, y, ["a", "b", "c"], cases)

    assert fit["kappa_std"] < 5.0
    assert fit["max_ci95_rel_half_width"] < mf.CI_REL_HALF_WIDTH_THRESHOLD
    assert fit["identifiable"] is True
    assert fit["rejection_reasons"] == []
    assert fit["coefficients"]["a"]["value"] == pytest.approx(2.0, abs=0.1)
    assert fit["coefficients"]["b"]["value"] == pytest.approx(3.0, abs=0.1)
    assert fit["coefficients"]["c"]["value"] == pytest.approx(5.0, abs=0.1)
    assert len(fit["loo_residuals"]) == len(rows)


def test_fit_rejected_via_high_condition_number():
    # n_pins nearly collinear with n_total -- the exact v1 failure mode
    # (design draft sec 1.1/Codex finding #1) T0b's net-drop design point
    # exists to break.
    rng = np.random.default_rng(1)
    n_total = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
    n_pins = n_total * 4.0 + rng.normal(0, 0.5, size=6)
    n_bins = np.array([16.0, 16.0, 64.0, 64.0, 256.0, 256.0])
    X = np.column_stack([n_total, n_pins, n_bins])
    y = 1000 + 2 * n_total + 3 * n_pins + 5 * n_bins + rng.normal(0, 1.0, size=6)
    cases = [f"c{i}" for i in range(6)]

    fit = mf.fit_ols_with_gate(X, y, ["a", "b", "c"], cases)

    assert fit["kappa_std"] > mf.KAPPA_THRESHOLD
    assert fit["identifiable"] is False
    assert any("kappa" in r for r in fit["rejection_reasons"])


def test_fit_rejected_via_wide_confidence_interval_despite_low_kappa():
    # Well-conditioned design (orthogonal factorial corners) but too few
    # points relative to noise -> wide per-coefficient CIs while
    # kappa(X_std) stays well under threshold. Exercises the CI leg of the
    # gate independently of the kappa leg.
    rng = np.random.default_rng(3)
    levels = [(100.0, 200.0), (50.0, 150.0), (16.0, 64.0)]
    combos = [(a, b, c) for a in levels[0] for b in levels[1] for c in levels[2]][:5]
    X = np.array(combos)
    y = 1000 + 2 * X[:, 0] + 3 * X[:, 1] + 5 * X[:, 2] + rng.normal(0, 80.0, size=5)
    cases = [f"c{i}" for i in range(5)]

    fit = mf.fit_ols_with_gate(X, y, ["a", "b", "c"], cases)

    assert fit["kappa_std"] < mf.KAPPA_THRESHOLD
    assert fit["max_ci95_rel_half_width"] > mf.CI_REL_HALF_WIDTH_THRESHOLD
    assert fit["identifiable"] is False
    assert any("half-width" in r for r in fit["rejection_reasons"])


def test_fit_raises_on_insufficient_points():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="not enough"):
        mf.fit_ols_with_gate(X, y, ["a", "b", "c"], ["c0", "c1", "c2", "c3"])


# ---------------------------------------------------------------------------
# build_model_fit end-to-end: file loading, schema mixing, exclusions.
# ---------------------------------------------------------------------------

# Deterministic 8-point 2^3-factorial fixture on (n_total, n_pins, n_bins)
# reproducing sec 1.1's v1 coefficients (87/73/160) with an intercept and
# small noise -- passes the identifiability gate (verified interactively
# during authoring: kappa(X_std) == 1.0, max CI rel half-width ~= 0.01).
_LEVELS = [(1.0e6, 2.0e6), (4.0e6, 8.0e6), (1024.0, 4096.0)]
_COMBOS = [(a, b, c) for a in _LEVELS[0] for b in _LEVELS[1] for c in _LEVELS[2]]
_Y = [1379165363.6, 1379650160.1, 1671167592.3, 1671660062.8,
      1466154084.8, 1466648849.1, 1758164479.2, 1758653778.8]


def _write_t0b_point(t0b_dir, tag, n_total, n_pins, n_bins, gp_peak_bytes, ok=True):
    rec = dict(ok=ok, tag=tag, n_physical=int(n_total * 0.8), n_filler=int(n_total * 0.2),
               n_total=n_total, n_pins=n_pins, n_bins=n_bins,
               gp_peak_alloc_gb=gp_peak_bytes / 2**30, t_gp=100.0 + 1e-5 * n_total,
               gp_iterations_run=2000)
    path = os.path.join(t0b_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(rec, f)
    return path


def test_build_model_fit_end_to_end(tmp_path):
    t0b_dir = str(tmp_path / "t0b")
    os.makedirs(t0b_dir)

    for i, ((n_total, n_pins, n_bins), y) in enumerate(zip(_COMBOS, _Y)):
        _write_t0b_point(t0b_dir, f"pt{i}", n_total, n_pins, n_bins, y)

    # a failed run must be skipped, not fed into the fit
    _write_t0b_point(t0b_dir, "oom_point", 9e9, 9e9, 9e9, 1.0, ok=False)

    # the --dry-run plan artifact living in the same directory must be
    # excluded, not misread as a case-level run.
    with open(os.path.join(t0b_dir, "dry_run_plan.json"), "w") as f:
        json.dump([{"tag": "not_a_run"}], f)

    case_json = str(tmp_path / "probe_gp_memory__adaptec1.json")
    with open(case_json, "w") as f:
        json.dump(dict(ok=True, config="adaptec1.json", num_physical=210904, num_filler=160067,
                        num_pins=919701, num_bins_x=512, num_bins_y=512,
                        gp_peak_alloc_mb=140.8, gp_s=14.4, gp_iterations=873), f)

    result = mf.build_model_fit(t0b_dir=t0b_dir, case_json_paths=[case_json])

    assert result["n_points_used"] == 9  # 8 t0b points + 1 case-level point
    assert len(result["design_matrix"]) == 9
    assert any("oom_point" in s["path"] for s in result["skipped"])
    assert all("dry_run_plan" not in row["source_path"] for row in result["design_matrix"])
    assert result["gp_memory_model"] is not None
    assert result["gp_runtime_model"] is not None
    assert isinstance(result["identifiable"], bool)
    assert result["design_matrix_units"]["gp_peak_bytes"] == mf.UNITS["gp_peak_bytes"]


def test_cli_writes_output_json(tmp_path):
    t0b_dir = str(tmp_path / "t0b")
    os.makedirs(t0b_dir)
    for i, ((n_total, n_pins, n_bins), y) in enumerate(zip(_COMBOS, _Y)):
        _write_t0b_point(t0b_dir, f"pt{i}", n_total, n_pins, n_bins, y)
    out_path = str(tmp_path / "model_fit.json")

    proc = subprocess.run(
        [sys.executable, "-m", "ioplace.diagnostics.m4_model_fit",
         "--t0b-dir", t0b_dir, "--out", out_path],
        cwd=_repo_root(), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    with open(out_path) as f:
        result = json.load(f)
    assert result["n_points_used"] == 8
    assert "identifiable" in result
