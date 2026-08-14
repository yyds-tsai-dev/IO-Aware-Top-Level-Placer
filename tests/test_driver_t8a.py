"""M4 T8a: driver instrumentation (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` T8a row /
sec 6.3 E1) -- adds the E1 gate fields (`num_unplaced_cells`,
`final_overflow`, `legalization_status`) and RESULT GATE provenance
(`run_id`, `status`, `schema_version`, `repo_commit`, `input_sha256`, `seed`)
to `run_placement.run_flat` / `run_placement_io.run_io`, and wires
`ioplace/profile.py`'s PhaseTimer/DeviceMemSampler in as four named phases
(`read`/`gp`/`lg`/`eval`), each with its own peak-memory reset and host-RSS
HWM snapshot.

Five things this file locks down:
  1. `_legalization_fields` (pure mapping, no DREAMPlace/CUDA needed) covers
     the success/failed/skipped branches directly -- this is T8a's required
     "legalization 失敗" 人工情境 test, exercised without needing a real
     DREAMPlace run that actually produces an illegal placement.
  2. A real run_flat/run_io on `simple.json` produces all the new fields
     with sane values (num_unplaced_cells==0, legalization_status=="success").
  3. phases has "read"/"gp"/"lg"/"eval" entries, each carrying
     t_s/peak_alloc_gb/peak_reserved_gb/host_rss_hwm_at_phase_end.
  4. RESULT GATE provenance: run_id is a fresh uuid4 per call, status=="ok",
     schema_version matches the module constant, repo_commit matches the
     actual git HEAD, input_sha256 covers the config path.
  5. Existing fields/bit-lock are untouched: peak_mem_mb_reset_semantics
     still True and peak_mem_mb still reflects a real (not stale-process)
     peak; --ft-reweight off's bit-identical lock (tests/test_ft_reweight.py)
     and the rho_max==0 observer-mode bit-identical lock
     (tests/test_driver_io.py) are exercised by their own existing tests,
     not duplicated here -- this file only adds new assertions on top of
     what those already cover.
"""
import json
import os
import subprocess
import uuid as uuid_mod

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement import (RESULT_SCHEMA_VERSION,
    _legalization_fields, run_flat)
from ioplace.drivers.run_placement_io import RESULT_FIELDS, run_io

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASE_KEYS = {"t_s", "peak_alloc_gb", "peak_reserved_gb", "host_rss_hwm_at_phase_end"}


def _git_head():
    return subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()


# -- 1. pure mapping: the "legalization failed" artificial scenario --------

def test_legalization_fields_success():
    assert _legalization_fields(True, 0, 1) == {
        "legalization_status": "success", "num_unplaced_cells": 0}


def test_legalization_fields_failed_artificial_scenario():
    """T8a acceptance's required 人工情境: a run where legality_check_op
    would have returned False and 5 movable cells ended up non-finite/
    out-of-die -- exercised directly against the pure mapping, no need to
    coerce a real DREAMPlace run into actually producing an illegal
    placement."""
    assert _legalization_fields(False, 5, 1) == {
        "legalization_status": "failed", "num_unplaced_cells": 5}


def test_legalization_fields_skipped_when_legalize_flag_off():
    assert _legalization_fields(None, 0, 0) == {
        "legalization_status": "skipped", "num_unplaced_cells": None}


# -- 2/3/4/5. real runs -----------------------------------------------------

def _assert_phase_shape(phases):
    for name in ("read", "gp", "lg", "eval"):
        assert name in phases, f"missing phase {name!r}"
        assert PHASE_KEYS <= phases[name].keys(), (name, phases[name].keys())
        assert phases[name]["t_s"] >= 0.0
        assert phases[name]["host_rss_hwm_at_phase_end"] > 0.0


def _assert_result_gate_provenance(res, config_json):
    assert isinstance(res["run_id"], str)
    uuid_mod.UUID(res["run_id"])                        # raises if not a uuid
    assert res["status"] == "ok"
    assert res["schema_version"] == RESULT_SCHEMA_VERSION
    assert res["repo_commit"] == _git_head()
    assert config_json in res["input_sha256"]
    assert len(res["input_sha256"][config_json]) == 64  # sha256 hex digest


@pytest.mark.slow
def test_flat_t8a_fields_on_simple(tmp_path):
    out = str(tmp_path / "flat.json")
    res = run_flat(SIMPLE, 4, "grid", 0, out)

    assert res["legalization_status"] == "success"
    assert res["num_unplaced_cells"] == 0
    assert np.isfinite(res["final_overflow"])

    _assert_phase_shape(res["phases"])
    assert res["t_read"] >= 0.0 and res["t_gp"] > 0.0 and res["t_eval"] >= 0.0
    assert res["device_used_gb"] >= 0.0
    assert res["host_peak_rss_gb"] > 0.0

    _assert_result_gate_provenance(res, SIMPLE)

    # existing fields/semantics untouched
    assert res["peak_mem_mb_reset_semantics"] is True
    assert res["peak_mem_mb"] >= 0.0
    assert res["mode"] == "flat"

    on_disk = json.load(open(out))
    for f in ("num_unplaced_cells", "final_overflow", "legalization_status",
             "run_id", "status", "schema_version", "repo_commit",
             "input_sha256", "phases"):
        assert f in on_disk


@pytest.mark.slow
def test_io_t8a_fields_on_simple(tmp_path):
    out = str(tmp_path / "io.json")
    res = run_io(SIMPLE, 4, "grid", 0, out, rho_max=0.05, every=10)

    assert res["legalization_status"] == "success"
    assert res["num_unplaced_cells"] == 0
    assert np.isfinite(res["final_overflow"])

    _assert_phase_shape(res["phases"])
    _assert_result_gate_provenance(res, SIMPLE)

    assert res["peak_mem_mb_reset_semantics"] is True
    assert res["mode"] == "io"

    on_disk = json.load(open(out))
    for f in ("num_unplaced_cells", "final_overflow", "legalization_status",
             "run_id", "status", "schema_version", "repo_commit",
             "input_sha256", "phases"):
        assert f in on_disk


@pytest.mark.slow
def test_run_id_is_unique_across_runs(tmp_path):
    a = run_flat(SIMPLE, 4, "grid", 0, str(tmp_path / "a.json"))
    b = run_flat(SIMPLE, 4, "grid", 0, str(tmp_path / "b.json"))
    assert a["run_id"] != b["run_id"]


def test_result_fields_declare_the_t8a_columns():
    for f in ("num_unplaced_cells", "final_overflow", "legalization_status",
             "run_id", "status", "schema_version", "repo_commit",
             "input_sha256", "phases", "t_read", "t_gp", "t_lg", "t_eval",
             "device_used_gb", "host_peak_rss_gb"):
        assert f in RESULT_FIELDS, f
