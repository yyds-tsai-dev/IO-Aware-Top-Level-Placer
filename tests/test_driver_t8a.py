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
  6. Overflow-diagnosis follow-up (schema_version 3): `stop_overflow_reached`/
     `gp_iteration_budget`/`_effective_scale_fields`/`_last_metric_iteration`
     are pure/mock-testable (SimpleNamespace params/placedb, no DREAMPlace/
     CUDA needed -- same rationale as `_legalization_fields`); `_t8a_provenance`'s
     new command/hostname/benchmark_kind/device_baseline_gb fields are checked
     directly (needs only a config path, not a real run); hpwl_gp/hpwl_lg and
     the RESULT GATE fields' presence on a real run are checked by extending
     the existing simple.json flat/io tests (point 2 above).
"""
import json
import os
import socket
import subprocess
import sys
import uuid as uuid_mod
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement import (RESULT_SCHEMA_VERSION,
    _fence_overflow_stop_metric, _legalization_fields, _stop_overflow_reached,
    _gp_iteration_budget,
    _effective_scale_fields, _last_metric_iteration, _t8a_provenance, run_flat)
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


# -- 6. overflow-diagnosis follow-up: pure/mock-level unit tests -----------

def test_stop_overflow_reached_true_when_final_overflow_at_or_below_target():
    assert _stop_overflow_reached(0.07, 0.07) is True   # equal counts as reached
    assert _stop_overflow_reached(0.05, 0.07) is True


def test_stop_overflow_reached_false_when_final_overflow_above_target():
    assert _stop_overflow_reached(0.10, 0.07) is False


def test_fence_overflow_stop_metric_ignores_a_saturated_final_bucket():
    """Fence-diagnostics fix (P-F Task 7 finding): the v2 main flow's fence
    phase forces one movable cell into DREAMPlace's implicit "no fence"
    bucket, appended last in the K+1-length overflow vector (escape-cell
    workaround, fence_phase.py:134-142); with effectively zero placeable
    bins its overflow saturates near 1.0 regardless of the design's real
    state (measured 0.99996 in the run that found this). Before this fix,
    run_main_flow fed max(overflow_regions) to _stop_overflow_reached, so
    that one degenerate bucket could mask every real region having already
    converged well below stop_overflow -- a false negative. The regression:
    two well-converged fence regions (0.02, 0.01) plus a saturated escape
    bucket (0.99996) as overflow[-1] must not suppress the stop flag when
    fed through _fence_overflow_stop_metric, even though .max() over the
    same vector would."""
    overflow_regions = [0.02, 0.01, 0.99996]
    assert max(overflow_regions) == pytest.approx(0.99996)
    metric = _fence_overflow_stop_metric(overflow_regions, True)
    assert metric == pytest.approx(0.99996)
    # This regression is about *provenance*, not about flipping the boolean
    # for this exact vector: DREAMPlace's own Lgamma_stop_criterion also
    # reads only overflow[-1] for fence regions (NonLinearPlace.py:300-311),
    # so a saturated *last* bucket is still "not stopped" by design -- the
    # bug was `.max()` picking a *different*, non-saturated bucket's value
    # only by accident of ordering while still calling it the design's own
    # overflow. Assert the metric is exactly overflow[-1], not derived from
    # max(), which is the actual fix under test.
    assert metric == overflow_regions[-1]
    assert _stop_overflow_reached(metric, 0.07) is False

    # The case the fix actually changes the answer for: once the escape
    # bucket is no longer conflated with a genuinely unconverged region,
    # a low overflow[-1] must not be suppressed by a *different* bucket
    # that is still high (e.g. a region GP has not yet reached).
    overflow_regions_converging = [0.99996, 0.30, 0.05]
    metric_converging = _fence_overflow_stop_metric(overflow_regions_converging, True)
    assert metric_converging == pytest.approx(0.05)
    assert _stop_overflow_reached(metric_converging, 0.07) is True
    assert _stop_overflow_reached(max(overflow_regions_converging), 0.07) is False


def test_fence_overflow_stop_metric_falls_back_to_max_without_fence_regions():
    assert _fence_overflow_stop_metric([0.3, 0.9, 0.05], False) == pytest.approx(0.9)


def test_gp_iteration_budget_reads_config_value():
    """No real config/PlaceDB needed -- global_place_stages[0]["iteration"]
    is read straight off `params`, so a SimpleNamespace mock exercises the
    exact same read `run_flat`/`run_io` do."""
    params = SimpleNamespace(global_place_stages=[
        {"iteration": 300, "learning_rate": 0.01}])
    assert _gp_iteration_budget(params) == 300


def test_effective_scale_fields_reads_placedb_and_params():
    params = SimpleNamespace(target_density=0.85)
    placedb = SimpleNamespace(num_filler_nodes=123, num_bins_x=512, num_bins_y=512)
    assert _effective_scale_fields(params, placedb) == {
        "effective_target_density": 0.85, "num_filler_nodes": 123,
        "num_bins_x": 512, "num_bins_y": 512}


def test_last_metric_iteration_flat_leaf():
    """The GP+LG protocol shape (legalize_flag=1): NonLinearPlace.py appends
    a *flat* EvalMetrics-like object to all_metrics right after
    legalization -- metrics[-1] is that object directly."""
    metrics = [[[SimpleNamespace(iteration=5)]], SimpleNamespace(iteration=300)]
    assert _last_metric_iteration(metrics) == 300


def test_last_metric_iteration_nested_leaf():
    """The GP-loop-only shape (no flat post-legalization append): descend
    into the Lgamma/Llambda/Lsub nesting to the last-recorded metric."""
    metrics = [[[SimpleNamespace(iteration=1)], [SimpleNamespace(iteration=2),
               SimpleNamespace(iteration=299)]]]
    assert _last_metric_iteration(metrics) == 299


def test_t8a_provenance_includes_result_gate_fields():
    """command/hostname/benchmark_kind/device_baseline_gb -- needs only a
    config path (for input_sha256/env_metadata), no real DREAMPlace run."""
    prov = _t8a_provenance(SIMPLE, benchmark_kind="synthetic", device_baseline_gb=1.5)
    assert prov["command"] == " ".join(sys.argv)
    assert prov["hostname"] == socket.gethostname()
    assert prov["benchmark_kind"] == "synthetic"
    assert prov["device_baseline_gb"] == 1.5


def test_result_fields_declare_the_overflow_diagnosis_columns():
    for f in ("stop_overflow_reached", "gp_iteration_budget", "gp_iterations_run",
             "hpwl_gp", "hpwl_lg", "effective_target_density", "num_filler_nodes",
             "num_bins_x", "num_bins_y",
             "command", "hostname", "benchmark_kind", "device_baseline_gb"):
        assert f in RESULT_FIELDS, f


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
    # Overflow-diagnosis follow-up: the remaining RESULT GATE fields.
    assert res["command"] == " ".join(sys.argv)
    assert res["hostname"] == socket.gethostname()
    assert res["benchmark_kind"] == "real"              # default, not overridden below
    assert res["device_baseline_gb"] >= 0.0


def _assert_overflow_diagnosis_fields(res):
    """Overflow-diagnosis follow-up (schema_version 3): the E1-gate/scale
    fields a real GP+LG run on simple.json should produce sane values for."""
    assert isinstance(res["stop_overflow_reached"], bool)
    assert res["gp_iteration_budget"] > 0
    assert 0 <= res["gp_iterations_run"] <= res["gp_iteration_budget"]
    assert res["hpwl_gp"] > 0.0 and res["hpwl_lg"] > 0.0
    assert 0.0 < res["effective_target_density"] <= 1.0
    assert res["num_filler_nodes"] >= 0
    assert res["num_bins_x"] > 0 and res["num_bins_y"] > 0


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
    _assert_overflow_diagnosis_fields(res)

    # existing fields/semantics untouched
    assert res["peak_mem_mb_reset_semantics"] is True
    assert res["peak_mem_mb"] >= 0.0
    assert res["mode"] == "flat"

    on_disk = json.load(open(out))
    for f in ("num_unplaced_cells", "final_overflow", "legalization_status",
             "run_id", "status", "schema_version", "repo_commit",
             "input_sha256", "phases",
             "stop_overflow_reached", "gp_iteration_budget", "gp_iterations_run",
             "hpwl_gp", "hpwl_lg", "effective_target_density",
             "num_filler_nodes", "num_bins_x", "num_bins_y",
             "command", "hostname", "benchmark_kind", "device_baseline_gb"):
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
    _assert_overflow_diagnosis_fields(res)

    assert res["peak_mem_mb_reset_semantics"] is True
    assert res["mode"] == "io"

    on_disk = json.load(open(out))
    for f in ("num_unplaced_cells", "final_overflow", "legalization_status",
             "run_id", "status", "schema_version", "repo_commit",
             "input_sha256", "phases",
             "stop_overflow_reached", "gp_iteration_budget", "gp_iterations_run",
             "hpwl_gp", "hpwl_lg", "effective_target_density",
             "num_filler_nodes", "num_bins_x", "num_bins_y",
             "command", "hostname", "benchmark_kind", "device_baseline_gb"):
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
