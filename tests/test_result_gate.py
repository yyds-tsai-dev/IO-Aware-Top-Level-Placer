"""M4 sec 7.0 RESULT GATE (`ioplace/bench/result_gate.py`). Pure-Python,
no CUDA needed anywhere in this module -- `check()`'s "must catch every
deliberately-broken sample" acceptance (sec 7.0) is exercised by starting
from one known-good synthetic record and breaking exactly one thing per
test."""
import pytest

from ioplace.bench.result_gate import (
    HW_BUDGET_GB, LIFETIME_SCHEMA_FIELDS, SPIKE_SCHEMA_FIELDS,
    assert_budget, check, check_table_uniqueness, feasibility_verdict,
)

# ---------------------------------------------------------------------------
# assert_budget (sec 1.4 B3)
# ---------------------------------------------------------------------------

def test_hw_budget_gb_table_values():
    assert HW_BUDGET_GB["NVIDIA L4"] == 19.5
    assert HW_BUDGET_GB["NVIDIA H100 80GB HBM3"] == 72.0
    assert HW_BUDGET_GB["_m2_10m_contract"] == 8.0


def test_assert_budget_passes_at_or_under_cap():
    assert_budget("NVIDIA L4", 19.5, "NVIDIA L4")     # exactly at cap
    assert_budget("NVIDIA L4", 10.0, "NVIDIA L4")      # under cap


def test_assert_budget_raises_over_cap():
    with pytest.raises(AssertionError):
        assert_budget("NVIDIA L4", 19.6, "NVIDIA L4")


def test_assert_budget_raises_for_unknown_source():
    with pytest.raises(AssertionError):
        assert_budget("NVIDIA L4", 5.0, "NVIDIA RTX 4090")


def test_assert_budget_m2_10m_contract_cap():
    assert_budget("NVIDIA L4", 8.0, "_m2_10m_contract")
    with pytest.raises(AssertionError):
        assert_budget("NVIDIA L4", 8.1, "_m2_10m_contract")


# ---------------------------------------------------------------------------
# feasibility_verdict (sec 2.2's three-state rule -- six branches)
# ---------------------------------------------------------------------------

def test_verdict_feasible():
    v = feasibility_verdict("ok", "completed", peak_gb=10.0, budget_gb=19.5,
                            oom_repeats=0, resident_lower_bound_gb=5.0)
    assert v == "feasible_l4_contract"


def test_verdict_infeasible_completed_over_budget():
    v = feasibility_verdict("ok", "completed", peak_gb=20.0, budget_gb=19.5,
                            oom_repeats=0, resident_lower_bound_gb=5.0)
    assert v == "infeasible_l4_contract"


def test_verdict_infeasible_oom_reproduced_twice():
    v = feasibility_verdict("ok", "oom", peak_gb=None, budget_gb=19.5,
                            oom_repeats=2, resident_lower_bound_gb=None)
    assert v == "infeasible_l4_contract"


def test_verdict_infeasible_analytic_lower_bound_exceeds_device():
    v = feasibility_verdict("ok", "crashed", peak_gb=None, budget_gb=19.5,
                            oom_repeats=0, resident_lower_bound_gb=25.0)
    assert v == "infeasible_l4_contract"


def test_verdict_invalid_experiment_status_not_ok():
    v = feasibility_verdict("contaminated", "completed", peak_gb=1.0, budget_gb=19.5,
                            oom_repeats=0, resident_lower_bound_gb=1.0)
    assert v == "invalid_measurement"


def test_verdict_invalid_unreproduced_oom_or_crash_fallback():
    # experiment_status=="ok" but neither the feasible nor any infeasible
    # condition is met (oom seen only once, no analytic lower bound) --
    # sec 2.2 names no 4th literal for this, folded into invalid_measurement.
    v = feasibility_verdict("ok", "oom", peak_gb=None, budget_gb=19.5,
                            oom_repeats=1, resident_lower_bound_gb=None)
    assert v == "invalid_measurement"
    v2 = feasibility_verdict("ok", "crashed", peak_gb=None, budget_gb=19.5,
                             oom_repeats=0, resident_lower_bound_gb=None)
    assert v2 == "invalid_measurement"


def test_verdict_always_returns_one_of_three_literals():
    literals = {"feasible_l4_contract", "infeasible_l4_contract", "invalid_measurement"}
    for experiment_status in ("ok", "contaminated", "instrumentation_error", "input_error"):
        for workload_status in ("completed", "oom", "crashed"):
            for peak_gb in (None, 1.0, 100.0):
                for oom_repeats in (None, 0, 1, 2, 3):
                    for rlb in (None, 1.0, 100.0):
                        v = feasibility_verdict(experiment_status, workload_status, peak_gb,
                                                19.5, oom_repeats, rlb)
                        assert v in literals


# ---------------------------------------------------------------------------
# check() -- sec 7.0's six rules
# ---------------------------------------------------------------------------

def _good_profile_record():
    return {
        "case": "toy", "K": 8, "rtype": "grid", "arm": "flat",
        "t_read": 1.0, "t_initialize": 0.1, "t_gp": 2.0, "t_lg": 0.5,
        "t_eval_total": 0.3, "t_op_total": 0.2, "t_diag_total": 0.1, "t_total": 4.0,
        "iter_ms": [1.0] * 10, "iter_ms_p50": 1.0, "iter_ms_p90": 1.0,
        "iter_ms_p99": 1.0, "iter_ms_max": 1.0,
        "op_fwd_ms": [1.0], "op_bwd_ms": [1.0], "eval_ms": [1.0],
        "phases": {}, "device_used_gb": 1.0, "host_peak_rss_gb": 1.0,
        "n_movable": 10, "n_physical": 12, "n_filler": 2, "n_nets": 8, "n_pins": 30,
        "lattice": 512, "n_bins": 64, "k_chunk": 1, "n_active": 8,
        "bytes_per_pin_gp": None, "ns_per_pin_op": None, "gb_per_mnet_eval": None,
        "env": {},
        "experiment_status": "ok", "workload_status": "completed",
        "benchmark_kind": "real",
        "run_id": "r-1", "repo_commit": "abc123", "dp_commit": "def456",
        "command": "pytest", "hostname": "host1", "gpu_name": "NVIDIA L4",
        "seed": 0, "input_sha256": {"cfg.json": "deadbeef"},
        "device_baseline_gb": 0.1,
        "n_iter_expected": 10,
    }


def test_check_accepts_a_good_profile_record():
    ok, reasons = check(_good_profile_record())
    assert ok, reasons


def test_check_catches_experiment_status_not_ok():
    r = _good_profile_record()
    r["experiment_status"] = "contaminated"
    ok, reasons = check(r)
    assert not ok
    assert any("experiment_status" in x for x in reasons)


def test_check_catches_invalid_workload_status():
    r = _good_profile_record()
    r["workload_status"] = "not_a_real_status"
    ok, reasons = check(r)
    assert not ok
    assert any("workload_status" in x for x in reasons)


def test_check_catches_missing_schema_field():
    r = _good_profile_record()
    del r["n_nets"]
    ok, reasons = check(r)
    assert not ok
    assert any("missing schema fields" in x for x in reasons)


def test_check_catches_invalid_benchmark_kind():
    r = _good_profile_record()
    r["benchmark_kind"] = "toy"
    ok, reasons = check(r)
    assert not ok
    assert any("benchmark_kind" in x for x in reasons)


def test_check_catches_nan_field():
    r = _good_profile_record()
    r["t_total"] = float("nan")
    ok, reasons = check(r)
    assert not ok
    assert any("non-finite" in x for x in reasons)


def test_check_catches_inf_in_nested_list():
    r = _good_profile_record()
    r["iter_ms"] = [1.0, float("inf"), 2.0]
    ok, reasons = check(r)
    assert not ok
    assert any("non-finite" in x for x in reasons)


@pytest.mark.parametrize("field", ["run_id", "repo_commit", "dp_commit", "command",
                                   "hostname", "gpu_name", "seed", "input_sha256"])
def test_check_catches_missing_provenance_field(field):
    r = _good_profile_record()
    r[field] = None
    ok, reasons = check(r)
    assert not ok
    assert any("provenance" in x for x in reasons)


def test_check_catches_empty_iter_ms_below_expected():
    r = _good_profile_record()
    r["n_iter_expected"] = 100
    r["iter_ms"] = [1.0, 2.0]      # far below 0.9*100
    ok, reasons = check(r)
    assert not ok
    assert any("rule 6" in x for x in reasons)


def test_check_catches_zero_n_nets():
    r = _good_profile_record()
    r["n_nets"] = 0
    ok, reasons = check(r)
    assert not ok
    assert any("rule 6" in x for x in reasons)


def test_check_multiple_breaks_reports_multiple_reasons():
    r = _good_profile_record()
    r["experiment_status"] = "crashed"
    r["t_total"] = float("nan")
    ok, reasons = check(r)
    assert not ok
    assert len(reasons) >= 2


# ---------------------------------------------------------------------------
# lifetime/spike schema-specific rule 6 mapping
# ---------------------------------------------------------------------------

def _good_lifetime_record():
    return {
        "record_kind": "lifetime", "case": "mempool_group", "K": 32, "rtype": "grid",
        "arm": "ours@M2", "benchmark_kind": "real",
        "budget_gb": 19.5, "budget_source": "NVIDIA L4",
        "experiment_status": "ok", "workload_status": "completed",
        "feasibility_verdict": "feasible_l4_contract",
        "device_baseline_gb": 0.1, "exclusivity_evidence": "baseline_only",
        "host_mem_available_gb_at_start": 100.0,
        "buffers": [], "phases": [], "totals": {},
        "n_callbacks_with_active": 5, "n_evals_while_active": 5,
        "of_on": 2.0, "gp_iterations_run": 5,
        "run_id": "r-1", "repo_commit": "abc", "dp_commit": "def",
        "command": "x", "hostname": "h", "gpu_name": "NVIDIA L4",
        "seed": 0, "input_sha256": {"cfg.json": "beef"},
    }


def test_check_lifetime_schema_accepts_good_record():
    ok, reasons = check(_good_lifetime_record(), schema=LIFETIME_SCHEMA_FIELDS)
    assert ok, reasons


def test_check_lifetime_schema_rule6_needs_at_least_3_active_callbacks():
    r = _good_lifetime_record()
    r["n_callbacks_with_active"] = 2
    ok, reasons = check(r, schema=LIFETIME_SCHEMA_FIELDS)
    assert not ok
    assert any("rule 6" in x for x in reasons)


def _good_spike_record():
    return {
        "record_kind": "spike", "case": "group_3x3", "K": 32, "benchmark_kind": "synthetic",
        "experiment_status": "ok", "workload_status": "completed",
        "feasibility_verdict": "feasible_l4_contract",
        "measured_peak_gb": 10.0, "budget_gb": 19.5, "budget_source": "NVIDIA L4",
        "baseline_reserved_gb": 0.2, "device_baseline_gb": 0.1,
        "resident_gb": 5.0, "resident_lower_bound_gb": 5.0, "max_phase_transient_gb": 5.0,
        "retry_count": 0, "n_interleaved_iters": 5,
        "run_id": "r-2", "repo_commit": "abc", "dp_commit": "def",
        "command": "x", "hostname": "h", "gpu_name": "NVIDIA L4",
        "seed": 0, "input_sha256": {"cfg.json": "beef"},
    }


def test_check_spike_schema_accepts_good_record():
    ok, reasons = check(_good_spike_record(), schema=SPIKE_SCHEMA_FIELDS)
    assert ok, reasons


def test_check_spike_schema_rule6_needs_at_least_3_interleaved_iters():
    r = _good_spike_record()
    r["n_interleaved_iters"] = 1
    ok, reasons = check(r, schema=SPIKE_SCHEMA_FIELDS)
    assert not ok
    assert any("rule 6" in x for x in reasons)


# ---------------------------------------------------------------------------
# check_table_uniqueness (rule 5)
# ---------------------------------------------------------------------------

def test_check_table_uniqueness_passes_on_distinct_run_ids():
    ok, dupes = check_table_uniqueness(["a", "b", "c"])
    assert ok and dupes == []


def test_check_table_uniqueness_catches_duplicate_run_id():
    ok, dupes = check_table_uniqueness(["a", "b", "a"])
    assert not ok and dupes == ["a"]
