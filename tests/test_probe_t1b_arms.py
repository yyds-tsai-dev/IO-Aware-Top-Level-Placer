"""M4 T1b (a) tests for `ioplace.diagnostics.probes_m4.probe_t1b_arms` --
design draft `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`
sec 1.4 B1 / T1b row (a).

CPU-only (task instruction: no GPU work while T8 is running): every test
here either exercises pure logic (`build_judgment`, `_prepared_config`,
CLI parsing) or mocks the subprocess-launching functions
(`_run_arm_subprocess`/`_run_protocol_subprocess`) entirely -- no real
`probe_t1b_arms.py --single-arm`/`--run-protocol` subprocess is ever
launched, and `run_one_arm`/`run_protocol_in_process` (which DO touch
CUDA/DREAMPlace) are never called.
"""
import json
import os

import pytest

from ioplace.diagnostics.probes_m4 import probe_t1b_arms as tb


def _arm(run_start_allocated_gb, run_end_allocated_gb=None, peak_alloc_gb=1.0):
    return {
        "run_start_allocated_gb": run_start_allocated_gb,
        "run_start_reserved_gb": run_start_allocated_gb,
        "run_end_allocated_gb": run_end_allocated_gb if run_end_allocated_gb is not None
                               else run_start_allocated_gb,
        "run_end_reserved_gb": run_start_allocated_gb,
        "peak_alloc_gb": peak_alloc_gb, "peak_reserved_gb": peak_alloc_gb,
    }


def test_arm_fields_schema_matches_fixture_helper():
    """`_arm()` above mirrors exactly what `run_one_arm` (untested here --
    it touches CUDA/DREAMPlace, out of scope for this CPU-only file) is
    documented to return; this pins ARM_FIELDS itself against drift."""
    assert set(_arm(0.1).keys()) == set(tb.ARM_FIELDS)


# ---------------------------------------------------------------------------
# build_judgment
# ---------------------------------------------------------------------------

def test_judgment_cumulative_hwm_when_reset_only_fixes_baseline():
    """reset_only's arm2 baseline matches arm1's (within tolerance) but
    same_process's does not -- classic 'counter wasn't reset' signature."""
    protocols = {
        "fresh_subprocess": {"arm1": _arm(0.10), "arm2": _arm(0.10)},
        "same_process": {"arm1": _arm(0.10), "arm2": _arm(0.50)},
        "reset_only": {"arm1": _arm(0.10), "arm2": _arm(0.101)},
        "teardown_gc": {"arm1": _arm(0.10), "arm2": _arm(0.101)},
    }
    j = tb.build_judgment(protocols)
    assert j["reset_only_fixes_baseline"] is True
    assert j["cause"] == "cumulative_hwm"


def test_judgment_retained_tensors_when_only_teardown_fixes_baseline():
    """reset_only's arm2 baseline stays elevated (reset doesn't free live
    tensors) but teardown_gc's comes back down -- B1's adjudicated cause."""
    protocols = {
        "fresh_subprocess": {"arm1": _arm(0.10), "arm2": _arm(0.10)},
        "same_process": {"arm1": _arm(0.10), "arm2": _arm(0.50)},
        "reset_only": {"arm1": _arm(0.10), "arm2": _arm(0.50)},
        "teardown_gc": {"arm1": _arm(0.10), "arm2": _arm(0.102)},
    }
    j = tb.build_judgment(protocols)
    assert j["reset_only_fixes_baseline"] is False
    assert j["teardown_within_2pct_tolerance"] is True
    assert j["cause"] == "retained_tensors"


def test_judgment_undetermined_when_neither_protocol_fixes_baseline():
    protocols = {
        "fresh_subprocess": {"arm1": _arm(0.10), "arm2": _arm(0.10)},
        "same_process": {"arm1": _arm(0.10), "arm2": _arm(0.50)},
        "reset_only": {"arm1": _arm(0.10), "arm2": _arm(0.50)},
        "teardown_gc": {"arm1": _arm(0.10), "arm2": _arm(0.45)},
    }
    j = tb.build_judgment(protocols)
    assert j["reset_only_fixes_baseline"] is False
    assert j["teardown_within_2pct_tolerance"] is False
    assert j["cause"] == "undetermined"


def test_judgment_records_delta_gb_and_pct_per_protocol():
    protocols = {
        "fresh_subprocess": {"arm1": _arm(0.10), "arm2": _arm(0.10)},
        "same_process": {"arm1": _arm(0.10), "arm2": _arm(0.20)},
        "reset_only": {"arm1": _arm(0.10), "arm2": _arm(0.10)},
        "teardown_gc": {"arm1": _arm(0.10), "arm2": _arm(0.10)},
    }
    j = tb.build_judgment(protocols)
    assert j["same_process"]["arm2_baseline_delta_gb"] == pytest.approx(0.10)
    assert j["same_process"]["arm2_baseline_delta_pct"] == pytest.approx(100.0)
    assert j["fresh_subprocess"]["arm2_baseline_delta_gb"] == pytest.approx(0.0)


def test_judgment_zero_baseline_falls_back_to_absolute_gb_tolerance():
    """When arm1's own baseline is ~0 GB, a percent delta is undefined
    (divide by ~0) -- `_pct_delta` returns None and the tolerance check
    falls back to an absolute-GB comparison instead of raising."""
    protocols = {
        "fresh_subprocess": {"arm1": _arm(0.0), "arm2": _arm(0.0)},
        "same_process": {"arm1": _arm(0.0), "arm2": _arm(0.0)},
        "reset_only": {"arm1": _arm(0.0), "arm2": _arm(0.0)},
        "teardown_gc": {"arm1": _arm(0.0), "arm2": _arm(0.0)},
    }
    j = tb.build_judgment(protocols)
    assert j["reset_only"]["arm2_baseline_delta_pct"] is None
    assert j["reset_only_fixes_baseline"] is True
    assert j["cause"] == "cumulative_hwm"


# ---------------------------------------------------------------------------
# _prepared_config
# ---------------------------------------------------------------------------

def test_prepared_config_overwrites_iteration_count(tmp_path):
    src = tmp_path / "cfg.json"
    src.write_text(json.dumps({
        "aux_input": "x.aux",
        "global_place_stages": [{"iteration": 1000, "learning_rate": 0.01}],
    }))
    out_path = tb._prepared_config(str(src), 7)
    try:
        with open(out_path) as f:
            cfg = json.load(f)
        assert cfg["global_place_stages"][0]["iteration"] == 7
        assert cfg["global_place_stages"][0]["learning_rate"] == 0.01
        assert cfg["aux_input"] == "x.aux"
        assert os.path.isabs(out_path)
    finally:
        os.remove(out_path)


# ---------------------------------------------------------------------------
# between-arm hooks (pure Python -- CPU-safe branches only; the
# torch.cuda.is_available() guards mean these are no-ops without a GPU,
# exercised here only for "does not raise")
# ---------------------------------------------------------------------------

def test_between_same_process_hook_is_a_no_op():
    tb._between_same_process({"a": object()})   # must not raise


def test_between_teardown_gc_hook_drains_live_objects_dict():
    live = {"params": object(), "placedb": object()}
    tb._between_teardown_gc(live)
    assert live == {}


# ---------------------------------------------------------------------------
# run() orchestration -- fully mocked at the subprocess-launch boundary
# ---------------------------------------------------------------------------

def _fake_protocol_result(arm1_gb, arm2_gb):
    return {"arm1": _arm(arm1_gb), "arm2": _arm(arm2_gb)}


def test_run_assembles_schema_with_mocked_subprocess_launches(tmp_path, monkeypatch):
    calls = []

    def fake_run_arm_subprocess(python, config_json, iterations, seed, out_path, timeout_s=600.0):
        calls.append(("arm", out_path))
        return _arm(0.10)

    def fake_run_protocol_subprocess(python, protocol, config_json, iterations, seed,
                                     out_path, timeout_s=600.0):
        calls.append(("protocol", protocol))
        return _fake_protocol_result(0.10, 0.101)

    monkeypatch.setattr(tb, "_run_arm_subprocess", fake_run_arm_subprocess)
    monkeypatch.setattr(tb, "_run_protocol_subprocess", fake_run_protocol_subprocess)

    # env_metadata() sha256-hashes every input_path -- needs a real
    # (content doesn't matter) file on disk, not a bare made-up name.
    config_json = str(tmp_path / "fake_config.json")
    with open(config_json, "w") as f:
        f.write("{}")

    out_json = str(tmp_path / "t1b_arms.json")
    record = tb.run(config_json=config_json, iterations=5, seed=0,
                    python="fake-python", out_json=out_json)

    assert record["record_kind"] == "t1b_arms"
    assert set(record["protocols"].keys()) == set(tb.PROTOCOLS)
    assert record["protocols"]["fresh_subprocess"]["arm1"] == _arm(0.10)
    assert "judgment" in record and "cause" in record["judgment"]
    assert os.path.exists(out_json)
    with open(out_json) as f:
        on_disk = json.load(f)
    assert on_disk["record_kind"] == "t1b_arms"

    # fresh_subprocess launches 2 arm subprocesses; the other 3 protocols
    # launch 1 protocol subprocess each.
    arm_calls = [c for c in calls if c[0] == "arm"]
    protocol_calls = [c for c in calls if c[0] == "protocol"]
    assert len(arm_calls) == 2
    assert {c[1] for c in protocol_calls} == set(tb.IN_PROCESS_PROTOCOLS)


def test_run_protocol_in_process_applies_hook_between_arms(monkeypatch):
    """run_protocol_in_process itself (not the subprocess launcher) --
    mocks run_one_arm so no CUDA/DREAMPlace call happens, and asserts the
    protocol's between-arm hook was invoked exactly once, between the two
    run_one_arm calls."""
    call_order = []

    def fake_run_one_arm(config_json, iterations, seed=0):
        call_order.append("run_one_arm")
        return _arm(0.1), {"live": object()}

    def fake_hook(live_objects):
        call_order.append("hook")

    monkeypatch.setattr(tb, "run_one_arm", fake_run_one_arm)
    monkeypatch.setattr(tb, "_BETWEEN_ARM_HOOKS", {"reset_only": fake_hook})

    result = tb.run_protocol_in_process("reset_only", "cfg.json", 5, 0)
    assert call_order == ["run_one_arm", "hook", "run_one_arm"]
    assert result == {"arm1": _arm(0.1), "arm2": _arm(0.1)}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = tb._parse_args([])
    assert args.config == tb.DEFAULT_CONFIG
    assert args.iterations == tb.DEFAULT_ITERATIONS
    assert args.single_arm is False
    assert args.run_protocol is None


def test_parse_args_run_protocol_rejects_fresh_subprocess():
    """fresh_subprocess is not an IN_PROCESS_PROTOCOLS choice -- it's
    handled by launching --single-arm twice instead (module docstring)."""
    with pytest.raises(SystemExit):
        tb._parse_args(["--run-protocol", "fresh_subprocess"])


def test_parse_args_single_arm_flag():
    args = tb._parse_args(["--single-arm", "--out", "o.json"])
    assert args.single_arm is True
    assert args.out == "o.json"
