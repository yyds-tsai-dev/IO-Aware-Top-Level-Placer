"""M4 T9 tests for `ioplace.bench.spike_30m` (parent) / `spike_30m_child`
(child) -- design draft `docs/superpowers/specs/2026-08-13-m4-scale-up-
design-draft.md` sec 5.3 layer 2 / T9 row / sec 2.2's three-state
feasibility rule.

No GPU work is run anywhere in this file (task instruction: CPU tests
only, GPU is busy with a concurrent T8 run). The state-machine tests below
mock `spike_30m._run_one_attempt`/`_preflight` entirely -- they never
launch `spike_30m_child.py` as a real subprocess, let alone touch CUDA.
`spike_30m_child`'s own tests are limited to its CLI validation and the
CPU-safe branch of its reflection-based `_resident_bytes` helper (which,
with an object whose tensors are all `is_cuda=False`, correctly returns 0
without needing a GPU at all).
"""
import argparse
import inspect
import json
import os

import pytest
import torch

from ioplace.bench import result_gate
from ioplace.bench import spike_30m as sp
from ioplace.bench import spike_30m_child as spc


# ---------------------------------------------------------------------------
# contract: parent never touches CUDA
# ---------------------------------------------------------------------------

def test_parent_module_source_never_imports_torch():
    """Static enforcement of the module's own documented contract ("this
    module must NEVER build a CUDA context") -- a plain `import torch`
    doesn't create a context by itself, but this module goes further and
    avoids even that, so grepping the source for the string is a cheap,
    exact way to catch a future accidental `import torch` creeping in."""
    src = inspect.getsource(sp)
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import torch"), \
            f"spike_30m.py must never import torch: found {stripped!r}"
        assert "torch.cuda" not in stripped, \
            f"spike_30m.py must never reference torch.cuda: found {stripped!r}"


def test_parent_budget_default_matches_l4_contract():
    assert sp.DEFAULT_BUDGET_GB == 19.5
    assert sp.DEFAULT_BUDGET_SOURCE == "NVIDIA L4"


# ---------------------------------------------------------------------------
# child: CLI validation
# ---------------------------------------------------------------------------

def test_child_rejects_n_iter_below_contract_minimum():
    with pytest.raises(ValueError, match="contract minimum"):
        spc._parse_args(["--cache-dir", "x", "--k", "32",
                         "--budget-gb", "19.5", "--budget-source", "NVIDIA L4",
                         "--out", "o.json", "--n-iter", "2"])


def test_child_accepts_contract_minimum_n_iter():
    args = spc._parse_args(["--cache-dir", "x", "--k", "16",
                            "--budget-gb", "19.5", "--budget-source", "NVIDIA L4",
                            "--out", "o.json", "--n-iter", "3"])
    assert args.n_iter == 3
    assert args.s4_scenario == "A"


def test_child_rejects_over_budget_cli_flag():
    """sec 1.4 B3: --budget-gb may never exceed the hardware contract."""
    with pytest.raises(AssertionError):
        spc.main(["--cache-dir", "x", "--k", "32", "--budget-gb", "40.0",
                 "--budget-source", "NVIDIA L4", "--out", "o.json"])


def test_child_resident_bytes_cpu_only_tensors_contribute_zero():
    class Fake:
        pass

    obj = Fake()
    obj.a = torch.zeros(1000)          # CPU tensor -- is_cuda False
    obj.b = [torch.zeros(10), torch.zeros(20)]
    obj.c = "not a tensor"
    assert spc._resident_bytes(obj) == 0


# ---------------------------------------------------------------------------
# child: incremental JSON write helper
# ---------------------------------------------------------------------------

def test_child_write_is_atomic_and_leaves_no_tmp_file(tmp_path):
    out = str(tmp_path / "r.json")
    spc._write(out, {"a": 1})
    spc._write(out, {"a": 2, "b": [1, 2, 3]})
    assert not os.path.exists(out + ".tmp")
    with open(out) as f:
        assert json.load(f) == {"a": 2, "b": [1, 2, 3]}


# ---------------------------------------------------------------------------
# parent: preflight against the CURRENT real GPU (no workload launched --
# these are plain nvidia-smi inspection calls, not CUDA work).
# ---------------------------------------------------------------------------

def test_gpu_name_matches_l4_budget_key_or_is_skippable():
    name = sp.gpu_name()
    if name is None:
        pytest.skip("nvidia-smi unavailable on this host")
    assert name in result_gate.HW_BUDGET_GB or name == "NVIDIA L4"


def test_preflight_detects_current_gpu_contention_if_present():
    """This host's GPU is expected to be busy with a concurrent T8 GPU run
    while this task executes -- if so, `_preflight` must correctly flag
    contamination (an external compute-apps PID or non-zero baseline)
    rather than silently reporting `ok=True`. Skips (does not fail) if the
    GPU happens to be idle when this test runs, since that's a precondition
    this test cannot control."""
    baseline = sp.device_used_gb()
    pids = sp.compute_app_pids()
    if baseline is None:
        pytest.skip("nvidia-smi unavailable on this host")
    if baseline < sp.CONTAMINATION_BASELINE_GIB and not pids:
        pytest.skip("GPU appears idle right now -- contention precondition not met")
    ok, status, details = sp._preflight(19.5, "NVIDIA L4", "results/m4/bench/tiled_cache/3x3_n2")
    assert ok is False
    assert status == "contaminated"


# ---------------------------------------------------------------------------
# parent: state machine (fully mocked -- no subprocess, no nvidia-smi)
# ---------------------------------------------------------------------------

def _args(tmp_path, **overrides):
    ap = argparse.Namespace(
        cache_dir=str(tmp_path / "cache"), k=32, rtype="grid", seed=0,
        n_iter=4, lr=1.0, tau=None, s4_scenario="A",
        budget_gb=19.5, budget_source="NVIDIA L4", max_attempts=3,
        child_timeout_s=1800.0, python=sp.DEFAULT_PYTHON,
        count_freeze=str(tmp_path / "count_freeze_missing.json"),
        out=str(tmp_path / "out.json"),
    )
    for k, v in overrides.items():
        setattr(ap, k, v)
    return ap


def _fake_ok_preflight(monkeypatch, gpu="NVIDIA L4"):
    monkeypatch.setattr(sp, "_preflight", lambda *a, **kw: (
        True, None, {"gpu_name": gpu, "baseline_device_used_gb": 0.1,
                     "exclusivity_evidence": "compute_apps_pid_confirmed", "cache_meta": {}}))


def _attempt(ws, postflight_ok=True, exit_code=None, peak_gb=5.0, resident_lb=4.0):
    if exit_code is None:
        exit_code = {"completed": 0, "oom": 3, "crashed": 1}[ws]
    return {
        "attempt": 0, "exit_code": exit_code, "wall_s": 1.0,
        "workload_status": ws, "postflight_ok": postflight_ok,
        "child_stderr_tail": "", "child_exit_code": exit_code,
        "device_used_peak_gb_nvsmi": peak_gb, "device_baseline_gb_nvsmi": 0.1,
        "partial": {"measured_peak_gb": peak_gb, "resident_lower_bound_gb": resident_lb,
                    "n_interleaved_iters": 4, "case": "toy", "input_sha256": "deadbeef"},
    }


def test_state_machine_immediate_feasible(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("completed", peak_gb=5.0)]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path))
    assert record["experiment_status"] == "ok"
    assert record["workload_status"] == "completed"
    assert record["feasibility_verdict"] == "feasible_l4_contract"
    assert record["retry_count"] == 0
    ok, reasons = result_gate.check(record, schema=result_gate.SPIKE_SCHEMA_FIELDS)
    assert ok, reasons


def test_state_machine_immediate_infeasible_over_budget(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("completed", peak_gb=25.0)]   # > 19.5 budget
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path))
    assert record["feasibility_verdict"] == "infeasible_l4_contract"
    assert record["experiment_status"] == "ok"


def test_state_machine_oom_reproduced_twice_is_infeasible(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("oom"), _attempt("oom")]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path))
    assert record["feasibility_verdict"] == "infeasible_l4_contract"
    assert record["oom_repeats"] == 2
    assert record["retry_count"] == 1


def test_state_machine_nonconsecutive_oom_pair_still_counts_as_reproduced(tmp_path, monkeypatch):
    """"重現兩次" (reproduced twice under the same contract) does not
    require the two OOMs to be consecutive attempts -- an intervening
    crashed/invalid attempt does not reset `oom_repeats`."""
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("oom"), _attempt("crashed"), _attempt("oom")]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    assert record["feasibility_verdict"] == "infeasible_l4_contract"
    assert record["oom_repeats"] == 2
    assert record["retry_count"] == 2


def test_state_machine_single_unreproduced_oom_exhausts_to_blocked_external(tmp_path, monkeypatch):
    """A single OOM (never reproduced, budget exhausted by other invalid
    attempts) must end as invalid_measurement/blocked_external -- NOT
    silently promoted to infeasible_l4_contract off one sample."""
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("oom"), _attempt("crashed"), _attempt("crashed")]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    assert record["feasibility_verdict"] == "invalid_measurement"
    assert record["closed"] == "blocked_external"
    assert record["oom_repeats"] == 1
    assert record["retry_count"] == 2


def test_state_machine_oom_then_completed_is_contradiction_contaminated(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("oom"), _attempt("completed")]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=2))
    # attempt 2 is a contradiction (oom seen, then completed) -> contaminated
    assert record["experiment_status"] == "contaminated"
    assert record["feasibility_verdict"] == "invalid_measurement"
    assert record["closed"] == "blocked_external"


def test_state_machine_crashed_is_instrumentation_error(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("crashed"), _attempt("crashed"), _attempt("crashed")]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    assert record["experiment_status"] == "instrumentation_error"
    assert record["feasibility_verdict"] == "invalid_measurement"
    assert record["closed"] == "blocked_external"
    assert record["retry_count"] == 2


def test_state_machine_postflight_failure_is_contaminated_priority_over_completed(tmp_path, monkeypatch):
    """Contamination takes priority over the OOM/completed verdict even
    when the child itself reports a clean exit."""
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("completed", postflight_ok=False),
            _attempt("completed", postflight_ok=True)]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    assert record["experiment_status"] == "ok"          # 2nd (clean) attempt wins
    assert record["feasibility_verdict"] == "feasible_l4_contract"
    assert record["retry_count"] == 1


def test_state_machine_resident_lower_bound_over_device_gib_is_infeasible(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("oom", resident_lb=25.0)]   # analytic lower bound alone > 21.7 GiB
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    assert record["feasibility_verdict"] == "infeasible_l4_contract"


def test_state_machine_preflight_failure_retries_and_can_recover(tmp_path, monkeypatch):
    seq = [(False, "contaminated", {"gpu_name": "NVIDIA L4", "baseline_device_used_gb": 2.0}),
          (True, None, {"gpu_name": "NVIDIA L4", "baseline_device_used_gb": 0.1, "cache_meta": {}})]
    monkeypatch.setattr(sp, "_preflight", lambda *a, **kw: seq.pop(0))
    calls = [_attempt("completed")]
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    assert record["feasibility_verdict"] == "feasible_l4_contract"
    assert record["retry_count"] == 1


def test_assembled_record_always_passes_schema_even_when_blocked(tmp_path, monkeypatch):
    _fake_ok_preflight(monkeypatch)
    calls = [_attempt("crashed")] * 3
    monkeypatch.setattr(sp, "_run_one_attempt", lambda *a, **kw: calls.pop(0))
    record = sp.run(_args(tmp_path, max_attempts=3))
    missing = [f for f in result_gate.SPIKE_SCHEMA_FIELDS if f not in record]
    assert not missing
    for f in result_gate._PROVENANCE_FIELDS:
        assert record.get(f) not in (None, "", {}), f"{f} missing/empty in a blocked record"
