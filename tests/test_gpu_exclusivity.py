"""M4 T1b (d) tests for `ioplace.diagnostics.probes_m4.gpu_exclusivity` --
design draft `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`
sec 1.4 D9 / T1b row (d).

CPU-only: every test here mocks `device_used_gb`/`compute_app_pids`
directly (module-level monkeypatch), never calling the real `nvidia-smi`
binary or touching CUDA -- consistent with this task's "no GPU work" scope
(GPU busy with a concurrent T8 run).
"""
import pytest

from ioplace.diagnostics.probes_m4 import gpu_exclusivity as gx


# ---------------------------------------------------------------------------
# preflight_exclusivity
# ---------------------------------------------------------------------------

def test_preflight_ok_when_baseline_low_and_no_other_pids(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.1)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])
    result = gx.preflight_exclusivity()
    assert result["ok"] is True
    assert result["reasons"] == []
    assert result["exclusivity_evidence"] == "compute_apps_pid_confirmed"


def test_preflight_contaminated_when_baseline_at_or_above_threshold(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: gx.CONTAMINATION_BASELINE_GIB)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])
    result = gx.preflight_exclusivity()
    assert result["ok"] is False
    assert any("baseline_device_used_gb" in r for r in result["reasons"])


def test_preflight_contaminated_when_other_pid_present(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.05)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [(4242, 512.0)])
    result = gx.preflight_exclusivity(self_test_pid=1000)
    assert result["ok"] is False
    assert any("other compute-apps PIDs" in r for r in result["reasons"])


def test_preflight_contaminated_when_baseline_query_fails(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: None)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])
    result = gx.preflight_exclusivity()
    assert result["ok"] is False
    assert any("unavailable" in r for r in result["reasons"])


def test_preflight_evidence_downgrades_to_baseline_only_when_query_unavailable(monkeypatch):
    """§6.1's documented degrade path: nvidia-smi --query-compute-apps
    itself failing (returns None, distinct from an empty-but-successful
    list) must never be silently treated as 'confirmed empty'."""
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.1)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: None)
    result = gx.preflight_exclusivity()
    assert result["exclusivity_evidence"] == "baseline_only"
    # an unavailable query is not itself contamination -- only baseline/pid
    # findings are -- so `ok` can still be True here.
    assert result["ok"] is True


def test_preflight_self_test_confirms_own_pid_upgrades_evidence(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.1)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [(1234, 800.0)])
    result = gx.preflight_exclusivity(self_test_pid=1234)
    assert result["self_test_own_pid_visible"] is True
    assert result["exclusivity_evidence"] == "compute_apps_pid_confirmed"
    assert result["ok"] is True   # the only PID present is our own


def test_preflight_self_test_cannot_see_own_pid_downgrades_evidence(monkeypatch):
    """This host has been observed to return an empty compute-apps list
    even while our own workload is running (design draft §6.1's recorded
    note) -- if the self-test can't see our OWN known-active PID, an
    empty/other-free reading is not trustworthy evidence of exclusivity
    either, so evidence must downgrade even though `ok` may still pass on
    the baseline/other-PID checks alone."""
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.1)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])   # empty, but we know 1234 is active
    result = gx.preflight_exclusivity(self_test_pid=1234)
    assert result["self_test_own_pid_visible"] is False
    assert result["exclusivity_evidence"] == "baseline_only"


def test_self_test_own_pid_visible_returns_none_when_query_fails(monkeypatch):
    monkeypatch.setattr(gx, "compute_app_pids", lambda: None)
    assert gx.self_test_own_pid_visible(1234) is None


def test_self_test_own_pid_visible_true_false(monkeypatch):
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [(1234, 100.0), (5, 1.0)])
    assert gx.self_test_own_pid_visible(1234) is True
    assert gx.self_test_own_pid_visible(999) is False


# ---------------------------------------------------------------------------
# postflight_exclusivity
# ---------------------------------------------------------------------------

def test_postflight_ok_within_tolerance(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.15)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])
    result = gx.postflight_exclusivity(baseline_before_gb=0.1)
    assert result["ok"] is True


def test_postflight_fails_when_delta_exceeds_tolerance(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 5.0)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])
    result = gx.postflight_exclusivity(baseline_before_gb=0.1)
    assert result["ok"] is False
    assert any("exceeds" in r for r in result["reasons"])


def test_postflight_fails_when_lingering_pid_present(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.1)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [(999, 10.0)])
    result = gx.postflight_exclusivity(baseline_before_gb=0.1)
    assert result["ok"] is False
    assert any("lingering" in r for r in result["reasons"])


def test_postflight_fails_when_query_unavailable(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: None)
    monkeypatch.setattr(gx, "compute_app_pids", lambda: [])
    result = gx.postflight_exclusivity(baseline_before_gb=0.1)
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# _nvsmi/compute_app_pids parsing (no monkeypatch -- exercises the real
# subprocess-output parser against synthetic text, not a live nvidia-smi
# call)
# ---------------------------------------------------------------------------

def test_compute_app_pids_parses_csv_lines(monkeypatch):
    monkeypatch.setattr(gx, "_nvsmi", lambda *a, **kw: "1234, 512 MiB\n5678, 1024 MiB\n")
    assert gx.compute_app_pids() == [(1234, 512.0), (5678, 1024.0)]


def test_compute_app_pids_empty_text_is_empty_list(monkeypatch):
    monkeypatch.setattr(gx, "_nvsmi", lambda *a, **kw: "\n")
    assert gx.compute_app_pids() == []


def test_compute_app_pids_none_when_nvsmi_unavailable(monkeypatch):
    monkeypatch.setattr(gx, "_nvsmi", lambda *a, **kw: None)
    assert gx.compute_app_pids() is None


def test_device_used_gb_converts_mib_to_gib(monkeypatch):
    monkeypatch.setattr(gx, "_nvsmi", lambda *a, **kw: "1024\n")
    assert gx.device_used_gb() == pytest.approx(1.0)


def test_device_used_gb_none_when_nvsmi_unavailable(monkeypatch):
    monkeypatch.setattr(gx, "_nvsmi", lambda *a, **kw: None)
    assert gx.device_used_gb() is None


def test_gpu_name_strips_header(monkeypatch):
    monkeypatch.setattr(gx, "_nvsmi", lambda *a, **kw: "NVIDIA L4\n")
    assert gx.gpu_name() == "NVIDIA L4"


# ---------------------------------------------------------------------------
# NvsmiPoller (background thread, but sampling a monkeypatched device_used_gb
# -- no real nvidia-smi/CUDA involved)
# ---------------------------------------------------------------------------

def test_nvsmi_poller_tracks_hwm_across_manual_samples(monkeypatch):
    values = iter([0.1, 5.0, 2.0])
    monkeypatch.setattr(gx, "device_used_gb", lambda: next(values))
    poller = gx.NvsmiPoller(interval_s=None)   # no background thread -- manual _sample_once only
    poller._sample_once()
    poller._sample_once()
    poller._sample_once()
    assert poller.peak_gb == pytest.approx(5.0)


def test_nvsmi_poller_start_stop_is_safe_to_call_twice(monkeypatch):
    monkeypatch.setattr(gx, "device_used_gb", lambda: 0.2)
    poller = gx.NvsmiPoller(interval_s=0.01).start()
    poller.stop()
    poller.stop()   # must not raise or resume sampling
    assert poller.peak_gb == pytest.approx(0.2)
