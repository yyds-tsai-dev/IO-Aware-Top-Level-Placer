import json
import pytest

from ioplace.bench import spike_30m as sp
from ioplace.bench.spike_30m import _preflight
from ioplace.bench.spike_30m_child import _resident_bytes


def _meta(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"n_nodes": 3, "manifest_output_sha256": {}}))
    return str(tmp_path)


def test_shared_preflight_allows_external_pids_and_records_margin(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "gpu_name", lambda: "NVIDIA H100 NVL")
    monkeypatch.setattr(sp, "device_used_gb", lambda: 20.0)
    monkeypatch.setattr(sp, "compute_app_pids", lambda: [1234, 5678])
    monkeypatch.setattr(sp, "MEM_AVAILABLE_MIN_GB", 0.0)
    monkeypatch.setattr(sp, "_nvsmi", lambda *a, **k:
                        "GPU-uuid, NVIDIA H100, 95830, 1000, 92000, 20\n")
    ok, status, d = _preflight(84.0, "NVIDIA H100 NVL", _meta(tmp_path), "shared", 4.0)
    assert ok and status is None
    assert d["shared_user_authorized"] is True
    assert d["shared_free_required_gb"] == 88.0
    assert d["preflight_compute_app_pids"] == [1234, 5678]


def test_shared_preflight_rejects_insufficient_free(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "gpu_name", lambda: "NVIDIA H100 NVL")
    monkeypatch.setattr(sp, "device_used_gb", lambda: 0.0)
    monkeypatch.setattr(sp, "compute_app_pids", lambda: [])
    monkeypatch.setattr(sp, "MEM_AVAILABLE_MIN_GB", 0.0)
    monkeypatch.setattr(sp, "_nvsmi", lambda *a, **k:
                        "GPU-uuid, NVIDIA H100, 95830, 90000, 5000, 20\n")
    ok, status, d = _preflight(84.0, "NVIDIA H100 NVL", _meta(tmp_path), "shared", 4.0)
    assert not ok and status == "input_error"
    assert d["shared_free_gb"] < d["shared_free_required_gb"]


def test_exclusive_preflight_rejects_external_pids(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "gpu_name", lambda: "NVIDIA H100 NVL")
    monkeypatch.setattr(sp, "device_used_gb", lambda: 0.0)
    monkeypatch.setattr(sp, "compute_app_pids", lambda: [4321])
    monkeypatch.setattr(sp, "MEM_AVAILABLE_MIN_GB", 0.0)
    ok, status, _ = _preflight(84.0, "NVIDIA H100 NVL", _meta(tmp_path), "exclusive", 4.0)
    assert not ok and status == "contaminated"


@pytest.mark.parametrize("workload,peak,iterations,expected", [
    ("oom", None, 0, "shared_oom_unattributed"),
    ("completed", 83., 4, "completed_within_registered_process_budget"),
    ("completed", 85., 4, "exceeds_registered_process_budget"),
    ("completed", 83., 3, "invalid_measurement"),
    ("completed", None, 4, "invalid_measurement"),
    ("completed", float("nan"), 4, "invalid_measurement"),
])
def test_shared_verdict(workload, peak, iterations, expected):
    assert sp.shared_verdict("ok", workload, peak, 84., iterations, 4) == expected


def test_shared_snapshot_failure_is_ineligible(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "gpu_name", lambda: "NVIDIA H100 NVL")
    monkeypatch.setattr(sp, "device_used_gb", lambda: 0.)
    monkeypatch.setattr(sp, "compute_app_pids", lambda: [])
    monkeypatch.setattr(sp, "_nvsmi", lambda *args: None)
    ok, status, detail = _preflight(84., "NVIDIA H100 NVL", _meta(tmp_path), "shared")
    assert not ok and status == "input_error"
    assert "nvidia_smi_error" in detail


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_resident_registered_buffer_and_view_dedup():
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.zeros(1024, device="cuda"))
            self.view = self.b[:100]
    m = M()
    assert _resident_bytes(m) == m.b.untyped_storage().nbytes()


def test_movable_step_preserves_outside_die_fixed_nodes():
    import torch
    from ioplace.bench.spike_30m_child import movable_step
    pos = torch.tensor([5., 7., -100., 120., 5., 7., -200., 250.], requires_grad=True)
    grad = torch.tensor([100., -100., 100., 100., -100., 100., 100., 100.])
    movable_step(pos, grad, 2, 4, (0., 0., 10., 10.), 1.)
    assert torch.equal(pos, torch.tensor([0.,10.,-100.,120.,10.,0.,-200.,250.]))


@pytest.mark.parametrize("visible", ["1", "GPU-selected"])
def test_nvsmi_honors_selected_device(monkeypatch, visible):
    from ioplace.diagnostics.probes_m4 import gpu_exclusivity as gpu
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    commands = []
    def capture(command, **kwargs):
        commands.append(command)
        return b"1\n"
    monkeypatch.setattr(gpu.subprocess, "check_output", capture)
    assert gpu._nvsmi("memory.used") == "1\n"
    assert commands[0][-2:] == ["--id", visible]
