"""Exceptions must release live samplers and per-run DREAMPlace hooks."""
import json
import os
from pathlib import Path

import pytest
import torch

from ioplace import profile, profile_lifetime
from ioplace.drivers import run_placement_io as driver


def _capture_samplers(monkeypatch, fail_stop=False):
    instances = []
    class Sampler(profile.DeviceMemSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(interval_s=.01)
            instances.append(self)
        def stop(self):
            super().stop()
            if fail_stop:
                raise RuntimeError("injected cleanup error")
            return self
    monkeypatch.setattr(driver, "DeviceMemSampler", Sampler)
    monkeypatch.setattr(profile, "DeviceMemSampler", Sampler)
    return instances


@pytest.mark.parametrize("lifetime", [False, True])
@pytest.mark.parametrize("fail_stop", [False, True])
def test_read_failure_releases_all_threads_and_preserves_original_error(tmp_path, monkeypatch, lifetime, fail_stop):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instances = _capture_samplers(monkeypatch, fail_stop)
    def fail_read(*args):
        assert len(instances) == (2 if lifetime else 1)
        assert all(item._thread.is_alive() for item in instances)
        raise ValueError("injected read error")
    monkeypatch.setattr(driver, "_load_dreamplace", fail_read)
    for _ in range(2):
        instances.clear()
        with pytest.raises(ValueError, match="injected read error") as error:
            driver.run_io("unused.json", 4, "grid", 0, str(tmp_path / "out.json"),
                          lifetime_out=str(tmp_path / "life.json") if lifetime else None)
        assert all(item._thread is None and item._stop_evt.is_set() for item in instances)
        assert not (tmp_path / "out.json").exists()
        if fail_stop:
            assert "injected cleanup error" in " ".join(error.value.__notes__)


@pytest.mark.slow
def test_failed_snapshot_callback_restores_real_hooks_and_threads(tmp_path, monkeypatch):
    from ioplace.dreamplace_env import setup_dreamplace
    root = Path(setup_dreamplace())
    import NonLinearPlace
    instances = _capture_samplers(monkeypatch)
    load = driver._load_dreamplace
    loaded, constructed = [], []
    def capture_load(config):
        result = load(config)
        loaded.append(result[0])
        return result
    monkeypatch.setattr(driver, "_load_dreamplace", capture_load)
    constructor = NonLinearPlace.NonLinearPlace.__init__
    def capture_placer(value, *args, **kwargs):
        constructor(value, *args, **kwargs)
        constructed.append((value, value.op_collections.legalize_op,
                            getattr(value, "iteration_callback", None)))
    monkeypatch.setattr(NonLinearPlace.NonLinearPlace, "__init__", capture_placer)
    config = json.loads((root / "install/test/simple.json").read_text())
    config.update(num_threads=4, plot_flag=0, num_bins_x=16, num_bins_y=16,
        global_place_stages=[dict(num_bins_x=16, num_bins_y=16, iteration=40,
            learning_rate=.01, wirelength="weighted_average", optimizer="nesterov")])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    def failure(*args):
        raise RuntimeError("injected snapshot callback error")
    # Fix round 1 (controller note (j) follow-up): a non-legacy policy also
    # owns a NormTraceWriter, registered on the same cleanup ExitStack as the
    # samplers/hooks above -- it must be closed (and its file left readable)
    # even when the run fails mid-callback, same as everything else here.
    norm_trace_path = str(tmp_path / "norm_trace.jsonl")
    with pytest.raises(RuntimeError, match="injected snapshot callback error"):
        driver.run_io(str(path), 4, "grid", 0, str(tmp_path / "result.json"),
            rho_max=.1, every=5, callback_order="atomic", check_invariant=True,
            no_diag=True, snapshot_iters=[5], snapshot_dir=str(tmp_path / "snapshots"),
            snapshot_grad_check_cb=failure, lifetime_out=str(tmp_path / "lifetime.json"),
            norm_policy="grandplan", norm_probe_every=5, norm_trace=norm_trace_path)
    assert len(instances) == 2 and all(item._thread is None for item in instances)
    assert all(not hasattr(params, "_extra_obj_terms") for params in loaded)
    assert constructed
    for placer, legalize, callback in constructed:
        assert placer.op_collections.legalize_op is legalize
        assert getattr(placer, "iteration_callback", None) is callback
        assert not hasattr(placer.optimizer.obj_and_grad_fn, "__wrapped__")
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "lifetime.json").exists()
    assert os.path.exists(norm_trace_path)
    from ioplace.norm_trace import read_norm_trace
    read_norm_trace(norm_trace_path)  # must not raise: writer was closed/flushed
