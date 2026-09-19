"""Resource limits for the recovery runner's fresh detailed-route children."""
import importlib.util
from pathlib import Path
import threading

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROAD_BIN", "/unused/openroad")
    path = Path(__file__).parents[1] / "results/route_gp_20260914/recover_large_20260915.py"
    spec = importlib.util.spec_from_file_location("recovery_parallelism_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STATUS = tmp_path
    return module


def test_router_threads_reserve_cpus_for_other_stages(runner, monkeypatch):
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _: set(range(8)))
    assert runner.routing_threads() == 4
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _: {0})
    assert runner.routing_threads() == 1
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _: set(range(48)))
    assert runner.routing_threads() == 24


def test_detailed_routes_exclude_each_other_and_release_on_failure(runner):
    entered = threading.Event()
    attempted = threading.Event()

    def other_route():
        attempted.set()
        with runner.detailed_route_slot():
            entered.set()

    with pytest.raises(RuntimeError, match="router failed"):
        with runner.detailed_route_slot():
            worker = threading.Thread(target=other_route)
            worker.start()
            assert attempted.wait(2)
            assert not entered.wait(.1)
            raise RuntimeError("router failed")
    worker.join(2)
    assert not worker.is_alive()
    assert entered.is_set()
