import importlib
import os
import subprocess
import sys
import pytest

from ioplace.paths import REPO_ROOT


def test_require_gr_in_loop_rejects_an_unset_env(monkeypatch):
    from ioplace.gr_in_loop import GR_IN_LOOP_ENV, require_gr_in_loop
    assert GR_IN_LOOP_ENV == "IOPLACE_ENABLE_GR_IN_LOOP"
    monkeypatch.delenv(GR_IN_LOOP_ENV, raising=False)
    with pytest.raises(RuntimeError, match="IOPLACE_ENABLE_GR_IN_LOOP"):
        require_gr_in_loop()
    monkeypatch.setenv(GR_IN_LOOP_ENV, "1")
    require_gr_in_loop()


def test_the_controller_gate_routes_through_require_gr_in_loop(monkeypatch):
    """P-H Task 8 owns the controller's RuntimeError and its message
    (tests/test_routing_gp_retirement.py). This only pins that the controller
    delegates to gr_in_loop instead of re-reading os.environ, so the repo has
    exactly one gate with exactly one message."""
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    module = importlib.import_module("ioplace.ops.routing_gp_controller")
    calls = []
    monkeypatch.setattr(module, "require_gr_in_loop", lambda: calls.append(1))
    module.RoutingGPController(object(), object())
    assert calls == [1]


def test_run_route_gp_cli_is_gated(tmp_path):
    script = os.path.join(str(REPO_ROOT), "src/scripts/run_route_gp.py")
    env = {key: value for key, value in os.environ.items()
           if key != "IOPLACE_ENABLE_GR_IN_LOOP"}
    blocked = subprocess.run([sys.executable, script, "--help"],
                             capture_output=True, text=True, env=env)
    assert blocked.returncode != 0
    assert "IOPLACE_ENABLE_GR_IN_LOOP" in blocked.stderr
    env["IOPLACE_ENABLE_GR_IN_LOOP"] = "1"
    allowed = subprocess.run([sys.executable, script, "--help"],
                             capture_output=True, text=True, env=env)
    assert allowed.returncode == 0
