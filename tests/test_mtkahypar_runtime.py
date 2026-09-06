import os
import subprocess
import sys
from pathlib import Path
import pytest
from ioplace.partition import mtkahypar_runtime as rt

class Fake:
    def __init__(self): self.calls=[]
    def initialize(self, n, print_warnings=False): self.calls.append((n, print_warnings)); return self

@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    monkeypatch.setattr(rt, "_state", dict(module=None, requested=None,
        effective=None, override=None, first_requested=None))

@pytest.mark.parametrize("value", ["nope", "0", "-1", "1.5", ""])
def test_invalid_override(monkeypatch, value):
    monkeypatch.setenv("IOPLACE_MTKAHYPAR_THREADS", value)
    with pytest.raises(ValueError): rt.initialize(Fake(), 4)

def test_first_only_and_default_request_retains_effective(monkeypatch):
    monkeypatch.delenv("IOPLACE_MTKAHYPAR_THREADS", raising=False); f=Fake()
    assert rt.initialize(f, 8) is f; assert rt.initialize(f, 4) is f
    assert f.calls == [(8, False)] and rt.metadata()["effective_native_threads"] == 8
    assert rt.metadata()["requested_threads"] == 4
    assert rt.metadata()["first_requested_threads"] == 8

def test_override_is_effective_and_cannot_change(monkeypatch):
    f=Fake(); monkeypatch.setenv("IOPLACE_MTKAHYPAR_THREADS", "1")
    rt.initialize(f, 8); assert f.calls == [(1, False)]
    monkeypatch.setenv("IOPLACE_MTKAHYPAR_THREADS", "2")
    with pytest.raises(RuntimeError): rt.initialize(f, 4)


def test_native_serial_override_across_both_callers_in_fresh_process():
    pytest.importorskip("mtkahypar")
    code = '''
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
from tests.test_bench_rent import _toy_two_cliques_netlist, _grid_torus_netlist
from ioplace.partition.mtkahypar_runner import partition_netlist
from ioplace.partition.mtkahypar_runtime import metadata
from ioplace.bench.rent import measure_rent
assignment = partition_netlist(_toy_two_cliques_netlist(), 2, threads=8)
assert assignment.shape == (8,) and set(assignment) == {0, 1}
assert np.all(assignment[:4] == assignment[0])
assert np.all(assignment[4:] == assignment[4])
for seed in range(3):
    result = measure_rent(_grid_torus_netlist(16), b_lo=4, b_hi=64,
        seed=seed, backend="auto", threads=4, n_bootstrap=5)
    assert result.backend == "mtkahypar"
    assert abs(result.p - 0.5) <= 0.05, result.p
    assert result.runtime["effective_native_threads"] == 1
    assert result.runtime["requested_threads"] == 4
    assert result.runtime["first_requested_threads"] == 8
print("serial native callers passed")
'''
    env = dict(os.environ, IOPLACE_MTKAHYPAR_THREADS="1", CUDA_VISIBLE_DEVICES="",
               OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "serial native callers passed" in proc.stdout
