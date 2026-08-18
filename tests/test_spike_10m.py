import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.diagnostics.spike_10m import run

# Tiny scale (not the 10M/30M spike sizes) so this runs fast and unmarked
# (matches tests/test_io_term_chunked.py's precedent of small synthetic
# cases run without @pytest.mark.slow).
_SMALL = dict(n_nodes=2_000, n_nets=2_400, n_pins=8_000, K=8)


def test_budget_gb_default_is_8_and_recorded_in_output():
    """M4 design draft sec 1.4 B3: --budget-gb defaults to 8.0, the M2 10M/
    8GB contract (spec: 'M3 自己的 G6 判準不變, 仍是 10M / 8GB'), and the
    threshold actually used is now part of the JSON output."""
    r = run(**_SMALL)
    assert r["budget_gb"] == 8.0
    assert r["ok"] == (r["peak_gb"] <= 8.0)


def test_budget_gb_is_parametrizable():
    """A 30M-scale peak (e.g. 12.32 GB) must not be misjudged as a failure
    against the fixed M2 10M/8GB threshold -- budget_gb makes the pass/fail
    line a parameter, not a hardcoded constant."""
    r_tight = run(**_SMALL, budget_gb=1e-9)
    assert r_tight["budget_gb"] == 1e-9
    assert r_tight["ok"] is False           # any nonzero peak fails a ~0 budget

    r_loose = run(**_SMALL, budget_gb=1e9)
    assert r_loose["budget_gb"] == 1e9
    assert r_loose["ok"] is True
