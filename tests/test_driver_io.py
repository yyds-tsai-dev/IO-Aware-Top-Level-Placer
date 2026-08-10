import json, os
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement_io import run_io, RESULT_FIELDS

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")

def test_result_fields_contract_is_declared():
    for f in ("mode", "io_count", "io_gp", "ft_count", "hard_lambda_sum", "hpwl",
              "lg_loss", "rho_max", "tau_hi", "tau_lo", "alpha_io", "w_mode", "d_max",
              "rho_margin", "lambda_io_final", "spearman_rho", "num_callbacks",
              "num_refreshes", "backtrack_median", "observer_mode", "dp_seed", "det",
              "trajectory"):
        assert f in RESULT_FIELDS

@pytest.mark.slow
def test_run_io_on_simple_writes_full_schema(tmp_path):
    out = str(tmp_path / "io.json")
    res = run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=out,
                 rho_max=0.05, every=10, check_invariant=True)
    assert os.path.exists(out) and os.path.exists(out + ".npz")
    on_disk = json.load(open(out))
    for f in RESULT_FIELDS:
        assert f in on_disk, f
    assert res["mode"] == "io"
    assert res["io_gp"] >= 0 and res["io_count"] >= 0
    assert res["lg_loss"] == res["io_count"] - res["io_gp"]
    assert res["num_callbacks"] > 0
    assert res["num_refreshes"] >= 1          # activation at least
    assert len(res["trajectory"]) == res["num_callbacks"]

@pytest.mark.slow
def test_dp_seed_changes_the_placement(tmp_path):
    a = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "a.json"), rho_max=0.0, dp_seed=1000)
    b = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "b.json"), rho_max=0.0, dp_seed=2000)
    xa = np.load(str(tmp_path / "a.json") + ".npz")["node_x"]
    xb = np.load(str(tmp_path / "b.json") + ".npz")["node_x"]
    assert not np.array_equal(xa, xb)

@pytest.mark.slow
def test_rho_zero_is_observer_mode_and_bit_identical_to_flat(tmp_path):
    """rho_max=0 and rho_margin=0 must leave the objective completely untouched
    (no attach_terms, no secant refresh) -> bit-identical to run_flat, but with the
    io_gp / lg_loss / hard_lambda_sum columns that T6's flat baseline needs."""
    from ioplace.drivers.run_placement import run_flat
    a = run_flat(SIMPLE, 4, "grid", 0, str(tmp_path / "flat.json"),
                 dp_seed=1000, deterministic=1)
    b = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "io0.json"), rho_max=0.0,
               dp_seed=1000, deterministic=1)
    xa = np.load(str(tmp_path / "flat.json") + ".npz")["node_x"]
    xb = np.load(str(tmp_path / "io0.json") + ".npz")["node_x"]
    assert np.array_equal(xa, xb)
    assert b["observer_mode"] is True and b["num_refreshes"] == 0
    assert b["io_count"] == a["io_count"] and b["io_gp"] >= 0

@pytest.mark.slow
def test_optimizer_lock_is_enforced(tmp_path):
    import Params
    from ioplace.drivers.run_placement import _load_dreamplace
    from ioplace.dp_hook import assert_optimizer_lock
    p, db = _load_dreamplace(SIMPLE)
    # before initialize, use_bb is the unresolved string 'auto' -- the lock
    # must refuse that state with a clear AssertionError (not a ValueError)
    with pytest.raises(AssertionError, match="unresolved"):
        assert_optimizer_lock(p)
    db.initialize(p)
    assert_optimizer_lock(p)                       # locked config passes
    p.global_place_stages[0]["Lsub_iteration"] = 2
    with pytest.raises(AssertionError):
        assert_optimizer_lock(p)
