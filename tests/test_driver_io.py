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
              "trajectory", "peak_mem_mb_reset_semantics", "diag_every", "no_diag"):
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
def test_peak_mem_io_is_flagged_and_not_the_process_cumulative_hwm(tmp_path):
    """Same B1 fix as run_flat (M4 design draft sec 1.4), applied to run_io:
    reset_peak_memory_stats() at the run's measurement start, necessary but
    not sufficient (see run_placement.py's matching comment) for per-run
    isolation."""
    junk = torch.empty(400_000_000, dtype=torch.uint8, device="cuda")  # ~400MB
    del junk
    torch.cuda.empty_cache()
    inflated_peak_mb = torch.cuda.max_memory_allocated() / 2**20

    res = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "io.json"), rho_max=0.05)
    assert res["peak_mem_mb_reset_semantics"] is True
    assert res["peak_mem_mb"] < inflated_peak_mb

@pytest.mark.slow
def test_diag_every_n_subsamples_and_no_diag_disables(tmp_path, monkeypatch):
    """B2 (M4 design draft sec 1.4 / Codex review): diagnostics() runs a
    full chunked fwd+bwd per non-empty degree bucket -- expensive at scale.
    --diag-every N / --no-diag must gate *only* that call; grad_l1_io/
    grad_l1_wl (needed every callback for the schedule's auto-normalization
    ratio, sec 5.2) are NOT reusable from diagnostics()'s backward passes
    (they run at different tau -- state.tau vs diagnostics's fixed
    0.05*L_R reference -- so they stay independent, computed every time)."""
    from ioplace.ops.io_term import IoTerm
    calls = {"n": 0}
    orig = IoTerm.diagnostics
    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)
    monkeypatch.setattr(IoTerm, "diagnostics", counting)

    res_every1 = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "io1.json"),
                        rho_max=0.05, every=10, diag_every=1)
    calls_every1 = calls["n"]; calls["n"] = 0

    res_every2 = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "io2.json"),
                        rho_max=0.05, every=10, diag_every=2)
    calls_every2 = calls["n"]; calls["n"] = 0

    # diag_every=1 must reproduce the pre-fix always-on behavior exactly.
    assert calls_every1 == res_every1["num_callbacks"]
    assert 0 < calls_every2 < calls_every1
    # `every`'s own callback cadence (num_callbacks/trajectory) is untouched.
    assert res_every2["num_callbacks"] == res_every1["num_callbacks"]

    res_nodiag = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "io3.json"),
                        rho_max=0.05, every=10, no_diag=True)
    assert calls["n"] == 0
    assert all(e["soft_lambda_ref_tau"] == 0.0 and e["frac_soft"] == 0.0
              and e["grad_share"] == [] for e in res_nodiag["trajectory"])
    # grad_l1_io/grad_l1_wl are independent of diag gating -- still present
    # and still real (nonzero) values on every callback.
    assert all("grad_l1_io" in e and "grad_l1_wl" in e for e in res_nodiag["trajectory"])
    assert res_nodiag["diag_every"] == 1 and res_nodiag["no_diag"] is True

@pytest.mark.slow
def test_lifetime_out_is_bit_exact_with_run_without_it(tmp_path):
    """M4 T2b's bit-exactness guarantee: attaching a LifetimeRecorder
    (lifetime_out=<path>) must not perturb io_count/ft_count/hpwl/tree_wl
    -- every profile_lifetime.py call site in run_io is behind `if rec is
    not None:` and only performs GPU-memory *introspection*
    (reset_peak_memory_stats/synchronize/memory_allocated/mem_get_info/
    scan_cuda_tensors' pure-Python object-graph walk), never touching
    `pos`/`pos.grad`, RNG state, or optimizer state -- see run_io's own
    "M4 T2b" comment block for the argument in full."""
    a = run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=str(tmp_path / "a.json"),
              rho_max=0.05, every=10, dp_seed=1000, deterministic=1)
    b = run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=str(tmp_path / "b.json"),
              rho_max=0.05, every=10, dp_seed=1000, deterministic=1,
              lifetime_out=str(tmp_path / "b.lifetime.json"))
    assert os.path.exists(tmp_path / "b.lifetime.json")
    assert a["io_count"] == b["io_count"]
    assert a["ft_count"] == b["ft_count"]
    assert a["hpwl"] == b["hpwl"]
    assert a["tree_wl"] == b["tree_wl"]
    xa = np.load(str(tmp_path / "a.json") + ".npz")["node_x"]
    xb = np.load(str(tmp_path / "b.json") + ".npz")["node_x"]
    assert np.array_equal(xa, xb)


@pytest.mark.slow
def test_lifetime_out_writes_a_result_gate_schema_record(tmp_path):
    from ioplace.bench.result_gate import LIFETIME_SCHEMA_FIELDS, check
    lifetime_path = str(tmp_path / "lt.json")
    run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=str(tmp_path / "c.json"),
          rho_max=0.05, of_on=2.0, every=1, dp_seed=1000, deterministic=1,
          lifetime_out=lifetime_path)
    raw = json.load(open(lifetime_path))
    assert "buffers" in raw and "phases" in raw and "totals" in raw
    for phase in raw["phases"]:
        for f in ("resident_gb", "alloc_start_gb", "unattributed_gb",
                  "transient_gb", "peak_alloc_gb", "peak_reserved_gb",
                  "device_used_peak_gb", "host_rss_hwm_at_phase_end"):
            assert f in phase


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
