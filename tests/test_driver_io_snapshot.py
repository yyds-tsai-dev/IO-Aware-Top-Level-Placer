"""M3 design draft v3.1 T0-a: driver snapshot instrumentation for P0b.

--snapshot-iters / --snapshot-dir (design v3.1 sec 3.4.1(a), sec 8 T0-a row):
write a trajectory-snapshot npz at each requested iteration, from *inside*
the same N=every evaluator-gated branch that already produces the
trajectory log. Acceptance criteria (sec 8 T0-a):
  1. not requesting --snapshot-iters leaves the run bit-identical to before
     T0-a existed.
  2. gradient consistency: g_wl_density + lambda_io*g_io (computed at a
     snapshot's stored tau) matches the driver's full-objective gradient at
     that same iteration, relative error < 1e-6.
"""
import os
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement_io import run_io

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")


@pytest.mark.slow
def test_snapshot_iters_does_not_change_the_final_placement(tmp_path):
    """T0-a acceptance criterion 1: adding --snapshot-iters must be a pure
    side channel -- the final placement npz has to match a run without the
    flag, bit-for-bit (shortened iteration budget via `every`/rho_max is
    fine per the task; determinism comes from dp_seed+deterministic=1, the
    same guarantee test_driver_io.py's existing bit-exact tests rely on)."""
    baseline = run_io(SIMPLE, k=4, rtype="grid", seed=0,
                      out_json=str(tmp_path / "baseline.json"),
                      rho_max=0.05, every=10, dp_seed=1000, deterministic=1)
    snapped = run_io(SIMPLE, k=4, rtype="grid", seed=0,
                     out_json=str(tmp_path / "snapped.json"),
                     rho_max=0.05, every=10, dp_seed=1000, deterministic=1,
                     snapshot_iters=[20, 40], snapshot_dir=str(tmp_path / "snaps"))
    xa = np.load(str(tmp_path / "baseline.json") + ".npz")["node_x"]
    xb = np.load(str(tmp_path / "snapped.json") + ".npz")["node_x"]
    ya = np.load(str(tmp_path / "baseline.json") + ".npz")["node_y"]
    yb = np.load(str(tmp_path / "snapped.json") + ".npz")["node_y"]
    assert np.array_equal(xa, xb)
    assert np.array_equal(ya, yb)
    # everything else in the JSON result (io_count, trajectory, ...) must
    # also be untouched -- the snapshot side channel must not perturb any
    # measurement, not just the final positions.
    assert baseline["io_count"] == snapped["io_count"]
    assert baseline["trajectory"] == snapped["trajectory"]


@pytest.mark.slow
def test_snapshot_files_have_expected_keys_and_shapes(tmp_path):
    snap_dir = str(tmp_path / "snaps")
    res = run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=str(tmp_path / "io.json"),
                rho_max=0.05, every=10, dp_seed=1000, deterministic=1,
                snapshot_iters="20,40", snapshot_dir=snap_dir)
    for it in (20, 40):
        p = os.path.join(snap_dir, f"it{it:04d}.npz")
        assert os.path.exists(p), p
        d = np.load(p)
        assert int(d["iteration"]) == it
        for key in ("overflow", "tau", "tau_rel", "gamma", "density_weight",
                    "ratio_ema", "lambda_io"):
            assert key in d, key
            assert np.asarray(d[key]).shape == ()
        assert d["node_x"].shape == d["node_y"].shape
        n_all = d["node_x"].shape[0]
        assert d["g_wl_density"].shape == (2 * n_all,)
        assert d["g_wl_density"].dtype == np.float32
        # tau_rel is tau / L_R (sec 3.4.1(a)): consistent with the run's
        # own tau_hi/tau_lo/of_on/of_end schedule.
        from ioplace.schedules import tau_rel_from_overflow
        expected = tau_rel_from_overflow(float(d["overflow"]), res["tau_hi"],
                                         res["tau_lo"], res["of_on"], res["of_end"])
        assert float(d["tau_rel"]) == pytest.approx(expected, rel=1e-9)


@pytest.mark.slow
def test_snapshot_iters_unreachable_raises(tmp_path):
    with pytest.raises(ValueError, match="unreachable"):
        run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=str(tmp_path / "io.json"),
              rho_max=0.05, every=10, dp_seed=1000, deterministic=1,
              snapshot_iters=[25], snapshot_dir=str(tmp_path / "snaps"))


def test_snapshot_iters_without_dir_raises():
    with pytest.raises(ValueError, match="snapshot_dir"):
        run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json="/tmp/unused.json",
              rho_max=0.05, every=10, dp_seed=1000, deterministic=1,
              snapshot_iters=[10])


@pytest.mark.slow
def test_gradient_consistency_g_wl_density_plus_lambda_io_g_io(tmp_path):
    """T0-a acceptance criterion 2: g_wl_density + lambda_io*g_io must equal
    the driver's actual full-objective gradient (wirelength + density_weight
    * density + lambda_io*L_IO) at that snapshot's exact position and tau,
    to relative L1 error < 1e-6.

    The reference is computed via placer.model.obj_fn(...).backward() on an
    independent clone -- i.e. the *raw* (unpreconditioned) autograd
    gradient, matching how g_wl_density itself is spec'd (design v3.1 sec 8
    T0-a row: "g_wl_density = grad(wirelength_op) + density_weight *
    grad(density_op)", no precondition_op in that formula) and how P0b
    actually consumes it (sec 3.4.2's M3 family builds a raw normalized
    direction from g_full = g_wl_density + lambda_io*C, never feeding it
    through DP's own optimizer preconditioner). placer.model.obj_and_grad_fn
    additionally applies precondition_op (elementwise divide by a
    density/pin-weight-derived, gradient-independent scalar) -- comparing
    against *that* would require replicating its update_mask/fix_nodes_mask/
    density_weight state and would no longer be testing this driver's own
    plumbing, so it is intentionally not what this test compares against."""
    checks = []

    def hook(iteration, pos, placer, io_term, state, g_wl_density):
        p_io = pos.detach().clone().requires_grad_(True)
        L_io = io_term(p_io, state.tau, lambda_io=1.0, lambda_margin=0.0)
        L_io.backward()
        g_io = p_io.grad.detach().clone()
        reconstructed = g_wl_density + state.lambda_io * g_io

        p_ref = pos.detach().clone().requires_grad_(True)
        obj = placer.model.obj_fn(p_ref)
        obj.backward()
        full_grad = p_ref.grad.detach().clone()
        checks.append((iteration, reconstructed, full_grad))

    run_io(SIMPLE, k=4, rtype="grid", seed=0, out_json=str(tmp_path / "io.json"),
          rho_max=0.05, every=10, dp_seed=1000, deterministic=1,
          snapshot_iters=[20, 40], snapshot_dir=str(tmp_path / "snaps"),
          snapshot_grad_check_cb=hook)

    assert len(checks) == 2
    for iteration, reconstructed, full_grad in checks:
        denom = float(full_grad.abs().sum().clamp_min(1e-12))
        rel_err = float((reconstructed - full_grad).abs().sum()) / denom
        assert rel_err < 1e-6, f"iteration {iteration}: rel_err={rel_err}"
