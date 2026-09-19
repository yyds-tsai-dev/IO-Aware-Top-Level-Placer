import pytest

torch = pytest.importorskip("torch")

from ioplace.norm import TermNormalizer
from ioplace.ops.ft_callback import publish_atomic
from ioplace.ops.ft_term import FtTerm
from ioplace.schedules import ScheduleState, activation_ramp

# Recorded on 2026-09-19 from the retired path at HEAD (commit 13246d8):
# ScheduleState(rho_max=0.4, n_ramp=20, f_ft_max=0.25, c_lip=1.0) driven with
# L_R=1000, gamma=100 through update_continuous + apply_ft_transaction +
# mark_refreshed. Deterministic, no RNG -- regenerate by running the loop in
# test_retired_path_golden_lambda_trajectory and printing `got`.
# (iteration, overflow, g_wl, g_io, g_ft, merged)
EVENTS = [(0, 0.95, 1000.0, 8.0, 3.0, 9.0),
          (50, 0.80, 1000.0, 10.0, 4.0, 12.0),
          (100, 0.60, 1200.0, 12.0, 5.0, 14.0),
          (150, 0.45, 1400.0, 15.0, 6.0, 17.0),
          (200, 0.30, 1600.0, 18.0, 7.0, 20.0),
          (250, 0.18, 1800.0, 20.0, 8.0, 23.0),
          (300, 0.10, 2000.0, 22.0, 9.0, 25.0)]

# (f_ft, kappa_ft, ratio_ema, Cmax, lambda_io, lambda_ft, obj_version)
GOLDEN = [(0.0, 0.0, 111.11111111111111, 1.0, 0.0, 0.0, 1),
          (0.0, 0.0, 97.22222222222223, 1.0, 0.0, 0.0, 3),
          (0.0, 0.0, 91.46825396825398, 1.0, 15.680272108843544, 0.0, 4),
          (0.0948345618198189, 0.23708640454954724, 86.91059757236229,
           2.185432022747736, 22.348439375750306, 5.298511138890168, 5),
          (0.21366514317815835, 0.5494246538866929, 83.45529878618115,
           3.7471232694334646, 8.605208480652106, 4.727913691105118, 6),
          (0.25, 0.625, 80.85808417569928, 4.125, 4.016786951044713,
           2.510491844402946, 7),
          (0.25, 0.6111111111111112, 80.42904208784964, 4.055555555555555,
           2.621086243264244, 1.6017749264392602, 8)]


def test_retired_path_golden_lambda_trajectory():
    """Locks the numbers the legacy arm must keep reproducing. If this test
    fails, schedules.py changed and the `--norm-policy legacy` guarantee is
    void -- do not update the constants, find the regression."""
    state = ScheduleState(rho_max=0.4, n_ramp=20, f_ft_max=0.25, c_lip=1.0)
    for (iteration, overflow, g_wl, g_io, g_ft, merged), want in zip(EVENTS, GOLDEN):
        state.update_continuous(iteration, overflow, 1000.0, 100.0)
        f_ft = state.apply_ft_transaction(iteration, state.tau / 1000.0, g_wl,
                                          g_io, g_ft, merged, 6.0, 100.0)
        state.mark_refreshed()
        got = (f_ft, state.kappa_ft, state.ratio_ema, state.Cmax, state.lambda_io,
               state.lambda_io * state.kappa_ft, state.obj_version)
        assert got == pytest.approx(want, rel=1e-12, abs=0.0)


def _toy():
    from tests.test_ft_term import _make, _pos
    nl, io_term, _, distance = _make(k=4, chunk=1)
    ft_term = FtTerm(io_term, distance)
    ft_term.set_home([0])
    return io_term, ft_term, _pos(nl), float(distance.max())


def _drive(state, normalizer, io_term, ft_term, pos, ecc_max):
    """Run the same seven callbacks through whichever path is wired up."""
    lambdas = []
    wirelength_op = lambda p: (p ** 2).sum()
    for iteration, overflow, _, _, _, _ in EVENTS:
        state.update_continuous(iteration, overflow, 100.0, 1.0)
        publish = lambda: publish_atomic(state, io_term, ft_term, wirelength_op,
                                         pos, iteration, state.tau / 100.0,
                                         ecc_max, 1.0)
        if normalizer is None:
            record = publish()
            lambdas.append((state.lambda_io, state.lambda_io * state.kappa_ft,
                            state.obj_version, record["grad_l1_io"]))
            state.mark_refreshed()
        else:
            txn = normalizer.transaction(iteration, overflow, state.tau, 1.0,
                                         legacy_publish=publish)
            lambdas.append((txn.lambdas["io"], txn.lambdas["ft"], txn.obj_version,
                            txn.legacy_record["grad_l1_io"]))
            normalizer.mark_refreshed()
    return lambdas


def test_legacy_policy_reproduces_the_retired_lambda_exactly():
    io_term, ft_term, pos, ecc_max = _toy()
    direct_state = ScheduleState(rho_max=0.4, n_ramp=20, f_ft_max=0.25, c_lip=1e6)
    direct = _drive(direct_state, None, io_term, ft_term, pos, ecc_max)

    io_term2, ft_term2, pos2, ecc_max2 = _toy()
    adapted_state = ScheduleState(rho_max=0.4, n_ramp=20, f_ft_max=0.25, c_lip=1e6)
    normalizer = TermNormalizer(policy="legacy", legacy_state=adapted_state)
    normalizer.register("io", object(), 1.0)
    normalizer.register("ft", object(), ecc_max2)
    adapted = _drive(adapted_state, normalizer, io_term2, ft_term2, pos2, ecc_max2)

    assert adapted == direct                      # bit-for-bit, not approx
    assert any(row[0] > 0.0 for row in adapted), "lambda never activated"


def test_legacy_policy_delegates_the_version_counters():
    io_term, ft_term, pos, ecc_max = _toy()
    state = ScheduleState(rho_max=0.4, n_ramp=20, f_ft_max=0.25, c_lip=1e6)
    normalizer = TermNormalizer(policy="legacy", legacy_state=state)
    normalizer.register("io", object(), 1.0)
    normalizer.register("ft", object(), ecc_max)
    ref_before = normalizer.lambdas
    state.update_continuous(0, 0.5, 100.0, 1.0)
    assert normalizer.obj_version == state.obj_version
    normalizer.transaction(0, 0.5, state.tau, 1.0,
                           legacy_publish=lambda: publish_atomic(
                               state, io_term, ft_term, lambda p: (p ** 2).sum(),
                               pos, 0, state.tau / 100.0, ecc_max, 1.0))
    assert normalizer.obj_version == state.obj_version
    assert normalizer.needs_refresh() and state.needs_refresh()
    # `self.lambdas` must be mutated in place, never rebound, on the legacy
    # path too: a driver's `term_fn` closure holds a reference to this exact
    # dict (fix round 1, I-1).
    assert normalizer.lambdas is ref_before
    normalizer.mark_refreshed()
    assert not normalizer.needs_refresh() and not state.needs_refresh()
    assert normalizer.lambdas is ref_before


def test_legacy_policy_requires_the_publish_callable():
    state = ScheduleState(rho_max=0.4)
    normalizer = TermNormalizer(policy="legacy", legacy_state=state)
    normalizer.register("io", object(), 1.0)
    with pytest.raises(ValueError):
        normalizer.transaction(0, 0.5, 10.0, 1.0)


def test_legacy_policy_fills_the_trace_row(tmp_path):
    from ioplace.norm_trace import NormTraceWriter, read_norm_trace
    io_term, ft_term, pos, ecc_max = _toy()
    state = ScheduleState(rho_max=0.4, n_ramp=0, f_ft_max=0.25, c_lip=1e6)
    path = tmp_path / "legacy.norm_trace.jsonl"
    with NormTraceWriter(str(path)) as writer:
        normalizer = TermNormalizer(policy="legacy", legacy_state=state,
                                    trace=writer)
        normalizer.register("io", object(), 1.0)
        normalizer.register("ft", object(), ecc_max)
        state.update_continuous(100, 0.10, 100.0, 1.0)
        normalizer.transaction(100, 0.10, state.tau, 1.0,
                               legacy_publish=lambda: publish_atomic(
                                   state, io_term, ft_term, lambda p: (p ** 2).sum(),
                                   pos, 100, state.tau / 100.0, ecc_max, 1.0))
        normalizer.mark_refreshed()
    row, = read_norm_trace(str(path))
    assert row["policy"] == "legacy"
    assert row["terms"]["io"]["lam"] == pytest.approx(state.lambda_io, rel=1e-12)
    assert row["terms"]["ft"]["lam"] == pytest.approx(
        state.lambda_io * state.kappa_ft, rel=1e-12)
    assert row["cmax"] == pytest.approx(state.Cmax, rel=1e-12)
    assert row["terms"]["io"]["wt"] == pytest.approx(
        state.rho * activation_ramp(100, state.it_activate, state.n_ramp),
        rel=1e-12)


def test_legacy_policy_rejects_grad_norms():
    """The non-legacy EMA path (`update_grad_norms`) must never run on a
    legacy call: gradients are measured by `publish_atomic` inside
    `legacy_publish`, not by a probe (fix round 1, I-6)."""
    state = ScheduleState(rho_max=0.4)
    normalizer = TermNormalizer(policy="legacy", legacy_state=state)
    normalizer.register("io", object(), 1.0)
    with pytest.raises(ValueError):
        normalizer.transaction(0, 0.5, 10.0, 1.0, grad_norms={"wl": 1.0, "io": 1.0})
