import math
import pytest

from ioplace.norm import (EPS, adaptive_lambda, clip_sum_to_cap, cmax_from_curvatures,
                          ema_update, grandplan_lambda, grandplan_weight,
                          parse_target_shares)
from ioplace.norm import NormTransaction, TermConfig, TermNormalizer, TermState, VersionPair
from ioplace.schedules import activation_ramp, lipschitz_cap


def test_ema_update_seeds_then_damps():
    assert ema_update(None, 100.0) == pytest.approx(100.0, rel=1e-12)
    assert ema_update(100.0, 50.0, ema=0.5) == pytest.approx(75.0, rel=1e-12)
    assert ema_update(100.0, 50.0, ema=0.0) == pytest.approx(50.0, rel=1e-12)


def test_grandplan_weight_steps_every_ramp_period():
    assert grandplan_weight(99, 100) == 0.0
    assert grandplan_weight(100, 100) == pytest.approx(0.05, rel=1e-12)
    assert grandplan_weight(199, 100) == pytest.approx(0.05, rel=1e-12)
    assert grandplan_weight(200, 100) == pytest.approx(0.10, rel=1e-12)
    assert grandplan_weight(1000, 100) == pytest.approx(0.50, rel=1e-12)


def test_grandplan_weight_saturates_exactly_at_wt_max():
    assert grandplan_weight(2000, 100) == 1.0
    assert grandplan_weight(50000, 100) == 1.0
    assert grandplan_weight(2000, 100, wt_max=0.4) == 0.4


def test_grandplan_weight_is_zero_before_activation():
    assert grandplan_weight(500, None) == 0.0


def test_grandplan_weight_rejects_nonpositive_period():
    with pytest.raises(ValueError):
        grandplan_weight(500, 100, ramp_period=0)


def test_grandplan_lambda_scales_ratio_and_only_zero_is_dead():
    """Controller ruling F1' (round 2): the relative `eps_rel * wl_norm`
    deadness threshold is gone -- it tracked `||grad WL||` rather than the
    term, and misclassified a healthy gradient in late GP. A tiny-but-nonzero
    gradient now yields the ordinary `wt * ratio` (bounded downstream by the
    Lipschitz cap, exactly as the retired path's `min(base, cap)` does)."""
    assert grandplan_lambda(0.05, 100.0, 10.0) == pytest.approx(5.0, rel=1e-12)
    assert grandplan_lambda(0.05, 1e9, 1.0) == pytest.approx(0.05e9, rel=1e-12)
    assert grandplan_lambda(0.05, 1e9, 0.0) == 0.0


def test_adaptive_lambda_is_a_fixed_point_at_the_target_share():
    # lam*g == f*G  =>  the multiplicative factor is 1 and momentum is a no-op
    got = adaptive_lambda(2.0, 10.0, 0.2, 100.0, 1000.0)
    assert got == pytest.approx(2.0, rel=1e-12)


def test_adaptive_lambda_applies_sqrt_update_then_momentum():
    got = adaptive_lambda(1.0, 10.0, 0.2, 100.0, 1000.0)
    expected = 0.75 * 1.0 + 0.25 * math.sqrt(2.0)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(1.1035533905932737, rel=1e-12)


def test_adaptive_lambda_bootstraps_undamped_from_the_grandplan_form():
    got = adaptive_lambda(0.0, 10.0, 0.2, 100.0, 1000.0)
    assert got == pytest.approx(0.2 * 1000.0 / 10.0, rel=1e-12)


def test_adaptive_lambda_guards_zero_share_and_dead_gradient():
    assert adaptive_lambda(1.0, 10.0, 0.0, 100.0, 1000.0) == 0.0
    assert adaptive_lambda(1.0, 0.0, 0.2, 100.0, 1000.0) == 0.0
    # Ruling F1': a tiny-but-nonzero gradient is measured, not discarded.
    assert adaptive_lambda(1.0, 0.5, 0.2, 100.0, 1000.0) > 0.0


def test_cmax_is_the_lambda_weighted_mean_curvature():
    # Controller ruling 2026-09-19 (fix round 1): Cmax = sum_t lambda_t*curv_t
    # / sum_t lambda_t, replacing the derived-kappa form this test used to
    # exercise (that form degenerated whenever lambda_io == 0).
    assert cmax_from_curvatures([]) == 1.0                          # no terms
    assert cmax_from_curvatures([(0.0, 6.0)]) == 1.0                # sum(lambda) == 0
    assert cmax_from_curvatures([(1.0, 1.0)]) == pytest.approx(1.0, rel=1e-12)
    assert cmax_from_curvatures([(1.0, 1.0), (1.0, 6.0)]) == pytest.approx(3.5, rel=1e-12)
    assert cmax_from_curvatures([(200.0, 1.0), (100.0, 6.0)]) == pytest.approx(
        800.0 / 300.0, rel=1e-12)                                   # 800/300 = 2.6666...


def test_clip_sum_to_cap_is_a_noop_below_the_cap():
    got, binding = clip_sum_to_cap({"io": 10.0, "ft": 5.0}, 100.0)
    assert got == {"io": 10.0, "ft": 5.0}
    assert binding is None
    got, binding = clip_sum_to_cap({"io": 10.0}, float("inf"))
    assert got == {"io": 10.0} and binding is None


def test_clip_sum_to_cap_rescales_and_names_the_dominant_term():
    got, binding = clip_sum_to_cap({"io": 200.0, "ft": 150.0}, 285.7142857142857)
    assert sum(got.values()) == pytest.approx(285.7142857142857, rel=1e-12)
    assert got["io"] == pytest.approx(163.26530612244898, rel=1e-12)
    assert got["ft"] == pytest.approx(122.44897959183673, rel=1e-12)
    assert binding == "io"


def test_clip_sum_to_cap_breaks_ties_alphabetically_and_tolerates_zero():
    _, binding = clip_sum_to_cap({"io": 5.0, "cap": 5.0}, 1.0)
    assert binding == "cap"
    got, binding = clip_sum_to_cap({"io": 0.0}, 1.0)
    assert got == {"io": 0.0} and binding is None


def test_parse_target_shares():
    assert parse_target_shares("io=0.3,ft=0.1") == {"io": 0.3, "ft": 0.1}
    assert parse_target_shares(" io = 0.3 ") == {"io": 0.3}
    assert parse_target_shares("") == {}
    assert parse_target_shares(None) == {}
    with pytest.raises(ValueError):
        parse_target_shares("io")


@pytest.mark.parametrize("spec", [
    "io=0.3,io=0.9",        # duplicate: last used to win, silently
    "io=nan",               # nan defeats every downstream `<= 0.0` guard
    "io=inf",
    "io=-0.2",
    "io=1.5",
    "io=0.3,ft=",           # empty value
])
def test_parse_target_shares_rejects_bad_values(spec):
    """Review I6: every one of these was silently accepted (or, for the empty
    value, raised an unhelpful float() ValueError)."""
    with pytest.raises(ValueError):
        parse_target_shares(spec)


def _norm_a(**kwargs):
    """Policy-A normalizer with one IO term, cap effectively disabled."""
    n = TermNormalizer(policy="grandplan", **kwargs)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    return n


class _FakeGrad:
    """Tiny torch-free stand-in for a gradient tensor: supports exactly the
    operations `_cancellation_ratio`/`_norm` use (`*` by a scalar, `+`,
    `.abs().sum()`), so the cancellation-ratio maths can be unit-tested
    without a real probe (Task 4 populates `_grad_cache` for real)."""

    def __init__(self, values):
        self.values = list(values)

    def __mul__(self, scalar):
        return _FakeGrad(v * scalar for v in self.values)

    def __add__(self, other):
        return _FakeGrad(a + b for a, b in zip(self.values, other.values))

    def abs(self):
        return _FakeGrad(abs(v) for v in self.values)

    def sum(self):
        return sum(self.values)


def test_normalizer_rejects_unknown_policy_and_norm_order():
    with pytest.raises(ValueError):
        TermNormalizer(policy="bogus")
    with pytest.raises(ValueError):
        TermNormalizer(norm_p=3)
    with pytest.raises(ValueError):
        TermNormalizer(policy="legacy")          # legacy needs a ScheduleState


def test_weights_has_no_pure_preview_under_the_legacy_policy():
    """`weights()` stays a NotImplementedError under `policy="legacy"` even
    with the adapter landed (Task 6): there is no pure preview to give.
    Coefficients on the legacy path come from `apply_ft_transaction`, which
    only runs -- and only measures gradients -- inside `legacy_publish`
    during `transaction()`; there is no side-effect-free way to compute them
    ahead of a real transaction."""
    from ioplace.schedules import ScheduleState
    n = TermNormalizer(policy="legacy", legacy_state=ScheduleState(rho_max=1.0))
    n.register("io", object(), 1.0)
    with pytest.raises(NotImplementedError):
        n.weights(0, 0.85, tau=1000.0, gamma=1e-12)


def test_register_rejects_duplicates_and_seeds_state():
    n = _norm_a()
    assert n.lambdas == {"io": 0.0}
    assert n.states["io"] == TermState()
    assert n.configs["io"] == TermConfig("io", 1.0, 0.0, 0.90, 0)
    with pytest.raises(ValueError):
        n.register("io", object(), 1.0)


def test_update_grad_norms_sets_instant_ratio_and_ema():
    n = _norm_a(ema=0.5)
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    assert n.states["io"].ratio_inst == pytest.approx(100.0, rel=1e-12)
    assert n.states["io"].ratio_ema == pytest.approx(100.0, rel=1e-12)
    n.update_grad_norms({"wl": 1000.0, "io": 20.0})
    assert n.states["io"].ratio_inst == pytest.approx(50.0, rel=1e-12)
    assert n.states["io"].ratio_ema == pytest.approx(75.0, rel=1e-12)


def test_zero_grad_norm_does_not_divide_by_zero():
    # Controller ruling F1' (round 2): an *exactly* zero gradient is the only
    # dead case left -- `ratio_inst` is None and `ratio_ema` is left exactly as
    # it was (None here, since this is the first measurement) rather than
    # seeded with `wl_norm / EPS`.
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 0.0})
    assert n.states["io"].ratio_inst is None
    assert n.states["io"].ratio_ema is None


def test_zero_gradient_probe_does_not_poison_the_ratio_ema():
    # Controller ruling (fix round 1): a zero (or near-zero) term gradient
    # must not seed ~1e33 into the EMA -- the next, informative probe should
    # seed it cleanly instead, exactly as if the dead probe never happened.
    n = _norm_a(ema=0.5)
    n.update_grad_norms({"wl": 1000.0, "io": 0.0})
    assert n.states["io"].ratio_inst is None
    assert n.states["io"].ratio_ema is None
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    assert n.states["io"].ratio_inst == pytest.approx(100.0, rel=1e-12)
    assert n.states["io"].ratio_ema == pytest.approx(100.0, rel=1e-12)


def test_term_stays_inactive_above_the_activation_overflow():
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    assert n.weights(0, overflow=0.95, tau=1000.0, gamma=1e-12) == {"io": 0.0}
    assert n.states["io"].active is False


def test_policy_a_lambda_is_stepped_weight_times_ema_ratio():
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    assert n.states["io"].it_activate == 0
    assert txn.lambdas["io"] == pytest.approx(0.05 * 100.0, rel=1e-12)
    n.mark_refreshed()
    n.update_grad_norms({"wl": 1000.0, "io": 20.0})       # ratio_ema -> 75
    txn = n.transaction(100, 0.60, tau=1000.0, gamma=1e-12)
    assert txn.lambdas["io"] == pytest.approx(0.10 * 75.0, rel=1e-12)


def test_the_committed_lambda_is_unramped_and_the_ramp_is_applied_per_iteration():
    """Review I1: `lambdas` (and the trace's `lam`) hold the UN-ramped
    coefficient; the activation ramp is applied by `applied_lambda` on every
    GP iteration, so it is visible between two `every`-gated transactions
    instead of being aliased away by them."""
    n = TermNormalizer(policy="grandplan")
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=20)
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    assert txn.lambdas["io"] == pytest.approx(0.05 * 100.0, rel=1e-12)
    assert n.applied_lambda("io", 0) == 0.0                    # ramp(0, 0, 20) == 0
    assert n.applied_lambda("io", 10) == pytest.approx(0.05 * 0.5 * 100.0, rel=1e-12)
    assert n.applied_lambda("io", 20) == pytest.approx(0.05 * 100.0, rel=1e-12)
    assert txn.row["terms"]["io"]["lam_applied"] == 0.0
    n.mark_refreshed()
    txn = n.transaction(10, 0.85, tau=1000.0, gamma=1e-12)
    assert txn.lambdas["io"] == pytest.approx(0.05 * 100.0, rel=1e-12)
    assert txn.row["terms"]["io"]["lam_applied"] == pytest.approx(
        0.05 * 0.5 * 100.0, rel=1e-12)


def test_the_external_activation_clock_wins_over_the_first_gated_probe():
    """Review I1a: `it_activate` is the ScheduleState's per-iteration
    activation instant, not the first `every`-gated callback that observed the
    overflow threshold (which is up to `every`=50 iterations later -- longer
    than the whole n_ramp=20 window)."""
    n = TermNormalizer(policy="grandplan", wt0=1.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=20)
    n.set_activation("io", 37)          # ScheduleState latched at iteration 37
    n.update_activation(50, 0.85)       # the first gated callback would say 50
    assert n.states["io"].it_activate == 37 and n.states["io"].active
    txn = n.transaction(50, 0.85, tau=1000.0, gamma=0.0,
                        grad_norms={"wl": 1000.0, "io": 10.0})
    lam = txn.lambdas["io"]
    assert lam == pytest.approx(100.0, rel=1e-12)
    for offset in range(0, 21):
        assert n.applied_lambda("io", 37 + offset) == pytest.approx(
            lam * activation_ramp(37 + offset, 37, 20), rel=1e-12)
    assert n.applied_lambda("io", 57) == pytest.approx(lam, rel=1e-12)
    assert n.applied_lambda("io", 500) == pytest.approx(lam, rel=1e-12)
    assert txn.row["terms"]["io"]["lam_applied"] == pytest.approx(
        lam * 13.0 / 20.0, rel=1e-12)


def test_per_term_weight_ceilings_saturate_independently():
    """Review I4: without per-term `wt_max` every grandplan term converges to
    the same ceiling, so FT's realised force share converges to IO's (an
    effective f_ft of 1.0 against legacy's 0.25)."""
    n = TermNormalizer(policy="grandplan", wt0=0.05, wt_step=0.05,
                       ramp_period=10, wt_max=1.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 1.0, activate_overflow=0.90, n_ramp=0,
               requires="io", wt_max=0.25)
    for iteration in range(0, 1000, 50):
        n.transaction(iteration, 0.85, tau=1000.0, gamma=0.0,
                      grad_norms={"wl": 1000.0, "io": 10.0, "ft": 10.0})
        n.mark_refreshed()
    assert n.states["io"].wt == pytest.approx(1.00, rel=1e-12)
    assert n.states["ft"].wt == pytest.approx(0.25, rel=1e-12)
    assert n.lambdas["ft"] / n.lambdas["io"] == pytest.approx(0.25, rel=1e-12)
    assert n.transaction(1000, 0.85, tau=1000.0, gamma=0.0).row[
        "terms"]["ft"]["wt_max"] == pytest.approx(0.25, rel=1e-12)


def test_dependent_term_is_zeroed_when_its_base_dies():
    """Review C1: `FtTerm` can only express the FT force as
    `lambda_io * kappa`, so a nonzero lambda_ft with a dead lambda_io is
    unapplicable -- and the pre-fix driver raised RuntimeError there, killing a
    multi-hour run on one ordinary probe. Worse, lambda_ft *rose* at exactly
    that probe (20 -> 30 in the review's reproduction), because FT's own EMA
    kept updating while IO's signal vanished."""
    n = TermNormalizer(policy="grandplan", wt0=1.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 1.0, activate_overflow=0.90, n_ramp=0,
               requires="io")
    healthy = n.transaction(0, 0.85, tau=1000.0, gamma=0.0,
                            grad_norms={"wl": 1000.0, "io": 100.0, "ft": 50.0})
    assert healthy.lambdas["io"] > 0.0 and healthy.lambdas["ft"] > 0.0
    n.mark_refreshed()
    # Exactly-zero IO gradient: no crossing to pull on, so IO is genuinely dead
    # and FT must die with it (controller ruling F1 keys the dependency on
    # "base lambda == 0 after the tiny-gradient logic").
    dead = n.transaction(50, 0.80, tau=1000.0, gamma=0.0,
                         grad_norms={"wl": 1000.0, "io": 0.0, "ft": 50.0})
    assert dead.lambdas == {"io": 0.0, "ft": 0.0}
    assert n.states["ft"].wt == 0.0
    n.mark_refreshed()
    recovered = n.transaction(100, 0.75, tau=1000.0, gamma=0.0,
                              grad_norms={"wl": 1000.0, "io": 100.0, "ft": 50.0})
    assert recovered.lambdas["io"] > 0.0 and recovered.lambdas["ft"] > 0.0


def test_requires_zeroing_happens_before_cmax_so_a_dead_base_does_not_inflate_it():
    """Ruling D-10 (P-D Task 8): a term registered with `requires=<base>`
    must be zeroed for the Cmax computation, not only for the final
    published lambda -- otherwise a dependent whose base just died still
    counts its (about to be discarded) pre-clip lambda*curvature in Cmax's
    weighted mean, inflating Cmax and over-clipping every surviving term.
    Harmless with only `io`/`ft` registered (this scenario always zeroed
    `ft`'s *published* lambda on the same probe `io`'s own gradient died, and
    no existing test read `cmax` at that probe), but registering a third,
    independent term (`cap`) makes the interaction observable and load-
    bearing, since `cap` has no `requires` relation of its own and would
    otherwise absorb the inflated Cmax's over-clipping."""
    n = TermNormalizer(policy="grandplan", wt0=1.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 6.0, activate_overflow=0.90, n_ramp=0,
               requires="io")
    n.transaction(0, 0.85, tau=1000.0, gamma=0.0,
                  grad_norms={"wl": 1000.0, "io": 100.0, "ft": 50.0})
    n.mark_refreshed()
    # io's gradient vanishes; ft's own gradient stays healthy, so ft's own
    # pre-clip lambda is still computed as nonzero before the `requires`
    # chain zeroes it for publication.
    txn = n.transaction(50, 0.80, tau=1000.0, gamma=0.0,
                        grad_norms={"wl": 1000.0, "io": 0.0, "ft": 50.0})
    assert txn.lambdas == {"io": 0.0, "ft": 0.0}
    # Buggy order: Cmax = curv_ft (6.0), since ft's un-zeroed pre-clip lambda
    # is the only nonzero weight in the mean. Fixed order: both pre-clip
    # lambdas are zero by the time Cmax is computed, so cmax_from_curvatures
    # falls back to its "no force" default of 1.0.
    assert txn.cmax == pytest.approx(1.0, rel=1e-12)


def test_a_tiny_but_nonzero_gradient_is_measured_and_only_bounded_by_the_cap():
    """Controller ruling F1' (round 2), superseding both earlier rules for
    `0 < grad <= eps_rel * ||grad WL||`: neither zero the coefficient (fix
    round 1) nor hand it the cap's whole headroom (ruling F1). The ratio is
    perfectly computable, so measure it, feed it through the policy, and let
    the shared Lipschitz cap bound the result -- which is the retired path's
    own `min(base, cap)`. Measured on `mempool_group`, the two superseded rules
    respectively switched the IO penalty off for the last quarter of GP and
    jumped it to 14x its last healthy value (stalling GP at overflow 0.71)."""
    n = TermNormalizer(policy="grandplan", wt0=1.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 1.0, activate_overflow=0.90, n_ramp=0,
               requires="io")
    n.transaction(0, 0.85, tau=100.0, gamma=20.0,
                  grad_norms={"wl": 1000.0, "io": 10.0, "ft": 20.0})
    n.mark_refreshed()
    assert n.states["io"].ratio_ema == pytest.approx(100.0, rel=1e-12)
    # 0.5 is below the retired eps_rel(1e-3) * 1000 == 1.0 threshold, and is
    # now an ordinary measurement.
    txn = n.transaction(50, 0.80, tau=100.0, gamma=20.0,
                        grad_norms={"wl": 1000.0, "io": 0.5, "ft": 20.0})
    assert n.states["io"].ratio_inst == pytest.approx(2000.0, rel=1e-12)
    assert n.states["io"].ratio_ema == pytest.approx(1050.0, rel=1e-12)
    cap = lipschitz_cap(100.0, 20.0, 1.0, 1.0)
    assert txn.cap == pytest.approx(cap, rel=1e-12)             # 500.0
    # pre-clip io = 1.0*1050, ft = 1.0*50 -> sum 1100 > cap, one common rescale
    assert sum(txn.lambdas.values()) == pytest.approx(cap, rel=1e-12)
    assert txn.lambdas["io"] == pytest.approx(cap * 1050.0 / 1100.0, rel=1e-12)
    assert txn.lambdas["ft"] == pytest.approx(cap * 50.0 / 1100.0, rel=1e-12)
    assert txn.lambdas["io"] / txn.lambdas["ft"] == pytest.approx(21.0, rel=1e-12)
    assert txn.cap_binding == "io"
    # not ruling F1's "take the whole headroom" behaviour
    assert txn.lambdas["io"] != pytest.approx(cap, rel=1e-6)


def test_an_exactly_zero_gradient_still_zeroes_the_coefficient():
    n = _norm_a()
    txn = n.transaction(0, 0.85, tau=100.0, gamma=20.0,
                        grad_norms={"wl": 1000.0, "io": 10.0})
    assert txn.lambdas["io"] > 0.0
    n.mark_refreshed()
    txn = n.transaction(50, 0.80, tau=100.0, gamma=20.0,
                        grad_norms={"wl": 1000.0, "io": 0.0})
    assert txn.lambdas["io"] == 0.0


def test_dependent_kappa_is_clamped_like_the_legacy_schedule():
    """Review M2: the non-legacy driver recovers `kappa = lam_ft/lam_io` by
    division, which a denormal base coefficient blows up. Mirrors
    `ScheduleState.kappa_max`."""
    n = TermNormalizer(policy="grandplan", wt0=1.0, kappa_max=100.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 1.0, activate_overflow=0.90, n_ramp=0,
               requires="io")
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=0.0,
                        grad_norms={"wl": 1000.0, "io": 1000.0, "ft": 2.0})
    assert txn.lambdas["io"] == pytest.approx(1.0, rel=1e-12)
    assert txn.lambdas["ft"] == pytest.approx(100.0, rel=1e-12)   # not 500.0
    assert txn.row["terms"]["ft"]["kappa_clamped"] is True
    assert txn.row["terms"]["io"]["kappa_clamped"] is False


def test_register_range_checks():
    """Review M3: a mis-registered term used to fail silently --
    `curvature=0` gave `cmax=0`, which `lipschitz_cap`'s `max(1.0, cmax)`
    swallowed, so the term got no cap protection and `cap_binding` became
    meaningless."""
    n = TermNormalizer(policy="grandplan")
    with pytest.raises(ValueError):
        n.register("a", object(), 0.0)                 # curvature < 1
    with pytest.raises(ValueError):
        n.register("a", object(), -5.0)
    with pytest.raises(ValueError):
        n.register("a", object(), float("nan"))
    with pytest.raises(ValueError):
        n.register("a", object(), 1.0, n_ramp=-1)
    with pytest.raises(ValueError):
        n.register("a", object(), 1.0, activate_overflow=0.0)
    with pytest.raises(ValueError):
        n.register("a", object(), 1.0, target_share=1.5)
    with pytest.raises(ValueError):
        n.register("a", object(), 1.0, wt_max=0.0)
    with pytest.raises(ValueError):
        n.register("a", object(), 1.0, ramp_period=-10)
    with pytest.raises(ValueError):
        n.register("a", object(), 1.0, requires="nope")
    # `--of-on 2` is a legitimate "always active" setting the driver passes
    # straight through, so activate_overflow has no upper bound.
    n.register("a", object(), 1.0, activate_overflow=2.0)
    assert n.configs["a"].activate_overflow == 2.0


def test_transaction_rejects_a_nonfinite_committed_lambda():
    """Review I6: no derived coefficient was checked for finiteness, so a
    `nan` ratio reached the objective term itself (every downstream guard --
    `lam <= 0.0`, `target_share <= 0.0` -- is False for `nan`)."""
    n = _norm_a()
    with pytest.raises(FloatingPointError):
        n.transaction(0, 0.85, tau=1000.0, gamma=1e-12,
                      grad_norms={"wl": 1000.0, "io": float("nan")})
    assert n.obj_version == 0                     # nothing was committed


def test_cap_uses_per_term_curvature_and_clips_the_sum():
    n = TermNormalizer(policy="grandplan", wt0=1.0, wt_max=1.0)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 6.0, activate_overflow=0.90, n_ramp=0)
    n.update_grad_norms({"wl": 1000.0, "io": 5.0, "ft": 10.0})   # ratios 200, 100
    txn = n.transaction(0, 0.85, tau=100.0, gamma=20.0)
    # pre-clip lambdas: io = wt*ratio_io = 1.0*200 = 200, ft = 1.0*100 = 100
    # Cmax is the lambda-weighted mean curvature (controller ruling
    # 2026-09-19): Cmax = (200*1 + 100*6) / (200 + 100) = 800/300 = 8/3
    assert txn.cmax == pytest.approx(8.0 / 3.0, rel=1e-12)
    # cap = c_lip*tau^2/(gamma*Cmax) = 100^2/(20*8/3) = 10000/53.333... = 187.5,
    # which is < 300 so the cap binds
    assert txn.cap == pytest.approx(lipschitz_cap(100.0, 20.0, 1.0, 8.0 / 3.0), rel=1e-12)
    assert txn.cap == pytest.approx(187.5, rel=1e-12)
    assert sum(txn.lambdas.values()) == pytest.approx(txn.cap, rel=1e-12)
    # scale = cap/total = 187.5/300 = 0.625 -> io = 125.0, ft = 62.5
    assert txn.lambdas["io"] == pytest.approx(125.0, rel=1e-12)
    assert txn.lambdas["ft"] == pytest.approx(62.5, rel=1e-12)
    # binding is the term with the largest pre-clip lambda*curv: io=200*1=200,
    # ft=100*6=600, so ft dominates the Lipschitz bound even though its raw
    # pre-clip lambda (100) is smaller than io's (200)
    assert txn.cap_binding == "ft"
    assert txn.lambdas["io"] / txn.lambdas["ft"] == pytest.approx(2.0, rel=1e-12)


def test_cap_does_not_bind_when_gamma_is_tiny():
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=0.0)
    assert txn.cap == float("inf")
    assert txn.cap_binding is None
    assert txn.row["cap"] is None                                   # inf is not JSON


def test_transaction_bumps_obj_version_exactly_once_and_demands_refresh():
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    assert n.obj_version == 0 and n.needs_refresh() is False
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    assert txn.obj_version == 1 and n.obj_version == 1
    assert txn.needs_refresh is True and n.needs_refresh() is True
    with pytest.raises(RuntimeError):
        n.transaction(50, 0.80, tau=1000.0, gamma=1e-12)
    n.mark_refreshed()
    assert n.needs_refresh() is False
    n.transaction(50, 0.80, tau=1000.0, gamma=1e-12)
    assert n.obj_version == 2


def test_transaction_accepts_grad_norms_inline():
    n = _norm_a()
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12,
                        grad_norms={"wl": 1000.0, "io": 10.0})
    assert txn.lambdas["io"] == pytest.approx(5.0, rel=1e-12)


def test_weights_is_a_pure_preview_that_does_not_bump_the_version():
    n = _norm_a()
    ref_before = n.lambdas
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    preview = n.weights(0, 0.85, tau=1000.0, gamma=1e-12)
    assert preview == {"io": pytest.approx(5.0, rel=1e-12)}
    assert n.obj_version == 0
    assert n.lambdas == {"io": 0.0}                                  # not committed
    assert n.transaction(0, 0.85, tau=1000.0, gamma=1e-12).lambdas == preview
    assert n.lambdas == preview                                      # now committed
    # `self.lambdas` must be mutated in place, never rebound: a driver's
    # `term_fn` closure holds a reference to this exact dict.
    assert n.lambdas is ref_before


def test_should_probe_follows_probe_every():
    n = _norm_a(probe_every=50)
    assert n.should_probe(0) and n.should_probe(100)
    assert not n.should_probe(51)


def test_row_carries_every_logged_field():
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    row = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12).row
    assert row["iteration"] == 0 and row["overflow"] == pytest.approx(0.85)
    assert row["policy"] == "grandplan" and row["norm_p"] == 1
    assert row["grad_l1_wl"] == pytest.approx(1000.0)
    assert row["obj_version"] == 1 and row["refreshed_version"] == 0
    io = row["terms"]["io"]
    assert io["grad_l1"] == pytest.approx(10.0)
    assert io["ratio_inst"] == pytest.approx(100.0)
    assert io["ratio_ema"] == pytest.approx(100.0)
    assert io["wt"] == pytest.approx(0.05)
    assert io["lam"] == pytest.approx(5.0)
    # realised share lam*||grad T|| / (||grad WL|| + sum lam*||grad T||)
    assert io["share"] == pytest.approx(50.0 / 1050.0, rel=1e-12)
    assert io["active"] is True
    assert io["lam_applied"] == pytest.approx(5.0)     # n_ramp=0 -> ramp is 1
    assert io["wt_max"] == pytest.approx(1.0)
    assert io["kappa_clamped"] is False
    assert row["cancellation_ratio"] is None                         # no probe cache


def test_cancellation_ratio_reads_the_seeded_probe_cache():
    # Task 4 populates `_grad_cache`/`_probed_this_callback` for real via a
    # probe; here we seed them directly to unit-test the maths in isolation.
    n = TermNormalizer(policy="grandplan", track_cancellation=True)
    n.register("io", object(), 1.0, n_ramp=0)
    n.register("ft", object(), 1.0, n_ramp=0)
    n.states["io"].grad_norm = 1.0
    n.states["ft"].grad_norm = 1.0
    n._probed_this_callback = True
    # aligned gradients: ||lam_io*g + lam_ft*g|| / (lam_io*1 + lam_ft*1) == 1.0
    n._grad_cache = {"io": _FakeGrad([1.0, 0.0]), "ft": _FakeGrad([1.0, 0.0])}
    assert n._cancellation_ratio({"io": 1.0, "ft": 1.0}) == pytest.approx(1.0, rel=1e-12)
    # opposed gradients: the merged sum cancels to zero
    n._grad_cache = {"io": _FakeGrad([1.0, 0.0]), "ft": _FakeGrad([-1.0, 0.0])}
    assert n._cancellation_ratio({"io": 1.0, "ft": 1.0}) == pytest.approx(0.0, rel=1e-12)


def test_cancellation_ratio_is_none_when_track_cancellation_is_false():
    n = TermNormalizer(policy="grandplan", track_cancellation=False)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    # Seeded exactly as the aligned-gradient case above would be, so the only
    # difference from a real hit is `track_cancellation=False` -- if the gate
    # were dead this would still report 1.0.
    n._probed_this_callback = True
    n._grad_cache = {"io": _FakeGrad([1.0, 0.0])}
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    assert txn.cancellation_ratio is None
    assert txn.row["cancellation_ratio"] is None


def test_version_pair_is_equal_only_when_both_members_are_refreshed():
    a, b = _norm_a(), _norm_a()
    pair = VersionPair(a, b)
    assert pair.obj_version == pair.refreshed_version
    a.update_grad_norms({"wl": 1000.0, "io": 10.0})
    a.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    assert pair.obj_version != pair.refreshed_version
    a.mark_refreshed()
    assert pair.obj_version == pair.refreshed_version


def test_oneshot_lambda_reproduces_the_retired_route_expression():
    assert TermNormalizer.oneshot_lambda(0.1, 1000.0, 4.0) == 0.1 * 1000.0 / 4.0
    assert TermNormalizer.oneshot_lambda(0.1, 1000.0, 0.0) == 0.0


def test_install_version_invariant_guards_the_normalizer():
    torch = pytest.importorskip("torch")
    from ioplace.dreamplace_env import setup_dreamplace
    setup_dreamplace()
    from NesterovAcceleratedGradientOptimizer import NesterovAcceleratedGradientOptimizer as NAG
    from ioplace.dp_hook import install_version_invariant

    def obj_and_grad_fn(p):
        if p.grad is not None:
            p.grad.zero_()
        o = 0.5 * (p * p).sum()
        o.backward()
        return o.detach(), p.grad

    p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = NAG([p], lr=0.01, obj_and_grad_fn=obj_and_grad_fn,
              constraint_fn=lambda t: None, use_bb=False)
    obj_and_grad_fn(p)
    opt.step()
    n = _norm_a()
    uninstall = install_version_invariant(opt, n)
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    n.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    with pytest.raises(AssertionError):
        opt.step()
    n.mark_refreshed()
    opt.step()
    uninstall()


def _norm_b(share=0.2, **kwargs):
    n = TermNormalizer(policy="adaptive", **kwargs)
    n.register("io", object(), 1.0, target_share=share, activate_overflow=0.90, n_ramp=0)
    return n


def test_policy_b_bootstraps_from_the_grandplan_form():
    n = _norm_b()
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=0.0,
                        grad_norms={"wl": 1000.0, "io": 10.0})
    assert txn.lambdas["io"] == pytest.approx(20.0, rel=1e-12)
    assert n.states["io"].wt == pytest.approx(0.2, rel=1e-12)


def test_policy_b_applies_sqrt_update_with_momentum_on_the_second_probe():
    n = _norm_b()
    n.transaction(0, 0.85, tau=1000.0, gamma=0.0, grad_norms={"wl": 1000.0, "io": 10.0})
    n.mark_refreshed()
    txn = n.transaction(50, 0.85, tau=1000.0, gamma=0.0,
                        grad_norms={"wl": 1000.0, "io": 10.0})
    assert txn.lambdas["io"] == pytest.approx(20.477225575051661, rel=1e-12)


def test_policy_b_converges_to_the_requested_force_share():
    n = _norm_b(share=0.2)
    for iteration in range(0, 50 * 150, 50):
        n.transaction(iteration, 0.85, tau=1000.0, gamma=0.0,
                      grad_norms={"wl": 1000.0, "io": 10.0})
        n.mark_refreshed()
    lam = n.lambdas["io"]
    assert lam == pytest.approx(25.0, rel=1e-6)          # lam*g/(wl+lam*g) == 0.2
    share = lam * 10.0 / (1000.0 + lam * 10.0)
    assert share == pytest.approx(0.2, rel=1e-6)


def test_policy_b_ramps_the_applied_coefficient_not_the_target_share():
    """Review I1c: the ramp no longer scales policy B's target share (which
    made the controller chase a moving fixed point during the soft start);
    the committed coefficient is computed at the full share and the ramp is
    applied per iteration to what the objective sees."""
    n = TermNormalizer(policy="adaptive")
    n.register("io", object(), 1.0, target_share=0.2, activate_overflow=0.90, n_ramp=20)
    n.transaction(0, 0.85, tau=1000.0, gamma=0.0, grad_norms={"wl": 1000.0, "io": 10.0})
    assert n.states["io"].wt == pytest.approx(0.2, rel=1e-12)
    assert n.lambdas["io"] == pytest.approx(0.2 * 1000.0 / 10.0, rel=1e-12)
    assert n.applied_lambda("io", 0) == 0.0              # ramp 0
    assert n.applied_lambda("io", 10) == pytest.approx(0.5 * n.lambdas["io"], rel=1e-12)
    assert n.applied_lambda("io", 20) == pytest.approx(n.lambdas["io"], rel=1e-12)


def test_policy_b_keeps_its_momentum_state_across_a_dead_probe():
    """Review I7: one dead probe used to overwrite `state.lam` with 0, so the
    next probe re-bootstrapped from policy A's form (20.0 here) instead of
    resuming the converged multiplicative state (25.0)."""
    n = _norm_b(share=0.2)
    for iteration in range(0, 50 * 150, 50):
        n.transaction(iteration, 0.85, tau=1000.0, gamma=0.0,
                      grad_norms={"wl": 1000.0, "io": 10.0})
        n.mark_refreshed()
    converged = n.lambdas["io"]
    assert converged == pytest.approx(25.0, rel=1e-6)
    n.transaction(10000, 0.85, tau=1000.0, gamma=0.0,
                  grad_norms={"wl": 1000.0, "io": 0.0})
    assert n.lambdas["io"] == 0.0                         # published: nothing
    assert n.states["io"].lam_state == pytest.approx(
        converged, rel=1e-12)                          # policy state: kept
    n.mark_refreshed()
    n.transaction(10050, 0.85, tau=1000.0, gamma=0.0,
                  grad_norms={"wl": 1000.0, "io": 10.0})
    assert n.lambdas["io"] == pytest.approx(converged, rel=1e-6)
    assert n.lambdas["io"] != pytest.approx(0.2 * 1000.0 / 10.0, rel=1e-3)


def test_policy_b_zero_target_share_leaves_the_term_off():
    n = _norm_b(share=0.0)
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=0.0,
                        grad_norms={"wl": 1000.0, "io": 10.0})
    assert txn.lambdas["io"] == 0.0


def test_policy_b_obeys_the_same_cap_as_policy_a():
    n = TermNormalizer(policy="adaptive")
    n.register("io", object(), 1.0, target_share=0.9,
               activate_overflow=0.90, n_ramp=0)
    n.register("ft", object(), 6.0, target_share=0.9,
               activate_overflow=0.90, n_ramp=0)
    txn = n.transaction(0, 0.85, tau=100.0, gamma=20.0,
                        grad_norms={"wl": 1000.0, "io": 5.0, "ft": 10.0})
    # pre-clip lambdas (bootstrap): io = 0.9*1000/5 = 180, ft = 0.9*1000/10 = 90
    # Cmax is the lambda-weighted mean curvature (controller ruling
    # 2026-09-19): Cmax = (180*1 + 90*6) / (180 + 90) = 720/270 = 8/3
    assert txn.cmax == pytest.approx(8.0 / 3.0, rel=1e-12)
    # cap = 100^2/(20*8/3) = 10000/53.333... = 187.5, which is < 270 so the cap binds
    assert txn.cap == pytest.approx(100.0 * 100.0 / (20.0 * 8.0 / 3.0), rel=1e-12)
    assert txn.cap == pytest.approx(187.5, rel=1e-12)
    assert sum(txn.lambdas.values()) == pytest.approx(txn.cap, rel=1e-12)
    # scale = cap/total = 187.5/270 = 0.69444... -> io = 125.0, ft = 62.5
    assert txn.lambdas["io"] == pytest.approx(125.0, rel=1e-12)
    assert txn.lambdas["ft"] == pytest.approx(62.5, rel=1e-12)
    # binding is the term with the largest pre-clip lambda*curv: io=180*1=180,
    # ft=90*6=540, so ft dominates even though its raw pre-clip lambda (90) is
    # smaller than io's (180)
    assert txn.cap_binding == "ft"
    assert txn.lambdas["io"] / txn.lambdas["ft"] == pytest.approx(2.0, rel=1e-12)


class _Poly(object):
    """Toy term: value = coefficient * sum(pos**2), so grad = 2*coefficient*pos."""

    def __init__(self, coefficient):
        self.coefficient = coefficient

    def value(self, pos, ctx):
        return self.coefficient * (pos ** 2).sum()


def _probe_setup(policy="grandplan", **kwargs):
    torch = pytest.importorskip("torch")
    n = TermNormalizer(policy=policy, num_movable=1, num_nodes=2,
                       wt0=0.05, **kwargs)
    pos = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    wl_fn = lambda p: (p ** 3).sum()
    ctx = {"iteration": 0, "overflow": 0.85, "tau": 1000.0, "gamma": 0.0}
    return torch, n, pos, wl_fn, ctx


def test_probe_masks_fixed_and_filler_entries_like_independent_gradient():
    torch, n, pos, wl_fn, ctx = _probe_setup()
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    norms = n.probe(0, pos, wl_fn, ctx)
    assert norms["wl"] == pytest.approx(30.0, rel=1e-12)      # 3 + 27, [1] and [3] masked
    assert norms["a"] == pytest.approx(8.0, rel=1e-12)        # 2 + 6
    assert n.states["a"].grad_norm == pytest.approx(8.0, rel=1e-12)
    assert n.states["a"].ratio_ema == pytest.approx(30.0 / 8.0, rel=1e-12)


def test_probe_does_not_touch_the_live_gradient_buffer():
    torch, n, pos, wl_fn, ctx = _probe_setup()
    pos.grad = torch.full_like(pos, 123.0)
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    n.probe(0, pos, wl_fn, ctx)
    assert bool(torch.all(pos.grad == 123.0))


def test_probe_honours_norm_p_two():
    torch, n, pos, wl_fn, ctx = _probe_setup(norm_p=2)
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    norms = n.probe(0, pos, wl_fn, ctx)
    assert norms["wl"] == pytest.approx(27.166155414412157, rel=1e-12)   # sqrt(9+729)
    assert norms["a"] == pytest.approx(6.324555320336759, rel=1e-12)     # sqrt(4+36)


def test_probe_terms_subsetting_keeps_previous_measurements():
    torch, n, pos, wl_fn, ctx = _probe_setup()
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("b", _Poly(-0.5), 1.0, activate_overflow=0.90, n_ramp=0)
    n.probe(0, pos, wl_fn, ctx)
    assert n.states["b"].grad_norm == pytest.approx(4.0, rel=1e-12)
    n.probe(50, pos, wl_fn, ctx, probe_terms=["a"])
    assert n.states["b"].grad_norm == pytest.approx(4.0, rel=1e-12)


def test_cancellation_ratio_is_zero_for_exactly_opposed_terms():
    torch, n, pos, wl_fn, ctx = _probe_setup()
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("b", _Poly(-0.5), 1.0, activate_overflow=0.90, n_ramp=0)
    n.probe(0, pos, wl_fn, ctx)
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=0.0)
    assert txn.lambdas["a"] == pytest.approx(0.05 * 30.0 / 8.0, rel=1e-12)
    assert txn.lambdas["b"] == pytest.approx(0.05 * 30.0 / 4.0, rel=1e-12)
    assert txn.cancellation_ratio == pytest.approx(0.0, abs=1e-15)
    assert txn.row["cancellation_ratio"] == pytest.approx(0.0, abs=1e-15)


def test_cancellation_ratio_is_one_for_aligned_terms():
    torch, n, pos, wl_fn, ctx = _probe_setup()
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("b", _Poly(0.5), 1.0, activate_overflow=0.90, n_ramp=0)
    n.probe(0, pos, wl_fn, ctx)
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=0.0)
    assert txn.cancellation_ratio == pytest.approx(1.0, rel=1e-12)
    # Fix round 1, I-1: transaction() must clear the probe cache once it has
    # used it, so a pos-sized tensor per term does not stay pinned for the
    # whole probe interval.
    assert n._grad_cache == {}
    n.mark_refreshed()
    # No probe() this callback -- the cache from iteration 0 must not be
    # reused against iteration 50's coefficients.
    txn2 = n.transaction(50, 0.80, tau=1000.0, gamma=0.0,
                         grad_norms={"wl": 1000.0, "a": 8.0, "b": 4.0})
    assert txn2.cancellation_ratio is None


def test_track_cancellation_off_keeps_no_tensors():
    torch, n, pos, wl_fn, ctx = _probe_setup(track_cancellation=False)
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    n.probe(0, pos, wl_fn, ctx)
    assert n._grad_cache == {}
    assert n.transaction(0, 0.85, tau=1000.0, gamma=0.0).cancellation_ratio is None


def test_probe_rejects_a_nonfinite_gradient():
    torch, n, pos, wl_fn, ctx = _probe_setup()

    class _Bad(object):
        def value(self, pos, ctx):
            return (pos * float("inf")).sum()

    n.register("bad", _Bad(), 1.0, activate_overflow=0.90, n_ramp=0)
    with pytest.raises(FloatingPointError):
        n.probe(0, pos, wl_fn, ctx)


def test_norm_term_adapters_expose_the_production_terms():
    torch = pytest.importorskip("torch")
    from ioplace.ops.ft_term import FtTerm
    from ioplace.ops.norm_terms import FtNormTerm, IoNormTerm
    from tests.test_ft_term import _make, _pos
    nl, io, _, distance = _make(k=4, chunk=1)
    ft = FtTerm(io, distance)
    ft.set_home([0])
    pos = _pos(nl, requires=False)
    ctx_a = {"iteration": 0, "overflow": 0.5, "tau": 8.0, "gamma": 1.0}
    ctx_b = {"iteration": 0, "overflow": 0.5, "tau": 4.0, "gamma": 1.0}
    # value(pos, ctx) reads ctx["tau"]: two different taus must not agree.
    assert float(IoNormTerm(io).value(pos, ctx_a)) != pytest.approx(
        float(IoNormTerm(io).value(pos, ctx_b)))
    assert float(FtNormTerm(ft).value(pos, ctx_a)) != pytest.approx(
        float(FtNormTerm(ft).value(pos, ctx_b)))
    # value(pos, ctx) is unweighted (lambda_io=1): io(pos, tau, 2.0) is 2x it.
    assert float(IoNormTerm(io).value(pos, ctx_a)) == pytest.approx(
        float(io(pos, 8.0, 1.0)), rel=1e-12)
    assert float(io(pos, 8.0, 2.0)) == pytest.approx(
        2.0 * float(IoNormTerm(io).value(pos, ctx_a)), rel=1e-12)
    assert float(FtNormTerm(ft).value(pos, ctx_a)) == pytest.approx(
        float(ft.ft_only(pos, 8.0)), rel=1e-12)


def test_probe_terms_subset_makes_cancellation_ratio_none():
    # Fix round 1, I-2: a partial (`probe_terms=`) probe caches gradients for
    # only the measured subset. If a later transaction's lambdas give a
    # nonzero coefficient to a term outside that subset, the cancellation
    # ratio's numerator/denominator would silently miss that term's force --
    # so cancellation_ratio must report None rather than a wrong number.
    torch, n, pos, wl_fn, ctx = _probe_setup()
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("b", _Poly(-0.5), 1.0, activate_overflow=0.90, n_ramp=0)
    n.probe(0, pos, wl_fn, ctx)
    n.mark_refreshed()
    n.probe(50, pos, wl_fn, ctx, probe_terms=["a"])
    txn = n.transaction(50, 0.85, tau=1000.0, gamma=0.0)
    assert txn.cancellation_ratio is None


def test_probe_requires_num_movable_and_num_nodes():
    # Fix round 1, I-3: without num_movable/num_nodes the fixed/filler mask
    # silently becomes a no-op (`_grad`'s guard is `is not None`), so probe()
    # must refuse to run rather than silently return polluted norms.
    torch = pytest.importorskip("torch")
    n = TermNormalizer(policy="grandplan", wt0=0.05)
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    pos = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    wl_fn = lambda p: (p ** 3).sum()
    ctx = {"iteration": 0, "overflow": 0.85, "tau": 1000.0, "gamma": 0.0}
    with pytest.raises(ValueError):
        n.probe(0, pos, wl_fn, ctx)


def test_probe_rejects_unknown_probe_terms():
    torch, n, pos, wl_fn, ctx = _probe_setup()
    n.register("a", _Poly(1.0), 1.0, activate_overflow=0.90, n_ramp=0)
    with pytest.raises(KeyError):
        n.probe(0, pos, wl_fn, ctx, probe_terms=["bogus"])


def test_probe_drives_the_real_io_and_ft_norm_terms():
    torch = pytest.importorskip("torch")
    from ioplace.ops.ft_term import FtTerm
    from ioplace.ops.norm_terms import FtNormTerm, IoNormTerm
    from tests.test_ft_term import _make, _pos
    nl, io, _, distance = _make(k=4, chunk=1)
    ft = FtTerm(io, distance)
    ft.set_home([0])
    pos = _pos(nl)
    pos.grad = torch.zeros_like(pos)
    n = TermNormalizer(policy="grandplan", num_movable=io.num_movable,
                       num_nodes=io.num_nodes, wt0=0.05)
    n.register("io", IoNormTerm(io), 1.0, activate_overflow=0.90, n_ramp=0)
    n.register("ft", FtNormTerm(ft), 1.0, activate_overflow=0.90, n_ramp=0)
    wl_fn = lambda p: (p ** 2).sum()
    ctx = {"iteration": 0, "overflow": 0.5, "tau": 8.0, "gamma": 1.0}
    norms = n.probe(0, pos, wl_fn, ctx)
    for name in ("wl", "io", "ft"):
        assert math.isfinite(norms[name]) and norms[name] > 0.0
    assert bool(torch.all(pos.grad == 0.0))
