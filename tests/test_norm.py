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


def test_grandplan_lambda_scales_ratio_and_guards_dead_terms():
    assert grandplan_lambda(0.05, 100.0, 1000.0, 10.0) == pytest.approx(5.0, rel=1e-12)
    # grad_norm <= eps_rel * wl_norm -> the term has no local signal
    assert grandplan_lambda(0.05, 1e9, 1000.0, 1.0) == 0.0
    assert grandplan_lambda(0.05, 1e9, 1000.0, 0.0) == 0.0


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
    assert adaptive_lambda(1.0, 0.5, 0.2, 100.0, 1000.0) == 0.0


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


def test_weights_rejects_the_legacy_policy_until_the_adapter_lands():
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
    n = _norm_a()
    n.update_grad_norms({"wl": 1000.0, "io": 0.0})
    assert math.isfinite(n.states["io"].ratio_ema)
    assert n.states["io"].ratio_ema == pytest.approx(1e33)   # wl_norm / EPS


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


def test_activation_ramp_is_shared_by_the_policy():
    n = TermNormalizer(policy="grandplan")
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=20)
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    txn = n.transaction(0, 0.85, tau=1000.0, gamma=1e-12)
    assert txn.lambdas["io"] == 0.0                            # ramp(0, 0, 20) == 0
    n.mark_refreshed()
    txn = n.transaction(10, 0.85, tau=1000.0, gamma=1e-12)
    assert txn.lambdas["io"] == pytest.approx(0.05 * 0.5 * 100.0, rel=1e-12)


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
