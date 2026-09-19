import math
import pytest

from ioplace.norm import (EPS, adaptive_lambda, clip_sum_to_cap, cmax_from_curvatures,
                          ema_update, grandplan_lambda, grandplan_weight,
                          parse_target_shares)


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


def test_cmax_generalises_derive_cmax():
    from ioplace.schedules import derive_cmax
    assert cmax_from_curvatures([]) == 1.0
    assert cmax_from_curvatures([(1.0, 1.0)]) == 1.0
    assert cmax_from_curvatures([(1.0, 1.0), (1.0, 6.0)]) == pytest.approx(6.0, rel=1e-12)
    assert cmax_from_curvatures([(0.5, 6.0)]) == pytest.approx(3.5, rel=1e-12)
    assert cmax_from_curvatures([(2.0, 0.5)]) == 1.0            # (curv-1)_+ hinge
    assert cmax_from_curvatures([(2.0, 6.0)]) == pytest.approx(derive_cmax(2.0, 6.0), rel=1e-12)


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
