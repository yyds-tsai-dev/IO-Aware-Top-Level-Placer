import pytest
from ioplace.schedules import (tau_rel_from_overflow, rho_from_overflow, activation_ramp,
                               lipschitz_cap, ScheduleState)

def test_tau_endpoints_and_monotonicity():
    assert tau_rel_from_overflow(0.90) == pytest.approx(0.30, rel=1e-12)
    assert tau_rel_from_overflow(0.07) == pytest.approx(0.03, rel=1e-12)
    assert tau_rel_from_overflow(1.50) == pytest.approx(0.30, rel=1e-12)   # clipped high
    assert tau_rel_from_overflow(0.00) == pytest.approx(0.03, rel=1e-12)   # clipped low
    xs = [0.9, 0.7, 0.5, 0.3, 0.15, 0.07]
    ys = [tau_rel_from_overflow(x) for x in xs]
    assert all(a > b for a, b in zip(ys, ys[1:]))

def test_tau_is_log_linear_in_overflow():
    import math
    mid = (0.90 + 0.07) / 2
    assert tau_rel_from_overflow(mid) == pytest.approx(math.sqrt(0.30 * 0.03), rel=1e-12)

def test_rho_ramps_from_zero_to_rho_max():
    assert rho_from_overflow(0.95, 0.2) == pytest.approx(0.0, abs=1e-15)
    assert rho_from_overflow(0.90, 0.2) == pytest.approx(0.0, abs=1e-15)
    assert rho_from_overflow(0.20, 0.2) == pytest.approx(0.2, rel=1e-12)
    assert rho_from_overflow(0.07, 0.2) == pytest.approx(0.2, rel=1e-12)
    assert rho_from_overflow(0.55, 0.2) == pytest.approx(0.1, rel=1e-12)

def test_activation_ramp_bounds():
    assert activation_ramp(100, 100) == pytest.approx(0.0)
    assert activation_ramp(110, 100, n_ramp=20) == pytest.approx(0.5)
    assert activation_ramp(120, 100, n_ramp=20) == pytest.approx(1.0)
    assert activation_ramp(500, 100, n_ramp=20) == pytest.approx(1.0)
    assert activation_ramp(90, 100, n_ramp=20) == pytest.approx(0.0)

def test_lipschitz_cap_formula():
    assert lipschitz_cap(53.4, 14.3) == pytest.approx(53.4 ** 2 / 14.3, rel=1e-12)
    assert lipschitz_cap(53.4, 14.3, c_lip=0.5) == pytest.approx(0.5 * 53.4 ** 2 / 14.3, rel=1e-12)
    assert lipschitz_cap(10.0, 0.0) == float("inf")

def test_state_activates_once_and_reports_discrete_event():
    s = ScheduleState(rho_max=0.1)
    assert s.update_continuous(0, overflow=0.99, L_R=1000.0, gamma=100.0) is False
    assert s.active is False and s.lambda_io == 0.0
    assert s.update_continuous(10, overflow=0.85, L_R=1000.0, gamma=100.0) is True   # activation
    assert s.active is True and s.it_activate == 10
    assert s.update_continuous(11, overflow=0.84, L_R=1000.0, gamma=100.0) is False  # no re-fire

def test_state_tau_tracks_overflow_every_iteration():
    s = ScheduleState(rho_max=0.1)
    s.update_continuous(10, 0.85, 1000.0, 100.0)
    t1 = s.tau
    s.update_continuous(11, 0.60, 1000.0, 100.0)
    assert s.tau < t1
    assert s.tau == pytest.approx(tau_rel_from_overflow(0.60) * 1000.0, rel=1e-12)

def test_lambda_is_zero_until_ratio_known_then_ramped():
    s = ScheduleState(rho_max=0.4, n_ramp=20)
    s.update_continuous(10, 0.85, 1000.0, 1e-12)   # gamma tiny -> cap ~ tau^2/gamma huge -> inactive
    assert s.lambda_io == 0.0                       # no ratio yet
    s.update_ratio(g_wl_l1=1000.0, g_io_l1=2.0)     # ratio = 500
    assert s.ratio_ema == pytest.approx(500.0, rel=1e-12)
    s.update_continuous(20, 0.55, 1000.0, 1e-12)    # rho = 0.4*0.5 = 0.2 ; ramp = 0.5
    assert s.lambda_io == pytest.approx(0.4 * 0.5 * 0.5 * 500.0, rel=1e-9)

def test_ratio_uses_ema_damping():
    s = ScheduleState(rho_max=0.1, ema=0.5)
    s.update_continuous(10, 0.85, 1000.0, 1e9)
    s.update_ratio(1000.0, 1.0)      # 1000
    s.update_ratio(1000.0, 10.0)     # raw 100 -> ema 0.5*1000 + 0.5*100 = 550
    assert s.ratio_ema == pytest.approx(550.0, rel=1e-12)

def test_lipschitz_cap_binds_lambda():
    s = ScheduleState(rho_max=1.0, n_ramp=0)
    s.update_continuous(10, 0.85, 1000.0, gamma=1.0)
    s.update_ratio(1e9, 1.0)
    s.update_continuous(11, 0.07, 1000.0, gamma=1.0)
    assert s.lambda_io == pytest.approx(lipschitz_cap(s.tau, 1.0), rel=1e-9)

def test_obj_version_only_bumps_on_discrete_events():
    s = ScheduleState(rho_max=0.1)
    s.update_continuous(10, 0.85, 1000.0, 1e9)      # activation -> discrete
    v0 = s.obj_version
    assert v0 == 1
    for it in range(11, 30):
        s.update_continuous(it, 0.80, 1000.0, 1e9)  # continuous drift only
    assert s.obj_version == v0
    s.update_ratio(1000.0, 2.0)
    assert s.obj_version == v0 + 1

def test_needs_refresh_and_mark_refreshed():
    s = ScheduleState(rho_max=0.1)
    assert s.needs_refresh() is False
    s.update_continuous(10, 0.85, 1000.0, 1e9)
    assert s.needs_refresh() is True
    s.mark_refreshed()
    assert s.needs_refresh() is False

def test_margin_toggle_is_a_discrete_event():
    s = ScheduleState(rho_max=0.1, rho_margin=0.05, of_margin=0.15)
    s.update_continuous(10, 0.85, 1000.0, 1e9); s.mark_refreshed()
    s.update_ratio(1000.0, 1.0); s.mark_refreshed()
    assert s.lambda_margin == 0.0
    assert s.update_continuous(100, 0.10, 1000.0, 1e9) is True     # margin switches on
    assert s.lambda_margin > 0.0
    assert s.update_continuous(101, 0.09, 1000.0, 1e9) is False

def test_zero_io_grad_does_not_divide_by_zero():
    s = ScheduleState(rho_max=0.1)
    s.update_continuous(10, 0.85, 1000.0, 1e9)
    s.update_ratio(g_wl_l1=1000.0, g_io_l1=0.0)
    assert s.ratio_ema is not None and s.ratio_ema == s.ratio_ema  # not NaN
