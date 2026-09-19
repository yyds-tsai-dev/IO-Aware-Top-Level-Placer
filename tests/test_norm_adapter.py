import json

import pytest
torch = pytest.importorskip("torch")

from ioplace.norm_adapter import (DEFAULT_IO_TARGET_SHARE,
                                  FT_ACTIVATE_OVERFLOW, NORM_POLICIES,
                                  LegacyNormAdapter, TermNormalizerAdapter,
                                  make_norm_adapter)
from ioplace.norm_trace import ROW_FIELDS, read_norm_trace


class _FakeIoTerm:
    num_movable, num_nodes = 2, 3

    def __call__(self, pos, tau, lambda_io, *args):
        return lambda_io * (pos ** 2).sum()


class _DeadIoTerm(_FakeIoTerm):
    """An IO term with an exactly-zero gradient -- the only thing the P-H fix
    wave (controller ruling F1') still classifies as dead."""

    def __call__(self, pos, tau, lambda_io, *args):
        return lambda_io * (pos * 0.0).sum()


class _FakeFtTerm:
    """`FtNormTerm` measures `ft_only(pos, tau)`; `FtTerm.forward`'s own
    signature is the driver's business, not the normalizer's."""

    def ft_only(self, pos, tau):
        return (pos ** 2).sum()


def _wirelength(pos):
    return (3.0 * pos).abs().sum()


def _pos():
    # x = pos[:3], y = pos[3:]; entries 2 and 5 are the fixed/filler slots the
    # normalizer's mask zeroes, so ||grad WL||_1 = 4*3 = 12 and
    # ||grad IO||_1 = |2*[1,2,4,5]| = 24 for every probe below.
    return torch.arange(6, dtype=torch.float64) + 1.0


def _config(**override):
    config = dict(L_R=100.0, rho_max=0.1, tau_hi=0.30, tau_lo=0.03, of_on=2.0,
                  of_end=0.5, of_full=1.0, f_ft_max=0.0, ft_ramp_mode="window",
                  tau_start=0.12, tau_full=0.05, io_term=_FakeIoTerm())
    config.update(override)
    return config


def _probe(adapter, iteration, config=None):
    config = _config() if config is None else config
    return adapter.probe(iteration, _pos(), io_term=config["io_term"],
                         ft_term=config.get("ft_term"),
                         wirelength_op=_wirelength,
                         ecc_max=config.get("ecc_max", 0.0), gamma=1.0)


def _activated(policy, **override):
    """Activate at iteration 0 so policy A's 20-iteration activation ramp is
    complete by the first probe at 50. Without the adapter's activation sync
    `io` would latch at the probe itself and its applied lambda would be
    exactly 0 there."""
    config = _config(**override)
    adapter = make_norm_adapter(policy, **config)
    adapter.begin_iteration(0, overflow=1.5, gamma=1.0)
    adapter.mark_refreshed()
    adapter.begin_iteration(50, overflow=1.5, gamma=1.0)
    return adapter, config


def test_policies_are_declared():
    assert NORM_POLICIES == ("legacy", "grandplan", "adaptive")


def test_legacy_adapter_activates_once_and_reports_tau():
    adapter = make_norm_adapter("legacy", **_config())
    assert adapter.policy == "legacy"
    assert adapter.begin_iteration(0, overflow=3.0, gamma=1.0) is False
    assert adapter.active is False and adapter.lambda_io == 0.0
    assert adapter.begin_iteration(10, overflow=1.5, gamma=1.0) is True
    assert adapter.active is True
    assert adapter.begin_iteration(20, overflow=1.5, gamma=1.0) is False
    assert adapter.tau == pytest.approx(adapter.tau_rel * 100.0)


def test_probe_bumps_obj_version_exactly_once_and_needs_a_refresh():
    adapter, config = _activated("legacy")
    before = adapter.obj_version
    row = _probe(adapter, 50, config)
    assert adapter.obj_version == before + 1
    assert adapter.needs_refresh() is True
    assert adapter.refreshed_version != adapter.obj_version
    adapter.mark_refreshed()
    assert adapter.needs_refresh() is False
    assert adapter.refreshed_version == adapter.obj_version
    row = adapter.write_trace_row(row, {"iteration": 50, "policy": "legacy"})
    for key in ("grad_l1_wl", "grad_l1_io", "grad_l1_ft", "ratio_ema",
                "lambda_io", "obj_version", "tau", "tau_rel", "kappa_ft"):
        assert key in row
    assert row["grad_l1_wl"] > 0 and row["grad_l1_io"] > 0
    assert adapter.lambda_io > 0


def test_unknown_policy_and_unknown_knob_are_rejected():
    with pytest.raises(ValueError, match="norm policy"):
        make_norm_adapter("bogus", **_config())
    with pytest.raises(TypeError, match="unexpected keyword"):
        make_norm_adapter("legacy", bogus_knob=1, **_config())
    with pytest.raises(ValueError, match="LegacyNormAdapter"):
        TermNormalizerAdapter("legacy", **_config())


def test_flag_combinations_the_landed_driver_rejects_are_rejected_here():
    """The `run_io` validation block (run_placement_io.py:237-265), minus the
    two rules the main flow cannot reach: it has no `--rho-margin` and no
    `--callback-order`, and its probe cadence *is* `--every`, so
    `probe_every % every` holds by construction."""
    with pytest.raises(ValueError, match="norm_p must be 1"):
        make_norm_adapter("legacy", norm_p=2, **_config())
    with pytest.raises(ValueError, match="rho-max"):
        make_norm_adapter("grandplan", **_config(rho_max=0.0))
    with pytest.raises(ValueError, match="io_term"):
        make_norm_adapter("grandplan", **_config(io_term=None))
    with pytest.raises(ValueError, match="unregistered"):
        make_norm_adapter("adaptive", target_shares="io=0.3,ft=0.05",
                          **_config())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_norm_adapter("adaptive", target_shares="io=1.5", **_config())


def test_the_ft_activation_gate_matches_the_legacy_driver():
    """The constant is duplicated to keep this module free of DREAMPlace
    imports; this is the lock that stops the copy from drifting."""
    run_io = pytest.importorskip("ioplace.drivers.run_placement_io")
    assert FT_ACTIVATE_OVERFLOW == run_io.FT_ACTIVATE_OVERFLOW


def test_grandplan_adapter_normalises_a_positive_lambda_io(tmp_path):
    """Policy A end to end: one probe of a term with a real positive gradient
    must leave a non-zero lambda_io and one schema-valid norm_trace row."""
    adapter, config = _activated("grandplan", out_dir=str(tmp_path))
    assert isinstance(adapter, TermNormalizerAdapter)
    row = _probe(adapter, 50, config)
    assert adapter.lambda_io > 0.0
    assert adapter.kappa_ft == 0.0                      # no FT term registered
    assert row["policy"] == "grandplan" and set(row) == set(ROW_FIELDS)
    assert row["grad_l1_wl"] == pytest.approx(12.0)
    assert row["terms"]["io"]["grad_l1"] == pytest.approx(24.0)
    assert row["terms"]["io"]["ratio_ema"] == pytest.approx(0.5)
    # wt = grandplan_weight(50, it_activate=0, wt0=0.05, ramp_period=100) =
    # 0.05, so the committed lam = wt * ratio_ema = 0.025, well under the
    # Lipschitz cap; the activation ramp is complete at iteration 50, so the
    # applied lambda equals it.
    assert row["terms"]["io"]["wt"] == pytest.approx(0.05)
    assert row["terms"]["io"]["lam"] == pytest.approx(0.025)
    assert row["terms"]["io"]["lam_applied"] == pytest.approx(0.025)
    assert adapter.lambda_io == pytest.approx(0.025)
    # the row reaches disk only once the Nesterov cache has been refreshed
    assert not (tmp_path / "norm_trace.jsonl").read_text()
    adapter.mark_refreshed()
    adapter.close()
    rows = read_norm_trace(str(tmp_path / "norm_trace.jsonl"))
    assert len(rows) == 1
    assert rows[0]["obj_version"] == rows[0]["refreshed_version"]


def test_io_activates_from_the_schedule_not_from_the_first_probe():
    """Review I1a. `ScheduleState` latches on every iteration; the normalizer's
    own gate would only run on the probe cadence, putting `it_activate` up to
    `probe_every` iterations late and aliasing the whole n_ramp soft start."""
    config = _config(io_term=_FakeIoTerm())
    adapter = make_norm_adapter("grandplan", **config)
    adapter.begin_iteration(0, overflow=3.0, gamma=1.0)       # above of_on
    assert adapter.normalizer.states["io"].active is False
    adapter.begin_iteration(7, overflow=1.5, gamma=1.0)       # schedule latches
    assert adapter.state.it_activate == 7
    assert adapter.normalizer.states["io"].it_activate == 7
    for iteration in (8, 30, 50):
        adapter.begin_iteration(iteration, overflow=1.5, gamma=1.0)
    assert adapter.normalizer.states["io"].it_activate == 7   # monotone


def test_the_applied_lambda_drifts_through_the_ramp_between_transactions(tmp_path):
    """Review I1b: `transaction()` commits an un-ramped lambda every
    `probe_every` iterations and `applied_lambda` re-scales it every
    iteration, so `adapter.lambda_io` moves without a new probe -- the same
    continuous drift the legacy arm gets from `update_continuous`."""
    config = _config(out_dir=str(tmp_path), io_term=_FakeIoTerm())
    adapter = make_norm_adapter("grandplan", **config)
    adapter.begin_iteration(50, overflow=1.5, gamma=1.0)      # activates at 50
    assert adapter.lambda_io == 0.0                           # ramp(50, 50) == 0
    _probe(adapter, 50, config)
    adapter.mark_refreshed()
    committed = adapter.normalizer.lambdas["io"]
    assert committed == pytest.approx(0.025)
    for iteration, fraction in ((55, 0.25), (60, 0.5), (70, 1.0), (90, 1.0)):
        adapter.begin_iteration(iteration, overflow=1.5, gamma=1.0)
        assert adapter.lambda_io == pytest.approx(committed * fraction)
        assert adapter.normalizer.lambdas["io"] == committed   # no new commit
    assert adapter.needs_refresh() is False                    # drift is not a bump
    adapter.close()


def test_ft_is_dependent_gated_and_ceilinged(tmp_path):
    """The three FT registration details the landed `_register_norm_terms`
    carries: `requires="io"`, its own `FT_ACTIVATE_OVERFLOW` gate, and the
    `f_ft_max * wt_max` weight ceiling under grandplan."""
    config = _config(out_dir=str(tmp_path), f_ft_max=0.2,
                     ft_term=_FakeFtTerm(), ecc_max=3.0)
    adapter = make_norm_adapter("grandplan", **config)
    assert adapter.normalizer.configs["ft"].requires == "io"
    assert adapter.normalizer.configs["ft"].wt_max == pytest.approx(0.2)
    assert adapter.normalizer.configs["ft"].curvature == pytest.approx(3.0)
    assert adapter.normalizer.configs["io"].wt_max is None    # normalizer-wide 1.0

    adapter.begin_iteration(0, overflow=1.5, gamma=1.0)       # io on, ft off
    assert adapter.normalizer.states["ft"].active is False
    adapter.mark_refreshed()
    adapter.begin_iteration(50, overflow=0.2, gamma=1.0)      # below 0.30
    assert adapter.normalizer.states["ft"].it_activate == 50
    row = _probe(adapter, 50, config)
    # both terms measure ||grad||_1 = 24 against ||grad WL||_1 = 12, so both
    # ratios are 0.5 and both weights are the first grandplan step 0.05
    assert row["terms"]["ft"]["wt_max"] == pytest.approx(0.2)
    assert row["terms"]["ft"]["lam"] == pytest.approx(0.025)
    # cmax = (0.025*1 + 0.025*3)/0.05 = 2, tau = tau_lo*L_R = 3, so the cap is
    # 3^2/2 = 4.5 and does not bind on a total lambda of 0.05
    assert row["cmax"] == pytest.approx(2.0)
    assert row["cap"] == pytest.approx(4.5) and row["cap_binding"] is None
    # ft's ramp starts at 50, io's at 0, so the applied kappa is still 0 here
    assert row["terms"]["ft"]["lam_applied"] == pytest.approx(0.0)
    assert adapter.kappa_ft == pytest.approx(0.0)
    adapter.mark_refreshed()
    adapter.begin_iteration(70, overflow=0.2, gamma=1.0)      # ft fully ramped
    assert adapter.kappa_ft == pytest.approx(1.0)
    assert adapter.lambda_io * adapter.kappa_ft == pytest.approx(0.025)
    adapter.close()


def test_a_dead_io_gradient_zeroes_the_dependent_ft_term(tmp_path):
    """Review C1: `FtTerm` applies `lambda_io * kappa`, so FT is unapplicable
    once lambda_io is 0. The normalizer publishes 0 for it rather than letting
    the driver assert."""
    config = _config(out_dir=str(tmp_path), io_term=_DeadIoTerm(),
                     f_ft_max=0.2, ft_term=_FakeFtTerm(), ecc_max=3.0)
    adapter = make_norm_adapter("grandplan", **config)
    adapter.begin_iteration(0, overflow=0.2, gamma=1.0)
    adapter.mark_refreshed()
    adapter.begin_iteration(50, overflow=0.2, gamma=1.0)
    row = _probe(adapter, 50, config)
    assert row["terms"]["io"]["grad_l1"] == 0.0
    assert row["terms"]["io"]["lam"] == 0.0
    assert row["terms"]["ft"]["grad_l1"] == pytest.approx(24.0)
    assert row["terms"]["ft"]["lam"] == 0.0                   # requires="io"
    assert adapter.lambda_io == 0.0 and adapter.kappa_ft == 0.0
    adapter.mark_refreshed()
    adapter.close()


def test_term_normalizer_adapter_bumps_obj_version_once_per_probe(tmp_path):
    adapter, config = _activated("grandplan", out_dir=str(tmp_path))
    assert adapter.needs_refresh() is False
    before = adapter.obj_version
    _probe(adapter, 50, config)
    # VersionPair sums the ScheduleState's counter and the normalizer's, so one
    # transaction is one bump even though two objects carry versions.
    assert adapter.obj_version == before + 1
    assert adapter.needs_refresh() is True
    adapter.mark_refreshed()
    assert adapter.needs_refresh() is False
    assert adapter.refreshed_version == adapter.obj_version
    adapter.begin_iteration(100, overflow=1.5, gamma=1.0)
    _probe(adapter, 100, config)
    assert adapter.obj_version == before + 2
    adapter.mark_refreshed()
    adapter.close()
    assert len(read_norm_trace(str(tmp_path / "norm_trace.jsonl"))) == 2


def test_a_transaction_without_a_probe_reuses_the_previous_measurements(tmp_path):
    """`probe_every` gates the gradient probe, not the transaction: the
    landed driver's `if normalizer.should_probe(iteration)` (:657). The main
    flow's forced last-iteration callback lands off the cadence and must not
    re-measure."""
    config = _config(out_dir=str(tmp_path), probe_every=50)
    adapter = make_norm_adapter("grandplan", **config)
    adapter.begin_iteration(0, overflow=1.5, gamma=1.0)
    adapter.mark_refreshed()
    adapter.begin_iteration(50, overflow=1.5, gamma=1.0)
    first = _probe(adapter, 50, config)
    adapter.mark_refreshed()
    adapter.begin_iteration(77, overflow=1.5, gamma=1.0)
    second = _probe(adapter, 77, config)
    adapter.mark_refreshed()
    adapter.close()
    assert second["probe_iteration"] == 50 and second["iteration"] == 77
    assert second["terms"]["io"]["ratio_ema"] == first["terms"]["io"]["ratio_ema"]
    assert len(read_norm_trace(str(tmp_path / "norm_trace.jsonl"))) == 2


def test_adaptive_policy_constructs_and_bootstraps_its_coefficient(tmp_path):
    """Policy B: `io` defaults to `DEFAULT_IO_TARGET_SHARE` and `ft` to
    `f_ft_max`, the landed driver's defaults (run_placement_io.py:102-111), and
    the first update takes `adaptive_lambda`'s bootstrap branch."""
    adapter, config = _activated("adaptive", out_dir=str(tmp_path))
    assert adapter.policy == "adaptive" and adapter.normalizer.policy == "adaptive"
    assert adapter.target_shares == {"io": 0.3}      # no FT term registered
    _probe(adapter, 50, config)
    # bootstrap: target_share * ||grad WL|| / ||grad T|| = 0.3 * 12 / 24
    assert adapter.lambda_io == pytest.approx(0.15)
    adapter.mark_refreshed()
    adapter.close()
    override = make_norm_adapter("adaptive", target_shares="io=0.3,ft=0.05",
                                 **_config(f_ft_max=0.2, ft_term=_FakeFtTerm(),
                                           ecc_max=3.0))
    assert override.target_shares == {"io": 0.3, "ft": 0.05}
    assert override.trace_path is None


def test_each_policy_writes_its_own_trace_file(tmp_path):
    """A-9: one schema per file name. norm_trace.jsonl is the design sec 4
    schema NormTraceWriter validates; the legacy row is publish_atomic's own
    dict and goes to legacy_trace.jsonl."""
    legacy_dir, norm_dir = str(tmp_path / "legacy"), str(tmp_path / "grandplan")
    legacy, legacy_config = _activated("legacy", out_dir=legacy_dir)
    assert legacy.trace_path.endswith("legacy_trace.jsonl")
    written = legacy.write_trace_row(_probe(legacy, 50, legacy_config),
                                     {"io_count": 7, "churn": 0.0})
    legacy.mark_refreshed()
    legacy.close()
    lines = [json.loads(line) for line
             in open(legacy.trace_path).read().splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0]["io_count"] == 7
    assert lines[0]["grad_l1_io"] > 0 and written["policy"] == "legacy"
    assert not (tmp_path / "legacy" / "norm_trace.jsonl").exists()

    grandplan, gp_config = _activated("grandplan", out_dir=norm_dir)
    assert grandplan.trace_path.endswith("norm_trace.jsonl")
    # the driver's extras must not contaminate the validated schema
    assert grandplan.write_trace_row(_probe(grandplan, 50, gp_config),
                                     {"io_count": 7, "churn": 0.0}) is None
    grandplan.mark_refreshed()
    grandplan.close()
    rows = read_norm_trace(grandplan.trace_path)
    assert len(rows) == 1 and set(rows[0]) == set(ROW_FIELDS)
    assert not (tmp_path / "grandplan" / "legacy_trace.jsonl").exists()


def test_the_default_io_target_share_matches_the_legacy_driver():
    """Same drift lock as the FT gate above: policy B's default IO force share
    is duplicated here so this module stays DREAMPlace-free, and the two arms
    are only comparable while the two copies agree."""
    run_io = pytest.importorskip("ioplace.drivers.run_placement_io")
    assert DEFAULT_IO_TARGET_SHARE == run_io.DEFAULT_IO_TARGET_SHARE


def test_an_ft_term_without_an_ft_ceiling_is_refused():
    """Task 5 review (folded into Task 7): under `grandplan`, `f_ft_max <= 0`
    makes the `f_ft_max * wt_max` ceiling 0, which `_register` drops -- FT
    would then ramp to the normalizer-wide wt_max under a flag that reads "FT
    off". `run_main_flow` never builds an ft_term with f_ft_max <= 0; any
    caller that does is asking for two contradictory things."""
    for policy in ("grandplan", "adaptive"):
        for f_ft_max in (0.0, -1.0):
            with pytest.raises(ValueError, match="f_ft_max"):
                make_norm_adapter(policy, **_config(f_ft_max=f_ft_max,
                                                    ft_term=_FakeFtTerm(),
                                                    ecc_max=3.0))
    # the same configuration with a positive ceiling still constructs
    assert make_norm_adapter("grandplan", **_config(
        f_ft_max=0.2, ft_term=_FakeFtTerm(), ecc_max=3.0)).target_shares == {
            "io": 0.3, "ft": 0.2}


def test_the_legacy_adapter_is_bit_exact_against_a_bare_schedule_state():
    """The adapter's whole justification is that `--norm-policy legacy`
    reproduces the pre-v2 coefficients *exactly*, not approximately. Drive a
    bare `ScheduleState` + `publish_atomic` through the same call sequence the
    adapter makes and compare with `==`, not `approx`."""
    from ioplace.ops.ft_callback import publish_atomic
    from ioplace.schedules import ScheduleState

    config = _config(f_ft_max=0.2, ft_term=_FakeFtTerm(), ecc_max=3.0)
    adapter = make_norm_adapter("legacy", **config)
    assert isinstance(adapter, LegacyNormAdapter)
    adapter.begin_iteration(0, overflow=1.5, gamma=1.0)
    adapter.mark_refreshed()
    adapter.begin_iteration(50, overflow=0.9, gamma=1.0)
    row = _probe(adapter, 50, config)

    # `_ScheduleBacked.__init__`'s ScheduleState, spelled out.
    state = ScheduleState(rho_max=0.1, tau_hi=0.30, tau_lo=0.03, of_on=2.0,
                          of_end=0.5, of_full=1.0, n_ramp=20, c_lip=1.0,
                          f_ft_max=0.2, ft_ramp_mode="window", tau_start=0.12,
                          tau_full=0.05, kappa_max=100.0, eps_rel=1e-3, ema=0.5)
    state.update_continuous(0, 1.5, 100.0, 1.0)
    state.mark_refreshed()
    state.update_continuous(50, 0.9, 100.0, 1.0)
    reference = publish_atomic(state, _FakeIoTerm(), _FakeFtTerm(), _wirelength,
                               _pos(), 50, state.tau / 100.0, 3.0, 1.0)

    assert adapter.lambda_io == state.lambda_io
    assert adapter.kappa_ft == state.kappa_ft
    assert adapter.tau == state.tau and adapter.tau_rel == state.tau / 100.0
    assert adapter.obj_version == state.obj_version
    # not a vacuous 0 == 0: this iteration has both coefficients switched on
    assert adapter.lambda_io > 0.0 and adapter.kappa_ft > 0.0
    for key, value in reference.items():
        assert row[key] == value, key
