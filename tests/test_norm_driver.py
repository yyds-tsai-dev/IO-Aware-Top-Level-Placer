import json
import os

import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers import run_placement
from ioplace.drivers.run_placement_io import RESULT_FIELDS, run_io
from ioplace.norm_trace import read_norm_trace

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")


def test_norm_fields_are_in_the_result_contract():
    for field in ("norm_policy", "norm_p", "norm_ramp_period", "norm_wt_max",
                  "norm_probe_every", "norm_target_share", "norm_trace",
                  "lambda_ft_final"):
        assert field in RESULT_FIELDS


def test_norm_flag_defaults_match_the_design():
    args = run_placement.build_parser().parse_args(
        ["--config", "c.json", "--mode", "io", "--out", "o.json"])
    assert args.norm_policy == "legacy"
    assert args.norm_p == 1
    assert args.norm_ramp_period == 100
    assert args.norm_wt_max == 1.0
    assert args.norm_probe_every == 50
    assert args.norm_target_share is None
    assert args.norm_trace is None


def test_norm_flags_appear_in_the_driver_help():
    text = run_placement.build_parser().format_help()
    for flag in ("--norm-policy", "--norm-p", "--norm-ramp-period",
                 "--norm-wt-max", "--norm-probe-every", "--norm-target-share",
                 "--norm-trace"):
        assert flag in text


def test_norm_flags_are_forwarded_to_run_io(monkeypatch):
    captured = {}

    def fake_run_io(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("ioplace.drivers.run_placement_io.run_io", fake_run_io)
    monkeypatch.setattr("sys.argv", [
        "run_placement", "--config", "c.json", "--mode", "io", "--out", "o.json",
        "--callback-order", "atomic", "--norm-policy", "adaptive", "--norm-p", "2",
        "--norm-ramp-period", "250", "--norm-wt-max", "0.4",
        "--norm-probe-every", "100", "--norm-target-share", "io=0.3,ft=0.1",
        "--norm-trace", "t.jsonl"])
    run_placement.main()
    assert captured["norm_policy"] == "adaptive"
    assert captured["norm_p"] == 2
    assert captured["norm_ramp_period"] == 250
    assert captured["norm_wt_max"] == 0.4
    assert captured["norm_probe_every"] == 100
    assert captured["norm_target_share"] == "io=0.3,ft=0.1"
    assert captured["norm_trace"] == "t.jsonl"


@pytest.mark.parametrize("kwargs,message", [
    (dict(norm_policy="bogus"), "norm_policy"),
    (dict(norm_p=3), "norm_p"),
    (dict(norm_policy="legacy", norm_p=2), "norm_p"),       # legacy measures L1 only
    (dict(norm_probe_every=0), "norm_probe_every"),
    (dict(norm_probe_every=75, norm_policy="grandplan"), "norm_probe_every"),  # not a multiple of every=50
    (dict(norm_policy="grandplan"), "atomic"),              # callback_order defaults legacy
    (dict(norm_policy="grandplan", callback_order="atomic", rho_max=0.0),
     "observer mode"),                                      # rho_max=0/rho_margin=0/wl_reweight=off
    # Review I2: rho_max never reaches the normalizer, so `--rho-max 0` (which
    # under legacy is a reweight-only run) silently turned the IO penalty on at
    # full strength instead.
    (dict(norm_policy="grandplan", callback_order="atomic", rho_max=0.0,
          wl_reweight="crossings"), "rho_max is inert"),
    (dict(norm_policy="adaptive", callback_order="atomic", rho_max=0.0,
          wl_reweight="crossings"), "rho_max is inert"),
    # Review I3: ScheduleState.ratio_ema is never written on a non-legacy arm,
    # so the margin term evaluates to 0.0 forever while --rho-margin is still
    # echoed into the result JSON.
    (dict(norm_policy="grandplan", callback_order="atomic", rho_max=0.4,
          rho_margin=0.05), "rho_margin is inert"),
    (dict(norm_policy="adaptive", callback_order="atomic", rho_max=0.4,
          rho_margin=0.05), "rho_margin is inert"),
])
def test_run_io_rejects_invalid_norm_configuration(kwargs, message, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        run_io("missing.json", 4, "grid", 0, str(tmp_path / "o.json"), **kwargs)
    assert message in str(excinfo.value)


def test_ft_is_registered_as_a_dependent_term_with_a_mirrored_wt_ceiling():
    """Reviews C1/I4/I6, on the driver's own registration contract (no GPU
    needed). `ft` must declare `requires="io"` so a dead IO probe degrades to
    lambda_ft=0 instead of the RuntimeError that used to kill the run, and
    under grandplan its weight ceiling must mirror legacy's f_ft_max rather
    than converging to IO's."""
    from ioplace.drivers.run_placement_io import (DEFAULT_IO_TARGET_SHARE,
                                                  FT_ACTIVATE_OVERFLOW,
                                                  _register_norm_terms)
    from ioplace.norm import TermNormalizer

    n = TermNormalizer(policy="grandplan", wt_max=0.4)
    _register_norm_terms(n, object(), object(), 6.0, {}, 0.90, 20, 0.25,
                         "grandplan", 0.4)
    assert n.configs["ft"].requires == "io"
    assert n.configs["io"].requires is None
    assert n.configs["ft"].wt_max == pytest.approx(0.25 * 0.4)
    assert n.configs["io"].wt_max is None                  # normalizer default
    assert n.configs["io"].target_share == DEFAULT_IO_TARGET_SHARE
    assert n.configs["io"].activate_overflow == 0.90
    assert n.configs["ft"].activate_overflow == FT_ACTIVATE_OVERFLOW

    # Under adaptive the target shares already set each term's budget, so no
    # weight-ceiling override is installed.
    n = TermNormalizer(policy="adaptive")
    _register_norm_terms(n, object(), object(), 6.0, {"io": 0.3, "ft": 0.1},
                         0.90, 20, 0.25, "adaptive", 1.0)
    assert n.configs["ft"].wt_max is None
    assert n.configs["ft"].target_share == pytest.approx(0.1)

    # Review I6: a typo'd share key used to fall back to the default silently.
    n = TermNormalizer(policy="grandplan")
    with pytest.raises(ValueError) as excinfo:
        _register_norm_terms(n, object(), object(), 6.0, {"ioo": 0.5}, 0.90,
                             20, 0.25, "grandplan", 1.0)
    assert "ioo" in str(excinfo.value) and "io" in str(excinfo.value)

    # FT off: `--norm-target-share ft=...` names a term that will not exist.
    n = TermNormalizer(policy="grandplan")
    with pytest.raises(ValueError):
        _register_norm_terms(n, object(), None, 1.0, {"ft": 0.1}, 0.90, 20,
                             0.0, "grandplan", 1.0)


def _small_config(tmp_path):
    cfg = json.load(open(SIMPLE))
    cfg.update(num_threads=4, plot_flag=0, num_bins_x=16, num_bins_y=16,
               global_place_stages=[dict(num_bins_x=16, num_bins_y=16, iteration=40,
                                         learning_rate=.01,
                                         wirelength="weighted_average",
                                         optimizer="nesterov")])
    path = tmp_path / "simple_norm.json"
    path.write_text(json.dumps(cfg))
    return str(path)


@pytest.mark.slow
def test_grandplan_policy_runs_and_writes_a_norm_trace(tmp_path):
    out = str(tmp_path / "grandplan.json")
    result = run_io(_small_config(tmp_path), 4, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, check_invariant=True,
                    norm_policy="grandplan", norm_probe_every=5,
                    norm_ramp_period=10, norm_wt_max=1.0)
    assert result["norm_policy"] == "grandplan"
    assert result["norm_trace"] == os.path.abspath(out + ".norm_trace.jsonl")
    rows = read_norm_trace(result["norm_trace"])
    assert rows, "no normalisation rows were emitted"
    assert [r["obj_version"] for r in rows] == sorted(set(r["obj_version"] for r in rows))
    assert all(r["refreshed_version"] == r["obj_version"] for r in rows)
    assert all(set(r["terms"]) == {"io", "ft"} for r in rows)
    assert any(r["terms"]["io"]["lam"] > 0.0 for r in rows)
    assert result["lambda_io_final"] > 0.0


@pytest.mark.slow
def test_adaptive_policy_runs_and_respects_the_requested_share(tmp_path):
    out = str(tmp_path / "adaptive.json")
    result = run_io(_small_config(tmp_path), 4, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, norm_policy="adaptive", norm_probe_every=5,
                    norm_target_share="io=0.3,ft=0.1")
    rows = read_norm_trace(result["norm_trace"])
    assert rows
    assert all(r["policy"] == "adaptive" for r in rows)
    assert all(r["terms"]["io"]["target_share"] == 0.3 for r in rows)
    assert all(r["terms"]["ft"]["target_share"] == 0.1 for r in rows)


@pytest.mark.slow
def test_legacy_policy_is_the_default_and_emits_no_trace(tmp_path):
    out = str(tmp_path / "legacy.json")
    result = run_io(_small_config(tmp_path), 4, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, check_invariant=True)
    assert result["norm_policy"] == "legacy"
    assert result["norm_trace"] is None
    assert not os.path.exists(out + ".norm_trace.jsonl")
    events = result["trajectory"]
    assert events and any(event["grad_l1_ft"] > 0 for event in events)
    assert all(event["obj_version"] == event["refreshed_version"] for event in events)


@pytest.mark.slow
def test_legacy_publish_atomic_is_wired_with_the_correct_tau_ecc_gamma(tmp_path, monkeypatch):
    """Fix round 1 regression (controller notes (a)/(c)): `--norm-policy
    legacy` must route through `normalizer.transaction(..., legacy_publish=
    ...)` with a closure that calls the retired `ops.ft_callback.
    publish_atomic` with exactly the `(tau_rel=state.tau/L_R,
    ecc_max=distance.max(), gamma)` triple the pre-P-H driver passed -- never
    a value derived through the normalizer's own (grandplan/adaptive)
    coefficient maths. A spy on the module attribute (the driver re-imports
    it from inside `cb()`'s legacy branch every qualifying callback, so
    patching the module attribute is observed) records the positional args
    and calls through to the real implementation, so the run's actual
    coefficients are unaffected by the spy.

    Fix round 2: the round-1 "gamma matches" check
    (`event["lambda_io"] <= lipschitz_cap(event["tau"], gamma, 1.0,
    event["Cmax"])`) was tautological -- `apply_ft_transaction` computed
    `event["lambda_io"]` as `min(base, lipschitz_cap(tau, gamma, ...))` from
    that exact spied `gamma`/`tau`/`Cmax`, so the bound holds for *any* gamma
    the driver happens to pass, not just the correct one. Replaced with (i)
    a schedule-independent property check (every spied gamma is finite and
    positive, and non-increasing wherever the matching trajectory overflow
    is non-increasing -- DREAMPlace's Lgamma schedule ties gamma to
    overflow) and (ii) a direct equality check against a new `entry["gamma"]`
    trajectory field the driver now records independently of the spy."""
    import math

    import ioplace.ops.ft_callback as ft_callback_mod
    from ioplace.drivers.run_placement import _load_dreamplace

    k = 4
    config = _small_config(tmp_path)
    calls = []
    original = ft_callback_mod.publish_atomic

    def spy(state, io_term, ft_term, wirelength_op, pos, iteration, tau_rel,
           ecc_max, gamma):
        calls.append((iteration, tau_rel, ecc_max, gamma))
        return original(state, io_term, ft_term, wirelength_op, pos, iteration,
                        tau_rel, ecc_max, gamma)

    monkeypatch.setattr(ft_callback_mod, "publish_atomic", spy)

    # P-F fix round 3: this test's schedule-property checks below (gamma
    # tracks DREAMPlace's own overflow-driven Lgamma schedule) were recorded
    # against the physical GP trajectory the lower-left anchor produces, the
    # only anchor that existed before the P-F range. run_io's own default
    # flipped to "center" (design v2 sec 7), and the anchor changes the real
    # IO gradient the optimizer descends, so it changes the trajectory these
    # checks are pinned to -- pin the anchor explicitly rather than letting
    # it silently track run_io's default.
    out = str(tmp_path / "legacy_spy.json")
    result = run_io(config, k, "grid", 0, out,
                    rho_max=.4, every=5, of_on=2., of_full=1.,
                    callback_order="atomic", f_ft_max=.25, ft_ramp_mode="constant",
                    no_diag=True, node_anchor="lower_left")
    assert calls, "publish_atomic was never called"

    # Recompute L_R independently the same way run_io does, from the config's
    # own die box and the test's own k -- not from anything the driver itself
    # derived.
    params, placedb = _load_dreamplace(config)
    placedb.initialize(params)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    L_R = ((die[2] - die[0]) * (die[3] - die[1]) / k) ** 0.5

    events_by_iter = {e["iteration"]: e for e in result["trajectory"] if "kappa_ft" in e}
    assert events_by_iter, "no legacy events with a kappa_ft record"
    calls_sorted = sorted(calls, key=lambda c: c[0])
    # Not vacuous: the schedule-property check below is pairwise, so it needs
    # at least two spied calls to say anything at all.
    assert len(calls_sorted) >= 2, "need >= 2 spied calls to check the schedule property"

    checked = 0
    for iteration, tau_rel, ecc_max, gamma in calls_sorted:
        event = events_by_iter.get(iteration)
        if event is None:
            continue
        assert tau_rel == pytest.approx(event["tau"] / L_R, rel=1e-9)
        assert ecc_max > 0.0    # f_ft_max > 0 in this run
        # (i) schedule-independent property: finite, positive.
        assert math.isfinite(gamma) and gamma > 0.0
        # (ii) direct equality against the driver's own recorded value,
        # independent of anything derived through the spy or through
        # apply_ft_transaction's own formulas.
        assert event["gamma"] == gamma
        checked += 1
    assert checked > 0

    # (i) continued: gamma must not increase wherever overflow does not
    # increase, matching DREAMPlace's Lgamma schedule (gamma tracks overflow).
    pairs_checked = 0
    for (it_a, *_), (it_b, *_) in zip(calls_sorted, calls_sorted[1:]):
        event_a, event_b = events_by_iter.get(it_a), events_by_iter.get(it_b)
        if event_a is None or event_b is None:
            continue
        gamma_a = next(g for i, _, _, g in calls_sorted if i == it_a)
        gamma_b = next(g for i, _, _, g in calls_sorted if i == it_b)
        if event_b["overflow"] <= event_a["overflow"]:
            assert gamma_b <= gamma_a + 1e-9
        pairs_checked += 1
    assert pairs_checked > 0
