# Normalisation Module (P-H) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three ad-hoc coefficient paths (λ_IO EMA + Lipschitz cap, κ_FT force share, the one-shot route λ) with one `TermNormalizer` that normalises every extra objective term by gradient-norm ratio under two selectable policies, emits a per-probe `norm_trace.jsonl`, and keeps the retired paths alive only as thin adapters.

**Architecture:** A pure-math layer (`src/ioplace/norm.py`) computes weights from gradient norms with no torch and no DREAMPlace dependency; a `TermNormalizer` class owns per-term registration, activation, EMA, the generalised Lipschitz cap, and the one-bump-per-transaction `obj_version` discipline; a torch-only `probe()` produces the isolated per-term gradients; a separate writer (`src/ioplace/norm_trace.py`) serialises one JSON row per transaction. `run_placement_io.py` keeps `ScheduleState` for τ/ρ only and routes all λ through the normalizer, with `--norm-policy legacy` delegating verbatim to the existing `publish_atomic`/`apply_ft_transaction` pair so recorded λ values are reproduced bit-exactly.

**Tech Stack:** Python 3.12, PyTorch 2.8.0+cu128 (CPU is enough for every non-slow test), NumPy, DREAMPlace (`NesterovAcceleratedGradientOptimizer`, `PlaceObj` extra-term patch), pytest 9.1.1.

**Spec:** [docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md](../specs/2026-09-19-v2-io-aware-redesign-design.md) — section 4 "Normalisation module (P-H)", plus the Decisions and the section 3/5/6 interface notes. If the spec file is not yet assembled when you start, the authoritative source for section 4 is the design draft it is being assembled from; do not invent requirements beyond it.

## Global Constraints

- Test protocol, verbatim from the spec section 9: `source src/scripts/env.sh`, then `"$IOPLACE_PYTHON" -m pytest`; `-m "not slow"` while iterating, full suite before declaring done.
- No new DREAMPlace patch, verbatim from the spec section 1: `m2-extra-obj-terms.patch` already adds extra terms *after* the fence branch of `PlaceObj.obj_fn` (verified, `PlaceObj.py:298-328`), so **no new DREAMPlace patch is needed** for any v2 term; `iteration-callback.patch` supplies the hull-rebuild and probe hook.
- Norm order: default `norm_p=1` (L1) with L2 as a switch — every calibrated constant (`ratio_ema`, `κ_ft`, `c_lip`) was fitted under L1.
- Shared across both policies: `ema=0.5`, the activation ramps, and the Lipschitz cap.
- Policy A `grandplan`: `λ_t = wt_t·‖∇WL‖_p/‖∇T_t‖_p`, `wt_t` starting at 0.05, stepped +0.05 every `ramp_period=100` iterations up to `wt_max=1.0`, gated on the activation overflow threshold.
- Policy B `adaptive`: target force shares `f_t`; `λ_t ← λ_t·(f_t·G/(λ_t‖∇T_t‖))^0.5` with momentum `λ_t ← 0.75·λ_t^prev + 0.25·λ_t^new`.
- Lipschitz cap survives in form: `lipschitz_cap(tau, gamma, c_lip, Cmax)` with `Cmax = 1 + Σ_t κ_t(curv_t−1)_+`, generalising `derive_cmax`. Declared curvatures: IO 1, FT `ecc_max`, capacity `max_s pen''`, pseudo-FT 1. λ's are clipped so their *sum* respects the cap; which term bound the cap is logged.
- Exactly one `obj_version` bump per transaction, followed by a required `refresh_nesterov_secant` then `mark_refreshed()`. This invariant holds per counter, not per callback: under the normalizer policies, `ScheduleState.obj_version` (τ/ρ) and `TermNormalizer.obj_version` (λ) each bump independently within one iteration; `VersionPair(state, normalizer)` is what makes summing them safe for `install_version_invariant` (Task 2).
- The three legacy paths (`schedules.py:94-104,159-176`; `schedules.py:66-81,178-237`; `ops/routing_gp_controller.py:64-78`) end as thin adapters over the new module or are deleted; the routing one is kept only behind `IOPLACE_ENABLE_GR_IN_LOOP=1`.
- The host is shared: check `nvidia-smi` and select a free device with `CUDA_VISIBLE_DEVICES` before any GPU work. At the time of writing GPU 3 was the only idle device.
- Python floor stays `>=3.9` (`pyproject.toml:8`); do not use 3.10+ syntax (`X | Y` annotations, `match`).
- Every commit message ends with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File Structure

**Created**

| File | Single responsibility |
|---|---|
| `src/ioplace/norm.py` | Pure normalisation math (`ema_update`, `grandplan_weight`, `grandplan_lambda`, `adaptive_lambda`, `cmax_from_curvatures`, `clip_sum_to_cap`, `parse_target_shares`) **and** the `TermNormalizer` state machine + `VersionPair`. |
| `src/ioplace/norm_trace.py` | `norm_trace.jsonl` row schema and append-only writer. Nothing else. |
| `src/ioplace/ops/norm_terms.py` | Adapters exposing `IoTerm`/`FtTerm` to the normalizer's `value(pos, ctx)` protocol. |
| `tests/test_norm.py` | Unit tests for the math layer, the state machine, both policies, the cap, probing, and the version discipline. |
| `tests/test_norm_trace.py` | Row-schema and writer round-trip tests. |
| `tests/test_norm_legacy_adapter.py` | Golden λ regression for the retired path and the legacy-policy equivalence test. |
| `tests/test_norm_driver.py` | Driver/CLI wiring tests (one slow end-to-end on `simple.json`). |
| `tests/test_norm_group_validation.py` | Slow, opt-in `mempool_group` λ-within-2× acceptance test. |

**Modified**

| File | Change |
|---|---|
| `src/ioplace/drivers/run_placement_io.py` | `run_io` gains the norm kwargs, builds the normalizer, routes every λ through `normalizer.transaction(...)`, refreshes once, records norm fields in `result`. |
| `src/ioplace/drivers/run_placement.py:448-537` | `main()` split into `build_parser()` + `main()`; five `--norm-*` flags plus `--norm-target-share`/`--norm-trace` forwarded to `run_io`. |
| `src/ioplace/ops/routing_gp_controller.py:17-35,64-78` | `IOPLACE_ENABLE_GR_IN_LOOP=1` gate in `__init__`; `_calibrate` reduced to a call to `TermNormalizer.oneshot_lambda`. |
| `docs/dev-env.md` | A "Normalisation module (P-H) flags" subsection listing the flags, defaults and the `norm_trace.jsonl` artefact. |

**Deliberately unchanged:** `src/ioplace/schedules.py` (τ/ρ schedules and the legacy `ScheduleState` stay; they become the thing the legacy adapter delegates to), `src/ioplace/ops/ft_callback.py` (`publish_atomic` *is* the legacy adapter body), `src/ioplace/dp_hook.py` (`install_version_invariant` is reused, not modified).

**One deliberate refinement of the spec's interface:** section 4 lists `term.grad_l1(pos)` alongside `term.value(pos, ctx)`. This plan implements the protocol as `value(pos, ctx)` **only**, with `TermNormalizer.probe` owning the backward, the fixed/filler masking and the `norm_p` choice. Rationale: one gradient definition for all five terms, and the cached gradient tensors are what `cancellation_ratio` needs. `IoTerm.io_grad_l1` stays in place for the legacy path.

**A second refinement:** the spec's `probe(iteration, pos, obj_and_grad)` becomes `probe(iteration, pos, wl_fn, ctx, probe_terms=None)` (Task 4) — `wl_fn`/`ctx` replace the single `obj_and_grad` callable so every term's `value(pos, ctx)` call shares one context dict, and `probe_terms` lets a caller measure a subset.

---

### Task 1: Pure normalisation math

**Files:**
- Create: `src/ioplace/norm.py`
- Test: `tests/test_norm.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `EPS = 1e-30`
  - `ema_update(prev, inst, ema=0.5) -> float` (`prev` may be `None`)
  - `grandplan_weight(iteration, it_activate, wt0=0.05, wt_step=0.05, ramp_period=100, wt_max=1.0) -> float`
  - `grandplan_lambda(wt, ratio, wl_norm, grad_norm, eps_rel=1e-3) -> float`
  - `adaptive_lambda(lam_prev, grad_norm, target_share, total_force, wl_norm, momentum=0.75, eps_rel=1e-3) -> float`
  - `cmax_from_curvatures(items) -> float`, `items` an iterable of `(kappa, curvature)` float pairs
  - `clip_sum_to_cap(lambdas, cap) -> (dict, str_or_None)`
  - `parse_target_shares(spec) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_norm.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'ioplace.norm'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/norm.py`:

```python
"""Unified gradient-norm normalisation for extra objective terms (v2 design
sec 4, P-H).

This module replaces three ad-hoc coefficient paths: the lambda_IO EMA plus
Lipschitz cap (`schedules.py:94-104,159-176`), the kappa_FT force share
(`schedules.py:66-81,178-237`), and the one-shot route lambda
(`ops/routing_gp_controller.py:64-78`).

Everything above `TermNormalizer` is a total function of floats: no torch, no
DREAMPlace, no CUDA, so the policy maths is testable on any host.
"""

EPS = 1e-30


def ema_update(prev, inst, ema=0.5):
    """Shared exponential moving average (design sec 4: both policies use
    `ema=0.5`). `prev is None` seeds the average with `inst`, matching
    `ScheduleState.update_ratio`'s first-sample behaviour."""
    return inst if prev is None else ema * prev + (1.0 - ema) * inst


def grandplan_weight(iteration, it_activate, wt0=0.05, wt_step=0.05,
                     ramp_period=100, wt_max=1.0):
    """Policy A's stepped weight: `wt0` at activation, `+wt_step` every
    `ramp_period` iterations, clamped at `wt_max`. Returns 0.0 before the term
    activates (`it_activate is None` or `iteration < it_activate`)."""
    if ramp_period <= 0:
        raise ValueError("ramp_period must be positive")
    if it_activate is None or iteration < it_activate:
        return 0.0
    steps = (iteration - it_activate) // ramp_period
    return min(wt0 + wt_step * steps, wt_max)


def grandplan_lambda(wt, ratio, wl_norm, grad_norm, eps_rel=1e-3):
    """Policy A coefficient: `lambda_t = wt_t * ratio`, where `ratio` is the
    (EMA-damped) `||grad WL||_p / ||grad T_t||_p`. The raw norms are passed
    separately only for the zero-gradient guard inherited from
    `schedules.derive_kappa_ft`: a term whose gradient is below
    `eps_rel * ||grad WL||_p` has no local signal and must never manufacture a
    penalty (and the division that produced `ratio` is then meaningless)."""
    if grad_norm <= eps_rel * wl_norm:
        return 0.0
    return wt * ratio


def adaptive_lambda(lam_prev, grad_norm, target_share, total_force, wl_norm,
                    momentum=0.75, eps_rel=1e-3):
    """Policy B coefficient: one DREAMPlace-4.0-style multiplicative force-share
    update `lam_new = lam_prev * sqrt(f_t * G / (lam_prev * ||grad T_t||))`
    followed by the momentum blend `momentum*lam_prev + (1-momentum)*lam_new`.

    `total_force` is `G = ||grad WL||_p + sum_t lam_t ||grad T_t||_p` evaluated
    with the *pre-update* coefficients.

    Bootstrap: with `lam_prev <= 0` the multiplicative form divides by zero, so
    the first value is taken from policy A's form at `wt = f_t` and returned
    undamped -- blending a bootstrap against `lam_prev == 0` would halve it for
    no reason."""
    if target_share <= 0.0 or grad_norm <= eps_rel * wl_norm:
        return 0.0
    if lam_prev <= 0.0:
        return target_share * wl_norm / grad_norm
    lam_new = lam_prev * (target_share * total_force /
                          max(lam_prev * grad_norm, EPS)) ** 0.5
    return momentum * lam_prev + (1.0 - momentum) * lam_new


def cmax_from_curvatures(items):
    """`Cmax = 1 + sum_t kappa_t (curv_t - 1)_+`, the N-term generalisation of
    `schedules.derive_cmax(kappa_ft, ecc_max_max)`. `items` is an iterable of
    `(kappa, curvature)` pairs over the *active* terms only."""
    return 1.0 + sum(kappa * max(curv - 1.0, 0.0) for kappa, curv in items)


def clip_sum_to_cap(lambdas, cap):
    """Scale every coefficient by one common factor so that `sum(lambdas)`
    respects `cap` (design sec 4: the cap binds the *sum*, not each term).

    Returns `(scaled, binding)`. `binding` is the name of the largest pre-scale
    coefficient -- the dominant contributor to the violation -- or `None` when
    the cap did not bind. Ties break alphabetically so the log is
    deterministic."""
    total = sum(lambdas.values())
    if total <= cap or total <= 0.0:
        return dict(lambdas), None
    scale = cap / total
    binding = max(sorted(lambdas), key=lambdas.__getitem__)
    return dict((name, value * scale) for name, value in lambdas.items()), binding


def parse_target_shares(spec):
    """Parse the `--norm-target-share` CLI value: `"io=0.3,ft=0.1"` ->
    `{"io": 0.3, "ft": 0.1}`. Empty or `None` -> `{}`."""
    if not spec:
        return {}
    shares = {}
    for item in spec.split(","):
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ValueError("target share must be name=value, got %r" % (item,))
        shares[name.strip()] = float(value)
    return shares
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/norm.py tests/test_norm.py
git commit -m "feat(norm): add pure gradient-norm normalisation maths" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `TermNormalizer` core — registration, policy A, cap, transaction discipline

**Files:**
- Modify: `src/ioplace/norm.py` (append below the pure functions from Task 1)
- Test: `tests/test_norm.py` (append)

**Interfaces:**
- Consumes: `ema_update`, `grandplan_weight`, `grandplan_lambda`, `cmax_from_curvatures`, `clip_sum_to_cap`, `EPS` from Task 1; `schedules.activation_ramp(iteration, it_activate, n_ramp=20)` and `schedules.lipschitz_cap(tau, gamma, c_lip=1.0, cmax=1.0)`; `dp_hook.install_version_invariant(optimizer, state)`.
- Produces:
  - `TermConfig(name, curvature=1.0, target_share=0.0, activate_overflow=0.90, n_ramp=20)` — dataclass
  - `TermState(grad_norm=0.0, ratio_inst=None, ratio_ema=None, wt=0.0, lam=0.0, active=False, it_activate=None)` — dataclass
  - `NormTransaction(lambdas, cmax, cap, cap_binding, cancellation_ratio, obj_version, row, legacy_record=None, needs_refresh=True)` — dataclass
  - `VersionPair(*states)` with `.obj_version` / `.refreshed_version` properties
  - `TermNormalizer(policy="grandplan", norm_p=1, ema=0.5, probe_every=50, wt0=0.05, wt_step=0.05, ramp_period=100, wt_max=1.0, momentum=0.75, c_lip=1.0, eps_rel=1e-3, num_movable=None, num_nodes=None, track_cancellation=True, legacy_state=None, trace=None)` with:
    - `register(name, term, curvature, target_share=0.0, activate_overflow=0.90, n_ramp=20) -> None` (`kappa` is not a registration input — see below)
    - `update_grad_norms(grad_norms) -> None`
    - `should_probe(iteration) -> bool`
    - `weights(iteration, overflow, tau, gamma) -> dict`
    - `transaction(iteration, overflow, tau, gamma, grad_norms=None, legacy_publish=None) -> NormTransaction`
    - `mark_refreshed() -> None`, `needs_refresh() -> bool`
    - `obj_version` / `refreshed_version` read-only properties
    - `lambdas` — live `dict` the driver's `term_fn` reads every iteration
    - `oneshot_lambda(strength, wl_norm, grad_norm, floor=1e-12) -> float` — staticmethod

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_norm.py`:

```python
from ioplace.norm import NormTransaction, TermConfig, TermNormalizer, TermState, VersionPair
from ioplace.schedules import activation_ramp, lipschitz_cap


def _norm_a(**kwargs):
    """Policy-A normalizer with one IO term, cap effectively disabled."""
    n = TermNormalizer(policy="grandplan", **kwargs)
    n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
    return n


def test_normalizer_rejects_unknown_policy_and_norm_order():
    with pytest.raises(ValueError):
        TermNormalizer(policy="bogus")
    with pytest.raises(ValueError):
        TermNormalizer(norm_p=3)
    with pytest.raises(ValueError):
        TermNormalizer(policy="legacy")          # legacy needs a ScheduleState


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
    txn = n.transaction(0, 0.85, tau=100.0, gamma=10.0)
    # pre-clip lambdas: io = wt*ratio_io = 1.0*200 = 200, ft = 1.0*100 = 100
    # derived kappa: kappa_io = 1.0 (by definition), kappa_ft = 100/200 = 0.5
    assert txn.cmax == pytest.approx(3.5, rel=1e-12)                # 1 + 0.5*(6-1)
    assert txn.cap == pytest.approx(lipschitz_cap(100.0, 10.0, 1.0, 3.5), rel=1e-12)
    assert sum(txn.lambdas.values()) == pytest.approx(txn.cap, rel=1e-12)
    assert txn.cap_binding == "io"                                  # 200 > 100
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
    n.update_grad_norms({"wl": 1000.0, "io": 10.0})
    preview = n.weights(0, 0.85, tau=1000.0, gamma=1e-12)
    assert preview == {"io": pytest.approx(5.0, rel=1e-12)}
    assert n.obj_version == 0
    assert n.lambdas == {"io": 0.0}                                  # not committed
    assert n.transaction(0, 0.85, tau=1000.0, gamma=1e-12).lambdas == preview
    assert n.lambdas == preview                                      # now committed


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
```

Append the version-invariant reuse test (it needs DREAMPlace's optimizer, so it
lives in the same file behind an importorskip):

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -v`
Expected: collection error — `ImportError: cannot import name 'TermNormalizer' from 'ioplace.norm'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/norm.py` (and add the two imports at the top of the file,
below the docstring: `from dataclasses import dataclass` and
`from ioplace.schedules import activation_ramp, lipschitz_cap`):

```python
def _json_cap(cap):
    """`inf` is not valid JSON; normalise it to `None` for logging. Shared by
    `TermNormalizer._row` and the driver's non-legacy trajectory entry (Task 7
    Step 3d)."""
    return None if cap == float("inf") else cap


@dataclass
class TermConfig:
    """Static registration data for one extra objective term. `kappa` is not
    stored here: it is derived every transaction from the live lambdas (see
    `register`'s docstring)."""
    name: str
    curvature: float = 1.0
    target_share: float = 0.0
    activate_overflow: float = 0.90
    n_ramp: int = 20


@dataclass
class TermState:
    """Live per-term state. `wt` holds policy A's stepped weight or policy B's
    ramped target share, whichever the active policy produced."""
    grad_norm: float = 0.0
    ratio_inst: float = None
    ratio_ema: float = None
    wt: float = 0.0
    lam: float = 0.0
    active: bool = False
    it_activate: int = None


@dataclass
class NormTransaction:
    """Result of one atomic coefficient update."""
    lambdas: dict
    cmax: float
    cap: float
    cap_binding: str
    cancellation_ratio: float
    obj_version: int
    row: dict
    legacy_record: dict = None
    needs_refresh: bool = True


class VersionPair:
    """Expose several version-carrying states as one, so a single
    `dp_hook.install_version_invariant` covers all of them. Stacking two
    invariant wrappers would not work: `refresh_nesterov_secant` unwraps only
    one `__wrapped__` level and the inner wrapper would assert against the
    refresh itself. Summation is sound because `refreshed_version <=
    obj_version` holds for every member, so the sums are equal iff every member
    is refreshed."""

    def __init__(self, *states):
        self.states = states

    @property
    def obj_version(self):
        return sum(s.obj_version for s in self.states)

    @property
    def refreshed_version(self):
        return sum(s.refreshed_version for s in self.states)


class TermNormalizer:
    """Gradient-norm normalisation for N extra objective terms (design sec 4).

    Generalises `ScheduleState.apply_ft_transaction`'s seven-step atomic
    discipline: measure norms, derive every coefficient, bump `obj_version`
    exactly once, hand the caller a flag to call `refresh_nesterov_secant`, then
    `mark_refreshed()`.
    """

    POLICIES = ("legacy", "grandplan", "adaptive")

    def __init__(self, policy="grandplan", norm_p=1, ema=0.5, probe_every=50,
                 wt0=0.05, wt_step=0.05, ramp_period=100, wt_max=1.0,
                 momentum=0.75, c_lip=1.0, eps_rel=1e-3,
                 num_movable=None, num_nodes=None, track_cancellation=True,
                 legacy_state=None, trace=None):
        if policy not in self.POLICIES:
            raise ValueError("policy must be one of %r, got %r" % (self.POLICIES, policy))
        if norm_p not in (1, 2):
            raise ValueError("norm_p must be 1 or 2, got %r" % (norm_p,))
        if probe_every <= 0:
            raise ValueError("probe_every must be positive")
        if policy == "legacy" and legacy_state is None:
            raise ValueError("policy 'legacy' requires legacy_state=<ScheduleState>")
        self.policy = policy
        self.norm_p = int(norm_p)
        self.ema = float(ema)
        self.probe_every = int(probe_every)
        self.wt0, self.wt_step = float(wt0), float(wt_step)
        self.ramp_period, self.wt_max = int(ramp_period), float(wt_max)
        self.momentum = float(momentum)
        self.c_lip, self.eps_rel = float(c_lip), float(eps_rel)
        self.num_movable, self.num_nodes = num_movable, num_nodes
        self.track_cancellation = bool(track_cancellation)
        self.trace = trace
        self._legacy = legacy_state
        self.configs, self.terms, self.states = {}, {}, {}
        self.lambdas = {}
        self.wl_norm = 0.0
        self._obj_version = 0
        self._refreshed_version = 0
        self._pending_row = None
        self._grad_cache = {}
        self._probed_this_callback = False

    # -- version discipline -------------------------------------------------
    @property
    def obj_version(self):
        return self._legacy.obj_version if self._legacy is not None else self._obj_version

    @property
    def refreshed_version(self):
        return (self._legacy.refreshed_version if self._legacy is not None
                else self._refreshed_version)

    def needs_refresh(self):
        return self.obj_version != self.refreshed_version

    def mark_refreshed(self):
        """Second half of the transaction: call this *after*
        `refresh_nesterov_secant(optimizer)`. Also finalises and emits the
        pending trace row, so a row is only written once its objective version
        is actually live in the optimizer's cache."""
        if self._legacy is not None:
            self._legacy.mark_refreshed()
        else:
            self._refreshed_version = self._obj_version
        row, self._pending_row = self._pending_row, None
        if row is not None:
            row["refreshed_version"] = self.refreshed_version
            if self.trace is not None:
                self.trace.write(row)

    # -- registration and measurement --------------------------------------
    def register(self, name, term, curvature, target_share=0.0,
                 activate_overflow=0.90, n_ramp=20):
        """Register one term. `term` exposes `value(pos, ctx) -> Tensor`;
        `probe()` owns the backward, the masking and the norm order.

        Declared curvatures (design sec 4): IO 1, FT `ecc_max`, capacity
        `max_s pen''`, pseudo-FT 1. `kappa_t` is derived each transaction as
        `lambda_t / lambda_io` (pre-clip), generalising
        `derive_cmax(kappa_ft, ecc_max)` -- it is not a registration input."""
        if name in self.configs:
            raise ValueError("term %r already registered" % (name,))
        self.configs[name] = TermConfig(name, float(curvature),
                                        float(target_share), float(activate_overflow),
                                        int(n_ramp))
        self.terms[name] = term
        self.states[name] = TermState()
        self.lambdas[name] = 0.0

    def should_probe(self, iteration):
        return iteration % self.probe_every == 0

    def update_grad_norms(self, grad_norms):
        """Absorb one probe's measurements. `grad_norms` maps `"wl"` and each
        registered term name to its `||.||_p` gradient norm; unmentioned terms
        keep their previous measurement (this is how `probe_terms` subsetting
        stays correct)."""
        self.wl_norm = float(grad_norms["wl"])
        for name, state in self.states.items():
            if name not in grad_norms:
                continue
            grad_norm = float(grad_norms[name])
            state.grad_norm = grad_norm
            state.ratio_inst = self.wl_norm / max(grad_norm, EPS)
            state.ratio_ema = ema_update(state.ratio_ema, state.ratio_inst, self.ema)

    # -- coefficient computation -------------------------------------------
    def _activate(self, iteration, overflow):
        """Latch each term on the first iteration at or below its activation
        overflow threshold. Monotone and idempotent, so both `weights()` and
        `transaction()` may call it."""
        for name, config in self.configs.items():
            state = self.states[name]
            if not state.active and overflow <= config.activate_overflow:
                state.active, state.it_activate = True, iteration

    def _compute(self, iteration, tau, gamma):
        """Pure: returns `(lambdas, weights, cmax, cap, cap_binding)` from the
        current measurements without mutating anything."""
        total_force = self.wl_norm + sum(
            self.states[n].lam * self.states[n].grad_norm
            for n in self.configs if self.states[n].active)
        lambdas, weights = {}, {}
        for name, config in self.configs.items():
            state = self.states[name]
            if not state.active or state.ratio_ema is None:
                lambdas[name], weights[name] = 0.0, 0.0
                continue
            ramp = activation_ramp(iteration, state.it_activate, config.n_ramp)
            if self.policy == "grandplan":
                weights[name] = ramp * grandplan_weight(
                    iteration, state.it_activate, self.wt0, self.wt_step,
                    self.ramp_period, self.wt_max)
                lambdas[name] = grandplan_lambda(weights[name], state.ratio_ema,
                                                 self.wl_norm, state.grad_norm,
                                                 self.eps_rel)
            else:
                weights[name] = ramp * config.target_share
                lambdas[name] = adaptive_lambda(state.lam, state.grad_norm,
                                                weights[name], total_force,
                                                self.wl_norm, self.momentum,
                                                self.eps_rel)
        # kappa is derived, not declared: kappa_io = 1.0 by definition, and
        # kappa_t = lambda_t / lambda_io (both pre-clip) for every other term,
        # generalising derive_cmax(kappa_ft, ecc_max)'s live ratio reading.
        # kappa_t = 0.0 when lambda_io is 0.0 (no IO signal to scale against).
        lam_io_preclip = lambdas.get("io", 0.0)
        cmax = cmax_from_curvatures(
            [(1.0 if name == "io" else
              (lambdas[name] / lam_io_preclip if lam_io_preclip != 0.0 else 0.0),
              self.configs[name].curvature)
             for name in self.configs if self.states[name].active])
        cap = lipschitz_cap(tau, gamma, self.c_lip, cmax)
        lambdas, binding = clip_sum_to_cap(lambdas, cap)
        return lambdas, weights, cmax, cap, binding

    def weights(self, iteration, overflow, tau, gamma):
        """Pure preview of the coefficients. Latches activation (monotone) but
        commits nothing and never bumps `obj_version`."""
        self._activate(iteration, overflow)
        return self._compute(iteration, tau, gamma)[0]

    def _norm(self, tensor):
        return (float(tensor.abs().sum()) if self.norm_p == 1
                else float(tensor.norm(p=2)))

    def _cancellation_ratio(self, lambdas):
        """`||sum_t lam_t grad T_t||_p / sum_t lam_t ||grad T_t||_p`, the N-term
        generalisation of `ScheduleState.cancellation_ratio`. `None` when this
        callback did not call `probe()` (the cache would otherwise mix stale
        gradients from an earlier probe with this transaction's coefficients)
        or when the probe kept no gradient tensors (`track_cancellation=False`)."""
        if not self._probed_this_callback:
            return None
        merged, denom = None, 0.0
        for name, grad in self._grad_cache.items():
            lam = lambdas.get(name, 0.0)
            if lam == 0.0:
                continue
            merged = grad * lam if merged is None else merged + grad * lam
            denom += lam * self.states[name].grad_norm
        if merged is None or denom <= 0.0:
            return None
        return self._norm(merged) / denom

    def _row(self, iteration, overflow, tau, gamma, cmax, cap, binding, cancellation):
        """One `norm_trace.jsonl` row. `grad_l1` keeps its name for continuity
        with the retired `grad_l1_io`/`grad_l1_ft` trajectory fields; it holds
        the `||.||_p` norm with `p` = this row's `norm_p`."""
        denom = self.wl_norm + sum(s.lam * s.grad_norm for s in self.states.values())
        return {
            "iteration": int(iteration),
            "overflow": float(overflow),
            "tau": float(tau),
            "gamma": float(gamma),
            "policy": self.policy,
            "norm_p": self.norm_p,
            "grad_l1_wl": self.wl_norm,
            "cmax": cmax,
            "cap": _json_cap(cap),
            "cap_binding": binding,
            "cancellation_ratio": cancellation,
            "obj_version": self.obj_version,
            "refreshed_version": self.refreshed_version,
            "terms": dict(
                (name, {"grad_l1": s.grad_norm, "ratio_inst": s.ratio_inst,
                        "ratio_ema": s.ratio_ema, "wt": s.wt,
                        "target_share": self.configs[name].target_share,
                        "lam": s.lam,
                        "share": (s.lam * s.grad_norm / denom) if denom > 0.0 else 0.0,
                        "active": s.active})
                for name, s in self.states.items()),
        }

    def transaction(self, iteration, overflow, tau, gamma, grad_norms=None,
                    legacy_publish=None):
        """Atomic coefficient update: derive every lambda, commit it, bump
        `obj_version` exactly once, and hand the driver a transaction whose
        `needs_refresh` flag means "call `refresh_nesterov_secant(optimizer)`
        then `mark_refreshed()` before the next optimizer step"."""
        if self._pending_row is not None:
            raise RuntimeError(
                "previous transaction was not refreshed: call "
                "refresh_nesterov_secant(optimizer) then mark_refreshed()")
        if grad_norms is not None:
            self.update_grad_norms(grad_norms)
        if self._legacy is not None:
            return self._legacy_transaction(iteration, overflow, tau, gamma,
                                            legacy_publish)
        self._activate(iteration, overflow)
        lambdas, weights, cmax, cap, binding = self._compute(iteration, tau, gamma)
        for name, state in self.states.items():
            state.lam, state.wt = lambdas[name], weights[name]
        self.lambdas = dict(lambdas)
        cancellation = self._cancellation_ratio(lambdas)
        self._probed_this_callback = False
        self._obj_version += 1
        row = self._row(iteration, overflow, tau, gamma, cmax, cap, binding,
                        cancellation)
        self._pending_row = row
        return NormTransaction(lambdas=dict(lambdas), cmax=cmax, cap=cap,
                               cap_binding=binding, cancellation_ratio=cancellation,
                               obj_version=self.obj_version, row=row)

    @staticmethod
    def oneshot_lambda(strength, wl_norm, grad_norm, floor=1e-12):
        """One-shot ratio normalisation for a term that is calibrated once per
        rebuild rather than every probe. The expression is written exactly as
        the retired `routing_gp_controller._calibrate` wrote it, so the adapter
        is bit-for-bit identical."""
        return strength * wl_norm / grad_norm if grad_norm > floor else 0.0
```

Also add a placeholder-free stub for the legacy branch so the module imports
cleanly before Task 6 fills it in — put it directly above `oneshot_lambda`:

```python
    def _legacy_transaction(self, iteration, overflow, tau, gamma, legacy_publish):
        raise NotImplementedError(
            "policy 'legacy' delegation is installed by the legacy adapter task")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -v`
Expected: 32 passed (15 from Task 1 + 17 new).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/norm.py tests/test_norm.py
git commit -m "feat(norm): add TermNormalizer with policy A, cap and version discipline" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Policy B `adaptive`

**Files:**
- Modify: `src/ioplace/norm.py` (`TermNormalizer.__init__` policy validation already accepts `"adaptive"`; `_compute` already branches — this task proves and fixes the branch end-to-end)
- Test: `tests/test_norm.py` (append)

**Interfaces:**
- Consumes: `adaptive_lambda(lam_prev, grad_norm, target_share, total_force, wl_norm, momentum=0.75, eps_rel=1e-3)` and `TermNormalizer.transaction(iteration, overflow, tau, gamma, grad_norms=None, legacy_publish=None) -> NormTransaction` from Tasks 1-2.
- Produces: no new names. Locks the semantics later tasks depend on: policy B reads `TermConfig.target_share`, multiplies it by the shared activation ramp, uses the *pre-update* coefficients for `G`, and stores the ramped share in `TermState.wt` (so `row["terms"][name]["wt"]` means "the effective share" under policy B and "the stepped weight" under policy A).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_norm.py`:

```python
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


def test_policy_b_ramp_scales_the_target_share_not_the_coefficient():
    n = TermNormalizer(policy="adaptive")
    n.register("io", object(), 1.0, target_share=0.2, activate_overflow=0.90, n_ramp=20)
    n.transaction(0, 0.85, tau=1000.0, gamma=0.0, grad_norms={"wl": 1000.0, "io": 10.0})
    assert n.lambdas["io"] == 0.0                        # ramp 0 -> share 0
    n.mark_refreshed()
    n.transaction(10, 0.85, tau=1000.0, gamma=0.0, grad_norms={"wl": 1000.0, "io": 10.0})
    assert n.states["io"].wt == pytest.approx(0.1, rel=1e-12)
    assert n.lambdas["io"] == pytest.approx(0.1 * 1000.0 / 10.0, rel=1e-12)


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
    # derived kappa: kappa_io = 1.0, kappa_ft = 90/180 = 0.5
    assert txn.cmax == pytest.approx(3.5, rel=1e-12)
    assert txn.cap == pytest.approx(100.0 * 100.0 / (20.0 * 3.5), rel=1e-12)  # 142.857...
    assert sum(txn.lambdas.values()) == pytest.approx(txn.cap, rel=1e-12)
    assert txn.cap_binding == "io"                                  # 180 > 90
    assert txn.lambdas["io"] / txn.lambdas["ft"] == pytest.approx(2.0, rel=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -k policy_b -v`
Expected: several FAIL. The `_compute` branch written in Task 2 already implements the formula, so the likely failure is in `test_policy_b_converges_to_the_requested_force_share` and `test_policy_b_ramp_scales_the_target_share_not_the_coefficient` if `total_force` was computed after committing, or if the ramp was applied to the coefficient. If every test passes on the first run, that is a valid outcome — record it and move to Step 4.

- [ ] **Step 3: Confirm the implementation**

Task 2's `_compute` already contains this branch; make no edit unless a
Task 3 test fails.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -v`
Expected: 38 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/norm.py tests/test_norm.py
git commit -m "feat(norm): lock policy B adaptive force-share semantics" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `probe()` — isolated per-term gradients and cancellation

**Files:**
- Modify: `src/ioplace/norm.py` (add `probe` and `_grad` to `TermNormalizer`)
- Create: `src/ioplace/ops/norm_terms.py`
- Test: `tests/test_norm.py` (append)

**Interfaces:**
- Consumes: `TermNormalizer.update_grad_norms`, `_norm`, `_cancellation_ratio`, `_grad_cache` from Task 2.
- Produces:
  - `TermNormalizer.probe(iteration, pos, wl_fn, ctx, probe_terms=None) -> dict` — returns `{"wl": float, <name>: float, ...}` and, when `track_cancellation` is set, caches the per-term gradient tensors for `_cancellation_ratio`.
  - `TermNormalizer._grad(fn, pos) -> Tensor` — one isolated backward with the fixed/filler masking of `ops/ft_callback.independent_gradient`.
  - `ioplace.ops.norm_terms.IoNormTerm(io_term)` with `value(pos, ctx) -> Tensor`
  - `ioplace.ops.norm_terms.FtNormTerm(ft_term)` with `value(pos, ctx) -> Tensor`
  - The `ctx` contract: `{"iteration": int, "overflow": float, "tau": float, "gamma": float}`; `tau` is the schedule's absolute τ (`ScheduleState.tau`), not `tau_rel`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_norm.py`:

```python
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
    pos = _pos(nl)
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -k "probe or cancellation or adapters" -v`
Expected: FAIL with `AttributeError: 'TermNormalizer' object has no attribute 'probe'` and `ModuleNotFoundError: No module named 'ioplace.ops.norm_terms'`.

- [ ] **Step 3: Write the implementation**

Add to `TermNormalizer` in `src/ioplace/norm.py`, directly above `_norm`:

```python
    def probe(self, iteration, pos, wl_fn, ctx, probe_terms=None):
        """One WL-only backward plus one isolated backward per registered term
        -- the pattern at `run_placement_io.py:465-481`, generalised to N terms.
        Five terms cost about +12% backward at `probe_every=50`; pass
        `probe_terms` to measure a subset (unmeasured terms keep their previous
        norms).

        `ctx` is handed to each term's `value(pos, ctx)`; the driver fills it
        with `{"iteration", "overflow", "tau", "gamma"}`.

        Returns the `||.||_p` norms and, with `track_cancellation` set, caches
        the gradient tensors so `transaction()` can report
        `cancellation_ratio`. The cache (and the fact that this callback
        probed) live only until the next `transaction()`, which clears the
        flag once it has used them for `cancellation_ratio`."""
        names = list(self.configs) if probe_terms is None else list(probe_terms)
        self._grad_cache = {}
        norms = {"wl": self._norm(self._grad(wl_fn, pos, "wl"))}
        for name in names:
            term = self.terms[name]
            grad = self._grad(lambda p: term.value(p, ctx), pos, name)
            norms[name] = self._norm(grad)
            if self.track_cancellation:
                self._grad_cache[name] = grad
        self.update_grad_norms(norms)
        self._probed_this_callback = True
        return norms

    def _grad(self, fn, pos, label):
        """Isolated forward+backward on a detached clone, with the fixed and
        filler entries zeroed exactly as `ops/ft_callback.independent_gradient`
        does, so the live `pos.grad` is never disturbed."""
        import torch
        leaf = pos.detach().clone().requires_grad_(True)
        grad, = torch.autograd.grad(fn(leaf), leaf)
        if self.num_movable is not None and self.num_nodes is not None:
            grad[self.num_movable:self.num_nodes] = 0
            grad[self.num_nodes + self.num_movable:] = 0
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("nonfinite probe gradient for term %r" % (label,))
        return grad
```

Create `src/ioplace/ops/norm_terms.py`:

```python
"""Adapters exposing the production IO and FT terms to `norm.TermNormalizer`.

The normalizer's term protocol is a single method, `value(pos, ctx) -> Tensor`,
returning the term's *unweighted* objective: the normalizer owns the backward,
the fixed/filler masking and the norm order, so every term's gradient norm is
measured identically (design sec 4).
"""


class IoNormTerm(object):
    """Unweighted `L_IO` (`lambda_io=1`, no margin) at the schedule's live tau."""

    name = "io"

    def __init__(self, io_term):
        self.io_term = io_term

    def value(self, pos, ctx):
        return self.io_term(pos, ctx["tau"], 1.0)


class FtNormTerm(object):
    """Unweighted feed-through part only -- `FtTerm.ft_only`, i.e. the same
    isolated quantity `ops/ft_callback.publish_atomic` measures for `kappa_ft`."""

    name = "ft"

    def __init__(self, ft_term):
        self.ft_term = ft_term

    def value(self, pos, ctx):
        return self.ft_term.ft_only(pos, ctx["tau"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm.py -v`
Expected: 47 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/norm.py src/ioplace/ops/norm_terms.py tests/test_norm.py
git commit -m "feat(norm): probe isolated per-term gradients and report cancellation" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `norm_trace.jsonl` writer

**Files:**
- Create: `src/ioplace/norm_trace.py`
- Modify: `src/ioplace/norm.py` (nothing structural — `mark_refreshed` already calls `self.trace.write(row)`; this task adds the schema check the writer enforces)
- Test: `tests/test_norm_trace.py`

**Interfaces:**
- Consumes: the row dict produced by `TermNormalizer._row` (Task 2) and emitted by `mark_refreshed` (Task 2).
- Produces:
  - `ioplace.norm_trace.ROW_FIELDS` — tuple, the exact top-level row keys
  - `ioplace.norm_trace.TERM_FIELDS` — tuple, the exact per-term keys
  - `ioplace.norm_trace.NormTraceWriter(path, validate=True)` with `write(row) -> None`, `close() -> None`, and context-manager support
  - `ioplace.norm_trace.read_norm_trace(path) -> list` — one dict per line, for tests and analysis

- [ ] **Step 1: Write the failing test**

Create `tests/test_norm_trace.py`:

```python
import json

import pytest

from ioplace.norm import TermNormalizer
from ioplace.norm_trace import (ROW_FIELDS, TERM_FIELDS, NormTraceWriter,
                                read_norm_trace)


def _row(**overrides):
    row = {"iteration": 100, "overflow": 0.42, "tau": 53.4, "gamma": 14.3,
           "policy": "grandplan", "norm_p": 1, "grad_l1_wl": 1000.0,
           "cmax": 3.5, "cap": 285.7142857142857, "cap_binding": "io",
           "cancellation_ratio": 0.83, "obj_version": 3, "refreshed_version": 3,
           "terms": {"io": {"grad_l1": 10.0, "ratio_inst": 100.0,
                            "ratio_ema": 95.0, "wt": 0.1, "target_share": 0.3,
                            "lam": 9.5, "share": 0.087, "active": True}}}
    row.update(overrides)
    return row


def test_row_fields_match_the_design_logging_list():
    assert ROW_FIELDS == ("iteration", "overflow", "tau", "gamma", "policy",
                          "norm_p", "grad_l1_wl", "cmax", "cap", "cap_binding",
                          "cancellation_ratio", "obj_version", "refreshed_version",
                          "terms")
    assert TERM_FIELDS == ("grad_l1", "ratio_inst", "ratio_ema", "wt",
                           "target_share", "lam", "share", "active")


def test_writer_appends_one_json_object_per_row(tmp_path):
    path = tmp_path / "sub" / "norm_trace.jsonl"
    writer = NormTraceWriter(str(path))
    writer.write(_row(iteration=0))
    writer.write(_row(iteration=50))
    writer.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["iteration"] for line in lines] == [0, 50]
    assert read_norm_trace(str(path)) == [_row(iteration=0), _row(iteration=50)]


def test_writer_flushes_so_a_killed_run_keeps_its_rows(tmp_path):
    path = tmp_path / "norm_trace.jsonl"
    writer = NormTraceWriter(str(path))
    writer.write(_row())
    assert len(path.read_text().splitlines()) == 1     # readable before close()
    writer.close()


def test_writer_rejects_an_unknown_or_missing_field(tmp_path):
    writer = NormTraceWriter(str(tmp_path / "t.jsonl"))
    with pytest.raises(ValueError):
        writer.write(_row(extra=1))
    bad = _row()
    del bad["cmax"]
    with pytest.raises(ValueError):
        writer.write(bad)
    bad = _row()
    bad["terms"]["io"]["surprise"] = 1
    with pytest.raises(ValueError):
        writer.write(bad)
    writer.close()


def test_writer_is_a_context_manager(tmp_path):
    path = tmp_path / "t.jsonl"
    with NormTraceWriter(str(path)) as writer:
        writer.write(_row())
    assert len(read_norm_trace(str(path))) == 1


def test_normalizer_emits_one_row_per_refreshed_transaction(tmp_path):
    path = tmp_path / "norm_trace.jsonl"
    with NormTraceWriter(str(path)) as writer:
        n = TermNormalizer(policy="grandplan", trace=writer)
        n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
        for iteration in (0, 50, 100):
            n.transaction(iteration, 0.85, tau=1000.0, gamma=1e-12,
                          grad_norms={"wl": 1000.0, "io": 10.0})
            n.mark_refreshed()
    rows = read_norm_trace(str(path))
    assert [r["iteration"] for r in rows] == [0, 50, 100]
    assert [r["obj_version"] for r in rows] == [1, 2, 3]
    # written at mark_refreshed(), so the row always records a live objective
    assert all(r["refreshed_version"] == r["obj_version"] for r in rows)
    assert [r["terms"]["io"]["wt"] for r in rows] == [0.05, 0.05, 0.10]


def test_no_row_is_written_for_a_transaction_that_is_never_refreshed(tmp_path):
    path = tmp_path / "norm_trace.jsonl"
    with NormTraceWriter(str(path)) as writer:
        n = TermNormalizer(policy="grandplan", trace=writer)
        n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
        n.transaction(0, 0.85, tau=1000.0, gamma=1e-12,
                      grad_norms={"wl": 1000.0, "io": 10.0})
    assert read_norm_trace(str(path)) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_trace.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'ioplace.norm_trace'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/norm_trace.py`:

```python
"""`norm_trace.jsonl`: one row per coefficient transaction (>= one per probe;
design sec 4).

One file per run, rewritten on open; flushed after every row, so a run killed
mid-GP keeps everything it logged. Non-finite values are serialised as
`NaN`/`Infinity`; `json.loads` accepts them, strict JSON parsers do not -- this
is a diagnostic log, and failing a multi-hour placement over one NaN would be
worse than logging it.
"""
import json
import os

ROW_FIELDS = ("iteration", "overflow", "tau", "gamma", "policy", "norm_p",
              "grad_l1_wl", "cmax", "cap", "cap_binding", "cancellation_ratio",
              "obj_version", "refreshed_version", "terms")

#: `grad_l1` holds the `||.||_p` norm with `p` = the row's `norm_p`; the name is
#: kept for continuity with the retired `grad_l1_io`/`grad_l1_ft` trajectory
#: fields. `wt` is policy A's stepped weight, policy B's ramped target share,
#: or (policy "legacy") `rho * activation_ramp(iteration, it_activate, n_ramp)`
#: -- the retired path's own ramped weight, for the same slot.
TERM_FIELDS = ("grad_l1", "ratio_inst", "ratio_ema", "wt", "target_share",
               "lam", "share", "active")


class NormTraceWriter(object):
    def __init__(self, path, validate=True):
        self.path = path
        self.validate = validate
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._stream = open(path, "w")

    def write(self, row):
        if self.validate:
            _check(row, ROW_FIELDS, "row")
            for name, term in row["terms"].items():
                _check(term, TERM_FIELDS, "term %r" % (name,))
        self._stream.write(json.dumps(row) + "\n")
        self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _check(mapping, fields, what):
    got, want = set(mapping), set(fields)
    if got != want:
        raise ValueError("%s fields %r do not match the schema (missing %r, "
                         "unexpected %r)" % (what, sorted(got), sorted(want - got),
                                             sorted(got - want)))


def read_norm_trace(path):
    """Parse a `norm_trace.jsonl` into a list of row dicts."""
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_trace.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/norm_trace.py tests/test_norm_trace.py
git commit -m "feat(norm): emit norm_trace.jsonl rows at refresh time" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Legacy adapter — `policy="legacy"` reproduces the retired λ

**Files:**
- Modify: `src/ioplace/norm.py` (replace the `_legacy_transaction` stub from Task 2)
- Test: `tests/test_norm_legacy_adapter.py`

**Interfaces:**
- Consumes: `schedules.ScheduleState` (`update_continuous`, `apply_ft_transaction`, `mark_refreshed`, `lambda_io`, `kappa_ft`, `Cmax`, `rho`, `active`, `it_activate`, `c_lip`); `ops.ft_callback.publish_atomic(state, io_term, ft_term, wirelength_op, pos, iteration, tau_rel, ecc_max, gamma) -> dict`; `TermNormalizer._row`, `NormTransaction` from Task 2.
- Produces:
  - `TermNormalizer.transaction(..., legacy_publish=<zero-arg callable returning publish_atomic's record>)` returning a `NormTransaction` whose `lambdas == {"io": state.lambda_io, "ft": state.lambda_io * state.kappa_ft}` and whose `legacy_record` is that record verbatim (the driver merges it into its trajectory entry, keeping the existing trajectory schema byte-for-byte).
  - Contract: under `policy="legacy"` the normalizer owns **no** coefficient maths. `obj_version`/`refreshed_version` delegate to the `ScheduleState`, so `apply_ft_transaction`'s single bump is still the only bump.

- [ ] **Step 1: Write the failing test**

Create `tests/test_norm_legacy_adapter.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from ioplace.norm import TermNormalizer
from ioplace.ops.ft_callback import publish_atomic
from ioplace.ops.ft_term import FtTerm
from ioplace.schedules import ScheduleState

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
    state.update_continuous(0, 0.5, 100.0, 1.0)
    assert normalizer.obj_version == state.obj_version
    normalizer.transaction(0, 0.5, state.tau, 1.0,
                           legacy_publish=lambda: publish_atomic(
                               state, io_term, ft_term, lambda p: (p ** 2).sum(),
                               pos, 0, state.tau / 100.0, ecc_max, 1.0))
    assert normalizer.obj_version == state.obj_version
    assert normalizer.needs_refresh() and state.needs_refresh()
    normalizer.mark_refreshed()
    assert not normalizer.needs_refresh() and not state.needs_refresh()


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_legacy_adapter.py -v`
Expected: `test_retired_path_golden_lambda_trajectory` PASSES (it only exercises today's `schedules.py`); the four adapter tests FAIL with `NotImplementedError: policy 'legacy' delegation is installed by the legacy adapter task`.

- [ ] **Step 3: Write the implementation**

Replace the `_legacy_transaction` stub in `src/ioplace/norm.py` with:

```python
    def _legacy_transaction(self, iteration, overflow, tau, gamma, legacy_publish):
        """Thin adapter over the retired path: `ops/ft_callback.publish_atomic`
        still measures the gradients, derives `kappa_ft` and runs
        `ScheduleState.apply_ft_transaction`, so `--norm-policy legacy`
        reproduces the recorded coefficients bit-for-bit. The normalizer only
        mirrors the results into its own state and trace row; it computes no
        coefficient of its own on this path."""
        if legacy_publish is None:
            raise ValueError("policy 'legacy' requires legacy_publish=<callable>")
        record = legacy_publish()
        state = self._legacy
        lambdas = {}
        self.wl_norm = record["grad_l1_wl"]
        io = self.states.get("io")
        if io is not None:
            lambdas["io"] = state.lambda_io
            io.grad_norm = record["grad_l1_io"]
            io.ratio_inst = record["ratio_inst"]
            io.ratio_ema = record["ratio_ema"]
            io.lam = state.lambda_io
            io.wt = state.rho * activation_ramp(iteration, state.it_activate,
                                                state.n_ramp)
            io.active = state.active
            io.it_activate = state.it_activate
        ft = self.states.get("ft")
        if ft is not None:
            lambdas["ft"] = state.lambda_io * state.kappa_ft
            ft.grad_norm = record["grad_l1_ft"]
            ft.lam = lambdas["ft"]
            ft.wt = record["f_ft"]
            ft.active = record["kappa_ft"] > 0.0
            ft.it_activate = state.it_activate
        self.lambdas = dict(lambdas)
        cap = lipschitz_cap(tau, gamma, state.c_lip, record["Cmax"])
        # `apply_ft_transaction` uses `min(base, cap)`, so the cap bound exactly
        # when the stored coefficient *is* the cap value.
        binding = "io" if state.lambda_io == cap else None
        row = self._row(iteration, overflow, tau, gamma, record["Cmax"], cap,
                        binding, record["cancellation_ratio"])
        self._pending_row = row
        return NormTransaction(lambdas=dict(lambdas), cmax=record["Cmax"], cap=cap,
                               cap_binding=binding,
                               cancellation_ratio=record["cancellation_ratio"],
                               obj_version=self.obj_version, row=row,
                               legacy_record=record)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_legacy_adapter.py tests/test_schedules.py tests/test_ft_callback.py -v`
Expected: all pass (5 new + 32 existing schedules tests + 3 ft_callback tests).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/norm.py tests/test_norm_legacy_adapter.py
git commit -m "feat(norm): add legacy policy adapter with golden lambda regression" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Driver wiring, CLI flags and documentation

**Files:**
- Modify: `src/ioplace/drivers/run_placement_io.py` (signature `:122-139`, validation `:140-154`, normalizer construction after `:279`, `term_fn` `:281-288`, callback `:458-461`, refresh `:584-588`, refreshed_version stamp `:589-590`, invariant `:592-594`, `lambda_io_final` `:612`, result dict `:683-734`, `RESULT_FIELDS` `:21-42`)
- Modify: `src/ioplace/drivers/run_placement.py:448-537`
- Modify: `docs/dev-env.md`
- Test: `tests/test_norm_driver.py`

**Interfaces:**
- Consumes: `ioplace.norm.TermNormalizer`, `ioplace.norm.VersionPair`, `ioplace.norm.parse_target_shares`, `ioplace.norm_trace.NormTraceWriter`, `ioplace.ops.norm_terms.IoNormTerm`, `ioplace.ops.norm_terms.FtNormTerm`, `ioplace.ops.ft_callback.publish_atomic`, `ioplace.dp_hook.install_version_invariant`.
- Produces:
  - `run_io(..., norm_policy="legacy", norm_p=1, norm_ramp_period=100, norm_wt_max=1.0, norm_probe_every=50, norm_target_share=None, norm_trace=None)`
  - `ioplace.drivers.run_placement.build_parser() -> argparse.ArgumentParser`
  - New `result` keys, all added to `RESULT_FIELDS`: `norm_policy`, `norm_p`, `norm_ramp_period`, `norm_wt_max`, `norm_probe_every`, `norm_target_share`, `norm_trace`, `lambda_ft_final`
  - Artefact: `<out>.norm_trace.jsonl` for non-legacy policies unless `--norm-trace` overrides it

- [ ] **Step 1: Write the failing tests**

Create `tests/test_norm_driver.py`:

```python
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
    (dict(norm_probe_every=0), "norm_probe_every"),
    (dict(norm_probe_every=75), "norm_probe_every"),        # not a multiple of every=50
    (dict(norm_policy="grandplan"), "atomic"),              # callback_order defaults legacy
])
def test_run_io_rejects_invalid_norm_configuration(kwargs, message, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        run_io("missing.json", 4, "grid", 0, str(tmp_path / "o.json"), **kwargs)
    assert message in str(excinfo.value)


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_driver.py -m "not slow" -v`
Expected: FAIL — `AttributeError: module 'ioplace.drivers.run_placement' has no attribute 'build_parser'` and `assert 'norm_policy' in RESULT_FIELDS`.

- [ ] **Step 3a: Extend `run_io`'s signature and validation**

In `src/ioplace/drivers/run_placement_io.py`, append to the keyword-only
parameters of `run_io` (after `discrete_max_active=65536`):

```python
           norm_policy="legacy", norm_p=1, norm_ramp_period=100,
           norm_wt_max=1.0, norm_probe_every=50, norm_target_share=None,
           norm_trace=None):
```

Immediately after the `home_period` validation (`if every <= 0 or home_period <= 0 or home_period % every: raise ...`), insert:

```python
    if norm_policy not in ("legacy", "grandplan", "adaptive"):
        raise ValueError("norm_policy must be legacy, grandplan or adaptive, "
                         "got %r" % (norm_policy,))
    if norm_p not in (1, 2):
        raise ValueError("norm_p must be 1 or 2, got %r" % (norm_p,))
    if norm_probe_every <= 0 or norm_probe_every % every:
        raise ValueError("norm_probe_every must be a positive multiple of every")
    if norm_policy != "legacy" and callback_order != "atomic":
        raise ValueError("norm_policy %r requires callback_order='atomic'"
                         % (norm_policy,))
```

Extend `RESULT_FIELDS` (`:21-42`) with a new trailing group:

```python
                 # v2 P-H (design sec 4): normalisation module configuration
                 # and artefact.
                 "norm_policy", "norm_p", "norm_ramp_period", "norm_wt_max",
                 "norm_probe_every", "norm_target_share", "norm_trace",
                 "lambda_ft_final")
```

- [ ] **Step 3b: Build the normalizer**

In `run_io`, immediately after the `observer_mode = (...)` assignment (`:279`),
insert:

```python
        from ioplace.norm import (TermNormalizer, VersionPair, _json_cap,
                                 parse_target_shares)
        from ioplace.norm_trace import NormTraceWriter
        from ioplace.ops.norm_terms import FtNormTerm, IoNormTerm
        shares = parse_target_shares(norm_target_share)
        norm_trace_path = norm_trace
        if norm_trace_path is None and norm_policy != "legacy" and not observer_mode:
            norm_trace_path = out_json + ".norm_trace.jsonl"
        trace_writer = None
        if norm_trace_path is not None and not observer_mode:
            trace_writer = NormTraceWriter(norm_trace_path)
            cleanup.callback(trace_writer.close)
        normalizer = TermNormalizer(
            policy=norm_policy, norm_p=norm_p, ema=state.ema,
            probe_every=norm_probe_every, ramp_period=norm_ramp_period,
            wt_max=norm_wt_max, c_lip=state.c_lip,
            num_movable=io_term.num_movable, num_nodes=io_term.num_nodes,
            # One cached gradient tensor per term is ~8 B/node/term; above 5M
            # nodes the cancellation diagnostic is not worth the residency.
            track_cancellation=(placedb.num_nodes <= 5_000_000),
            legacy_state=state if norm_policy == "legacy" else None,
            trace=trace_writer)
        normalizer.register("io", IoNormTerm(io_term), 1.0,
                            target_share=shares.get("io", 0.3),
                            activate_overflow=of_on, n_ramp=state.n_ramp)
        if ft_term is not None:
            # FT curvature is ecc_max (design sec 4). Its overflow gate is 0.30:
            # the tau_rel window the legacy ramp used (0.12 -> 0.05) maps through
            # tau_rel_from_overflow to overflow 0.57 -> 0.25, and 0.30 is the
            # same threshold the capacity term uses (design sec 5).
            normalizer.register("ft", FtNormTerm(ft_term), float(distance.max()),
                                target_share=shares.get("ft", f_ft_max),
                                activate_overflow=0.30, n_ramp=state.n_ramp)
```

- [ ] **Step 3c: Route `term_fn` through the normalizer**

Replace `term_fn` (`:281-288`) with:

```python
        def term_fn(pos):
            if norm_policy == "legacy":
                # Pass state.kappa_ft verbatim: recovering it as lambda_ft/lambda_io
                # would differ in the last ulp and break the legacy guarantee.
                lam_io, kappa = state.lambda_io, state.kappa_ft
            else:
                lam_io = normalizer.lambdas.get("io", 0.0)
                lam_ft = normalizer.lambdas.get("ft", 0.0)
                if lam_io <= 0.0 and lam_ft > 0.0:
                    raise RuntimeError(
                        "FtTerm expresses the FT coefficient as lambda_io*kappa, so "
                        "a nonzero lambda_ft with lambda_io == 0 cannot be applied")
                # Non-legacy: this path has no bit-exact guarantee to preserve
                # (unlike legacy above), so recovering kappa by division here,
                # accepting last-ulp rounding against the traced lambda_ft, is fine.
                kappa = lam_ft / lam_io if lam_io > 0.0 else 0.0
            if not state.active or (lam_io == 0.0 and state.lambda_margin == 0.0):
                return pos.new_zeros(())
            if ft_term is not None:
                return ft_term(pos, state.tau, lam_io, kappa,
                               state.lambda_margin, state.margin_m, state.margin_tau)
            return io_term(pos, state.tau, lam_io, state.lambda_margin,
                           state.margin_m, state.margin_tau)
```

- [ ] **Step 3d: Route the callback through the normalizer**

Replace the `from ioplace.ops.ft_callback import publish_atomic` /
`entry.update(publish_atomic(...))` pair (`:458-461`) with:

```python
                    wirelength_op = placer.model.op_collections.wirelength_op
                    if norm_policy == "legacy":
                        from ioplace.ops.ft_callback import publish_atomic
                        txn = normalizer.transaction(
                            iteration, of, state.tau, gamma,
                            legacy_publish=lambda: publish_atomic(
                                state, io_term, ft_term, wirelength_op, pos,
                                iteration, state.tau / L_R,
                                float(distance.max()) if distance is not None else 0.,
                                gamma))
                        entry.update(txn.legacy_record)
                    else:
                        ctx_norm = {"iteration": iteration, "overflow": of,
                                    "tau": state.tau, "gamma": gamma}
                        if normalizer.should_probe(iteration):
                            normalizer.probe(iteration, pos, wirelength_op, ctx_norm)
                        txn = normalizer.transaction(iteration, of, state.tau, gamma)
                        ft_state = normalizer.states.get("ft")
                        entry.update(grad_l1_wl=normalizer.wl_norm,
                                     grad_l1_io=normalizer.states["io"].grad_norm,
                                     grad_l1_ft=ft_state.grad_norm if ft_state else 0.,
                                     ratio_inst=normalizer.states["io"].ratio_inst,
                                     ratio_ema=normalizer.states["io"].ratio_ema,
                                     lambda_io=txn.lambdas.get("io", 0.),
                                     obj_version=txn.obj_version)
                    entry.update(lambda_ft=txn.lambdas.get("ft", 0.),
                                 norm_cmax=txn.cmax,
                                 norm_cap=_json_cap(txn.cap),
                                 cap_binding=txn.cap_binding,
                                 cancellation_ratio=txn.cancellation_ratio)
```

- [ ] **Step 3e: Refresh, invariant and final coefficients**

Replace the refresh block (`:584-588`) with:

```python
            # 順序不可換:先讓新 τ/λ/w 生效,再 refresh。
            if not observer_mode and (discrete or state.needs_refresh()
                                      or normalizer.needs_refresh()):
                refresh_nesterov_secant(placer.optimizer)
                state.mark_refreshed()
                normalizer.mark_refreshed()      # also emits the pending trace row
                cb_state["num_refreshes"] += 1
```

Replace the pre-existing `refreshed_version` stamp (`:589-590`,
`trajectory[-1]["refreshed_version"] = state.refreshed_version`) with:

```python
            if callback_order == "atomic" and trajectory and trajectory[-1]["iteration"] == iteration:
                trajectory[-1]["refreshed_version"] = (
                    state.refreshed_version if norm_policy == "legacy"
                    else normalizer.refreshed_version)
```

This keeps the trajectory row's `obj_version`/`refreshed_version` pair on the
same counter: legacy already writes `obj_version=txn.obj_version` from
`state` via `txn.legacy_record`/`state` delegation, and non-legacy writes
`obj_version=txn.obj_version` from the normalizer (Step 3d) -- so its
`refreshed_version` must come from the same source, not from `state`.

Replace the invariant install (`:592-594`) with:

```python
            if check_invariant and not cb_state["installed_invariant"]:
                # Under policy="legacy" the normalizer's obj_version/
                # refreshed_version delegate to `state`, so this VersionPair
                # sums the same counter twice; equality still holds iff both
                # are refreshed, so the invariant stays sound.
                cleanup.callback(install_version_invariant(
                    placer.optimizer, VersionPair(state, normalizer)))
                cb_state["installed_invariant"] = True
```

Replace `lambda_io_final = state.lambda_io` (`:612`) with:

```python
        lambda_io_final = (state.lambda_io if norm_policy == "legacy"
                           else normalizer.lambdas.get("io", 0.0))
        lambda_ft_final = (state.lambda_io * state.kappa_ft if norm_policy == "legacy"
                           else normalizer.lambdas.get("ft", 0.0))
```

Add to the `result` dict, next to `"margin_m": margin_m, "lambda_io_final": lambda_io_final,`:

```python
            "lambda_ft_final": lambda_ft_final,
            "norm_policy": norm_policy, "norm_p": norm_p,
            "norm_ramp_period": norm_ramp_period, "norm_wt_max": norm_wt_max,
            "norm_probe_every": norm_probe_every,
            "norm_target_share": norm_target_share,
            "norm_trace": (os.path.abspath(norm_trace_path)
                           if norm_trace_path is not None and not observer_mode
                           else None),
```

- [ ] **Step 3f: Add the CLI flags**

In `src/ioplace/drivers/run_placement.py`, rename `def main():` to
`def build_parser():`, end it with `return ap` in place of
`args = ap.parse_args()` and the dispatch, then add a new `main()` below it that
keeps the existing dispatch unchanged except for the forwarded norm kwargs.
Insert these arguments just before `args = ap.parse_args()` used to be:

```python
    # v2 P-H (design sec 4): gradient-norm normalisation for the extra
    # objective terms. mode=io only; no-op for the other modes.
    ap.add_argument("--norm-policy", choices=["legacy", "grandplan", "adaptive"],
                    default="legacy",
                    help="coefficient normalisation policy: 'legacy' reproduces "
                         "the retired lambda_IO EMA + kappa_FT force share exactly; "
                         "'grandplan' uses lambda_t = wt_t*||grad WL||_p/||grad T_t||_p "
                         "with wt stepping +0.05 every --norm-ramp-period iterations; "
                         "'adaptive' drives each term to a target force share")
    ap.add_argument("--norm-p", type=int, choices=[1, 2], default=1,
                    help="gradient norm order for every normalisation probe "
                         "(default 1: every calibrated constant was fitted under L1)")
    ap.add_argument("--norm-ramp-period", type=int, default=100,
                    help="policy grandplan: iterations between +0.05 wt steps")
    ap.add_argument("--norm-wt-max", type=float, default=1.,
                    help="policy grandplan: upper bound on wt (default 1.0)")
    ap.add_argument("--norm-probe-every", type=int, default=50,
                    help="iterations between normalisation probes; must be a "
                         "positive multiple of --every")
    ap.add_argument("--norm-target-share", default=None,
                    help="policy adaptive: per-term target force shares, "
                         "e.g. 'io=0.3,ft=0.1'")
    ap.add_argument("--norm-trace", default=None,
                    help="write norm_trace.jsonl here (default "
                         "<out>.norm_trace.jsonl for non-legacy policies, "
                         "no trace for legacy)")
```

and extend the `run_io(...)` call in `main()` with:

```python
              norm_policy=args.norm_policy, norm_p=args.norm_p,
              norm_ramp_period=args.norm_ramp_period, norm_wt_max=args.norm_wt_max,
              norm_probe_every=args.norm_probe_every,
              norm_target_share=args.norm_target_share, norm_trace=args.norm_trace,
```

`main()` starts with `args = build_parser().parse_args()`; everything else in
the dispatch is unchanged.

**Default note:** `--norm-policy` defaults to `legacy` here so P-H ships
without changing production behaviour. The default flips to `grandplan` only
when P-B's `run_main_flow.py` lands (see
`docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md`) -- that switch is
P-B's responsibility, not this task's.

- [ ] **Step 3g: Document the flags**

In `docs/dev-env.md`, immediately after the paragraph ending "Report missing
benchmark data or unavailable GPUs explicitly.", insert:

```markdown
### Normalisation module (P-H) flags

`ioplace.drivers.run_placement --mode io` normalises every extra objective term
through `ioplace.norm.TermNormalizer` (v2 design section 4):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--norm-policy` | `legacy` | `legacy` reproduces the retired λ_IO EMA + κ_FT force share exactly; `grandplan` is `λ_t = wt_t·‖∇WL‖_p/‖∇T_t‖_p`; `adaptive` targets a per-term force share |
| `--norm-p` | `1` | Gradient norm order (L1 default, L2 switch) |
| `--norm-ramp-period` | `100` | Policy `grandplan`: iterations between `+0.05` steps of `wt`, from `0.05` |
| `--norm-wt-max` | `1.0` | Policy `grandplan`: upper bound on `wt` |
| `--norm-probe-every` | `50` | Iterations between probes; must be a positive multiple of `--every` |
| `--norm-target-share` | unset | Policy `adaptive`: `io=0.3,ft=0.1` |
| `--norm-trace` | unset | Trace path; defaults to `<out>.norm_trace.jsonl` for non-legacy policies |

Non-legacy policies require `--callback-order atomic`. `norm_trace.jsonl` gets
one row per coefficient transaction (>= one per probe, since a transaction may
reuse the previous probe's measurements), with the per-term gradient norm,
instantaneous and EMA ratio, weight, coefficient, realised force share,
`cap_binding`, `cancellation_ratio` and the objective/refresh versions.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_driver.py -m "not slow" -v`
Expected: 9 passed.

Then, with a free GPU (`nvidia-smi`, then `export CUDA_VISIBLE_DEVICES=<idle>`):
Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_driver.py tests/test_driver_io.py -v`
Expected: all pass, including the three slow driver tests.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/drivers/run_placement_io.py src/ioplace/drivers/run_placement.py \
        docs/dev-env.md tests/test_norm_driver.py
git commit -m "feat(driver): route IO/FT coefficients through TermNormalizer" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Retire the routing one-shot ratio behind `IOPLACE_ENABLE_GR_IN_LOOP`

**Files:**
- Modify: `src/ioplace/ops/routing_gp_controller.py:1-8` (imports), `:17-35` (`__init__` gate), `:64-78` (`_calibrate`)
- Modify: `src/scripts/run_route_gp.py:1` (module docstring)
- Modify: `tests/test_routing_gp_driver.py:41` (subprocess `env=` for the now-gated construction)
- Test: `tests/test_routing_gp_retirement.py`

**Interfaces:**
- Consumes: `TermNormalizer.oneshot_lambda(strength, wl_norm, grad_norm, floor=1e-12) -> float` from Task 2.
- Produces:
  - `RoutingGPController.__init__` raises `RuntimeError` unless `os.environ.get("IOPLACE_ENABLE_GR_IN_LOOP") == "1"`.
  - `_calibrate` keeps its return dict unchanged (`paper_value`, `paper_gradient_l1`, the per-component `*_value`/`*_gradient_l1` keys, `wa_gradient_l1`, `route_lambda`, `requested_route_strength`, `effective_route_strength`) — only the coefficient expression moves into the normalizer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_routing_gp_retirement.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from ioplace.norm import TermNormalizer
from ioplace.ops.routing_gp_controller import RoutingGPController


def test_controller_is_retired_unless_the_escape_hatch_is_set(monkeypatch):
    monkeypatch.delenv("IOPLACE_ENABLE_GR_IN_LOOP", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        RoutingGPController(object(), object())
    assert "IOPLACE_ENABLE_GR_IN_LOOP" in str(excinfo.value)


def test_controller_constructs_behind_the_escape_hatch(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    controller = RoutingGPController(object(), object(), route_strength=.25)
    assert controller.route_strength == .25
    assert controller.route_lambda == 0.


def test_controller_still_validates_its_arguments(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    with pytest.raises(ValueError):
        RoutingGPController(object(), object(), mode="bogus")
    with pytest.raises(ValueError):
        RoutingGPController(object(), object(), start=0)
    with pytest.raises(ValueError):
        RoutingGPController(object(), object(), tau=0.)


def test_oneshot_lambda_matches_the_retired_expression():
    for strength, wa_l1, norm in ((.1, 1234.5, 6.78), (.25, 1e6, 3.),
                                  (1., 7.5, 2.5)):
        assert (TermNormalizer.oneshot_lambda(strength, wa_l1, norm)
                == strength * wa_l1 / norm)
    assert TermNormalizer.oneshot_lambda(.1, 1000., 1e-13) == 0.


def test_calibrate_uses_the_normalizer(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")

    class _Term(object):
        def paper(self, pos, gamma):
            return (pos ** 2).sum()

        def components(self, pos, tau, gamma):
            value = (pos ** 3).sum()
            return {"io": value, "wirelength": value, "congestion": value,
                    "objective": value}

    class _Model(object):
        class _Ops(object):
            wirelength_op = staticmethod(lambda p: 4. * p.sum())
        op_collections = _Ops()

    class _Placer(object):
        model = _Model()

    controller = RoutingGPController(_Term(), _Placer(), mode="joint",
                                     route_strength=.1)
    controller.gamma = 1.
    pos = torch.tensor([1., 2., 3.], dtype=torch.float64)
    stats = controller._calibrate(pos)
    wa_l1 = 12.                                     # |4| three times
    route_l1 = 3. + 12. + 27.                       # |3 p^2|
    assert stats["wa_gradient_l1"] == pytest.approx(wa_l1, rel=1e-12)
    assert stats["route_lambda"] == .1 * wa_l1 / route_l1
    assert controller.route_lambda == stats["route_lambda"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_routing_gp_retirement.py -v`
Expected: the first test FAILs (`DID NOT RAISE RuntimeError`); `test_calibrate_uses_the_normalizer` FAILs on the missing gate too.

- [ ] **Step 3: Write the implementation**

In `src/ioplace/ops/routing_gp_controller.py`, add `import os` to the standard
library imports and `from ioplace.norm import TermNormalizer` to the project
imports, then insert at the top of `__init__`, before the `mode` check:

```python
        if os.environ.get("IOPLACE_ENABLE_GR_IN_LOOP") != "1":
            raise RuntimeError(
                "in-loop GR is retired by the v2 design (sec 1): the final GRT "
                "protocol runs once, after placement. Set "
                "IOPLACE_ENABLE_GR_IN_LOOP=1 to use this unmaintained path.")
```

and replace the coefficient lines of `_calibrate` (`:72-75`) with:

```python
        norm = stats["route_gradient_l1"]
        # v2 P-H (design sec 4): the one-shot ratio normalisation now lives in
        # ioplace.norm; this is the third and last of the retired coefficient
        # paths, kept only as an adapter.
        self.route_lambda = TermNormalizer.oneshot_lambda(self.route_strength,
                                                          wa_l1, norm)
        if not math.isfinite(self.route_lambda):
            raise FloatingPointError("nonfinite routing coefficient")
```

In `src/scripts/run_route_gp.py`, replace the module docstring with:

```python
"""WA, paper Eq.7, and routing-gradient GP with measured OpenROAD feedback.

Retired by the v2 design (sec 1): requires IOPLACE_ENABLE_GR_IN_LOOP=1 and is
unmaintained. The supported protocol is one GRT call after placement.
"""
```

- [ ] **Step 3b: Keep the driver integration test exercising the gated path**

`tests/test_routing_gp_driver.py`'s `test_real_gp_route_gradient_and_two_later_router_observations`
launches `src/scripts/run_route_gp.py` as a subprocess, which now constructs
`RoutingGPController` and would raise unless the escape hatch is set. Pass it
through explicitly on that test's `subprocess.run` call:

```python
        run = subprocess.run(command, capture_output=True, text=True,
                             env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"})
```

Note for a later plan: P-B's Task 8
(`docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md`) relocates this gate
into `src/ioplace/gr_in_loop.py` as the single source of truth; the env var
name stays the same, only its enforcement point moves.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_routing_gp_retirement.py -v`
Expected: 5 passed.

Then confirm nothing else constructs the controller:
Run: `source src/scripts/env.sh && grep -rn "RoutingGPController" src/ tests/ && "$IOPLACE_PYTHON" -m pytest -m "not slow" -q`
Expected: `src/ioplace/ops/routing_gp_controller.py`, `src/scripts/run_route_gp.py`,
`tests/test_routing_gp_retirement.py` and `tests/test_routing_gp_driver.py:46`
(a pre-existing source-hash string literal naming the module path, not a
construction) match; the suite passes.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/routing_gp_controller.py src/scripts/run_route_gp.py \
        tests/test_routing_gp_retirement.py tests/test_routing_gp_driver.py
git commit -m "refactor(routing): reduce the one-shot route lambda to a norm adapter" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `mempool_group` acceptance — λ within 2× of the retired path

**Files:**
- Create: `tests/test_norm_group_validation.py`
- Create (run artefacts, not committed): `runs/norm-validation/`

**Interfaces:**
- Consumes: the `run_io` result JSON (`trajectory[*].iteration`, `trajectory[*].lambda_io`) from `--norm-policy legacy`, and `norm_trace.jsonl` rows (`iteration`, `terms.io.lam`) from `--norm-policy grandplan` — both produced by Task 7.
- Produces: the design's "done" evidence for P-H. Nothing imports this test.

**Why `--norm-wt-max 0.4`:** policy A's `wt` and the legacy path's `ρ_max` occupy the same slot in `λ = weight · ratio_ema`. Running policy A at `wt_max=1.0` against `--rho-max .40` would compare a deliberately different ramp, not the normalisation machinery, and would sit at 2.5× by construction. Matching them isolates what this subproject actually changed.

**Interpretation caveat:** matching `wt_max` to `rho_max` does not make the two
paths' `ratio_ema` denominators equal. Legacy's `ratio_ema` denominator is the
*merged* `‖∇IO + κ_FT·∇FT‖` (schedules.py's cancellation-aware measurement),
while policy A's `ratio_ema` denominator is the *isolated* `‖∇IO‖` that
`probe()` measures per term (C-1's derived-`kappa` `Cmax` also differs from
legacy's stored `Cmax` in general). A 2x miss in this acceptance test may
therefore stem from either the normalisation change itself or from this
pre-existing difference in what the two paths measure — do not attribute a
failure to one cause without checking both traces.

- [ ] **Step 1: Write the failing test**

Create `tests/test_norm_group_validation.py`:

```python
"""P-H acceptance (design sec 9, "Done per subproject" H): a group-scale run
whose lambda is within 2x of the retired path's lambda at matched iterations.

Opt-in: produce the two runs with the commands in
docs/superpowers/plans/2026-09-19-v2-p-h-normalisation.md Task 9, then point
IOPLACE_NORM_VALIDATION_DIR at the directory holding them.
"""
import json
import os
import pathlib

import pytest

from ioplace.norm_trace import read_norm_trace


@pytest.mark.slow
def test_grandplan_lambda_is_within_2x_of_the_retired_path():
    directory = os.environ.get("IOPLACE_NORM_VALIDATION_DIR")
    if not directory:
        pytest.skip("set IOPLACE_NORM_VALIDATION_DIR to the validation run directory")
    root = pathlib.Path(directory)
    legacy = json.loads((root / "legacy.json").read_text())
    assert legacy["norm_policy"] == "legacy"
    legacy_lambda = dict((int(event["iteration"]), float(event["lambda_io"]))
                         for event in legacy["trajectory"]
                         if float(event.get("lambda_io", 0.)) > 0.)

    grandplan = json.loads((root / "grandplan.json").read_text())
    assert grandplan["norm_policy"] == "grandplan"
    assert grandplan["norm_wt_max"] == legacy["rho_max"], \
        "compare matched weight ceilings, not different ramps"
    rows = read_norm_trace(grandplan["norm_trace"])
    new_lambda = dict((int(row["iteration"]), float(row["terms"]["io"]["lam"]))
                      for row in rows if float(row["terms"]["io"]["lam"]) > 0.)

    matched = sorted(set(legacy_lambda) & set(new_lambda))
    assert len(matched) >= 20, "only %d matched active iterations" % (len(matched),)
    tail = matched[len(matched) // 2:]          # after wt has finished ramping
    worst_iteration = max(tail, key=lambda i: max(new_lambda[i] / legacy_lambda[i],
                                                  legacy_lambda[i] / new_lambda[i]))
    worst = max(new_lambda[worst_iteration] / legacy_lambda[worst_iteration],
                legacy_lambda[worst_iteration] / new_lambda[worst_iteration])
    assert worst <= 2.0, (
        "lambda ratio %.3f at iteration %d (legacy %.6g, grandplan %.6g) over %d "
        "matched iterations" % (worst, worst_iteration,
                                legacy_lambda[worst_iteration],
                                new_lambda[worst_iteration], len(tail)))
```

- [ ] **Step 2: Run the test to verify it skips**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_norm_group_validation.py -v`
Expected: 1 skipped — "set IOPLACE_NORM_VALIDATION_DIR to the validation run directory".

- [ ] **Step 3: Reserve a GPU and build the validation config**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

Expected: a table of four H100 NVLs. Pick an index with low `memory.used` and
0% utilisation (index 3 was the idle one when this plan was written) and:

```bash
cd /ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
mkdir -p runs/norm-validation
"$IOPLACE_PYTHON" - <<'PY'
import json, pathlib
config = json.load(open("results/recovery_visible_20260906/configs/mempool_group.json"))
config.update(gpu=1, num_threads=16, plot_flag=0, detailed_place_flag=0)
pathlib.Path("runs/norm-validation/mempool_group.json").write_text(
    json.dumps(config, indent=1))
print(config["def_input"])
PY
```

Expected: prints
`/ldaphome/yyds-tsai-dev/benchmarks/ispd25/archive/extracted/visible/mempool_group/mempool_group.def`.
If that file does not exist, stop and report the missing benchmark — do not
substitute another case.

- [ ] **Step 4: Run the legacy arm**

```bash
"$IOPLACE_PYTHON" -m ioplace.drivers.run_placement \
  --config runs/norm-validation/mempool_group.json --mode io --k 16 --rtype grid \
  --seed 0 --dp-seed 1000 --deterministic 1 --every 50 --rho-max .40 \
  --tau-hi .30 --tau-lo .03 --callback-order atomic --no-diag \
  --f-ft-max .25 --home-period 50 --ft-ramp-mode window --tau-start .12 --tau-full .05 \
  --norm-policy legacy --out runs/norm-validation/legacy.json
```

Expected: exit 0 after roughly 1–2 h (group flat GP+LG is 660–2000 s; the IO+FT
terms and the evaluator callbacks dominate the rest). `runs/norm-validation/legacy.json`
exists with `"norm_policy": "legacy"` and a non-empty `trajectory`.

- [ ] **Step 5: Run the grandplan arm at the matched weight ceiling**

```bash
"$IOPLACE_PYTHON" -m ioplace.drivers.run_placement \
  --config runs/norm-validation/mempool_group.json --mode io --k 16 --rtype grid \
  --seed 0 --dp-seed 1000 --deterministic 1 --every 50 --rho-max .40 \
  --tau-hi .30 --tau-lo .03 --callback-order atomic --no-diag \
  --f-ft-max .25 --home-period 50 --ft-ramp-mode window --tau-start .12 --tau-full .05 \
  --norm-policy grandplan --norm-p 1 --norm-probe-every 50 \
  --norm-ramp-period 100 --norm-wt-max .40 \
  --out runs/norm-validation/grandplan.json
```

Expected: exit 0; `runs/norm-validation/grandplan.json.norm_trace.jsonl` exists
with one row per probe.

- [ ] **Step 6: Run the adaptive arm so both policies have group-scale evidence**

```bash
"$IOPLACE_PYTHON" -m ioplace.drivers.run_placement \
  --config runs/norm-validation/mempool_group.json --mode io --k 16 --rtype grid \
  --seed 0 --dp-seed 1000 --deterministic 1 --every 50 --rho-max .40 \
  --tau-hi .30 --tau-lo .03 --callback-order atomic --no-diag \
  --f-ft-max .25 --home-period 50 --ft-ramp-mode window --tau-start .12 --tau-full .05 \
  --norm-policy adaptive --norm-p 1 --norm-probe-every 50 \
  --norm-target-share io=0.3,ft=0.1 \
  --out runs/norm-validation/adaptive.json
```

Expected: exit 0; the trace's `terms.io.share` approaches 0.3 over the run.
Report the final realised shares with:

```bash
"$IOPLACE_PYTHON" -c "
from ioplace.norm_trace import read_norm_trace
rows = read_norm_trace('runs/norm-validation/adaptive.json.norm_trace.jsonl')
print(rows[-1]['iteration'], {n: round(t['share'], 4) for n, t in rows[-1]['terms'].items()})"
```

- [ ] **Step 7: Run the acceptance test**

```bash
IOPLACE_NORM_VALIDATION_DIR=runs/norm-validation \
  "$IOPLACE_PYTHON" -m pytest tests/test_norm_group_validation.py -v
```

Expected: 1 passed. On failure the message names the worst iteration and both
coefficients — report that, do not loosen the threshold.

- [ ] **Step 8: Run the full suite**

```bash
"$IOPLACE_PYTHON" -m pytest
```

Expected: the whole suite passes, including the slow placement integration tests.

- [ ] **Step 9: Commit**

```bash
git add tests/test_norm_group_validation.py
git commit -m "test(norm): add group-scale lambda parity acceptance for P-H" \
           -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage.** Every section-4 requirement maps to a task:

| Spec requirement | Task |
|---|---|
| New `src/ioplace/norm.py`, one class `TermNormalizer` | 1, 2 |
| `register(name, term, curvature)` | 2 |
| `probe(...)` every `probe_every=50`, WL-only backward + one isolated backward per term, `probe_terms` subsetting | 4 |
| `weights(iteration, overflow, tau, gamma) -> {name: λ}` | 2 |
| `transaction(...)`: one `obj_version` bump, refresh flag, `mark_refreshed()` | 2 (enforced: a second transaction before `mark_refreshed()` raises) |
| Policy A `grandplan` (wt 0.05, +0.05 per `ramp_period=100`, `wt_max=1.0`, overflow-gated) | 1, 2 |
| Policy B `adaptive` (target shares, sqrt update, momentum 0.75) | 1, 3 |
| Shared `ema=0.5`, activation ramps, cap | 1, 2, 3 |
| `Cmax = 1 + Σ_t κ_t(curv_t−1)_+` generalising `derive_cmax`; sum-clipping; `cap_binding` logged | 1, 2 |
| `norm_p=1` default with L2 switch | 2, 4, 7 |
| `norm_trace.jsonl` with the listed fields | 2, 5 |
| Three legacy paths reduced to adapters; routing behind `IOPLACE_ENABLE_GR_IN_LOOP=1` | 6, 7, 8 |
| Section 9 `tests/test_norm.py`: policy-A ramp, policy-B momentum, cap binding, one bump per transaction, refresh required (reusing `dp_hook.install_version_invariant`) | 1–4 |
| CLI flags `--norm-policy/--norm-p/--norm-ramp-period/--norm-wt-max/--norm-probe-every` | 7 |
| "Done" for H: both policies, trace emitted, legacy paths as adapters, group-scale λ within 2× | 9 |

Two deliberate scope boundaries, both stated in the tasks rather than left implicit:

- The spec lists `cap`, `pseudo_ft` and `group` as registrable terms. `register()` is term-agnostic and their declared curvatures are documented in its docstring (Task 2), but only `io` and `ft` are registered by the driver here — the other three terms are produced by P-D, P-E and P-C and will call `register()` themselves. No task in this plan should create them.
- The spec's `term.grad_l1(pos)` is folded into `TermNormalizer.probe` (Task 4), so all five terms share one gradient definition, one masking rule and one norm order — and the cached tensors are what `cancellation_ratio` needs. `IoTerm.io_grad_l1` is untouched and still serves the legacy callback path.

**2. Placeholder scan.** Searched the plan for `TBD`, `TODO`, "similar to Task", "fill in", "implement later", "add appropriate error handling", "handle edge cases", "as needed" and trailing ellipses: no matches. Every code step carries complete, runnable code; the only `NotImplementedError` in the plan is an explicit, tested stub in Task 2 that Task 6 replaces, and Task 6's Step 2 names the exact error text it expects to see first. Task 3 Step 2 is the one step whose tests may pass immediately (Task 2 already wrote that branch); the step says so and tells the executor what to do in that case rather than pretending a failure.

**3. Type consistency.** Cross-checked every name used across task boundaries:

- `TermNormalizer.transaction(iteration, overflow, tau, gamma, grad_norms=None, legacy_publish=None)` — same signature in Tasks 2, 3, 5, 6, 7, 9.
- `NormTransaction` fields `lambdas / cmax / cap / cap_binding / cancellation_ratio / obj_version / row / legacy_record / needs_refresh` — declared in Task 2, consumed in 4, 6, 7.
- `TermState` fields `grad_norm / ratio_inst / ratio_ema / wt / lam / active / it_activate` — declared in Task 2, written by the legacy adapter in Task 6, read by the driver in Task 7.
- Row keys: `grad_l1_wl` at top level, `grad_l1` per term — identical in `TermNormalizer._row` (Task 2), `ROW_FIELDS`/`TERM_FIELDS` (Task 5), and the assertions in Tasks 5, 6, 9.
- `probe(iteration, pos, wl_fn, ctx, probe_terms=None)` and the `ctx` keys `{"iteration", "overflow", "tau", "gamma"}` — Task 4 defines them, Task 7 supplies exactly those keys, `IoNormTerm`/`FtNormTerm` read only `ctx["tau"]`.
- `VersionPair` (Task 2) is the single object handed to `install_version_invariant` (Task 7); the plan explains why two stacked wrappers would break `refresh_nesterov_secant`'s single-level `__wrapped__` unwrap.
- `TermNormalizer.oneshot_lambda` (Task 2) is the only coefficient expression in Task 8, written in the same operand order as the retired `_calibrate` so the adapter is bit-exact.
- `read_norm_trace` (Task 5) is the only reader used by Tasks 7 and 9.
- λ_ft is always `λ_io·κ` on the `FtTerm` side; Task 7's `term_fn` passes `state.kappa_ft` verbatim on the legacy path (recovering it by division would differ in the last ulp) and raises rather than silently dropping a nonzero λ_ft when λ_io is zero.
