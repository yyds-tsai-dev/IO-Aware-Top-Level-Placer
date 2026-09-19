"""Unified gradient-norm normalisation for extra objective terms (v2 design
sec 4, P-H).

This module replaces three ad-hoc coefficient paths: the lambda_IO EMA plus
Lipschitz cap (`schedules.py:94-104,159-176`), the kappa_FT force share
(`schedules.py:66-81,178-237`), and the one-shot route lambda
(`ops/routing_gp_controller.py:64-78`).

Everything above `TermNormalizer` is a total function of floats: no torch, no
DREAMPlace, no CUDA, so the policy maths is testable on any host.
"""

import math
from dataclasses import dataclass
from typing import Optional

from ioplace.schedules import activation_ramp, lipschitz_cap

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
    """`Cmax = sum_t lambda_t*curv_t / sum_t lambda_t`, the pre-clip
    lambda-weighted mean curvature (controller ruling 2026-09-19, fix round 1
    -- replaces the derived-kappa form `1 + sum_t kappa_t(curv_t-1)_+`, which
    degenerated whenever `lambda_io == 0`). This form is exactly the legacy
    bound `sum_t lambda_t*curv_t <= c_lip*tau^2/gamma` when only IO is active
    (`curv_io == 1`), and has no such degeneracy. `items` is an iterable of
    `(lambda, curvature)` pairs over the *active* terms only. Returns `1.0`
    when the total lambda is zero (including an empty `items`): there is no
    force to weight the mean by."""
    items = list(items)
    total = sum(lam for lam, _ in items)
    if total <= 0.0:
        return 1.0
    return sum(lam * curv for lam, curv in items) / total


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
    `{"io": 0.3, "ft": 0.1}`. Empty or `None` -> `{}`.

    Value validation (review I6): duplicates, non-finite values and values
    outside `[0, 1]` are rejected here. A `nan` share is otherwise invisible
    -- `adaptive_lambda`'s `target_share <= 0.0` guard is False for `nan`, the
    sqrt update yields `nan`, and `clip_sum_to_cap` then propagates it to
    every term. Unknown *names* cannot be checked here (this function has no
    term registry); the driver validates them against `normalizer.configs`
    right after it registers its terms."""
    if not spec:
        return {}
    shares = {}
    for item in spec.split(","):
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError("target share must be name=value, got %r" % (item,))
        if name in shares:
            raise ValueError("duplicate target share for %r in %r" % (name, spec))
        share = float(value)
        if not math.isfinite(share):
            raise ValueError("target share for %r must be finite, got %r"
                             % (name, share))
        if share < 0.0 or share > 1.0:
            raise ValueError("target share for %r must lie in [0, 1], got %r"
                             % (name, share))
        shares[name] = share
    return shares


def _json_cap(cap):
    """`inf` is not valid JSON; normalise it to `None` for logging. Shared by
    `TermNormalizer._row` and the driver's non-legacy trajectory entry (Task 7
    Step 3d)."""
    return None if cap == float("inf") else cap


@dataclass
class TermConfig:
    """Static registration data for one extra objective term. There is no
    per-term `kappa`: `Cmax` is the pre-clip lambda-weighted mean of every
    active term's `curvature` (controller ruling 2026-09-19; see `register`'s
    docstring and `_compute`).

    `requires` (review C1) names the term this one's coefficient is expressed
    *relative to* -- `ft`'s applied force is `lambda_io * kappa`, so a
    `lambda_ft > 0` with `lambda_io == 0` cannot be applied at all. The
    `wt0`/`wt_step`/`ramp_period`/`wt_max` fields (review I4) are per-term
    overrides of the normalizer-wide policy-A weight schedule; `None` means
    "use the normalizer's value"."""
    name: str
    curvature: float = 1.0
    target_share: float = 0.0
    activate_overflow: float = 0.90
    n_ramp: int = 20
    requires: Optional[str] = None
    wt0: Optional[float] = None
    wt_step: Optional[float] = None
    ramp_period: Optional[int] = None
    wt_max: Optional[float] = None


@dataclass
class TermState:
    """Live per-term state. `wt` holds policy A's stepped weight or policy B's
    target share, whichever the active policy produced.

    Three coefficient slots, not one (reviews I1/I7):

    * `lam` -- the committed, UN-ramped coefficient this transaction
      published. This is what the trace's `lam` column and `self.lambdas`
      carry.
    * `lam_applied` -- `lam` scaled by this term's activation ramp at the
      transaction's iteration. The driver recomputes the same product every GP
      iteration through `applied_lambda()`, so the coefficient actually
      applied drifts continuously across the ramp window exactly as the
      retired path's `rho * ramp * ratio_ema` did; this field records its
      value at the transaction itself.
    * `lam_state` -- policy B's momentum state. A probe that publishes 0 (a
      dead gradient, or a dependent term whose base died) leaves this
      untouched, so the next healthy probe resumes the multiplicative update
      from the converged value instead of re-bootstrapping (review I7)."""
    grad_norm: float = 0.0
    ratio_inst: Optional[float] = None
    ratio_ema: Optional[float] = None
    wt: float = 0.0
    lam: float = 0.0
    lam_applied: float = 0.0
    lam_state: float = 0.0
    kappa_clamped: bool = False
    active: bool = False
    it_activate: Optional[int] = None


@dataclass
class NormTransaction:
    """Result of one atomic coefficient update."""
    lambdas: dict
    cmax: float
    cap: float
    cap_binding: Optional[str]
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
                 momentum=0.75, c_lip=1.0, eps_rel=1e-3, kappa_max=100.0,
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
        # Mirrors ScheduleState.kappa_max (schedules.py): the non-legacy arm
        # recovers kappa = lam_dependent/lam_base, which a denormal base
        # coefficient would blow up (review M2).
        self.kappa_max = float(kappa_max)
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
        self._probed_names = set()
        self._probed_this_callback = False
        self._last_probe_iteration = None
        # Terms whose activation clock is supplied from outside (review I1a);
        # `_activate`'s own overflow gate is only a fallback for the rest.
        self._external_activation = set()

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
                 activate_overflow=0.90, n_ramp=20, requires=None,
                 wt0=None, wt_step=None, ramp_period=None, wt_max=None):
        """Register one term. `term` exposes `value(pos, ctx) -> Tensor`;
        `probe()` owns the backward, the masking and the norm order.

        Declared curvatures (design sec 4): IO 1, FT `ecc_max`, capacity
        `max_s pen''`, pseudo-FT 1. `curvature` feeds `Cmax`'s pre-clip
        lambda-weighted mean over every active term (controller ruling
        2026-09-19, `_compute`) -- there is no per-term kappa to register.

        `requires` (review C1) names an already-registered term this one's
        coefficient is expressed relative to: `_compute` publishes 0 for this
        term whenever that base term's coefficient is 0, because the force
        `lam_base * kappa` cannot be applied without a base. It also bounds
        the recovered `kappa = lam/lam_base` by `kappa_max` (review M2).

        `wt0`/`wt_step`/`ramp_period`/`wt_max` (review I4) override the
        normalizer-wide policy-A weight schedule for this term only; `None`
        (the default) keeps the normalizer's value, so registering two terms
        without overrides behaves exactly as before.

        Range checks (review M3): a mis-registered term used to fail silently
        -- `curvature=0` made `cmax=0`, which `lipschitz_cap`'s `max(1.0,
        cmax)` swallowed, so the term got no cap protection at all and
        `cap_binding` became meaningless."""
        if name in self.configs:
            raise ValueError("term %r already registered" % (name,))
        curvature = float(curvature)
        if not math.isfinite(curvature) or curvature < 1.0:
            raise ValueError("curvature must be a finite value >= 1.0 (1.0 = "
                             "no extra curvature), got %r for term %r"
                             % (curvature, name))
        n_ramp = int(n_ramp)
        if n_ramp < 0:
            raise ValueError("n_ramp must be >= 0, got %r for term %r"
                             % (n_ramp, name))
        activate_overflow = float(activate_overflow)
        # Upper bound intentionally not enforced: `run_io` passes `--of-on`
        # straight through as the IO term's gate, and `--of-on 2` is the
        # established "always active" setting in the driver tests.
        if not math.isfinite(activate_overflow) or activate_overflow <= 0.0:
            raise ValueError("activate_overflow must be a finite value > 0, "
                             "got %r for term %r" % (activate_overflow, name))
        target_share = float(target_share)
        if not math.isfinite(target_share) or not 0.0 <= target_share <= 1.0:
            raise ValueError("target_share must lie in [0, 1], got %r for "
                             "term %r" % (target_share, name))
        if requires is not None and requires not in self.configs:
            raise ValueError("term %r requires %r, which is not registered "
                             "(registered: %r)"
                             % (name, requires, sorted(self.configs)))
        overrides = {}
        for label, value in (("wt0", wt0), ("wt_step", wt_step),
                             ("ramp_period", ramp_period), ("wt_max", wt_max)):
            if value is None:
                overrides[label] = None
                continue
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be > 0 when given, got %r for term %r"
                                 % (label, value, name))
            overrides[label] = int(value) if label == "ramp_period" else value
        self.configs[name] = TermConfig(name, curvature, target_share,
                                        activate_overflow, n_ramp, requires,
                                        overrides["wt0"], overrides["wt_step"],
                                        overrides["ramp_period"],
                                        overrides["wt_max"])
        self.terms[name] = term
        self.states[name] = TermState()
        self.lambdas[name] = 0.0

    def _schedule(self, name, attr):
        """Per-term policy-A schedule value with the normalizer-wide default
        (review I4)."""
        value = getattr(self.configs[name], attr)
        return getattr(self, attr) if value is None else value

    def should_probe(self, iteration):
        return iteration % self.probe_every == 0

    def update_grad_norms(self, grad_norms):
        """Absorb one probe's measurements. `grad_norms` maps `"wl"` and each
        registered term name to its `||.||_p` gradient norm; unmentioned terms
        keep their previous measurement (this is how `probe_terms` subsetting
        stays correct).

        Controller ruling (fix round 1): a term whose measured gradient is at
        or below `eps_rel * wl_norm` (including exactly zero) has no local
        signal, and dividing by it would seed the EMA with a huge, meaningless
        ratio (`wl_norm / EPS` ~ 1e30-ish) that then poisons `ratio_ema` for
        many probes. Such a probe leaves `ratio_ema` untouched (`None` if
        never seeded) and records `ratio_inst = None` so the trace row shows
        the dead probe rather than a manufactured number."""
        self.wl_norm = float(grad_norms["wl"])
        for name, state in self.states.items():
            if name not in grad_norms:
                continue
            grad_norm = float(grad_norms[name])
            state.grad_norm = grad_norm
            if grad_norm <= self.eps_rel * self.wl_norm:
                state.ratio_inst = None
                continue
            state.ratio_inst = self.wl_norm / max(grad_norm, EPS)
            state.ratio_ema = ema_update(state.ratio_ema, state.ratio_inst, self.ema)

    # -- coefficient computation -------------------------------------------
    def _activate(self, iteration, overflow):
        """Latch each term on the first iteration at or below its activation
        overflow threshold. Monotone and idempotent, so both `weights()` and
        `transaction()` may call it. Terms with an external activation clock
        (`set_activation`) are skipped -- their clock is authoritative."""
        for name, config in self.configs.items():
            state = self.states[name]
            if name in self._external_activation:
                continue
            if not state.active and overflow <= config.activate_overflow:
                state.active, state.it_activate = True, iteration

    def set_activation(self, name, it_activate):
        """Install an external activation clock for one term (review I1a).

        `TermState.it_activate` used to latch on the first `every`-gated
        callback at or below the term's overflow threshold, which differs from
        the true activation iteration by up to `every` (50 by default) -- and
        the activation ramp is `n_ramp=20` iterations long, so the whole soft
        start was aliased away. The driver owns a per-iteration clock already
        (`ScheduleState.it_activate`, latched inside `update_continuous` on
        every GP iteration); this hands it to the normalizer so both arms ramp
        off the same instant. Monotone and idempotent: the first non-`None`
        value wins."""
        if it_activate is None:
            return
        state = self.states[name]
        self._external_activation.add(name)
        if not state.active:
            state.active, state.it_activate = True, int(it_activate)

    def update_activation(self, iteration, overflow):
        """Per-iteration activation latch for terms with no external clock
        (review I1a). The driver calls this every GP iteration, not only on
        `every`-gated callbacks, so `it_activate` is the real activation
        iteration."""
        self._activate(iteration, overflow)

    def applied_lambda(self, name, iteration):
        """The coefficient to actually apply at `iteration`: the committed
        (un-ramped) lambda scaled by this term's activation ramp (review I1b).

        `self.lambdas` only changes inside `transaction()`, i.e. every `every`
        iterations. The retired path recomputed `lambda_io = rho * ramp *
        ratio_ema` on *every* GP iteration (`ScheduleState.update_continuous`),
        so its coefficient drifted continuously through the ramp window
        instead of stepping once. A driver's `term_fn` calls this every
        iteration to reproduce that drift for the non-legacy policies; the
        drift carries no `obj_version` bump, exactly as it did not on the
        retired path (the ramp is a known, monotone function of the iteration
        counter, not a new measurement)."""
        state = self.states.get(name)
        if state is None or state.it_activate is None:
            return 0.0
        lam = self.lambdas.get(name, 0.0)
        if lam == 0.0:
            return 0.0
        return lam * activation_ramp(iteration, state.it_activate,
                                     self.configs[name].n_ramp)

    def _compute(self, iteration, tau, gamma):
        """Pure: returns `(lambdas, weights, cmax, cap, cap_binding,
        lam_states, kappa_clamped)` from the current measurements without
        mutating anything.

        `lambdas` are the committed, UN-ramped coefficients (the activation
        ramp is applied per iteration by `applied_lambda`, review I1b).
        `lam_states` are policy B's momentum states, which a dead or
        dependency-zeroed probe leaves untouched (review I7).

        Tiny-gradient handling (controller ruling F1): a term whose gradient
        is at or below `eps_rel * ||grad WL||` but not exactly zero has no
        usable ratio, but it is *not* dead -- the retired path let `ratio_inst`
        explode in that regime and simply saturated `lambda_io` at the
        Lipschitz cap, which is why its IO force persisted to the end of GP.
        Zeroing the coefficient instead (the pre-F1 behaviour) collapsed the
        IO term around iteration 1000 of the group-scale acceptance run. Such
        a term therefore takes the cap's remaining headroom, and `ratio_ema`
        stays untouched. A gradient of exactly 0.0 still gets 0: there is no
        crossing to pull on, so any coefficient is inert.

        Caveat on the headroom: `Cmax` (and so the cap) is derived from the
        other terms' pre-clip lambdas, as the ruling specifies, so a saturating
        term with `curvature > Cmax` can push the realised weighted mean above
        the bound. `io` (curvature 1) is the term that saturates in practice;
        for it the bound is exact."""
        total_force = self.wl_norm + sum(
            self.states[n].lam_state * self.states[n].grad_norm
            for n in self.configs if self.states[n].active)
        lambdas, weights, lam_states = {}, {}, {}
        saturating, healthy = [], []
        for name, config in self.configs.items():
            state = self.states[name]
            lam_states[name] = state.lam_state
            if not state.active or state.ratio_ema is None:
                lambdas[name], weights[name] = 0.0, 0.0
                continue
            if self.policy == "grandplan":
                weights[name] = grandplan_weight(
                    iteration, state.it_activate, self._schedule(name, "wt0"),
                    self._schedule(name, "wt_step"),
                    self._schedule(name, "ramp_period"),
                    self._schedule(name, "wt_max"))
            elif self.policy == "adaptive":
                weights[name] = config.target_share
            else:
                raise ValueError(
                    "policy %r has no coefficient rule ('grandplan' and "
                    "'adaptive' compute here; 'legacy' must delegate before "
                    "reaching _compute)" % (self.policy,))
            if state.grad_norm == 0.0:
                lambdas[name] = 0.0
                continue
            if state.grad_norm <= self.eps_rel * self.wl_norm:
                lambdas[name] = 0.0          # placeholder; resolved after the cap
                saturating.append(name)
                continue
            if self.policy == "grandplan":
                lambdas[name] = grandplan_lambda(weights[name], state.ratio_ema,
                                                 self.wl_norm, state.grad_norm,
                                                 self.eps_rel)
            else:
                lambdas[name] = adaptive_lambda(state.lam_state, state.grad_norm,
                                                weights[name], total_force,
                                                self.wl_norm, self.momentum,
                                                self.eps_rel)
            healthy.append(name)
        # Cmax is the pre-clip lambda-weighted mean curvature (controller
        # ruling 2026-09-19, fix round 1): Cmax = sum_t lambda_t*curv_t /
        # sum_t lambda_t. This replaces the derived-kappa form
        # `1 + kappa_t(curv_t-1)_+`, which divided by lambda_io and so
        # degenerated whenever lambda_io == 0. The new form is exactly the
        # legacy bound sum_t lambda_t*curv_t <= c_lip*tau^2/gamma when only IO
        # is active (curv_io == 1), with no such degeneracy.
        preclip = lambdas
        cmax = cmax_from_curvatures(
            [(preclip[name], self.configs[name].curvature)
             for name in self.configs if self.states[name].active])
        cap = lipschitz_cap(tau, gamma, self.c_lip, cmax)
        total = sum(preclip.values())
        # cap_binding is the term with the largest pre-clip lambda_t*curv_t --
        # the dominant contributor to the Lipschitz bound this cap enforces --
        # not merely the largest raw lambda_t (clip_sum_to_cap's own binding
        # answers that different question). Ties break alphabetically,
        # matching clip_sum_to_cap's convention. `None` exactly when
        # clip_sum_to_cap will not scale (same condition it uses internally).
        if total > cap and total > 0.0:
            binding = max(sorted(preclip),
                         key=lambda n: preclip[n] * self.configs[n].curvature)
        else:
            binding = None
        lambdas, _ = clip_sum_to_cap(preclip, cap)
        if saturating:
            if math.isfinite(cap):
                headroom = max(cap - sum(lambdas.values()), 0.0) / len(saturating)
                for name in saturating:
                    lambdas[name] = headroom
            else:
                # gamma <= 0: there is no Lipschitz bound to saturate against,
                # so hold the last published coefficient rather than invent an
                # unbounded one.
                for name in saturating:
                    lambdas[name] = max(self.states[name].lam_state, 0.0)
        for name in healthy:
            lam_states[name] = lambdas[name]
        # Review C1: a term registered with `requires=<base>` expresses its
        # force as `lambda_base * kappa`, so it cannot be applied at all once
        # the base coefficient is zero -- and the retired failure mode was
        # worse than useless, because the FT coefficient *rose* at exactly the
        # probe where IO's signal vanished. Zero the published coefficient (and
        # its weight) but leave `lam_states` alone, so policy B resumes from
        # its converged value when the base recovers (review I7). Iterated so a
        # chain of dependencies collapses in one transaction.
        for _ in range(len(self.configs)):
            changed = False
            for name, config in self.configs.items():
                base = config.requires
                if base is None or lambdas.get(name, 0.0) == 0.0:
                    continue
                if lambdas.get(base, 0.0) <= 0.0:
                    lambdas[name], weights[name] = 0.0, 0.0
                    changed = True
            if not changed:
                break
        # Review M2: mirror `ScheduleState.kappa_max`. The applied force is
        # already bounded by the cap; this bounds the *conditioning* of the
        # kappa = lam/lam_base the driver recovers by division.
        kappa_clamped = dict((name, False) for name in self.configs)
        for name, config in self.configs.items():
            base = config.requires
            if base is None or lambdas.get(name, 0.0) <= 0.0:
                continue
            ceiling = self.kappa_max * lambdas.get(base, 0.0)
            if lambdas[name] > ceiling:
                lambdas[name] = ceiling
                kappa_clamped[name] = True
        return lambdas, weights, cmax, cap, binding, lam_states, kappa_clamped

    def weights(self, iteration, overflow, tau, gamma):
        """Pure preview of the coefficients. Latches activation (monotone) but
        commits nothing and never bumps `obj_version`."""
        if self._legacy is not None:
            raise NotImplementedError(
                "policy 'legacy' has no pure preview: coefficients only "
                "exist after publish_atomic measures gradients")
        self._activate(iteration, overflow)
        return self._compute(iteration, tau, gamma)[0]

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
        probed) live only until the next `transaction()`, which clears them
        once it has used them for `cancellation_ratio`.

        Raises `ValueError` if this normalizer was built without
        `num_movable`/`num_nodes` (fix round 1, I-3): without them the fixed
        and filler mask silently becomes a no-op, and the returned norms would
        silently include fixed/filler gradient mass. Raises `KeyError` if
        `probe_terms` names a term that was never `register()`ed."""
        if self.num_movable is None or self.num_nodes is None:
            raise ValueError(
                "probe() requires num_movable/num_nodes for the fixed/filler mask")
        names = list(self.configs) if probe_terms is None else list(probe_terms)
        unknown = [name for name in names if name not in self.terms]
        if unknown:
            raise KeyError("probe_terms %r not registered (registered: %r)" %
                           (unknown, sorted(self.terms)))
        self._last_probe_iteration = iteration
        self._probed_names = set(names)
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

    def _grad(self, fn, pos, label="?"):
        """Isolated forward+backward on a detached clone, with the fixed and
        filler entries zeroed exactly as `ops/ft_callback.independent_gradient`
        does, so the live `pos.grad` is never disturbed."""
        import torch
        leaf = pos.detach().clone().requires_grad_(True)
        try:
            grad, = torch.autograd.grad(fn(leaf), leaf)
        except Exception as exc:
            raise RuntimeError("norm probe term %r: %s" % (label, exc)) from exc
        if self.num_movable is not None and self.num_nodes is not None:
            grad[self.num_movable:self.num_nodes] = 0
            grad[self.num_nodes + self.num_movable:] = 0
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("nonfinite probe gradient for term %r" % (label,))
        return grad

    def _norm(self, tensor):
        return (float(tensor.abs().sum()) if self.norm_p == 1
                else float(tensor.norm(p=2)))

    def _cancellation_ratio(self, lambdas):
        """`||sum_t lam_t grad T_t||_p / sum_t lam_t ||grad T_t||_p`, the N-term
        generalisation of `ScheduleState.cancellation_ratio`. `None` when this
        normalizer was built with `track_cancellation=False`, when this
        callback did not call `probe()` (the cache would otherwise mix stale
        gradients from an earlier probe with this transaction's coefficients),
        when the probe kept no gradient tensors, or when the most recent probe
        used `probe_terms` to measure only a subset of terms and one of the
        *other*, unprobed terms has a nonzero coefficient this transaction
        (fix round 1, I-2: mixing this transaction's full-lambda coefficients
        with a partial gradient cache would silently under-count the
        denominator and misreport the ratio)."""
        if not self.track_cancellation:
            return None
        if not self._probed_this_callback:
            return None
        if any(lam != 0.0 and name not in self._grad_cache
               for name, lam in lambdas.items()):
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
            "probe_iteration": (None if self._last_probe_iteration is None
                               else int(self._last_probe_iteration)),
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
                        "wt_max": self._schedule(name, "wt_max"),
                        "target_share": self.configs[name].target_share,
                        "lam": s.lam, "lam_applied": s.lam_applied,
                        "share": (s.lam * s.grad_norm / denom) if denom > 0.0 else 0.0,
                        "kappa_clamped": s.kappa_clamped,
                        "active": s.active})
                for name, s in self.states.items()),
        }

    def transaction(self, iteration, overflow, tau, gamma, grad_norms=None,
                    legacy_publish=None):
        """Atomic coefficient update: derive every lambda, commit it, bump
        `obj_version` exactly once, and hand the driver a transaction whose
        `needs_refresh` flag means "call `refresh_nesterov_secant(optimizer)`
        then `mark_refreshed()` before the next optimizer step".

        The re-entrancy guard below is path-independent: it fires whenever
        this normalizer's own `_pending_row` is still set, i.e. the previous
        transaction's row was never finalised by `mark_refreshed()`. This
        matters for `policy="legacy"`: `obj_version`/`refreshed_version`
        delegate entirely to the `ScheduleState`, whose own
        `update_continuous()` -- called by the driver *before* this method,
        exactly as the retired path always did -- may bump `obj_version`
        itself (a discrete activation or margin toggle) without an
        intervening refresh; that bump and this transaction's
        `apply_ft_transaction` bump are both reconciled by one
        `mark_refreshed()` call at the end of the iteration (see the retired
        golden trajectory and `run_placement_io.py`'s own
        `discrete or state.needs_refresh()` refresh gate). Guarding on
        `self.needs_refresh()` would misfire on that legitimate
        same-iteration bump; guarding on `_pending_row` instead tracks only
        *this* normalizer's own uncommitted transaction, so legacy gets a
        real double-transaction detector without being fooled by
        `ScheduleState`'s independent version bump."""
        if self._pending_row is not None:
            raise RuntimeError(
                "previous transaction was not refreshed: call "
                "refresh_nesterov_secant(optimizer) then mark_refreshed()")
        if self._legacy is not None:
            if grad_norms is not None:
                raise ValueError(
                    "policy 'legacy' does not accept grad_norms: gradients "
                    "are measured by publish_atomic (via legacy_publish), "
                    "not update_grad_norms")
            return self._legacy_transaction(iteration, overflow, tau, gamma,
                                            legacy_publish)
        if grad_norms is not None:
            self.update_grad_norms(grad_norms)
        self._activate(iteration, overflow)
        (lambdas, weights, cmax, cap, binding, lam_states,
         kappa_clamped) = self._compute(iteration, tau, gamma)
        # Review I6: no derived coefficient was ever checked for finiteness,
        # unlike the probe gradients. A `nan` share or ratio otherwise reaches
        # the objective term itself, where every downstream guard
        # (`lam <= 0.0`, `target_share <= 0.0`) silently evaluates False.
        # Raised before anything is committed, so the normalizer stays usable.
        for name, value in lambdas.items():
            if not math.isfinite(value):
                raise FloatingPointError("non-finite lambda for %r" % (name,))
        for name, state in self.states.items():
            state.lam, state.wt = lambdas[name], weights[name]
            state.lam_state = lam_states[name]
            state.kappa_clamped = kappa_clamped.get(name, False)
            state.lam_applied = (
                0.0 if state.it_activate is None else
                lambdas[name] * activation_ramp(iteration, state.it_activate,
                                                self.configs[name].n_ramp))
        # Mutate the live dict in place: `self.lambdas` is the object a
        # driver's `term_fn` closure captured a reference to, and rebinding
        # `self.lambdas = dict(lambdas)` would leave that closure reading a
        # stale copy forever.
        self.lambdas.clear()
        self.lambdas.update(lambdas)
        cancellation = self._cancellation_ratio(lambdas)
        self._grad_cache = {}
        self._probed_names = set()
        self._probed_this_callback = False
        self._obj_version += 1
        row = self._row(iteration, overflow, tau, gamma, cmax, cap, binding,
                        cancellation)
        self._pending_row = row
        return NormTransaction(lambdas=dict(lambdas), cmax=cmax, cap=cap,
                               cap_binding=binding, cancellation_ratio=cancellation,
                               obj_version=self.obj_version, row=row)

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
            # Guard against `state.it_activate is None` before the term ever
            # activates (e.g. the first callback lands above `of_on`):
            # `activation_ramp` requires a numeric `it_activate`, and
            # `update_continuous` already forces `state.rho == 0.0` while
            # inactive, so the ramped weight is 0.0 either way -- this guard
            # changes no coefficient, only avoids the `None` arithmetic.
            io.wt = (state.rho * activation_ramp(iteration, state.it_activate,
                                                 state.n_ramp)
                     if state.active else 0.0)
            io.active = state.active
            io.it_activate = state.it_activate
            # The retired path bakes its own ramp into `lambda_io`
            # (`update_continuous`/`apply_ft_transaction`), so the committed
            # and applied coefficients are the same number here. Additive
            # trace bookkeeping only -- no coefficient changes on this path.
            io.lam_applied = state.lambda_io
            io.lam_state = state.lambda_io
            io.kappa_clamped = False
        ft = self.states.get("ft")
        if ft is not None:
            lambdas["ft"] = state.lambda_io * state.kappa_ft
            ft.grad_norm = record["grad_l1_ft"]
            ft.lam = lambdas["ft"]
            ft.wt = record["f_ft"]
            ft.active = record["kappa_ft"] > 0.0
            ft.it_activate = state.it_activate
            ft.lam_applied = lambdas["ft"]
            ft.lam_state = lambdas["ft"]
            ft.kappa_clamped = bool(record["kappa_clamped"])
        # Mutate the live dict in place, exactly like the non-legacy branch
        # above: `self.lambdas` is the object a driver's `term_fn` closure
        # captured a reference to, and rebinding it would leave that closure
        # reading a stale copy forever.
        self.lambdas.clear()
        self.lambdas.update(lambdas)
        cap = lipschitz_cap(state.tau, gamma, state.c_lip, record["Cmax"])
        # `apply_ft_transaction` uses `min(base, cap)`, so the cap bound exactly
        # when the stored coefficient *is* the cap value.
        binding = "io" if state.lambda_io == cap else None
        # `cmax` here is the legacy `Cmax` from `apply_ft_transaction`'s record
        # (`1 + kappa_ft*(ecc_max_max-1)_+`), not the new pre-clip
        # lambda-weighted-mean form `_compute` uses for the other two
        # policies -- the legacy path owns no coefficient maths of its own,
        # so `_row` is handed the retired value verbatim.
        row = self._row(iteration, overflow, tau, gamma, record["Cmax"], cap,
                        binding, record["cancellation_ratio"])
        self._pending_row = row
        return NormTransaction(lambdas=dict(lambdas), cmax=record["Cmax"], cap=cap,
                               cap_binding=binding,
                               cancellation_ratio=record["cancellation_ratio"],
                               obj_version=self.obj_version, row=row,
                               legacy_record=record)

    @staticmethod
    def oneshot_lambda(strength, wl_norm, grad_norm, floor=1e-12):
        """One-shot ratio normalisation for a term that is calibrated once per
        rebuild rather than every probe. The expression is written exactly as
        the retired `routing_gp_controller._calibrate` wrote it, so the adapter
        is bit-for-bit identical."""
        return strength * wl_norm / grad_norm if grad_norm > floor else 0.0
