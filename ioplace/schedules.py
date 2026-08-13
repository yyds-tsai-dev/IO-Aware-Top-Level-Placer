"""tau / rho / lambda_io schedules (design v2 sec 4.2, 5.2, 6.4.2; v3.1 sec
5.2/5.5 FT force-share additions)."""
import math
from dataclasses import dataclass, field

EPS = 1e-30


def tau_rel_from_overflow(of, tau_hi=0.30, tau_lo=0.03, of_on=0.90, of_end=0.07):
    if of_on <= of_end:
        raise ValueError("of_on must exceed of_end")
    frac = (of - of_end) / (of_on - of_end)
    frac = min(max(frac, 0.0), 1.0)
    return tau_lo * (tau_hi / tau_lo) ** frac


def rho_from_overflow(of, rho_max, of_on=0.90, of_full=0.20):
    if of_on <= of_full:
        raise ValueError("of_on must exceed of_full")
    frac = (of_on - of) / (of_on - of_full)
    return rho_max * min(max(frac, 0.0), 1.0)


def activation_ramp(iteration, it_activate, n_ramp=20):
    if n_ramp <= 0:
        return 1.0 if iteration >= it_activate else 0.0
    return min(max((iteration - it_activate) / float(n_ramp), 0.0), 1.0)


def ft_activation_ramp(tau_rel, tau_start=0.12, tau_full=0.05, mode="window"):
    """FT force-share ramp as a function of tau_rel (design v3.1 sec 5.2).

    Two-point log interpolation: 0 at tau_rel >= tau_start, 1 at
    tau_rel <= tau_full, log-linear in between:
        clip( ln(tau_start/tau_rel) / ln(tau_start/tau_full), 0, 1 )
    `mode="constant"` returns 1.0 unconditionally -- the `--ft-ramp-mode
    constant` arm (sec 5.2 last row / Codex F5's fix for the `of_ft_full ==
    of_on` division-by-zero in the old always-on scheme).

    Sample-and-hold semantics (i.e. whether this is evaluated every
    iteration or only every N=50-iteration callback) are the caller's
    responsibility -- this is a pure function of whatever tau_rel it is
    handed.

    NOTE (naming): this is deliberately NOT named bare `activation_ramp`.
    The task that requested this function specified that literal name, but
    `activation_ramp(iteration, it_activate, n_ramp)` above is a pre-existing,
    actively-used, differently-signatured function -- design v3.1 sec 5.2's
    `lambda_io` row cites it unchanged at `schedules.py:22-25` for lambda_io's
    own iteration-count ramp. Reusing the name would have silently shadowed
    it and broken `update_continuous`/`test_activation_ramp_bounds`. Flagged
    in the task report; confirmed by the scheduler to use this name instead.
    """
    if mode == "constant":
        return 1.0
    if mode != "window":
        raise ValueError(f"unknown ft_ramp_mode: {mode!r}")
    if tau_start <= tau_full:
        raise ValueError("tau_start must exceed tau_full")
    if tau_rel <= 0.0:
        return 1.0
    frac = math.log(tau_start / tau_rel) / math.log(tau_start / tau_full)
    return min(max(frac, 0.0), 1.0)


def derive_kappa_ft(f_ft, g_io_l1, g_ft_l1, kappa_max=100.0, eps_rel=1e-3):
    """FT force-share kappa (design v3.1 sec 5.2/5.5 step 3).

    Zero-gradient branch: g_ft_l1 <= eps_rel * g_io_l1 => kappa_ft = 0 (never
    a near-zero division, never a manufactured penalty when FT has no local
    gradient signal). Otherwise clip(f_ft * g_io_l1 / g_ft_l1, 0, kappa_max).

    `f_ft` here is the *fully ramped* force-share fraction (f_ft_max times
    the sec 5.2 tau_rel ramp, i.e. f_ft_max * ft_activation_ramp(tau_rel))
    -- see the docstring note in this module's apply_ft_transaction for why
    this differs from the task's shorthand "f_ft * ramp * ..." summary
    formula.
    """
    if g_ft_l1 <= eps_rel * g_io_l1:
        return 0.0
    return min(max(f_ft * g_io_l1 / g_ft_l1, 0.0), kappa_max)


def derive_cmax(kappa_ft, ecc_max_max):
    """Cmax upper bound for the merged (e,k) coefficient (design v3.1 sec
    5.2/5.5 step 5): 1 + kappa_ft * (max_h ecc_max[h] - 1)_+.

    kappa_ft == 0 => Cmax == 1 identically (no manufactured penalty on
    pure-IO steps), by construction of the formula -- no special case needed.
    """
    return 1.0 + kappa_ft * max(ecc_max_max - 1.0, 0.0)


def lipschitz_cap(tau, gamma, c_lip=1.0, cmax=1.0):
    """Step-size upper bound: c_lip * tau^2 / (gamma * max(1, cmax)).

    `cmax` defaults to 1.0 (Cmax's own rest value), so existing call sites
    that pass only (tau, gamma[, c_lip]) are bit-for-bit unchanged (design
    v3.1 sec 5.2/T3 item 4: the Cmax Lipschitz guard is an additive
    parameterization of this pre-existing cap, not a new formula).
    """
    if gamma <= 0.0:
        return float("inf")
    return c_lip * tau * tau / (gamma * max(1.0, cmax))


@dataclass
class ScheduleState:
    rho_max: float
    tau_hi: float = 0.30
    tau_lo: float = 0.03
    of_on: float = 0.90
    of_end: float = 0.07
    of_full: float = 0.20
    n_ramp: int = 20
    c_lip: float = 1.0
    alpha_io: float = 0.0
    rho_margin: float = 0.0
    margin_m: float = 0.0
    margin_tau: float = 1.0
    of_margin: float = 0.15
    # -- v3.1 FT force-share config (sec 5.2/5.5, T3) --
    f_ft_max: float = 0.0
    tau_start: float = 0.12
    tau_full: float = 0.05
    ft_ramp_mode: str = "window"
    kappa_max: float = 100.0
    eps_rel: float = 1e-3
    ema: float = 0.5
    active: bool = False
    it_activate: int = None
    tau: float = 0.0
    rho: float = 0.0
    lambda_io: float = 0.0
    lambda_margin: float = 0.0
    ratio_ema: float = None
    # -- v3.1 FT force-share derived state (sec 5.2/5.5, T3) --
    kappa_ft: float = 0.0
    Cmax: float = 1.0
    ratio_inst: float = None
    cancellation_ratio: float = None
    home_version: int = 0
    obj_version: int = 0
    refreshed_version: int = 0
    _margin_on: bool = field(default=False, repr=False)

    def update_continuous(self, iteration, overflow, L_R, gamma):
        discrete = False
        if not self.active and overflow <= self.of_on:
            self.active = True
            self.it_activate = iteration
            self.obj_version += 1
            discrete = True
        self.tau = tau_rel_from_overflow(overflow, self.tau_hi, self.tau_lo,
                                         self.of_on, self.of_end) * L_R
        if not self.active:
            self.rho, self.lambda_io, self.lambda_margin = 0.0, 0.0, 0.0
            return discrete
        self.rho = rho_from_overflow(overflow, self.rho_max, self.of_on, self.of_full)
        ramp = activation_ramp(iteration, self.it_activate, self.n_ramp)
        base = 0.0 if self.ratio_ema is None else self.rho * ramp * self.ratio_ema
        self.lambda_io = min(base, lipschitz_cap(self.tau, gamma, self.c_lip, self.Cmax))
        want_margin = (self.rho_margin > 0.0) and (overflow <= self.of_margin)
        if want_margin != self._margin_on:
            self._margin_on = want_margin
            self.obj_version += 1
            discrete = True
        self.lambda_margin = (self.rho_margin * ramp * (self.ratio_ema or 0.0)
                              if self._margin_on else 0.0)
        return discrete

    def update_ratio(self, g_wl_l1, g_io_l1):
        raw = g_wl_l1 / max(g_io_l1, EPS)
        self.ratio_ema = raw if self.ratio_ema is None else (
            self.ema * self.ratio_ema + (1.0 - self.ema) * raw)
        self.obj_version += 1

    def apply_ft_transaction(self, iteration, tau_rel, g_wl_l1, g_io_l1, g_ft_l1,
                              merged_l1, ecc_max_max, gamma):
        """Seven-step atomic FT callback transaction (design v3.1 sec 5.5),
        the `--callback-order atomic` (default) path run on the N=50
        evaluator-gated event.

        Steps 1-2 (evaluator call / home_e refresh, measuring g_wl/g_io/g_ft)
        happen in the caller before this is invoked -- schedules.py owns no
        evaluator or gradient-vector state. This method performs steps 3-7:
        derive kappa_ft, update the merged-norm ratio_ema, derive Cmax,
        recompute lambda_io from the freshly-updated values, and bump
        obj_version exactly once (never on an intermediate step). Requires
        `update_continuous(iteration, ...)` to have already run this same
        iteration so self.tau/self.rho/self.active/self.it_activate are
        current. Caller still owns the actual Nesterov secant refresh side
        effect and must call `mark_refreshed()` once that is done (step 7's
        second half).

        Legacy-order regression lock: the pre-existing `update_continuous` +
        `update_ratio` pair, called in the pre-existing order, remains
        untouched and is the `--callback-order legacy` path (sec 5.5 note 2)
        -- it is not affected by this method's existence.

        Returns f_ft (the force-share fraction sampled at tau_rel), for
        trajectory logging.

        NOTE (formula source): the task summary describing this step wrote
        `kappa = clip(f_ft * ramp * g_io_l1 / max(g_ft_l1, eps), 0, kappa_max)`
        -- an extra standalone `ramp` factor alongside `f_ft`. Design v3.1
        sec 5.2/5.5 body defines `f_ft` itself as *already* ramped
        (`f_ft = f_ft_max * ft_activation_ramp(tau_rel)`, sec 5.2's `f_ft`
        row) and `kappa_ft = clip(f_ft(tau_rel) * g_io/g_ft, 0, kappa_max)`
        (sec 5.5 step 3) -- a single multiplication, not a second one. The
        summary's extra `* ramp` would double-apply the ramp. Implemented
        per the v3.1 body (single multiplication); flagged as a
        summary/body discrepancy in the task report as instructed. The
        body's zero-gradient guard (`g_ft <= eps_rel * g_io => kappa = 0`)
        is used as-is rather than the summary's `max(g_ft_l1, eps)` floor --
        the guard already makes the division safe, so no separate epsilon
        floor on the denominator is needed (see derive_kappa_ft).
        """
        f_ft = self.f_ft_max * ft_activation_ramp(tau_rel, self.tau_start, self.tau_full,
                                                  self.ft_ramp_mode)
        self.kappa_ft = derive_kappa_ft(f_ft, g_io_l1, g_ft_l1, self.kappa_max,
                                        self.eps_rel)
        raw = g_wl_l1 / max(merged_l1, EPS)
        self.ratio_inst = raw
        self.ratio_ema = raw if self.ratio_ema is None else (
            self.ema * self.ratio_ema + (1.0 - self.ema) * raw)
        self.Cmax = derive_cmax(self.kappa_ft, ecc_max_max)
        self.cancellation_ratio = merged_l1 / max(
            g_io_l1 + self.kappa_ft * g_ft_l1, EPS)
        if self.active:
            ramp_iter = activation_ramp(iteration, self.it_activate, self.n_ramp)
            base = self.rho * ramp_iter * self.ratio_ema
            self.lambda_io = min(base, lipschitz_cap(self.tau, gamma, self.c_lip,
                                                      self.Cmax))
        self.obj_version += 1
        self.home_version = self.obj_version
        return f_ft

    def mark_refreshed(self):
        self.refreshed_version = self.obj_version

    def needs_refresh(self):
        return self.obj_version != self.refreshed_version
