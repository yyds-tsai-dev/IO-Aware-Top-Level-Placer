"""tau / rho / lambda_io schedules (design v2 sec 4.2, 5.2, 6.4.2)."""
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


def lipschitz_cap(tau, gamma, c_lip=1.0):
    if gamma <= 0.0:
        return float("inf")
    return c_lip * tau * tau / gamma


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
    ema: float = 0.5
    active: bool = False
    it_activate: int = None
    tau: float = 0.0
    rho: float = 0.0
    lambda_io: float = 0.0
    lambda_margin: float = 0.0
    ratio_ema: float = None
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
        self.lambda_io = min(base, lipschitz_cap(self.tau, gamma, self.c_lip))
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

    def mark_refreshed(self):
        self.refreshed_version = self.obj_version

    def needs_refresh(self):
        return self.obj_version != self.refreshed_version
