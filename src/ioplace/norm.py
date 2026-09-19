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
