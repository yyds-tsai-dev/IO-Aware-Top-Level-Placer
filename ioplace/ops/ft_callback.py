"""Atomic publication of IO/FT coefficients using independent gradients."""
import torch

from ioplace.schedules import derive_kappa_ft, ft_activation_ramp


def independent_gradient(fn, pos, num_movable, num_nodes):
    leaf = pos.detach().clone().requires_grad_(True)
    value = fn(leaf)
    gradient, = torch.autograd.grad(value, leaf)
    gradient[num_movable:num_nodes] = 0
    gradient[num_nodes + num_movable:] = 0
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("nonfinite independent objective gradient")
    return gradient


def publish_atomic(state, io_term, ft_term, wirelength_op, pos, iteration,
                   tau_rel, ecc_max, gamma):
    """Weights/home must already be refreshed; caller refreshes Nesterov next."""
    args = (pos, io_term.num_movable, io_term.num_nodes)
    wl = independent_gradient(wirelength_op, *args)
    io = independent_gradient(lambda p: io_term(p, state.tau, 1.), *args)
    ft = (independent_gradient(lambda p: ft_term.ft_only(p, state.tau), *args)
          if ft_term is not None else None)
    wl_l1, io_l1 = float(wl.abs().sum()), float(io.abs().sum())
    ft_l1 = float(ft.abs().sum()) if ft is not None else 0.
    f = state.f_ft_max * ft_activation_ramp(tau_rel, state.tau_start,
                                           state.tau_full, state.ft_ramp_mode)
    kappa = derive_kappa_ft(f, io_l1, ft_l1, state.kappa_max, state.eps_rel)
    merged_l1 = float((io + kappa * ft).abs().sum()) if kappa else io_l1
    old_version = state.obj_version
    state.apply_ft_transaction(iteration, tau_rel, wl_l1, io_l1, ft_l1,
                               merged_l1, ecc_max, gamma)
    assert state.obj_version == old_version + 1
    assert state.kappa_ft == kappa
    applied = state.lambda_io * kappa * ft_l1
    return dict(
        grad_l1_wl=wl_l1, grad_l1_io=io_l1, grad_l1_ft=ft_l1,
        grad_l1_merged=merged_l1, f_ft=f, kappa_ft=kappa,
        kappa_clamped=bool(ft_l1 > state.eps_rel * io_l1
                           and f * io_l1 / ft_l1 > state.kappa_max),
        Cmax=state.Cmax, ratio_inst=state.ratio_inst, ratio_ema=state.ratio_ema,
        f_effective=kappa * ft_l1 / io_l1 if io_l1 else None,
        applied_ft_force_l1=applied,
        cancellation_ratio=state.cancellation_ratio if applied > 0 else None,
        lambda_io=state.lambda_io, obj_version=state.obj_version,
        home_version=state.home_version,
    )
