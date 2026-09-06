"""S4a-star surrogate with bounded four-pass IO/FT forward and backward.

S4a is sum(q[e,k] * max(D[home[e],k]-1,0)), not true pure-feedthrough.
The original zero-FT arithmetic is retained through direct IoTerm dispatch.
"""
import torch
from .io_term import IoTerm, IoTermRef, margin_penalty
from .soft_assign import softmax_stats, region_sdf_l1, chunk_p_ell, d_star_from_m, _chunks

class _FtFn(torch.autograd.Function):
    """Chunked-k forward/backward (design v2 sec 2.5's four-pass structure).

    Resident state saved for backward: m (N,), t (N,), argmax (N,), lam (E,),
    plus the static topology held on `meta` (an IoTerm instance -- rects,
    rect2region, node_idx, net_idx, w, k_chunk, ...). NOTHING of shape
    (N,K)/(P,K) is ever allocated; every per-chunk tensor is (N,c) or (P',c)
    with c == meta.k_chunk << K.
    """

    @staticmethod
    def forward(ctx, x, y, meta, D, home, tau, lambda_io, io_scale, ft_scale, lambda_margin, margin_m, margin_tau):
        rects = meta.rects.to(dtype=x.dtype)

        # FWD-1 (reduce): m, t, argmax over ALL K regions -- already chunk-aware.
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau, chunk=meta.k_chunk)

        # FWD-2 (accumulate): per chunk, p/ell -> S_chunk (fp64 index_add_) -> lam.
        lam = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
        ft = lam.new_zeros(())
        for lo, hi in _chunks(meta.K, meta.k_chunk):
            sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)          # (N, c)
            _, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)                      # (N, c)
            S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                              device=x.device).index_add_(0, meta.net_idx,
                                                           ell_c[meta.node_idx].double())
            # production habit: -expm1(S) instead of (1 - exp(S)) -- S itself
            # stays the fp64 accumulator from index_add_ above (contract unchanged).
            q = -torch.expm1(S_c)
            lam = lam + q.sum(dim=1)
            a = (D[home, lo:hi].double() - 1.0).clamp_min(0)
            ft = ft + (a * q).sum()

        L_io = (meta.w * (lam - 1.0).clamp(min=0)).sum()
        if lambda_margin != 0.0:                          # production habit: short-circuit
            d_star = d_star_from_m(m, tau)
            L_margin = margin_penalty(d_star, margin_m, margin_tau)
        else:
            L_margin = torch.zeros((), dtype=L_io.dtype, device=x.device)

        meta.last_peak_chunk_elems = meta.k_chunk * max(meta.num_physical, meta._n_pins_dedup)
        ctx.save_for_backward(x, y, m, t, am, lam, home, meta.w.detach().clone())
        ctx.D, ctx.io_scale, ctx.ft_scale = D, io_scale, ft_scale
        ctx.meta = meta
        ctx.tau, ctx.lambda_io, ctx.lambda_margin = tau, lambda_io, lambda_margin
        ctx.margin_m, ctx.margin_tau = margin_m, margin_tau
        return lambda_io * (io_scale * L_io + ft_scale * ft) + lambda_margin * L_margin

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gout):
        x, y, m, t, am, lam, home, weights = ctx.saved_tensors
        meta = ctx.meta
        tau, lambda_io, lambda_margin = ctx.tau, ctx.lambda_io, ctx.lambda_margin
        margin_m, margin_tau = ctx.margin_m, ctx.margin_tau
        rects = meta.rects.to(dtype=x.dtype)
        n = x.shape[0]

        with torch.no_grad():
            io_coeff = ctx.io_scale * weights * (lam > 1.0).double()

            # BWD-1 (reduce): A_i = sum_k c_{i,k} * p_{i,k}  (leave-one-out coupling term).
            A = torch.zeros(n, dtype=torch.float64, device=x.device)
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx, ell_pins)
                a = (ctx.D[home, lo:hi].double() - 1.0).clamp_min(0)
                coeff = io_coeff.unsqueeze(1) + ctx.ft_scale * a
                c_pins = coeff[meta.net_idx] * torch.exp(S_c[meta.net_idx] - ell_pins)   # (P', c)
                c_nodes = torch.zeros((n, hi - lo), dtype=torch.float64,
                                      device=x.device).index_add_(0, meta.node_idx, c_pins)
                A = A + (c_nodes * p_c.double()).sum(dim=1)

            if lambda_margin != 0.0:
                d_star = d_star_from_m(m, tau)
                margin_w = torch.sigmoid((d_star + margin_m) / margin_tau) / margin_tau  # (N,)
            else:
                margin_w = None

            # BWD-2 (scatter): recompute again; dL/dz = p*(c - A); dL/dx += dL/dz*(-1/tau)*d(sdf)/dx.
            # d(sdf_chunk)/dx is obtained via a *local* autograd.grad on the same
            # (differentiable) region_sdf_l1 the reference uses -- (N,c)/(P,c) sized,
            # never (N,K) -- so ties/boundary subgradients match IoTermRef exactly.
            gx = torch.zeros(n, dtype=torch.float64, device=x.device)
            gy = torch.zeros(n, dtype=torch.float64, device=x.device)
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    yg = y.detach().requires_grad_(True)
                    sdf_c = region_sdf_l1(xg, yg, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c.detach(), m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx, ell_pins)
                a = (ctx.D[home, lo:hi].double() - 1.0).clamp_min(0)
                coeff = io_coeff.unsqueeze(1) + ctx.ft_scale * a
                c_pins = coeff[meta.net_idx] * torch.exp(S_c[meta.net_idx] - ell_pins)
                c_nodes = torch.zeros((n, hi - lo), dtype=torch.float64,
                                      device=x.device).index_add_(0, meta.node_idx, c_pins)
                dLdz = p_c.double() * (c_nodes - A.unsqueeze(1))                # (N, c) fp64

                w_out = lambda_io * dLdz * (-1.0 / tau)
                if margin_w is not None:
                    is_am = (am.unsqueeze(1) == torch.arange(lo, hi, device=x.device).unsqueeze(0))
                    w_out = w_out + lambda_margin * margin_w.unsqueeze(1) * is_am.double()

                gxc, gyc = torch.autograd.grad(sdf_c, [xg, yg], grad_outputs=w_out.to(sdf_c.dtype))
                gx = gx + gxc.double()
                gy = gy + gyc.double()

            gx = gx * gout.double()
            gy = gy * gout.double()
            gx[meta.num_movable:] = 0.0        # I4: fixed + filler never move
            gy[meta.num_movable:] = 0.0
        return (gx.to(x.dtype), gy.to(y.dtype)) + (None,) * 11



class FtTerm(torch.nn.Module):
    def __init__(self, io_term, D):
        super().__init__()
        if not isinstance(io_term, IoTerm):
            raise TypeError("production FtTerm requires production IoTerm")
        self.io_term = io_term
        self.register_buffer("D", torch.as_tensor(D, dtype=torch.float64,
                                                  device=io_term.rects.device).clone())
        if self.D.shape != (io_term.K, io_term.K) or not bool(torch.isfinite(self.D).all()):
            raise ValueError("D must be a finite K by K distance table")
        self.register_buffer("home", None)

    def set_home(self, home):
        h = torch.as_tensor(home, dtype=torch.int64, device=self.D.device)
        if h.shape != (self.io_term.n_active,) or bool(((h < 0) | (h >= self.io_term.K)).any()):
            raise ValueError("home must contain one valid region per active net")
        self.home = h.detach().clone()

    def _evaluate_parts(self, pos, tau, lambda_io, io_scale, ft_scale, lambda_margin, margin_m, margin_tau):
        if self.home is None:
            raise RuntimeError("set_home is required for nonzero FT")
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        return _FtFn.apply(x, y, meta, self.D, self.home, tau, lambda_io,
                           io_scale, ft_scale, lambda_margin, margin_m, margin_tau)

    def forward(self, pos, tau, lambda_io, kappa_ft, lambda_margin=0., margin_m=0., margin_tau=1.):
        if kappa_ft == 0:
            return self.io_term(pos, tau, lambda_io, lambda_margin, margin_m, margin_tau)
        return self._evaluate_parts(pos, tau, lambda_io, 1., kappa_ft, lambda_margin, margin_m, margin_tau)

    def ft_only(self, pos, tau):
        return self._evaluate_parts(pos, tau, 1., 0., 1., 0., 0., 1.)


class FtTermRef(torch.nn.Module):
    """Independent dense autograd oracle. For small numerical tests only."""
    def __init__(self, io_term, D):
        super().__init__()
        self.io_term = io_term
        self.register_buffer("D", torch.as_tensor(D, dtype=torch.float64,
                                                  device=io_term.rects.device).clone())
        self.register_buffer("home", None)

    def set_home(self, home):
        h = torch.as_tensor(home, dtype=torch.int64, device=self.D.device)
        if h.shape != (self.io_term.n_active,) or bool(((h < 0) | (h >= self.io_term.K)).any()):
            raise ValueError("home must contain one valid region per active net")
        self.home = h.detach().clone()

    def _values(self, pos, tau):
        if self.home is None:
            raise RuntimeError("set_home is required for nonzero FT")
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        x = torch.cat((x[:meta.num_movable], x[meta.num_movable:].detach()))
        y = torch.cat((y[:meta.num_movable], y[meta.num_movable:].detach()))
        rects = meta.rects.to(dtype=x.dtype)
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau)
        sdf = region_sdf_l1(x, y, rects, meta.rect2region, 0, meta.K)
        _, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
        S = torch.zeros((meta.n_active, meta.K), dtype=torch.float64, device=pos.device)
        S = S.index_add(0, meta.net_idx, ell[meta.node_idx].double())
        q = -torch.expm1(S)
        L_io = (meta.w * (q.sum(1) - 1).clamp_min(0)).sum()
        coeff = (self.D[self.home] - 1.).clamp_min(0)
        return L_io, (coeff * q).sum(), d_star_from_m(m, tau)

    def forward(self, pos, tau, lambda_io, kappa_ft, lambda_margin=0., margin_m=0., margin_tau=1.):
        if kappa_ft == 0:
            return self.io_term(pos, tau, lambda_io, lambda_margin, margin_m, margin_tau)
        io, ft, ds = self._values(pos, tau)
        margin = margin_penalty(ds, margin_m, margin_tau) if lambda_margin else io.new_zeros(())
        return lambda_io * (io + kappa_ft * ft) + lambda_margin * margin

    def ft_only(self, pos, tau):
        return self._values(pos, tau)[1]
