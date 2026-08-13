"""S2 product-form IO surrogate -- reference (non-chunked) implementation.

*** This module's IoTermRef is the SEMANTIC reference only. It materializes
(N,K)/(P,K) tensors and MUST NOT be used by any driver (design v2 sec 2.5 /
Global Constraints). The production path is IoTerm in this same file, added by
Task 2b, which is bound to IoTermRef by an equivalence test. ***
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from ioplace.ops.soft_assign import (rect_table, region_sdf_l1, softmax_stats,
                                     chunk_p_ell, d_star_from_m, _chunks)

DEG_BUCKET_EDGES = (2, 3, 4, 8, 16, 32, 64, 100)
DEG_BUCKET_LABELS = ("2", "3", "4-7", "8-15", "16-31", "32-63", "64-99")


@dataclass
class NetCsr:
    flat_net2node: np.ndarray
    net2node_start: np.ndarray
    net_ids: np.ndarray
    degrees: np.ndarray
    pin_degrees: np.ndarray
    deg_bucket: np.ndarray
    n_nets_total: int


def net_mask_io(nl, ignore_net_degree):
    deg = nl.net_degrees
    return (deg >= 2) & (deg < ignore_net_degree)


def build_net_node_csr(nl, ignore_net_degree):
    mask = net_mask_io(nl, ignore_net_degree)
    nodes_of_pin = nl.pin2node[nl.flat_net2pin].astype(np.int64)
    net_of_pin = nl.pin2net[nl.flat_net2pin].astype(np.int64)
    pin_keep = mask[net_of_pin]
    key = net_of_pin[pin_keep] * np.int64(nl.num_physical) + nodes_of_pin[pin_keep]
    key = np.unique(key)                       # sorted -> grouped by net, dedup'd
    unet = key // np.int64(nl.num_physical)
    unode = key % np.int64(nl.num_physical)
    ids_all, counts = np.unique(unet, return_counts=True)
    ok = counts >= 2                           # nets collapsing to 1 node contribute nothing
    keep = np.repeat(ok, counts); flat = unode[keep]; degs = counts[ok]; ids = ids_all[ok]
    start = np.concatenate([[0], np.cumsum(degs)]).astype(np.int64)
    pin_deg = nl.net_degrees[ids].astype(np.int64)
    bucket = np.clip(np.searchsorted(np.array(DEG_BUCKET_EDGES[1:]), pin_deg, side="right"),
                     0, len(DEG_BUCKET_LABELS) - 1)
    return NetCsr(flat, start, ids, degs, pin_deg, bucket, int(nl.num_nets))


def margin_penalty(d_star, m, tau_m):
    """design v2 sec 9.2: softplus((d*+m)/tau_m); d* < 0 inside a region."""
    return F.softplus((d_star + m) / tau_m).sum()


class IoTermRef(torch.nn.Module):
    """Invariants (design v2 sec 3.1 / 3.2.2 / 3.2.3):
      I1  L_IO = sum_e w_e * max(sum_k q_{e,k} - 1, 0),  q = 1 - exp(S)
      I2  ell is produced by soft_assign.chunk_p_ell (stable form, sec 2.2) --
          never torch.log1p(-softmax(...)) without a guard
      I3  S is accumulated in float64 (index_add_) regardless of pos dtype
      I4  gradient of every index >= num_movable (terminals AND fillers) is 0
      I5  the value returned is already weighted: lambda_io*L_IO + lambda_margin*L_margin
    """

    def __init__(self, csr, rects, rect2region, K, num_movable, num_physical,
                 num_nodes, device="cuda", w_mode="unit"):
        super().__init__()
        if w_mode not in ("unit", "inv_deg"):
            raise ValueError(f"w_mode must be 'unit' or 'inv_deg', got {w_mode!r}")
        self.K = int(K)
        self.num_movable = int(num_movable)
        self.num_physical = int(num_physical)
        self.num_nodes = int(num_nodes)
        self.w_mode = w_mode
        self.n_active = int(len(csr.net_ids))

        degrees_i64 = torch.as_tensor(csr.degrees, dtype=torch.int64, device=device)
        if w_mode == "unit":
            w = torch.ones(self.n_active, dtype=torch.float64, device=device)
        else:  # inv_deg -- w_e = 1/(deg'-1); NetCsr only keeps deg' >= 2, so no div-by-0
            w = 1.0 / (degrees_i64.double() - 1.0)

        self.register_buffer("node_idx", torch.as_tensor(csr.flat_net2node, dtype=torch.int64,
                                                          device=device))
        self.register_buffer("net_idx", torch.repeat_interleave(
            torch.arange(self.n_active, dtype=torch.int64, device=device), degrees_i64))
        self.register_buffer("w", w)
        self.register_buffer("rects", torch.as_tensor(rects, dtype=torch.float64, device=device))
        self.register_buffer("rect2region", torch.as_tensor(rect2region, dtype=torch.int64,
                                                             device=device))
        # not part of the documented buffer set, but needed internally by
        # diagnostics() to attribute gradient share back to each net's degree bucket
        self.register_buffer("deg_bucket", torch.as_tensor(csr.deg_bucket, dtype=torch.int64,
                                                            device=device))

    def _split_xy(self, pos):
        """I4: slice pos into (x,y) over the physical nodes, detaching the
        terminal tail so it still participates in the forward value (its
        region membership is real) but never receives gradient. Filler
        positions (index >= num_physical) are never indexed at all."""
        x = pos[:self.num_physical]
        y = pos[self.num_nodes:self.num_nodes + self.num_physical]
        x = torch.cat([x[:self.num_movable], x[self.num_movable:].detach()])
        y = torch.cat([y[:self.num_movable], y[self.num_movable:].detach()])
        return x, y

    def _forward_io(self, x, y, tau):
        """-> (L_io scalar, lam (n_active,), d_star (N,))"""
        m, t, am = softmax_stats(x, y, self.rects, self.rect2region, self.K, tau)
        sdf = region_sdf_l1(x, y, self.rects, self.rect2region, 0, self.K)
        p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
        S = torch.zeros((self.n_active, self.K), dtype=torch.float64,
                        device=x.device).index_add_(0, self.net_idx, ell[self.node_idx].double())
        lam = (1.0 - torch.exp(S)).sum(dim=1)
        return (self.w * (lam - 1.0).clamp(min=0)).sum(), lam, d_star_from_m(m, tau)

    def forward(self, pos, tau, lambda_io, lambda_margin=0.0, margin_m=0.0, margin_tau=1.0):
        x, y = self._split_xy(pos)
        L_io, lam, d_star = self._forward_io(x, y, tau)
        L_margin = margin_penalty(d_star, margin_m, margin_tau)
        return lambda_io * L_io + lambda_margin * L_margin

    def io_grad_l1(self, pos, tau) -> float:
        """Independent fwd+bwd of unweighted (lambda_io=1, no margin) L_IO;
        returns ||grad L_IO||_1 (design v2 sec 5.2's free-riding diagnostic)."""
        p = pos.detach().clone().requires_grad_(True)
        L = self.forward(p, tau, lambda_io=1.0, lambda_margin=0.0)
        L.backward()
        return float(p.grad.abs().sum())

    def diagnostics(self, pos, tau) -> dict:
        p = pos.detach().clone().requires_grad_(True)
        x, y = self._split_xy(p)

        with torch.no_grad():
            _, t_mv, _ = softmax_stats(x[:self.num_movable], y[:self.num_movable],
                                       self.rects, self.rect2region, self.K, tau)
            p_max = 1.0 / (1.0 + t_mv)          # e[argmax]=1 -> p_max = 1/(1+t)
            frac_soft = float((p_max < (1.0 - 1e-3)).double().mean())

        L_io, lam, _ = self._forward_io(x, y, tau)
        contrib = self.w * (lam - 1.0).clamp(min=0)     # (n_active,), sums to L_io

        n_b = len(DEG_BUCKET_LABELS)
        grad_share = np.zeros(n_b, dtype=np.float64)
        n_nonempty = 0
        for b in range(n_b):
            bucket_mask = (self.deg_bucket == b).double()
            if float(bucket_mask.sum()) == 0.0:
                continue
            n_nonempty += 1
            (g,) = torch.autograd.grad(contrib, p, grad_outputs=bucket_mask, retain_graph=True)
            grad_share[b] = float(g.abs().sum())
        total = grad_share.sum()
        if total > 0.0:
            grad_share = grad_share / total

        return {
            # raw sum_e lambda_e (no -1, no clamp, no w_e) -- distinct from l_io,
            # which is the actual (weighted, clamped) objective term
            "soft_lambda_sum": float(lam.detach().sum()),
            "frac_soft": frac_soft,
            "grad_share": grad_share,
            "l_io": float(L_io.detach()),
            # M4 design draft sec 1.4 B2 (Codex review): the driver's per-
            # callback cost of "diagnostics" is this call's n_nonempty
            # bucket backward passes *plus* the one separate io_grad_l1
            # backward the driver always takes right before it -- general
            # formula, not the fixed "7 buckets + 1 = 8" the draft's first
            # pass assumed (a run whose nets miss some buckets does fewer).
            "n_backward_passes": 1 + n_nonempty,
        }


class _IoFn(torch.autograd.Function):
    """Chunked-k forward/backward (design v2 sec 2.5's four-pass structure).

    Resident state saved for backward: m (N,), t (N,), argmax (N,), lam (E,),
    plus the static topology held on `meta` (an IoTerm instance -- rects,
    rect2region, node_idx, net_idx, w, k_chunk, ...). NOTHING of shape
    (N,K)/(P,K) is ever allocated; every per-chunk tensor is (N,c) or (P',c)
    with c == meta.k_chunk << K.
    """

    @staticmethod
    def forward(ctx, x, y, meta, tau, lambda_io, lambda_margin, margin_m, margin_tau):
        rects = meta.rects.to(dtype=x.dtype)

        # FWD-1 (reduce): m, t, argmax over ALL K regions -- already chunk-aware.
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau, chunk=meta.k_chunk)

        # FWD-2 (accumulate): per chunk, p/ell -> S_chunk (fp64 index_add_) -> lam.
        lam = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
        for lo, hi in _chunks(meta.K, meta.k_chunk):
            sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)          # (N, c)
            _, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)                      # (N, c)
            S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                              device=x.device).index_add_(0, meta.net_idx,
                                                           ell_c[meta.node_idx].double())
            # production habit: -expm1(S) instead of (1 - exp(S)) -- S itself
            # stays the fp64 accumulator from index_add_ above (contract unchanged).
            lam = lam + (-torch.expm1(S_c)).sum(dim=1)

        L_io = (meta.w * (lam - 1.0).clamp(min=0)).sum()
        if lambda_margin != 0.0:                          # production habit: short-circuit
            d_star = d_star_from_m(m, tau)
            L_margin = margin_penalty(d_star, margin_m, margin_tau)
        else:
            L_margin = torch.zeros((), dtype=L_io.dtype, device=x.device)

        meta.last_peak_chunk_elems = meta.k_chunk * max(meta.num_physical, meta._n_pins_dedup)
        ctx.save_for_backward(x, y, m, t, am, lam)
        ctx.meta = meta
        ctx.tau, ctx.lambda_io, ctx.lambda_margin = tau, lambda_io, lambda_margin
        ctx.margin_m, ctx.margin_tau = margin_m, margin_tau
        return lambda_io * L_io + lambda_margin * L_margin

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gout):
        x, y, m, t, am, lam = ctx.saved_tensors
        meta = ctx.meta
        tau, lambda_io, lambda_margin = ctx.tau, ctx.lambda_io, ctx.lambda_margin
        margin_m, margin_tau = ctx.margin_m, ctx.margin_tau
        rects = meta.rects.to(dtype=x.dtype)
        n = x.shape[0]

        with torch.no_grad():
            w_pins = (meta.w * (lam > 1.0).double())[meta.net_idx]        # (P',) fp64

            # BWD-1 (reduce): A_i = sum_k c_{i,k} * p_{i,k}  (leave-one-out coupling term).
            A = torch.zeros(n, dtype=torch.float64, device=x.device)
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx, ell_pins)
                c_pins = w_pins.unsqueeze(1) * torch.exp(S_c[meta.net_idx] - ell_pins)   # (P', c)
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
                c_pins = w_pins.unsqueeze(1) * torch.exp(S_c[meta.net_idx] - ell_pins)
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
        return gx.to(x.dtype), gy.to(y.dtype), None, None, None, None, None, None


class IoTerm(torch.nn.Module):
    """Chunked-k production IO term (design v2 sec 2.5). Same value/gradient
    contract as IoTermRef (invariants I1-I5 there apply verbatim) but routed
    through _IoFn so no forward/backward path ever allocates an (N,K) or
    (P,K) tensor. **This is the only class a driver may use.**

    Production habits (T2b coordinator additions, vs. the IoTermRef port):
      - lambda_margin == 0.0 short-circuits the margin term entirely (forward
        skips margin_penalty; backward skips its d(sdf_argmax)/dx pass).
      - the S accumulator stays fp64 (contract unchanged); (1 - exp(S)) is
        computed as -torch.expm1(S) for accuracy when S is near 0.
      - `rects` is cast to pos's own dtype at each call instead of being
        forced to fp64 like IoTermRef -- an fp32 `pos` keeps every per-chunk
        tensor fp32 (half the memory of an fp64 rects promoting everything).
    """

    def __init__(self, csr, rects, rect2region, K, num_movable, num_physical,
                 num_nodes, device="cuda", w_mode="unit", chunk_budget: int = 8_000_000):
        super().__init__()
        if w_mode not in ("unit", "inv_deg"):
            raise ValueError(f"w_mode must be 'unit' or 'inv_deg', got {w_mode!r}")
        self.K = int(K)
        self.num_movable = int(num_movable)
        self.num_physical = int(num_physical)
        self.num_nodes = int(num_nodes)
        self.w_mode = w_mode
        self.n_active = int(len(csr.net_ids))

        degrees_i64 = torch.as_tensor(csr.degrees, dtype=torch.int64, device=device)
        if w_mode == "unit":
            w = torch.ones(self.n_active, dtype=torch.float64, device=device)
        else:  # inv_deg -- w_e = 1/(deg'-1); NetCsr only keeps deg' >= 2, so no div-by-0
            w = 1.0 / (degrees_i64.double() - 1.0)

        self.register_buffer("node_idx", torch.as_tensor(csr.flat_net2node, dtype=torch.int64,
                                                          device=device))
        self.register_buffer("net_idx", torch.repeat_interleave(
            torch.arange(self.n_active, dtype=torch.int64, device=device), degrees_i64))
        self.register_buffer("w", w)
        self.register_buffer("rects", torch.as_tensor(rects, device=device))
        self.register_buffer("rect2region", torch.as_tensor(rect2region, dtype=torch.int64,
                                                             device=device))
        # not part of the documented buffer set, but needed internally by
        # diagnostics() to attribute gradient share back to each net's degree bucket
        self.register_buffer("deg_bucket", torch.as_tensor(csr.deg_bucket, dtype=torch.int64,
                                                            device=device))

        self._n_pins_dedup = int(self.node_idx.numel())
        denom = max(self.num_physical, self._n_pins_dedup, 1)
        self.k_chunk = max(1, min(self.K, chunk_budget // denom))
        self.last_peak_chunk_elems = self.k_chunk * max(self.num_physical, self._n_pins_dedup)

    def forward(self, pos, tau, lambda_io, lambda_margin=0.0, margin_m=0.0, margin_tau=1.0):
        x = pos[:self.num_physical]
        y = pos[self.num_nodes:self.num_nodes + self.num_physical]
        return _IoFn.apply(x, y, self, tau, lambda_io, lambda_margin, margin_m, margin_tau)

    def io_grad_l1(self, pos, tau) -> float:
        """Independent fwd+bwd of unweighted (lambda_io=1, no margin) L_IO;
        returns ||grad L_IO||_1 (design v2 sec 5.2's free-riding diagnostic)."""
        p = pos.detach().clone().requires_grad_(True)
        L = self.forward(p, tau, lambda_io=1.0, lambda_margin=0.0)
        L.backward()
        return float(p.grad.abs().sum())

    def diagnostics(self, pos, tau) -> dict:
        """Reporting-only (design v2 sec 3.2.5), not on the training hot
        path, but still bound by the chunked memory contract. An earlier
        version built one differentiable graph spanning every K-chunk and
        called autograd.grad on it once per degree bucket with
        retain_graph=True -- that keeps every chunk's intermediates alive
        across all 7 calls, which in aggregate is exactly the (N,K)
        materialization sec 2.5 forbids, and OOMs on bigblue4-scale real
        data (Task 2b review Finding C1). Fixed two ways: (a) the no-grad
        totals (frac_soft/lam/contrib/L_io) run under torch.no_grad() on
        pos.detach(), no graph at all; (b) grad_share reuses _IoFn itself --
        lam does not depend on w (only the final L_io = sum_e
        w_e*clamp(lam_e-1,0) does), so masking w to zero out nets outside
        the current bucket and calling the ordinary (already memory-safe)
        forward()+backward() gives exactly that bucket's gradient, at the
        cost of redoing the full chunked fwd+bwd once per bucket instead of
        sharing one graph -- more compute, still bounded memory."""
        with torch.no_grad():
            x = pos.detach()[:self.num_physical]
            y = pos.detach()[self.num_nodes:self.num_nodes + self.num_physical]
            rects = self.rects.to(dtype=x.dtype)

            _, t_mv, _ = softmax_stats(x[:self.num_movable], y[:self.num_movable],
                                       rects, self.rect2region, self.K, tau, chunk=self.k_chunk)
            p_max = 1.0 / (1.0 + t_mv)          # e[argmax]=1 -> p_max = 1/(1+t)
            frac_soft = float((p_max < (1.0 - 1e-3)).double().mean())

            m, t, am = softmax_stats(x, y, rects, self.rect2region, self.K, tau, chunk=self.k_chunk)
            lam = torch.zeros(self.n_active, dtype=torch.float64, device=x.device)
            for lo, hi in _chunks(self.K, self.k_chunk):
                sdf_c = region_sdf_l1(x, y, rects, self.rect2region, lo, hi)
                _, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
                S_c = torch.zeros((self.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, self.net_idx,
                                                               ell_c[self.node_idx].double())
                lam = lam + (-torch.expm1(S_c)).sum(dim=1)
            contrib = self.w * (lam - 1.0).clamp(min=0)     # (n_active,), sums to L_io
            L_io = contrib.sum()

        n_b = len(DEG_BUCKET_LABELS)
        grad_share = np.zeros(n_b, dtype=np.float64)
        w_full = self.w
        n_nonempty = 0
        try:
            for b in range(n_b):
                mask = (self.deg_bucket == b)
                if not bool(mask.any()):
                    continue
                n_nonempty += 1
                self.w = w_full * mask.double()
                pb = pos.detach().clone().requires_grad_(True)
                self.forward(pb, tau, lambda_io=1.0, lambda_margin=0.0).backward()
                grad_share[b] = float(pb.grad.abs().sum())
        finally:
            self.w = w_full
        total = grad_share.sum()
        if total > 0.0:
            grad_share = grad_share / total

        return {
            "soft_lambda_sum": float(lam.sum()),
            "frac_soft": frac_soft,
            "grad_share": grad_share,
            "l_io": float(L_io),
            # see IoTermRef.diagnostics's matching comment: general formula,
            # not a fixed 8 (M4 design draft sec 1.4 B2, Codex review).
            "n_backward_passes": 1 + n_nonempty,
        }
