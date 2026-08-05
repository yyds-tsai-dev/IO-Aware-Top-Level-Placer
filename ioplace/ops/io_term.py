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
                                     chunk_p_ell, d_star_from_m)

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
    keep = mask[net_of_pin]
    key = net_of_pin[keep] * np.int64(nl.num_physical) + nodes_of_pin[keep]
    key = np.unique(key)                       # sorted -> grouped by net, dedup'd
    unet = key // np.int64(nl.num_physical)
    unode = key % np.int64(nl.num_physical)
    net_ids, counts = np.unique(unet, return_counts=True)
    ok = counts >= 2                           # nets collapsing to 1 node contribute nothing
    starts = np.concatenate([[0], np.cumsum(counts)])
    seg = [(net_ids[i], starts[i], starts[i + 1]) for i in range(len(net_ids)) if ok[i]]
    flat = np.concatenate([unode[a:b] for _, a, b in seg]) if seg else np.zeros(0, np.int64)
    degs = np.array([b - a for _, a, b in seg], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(degs)]).astype(np.int64)
    ids = np.array([n for n, _, _ in seg], dtype=np.int64)
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
        for b in range(n_b):
            bucket_mask = (self.deg_bucket == b).double()
            if float(bucket_mask.sum()) == 0.0:
                continue
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
        }
