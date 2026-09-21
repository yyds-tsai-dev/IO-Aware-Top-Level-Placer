"""Differentiable per-segment boundary capacity term (v2 design sec 5, P-D).

Demand, as this plan reads sec 5 (interpretations D-1/D-2 in
docs/superpowers/plans/2026-09-19-v2-p-d-capacity.md):

    D_s = sum over candidates c on segment s of
              w_{e(c)} * count(c) * q_{e(c),u(c)} * q_{e(c),v(c)} * alpha_c

  * count(c) is the observed crossing count `Candidates.count` carries for
    that (net, demand pair, segment) item. The capacity semantics string
    (`capacity.npz`: "usable tracks crossing the segment; one net crossing
    consumes one track") and evaluator_ref's own accumulation
    (`candidate_counts[key] += 1` per crossing, `evaluator_ref.py`) both charge
    per crossing, not per candidate item; dropping this factor (Task 4 fix
    round 1 item 3) undercounted D_s for any (net, pair, boundary) that
    crosses more than once and broke the "the term optimises what the
    evaluator measures" property the whole subproject leans on.
  * (u, v) is the candidate's *demand pair* -- the regions of the two MST-edge
    endpoints whose leg produced the crossing, not the segment's own boundary
    pair. For an adjacent-pair leg they coincide and this is sec 5's formula
    verbatim; for a feed-through they do not, and that is exactly what makes
    sec 5's "a traversing net crossed entry and exit segments, so both take
    demand" true (with the segment's own pair, q_{e,middle} == 0 kills both).
  * alpha is a softmax over the candidates of one (net, u, v, boundary pair)
    group, by -L1(net soft pin centroid, segment)/tau_b. Alternatives on one
    boundary share a group and one unit of demand; a feed-through's entry and
    exit lie on different boundaries, so they are different groups and each
    takes the full product.

Penalty, verbatim from sec 5:

    d_s = (D_s - C_s)/C_s,   L_cap = sum_s (2[d_s]_+^3 + [d_s]_+^2)

C1 at the knee, exactly zero below capacity. Phase-1 sec 5.4's
softplus((D-C)/C) is rejected there and must not reappear. For C_s == 0 the
unit rule forbids an epsilon, so d_s := D_s (interpretation D-3): finite,
zero at zero demand, and the penalty expression itself is unchanged.
"""
import numpy as np
import torch

from ioplace.ops.soft_assign import (_chunks, chunk_p_ell, region_sdf_l1,
                                     softmax_stats)

# pen''(d) = 12d + 2 is unbounded, so sec 4's "max_s pen''" has no supremum.
# The declared curvature is pen'' at the overflow level the design tolerates.
CAP_CURVATURE_DREF = 1.0


def cap_curvature(d_ref=CAP_CURVATURE_DREF):
    """pen''(d_ref) = 12*d_ref + 2 -- the value handed to
    `TermNormalizer.register(..., curvature=...)`. Cmax is a lambda-weighted
    *mean* over active terms (`norm.cmax_from_curvatures`), so a representative
    curvature is what the Lipschitz cap wants."""
    return 12.0 * float(d_ref) + 2.0


def cap_penalty(d):
    """2[d]_+^3 + [d]_+^2, elementwise. Exactly 0 for d <= 0."""
    if torch.is_tensor(d):
        r = d.clamp(min=0.0)
    else:
        r = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
    return 2.0 * r ** 3 + r ** 2


def cap_penalty_grad(d):
    """pen'(d) = 6[d]_+^2 + 2[d]_+. pen'(0) == 0 from both sides (C1)."""
    if torch.is_tensor(d):
        r = d.clamp(min=0.0)
    else:
        r = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
    return 6.0 * r ** 2 + 2.0 * r


def normalised_overflow(demand, capacity):
    """(d, inv) with d = (D - C)/C and inv = dd/dD.

    Interpretation D-3 (unit rule: zero-capacity segments stay blocked with a
    finite penalty, no epsilon substitution): where C == 0, d := D and
    inv := 1. `safe` keeps the *unused* branch of torch.where finite, because
    a NaN/inf there still poisons the backward pass."""
    positive = capacity > 0
    safe = torch.where(positive, capacity, torch.ones_like(capacity))
    d = torch.where(positive, (demand - capacity) / safe, demand)
    inv = torch.where(positive, 1.0 / safe, torch.ones_like(safe))
    return d, inv


def segment_softmax(u, group, n_groups):
    """softmax of `u` within each `group`. Shift-invariant (per-group max
    subtracted) so a large |u| cannot overflow."""
    peak = torch.full((n_groups,), -float("inf"), dtype=u.dtype, device=u.device)
    peak.scatter_reduce_(0, group, u, reduce="amax", include_self=True)
    e = torch.exp(u - peak[group])
    total = torch.zeros(n_groups, dtype=u.dtype, device=u.device)
    total.index_add_(0, group, e)
    return e / total[group]


def l1_point_box_dist(cx, cy, box):
    """L1 distance from (cx, cy) to the axis-aligned box [x0,y0,x1,y1]; zero
    inside. A segment's box is degenerate in one axis (a vertical segment has
    x0 == x1), so this is |cx - X| plus the clamped y distance -- one
    expression covering both orientations."""
    dx = (box[:, 0] - cx).clamp(min=0.0) + (cx - box[:, 2]).clamp(min=0.0)
    dy = (box[:, 1] - cy).clamp(min=0.0) + (cy - box[:, 3]).clamp(min=0.0)
    return dx + dy


def l1_point_box_grad(cx, cy, box):
    """(d d1/d cx, d d1/d cy).

    Task 4 fix round 1 item 2: `torch.clamp(min=...)`'s backward passes
    gradient through *at* the boundary itself (`x >= min`, not `x > min` --
    verified against autograd: `x.clamp(min=0)` at `x == 0` has grad 1, not
    0). `l1_point_box_dist` is built from two such clamps per axis, so the
    subgradient at a face (cx exactly on `box[:,0]`/`box[:,2]`, or cy on
    `box[:,1]`/`box[:,3]`) is +-1, matching `CapTermRef`'s own autograd there
    -- a strict `>`/`<` disagreed by exactly 1 at an endpoint-aligned
    centroid, which is silent everywhere `CapTermRef` is the only consumer
    but a hard failure of `CapTerm`'s (Task 5) `rel <= 1e-10` parity contract.
    It is only on a *degenerate* axis (`box[:,0] == box[:,2]`, a vertical
    segment's x-extent, or the H analogue in y) that the two boundary terms
    are simultaneously satisfied and cancel to a true 0."""
    ddx = (cx >= box[:, 2]).to(cx.dtype) - (cx <= box[:, 0]).to(cx.dtype)
    ddy = (cy >= box[:, 3]).to(cy.dtype) - (cy <= box[:, 1]).to(cy.dtype)
    return ddx, ddy


class _CapBase(torch.nn.Module):
    """Shared candidate/segment bookkeeping for CapTermRef and CapTerm."""

    def __init__(self, io_term, seg_box, seg_capacity, *, tau_b,
                 curvature_dref=CAP_CURVATURE_DREF):
        super().__init__()
        self.io_term = io_term
        device = io_term.rects.device
        box = torch.as_tensor(np.asarray(seg_box, dtype=np.float64),
                              dtype=torch.float64, device=device)
        capacity = torch.as_tensor(np.asarray(seg_capacity, dtype=np.float64),
                                   dtype=torch.float64, device=device)
        if box.ndim != 2 or box.shape[1] != 4:
            raise ValueError("seg_box must be (S,4)")
        if capacity.shape != (box.shape[0],):
            raise ValueError("seg_capacity must carry one value per segment")
        if not bool(torch.isfinite(capacity).all()) or bool((capacity < 0).any()):
            raise ValueError("finite nonnegative capacity required")
        if not (tau_b > 0):
            raise ValueError("tau_b must be positive")
        self.tau_b = float(tau_b)
        # pen'' is evaluated here for the declared curvature; Task 8's
        # --cap-curvature-dref moves it.
        self.curvature_dref = float(curvature_dref)
        self.register_buffer("seg_box", box)
        self.register_buffer("C", capacity)
        self.register_buffer("deg", torch.bincount(
            io_term.net_idx, minlength=io_term.n_active).to(torch.float64))
        self.n_groups = 0
        self.last_demand = None
        self.last_d = None
        for name in ("cand_net", "cand_u", "cand_v", "cand_seg", "cand_group",
                     "cand_count"):
            self.register_buffer(name,
                                 torch.zeros(0, dtype=torch.int64, device=device))

    @property
    def num_segments(self):
        return int(self.C.numel())

    def set_candidates(self, cand):
        """Install a `region_segments.Candidates` (already remapped into the IO
        term's active-net index space). Called by the driver on the
        `home_period` cadence, exactly like `FtTerm.set_home`."""
        device = self.C.device
        meta = self.io_term
        arrays = {}
        for name, values in (("cand_net", cand.net), ("cand_u", cand.u),
                             ("cand_v", cand.v), ("cand_seg", cand.seg),
                             ("cand_group", cand.group), ("cand_count", cand.count)):
            arrays[name] = torch.as_tensor(np.asarray(values, dtype=np.int64),
                                           dtype=torch.int64, device=device)
        size = arrays["cand_net"].numel()
        if any(t.numel() != size for t in arrays.values()):
            raise ValueError("candidate arrays must be the same length")
        if size:
            if bool(((arrays["cand_net"] < 0)
                     | (arrays["cand_net"] >= meta.n_active)).any()):
                raise ValueError("candidate net index outside the active nets")
            for key in ("cand_u", "cand_v"):
                if bool(((arrays[key] < 0) | (arrays[key] >= meta.K)).any()):
                    raise ValueError("candidate region id outside [0,K)")
            if bool((arrays["cand_u"] >= arrays["cand_v"]).any()):
                raise ValueError("candidate demand pairs must satisfy u < v")
            if bool(((arrays["cand_seg"] < 0)
                     | (arrays["cand_seg"] >= self.num_segments)).any()):
                raise ValueError("candidate segment id outside the table")
            if bool((arrays["cand_count"] < 0).any()):
                raise ValueError("candidate crossing counts must be non-negative")
            groups = arrays["cand_group"]
            if int(groups.min()) < 0 or int(groups.max()) >= int(cand.n_groups):
                raise ValueError("candidate group ids are not dense")
        for name, tensor in arrays.items():
            setattr(self, name, tensor)
        self.n_groups = int(cand.n_groups)

    def w_cand(self):
        """w_e per candidate, read live from the IO term so the reweighting
        paths (`--alpha-io`, `--ft-reweight`) are picked up."""
        return self.io_term.w[self.cand_net]

    def _centroids(self, x, y):
        """The net's soft pin centroid c_e: one index_add over the IO CSR's
        deduplicated node list (sec 5), in the same anchored coordinates the
        IO term uses, so P-F's --node-anchor choice carries through."""
        meta = self.io_term
        cx = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
        cy = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
        cx.index_add_(0, meta.net_idx, x[meta.node_idx].double())
        cy.index_add_(0, meta.net_idx, y[meta.node_idx].double())
        return cx / self.deg, cy / self.deg

    def _alpha(self, cx, cy):
        d1 = l1_point_box_dist(cx[self.cand_net], cy[self.cand_net],
                               self.seg_box[self.cand_seg])
        return segment_softmax(-d1 / self.tau_b, self.cand_group, self.n_groups)


class CapTermRef(_CapBase):
    """Dense autograd oracle -- materialises an (E,K) `q`, so it is for tests
    and small-scale probes only, exactly like `IoTermRef`/`FtTermRef`. The
    production term is `CapTerm` in this module and is bound to this class by
    an equivalence test."""

    def _demand(self, pos, tau):
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        x = torch.cat((x[:meta.num_movable], x[meta.num_movable:].detach()))
        y = torch.cat((y[:meta.num_movable], y[meta.num_movable:].detach()))
        # Task 4 fix round 1 item 1: the sec 7 anchor must be applied before
        # anything (q's region SDF *and* the net centroid) reads x/y, exactly
        # like IoTermRef/IoTerm._split_xy -- otherwise membership and the
        # centroid silently fall back to the cell lower-left even when the
        # flow's default (and the freeze's own membership read, freeze.py)
        # is node_anchor="center". A constant per-node offset, so it changes
        # no gradient -- only where the region SDF/centroid are evaluated.
        x, y = meta._anchor_xy(x, y)
        rects = meta.rects.to(dtype=x.dtype)
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau)
        sdf = region_sdf_l1(x, y, rects, meta.rect2region, 0, meta.K)
        _p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
        S = torch.zeros((meta.n_active, meta.K), dtype=torch.float64,
                        device=x.device)
        S = S.index_add(0, meta.net_idx, ell[meta.node_idx].double())
        q = -torch.expm1(S)
        cx, cy = self._centroids(x, y)
        alpha = self._alpha(cx, cy)
        # Task 4 fix round 1 item 3: charge per observed crossing
        # (`cand.count`), not once per (net, pair, boundary) candidate item --
        # see the module docstring's D_s formula.
        product = (self.w_cand() * self.cand_count.to(torch.float64)
                   * q[self.cand_net, self.cand_u]
                   * q[self.cand_net, self.cand_v] * alpha)
        demand = torch.zeros(self.num_segments, dtype=torch.float64,
                             device=x.device)
        return demand.index_add(0, self.cand_seg, product)

    def demand(self, pos, tau):
        return self._demand(pos, tau)

    def forward(self, pos, tau, lambda_cap):
        if lambda_cap == 0.0 or self.cand_net.numel() == 0:
            # fp64, not pos.new_zeros(()): the non-short-circuit branch below
            # always returns a float64 scalar (C/demand/cap_penalty are all
            # float64), so an fp32 pos must not silently change this
            # branch's return dtype (minor, fix round 1 free item).
            return torch.zeros((), dtype=torch.float64, device=pos.device)
        demand = self._demand(pos, tau)
        d, _inv = normalised_overflow(demand, self.C)
        self.last_demand, self.last_d = demand.detach(), d.detach()
        return lambda_cap * cap_penalty(d).sum()


class _CapFn(torch.autograd.Function):
    """Chunked-k forward/backward, structurally identical to `_FtFn`
    (ops/ft_term.py): FWD-1 reduce (m/t/argmax over all K), FWD-2 accumulate
    (per chunk q -> the per-candidate cofactor caches), BWD-1 reduce (A_i over
    all K), BWD-2 scatter (dL/dz -> positions via a local autograd.grad on
    region_sdf_l1). Nothing of shape (N,K)/(P,K)/(E,K) is ever allocated; the
    only candidate-sized state is qa/qb, two (M,) float64 arrays -- sec 5's
    "both cofactors are cached per candidate ... so no (E,K) tensor appears".

    `x`/`y` arrive already anchored (CapTerm.forward applies
    `meta._anchor_xy` first, as `CapTermRef._demand` and `IoTerm.forward`
    do), so q's region SDF and the net centroid read the same point.

    Gradient contract (CapTermRef's autograd, audited in Task 4):
      g_c   = pen'(d_s) * inv_s                       (dL/dD_s, s = seg(c))
      P_c   = w_e * count_c * q_{e,u} * q_{e,v}
      q:     dL/dq_{e,u} += g_c * P_c/q_u * alpha_c  (exact product rule)
      alpha: dL/du_c = alpha_c (g_c P_c - sum_{c' in group} alpha_c' g_c' P_c'),
             u_c = -d1_c / tau_b, d1 through l1_point_box_grad, c_e = mean.
    """

    @staticmethod
    def forward(ctx, x, y, meta, cap, tau, lambda_cap):
        rects = meta.rects.to(dtype=x.dtype)
        # FWD-1 (reduce): m, t, argmax over ALL K regions.
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau,
                                 chunk=meta.k_chunk)
        size = cap.cand_net.numel()
        qa = torch.zeros(size, dtype=torch.float64, device=x.device)
        qb = torch.zeros(size, dtype=torch.float64, device=x.device)
        # FWD-2 (accumulate): per chunk q -> the two (M,) cofactor caches.
        for lo, hi in _chunks(meta.K, meta.k_chunk):
            sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)
            _p_c, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
            S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                              device=x.device).index_add_(
                                  0, meta.net_idx, ell_c[meta.node_idx].double())
            q_c = -torch.expm1(S_c)
            sel = (cap.cand_u >= lo) & (cap.cand_u < hi)
            if bool(sel.any()):
                qa[sel] = q_c[cap.cand_net[sel], cap.cand_u[sel] - lo]
            sel = (cap.cand_v >= lo) & (cap.cand_v < hi)
            if bool(sel.any()):
                qb[sel] = q_c[cap.cand_net[sel], cap.cand_v[sel] - lo]

        cx, cy = cap._centroids(x, y)
        alpha = cap._alpha(cx, cy)
        # w_e * count_c: charge per observed crossing, exactly as
        # CapTermRef._demand does (Task 4 fix round 1 item 3).
        wc = cap.w_cand().double() * cap.cand_count.to(torch.float64)
        demand = torch.zeros(cap.num_segments, dtype=torch.float64,
                             device=x.device).index_add_(
                                 0, cap.cand_seg, wc * qa * qb * alpha)
        d, inv = normalised_overflow(demand, cap.C)
        value = cap_penalty(d).sum()

        cap.last_demand, cap.last_d = demand.detach(), d.detach()
        meta.last_peak_chunk_elems = meta.k_chunk * max(meta.num_physical,
                                                        meta._n_pins_dedup)
        ctx.save_for_backward(x, y, m, t, am, qa, qb, alpha, d, inv, cx, cy,
                              wc.detach().clone())
        ctx.meta, ctx.cap, ctx.tau, ctx.lambda_cap = meta, cap, tau, lambda_cap
        return lambda_cap * value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gout):
        x, y, m, t, am, qa, qb, alpha, d, inv, cx, cy, wc = ctx.saved_tensors
        meta, cap, tau, lambda_cap = ctx.meta, ctx.cap, ctx.tau, ctx.lambda_cap
        rects = meta.rects.to(dtype=x.dtype)
        n = x.shape[0]

        with torch.no_grad():
            # dL/dD_s, held fixed while D_s is differentiated (chain rule).
            g_seg = cap_penalty_grad(d) * inv
            g_cand = g_seg[cap.cand_seg]

            # ---- alpha path: closed form, only (M,)/(E,) tensors. Not gated
            # on q; a singleton group gives exactly 0 here (limitation D-4).
            dl_dalpha = g_cand * wc * qa * qb
            pooled = torch.zeros(cap.n_groups, dtype=torch.float64,
                                 device=x.device)
            pooled.index_add_(0, cap.cand_group, alpha * dl_dalpha)
            dl_du = alpha * (dl_dalpha - pooled[cap.cand_group])
            dl_dd1 = -dl_du / cap.tau_b                  # u = -d1 / tau_b
            ddx, ddy = l1_point_box_grad(cx[cap.cand_net], cy[cap.cand_net],
                                         cap.seg_box[cap.cand_seg])
            gcx = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
            gcy = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
            gcx.index_add_(0, cap.cand_net, dl_dd1 * ddx)
            gcy.index_add_(0, cap.cand_net, dl_dd1 * ddy)
            gx = torch.zeros(n, dtype=torch.float64, device=x.device)
            gy = torch.zeros(n, dtype=torch.float64, device=x.device)
            gx.index_add_(0, meta.node_idx, (gcx / cap.deg)[meta.net_idx])
            gy.index_add_(0, meta.node_idx, (gcy / cap.deg)[meta.net_idx])

            # ---- q path: the exact product derivative, both cofactors ----
            coeff_u = g_cand * wc * qb * alpha           # dL/dq_{e,u}
            coeff_v = g_cand * wc * qa * alpha           # dL/dq_{e,v}

            def coeff_chunk(lo, hi):
                out = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device)
                sel = (cap.cand_u >= lo) & (cap.cand_u < hi)
                if bool(sel.any()):
                    out.index_put_((cap.cand_net[sel], cap.cand_u[sel] - lo),
                                   coeff_u[sel], accumulate=True)
                sel = (cap.cand_v >= lo) & (cap.cand_v < hi)
                if bool(sel.any()):
                    out.index_put_((cap.cand_net[sel], cap.cand_v[sel] - lo),
                                   coeff_v[sel], accumulate=True)
                return out

            # BWD-1 (reduce): A_i = sum over ALL k of c_{i,k} * p_{i,k}.
            acc = torch.zeros(n, dtype=torch.float64, device=x.device)
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx,
                                                              ell_pins)
                c_pins = (coeff_chunk(lo, hi)[meta.net_idx]
                          * torch.exp(S_c[meta.net_idx] - ell_pins))
                c_nodes = torch.zeros((n, hi - lo), dtype=torch.float64,
                                      device=x.device).index_add_(
                                          0, meta.node_idx, c_pins)
                acc = acc + (c_nodes * p_c.double()).sum(dim=1)

            # BWD-2 (scatter): dL/dz = p*(c - A); dL/dx += dL/dz * (-1/tau) *
            # d(sdf)/dx, through a *local* autograd.grad on the very same
            # region_sdf_l1 the reference differentiates, so ties and boundary
            # subgradients match CapTermRef exactly.
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    yg = y.detach().requires_grad_(True)
                    sdf_c = region_sdf_l1(xg, yg, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c.detach(), m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx,
                                                              ell_pins)
                c_pins = (coeff_chunk(lo, hi)[meta.net_idx]
                          * torch.exp(S_c[meta.net_idx] - ell_pins))
                c_nodes = torch.zeros((n, hi - lo), dtype=torch.float64,
                                      device=x.device).index_add_(
                                          0, meta.node_idx, c_pins)
                dl_dz = p_c.double() * (c_nodes - acc.unsqueeze(1))
                w_out = dl_dz * (-1.0 / tau)
                gxc, gyc = torch.autograd.grad(sdf_c, [xg, yg],
                                               grad_outputs=w_out.to(sdf_c.dtype))
                gx = gx + gxc.double()
                gy = gy + gyc.double()

            scale = lambda_cap * gout.double()
            gx = gx * scale
            gy = gy * scale
            gx[meta.num_movable:] = 0.0       # fixed + filler never move
            gy[meta.num_movable:] = 0.0
        return gx.to(x.dtype), gy.to(y.dtype), None, None, None, None


class CapTerm(_CapBase):
    """Chunked production capacity term. Same value/gradient contract as
    `CapTermRef`, routed through `_CapFn` so no forward or backward path ever
    allocates an (N,K)/(P,K)/(E,K) tensor. **This is the only class a driver
    may use.** `lambda_cap` is an argument: a caller must pass the
    normalizer's applied value (`applied_lambda(name, iteration)`, ruling
    D-6), never read `lambdas[name]`."""

    @property
    def curvature(self):
        """Handed to `TermNormalizer.register(..., curvature=...)` (sec 4)."""
        return cap_curvature(self.curvature_dref)

    def forward(self, pos, tau, lambda_cap):
        if lambda_cap == 0.0 or self.cand_net.numel() == 0:
            return pos.new_zeros(())
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        # sec 7 anchor, applied once before q's SDF *and* the centroid read
        # x/y -- the same order as CapTermRef._demand and IoTerm.forward.
        x, y = meta._anchor_xy(x, y)
        return _CapFn.apply(x, y, meta, self, tau, lambda_cap)

    def demand(self, pos, tau):
        """Per-segment soft demand D_s, no grad -- the GP surrogate side of the
        sec 9 rank-correlation experiment."""
        with torch.no_grad():
            if self.cand_net.numel() == 0:
                return torch.zeros(self.num_segments, dtype=torch.float64,
                                   device=self.C.device)
            self.forward(pos.detach(), tau, 1.0)
            return self.last_demand.clone()

    def diagnostics(self, pos, tau):
        """Reporting only. The driver logs `max_d` every probe so the declared
        curvature (pen'' at d_ref) stays auditable against real data."""
        with torch.no_grad():
            if self.cand_net.numel() == 0:
                return {"l_cap": 0.0, "demand_total": 0.0, "max_d": 0.0,
                        "num_over_capacity": 0, "num_candidates": 0,
                        "num_groups": 0, "frac_singleton_groups": 0.0}
            value = float(self.forward(pos.detach(), tau, 1.0))
            sizes = torch.bincount(self.cand_group, minlength=self.n_groups)
            return {"l_cap": value,
                    "demand_total": float(self.last_demand.sum()),
                    "max_d": float(self.last_d.max()),
                    "num_over_capacity": int((self.last_d > 0).sum()),
                    "num_candidates": int(self.cand_net.numel()),
                    "num_groups": int(self.n_groups),
                    # limitation D-4: a group of size 1 has no alpha gradient
                    # in any phase; on grid geometry every group is one.
                    "frac_singleton_groups": float((sizes == 1).double().mean())}


class CapNormTerm(object):
    """`TermNormalizer` adapter, alongside `ops/norm_terms.IoNormTerm` and
    `FtNormTerm`: `value(pos, ctx)` returns the *unweighted* L_cap at the
    schedule's live tau, and the normalizer owns the backward, the
    fixed/filler masking and the norm order (design sec 4)."""

    def __init__(self, cap_term):
        self.cap_term = cap_term

    def value(self, pos, ctx):
        return self.cap_term(pos, ctx["tau"], 1.0)
