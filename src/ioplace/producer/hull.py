"""Per-partition convex hulls for the GrandPlan grouping loss.

Algorithm 1 (directional-extrema candidate reduction) and the area cap of
docs/research/2026-09-18-grandplan-digest.md section 2.1, with the constants
fixed by the v2 spec section 2: m=16, q=0.90, alpha=0.25, K_dir=64, so quickhull
never sees more than 2*m*(K_dir+1) = 2080 points per region even at 11M cells.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

DIRECTIONS_M = 16
QUANTILE_Q = 0.90
BAND_ALPHA = 0.25
K_DIR = 64
MACRO_MAX_POINTS = 64


def reduce_candidates(pts, m=DIRECTIONS_M, q=QUANTILE_Q, alpha=BAND_ALPHA,
                      k_dir=K_DIR):
    """Algorithm 1. For each of m equally spaced directions, keep the quantile
    band [t, t + alpha*(s_max - t)] with t = quantile(s, q), capped at the k_dir
    projections closest to t, PLUS that direction's support point (ruling D1);
    repeat with the mirrored direction (which is the paper's
    t_lo = quantile(s, 1-q) branch); dedup.

    The output bound is 2*m*(k_dir + 1) before dedup; antipodal direction pairs select
    largely the same points, which is why the spec quotes "<=1024 points/region"
    at these defaults. Fully deterministic: no RNG, and np.unique sorts.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= k_dir:
        return np.unique(pts, axis=0)
    keep = np.zeros(len(pts), dtype=bool)
    for j in range(m):
        th = j * (2.0 * np.pi / m)
        s = pts[:, 0] * np.cos(th) + pts[:, 1] * np.sin(th)
        for sign in (1.0, -1.0):
            ss = sign * s
            t = float(np.quantile(ss, q))
            hi = float(ss.max())
            band = np.nonzero((ss >= t) & (ss <= t + alpha * (hi - t)))[0]
            if len(band) > k_dir:
                order = np.argsort(ss[band] - t, kind="stable")
                band = band[order[:k_dir]]
            keep[band] = True
            # Ruling D1, a NAMED DEVIATION from the digest's literal band: the
            # band [t, t+alpha*(s_max-t)] plus the "keep the k_dir closest to
            # t" cap selects a shell just above the q-quantile and throws the
            # support points away -- measured, the hull of the reduced set
            # collapses onto the dense cluster (area 0.95 on a cloud whose
            # true hull is 100). Always retain this direction's argmax.
            keep[int(np.argmax(ss))] = True
    return np.unique(pts[keep], axis=0)


def polygon_area(verts):
    """Shoelace absolute area."""
    v = np.asarray(verts, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ccw(verts):
    v = np.asarray(verts, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return v if signed > 0 else v[::-1].copy()


def convex_hull(pts):
    """Quickhull (scipy.spatial.ConvexHull is Qhull) over the deduplicated
    points, returned counter-clockwise. Fewer than three distinct points, or a
    collinear set (QhullError), falls back to the axis-aligned bounding box
    inflated to a non-degenerate rectangle -- the anchor tables downstream need
    a polygon with positive area and at least three vertices."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    uniq = np.unique(pts, axis=0)
    if len(uniq) >= 3:
        try:
            return _ccw(uniq[ConvexHull(uniq).vertices])
        except QhullError:
            pass
    if len(uniq) == 0:
        raise ValueError("convex_hull needs at least one point")
    lo, hi = uniq.min(axis=0), uniq.max(axis=0)
    # Scale eps to the coordinate magnitude, not just the (possibly zero)
    # span (hi - lo): a single point far from the origin otherwise inflates
    # to an absolute-1e-9-sided box whose area underflows to zero -- not
    # merely because the box is tiny, but because the shoelace formula
    # differences O(coord**2)-sized terms, which loses everything below
    # ~coord**2 * 2**-52. The scale factor here must clear that floor (empirically
    # verified down to 1e-7); 1e-6 matches the span-relative term above and
    # leaves a >10x margin.
    scale = np.maximum(np.abs(lo), np.abs(hi))
    eps = np.maximum((hi - lo) * 1e-6, np.maximum(scale * 1e-6, 1e-9))
    hi = np.maximum(hi, lo + eps)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def shrink_to_area(verts, a_max, rel_tol=1e-3, max_iter=60):
    """Area cap: uniformly scale the hull vertices toward their centroid until
    Area(H) <= A_max (digest section 2.1). Spec section 2 mandates bisection to
    1e-3 relative area; the closed form s = sqrt(a_max/a) is the oracle the unit
    test checks this against."""
    if a_max <= 0.0:
        raise ValueError("shrink_to_area requires a_max > 0")
    v = np.asarray(verts, dtype=np.float64)
    a = polygon_area(v)
    if a <= a_max or a <= 0.0:
        return v
    c = v.mean(axis=0)
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        s = 0.5 * (lo + hi)
        a_s = polygon_area(c + s * (v - c))
        if a_s > a_max:
            hi = s
        else:
            lo = s
            if (a_max - a_s) <= rel_tol * a_max:
                break
    return c + lo * (v - c)


def macro_pseudo_points(x, y, w, h, pitch_x, pitch_y,
                        max_per_macro=MACRO_MAX_POINTS):
    """Macro enrichment (digest section 2.1): a regular grid of points inside
    each macro footprint at the mean standard-cell pitch, capped at
    max_per_macro per macro (spec section 2). x, y are lower-left corners.
    Points sit at sub-cell centres, which lie strictly inside the footprint
    for macros with positive width and height; a macro with zero width and/or
    height has no interior to be strictly inside, so its point(s) fall on
    that degenerate edge/corner instead."""
    if pitch_x <= 0.0 or pitch_y <= 0.0:
        raise ValueError("macro_pseudo_points requires positive pitch_x and pitch_y")
    cap_side = max(1, int(np.floor(np.sqrt(max_per_macro))))
    out = []
    for xi, yi, wi, hi in zip(np.asarray(x, dtype=np.float64),
                              np.asarray(y, dtype=np.float64),
                              np.asarray(w, dtype=np.float64),
                              np.asarray(h, dtype=np.float64)):
        nx = int(min(max(1, round(wi / pitch_x)), cap_side))
        ny = int(min(max(1, round(hi / pitch_y)), cap_side))
        gx = xi + (np.arange(nx) + 0.5) * (wi / nx)
        gy = yi + (np.arange(ny) + 0.5) * (hi / ny)
        out.append(np.stack(np.meshgrid(gx, gy, indexing="ij"),
                            axis=-1).reshape(-1, 2))
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(out, axis=0)


def build_hull(pts, a_max, m=DIRECTIONS_M, q=QUANTILE_Q, alpha=BAND_ALPHA,
               k_dir=K_DIR):
    """Algorithm 1 -> quickhull -> area cap, the per-rebuild pipeline."""
    return shrink_to_area(
        convex_hull(reduce_candidates(pts, m=m, q=q, alpha=alpha, k_dir=k_dir)),
        a_max)


@dataclass
class AnchorTables:
    """Eq.1's two anchor fields, rasterised onto a lattice x lattice bin grid.

    Both offset tables are stored in BIN WIDTHS, i.e. the x component divided
    by (xh-xl)/L and the y component by (yh-yl)/L -- NOT in the die's own
    units. The consumer reconstructs
    `anchor = centre(bin(x)) + offset * (bin_w, bin_h)`;
    `grouping_term.GroupingTerm._lookup` is the one place that does it.

    pull_off (K, L*L, 2) fp16 : (Pi_{H_k}(centre(b)) - centre(b)) / bin size,
                                zero inside H_k.
    pull_on  (K, L*L)   bool  : centre(b) is OUTSIDE H_k (Eq.1's 1{x not in Omega_k}).
    push_off (K, L*L, 2) fp16 : mean over foreign hulls s != k containing b of
                                (Pi_{dH_s}(centre(b)) - centre(b)) / bin size.
    push_cnt (K, L*L)   uint8 : how many foreign hulls contain centre(b).

    Offsets rather than absolute coordinates because fp16 has ~11 mantissa bits:
    an absolute die coordinate would quantise to ~0.1% of the die. An offset is
    bounded by the die extent E, so fp16's 2^-11 relative error gives an
    absolute error of at most 2**-11 * E == (E/512)/4 == 1/4 bin width,
    independent of die size -- the 512 lattice and fp16's 11 mantissa bits
    cancel. That is well under the >= sqrt(2)/2 bin the frozen rasterisation
    itself already costs (projection is 1-Lipschitz, so using Pi_H(centre(b))
    in place of Pi_H(x_cell) can differ by up to a bin's half-diagonal): fp16
    does not add a dominant error term, and an fp32 pull table would cost
    +16 MiB at K=16 (+32 MiB at K=32) of GPU memory for no measurable gain.

    Dividing by the bin size does not change that bound -- it is the SAME
    statement in different units (a normalised offset is at most L, and
    2**-11 * L == L/2048 == 0.25 bin at L = 512) -- but it removes the bound's
    dependence on absolute die size from the fp16 *range*. Storing raw scaled
    units made the largest representable offset the die extent, so the
    `>= 32768` overflow guard below was a hard ceiling on die size: P-C Task 10
    measured `mempool_tile_wrap` at 4168 scaled units with 127,433 movable
    cells, and a 30M-cell NanGate45 die is ~15.4x that linearly (~64,000),
    which would have aborted inside the first hull rebuild -- at the scale this
    project exists to reach. Normalised, the largest magnitude any die can
    produce is L itself, so the guard is on the lattice and is unreachable at
    every lattice this codebase uses.

    (mean, count) rather than one anchor per foreign hull because
    sum_s ||x - a_s||^2 = n*||x - abar||^2 + (sum_s ||a_s||^2 - n*||abar||^2):
    the bracket does not depend on x, so this reproduces Eq.2's push gradient
    exactly and Eq.1's push value up to a frozen constant, while keeping the
    per-cell cost O(1) instead of O(K) -- the hotspot spec section 2 calls out.
    """
    lattice: int
    die: tuple
    pull_off: torch.Tensor
    pull_on: torch.Tensor
    push_off: torch.Tensor
    push_cnt: torch.Tensor

    @property
    def k(self):
        return int(self.pull_off.shape[0])


def nearest_on_polygon_boundary(px, py, verts, chunk=16384):
    """Nearest point on a CONVEX polygon's boundary, plus an inside test.

    For a convex H this single routine yields both of Eq.1's anchors: the
    nearest boundary point is Pi_{H}(x) when x is outside H and Pi_{dH}(x) when
    x is inside it. Ties between equidistant edges break on the first edge in
    vertex order (torch.argmin's own rule) -- deterministic, which is all the
    frozen-anchor semantics require.

    px, py: (B,) float64 tensors. verts: (V,2) counter-clockwise.
    Returns (proj (B,2) float64, inside (B,) bool) on px's device.
    """
    dev = px.device
    v = torch.as_tensor(np.asarray(verts, dtype=np.float64), device=dev)
    a = v
    e = torch.roll(v, -1, dims=0) - v                       # (V,2) edge vectors
    ee = (e * e).sum(dim=1).clamp(min=1e-30)                # (V,)
    nrm = torch.stack([e[:, 1], -e[:, 0]], dim=1)           # outward normal (CCW)
    n = int(px.shape[0])
    proj = torch.empty((n, 2), dtype=torch.float64, device=dev)
    inside = torch.empty((n,), dtype=torch.bool, device=dev)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        qx, qy = px[lo:hi], py[lo:hi]
        wx = qx.unsqueeze(1) - a[:, 0].unsqueeze(0)         # (c,V)
        wy = qy.unsqueeze(1) - a[:, 1].unsqueeze(0)
        t = ((wx * e[:, 0] + wy * e[:, 1]) / ee).clamp(0.0, 1.0)
        cxv = a[:, 0] + t * e[:, 0]
        cyv = a[:, 1] + t * e[:, 1]
        d2 = (qx.unsqueeze(1) - cxv) ** 2 + (qy.unsqueeze(1) - cyv) ** 2
        j = d2.argmin(dim=1, keepdim=True)
        proj[lo:hi, 0] = cxv.gather(1, j).squeeze(1)
        proj[lo:hi, 1] = cyv.gather(1, j).squeeze(1)
        inside[lo:hi] = ((wx * nrm[:, 0] + wy * nrm[:, 1]) <= 0.0).all(dim=1)
    return proj, inside


FP16_OFFSET_LIMIT = 32768.0


def check_anchor_table_range(die, lattice):
    """Precondition of `anchor_tables`' fp16 offset store, factored out so a
    driver can fail fast BEFORE it starts a GP instead of aborting inside the
    first hull rebuild (P-C Task 10 fix round 1, finding I3).

    Two conditions, both `ValueError` rather than `assert` because python -O
    strips asserts and these are input contracts:

    * The die must have a strictly positive extent on both axes. The offset
      tables are normalised by the bin size, so a degenerate die would divide
      by zero and store `nan`/`inf`.
    * A normalised offset is at most the lattice L (the largest offset an axis
      can produce is that axis's whole extent, which is L bins), so L must stay
      under fp16's range with margin. `FP16_OFFSET_LIMIT` keeps the 2x margin
      against fp16's 65504 that the previous, die-extent-based form used; at
      the lattice 512 this codebase uses, the limit is 64x away and unreachable
      -- which is the point of the normalisation (see `AnchorTables`).
    """
    xl, yl, xh, yh = (float(v) for v in die)
    if not (xh > xl and yh > yl):
        raise ValueError(
            f"anchor_tables: die {(xl, yl, xh, yh)!r} must have a positive "
            "extent on both axes; the fp16 offsets are stored in bin widths, "
            "so a zero extent divides by zero")
    L = int(lattice)
    if L <= 0:
        raise ValueError(f"anchor_tables: lattice must be positive, got {L!r}")
    if L >= FP16_OFFSET_LIMIT:
        raise ValueError(
            f"anchor_tables: lattice {L} >= {FP16_OFFSET_LIMIT} would overflow "
            "the fp16 pull_off/push_off tables, whose offsets are stored in "
            "bin widths and so are bounded by the lattice")


def anchor_tables(hulls, die, lattice, device="cuda", bin_chunk=16384):
    """Rasterise Eq.1's anchors for every hull onto the lattice bin grid.

    hulls: list of (V,2) counter-clockwise vertex arrays, one per region, in
    region-id order. die: (xl, yl, xh, yh) in the SAME coordinate system the
    consumer's positions are in (inside GP that is the scaled post-initialize()
    system, not the native one).

    The returned offsets are in BIN WIDTHS -- see `AnchorTables`.
    """
    check_anchor_table_range(die, lattice)
    xl, yl, xh, yh = (float(v) for v in die)
    L = int(lattice)
    cw, ch = (xh - xl) / L, (yh - yl) / L
    bin_size = torch.tensor([cw, ch], dtype=torch.float64, device=device)
    idx = torch.arange(L, dtype=torch.float64, device=device)
    gx = (xl + (idx + 0.5) * cw).repeat(L)                  # bin b = iy*L + ix
    gy = (yl + (idx + 0.5) * ch).repeat_interleave(L)
    centre = torch.stack([gx, gy], dim=1)
    K, B = len(hulls), L * L

    pull_off = torch.zeros((K, B, 2), dtype=torch.float16, device=device)
    pull_on = torch.zeros((K, B), dtype=torch.bool, device=device)
    own_off = torch.zeros((K, B, 2), dtype=torch.float32, device=device)
    own_in = torch.zeros((K, B), dtype=torch.bool, device=device)
    sum_off = torch.zeros((B, 2), dtype=torch.float64, device=device)
    sum_n = torch.zeros((B,), dtype=torch.int32, device=device)

    zero2 = torch.zeros((B, 2), dtype=torch.float64, device=device)
    for k, verts in enumerate(hulls):
        # Ruling D11: nearest_on_polygon_boundary builds (chunk, V) temporaries
        # and nothing bounds V below 2*m*(k_dir+1); at V=2048 a 16384 chunk
        # would be ~268 MiB per temporary. Measured V after reduction is 6-13,
        # so this clamp never binds in practice and costs nothing.
        v_count = len(np.asarray(verts, dtype=np.float64).reshape(-1, 2))
        chunk = max(1024, bin_chunk // max(1, v_count // 16))
        proj, inside = nearest_on_polygon_boundary(gx, gy, verts, chunk=chunk)
        # in BIN WIDTHS from here down: every consumer of pull_off/push_off
        # multiplies by `bin_size` again (AnchorTables' docstring).
        off = (proj - centre) / bin_size
        pull_off[k] = torch.where(inside.unsqueeze(1), zero2, off).half()
        pull_on[k] = ~inside
        own_off[k] = off.float()
        own_in[k] = inside
        sum_off += torch.where(inside.unsqueeze(1), off, zero2)
        sum_n += inside.to(torch.int32)

    push_off = torch.zeros((K, B, 2), dtype=torch.float16, device=device)
    push_cnt = torch.zeros((K, B), dtype=torch.uint8, device=device)
    for k in range(K):
        n = (sum_n - own_in[k].to(torch.int32)).clamp(min=0)
        s = sum_off - torch.where(own_in[k].unsqueeze(1),
                                  own_off[k].double(), zero2)
        mean = s / n.clamp(min=1).unsqueeze(1).double()
        push_off[k] = torch.where((n > 0).unsqueeze(1), mean, zero2).half()
        push_cnt[k] = n.clamp(max=255).to(torch.uint8)

    return AnchorTables(lattice=L, die=(xl, yl, xh, yh), pull_off=pull_off,
                        pull_on=pull_on, push_off=push_off, push_cnt=push_cnt)


QUANTILE_SUBSAMPLE = 8_000_000


def reduce_candidates_torch(x, y, m=DIRECTIONS_M, q=QUANTILE_Q,
                            alpha=BAND_ALPHA, k_dir=K_DIR,
                            quantile_subsample=QUANTILE_SUBSAMPLE):
    """Algorithm 1 on the device (spec section 2: "Algorithm-1 candidate
    reduction on GPU"). Same semantics as reduce_candidates; only the
    projections, the quantile and the top-k live on the GPU, and only the
    surviving <=2*m*k_dir candidates come back to the host for quickhull.

    torch.quantile has an input-element limit, so above quantile_subsample
    (spec's QUANTILE_SUBSAMPLE=8_000_000 by default; exposed as a keyword only
    so tests can drive the branch without allocating 8e6 points) the threshold
    is estimated from a deterministic stride subsample -- a threshold
    estimate, not a filter: the band test still runs over every point.

    The within-band cutoff is a stable ascending sort of (ss[idx] - t), not
    torch.topk: topk makes no tie-break guarantee, while np.argsort(kind=
    "stable") on the CPU path breaks ties by ascending original index. A
    stable sort over idx (itself ascending, since it comes from nonzero)
    reproduces that exact ordering, which matters because chip data is
    grid-aligned and legitimately ties at the band edge (unlike a continuous
    synthetic cloud, where ties have measure zero).

    Like the numpy path, each direction's support point is always kept
    (ruling D1), so the output bound is 2*m*(k_dir + 1) before dedup.
    """
    n = int(x.numel())
    if n <= k_dir:
        return np.unique(torch.stack([x, y], dim=1).double().cpu().numpy(), axis=0)
    keep = torch.zeros(n, dtype=torch.bool, device=x.device)
    stride = max(1, n // quantile_subsample + (1 if n % quantile_subsample else 0))
    for j in range(m):
        th = j * (2.0 * math.pi / m)
        s = x.double() * math.cos(th) + y.double() * math.sin(th)
        for sign in (1.0, -1.0):
            ss = sign * s
            t = torch.quantile(ss[::stride].contiguous(), q)
            hi = ss.max()
            idx = ((ss >= t) & (ss <= t + alpha * (hi - t))).nonzero(as_tuple=True)[0]
            if idx.numel() > k_dir:
                order = torch.sort(ss[idx] - t, stable=True).indices
                idx = idx[order[:k_dir]]
            keep[idx] = True
            # Ruling D1, same named deviation as the numpy path: the band plus
            # the k_dir cap drops the support points, and the hull of the
            # reduced set collapses (measured area 95.78 against a true 100).
            # No int()/.item() here: indexing keep with the 0-dim argmax
            # tensor stays device-side and avoids a host sync every iteration.
            keep[torch.argmax(ss)] = True
    return np.unique(
        torch.stack([x[keep], y[keep]], dim=1).double().cpu().numpy(), axis=0)
