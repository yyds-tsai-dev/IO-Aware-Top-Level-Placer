"""Per-partition convex hulls for the GrandPlan grouping loss.

Algorithm 1 (directional-extrema candidate reduction) and the area cap of
docs/research/2026-09-18-grandplan-digest.md section 2.1, with the constants
fixed by the v2 spec section 2: m=16, q=0.90, alpha=0.25, K_dir=64, so quickhull
never sees more than 2*m*(K_dir+1) = 2080 points per region even at 11M cells.
"""
import numpy as np
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
