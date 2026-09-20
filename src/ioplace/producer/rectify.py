"""Bin map -> rectilinear RegionSet with a hard rect_max budget.

spec section 2 / section 10 risk 1: maximal horizontal strips merged vertically,
then rect_max=8 enforced by filling the smallest notches, then
RegionSet.validate(). The budget is a memory contract -- ops/soft_assign.py's
region_sdf_l1 builds (N, rects-per-chunk) temporaries.
"""
import numpy as np
from scipy import ndimage

from ioplace.regions import RegionSet, RegionSpec

_CC = ndimage.generate_binary_structure(2, 1)      # 4-connected


def _row_runs(row):
    """Maximal True runs of a 1-D boolean row as (x0, x1) half-open pairs."""
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def mask_to_rects(mask):
    """Maximal horizontal strips merged vertically: identical [x0,x1) runs in
    consecutive rows become one rectangle. Returns (R,4) int64 half-open
    [x0, y0, x1, y1) in bin coordinates, sorted in raster order."""
    mask = np.asarray(mask, dtype=bool)
    ny = mask.shape[0]
    open_runs = {}
    rects = []
    for y in range(ny + 1):
        runs = set() if y == ny else set(_row_runs(mask[y]))
        for key in [k for k in open_runs if k not in runs]:
            rects.append((key[0], open_runs.pop(key), key[1], y))
        for key in runs:
            open_runs.setdefault(key, y)
    rects.sort(key=lambda r: (r[1], r[0]))
    return np.array(rects, dtype=np.int64).reshape(-1, 4)


def region_rect_counts(labels, k):
    lab = np.asarray(labels)
    return [len(mask_to_rects(lab == kk)) for kk in range(k)]


def _connected(mask):
    return bool(mask.any()) and ndimage.label(mask, structure=_CC)[1] == 1


def _feasible(lab, r, sel):
    """A candidate notch fill is feasible iff every donor stays non-empty and
    4-connected and the target stays 4-connected."""
    if not _connected((lab == r) | sel):
        return False
    for d in np.unique(lab[sel]):
        if int(d) == r:
            continue
        if not _connected((lab == int(d)) & ~sel):
            return False
    return True


def _fill_one_notch(lab, r):
    """Fill one notch of region r, preferring the smallest candidate that
    strictly reduces r's rectangle count and falling back to the smallest
    feasible candidate otherwise. Returns True if the map changed."""
    mask = lab == r
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    hole = np.zeros_like(mask)
    hole[y0:y1, x0:x1] = ~mask[y0:y1, x0:x1]
    if not hole.any():
        return False                                   # already equal to its bbox
    cc, n = ndimage.label(hole, structure=_CC)
    comps = []
    for c in range(1, n + 1):
        sel = cc == c
        comps.append((int(sel.sum()), int(np.flatnonzero(sel.reshape(-1))[0]), sel))
    comps.sort(key=lambda t: (t[0], t[1]))             # smallest notch first
    before = len(mask_to_rects(mask))
    fallback = None
    for _, _, sel in comps:
        if not _feasible(lab, r, sel):
            continue
        if fallback is None:
            fallback = sel
        if len(mask_to_rects(mask | sel)) < before:
            lab[sel] = np.int16(r)
            return True
    if fallback is None:
        return False
    lab[fallback] = np.int16(r)
    return True


def _halo(mask):
    """4-neighbourhood of a boolean mask (the mask itself is not excluded)."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _shed_one_strip(lab, r):
    """Second move (ruling D3): give away region r's SMALLEST maximal strip.

    mask_to_rects decomposes r into maximal horizontal strips merged
    vertically, so every rect is a set of whole row runs; deleting one deletes
    exactly those runs and leaves every other rect's decomposition untouched,
    i.e. r's rect count falls by exactly one whenever this fires. The strip
    goes to the majority label among its own 4-neighbours outside r (ties ->
    lowest id, np.argmax's rule), and the move is accepted only if r and the
    receiver both stay non-empty and 4-connected.

    This is feasible where absorption is not: _fill_one_notch needs a bbox hole
    whose EVERY donor survives losing it, and on a 16-region 64^2 map almost
    every hole straddles several donors; shedding needs one receiver. Mutates
    `lab` in place and returns whether it fired.
    """
    mask = lab == r
    rects = mask_to_rects(mask)
    if len(rects) <= 1:
        return False
    area = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    order = sorted(range(len(rects)),
                   key=lambda i: (int(area[i]), int(rects[i, 1]),
                                  int(rects[i, 0])))
    for i in order:
        x0, y0, x1, y1 = (int(v) for v in rects[i])
        sel = np.zeros_like(mask)
        sel[y0:y1, x0:x1] = True
        nb = lab[_halo(sel) & ~sel]
        nb = nb[nb != r]
        if not nb.size:
            continue
        d = int(np.bincount(nb).argmax())
        if not _connected(mask & ~sel):
            continue
        if not _connected((lab == d) | sel):
            continue
        lab[sel] = np.int16(d)
        return True
    return False


def enforce_rect_max(labels, k, rect_max=8):
    """spec section 2's hard rect cap, on the region with the most rectangles:
    absorption (_fill_one_notch) first, then shedding (_shed_one_strip).

    Termination (ruling D4): there is NO monotone potential here -- a donor
    that extends beyond r's bbox keeps its own bbox while losing bins to r, so
    sum_r |bbox(r) \\ r| can stay flat, and the arg-max target switches between
    regions. What makes the loop safe is the k*B^2 pass budget below: every
    pass strictly grows (absorption) or strictly shrinks (shedding) the current
    target region, and the budget bound is the guard.

    On exhaustion this raises; run_region_producer catches that and re-runs the
    whole extraction at --extract-bins 32 before letting it escape (ruling D3).
    """
    lab = np.asarray(labels, dtype=np.int16).copy()
    budget = k * lab.size + 1
    for _ in range(budget):
        counts = region_rect_counts(lab, k)
        worst = int(np.argmax(counts))
        if counts[worst] <= rect_max:
            return lab
        if not _fill_one_notch(lab, worst) and not _shed_one_strip(lab, worst):
            raise RuntimeError(
                f"region {worst} has {counts[worst]} rects (> {rect_max}) and "
                "neither a feasible notch fill nor a feasible shed remains; "
                "lower --extract-bins or K")
    # Budget exhausted (ruling D4): name the region and count the arg-max
    # target had on this final pass so Task 10's --extract-bins 32 fallback
    # can log which region forced the retry, same as the in-loop RuntimeError
    # above.
    counts = region_rect_counts(lab, k)
    worst = int(np.argmax(counts))
    raise RuntimeError(
        f"enforce_rect_max did not converge within {budget} passes: region "
        f"{worst} still has {counts[worst]} rects (> {rect_max}); the arg-max "
        "target kept switching between regions (ruling D4 -- no monotone "
        "potential excludes a cycle)")


def rects_to_regionset(labels, k, die, lattice=512, name_fmt="P{}"):
    """Bin-coordinate rectangles -> a RegionSet on `lattice`. Requires
    lattice % bins == 0 so every edge lands exactly on the lattice (64 | 512 and
    32 | 512, spec section 2), which is what makes validate() pass."""
    lab = np.asarray(labels)
    b = lab.shape[0]
    assert lab.shape[0] == lab.shape[1], "label map must be square"
    assert lattice % b == 0, \
        f"lattice {lattice} must be a multiple of the bin count {b}"
    xl, yl, xh, yh = (float(v) for v in die)
    cw, ch = (xh - xl) / b, (yh - yl) / b
    regions = []
    for kk in range(k):
        r = mask_to_rects(lab == kk).astype(np.float64)
        assert len(r), f"region {kk} is empty; cannot build a RegionSpec"
        regions.append(RegionSpec(name_fmt.format(kk), np.stack([
            xl + r[:, 0] * cw, yl + r[:, 1] * ch,
            xl + r[:, 2] * cw, yl + r[:, 3] * ch], axis=1)))
    rs = RegionSet(die=(xl, yl, xh, yh), lattice=int(lattice), regions=regions)
    rs.validate()
    return rs
