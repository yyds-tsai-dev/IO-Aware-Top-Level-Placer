"""Phase-2 freeze: cell-centre membership, churn, and the stop criterion
(design v2 sec 3 phase 2).

Freeze when all three hold:
  * `overflow <= 0.15`
  * `tau_rel <= 0.05` (the FT full-ramp point, schedules.py:125)
  * membership churn over the last 50 iterations `<= 0.5%`

Membership is the argmax region of the **cell centre**. `region_sdf_l1`
returns a signed distance -- negative inside the owning rect
(soft_assign.py:27's `max(dx,dy).clamp(max=0)`), non-negative outside -- so
`argmax_k(-SDF_k)` is the containing region independently of tau, and this
module needs only the first of `softmax_stats`' two passes
(soft_assign.py:40-58).
"""
from collections import deque

import numpy as np

from ioplace.artifacts import FREEZE_SCHEMA_VERSION
from ioplace.ops.soft_assign import region_sdf_l1


def _chunk_bounds(K, chunk):
    if chunk is None or chunk >= K:
        return [(0, K)]
    return [(lo, min(lo + chunk, K)) for lo in range(0, K, chunk)]


def argmax_region(x, y, rects, rect2region, K, chunk=None):
    """(N,) int64 owning-region index, chunked over regions like softmax_stats."""
    import torch
    best = torch.full((x.shape[0],), -float("inf"), dtype=x.dtype, device=x.device)
    out = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    for lo, hi in _chunk_bounds(K, chunk):
        z = -region_sdf_l1(x, y, rects, rect2region, lo, hi)
        cm, ci = z.max(dim=1)
        upd = cm > best                       # strict > keeps first-occurrence ties
        best = torch.where(upd, cm, best)
        out = torch.where(upd, ci + lo, out)
    return out


def cell_centers(x, y, size_x, size_y):
    return x + 0.5 * size_x, y + 0.5 * size_y


class FreezeMonitor:
    """Tracks membership churn over a fixed iteration window and evaluates the
    three-part freeze criterion. Observation cadence is the caller's
    (the driver's `--every` evaluator-gated callback); churn is always taken
    against the newest retained sample at least `window` iterations old, so a
    cadence change cannot silently change the meaning of the number."""

    def __init__(self, *, window=50, overflow_max=0.15, tau_rel_max=0.05,
                 churn_max=0.005):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self.overflow_max = float(overflow_max)
        self.tau_rel_max = float(tau_rel_max)
        self.churn_max = float(churn_max)
        self.churn = None
        self.last_iteration = None
        self._samples = deque()               # (iteration, argmax int64 numpy)

    def observe(self, iteration, argmax):
        arr = np.asarray(argmax.cpu() if hasattr(argmax, "cpu") else argmax,
                         dtype=np.int64)
        reference = None
        for it, part in self._samples:
            if it <= iteration - self.window:
                reference = (it, part)        # newest sample a full window back
        self.churn = (float(np.mean(arr != reference[1]))
                      if reference is not None else None)
        self._samples.append((int(iteration), arr))
        # keep only what a future comparison can still need: the newest sample
        # older than the window, plus everything after it.
        while len(self._samples) > 2 and self._samples[1][0] <= iteration - self.window:
            self._samples.popleft()
        self.last_iteration = int(iteration)
        return self.churn

    def reasons(self, overflow, tau_rel):
        return {"overflow_ok": bool(overflow <= self.overflow_max),
                "tau_ok": bool(tau_rel <= self.tau_rel_max),
                "churn_ok": bool(self.churn is not None
                                 and self.churn <= self.churn_max)}

    def should_freeze(self, overflow, tau_rel):
        return all(self.reasons(overflow, tau_rel).values())


def ensure_nonempty_regions(part, k, cx, cy, centers):
    """Guarantee every region owns at least one cell.

    An empty *real* region makes `PlaceDB.calc_num_filler_for_fence_region`
    take `np.percentile` of an empty movable-size slice (PlaceDB.py:687).
    Under the installed numpy (1.26.4) that raises `IndexError: index -1 is out
    of bounds for axis 0 with size 0` on the spot, inside `initialize()` --
    verified on this host 2026-09-19 (pre-flight E-3). It never reaches the
    `int(round(nan))` at PlaceDB.py:728, so do not go looking for a
    `ValueError`. The escape-cell workaround
    (run_placement_two_stage.py:192-233) only covers the *implicit* bucket, so
    the freeze must repair real regions itself: move the single cell whose
    centre is closest (L1) to the empty region's centre, preferring cells from
    regions that own more than one.
    """
    part = np.asarray(part, dtype=np.int32).copy()
    centers = np.asarray(centers, dtype=np.float64)
    moves = []
    for region in range(int(k)):
        counts = np.bincount(part, minlength=int(k))
        if counts[region] > 0:
            continue
        donor_ok = counts[part] >= 2
        pool = np.flatnonzero(donor_ok) if donor_ok.any() else np.arange(len(part))
        distance = (np.abs(cx[pool] - centers[region, 0])
                    + np.abs(cy[pool] - centers[region, 1]))
        cell = int(pool[int(np.argmin(distance))])
        moves.append({"region": region, "cell": cell, "from": int(part[cell]),
                      "distance": float(distance.min())})
        part[cell] = region
    return part, moves


def region_cell_stats(part, size_x, size_y, rs):
    """The four per-region arrays `freeze.json` carries.

    This is the **single** implementation of the per-region count / cell-area /
    region-area / utilisation arithmetic in the v2 flow:
    `main_flow_metrics.region_area_balance` (Task 6) calls it and only adds the
    max/min/ratio/deviation summaries that belong to `result.json` (pre-flight
    amendment D-2). It lives here rather than there because Task 3 lands first
    and `main_flow_metrics` must stay importable without torch -- this module
    pulls in `ops/soft_assign`, which imports torch -- so Task 6 takes it as a
    local import inside the function.

    `region_cell_area` and `region_area` are **not** scale-invariant; only
    `region_utilization` is (both of its terms carry `scale_factor**2`). Every
    caller must therefore pass sizes and a `RegionSet` in the *same* frame, and
    both call sites in `run_main_flow` pass **native-unit** sizes with the
    native `RegionSet`, because `freeze.json` is a native-unit artefact and
    `result.json` must quote the same numbers.
    """
    part = np.asarray(part, dtype=np.int64)
    area = np.asarray(size_x, dtype=np.float64) * np.asarray(size_y, dtype=np.float64)
    counts = np.bincount(part, minlength=rs.k)[:rs.k]
    cell_area = np.bincount(part, weights=area, minlength=rs.k)[:rs.k]
    region_area = np.empty(rs.k, dtype=np.float64)
    for rid, region in enumerate(rs.regions):
        rects = np.asarray(region.rects, dtype=np.float64).reshape(-1, 4)
        region_area[rid] = float(np.sum((rects[:, 2] - rects[:, 0])
                                        * (rects[:, 3] - rects[:, 1])))
    return {"region_cell_count": counts.astype(int).tolist(),
            "region_cell_area": cell_area.tolist(),
            "region_area": region_area.tolist(),
            "region_utilization": (cell_area / region_area).tolist()}


def freeze_record(*, iteration, reason, overflow, tau, tau_rel, churn, k,
                  io_soft, membership_npz, soft_npz, repaired_empty_regions,
                  gp_iterations_soft, density_weight_soft, stats):
    record = {"schema_version": FREEZE_SCHEMA_VERSION,
              "iteration": int(iteration), "reason": str(reason),
              "overflow": float(overflow), "tau": float(tau),
              "tau_rel": float(tau_rel),
              "churn": None if churn is None else float(churn),
              "k": int(k), "io_soft": int(io_soft),
              "membership_npz": str(membership_npz), "soft_npz": str(soft_npz),
              "repaired_empty_regions": list(repaired_empty_regions),
              "gp_iterations_soft": int(gp_iterations_soft),
              "density_weight_soft": float(density_weight_soft)}
    record.update(stats)
    return record
