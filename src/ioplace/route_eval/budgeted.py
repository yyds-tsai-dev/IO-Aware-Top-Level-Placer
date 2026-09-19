"""Geometry proposals that trade wirelength for fewer region crossings.

Pin positions and MST connectivity stay fixed. Search both L orientations and
three-segment H-V-H / V-H-V paths on region-band center tracks, including
tracks outside the pin bounding box. Each edge obeys its own Manhattan-length
budget, hence the sum of edge lengths obeys the same net/design budget.

This is an obstacle-free routing estimator, not detailed routing: it does not
model tracks, layers, capacity or DRC. Crossings and lengths count MST branch
multiplicity, matching the legacy evaluator; shared geometry is not deduped.
The candidate family is bounded, so no global optimality claim is made.
"""
import math
import numpy as np


class BudgetedRouter:
    def __init__(self, region_grid, *, resources=None):
        self.resources = resources
        if resources is not None and resources.rg is not region_grid:
            raise ValueError("resources must use the same RegionGrid instance")
        self.rg = region_grid
        g = region_grid.grid
        self.h = np.pad(np.cumsum(g[:, 1:] != g[:, :-1], axis=1), ((0, 0), (1, 0)))
        self.v = np.pad(np.cumsum(g[1:, :] != g[:-1, :], axis=0), ((1, 0), (0, 0)))
        # Interior cell-center tracks on both sides of each region-band boundary.
        xc = np.r_[0, np.flatnonzero(np.any(g[:, 1:] != g[:, :-1], axis=0)) + 1, g.shape[1]]
        yc = np.r_[0, np.flatnonzero(np.any(g[1:, :] != g[:-1, :], axis=1)) + 1, g.shape[0]]
        xl, yl, _, _ = region_grid.die
        self.x_tracks = xl + (xc[:-1] + .5) * region_grid.cell_w
        self.y_tracks = yl + (yc[:-1] + .5) * region_grid.cell_h
        # Both sides of each boundary matter when the detour allowance is tight.
        self.x_tracks = np.unique(np.r_[self.x_tracks, xl + (xc[1:] - .5) * region_grid.cell_w])
        self.y_tracks = np.unique(np.r_[self.y_tracks, yl + (yc[1:] - .5) * region_grid.cell_h])

    def _h(self, x0, x1, y):
        a, row = self.rg.to_idx(x0, y)
        b, _ = self.rg.to_idx(x1, y)
        return np.abs(self.h[row, b] - self.h[row, a])

    def _v(self, y0, y1, x):
        col, a = self.rg.to_idx(x, y0)
        _, b = self.rg.to_idx(x, y1)
        return np.abs(self.v[b, col] - self.v[a, col])

    def route_edges(self, starts, ends, *, wirelength_budget=.05):
        """Return explicit connected polylines, costs and unchanged endpoints.

        Inputs are (N,2) physical coordinates in the closed die. Call in chunks
        for large netlists; memory is O(N + lattice area), independent of tracks.
        Zero budget still permits a better shortest L/Z route.
        """
        if not math.isfinite(wirelength_budget) or wirelength_budget < 0:
            raise ValueError("wirelength_budget must be finite and nonnegative")
        a, b = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape:
            raise ValueError("starts and ends must have matching (N,2) shapes")
        xl, yl, xh, yh = self.rg.die
        for points in (a, b):
            if (not np.isfinite(points).all() or (points < [xl, yl]).any()
                    or (points > [xh, yh]).any()):
                raise ValueError("route endpoints must be finite and inside the die")
        x0, y0 = a.T
        x1, y1 = b.T
        length = np.abs(b - a).sum(axis=1)
        base_io = self._h(x0, x1, y0) + self._v(y0, y1, x1)
        best_io, best_length = base_io.copy(), length.copy()
        points = np.stack([a, np.column_stack([x1, y0]),
                           np.column_stack([x1, y0]), b], axis=1)
        if self.resources is not None:
            resource_cost, feasible = self.resources.evaluate(points)
            best_score = np.where(feasible, base_io + resource_cost, np.inf)
        else:
            best_score = best_io.astype(float)
        # Zero crossings at Manhattan length is already a lower bound. Avoid
        # searching these overwhelmingly common local edges in large designs.
        active = np.arange(len(a)) if self.resources is not None else np.flatnonzero(base_io)
        if len(active) != len(a):
            if len(active):
                sub = self.route_edges(a[active], b[active], wirelength_budget=wirelength_budget)
                points[active] = sub["points"]
                best_io[active] = sub["crossings"]
                best_length[active] = sub["wirelength"]
            return {"points": points, "baseline_crossings": base_io,
                    "crossings": best_io, "baseline_wirelength": length,
                    "wirelength": best_length, "wirelength_budget": wirelength_budget}

        def consider(track, vertical_first):
            nonlocal best_io, best_length, best_score
            if vertical_first:
                mid = np.broadcast_to(track, y0.shape)
                cost = np.abs(y0-mid) + np.abs(x0-x1) + np.abs(y1-mid)
                io = self._v(y0, mid, x0) + self._h(x0, x1, mid) + self._v(mid, y1, x1)
                p, q = np.column_stack([x0, mid]), np.column_stack([x1, mid])
            else:
                mid = np.broadcast_to(track, x0.shape)
                cost = np.abs(x0-mid) + np.abs(y0-y1) + np.abs(x1-mid)
                io = self._h(x0, mid, y0) + self._v(y0, y1, mid) + self._h(mid, x1, y1)
                p, q = np.column_stack([mid, y0]), np.column_stack([mid, y1])
            candidate = np.stack([a, p, q, b], axis=1)
            if self.resources is not None:
                penalty, feasible = self.resources.evaluate(candidate)
                score = np.where(feasible, io + penalty, np.inf)
            else:
                score = io.astype(float)
            use = ((cost <= length * (1 + wirelength_budget)) & np.isfinite(score)
                   & ((score < best_score) | ((score == best_score) & (cost < best_length))))
            best_score[use] = score[use]
            best_io[use], best_length[use] = io[use], cost[use]
            points[use, 1], points[use, 2] = p[use], q[use]

        consider(y1, True)  # vertical-first L, often improves IO at zero WL cost
        x_tracks, y_tracks = self.x_tracks, self.y_tracks
        if self.resources is not None:
            # Obstacles can create useful tracks unrelated to region bands.
            x_tracks = xl + (np.arange(self.rg.grid.shape[1])+.5)*self.rg.cell_w
            y_tracks = yl + (np.arange(self.rg.grid.shape[0])+.5)*self.rg.cell_h
        for track in x_tracks:
            consider(track, False)
        for track in y_tracks:
            consider(track, True)
        if not np.isfinite(best_score).all():
            raise ValueError("no resource-feasible L/Z candidate within wirelength budget")
        return {"points": points, "baseline_crossings": base_io,
                "crossings": best_io, "baseline_wirelength": length,
                "wirelength": best_length, "wirelength_budget": wirelength_budget}


def route_net(router, pin_x, pin_y, *, wirelength_budget=.05, topology="mst",
              shared_branches=False, coordinate_scale=1000.):
    """MST (legacy default) or FLUTE with opt-in per-net shared-segment union.

    FLUTE always reports union metrics. Branch-wise fields remain explicit for
    comparisons; use result['union'] for deduplicated IO and wirelength.
    """
    from ioplace.evaluator_ref import net_mst_edges
    px, py = np.asarray(pin_x), np.asarray(pin_y)
    pins = np.column_stack([px, py])
    if topology == "mst":
        edges = net_mst_edges(px, py)
        starts, ends = pins[edges[:, 0]], pins[edges[:, 1]]
        provenance = {"topology": "mst", "mst_edges": edges}
    elif topology == "flute":
        from ioplace.route_eval.topology import flute_edges
        geometry, provenance = flute_edges(pins, coordinate_scale=coordinate_scale)
        starts, ends = geometry[:, 0], geometry[:, 1]
    else:
        raise ValueError("topology must be mst or flute")
    result = router.route_edges(starts, ends, wirelength_budget=wirelength_budget)
    result.update(provenance)
    if shared_branches or topology == "flute":
        from ioplace.route_eval.topology import union_metrics
        result["union"] = union_metrics(router.rg, result["points"])
        baseline = np.stack([starts, np.column_stack([ends[:, 0], starts[:, 1]]), ends], axis=1)
        result["baseline_union"] = union_metrics(router.rg, baseline)
        before, after = result["baseline_union"], result["union"]
        if router.resources is not None:
            raise ValueError("shared-tree resource accounting requires joint demand updates; use route_edges")
        if (after["wirelength"] > before["wirelength"]*(1+wirelength_budget)
                or (after["crossings"], after["wirelength"]) > (before["crossings"], before["wirelength"])):
            result["points"] = np.stack([starts, baseline[:, 1], baseline[:, 1], ends], axis=1)
            result["crossings"] = result["baseline_crossings"].copy()
            result["wirelength"] = result["baseline_wirelength"].copy()
            result["union"] = before
            result["union_budget_fallback"] = True
        else:
            result["union_budget_fallback"] = False
    result["hpwl"] = float(np.ptp(px) + np.ptp(py)) if len(px) else 0.
    return result
