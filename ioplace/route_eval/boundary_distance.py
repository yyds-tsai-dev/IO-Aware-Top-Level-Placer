"""Length-weighted proximity to internal boundaries of a rectangular grid.

Distance uses lattice cells perpendicular to each boundary. It is not the
run-length threshold used by the crossing evaluator. WIRE segment multiplicity
is retained, consistently with route_wl; this is not a metal-union measure.
"""
import numpy as np

DEFAULT_DISTANCES = (0., .25, .5, 1., 2., 4., 8., 16., 32., 64., 128.)


def full_grid_cuts(rg):
    vertical = rg.grid[:, 1:] != rg.grid[:, :-1]
    horizontal = rg.grid[1:, :] != rg.grid[:-1, :]
    vc, hr = np.any(vertical, axis=0), np.any(horizontal, axis=1)
    if not np.array_equal(vertical, np.broadcast_to(vc, vertical.shape)) or not np.array_equal(
            horizontal, np.broadcast_to(hr[:, None], horizontal.shape)):
        raise ValueError("distance diagnostic requires full rectangular grid cuts")
    return (rg.die[0] + (np.flatnonzero(vc)+1)*rg.cell_w,
            rg.die[1] + (np.flatnonzero(hr)+1)*rg.cell_h)


def band_overlap(lo, hi, cuts, radius):
    """Length of each interval intersecting the UNION of closed cut bands."""
    merged = []
    for cut in cuts:
        a, b = cut-radius, cut+radius
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    result = np.zeros(len(lo), dtype=np.float64)
    for a, b in merged:
        result += np.maximum(0., np.minimum(hi, b)-np.maximum(lo, a))
    return result


def wire_boundary_distance(segments, rg, coord_map, net_mask=None,
                           distances=DEFAULT_DISTANCES, chunk_size=250_000):
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all()
            or np.any(distances < 0) or np.any(np.diff(distances) <= 0) or chunk_size <= 0):
        raise ValueError("finite increasing nonnegative distances and positive chunk size required")
    scale = float(coord_map.scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite CoordMap scale required")
    cuts_x, cuts_y = full_grid_cuts(rg)
    sx, sy = coord_map.shift_factor
    cuts_x, cuts_y = cuts_x/scale+sx, cuts_y/scale+sy
    xl, yl, xh, yh = rg.die
    xl, xh, yl, yh = xl/scale+sx, xh/scale+sx, yl/scale+sy, yh/scale+sy
    cell_w, cell_h = rg.cell_w/scale, rg.cell_h/scale
    if net_mask is not None:
        net_mask = np.asarray(net_mask, dtype=bool)
        if net_mask.shape != (segments.num_nets,):
            raise ValueError("net mask must use segment net-index space")
    selected = segments.wire_mask()
    if net_mask is not None:
        selected &= net_mask[segments.seg_net_id]
    lengths = segments.segment_length()
    zero_count = int(np.count_nonzero(selected & (lengths == 0)))
    indices = np.flatnonzero(selected & (lengths > 0))
    any_length = np.zeros(len(distances)); parallel_length = np.zeros(len(distances))
    total = inside = horizontal_inside = vertical_inside = 0.
    for start in range(0, len(indices), chunk_size):
        ids = indices[start:start+chunk_size]
        x0, y0, x1, y1 = [np.asarray(v[ids], dtype=np.float64) for v in
            (segments.seg_x0, segments.seg_y0, segments.seg_x1, segments.seg_y1)]
        if not all(np.isfinite(v).all() for v in (x0,y0,x1,y1)):
            raise ValueError("nonfinite wire coordinates")
        h, v = y0 == y1, x0 == x1
        if not np.all(h | v):
            raise ValueError("non-Manhattan WIRE cannot enter distance diagnostic")
        total += float(lengths[ids].sum(dtype=np.float64))
        for horizontal, mask, axis0, axis1, fixed, amin, amax, fmin, fmax, parallel, transverse, pc, tc in (
                (True,h,x0,x1,y0,xl,xh,yl,yh,cuts_y,cuts_x,cell_h,cell_w),
                (False,v,y0,y1,x0,yl,yh,xl,xh,cuts_x,cuts_y,cell_w,cell_h)):
            mask = mask & (fixed >= fmin) & (fixed <= fmax)
            lo = np.maximum(np.minimum(axis0[mask],axis1[mask]), amin)
            hi = np.minimum(np.maximum(axis0[mask],axis1[mask]), amax)
            keep = hi > lo
            lo, hi, constant = lo[keep], hi[keep], fixed[mask][keep]
            span = hi-lo
            amount = float(span.sum()); inside += amount
            if horizontal: horizontal_inside += amount
            else: vertical_inside += amount
            nearest = np.full(len(span), np.inf)
            for cut in parallel:
                nearest = np.minimum(nearest, np.abs(constant-cut)/pc)
            for i, distance in enumerate(distances):
                along = nearest <= distance
                plen = float(span[along].sum())
                parallel_length[i] += plen
                any_length[i] += plen + float(band_overlap(lo[~along],hi[~along],transverse,distance*tc).sum())
    available = inside > 0
    def fraction(values):
        return np.clip(values/inside,0.,1.).tolist() if available else [None]*len(distances)
    return dict(available=available, distance_cells=distances.tolist(),
        distance_definition="min axis-normalized perpendicular distance to internal full-grid boundaries",
        weight="decoded WIRE segment length in DBU, retaining multiplicity, clipped to die",
        positive_wire_segments=len(indices), zero_length_wire_rows=zero_count,
        total_wire_length_dbu=total, inside_die_wire_length_dbu=inside,
        outside_die_wire_length_dbu=max(0.,total-inside),
        horizontal_inside_length_dbu=horizontal_inside, vertical_inside_length_dbu=vertical_inside,
        any_boundary_cdf=fraction(any_length), parallel_boundary_cdf=fraction(parallel_length),
        vertical_cut_count=len(cuts_x), horizontal_cut_count=len(cuts_y))
