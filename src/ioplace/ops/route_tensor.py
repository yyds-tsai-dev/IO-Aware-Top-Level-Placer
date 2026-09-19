"""Sparse, device-resident physical route unions and resource crossings.

Axes are 0 for horizontal (track y, span x), 1 for vertical. Sorting, exact
track grouping, interval membership, and sparse support are discrete decisions.
Autograd differentiates the selected smooth piece. At coincident tracks the
grouping is frozen: independently separating equal tracks is discontinuous;
moving their shared coordinate together has the usual union derivative.
No routing, host coordinate copies, or segment-by-resource dense matrix occurs.
"""

import math

import torch


def union_intervals(net_ids, axes, tracks, low, high):
    """Exact collinear per-net union; normalize endpoints and drop zero lengths.

    All arguments are one-dimensional same-device tensors. IDs/axes are int64;
    coordinates share a floating dtype. Output has the same five named fields,
    ordered by (net, axis, track, low). At endpoint ties one representative is
    selected, so gradients are interpreted on a fixed sorting/grouping piece.
    """
    lo, hi = torch.minimum(low, high), torch.maximum(low, high)
    keep = hi.detach() > lo.detach()
    net_ids, axes, tracks, lo, hi = (v[keep] for v in (net_ids, axes, tracks, lo, hi))
    n = lo.numel()
    if not n:
        return dict(net_ids=net_ids, axes=axes, tracks=tracks, low=lo, high=hi)
    order = torch.arange(n, device=lo.device)
    for key in (lo, tracks, axes, net_ids):
        order = order[torch.argsort(key.detach()[order], stable=True)]
    net_ids, axes, tracks, lo, hi = (v[order] for v in (net_ids, axes, tracks, lo, hi))
    group_start = torch.ones(n, device=lo.device, dtype=torch.bool)
    group_start[1:] = ((net_ids[1:] != net_ids[:-1]) | (axes[1:] != axes[:-1])
                       | (tracks.detach()[1:] != tracks.detach()[:-1]))
    group = group_start.long().cumsum(0) - 1
    # Integer rank encodes prefix maxima without adding large float offsets to
    # coordinates. Gather original highs to retain the selected derivative.
    rank_order = torch.argsort(hi.detach(), stable=True)
    rank = torch.empty_like(rank_order)
    rank[rank_order] = torch.arange(n, device=lo.device)
    prefix_index = torch.cummax(group * (n + 1) + rank, 0).indices
    prefix_hi = hi[prefix_index]
    starts = group_start.clone()
    starts[1:] |= lo.detach()[1:] > prefix_hi.detach()[:-1]
    ends = torch.ones_like(starts)
    ends[:-1] = starts[1:]
    return dict(net_ids=net_ids[starts], axes=axes[starts], tracks=tracks[starts],
                low=lo[starts], high=prefix_hi[ends])


def _smoothstep(t):
    """Quintic compact C2 step; clamping also bounds its polynomial numerics."""
    t = t.clamp(-1, 1)
    return .5 + t * (15 / 16 + t.square() * (-5 / 8 + t.square() * (3 / 16)))


def grid_crossings(union, x_edges, y_edges, tau, *, hard=False):
    """Sparse crossing incidences with physical union-segment identities.

    Resource IDs match ``ResourceGrid.edge_keys``: horizontal rows first, then
    vertical rows. Smooth values are squared longitudinal compact windows times
    a transverse compact window. Outer transverse bins extend to infinity.
    ``hard=True`` uses low < cut <= high and lower <= track < upper exactly,
    returning detached 0/1 occupancy for discrete evaluation.

    Support enumeration uses detached searchsorted and repeat_interleave, with
    storage proportional to segments plus actual nearby incidences. Its changing
    support is differentiable because smooth values and derivatives vanish at
    the support boundary. Grid boundaries and finite positive scalar tau are
    fixed inputs. A zero-dimensional tensor tau is also accepted; validating a
    CUDA scalar synchronizes, so a Python float avoids that synchronization.
    """
    if getattr(tau, "ndim", 0) != 0:
        raise ValueError("tau must be a finite positive scalar")
    finite = torch.isfinite(tau) if torch.is_tensor(tau) else math.isfinite(tau)
    if not finite or tau <= 0:
        raise ValueError("tau must be a finite positive scalar")
    nx, ny = x_edges.numel() - 1, y_edges.numel() - 1
    horizontal_count = ny * (nx - 1)
    out = {key: [] for key in ("net_ids", "edge_ids", "values", "segment_ids")}
    margin = 0. if hard else tau
    for axis, span_edges, track_edges in ((0, x_edges, y_edges), (1, y_edges, x_edges)):
        cuts = span_edges[1:-1]
        if not cuts.numel():
            continue
        segments = torch.nonzero(union["axes"] == axis, as_tuple=True)[0]
        low, high, track = (union[k][segments] for k in ("low", "high", "tracks"))
        # Hard lower-open/upper-closed intervals include an exact upper cut.
        cut_begin = torch.searchsorted(cuts, (low - margin).detach().contiguous(), right=True)
        cut_end = torch.searchsorted(cuts, (high + margin).detach().contiguous(), right=hard)
        internal_tracks = track_edges[1:-1]
        bin_begin = torch.searchsorted(internal_tracks, (track - margin).detach().contiguous(), right=True)
        bin_end = torch.searchsorted(internal_tracks, (track + margin).detach().contiguous(), right=hard) + 1
        cut_count, bin_count = cut_end - cut_begin, bin_end - bin_begin
        counts = cut_count * bin_count
        local = torch.repeat_interleave(torch.arange(segments.numel(), device=low.device), counts)
        offsets = torch.arange(local.numel(), device=low.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        cut_id = cut_begin[local] + offsets.remainder(cut_count[local])
        bin_id = bin_begin[local] + torch.div(offsets, cut_count[local], rounding_mode="floor")
        if axis == 0:
            edge_ids = bin_id * (nx - 1) + cut_id
        else:
            edge_ids = horizontal_count + cut_id * nx + bin_id
        if hard:
            values = torch.ones(local.numel(), device=low.device, dtype=low.dtype)
        else:
            longitudinal = (_smoothstep((cuts[cut_id] - low[local]) / tau)
                            - _smoothstep((cuts[cut_id] - high[local]) / tau))
            lower = _smoothstep((track[local] - track_edges[bin_id]) / tau)
            upper = _smoothstep((track[local] - track_edges[bin_id + 1]) / tau)
            lower = torch.where(bin_id == 0, torch.ones_like(lower), lower)
            upper = torch.where(bin_id == track_edges.numel() - 2, torch.zeros_like(upper), upper)
            values = longitudinal.square() * (lower - upper)
        out["net_ids"].append(union["net_ids"][segments[local]])
        out["edge_ids"].append(edge_ids)
        out["values"].append(values)
        out["segment_ids"].append(segments[local])
    return {key: torch.cat(parts) if parts else union["low" if key == "values" else "net_ids"][:0]
            for key, parts in out.items()}


def joint_demand(crossings, edge_count, background=None):
    """Max occupancy within each net/resource, sum across nets, add fixed usage."""
    values = crossings["values"]
    demand = values.new_zeros(edge_count) if background is None else background.clone()
    # Preserve a zero derivative for empty physical routes.
    demand = demand + values.sum() * 0
    if not values.numel():
        return demand
    keys = crossings["net_ids"] * edge_count + crossings["edge_ids"]
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    occupancy = values.new_zeros(unique.numel()).scatter_reduce(0, inverse, values, reduce="amax", include_self=False)
    return demand.scatter_add(0, unique.remainder(edge_count), occupancy)


def congestion_cost(demand, capacity):
    """Summed joint congestion penalty, matching the hard resource evaluator."""
    safe = capacity.clamp_min(1)
    return ((demand / safe).square() + 4 * (torch.relu(demand - capacity) / safe).square()).sum()
