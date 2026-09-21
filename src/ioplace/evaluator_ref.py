import numpy as np
from dataclasses import dataclass, field
from ioplace.netlist import pin_positions
from ioplace.region_graph import region_graph as build_region_graph, next_hop_table, steiner_tree_stats
from ioplace.region_segments import edge_segment_ids, segment_utilisation
from ioplace.straddle import straddle_diagnostics

@dataclass
class EvalResult:
    io_count: int
    ft_count: int
    tree_wl: float
    hpwl: float
    per_net_crossings: np.ndarray
    per_net_ft: np.ndarray
    boundary_pair_demand: dict
    large_net_lb: int
    hard_lambda_sum: int = 0
    per_net_lambda: np.ndarray = None
    # M3 T1 (docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md
    # §2/§2.5): region-graph (G_R) Steiner routing, as opposed to the MST-geometry
    # routing the legacy io_count/ft_count fields above are computed from.
    # io_rg = Σ_e ST_e ; ft_rg = io_rg - hard_lambda_sum (== Σ_e FT_e via RG).
    io_rg: int = 0
    ft_rg: int = 0
    per_net_steiner: np.ndarray = None   # (E,) int32, ST_e
    per_net_home: np.ndarray = None      # (E,) uint8, argmax-pin-count region (ties -> smallest id)
    # v2 P-F (design v2 sec 7 diagnostics 1-3). Defaults are the "not computed"
    # state, which evaluate(straddle=False) and any pre-P-F caller both land on.
    straddle_cells: int = 0
    straddle_area_fraction: float = 0.0
    straddle_pin_split_nets: int = 0
    straddle_out_area: float = 0.0
    straddle_movable_area: float = 0.0
    straddle_wide_cells: int = 0
    per_node_straddle: np.ndarray = None   # (num_physical,) uint8
    per_net_pin_split: np.ndarray = None   # (num_nets,) int32, signed
    # v2 P-D (design sec 5): the per-segment hard check. All None unless
    # `evaluate(..., segments=...)` was asked for, so "not measured" and
    # "measured as zero" stay distinguishable.
    segment_demand: np.ndarray = None      # (S,) int64
    segment_capacity: np.ndarray = None    # (S,) float64
    segment_util: np.ndarray = None        # (S,) float64, inf where C == 0 < D
    num_over_capacity: int = None
    num_zero_capacity_segments: int = None
    zero_capacity_demand: int = None
    max_util: float = None
    p99_util: float = None
    # capped-candidate source arrays, ascending by ((net*K+u)*K+v)*S+seg --
    # the ordering that makes ref/GPU parity bit-exact.
    cand_net: np.ndarray = None
    cand_u: np.ndarray = None
    cand_v: np.ndarray = None
    cand_seg: np.ndarray = None
    cand_count: np.ndarray = None
    cand_dropped: int = None               # crossings on legs with u == v

def net_mst_edges(px, py):
    d = len(px)
    if d <= 1:
        return np.empty((0, 2), dtype=np.int32)
    if d == 2:
        return np.array([[0, 1]], dtype=np.int32)
    dist = np.abs(px[:, None] - px[None, :]) + np.abs(py[:, None] - py[None, :])
    in_tree = np.zeros(d, dtype=bool)
    in_tree[0] = True
    best_cost = dist[0].copy()
    best_from = np.zeros(d, dtype=np.int32)
    edges = np.empty((d - 1, 2), dtype=np.int32)
    for t in range(d - 1):
        masked = np.where(in_tree, np.inf, best_cost)
        j = int(np.argmin(masked))
        edges[t] = (best_from[j], j)
        in_tree[j] = True
        upd = dist[j] < best_cost
        best_cost = np.where(upd, dist[j], best_cost)
        best_from = np.where(upd, j, best_from)
    return edges

def net_mst_length(px, py):
    e = net_mst_edges(px, py)
    if len(e) == 0:
        return 0.0
    return float(np.sum(np.abs(px[e[:, 0]] - px[e[:, 1]]) +
                        np.abs(py[e[:, 0]] - py[e[:, 1]])))

def _walk_segment(rg, x0, y0, x1, y1, regions, pairs):
    """沿軸對齊線段在 lattice 上走,累計 region 集合與 crossing;回傳 crossing 數。"""
    ids0 = rg.region_of_points(np.array([x0]), np.array([y0]))[0]
    if x0 == x1 and y0 == y1:
        regions.add(int(ids0))
        return 0
    if y0 == y1:  # 水平
        ix0, iy = rg.to_idx(np.array([min(x0, x1)]), np.array([y0]))
        ix1, _ = rg.to_idx(np.array([max(x0, x1)]), np.array([y0]))
        row = rg.grid[iy[0], ix0[0]:ix1[0] + 1]
    else:         # 垂直
        ix, iy0 = rg.to_idx(np.array([x0]), np.array([min(y0, y1)]))
        _, iy1 = rg.to_idx(np.array([x0]), np.array([max(y0, y1)]))
        row = rg.grid[iy0[0]:iy1[0] + 1, ix[0]]
    regions.update(np.unique(row).tolist())
    diff_pos = np.nonzero(np.diff(row))[0]
    for p in diff_pos:
        a, b = int(row[p]), int(row[p + 1])
        pairs.append((min(a, b), max(a, b)))
    return len(diff_pos)

def edge_regions_and_crossings(rg, x0, y0, x1, y1):
    regions, pairs = set(), []
    n1 = _walk_segment(rg, x0, y0, x1, y0, regions, pairs)   # 水平段
    n2 = _walk_segment(rg, x1, y0, x1, y1, regions, pairs)   # 垂直段
    return regions, n1 + n2, pairs

def evaluate(nl, node_x, node_y, rg, max_degree=256, *, route_wirelength_budget=None,
             straddle=True, segments=None, segment_capacity=None,
             capacity_candidates=False):
    """Evaluate legacy MST L-routes or opt into budgeted detour geometry.

    A budget of .05 allows +5% per MST branch; pin HPWL and connectivity stay
    fixed. The opt-in mode recomputes crossings, FT and pair demand from the
    selected segments. It requires in-die pins and models no routing obstacles.
    Nets above max_degree retain the legacy lower-bound treatment.

    straddle=False skips the sec 7 diagnostics; the legacy fields are bit-identical either way.

    v2 P-D (design sec 5): with `segments` (a region_segments.SegmentTable
    built from this same `rg`) every unit crossing is additionally mapped to
    its segment id and accumulated into `segment_demand`; with
    `segment_capacity` the utilisation fields are filled from the shared
    `region_segments.segment_utilisation` helper; with `capacity_candidates`
    the per-(net, demand pair, segment) crossing counts the capacity term's
    candidate refresh consumes are returned too.
    """
    if segments is not None:
        if (segments.k != rg.k or segments.grid_shape() != rg.grid.shape
                or segments.die != tuple(float(v) for v in rg.die)):
            raise ValueError("segment table was built for a different region grid")
    elif segment_capacity is not None or capacity_candidates:
        raise ValueError("segment_capacity/capacity_candidates need segments=")
    router = None
    if route_wirelength_budget is not None:
        from ioplace.route_eval.budgeted import BudgetedRouter
        router = BudgetedRouter(rg)
        router.route_edges(np.empty((0, 2)), np.empty((0, 2)),
                           wirelength_budget=route_wirelength_budget)
    px, py = pin_positions(nl, node_x, node_y)
    start = nl.flat_net2pin_start
    n_nets = nl.num_nets
    per_net_crossings = np.zeros(n_nets, dtype=np.int32)
    per_net_ft = np.zeros(n_nets, dtype=np.int32)
    per_net_lambda = np.zeros(n_nets, dtype=np.int32)
    # M3 T1: region-graph (G_R) Steiner routing, computed alongside (but
    # independently of) the legacy MST-geometry fields above -- see EvalResult.
    per_net_steiner = np.zeros(n_nets, dtype=np.int32)
    per_net_home = np.zeros(n_nets, dtype=np.uint8)
    rg_adj, rg_D, rg_ell = build_region_graph(rg)
    rg_next_hop = next_hop_table(rg_adj, rg_D)
    pair_demand = {}
    segment_demand = (np.zeros(segments.num_segments, dtype=np.int64)
                      if segments is not None else None)
    candidate_counts = {} if capacity_candidates else None
    candidates_dropped = 0
    tree_wl = 0.0
    hpwl = 0.0
    large_lb = 0
    pin_rid_all = rg.region_of_points(px, py)
    for net in range(n_nets):
        s, e = start[net], start[net + 1]
        d = e - s
        if d <= 1:
            continue
        pin_idx = nl.flat_net2pin[s:e]
        nx_, ny_ = px[pin_idx], py[pin_idx]
        hpwl += (nx_.max() - nx_.min()) + (ny_.max() - ny_.min())
        net_pin_rids = pin_rid_all[pin_idx]
        pin_regions = set(net_pin_rids.tolist())
        per_net_lambda[net] = len(pin_regions)
        counts = np.bincount(net_pin_rids.astype(np.int64), minlength=rg.k)
        per_net_home[net] = np.argmax(counts)  # ties -> first (smallest) index
        st, _ft_rg, _exact = steiner_tree_stats(rg_D, rg_next_hop, pin_regions)
        per_net_steiner[net] = st
        if d > max_degree:
            lb = len(pin_regions) - 1
            per_net_crossings[net] = lb
            large_lb += lb
            continue
        edges = net_mst_edges(nx_, ny_)
        routes = None
        if router is not None:
            pins = np.column_stack([nx_, ny_])
            routes = router.route_edges(pins[edges[:, 0]], pins[edges[:, 1]],
                                       wirelength_budget=route_wirelength_budget)
        passed = set()
        ncross = 0
        for edge_index, (a, b) in enumerate(edges):
            if routes is None:
                regs, nc, pairs = edge_regions_and_crossings(
                    rg, nx_[a], ny_[a], nx_[b], ny_[b])
                length = abs(nx_[a] - nx_[b]) + abs(ny_[a] - ny_[b])
            else:
                regs, pairs = set(), []
                points = routes["points"][edge_index]
                nc = sum(_walk_segment(rg, *p, *q, regs, pairs)
                         for p, q in zip(points[:-1], points[1:]))
                length = routes["wirelength"][edge_index]
            passed |= regs
            ncross += nc
            for pr in pairs:
                pair_demand[pr] = pair_demand.get(pr, 0) + 1
            tree_wl += length
            if segments is not None:
                if routes is None:
                    seg_ids = edge_segment_ids(rg, segments, nx_[a], ny_[a],
                                               nx_[b], ny_[b])
                else:
                    seg_ids = np.concatenate(
                        [edge_segment_ids(rg, segments, p[0], p[1], q[0], q[1])
                         for p, q in zip(points[:-1], points[1:])]
                        or [np.zeros(0, dtype=np.int32)])
                # the sec 5 "free assertion on slice lengths": the CSR slice
                # and the independent lattice walk must agree exactly.
                assert len(seg_ids) == nc, (
                    "segment lookup disagrees with the lattice walk: "
                    "%d vs %d" % (len(seg_ids), nc))
                np.add.at(segment_demand, seg_ids, 1)
                if candidate_counts is not None:
                    ua, ub = int(net_pin_rids[a]), int(net_pin_rids[b])
                    if ua == ub:
                        candidates_dropped += len(seg_ids)
                    else:
                        lo_r, hi_r = min(ua, ub), max(ua, ub)
                        for sid in seg_ids:
                            key = (net, lo_r, hi_r, int(sid))
                            candidate_counts[key] = candidate_counts.get(key, 0) + 1
        per_net_crossings[net] = ncross
        per_net_ft[net] = len(passed - pin_regions)
    hard_lambda_sum = int(np.maximum(per_net_lambda - 1, 0).sum())
    io_rg = int(per_net_steiner.sum())
    ft_rg = io_rg - hard_lambda_sum
    st = straddle_diagnostics(nl, node_x, node_y, rg,
                              pin_rid=pin_rid_all) if straddle else None
    capacity_fields = {}
    if segments is not None:
        capacity_fields["segment_demand"] = segment_demand
        assert int(segment_demand.sum()) == int(per_net_crossings.sum()) - large_lb, \
            "per-segment demand does not reconcile with the Ph/Pv crossing count"
        if segment_capacity is not None:
            capacity = np.asarray(segment_capacity, dtype=np.float64)
            util, scalars = segment_utilisation(segment_demand, capacity)
            capacity_fields["segment_capacity"] = capacity
            capacity_fields["segment_util"] = util
            for name in ("num_over_capacity", "num_zero_capacity_segments",
                         "zero_capacity_demand", "max_util", "p99_util"):
                capacity_fields[name] = scalars[name]
    if candidate_counts is not None:
        size = segments.num_segments
        keys = np.array(sorted(candidate_counts), dtype=np.int64).reshape(-1, 4)
        counts = np.array([candidate_counts[tuple(int(v) for v in row)]
                           for row in keys], dtype=np.int64)
        # sort by the composite key torch.unique will produce on the GPU side
        composite = (((keys[:, 0] * rg.k + keys[:, 1]) * rg.k + keys[:, 2]) * size
                     + keys[:, 3]) if keys.size else np.zeros(0, dtype=np.int64)
        order = np.argsort(composite, kind="stable")
        capacity_fields.update(
            cand_net=keys[order, 0] if keys.size else np.zeros(0, dtype=np.int64),
            cand_u=keys[order, 1] if keys.size else np.zeros(0, dtype=np.int64),
            cand_v=keys[order, 2] if keys.size else np.zeros(0, dtype=np.int64),
            cand_seg=keys[order, 3] if keys.size else np.zeros(0, dtype=np.int64),
            cand_count=counts[order] if keys.size else np.zeros(0, dtype=np.int64),
            cand_dropped=int(candidates_dropped))
    return EvalResult(
        io_count=int(per_net_crossings.sum()),
        ft_count=int(per_net_ft.sum()),
        tree_wl=float(tree_wl), hpwl=float(hpwl),
        per_net_crossings=per_net_crossings, per_net_ft=per_net_ft,
        boundary_pair_demand=pair_demand, large_net_lb=int(large_lb),
        hard_lambda_sum=hard_lambda_sum,
        per_net_lambda=per_net_lambda,
        io_rg=io_rg, ft_rg=ft_rg,
        per_net_steiner=per_net_steiner, per_net_home=per_net_home,
        straddle_cells=st.straddle_cells if st else 0,
        straddle_area_fraction=st.straddle_area_fraction if st else 0.0,
        straddle_pin_split_nets=st.straddle_pin_split_nets if st else 0,
        straddle_out_area=st.straddle_out_area if st else 0.0,
        straddle_movable_area=st.straddle_movable_area if st else 0.0,
        straddle_wide_cells=st.straddle_wide_cells if st else 0,
        per_node_straddle=st.per_node_straddle if st else None,
        per_net_pin_split=st.per_net_pin_split if st else None,
        **capacity_fields)
