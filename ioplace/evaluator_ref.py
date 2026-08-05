import numpy as np
from dataclasses import dataclass, field
from ioplace.netlist import pin_positions

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

def evaluate(nl, node_x, node_y, rg, max_degree=256):
    px, py = pin_positions(nl, node_x, node_y)
    start = nl.flat_net2pin_start
    n_nets = nl.num_nets
    per_net_crossings = np.zeros(n_nets, dtype=np.int32)
    per_net_ft = np.zeros(n_nets, dtype=np.int32)
    pair_demand = {}
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
        pin_regions = set(pin_rid_all[pin_idx].tolist())
        if d > max_degree:
            lb = len(pin_regions) - 1
            per_net_crossings[net] = lb
            large_lb += lb
            continue
        edges = net_mst_edges(nx_, ny_)
        passed = set()
        ncross = 0
        for (a, b) in edges:
            regs, nc, pairs = edge_regions_and_crossings(
                rg, nx_[a], ny_[a], nx_[b], ny_[b])
            passed |= regs
            ncross += nc
            for pr in pairs:
                pair_demand[pr] = pair_demand.get(pr, 0) + 1
            tree_wl += abs(nx_[a] - nx_[b]) + abs(ny_[a] - ny_[b])
        per_net_crossings[net] = ncross
        per_net_ft[net] = len(passed - pin_regions)
    return EvalResult(
        io_count=int(per_net_crossings.sum()),
        ft_count=int(per_net_ft.sum()),
        tree_wl=float(tree_wl), hpwl=float(hpwl),
        per_net_crossings=per_net_crossings, per_net_ft=per_net_ft,
        boundary_pair_demand=pair_demand, large_net_lb=int(large_lb))
