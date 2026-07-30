import numpy as np

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
