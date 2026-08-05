"""Probe 2: S1+S2 cost/memory + tau-band statistics on a REAL final placement.

Uses the M1 reweight final coordinates (results/m1/adaptec1_reweight_k16_grid.json.npz)
and the same region set the driver would build.
"""
import json, sys, time, os
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer")
import numpy as np, torch
from ioplace.netlist import load_netlist
from ioplace.drivers.run_placement import get_regions_for

DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
dev = "cuda"

def sdf_l1(x, y, R):
    """(N,) x (K,4) -> (N,K) L1 signed distance to axis-aligned rect."""
    dx = torch.maximum(R[:, 0].unsqueeze(0) - x.unsqueeze(1), x.unsqueeze(1) - R[:, 2].unsqueeze(0))
    dy = torch.maximum(R[:, 1].unsqueeze(0) - y.unsqueeze(1), y.unsqueeze(1) - R[:, 3].unsqueeze(0))
    out = dx.clamp(min=0) + dy.clamp(min=0)
    ins = torch.maximum(dx, dy).clamp(max=0)
    return out + ins

def run(case: str = "adaptec1", k: int = 16) -> dict:
    CASE, K = case, k
    cfg = f"{DP}/install/test/ispd2005/{CASE}.json"
    npz = f"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m1/{CASE}_reweight_k{K}_grid.json.npz"

    nl, placedb, params = load_netlist(cfg)
    d = np.load(npz)
    node_x, node_y = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, K, "grid", 0)
    rects = np.stack([r.rects[0] for r in rs.regions])          # (K,4) single-rect grid
    deg = nl.net_degrees
    header = {
        "case": CASE, "k": K,
        "num_physical": nl.num_physical, "num_movable": nl.num_movable, "num_nets": nl.num_nets,
        "num_pins": len(nl.pin2net), "die": list(die),
        "degree_max": int(deg.max()), "n_deg_gt_100": int((deg > 100).sum()),
        "n_deg_gt_256": int((deg > 256).sum()),
        "pins_in_deg_gt_100_nets": int(deg[deg > 100].sum()),
        "pct_pins_in_deg_gt_100_nets": 100 * deg[deg > 100].sum() / deg.sum(),
    }
    area = (die[2]-die[0])*(die[3]-die[1]); L_R = (area/K)**0.5
    header.update(die_area=area, L_R=L_R, bin_size=(die[2]-die[0])/512)

    NX = torch.as_tensor(node_x, dtype=torch.float32, device=dev)
    NY = torch.as_tensor(node_y, dtype=torch.float32, device=dev)
    R = torch.as_tensor(rects, dtype=torch.float32, device=dev)                                 # (K,4)
    pin2node = torch.as_tensor(nl.pin2node.astype(np.int64), dtype=torch.int64, device=dev)
    pin2net  = torch.as_tensor(nl.pin2net.astype(np.int64),  dtype=torch.int64, device=dev)
    poffx = torch.as_tensor(nl.pin_offset_x, dtype=torch.float32, device=dev)
    poffy = torch.as_tensor(nl.pin_offset_y, dtype=torch.float32, device=dev)
    n_nets = nl.num_nets

    def io_surrogate(x, y, tau, use_pins=True):
        dist = sdf_l1(x, y, R)                       # (N,K)
        p = torch.softmax(-dist / tau, dim=1)        # (N,K)
        pp = p[pin2node] if use_pins else p          # (P,K)
        ell = torch.log1p(-pp.clamp(max=1 - 1e-7))   # (P,K)
        S = torch.zeros((n_nets, K), device=dev, dtype=ell.dtype)
        S.index_add_(0, pin2net, ell)                # (n_nets,K)
        q = 1.0 - torch.exp(S)
        return (q.sum(dim=1) - 1.0).clamp(min=0).sum(), p

    taus = []
    for tau_rel in (0.5, 0.2, 0.1, 0.05, 0.02):
        tau = tau_rel * L_R
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        X = NX.clone().requires_grad_(True); Y = NY.clone().requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.time()
        L, p = io_surrogate(X, Y, tau)
        torch.cuda.synchronize(); t1 = time.time()
        g = torch.autograd.grad(L, [X, Y])
        torch.cuda.synchronize(); t2 = time.time()
        gm = (g[0].abs() + g[1].abs())
        nz = (gm > 1e-6 * gm.max()).float().mean().item()
        with torch.no_grad():
            pmax = p.max(dim=1).values
            frac_soft = (pmax < 0.99).float().mean().item()
        taus.append({
            "tau_rel": tau_rel, "tau": tau, "L_io": L.item(),
            "fwd_ms": 1000*(t1-t0), "bwd_ms": 1000*(t2-t1),
            "peak_mb": torch.cuda.max_memory_allocated()/2**20,
            "nan": bool(torch.isnan(g[0]).any()),
            "frac_cells_with_grad": nz, "frac_pmax_lt_0_99": frac_soft,
        })
        del X, Y, L, p, g, gm

    return {**header, "taus": taus}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
