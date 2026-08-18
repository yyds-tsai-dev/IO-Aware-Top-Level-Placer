"""Probe 6: peak memory / time vs K, using bigblue4 coords from the k16 npz."""
import json, sys, time
sys.path.insert(0,"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer")
import numpy as np, torch
from ioplace.netlist import load_netlist
from ioplace.drivers.run_placement import get_regions_for

DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
dev = "cuda"

def run(case: str = "adaptec1", k: int = 16) -> dict:
    """Sweep region-count K in (8,16,32) for `case`, using node coordinates from
    the `k`-grid M1 reweight npz (k only selects the coordinate snapshot; the
    region-count sweep itself is fixed, matching the original script)."""
    CASE = case
    nl, placedb, params = load_netlist(f"{DP}/install/test/ispd2005/{CASE}.json")
    d = np.load(f"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m1/{CASE}_reweight_k{k}_grid.json.npz")
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    p2n = torch.as_tensor(nl.pin2node.astype(np.int64), device=dev)
    p2e = torch.as_tensor(nl.pin2net.astype(np.int64), device=dev)
    NX = torch.as_tensor(d["node_x"], dtype=torch.float32, device=dev)
    NY = torch.as_tensor(d["node_y"], dtype=torch.float32, device=dev)
    n_nets = nl.num_nets

    sweep = []
    for K in (8, 16, 32):
        rs = get_regions_for(die, K, "grid", 0)
        R = torch.as_tensor(np.stack([r.rects[0] for r in rs.regions]), dtype=torch.float32, device=dev)
        L_R = (((die[2]-die[0])*(die[3]-die[1]))/K)**0.5; tau = 0.1*L_R
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = NX.clone().requires_grad_(True); y = NY.clone().requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.time()
        dx = torch.maximum(R[:,0][None]-x[:,None], x[:,None]-R[:,2][None])
        dy = torch.maximum(R[:,1][None]-y[:,None], y[:,None]-R[:,3][None])
        dist = dx.clamp(min=0)+dy.clamp(min=0)+torch.maximum(dx,dy).clamp(max=0)
        p = torch.softmax(-dist/tau, 1)
        ell = torch.log1p(-p[p2n].clamp(max=1-1e-7))
        S = torch.zeros((n_nets,K), device=dev).index_add_(0, p2e, ell)
        L = ((1-torch.exp(S)).sum(1)-1).clamp(min=0).sum()
        torch.cuda.synchronize(); t1 = time.time()
        g = torch.autograd.grad(L, [x, y]); torch.cuda.synchronize(); t2 = time.time()
        sweep.append({
            "k_regions": K, "fwd_ms": 1000*(t1-t0), "bwd_ms": 1000*(t2-t1),
            "peak_mb": torch.cuda.max_memory_allocated()/2**20, "L": L.item(),
        })
        del x, y, dx, dy, dist, p, ell, S, L, g

    return {"case": CASE, "coord_k": k, "N": nl.num_physical, "n_nets": n_nets,
            "n_pins": len(nl.pin2net), "sweep": sweep}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
