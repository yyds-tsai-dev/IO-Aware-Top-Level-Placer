"""Probe 5: elementwise clamped-naive vs stable LOO; fp64 reference; dead-zone fraction."""
import json, sys, time
sys.path.insert(0,"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer")
import numpy as np, torch
from ioplace.netlist import load_netlist
from ioplace.drivers.run_placement import get_regions_for

DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
dev = "cuda"

def run(case: str = "adaptec1", k: int = 16) -> dict:
    CASE, K = case, k
    nl, placedb, params = load_netlist(f"{DP}/install/test/ispd2005/{CASE}.json")
    d = np.load(f"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m1/{CASE}_reweight_k{K}_grid.json.npz")
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, K, "grid", 0); Rn = np.stack([r.rects[0] for r in rs.regions])
    L_R = (((die[2]-die[0])*(die[3]-die[1]))/K)**0.5

    def _run_dtype(dtype):
        T = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=dev)
        NX, NY, R = T(d["node_x"]), T(d["node_y"]), T(Rn)
        p2n = torch.as_tensor(nl.pin2node.astype(np.int64), device=dev)
        p2e = torch.as_tensor(nl.pin2net.astype(np.int64), device=dev)
        n_nets = nl.num_nets
        def sdf(x, y):
            dx = torch.maximum(R[:,0][None]-x[:,None], x[:,None]-R[:,2][None])
            dy = torch.maximum(R[:,1][None]-y[:,None], y[:,None]-R[:,3][None])
            return dx.clamp(min=0)+dy.clamp(min=0)+torch.maximum(dx,dy).clamp(max=0)
        def naive(x, y, tau, cl):
            p = torch.softmax(-sdf(x,y)/tau, 1)
            ell = torch.log1p(-p[p2n].clamp(max=1-cl))
            S = torch.zeros((n_nets,K), device=dev, dtype=dtype).index_add_(0, p2e, ell)
            return ((1-torch.exp(S)).sum(1)-1).clamp(min=0).sum(), p
        def stable(x, y, tau, floor=1e-30):
            z = -sdf(x,y)/tau; m, am = z.max(1, keepdim=True); e = torch.exp(z-m)
            oh = torch.zeros_like(e).scatter_(1, am, 1.0)
            t = (e*(1-oh)).sum(1, keepdim=True).clamp_min(floor); s = 1+t
            p = e/s; omp = torch.where(oh.bool(), (t/s).expand_as(e), (s-e)/s).clamp_min(floor)
            ell = torch.log(omp); ellp = ell[p2n]
            S = torch.zeros((n_nets,K), device=dev, dtype=dtype).index_add_(0, p2e, ellp)
            lam = (1-torch.exp(S)).sum(1); L = (lam-1).clamp(min=0).sum()
            with torch.no_grad():
                c = torch.exp(S[p2e]-ellp)*((lam-1)>0).to(dtype)[p2e].unsqueeze(1)
            return L, (p[p2n]*c).sum(), p
        out = {}
        for tau_rel in (0.2, 0.05, 0.02):
            tau = tau_rel*L_R
            x = NX.clone().requires_grad_(True); y = NY.clone().requires_grad_(True)
            Ln, p = naive(x, y, tau, 1e-7 if dtype == torch.float32 else 1e-15)
            gn = torch.autograd.grad(Ln, [x, y]); gn = torch.cat(gn)
            x2 = NX.clone().requires_grad_(True); y2 = NY.clone().requires_grad_(True)
            Ls, sur, p2 = stable(x2, y2, tau); gs = torch.cat(torch.autograd.grad(sur, [x2, y2]))
            pm = p.max(1).values
            out[tau_rel] = (Ln.item(), Ls.item(), gn, gs,
                             (pm > 1-(1e-7 if dtype == torch.float32 else 1e-15)).float().mean().item())
        return out

    f32 = _run_dtype(torch.float32); f64 = _run_dtype(torch.float64)
    taus = []
    for tr in f32:
        Ln, Ls, gn, gs, dead = f32[tr]; _, _, gn64, gs64, dead64 = f64[tr]
        rel = lambda a, b: ((a-b).abs().sum()/b.abs().sum()).item()
        taus.append({
            "tau_rel": tr, "L_naive": Ln, "L_stable": Ls,
            "rel_l1_fp32_naive_vs_stable": rel(gn, gs),
            "rel_l1_fp32stable_vs_fp64stable": rel(gs.double(), gs64),
            "rel_l1_fp32naive_vs_fp64stable": rel(gn.double(), gs64),
            "deadzone_frac": dead,
            "maxrelerr_elem": ((gn-gs).abs()/(gs.abs()+1e-12)).max().item(),
        })
    return {"case": CASE, "k": K, "taus": taus}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
