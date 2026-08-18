"""Probe 4: correct log-space LOO surrogate gradient + WA-WL gradient norm ratio."""
import json, sys, time
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer")
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
    rs = get_regions_for(die, K, "grid", 0)
    Rn = np.stack([r.rects[0] for r in rs.regions]); L_R = (((die[2]-die[0])*(die[3]-die[1]))/K)**0.5
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    NX, NY, R = T(d["node_x"]), T(d["node_y"]), T(Rn)
    p2n, p2e = T(nl.pin2node.astype(np.int64), torch.int64), T(nl.pin2net.astype(np.int64), torch.int64)
    poffx, poffy = T(nl.pin_offset_x), T(nl.pin_offset_y)
    n_nets = nl.num_nets; N = nl.num_physical
    deg = T(nl.net_degrees.astype(np.float32))
    bx = (die[2]-die[0])/512; by = (die[3]-die[1])/512
    base_gamma = params.gamma*(bx+by)

    def sdf(x, y):
        dx = torch.maximum(R[:,0][None]-x[:,None], x[:,None]-R[:,2][None])
        dy = torch.maximum(R[:,1][None]-y[:,None], y[:,None]-R[:,3][None])
        return dx.clamp(min=0)+dy.clamp(min=0)+torch.maximum(dx,dy).clamp(max=0)

    def stable_p_and_ell(x, y, tau, floor=1e-30):
        z = -sdf(x,y)/tau
        m, am = z.max(1, keepdim=True)
        e = torch.exp(z-m)                       # (N,K), e[argmax]=1
        onehot = torch.zeros_like(e).scatter_(1, am, 1.0)
        t = (e*(1-onehot)).sum(1, keepdim=True).clamp_min(floor)   # sum over j != argmax, EXACT
        s = 1.0+t
        p = e/s
        omp = (s-e)/s                                             # 1-p, cancellation only at argmax
        omp = torch.where(onehot.bool(), (t/s).expand_as(omp), omp).clamp_min(floor)
        return p, torch.log(omp)

    def io_forward_and_grad(x0, y0, tau, w=None):
        """L_io value + dL/dx,dL/dy via detached leave-one-out coefficients on p."""
        x = x0.detach().clone().requires_grad_(True); y = y0.detach().clone().requires_grad_(True)
        p, ell = stable_p_and_ell(x,y,tau)
        ellp = ell[p2n]
        S = torch.zeros((n_nets,K), device=dev).index_add_(0,p2e,ellp)
        lam = (1.0-torch.exp(S)).sum(1)                      # expected #regions touched
        wv = torch.ones(n_nets, device=dev) if w is None else w
        L = (wv*(lam-1.0).clamp(min=0)).sum()
        with torch.no_grad():
            active = ((lam-1.0)>0).float()*wv
            c = torch.exp(S[p2e]-ellp)*active[p2e].unsqueeze(1)   # dL/dp_{i,k}, (P,K)
        surrog = (p[p2n]*c).sum()
        gx, gy = torch.autograd.grad(surrog, [x,y])
        return L.item(), gx, gy

    def wa_wl_grad(x0, y0, gamma):
        x = x0.detach().clone().requires_grad_(True); y = y0.detach().clone().requires_grad_(True)
        px = x[p2n]+poffx; py = y[p2n]+poffy
        tot = 0
        for c in (px, py):
            mx = torch.full((n_nets,),-1e30,device=dev).scatter_reduce_(0,p2e,c,reduce="amax",include_self=True)
            mn = torch.full((n_nets,), 1e30,device=dev).scatter_reduce_(0,p2e,c,reduce="amin",include_self=True)
            ep = torch.exp((c-mx[p2e])/gamma); en = torch.exp((mn[p2e]-c)/gamma)
            z = lambda v: torch.zeros(n_nets,device=dev).index_add_(0,p2e,v)
            tot = tot + (z(c*ep)/z(ep) - z(c*en)/z(en)).sum()
        gx, gy = torch.autograd.grad(tot, [x,y]); return tot.item(), gx, gy

    gammas = []
    gWL = None
    for gname, gamma in (("gamma_init(10x)", 10*base_gamma),
                          ("gamma_final(~of=0.07)", base_gamma*10**((0.07-0.1)*20/9-1))):
        wl, gwx, gwy = wa_wl_grad(NX, NY, gamma)
        g_l1 = torch.cat([gwx, gwy]).abs().sum().item()
        gammas.append({"name": gname, "gamma": gamma, "wl": wl, "g_wl_l1": g_l1})
        if gname.startswith("gamma_final"):
            gWL = g_l1

    taus = []
    for tau_rel in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        tau = tau_rel*L_R
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        L, gx, gy = io_forward_and_grad(NX, NY, tau); torch.cuda.synchronize(); t1 = time.time()
        g1 = torch.cat([gx, gy]).abs().sum().item()
        taus.append({
            "tau_rel": tau_rel, "tau": tau, "L_io": L, "g_io_l1": g1,
            "lam_io_star": gWL/g1, "fwd_bwd_ms": 1000*(t1-t0),
            "peak_mb": torch.cuda.max_memory_allocated()/2**20,
            "nan": bool(torch.isnan(gx).any()),
        })

    return {"case": CASE, "k": K, "N": N, "n_nets": n_nets, "n_pins": len(nl.pin2net),
            "L_R": L_R, "base_gamma": base_gamma, "gammas": gammas, "taus": taus}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
