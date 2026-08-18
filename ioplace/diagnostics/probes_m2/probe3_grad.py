"""Probe 3: clamped-naive autograd vs log-space leave-one-out backward, on real coords.
Also: gradient-norm ratio vs the WA-wirelength gradient (for lambda_io auto-normalisation).
"""
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
    Rn = np.stack([r.rects[0] for r in rs.regions])
    L_R = (((die[2]-die[0])*(die[3]-die[1]))/K)**0.5
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    NX, NY, R = T(d["node_x"]), T(d["node_y"]), T(Rn)
    pin2node, pin2net = T(nl.pin2node.astype(np.int64), torch.int64), T(nl.pin2net.astype(np.int64), torch.int64)
    n_nets, P = nl.num_nets, len(nl.pin2net)

    def sdf(x, y):
        dx = torch.maximum(R[:,0][None]-x[:,None], x[:,None]-R[:,2][None])
        dy = torch.maximum(R[:,1][None]-y[:,None], y[:,None]-R[:,3][None])
        return dx.clamp(min=0)+dy.clamp(min=0) + torch.maximum(dx,dy).clamp(max=0)

    def loss_naive(x, y, tau, clamp):
        p = torch.softmax(-sdf(x,y)/tau, 1)
        ell = torch.log1p(-p[pin2node].clamp(max=1-clamp) if clamp else -p[pin2node])
        S = torch.zeros((n_nets,K), device=dev).index_add_(0, pin2net, ell)
        return (1.0-torch.exp(S)).sum(1).sub(1.0).clamp(min=0).sum()

    def stable_p_and_ell(x, y, tau, floor=1e-30):
        """Numerically-safe (p, log(1-p)), matching probe4_loo_grad.stable_p_and_ell:
        the argmax column of 1-p is recomputed from the exact leave-argmax-out sum
        `t` instead of via cancellation (s-e)/s, which loses precision there."""
        z = -sdf(x, y)/tau
        m, am = z.max(1, keepdim=True)
        e = torch.exp(z-m)                       # (N,K), e[argmax]=1
        onehot = torch.zeros_like(e).scatter_(1, am, 1.0)
        t = (e*(1-onehot)).sum(1, keepdim=True).clamp_min(floor)   # sum over j != argmax, EXACT
        s = 1.0+t
        p = e/s
        omp = (s-e)/s                                             # 1-p, cancellation only at argmax
        omp = torch.where(onehot.bool(), (t/s).expand_as(omp), omp).clamp_min(floor)
        return p, torch.log(omp)

    def grad_loo(x, y, tau):
        """log-space leave-one-out backward: dL/dp_{i,k} = exp(S_{e,k} - ell_{i,k})."""
        x = x.detach().clone().requires_grad_(True); y = y.detach().clone().requires_grad_(True)
        p, ell = stable_p_and_ell(x, y, tau)          # 見 probe4_loo_grad.stable_p_and_ell
        ellp = ell[pin2node]
        S = torch.zeros((n_nets, K), device=dev).index_add_(0, pin2net, ellp)
        lam = (1.0 - torch.exp(S)).sum(1)
        with torch.no_grad():
            c = torch.exp(S[pin2net] - ellp) * ((lam - 1.0) > 0).float()[pin2net].unsqueeze(1)
        surrog = (p[pin2node] * c).sum()              # 係數作用在 p,不是 ell
        return torch.autograd.grad(surrog, [x, y])

    taus = []
    for tau_rel in (0.2, 0.05, 0.02):
        tau = tau_rel*L_R
        # naive, no clamp
        x = NX.clone().requires_grad_(True); y = NY.clone().requires_grad_(True)
        L = loss_naive(x, y, tau, clamp=0.0); g0 = torch.autograd.grad(L, [x,y])
        # naive, clamped
        x2 = NX.clone().requires_grad_(True); y2 = NY.clone().requires_grad_(True)
        L2 = loss_naive(x2, y2, tau, clamp=1e-7); g1 = torch.autograd.grad(L2, [x2,y2])
        # log-space LOO
        torch.cuda.reset_peak_memory_stats(); t0=time.time()
        g2 = grad_loo(NX, NY, tau); torch.cuda.synchronize(); t1=time.time()
        n0 = torch.isnan(g0[0]).sum().item()+torch.isnan(g0[1]).sum().item()
        a = torch.cat([g1[0],g1[1]]); b = torch.cat([g2[0],g2[1]])
        rel = ((a-b).abs().sum()/b.abs().sum()).item()
        zero_naive_nonzero_loo = ((a==0)&(b.abs()>1e-8)).sum().item()
        taus.append({
            "tau_rel": tau_rel,
            "nan_count": int(n0),
            "g_loo_l1": b.abs().sum().item(),
            "g_clampnaive_l1": a.abs().sum().item(),
            "rel_l1_diff": rel,
            "cells_naive0_loo_nonzero": int(zero_naive_nonzero_loo),
            "loo_bwd_ms": 1000*(t1-t0),
            "peak_mb": torch.cuda.max_memory_allocated()/2**20,
        })

    return {"case": CASE, "k": K, "taus": taus}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
