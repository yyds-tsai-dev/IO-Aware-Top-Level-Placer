"""Probe 1: numerical behaviour of the product-form IO surrogate.

Q1: does naive autograd on q = 1 - exp(sum(log1p(-p))) produce NaN / zero grads
    where the true leave-one-out gradient is O(1)?
Q2: does the log-space leave-one-out backward fix it?
"""
import json
import torch

dev = "cuda"

def softmax_p(z, tau):
    return torch.softmax(-z / tau, dim=-1)

def run() -> dict:
    torch.manual_seed(0)
    # --- synthetic: one net of degree d, K regions, one pin deep inside region 0 ---
    K = 16
    cases = []
    for dtype in (torch.float32,):
        for d, deep in [(2, 1), (4, 1), (8, 4), (32, 16)]:
            # signed distances: pin 0..deep-1 deep inside region 0 (d_0 = -R), others far
            R = 1000.0
            dist = torch.full((d, K), R, dtype=dtype, device=dev)
            dist[:deep, 0] = -R                 # deep inside region 0
            for i in range(deep, d):
                dist[i, (i % (K - 1)) + 1] = -R  # deep inside some other region
            tau = 50.0
            dist = dist.clone().requires_grad_(True)
            p = softmax_p(dist, tau)
            # naive path
            ell = torch.log1p(-p)                       # no clamp -> log(0) possible
            S = ell.sum(dim=0)                          # (K,)
            q = 1.0 - torch.exp(S)
            L = q.sum()
            g_naive = torch.autograd.grad(L, dist, retain_graph=False)[0]
            # exact leave-one-out: dq_k/dp_ik = exp(S_k - ell_ik)
            with torch.no_grad():
                p2 = softmax_p(dist, tau)
                ell2 = torch.log1p(-p2.clamp(max=1 - 1e-7))
                S2 = ell2.sum(dim=0)
                loo = torch.exp(S2.unsqueeze(0) - ell2)   # (d,K) = prod_{j!=i}(1-p_j)
            cases.append({
                "dtype": str(dtype), "d": d, "deep": deep, "tau": tau,
                "p_max": p.max().item(),
                "S0": S[0].item(), "exp_S0": torch.exp(S[0]).item(),
                "naive_grad_nan": bool(torch.isnan(g_naive).any().item()),
                "naive_g_abs_max": g_naive.abs().max().item(),
                "loo_max": loo.max().item(),
            })
    return {"K": K, "cases": cases}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
