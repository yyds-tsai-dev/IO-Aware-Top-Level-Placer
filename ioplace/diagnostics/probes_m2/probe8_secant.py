"""Probe 8 (v2 validation): does refresh_nesterov_secant() work against the REAL
NesterovAcceleratedGradientOptimizer state layout, and is the H1 failure real?"""
import json, sys
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install")
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/dreamplace")
import torch
from NesterovAcceleratedGradientOptimizer import NesterovAcceleratedGradientOptimizer as NAG

state = {"ver": 0}
A = torch.tensor([3.0, 1.0]); ctr = torch.tensor([5.0, -5.0])

def obj_and_grad_fn(p):
    if p.grad is not None: p.grad.zero_()
    o = 0.5*(A*p*p).sum()
    if state["ver"] == 1:                      # objective jump
        o = o + 50.0*((p-ctr)**2).sum()
    o.backward()
    return o.detach(), p.grad

def refresh_nesterov_secant(opt):
    g = opt.param_groups[0]
    if not g["g_k"]: return
    f = opt.obj_and_grad_fn
    ok, gk = f(g["v_k"][0]); g["g_k"][0].copy_(gk.data); g["obj_k"][0].copy_(ok.data)
    if g["g_k_1"]:
        ok1, gk1 = f(g["v_k_1"][0]); g["g_k_1"][0].copy_(gk1.data); g["obj_k_1"][0].copy_(ok1.data)
        dv = (g["v_k"][0].data - g["v_k_1"][0].data).norm(p=2)
        dg = (g["g_k"][0] - g["g_k_1"][0]).norm(p=2)
        if dv > 0 and dg > 0: g["alpha_k"][0].copy_(dv/dg)

def _run_trial(do_refresh):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    state["ver"] = 0
    opt = NAG([p], lr=0.01, obj_and_grad_fn=obj_and_grad_fn,
              constraint_fn=lambda t: None, use_bb=False)
    obj_and_grad_fn(p)                                  # prime p.grad (DP does this too)
    for _ in range(5):
        opt.step()
    g = opt.param_groups[0]
    state["ver"] = 1                                    # <-- objective changes here
    if do_refresh: refresh_nesterov_secant(opt)
    vk = g["v_k"][0].data.clone()
    gk_cached = g["g_k"][0].clone()
    true_new = (A*vk + 100.0*(vk-ctr))                  # grad of new objective at v_k
    true_old = (A*vk)
    n0 = g["obj_eval_count"]; a_before = g["alpha_k"][0].item()
    opt.step()
    return dict(err_vs_new=(gk_cached-true_new).abs().max().item(),
                err_vs_old=(gk_cached-true_old).abs().max().item(),
                alpha_before=a_before, alpha_after=g["alpha_k"][0].item(),
                obj_evals_in_step=g["obj_eval_count"]-n0)

def run() -> dict:
    return {"no_refresh": _run_trial(False), "with_refresh": _run_trial(True)}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
