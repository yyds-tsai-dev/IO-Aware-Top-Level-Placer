"""M3 P0 probe (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 3.1): soft-regime FT surrogate candidate comparison. This is the
blocking experiment demanded by the adversarial review before v2 sec 3 can be
written: `docs/reviews/2026-08-13-m3-draft-v1-adversarial-opus.md` F1 showed
that the S4b-current candidate, evaluated in the *soft* (non-hard) regime the
optimizer actually sees during GP, overshoots true `ft_rg` by 5.14x and puts
83% of its mass on nets whose true FT is 0 -- the opposite of the hard-limit
numbers design draft sec 3.2 used to pick a winner. F2 traces the failure to
a `(lambda_e - 1)` baseline that goes negative-coefficient at kappa_ft>1; F10
fixes two NaN/underflow traps in the R_e formula. Codex finding 15 requires
probes to be hermetic (repo-relative imports/IO, no /tmp, full provenance).

Shared quantities (all candidates, w_e=1 throughout):
  p_{i,k}   : S1 L1-SDF soft assignment (ioplace/ops/soft_assign.py)
  S_{e,k}   = sum_{i in e} log(1 - p_{i,k})            (soft_assign's `ell`)
  q_{e,k}   = 1 - exp(S_{e,k}),  lambda_e = sum_k q_{e,k}      (soft, M2 S2)
  home_e    : hard argmax pin-count region (ties -> smallest index)
  Lambda_bar_e : hard touched-region count (region-bitmask, `touched.sum()`)
  D[a,b]    : region-adjacency-graph hop distance (Floyd-Warshall, K<=32)
  ecc_max[h]= max_k D[h,k]

Candidates (13 = 1 + 4 families x 3 betas; design draft sec 3.1, Opus F1
fix-list items (a)-(d), Opus F2, Opus F10):
  S4a-soft            : L_e = sum_k q_{e,k} * relu(D[home_e,k] - 1)     (no beta)
  S4b-current          : L_e = relu(R_e - (lambda_e - 1))               beta in {0.25,0.5,1.0}
  S4b-hardbase (F2)    : L_e = relu(R_e - (Lambda_bar_e - 1))           beta in {0.25,0.5,1.0}
  S4b-gated             : L_e = relu(R~_e - (lambda_e - 1))            beta in {0.25,0.5,1.0}
  S4b-gated-hardbase    : L_e = relu(R~_e - (Lambda_bar_e - 1))        beta in {0.25,0.5,1.0}

where (Opus F10: epsilon=0, not max(N,eps); exponentials in fp64):
  g_{e,k} = exp((D[home_e,k] - ecc_max[home_e]) / beta)                (fp64, in (0,1])
  N_e = sum_k q*_{e,k} * g_{e,k},  M_e = sum_k q*_{e,k} * g_{e,k} * D[home_e,k]
  R_e = M_e/N_e if N_e>0 else 0
  q~_{e,k} = relu(q_{e,k} - 0.5) / 0.5           (S4b-gated's membership gate;
                                                   q* is q for S4b-current/hardbase,
                                                   q~ for S4b-gated/gated-hardbase.
                                                   lambda_e in the baseline always
                                                   uses the *original* q, per spec.)

Eligibility (Opus F1 judgement): ratio_vs_true in [0.7, 1.5] AND mass_on_ft0 <
0.20 across all 6 placements x 3 tau_rel = 18 cells.

T0-b (design draft sec 8): reissued hermetic -- `REPO` derived from
`__file__` (overridable with `--repo-root`), atomic write, `exactness` added
to the unified `env` provenance schema.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_ft_surrogate_soft

Writes results/m3/probes/probe_ft_surrogate_soft.json directly (not via stdout
redirection) -- DREAMPlace's PlaceDB loader writes its own INFO/WARNING lines
to stdout, which would otherwise interleave with (and corrupt) a `> out.json`
redirect; see probe_m3_rg.py / probe_beta_tau.py's same convention.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

import numpy as np
import scipy
import scipy.stats as st
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.ops.soft_assign import rect_table, region_sdf_l1, softmax_stats, chunk_p_ell
from ioplace.ops.io_term import build_net_node_csr

from ioplace.paths import REPO_ROOT
REPO = str(REPO_ROOT)
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"
DEV = "cuda"

PLACEMENTS = (
    dict(tag="adaptec1_A0_k16_grid", npz="results/m2/ablation/adaptec1_A0_k16_grid.json.npz",
         k=16, rtype="grid", seed=0),
    dict(tag="adaptec1_A2_k16_grid", npz="results/m2/ablation/adaptec1_A2_k16_grid.json.npz",
         k=16, rtype="grid", seed=0),
    dict(tag="adaptec1_A0_k32_grid", npz="results/m2/ablation/adaptec1_A0_k32_grid.json.npz",
         k=32, rtype="grid", seed=0),
    dict(tag="adaptec1_A2_k32_grid", npz="results/m2/ablation/adaptec1_A2_k32_grid.json.npz",
         k=32, rtype="grid", seed=0),
    dict(tag="adaptec1_A0_k16_slicing", npz="results/m2/ablation/adaptec1_A0_k16_slicing.json.npz",
         k=16, rtype="slicing", seed=0),
    dict(tag="adaptec1_A2_k16_slicing", npz="results/m2/ablation/adaptec1_A2_k16_slicing.json.npz",
         k=16, rtype="slicing", seed=0),
)
TAU_RELS = (0.30, 0.10, 0.03)
BETAS = (0.25, 0.5, 1.0)

ELIGIBILITY_RATIO_RANGE = (0.7, 1.5)
ELIGIBILITY_MASS_ON_FT0_MAX = 0.20

CANDIDATES = (
    dict(name="S4a-soft", family="S4a-soft", beta=None),
    *[dict(name=f"S4b-current_beta{b}", family="S4b-current", beta=b) for b in BETAS],
    *[dict(name=f"S4b-hardbase_beta{b}", family="S4b-hardbase", beta=b) for b in BETAS],
    *[dict(name=f"S4b-gated_beta{b}", family="S4b-gated", beta=b) for b in BETAS],
    *[dict(name=f"S4b-gated-hardbase_beta{b}", family="S4b-gated-hardbase", beta=b) for b in BETAS],
)
assert len(CANDIDATES) == 13

# Opus F1's reproduction target (sanity check on the S4b-current arm that
# originally exposed the soft-regime blowup): adaptec1 A2 k16 grid,
# tau_rel=0.03, beta=0.5 -> total ~= 17389, ratio ~= 5.14, mass_on_ft0 ~= 0.83.
SANITY_CHECK_TARGET = dict(
    placement="adaptec1_A2_k16_grid", tau_rel=0.03, candidate="S4b-current_beta0.5",
    total=17389, ratio_vs_true=5.14, mass_on_ft0=0.83,
)


def region_graph(rg):
    """Shared-lattice-edge adjacency + all-pairs hop distance (Floyd-Warshall,
    K<=32 -> a cheap dense table). Copied verbatim from probe_m3_rg.py /
    probe_beta_tau.py (design draft sec 2.2's region-adjacency graph)."""
    g = rg.grid.astype(np.int64)
    k = rg.k
    adj = np.zeros((k, k), dtype=np.int64)
    a, b = g[:, :-1], g[:, 1:]
    m = a != b
    np.add.at(adj, (a[m], b[m]), 1)
    np.add.at(adj, (b[m], a[m]), 1)
    a, b = g[:-1, :], g[1:, :]
    m = a != b
    np.add.at(adj, (a[m], b[m]), 1)
    np.add.at(adj, (b[m], a[m]), 1)
    INF = 10 ** 6
    D = np.where(adj > 0, 1, INF).astype(np.int64)
    np.fill_diagonal(D, 0)
    for kk in range(k):
        D = np.minimum(D, D[:, kk:kk + 1] + D[kk:kk + 1, :])
    return adj, D


def batched_prim(Dsub):
    """Metric-closure MST cost per net (Prim's algorithm, batched over nets)
    for Lambda>=4 terminal sets -- an upper bound on the exact Steiner cost
    (design draft sec 2.4 L2). Copied verbatim from probe_m3_rg.py."""
    B, t, _ = Dsub.shape
    intree = torch.zeros((B, t), dtype=torch.bool, device=Dsub.device)
    intree[:, 0] = True
    best = Dsub[:, 0, :].clone().double()
    total = torch.zeros(B, dtype=torch.float64, device=Dsub.device)
    INF = 1e12
    for _ in range(t - 1):
        masked = torch.where(intree, torch.tensor(INF, device=Dsub.device, dtype=torch.float64), best)
        v, j = masked.min(dim=1)
        total += v
        intree.scatter_(1, j.unsqueeze(1), True)
        nd = Dsub.gather(1, j.view(B, 1, 1).expand(B, 1, t)).squeeze(1).double()
        best = torch.minimum(best, nd)
    return total


def per_net_ft_rg(nl, node_x, node_y, rg, D, dev):
    """True per-net ft_rg = ST_e - max(Lambda_e-1, 0) (design draft sec 2.2
    identity), over ALL nl.num_nets nets. Lambda<=3 exact (Lambda=2 direct
    D-table lookup, Lambda=3 min-over-Steiner-point), Lambda>=4 metric-closure
    MST upper bound (sec 2.4 L2) -- algorithm matches probe_m3_rg.py exactly.
    Returns (ft_rg, lambda_bar) both (num_nets,) int64 numpy arrays."""
    K = rg.k
    bm = rg.pin_region_bitmask(nl, node_x, node_y)
    touched = ((bm[:, None].astype(np.uint64) >> np.arange(K, dtype=np.uint64)[None, :])
               & np.uint64(1)).astype(bool)
    deg = nl.net_degrees
    touched[deg < 2] = False
    lam_hard = touched.sum(1)
    Dt = torch.as_tensor(D, device=dev)
    steiner = np.zeros(nl.num_nets, dtype=np.int64)
    for t in range(2, K + 1):
        sel = np.nonzero(lam_hard == t)[0]
        if len(sel) == 0:
            continue
        term = np.nonzero(touched[sel])[1].reshape(len(sel), t)
        tt = torch.as_tensor(term, device=dev)
        if t == 2:
            steiner[sel] = D[term[:, 0], term[:, 1]]
        elif t == 3:
            s = (Dt[:, tt[:, 0]] + Dt[:, tt[:, 1]] + Dt[:, tt[:, 2]]).min(dim=0).values
            steiner[sel] = s.cpu().numpy()
        else:
            sub = Dt[tt.unsqueeze(2), tt.unsqueeze(1)]
            steiner[sel] = batched_prim(sub).cpu().numpy().astype(np.int64)
    ft_rg = steiner - np.maximum(lam_hard - 1, 0)
    return ft_rg, lam_hard


def _lambda_bucket_masks(lam_hard_active):
    yield "1", (lam_hard_active == 1)
    yield "2", (lam_hard_active == 2)
    yield "3", (lam_hard_active == 3)
    yield "4+", (lam_hard_active >= 4)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_relpaths):
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": f"{sys.executable} -m ioplace.diagnostics.probes_m3.probe_ft_surrogate_soft",
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": "ft_true (per_net_ft_rg) is exact for Lambda<=3, metric-closure MST upper "
                    "bound for Lambda>=4 (design draft sec 2.4 L2); the 13 candidate surrogates "
                    "themselves are differentiable proxies, not routing costs -- exactness N/A",
    }


def _load_case(spec, nl, placedb, csr, dev=DEV):
    K = spec["k"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, K, spec["rtype"], spec["seed"])
    rg = RegionGrid(rs)
    rects, r2k = rect_table(rs)

    npz_path = os.path.join(REPO, spec["npz"])
    d = np.load(npz_path)
    node_x, node_y = d["node_x"], d["node_y"]

    adj, D = region_graph(rg)
    ecc_max = D.max(axis=1)

    ft_rg_all, lam_hard_all = per_net_ft_rg(nl, node_x, node_y, rg, D, dev)

    # home_e: hard pin-count-argmax region (ties -> smallest index, matching
    # np.argmax's first-occurrence tie-break) -- computed once per placement
    # over ALL nets (probe_m3_rg.py's / probe_beta_tau.py's convention), then
    # restricted to the csr's active net set below.
    ppx, ppy = pin_positions(nl, node_x, node_y)
    pin_rid = rg.region_of_points(ppx, ppy).astype(np.int64)
    cnt = np.zeros((nl.num_nets, K), dtype=np.int64)
    np.add.at(cnt, (nl.pin2net, pin_rid), 1)
    home_all = cnt.argmax(1)

    net_ids = csr.net_ids
    home_active = home_all[net_ids]
    lam_hard_active = lam_hard_all[net_ids]
    ft_rg_active = ft_rg_all[net_ids]

    D_t = torch.as_tensor(D, dtype=torch.float64, device=dev)
    ecc_max_t = torch.as_tensor(ecc_max, dtype=torch.float64, device=dev)
    home_t = torch.as_tensor(home_active, dtype=torch.int64, device=dev)
    D_home = D_t[home_t]                       # (E,K) fp64 -- gathered once, reused every tau/candidate
    ecc_home = ecc_max_t[home_t]                # (E,)
    Lam_hard_t = torch.as_tensor(lam_hard_active, dtype=torch.float64, device=dev)  # (E,)

    n_active = int(len(net_ids))
    degrees_i64 = torch.as_tensor(csr.degrees, dtype=torch.int64, device=dev)
    node_idx = torch.as_tensor(csr.flat_net2node, dtype=torch.int64, device=dev)
    net_idx = torch.repeat_interleave(torch.arange(n_active, dtype=torch.int64, device=dev),
                                      degrees_i64)

    rects_t = torch.as_tensor(rects, dtype=torch.float64, device=dev)
    r2k_t = torch.as_tensor(r2k, dtype=torch.int64, device=dev)
    x = torch.as_tensor(node_x, dtype=torch.float64, device=dev)
    y = torch.as_tensor(node_y, dtype=torch.float64, device=dev)

    L_R = ((die[2] - die[0]) * (die[3] - die[1]) / K) ** 0.5

    # True per-lambda-bucket FT mass share -- placement-level (independent of
    # tau/candidate), the "true_share" half of each cell's mass_by_lambda_bucket.
    ft_true_total = float(ft_rg_active.sum())
    bucket_true_share = {}
    for label, mask in _lambda_bucket_masks(lam_hard_active):
        bucket_true_share[label] = (float(ft_rg_active[mask].sum() / ft_true_total)
                                    if ft_true_total > 0 else 0.0)

    return dict(
        nl=nl, K=K, n_active_nets=n_active, node_idx=node_idx, net_idx=net_idx,
        rects_t=rects_t, r2k_t=r2k_t, D_home=D_home, ecc_home=ecc_home,
        Lam_hard=Lam_hard_t, lam_hard_active=lam_hard_active,
        ft_rg_active=ft_rg_active, ft_true_total=ft_true_total,
        bucket_true_share=bucket_true_share,
        x=x, y=y, L_R=L_R, npz_sha256=_sha256(npz_path),
    )


def _build_candidate(family, beta, q, D_home, ecc_home, lam_soft, Lam_hard):
    """Returns (li, N_or_None, q_r_or_None, exponent_or_None). `li` is the
    (E,) per-net differentiable surrogate contribution (before the final sum
    over e); `N` is the (E,) fp64 R_e-accumulator (None for S4a-soft, which
    has no such accumulator -- it has no beta and no N_e/M_e)."""
    if family == "S4a-soft":
        li = (q * torch.clamp(D_home - 1.0, min=0.0)).sum(dim=1)
        return li, None, None, None

    q_r = q if family in ("S4b-current", "S4b-hardbase") else torch.clamp(q - 0.5, min=0.0) / 0.5
    exponent = (D_home - ecc_home.unsqueeze(1)) / beta        # (E,K) fp64, <=0
    g = torch.exp(exponent)                                   # (E,K) fp64
    N = (q_r * g).sum(dim=1)
    M = (q_r * g * D_home).sum(dim=1)
    # Opus F10: R_e := 0 when N_e==0 (epsilon=0, not max(N,eps)). g_{e,k}>0 for
    # every k, so N_e==0 iff q_r is all-zero for that net, which forces M_e==0
    # too -- substituting a safe denominator (1 instead of 0) where N_e==0
    # leaves the *value* identical to torch.where(N>0, M/N, 0) while keeping
    # M/N_safe finite everywhere, which autograd needs to avoid the classic
    # NaN-in-torch.where trap (the masked-out branch's local gradient -M/N^2
    # is still evaluated, and NaN*0==NaN in IEEE754).
    N_safe = torch.where(N > 0, N, torch.ones_like(N))
    R = torch.where(N > 0, M / N_safe, torch.zeros_like(N))
    baseline = (lam_soft - 1.0) if family in ("S4b-current", "S4b-gated") else (Lam_hard - 1.0)
    li = torch.relu(R - baseline)
    return li, N, q_r, exponent


def _safe_spearman(a, b):
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    rho = st.spearmanr(a, b).correlation
    return float(rho) if np.isfinite(rho) else None


def _run_tau(case, tau_rel, placement_tag):
    nl = case["nl"]
    K = case["K"]
    tau = tau_rel * case["L_R"]
    x = case["x"].clone().requires_grad_(True)
    y = case["y"].clone().requires_grad_(True)

    m, t, am = softmax_stats(x, y, case["rects_t"], case["r2k_t"], K, tau)
    sdf = region_sdf_l1(x, y, case["rects_t"], case["r2k_t"], 0, K)
    p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
    ell_pins = ell[case["node_idx"]].double()
    S = torch.zeros((case["n_active_nets"], K), dtype=torch.float64,
                    device=x.device).index_add_(0, case["net_idx"], ell_pins)
    q = -torch.expm1(S)                        # (E,K) fp64
    lam_soft = q.sum(dim=1)                     # (E,)

    L_IO = torch.relu(lam_soft - 1.0).sum()
    gx_io, gy_io = torch.autograd.grad(L_IO, [x, y], retain_graph=True)
    nm = nl.num_movable
    grad_l1_io = float(gx_io[:nm].abs().sum() + gy_io[:nm].abs().sum())

    ft_true = case["ft_rg_active"]
    ft_true_sum = case["ft_true_total"]

    cells = []
    for cand in CANDIDATES:
        li, N, q_r, exponent = _build_candidate(cand["family"], cand["beta"], q,
                                                  case["D_home"], case["ecc_home"],
                                                  lam_soft, case["Lam_hard"])
        L = li.sum()
        gx, gy = torch.autograd.grad(L, [x, y], retain_graph=True)
        grad_l1 = float(gx[:nm].abs().sum() + gy[:nm].abs().sum())

        li_np = li.detach().cpu().numpy()
        total = float(li_np.sum())
        ratio_vs_true = (total / ft_true_sum) if ft_true_sum > 0 else None
        mass_on_ft0 = float(li_np[ft_true == 0].sum() / total) if total > 0 else 0.0

        mass_by_bucket = {}
        for label, mask in _lambda_bucket_masks(case["lam_hard_active"]):
            surrogate_share = float(li_np[mask].sum() / total) if total > 0 else 0.0
            mass_by_bucket[label] = dict(surrogate_share=surrogate_share,
                                          true_share=case["bucket_true_share"][label])

        spearman_full = _safe_spearman(li_np, ft_true)
        pos_mask = ft_true > 0
        spearman_ftpos = _safe_spearman(li_np[pos_mask], ft_true[pos_mask]) if pos_mask.any() else None

        if N is not None:
            N_np = N.detach().cpu().numpy()
            nz = N_np[N_np > 0]
            n_min_fp64 = float(nz.min()) if len(nz) > 0 else None
            with torch.no_grad():
                q32 = q_r.detach().float()
                exp32 = exponent.detach().float()
                g32 = torch.exp(exp32)
                N32 = (q32 * g32).sum(dim=1)
                n_underflow_fp32_count = int(((N32 == 0.0) & (N.detach() > 0.0)).sum().item())
        else:
            n_min_fp64 = None
            n_underflow_fp32_count = None

        eligible = bool(
            ratio_vs_true is not None
            and ELIGIBILITY_RATIO_RANGE[0] <= ratio_vs_true <= ELIGIBILITY_RATIO_RANGE[1]
            and mass_on_ft0 < ELIGIBILITY_MASS_ON_FT0_MAX
        )

        cells.append(dict(
            placement=placement_tag, tau_rel=tau_rel, candidate=cand["name"],
            family=cand["family"], beta=cand["beta"],
            total=total, ratio_vs_true=ratio_vs_true, mass_on_ft0=mass_on_ft0,
            mass_by_lambda_bucket=mass_by_bucket,
            spearman_full=spearman_full, spearman_ftpos=spearman_ftpos,
            grad_l1=grad_l1, grad_l1_io=grad_l1_io,
            grad_ratio=(grad_l1 / grad_l1_io) if grad_l1_io > 0 else None,
            n_min_fp64=n_min_fp64, n_underflow_fp32_count=n_underflow_fp32_count,
            eligible=eligible,
        ))
    return cells


def _cell_ref(cell, full=False):
    if cell is None:
        return None
    ref = dict(placement=cell["placement"], tau_rel=cell["tau_rel"],
               ratio_vs_true=cell["ratio_vs_true"], mass_on_ft0=cell["mass_on_ft0"])
    if full:
        ref["total"] = cell["total"]
    return ref


def _build_summary(all_cells):
    by_candidate = {}
    for cell in all_cells:
        by_candidate.setdefault(cell["candidate"], []).append(cell)

    per_candidate = []
    for cand in CANDIDATES:
        cells = by_candidate[cand["name"]]
        n_eligible = sum(1 for c in cells if c["eligible"])
        ratio_cells = [c for c in cells if c["ratio_vs_true"] is not None]
        worst_ratio_cell = (max(ratio_cells, key=lambda c: abs(c["ratio_vs_true"] - 1.0))
                            if ratio_cells else None)
        worst_mass_cell = max(cells, key=lambda c: c["mass_on_ft0"]) if cells else None
        per_candidate.append(dict(
            candidate=cand["name"], family=cand["family"], beta=cand["beta"],
            eligible_all_18=(n_eligible == len(cells)),
            n_eligible=n_eligible, n_cells=len(cells),
            worst_ratio_cell=_cell_ref(worst_ratio_cell),
            worst_mass_on_ft0_cell=_cell_ref(worst_mass_cell),
        ))

    sanity_cell = next((c for c in all_cells
                        if c["placement"] == SANITY_CHECK_TARGET["placement"]
                        and c["tau_rel"] == SANITY_CHECK_TARGET["tau_rel"]
                        and c["candidate"] == SANITY_CHECK_TARGET["candidate"]), None)
    sanity_check = dict(target=SANITY_CHECK_TARGET, actual=_cell_ref(sanity_cell, full=True))

    return dict(per_candidate=per_candidate, sanity_check=sanity_check)


def run():
    params, placedb = _load_dreamplace(CFG)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    ignore_net_degree = int(params.ignore_net_degree)
    csr = build_net_node_csr(nl, ignore_net_degree)

    all_cells = []
    placement_meta = []
    for spec in PLACEMENTS:
        case = _load_case(spec, nl, placedb, csr, DEV)
        placement_meta.append(dict(
            tag=spec["tag"], npz=spec["npz"], k=spec["k"], rtype=spec["rtype"], seed=spec["seed"],
            n_active_nets=case["n_active_nets"], ft_true_total=case["ft_true_total"],
            lambda_bucket_true_share=case["bucket_true_share"],
        ))
        for tau_rel in TAU_RELS:
            all_cells.extend(_run_tau(case, tau_rel, spec["tag"]))

    return {
        "probe": "probe_ft_surrogate_soft",
        "env": _env_metadata([spec["npz"] for spec in PLACEMENTS]),
        "config": dict(
            tau_rels=list(TAU_RELS), betas=list(BETAS),
            candidates=[dict(name=c["name"], family=c["family"], beta=c["beta"]) for c in CANDIDATES],
            eligibility_ratio_range=list(ELIGIBILITY_RATIO_RANGE),
            eligibility_mass_on_ft0_max=ELIGIBILITY_MASS_ON_FT0_MAX,
            ignore_net_degree=ignore_net_degree,
            ft_true_is_ub_for_lam_ge4=True,
        ),
        "placements": placement_meta,
        "cells": all_cells,
        "summary": _build_summary(all_cells),
    }


def _atomic_write_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, out_path)


OUT_RELPATH = "results/m3/probes/probe_ft_surrogate_soft.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None,
                    help="override auto-detected repo root (default: derived from __file__)")
    args = ap.parse_args()
    if args.repo_root:
        REPO = os.path.abspath(args.repo_root)

    result = run()
    out_path = os.path.join(REPO, OUT_RELPATH)
    _atomic_write_json(result, out_path)
    print(f"[probe_ft_surrogate_soft] wrote {out_path}")
