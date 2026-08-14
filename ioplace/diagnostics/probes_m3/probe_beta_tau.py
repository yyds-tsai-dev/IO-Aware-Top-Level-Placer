"""M3 T0 probe P4 (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 3.4, L4/L6): `L_FT` gradient L1-norm surface over (beta, tau_rel), plus the
fp32-underflow evidence for L6.

Rebuilds the full sec 3.1 forward on real (GPU torch, leaf-tensor) coordinates:

  p_{i,k}   : S1 L1-SDF soft assignment (ioplace/ops/soft_assign.py), tau =
              tau_rel * L_R, L_R = sqrt(A_die/K) -- the design v2 sec 2.1/4.1
              tau normalisation ioplace/ops/io_term.py inherits verbatim.
  S_{e,k}   = sum_{i in e} log(1 - p_{i,k})            (io_term.py's `ell`)
  q_{e,k}   = 1 - exp(S_{e,k}),  lambda_e = sum_k q_{e,k}
  home_e    = hard argmax pin-count region (ties -> smallest index)
  D, ecc_max: region adjacency graph (Floyd-Warshall hop distance), see
              probe_m3_rg.py's region_graph()
  a_{e,k}   = q_{e,k} * exp((D[home_e,k] - ecc_max[home_e]) / beta)
  N_e = sum_k a_{e,k},  M_e = sum_k a_{e,k}*D[home_e,k],  R_e = M_e/max(N_e,1e-30)
  L_FT = sum_e ReLU(R_e - (lambda_e - 1)),  L_IO = sum_e ReLU(lambda_e - 1)  (w_e=1)

N_e/M_e accumulate in fp64 (matching io_term.py's S accumulator contract).
E x K fits in one batch for adaptec1 k16 (E~221k, K=16) -- no chunking needed.

Sweeps beta in {0.25, 0.5, 1.0, 2.0} x tau_rel in {0.30, 0.10, 0.03} (12 cells)
on both adaptec1_A0_k16_grid (flat) and adaptec1_k16_rho0.40_annealed (M2
best). Per cell: L_FT, L_IO, grad_l1_ft = ||dL_FT/dpos||_1 and grad_l1_io
likewise (movable cells only), their ratio, fp64 N_e min, and the count of
nets whose N_e would underflow to 0 if a_{e,k} were accumulated in fp32
instead (L6 evidence). Judgement hook: ratio < 0.05 is the G1 dead-zone.

T0-b (design draft sec 8): reissued hermetic -- `REPO` derived from
`__file__` (overridable with `--repo-root`), atomic write, `exactness` added
to the unified `env` provenance schema.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_beta_tau

Writes results/m3/probes/probe_beta_tau.json directly (not via stdout
redirection): DREAMPlace's PlaceDB loader writes its own INFO/WARNING lines to
stdout, which would otherwise interleave with (and corrupt) a `> out.json`
redirect -- see probe_m3_rg.py's same direct-file-write convention.
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
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.ops.soft_assign import rect_table, region_sdf_l1, softmax_stats, chunk_p_ell
from ioplace.ops.io_term import build_net_node_csr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"
DEV = "cuda"

RUNS = [
    dict(tag="adaptec1_A0_k16_grid", npz="results/m2/ablation/adaptec1_A0_k16_grid.json.npz"),
    dict(tag="adaptec1_k16_rho0.40_annealed", npz="results/m2/sweep/adaptec1_k16_rho0.40_annealed.json.npz"),
]
K = 16
RTYPE = "grid"
BETAS = (0.25, 0.5, 1.0, 2.0)
TAU_RELS = (0.30, 0.10, 0.03)
G1_DEAD_ZONE = 0.05


def region_graph(rg):
    """Copied from probe_m3_rg.py: shared-lattice-edge adjacency + all-pairs
    hop distance (Floyd-Warshall, K<=32 -> a cheap dense table)."""
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
        "dp_commit": _git_head(DP),
        "repo_commit": _git_head(REPO),
        "command": " ".join([sys.executable, "-m", "ioplace.diagnostics.probes_m3.probe_beta_tau"]
                            + sys.argv[1:]),
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": "exact forward evaluation of the S1/S2 soft-assign surrogate at each "
                    "(beta, tau_rel) grid cell -- not a routing-cost approximation",
    }


def _load_case(npz_relpath):
    params, placedb = _load_dreamplace(CFG)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, K, RTYPE, 0)
    rg = RegionGrid(rs)
    rects, r2k = rect_table(rs)
    ignore_net_degree = int(params.ignore_net_degree)
    csr = build_net_node_csr(nl, ignore_net_degree)
    d = np.load(os.path.join(REPO, npz_relpath))
    node_x, node_y = d["node_x"], d["node_y"]
    L_R = ((die[2] - die[0]) * (die[3] - die[1]) / K) ** 0.5

    # home_e (hard, from actual pin positions; ties -> smallest region index,
    # matching np.argmax's first-occurrence tie-break) -- computed once per
    # placement, over ALL nets, then restricted to the csr's active nets below.
    ppx, ppy = pin_positions(nl, node_x, node_y)
    pin_rid = rg.region_of_points(ppx, ppy).astype(np.int64)
    cnt = np.zeros((nl.num_nets, K), dtype=np.int64)
    np.add.at(cnt, (nl.pin2net, pin_rid), 1)
    home_all = cnt.argmax(1)

    adj, D = region_graph(rg)
    ecc_max = D.max(axis=1)

    n_active = len(csr.net_ids)
    degrees_i64 = torch.as_tensor(csr.degrees, dtype=torch.int64, device=DEV)
    node_idx = torch.as_tensor(csr.flat_net2node, dtype=torch.int64, device=DEV)
    net_idx = torch.repeat_interleave(torch.arange(n_active, dtype=torch.int64, device=DEV),
                                      degrees_i64)
    home_e = torch.as_tensor(home_all[csr.net_ids], dtype=torch.int64, device=DEV)
    D_t = torch.as_tensor(D, dtype=torch.float64, device=DEV)
    ecc_max_t = torch.as_tensor(ecc_max, dtype=torch.float64, device=DEV)
    D_home = D_t[home_e]                 # (E,K) fp64 -- gathered once, reused every cell
    ecc_home = ecc_max_t[home_e]         # (E,)

    rects_t = torch.as_tensor(rects, dtype=torch.float64, device=DEV)
    r2k_t = torch.as_tensor(r2k, dtype=torch.int64, device=DEV)
    x = torch.as_tensor(node_x, dtype=torch.float64, device=DEV)
    y = torch.as_tensor(node_y, dtype=torch.float64, device=DEV)

    return dict(nl=nl, n_active=n_active, node_idx=node_idx, net_idx=net_idx,
                rects_t=rects_t, r2k_t=r2k_t, D_home=D_home, ecc_home=ecc_home,
                x=x, y=y, L_R=L_R, npz=npz_relpath)


def _forward(case, tau, beta):
    """One fwd (+ two independent grad calls sharing its graph). Returns
    L_FT, L_IO (python floats), grad_l1_ft, grad_l1_io (movable cells only),
    N_e (fp64, (E,)), and the fp32-accumulated N_e for the L6 underflow count."""
    nl = case["nl"]
    x = case["x"].clone().requires_grad_(True)
    y = case["y"].clone().requires_grad_(True)

    m, t, am = softmax_stats(x, y, case["rects_t"], case["r2k_t"], K, tau)
    sdf = region_sdf_l1(x, y, case["rects_t"], case["r2k_t"], 0, K)
    p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)                 # (N,K)
    ell_pins = ell[case["node_idx"]].double()                   # (P',K) fp64
    S = torch.zeros((case["n_active"], K), dtype=torch.float64, device=x.device
                    ).index_add_(0, case["net_idx"], ell_pins)
    q = -torch.expm1(S)                                         # (E,K) fp64
    lam = q.sum(dim=1)                                          # (E,)

    exponent = (case["D_home"] - case["ecc_home"].unsqueeze(1)) / beta   # (E,K) fp64, <= 0
    a = q * torch.exp(exponent)                                 # (E,K) fp64
    N = a.sum(dim=1)
    Mn = (a * case["D_home"]).sum(dim=1)
    R = Mn / N.clamp(min=1e-30)

    L_FT = torch.relu(R - (lam - 1.0)).sum()
    L_IO = torch.relu(lam - 1.0).sum()

    gx_ft, gy_ft = torch.autograd.grad(L_FT, [x, y], retain_graph=True)
    gx_io, gy_io = torch.autograd.grad(L_IO, [x, y])

    nm = nl.num_movable
    grad_l1_ft = float(gx_ft[:nm].abs().sum() + gy_ft[:nm].abs().sum())
    grad_l1_io = float(gx_io[:nm].abs().sum() + gy_io[:nm].abs().sum())

    with torch.no_grad():
        q32 = q.float()
        exponent32 = exponent.float()
        a32 = q32 * torch.exp(exponent32)
        N32 = a32.sum(dim=1)                        # fp32 accumulation
        underflow = ((N32 == 0.0) & (N > 0.0)).sum().item()

    return {
        "L_FT": float(L_FT.detach()), "L_IO": float(L_IO.detach()),
        "grad_l1_ft": grad_l1_ft, "grad_l1_io": grad_l1_io,
        "N_min_fp64": float(N.min()), "n_underflow_fp32": int(underflow),
        "n_active_nets": case["n_active"],
    }


def run() -> dict:
    results = []
    for r in RUNS:
        case = _load_case(r["npz"])
        grid = []
        for beta in BETAS:
            for tau_rel in TAU_RELS:
                tau = tau_rel * case["L_R"]
                cell = _forward(case, tau, beta)
                ratio = cell["grad_l1_ft"] / cell["grad_l1_io"] if cell["grad_l1_io"] > 0 else float("inf")
                cell.update({"beta": beta, "tau_rel": tau_rel, "tau": tau,
                             "ratio_ft_over_io": ratio,
                             "g1_dead_zone": bool(ratio < G1_DEAD_ZONE)})
                grid.append(cell)

        beta_summary = []
        for beta in BETAS:
            cells = [c for c in grid if c["beta"] == beta]
            beta_summary.append({
                "beta": beta,
                "any_g1_dead_zone": bool(any(c["g1_dead_zone"] for c in cells)),
                "tau_rels_triggering_g1": [c["tau_rel"] for c in cells if c["g1_dead_zone"]],
                "min_ratio": min(c["ratio_ft_over_io"] for c in cells),
            })

        results.append({
            "tag": r["tag"], "npz": r["npz"], "k": K, "rtype": RTYPE,
            "L_R": case["L_R"], "n_active_nets": case["n_active"],
            "grid": grid, "beta_summary": beta_summary,
        })

    return {
        "env": _env_metadata([r["npz"] for r in RUNS]),
        "betas": list(BETAS), "tau_rels": list(TAU_RELS),
        "g1_dead_zone_threshold": G1_DEAD_ZONE,
        "runs": results,
    }


def _atomic_write_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, out_path)


OUT_RELPATH = "results/m3/probes/probe_beta_tau.json"

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
    print(f"[probe_beta_tau] wrote {out_path}")
