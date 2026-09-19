"""M3 T0 probe (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 2.2): region-adjacency-graph (`G_R`) Steiner routing vs the legacy MST-walk
routing (`io_mst`/`ft_mst`), plus the `S4a-star` hard-analogue surrogate and
boundary-pair demand vs shared-boundary length. Cited directly by design draft
sec 2.2's evidence table.

`region_graph`/`batched_prim` are reused (imported, not copy-pasted) by
`probe_m3_bb.py` and `probe_m3_surrogate.py`.

T0-b (design draft sec 8): reissued hermetic -- repo-relative paths (derived
from `__file__`, `--repo-root` overridable), atomic write, unified `env`
provenance schema. Numeric content is unchanged from the pre-T0-b version;
only the top-level JSON shape changed (bare list -> `{"env":..., "runs":[...]}`)
to have somewhere to attach provenance.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_m3_rg

Writes results/m3/probes/probe_m3_rg.json.
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
import scipy.stats as st

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext

from ioplace.paths import REPO_ROOT
REPO = str(REPO_ROOT)
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"
DEV = "cuda"


def region_graph(rg):
    g = rg.grid.astype(np.int64); K = rg.k
    adj = np.zeros((K, K), dtype=np.int64)       # shared-lattice-edge count
    a, b = g[:, :-1], g[:, 1:]                   # horizontal neighbours
    m = a != b
    np.add.at(adj, (a[m], b[m]), 1); np.add.at(adj, (b[m], a[m]), 1)
    a, b = g[:-1, :], g[1:, :]
    m = a != b
    np.add.at(adj, (a[m], b[m]), 1); np.add.at(adj, (b[m], a[m]), 1)
    INF = 10**6
    D = np.where(adj > 0, 1, INF).astype(np.int64)
    np.fill_diagonal(D, 0)
    for k in range(K):                            # Floyd-Warshall, K<=32
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return adj, D

def batched_prim(Dsub):                           # Dsub (B,t,t) torch
    B, t, _ = Dsub.shape
    intree = torch.zeros((B, t), dtype=torch.bool, device=Dsub.device); intree[:, 0] = True
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

def analyse(tag, npz, k, rtype, nl, placedb, dev="cuda"):
    d = np.load(npz)
    nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, 0))
    ctx = GpuEvalContext(nl, rg, device=dev)
    res = ctx.evaluate(nx_, ny_)
    adj, D = region_graph(rg)
    K = rg.k
    bm = rg.pin_region_bitmask(nl, nx_, ny_)                   # (E,) uint64
    touched = ((bm[:, None].astype(np.uint64) >> np.arange(K, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    deg = nl.net_degrees
    touched[deg < 2] = False
    lam = touched.sum(1)
    # per-net "home" region = region holding the most pins (proxy for argmax q)
    px, py = nl.pin_offset_x, nl.pin_offset_y
    ppx, ppy = pin_positions(nl, nx_, ny_)
    pin_rid = rg.region_of_points(ppx, ppy).astype(np.int64)
    cnt = np.zeros((nl.num_nets, K), dtype=np.int32)
    np.add.at(cnt, (nl.pin2net, pin_rid), 1)
    home = cnt.argmax(1)
    Dt = torch.as_tensor(D, device=dev)
    steiner = np.zeros(nl.num_nets, dtype=np.int64)
    for t in range(2, K + 1):
        sel = np.nonzero(lam == t)[0]
        if len(sel) == 0: continue
        term = np.nonzero(touched[sel])[1].reshape(len(sel), t)
        tt = torch.as_tensor(term, device=dev)
        if t == 2:
            steiner[sel] = D[term[:, 0], term[:, 1]]
        elif t == 3:
            s = (Dt[:, tt[:, 0]] + Dt[:, tt[:, 1]] + Dt[:, tt[:, 2]]).min(dim=0).values
            steiner[sel] = s.cpu().numpy()
        else:
            sub = Dt[tt.unsqueeze(2), tt.unsqueeze(1)]        # (B,t,t)
            steiner[sel] = batched_prim(sub).cpu().numpy().astype(np.int64)   # metric-closure MST (UB)
    ft_rg = steiner - np.maximum(lam - 1, 0)
    # star surrogate (hard analogue of the S4 proposal)
    Dh = D[home]                                              # (E,K)
    star_ft = np.where(touched, np.maximum(Dh - 1, 0), 0).sum(1)
    star_io = np.where(touched, Dh, 0).sum(1)
    ok = deg >= 2
    out = dict(tag=tag, k=k, rtype=rtype,
        io_mst=int(res.io_count), ft_mst=int(res.ft_count),
        hard_lambda_sum=int(res.hard_lambda_sum),
        io_rg=int(steiner.sum()), ft_rg=int(ft_rg.sum()),
        star_io=int(star_io.sum()), star_ft=int(star_ft.sum()),
        detour_mst=int(res.io_count - res.hard_lambda_sum),
        n_lam_ge4=int((lam >= 4).sum()), n_lam2=int((lam == 2).sum()),
        n_lam3=int((lam == 3).sum()), n_lam1=int((lam == 1).sum()),
        n_ft_pos=int((ft_rg > 0).sum()), n_ftmst_pos=int((res.per_net_ft > 0).sum()))
    m = ok & (ft_rg + res.per_net_ft > 0)
    out["spearman_ftrg_vs_ftmst"] = float(st.spearmanr(ft_rg[m], res.per_net_ft[m]).correlation)
    m2 = ok & (ft_rg > 0)
    out["spearman_ftrg_vs_starft"] = float(st.spearmanr(ft_rg[m2], star_ft[m2]).correlation)
    out["star_ft_over_ft_rg"] = float(star_ft.sum() / max(ft_rg.sum(), 1))
    # boundary-pair demand vs shared boundary length (in lattice edges)
    dem = res.boundary_pair_demand
    rows = []
    for (a, b), c in sorted(dem.items(), key=lambda kv: -kv[1]):
        rows.append((a, b, c, int(adj[a, b])))
    out["n_adj_pairs"] = int((np.triu(adj, 1) > 0).sum())
    out["demand_top"] = rows[:6]
    out["demand_per_len"] = sorted([c / l for (_, _, c, l) in rows if l > 0])
    dpl = np.array(out["demand_per_len"])
    out["demand_per_len_stats"] = dict(n=len(dpl), min=float(dpl.min()), med=float(np.median(dpl)),
                                       max=float(dpl.max()), max_over_med=float(dpl.max()/np.median(dpl)))
    del out["demand_per_len"]
    return out


RUNS = [("A0_flat", "results/m2/ablation/adaptec1_A0_k16_grid.json.npz", 16, "grid"),
        ("A2_best", "results/m2/ablation/adaptec1_A2_k16_grid.json.npz", 16, "grid"),
        ("A0_k32",  "results/m2/ablation/adaptec1_A0_k32_grid.json.npz", 32, "grid"),
        ("A2_k32",  "results/m2/ablation/adaptec1_A2_k32_grid.json.npz", 32, "grid"),
        ("A0_slic", "results/m2/ablation/adaptec1_A0_k16_slicing.json.npz", 16, "slicing"),
        ("A2_slic", "results/m2/ablation/adaptec1_A2_k16_slicing.json.npz", 16, "slicing")]


# ---------------------------------------------------------------------------
# provenance (pattern established by probe_p0b.py / probe_ft_surrogate_soft.py)
# ---------------------------------------------------------------------------
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
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": " ".join([sys.executable, "-m", "ioplace.diagnostics.probes_m3.probe_m3_rg"]
                            + sys.argv[1:]),
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": {
            "Lambda_le_3": "exact (direct D-table lookup / Steiner-point-search over K choices)",
            "Lambda_ge_4": "metric-closure MST upper bound (design draft sec 2.4 L2)",
        },
    }


def _atomic_write_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, out_path)


def run():
    params, placedb = _load_dreamplace(CFG)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    runs = [analyse(t, os.path.join(REPO, p), k, r, nl, placedb, DEV) for (t, p, k, r) in RUNS]
    return {"env": _env_metadata([p for (_, p, _, _) in RUNS]), "runs": runs}


OUT_RELPATH = "results/m3/probes/probe_m3_rg.json"

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
    print(f"[probe_m3_rg] wrote {out_path}")
