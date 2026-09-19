"""M3 T0 probe (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 2.2/3.1): S4a/S4b/S4c/S4d hard-analogue surrogate comparison against
`ft_rg`, same region-graph machinery as `probe_m3_rg.py` (imported from it,
not copy-pasted -- see that module's `region_graph`/`batched_prim`).

T0-b (design draft sec 8): reissued hermetic -- repo-relative paths (derived
from `__file__`, `--repo-root` overridable), atomic write, unified `env`
provenance schema. Also fixes the pre-T0-b version's `exec(open("/tmp/probe_
m3_rg.py")...)` hack (an actual /tmp read dependency) by importing
probe_m3_rg's functions as a normal Python module. Numeric content is
unchanged; only the top-level JSON shape changed (bare list ->
`{"env":..., "runs":[...]}`).

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m3.probe_m3_surrogate

Writes results/m3/probes/probe_m3_surrogate.json.
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
from ioplace.diagnostics.probes_m3.probe_m3_rg import region_graph, batched_prim

from ioplace.paths import REPO_ROOT
REPO = str(REPO_ROOT)
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"
DEV = "cuda"


def cmp_surrogates(tag, npz, k, rtype, nl, placedb, dev="cuda"):
    d = np.load(npz); nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, 0)); K = rg.k
    ctx = GpuEvalContext(nl, rg, device=dev); res = ctx.evaluate(nx_, ny_)
    adj, D = region_graph(rg)
    bm = rg.pin_region_bitmask(nl, nx_, ny_)
    touched = ((bm[:, None].astype(np.uint64) >> np.arange(K, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    deg = nl.net_degrees; touched[deg < 2] = False; lam = touched.sum(1)
    ppx, ppy = pin_positions(nl, nx_, ny_)
    pin_rid = rg.region_of_points(ppx, ppy).astype(np.int64)
    cnt = np.zeros((nl.num_nets, K), dtype=np.int32); np.add.at(cnt, (nl.pin2net, pin_rid), 1)
    home = cnt.argmax(1)
    Dt = torch.as_tensor(D, device=dev); steiner = np.zeros(nl.num_nets, dtype=np.int64)
    for t in range(2, K + 1):
        sel = np.nonzero(lam == t)[0]
        if len(sel) == 0: continue
        term = np.nonzero(touched[sel])[1].reshape(len(sel), t); tt = torch.as_tensor(term, device=dev)
        if t == 2: steiner[sel] = D[term[:,0], term[:,1]]
        elif t == 3: steiner[sel] = (Dt[:,tt[:,0]]+Dt[:,tt[:,1]]+Dt[:,tt[:,2]]).min(dim=0).values.cpu().numpy()
        else:
            sub = Dt[tt.unsqueeze(2), tt.unsqueeze(1)]
            steiner[sel] = batched_prim(sub).cpu().numpy().astype(np.int64)
    ft_rg = steiner - np.maximum(lam - 1, 0)
    sum_all = np.zeros((len(lam), K), dtype=np.int64)
    ecc_all = np.zeros((len(lam), K), dtype=np.int64)
    for h in range(K):
        Dh = D[h][None, :]
        sum_all[:, h] = np.where(touched, np.maximum(Dh-1, 0), 0).sum(1)
        ecc_all[:, h] = np.where(touched, Dh, 0).max(1)
    lm1 = np.maximum(lam - 1, 0)
    S = dict(
        s4a_sum_home = sum_all[np.arange(len(lam)), home],
        s4b_ecc_home = np.maximum(ecc_all[np.arange(len(lam)), home] - lm1, 0),
        s4c_sum_best = sum_all.min(1),
        s4d_ecc_best = np.maximum(ecc_all.min(1) - lm1, 0),
    )
    out = dict(tag=tag, k=k, rtype=rtype, ft_rg=int(ft_rg.sum()), ft_mst=int(res.ft_count))
    m = deg >= 2
    for nm, v in S.items():
        out[nm] = int(v.sum()); out[nm+"_ratio"] = round(float(v.sum()/max(ft_rg.sum(),1)), 3)
        mm = m & ((v + ft_rg) > 0)
        out[nm+"_rho"] = round(float(st.spearmanr(ft_rg[mm], v[mm]).correlation), 3)
        # share of surrogate mass vs share of true ft, by lambda bucket
        sh = {}
        for lo, hi, b in ((2,2,"L2"),(3,3,"L3"),(4,99,"L4p")):
            bm2 = (lam>=lo)&(lam<=hi)
            sv = v[bm2].sum()/max(v.sum(),1); sf = ft_rg[bm2].sum()/max(ft_rg.sum(),1)
            sh[b] = (round(float(sv),3), round(float(sf),3), round(float(sv/max(sf,1e-9)),2))
        out[nm+"_share_vs_true"] = sh
    return out


RUNS = [("A0_k16", "results/m2/ablation/adaptec1_A0_k16_grid.json.npz", 16, "grid"),
        ("A2_k16", "results/m2/ablation/adaptec1_A2_k16_grid.json.npz", 16, "grid"),
        ("A2_k32", "results/m2/ablation/adaptec1_A2_k32_grid.json.npz", 32, "grid"),
        ("A2_slic","results/m2/ablation/adaptec1_A2_k16_slicing.json.npz", 16, "slicing")]


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
        "command": " ".join([sys.executable, "-m", "ioplace.diagnostics.probes_m3.probe_m3_surrogate"]
                            + sys.argv[1:]),
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": {
            "Lambda_le_3": "exact (direct D-table lookup / Steiner-point-search over K choices)",
            "Lambda_ge_4": "metric-closure MST upper bound (design draft sec 2.4 L2)",
            "s4a_s4b_s4c_s4d": "hard-analogue surrogates, not routing costs -- exactness N/A",
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
    runs = [cmp_surrogates(t, os.path.join(REPO, f), k, r, nl, placedb, DEV) for (t, f, k, r) in RUNS]
    return {"env": _env_metadata([f for (_, f, _, _) in RUNS]), "runs": runs}


OUT_RELPATH = "results/m3/probes/probe_m3_surrogate.json"

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
    print(f"[probe_m3_surrogate] wrote {out_path}")
