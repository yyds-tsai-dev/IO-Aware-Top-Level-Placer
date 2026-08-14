"""M3 T0 probe (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 4.1): per-region area utilisation (total-area denominator). Data source
for `probe_free_area_util.py`'s free-area variant.

T0-b (design draft sec 8): reissued hermetic -- repo-relative paths (derived
from `__file__`, `--repo-root` overridable), atomic write, unified `env`
provenance schema. Numeric content is unchanged; only the top-level JSON
shape changed (bare list -> `{"env":..., "runs":[...]}`).

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_m3_util

Writes results/m3/probes/probe_m3_util.json.
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

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
CFG = f"{DP}/install/test/ispd2005/adaptec1.json"


def util(tag, npz, k, rtype, nl, placedb):
    d = np.load(npz); nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, k, rtype, 0); rg = RegionGrid(rs)
    nm = nl.num_movable
    cx = nx_[:nm] + nl.node_size_x[:nm] / 2.0
    cy = ny_[:nm] + nl.node_size_y[:nm] / 2.0
    rid = rg.region_of_points(cx, cy)
    area = nl.node_size_x[:nm] * nl.node_size_y[:nm]
    acc = np.zeros(k); np.add.at(acc, rid, area)
    rarea = np.array([sum((r[2]-r[0])*(r[3]-r[1]) for r in reg.rects) for reg in rs.regions])
    u = acc / rarea
    return dict(tag=tag, k=k, rtype=rtype, mean=float(u.mean()), min=float(u.min()),
                max=float(u.max()), max_over_mean=float(u.max()/u.mean()),
                util=[round(float(v),4) for v in u])


RUNS = [("A0_k16","results/m2/ablation/adaptec1_A0_k16_grid.json.npz",16,"grid"),
        ("A2_k16","results/m2/ablation/adaptec1_A2_k16_grid.json.npz",16,"grid"),
        ("A2_k32","results/m2/ablation/adaptec1_A2_k32_grid.json.npz",32,"grid"),
        ("A2_slic","results/m2/ablation/adaptec1_A2_k16_slicing.json.npz",16,"slicing")]


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
        "numpy_version": np.__version__,
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": " ".join([sys.executable, "-m", "ioplace.diagnostics.probes_m3.probe_m3_util"]
                            + sys.argv[1:]),
        "argv": list(sys.argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
        "exactness": "exact (plain rectangle area / cell-center region-membership sums, no approximation)",
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
    runs = [util(t, os.path.join(REPO, f), k, r, nl, placedb) for (t, f, k, r) in RUNS]
    return {"env": _env_metadata([f for (_, f, _, _) in RUNS]), "runs": runs}


OUT_RELPATH = "results/m3/probes/probe_m3_util.json"

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
    print(f"[probe_m3_util] wrote {out_path}")
