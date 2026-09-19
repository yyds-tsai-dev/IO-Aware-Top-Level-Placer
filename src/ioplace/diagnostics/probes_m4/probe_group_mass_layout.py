"""M4 T3a/T6 probe (2026-08-14 T6 adjudication doc
`docs/results/2026-08-14-m4-t6-adjudication.md` "Bug A" evidence): the
mass-weighted spatial layout of `mempool_cluster.def`'s 4 dominant
`gen_groups[i].i_group` prefixes -- per-group mean/std position, pairwise
cosine similarity of their 64x64 density maps, mass-weighted bin
dominance, fixed-midline quadrant mass fractions, and true (mass) centroid
pairwise distances -- independent of `ioplace/bench/hierarchy_gate.py`'s
own G-D classifier (a standalone cross-check, per the probe/gate split
established by T0's migrated probes; both should agree since they compute
the same underlying quantities from the same file).

Migrated from `/tmp/p3b_mass_cluster.py`: sealed (repo-relative I/O, `env`
provenance block, argparse instead of a hardcoded path) and now writes its
JSON to `results/m4/probes/` instead of stdout. Only reads `COMPONENTS`
(breaks at `END COMPONENTS`, never touches the much larger `NETS`
section) -- this predates Bug B and was never affected by it.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_group_mass_layout \\
        [--def <mempool_cluster.def>] [--top-n 4] [--grid 64]
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from array import array

import numpy as np

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DEFAULT_DEF = ("/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/"
               "ISPD2025_benchmarks/visible/mempool_cluster/mempool_cluster.def")
TOP = "__TOP__"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_paths):
    return dict(
        command=" ".join(sys.argv), python_executable=sys.executable,
        python_version=platform.python_version(), hostname=socket.gethostname(),
        utc_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repo_commit=_git_head(REPO),
        input_sha256={p: _sha256(p) for p in input_paths},
    )


def run(def_path, top_n=4, grid=64):
    section = "pre"
    die = None
    keys, klist = {}, []
    gid, X, Y = array("h"), array("d"), array("d")

    t0 = time.time()
    with open(def_path, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("DIEAREA "):
                    t = line.split()
                    die = [float(t[2]), float(t[3]), float(t[6]), float(t[7])]
                elif line.startswith("COMPONENTS "):
                    section = "comp"
                continue
            if line.startswith("END COMPONENTS"):
                break
            if not line.startswith("- "):
                continue
            toks = line.split()
            nm = toks[1]
            i = nm.find("/")
            k = nm[:i] if i != -1 else TOP
            ki = keys.get(k)
            if ki is None:
                ki = keys[k] = len(klist)
                klist.append(k)
            si = None
            for j in range(3, len(toks)):
                if toks[j] in ("FIXED", "COVER", "PLACED", "UNPLACED"):
                    si = j
                    break
            if si is None or si + 3 >= len(toks) or toks[si + 1] != "(":
                continue
            gid.append(ki); X.append(float(toks[si + 2])); Y.append(float(toks[si + 3]))
    scan_s = time.time() - t0

    gid = np.frombuffer(gid, dtype=np.int16)
    X = np.frombuffer(X, dtype=np.float64)
    Y = np.frombuffer(Y, dtype=np.float64)
    cnt = np.bincount(gid, minlength=len(klist))
    order = np.argsort(-cnt)[:top_n]
    names = [klist[i] for i in order]
    W, H = die[2] - die[0], die[3] - die[1]
    ext = max(W, H)

    per_group = []
    dens = np.zeros((top_n, grid * grid))
    bx = np.clip(((X - die[0]) / W * grid).astype(int), 0, grid - 1)
    by = np.clip(((Y - die[1]) / H * grid).astype(int), 0, grid - 1)
    bin_id = bx * grid + by
    mx, my = [], []
    for r, gi in enumerate(order):
        m = gid == gi
        x, y = X[m], Y[m]
        dens[r] = np.bincount(bin_id[m], minlength=grid * grid)
        cx, cy = float(x.mean()), float(y.mean())
        mx.append(cx); my.append(cy)
        per_group.append(dict(
            prefix=names[r], cells=int(m.sum()),
            mean_u=(cx - die[0]) / W, mean_v=(cy - die[1]) / H,
            std_u=float(x.std()) / W, std_v=float(y.std()) / H,
        ))

    D = dens / dens.sum(axis=1, keepdims=True)
    cosine_matrix = [[float(D[i] @ D[j] / (np.linalg.norm(D[i]) * np.linalg.norm(D[j])))
                       for j in range(top_n)] for i in range(top_n)]

    tot = dens.sum(axis=0)
    frac = np.divide(dens, np.maximum(tot, 1))
    maxf = frac.max(axis=0)
    w = tot / tot.sum()
    mean_max_share = float((maxf * w).sum())
    mass_above = {f"{thr:.1f}": float(w[maxf > thr].sum()) for thr in (0.4, 0.5, 0.7, 0.9)}

    # Fixed-midline quadrant mass fractions (Q = xhi*2 + yhi) -- diagnostic
    # only, not the searched-split-line score `hierarchy_gate.classify_
    # arrangement`'s `score_2x2` computes.
    q = (bx >= grid // 2).astype(int) * 2 + (by >= grid // 2).astype(int)
    quadrant_mass_frac = {}
    for r, gi in enumerate(order):
        m = gid == gi
        quadrant_mass_frac[names[r]] = (np.bincount(q[m], minlength=4) / m.sum()).tolist()

    centroid_dist_matrix = [[round(math.hypot(mx[i] - mx[j], my[i] - my[j]) / ext, 6)
                              for j in range(top_n)] for i in range(top_n)]

    return dict(
        def_path=os.path.abspath(def_path), scan_s=scan_s, grid=grid, top_n=top_n,
        die_bbox=die, die_extent=ext, top_prefixes=names,
        per_group=per_group,
        bin_density_cosine_matrix=cosine_matrix,
        bin_dominance=dict(mean_max_share=mean_max_share,
                            uniform_mix_share=1.0 / top_n,
                            mass_frac_above_dominance_threshold=mass_above),
        quadrant_mass_frac_fixed_midline=quadrant_mass_frac,
        centroid_distance_matrix=centroid_dist_matrix,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="def_path", default=DEFAULT_DEF)
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--grid", type=int, default=64)
    args = ap.parse_args(argv)

    result = run(args.def_path, top_n=args.top_n, grid=args.grid)
    result["env"] = _env_metadata([args.def_path])
    out_path = os.path.join(REPO, "results", "m4", "probes", "probe_group_mass_layout.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[probe_group_mass_layout] scan_s={result['scan_s']:.1f} "
          f"mean_max_share={result['bin_dominance']['mean_max_share']:.4f} wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
