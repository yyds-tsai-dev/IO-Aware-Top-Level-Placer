"""M4 T3a probe (2026-08-14 T6 adjudication doc
`docs/results/2026-08-14-m4-t6-adjudication.md` sec 4 前置 3 / evidence
summary): the mass-weighted hypothesis-assignment arrangement score for
`mempool_cluster.def`'s 4 dominant `gen_groups[i].i_group` prefixes -- a
standalone cross-check of `ioplace/bench/hierarchy_gate.py`'s own G-D
`classify_arrangement` (both implement the same scoring rule
independently; they should agree, per the probe/gate split established by
T0's migrated probes). Also reports the *fixed-midline* 2x2 score (cx=cy=
0.5, no line search) alongside the searched-line score, since the
adjudication doc quotes both ("最佳象限指派可解釋 66.0%(自由切割線下
68.0%)").

Migrated from `/tmp/p5_arrange.py`: sealed (repo-relative I/O, `env`
provenance block, argparse instead of a hardcoded path) and now writes its
JSON to `results/m4/probes/` instead of stdout. Only reads `COMPONENTS`
(breaks at `END COMPONENTS`) -- geometry-only, no `NETS` scan, so
unaffected by Bug B.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_arrangement_score \\
        [--def <mempool_cluster.def>]
"""
import argparse
import datetime
import hashlib
import itertools
import json
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
CUT_LO, CUT_HI, CUT_STEP = 0.20, 0.80, 0.02


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


def _score(order, gid, labels, nreg):
    """`M[r, c]` = fraction of group r's own cell mass with region label
    c; returns `(M, (best_score, best_perm))` maximizing
    `mean_r(M[r, perm[r]])` over permutations of `range(nreg)`."""
    M = np.zeros((4, nreg))
    for r, gi in enumerate(order):
        m = gid == gi
        M[r] = np.bincount(labels[m], minlength=nreg) / m.sum()
    best = None
    for perm in itertools.permutations(range(nreg), 4) if nreg >= 4 else []:
        s = sum(M[r, perm[r]] for r in range(4)) / 4
        if best is None or s > best[0]:
            best = (s, perm)
    return M, best


def _cut_grid():
    n = round((CUT_HI - CUT_LO) / CUT_STEP)
    return [round(CUT_LO + i * CUT_STEP, 10) for i in range(n + 1)]


def run(def_path):
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
            k = nm[:i] if i != -1 else "__TOP__"
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
    order = np.argsort(-cnt)[:4]
    names = [klist[i] for i in order]
    W, H = die[2] - die[0], die[3] - die[1]
    u, v = (X - die[0]) / W, (Y - die[1]) / H

    t1 = time.time()
    hypotheses = {}
    q2_fixed = (u >= 0.5).astype(int) * 2 + (v >= 0.5).astype(int)
    M, best = _score(order, gid, q2_fixed, 4)
    hypotheses["2x2_fixed_midline"] = dict(score=best[0], perm=list(best[1]), M=M.tolist())

    sx = np.clip((u * 4).astype(int), 0, 3)
    M, best = _score(order, gid, sx, 4)
    hypotheses["1x4_vertical_strips"] = dict(score=best[0], perm=list(best[1]), M=M.tolist())

    sy = np.clip((v * 4).astype(int), 0, 3)
    M, best = _score(order, gid, sy, 4)
    hypotheses["4x1_horizontal_strips"] = dict(score=best[0], perm=list(best[1]), M=M.tolist())

    best_2x2 = None
    for cx in _cut_grid():
        for cy in _cut_grid():
            lab = ((u >= cx).astype(int) * 2 + (v >= cy).astype(int)).astype(np.int64)
            M, b = _score(order, gid, lab, 4)
            if best_2x2 is None or b[0] > best_2x2[0]:
                best_2x2 = (b[0], cx, cy, b[1], M)
    hypotheses["2x2_searched_split_lines"] = dict(
        score=best_2x2[0], cx=best_2x2[1], cy=best_2x2[2],
        perm=list(best_2x2[3]), M=best_2x2[4].tolist())
    grid_search_s = time.time() - t1

    return dict(
        def_path=os.path.abspath(def_path), scan_s=scan_s, grid_search_s=grid_search_s,
        die_bbox=die, top_prefixes=names, hypotheses=hypotheses,
        cut_grid=dict(lo=CUT_LO, hi=CUT_HI, step=CUT_STEP),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="def_path", default=DEFAULT_DEF)
    args = ap.parse_args(argv)

    result = run(args.def_path)
    result["env"] = _env_metadata([args.def_path])
    out_path = os.path.join(REPO, "results", "m4", "probes", "probe_arrangement_score.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    h = result["hypotheses"]
    print(f"[probe_arrangement_score] scan_s={result['scan_s']:.1f} "
          f"2x2_searched={h['2x2_searched_split_lines']['score']:.4f} "
          f"1x4={h['1x4_vertical_strips']['score']:.4f} "
          f"4x1={h['4x1_horizontal_strips']['score']:.4f} wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
