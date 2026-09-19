"""M4 T6 probe (2026-08-14 T6 adjudication doc
`docs/results/2026-08-14-m4-t6-adjudication.md` sec 2(c) / §3.2's glue
distance-kernel discussion): within a single `mempool_group.def` (the
level *below* the 4-group cluster), fits a power-law `lambda(d) = lambda_0
* d^-alpha` between the 16 `gen_tiles[i].i_tile` blocks' cross-tile net
count and their (mass, not bbox) true-centroid distance -- the adjudication
doc's "下一層(16 tile,120 pair)實測 α=1.915、corr=-0.785" evidence that
the cluster level's flat inter-group connectivity (α~0.10, statistically
indistinguishable from 0) is a top-level-specific finding, not a property
of the RTL hierarchy in general.

Migrated from `/tmp/p4_tilepairs.py`: sealed (repo-relative I/O, `env`
provenance block, argparse instead of a hardcoded path) and now writes its
JSON to `results/m4/probes/` instead of stdout. Two-pass over `mempool_
group.def` on the *same* open file handle (pass 1: `COMPONENTS` -> mass
centroids; pass 2, continuing from where pass 1 left off: `NETS` -> tile
pair net counts) -- avoids reopening/reseeking a 2.3 GB file. `NETS`
pin-tuple scanning here already excludes `( PIN ... )`/`( * ... )` tuples
(pre-dates Bug B and, unlike Bug A/B, was never affected by it: p4 stops
per-line iteration at the first `+` token exactly like `hierarchy_gate.
scan_def`'s fix).

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_tile_distance_decay \\
        [--group-def <mempool_group.def>]
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
from collections import Counter

import numpy as np

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DEFAULT_GROUP_DEF = ("/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/"
                      "ISPD2025_benchmarks/visible/mempool_group/mempool_group.def")
TOP = "__TOP__"


def _group_key(name):
    i = name.find("/")
    return name[:i] if i != -1 else TOP


def _degree_bucket(d):
    for b in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 256, 1024):
        if d <= b:
            return b
    return 100000


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


def run(group_def):
    sx, sy, nc = Counter(), Counter(), Counter()
    die = None
    section = "pre"
    t0 = time.time()

    with open(group_def, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("DIEAREA "):
                    t = line.split()
                    die = [float(t[2]), float(t[3]), float(t[6]), float(t[7])]
                elif line.startswith("COMPONENTS "):
                    section = "comp"
                continue
            if line.startswith("END COMPONENTS"):
                section = "between"
                break
            if not line.startswith("- "):
                continue
            toks = line.split()
            k = _group_key(toks[1])
            nc[k] += 1
            si = None
            for j in range(3, len(toks)):
                if toks[j] in ("FIXED", "COVER", "PLACED", "UNPLACED"):
                    si = j
                    break
            if si is not None and si + 3 < len(toks) and toks[si + 1] == "(":
                sx[k] += float(toks[si + 2]); sy[k] += float(toks[si + 3])
        cent = {k: (sx[k] / nc[k], sy[k] / nc[k]) for k in nc}

        # Pass 2 (continuing the same file handle): NETS -> tile-pair net counts.
        pair, internal, touch, npins = Counter(), Counter(), Counter(), Counter()
        deg_cross, nblk_hist = Counter(), Counter()
        n_nets = 0
        cur, in_pins = None, True

        def flush():
            nonlocal cur
            if cur is None:
                return
            ks = list(cur)
            for k in ks:
                touch[k] += 1; npins[k] += cur[k]
            if len(ks) == 1:
                internal[ks[0]] += 1
            else:
                nblk_hist[len(ks)] += 1
                deg_cross[_degree_bucket(sum(cur.values()))] += 1
                ks2 = sorted(ks)
                for i in range(len(ks2)):
                    for j in range(i + 1, len(ks2)):
                        pair[(ks2[i], ks2[j])] += 1
            cur = None

        for line in f:
            if section == "between":
                if line.startswith("NETS "):
                    section = "nets"
                continue
            if line.startswith("END NETS"):
                flush()
                break
            if line.startswith("- "):
                flush()
                n_nets += 1
                cur = Counter()
                in_pins = True
            if not in_pins:
                continue
            toks = line.split()
            i, n = 0, len(toks)
            while i < n:
                if toks[i] == "+":
                    in_pins = False
                    break
                if toks[i] == "(" and i + 3 < n and toks[i + 3] == ")":
                    inst = toks[i + 1]
                    if inst not in ("PIN", "*"):
                        cur[_group_key(inst)] += 1
                    i += 4
                else:
                    i += 1
    scan_s = time.time() - t0

    tiles = sorted((k for k in nc if k.startswith("gen_tiles")),
                    key=lambda k: int(k.split("[")[1].split("\\")[0]))
    W, H = die[2] - die[0], die[3] - die[1]
    ext = max(W, H)
    rows = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            a, b = tiles[i], tiles[j]
            c = pair.get((min(a, b), max(a, b)), 0)
            d = math.hypot(cent[a][0] - cent[b][0], cent[a][1] - cent[b][1]) / ext
            rows.append((d, c, a, b))
    rows.sort()
    # `.reshape(-1, 2)` keeps this a proper (n, 2) array even when `rows` is
    # empty (a design with <2 `gen_tiles[...]` prefixes) -- `np.array([])`
    # on an empty list is 1-D and `arr[:, 1]` below would raise.
    arr = np.array([(d, c) for d, c, _, _ in rows], dtype=float).reshape(-1, 2)

    fit = dict(alpha=None, log_slope=None, lambda0=None, n_nonzero_pairs=0, corr_log_d_log_lambda=None)
    m = arr[:, 1] > 0
    if m.sum() >= 5:
        lg = np.polyfit(np.log(arr[m, 0]), np.log(arr[m, 1]), 1)
        r = float(np.corrcoef(np.log(arr[m, 0]), np.log(arr[m, 1]))[0, 1])
        fit = dict(alpha=float(-lg[0]), log_slope=float(lg[0]), lambda0=float(math.exp(lg[1])),
                   n_nonzero_pairs=int(m.sum()), corr_log_d_log_lambda=r)

    per_tile = {t: dict(internal=int(internal[t]), touch=int(touch[t]),
                         cross=int(touch[t] - internal[t])) for t in tiles}
    inter = Counter()
    for (a, b), c in pair.items():
        if not a.startswith("gen_tiles"):
            inter[a] += c
        if not b.startswith("gen_tiles"):
            inter[b] += c

    return dict(
        group_def=os.path.abspath(group_def), scan_s=scan_s, n_nets=n_nets,
        n_tile_pairs=len(rows), total_tile_pair_net_incidences=int(arr[:, 1].sum()) if len(rows) else 0,
        distance_range=[float(arr[0, 0]), float(arr[-1, 0])] if len(rows) else None,
        count_min=int(arr[:, 1].min()) if len(rows) else None,
        count_max=int(arr[:, 1].max()) if len(rows) else None,
        count_mean=float(arr[:, 1].mean()) if len(rows) else None,
        closest_12_pairs=[[round(d, 4), int(c)] for d, c, _, _ in rows[:12]],
        farthest_12_pairs=[[round(d, 4), int(c)] for d, c, _, _ in rows[-12:]],
        power_law_fit=fit,
        blocks_per_net_hist_cross_only={str(k): v for k, v in sorted(nblk_hist.items())},
        deg_hist_cross={str(k): v for k, v in sorted(deg_cross.items())},
        top_non_tile_blocks_by_cross_incidence=[[k, int(v), int(nc[k])] for k, v in inter.most_common(8)],
        per_tile=per_tile,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-def", default=DEFAULT_GROUP_DEF)
    args = ap.parse_args(argv)

    result = run(args.group_def)
    result["env"] = _env_metadata([args.group_def])
    out_path = os.path.join(REPO, "results", "m4", "probes", "probe_tile_distance_decay.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    fit = result["power_law_fit"]
    print(f"[probe_tile_distance_decay] scan_s={result['scan_s']:.1f} "
          f"alpha={fit['alpha']} corr={fit['corr_log_d_log_lambda']} wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
