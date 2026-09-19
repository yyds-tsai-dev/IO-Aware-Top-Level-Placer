"""M4 T3a/T6 probe (2026-08-14 T6 adjudication doc
`docs/results/2026-08-14-m4-t6-adjudication.md` sec 4 前置 + evidence
summary "真實跨 group 統計"): the pairwise (six unordered `gen_groups[i]`/
`gen_groups[j]` combinations) and per-group cross-boundary net statistics
for `mempool_cluster.def`'s `NETS` section -- what `ioplace/bench/
hierarchy_gate.py`'s G-D gate itself does *not* break down (its
`nets_crossing_or_top` bucket field is an aggregate "this bucket is touched
by >=1 other bucket" count per group, not a pairwise matrix).

Migrated from `/tmp/p1_clean_cross.py` (already had Bug B's `+`-clause
truncation fix baked in -- written to investigate Bug B in the first
place): sealed (repo-relative I/O, `env` provenance block, argparse
instead of a hardcoded path) and now writes its JSON to `results/m4/
probes/` instead of stdout.

Same NETS-record state machine as `hierarchy_gate.scan_def`'s Bug-B fix
(stop scanning pin tuples at a net record's first `+` token; exclude
`( PIN ... )` and `( * ... )` wildcard tuples) -- kept as an independently
written implementation here (not imported from `hierarchy_gate`) so the
two serve as a cross-check of each other, per the probe/gate split
established by T0's migrated probes.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_cross_group_stats \\
        [--def <mempool_cluster.def>]
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
import time
from collections import Counter

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DEFAULT_DEF = ("/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/"
               "ISPD2025_benchmarks/visible/mempool_cluster/mempool_cluster.def")
TOP = "__TOP__"


def _group_key(name):
    i = name.find("/")
    return name[:i] if i != -1 else TOP


def _degree_bucket(d):
    for b in (8, 16, 32, 64, 256, 1024):
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


def run(def_path):
    flat = set()  # top-level instance names with no "/" (seen in COMPONENTS)
    section = "pre"
    n_nets = n_wildcard_tuples = n_nets_with_plus = n_bogus_noslash_tokens = 0
    cross_pin_total = 0
    nets_by_groupset = Counter()  # "{n_groups}g_top{0|1}" -> net count
    pair = Counter()  # (group_a, group_b) -> net count, a<b, both real groups (excludes TOP)
    ngroups_hist = Counter()  # n_groups touched (>=2 only) -> net count
    deg_hist_cross = Counter()  # degree bucket -> net count, crossing nets only
    deg_hist_all = Counter()  # degree bucket -> net count, all nets
    pinshare = Counter()  # (n_groups, per-group pin-count-bucket tuple) -> net count
    per_group_internal = Counter()
    per_group_touch = Counter()
    per_group_pins = Counter()

    cur = None
    in_pins = True

    def flush():
        nonlocal cross_pin_total
        if cur is None:
            return
        gs = set(cur)
        groups = sorted(g for g in gs if g != TOP)
        deg = sum(cur.values())
        deg_hist_all[_degree_bucket(deg)] += 1
        for g in gs:
            per_group_touch[g] += 1
            per_group_pins[g] += cur[g]
        if len(groups) <= 1 and TOP not in gs:
            if groups:
                per_group_internal[groups[0]] += 1
        key = f"{len(groups)}g_top{int(TOP in gs)}"
        nets_by_groupset[key] += 1
        if len(groups) >= 2:
            ngroups_hist[len(groups)] += 1
            deg_hist_cross[_degree_bucket(deg)] += 1
            cross_pin_total += deg
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    pair[(groups[i], groups[j])] += 1
            share = tuple(sorted((cur[g] for g in groups), reverse=True))
            pinshare[(len(groups), tuple(_degree_bucket(x) for x in share[:4]))] += 1

    t0 = time.time()
    with open(def_path, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("COMPONENTS "):
                    section = "comp"
                continue
            if section == "comp":
                if line.startswith("END COMPONENTS"):
                    section = "between"
                    continue
                if line.startswith("- "):
                    nm = line.split(maxsplit=2)[1]
                    if "/" not in nm:
                        flat.add(nm)
                continue
            if section == "between":
                if line.startswith("NETS "):
                    section = "nets"
                continue
            # section == "nets"
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
            if "(" not in line and "+" not in line:
                continue
            toks = line.split()
            i, n = 0, len(toks)
            while i < n:
                t = toks[i]
                if t == "+":
                    in_pins = False
                    n_nets_with_plus += 1
                    break
                if t == "(" and i + 3 < n and toks[i + 3] == ")":
                    inst = toks[i + 1]
                    if inst == "PIN":
                        pass
                    elif inst == "*":
                        n_wildcard_tuples += 1
                    elif "/" in inst:
                        cur[_group_key(inst)] += 1
                    elif inst in flat:
                        cur[TOP] += 1
                    else:
                        n_bogus_noslash_tokens += 1
                    i += 4
                else:
                    i += 1
    scan_s = time.time() - t0

    six_pairs = {f"{a}|{b}": v for (a, b), v in sorted(pair.items())}
    return dict(
        def_path=os.path.abspath(def_path), scan_s=scan_s,
        n_nets=n_nets, n_flat_top_insts=len(flat),
        n_wildcard_tuples=n_wildcard_tuples, n_nets_with_plus=n_nets_with_plus,
        n_bogus_noslash_tokens=n_bogus_noslash_tokens,
        per_group_internal=dict(per_group_internal), per_group_touch=dict(per_group_touch),
        per_group_pins=dict(per_group_pins),
        # per-group *crossing* net count (touch - internal), the quantity
        # the 2026-08-14 adjudication doc calls "每 group 跨界 net".
        per_group_crossing=({g: per_group_touch[g] - per_group_internal.get(g, 0)
                              for g in per_group_touch if g != TOP}),
        nets_by_groupset=dict(sorted(nets_by_groupset.items())),
        six_group_pair_net_counts=six_pairs,
        ngroups_hist=dict(sorted(ngroups_hist.items())),
        deg_hist_cross={str(k): v for k, v in sorted(deg_hist_cross.items())},
        deg_hist_all={str(k): v for k, v in sorted(deg_hist_all.items())},
        cross_pin_total=cross_pin_total,
        pinshare_top25={f"{k[0]}:{k[1]}": v for k, v in pinshare.most_common(25)},
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="def_path", default=DEFAULT_DEF)
    args = ap.parse_args(argv)

    result = run(args.def_path)
    result["env"] = _env_metadata([args.def_path])
    out_path = os.path.join(REPO, "results", "m4", "probes", "probe_cross_group_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[probe_cross_group_stats] scan_s={result['scan_s']:.1f} "
          f"six_pairs={result['six_group_pair_net_counts']} wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
