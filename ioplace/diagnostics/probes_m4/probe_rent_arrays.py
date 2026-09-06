"""M4 T6/T6B Rent-exponent probe (2026-08-15 T6 holdout adjudication doc
`docs/results/2026-08-15-m4-t6-holdout-adjudication.md` sec 8 item 1): the
generic, target-parameterized replacement for the two ad hoc `/tmp` scripts
that produced T6's H1/H2/H6 holdout judgements -- `measure_rent_cluster.py`
(one hardcoded call on `mempool_cluster.def`) and `measure_rent_synthetic.py`
(a hardcoded `{1x1_source,1x2_n1,1x2_n2,2x2_n1,2x2_n2}` loop). Both were
rescued to `results/m4/bench/rescue/` (2026-08-15, after nearly being lost
-- they were never checked into the repo) but were not runnable from a
fresh checkout until this migration; this probe folds their shared logic
(load one Bookshelf/DEF netlist, call `rent.measure_rent` with T6's frozen
parameters, write the per-level table + fitted p) into one CLI invocation
per target instead of two hardcoded scripts.

One invocation measures one target's Rent exponent. T6's per-array table
(H1/H2/H6's `{1x1_source, 1x2_n1, 1x2_n2, 2x2_n1, 2x2_n2}` + `cluster`) and
T6B's new artifacts (adjudication doc sec 4.1 B-0..B-6: `3x3_n1`, `3x3_n2`,
and `*-noglue` variants once `build_t6_arrays.py` produces them) are each
one run of this probe; `ioplace/bench/assemble_t6.py` glues the per-target
outputs into `verify_group2x2_*.json` + `t6_summary.json`.

Rent parameters are the adjudication doc's sec 4.1 frozen convention for
*every* T6/T6B artifact ("所有項目一律 ... Rent 一律 `backend="mtkahypar"`,
`threads=16`, `b_lo=1e3`, `b_hi=1e6`, `n_bootstrap=200`,
`max_net_degree=100`（兩邊同套）") -- hardcoded as module constants below,
deliberately *not* exposed on the CLI, so a future run cannot silently pick
a different ruler than the one H1/H2/H6 already compared real vs synthetic
with (the 2026-08-15 doc's sec 1.2/7-2 finding was exactly that mixing
rulers mid-comparison is what broke H1's absolute leg).

Target resolution (`--target`, or `--prefix` to bypass it entirely):
  - `cluster`                     -> the real `mempool_cluster.def` (DEF
    loader, `rent.load_def_netlist`)
  - `group` / `source` / `1x1_source` -> the real `mempool_group` Bookshelf
    export (`results/m4/bench/mempool_group_export/mempool_group`)
  - anything else                 -> `results/m4/bench/arrays/<target>/
    <target>` (the existing `build_t6_arrays.py` dir-equals-prefix
    convention already used by `1x2_n1`/`2x2_n2`/etc; this probe invents no
    new naming, so it transparently picks up future `3x3_n2`/`*-noglue`
    array directories once T6B/T7 produce them)
  - `--prefix <path>`             -> an arbitrary Bookshelf prefix,
    overriding target resolution entirely (the task's "任意 bookshelf 前綴"
    escape hatch)

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_rent_arrays \\
        --target 2x2_n2
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_rent_arrays \\
        --target cluster --out results/m4/probes/probe_rent_arrays__cluster.json
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_rent_arrays \\
        --prefix /some/other/bookshelf_prefix --out /tmp/x.json
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

import numpy as np

from ioplace.bench import rent

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DP = os.environ.get("DREAMPLACE_ROOT", os.path.join(os.path.dirname(REPO), "DREAMPlace"))
SOURCE_PREFIX = os.path.join(REPO, "results/m4/bench/mempool_group_export/mempool_group")
CLUSTER_DEF = ("/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/"
               "ISPD2025_benchmarks/visible/mempool_cluster/mempool_cluster.def")
ALIASES = {"group": SOURCE_PREFIX, "source": SOURCE_PREFIX, "1x1_source": SOURCE_PREFIX}

# Adjudication doc sec 4.1's frozen ruler for every T6/T6B artifact -- not a
# CLI knob, see module docstring.
BACKEND = "mtkahypar"
THREADS = 16
B_LO = 1e3
B_HI = 1e6
N_BOOTSTRAP = 200
MAX_NET_DEGREE = 100


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


def resolve_target(target, prefix_override):
    """-> (kind, path) where kind is "def" (DEF loader) or "bookshelf"
    (Bookshelf-prefix loader). See module docstring's target-resolution
    section."""
    if prefix_override:
        return "bookshelf", prefix_override
    if target == "cluster":
        return "def", CLUSTER_DEF
    if target in ALIASES:
        return "bookshelf", ALIASES[target]
    return "bookshelf", os.path.join(REPO, "results", "m4", "bench", "arrays", target, target)


def _bookshelf_input_paths(prefix):
    return [prefix + ext for ext in (".nodes", ".nets", ".pl") if os.path.exists(prefix + ext)]


def run(target=None, prefix=None, seed=0):
    """Loads the resolved target's netlist and measures its Rent exponent
    with the frozen T6/T6B parameters. Returns `(result_dict,
    net_degrees_array)` -- the caller decides whether/where to persist the
    degree array (needed downstream for H3's KS test, per
    `assemble_t6.py`)."""
    kind, path = resolve_target(target, prefix)
    t0 = time.time()
    if kind == "def":
        nl, meta = rent.load_def_netlist(path)
        input_paths = [path]
    else:
        nl = rent.load_bookshelf_netlist(path)
        meta = None
        input_paths = _bookshelf_input_paths(path)
    load_s = time.time() - t0
    print(f"[probe_rent_arrays] target={target!r} prefix={path!r} kind={kind} "
          f"load_s={load_s:.1f} num_physical={nl.num_physical} num_nets={nl.num_nets}", flush=True)

    res = rent.measure_rent(nl, b_lo=B_LO, b_hi=B_HI, seed=seed, backend=BACKEND,
                             threads=THREADS, n_bootstrap=N_BOOTSTRAP, max_net_degree=MAX_NET_DEGREE)
    total_s = time.time() - t0
    print(f"[probe_rent_arrays] target={target!r} p={res.p:.4f} "
          f"ci=({res.p_ci_lo:.4f},{res.p_ci_hi:.4f}) backend={res.backend} "
          f"levels_used={res.levels_used} total_s={total_s:.1f}", flush=True)

    out = {
        "target": target, "kind": kind, "path": os.path.abspath(path),
        "p": res.p, "p_ci_lo": res.p_ci_lo, "p_ci_hi": res.p_ci_hi, "log_t": res.log_t,
        "backend": res.backend, "levels_used": res.levels_used, "n_bootstrap": res.n_bootstrap,
        "levels": [{"level": rl.level, "n_blocks": rl.n_blocks, "avg_block_size": rl.avg_block_size,
                    "avg_terminals": rl.avg_terminals} for rl in res.levels],
        "load_s": load_s, "total_s": total_s,
        "net_degrees_max": int(nl.net_degrees.max()), "num_nets": int(nl.num_nets),
        "num_physical": int(nl.num_physical),
        "rent_params": {"backend": BACKEND, "threads": THREADS, "b_lo": B_LO, "b_hi": B_HI,
                         "n_bootstrap": N_BOOTSTRAP, "max_net_degree": MAX_NET_DEGREE, "seed": seed},
        "input_paths": input_paths,
    }
    if meta is not None:
        out["meta"] = meta
    return out, nl.net_degrees


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                     help="cluster | group/source/1x1_source | <shape>_<norm> "
                          "(e.g. 2x2_n2, 3x3_n2 once built)")
    ap.add_argument("--prefix", default=None,
                     help="arbitrary Bookshelf prefix, overrides --target resolution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                     help="output JSON path (default: results/m4/probes/"
                          "probe_rent_arrays__<target-or-prefix-basename>.json)")
    ap.add_argument("--save-degrees", action="store_true", default=True,
                     help="also save the net-degree array (needed for H3) "
                          "next to --out as <out-stem>.net_degrees.npy (default: on)")
    ap.add_argument("--no-save-degrees", dest="save_degrees", action="store_false")
    args = ap.parse_args(argv)

    if args.target is None and args.prefix is None:
        ap.error("one of --target or --prefix is required")

    name = args.target if args.target is not None else os.path.basename(args.prefix.rstrip("/"))
    out_path = args.out or os.path.join(REPO, "results", "m4", "probes", f"probe_rent_arrays__{name}.json")

    result, net_degrees = run(target=args.target, prefix=args.prefix, seed=args.seed)
    result["env"] = _env_metadata(result.pop("input_paths"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[probe_rent_arrays] wrote {out_path}")

    if args.save_degrees:
        stem = out_path[:-len(".json")] if out_path.endswith(".json") else out_path
        deg_path = stem + ".net_degrees.npy"
        np.save(deg_path, net_degrees.astype(np.int32))
        print(f"[probe_rent_arrays] wrote {deg_path}")

    return result


if __name__ == "__main__":
    main()
