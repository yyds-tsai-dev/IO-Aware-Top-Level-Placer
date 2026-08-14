"""M4 T3 Lambda distribution (design draft `docs/superpowers/specs/2026-08-
13-m4-scale-up-design-draft.md` M4-L10 / sec 4.4): one `ioplace.
evaluator_ref.evaluate()` pass (K=16, grid regions) over `mempool_tile_wrap`
at its post-GP (`random_center_init_flag=1`, DREAMPlace's config default)
positions, to measure what fraction of nets fall in `per_net_lambda in
[4,8]` -- sec 4.4's Dreyfus-Wagner DP-table guardrail (`B * 2^Lambda * K * 4
B <= 512 MB`) is a batching concern specifically for that degree range, and
the NanGate45 corpus's own Lambda distribution was unmeasured until T3.

Run *after* `probe_warmstart_sanity.py`, which saves the positions this
probe reuses (`<case>_warmstart_on.npz`) rather than re-running GP -- kept
as its own script/process (own external `timeout`) because `evaluate()`'s
per-net MST + region-graph walk is the expensive part `probe_warmstart_
sanity.py` deliberately avoids (see that module's docstring); bundling both
into one script's timeout budget was tried first and blew the 120s budget.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_lambda_dist \\
        <config.json> <case_name>
"""
import json
import os
import sys
import time

import numpy as np

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.evaluator_ref import evaluate
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"

LAMBDA_K, LAMBDA_RTYPE, LAMBDA_SEED = 16, "grid", 0


def run(config_json, npz_path):
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    placedb.initialize(params)

    pos = np.load(npz_path)
    node_x, node_y = pos["node_x"], pos["node_y"]

    nl = netlist_from_placedb(placedb)
    nl.node_x, nl.node_y = node_x, node_y
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, LAMBDA_K, LAMBDA_RTYPE, LAMBDA_SEED))
    res = evaluate(nl, node_x, node_y, rg)

    lam = res.per_net_lambda
    n_nets = len(lam)
    frac_4_8 = float(((lam >= 4) & (lam <= 8)).sum() / n_nets) if n_nets else None

    return dict(lambda_k16=dict(
        k=LAMBDA_K, rtype=LAMBDA_RTYPE, seed=LAMBDA_SEED,
        source="mempool_tile_wrap_warmstart_on.npz (random_center_init_flag=1, post-GP)",
        n_nets=n_nets, frac_lambda_4_to_8=frac_4_8, runtime_s=time.time() - t0,
        lambda_histogram={int(v): int(c) for v, c in zip(*np.unique(lam, return_counts=True))},
    ))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json> <case_name>")
    cfg = os.path.abspath(sys.argv[1])
    case = sys.argv[2]
    npz = os.path.join(REPO, "results", "m4", "corpus", f"{case}_warmstart_on.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"{npz} missing -- run probe_warmstart_sanity.py for {case} first")

    addition = run(cfg, npz)

    out_path = os.path.join(REPO, "results", "m4", "corpus", f"{case}.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(addition)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=1, sort_keys=True)

    lk = addition["lambda_k16"]
    print(f"[probe_lambda_dist] wrote {out_path}: n_nets={lk['n_nets']} "
          f"frac_lambda_4_8={lk['frac_lambda_4_to_8']:.4f} runtime_s={lk['runtime_s']:.1f}")
