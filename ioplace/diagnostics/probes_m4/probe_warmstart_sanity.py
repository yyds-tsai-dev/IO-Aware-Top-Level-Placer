"""M4 T3 warm-start sanity (design draft `docs/superpowers/specs/2026-08-13-
m4-scale-up-design-draft.md` T3 row / sec 8's R2 warm-start risk item): two
short GP sanity runs on `mempool_tile_wrap` -- `random_center_init_flag` on
(DREAMPlace's default: movable cells randomized around the die center,
discarding the DEF's own `PLACED` coordinates) vs off (DREAMPlace's
`BasicPlace.py:269-289` seeds `self.init_pos` straight from
`placedb.node_x/node_y`, i.e. the *input* DEF's placed coordinates, and
only overwrites the *movable* slice when the flag is on) -- recording
initial/final HPWL for each and whether the GP-start position actually
covers the input coordinates, so R2's warm-start decision (deferred to T8)
has facts to work from rather than a guess. This is the one step in M4
T3/T3a allowed to touch the GPU (`ioplace/drivers/run_placement.py`'s GP
path runs on CUDA when the config says so).

HPWL here is computed directly (per-net pin bbox sum via
`ioplace.netlist.pin_positions`), *not* via `ioplace.evaluator_ref.
evaluate()` -- `evaluate()`'s per-net loop also builds an MST + walks the
region grid for IO/FT/Steiner stats, which on `mempool_tile_wrap` (145,589
nets) is the dominant cost (~tens of seconds *per call*) and would blow the
design draft's 120s per-script timeout budget across the 4 HPWL calls (2
arms x initial/final) this probe needs. `probe_lambda_dist.py` is the
separate script (design draft M4-L10) that pays for one real `evaluate()`
call, reusing this probe's saved final positions instead of re-running GP.

Merges its results into the `results/m4/corpus/mempool_tile_wrap.json`
already written by `probe_corpus_stats.py` (adds a `warmstart` key; leaves
every other key alone) -- run this *after* `probe_corpus_stats.py`, not
instead of it. Also saves the `random_center_init_flag=1` (config default)
arm's final positions to `<case>_warmstart_on.npz` for `probe_lambda_dist.
py` to reuse.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_warmstart_sanity \\
        <config.json> <case_name>
"""
import json
import os
import sys
import time

import numpy as np

from ioplace.drivers.run_placement import _load_dreamplace, extract_final_positions
from ioplace.netlist import netlist_from_placedb, pin_positions

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"


def _fast_hpwl(placedb, node_x, node_y):
    """Sum of per-net (x-extent + y-extent) over each net's pins -- the
    plain half-perimeter wirelength, computed directly rather than via
    `evaluator_ref.evaluate()` (see module docstring)."""
    nl = netlist_from_placedb(placedb)
    px, py = pin_positions(nl, node_x, node_y)
    start, net2pin = nl.flat_net2pin_start, nl.flat_net2pin
    total = 0.0
    for net in range(nl.num_nets):
        s, e = start[net], start[net + 1]
        if e - s <= 1:
            continue
        idx = net2pin[s:e]
        xs, ys = px[idx], py[idx]
        total += float(xs.max() - xs.min()) + float(ys.max() - ys.min())
    return total


def _run_one_arm(config_json, random_center_init_flag):
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    params.random_center_init_flag = random_center_init_flag
    placedb.initialize(params)
    n_phys = placedb.num_physical_nodes
    input_x = np.array(placedb.node_x[:n_phys], dtype=np.float64)
    input_y = np.array(placedb.node_y[:n_phys], dtype=np.float64)
    input_hpwl = _fast_hpwl(placedb, input_x, input_y)

    # Mirrors `ioplace.drivers.run_placement._place()`, but captures
    # `pos[0]` right after `NonLinearPlace` construction -- i.e. before
    # `placer(...)` runs GP and mutates it in place -- so the coverage check
    # below is against the actual *GP-start* position, not the final one.
    # `NonLinearPlace` is only importable after `_load_dreamplace()` (via
    # `setup_dreamplace()`) has pushed `$DP/install` onto `sys.path`.
    import NonLinearPlace
    lr = params.global_place_stages[0]["learning_rate"]
    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    gp_start = placer.pos[0].data.clone().cpu().numpy()
    placer(params, placedb, lr)

    final_x, final_y = extract_final_positions(placer, placedb)
    final_hpwl = _fast_hpwl(placedb, final_x, final_y)

    # Coverage check: with the flag off, the *movable* segment of
    # DREAMPlace's own GP-start pos should equal placedb's input coords
    # exactly (`BasicPlace.py:271` copies `placedb.node_x` into
    # `init_pos[0:num_physical_nodes]` unconditionally, and only overwrites
    # `[0:num_movable_nodes]` when the flag is on, `BasicPlace.py:272-288`).
    nm = placedb.num_movable_nodes
    n_nodes = len(gp_start) // 2  # `pos[0]` layout: `[x_all | y_all]`, each `num_nodes` long.
    gp_start_x, gp_start_y = gp_start[0:n_phys], gp_start[n_nodes:n_nodes + n_phys]
    covers_input = bool(
        np.allclose(gp_start_x[:nm], input_x[:nm]) and np.allclose(gp_start_y[:nm], input_y[:nm]))

    return dict(
        random_center_init_flag=random_center_init_flag,
        runtime_s=time.time() - t0,
        input_hpwl=input_hpwl, final_hpwl=final_hpwl,
        covers_input_coords=covers_input,
    ), (final_x, final_y)


def run(config_json):
    on_result, on_final_pos = _run_one_arm(config_json, random_center_init_flag=1)
    off_result, _off_final_pos = _run_one_arm(config_json, random_center_init_flag=0)
    return dict(warmstart=dict(on=on_result, off=off_result)), on_final_pos


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json> <case_name>")
    cfg = os.path.abspath(sys.argv[1])
    case = sys.argv[2]
    addition, (on_final_x, on_final_y) = run(cfg)

    out_path = os.path.join(REPO, "results", "m4", "corpus", f"{case}.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(addition)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=1, sort_keys=True)

    npz_path = os.path.join(REPO, "results", "m4", "corpus", f"{case}_warmstart_on.npz")
    np.savez_compressed(npz_path, node_x=on_final_x, node_y=on_final_y)

    on, off = addition["warmstart"]["on"], addition["warmstart"]["off"]
    print(f"[probe_warmstart_sanity] wrote {out_path} and {npz_path}: "
          f"on(input={on['input_hpwl']:.4e} final={on['final_hpwl']:.4e} "
          f"covers_input={on['covers_input_coords']} t={on['runtime_s']:.1f}s) "
          f"off(input={off['input_hpwl']:.4e} final={off['final_hpwl']:.4e} "
          f"covers_input={off['covers_input_coords']} t={off['runtime_s']:.1f}s)")
