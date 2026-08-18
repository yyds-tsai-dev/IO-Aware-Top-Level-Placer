"""M4 T3 corpus-ingestion stats (design draft `docs/superpowers/specs/2026-
08-13-m4-scale-up-design-draft.md` T3 row / sec 2.4 M4-L3): read-only
`PlaceDB.read()` timing + host peak RSS + die/area utilization for one of
the four T3 corpus configs (`benchmarks/ispd25/{mempool_tile_wrap,
mempool_group,mempool_cluster}.json`, `benchmarks/ispd2015_superblue12_m4.
json`), plus the sec 2.4 M4-L3 `target_density` reverse-derivation.

This is a sibling of `probe_host_rss.py` (same read-only `PlaceDB.read()` +
`area_util` measurement), kept separate because: (a) its output schema is
the T3-specific `results/m4/corpus/<case>.json` (nodes/nets/pins/die/util/
read_s/peak_rss/sha256), not `probe_host_rss.py`'s `results/m4/probes/
probe_host_rss__<case>.json`; (b) it adds the M4-L3 `target_density`
suggestion, which `probe_host_rss.py` (T0, `adaptec1`-only, no NanGate45
config to react to yet) intentionally does not compute.

M4-L3 target_density rule: DREAMPlace's own `PlaceDB.py` already has a
margin convention for when a *configured* `target_density` is too low for
the measured utilization -- the per-fence-region path
(`calc_num_filler_for_fence_region`, `PlaceDB.py:718-726`) computes
`target_density_fence_region = min(1, utilization + 0.01)` when
`target_density < utilization`, floored at `max(0.35, ...)`; the top-level
path (`PlaceDB.py:842-846`) does the analogous `min(cell_utilization, 1.0)`
override. Design draft sec 2.4 M4-L3 only points at the numerator/
denominator (`total_movable_node_area / free_area`) and leaves the margin
unspecified, so this probe reuses DREAMPlace's own margin convention rather
than inventing a new one: `suggested_target_density = min(1.0, max(0.35,
area_util + 0.01))`. This is a *suggestion* written into the stats JSON for
a human/T8b to review and bake into the config -- this probe does not
mutate the input config file itself.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_corpus_stats \\
        <config.json> <case_name>
"""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

from ioplace.dreamplace_env import setup_dreamplace

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def suggested_target_density(area_util):
    """sec 2.4 M4-L3 rule, see module docstring: reuse DREAMPlace's own
    `PlaceDB.py:718-726` margin (`min(1, utilization + 0.01)`, floored at
    0.35) rather than inventing a new margin."""
    return min(1.0, max(0.35, area_util + 0.01))


def run(config_json, case) -> dict:
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    # Configs hold absolute paths (this repo's own convention for the T3
    # corpus configs, unlike DREAMPlace's own test configs which are
    # relative to $DP/install) -- the chdir is still needed because
    # Params.load()/PlaceDB.read() resolve *relative* paths (e.g. the
    # ispd2015 superblue12 config reuses DP's own relative paths) against
    # cwd, same chdir bracket as run_placement.py's `_load_dreamplace`.
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))
    try:
        params.load(config_json)
        configured_target_density = float(params.target_density)

        t0 = time.time()
        db = PlaceDB.PlaceDB()
        db.read(params)
        t_read = time.time() - t0
        rss_after_read_gb = _rss_gb()

        t1 = time.time()
        db.initialize(params)
        t_init = time.time() - t1
        rss_after_init_gb = _rss_gb()
    finally:
        os.chdir(cwd)

    die = [float(db.xl), float(db.yl), float(db.xh), float(db.yh)]
    free_area = (db.xh - db.xl) * (db.yh - db.yl) - db.total_fixed_node_area
    area_util = float(db.total_movable_node_area / free_area) if free_area > 0 else None

    input_paths = [config_json]
    lef = params.lef_input
    input_paths += lef if isinstance(lef, list) else [lef]
    if params.def_input:
        input_paths.append(params.def_input)

    return dict(
        case=case,
        config=config_json,
        # Peak RSS: this probe does exactly one PlaceDB.read()+initialize()
        # per process invocation (never looped over multiple cases in the
        # same interpreter) specifically so `ru_maxrss` -- a process-
        # lifetime high-water mark that cannot be reset (design draft sec
        # 1.4 B1/6.1) -- is a clean, uncontaminated peak for *this* case.
        read_s=t_read, init_s=t_init,
        peak_rss_gb=rss_after_init_gb,
        rss_after_read_gb=rss_after_read_gb, rss_after_init_gb=rss_after_init_gb,
        num_nodes=int(db.num_physical_nodes), num_movable=int(db.num_movable_nodes),
        num_terminals=int(db.num_terminals), num_terminal_NIs=int(db.num_terminal_NIs),
        num_nets=int(db.num_nets), num_pins=int(len(db.pin2node_map)),
        num_filler=int(getattr(db, "num_filler_nodes", 0)),
        die=die, row_height=float(db.row_height), site_width=float(db.site_width),
        total_movable_area=float(db.total_movable_node_area),
        total_fixed_area=float(db.total_fixed_node_area),
        free_area=float(free_area),
        area_util=area_util,
        configured_target_density=configured_target_density,
        # sec 2.4 M4-L3.
        suggested_target_density=(suggested_target_density(area_util)
                                   if area_util is not None else None),
        sha256={os.path.abspath(p): _sha256(p) for p in input_paths},
        env=dict(
            dp_commit=_git_head(DP), ioplace_commit=_git_head(REPO),
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json> <case_name>")
    cfg = os.path.abspath(sys.argv[1])
    case = sys.argv[2]
    result = run(cfg, case)
    out_path = os.path.join(REPO, "results", "m4", "corpus", f"{case}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[probe_corpus_stats] wrote {out_path}: "
          f"num_movable={result['num_movable']} num_nets={result['num_nets']} "
          f"num_pins={result['num_pins']} area_util={result['area_util']} "
          f"suggested_target_density={result['suggested_target_density']} "
          f"read_s={result['read_s']:.1f} peak_rss_gb={result['peak_rss_gb']:.2f}")
