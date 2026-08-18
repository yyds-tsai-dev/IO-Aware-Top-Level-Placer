"""M4 T0 probe (design draft `docs/superpowers/specs/2026-08-13-m4-scale-
up-design-draft.md` sec 1.1's "host RSS" column / sec 2.4 M4-L3): read-only
`PlaceDB.read()`/`initialize()` timing and host peak RSS, plus die/area
utilization -- no GP, no evaluator. This is the probe behind the draft's
`host_RSS(PlaceDB.read) ~= 1.35-1.42 KB/pin (LEF/DEF) / 0.88 KB/pin
(Bookshelf)` model (sec 1.1) and the per-case `target_density` reverse-
derivation call-out (sec 2.4 M4-L3: `total_movable_node_area / free_area`).

Migrated from `/tmp/probe_m4_read.py` (design draft sec 11): sealed
(repo-relative paths only, no /tmp dependency) and now writes its JSON
directly to `results/m4/probes/` instead of stdout (same
direct-file-write convention as `ioplace/diagnostics/probes_m3/
probe_free_area_util.py` -- DREAMPlace's loader's own INFO/WARNING lines
would otherwise interleave with a `> out.json` shell redirect).

Only verified here on `adaptec1` (small, ISPD2005, config already exists);
the mempool_* (ISPD2025/NanGate45) cases from design draft sec 1.1's table
need a DREAMPlace config that doesn't exist in this repo yet (M4 T3's job
-- "產 benchmarks/ispd25/*.json config") so their re-run is deferred to
that task, not T0.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_host_rss <config.json>
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
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_abspaths):
    import numpy as np
    import torch
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "dp_commit": _git_head(DP),
        "ioplace_commit": _git_head(REPO),
        "input_sha256": {p: _sha256(p) for p in input_abspaths},
    }


def run(config_json) -> dict:
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    # Configs hold paths relative to $DP/install (same chdir bracket
    # ioplace.drivers.run_placement._load_dreamplace uses).
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))
    try:
        params.load(config_json)

        t0 = time.time()
        rss0 = _rss_gb()
        db = PlaceDB.PlaceDB()
        db.read(params)
        t_read = time.time() - t0
        rss1 = _rss_gb()

        t1 = time.time()
        db.initialize(params)
        t_init = time.time() - t1
        rss2 = _rss_gb()
    finally:
        os.chdir(cwd)

    free_area = (db.xh - db.xl) * (db.yh - db.yl) - db.total_fixed_node_area
    return dict(
        config=config_json,
        read_s=t_read, init_s=t_init,
        rss_before_gb=rss0, rss_after_read_gb=rss1, rss_after_init_gb=rss2,
        num_physical=int(db.num_physical_nodes), num_movable=int(db.num_movable_nodes),
        num_terminals=int(db.num_terminals), num_terminal_NIs=int(db.num_terminal_NIs),
        num_nets=int(db.num_nets), num_pins=int(len(db.pin2node_map)),
        num_filler=int(getattr(db, "num_filler_nodes", 0)),
        die=[float(db.xl), float(db.yl), float(db.xh), float(db.yh)],
        row_height=float(db.row_height), site_width=float(db.site_width),
        total_movable_area=float(db.total_movable_node_area),
        total_fixed_area=float(db.total_fixed_node_area),
        # sec 2.4 M4-L3: the denominator DREAMPlace's own target_density is
        # implicitly measured against -- utilization near/above 1.0 with the
        # ISPD2005-derived default target_density (0.835) means excess filler.
        area_util=float(db.total_movable_node_area / free_area) if free_area > 0 else None,
        env=_env_metadata([config_json]),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json>")
    cfg = os.path.abspath(sys.argv[1])
    result = run(cfg)
    case = os.path.splitext(os.path.basename(cfg))[0]
    out_path = os.path.join(REPO, "results", "m4", "probes", f"probe_host_rss__{case}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_host_rss] wrote {out_path}")
