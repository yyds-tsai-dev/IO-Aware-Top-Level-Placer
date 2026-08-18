"""M4 T0 probe (design draft `docs/superpowers/specs/2026-08-13-m4-scale-
up-design-draft.md` sec 1.1 "次級事實 1" / sec 11's evidence appendix):
read-only PlaceDB movable-macro statistics -- how many movable nodes are
"macros" (row height > 4x the standard row height) and what fraction of
movable area they account for, plus average net degree.

Migrated from `/tmp/probe_m4_macro.py` (design draft sec 11): sealed
(repo-relative paths only, no /tmp dependency) and now writes its JSON
directly to `results/m4/probes/` instead of stdout (DREAMPlace's PlaceDB
loader writes its own INFO/WARNING lines to stdout, which would otherwise
interleave with a `> out.json` shell redirect -- same convention as
`ioplace/diagnostics/probes_m3/probe_m3_rg.py`).

Confirms the draft's sec 1.1 finding that the mempool_* (ISPD2025/NanGate45)
family has zero movable macros (SRAM is entirely in num_terminals) -- this
probe was only re-verified here on `adaptec1` (ISPD2005; no movable macros
there either, which is the expected/uninteresting case); the mempool_*
cases need a DREAMPlace config that doesn't exist yet (M4 T3's job) so
their re-run is deferred to that task, not T0.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_macro_stats <config.json>
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from ioplace.dreamplace_env import setup_dreamplace

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_abspaths):
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
        db = PlaceDB.PlaceDB()
        db.read(params)
    finally:
        os.chdir(cwd)

    nm = db.num_movable_nodes
    h = np.asarray(db.node_size_y[:nm])
    w = np.asarray(db.node_size_x[:nm])
    rh = db.row_height
    macro = h > 4 * rh

    return dict(
        config=config_json,
        num_movable=int(nm), row_height=float(rh),
        n_movable_macro=int(macro.sum()),
        movable_macro_area_frac=(float((w[macro] * h[macro]).sum() / (w * h).sum())
                                 if macro.any() else 0.0),
        max_h_in_rows=float(h.max() / rh) if nm > 0 else 0.0,
        num_terminals=int(db.num_terminals),
        num_terminal_NIs=int(db.num_terminal_NIs),
        avg_net_degree=float(len(db.pin2node_map) / db.num_nets),
        env=_env_metadata([config_json]),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json>")
    cfg = os.path.abspath(sys.argv[1])
    result = run(cfg)
    case = os.path.splitext(os.path.basename(cfg))[0]
    out_path = os.path.join(REPO, "results", "m4", "probes", f"probe_macro_stats__{case}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_macro_stats] wrote {out_path}")
