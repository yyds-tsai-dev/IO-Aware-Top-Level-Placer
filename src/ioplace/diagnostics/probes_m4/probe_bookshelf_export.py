"""M4 T0 probe -- NOT migrated from `/tmp` (no such probe was left there;
written fresh per T0's instruction to reconstruct it from design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 3.1's
evidence 1/2):

  Evidence 1: DREAMPlace's own `PlaceDB.write(params, path,
  place_io.SolutionFileFormat.BOOKSHELFALL)` (`$DP/dreamplace/PlaceDB.py:
  1010-1033` -> `BookShelfWriter::writeAll`,
  `$DP/dreamplace/ops/place_io/src/BookshelfWriter.cpp:22-51,100-138`) can
  losslessly export ANY loaded PlaceDB -- LEF/DEF-origin or already
  Bookshelf-origin -- to a complete `.nodes/.nets/.wts/.pl/.scl/.shapes/
  .route/.aux` set, `.nets` including per-pin offsets relative to each
  cell's center. This probe actually exercises that writer (not just cites
  the code) and times it.

  Evidence 2: Bookshelf is >3x smaller than DEF (design draft's own
  bigblue4 numbers: .nets 345.3MB/8.73M pins = 39.6 B/pin; .nodes 15.0
  B/node; .pl 16.5 B/node -- measured off ISPD2005's *existing* Bookshelf
  files on disk, not a re-export). This probe recomputes the same
  B/pin-B/node ratios off its OWN fresh BOOKSHELFALL export, so the
  self-consistency of the design draft's ratios doesn't rely on trusting a
  file this repo never touches.

T0 scope: verified here on `adaptec1` (small, ISPD2005 Bookshelf-origin --
exercises the writer's *format* faithfully even though the source is
already Bookshelf; a genuine LEF/DEF -> Bookshelf conversion demo on
`mempool_tile_wrap` needs a DREAMPlace config this repo doesn't have yet,
M4 T3's job, so that re-run is deferred, not T0's). Output Bookshelf files
are written under `results/m4/probes/bookshelf_export/<case>/` and are NOT
committed (large, derived, reproducible -- only the summary JSON is meant
to be tracked); `.pl` is additionally covered by `.gitignore`'s
`results/**/*.pl`.

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_bookshelf_export <config.json>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"

# The BookShelfWriter::writeAll suffixes actually produced (BookshelfWriter.cpp
# writeAux/writeNodes/writeNets/writeWts/writeScl/writeShapes/writePlx/writeRoute).
_SUFFIXES = ("aux", "nodes", "nets", "wts", "pl", "scl", "shapes", "route")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(input_abspaths):
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "dp_commit": _git_head(DP),
        "ioplace_commit": _git_head(REPO),
        "input_sha256": {p: _sha256(p) for p in input_abspaths},
    }


def run(config_json, out_prefix) -> dict:
    params, placedb = _load_dreamplace(config_json)
    # _load_dreamplace() (via setup_dreamplace()) is what pushes $DP/install
    # onto sys.path -- only importable after that call, not before.
    import dreamplace.ops.place_io.place_io as place_io_mod
    params.random_seed = 1000
    params.deterministic_flag = 1
    placedb.initialize(params)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    t0 = time.time()
    placedb.write(params, out_prefix, place_io_mod.SolutionFileFormat.BOOKSHELFALL)
    write_s = time.time() - t0

    n_physical = int(placedb.num_physical_nodes)
    n_pins = int(len(placedb.pin2node_map))
    file_sizes = {}
    for suf in _SUFFIXES:
        p = f"{out_prefix}.{suf}"
        if os.path.exists(p):
            file_sizes[suf] = os.path.getsize(p)

    ratios = {}
    if "nodes" in file_sizes and n_physical > 0:
        ratios["bytes_per_node_nodes_file"] = file_sizes["nodes"] / n_physical
    if "nets" in file_sizes and n_pins > 0:
        ratios["bytes_per_pin_nets_file"] = file_sizes["nets"] / n_pins
    if "pl" in file_sizes and n_physical > 0:
        ratios["bytes_per_node_pl_file"] = file_sizes["pl"] / n_physical

    return dict(
        config=config_json, out_prefix=out_prefix, write_s=write_s,
        n_physical=n_physical, n_movable=int(placedb.num_movable_nodes),
        n_nets=int(placedb.num_nets), n_pins=n_pins,
        file_sizes_bytes=file_sizes, ratios=ratios,
        env=_env_metadata([config_json]),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json>")
    cfg = os.path.abspath(sys.argv[1])
    case = os.path.splitext(os.path.basename(cfg))[0]
    prefix = os.path.join(REPO, "results", "m4", "probes", "bookshelf_export", case, case)
    result = run(cfg, prefix)
    out_path = os.path.join(REPO, "results", "m4", "probes", f"probe_bookshelf_export__{case}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_bookshelf_export] wrote {out_path}")
