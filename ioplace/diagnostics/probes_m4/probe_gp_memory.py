"""M4 T0 probe (design draft `docs/superpowers/specs/2026-08-13-m4-scale-
up-design-draft.md` sec 1.1's full case table + GP memory fit
`GP_peak_bytes ~= 87*N_total + 73*N_pins + 160*n_bins`): `PlaceDB.read()` +
`initialize()` + DREAMPlace GP+LG (via `ioplace.drivers.run_placement`'s
`_load_dreamplace`/`_place`, the same entry points the real drivers use),
timing each phase and recording host peak RSS and GPU peak alloc/reserved.

Migrated and merged from `/tmp/probe_m4_mem.py` (GP peak, no host RSS) and
`/tmp/probe_m4_full.py` (adds host RSS + try/except OOM handling) -- design
draft sec 11 maps both onto this one target file. Sealed (repo-relative
paths only, no /tmp dependency); writes its JSON directly to
`results/m4/probes/` (DREAMPlace's own INFO/WARNING log lines would
otherwise interleave with a `> out.json` shell redirect -- same convention
as `ioplace/diagnostics/probes_m3/probe_free_area_util.py`).

Per T0's scope, only re-verified here on `adaptec1` (small, config already
exists, ~15-30s). The design draft's `mempool_group`/`mempool_cluster` rows
(minutes to ~6 minutes) and the 6.2M/12.3M/27.7M tiler-synthesized rows are
explicitly deferred to later M4 experiments (T6/T8), not T0 -- re-running
them needs either a NanGate45 config this repo doesn't have yet (M4 T3) or
the tiler (M4 T4), neither of which exists yet.

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_gp_memory <config.json>
"""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace, _place

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
    out = {"config": config_json, "rss_start_gb": _rss_gb()}
    t0 = time.time()
    try:
        params, placedb = _load_dreamplace(config_json)
        params.random_seed = 1000
        params.deterministic_flag = 1
        out["read_s"] = time.time() - t0
        out["rss_after_read_gb"] = _rss_gb()

        t1 = time.time()
        placedb.initialize(params)
        out["init_s"] = time.time() - t1
        out.update(
            num_physical=int(placedb.num_physical_nodes),
            num_movable=int(placedb.num_movable_nodes),
            num_terminals=int(placedb.num_terminals),
            num_terminal_NIs=int(placedb.num_terminal_NIs),
            num_nets=int(placedb.num_nets),
            num_pins=int(len(placedb.pin2node_map)),
            num_filler=int(getattr(placedb, "num_filler_nodes", 0)),
            num_bins_x=int(placedb.num_bins_x), num_bins_y=int(placedb.num_bins_y),
            rss_after_init_gb=_rss_gb(),
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t2 = time.time()
        placer, metrics = _place(params, placedb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # metrics[-1] is a flat EvalMetrics only when legalize_flag=1 appends
        # one (NonLinearPlace.py:891-892); GP-only configs (legalize_flag=0,
        # e.g. the overflow-diagnosis probe) leave the nested per-stage list
        # structure, so unwrap trailing lists until we reach the object.
        _last_metric = metrics[-1]
        while isinstance(_last_metric, (list, tuple)):
            _last_metric = _last_metric[-1]
        out.update(
            gp_s=time.time() - t2,
            gp_peak_alloc_mb=(torch.cuda.max_memory_allocated() / 2**20
                              if torch.cuda.is_available() else 0.0),
            gp_peak_reserved_mb=(torch.cuda.max_memory_reserved() / 2**20
                                 if torch.cuda.is_available() else 0.0),
            # Overflow-diagnosis follow-up: final_overflow -- float() on a
            # 0-dim tensor already calls .item() implicitly, same pattern
            # ioplace/drivers/run_placement.py's run_flat/run_io use.
            # gp_iterations -- metrics is _place()'s `placer(params, placedb,
            # lr)` return value; under this repo's GP+LG protocol
            # (legalize_flag=1) NonLinearPlace.py:891-892 appends a *flat*
            # EvalMetrics object to it right after legalization, so
            # metrics[-1] is that object directly (verified against
            # $DP/install/dreamplace/NonLinearPlace.py's metrics structure).
            final_overflow=float(placer.model.overflow.max()),
            gp_iterations=int(_last_metric.iteration),
            rss_end_gb=_rss_gb(), ok=True,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        out.update(
            ok=False, error=f"{type(e).__name__}: {e}"[:400], rss_fail_gb=_rss_gb(),
            gpu_alloc_at_fail_mb=(torch.cuda.max_memory_allocated() / 2**20
                                  if torch.cuda.is_available() else 0.0),
        )
    out["env"] = _env_metadata([config_json])
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json>")
    cfg = os.path.abspath(sys.argv[1])
    result = run(cfg)
    case = os.path.splitext(os.path.basename(cfg))[0]
    out_path = os.path.join(REPO, "results", "m4", "probes", f"probe_gp_memory__{case}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_gp_memory] wrote {out_path}")
