"""M4 T0 probe (design draft `docs/superpowers/specs/2026-08-13-m4-scale-
up-design-draft.md` sec 1.3 / sec 5.3 layer 2): parametrizes
`ioplace.diagnostics.spike_10m.run()` to any (n_nodes, n_nets, K) scale --
the same synthetic-topology `IoTerm` fwd+bwd spike M2 froze the chunked-k
interface on at 10M, re-run at 30M for the design draft's "84 B/pin,
K-independent" chunked-memory-contract confirmation, and also adds host
peak RSS (spike_10m.run() itself only reports GPU figures).

Migrated from `/tmp/probe_m4_spike30.py` (design draft sec 11): sealed
(repo-relative paths only, no /tmp dependency) and now writes its JSON
directly to `results/m4/probes/` instead of `print("SPIKE " + ...)` to
stdout. `--budget-gb` is threaded through to `spike_10m.run()` (M4 T1's B3
fix, `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec
1.4) so this probe can be pointed at a 30M-scale budget without the
10M-sized 8.0 GB default misjudging it.

T0 only verifies this at a small scale (well under the 10M/30M spike sizes
the design draft measured -- those runs take from under a minute to
several minutes and are deferred to later M4 experiments per T0's scope).

Usage:
    PYTHONPATH=src $PY -m ioplace.diagnostics.probes_m4.probe_spike_scale \\
        --n-nodes 200000 --n-nets 240000 --K 16
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import torch

from ioplace.diagnostics.spike_10m import run as spike_run, DEGREE_DIST_JSON
from ioplace.profile import host_rss_gb

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


def _env_metadata(input_relpaths):
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "dp_commit": _git_head(DP),
        "ioplace_commit": _git_head(REPO),
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in input_relpaths},
    }


def run(n_nodes, n_nets, K, budget_gb=8.0) -> dict:
    r = spike_run(n_nodes=n_nodes, n_nets=n_nets, n_pins=int(n_nets * 3.5), K=K,
                  budget_gb=budget_gb)
    r["host_rss_gb"] = host_rss_gb()
    r["env"] = _env_metadata([DEGREE_DIST_JSON])
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=10_000_000)
    ap.add_argument("--n-nets", type=int, default=12_000_000)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--budget-gb", type=float, default=8.0)
    args = ap.parse_args()

    result = run(args.n_nodes, args.n_nets, args.K, budget_gb=args.budget_gb)
    out_path = os.path.join(
        REPO, "results", "m4", "probes",
        f"probe_spike_scale__{args.n_nodes}_{args.n_nets}_k{args.K}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe_spike_scale] wrote {out_path}")
