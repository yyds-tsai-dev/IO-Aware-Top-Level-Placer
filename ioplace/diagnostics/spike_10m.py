"""Task 2b Step 5: 10M-scale IoTerm feasibility spike.

design v2 sec 2.5's gate on the chunked-k interface freeze: "10M 級預估...
由 T2 Step 6 的 feasibility spike 在 L4 上以合成拓撲實測驗證;spike 不通過就不
得凍結 T2 介面。" No real netlist at this scale exists in the repo's test
data, so this synthesizes one: N=10M node positions uniform over a synthetic
die, and per-net degree resampled from bigblue4's *real* bucket ratios
(results/m2/probes/degree_distribution.json) so the connectivity shape is
production-like rather than uniform-random. bigblue4's own bucket-weighted
mean degree is ~4.1, not the nominal n_pins/n_nets = 40e6/12e6 ~= 3.33, so
the resampled pin count comes out close to but not exactly n_pins -- see
task-2b-report.md for the actual figure; n_pins is a target/label here, not
an enforced count (forcing an exact match would mean distorting the bucket
ratios, defeating the point of resampling from real data).

ok = peak_gb <= 8.0 and not OOM. A `false`/OOM result BLOCKS T2b -- it does
not proceed to T4 (design v2 sec 2.5).

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.spike_10m > results/m2/spike/spike_10m.json
"""
import json
import time

import numpy as np
import torch

from ioplace.drivers.run_placement import get_regions_for
from ioplace.netlist import Netlist
from ioplace.ops.soft_assign import rect_table
from ioplace.ops.io_term import build_net_node_csr, IoTerm, DEG_BUCKET_EDGES

DEGREE_DIST_JSON = "results/m2/probes/degree_distribution.json"
SEED = 0


def _synthesize_netlist(n_nodes, n_nets, die):
    """Vectorized synthetic netlist (no Python loop over nets/pins). Per-net
    degree: pick a bigblue4 degree bucket by its real proportion, then an
    integer degree uniformly within that bucket's [lo, hi) range. Pin->node
    assignment draws with replacement -- collisions are negligible at this
    N and are handled by build_net_node_csr's own dedup regardless."""
    rng = np.random.default_rng(SEED)
    bucket_counts = json.load(open(DEGREE_DIST_JSON))["bigblue4"]["bucket_counts"]
    probs = np.array(list(bucket_counts.values()), dtype=np.float64)
    probs /= probs.sum()
    lo = np.array(DEG_BUCKET_EDGES[:-1], dtype=np.int64)
    hi = np.array(DEG_BUCKET_EDGES[1:], dtype=np.int64)

    cdf = np.cumsum(probs)
    bucket = np.clip(np.searchsorted(cdf, rng.random(n_nets), side="right"), 0, len(probs) - 1)
    degrees = (lo[bucket] + rng.integers(0, hi[bucket] - lo[bucket], size=n_nets)).astype(np.int32)

    flat_net2pin_start = np.concatenate([[0], np.cumsum(degrees)]).astype(np.int32)
    total_pins = int(flat_net2pin_start[-1])

    pin2node = rng.integers(0, n_nodes, size=total_pins, dtype=np.int32)
    pin2net = np.repeat(np.arange(n_nets, dtype=np.int32), degrees)
    flat_net2pin = np.arange(total_pins, dtype=np.int32)

    xl, yl, xh, yh = die
    node_x = rng.uniform(xl, xh, n_nodes).astype(np.float32)
    node_y = rng.uniform(yl, yh, n_nodes).astype(np.float32)

    nl = Netlist(
        node_x=node_x, node_y=node_y,
        node_size_x=np.ones(n_nodes, dtype=np.float32), node_size_y=np.ones(n_nodes, dtype=np.float32),
        num_movable=n_nodes, num_terminals=0, num_terminal_NIs=0,
        pin_offset_x=np.zeros(total_pins, dtype=np.float32), pin_offset_y=np.zeros(total_pins, dtype=np.float32),
        pin2node=pin2node, pin2net=pin2net,
        flat_net2pin=flat_net2pin, flat_net2pin_start=flat_net2pin_start,
        xl=xl, yl=yl, xh=xh, yh=yh)
    return nl, total_pins


def run(n_nodes=10_000_000, n_nets=12_000_000, n_pins=40_000_000, K=32) -> dict:
    die = (0.0, 0.0, 100_000.0, 100_000.0)

    t0 = time.time()
    nl, total_pins = _synthesize_netlist(n_nodes, n_nets, die)
    synth_s = time.time() - t0

    rs = get_regions_for(die, K, "grid", SEED)
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)

    result = {"n_nodes": n_nodes, "n_nets": n_nets, "n_pins_target": n_pins,
              "n_pins_actual": total_pins, "K": K, "synth_s": synth_s, "ok": False}
    torch.cuda.reset_peak_memory_stats()
    try:
        term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=K,
                      num_movable=nl.num_movable, num_physical=nl.num_physical,
                      num_nodes=nl.num_physical, device="cuda")
        pos = torch.cat([torch.as_tensor(nl.node_x), torch.as_tensor(nl.node_y)]
                       ).cuda().requires_grad_(True)
        L_R = ((die[2] - die[0]) * (die[3] - die[1]) / K) ** 0.5
        tau = 0.1 * L_R

        torch.cuda.synchronize(); t0 = time.time()
        L = term(pos, tau, 1.0)
        torch.cuda.synchronize(); t1 = time.time()
        L.backward()
        torch.cuda.synchronize(); t2 = time.time()

        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        result.update({
            "fwd_ms": 1000 * (t1 - t0), "bwd_ms": 1000 * (t2 - t1),
            "peak_gb": peak_gb, "k_chunk": term.k_chunk, "n_active": term.n_active,
            "loss": float(L.detach()), "ok": bool(peak_gb <= 8.0),
        })
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        result.update({"peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                       "oom": True, "ok": False, "error": str(e)})
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
