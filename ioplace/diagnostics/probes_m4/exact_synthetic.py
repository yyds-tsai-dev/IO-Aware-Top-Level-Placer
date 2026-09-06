"""Deterministic exact-count synthetic Netlist generator."""
import math
import numpy as np
from ioplace.netlist import Netlist

def exact_synthetic(n_nodes, n_nets, n_pins, seed=0, layout="local",
                    die=(0., 0., 100000., 100000.)):
    vals = (n_nodes, n_nets, n_pins)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0 for v in vals):
        raise ValueError("n_nodes, n_nets, and n_pins must be positive integers")
    if layout not in ("local", "uniform"):
        raise ValueError("layout must be 'local' or 'uniform'")
    if any(v > np.iinfo(np.int32).max for v in vals):
        raise ValueError("counts exceed int32")
    base, rem = divmod(n_pins, n_nets)
    if base < 2 or base + (rem > 0) >= 100 or base + (rem > 0) > n_nodes:
        raise ValueError("requested degree violates 2 <= degree < 100 or node capacity")
    deg = np.full(n_nets, base, dtype=np.int32); deg[:rem] += 1
    start = np.empty(n_nets + 1, dtype=np.int32); start[0] = 0
    start[1:] = np.cumsum(deg, dtype=np.int64).astype(np.int32)
    pin2net = np.repeat(np.arange(n_nets, dtype=np.int32), deg)
    local_i = np.arange(n_pins, dtype=np.int64) - np.repeat(start[:-1].astype(np.int64), deg)
    rng = np.random.default_rng(seed)
    offsets = np.repeat(rng.integers(0, n_nodes, size=n_nets, dtype=np.int64), deg)
    pin2node = ((offsets + local_i) % n_nodes).astype(np.int32)
    xl, yl, xh, yh = map(float, die)
    if not all(math.isfinite(v) for v in (xl,yl,xh,yh)) or not (xh > xl and yh > yl):
        raise ValueError("die must have finite positive extent")
    if layout == "uniform":
        x = rng.uniform(xl, xh, n_nodes); y = rng.uniform(yl, yh, n_nodes)
    else:
        side = math.ceil(math.sqrt(n_nodes)); ix = np.arange(n_nodes) % side; iy = np.arange(n_nodes) // side
        x = xl + (ix + .5) * (xh-xl) / side; y = yl + (iy + .5) * (yh-yl) / side
    z = np.ones(n_nodes, dtype=np.float64)
    return Netlist(x.astype(np.float64), y.astype(np.float64), z, z.copy(), n_nodes, 0, 0,
                   np.zeros(n_pins, dtype=np.float64), np.zeros(n_pins, dtype=np.float64),
                   pin2node, pin2net, np.arange(n_pins, dtype=np.int32), start,
                   xl, yl, xh, yh)
