"""Small source-only compatibility helper; never reads a replicated PlaceDB."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def native_pin_selection(degrees, pin2node, name2local, pin_names=None):
    if np.any(np.asarray(degrees) < 0) or int(np.sum(degrees)) != len(pin2node):
        raise ValueError("raw net degrees do not match the pin array")
    if np.any((np.asarray(pin2node) < 0) | (np.asarray(pin2node) >= len(name2local))):
        raise ValueError("raw pin node index outside the source tile")
    source = Path(__file__).with_suffix(".cpp")
    compiler = os.environ.get("CXX", "g++")
    identity = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    digest = hashlib.sha256(source.read_bytes() + identity.encode()).hexdigest()
    cache = Path(tempfile.gettempdir()) / f"ioplace-native-sort-{os.getuid()}"
    cache.mkdir(exist_ok=True)
    library = cache / (digest + ".so")
    if not library.exists():
        temporary = cache / f"{digest}.{os.getpid()}.so"
        subprocess.run([compiler, "-std=c++17", "-O3", "-shared", "-fPIC",
                        str(source), "-o", str(temporary)], check=True, capture_output=True)
        os.replace(temporary, library)
    ranks = np.empty(len(name2local), dtype=np.int64)
    for rank, name in enumerate(sorted(name2local, key=lambda v: v.encode("utf-8"))):
        ranks[name2local[name]] = rank
    starts = np.r_[0, np.cumsum(degrees)].astype(np.int64)
    nodes = np.ascontiguousarray(pin2node, dtype=np.int64)
    pin_ranks = np.zeros(len(nodes), dtype=np.int64)
    if pin_names:
        name_rank = {name: rank+1 for rank, name in enumerate(sorted(set(pin_names.values()), key=lambda v: v.encode("utf-8")))}
        for pin, name in pin_names.items():
            pin_ranks[pin] = name_rank[name]
    selected = np.empty(len(nodes), dtype=np.int64)
    out_starts = np.empty(len(starts), dtype=np.int64)
    lib = ctypes.CDLL(str(library))
    func = lib.select_native_pins
    vector = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    func.argtypes = [ctypes.c_int64] + [vector] * 6
    func.restype = ctypes.c_int64
    count = func(len(degrees), starts, nodes, ranks, pin_ranks, selected, out_starts)
    return selected[:count].copy(), np.diff(out_starts), dict(
        compiler=identity, helper_sha256=digest, policy="std_sort_node_name_pin_name_unique_node")


def native_net_pin_order(degrees, pin_is_output):
    """Match createPin: append each pin, then swap front/back for each output.

    Global pin arrays stay in canonical lexical order. Only the net's pin
    traversal changes; repeated outputs form a rotation, not a stable partition.
    """
    degrees = np.asarray(degrees, dtype=np.int64)
    output = np.asarray(pin_is_output, dtype=bool)
    if np.any(degrees < 0) or output.ndim != 1 or int(degrees.sum()) != len(output):
        raise ValueError("canonical degrees do not match output-pin flags")
    order = np.arange(len(output), dtype=np.int64)
    positions = np.flatnonzero(output)
    if not len(positions):
        return order
    starts = np.r_[0, np.cumsum(degrees)]
    nets = np.searchsorted(starts, positions, side="right") - 1
    first = np.r_[True, nets[1:] != nets[:-1]]
    last = np.r_[nets[1:] != nets[:-1], True]
    previous = np.r_[0, positions[:-1]]
    order[positions] = np.where(first, starts[nets], previous)
    order[starts[nets[last]]] = positions[last]
    return order


def round_away(values):
    values = np.asarray(values)
    return np.copysign(np.floor(np.abs(values) + .5), values)


def normalize_geometry(width, height, orientation, fixed, pin_nodes, raw_x, raw_y):
    """Native order: round center offsets, then rotate/reflect fixed nodes only."""
    width, height = np.asarray(width), np.asarray(height)
    orientation = np.where(fixed, orientation, "N")
    rotated = np.isin(orientation, ["E", "W", "FE", "FW"])
    stored_w, stored_h = np.where(rotated, height, width), np.where(rotated, width, height)
    w, h = width[pin_nodes], height[pin_nodes]
    u, v = round_away(raw_x+w/2), round_away(raw_y+h/2)
    x, y = u.copy(), v.copy()
    for orient, xx, yy in (("S", w-u, h-v), ("E", v, w-u), ("W", h-v, u),
                            ("FN", w-u, v), ("FS", u, h-v), ("FE", h-v, w-u), ("FW", v, u)):
        mask = orientation[pin_nodes] == orient
        x[mask], y[mask] = xx[mask], yy[mask]
    return stored_w, stored_h, x, y
