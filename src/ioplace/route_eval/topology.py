"""FLUTE topology and geometric union accounting, separate from router prediction.

FLUTE works on quantized coordinates; terminal stubs reconnect the exact input
pins. Missing source/LUTs fail explicitly, never silently fall back to MST.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import numpy as np

_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _library():
    from ioplace.paths import REPO_ROOT
    root = Path(os.environ.get("IOPLACE_FLUTE_SOURCE", str(Path(REPO_ROOT) /
                "third_party/DREAMPlace/thirdparty/flute")))
    source = Path(__file__).with_name("flute_bridge.cpp")
    files = [source, root / "flute.cpp", root / "flute.hpp",
             root / "lut.ICCAD2015/POWV9.dat", root / "lut.ICCAD2015/POST9.dat"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    compiler = os.environ.get("CXX", "g++")
    version = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256((str(hashes) + version.decode()).encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"ioplace-flute-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    target = directory / (key + ".so")
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".so") as temporary:
            subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
                "-I", str(root), str(source), str(files[1]), "-o", temporary.name], check=True)
            # Copy via replace while keeping NamedTemporaryFile cleanup valid.
            pending = Path(temporary.name + ".ready")
            pending.write_bytes(Path(temporary.name).read_bytes())
            pending.replace(target)
    lib = ctypes.CDLL(str(target))
    lib.ioplace_flute_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ioplace_flute_init.restype = None
    lib.ioplace_flute.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p]
    lib.ioplace_flute.restype = ctypes.c_int
    lib.ioplace_flute_init(os.fsencode(files[3]), os.fsencode(files[4]))
    return lib, {"source_sha256": hashes, "compiler": version.decode().splitlines()[0],
                 "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def flute_edges(pins, *, coordinate_scale=1000., accuracy=3):
    """Return geometric tree edges (E,2,2), reconnecting quantized terminals.

    Supports <=256 distinct quantized pins. Coordinates are translated before
    quantizing and the worst-case tree sum is checked against FLUTE's int32.
    This is a CPU diagnostic backend, not a batched GPU implementation.
    """
    pins = np.asarray(pins, dtype=np.float64)
    if pins.ndim != 2 or pins.shape[1] != 2 or not np.isfinite(pins).all():
        raise ValueError("finite (N,2) pins required")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0 or not 1 <= accuracy <= 10:
        raise ValueError("positive finite scale and accuracy in [1,10] required")
    if len(pins) < 2:
        return np.empty((0, 2, 2)), {"topology": "flute", "coordinate_scale": coordinate_scale}
    origin = pins.min(axis=0)
    rounded = np.rint((pins-origin) * coordinate_scale)
    degree = len(np.unique(rounded, axis=0))
    if degree > 256 or rounded.max() * max(2*degree, 2) >= np.iinfo(np.int32).max:
        raise ValueError("FLUTE degree or int32 tree-length range exceeded")
    unique = np.unique(rounded.astype(np.int32), axis=0)
    edges = []
    provenance = {}
    if degree >= 2:
        x, y = (np.ascontiguousarray(unique[:, i]) for i in range(2))
        output = np.empty((2*degree-2, 3), dtype=np.int32)
        with _LOCK:
            lib, provenance = _library()
            code = lib.ioplace_flute(degree, x.ctypes.data, y.ctypes.data,
                                     accuracy, output.ctypes.data)
        if code or np.any(output[:, 2] < 0) or np.any(output[:, 2] >= len(output)):
            raise RuntimeError("invalid FLUTE tree")
        locations = output[:, :2] / coordinate_scale + origin
        for i, parent in enumerate(output[:, 2]):
            if not np.array_equal(locations[i], locations[parent]):
                edges.append([locations[i], locations[parent]])
    snapped = rounded / coordinate_scale + origin
    edges.extend([p, q] for p, q in zip(pins, snapped) if not np.array_equal(p, q))
    return np.asarray(edges, dtype=float).reshape(-1, 2, 2), {
        **provenance, "topology": "flute", "coordinate_scale": coordinate_scale,
        "accuracy": accuracy, "unique_quantized_pins": degree,
        "max_terminal_snap": float(np.abs(snapped-pins).max())}


def flute_tree(pins, *, coordinate_scale=1000., accuracy=3):
    """Return FLUTE's raw nodes while preserving physical terminal identity.

    Unlike :func:`flute_edges`, this representation retains zero-length
    terminal-to-Steiner branches.  FLUTE sorts its first ``degree`` terminal
    rows, so ``terminal_nodes`` maps each original physical pin to its raw row.
    Duplicate physical pins remain distinct terminals.
    """
    pins = np.asarray(pins, dtype=np.float64)
    if pins.ndim != 2 or pins.shape[1] != 2 or not np.isfinite(pins).all():
        raise ValueError("finite (N,2) pins required")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0 or not 1 <= accuracy <= 10:
        raise ValueError("positive finite scale and accuracy in [1,10] required")
    degree = len(pins)
    if degree < 2:
        snapped = pins.copy()
        return {
            "positions": snapped.copy(),
            "parents": np.arange(degree, dtype=np.int32),
            "terminal_nodes": np.arange(degree, dtype=np.int32),
            "snapped_pins": snapped,
        }, {"topology": "flute", "coordinate_scale": coordinate_scale,
            "accuracy": accuracy, "unique_quantized_pins": degree,
            "physical_pins": degree, "max_terminal_snap": 0.}
    origin = pins.min(axis=0)
    rounded = np.rint((pins - origin) * coordinate_scale)
    if degree > 256 or rounded.max() * max(2 * degree, 2) >= np.iinfo(np.int32).max:
        raise ValueError("FLUTE degree or int32 tree-length range exceeded")
    integer = rounded.astype(np.int32)
    x, y = (np.ascontiguousarray(integer[:, i]) for i in range(2))
    output = np.empty((2 * degree - 2, 3), dtype=np.int32)
    with _LOCK:
        lib, provenance = _library()
        code = lib.ioplace_flute(degree, x.ctypes.data, y.ctypes.data,
                                 accuracy, output.ctypes.data)
    if code or np.any(output[:, 2] < 0) or np.any(output[:, 2] >= len(output)):
        raise RuntimeError("invalid FLUTE tree")

    # Raw terminal slots are sorted by FLUTE, rather than retaining input order.
    # Queues preserve the multiplicity of duplicate physical terminals.
    slots = {}
    for slot, xy in enumerate(output[:degree, :2]):
        slots.setdefault(tuple(map(int, xy)), []).append(slot)
    terminal_nodes = np.empty(degree, dtype=np.int32)
    for original, xy in enumerate(integer):
        queue = slots.get(tuple(map(int, xy)))
        if not queue:
            raise RuntimeError("FLUTE terminal mapping is inconsistent")
        terminal_nodes[original] = queue.pop(0)
    if any(slots.values()):
        raise RuntimeError("FLUTE terminal mapping is incomplete")

    snapped = rounded / coordinate_scale + origin
    positions = output[:, :2] / coordinate_scale + origin
    return {
        "positions": positions,
        "parents": output[:, 2].copy(),
        "terminal_nodes": terminal_nodes,
        "snapped_pins": snapped,
    }, {
        **provenance, "topology": "flute", "coordinate_scale": coordinate_scale,
        "accuracy": accuracy, "unique_quantized_pins": len(np.unique(integer, axis=0)),
        "physical_pins": degree,
        "max_terminal_snap": float(np.abs(snapped - pins).max()),
    }


def segment_union(polylines):
    """Merge collinear overlapping intervals per net; preserve distinct tracks."""
    groups = {}
    for line in np.asarray(polylines, dtype=float):
        for a, b in zip(line[:-1], line[1:]):
            if np.array_equal(a, b):
                continue
            if not np.isfinite([a, b]).all() or (a[0] != b[0] and a[1] != b[1]):
                raise ValueError("finite rectilinear segments required")
            axis = 0 if a[1] == b[1] else 1
            key = (axis, a[1-axis])
            groups.setdefault(key, []).append(sorted((a[axis], b[axis])))
    result = []
    for (axis, track), intervals in sorted(groups.items()):
        merged = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(high, merged[-1][1])
            else:
                merged.append([low, high])
        for low, high in merged:
            a, b = [0., 0.], [0., 0.]
            a[axis], b[axis] = low, high
            a[1-axis] = b[1-axis] = track
            result.append([a, b])
    return np.asarray(result, dtype=float).reshape(-1, 2, 2)


def union_metrics(region_grid, polylines):
    from ioplace.evaluator_ref import _walk_segment
    segments = segment_union(polylines)
    passed, pairs = set(), []
    io = sum(_walk_segment(region_grid, *a, *b, passed, pairs) for a, b in segments)
    return {"crossings": int(io), "wirelength": float(np.abs(np.diff(segments, axis=1)).sum()),
            "segments": segments, "passed_regions": passed, "boundary_pairs": pairs}
