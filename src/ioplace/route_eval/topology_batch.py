"""Native OpenMP batching for raw FLUTE trees."""
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
    root = Path(os.environ.get(
        "IOPLACE_FLUTE_SOURCE",
        str(Path(REPO_ROOT) / "third_party/DREAMPlace/thirdparty/flute")))
    source = Path(__file__).with_name("flute_batch.cpp")
    files = [source, root / "flute.cpp", root / "flute.hpp",
             root / "lut.ICCAD2015/POWV9.dat", root / "lut.ICCAD2015/POST9.dat"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    compiler = os.environ.get("CXX", "g++")
    version = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256((str(hashes) + version.decode() + "-fopenmp").encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"ioplace-flute-batch-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    target = directory / (key + ".so")
    with _LOCK:
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".so") as temporary:
                subprocess.run([
                    compiler, "-std=c++17", "-O3", "-fPIC", "-shared", "-fopenmp",
                    "-I", str(root), str(source), str(files[1]), "-o", temporary.name,
                ], check=True)
                pending = Path(temporary.name + ".ready")
                pending.write_bytes(Path(temporary.name).read_bytes())
                pending.replace(target)
        library = ctypes.CDLL(str(target))
        library.ioplace_flute_batch_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        library.ioplace_flute_batch_init.restype = None
        library.ioplace_flute_batch_init_count.argtypes = []
        library.ioplace_flute_batch_init_count.restype = ctypes.c_int
        library.ioplace_flute_batch_init(os.fsencode(files[3]), os.fsencode(files[4]))
    library.ioplace_flute_batch.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    library.ioplace_flute_batch.restype = ctypes.c_int
    provenance = {
        "backend": "native-openmp-flute", "source_sha256": hashes,
        "compiler": version.decode().splitlines()[0],
        "library_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "openmp": True,
        "lut_initializations": library.ioplace_flute_batch_init_count(),
    }
    return library, provenance


def batch_flute_trees(pins, starts, coordinate_scale=1000., accuracy=3, threads=8):
    """Build packed raw FLUTE trees for degree-2..256 nets."""
    pins = np.ascontiguousarray(pins, dtype=np.float64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    if pins.ndim != 2 or pins.shape[1] != 2 or not np.isfinite(pins).all():
        raise ValueError("finite (P,2) pins required")
    if starts.ndim != 1 or len(starts) < 1 or starts[0] != 0 or starts[-1] != len(pins) \
            or np.any(np.diff(starts) < 0):
        raise ValueError("valid monotonic starts CSR required")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("positive finite coordinate scale required")
    if not isinstance(accuracy, (int, np.integer)) or not 1 <= int(accuracy) <= 10:
        raise ValueError("accuracy must be in [1,10]")
    if not isinstance(threads, (int, np.integer)) or int(threads) < 1:
        raise ValueError("positive integer threads required")
    degrees = np.diff(starts)
    if len(degrees) and (np.any(degrees < 2) or np.any(degrees > 256)):
        raise ValueError("FLUTE degree must be in [2,256]")
    branch_starts = np.empty(len(starts), dtype=np.int64)
    branch_starts[0] = 0
    if len(degrees):
        np.cumsum(2 * degrees - 2, out=branch_starts[1:])
    row_count = int(branch_starts[-1])
    positions = np.empty((row_count, 2), dtype=np.float64)
    parents = np.empty(row_count, dtype=np.int64)
    terminal_nodes = np.empty(len(pins), dtype=np.int64)
    library, base_provenance = _library()
    error_net = ctypes.c_int64(-1)
    code = library.ioplace_flute_batch(
        len(degrees), pins.ctypes.data, starts.ctypes.data, float(coordinate_scale),
        int(accuracy), int(threads), branch_starts.ctypes.data,
        positions.ctypes.data, parents.ctypes.data, terminal_nodes.ctypes.data,
        ctypes.byref(error_net))
    if code:
        messages = {-1: "invalid native batch arguments", -2: "invalid FLUTE degree",
                    -3: "non-finite pin coordinate", -4: "FLUTE int32 tree range exceeded"}
        raise ValueError(f"{messages.get(code, 'native FLUTE failure')} at net {error_net.value}")
    if len(terminal_nodes) and np.any(terminal_nodes < 0):
        raise RuntimeError("FLUTE terminal mapping failed")
    tree = {"positions": positions, "parents": parents,
            "terminal_nodes": terminal_nodes, "branch_starts": branch_starts}
    provenance = {**base_provenance, "coordinate_scale": float(coordinate_scale),
                  "accuracy": int(accuracy), "threads": int(threads),
                  "net_count": len(degrees), "pin_count": len(pins),
                  "raw_row_count": row_count}
    return tree, provenance
