"""M4 T9 replication-constructor `ioplace.netlist.Netlist` builder (design
draft `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec
5.3 layer 2 / T9 row): builds the 27.7M-node 3x3 array's `Netlist` directly
from `mempool_group`'s single-tile Bookshelf export + numpy translation/
replication, WITHOUT ever parsing the 16.7GB 3x3 `.nets` file DREAMPlace's
own `PlaceDB.read` would have to (T9 must not go through `PlaceDB` at all --
sec 5.3's "不經 PlaceDB").

**Verified fact this module is built on** (T6/T7's own artifacts, see
`results/m4/bench/arrays/3x3_n2/3x3_n2.manifest.json`): the 3x3 array is
*exactly* `tile_bookshelf.tile()`'s 9x replication of the single source
tile (block order i-major/j-minor, `t{i}_{j}/` name prefix, coordinate
translation `(i*W, j*H)`) plus a **contiguous tail** of degree-2 glue nets
appended by `glue_gen.append_glue_nets` (`NetDegree : 2 glue{k}` followed by
two `t{i}_{j}/{node} I : 0 0` pin lines, `k` in `0..glue.n_nets-1`, strictly
in file order at the very end of `.nets`). That structure is what makes
in-memory replication possible: everything derived from the *source* tile
(one ~12s / ~0.7GB parse, per T0's `probe_bookshelf_export.py` timing) is
pure numpy translate/tile/concatenate; the ONLY new information in the 3x3
array beyond 9x the source is the glue tail, which is small (~59k nets /
~119k pins for the primary 3x3_n2 array) and is read via a bounded tail-seek
of `.nets`, never a full scan.

**Movable-first repermutation (required, not optional):** the source tile's
own `.nodes` is already movable-first (verified: `mempool_group_export`'s
first 3,077,669 records have no 4th token, the next 11,742 all do) --
but replicating it tile-major (`t_idx = i*C+j`, `[tile0 movable][tile0
term.][tile1 movable][tile1 term.]...`) is NOT globally movable-first, and
both `Netlist.num_movable` (a prefix count) and `IoTerm`'s backward pass
(`gx[meta.num_movable:] = 0.0`, `ioplace/ops/io_term.py`) assume it is. This
module stably orders movable cells, regular fixed terminals, then terminal_NIs,
matching native PlaceDB, and remaps all pin references. Cache schema 2 also
converts center-relative Bookshelf pin offsets into native lower-left offsets
(integer rounding with ties away from zero). Legacy caches must be rebuilt.

Public API:
    build_tiled_netlist_cache(manifest_path, out_dir) -> dict   # one-time
    load_tiled_netlist(cache_dir, mmap=True) -> (Netlist, meta)
    verify_against_bookshelf(cache_dir, prefix, mode="full"|"window") -> dict

The constructor and source verifier are CPU-only. Native regression tests use
the uninitialized PlaceDB reader in raw Bookshelf units; initialization would
introduce site-width scaling and invalidate a direct coordinate comparison.
"""
import json
import os
import re
import time

import numpy as np

from ioplace.bench import tile_bookshelf as tb
from ioplace.netlist import Netlist
from ioplace.bench.native_pin_order import native_pin_selection, normalize_geometry, round_away

# Cache array files written by build_tiled_netlist_cache / read by
# load_tiled_netlist. dtypes match ioplace.netlist.netlist_from_placedb's
# own convention exactly (float64 positions/sizes/offsets, int32 index
# arrays) so a Netlist built here is indistinguishable in dtype from one
# built off a real PlaceDB.
_FLOAT_ARRAYS = ("node_x", "node_y", "node_size_x", "node_size_y",
                 "pin_offset_x", "pin_offset_y")
_INT32_ARRAYS = ("pin2node", "pin2net", "flat_net2pin", "flat_net2pin_start")
# inv_perm is int64 (indexes up to n_nodes_total, kept wide for headroom --
# it is only ever used by verify_against_bookshelf, not by load_tiled_netlist/
# the hot GPU path, so the extra 4 bytes/node is not on any memory-critical
# path).
_EXTRA_ARRAYS = ("inv_perm",)

_TILE_PREFIX_RE = re.compile(r"^t(\d+)_(\d+)/(.+)$")


def _split_tile_prefix(full_name):
    m = _TILE_PREFIX_RE.match(full_name)
    if not m:
        raise ValueError(f"name {full_name!r} does not match the t{{i}}_{{j}}/... "
                         "tile-prefix convention (tile_bookshelf.tile's own naming)")
    return int(m.group(1)), int(m.group(2)), m.group(3)


# ---------------------------------------------------------------------------
# single-tile (source) Bookshelf parser
# ---------------------------------------------------------------------------

class SourceTile:
    """In-memory parse of one single-tile Bookshelf design (numpy arrays,
    file order preserved -- i.e. movable-first, matching every source this
    module is defined over)."""

    __slots__ = ("n_nodes", "n_terminals", "n_terminal_ni",
                "node_size_x", "node_size_y", "node_orientation", "is_terminal", "is_terminal_ni", "name2local",
                "n_nets", "n_pins", "net_degrees",
                "pin2node", "pin2net", "pin_offset_x", "pin_offset_y", "pin_names")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _read_nodes(path):
    with open(path) as f:
        header = next(f)
        if not header.startswith("UCLA nodes"):
            raise ValueError(f"{path}: bad header {header!r}")
        n_nodes = n_terminals = None
        for line in f:
            s = line.strip()
            if s.startswith("NumNodes"):
                n_nodes = int(s.split(":")[1])
            elif s.startswith("NumTerminals"):
                n_terminals = int(s.split(":")[1])
                break
        if n_nodes is None or n_terminals is None:
            raise ValueError(f"{path}: missing NumNodes/NumTerminals header")

        names = [None] * n_nodes
        size_x = np.empty(n_nodes, dtype=np.float64)
        size_y = np.empty(n_nodes, dtype=np.float64)
        is_terminal = np.zeros(n_nodes, dtype=bool)
        is_terminal_ni = np.zeros(n_nodes, dtype=bool)
        n_ni = 0
        i = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            names[i] = parts[0]
            size_x[i] = float(parts[1])
            size_y[i] = float(parts[2])
            if len(parts) >= 4:
                is_terminal[i] = True
                if parts[3] == "terminal_NI":
                    n_ni += 1
                    is_terminal_ni[i] = True
            i += 1
    if i != n_nodes:
        raise ValueError(f"{path}: header says NumNodes={n_nodes} but {i} records were read")
    name2local = {name: idx for idx, name in enumerate(names)}
    if len(name2local) != n_nodes:
        raise ValueError(f"{path}: duplicate node names (expected {n_nodes} unique)")
    return names, name2local, size_x, size_y, is_terminal, is_terminal_ni, n_ni, n_nodes, n_terminals


def _read_pl(path, name2local, n_nodes):
    x = np.zeros(n_nodes, dtype=np.float64)
    y = np.zeros(n_nodes, dtype=np.float64)
    seen = np.zeros(n_nodes, dtype=bool)
    orientation = np.full(n_nodes, "N", dtype="U7")
    with open(path) as f:
        header = next(f)
        if not header.startswith("UCLA pl"):
            raise ValueError(f"{path}: bad header {header!r}")
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            idx = name2local[parts[0]]
            x[idx] = float(parts[1])
            y[idx] = float(parts[2])
            if len(parts) > 4:
                orientation[idx] = parts[4]
            seen[idx] = True
    n_missing = int((~seen).sum())
    if n_missing:
        raise ValueError(f"{path}: {n_missing} node(s) have no .pl record")
    return x, y, orientation


def _read_nets(path, name2local, *, include_pin_names=False):
    pin_names = {}
    with open(path) as f:
        header = next(f)
        if not header.startswith("UCLA nets"):
            raise ValueError(f"{path}: bad header {header!r}")
        n_nets = n_pins = None
        for line in f:
            s = line.strip()
            if s.startswith("NumNets"):
                n_nets = int(s.split(":")[1])
            elif s.startswith("NumPins"):
                n_pins = int(s.split(":")[1])
                break
        if n_nets is None or n_pins is None:
            raise ValueError(f"{path}: missing NumNets/NumPins header")

        net_degrees = np.empty(n_nets, dtype=np.int64)
        pin2node = np.empty(n_pins, dtype=np.int64)
        pin2net = np.empty(n_pins, dtype=np.int64)
        pin_offset_x = np.empty(n_pins, dtype=np.float64)
        pin_offset_y = np.empty(n_pins, dtype=np.float64)

        net_i = -1
        pin_i = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("NetDegree"):
                net_i += 1
                _, rest = s.split(":", 1)
                deg_str, _name = rest.split(None, 1)
                net_degrees[net_i] = int(deg_str)
                continue
            parts = s.split()
            # parts: name, direction(I/O), ':', dx, dy
            pin2node[pin_i] = name2local[parts[0]]
            pin2net[pin_i] = net_i
            pin_offset_x[pin_i] = float(parts[3])
            pin_offset_y[pin_i] = float(parts[4])
            if len(parts) >= 9:
                pin_names[pin_i] = parts[8]
            pin_i += 1
    if net_i + 1 != n_nets:
        raise ValueError(f"{path}: header NumNets={n_nets} but {net_i + 1} NetDegree records read")
    if pin_i != n_pins:
        raise ValueError(f"{path}: header NumPins={n_pins} but {pin_i} pin records read")
    result = net_degrees, pin2node, pin2net, pin_offset_x, pin_offset_y, n_nets, n_pins
    return (*result, pin_names) if include_pin_names else result


def parse_source_tile(prefix):
    """Parses `<prefix>.{nodes,pl,nets}` (via its `.aux`) into a
    `SourceTile`. This is the "parse once" step the module docstring's
    ~12s/~0.7GB budget refers to -- O(source tile size), never the
    replicated array's."""
    paths = tb.read_aux(prefix + ".aux")
    (names, name2local, size_x, size_y, is_terminal, is_terminal_ni, n_ni,
     n_nodes, n_terminals) = _read_nodes(paths["nodes"])
    pl_x, pl_y, orientation = _read_pl(paths["pl"], name2local, n_nodes)
    (net_degrees, pin2node, pin2net, pin_off_x, pin_off_y,
     n_nets, n_pins, pin_names) = _read_nets(paths["nets"], name2local, include_pin_names=True)

    return SourceTile(
        n_nodes=n_nodes, n_terminals=n_terminals, n_terminal_ni=n_ni,
        node_size_x=size_x, node_size_y=size_y, is_terminal=is_terminal,
        is_terminal_ni=is_terminal_ni,
        node_orientation=orientation, pin_names=pin_names,
        name2local=name2local,
        n_nets=n_nets, n_pins=n_pins, net_degrees=net_degrees,
        pin2node=pin2node, pin2net=pin2net,
        pin_offset_x=pin_off_x, pin_offset_y=pin_off_y,
    ), (pl_x, pl_y), paths


# ---------------------------------------------------------------------------
# glue tail (tail-seek, never a full .nets scan)
# ---------------------------------------------------------------------------

def _parse_glue_lines(lines, n_glue_nets, n_glue_pins):
    """Returns a list of {"name", "pins": [(i,j,node_name), ...]} or None
    if `lines` doesn't contain a complete, well-formed run of exactly
    `n_glue_nets` glue-net blocks (signals the caller to retry with a
    bigger tail buffer)."""
    nets = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if not s.startswith("NetDegree"):
            return None
        _, rest = s.split(":", 1)
        deg_str, name = rest.split()
        deg = int(deg_str)
        if deg != 2:
            raise ValueError(f"glue net {name!r} has degree {deg}, expected 2 "
                             "(only bipartite tile-pair glue nets are in scope, "
                             "matching glue_gen.sample_glue_nets)")
        pins = []
        for _ in range(deg):
            i += 1
            if i >= n:
                return None
            parts = lines[i].strip().split()
            if not parts:
                return None
            ti, tj, node_name = _split_tile_prefix(parts[0])
            pins.append((ti, tj, node_name, float(parts[3]), float(parts[4])))
        nets.append({"name": name, "pins": pins})
        i += 1
    if len(nets) != n_glue_nets:
        return None
    if sum(len(net["pins"]) for net in nets) != n_glue_pins:
        return None
    for k, net in enumerate(nets):
        if net["name"] != f"glue{k}":
            raise ValueError(f"glue net order mismatch at k={k}: got {net['name']!r} "
                             "(glue_gen.append_glue_nets writes glue0..glueN-1 strictly "
                             "in order)")
    return nets


def _read_glue_tail(nets_path, n_glue_nets, n_glue_pins,
                    initial_bytes=8_000_000, max_tries=8):
    """Reads glue nets `glue0..glue{n_glue_nets-1}` from the tail of
    `nets_path` via bounded `seek()`s -- NEVER a full-file scan (module
    docstring). Doubles the tail-buffer size and retries if the marker
    ("NetDegree : 2 glue0") isn't found or a block is truncated by the
    buffer boundary, up to `max_tries`, capped at the file's own size."""
    if n_glue_nets == 0:
        return []
    size = os.path.getsize(nets_path)
    buf_bytes = min(size, initial_bytes)
    last_err = None
    for _ in range(max_tries):
        start = max(0, size - buf_bytes)
        with open(nets_path, "rb") as f:
            f.seek(start)
            chunk = f.read()
        text = chunk.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        if start > 0 and lines:
            lines = lines[1:]           # drop a possibly-truncated first line
        start_idx = None
        for idx, line in enumerate(lines):
            s = line.strip()
            if s.startswith("NetDegree") and s.rsplit(None, 1)[-1] == "glue0":
                start_idx = idx
                break
        if start_idx is not None:
            try:
                nets = _parse_glue_lines(lines[start_idx:], n_glue_nets, n_glue_pins)
            except ValueError as e:
                last_err = e
                nets = None
            if nets is not None:
                return nets
        if buf_bytes >= size:
            break
        buf_bytes = min(size, buf_bytes * 4)
    if last_err is not None:
        raise last_err
    raise ValueError(f"{nets_path}: could not locate a complete glue0..glue{n_glue_nets - 1} "
                     f"tail within {buf_bytes} bytes of the file end")


# ---------------------------------------------------------------------------
# build / load / verify
# ---------------------------------------------------------------------------

def _source_die_bounds(paths):
    rows = tb.read_scl(paths["scl"])
    if not rows:
        raise ValueError(f"{paths['scl']}: no rows")
    xl = min(r.x0 for r in rows)
    xr = max(r.x0 + r.num_sites * r.sitewidth for r in rows)
    yl = min(r.y for r in rows)
    yr = max(r.y + r.height for r in rows)
    return float(xl), float(yl), float(xr), float(yr)


def build_tiled_netlist_cache(manifest_path, out_dir):
    """One-time construction: parses the manifest's `source_prefix` tile
    once (numpy arrays), replicates it R*C times via translate/tile
    (never touching the R*C array's own multi-GB `.nodes`/`.pl`/`.nets`
    files), reads the glue tail of `<dst_prefix>.nets` (bounded seek, sec
    above), applies the movable-first stable permutation, and writes the
    resulting `Netlist` arrays + `meta.json` to `out_dir` as individual
    `.npy` files (so `load_tiled_netlist(..., mmap=True)` can mmap each
    one independently). Returns the meta dict (also written to
    `<out_dir>/meta.json`).

    `manifest_path` is `tile_bookshelf.tile`'s own `<dst_prefix>.
    manifest.json` (after `glue_gen.append_glue_nets` has updated its
    "glue" section) -- `dst_prefix` itself is derived by stripping the
    ".manifest.json" suffix, matching that module's own naming
    convention.
    """
    if not manifest_path.endswith(".manifest.json"):
        raise ValueError(f"manifest_path={manifest_path!r} must end with '.manifest.json'")
    dst_prefix = manifest_path[: -len(".manifest.json")]

    with open(manifest_path) as f:
        manifest = json.load(f)
    R, C = int(manifest["R"]), int(manifest["C"])
    W, H = manifest["tile_width"], manifest["tile_height"]
    src_prefix = manifest["source_prefix"]

    # Verify the *source* files (cheap, O(source size)) match what the
    # manifest recorded -- catches a stale/mismatched source without ever
    # touching the R*C array's own multi-GB files.
    src_paths = tb.read_aux(src_prefix + ".aux")
    for suf, expected in manifest.get("source_sha256", {}).items():
        if suf not in src_paths:
            continue
        actual = tb.sha256_file(src_paths[suf])
        if actual != expected:
            raise ValueError(f"source .{suf} sha256 mismatch vs manifest: "
                             f"expected {expected}, got {actual}")

    t0 = time.time()
    src, (pl_x, pl_y), src_paths = parse_source_tile(src_prefix)
    parse_source_s = time.time() - t0

    xl, yl, xr, yr = _source_die_bounds(src_paths)
    if (xr - xl) != W or (yr - yl) != H:
        raise ValueError(f"source .scl row bbox ({xr - xl}x{yr - yl}) does not match "
                         f"manifest tile_width/tile_height ({W}x{H})")

    selected, canonical_degrees, sort_provenance = native_pin_selection(
        src.net_degrees, src.pin2node, src.name2local, src.pin_names)
    canonical_nodes = src.pin2node[selected]
    stored_w, stored_h, canonical_x, canonical_y = normalize_geometry(
        src.node_size_x, src.node_size_y, src.node_orientation, src.is_terminal,
        canonical_nodes, src.pin_offset_x[selected], src.pin_offset_y[selected])
    n_src_nodes, n_src_nets, n_src_pins = src.n_nodes, src.n_nets, len(selected)
    n_tiles = R * C

    # ---- replicate node arrays, tile-major (t_idx = i*C + j) ----
    node_x = np.empty(n_tiles * n_src_nodes, dtype=np.float64)
    node_y = np.empty(n_tiles * n_src_nodes, dtype=np.float64)
    for i in range(R):
        for j in range(C):
            t_idx = i * C + j
            s, e = t_idx * n_src_nodes, (t_idx + 1) * n_src_nodes
            node_x[s:e] = round_away(pl_x + i * W)
            node_y[s:e] = round_away(pl_y + j * H)
    node_size_x = np.tile(stored_w, n_tiles)
    node_size_y = np.tile(stored_h, n_tiles)
    is_terminal = np.tile(src.is_terminal, n_tiles)
    is_terminal_ni = np.tile(src.is_terminal_ni, n_tiles)

    # ---- replicate pin arrays, same tile order (net-major within tile) ----
    tile_node_offset = (np.arange(n_tiles, dtype=np.int64) * n_src_nodes)
    tile_net_offset = (np.arange(n_tiles, dtype=np.int64) * n_src_nets)
    pin2node_base = np.tile(canonical_nodes, n_tiles) + np.repeat(tile_node_offset, n_src_pins)
    pin2net_base = np.tile(src.pin2net[selected], n_tiles) + np.repeat(tile_net_offset, n_src_pins)
    pin_offset_x_base = np.tile(canonical_x, n_tiles)
    pin_offset_y_base = np.tile(canonical_y, n_tiles)
    net_degrees_base = np.tile(canonical_degrees, n_tiles)

    n_nodes_total = n_tiles * n_src_nodes
    n_nets_base = n_tiles * n_src_nets
    n_pins_base = n_tiles * n_src_pins

    base = manifest.get("base", {})
    if base:
        assert n_nodes_total == base["n_nodes"], (n_nodes_total, base["n_nodes"])
        assert n_nets_base == base["n_nets"], (n_nets_base, base["n_nets"])
        assert n_tiles * src.n_pins == base["n_pins"], (n_tiles * src.n_pins, base["n_pins"])
    # cheap invariant (module docstring): per-tile net count is constant.
    assert n_nets_base == n_tiles * n_src_nets

    # ---- glue tail ----
    glue_spec = manifest.get("glue") or {}
    n_glue_nets = int(glue_spec.get("n_nets") or 0)
    n_glue_pins = int(glue_spec.get("n_pins") or 0)
    if n_glue_nets:
        glue_nets = _read_glue_tail(dst_prefix + ".nets", n_glue_nets, n_glue_pins)
        glue_pin2node = np.empty(2 * n_glue_nets, dtype=np.int64)
        k2 = 0
        for net in glue_nets:
            (i0, j0, name0, ox0, oy0), (i1, j1, name1, ox1, oy1) = net["pins"]
            if (i0, j0) == (i1, j1):
                raise ValueError(f"glue net {net['name']!r} does not span two distinct "
                                 f"tiles: both endpoints are in tile ({i0},{j0})")
            glue_pin2node[k2] = (i0 * C + j0) * n_src_nodes + src.name2local[name0]
            glue_pin2node[k2 + 1] = (i1 * C + j1) * n_src_nodes + src.name2local[name1]
            k2 += 2
        assert k2 == 2 * n_glue_nets
        glue_pin2net = np.repeat(n_nets_base + np.arange(n_glue_nets, dtype=np.int64), 2)
        glue_pin_offset_x = np.empty(2 * n_glue_nets, dtype=np.float64)
        glue_pin_offset_y = np.empty(2 * n_glue_nets, dtype=np.float64)
        for k, net in enumerate(glue_nets):
            (i0, j0, n0, ox0, oy0), (i1, j1, n1, ox1, oy1) = net["pins"]
            for q, (ti, tj, name, ox, oy) in enumerate(((i0,j0,n0,ox0,oy0),(i1,j1,n1,ox1,oy1))):
                local = src.name2local[name]
                _, _, gx, gy = normalize_geometry(
                    src.node_size_x[local:local+1], src.node_size_y[local:local+1],
                    src.node_orientation[local:local+1], src.is_terminal[local:local+1],
                    np.array([0]), np.array([ox]), np.array([oy]))
                glue_pin_offset_x[2*k+q], glue_pin_offset_y[2*k+q] = gx[0], gy[0]
        glue_degrees = np.full(n_glue_nets, 2, dtype=np.int64)

        pin2node_pre = np.concatenate([pin2node_base, glue_pin2node])
        pin2net_pre = np.concatenate([pin2net_base, glue_pin2net])
        pin_offset_x = np.concatenate([pin_offset_x_base, glue_pin_offset_x])
        pin_offset_y = np.concatenate([pin_offset_y_base, glue_pin_offset_y])
        net_degrees = np.concatenate([net_degrees_base, glue_degrees])
    else:
        pin2node_pre, pin2net_pre = pin2node_base, pin2net_base
        pin_offset_x, pin_offset_y = pin_offset_x_base, pin_offset_y_base
        net_degrees = net_degrees_base

    n_nets_total = n_nets_base + n_glue_nets
    n_pins_total = n_pins_base + n_glue_pins
    num_terminal_ni_total = n_tiles * src.n_terminal_ni

    # ---- cheap invariants (module docstring), pre-permutation ----
    assert int(net_degrees.sum()) == n_pins_total, "sum(degrees) != NumPins"
    assert int(pin2node_pre.max()) < n_nodes_total, "pin2node out of range"
    assert int((~is_terminal).sum()) + n_tiles * (src.n_terminals - src.n_terminal_ni) + num_terminal_ni_total == n_nodes_total

    # ---- movable-first stable repermutation ----
    node_class = np.where(~is_terminal, 0, np.where(is_terminal_ni, 2, 1))
    perm = np.argsort(node_class, kind="stable")          # perm[new_idx] = old_idx
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.shape[0], dtype=perm.dtype)   # inv_perm[old_idx] = new_idx

    node_x = node_x[perm]
    node_y = node_y[perm]
    node_size_x = node_size_x[perm]
    node_size_y = node_size_y[perm]
    is_terminal_sorted = is_terminal[perm]

    num_movable = int((~is_terminal).sum())
    assert not is_terminal_sorted[:num_movable].any(), "movable-first permutation broken"
    assert is_terminal_sorted[num_movable:].all(), "movable-first permutation broken"

    pin2node = inv_perm[pin2node_pre]
    pin2net = pin2net_pre

    if int(pin2node.max()) >= 2**31 - 1 or n_nodes_total >= 2**31 - 1:
        raise ValueError("pin2node/node count exceeds int32 range; widen storage dtype")
    if n_nets_total >= 2**31 - 1:
        raise ValueError("n_nets_total exceeds int32 range; widen storage dtype")

    flat_net2pin_start = np.concatenate([[0], np.cumsum(net_degrees)]).astype(np.int32)
    flat_net2pin = np.arange(n_pins_total, dtype=np.int32)

    num_terminals_total = n_tiles * (src.n_terminals - src.n_terminal_ni)
    num_terminal_ni_total = n_tiles * src.n_terminal_ni

    arrays = {
        "node_x": node_x, "node_y": node_y,
        "node_size_x": node_size_x, "node_size_y": node_size_y,
        "pin_offset_x": pin_offset_x.astype(np.float64),
        "pin_offset_y": pin_offset_y.astype(np.float64),
        "pin2node": pin2node.astype(np.int32),
        "pin2net": pin2net.astype(np.int32),
        "flat_net2pin": flat_net2pin,
        "flat_net2pin_start": flat_net2pin_start,
        "inv_perm": inv_perm,
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, name + ".npy"), arr)

    meta = {
        "manifest_path": os.path.abspath(manifest_path),
        "dst_prefix": os.path.abspath(dst_prefix),
        "source_prefix": os.path.abspath(src_prefix),
        "R": R, "C": C, "tile_width": W, "tile_height": H,
        "n_src_nodes": n_src_nodes, "n_src_nets": n_src_nets, "n_src_pins": n_src_pins,
        "n_src_pins_raw": src.n_pins, "n_src_pins_canonical": n_src_pins,
        "n_pins_base_raw": n_tiles * src.n_pins,
        "n_pins_raw": n_tiles * src.n_pins + n_glue_pins,
        "n_duplicate_pins_removed": n_tiles * (src.n_pins - n_src_pins),
        "native_sort_implementation": sort_provenance,
        "orientation_policy": "native_fixed_only_N_normalization",
        "n_nodes": n_nodes_total,
        "num_movable": num_movable,
        "num_terminals": num_terminals_total,
        "num_terminal_NIs": num_terminal_ni_total,
        "n_nets": n_nets_total, "n_pins": n_pins_total,
        "n_nets_base": n_nets_base, "n_pins_base": n_pins_base,
        "n_glue_nets": n_glue_nets, "n_glue_pins": n_glue_pins,
        "xl": xl, "yl": yl, "xh": xl + R * W, "yh": yl + C * H,
        "schema_version": 3,
        "node_order": "movable_fixed_terminal_ni",
        "pin_offset_origin": "lower_left",
        "parse_source_s": parse_source_s,
        "manifest_source_sha256": manifest.get("source_sha256", {}),
        "manifest_output_sha256": manifest.get("output_sha256", {}),
    }
    np.save(os.path.join(out_dir, "source_selected_pin_indices.npy"), selected)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def load_tiled_netlist(cache_dir, mmap=True):
    """Loads the `Netlist` (+ meta dict) written by `build_tiled_netlist_
    cache`. `mmap=True` (default) opens every array with `np.load(...,
    mmap_mode="r")` -- essential at the 27.7M-node/108M-pin scale so
    merely constructing the `Netlist` doesn't require reading the whole
    cache into host RAM up front; `mmap=False` is for small toy-fixture
    tests where an ordinary in-memory array is simpler to assert on."""
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("schema_version") != 3 or meta.get("node_order") != "movable_fixed_terminal_ni" or meta.get("pin_offset_origin") != "lower_left":
        raise ValueError("unsupported or absent cache schema; rebuild cache with build_tiled_netlist_cache")
    mode = "r" if mmap else None

    def _load(name):
        return np.load(os.path.join(cache_dir, name + ".npy"), mmap_mode=mode)

    nl = Netlist(
        node_x=_load("node_x"), node_y=_load("node_y"),
        node_size_x=_load("node_size_x"), node_size_y=_load("node_size_y"),
        num_movable=meta["num_movable"], num_terminals=meta["num_terminals"],
        num_terminal_NIs=meta["num_terminal_NIs"],
        pin_offset_x=_load("pin_offset_x"), pin_offset_y=_load("pin_offset_y"),
        pin2node=_load("pin2node"), pin2net=_load("pin2net"),
        flat_net2pin=_load("flat_net2pin"), flat_net2pin_start=_load("flat_net2pin_start"),
        xl=meta["xl"], yl=meta["yl"], xh=meta["xh"], yh=meta["yh"],
    )
    return nl, meta


def verify_against_bookshelf(cache_dir, prefix, mode="full"):
    """Re-derives node positions/sizes and net/pin structure directly by
    streaming the on-disk Bookshelf array at `prefix` (the R*C array's own
    prefix, e.g. `results/m4/bench/arrays/3x3_n2/3x3_n2` -- NOT the
    source tile) and cross-checks every record against the cache built by
    `build_tiled_netlist_cache` at `cache_dir`, via `inv_perm.npy` (never
    trusts the replication arithmetic on its own -- an independent,
    from-the-actual-files re-derivation).

    `mode="window"`: only the first `_WINDOW_RECORDS` node/pl/net records
    are streamed (fast, toy-fixture/CI-friendly) -- PLUS the glue tail,
    always checked in full regardless of mode (cheap, and it's exactly the
    part of the array replication doesn't cover, so it's the highest-value
    check to never skip).
    `mode="full"`: every node/pl/net record is streamed (design draft's
    "~5 min CPU" one-time verification on the real 3x3 array -- bounded
    host memory throughout, one line at a time, same streaming style as
    `tile_bookshelf.py`/`glue_gen.py`).

    Returns a dict with `ok`, per-file checked/mismatched counts, and
    `elapsed_s`; raises only on structural parse errors (a genuine
    mismatch is reported in the returned dict, not an exception, so a
    caller can still inspect *how many* records disagreed).
    """
    if mode not in ("full", "window"):
        raise ValueError(f"mode must be 'full' or 'window', got {mode!r}")
    t0 = time.time()

    nl, meta = load_tiled_netlist(cache_dir, mmap=True)
    inv_perm = np.load(os.path.join(cache_dir, "inv_perm.npy"), mmap_mode="r")
    n_src_nodes = meta["n_src_nodes"]
    C = meta["C"]

    src, (pl_x, pl_y), src_paths = parse_source_tile(meta["source_prefix"])
    selected, canonical_degrees, _ = native_pin_selection(
        src.net_degrees, src.pin2node, src.name2local, src.pin_names)
    expected_w, expected_h, expected_x, expected_y = normalize_geometry(
        src.node_size_x, src.node_size_y, src.node_orientation, src.is_terminal,
        src.pin2node[selected], src.pin_offset_x[selected], src.pin_offset_y[selected])
    raw_to_canonical = np.full(src.n_pins, -1, dtype=np.int64)
    raw_to_canonical[selected] = np.arange(len(selected))

    window = None if mode == "full" else 200_000
    result = {
        "mode": mode, "ok": True, "errors": [],
        "n_checked_nodes": 0, "n_mismatched_nodes": 0,
        "n_checked_pl": 0, "n_mismatched_pl": 0,
        "n_checked_nets": 0, "n_mismatched_nets": 0,
        "n_checked_pins": 0, "n_mismatched_pins": 0,
        "n_checked_glue_nets": 0, "n_mismatched_glue_nets": 0,
        "n_checked_raw_pins": 0, "n_checked_canonical_pins": 0,
        "n_duplicate_pins_removed": meta["n_duplicate_pins_removed"],
    }

    def _fail(bucket, msg, cap=50):
        result["ok"] = False
        result[f"n_mismatched_{bucket}"] += 1
        if len(result["errors"]) < cap:
            result["errors"].append(msg)

    def _stored_idx(ti, tj, local_name):
        t_idx = ti * C + tj
        local_idx = src.name2local[local_name]
        return int(inv_perm[t_idx * n_src_nodes + local_idx])

    # ---- .nodes (size only -- position lives in .pl) ----
    with open(prefix + ".nodes") as f:
        next(f)
        for line in f:
            if line.strip().startswith("NumTerminals"):
                break
        n_checked = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            if window is not None and n_checked >= window:
                break
            parts = s.split()
            ti, tj, local_name = _split_tile_prefix(parts[0])
            idx = _stored_idx(ti, tj, local_name)
            result["n_checked_nodes"] += 1
            n_checked += 1
            local = src.name2local[local_name]
            if (float(parts[1]) != src.node_size_x[local] or float(parts[2]) != src.node_size_y[local]
                    or nl.node_size_x[idx] != expected_w[local]
                    or nl.node_size_y[idx] != expected_h[local]):
                _fail("nodes", f".nodes size mismatch for {parts[0]!r}")

    # ---- .pl (position) ----
    with open(prefix + ".pl") as f:
        next(f)
        n_checked = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            if window is not None and n_checked >= window:
                break
            parts = s.split()
            ti, tj, local_name = _split_tile_prefix(parts[0])
            idx = _stored_idx(ti, tj, local_name)
            result["n_checked_pl"] += 1
            n_checked += 1
            local = src.name2local[local_name]
            if (float(parts[1]) != pl_x[local] + ti*meta["tile_width"]
                    or float(parts[2]) != pl_y[local] + tj*meta["tile_height"]
                    or parts[4] != src.node_orientation[local]
                    or nl.node_x[idx] != round_away(float(parts[1]))
                    or nl.node_y[idx] != round_away(float(parts[2]))):
                _fail("pl", f".pl position mismatch for {parts[0]!r}")

    # ---- .nets (base + glue): pin order in the file == cache pin array
    # order (both concatenate tile-major then glue, identically) ----
    net_degrees = np.diff(nl.flat_net2pin_start)
    n_base_nets = meta["n_nets_base"]
    with open(prefix + ".nets") as f:
        header = next(f)
        for line in f:
            if line.strip().startswith("NumPins"):
                break
        net_i = pin_i = 0
        n_checked_nets_local = 0
        stop_base_stream = window is not None
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("NetDegree"):
                if stop_base_stream and net_i >= window and net_i < n_base_nets:
                    break  # window mode: stop the (huge) base-net stream early;
                           # the glue tail is checked separately below regardless.
                deg = int(s.split(":", 1)[1].split()[0])
                raw_degree = int(src.net_degrees[net_i % src.n_nets]) if net_i < n_base_nets else 2
                canonical_degree = int(canonical_degrees[net_i % src.n_nets]) if net_i < n_base_nets else 2
                if deg != raw_degree or net_i >= len(net_degrees) or net_degrees[net_i] != canonical_degree:
                    _fail("nets", f"net #{net_i} raw/canonical degree mismatch")
                result["n_checked_nets"] += 1
                net_i += 1
                continue
            parts = s.split()
            ti, tj, local_name = _split_tile_prefix(parts[0])
            expected_stored = _stored_idx(ti, tj, local_name)
            result["n_checked_pins"] += 1
            result["n_checked_raw_pins"] += 1
            raw_x, raw_y = float(parts[3]), float(parts[4])
            local = src.name2local[local_name]
            if pin_i < meta["n_pins_base_raw"]:
                tile, raw_i = divmod(pin_i, src.n_pins)
                source_node = int(src.pin2node[raw_i])
                if ((ti*C + tj) != tile or local != source_node or raw_x != src.pin_offset_x[raw_i]
                        or raw_y != src.pin_offset_y[raw_i]):
                    _fail("pins", f"raw pin #{pin_i} ({parts[0]!r}) replication mismatch")
                canonical_i = raw_to_canonical[raw_i]
                if canonical_i >= 0:
                    stored_pin = tile * len(selected) + canonical_i
                    result["n_checked_canonical_pins"] += 1
                    if (int(nl.pin2node[stored_pin]) != expected_stored
                            or nl.pin_offset_x[stored_pin] != expected_x[canonical_i]
                            or nl.pin_offset_y[stored_pin] != expected_y[canonical_i]):
                        _fail("pins", f"canonical pin #{stored_pin} ({parts[0]!r}) mismatch")
            pin_i += 1

    # ---- glue tail: always checked in full, regardless of mode ----
    n_glue_nets = meta["n_glue_nets"]
    n_glue_pins = meta["n_glue_pins"]
    if n_glue_nets:
        glue_nets = _read_glue_tail(prefix + ".nets", n_glue_nets, n_glue_pins)
        base_pin_count = meta["n_pins_base"]
        for k, net in enumerate(glue_nets):
            result["n_checked_glue_nets"] += 1
            (i0, j0, n0, ox0, oy0), (i1, j1, n1, ox1, oy1) = net["pins"]
            expected = [_stored_idx(i0, j0, n0), _stored_idx(i1, j1, n1)]
            got = [int(nl.pin2node[base_pin_count + 2 * k]),
                  int(nl.pin2node[base_pin_count + 2 * k + 1])]
            if got != expected:
                _fail("glue_nets", f"glue net #{k} pin2node mismatch: file-derived={expected} cache={got}")
            for q, (name, ox, oy) in enumerate(((n0, ox0, oy0), (n1, ox1, oy1))):
                local = src.name2local[name]
                _, _, ex, ey = normalize_geometry(
                    src.node_size_x[local:local+1], src.node_size_y[local:local+1],
                    src.node_orientation[local:local+1], src.is_terminal[local:local+1],
                    np.array([0]), np.array([ox]), np.array([oy]))
                result["n_checked_canonical_pins"] += 1
                if nl.pin_offset_x[base_pin_count + 2*k + q] != ex[0] or nl.pin_offset_y[base_pin_count + 2*k + q] != ey[0]:
                    _fail("glue_nets", f"glue net #{k} offset mismatch")

    # Raw fingerprints also cover discarded pin names/directions and records
    # outside the canonical representation. Window mode deliberately omits
    # this full-file work and is never reported as exhaustive verification.
    if mode == "full":
        result["raw_file_sha256"] = {}
        for kind, expected in meta["manifest_output_sha256"].items():
            actual = tb.sha256_file(prefix + "." + kind)
            result["raw_file_sha256"][kind] = actual
            if actual != expected:
                result["ok"] = False
                if len(result["errors"]) < 50:
                    result["errors"].append(f"raw .{kind} SHA256 differs from manifest")

    result["elapsed_s"] = time.time() - t0
    return result
