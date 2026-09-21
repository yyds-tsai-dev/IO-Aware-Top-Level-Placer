"""Boundary segments of a RegionGrid (v2 design sec 5, subproject P-D).

A *segment* is one maximal run of lattice boundary unit edges that separates
the same unordered region pair along the same boundary line. Everything in
P-D -- the capacity extractor, the differentiable term, both evaluators --
keys off the integer segment ids this module hands out, and off nothing else
(the unit rule, round-feedback spec:169-196: the GCell grid is touched once,
offline, and its output is a scalar keyed by segment id).

Orientation convention, fixed here once:

  ORIENT_V  from grid[:, :-1] != grid[:, 1:].  Boundary line x = xl +
            (line+1)*cell_w, spanning rows [lo, hi).  Wires cross it
            *horizontally*.  One unit edge is cell_h long.
  ORIENT_H  from grid[:-1, :] != grid[1:, :].  Boundary line y = yl +
            (line+1)*cell_h, spanning columns [lo, hi).  Wires cross it
            *vertically*.  One unit edge is cell_w long.

Segment ids are assigned V first, ordered by (line, lo), then H, ordered by
(line, lo).  The order is a contract: evaluator_ref, evaluator_gpu and
capacity.npz all index the same table and a reordering would silently
mismatch them.  `segments_digest` fingerprints it.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

SEGMENT_SCHEMA_VERSION = 1

ORIENT_V = 0
ORIENT_H = 1

# The scalar capacity fields every driver's result.json carries (the
# _pack_<x>_metrics convention P-F introduced for the straddle block).
CAPACITY_SCALARS = (
    "num_segments", "segment_demand_total", "num_over_capacity",
    "num_zero_capacity_segments", "zero_capacity_demand",
    "max_util", "p99_util",
)


@dataclass(frozen=True)
class SegmentTable:
    """Frozen per-segment geometry plus the two lookup structures sec 5 asks
    for: a unit-edge -> segment-id raster (2 x L x (L-1) int32, ~2 MB at
    L=512) and per-row/column CSR lists of boundary positions."""

    orient: np.ndarray        # (S,) int8, ORIENT_V / ORIENT_H
    line: np.ndarray          # (S,) int32, boundary line index
    lo: np.ndarray            # (S,) int32, inclusive start along the run axis
    hi: np.ndarray            # (S,) int32, exclusive end
    pair_a: np.ndarray        # (S,) int16, min region id
    pair_b: np.ndarray        # (S,) int16, max region id
    length_units: np.ndarray  # (S,) int32 == hi - lo
    length: np.ndarray        # (S,) float64, physical length in die units
    box: np.ndarray           # (S,4) float64 [x0,y0,x1,y1]; degenerate in one axis
    edge_seg_v: np.ndarray    # (ny, nx-1) int32, -1 where there is no boundary
    edge_seg_h: np.ndarray    # (ny-1, nx) int32
    row_ptr: np.ndarray       # (ny+1,) int64
    row_col: np.ndarray       # (Ev,) int32, ascending within a row
    row_seg: np.ndarray       # (Ev,) int32
    col_ptr: np.ndarray       # (nx+1,) int64
    col_row: np.ndarray       # (Eh,) int32, ascending within a column
    col_seg: np.ndarray       # (Eh,) int32
    k: int
    lattice: int
    die: tuple

    @property
    def num_segments(self):
        return int(self.orient.shape[0])

    def grid_shape(self):
        return (self.edge_seg_h.shape[0] + 1, self.edge_seg_v.shape[1] + 1)

    def pair_key(self):
        """(S,) int64 = pair_a*k + pair_b -- the boundary-pair identity used
        to group alternative segments (interpretation D-2)."""
        return self.pair_a.astype(np.int64) * self.k + self.pair_b.astype(np.int64)


def _runs(key, run_len):
    """Run-length-encode a 2-D `key` array (0 == no boundary) along its run
    axis. `key` is already laid out so that a C-order ravel walks the run
    axis fastest. Returns (starts, ends, flat_ids) where flat_ids is -1 off
    the boundary and a dense 0-based run index on it."""
    flat = key.ravel()
    prev = np.zeros_like(flat)
    prev[1:] = flat[:-1]
    first_in_line = (np.arange(flat.size) % run_len) == 0
    start = (flat != 0) & (first_in_line | (prev != flat))
    nxt = np.zeros_like(flat)
    nxt[:-1] = flat[1:]
    last_in_line = (np.arange(flat.size) % run_len) == (run_len - 1)
    end = (flat != 0) & (last_in_line | (nxt != flat))
    starts = np.nonzero(start)[0]
    ends = np.nonzero(end)[0]
    assert starts.shape == ends.shape, "run-length encoding lost a run"
    ids = np.cumsum(start) - 1
    ids = np.where(flat != 0, ids, -1).astype(np.int64)
    return starts, ends, ids


def _boundary_key(a, b, k):
    """0 where a == b, else min*k + max + 1 (so 0 is never a valid pair)."""
    diff = a != b
    return np.where(diff, np.minimum(a, b) * k + np.maximum(a, b) + 1, 0)


def enumerate_segments(rg):
    """Enumerate the boundary segments of a RegionGrid. Pure numpy, O(L^2)."""
    k = int(rg.k)
    grid = rg.grid.astype(np.int64)
    ny, nx = grid.shape
    xl, yl, _xh, _yh = rg.die
    cw, ch = float(rg.cell_w), float(rg.cell_h)

    # ---- vertical segments: runs down the rows of each column boundary ----
    key_v = _boundary_key(grid[:, :-1], grid[:, 1:], k)          # (ny, nx-1)
    # transpose so a C-order ravel walks rows fastest within one column
    starts_v, ends_v, ids_v = _runs(key_v.T.copy(), ny)
    line_v = (starts_v // ny).astype(np.int32)
    lo_v = (starts_v % ny).astype(np.int32)
    hi_v = (ends_v % ny + 1).astype(np.int32)
    pk_v = key_v.T.ravel()[starts_v] - 1
    n_v = int(starts_v.size)
    edge_seg_v = np.where(ids_v >= 0, ids_v, -1).reshape(nx - 1, ny).T.astype(np.int32)

    # ---- horizontal segments: runs along the columns of each row boundary ----
    key_h = _boundary_key(grid[:-1, :], grid[1:, :], k)          # (ny-1, nx)
    starts_h, ends_h, ids_h = _runs(key_h, nx)
    line_h = (starts_h // nx).astype(np.int32)
    lo_h = (starts_h % nx).astype(np.int32)
    hi_h = (ends_h % nx + 1).astype(np.int32)
    pk_h = key_h.ravel()[starts_h] - 1
    edge_seg_h = np.where(ids_h >= 0, ids_h + n_v, -1).reshape(ny - 1, nx).astype(np.int32)

    orient = np.concatenate([np.full(n_v, ORIENT_V, dtype=np.int8),
                             np.full(starts_h.size, ORIENT_H, dtype=np.int8)])
    line = np.concatenate([line_v, line_h]).astype(np.int32)
    lo = np.concatenate([lo_v, lo_h]).astype(np.int32)
    hi = np.concatenate([hi_v, hi_h]).astype(np.int32)
    pk = np.concatenate([pk_v, pk_h]).astype(np.int64)
    pair_a = (pk // k).astype(np.int16)
    pair_b = (pk % k).astype(np.int16)
    length_units = (hi - lo).astype(np.int32)

    unit = np.where(orient == ORIENT_V, ch, cw)
    length = length_units.astype(np.float64) * unit

    box = np.empty((orient.size, 4), dtype=np.float64)
    is_v = orient == ORIENT_V
    bx = xl + (line.astype(np.float64) + 1.0) * cw
    by = yl + (line.astype(np.float64) + 1.0) * ch
    box[:, 0] = np.where(is_v, bx, xl + lo.astype(np.float64) * cw)
    box[:, 2] = np.where(is_v, bx, xl + hi.astype(np.float64) * cw)
    box[:, 1] = np.where(is_v, yl + lo.astype(np.float64) * ch, by)
    box[:, 3] = np.where(is_v, yl + hi.astype(np.float64) * ch, by)

    # ---- CSR indexes ----
    ii, jj = np.nonzero(edge_seg_v >= 0)                # row-major: (i, then j)
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(ii, minlength=ny))]).astype(np.int64)
    row_col = jj.astype(np.int32)
    row_seg = edge_seg_v[ii, jj].astype(np.int32)

    hi_i, hi_j = np.nonzero(edge_seg_h >= 0)
    order = np.lexsort((hi_i, hi_j))                     # (j, then i)
    hi_i, hi_j = hi_i[order], hi_j[order]
    col_ptr = np.concatenate([[0], np.cumsum(np.bincount(hi_j, minlength=nx))]).astype(np.int64)
    col_row = hi_i.astype(np.int32)
    col_seg = edge_seg_h[hi_i, hi_j].astype(np.int32)

    return SegmentTable(orient=orient, line=line, lo=lo, hi=hi,
                        pair_a=pair_a, pair_b=pair_b,
                        length_units=length_units, length=length, box=box,
                        edge_seg_v=edge_seg_v, edge_seg_h=edge_seg_h,
                        row_ptr=row_ptr, row_col=row_col, row_seg=row_seg,
                        col_ptr=col_ptr, col_row=col_row, col_seg=col_seg,
                        k=k, lattice=int(rg.nx), die=tuple(float(v) for v in rg.die))


def segments_digest(table):
    """sha256 over the defining arrays plus the lattice identity. Written into
    capacity.npz and evaluation.npz so a capacity file can never be paired
    with a different geometry."""
    digest = hashlib.sha256()
    digest.update(str(SEGMENT_SCHEMA_VERSION).encode())
    digest.update(str((table.k, table.lattice, table.die)).encode())
    for arr in (table.orient, table.line, table.lo, table.hi,
                table.pair_a, table.pair_b):
        contiguous = np.ascontiguousarray(arr)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
