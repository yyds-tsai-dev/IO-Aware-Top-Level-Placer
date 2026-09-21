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


def segment_ids_on_row(table, row, lo, hi):
    """Vertical segments crossed by a horizontal leg in grid row `row` running
    between grid columns `lo` and `hi` (inclusive, either order). The leg
    crosses the column boundaries j with min(lo,hi) <= j < max(lo,hi), so this
    is one contiguous CSR slice and costs the number of crossings, not the
    length of the leg (sec 5)."""
    n_rows = table.row_ptr.shape[0] - 1
    if not (0 <= row < n_rows):
        raise IndexError(f"row {row} out of bounds for a table with {n_rows} rows")
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    s0, s1 = int(table.row_ptr[row]), int(table.row_ptr[row + 1])
    cols = table.row_col[s0:s1]
    i0 = int(np.searchsorted(cols, a, side="left"))
    i1 = int(np.searchsorted(cols, b, side="left"))
    return table.row_seg[s0 + i0:s0 + i1]


def segment_ids_on_col(table, col, lo, hi):
    """Horizontal segments crossed by a vertical leg in grid column `col`."""
    n_cols = table.col_ptr.shape[0] - 1
    if not (0 <= col < n_cols):
        raise IndexError(f"col {col} out of bounds for a table with {n_cols} cols")
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    s0, s1 = int(table.col_ptr[col]), int(table.col_ptr[col + 1])
    rows = table.col_row[s0:s1]
    i0 = int(np.searchsorted(rows, a, side="left"))
    i1 = int(np.searchsorted(rows, b, side="left"))
    return table.col_seg[s0 + i0:s0 + i1]


def edge_segment_ids(rg, table, x0, y0, x1, y1):
    """Segments crossed by the L-route (x0,y0) -> (x1,y0) -> (x1,y1), in that
    order -- the same geometry `evaluator_ref.edge_regions_and_crossings`
    walks, so `len(...)` equals that function's crossing count."""
    ax, ay = rg.to_idx(np.asarray([x0], dtype=np.float64),
                       np.asarray([y0], dtype=np.float64))
    bx, by = rg.to_idx(np.asarray([x1], dtype=np.float64),
                       np.asarray([y1], dtype=np.float64))
    horizontal = segment_ids_on_row(table, int(ay[0]), int(ax[0]), int(bx[0]))
    vertical = segment_ids_on_col(table, int(bx[0]), int(ay[0]), int(by[0]))
    return np.concatenate([horizontal, vertical])


def segment_utilisation(demand, capacity):
    """(util, scalars) from per-segment demand and capacity.

    Unit rule (round-feedback spec:169-196): a zero-capacity segment is
    blocked, not scaled -- its utilisation is reported as `inf` when it
    carries demand and 0.0 when it does not, and it is *excluded* from
    max_util/p99_util (which are the statistics of the routable segments) but
    *counted* in num_over_capacity and reported separately. No epsilon is
    substituted anywhere.

    `util` here is a diagnostic ratio only. Ruling D-3's `C_s = 0 -> d_s :=
    D_s` governs the differentiable penalty term `d_s` that a later task
    builds on top of segment_demand/segment_capacity -- it does not apply to
    this function's `util`, which stays `inf`/`0.0` at zero capacity as
    described above. Do not conflate the two: applying D-3's substitution to
    `util` instead of `d_s` would be a different (and wrong) quantity.

    Both evaluators call this one function on the same bit-exact
    `segment_demand`, which is what makes segment_util/max_util/p99_util
    bit-exact across CPU and GPU rather than merely close."""
    demand = np.asarray(demand, dtype=np.int64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if demand.shape != capacity.shape:
        raise ValueError("segment demand and capacity must have the same shape")
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError("segment demand and capacity must be non-negative")
    positive = capacity > 0.0
    util = np.zeros(demand.shape, dtype=np.float64)
    np.divide(demand, capacity, out=util, where=positive)
    util[(~positive) & (demand > 0)] = np.inf
    routable = util[positive]
    scalars = {
        "num_segments": int(demand.size),
        "segment_demand_total": int(demand.sum()),
        "num_over_capacity": int((demand > capacity).sum()),
        "num_zero_capacity_segments": int((~positive).sum()),
        "zero_capacity_demand": int(demand[~positive].sum()),
        "max_util": float(routable.max()) if routable.size else 0.0,
        "p99_util": float(np.percentile(routable, 99)) if routable.size else 0.0,
    }
    assert tuple(scalars) == CAPACITY_SCALARS, "CAPACITY_SCALARS drifted"
    return util, scalars


@dataclass
class Candidates:
    """The capped per-net candidate list the capacity term differentiates.

    One row per (net, demand pair (u,v), segment). `group` is the dense id of
    the alpha-softmax group (net, u, v, boundary pair of seg) -- see
    interpretation D-2 in the plan: alternatives on one boundary pair share a
    group and a single w*q_u*q_v; a feed-through's entry and exit segments sit
    in different groups and each take the full product."""

    net: np.ndarray       # (M,) int64
    u: np.ndarray         # (M,) int64, demand-pair low region
    v: np.ndarray         # (M,) int64, demand-pair high region
    seg: np.ndarray       # (M,) int64
    group: np.ndarray     # (M,) int64, dense and non-decreasing
    count: np.ndarray     # (M,) int64, observed crossings (diagnostics only)
    n_groups: int

    def is_empty(self):
        return int(self.net.shape[0]) == 0

    def remap_nets(self, mapping):
        """Translate global net ids through `mapping` (an (n_nets,) int64 array
        with -1 for nets the differentiable term does not carry -- the IO CSR
        drops nets above `ignore_net_degree` and nets collapsing to one node,
        `ops/io_term.build_net_node_csr`), dropping the rest and recompacting
        the group ids."""
        mapping = np.asarray(mapping, dtype=np.int64)
        new_net = mapping[self.net]
        keep = new_net >= 0
        group = self.group[keep]
        _uniq, dense = np.unique(group, return_inverse=True)
        return Candidates(net=new_net[keep], u=self.u[keep], v=self.v[keep],
                          seg=self.seg[keep], group=dense.astype(np.int64),
                          count=self.count[keep], n_groups=int(_uniq.size))


def _rank_within(owner, order):
    """Position of each element within its `owner` run, for an array already
    sorted so that equal owners are contiguous and the desired ranking order
    holds inside each run. `order` is that sort permutation."""
    owner_sorted = owner[order]
    n = owner_sorted.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    new_run[1:] = owner_sorted[1:] != owner_sorted[:-1]
    positions = np.arange(n, dtype=np.int64)
    run_start = np.maximum.accumulate(np.where(new_run, positions, 0))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = positions - run_start
    return rank


def select_candidates(net, u, v, seg, count, table, m_pairs=4, m_seg=2):
    """Cap a net's crossed (demand pair, segment) items at `m_pairs` groups and
    `m_seg` segments per group (spec sec 5's m_pairs=4 / m_seg=2 budget, read
    through interpretation D-2).

    Inputs are the evaluator's per-(net, u, v, seg) crossing counts with u < v
    and u != v; both evaluators produce them in the same ascending composite-key
    order, so this function's output is bit-identical between them. Every
    tie-break is total: groups by (-total count, group key), segments within a
    group by (-count, segment id)."""
    net = np.asarray(net, dtype=np.int64)
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    seg = np.asarray(seg, dtype=np.int64)
    count = np.asarray(count, dtype=np.int64)
    if net.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return Candidates(net=z, u=z, v=z, seg=z, group=z, count=z, n_groups=0)
    if (u >= v).any():
        raise ValueError("candidate demand pairs must satisfy u < v "
                         "(same-region legs are dropped by the caller)")
    k = np.int64(table.k)
    spair = table.pair_key()[seg]
    # (net, u, v, boundary pair) -> dense group id, ascending by that tuple
    gkey = ((net * k + u) * k + v) * (k * k) + spair
    uniq_g, gid = np.unique(gkey, return_inverse=True)
    gid = gid.astype(np.int64)
    # bincount only takes float64 weights; count.sum() per group is well
    # under 2**53 at any realistic net degree / crossing count, so the
    # float64 accumulation is exact and the cast back to int64 loses nothing.
    gtot = np.bincount(gid, weights=count.astype(np.float64),
                       minlength=uniq_g.size).astype(np.int64)
    gnet = np.zeros(uniq_g.size, dtype=np.int64)
    gnet[gid] = net

    # ---- cap the number of groups per net ----
    g_order = np.lexsort((uniq_g, -gtot, gnet))
    g_rank = _rank_within(gnet, g_order)
    group_keep = g_rank < m_pairs

    # ---- cap the number of segments per kept group ----
    keep = group_keep[gid]
    i_order = np.lexsort((seg[keep], -count[keep], gid[keep]))
    i_rank = _rank_within(gid[keep], i_order)
    item_keep = i_rank < m_seg

    idx = np.nonzero(keep)[0][item_keep]
    idx = idx[np.lexsort((seg[idx], gid[idx]))]          # (group, seg) ascending
    _uniq, dense = np.unique(gid[idx], return_inverse=True)
    return Candidates(net=net[idx], u=u[idx], v=v[idx], seg=seg[idx],
                      group=dense.astype(np.int64), count=count[idx],
                      n_groups=int(_uniq.size))
