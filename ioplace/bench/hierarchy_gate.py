"""M4 T3a hierarchy gate (design draft `docs/superpowers/specs/2026-08-13-
m4-scale-up-design-draft.md` sec 3.1a): streams `mempool_cluster.def`'s
`COMPONENTS`/`NETS` sections (9.7 GB text) -- and, for comparison,
`mempool_group.def`'s own `COMPONENTS`/`NETS` sections (2.3 GB) -- to test
whether "cluster = 4 groups" actually holds, and if so what geometric
arrangement the 4 groups are in (2x2 / 1x4 / 4x1). This is a *blocking*
gate: sec 3.1's `11,310,807/4 = 2,827,702` (8.1% smaller than standalone
`mempool_group`) is cell-count-ratio evidence only -- it doesn't prove the
four quarters are each a same-shaped copy of `mempool_group`, doesn't rule
out synthesis pruning/shared top-level logic/different parameterization,
and (per Codex finding #7) the 8.1% gap actively argues *against* a naive
"4 identical copies" reading. G-A..G-E below turn that into five falsifiable
checks. `cluster_stats.py`/T4's glue-net generation and T6's tiler
calibration must not run until this gate passes -- this module only
produces the verdict, nothing downstream.

Never loads either DEF whole: both are read as an in-order stream of lines,
one record (a handful of lines) held in memory at a time. Per-prefix
accumulators (`_Bucket`) are bounded by the number of distinct top-level
instance-name prefixes (single digits) times the number of distinct LEF
cell names (~150) -- negligible next to the file sizes. G-D's classifier
(see below) additionally retains one `(bucket_index, x, y)` triple per
*placed* cell scanned during the same pass (`collect_geometry=True`) --
bounded to ~20 bytes/cell, ~226 MB for `mempool_cluster.def`'s 11.31M
cells, not "whole file in memory".

**2026-08-14 T6 adjudication (`docs/results/2026-08-14-m4-t6-adjudication.
md` sec 4 前置 1-3) retracted commit 309e5d7's G-D/`cluster_stats` numbers**
over two bugs fixed here:

  - **Bug B** (`NETS` `+`-clause truncation): the pin-tuple scanner did not
    stop at a net record's first `+` token, so `+ ROUTED ... ( x y )`
    routing coordinates and `( * pinname )` wildcard tuples were counted as
    real instance pins -- inflating cross-group net/pin counts ~7.2x.
    Fixed by tracking `after_plus` per net record; `n_wildcard_tuples` and
    `n_post_plus_tuples_skipped` in `scan_def`'s meta dict quantify the
    fix's effect.
  - **Bug A** (bbox-derived centroid): a *placed* chip's per-group bounding
    box is an extreme-value statistic set by a handful of outlier cells,
    not a robust group extent -- its midpoint is not a meaningful centroid.
    `_Bucket` now also accumulates `sum_x`/`sum_y`/`n_placed` for a mass
    centroid; the old bbox is kept (renamed `bbox_outlier_extent`, with a
    docstring warning) as a diagnostic, not as G-D's geometry signal.

G-D's classifier was also rewritten from a bbox-centroid distance-level
classifier (retracted -- it gave the *opposite* answer to the mass-weighted
signal on the real chip) to a mass-weighted hypothesis-assignment score:
see `classify_arrangement` below.

DEF format notes this parser relies on (from the file's own header:
`DIVIDERCHAR "/"`, `BUSBITCHARS "[]"`):
  - Instance/net/pin names are `\\`-escaped hierarchical paths (`\\[`, `\\]`
    escape literal bus-bit-like brackets, e.g.
    `gen_groups\\[2\\].i_group/gen_tiles\\[11\\].i_tile/.../g17170`). The
    *raw* (still-escaped) path segment before the first literal `/` is the
    "prefix" this gate groups by -- verified against a sample scan of
    `mempool_cluster.def` to have no escaped `/` (`\\/`) anywhere in
    `COMPONENTS`, so splitting on a bare `/` is safe.
  - Some instances have no `/` at all (e.g. `wake_up_q_reg\\[212\\]`) --
    flat top-level cells with no group prefix; bucketed under whatever
    `group_key` returns for them (`top_level_prefix` returns None, folded
    into the residual bucket by the caller).
  - `COMPONENTS` records are `- name cell [+ SOURCE TIMING] + <STATUS> ( x
    y ) ORIENT` then a lone ` ;` line (`STATUS in {FIXED,COVER,PLACED,
    UNPLACED}`), or 3 lines for `FIXED` macros with a `+ HALO` continuation.
  - `NETS` records list pins as `( instname pinname )` tuples (possibly
    wrapped across many lines for high-fanout nets -- the pre-CTS `clk_i`
    net in `mempool_cluster.def` alone spans >1M lines), then usually a `+`
    clause (`+ ROUTED ...`, `+ USE ...`, ...) that is *not* pin tuples --
    any `( ... )` tuple after the record's first `+` (routing coordinates,
    property values) must not be scanned as a pin. A `( PIN name )` tuple
    refers to a top-level design port, not an instance, and a
    `( * pinname )` tuple is a DEF wildcard/MUSTJOIN pattern, not a
    concrete instance pin -- both are excluded from per-prefix pin counts.
    Net records also terminate with a lone ` ;` line.

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.hierarchy_gate \\
        --cluster-def <mempool_cluster.def> --group-def <mempool_group.def> \\
        --out results/m4/corpus/hierarchy_gate.json
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from array import array
from collections import Counter, defaultdict

import numpy as np

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"

STATUS_WORDS = frozenset({"FIXED", "COVER", "PLACED", "UNPLACED"})

# sec 3.1a thresholds.
G_A_COVERAGE_MIN = 0.95
G_B_SIZE_TOL = 0.15
G_C_COSINE_MIN = 0.95

# G-D (2026-08-14 T6 adjudication doc sec 4 前置 3): mass-weighted
# hypothesis-assignment classifier. `score_2x2` searches axis-aligned split
# lines (cx, cy) over this grid (both endpoints inclusive); `score_1x4`/
# `score_4x1` use fixed equal-width quartiles along a single axis (no
# search -- there's no "which line" question for 4 abutted strips of equal
# width, only "which axis"). A candidate arrangement passes iff its score
# clears both a floor and a lead over the next-best of the three
# hypotheses -- see `classify_arrangement`.
G_D_CUT_LO, G_D_CUT_HI, G_D_CUT_STEP = 0.20, 0.80, 0.02
G_D_BIN_DOMINANCE_GRID = 64
G_D_PASS_SCORE_MIN = 0.50
G_D_PASS_MARGIN_MIN = 0.10

TOP_LEVEL_RESIDUAL_KEY = "__TOP__"


def _unescape(name):
    """Display-only: `gen_groups\\[2\\].i_group` -> `gen_groups[2].i_group`."""
    return name.replace("\\[", "[").replace("\\]", "]")


def top_level_prefix(name):
    """Raw (still `\\`-escaped) path segment before the first literal `/`,
    or `TOP_LEVEL_RESIDUAL_KEY` if `name` has no `/` (a flat top-level
    instance, e.g. `wake_up_q_reg\\[212\\]`)."""
    i = name.find("/")
    return name[:i] if i != -1 else TOP_LEVEL_RESIDUAL_KEY


def _all_one_bucket(_name):
    """`group_key` for a non-hierarchical reference design (standalone
    `mempool_group.def`): everything collapses into a single bucket."""
    return "_ALL_"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


class _Bucket:
    __slots__ = ("cells", "fixed_cells", "celltype", "pins", "nets_internal",
                 "nets_crossing_or_top", "xmin", "xmax", "ymin", "ymax",
                 "sum_x", "sum_y", "n_placed")

    def __init__(self):
        self.cells = 0
        self.fixed_cells = 0
        self.celltype = Counter()
        self.pins = 0
        self.nets_internal = 0
        self.nets_crossing_or_top = 0
        self.xmin = self.ymin = math.inf
        self.xmax = self.ymax = -math.inf
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.n_placed = 0

    def add_coord(self, x, y):
        if x < self.xmin: self.xmin = x
        if x > self.xmax: self.xmax = x
        if y < self.ymin: self.ymin = y
        if y > self.ymax: self.ymax = y
        self.sum_x += x
        self.sum_y += y
        self.n_placed += 1

    def merge(self, other):
        self.cells += other.cells
        self.fixed_cells += other.fixed_cells
        self.celltype += other.celltype
        self.pins += other.pins
        self.nets_internal += other.nets_internal
        self.nets_crossing_or_top += other.nets_crossing_or_top
        self.xmin = min(self.xmin, other.xmin); self.xmax = max(self.xmax, other.xmax)
        self.ymin = min(self.ymin, other.ymin); self.ymax = max(self.ymax, other.ymax)
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.n_placed += other.n_placed

    def to_dict(self):
        has_bbox = self.xmin != math.inf
        # `bbox_outlier_extent`: the min/max coordinate envelope of this
        # bucket's placed cells. On a *placed* chip (as opposed to a
        # synthetic tiler's abutted output) this is an extreme-value
        # statistic set by whichever handful of cells sit furthest from the
        # bulk of the group -- it is NOT a robust measure of where the
        # group actually sits (2026-08-14 T6 adjudication doc "Bug A": the
        # real cluster's bbox midpoints collapsed to ~the die center and
        # gave the *opposite* arrangement answer to the mass-weighted
        # signal). `centroid` below is the mass centroid
        # (`sum_x/n_placed`, `sum_y/n_placed`), used by G-D instead.
        bbox_outlier_extent = [self.xmin, self.ymin, self.xmax, self.ymax] if has_bbox else None
        centroid = ([self.sum_x / self.n_placed, self.sum_y / self.n_placed]
                    if self.n_placed else None)
        return dict(
            cells=self.cells, fixed_cells=self.fixed_cells,
            movable_cells=self.cells - self.fixed_cells,
            pins=self.pins, nets_internal=self.nets_internal,
            nets_touching=self.nets_internal + self.nets_crossing_or_top,
            celltype_hist={k: v for k, v in self.celltype.most_common()},
            bbox_outlier_extent=bbox_outlier_extent, centroid=centroid,
            n_placed=self.n_placed,
        )


def scan_def(def_path, group_key, collect_geometry=False):
    """Single streaming pass over `def_path`'s `COMPONENTS` and `NETS`
    sections (everything before `COMPONENTS` and after `END NETS` is
    skipped without being tokenized). `group_key(raw_instance_name) ->
    str` assigns each instance to a bucket key.

    If `collect_geometry` is set, every *placed* `COMPONENTS` cell's
    `(bucket_key, x, y)` is also appended to per-key arrays (kept for ALL
    buckets seen, not just an eventual top-4 -- the caller doesn't know
    which prefixes are dominant until the whole pass finishes) for G-D's
    `classify_arrangement`; this is the only opt-in departure from "one
    record at a time in memory" (see module docstring for the bound).

    Returns `(buckets: {key: _Bucket}, meta: dict, geometry: dict | None)`;
    `geometry` (when `collect_geometry`) is `{"keys": [key, ...],
    "gid": np.ndarray[int16] (index into "keys"), "x": np.ndarray[float64],
    "y": np.ndarray[float64]}`, one entry per placed component scanned."""
    buckets = defaultdict(_Bucket)
    declared_components = declared_nets = die_bbox = None
    scanned_components = scanned_nets = scanned_pin_tuples = skipped_top_port_tuples = 0
    n_wildcard_tuples = n_post_plus_tuples_skipped = 0
    section = "pre"  # pre -> components -> between -> nets -> done
    cur_net_prefixes = None  # set[str] | None, only while inside a NETS record
    after_plus = False  # True once a NETS record's first `+` token has been seen

    geom_keys = {}  # key -> index into geom_klist, only populated if collect_geometry
    geom_klist = []
    geom_gid = array("h") if collect_geometry else None
    geom_x = array("d") if collect_geometry else None
    geom_y = array("d") if collect_geometry else None

    def flush_net():
        nonlocal cur_net_prefixes
        if cur_net_prefixes is None:
            return
        if len(cur_net_prefixes) == 1:
            (key,) = cur_net_prefixes
            buckets[key].nets_internal += 1
        else:
            for key in cur_net_prefixes:
                buckets[key].nets_crossing_or_top += 1
        cur_net_prefixes = None

    t0 = time.time()
    with open(def_path, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("DIEAREA "):
                    # `DIEAREA ( xl yl ) ( xh yh ) ;`
                    toks = line.split()
                    die_bbox = [float(toks[2]), float(toks[3]), float(toks[6]), float(toks[7])]
                elif line.startswith("COMPONENTS "):
                    declared_components = int(line.split()[1])
                    section = "components"
                continue

            if section == "components":
                if line.startswith("END COMPONENTS"):
                    section = "between"
                    continue
                if not line.startswith("- "):
                    continue  # continuation line (` ;`, `+ HALO ...`)
                toks = line.split()
                name, cell = toks[1], toks[2]
                key = group_key(name)
                b = buckets[key]
                b.cells += 1
                b.celltype[cell] += 1
                status_idx = None
                for i in range(3, len(toks)):
                    if toks[i] in STATUS_WORDS:
                        status_idx = i
                        if toks[i] == "FIXED":
                            b.fixed_cells += 1
                        break
                if (status_idx is not None and status_idx + 3 < len(toks)
                        and toks[status_idx + 1] == "("):
                    x = float(toks[status_idx + 2])
                    y = float(toks[status_idx + 3])
                    b.add_coord(x, y)
                    if collect_geometry:
                        ki = geom_keys.get(key)
                        if ki is None:
                            ki = geom_keys[key] = len(geom_klist)
                            geom_klist.append(key)
                        geom_gid.append(ki)
                        geom_x.append(x)
                        geom_y.append(y)
                scanned_components += 1
                continue

            if section == "between":
                if line.startswith("NETS "):
                    declared_nets = int(line.split()[1])
                    section = "nets"
                continue

            if section == "nets":
                if line.startswith("END NETS"):
                    flush_net()
                    section = "done"
                    break
                if line.startswith("- "):
                    flush_net()
                    scanned_nets += 1
                    cur_net_prefixes = set()
                    after_plus = False
                if "(" not in line and "+" not in line:
                    continue
                toks = line.split()
                i, n = 0, len(toks)
                while i < n:
                    t = toks[i]
                    if not after_plus and t == "+":
                        # Everything from here to this net record's
                        # terminating ` ;` is a `+ ROUTED`/`+ USE`/... clause
                        # (routing coordinates, property values) -- not pin
                        # tuples (Bug B).
                        after_plus = True
                        i += 1
                        continue
                    # A pin tuple is exactly 4 tokens: `( instname pinname
                    # )`, including the `( PIN <portname> )` case for
                    # top-level design ports and the `( * pinname )`
                    # DEF wildcard/MUSTJOIN case.
                    if t == "(" and i + 3 < n and toks[i + 3] == ")":
                        if after_plus:
                            n_post_plus_tuples_skipped += 1
                            i += 4
                            continue
                        inst = toks[i + 1]
                        if inst == "PIN":
                            skipped_top_port_tuples += 1
                        elif inst == "*":
                            n_wildcard_tuples += 1
                        else:
                            key = group_key(inst)
                            buckets[key].pins += 1
                            scanned_pin_tuples += 1
                            if cur_net_prefixes is not None:
                                cur_net_prefixes.add(key)
                        i += 4
                    else:
                        i += 1
                continue
    elapsed = time.time() - t0

    meta = dict(
        declared_components=declared_components, scanned_components=scanned_components,
        declared_nets=declared_nets, scanned_nets=scanned_nets,
        scanned_pin_tuples=scanned_pin_tuples, skipped_top_port_tuples=skipped_top_port_tuples,
        n_wildcard_tuples=n_wildcard_tuples, n_post_plus_tuples_skipped=n_post_plus_tuples_skipped,
        scan_s=elapsed, die_bbox=die_bbox,
    )
    geometry = None
    if collect_geometry:
        geometry = dict(
            keys=geom_klist,
            gid=np.frombuffer(geom_gid, dtype=np.int16),
            x=np.frombuffer(geom_x, dtype=np.float64),
            y=np.frombuffer(geom_y, dtype=np.float64),
        )
    return dict(buckets), meta, geometry


def _cosine(hist_a, hist_b):
    keys = set(hist_a) | set(hist_b)
    if not keys:
        return None
    dot = sum(hist_a.get(k, 0) * hist_b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in hist_a.values()))
    nb = math.sqrt(sum(v * v for v in hist_b.values()))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def _cut_grid():
    n = round((G_D_CUT_HI - G_D_CUT_LO) / G_D_CUT_STEP)
    return [round(G_D_CUT_LO + i * G_D_CUT_STEP, 10) for i in range(n + 1)]


def _assignment_score(u_by_group, v_by_group, labels_fn, n_regions=4):
    """`u_by_group`/`v_by_group`: length-4 lists of each group's own cells'
    `u`/`v` arrays. `labels_fn(u, v) -> int array in [0, n_regions)` maps
    cell coordinates to a region id. Returns `(best_score, best_perm, M)`:
    `M[r, c]` = fraction of group r's own cell mass landing in region c;
    `best_perm` is the permutation of `range(n_regions)` (a 1-1 group ->
    region assignment) maximizing `mean_r(M[r, perm[r]])` ("assignment
    score" -- ties broken by permutation order)."""
    n_groups = len(u_by_group)
    M = np.zeros((n_groups, n_regions))
    for r in range(n_groups):
        n_r = len(u_by_group[r])
        if n_r == 0:
            continue
        labels = labels_fn(u_by_group[r], v_by_group[r])
        M[r] = np.bincount(labels, minlength=n_regions) / n_r
    best_score, best_perm = -1.0, None
    for perm in itertools.permutations(range(n_regions), n_groups):
        s = sum(M[r, perm[r]] for r in range(n_groups)) / n_groups
        if s > best_score:
            best_score, best_perm = s, perm
    return float(best_score), list(best_perm), M


def _score_2x2(u_by_group, v_by_group):
    """Grid search over split lines `(cx, cy)` -- quadrant label = `2 *
    [u>=cx] + [v>=cy]` (0=LL, 1=LU, 2=RL, 3=RU relative to the split)."""
    best = None
    for cx in _cut_grid():
        for cy in _cut_grid():
            def labels_fn(u, v, cx=cx, cy=cy):
                return (u >= cx).astype(np.int64) * 2 + (v >= cy).astype(np.int64)
            score, perm, M = _assignment_score(u_by_group, v_by_group, labels_fn)
            if best is None or score > best["score"]:
                best = dict(score=score, cx=cx, cy=cy, perm=perm, M=M)
    return best


def _score_strips(coord_by_group, n_regions=4):
    """Fixed equal-width quartiles along a single axis (`coord_by_group` is
    `u_by_group` for the 1x4/vertical-strip hypothesis, `v_by_group` for
    4x1/horizontal-strip) -- no line search: 4 abutted strips of a single
    axis have only one free parameter (which axis), not a cut position."""
    def labels_fn(coord, _other):
        return np.clip((coord * n_regions).astype(np.int64), 0, n_regions - 1)
    zeros = [np.zeros(0) for _ in coord_by_group]
    score, perm, M = _assignment_score(coord_by_group, zeros, labels_fn, n_regions)
    return dict(score=score, perm=perm, M=M)


def _bin_dominance(u, v, gid, n_groups=4, grid=G_D_BIN_DOMINANCE_GRID):
    """64x64 (default) mass-weighted bin occupancy: for each spatial bin,
    which group holds the most mass in it, and how concentrated that
    dominance is overall (`mean_max_share`, mass-weighted; uniform 4-way
    mixing would give `1/n_groups`)."""
    bx = np.clip((u * grid).astype(np.int64), 0, grid - 1)
    by = np.clip((v * grid).astype(np.int64), 0, grid - 1)
    bin_id = bx * grid + by
    dens = np.zeros((n_groups, grid * grid))
    for r in range(n_groups):
        m = gid == r
        dens[r] = np.bincount(bin_id[m], minlength=grid * grid)
    tot = dens.sum(axis=0)
    frac = np.divide(dens, np.maximum(tot, 1))
    dominant = frac.argmax(axis=0)
    maxf = frac.max(axis=0)
    tot_sum = tot.sum()
    w = tot / tot_sum if tot_sum else np.zeros_like(tot)
    mean_max_share = float((maxf * w).sum())
    thresholds = (0.4, 0.5, 0.7, 0.9)
    mass_above = {f"{thr:.1f}": float(w[maxf > thr].sum()) for thr in thresholds}
    return dict(
        grid=grid, uniform_mix_share=1.0 / n_groups,
        mean_max_share=mean_max_share,
        mass_frac_above_dominance_threshold=mass_above,
        dominant_group_index=[int(x) for x in np.where(tot > 0, dominant, -1)],
    )


def _side_purity(coord_by_group, cut, perm, side_bit):
    """`side_bit(label) -> 0/1` maps a 2x2 quadrant label to which of the 2
    sides of `cut` it belongs to. Returns the mass-weighted fraction of
    each group's OWN cells landing on ITS assigned side (from `perm`, the
    winning `score_2x2` assignment) -- "purity" of the coarser 1-D
    left/right (or top/bottom) split induced by the 2x2 winner."""
    total = correct = 0
    for r, coords in enumerate(coord_by_group):
        n_r = len(coords)
        total += n_r
        if n_r == 0:
            continue
        hi = side_bit(perm[r]) == 1
        correct += int((coords >= cut).sum() if hi else (coords < cut).sum())
    return correct / total if total else None


def _centroid_distance_matrix(centroids, die_extent):
    """Pairwise Euclidean distance between mass centroids, normalized by
    the die's own linear extent (so distances are comparable across
    designs). `centroids[i]` may be `None` (bucket had no placed cells) --
    that row/col is left `None`."""
    n = len(centroids)
    mat = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0.0
                continue
            if centroids[i] is None or centroids[j] is None:
                continue
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            d = math.hypot(dx, dy)
            mat[i][j] = d / die_extent if die_extent else d
    return mat


def classify_arrangement(geometry, top4_keys, top4_centroids, die_bbox):
    """G-D (2026-08-14 T6 adjudication doc sec 4 前置 3): mass-weighted
    hypothesis-assignment classifier, replacing the retracted bbox
    distance-level classifier. Scores three hypotheses (`2x2`/`1x4`/`4x1`)
    by the best group->region assignment's mean own-region mass fraction;
    the winner is whichever scores highest, and the gate only calls it
    *determinable* (`pass_`) if that winner both clears
    `G_D_PASS_SCORE_MIN` and leads the second-best hypothesis by at least
    `G_D_PASS_MARGIN_MIN` (`PASS_2x2 = score_2x2 >= 0.50 and margin >=
    0.10`, generalized to whichever hypothesis is actually the winner).

    `geometry`: `scan_def(..., collect_geometry=True)`'s geometry dict (or
    `None` if the caller didn't collect one -- G-D can't run).
    `top4_keys`: the 4 dominant bucket keys, in the same order as
    `top4_centroids` (each a `[x, y]` mass centroid or `None`)."""
    result = dict()
    if geometry is None or die_bbox is None or len(top4_keys) != 4:
        result.update(pass_=False, arrangement="other", protocol="other",
                       reason="geometry_or_die_bbox_unavailable")
        return result

    key_to_idx = {k: i for i, k in enumerate(geometry["keys"])}
    src_idx = [key_to_idx.get(k) for k in top4_keys]
    if any(i is None for i in src_idx):
        result.update(pass_=False, arrangement="other", protocol="other",
                       reason="a_top4_prefix_had_no_placed_cells")
        return result

    W = die_bbox[2] - die_bbox[0]
    H = die_bbox[3] - die_bbox[1]
    die_extent = max(W, H)
    if W <= 0 or H <= 0:
        result.update(pass_=False, arrangement="other", protocol="other",
                       reason="degenerate_die_bbox")
        return result

    gid_raw = geometry["gid"]
    remap = np.full(len(geometry["keys"]), -1, dtype=np.int64)
    for r, gi in enumerate(src_idx):
        remap[gi] = r
    gid = remap[gid_raw]
    keep = gid >= 0
    gid = gid[keep]
    u_all = (geometry["x"][keep] - die_bbox[0]) / W
    v_all = (geometry["y"][keep] - die_bbox[1]) / H

    u_by_group = [u_all[gid == r] for r in range(4)]
    v_by_group = [v_all[gid == r] for r in range(4)]

    best_2x2 = _score_2x2(u_by_group, v_by_group)
    strips_1x4 = _score_strips(u_by_group)
    strips_4x1 = _score_strips(v_by_group)

    scores = dict(**{"2x2": best_2x2["score"], "1x4": strips_1x4["score"],
                      "4x1": strips_4x1["score"]})
    best_key = max(scores, key=scores.get)
    second_best = max(sc for k, sc in scores.items() if k != best_key)
    margin = scores[best_key] - second_best
    determinable = scores[best_key] >= G_D_PASS_SCORE_MIN and margin >= G_D_PASS_MARGIN_MIN

    bin_dom = _bin_dominance(u_all, v_all, gid)
    cdist = _centroid_distance_matrix(top4_centroids, die_extent)

    lr_purity = _side_purity(u_by_group, best_2x2["cx"], best_2x2["perm"], lambda lbl: lbl // 2)
    tb_purity = _side_purity(v_by_group, best_2x2["cy"], best_2x2["perm"], lambda lbl: lbl % 2)

    quad_names = ["LL", "LU", "RL", "RU"]  # label = 2*[u>=cx] + [v>=cy]
    strip_names = ["s0", "s1", "s2", "s3"]  # increasing along the strip axis
    labels_unescaped = [_unescape(k) for k in top4_keys]

    def _assignment(perm, names):
        return {labels_unescaped[r]: names[perm[r]] for r in range(4)}

    result.update(
        pass_=determinable,
        arrangement=best_key, protocol=best_key,
        scores=scores, winning_score=scores[best_key], margin=margin,
        pass_score_min=G_D_PASS_SCORE_MIN, pass_margin_min=G_D_PASS_MARGIN_MIN,
        score_2x2=dict(score=best_2x2["score"], cx=best_2x2["cx"], cy=best_2x2["cy"],
                        assignment=_assignment(best_2x2["perm"], quad_names),
                        confusion_matrix=best_2x2["M"].tolist()),
        score_1x4=dict(score=strips_1x4["score"],
                        assignment=_assignment(strips_1x4["perm"], strip_names),
                        confusion_matrix=strips_1x4["M"].tolist()),
        score_4x1=dict(score=strips_4x1["score"],
                        assignment=_assignment(strips_4x1["perm"], strip_names),
                        confusion_matrix=strips_4x1["M"].tolist()),
        bin_dominance=bin_dom,
        centroid_distance_matrix=cdist, centroid_distance_labels=labels_unescaped,
        left_right_purity=lr_purity, top_bottom_purity=tb_purity,
        die_extent=die_extent, n_cells_classified=int(len(gid)),
    )
    return result


def run(cluster_def, group_def):
    cluster_buckets, cluster_meta, cluster_geom = scan_def(
        cluster_def, top_level_prefix, collect_geometry=True)
    group_buckets, group_meta, _ = scan_def(group_def, _all_one_bucket)
    standalone = group_buckets.get("_ALL_", _Bucket())

    ranked = sorted(cluster_buckets.items(), key=lambda kv: -kv[1].cells)
    top4 = ranked[:4]
    rest = ranked[4:]
    total_cells = sum(b.cells for b in cluster_buckets.values())
    top4_cells = sum(b.cells for _, b in top4)
    coverage = (top4_cells / total_cells) if total_cells else 0.0

    # G-A: 4 dominant prefixes covering >=95% of cells.
    g_a_pass = len(ranked) >= 4 and coverage >= G_A_COVERAGE_MIN
    g_a = dict(pass_=g_a_pass, coverage=coverage, threshold=G_A_COVERAGE_MIN,
               n_distinct_prefixes=len(ranked), top4_prefixes=[_unescape(k) for k, _ in top4])

    # G-E: top-level residual (everything outside the top4).
    n_top = total_cells - top4_cells
    residual = _Bucket()
    for _, b in rest:
        residual.merge(b)
    g_e = dict(n_top=n_top, n_top_frac=(n_top / total_cells if total_cells else None),
               residual_prefixes=[_unescape(k) for k, _ in rest],
               residual=residual.to_dict())

    # G-B (size) + G-C (cell-type composition) vs standalone `mempool_group`.
    per_group = []
    g_b_pass = g_c_pass = True
    for key, b in top4:
        bd = b.to_dict()
        rel_diff = ((b.cells - standalone.cells) / standalone.cells
                    if standalone.cells else None)
        size_ok = rel_diff is not None and abs(rel_diff) <= G_B_SIZE_TOL
        cos = _cosine(b.celltype, standalone.celltype)
        cos_ok = cos is not None and cos >= G_C_COSINE_MIN
        g_b_pass = g_b_pass and size_ok
        g_c_pass = g_c_pass and cos_ok
        per_group.append(dict(
            prefix=_unescape(key), rel_diff_vs_standalone=rel_diff, size_within_tolerance=size_ok,
            celltype_cosine_vs_standalone=cos, celltype_cosine_pass=cos_ok, **bd,
        ))
    identity_holds = (top4_cells + n_top == total_cells)
    g_b_pass = g_b_pass and identity_holds
    g_b = dict(pass_=g_b_pass, tolerance=G_B_SIZE_TOL, identity_holds=identity_holds,
               standalone_cells=standalone.cells, cluster_total_cells=total_cells,
               top4_cells=top4_cells, n_top=n_top)
    g_c = dict(pass_=g_c_pass, threshold=G_C_COSINE_MIN)

    # G-D: mass-weighted hypothesis-assignment geometric classification.
    top4_keys = [k for k, _ in top4]
    top4_centroids = [pg["centroid"] for pg in per_group]
    g_d = classify_arrangement(cluster_geom, top4_keys, top4_centroids,
                                cluster_meta.get("die_bbox"))
    arrangement, protocol = g_d["arrangement"], g_d["protocol"]

    # Independent corroborating (or contradicting) signal for "cluster ~ 4x
    # group": the *die* itself (not cell placement) should be ~2x the
    # standalone group's die in each linear dimension if the floorplan is a
    # 2x2 array of group-sized quadrants.
    die_bbox = cluster_meta.get("die_bbox")
    standalone_die = group_meta.get("die_bbox")
    die_linear_ratio_vs_standalone = None
    if die_bbox is not None and standalone_die is not None:
        cw, ch = die_bbox[2] - die_bbox[0], die_bbox[3] - die_bbox[1]
        sw, sh = standalone_die[2] - standalone_die[0], standalone_die[3] - standalone_die[1]
        if sw > 0 and sh > 0:
            die_linear_ratio_vs_standalone = [cw / sw, ch / sh]
    g_d.update(die_bbox=die_bbox, standalone_die_bbox=standalone_die,
               die_linear_ratio_vs_standalone=die_linear_ratio_vs_standalone)

    gates_pass = g_a["pass_"] and g_b["pass_"] and g_c["pass_"] and g_d["pass_"]
    verdict = f"PASS_{protocol}" if gates_pass else "FAIL"

    return dict(
        verdict=verdict, arrangement=arrangement, protocol=protocol,
        cluster_def=os.path.abspath(cluster_def), group_def=os.path.abspath(group_def),
        cluster_scan_meta=cluster_meta, group_scan_meta=group_meta,
        standalone_reference=dict(source="group_def_raw_components", **standalone.to_dict()),
        g_a=g_a, g_b=g_b, g_c=g_c, g_d=g_d, g_e=g_e,
        per_group=per_group,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-def", required=True)
    ap.add_argument("--group-def", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    result = run(args.cluster_def, args.group_def)
    result["provenance"] = dict(
        repo_commit=_git_head(REPO),
        cluster_def_sha256=_sha256(args.cluster_def),
        group_def_sha256=_sha256(args.group_def),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[hierarchy_gate] verdict={result['verdict']} arrangement={result['arrangement']} "
          f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
