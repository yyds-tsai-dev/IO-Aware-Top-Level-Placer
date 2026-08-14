"""M4 T3a hierarchy gate (design draft `docs/superpowers/specs/2026-08-13-
m4-scale-up-design-draft.md` sec 3.1a): streams `mempool_cluster.def`'s
`COMPONENTS`/`NETS` sections (9.7 GB text) -- and, for comparison,
`mempool_group.def`'s own `COMPONENTS`/`NETS` sections (2.3 GB) -- to test
whether "cluster = 4 groups" actually holds, and if so what geometric
arrangement the 4 groups are in (2x2 / 1x4 / other). This is a *blocking*
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
cell names (~150) -- negligible next to the file sizes.

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
  - `NETS` records list pins as `( instname pinname )` tuples, possibly
    wrapped across many lines for high-fanout nets (the pre-CTS `clk_i` net
    in `mempool_cluster.def` alone spans >1M lines); a `( PIN name )` tuple
    refers to a top-level design port, not an instance, and is excluded
    from per-prefix pin counts. Net records also terminate with a lone
    ` ;` line.

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.hierarchy_gate \\
        --cluster-def <mempool_cluster.def> --group-def <mempool_group.def> \\
        --out results/m4/corpus/hierarchy_gate.json
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"

STATUS_WORDS = frozenset({"FIXED", "COVER", "PLACED", "UNPLACED"})

# sec 3.1a thresholds.
G_A_COVERAGE_MIN = 0.95
G_B_SIZE_TOL = 0.15
G_C_COSINE_MIN = 0.95
# Geometric-arrangement classification tolerance: two centroid-pair
# distances are treated as "the same distance level" if they differ by less
# than this fraction of their level's running-average distance. The real
# cluster's 4 groups are hand-placed (not abutted tiler output), so this
# needs more slack than an exact-congruence check would.
_DIST_REL_TOL = 0.10
# Diagnostic-only threshold (does not gate pass/fail): centroid separations
# below this fraction of the die's own extent are flagged as "no meaningful
# spatial separation" rather than trusted as a real 2x2/1x4 signal.
_COLOCATED_REL_TOL = 0.05

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
                 "nets_crossing_or_top", "xmin", "xmax", "ymin", "ymax")

    def __init__(self):
        self.cells = 0
        self.fixed_cells = 0
        self.celltype = Counter()
        self.pins = 0
        self.nets_internal = 0
        self.nets_crossing_or_top = 0
        self.xmin = self.ymin = math.inf
        self.xmax = self.ymax = -math.inf

    def add_coord(self, x, y):
        if x < self.xmin: self.xmin = x
        if x > self.xmax: self.xmax = x
        if y < self.ymin: self.ymin = y
        if y > self.ymax: self.ymax = y

    def merge(self, other):
        self.cells += other.cells
        self.fixed_cells += other.fixed_cells
        self.celltype += other.celltype
        self.pins += other.pins
        self.nets_internal += other.nets_internal
        self.nets_crossing_or_top += other.nets_crossing_or_top
        self.xmin = min(self.xmin, other.xmin); self.xmax = max(self.xmax, other.xmax)
        self.ymin = min(self.ymin, other.ymin); self.ymax = max(self.ymax, other.ymax)

    def to_dict(self):
        has_bbox = self.xmin != math.inf
        bbox = [self.xmin, self.ymin, self.xmax, self.ymax] if has_bbox else None
        centroid = [(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2] if has_bbox else None
        return dict(
            cells=self.cells, fixed_cells=self.fixed_cells,
            movable_cells=self.cells - self.fixed_cells,
            pins=self.pins, nets_internal=self.nets_internal,
            nets_touching=self.nets_internal + self.nets_crossing_or_top,
            celltype_hist={k: v for k, v in self.celltype.most_common()},
            bbox=bbox, centroid=centroid,
        )


def scan_def(def_path, group_key):
    """Single streaming pass over `def_path`'s `COMPONENTS` and `NETS`
    sections (everything before `COMPONENTS` and after `END NETS` is
    skipped without being tokenized). `group_key(raw_instance_name) ->
    str` assigns each instance to a bucket key. Returns `(buckets: {key:
    _Bucket}, meta: dict)`."""
    buckets = defaultdict(_Bucket)
    declared_components = declared_nets = die_bbox = None
    scanned_components = scanned_nets = scanned_pin_tuples = skipped_top_port_tuples = 0
    section = "pre"  # pre -> components -> between -> nets -> done
    cur_net_prefixes = None  # set[str] | None, only while inside a NETS record

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
                    b.add_coord(float(toks[status_idx + 2]), float(toks[status_idx + 3]))
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
                if "(" not in line:
                    continue
                toks = line.split()
                i, n = 0, len(toks)
                while i < n:
                    # A pin tuple is exactly 4 tokens: `( instname pinname
                    # )`, including the `( PIN <portname> )` case for
                    # top-level design ports.
                    if toks[i] == "(" and i + 3 < n and toks[i + 3] == ")":
                        inst = toks[i + 1]
                        if inst != "PIN":
                            key = group_key(inst)
                            buckets[key].pins += 1
                            scanned_pin_tuples += 1
                            if cur_net_prefixes is not None:
                                cur_net_prefixes.add(key)
                        else:
                            skipped_top_port_tuples += 1
                        i += 4
                    else:
                        i += 1
                continue
    elapsed = time.time() - t0

    meta = dict(
        declared_components=declared_components, scanned_components=scanned_components,
        declared_nets=declared_nets, scanned_nets=scanned_nets,
        scanned_pin_tuples=scanned_pin_tuples, skipped_top_port_tuples=skipped_top_port_tuples,
        scan_s=elapsed, die_bbox=die_bbox,
    )
    return dict(buckets), meta


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


def _classify_arrangement(centroids):
    """`centroids`: list of 4 `(cx, cy)`. Returns `(arrangement,
    distance_levels)`; `distance_levels` is a list of `{dist, pair_count,
    pairs}` sorted ascending -- sec 3.1a G-D / sec 3.2's "6 unordered pairs
    / distinct distance levels" DoF accounting."""
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            pairs.append((i, j, math.hypot(dx, dy)))
    pairs.sort(key=lambda p: p[2])

    levels = []
    for i, j, d in pairs:
        placed = False
        for lvl in levels:
            if lvl["dist"] > 0 and abs(d - lvl["dist"]) / lvl["dist"] <= _DIST_REL_TOL:
                lvl["pairs"].append([i, j])
                lvl["dist"] = (lvl["dist"] * lvl["pair_count"] + d) / (lvl["pair_count"] + 1)
                lvl["pair_count"] += 1
                placed = True
                break
        if not placed:
            levels.append({"dist": d, "pair_count": 1, "pairs": [[i, j]]})

    counts = [lvl["pair_count"] for lvl in levels]
    if len(levels) == 2 and counts == [4, 2]:
        arrangement = "2x2"
    elif len(levels) == 3 and counts == [3, 2, 1]:
        arrangement = "1x4"
    else:
        arrangement = "other"
    return arrangement, levels


def run(cluster_def, group_def):
    cluster_buckets, cluster_meta = scan_def(cluster_def, top_level_prefix)
    group_buckets, group_meta = scan_def(group_def, _all_one_bucket)
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

    # G-D: geometric arrangement from group centroids.
    centroids = [pg["centroid"] for pg in per_group]
    bbox_ok = all(c is not None for c in centroids)
    if bbox_ok:
        arrangement, distance_levels = _classify_arrangement(centroids)
    else:
        arrangement, distance_levels = "other", []
    # sec 3.1a's degenerate-arrangement fallback: an "other" geometry still
    # needs a protocol assignment for T6 (>=3 distinct distance levels ->
    # follow the 1x4 hold-out protocol; ==2 -> follow the 2x2 protocol,
    # acknowledging 0 lack-of-fit DoF) -- this does not affect G-D pass/fail,
    # which only requires that an arrangement be determinable at all.
    protocol = arrangement
    if arrangement == "other":
        protocol = "1x4" if len(distance_levels) >= 3 else "2x2"

    # Diagnostic (not itself a pass/fail input): on a *placed* chip (as
    # opposed to a synthetic tiler's abutted output), instance-hierarchy
    # membership need not correlate with physical placement locality --
    # Innovus's placer is free to scatter a group's own cells anywhere that
    # minimizes wirelength/congestion, including across the whole die. If
    # every group's bbox already spans ~the full die, the resulting
    # centroids collapse to ~the die center regardless of `arrangement`'s
    # nominal classification, and the `protocol` selection above is not
    # evidence of a genuine spatial layout -- flagged here so it isn't
    # silently treated as a confident 2x2/1x4 physical tiling signal.
    die_bbox = cluster_meta.get("die_bbox")
    max_pair_dist = max((lvl["dist"] for lvl in distance_levels), default=None)
    die_extent = geometric_separation_negligible = sep_frac = None
    if die_bbox is not None:
        die_extent = max(die_bbox[2] - die_bbox[0], die_bbox[3] - die_bbox[1])
        if max_pair_dist is not None and die_extent > 0:
            sep_frac = max_pair_dist / die_extent
            geometric_separation_negligible = sep_frac < _COLOCATED_REL_TOL

    # Independent corroborating (or contradicting) signal for "cluster ~ 4x
    # group": the *die* itself (not cell placement) should be ~2x the
    # standalone group's die in each linear dimension if the floorplan is a
    # 2x2 array of group-sized quadrants.
    standalone_die = group_meta.get("die_bbox")
    die_linear_ratio_vs_standalone = None
    if die_bbox is not None and standalone_die is not None:
        cw, ch = die_bbox[2] - die_bbox[0], die_bbox[3] - die_bbox[1]
        sw, sh = standalone_die[2] - standalone_die[0], standalone_die[3] - standalone_die[1]
        if sw > 0 and sh > 0:
            die_linear_ratio_vs_standalone = [cw / sw, ch / sh]

    g_d = dict(pass_=bbox_ok, arrangement=arrangement, protocol=protocol,
               bbox_available=bbox_ok, distance_levels=distance_levels,
               die_bbox=die_bbox, die_extent=die_extent,
               max_pairwise_centroid_dist=max_pair_dist,
               max_centroid_separation_frac_of_die=sep_frac,
               geometric_separation_negligible=geometric_separation_negligible,
               standalone_die_bbox=standalone_die,
               die_linear_ratio_vs_standalone=die_linear_ratio_vs_standalone,
               group_bboxes=[dict(prefix=pg["prefix"], bbox=pg["bbox"], centroid=pg["centroid"])
                             for pg in per_group])

    gates_pass = g_a["pass_"] and g_b["pass_"] and g_c["pass_"] and g_d["pass_"]
    if not gates_pass:
        verdict = "FAIL"
    elif protocol == "1x4":
        verdict = "PASS_1x4"
    else:
        verdict = "PASS"

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
