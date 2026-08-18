"""M4 T6 `cluster_stats.py` (design draft `docs/superpowers/specs/2026-08-13-
m4-scale-up-design-draft.md` sec 3.2 module (b): "T3a 通過後:從 cluster 量跨
group net 統計(正規化後)"; sole authoritative protocol for its contents is
`docs/results/2026-08-14-m4-t6-adjudication.md` sec 2 "glue 統計與距離核" and
sec 4 "T6 本體" table).

Assembles the calibration statistics `glue_gen.py`'s power-law kernel needs
from two already-produced artifacts:

  - `results/m4/corpus/hierarchy_gate.json` (T3a, commit ea91dc2 -- mass-
    centroid + `+`-clause-truncation bugs fixed, verdict `PASS_2x2`): the
    G-D `score_2x2.assignment` (which `gen_groups[i]` sits in which quadrant)
    and the four groups' cell counts.
  - `results/m4/probes/probe_cross_group_stats.json` (T3a/T6 probe, same
    Bug-B-fixed NETS scan as `hierarchy_gate.scan_def`, independently
    written): the six unordered `gen_groups[i]`/`gen_groups[j]` pair net
    counts, net-degree histograms, and pinshare distribution.

Adjacent (grid distance d=1) vs diagonal (d=sqrt(2)) pair classification
comes directly from the G-D assignment's quadrant labels (`"LU"/"RU"/"LL"/
"RL"`, parsed as (col, row)) -- not hardcoded -- so this module stays
correct if a future gate rerun changes which `gen_groups[i]` lands where.

Closed-form 2x2 MoM (`glue_gen.solve_mom_2x2`, sec 3.2): `lambda_adj`/
`lambda_diag` (means of the 4 adjacent / 2 diagonal raw pair counts) solve
for `(lambda_0, alpha)`.

Overdispersion: `var/mean` (ddof=1) of the six raw pair counts; sec 3.2(c)'s
`var/mean > 3` rule flags every Poisson-premised test as unreliable (2026-
08-14 adjudication sec 2b: measured 16.9, publishable statement is `|alpha|
<= 0.2`, main-line `alpha=0.10`, "distance response not statistically
resolved").

`lambda_0_tile`: a Rent-consistent scale conversion of `lambda_0` (measured
on real cluster groups, avg 2,824,617 cells = `g_b.top4_cells/4`) to the
actual `mempool_group` standalone Bookshelf tile size `tile_bookshelf.py`
replicates (3,077,989 cells = `g_b.standalone_cells`, 8.97% larger) --
`lambda_0 * (tile_cells/group_cells)^p`, `p` = mean of the four groups'
single-point Rent-exponent estimate `log(per_group_crossing) /
log(per_group_cells)` (>= 0.630, <= 0.636 across the four groups). This
reproduces the adjudication doc's `lambda_0 x (3.0780/2.8246)^0.633 = 4387`
(3.078M/2.8246M being the same two cell counts in millions, 0.633 the same
mean-p, rounded).

Net-level bootstrap (adjudication sec 4 "不確定度 net-level bootstrap
>=1000"): rescans `mempool_cluster.def`'s `NETS` section once (same Bug-B-
fixed state machine as `hierarchy_gate.scan_def`/`probe_cross_group_stats.
run`, independently written here as a third cross-check), retaining only
the ~22.4K nets that touch >=2 of the four `gen_groups[i]` -- each net's
"which groups" bitmask is all a bootstrap replicate needs, not its pins.
Each replicate then draws a Poisson(1) resample weight per informative net
(the large-N Poisson-bootstrap approximation to nonparametric case
resampling over the full ~12.7M-net population: a net's case-resampling
count under `Binomial(n=12712800, p=1/12712800)` is `Poisson(1)` to > 4
significant figures at this N -- the ~12.69M zero-touching nets contribute
0 to every replicate under either the exact or the approximate scheme, so
only the informative nets' weights need be drawn), recomputes the six pair
sums, and refits `(lambda_0, alpha)`. >=1000 replicates -> percentile 95%
CI.

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.cluster_stats \\
        --hierarchy-gate results/m4/corpus/hierarchy_gate.json \\
        --cross-group-stats results/m4/probes/probe_cross_group_stats.json \\
        --out results/m4/bench/cluster_stats.json
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter

import numpy as np

from ioplace.bench import glue_gen

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"

# sec 3.2(c): var/mean > 3 over the six raw pair counts -> every
# Poisson-premised test (incl. the bootstrap CI's implicit sampling model)
# must be flagged unreliable in the report.
OVERDISPERSION_UNRELIABLE_THRESHOLD = 3.0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _quadrant_rc(label):
    """"LU"/"RU"/"LL"/"RL" -> (col, row): col 0=left/1=right (1st char),
    row 0=up/1=low (2nd char, 'U' vs 'L')."""
    col = 0 if label[0] == "L" else 1
    row = 0 if label[1] == "U" else 1
    return col, row


def classify_pairs(assignment):
    """`assignment`: {"gen_groups[i].i_group": quadrant_label, ...} (G-D
    `score_2x2.assignment`). Returns {(gi, gj): "adjacent"|"diagonal"} for
    every unordered pair of the 4 group indices (0-3), gi < gj."""
    rc = {}
    for name, label in assignment.items():
        i = int(name.split("[", 1)[1].split("]", 1)[0])
        rc[i] = _quadrant_rc(label)
    out = {}
    idxs = sorted(rc)
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            gi, gj = idxs[a], idxs[b]
            (ci, ri), (cj, rj) = rc[gi], rc[gj]
            out[(gi, gj)] = "adjacent" if (ci == cj or ri == rj) else "diagonal"
    return out


def _raw_pair_counts(six_group_pair_net_counts, pair_kind):
    """`six_group_pair_net_counts`: probe's `{"gen_groups\\[i\\].i_group|
    gen_groups\\[j\\].i_group": count}` (raw DEF-escaped keys). Returns
    {(gi, gj): count} restricted to the 6 real-group pairs `pair_kind`
    classifies (drops any dma/idma special-prefix pairs the probe also
    carries)."""
    out = {}
    for (gi, gj) in pair_kind:
        key_a = f"gen_groups\\[{gi}\\].i_group|gen_groups\\[{gj}\\].i_group"
        key_b = f"gen_groups\\[{gj}\\].i_group|gen_groups\\[{gi}\\].i_group"
        if key_a in six_group_pair_net_counts:
            out[(gi, gj)] = six_group_pair_net_counts[key_a]
        elif key_b in six_group_pair_net_counts:
            out[(gi, gj)] = six_group_pair_net_counts[key_b]
        else:
            raise KeyError(f"neither {key_a!r} nor {key_b!r} in six_group_pair_net_counts")
    return out


def overdispersion(counts):
    """sample var/mean (ddof=1) of `counts` -- sec 3.2(c)'s overdispersion
    statistic. Returns (var_over_mean, unreliable: bool)."""
    arr = np.asarray(list(counts), dtype=np.float64)
    mean = float(arr.mean())
    var = float(arr.var(ddof=1))
    ratio = var / mean if mean else float("nan")
    return ratio, ratio > OVERDISPERSION_UNRELIABLE_THRESHOLD


def rent_p_scale_exponent(hierarchy_gate, cross_group_stats):
    """Mean of the four groups' single-point Rent-exponent estimate
    `log(per_group_crossing) / log(per_group_cells)` -- the `p` the
    adjudication doc's `lambda_0_tile` scale conversion uses."""
    cells = {e["prefix"]: e["cells"] for e in hierarchy_gate["per_group"]}
    crossing = cross_group_stats["per_group_crossing"]
    ps = []
    for i in range(4):
        name = f"gen_groups[{i}].i_group"
        key = f"gen_groups\\[{i}\\].i_group"
        ps.append(math.log(crossing[key]) / math.log(cells[name]))
    return float(np.mean(ps)), ps


def lambda_0_tile_scale(lambda_0, standalone_cells, avg_group_cells, p):
    ratio = standalone_cells / avg_group_cells
    return lambda_0 * ratio ** p, ratio


# ---------------------------------------------------------------------------
# Net-level bootstrap: one DEF rescan -> per-net group-touch bitmask, then
# >=1000 in-memory Poisson(1)-weighted resamples.
# ---------------------------------------------------------------------------

def _group_key(name):
    i = name.find("/")
    return name[:i] if i != -1 else None


_GROUP_PREFIXES = [f"gen_groups\\[{i}\\].i_group" for i in range(4)]
_GROUP_INDEX = {p: i for i, p in enumerate(_GROUP_PREFIXES)}


def scan_net_group_bitmasks(def_path):
    """Streams `def_path`'s `NETS` section (same Bug-B-fixed `+`-clause-
    stop / wildcard-tuple-exclusion state machine as `hierarchy_gate.
    scan_def` and `probe_cross_group_stats.run`, independently written here
    as a third cross-check) and returns a uint8 array of length (# nets
    touching >=2 of the four `gen_groups[i]`) holding each such net's
    4-bit group-touch bitmask (bit i set iff the net has >=1 pin in
    `gen_groups[i]`). Also returns `six_group_pair_net_counts` recomputed
    from these bitmasks, for a self-check against
    `probe_cross_group_stats.json`'s own count."""
    section = "pre"
    cur = None
    in_pins = True
    masks = []
    pair_check = Counter()

    def flush():
        if cur is None or cur == 0:
            return
        idxs = [i for i in range(4) if cur & (1 << i)]
        if len(idxs) >= 2:
            masks.append(cur)
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pair_check[(idxs[a], idxs[b])] += 1

    t0 = time.time()
    with open(def_path, "r", buffering=1 << 24) as f:
        for line in f:
            if section == "pre":
                if line.startswith("NETS "):
                    section = "nets"
                continue
            if line.startswith("END NETS"):
                flush()
                break
            if line.startswith("- "):
                flush()
                cur = 0
                in_pins = True
            if not in_pins:
                continue
            if "(" not in line and "+" not in line:
                continue
            toks = line.split()
            i, n = 0, len(toks)
            while i < n:
                t = toks[i]
                if t == "+":
                    in_pins = False
                    break
                if t == "(" and i + 3 < n and toks[i + 3] == ")":
                    inst = toks[i + 1]
                    if inst not in ("PIN", "*") and "/" in inst:
                        gk = _group_key(inst)
                        gi = _GROUP_INDEX.get(gk)
                        if gi is not None:
                            cur |= (1 << gi)
                    i += 4
                else:
                    i += 1
    scan_s = time.time() - t0
    return (np.asarray(masks, dtype=np.uint8), scan_s,
            {(a, b): v for (a, b), v in pair_check.items()})


_BIT_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def _pair_sums_from_masks(masks, weights):
    """masks: uint8 array (bit i set iff net touches group i); weights:
    same-length array of per-net multiplicities. Returns {(gi,gj): weighted
    sum of nets touching both}."""
    out = {}
    for (gi, gj) in _BIT_PAIRS:
        both = ((masks & (1 << gi)) != 0) & ((masks & (1 << gj)) != 0)
        out[(gi, gj)] = float(weights[both].sum())
    return out


def bootstrap_lambda_alpha(masks, pair_kind, n_bootstrap=1200, seed=0, alpha_level=0.05):
    """>=1000-replicate net-level bootstrap 95% CI for (lambda_0, alpha).
    `pair_kind`: {(gi,gj): "adjacent"|"diagonal"} from `classify_pairs`.
    Poisson(1) case-weight approximation -- see module docstring."""
    rng = np.random.default_rng(seed)
    adj_pairs = [p for p, k in pair_kind.items() if k == "adjacent"]
    diag_pairs = [p for p, k in pair_kind.items() if k == "diagonal"]
    n = len(masks)
    lam0s, alphas, lam_adjs, lam_diags = [], [], [], []
    for _ in range(n_bootstrap):
        w = rng.poisson(1.0, size=n).astype(np.float64)
        sums = _pair_sums_from_masks(masks, w)
        lam_adj = float(np.mean([sums[p] for p in adj_pairs]))
        lam_diag = float(np.mean([sums[p] for p in diag_pairs]))
        if lam_adj <= 0 or lam_diag <= 0:
            continue
        res = glue_gen.solve_mom_2x2(lam_adj, lam_diag)
        if res.status != "ok":
            continue
        lam0s.append(res.lambda_0)
        alphas.append(res.alpha)
        lam_adjs.append(lam_adj)
        lam_diags.append(lam_diag)

    def _ci(values):
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = np.quantile(arr, [alpha_level / 2, 1 - alpha_level / 2])
        return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(arr))}

    return {
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_used": len(lam0s),
        "seed": seed,
        "lambda_0_ci": _ci(lam0s),
        "alpha_ci": _ci(alphas),
        "lambda_adj_ci": _ci(lam_adjs),
        "lambda_diag_ci": _ci(lam_diags),
    }


def run(hierarchy_gate_path, cross_group_stats_path, cluster_def=None,
        n_bootstrap=1200, seed=0):
    with open(hierarchy_gate_path) as f:
        hg = json.load(f)
    with open(cross_group_stats_path) as f:
        cg = json.load(f)

    if hg["verdict"] != "PASS_2x2":
        raise ValueError(f"hierarchy_gate verdict={hg['verdict']!r}, expected PASS_2x2 "
                          f"(T3a must pass before cluster_stats can run)")

    assignment = hg["g_d"]["score_2x2"]["assignment"]
    pair_kind = classify_pairs(assignment)
    raw_counts = _raw_pair_counts(cg["six_group_pair_net_counts"], pair_kind)

    adj_counts = [raw_counts[p] for p, k in pair_kind.items() if k == "adjacent"]
    diag_counts = [raw_counts[p] for p, k in pair_kind.items() if k == "diagonal"]
    lambda_adj = float(np.mean(adj_counts))
    lambda_diag = float(np.mean(diag_counts))

    all_six = list(raw_counts.values())
    overdisp_ratio, overdisp_unreliable = overdispersion(all_six)

    mom = glue_gen.solve_mom_2x2(lambda_adj, lambda_diag)

    standalone_cells = hg["g_b"]["standalone_cells"]
    avg_group_cells = hg["g_b"]["top4_cells"] / 4.0
    p_scale, p_scale_per_group = rent_p_scale_exponent(hg, cg)
    lambda_0_tile, tile_group_ratio = lambda_0_tile_scale(
        mom.lambda_0, standalone_cells, avg_group_cells, p_scale)

    result = {
        "assignment": assignment,
        "pair_kind": {f"{gi}|{gj}": k for (gi, gj), k in pair_kind.items()},
        "six_pair_raw_counts": {f"{gi}|{gj}": v for (gi, gj), v in raw_counts.items()},
        "lambda_adj": lambda_adj,
        "lambda_diag": lambda_diag,
        "overdispersion_var_over_mean": overdisp_ratio,
        "overdispersion_unreliable": overdisp_unreliable,
        "overdispersion_threshold": OVERDISPERSION_UNRELIABLE_THRESHOLD,
        "mom": {"lambda_0": mom.lambda_0, "alpha": mom.alpha, "status": mom.status,
                "detail": mom.detail},
        "lambda_0_tile": {
            "value": lambda_0_tile,
            "standalone_group_cells": standalone_cells,
            "avg_cluster_group_cells": avg_group_cells,
            "cell_ratio": tile_group_ratio,
            "rent_p_used": p_scale,
            "rent_p_per_group": p_scale_per_group,
            "note": "lambda_0 * (standalone_cells/avg_cluster_group_cells)^rent_p_used "
                    "-- Rent-consistent scale conversion from the real (smaller) cluster "
                    "group size to the actual (larger) mempool_group Bookshelf tile size "
                    "tile_bookshelf.py replicates; reproduces the adjudication doc's "
                    "lambda_0 x (3.0780/2.8246)^0.633 = 4387",
        },
        "degree_histogram": {
            "deg_hist_all": cg["deg_hist_all"],
            "deg_hist_cross": cg["deg_hist_cross"],
        },
        "pinshare_top25": cg["pinshare_top25"],
        "ngroups_hist": cg["ngroups_hist"],
        "cross_pin_total": cg["cross_pin_total"],
    }

    if cluster_def:
        masks, scan_s, pair_check = scan_net_group_bitmasks(cluster_def)
        mismatches = {f"{a}|{b}": (v, raw_counts.get((a, b)))
                      for (a, b), v in pair_check.items() if v != raw_counts.get((a, b))}
        boot = bootstrap_lambda_alpha(masks, pair_kind, n_bootstrap=n_bootstrap, seed=seed)
        result["net_level_bootstrap"] = boot
        result["bootstrap_rescan"] = {
            "def_path": os.path.abspath(cluster_def),
            "scan_s": scan_s,
            "n_informative_nets": int(len(masks)),
            "self_check_pair_counts": {f"{a}|{b}": v for (a, b), v in pair_check.items()},
            "self_check_mismatches_vs_probe": mismatches,
        }

    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy-gate", default=os.path.join(REPO, "results", "m4", "corpus",
                                                               "hierarchy_gate.json"))
    ap.add_argument("--cross-group-stats", default=os.path.join(REPO, "results", "m4", "probes",
                                                                  "probe_cross_group_stats.json"))
    ap.add_argument("--cluster-def", default=None,
                     help="if given, rescans this DEF's NETS section for the net-level "
                          "bootstrap CI; defaults to hierarchy_gate.json's own cluster_def")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "m4", "bench",
                                                    "cluster_stats.json"))
    args = ap.parse_args(argv)

    cluster_def = args.cluster_def
    if cluster_def is None and not args.no_bootstrap:
        with open(args.hierarchy_gate) as f:
            cluster_def = json.load(f)["cluster_def"]

    result = run(args.hierarchy_gate, args.cross_group_stats,
                 cluster_def=None if args.no_bootstrap else cluster_def,
                 n_bootstrap=args.n_bootstrap, seed=args.seed)
    result["provenance"] = dict(
        repo_commit=_git_head(REPO),
        hierarchy_gate_sha256=_sha256(args.hierarchy_gate),
        cross_group_stats_sha256=_sha256(args.cross_group_stats),
        cluster_def_sha256=_sha256(cluster_def) if cluster_def else None,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[cluster_stats] lambda_adj={result['lambda_adj']:.1f} "
          f"lambda_diag={result['lambda_diag']:.1f} "
          f"alpha={result['mom']['alpha']:.4f} lambda_0={result['mom']['lambda_0']:.1f} "
          f"lambda_0_tile={result['lambda_0_tile']['value']:.1f} "
          f"overdispersion={result['overdispersion_var_over_mean']:.1f} "
          f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
