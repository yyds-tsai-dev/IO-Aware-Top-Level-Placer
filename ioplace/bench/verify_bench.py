"""M4 T5 verifier (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 3.3).

fit vs holdout (Codex #6): **fit metrics never gate pass/fail** (their
values were fit *from* the data they're being compared to -- reporting
them here is a documentation aid, not a check function returning
status). **Holdout metrics gate, are pre-registered, and are evaluated
exactly once** -- there is deliberately no "keep tuning until it passes"
path through this module's functions.

This task's scope (design draft's T5 row): V0 (file-level structural
integrity) and H5' (the construction method's arithmetic self-check,
downgraded in v2 from a calibration target/V5) are fully implemented and
hard-gating. V1 (DREAMPlace-readable) and the H1-H4 holdout metrics are
*framework* here -- each needs a real large-scale comparison case
(`mempool_cluster` vs. a calibrated 2x2/1x4 synthetic array) that is a
product of T3a/T6, neither of which exists yet -- so every one of those
functions computes its metric when given real inputs, but returns
`status="not_run"` when the required comparison data isn't supplied
(never silently "passes").
"""
import os

from ioplace.bench import tile_bookshelf as tb


def _count_nodes_body(nodes_path):
    names = []
    with open(nodes_path) as f:
        started = False
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumTerminals"):
                    started = True
                continue
            if not s:
                continue
            names.append(s.split()[0])
    return names


def _iter_nets(nets_path):
    """Yields (netname, [(nodename, direction), ...]) per net block."""
    with open(nets_path) as f:
        started = False
        cur_name, cur_pins = None, []
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumPins"):
                    started = True
                continue
            if not s:
                continue
            if s.startswith("NetDegree"):
                if cur_name is not None:
                    yield cur_name, cur_pins
                _, rest = s.split(":", 1)
                _, cur_name = rest.split()
                cur_pins = []
            else:
                parts = s.split()
                cur_pins.append((parts[0], parts[1]))
        if cur_name is not None:
            yield cur_name, cur_pins


def _scan_nets(nets_path):
    """Single streaming pass over `_iter_nets(nets_path)`: (n_nets,
    n_pins, n_self_loop, n_degree_le1, referenced) -- `referenced` is the
    set of every node name touched by any net (bounded by node count, not
    pin count). Shared by the target and (when given) source scans in
    `check_v0_structural` below so neither materializes `_iter_nets`'s
    output as a list."""
    n_nets = n_pins = n_self_loop = n_degree_le1 = 0
    referenced = set()
    for _, pins in _iter_nets(nets_path):
        n_nets += 1
        n_pins += len(pins)
        names = [p[0] for p in pins]
        if len(set(names)) < len(names):
            n_self_loop += 1
        if len(pins) <= 1:
            n_degree_le1 += 1
        referenced.update(names)
    return n_nets, n_pins, n_self_loop, n_degree_le1, referenced


def check_v0_structural(prefix, manifest=None, source_prefix=None, allowed_unconnected=()):
    """Sec 3.3 V0, hard/mechanical: `.aux`-referenced files all exist;
    NumNodes/NumNets/NumPins headers match actual body line counts and
    (if `manifest` given) the manifest's base+glue totals; the design's
    self-loop and degree<=1 net fractions are each not higher than
    `source_prefix`'s (if given); every node is referenced by at least one
    net or is explicitly in `allowed_unconnected` (or, if both `manifest`
    and `source_prefix` are given, the unreferenced count is exactly the
    source's own unreferenced count x R x C -- see below); `.scl` row
    coverage has no overlap. Returns
    {"status": "ok"|"fail", "checks": {...}, "errors": [...]}.

    **2026-08-15 T6 holdout adjudication sec 4.2/8-4, three sub-checks
    reinterpreted (not relaxed -- these were misfiring on every real
    array):**

    1. Self-loops (a net with a repeated node) are legal and common in
       real hierarchical netlists -- `mempool_group` itself has 173,488
       (4.951% of its own nets). Judged relative to `source_prefix`
       instead of requiring exactly zero, same shape as the
       `degree_le1_not_worse_than_source` check this reuses.
    2. `.scl` row coverage now only checks for *overlap*, not gap -- a
       macro's footprint legitimately has no placement row under it (any
       macro-containing design), matching `tile_bookshelf.
       assert_rows_no_overlap_no_gap` (`tile_bookshelf.py:285`), which
       made this exact call during T6 construction; this brings
       `verify_bench` back in sync with that reinterpretation.
    3. Unreferenced-node count: the tiler replicates the source exactly
       R*C times and glue nets only ever *add* pin references, so a node
       unreferenced in the source is unreferenced in every tile-copy of
       it -- checked as `unreferenced_count == source_count * R * C`
       (needs both `manifest` and `source_prefix`; falls back to the old
       "must be exactly zero, minus declared exceptions" bar otherwise).

    **2026-08-15 scale fix (sec 4.2/8-4):** the target `.nets` file is
    scanned in a single streaming pass (`_scan_nets`, reused for the
    optional source scan) instead of materializing
    `list(_iter_nets(...))` -- fine at 2x2 (48M pins), but a 3x3 array's
    nets file is 108M pins, and holding that as a list of Python objects
    multiplies its on-disk footprint several-fold in RAM (the same
    failure mode `glue_gen.append_glue_nets` was already fixed for)."""
    errors = []
    checks = {}

    try:
        paths = tb.read_aux(prefix + ".aux")
    except Exception as e:
        return {"status": "fail", "checks": {"aux_readable": False}, "errors": [f".aux unreadable: {e}"]}
    checks["aux_readable"] = True

    checks["aux_files_exist"] = True
    for suf, p in paths.items():
        if not os.path.exists(p):
            checks["aux_files_exist"] = False
            errors.append(f"missing file referenced by .aux ({suf}): {p}")
    if not checks["aux_files_exist"]:
        return {"status": "fail", "checks": checks, "errors": errors}

    n_nodes_hdr, n_terminals_hdr = tb._nodes_header(paths["nodes"])
    node_names = _count_nodes_body(paths["nodes"])
    checks["nodes_header_matches_body"] = (len(node_names) == n_nodes_hdr)
    if len(node_names) != n_nodes_hdr:
        errors.append(f"NumNodes header={n_nodes_hdr} but body has {len(node_names)} records")

    n_nets_hdr, n_pins_hdr = tb._nets_header(paths["nets"])
    n_nets_body, n_pins_body, n_self_loop, n_degree_le1, referenced = _scan_nets(paths["nets"])
    checks["nets_header_matches_body"] = (n_nets_body == n_nets_hdr and n_pins_body == n_pins_hdr)
    if n_nets_body != n_nets_hdr:
        errors.append(f"NumNets header={n_nets_hdr} but body has {n_nets_body} net blocks")
    if n_pins_body != n_pins_hdr:
        errors.append(f"NumPins header={n_pins_hdr} but body has {n_pins_body} pin records")

    if manifest is not None:
        base = manifest["base"]
        glue = manifest.get("glue") or {"n_nets": 0, "n_pins": 0}
        expect_nodes = base["n_nodes"]
        expect_nets = base["n_nets"] + glue.get("n_nets", 0)
        expect_pins = base["n_pins"] + glue.get("n_pins", 0)
        checks["matches_manifest"] = (n_nodes_hdr == expect_nodes and n_nets_hdr == expect_nets
                                       and n_pins_hdr == expect_pins)
        if not checks["matches_manifest"]:
            errors.append(
                f"header (nodes={n_nodes_hdr},nets={n_nets_hdr},pins={n_pins_hdr}) != "
                f"manifest base+glue (nodes={expect_nodes},nets={expect_nets},pins={expect_pins})")

    self_loop_fraction = (n_self_loop / n_nets_body) if n_nets_body else 0.0
    checks["self_loop_fraction"] = self_loop_fraction

    frac_le1 = (n_degree_le1 / n_nets_body) if n_nets_body else 0.0
    checks["degree_le1_fraction"] = frac_le1

    src_scan = None
    if source_prefix is not None:
        src_paths = tb.read_aux(source_prefix + ".aux")
        src_scan = _scan_nets(src_paths["nets"])

        n_src_nets, _, n_src_self_loop, n_src_degree_le1, _ = src_scan
        src_self_loop_fraction = (n_src_self_loop / n_src_nets) if n_src_nets else 0.0
        checks["source_self_loop_fraction"] = src_self_loop_fraction
        checks["self_loop_fraction_not_worse_than_source"] = (
            self_loop_fraction <= src_self_loop_fraction + 1e-9)
        if self_loop_fraction > src_self_loop_fraction + 1e-9:
            errors.append(f"self-loop fraction {self_loop_fraction:.6f} ({n_self_loop} nets) > "
                           f"source's {src_self_loop_fraction:.6f} ({n_src_self_loop} nets)")

        src_frac = (n_src_degree_le1 / n_src_nets) if n_src_nets else 0.0
        checks["degree_le1_not_worse_than_source"] = (frac_le1 <= src_frac + 1e-9)
        if frac_le1 > src_frac + 1e-9:
            errors.append(f"degree<=1 fraction {frac_le1:.4f} > source's {src_frac:.4f}")

    allowed = set(allowed_unconnected)
    unreferenced_count = sum(1 for n in node_names if n not in referenced and n not in allowed)
    checks["unreferenced_count"] = unreferenced_count
    if source_prefix is not None and manifest is not None:
        src_paths = tb.read_aux(source_prefix + ".aux")
        src_node_names = _count_nodes_body(src_paths["nodes"])
        src_referenced = src_scan[4]
        source_unreferenced_count = sum(1 for n in src_node_names if n not in src_referenced)
        expect_unreferenced = source_unreferenced_count * manifest["R"] * manifest["C"]
        checks["source_unreferenced_count"] = source_unreferenced_count
        checks["all_nodes_referenced_or_declared_unconnected"] = (
            unreferenced_count == expect_unreferenced)
        if unreferenced_count != expect_unreferenced:
            errors.append(f"unreferenced node count {unreferenced_count} != source_unreferenced_count "
                           f"({source_unreferenced_count}) x R x C ({manifest['R']}x{manifest['C']}"
                           f"={expect_unreferenced})")
    else:
        # framework fallback (no source_prefix/manifest pair to compare
        # against): keep the pre-2026-08-15 "must be exactly zero, minus
        # declared exceptions" bar.
        checks["all_nodes_referenced_or_declared_unconnected"] = (unreferenced_count == 0)
        if unreferenced_count:
            errors.append(f"{unreferenced_count} node(s) referenced by no net and not declared "
                           f"unconnected")

    rows = tb.read_scl(paths["scl"])
    by_xband = {}
    for r in rows:
        key = (r.x0, r.x0 + r.num_sites * r.sitewidth)
        by_xband.setdefault(key, []).append(r)
    row_errors = []
    for key, rs in by_xband.items():
        rs_sorted = sorted(rs, key=lambda r: r.y)
        for a, b in zip(rs_sorted, rs_sorted[1:]):
            end_a = a.y + a.height
            if end_a > b.y:
                row_errors.append(f"x-band {key}: row overlap between y={a.y}+{a.height}={end_a} "
                                   f"and next row y={b.y} (overlap={end_a - b.y})")
    checks["rows_no_overlap_no_gap"] = (len(row_errors) == 0)
    errors.extend(row_errors)

    return {"status": "ok" if not errors else "fail", "checks": checks, "errors": errors}


def check_v1_dreamplace_readable(config_json=None):
    """Sec 3.3 V1: DREAMPlace's own `PlaceDB.read` succeeds and
    num_movable/num_nets/num_pins equal the manifest. Framework only in
    this task -- needs a DREAMPlace config for the (large) case under
    test, which T4/T5 don't produce; callers with a config can pass it in
    and get a real check, everyone else gets "not_run"."""
    if config_json is None:
        return {"status": "not_run", "reason": "no config_json supplied (T5 scope: framework only)"}
    from ioplace.netlist import load_netlist
    nl, placedb, params = load_netlist(config_json)
    return {"status": "ok", "num_movable": int(placedb.num_movable_nodes),
            "num_nets": int(placedb.num_nets), "num_pins": int(len(placedb.pin2node_map))}


def _not_run_holdout(name, needed):
    return {"status": "not_run", "metric": name,
            "reason": f"needs {needed} (T3a/T6 products that do not exist in this task's scope)"}


def check_h1_rent_shape(p_synthetic=None, p_real=None, threshold=0.05, band=(0.55, 0.80)):
    """H1: |p_syn - p_real| <= threshold and both in `band`. Framework:
    computes the real check when both p's are supplied (e.g. from
    `rent.measure_rent` on a real synthetic array and a real
    `mempool_cluster`); "not_run" otherwise.

    `band=None` (2026-08-15 T6 holdout adjudication sec 4.1/4.4/8-4, T6B's
    B-0..B-6 use): the 27.7M 3x3 array has no same-scale real reference to
    put an absolute band around (sec 4.1's H1/H2 row: `not_applicable`),
    so only the relative leg (`|p_syn - p_real| <= threshold`) is
    evaluated; `band_status="not_applicable"` records that the absolute
    leg was skipped, not passed. The *default* band (0.55, 0.80) is
    untouched -- this is an opt-in per-call parameter, not a threshold
    change (sec 8-4: "不得改動 default band 的數值")."""
    if p_synthetic is None or p_real is None:
        return _not_run_holdout("H1_rent_shape", "p_synthetic and p_real (rent.measure_rent on a real cluster)")
    diff = abs(p_synthetic - p_real)
    rel_ok = diff <= threshold
    if band is None:
        return {"status": "ok" if rel_ok else "fail", "metric": "H1_rent_shape",
                "p_synthetic": p_synthetic, "p_real": p_real, "diff": diff,
                "threshold": threshold, "band": None, "band_status": "not_applicable"}
    ok = rel_ok and band[0] <= p_synthetic <= band[1] and band[0] <= p_real <= band[1]
    return {"status": "ok" if ok else "fail", "metric": "H1_rent_shape",
            "p_synthetic": p_synthetic, "p_real": p_real, "diff": diff,
            "threshold": threshold, "band": band, "band_status": "evaluated"}


def check_h2_cut_histogram(synthetic_levels=None, real_levels=None, rel_tol=0.25):
    """H2: per-level average cut relative error <= rel_tol. `*_levels`:
    {B_level_index: avg_cut}. Framework; "not_run" without both."""
    if synthetic_levels is None or real_levels is None:
        return _not_run_holdout("H2_cut_histogram", "synthetic_levels and real_levels (per-level avg cut)")
    common = sorted(set(synthetic_levels) & set(real_levels))
    per_level = {}
    ok = bool(common)
    for lvl in common:
        real_v = real_levels[lvl]
        rel = abs(synthetic_levels[lvl] - real_v) / real_v if real_v else float("inf")
        per_level[lvl] = rel
        ok = ok and (rel <= rel_tol)
    return {"status": "ok" if ok else "fail", "metric": "H2_cut_histogram",
            "per_level_rel_err": per_level, "rel_tol": rel_tol}


def check_h3_degree_ks(synthetic_degrees=None, real_degrees=None, ks_threshold=0.05,
                        bucket_rel_tol=None):
    """H3: KS statistic between the full net-degree distributions <=
    ks_threshold. Framework; "not_run" without both degree arrays.

    `bucket_rel_tol` (2026-08-14 T6 adjudication doc sec 1 "H3 degree KS
    存活 + 加 tail 揭露"; T6 body table: "用 io_term.DEG_BUCKET_EDGES 同一套
    切法,逐桶相對誤差 <= 20%"): if given, also buckets both degree arrays
    with `io_term.DEG_BUCKET_EDGES`/`DEG_BUCKET_LABELS` (the same edges the
    IO surrogate itself buckets net degree into) and checks each bucket's
    relative frequency error, plus discloses each side's max degree (a
    single overall KS can hide a thin, high-degree tail that a per-bucket
    check makes visible). `status` is "ok" only if the KS check *and* every
    bucket both pass -- the KS-only call (bucket_rel_tol=None) keeps its
    original behavior/signature for existing callers."""
    if synthetic_degrees is None or real_degrees is None:
        return _not_run_holdout("H3_degree_ks", "synthetic_degrees and real_degrees (full net-degree arrays)")
    import numpy as np
    a = np.sort(np.asarray(synthetic_degrees, dtype=np.float64))
    b = np.sort(np.asarray(real_degrees, dtype=np.float64))
    grid = np.union1d(a, b)
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    result = {"status": "ok" if ks <= ks_threshold else "fail", "metric": "H3_degree_ks",
              "ks": ks, "threshold": ks_threshold}
    if bucket_rel_tol is None:
        return result

    from ioplace.ops.io_term import DEG_BUCKET_EDGES, DEG_BUCKET_LABELS
    edges = np.array(DEG_BUCKET_EDGES[1:])
    labels = list(DEG_BUCKET_LABELS) + [f">={DEG_BUCKET_EDGES[-1]}"]

    def _bucket_hist(degs):
        idx = np.clip(np.searchsorted(edges, degs, side="right"), 0, len(labels) - 1)
        counts = np.bincount(idx, minlength=len(labels))
        return counts / counts.sum() if counts.sum() else counts.astype(np.float64)

    frac_a, frac_b = _bucket_hist(a), _bucket_hist(b)
    per_bucket = {}
    buckets_ok = True
    for i, label in enumerate(labels):
        rel = abs(frac_a[i] - frac_b[i]) / frac_b[i] if frac_b[i] else (float("inf") if frac_a[i] else 0.0)
        ok = rel <= bucket_rel_tol
        buckets_ok = buckets_ok and ok
        per_bucket[label] = {"synthetic_frac": float(frac_a[i]), "real_frac": float(frac_b[i]),
                              "rel_err": float(rel), "ok": ok}
    result["per_bucket"] = per_bucket
    result["bucket_rel_tol"] = bucket_rel_tol
    result["max_degree_synthetic"] = float(a[-1]) if len(a) else None
    result["max_degree_real"] = float(b[-1]) if len(b) else None
    result["status"] = "ok" if (ks <= ks_threshold and buckets_ok) else "fail"
    return result


def check_h4_interface_dist_ks(synthetic_dist=None, real_dist=None, ks_threshold=0.10):
    """H4: KS statistic between interface-cell-to-tile-boundary distance
    distributions <= ks_threshold. Framework; "not_run" without both."""
    if synthetic_dist is None or real_dist is None:
        return _not_run_holdout("H4_interface_dist_ks", "synthetic_dist and real_dist (boundary-distance arrays)")
    import numpy as np
    a = np.sort(np.asarray(synthetic_dist, dtype=np.float64))
    b = np.sort(np.asarray(real_dist, dtype=np.float64))
    grid = np.union1d(a, b)
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    return {"status": "ok" if ks <= ks_threshold else "fail", "metric": "H4_interface_dist_ks",
            "ks": ks, "threshold": ks_threshold}


def check_h5_k_grid_lambda_ratio(hard_lambda_sum_synthetic=None, n_nets_synthetic=None,
                                  hard_lambda_sum_real=None, n_nets_real=None, band=(0.7, 1.4)):
    """H5: (hard_lambda_sum/n_nets)_synthetic / (...)_real falls in `band`
    (v2.1 tightened from v1's [0.5, 2.0]). Framework; "not_run" without
    all four inputs (K=16 grid `evaluator_gpu` runs on both a real
    synthetic array and `mempool_cluster` -- T8/T9 products)."""
    if None in (hard_lambda_sum_synthetic, n_nets_synthetic, hard_lambda_sum_real, n_nets_real):
        return _not_run_holdout("H5_k_grid_lambda_ratio",
                                 "hard_lambda_sum/n_nets for both the synthetic array and the real cluster (K=16)")
    lam_syn = hard_lambda_sum_synthetic / n_nets_synthetic
    lam_real = hard_lambda_sum_real / n_nets_real
    ratio = lam_syn / lam_real if lam_real else float("inf")
    ok = band[0] <= ratio <= band[1]
    return {"status": "ok" if ok else "fail", "metric": "H5_k_grid_lambda_ratio",
            "ratio": ratio, "band": band, "lambda_synthetic": lam_syn, "lambda_real": lam_real}


def check_b2_h5_matched_resolution(hard_lambda_sum_synthetic_k16=None, n_nets_synthetic_k16=None,
                                    hard_lambda_sum_real_k4=None, n_nets_real_k4=None,
                                    hard_lambda_sum_real_k16=None, n_nets_real_k16=None,
                                    band=(0.7, 1.4)):
    """B-2 (2026-08-15 T6 holdout adjudication sec 4.1/4.4/8-5): H5 has
    never been evaluated (no `hard_lambda_sum`/`n_nets` field in any
    `verify_group2x2_*.json`) -- this is the equal-*per-tile*-resolution
    caller for it that T6B defines before any measurement is taken (a
    legitimate pre-registration, sec 4.4).

    `hard_lambda_sum/n_nets` is extremely sensitive to grid resolution;
    comparing the 2x2 synthetic array and the single-tile source
    `mempool_group` both at the *same raw* K=16 is resolution-mismatched
    -- the array's die is 4x the source's, so K=16 on it is 1/4 the
    per-tile resolution of K=16 on the source alone, which would make
    `syn < real` by construction (sec 4.4). The matched-resolution pairing
    this function gates on instead:

        synthetic: `ioplace.regions.make_grid_regions(die_2x2, 4, 4)`
                   (K=16 over the whole 2x2 array -- 2x2 regions per tile)
        real:      `ioplace.regions.make_grid_regions(die_group, 2, 2)`
                   (K=4 over the single source tile -- the same 2x2
                   regions per tile)

    Band `[0.7, 1.4]` is spec's original H5 band, unmoved (sec 4.1's B-2
    row: "門檻沿用 spec §3.3 H5 原值").

    This is *framework*, same shape as `check_h5_k_grid_lambda_ratio`:
    each side's `hard_lambda_sum`/`n_nets` are `evaluator_gpu`/
    `evaluator_ref` outputs on real region-graph runs (T8/T9 products),
    supplied by the caller, not computed here -- "not_run" without the
    four matched-resolution inputs.

    `hard_lambda_sum_real_k16`/`n_nets_real_k16` are optional: if given
    (alongside the matched-resolution inputs), the *mismatched* both-K=16
    ratio is also reported, under `diagnostic_mismatched_resolution` --
    diagnostic only, sec 4.1: "同時支援輸出未匹配口徑(兩邊 K=16)作診斷欄位",
    never gates `status`."""
    matched = check_h5_k_grid_lambda_ratio(
        hard_lambda_sum_synthetic=hard_lambda_sum_synthetic_k16, n_nets_synthetic=n_nets_synthetic_k16,
        hard_lambda_sum_real=hard_lambda_sum_real_k4, n_nets_real=n_nets_real_k4, band=band)
    if matched["status"] == "not_run":
        matched["metric"] = "B2_H5_k_grid_lambda_ratio_matched_resolution"
        return matched

    result = dict(matched)
    result["metric"] = "B2_H5_k_grid_lambda_ratio_matched_resolution"
    result["resolution"] = {
        "synthetic": {"k": 16, "grid": "make_grid_regions(die_2x2, 4, 4)", "regions_per_tile": "2x2"},
        "real": {"k": 4, "grid": "make_grid_regions(die_group, 2, 2)", "regions_per_tile": "2x2"},
    }
    if hard_lambda_sum_real_k16 is not None and n_nets_real_k16 is not None:
        mismatched = check_h5_k_grid_lambda_ratio(
            hard_lambda_sum_synthetic=hard_lambda_sum_synthetic_k16, n_nets_synthetic=n_nets_synthetic_k16,
            hard_lambda_sum_real=hard_lambda_sum_real_k16, n_nets_real=n_nets_real_k16, band=band)
        result["diagnostic_mismatched_resolution"] = {
            "ratio": mismatched["ratio"], "real_k": 16,
            "note": "both sides K=16 -- resolution-mismatched (real's per-tile resolution is 4x "
                    "the synthetic array's at the same raw K), diagnostic only, does not gate status",
        }
    return result


def check_h5prime_self_consistency(manifest, tol=1e-3):
    """H5' (originally V5, downgraded sec 3.3): the construction method's
    own arithmetic identity -- the manifest's actual glue net count must
    equal C(m) = sum_{u<v} lambda_0 * phi(dist(u,v), alpha) to within
    `tol` relative error, and the implied cross-tile-net share
    `C(m) / (m*N_tile + C(m))` is reported (not gated -- v1's `(R*C)^p`
    dimensional claim was dropped in v2, sec 3.3).

    This is a self-check of the *generator*, not a validation against
    real cluster data (none exists at this task's scope): it only fails
    if the manifest's glue count disagrees with its own recorded
    (lambda_0, alpha) -- e.g. `sample_glue_net_count(..., mode="expected")`
    output, or aggregated `mode="poisson"` counts over enough seeds that
    sampling noise has mostly averaged out. A single `mode="poisson"`
    seed's raw count is *not* expected to pass this at the 0.1% level by
    construction (Poisson relative fluctuation ~ 1/sqrt(lambda)).

    **Known formula-category bug, disclosed not fixed here (2026-08-15 T6
    holdout adjudication doc Appendix A.2/A.4):** `expected_glue_total` is
    always the *N1* closed form `sum(lambda_0*phi(d))`, regardless of the
    manifest's own `normalization`; called against an N2-normalized
    manifest this compares the wrong quantity (T6's `verify_group2x2_n2.
    json` H5' `fail`, rel_err=1.24e-2, is exactly this -- see that file's
    `h5prime_annotation` field for the verbatim correction text). This
    function is kept byte-for-byte unchanged because T6's already-evaluated
    diagnostics are frozen (not re-run, sec 4.0's "已評過的指標不得重評" --
    this check was never gating for T6 either way, `diagnostics_not_gating`
    only). T6B's B-5 row does **not** call this function any more --
    `check_b5a_recipe_arithmetic`/`check_b5b_manifest_fidelity`/
    `check_b5c_sampling_noise` below replace it there (Appendix A.3's
    three-way split)."""
    from ioplace.bench import glue_gen
    glue = manifest.get("glue") or {}
    if not glue or not glue.get("n_nets") or glue.get("lambda_0") is None or glue.get("alpha") is None:
        return {"status": "not_run", "reason": "manifest has no fitted glue (kernel/lambda_0/alpha unset)"}
    R, C = manifest["R"], manifest["C"]
    expected = glue_gen.expected_glue_total(glue["lambda_0"], glue["alpha"], R, C)
    actual = glue["n_nets"]
    rel_err = abs(actual - expected) / expected if expected > 0 else float("inf")
    m = R * C
    n_tile = manifest["base"]["source_n_nets"]
    share = expected / (m * n_tile + expected) if (m * n_tile + expected) > 0 else 0.0
    return {"status": "ok" if rel_err <= tol else "fail", "expected": expected, "actual": actual,
            "rel_err": rel_err, "tol": tol, "implied_cross_tile_share": share}


# ---------------------------------------------------------------------------
# B-5a/B-5b/B-5c (2026-08-15 T6 holdout adjudication doc Appendix A.3):
# `check_h5prime_self_consistency` above conflated two independent problems
# (Appendix A.2's "比錯了量,兩層"): (1) it always used N1's formula even
# against N2-normalized manifests, and (2) it compared a single Poisson-
# sampled realization to a formula's *expectation* at a 0.1% tolerance no
# amount of correct-formula bookkeeping can generally meet (Poisson relative
# fluctuation ~ 1/sqrt(lambda)). Appendix A.1's three-condition rule allows
# fixing (1) (a pre-registered docstring already excluded (2)'s use case,
# sec 8-4/A.2's "第 (2) 條:依據早於量測") without it counting as an
# after-the-fact threshold relaxation -- but the fix does not re-glue those
# two problems back together: B-5a is the decisive (Poisson-noise-free)
# recipe-arithmetic check, B-5b is exact manifest bookkeeping, and B-5c
# reports the actual-vs-expected sampling noise as a disclosed z-score,
# never gating.
# ---------------------------------------------------------------------------

def check_b5a_recipe_arithmetic(expected_pair_counts, normalization, R, C, lambda_0, alpha,
                                 budget_pairs=3.0, tol=1e-3):
    """B-5a (Appendix A.3 table): decisive, Poisson-noise-free rounding
    self-check of the glue recipe -- `|sum(round(c(u,v))) - ref_total| /
    ref_total <= tol`, where `c(u,v)` is `expected_pair_counts` (manifest's
    own `t6.expected_pair_counts`, "該欄位已是正確值" -- already computed by
    that array's own normalization's construction formula at build time,
    `build_t6_arrays._expected_pair_counts`'s dispatch) and `ref_total` is
    the *independently recomputed* closed-form total for the declared
    `normalization`:

        n2: budget_pairs * lambda_0 * R * C / 2   (Appendix A.3/5.3's "N2
            之下核的選擇完全不改變 glue 總數" identity -- every kernel/alpha
            sums to the same m*B/2, so this needs no per-pair recompute)
        n1: glue_gen.expected_glue_total(lambda_0, alpha, R, C)           (the
            same Sum(lambda_0*phi(d)) `expected_pair_counts` should already
            sum to, if it was built with N1's formula)

    Never touches the array's *actual* materialized/Poisson-sampled net
    count -- that comparison is what made the old `check_h5prime_self_
    consistency` non-decisive (B-5c below reports it separately, as a
    disclosed z-score, not gated at a 1e-3 bar Poisson noise cannot
    generally meet).

    Guard (Appendix A.3's "必加防呆,對 N2 陣列若誤用 N1 公式要能偵測為
    normalization_mismatch"): if `rel_err` against the declared
    normalization's own `ref_total` fails, but `expected_pair_counts`'
    *unrounded* total instead agrees with the *other* normalization's
    `ref_total` to within `tol`, that is diagnostic of exactly the original
    H5' bug (the values were built with the wrong formula), not a genuine
    rounding/construction defect -- reported as `status="normalization_
    mismatch"` instead of a numeric "fail" so the two are not confused."""
    from ioplace.bench import glue_gen
    if normalization not in ("n1", "n2"):
        raise ValueError(f"unknown normalization {normalization!r}; must be 'n1' or 'n2'")

    unrounded_total = sum(expected_pair_counts.values())
    rounded_total = sum(round(v) for v in expected_pair_counts.values())
    n2_ref_total = budget_pairs * lambda_0 * R * C / 2.0
    n1_ref_total = glue_gen.expected_glue_total(lambda_0, alpha, R, C)
    ref_total = n2_ref_total if normalization == "n2" else n1_ref_total
    other_normalization = "n1" if normalization == "n2" else "n2"
    other_ref_total = n1_ref_total if normalization == "n2" else n2_ref_total

    rel_err = abs(rounded_total - ref_total) / ref_total if ref_total else float("inf")
    rel_err_other = (abs(unrounded_total - other_ref_total) / other_ref_total
                      if other_ref_total else float("inf"))

    if rel_err > tol and rel_err_other <= tol:
        return {"status": "normalization_mismatch", "metric": "B5a_recipe_arithmetic",
                "normalization": normalization, "rounded_total": rounded_total,
                "unrounded_total": unrounded_total, "ref_total": ref_total, "rel_err": rel_err,
                "other_normalization": other_normalization, "other_ref_total": other_ref_total,
                "rel_err_vs_other_normalization": rel_err_other, "tol": tol,
                "reason": f"expected_pair_counts matches {other_normalization}'s closed-form "
                          f"total, not the declared {normalization}'s -- the supplied values "
                          f"were very likely computed with the wrong formula (2026-08-15 "
                          f"adjudication Appendix A.2's original H5' bug), not a genuine "
                          f"rounding/construction defect"}
    return {"status": "ok" if rel_err <= tol else "fail", "metric": "B5a_recipe_arithmetic",
            "normalization": normalization, "rounded_total": rounded_total,
            "unrounded_total": unrounded_total, "ref_total": ref_total, "rel_err": rel_err, "tol": tol}


def check_b5b_manifest_fidelity(manifest):
    """B-5b (Appendix A.3 table): exact bookkeeping identity between a
    built array's `glue` section and its own `t6.sampled_pair_counts` --
    `glue.n_nets == sum(sampled_pair_counts.values())` and `glue.n_pins ==
    2 * glue.n_nets` (every glue net is degree-2, sec 3.2/`build_t6_arrays`'
    known simplification). Exact equality, no tolerance -- this is a
    bookkeeping check (did `append_glue_nets` record what was actually
    sampled), not a statistical one."""
    glue = manifest.get("glue") or {}
    t6 = manifest.get("t6") or {}
    sampled_pair_counts = t6.get("sampled_pair_counts") or {}
    sum_sampled = sum(sampled_pair_counts.values())
    n_nets, n_pins = glue.get("n_nets"), glue.get("n_pins")
    nets_match = (n_nets == sum_sampled)
    pins_match = (n_pins == 2 * n_nets) if n_nets is not None else False
    ok = bool(sampled_pair_counts) and nets_match and pins_match
    return {"status": "ok" if ok else "fail", "metric": "B5b_manifest_fidelity",
            "glue_n_nets": n_nets, "sum_sampled_pair_counts": sum_sampled, "nets_match": nets_match,
            "glue_n_pins": n_pins, "expected_n_pins": (2 * n_nets) if n_nets is not None else None,
            "pins_match": pins_match}


def check_b5c_sampling_noise(manifest, expected_pair_counts=None):
    """B-5c (Appendix A.3 table): reports the Poisson sampling-noise
    z-score of the array's *actual* materialized glue-net count against
    the (unrounded) expected total -- `z = (actual - sum(expected_pair_
    counts)) / sqrt(sum(expected_pair_counts))`. Disclosure only: `status`
    never gates on `z` (Appendix A.3: "只入輸出不入 status") -- a single
    `mode="poisson"` seed's count is expected to differ from its own
    expectation by O(1) standard deviations, that is not evidence of a
    construction defect by itself."""
    import math
    glue = manifest.get("glue") or {}
    t6 = manifest.get("t6") or {}
    if expected_pair_counts is None:
        expected_pair_counts = t6.get("expected_pair_counts") or {}
    total_expected = sum(expected_pair_counts.values())
    actual = glue.get("n_nets")
    z = ((actual - total_expected) / math.sqrt(total_expected))
    return {"status": "reported", "metric": "B5c_sampling_noise", "actual": actual,
            "expected_total": total_expected, "z": z,
            "note": "disclosure only, does not gate status (Appendix A.3)"}


def check_b1_rent_invariance_did(p_3x3_glue, p_3x3_noglue, p_2x2_glue, p_2x2_noglue):
    """B-1 (2026-08-15 T6 holdout adjudication sec 4.1/4.3/8-5):
    difference-in-differences Rent shape-invariance check for T6B's 3x3
    artifact. A 3x3 array's 9 tiles can't be halved into an integer tile
    count, so recursive bisection systematically inflates low-level
    terminal counts near the fit window regardless of glue -- a
    combinatorial artifact of the *shape*, not evidence about glue. Each
    array's own zero-glue control isolates it: only the *increment* from
    adding glue is compared across shapes.

        Delta_glue(s) = p(s, N2 glue) - p(s, no glue)
        DiD = |Delta_glue(3x3) - Delta_glue(2x2)|

    Threshold is 0.03 -- H6's original threshold (sec 5.2/8-5's decided
    rule: T6B reuses pre-registered numbers, it does not mint new ones),
    hardcoded here, not a parameter (sec 8-5: "門檻常數 0.03 寫死、無參數
    介面")."""
    delta_3x3 = p_3x3_glue - p_3x3_noglue
    delta_2x2 = p_2x2_glue - p_2x2_noglue
    did = abs(delta_3x3 - delta_2x2)
    threshold = 0.03
    return {"status": "ok" if did <= threshold else "fail", "metric": "B1_rent_invariance_did",
            "delta_glue_3x3": delta_3x3, "delta_glue_2x2": delta_2x2, "did": did,
            "threshold": threshold}
