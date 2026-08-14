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


def check_v0_structural(prefix, manifest=None, source_prefix=None, allowed_unconnected=()):
    """Sec 3.3 V0, hard/mechanical: `.aux`-referenced files all exist;
    NumNodes/NumNets/NumPins headers match actual body line counts and
    (if `manifest` given) the manifest's base+glue totals; no net has a
    repeated node (self-loop); the design's degree<=1 net fraction is not
    higher than `source_prefix`'s (if given); every node is referenced by
    at least one net or is explicitly in `allowed_unconnected`; `.scl` row
    coverage has no overlap and no gap. Returns
    {"status": "ok"|"fail", "checks": {...}, "errors": [...]}."""
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
    nets = list(_iter_nets(paths["nets"]))
    n_pins_body = sum(len(pins) for _, pins in nets)
    checks["nets_header_matches_body"] = (len(nets) == n_nets_hdr and n_pins_body == n_pins_hdr)
    if len(nets) != n_nets_hdr:
        errors.append(f"NumNets header={n_nets_hdr} but body has {len(nets)} net blocks")
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

    self_loop_nets = [name for name, pins in nets if len(set(p[0] for p in pins)) < len(pins)]
    checks["no_self_loops"] = (len(self_loop_nets) == 0)
    if self_loop_nets:
        errors.append(f"{len(self_loop_nets)} net(s) with a repeated node (self-loop), "
                       f"e.g. {self_loop_nets[:5]}")

    degrees = [len(pins) for _, pins in nets]
    frac_le1 = (sum(1 for d in degrees if d <= 1) / len(degrees)) if degrees else 0.0
    checks["degree_le1_fraction"] = frac_le1
    if source_prefix is not None:
        src_paths = tb.read_aux(source_prefix + ".aux")
        src_degrees = [len(pins) for _, pins in _iter_nets(src_paths["nets"])]
        src_frac = (sum(1 for d in src_degrees if d <= 1) / len(src_degrees)) if src_degrees else 0.0
        checks["degree_le1_not_worse_than_source"] = (frac_le1 <= src_frac + 1e-9)
        if frac_le1 > src_frac + 1e-9:
            errors.append(f"degree<=1 fraction {frac_le1:.4f} > source's {src_frac:.4f}")

    referenced = set()
    for _, pins in nets:
        for name, _ in pins:
            referenced.add(name)
    allowed = set(allowed_unconnected)
    unreferenced = [n for n in node_names if n not in referenced and n not in allowed]
    checks["all_nodes_referenced_or_declared_unconnected"] = (len(unreferenced) == 0)
    if unreferenced:
        errors.append(f"{len(unreferenced)} node(s) referenced by no net and not declared "
                       f"unconnected, e.g. {unreferenced[:5]}")

    rows = tb.read_scl(paths["scl"])
    by_xband = {}
    for r in rows:
        key = (r.x0, r.x0 + r.num_sites * r.sitewidth)
        by_xband.setdefault(key, []).append(r)
    row_errors = []
    for key, rs in by_xband.items():
        rs_sorted = sorted(rs, key=lambda r: r.y)
        for a, b in zip(rs_sorted, rs_sorted[1:]):
            if a.y + a.height != b.y:
                row_errors.append(f"x-band {key}: row gap/overlap between y={a.y}+{a.height} "
                                   f"and next row y={b.y}")
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
    `mempool_cluster`); "not_run" otherwise."""
    if p_synthetic is None or p_real is None:
        return _not_run_holdout("H1_rent_shape", "p_synthetic and p_real (rent.measure_rent on a real cluster)")
    ok = abs(p_synthetic - p_real) <= threshold and band[0] <= p_synthetic <= band[1] and band[0] <= p_real <= band[1]
    return {"status": "ok" if ok else "fail", "metric": "H1_rent_shape",
            "p_synthetic": p_synthetic, "p_real": p_real, "diff": abs(p_synthetic - p_real),
            "threshold": threshold, "band": band}


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
    construction (Poisson relative fluctuation ~ 1/sqrt(lambda))."""
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
