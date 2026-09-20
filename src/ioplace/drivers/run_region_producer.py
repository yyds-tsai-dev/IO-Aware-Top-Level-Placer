"""Driver 1 of the v2 flow: the region producer (spec section 1 / section 2).

In:  a DREAMPlace config, K, and a membership source.
Out: regions.json, seed.npz, membership.npz, producer.json in one directory.

Pipeline (digest section 5's stages 1-2): membership prior -> flat GP with the
GrandPlan grouping loss (hull tables rebuilt every T_hull from the iteration
callback) -> flat LG -> density-argmax extraction -> SA -> rect_max -> RegionSet.

Coordinate contract: everything inside the GP is in the SCALED post-initialize()
system; every geometric quantity written out is in the NATIVE post-read() system
(x_native = x_scaled / scale_factor + shift_factor[0]). The two exceptions are
producer.json's `hpwl_gp`/`hpwl_lg`, which stay SCALED -- see where they are
recorded below.

Normalisation: Eq.3's coefficient for the grouping term is derived by
ioplace.norm.TermNormalizer (policy "grandplan", norm_p=1) -- spec section 0's
single-owner rule, ruling D5. This driver holds no schedule state of its own.

dp_hook.assert_optimizer_lock passes on mempool_tile_wrap (use_bb -> 0,
macro_place_flag -> 0, no movable macros) and would FAIL on any design that has
movable macros, because DREAMPlace then selects the bb optimizer and the
surrogate-only forward is not behaviourally equivalent under it.
"""
import argparse
import os
import socket
import sys
import time

import numpy as np

from ioplace.artifacts import (placedb_identity_sha256, save_membership,
                               save_positions, save_producer_json)
from ioplace.dp_hook import (assert_optimizer_lock, attach_terms,
                             install_version_invariant,
                             refresh_nesterov_secant)
from ioplace.drivers.run_placement import (_load_dreamplace,
                                           extract_final_positions)
from ioplace.netlist import netlist_from_placedb
from ioplace.norm import TermNormalizer
from ioplace.producer import extract as extract_mod
from ioplace.producer import hull as hull_mod
from ioplace.producer import rectify as rectify_mod
from ioplace.producer import sa as sa_mod
from ioplace.producer.grouping_term import GroupingTerm
from ioplace.producer.membership import build_membership
from ioplace.paths import REPO_ROOT
from ioplace.profile import PhaseTimer, env_metadata

LATTICE = 512

# Ruling D5/D6 follow-up (controller, 2026-09-20): the grouping term registers
# with n_ramp = 0, NOT `register`'s default 20. GrandPlan applies the grouping
# loss from iteration 0 and Eq.3's own `wt` schedule (0.05, +0.05 every 100
# iterations) already supplies the gentle onset; a second, 20-iteration
# activation ramp on top of it would make lambda exactly 0 on the activating
# transaction and, at the default t_hull = probe_every = 50, leave the term
# inert for iterations 0-49 of a run whose hull is rebuilt only 20 times.
# With n_ramp = 0 `activation_ramp` is the step 1{iteration >= it_activate}
# (schedules.py:24-27), so `applied_lambda` == `self.lambdas[...]` for every
# iteration from the activating transaction onward -- which is why the term
# closure below may read `normalizer.lambdas` directly and still reproduce the
# per-iteration coefficient exactly.
GROUP_N_RAMP = 0


class _GroupAdapter(object):
    """TermNormalizer's term protocol is one method, `value(pos, ctx) -> Tensor`
    returning the term's UNWEIGHTED objective; the normalizer owns the backward,
    the fixed/filler masking and the norm order (ioplace/ops/norm_terms.py).
    GroupingTerm's unweighted energy is `forward(pos, 1.0)`; `ctx` carries the
    driver's {iteration, overflow, tau, gamma} and this term needs none of it.
    """

    def __init__(self, term):
        self.term = term

    def value(self, pos, ctx):
        return self.term.forward(pos, 1.0)


def check_membership(part, k, source):
    """Refuse an unusable membership prior BEFORE the GP starts: a label
    outside `[0, k)` (fix round 2, minor 4), or a region with no movable cells
    (fix round 1, finding I1).

    An out-of-range label is not cosmetic. `GroupingTerm.set_tables` asserts
    `tables.k > part.max()`, but that fires inside the FIRST iteration
    callback, minutes into a run; below it, `np.bincount(...)[:k]` would have
    silently dropped the overflowing label, so the region-count report and
    `ea_scaled` would already have been computed as if those cells did not
    exist. A negative label is worse still -- `np.bincount` raises its own
    opaque "argument must have no negative elements" from inside the count.

    There is no safe placeholder hull for an empty region.
    `hull.anchor_tables` reads "bin centre is inside hull s" as "region s
    pushes every cell of every other region away from bin b"
    (`hull.py:294-300`), and `nearest_on_polygon_boundary`'s half-plane test is
    `<= 0` on every edge (`hull.py:222,238`), so BOTH candidate placeholders --
    the die box and a zero-area polygon -- test as containing every bin. Either
    one would give every other region a spurious die-wide push and silently
    corrupt Eq.1 rather than fail.

    Reachable in practice, which is why this is a check and not an assert:
    Mt-KaHyPar balances over the whole hypergraph and `membership.py:18-27`
    then truncates its labels to the movable prefix, so a part can be empty on
    the movables while being non-empty overall.

    A caller who hits this should lower `--k` (the usual cause is asking for
    more regions than the prior can populate) or switch `--membership`; there
    is no in-driver repair, because inventing geometry for a region with no
    cells is exactly the corruption above.
    """
    labels = np.asarray(part, dtype=np.int64).ravel()
    if labels.size:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= k:
            bad = np.flatnonzero((labels < 0) | (labels >= k))
            raise ValueError(
                "membership source %r produced %d label(s) outside [0, %d): "
                "range [%d, %d], first offenders at movable indices %r. The "
                "grouping term indexes the anchor tables by label, so a label "
                "outside the region range has no hull to be pulled to. Raise "
                "--k to at least %d, or fix the prior."
                % (source, bad.size, k, lo, hi, bad[:5].tolist(), hi + 1))
    counts = np.bincount(labels, minlength=k)
    empty = np.flatnonzero(counts[:k] == 0).tolist()
    if empty:
        raise ValueError(
            "membership source %r left region(s) %r with no movable cells "
            "(counts=%r): the grouping term cannot build a hull for an empty "
            "region, and every placeholder hull contains every bin and would "
            "push all other regions away. Lower --k or use a different "
            "--membership prior." % (source, empty, counts[:k].tolist()))


def rectify_with_fallback(build, rectify_fn, requested_bins, fallback_bins=32):
    """Ruling D3: `--extract-bins 32` is a driver fallback, not an operator
    instruction.

    `build(bins) -> (labels, sa_report)` runs extraction + SA at `bins`;
    `rectify_fn(labels) -> labels` enforces the rect budget. A RuntimeError
    from either half at the requested bin count -- in practice
    `enforce_rect_max` exhausting both its moves (spec section 10 risk 1) --
    retries the WHOLE thing at `fallback_bins` before it is allowed to escape.
    At or below `fallback_bins` there is nothing left to fall back to, so the
    error propagates.

    Returns `(bins_used, labels, sa_report, path, reason)` with `path` one of
    "direct" / "fallback_32" and `reason` the first failure's message (None on
    the direct path).
    """
    try:
        labels, report = build(requested_bins)
        return requested_bins, rectify_fn(labels), report, "direct", None
    except RuntimeError as exc:
        if requested_bins <= fallback_bins:
            raise
        reason = str(exc)
    labels, report = build(fallback_bins)
    return (fallback_bins, rectify_fn(labels), report,
            "fallback_%d" % fallback_bins, reason)


def run_producer(config_json, out_dir, *, k=16, membership_source="mtkahypar",
                 extract_bins=64, rect_max=8, seed=0, epsilon=0.03, t_hull=50,
                 probe_every=50, hierarchy_depth=1, sa_seed=0, alpha_pull=1.0,
                 alpha_push=1.0, skip_sa=False, dp_seed=None,
                 deterministic=None, gp_iterations=None):
    import torch

    t_start = time.time()
    timer = PhaseTimer()
    os.makedirs(out_dir, exist_ok=True)

    # ---- read: native post-read coordinate system ------------------------
    with timer.phase("read"):
        params, placedb = _load_dreamplace(config_json)
        if dp_seed is not None:
            params.random_seed = dp_seed
        if deterministic is not None:
            params.deterministic_flag = deterministic
        if gp_iterations is not None:
            # simple.json does not set global_place_stages, so DREAMPlace's
            # params.json default (iteration: 1000) applies; the slow tests
            # pass 200 so five end-to-end runs stay in review-cycle time.
            params.global_place_stages[0]["iteration"] = int(gp_iterations)
        die_native = (float(placedb.xl), float(placedb.yl),
                      float(placedb.xh), float(placedb.yh))
        nl0 = netlist_from_placedb(placedb)
        # Before initialize(): the digest covers node sizes, which PlaceDB.scale()
        # multiplies by scale_factor (PlaceDB.py:160-161). Hashing here is what
        # makes the main flow's expect_sha256 check agree across processes.
        fingerprint = placedb_identity_sha256(placedb)
        node_names = getattr(placedb, "node_names", None)

    import NonLinearPlace   # importable only after _load_dreamplace's setup

    with timer.phase("prior"):
        part = build_membership(membership_source, nl=nl0, node_names=node_names,
                                num_movable=int(placedb.num_movable_nodes), k=k,
                                epsilon=epsilon, seed=seed, depth=hierarchy_depth)
        check_membership(part, k, membership_source)

    # ---- initialize: scaled coordinate system ----------------------------
    with timer.phase("initialize"):
        placedb.initialize(params)
        # Passes on the acceptance case: mempool_tile_wrap resolves use_bb -> 0
        # and macro_place_flag -> 0 because it has NO movable macros. It would
        # FAIL on any design that does -- DREAMPlace then selects the bb
        # optimizer, which the surrogate-only forward is not equivalent under.
        assert_optimizer_lock(params)
        die_scaled = (float(placedb.xl), float(placedb.yl),
                      float(placedb.xh), float(placedb.yh))
        # Fix round 1, finding I3. `anchor_tables` is first called from inside
        # the iteration callback, so a violated precondition used to abort
        # mid-GP after minutes of setup. Check the exact same predicate here,
        # before NonLinearPlace is even constructed.
        hull_mod.check_anchor_table_range(die_scaled, LATTICE)
        scale = float(params.scale_factor)
        shift = (float(params.shift_factor[0]), float(params.shift_factor[1]))
        m = int(placedb.num_movable_nodes)
        n_all = int(placedb.num_nodes)
        n_phys = int(placedb.num_physical_nodes)
        size_x = np.asarray(placedb.node_size_x[:m], dtype=np.float64)
        size_y = np.asarray(placedb.node_size_y[:m], dtype=np.float64)
        cell_area = size_x * size_y
        target_density = float(params.target_density)
        ea_scaled = np.array(
            [cell_area[part == kk].sum() / target_density for kk in range(k)])
        row_h = float(placedb.row_height)
        is_macro = size_y > 2.0 * row_h
        is_std = ~is_macro
        pitch_x = float(size_x[is_std].mean()) if is_std.any() else float(size_x.mean())
        pitch_y = float(size_y[is_std].mean()) if is_std.any() else float(size_y.mean())
        # MOVABLE macros only: `size_x`/`size_y` are the [:num_movable] prefix,
        # so fixed macros (all of mempool_tile_wrap's fakeram45 instances are
        # fixed terminals) are outside this slice and never enrich a hull --
        # and their occupied area is likewise absent from EA_k. On the
        # acceptance case macro_idx is empty and this whole branch is dead.
        macro_idx = np.nonzero(is_macro)[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    term = GroupingTerm(part=part, node_size_x=size_x, node_size_y=size_y,
                        num_movable=m, num_nodes=n_all, alpha_pull=alpha_pull,
                        alpha_push=alpha_push, device=device)
    # Ruling D5: one owner for every extra term's coefficient. policy
    # "grandplan" is Eq.3 (lam = wt * EMA(||grad WL||_1 / ||grad Group||_1)) with
    # digest section 4.1's stepped ramp; activate_overflow=1.0 latches the term
    # on the first transaction, reproducing the grouping loss's it_activate=0
    # "on for the whole flat GP" semantics (digest section 5). curvature=1.0
    # because the term is an exact quadratic spring. n_ramp=GROUP_N_RAMP (0):
    # see that constant's comment -- Eq.3's own `wt` ramp is the soft start.
    normalizer = TermNormalizer(policy="grandplan", norm_p=1,
                                probe_every=probe_every, num_movable=m,
                                num_nodes=n_all)
    normalizer.register("group", _GroupAdapter(term), curvature=1.0,
                        activate_overflow=1.0, n_ramp=GROUP_N_RAMP)
    attach_terms(params, [lambda pos: term(pos, normalizer.lambdas.get("group", 0.0))])

    part_t = torch.as_tensor(part.astype(np.int64), device=device)
    half_x = torch.as_tensor(0.5 * size_x, dtype=torch.float64, device=device)
    half_y = torch.as_tensor(0.5 * size_y, dtype=torch.float64, device=device)
    def rebuild(pos):
        """Recompute every partition's hull from the current cell centres and
        re-rasterise the anchor tables. Frozen until the next rebuild."""
        with torch.no_grad():
            cx = pos[:m].double() + half_x
            cy = pos[n_all:n_all + m].double() + half_y
        hulls = []
        for kk in range(k):
            sel = (part_t == kk).nonzero(as_tuple=True)[0]
            # Unreachable: `part` is fixed for the whole run and
            # `check_membership` rejected an empty region before the GP
            # started. Kept as an assert rather than a placeholder hull
            # because there is no correct placeholder -- see that function
            # (fix round 1, finding I1).
            assert sel.numel() > 0, (
                "region %d became empty after check_membership" % kk)
            if cx.is_cuda:
                pts = hull_mod.reduce_candidates_torch(cx[sel], cy[sel])
            else:
                pts = hull_mod.reduce_candidates(
                    torch.stack([cx[sel], cy[sel]], dim=1).numpy())
            mk = macro_idx[part[macro_idx] == kk]
            if len(mk):
                mx = cx[mk].cpu().numpy() - 0.5 * size_x[mk]
                my = cy[mk].cpu().numpy() - 0.5 * size_y[mk]
                pts = np.vstack([pts, hull_mod.macro_pseudo_points(
                    mx, my, size_x[mk], size_y[mk], pitch_x, pitch_y)])
            hulls.append(hull_mod.build_hull(pts, a_max=float(ea_scaled[kk])))
        term.set_tables(hull_mod.anchor_tables(hulls, die_scaled, LATTICE,
                                               device=device))

    probes = []
    cb_state = {"last_iteration": -1, "invariant": None}

    def cb(iteration, pos):
        cb_state["last_iteration"] = iteration
        # A hull rebuild and a lambda change are both DISCRETE objective
        # changes. `normalizer.transaction` is what bumps obj_version, so a
        # rebuild iteration must run one too even when it is not a probe
        # iteration -- otherwise the install_version_invariant below would have
        # nothing to catch and the secant cache would go stale silently.
        discrete = False
        if iteration % t_hull == 0:
            rebuild(pos)
            discrete = True
        ctx = {"iteration": iteration, "overflow": 0.0, "tau": 0.0,
               "gamma": 0.0}
        if normalizer.should_probe(iteration) and term.tables is not None:
            normalizer.probe(iteration, pos,
                             placer.model.op_collections.wirelength_op, ctx)
            discrete = True
        if discrete:
            # tau/gamma = 0 disables the Lipschitz cap (lipschitz_cap returns
            # inf for gamma <= 0): that cap bounds lambda against the IO term's
            # smoothing tau, and the grouping term is an exact quadratic with
            # no smoothing parameter -- GrandPlan Eq.3 has no cap either.
            # overflow = 0 with activate_overflow = 1.0 keeps the term latched
            # on from the first callback; the producer has no overflow gate.
            txn = normalizer.transaction(iteration, 0.0, 0.0, 0.0)
            st = normalizer.states["group"]
            probes.append({"iteration": int(iteration),
                           "grad_l1_wl": normalizer.wl_norm,
                           "grad_l1_group": st.grad_norm,
                           "ratio_ema": st.ratio_ema, "wt": st.wt,
                           "lambda_group": txn.lambdas["group"],
                           "obj_version": txn.obj_version})
        # Order is fixed: the new tables / lambda take effect first, then the
        # secant cache is rebuilt under the new objective
        # (run_placement_io.py:690-694, design v2 sec 6.4).
        if normalizer.needs_refresh():
            refresh_nesterov_secant(placer.optimizer)
            normalizer.mark_refreshed()
        if cb_state["invariant"] is not placer.optimizer:
            # Ruling D6. NonLinearPlace sets self.optimizer inside __call__
            # (NonLinearPlace.py:235), not in the constructor, so this is the
            # earliest point it exists -- the same lazy install
            # run_placement_io.py:704-708 does. Installing AFTER the refresh
            # above means it never sees a pending bump of its own making.
            # Keyed on the optimizer object rather than a one-shot bool so a
            # multi-stage `global_place_stages` (one optimizer per stage,
            # NonLinearPlace.py:235) gets the invariant on every stage instead
            # of silently losing it after the first; identity keying also makes
            # double-wrapping the same optimizer impossible.
            install_version_invariant(placer.optimizer, normalizer)
            cb_state["invariant"] = placer.optimizer

    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

    orig_legalize = placer.op_collections.legalize_op
    hpwl = {}
    gp_phase = timer.phase("gp")
    gp_phase.__enter__()

    def _timed_legalize(p):
        gp_phase.__exit__(None, None, None)
        with torch.no_grad():
            # SCALED units (hpwl_op runs on the post-initialize() placedb);
            # recorded that way on purpose -- see the payload below.
            hpwl["hpwl_gp"] = float(placer.op_collections.hpwl_op(p))
        with timer.phase("lg"):
            out = orig_legalize(p)
        with torch.no_grad():
            hpwl["hpwl_lg"] = float(placer.op_collections.hpwl_op(out))
        return out

    placer.op_collections.legalize_op = _timed_legalize
    placer.iteration_callback = cb
    placer(params, placedb, params.global_place_stages[0]["learning_rate"])
    if "gp" not in timer.phases:
        gp_phase.__exit__(None, None, None)
    final_overflow = float(placer.model.overflow.max())

    # ---- back to native units -------------------------------------------
    x_s, y_s = extract_final_positions(placer, placedb)
    x_n = x_s / scale + shift[0]
    y_n = y_s / scale + shift[1]
    w_n = np.asarray(placedb.node_size_x[:n_phys], dtype=np.float64) / scale
    h_n = np.asarray(placedb.node_size_y[:n_phys], dtype=np.float64) / scale
    ea_native = ea_scaled / (scale * scale)

    def _bin_area(bins):
        return (((die_native[2] - die_native[0]) / bins) *
                ((die_native[3] - die_native[1]) / bins))

    def build(bins):
        """Extraction + SA at `bins`. Re-entering a PhaseTimer phase overwrites
        it, so runtime_s["extract"]/["sa"] always describe the run that
        produced the emitted geometry, not a discarded first attempt."""
        with timer.phase("extract"):
            labels0 = extract_mod.extract(x_n[:m], y_n[:m], w_n[:m], h_n[:m],
                                          part, k, die_native, out_bins=bins)
        with timer.phase("sa"):
            if skip_sa:
                return labels0.copy(), {"skipped": True}
            return sa_mod.anneal(labels0, k, ea_native, _bin_area(bins),
                                 sa_mod.SaConfig(seed=sa_seed))

    def _rectify(labels):
        with timer.phase("rectify"):
            return rectify_mod.enforce_rect_max(labels, k, rect_max=rect_max)

    requested_bins = int(extract_bins)
    bins_used, labels, sa_report, rect_max_path, fallback_reason = \
        rectify_with_fallback(build, _rectify, requested_bins)
    # Ruling D3: which path was taken is recorded inside the existing `sa`
    # dict, so artifacts.PRODUCER_FIELDS is unchanged.
    sa_report["extract_bins_requested"] = requested_bins
    sa_report["rect_max_path"] = rect_max_path
    if fallback_reason is not None:
        sa_report["rect_max_fallback_reason"] = fallback_reason
    bin_area = _bin_area(bins_used)
    rs = rectify_mod.rects_to_regionset(labels, k, die_native, lattice=LATTICE)

    # ---- artefacts -------------------------------------------------------
    rs.to_json(os.path.join(out_dir, "regions.json"))
    save_positions(os.path.join(out_dir, "seed.npz"), x_n, y_n, die=die_native,
                   shift_factor=shift, scale_factor=scale,
                   placedb_sha256=fingerprint, kind="seed")
    save_membership(os.path.join(out_dir, "membership.npz"), part,
                    source=membership_source, k=k, seed=seed, epsilon=epsilon)

    counts = rectify_mod.region_rect_counts(labels, k)
    bins_per_region = np.bincount(labels.ravel(), minlength=k)[:k]
    region_area = bins_per_region.astype(np.float64) * bin_area
    cell_area_native = cell_area / (scale * scale)
    region_cell_area = np.array(
        [cell_area_native[part == kk].sum() for kk in range(k)])
    util = np.divide(region_cell_area, region_area,
                     out=np.zeros(k), where=region_area > 0)
    runtime = {name: float(p["t_s"]) for name, p in timer.phases.items()}
    runtime["total"] = time.time() - t_start
    group_state = normalizer.states["group"]
    # DEVIATION from the Task 10 brief (see the task report): the brief read
    # `torch.cuda.max_memory_allocated()` here. PhaseTimer calls
    # `reset_peak_memory_stats()` on every phase ENTRY (profile.py:241-245), so
    # that counter reports the peak since the last phase began -- which is
    # `rectify`, a pure-numpy phase -- and silently discards the GP peak this
    # field exists to report (measured 5.0 MB on simple.json, against a
    # K*512*512 fp16 anchor table allocated inside `gp`). Take the max over the
    # per-phase peaks PhaseTimer already recorded instead; its own docstring
    # carries the caveat that a reset clears counters, not still-live tensors.
    phase_peaks = [p["peak_alloc_gb"] for p in timer.phases.values()
                   if p.get("peak_alloc_gb") is not None]
    peak_mem_mb = (max(phase_peaks) * 1024.0) if phase_peaks else 0.0

    payload = {
        "config": os.path.abspath(config_json), "out_dir": os.path.abspath(out_dir),
        "k": int(k), "membership_source": membership_source,
        "membership_seed": int(seed), "epsilon": float(epsilon),
        "hierarchy_depth": int(hierarchy_depth),
        "extract_bins": int(bins_used), "fine_bins": int(extract_mod.FINE_BINS),
        "lattice": LATTICE, "rect_max": int(rect_max),
        "t_hull": int(t_hull), "probe_every": int(probe_every),
        "n_hull_rebuilds": int(term.n_rebuilds),
        "alpha_pull": float(alpha_pull), "alpha_push": float(alpha_push),
        "wt_final": float(group_state.wt),
        "lambda_group_final": float(normalizer.lambdas.get("group", 0.0)),
        "ratio_ema_final": (None if group_state.ratio_ema is None
                            else float(group_state.ratio_ema)),
        "die_native": list(die_native), "die_scaled": list(die_scaled),
        "shift_factor": list(shift), "scale_factor": scale,
        "placedb_sha256": fingerprint,
        "num_movable": m, "num_physical": n_phys, "num_nodes": n_all,
        "num_nets": int(nl0.num_nets), "target_density": target_density,
        "gp_iterations_run": int(cb_state["last_iteration"]) + 1,
        "final_overflow": final_overflow,
        # SCALED, not native, unlike every other geometric field here:
        # `hpwl_op` runs on the post-initialize() placedb, so these are
        # native_hpwl / site_width. Left scaled deliberately, for comparability
        # with drivers/run_placement.py:386-390, which records them the same
        # way; `scale_factor` above makes the conversion recoverable.
        "hpwl_gp": hpwl.get("hpwl_gp"), "hpwl_lg": hpwl.get("hpwl_lg"),
        "sa": sa_report, "sa_seed": int(sa_seed),
        "rects_per_region": [int(c) for c in counts],
        "rect_max_observed": int(max(counts)),
        "region_bins": [int(b) for b in bins_per_region],
        "region_area": region_area.tolist(),
        "region_cell_area": region_cell_area.tolist(),
        "region_utilisation": util.tolist(),
        "area_balance": {"max_util": float(util.max()),
                         "min_util": float(util.min()),
                         "max_over_min": (float(util.max() / util.min())
                                          if util.min() > 0 else None)},
        "probes": probes,
        "runtime_s": runtime,
        "peak_mem_mb": peak_mem_mb,
        "command": " ".join(sys.argv), "hostname": socket.gethostname(),
        "env": env_metadata(str(REPO_ROOT), os.environ.get("DREAMPLACE_ROOT", ""),
                            input_paths=(config_json,)),
    }
    save_producer_json(os.path.join(out_dir, "producer.json"), payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="v2 region producer (P-C)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--membership", choices=["mtkahypar", "hierarchy"],
                    default="mtkahypar")
    ap.add_argument("--extract-bins", type=int, choices=[32, 64], default=64,
                    help="64 = spec default; 32 = faithful GrandPlan for arm "
                         "(e), and the automatic fallback if the rect budget "
                         "is unreachable at 64")
    ap.add_argument("--rect-max", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--t-hull", type=int, default=50)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--hierarchy-depth", type=int, default=1)
    ap.add_argument("--sa-seed", type=int, default=0)
    ap.add_argument("--alpha-pull", type=float, default=1.0)
    ap.add_argument("--alpha-push", type=float, default=1.0)
    ap.add_argument("--no-sa", action="store_true")
    ap.add_argument("--dp-seed", type=int, default=None)
    ap.add_argument("--deterministic", type=int, default=None)
    ap.add_argument("--gp-iterations", type=int, default=None,
                    help="override global_place_stages[0]['iteration']")
    a = ap.parse_args()
    run_producer(a.config, a.out, k=a.k, membership_source=a.membership,
                 extract_bins=a.extract_bins, rect_max=a.rect_max, seed=a.seed,
                 epsilon=a.epsilon, t_hull=a.t_hull, probe_every=a.probe_every,
                 hierarchy_depth=a.hierarchy_depth, sa_seed=a.sa_seed,
                 alpha_pull=a.alpha_pull, alpha_push=a.alpha_push,
                 skip_sa=a.no_sa, dp_seed=a.dp_seed,
                 deterministic=a.deterministic, gp_iterations=a.gp_iterations)


if __name__ == "__main__":
    main()
