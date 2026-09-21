"""v2 main flow: soft-assign GP -> freeze -> fence GP -> fence LG -> evaluator.

Design v2 sec 3. Forked from run_placement_io.py, which stays as the
single-phase legacy driver and is imported here only for its resource-cleanup
helpers. The four phases live in two DREAMPlace instances because fence data
may only be injected between read() and initialize() (fence_inject.py:9-16);
they communicate through validated artefacts, so `--phase {all,soft,fence}`
can re-run either half from disk.
"""
import argparse
import json
import os
import time

import numpy as np

from ioplace.artifacts import (MAIN_FLOW_RESULT_SCHEMA_VERSION, file_sha256,
                               load_freeze, load_membership, load_positions,
                               placedb_identity_sha256, save_freeze,
                               save_membership, save_positions, save_result,
                               scaled_region_set)
from ioplace.dp_hook import (assert_optimizer_lock, attach_terms, detach_terms,
                             install_version_invariant, refresh_nesterov_secant)
from ioplace.drivers.run_placement import (_effective_scale_fields,
                                           _fence_overflow_stop_metric,
                                           _gp_iteration_budget,
                                           _legalization_diagnostics,
                                           _load_dreamplace,
                                           _pack_eval_metrics,
                                           _pack_straddle_metrics,
                                           _stop_overflow_reached,
                                           _t8a_provenance,
                                           extract_final_positions,
                                           get_regions_for)
from ioplace.drivers.run_placement_io import (_cleanup_once, _install_attribute,
                                              _io_cleanup)
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.export.evaluation import save_evaluation
from ioplace.fence_phase import build_fence_placedb, install_density_weight_clamp
from ioplace.freeze import (FreezeMonitor, argmax_region, cell_centers,
                            ensure_nonempty_regions, freeze_record,
                            region_cell_stats)
from ioplace.init_pos import INIT_MODES, apply_init, region_centers
from ioplace.main_flow_metrics import (fence_compliance, io_accounting,
                                       phase_summary, region_area_balance)
from ioplace.netlist import netlist_from_placedb
from ioplace.norm_adapter import NORM_POLICIES, make_norm_adapter
from ioplace.ops.io_term import IoTerm, build_net_node_csr
from ioplace.ops.soft_assign import rect_table
from ioplace.profile import DeviceMemSampler, PhaseTimer, release_cuda_scratch
from ioplace.region_grid import RegionGrid
from ioplace.regions import RegionSet

SOFT_NPZ = "soft.npz"
FREEZE_JSON = "freeze.json"
FROZEN_MEMBERSHIP_NPZ = "frozen_membership.npz"
PLACEMENT_NPZ = "placement.npz"
EVALUATION_NPZ = "evaluation.npz"
NORM_TRACE = "norm_trace.jsonl"
LEGACY_TRACE = "legacy_trace.jsonl"
REGIONS_JSON = "regions.json"
RESULT_JSON = "result.json"
SOFT_RESULT_JSON = "soft_result.json"

#: `update_density_weight_op_overflow` clamps the freshly recomputed weight to
#: 10 itself (`$DREAMPLACE_ROOT/install/dreamplace/PlaceObj.py:875`,
#: `(density_weight_u * density_weight_s).clamp(max=10)`). Our own ceiling is
#: `density_clamp_hi * density_weight_soft`, which can sit above that, in which
#: case DREAMPlace's cap binds first and ours never does -- the clamp log says
#: so rather than leaving a reader to compare the two numbers by hand (Task 4
#: review).
DREAMPLACE_DENSITY_WEIGHT_MAX = 10.0


class _FreezeReached(Exception):
    """Raised from the phase-1 iteration callback to end the soft GP.

    NonLinearPlace ignores the callback's return value
    (NonLinearPlace.py:521-523), so an exception is the only way to stop the
    loop from the driver without touching DREAMPlace source. Phase 1 runs with
    legalize_flag=0, so nothing *this driver reads* is skipped by unwinding:
    the tail it skips is placedb.apply() (NonLinearPlace.py:942), plotting and
    the legalize/DP branches, and the driver takes its positions from
    placer.pos[0] on both the freeze and the gp_end path. placedb.node_x does
    stay at its pre-GP values here, unlike on the gp_end path -- nothing below
    reads it (amendment D-12).
    """

    def __init__(self, snapshot):
        super().__init__("freeze criterion reached")
        self.snapshot = snapshot


def _die_of(placedb):
    return (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))


def _flag_dreamplace_density_cap(log):
    """Annotate every density-weight clamp entry with whether our own upper
    bound sits above DREAMPlace's hard `clamp(max=10)` (Task 4 review). Pure
    and in-place, so it is testable without a placement."""
    for entry in log:
        entry["hi_abs_capped_by_dreamplace"] = bool(
            float(entry["hi_abs"]) > DREAMPLACE_DENSITY_WEIGHT_MAX)
    return log


def _resolve_regions(die_native, k, rtype, seed, regions_json):
    if regions_json:
        rs = RegionSet.from_json(regions_json)
        rs.validate()
        # IoTerm(K=k), argmax_region(..., k), region_centers(rs), freeze_record
        # and save_membership all assume the file's k IS --k; arms (b)/ours pass
        # a producer regions.json and would otherwise silently run a K=16
        # geometry against a K=4 term (pre-flight amendment C-6).
        if rs.k != k:
            raise ValueError(f"{regions_json} has k={rs.k}, --k is {k}")
        span = max(die_native[2] - die_native[0], die_native[3] - die_native[1])
        if not np.allclose(rs.die, die_native, atol=1e-6 * span, rtol=0.0):
            raise ValueError(f"{regions_json} die {tuple(rs.die)} != placedb die "
                             f"{die_native}; regions.json must be in native "
                             "post-read units")
        return rs, "file"
    return get_regions_for(die_native, k, rtype, seed), "builtin"


def _resolve_prior(membership_npz, remap_blocks, *, nl_fn, rs, num_movable):
    """Load the soft-phase membership prior and, when it carries partitioner
    block ids, remap them onto geometric regions.

    mtkahypar block ids have no geometric meaning (M0 finding, see
    run_placement_two_stage.assign_blocks_to_regions' docstring), so feeding
    them straight in as region ids can throw two heavily connected blocks onto
    opposite die corners. `auto` decides from the artefact's own `source`.

    `nl_fn` is a zero-arg factory, not a netlist: `netlist_from_placedb` copies
    pin2node/pin2net/flat_net2pin, several GB at 10M-30M cells, and the remap
    is the only consumer -- with `remap_blocks="off"` or a non-mtkahypar
    artefact it must never be built (pre-flight amendment D-7). It cannot be
    hoisted and shared with the caller's own netlist either: that one is built
    after `placedb.initialize()`, and `netlist_from_placedb` captures node
    positions and sizes, which `scale()` has rewritten by then.
    """
    mem = load_membership(membership_npz, expect_num_movable=num_movable,
                          expect_k=rs.k)
    if remap_blocks == "auto":
        remap = mem.source.startswith("mtkahypar")
    elif remap_blocks == "on":
        remap = True
    elif remap_blocks == "off":
        remap = False
    else:
        raise ValueError(f"unknown remap_blocks {remap_blocks!r}")
    part = np.asarray(mem.part, dtype=np.int32)
    if remap:
        from ioplace.drivers import run_placement_two_stage
        part = run_placement_two_stage.assign_blocks_to_regions(part, nl_fn(), rs)
    return np.asarray(part, dtype=np.int32), {
        "source": mem.source, "k": mem.k, "seed": mem.seed,
        "epsilon": mem.epsilon, "remapped": bool(remap)}


def _to_native(values, shift, scale):
    return np.asarray(values, dtype=np.float64) / scale + shift


def run_soft_phase(config_json, out_dir, *, k, rtype, seed, regions_json=None,
                   init="die_center", seed_npz=None, membership_npz=None,
                   remap_blocks="auto", norm_policy="grandplan",
                   norm_target_share=None, every=50,
                   home_period=None, rho_max=0.1, f_ft_max=0.0, tau_hi=0.30,
                   tau_lo=0.03, of_on=0.90, of_end=None, of_full=0.20,
                   ft_ramp_mode="window", tau_start=0.12, tau_full=0.05,
                   freeze_window=50, freeze_overflow=0.15, freeze_tau_rel=0.05,
                   freeze_churn=0.005, argmax_chunk=4, ignore_net_degree=None,
                   w_mode="unit", dp_seed=None, deterministic=None,
                   check_invariant=False, timer=None, node_anchor="center"):
    """Phase 1 + phase 2. Writes soft.npz, frozen_membership.npz, freeze.json
    and the active policy's normalisation trace (norm_trace.jsonl, or
    legacy_trace.jsonl under --norm-policy legacy); returns the record the
    caller folds into result.json as `soft_summary`."""
    import torch
    # P-F fix round 1 item 4: run_main_flow is a driver too, and freeze.py
    # already takes membership at the cell centre (freeze.py:9) -- wire the
    # same --node-anchor handling here as run_placement_io.run_io (default
    # 'center', 'pin' rejected before any CUDA/_load_dreamplace call) so this
    # driver stops optimising at the lower-left while freezing at the centre.
    if node_anchor not in ("lower_left", "center", "pin"):
        raise ValueError("node_anchor must be lower_left, center or pin, got %r"
                         % (node_anchor,))
    if node_anchor == "pin":
        raise ValueError(
            "node_anchor='pin' is an IoTermRef-only bias probe (design v2 sec 7); "
            "no driver may run it -- use src/scripts/run_anchor_comparison.py")
    if every <= 0 or freeze_window % every:
        raise ValueError("freeze_window must be a positive multiple of --every "
                         f"(got window={freeze_window}, every={every})")
    home_period = every if home_period is None else home_period
    if home_period % every:
        raise ValueError("home_period must be a multiple of --every")
    if f_ft_max > 0 and rho_max <= 0:
        raise ValueError("--f-ft-max > 0 needs an enabled IO term (--rho-max > 0): "
                         "kappa_ft multiplies lambda_io")
    if init == "region_center" and not membership_npz:
        raise ValueError("--init region_center requires --membership (the prior "
                         "each cell is centred on)")
    if init == "seed" and not seed_npz:
        raise ValueError("--init seed requires --seed-npz")
    os.makedirs(out_dir, exist_ok=True)
    timer = PhaseTimer() if timer is None else timer

    with _io_cleanup() as cleanup:
        with timer.phase("read_soft"):
            params, placedb = _load_dreamplace(config_json)
            cleanup.callback(detach_terms, params)
            if dp_seed is not None:
                params.random_seed = dp_seed
            if deterministic is not None:
                params.deterministic_flag = deterministic
            # Phase 1 never legalises: LG is phase 4, on the fence instance.
            params.legalize_flag = 0
            die_native = _die_of(placedb)
            sha = placedb_identity_sha256(placedb)
            rs_native, region_source = _resolve_regions(die_native, k, rtype, seed,
                                                        regions_json)
            rs_native.to_json(os.path.join(out_dir, REGIONS_JSON))
            m = placedb.num_movable_nodes
            prior_part, prior_info = (None, None)
            if membership_npz:
                prior_part, prior_info = _resolve_prior(
                    membership_npz, remap_blocks,
                    nl_fn=lambda: netlist_from_placedb(placedb),
                    rs=rs_native, num_movable=m)
            seed_positions = None
            if seed_npz:
                seed_positions = load_positions(
                    seed_npz, expect_num_physical=placedb.num_physical_nodes,
                    expect_sha256=sha)
            init_info = apply_init(placedb, params, init, region_set=rs_native,
                                   part=prior_part, positions=seed_positions,
                                   rng_seed=int(params.random_seed))
            placedb.initialize(params)
            shift = (float(params.shift_factor[0]), float(params.shift_factor[1]))
            scale = float(params.scale_factor)
        assert_optimizer_lock(params)
        import NonLinearPlace

        gp_phase = timer.phase("gp_soft")
        gp_phase.__enter__()
        nl = netlist_from_placedb(placedb)
        rs_scaled = scaled_region_set(rs_native, shift, scale)
        rg = RegionGrid(rs_scaled)
        ctx = GpuEvalContext(nl, rg, device="cuda")
        rects, r2k = rect_table(rs_scaled)
        if of_end is None:
            of_end = float(params.stop_overflow)
        if ignore_net_degree is None:
            ignore_net_degree = int(params.ignore_net_degree)
        die_scaled = _die_of(placedb)
        L_R = ((die_scaled[2] - die_scaled[0]) * (die_scaled[3] - die_scaled[1]) / k) ** 0.5

        csr = build_net_node_csr(nl, ignore_net_degree)
        io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                         num_movable=nl.num_movable, num_physical=nl.num_physical,
                         num_nodes=placedb.num_nodes, device="cuda", w_mode=w_mode,
                         node_anchor=node_anchor,
                         node_size_x=nl.node_size_x, node_size_y=nl.node_size_y)
        ft_term, distance = None, None
        if f_ft_max > 0:
            from ioplace.region_graph import region_graph
            from ioplace.ops.ft_term import FtTerm
            _, distance, _ = region_graph(rg)
            ft_term = FtTerm(io_term, distance)
        ecc_max = float(distance.max()) if distance is not None else 0.0

        # The adapter owns its own trace file (legacy_trace.jsonl or
        # norm_trace.jsonl, amendment A-9) and, under grandplan/adaptive, the
        # TermNormalizer: it registers `io`/`ft` here, in the constructor, so
        # every term's activation gate is evaluated on every GP iteration
        # (review I1a), and it needs num_movable/num_nodes for the fixed/filler
        # gradient mask. `out_dir` is where the trace lands; unknown-to-legacy
        # keys -- io_term/ft_term/ecc_max included -- are dropped by
        # make_norm_adapter, so this one call serves every policy. It also owns
        # the flag-combination validation the legacy driver keeps in `run_io`
        # (legacy + norm_p != 1, non-legacy + rho_max == 0).
        adapter = make_norm_adapter(norm_policy, L_R=L_R, rho_max=rho_max,
                                    tau_hi=tau_hi, tau_lo=tau_lo, of_on=of_on,
                                    of_end=of_end, of_full=of_full,
                                    f_ft_max=f_ft_max, ft_ramp_mode=ft_ramp_mode,
                                    tau_start=tau_start, tau_full=tau_full,
                                    out_dir=out_dir, probe_every=every,
                                    target_shares=norm_target_share,
                                    io_term=io_term, ft_term=ft_term,
                                    ecc_max=ecc_max,
                                    num_movable=nl.num_movable,
                                    num_nodes=placedb.num_nodes)
        _cleanup_once(cleanup, adapter.close)
        monitor = FreezeMonitor(window=freeze_window, overflow_max=freeze_overflow,
                                tau_rel_max=freeze_tau_rel, churn_max=freeze_churn)

        def term_fn(pos):
            if not adapter.active or adapter.lambda_io == 0.0:
                return pos.new_zeros(())
            if ft_term is not None:
                return ft_term(pos, adapter.tau, adapter.lambda_io, adapter.kappa_ft)
            return io_term(pos, adapter.tau, adapter.lambda_io)

        attach_terms(params, [term_fn])
        np.random.seed(params.random_seed)
        placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

        n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
        total_iterations = params.global_place_stages[0]["iteration"]
        rects_t = torch.as_tensor(rects, device="cuda")
        r2k_t = torch.as_tensor(r2k, device="cuda")
        size_x = torch.as_tensor(np.asarray(placedb.node_size_x[:m], dtype=np.float64),
                                 device="cuda")
        size_y = torch.as_tensor(np.asarray(placedb.node_size_y[:m], dtype=np.float64),
                                 device="cuda")
        cb_state = {"last_iteration": -1, "num_refreshes": 0, "installed": False,
                    "overflow": float("nan"), "probes": 0,
                    "probe_samples": []}

        def snapshot(iteration, pos, overflow, reason, io_count, argmax):
            return {"iteration": int(iteration), "reason": reason,
                    "overflow": float(overflow), "tau": adapter.tau,
                    "tau_rel": adapter.tau_rel, "churn": monitor.churn,
                    "io_count": int(io_count),
                    "argmax": np.asarray(argmax.cpu(), dtype=np.int32),
                    "node_x": pos.data[:n_phys].detach().cpu().numpy().astype(np.float64),
                    "node_y": pos.data[n_all:n_all + n_phys].detach().cpu().numpy().astype(np.float64)}

        def centre_argmax(pos):
            x = pos.data[:m].double()
            y = pos.data[n_all:n_all + m].double()
            cx, cy = cell_centers(x, y, size_x, size_y)
            # Chunk over regions: region_sdf_l1 materialises (N, R) and this
            # function a further (N, K). At 30M cells x K=16 in float64 that is
            # ~7.7 GB per gated callback (amendment D-8).
            return argmax_region(cx, cy, rects_t.double(), r2k_t, k,
                                 chunk=argmax_chunk)

        def cb(iteration, pos):
            cb_state["last_iteration"] = iteration
            overflow = float(placer.model.overflow.max())
            cb_state["overflow"] = overflow
            gamma = float(placer.model.gamma)
            discrete = adapter.begin_iteration(iteration, overflow, gamma)
            if (iteration > 0 and iteration % every == 0) or iteration == total_iterations - 1:
                res = ctx.evaluate(pos.data[:n_phys], pos.data[n_all:n_all + n_phys])
                if ft_term is not None and (ft_term.home is None
                                            or iteration % home_period == 0):
                    homes = res.per_net_home[csr.net_ids]
                    ft_term.set_home(homes)
                row = adapter.probe(iteration, pos, io_term=io_term, ft_term=ft_term,
                                    wirelength_op=placer.model.op_collections.wirelength_op,
                                    ecc_max=ecc_max, gamma=gamma)
                argmax = centre_argmax(pos)
                # lambda_io/kappa_ft are the *applied* coefficients under
                # every policy (review I1: the non-legacy adapters read them
                # off TermNormalizer.applied_lambda, the legacy one bakes its
                # ramp into lambda_io), so the two arms' probe_samples are
                # directly comparable. lambda_ft is their product.
                sample = {"iteration": int(iteration), "overflow": overflow,
                          "io_count": int(res.io_count),
                          "ft_count": int(res.ft_count),
                          "lambda_io": adapter.lambda_io,
                          "kappa_ft": adapter.kappa_ft,
                          "churn": monitor.observe(iteration, argmax)}
                cb_state["probe_samples"].append(sample)
                # The adapter owns the trace: under legacy this writes
                # publish_atomic's row plus `sample` to legacy_trace.jsonl;
                # under grandplan/adaptive it is a no-op, because
                # TermNormalizer already wrote the design sec 4 row to
                # norm_trace.jsonl at mark_refreshed() time and NormTraceWriter
                # rejects any extra key (amendment A-9). `sample` reaches
                # result.json through soft_summary["probe_samples"] either way.
                adapter.write_trace_row(row, sample)
                cb_state["probes"] += 1
                if monitor.should_freeze(overflow, adapter.tau_rel):
                    # This skips the refresh block below, leaving the adapter in
                    # needs_refresh(). Deliberate: the GP loop is over, nothing
                    # evaluates the objective again, and the freeze path reads
                    # positions rather than gradients (amendment D-13).
                    raise _FreezeReached(snapshot(iteration, pos, overflow,
                                                  "criterion", res.io_count, argmax))
            if discrete or adapter.needs_refresh():
                refresh_nesterov_secant(placer.optimizer)
                adapter.mark_refreshed()
                cb_state["num_refreshes"] += 1
            if check_invariant and not cb_state["installed"]:
                cleanup.callback(install_version_invariant(placer.optimizer, adapter))
                cb_state["installed"] = True

        _install_attribute(cleanup, placer, "iteration_callback", cb)
        lr = params.global_place_stages[0]["learning_rate"]
        try:
            placer(params, placedb, lr)
            pos = placer.pos[0]
            argmax = centre_argmax(pos)
            res = ctx.evaluate(pos.data[:n_phys], pos.data[n_all:n_all + n_phys])
            monitor.observe(cb_state["last_iteration"] + 1, argmax)
            # Both paths carry the same one-step skew and neither can avoid it:
            # DREAMPlace writes model.overflow at NonLinearPlace.py:419, BEFORE
            # that iteration's optimizer step (:447-459), and fires the callback
            # after it (:521-523, see its own ":525 reports the metric before
            # step"). So every `overflow` in this driver -- probe_samples, the
            # freeze criterion, freeze.json and run_fence_gp's final_overflow --
            # is evaluated one step before the positions it is filed with. The
            # fresh read here only removes the dependency on cb_state; it is
            # value-identical, because nothing updates model.overflow between the
            # last callback and placer(...) returning.
            overflow = float(placer.model.overflow.max())
            shot = snapshot(cb_state["last_iteration"], pos, overflow,
                            "gp_end", res.io_count, argmax)
        except _FreezeReached as event:
            shot = event.snapshot
        density_weight_soft = float(placer.model.density_weight.detach().max())
        gp_phase.__exit__(None, None, None)

        with timer.phase("freeze"):
            centers_scaled = region_centers(rs_scaled)
            cx = shot["node_x"][:m] + np.asarray(placedb.node_size_x[:m], dtype=np.float64) / 2.0
            cy = shot["node_y"][:m] + np.asarray(placedb.node_size_y[:m], dtype=np.float64) / 2.0
            part, repaired = ensure_nonempty_regions(shot["argmax"], k, cx, cy,
                                                     centers_scaled)
            native_x = _to_native(shot["node_x"], shift[0], scale)
            native_y = _to_native(shot["node_y"], shift[1], scale)
            size_x_native = np.asarray(placedb.node_size_x[:m], dtype=np.float64) / scale
            size_y_native = np.asarray(placedb.node_size_y[:m], dtype=np.float64) / scale
            soft_path = os.path.join(out_dir, SOFT_NPZ)
            membership_path = os.path.join(out_dir, FROZEN_MEMBERSHIP_NPZ)
            save_positions(soft_path, native_x, native_y, die=die_native,
                           shift_factor=shift, scale_factor=scale,
                           placedb_sha256=sha, kind="soft")
            save_membership(membership_path, part, source="freeze", k=k,
                            seed=seed, epsilon=0.0)
            record = freeze_record(
                iteration=shot["iteration"], reason=shot["reason"],
                overflow=shot["overflow"], tau=shot["tau"], tau_rel=shot["tau_rel"],
                churn=shot["churn"], k=k, io_soft=shot["io_count"],
                membership_npz=os.path.abspath(membership_path),
                soft_npz=os.path.abspath(soft_path),
                repaired_empty_regions=repaired,
                gp_iterations_soft=cb_state["last_iteration"] + 1,
                density_weight_soft=density_weight_soft,
                stats=region_cell_stats(part, size_x_native, size_y_native, rs_native),
                node_anchor=node_anchor)
            save_freeze(os.path.join(out_dir, FREEZE_JSON), record)
        detach_terms(params)

    return {"freeze": record, "part": part, "soft_npz": soft_path,
            "membership_npz": membership_path, "regions_json":
            os.path.join(out_dir, REGIONS_JSON), "region_source": region_source,
            "init": init_info, "prior": prior_info,
            "norm_policy": norm_policy, "trace_path": adapter.trace_path,
            "probe_samples": cb_state["probe_samples"],
            "num_probes": cb_state["probes"],
            "num_refreshes": cb_state["num_refreshes"],
            "gp_iterations_soft": cb_state["last_iteration"] + 1,
            "density_weight_soft": density_weight_soft,
            "placedb_sha256": sha, "die_native": die_native}


def run_fence_gp(config_json, out_dir, *, region_set, part, positions,
                 reference_density_weight, density_clamp_lo=0.25,
                 density_clamp_hi=4.0, dp_seed=None, deterministic=None,
                 extra_terms=(), timer=None):
    """Phase 3 + phase 4 + evaluator.

    `extra_terms` is the documented attachment point for P-D's capacity term
    and P-E's pseudo-FT term. It is empty by default: after the freeze the IO
    and FT terms are OFF (design v2 sec 3, "terms after the freeze").
    """
    import torch
    timer = PhaseTimer() if timer is None else timer
    clamp_log = []

    with _io_cleanup() as cleanup:
        with timer.phase("read_fence"):
            params, placedb, info = build_fence_placedb(
                config_json, region_set, part, positions, dp_seed=dp_seed,
                deterministic=deterministic)
            cleanup.callback(detach_terms, params)
            scale_fields = _effective_scale_fields(params, placedb)
        assert_optimizer_lock(params)
        import NonLinearPlace

        gp_phase = timer.phase("gp_fence")
        gp_phase.__enter__()
        nl = netlist_from_placedb(placedb)
        rs_scaled = scaled_region_set(region_set, info["shift_factor"],
                                      info["scale_factor"])
        rg = RegionGrid(rs_scaled)
        ctx = GpuEvalContext(nl, rg, device="cuda")
        install_density_weight_clamp(cleanup, reference_density_weight, clamp_log,
                                     lo=density_clamp_lo, hi=density_clamp_hi)
        if extra_terms:
            attach_terms(params, list(extra_terms))
        np.random.seed(params.random_seed)
        placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

        n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
        holder = {}
        original_legalize = placer.op_collections.legalize_op

        def timed_legalize(pos):
            gp_phase.__exit__(None, None, None)
            with torch.no_grad():
                holder["hpwl_gp"] = float(placer.op_collections.hpwl_op(pos))
            # exact io(fence GP): measured on the GP positions handed to LG, so
            # lg_loss can never be contaminated by a stale periodic callback.
            holder["io_fence_gp"] = int(ctx.evaluate(
                pos.data[:n_phys], pos.data[n_all:n_all + n_phys]).io_count)
            with timer.phase("lg"):
                out = original_legalize(pos)
            with torch.no_grad():
                holder["hpwl_lg"] = float(placer.op_collections.hpwl_op(out))
            return out

        _install_attribute(cleanup, placer.op_collections, "legalize_op", timed_legalize)
        cb_state = {"last_iteration": -1}

        def cb(iteration, pos):
            cb_state["last_iteration"] = iteration

        _install_attribute(cleanup, placer, "iteration_callback", cb)
        lr = params.global_place_stages[0]["learning_rate"]
        placer(params, placedb, lr)
        # Key the guard off *this* phase's own name: `timer` is shared across
        # all four phases here, so testing for "lg" would silently stop closing
        # gp_fence the day any other phase is named lg (amendment D-9).
        if "gp_fence" not in timer.phases:
            gp_phase.__exit__(None, None, None)

        with timer.phase("eval"):
            node_x, node_y = extract_final_positions(placer, placedb)
            legal_fields = _legalization_diagnostics(placer, placedb, params,
                                                     node_x, node_y)
            res = ctx.evaluate(node_x, node_y)
            metrics = _pack_eval_metrics(res)
            evaluation_path = os.path.join(out_dir, EVALUATION_NPZ)
            save_evaluation(evaluation_path, nl, rg, res, node_x, node_y,
                            placedb.net_names,
                            provenance={"config": os.path.abspath(config_json),
                                        "placement_stage": "fence_gp_lg",
                                        "shift_factor": list(info["shift_factor"]),
                                        "scale_factor": info["scale_factor"]})
            placement_path = os.path.join(out_dir, PLACEMENT_NPZ)
            native_placement_x = _to_native(node_x, info["shift_factor"][0],
                                            info["scale_factor"])
            native_placement_y = _to_native(node_y, info["shift_factor"][1],
                                            info["scale_factor"])
            save_positions(placement_path, native_placement_x, native_placement_y,
                           die=info["die_native"], shift_factor=info["shift_factor"],
                           scale_factor=info["scale_factor"],
                           placedb_sha256=info["placedb_sha256"], kind="placement")
            # Fence-diagnostics fix (P-F Task 7 finding): final_overflow (kept
            # for continuity) is the raw max across all K+1 fence-mode
            # buckets, which in this driver is usually dominated by the
            # degenerate escape-cell bucket (fence_phase.py:134-142) rather
            # than the design's real overflow -- see
            # _fence_overflow_stop_metric's docstring. final_overflow_regions
            # is the full per-bucket vector so a reader can tell the two
            # apart instead of taking final_overflow at face value.
            overflow_regions = [float(v) for v in placer.model.overflow.reshape(-1)]
            final_overflow = float(placer.model.overflow.max())
            final_overflow_stop_metric = _fence_overflow_stop_metric(
                overflow_regions, len(placedb.regions) > 0)
            m = placedb.num_movable_nodes
            compliance = fence_compliance(
                rg, node_x, node_y, part,
                np.asarray(placedb.node_size_x[:m], dtype=np.float64),
                np.asarray(placedb.node_size_y[:m], dtype=np.float64))
            # Native units, native RegionSet: region_area/region_cell_area are
            # not scale-invariant, and freeze.json already carries them in the
            # native frame via region_cell_stats -- result.json must quote the
            # same numbers (amendment D-2). fence_compliance above stays in the
            # scaled frame because it is pure geometry against `rg`.
            size_x_native = (np.asarray(placedb.node_size_x[:m], dtype=np.float64)
                             / info["scale_factor"])
            size_y_native = (np.asarray(placedb.node_size_y[:m], dtype=np.float64)
                             / info["scale_factor"])
            balance = region_area_balance(part, size_x_native, size_y_native,
                                          region_set)
        detach_terms(params)

    io_fence_gp = holder.get("io_fence_gp")
    return {"metrics": metrics,
            # R-1 (task 5 pre-flight ruling): _pack_straddle_metrics needs the
            # raw EvalResult, which "metrics" (already packed by
            # _pack_eval_metrics) does not carry.
            "eval_result": res,
            "io_fence_gp": metrics["io_count"] if io_fence_gp is None else io_fence_gp,
            # Where io_fence_gp actually came from. io_identity_residual is 0
            # for any three inputs and can never detect the fallback; this can
            # (amendment D-1). "fallback" means legalize_op never fired -- the
            # config had legalize_flag=0 -- so io_fence_gp is the post-"LG"
            # number and lg_loss is 0 by definition rather than by measurement.
            "io_fence_gp_source": "fallback" if io_fence_gp is None else "legalize_op",
            "hpwl_gp": holder.get("hpwl_gp"), "hpwl_lg": holder.get("hpwl_lg"),
            "fence_compliance": compliance, "region_area_balance": balance,
            "density_weight_clamp": _flag_dreamplace_density_cap(clamp_log),
            "escape_cell": info["escape_cell"],
            "escape_from": info["escape_from"], "legal_fields": legal_fields,
            "scale_fields": scale_fields, "final_overflow": final_overflow,
            "final_overflow_regions": overflow_regions,
            "final_overflow_stop_metric": final_overflow_stop_metric,
            "stop_overflow_reached": _stop_overflow_reached(
                final_overflow_stop_metric, params.stop_overflow),
            "gp_iterations_fence": cb_state["last_iteration"] + 1,
            "gp_iteration_budget": _gp_iteration_budget(params),
            "placement_npz": placement_path, "evaluation_npz": evaluation_path,
            "params_seed": int(params.random_seed),
            "deterministic": int(params.deterministic_flag)}


def run_main_flow(config_json, out_dir, *, k=16, rtype="grid", seed=0,
                  regions_json=None, phase="all", init="die_center",
                  seed_npz=None, membership_npz=None, remap_blocks="auto",
                  norm_policy="grandplan", norm_target_share=None, every=50,
                  home_period=None, rho_max=0.1,
                  f_ft_max=0.0, tau_hi=0.30, tau_lo=0.03, of_on=0.90,
                  of_end=None, of_full=0.20, ft_ramp_mode="window",
                  tau_start=0.12, tau_full=0.05, freeze_window=50,
                  freeze_overflow=0.15, freeze_tau_rel=0.05, freeze_churn=0.005,
                  argmax_chunk=4, density_clamp_lo=0.25, density_clamp_hi=4.0,
                  ignore_net_degree=None, w_mode="unit", dp_seed=None,
                  deterministic=None, check_invariant=False,
                  benchmark_kind="real", extra_terms=(), node_anchor=None):
    import torch
    if phase not in ("all", "soft", "fence"):
        raise ValueError(f"unknown phase {phase!r}")
    if init not in INIT_MODES:
        raise ValueError(f"unknown init {init!r}")
    if norm_policy not in NORM_POLICIES:
        raise ValueError(f"unknown norm policy {norm_policy!r}")
    # Fence-diagnostics fix (P-F Task 7 finding): `node_anchor=None` means "not
    # given" -- distinct from an explicit "center" -- so a --phase fence call
    # that omits --node-anchor can silently take the anchor the soft phase
    # recorded in freeze.json, while one that passes a *conflicting* explicit
    # value still raises instead of the old behaviour of always trusting this
    # argument (a fence-only invocation would then just relabel whatever the
    # soft phase actually used). `node_anchor_given` is threaded through to
    # the freeze-vs-argument comparison below; `effective_node_anchor` is what
    # actually runs when a value is needed now (the soft phase, or the
    # up-front validation).
    node_anchor_given = node_anchor is not None
    effective_node_anchor = node_anchor if node_anchor_given else "center"
    # P-F fix round 1 item 4: reject here too (not just inside run_soft_phase)
    # so --phase fence -- which never calls run_soft_phase, since IO/FT are
    # off after the freeze -- still rejects a bad --node-anchor up front,
    # before os.makedirs/torch.cuda.mem_get_info below.
    if effective_node_anchor not in ("lower_left", "center", "pin"):
        raise ValueError("node_anchor must be lower_left, center or pin, got %r"
                         % (effective_node_anchor,))
    if effective_node_anchor == "pin":
        raise ValueError(
            "node_anchor='pin' is an IoTermRef-only bias probe (design v2 sec 7); "
            "no driver may run it -- use src/scripts/run_anchor_comparison.py")
    os.makedirs(out_dir, exist_ok=True)
    device_baseline_gb = 0.0
    if torch.cuda.is_available():
        free0, total0 = torch.cuda.mem_get_info()
        device_baseline_gb = (total0 - free0) / 2 ** 30
    t0 = time.time()
    timer = PhaseTimer()
    sampler = DeviceMemSampler()
    sampler.start()
    try:
        soft, soft_summary = None, None
        if phase in ("all", "soft"):
            soft = run_soft_phase(
                config_json, out_dir, k=k, rtype=rtype, seed=seed,
                regions_json=regions_json, init=init, seed_npz=seed_npz,
                membership_npz=membership_npz, remap_blocks=remap_blocks,
                norm_policy=norm_policy, norm_target_share=norm_target_share,
                every=every, home_period=home_period,
                rho_max=rho_max, f_ft_max=f_ft_max, tau_hi=tau_hi, tau_lo=tau_lo,
                of_on=of_on, of_end=of_end, of_full=of_full,
                ft_ramp_mode=ft_ramp_mode, tau_start=tau_start, tau_full=tau_full,
                freeze_window=freeze_window, freeze_overflow=freeze_overflow,
                freeze_tau_rel=freeze_tau_rel, freeze_churn=freeze_churn,
                argmax_chunk=argmax_chunk,
                ignore_net_degree=ignore_net_degree, w_mode=w_mode,
                dp_seed=dp_seed, deterministic=deterministic,
                check_invariant=check_invariant, timer=timer,
                node_anchor=effective_node_anchor)
            # Everything the soft phase decided that result.json would otherwise
            # lose: region_source, the resolved init record, the prior
            # (including `remapped`, which is the --remap-blocks auto decision
            # a reviewer of arms (a)/(b) will ask about), the probe samples and
            # which trace file the policy wrote (amendment D-6). `part` is a
            # numpy array and `freeze` is already result["freeze"].
            soft_summary = {key: value for key, value in soft.items()
                            if key not in ("part", "freeze")}
            torch.cuda.empty_cache()
            if phase == "soft":
                with open(os.path.join(out_dir, SOFT_RESULT_JSON), "w") as stream:
                    json.dump({key: value for key, value in soft.items()
                               if key != "part"}, stream, indent=1, default=str)
                return soft

        soft_path = os.path.join(out_dir, SOFT_NPZ)
        freeze_path = os.path.join(out_dir, FREEZE_JSON)
        membership_path = os.path.join(out_dir, FROZEN_MEMBERSHIP_NPZ)
        regions_path = regions_json or os.path.join(out_dir, REGIONS_JSON)
        for required in (soft_path, freeze_path, membership_path, regions_path):
            if not os.path.exists(required):
                raise FileNotFoundError(
                    f"--phase fence needs {os.path.basename(required)} in {out_dir}; "
                    "run --phase soft (or --phase all) first")
        record = load_freeze(freeze_path)
        # Fence-diagnostics fix (P-F Task 7 finding): freeze.json is
        # authoritative for the anchor the soft phase actually ran with,
        # whether it ran in this same process (phase="all") or a separate
        # earlier invocation (phase="fence" reading its artefacts back). A
        # fence-only call that explicitly passes a *conflicting*
        # --node-anchor is almost certainly a mistake -- raise instead of
        # silently preferring either value; one that omits it (the common
        # case) just inherits the recorded anchor.
        if (phase == "fence" and node_anchor_given
                and effective_node_anchor != record["node_anchor"]):
            raise ValueError(
                f"--node-anchor {effective_node_anchor!r} disagrees with "
                f"node_anchor {record['node_anchor']!r} recorded in "
                f"{freeze_path} by the soft phase; omit --node-anchor on a "
                "--phase fence run to use the recorded value, or pass the "
                "same anchor")
        node_anchor = record["node_anchor"]
        positions = load_positions(soft_path)
        region_set = RegionSet.from_json(regions_path)
        region_set.validate()
        part = load_membership(membership_path, expect_k=region_set.k,
                               require_nonempty=True).part

        fence = run_fence_gp(
            config_json, out_dir, region_set=region_set, part=part,
            positions=positions,
            reference_density_weight=record["density_weight_soft"],
            density_clamp_lo=density_clamp_lo,
            density_clamp_hi=density_clamp_hi, dp_seed=dp_seed,
            deterministic=deterministic, extra_terms=extra_terms, timer=timer)
    finally:
        sampler.stop()
        # Fix wave item 4: the fence branch leaves a fixed 32 MiB cuBLAS
        # workspace behind (PlaceObj.py:321's wirelength + density_weight
        # .dot(density), taken only when regions are fenced) that gc/
        # empty_cache() alone cannot reclaim. This runs after the sampler
        # has stopped and after every phase peak this run reports has
        # already been captured inside the `try`, so it cannot perturb
        # this run's own numbers.
        release_cuda_scratch()

    accounting = io_accounting(record["io_soft"], fence["io_fence_gp"],
                               fence["metrics"]["io_count"])
    artefacts = {name: {"path": os.path.abspath(path),
                        "sha256": file_sha256(path)}
                 for name, path in (
                     ("regions_json", regions_path), ("soft_npz", soft_path),
                     ("freeze_json", freeze_path),
                     ("membership_npz", membership_path),
                     ("placement_npz", fence["placement_npz"]),
                     ("evaluation_npz", fence["evaluation_npz"]),
                     # whichever the policy wrote (amendment A-9); the
                     # os.path.exists filter drops the other
                     ("norm_trace", os.path.join(out_dir, NORM_TRACE)),
                     ("legacy_trace", os.path.join(out_dir, LEGACY_TRACE)))
                 if os.path.exists(path)}
    result = {
        **fence["metrics"], **accounting, **_pack_straddle_metrics(fence["eval_result"]),
        "mode": "main_flow", "schema_version": MAIN_FLOW_RESULT_SCHEMA_VERSION,
        "config": os.path.abspath(config_json), "k": int(region_set.k),
        "rtype": rtype, "seed": seed, "init": init, "norm_policy": norm_policy,
        "phase": phase, "regions_json": os.path.abspath(regions_path),
        "dp_seed": fence["params_seed"], "det": fence["deterministic"],
        "runtime_s": time.time() - t0,
        # v2 P-F (design sec 7): the anchor the soft phase ran with.
        "node_anchor": node_anchor,
        "hpwl_gp": fence["hpwl_gp"], "hpwl_lg": fence["hpwl_lg"],
        "fence_compliance": fence["fence_compliance"]["lower_left"],
        "fence_compliance_center": fence["fence_compliance"]["center"],
        "region_area_balance": fence["region_area_balance"],
        "freeze": record, "density_weight_clamp": fence["density_weight_clamp"],
        "escape_cell": {"index": fence["escape_cell"], "from": fence["escape_from"]},
        "gp_iterations_soft": record["gp_iterations_soft"],
        "gp_iterations_fence": fence["gp_iterations_fence"],
        "gp_iteration_budget": fence["gp_iteration_budget"],
        "io_fence_gp_source": fence["io_fence_gp_source"],
        "soft_summary": soft_summary,
        "final_overflow": fence["final_overflow"],
        "final_overflow_regions": fence["final_overflow_regions"],
        "final_overflow_stop_metric": fence["final_overflow_stop_metric"],
        "stop_overflow_reached": fence["stop_overflow_reached"],
        "artifacts": artefacts,
        **fence["legal_fields"],
        **{key: fence["scale_fields"][key] for key in
           ("effective_target_density", "num_filler_nodes", "num_bins_x", "num_bins_y")},
        **phase_summary(timer, sampler),
        **_t8a_provenance(config_json, benchmark_kind=benchmark_kind,
                          device_baseline_gb=device_baseline_gb),
    }
    # _t8a_provenance carries run_placement's own RESULT_SCHEMA_VERSION (3);
    # it is spread last, so restore this driver's schema version afterwards.
    result["schema_version"] = MAIN_FLOW_RESULT_SCHEMA_VERSION
    save_result(os.path.join(out_dir, RESULT_JSON), result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="v2 main flow: soft GP -> freeze -> fence GP -> fence LG -> evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase", choices=["all", "soft", "fence"], default="all")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regions", default=None,
                        help="producer regions.json (native post-read units); "
                             "omit to use the built-in grid")
    parser.add_argument("--init", choices=list(INIT_MODES), default="die_center")
    parser.add_argument("--seed-npz", default=None)
    parser.add_argument("--membership", default=None)
    parser.add_argument("--remap-blocks", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--norm-policy", choices=list(NORM_POLICIES),
                        default="grandplan")
    parser.add_argument("--norm-target-share", default=None,
                        help="policy B force shares as fractions of the total "
                             "force, e.g. 'io=0.3,ft=0.1'; defaults io=0.3, "
                             "ft=--f-ft-max, and a name no registered term "
                             "answers to is an error. Ignored by "
                             "--norm-policy legacy")
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--home-period", type=int, default=None)
    parser.add_argument("--rho-max", type=float, default=0.1)
    parser.add_argument("--f-ft-max", type=float, default=0.0)
    parser.add_argument("--tau-hi", type=float, default=0.30)
    parser.add_argument("--tau-lo", type=float, default=0.03)
    parser.add_argument("--of-on", type=float, default=0.90)
    parser.add_argument("--of-end", type=float, default=None)
    parser.add_argument("--of-full", type=float, default=0.20)
    parser.add_argument("--ft-ramp-mode", choices=["window", "constant"], default="window")
    parser.add_argument("--tau-start", type=float, default=0.12)
    parser.add_argument("--tau-full", type=float, default=0.05)
    parser.add_argument("--freeze-window", type=int, default=50)
    parser.add_argument("--freeze-overflow", type=float, default=0.15)
    parser.add_argument("--freeze-tau-rel", type=float, default=0.05)
    parser.add_argument("--freeze-churn", type=float, default=0.005)
    parser.add_argument("--argmax-chunk", type=int, default=4,
                        help="regions per chunk in the freeze argmax; the "
                             "unchunked (N, R) + (N, K) intermediates are "
                             "~7.7 GB at 30M cells x K=16 in float64")
    parser.add_argument("--density-clamp-lo", type=float, default=0.25)
    parser.add_argument("--density-clamp-hi", type=float, default=4.0)
    parser.add_argument("--d-max", type=int, default=None, dest="ignore_net_degree")
    parser.add_argument("--w-mode", default="unit", choices=["unit", "inv_deg"])
    parser.add_argument("--dp-seed", type=int, default=None)
    parser.add_argument("--deterministic", type=int, default=None)
    parser.add_argument("--check-invariant", action="store_true")
    parser.add_argument("--benchmark-kind", default="real", choices=["real", "synthetic"])
    # v2 P-F (design sec 7): the anchor at which the soft region assignment is
    # evaluated. --phase fence is a no-op for placement (IO/FT are off after
    # the freeze) -- fence-diagnostics fix (P-F Task 7 finding): default is
    # None ("not given"), not "center", so run_main_flow can tell a fence-only
    # call that omits this flag (inherit the anchor freeze.json recorded)
    # apart from one that explicitly passes a conflicting value (raise).
    parser.add_argument("--node-anchor", choices=["lower_left", "center", "pin"],
                        default=None,
                        help="anchor for the soft region assignment: 'center' "
                             "(the default when --phase is soft/all) "
                             "evaluates the SDF at x+0.5*w, y+0.5*h, matching "
                             "the freeze rule (freeze.py's membership is "
                             "already the cell centre) and whole-cell fence "
                             "ownership; 'lower_left' is the legacy anchor; "
                             "'pin' is rejected by every driver -- it exists "
                             "only in IoTermRef as a small-scale bias probe; "
                             "no-op for placement under --phase fence (IO/FT "
                             "are off after the freeze) -- omit it there to "
                             "report the anchor freeze.json recorded from the "
                             "soft phase, or pass the same one it used")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_main_flow(
        args.config, args.out_dir, k=args.k, rtype=args.rtype, seed=args.seed,
        regions_json=args.regions, phase=args.phase, init=args.init,
        seed_npz=args.seed_npz, membership_npz=args.membership,
        remap_blocks=args.remap_blocks, norm_policy=args.norm_policy,
        norm_target_share=args.norm_target_share,
        every=args.every, home_period=args.home_period, rho_max=args.rho_max,
        f_ft_max=args.f_ft_max, tau_hi=args.tau_hi, tau_lo=args.tau_lo,
        of_on=args.of_on, of_end=args.of_end, of_full=args.of_full,
        ft_ramp_mode=args.ft_ramp_mode, tau_start=args.tau_start,
        tau_full=args.tau_full, freeze_window=args.freeze_window,
        freeze_overflow=args.freeze_overflow, freeze_tau_rel=args.freeze_tau_rel,
        freeze_churn=args.freeze_churn, argmax_chunk=args.argmax_chunk,
        density_clamp_lo=args.density_clamp_lo,
        density_clamp_hi=args.density_clamp_hi,
        ignore_net_degree=args.ignore_net_degree, w_mode=args.w_mode,
        dp_seed=args.dp_seed, deterministic=args.deterministic,
        check_invariant=args.check_invariant, benchmark_kind=args.benchmark_kind,
        node_anchor=args.node_anchor)
    return result


if __name__ == "__main__":
    main()
