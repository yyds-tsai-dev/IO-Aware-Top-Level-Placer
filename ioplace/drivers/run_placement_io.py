import json, os, time
import numpy as np
import scipy.stats
from ioplace.drivers.run_placement import (_load_dreamplace, extract_final_positions,
    _evaluate_and_pack, get_regions_for, _legalization_diagnostics, _phase_summary,
    _t8a_provenance)
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.ops.soft_assign import rect_table
from ioplace.ops.io_term import build_net_node_csr, IoTerm
from ioplace.profile import PhaseTimer, DeviceMemSampler
from ioplace.schedules import ScheduleState
from ioplace.dp_hook import (attach_terms, detach_terms, assert_optimizer_lock,
                             refresh_nesterov_secant, install_version_invariant)
from ioplace.reweight import update_net_weights

RESULT_FIELDS = ("mode", "config", "k", "rtype", "seed", "dp_seed", "det",
                 "io_count", "io_gp", "ft_count", "hard_lambda_sum", "tree_wl", "hpwl",
                 "io_rg", "ft_rg",
                 "lg_loss", "runtime_s", "peak_mem_mb", "peak_mem_mb_reset_semantics",
                 "rho_max", "tau_hi", "tau_lo",
                 "of_on", "of_end", "alpha_io", "w_mode", "d_max", "rho_margin",
                 "margin_m", "lambda_io_final", "spearman_rho", "num_callbacks",
                 "num_refreshes", "backtrack_median", "observer_mode",
                 "diag_every", "no_diag", "ft_reweight", "alpha_ft", "trajectory",
                 # M4 T8a (design draft T8a row / sec 6.3 E1 / sec 7.0 RESULT GATE):
                 "num_unplaced_cells", "final_overflow", "legalization_status",
                 "run_id", "status", "schema_version", "repo_commit", "input_sha256",
                 "env", "phases", "t_read", "t_gp", "t_lg", "t_eval",
                 "device_used_gb", "host_peak_rss_gb")


def _normalize_snapshot_iters(spec):
    """Accept None, a comma-separated string ("300,350,400"), or any iterable
    of ints; return a frozenset of ints (or None). String form mirrors the
    M3 design draft v3.1 T0-a CLI shorthand (`--snapshot-iters
    "300,350,400,450,500,550"`) so a future CLI wrapper can pass the raw
    flag value straight through without its own parsing."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [s for s in spec.split(",") if s.strip()]
    iters = frozenset(int(s) for s in spec)
    return iters if iters else None


def _raw_wl_density_grad(model, p):
    """Raw (unpreconditioned) gradient of wirelength + density_weight*density
    w.r.t. leaf tensor `p`, via the model's own obj_fn -- reusing it (rather
    than reimplementing the quad-penalty/fence-region composition) avoids
    drifting out of sync with PlaceObj.obj_fn. Any io-aware extra objective
    terms (design v2 sec 6.2's params-borne attach_terms) are temporarily
    detached so only the base DREAMPlace wl+density objective contributes --
    this is exactly the M3 design draft v3.1 T0-a `g_wl_density` field.

    `p` must be an independent leaf tensor (e.g.
    pos.detach().clone().requires_grad_(True)), never the live optimizer's
    own `pos` -- this function's backward() must not perturb pos.grad or any
    optimizer/Nesterov-secant state, since T0-a's invariant is that adding
    --snapshot-iters leaves the run bit-identical to a run without it."""
    saved_terms = model.extra_obj_terms
    model.extra_obj_terms = []
    try:
        obj = model.obj_fn(p)
        obj.backward()
        return p.grad.detach().clone()
    finally:
        model.extra_obj_terms = saved_terms


def run_io(config_json, k, rtype, seed, out_json, *,
           rho_max=0.1, tau_hi=0.30, tau_lo=0.03,
           of_on=0.90, of_end=None, of_full=0.20,
           alpha_io=0.0, cap=64.0, ignore_net_degree=None,
           rho_margin=0.0, margin_m=None, margin_tau=None,
           of_margin=0.15, w_mode="unit", every=50,
           dp_seed=None, deterministic=None, check_invariant=False,
           diag_every=1, no_diag=False,
           ft_reweight="off", alpha_ft=0.5,
           snapshot_iters=None, snapshot_dir=None, snapshot_grad_check_cb=None):
    import torch
    t0 = time.time()
    # M4 T8a: replaces the old single reset_peak_memory_stats() call (sec
    # 1.4 B1's fix) with PhaseTimer's per-phase reset -- see
    # run_placement._phase_summary's docstring for why peak_mem_mb is
    # recomputed from the per-phase peaks rather than a single end-of-run
    # read.
    timer = PhaseTimer()
    sampler = DeviceMemSampler()
    sampler.start()

    with timer.phase("read"):
        params, placedb = _load_dreamplace(config_json)
        if dp_seed is not None:
            params.random_seed = dp_seed
        if deterministic is not None:
            params.deterministic_flag = deterministic
        placedb.initialize(params)
    # design v2 sec 3.2.4: use_bb is only resolved to a concrete 0/1 by
    # PlaceDB.py:837, which runs inside initialize() -- must check after.
    assert_optimizer_lock(params)
    # NonLinearPlace is a bare top-level module inside $DREAMPLACE_ROOT/install
    # (see Global Constraints), only importable once _load_dreamplace has
    # called setup_dreamplace() and pushed install/ onto sys.path.
    import NonLinearPlace

    # M4 T8a: "gp" phase opened manually (not `with`) -- it must close
    # mid-call, at the exact point NonLinearPlace.__call__ invokes
    # legalize_op (see the op_collections.legalize_op monkeypatch below,
    # inserted once `placer` exists). Everything from here through
    # `placer(params, placedb, lr)`'s GP loop happens inside this window.
    gp_phase = timer.phase("gp")
    gp_phase.__enter__()

    nl = netlist_from_placedb(placedb)           # initialize 後(scale 後)座標系
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, k, rtype, seed)
    rg = RegionGrid(rs)
    ctx = GpuEvalContext(nl, rg, device="cuda")

    if of_end is None:
        of_end = float(params.stop_overflow)
    if ignore_net_degree is None:
        ignore_net_degree = int(params.ignore_net_degree)
    if margin_m is None:
        margin_m = 2.0 * float(np.median(nl.node_size_y[:nl.num_movable]))
    if margin_tau is None:
        margin_tau = margin_m / 2.0

    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, ignore_net_degree)
    io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                     num_movable=nl.num_movable, num_physical=nl.num_physical,
                     num_nodes=placedb.num_nodes, device="cuda", w_mode=w_mode)

    L_R = ((die[2] - die[0]) * (die[3] - die[1]) / k) ** 0.5

    state = ScheduleState(rho_max=rho_max, tau_hi=tau_hi, tau_lo=tau_lo,
                          of_on=of_on, of_end=of_end, of_full=of_full,
                          alpha_io=alpha_io, rho_margin=rho_margin,
                          margin_m=margin_m, margin_tau=margin_tau, of_margin=of_margin)

    # design v2 sec 5.2/6.4: rho_max==0 and rho_margin==0 is a *provable* no-op
    # -- the term is never attached, so the objective is bit-identical to
    # run_flat. T6's flat baseline runs through this path to get the
    # io_gp/lg_loss/hard_lambda_sum columns it needs.
    observer_mode = (rho_max == 0.0 and rho_margin == 0.0)

    def term_fn(pos):
        if not state.active or (state.lambda_io == 0.0 and state.lambda_margin == 0.0):
            return pos.new_zeros(())
        return io_term(pos, state.tau, state.lambda_io, state.lambda_margin,
                       state.margin_m, state.margin_tau)

    if not observer_mode:
        attach_terms(params, [term_fn])   # must precede NonLinearPlace(...) construction

    # Same init_pos determinism guard as run_placement._place() / run_reweight:
    # BasicPlace draws centre-noise/filler init from numpy's global RNG, seeded
    # only by Placer.py's flow which we bypass here.
    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

    # M4 T8a: same op_collections.legalize_op timing wrap as
    # run_placement.run_flat -- see that function's comment for why this is
    # the only non-invasive way (no DREAMPlace source touched) to observe
    # the GP/LG boundary.
    orig_legalize = placer.op_collections.legalize_op
    def _timed_legalize(pos):
        gp_phase.__exit__(None, None, None)
        with timer.phase("lg"):
            out = orig_legalize(pos)
        return out
    placer.op_collections.legalize_op = _timed_legalize

    n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
    total_iterations = params.global_place_stages[0]["iteration"]

    if diag_every < 1:
        raise ValueError(f"diag_every must be >= 1, got {diag_every}")
    if ft_reweight not in ("off", "on"):
        raise ValueError(f"ft_reweight must be 'off' or 'on', got {ft_reweight!r}")
    if ft_reweight == "on" and alpha_io > 0:
        # M3 Phase B (adjudication doc sec C item 2): the ft_rg-signalled
        # reweight below and the pre-existing alpha_io (IO-crossings)
        # reweight both write io_term.w from a *different* per-net signal --
        # running both at once is not defined by the M1 formula family
        # reused here (which writes one w = 1 + alpha*min(signal, cap) per
        # net, not a sum of two). Pick one signal per run.
        raise ValueError("ft_reweight='on' and alpha_io>0 are mutually "
                         "exclusive (both write io_term.w)")

    # design v3.1 T0-a: snapshots are written from *inside* the same N=50
    # evaluator-gated branch that already produces the trajectory log
    # (`iteration % every == 0`, or the final force_eval iteration) -- never
    # on an independent per-iteration check. A requested iteration that
    # can't land on that branch would silently produce fewer snapshots than
    # asked for, so fail fast instead (design v3.1 sec 3.4.0's "any failure
    # must not be silently skipped" spirit, applied to this precondition).
    snapshot_iters = _normalize_snapshot_iters(snapshot_iters)
    if snapshot_iters:
        if not snapshot_dir:
            raise ValueError("snapshot_iters given without snapshot_dir")
        unreachable = sorted(it for it in snapshot_iters
                             if not (it > 0 and it % every == 0)
                             and it != total_iterations - 1)
        if unreachable:
            raise ValueError(
                f"snapshot_iters {unreachable} unreachable: must be a positive "
                f"multiple of every={every} (or the final iteration "
                f"{total_iterations - 1}) -- snapshots are taken inside the "
                "same N=every evaluator-gated branch as the trajectory log")
        os.makedirs(snapshot_dir, exist_ok=True)

    trajectory = []
    cb_state = {"num_refreshes": 0, "io_gp": 0, "prev_obj_evals": 0,
               "obj_evals_per_iter": [], "installed_invariant": False,
               "diag_occurrences": 0}

    def cb(iteration, pos):
        of = float(placer.model.overflow.max())
        gamma = float(placer.model.gamma)
        discrete = state.update_continuous(iteration, of, L_R, gamma)

        # Per-iteration line-search cost (design v2 sec 9.1 F3: the *median
        # per scheduled iteration* obj_eval_count increment, threshold >= 5).
        # Tracked on every callback, not just the `every`-gated evaluator
        # entries below -- a window-cumulative delta would read ~`every` for a
        # perfectly healthy 1 eval/iter run and falsely trip F3.
        n_now = placer.optimizer.param_groups[0]["obj_eval_count"]
        obj_evals = n_now - cb_state["prev_obj_evals"]
        cb_state["prev_obj_evals"] = n_now
        cb_state["obj_evals_per_iter"].append(obj_evals)

        # Step 8: io_gp must reflect the *last* callback's exact io_count; since
        # GP may stop before hitting total_iterations, force one extra evaluator
        # call right at the iteration budget's edge so a run that uses the full
        # budget still gets an up-to-date io_gp (best-effort -- a run that stops
        # earlier via Lgamma_stop_criterion is covered by the periodic `every`
        # evaluator calls below instead).
        force_eval = (iteration == total_iterations - 1)
        if (iteration > 0 and iteration % every == 0) or force_eval:
            node_x = pos.data[:n_phys]
            node_y = pos.data[n_all:n_all + n_phys]
            res = ctx.evaluate(node_x, node_y)
            cb_state["io_gp"] = res.io_count

            entry = {"iteration": iteration, "overflow": of, "tau": state.tau,
                     "lambda_io": state.lambda_io, "obj_evals": obj_evals,
                     "io_count": res.io_count, "ft_count": res.ft_count,
                     "hard_lambda_sum": res.hard_lambda_sum,
                     "soft_lambda_ref_tau": 0.0, "grad_l1_io": 0.0, "grad_l1_wl": 0.0,
                     "frac_soft": 0.0, "grad_share": []}

            if not observer_mode:
                # auto-normalization (design v2 sec 5.2): one extra backward on
                # the WL-only op, L1 norm, taken *before* the io_term backward
                # below so its grad is isolated. Always computed, independent
                # of --diag-every/--no-diag below: state.update_ratio() feeds
                # the schedule's live lambda_io calibration every callback,
                # not just the sampled ones diagnostics() reports on.
                wl = placer.model.op_collections.wirelength_op(pos)
                wl.backward()
                g_wl_l1 = float(pos.grad.abs().sum())
                pos.grad.zero_()
                g_io_l1 = io_term.io_grad_l1(pos.detach(), state.tau)
                state.update_ratio(g_wl_l1, g_io_l1)
                entry["grad_l1_io"] = g_io_l1
                entry["grad_l1_wl"] = g_wl_l1

                if alpha_io > 0:
                    c_e = res.per_net_crossings[csr.net_ids].astype(np.float64)
                    new_w = 1.0 + alpha_io * np.minimum(c_e, cap)
                    io_term.w.copy_(torch.as_tensor(new_w, dtype=io_term.w.dtype,
                                                     device=io_term.w.device))
                    state.obj_version += 1

                # M3 Phase B (adjudication doc
                # docs/results/2026-08-14-m3-s4-adjudication.md sec C item 2,
                # the preregistered fallback route: "M1 式離散 reweight",
                # same w = 1 + alpha*min(signal, cap) family as alpha_io
                # above -- ioplace/reweight.py's update_net_weights, reused
                # directly, not reimplemented) -- but the per-net signal is
                # ft_rg instead of IO crossings: FT_rg_e = ST_e -
                # max(Lambda_e - 1, 0), the per-net analogue of the scalar
                # ft_rg = io_rg - hard_lambda_sum EvalResult already returns
                # (evaluator_gpu.py:594; per-net form already used by
                # probe_p0b.py's compute_before/process_random_direction).
                # ST_e >= Lambda_e - 1 always (a Steiner tree spanning
                # Lambda_e terminals needs >= Lambda_e-1 unit-weight region-
                # graph hops), so FT_rg_e >= 0 -- same non-negative-count
                # precondition update_net_weights' min(., cap) form assumes
                # for per_net_crossings. This is a coefficient update only
                # (no new gradient path, no new persistent state): it writes
                # the same io_term.w buffer the IO term's forward already
                # reads every iteration.
                if ft_reweight == "on":
                    lam_e = res.per_net_lambda[csr.net_ids].astype(np.float64)
                    st_e = res.per_net_steiner[csr.net_ids].astype(np.float64)
                    ft_rg_e = st_e - np.maximum(lam_e - 1.0, 0.0)
                    update_net_weights(io_term.w, ft_rg_e, alpha=alpha_ft, cap=cap)
                    state.obj_version += 1

                # M4 design draft sec 1.4 B2: diagnostics() runs one full
                # chunked fwd+bwd per non-empty degree bucket (io_term.py's
                # own n_backward_passes field) -- expensive at scale (30M:
                # ~213s/callback per the draft). --diag-every N samples this
                # call every N-th `every`-gated occurrence (N=1 default =
                # every occurrence, i.e. unchanged pre-fix behavior);
                # --no-diag drops it entirely. g_io_l1 above is NOT reused
                # here (Codex review of the draft): diagnostics's
                # grad_share is a *bucket-masked* backward at a fixed
                # reference tau (0.05*L_R), while g_io_l1 is the full
                # (unmasked) gradient at the schedule's live state.tau --
                # bucket-masked L1 norms don't reconstruct the unmasked L1
                # norm (cross-bucket gradient contributions at a shared node
                # can cancel), and the two tau values generally differ, so
                # there is no single backward pass both could share.
                do_diag = (not no_diag) and (cb_state["diag_occurrences"] % diag_every == 0)
                cb_state["diag_occurrences"] += 1
                if do_diag:
                    diag = io_term.diagnostics(pos, 0.05 * L_R)
                    entry["soft_lambda_ref_tau"] = diag["soft_lambda_sum"]
                    entry["frac_soft"] = diag["frac_soft"]
                    entry["grad_share"] = diag["grad_share"].tolist()

            trajectory.append(entry)

            # design v3.1 T0-a: trajectory-snapshot instrumentation for P0b
            # (sec 3.4.1(a)). Everything below operates on an independent
            # detached+cloned leaf tensor, never the live `pos`/`pos.grad` --
            # this is what makes the bit-exact invariant (a run with
            # --snapshot-iters must match a run without it) hold regardless
            # of what this block does.
            if snapshot_iters and iteration in snapshot_iters:
                p_wld = pos.detach().clone().requires_grad_(True)
                g_wl_density = _raw_wl_density_grad(placer.model, p_wld)
                tau_rel = state.tau / L_R
                np.savez_compressed(
                    os.path.join(snapshot_dir, f"it{iteration:04d}.npz"),
                    iteration=iteration, overflow=of, tau=state.tau, tau_rel=tau_rel,
                    gamma=gamma, density_weight=float(placer.model.density_weight),
                    ratio_ema=(state.ratio_ema if state.ratio_ema is not None
                              else float("nan")),
                    lambda_io=state.lambda_io,
                    node_x=pos.data[:n_all].detach().cpu().numpy(),
                    node_y=pos.data[n_all:2 * n_all].detach().cpu().numpy(),
                    g_wl_density=g_wl_density.cpu().numpy().astype(np.float32))
                if snapshot_grad_check_cb is not None:
                    snapshot_grad_check_cb(iteration, pos, placer, io_term, state,
                                           g_wl_density)

        # 順序不可換:先讓新 τ/λ/w 生效,再 refresh。
        if not observer_mode and (discrete or state.needs_refresh()):
            refresh_nesterov_secant(placer.optimizer)
            state.mark_refreshed()
            cb_state["num_refreshes"] += 1

        if check_invariant and not cb_state["installed_invariant"]:
            install_version_invariant(placer.optimizer, state)
            cb_state["installed_invariant"] = True

    placer.iteration_callback = cb
    lr = params.global_place_stages[0]["learning_rate"]
    placer(params, placedb, lr)
    if "gp" not in timer.phases:
        # params.legalize_flag was off -- _timed_legalize (and so gp_phase's
        # own close) never fired.
        gp_phase.__exit__(None, None, None)
    lambda_io_final = state.lambda_io
    final_overflow = float(placer.model.overflow.max())

    with timer.phase("eval"):
        node_x, node_y = extract_final_positions(placer, placedb)
        legal_fields = _legalization_diagnostics(placer, placedb, params, node_x, node_y)
        _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
        res = ctx.evaluate(node_x, node_y)
        m = res.per_net_crossings > 0
        if np.count_nonzero(m) >= 2:
            spearman_rho = float(scipy.stats.spearmanr(
                res.per_net_crossings[m], (res.per_net_lambda - 1)[m]).correlation)
        else:
            spearman_rho = float("nan")

        detach_terms(params)

    sampler.stop()

    backtrack_median = float(np.median(cb_state["obj_evals_per_iter"])) \
        if cb_state["obj_evals_per_iter"] else 0.0

    result = {
        "mode": "io", "config": config_json, "k": k, "rtype": rtype, "seed": seed,
        "dp_seed": int(params.random_seed), "det": int(params.deterministic_flag),
        "io_count": metrics["io_count"], "io_gp": cb_state["io_gp"],
        "ft_count": metrics["ft_count"], "hard_lambda_sum": res.hard_lambda_sum,
        "tree_wl": metrics["tree_wl"], "hpwl": metrics["hpwl"],
        # T1 region-graph fields (evaluator_gpu.py, commit 0a2e32c) off the
        # same final-position `res` evaluate() call already used for
        # spearman_rho above -- no extra evaluator pass.
        "io_rg": res.io_rg, "ft_rg": res.ft_rg,
        "lg_loss": metrics["io_count"] - cb_state["io_gp"],
        "runtime_s": time.time() - t0,
        # sec 1.4 B1: see run_flat's matching field for what this does/does
        # not guarantee (per-run reset, not full process isolation).
        # peak_mem_mb itself is now supplied by _phase_summary() below (see
        # run_placement._phase_summary's docstring for why).
        "peak_mem_mb_reset_semantics": True,
        "rho_max": rho_max, "tau_hi": tau_hi, "tau_lo": tau_lo,
        "of_on": of_on, "of_end": of_end, "alpha_io": alpha_io,
        "w_mode": w_mode, "d_max": ignore_net_degree, "rho_margin": rho_margin,
        "margin_m": margin_m, "lambda_io_final": lambda_io_final,
        "spearman_rho": spearman_rho, "num_callbacks": len(trajectory),
        "num_refreshes": cb_state["num_refreshes"],
        "backtrack_median": backtrack_median, "observer_mode": observer_mode,
        "diag_every": diag_every, "no_diag": no_diag,
        "ft_reweight": ft_reweight, "alpha_ft": alpha_ft,
        "trajectory": trajectory,
        "final_overflow": final_overflow,
        **legal_fields,
        **_phase_summary(timer, sampler), **_t8a_provenance(config_json),
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result
