"""Driver for `--mode io`.

v2 P-H: coefficient derivation for the extra objective terms is routed
through `ioplace.norm.TermNormalizer` (see `run_io`'s `norm_policy` family of
keyword arguments). `--norm-policy legacy` (the default) is a thin adapter
over the retired `ops/ft_callback.publish_atomic` + `ScheduleState` path and
reproduces its coefficients bit-for-bit; `grandplan`/`adaptive` compute their
own coefficients from `ioplace.ops.norm_terms.IoNormTerm`/`FtNormTerm`
gradient probes. `IoNormTerm` measures the *unweighted* IO term only, with no
margin contribution -- the same quantity `ops/ft_callback.publish_atomic`
isolates for its own `kappa_ft` derivation, so the two paths stay comparable.
"""
import json, os, time
from contextlib import ExitStack, contextmanager
import numpy as np
import scipy.stats
from ioplace.drivers.run_placement import (_load_dreamplace, extract_final_positions,
    _pack_eval_metrics, get_regions_for, _legalization_diagnostics, _phase_summary,
    _t8a_provenance, _stop_overflow_reached, _gp_iteration_budget, _effective_scale_fields)
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.ops.soft_assign import rect_table
from ioplace.ops.io_term import build_net_node_csr, IoTerm
from ioplace.profile import PhaseTimer, DeviceMemSampler
from ioplace.profile_lifetime import LifetimeRecorder
from ioplace.schedules import ScheduleState
from ioplace.dp_hook import (attach_terms, detach_terms, assert_optimizer_lock,
                             refresh_nesterov_secant, install_version_invariant)
from ioplace.reweight import update_net_weights
from ioplace.export.def_export import export_def

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
                 "device_used_gb", "host_peak_rss_gb",
                 # Overflow-diagnosis follow-up (schema_version 3): E1/RESULT
                 # GATE gaps closed after the T8a overflow-diagnosis adjudication.
                 "stop_overflow_reached", "gp_iteration_budget", "gp_iterations_run",
                 "hpwl_gp", "hpwl_lg", "effective_target_density",
                 "num_filler_nodes", "num_bins_x", "num_bins_y",
                 "command", "hostname", "benchmark_kind", "device_baseline_gb",
                 # M4 T2b (probe_lifetime_gate.py's artifact postcondition):
                 "n_callbacks_with_active", "n_evals_while_active",
                 # v2 P-H (design sec 4): normalisation module configuration
                 # and artefact.
                 "norm_policy", "norm_p", "norm_ramp_period", "norm_wt_max",
                 "norm_probe_every", "norm_target_share", "norm_trace",
                 "lambda_ft_final")


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


@contextmanager
def _io_cleanup():
    """Close every run-owned resource while retaining an original failure."""
    cleanup = ExitStack()
    try:
        yield cleanup
    except BaseException as error:
        try:
            cleanup.close()
        except BaseException as cleanup_error:
            error.add_note(f"IO resource cleanup also failed: {cleanup_error!r}")
        raise
    else:
        cleanup.close()


def _cleanup_once(cleanup, callback):
    done = False
    def close():
        nonlocal done
        if not done:
            callback()
            done = True
    cleanup.callback(close)
    return close


def _install_attribute(cleanup, obj, name, value):
    missing = object()
    original = getattr(obj, name, missing)
    def restore():
        if original is missing:
            delattr(obj, name)
        else:
            setattr(obj, name, original)
    cleanup.callback(restore)
    setattr(obj, name, value)


def run_io(config_json, k, rtype, seed, out_json, *,
           rho_max=0.1, tau_hi=0.30, tau_lo=0.03,
           of_on=0.90, of_end=None, of_full=0.20,
           alpha_io=0.0, cap=64.0, ignore_net_degree=None,
           rho_margin=0.0, margin_m=None, margin_tau=None,
           of_margin=0.15, w_mode="unit", every=50,
           dp_seed=None, deterministic=None, check_invariant=False,
           diag_every=1, no_diag=False,
           ft_reweight="off", alpha_ft=0.5,
           snapshot_iters=None, snapshot_dir=None, snapshot_grad_check_cb=None,
           emit_def=None,
           emit_eval=None,
           benchmark_kind="real",
           lifetime_out=None, callback_order="legacy", f_ft_max=0.,
           ft_ramp_mode="window", tau_start=.12, tau_full=.05,
           home_period=None, topology_diagnostics=False,
           wl_reweight="off", alpha_wl=.2, wl_cap=10.,
           discrete_mode="none", discrete_max_active=65536,
           norm_policy="legacy", norm_p=1, norm_ramp_period=100,
           norm_wt_max=1.0, norm_probe_every=50, norm_target_share=None,
           norm_trace=None):
    if discrete_mode not in ("none","ce","refine","ce_refine") or discrete_max_active<0:
        raise ValueError("invalid discrete postprocess configuration")
    if callback_order not in ("legacy", "atomic"):
        raise ValueError("callback_order must be legacy or atomic")
    if wl_reweight not in ("off", "crossings", "ft_rg"):
        raise ValueError("wl_reweight must be off, crossings or ft_rg")
    if f_ft_max < 0 or (f_ft_max > 0 and (callback_order != "atomic" or rho_max <= 0)):
        raise ValueError("positive FT requires atomic callbacks and enabled IO")
    if wl_reweight != "off" and callback_order != "atomic":
        raise ValueError("WL reweight requires atomic callbacks")
    if f_ft_max > 0 and (alpha_io > 0 or ft_reweight != "off" or wl_reweight != "off" or rho_margin > 0):
        raise ValueError("P0c FT excludes reweighting and margin objectives")
    home_period = every if home_period is None else home_period
    if every <= 0 or home_period <= 0 or home_period % every:
        raise ValueError("home_period must be a positive multiple of every")
    if norm_policy not in ("legacy", "grandplan", "adaptive"):
        raise ValueError("norm_policy must be legacy, grandplan or adaptive, "
                         "got %r" % (norm_policy,))
    if norm_p not in (1, 2):
        raise ValueError("norm_p must be 1 or 2, got %r" % (norm_p,))
    if norm_policy == "legacy" and norm_p != 1:
        raise ValueError("norm_policy 'legacy' measures L1 gradients inside "
                         "publish_atomic; norm_p must be 1, got %r" % (norm_p,))
    if norm_probe_every <= 0:
        raise ValueError("norm_probe_every must be a positive multiple of every")
    if norm_policy != "legacy" and norm_probe_every % every:
        raise ValueError("norm_probe_every must be a positive multiple of every")
    if norm_policy != "legacy" and callback_order != "atomic":
        raise ValueError("norm_policy %r requires callback_order='atomic'"
                         % (norm_policy,))
    if norm_policy != "legacy" and (rho_max == 0.0 and rho_margin == 0.0
                                    and wl_reweight == "off"):
        raise ValueError("norm_policy %r does nothing in observer mode "
                         "(rho_max=0, rho_margin=0, wl_reweight=off)"
                         % (norm_policy,))
    import torch
    # Overflow-diagnosis follow-up: device_baseline_gb -- see
    # run_placement.run_flat's matching comment for why this must be the
    # first CUDA call in the function, before PhaseTimer/DeviceMemSampler
    # or _load_dreamplace touch CUDA themselves.
    if torch.cuda.is_available():
        free0, total0 = torch.cuda.mem_get_info()
        device_baseline_gb = (total0 - free0) / 2**30
    else:
        device_baseline_gb = 0.0

    t0 = time.time()
    # M4 T2b (design draft sec 2.2/7.1 T2b): when lifetime_out is given, a
    # LifetimeRecorder takes over full-lifetime buffer tracking + per-phase
    # GPU-peak accounting (sec 2.2's corrected peak ~= resident + max
    # transient formula) for this run -- PhaseTimer(reset_peak=False) below
    # makes the ordinary phase timer skip its own reset_peak_memory_stats()
    # calls so the two trackers never fight over the same process-wide CUDA
    # counter (see profile.py's PhaseTimer/_Phase docstrings). Every rec.*
    # call in this function is behind `if rec is not None:` -- with
    # lifetime_out=None (every existing caller), rec is None, this whole
    # block is a no-op, and the rest of this function's control flow,
    # tensor ops, and RNG draws are byte-for-byte what they were before T2b.
    with _io_cleanup() as cleanup:
        rec = LifetimeRecorder() if lifetime_out is not None else None
        if rec is not None:
            stop_rec = _cleanup_once(cleanup, rec.stop)
            rec.mark("cuda_baseline", scan=False)   # no roots registered yet
        # M4 T8a: replaces the old single reset_peak_memory_stats() call (sec
        # 1.4 B1's fix) with PhaseTimer's per-phase reset -- see
        # run_placement._phase_summary's docstring for why peak_mem_mb is
        # recomputed from the per-phase peaks rather than a single end-of-run
        # read.
        timer = PhaseTimer(reset_peak=(rec is None))
        sampler = DeviceMemSampler()
        stop_sampler = _cleanup_once(cleanup, sampler.stop)
        sampler.start()

        if rec is not None:
            rec.phase_begin("read")
        with timer.phase("read"):
            params, placedb = _load_dreamplace(config_json)
            cleanup.callback(detach_terms, params)
            if discrete_mode!="none" and not params.legalize_flag:
                raise ValueError("discrete postprocess requires a legal GP+LG incumbent")
            if dp_seed is not None:
                params.random_seed = dp_seed
            if deterministic is not None:
                params.deterministic_flag = deterministic
            placedb.initialize(params)
            # Overflow-diagnosis follow-up: see run_placement._effective_scale_fields's
            # docstring for why this must be read after initialize().
            scale_fields = _effective_scale_fields(params, placedb)
        if rec is not None:
            rec.phase_end("read")
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
        if rec is not None:
            rec.mark("netlist_built", scan=False)    # still no roots registered
            rec.phase_begin("initialize")
        die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
        rs = get_regions_for(die, k, rtype, seed)
        rg = RegionGrid(rs)
        ctx = GpuEvalContext(nl, rg, device="cuda")
        if rec is not None:
            rec.add_root("eval_ctx", ctx)
            rec.phase_end("initialize")

        if of_end is None:
            of_end = float(params.stop_overflow)
        if ignore_net_degree is None:
            ignore_net_degree = int(params.ignore_net_degree)
        if margin_m is None:
            margin_m = 2.0 * float(np.median(nl.node_size_y[:nl.num_movable]))
        if margin_tau is None:
            margin_tau = margin_m / 2.0

        if rec is not None:
            rec.phase_begin("io_term_build")
        rects, r2k = rect_table(rs)
        csr = build_net_node_csr(nl, ignore_net_degree)
        io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                         num_movable=nl.num_movable, num_physical=nl.num_physical,
                         num_nodes=placedb.num_nodes, device="cuda", w_mode=w_mode)
        ft_term = None
        distance = None
        if f_ft_max > 0 or topology_diagnostics:
            from ioplace.region_graph import region_graph
            _, distance, _ = region_graph(rg)
        if f_ft_max > 0:
            from ioplace.ops.ft_term import FtTerm
            ft_term = FtTerm(io_term, distance)
        if rec is not None:
            rec.add_root("io_term", io_term)
            rec.phase_end("io_term_build")

        L_R = ((die[2] - die[0]) * (die[3] - die[1]) / k) ** 0.5

        state = ScheduleState(rho_max=rho_max, tau_hi=tau_hi, tau_lo=tau_lo,
                              of_on=of_on, of_end=of_end, of_full=of_full,
                              alpha_io=alpha_io, rho_margin=rho_margin,
                              margin_m=margin_m, margin_tau=margin_tau, of_margin=of_margin,
                              f_ft_max=f_ft_max, ft_ramp_mode=ft_ramp_mode,
                              tau_start=tau_start, tau_full=tau_full)

        # design v2 sec 5.2/6.4: rho_max==0 and rho_margin==0 is a *provable* no-op
        # -- the term is never attached, so the objective is bit-identical to
        # run_flat. T6's flat baseline runs through this path to get the
        # io_gp/lg_loss/hard_lambda_sum columns it needs.
        observer_mode = (rho_max == 0.0 and rho_margin == 0.0 and wl_reweight == "off")

        from ioplace.norm import (TermNormalizer, VersionPair, _json_cap,
                                 parse_target_shares)
        from ioplace.norm_trace import NormTraceWriter
        from ioplace.ops.norm_terms import FtNormTerm, IoNormTerm
        shares = parse_target_shares(norm_target_share)
        norm_trace_path = norm_trace
        if norm_trace_path is None and norm_policy != "legacy" and not observer_mode:
            norm_trace_path = out_json + ".norm_trace.jsonl"
        trace_writer = None
        if norm_trace_path is not None and not observer_mode:
            trace_writer = NormTraceWriter(norm_trace_path)
            cleanup.callback(trace_writer.close)
        normalizer = TermNormalizer(
            policy=norm_policy, norm_p=norm_p, ema=state.ema,
            probe_every=norm_probe_every, ramp_period=norm_ramp_period,
            wt_max=norm_wt_max, c_lip=state.c_lip,
            num_movable=io_term.num_movable, num_nodes=io_term.num_nodes,
            # One cached gradient tensor per term is ~8 B/node/term; above 5M
            # nodes the cancellation diagnostic is not worth the residency.
            track_cancellation=(placedb.num_nodes <= 5_000_000),
            legacy_state=state if norm_policy == "legacy" else None,
            trace=trace_writer)
        normalizer.register("io", IoNormTerm(io_term), 1.0,
                            target_share=shares.get("io", 0.3),
                            activate_overflow=of_on, n_ramp=state.n_ramp)
        if ft_term is not None:
            # FT curvature is ecc_max (design sec 4). Its overflow gate is 0.30:
            # the tau_rel window the legacy ramp used (0.12 -> 0.05) maps through
            # tau_rel_from_overflow to overflow 0.57 -> 0.25, and 0.30 is the
            # same threshold the capacity term uses (design sec 5).
            normalizer.register("ft", FtNormTerm(ft_term), float(distance.max()),
                                target_share=shares.get("ft", f_ft_max),
                                activate_overflow=0.30, n_ramp=state.n_ramp)

        def term_fn(pos):
            if norm_policy == "legacy":
                # Pass state.kappa_ft verbatim: recovering it as lambda_ft/lambda_io
                # would differ in the last ulp and break the legacy guarantee.
                lam_io, kappa = state.lambda_io, state.kappa_ft
            else:
                lam_io = normalizer.lambdas.get("io", 0.0)
                lam_ft = normalizer.lambdas.get("ft", 0.0)
                # Defensive only: the callback already rejects this combination
                # right after normalizer.transaction() (with iteration/overflow
                # context for diagnosability, fix round 1) -- this should be
                # unreachable by the time term_fn runs.
                assert not (lam_io <= 0.0 and lam_ft > 0.0), (
                    "FtTerm expresses the FT coefficient as lambda_io*kappa, so "
                    "a nonzero lambda_ft with lambda_io == 0 cannot be applied")
                # Non-legacy: this path has no bit-exact guarantee to preserve
                # (unlike legacy above), so recovering kappa by division here,
                # accepting last-ulp rounding against the traced lambda_ft, is fine.
                kappa = lam_ft / lam_io if lam_io > 0.0 else 0.0
            if not state.active or (lam_io == 0.0 and state.lambda_margin == 0.0):
                return pos.new_zeros(())
            if ft_term is not None:
                return ft_term(pos, state.tau, lam_io, kappa,
                               state.lambda_margin, state.margin_m, state.margin_tau)
            return io_term(pos, state.tau, lam_io, state.lambda_margin,
                           state.margin_m, state.margin_tau)

        if not observer_mode:
            attach_terms(params, [term_fn])   # must precede NonLinearPlace(...) construction

        # Same init_pos determinism guard as run_placement._place() / run_reweight:
        # BasicPlace draws centre-noise/filler init from numpy's global RNG, seeded
        # only by Placer.py's flow which we bypass here.
        if rec is not None:
            rec.phase_begin("placer_build")
        np.random.seed(params.random_seed)
        placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
        if rec is not None:
            rec.add_root("placer", placer)
            rec.phase_end("placer_build")
            rec.phase_begin("gp")

        # M4 T8a: same op_collections.legalize_op timing wrap as
        # run_placement.run_flat -- see that function's comment for why this is
        # the only non-invasive way (no DREAMPlace source touched) to observe
        # the GP/LG boundary. Overflow-diagnosis follow-up: also grabs
        # hpwl_gp/hpwl_lg the same way run_flat's wrapper does (see that
        # wrapper's comment for why the torch.no_grad() is needed -- `pos` is
        # the live optimizer's requires_grad=True leaf tensor).
        orig_legalize = placer.op_collections.legalize_op
        hpwl_holder = {}
        def _timed_legalize(pos):
            gp_phase.__exit__(None, None, None)
            with torch.no_grad():
                hpwl_holder["hpwl_gp"] = float(placer.op_collections.hpwl_op(pos))
            with timer.phase("lg"):
                out = orig_legalize(pos)
            with torch.no_grad():
                hpwl_holder["hpwl_lg"] = float(placer.op_collections.hpwl_op(out))
            return out
        _install_attribute(cleanup, placer.op_collections, "legalize_op", _timed_legalize)

        n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
        total_iterations = params.global_place_stages[0]["iteration"]
        gp_iteration_budget = _gp_iteration_budget(params)

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
        previous_topology = None
        topology_net_conversion = 0
        previous_home = None
        cb_state = {"num_refreshes": 0, "io_gp": 0, "prev_obj_evals": 0,
                   "obj_evals_per_iter": [], "installed_invariant": False,
                   "diag_occurrences": 0, "last_iteration": -1,
                   # M4 T2b (probe_lifetime_gate.py's artifact postcondition):
                   # counts, not booleans -- "the IO term was actually active
                   # (state.active, i.e. past the of_on overflow threshold) for
                   # >=3 callbacks/evaluator calls", the guard against a
                   # lifetime probe that measured memory before the op it's
                   # trying to characterize ever turned on.
                   "n_callbacks_with_active": 0, "n_evals_while_active": 0}

        def cb(iteration, pos):
            nonlocal previous_topology, topology_net_conversion, previous_home
            # Overflow-diagnosis follow-up: gp_iterations_run source for run_io
            # (unlike run_flat, run_io never captures NonLinearPlace.__call__'s
            # own metrics return value -- see run_placement._last_metric_iteration's
            # docstring for that path). Tracked on every callback, not just the
            # `every`-gated block below, since iteration_callback fires every GP
            # iteration (NonLinearPlace.py:521) and a run stopped early by
            # Lgamma_stop_criterion must still report the iteration it actually
            # stopped at.
            cb_state["last_iteration"] = iteration
            of = float(placer.model.overflow.max())
            gamma = float(placer.model.gamma)
            discrete = state.update_continuous(iteration, of, L_R, gamma)
            if not observer_mode and state.active:
                cb_state["n_callbacks_with_active"] += 1

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
                # M4 T2b: fine-grained buffer-lifetime checkpoints within the
                # single long-open "gp" phase -- only at the (small) set of
                # scan_iters iterations (default {0,1,2,-1}), not every
                # `every`-gated callback, since a full resident scan at 30M
                # scale is not free. Guarded by `rec is not None` exactly like
                # every other T2b call site.
                do_scan = rec is not None and rec.wants_iter(iteration, force_eval)
                if do_scan:
                    rec.mark(f"gp_iter_{iteration}")
                node_x = pos.data[:n_phys]
                node_y = pos.data[n_all:n_all + n_phys]
                res = ctx.evaluate(node_x, node_y)
                if do_scan:
                    rec.mark(f"after_ctx_evaluate_{iteration}")
                if not observer_mode and state.active:
                    cb_state["n_evals_while_active"] += 1
                cb_state["io_gp"] = res.io_count

                entry = {"iteration": iteration, "overflow": of, "tau": state.tau,
                         "lambda_io": state.lambda_io, "obj_evals": obj_evals,
                         "io_count": res.io_count, "ft_count": res.ft_count,
                         "hard_lambda_sum": res.hard_lambda_sum,
                         "soft_lambda_ref_tau": 0.0, "grad_l1_io": 0.0, "grad_l1_wl": 0.0,
                         "frac_soft": 0.0, "grad_share": []}

                if not observer_mode and callback_order == "atomic":
                    if wl_reweight != "off":
                        signal = (res.per_net_crossings if wl_reweight == "crossings" else
                                  res.per_net_steiner - np.maximum(res.per_net_lambda - 1, 0))
                        update_net_weights(placer.data_collections.net_weights, signal,
                                           alpha=alpha_wl, cap=wl_cap)
                    if alpha_io > 0:
                        update_net_weights(io_term.w, res.per_net_crossings[csr.net_ids],
                                           alpha=alpha_io, cap=cap)
                    if ft_reweight == "on":
                        signal = res.per_net_steiner - np.maximum(res.per_net_lambda - 1, 0)
                        update_net_weights(io_term.w, signal[csr.net_ids], alpha=alpha_ft, cap=cap)
                    if ft_term is not None and (ft_term.home is None or iteration % home_period == 0):
                        homes = res.per_net_home[csr.net_ids]
                        entry["home_churn"] = (float(np.mean(homes != previous_home))
                                                if previous_home is not None and len(homes) else None)
                        ft_term.set_home(homes)
                        previous_home = homes.copy()
                    wirelength_op = placer.model.op_collections.wirelength_op
                    if norm_policy == "legacy":
                        from ioplace.ops.ft_callback import publish_atomic
                        txn = normalizer.transaction(
                            iteration, of, state.tau, gamma,
                            legacy_publish=lambda: publish_atomic(
                                state, io_term, ft_term, wirelength_op, pos,
                                iteration, state.tau / L_R,
                                float(distance.max()) if distance is not None else 0.,
                                gamma))
                        entry.update(txn.legacy_record)
                    else:
                        ctx_norm = {"iteration": iteration, "overflow": of,
                                    "tau": state.tau, "gamma": gamma}
                        if normalizer.should_probe(iteration):
                            normalizer.probe(iteration, pos, wirelength_op, ctx_norm)
                        txn = normalizer.transaction(iteration, of, state.tau, gamma)
                        lam_io_txn = txn.lambdas.get("io", 0.0)
                        lam_ft_txn = txn.lambdas.get("ft", 0.0)
                        if lam_io_txn <= 0.0 and lam_ft_txn > 0.0:
                            raise RuntimeError(
                                "FtTerm expresses the FT coefficient as lambda_io*kappa, "
                                "so a nonzero lambda_ft with lambda_io == 0 cannot be "
                                "applied (iteration=%d, overflow=%.6f)" % (iteration, of))
                        ft_state = normalizer.states.get("ft")
                        entry.update(grad_l1_wl=normalizer.wl_norm,
                                     grad_l1_io=normalizer.states["io"].grad_norm,
                                     grad_l1_ft=ft_state.grad_norm if ft_state else 0.,
                                     ratio_inst=normalizer.states["io"].ratio_inst,
                                     ratio_ema=normalizer.states["io"].ratio_ema,
                                     lambda_io=txn.lambdas.get("io", 0.),
                                     obj_version=txn.obj_version)
                    entry.update(lambda_ft=txn.lambdas.get("ft", 0.),
                                 norm_cmax=txn.cmax,
                                 norm_cap=_json_cap(txn.cap),
                                 cap_binding=txn.cap_binding,
                                 cancellation_ratio=txn.cancellation_ratio)
                    weights = placer.data_collections.net_weights
                    entry["wl_weights_min"] = float(weights.min())
                    entry["wl_weights_max"] = float(weights.max())
                elif not observer_mode:
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
                    if do_scan:
                        rec.mark(f"after_wl_io_grad_{iteration}")
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
                        if do_scan:
                            rec.mark(f"after_diagnostics_{iteration}")
                        entry["soft_lambda_ref_tau"] = diag["soft_lambda_sum"]
                        entry["frac_soft"] = diag["frac_soft"]
                        entry["grad_share"] = diag["grad_share"].tolist()

                if topology_diagnostics:
                    from ioplace.diagnostics.probes_m3.probe_p0b import (
                        _touched_lists, classify_topology, transition_matrix, TOPO_IDX)
                    touched = _touched_lists(rg, nl, node_x.detach().cpu().numpy(),
                                              node_y.detach().cpu().numpy(), csr.net_ids, k)
                    lam = res.per_net_lambda[csr.net_ids]
                    ft_rg = res.per_net_steiner[csr.net_ids] - np.maximum(lam - 1, 0)
                    current = classify_topology(lam, ft_rg, touched,
                                                res.per_net_home[csr.net_ids], distance)
                    if previous_topology is not None:
                        transitions = transition_matrix(previous_topology, current)
                        topology_net_conversion += (transitions[TOPO_IDX["chain"]][TOPO_IDX["star"]]
                                                    - transitions[TOPO_IDX["star"]][TOPO_IDX["chain"]])
                        entry["topology_transitions"] = transitions
                    entry["topology_population"] = len(current)
                    entry["net_chain_to_star"] = topology_net_conversion
                    previous_topology = current
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
                    # Fix round 1: under a non-legacy policy the live coefficient
                    # lives on the normalizer (state.lambda_io/ratio_ema are stale
                    # there -- term_fn never writes them back), so pull the
                    # snapshot's own lambda_io/ratio_ema from the same source
                    # term_fn actually reads.
                    if norm_policy == "legacy":
                        snap_lambda_io, snap_ratio_ema = state.lambda_io, state.ratio_ema
                    else:
                        snap_lambda_io = normalizer.lambdas.get("io", 0.0)
                        snap_ratio_ema = normalizer.states["io"].ratio_ema
                    np.savez_compressed(
                        os.path.join(snapshot_dir, f"it{iteration:04d}.npz"),
                        iteration=iteration, overflow=of, tau=state.tau, tau_rel=tau_rel,
                        gamma=gamma, density_weight=float(placer.model.density_weight),
                        ratio_ema=(snap_ratio_ema if snap_ratio_ema is not None
                                  else float("nan")),
                        lambda_io=snap_lambda_io,
                        node_x=pos.data[:n_all].detach().cpu().numpy(),
                        node_y=pos.data[n_all:2 * n_all].detach().cpu().numpy(),
                        g_wl_density=g_wl_density.cpu().numpy().astype(np.float32))
                    if snapshot_grad_check_cb is not None:
                        snapshot_grad_check_cb(iteration, pos, placer, io_term, state,
                                               g_wl_density)

            # 順序不可換:先讓新 τ/λ/w 生效,再 refresh。
            if not observer_mode and (discrete or state.needs_refresh()
                                      or normalizer.needs_refresh()):
                refresh_nesterov_secant(placer.optimizer)
                state.mark_refreshed()
                normalizer.mark_refreshed()      # also emits the pending trace row
                cb_state["num_refreshes"] += 1
            if callback_order == "atomic" and trajectory and trajectory[-1]["iteration"] == iteration:
                trajectory[-1]["refreshed_version"] = (
                    state.refreshed_version if norm_policy == "legacy"
                    else normalizer.refreshed_version)

            if check_invariant and not cb_state["installed_invariant"]:
                # Under policy="legacy" the normalizer's obj_version/
                # refreshed_version delegate to `state`, so this VersionPair
                # sums the same counter twice; equality still holds iff both
                # are refreshed, so the invariant stays sound.
                cleanup.callback(install_version_invariant(
                    placer.optimizer, VersionPair(state, normalizer)))
                cb_state["installed_invariant"] = True

        _install_attribute(cleanup, placer, "iteration_callback", cb)
        lr = params.global_place_stages[0]["learning_rate"]
        placer(params, placedb, lr)
        if "gp" not in timer.phases:
            # params.legalize_flag was off -- _timed_legalize (and so gp_phase's
            # own close) never fired.
            gp_phase.__exit__(None, None, None)
        if rec is not None:
            # LifetimeRecorder's own "gp" phase spans the whole placer(...) call
            # (including any mid-call legalize sub-call) -- unlike PhaseTimer's
            # "gp"/"lg" split (which _timed_legalize closes/reopens mid-call to
            # observe the GP/LG wall-clock boundary), the recorder has no need
            # to split them: it isn't measuring wall time, and its fine-grained
            # per-iteration signal comes from the scan_iters mark() checkpoints
            # inside cb(), not from a phase boundary.
            rec.phase_end("gp")
        lambda_io_final = (state.lambda_io if norm_policy == "legacy"
                           else normalizer.lambdas.get("io", 0.0))
        lambda_ft_final = (state.lambda_io * state.kappa_ft if norm_policy == "legacy"
                           else normalizer.lambdas.get("ft", 0.0))
        final_overflow = float(placer.model.overflow.max())
        stop_overflow_reached = _stop_overflow_reached(final_overflow, params.stop_overflow)
        gp_iterations_run = cb_state["last_iteration"] + 1

        discrete_result=None
        if discrete_mode!="none":
            from ioplace.ops.discrete_postprocess import run_discrete_postprocess
            with timer.phase("discrete"):
                discrete_result,discrete_arrays=run_discrete_postprocess(nl,rs,placer,placedb,params,
                    orig_legalize,ctx,mode=discrete_mode,max_active=discrete_max_active)
                os.makedirs(os.path.dirname(out_json) or ".",exist_ok=True)
                sidecar=out_json+".discrete.npz"
                np.savez_compressed(sidecar,**discrete_arrays)
                import hashlib
                with open(sidecar,"rb") as stream:
                    discrete_result["sidecar_sha256"]=hashlib.sha256(stream.read()).hexdigest()
                discrete_result["sidecar"]=os.path.abspath(sidecar)

        if rec is not None:
            rec.phase_begin("eval")
        with timer.phase("eval"):
            node_x, node_y = extract_final_positions(placer, placedb)
            legal_fields = _legalization_diagnostics(placer, placedb, params, node_x, node_y)
            res = ctx.evaluate(node_x, node_y)
            metrics = _pack_eval_metrics(res)
            metrics.update(evaluation_backend="gpu", cpu_reference_full_evaluation=False)
            m = res.per_net_crossings > 0
            if np.count_nonzero(m) >= 2:
                spearman_rho = float(scipy.stats.spearmanr(
                    res.per_net_crossings[m], (res.per_net_lambda - 1)[m]).correlation)
            else:
                spearman_rho = float("nan")

            # Stage 2 S1 (spec sec 5.1/10): sidecar DEF export of the final GP+LG
            # placement, off by default (emit_def=None) -- same opt-in shape as
            # snapshot_iters/snapshot_dir above. `rs` is the RegionSet already
            # built for this run (get_regions_for above); export_def threads it
            # through untouched into regions.json.
            if emit_def is not None:
                export_def(placedb, params, node_x, node_y, emit_def, rs)

            # Route calibration needs full net-aligned evidence at the exact
            # placement exported above. Reuse the final evaluator result.
            if emit_eval is None and emit_def is not None:
                emit_eval = os.path.join(emit_def, f"evaluator_k{k}.npz")
            if emit_eval is not None:
                from ioplace.export.evaluation import save_evaluation
                save_evaluation(emit_eval, netlist_from_placedb(placedb), rg,
                                res, node_x, node_y, placedb.net_names,
                                provenance={"config": os.path.abspath(config_json),
                                            "placement_stage": "gp_lg",
                                            "def_directory": os.path.abspath(emit_def) if emit_def else None})

            detach_terms(params)
            if rec is not None:
                rec.mark("teardown")

        if rec is not None:
            rec.phase_end("eval")
        stop_sampler()
        if rec is not None:
            stop_rec()
            lifetime_record = rec.to_record()
            os.makedirs(os.path.dirname(lifetime_out) or ".", exist_ok=True)
            with open(lifetime_out, "w") as f:
                json.dump(lifetime_record, f, indent=1)

        backtrack_median = float(np.median(cb_state["obj_evals_per_iter"])) \
            if cb_state["obj_evals_per_iter"] else 0.0

        result = {
            **metrics,
            "mode": "io", "config": config_json, "k": k, "rtype": rtype, "seed": seed,
            "workload_status": "completed",
            "generator_verified": False if benchmark_kind == "synthetic" else None,
            "evaluator_file": os.path.abspath(emit_eval) if emit_eval else None,
            "dp_seed": int(params.random_seed), "det": int(params.deterministic_flag),
            "io_gp": cb_state["io_gp"],
            "lg_loss": (discrete_result["baseline"]["io_count"] if discrete_result is not None
                        else metrics["io_count"]) - cb_state["io_gp"],
            "discrete_delta_io": (metrics["io_count"]-discrete_result["baseline"]["io_count"]
                                  if discrete_result is not None else None),
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
            "lambda_ft_final": lambda_ft_final,
            "norm_policy": norm_policy, "norm_p": norm_p,
            "norm_ramp_period": norm_ramp_period, "norm_wt_max": norm_wt_max,
            "norm_probe_every": norm_probe_every,
            "norm_target_share": norm_target_share,
            "norm_trace": (os.path.abspath(norm_trace_path)
                           if norm_trace_path is not None and not observer_mode
                           else None),
            "spearman_rho": spearman_rho, "num_callbacks": len(trajectory),
            "num_refreshes": cb_state["num_refreshes"],
            "backtrack_median": backtrack_median, "observer_mode": observer_mode,
            "diag_every": diag_every, "no_diag": no_diag,
            "ft_reweight": ft_reweight, "alpha_ft": alpha_ft,
            "callback_order": callback_order, "f_ft_max": f_ft_max,
            "ft_ramp_mode": ft_ramp_mode, "tau_start": tau_start, "tau_full": tau_full,
            "home_period": home_period, "topology_diagnostics": topology_diagnostics,
            "net_chain_to_star": topology_net_conversion if topology_diagnostics else None,
            "wl_reweight": wl_reweight, "alpha_wl": alpha_wl, "wl_cap": wl_cap,
            "discrete_mode": discrete_mode, "discrete_result": discrete_result,
            "trajectory": trajectory,
            "final_overflow": final_overflow,
            "stop_overflow_reached": stop_overflow_reached,
            "gp_iteration_budget": gp_iteration_budget,
            "gp_iterations_run": gp_iterations_run,
            "hpwl_gp": hpwl_holder.get("hpwl_gp"), "hpwl_lg": hpwl_holder.get("hpwl_lg"),
            # M4 T2b: probe_lifetime_gate.py's artifact postcondition inputs
            # (n_callbacks_with_active >= 3 and n_evals_while_active >= 3) --
            # populated unconditionally (not just when lifetime_out is given),
            # since they're cheap booleans-turned-counters already tracked in
            # cb_state regardless of whether a LifetimeRecorder is attached.
            "n_callbacks_with_active": cb_state["n_callbacks_with_active"],
            "n_evals_while_active": cb_state["n_evals_while_active"],
            **scale_fields,
            **legal_fields,
            **_phase_summary(timer, sampler),
            **_t8a_provenance(config_json, benchmark_kind=benchmark_kind,
                              device_baseline_gb=device_baseline_gb),
        }
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=1)
        return result
