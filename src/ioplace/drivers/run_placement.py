import argparse, json, os, socket, sys, time, uuid
import numpy as np
from ioplace.dreamplace_env import setup_dreamplace
from ioplace.regions import make_grid_regions, make_slicing_regions
from ioplace.region_grid import RegionGrid
from ioplace.netlist import netlist_from_placedb
from ioplace.evaluator_ref import evaluate
from ioplace.profile import PhaseTimer, DeviceMemSampler, host_rss_gb, env_metadata

GRID_SHAPES = {4: (2, 2), 8: (4, 2), 16: (4, 4), 32: (8, 4)}

# M4 T8a (design draft sec 7.0 RESULT GATE / T8a row): bumped when the
# driver JSON's field set changes. 1 = pre-T8a (no phase timing / E1 /
# provenance fields); 2 = T8a's own addition; 3 = overflow-diagnosis
# follow-up: stop_overflow_reached/gp_iteration_budget/gp_iterations_run/
# hpwl_gp/hpwl_lg/effective_target_density/num_filler_nodes/num_bins_x/
# num_bins_y, plus the RESULT GATE provenance gaps (command/hostname/
# benchmark_kind/device_baseline_gb).
RESULT_SCHEMA_VERSION = 3

def get_regions_for(die, k, rtype, seed, lattice=512):
    if rtype == "grid":
        nx, ny = GRID_SHAPES[k]
        return make_grid_regions(die, nx, ny, lattice=lattice)
    if rtype == "slicing":
        return make_slicing_regions(die, k, seed=seed, lattice=lattice)
    raise ValueError(rtype)

def _load_dreamplace(config_json):
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))   # config 內是相對路徑
    try:
        params.load(config_json)
        # M0/M1 protocol: GP+LG only (spec §8), DP off regardless of what the
        # input config says.
        #
        # `detailed_place_engine`/`detailed_place_command` (params.json: "external
        # detailed placement engine to be called after placement") only matter to
        # dreamplace/Placer.py's top-level place() driver, which invokes an *external*
        # DP tool (e.g. ntuplace) as a subprocess after the whole flow finishes. Our
        # driver never calls Placer.py -- it drives NonLinearPlace directly -- so that
        # field is a no-op for us either way; kept for documentation/defensiveness.
        #
        # The field NonLinearPlace.__call__ actually branches on (source: $DP/install/
        # dreamplace/NonLinearPlace.py:920, `if params.detailed_place_flag:` guarding
        # the internal GPU detailed-placement op) is `detailed_place_flag`. Some configs
        # (e.g. install/test/ispd2005/adaptec1.json) set it to 1, so it must be forced
        # off explicitly -- setting only detailed_place_engine would silently leave
        # internal DP running for those configs.
        params.detailed_place_engine = ""
        params.detailed_place_flag = 0
        params.plot_flag = 0
        placedb = PlaceDB.PlaceDB()
        placedb.read(params)
        return params, placedb
    finally:
        os.chdir(cwd)

def _place(params, placedb):
    import NonLinearPlace
    lr = params.global_place_stages[0]["learning_rate"]
    # BasicPlace.py:272-289/352-362 draws init_pos (centre noise + filler
    # positions) from numpy's *global* RNG; only torch is reseeded per run
    # (BasicPlace.py:265). Placer.py:36 seeds numpy for the reference flow,
    # which we bypass -- without this every placement in a process starts from
    # a different init_pos (measured ~0.8% run-to-run on adaptec1 io_count).
    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb, lr)
    return placer, metrics

def extract_final_positions(placer, placedb):
    """Return the final (node_x, node_y) after GP+LG, len == num_physical_nodes,
    in placedb's internal (shifted+scaled) coordinate system.

    Investigation (Task 7 Step 1; see $DP/install/dreamplace/NonLinearPlace.py:930-937
    and PlaceDB.py:1121-1133 `apply()`): NonLinearPlace.__call__ ends with

        cur_pos = self.pos[0].data.clone().cpu().numpy()
        placedb.apply(params, cur_pos[:num_movable_nodes],
                      cur_pos[num_nodes:num_nodes + num_movable_nodes])

    and `PlaceDB.apply()` writes that back into `self.node_x[:num_movable_nodes]` /
    `self.node_y[:num_movable_nodes]` *before* unscaling (the unscale inside apply()
    only feeds the separate rawdb/C++ mirror used for file export). So by the time
    `placer(params, placedb, lr)` returns, `placedb.node_x`/`node_y` already hold the
    final placement for the movable segment, in the same scaled coordinate system as
    `placedb.xl/yl/xh/yh`. The fixed/terminal segment (indices
    num_movable_nodes:num_physical_nodes) is never touched by apply() because it's
    never touched by the optimizer either -- it was already correct (and already in
    that same scaled system) since `placedb.initialize()`'s `scale()` call.

    This is candidate A from the brief, confirmed empirically: after a full `simple`
    GP+LG run, `placedb.node_x/node_y` and `placer.pos[0]` (candidate B, sliced per
    the brief's `pos[:n_phys]` / `pos[n_all:n_all+n_phys]` scheme) were bit-for-bit
    identical over all num_physical_nodes entries (movable and fixed alike). That
    equality is structural, not a coincidence of the `simple` benchmark: `apply()`
    is always the last data-mutating step of `__call__`, and it always assigns from
    a fresh clone of the same
    `self.pos[0]` tensor that candidate B reads. Candidate A is used here because it
    is simpler (no reliance on `pos[0]`'s internal `[x_all | y_all]` layout) and
    matches how the rest of ioplace (`netlist_from_placedb`, Task 2) already reads
    positions off placedb. `placer` is accepted for interface symmetry with future
    drivers (Tasks 9/12) and is intentionally unused here.
    """
    n_phys = placedb.num_physical_nodes
    node_x = np.array(placedb.node_x[:n_phys], dtype=np.float64)
    node_y = np.array(placedb.node_y[:n_phys], dtype=np.float64)
    return node_x, node_y

def _pack_eval_metrics(res):
    return {"io_count": res.io_count, "ft_count": res.ft_count,
               "tree_wl": res.tree_wl, "hpwl": res.hpwl,
               "large_net_lb": res.large_net_lb,
               "hard_lambda_sum": res.hard_lambda_sum,
               "io_rg": res.io_rg, "ft_rg": res.ft_rg}


def _pack_straddle_metrics(res):
    """v2 P-F (design sec 7 diagnostics 1-3). Deliberately separate from
    _pack_eval_metrics: that one also feeds run_flat / run_two_stage /
    run_reweight, none of which has a RESULT_FIELDS gate, so widening it would
    silently change three other drivers' result.json."""
    from ioplace.straddle import STRADDLE_SCALARS
    out = {}
    for name in STRADDLE_SCALARS:
        value = getattr(res, name)
        out[name] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return out


def _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed, *, include_result=False):
    nl = netlist_from_placedb(placedb)
    nl.node_x, nl.node_y = node_x, node_y
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    res = evaluate(nl, node_x, node_y, rg)
    metrics = _pack_eval_metrics(res)
    return (rg, metrics, res) if include_result else (rg, metrics)


def _repo_root():
    from ioplace.paths import REPO_ROOT
    return str(REPO_ROOT)


def _dp_root():
    from ioplace.dreamplace_env import DEFAULT_ROOT
    return os.environ.get("DREAMPLACE_ROOT", DEFAULT_ROOT)


def _t8a_provenance(config_json, *, benchmark_kind="real", device_baseline_gb=0.0):
    """M4 T8a (design draft sec 7.0 RESULT GATE, narrowed to this task's
    field list): run_id / status / schema_version / repo_commit /
    input_sha256, built from `profile.env_metadata()` ("接上
    ioplace/profile.py"). `status` is only ever "ok" here -- this driver
    only reaches its final json.dump() on a clean return (an exception
    propagates with no file written at all), so there is no path that could
    write a misleading "ok". The fuller experiment_status/workload_status
    split (design draft sec 2.2/7.0, for tasks like T2b/T9 that must record
    OOM/crash as a *legitimate* terminal state) is out of T8a's scope.

    Overflow-diagnosis follow-up: the remaining RESULT GATE fields
    (command/hostname/benchmark_kind/device_baseline_gb) folded into the
    same provenance dict -- `command`/`hostname` are pulled straight off
    the process (sys.argv, socket.gethostname()); `benchmark_kind` and
    `device_baseline_gb` are supplied by the caller since they need
    context this function doesn't have (a CLI flag, and a pre-run
    mem_get_info() reading taken before this function is even entered)."""
    env = env_metadata(_repo_root(), _dp_root(), input_paths=(config_json,))
    return {"run_id": str(uuid.uuid4()), "status": "ok",
            "schema_version": RESULT_SCHEMA_VERSION,
            "repo_commit": env["ioplace_commit"], "input_sha256": env["input_sha256"],
            "env": env,
            "command": " ".join(sys.argv), "hostname": socket.gethostname(),
            "benchmark_kind": benchmark_kind, "device_baseline_gb": device_baseline_gb}


def _last_metric_iteration(metrics):
    """`NonLinearPlace.__call__`'s return value (`all_metrics`) is the same
    list object its internals call `Lgamma_metrics` -- nested
    Lgamma/Llambda/Lsub sub-lists while the GP loop itself is running
    (`all_metrics[-1]` is a list of lists), but a *flat* `EvalMetrics`
    object once legalization runs (source: $DP/install/dreamplace/
    NonLinearPlace.py:891-892, appended directly to `all_metrics` right
    before its own `iteration += 1`) -- same shape if detailed placement
    ran (:933-934), which our driver never enables (`_load_dreamplace`
    forces `detailed_place_flag = 0`). Descending into `metrics[-1]` until
    a non-list leaf is reached covers both shapes: under this repo's GP+LG
    protocol (`legalize_flag=1`) the leaf is that post-legalization
    EvalMetrics, whose `.iteration` is exactly the GP loop's final
    iteration count (LG's own `iteration += 1` happens *after* this object
    is constructed); if legalize_flag were ever 0 it falls back to the GP
    loop's own last-recorded metric instead."""
    m = metrics[-1]
    while isinstance(m, list):
        m = m[-1]
    return int(m.iteration)


def _stop_overflow_reached(final_overflow, stop_overflow):
    """Overflow-diagnosis follow-up: pure comparison, split out from
    run_flat/run_io's result assembly so it's directly unit-testable (both
    branches) without a real DREAMPlace run -- same rationale as
    `_legalization_fields` above."""
    return bool(final_overflow <= float(stop_overflow))


def _gp_iteration_budget(params):
    """Overflow-diagnosis follow-up: the configured GP iteration budget
    (`global_place_stages[0]["iteration"]`), split out for the same
    mock-params testability reason as `_stop_overflow_reached`."""
    return int(params.global_place_stages[0]["iteration"])


def _effective_scale_fields(params, placedb):
    """Overflow-diagnosis follow-up: `effective_target_density`/
    `num_filler_nodes`/`num_bins_x`/`num_bins_y` (design draft's RESULT
    GATE 缺項), split out for mock-placedb/params testability. Must only be
    called *after* `placedb.initialize(params)` -- PlaceDB.py:842-846 may
    rewrite `params.target_density` (clamps it up to cell_utilization if the
    configured value is smaller), so reading it any earlier would risk the
    pre-clamp value."""
    result = {"effective_target_density": float(params.target_density),
            "num_filler_nodes": int(placedb.num_filler_nodes),
            "num_bins_x": int(placedb.num_bins_x), "num_bins_y": int(placedb.num_bins_y)}
    for key in ("num_physical_nodes","num_movable_nodes","num_nodes","num_nets",
                "num_terminals","num_terminal_NIs"):
        if hasattr(placedb,key):
            result[key]=int(getattr(placedb,key))
    if hasattr(placedb,"pin2node_map"):
        result["n_pins_canonical"]=len(placedb.pin2node_map)
    if getattr(params,"aux_input",None):
        from ioplace.bench.tile_bookshelf import _nets_header
        from ioplace.dreamplace_env import setup_dreamplace
        aux=params.aux_input
        if not os.path.isabs(aux):
            aux=os.path.join(setup_dreamplace(),"install",aux)
        with open(aux) as stream:
            nets=[token for line in stream for token in line.split("#",1)[0].split()
                  if token.endswith(".nets")]
        if len(nets)==1:
            _,result["n_pins_raw"]=_nets_header(os.path.join(os.path.dirname(aux),nets[0]))
            result["raw_pin_count_source"]="Bookshelf NumPins header"
    return result


def _phase_summary(timer, sampler):
    """M4 T8a: assembles the read/gp/lg/eval phase-timing/memory fields
    (design draft sec 6.1) from a `profile.PhaseTimer` + `profile.DeviceMemSampler`
    that covered the whole run.

    `peak_mem_mb` is recomputed here as the max *phase* `peak_alloc_gb`
    rather than a single end-of-run `torch.cuda.max_memory_allocated()`
    call: each phase's own `reset_peak_memory_stats()` (sec 1.4 B1's fix,
    now applied per-phase instead of once at run start) means the global
    allocator counter holds only the *last* phase's peak by the time the run
    finishes -- a single end-of-run read would silently under-report
    peak_mem_mb without changing its name or documented "this run's overall
    GPU peak, MB" semantics. Taking the max across every recorded phase
    reconstructs the true run-wide peak instead, since every GPU-allocating
    step of the run happens inside some phase's window."""
    # M4 T2b: `PhaseTimer(reset_peak=False)` phases (a `LifetimeRecorder`
    # is active and owns GPU-peak accounting instead -- see profile.py's
    # `_Phase.__exit__`) write `peak_alloc_gb=None`, not `0.0` -- filtered
    # out of the max() below rather than treated as a genuine zero-peak
    # phase (which would silently drag peak_mem_mb down).
    phases = timer.phases
    peak_alloc_gb = max((p["peak_alloc_gb"] for p in phases.values()
                        if p.get("peak_alloc_gb") is not None), default=0.0)
    host_peaks = [p.get("host_rss_hwm_at_phase_end", 0.0) for p in phases.values()] + [host_rss_gb()]
    return {
        "phases": phases,
        "t_read": phases.get("read", {}).get("t_s", 0.0),
        "t_gp": phases.get("gp", {}).get("t_s", 0.0),
        "t_lg": phases.get("lg", {}).get("t_s", 0.0),
        "t_eval": phases.get("eval", {}).get("t_s", 0.0),
        "device_used_gb": sampler.device_used_gb,
        "host_peak_rss_gb": max(host_peaks),
        "peak_mem_mb": peak_alloc_gb * 1024.0,
    }


def _legalization_fields(legal, num_unplaced, legalize_flag):
    """M4 T8a: maps a raw `legality_check_op()` bool + unplaced-cell count
    into the E1 JSON fields. Pure/synchronous so the "legalization failed"
    branch (design draft T8a's required 人工情境 test) is directly
    exercisable without needing a DREAMPlace run that actually produces an
    illegal placement."""
    if not legalize_flag:
        return {"legalization_status": "skipped", "num_unplaced_cells": None}
    return {"legalization_status": "success" if legal else "failed",
            "num_unplaced_cells": int(num_unplaced)}


def _legalization_diagnostics(placer, placedb, params, node_x, node_y):
    """M4 T8a: `legality_check_op` is already built and already called
    internally by `op_collections.legalize_op` (BasicPlace.py's
    `build_legalization_op`) -- but only to `logging.error()` on failure,
    the bool itself never reaches the caller. This calls the same,
    already-public op again (read-only, on the final full `pos` tensor) to
    surface it.

    `num_unplaced_cells` has no DREAMPlace-native definition
    (`legality_check_cpp` reports one aggregate legal/illegal bool for the
    whole design, not a per-cell breakdown, and there is no C++ op this
    driver is allowed to add) -- defined here as the count of movable cells
    whose final position is non-finite or outside the die box: cells GP/LG
    failed to leave in *any* valid location. This is a strict subset of what
    a full overlap-aware legality check would flag, but computable without a
    new op."""
    if not params.legalize_flag:
        return _legalization_fields(None, 0, params.legalize_flag)
    legal = bool(placer.op_collections.legality_check_op(placer.pos[0]))
    n_mov = placedb.num_movable_nodes
    xl, yl, xh, yh = float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh)
    x, y = node_x[:n_mov], node_y[:n_mov]
    invalid = ~np.isfinite(x) | ~np.isfinite(y) | (x < xl) | (x > xh) | (y < yl) | (y > yh)
    return _legalization_fields(legal, int(np.count_nonzero(invalid)), params.legalize_flag)


def run_flat(config_json, k, rtype, seed, out_json, *, dp_seed=None, deterministic=None,
            benchmark_kind="real", emit_eval=None):
    import torch
    # Overflow-diagnosis follow-up: device_baseline_gb -- whole-device usage
    # already present before this run touches CUDA at all (run start, prior
    # to any CUDA configuration), so a downstream reader can tell "high
    # device_used_gb because of us" apart from "high because something else
    # already had the device". Must be the very first CUDA call in this
    # function, before PhaseTimer/DeviceMemSampler (both of which touch
    # CUDA themselves) or _load_dreamplace.
    if torch.cuda.is_available():
        free0, total0 = torch.cuda.mem_get_info()
        device_baseline_gb = (total0 - free0) / 2**30
    else:
        device_baseline_gb = 0.0

    t0 = time.time()
    timer = PhaseTimer()
    sampler = DeviceMemSampler()
    sampler.start()

    # M4 T8a: "read" phase = load config + read + initialize placedb (sec
    # 1.4 B1's per-run reset, now folded into PhaseTimer's per-phase reset --
    # see _phase_summary's docstring for why peak_mem_mb is recomputed
    # rather than read once at the end).
    with timer.phase("read"):
        params, placedb = _load_dreamplace(config_json)
        if dp_seed is not None:
            params.random_seed = dp_seed
        if deterministic is not None:
            params.deterministic_flag = deterministic
        placedb.initialize(params)
        # Overflow-diagnosis follow-up: must be read after initialize() --
        # see _effective_scale_fields's docstring for why.
        scale_fields = _effective_scale_fields(params, placedb)

    # NonLinearPlace is a bare top-level module inside $DREAMPLACE_ROOT/install
    # (see Global Constraints), only importable once _load_dreamplace (above)
    # has called setup_dreamplace() and pushed install/ onto sys.path.
    import NonLinearPlace

    # "gp" phase: opened manually (not `with`) because it must close mid-call,
    # at the exact point NonLinearPlace.__call__ invokes legalize_op -- see
    # the op_collections.legalize_op monkeypatch below, the only place
    # available from the driver side to observe the GP/LG boundary without
    # touching DREAMPlace source (op_collections.legalize_op is a plain
    # mutable PlaceOpCollection field, BasicPlace.py:237, already the same
    # kind of extension point run_placement_io.py's attach_terms/
    # iteration_callback use).
    gp_phase = timer.phase("gp")
    gp_phase.__enter__()
    # Same init_pos determinism guard as _place() (see that function's
    # comment): BasicPlace draws centre-noise/filler init from numpy's
    # global RNG, seeded only by Placer.py's flow which we bypass here.
    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

    # Overflow-diagnosis follow-up: hpwl_gp/hpwl_lg -- hpwl_op(pos) called on
    # the exact same full pos tensor NonLinearPlace itself passes into
    # legalize_op, once right before (still GP's output) and once on the
    # legalized result it returns; stashed in a dict rather than a bare
    # nonlocal since it must survive both the closure and the case where
    # legalize_flag is off (never populated -> stays absent, matching
    # _legalization_fields's None-when-skipped convention below). `pos` is
    # the live optimizer's own requires_grad=True leaf tensor, so hpwl_op(pos)
    # would otherwise build (and leak, via hpwl_holder's retained reference)
    # a full autograd graph for a value we only read -- wrapped in
    # torch.no_grad() to match how DREAMPlace's own EvalMetrics.evaluate()
    # ($DP/install/dreamplace/EvalMetrics.py:109) calls this same op.
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
    placer.op_collections.legalize_op = _timed_legalize

    gp_iteration_budget = _gp_iteration_budget(params)
    lr = params.global_place_stages[0]["learning_rate"]
    dp_metrics = placer(params, placedb, lr)
    if "gp" not in timer.phases:
        # params.legalize_flag was off -- _timed_legalize (and so gp_phase's
        # own close) never fired.
        gp_phase.__exit__(None, None, None)

    final_overflow = float(placer.model.overflow.max())
    stop_overflow_reached = _stop_overflow_reached(final_overflow, params.stop_overflow)
    gp_iterations_run = _last_metric_iteration(dp_metrics)

    with timer.phase("eval"):
        node_x, node_y = extract_final_positions(placer, placedb)
        legal_fields = _legalization_diagnostics(placer, placedb, params, node_x, node_y)
        rg, metrics, eval_result = _evaluate_and_pack(
            placedb, node_x, node_y, k, rtype, seed, include_result=True)
        if emit_eval is not None:
            from ioplace.export.evaluation import save_evaluation
            save_evaluation(emit_eval, netlist_from_placedb(placedb), rg, eval_result,
                            node_x, node_y, placedb.net_names,
                            provenance={"config": os.path.abspath(config_json),
                                        "placement_stage": "gp_lg"})

    sampler.stop()

    result = {"mode": "flat", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "dp_seed": int(params.random_seed),
              "det": int(params.deterministic_flag), "runtime_s": time.time() - t0,
              # sec 1.4 B1: marks this JSON as post-fix (per-run reset at
              # measurement start) so it can be told apart from pre-fix
              # results whose peak_mem_mb was the process cumulative HWM
              # (sec 1.4 B1 gate M4-G7 -- those must be flagged as stale,
              # not silently compared against this field).
              "peak_mem_mb_reset_semantics": True,
              "workload_status": "completed",
              "generator_verified": False if benchmark_kind == "synthetic" else None,
              "evaluator_file": os.path.abspath(emit_eval) if emit_eval else None,
              "final_overflow": final_overflow,
              "stop_overflow_reached": stop_overflow_reached,
              "gp_iteration_budget": gp_iteration_budget,
              "gp_iterations_run": gp_iterations_run,
              "hpwl_gp": hpwl_holder.get("hpwl_gp"), "hpwl_lg": hpwl_holder.get("hpwl_lg"),
              **scale_fields,
              **legal_fields, **metrics,
              **_phase_summary(timer, sampler),
              **_t8a_provenance(config_json, benchmark_kind=benchmark_kind,
                                device_baseline_gb=device_baseline_gb)}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    return result

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["flat", "two_stage", "reweight", "io"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reweight-every", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rho-max", type=float, default=0.1)
    ap.add_argument("--tau-hi", type=float, default=0.30)
    ap.add_argument("--tau-lo", type=float, default=0.03)
    ap.add_argument("--alpha-io", type=float, default=0.0)
    ap.add_argument("--rho-margin", type=float, default=0.0)
    ap.add_argument("--w-mode", default="unit", choices=["unit", "inv_deg"])
    ap.add_argument("--d-max", type=int, default=None)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--dp-seed", type=int, default=None)
    ap.add_argument("--deterministic", type=int, default=None)
    # sec 1.4 B2: gates run_placement_io.py's expensive per-callback
    # io_term.diagnostics() call (mode="io" only; no-op for other modes).
    ap.add_argument("--diag-every", type=int, default=1,
                    help="run io_term.diagnostics() every N-th `--every`-gated "
                         "callback (default 1 = every occurrence, matching "
                         "pre-B2 behavior)")
    ap.add_argument("--no-diag", action="store_true",
                    help="never run io_term.diagnostics() (overrides --diag-every)")
    # M3 Phase B (docs/results/2026-08-14-m3-s4-adjudication.md sec C item 2):
    # dynamic per-net ft_rg reweight, same M1 formula family as --alpha-io
    # (w = 1 + alpha*min(signal, cap)) with the signal switched to ft_rg.
    ap.add_argument("--ft-reweight", default="off", choices=["off", "on"])
    ap.add_argument("--alpha-ft", type=float, default=0.5)
    ap.add_argument("--callback-order", choices=["legacy", "atomic"], default="legacy")
    ap.add_argument("--f-ft-max", type=float, default=0.)
    ap.add_argument("--ft-ramp-mode", choices=["window", "constant"], default="window")
    ap.add_argument("--tau-start", type=float, default=.12)
    ap.add_argument("--tau-full", type=float, default=.05)
    ap.add_argument("--home-period", type=int, default=None)
    ap.add_argument("--topology-diagnostics", action="store_true")
    ap.add_argument("--wl-reweight", choices=["off", "crossings", "ft_rg"], default="off")
    ap.add_argument("--alpha-wl", type=float, default=.2)
    ap.add_argument("--wl-cap", type=float, default=10.)
    ap.add_argument("--cap", type=float, default=64.)
    # Stage 2 S1 (spec sec 5.1/10): opt-in sidecar DEF export
    # (out.def/regions.json/netmap.json/coord.json) of the final GP+LG
    # placement, mode="io" only. Default None/off leaves existing behavior
    # unchanged.
    ap.add_argument("--emit-def", default=None,
                    help="if set, write out.def/regions.json/netmap.json/"
                         "coord.json (Stage 2 S1) into this directory after "
                         "GP+LG (mode=io only)")
    ap.add_argument("--emit-eval", default=None,
                    help="write aligned per-net evaluator evidence to this NPZ (flat/io)")
    # Overflow-diagnosis follow-up (RESULT GATE): records whether a run is
    # a real benchmark (ISPD/mempool-style) or a synthetic/tiler-generated
    # one -- only meaningful for "flat"/"io" (T8a's field-addition scope).
    ap.add_argument("--benchmark-kind", default="real", choices=["real", "synthetic"])
    ap.add_argument("--discrete-mode",choices=["none","ce","refine","ce_refine"],default="none")
    ap.add_argument("--discrete-max-active",type=int,default=65536)
    # v2 P-H (design sec 4): gradient-norm normalisation for the extra
    # objective terms. mode=io only; no-op for the other modes.
    ap.add_argument("--norm-policy", choices=["legacy", "grandplan", "adaptive"],
                    default="legacy",
                    help="coefficient normalisation policy: 'legacy' reproduces "
                         "the retired lambda_IO EMA + kappa_FT force share exactly; "
                         "'grandplan' uses lambda_t = wt_t*||grad WL||_p/||grad T_t||_p "
                         "with wt stepping +0.05 every --norm-ramp-period iterations; "
                         "'adaptive' drives each term to a target force share")
    ap.add_argument("--norm-p", type=int, choices=[1, 2], default=1,
                    help="gradient norm order for every normalisation probe "
                         "(default 1: every calibrated constant was fitted under L1)")
    ap.add_argument("--norm-ramp-period", type=int, default=100,
                    help="policy grandplan: iterations between +0.05 wt steps")
    ap.add_argument("--norm-wt-max", type=float, default=1.,
                    help="policy grandplan: upper bound on wt (default 1.0)")
    ap.add_argument("--norm-probe-every", type=int, default=50,
                    help="iterations between normalisation probes; must be a "
                         "positive multiple of --every")
    ap.add_argument("--norm-target-share", default=None,
                    help="policy adaptive: per-term target force shares, "
                         "e.g. 'io=0.3,ft=0.1'")
    ap.add_argument("--norm-trace", default=None,
                    help="write norm_trace.jsonl here (default "
                         "<out>.norm_trace.jsonl for non-legacy policies, "
                         "no trace for legacy)")
    # v2 P-F (design sec 7): the anchor at which the soft region assignment is
    # evaluated. mode=io only; no-op for the other modes.
    ap.add_argument("--node-anchor", choices=["lower_left", "center", "pin"],
                    default="center",
                    help="anchor for the soft region assignment: 'center' "
                         "(default) evaluates the SDF at x+0.5*w, y+0.5*h, "
                         "matching the freeze rule and whole-cell fence "
                         "ownership; 'lower_left' is the legacy anchor; 'pin' "
                         "is rejected by every driver -- it exists only in "
                         "IoTermRef as a small-scale bias probe")
    return ap


def main():
    args = build_parser().parse_args()
    if args.mode == "flat":
        run_flat(args.config, args.k, args.rtype, args.seed, args.out,
                 dp_seed=args.dp_seed, deterministic=args.deterministic,
                 benchmark_kind=args.benchmark_kind, emit_eval=args.emit_eval)
    elif args.mode == "two_stage":
        from ioplace.drivers.run_placement_two_stage import run_two_stage  # Task 9
        run_two_stage(args.config, args.k, args.rtype, args.seed, args.out)
    elif args.mode == "reweight":
        from ioplace.drivers.run_placement_reweight import run_reweight    # Task 12
        run_reweight(args.config, args.k, args.rtype, args.seed, args.out,
                     every=args.reweight_every, alpha=args.alpha)
    else:
        from ioplace.drivers.run_placement_io import run_io
        run_io(args.config, args.k, args.rtype, args.seed, args.out,
              rho_max=args.rho_max, tau_hi=args.tau_hi, tau_lo=args.tau_lo,
              alpha_io=args.alpha_io, rho_margin=args.rho_margin, w_mode=args.w_mode,
              ignore_net_degree=args.d_max, every=args.every,
              dp_seed=args.dp_seed, deterministic=args.deterministic,
              diag_every=args.diag_every, no_diag=args.no_diag,
              ft_reweight=args.ft_reweight, alpha_ft=args.alpha_ft,
              callback_order=args.callback_order, f_ft_max=args.f_ft_max,
              ft_ramp_mode=args.ft_ramp_mode, tau_start=args.tau_start, tau_full=args.tau_full,
              home_period=args.home_period, topology_diagnostics=args.topology_diagnostics,
              wl_reweight=args.wl_reweight, alpha_wl=args.alpha_wl, wl_cap=args.wl_cap,
              cap=args.cap,
              discrete_mode=args.discrete_mode,discrete_max_active=args.discrete_max_active,
              emit_def=args.emit_def,
              emit_eval=args.emit_eval,
              benchmark_kind=args.benchmark_kind,
              norm_policy=args.norm_policy, norm_p=args.norm_p,
              norm_ramp_period=args.norm_ramp_period, norm_wt_max=args.norm_wt_max,
              norm_probe_every=args.norm_probe_every,
              norm_target_share=args.norm_target_share, norm_trace=args.norm_trace,
              node_anchor=args.node_anchor)

if __name__ == "__main__":
    main()
