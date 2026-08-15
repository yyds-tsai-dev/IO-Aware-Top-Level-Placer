import argparse, json, os, time, uuid
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
# provenance fields); 2 = this task's addition.
RESULT_SCHEMA_VERSION = 2

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

def _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed):
    nl = netlist_from_placedb(placedb)
    nl.node_x, nl.node_y = node_x, node_y
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    res = evaluate(nl, node_x, node_y, rg)
    return rg, {"io_count": res.io_count, "ft_count": res.ft_count,
                "tree_wl": res.tree_wl, "hpwl": res.hpwl,
                "large_net_lb": res.large_net_lb}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _dp_root():
    from ioplace.dreamplace_env import DEFAULT_ROOT
    return os.environ.get("DREAMPLACE_ROOT", DEFAULT_ROOT)


def _t8a_provenance(config_json):
    """M4 T8a (design draft sec 7.0 RESULT GATE, narrowed to this task's
    field list): run_id / status / schema_version / repo_commit /
    input_sha256, built from `profile.env_metadata()` ("接上
    ioplace/profile.py"). `status` is only ever "ok" here -- this driver
    only reaches its final json.dump() on a clean return (an exception
    propagates with no file written at all), so there is no path that could
    write a misleading "ok". The fuller experiment_status/workload_status
    split (design draft sec 2.2/7.0, for tasks like T2b/T9 that must record
    OOM/crash as a *legitimate* terminal state) is out of T8a's scope."""
    env = env_metadata(_repo_root(), _dp_root(), input_paths=(config_json,))
    return {"run_id": str(uuid.uuid4()), "status": "ok",
            "schema_version": RESULT_SCHEMA_VERSION,
            "repo_commit": env["ioplace_commit"], "input_sha256": env["input_sha256"],
            "env": env}


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
    phases = timer.phases
    peak_alloc_gb = max((p.get("peak_alloc_gb", 0.0) for p in phases.values()), default=0.0)
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


def run_flat(config_json, k, rtype, seed, out_json, *, dp_seed=None, deterministic=None):
    import torch
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

    orig_legalize = placer.op_collections.legalize_op
    def _timed_legalize(pos):
        gp_phase.__exit__(None, None, None)
        with timer.phase("lg"):
            out = orig_legalize(pos)
        return out
    placer.op_collections.legalize_op = _timed_legalize

    lr = params.global_place_stages[0]["learning_rate"]
    placer(params, placedb, lr)
    if "gp" not in timer.phases:
        # params.legalize_flag was off -- _timed_legalize (and so gp_phase's
        # own close) never fired.
        gp_phase.__exit__(None, None, None)

    final_overflow = float(placer.model.overflow.max())

    with timer.phase("eval"):
        node_x, node_y = extract_final_positions(placer, placedb)
        legal_fields = _legalization_diagnostics(placer, placedb, params, node_x, node_y)
        _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)

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
              "final_overflow": final_overflow,
              **legal_fields, **metrics,
              **_phase_summary(timer, sampler), **_t8a_provenance(config_json)}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result

def main():
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
    # Stage 2 S1 (spec sec 5.1/10): opt-in sidecar DEF export
    # (out.def/regions.json/netmap.json/coord.json) of the final GP+LG
    # placement, mode="io" only. Default None/off leaves existing behavior
    # unchanged.
    ap.add_argument("--emit-def", default=None,
                    help="if set, write out.def/regions.json/netmap.json/"
                         "coord.json (Stage 2 S1) into this directory after "
                         "GP+LG (mode=io only)")
    args = ap.parse_args()
    if args.mode == "flat":
        run_flat(args.config, args.k, args.rtype, args.seed, args.out,
                 dp_seed=args.dp_seed, deterministic=args.deterministic)
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
              emit_def=args.emit_def)

if __name__ == "__main__":
    main()
