"""M4 T0b model-discipline runner (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.1 T0b
row, v2.1 per Codex D4): produces the 11-point factorial probe the v1 GP
memory model (`GP_peak_bytes ~= 87*N_total + 73*N_pins + 160*n_bins`) needs
to have its 3 coefficients independently identifiable, which the v1 five
case-level points cannot give it (they co-vary format/filler/bins/topology
all at once).

Two design points, both on `mempool_group` (GP-only, det=1, seed 1000,
legalize_flag=0 -- no evaluator, no LG):

  1. bins x target_density factorial: num_bins_x/y in {1024, 2048, 4096} x
     target_density in {0.70, 0.835, 0.90} (9 points). Varies n_bins
     directly and N_total indirectly (target_density drives the filler
     count DREAMPlace adds -- see `_effective_scale_fields` in
     `ioplace/drivers/run_placement.py`), while N_pins stays fixed (same
     source netlist every point) -- this axis alone cannot identify the
     N_pins coefficient (Codex D4's finding).
  2. net-drop variants: `ioplace/bench/net_drop.py`'s 25%/50% variants of
     the same source netlist (bins/target_density held at the group main
     config's own values, 2048^2 / 0.714) -- moves N_pins independently of
     N_total and n_bins, closing the D4 gap.

9 + 2 = 11 points total. Every config is generated programmatically to a
scratch directory (never hand-authored/committed) so re-running this module
regenerates byte-identical configs from the constants below -- see
`build_points`/`write_scratch_configs`.

Every run writes `results/m4/scaling/t0b/<tag>.json`, built on
`ioplace/profile.py`'s `PhaseTimer`/`DeviceMemSampler`/`EventTimer`/
`build_profile_record` (sec 6.1/6.2's shared profiling infra) plus a
per-iteration CUDA-event timing histogram (`iter_ms`, via the same
`placer.iteration_callback` extension point `ioplace/drivers/
run_placement_io.py` already uses -- read-only reference, this module does
not import or modify anything under `ioplace/drivers/`).

Torch/DREAMPlace imports are deferred into `run_point` (not module level)
so `--dry-run`/`build_plan`/`build_points` are importable and testable
without a CUDA device or even torch installed -- this repo's GPU is
occupied by a concurrent cluster run as of this module's authoring pass
(design draft sec 7.1's red line: T0b's actual 11-point matrix is
**--dry-run only** until the GPU is free; only `ioplace/bench/net_drop.py`'s
CPU-streaming variant generation runs for real ahead of that).

CLI:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.run_t0b_matrix --dry-run
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.run_t0b_matrix --only group__bins2048__td0.714
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.run_t0b_matrix   # all 11, real GPU runs
"""
import argparse
import json
import os
import socket
import sys
import time
import uuid

from ioplace.bench import net_drop

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DP = os.environ.get("DREAMPLACE_ROOT", os.path.join(os.path.dirname(REPO), "DREAMPlace"))

GROUP_CONFIG = os.path.join(REPO, "benchmarks", "ispd25", "mempool_group.json")
NETDROP_SOURCE_PREFIX = os.path.join(
    REPO, "results", "m4", "bench", "mempool_group_export", "mempool_group")
NETDROP_OUT_DIR = os.path.join(REPO, "results", "m4", "bench", "netdrop")

SCRATCH_CONFIG_DIR_DEFAULT = os.path.join(REPO, "results", "m4", "scaling", "t0b", "_scratch_configs")
OUT_DIR_DEFAULT = os.path.join(REPO, "results", "m4", "scaling", "t0b")
PLAN_PATH_DEFAULT = os.path.join(REPO, "results", "m4", "scaling", "t0b", "dry_run_plan.json")

# Design point 1's factorial. `target_density`'s label preserves each
# value's own decimal precision (matches `scripts/gen_t0b_configs.py`'s
# convention) so 0.70/0.90 don't spuriously gain a 3rd digit and 0.835
# doesn't get truncated.
BINS = (1024, 2048, 4096)
TARGET_DENSITIES = (("0.70", 0.70), ("0.835", 0.835), ("0.90", 0.90))

# Design point 2: net-drop fractions, and the seed net_drop.py's own RNG
# (net *selection*) uses -- unrelated to DREAMPlace's RANDOM_SEED below.
NET_DROP_FRACTIONS = (0.25, 0.50)
NET_DROP_SEED = 0

# Every point in this matrix shares these DREAMPlace params (task's fixed
# protocol: "GP-only, det=1, seed 1000, legalize_flag=0").
ITERATION = 2000
RANDOM_SEED = 1000
DETERMINISTIC_FLAG = 1
LEGALIZE_FLAG = 0


def _bins_density_config(num_bins, target_density):
    """Design point 1: a `mempool_group.json` variant with `num_bins_x/y`
    (both the top-level fields *and* `global_place_stages[0]`'s own copy --
    DREAMPlace reads both, see `ioplace/drivers/run_placement.py`'s
    `_effective_scale_fields`) and `target_density` overridden; every other
    field (LEF/DEF inputs, wirelength/optimizer choice, gamma, ...)
    untouched."""
    with open(GROUP_CONFIG) as f:
        cfg = json.load(f)
    cfg["num_bins_x"] = num_bins
    cfg["num_bins_y"] = num_bins
    cfg["global_place_stages"][0]["num_bins_x"] = num_bins
    cfg["global_place_stages"][0]["num_bins_y"] = num_bins
    cfg["global_place_stages"][0]["iteration"] = ITERATION
    cfg["target_density"] = target_density
    cfg["random_seed"] = RANDOM_SEED
    cfg["deterministic_flag"] = DETERMINISTIC_FLAG
    cfg["legalize_flag"] = LEGALIZE_FLAG
    cfg["detailed_place_flag"] = 0
    return cfg


def net_drop_prefix(drop_fraction, seed=NET_DROP_SEED):
    return net_drop.derive_dst_prefix(NETDROP_OUT_DIR, NETDROP_SOURCE_PREFIX, drop_fraction, seed)


def _net_drop_config(drop_fraction, seed=NET_DROP_SEED):
    """Design point 2: a Bookshelf-format sibling of `mempool_group.json`
    pointed at `net_drop.py`'s `drop_fraction`/`seed` variant instead of the
    source LEF/DEF, bins/target_density left at the group main config's own
    values (2048^2 / 0.714 as of this module's authoring pass) -- this
    design point's whole purpose is moving N_pins independently, not
    bins/density. `lef_input`/`def_input` are dropped and the Bookshelf
    read-path fields the LEF/DEF config doesn't carry (`scale_factor`,
    `gift_init_flag`, `sort_nets_by_degree`, `num_threads`) are filled in to
    match this repo's existing Bookshelf-config convention (`benchmarks/
    ispd25/synthetic_1x2_n2.json`)."""
    with open(GROUP_CONFIG) as f:
        base = json.load(f)
    # sol_file_format="DEF" only matters to a placedb.write() call this GP-only
    # matrix never makes, but it's meaningless for a Bookshelf-sourced PlaceDB
    # (no rawdb DEF backing) and no Bookshelf config elsewhere in this repo
    # (bigblue4/synthetic_1x2_n2/synthetic_2x2_n2) carries it -- dropped to
    # match that convention rather than leave a dead, format-mismatched field.
    cfg = {k: v for k, v in base.items()
           if k not in ("lef_input", "def_input", "sol_file_format")}
    cfg["aux_input"] = net_drop_prefix(drop_fraction, seed) + ".aux"
    cfg.setdefault("scale_factor", 1.0)
    cfg.setdefault("gift_init_flag", 0)
    cfg.setdefault("sort_nets_by_degree", 0)
    cfg.setdefault("num_threads", 16)
    cfg["global_place_stages"][0]["iteration"] = ITERATION
    cfg["random_seed"] = RANDOM_SEED
    cfg["deterministic_flag"] = DETERMINISTIC_FLAG
    cfg["legalize_flag"] = LEGALIZE_FLAG
    cfg["detailed_place_flag"] = 0
    return cfg


def build_points():
    """The 11 (tag, design_point, config, meta) points -- 9 bins x density
    + 2 net-drop, in a fixed generation order so re-running is
    deterministic."""
    points = []
    for num_bins in BINS:
        for td_label, td_value in TARGET_DENSITIES:
            tag = f"group__bins{num_bins}__td{td_label}"
            points.append(dict(
                tag=tag, design_point="bins_density",
                config=_bins_density_config(num_bins, td_value),
                meta=dict(num_bins=num_bins, target_density=td_value),
            ))
    for frac in NET_DROP_FRACTIONS:
        pct = int(round(frac * 100))
        tag = f"group__netdrop{pct}__seed{NET_DROP_SEED}"
        points.append(dict(
            tag=tag, design_point="net_drop",
            config=_net_drop_config(frac, NET_DROP_SEED),
            meta=dict(drop_fraction=frac, net_drop_seed=NET_DROP_SEED,
                      net_drop_prefix=net_drop_prefix(frac, NET_DROP_SEED)),
        ))
    assert len(points) == 11, f"expected 11 T0b design points, got {len(points)}"
    tags = [p["tag"] for p in points]
    assert len(set(tags)) == len(tags), f"duplicate T0b tags: {tags}"
    return points


def write_scratch_configs(points, scratch_dir):
    """Writes every point's config to `<scratch_dir>/<tag>.json`
    (deterministic content -- see module docstring). Returns {tag:
    config_path}."""
    os.makedirs(scratch_dir, exist_ok=True)
    paths = {}
    for p in points:
        path = os.path.join(scratch_dir, f"{p['tag']}.json")
        with open(path, "w") as f:
            json.dump(p["config"], f, indent=4, sort_keys=True)
            f.write("\n")
        paths[p["tag"]] = path
    return paths


def build_plan(scratch_dir=SCRATCH_CONFIG_DIR_DEFAULT, out_dir=OUT_DIR_DEFAULT):
    """Generates every point's config to `scratch_dir` and returns the
    11-entry run plan (tag, design_point, config_path, out_json, meta) --
    the artifact `--dry-run` reports without executing any DREAMPlace
    run."""
    points = build_points()
    config_paths = write_scratch_configs(points, scratch_dir)
    plan = []
    for p in points:
        plan.append(dict(
            tag=p["tag"], design_point=p["design_point"],
            config_path=config_paths[p["tag"]],
            out_json=os.path.join(out_dir, f"{p['tag']}.json"),
            meta=p["meta"],
        ))
    return plan


def run_point(entry):
    """Executes one design point's DREAMPlace GP-only run (real GPU work --
    never called by `--dry-run`). Equivalent to `probe_gp_memory.py`'s
    `_load_dreamplace` + `_place` path, but with `placer.iteration_callback`
    (the same extension point `ioplace/drivers/run_placement_io.py` uses,
    read-only reference -- nothing under `ioplace/drivers/` is imported or
    modified here) wired to a `profile.py` `EventTimer` for a per-iteration
    CUDA-event timing histogram, which `_place`'s one-shot call cannot
    expose."""
    import numpy as np
    import torch
    from ioplace.drivers.run_placement import (
        _load_dreamplace, _last_metric_iteration, _gp_iteration_budget,
        _stop_overflow_reached,
    )
    from ioplace.profile import (
        PhaseTimer, DeviceMemSampler, EventTimer, env_metadata, build_profile_record,
    )

    tag = entry["tag"]
    config_path = entry["config_path"]
    run_id = str(uuid.uuid4())
    t_wall_start = time.time()

    device_baseline_gb = 0.0
    if torch.cuda.is_available():
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info()
        device_baseline_gb = (total - free) / 2**30

    timer = PhaseTimer()
    sampler = DeviceMemSampler().start()

    record = dict(
        tag=tag, design_point=entry["design_point"], meta=entry["meta"],
        config_path=config_path, run_id=run_id,
        device_baseline_gb=device_baseline_gb, ok=True,
    )
    try:
        with timer.phase("read"):
            params, placedb = _load_dreamplace(config_path)
        with timer.phase("initialize"):
            placedb.initialize(params)

        scale_meta = dict(
            n_movable=int(placedb.num_movable_nodes),
            n_physical=int(placedb.num_physical_nodes),
            n_filler=int(getattr(placedb, "num_filler_nodes", 0)),
            n_nets=int(placedb.num_nets),
            n_pins=int(len(placedb.pin2node_map)),
            n_bins=int(placedb.num_bins_x) * int(placedb.num_bins_y),
            lattice=None, k_chunk=None, n_active=None,
        )

        event_timer = EventTimer()
        cb_state = {"first": True}

        def _cb(iteration, pos):
            # One CUDA-event pair per callback-to-callback interval (sec
            # 6.1's per-iteration histogram): the first callback only opens
            # an interval (nothing to close yet); every later callback
            # closes the previous interval before opening the next; the
            # final interval is closed once outside the loop, after
            # placer(...) returns.
            if not cb_state["first"]:
                event_timer.stop()
            event_timer.start()
            cb_state["first"] = False

        import NonLinearPlace
        with timer.phase("gp"):
            lr = params.global_place_stages[0]["learning_rate"]
            # Same non-torch RNG seeding as _place() (ioplace/drivers/
            # run_placement.py) -- BasicPlace.py draws init_pos noise from
            # numpy's global RNG, only torch is reseeded per run internally.
            np.random.seed(params.random_seed)
            placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
            placer.iteration_callback = _cb
            metrics = placer(params, placedb, lr)
            if not cb_state["first"]:
                event_timer.stop()

        iter_ms = event_timer.elapsed_ms()
        sampler.stop()

        gp_phase = timer.phases["gp"]
        final_overflow = float(placer.model.overflow.max())
        gp_iterations_run = _last_metric_iteration(metrics)
        gp_iteration_budget = _gp_iteration_budget(params)

        env = env_metadata(REPO, DP, input_paths=[config_path])
        prof = build_profile_record(
            case=tag, K=None, rtype=None, arm="gp_only",
            phase_timer=timer, iter_ms=iter_ms,
            device_used_gb=sampler.device_used_gb,
            scale_meta=scale_meta, env=env,
        )
        record.update(prof)
        record.update(
            n_total=scale_meta["n_physical"] + scale_meta["n_filler"],
            gp_peak_alloc_gb=gp_phase["peak_alloc_gb"],
            gp_peak_reserved_gb=gp_phase["peak_reserved_gb"],
            final_overflow=final_overflow,
            gp_iterations_run=gp_iterations_run,
            gp_iteration_budget=gp_iteration_budget,
            stop_overflow_reached=_stop_overflow_reached(final_overflow, params.stop_overflow),
            random_seed=int(params.random_seed),
            deterministic_flag=int(params.deterministic_flag),
            legalize_flag=int(params.legalize_flag),
            hostname=socket.gethostname(),
            command=f"{sys.executable} -m ioplace.diagnostics.probes_m4.run_t0b_matrix --only {tag}",
            t_wall_s=time.time() - t_wall_start,
        )
    except RuntimeError as e:
        sampler.stop()
        if "out of memory" not in str(e).lower():
            raise
        record.update(ok=False, error=f"{type(e).__name__}: {e}"[:400],
                       t_wall_s=time.time() - t_wall_start)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="generate all 11 configs + the run plan, execute nothing")
    ap.add_argument("--only", metavar="TAG", default=None,
                     help="run (or, with --dry-run, report) a single point by tag")
    ap.add_argument("--scratch-dir", default=SCRATCH_CONFIG_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--plan-out", default=PLAN_PATH_DEFAULT)
    args = ap.parse_args()

    plan = build_plan(scratch_dir=args.scratch_dir, out_dir=args.out_dir)
    if args.only is not None:
        plan = [p for p in plan if p["tag"] == args.only]
        if not plan:
            raise SystemExit(f"--only {args.only!r}: no such tag (run with --dry-run to list tags)")

    if args.dry_run:
        os.makedirs(os.path.dirname(args.plan_out), exist_ok=True)
        with open(args.plan_out, "w") as f:
            json.dump(plan, f, indent=1, sort_keys=True)
        print(f"[run_t0b_matrix] --dry-run: wrote plan for {len(plan)} point(s) to {args.plan_out}")
        for p in plan:
            print(f"  {p['tag']:32s} [{p['design_point']:12s}] "
                  f"config={p['config_path']} -> out={p['out_json']}")
        return

    for p in plan:
        print(f"[run_t0b_matrix] running {p['tag']} ...")
        record = run_point(p)
        os.makedirs(os.path.dirname(p["out_json"]), exist_ok=True)
        with open(p["out_json"], "w") as f:
            json.dump(record, f, indent=1, sort_keys=True)
        status = "ok" if record.get("ok") else f"FAILED: {record.get('error')}"
        print(f"[run_t0b_matrix] {p['tag']}: {status} -> {p['out_json']}")


if __name__ == "__main__":
    main()
