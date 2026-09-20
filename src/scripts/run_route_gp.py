"""WA, paper Eq.7, and routing-gradient GP with measured OpenROAD feedback.

Retired by the v2 design (sec 1): requires IOPLACE_ENABLE_GR_IN_LOOP=1 and is
unmaintained. The supported protocol is one GRT call after placement.
"""
from ioplace.gr_in_loop import require_gr_in_loop

require_gr_in_loop()

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from ioplace.dp_hook import attach_terms, assert_optimizer_lock
from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for, _pack_eval_metrics
from ioplace.export.def_export import export_def
from ioplace.netlist import netlist_from_placedb
from ioplace.ops.route_gp import FrozenJointRouteCost
from ioplace.ops.routing_gp_controller import RoutingGPController
from ioplace.ops.steiner_gp import tensor_digest
from ioplace.region_grid import RegionGrid
from ioplace.regions import RegionSet
from ioplace.route_eval.joint import ResourceGrid
from ioplace.route_eval.online_openroad import digest, run_openroad, load_observation


def routing_policy(args, *, final=False):
    policy=dict(congestion_iterations=args.final_grt_iterations if final else args.feedback_grt_iterations,
        allow_congestion=args.final_allow_congestion if final else args.feedback_allow_congestion,
        threads=args.grt_threads)
    if getattr(args,'grt_signal_layers',None):policy['signal_layers']=args.grt_signal_layers
    return policy


def repair_routed_snapshot(export_dir, snapshot, *, num_nodes, movable_names,
                           lefs, binary, legality_check, threads=16):
    """Repair actual fragmented-row legality without changing the source tensor.

    The repaired DEF and ordered orientation sidecar describe the routed state.
    The returned tensor contains its positions; orientations are returned apart
    because the live GP tensor has no orientation degrees of freedom.
    """
    from ioplace.route_eval.placement_openroad import legalize_export
    directory = Path(export_dir)
    original_digest = tensor_digest(snapshot)
    shutil.copy2(directory / "out.def", directory / "dreamplace.def")
    repair_input = directory / "dreamplace_export"
    repair_input.mkdir()
    # Keep receipt input paths immutable when the repaired out.def is published.
    os.link(directory / "dreamplace.def", repair_input / "out.def")
    result = legalize_export(repair_input, lefs, directory / "row_legalization", binary,
                             movable_names, threads=threads)
    x, y = np.asarray(result["x_dbu"]), np.asarray(result["y_dbu"])
    orientations = list(result["orientations"])
    count = len(movable_names)
    if x.shape != (count,) or y.shape != (count,) or len(orientations) != count or not np.isfinite([x, y]).all():
        raise ValueError("row legalization returned invalid ordered movable coordinates")
    coord = json.loads((directory / "coord.json").read_text())
    scale, shift = float(coord["scale_factor"]), np.asarray(coord["shift_factor"], dtype=float)
    if not np.isfinite(scale) or scale <= 0 or shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("invalid routing coordinate sidecar")
    repaired = snapshot.detach().clone()
    repaired[:count] = torch.as_tensor((x - shift[0]) * scale, device=repaired.device, dtype=repaired.dtype)
    repaired[num_nodes:num_nodes+count] = torch.as_tensor((y - shift[1]) * scale, device=repaired.device, dtype=repaired.dtype)
    if not bool(legality_check(repaired)):
        raise RuntimeError("row-repaired placement failed DREAMPlace legality check")
    if tensor_digest(snapshot) != original_digest:
        raise RuntimeError("row legalization mutated its source placement")
    if (not torch.equal(repaired[count:num_nodes], snapshot[count:num_nodes])
            or not torch.equal(repaired[num_nodes+count:], snapshot[num_nodes+count:])):
        raise RuntimeError("row legalization changed fixed/filler coordinates")
    shutil.copy2(result["legal_def_path"], directory / "out.def")
    orientation_path = directory / "movable_orientations.json"
    orientation_path.write_text(json.dumps(orientations))
    receipt = dict(result["receipt"])
    receipt.update(receipt_path=result["receipt_path"], receipt_sha256=digest(result["receipt_path"]),
                   orientation_path=str(orientation_path.resolve()), orientation_sha256=digest(orientation_path),
                   repaired_position_sha256=tensor_digest(repaired))
    return repaired, receipt, orientations


def observation_publication(oracle_rows, trace):
    """Report actual later-step consumption, including early-stop pending data."""
    consumed, pending, publication = [], [], []
    for measured in oracle_rows:
        version = measured.get("observation_version")
        if version is None:
            continue
        later = next((row for row in trace if row["active"]
            and row["iteration"] > measured["iteration"]
            and row["published_observation_version"] >= version), None)
        record = dict(observation_version=version, measured_after_iteration=measured["iteration"],
                      first_consuming_iteration=None, generation=None)
        if later is None:
            pending.append(version)
        else:
            consumed.append(version)
            record.update(first_consuming_iteration=later["iteration"], generation=later["generation"])
        publication.append(record)
    return dict(consumed_observation_versions=consumed, pending_observation_versions=pending,
                publication=publication)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["wa_standard", "wa", "paper", "joint"], default="joint")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--rebuild", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-overflow", type=float)
    parser.add_argument("--openroad")
    parser.add_argument("--calibration", help="legal DEF export directory with coordinate/region/net sidecars")
    parser.add_argument("--calibration-placement", help="physical node_x/node_y npz paired with calibration DEF")
    parser.add_argument("--router-every", type=int, default=0)
    parser.add_argument("--feedback-grt-iterations", type=int, default=50)
    parser.add_argument("--feedback-allow-congestion", action="store_true")
    parser.add_argument("--final-grt-iterations", type=int, default=50)
    parser.add_argument("--final-allow-congestion", action="store_true")
    parser.add_argument("--grt-threads", type=int, default=4)
    parser.add_argument("--grt-signal-layers", help="explicit signal routing range, e.g. metal2-metal10")
    parser.add_argument("--route-strength", type=float, default=.1)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--hot-nets", type=int, default=16)
    parser.add_argument("--congestion-weight", type=float, default=.1)
    parser.add_argument("--wirelength-weight", type=float, default=.001)
    args = parser.parse_args()
    if min(args.feedback_grt_iterations,args.final_grt_iterations,args.grt_threads)<1:
        parser.error("positive feedback/final GRT iteration limits and threads required")
    if args.mode == "joint" and not args.openroad:
        parser.error("joint routing-gradient GP requires --openroad")
    if args.iterations <= args.start or args.start < 1 or args.rebuild < 1 or args.router_every < 0:
        parser.error("need iterations > start >= 1, positive rebuild, and nonnegative router interval")
    if args.calibration_placement and not args.calibration:
        parser.error("--calibration-placement requires --calibration")
    for name, value in (("route strength", args.route_strength), ("congestion weight", args.congestion_weight),
                        ("wirelength weight", args.wirelength_weight)):
        if not np.isfinite(value) or value < 0:
            parser.error(f"finite nonnegative {name} required")
    if args.hot_nets < 0 or (args.tau is not None and (not np.isfinite(args.tau) or args.tau <= 0)):
        parser.error("nonnegative hot-net count and finite positive tau required")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = Path(args.config).resolve()
    settings = json.loads(config.read_text())
    params, db = _load_dreamplace(str(config))
    if len(params.global_place_stages) != 1 or params.global_place_stages[0]["wirelength"] != "weighted_average":
        raise ValueError("routing GP requires one weighted-average GP stage")
    params.global_place_stages[0]["iteration"] = args.iterations
    params.global_place_stages[0]["Lsub_iteration"] = 1
    params.use_bb = 0
    if args.seed is not None:
        params.random_seed = args.seed
    if args.stop_overflow is not None:
        params.stop_overflow = args.stop_overflow
    db.initialize(params)
    assert_optimizer_lock(params)
    if (len(db.regions) or params.routability_opt_flag or params.timing_opt_flag or params.macro_place_flag
            or not params.global_place_flag or not params.legalize_flag or params.detailed_place_flag):
        raise ValueError("routing GP requires flat static-pin GP+LG without macros, fences, timing, or routability mutation")

    # export_def applies coordinates/status/orientation to rawdb. Own a separate
    # raw database so intermediate exports cannot mutate GP's database mirror.
    export_params, export_db = _load_dreamplace(str(config))
    export_db.initialize(export_params)
    if export_db.rawdb is db.rawdb:
        raise RuntimeError("export database must be isolated from the GP database")
    torch.set_num_threads(params.num_threads)
    np.random.seed(params.random_seed)
    import NonLinearPlace, PlaceObj, NesterovAcceleratedGradientOptimizer, BasicPlace
    placer = NonLinearPlace.NonLinearPlace(params, db, None)
    nl = netlist_from_placedb(db)
    n, nm, nall = nl.num_physical, nl.num_movable, db.num_nodes
    movable_names = [name.decode() if isinstance(name, bytes) else str(name) for name in db.node_names[:nm]]
    fixed_x, fixed_y = nl.node_x[nm:].copy(), nl.node_y[nm:].copy()
    original_legalize = placer.op_collections.legalize_op
    initial_sha = tensor_digest(placer.pos[0])

    def physical(pos):
        array = pos.detach().cpu().numpy()
        return array[:n].copy(), array[nall:nall+n].copy()

    def save_positions(path, pos, full=False):
        x, y = physical(pos)
        fields = dict(node_x=x, node_y=y)
        if full:
            fields["full_pos"] = pos.detach().cpu().numpy()
        np.savez_compressed(path, **fields)

    def check_snapshot(pos):
        x, y = physical(pos)
        fixed = np.array_equal(x[nm:], fixed_x) and np.array_equal(y[nm:], fixed_y)
        legal = bool(placer.op_collections.legality_check_op(pos))
        if not fixed or not legal or not bool(torch.isfinite(pos).all()):
            raise RuntimeError("router snapshot must be finite, legal, and preserve fixed coordinates")
        return x, y

    calibration = Path(args.calibration).resolve() if args.calibration else None
    rs = (RegionSet.from_json(calibration / "regions.json") if calibration
          else get_regions_for((nl.xl, nl.yl, nl.xh, nl.yh), 32, "slicing", 0))
    if not np.allclose(rs.die, [nl.xl, nl.yl, nl.xh, nl.yh], rtol=0, atol=1e-5):
        raise ValueError("calibration region die does not match the GP design")
    rg = RegionGrid(rs)
    tau = args.tau if args.tau is not None else 2 * min(rg.cell_w, rg.cell_h)
    save_positions(out / "initial.npz", placer.pos[0], full=True)
    (out / "config.json").write_text(config.read_text())
    from ioplace.paths import REPO_ROOT
    from ioplace.dreamplace_env import setup_dreamplace
    install = Path(setup_dreamplace()) / "install"
    def input_path(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() or path.exists() else (install / path).resolve()
    lefs = [input_path(p) for p in settings.get("lef_input", [])]
    inputs = [config, *lefs]
    for key in ("def_input", "verilog_input", "aux_input"):
        if settings.get(key):
            inputs.append(input_path(settings[key]))
    sources = list((Path(REPO_ROOT) / "src/ioplace").rglob("*.py"))
    sources += list((Path(REPO_ROOT) / "src/ioplace").rglob("*.cpp")) + [Path(__file__).resolve()]
    sources += [Path(m.__file__).resolve() for m in (NonLinearPlace, PlaceObj, NesterovAcceleratedGradientOptimizer, BasicPlace)]
    sources += [install.parent / "dreamplace/ops/place_io/src" / name
                for name in ("PlaceDB.cpp", "PyPlaceDB.cpp")]
    protocol = dict(argv=sys.argv, settings=vars(args), effective_seed=params.random_seed,
        effective_stop_overflow=params.stop_overflow, initial_sha256=initial_sha,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), device=str(placer.pos[0].device),
        source_sha256={str(p): digest(p) for p in sources}, config_sha256=digest(config),
        input_sha256={str(p): digest(p) for p in inputs},
        tool_sha256={str(Path(args.openroad).resolve()): digest(args.openroad)} if args.openroad else {},
        objective="WA+density+paper interior correction+route_lambda*(weighted physical IO+joint capacity penalty+route length+measured edge prices)",
        paper_mask="whole-pin strict bbox interior; fixed trunk constant dropped",
        tau=tau, export_database_isolated=True,
        row_legalization="common isolated OpenROAD detailed placement before every calibration, online, and final GRT",
        orientation_semantics="repaired DEF/sidecar orientations used by measured prediction and final evaluator; active GP retains canonical static pin offsets")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    oracle_rows = []

    def route_snapshot(label, source, *, iteration=-1, prepared=None, already_legal=False):
        live_before = tensor_digest(placer.pos[0])
        db_x, db_y = db.node_x.copy(), db.node_y.copy()
        snapshot = source.detach().clone()
        if prepared is None and not already_legal:
            with torch.no_grad():
                snapshot = original_legalize(snapshot).detach().clone()
        x, y = check_snapshot(snapshot)
        directory = out / label
        directory.mkdir()
        save_positions(directory / "dreamplace.npz", snapshot, full=True)
        if prepared is not None:
            for name in ("out.def", "coord.json", "regions.json", "netmap.json"):
                shutil.copy2(prepared / name, directory / ("input_" + name))
        # The authoritative measured tensor owns the routed DEF even when a
        # supplied calibration export contains an older placement. Preserve the
        # input export for provenance, then regenerate all routing sidecars.
        export_def(export_db, export_params, x, y, str(directory), rs)
        snapshot, row_repair, orientations = repair_routed_snapshot(directory, snapshot,
            num_nodes=nall, movable_names=movable_names, lefs=lefs, binary=args.openroad,
            legality_check=placer.op_collections.legality_check_op)
        check_snapshot(snapshot)
        save_positions(directory / "placement.npz", snapshot, full=True)
        run_openroad(directory / "out.def", lefs, directory / "grt", args.openroad,
            **routing_policy(args,final=label=="routed"))
        observation = load_observation(directory / "grt", directory / "coord.json", directory / "regions.json", directory / "netmap.json")
        observation["movable_orientations"] = orientations
        live_after = tensor_digest(placer.pos[0])
        arrays_unchanged = np.array_equal(db.node_x, db_x) and np.array_equal(db.node_y, db_y)
        if live_before != live_after or not arrays_unchanged:
            raise RuntimeError("routing snapshot mutated live GP placement/database")
        row = dict(label=label, iteration=int(iteration), legal=True, fixed_unchanged=True,
            actual_io=observation["actual_io"], actual_wirelength=observation["actual_wirelength"],
            source_sha256=observation["source_sha256"], placement_sha256=observation["placement_sha256"],
            measured_position_sha256=tensor_digest(snapshot), live_position_before=live_before, live_position_after=live_after,
            db_arrays_unchanged=bool(arrays_unchanged), export_database_isolated=True,
            snapshot=str(directory / "placement.npz"), snapshot_sha256=digest(directory / "placement.npz"),
            row_legalization=row_repair,
            **{k:observation[k] for k in ("native_congestion","routing_policy","router_elapsed_s",
                "aggregated_resource_overflow","preferred_layer_overflow","net_audit",
                "routed_net_count","routed_net_names_sha256")})
        oracle_rows.append(row)
        (out / "oracle.json").write_text(json.dumps(oracle_rows, indent=2))
        print("ROUTER", label, row["actual_io"], flush=True)
        return observation, snapshot, row

    baseline = measured = baseline_row = None
    if args.openroad and (args.mode == "joint" or calibration is not None):
        if not lefs or not settings.get("def_input"):
            raise ValueError("OpenROAD requires LEF/DEF inputs")
        if calibration:
            placement_path = Path(args.calibration_placement).resolve() if args.calibration_placement else calibration / "placement.npz"
            if not placement_path.exists():
                raise ValueError("calibration requires its paired placement.npz or --calibration-placement")
            with np.load(placement_path) as saved:
                x, y = saved["node_x"], saved["node_y"]
                if x.shape != (n,) or y.shape != (n,):
                    raise ValueError("calibration placement has different node count")
                measured = placer.pos[0].detach().clone()
                measured[:n] = torch.as_tensor(x, device=measured.device, dtype=measured.dtype)
                measured[nall:nall+n] = torch.as_tensor(y, device=measured.device, dtype=measured.dtype)
            protocol["calibration_inputs_sha256"] = {str(calibration / name): digest(calibration / name)
                for name in ("out.def", "coord.json", "regions.json", "netmap.json")}
            protocol["calibration_inputs_sha256"][str(placement_path)] = digest(placement_path)
            baseline, measured, baseline_row = route_snapshot("calibration", measured, prepared=calibration)
        else:
            baseline, measured, baseline_row = route_snapshot("calibration", placer.pos[0])
    grid = baseline["grid"] if baseline else ResourceGrid(
        np.linspace(nl.xl, nl.xh, 33), np.linspace(nl.yl, nl.yh, 33))
    capacity = baseline["capacity"] if baseline else np.ones(grid.edge_count)
    term = FrozenJointRouteCost(nl, rg, grid, capacity, num_nodes=nall,
        net_mask=placer.data_collections.net_mask_ignore_large_degrees,
        net_weights=placer.data_collections.net_weights,
        hot_nets=args.hot_nets if args.mode == "joint" else 0,
        congestion_weight=args.congestion_weight, wirelength_weight=args.wirelength_weight,
        flute_threads=params.num_threads)
    if args.mode == "joint":
        baseline_row.update(term.assimilate(baseline, measured))
        (out / "oracle.json").write_text(json.dumps(oracle_rows, indent=2))

    def online(iteration, pos):
        # Every assimilated online observation must have a subsequent GP step.
        if iteration + 1 >= args.iterations:
            return
        observation, snapshot, row = route_snapshot(f"step{iteration+1:06d}", pos, iteration=iteration)
        row.update(term.assimilate(observation, snapshot))
        (out / "oracle.json").write_text(json.dumps(oracle_rows, indent=2))

    controller = RoutingGPController(term, placer, mode=args.mode, start=args.start,
        rebuild_every=args.rebuild, tau=tau, route_strength=args.route_strength,
        router_every=args.router_every, router_callback=online)
    attach_terms(params, [] if args.mode == "wa_standard" else [controller])
    placer.iteration_callback = controller.callback
    boundary = {}
    def legalize_final(pos):
        save_positions(out / "gp.npz", pos, full=True)
        boundary["gp_sha256"] = tensor_digest(pos)
        boundary["gp_hpwl"] = float(placer.op_collections.hpwl_op(pos).detach())
        return original_legalize(pos)
    placer.op_collections.legalize_op = legalize_final
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    placer(params, db, params.global_place_stages[0]["learning_rate"])
    save_positions(out / "dreamplace_legal.npz", placer.pos[0], full=True)
    check_snapshot(placer.pos[0])
    if args.mode in ("paper", "joint") and not controller.rebuilds:
        raise RuntimeError("GP stopped before the requested objective activated")
    # Final GRT may be expensive. Persist causal evidence before it starts.
    (out / "gp_completed.json").write_text(json.dumps(dict(mode=args.mode,initial_sha256=initial_sha,
        **boundary,trace=controller.trace,gradient_audit=controller.gradient_audit,oracle=oracle_rows,
        legal=True,fixed_unchanged=True,final_grt_pending=bool(args.openroad),
        **observation_publication(oracle_rows,controller.trace)),indent=2))
    if args.openroad:
        final_observation, final_pos, final_row = route_snapshot("routed", placer.pos[0],
            iteration=len(controller.trace), already_legal=True)
        from ioplace.ops.placement_offsets import oriented_netlist
        final_nl = oriented_netlist(nl, final_observation["movable_orientations"])
    else:
        final_observation = None
        final_pos, final_nl = placer.pos[0], nl
    x, y = physical(final_pos)
    save_positions(out / "legal.npz", final_pos, full=True)
    from ioplace.evaluator_gpu import GpuEvalContext
    context = GpuEvalContext(final_nl, rg, device="cuda" if params.gpu else "cpu")
    report = dict(mode=args.mode, initial_sha256=initial_sha, **boundary,
        legal=True, fixed_unchanged=True, trace=controller.trace,
        topology_rebuilds=len(controller.rebuilds), topologies=controller.rebuilds,
        cache_refreshes=controller.cache_refreshes, gradient_audit=controller.gradient_audit,
        iterations=len(controller.trace), observations=term.observations, oracle=oracle_rows,
        final=_pack_eval_metrics(context.evaluate(x, y)), legal_sha256=digest(out / "legal.npz"),
        final_position_sha256=tensor_digest(final_pos), dreamplace_legal_sha256=digest(out / "dreamplace_legal.npz"),
        final_pin_offset_model="measured row orientations" if final_observation else "canonical static orientations",
        elapsed_s=time.time()-started, **observation_publication(oracle_rows, controller.trace))
    if final_observation:
        report["router"] = {k: final_observation[k] for k in ("actual_io", "actual_wirelength", "source_sha256", "placement_sha256",
            "native_congestion","routing_policy","router_elapsed_s","aggregated_resource_overflow",
            "preferred_layer_overflow","net_audit","routed_net_count","routed_net_names_sha256")}
    report["artifact_sha256"] = {str(path.relative_to(out)): digest(path) for path in out.rglob("*") if path.is_file()}
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print("DONE", args.mode, "steps", len(controller.trace), "rebuilds", len(controller.rebuilds), "legal", True, flush=True)


if __name__ == "__main__":
    main()
