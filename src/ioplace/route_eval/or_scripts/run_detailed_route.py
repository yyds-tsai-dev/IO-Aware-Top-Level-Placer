"""Run full OpenROAD global+detailed routing and emit auditable artifacts.

This file runs under ``openroad -python``.  Detailed routing deliberately uses
the tool's default iteration limit; no shortened ``droute_end_iter`` is set.
"""
import json
import os
from pathlib import Path
import sys

try:
    from .checked_tcl import checked_eval
except ImportError:  # Standalone ``openroad -python path/to/script.py``.
    from checked_tcl import checked_eval


def _tcl_path(path):
    path = str(path)
    if any(char in path for char in "{}\n\r"):
        raise ValueError("unsupported Tcl path")
    return "{" + path + "}"


def route_commands(settings):
    """Commands kept explicit so the routing policy is independently testable."""
    return [
        "check_placement -verbose",
        f"source {_tcl_path(settings['clear_script'])}",
        f"ioplace_clear_signal_routing $block {_tcl_path(settings['cleared'])}",
        f"write_def {_tcl_path(settings['routing_input'])}",
        f"set_thread_count {int(settings['threads'])}",
        "global_route -allow_congestion "
        f"-congestion_report_file {_tcl_path(settings['congestion'])} "
        f"-guide_file {_tcl_path(settings['guide'])}",
        f"detailed_route -output_drc {_tcl_path(settings['drc'])} -verbose 1",
        f"write_def {_tcl_path(settings['routed_def'])}",
    ]


def count_drc(text):
    return sum(line.lstrip().lower().startswith("violation type:")
               for line in text.splitlines())


def _routing_status(block):
    trivial, nontrivial = 0, []
    for net in block.getNets():
        if net.isSpecial() or net.getWire() is not None:
            continue
        terminals = len(net.getITerms()) + len(net.getBTerms())
        if terminals >= 2:
            nontrivial.append(net.getName())
        else:
            trivial += 1
    return {"unwired_net_count": trivial + len(nontrivial),
            "unwired_trivial_net_count": trivial,
            "unrouted_nontrivial_net_count": len(nontrivial),
            "unrouted_nontrivial_net_names": nontrivial}


def _snapshot(block, path):
    """Stream cells, connectivity, and fixed terminal geometry."""
    with open(path, "w") as stream:
        for inst in block.getInsts():
            x, y = inst.getOrigin()
            master = inst.getMaster().getName()
            stream.write(f"I\t{inst.getName()}\t{master}\t{x}\t{y}\t"
                         f"{inst.getOrient()}\t{inst.getPlacementStatus()}\n")
        for net in block.getNets():
            if not net.isSpecial():
                endpoints = [f"I:{iterm.getInst().getName()}/{iterm.getMTerm().getName()}"
                             for iterm in net.getITerms()]
                endpoints.extend(f"B:{bterm.getName()}" for bterm in net.getBTerms())
                stream.write(f"N\t{net.getName()}\t{','.join(sorted(endpoints))}\n")
        for bterm in block.getBTerms():
            for index, bpin in enumerate(bterm.getBPins()):
                boxes = []
                for box in bpin.getBoxes():
                    layer = box.getTechLayer()
                    layer_name = layer.getName() if layer is not None else ""
                    boxes.append(f"{layer_name}:{box.xMin()},{box.yMin()},"
                                 f"{box.xMax()},{box.yMax()}")
                stream.write(f"B\t{bterm.getName()}\t{index}\t"
                             f"{bpin.getPlacementStatus()}\t{';'.join(sorted(boxes))}\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: openroad -python run_detailed_route.py settings.json")
    settings = json.loads(Path(argv[0]).read_text())

    import openroad
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dump_segments

    tech = openroad.Tech()
    for lef in settings["lefs"]:
        tech.readLef(lef)
    design = openroad.Design(tech)
    design.readDef(settings["def"])
    block = design.getBlock()
    _snapshot(block, settings["identity_before"])
    component_count = len(block.getInsts())
    net_count = sum(not net.isSpecial() for net in block.getNets())
    checked_eval(design, "set block [[[ord::get_db] getChip] getBlock]", "get-block")
    for index, command in enumerate(route_commands(settings)):
        checked_eval(design, command, f"route-{index}")
    _snapshot(block, settings["identity_after"])

    native_wirelength = 0
    for net in block.getNets():
        if net.isSpecial():
            continue
        wire = net.getWire()
        if wire is not None:
            native_wirelength += int(wire.getLength())

    rows, names, has_wire, layers, vias, segment_meta = dump_segments.dump_segments(
        settings["lefs"], settings["routed_def"])
    dump_segments.write_outputs(settings["segments"], rows, names, has_wire,
                                layers, vias, segment_meta, settings["segments_json"])
    decoded = sum(abs(x1 - x0) + abs(y1 - y0)
                  for _, kind, _, x0, y0, x1, y1, *_ in rows
                  if kind == dump_segments.KIND_WIRE)
    rect = sum(max(abs(x1 - x0), abs(y1 - y0)) -
               min(abs(x1 - x0), abs(y1 - y0))
               for _, kind, _, x0, y0, x1, y1, *_ in rows
               if kind == dump_segments.KIND_RECT)
    drc_text = Path(settings["drc"]).read_text() if Path(settings["drc"]).exists() else ""
    metrics = {
        "drc_count": count_drc(drc_text),
        "native_wirelength_dbu": native_wirelength,
        "decoded_wirelength_dbu": decoded,
        "rect_reconciliation_dbu": rect,
        "component_count": component_count,
        "net_count": net_count,
        "routed_net_count": segment_meta["routed_net_count"],
        "unrouted_net_count": segment_meta["unrouted_net_count"],
        "units_distance_microns": segment_meta["units_distance_microns"],
        "openroad_version": segment_meta["openroad_version"],
        **_routing_status(block),
    }
    Path(settings["metrics"]).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
