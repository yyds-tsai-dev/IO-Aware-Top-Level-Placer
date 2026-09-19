"""Repair placement with OpenROAD detailed placement, without routing."""
import json
import math
from pathlib import Path
import sys

import numpy as np

try:
    from .checked_tcl import checked_eval
except ImportError:
    from checked_tcl import checked_eval


def repair_commands(bound_microns, threads):
    return [f"set_thread_count {int(threads)}",
            f"detailed_placement -max_displacement {{{int(bound_microns)} {int(bound_microns)}}}",
            "check_placement -verbose"]


def _replace_components(original_path, openroad_path, output_path):
    """Stream OpenROAD's repaired COMPONENTS into the otherwise original DEF."""
    with open(openroad_path) as raw:
        for line in raw:
            if line.startswith("COMPONENTS "):
                component_lines = [line]
                break
        else:
            raise ValueError("OpenROAD DEF has no COMPONENTS section")
        for line in raw:
            component_lines.append(line)
            if line.rstrip("\n") == "END COMPONENTS":
                break
        else:
            raise ValueError("OpenROAD DEF has unterminated COMPONENTS section")
    with open(original_path) as original, open(output_path, "w") as output:
        replaced = False
        skipping = False
        for line in original:
            if not replaced and line.startswith("COMPONENTS "):
                output.writelines(component_lines)
                replaced = skipping = True
                continue
            if skipping:
                if line.rstrip("\n") == "END COMPONENTS":
                    skipping = False
                continue
            output.write(line)
    if not replaced or skipping:
        raise ValueError("original DEF has invalid COMPONENTS section")


def _box(box):
    return box.xMin(), box.yMin(), box.xMax(), box.yMax()


def _immutable_snapshot(block, movable, path):
    """Stream every invariant while omitting allowed movable pose changes."""
    with open(path, "w") as stream:
        for inst in block.getInsts():
            master = inst.getMaster().getName()
            if inst.getName() in movable:
                stream.write(f"M\t{inst.getName()}\t{master}\n")
            else:
                x, y = inst.getOrigin()
                stream.write(f"F\t{inst.getName()}\t{master}\t{x}\t{y}\t"
                             f"{inst.getOrient()}\t{inst.getPlacementStatus()}\t"
                             f"{','.join(map(str, _box(inst.getBBox())))}\n")
        for net in block.getNets():
            if net.isSpecial():
                continue
            endpoints = [f"I:{it.getInst().getName()}/{it.getMTerm().getName()}"
                         for it in net.getITerms()]
            endpoints.extend(f"B:{bt.getName()}" for bt in net.getBTerms())
            stream.write(f"N\t{net.getName()}\t{','.join(sorted(endpoints))}\n")
        for bterm in block.getBTerms():
            for index, bpin in enumerate(bterm.getBPins()):
                boxes = []
                for box in bpin.getBoxes():
                    layer = box.getTechLayer()
                    layer_name = layer.getName() if layer is not None else ""
                    boxes.append(f"{layer_name}:{','.join(map(str, _box(box)))}")
                stream.write(f"B\t{bterm.getName()}\t{index}\t"
                             f"{bpin.getPlacementStatus()}\t{';'.join(sorted(boxes))}\n")


def _positions(block, names):
    x, y, orient = [], [], []
    for name in names:
        inst = block.findInst(name)
        if inst is None:
            raise ValueError(f"movable instance not found: {name}")
        bbox = inst.getBBox()
        x.append(bbox.xMin()); y.append(bbox.yMin()); orient.append(str(inst.getOrient()))
    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64), orient


def _full_design_bound_microns(block):
    units = block.getDbUnitsPerMicron()
    boxes = [block.getDieArea(), block.getCoreArea()]
    span = max(max(box.xMax() - box.xMin(), box.yMax() - box.yMin()) for box in boxes)
    return int(math.ceil(span / units))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    settings = json.loads(Path(argv[-1]).read_text())
    names = json.loads(Path(settings["movable_names"]).read_text())
    if len(names) != len(set(names)):
        raise ValueError("duplicate movable instance name")
    movable = set(names)
    import openroad
    tech = openroad.Tech()
    for lef in settings["lefs"]:
        tech.readLef(lef)
    verify_only = "--verify-only" in argv[:-1]
    design = openroad.Design(tech)
    design.readDef(settings["legalized_def"] if verify_only else settings["def"])
    block = design.getBlock()
    if verify_only:
        checked_eval(design, "check_placement -verbose", "persisted-placement")
        x, y, orientations = _positions(block, names)
        with np.load(settings["coordinates"], allow_pickle=False) as expected:
            if (expected["names"].astype(str).tolist() != names or
                    not np.array_equal(expected["x_dbu"], x) or
                    not np.array_equal(expected["y_dbu"], y) or
                    expected["orientations"].astype(str).tolist() != orientations):
                raise RuntimeError("persisted movable pose differs from repair output")
        _immutable_snapshot(block, movable, settings["identity_after"])
        np.savez_compressed(settings["coordinates"], names=np.asarray(names),
                            x_dbu=x, y_dbu=y, orientations=np.asarray(orientations))
        return
    if len(movable) > len(block.getInsts()):
        raise ValueError("more movable names than instances")
    before_x, before_y, before_orient = _positions(block, names)
    _immutable_snapshot(block, movable, settings["identity_before"])
    print("IOPLACE_EXPECTED_PRECHECK_BEGIN", flush=True)
    before_code = int(design.evalTclString("catch {check_placement}"))
    print(f"IOPLACE_EXPECTED_PRECHECK_END code={before_code}", flush=True)
    bound = _full_design_bound_microns(block)
    for index, command in enumerate(repair_commands(bound, settings["threads"])):
        checked_eval(design, command, f"placement-{index}")
    checked_eval(design, f"write_def {{{settings['openroad_legalized_def']}}}",
                 "openroad-legalized-def")
    after_x, after_y, after_orient = _positions(block, names)
    _replace_components(settings["def"], settings["openroad_legalized_def"],
                        settings["legalized_def"])
    np.savez_compressed(settings["coordinates"], names=np.asarray(names),
                        x_dbu=after_x, y_dbu=after_y,
                        orientations=np.asarray(after_orient))
    displacement = np.abs(after_x - before_x) + np.abs(after_y - before_y)
    units = block.getDbUnitsPerMicron()
    report = {
        "before_check_code": before_code,
        "displacement_bound_microns": bound,
        "movable_count": len(names),
        "instance_count": len(block.getInsts()),
        "changed_location_count": int(np.count_nonzero(displacement)),
        "changed_orientation_count": sum(a != b for a, b in zip(before_orient, after_orient)),
        "average_displacement_dbu": float(displacement.mean()) if len(displacement) else 0.,
        "maximum_displacement_dbu": int(displacement.max()) if len(displacement) else 0,
        "average_displacement_microns": float(displacement.mean() / units) if len(displacement) else 0.,
        "maximum_displacement_microns": float(displacement.max() / units) if len(displacement) else 0.,
        "units_distance_microns": units,
        "openroad_version": openroad.openroad_version(),
    }
    Path(settings["report"]).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
