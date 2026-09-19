"""Dump placed OpenDB node and signal-pin geometry for evaluator alignment.

Run with the embedded OpenROAD Python, for example::

    openroad -python dump_pin_geometry.py --lef tech.lef --lef cells.lef \
      --def fixed.def --out pin_geometry.npz

The node order is movable instances, fixed instances, then BTerms.  Pin
coordinates come directly from OpenDB ``getBBox`` after LEF/DEF transforms;
no rectangle-center averaging or synthetic coordinates are used.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess

import numpy as np

import odb
import openroad


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bbox(box):
    if box is None:
        raise ValueError("OpenDB returned no pin/node bbox")
    values = (float(box.xMin()), float(box.yMin()),
              float(box.xMax()), float(box.yMax()))
    if not np.all(np.isfinite(values)) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"invalid OpenDB bbox: {values}")
    return values


def _node_record(name, kind, box, fixed):
    x0, y0, x1, y1 = _bbox(box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"non-positive bbox for node {name}: {(x0, y0, x1, y1)}")
    return {"name": name, "kind": kind, "x": x0, "y": y0,
            "w": x1 - x0, "h": y1 - y0, "fixed": bool(fixed)}


def _fixed_inst(inst):
    status = str(inst.getPlacementStatus()).upper()
    return bool(inst.isFixed()) or any(x in status for x in ("LOCKED", "FIRM", "COVER"))


def dump_pin_geometry(lefs, def_path):
    db = odb.dbDatabase.create()
    for lef in lefs:
        odb.read_lef(db, lef)
    odb.read_def(db.getTech(), def_path)
    block = db.getChip().getBlock()

    insts = list(block.getInsts())
    movable = [x for x in insts if not _fixed_inst(x)]
    fixed = [x for x in insts if _fixed_inst(x)]
    nodes = []
    node_id = {}
    for inst in movable + fixed:
        idx = len(nodes)
        node_id[inst.getName()] = idx
        rec = _node_record(inst.getName(), "inst", inst.getBBox(), _fixed_inst(inst))
        rec["orientation"] = str(inst.getOrient())
        rec["status"] = str(inst.getPlacementStatus())
        nodes.append(rec)
    for bterm in block.getBTerms():
        idx = len(nodes)
        node_id[("BTerm", bterm.getName())] = idx
        rec = _node_record(bterm.getName(), "bterm", bterm.getBBox(), True)
        rec["orientation"] = "NONE"
        rec["status"] = "FIXED"
        nodes.append(rec)

    pin_node, pin_net, pin_offset, pin_names = [], [], [], []
    net_names, net2pin_start, net2pin = [], [0], []
    endpoint_seen = set()
    special_names = [net.getName() for net in block.getNets() if net.isSpecial()]
    for net in block.getNets():
        if net.isSpecial():
            continue
        net_id = len(net_names)
        net_names.append(net.getName())
        endpoints = []
        for iterm in net.getITerms():
            inst_name = iterm.getInst().getName()
            endpoint = ("ITerm", inst_name, iterm.getMTerm().getName())
            node = node_id[inst_name]
            box = iterm.getBBox()
            x0, y0, x1, y1 = _bbox(box)
            n = nodes[node]
            endpoints.append(endpoint)
            pin_node.append(node); pin_net.append(net_id)
            pin_offset.append(((x0 + x1) / 2 - n["x"], (y0 + y1) / 2 - n["y"]))
            pin_names.append(json.dumps(endpoint, separators=(",", ":")))
        for bterm in net.getBTerms():
            endpoint = ("BTerm", bterm.getName())
            node = node_id[("BTerm", bterm.getName())]
            x0, y0, x1, y1 = _bbox(bterm.getBBox())
            n = nodes[node]
            endpoints.append(endpoint)
            pin_node.append(node); pin_net.append(net_id)
            pin_offset.append(((x0 + x1) / 2 - n["x"], (y0 + y1) / 2 - n["y"]))
            pin_names.append(json.dumps(endpoint, separators=(",", ":")))
        if len(set(endpoints)) != len(endpoints):
            raise ValueError(f"duplicate endpoint identity on net {net.getName()}")
        endpoint_seen.update((net_id, x) for x in endpoints)
        net2pin.extend(range(len(pin_node) - len(endpoints), len(pin_node)))
        net2pin_start.append(len(net2pin))

    die = block.getDieArea()
    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "def_path": def_path, "def_sha256": _sha256(def_path),
        "lef_paths": list(lefs), "lef_sha256": {p: _sha256(p) for p in lefs},
        "source_sha256": {"def": _sha256(def_path), "lef": {p: _sha256(p) for p in lefs}},
        "openroad_version": openroad.openroad_version(),
        "dbu_per_micron": block.getDbUnitsPerMicron(),
        "die_bbox": list(_bbox(die)), "num_movable": len(movable),
        "num_instances": len(insts), "num_physical": len(nodes), "num_fixed": len(fixed),
        "num_bterms": len(list(block.getBTerms())), "num_nodes": len(nodes),
        "num_pins": len(pin_node), "num_nets": len(net_names),
        "excluded_special_net_names": special_names,
        "pin_center_convention": "exact_half_DBU_center_of_transformed_union_bbox",
        "pin_center_note": "Uses OpenDB getBBox after LEF/DEF orientation transforms; "
                           "DREAMPlace integer-rounding conventions are not applied.",
    }
    return nodes, pin_node, pin_net, pin_offset, pin_names, net_names, net2pin_start, net2pin, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lef", action="append", required=True)
    ap.add_argument("--def", dest="def_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = dump_pin_geometry(args.lef, args.def_path)
    nodes, pin_node, pin_net, pin_offset, pin_names, net_names, starts, flat, meta = result
    def strings(values):
        if not values:
            return np.array([], dtype="<U1")
        return np.asarray(values, dtype=np.str_)
    np.savez_compressed(args.out, node_names=strings([n["name"] for n in nodes]),
                        node_kind=strings([n["kind"] for n in nodes]),
                        node_orientation=strings([n["orientation"] for n in nodes]),
                        node_status=strings([n["status"] for n in nodes]),
                        node_x=np.array([n["x"] for n in nodes], dtype=np.float64),
                        node_y=np.array([n["y"] for n in nodes], dtype=np.float64),
                        node_w=np.array([n["w"] for n in nodes], dtype=np.float64),
                        node_h=np.array([n["h"] for n in nodes], dtype=np.float64),
                        node_fixed=np.array([n["fixed"] for n in nodes], dtype=bool),
                        pin_node=np.array(pin_node, dtype=np.int32), pin_net=np.array(pin_net, dtype=np.int32),
                        pin_offset=np.array(pin_offset, dtype=np.float64),
                        pin_names=strings(pin_names), net_names=strings(net_names),
                        net2pin_start=np.array(starts, dtype=np.int32), net2pin=np.array(flat, dtype=np.int32))
    with open(args.out + ".json", "w") as f: json.dump(meta, f, indent=2); f.write("\n")
    print(f"wrote {args.out}: nodes={len(nodes)} pins={len(pin_node)} nets={len(net_names)}")


if __name__ == "__main__":
    main()
