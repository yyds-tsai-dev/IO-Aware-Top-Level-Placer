"""Evaluate the actual router input/output DEF, using the frozen S1 coordinates.

No placement is run. OpenDB supplies transformed pin bounding-box centers and
instance positions, without loading source Verilog; the S1 coord.json transform, not a fresh initialization, maps them
to the exact evaluator lattice. Net identities are matched by name.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ioplace.netlist import Netlist
from ioplace.export.evaluation import normalized_names, save_evaluation
from ioplace.route_eval.route_crossings import CoordMap, load_net_order, region_grid_from_json


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align_netlist(nl, actual_names, target_names, excluded_special=()):
    actual, target = normalized_names(actual_names), normalized_names(target_names)
    if len(actual) != nl.num_nets or len(set(actual)) != len(actual):
        raise ValueError("native net names are not a bijection")
    mapping = {name: i for i, name in enumerate(actual)}
    missing = set(target)-set(actual)
    if (set(actual)-set(target) or not missing.issubset(set(excluded_special))
            or len(set(target)) != len(target)):
        raise ValueError("router DEF and original netmap have different net sets (signal nets)")
    target_map = {name:i for i,name in enumerate(target)}
    inverse = np.asarray([target_map[name] for name in actual], dtype=np.int32)
    degrees = np.zeros(len(target), dtype=np.int32)
    degrees[inverse] = nl.net_degrees
    starts = np.r_[0, np.cumsum(degrees)].astype(np.int32)
    order = np.argsort(inverse)
    old_starts = nl.flat_net2pin_start[order]
    target_starts = np.r_[0,np.cumsum(nl.net_degrees[order])][:-1]
    indices = np.arange(len(nl.flat_net2pin))+np.repeat(old_starts-target_starts,nl.net_degrees[order])
    return replace(nl,pin2net=inverse[nl.pin2net],flat_net2pin=nl.flat_net2pin[indices],
                   flat_net2pin_start=starts)


def read_router_netlist(pin_geometry, routed_def, coord, names):
    with open(str(pin_geometry)+".json") as stream:
        meta = json.load(stream)
    if meta["def_sha256"] != sha(routed_def):
        raise ValueError("OpenDB pin geometry belongs to a different DEF")
    for path,digest in meta["lef_sha256"].items():
        if sha(path) != digest:
            raise ValueError("OpenDB pin geometry LEF changed")
    with np.load(pin_geometry,allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    die = meta["die_bbox"]
    nl = Netlist(node_x=data["node_x"],node_y=data["node_y"],node_size_x=data["node_w"],
        node_size_y=data["node_h"],num_movable=meta["num_movable"],num_terminals=meta["num_fixed"],
        num_terminal_NIs=meta["num_bterms"],pin_offset_x=data["pin_offset"][:,0],
        pin_offset_y=data["pin_offset"][:,1],pin2node=data["pin_node"],pin2net=data["pin_net"],
        flat_net2pin=data["net2pin"],flat_net2pin_start=data["net2pin_start"],
        xl=die[0],yl=die[1],xh=die[2],yh=die[3])
    nl = align_netlist(nl,data["net_names"],names,meta.get("excluded_special_net_names", []))
    scale = coord.scale_factor
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("S1 coordinate scale must be finite and positive")
    x, y = coord.to_internal(nl.node_x, nl.node_y)
    xl, yl = coord.to_internal(nl.xl, nl.yl)
    xh, yh = coord.to_internal(nl.xh, nl.yh)
    return replace(nl, node_x=x, node_y=y, node_size_x=nl.node_size_x * scale,
                   node_size_y=nl.node_size_y * scale,
                   pin_offset_x=nl.pin_offset_x * scale, pin_offset_y=nl.pin_offset_y * scale,
                   xl=float(xl), yl=float(yl), xh=float(xh), yh=float(yh))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "def", "regions", "netmap", "coord", "out"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--pin-geometry",required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-degree", type=int, default=256)
    args = ap.parse_args()
    names = load_net_order(args.netmap)
    rg = region_grid_from_json(args.regions)
    nl = read_router_netlist(args.pin_geometry, getattr(args, "def"), CoordMap.from_json(args.coord), names)
    if args.device == "cpu":
        from ioplace.evaluator_ref import evaluate
        result = evaluate(nl, nl.node_x, nl.node_y, rg, max_degree=args.max_degree)
    else:
        from ioplace.evaluator_gpu import GpuEvalContext
        result = GpuEvalContext(nl, rg, device=args.device, max_degree=args.max_degree).evaluate(nl.node_x, nl.node_y)
    meta = save_evaluation(args.out, nl, rg, result, nl.node_x, nl.node_y, names,
        max_degree=args.max_degree,
        provenance=dict(placement_stage="router_def", router_def=os.path.abspath(getattr(args, "def")),
            router_def_sha256=sha(getattr(args, "def")), config_sha256=sha(args.config),
            coord_sha256=sha(args.coord), netmap_sha256=sha(args.netmap),
            pin_geometry_sha256=sha(args.pin_geometry),
            pin_geometry_metadata=json.load(open(args.pin_geometry+".json"))))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
