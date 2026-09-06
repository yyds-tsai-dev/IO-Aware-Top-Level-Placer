"""Compare boundary demand on exactly the eligible routed-net population."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stage2_recompute_evaluator import read_router_netlist, sha
from ioplace.export.evaluation import boundary_statistics, load_evaluation, array_digest
from ioplace.route_eval.route_crossings import CoordMap, load_net_order, region_grid_from_json, evaluate_route_from_files


def filter_net_pins(nl, mask):
    """Retain original net-index space, with empty nets outside the mask."""
    indices=nl.flat_net2pin[np.repeat(mask,nl.net_degrees)]
    degrees=np.where(mask,nl.net_degrees,0)
    return replace(nl,pin2node=nl.pin2node[indices],pin2net=nl.pin2net[indices],
        pin_offset_x=nl.pin_offset_x[indices],pin_offset_y=nl.pin_offset_y[indices],
        flat_net2pin=np.arange(len(indices),dtype=np.int32),
        flat_net2pin_start=np.r_[0,np.cumsum(degrees)].astype(np.int32))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("crossings","evaluator","pin-geometry","routed-def","regions","coord","netmap","segments","out"):
        parser.add_argument("--"+flag,required=True)
    args=parser.parse_args()
    crossing=json.loads(Path(args.crossings).read_text())
    evidence=load_evaluation(args.evaluator)
    if crossing["evaluator_sha256"]!=sha(args.evaluator):
        raise ValueError("boundary evaluator digest mismatch")
    mask=np.asarray(crossing["per_net_calibration_eligible"],dtype=bool)
    if not crossing["unrouted_signal_gate_pass"]:
        mask[:]=False
    if not mask.any():
        Path(args.out).write_text(json.dumps(dict(not_evaluable=True,
            reason="no eligible population or unrouted-net gate failed",
            crossings_sha256=sha(args.crossings),evaluator_sha256=sha(args.evaluator)))+"\n")
        return
    names=load_net_order(args.netmap);rg=region_grid_from_json(args.regions)
    nl=read_router_netlist(args.pin_geometry,args.routed_def,CoordMap.from_json(args.coord),names)
    selected=filter_net_pins(nl,mask)
    from ioplace.evaluator_gpu import GpuEvalContext
    result=GpuEvalContext(selected,rg,device="cuda").evaluate(nl.node_x,nl.node_y)
    route=evaluate_route_from_files(args.segments,args.regions,args.netmap,args.coord,delta=2,net_mask=mask)
    expected=int(np.asarray(crossing["per_net_route_cross_dw"])[mask].sum())
    if route.total_route_cross_dw!=expected or result.io_count!=int(evidence["per_net_crossings"][mask].sum()):
        raise ValueError("eligible rerun changed per-net integer totals")
    ep,ev,lengths,es=boundary_statistics(rg,result.boundary_pair_demand)
    rp,rv,rl,rs=boundary_statistics(rg,route.route_pair_demand)
    if ep!=rp or not np.array_equal(lengths,rl):
        raise ValueError("boundary physical adjacency mismatch")
    output=dict(population="same eligible routed signal nets as formal calibration",n_nets=int(mask.sum()),
        mask_sha256=array_digest(mask),crossings_sha256=sha(args.crossings),
        evaluator_sha256=sha(args.evaluator),segments_sha256=sha(args.segments),
        region_sha256=evidence["metadata"]["region_sha256"],
        placement_sha256=evidence["metadata"]["placement_sha256"],
        pairs=[dict(pair=list(pair),length=float(length),evaluator=int(a),route=int(b))
               for pair,length,a,b in zip(ep,lengths,ev,rv)],
        evaluator_statistics=es,route_statistics=rs)
    Path(args.out).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
