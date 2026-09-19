"""Recompute frozen final placements for boundary-demand and seed-noise evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_ft_followup import read_run
from ioplace.netlist import load_netlist
from ioplace.drivers.run_placement import get_regions_for
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.export.evaluation import boundary_statistics, array_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    names=([f"A{i}_seed1000" for i in range(6)]
        +[f"{arm}_seed{seed}" for arm in ("A0","A2") for seed in (1001,1002,1003)]
        +[f"P{i}_seed1000" for i in range(4)]+[f"flat_seed{seed}" for seed in range(1000,1005)])
    first=read_run(root,names[0]);nl,db,params=load_netlist(first["config"])
    rg=RegionGrid(get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),first["k"],first["rtype"],first["seed"]))
    context=GpuEvalContext(nl,rg,device="cuda")
    output={}
    for name in names:
        metrics=read_run(root,name)
        coordinates=root/(name+".json.npz")
        with np.load(coordinates,allow_pickle=False) as data:
            x,y=data["node_x"],data["node_y"]
        result=context.evaluate(x,y)
        for key in ("io_count","ft_count","hard_lambda_sum","io_rg","ft_rg"):
            if key in metrics and int(getattr(result,key))!=metrics[key]:
                raise ValueError(f"{name} frozen evaluator mismatch: {key}")
        if not np.isclose(result.hpwl,metrics["hpwl"],rtol=1e-12):
            raise ValueError(f"{name} HPWL mismatch")
        pairs,demand,lengths,stats=boundary_statistics(rg,result.boundary_pair_demand)
        output[name]=dict(metrics_sha256=sha(root/(name+".json")),
            coordinates_sha256=sha(coordinates),placement_sha256=array_digest(x,y),
            scalars_verified=True,**stats,
            pairs=[dict(pair=list(pair),demand=int(value),length=float(length))
                   for pair,value,length in zip(pairs,demand,lengths)])
        print("verified",name,flush=True)
    noise={}
    paired={}
    for kind in ("boundary_demand","boundary_demand_per_length"):
        noise[kind]={};paired[kind]={}
        for key in ("total","max","mean","p90","gini"):
            values=[output[f"flat_seed{seed}"][kind][key] for seed in range(1000,1005)]
            noise[kind][key]=dict(values=values,mean=float(np.mean(values)),std_seed=float(np.std(values,ddof=1)))
            differences=np.array([output[f"A2_seed{seed}"][kind][key]-output[f"A0_seed{seed}"][kind][key]
                                  for seed in (1001,1002,1003)])
            paired[kind][key]=dict(differences=differences.tolist(),mean=float(differences.mean()),
                one_sided_95_upper=float(differences.mean()+t.ppf(.95,2)*differences.std(ddof=1)/np.sqrt(3)))
    result=dict(scope="adaptec1 K16; frozen final geometry recomputation",source_sha256=sha(Path(__file__)),
        runs=output,flat_seed_noise=noise,paired_A2_minus_A0=paired,
        interpretation="descriptive diagnostics; no new quality acceptance thresholds chosen from these results")
    (root/"boundary_probes.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
