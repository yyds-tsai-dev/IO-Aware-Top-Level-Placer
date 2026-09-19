"""Measure bounded active M5 work on an exact1M/10M/30M synthetic netlist."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes",type=int,choices=[1000000,10000000,30000000],required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("preserve existing scale observation")
    sys.path.insert(0,str(args.source.resolve()))
    import numpy as np
    import torch
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.drivers.run_placement import get_regions_for
    from ioplace.region_grid import RegionGrid
    from ioplace.evaluator_gpu import GpuEvalContext
    from ioplace.ops.discrete_placement import prepare_discrete,decode_placement,refine_placement
    from ioplace.bench.spike_30m import device_snapshot
    torch.set_num_threads(8)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    record=dict(status="running",pid=os.getpid(),started=time.time(),N=args.nodes,
        E=6*args.nodes//5,P=24*args.nodes//5,K=16,max_active=65536,
        benchmark_kind="synthetic",generator_verified=False,quality_claims_prohibited=True,
        device_before=device_snapshot(),source_snapshot=str(args.source.resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def persist():
        record["host_peak_rss_gib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        temp=args.out.with_suffix(".tmp");temp.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n");temp.replace(args.out)
    persist()
    try:
        started=time.perf_counter()
        nl=exact_synthetic(record["N"],record["E"],record["P"],seed=0,layout="local")
        record["generate_s"]=time.perf_counter()-started
        rs=get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),16,"grid",0);rg=RegionGrid(rs)
        torch.cuda.reset_peak_memory_stats()
        context=GpuEvalContext(nl,rg,device="cuda")
        baseline=context.evaluate(nl.node_x,nl.node_y)
        record["baseline"]=dict(io_count=baseline.io_count,ft_count=baseline.ft_count,hpwl=baseline.hpwl)
        prepared=prepare_discrete(nl,rs,nl.node_x,nl.node_y,max_active=65536)
        record["preparation"]=prepared["diagnostics"];persist()
        x,y,ce=decode_placement(prepared,nl.node_x,nl.node_y,baseline.io_count,baseline.ft_count)
        record["ce"]=dict(decode_s=ce["decode_s"],objective_initial=float(ce["objective_trace"][0]),
            objective_final=float(ce["objective_trace"][-1]),weighted_identity_max_error=ce["weighted_identity_max_error"],
            provenance=ce["provenance"]);persist()
        x,y,refine=refine_placement(prepared,x,y,baseline.io_count,baseline.ft_count)
        record["refine"]={key:refine[key] for key in ("refine_s","moves_accepted","swaps_attempted","swaps_accepted")}
        final=context.evaluate(x,y)
        record.update(status="completed",ended=time.time(),
            process_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            process_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            final_unlegalized=dict(io_count=final.io_count,ft_count=final.ft_count,hpwl=final.hpwl),
            device_after=device_snapshot(),physical_legality="not_evaluated_synthetic_component_scale_only")
    except Exception as error:
        record.update(status="failed",error=repr(error),ended=time.time());raise
    finally:persist()


if __name__=="__main__":main()
