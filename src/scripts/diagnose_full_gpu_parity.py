"""Preserved, per-net localization of the first full-scale integer parity failure."""
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[1]
REPO = REPO.parent if REPO.name == "src" else REPO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_full_gpu_evaluator import ROOT,CACHE,sha,write
from ioplace.bench.bookshelf_netlist import load_tiled_netlist
from ioplace.drivers.run_placement import get_regions_for,_pack_eval_metrics
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.evaluator_ref import evaluate
from ioplace.region_grid import RegionGrid
OUT=ROOT/'parity_diagnosis_v1'

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    reference_path=ROOT/'full_3x3_flat_k16.json'
    historical=json.loads(reference_path.read_text())
    protocol=dict(source_sha256={str(q):sha(q) for q in [Path(__file__),Path(__file__).resolve().parents[1] / 'ioplace/evaluator_gpu.py',Path(__file__).resolve().parents[1] / 'ioplace/evaluator_ref.py',Path(__file__).resolve().parents[1] / 'ioplace/region_grid.py']}, reference_sha256=sha(reference_path), first_failure_sha256=sha(ROOT/'posthoc_gpu_validation.json'), boundary_risk_tolerance=1e-4, selector='CPU region spread OR GPU IO/FT positive OR degree>256 OR internal region boundary distance<=1e-4', cpu_omitted_io_ft=0, historical_reconstruction_required=True)
    write(OUT/'protocol.json',protocol)
    result=dict(status='running',pid=os.getpid(),started=time.time(),iterations=[])
    write(OUT/'result.json',result)
    start=time.perf_counter()
    raw, _ = load_tiled_netlist(str(CACHE), mmap=True)
    with np.load(str(reference_path)+".npz", allow_pickle=False) as saved:
        x, y = saved["node_x"], saved["node_y"]
    if x.shape != (raw.num_physical,) or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("coordinate count/finite check failed")
    for positions, cache_positions, shift in ((x,raw.node_x,10260.),(y,raw.node_y,10080.)):
        fixed = np.array(cache_positions[raw.num_movable:], dtype=np.float32)
        fixed -= shift
        fixed *= 1/380.
        if not np.array_equal(positions[raw.num_movable:],fixed.astype(np.float64)):
            raise ValueError("fixed/NI tail geometry or identity mismatch")
    def scale(values):
        value = np.array(values, dtype=np.float32, copy=True)
        value *= 1/380.
        return value.astype(np.float64)
    nl = replace(raw,node_x=x,node_y=y,node_size_x=scale(raw.node_size_x),node_size_y=scale(raw.node_size_y),
        pin_offset_x=scale(raw.pin_offset_x),pin_offset_y=scale(raw.pin_offset_y),
        xl=0.,yl=0.,xh=(raw.xh-10260.)/380.,yh=(raw.yh-10080.)/380.)
    result.update(fixed_and_ni_tail_exact=True, die=[nl.xl,nl.yl,nl.xh,nl.yh],
                  load_convert_s_after_hash_reads=time.perf_counter()-start)
    regions = RegionGrid(get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),16,"grid",1000,lattice=512))

    context=GpuEvalContext(nl,regions,device='cuda',max_degree=256)
    keys=('per_net_crossings','per_net_ft','per_net_lambda','per_net_steiner','per_net_home')
    arrays=None
    for iteration in range(3):
        start=time.perf_counter()
        value=context.evaluate(x,y)
        torch.cuda.synchronize()
        current={key:np.array(getattr(value,key),copy=True) for key in keys}
        equal=True if arrays is None else all(np.array_equal(arrays[k],current[k]) for k in keys)
        result['iterations'].append(dict(iteration=iteration,seconds=time.perf_counter()-start,metrics=_pack_eval_metrics(value),equal_first_per_net=equal))
        if arrays is None:
            arrays=current
            for key,a in arrays.items(): np.save(OUT/(key+'.npy'),a)
        write(OUT/'result.json',result)
    del context,value,current
    gc.collect(); torch.cuda.empty_cache()
    degree=nl.net_degrees
    minimum=np.full(nl.num_nets,regions.k,dtype=np.int16)
    maximum=np.full(nl.num_nets,-1,dtype=np.int16)
    risk=np.zeros(nl.num_nets,dtype=bool)
    xb=(np.flatnonzero(np.any(np.diff(regions.grid,axis=1)!=0,axis=0))+1)*regions.cell_w+regions.die[0]
    yb=(np.flatnonzero(np.any(np.diff(regions.grid,axis=0)!=0,axis=1))+1)*regions.cell_h+regions.die[1]
    start=time.perf_counter()
    for lo in range(0,len(nl.pin2node),2_000_000):
        hi=min(lo+2_000_000,len(nl.pin2node))
        nodes=nl.pin2node[lo:hi]; nets=nl.pin2net[lo:hi]
        px=x[nodes]+nl.pin_offset_x[lo:hi]; py=y[nodes]+nl.pin_offset_y[lo:hi]
        rid=regions.region_of_points(px,py)
        np.minimum.at(minimum,nets,rid); np.maximum.at(maximum,nets,rid)
        near=np.zeros(hi-lo,dtype=bool)
        for b in xb: near |= np.abs(px-b)<=1e-4
        for b in yb: near |= np.abs(py-b)<=1e-4
        risk[nets[near]]=True
    select=((minimum!=maximum)|(arrays['per_net_crossings']>0)|(arrays['per_net_ft']>0)|(degree>256)|risk)&(degree>=2)
    ids=np.flatnonzero(select)
    selected_degree=degree[ids]
    starts=np.r_[0,np.cumsum(selected_degree,dtype=np.int64)]
    index=np.repeat(nl.flat_net2pin_start[ids]-starts[:-1],selected_degree)+np.arange(starts[-1])
    pins=nl.flat_net2pin[index]
    compact=replace(nl,pin_offset_x=nl.pin_offset_x[pins],pin_offset_y=nl.pin_offset_y[pins],pin2node=nl.pin2node[pins],pin2net=np.repeat(np.arange(len(ids),dtype=np.int32),selected_degree),flat_net2pin=np.arange(len(pins),dtype=np.int32),flat_net2pin_start=starts.astype(np.int32))
    result.update(selector_s=time.perf_counter()-start,selected_nets=len(ids),selected_pins=len(pins),boundary_risk_nets=int(risk.sum()),cpu_multi_region_nets=int(((minimum!=maximum)&(degree>=2)).sum()))
    np.save(OUT/'selected_net_ids.npy',ids)
    write(OUT/'result.json',result)
    print(json.dumps(result),flush=True)
    start=time.perf_counter()
    cpu=evaluate(compact,x,y,regions,max_degree=256)
    result['cpu_subset_seconds']=time.perf_counter()-start
    result['cpu_subset_metrics']=_pack_eval_metrics(cpu)
    result['historical_integer_reconstruction']={key:getattr(cpu,key)==historical[key] for key in ('io_count','ft_count','hard_lambda_sum','io_rg','ft_rg')}
    mismatch=np.zeros(len(ids),dtype=bool)
    for key in keys:
        a=getattr(cpu,key)
        np.save(OUT/('cpu_selected_'+key+'.npy'),a)
        mismatch |= a!=arrays[key][ids]
    bad_ids=ids[mismatch]
    result['mismatching_nets']=bad_ids.tolist()
    result['per_net_differences']=[dict(net=int(ids[i]),degree=int(degree[ids[i]]),cpu={k:int(getattr(cpu,k)[i]) for k in keys},gpu={k:int(arrays[k][ids[i]]) for k in keys}) for i in np.flatnonzero(mismatch)]
    # Tiny standalone geometry fixture retains every mismatching net's ordered pins.
    mask=np.repeat(mismatch,selected_degree)
    badpins=compact.pin2node[mask]
    np.savez(OUT/'mismatch_geometry.npz',net_ids=bad_ids,degrees=degree[bad_ids],px=x[badpins]+compact.pin_offset_x[mask],py=y[badpins]+compact.pin_offset_y[mask],node_ids=badpins,offset_x=compact.pin_offset_x[mask],offset_y=compact.pin_offset_y[mask],die=np.array(regions.die))
    result.update(status='completed',ended=time.time())
    write(OUT/'result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
