"""Apply route feedback to a legal checkpoint, preserving full-net gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from ioplace.drivers.run_placement import _load_dreamplace, _place, get_regions_for, _pack_eval_metrics
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.ops.route_feedback import RouteFeedback, closed_loop


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--placement',help='Saved legal node_x/node_y NPZ; otherwise run fresh GP+LG')
    p.add_argument('--out',required=True)
    p.add_argument('--k',type=int,default=32)
    p.add_argument('--rtype',choices=['slicing','grid'],default='slicing')
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--emit-def',action='store_true')
    p.add_argument('--proposal-mode',choices=['bend','cost_delta_swap'],default='cost_delta_swap')
    p.add_argument('--max-active',type=int,default=512)
    p.add_argument('--max-displacement-cells',type=float,default=8.)
    p.add_argument('--flute-sample',type=int,default=256)
    args=p.parse_args()
    if args.rounds<0 or args.flute_sample<0:p.error('nonnegative rounds/sample required')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    config=Path(args.config).resolve()
    from ioplace.paths import REPO_ROOT
    source=list((Path(REPO_ROOT)/'src/ioplace').rglob('*.py'))+[Path(__file__)]
    protocol=dict(argv=sys.argv,config=str(config),config_sha256=digest(config),
        placement=args.placement,placement_sha256=digest(args.placement) if args.placement else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started=time.time(),
        source_sha256={str(f):digest(f) for f in source},
        region_seed=0,placement_seed=args.seed,k=args.k,rtype=args.rtype,
        hpwl_budget=.05,route_budget=.05,fractions=[.25,.5,1.],rounds=args.rounds)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    params,db=_load_dreamplace(str(config))
    params.random_seed=args.seed;params.deterministic_flag=1
    db.initialize(params)
    import torch
    torch.set_num_threads(params.num_threads)
    if args.placement:
        import NonLinearPlace
        np.random.seed(args.seed)
        placer=NonLinearPlace.NonLinearPlace(params,db,None)
        with np.load(args.placement) as data:x,y=data['node_x'].copy(),data['node_y'].copy()
    else:
        placer,_=_place(params,db)
        x,y=db.node_x[:db.num_physical_nodes].copy(),db.node_y[:db.num_physical_nodes].copy()
    nl=netlist_from_placedb(db);n,nm,nall=nl.num_physical,nl.num_movable,db.num_nodes
    if (x.shape!=y.shape or x.shape!=(n,) or not np.array_equal(x[nm:],nl.node_x[nm:])
            or not np.array_equal(y[nm:],nl.node_y[nm:])):
        raise ValueError('checkpoint shape/fixed-node identity mismatch')
    rs=get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),args.k,args.rtype,0)
    rg=RegionGrid(rs)
    from ioplace.evaluator_gpu import GpuEvalContext
    ctx=GpuEvalContext(nl,rg,device='cuda' if params.gpu else 'cpu')
    template=placer.pos[0].detach().clone()
    def position(x,y):
        pos=template.clone()
        pos[:n]=torch.as_tensor(x,dtype=pos.dtype,device=pos.device)
        pos[nall:nall+n]=torch.as_tensor(y,dtype=pos.dtype,device=pos.device)
        return pos
    def legalize(x,y):
        with torch.no_grad():pos=placer.op_collections.legalize_op(position(x,y))
        return pos[:n].cpu().numpy().astype(float),pos[nall:nall+n].cpu().numpy().astype(float)
    def is_legal(x,y):
        with torch.no_grad():return bool(placer.op_collections.legality_check_op(position(x,y)))
    checkpoints=[]
    sample=np.flatnonzero((nl.net_degrees>=3)&(nl.net_degrees<=32))[:args.flute_sample]
    def checkpoint(label,x,y,metrics):
        directory=out/label;directory.mkdir()
        np.savez_compressed(directory/'placement.npz',node_x=x,node_y=y)
        row=dict(label=label,metrics=metrics,placement_sha256=digest(directory/'placement.npz'))
        if args.emit_def:
            from ioplace.export.def_export import export_def
            export_def(db,params,x,y,str(directory),rs)
            predicted=ctx.evaluate(x,y)
            np.savez_compressed(directory/'prediction.npz',per_net_crossings=predicted.per_net_crossings)
        from ioplace.route_eval.budgeted import route_net
        rows=[]
        for net in sample:
            pins=nl.flat_net2pin[nl.flat_net2pin_start[net]:nl.flat_net2pin_start[net+1]]
            px=x[nl.pin2node[pins]]+nl.pin_offset_x[pins]
            py=y[nl.pin2node[pins]]+nl.pin_offset_y[pins]
            if np.any(px<nl.xl) or np.any(px>nl.xh) or np.any(py<nl.yl) or np.any(py>nl.yh):
                rows.append(dict(net=int(net),excluded='outside_die'));continue
            item=dict(net=int(net))
            for topology in ('mst','flute'):
                route=route_net(feedback.router,px,py,topology=topology,shared_branches=True)
                item[topology]={key:{k:route[key][k] for k in ('crossings','wirelength')}
                                for key in ('baseline_union','union')}
            rows.append(item)
        (directory/'flute_sample.json').write_text(json.dumps(rows,indent=2))
        row['flute_sample_nets']=len(rows)
        checkpoints.append(row)
        (out/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2))
        print(label,json.dumps(metrics),flush=True)
    feedback=RouteFeedback(nl,rg,x,y,max_displacement_cells=args.max_displacement_cells)
    xx,yy,report=closed_loop(feedback,x,y,legalize=legalize,is_legal=is_legal,
        full_metrics=lambda a,b:_pack_eval_metrics(ctx.evaluate(a,b)),rounds=args.rounds,checkpoint=checkpoint,
        proposal_mode=args.proposal_mode,max_active=args.max_active)
    np.savez_compressed(out/'selected.npz',node_x=xx,node_y=yy)
    report.update(selected_sha256=digest(out/'selected.npz'),checkpoints=checkpoints,
                  elapsed_s=time.time()-protocol['started'])
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('DONE',report['accepted_rounds'],flush=True)


if __name__=='__main__':main()
