"""Paired WA / paper Eq.7 global placement, legalization, and optional GRT."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from ioplace.drivers.run_placement import _load_dreamplace,extract_final_positions,get_regions_for,_pack_eval_metrics
from ioplace.dp_hook import attach_terms,assert_optimizer_lock
from ioplace.netlist import netlist_from_placedb
from ioplace.ops.steiner_wirelength import FrozenSteinerWirelength
from ioplace.ops.steiner_gp import SteinerGPController,tensor_digest
from ioplace.route_eval.online_openroad import digest,run_openroad,load_observation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--mode',choices=['wa','wa_standard','paper'],default='paper')
    parser.add_argument('--iterations',type=int,default=500);parser.add_argument('--start',type=int,default=200)
    parser.add_argument('--rebuild',type=int,default=1);parser.add_argument('--seed',type=int)
    parser.add_argument('--stop-overflow',type=float);parser.add_argument('--openroad')
    args=parser.parse_args()
    if args.iterations<=args.start or args.start<1 or args.rebuild<1:raise ValueError('need iterations > start >= 1')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time();params,db=_load_dreamplace(str(Path(args.config).resolve()))
    if len(params.global_place_stages)!=1 or params.global_place_stages[0]['wirelength']!='weighted_average':
        raise ValueError('paper Eq.7 requires one weighted-average GP stage')
    params.global_place_stages[0]['iteration']=args.iterations
    params.global_place_stages[0]['Lsub_iteration']=1
    params.use_bb=0
    if args.seed is not None:params.random_seed=args.seed
    if args.stop_overflow is not None:params.stop_overflow=args.stop_overflow
    db.initialize(params);assert_optimizer_lock(params)
    if (len(db.regions)>0 or params.routability_opt_flag or params.timing_opt_flag
            or params.macro_place_flag):
        raise ValueError('paper GP runner requires flat, static-pin wirelength placement')
    torch.set_num_threads(params.num_threads);np.random.seed(params.random_seed)
    import NonLinearPlace,PlaceObj,NesterovAcceleratedGradientOptimizer
    placer=NonLinearPlace.NonLinearPlace(params,db,None)
    nl=netlist_from_placedb(db)
    term=FrozenSteinerWirelength(nl,db.num_nodes,
        net_mask=placer.data_collections.net_mask_ignore_large_degrees,
        net_weights=placer.data_collections.net_weights)
    controller=SteinerGPController(term,placer,enabled=args.mode=='paper',start=args.start,
        rebuild_every=args.rebuild,match_refresh=args.mode!='wa_standard')
    attach_terms(params,[controller]);placer.iteration_callback=controller.callback
    fixed_x=nl.node_x[nl.num_movable:].copy();fixed_y=nl.node_y[nl.num_movable:].copy()
    initial_sha=tensor_digest(placer.pos[0]);n=nl.num_physical;nall=db.num_nodes
    def save_positions(path,pos):
        values=pos.detach().cpu().numpy()
        np.savez_compressed(path,node_x=values[:n],node_y=values[nall:nall+n])
    save_positions(out/'initial.npz',placer.pos[0])
    from ioplace.paths import REPO_ROOT
    sources=list((Path(REPO_ROOT)/'src/ioplace').rglob('*.py'))+list((Path(REPO_ROOT)/'src/ioplace').rglob('*.cpp'))+[Path(__file__)]
    sources.extend(Path(m.__file__) for m in (NonLinearPlace,PlaceObj,NesterovAcceleratedGradientOptimizer))
    protocol=dict(argv=sys.argv,source_sha256={str(p):digest(p) for p in sources},
        config_sha256=digest(args.config),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        settings=vars(args),effective_seed=params.random_seed,initial_sha256=initial_sha,
        effective_stop_overflow=params.stop_overflow,device=str(placer.pos[0].device))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    original_legalize=placer.op_collections.legalize_op
    boundary={}
    def legalize(pos):
        save_positions(out/'gp.npz',pos)
        boundary['gp_sha256']=tensor_digest(pos)
        boundary['gp_hpwl']=float(placer.op_collections.hpwl_op(pos).detach())
        result=original_legalize(pos)
        return result
    placer.op_collections.legalize_op=legalize
    metrics=placer(params,db,params.global_place_stages[0]['learning_rate'])
    x,y=extract_final_positions(placer,db)
    np.savez_compressed(out/'legal.npz',node_x=x,node_y=y)
    legal=bool(placer.op_collections.legality_check_op(placer.pos[0]))
    fixed_unchanged=np.array_equal(x[nl.num_movable:],fixed_x) and np.array_equal(y[nl.num_movable:],fixed_y)
    if not legal or not fixed_unchanged:raise RuntimeError('GP+LG produced illegal or changed fixed placement')
    if args.mode=='paper' and not controller.rebuilds:raise RuntimeError('GP stopped before paper objective activated')
    rs=get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),32,'slicing',0)
    from ioplace.region_grid import RegionGrid
    from ioplace.evaluator_gpu import GpuEvalContext
    context=GpuEvalContext(nl,RegionGrid(rs),device='cuda' if params.gpu else 'cpu')
    report=dict(mode=args.mode,initial_sha256=initial_sha,**boundary,
        legal=legal,fixed_unchanged=bool(fixed_unchanged),trace=controller.trace,
        topology_rebuilds=len(controller.rebuilds),topologies=controller.rebuilds,
        cache_refreshes=controller.cache_refreshes,
        gradient_audit=controller.gradient_audit,
        iterations=len(controller.trace),elapsed_s=time.time()-started,
        final=_pack_eval_metrics(context.evaluate(x,y)),legal_sha256=digest(out/'legal.npz'))
    if args.openroad:
        from ioplace.export.def_export import export_def
        exported=out/'routed';exported.mkdir()
        export_def(db,params,x,y,str(exported),rs)
        settings=json.loads(Path(args.config).read_text())
        run_openroad(exported/'out.def',settings['lef_input'],exported/'grt',args.openroad)
        observed=load_observation(exported/'grt',exported/'coord.json',exported/'regions.json',exported/'netmap.json')
        report['router']={k:observed[k] for k in ('actual_io','actual_wirelength','source_sha256','placement_sha256')}
        report['router']['net_ids']=sorted(observed['net_io'])
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('DONE',args.mode,'steps',len(controller.trace),'rebuilds',len(controller.rebuilds),'legal',legal,flush=True)


if __name__=='__main__':main()
