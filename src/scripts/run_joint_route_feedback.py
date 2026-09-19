"""Online FLUTE/shared-capacity placement with measured OpenROAD feedback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from ioplace.drivers.run_placement import _load_dreamplace,get_regions_for,_pack_eval_metrics
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.export.def_export import export_def
from ioplace.route_eval.online_openroad import run_openroad,load_observation,digest
from ioplace.route_eval.joint import JointRoutingState
from ioplace.ops.joint_route_feedback import JointRouteFeedback,online_loop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--placement',required=True)
    parser.add_argument('--openroad',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--max-active',type=int,default=16)
    parser.add_argument('--neighbors',type=int,default=8)
    parser.add_argument('--max-displacement-cells',type=float,default=32.)
    parser.add_argument('--no-learn',action='store_true')
    parser.add_argument('--congestion-weight',type=float,default=.1)
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    config=Path(args.config).resolve();settings=json.loads(config.read_text())
    if not settings.get('lef_input') or not settings.get('def_input'):
        raise ValueError('online GRT requires physical LEF/DEF inputs')
    started=time.time()
    from ioplace.paths import REPO_ROOT
    sources=list((Path(REPO_ROOT)/'src/ioplace').rglob('*.py'))+list((Path(REPO_ROOT)/'src/ioplace').rglob('*.cpp'))+[Path(__file__)]
    protocol=dict(argv=sys.argv,source_sha256={str(p):digest(p) for p in sources},
        config_sha256=digest(config),placement_sha256=digest(args.placement),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),settings=vars(args),started=started)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    params,db=_load_dreamplace(str(config));db.initialize(params)
    import torch
    import NonLinearPlace
    torch.set_num_threads(params.num_threads);np.random.seed(params.random_seed)
    placer=NonLinearPlace.NonLinearPlace(params,db,None)
    nl=netlist_from_placedb(db);n,nm,nall=nl.num_physical,nl.num_movable,db.num_nodes
    with np.load(args.placement) as data:x,y=data['node_x'].astype(float),data['node_y'].astype(float)
    if (x.shape!=(n,) or y.shape!=x.shape or not np.array_equal(x[nm:],nl.node_x[nm:])
            or not np.array_equal(y[nm:],nl.node_y[nm:])):raise ValueError('checkpoint/fixed-node identity mismatch')
    rs=get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),32,'slicing',0);rg=RegionGrid(rs)
    template=placer.pos[0].detach().clone()
    def is_legal(a,b):
        with torch.no_grad():
            pos=template.clone()
            pos[:n]=torch.as_tensor(a,device=pos.device,dtype=pos.dtype)
            pos[nall:nall+n]=torch.as_tensor(b,device=pos.device,dtype=pos.dtype)
            return bool(placer.op_collections.legality_check_op(pos))
    from ioplace.evaluator_gpu import GpuEvalContext
    context=GpuEvalContext(nl,rg,device='cuda' if params.gpu else 'cpu')
    full=lambda a,b:_pack_eval_metrics(context.evaluate(a,b))
    expected_nets=None;resource_edges=None;oracle_rows=[]
    def oracle(label,a,b,state):
        nonlocal expected_nets,resource_edges
        directory=out/label;directory.mkdir()
        np.savez_compressed(directory/'placement.npz',node_x=a,node_y=b)
        export_def(db,params,a,b,str(directory),rs)
        run_openroad(directory/'out.def',settings['lef_input'],directory/'grt',args.openroad)
        observation=load_observation(directory/'grt',directory/'coord.json',directory/'regions.json',directory/'netmap.json')
        identities=set(observation['net_io'])
        edges=(observation['grid'].x_edges,observation['grid'].y_edges)
        if expected_nets is None:
            expected_nets=identities;resource_edges=edges
        elif identities!=expected_nets or not all(np.array_equal(a,b) for a,b in zip(edges,resource_edges)):
            raise ValueError('router net cohort or resource grid changed across checkpoints')
        row=dict(label=label,actual_io=observation['actual_io'],actual_wirelength=observation['actual_wirelength'],
                 source_sha256=observation['source_sha256'],placement_sha256=digest(directory/'placement.npz'))
        oracle_rows.append(row)
        (out/'oracle.json').write_text(json.dumps(oracle_rows,indent=2))
        print('ROUTER',label,row['actual_io'],flush=True)
        return observation
    baseline=oracle('baseline',x,y,None)
    state=JointRoutingState(rg,baseline['grid'],baseline['capacity'],congestion_weight=args.congestion_weight)
    feedback=JointRouteFeedback(nl,rg,x,y,state=state,max_displacement_cells=args.max_displacement_cells)
    print('FLUTE_COHORT',json.dumps(feedback.cohort),flush=True)
    (out/'cohort.json').write_text(json.dumps(feedback.cohort,indent=2))
    def checkpoint(iteration,a,b,state,record):
        directory=out/f'round{iteration}_retained';directory.mkdir()
        np.savez_compressed(directory/'placement.npz',node_x=a,node_y=b)
        np.savez_compressed(directory/'resources.npz',capacity=state.capacity,demand=state.demand,
            background=state.background,edge_prices=state.edge_prices)
        ids=sorted(state.routes)
        np.savez_compressed(directory/'flute.npz',net_ids=np.asarray(ids),
            crossings=np.array([state.routes[n].crossings for n in ids]),
            wirelength=np.array([state.routes[n].wirelength for n in ids]),
            weights=np.array([state.net_weights.get(n,1.) for n in ids]))
        (directory/'round.json').write_text(json.dumps(record,indent=2))
        np.testing.assert_array_equal(state.demand,state.recompute_demand())
        print('ROUND',iteration,'generation',record['generation'],'accepted',record['router_accepted'],flush=True)
    xx,yy,report=online_loop(feedback,x,y,oracle=oracle,is_legal=is_legal,full_metrics=full,
        rounds=args.rounds,max_active=args.max_active,neighbors=args.neighbors,learn=not args.no_learn,
        baseline_observation=baseline,checkpoint=checkpoint)
    np.savez_compressed(out/'selected.npz',node_x=xx,node_y=yy)
    report.update(selected_sha256=digest(out/'selected.npz'),elapsed_s=time.time()-started,
                  original_legacy=full(x,y),final_legacy=full(xx,yy),oracle=oracle_rows)
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('DONE',report['accepted_rounds'],report['original_actual_io'],report['final_actual_io'],flush=True)


if __name__=='__main__':main()
