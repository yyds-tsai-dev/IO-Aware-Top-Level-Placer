"""Independent OpenROAD GRT on exported closed-loop checkpoints.

Reports actual global-route centerline crossings separately from geometric
opportunity. Layers remain distinct; overlapping branches merge per net/layer.
The router receives no partition information. Detailed routing is not claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from ioplace.regions import RegionSet
from ioplace.region_grid import RegionGrid
from ioplace.route_eval.topology import union_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tcl_path(path):
    value=str(Path(path).resolve())
    if any(c in value for c in '{}\n\r'):raise ValueError('unsupported Tcl path')
    return '{'+value+'}'


def parse_segments(path,coord,rg):
    nets={};name=None;opened=False
    for raw in Path(path).read_text().splitlines():
        line=raw.strip()
        if not line:continue
        if line=='(':opened=True;continue
        if line==')':opened=False;name=None;continue
        if not opened:
            name=line
            if name in nets:raise ValueError('duplicate routed net')
            nets[name]={}
        else:
            fields=line.split()
            if len(fields)!=6:raise ValueError('invalid global route segment')
            x0,y0,l0,x1,y1,l1=fields
            if l0!=l1:
                if (x0,y0)!=(x1,y1):raise ValueError('nonzero lateral via segment')
                continue
            segment=(np.array([[int(x0),int(y0)],[int(x1),int(y1)]],dtype=float)
                     -np.asarray(coord['shift_factor']))*coord['scale_factor']
            nets[name].setdefault(l0,[]).append(segment)
    if opened:raise ValueError('unterminated route')
    result={}
    for name,layers in nets.items():
        io=0;length=0.;outside=0
        for segments in layers.values():
            segments=np.asarray(segments)
            metric=union_metrics(rg,segments)
            io+=metric['crossings'];length+=metric['wirelength']
            outside+=int(np.count_nonzero(np.any((segments<rg.die[:2])|(segments>rg.die[2:]),axis=2)))
        result[name]=dict(io=io,wirelength=length,outside_region_die_endpoints=outside)
    return result


def select_router_checkpoint(summary):
    """Only evaluator-accepted checkpoints can pass the independent veto."""
    baseline=summary['baseline']
    candidates=[]
    for label,metric in summary.items():
        if not label.endswith('_accepted'):continue
        control=summary[label.replace('_accepted','_control')]
        if (metric['common_net_io'] < min(baseline['common_net_io'],control['common_net_io'])
                and metric['in_die_common_io'] < min(baseline['in_die_common_io'],control['in_die_common_io'])
                and metric['common_net_wirelength'] <= baseline['common_net_wirelength']*1.05):
            candidates.append(label)
    return min(candidates,key=lambda label:(summary[label]['common_net_io'],
                    summary[label]['common_net_wirelength'])) if candidates else 'baseline'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--config',required=True)
    p.add_argument('--openroad',required=True);p.add_argument('--reuse-verified',action='store_true');args=p.parse_args()
    root=Path(args.run).resolve();config=json.loads(Path(args.config).read_text())
    rows=json.loads((root/'checkpoints.json').read_text())
    from ioplace.paths import REPO_ROOT
    clear=Path(REPO_ROOT)/'src/ioplace/route_eval/or_scripts/clear_signal_routing.tcl'
    all_metrics={};receipts=[]
    for row in rows:
        directory=root/row['label'];output=directory/'grt'
        reuse=output.exists() and args.reuse_verified
        output.mkdir(exist_ok=reuse)
        coord=json.loads((directory/'coord.json').read_text())
        rg=RegionGrid(RegionSet.from_json(directory/'regions.json'))
        script=output/'route.tcl'
        commands=['set_thread_count 4']
        commands += ['read_lef '+tcl_path(lef) for lef in config['lef_input']]
        commands += ['read_def '+tcl_path(directory/'out.def'),
            'source '+tcl_path(clear),
            'ioplace_clear_signal_routing [ord::get_db_block] '+tcl_path(output/'cleared.txt'),
            'check_placement -verbose',
            'global_route -congestion_iterations 50 -verbose',
            'grt::write_segments '+tcl_path(output/'segments.txt')]
        expected_script='\n'.join(commands)+'\n'
        if reuse:
            receipt=json.loads((output/'receipt.json').read_text())
            if (receipt['returncode'] or receipt['input_def_sha256']!=digest(directory/'out.def')
                    or receipt['openroad_sha256']!=digest(args.openroad)
                    or receipt['lef_sha256']!={f:digest(f) for f in config['lef_input']}
                    or receipt['script_sha256']!=digest(script)
                    or script.read_text()!=expected_script
                    or receipt['segments_sha256']!=digest(output/'segments.txt')):
                raise ValueError('existing route receipt no longer matches inputs/outputs')
        else:
            script.write_text(expected_script)
            command=[args.openroad,'-exit',str(script)]
            with (output/'run.log').open('w') as log:
                run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            receipt=dict(label=row['label'],returncode=run.returncode,command=command,
                input_def_sha256=digest(directory/'out.def'),openroad_sha256=digest(args.openroad),
                lef_sha256={f:digest(f) for f in config['lef_input']},script_sha256=digest(script))
            (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
            if run.returncode:raise RuntimeError(f'OpenROAD failed: {output}/run.log')
        metrics=parse_segments(output/'segments.txt',coord,rg)
        if not metrics:raise ValueError('empty GRT result')
        receipt['segments_sha256']=digest(output/'segments.txt')
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
        all_metrics[row['label']]=metrics;receipts.append(receipt)
        (output/'per_net.json').write_text(json.dumps(metrics,indent=2))
        print('ROUTED',row['label'],len(metrics),flush=True)
    common=set.intersection(*(set(nets) for nets in all_metrics.values()))
    summary={}
    for label,nets in all_metrics.items():
        summary[label]=dict(common_net_io=sum(nets[n]['io'] for n in common),
            common_net_wirelength=sum(nets[n]['wirelength'] for n in common),
            routed_nets=len(nets),excluded_not_common=len(set(nets)-common),
            outside_region_die_endpoints=sum(nets[n]['outside_region_die_endpoints'] for n in common))
    in_die_common = {n for n in common if all(nets[n]['outside_region_die_endpoints']==0
                                              for nets in all_metrics.values())}
    for label,nets in all_metrics.items():
        summary[label]['in_die_common_io'] = sum(nets[n]['io'] for n in in_die_common)
        summary[label]['in_die_common_wirelength'] = sum(nets[n]['wirelength'] for n in in_die_common)
    for label in summary:
        path=root/label/'prediction.npz'
        if path.exists():
            netmap=json.loads((root/label/'netmap.json').read_text())
            with np.load(path) as archive:counts=archive['per_net_crossings']
            by_name={name:int(counts[int(i)]) for i,name in netmap.items()}
            if not common <= set(by_name):raise ValueError('routed net missing from prediction identity')
            summary[label]['predicted_legacy_common_io']=sum(by_name[n] for n in common)
            summary[label]['predicted_legacy_in_die_io']=sum(by_name[n] for n in in_die_common)
    baseline=summary['baseline']
    for row in summary.values():
        row['io_delta']=row['common_net_io']-baseline['common_net_io']
        row['wirelength_delta']=row['common_net_wirelength']-baseline['common_net_wirelength']
    report=dict(analysis_source_sha256=digest(__file__),router='OpenROAD GRT',detailed_routing_verified=False,common_nets=len(common),in_die_common_nets=len(in_die_common),
        net_names_sha256=hashlib.sha256('\n'.join(sorted(common)).encode()).hexdigest(),
        scope='per-layer global-route segment union; lattice crossings clamp outside-core endpoints',
        checkpoints=summary,receipts=receipts)
    # Independent validation can veto evaluator-accepted checkpoints. Rejected
    # search proposals remain diagnostic and cannot become the selected result.
    selected=select_router_checkpoint(summary)
    report['router_selected_checkpoint']=selected
    report['router_confirmed_improvement']=selected!='baseline'
    selected_path=root/selected/'placement.npz'
    import shutil
    shutil.copyfile(selected_path,root/'router_selected.npz')
    (root/'router_selected.json').write_text(json.dumps(dict(checkpoint=selected,
        placement=str(root/'router_selected.npz'),source_placement=str(selected_path),sha256=digest(selected_path),
        improvement_confirmed=selected!='baseline',
        criteria='strict GRT IO improvement over baseline/control on full-common AND in-die-common nets; +5% GRT length'),indent=2))
    (root/'openroad_grt.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
