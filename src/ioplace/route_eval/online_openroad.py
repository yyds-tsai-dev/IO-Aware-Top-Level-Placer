"""Measured GRT routes/resources for online placement feedback."""
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from ioplace.region_grid import RegionGrid
from ioplace.regions import RegionSet
from ioplace.route_eval.joint import ResourceGrid
from ioplace.route_eval.topology import union_metrics


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


def layer_resource_usage(grid,layers):
    """Unique occupancy within a net/layer; sum multiplicity across layers."""
    keys=[grid.edge_keys(np.asarray(v,dtype=float).reshape(-1,2,2)) for v in layers.values()]
    unique,counts=np.unique(np.concatenate(keys) if keys else np.array([],dtype=int),return_counts=True)
    return dict(keys=unique.tolist(),counts=counts.tolist())


def parse_native_congestion(text):
    """Read the final native report; layer-summed usage can hide overflow."""
    marker = 'Final congestion report:'
    if marker not in text:
        raise ValueError('missing native final congestion report')
    final = text.rsplit(marker, 1)[1]
    row = re.search(r'^Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$', final, re.M)
    if row is None:
        raise ValueError('invalid native final congestion total')
    resource, demand, percent, horizontal, vertical, overflow = row.groups()
    return dict(resource=int(resource), demand=int(demand), usage_percent=float(percent),
        max_horizontal_overflow=int(horizontal), max_vertical_overflow=int(vertical), total_overflow=int(overflow))


def run_openroad(def_path,lefs,out,binary,*,congestion_iterations=50,allow_congestion=False,threads=4,signal_layers=None):
    for name, value in (('congestion_iterations', congestion_iterations), ('threads', threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f'positive integer {name} required')
    if type(allow_congestion) is not bool:
        raise ValueError('boolean allow_congestion required')
    if signal_layers is not None and (not isinstance(signal_layers,str)
            or re.fullmatch(r'[A-Za-z0-9_]+-[A-Za-z0-9_]+',signal_layers) is None):
        raise ValueError('signal_layers must be a simple first-last layer range')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).parent/'or_scripts/dump_online_route.py'
    clear=source.with_name('clear_signal_routing.tcl')
    checked=source.with_name('checked_tcl.py')
    paths=[Path(def_path).resolve(),*[Path(p).resolve() for p in lefs],source,clear,checked,Path(binary).resolve()]
    if any(any(c in str(p) for c in '{}\n\r') for p in [out,*paths]):raise ValueError('unsupported Tcl path')
    settings=dict(def_=str(paths[0]),lefs=list(map(str,paths[1:1+len(lefs)])),clear_script=str(clear),
        cleared=str(out/'cleared.txt'),segments=str(out/'segments.txt'),resources=str(out/'resources.json'),
        congestion_iterations=congestion_iterations,allow_congestion=allow_congestion,threads=threads,
        timings=str(out/'timings.json'),net_audit=str(out/'net_audit.json'),net_manifest=str(out/'net_manifest.jsonl'))
    if signal_layers is not None:settings['signal_layers']=signal_layers
    settings['def']=settings.pop('def_')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    command=[str(binary),'-exit','-python',str(source),str(out/'settings.json')]
    input_hashes={str(p):digest(p) for p in [*paths,out/'settings.json']}
    started=time.perf_counter()
    with (out/'run.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    with (out/'run.log').open(errors='replace') as log:
        errors=[line.strip() for line in log if '[ERROR ' in line]
    returncode=result.returncode or (1 if errors else 0)
    receipt=dict(command=command,tool_returncode=result.returncode,returncode=returncode,
        logged_errors=errors,inputs=input_hashes,elapsed_s=time.perf_counter()-started,
        routing_policy=dict(congestion_iterations=congestion_iterations,allow_congestion=allow_congestion,threads=threads))
    if signal_layers is not None:receipt['routing_policy']['signal_layers']=signal_layers
    receipt['outputs']={'run.log':digest(out/'run.log')}
    try:
        receipt['native_congestion']=parse_native_congestion((out/'run.log').read_text(errors='replace'))
    except ValueError as error:
        receipt['congestion_parse_error']=str(error)
        if returncode==0:
            receipt.update(returncode=1,validation_error=str(error));returncode=1
    if returncode==0:
        try:
            receipt['outputs'].update({name:digest(out/name) for name in
                ('segments.txt','resources.json','timings.json','net_audit.json','net_manifest.jsonl')})
        except (ValueError, OSError) as error:
            receipt.update(returncode=1,validation_error=str(error))
            returncode=1
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    if returncode:raise RuntimeError(f'OpenROAD online route failed; see {out}/run.log')
    return receipt


def load_observation(out,coord_path,regions_path,netmap_path):
    out=Path(out)
    receipt=json.loads((out/'receipt.json').read_text())
    if receipt['returncode'] or any(digest(out/name)!=sha for name,sha in receipt['outputs'].items()):
        raise ValueError('invalid online routing receipt')
    coord=json.loads(Path(coord_path).read_text());raw=json.loads((out/'resources.json').read_text())
    shift=np.asarray(coord['shift_factor']);scale=float(coord['scale_factor'])
    rg=RegionGrid(RegionSet.from_json(regions_path))
    grid=ResourceGrid((np.asarray(raw['x_edges_dbu'])-shift[0])*scale,
                      (np.asarray(raw['y_edges_dbu'])-shift[1])*scale)
    capacity=np.r_[np.asarray(raw['horizontal_capacity']).ravel(),np.asarray(raw['vertical_capacity']).ravel()].astype(float)
    usage=np.r_[np.asarray(raw['horizontal_usage']).ravel(),np.asarray(raw['vertical_usage']).ravel()].astype(float)
    if capacity.shape!=(grid.edge_count,) or usage.shape!=capacity.shape:raise ValueError('invalid OpenDB resource dimensions')
    names={name:int(net) for net,name in json.loads(Path(netmap_path).read_text()).items()}
    nets={};current=None;opened=False
    for raw_line in (out/'segments.txt').read_text().splitlines():
        line=raw_line.strip()
        if not line:continue
        if line=='(':
            if current is None or opened:raise ValueError('invalid route start')
            opened=True;continue
        if line==')':
            if not opened:raise ValueError('invalid route end')
            opened=False;current=None;continue
        if not opened:
            if line not in names:raise ValueError('router changed net identity')
            current=names[line]
            if current in nets:raise ValueError('duplicate routed net')
            nets[current]={}
            continue
        fields=line.split()
        if len(fields)!=6:raise ValueError('invalid GRT segment')
        x0,y0,l0,x1,y1,l1=fields
        a,b=np.asarray([[int(x0),int(y0)],[int(x1),int(y1)]],dtype=float)
        if l0!=l1:
            if not np.array_equal(a,b):raise ValueError('nonzero lateral via')
            continue
        segment=(np.array([a,b])-shift)*scale
        nets[current].setdefault(l0,[]).append(segment)
    if opened or current is not None:raise ValueError('unterminated GRT net')
    net_io={};net_keys={};net_usage={};net_wirelength={};outside={}
    for net,layers in nets.items():
        io=0;length=0.;segments=[];out_count=0
        for values in layers.values():
            geometry=np.asarray(values,dtype=float).reshape(-1,2,2)
            metric=union_metrics(rg,geometry)
            io+=metric['crossings'];length+=metric['wirelength'];segments.extend(metric['segments'])
            out_count+=int(np.any((geometry<rg.die[:2])|(geometry>rg.die[2:]),axis=2).sum())
        net_io[net]=io;net_wirelength[net]=length;outside[net]=out_count
        net_keys[net]=grid.edge_keys(np.asarray(segments).reshape(-1,2,2)).tolist()
        net_usage[net]=layer_resource_usage(grid,layers)
    return dict(grid=grid,capacity=capacity,usage=usage,net_io=net_io,net_keys=net_keys,net_usage=net_usage,
        net_wirelength=net_wirelength,outside_endpoints=outside,
        actual_io=sum(net_io.values()),actual_wirelength=sum(net_wirelength.values()),
        source_sha256=digest(out/'receipt.json'),placement_sha256=receipt['inputs'][json.loads((out/'settings.json').read_text())['def']],
        provenance=str(out/'receipt.json'),resource_semantics=raw['capacity_semantics'],
        native_congestion=receipt.get('native_congestion'),routing_policy=receipt.get('routing_policy'),
        router_elapsed_s=receipt.get('elapsed_s'),
        aggregated_resource_overflow=float(np.maximum(usage-capacity,0).sum()),
        preferred_layer_overflow=raw.get('preferred_layer_overflow'),
        net_audit=json.loads((out/'net_audit.json').read_text()) if (out/'net_audit.json').exists() else None,
        routed_net_count=len(nets),routed_net_names_sha256=hashlib.sha256(json.dumps(
            sorted(name for name,net in names.items() if net in nets),separators=(',',':')).encode()).hexdigest())
