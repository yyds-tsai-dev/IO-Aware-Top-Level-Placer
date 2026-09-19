"""One-shot OpenROAD process: route clean signal nets, dump GRT resources."""
import ctypes
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time

try:
    from .checked_tcl import checked_eval
except ImportError:  # Standalone ``openroad -python path/to/script.py``.
    from checked_tcl import checked_eval


def u8(value):
    # Some installed SWIG builds lack a uint8_t scalar typemap and return an
    # owning pointer to a copied byte. Keep the byte alive while reading it.
    # The missing SWIG destructor cannot free it; this bounded one-shot process
    # releases these small allocations at exit, without per-byte warning spam.
    if isinstance(value,int):return value
    if 'uint8_t *' not in str(value):raise TypeError('unexpected OpenDB byte representation')
    result=int(ctypes.c_uint8.from_address(int(value)).value)
    value.own(False)
    return result


def _run_commands(design, settings):
    route = 'global_route -congestion_iterations '+str(settings.get('congestion_iterations',50))
    if settings.get('allow_congestion',False):
        route += ' -allow_congestion'
    commands = [
        'set_thread_count '+str(settings.get('threads',4)),
        'source {'+settings['clear_script']+'}',
        'ioplace_clear_signal_routing [ord::get_db_block] {'+settings['cleared']+'}',
        'check_placement -verbose',
        route+' -verbose',
        'grt::write_segments {'+settings['segments']+'}',
    ]
    if settings.get('signal_layers'):
        commands.insert(4,'set_routing_layers -signal '+settings['signal_layers'])
    timings=[]
    for index, command in enumerate(commands):
        started=time.perf_counter()
        if settings.get('timings'):
            Path(settings['timings']).write_text(json.dumps(dict(completed=timings,running=command)))
        checked_eval(design, command, f'online-route-{index}')
        timings.append(dict(command=command,elapsed_s=time.perf_counter()-started))
    return timings


def audit_netlist(block, path):
    """Stream exact connectivity and inspect high-degree nets without exclusion."""
    hashes=[];largest=[];types={};degrees={str(n):0 for n in (100,1000,10000)}
    count=0;special=0;nontrivial=0
    with Path(path).open('w') as stream:
        for net in block.getNets():
            endpoints=sorted([['I',pin.getInst().getName(),pin.getMTerm().getName()] for pin in net.getITerms()]
                +[['B',pin.getName()] for pin in net.getBTerms()])
            degree=len(endpoints);kind=str(net.getSigType());is_special=bool(net.isSpecial())
            row=dict(name=net.getName(),signal_type=kind,special=is_special,degree=degree,endpoints=endpoints)
            encoded=json.dumps(row,sort_keys=True,separators=(',',':'))
            stream.write(encoded+'\n');hashes.append(hashlib.sha256(encoded.encode()).digest())
            count+=1;special+=is_special;nontrivial+=not is_special and degree>=2
            types[kind]=types.get(kind,0)+1
            for threshold in degrees:
                degrees[threshold]+=degree>int(threshold)
            heapq.heappush(largest,(degree,net.getName(),kind,is_special))
            if len(largest)>20:heapq.heappop(largest)
    cohort=hashlib.sha256()
    for sha in sorted(hashes):cohort.update(sha)
    return dict(net_count=count,special_net_count=special,nontrivial_nonspecial_net_count=nontrivial,
        signal_type_counts=types,degree_above_counts=degrees,connectivity_sha256=cohort.hexdigest(),
        largest_nets=[dict(degree=d,name=n,signal_type=k,special=s) for d,n,k,s in sorted(largest,reverse=True)],
        requested_exclusions=[],sampling=False)


def main():
    from openroad import Tech, Design
    settings=json.loads(Path(sys.argv[-1]).read_text())
    started=time.perf_counter();tech=Tech()
    for lef in settings['lefs']:tech.readLef(lef)
    design=Design(tech);design.readDef(settings['def'])
    loaded=time.perf_counter()
    audit=audit_netlist(design.getBlock(),settings['net_manifest'])
    Path(settings['net_audit']).write_text(json.dumps(audit,indent=2))
    audited=time.perf_counter()
    timings=_run_commands(design, settings)
    resources_started=time.perf_counter()
    block=design.getBlock();grid=block.getGCellGrid()
    xs,ys=list(grid.getGridX()),list(grid.getGridY())
    die=block.getDieArea()
    x_edges=xs+[die.xMax()];y_edges=ys+[die.yMax()]
    if x_edges[-1]<=x_edges[-2] or y_edges[-1]<=y_edges[-2]:raise ValueError('invalid final GCell boundary')
    nx,ny=len(xs),len(ys)
    hcap=[[0]*(nx-1) for _ in range(ny)];huse=[[0]*(nx-1) for _ in range(ny)]
    vcap=[[0]*nx for _ in range(ny-1)];vuse=[[0]*nx for _ in range(ny-1)]
    layers=[];preferred_layer_overflow=0
    for layer in tech.getDB().getTech().getLayers():
        if layer.getRoutingLevel()<=0:continue
        horizontal=str(layer.getDirection())=='HORIZONTAL'
        vertical=str(layer.getDirection())=='VERTICAL'
        if not horizontal and not vertical:continue
        layers.append(dict(name=layer.getName(),direction=str(layer.getDirection())))
        for y in range(ny if horizontal else ny-1):
            for x in range(nx-1 if horizontal else nx):
                capacity=u8(grid.getCapacity(layer,x,y));usage=u8(grid.getUsage(layer,x,y))
                preferred_layer_overflow+=max(usage-capacity,0)
                if horizontal:hcap[y][x]+=capacity;huse[y][x]+=usage
                else:vcap[y][x]+=capacity;vuse[y][x]+=usage
    result=dict(x_edges_dbu=x_edges,y_edges_dbu=y_edges,horizontal_capacity=hcap,
        vertical_capacity=vcap,horizontal_usage=huse,vertical_usage=vuse,layers=layers,
        capacity_semantics='OpenDB total track capacity; usage includes blockages and wires; preferred-direction layers summed',
        uint8_backend=True,preferred_layer_overflow=preferred_layer_overflow)
    Path(settings['resources']).write_text(json.dumps(result))
    Path(settings['timings']).write_text(json.dumps(dict(completed=timings,running=None,
        read_lef_def_s=loaded-started,net_audit_s=audited-loaded,
        resource_dump_s=time.perf_counter()-resources_started,total_s=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
