"""Construct pin-aware sparse boundary factors and decode their CE surrogate."""
from dataclasses import replace
import time
import numpy as np
from ioplace.region_grid import RegionGrid
from ioplace.ops.discrete_ce import conditional_expectation,net_metrics


def _active_cells(nl,x,y,rg,rectangles,limit,chunk=262144):
    selected=np.empty(0,dtype=np.int32);distance=np.empty(0)
    loads=np.zeros(rg.k,dtype=np.float64)
    for low in range(0,nl.num_movable,chunk):
        high=min(low+chunk,nl.num_movable)
        ids=np.arange(low,high,dtype=np.int32)
        home=rg.region_of_points(x[low:high],y[low:high])
        box=rectangles[home]
        d=np.minimum.reduce((np.where(box[:,0]>rg.die[0],np.abs(x[low:high]-box[:,0]),np.inf),
            np.where(box[:,2]<rg.die[2],np.abs(x[low:high]-box[:,2]),np.inf),
            np.where(box[:,1]>rg.die[1],np.abs(y[low:high]-box[:,1]),np.inf),
            np.where(box[:,3]<rg.die[3],np.abs(y[low:high]-box[:,3]),np.inf)))
        area=nl.node_size_x[low:high]*nl.node_size_y[low:high]
        loads+=np.bincount(home,weights=area,minlength=rg.k)
        finite=np.isfinite(d)
        ids=np.r_[selected,ids[finite]];d=np.r_[distance,d[finite]]
        keep=np.lexsort((ids,d))[:limit]
        selected,distance=ids[keep],d[keep]
    return selected,distance,loads


def prepare_discrete(nl,regions,node_x,node_y,*,max_active=65536,tau_fraction=.03):
    """O(N+P) scans plus bounded active sorting; no full-cell region matrix."""
    started=time.perf_counter()
    if max_active<0 or not np.isfinite(tau_fraction) or tau_fraction<=0:
        raise ValueError("positive temperature and nonnegative active limit required")
    if any(np.asarray(region.rects).shape!=(1,4) for region in regions.regions):
        raise ValueError("boundary projection currently requires one rectangle per region")
    rg=RegionGrid(regions)
    if rg.k>64:
        raise ValueError("region masks support at most64 regions")
    x=np.asarray(node_x,dtype=np.float64);y=np.asarray(node_y,dtype=np.float64)
    if x.shape!=(nl.num_physical,) or y.shape!=x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("finite physical coordinates required")
    rectangles=np.array([region.rects[0] for region in regions.regions])
    active,distance,target=_active_cells(nl,x,y,rg,rectangles,min(max_active,nl.num_movable))
    B=len(active);home=rg.region_of_points(x[active],y[active]).astype(np.int32)
    neighbors=[set() for _ in range(rg.k)]
    for a,b in ((rg.grid[:,:-1],rg.grid[:,1:]),(rg.grid[:-1,:],rg.grid[1:,:])):
        different=a!=b
        for left,right in np.unique(np.column_stack((a[different],b[different])),axis=0):
            neighbors[int(left)].add(int(right));neighbors[int(right)].add(int(left))
    C=1+max(map(len,neighbors),default=0)
    characteristic=((nl.xh-nl.xl)*(nl.yh-nl.yl)/rg.k)**.5
    tau=tau_fraction*characteristic
    eps=max(min(rg.cell_w,rg.cell_h)*1e-3,
            max(abs(v) for v in rg.die)*np.finfo(np.float32).eps*4)
    if np.any(rectangles[:,2]-rectangles[:,0]<=2*eps) or np.any(rectangles[:,3]-rectangles[:,1]<=2*eps):
        raise ValueError("region too narrow for robust interior anchor projection")
    all_x=np.clip(x[active,None],rectangles[None,:,0]+eps,rectangles[None,:,2]-eps)
    all_y=np.clip(y[active,None],rectangles[None,:,1]+eps,rectangles[None,:,3]-eps)
    all_x[np.arange(B),home]=x[active];all_y[np.arange(B),home]=y[active]
    all_distance=np.abs(all_x-x[active,None])+np.abs(all_y-y[active,None])
    all_probability=np.exp(-all_distance/tau)
    all_probability/=all_probability.sum(axis=1,keepdims=True)
    candidate_region=np.repeat(home[:,None],C,axis=1)
    valid=np.zeros((B,C),dtype=bool);valid[:,0]=True
    for region in range(rg.k):
        rows=np.flatnonzero(home==region)
        support=[region,*sorted(neighbors[region])]
        candidate_region[rows,:len(support)]=support
        valid[rows,:len(support)]=True
    row=np.arange(B)[:,None]
    candidate_x=all_x[row,candidate_region];candidate_y=all_y[row,candidate_region]
    probability=all_probability[row,candidate_region]*valid
    retained=probability.sum(axis=1)
    probability/=retained[:,None]
    area=(nl.node_size_x[active]*nl.node_size_y[active]).astype(np.float64)
    inactive=target-np.bincount(home,weights=area,minlength=rg.k)
    unary=.01*(np.abs(candidate_x-x[active,None])+np.abs(candidate_y-y[active,None]))/max(B,1)/characteristic
    mapping=np.full(nl.num_physical,-1,dtype=np.int32);mapping[active]=np.arange(B,dtype=np.int32)
    degrees=nl.net_degrees
    collected=[];unsupported=0
    for low in range(0,len(nl.pin2node),2000000):
        high=min(low+2000000,len(nl.pin2node))
        mask=mapping[nl.pin2node[low:high]]>=0
        supported=(degrees[nl.pin2net[low:high]]>=2)&(degrees[nl.pin2net[low:high]]<=256)
        unsupported+=int((mask&~supported).sum())
        collected.append((np.flatnonzero(mask&supported)+low).astype(np.int32))
    active_pins=np.concatenate(collected) if collected else np.empty(0,dtype=np.int32)
    key=mapping[nl.pin2node[active_pins]].astype(np.int64)*nl.num_nets+nl.pin2net[active_pins]
    order=np.argsort(key,kind="stable");key=key[order];active_pins=active_pins[order]
    inc_start=np.r_[np.flatnonzero(np.r_[True,np.diff(key)!=0]),len(key)] if len(key) else np.array([0])
    unique=key[inc_start[:-1]]
    cells=(unique//max(nl.num_nets,1)).astype(np.int32)
    global_nets=(unique%max(nl.num_nets,1)).astype(np.int32)
    factor_nets=np.unique(global_nets)
    factor_id=np.searchsorted(factor_nets,global_nets).astype(np.int32)
    cell_start=np.r_[0,np.cumsum(np.bincount(cells,minlength=B))].astype(np.int32)
    masks=np.empty((len(unique),C),dtype=np.uint64)
    pin_cell=mapping[nl.pin2node[active_pins]]
    for c in range(C):
        rid=rg.region_of_points(candidate_x[pin_cell,c]+nl.pin_offset_x[active_pins],
                                candidate_y[pin_cell,c]+nl.pin_offset_y[active_pins]).astype(np.uint64)
        bits=np.left_shift(np.uint64(1),rid)
        if len(bits):masks[:,c]=np.bitwise_or.reduceat(bits,inc_start[:-1])
    factor_degrees=degrees[factor_nets]
    starts=np.r_[0,np.cumsum(factor_degrees)].astype(np.int32)
    offsets=nl.flat_net2pin_start[factor_nets]-starts[:-1]
    indices=np.arange(starts[-1],dtype=np.int64)+np.repeat(offsets,factor_degrees)
    pins=nl.flat_net2pin[indices]
    compact=replace(nl,node_x=x,node_y=y,pin2node=nl.pin2node[pins],
        pin2net=np.repeat(np.arange(len(factor_nets),dtype=np.int32),factor_degrees),
        pin_offset_x=nl.pin_offset_x[pins],pin_offset_y=nl.pin_offset_y[pins],
        flat_net2pin=np.arange(len(pins),dtype=np.int32),flat_net2pin_start=starts)
    fixed=np.zeros(len(factor_nets),dtype=np.uint64)
    deterministic=mapping[compact.pin2node]<0
    rid=rg.region_of_points(x[compact.pin2node[deterministic]]+compact.pin_offset_x[deterministic],
                            y[compact.pin2node[deterministic]]+compact.pin_offset_y[deterministic]).astype(np.uint64)
    np.bitwise_or.at(fixed,compact.pin2net[deterministic],np.left_shift(np.uint64(1),rid))
    metric=net_metrics(compact,rg)
    return dict(active=active,home=home,candidate_x=candidate_x,candidate_y=candidate_y,
        probabilities=probability,candidate_region=candidate_region,area=area,
        region_area=(rectangles[:,2]-rectangles[:,0])*(rectangles[:,3]-rectangles[:,1]),
        target=target,inactive_load=inactive,unary=unary,cell_start=cell_start,factor_id=factor_id,
        candidate_mask=masks,fixed_mask=fixed,passed_mask=metric["per_net_passed_mask"],
        factor_net_ids=factor_nets,compact_netlist=compact,region_grid=rg,
        metrics_before=metric,diagnostics=dict(active_cells=B,candidate_slots=C,
            supported_incident_nets=len(factor_nets),supported_incident_pins=len(pins),
            excluded_active_pins=unsupported,temperature=tau,characteristic_region_length=characteristic,
            max_active_distance=float(distance.max()) if len(distance) else 0.,
            discarded_probability_mass_mean=float((1-retained).mean()) if B else 0.,
            discarded_probability_mass_max=float((1-retained).max()) if B else 0.,
            preparation_s=time.perf_counter()-started,region_projection_epsilon=eps,
            balance_scope="anchor-load deviation from incumbent; physical feasibility deferred to legalizer"))


def decode_placement(prepared,node_x,node_y,baseline_io,baseline_ft):
    started=time.perf_counter()
    fields=("probabilities","candidate_region","area","region_area","target","inactive_load",
            "unary","cell_start","factor_id","candidate_mask","fixed_mask","passed_mask")
    result=conditional_expectation(**{key:prepared[key] for key in fields},
        io_weight=1/max(baseline_io,1),ft_weight=1/max(baseline_ft,1),balance_weight=.1)
    x=np.array(node_x,dtype=np.float64,copy=True);y=np.array(node_y,dtype=np.float64,copy=True)
    rows=np.arange(len(prepared["active"]));choices=result["choices"]
    x[prepared["active"]]=prepared["candidate_x"][rows,choices]
    y[prepared["active"]]=prepared["candidate_y"][rows,choices]
    result["decode_s"]=time.perf_counter()-started
    return x,y,result


def adjacent_swap_pairs(prepared,x,y,limit=None):
    from scipy.spatial import cKDTree
    active=prepared["active"];rg=prepared["region_grid"]
    home=rg.region_of_points(x[active],y[active])
    points=np.column_stack((x[active],y[active]))
    pairs={}
    adjacency=set()
    for a,b in ((rg.grid[:,:-1],rg.grid[:,1:]),(rg.grid[:-1,:],rg.grid[1:,:])):
        different=a!=b
        adjacency.update(tuple(sorted(map(int,pair))) for pair in np.unique(np.column_stack((a[different],b[different])),axis=0))
    for a,b in sorted(adjacency):
        left=np.flatnonzero(home==a);right=np.flatnonzero(home==b)
        if not len(left) or not len(right):continue
        distances,indices=cKDTree(points[right]).query(points[left],k=1)
        for i,j,distance in zip(left,right[indices],distances):
            pair=tuple(sorted((int(i),int(j))))
            pairs[pair]=float(distance)
    limit=len(active) if limit is None else limit
    ordered=sorted(pairs,key=lambda pair:(pairs[pair],pair))[:limit]
    return np.asarray(ordered,dtype=np.int32).reshape(-1,2)


def refine_placement(prepared,node_x,node_y,baseline_io,baseline_ft,*,swap_pairs=None):
    """Compiled exact supported-net moves/swaps; legal/full-net gate is external."""
    import ctypes
    from ioplace.ops.discrete_ce import _native
    started=time.perf_counter()
    nl=prepared["compact_netlist"];rg=prepared["region_grid"]
    x=np.array(node_x,dtype=np.float64,copy=True);y=np.array(node_y,dtype=np.float64,copy=True)
    active=np.ascontiguousarray(prepared["active"],dtype=np.int32)
    B,C=prepared["candidate_x"].shape;F=nl.num_nets;K=rg.k
    if x.shape!=(nl.num_physical,) or y.shape!=x.shape or np.any(active<0) or np.any(active>=len(x)) or len(set(active))!=B:
        raise ValueError("invalid active cell coordinates")
    if swap_pairs is None:swap_pairs=adjacent_swap_pairs(prepared,x,y)
    swap_pairs=np.asarray(swap_pairs,dtype=np.int32).reshape(-1,2)
    if np.any((swap_pairs<0)|(swap_pairs>=B)) or np.any(swap_pairs[:,0]==swap_pairs[:,1]):
        raise ValueError("invalid swap pair")
    before_home=prepared["home"];now_home=rg.region_of_points(x[active],y[active])
    area=prepared["area"]
    load=prepared["target"]+np.bincount(now_home,weights=area,minlength=K)-np.bincount(before_home,weights=area,minlength=K)
    # compact_netlist positions are the immutable incumbent used to prepare factors.
    original_x=nl.node_x[active];original_y=nl.node_y[active]
    order=nl.flat_net2pin
    values=[np.ascontiguousarray(value,dtype=dtype) for value,dtype in (
        (nl.flat_net2pin_start,np.int32),(nl.pin2node[order],np.int32),
        (nl.pin_offset_x[order],np.float64),(nl.pin_offset_y[order],np.float64),
        (x,np.float64),(y,np.float64),(rg.grid,np.int16),(active,np.int32),
        (prepared["candidate_x"],np.float64),(prepared["candidate_y"],np.float64),
        (prepared["cell_start"],np.int32),(prepared["factor_id"],np.int32),
        (swap_pairs[:,0],np.int32),(swap_pairs[:,1],np.int32),(area,np.float64),
        (prepared["region_area"],np.float64),(prepared["target"],np.float64),(load,np.float64),
        (original_x,np.float64),(original_y,np.float64))]
    P=len(nl.pin2node);S=len(swap_pairs);I=len(prepared["factor_id"]);N=len(x)
    shapes=((F+1,),(P,),(P,),(P,),(N,),(N,),(rg.nx,rg.nx),(B,),
            (B,C),(B,C),(B+1,),(I,),(S,),(S,),(B,),(K,),(K,),(K,),(B,),(B,))
    if (any(v.shape!=shape for v,shape in zip(values,shapes)) or np.any(values[14]<0)
            or np.any(values[15]<=0)):
        raise ValueError("invalid refinement buffer shapes or areas")
    if any(not np.all(np.isfinite(v)) for v in values):raise ValueError("nonfinite refinement input")
    starts,nodes=values[:2]
    if (starts.shape!=(F+1,) or starts[0]!=0 or starts[-1]!=len(nodes)
            or np.any(np.diff(starts)<0) or np.any(np.diff(starts)>256)
            or np.any((nodes<0)|(nodes>=len(x)))
            or values[10].shape!=(B+1,) or values[10][0]!=0 or values[10][-1]!=len(values[11])
            or np.any(np.diff(values[10])<0) or np.any((values[11]<0)|(values[11]>=F))):
        raise ValueError("invalid refinement topology")
    moves=np.empty(B,dtype=np.int32);swaps=np.empty(len(swap_pairs),dtype=np.int32)
    trace=np.empty(B+len(swap_pairs)+1,dtype=np.float64);final_load=np.empty(K,dtype=np.float64)
    library,provenance=_native();fn=library.ioplace_refine
    fn.restype=ctypes.c_int
    fn.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*20+[ctypes.c_double]*8+[ctypes.c_void_p]*4
    ptr=lambda value:ctypes.c_void_p(value.ctypes.data)
    alpha,beta=1/max(baseline_io,1),1/max(baseline_ft,1)
    displacement_weight=.01/max(B,1)/prepared["diagnostics"]["characteristic_region_length"]
    code=fn(B,C,F,K,rg.nx,len(swap_pairs),*map(ptr,values),alpha,beta,.1,displacement_weight,
            *map(float,rg.die),*map(ptr,[moves,swaps,trace,final_load]))
    if code:raise RuntimeError(f"native refinement failed: {code}")
    x,y=values[4:6]
    metric=net_metrics(nl,rg,x,y)
    expected=alpha*metric["io_count"]+beta*metric["ft_count"]
    expected+=.1*np.sum(((final_load-prepared["target"])/prepared["region_area"])**2)
    expected+=displacement_weight*np.sum(np.abs(x[active]-original_x)+np.abs(y[active]-original_y))
    tolerance=1e-9*max(1.,abs(float(trace[0])))
    actual_load=prepared["target"]+np.bincount(rg.region_of_points(x[active],y[active]),weights=area,minlength=K)-np.bincount(before_home,weights=area,minlength=K)
    if (not np.all(np.isfinite(trace)) or np.any(np.diff(trace)>tolerance)
            or abs(expected-trace[-1])>tolerance or not np.allclose(final_load,actual_load,atol=1e-7,rtol=1e-12)):
        raise ArithmeticError("incremental refinement disagrees with full supported-net metrics/load")
    return x,y,dict(moves_accepted=int((moves>=0).sum()),swaps_attempted=len(swaps),
        swaps_accepted=int(swaps.sum()),move_choices=moves,swap_decisions=swaps,swap_pairs=swap_pairs,
        objective_trace=trace,metrics_after=metric,refine_s=time.perf_counter()-started,
        full_recompute_objective=expected,provenance=provenance,
        acceptance_scope="intermediate supported-net objective; final full HPWL and legality checks required")
