"""Sparse cell-categorical conditional expectation, with a compiled hot loop.

One incidence represents a cell/net pair. Its candidate mask is the union of
all pins of that cell on that net. Pins on one cell are not independent trials.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def _native():
    source=Path(__file__).with_name("discrete_core.cpp")
    compiler=os.environ.get("CXX","g++")
    version=subprocess.check_output([compiler,"--version"])
    digest=hashlib.sha256(source.read_bytes()+version).hexdigest()
    directory=Path(tempfile.gettempdir())/f"ioplace-discrete-{os.getuid()}"
    directory.mkdir(mode=0o700,exist_ok=True)
    library=directory/(digest+".so")
    if not library.exists():
        temporary=directory/(digest+f".{os.getpid()}.so")
        subprocess.run([compiler,"-std=c++17","-O3","-fPIC","-shared",str(source),"-o",str(temporary)],check=True)
        temporary.replace(library)
    loaded=ctypes.CDLL(str(library))
    fn=loaded.ioplace_ce
    fn.restype=ctypes.c_int
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*12+[ctypes.c_double]*3+[ctypes.c_void_p]*3
    return loaded,dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),compiler=version.decode().splitlines()[0])


def conditional_expectation(probabilities,candidate_region,area,region_area,target,inactive_load,
                            unary,cell_start,factor_id,candidate_mask,fixed_mask,passed_mask,
                            *,io_weight=1.,ft_weight=1.,balance_weight=.1):
    """Return deterministic choices and the exact frozen-surrogate objective trace."""
    probabilities=np.ascontiguousarray(probabilities,dtype=np.float64)
    if probabilities.ndim!=2 or not probabilities.shape[1]:
        raise ValueError("probabilities must be a cell-by-candidate matrix")
    B,C=probabilities.shape
    arrays=[np.ascontiguousarray(value,dtype=dtype) for value,dtype in (
        (candidate_region,np.int32),(area,np.float64),(region_area,np.float64),
        (target,np.float64),(inactive_load,np.float64),(unary,np.float64),
        (cell_start,np.int32),(factor_id,np.int32),(candidate_mask,np.uint64),
        (fixed_mask,np.uint64),(passed_mask,np.uint64))]
    region,area,A,target,inactive,unary,starts,factors,masks,fixed,passed=arrays
    K,F,I=len(A),len(fixed),len(factors)
    expected=((B,C),(B,),(K,),(K,),(K,),(B,C),(B+1,),(I,),(I,C),(F,),(F,))
    if any(value.shape!=shape for value,shape in zip(arrays,expected)) or not 1<=K<=64:
        raise ValueError("categorical factor array shape mismatch")
    if any(not np.all(np.isfinite(value)) for value in (probabilities,area,A,target,inactive,unary)):
        raise ValueError("nonfinite categorical data")
    if (np.any(probabilities<0) or not np.allclose(probabilities.sum(axis=1),1,atol=1e-12,rtol=0)
            or np.any(area<0) or np.any(A<=0) or np.any((region<0)|(region>=K))):
        raise ValueError("invalid probabilities, areas, or regions")
    if starts[0]!=0 or starts[-1]!=I or np.any(np.diff(starts)<0) or np.any((factors<0)|(factors>=F)):
        raise ValueError("invalid incidence CSR")
    if np.any(masks==0) or (K<64 and any(np.any(value>>np.uint64(K)) for value in (masks,fixed,passed))):
        raise ValueError("candidate masks must represent nonempty pins in valid regions")
    for i in range(B):
        row=factors[starts[i]:starts[i+1]]
        if len(np.unique(row))!=len(row):
            raise ValueError("combine all same-cell pins into one cell/net incidence")
    if np.any((np.bincount(factors,minlength=F)==0)&(fixed==0)):
        raise ValueError("empty net factor")
    weights=np.asarray([io_weight,ft_weight,balance_weight],dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError("weights must be finite and nonnegative")
    choices=np.empty(B,dtype=np.int32);trace=np.empty(B+1,dtype=np.float64)
    identity=np.zeros(1,dtype=np.float64)
    library,provenance=_native()
    fn=library.ioplace_ce
    pointer=lambda value:ctypes.c_void_p(value.ctypes.data)
    code=fn(B,C,K,F,I,*map(pointer,[probabilities,*arrays]),*map(float,weights),
            pointer(choices),pointer(trace),pointer(identity))
    if code or not np.all(np.isfinite(trace)):
        raise RuntimeError("native categorical expectation failed")
    tolerance=1e-9*max(1.,abs(float(trace[0])))
    if np.any(np.diff(trace)>tolerance) or identity[0]>tolerance:
        raise ArithmeticError("conditional expectation identity or monotonicity failed")
    return dict(choices=choices,objective_trace=trace,weighted_identity_max_error=float(identity[0]),
                numerical_tolerance=tolerance,provenance=provenance,
                objective_scope="expected touched-region IO plus frozen-tree FT absence, displacement and anchor-load balance")


def net_metrics(nl,rg,node_x=None,node_y=None):
    """Compiled reference-compatible Prim/lattice metrics for degree<=256 nets."""
    x=np.ascontiguousarray(nl.node_x if node_x is None else node_x,dtype=np.float64)
    y=np.ascontiguousarray(nl.node_y if node_y is None else node_y,dtype=np.float64)
    if x.shape!=(nl.num_physical,) or y.shape!=x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("finite physical node positions required")
    order=nl.flat_net2pin
    nodes=np.ascontiguousarray(nl.pin2node[order],dtype=np.int32)
    ox=np.ascontiguousarray(nl.pin_offset_x[order],dtype=np.float64)
    oy=np.ascontiguousarray(nl.pin_offset_y[order],dtype=np.float64)
    starts=np.ascontiguousarray(nl.flat_net2pin_start,dtype=np.int32)
    grid=np.ascontiguousarray(rg.grid,dtype=np.int16)
    if (len(starts)!=nl.num_nets+1 or starts[0]!=0 or starts[-1]!=len(nodes)
            or np.any(np.diff(starts)<0) or np.any(np.diff(starts)>256)
            or np.any((nodes<0)|(nodes>=len(x))) or np.any((grid<0)|(grid>=64))
            or not np.all(np.isfinite(ox)) or not np.all(np.isfinite(oy))):
        raise ValueError("invalid or unsupported metric topology")
    F=nl.num_nets
    io=np.empty(F,dtype=np.int32);ft=np.empty(F,dtype=np.int32)
    hpwl=np.empty(F,dtype=np.float64);tree=np.empty(F,dtype=np.float64);passed=np.empty(F,dtype=np.uint64)
    library,provenance=_native();fn=library.ioplace_net_metrics
    fn.restype=ctypes.c_int
    fn.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*7+[ctypes.c_double]*4+[ctypes.c_void_p]*5
    ptr=lambda value:ctypes.c_void_p(value.ctypes.data)
    code=fn(F,rg.nx,*map(ptr,[starts,nodes,ox,oy,x,y,grid]),*map(float,rg.die),
            *map(ptr,[io,ft,hpwl,tree,passed]))
    if code:
        raise RuntimeError(f"native net metrics failed: {code}")
    return dict(per_net_crossings=io,per_net_ft=ft,per_net_hpwl=hpwl,per_net_tree_wl=tree,
                per_net_passed_mask=passed,io_count=int(io.sum(dtype=np.int64)),
                ft_count=int(ft.sum(dtype=np.int64)),hpwl=float(hpwl.sum()),tree_wl=float(tree.sum()),
                provenance=provenance)
