"""Transactional FLUTE trees and joint, per-net 2-D routing-resource demand.

Physical IO/length use a collinear geometric union. Resource demand uses one
unit per net per crossed resource-grid edge, independent of branch multiplicity.
Different nets accumulate. Layer assignment and detailed routing remain external.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from ioplace.route_eval.topology import flute_edges, union_metrics


class ResourceGrid:
    """Possibly nonuniform resource cells, independent of the IO region lattice."""
    def __init__(self, x_edges, y_edges):
        self.x_edges, self.y_edges = (np.asarray(v,dtype=float).copy() for v in (x_edges,y_edges))
        for v in (self.x_edges,self.y_edges):
            if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or np.any(np.diff(v)<=0):
                raise ValueError('finite strictly increasing resource-grid boundaries required')
        self.nx,self.ny=len(self.x_edges)-1,len(self.y_edges)-1
        self.horizontal_count=self.ny*(self.nx-1)
        self.edge_count=self.horizontal_count+(self.ny-1)*self.nx
        self.x_centers=(self.x_edges[:-1]+self.x_edges[1:])/2
        self.y_centers=(self.y_edges[:-1]+self.y_edges[1:])/2

    def edge_keys(self, segments):
        segments=np.asarray(segments,dtype=float).reshape(-1,2,2)
        if not np.isfinite(segments).all():raise ValueError('nonfinite resource segment')
        keys=[]
        for a,b in segments:
            if a[0]!=b[0] and a[1]!=b[1]:raise ValueError('resource segments must be rectilinear')
            ax,bx=np.clip(np.searchsorted(self.x_edges,[a[0],b[0]],side='right')-1,0,self.nx-1)
            ay,by=np.clip(np.searchsorted(self.y_edges,[a[1],b[1]],side='right')-1,0,self.ny-1)
            if ax!=bx:keys.extend(ay*(self.nx-1)+np.arange(min(ax,bx),max(ax,bx)))
            if ay!=by:keys.extend(self.horizontal_count+np.arange(min(ay,by),max(ay,by))*self.nx+bx)
        return np.unique(np.asarray(keys,dtype=np.int64))


@dataclass(frozen=True)
class SharedRoute:
    pins: np.ndarray
    segments: np.ndarray
    keys: np.ndarray
    crossings: int
    wirelength: float
    topology: str = 'flute'


def _readonly(array):
    result=np.array(array,copy=True)
    result.flags.writeable=False
    return result


class JointRoutingState:
    def __init__(self, region_grid, resource_grid, capacity, *, background=None,
                 congestion_weight=.1, wirelength_weight=.001, wirelength_budget=.05,
                 coordinate_scale=1000., candidate_tracks=6):
        self.rg,self.grid=region_grid,resource_grid
        self.capacity=np.asarray(capacity,dtype=float).copy()
        self.background=(np.zeros(resource_grid.edge_count) if background is None
                         else np.asarray(background,dtype=float).copy())
        for value in (self.capacity,self.background):
            if value.shape!=(resource_grid.edge_count,) or not np.isfinite(value).all() or (value<0).any():
                raise ValueError('nonnegative finite capacity/background vector matching resource grid required')
        if (not np.isfinite([congestion_weight,wirelength_weight,wirelength_budget,coordinate_scale]).all()
                or min(congestion_weight,wirelength_weight,wirelength_budget)<0
                or coordinate_scale<=0 or candidate_tracks<1):
            raise ValueError('invalid joint routing settings')
        self.congestion_weight=congestion_weight
        self.wirelength_weight=wirelength_weight
        self.wirelength_budget=wirelength_budget
        self.coordinate_scale=coordinate_scale
        self.candidate_tracks=candidate_tracks
        self.demand=self.background.copy()
        self.routes={}
        self.net_weights={}
        self.edge_prices=np.zeros(resource_grid.edge_count)
        self.generation=0

    def fork(self):
        other=object.__new__(type(self))
        other.__dict__=self.__dict__.copy()
        for name in ('capacity','background','demand','edge_prices'):
            setattr(other,name,getattr(self,name).copy())
        other.routes=self.routes.copy()  # SharedRoute buffers are immutable.
        other.net_weights=self.net_weights.copy()
        return other

    def _penalty(self, demand, keys=None):
        cap=self.capacity if keys is None else self.capacity[keys]
        safe=np.maximum(cap,1.)
        return (demand/safe)**2+4*(np.maximum(demand-cap,0)/safe)**2

    def _record(self,segments,pins=None,topology='flute'):
        metric=union_metrics(self.rg,np.asarray(segments,dtype=float).reshape(-1,2,2))
        points=np.empty((0,2)) if pins is None else np.asarray(pins,dtype=float)
        return SharedRoute(_readonly(points),_readonly(metric['segments']),
            _readonly(self.grid.edge_keys(metric['segments'])),metric['crossings'],metric['wirelength'],topology)

    def remove(self,net_id):
        previous=self.routes.pop(net_id,None)
        if previous is not None:self.demand[previous.keys]-=1
        return previous

    def _commit(self,net_id,route):
        if np.any(self.capacity[route.keys]<=0):raise ValueError('no resource-feasible shared tree')
        self.remove(net_id)
        self.routes[net_id]=route
        self.demand[route.keys]+=1
        return route

    def install_segments(self,net_id,segments,*,pins=None):
        return self._commit(net_id,self._record(segments,pins))

    def recompute_demand(self):
        demand=self.background.copy()
        for route in self.routes.values():demand[route.keys]+=1
        return demand

    def metrics(self):
        return dict(crossings=sum(r.crossings for r in self.routes.values()),
            weighted_crossings=sum(self.net_weights.get(n,1.)*r.crossings for n,r in self.routes.items()),
            wirelength=sum(r.wirelength for r in self.routes.values()),
            overflow=float(np.maximum(self.demand-self.capacity,0).sum()),
            congestion=float(self._penalty(self.demand).sum()),
            objective=float(sum(self.net_weights.get(n,1.)*r.crossings+self.wirelength_weight*r.wirelength
                                for n,r in self.routes.items())
                +self.congestion_weight*self._penalty(self.demand).sum()
                +np.dot(self.edge_prices,self.demand-self.background)),
            routed_nets=len(self.routes),multi_pin_nets=sum(len(r.pins)>2 for r in self.routes.values()),
            generation=self.generation,
            demand_sha256=hashlib.sha256(self.demand.tobytes()).hexdigest())

    def _candidate_rank(self,net_id,route,without):
        keys=route.keys
        incremental=(self._penalty(without[keys]+1,keys)-self._penalty(without[keys],keys)).sum()
        objective=(self.net_weights.get(net_id,1.)*route.crossings+self.wirelength_weight*route.wirelength
                   +self.congestion_weight*incremental+self.edge_prices[keys].sum())
        return (int(np.count_nonzero(self.capacity[keys]<=0)),float(objective),route.wirelength)

    def _polylines(self,a,b):
        yield np.array([a,[b[0],a[1]],[b[0],a[1]],b])
        yield np.array([a,[a[0],b[1]],[a[0],b[1]],b])
        for axis,tracks in ((0,self.grid.x_centers),(1,self.grid.y_centers)):
            low,high=sorted((a[axis],b[axis]))
            distance=np.maximum(low-tracks,0)+np.maximum(tracks-high,0)
            # Cover both endpoint neighborhoods and the middle, not only the
            # first tracks in a long bounding box with tied zero distance.
            anchors=np.array([a[axis],b[axis],(low+high)/2])
            indices=set(np.argsort(distance,kind='stable')[:self.candidate_tracks].tolist())
            for anchor in anchors:
                indices.update(np.argsort(np.abs(tracks-anchor),kind='stable')[:2].tolist())
            for index in sorted(indices):
                track=tracks[index]
                p,q=a.copy(),b.copy();p[axis]=q[axis]=track
                line=np.array([a,p,q,b])
                # A branch bound supplements the net union bound below.
                if np.abs(np.diff(line,axis=0)).sum()<=np.abs(b-a).sum()*(1+self.wirelength_budget)+1e-9:
                    yield line

    def replace(self,net_id,pins):
        """Rebuild FLUTE and optimize whole-tree unions against other nets.

        Replacement is atomic: even failed geometry/resource checks leave the
        previous route and demand intact. Call on a fork for placement trials.
        """
        pins=np.asarray(pins,dtype=float)
        if (pins.ndim!=2 or pins.shape[1]!=2 or not np.isfinite(pins).all()
                or (pins<self.rg.die[:2]).any() or (pins>self.rg.die[2:]).any()):
            raise ValueError('joint route pins must be finite and inside region die')
        edges,_=flute_edges(pins,coordinate_scale=self.coordinate_scale)
        a,b=edges[:,0],edges[:,1]
        lines=np.stack([a,np.column_stack((b[:,0],a[:,1])),np.column_stack((b[:,0],a[:,1])),b],axis=1)
        def record(current):
            pieces=np.stack((current[:,:-1],current[:,1:]),axis=2).reshape(-1,2,2)
            return self._record(pieces,pins)
        best=record(lines)
        baseline_length=best.wirelength
        without=self.demand.copy()
        if net_id in self.routes:without[self.routes[net_id].keys]-=1
        rank=self._candidate_rank(net_id,best,without)
        for i,(start,end) in enumerate(edges):
            keep=lines[i].copy()
            for line in self._polylines(start,end):
                if (line<self.rg.die[:2]).any() or (line>self.rg.die[2:]).any():continue
                lines[i]=line
                candidate=record(lines)
                if candidate.wirelength>baseline_length*(1+self.wirelength_budget)+1e-9:continue
                candidate_rank=self._candidate_rank(net_id,candidate,without)
                if candidate_rank<rank:
                    best,rank,keep=candidate,candidate_rank,line.copy()
            lines[i]=keep
        if rank[0]:raise ValueError('no resource-feasible shared FLUTE tree within budget')
        return self._commit(net_id,best)
