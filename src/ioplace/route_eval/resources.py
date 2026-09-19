"""Frozen 2-D lattice resource costs for candidate routing.

Capacity zero forbids an edge. Positive capacity incurs incremental squared
utilization cost using externally supplied background demand. This does not
reserve resources between nets or model layers, vias, or detailed-route DRC.
"""
import numpy as np


class RoutingResources:
    def __init__(self, region_grid, horizontal_capacity, vertical_capacity, *,
                 horizontal_demand=None, vertical_demand=None, congestion_weight=1.):
        self.rg = region_grid
        ny, nx = region_grid.grid.shape
        if not np.isfinite(congestion_weight) or congestion_weight < 0:
            raise ValueError('finite nonnegative congestion weight required')
        self.cost, self.blocked = [], []
        for capacity, demand, shape, axis in (
                (horizontal_capacity, horizontal_demand, (ny,nx-1), 1),
                (vertical_capacity, vertical_demand, (ny-1,nx), 0)):
            capacity = np.asarray(capacity, dtype=float)
            demand = np.zeros(shape) if demand is None else np.asarray(demand, dtype=float)
            if (capacity.shape != shape or demand.shape != shape or not np.isfinite(capacity).all()
                    or not np.isfinite(demand).all() or (capacity < 0).any() or (demand < 0).any()):
                raise ValueError('resource shapes must match lattice edges; finite nonnegative values required')
            blocked = capacity == 0
            safe_capacity = np.where(blocked, 1., capacity)
            cost = congestion_weight * (2*demand+1) / safe_capacity**2
            pad = ((0,0),(1,0)) if axis == 1 else ((1,0),(0,0))
            self.cost.append(np.pad(np.cumsum(cost, axis=axis),pad))
            self.blocked.append(np.pad(np.cumsum(blocked, axis=axis),pad))

    def evaluate(self, points):
        points = np.asarray(points)
        cost, blocked = np.zeros(len(points)), np.zeros(len(points),dtype=np.int64)
        for a,b in zip(points[:,:-1].transpose(1,0,2), points[:,1:].transpose(1,0,2)):
            ax,ay=self.rg.to_idx(a[:,0],a[:,1]);bx,by=self.rg.to_idx(b[:,0],b[:,1])
            for total, fields in ((cost,self.cost),(blocked,self.blocked)):
                total += np.abs(fields[0][ay,bx]-fields[0][ay,ax])
                total += np.abs(fields[1][by,bx]-fields[1][ay,bx])
        return cost, blocked == 0
