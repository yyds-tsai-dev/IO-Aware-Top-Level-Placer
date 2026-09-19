"""Frozen FLUTE/detour graph with genuine GP IO, length and capacity derivatives.

Raw terminal coordinates remain live; Steiner nodes and selected L/Z track
choices remain fixed until rebuild. GPU geometric union is recomputed in each
forward. Its derivatives use the documented piecewise/frozen-track grouping.
"""
import hashlib
from copy import copy
import math
import numpy as np
import torch
from ioplace.route_eval.topology_batch import batch_flute_trees
from ioplace.route_eval.joint import JointRoutingState, ResourceGrid, SharedRoute
from ioplace.route_eval.topology import segment_union
from ioplace.ops.route_tensor import union_intervals,grid_crossings,joint_demand,congestion_cost


class FrozenJointRouteCost:
    def __init__(self,nl,region_grid,resource_grid,capacity,*,num_nodes,
                 background=None,net_mask=None,net_weights=None,hot_nets=16,
                 wirelength_budget=.05,coordinate_scale=1000.,flute_threads=8,
                 io_weight=1.,congestion_weight=.1,wirelength_weight=.001):
        self.nl,self.rg,self.grid=nl,region_grid,resource_grid
        self.num_nodes=int(num_nodes);self.net_mask=net_mask;self.net_weights=net_weights
        self.capacity_np=np.asarray(capacity,dtype=float).copy()
        self.background_np=(np.zeros(resource_grid.edge_count) if background is None
                            else np.asarray(background,dtype=float).copy())
        for array in (self.capacity_np,self.background_np):
            if array.shape!=(resource_grid.edge_count,) or not np.isfinite(array).all() or (array<0).any():
                raise ValueError('invalid capacity/background')
        if hot_nets<0 or not np.isfinite(wirelength_budget) or wirelength_budget<0:
            raise ValueError('invalid hot-net count/budget')
        self.hot_nets=int(hot_nets);self.budget=wirelength_budget
        self.scale=coordinate_scale;self.threads=flute_threads
        self.weights=(io_weight,congestion_weight,wirelength_weight)
        if not all(np.isfinite(w) and w>=0 for w in self.weights):raise ValueError('invalid routing weights')
        self.generation=0
        self.io_calibration_np=np.ones(nl.num_nets)
        self.edge_prices_np=np.zeros(resource_grid.edge_count)
        self.observation_version=0;self.published_observation_version=0
        self.observations=[]

    def _tensor(self,value,pos,integer=False):
        return torch.as_tensor(value,dtype=torch.long if integer else pos.dtype,device=pos.device)

    def _ends(self,pos):
        a=torch.where(self.a_idx>=0,pos[self.a_idx.clamp_min(0)]+self.a_value,self.a_value)
        b=torch.where(self.b_idx>=0,pos[self.b_idx.clamp_min(0)]+self.b_value,self.b_value)
        return a,b

    def intervals(self,pos):
        a,b=self._ends(pos)
        # Four-point templates: H, V, Z through fixed x, Z through fixed y.
        horizontal=(self.modes==0)|(self.modes==2)
        xtrack=torch.where(self.modes==2,self.tracks,b[:,0])
        ytrack=torch.where(self.modes==3,self.tracks,b[:,1])
        p=torch.stack((torch.where(horizontal,xtrack,a[:,0]),
                       torch.where(horizontal,a[:,1],ytrack)),dim=1)
        q=torch.stack((torch.where(horizontal,xtrack,b[:,0]),
                       torch.where(horizontal,b[:,1],ytrack)),dim=1)
        # H/V use a duplicated elbow. Z modes keep both elbows distinct.
        q=torch.where(((self.modes==0)|(self.modes==1))[:,None],p,q)
        points=torch.stack((a,p,q,b),dim=1)
        axes=self.axes_templates[self.modes].reshape(-1)
        starts,ends=points[:,:-1].reshape(-1,2),points[:,1:].reshape(-1,2)
        index=axes[:,None]
        return union_intervals(self.edge_nets.repeat_interleave(3),axes,
            starts.gather(1,1-index).flatten(),starts.gather(1,index).flatten(),
            ends.gather(1,index).flatten())

    def components(self,pos,*,tau,gamma,hard=False):
        if self.generation==0:raise RuntimeError('route graph must be rebuilt before evaluation')
        if not math.isfinite(float(gamma)) or float(gamma)<=0:raise ValueError('finite positive gamma required')
        union=self.intervals(pos)
        io_events=grid_crossings(union,self.io_x,self.io_y,tau,hard=hard)
        io_values=io_events['values']*self.boundaries[io_events['edge_ids']]
        io_raw=io_values.sum()
        io=(io_values*self.io_weights[io_events['net_ids']]).sum()
        events=grid_crossings(union,self.resource_x,self.resource_y,tau,hard=hard)
        demand=joint_demand(events,self.grid.edge_count,self.background)
        congestion=congestion_cost(demand,self.capacity)
        price=(self.edge_prices*(demand-self.background)).sum()
        length=union['high']-union['low']
        wirelength=length.sum() if hard else (length*torch.tanh(length/(2*gamma))).sum()
        objective=self.weights[0]*io+self.weights[1]*congestion+self.weights[2]*wirelength+price
        return dict(io=io,io_raw=io_raw,congestion=congestion,price=price,wirelength=wirelength,demand=demand,objective=objective,
                    io_events=io_events,resource_events=events)

    def __call__(self,pos,*,tau,gamma):
        return self.components(pos,tau=tau,gamma=gamma)['objective']

    def paper(self,pos,gamma):
        """Published Eq.7 interior correction from the same raw FLUTE batch."""
        if not math.isfinite(float(gamma)) or float(gamma)<=0:raise ValueError('finite positive gamma required')
        points=torch.where(self.paper_idx>=0,pos[self.paper_idx.clamp_min(0)]+self.paper_value,self.paper_value)
        delta=points-self.paper_anchor
        value=(delta*torch.tanh(delta/(2*gamma))).sum(dim=1)
        return (value*self.base_net_weights[self.paper_net_ids]).sum()

    def assimilate(self,observation,measured_pos):
        """Validate a routed legal placement; queue learned data for next rebuild.

        The prediction denominator is rebuilt at the actually routed positions,
        not at a different, possibly illegal current GP position. The active
        frozen graph remains unchanged until the controller publishes a rebuild.
        """
        grid=observation['grid']
        if not np.array_equal(grid.x_edges,self.grid.x_edges) or not np.array_equal(grid.y_edges,self.grid.y_edges):
            raise ValueError('router resource grid changed')
        usage=np.asarray(observation['usage'],float);capacity=np.asarray(observation['capacity'],float)
        if (usage.shape!=self.capacity_np.shape or capacity.shape!=usage.shape
                or not np.isfinite(usage).all() or not np.isfinite(capacity).all()
                or (usage<0).any() or (capacity<0).any()):raise ValueError('invalid router resources')
        actual={int(k):float(v) for k,v in observation['net_io'].items()}
        if any(not np.isfinite(v) or v<0 for v in actual.values()):raise ValueError('invalid router net IO')
        totals=np.asarray([observation['actual_io'],observation['actual_wirelength']],float)
        if not np.isfinite(totals).all() or (totals<0).any() or not np.isclose(totals[0],sum(actual.values()),rtol=0,atol=1e-6):
            raise ValueError('invalid router totals')
        if any(k<0 or k>=self.nl.num_nets for k in actual):raise ValueError('invalid router net ID')
        measured=copy(self)
        orientations=observation.get('movable_orientations')
        if orientations is not None:
            from ioplace.ops.placement_offsets import oriented_netlist
            measured.nl=oriented_netlist(self.nl,orientations)
        measured.rebuild(measured_pos)
        with torch.no_grad():
            parts=measured.components(measured_pos,tau=1.,gamma=1.,hard=True)
            events=parts['io_events']
            raw=events['values']*measured.boundaries[events['edge_ids']]
            predicted=measured_pos.new_zeros(self.nl.num_nets).scatter_add(0,events['net_ids'],raw).cpu().numpy()
            union=measured.intervals(measured_pos)
            lengths=measured_pos.new_zeros(self.nl.num_nets).scatter_add(
                0,union['net_ids'],union['high']-union['low']).cpu().numpy()
        foreground=np.zeros_like(usage);weights=self.io_calibration_np.copy()
        for net in measured.selected_nets:
            if net not in actual and lengths[net]>1e-9:raise ValueError('router omitted supported nonzero net')
            weights[net]=.5*weights[net]+.5*np.clip((actual.get(int(net),0.)+1)/(predicted[net]+1),.25,8.)
            pernet=observation['net_usage'].get(int(net))
            if pernet is None:
                if lengths[net]>1e-9:raise ValueError('router omitted net layer usage')
                continue
            keys=np.asarray(pernet['keys'],dtype=int);counts=np.asarray(pernet['counts'],float)
            if (keys.ndim!=1 or counts.shape!=keys.shape or len(np.unique(keys))!=len(keys)
                    or (keys<0).any() or (keys>=len(usage)).any() or not np.isfinite(counts).all()
                    or (counts<0).any()):raise ValueError('invalid router layer demand')
            foreground[keys]+=counts
        background=np.maximum(usage-foreground,0.)
        prices=.5*self.edge_prices_np+.5*(usage/np.maximum(capacity,1.))**2
        record=dict(observation_version=self.observation_version+1,
            source_sha256=observation['source_sha256'],placement_sha256=observation['placement_sha256'],
            measured_position_sha256=hashlib.sha256(measured_pos.detach().cpu().numpy().tobytes()).hexdigest(),
            changed_net_weights=int(np.count_nonzero(weights!=self.io_calibration_np)),
            background_sha256=hashlib.sha256(background.tobytes()).hexdigest(),
            calibration_predictor='same hot-net FLUTE/detour policy at measured positions',
            calibration_hot_nets=self.hot_nets,
            measured_pin_model='oriented row pins' if orientations is not None else 'static GP offsets',
            measured_pin_offsets_sha256=hashlib.sha256(measured.nl.pin_offset_x.tobytes()+measured.nl.pin_offset_y.tobytes()).hexdigest())
        self.io_calibration_np=weights;self.capacity_np=capacity.copy()
        self.background_np=background;self.edge_prices_np=prices
        self.observation_version+=1;self.observations=[*self.observations,record]
        return record

    @staticmethod
    def _line(a,b,mode,track):
        if mode==0:return np.array([a,[b[0],a[1]],[b[0],a[1]],b])
        if mode==1:return np.array([a,[a[0],b[1]],[a[0],b[1]],b])
        if mode==2:return np.array([a,[track,a[1]],[track,b[1]],b])
        return np.array([a,[a[0],track],[b[0],track],b])

    def _choices(self,a,b):
        yield 0,0.,self._line(a,b,0,0.)
        yield 1,0.,self._line(a,b,1,0.)
        for mode,axis,tracks in ((2,0,self.grid.x_centers),(3,1,self.grid.y_centers)):
            candidates=set()
            for anchor in (a[axis],b[axis],(a[axis]+b[axis])/2):
                candidates.update(np.argsort(np.abs(tracks-anchor),kind='stable')[:2].tolist())
            for index in sorted(candidates):
                track=float(np.asarray(tracks[index],dtype=a.dtype));line=self._line(a,b,mode,track)
                if np.abs(np.diff(line,axis=0)).sum()<=np.abs(b-a).sum()*(1+self.budget)+1e-9:
                    yield mode,float(track),line

    def rebuild(self,pos):
        trial=copy(self)
        metadata=trial._rebuild(pos)
        self.__dict__.update(trial.__dict__)
        return metadata

    def _rebuild(self,pos):
        if pos.ndim!=1 or len(pos)!=2*self.num_nodes:raise ValueError('invalid flattened positions')
        coords=pos.detach().cpu().numpy()
        if not np.isfinite(coords).all():raise ValueError('nonfinite positions')
        degrees=np.asarray(self.nl.net_degrees)
        enabled=np.ones(len(degrees),dtype=bool) if self.net_mask is None else np.asarray(
            self.net_mask.detach().cpu() if torch.is_tensor(self.net_mask) else self.net_mask,dtype=bool)
        if enabled.shape!=degrees.shape:raise ValueError('invalid net mask shape')
        supported=enabled&(degrees>=2)&(degrees<=256)
        selected=np.flatnonzero(supported)
        self.selected_nets=selected
        chosen_pins=np.asarray(self.nl.flat_net2pin)[np.repeat(supported,degrees)]
        owners=np.asarray(self.nl.pin2node)[chosen_pins]
        offsets=np.column_stack((self.nl.pin_offset_x[chosen_pins],self.nl.pin_offset_y[chosen_pins]))
        pins=np.column_stack((coords[owners],coords[self.num_nodes+owners]))+offsets
        starts=np.r_[0,np.cumsum(degrees[selected])]
        tree,provenance=batch_flute_trees(pins,starts,coordinate_scale=self.scale,threads=self.threads)
        rows=tree['positions'];parents=tree['parents'];terminal=tree['terminal_nodes']
        raw_idx=np.full(rows.shape,-1,dtype=np.int64);raw_value=rows.copy()
        movable=owners<self.nl.num_movable
        raw_idx[terminal[movable]]=np.column_stack((owners[movable],self.num_nodes+owners[movable]))
        raw_value[terminal[movable]]=offsets[movable]
        raw_value[terminal[~movable]]=pins[~movable]
        if len(selected):
            low=np.minimum.reduceat(pins,starts[:-1]);high=np.maximum.reduceat(pins,starts[:-1])
            inner=np.all((pins>np.repeat(low,degrees[selected],axis=0))
                &(pins<np.repeat(high,degrees[selected],axis=0)),axis=1)&np.repeat(degrees[selected]>3,degrees[selected])
        else:inner=np.zeros(0,dtype=bool)
        self.paper_idx=self._tensor(raw_idx[terminal[inner]],pos,True)
        self.paper_value=self._tensor(raw_value[terminal[inner]],pos)
        self.paper_anchor=self._tensor(rows[parents[terminal[inner]]],pos)
        self.paper_net_ids=self._tensor(np.repeat(selected,degrees[selected])[inner],pos,True)
        source=np.flatnonzero(parents!=np.arange(len(parents)))
        raw_nets=np.repeat(selected,np.diff(tree['branch_starts']))
        self.edge_nets=self._tensor(raw_nets[source],pos,True)
        self.a_idx=self._tensor(raw_idx[source],pos,True);self.b_idx=self._tensor(raw_idx[parents[source]],pos,True)
        self.a_value=self._tensor(raw_value[source],pos);self.b_value=self._tensor(raw_value[parents[source]],pos)
        self.modes=torch.zeros(len(source),device=pos.device,dtype=torch.long)
        self.tracks=pos.new_zeros(len(source))
        self.axes_templates=self._tensor([[0,0,1],[1,1,0],[0,1,0],[1,0,1]],pos,True)
        self.capacity=self._tensor(self.capacity_np,pos);self.background=self._tensor(self.background_np,pos)
        self.edge_prices=self._tensor(self.edge_prices_np,pos)
        self.base_net_weights=self._tensor(np.ones(self.nl.num_nets) if self.net_weights is None else self.net_weights,pos)
        if self.base_net_weights.shape!=(self.nl.num_nets,) or not bool(torch.isfinite(self.base_net_weights).all()) or bool((self.base_net_weights<0).any()):
            raise ValueError('invalid net weights')
        self.io_weights=self.base_net_weights*self._tensor(self.io_calibration_np,pos)
        self.resource_x=self._tensor(self.grid.x_edges,pos);self.resource_y=self._tensor(self.grid.y_edges,pos)
        xl,yl,xh,yh=self.rg.die
        self.io_x=self._tensor(np.linspace(xl,xh,self.rg.nx+1),pos)
        self.io_y=self._tensor(np.linspace(yl,yh,self.rg.ny+1),pos)
        ids=self.rg.grid
        self.boundaries=self._tensor(np.r_[(ids[:,1:]!=ids[:,:-1]).ravel(),(ids[1:,:]!=ids[:-1,:]).ravel()],pos)
        # CPU candidates use exactly the boundaries represented on the GPU.
        self.effective_grid=ResourceGrid(self.resource_x.cpu().numpy(),self.resource_y.cpu().numpy())
        self.effective_io_grid=ResourceGrid(self.io_x.cpu().numpy(),self.io_y.cpu().numpy())
        self.boundaries_np=self.boundaries.cpu().numpy()
        self.hot_demand_consistent=True
        self.generation+=1
        self.published_observation_version=self.observation_version
        with torch.no_grad():
            before=self.components(pos,tau=1.,gamma=1.,hard=True)
            initial_wirelength=float(before['wirelength'])
            initial_objective=float(before['objective'])
            trials=self._select_detours(pos,selected,before) if self.hot_nets else 0
            after=self.components(pos,tau=1.,gamma=1.,hard=True)
            if trials and not np.allclose(self.hot_final_demand,after['demand'].cpu().numpy(),rtol=1e-6,atol=1e-5):
                raise RuntimeError('hot candidate and full tensor demand disagree')
        return dict(generation=self.generation,eligible_nets=len(selected),paper_interior_pins=int(inner.sum()),
            published_observation_version=self.published_observation_version,
            eligible_two_pin_nets=int(np.sum(degrees[selected]==2)),
            eligible_multi_pin_nets=int(np.sum(degrees[selected]>2)),
            excluded_degree=int(np.sum(enabled&(degrees>256))),masked_nets=int(np.sum(~enabled)),
            raw_edges=len(source),detour_net_trials=trials,hot_demand_consistent=self.hot_demand_consistent,
            blocked_net_edge_uses_before=float((before['demand']-self.background)[self.capacity<=0].sum()),
            blocked_net_edge_uses_after=float((after['demand']-self.background)[self.capacity<=0].sum()),
            blocked_edges_before=int(((before['demand']>self.background)&(self.capacity<=0)).sum()),
            blocked_edges_after=int(((after['demand']>self.background)&(self.capacity<=0)).sum()),
            selection_order='blocked resource count, scalar objective, wirelength',
            hard_wirelength_before=initial_wirelength,hard_wirelength_after=float(after['wirelength']),
            hard_objective_before=initial_objective,hard_objective_after=float(after['objective']),
            hard_io=float(after['io_raw']),hard_weighted_io=float(after['io']),hard_overflow=float(torch.relu(after['demand']-self.capacity).sum()),
            demand_sha256=hashlib.sha256(after['demand'].cpu().numpy().tobytes()).hexdigest(),
            flute_provenance=provenance)

    def _hard_record(self,lines):
        segments=segment_union(lines)
        keys=self.effective_grid.edge_keys(segments)
        # Distinct physical segments count separately; resource keys are unique.
        io=sum(self.boundaries_np[self.effective_io_grid.edge_keys([segment])].sum()
               for segment in segments)
        return SharedRoute(np.empty((0,2)),segments,keys,int(io),
                           float(np.abs(np.diff(segments,axis=1)).sum()))

    def _select_detours(self,pos,selected,before):
        if not len(selected):return 0
        score=pos.new_zeros(self.nl.num_nets)
        events=before['io_events']
        score.scatter_add_(0,events['net_ids'],events['values']*self.boundaries[events['edge_ids']])
        events=before['resource_events'];keys=events['edge_ids']
        pressure=(before['demand'][keys]/self.capacity[keys].clamp_min(1))**2
        score.scatter_add_(0,events['net_ids'],events['values']*pressure*self.weights[1])
        hot=torch.argsort(score,descending=True,stable=True)[:self.hot_nets].cpu().numpy()
        a,b=(v.detach().cpu().numpy() for v in self._ends(pos))
        edge_nets=self.edge_nets.cpu().numpy()
        modes=np.zeros(len(a),dtype=np.int64);tracks=np.zeros(len(a))
        demand=before['demand'].cpu().numpy().astype(float);trials=0
        scorer=JointRoutingState(self.rg,self.effective_grid,self.capacity.cpu().numpy(),
            congestion_weight=self.weights[1],wirelength_weight=self.weights[2],wirelength_budget=self.budget)
        weights=self.io_weights.detach().cpu().numpy()
        scorer.net_weights={int(n):float(self.weights[0]*weights[n]) for n in selected}
        scorer.edge_prices=self.edge_prices.cpu().numpy().copy()
        for net in hot:
            indices=np.flatnonzero(edge_nets==net)
            if not len(indices):continue
            trials+=1
            lines=np.stack([self._line(a[i],b[i],0,0.) for i in indices])
            def record():
                return self._hard_record(lines)
            best=record();old_keys=best.keys;without=demand.copy();without[old_keys]-=1
            baseline_length=best.wirelength;rank=scorer._candidate_rank(int(net),best,without)
            for local,edge in enumerate(indices):
                keep=lines[local].copy();chosen=(0,0.)
                for mode,track,line in self._choices(a[edge],b[edge]):
                    lines[local]=line;candidate=record()
                    if candidate.wirelength>baseline_length*(1+self.budget)+1e-8:continue
                    proposed=scorer._candidate_rank(int(net),candidate,without)
                    if proposed<rank:
                        best,rank,keep,chosen=candidate,proposed,line.copy(),(mode,track)
                lines[local]=keep;modes[edge],tracks[edge]=chosen
            demand[old_keys]-=1;demand[best.keys]+=1
        self.hot_final_demand=demand
        self.modes.copy_(self._tensor(modes,pos,True));self.tracks.copy_(self._tensor(tracks,pos))
        return trials
