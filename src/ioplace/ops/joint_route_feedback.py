"""Placement proposals scored by multi-pin FLUTE and joint routing resources."""
from dataclasses import dataclass
from copy import copy
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from ioplace.route_eval.joint import JointRoutingState, ResourceGrid
from ioplace.route_eval.topology import flute_edges


def rsmt_anchors(pins,edges):
    """Fig. 5 / Algorithm 2 anchor multiset, adapted to exact pin coordinates.

    Direct pin-to-pin branches use a midpoint; other branches use the adjacent
    Steiner endpoint. Colocated pins retain self anchors for zero-length links.
    Unlike the paper's quadtree refinement, callers score legal site swaps.
    """
    pins=np.asarray(pins,dtype=float)
    physical={tuple(p) for p in pins};adjacency={}
    seen=set()
    for a,b in np.asarray(edges,dtype=float).reshape(-1,2,2):
        aa,bb=tuple(a),tuple(b)
        edge=tuple(sorted((aa,bb)))
        if aa==bb or edge in seen:continue
        seen.add(edge)
        midpoint=(a+b)/2
        adjacency.setdefault(aa,[]).append(midpoint if aa in physical and bb in physical else b)
        adjacency.setdefault(bb,[]).append(midpoint if aa in physical and bb in physical else a)
    result=[]
    for p in pins:
        anchors=list(adjacency.get(tuple(p),[]));degree=len(anchors)
        overlap=np.count_nonzero(np.all(pins==p,axis=1))>1
        extra=degree if degree in (1,2) and overlap else int(degree==3)
        anchors.extend([p]*extra)
        result.append(np.asarray(anchors or [p],dtype=float).reshape(-1,2))
    return result


@dataclass
class JointProposal:
    x: np.ndarray
    y: np.ndarray
    state: JointRoutingState
    diagnostics: dict


def accept_joint_move(original,incumbent,candidate,*,original_hpwl,candidate_hpwl,
                      legal,fixed_unchanged,hpwl_budget=.05):
    """Compare costs only within one learned generation; budgets never drift."""
    if not np.isfinite(hpwl_budget) or hpwl_budget<0:raise ValueError('invalid budget')
    reasons=[]
    if not legal:reasons.append('illegal')
    if not fixed_unchanged:reasons.append('fixed_nodes_changed')
    if candidate is None:return dict(accepted=False,reasons=reasons+['invalid_candidate'])
    for name,metric in (('original',original),('incumbent',incumbent),('candidate',candidate)):
        if not all(np.isfinite(metric[k]) and metric[k]>=0 for k in ('objective','wirelength','overflow')):
            reasons.append(name+'_invalid')
    if not np.isfinite([original_hpwl,candidate_hpwl]).all() or min(original_hpwl,candidate_hpwl)<0:
        reasons.append('invalid_hpwl')
    if candidate['generation']!=incumbent['generation']:reasons.append('observation_generation_mismatch')
    if candidate['objective']>=incumbent['objective']-1e-9:reasons.append('no_joint_cost_improvement')
    if candidate['wirelength']>original['wirelength']*(1+hpwl_budget)+1e-9:reasons.append('original_route_budget')
    if candidate_hpwl>original_hpwl*(1+hpwl_budget)+1e-9:reasons.append('original_hpwl_budget')
    if candidate['overflow']>incumbent['overflow']+1e-9:reasons.append('increased_joint_overflow')
    return dict(accepted=not reasons,reasons=reasons)


class JointRouteFeedback:
    def __init__(self,nl,region_grid,node_x,node_y,*,state=None,max_degree=256,
                 max_displacement_cells=8.,congestion_weight=.1,wirelength_budget=.05):
        if not 2<=max_degree<=256 or not np.isfinite(max_displacement_cells) or max_displacement_cells<=0:
            raise ValueError('invalid joint feedback degree/displacement')
        self.nl,self.rg,self.cap=nl,region_grid,max_displacement_cells
        if state is None:
            grid=ResourceGrid(np.linspace(nl.xl,nl.xh,region_grid.nx+1),
                              np.linspace(nl.yl,nl.yh,region_grid.ny+1))
            state=JointRoutingState(region_grid,grid,np.full(grid.edge_count,10.),
                                    congestion_weight=congestion_weight,wirelength_budget=wirelength_budget)
        if state.routes:raise ValueError('joint feedback requires an initially empty routing state')
        self.state=state
        self.net_ids=[];outside=0
        for net in np.flatnonzero((nl.net_degrees>=2)&(nl.net_degrees<=max_degree)):
            pins=self.pin_positions(int(net),node_x,node_y)
            if (pins<region_grid.die[:2]).any() or (pins>region_grid.die[2:]).any():outside+=1;continue
            self.net_ids.append(int(net))
            state.replace(int(net),pins)
        supported=set(self.net_ids)
        self.incident=[set() for _ in range(nl.num_movable)]
        for node,net in zip(nl.pin2node,nl.pin2net):
            if node<nl.num_movable:self.incident[int(node)].add(int(net))
        # Do not move cells with any unmodeled nontrivial net. Router-measured
        # excluded-net demand remains background, so moving it would be stale.
        self.movable=np.array([all(net in supported or nl.net_degrees[net]<2 for net in row)
                               for row in self.incident])
        ids=np.asarray(self.net_ids,dtype=np.int64)
        self.cohort=dict(eligible_nets=len(ids),eligible_multi_pin_nets=int(np.sum(nl.net_degrees[ids]>2)),
            eligible_two_pin_nets=int(np.sum(nl.net_degrees[ids]==2)),excluded_outside_die=outside,
            excluded_degree=int(np.sum((nl.net_degrees<2)|(nl.net_degrees>max_degree))),
            net_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            max_degree=max_degree,frozen_nodes_with_unmodeled_incidents=int(np.count_nonzero(~self.movable)))
        self.observations=[]

    def pin_indices(self,net):
        return self.nl.flat_net2pin[self.nl.flat_net2pin_start[net]:self.nl.flat_net2pin_start[net+1]]

    def pin_positions(self,net,x,y):
        pins=self.pin_indices(net);nodes=self.nl.pin2node[pins]
        return np.column_stack((np.asarray(x)[nodes]+self.nl.pin_offset_x[pins],
                                np.asarray(y)[nodes]+self.nl.pin_offset_y[pins]))

    def priorities(self):
        result=np.zeros(self.nl.num_movable)
        for net,route in self.state.routes.items():
            # FLUTE geometry and joint resource price both affect which cells
            # receive a finite-difference/swap evaluation budget.
            pressure=(self.state.edge_prices[route.keys].sum()
                +np.maximum(self.state.demand[route.keys]-self.state.capacity[route.keys],0).sum())
            weight=self.state.net_weights.get(net,1.)
            value=weight*(route.crossings+self.state.wirelength_weight*route.wirelength)+pressure
            nodes=np.unique(self.nl.pin2node[self.pin_indices(net)])
            nodes=nodes[nodes<self.nl.num_movable]
            result[nodes]+=value
        result[~self.movable]=0
        return result

    def topology_anchors(self,node,x,y):
        """Paper-inspired branch anchors, plus the per-net weighted median.

        Preserve multiset multiplicity for the normalized Manhattan cost in
        Eq. 8. Use its weighted median and individual anchors as search seeds.
        Rebuilt joint FLUTE cost determines acceptance of each legal swap.
        """
        anchors=[]
        weights=[]
        for net in sorted(self.incident[node]):
            if net not in self.state.routes:continue
            route=self.state.routes[net]
            edges,_=flute_edges(route.pins,coordinate_scale=self.state.coordinate_scale)
            per_pin=rsmt_anchors(route.pins,edges)
            per_net=[]
            for index,pin in enumerate(self.pin_indices(net)):
                if self.nl.pin2node[pin]!=node:continue
                offset=np.array([self.nl.pin_offset_x[pin],self.nl.pin_offset_y[pin]])
                per_net.extend(per_pin[index]-offset)
            if per_net:
                anchors.extend(per_net)
                weights.extend([self.state.net_weights.get(net,1.)/len(per_net)]*len(per_net))
        anchors=np.asarray(anchors,dtype=float).reshape(-1,2)
        if not len(anchors):return anchors
        weights=np.asarray(weights);median=[]
        for axis in (0,1):
            order=np.argsort(anchors[:,axis],kind='stable')
            median.append(anchors[order[np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)],axis])
        return np.unique(np.vstack((anchors,median)),axis=0)

    def propose(self,x,y,*,max_active=32,neighbors=8,fraction=1.):
        if max_active<0 or neighbors<1 or not np.isfinite(fraction) or not 0<fraction<=1:
            raise ValueError('invalid joint proposal settings')
        x,y=np.array(x,dtype=float,copy=True),np.array(y,dtype=float,copy=True)
        current=self.state.fork();nm=self.nl.num_movable
        if not nm:return JointProposal(x,y,current,dict(attempted=0,accepted_swaps=[],multi_pin_rebuilds=0))
        priority=self.priorities()
        active=np.flatnonzero(priority>0)
        active=sorted(active,key=lambda i:(-priority[i],int(i)))[:max_active]
        tree=cKDTree(np.column_stack((x[:nm],y[:nm])))
        used=set();accepted=[];attempted=multi=0;global_best=None
        for node in active:
            if int(node) in used:continue
            target=np.zeros(2);weight=0.
            for net in self.incident[node]:
                if net not in current.routes:continue
                route=current.routes[net]
                w=current.net_weights.get(net,1.)*max(route.crossings,1)
                target+=route.pins.mean(axis=0)*w;weight+=w
            center=np.array([x[node],y[node]])
            target=target/weight if weight else center
            direction=target-center
            distance=np.linalg.norm(direction/[self.rg.cell_w,self.rg.cell_h])
            target=center+direction*min(1.,self.cap*fraction/max(distance,1e-300))
            _,near=tree.query(center,k=min(neighbors,nm))
            _,toward=tree.query(target,k=min(neighbors,nm))
            candidates=set(np.atleast_1d(near).tolist()+np.atleast_1d(toward).tolist())
            for anchor in self.topology_anchors(node,x,y):
                _,anchored=tree.query(anchor,k=min(neighbors,nm))
                candidates.update(np.atleast_1d(anchored).tolist())
            best=None;before=current.metrics()
            for other in sorted(candidates):
                if other==node or other in used or not self.movable[other]:continue
                if (self.nl.node_size_x[node]!=self.nl.node_size_x[other]
                        or self.nl.node_size_y[node]!=self.nl.node_size_y[other]):continue
                shift=np.array([x[other]-x[node],y[other]-y[node]])
                if np.linalg.norm(shift/[self.rg.cell_w,self.rg.cell_h])>self.cap*fraction:continue
                ids=sorted((self.incident[node]|self.incident[other])&set(current.routes))
                trial=current.fork()
                for net in ids:trial.remove(net)
                x[node],x[other]=x[other],x[node];y[node],y[other]=y[other],y[node]
                attempted+=1
                try:
                    for net in ids:
                        trial.replace(net,self.pin_positions(net,x,y))
                        multi+=int(self.nl.net_degrees[net]>2)
                    metric=trial.metrics()
                except ValueError:
                    metric=None
                finally:
                    x[node],x[other]=x[other],x[node];y[node],y[other]=y[other],y[node]
                if (metric is not None and metric['objective']<before['objective']-1e-9
                        and metric['overflow']<=before['overflow']+1e-9):
                    rank=(metric['objective'],metric['wirelength'],other)
                    if best is None or rank<best[0]:best=(rank,other,trial,metric)
            if best and (global_best is None or best[0]<global_best[0]):
                global_best=(*best,int(node),before)
        if global_best:
            _,other,current,metric,node,before=global_best
            x[node],x[other]=x[other],x[node];y[node],y[other]=y[other],y[node]
            accepted.append(dict(nodes=[int(node),int(other)],objective_before=before['objective'],
                                 objective_after=metric['objective'],generation=current.generation))
        return JointProposal(x,y,current,dict(attempted=attempted,accepted_swaps=accepted,
            multi_pin_rebuilds=multi,active_nodes=list(map(int,active)),
            priority_sha256=hashlib.sha256(priority.tobytes()).hexdigest(),generation=current.generation))

    def observe(self,observation,*,evaluated_state,retained):
        """Learn from routed proposals, including rejected placements.

        Only an observation of the retained placement refreshes background
        demand. A rejected placement still updates per-net error weights and
        spatial resource prices; its geometry is never committed.
        """
        state=self.state
        grid=observation.get('grid')
        if grid is not None and (not np.array_equal(grid.x_edges,state.grid.x_edges)
                or not np.array_equal(grid.y_edges,state.grid.y_edges)):
            raise ValueError('router resource grid changed')
        usage=np.asarray(observation['usage'],dtype=float)
        capacity=np.asarray(observation['capacity'],dtype=float)
        if (usage.shape!=state.demand.shape or capacity.shape!=usage.shape
                or not np.isfinite(usage).all() or not np.isfinite(capacity).all()
                or (usage<0).any() or (capacity<0).any()):
            raise ValueError('invalid router resource observation')
        net_io={int(k):float(v) for k,v in observation['net_io'].items()}
        if any(not np.isfinite(v) or v<0 for v in net_io.values()):raise ValueError('invalid measured net IO')
        missing=set(evaluated_state.routes)-set(net_io)
        if any(evaluated_state.routes[n].wirelength>1e-9 for n in missing):
            raise ValueError('router observation omitted supported nonzero nets')
        if (not np.isfinite([observation['actual_io'],observation['actual_wirelength']]).all()
                or min(observation['actual_io'],observation['actual_wirelength'])<0
                or not np.isclose(sum(net_io.values()),observation['actual_io'])):
            raise ValueError('invalid router totals')
        prior=state.metrics()
        trial=state.fork()
        observed_foreground=np.zeros_like(usage)
        for net,route in evaluated_state.routes.items():
            actual=net_io.get(net,0.)
            ratio=np.clip((actual+1)/(route.crossings+1),.25,8.)
            trial.net_weights[net]=.5*trial.net_weights.get(net,1.)+.5*float(ratio)
            measured=observation.get('net_usage',{}).get(net)
            if measured is None:
                keys=np.unique(np.asarray(observation['net_keys'].get(net,[]),dtype=int))
                counts=np.ones(len(keys))
            else:
                keys=np.asarray(measured['keys'],dtype=int)
                counts=np.asarray(measured['counts'],dtype=float)
                if (keys.ndim!=1 or counts.shape!=keys.shape or len(np.unique(keys))!=len(keys)
                        or not np.isfinite(counts).all() or (counts<0).any()):
                    raise ValueError('invalid layer-summed net usage')
            if np.any(keys<0) or np.any(keys>=len(usage)):raise ValueError('invalid observed resource edge')
            observed_foreground[keys]+=counts
        trial.edge_prices=.5*trial.edge_prices+.5*(usage/np.maximum(capacity,1.))**2
        if retained:
            trial.capacity=capacity.copy()
            trial.background=np.maximum(usage-observed_foreground,0.)
            trial.demand=trial.recompute_demand()
        trial.generation+=1
        record=dict(generation=trial.generation,retained=retained,
            source_sha256=observation['source_sha256'],placement_sha256=observation['placement_sha256'],
            objective_before=prior['objective'],objective_after=trial.metrics()['objective'],
            changed_net_weights=int(sum(trial.net_weights.get(n,1.)!=state.net_weights.get(n,1.) for n in trial.routes)),
            edge_prices_sha256=hashlib.sha256(trial.edge_prices.tobytes()).hexdigest(),
            background_sha256=hashlib.sha256(trial.background.tobytes()).hexdigest())
        self.state=trial
        self.observations.append(record)
        return record

    def reroute(self,x,y):
        """Rebuild all committed trees under the latest observation generation."""
        trial=self.state.fork()
        for net in self.net_ids:trial.remove(net)
        for net in self.net_ids:trial.replace(net,self.pin_positions(net,x,y))
        self.state=trial
        return trial.metrics()


def online_loop(feedback,x,y,*,oracle,is_legal,full_metrics,rounds=2,max_active=32,
                neighbors=8,hpwl_budget=.05,learn=True,baseline_observation=None,checkpoint=None):
    """Route, learn, propose, independently validate, and continue after vetoes."""
    if rounds<1:raise ValueError('at least one online round required')
    x,y=np.array(x,dtype=float,copy=True),np.array(y,dtype=float,copy=True)
    nm=feedback.nl.num_movable
    fixed=x[nm:].copy(),y[nm:].copy()
    def valid(a,b):
        return (np.isfinite(a).all() and np.isfinite(b).all()
            and np.array_equal(a[nm:],fixed[0]) and np.array_equal(b[nm:],fixed[1]) and is_legal(a,b))
    if not valid(x,y):raise ValueError('online loop needs a legal finite baseline')
    baseline=baseline_observation or oracle('baseline',x,y,feedback.state)
    original_actual=baseline;retained_actual=baseline
    original=feedback.state.metrics();original_hpwl=full_metrics(x,y)['hpwl']
    def prepare_observation(observation,evaluated,retained,a,b):
        # Validate on an isolated wrapper even when learning is disabled.
        # Publish only after observation validation and full rerouting succeed.
        working=copy(feedback)
        working.state=(evaluated if retained else feedback.state).fork()
        working.observations=list(feedback.observations)
        working.observe(observation,evaluated_state=evaluated,retained=retained)
        if learn:working.reroute(a,b)
        return working
    baseline_work=prepare_observation(baseline,feedback.state,True,x,y)
    if learn:
        feedback.state=baseline_work.state;feedback.observations=baseline_work.observations
    history=[];accepted=0
    for iteration in range(rounds):
        incumbent=feedback.state.metrics()
        record=dict(round=iteration,generation=feedback.state.generation,incumbent=incumbent,
            last_observation=feedback.observations[-1]['source_sha256'] if feedback.observations else None)
        proposal=feedback.propose(x,y,max_active=max_active,neighbors=neighbors)
        metric=proposal.state.metrics();legal=valid(proposal.x,proposal.y)
        hpwl=full_metrics(proposal.x,proposal.y)['hpwl'] if legal else float('inf')
        verdict=accept_joint_move(original,incumbent,metric,original_hpwl=original_hpwl,
            candidate_hpwl=hpwl,legal=legal,fixed_unchanged=legal,hpwl_budget=hpwl_budget)
        record.update(candidate=metric,proposal=proposal.diagnostics,surrogate=verdict,
                      router_accepted=False,hpwl=hpwl)
        observation=None
        if verdict['accepted']:
            label=f'round{iteration}_candidate'
            observation=oracle(label,proposal.x,proposal.y,proposal.state)
            # Actual IO improvement is required; equal IO with lower WL remains
            # a diagnostic improvement and cannot become an IO claim.
            router_ok=(observation['actual_io']<retained_actual['actual_io']
                and observation['actual_wirelength']<=original_actual['actual_wirelength']*(1+hpwl_budget))
            working=prepare_observation(observation,proposal.state,router_ok,
                proposal.x if router_ok else x,proposal.y if router_ok else y)
            record['router']={k:observation[k] for k in ('actual_io','actual_wirelength','source_sha256','placement_sha256')}
            record['router_accepted']=bool(router_ok)
            if router_ok:
                x,y=proposal.x.copy(),proposal.y.copy();feedback.state=proposal.state
                retained_actual=observation;accepted+=1
            if learn:
                feedback.state=working.state;feedback.observations=working.observations
        if checkpoint:checkpoint(iteration,x,y,feedback.state,record)
        history.append(record)
        # Do not terminate on a router veto: its learned error must reach the
        # next round. No proposal still produces explicit per-round evidence.
    return x,y,dict(rounds=history,observations=feedback.observations,cohort=feedback.cohort,
        accepted_rounds=accepted,learning_enabled=learn,
        original_actual_io=original_actual['actual_io'],final_actual_io=retained_actual['actual_io'],
        original_actual_wirelength=original_actual['actual_wirelength'],
        final_actual_wirelength=retained_actual['actual_wirelength'],final=feedback.state.metrics())
