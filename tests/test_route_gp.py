import numpy as np
import pytest
import torch
from ioplace.netlist import Netlist
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from ioplace.route_eval.joint import ResourceGrid


def fixture():
    points=np.array([[4.8,2.3],[8.2,3.1],[1.3,7.2]])
    n=len(points)
    nl=Netlist(points[:,0],points[:,1],np.ones(n)*.1,np.ones(n)*.1,n,0,0,
        np.zeros(2),np.zeros(2),np.array([0,1]),np.zeros(2,dtype=int),
        np.array([0,1]),np.array([0,2]),0,0,10,10)
    rg=RegionGrid(make_grid_regions((0,0,10,10),2,1,lattice=10))
    grid=ResourceGrid(np.arange(11),np.arange(11))
    pos=torch.tensor(np.r_[points[:,0],points[:,1]],dtype=torch.float64,requires_grad=True)
    return nl,rg,grid,pos


def test_route_io_cost_reaches_two_pin_gp_gradient_and_decreases_on_descent():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,rg,grid,pos=fixture()
    op=FrozenJointRouteCost(nl,rg,grid,np.full(grid.edge_count,10.),num_nodes=3,hot_nets=0)
    meta=op.rebuild(pos)
    cost=op.components(pos,tau=1.,gamma=.1)
    g,=torch.autograd.grad(cost['io'],pos)
    assert g[0]<0  # left endpoint should move right across region boundary
    assert g[2]==g[5]==0  # disconnected cell
    moved=pos.detach()-.01*g
    assert op.components(moved,tau=1.,gamma=.1)['io']<cost['io']
    hard=op.components(pos,tau=1.,gamma=.1,hard=True)
    assert hard['io']==1
    assert float(hard['wirelength'].detach())==pytest.approx(4.2)
    assert hard['demand'][118]==1  # vertical leg crosses y=3 in x-bin 8
    assert meta['eligible_two_pin_nets']==1


def test_frozen_route_full_gradient_and_background_coupling():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,rg,grid,pos=fixture()
    op=FrozenJointRouteCost(nl,rg,grid,np.full(grid.edge_count,1.2),num_nodes=3,hot_nets=0)
    op.rebuild(pos)
    assert torch.autograd.gradcheck(lambda p:op(p,tau=.7,gamma=.15),(pos,),eps=1e-6,atol=1e-4,rtol=1e-3)
    a=op.components(pos,tau=.7,gamma=.15)['congestion']
    ga,=torch.autograd.grad(a,pos)
    pressured=FrozenJointRouteCost(nl,rg,grid,np.full(grid.edge_count,1.2),num_nodes=3,
        hot_nets=0,background=np.full(grid.edge_count,.9))
    pressured.rebuild(pos)
    gb,=torch.autograd.grad(pressured.components(pos,tau=.7,gamma=.15)['congestion'],pos)
    assert gb.abs().sum()>ga.abs().sum()


def test_rebuild_uses_whole_net_detour_budget_and_records_joint_demand():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,rg,grid,pos=fixture()
    op=FrozenJointRouteCost(nl,rg,grid,np.full(grid.edge_count,10.),num_nodes=3,hot_nets=1)
    meta=op.rebuild(pos)
    assert meta['generation']==1
    assert meta['detour_net_trials']>0
    assert meta['hard_wirelength_after']<=meta['hard_wirelength_before']*1.05+1e-8
    assert np.isfinite(meta['hard_objective_after'])


def test_router_observation_is_applied_only_at_next_rebuild():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,rg,grid,pos=fixture()
    op=FrozenJointRouteCost(nl,rg,grid,np.full(grid.edge_count,10.),num_nodes=3,hot_nets=0)
    op.rebuild(pos)
    before=op.components(pos,tau=1.,gamma=.1,hard=True)
    demand=before['demand'].detach().numpy();keys=np.flatnonzero(demand)
    obs=dict(grid=grid,capacity=np.full(grid.edge_count,10.),usage=demand+.9,
        net_io={0:5},net_usage={0:dict(keys=keys.tolist(),counts=[1]*len(keys))},
        actual_io=5,actual_wirelength=4.2,source_sha256='fixture',placement_sha256='placed-fixture')
    op.assimilate(obs,pos.detach())
    assert op.observation_version==1 and op.published_observation_version==0
    assert op.components(pos,tau=1.,gamma=.1,hard=True)['io']==before['io']
    op.rebuild(pos)
    assert op.published_observation_version==1
    after=op.components(pos,tau=1.,gamma=.1,hard=True)
    assert after['io']>before['io']
    np.testing.assert_allclose(op.background.cpu(),.9)


def test_batched_paper_term_matches_scalar_raw_flute_gradient():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    from ioplace.ops.steiner_wirelength import FrozenSteinerWirelength
    from tests.test_steiner_wirelength import _netlist,_pos
    points=np.array([(2.,8.),(0.,5.),(4.,6.),(9.,4.),(2.,2.),(6.,0.)])
    nl=_netlist(points);nl.net_degrees=np.array([6])
    pos=_pos(points,requires_grad=True)
    rg=RegionGrid(make_grid_regions((0,0,10,10),2,1,lattice=10))
    grid=ResourceGrid(np.arange(11),np.arange(11))
    routed=FrozenJointRouteCost(nl,rg,grid,np.full(grid.edge_count,10.),num_nodes=6,hot_nets=0)
    scalar=FrozenSteinerWirelength(nl,6)
    routed.rebuild(pos);scalar.rebuild(pos)
    assert routed.paper(pos,1.).item()==pytest.approx(3.0463766238230594)
    ga,=torch.autograd.grad(routed.paper(pos,1.),pos)
    gb,=torch.autograd.grad(scalar(pos,1.),pos)
    torch.testing.assert_close(ga,gb)


def test_float32_hot_candidates_share_exact_resource_and_io_boundaries():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,_,_,pos=fixture()
    rg=RegionGrid(make_grid_regions((0,0,1.4,1.4),2,1,lattice=2))
    grid=ResourceGrid([0,.7,1.4],[0,.7,1.4])
    pos=torch.tensor([.2,.7,.1,.2,.2,1.],dtype=torch.float32)
    op=FrozenJointRouteCost(nl,rg,grid,np.ones(grid.edge_count),num_nodes=3,hot_nets=0)
    op.rebuild(pos)
    union=op.intervals(pos)
    a,b=op._ends(pos)
    record=op._hard_record(np.stack([op._line(x,y,0,0.) for x,y in zip(a.numpy(),b.numpy())]))
    hard=op.components(pos,tau=.1,gamma=.1,hard=True)
    np.testing.assert_array_equal(record.keys,np.flatnonzero(hard['demand'].numpy()))
    assert record.crossings==hard['io_raw'].item()==1
    op.hot_nets=1
    meta=op.rebuild(pos)
    assert meta['hot_demand_consistent']


@pytest.mark.parametrize('field,value',[('actual_io',float('nan')),('actual_wirelength',-1),('actual_io',2)])
def test_bad_router_totals_cannot_publish_partial_state(field,value):
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,rg,grid,pos=fixture()
    op=FrozenJointRouteCost(nl,rg,grid,np.ones(grid.edge_count),num_nodes=3,hot_nets=0)
    op.rebuild(pos)
    obs=dict(grid=grid,capacity=np.ones(grid.edge_count),usage=np.ones(grid.edge_count),
        net_io={0:1},net_usage={0:dict(keys=[],counts=[])},actual_io=1,actual_wirelength=4.2,
        source_sha256='x',placement_sha256='y')
    obs[field]=value
    with pytest.raises(ValueError):op.assimilate(obs,pos.detach())
    assert op.observation_version==0 and op.generation==1
    np.testing.assert_array_equal(op.io_calibration_np,[1])


def test_two_hot_nets_preserve_shared_resource_demand_at_float32_boundary():
    from ioplace.ops.route_gp import FrozenJointRouteCost
    points=np.array([[.2,.2],[.7,.2],[.3,.2],[.7,.2]])
    nl=Netlist(points[:,0],points[:,1],np.full(4,.01),np.full(4,.01),4,0,0,
        np.zeros(4),np.zeros(4),np.arange(4),np.repeat([0,1],2),np.arange(4),np.array([0,2,4]),0,0,1.4,1.4)
    rg=RegionGrid(make_grid_regions((0,0,1.4,1.4),2,1,lattice=2))
    grid=ResourceGrid([0,.7,1.4],[0,.7,1.4])
    pos=torch.tensor(np.r_[points[:,0],points[:,1]],dtype=torch.float32)
    op=FrozenJointRouteCost(nl,rg,grid,np.ones(grid.edge_count),num_nodes=4,hot_nets=2)
    meta=op.rebuild(pos)
    assert meta['detour_net_trials']==2 and meta['hot_demand_consistent']
    hard=op.components(pos,tau=.1,gamma=.1,hard=True)
    assert hard['demand'][0]==2 and hard['io_raw']==2
    np.testing.assert_array_equal(op.hot_final_demand,hard['demand'].numpy())
