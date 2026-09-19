import numpy as np
from ioplace.netlist import Netlist
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def three_pin_case():
    # A legal same-size swap can move the only right-side pin left; no 2-pin nets.
    x=np.array([1.,2.,8.,3.]);y=np.array([1.,3.,2.,2.])
    nl=Netlist(x,y,np.full(4,.25),np.full(4,.25),4,0,0,
        np.zeros(3),np.zeros(3),np.array([0,1,2]),np.zeros(3,dtype=int),
        np.arange(3),np.array([0,3]),0.,0.,10.,10.)
    rg=RegionGrid(make_grid_regions((0,0,10,10),2,1,lattice=10))
    return nl,rg,x,y


def test_multi_pin_only_net_drives_actual_position_proposal_and_acceptance():
    # Bug caught: using degree-two cohort or retaining legacy MST as the gate.
    from ioplace.ops.joint_route_feedback import JointRouteFeedback, accept_joint_move
    nl,rg,x,y=three_pin_case()
    feedback=JointRouteFeedback(nl,rg,x,y,max_displacement_cells=8,congestion_weight=0.)
    before=feedback.state.metrics()
    candidate=feedback.propose(x,y,max_active=4,neighbors=4)
    assert candidate.state.metrics()['crossings']==0
    assert before['crossings']==1
    assert candidate.diagnostics['multi_pin_rebuilds']>0
    assert not np.array_equal(candidate.x,x)
    assert sorted(zip(candidate.x,candidate.y))==sorted(zip(x,y))
    verdict=accept_joint_move(before,before,candidate.state.metrics(),original_hpwl=20.,candidate_hpwl=10.,legal=True,fixed_unchanged=True)
    assert verdict['accepted']
    np.testing.assert_array_equal(feedback.state.demand,feedback.state.recompute_demand())
    assert feedback.state.metrics()['crossings']==1  # proposing never commits


def test_final_gate_uses_joint_cost_budgets_and_same_observation_generation():
    from ioplace.ops.joint_route_feedback import accept_joint_move
    base=dict(objective=20.,wirelength=100.,overflow=0.,generation=0)
    candidate=dict(base,objective=19.,wirelength=104.)
    args=dict(original_hpwl=100.,candidate_hpwl=104.,legal=True,fixed_unchanged=True)
    assert accept_joint_move(base,base,candidate,**args)['accepted']
    assert not accept_joint_move(base,base,dict(candidate,generation=1),**args)['accepted']
    assert not accept_joint_move(base,base,dict(candidate,overflow=1.),**args)['accepted']
    assert not accept_joint_move(base,base,dict(candidate,wirelength=106.),**args)['accepted']
    assert not accept_joint_move(base,base,candidate,**dict(args,candidate_hpwl=106.))['accepted']
    assert not accept_joint_move(base,base,candidate,**dict(args,fixed_unchanged=False))['accepted']


def test_fixed_nodes_never_swap_and_unsupported_incidents_are_explicit():
    from ioplace.ops.joint_route_feedback import JointRouteFeedback
    nl,rg,x,y=three_pin_case();nl.num_movable=3;nl.num_terminals=1
    feedback=JointRouteFeedback(nl,rg,x,y,max_displacement_cells=8)
    trial=feedback.propose(x,y,max_active=3,neighbors=3)
    assert trial.x[3]==x[3] and trial.y[3]==y[3]
    assert feedback.cohort['eligible_multi_pin_nets']==1
    assert feedback.cohort['eligible_two_pin_nets']==0


def test_proposal_anchor_uses_incident_steiner_structure_not_only_net_centroid():
    from ioplace.ops.joint_route_feedback import JointRouteFeedback
    nl,rg,x,y=three_pin_case();f=JointRouteFeedback(nl,rg,x,y,congestion_weight=0.)
    anchors=f.topology_anchors(2,x,y)
    assert any(np.allclose(anchor,[2.,2.]) for anchor in anchors)
    assert not all(np.allclose(anchor,[11/3,2.]) for anchor in anchors)


def test_paper_anchors_midpoints_steiner_and_colocated_multiplicity():
    from ioplace.ops.joint_route_feedback import rsmt_anchors
    pins=np.array([[0.,0.],[4.,0.]])
    anchors=rsmt_anchors(pins,np.array([pins]))
    np.testing.assert_array_equal(anchors[0],[[2.,0.]])
    np.testing.assert_array_equal(anchors[1],[[2.,0.]])
    # Fig. 5(c): overlapping cell at degree two gets two self anchors.
    pins=np.array([[0.,0.],[0.,0.],[-4.,0.],[2.,2.]])
    edges=np.array([[[0,0],[-4,0]],[[0,0],[0,2]],[[0,2],[2,2]]],float)
    anchors=rsmt_anchors(pins,edges)
    np.testing.assert_array_equal(anchors[0],[[-2,0],[0,2],[0,0],[0,0]])
    # Fig. 5(d): pin on degree-three Steiner junction also keeps a self anchor.
    pins=np.array([[0.,0.],[-4.,0.],[4.,0.],[0.,4.]])
    edges=np.array([[[0,0],[-4,0]],[[0,0],[4,0]],[[0,0],[0,4]]],float)
    anchors=rsmt_anchors(pins,edges)
    assert len(anchors[0])==4
    np.testing.assert_array_equal(anchors[0][-1],[0,0])
