import numpy as np
from ioplace.netlist import Netlist
from ioplace.ops.route_feedback import RouteFeedback, closed_loop, accept_move
from ioplace.evaluator_ref import evaluate
from tests.test_budgeted_routing import dense_regions
from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid


def fixture():
    x, y = np.array([5.,95.]), np.array([50.,50.])
    nl = Netlist(x,y,np.ones(2),np.ones(2),2,0,0,np.zeros(2),np.zeros(2),
        np.arange(2),np.zeros(2,dtype=int),np.arange(2),np.array([0,2]),0,0,100,100)
    return nl,x,y


def run(rg, legalize=lambda x,y:(x,y)):
    nl,x,y=fixture()
    feedback=RouteFeedback(nl,rg,x,y)
    def metrics(x,y):
        r=evaluate(nl,x,y,rg)
        return dict(io_count=r.io_count,ft_count=r.ft_count,hpwl=r.hpwl)
    return closed_loop(feedback,x,y,legalize=legalize,is_legal=lambda x,y:True,full_metrics=metrics)


def test_closed_loop_moves_pins_and_improves_fresh_placement_evaluation():
    x,y,report=run(dense_regions())
    assert report['accepted_rounds']==1
    assert report['original']['io_count']==11 and report['final']['io_count']==0
    assert report['final']['opportunity_wirelength']<report['original']['opportunity_wirelength']
    assert (y<49).all()
    assert report['actual_router_io'] is None


def test_uniform_grid_and_legalizer_erased_signal_are_noops():
    for rg,legalizer in [(RegionGrid(make_grid_regions((0,0,100,100),2,2,lattice=100)),lambda x,y:(x,y)),
                         (dense_regions(),lambda x,y:(x,np.full(2,50.)))]:
        x,y,r=run(rg,legalizer)
        assert r['accepted_rounds']==0
        np.testing.assert_array_equal(y,[50,50])


def test_fixed_nodes_and_aggregated_conflicting_pin_forces():
    nl,x,y=fixture();nl.num_movable=1;nl.num_terminals=1
    fb=RouteFeedback(nl,dense_regions(),x,y)
    _,d=fb.evaluate(x,y,signal=True)
    assert d.shape==(1,2) and d[0,1]<0


def test_original_budget_and_control_prevent_false_acceptance():
    original=dict(hpwl=100.,io_count=20,opportunity_io=10,opportunity_wirelength=100.)
    incumbent=dict(hpwl=104.,io_count=19,opportunity_io=9,opportunity_wirelength=104.)
    proposal=dict(hpwl=108.,io_count=18,opportunity_io=8,opportunity_wirelength=108.)
    assert not accept_move(original,incumbent,proposal,incumbent)['accepted']
    proposal.update(hpwl=100.,opportunity_wirelength=100.)
    assert accept_move(original,incumbent,proposal,incumbent)['accepted']
    assert not accept_move(original,incumbent,proposal,proposal)['accepted']


def test_cost_difference_swaps_preserve_rectangles_and_recompute_incident_cost():
    from ioplace.ops.route_feedback import cost_delta_swaps
    nl,x,y=fixture()
    nl.node_x=x=np.array([5.,95.,5.,95.]);nl.node_y=y=np.array([50.,50.,48.,48.])
    nl.node_size_x=nl.node_size_y=np.ones(4);nl.num_movable=4
    fb=RouteFeedback(nl,dense_regions(),x,y)
    _,direction=fb.evaluate(x,y,signal=True)
    xx,yy,detail=cost_delta_swaps(fb,x,y,direction,1.)
    assert len(detail['accepted_swaps'])>=1
    assert sorted(zip(xx,yy))==sorted(zip(x,y))
    assert evaluate(nl,xx,yy,fb.rg).io_count < evaluate(nl,x,y,fb.rg).io_count
    assert fb.evaluate(xx,yy)['opportunity_io']<=fb.evaluate(x,y)['opportunity_io']


def test_fixed_node_corruption_and_illegal_proposals_are_rejected():
    nl,x,y=fixture();nl.num_movable=1;nl.num_terminals=1
    fb=RouteFeedback(nl,dense_regions(),x,y)
    def corrupt(a,b):b[-1]+=1;return a,b
    xx,yy,r=closed_loop(fb,x,y,legalize=corrupt,is_legal=lambda a,b:True,
        full_metrics=lambda a,b:dict(io_count=11,hpwl=90))
    assert r['accepted_rounds']==0
    np.testing.assert_array_equal(yy,y)
