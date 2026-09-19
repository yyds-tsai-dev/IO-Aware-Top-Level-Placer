import numpy as np
from tests.test_joint_route_feedback import three_pin_case


def observation(feedback, io=4):
    s=feedback.state
    return dict(net_io={0:io},net_keys={0:s.routes[0].keys.tolist()},
        usage=s.demand.copy(),capacity=s.capacity.copy(),actual_io=io,
        actual_wirelength=20.,placement_sha256='fixture',source_sha256='fixture-router')


def test_measured_router_error_changes_next_multi_pin_update_costs():
    # Bug caught: storing router metrics without using them in the optimizer.
    from ioplace.ops.joint_route_feedback import JointRouteFeedback
    nl,rg,x,y=three_pin_case();f=JointRouteFeedback(nl,rg,x,y,congestion_weight=0.)
    before=f.state.metrics()['objective'];priority=f.priorities().copy()
    old=f.state.fork()
    record=f.observe(observation(f,5),evaluated_state=old,retained=True)
    assert f.state.generation==1
    assert f.state.net_weights[0]>1
    assert f.state.metrics()['objective']>before
    assert np.any(f.priorities()!=priority)
    assert record['source_sha256']=='fixture-router'
    trial=f.propose(x,y,max_active=4,neighbors=4)
    assert trial.diagnostics['generation']==1


def test_rejected_router_observation_reweights_next_round_without_committing_positions():
    # Bug caught: terminating on first veto or learning only accepted router runs.
    from ioplace.ops.joint_route_feedback import JointRouteFeedback, online_loop
    nl,rg,x,y=three_pin_case();f=JointRouteFeedback(nl,rg,x,y,congestion_weight=0.,max_displacement_cells=8.)
    def oracle(label,a,b,state):
        io=1 if label=='baseline' else 5
        return dict(net_io={0:io},net_keys={0:state.routes[0].keys.tolist()},
            usage=state.demand.copy(),capacity=state.capacity.copy(),actual_io=io,
            actual_wirelength=20.,placement_sha256=label,source_sha256='measured-'+label)
    xx,yy,report=online_loop(f,x,y,oracle=oracle,is_legal=lambda a,b:True,
        full_metrics=lambda a,b:dict(hpwl=20.),rounds=2,max_active=4,neighbors=4)
    np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(yy,y)
    assert len(report['rounds'])==2
    assert not report['rounds'][0]['router_accepted']
    assert report['rounds'][1]['generation']>report['rounds'][0]['generation']
    assert report['rounds'][1]['incumbent']['objective']!=report['rounds'][0]['incumbent']['objective']
    assert report['rounds'][1]['last_observation']=='measured-round0_candidate'


def test_router_usage_background_does_not_double_count_modeled_shared_net():
    from ioplace.ops.joint_route_feedback import JointRouteFeedback
    nl,rg,x,y=three_pin_case();f=JointRouteFeedback(nl,rg,x,y)
    o=observation(f);o['usage']=o['usage']+2
    f.observe(o,evaluated_state=f.state.fork(),retained=True)
    np.testing.assert_array_equal(f.state.background,np.full(f.state.grid.edge_count,2.))
    np.testing.assert_array_equal(f.state.demand,f.state.recompute_demand())


def test_openroad_adapter_reads_real_shared_routes_and_capacities(tmp_path):
    # Real external boundary: catches unit, direction, capacity, and net-map errors.
    import os
    import json
    import pytest
    from pathlib import Path
    from ioplace.paths import REPO_ROOT
    from ioplace.route_eval.online_openroad import run_openroad, load_observation
    binary=os.environ.get('OPENROAD_BIN')
    if not binary or not Path(binary).exists():pytest.skip('set OPENROAD_BIN for live GRT test')
    case=Path(REPO_ROOT)/'results/route_feedback_20260914/gcd_swap32_seed1000/baseline'
    if not (case/'out.def').exists():pytest.skip('GCD legal checkpoint unavailable')
    lef=Path(REPO_ROOT)/'third_party/OpenROAD/src/grt/test/Nangate45/Nangate45.lef'
    receipt=run_openroad(case/'out.def',[lef],tmp_path/'grt',binary)
    result=load_observation(tmp_path/'grt',case/'coord.json',case/'regions.json',case/'netmap.json')
    assert receipt['returncode']==0
    assert result['actual_io']==414
    assert len(result['net_io'])==563
    assert np.any(result['capacity']>0)
    assert np.any(result['usage']>0)
    assert result['grid'].edge_count==len(result['capacity'])==len(result['usage'])
    assert result['source_sha256']


def test_layer_summed_foreground_leaves_no_phantom_background():
    from ioplace.ops.joint_route_feedback import JointRouteFeedback
    from ioplace.route_eval.online_openroad import layer_resource_usage
    nl,rg,x,y=three_pin_case();f=JointRouteFeedback(nl,rg,x,y)
    route=f.state.routes[0]
    measured=layer_resource_usage(f.state.grid,{'M2':route.segments,'M4':route.segments})
    o=observation(f);o['net_usage']={0:measured};o['usage']=2*f.state.demand+3
    f.observe(o,evaluated_state=f.state.fork(),retained=True)
    f.state.remove(0)
    np.testing.assert_array_equal(f.state.demand,np.full(f.state.grid.edge_count,3.))


def test_invalid_improving_observation_never_commits_candidate():
    import pytest
    from ioplace.ops.joint_route_feedback import JointRouteFeedback,online_loop
    nl,rg,x,y=three_pin_case();f=JointRouteFeedback(nl,rg,x,y,congestion_weight=0.)
    initial=f.state;weights=initial.net_weights.copy();history=list(f.observations)
    def oracle(label,a,b,state):
        o=observation(f,1)
        if label!='baseline':o.update(net_io={},actual_io=0)
        return o
    with pytest.raises(ValueError,match='omitted supported'):
        online_loop(f,x,y,oracle=oracle,is_legal=lambda a,b:True,
                    full_metrics=lambda a,b:dict(hpwl=20.),rounds=1,learn=False,max_active=4,neighbors=4)
    assert f.state is initial
    assert f.state.net_weights==weights and f.observations==history
    assert f.state.generation==0
