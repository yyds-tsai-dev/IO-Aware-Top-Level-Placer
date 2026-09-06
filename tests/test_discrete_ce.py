import itertools
import numpy as np
import pytest
from ioplace.ops.discrete_ce import conditional_expectation, net_metrics


def fixture():
    return dict(probabilities=np.array([[.2,.3,.5],[0.,1.,0.],[.5,.5,0.]]),
        candidate_region=np.array([[0,1,1],[1,2,0],[2,0,1]]),area=np.array([1.,2.,3.]),
        region_area=np.array([10.,12.,14.]),target=np.array([2.,3.,5.]),inactive_load=np.array([.5,1.,2.]),
        unary=np.array([[0.,.02,.04],[.03,0.,.01],[0.,.01,.02]]),
        cell_start=np.array([0,2,4,6]),factor_id=np.array([0,1,0,2,1,2]),
        candidate_mask=np.array([[1,2,3],[1,3,2],[2,4,1],[2,4,1],[4,1,2],[4,3,2]],dtype=np.uint64),
        fixed_mask=np.array([0,4,0],dtype=np.uint64),passed_mask=np.array([7,7,7],dtype=np.uint64),
        io_weight=.1,ft_weight=.2,balance_weight=.3)


def deterministic_cost(data,choices):
    present=list(map(int,data["fixed_mask"]))
    loads=data["inactive_load"].copy()
    unary=0.
    for cell,candidate in enumerate(choices):
        for incidence in range(data["cell_start"][cell],data["cell_start"][cell+1]):
            present[data["factor_id"][incidence]] |= int(data["candidate_mask"][incidence,candidate])
        loads[data["candidate_region"][cell,candidate]]+=data["area"][cell]
        unary+=data["unary"][cell,candidate]
    io=sum(mask.bit_count()-1 for mask in present)
    ft=sum((int(passed)&~mask).bit_count() for passed,mask in zip(data["passed_mask"],present))
    balance=np.sum(((loads-data["target"])/data["region_area"])**2)
    return data["io_weight"]*io+data["ft_weight"]*ft+unary+data["balance_weight"]*balance


def enumerated_expectation(data,probabilities):
    total=0.
    for choices in itertools.product(range(probabilities.shape[1]),repeat=len(probabilities)):
        weight=np.prod([probabilities[i,c] for i,c in enumerate(choices)])
        total+=weight*deterministic_cost(data,choices)
    return total


def test_native_ce_matches_exhaustive_expectation_at_every_conditioning_step():
    data=fixture();result=conditional_expectation(**data)
    probability=data["probabilities"].copy()
    assert result["objective_trace"][0]==pytest.approx(enumerated_expectation(data,probability),abs=1e-12)
    for cell,choice in enumerate(result["choices"]):
        before=enumerated_expectation(data,probability)
        candidates=[]
        for c in range(probability.shape[1]):
            trial=probability.copy();trial[cell]=0;trial[cell,c]=1
            candidates.append(enumerated_expectation(data,trial))
        assert np.dot(probability[cell],candidates)==pytest.approx(before,abs=1e-12)
        assert candidates[choice]==pytest.approx(min(candidates),abs=1e-12)
        probability[cell]=0;probability[cell,choice]=1
        assert result["objective_trace"][cell+1]==pytest.approx(enumerated_expectation(data,probability),abs=1e-12)
    assert np.all(np.diff(result["objective_trace"])<=1e-12)
    assert result["weighted_identity_max_error"]<1e-12


def test_multiple_pins_on_one_cell_are_a_union_not_independent_trials():
    data=fixture()
    data["candidate_mask"][:]=3  # every candidate for these cell/net pairs occupies regions0 and1
    result=conditional_expectation(**data)
    assert result["objective_trace"][0]==pytest.approx(enumerated_expectation(data,data["probabilities"]),abs=1e-12)


def test_deterministic_probabilities_zero_factors_and_zero_ft_weight():
    data=fixture();data["probabilities"][:]=[1.,0.,0.];data["ft_weight"]=0.
    result=conditional_expectation(**data)
    assert result["objective_trace"][0]==pytest.approx(deterministic_cost(data,[0,0,0]),abs=1e-12)
    assert np.isfinite(result["objective_trace"]).all()


def test_duplicate_cell_net_incidence_rejected():
    data=fixture();data["factor_id"][1]=0
    with pytest.raises(ValueError,match="combine all same-cell pins"):
        conditional_expectation(**data)


@pytest.mark.parametrize("seed",[0,7,31])
def test_compiled_pin_metrics_match_reference_with_ties_offsets_and_boundaries(seed):
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.regions import make_grid_regions
    from ioplace.region_grid import RegionGrid
    from ioplace.evaluator_ref import evaluate,net_mst_edges,edge_regions_and_crossings
    from ioplace.netlist import pin_positions
    rng=np.random.default_rng(seed)
    nl=exact_synthetic(80,30,137,seed=seed,layout="uniform",die=(-3.7,1.3,55.2,100.8))
    nl.pin2node[:8]=0  # correlated pins on one cell, including coincident ones
    nl.pin_offset_x[:]=rng.integers(-2,3,len(nl.pin2node))*.5
    nl.pin_offset_y[:]=rng.integers(-2,3,len(nl.pin2node))*.5
    rg=RegionGrid(make_grid_regions((nl.xl,nl.yl,nl.xh,nl.yh),3,2,lattice=64))
    nl.node_x[:4]=[nl.xl,nl.xh,nl.xl+21*rg.cell_w,nl.xl+43*rg.cell_w]
    nl.node_y[:4]=[nl.yl,nl.yh,nl.yl+32*rg.cell_h,nl.yl+32*rg.cell_h]
    result=net_metrics(nl,rg);ref=evaluate(nl,nl.node_x,nl.node_y,rg)
    assert np.array_equal(result["per_net_crossings"],ref.per_net_crossings)
    assert np.array_equal(result["per_net_ft"],ref.per_net_ft)
    assert result["hpwl"]==pytest.approx(ref.hpwl,rel=1e-12)
    assert result["tree_wl"]==pytest.approx(ref.tree_wl,rel=1e-12)
    px,py=pin_positions(nl)
    for e in range(nl.num_nets):
        ids=nl.flat_net2pin[nl.flat_net2pin_start[e]:nl.flat_net2pin_start[e+1]]
        passed=set()
        for a,b in net_mst_edges(px[ids],py[ids]):
            passed.update(edge_regions_and_crossings(rg,px[ids[a]],py[ids[a]],px[ids[b]],py[ids[b]])[0])
        assert int(result["per_net_passed_mask"][e])==sum(1<<r for r in passed)


def test_sparse_geometry_factors_union_pin_offsets_and_preserve_fixed_tail():
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.regions import make_grid_regions
    from ioplace.evaluator_ref import evaluate
    from ioplace.ops.discrete_placement import prepare_discrete,decode_placement
    nl=exact_synthetic(40,20,73,seed=2,layout="uniform",die=(0.,0.,64.,64.))
    nl.num_movable=35;nl.num_terminals=5
    nl.node_x[35:]=[-10,80,90,100,110]
    nl.pin_offset_x[:]=np.arange(73)%3-1
    nl.pin_offset_y[:]=np.arange(73)%5-2
    regions=make_grid_regions((0,0,64,64),2,2,lattice=32)
    prepared=prepare_discrete(nl,regions,nl.node_x,nl.node_y,max_active=12)
    assert len(prepared["active"])==12
    assert np.all(prepared["active"]<35)
    assert np.allclose(prepared["probabilities"].sum(axis=1),1,atol=1e-12)
    for cell,node in enumerate(prepared["active"]):
        for incidence in range(prepared["cell_start"][cell],prepared["cell_start"][cell+1]):
            global_net=prepared["factor_net_ids"][prepared["factor_id"][incidence]]
            pins=np.flatnonzero((nl.pin2node==node)&(nl.pin2net==global_net))
            for c in range(prepared["probabilities"].shape[1]):
                rid=prepared["region_grid"].region_of_points(prepared["candidate_x"][cell,c]+nl.pin_offset_x[pins],
                                                            prepared["candidate_y"][cell,c]+nl.pin_offset_y[pins])
                assert int(prepared["candidate_mask"][incidence,c])==sum(1<<int(r) for r in set(rid))
    before=evaluate(nl,nl.node_x,nl.node_y,prepared["region_grid"])
    x,y,result=decode_placement(prepared,nl.node_x,nl.node_y,before.io_count,before.ft_count)
    assert np.array_equal(x[35:],nl.node_x[35:])
    assert np.array_equal(y[35:],nl.node_y[35:])
    assert np.all(np.diff(result["objective_trace"])<=result["numerical_tolerance"])


def test_single_region_has_no_internal_boundary_candidates():
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.regions import make_grid_regions
    from ioplace.ops.discrete_placement import prepare_discrete,decode_placement
    nl=exact_synthetic(20,5,20)
    prepared=prepare_discrete(nl,make_grid_regions((0,0,100000,100000),1,1),nl.node_x,nl.node_y)
    assert not len(prepared["active"])
    x,y,result=decode_placement(prepared,nl.node_x,nl.node_y,0,0)
    assert np.array_equal(x,nl.node_x) and np.array_equal(y,nl.node_y)


@pytest.mark.parametrize("decoded",[False,True])
def test_compiled_moves_and_unequal_area_swaps_match_full_recomputation(decoded):
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.regions import make_grid_regions
    from ioplace.evaluator_ref import evaluate
    from ioplace.ops.discrete_placement import prepare_discrete,decode_placement,refine_placement
    nl=exact_synthetic(50,30,161,seed=91,layout="uniform",die=(0.,0.,64.,64.))
    nl.num_movable=45;nl.num_terminals=5
    nl.node_size_x[:]=np.arange(50)%4+1
    nl.pin_offset_x[:]=np.arange(161)%3*.25
    nl.pin_offset_y[:]=np.arange(161)%5*.25
    prepared=prepare_discrete(nl,make_grid_regions((0,0,64,64),2,2,lattice=32),nl.node_x,nl.node_y,max_active=20)
    before=evaluate(nl,nl.node_x,nl.node_y,prepared["region_grid"])
    x,y=nl.node_x.copy(),nl.node_y.copy()
    if decoded:x,y,_=decode_placement(prepared,x,y,before.io_count,before.ft_count)
    # Explicit pairs exercise factors shared by both swap cells and unequal sizes.
    swaps=np.array([[i,i+1] for i in range(19)],dtype=np.int32)
    xx,yy,result=refine_placement(prepared,x,y,before.io_count,before.ft_count,swap_pairs=swaps)
    after=evaluate(prepared["compact_netlist"],xx,yy,prepared["region_grid"])
    assert after.io_count==result["metrics_after"]["io_count"]
    assert after.ft_count==result["metrics_after"]["ft_count"]
    assert after.hpwl==pytest.approx(result["metrics_after"]["hpwl"],rel=1e-12)
    assert np.array_equal(xx[45:],nl.node_x[45:]) and np.array_equal(yy[45:],nl.node_y[45:])
    assert np.all(np.diff(result["objective_trace"])<=1e-9)
    assert result["objective_trace"][-1]==pytest.approx(result["full_recompute_objective"],abs=1e-9)
