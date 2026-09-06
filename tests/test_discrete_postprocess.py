import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ioplace.ops.discrete_postprocess import acceptance,run_discrete_postprocess


def test_full_hpwl_and_lg_only_control_both_gate_acceptance():
    baseline=dict(io_count=100,ft_count=50,hpwl=1000.)
    proposal=dict(io_count=80,ft_count=30,hpwl=1100.)
    assert "full_hpwl_guard" in acceptance(baseline,proposal,legal=True,fixed_unchanged=True)["reasons"]
    proposal["hpwl"]=1000.
    control=dict(io_count=70,ft_count=25,hpwl=1000.)
    assert not acceptance(baseline,proposal,legal=True,fixed_unchanged=True,control=control)["accepted"]
    assert acceptance(baseline,proposal,legal=True,fixed_unchanged=True)["accepted"]
    assert not acceptance(baseline,proposal,legal=True,fixed_unchanged=True,control_valid=False)["accepted"]


def test_failed_fixed_node_check_restores_incumbent_and_recomputes_assignments():
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.regions import make_grid_regions
    from ioplace.region_grid import RegionGrid
    from ioplace.evaluator_ref import evaluate
    nl=exact_synthetic(12,6,25,seed=12,layout="uniform",die=(0.,0.,64.,64.))
    nl.num_movable=10;nl.num_terminals=2
    regions=make_grid_regions((0,0,64,64),2,2,lattice=32);rg=RegionGrid(regions)
    tensor=torch.tensor(np.r_[nl.node_x,nl.node_y],dtype=torch.float32,requires_grad=True)
    original=tensor.detach().clone()
    placer=SimpleNamespace(pos=[tensor],op_collections=SimpleNamespace(legality_check_op=lambda pos:True))
    class DB:
        num_nodes=12
        def apply(self,params,x,y):self.applied=(x.copy(),y.copy())
    db=DB()
    context=SimpleNamespace(evaluate=lambda x,y:evaluate(nl,x,y,rg))
    def corrupt_fixed(pos):
        pos=pos.clone();pos[10]+=2
        return pos
    report,arrays=run_discrete_postprocess(nl,regions,placer,db,None,corrupt_fixed,context,
                                         mode="ce_refine",max_active=6)
    assert not report["accepted"] and "fixed_nodes_changed" in report["reasons"]
    assert torch.equal(placer.pos[0],original)
    assert np.array_equal(arrays["final_region_assignment"],rg.region_of_points(original[:12],original[12:]))
    assert np.array_equal(db.applied[0],original[:10].numpy())


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_real_driver_discrete_postprocess_exports_legal_final_geometry(tmp_path):
    from ioplace.drivers.run_placement_io import run_io
    from ioplace.netlist import load_netlist
    from ioplace.drivers.run_placement import get_regions_for
    from ioplace.region_grid import RegionGrid
    dp=Path(os.environ["DREAMPLACE_ROOT"])
    config=json.loads((dp/"install/test/simple.json").read_text())
    config.update(num_threads=4,plot_flag=0,num_bins_x=16,num_bins_y=16,
        global_place_stages=[dict(num_bins_x=16,num_bins_y=16,iteration=40,learning_rate=.01,
                                 wirelength="weighted_average",optimizer="nesterov")])
    path=tmp_path/"discrete.json";path.write_text(json.dumps(config))
    out=tmp_path/"result.json"
    result=run_io(str(path),4,"grid",0,str(out),rho_max=0,every=5,no_diag=True,
                  discrete_mode="ce_refine",discrete_max_active=64)
    assert result["legalization_status"]=="success"
    assert result["discrete_result"]["metric_scope"]=="full netlist after legalization"
    assert result["n_pins_raw"]>=result["n_pins_canonical"]
    coordinates=np.load(str(out)+".npz")
    sidecar=np.load(str(out)+".discrete.npz")
    nl,_,_=load_netlist(str(path))
    rg=RegionGrid(get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),4,"grid",0))
    expected=rg.region_of_points(coordinates["node_x"],coordinates["node_y"])
    assert np.array_equal(sidecar["final_region_assignment"],expected)
    final=result["discrete_result"]["final"]
    assert result["io_count"]==final["io_count"] and result["ft_count"]==final["ft_count"]
