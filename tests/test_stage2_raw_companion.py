import copy
import numpy as np
import pytest
from scripts.stage2_raw_companion import project_raw

def fixture():
    n=6
    ev={"per_net_steiner":[1,2,3,4,5,6],"per_net_crossings":[1]*n,
        "per_net_lambda":[2]*n,"net_degrees":[2,1,257,2,2,2],
        "metadata":{"max_degree":256}}
    sample={"delta":2,"io_rg":10,"io_mst":12,"route_cross_raw":9,
      "per_net_route_cross_dw":np.array([1,2,3,1,0,1]),
      "evaluator":ev,"unmatched_net_indices":[],
      "per_net_calibration_eligible":[True,True,True,False,True,True]}
    primary={"delta":2,"per_net_calibration_eligible":sample["per_net_calibration_eligible"],
      "per_net_has_routed_wire":[True]*n,"per_net_route_ft":[1]*n,"per_net_route_wl":[1]*n,
      "route_pair_demand":{"0,1":1},"unrouted_signal_gate_pass":True,
      "evaluator_sha256":"e","segments_sha256":"s","routed_def_sha256":"r",
      "per_net_route_cross_raw":[4,20,30,10,0,5]}
    raw=dict(primary,delta=0,per_net_route_cross_dw=[4,20,30,10,0,5],per_net_lambda_route=[3]*n)
    return sample,primary,raw

def test_project_raw_uses_formal_mask_and_preserves_input():
    s,p,r=fixture(); before=copy.deepcopy(s)
    out=project_raw(s,p,r)
    assert out["route_cross_dw"]==9  # degree1/large/ineligible excluded
    assert out["route_minus_io_rg"] == -1 and out["io_mst_minus_route"] == 3
    np.testing.assert_array_equal(out["per_net_route_cross_dw"], [4,20,30,10,0,5])
    np.testing.assert_array_equal(out["per_net_lambda_route"], [3]*6)
    assert out["route_target"]=="route_cross_raw" and out["delta"]==0
    assert s["delta"] == before["delta"] and np.array_equal(s["per_net_route_cross_dw"], before["per_net_route_cross_dw"])

@pytest.mark.parametrize("key", ["per_net_route_ft","route_pair_demand","per_net_route_wl","evaluator_sha256","segments_sha256","routed_def_sha256","per_net_calibration_eligible","per_net_route_cross_raw"])
def test_delta_invariant_corruption_rejected(key):
    s,p,r=fixture(); r=copy.deepcopy(r); r[key]=([] if key.endswith("sha256") else ({} if key=="route_pair_demand" else [False]*6))
    with pytest.raises(ValueError): project_raw(s,p,r)

@pytest.mark.parametrize("which", ["primary_delta","raw_delta","raw_values","raw_shape"])
def test_delta_and_raw_crossing_contract_rejected(which):
    s,p,r=fixture()
    if which=="primary_delta": p["delta"]=1
    elif which=="raw_delta": r["delta"]=2
    elif which=="raw_values": r["per_net_route_cross_raw"][0]=99
    else:
        r["per_net_route_cross_dw"]=[1]
        r["per_net_route_cross_raw"]=[1]
        p["per_net_route_cross_raw"]=[1]
    with pytest.raises(ValueError): project_raw(s,p,r)
