import json, hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from ioplace.diagnostics import stage2_calibration as sc
from scripts.stage2_s8_extract_crossings import routing_population, validate_route_evidence

def _sample(n=9):
    ev={"per_net_steiner":list(range(n)),"per_net_crossings":[1]*n,
        "per_net_lambda":[2]*n,"net_degrees":[2]*n,
        "metadata":{"max_degree":256}}
    return {"sample_id":"toy","evaluator":ev,"per_net_route_cross_dw":[1]*n,
            "unmatched_net_indices":[8],"per_net_calibration_eligible":[True]*n,
            "unrouted_signal_gate_pass":True}

def test_paired_arrays_excludes_unrouted_and_reports_formal_population():
    route, models, degree, matched, formal = sc._paired_arrays(_sample())
    assert len(route)==9 and formal.sum()==8 and not formal[8]
    assert sc.table2_per_net_correlation([_sample()])["rows"][0]["n"] == 8

def test_zero_denominator_and_rank_deficient_are_graceful():
    t={"pooled":{"R_lam_minus_1":None,"R_io_rg":None,"R_io_mst":None}}
    assert sc.judge_c1(t)["verdict"] == "not_evaluable"
    c2,c3=sc.judge_c2_c3([],t,{"identifiable":False,"coefficients":{"alpha":{"value":0},"beta":{"value":0},"gamma":{"value":0}}})
    assert c2["verdict"] == "not_evaluable" and c3["verdict"] == "not_evaluable"
    assert sc._corr([1,1],[2,3])["spearman"] is None

def test_route_ft_unknown_eligibility_does_not_claim_complete():
    x=_sample(); x["evaluator"]["per_net_lambda"]=[1]*9
    assert sc._paired_arrays(x)[4].sum()==8

def test_delta_scan_rejects_swapped_artifact(tmp_path):
    case=str(tmp_path); p=tmp_path/'row.json'; p.write_text(json.dumps({"delta":0,"total_route_cross_dw":1,"total_route_cross_raw":1,"routed_def_sha256":"r","segments_sha256":"s","evaluator_sha256":"e"}))
    primary={"routed_def_sha256":"r","segments_sha256":"s","evaluator_sha256":"e"}
    scan={**primary,"rows":[{"delta":i,"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"route_cross_dw":1,"route_cross_raw":1} for i in [0,1,2,4]]}
    (tmp_path/'delta_scan_k16.json').write_text(json.dumps(scan))
    with pytest.raises(ValueError): sc.load_delta_scan(case,16,primary)

def test_routing_population_uses_positive_wire_and_excludes_special():
    seg=SimpleNamespace(net_names=np.array(["a","b","VDD"]),seg_net_id=np.array([0,1]),
                        wire_mask=lambda:np.array([True,False]),segment_length=lambda:np.array([2.,3.]))
    ev={"metadata":{"max_degree":256,"provenance":{"pin_geometry_metadata":{"excluded_special_net_names":["VDD"]}}},"net_degrees":np.array([2,2,2])}
    out=routing_population(seg,["a","b","VDD"],ev)
    assert out["per_net_has_routed_wire"]==[True,False,False]
    assert out["unrouted_net_indices"]==[1]

def test_coord_only_mismatch_rejected(tmp_path):
    def f(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    routed=tmp_path/'r.def'; seg=tmp_path/'s.npz'; coord=tmp_path/'c.json'; netmap=tmp_path/'n.json'; prov=tmp_path/'p.json'
    for p in (routed,seg,coord,netmap): p.write_bytes(b'x')
    ev={"metadata":{"provenance":{"placement_stage":"router_def","router_def_sha256":f(routed),"coord_sha256":"bad","netmap_sha256":f(netmap)}}}
    prov.write_text(json.dumps({"routed_def_sha256":f(routed),"segments_sha256":f(seg),"verified":True}))
    with pytest.raises(ValueError): validate_route_evidence(ev,str(routed),str(seg),str(prov),str(coord),str(netmap))


def test_c4_compares_common_nets_when_both_arms_pass_unrouted_gate():
    samples = []
    for arm, missing in (("flat_k16", 98), ("ours_k16", 99)):
        sample = _sample(100)
        sample.update(design="design", arm=arm, k_placement=16,
                      sample_id="design__"+arm, unmatched_net_indices=[missing],
                      lam_minus_1=0, route_cross_dw=0)
        names = np.array([f"n{i}" for i in range(100)])
        sample["evaluator"]["net_names"] = names
        gp = np.full(100, 3 if arm.startswith("flat") else 2)
        route = np.ones(100, dtype=np.int64)
        if arm.startswith("ours"):
            route[:10] = 0
            route[98] = 1000  # This net is unavailable in the flat arm.
        sample["pre_route_evaluator"] = dict(net_names=names, per_net_lambda=gp)
        sample["per_net_route_cross_dw"] = route
        samples.append(sample)
    row = sc.judge_c4(samples)["formal_rows"][0]
    assert row["common_eligible_nets"] == 98
    assert row["pre_route_delta"] == -98 and row["post_route_delta"] == -10
    assert not row["sign_flip"]
    samples[1]["unrouted_signal_gate_pass"] = False
    assert sc.judge_c4(samples)["formal_verdict"] == "not_evaluable"


def test_empty_eligible_population_is_not_evaluable_in_bucket_and_boundary_tables():
    sample = _sample()
    sample["unrouted_signal_gate_pass"] = False
    assert sc.table3_degree_bucket([sample])["not_evaluable"]
    sample["boundary_evidence"] = dict(not_evaluable=True, reason="unrouted gate")
    assert sc.table7_boundary_pair_demand([sample])["not_evaluable"]
