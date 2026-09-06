"""Exercise complete persisted calibration chains, including rejection gates."""
import json
from pathlib import Path

import numpy as np
import pytest

from ioplace.diagnostics import stage2_calibration as sc
from ioplace.export.evaluation import array_digest


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def _case(root, design, arm, coefficients=(2, 3, 1), rank_deficient=False):
    case = root / f"{design}__{arm}"
    run = case / "or_run"
    run.mkdir(parents=True)
    k = int(arm.rsplit("k", 1)[1])
    l = np.r_[np.tile([1, 1, 2, 2, 3, 3], 9), [1000, 0, 1000, 1000, 1, 0]]
    f = np.r_[np.tile([0, 1, 0, 2, 1, 3], 9), [1000, 0, 1000, 1000, 1, 0]]
    e = np.r_[np.tile([0, 0, 1, 1, 2, 1], 9), [1000, 0, 1000, 1000, 1, 0]]
    if rank_deficient:
        e[:] = 0
    n = len(l)
    names = np.array([f"n{i}" for i in range(n)])
    degree = np.r_[np.full(54, 2), [257, 1, 2, 2, 2, 0]]
    formal = np.ones(n, dtype=bool)
    formal[[54, 55, 56, 57, 59]] = False
    route = coefficients[0]*l + coefficients[1]*f + coefficients[2]*e
    # Arbitrarily bad excluded rows must not influence fitted coefficients.
    route[[54, 56, 57]] = 999999
    route_ft = route-l
    route_ft[[57, 58]] = -1
    totals = dict(hard_lambda_sum=int(l.sum()), io_rg=int((l+f).sum()),
                  ft_rg=int(f.sum()), io_count=int((l+f+e).sum()),
                  ft_count=int((f+e).sum()), tree_wl=100., hpwl=80., large_net_lb=0)
    metrics = _write(case / "metrics.json", dict(k=k, **totals))
    for name in ("fixed.def", "routed.def", "pin_geometry.npz", "pin_geometry_input.npz",
                 "pin_geometry.npz.json", "pin_geometry_input.npz.json", "segments.npz"):
        (run / name).write_text(name)
    routed_sha = sc.sha256_file(run / "routed.def")
    segment_sha = sc.sha256_file(run / "segments.npz")
    ev_paths = []
    for stage, name in (("gp_lg", f"evaluator_k{k}.npz"), ("router_def", f"evaluator_routed_k{k}.npz")):
        meta = dict(schema_version=1, k=k, max_degree=256, num_nets=n,
                    num_physical=120, num_movable=120, totals=totals,
                    region_sha256=f"region{k}", placement_sha256=stage,
                    net_order_sha256=array_digest(names),
                    provenance=dict(placement_stage=stage, router_def_sha256=routed_sha))
        path = case / name
        np.savez(path, per_net_crossings=l+f+e, per_net_ft=f+e, per_net_lambda=l+1,
                 per_net_steiner=l+f, per_net_home=np.zeros(n, dtype=int),
                 net_degrees=degree, pin_region_mask=np.zeros(n, dtype=np.uint64),
                 net_names=names, metadata=np.array(json.dumps(meta)))
        ev_paths.append(path)
    ev_sha = sc.sha256_file(ev_paths[1])
    identity = dict(overall_pass=True, input_def_sha256=sc.sha256_file(run / "fixed.def"),
        routed_def_sha256=routed_sha, before_geometry_sha256=sc.sha256_file(run / "pin_geometry_input.npz"),
        after_geometry_sha256=sc.sha256_file(run / "pin_geometry.npz"))
    _write(run / "verify_identity.json", identity)
    _write(run / "verify_s2.json", dict(overall_pass=True))
    _write(run / "segments.provenance.json", dict(verified=True, segments_sha256=segment_sha,
                                                  routed_def_sha256=routed_sha))
    wire = np.ones(n, dtype=bool)
    wire[[55, 56, 57, 59]] = False
    scan_rows = []
    for delta in (0, 1, 2, 4):
        path = case / (f"crossings_k{k}.json" if delta == 2 else f"crossings_k{k}_delta{delta}.json")
        actual = route + (3 if delta == 0 else 1 if delta == 1 else 0)
        crossing = dict(delta=delta, num_nets=n, num_unmatched_nets=1,
            unmatched_net_indices=[57], per_net_calibration_eligible=formal.tolist(),
            per_net_has_routed_wire=wire.tolist(), unrouted_signal_fraction=1/57,
            unrouted_signal_gate_pass=True, evaluator_file=str(ev_paths[1]), evaluator_sha256=ev_sha,
            evaluator_region_sha256=f"region{k}", evaluator_placement_sha256="router_def",
            evaluator_net_order_sha256=array_digest(names), routed_def_sha256=routed_sha,
            segments_sha256=segment_sha, per_net_route_ft=route_ft.tolist(),
            per_net_lambda_route=(l+1).tolist(), per_net_route_cross_dw=actual.tolist(),
            per_net_route_cross_raw=(route+3).tolist(), per_net_route_wl=route.tolist(),
            total_route_cross_dw=int(actual.sum()), total_route_cross_raw=int((route+3).sum()),
            total_route_wl=int(route.sum()), route_pair_demand={"0,1": int(actual.sum())})
        _write(path, crossing)
        scan_rows.append(dict(delta=delta, path=str(path), sha256=sc.sha256_file(path),
                              route_cross_dw=int(actual.sum()), route_cross_raw=int((route+3).sum())))
    _write(case / f"delta_scan_k{k}.json", dict(rows=scan_rows, routed_def_sha256=routed_sha,
        segments_sha256=segment_sha, evaluator_sha256=ev_sha))
    _write(case / f"boundary_evidence_k{k}.json", dict(population="formal", n_nets=int(formal.sum()),
        evaluator_sha256=ev_sha, crossings_sha256=sc.sha256_file(case / f"crossings_k{k}.json"),
        pairs=[dict(pair=[0, 1], length=10, evaluator=2, route=3),
               dict(pair=[1, 2], length=20, evaluator=4, route=7)],
        evaluator_statistics={}, route_statistics={}))
    inputs = [run / "fixed.def", run / "routed.def", ev_paths[0], metrics]
    outputs = [p for p in case.rglob("*") if p.is_file() and p not in inputs]
    _write(case / "evidence.execution.json", dict(status="completed",
        inputs={str(p): sc.sha256_file(p) for p in inputs},
        outputs={str(p): sc.sha256_file(p) for p in outputs}))
    return case


def _cohort(root, *, changed=False, rank_deficient=False, count=3):
    pairs = []
    for i in range(count):
        for arm in ("flat_k16", "ours_k16", "flat_k32", "ours_k32"):
            design = f"design{i}"
            _case(root, design, arm, (8, 12, 4) if changed and i == 2 else (2, 3, 1), rank_deficient)
            pairs.append((design, arm))
    return pairs


def test_complete_three_design_cohort_serializes_and_respects_population(tmp_path):
    pairs = _cohort(tmp_path)
    _write(tmp_path / "cohort.json", [dict(design=design,arm=arm) for design,arm in pairs])
    result = sc.build_calibration(str(tmp_path), sample_pairs=json.loads(json.dumps(pairs)))
    json.dumps(result, default=sc._json_default, allow_nan=False)
    assert len(result["valid_samples"]) == 12
    assert result["complete_registered_cohort"] and result["registered_cohort_size"] == 12
    assert result["c5_transfer_cv"]["verdict"] == "pass"
    assert result["g4_raw_vs_dw2_drift"]["complete"]
    for sample in result["valid_samples"]:
        scan = sample["delta_scan"]
        row = next(row for row in scan["rows"] if row["delta"] == 2)
        whole = next(row for row in scan["whole_design_supplementary"] if row["delta"] == 2)
        assert row["route_cross_dw"] == sample["route_cross_dw"]
        assert row["route_cross_raw"] == sample["route_cross_raw"]
        assert whole["route_cross_dw"] > 1000 * row["route_cross_dw"]
    assert len(result["c4_sign_invariance"]["formal_rows"]) == 6
    assert result["c4_sign_invariance"]["formal_verdict"] == "sign_invariant"
    for key, expected in zip(("alpha", "beta", "gamma"), (2, 3, 1)):
        assert result["regression"]["coefficients"][key]["value"] == pytest.approx(expected)
    assert all(s["calibration_nets"] == 55 for s in result["valid_samples"])
    assert all(row["route_ft"] is None for row in result["table6_ft_three_value"]["rows"])


@pytest.mark.parametrize("options, verdict", [
    ({"changed": True}, "fail"), ({"count": 2}, "not_evaluable"),
    ({"rank_deficient": True}, "not_evaluable"),
])
def test_transfer_requires_three_identifiable_consistent_designs(tmp_path, options, verdict):
    result = sc.build_calibration(str(tmp_path), sample_pairs=_cohort(tmp_path, **options))
    assert result["c5_transfer_cv"]["verdict"] == verdict
    if options.get("rank_deficient"):
        assert "io_calibrated" not in result["c2_regression_calibration"]
        assert result["c3_kappa_ft"]["verdict"] == "not_evaluable"


@pytest.mark.parametrize("mutation", ["omitted_identity", "false_identity", "changed_def", "empty_outputs"])
def test_paired_identity_chain_fails_closed(tmp_path, mutation):
    case = _case(tmp_path, "design", "flat_k16")
    receipt_path = case / "evidence.execution.json"
    receipt = json.loads(receipt_path.read_text())
    identity_path = case / "or_run/verify_identity.json"
    if mutation == "omitted_identity":
        del receipt["outputs"][str(identity_path)]
    elif mutation == "false_identity":
        identity = json.loads(identity_path.read_text())
        identity["overall_pass"] = False
        _write(identity_path, identity)
        receipt["outputs"][str(identity_path)] = sc.sha256_file(identity_path)
    elif mutation == "changed_def":
        (case / "or_run/fixed.def").write_text("changed input")
    else:
        receipt["outputs"] = {}
    _write(receipt_path, receipt)
    with pytest.raises(ValueError):
        sc.build_sample(str(tmp_path), "design", "flat_k16")
