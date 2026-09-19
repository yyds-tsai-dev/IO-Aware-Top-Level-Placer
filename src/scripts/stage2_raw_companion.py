"""G4 raw-crossing companion; primary delta2 evidence remains immutable."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ioplace.diagnostics import stage2_calibration as cal
from ioplace.export.evaluation import array_digest


def project_raw(sample, primary, raw):
    """Project only delta-sensitive values after checking shared evidence."""
    if primary["delta"] != 2 or raw["delta"] != 0:
        raise ValueError("expected primary delta2 and companion delta0")
    for key in ("per_net_calibration_eligible", "per_net_has_routed_wire",
                "per_net_route_ft", "per_net_route_wl", "route_pair_demand",
                "unrouted_signal_gate_pass", "evaluator_sha256", "segments_sha256",
                "routed_def_sha256", "per_net_route_cross_raw"):
        if primary[key] != raw[key]:
            raise ValueError(f"raw companion changed delta-invariant evidence: {key}")
    route = np.asarray(raw["per_net_route_cross_dw"], dtype=np.int64)
    if not np.array_equal(route, raw["per_net_route_cross_raw"]):
        raise ValueError("delta0 does not equal raw crossings")
    if route.shape != sample["per_net_route_cross_dw"].shape:
        raise ValueError("raw route net-index shape mismatch")
    result = dict(sample)
    formal = cal._paired_arrays(sample)[4]
    total = int(route[formal].sum())
    result.update(delta=0, route_target="route_cross_raw", route_cross_dw=total,
        per_net_route_cross_dw=route,
        per_net_lambda_route=np.asarray(raw["per_net_lambda_route"], dtype=np.int64),
        route_minus_io_rg=total-sample["io_rg"], io_mst_minus_route=sample["io_mst"]-total)
    if total != sample["route_cross_raw"]:
        raise ValueError("raw formal total differs from frozen primary evidence")
    return result


def summarize(samples):
    output = {name: getattr(cal, name)(samples) for name in (
        "table1_total_ratio", "table2_per_net_correlation", "table3_degree_bucket",
        "table4_lambda_route_bucket", "table5_three_value_decomposition",
        "table6_ft_three_value", "table7_boundary_pair_demand")}
    fit = cal.fit_regression(samples)
    fit["regressor_definitions"]["y"] = "route_cross_raw (= route_cross_dw(0))"
    output["regression"] = fit
    output["per_design_regression_informational"] = cal.fit_per_design_regression(samples)
    output["c1_model_selection"] = cal.judge_c1(output["table1_total_ratio"], output["table2_per_net_correlation"])
    output["c2_regression_calibration"], output["c3_kappa_ft"] = cal.judge_c2_c3(samples, output["table1_total_ratio"], fit)
    output["c4_sign_invariance"] = cal.judge_c4(samples)
    for row in output["c4_sign_invariance"]["rows"]:
        row["post_route_metric"] = "route_cross_raw"
    output["c5_transfer_cv"] = cal.judge_c5(samples)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples, inputs, populations = [], {}, []
    for row in json.loads((args.root / "cohort.json").read_text()):
        sample = cal.build_sample(str(args.root), row["design"], row["arm"])
        path = Path(sample["source_paths"]["crossings_matched"])
        raw_path = path.with_name(path.stem+"_delta0.json")
        receipt = json.loads((path.parent / "evidence.execution.json").read_text())
        if receipt["outputs"].get(str(raw_path.resolve())) != cal.sha256_file(raw_path):
            raise ValueError("unregistered raw companion")
        primary, raw = (json.loads(p.read_text()) for p in (path, raw_path))
        projected = project_raw(sample, primary, raw)
        samples.append(projected)
        target, models, _, _, formal = cal._paired_arrays(sample)
        model_active = np.logical_or.reduce([v > 0 for v in models.values()])
        dw_active = formal & ((target > 0) | model_active)
        raw_active = formal & ((projected["per_net_route_cross_dw"] > 0) | model_active)
        populations.append(dict(sample_id=sample["sample_id"], formal_n=int(formal.sum()),
            n_active_dw2=int(dw_active.sum()), n_active_raw=int(raw_active.sum()),
            raw_only_additions=int((raw_active & ~dw_active).sum()),
            dw2_mask_sha256=array_digest(dw_active), raw_mask_sha256=array_digest(raw_active)))
        inputs[sample["sample_id"]] = {str(p.resolve()): cal.sha256_file(p) for p in (path, raw_path)}
        print("validated", sample["sample_id"], flush=True)
    output = summarize(samples)
    output.update(status="completed", target="route_cross_raw (= delta0)",
        role="G4 supplementary dual-value conclusions; delta2 remains primary",
        field_convention="legacy route_cross_dw field names in tables denote target delta0 here",
        population="same formal eligible nets; active correlation/regression mask follows the existing target>0 OR any_model>0 rule",
        invariant_tables="FT and boundary-pair demand use raw visited regions/physical transitions and are delta-independent; exact equality checked",
        active_population_comparison=populations,
        inputs=inputs, source_sha256={str(Path(p).resolve()): cal.sha256_file(p) for p in
            (__file__, cal.__file__)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(".tmp")
    temp.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    temp.replace(args.out)


if __name__ == "__main__":
    main()
