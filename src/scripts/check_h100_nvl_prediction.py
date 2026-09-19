"""Adjudicate the frozen NVL protocol without misattributing shared memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.m4_check_prediction import check_wall_time,check_all_phases,check_host_rss,build_mainline_hypothesis_verdict


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root
    prediction_path=root/"forecast/h100_nvl_prediction.json"
    protocol_path=root/"forecast/protocol.json"
    actual_path=root/"full_3x3_flat_k16.json"
    receipt_path=root/"full_3x3_flat_k16.execution.json"
    prediction,protocol,actual,receipt=(json.loads(p.read_text()) for p in
        (prediction_path,protocol_path,actual_path,receipt_path))
    if (receipt["status"]!="completed" or receipt["returncode"]!=0
            or receipt["output_sha256"]!=sha(actual_path)):
        raise ValueError("full workload did not complete with matching output")
    if (prediction["status"]!="frozen" or prediction["execution_protocol_sha256"]!=sha(protocol_path)
            or receipt["protocol_sha256"]!=sha(protocol_path) or receipt["prediction_sha256"]!=sha(prediction_path)
            or receipt["started"]<=protocol["registered_at_unix"]):
        raise ValueError("prospective prediction/protocol mismatch")
    if (receipt["gpu_uuid"]!=protocol["gpu_uuid"] or receipt["config_sha256"]!=protocol["config_sha256"]
            or sha(Path(protocol["config"]))!=protocol["config_sha256"]
            or receipt["source_snapshot_digest"]!=protocol["source_snapshot_digest"]
            or receipt["measurement_mode"]!="shared"):
        raise ValueError("hardware, config, source or measurement protocol mismatch")
    for relative,digest in protocol["source_sha256"].items():
        if sha(Path(protocol["source_snapshot"])/relative)!=digest:
            raise ValueError(f"frozen source changed: {relative}")
    for name,key in (("full_3x3_flat_k16.log","log_sha256"),
                     ("full_3x3_flat_k16.device.jsonl","device_history_sha256")):
        if sha(root/name)!=receipt[key]:
            raise ValueError(f"execution artifact changed: {name}")
    if sha(Path(protocol["input_manifest"]))!=protocol["input_manifest_sha256"]:
        raise ValueError("registered input manifest changed")
    for key,expected in dict(k=16,rtype="grid",mode="io",rho_max=0,dp_seed=1000,
            callback_order="legacy",f_ft_max=0,ft_reweight="off",wl_reweight="off",no_diag=True).items():
        if actual.get(key)!=expected:
            raise ValueError(f"actual arm differs from frozen protocol: {key}")
    import numpy as np
    coordinate_path=Path(str(actual_path)+".npz")
    with np.load(coordinate_path,allow_pickle=False) as archive:
        if len(archive["node_x"])!=protocol["physical_nodes"] or len(archive["node_y"])!=protocol["physical_nodes"]:
            raise ValueError("final placement physical node count mismatch")
        if not np.isfinite(archive["node_x"]).all() or not np.isfinite(archive["node_y"]).all():
            raise ValueError("nonfinite final placement coordinates")
    wall=check_wall_time(actual,prediction)
    phases=check_all_phases(actual,prediction)
    hypothesis=build_mainline_hypothesis_verdict(phases,wall)
    peaks=[phase.get("peak_alloc_gb") for phase in actual.get("phases",{}).values()]
    peaks=[value for value in peaks if value is not None]
    history=[json.loads(line) for line in (root/"full_3x3_flat_k16.device.jsonl").read_text().splitlines()]
    whole_peak=max((row["used_gib"] for row in history if "used_gib" in row),default=None)
    result=dict(protocol_compatible=True,receipt_sha256=sha(receipt_path),
        prediction_sha256=sha(prediction_path),actual_sha256=sha(actual_path),
        phase_checks=phases,wall_time=wall,mainline_hypothesis=hypothesis,
        host_rss=check_host_rss(actual,prediction),
        gpu_memory=dict(verdict="not_applicable_shared_attribution",whole_card_peak_gib=whole_peak,
            process_peak_allocated_gib=max(peaks) if peaks else actual.get("peak_mem_mb",0)/1024,
            analytic_forecast=prediction["gpu_memory_forecast"],
            reason="whole-card usage and process analytic bounds have different attribution"),
        legality={key:actual.get(key) for key in ("legalization_status","num_unplaced_cells","stop_overflow_reached","final_overflow")},
        registered_counts={key:protocol[key] for key in ("physical_nodes","movable_nodes","nets","raw_pins","canonical_pins")},
        count_scope="physical coordinate length observed; pin/net counts from verified source/cache protocol",
        supervisor_wall_s=receipt["supervisor_wall_s"],historical_sxm_forecast="unchanged; inapplicable SKU",
        source_scope="full run uses frozen source snapshot; later development is separate",
        coordinate_sha256=sha(coordinate_path),coordinate_hash_scope="computed during adjudication; supervisor bound JSON/log/device history",
        evaluation_procedure="frozen final phase runs serial CPU reference then GPU evaluator",
        exclusive_speedup_claim=False,quality_claims_prohibited=True)
    if not hypothesis.get("confirmed"):
        result["required_disclosure"]="Frozen runtime assumption was not confirmed; explain phase differences and refit separately without widening this forecast."
    (root/"prediction_check.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
