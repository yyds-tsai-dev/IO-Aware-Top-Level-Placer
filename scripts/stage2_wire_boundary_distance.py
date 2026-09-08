"""G4 posthoc wire-distance diagnostic on the validated primary population."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ioplace.diagnostics.stage2_calibration import build_sample, sha256_file
from ioplace.export.evaluation import array_digest
from ioplace.route_eval.boundary_distance import wire_boundary_distance, DEFAULT_DISTANCES
from ioplace.route_eval.route_crossings import (CoordMap, align_net_indices,
    load_net_order, region_grid_from_json)
from ioplace.route_eval.segments import load_segments


def diagnose(root, design, arm):
    sample = build_sample(str(root), design, arm)  # validates full receipt/hash chain
    case = (root / sample["sample_id"]).resolve()
    path = Path(sample["source_paths"]["crossings_matched"])
    crossing = json.loads(path.read_text())
    receipt = json.loads((case / "evidence.execution.json").read_text())
    recorded = dict(receipt["inputs"], **receipt["outputs"])
    paths = {key: Path(crossing[key]).resolve() for key in
             ("segments", "regions", "netmap", "coord", "evaluator_file")}
    paths.update(crossings=path.resolve(), receipt=case / "evidence.execution.json",
                 routed_def=case / "or_run/routed.def",
                 segments_metadata=paths["segments"].with_suffix(".json"))
    for key in ("segments", "regions", "netmap", "coord", "evaluator_file", "crossings"):
        if recorded.get(str(paths[key])) != sha256_file(paths[key]):
            raise ValueError(f"unregistered or changed diagnostic input: {key}")
    segments = load_segments(paths["segments"])
    names = load_net_order(paths["netmap"])
    mapping, _ = align_net_indices(segments, names)
    eligible = np.asarray(crossing["per_net_calibration_eligible"], dtype=bool)
    if eligible.shape != mapping.shape or np.any(mapping[eligible] < 0):
        raise ValueError("formal population cannot align to routed net names")
    if not crossing["unrouted_signal_gate_pass"]:
        raise ValueError("formal diagnostic requires the <=2% coverage gate")
    mask = np.zeros(segments.num_nets, dtype=bool)
    mask[mapping[eligible]] = True
    if int(mask.sum()) != sample["calibration_nets"]:
        raise ValueError("diagnostic population differs from primary calibration")
    rg = region_grid_from_json(paths["regions"])
    coord = CoordMap.from_json(paths["coord"])
    return dict(sample_id=sample["sample_id"], k=sample["k_placement"],
        population="primary formal eligible routed signal nets",
        n_nets=int(mask.sum()), evaluator_order_mask_sha256=array_digest(eligible),
        inputs={key: dict(path=str(value), sha256=sha256_file(value)) for key,value in paths.items()},
        formal=wire_boundary_distance(segments, rg, coord, net_mask=mask),
        whole_design_supplementary=wire_boundary_distance(segments, rg, coord))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol["source_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError("diagnostic source changed after protocol registration")
    outputs = {}
    for entry in json.loads((args.root / "cohort.json").read_text()):
        row = diagnose(args.root, entry["design"], entry["arm"])
        path = args.out / (row["sample_id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
        outputs[str(path.resolve())] = sha256_file(path)
        print(row["sample_id"], row["formal"]["any_boundary_cdf"], flush=True)
    (args.out / "execution.json").write_text(json.dumps(dict(status="completed",
        protocol_sha256=sha256_file(protocol_path), outputs=outputs), indent=2)+"\n")


if __name__ == "__main__":
    main()
