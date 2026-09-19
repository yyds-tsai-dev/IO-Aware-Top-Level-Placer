"""Decode verified routes, recompute colocated evaluator evidence, and scan delta.

Every subprocess and all input/output hashes are recorded. A completed case is
reused only when its receipt still matches the current files. GP evidence is
retained separately from the evaluator recomputed on the routed geometry.
"""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ioplace.export.evaluation import load_evaluation


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def execute(command, log, *, tolerate_system_exit_zero=False):
    start = time.time()
    for previous in (log,log.with_suffix(".execution.json")):
        if previous.exists():
            previous.rename(previous.with_name(previous.name+f".previous-{time.time_ns()}"))
    with log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, stdout=stream,
                                   stderr=subprocess.STDOUT)
        execution = dict(command=list(map(str, command)), pid=process.pid, started=start)
        write(log.with_suffix(".execution.json"), execution)
        code = process.wait()
    execution.update(ended=time.time(), returncode=code, log_sha256=sha(log))
    # OpenROAD's embedded Python catches SystemExit(0) as an interpreter error.
    quirk = tolerate_system_exit_zero and code != 0 and "SystemExit: 0" in log.read_text()
    execution["embedded_python_system_exit_zero"] = bool(quirk)
    write(log.with_suffix(".execution.json"), execution)
    if code and not quirk:
        raise RuntimeError(f"command exited {code}; see {log}")
    return execution


def process_case(root, row, openroad):
    name = row["design"] + "__" + row["arm"]
    case = root / name
    run = case / "or_run"
    routed = run / "routed.def"
    receipt = case / "evidence.execution.json"
    config = Path(row["config"])
    source_files = [Path(__file__), Path(__file__).resolve().parents[1] / "scripts/stage2_s8_extract_crossings.py",
        Path(__file__).resolve().parents[1] / "scripts/stage2_recompute_evaluator.py", Path(__file__).resolve().parents[1] / "ioplace/export/evaluation.py",
        Path(__file__).resolve().parents[1] / "scripts/stage2_boundary_evidence.py",
        Path(__file__).resolve().parents[1] / "scripts/stage2_verify_identity.py",
        Path(__file__).resolve().parents[1] / "ioplace/evaluator_gpu.py", Path(__file__).resolve().parents[1] / "ioplace/route_eval/route_crossings.py",
        Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/dump_segments.py",
        Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/dump_pin_geometry.py",
        Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/verify_routed_def.py"]
    snapshot_manifest = REPO / "source_manifest.json"
    if snapshot_manifest.exists():
        manifest = json.loads(snapshot_manifest.read_text())
        for relative, digest in manifest["files"].items():
            path = REPO / relative
            if sha(path) != digest:
                raise ValueError(f"frozen evidence source changed: {relative}")
            source_files.append(path)
        source_files.append(snapshot_manifest)
    inputs = [routed, run / "fixed.def", config, case / "netmap.json", case / "coord.json",
              case / "metrics.json", case / f"evaluator_k{row['k']}.npz", Path(openroad),
              *source_files]
    inputs.extend(Path(path) for path in json.loads(config.read_text())["lef_input"])
    regions = {k: root / (row["design"] + f"__flat_k{k}") / "regions.json" for k in (16, 32)}
    inputs.extend(regions.values())
    hashes = {str(path): sha(path) for path in inputs}
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (old.get("status") == "completed" and old.get("inputs") == hashes
                and all(Path(path).exists() and sha(path) == digest
                        for path, digest in old["outputs"].items())):
            print("verified skip", name, flush=True)
            return
        receipt.rename(receipt.with_name(receipt.name+f".previous-{time.time_ns()}"))
    record = dict(status="running", pid=os.getpid(), started=time.time(), inputs=hashes,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), commands=[])
    write(receipt, record)
    outputs = []
    try:
        lefs = json.loads(config.read_text())["lef_input"]
        lef_args = [part for lef in lefs for part in ("--lef", lef)]
        segments, verify = run / "segments.npz", run / "verify_s2.json"
        for path in (segments, verify):
            if path.exists():
                path.rename(path.with_name(path.name + f".previous-{time.time_ns()}"))
        record["commands"].append(execute([openroad, "-python",
            Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/dump_segments.py", *lef_args,
            "--def", routed, "--out", segments, "--selfcheck", "--design-name", name],
            run / "dump_segments.log"))
        record["commands"].append(execute([openroad, "-python",
            Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/verify_routed_def.py", *lef_args,
            "--def", routed, "--sample", "1000", "--seed", "0",
            "--report-out", run / "verify_wire_length.rpt", "--json-out", verify],
            run / "verify_s2.log", tolerate_system_exit_zero=True))
        if not json.loads(verify.read_text())["overall_pass"]:
            raise ValueError("route wire-length/parser verification failed")
        import numpy as np
        with np.load(segments, allow_pickle=False) as archive:
            if not np.any(archive["seg_kind"] == 0):
                raise ValueError("empty decoded wire segments")
        provenance = run / "segments.provenance.json"
        write(provenance, dict(verified=True, routed_def_sha256=sha(routed),
              segments_sha256=sha(segments), verify_sha256=sha(verify),
              tool_sha256=hashes[str(Path(openroad))], inputs=hashes))
        outputs.extend((segments, verify, provenance))
        geometry = run / "pin_geometry.npz"
        record["commands"].append(execute([openroad,"-python",
            Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/dump_pin_geometry.py",*lef_args,
            "--def",routed,"--out",geometry],run / "dump_pin_geometry.log"))
        outputs.extend((geometry,Path(str(geometry)+".json")))
        input_geometry=run / "pin_geometry_input.npz"
        original_def=run / "fixed.def"
        record["commands"].append(execute([openroad,"-python",
            Path(__file__).resolve().parents[1] / "ioplace/route_eval/or_scripts/dump_pin_geometry.py",*lef_args,
            "--def",original_def,"--out",input_geometry],run / "dump_input_pin_geometry.log"))
        identity=run / "verify_identity.json"
        record["commands"].append(execute([sys.executable,Path(__file__).resolve().parents[1] / "scripts/stage2_verify_identity.py",
            "--before",input_geometry,"--after",geometry,"--out",identity],run / "verify_identity.log"))
        outputs.extend((input_geometry,Path(str(input_geometry)+".json"),identity))
        for k in (16, 32):
            evaluator = case / f"evaluator_routed_k{k}.npz"
            common = ["--regions", regions[k], "--netmap", case / "netmap.json",
                      "--coord", case / "coord.json"]
            record["commands"].append(execute([sys.executable,
                Path(__file__).resolve().parents[1] / "scripts/stage2_recompute_evaluator.py", "--config", config,
                "--def", routed, "--pin-geometry",geometry,*common,"--out",evaluator],
                run / f"recompute_k{k}.log"))
            outputs.append(evaluator)
            scan = []
            for delta in (0, 1, 2, 4):
                out = case / (f"crossings_k{k}.json" if delta == 2 else f"crossings_k{k}_delta{delta}.json")
                record["commands"].append(execute([sys.executable,
                    Path(__file__).resolve().parents[1] / "scripts/stage2_s8_extract_crossings.py", "--segments", segments,
                    *common, "--evaluator", evaluator, "--routed-def", routed,
                    "--segments-provenance", provenance, "--delta", str(delta), "--out", out],
                    run / f"extract_k{k}_delta{delta}.log"))
                value = json.loads(out.read_text())
                scan.append(dict(delta=delta, route_cross_dw=value["total_route_cross_dw"],
                                 route_cross_raw=value["total_route_cross_raw"],
                                 path=str(out), sha256=sha(out)))
                outputs.append(out)
            scan_path = case / f"delta_scan_k{k}.json"
            write(scan_path, dict(rows=scan, routed_def_sha256=sha(routed),
                                 evaluator_sha256=sha(evaluator), segments_sha256=sha(segments)))
            outputs.append(scan_path)
            boundary=case / f"boundary_evidence_k{k}.json"
            record["commands"].append(execute([sys.executable,Path(__file__).resolve().parents[1] / "scripts/stage2_boundary_evidence.py",
                "--crossings",case / f"crossings_k{k}.json","--evaluator",evaluator,
                "--pin-geometry",geometry,"--routed-def",routed,"--segments",segments,
                *common,"--out",boundary],run / f"boundary_k{k}.log"))
            outputs.append(boundary)
            gp = case / f"evaluator_k{k}.npz"
            if gp.exists():
                before, after = load_evaluation(gp)["metadata"], load_evaluation(evaluator)["metadata"]
                drift = case / f"router_geometry_delta_k{k}.json"
                write(drift, dict(gp_evaluator_sha256=sha(gp), routed_evaluator_sha256=sha(evaluator),
                    before=before["totals"], after=after["totals"],
                    delta={key: after["totals"][key]-value for key,value in before["totals"].items()},
                    note="includes DEF coordinate quantization and any router geometry change"))
                outputs.append(drift)
        record.update(status="completed", outputs={str(path):sha(path) for path in outputs})
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["ended"] = time.time()
        write(receipt, record)
    print("completed", name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", default="*")
    parser.add_argument("--watch", action="store_true", help="wait for all selected route receipts to complete")
    parser.add_argument("--openroad", default=os.environ.get("OPENROAD_BIN"))
    args = parser.parse_args()
    if not args.openroad:
        parser.error("source scripts/openroad_env.sh or set --openroad")
    rows = json.loads((args.root / "cohort.json").read_text())
    pending = [row for row in rows if fnmatch.fnmatch(row["design"]+"__"+row["arm"], args.case)]
    while pending:
        waiting = []
        for row in pending:
            case = args.root / (row["design"]+"__"+row["arm"])
            route_receipt = case / "or_run/execution.json"
            if args.watch and (not route_receipt.exists() or
                    json.loads(route_receipt.read_text()).get("status") != "completed"):
                waiting.append(row)
                continue
            process_case(args.root.resolve(), row, str(Path(args.openroad).resolve()))
        pending = waiting
        if pending:
            time.sleep(20)


if __name__ == "__main__":
    main()
