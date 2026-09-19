"""Execute the prospectively frozen shared-NVL full GP/LG/evaluator workload."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
ROOT = REPO / "results/h100_nvl_followup_20260906"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ioplace.bench.spike_30m import device_snapshot


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


def validate_inputs(protocol):
    for path, expected in ((protocol["config"], protocol["config_sha256"]),
                           (protocol["input_manifest"], protocol["input_manifest_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    source = Path(protocol["source_snapshot"])
    for path, expected in protocol["source_sha256"].items():
        if sha(source / path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    cache = REPO / "results/recovery_visible_20260906/cache/3x3_n2"
    verification = json.loads((cache / "verification.json").read_text())
    if not verification["verify"]["ok"] or verification["verify"]["mode"] != "full":
        raise ValueError("full cache verification required")
    if sha(cache / "meta.json") != protocol["cache_meta_sha256"]:
        raise ValueError("frozen cache metadata changed")
    for scenario in protocol["t9_scenarios"]:
        result = json.loads((ROOT / f"t9_{scenario}.json").read_text())
        if (result["feasibility_verdict"] != "completed_within_registered_process_budget"
                or result["n_interleaved_iters"] < protocol["t9_n_iter"]
                or result["n_pins_canonical"] != protocol["canonical_pins"]):
            raise ValueError(f"T9 {scenario} did not meet the registered process gate")


def main():
    protocol_path = ROOT / "forecast/protocol.json"
    prediction_path = ROOT / "forecast/h100_nvl_prediction.json"
    protocol, prediction = (json.loads(path.read_text()) for path in (protocol_path, prediction_path))
    if prediction["execution_protocol_sha256"] != sha(protocol_path) or prediction["status"] != "frozen":
        raise ValueError("prediction/protocol freeze mismatch")
    validate_inputs(protocol)
    os.environ["CUDA_VISIBLE_DEVICES"] = protocol["gpu_uuid"]
    before = device_snapshot()
    if before["uuid"] != protocol["gpu_uuid"] or before["free_gib"] < (
            protocol["process_budget_gib"] + protocol["launch_free_memory_margin_gib"]):
        raise RuntimeError("registered GPU currently lacks launch memory margin")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0])/2**20
    if available < protocol["host_available_min_gib"]:
        raise RuntimeError("host currently lacks registered available memory")
    output = ROOT / "full_3x3_flat_k16.json"
    receipt = ROOT / "full_3x3_flat_k16.execution.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("preserve the previous attempted run before an explicitly labeled rerun")
    cmd = [sys.executable, "-m", "ioplace.drivers.run_placement", "--config", protocol["config"],
        "--mode", "io", "--k", "16", "--rtype", "grid", "--seed", "1000",
        "--dp-seed", "1000", "--deterministic", "1", "--rho-max", "0",
        "--callback-order", "legacy", "--f-ft-max", "0", "--ft-reweight", "off",
        "--wl-reweight", "off", "--every", "50", "--no-diag",
        "--benchmark-kind", "synthetic", "--out", str(output)]
    env = dict(os.environ, PYTHONPATH=protocol["source_snapshot"], PYTHONDONTWRITEBYTECODE="1")
    started = time.time()
    record = dict(status="running", started=started, command=cmd, measurement_mode="shared",
        protocol_sha256=sha(protocol_path), prediction_sha256=sha(prediction_path),
        config_sha256=protocol["config_sha256"], gpu_uuid=before["uuid"],
        source_snapshot_digest=protocol["source_snapshot_digest"], preflight=before,
        host_available_gib=available, supervisor_source_sha256=sha(__file__),
        dreamplace_commit=subprocess.check_output(["git", "-C", os.environ["DREAMPLACE_ROOT"],
                                                  "rev-parse", "HEAD"], text=True).strip(),
        dreamplace_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "-C",
                                  os.environ["DREAMPLACE_ROOT"], "diff", "HEAD"])).hexdigest())
    history_path = ROOT / "full_3x3_flat_k16.device.jsonl"
    log_path = ROOT / "full_3x3_flat_k16.log"
    with log_path.open("w") as log, history_path.open("w") as history:
        process = subprocess.Popen(cmd, cwd=protocol["source_snapshot"], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = process.pid
        write(receipt, record)
        timed_out = False
        while process.poll() is None:
            try:
                sample = device_snapshot()
            except Exception as error:
                sample = dict(time=time.time(), snapshot_error=str(error))
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                sample["process_status_memory"] = {line.split(":",1)[0]:line.split(":",1)[1].strip()
                    for line in status.splitlines() if line.startswith(("VmRSS:", "VmHWM:"))}
            except FileNotFoundError:
                pass
            history.write(json.dumps(sample)+"\n")
            history.flush()
            if time.time()-started > protocol["full_run_timeout_s"]:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(2)
        code = process.wait()
    record.update(ended=time.time(), returncode=code, timed_out=timed_out,
                  supervisor_wall_s=time.time()-started, log_sha256=sha(log_path),
                  device_history_sha256=sha(history_path), status="completed" if code == 0 else "failed")
    if output.exists():
        record["output_sha256"] = sha(output)
    elif code == 0:
        record.update(status="failed", error="child exited zero without output")
    write(receipt, record)
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
