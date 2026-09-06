"""Finish analysis when the already-running registered experiments complete."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
H100 = REPO / "results/h100_nvl_followup_20260906"
STAGE2 = REPO / "results/stage2_followup_20260906"
STATE = REPO / "results/finalize_followup_20260906.json"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(value):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(STATE)


def execute(command, log):
    if log.exists():
        raise FileExistsError(f"preserve earlier finalization attempt: {log}")
    record = dict(command=list(map(str, command)), started=time.time())
    with log.open("w") as stream:
        proc = subprocess.Popen(record["command"], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        record["pid"] = proc.pid
        code = proc.wait()
    record.update(returncode=code, ended=time.time(), log_sha256=sha(log))
    if code:
        raise RuntimeError(f"analysis failed with {code}; {log}")
    return record


def main():
    if STATE.exists():
        raise FileExistsError("inspect previous finalization state before restarting")
    state = dict(status="waiting", pid=os.getpid(), started=time.time(), h100="waiting", stage2="waiting", commands=[])
    write(state)
    cohort = json.loads((STAGE2 / "cohort.json").read_text())
    try:
        while state["h100"] != "completed" or state["stage2"] != "completed":
            receipt = json.loads((H100 / "full_3x3_flat_k16.execution.json").read_text())
            if receipt["status"] == "failed":
                raise RuntimeError("full H100 workload failed; preserve first-run evidence")
            if state["h100"] == "waiting" and receipt["status"] == "completed":
                state["h100"] = "analyzing"; write(state)
                for name in ("check_h100_nvl_prediction", "report_h100_nvl_followup"):
                    script = REPO / "scripts" / (name+".py")
                    command = [sys.executable, script, "--root", H100]
                    result = execute(command, H100 / (name+".finalize.log"))
                    result["script_sha256"] = sha(script)
                    state["commands"].append(result)
                state["h100"] = "completed"
                print("H100 adjudication/report completed", flush=True); write(state)
            ready = True
            for row in cohort:
                path = STAGE2 / (row["design"]+"__"+row["arm"]) / "evidence.execution.json"
                if not path.exists():
                    ready = False
                    continue
                status = json.loads(path.read_text())["status"]
                if status == "failed":
                    raise RuntimeError(f"paired evidence failed: {path}")
                ready &= status == "completed"
            if state["stage2"] == "waiting" and ready:
                state["stage2"] = "analyzing"; write(state)
                output = STAGE2 / "calibration"
                state["commands"].append(execute([sys.executable,"-m","ioplace.diagnostics.stage2_calibration",
                    "--s8-dir",STAGE2,"--out-dir",output],STAGE2 / "calibration.finalize.log"))
                tables = json.loads((output / "tables.json").read_text())
                if not tables["complete_registered_cohort"] or len(tables["valid_samples"]) != 12:
                    raise ValueError("final calibration does not cover the registered cohort")
                script = REPO / "scripts/report_stage2_followup.py"
                state["commands"].append(execute([sys.executable,script,"--tables",output / "tables.json"],
                    STAGE2 / "report.finalize.log"))
                state["stage2"] = "completed"
                print("Stage2 full calibration/report completed", flush=True); write(state)
            if state["h100"] != "completed" or state["stage2"] != "completed":
                time.sleep(20)
        state.update(status="completed", ended=time.time())
    except BaseException as error:
        state.update(status="failed", error=repr(error), ended=time.time())
        raise
    finally:
        write(state)


if __name__ == "__main__":
    main()
