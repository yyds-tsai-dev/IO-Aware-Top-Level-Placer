"""Preserve live FFT workers, then route only distinct remaining placements."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
ROOT=REPO/"results/stage2_followup_20260906"
DISPATCHER=84136
WORKERS=(748799,748804)


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)


def live(pid):
    try:return Path(f"/proc/{pid}/stat").read_text().split(")",1)[1].split()[0]!="Z"
    except FileNotFoundError:return False


def reuse_flat(design):
    source=ROOT/(design+"__flat_k16");target=ROOT/(design+"__flat_k32")
    for name in ("out.def","coord.json","netmap.json"):
        if sha(source/name)!=sha(target/name):raise ValueError(f"flat inputs differ: {design} {name}")
    left=json.loads((source/"metrics.json").read_text())["config"]
    right=json.loads((target/"metrics.json").read_text())["config"]
    if sha(left)!=sha(right):raise ValueError("flat config mismatch")
    route=source/"or_run";destination=target/"or_run";destination.mkdir(exist_ok=True)
    receipt=json.loads((route/"execution.json").read_text())
    identity=json.loads((route/"verify_identity.json").read_text())
    if receipt["status"]!="completed" or not identity["overall_pass"]:
        raise ValueError("canonical route identity incomplete")
    if sha(route/"routed.def")!=receipt["artifacts"]["routed.def"]:
        raise ValueError("canonical routed geometry changed")
    # Publish routed.def last, so the old dispatcher's existence check never sees
    # a route without its complete reuse provenance and identity evidence.
    for path in route.iterdir():
        if (not path.is_file() or "previous" in path.name or path.suffix==".tcl"
                or path.name in ("routed.def","execution.json")):
            continue
        dest=destination/path.name
        if dest.exists():raise FileExistsError(f"preserve existing target artifact: {dest}")
        shutil.copy2(path,dest)
    reused=dict(receipt,run_dir=str(target),record_kind="route_reuse",reused_from=str(source),
        reuse_time=time.time(),reuse_basis="byte-identical DEF, coordinate transform, netmap, config and referenced LEFs",
        target_def_sha256=sha(target/"out.def"),source_execution_sha256=sha(route/"execution.json"))
    write(destination/"execution.json",reused)
    os.link(route/"routed.def",destination/"routed.def")


def route_case(case):
    output=ROOT/(case+".unique_route.log")
    env=dict(os.environ,STAGE2_S8_ROOT=str(ROOT),STAGE2_ROUTE_THREADS="8",STAGE2_ROUTE_TIMEOUT_S="43200")
    print("route",case,flush=True)
    with output.open("w") as stream:
        result=subprocess.run(["bash",str(Path(__file__).resolve().parents[1] / "scripts/stage2_s8_route.sh"),"--shard="+case],
                              cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"route failed: {case}, see {output}")
    if case.endswith("__flat_k16"):reuse_flat(case.split("__")[0])
    print("completed",case,flush=True)


def main():
    state_path=ROOT/"unique_route_scheduler.json"
    if state_path.exists():raise FileExistsError("scheduler state already exists; inspect before resuming")
    command=Path(f"/proc/{DISPATCHER}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    if "bash scripts/stage2_s8_route.sh" not in command:
        raise ValueError("dispatcher PID identity changed")
    state=dict(started=time.time(),dispatcher=DISPATCHER,workers=WORKERS,status="waiting_current_fft",
        reason="reuse exactly identical flat inputs; preserve2-worker CPU concurrency")
    write(state_path,state)
    os.kill(DISPATCHER,signal.SIGSTOP)
    while any(live(pid) for pid in WORKERS):time.sleep(10)
    state["status"]="routing_unique_remaining";write(state_path,state)
    cases=[design+"__"+arm for arm in ("flat_k16","ours_k16","ours_k32")
           for design in ("des_perf_1","mempool_tile_wrap")]
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(route_case,case):case for case in cases}
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:errors.append(dict(case=futures[future],error=repr(error)))
    state.update(status="failed" if errors else "completed",errors=errors,ended=time.time())
    write(state_path,state)
    if errors:
        raise RuntimeError("unique routing failed; original dispatcher remains paused to avoid duplicate work")
    os.kill(DISPATCHER,signal.SIGCONT)
    state["original_dispatcher_resumed"]=True;write(state_path,state)


if __name__=="__main__":main()
