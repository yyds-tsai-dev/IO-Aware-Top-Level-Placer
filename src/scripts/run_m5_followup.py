"""Frozen M5 four-arm comparisons, followed by conditional seed confirmation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)


def initialize(root):
    manifest=root/"manifest.json"
    if manifest.exists():return json.loads(manifest.read_text())
    root.mkdir(parents=True,exist_ok=True);configs=root/"configs";configs.mkdir(exist_ok=True)
    dp=Path(os.environ["DREAMPLACE_ROOT"])
    templates=dict(adaptec1=dp/"install/test/ispd2005/adaptec1.json",
        bigblue4=dp/"install/test/ispd2005/bigblue4.json",
        mempool_tile_wrap=REPO/"results/stage2_followup_20260906/configs/mempool_tile_wrap.json")
    paths={};inputs={}
    for design,template in templates.items():
        config=json.loads(template.read_text());config.update(gpu=1,num_threads=8,detailed_place_flag=0,plot_flag=0)
        path=configs/(design+".json");write(path,config);paths[design]=str(path)
        files=[]
        for key in ("lef_input","def_input","verilog_input","sdc_input","aux_input"):
            value=config.get(key)
            if value:
                files.extend(value if isinstance(value,list) else [value])
        resolved=[Path(p) if os.path.isabs(p) else dp/"install"/p for p in files]
        if config.get("aux_input"):
            aux=Path(config["aux_input"])
            if not aux.is_absolute():aux=dp/"install"/aux
            tokens=[token for line in aux.read_text().splitlines() for token in line.split("#",1)[0].split()
                    if token.endswith((".nodes",".nets",".wts",".pl",".scl",".shapes"))]
            resolved.extend(aux.parent/token for token in tokens)
        inputs[design]={str(p.resolve()):sha(p) for p in resolved}
    snapshot=root/"source"
    shutil.copytree(Path(__file__).resolve().parents[1] / "ioplace",snapshot/"ioplace",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    hashes={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    record=dict(frozen_at=time.time(),source_sha256=hashes,configs=paths,
        config_sha256={design:sha(Path(path)) for design,path in paths.items()},input_sha256=inputs,
        registration_sha256=sha(REPO/"docs/experiments/2026-09-06-m5-preregistration.md"),
        arms=["none","ce","refine","ce_refine"],k=16,rtype="grid",rho_max=0,
        initial_seed=1000,confirmation_seeds=[1001,1002,1003],max_active=65536,timeout_s=7200)
    write(manifest,record);return record


def execute(root,manifest,design,mode,seed,gpu):
    name=f"{design}__{mode}__seed{seed}";directory=root/"runs";directory.mkdir(exist_ok=True)
    output=directory/(name+".json");receipt=directory/(name+".execution.json")
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if (prior.get("returncode")==0 and output.exists() and prior["output_sha256"]==sha(output)
                and prior["manifest_sha256"]==sha(root/"manifest.json")):
            print("verified skip",name,flush=True);return json.loads(output.read_text())
        raise ValueError(f"preserve failed M5 run before an explicitly labeled retry: {name}")
    config=manifest["configs"][design]
    if sha(Path(config))!=manifest["config_sha256"][design]:raise ValueError("M5 config changed")
    for path,digest in manifest["input_sha256"][design].items():
        if sha(path)!=digest:raise ValueError("M5 input changed")
    snapshot=root/"source"
    command=[sys.executable,"-m","ioplace.drivers.run_placement","--config",config,
        "--mode","io","--k","16","--rtype","grid","--seed","0","--dp-seed",str(seed),
        "--deterministic","1","--rho-max","0","--every","50","--no-diag",
        "--discrete-mode",mode,"--discrete-max-active","65536","--out",str(output)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(snapshot),PYTHONDONTWRITEBYTECODE="1")
    log=directory/(name+".log")
    from ioplace.bench.spike_30m import device_snapshot
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    snapshot_before=device_snapshot()
    if snapshot_before["free_gib"]<12:
        raise RuntimeError("selected shared GPU lacks M5 launch headroom")
    record=dict(command=command,started=time.time(),manifest_sha256=sha(root/"manifest.json"),
                gpu=str(gpu),device_before=snapshot_before,measurement_mode="shared")
    print("start",name,"GPU",gpu,flush=True)
    with log.open("w") as stream:
        process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record["pid"]=process.pid;write(receipt,record)
        try:code=process.wait(timeout=manifest["timeout_s"])
        except subprocess.TimeoutExpired:
            process.kill();code=process.wait();record["timed_out"]=True
    record.update(returncode=code,ended=time.time(),log_sha256=sha(log),output_sha256=sha(output) if output.exists() else None)
    record["device_after"]=device_snapshot()
    write(receipt,record)
    print("finish",name,code,flush=True)
    if code:raise RuntimeError(f"M5 workload failed: {name}, see {log}")
    return json.loads(output.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--design",required=True,choices=["adaptec1","bigblue4","mempool_tile_wrap"])
    parser.add_argument("--gpu",required=True)
    parser.add_argument("--initialize-only",action="store_true")
    args=parser.parse_args();root=args.root.resolve();manifest=initialize(root)
    if args.initialize_only:return
    for path,digest in manifest["source_sha256"].items():
        if sha(root/"source"/path)!=digest:raise ValueError("M5 frozen source changed")
    results={mode:execute(root,manifest,args.design,mode,1000,args.gpu) for mode in manifest["arms"]}
    accepted=[mode for mode,value in results.items() if mode!="none" and value["discrete_result"]["accepted"]]
    write(root/(args.design+".screening.json"),dict(accepted_modes=accepted,
        results={mode:{key:value[key] for key in ("io_count","ft_count","hpwl","legalization_status")}
                 for mode,value in results.items()}))
    if accepted:
        for seed in manifest["confirmation_seeds"]:
            for mode in ["none",*accepted]:execute(root,manifest,args.design,mode,seed,args.gpu)


if __name__=="__main__":main()
