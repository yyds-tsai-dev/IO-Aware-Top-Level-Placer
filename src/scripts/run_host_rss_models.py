"""Native raw Bookshelf read HWM: training, frozen fit, then 2x2 holdout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
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
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)


def read_point(config,out):
    from ioplace.dreamplace_env import setup_dreamplace
    from ioplace.bench.tile_bookshelf import read_aux,_nets_header
    setup_dreamplace()
    import Params,PlaceDB
    params=Params.Params();params.load(str(config));params.gpu=0
    paths=read_aux(params.aux_input)
    raw_nets,raw_pins=_nets_header(paths["nets"])
    host_before=Path("/proc/meminfo").read_text()
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    started=time.perf_counter()
    db=PlaceDB.PlaceDB();db.read(params)
    elapsed=time.perf_counter()-started
    hwm=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    # Names are already materialized by read(); the HWM above precedes this sum.
    name_bytes=sum(len(name) for name in db.node_names)+sum(len(name) for name in db.net_names)
    record=dict(status="completed",config=str(config),config_sha256=sha(config),
        phase="native_raw_read_without_initialize",read_s=elapsed,host_peak_rss_gib=hwm,
        baseline_hwm_gib=baseline,N=int(db.num_physical_nodes),P_raw=int(raw_pins),
        P_canonical=len(db.pin2node_map),E_raw=int(raw_nets),E_native=int(db.num_nets),
        name_bytes=int(name_bytes),pid=os.getpid(),python=sys.version,
        host_meminfo_before=host_before,
        memory_scope="absolute process RUSAGE_SELF HWM; GiB; no PlaceDB.initialize or GPU context",
        dreamplace_commit=subprocess.check_output(["git","-C",os.environ["DREAMPLACE_ROOT"],"rev-parse","HEAD"],text=True).strip(),
        source_sha256=sha(__file__))
    write(out,record)


def alias_prefix(prefix,alias_name="rssInput"):
    from ioplace.bench.tile_bookshelf import read_aux
    prefix=Path(prefix);paths=read_aux(str(prefix)+".aux")
    alias=prefix.parent/"rss_alias";alias.mkdir(exist_ok=True)
    for ext in ("nodes","nets","wts","pl","scl"):
        path=alias/(alias_name+"."+ext)
        target=Path(paths[ext]).resolve()
        if not path.exists():path.symlink_to(target)
        if path.resolve()!=target:raise ValueError("RSS alias points to different input")
    aux=alias/(alias_name+".aux")
    aux.write_text("RowBasedPlacement : "+" ".join(alias_name+"."+s for s in ("nodes","nets","wts","pl","scl"))+"\n")
    return aux,paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.config:
        read_point(args.config.resolve(),args.out);return
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    from ioplace.bench.net_drop import drop_nets
    from ioplace.diagnostics.component_models import fit_model,predict
    import numpy as np
    recovery=REPO/"results/recovery_visible_20260906"
    group=recovery/"mempool_group_export/mempool_group"
    for fraction,label in ((.25,"groupDrop25"),(.5,"groupDrop50")):
        target=root/"inputs"/label/label
        if not Path(str(target)+".manifest.json").exists():
            print("build",label,flush=True);drop_nets(str(group),str(target),fraction,seed=0)
    dp=Path(os.environ["DREAMPLACE_ROOT"])
    prefixes=dict(adaptec1=dp/"benchmarks/ispd2005/adaptec1/adaptec1",
        bigblue4=dp/"benchmarks/ispd2005/bigblue4/bigblue4",group=group,
        groupDrop25=root/"inputs/groupDrop25/groupDrop25",groupDrop50=root/"inputs/groupDrop50/groupDrop50",
        array1x2=recovery/"arrays/1x2_n2/1x2_n2",array2x2=recovery/"arrays/2x2_n2/2x2_n2")
    configs={};input_hashes={}
    for name,prefix in prefixes.items():
        aux,paths=alias_prefix(prefix)
        cfg=root/(name+".config.json")
        write(cfg,dict(aux_input=str(aux),gpu=0,dtype="float64",sort_nets_by_degree=0,num_threads=8))
        configs[name]=cfg
        input_hashes[name]={path:sha(path) for path in paths.values()}
    manifest=root/"matrix.json"
    if not manifest.exists():
        write(manifest,dict(created=time.time(),input_sha256=input_hashes,source_sha256=sha(__file__),
            registration_sha256=sha(REPO/"docs/experiments/2026-09-06-component-model-preregistration.md"),
            train=list(prefixes)[:-1],holdout="array2x2",repetitions=3,timeout_s=3600))
    elif json.loads(manifest.read_text())["input_sha256"]!=input_hashes:
        raise ValueError("host matrix input changed")
    def measure(names):
        jobs=[(name,rep) for name in names for rep in range(3)]
        random.Random(20260906).shuffle(jobs)
        results={name:[] for name in names}
        for name,rep in jobs:
            out=root/f"{name}_rep{rep}.json";receipt=root/f"{name}_rep{rep}.execution.json"
            if receipt.exists():
                previous=json.loads(receipt.read_text())
                if previous.get("returncode")!=0 or not out.exists() or previous["output_sha256"]!=sha(out):
                    raise ValueError("preserve failed host attempt before rerun")
            else:
                cmd=[sys.executable,str(Path(__file__).resolve()),"--config",str(configs[name]),"--out",str(out)]
                print("read start",name,rep,flush=True)
                record=dict(command=cmd,started=time.time(),matrix_sha256=sha(manifest))
                with (root/f"{name}_rep{rep}.log").open("w") as log:
                    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
                    record["pid"]=process.pid;write(receipt,record)
                    try:code=process.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        process.kill();code=process.wait();record["timed_out"]=True
                record.update(returncode=code,ended=time.time(),output_sha256=sha(out) if out.exists() else None)
                write(receipt,record)
                if code:raise RuntimeError(f"host read failed: {name} repetition{rep}, exit{code}")
                print("read finish",name,rep,flush=True)
            results[name].append(json.loads(out.read_text()))
        return results
    training=measure(list(prefixes)[:-1])
    fit_path=root/"fit_frozen.json"
    if not fit_path.exists():
        features=[[1,values[0]["N"],values[0]["P_raw"]] for values in training.values()]
        fits={target:fit_model(features,[np.mean([v[target] for v in values]) for values in training.values()],
                names=["intercept","N","P_raw"],group_ids=list(training))
              for target in ("host_peak_rss_gib","read_s")}
        write(fit_path,dict(frozen_at=time.time(),matrix_sha256=sha(manifest),fits=fits,
            fitter_sha256=sha(Path(__file__).resolve().parents[1] / "ioplace/diagnostics/component_models.py")))
    fits=json.loads(fit_path.read_text())["fits"]
    holdout=measure(["array2x2"])["array2x2"]
    x=[1,holdout[0]["N"],holdout[0]["P_raw"]]
    checks={target:dict(**predict(model,x),actual=float(np.mean([v[target] for v in holdout])))
            for target,model in fits.items()}
    write(root/"holdout_check.json",dict(fit_sha256=sha(fit_path),checks=checks))


if __name__=="__main__":
    main()
