"""Fresh-process component model matrix; see the frozen September6 protocol."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def run_point(recipe, output):
    import numpy as np
    import torch
    from ioplace.bench.spike_30m import device_snapshot
    from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic
    from ioplace.drivers.run_placement import get_regions_for
    from ioplace.region_grid import RegionGrid
    from ioplace.ops.soft_assign import rect_table
    from ioplace.ops.io_term import build_net_node_csr, IoTerm
    from ioplace.evaluator_gpu import GpuEvalContext
    torch.set_num_threads(8)
    record = dict(recipe=recipe, status="running", pid=os.getpid(), started=time.time(),
                  gpu_before=device_snapshot(), dtype="float64", benchmark_kind="synthetic",
                  generator_verified=False, torch_version=torch.__version__, cuda_version=torch.version.cuda)
    write(output, record)
    try:
        if "cache" in recipe:
            from ioplace.bench.bookshelf_netlist import load_tiled_netlist
            nl, meta = load_tiled_netlist(recipe["cache"], mmap=True)
            record["input_manifest_sha256"] = sha(meta["manifest_path"])
            record["raw_pins"] = meta["n_pins_raw"]
        else:
            nl = exact_synthetic(recipe["N"],recipe["E"],recipe["P"],seed=recipe["seed"],layout=recipe["layout"])
            record["raw_pins"] = recipe["P"]
        N, E, P, K = nl.num_physical, nl.num_nets, len(nl.pin2node), recipe["K"]
        record.update(N=N, E=E, P=P, K=K, num_movable=nl.num_movable)
        die = (nl.xl,nl.yl,nl.xh,nl.yh)
        regions = get_regions_for(die,K,"grid",0)
        rg = RegionGrid(regions)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if recipe["component"] == "evaluator":
            component = GpuEvalContext(nl,rg,device="cuda")
            x = torch.as_tensor(np.array(nl.node_x),dtype=torch.float64,device="cuda")
            y = torch.as_tensor(np.array(nl.node_y),dtype=torch.float64,device="cuda")
            record.update(mst_chunk_budget=8000000,seg_chunk_budget=8000000,edge_batch_size=1000000,max_degree=256)
        else:
            csr = build_net_node_csr(nl,100)
            rects,r2k = rect_table(regions)
            component = IoTerm(csr,rects,r2k,K,nl.num_movable,N,N,device="cuda",
                               chunk_budget=recipe["chunk_budget"])
            pos = torch.as_tensor(np.r_[nl.node_x,nl.node_y],dtype=torch.float64,device="cuda").requires_grad_(True)
            record.update(P_dedup=component._n_pins_dedup,E_active=component.n_active,c=component.k_chunk,
                          Q=component.k_chunk*max(N,component._n_pins_dedup))
        torch.cuda.synchronize()
        record["construction_s"] = time.perf_counter()-start
        record["construction_peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        record["iterations"] = []
        for iteration in range(3):
            torch.cuda.synchronize()
            wall = time.perf_counter()
            begin,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            if recipe["component"] == "evaluator":
                with torch.no_grad():
                    value = component.evaluate(x,y)
                end.record(); torch.cuda.synchronize()
                detail = dict(io_count=value.io_count,ft_count=value.ft_count,hpwl=value.hpwl)
            else:
                tau = .1*((die[2]-die[0])*(die[3]-die[1])/K)**.5
                loss = component(pos,tau,1.)
                forward_done = torch.cuda.Event(enable_timing=True)
                forward_done.record()
                torch.cuda.synchronize()
                forward_wall = time.perf_counter()-wall
                backward_start = time.perf_counter()
                loss.backward()
                end.record(); torch.cuda.synchronize()
                detail = dict(loss=float(loss.detach()),forward_wall_s=forward_wall,
                              backward_wall_s=time.perf_counter()-backward_start,
                              forward_event_ms=begin.elapsed_time(forward_done),
                              backward_event_ms=forward_done.elapsed_time(end))
                pos.grad = None
            row = dict(iteration=iteration,wall_s=time.perf_counter()-wall,
                       cuda_event_ms=begin.elapsed_time(end),**detail)
            record["iterations"].append(row)
            record.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            write(output,record)
        record.update(status="completed",gpu_after=device_snapshot(),ended=time.time(),
                      warm_mean_wall_s=float(np.mean([r["wall_s"] for r in record["iterations"][1:]])),
                      cuda_event_scope="elapsed CUDA stream range, may include CPU scheduling gaps")
    except Exception as error:
        record.update(status="oom" if "out of memory" in str(error).lower() else "failed",error=repr(error))
        raise
    finally:
        write(output,record)


def matrix():
    recipes = []
    for N,E,P,K in itertools.product((500000,2000000),(250000,1000000),(2000000,8000000),(8,32)):
        for component,budget in [("evaluator",None),("io_term",4000000),("io_term",16000000)]:
            base = dict(N=N,E=E,P=P,K=K,component=component,chunk_budget=budget,seed=0,layout="local")
            recipe_id = f"{component}_N{N}_E{E}_P{P}_K{K}_C{budget}"
            for repetition in range(3):
                recipes.append(dict(**base,recipe_id=recipe_id,repetition=repetition))
    random.Random(20260906).shuffle(recipes)
    return recipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--point",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    if args.point:
        run_point(json.loads(args.point.read_text()),args.out)
        return
    if not args.root:
        parser.error("--root required for matrix")
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/"matrix.json"
    if not manifest_path.exists():
        snapshot=root/"source"
        shutil.copytree(Path(__file__).resolve().parents[1] / "ioplace",snapshot/"ioplace",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        (snapshot/"scripts").mkdir()
        shutil.copy2(__file__,snapshot/"scripts/run_component_models.py")
        hashes={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
        write(manifest_path,dict(created=time.time(),source_sha256=hashes,recipes=matrix(),
              registration_sha256=sha(REPO/"docs/experiments/2026-09-06-component-model-preregistration.md")))
    manifest=json.loads(manifest_path.read_text());snapshot=root/"source"
    for path,digest in manifest["source_sha256"].items():
        if sha(snapshot/path)!=digest:
            raise ValueError("matrix frozen source changed")
    env=dict(os.environ,PYTHONPATH=str(snapshot),PYTHONDONTWRITEBYTECODE="1")
    jobs=root/"points";jobs.mkdir(exist_ok=True)
    recipes=manifest["recipes"][:args.limit] if args.limit else manifest["recipes"]
    for recipe in recipes:
        name=recipe["recipe_id"]+f"_rep{recipe['repetition']}"
        point,output,receipt=(jobs/(name+suffix) for suffix in (".recipe.json",".json",".execution.json"))
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get("returncode")==0 and output.exists() and old.get("output_sha256")==sha(output):
                print("verified skip",name,flush=True);continue
            print("preserved failed recipe",name,flush=True);continue
        write(point,recipe)
        command=[sys.executable,str(snapshot/"scripts/run_component_models.py"),"--point",str(point),"--out",str(output)]
        log=jobs/(name+".log")
        execution=dict(command=command,started=time.time(),matrix_sha256=sha(manifest_path))
        print("start",name,flush=True)
        with log.open("w") as stream:
            process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT)
            execution["pid"]=process.pid;write(receipt,execution)
            try:
                code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill();code=process.wait();execution["timed_out"]=True
        execution.update(returncode=code,ended=time.time(),log_sha256=sha(log),
                         output_sha256=sha(output) if output.exists() else None)
        write(receipt,execution)
        print("finish",name,code,flush=True)


if __name__=="__main__":
    main()
