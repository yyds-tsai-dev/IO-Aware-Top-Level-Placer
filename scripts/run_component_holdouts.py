"""Run the36 registered real-cache holdouts against a frozen synthetic fit."""
import hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/component_models_20260906"
OUT=BASE/"holdouts"
SNAPSHOT=BASE/"source"
RUN=SNAPSHOT/"scripts/run_component_models.py"
FIT=BASE/"fits_frozen.json"
CACHE=ROOT/"results/recovery_visible_20260906/cache"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def write(path,data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n");temp.replace(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[]
    frozen=json.loads((BASE/"matrix.json").read_text())
    for path,digest in frozen["source_sha256"].items():
        if sha(SNAPSHOT/path)!=digest:raise ValueError("frozen source changed")
    for shape in ("1x2_n2","2x2_n2"):
        gate=CACHE/shape/"verification.json"
        while not gate.exists():time.sleep(10)
        v=json.loads(gate.read_text())["verify"]
        if not v["ok"] or v["mode"]!="full":raise ValueError("cache not fully verified")
        for K in (16,32):
            for component,budget in (("evaluator",None),("io_term",4000000),("io_term",16000000)):
                for rep in range(3):
                    cases.append(dict(component=component,K=K,chunk_budget=budget,seed=0,
                        layout="real_tiled",cache=str(CACHE/shape),
                        recipe_id=f"{shape}_{component}_K{K}_C{budget}",repetition=rep))
    random.Random(20260906).shuffle(cases)
    assert len(cases)==36
    for i,recipe in enumerate(cases):
        name=recipe["recipe_id"]+f"_rep{recipe['repetition']}"
        output,point,receipt,log=(OUT/(name+s) for s in (".json",".recipe.json",".execution.json",".log"))
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get("returncode")==0 and old["fits_sha256"]==sha(FIT) and output.exists() and old["output_sha256"]==sha(output):
                print("verified skip",name,flush=True);continue
            print("preserve failed holdout",name,flush=True);continue
        write(point,recipe)
        cmd=[sys.executable,str(RUN),"--point",str(point),"--out",str(output)]
        record=dict(recipe=recipe,recipe_sha256=sha(point),fits_sha256=sha(FIT),started=time.time(),command=cmd)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="3",PYTHONPATH=str(SNAPSHOT),PYTHONDONTWRITEBYTECODE="1")
        with log.open("w") as stream:
            process=subprocess.Popen(cmd,cwd=SNAPSHOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record["pid"]=process.pid;write(receipt,record)
            print("start",i+1,"/36",name,process.pid,flush=True)
            try:code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill();code=process.wait();record["timed_out"]=True
        record.update(returncode=code,ended=time.time(),log_sha256=sha(log),output_sha256=sha(output) if output.exists() else None)
        write(receipt,record)
        print("finish",name,code,flush=True)

if __name__=="__main__":main()
