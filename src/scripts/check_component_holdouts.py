"""Check real-cache outcomes against the earlier frozen component fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ioplace.diagnostics.component_models import predict


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root
    fit_path=root/"fits_frozen.json";frozen=json.loads(fit_path.read_text())
    groups={};failures=[]
    for path in sorted((root/"holdouts").glob("*.execution.json")):
        receipt=json.loads(path.read_text())
        out=path.with_name(path.name.replace(".execution.json",".json"))
        if receipt["fits_sha256"]!=sha(fit_path):raise ValueError("holdout changed frozen fit")
        if receipt.get("returncode")!=0 or not out.exists():
            failures.append(dict(path=str(path),receipt=receipt));continue
        if sha(out)!=receipt["output_sha256"]:raise ValueError("holdout artifact hash mismatch")
        if receipt["started"]<=frozen["frozen_at"]:raise ValueError("holdout preceded fit freeze")
        value=json.loads(out.read_text())
        groups.setdefault(value["recipe"]["recipe_id"],[]).append(value)
    rows=[]
    for recipe,values in groups.items():
        if len(values)!=3 or {v["recipe"]["repetition"] for v in values}!={0,1,2}:
            failures.append(dict(recipe=recipe,reason="incomplete repetition group"));continue
        first=values[0];component=first["recipe"]["component"]
        features=([1,first["N"],first["P"],first["E"],first["E"]*first["K"]] if component=="evaluator"
                  else [1,first["Q"],first["P_dedup"],first["E_active"]])
        for target,model in frozen["fits"][component].items():
            prediction=predict(model,features)
            actual=float(np.mean([v[target] for v in values]))
            interval=prediction["pi95"]
            rows.append(dict(recipe=recipe,component=component,target=target,actual=actual,
                std_process=float(np.std([v[target] for v in values],ddof=1)),**prediction,
                point_nonnegative=prediction["point"]>=0,
                relative_error=(prediction["point"]-actual)/actual if actual else None,
                in_frozen_pi=interval[0]<=actual<=interval[1] if interval else None,
                interpretation="unvalidated extrapolation; identification gate failed" if not model["identifiable"] else
                    "external holdout validation of an identified conditional model"))
    if len(groups)!=12:
        failures.append(dict(reason="expected12 complete holdout recipe groups",actual=len(groups)))
    result=dict(fits_sha256=sha(fit_path),n_recipes=len(groups),n_successful_processes=sum(map(len,groups.values())),
        failures=failures,rows=rows,extrapolation_allowed=False,
        reason="all original primary component fits failed the preregistered coefficient interval gate")
    (root/"holdout_check.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("n_recipes","n_successful_processes","failures")},indent=2))


if __name__=="__main__":main()
