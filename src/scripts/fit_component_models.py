"""Aggregate fresh-process replicates and freeze component fits before holdouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ioplace.diagnostics.component_models import fit_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve frozen fit; use a new explicitly labeled output")
    manifest=json.loads((args.root/"matrix.json").read_text())
    groups, failures={},[]
    for recipe in manifest["recipes"]:
        name=recipe["recipe_id"]+f"_rep{recipe['repetition']}"
        output=args.root/"points"/(name+".json")
        execution=args.root/"points"/(name+".execution.json")
        receipt=json.loads(execution.read_text()) if execution.exists() else {}
        if receipt.get("returncode")!=0 or not output.exists():
            failures.append(dict(name=name,receipt=receipt));continue
        if receipt["output_sha256"]!=sha(output):
            raise ValueError(f"measurement digest mismatch {name}")
        value=json.loads(output.read_text())
        if value["status"]!="completed":
            failures.append(dict(name=name,status=value["status"]));continue
        groups.setdefault(recipe["recipe_id"],[]).append(value)
    fits={}
    for component in ("evaluator","io_term"):
        rows=[]
        for key,values in groups.items():
            if values[0]["recipe"]["component"]!=component:
                continue
            if len(values)!=3 or {v["recipe"]["repetition"] for v in values}!={0,1,2}:
                failures.append(dict(recipe=key,reason="incomplete replicate group"));continue
            first=values[0]
            if component=="evaluator":
                features=[1,first["N"],first["P"],first["E"],first["E"]*first["K"]]
                names=["intercept","N","P","E","E_K"]
            else:
                features=[1,first["Q"],first["P_dedup"],first["E_active"]]
                names=["intercept","Q","P_dedup","E_active"]
            rows.append(dict(id=key,features=features,values=values))
        if not rows:
            fits[component]=dict(status="not_evaluable",reason="no complete recipe groups");continue
        fits[component]={target:fit_model([r["features"] for r in rows],
                [float(np.mean([v[target] for v in r["values"]])) for r in rows],
                names=names,group_ids=[r["id"] for r in rows])
            for target in ("peak_allocated_gib","peak_reserved_gib","warm_mean_wall_s")}
    result=dict(frozen_at=time.time(),matrix_sha256=sha(args.root/"matrix.json"),
        status="complete_matrix" if not failures else "incomplete_matrix",
        failures=failures,fits=fits,fitter_source_sha256=sha(Path(__file__)),
        regression_source_sha256=sha(Path(__file__).resolve().parents[1]/"ioplace/diagnostics/component_models.py"),
        historical_gp_fit_verdict="unchanged",holdout_status="not_run")
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(args.out)


if __name__=="__main__":
    main()
