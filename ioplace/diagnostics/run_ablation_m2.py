"""design v2 sec 8.1: M2 ablation matrix A0-A7 + scale/shape confirmation (T8).

20 arms, idempotent (existing files are skipped, so the run is resumable).
adaptec1 arms run first (~2 min each) so the diagnostics/report tooling has
data early; bigblue4 arms (~9-20 min each) run last. (rho*, tau*) come from
T7's gates.json best arm; the det regime and dp_seed come from T6.

    PYTHONPATH=. $PY -m ioplace.diagnostics.run_ablation_m2

Task 8 Step 2 (per-degree-bucket diagnostic, design v2 sec 3.2.1/8.2) lives in
this module too, as a second CLI stage that never touches GPU or runs a
placement -- it only re-reads each A2/A4/A6 run's own JSON (grad_share, GP
schedule fields) and its saved node_x/node_y .npz, then recomputes per-net
crossing/lambda data with the CPU evaluator_ref.evaluate:

    PYTHONPATH=. $PY -m ioplace.diagnostics.run_ablation_m2 --stage buckets

`--run case:arm:path` (repeatable) overrides the results/m2/ablation/*.json
directory scan with an explicit list, e.g. to smoke-test against
results/m2/sweep/*.json (whose filenames don't carry an arm token) before the
real ablation files exist.
"""
import json, os, re, shutil

import numpy as np

from ioplace.drivers.run_placement import run_flat, _load_dreamplace, get_regions_for
from ioplace.drivers.run_placement_reweight import run_reweight
from ioplace.drivers.run_placement_io import run_io
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate
from ioplace.ops.io_term import build_net_node_csr, DEG_BUCKET_LABELS

DPTEST = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/test/ispd2005"
CFGS = {"adaptec1": f"{DPTEST}/adaptec1.json", "bigblue4": f"{DPTEST}/bigblue4.json"}
NOISE = "results/m2/noise/summary.json"
GATES = "results/m2/sweep/gates.json"
OUT = "results/m2/ablation"
DP_SEED = 1000


def _best_params():
    gates = json.load(open(GATES))
    best = next(a for a in gates["arms"] if a["name"] == gates["best"])
    return best["rho_max"], best["tau_hi"], best["tau_lo"]


def main():
    os.makedirs(OUT, exist_ok=True)
    det = int(json.load(open(NOISE))["regime"])
    rho, hi, lo = _best_params()
    a2 = dict(rho_max=rho, tau_hi=hi, tau_lo=lo, every=50,
              dp_seed=DP_SEED, deterministic=det)

    def path(case, arm, k, rtype):
        return f"{OUT}/{case}_{arm}_k{k}_{rtype}.json"

    def flat(case, k, rtype):
        out = path(case, "A0", k, rtype)
        if not os.path.exists(out):
            run_flat(CFGS[case], k, rtype, 0, out, dp_seed=DP_SEED, deterministic=det)
            print("done:", out, flush=True)

    def reweight(case, k):
        out = path(case, "A1", k, "grid")
        if not os.path.exists(out):
            run_reweight(CFGS[case], k, "grid", 0, out, every=100, alpha=0.2,
                         dp_seed=DP_SEED, deterministic=det)
            print("done:", out, flush=True)

    def io(case, arm, k, rtype="grid", **extra):
        out = path(case, arm, k, rtype)
        if not os.path.exists(out):
            run_io(CFGS[case], k, rtype, 0, out, **{**a2, **extra})
            print("done:", out, flush=True)

    # A7: T7 already ran exactly this arm -- copy, don't burn 2 GPU minutes
    a7_src = "results/m2/sweep/adaptec1_k16_A7_margin.json"
    a7_dst = path("adaptec1", "A7", 16, "grid")
    if not os.path.exists(a7_dst):
        shutil.copy(a7_src, a7_dst)
        shutil.copy(a7_src + ".npz", a7_dst + ".npz")
        print("copied:", a7_dst, flush=True)

    # ---- adaptec1 (fast arms first) ----
    for k in (8, 16, 32):
        flat("adaptec1", k, "grid")
    flat("adaptec1", 16, "slicing")
    reweight("adaptec1", 16)
    for k in (8, 16, 32):
        io("adaptec1", "A2", k)
    io("adaptec1", "A2", 16, rtype="slicing")
    io("adaptec1", "A3", 16, alpha_io=0.2)
    io("adaptec1", "A4", 16, w_mode="inv_deg")
    io("adaptec1", "A5", 16, tau_hi=0.10, tau_lo=0.10)
    io("adaptec1", "A6", 16, ignore_net_degree=256)

    # ---- bigblue4 (scale confirmation) ----
    flat("bigblue4", 16, "grid")
    reweight("bigblue4", 16)
    io("bigblue4", "A2", 16)
    io("bigblue4", "A3", 16, alpha_io=0.2)
    io("bigblue4", "A4", 16, w_mode="inv_deg")
    io("bigblue4", "A6", 16, ignore_net_degree=256)
    print("ablation matrix complete", flush=True)


# ---------------------------------------------------------------------------
# Task 8 Step 2: per-degree-bucket diagnostic (design v2 sec 3.2.1 / 8.2)
# ---------------------------------------------------------------------------

_ABLATION_NAME_RE = re.compile(
    r"^(?P<case>[A-Za-z0-9]+)_(?P<arm>A\d+)_k(?P<k>\d+)_(?P<rtype>grid|slicing)\.json$")
BUCKET_ARMS = ("A2", "A4", "A6")
BUCKETS_OUT = f"{OUT}/degree_buckets.json"


def find_ablation_bucket_runs(ablation_dir=OUT):
    """Scan `ablation_dir` for A2/A4/A6 run JSONs named per run_ablation_m2's
    own `{case}_{arm}_k{K}_{rtype}.json` convention (see `path()` in `main`).
    Returns a sorted list of (case, arm, path) triples."""
    runs = []
    for fn in sorted(os.listdir(ablation_dir)):
        m = _ABLATION_NAME_RE.match(fn)
        if m and m.group("arm") in BUCKET_ARMS:
            runs.append((m.group("case"), m.group("arm"), os.path.join(ablation_dir, fn)))
    return runs


def _rebuild_nl_rg(run):
    """Reconstruct the (nl, rg) pair evaluator_ref.evaluate needs, the same
    way run_placement._evaluate_and_pack does: _load_dreamplace + initialize
    (CPU, no placement -- NonLinearPlace is never constructed) then the same
    get_regions_for(die, k, rtype, seed) region set the run itself used."""
    params, placedb = _load_dreamplace(run["config"])
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, run["k"], run["rtype"], run["seed"]))
    return nl, rg


def compute_degree_buckets_for_run(path, case, arm):
    """Per Task 8 Step 2 / design v2 sec 3.2.1: for one A2/A4/A6 run, combine
    its own trajectory[-1]["grad_share"] (already computed on-GPU during the
    run) with a CPU evaluator_ref.evaluate recompute of the saved final
    placement, bucketed by net degree exactly like IoTerm does (same
    build_net_node_csr call, same ignore_net_degree=d_max, same
    DEG_BUCKET_LABELS) so grad_share/io_share/lambda_share share one
    denominator population. Returns a list of 7 row dicts (one per bucket)."""
    run = json.load(open(path))
    grad_share = run["trajectory"][-1]["grad_share"]
    npz = np.load(path + ".npz")
    node_x, node_y = npz["node_x"], npz["node_y"]

    nl, rg = _rebuild_nl_rg(run)
    res = evaluate(nl, node_x, node_y, rg)
    csr = build_net_node_csr(nl, run["d_max"])

    crossings = res.per_net_crossings[csr.net_ids].astype(np.float64)
    lam_excess = np.maximum(res.per_net_lambda[csr.net_ids].astype(np.float64) - 1.0, 0.0)
    total_cross = float(crossings.sum())
    total_lam = float(lam_excess.sum())

    rows = []
    for b, label in enumerate(DEG_BUCKET_LABELS):
        mask = csr.deg_bucket == b
        io_share = float(crossings[mask].sum() / total_cross) if total_cross > 0 else 0.0
        lambda_share = float(lam_excess[mask].sum() / total_lam) if total_lam > 0 else 0.0
        g = float(grad_share[b])
        ratio = (g / io_share) if io_share != 0.0 else None
        rows.append({"case": case, "arm": arm, "bucket": label, "grad_share": g,
                     "io_share": io_share, "lambda_share": lambda_share, "ratio": ratio})
    return rows


def render_degree_buckets_markdown(rows):
    lines = ["| case | arm | bucket | grad_share | io_share | lambda_share | grad_share/io_share |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        ratio = "—" if r["ratio"] is None else f"{r['ratio']:.3f}"
        lines.append(f"| {r['case']} | {r['arm']} | {r['bucket']} | "
                     f"{r['grad_share']:.4f} | {r['io_share']:.4f} | "
                     f"{r['lambda_share']:.4f} | {ratio} |")
    return "\n".join(lines)


def run_degree_buckets_stage(runs, out_path=BUCKETS_OUT):
    """`runs` is a list of (case, arm, path) triples (see
    find_ablation_bucket_runs). Writes `out_path` as a flat list of row dicts
    (design v2 sec 3.2.1's A4 override condition is decided from this table
    at T9, not here) and prints the markdown table to stdout."""
    rows = []
    for case, arm, path in runs:
        rows.extend(compute_degree_buckets_for_run(path, case, arm))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    md = render_degree_buckets_markdown(rows)
    print(md, flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["matrix", "buckets"], default="matrix",
                    help="matrix (default): run the A0-A7 ablation matrix (Step 1). "
                         "buckets: Task 8 Step 2 per-degree-bucket diagnostic.")
    ap.add_argument("--ablation-dir", default=OUT,
                    help="[buckets] directory to scan for {case}_{arm}_k{K}_{rtype}.json "
                         "files; ignored if --run is given.")
    ap.add_argument("--run", action="append", default=[],
                    help="[buckets] explicit case:arm:path triple, repeatable; overrides "
                         "--ablation-dir. For smoke-testing against files that don't follow "
                         "the ablation naming convention, e.g. "
                         "adaptec1:A2:results/m2/sweep/adaptec1_k16_rho0.40_annealed.json")
    ap.add_argument("--out", default=BUCKETS_OUT, help="[buckets] output JSON path.")
    args = ap.parse_args()

    if args.stage == "matrix":
        main()
    else:
        if args.run:
            runs = [tuple(r.split(":", 2)) for r in args.run]
        else:
            runs = find_ablation_bucket_runs(args.ablation_dir)
        run_degree_buckets_stage(runs, args.out)
