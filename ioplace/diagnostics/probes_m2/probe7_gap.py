"""Probe 7: how much of io_count can the 'lambda-1' (region-presence) objective explain?
S2 minimises E[lambda-1]; the metric is MST boundary crossings. Measure the gap."""
import json, sys
sys.path.insert(0,"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer")
import numpy as np
from scipy.stats import spearmanr
from ioplace.netlist import load_netlist
from ioplace.drivers.run_placement import get_regions_for
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext

DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"

def run(case: str = "adaptec1", k: int = 16) -> dict:
    """Sweeps a fixed (case,K) list -- including the adaptec1 k=8 point the v1
    design-doc table omitted -- so `case`/`k` are accepted for signature
    uniformity with the other probes but unused here."""
    cases = []
    for CASE, K in (("adaptec1", 8), ("adaptec1", 16), ("adaptec1", 32), ("bigblue4", 16)):
        nl, placedb, params = load_netlist(f"{DP}/install/test/ispd2005/{CASE}.json")
        d = np.load(f"/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m1/{CASE}_reweight_k{K}_grid.json.npz")
        die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
        rg = RegionGrid(get_regions_for(die, K, "grid", 0))
        nx, ny = d["node_x"], d["node_y"]
        ctx = GpuEvalContext(nl, rg, device="cuda"); res = ctx.evaluate(nx, ny)
        bm = rg.pin_region_bitmask(nl, nx, ny)
        pc = np.array([bin(int(v)).count("1") for v in bm])
        lam1 = np.maximum(pc-1, 0)
        deg = nl.net_degrees
        m = deg >= 2
        lam1_sum = int(lam1[m].sum())

        # GP-end rank correlation (design v2 sec 3.2.5): only nets that actually cross.
        spear_mask = m & (res.per_net_crossings > 0)
        if spear_mask.sum() >= 2:
            rho, pval = spearmanr(res.per_net_crossings[spear_mask], lam1[spear_mask])
        else:
            rho, pval = float("nan"), float("nan")

        cases.append({
            "case": CASE, "k": K,
            "io_count": int(res.io_count),
            "sum_lambda_minus_1": lam1_sum,
            "ratio_io_over_lambda_minus_1": res.io_count / max(lam1_sum, 1),
            "nets_with_cross_gt_lam1": int(((res.per_net_crossings > lam1) & m).sum()),
            "n_nets_deg_ge2": int(m.sum()),
            "excess_crossings": int(np.maximum(res.per_net_crossings-lam1, 0)[m].sum()),
            "spearman_rho": float(rho), "spearman_pvalue": float(pval),
            "n_nets_spearman": int(spear_mask.sum()),
        })
    return {"cases": cases}

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
