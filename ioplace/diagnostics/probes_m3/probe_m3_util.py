import sys, json
import numpy as np
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/.claude/worktrees/m2-bg")
from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
BASE = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/.claude/worktrees/m2-bg/results/m2"

def util(tag, npz, k, rtype, nl, placedb):
    d = np.load(npz); nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, k, rtype, 0); rg = RegionGrid(rs)
    nm = nl.num_movable
    cx = nx_[:nm] + nl.node_size_x[:nm] / 2.0
    cy = ny_[:nm] + nl.node_size_y[:nm] / 2.0
    rid = rg.region_of_points(cx, cy)
    area = nl.node_size_x[:nm] * nl.node_size_y[:nm]
    acc = np.zeros(k); np.add.at(acc, rid, area)
    rarea = np.array([sum((r[2]-r[0])*(r[3]-r[1]) for r in reg.rects) for reg in rs.regions])
    u = acc / rarea
    return dict(tag=tag, k=k, rtype=rtype, mean=float(u.mean()), min=float(u.min()),
                max=float(u.max()), max_over_mean=float(u.max()/u.mean()),
                util=[round(float(v),4) for v in u])

CFG = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/test/ispd2005/adaptec1.json"
params, placedb = _load_dreamplace(CFG); placedb.initialize(params)
nl = netlist_from_placedb(placedb)
out=[]
for t,f,k,r in [("A0_k16","ablation/adaptec1_A0_k16_grid",16,"grid"),
                ("A2_k16","ablation/adaptec1_A2_k16_grid",16,"grid"),
                ("A2_k32","ablation/adaptec1_A2_k32_grid",32,"grid"),
                ("A2_slic","ablation/adaptec1_A2_k16_slicing",16,"slicing")]:
    out.append(util(t, BASE+"/"+f+".json.npz", k, r, nl, placedb))
open("/tmp/probe_m3_util.json","w").write(json.dumps(out, indent=1))
