import sys, json
import numpy as np, torch
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/.claude/worktrees/m2-bg")
exec(open("/tmp/probe_m3_rg.py").read().split("params, placedb = _load_dreamplace")[0])

def lam_breakdown(tag, npz, k, rtype, nl, placedb, dev="cuda"):
    d = np.load(npz); nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, 0))
    ctx = GpuEvalContext(nl, rg, device=dev); res = ctx.evaluate(nx_, ny_)
    adj, D = region_graph(rg); K = rg.k
    bm = rg.pin_region_bitmask(nl, nx_, ny_)
    touched = ((bm[:, None].astype(np.uint64) >> np.arange(K, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    deg = nl.net_degrees; touched[deg < 2] = False; lam = touched.sum(1)
    from ioplace.netlist import pin_positions
    ppx, ppy = pin_positions(nl, nx_, ny_)
    pin_rid = rg.region_of_points(ppx, ppy).astype(np.int64)
    cnt = np.zeros((nl.num_nets, K), dtype=np.int32); np.add.at(cnt, (nl.pin2net, pin_rid), 1)
    home = cnt.argmax(1)
    Dt = torch.as_tensor(D, device=dev); steiner = np.zeros(nl.num_nets, dtype=np.int64)
    for t in range(2, K + 1):
        sel = np.nonzero(lam == t)[0]
        if len(sel) == 0: continue
        term = np.nonzero(touched[sel])[1].reshape(len(sel), t); tt = torch.as_tensor(term, device=dev)
        if t == 2: steiner[sel] = D[term[:, 0], term[:, 1]]
        elif t == 3: steiner[sel] = (Dt[:, tt[:,0]]+Dt[:, tt[:,1]]+Dt[:, tt[:,2]]).min(dim=0).values.cpu().numpy()
        else:
            sub = Dt[tt.unsqueeze(2), tt.unsqueeze(1)]
            steiner[sel] = batched_prim(sub).cpu().numpy().astype(np.int64)
    ft_rg = steiner - np.maximum(lam - 1, 0)
    Dh = D[home]; star_ft = np.where(touched, np.maximum(Dh-1,0), 0).sum(1)
    out = dict(tag=tag, k=k, rtype=rtype, io_mst=int(res.io_count), ft_mst=int(res.ft_count),
        hard_lambda_sum=int(res.hard_lambda_sum), io_rg=int(steiner.sum()), ft_rg=int(ft_rg.sum()),
        star_ft=int(star_ft.sum()), detour_mst=int(res.io_count-res.hard_lambda_sum))
    for lo, hi, nm in ((2,2,"lam2"),(3,3,"lam3"),(4,99,"lam4p")):
        m = (lam >= lo) & (lam <= hi)
        out["ft_rg_"+nm] = int(ft_rg[m].sum()); out["star_ft_"+nm] = int(star_ft[m].sum())
        out["n_"+nm] = int(m.sum())
    dem = res.boundary_pair_demand
    dpl = np.array(sorted(c/adj[a,b] for (a,b),c in dem.items() if adj[a,b] > 0))
    out["demand_per_len"] = dict(n=len(dpl), med=float(np.median(dpl)), max=float(dpl.max()),
                                 max_over_med=float(dpl.max()/np.median(dpl)))
    return out

BB = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/test/ispd2005/bigblue4.json"
res = []
params, placedb = _load_dreamplace(BB); placedb.initialize(params)
nl = netlist_from_placedb(placedb)
for tag, f in (("bb4_A0", "bigblue4_A0_k16_grid"), ("bb4_A2", "bigblue4_A2_k16_grid"),
               ("bb4_A1", "bigblue4_A1_k16_grid")):
    res.append(lam_breakdown(tag, BASE + "/ablation/" + f + ".json.npz", 16, "grid", nl, placedb))
open("/tmp/probe_m3_bb.json","w").write(json.dumps(res, indent=1))
