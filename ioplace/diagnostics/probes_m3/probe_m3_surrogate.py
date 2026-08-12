import sys, json
import numpy as np, torch, scipy.stats as st
sys.path.insert(0, "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/.claude/worktrees/m2-bg")
exec(open("/tmp/probe_m3_rg.py").read().split("params, placedb = _load_dreamplace")[0])

def cmp_surrogates(tag, npz, k, rtype, nl, placedb, dev="cuda"):
    d = np.load(npz); nx_, ny_ = d["node_x"], d["node_y"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, 0)); K = rg.k
    ctx = GpuEvalContext(nl, rg, device=dev); res = ctx.evaluate(nx_, ny_)
    adj, D = region_graph(rg)
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
        if t == 2: steiner[sel] = D[term[:,0], term[:,1]]
        elif t == 3: steiner[sel] = (Dt[:,tt[:,0]]+Dt[:,tt[:,1]]+Dt[:,tt[:,2]]).min(dim=0).values.cpu().numpy()
        else:
            sub = Dt[tt.unsqueeze(2), tt.unsqueeze(1)]
            steiner[sel] = batched_prim(sub).cpu().numpy().astype(np.int64)
    ft_rg = steiner - np.maximum(lam - 1, 0)
    Dall = D[None,:,:]                    # (1,K,K)
    T = touched[:,None,:]                 # (E,1,K)
    Dm = np.where(T, Dall, 0)             # (E,K,K) -- too big; do per-root loop instead
    del Dm
    sum_all = np.zeros((len(lam), K), dtype=np.int64)
    ecc_all = np.zeros((len(lam), K), dtype=np.int64)
    for h in range(K):
        Dh = D[h][None, :]
        sum_all[:, h] = np.where(touched, np.maximum(Dh-1, 0), 0).sum(1)
        ecc_all[:, h] = np.where(touched, Dh, 0).max(1)
    lm1 = np.maximum(lam - 1, 0)
    S = dict(
        s4a_sum_home = sum_all[np.arange(len(lam)), home],
        s4b_ecc_home = np.maximum(ecc_all[np.arange(len(lam)), home] - lm1, 0),
        s4c_sum_best = sum_all.min(1),
        s4d_ecc_best = np.maximum(ecc_all.min(1) - lm1, 0),
    )
    out = dict(tag=tag, k=k, rtype=rtype, ft_rg=int(ft_rg.sum()), ft_mst=int(res.ft_count))
    m = deg >= 2
    for nm, v in S.items():
        out[nm] = int(v.sum()); out[nm+"_ratio"] = round(float(v.sum()/max(ft_rg.sum(),1)), 3)
        mm = m & ((v + ft_rg) > 0)
        out[nm+"_rho"] = round(float(st.spearmanr(ft_rg[mm], v[mm]).correlation), 3)
        # share of surrogate mass vs share of true ft, by lambda bucket
        sh = {}
        for lo, hi, b in ((2,2,"L2"),(3,3,"L3"),(4,99,"L4p")):
            bm2 = (lam>=lo)&(lam<=hi)
            sv = v[bm2].sum()/max(v.sum(),1); sf = ft_rg[bm2].sum()/max(ft_rg.sum(),1)
            sh[b] = (round(float(sv),3), round(float(sf),3), round(float(sv/max(sf,1e-9)),2))
        out[nm+"_share_vs_true"] = sh
    return out

CFG = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace/install/test/ispd2005/adaptec1.json"
params, placedb = _load_dreamplace(CFG); placedb.initialize(params)
nl = netlist_from_placedb(placedb)
runs = [("A0_k16", "ablation/adaptec1_A0_k16_grid", 16, "grid"),
        ("A2_k16", "ablation/adaptec1_A2_k16_grid", 16, "grid"),
        ("A2_k32", "ablation/adaptec1_A2_k32_grid", 32, "grid"),
        ("A2_slic","ablation/adaptec1_A2_k16_slicing", 16, "slicing")]
out = [cmp_surrogates(t, BASE+"/"+f+".json.npz", k, r, nl, placedb) for (t,f,k,r) in runs]
open("/tmp/probe_m3_surrogate.json","w").write(json.dumps(out, indent=1))
