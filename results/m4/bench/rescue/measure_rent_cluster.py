import json
import resource
import time

from ioplace.bench import rent

t0 = time.time()
nl, meta = rent.load_def_netlist(
    "/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/visible/mempool_cluster/mempool_cluster.def")
t_load = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"[cluster] load: {t_load:.1f}s meta={meta} rss_hwm={rss:.2f}GB", flush=True)

res = rent.measure_rent(nl, b_lo=1e3, b_hi=1e6, seed=0, backend="mtkahypar", threads=16,
                         n_bootstrap=200, max_net_degree=100)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"[cluster] p={res.p:.4f} ci=({res.p_ci_lo:.4f},{res.p_ci_hi:.4f}) "
      f"backend={res.backend} levels_used={res.levels_used} rss_hwm={rss:.2f}GB "
      f"total_s={time.time()-t0:.1f}", flush=True)

out = {
    "p": res.p, "p_ci_lo": res.p_ci_lo, "p_ci_hi": res.p_ci_hi, "log_t": res.log_t,
    "backend": res.backend, "levels_used": res.levels_used, "n_bootstrap": res.n_bootstrap,
    "levels": [{"level": rl.level, "n_blocks": rl.n_blocks, "avg_block_size": rl.avg_block_size,
                "avg_terminals": rl.avg_terminals} for rl in res.levels],
    "meta": meta, "load_s": t_load, "total_s": time.time() - t0,
    "net_degrees_max": int(nl.net_degrees.max()), "num_nets": int(nl.num_nets),
    "num_physical": int(nl.num_physical),
}
with open("/tmp/rent_cluster_result.json", "w") as f:
    json.dump(out, f, indent=1)

import numpy as np
np.save("/tmp/rent_cluster_net_degrees.npy", nl.net_degrees.astype(np.int32))
print("DONE", flush=True)
