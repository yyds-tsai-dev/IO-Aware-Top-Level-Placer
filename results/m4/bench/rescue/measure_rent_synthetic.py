import json
import time

import numpy as np

from ioplace.bench import rent

CASES = {
    "1x1_source": "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m4/bench/mempool_group_export/mempool_group",
    "1x2_n1": "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m4/bench/arrays/1x2_n1/1x2_n1",
    "1x2_n2": "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m4/bench/arrays/1x2_n2/1x2_n2",
    "2x2_n1": "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m4/bench/arrays/2x2_n1/2x2_n1",
    "2x2_n2": "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m4/bench/arrays/2x2_n2/2x2_n2",
}

results = {}
try:
    with open("/tmp/rent_synthetic_result.json") as f:
        results = json.load(f)
    print(f"resuming, already have: {list(results.keys())}", flush=True)
except FileNotFoundError:
    pass

for name, prefix in CASES.items():
    if name in results:
        print(f"[{name}] already done, skipping", flush=True)
        continue
    t0 = time.time()
    nl = rent.load_bookshelf_netlist(prefix)
    t_load = time.time() - t0
    print(f"[{name}] load: {t_load:.1f}s num_physical={nl.num_physical} num_nets={nl.num_nets}",
          flush=True)

    res = rent.measure_rent(nl, b_lo=1e3, b_hi=1e6, seed=0, backend="mtkahypar", threads=16,
                             n_bootstrap=200, max_net_degree=100)
    t_total = time.time() - t0
    print(f"[{name}] p={res.p:.4f} ci=({res.p_ci_lo:.4f},{res.p_ci_hi:.4f}) "
          f"backend={res.backend} levels_used={res.levels_used} total_s={t_total:.1f}", flush=True)

    results[name] = {
        "p": res.p, "p_ci_lo": res.p_ci_lo, "p_ci_hi": res.p_ci_hi, "log_t": res.log_t,
        "backend": res.backend, "levels_used": res.levels_used, "n_bootstrap": res.n_bootstrap,
        "levels": [{"level": rl.level, "n_blocks": rl.n_blocks, "avg_block_size": rl.avg_block_size,
                    "avg_terminals": rl.avg_terminals} for rl in res.levels],
        "load_s": t_load, "total_s": t_total,
        "num_physical": int(nl.num_physical), "num_nets": int(nl.num_nets),
        "net_degrees_max": int(nl.net_degrees.max()),
    }
    np.save(f"/tmp/rent_synth_{name}_net_degrees.npy", nl.net_degrees.astype(np.int32))
    with open("/tmp/rent_synthetic_result.json", "w") as f:
        json.dump(results, f, indent=1)

print("DONE", flush=True)
