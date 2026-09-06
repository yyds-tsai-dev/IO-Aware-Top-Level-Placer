import numpy as np
from . import mtkahypar_runtime

def _hyperedges(nl):
    """Per-net, deduplicated, 0-indexed node id lists (installed mtkahypar==1.6.2
    Python API takes hyperedges as a nested list-of-lists, not a flat CSR pair)."""
    edges = []
    for net in range(nl.num_nets):
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        nodes = np.unique(nl.pin2node[nl.flat_net2pin[s:e]])
        edges.append([int(v) for v in nodes])
    return edges

def partition_netlist(nl, k, epsilon=0.03, seed=0, threads=8):
    import mtkahypar
    # print_warnings=False: mtkahypar.initialize() sets up a process-global TBB
    # thread pool; calling it again (e.g. Task 9 sweeping k in {8,16,32} within
    # one process) is harmless but otherwise prints "Mt-KaHyPar is already
    # initialized" to stderr on every call after the first.
    mtk = mtkahypar_runtime.initialize(mtkahypar, threads)
    ctx = mtk.context_from_preset(mtkahypar.PresetType.DEFAULT)
    ctx.set_partitioning_parameters(k, epsilon, mtkahypar.Objective.KM1)
    mtkahypar.set_seed(seed)
    edges = _hyperedges(nl)
    hg = mtk.create_hypergraph(ctx, nl.num_physical, nl.num_nets, edges)
    part = hg.partition(ctx)
    return np.array([part.block_id(v) for v in range(nl.num_physical)],
                    dtype=np.int32)
