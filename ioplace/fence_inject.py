import numpy as np
from ioplace.region_grid import RegionGrid

INT32_MAX = np.iinfo(np.int32).max

def inject_fence_regions(placedb, rs, parts):
    """Fill placedb's four fence-region fields in place.

    Must be called after `placedb.read(params)` and before
    `placedb.initialize(params)` -- `initialize()` consumes `placedb.regions`
    / `node2fence_region_map` to build filler nodes and multi-region density
    ops, and DREAMPlace source is off-limits, so the only place to inject
    fence data is this window between read() and initialize().
    """
    assert not hasattr(placedb, "filler_start_map"), \
        "inject must happen after read() and BEFORE initialize()"
    n_assign = placedb.num_movable_nodes + placedb.num_terminals
    assert len(parts) == placedb.num_movable_nodes
    regions = [np.asarray(r.rects, dtype=placedb.dtype).reshape(-1, 4)
               for r in rs.regions]
    placedb.regions = regions
    placedb.flat_region_boxes = np.concatenate(regions, axis=0)
    counts = [len(r) for r in regions]
    placedb.flat_region_boxes_start = np.concatenate(
        [[0], np.cumsum(counts)]).astype(np.int32)
    rg = RegionGrid(rs)
    m = placedb.num_movable_nodes
    fence_map = np.full(n_assign, INT32_MAX, dtype=np.int32)
    fence_map[:m] = parts.astype(np.int32)
    term_sl = slice(m, n_assign)
    fence_map[m:] = rg.region_of_points(
        np.asarray(placedb.node_x[term_sl], dtype=np.float64),
        np.asarray(placedb.node_y[term_sl], dtype=np.float64)).astype(np.int32)
    placedb.node2fence_region_map = fence_map
