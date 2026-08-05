import numpy as np
import pytest
from types import SimpleNamespace
from ioplace.fence_inject import inject_fence_regions
from ioplace.netlist import Netlist
from ioplace.regions import make_grid_regions

def _fake_placedb():
    return SimpleNamespace(
        dtype=np.float32,
        num_movable_nodes=3, num_terminals=1, num_terminal_NIs=0,
        node_x=np.array([10., 30., 50., 90.]), node_y=np.array([10., 10., 40., 90.]),
        xl=0., yl=0., xh=100., yh=100.)

def test_inject_fills_four_fields():
    db = _fake_placedb()
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    parts = np.array([0, 0, 1], dtype=np.int32)
    inject_fence_regions(db, rs, parts)
    assert len(db.regions) == 4
    assert db.flat_region_boxes.shape == (4, 4) and db.flat_region_boxes.ndim == 2
    assert db.flat_region_boxes.dtype == np.float32
    assert list(db.flat_region_boxes_start) == [0, 1, 2, 3, 4]
    assert db.node2fence_region_map.dtype == np.int32
    assert list(db.node2fence_region_map[:3]) == [0, 0, 1]
    assert db.node2fence_region_map[3] == 3     # terminal (90,90) -> P3 (geometric)

def test_inject_rejects_wrong_parts_length():
    db = _fake_placedb()
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    with pytest.raises(AssertionError):
        inject_fence_regions(db, rs, np.array([0, 1], dtype=np.int32))

def test_pick_escape_cell_avoids_singletons():
    from ioplace.drivers.run_placement_two_stage import _pick_escape_cell
    # 5 movable cells, k=3 blocks; block 2 is a singleton (cell 4 only).
    parts = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    # area = [2, 1, 3, 4, 0.5] -- cell 4 (the singleton block) is the
    # smallest-area cell overall, so a correct implementation must skip it
    # and instead pick the smallest-area cell among the non-singleton pool
    # (cell 1, area=1), not the global minimum.
    node_size_x = np.array([2., 1., 3., 4., 0.5])
    node_size_y = np.array([1., 1., 1., 1., 1.])
    node2fence_region_map = np.full(5, -1, dtype=np.int32)  # unused by helper
    idx = _pick_escape_cell(node2fence_region_map, parts, node_size_x, node_size_y, 3)
    assert 0 <= idx < len(parts)
    assert parts[idx] != 2      # not a member of the singleton block
    assert idx == 1             # min-area cell among the non-singleton pool

def test_pick_escape_cell_all_singletons_fallback():
    from ioplace.drivers.run_placement_two_stage import _pick_escape_cell
    # 3 movable cells, k=3 blocks, every block has exactly 1 cell -- no
    # non-singleton pool exists, so the helper must fall back to the
    # smallest-area cell over the entire population.
    parts = np.array([0, 1, 2], dtype=np.int32)
    node_size_x = np.array([3., 1., 2.])
    node_size_y = np.array([1., 1., 1.])
    node2fence_region_map = np.full(3, -1, dtype=np.int32)  # unused by helper
    idx = _pick_escape_cell(node2fence_region_map, parts, node_size_x, node_size_y, 3)
    assert 0 <= idx < len(parts)
    assert idx == 1              # global min-area fallback

def _netlist_from_nets(nets, num_movable):
    """Build a minimal Netlist from a list of nets (each a list of node ids);
    no terminals, unit pin offsets, positions unused by the block<->region
    assignment (it only reads connectivity + region geometry)."""
    pins, p2n = [], []
    for i, ns in enumerate(nets):
        pins += ns
        p2n += [i] * len(ns)
    start = np.concatenate([[0], np.cumsum([len(n) for n in nets])]).astype(np.int32)
    return Netlist(
        node_x=np.zeros(num_movable), node_y=np.zeros(num_movable),
        node_size_x=np.ones(num_movable), node_size_y=np.ones(num_movable),
        num_movable=num_movable, num_terminals=0, num_terminal_NIs=0,
        pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
        pin2node=np.array(pins, dtype=np.int32),
        pin2net=np.array(p2n, dtype=np.int32),
        flat_net2pin=np.arange(len(pins), dtype=np.int32),
        flat_net2pin_start=start,
        xl=0., yl=0., xh=100., yh=100.)

def test_assign_blocks_prefers_adjacent_regions():
    # M0 finding: mtkahypar block ids carry no geometric meaning, so a
    # naive block-id == region-id injection can throw two heavily-connected
    # blocks onto opposite die corners. assign_blocks_to_regions must instead
    # place heavily-connected blocks in geometrically adjacent regions.
    from ioplace.drivers.run_placement_two_stage import assign_blocks_to_regions
    # 4 movable cells, 1 per block (parts[i] == block i). Heavy connectivity
    # 0<->1 (5 nets) and 2<->3 (5 nets); zero connectivity for every other
    # block pair (0-2, 0-3, 1-2, 1-3).
    nets = [[0, 1]] * 5 + [[2, 3]] * 5
    nl = _netlist_from_nets(nets, num_movable=4)
    parts = np.array([0, 1, 2, 3], dtype=np.int32)
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    # make_grid_regions row-major layout over a 2x2 grid of the die above ->
    # region centers P0=(25,25) P1=(75,25) P2=(25,75) P3=(75,75). Minimum
    # Manhattan distance between any two distinct regions is 50 (the 4
    # edge-adjacent pairs); the 2 diagonal pairs (P0,P3)/(P1,P2) are 100.
    centers = {0: (25., 25.), 1: (75., 25.), 2: (25., 75.), 3: (75., 75.)}

    def dist(r1, r2):
        (x1, y1), (x2, y2) = centers[r1], centers[r2]
        return abs(x1 - x2) + abs(y1 - y2)

    new_parts = assign_blocks_to_regions(parts, nl, rs)
    assert len(new_parts) == 4
    assert set(new_parts.tolist()) == {0, 1, 2, 3}   # still a bijection
    min_dist = min(dist(a, b) for a in range(4) for b in range(4) if a != b)
    # block 0's and block 1's assigned regions must be geometrically
    # adjacent (minimal possible center distance) -- likewise 2 and 3.
    assert dist(int(new_parts[0]), int(new_parts[1])) == min_dist
    assert dist(int(new_parts[2]), int(new_parts[3])) == min_dist

def test_assign_blocks_deterministic():
    from ioplace.drivers.run_placement_two_stage import assign_blocks_to_regions
    # Asymmetric connectivity (no tie symmetry to hide nondeterminism behind):
    # 0<->1 heavy, 1<->2 medium, 2<->3 light, 0<->3 absent.
    nets = [[0, 1]] * 6 + [[1, 2]] * 3 + [[2, 3]] * 1
    nl = _netlist_from_nets(nets, num_movable=4)
    parts = np.array([0, 1, 2, 3], dtype=np.int32)
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    r1 = assign_blocks_to_regions(parts, nl, rs)
    r2 = assign_blocks_to_regions(parts, nl, rs)
    assert np.array_equal(r1, r2)

# NOTE on the benchmark choice below (see task-9-report.md for full detail):
#
# Brief Step 1 originally specified `install/test/simple.json` here. Simple
# (8 movable cells) is too small/size-homogeneous for the multi-fence-region
# solver to legalize reliably -- fence_compliance was observed at 0.5-0.875
# across runs, and DREAMPlace's filler-sizing heuristic can even crash
# initialize() non-deterministically on it -- the "small case instability"
# brief Step 4 anticipated. Per brief Step 4's own fallback ("若 simple ...
# 改用 adaptec1.json"), this test targets adaptec1 instead; `simple` is not
# used as an integration case (known, out-of-scope small-case instability).
#
# adaptec1 previously hit a separate DREAMPlace bug in
# dreamplace.ops.fence_region.fence_region.slice_non_fence_region(): 480/543
# terminals have a bbox edge outside the core die bbox (padframe-style fixed
# macros), producing a die slice that never overlaps the die, so shapely's
# intersection is topologically empty -- but under shapely 2.1.2 an
# empty-result Polygon's `.bounds` is `(nan, nan, nan, nan)` (len 4) instead
# of `()`, slipping past the `len(intersect.bounds) == 4` emptiness filter
# and poisoning the region's density map with NaN, crashing NonLinearPlace's
# first obj/grad eval. Fixed upstream via a minimal, coordinator-authorized
# compatibility patch on the `io-aware` branch of $DREAMPLACE_ROOT (see
# ioplace/dp_patch/shapely2-compat.patch and task-9-report.md) -- this test
# now runs the real assertions, not xfail.
@pytest.mark.slow
def test_two_stage_adaptec1(tmp_path):
    import os
    from ioplace.drivers.run_placement_two_stage import run_two_stage
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    res = run_two_stage(os.path.join(root, "install/test/ispd2005/adaptec1.json"),
                        4, "grid", 0, str(tmp_path / "s2.json"))
    assert res["mode"] == "two_stage" and res["io_count"] >= 0
    # fence placement 後,movable cells 幾何落點應與指派 region 一致(LG 後)
    assert res["fence_compliance"] >= 0.9
