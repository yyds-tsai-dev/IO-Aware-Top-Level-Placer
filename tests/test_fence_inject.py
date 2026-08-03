import numpy as np
import pytest
from types import SimpleNamespace
from ioplace.fence_inject import inject_fence_regions
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
