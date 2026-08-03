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

# NOTE on the benchmark choice below (see task-9-report.md for full detail):
#
# Brief Step 1 originally specified `install/test/simple.json` here. Simple
# (8 movable cells) runs GP+LG to completion without crashing, but with only
# ~1-3 cells per region the multi-fence-region solver cannot legalize
# cleanly: LG logs "out of fence region" errors and fence_compliance == 0.5,
# far below the 0.9 bar -- the "small case instability" the brief's own
# Step 4 anticipated.
#
# Per brief Step 4's own fallback ("若 simple ... 改用 adaptec1.json"), this
# test targets adaptec1 instead. adaptec1 hits a *different*, deeper issue:
# PlaceDB.initialize() unconditionally calls
# dreamplace.ops.fence_region.fence_region.slice_non_fence_region() for
# every fence region, which slices the die by every region/terminal box
# edge x-coordinate and intersects each slice with the non-fence polygon.
# adaptec1 has 480/543 terminals whose bbox extends outside the placeable
# core (padframe-style fixed macros, e.g. terminal 28 at x=9552,y=22 with
# size 72x432 against die (459,459,11151,11139)) -- entirely expected for a
# real macro-heavy design. A slice built from two such out-of-core
# x-coordinates never overlaps the die, so
# `non_fence_region.intersection(cvx_hull)` is topologically empty; under
# the installed shapely (2.1.2), that empty-result Polygon's `.bounds` is
# `(nan, nan, nan, nan)` (len 4) instead of `()`, so it slips past
# `slice_non_fence_region`'s `len(intersect.bounds) == 4` emptiness filter.
# The resulting NaN box feeds straight into the region's electric-potential
# density map, and `NonLinearPlace`'s very first objective/gradient
# evaluation crashes with `TypeError: fixed_density_map(): incompatible
# function arguments` (a NaN-valued 0-d tensor lands where a plain float is
# expected). Reproduced directly against 4 fence regions x 543 real
# terminals outside this driver; independent of k, rtype and of the
# fence_inject escape-valve workaround above -- it is a pre-existing
# DREAMPlace/shapely version incompatibility in
# dreamplace/ops/fence_region/fence_region.py::slice_non_fence_region
# (merge=True path), not something introduced by our injection. DREAMPlace
# source is off-limits (Global Constraints), so this cannot be fixed from
# our driver; xfail documents it and will flip to XPASS (a "please
# investigate" signal) if a DREAMPlace/shapely upgrade ever resolves it.
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DREAMPlace bug: slice_non_fence_region's merge=True path treats "
           "shapely's NaN-bounds empty-Polygon result (for a die slice built "
           "from an out-of-core terminal x-coordinate) as a real box, "
           "poisoning the region's density map with NaN and crashing "
           "NonLinearPlace's first obj/grad eval with a TypeError. Hit "
           "reliably on adaptec1 (480/543 terminals extend outside the core "
           "bbox); not fixable without editing DREAMPlace source. See "
           "task-9-report.md.",
    raises=TypeError, strict=False)
def test_two_stage_adaptec1(tmp_path):
    import os
    from ioplace.drivers.run_placement_two_stage import run_two_stage
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    res = run_two_stage(os.path.join(root, "install/test/ispd2005/adaptec1.json"),
                        4, "grid", 0, str(tmp_path / "s2.json"))
    assert res["mode"] == "two_stage" and res["io_count"] >= 0
    # fence placement 後,movable cells 幾何落點應與指派 region 一致(LG 後)
    assert res["fence_compliance"] >= 0.9
