import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.freeze import (FreezeMonitor, argmax_region, cell_centers,
                            ensure_nonempty_regions, freeze_record,
                            region_cell_stats)
from ioplace.init_pos import region_centers
from ioplace.ops.soft_assign import rect_table
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def _grid():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    rects, r2k = rect_table(rs)
    return rs, torch.as_tensor(rects), torch.as_tensor(r2k)


def test_argmax_region_matches_region_of_points_at_cell_centres():
    rs, rects, r2k = _grid()
    x = torch.tensor([10., 60., 10., 60.], dtype=torch.float64)
    y = torch.tensor([10., 10., 60., 60.], dtype=torch.float64)
    got = argmax_region(x, y, rects, r2k, rs.k)
    assert got.tolist() == RegionGrid(rs).region_of_points(x.numpy(), y.numpy()).tolist()
    assert argmax_region(x, y, rects, r2k, rs.k, chunk=1).tolist() == got.tolist()


def test_cell_centre_anchor_differs_from_lower_left_for_a_straddling_cell():
    rs, rects, r2k = _grid()
    x = torch.tensor([40.], dtype=torch.float64)       # lower-left in region 0
    y = torch.tensor([10.], dtype=torch.float64)
    size_x = torch.tensor([30.], dtype=torch.float64)  # centre at x=55 -> region 1
    size_y = torch.tensor([2.], dtype=torch.float64)
    assert argmax_region(x, y, rects, r2k, rs.k).tolist() == [0]
    cx, cy = cell_centers(x, y, size_x, size_y)
    assert argmax_region(cx, cy, rects, r2k, rs.k).tolist() == [1]


def test_monitor_churn_uses_the_sample_one_window_back():
    monitor = FreezeMonitor(window=50)
    base = np.array([0, 1, 2, 3], dtype=np.int64)
    assert monitor.observe(0, base) is None            # nothing to compare to yet
    assert monitor.observe(50, base.copy()) == 0.0
    changed = base.copy(); changed[0] = 3
    assert monitor.observe(100, changed) == 0.25
    assert monitor.churn == 0.25 and monitor.last_iteration == 100


def test_criterion_needs_all_three_conditions():
    monitor = FreezeMonitor(window=50, overflow_max=.15, tau_rel_max=.05,
                            churn_max=.005)
    base = np.zeros(1000, dtype=np.int64)
    monitor.observe(0, base)
    monitor.observe(50, base.copy())                   # churn 0.0
    assert monitor.should_freeze(overflow=.10, tau_rel=.04)
    assert not monitor.should_freeze(overflow=.16, tau_rel=.04)
    assert not monitor.should_freeze(overflow=.10, tau_rel=.06)
    churned = base.copy(); churned[:10] = 1            # churn 1% > 0.5%
    monitor.observe(100, churned)
    assert not monitor.should_freeze(overflow=.10, tau_rel=.04)
    flags = monitor.reasons(overflow=.10, tau_rel=.04)
    assert flags == {"overflow_ok": True, "tau_ok": True, "churn_ok": False}


def test_criterion_never_fires_before_a_full_window():
    monitor = FreezeMonitor(window=50)
    monitor.observe(10, np.zeros(4, dtype=np.int64))
    assert monitor.churn is None
    assert not monitor.should_freeze(overflow=.0, tau_rel=.0)


def test_ensure_nonempty_regions_moves_the_nearest_cell_only():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    centers = region_centers(rs)
    part = np.array([0, 0, 0, 1], dtype=np.int32)      # regions 2 and 3 empty
    cx = np.array([10., 20., 30., 60.])
    cy = np.array([10., 10., 70., 10.])
    repaired, moves = ensure_nonempty_regions(part, 4, cx, cy, centers)
    assert np.bincount(repaired, minlength=4).min() >= 1
    assert len(moves) == 2
    assert moves[0]["region"] == 2 and moves[0]["cell"] == 2   # (30,70) is nearest
    assert int(repaired[2]) == 2


def test_region_cell_stats_reports_area_and_utilisation():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    part = np.array([0, 0, 1], dtype=np.int32)
    size_x = np.array([10., 10., 5.])
    size_y = np.array([10., 10., 4.])
    stats = region_cell_stats(part, size_x, size_y, rs)
    assert stats["region_cell_count"] == [2, 1, 0, 0]
    assert stats["region_cell_area"] == [200., 20., 0., 0.]
    assert stats["region_area"] == [2500.] * 4
    assert stats["region_utilization"] == [0.08, 0.008, 0., 0.]


def test_freeze_record_is_schema_complete(tmp_path):
    from ioplace.artifacts import save_freeze
    record = freeze_record(
        iteration=300, reason="criterion", overflow=.12, tau=1.5, tau_rel=.04,
        churn=.001, k=4, io_soft=1234, membership_npz="frozen_membership.npz",
        soft_npz="soft.npz", repaired_empty_regions=[], gp_iterations_soft=301,
        density_weight_soft=1.5e-5,
        stats={"region_cell_count": [1, 1, 1, 1], "region_cell_area": [1.] * 4,
               "region_area": [4.] * 4, "region_utilization": [.25] * 4},
        node_anchor="lower_left")
    save_freeze(str(tmp_path / "freeze.json"), record)      # must not raise
    assert record["schema_version"] == 1 and record["reason"] == "criterion"
    # Fence-diagnostics fix (P-F Task 7 finding): a separate --phase fence
    # invocation reads this back to report the anchor the soft phase actually
    # used, instead of trusting its own --node-anchor argument.
    assert record["node_anchor"] == "lower_left"
