import numpy as np
import pytest

from ioplace.main_flow_metrics import (fence_compliance, io_accounting,
                                       phase_summary, region_area_balance)
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def test_io_accounting_closes_the_identity():
    out = io_accounting(io_soft=1000, io_fence_gp=1120, io_final=1155)
    assert out["io_delta_at_freeze"] == 120
    assert out["lg_loss"] == 35
    assert out["io_identity_residual"] == 0
    assert out["io_soft"] == 1000 and out["io_fence_gp"] == 1120
    assert out["io_count"] == 1155


def test_region_area_balance_reports_utilisation_and_count_deviation():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)   # 2500 each
    part = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    size_x = np.array([10., 10., 10., 5., 5.])
    size_y = np.array([10., 10., 10., 10., 10.])
    out = region_area_balance(part, size_x, size_y, rs)
    assert out["region_cell_count"] == [2, 1, 1, 1]
    assert out["region_utilization"] == [0.08, 0.04, 0.02, 0.02]
    assert out["utilization_max"] == 0.08 and out["utilization_min"] == 0.02
    assert out["utilization_ratio"] == 4.0
    assert out["cell_count_max"] == 2 and out["cell_count_min"] == 1
    assert out["cell_count_deviation"] == pytest.approx(0.6)   # |2-1.25|/1.25


def test_region_area_balance_survives_an_empty_region():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    out = region_area_balance(np.array([0, 0], dtype=np.int32),
                              np.array([1., 1.]), np.array([1., 1.]), rs)
    assert out["utilization_min"] == 0.0
    assert out["utilization_ratio"] is None
    assert out["empty_regions"] == [1, 2, 3]


def test_fence_compliance_distinguishes_lower_left_from_centre():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    rg = RegionGrid(rs)
    node_x = np.array([10., 40.])
    node_y = np.array([10., 10.])
    size_x = np.array([2., 30.])          # cell 1 centre at x=55 -> region 1
    size_y = np.array([2., 2.])
    part = np.array([0, 1], dtype=np.int32)
    out = fence_compliance(rg, node_x, node_y, part, size_x, size_y)
    assert out["lower_left"] == 0.5
    assert out["center"] == 1.0
    assert fence_compliance(rg, node_x, node_y, part)["center"] is None


def test_phase_summary_takes_the_max_peak_and_skips_unmeasured_phases():
    class _Timer:
        phases = {"gp_soft": {"t_s": 12.5, "peak_alloc_gb": 1.5,
                              "host_rss_hwm_at_phase_end": 3.0},
                  "gp_fence": {"t_s": 30.0, "peak_alloc_gb": 4.0,
                               "host_rss_hwm_at_phase_end": 5.0},
                  "lg": {"t_s": 1.0, "peak_alloc_gb": None,
                         "host_rss_hwm_at_phase_end": 5.0}}

    class _Sampler:
        device_used_gb = 7.25

    out = phase_summary(_Timer(), _Sampler(), host_rss=6.0)
    assert out["t_gp_soft"] == 12.5 and out["t_gp_fence"] == 30.0 and out["t_lg"] == 1.0
    # every declared phase gets a key, so result.json's contract cannot break
    # when only half the flow ran
    assert out["t_read_soft"] == 0.0 and out["t_freeze"] == 0.0
    assert out["t_read_fence"] == 0.0 and out["t_eval"] == 0.0
    assert out["peak_mem_mb"] == 4.0 * 1024.0
    assert out["peak_mem_mb_by_phase"]["gp_fence"] == 4.0 * 1024.0
    assert out["peak_mem_mb_by_phase"]["lg"] is None
    assert out["device_used_gb"] == 7.25
    assert out["host_peak_rss_gb"] == 6.0
