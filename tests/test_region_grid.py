import numpy as np
import pytest
from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from tests.test_netlist import make_tiny_netlist

DIE = (0., 0., 100., 100.)

def test_grid_ids_match_geometry():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)   # P0=左下 P1=右下 P2=左上 P3=右上
    rg = RegionGrid(rs)
    assert rg.grid.shape == (10, 10)
    ids = rg.region_of_points(np.array([10., 90., 10., 90.]),
                              np.array([10., 10., 90., 90.]))
    assert list(ids) == [0, 1, 2, 3]

def test_points_on_die_edge_clip():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    rg = RegionGrid(rs)
    ids = rg.region_of_points(np.array([0., 100.]), np.array([0., 100.]))
    assert list(ids) == [0, 3]

def test_pin_region_bitmask():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    rg = RegionGrid(rs)
    nl = make_tiny_netlist()
    # cells at (10,10)=P0,(30,10)=P0,(50,40)=P1(x=50 → 右半),(90,90)=P3
    bm = rg.pin_region_bitmask(nl, nl.node_x, nl.node_y)
    assert bm[0] == np.uint64(0b0001)            # n0: P0,P0
    assert bm[1] == np.uint64(0b1011)            # n1: P0,P1,P3
