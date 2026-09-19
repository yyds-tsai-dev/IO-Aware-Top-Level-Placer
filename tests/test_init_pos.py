import numpy as np
import pytest
from types import SimpleNamespace

from ioplace.init_pos import INIT_MODES, apply_init, region_centers
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def _placedb():
    return SimpleNamespace(
        num_movable_nodes=4, num_terminals=1, num_terminal_NIs=0,
        num_physical_nodes=5,
        node_x=np.array([1., 2., 3., 4., 90.]),
        node_y=np.array([5., 6., 7., 8., 90.]),
        node_size_x=np.array([2., 2., 40., 2., 1.]),
        node_size_y=np.array([2., 2., 2., 2., 1.]),
        xl=0., yl=0., xh=100., yh=100.)


def _params():
    return SimpleNamespace(random_center_init_flag=1, random_seed=1000)


def test_modes_are_declared():
    assert INIT_MODES == ("die_center", "region_center", "seed")


def test_region_centers_are_area_weighted():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    assert np.allclose(region_centers(rs),
                       [[25., 25.], [75., 25.], [25., 75.], [75., 75.]])


def test_die_center_leaves_positions_to_dreamplace():
    db, params = _placedb(), _params()
    before_x = db.node_x.copy()
    info = apply_init(db, params, "die_center")
    assert params.random_center_init_flag == 1
    assert np.array_equal(db.node_x, before_x)
    assert info["mode"] == "die_center"


def test_region_center_puts_every_cell_centre_in_its_own_region():
    db, params = _placedb(), _params()
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    part = np.array([0, 1, 2, 3], dtype=np.int32)
    info = apply_init(db, params, "region_center", region_set=rs, part=part,
                      rng_seed=1000)
    assert params.random_center_init_flag == 0
    m = db.num_movable_nodes
    cx = db.node_x[:m] + db.node_size_x[:m] / 2.
    cy = db.node_y[:m] + db.node_size_y[:m] / 2.
    assert np.array_equal(RegionGrid(rs).region_of_points(cx, cy), part)
    # the 40-wide cell 2 must have its centre, not its lower-left, on the centre
    assert db.node_x[2] < 25.
    assert np.array_equal(db.node_x[m:], [90.])      # fixed nodes untouched
    assert info["mode"] == "region_center" and info["noise_scale_x"] == 0.1


def test_region_center_is_deterministic_per_seed():
    def run(seed):
        db, params = _placedb(), _params()
        rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
        apply_init(db, params, "region_center", region_set=rs,
                   part=np.array([0, 1, 2, 3], dtype=np.int32), rng_seed=seed)
        return db.node_x.copy()
    assert np.array_equal(run(1000), run(1000))
    assert not np.array_equal(run(1000), run(2000))


def test_seed_mode_writes_movable_slice_and_validates_fixed_slice():
    from ioplace.artifacts import Positions
    db, params = _placedb(), _params()
    good = Positions(node_x=np.array([10., 11., 12., 13., 90.]),
                     node_y=np.array([20., 21., 22., 23., 90.]),
                     die=(0., 0., 100., 100.), shift_factor=(0., 0.),
                     scale_factor=1., placedb_sha256="x", kind="seed")
    apply_init(db, params, "seed", positions=good)
    assert params.random_center_init_flag == 0
    assert np.array_equal(db.node_x, [10., 11., 12., 13., 90.])
    assert np.array_equal(db.node_y, [20., 21., 22., 23., 90.])

    db2 = _placedb()
    moved = Positions(node_x=np.array([10., 11., 12., 13., 91.]),
                      node_y=np.array([20., 21., 22., 23., 90.]),
                      die=(0., 0., 100., 100.), shift_factor=(0., 0.),
                      scale_factor=1., placedb_sha256="x", kind="seed")
    with pytest.raises(ValueError, match="fixed nodes"):
        apply_init(db2, _params(), "seed", positions=moved)

    db3 = _placedb()
    short = Positions(node_x=np.array([1., 2.]), node_y=np.array([1., 2.]),
                      die=(0., 0., 100., 100.), shift_factor=(0., 0.),
                      scale_factor=1., placedb_sha256="x", kind="seed")
    with pytest.raises(ValueError, match="num_physical"):
        apply_init(db3, _params(), "seed", positions=short)


def test_unknown_mode_and_missing_inputs_are_rejected():
    db, params = _placedb(), _params()
    with pytest.raises(ValueError, match="mode"):
        apply_init(db, params, "warp")
    with pytest.raises(ValueError, match="region_set and part"):
        apply_init(db, params, "region_center")
    with pytest.raises(ValueError, match="positions"):
        apply_init(db, params, "seed")
