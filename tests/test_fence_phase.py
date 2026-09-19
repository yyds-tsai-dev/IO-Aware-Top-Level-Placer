import dataclasses
import os
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.dreamplace_env import setup_dreamplace
setup_dreamplace()          # sys.path only -- needed for the `import PlaceObj`
                            # tests below; no LEF/DEF read, CPU-only.

from ioplace.fence_phase import (build_fence_placedb, clamp_density_weight,
                                 install_density_weight_clamp)

GCD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results/route_feedback_20260914/gcd.json")


class _FlatModel:
    def __init__(self, value):
        self.density_weight = torch.tensor([value], dtype=torch.float64)


class _FenceModel:
    def __init__(self, values, dtype=torch.float64):
        self.density_weight = torch.tensor(values, dtype=dtype)
        self.density_weight_u = torch.tensor(values, dtype=dtype) * 2.0
        self.density_weight_step_size_inc_low = 1.03
        self.density_weight_step_size = 0.0


def test_clamp_is_a_noop_inside_the_band():
    model = _FlatModel(2.0)
    log = clamp_density_weight(model, reference=1.0)
    assert model.density_weight.tolist() == [2.0]
    assert log["bound"] is False and log["lo_abs"] == 0.25 and log["hi_abs"] == 4.0
    # lo_abs/hi_abs are absolute bounds, not the [0.25, 4] multipliers: they
    # only coincide with them when reference == 1 (amendment D-14).
    scaled = clamp_density_weight(_FlatModel(2.0), reference=2.0)
    assert scaled["lo_abs"] == 0.5 and scaled["hi_abs"] == 8.0
    assert scaled["bound"] is False


def test_clamp_binds_above_and_below():
    high = _FlatModel(40.0)
    assert clamp_density_weight(high, reference=1.0)["bound"] is True
    assert high.density_weight.tolist() == [4.0]
    low = _FlatModel(0.01)
    assert clamp_density_weight(low, reference=1.0)["bound"] is True
    assert low.density_weight.tolist() == [0.25]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clamp_keeps_the_fence_subgradient_state_consistent(dtype):
    model = _FenceModel([0.5, 8.0, 1.0], dtype=dtype)
    log = clamp_density_weight(model, reference=1.0)
    assert model.density_weight.tolist() == [0.5, 4.0, 1.0]
    # u is scaled by the same elementwise ratio: [1.0, 16.0, 2.0] * [1, .5, 1]
    assert model.density_weight_u.tolist() == [1.0, 8.0, 2.0]
    expected = 0.03 * float(torch.tensor([1.0, 8.0, 2.0], dtype=dtype).norm(p=2))
    assert model.density_weight_step_size == pytest.approx(expected, rel=1e-6)
    assert log["before"] == [0.5, 8.0, 1.0] and log["after"] == [0.5, 4.0, 1.0]


def test_clamp_leaves_zero_density_weight_entries_untouched():
    # A zero entry has no density_weight_u counterpart to rescale, so raising
    # it to lo_abs would be undone by the very next overflow-based update.
    model = _FenceModel([0.0, 8.0, 1.0])
    log = clamp_density_weight(model, reference=1.0)
    assert model.density_weight.tolist() == [0.0, 4.0, 1.0]
    assert model.density_weight_u.tolist() == [0.0, 8.0, 2.0]
    assert log["before"][0] == 0.0 and log["after"][0] == 0.0


def test_clamp_rejects_a_nonpositive_reference():
    with pytest.raises(ValueError, match="reference"):
        clamp_density_weight(_FlatModel(1.0), reference=0.0)


def test_install_density_weight_clamp_is_scoped_to_the_cleanup():
    import PlaceObj
    from ioplace.drivers.run_placement_io import _io_cleanup

    original = PlaceObj.PlaceObj.__dict__["initialize_density_weight"]

    with _io_cleanup() as cleanup:
        install_density_weight_clamp(cleanup, reference=1.0, log=[])
        assert PlaceObj.PlaceObj.initialize_density_weight is not original
    assert PlaceObj.PlaceObj.__dict__["initialize_density_weight"] is original

    with pytest.raises(RuntimeError):
        with _io_cleanup() as cleanup:
            install_density_weight_clamp(cleanup, reference=1.0, log=[])
            assert PlaceObj.PlaceObj.initialize_density_weight is not original
            raise RuntimeError("boom")
    assert PlaceObj.PlaceObj.__dict__["initialize_density_weight"] is original


def test_install_density_weight_clamp_wraps_and_clamps_in_one_call():
    import PlaceObj
    from ioplace.drivers.run_placement_io import _io_cleanup

    saved = PlaceObj.PlaceObj.__dict__["initialize_density_weight"]
    PlaceObj.PlaceObj.initialize_density_weight = \
        lambda self, params, placedb: self.density_weight
    try:
        log = []
        with _io_cleanup() as cleanup:
            install_density_weight_clamp(cleanup, reference=1.0, log=log)
            instance = PlaceObj.PlaceObj.__new__(PlaceObj.PlaceObj)
            instance.density_weight = torch.tensor([40.0])
            result = instance.initialize_density_weight(None, None)
        assert result is instance.density_weight
        assert result.tolist() == [4.0]
        assert len(log) == 1
    finally:
        PlaceObj.PlaceObj.initialize_density_weight = saved


@pytest.mark.slow
def test_build_fence_placedb_injects_fences_and_scales_the_warm_start():
    from ioplace.artifacts import Positions, placedb_identity_sha256
    from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
    if not os.path.exists(GCD):
        pytest.skip("GCD benchmark required")
    params0, db0 = _load_dreamplace(GCD)
    m, n_phys = db0.num_movable_nodes, db0.num_physical_nodes
    die = (float(db0.xl), float(db0.yl), float(db0.xh), float(db0.yh))
    rs = get_regions_for(die, 4, "grid", 0)
    sha = placedb_identity_sha256(db0)
    from ioplace.region_grid import RegionGrid
    cx = np.asarray(db0.node_x[:m], np.float64) + np.asarray(db0.node_size_x[:m], np.float64) / 2
    cy = np.asarray(db0.node_y[:m], np.float64) + np.asarray(db0.node_size_y[:m], np.float64) / 2
    part = RegionGrid(rs).region_of_points(cx, cy).astype(np.int32)
    seed = Positions(node_x=np.asarray(db0.node_x[:n_phys], np.float64),
                     node_y=np.asarray(db0.node_y[:n_phys], np.float64),
                     die=die, shift_factor=(0., 0.), scale_factor=1.,
                     placedb_sha256=sha, kind="soft")

    params, db, info = build_fence_placedb(GCD, rs, part, seed)
    assert len(db.regions) == 4
    assert info["placedb_sha256"] == sha
    escape = info["escape_cell"]
    assert db.node2fence_region_map[escape] == 4          # implicit no-fence bucket
    mask = np.ones(m, dtype=bool); mask[escape] = False
    assert np.array_equal(db.node2fence_region_map[:m][mask], part[mask])
    # warm start survives initialize()'s scale(): native -> (v - shift) * scale
    shift, scale = info["shift_factor"], info["scale_factor"]
    expected_x = (seed.node_x[:m] - shift[0]) * scale
    expected_y = (seed.node_y[:m] - shift[1]) * scale
    assert np.allclose(np.asarray(db.node_x[:m], np.float64), expected_x, atol=1e-3)
    assert np.allclose(np.asarray(db.node_y[:m], np.float64), expected_y, atol=1e-3)
    assert params.random_center_init_flag == 0

    # negative guards -- reuse db0/rs/part/seed; build_fence_placedb always
    # reads its own fresh PlaceDB internally (the phase-3 contract requires
    # a fresh instance regardless), so no additional top-level GCD read is
    # added here beyond what each guard call itself needs.
    with pytest.raises(ValueError, match="membership has"):
        build_fence_placedb(GCD, rs, part[:-1], seed)

    foreign_rs = get_regions_for((0.0, 0.0, 100.0, 100.0), 4, "grid", 0)
    with pytest.raises(ValueError, match="native post-read units"):
        build_fence_placedb(GCD, foreign_rs, part, seed)

    bad_seed = dataclasses.replace(seed, placedb_sha256="0" * 64)
    with pytest.raises(ValueError, match="different netlist"):
        build_fence_placedb(GCD, rs, part, bad_seed)
