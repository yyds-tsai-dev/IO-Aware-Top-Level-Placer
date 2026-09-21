import json
import os
import stat

import numpy as np
import pytest

from ioplace.capacity.extract import (CAPACITY_SEMANTICS, gcell_capacity,
                                      lef_capacity, load_capacity,
                                      parse_tech_lef_layers, save_capacity,
                                      track_density)
from ioplace.region_grid import RegionGrid
from ioplace.region_segments import enumerate_segments, segments_digest
from ioplace.regions import make_grid_regions

DIE = (0., 0., 3000., 3000.)
TECH_LEF = ("/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/"
            "NangateOpenCellLibrary.tech.lef")
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "capacity",
                       "tiny_resources.json")

# metal2-metal10 (the sec 8 GRT routing-layer range) track densities for the
# real NanGate45 tech LEF, hand-derived from its PITCH/DIRECTION table:
#   horizontal metal3/5/7/9 = 1/.14 + 1/.28 + 1/.8 + 1/1.6
#   vertical  metal2/4/6/8/10 = 1/.19 + 1/.28 + 1/.28 + 1/.8 + 1/1.6
# NOTE: the plan brief (task-3-brief.md) transcribed the vertical constant as
# 14.280015037593985 -- a single-digit ("1" -> "0") slip at the fourth
# decimal. 1/.19 + 1/.28 + 1/.28 + 1/.8 + 1/1.6 == 14.281015037593985 exactly
# (confirmed independently by the coordinator); the value below is the
# corrected one. Named here, rather than repeated as a second hard-coded
# literal in the LEF-fallback test below, so the relationship is self-evident
# and no third transcription can drift from it.
HORIZONTAL_M2_M10 = 12.589285714285714
VERTICAL_M2_M10 = 14.281015037593985


def _table():
    """2x2 regions on a lattice-4 grid over a 3000x3000 die: four segments,
    V at x=1500 over y in [0,1500] and [1500,3000], H at y=1500 over
    x in [0,1500] and [1500,3000]."""
    return enumerate_segments(RegionGrid(make_grid_regions(DIE, 2, 2, lattice=4)))


def test_gcell_capacity_is_the_overlap_weighted_row_sum():
    table = _table()
    with open(FIXTURE) as stream:
        resources = json.load(stream)
    capacity = gcell_capacity(table, resources, shift_factor=(0., 0.),
                              scale_factor=1.)
    # V at x=1500 -> GCell column 1 -> hcap[:,1] = [20,40,60].
    #   y in [0,1500]:    1.0*20 + 0.5*40           = 40
    #   y in [1500,3000]: 0.5*40 + 1.0*60           = 80
    # H at y=1500 -> GCell row 1 -> vcap[1,:] = [11,12,13].
    #   x in [0,1500]:    1.0*11 + 0.5*12           = 17
    #   x in [1500,3000]: 0.5*12 + 1.0*13           = 19
    np.testing.assert_allclose(capacity, [40., 80., 17., 19.])


def test_gcell_capacity_rejects_a_mismatched_resource_grid():
    table = _table()
    with open(FIXTURE) as stream:
        resources = json.load(stream)
    resources["horizontal_capacity"] = [[1, 2, 3]]
    with pytest.raises(ValueError, match="resource dimensions"):
        gcell_capacity(table, resources)


@pytest.mark.skipif(not os.path.exists(TECH_LEF), reason="NanGate45 tech LEF absent")
def test_nangate45_track_densities_come_from_the_real_pitches():
    layers = parse_tech_lef_layers(TECH_LEF)
    assert [layer["name"] for layer in layers] == \
        ["metal%d" % i for i in range(1, 11)]
    assert layers[0]["pitch"] == pytest.approx(0.14)
    assert layers[0]["direction"] == "HORIZONTAL"
    assert layers[9]["pitch"] == pytest.approx(1.6)
    assert layers[9]["direction"] == "VERTICAL"
    assert track_density(layers, "HORIZONTAL", ("metal2", "metal10")) == \
        pytest.approx(HORIZONTAL_M2_M10)
    assert track_density(layers, "VERTICAL", ("metal2", "metal10")) == \
        pytest.approx(VERTICAL_M2_M10)
    # unrestricted, metal1 joins the horizontal set
    assert track_density(layers, "HORIZONTAL") == pytest.approx(19.732142857142854)
    assert track_density(layers, "VERTICAL") == pytest.approx(VERTICAL_M2_M10)


@pytest.mark.skipif(not os.path.exists(TECH_LEF), reason="NanGate45 tech LEF absent")
def test_lef_fallback_is_rho_times_length_in_microns():
    table = _table()
    layers = parse_tech_lef_layers(TECH_LEF)
    capacity = lef_capacity(table, layers, scale_factor=1., dbu_per_micron=2000.,
                            layer_range=("metal2", "metal10"))
    # each segment is 1500 dbu = 0.75 um long; a vertical segment is crossed
    # by horizontal layers and vice versa (sec 5's "layers perpendicular to
    # the segment").
    np.testing.assert_allclose(capacity[:2], [HORIZONTAL_M2_M10 * 0.75] * 2)
    np.testing.assert_allclose(capacity[2:], [VERTICAL_M2_M10 * 0.75] * 2)


def test_capacity_npz_round_trips_with_a_receipt_hash(tmp_path):
    table = _table()
    capacity = np.array([40., 80., 17., 0.])
    path = tmp_path / "capacity.npz"
    metadata = save_capacity(path, table, capacity, source="openroad",
                             receipt_sha256="a" * 64,
                             extra={"uint8_backend": True,
                                    "congestion_iterations": 5})
    assert metadata["capacity_semantics"] == CAPACITY_SEMANTICS
    assert metadata["capacity_semantics"] == (
        "usable tracks crossing the segment; "
        "one net crossing consumes one track")
    assert metadata["capacity_source"] == "openroad"
    assert metadata["receipt_sha256"] == "a" * 64
    assert metadata["segments_sha256"] == segments_digest(table)
    assert metadata["zero_capacity_segments"] == 1
    assert metadata["uint8_backend"] is True

    loaded = load_capacity(path, table=table)
    np.testing.assert_array_equal(loaded["capacity"], capacity)
    assert loaded["metadata"] == metadata


def test_loading_against_a_different_geometry_fails_loudly(tmp_path):
    table = _table()
    path = tmp_path / "capacity.npz"
    save_capacity(path, table, np.ones(table.num_segments), source="lef_pitch")
    other = enumerate_segments(RegionGrid(make_grid_regions(DIE, 4, 4, lattice=4)))
    with pytest.raises(ValueError, match="segment table"):
        load_capacity(path, table=other)


def test_save_capacity_rejects_a_negative_or_nonfinite_capacity(tmp_path):
    table = _table()
    for bad in (np.array([1., -1., 1., 1.]), np.array([1., np.inf, 1., 1.])):
        with pytest.raises(ValueError, match="finite nonnegative"):
            save_capacity(tmp_path / "bad.npz", table, bad, source="lef_pitch")


def test_zero_capacity_is_preserved_not_floored(tmp_path):
    """Unit rule: zero-capacity segments stay blocked; no epsilon substitution
    anywhere in the extractor."""
    table = _table()
    path = tmp_path / "capacity.npz"
    save_capacity(path, table, np.zeros(table.num_segments), source="lef_pitch")
    loaded = load_capacity(path, table=table)
    assert (loaded["capacity"] == 0.).all()


def test_save_capacity_honours_the_umask(tmp_path):
    """artifacts._restore_umask_permissions: mkstemp() creates the temp file
    at 0600 and os.replace() preserves that mode across the rename, so
    without the fix every capacity.npz would come out unreadable by anyone
    but its writer regardless of the process umask."""
    table = _table()
    path = tmp_path / "capacity.npz"
    old_umask = os.umask(0o022)
    try:
        save_capacity(path, table, np.ones(table.num_segments), source="lef_pitch")
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == (0o666 & ~0o022)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("IOPLACE_OPENROAD_TESTS") != "1"
    or not os.access(os.environ.get("OPENROAD_BIN", ""), os.X_OK),
    reason="needs IOPLACE_OPENROAD_TESTS=1 and an executable $OPENROAD_BIN")
def test_real_openroad_extraction_on_gcd(tmp_path):
    """The one test that actually runs the router. Everything the extractor
    computes is already covered by the recorded fixture above; this checks the
    plumbing -- that run_openroad's resources.json still has the shape
    gcell_capacity expects, and that a real extraction produces finite,
    mostly-positive capacities."""
    import json as _json
    from ioplace.capacity.extract import run_extraction
    from ioplace.regions import RegionSet
    case = _json.load(open("results/route_feedback_20260914/gcd.json"))
    lefs = case["lef_input"]
    def_path = case["def_input"]
    resources, receipt_sha = run_extraction(
        def_path, lefs, tmp_path / "or", os.environ["OPENROAD_BIN"],
        congestion_iterations=5, signal_layers="metal2-metal10", threads=4)
    assert len(receipt_sha) == 64
    for key in ("x_edges_dbu", "y_edges_dbu", "horizontal_capacity",
                "vertical_capacity"):
        assert key in resources
    die = (0., 0., float(resources["x_edges_dbu"][-1]),
           float(resources["y_edges_dbu"][-1]))
    table = enumerate_segments(RegionGrid(
        __import__("ioplace.regions", fromlist=["make_grid_regions"])
        .make_grid_regions(die, 2, 2, lattice=8)))
    capacity = gcell_capacity(table, resources)
    assert np.isfinite(capacity).all() and (capacity >= 0).all()
    assert (capacity > 0).sum() >= table.num_segments // 2
