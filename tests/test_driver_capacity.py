import json
import os

import numpy as np
import pytest

from ioplace.region_segments import CAPACITY_SCALARS


def test_pack_capacity_metrics_has_exactly_the_scalar_names():
    from ioplace.drivers.run_placement import _pack_capacity_metrics
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, table = _strip_case()
    res = evaluate_ref(nl, nl.node_x, nl.node_y, rg, segments=table,
                       segment_capacity=np.array([0.5, 4.0]))
    packed = _pack_capacity_metrics(res)
    assert tuple(packed) == CAPACITY_SCALARS
    assert isinstance(packed["num_over_capacity"], int)
    assert isinstance(packed["max_util"], float)
    assert packed["num_over_capacity"] == 1


def test_pack_capacity_metrics_is_all_none_without_a_measurement():
    from ioplace.drivers.run_placement import _pack_capacity_metrics
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, _table = _strip_case()
    packed = _pack_capacity_metrics(evaluate_ref(nl, nl.node_x, nl.node_y, rg))
    assert tuple(packed) == CAPACITY_SCALARS
    assert set(packed.values()) == {None}


def test_result_fields_carry_every_capacity_scalar():
    from ioplace.drivers.run_placement_io import RESULT_FIELDS
    for name in CAPACITY_SCALARS:
        assert name in RESULT_FIELDS, name
    for name in ("capacity", "capacity_source", "lambda_cap_final",
                 "cap_frac_singleton_groups"):
        assert name in RESULT_FIELDS, name


def test_the_cli_exposes_the_capacity_flags():
    from ioplace.drivers.run_placement import build_parser
    args = build_parser().parse_args(
        ["--mode", "io", "--config", "c.json", "--out", "o.json",
         "--capacity", "cap.npz", "--cap-tau-b-cells", "3",
         "--cap-m-pairs", "2", "--cap-m-seg", "1",
         "--cap-curvature-dref", "0.5"])
    assert args.capacity == "cap.npz"
    assert args.cap_tau_b_cells == 3.0
    assert args.cap_m_pairs == 2 and args.cap_m_seg == 1
    assert args.cap_curvature_dref == 0.5
    default = build_parser().parse_args(
        ["--mode", "io", "--config", "c.json", "--out", "o.json"])
    assert default.capacity is None
    assert default.cap_tau_b_cells == 2.0
    assert (default.cap_m_pairs, default.cap_m_seg) == (4, 2)


def test_capacity_needs_a_matching_geometry(tmp_path):
    """The driver must refuse a capacity.npz built for different regions
    before it touches CUDA."""
    from ioplace.capacity.extract import save_capacity
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import make_grid_regions
    from ioplace.drivers.run_placement_io import load_capacity_for_grid
    die = (0., 0., 100., 100.)
    table = enumerate_segments(RegionGrid(make_grid_regions(die, 2, 2, lattice=20)))
    path = tmp_path / "capacity.npz"
    save_capacity(path, table, np.ones(table.num_segments), source="lef_pitch")
    good = RegionGrid(make_grid_regions(die, 2, 2, lattice=20))
    assert load_capacity_for_grid(str(path), good)[1].shape == \
        (table.num_segments,)
    bad = RegionGrid(make_grid_regions(die, 4, 4, lattice=20))
    with pytest.raises(ValueError, match="different segment table"):
        load_capacity_for_grid(str(path), bad)


@pytest.mark.slow
@pytest.mark.gpu
def test_gcd_end_to_end_with_capacity(tmp_path):
    """sec 9's small end-to-end: build a LEF-fallback capacity.npz for GCD,
    run the IO driver with --capacity, and check that the capacity term was
    actually live -- registered, normalised, and reported."""
    import subprocess
    import sys
    from ioplace.capacity.extract import save_capacity, lef_capacity, \
        parse_tech_lef_layers
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    case_path = "results/route_feedback_20260914/gcd.json"
    if not os.path.exists(case_path):
        pytest.skip("the GCD case description is absent")
    tech = ("/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/"
            "NangateOpenCellLibrary.tech.lef")
    if not os.path.exists(tech):
        pytest.skip("NanGate45 tech LEF absent")

    # The die comes from the same place the driver reads it, so the segment
    # fingerprints match: run one throwaway read first.
    from ioplace.dreamplace_env import setup_dreamplace
    setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    params.load(case_path)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    die = (float(placedb.xl), float(placedb.yl),
           float(placedb.xh), float(placedb.yh))
    scale = float(getattr(placedb, "scale_factor", 1.0) or 1.0)

    # Task 8 fix: reuse get_regions_for (the exact call run_io makes
    # internally for --k 4 --rtype grid) rather than hand-rolling
    # make_grid_regions with a guessed lattice -- the brief's original
    # `make_grid_regions(die, 2, 2, lattice=64)` used lattice=64, but
    # get_regions_for's own default is lattice=512, so the two segment
    # tables' fingerprints never matched and load_capacity_for_grid
    # correctly refused the file as "a different segment table".
    from ioplace.drivers.run_placement import get_regions_for
    rs = get_regions_for(die, 4, "grid", 0)
    table = enumerate_segments(RegionGrid(rs))
    layers = parse_tech_lef_layers(tech)
    capacity = lef_capacity(table, layers, scale_factor=scale,
                            dbu_per_micron=2000., layer_range=("metal2", "metal10"))
    # scale it down so the run actually violates something and the term fires
    capacity = np.maximum(capacity * 0.02, 0.0)
    cap_path = tmp_path / "capacity.npz"
    save_capacity(cap_path, table, capacity, source="lef_pitch")

    out = tmp_path / "result.json"
    # Task 8 fix: _load_dreamplace chdirs into $DREAMPLACE_ROOT/install before
    # calling params.load(config_json) (run_placement.py's own comment on
    # that chdir: "config 內是相對路徑" -- the *config file's own* internal
    # paths are relative to DREAMPlace's install dir, not the CLI's --config
    # argument itself). A relative --config is resolved against the
    # subprocess's cwd at chdir time, not the caller's, so it must be made
    # absolute before crossing the subprocess boundary; the brief's own
    # in-process `params.load(case_path)` call above works unmodified only
    # because it runs before any chdir happens.
    subprocess.run(
        [sys.executable, "-m", "ioplace.drivers.run_placement",
         "--mode", "io", "--config", os.path.abspath(case_path), "--k", "4",
         "--rtype", "grid", "--out", str(out), "--every", "50",
         "--rho-max", "0.1", "--norm-policy", "grandplan",
         "--callback-order", "atomic", "--capacity", str(cap_path)],
        check=True)
    result = json.load(open(out))
    assert result["capacity"] == os.path.abspath(str(cap_path))
    assert result["capacity_source"] == "lef_pitch"
    assert result["num_segments"] == table.num_segments
    assert result["segment_demand_total"] >= 0
    assert result["max_util"] is not None
    assert result["lambda_cap_final"] is not None
    trace = [json.loads(line) for line
             in open(result["norm_trace"]).read().splitlines() if line.strip()]
    assert trace, "no norm_trace rows"
    assert any("cap" in row["terms"] for row in trace)
    assert any(row["terms"]["cap"]["active"] for row in trace)
