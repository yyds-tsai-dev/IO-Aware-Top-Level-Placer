import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate
from ioplace.export.evaluation import (
    SCHEMA_VERSION, load_evaluation, pin_regions_from_evaluation, save_evaluation,
)
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from tests.test_netlist import make_tiny_netlist


def _evidence(tmp_path):
    nl = make_tiny_netlist()
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10))
    result = evaluate(nl, nl.node_x, nl.node_y, rg)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y, ["n0", "n1"])
    return path, nl, rg, result


def test_evidence_preserves_pin_regions_and_exact_evaluator_results(tmp_path):
    path, nl, rg, result = _evidence(tmp_path)
    data = load_evaluation(path, net_names=[b"n0", b"n1"], rg=rg)
    assert pin_regions_from_evaluation(data) == {0: [0], 1: [0, 1, 3]}
    np.testing.assert_array_equal(data["net_degrees"], [2, 3])
    np.testing.assert_array_equal(data["per_net_crossings"], [0, 2])
    assert {tuple(p): v for p, v in zip(data["boundary_pairs"], data["boundary_demand"]) if v} == result.boundary_pair_demand
    assert len(data["boundary_length"]) == 4
    assert np.all(data["boundary_length"] == 50.)
    assert data["metadata"]["totals"]["io_count"] == result.io_count


def test_wrong_net_order_or_region_geometry_cannot_be_paired(tmp_path):
    path, _, _, _ = _evidence(tmp_path)
    with pytest.raises(ValueError, match="net-name orders"):
        load_evaluation(path, net_names=["n1", "n0"])
    other = RegionGrid(make_grid_regions((0., 0., 200., 100.), 2, 2, lattice=10))
    with pytest.raises(ValueError, match="region geometries"):
        load_evaluation(path, rg=other)


def test_corrupted_per_net_values_fail_total_validation(tmp_path):
    path, _, _, _ = _evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["per_net_crossings"][0] = 10
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="total mismatch"):
        load_evaluation(path)


def _straddling_evidence(tmp_path, straddle=True):
    from tests.test_straddle import _corner_case
    nl = _corner_case()
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10))
    result = evaluate(nl, nl.node_x, nl.node_y, rg, straddle=straddle)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y,
                    ["n0", "n1", "n2", "n3"])
    return path, nl, rg, result


def test_schema_2_round_trips_the_straddle_block(tmp_path):
    """Name kept for continuity with P-F's original test; the straddle block
    itself is schema 2's addition, but P-D Task 8 bumped SCHEMA_VERSION to 3
    for the sec 5 capacity block, so a freshly written archive now carries 3
    -- the straddle block must still round-trip unchanged under it."""
    path, nl, rg, result = _straddling_evidence(tmp_path)
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["schema_version"] == SCHEMA_VERSION
    straddle = data["metadata"]["straddle"]
    assert straddle["straddle_cells"] == 2
    assert straddle["straddle_pin_split_nets"] == 1
    assert straddle["straddle_area_fraction"] == pytest.approx(20. / 48.)
    assert straddle["straddle_out_area"] == pytest.approx(20.)
    assert straddle["straddle_movable_area"] == pytest.approx(48.)
    assert straddle["straddle_wide_cells"] == 0
    assert straddle["anchor"] == "center" and straddle["box"] == "closed_four_corner"
    np.testing.assert_array_equal(data["per_node_straddle"], [0, 1, 1, 0])
    np.testing.assert_array_equal(data["per_net_pin_split"], [-1, 0, 1, -1])
    # Fix round 1 item 2: an int array silently becoming float through npz is
    # the classic failure in this exact code path, and the integer straddle
    # fields carry a bit-exactness contract (global constraints).
    assert data["per_node_straddle"].dtype == np.uint8
    assert data["per_net_pin_split"].dtype == np.int32


def test_evidence_without_diagnostics_records_that_fact(tmp_path):
    path, _, rg, _ = _straddling_evidence(tmp_path, straddle=False)
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["schema_version"] == SCHEMA_VERSION
    assert data["metadata"]["straddle"] is None
    assert "per_node_straddle" not in data and "per_net_pin_split" not in data


def test_corrupted_straddle_arrays_fail_total_validation(tmp_path):
    path, _, _, _ = _straddling_evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["per_node_straddle"][0] = 1          # now 3 straddlers, metadata says 2
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="total mismatch: per_node_straddle"):
        load_evaluation(path)


def test_a_schema_1_archive_still_loads(tmp_path):
    path, _, _, _ = _straddling_evidence(tmp_path)
    import json
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema_version"] = 1
    metadata.pop("straddle")
    arrays.pop("per_node_straddle")
    arrays.pop("per_net_pin_split")
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    data = load_evaluation(path)
    assert data["metadata"]["schema_version"] == 1
    assert data["metadata"].get("straddle") is None


def test_a_future_schema_version_is_rejected_not_silently_misparsed(tmp_path):
    """Fix round 1 item 1. The contract (SUPPORTED_SCHEMA_VERSIONS,
    export/evaluation.py) is read-v1/v2/v3, reject anything else -- in
    particular a FUTURE version this reader was never taught. Originally
    written against schema 3 as the future version; P-D Task 8 took schema 3
    for the sec 5 capacity block (global constraints' schema-version rule),
    so the still-genuinely-unsupported probe moved to 4 -- the read-vs-reject
    line moves again the day some later subproject claims it. An untested
    rejection path is exactly what would silently start mis-parsing the day
    that version bump lands."""
    path, _, _, _ = _straddling_evidence(tmp_path)
    import json
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema_version"] = 4
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="unsupported evaluator evidence schema"):
        load_evaluation(path)


def test_an_unknown_non_numeric_schema_version_is_also_rejected(tmp_path):
    """Same contract, exercised from the other direction: schema_version
    missing/garbage (not just numerically ahead) must not be treated as
    supported either."""
    path, _, _, _ = _straddling_evidence(tmp_path)
    import json
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema_version"] = "not-a-version"
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="unsupported evaluator evidence schema"):
        load_evaluation(path)


def _capacity_evidence(tmp_path, with_capacity=True):
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from ioplace.region_segments import enumerate_segments
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, table = _strip_case()
    capacity = np.array([0.5, 4.0]) if with_capacity else None
    result = evaluate_ref(nl, nl.node_x, nl.node_y, rg, segments=table,
                          segment_capacity=capacity)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y, ["n0"],
                    segments=table)
    return path, nl, rg, result, table


def test_capacity_block_round_trips(tmp_path):
    path, _nl, rg, result, table = _capacity_evidence(tmp_path)
    data = load_evaluation(path, rg=rg)
    block = data["metadata"]["capacity"]
    from ioplace.region_segments import CAPACITY_SCALARS, segments_digest
    for name in CAPACITY_SCALARS:
        assert name in block
    assert block["num_over_capacity"] == 1
    assert block["segment_demand_total"] == 2
    assert block["segments_sha256"] == segments_digest(table)
    assert block["capacity_semantics"] == (
        "usable tracks crossing the segment; "
        "one net crossing consumes one track")
    np.testing.assert_array_equal(data["segment_demand"], result.segment_demand)
    np.testing.assert_array_equal(data["segment_capacity"], [0.5, 4.0])
    np.testing.assert_array_equal(data["segment_util"], result.segment_util)


def test_evidence_without_capacity_records_that_fact(tmp_path):
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, _table = _strip_case()
    result = evaluate_ref(nl, nl.node_x, nl.node_y, rg)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y, ["n0"])
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["capacity"] is None
    assert "segment_demand" not in data


def test_a_corrupted_segment_demand_fails_total_validation(tmp_path):
    import json
    path, _nl, _rg, _result, _table = _capacity_evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["segment_demand"][0] += 5
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="total mismatch: segment_demand"):
        load_evaluation(path)


def test_older_schema_archives_still_load(tmp_path):
    """The SUPPORTED_SCHEMA_VERSIONS convention P-F introduced: results/ holds
    historical evidence the route-calibration tooling still pairs against."""
    import json
    from ioplace.export.evaluation import SUPPORTED_SCHEMA_VERSIONS
    path, _nl, _rg, _result, _table = _capacity_evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema_version"] = SUPPORTED_SCHEMA_VERSIONS[0]
    metadata.pop("capacity")
    for key in ("segment_demand", "segment_capacity", "segment_util"):
        arrays.pop(key)
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    data = load_evaluation(path)
    assert data["metadata"]["schema_version"] == SUPPORTED_SCHEMA_VERSIONS[0]
    assert data["metadata"].get("capacity") is None
