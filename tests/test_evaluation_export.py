import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate
from ioplace.export.evaluation import (
    load_evaluation, pin_regions_from_evaluation, save_evaluation,
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
    path, nl, rg, result = _straddling_evidence(tmp_path)
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["schema_version"] == 2
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


def test_evidence_without_diagnostics_records_that_fact(tmp_path):
    path, _, rg, _ = _straddling_evidence(tmp_path, straddle=False)
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["schema_version"] == 2
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
