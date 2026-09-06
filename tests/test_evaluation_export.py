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
