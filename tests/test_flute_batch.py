import numpy as np
import pytest
import json
import subprocess
import sys

from ioplace.route_eval.topology import flute_tree
from ioplace.route_eval.topology_batch import batch_flute_trees


def _per_net(tree, net):
    lo, hi = tree["branch_starts"][net:net + 2]
    return (tree["positions"][lo:hi], tree["parents"][lo:hi] - lo)


def test_batch_matches_raw_flute_with_duplicates_zero_edges_and_global_ids():
    nets = [
        np.array([[0., 0.], [4., 3.]]),
        np.array([[0., 0.], [10., 10.], [2., 6.], [2., 6.], [7., 4.]]),
        np.array([[0., 0.], [0., 2.], [1., 1.], [2., 0.], [2., 2.]]),
    ]
    starts = np.r_[0, np.cumsum([len(net) for net in nets])]
    pins = np.concatenate(nets)
    batch, provenance = batch_flute_trees(pins, starts, threads=2)
    assert batch["branch_starts"].tolist() == [0, 2, 10, 18]
    assert provenance["backend"] == "native-openmp-flute"
    for net_id, pins_net in enumerate(nets):
        raw, _ = flute_tree(pins_net)
        positions, parents = _per_net(batch, net_id)
        np.testing.assert_array_equal(positions, raw["positions"])
        np.testing.assert_array_equal(parents, raw["parents"])
        pin_lo = starts[net_id]
        row_lo = batch["branch_starts"][net_id]
        np.testing.assert_array_equal(
            batch["terminal_nodes"][pin_lo:starts[net_id + 1]] - row_lo,
            raw["terminal_nodes"])
    center = starts[2] + 2
    row = batch["terminal_nodes"][center]
    np.testing.assert_array_equal(batch["positions"][row],
                                  batch["positions"][batch["parents"][row]])


def test_empty_batch_and_serial_parallel_are_bitwise_identical():
    empty, meta = batch_flute_trees(np.empty((0, 2)), np.array([0]), threads=8)
    assert empty["positions"].shape == (0, 2)
    assert empty["parents"].shape == empty["terminal_nodes"].shape == (0,)
    assert empty["branch_starts"].tolist() == [0]
    assert meta["net_count"] == meta["pin_count"] == meta["raw_row_count"] == 0

    rng = np.random.default_rng(19)
    degrees = rng.integers(2, 18, 80)
    starts = np.r_[0, np.cumsum(degrees)]
    pins = rng.normal(size=(starts[-1], 2))
    serial, _ = batch_flute_trees(pins, starts, threads=1)
    parallel, _ = batch_flute_trees(pins, starts, threads=8)
    for key in ("positions", "parents", "terminal_nodes", "branch_starts"):
        np.testing.assert_array_equal(serial[key], parallel[key])


@pytest.mark.parametrize("pins,starts,message", [
    (np.zeros((1, 2)), np.array([0, 1]), "degree"),
    (np.zeros((257, 2)), np.array([0, 257]), "degree"),
    (np.array([[0., 0.], [np.nan, 1.]]), np.array([0, 2]), "finite"),
    (np.array([[0., 0.], [1e20, 0.]]), np.array([0, 2]), "range"),
    (np.zeros((4, 2)), np.array([0, 3, 2, 4]), "starts"),
])
def test_rejects_invalid_degrees_coordinates_and_csr(pins, starts, message):
    with pytest.raises(ValueError, match=message):
        batch_flute_trees(pins, starts)


def test_coordinate_scale_accuracy_and_threads_validation():
    pins = np.zeros((2, 2))
    starts = np.array([0, 2])
    for kwargs, message in [
        ({"coordinate_scale": 0}, "scale"), ({"accuracy": 0}, "accuracy"),
        ({"accuracy": 11}, "accuracy"), ({"threads": 0}, "threads"),
    ]:
        with pytest.raises(ValueError, match=message):
            batch_flute_trees(pins, starts, **kwargs)


def test_quantization_half_ties_match_numpy_rint_scalar_contract():
    pins = np.array([[0., 0.], [.0005, 1.], [.0015, 2.], [1., 3.]])
    batch, _ = batch_flute_trees(pins, np.array([0, 4]),
                                 coordinate_scale=1000., threads=1)
    scalar, _ = flute_tree(pins, coordinate_scale=1000.)
    np.testing.assert_array_equal(batch["positions"], scalar["positions"])
    np.testing.assert_array_equal(batch["parents"], scalar["parents"])
    np.testing.assert_array_equal(batch["terminal_nodes"], scalar["terminal_nodes"])


def test_concurrent_first_use_initializes_native_lut_once():
    code = r'''
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ioplace.route_eval.topology_batch import batch_flute_trees
pins=np.array([[0.,0.],[1.,1.]])
starts=np.array([0,2])
with ThreadPoolExecutor(max_workers=8) as pool:
    values=list(pool.map(lambda _: batch_flute_trees(pins,starts)[1],range(8)))
print(json.dumps([value["lut_initializations"] for value in values]))
'''
    completed = subprocess.run([sys.executable, "-c", code], check=True,
                               text=True, capture_output=True)
    assert json.loads(completed.stdout) == [1] * 8
