import json
from types import SimpleNamespace

import numpy as np
import pytest

from ioplace.artifacts import (
    FREEZE_FIELDS, MAIN_FLOW_RESULT_FIELDS, MAIN_FLOW_RESULT_SCHEMA_VERSION,
    PRODUCER_FIELDS, PRODUCER_SCHEMA_VERSION, Membership, Positions,
    file_sha256, load_freeze, load_membership, load_positions,
    load_producer_json, placedb_identity_sha256, save_freeze, save_membership,
    save_positions, save_producer_json, save_result, scaled_region_set)
from ioplace.regions import make_grid_regions


def _fake_placedb(**override):
    base = dict(num_movable_nodes=3, num_physical_nodes=4, num_nets=2,
                node_size_x=np.array([1., 1., 1., 2.]),
                node_size_y=np.array([1., 1., 1., 2.]),
                pin2node_map=np.array([0, 1, 2, 3], dtype=np.int32),
                pin2net_map=np.array([0, 0, 1, 1], dtype=np.int32),
                flat_net2pin_start_map=np.array([0, 2, 4], dtype=np.int32))
    base.update(override)
    return SimpleNamespace(**base)


def test_placedb_identity_is_stable_and_connectivity_sensitive():
    first = placedb_identity_sha256(_fake_placedb())
    assert first == placedb_identity_sha256(_fake_placedb())
    assert len(first) == 64
    assert first != placedb_identity_sha256(
        _fake_placedb(pin2node_map=np.array([0, 1, 3, 2], dtype=np.int32)))
    assert first != placedb_identity_sha256(
        _fake_placedb(node_size_x=np.array([1., 1., 1., 3.])))


def test_placedb_identity_ignores_node_positions():
    """Counts, node sizes and connectivity only -- never a coordinate. Node
    sizes do change under PlaceDB.scale(), which is why the docstring pins the
    fingerprint to the pre-initialize() point of the lifecycle; the producer
    (P-C Task 10) hashes in its `read` phase for exactly that reason."""
    db = _fake_placedb()
    before = placedb_identity_sha256(db)
    db.node_x = np.array([1., 2., 3., 4.])
    db.xl, db.yl, db.xh, db.yh = 0., 0., 10., 10.
    assert placedb_identity_sha256(db) == before


def _positions(tmp_path, **override):
    kwargs = dict(die=(0., 0., 100., 200.), shift_factor=(0., 0.), scale_factor=2.,
                  placedb_sha256="abc", kind="seed")
    kwargs.update(override)
    path = str(tmp_path / "seed.npz")
    save_positions(path, np.array([1., 2., 3.]), np.array([4., 5., 6.]), **kwargs)
    return path


def test_positions_round_trip_preserves_every_field(tmp_path):
    path = _positions(tmp_path)
    got = load_positions(path)
    assert isinstance(got, Positions)
    assert np.array_equal(got.node_x, [1., 2., 3.])
    assert np.array_equal(got.node_y, [4., 5., 6.])
    assert got.die == (0., 0., 100., 200.)
    assert got.shift_factor == (0., 0.)
    assert got.scale_factor == 2.
    assert got.placedb_sha256 == "abc" and got.kind == "seed"
    assert got.num_physical == 3


def test_positions_reject_bad_writes(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        save_positions(str(tmp_path / "a.npz"), [1.], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=1., placedb_sha256="a",
                       kind="bogus")
    with pytest.raises(ValueError, match="same length"):
        save_positions(str(tmp_path / "b.npz"), [1., 2.], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=1., placedb_sha256="a",
                       kind="seed")
    with pytest.raises(ValueError, match="finite"):
        save_positions(str(tmp_path / "c.npz"), [np.nan], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=1., placedb_sha256="a",
                       kind="seed")
    with pytest.raises(ValueError, match="scale_factor"):
        save_positions(str(tmp_path / "d.npz"), [1.], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=0., placedb_sha256="a",
                       kind="seed")


def test_positions_load_enforces_identity_and_size(tmp_path):
    path = _positions(tmp_path)
    load_positions(path, expect_num_physical=3, expect_sha256="abc")
    with pytest.raises(ValueError, match="num_physical"):
        load_positions(path, expect_num_physical=4)
    with pytest.raises(ValueError, match="placedb fingerprint"):
        load_positions(path, expect_sha256="other")


def test_membership_round_trip_and_validation(tmp_path):
    path = str(tmp_path / "m.npz")
    save_membership(path, np.array([0, 1, 1, 3]), source="mtkahypar", k=4,
                    seed=7, epsilon=0.03)
    got = load_membership(path, expect_num_movable=4, expect_k=4)
    assert isinstance(got, Membership)
    assert got.part.dtype == np.int32 and got.part.tolist() == [0, 1, 1, 3]
    assert got.source == "mtkahypar" and got.k == 4 and got.seed == 7
    assert got.epsilon == 0.03 and got.num_movable == 4
    with pytest.raises(ValueError, match="empty regions"):
        load_membership(path, require_nonempty=True)
    with pytest.raises(ValueError, match="num_movable"):
        load_membership(path, expect_num_movable=5)
    with pytest.raises(ValueError, match="out of range"):
        save_membership(str(tmp_path / "bad.npz"), np.array([0, 4]), source="x", k=4)


def test_freeze_record_requires_every_declared_field(tmp_path):
    record = {name: 0 for name in FREEZE_FIELDS}
    record["schema_version"] = 1
    path = str(tmp_path / "freeze.json")
    save_freeze(path, record)
    assert load_freeze(path)["schema_version"] == 1
    del record["churn"]
    with pytest.raises(ValueError, match="churn"):
        save_freeze(str(tmp_path / "bad.json"), record)


def test_result_requires_every_declared_field(tmp_path):
    record = {name: 0 for name in MAIN_FLOW_RESULT_FIELDS}
    record["schema_version"] = MAIN_FLOW_RESULT_SCHEMA_VERSION
    path = str(tmp_path / "result.json")
    save_result(path, record)
    assert json.load(open(path))["schema_version"] == MAIN_FLOW_RESULT_SCHEMA_VERSION
    del record["lg_loss"]
    with pytest.raises(ValueError, match="lg_loss"):
        save_result(str(tmp_path / "bad.json"), record)


def test_producer_json_requires_every_field_and_stamps_the_version(tmp_path):
    record = {name: 0 for name in PRODUCER_FIELDS}
    record["k"] = 16
    record["extract_bins"] = 64
    path = str(tmp_path / "producer.json")
    save_producer_json(path, record)
    on_disk = json.load(open(path))
    assert on_disk["k"] == 16 and on_disk["extract_bins"] == 64
    assert on_disk["schema_version"] == PRODUCER_SCHEMA_VERSION
    assert load_producer_json(path)["rects_per_region"] == 0
    del record["rects_per_region"]
    with pytest.raises(ValueError, match="rects_per_region"):
        save_producer_json(str(tmp_path / "bad.json"), record)


def test_scaled_region_set_matches_placedb_scale_and_still_validates():
    native = make_grid_regions((10., 20., 110., 220.), 2, 2, lattice=8)
    scaled = scaled_region_set(native, (10., 20.), 2.)
    scaled.validate()
    assert scaled.die == (0., 0., 200., 400.)
    assert scaled.lattice == native.lattice and scaled.k == native.k
    for a, b in zip(native.regions, scaled.regions):
        expected = (np.asarray(a.rects) - np.array([10., 20., 10., 20.])) * 2.
        assert np.allclose(np.asarray(b.rects), expected)
        assert a.name == b.name


def test_file_sha256_is_stable_and_content_sensitive(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    assert file_sha256(str(first)) == file_sha256(str(second))
    second.write_bytes(b"hellp")
    assert file_sha256(str(first)) != file_sha256(str(second))
