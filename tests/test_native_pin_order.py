"""Bounded CPU regressions for the native pin canonicalization helpers."""

import numpy as np
import pytest

from ioplace.bench.native_pin_order import native_pin_selection, normalize_geometry, round_away
from pathlib import Path
from ioplace.bench import bookshelf_netlist as bn, tile_bookshelf as tb
from tests.test_bench_tile_bookshelf import write_toy_bookshelf
from tests.test_bench_bookshelf_netlist import _assert_native_equivalent


def _native_case(tmp_path, nodes=None, placement=None, nets=None):
    source = write_toy_bookshelf(str(tmp_path / "source" / "toy"))
    for suffix, transform in (("nodes", nodes), ("pl", placement), ("nets", nets)):
        if transform:
            path = Path(source + "." + suffix)
            path.write_text(transform(path.read_text()))
    prefix = str(tmp_path / "array" / "replica")
    tb.tile(source, prefix, R=1, C=2, seed=0)
    cache = str(tmp_path / "cache")
    meta = bn.build_tiled_netlist_cache(prefix + ".manifest.json", cache)
    cached, native, db = _assert_native_equivalent(prefix, cache)
    return cached, native, db, meta


@pytest.mark.parametrize("degree,retained", [(2, 0), (16, 0), (17, 8), (20, 10), (33, 25)])
def test_unnamed_duplicate_ties_match_actual_native_reader(tmp_path, degree, retained):
    text = (f"UCLA nets 1.0\n\nNumNets : 1\nNumPins : {degree}\n"
            f"NetDegree : {degree} duplicate\n" +
            "".join(f"o0 O : {i} 0\n" for i in range(degree)))
    cached, _, _, meta = _native_case(tmp_path, nets=lambda _: text)
    np.testing.assert_array_equal(cached.pin_offset_x, [retained+1., retained+1.])
    assert cached.num_nets == 2 and cached.pin2node.size == 2
    assert meta["n_pins_raw"] == 2*degree
    assert meta["n_duplicate_pins_removed"] == 2*(degree-1)


@pytest.mark.parametrize("name,old_size", [("o0", "2 2"), ("f0", "4 4"), ("p0", "0 0")])
@pytest.mark.parametrize("orientation", ["N", "S", "E", "W", "FN", "FS", "FE", "FW"])
def test_all_node_classes_and_orientations_match_native(tmp_path, name, old_size, orientation):
    def placement(text):
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] == name:
                parts[4] = orientation
                line = " ".join(parts)
            rows.append(line)
        return "\n".join(rows) + "\n"
    nets = ("UCLA nets 1.0\n\nNumNets : 1\nNumPins : 2\n"
            f"NetDegree : 2 orientation\n{name} O : -1 -2\no1 I : 0 0\n")
    _native_case(tmp_path, nodes=lambda t: t.replace(f"{name} {old_size}", f"{name} 6 10"),
                 placement=placement, nets=lambda _: nets)


def test_named_duplicates_and_fractional_positions_match_native(tmp_path):
    text = ("UCLA nets 1.0\n\nNumNets : 1\nNumPins : 2\nNetDegree : 2 named\n"
            "o0 O : 10 0 : 1 1 z\no0 O : 20 0 : 1 1 a\n")
    cached, _, _, _ = _native_case(tmp_path, nets=lambda _: text,
        placement=lambda t: t.replace("o0 0 0", "o0 -0.5 1.5"))
    assert cached.pin_offset_x[0] == 21.
    assert cached.node_x[0] == -1. and cached.node_y[0] == 2.


def test_verifier_checks_discarded_raw_pin_offsets(tmp_path):
    text = ("UCLA nets 1.0\n\nNumNets : 1\nNumPins : 17\nNetDegree : 17 duplicate\n" +
            "".join(f"o0 O : {i} 0\n" for i in range(17)))
    _, _, _, meta = _native_case(tmp_path, nets=lambda _: text)
    path = Path(meta["dst_prefix"] + ".nets")
    path.write_text(path.read_text().replace("t0_0/o0 O : 0 0", "t0_0/o0 O : 99 0", 1))
    result = bn.verify_against_bookshelf(str(tmp_path / "cache"), meta["dst_prefix"], mode="full")
    assert not result["ok"] and result["n_mismatched_pins"] >= 1
    assert result["n_checked_raw_pins"] == 34
    assert result["n_checked_canonical_pins"] == 2


@pytest.mark.parametrize("degree", [2, 16, 17, 20, 33])
def test_duplicate_pins_canonicalize_by_node_and_name(degree):
    names = {"o0": 0}
    nodes = np.zeros(degree, dtype=np.int64)
    selected, out_degrees, _ = native_pin_selection(
        np.array([degree]), nodes, names,
        {i: chr(ord("z") - (i % 26)) for i in range(degree)})
    assert selected.tolist() == [min(degree - 1, 25)]
    assert out_degrees.tolist() == [1]


@pytest.mark.parametrize("orientation,expected", [
    ("N", (6, 10, 2, 3)), ("S", (6, 10, 4, 7)),
    ("E", (10, 6, 3, 4)), ("W", (10, 6, 7, 2)),
    ("FN", (6, 10, 4, 3)), ("FS", (6, 10, 2, 7)),
    ("FE", (10, 6, 7, 4)), ("FW", (10, 6, 3, 2)),
])
def test_fixed_orientation_geometry(orientation, expected):
    w, h, x, y = normalize_geometry(
        np.array([6., 6.]), np.array([10., 10.]),
        np.array([orientation, "N"]), np.array([True, False]),
        np.array([0]), np.array([-1.]), np.array([-2.]))
    assert (w[0], h[0], x[0], y[0]) == expected


def test_negative_half_rounds_away_from_zero():
    np.testing.assert_array_equal(round_away([-.5, 1.5]), [-1., 2.])


def test_duplicate_pin_names_choose_lexicographically_first():
    selected, degrees, _ = native_pin_selection(
        np.array([2]), np.array([0, 0]), {"o0": 0}, {0: "z", 1: "a"})
    assert selected.tolist() == [1]
    assert degrees.tolist() == [1]


def test_malformed_degree_and_pin_indices_rejected():
    with pytest.raises(ValueError):
        native_pin_selection(np.array([2]), np.array([0]), {0: "o0"})
    with pytest.raises(ValueError):
        native_pin_selection(np.array([1]), np.array([2]), {"o0": 0})
