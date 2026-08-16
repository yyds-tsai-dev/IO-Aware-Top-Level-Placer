"""M4 T9 tests for `ioplace.bench.bookshelf_netlist` (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 5.3
layer 2 / T9 row): replication-constructor correctness (translation,
`t{i}_{j}/` prefix resolution, glue tail), the required movable-first
repermutation + pin2node remap, the module's own "cheap invariants", and
`verify_against_bookshelf`'s ability to catch a deliberately-broken array.

All tests run against the same toy fixture `tile_bookshelf`/`glue_gen`'s
own test suites use (`write_toy_bookshelf`: 5 movable + 1 IO pin + 1 fixed
macro, 3 nets) -- never the real ~27.7M-node 3x3 array (this task's own
instructions: write real code, don't run it against the multi-GB corpus
right now, host RAM is shared with a concurrent GPU run).
"""
import json

import numpy as np
import pytest

from ioplace.bench import bookshelf_netlist as bn
from ioplace.bench import glue_gen
from ioplace.bench import tile_bookshelf as tb
from tests.test_bench_tile_bookshelf import write_toy_bookshelf


def _build_cache(tmp_path, R, C, seed=0, with_glue=True, glue_seed=1):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=R, C=C, seed=seed)

    if with_glue:
        rng = np.random.default_rng(glue_seed)
        lambda_0, alpha = 3.0, 0.8
        counts = glue_gen.sample_glue_net_count(rng, lambda_0, alpha, R=R, C=C)
        assert sum(counts.values()) > 0, "toy lambda_0 chosen so this is overwhelmingly likely"
        candidate_nodes = {(i, j): ["o0", "o1", "o2"] for i in range(R) for j in range(C)}
        nets = glue_gen.sample_glue_nets(rng, counts, candidate_nodes)
        glue_gen.append_glue_nets(dst, nets, lambda_0, alpha, seed=glue_seed)

    cache_dir = str(tmp_path / "cache")
    meta = bn.build_tiled_netlist_cache(dst + ".manifest.json", cache_dir)
    return src, dst, cache_dir, meta


# ---------------------------------------------------------------------------
# parse_source_tile
# ---------------------------------------------------------------------------

def test_parse_source_tile_reads_toy_design(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    tile, (pl_x, pl_y), _ = bn.parse_source_tile(src)
    assert tile.n_nodes == 7
    assert tile.n_terminals == 2
    assert tile.n_terminal_ni == 1
    assert tile.n_nets == 3
    assert tile.n_pins == 7
    # movable-first: o0..o4 (idx 0-4) not terminal, p0 (idx5)/f0 (idx6) terminal
    assert list(tile.is_terminal) == [False] * 5 + [True, True]
    assert tile.name2local["o2"] == 2 and tile.name2local["f0"] == 6
    # .pl positions in file order (name-indexed, robust to file-order match)
    assert (pl_x[tile.name2local["o2"]], pl_y[tile.name2local["o2"]]) == (4.0, 0.0)
    assert (pl_x[tile.name2local["f0"]], pl_y[tile.name2local["f0"]]) == (6.0, 0.0)
    # net n0: o0(O,-1,-1) o1(I,-1,-1) f0(I,0,0)
    assert tile.net_degrees.tolist() == [3, 2, 2]
    n0_pins = [i for i in range(tile.n_pins) if tile.pin2net[i] == 0]
    assert [tile.pin2node[i] for i in n0_pins] == [
        tile.name2local["o0"], tile.name2local["o1"], tile.name2local["f0"]]
    assert [(tile.pin_offset_x[i], tile.pin_offset_y[i]) for i in n0_pins] == [
        (-1.0, -1.0), (-1.0, -1.0), (0.0, 0.0)]


# ---------------------------------------------------------------------------
# build_tiled_netlist_cache: base replication (no glue)
# ---------------------------------------------------------------------------

def test_base_counts_are_exactly_rxc_times_source(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=False)
    assert meta["n_nodes"] == 7 * 4
    assert meta["num_movable"] == 5 * 4
    assert meta["num_terminals"] == 2 * 4
    assert meta["num_terminal_NIs"] == 1 * 4
    assert meta["n_nets"] == 3 * 4
    assert meta["n_pins"] == 7 * 4
    assert meta["n_glue_nets"] == 0 and meta["n_glue_pins"] == 0
    assert meta["node_order"] == "movable_first_permuted"


def test_translation_matches_i_w_j_h(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=False)
    nl, meta = bn.load_tiled_netlist(cache_dir, mmap=False)
    inv_perm = np.load(cache_dir + "/inv_perm.npy")

    src_tile, _, _ = bn.parse_source_tile(src)
    n_src = src_tile.n_nodes

    def stored(i, j, name):
        return inv_perm[(i * 2 + j) * n_src + src_tile.name2local[name]]

    # W=6, H=4 (matches test_bench_tile_bookshelf's own toy geometry)
    assert meta["tile_width"] == 6 and meta["tile_height"] == 4
    assert (nl.node_x[stored(0, 0, "o0")], nl.node_y[stored(0, 0, "o0")]) == (0.0, 0.0)
    assert (nl.node_x[stored(1, 0, "o0")], nl.node_y[stored(1, 0, "o0")]) == (6.0, 0.0)
    assert (nl.node_x[stored(0, 1, "o0")], nl.node_y[stored(0, 1, "o0")]) == (0.0, 4.0)
    assert (nl.node_x[stored(1, 1, "o2")], nl.node_y[stored(1, 1, "o2")]) == (4.0 + 6.0, 0.0 + 4.0)


def test_movable_first_permutation_and_io_term_backward_assumption(tmp_path):
    """`Netlist.num_movable` must be a true prefix count: every stored
    index < num_movable is movable, every index >= num_movable is a
    terminal (matches `ioplace.ops.io_term.IoTerm`'s backward invariant
    `gx[meta.num_movable:] = 0.0`)."""
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=3, with_glue=False)
    nl, meta = bn.load_tiled_netlist(cache_dir, mmap=False)
    inv_perm = np.load(cache_dir + "/inv_perm.npy")
    src_tile, _, _ = bn.parse_source_tile(src)
    n_src = src_tile.n_nodes

    is_terminal_pre = np.tile(src_tile.is_terminal, 6)
    is_terminal_stored = np.empty_like(is_terminal_pre)
    is_terminal_stored[inv_perm] = is_terminal_pre
    assert not is_terminal_stored[: nl.num_movable].any()
    assert is_terminal_stored[nl.num_movable:].all()
    assert nl.num_movable == 5 * 6


# ---------------------------------------------------------------------------
# glue tail
# ---------------------------------------------------------------------------

def test_glue_pin2node_resolves_to_correct_stored_indices(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=True)
    assert meta["n_glue_nets"] > 0
    nl, meta = bn.load_tiled_netlist(cache_dir, mmap=False)
    inv_perm = np.load(cache_dir + "/inv_perm.npy")
    src_tile, _, _ = bn.parse_source_tile(src)
    n_src = src_tile.n_nodes

    glue_nets = bn._read_glue_tail(dst + ".nets", meta["n_glue_nets"], meta["n_glue_pins"])
    assert len(glue_nets) == meta["n_glue_nets"]
    base_pins = meta["n_pins_base"]
    for k, net in enumerate(glue_nets):
        (i0, j0, n0), (i1, j1, n1) = net["pins"]
        expected0 = inv_perm[(i0 * 2 + j0) * n_src + src_tile.name2local[n0]]
        expected1 = inv_perm[(i1 * 2 + j1) * n_src + src_tile.name2local[n1]]
        got0 = int(nl.pin2node[base_pins + 2 * k])
        got1 = int(nl.pin2node[base_pins + 2 * k + 1])
        assert (got0, got1) == (expected0, expected1)
        assert (i0, j0) != (i1, j1), "every glue net must span two distinct tiles"


def test_glue_tail_reader_retries_with_bigger_buffer(tmp_path):
    """Forces the initial tail-buffer to be far smaller than the glue
    section (a handful of bytes) so `_read_glue_tail` must exercise its
    doubling-retry loop before it finds the `glue0` marker."""
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=True)
    n = meta["n_glue_nets"]
    p = meta["n_glue_pins"]
    assert n > 0
    nets_small_buf = bn._read_glue_tail(dst + ".nets", n, p, initial_bytes=8)
    nets_normal = bn._read_glue_tail(dst + ".nets", n, p)
    assert nets_small_buf == nets_normal


def test_glue_net_degree_other_than_2_is_rejected(tmp_path):
    lines = ["NetDegree : 3 glue0", "    t0_0/o0 I : 0 0",
            "    t0_1/o0 I : 0 0", "    t1_0/o0 I : 0 0"]
    with pytest.raises(ValueError, match="degree 3"):
        bn._parse_glue_lines(lines, n_glue_nets=1, n_glue_pins=3)


def test_glue_net_out_of_order_name_is_rejected():
    lines = ["NetDegree : 2 glue1", "    t0_0/o0 I : 0 0", "    t0_1/o0 I : 0 0"]
    with pytest.raises(ValueError, match="order mismatch"):
        bn._parse_glue_lines(lines, n_glue_nets=1, n_glue_pins=2)


def test_split_tile_prefix_rejects_unprefixed_name():
    with pytest.raises(ValueError, match="tile-prefix"):
        bn._split_tile_prefix("not_a_tiled_name")


# ---------------------------------------------------------------------------
# cheap invariants (module docstring)
# ---------------------------------------------------------------------------

def test_cheap_invariants_hold_with_glue(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=True)
    nl, meta = bn.load_tiled_netlist(cache_dir, mmap=False)

    net_degrees = np.diff(nl.flat_net2pin_start)
    assert int(net_degrees.sum()) == nl.pin2node.shape[0]      # sum(degrees) == NumPins
    assert int(nl.pin2node.max()) < nl.num_physical             # max(pin2node) < NumNodes

    # per-tile net count is constant (base nets only; glue nets are the
    # tail and not attributed to a single tile)
    n_src_nets = meta["n_src_nets"]
    n_tiles = meta["R"] * meta["C"]
    assert meta["n_nets_base"] == n_tiles * n_src_nets


# ---------------------------------------------------------------------------
# load_tiled_netlist mmap mode
# ---------------------------------------------------------------------------

def test_load_mmap_true_and_false_agree(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=1, C=2, with_glue=True)
    nl_mmap, meta_mmap = bn.load_tiled_netlist(cache_dir, mmap=True)
    nl_mem, meta_mem = bn.load_tiled_netlist(cache_dir, mmap=False)
    assert meta_mmap == meta_mem
    np.testing.assert_array_equal(np.asarray(nl_mmap.node_x), np.asarray(nl_mem.node_x))
    np.testing.assert_array_equal(np.asarray(nl_mmap.pin2node), np.asarray(nl_mem.pin2node))


# ---------------------------------------------------------------------------
# verify_against_bookshelf
# ---------------------------------------------------------------------------

def test_verify_against_bookshelf_window_mode_passes_on_correct_array(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=True)
    result = bn.verify_against_bookshelf(cache_dir, dst, mode="window")
    assert result["ok"] is True, result["errors"]
    assert result["n_mismatched_nodes"] == 0
    assert result["n_mismatched_pl"] == 0
    assert result["n_mismatched_pins"] == 0
    assert result["n_checked_glue_nets"] == meta["n_glue_nets"]
    assert result["n_mismatched_glue_nets"] == 0


def test_verify_against_bookshelf_full_mode_passes_on_correct_array(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=True)
    result = bn.verify_against_bookshelf(cache_dir, dst, mode="full")
    assert result["ok"] is True, result["errors"]
    assert result["n_checked_nodes"] == meta["n_nodes"]
    assert result["n_checked_pl"] == meta["n_nodes"]
    assert result["n_checked_nets"] == meta["n_nets"]
    # full mode's main .nets stream is not bounded by `window`, so it covers
    # the glue tail too (in addition to the always-on separate glue check
    # below) -- n_checked_pins == every pin in the array, base + glue.
    assert result["n_checked_pins"] == meta["n_pins"]
    assert result["n_mismatched_nodes"] == 0
    assert result["n_mismatched_pl"] == 0
    assert result["n_mismatched_nets"] == 0
    assert result["n_mismatched_pins"] == 0


def test_verify_against_bookshelf_catches_a_corrupted_pl_record(tmp_path):
    """RESULT-GATE-style rigor: verify_against_bookshelf must actually
    catch a deliberately-broken array, not just rubber-stamp a correct
    one."""
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=2, C=2, with_glue=False)
    with open(dst + ".pl") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("t0_0/o0 "):
            parts = line.split()
            parts[1] = str(int(parts[1]) + 1000)   # corrupt x position
            lines[i] = " ".join(parts) + "\n"
            break
    else:
        raise AssertionError("t0_0/o0 not found in .pl -- fixture changed?")
    with open(dst + ".pl", "w") as f:
        f.writelines(lines)

    result = bn.verify_against_bookshelf(cache_dir, dst, mode="full")
    assert result["ok"] is False
    assert result["n_mismatched_pl"] == 1
    assert any("t0_0/o0" in e for e in result["errors"])


def test_verify_against_bookshelf_catches_a_corrupted_net_pin(tmp_path):
    src, dst, cache_dir, meta = _build_cache(tmp_path, R=1, C=2, with_glue=False)
    with open(dst + ".nets") as f:
        text = f.read()
    # swap one base pin's node reference from t0_0/o1 to t0_1/o1 (still a
    # syntactically valid record, but structurally wrong for the cache).
    corrupted = text.replace("    t0_0/o1 I : -1 -1\n", "    t0_1/o1 I : -1 -1\n", 1)
    assert corrupted != text, "expected pin line not found -- fixture changed?"
    with open(dst + ".nets", "w") as f:
        f.write(corrupted)

    result = bn.verify_against_bookshelf(cache_dir, dst, mode="full")
    assert result["ok"] is False
    assert result["n_mismatched_pins"] >= 1


# ---------------------------------------------------------------------------
# source/manifest sha256 mismatch guard
# ---------------------------------------------------------------------------

def test_build_rejects_source_that_no_longer_matches_manifest_sha256(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    tb.tile(src, dst, R=1, C=1, seed=0)

    with open(src + ".nodes", "a") as f:
        f.write("extra_stray_line_that_changes_the_hash\n")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        bn.build_tiled_netlist_cache(dst + ".manifest.json", str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# equivalence proof (design draft R4 / T9's own instruction): replication
# vs `PlaceDB.read` on the real 1x2 array. This needs a real DREAMPlace
# PlaceDB.read of a 6.16M-node design and touches CUDA config fields --
# written per spec ("寫好不跑") but never run by the default suite; GPU is
# busy with a concurrent T8 run on this box regardless.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_replication_equals_placedb_read_on_real_1x2_array_movable_first():
    """1x2 array built two independent ways must agree after both are
    reduced to the same movable-first-permuted representation:
      (a) this module's `build_tiled_netlist_cache` off
          `results/m4/bench/arrays/1x2_n2/1x2_n2.manifest.json`;
      (b) `ioplace.netlist.load_netlist` -> `PlaceDB.read` off
          `benchmarks/ispd25/synthetic_1x2_n2.json` (already movable-first
          by DREAMPlace's own PlaceDB convention).
    Compared as per-net node-ID sets (order-independent) rather than raw
    pin-index arrays, since PlaceDB's own internal pin ordering is not
    contractually required to match this module's net-major convention --
    only net/node connectivity and node positions/sizes need to agree.
    """
    import tempfile

    from ioplace.netlist import load_netlist

    manifest_path = "results/m4/bench/arrays/1x2_n2/1x2_n2.manifest.json"
    config_json = "benchmarks/ispd25/synthetic_1x2_n2.json"

    with tempfile.TemporaryDirectory() as cache_dir:
        meta = bn.build_tiled_netlist_cache(manifest_path, cache_dir)
        nl_rep, _ = bn.load_tiled_netlist(cache_dir, mmap=True)

    nl_pdb, _placedb, _params = load_netlist(config_json)

    assert nl_rep.num_movable == nl_pdb.num_movable
    assert nl_rep.num_physical == nl_pdb.num_physical
    assert nl_rep.num_nets == nl_pdb.num_nets

    def net_node_id_sets(nl):
        sets = []
        for net_i in range(nl.num_nets):
            lo, hi = nl.flat_net2pin_start[net_i], nl.flat_net2pin_start[net_i + 1]
            pins = nl.flat_net2pin[lo:hi]
            sets.append(frozenset(int(nl.pin2node[p]) for p in pins))
        return sets

    assert sorted(net_node_id_sets(nl_rep), key=lambda s: (len(s), sorted(s))) == \
        sorted(net_node_id_sets(nl_pdb), key=lambda s: (len(s), sorted(s)))

    order_rep = np.argsort(np.lexsort((nl_rep.node_y, nl_rep.node_x)))
    order_pdb = np.argsort(np.lexsort((nl_pdb.node_y, nl_pdb.node_x)))
    np.testing.assert_array_equal(np.sort(nl_rep.node_x), np.sort(nl_pdb.node_x))
    np.testing.assert_array_equal(np.sort(nl_rep.node_y), np.sort(nl_pdb.node_y))
