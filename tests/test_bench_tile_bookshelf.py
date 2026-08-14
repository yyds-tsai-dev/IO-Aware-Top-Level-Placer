import os

from ioplace.bench import tile_bookshelf as tb

# Toy source design shared by the T4 tests: 5 movable cells (o0..o4), 1 IO
# pin (p0, terminal_NI), 1 fixed macro (f0, terminal), 3 nets, 2 rows.
# xl/xr/yl/yr of the row bounding box are 0/6/0/4 -> W=6, H=4.
_TOY_NODES = """UCLA nodes 1.0

NumNodes : 7
NumTerminals : 2

o0 2 2
o1 2 2
o2 2 2
o3 2 2
o4 2 2
p0 0 0 terminal_NI
f0 4 4 terminal
"""

_TOY_PL = """UCLA pl 1.0

o0 0 0 : N
o1 2 0 : N
o2 4 0 : N
o3 0 2 : N
o4 2 2 : N
p0 10 10 : N /FIXED_NI
f0 6 0 : N /FIXED
"""

_TOY_SCL = """UCLA scl 1.0

NumRows : 2

CoreRow Horizontal
\tCoordinate : 0
\tHeight : 2
\tSitewidth : 1
\tSitespacing : 1
\tSiteorient : 0
\tSitesymmetry : 1
\tSubrowOrigin : 0 NumSites : 6
End
CoreRow Horizontal
\tCoordinate : 2
\tHeight : 2
\tSitewidth : 1
\tSitespacing : 1
\tSiteorient : 0
\tSitesymmetry : 1
\tSubrowOrigin : 0 NumSites : 6
End
"""

_TOY_NETS = """UCLA nets 1.0

NumNets : 3
NumPins : 7

NetDegree : 3 n0
    o0 O : -1 -1
    o1 I : -1 -1
    f0 I : 0 0
NetDegree : 2 n1
    o2 O : -1 -1
    o3 I : -1 -1
NetDegree : 2 n2
    o4 O : -1 -1
    p0 I : 0 0
"""

_TOY_WTS = "UCLA wts 1.0\n\n"

_TOY_AUX = "RowBasedPlacement : toy.nodes toy.nets toy.wts toy.pl toy.scl\n"


def write_toy_bookshelf(prefix):
    """Write the shared toy design at `<prefix>.{aux,nodes,nets,wts,pl,scl}`.
    Used by this file and by test_bench_glue_gen.py / test_bench_verify.py."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    with open(prefix + ".nodes", "w") as f:
        f.write(_TOY_NODES)
    with open(prefix + ".pl", "w") as f:
        f.write(_TOY_PL)
    with open(prefix + ".scl", "w") as f:
        f.write(_TOY_SCL)
    with open(prefix + ".nets", "w") as f:
        f.write(_TOY_NETS)
    with open(prefix + ".wts", "w") as f:
        f.write(_TOY_WTS)
    with open(prefix + ".aux", "w") as f:
        f.write(_TOY_AUX)
    return prefix


def test_2x2_counts_are_exactly_4x_source(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=2, C=2, seed=0)

    base = manifest["base"]
    assert base["source_n_nodes"] == 7
    assert base["source_n_terminals"] == 2
    assert base["source_n_nets"] == 3
    assert base["source_n_pins"] == 7
    assert base["source_n_rows"] == 2

    assert base["n_nodes"] == 7 * 4
    assert base["n_terminals"] == 2 * 4
    assert base["n_nets"] == 3 * 4
    assert base["n_pins"] == 7 * 4
    assert base["n_rows"] == 2 * 4
    assert manifest["glue"] == {"n_nets": 0, "n_pins": 0, "kernel": None}

    # cross-check against the actual files on disk, not just the manifest
    with open(dst + ".nodes") as f:
        body = [l for l in f if l.strip() and not l.startswith(("UCLA", "NumNodes", "NumTerminals"))]
    assert len(body) == 7 * 4
    with open(dst + ".pl") as f:
        body = [l for l in f if l.strip() and not l.startswith("UCLA")]
    assert len(body) == 7 * 4
    with open(dst + ".nets") as f:
        lines = [l for l in f if l.strip() and not l.startswith(("UCLA", "NumNets", "NumPins"))]
    assert sum(1 for l in lines if "NetDegree" in l) == 3 * 4
    assert sum(1 for l in lines if "NetDegree" not in l) == 7 * 4
    with open(dst + ".scl") as f:
        n_core_row = sum(1 for l in f if l.strip() == "CoreRow Horizontal")
    assert n_core_row == 2 * 4


def test_names_are_prefixed_t_i_j(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    tb.tile(src, dst, R=2, C=3, seed=0)
    with open(dst + ".nodes") as f:
        names = {l.split()[0] for l in f if l.strip() and not l.startswith(("UCLA", "NumNodes", "NumTerminals"))}
    expected_prefixes = {f"t{i}_{j}/" for i in range(2) for j in range(3)}
    got_prefixes = {n.rsplit("/", 1)[0] + "/" for n in names}
    assert got_prefixes == expected_prefixes
    assert all(n.split("/", 1)[1] in {"o0", "o1", "o2", "o3", "o4", "p0", "f0"} for n in names)


def test_coordinates_are_translated_by_i_w_j_h(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=2, C=2, seed=0)
    assert manifest["tile_width"] == 6
    assert manifest["tile_height"] == 4

    positions = {}
    with open(dst + ".pl") as f:
        for l in f:
            if not l.strip() or l.startswith("UCLA"):
                continue
            parts = l.split()
            positions[parts[0]] = (int(parts[1]), int(parts[2]))

    assert positions["t0_0/o0"] == (0, 0)
    assert positions["t1_0/o0"] == (0 + 1 * 6, 0)
    assert positions["t0_1/o0"] == (0, 0 + 1 * 4)
    assert positions["t1_1/o0"] == (1 * 6, 1 * 4)
    # a movable cell not at the source origin still gets the same offset
    assert positions["t1_1/o2"] == (4 + 6, 0 + 4)


def test_fixed_macro_and_io_pin_status_preserved_under_translation(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    tb.tile(src, dst, R=1, C=2, seed=0)

    with open(dst + ".nodes") as f:
        node_lines = {l.split()[0]: l for l in f if l.strip() and not l.startswith(("UCLA", "NumNodes", "NumTerminals"))}
    assert node_lines["t0_1/f0"].strip().endswith("terminal")
    assert not node_lines["t0_1/f0"].strip().endswith("terminal_NI")
    assert node_lines["t0_1/p0"].strip().endswith("terminal_NI")
    assert node_lines["t0_1/o0"].strip() == "t0_1/o0 2 2"

    with open(dst + ".pl") as f:
        pl_lines = {l.split()[0]: l.strip() for l in f if l.strip() and not l.startswith("UCLA")}
    assert pl_lines["t0_1/f0"] == "t0_1/f0 6 4 : N /FIXED"  # src (6,0) + (0*W, 1*H)=(0,4)
    assert pl_lines["t0_1/p0"] == "t0_1/p0 10 14 : N /FIXED_NI"


def test_row_overlap_is_caught(tmp_path):
    """A source with a deliberate row overlap must make tile() raise, not
    silently produce an array whose rows physically collide.

    (2026-08-15 M4 T6 bugfix 2: `assert_rows_no_overlap_no_gap` no longer
    rejects *gaps* -- a macro-obstructed real design legitimately has rows
    that don't cover its macros' footprints, so "no gap" is unsatisfiable
    for any macro-containing benchmark and isn't a construction defect; see
    that function's docstring. Only overlaps -- rows physically colliding,
    which *would* indicate a real translation bug -- are still checked.)"""
    src = str(tmp_path / "src" / "bad")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src + ".nodes", "w") as f:
        f.write(_TOY_NODES)
    with open(src + ".pl", "w") as f:
        f.write(_TOY_PL)
    with open(src + ".nets", "w") as f:
        f.write(_TOY_NETS)
    with open(src + ".wts", "w") as f:
        f.write(_TOY_WTS)
    with open(src + ".aux", "w") as f:
        f.write(_TOY_AUX.replace("toy.", "bad."))
    # second row starts at y=1 instead of y=2, while the first row (height 2)
    # still ends at y=2 -> a 1-unit overlap
    bad_scl = _TOY_SCL.replace("Coordinate : 2", "Coordinate : 1")
    with open(src + ".scl", "w") as f:
        f.write(bad_scl)

    dst = str(tmp_path / "out" / "arr")
    try:
        tb.tile(src, dst, R=1, C=2, seed=0)
        assert False, "expected AssertionError for a row overlap"
    except AssertionError as e:
        assert "overlap" in str(e)


def test_row_gap_is_allowed(tmp_path):
    """A source with a row gap (e.g. under a macro) must tile() cleanly --
    2026-08-15 M4 T6 bugfix 2: gaps are a legitimate feature of macro-
    containing floorplans, not a construction defect (see
    `assert_rows_no_overlap_no_gap`'s docstring)."""
    src = str(tmp_path / "src" / "gappy")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src + ".nodes", "w") as f:
        f.write(_TOY_NODES)
    with open(src + ".pl", "w") as f:
        f.write(_TOY_PL)
    with open(src + ".nets", "w") as f:
        f.write(_TOY_NETS)
    with open(src + ".wts", "w") as f:
        f.write(_TOY_WTS)
    with open(src + ".aux", "w") as f:
        f.write(_TOY_AUX.replace("toy.", "gappy."))
    # second row starts at y=3 instead of y=2 -> a 1-unit gap (e.g. a macro)
    gappy_scl = _TOY_SCL.replace("Coordinate : 2", "Coordinate : 3")
    with open(src + ".scl", "w") as f:
        f.write(gappy_scl)

    dst = str(tmp_path / "out" / "arr")
    tb.tile(src, dst, R=1, C=2, seed=0)  # must not raise


def test_same_seed_is_bit_for_bit_reproducible(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst1 = str(tmp_path / "out1" / "arr")
    dst2 = str(tmp_path / "out2" / "arr")
    m1 = tb.tile(src, dst1, R=2, C=2, seed=7)
    m2 = tb.tile(src, dst2, R=2, C=2, seed=7)
    assert m1["output_sha256"] == m2["output_sha256"]
    for suf in tb.CORE_SUFFIXES:
        with open(dst1 + "." + suf, "rb") as f1, open(dst2 + "." + suf, "rb") as f2:
            assert f1.read() == f2.read()


def test_manifest_records_source_and_output_sha256(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=1, C=1, seed=3)
    assert manifest["seed"] == 3
    assert manifest["R"] == 1 and manifest["C"] == 1
    assert set(manifest["source_sha256"]) == set(tb.CORE_SUFFIXES)
    assert set(manifest["output_sha256"]) == set(tb.CORE_SUFFIXES)
    for suf in tb.CORE_SUFFIXES:
        assert manifest["output_sha256"][suf] == tb.sha256_file(dst + "." + suf)
