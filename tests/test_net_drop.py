import json
import os
import subprocess
import sys

from ioplace.bench import net_drop
from ioplace.bench import tile_bookshelf as tb
from ioplace.bench import verify_bench

# Toy source design: 4 movable nodes (o0..o3, no terminals), 8 nets, each net
# connecting *all four* nodes (degree 4). Every node is referenced by every
# net -> dropping up to half the nets can never leave a node fully
# unreferenced, so `verify_bench.check_v0_structural`'s default
# "all_nodes_referenced" bar stays satisfiable regardless of which nets a
# given seed happens to pick.
_TOY_NODES = """UCLA nodes 1.0

NumNodes : 4
NumTerminals : 0

o0 2 2
o1 2 2
o2 2 2
o3 2 2
"""

_TOY_PL = """UCLA pl 1.0

o0 0 0 : N
o1 2 0 : N
o2 4 0 : N
o3 6 0 : N
"""

_TOY_SCL = """UCLA scl 1.0

NumRows : 1

CoreRow Horizontal
\tCoordinate : 0
\tHeight : 2
\tSitewidth : 1
\tSitespacing : 1
\tSiteorient : 0
\tSitesymmetry : 1
\tSubrowOrigin : 0 NumSites : 8
End
"""

_TOY_WTS = "UCLA wts 1.0\n\n"

_ONE_NET = """NetDegree : 4 {name}
    o0 O : -1 -1
    o1 I : -1 -1
    o2 I : -1 -1
    o3 I : -1 -1
"""

_TOY_NETS = "UCLA nets 1.0\n\nNumNets : 8\nNumPins : 32\n\n" + "".join(
    _ONE_NET.format(name=f"n{i}") for i in range(8)
)

_TOY_AUX = "RowBasedPlacement : toy8.nodes toy8.nets toy8.wts toy8.pl toy8.scl\n"


def write_toy_bookshelf(prefix):
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


def _node_names(nodes_path):
    names = []
    with open(nodes_path) as f:
        started = False
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumTerminals"):
                    started = True
                continue
            if not s:
                continue
            names.append(s.split()[0])
    return names


def test_deletion_count_and_manifest_correct(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy8"))
    dst = str(tmp_path / "out25" / "toy8")
    manifest = net_drop.drop_nets(src, dst, drop_fraction=0.25, seed=0)

    assert manifest["n_nets_source"] == 8
    assert manifest["n_nets_dropped"] == 2  # round(0.25 * 8)
    assert manifest["n_nets_kept"] == 6
    assert manifest["n_pins_source"] == 32
    assert manifest["n_pins_dropped"] == 8  # 2 nets * degree 4
    assert manifest["n_pins_kept"] == 24
    assert manifest["seed"] == 0
    assert manifest["drop_fraction"] == 0.25
    assert set(manifest["source_sha256"]) == set(tb.CORE_SUFFIXES)
    assert set(manifest["output_sha256"]) == set(tb.CORE_SUFFIXES)

    # cross-check against the actual .nets file on disk, not just the manifest
    n_nets_hdr, n_pins_hdr = tb._nets_header(dst + ".nets")
    assert n_nets_hdr == 6
    assert n_pins_hdr == 24
    with open(dst + ".nets") as f:
        body = [l for l in f if l.strip() and not l.startswith(("UCLA", "NumNets", "NumPins"))]
    assert sum(1 for l in body if l.startswith("NetDegree")) == 6
    assert sum(1 for l in body if not l.startswith("NetDegree")) == 24

    # a 50% drop on the same source
    dst50 = str(tmp_path / "out50" / "toy8")
    manifest50 = net_drop.drop_nets(src, dst50, drop_fraction=0.50, seed=0)
    assert manifest50["n_nets_dropped"] == 4
    assert manifest50["n_nets_kept"] == 4
    assert manifest50["n_pins_dropped"] == 16
    assert manifest50["n_pins_kept"] == 16


def test_same_seed_is_bit_for_bit_reproducible(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy8"))
    dst1 = str(tmp_path / "out1" / "toy8")
    dst2 = str(tmp_path / "out2" / "toy8")
    m1 = net_drop.drop_nets(src, dst1, drop_fraction=0.5, seed=7)
    m2 = net_drop.drop_nets(src, dst2, drop_fraction=0.5, seed=7)
    assert m1["output_sha256"] == m2["output_sha256"]
    for suf in tb.CORE_SUFFIXES:
        with open(dst1 + "." + suf, "rb") as f1, open(dst2 + "." + suf, "rb") as f2:
            assert f1.read() == f2.read()

    # a different seed is allowed to (and, on this fixture, does) pick a
    # different set of nets dropped -- reproducibility is per-seed, not
    # seed-independent.
    dst3 = str(tmp_path / "out3" / "toy8")
    m3 = net_drop.drop_nets(src, dst3, drop_fraction=0.5, seed=13)
    with open(dst1 + ".nets", "rb") as f1, open(dst3 + ".nets", "rb") as f3:
        assert f1.read() != f3.read()


def test_node_set_is_unchanged(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy8"))
    dst = str(tmp_path / "out" / "toy8")
    net_drop.drop_nets(src, dst, drop_fraction=0.5, seed=1)

    src_names = _node_names(src + ".nodes")
    dst_names = _node_names(dst + ".nodes")
    assert dst_names == src_names  # order-preserving verbatim copy
    assert set(dst_names) == {"o0", "o1", "o2", "o3"}

    # the .nodes file itself is byte-for-byte the source (verbatim copy)
    with open(src + ".nodes", "rb") as fa, open(dst + ".nodes", "rb") as fb:
        assert fa.read() == fb.read()


def test_output_is_readable_by_existing_bookshelf_reader(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy8"))
    dst = str(tmp_path / "out" / "toy8")
    net_drop.drop_nets(src, dst, drop_fraction=0.25, seed=0)

    # tile_bookshelf's own .aux/header reader
    paths = tb.read_aux(dst + ".aux")
    assert set(paths) == set(tb.CORE_SUFFIXES)
    for p in paths.values():
        assert os.path.exists(p)
    n_nodes_hdr, n_terminals_hdr = tb._nodes_header(paths["nodes"])
    assert n_nodes_hdr == 4
    assert n_terminals_hdr == 0

    # verify_bench's structural reader/checker (V0): headers match bodies,
    # every node still referenced by at least one surviving net, no row
    # overlap.
    result = verify_bench.check_v0_structural(dst)
    assert result["status"] == "ok", result["errors"]
    assert result["checks"]["nodes_header_matches_body"]
    assert result["checks"]["nets_header_matches_body"]
    assert result["checks"]["all_nodes_referenced_or_declared_unconnected"]
    assert result["checks"]["unreferenced_count"] == 0


def test_derive_dst_prefix_naming():
    assert (net_drop.derive_dst_prefix("results/m4/bench/netdrop",
                                        "results/m4/bench/mempool_group_export/mempool_group",
                                        0.25, 0)
            == "results/m4/bench/netdrop/mempool_group__drop0.25__seed0")
    # trailing slash on the source prefix's directory must not leak into the
    # derived basename, and a bare fraction (0.5, not 0.50) still formats to
    # 2 decimals so 25%/50% file names sort and read consistently.
    assert (net_drop.derive_dst_prefix("out", "src/toy8", 0.5, 7)
            == os.path.join("out", "toy8__drop0.50__seed7"))


def test_cli_source_drop_seed_out_dir_matches_derive_dst_prefix(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy8"))
    out_dir = str(tmp_path / "netdrop_out")
    proc = subprocess.run(
        [sys.executable, "-m", "ioplace.bench.net_drop",
         "--source", src, "--drop", "0.25", "--seed", "0", "--out-dir", out_dir],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads(proc.stdout)
    assert manifest["n_nets_dropped"] == 2

    dst_prefix = net_drop.derive_dst_prefix(out_dir, src, 0.25, 0)
    assert os.path.exists(dst_prefix + ".manifest.json")
    with open(dst_prefix + ".manifest.json") as f:
        on_disk_manifest = json.load(f)
    assert on_disk_manifest == manifest
