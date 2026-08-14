import os

import pytest

from ioplace.bench import export_bookshelf
from ioplace.bench import tile_bookshelf as tb

DP_ROOT = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")


@pytest.mark.slow
def test_export_adaptec1_is_readable_by_the_streaming_tiler(tmp_path):
    """T4 scope note: only verified here on a small case (adaptec1
    re-export). Round-trips export_bookshelf's output through
    tile_bookshelf.tile(R=1, C=1) (an identity tiling) to prove the two
    modules' Bookshelf dialect actually agrees -- sec 3.2's "(c) reads
    (a)" -- not just that export_bookshelf runs."""
    cfg = os.path.join(DP_ROOT, "install/test/ispd2005/adaptec1.json")
    out_prefix = str(tmp_path / "adaptec1_reexport" / "adaptec1")
    result = export_bookshelf.export(cfg, out_prefix)

    assert result["n_movable"] > 0
    assert result["n_nets"] > 0
    assert result["n_pins"] > 0
    assert result["write_s"] > 0
    for suf in tb.CORE_SUFFIXES:
        assert os.path.exists(out_prefix + "." + suf)

    dst_prefix = str(tmp_path / "retiled" / "adaptec1")
    manifest = tb.tile(out_prefix, dst_prefix, R=1, C=1, seed=0)
    assert manifest["base"]["n_nodes"] == result["n_physical"]
    assert manifest["base"]["n_nets"] == result["n_nets"]
    assert manifest["base"]["n_pins"] == result["n_pins"]


@pytest.mark.slow
def test_export_is_seed_deterministic(tmp_path):
    cfg = os.path.join(DP_ROOT, "install/test/ispd2005/adaptec1.json")
    out1 = str(tmp_path / "a" / "adaptec1")
    out2 = str(tmp_path / "b" / "adaptec1")
    r1 = export_bookshelf.export(cfg, out1, random_seed=42)
    r2 = export_bookshelf.export(cfg, out2, random_seed=42)
    assert r1["n_physical"] == r2["n_physical"]
    assert r1["n_nets"] == r2["n_nets"]
    assert r1["n_pins"] == r2["n_pins"]
    for suf in tb.CORE_SUFFIXES:
        assert tb.sha256_file(out1 + "." + suf) == tb.sha256_file(out2 + "." + suf)
