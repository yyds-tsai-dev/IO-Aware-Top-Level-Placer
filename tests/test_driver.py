import json, os
import numpy as np
import pytest
from ioplace.drivers.run_placement import get_regions_for

def test_get_regions_grid_k16():
    rs = get_regions_for((0., 0., 1000., 1000.), 16, "grid", 0)
    assert rs.k == 16
    rs.validate()

def test_get_regions_scale_equivariant():
    rs1 = get_regions_for((0., 0., 1000., 1000.), 8, "slicing", 5)
    rs2 = get_regions_for((0., 0., 2000., 2000.), 8, "slicing", 5)
    for a, b in zip(rs1.regions, rs2.regions):
        assert np.allclose(np.asarray(a.rects) * 2.0, np.asarray(b.rects))

@pytest.mark.slow
def test_run_flat_simple(tmp_path):
    from ioplace.drivers.run_placement import run_flat
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    out = str(tmp_path / "simple_flat.json")
    res = run_flat(os.path.join(root, "install/test/simple.json"), 4, "grid", 0, out)
    assert res["io_count"] >= 0 and res["hpwl"] > 0
    assert os.path.exists(out) and os.path.exists(out + ".npz")
    saved = json.load(open(out))
    assert saved["mode"] == "flat"

@pytest.mark.slow
def test_load_dreamplace_forces_dp_off():
    # adaptec1.json 出廠 detailed_place_flag=1;driver 必須強制為 0(GP+LG only protocol)
    from ioplace.drivers.run_placement import _load_dreamplace
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    cfg = os.path.join(root, "install/test/ispd2005/adaptec1.json")
    params, placedb = _load_dreamplace(cfg)
    assert params.detailed_place_flag == 0
    assert params.detailed_place_engine == ""
    assert placedb.num_movable_nodes > 0
