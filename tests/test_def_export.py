import json, os
from types import SimpleNamespace

import numpy as np
import pytest

from ioplace.export.def_export import export_def
from ioplace.regions import RegionSet

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
FFT1 = os.path.join(DP, "install", "test", "ispd2015", "lefdef", "mgc_fft_1.json")


def test_export_def_rejects_unresolved_scale_factor(tmp_path):
    """spec sec 7.3/L-Q2-a: ISPD2015 configs leave scale_factor at 0.0 until
    placedb.initialize() resolves it (PlaceDB.py:764-767) -- calling
    export_def before that must fail loudly, not silently divide by zero or
    fall back to an incorrect hardcoded 1.0. This check happens before
    export_def ever touches `placedb`, so a bare params stub is enough."""
    params = SimpleNamespace(shift_factor=[0.0, 0.0], scale_factor=0.0)
    with pytest.raises(ValueError, match="scale_factor"):
        export_def(None, params, np.zeros(1), np.zeros(1), str(tmp_path), None)


def _reread_out_def(config_json, out_def):
    """Load `out_def` back through DREAMPlace's own reader (same code path
    as `run_placement._load_dreamplace`, minus the DP-off flags which don't
    matter for a read-only re-parse), pointed at our exported DEF instead of
    the original one. This is the "golden oracle" for the round-trip
    assertions below: rather than hand-roll a DEF text parser, reuse the
    already-trusted parser DREAMPlace itself uses, so a divergence in
    COMPONENTS/NETS content is caught the same way a real consumer would hit
    it."""
    from ioplace.dreamplace_env import setup_dreamplace
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))
    try:
        params.load(config_json)
        params.def_input = out_def
        placedb = PlaceDB.PlaceDB()
        placedb.read(params)
        placedb.initialize(params)
        return params, placedb
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_export_def_mgc_fft_1_golden(tmp_path):
    """S1 golden test (design spec sec 10 S1 row): out.def / regions.json /
    netmap.json / coord.json on a real ISPD2015 LEF/DEF design. No GP/LG is
    needed for this -- the round-trip properties under test (COMPONENTS
    count, coordinate mapping, net-name bijection) hold on any valid
    node_x/node_y, so this uses placedb's as-initialized positions and stays
    far cheaper than a full placement run."""
    from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for

    params, placedb = _load_dreamplace(FFT1)
    placedb.initialize(params)   # resolves params.scale_factor (was 0.0)

    n_phys = placedb.num_physical_nodes
    node_x = np.array(placedb.node_x[:n_phys], dtype=np.float64)
    node_y = np.array(placedb.node_y[:n_phys], dtype=np.float64)

    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs = get_regions_for(die, 8, "grid", 0)

    out_dir = str(tmp_path / "def_export")
    paths = export_def(placedb, params, node_x, node_y, out_dir, rs)
    assert set(paths) == {"out_def", "regions_json", "netmap_json", "coord_json"}
    for p in paths.values():
        assert os.path.exists(p)

    # -- netmap.json: {net_index: net_name} bijection with placedb.net_names
    # (spec sec 7.2) --
    net_names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in placedb.net_names]
    netmap = json.load(open(paths["netmap_json"]))
    assert len(netmap) == len(net_names)
    assert [netmap[str(i)] for i in range(len(net_names))] == net_names

    # -- regions.json: round-trips through RegionSet.to_json/from_json,
    # carries the same K-grid geometry used for this run (R1 -- kept out of
    # the DEF itself, spec sec 5.1) --
    rs_loaded = RegionSet.from_json(paths["regions_json"])
    assert rs_loaded.k == rs.k == 8
    assert rs_loaded.die == tuple(die)

    # -- coord.json: matches params/placedb, not a hardcoded 1.0 (spec sec
    # 7.3/L-Q2-a) --
    coord = json.load(open(paths["coord_json"]))
    assert coord["scale_factor"] == pytest.approx(float(params.scale_factor))
    assert coord["scale_factor"] != 0.0
    assert coord["shift_factor"] == [float(params.shift_factor[0]),
                                     float(params.shift_factor[1])]
    assert coord["def_units_per_micron"] == int(placedb.rawdb.defUnit())
    assert coord["xl"] == pytest.approx(float(placedb.xl))
    assert coord["yl"] == pytest.approx(float(placedb.yl))
    assert coord["xh"] == pytest.approx(float(placedb.xh))
    assert coord["yh"] == pytest.approx(float(placedb.yh))

    # -- out.def: re-read through DREAMPlace's own reader. DefWriter.cpp is
    # a passthrough of the input DEF outside COMPONENTS/ROW (spec sec 3.3),
    # so this both proves the red-line instance/net-set invariant and
    # exercises that passthrough claim directly, rather than trusting it. --
    p2, db2 = _reread_out_def(FFT1, paths["out_def"])

    assert db2.num_movable_nodes == placedb.num_movable_nodes
    assert db2.num_terminals == placedb.num_terminals
    assert db2.num_terminal_NIs == placedb.num_terminal_NIs

    # DEF COMPONENTS = movable + fixed cells only; IO pins (terminal_NI) are
    # explicitly excluded from vNodeIndex on the C++ side (PlaceDB.cpp
    # add_def_pin: "exclude io pins"), so the instance-set check is scoped
    # to the first num_movable+num_terminals names, not all num_physical.
    n_comp = placedb.num_movable_nodes + placedb.num_terminals
    names1 = [n.decode() for n in placedb.node_names[:n_comp]]
    names2 = [n.decode() for n in db2.node_names[:n_comp]]
    assert names1 == names2                       # instance set unchanged

    net_names2 = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in db2.net_names]
    assert net_names2 == net_names                 # net set unchanged

    # -- coordinate round trip (spec sec 7.3/sec 10 S1 row): movable-cell
    # DEF positions mapped back to DBU must match the exported node_x/
    # node_y to within 1 DBU. Compared in DBU space (not placedb's internal
    # scaled space) so this doesn't depend on db2's independently-derived
    # shift/scale factors exactly matching params's bit-for-bit -- only that
    # they describe the same physical mapping (same tech.lef site_width).
    n_mov = placedb.num_movable_nodes
    dbu_x1 = node_x[:n_mov] / float(params.scale_factor) + params.shift_factor[0]
    dbu_y1 = node_y[:n_mov] / float(params.scale_factor) + params.shift_factor[1]
    got_x = np.array(db2.node_x[:n_mov], dtype=np.float64)
    got_y = np.array(db2.node_y[:n_mov], dtype=np.float64)
    dbu_x2 = got_x / float(p2.scale_factor) + p2.shift_factor[0]
    dbu_y2 = got_y / float(p2.scale_factor) + p2.shift_factor[1]
    assert np.max(np.abs(dbu_x1 - dbu_x2)) <= 1.0
    assert np.max(np.abs(dbu_y1 - dbu_y2)) <= 1.0
