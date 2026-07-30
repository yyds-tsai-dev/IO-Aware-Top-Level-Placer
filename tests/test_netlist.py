import numpy as np
import pytest
from ioplace.netlist import Netlist, pin_positions

def make_tiny_netlist():
    # 3 cells + 1 fixed, 2 nets: n0={c0,c1}, n1={c1,c2,f3}
    return Netlist(
        node_x=np.array([10., 30., 50., 90.]), node_y=np.array([10., 10., 40., 90.]),
        node_size_x=np.ones(4), node_size_y=np.ones(4),
        num_movable=3, num_terminals=1, num_terminal_NIs=0,
        pin_offset_x=np.zeros(5), pin_offset_y=np.zeros(5),
        pin2node=np.array([0, 1, 1, 2, 3], dtype=np.int32),
        pin2net=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        flat_net2pin=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        flat_net2pin_start=np.array([0, 2, 5], dtype=np.int32),
        xl=0., yl=0., xh=100., yh=100.)

def test_netlist_properties():
    nl = make_tiny_netlist()
    assert nl.num_physical == 4 and nl.num_nets == 2
    assert list(nl.net_degrees) == [2, 3]

def test_pin_positions_default_and_override():
    nl = make_tiny_netlist()
    px, py = pin_positions(nl)
    assert px[0] == 10. and px[2] == 30.
    px2, _ = pin_positions(nl, node_x=nl.node_x + 5., node_y=nl.node_y)
    assert px2[0] == 15.

@pytest.mark.slow
def test_load_netlist_simple():
    from ioplace.netlist import load_netlist
    import os
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    nl, placedb, params = load_netlist(os.path.join(root, "install/test/simple.json"))
    assert nl.num_movable > 0 and nl.num_nets > 0
    assert len(nl.flat_net2pin_start) == nl.num_nets + 1
    assert nl.net_degrees.min() >= 1
