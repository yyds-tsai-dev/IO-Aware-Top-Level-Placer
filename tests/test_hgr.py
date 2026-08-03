import numpy as np
import pytest
from ioplace.partition.hgr import write_hgr
from tests.test_netlist import make_tiny_netlist

def test_write_hgr(tmp_path):
    nl = make_tiny_netlist()
    p = tmp_path / "t.hgr"
    n = write_hgr(nl, str(p))
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "2 4"          # 2 nets, 4 nodes
    assert lines[1] == "1 2"          # n0 = {c0,c1} -> 1-indexed
    assert lines[2] == "2 3 4"        # n1 = {c1,c2,f3}
    assert n == 2

def test_partition_two_clusters():
    mtk = pytest.importorskip("mtkahypar")
    from ioplace.partition.mtkahypar_runner import partition_netlist
    from ioplace.netlist import Netlist
    # two 4-clique clusters bridged by 1 net -> k=2 should separate them
    p2n, pins = [], []
    nets = [[0,1],[1,2],[2,3],[0,3],[4,5],[5,6],[6,7],[4,7],[3,4]]
    for i, ns in enumerate(nets):
        pins += ns; p2n += [i]*len(ns)
    start = np.searchsorted(np.array(p2n), np.arange(len(nets)+1))
    nl = Netlist(node_x=np.zeros(8), node_y=np.zeros(8),
                 node_size_x=np.ones(8), node_size_y=np.ones(8),
                 num_movable=8, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                 pin2node=np.array(pins, np.int32), pin2net=np.array(p2n, np.int32),
                 flat_net2pin=np.arange(len(pins), dtype=np.int32),
                 flat_net2pin_start=start.astype(np.int32),
                 xl=0., yl=0., xh=10., yh=10.)
    parts = partition_netlist(nl, 2, seed=1)
    assert len(parts) == 8 and set(parts) == {0, 1}
    assert len(set(parts[:4])) == 1 and len(set(parts[4:])) == 1
