import numpy as np
import pytest
from ioplace.netlist import Netlist
from ioplace.producer import membership


def _names(spec):
    out = []
    for prefix, n in spec:
        out.extend([f"{prefix}{i}".encode() for i in range(n)])
    return np.array(out)


def test_hierarchy_groups_share_a_label_and_are_balanced():
    """LPT packing: a(10)->bucket0, b(6)->bucket1, c(4)->bucket1 (now 10),
    d(1)->bucket0 on the tie. Two buckets of 11 and 10."""
    names = _names([("a/x", 10), ("b/y", 6), ("c/z", 4), ("d", 1)])
    part = membership.hierarchy_membership(names, 21, 2)
    assert part.dtype == np.int32 and part.shape == (21,)
    assert len(set(part[:10].tolist())) == 1
    assert len(set(part[10:16].tolist())) == 1
    assert part[0] != part[10]
    assert sorted(np.bincount(part, minlength=2).tolist()) == [10, 11]


def test_hierarchy_is_deterministic():
    names = _names([("a/x", 7), ("b/y", 5), ("c/z", 3)])
    a = membership.hierarchy_membership(names, 15, 3)
    b = membership.hierarchy_membership(names, 15, 3)
    assert np.array_equal(a, b)


def test_hierarchy_depth_splits_deeper_prefixes():
    names = np.array([b"top/a/x0", b"top/a/x1", b"top/b/y0", b"top/b/y1"])
    with pytest.raises(ValueError):
        membership.hierarchy_membership(names, 4, 2, depth=1)   # one prefix only
    part = membership.hierarchy_membership(names, 4, 2, depth=2)
    assert part[0] == part[1] and part[2] == part[3] and part[0] != part[2]


def test_hierarchy_rejects_fewer_prefix_groups_than_k():
    names = _names([("a/x", 4), ("b/y", 4)])
    with pytest.raises(ValueError, match="prefix groups"):
        membership.hierarchy_membership(names, 8, 4)


def test_hierarchy_accepts_str_names_too():
    names = ["a/x0", "a/x1", "b/y0", "b/y1"]
    part = membership.hierarchy_membership(names, 4, 2)
    assert part[0] == part[1] and part[0] != part[2]


def _two_clique_netlist(n_per=20):
    """Two 20-node cliques joined by a single net -- a hypergraph any
    partitioner must cut in exactly one place."""
    nets = []
    for base in (0, n_per):
        for i in range(n_per - 1):
            nets.append([base + i, base + i + 1])
        nets.append(list(range(base, base + n_per)))
    nets.append([0, n_per])
    pins, p2n, p2e, start = [], [], [], [0]
    for e, nodes in enumerate(nets):
        pins.extend(nodes)
        p2n.extend(nodes)
        p2e.extend([e] * len(nodes))
        start.append(len(p2n))
    n = 2 * n_per
    return Netlist(
        node_x=np.zeros(n), node_y=np.zeros(n),
        node_size_x=np.ones(n), node_size_y=np.ones(n),
        num_movable=n, num_terminals=0, num_terminal_NIs=0,
        pin_offset_x=np.zeros(len(p2n)), pin_offset_y=np.zeros(len(p2n)),
        pin2node=np.array(p2n, dtype=np.int32),
        pin2net=np.array(p2e, dtype=np.int32),
        flat_net2pin=np.arange(len(p2n), dtype=np.int32),
        flat_net2pin_start=np.array(start, dtype=np.int32),
        xl=0.0, yl=0.0, xh=100.0, yh=100.0)


def test_mtkahypar_membership_shape_range_and_determinism():
    pytest.importorskip("mtkahypar")
    nl = _two_clique_netlist()
    a = membership.mtkahypar_membership(nl, 2, seed=0)
    assert a.dtype == np.int32 and a.shape == (nl.num_movable,)
    assert int(a.min()) >= 0 and int(a.max()) < 2
    b = membership.mtkahypar_membership(nl, 2, seed=0)
    assert np.array_equal(a, b)


def test_build_membership_dispatches_and_rejects_unknown_sources():
    names = _names([("a/x", 4), ("b/y", 4)])
    part = membership.build_membership(
        "hierarchy", nl=None, node_names=names, num_movable=8, k=2)
    assert part.shape == (8,)
    with pytest.raises(ValueError, match="membership source"):
        membership.build_membership(
            "magic", nl=None, node_names=names, num_movable=8, k=2)
