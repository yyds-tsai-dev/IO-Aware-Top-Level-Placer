import numpy as np
import pytest
from ioplace.netlist import Netlist
from ioplace.partition.mtkahypar_runner import partition_netlist
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


def test_hierarchy_depth_caps_when_name_is_shallower_than_depth():
    """A name with fewer path components than `depth` uses its full
    available hierarchy (all components but the leaf), not a truncated
    first-`depth` slice and not the "" catch-all reserved for names with
    no '/' at all. "top/mid/x*" has exactly depth=3 components (capped to
    "top/mid"); "top/y*" has only 2 (capped to "top"): both are shallower
    than-or-equal-to depth, so this pins the fallback branch of `_prefix`,
    not its `len(parts) > depth` branch."""
    names = np.array([b"top/mid/x0", b"top/mid/x1", b"top/y0", b"top/y1"])
    part = membership.hierarchy_membership(names, 4, 2, depth=3)
    assert part[0] == part[1]      # "top/mid/x0", "top/mid/x1" -> "top/mid"
    assert part[2] == part[3]      # "top/y0", "top/y1" -> "top" (capped)
    assert part[0] != part[2]


def test_hierarchy_ignores_names_past_num_movable():
    """`node_names` may run past `num_movable` (real netlists append
    terminal/IO names after the movable prefix); hierarchy_membership must
    key off the first `num_movable` entries. A slice bug that took the last
    `num_movable` instead would read "b/y*" and "z/term*" here instead of
    "a/x*" and "b/y*", changing every label."""
    base = _names([("a/x", 5), ("b/y", 3)])          # exactly num_movable=8
    tail = _names([("z/term", 4)])                     # never movable
    names_with_tail = np.concatenate([base, tail])      # 12 names, 8 movable
    part_with_tail = membership.hierarchy_membership(names_with_tail, 8, 2)
    part_reference = membership.hierarchy_membership(base, 8, 2)
    assert np.array_equal(part_with_tail, part_reference)


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


def _two_clique_netlist_with_terminals(n_per=20, n_term=4):
    """Same two-clique hypergraph as `_two_clique_netlist`, plus `n_term`
    extra fixed (non-movable) physical nodes appended after the movable
    prefix and never referenced by any net, so num_movable < num_physical --
    the shape the brief warned `mtkahypar_membership`'s
    `part[:nl.num_movable]` truncation has to get right."""
    nl = _two_clique_netlist(n_per)
    pad = lambda arr, extra: np.concatenate([arr, extra])
    return Netlist(
        node_x=pad(nl.node_x, np.zeros(n_term)),
        node_y=pad(nl.node_y, np.zeros(n_term)),
        node_size_x=pad(nl.node_size_x, np.ones(n_term)),
        node_size_y=pad(nl.node_size_y, np.ones(n_term)),
        num_movable=nl.num_movable, num_terminals=n_term, num_terminal_NIs=0,
        pin_offset_x=nl.pin_offset_x, pin_offset_y=nl.pin_offset_y,
        pin2node=nl.pin2node, pin2net=nl.pin2net,
        flat_net2pin=nl.flat_net2pin, flat_net2pin_start=nl.flat_net2pin_start,
        xl=nl.xl, yl=nl.yl, xh=nl.xh, yh=nl.yh)


def test_mtkahypar_membership_truncates_to_movable_not_physical():
    pytest.importorskip("mtkahypar")
    nl = _two_clique_netlist_with_terminals()
    assert nl.num_movable < nl.num_physical
    full = partition_netlist(nl, 2, seed=0)      # untruncated, num_physical-long
    a = membership.mtkahypar_membership(nl, 2, seed=0)
    assert a.dtype == np.int32 and a.shape == (nl.num_movable,)
    assert a.shape != (nl.num_physical,)
    # the movable prefix, in order -- not the terminal suffix and not the
    # full physical-length array
    assert np.array_equal(a, full[:nl.num_movable])


def test_build_membership_dispatches_and_rejects_unknown_sources():
    names = _names([("a/x", 4), ("b/y", 4)])
    part = membership.build_membership(
        "hierarchy", nl=None, node_names=names, num_movable=8, k=2)
    assert part.shape == (8,)
    with pytest.raises(ValueError, match="membership source"):
        membership.build_membership(
            "magic", nl=None, node_names=names, num_movable=8, k=2)
