import numpy as np
import pytest
from ioplace.diagnostics.probes_m4.exact_synthetic import exact_synthetic

def test_exact_counts_and_csr_nonmultiple():
    n = exact_synthetic(101, 7, 23, seed=4)
    assert n.num_physical == 101 and n.num_nets == 7
    assert len(n.pin2node) == len(n.pin2net) == len(n.flat_net2pin) == 23
    assert n.flat_net2pin_start[-1] == 23
    assert np.array_equal(n.flat_net2pin_start[1:] - n.flat_net2pin_start[:-1], n.net_degrees)
    for e in range(n.num_nets):
        assert len(set(n.pin2node[n.flat_net2pin_start[e]:n.flat_net2pin_start[e+1]])) == n.net_degrees[e]

def test_connectivity_same_across_layouts_but_coordinates_differ():
    a = exact_synthetic(100, 5, 25, seed=2, layout="local")
    b = exact_synthetic(100, 5, 25, seed=2, layout="uniform")
    assert np.array_equal(a.pin2node, b.pin2node)
    assert not np.array_equal(a.node_x, b.node_x)

@pytest.mark.parametrize("args", [(10, 2, 3), (10, 2, 200), (1, 2, 4)])
def test_invalid_degree_rejected(args):
    with pytest.raises(ValueError): exact_synthetic(*args)

def test_deterministic_and_cyclic_wrap():
    a = exact_synthetic(5, 2, 8, seed=0); b = exact_synthetic(5, 2, 8, seed=0)
    assert np.array_equal(a.pin2node, b.pin2node)
    assert np.all(a.pin2node < 5)

def test_invalid_layout_and_die():
    with pytest.raises(ValueError): exact_synthetic(10, 2, 4, layout="bad")
    with pytest.raises(ValueError): exact_synthetic(10, 2, 4, die=(0, 0, 0, 1))
