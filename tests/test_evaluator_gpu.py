import numpy as np
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate
from ioplace.evaluator_gpu import evaluate_gpu
from tests.test_evaluator_ref import DIE  # (0,0,100,100)

def _random_case(rng, n_cells=40, n_nets=25, max_d=12):
    from ioplace.netlist import Netlist
    nx_, ny_ = rng.uniform(0.5, 99.5, n_cells), rng.uniform(0.5, 99.5, n_cells)
    pins, p2n = [], []
    for net in range(n_nets):
        d = int(rng.integers(2, max_d))
        pins += list(rng.choice(n_cells, d, replace=False)); p2n += [net] * d
    p2n = np.array(p2n, np.int32); pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(n_nets + 1)).astype(np.int32)
    return Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                   node_size_y=np.ones(n_cells), num_movable=n_cells,
                   num_terminals=0, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                   pin2node=pins, pin2net=p2n,
                   flat_net2pin=np.arange(len(pins), dtype=np.int32),
                   flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gpu_matches_reference(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand


def test_gpu_matches_reference_large_net():
    """A net with degree > max_degree exercises the presence-lower-bound branch
    (GpuEvalContext.large_net_ids_t / per_net_crossings lower bound), which the
    5-seed test above never triggers (those nets all have degree < 12). Small case:
    max_degree=8 and one net with degree 20 forces exactly that net through the
    large-net path while the rest still go through the normal MST/FT path."""
    from ioplace.netlist import Netlist
    rng = np.random.default_rng(11)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    n_cells = 25
    nx_, ny_ = rng.uniform(0.5, 99.5, n_cells), rng.uniform(0.5, 99.5, n_cells)
    degree_plan = [2, 3, 4, 20, 6]  # net index 3 has degree 20 > max_degree=8
    pins, p2n = [], []
    for net, d in enumerate(degree_plan):
        pins += list(rng.choice(n_cells, d, replace=False))
        p2n += [net] * d
    p2n = np.array(p2n, np.int32)
    pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(len(degree_plan) + 1)).astype(np.int32)
    nl = Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                 node_size_y=np.ones(n_cells), num_movable=n_cells,
                 num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                 pin2node=pins, pin2net=p2n,
                 flat_net2pin=np.arange(len(pins), dtype=np.int32),
                 flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg, max_degree=8)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg, max_degree=8)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand
    assert gpu.large_net_lb == ref.large_net_lb
    assert gpu.large_net_lb > 0  # sanity: the large-net branch was actually exercised


def test_gpu_matches_reference_on_lattice_boundaries():
    """C1 regression (whole-branch review): GpuEvalContext._to_idx used to divide
    by the plain python floats self.cell_w/self.cell_h. CUDA compiles
    `tensor / python_float` as a reciprocal-multiply (x * (1/c)), and (1/c) is
    itself rounded -- so a coordinate that lands exactly on a lattice boundary can
    come out one ULP low (e.g. 2673.0 / (10692/512) -> 127.99999999999999, not
    128.0), which `.to(torch.int64)` truncates down to 127 instead of 128: an
    off-by-one lattice-cell mis-assignment exactly at region boundaries. Plain
    `tensor / tensor` (even a 0-dim one) instead correctly rounds, matching numpy.

    This die (10692x10680, lattice=512 -> cell_w=10692/512=20.8828125,
    cell_h=10680/512=20.859375) and a 4x4 grid partition (region boundaries at
    lattice index 128/256/384) are not arbitrary: empirically (see the C1 fix in
    evaluator_gpu.py) this exact cell_w mis-rounds 80 of the 511 possible
    lattice-boundary indices when divided as tensor/python-float, including
    indices 128 and 256 -- exactly the region-column boundaries hit below by
    node_x 2673.0 (=128*cell_w) and 5346.0 (=256*cell_w). Before the C1 fix this
    test fails (GPU under-counts crossings by 1 on both boundary-straddling nets,
    per_net_crossings [1,1,3] (ref) vs [0,0,3] (gpu)); after the fix it's exact.
    """
    from ioplace.netlist import Netlist
    DIE_ND = (0., 0., 10692., 10680.)
    rs = make_grid_regions(DIE_ND, 4, 4, lattice=512)
    rg = RegionGrid(rs)

    # 5 cells, one pin each (offset 0,0):
    #   0: (2673.0, 5000.0) -- x exactly on the col0/col1 region boundary (idx 128)
    #   1: (2000.0, 5000.0) -- safely inside col0, same row as cell 0
    #   2: (5346.0, 5000.0) -- x exactly on the col1/col2 region boundary (idx 256)
    #   3: (4700.0, 5000.0) -- safely inside col1, same row as cell 2
    #   4: (9000.0, 2670.0) -- y exactly on a row boundary (idx 128) but a
    #      non-triggering axis (cell_h happens not to mis-round here -- included to
    #      show the fix doesn't disturb the already-correct axis), x safely inside col3
    node_x = np.array([2673.0, 2000.0, 5346.0, 4700.0, 9000.0], dtype=np.float64)
    node_y = np.array([5000.0, 5000.0, 5000.0, 5000.0, 2670.0], dtype=np.float64)

    # net0 = {0,1} straddles the idx-128 boundary; net1 = {2,3} straddles idx-256;
    # net2 = {1,3,4} is a plain 3-pin net (no boundary pin) exercising MST with the
    # fix in place, alongside the two boundary-triggering 2-pin nets.
    pin2node = np.array([0, 1, 2, 3, 1, 3, 4], dtype=np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    flat_net2pin = np.arange(7, dtype=np.int32)
    flat_net2pin_start = np.array([0, 2, 4, 7], dtype=np.int32)

    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=np.ones(5), node_size_y=np.ones(5),
                 num_movable=5, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(7), pin_offset_y=np.zeros(7),
                 pin2node=pin2node, pin2net=pin2net,
                 flat_net2pin=flat_net2pin, flat_net2pin_start=flat_net2pin_start,
                 xl=0., yl=0., xh=10692., yh=10680.)

    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    assert ref.per_net_crossings.tolist() == [1, 1, 3]
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand


def test_gpu_matches_reference_k64():
    """8x8 region grid (K=64) -- the required 5-seed test above only covers K=16
    (4x4). This exercises the K<=64 bitmask-vectorization boundary GpuEvalContext
    now asserts on construction (pin_bm/passed_bm packed into a single int64)."""
    rng = np.random.default_rng(64)
    rg = RegionGrid(make_grid_regions(DIE, 8, 8, lattice=32))
    nl = _random_case(rng, n_cells=80, n_nets=50, max_d=20)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand
