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
