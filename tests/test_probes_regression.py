import numpy as np
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate
from tests.test_evaluator_gpu import _random_case

DIE = (0., 0., 100., 100.)

def _sdf_l1(x, y, rects):
    dx = torch.maximum(rects[:, 0][None] - x[:, None], x[:, None] - rects[:, 2][None])
    dy = torch.maximum(rects[:, 1][None] - y[:, None], y[:, None] - rects[:, 3][None])
    return dx.clamp(min=0) + dy.clamp(min=0) + torch.maximum(dx, dy).clamp(max=0)

def _toy(dev, tau, clamp):
    """1 net of 4 nodes, all deep inside region 0 -> softmax saturates to exactly 1.0."""
    rects = torch.tensor([[0., 0., 50., 100.], [50., 0., 100., 100.]], device=dev)
    x = torch.tensor([5., 6., 7., 8.], device=dev, requires_grad=True)
    y = torch.tensor([50., 50., 50., 50.], device=dev, requires_grad=True)
    p = torch.softmax(-_sdf_l1(x, y, rects) / tau, dim=1)
    pp = p if clamp is None else p.clamp(max=1 - clamp)
    ell = torch.log1p(-pp)
    S = ell.sum(dim=0, keepdim=True)
    L = (1 - torch.exp(S)).sum()
    g = torch.autograd.grad(L, [x, y], allow_unused=True)
    return g

def test_unguarded_naive_autograd_produces_nan():
    """RED evidence for design v2 R3: without a clamp/floor guard the product-form
    surrogate yields 0*inf = NaN exactly where cells sit deep inside a region."""
    gx, gy = _toy("cuda", tau=1.0, clamp=None)
    assert torch.isnan(gx).any() or torch.isnan(gy).any()

def test_guarded_path_is_nan_free():
    gx, gy = _toy("cuda", tau=1.0, clamp=1e-7)
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hard_lambda_sum_is_a_lower_bound_on_io_count(seed):
    """design v2 sec 3.2.5: sum_e (Lambda_e - 1) <= io_count always holds."""
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(rng)
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    bm = rg.pin_region_bitmask(nl, nl.node_x, nl.node_y)
    lam = np.array([bin(int(v)).count("1") for v in bm])
    deg2 = nl.net_degrees >= 2
    assert int(np.maximum(lam - 1, 0)[deg2].sum()) <= res.io_count
