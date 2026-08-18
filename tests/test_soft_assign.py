import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.regions import make_grid_regions, RegionSet, RegionSpec
from ioplace.region_grid import RegionGrid
from ioplace.ops.soft_assign import (rect_table, region_sdf_l1, softmax_stats,
                                     chunk_p_ell, d_star_from_m)

DIE = (0., 0., 100., 100.)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _t(a, dtype=torch.float64):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=DEV)

def _tables(rs):
    r, r2k = rect_table(rs)
    return _t(r), torch.as_tensor(r2k, device=DEV)

def test_rect_table_shapes_and_mapping():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    rects, r2k = rect_table(rs)
    assert rects.shape == (4, 4) and r2k.shape == (4,)
    assert list(r2k) == [0, 1, 2, 3]
    assert np.allclose(rects[0], [0., 0., 50., 50.])
    assert np.allclose(rects[3], [50., 50., 100., 100.])

def test_sdf_single_rect_analytic():
    """K=2 vertical split: P0=[0,50]x[0,100], P1=[50,100]x[0,100]."""
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    rects, r2k = _tables(rs)
    x = _t([25., 75., 60., 0., 50.]); y = _t([50., 50., 110., 0., 50.])
    d = region_sdf_l1(x, y, rects, r2k, 0, 2)
    assert d.shape == (5, 2)
    assert d[0].tolist() == pytest.approx([-25., 25.])   # inside P0, 25 from boundary
    assert d[1].tolist() == pytest.approx([25., -25.])
    assert d[2].tolist() == pytest.approx([20., 10.])    # outside die (corner), L1
    assert d[3].tolist() == pytest.approx([0., 50.])     # die corner, on P0 boundary
    assert d[4].tolist() == pytest.approx([0., 0.])      # exactly on the shared edge

def test_sdf_rectilinear_hard_min():
    """L-shaped region 0 = [0,10]x[0,20] U [10,20]x[0,10]; region 1 = the rest."""
    rs = RegionSet(die=(0., 0., 40., 20.), lattice=4, regions=[
        RegionSpec("L", np.array([[0., 0., 10., 20.], [10., 0., 20., 10.]])),
        RegionSpec("R", np.array([[10., 10., 20., 20.], [20., 0., 40., 20.]]))])
    rects, r2k = _tables(rs)
    assert rects.shape == (4, 4) and list(r2k.cpu().numpy()) == [0, 0, 1, 1]
    x = _t([15., 5., 15., 25.]); y = _t([15., 5., 5., 5.])
    d = region_sdf_l1(x, y, rects, r2k, 0, 1)[:, 0]
    assert d.tolist() == pytest.approx([5., -5., -5., 5.])

def test_sdf_chunk_slice_matches_full():
    rs = make_grid_regions(DIE, 4, 4, lattice=16)
    rects, r2k = _tables(rs)
    rng = np.random.default_rng(0)
    x = _t(rng.uniform(0, 100, 200)); y = _t(rng.uniform(0, 100, 200))
    full = region_sdf_l1(x, y, rects, r2k, 0, 16)
    for lo, hi in [(0, 4), (4, 9), (9, 16)]:
        assert torch.equal(region_sdf_l1(x, y, rects, r2k, lo, hi), full[:, lo:hi])

@pytest.mark.parametrize("chunk", [None, 1, 3, 16])
def test_softmax_stats_chunk_invariant(chunk):
    rs = make_grid_regions(DIE, 4, 4, lattice=16)
    rects, r2k = _tables(rs)
    rng = np.random.default_rng(1)
    x = _t(rng.uniform(0, 100, 500)); y = _t(rng.uniform(0, 100, 500))
    m, t, am = softmax_stats(x, y, rects, r2k, 16, tau=7.0, chunk=chunk)
    m0, t0, am0 = softmax_stats(x, y, rects, r2k, 16, tau=7.0, chunk=None)
    assert torch.equal(m, m0) and torch.equal(am, am0)
    assert torch.allclose(t, t0, rtol=0, atol=1e-12)

def test_t_excludes_argmax_and_never_cancels():
    """t must be summed over j != argmax; s-1 would lose all precision when the
    winner dominates (design v2 sec 2.2). gap=500 keeps exp(-gap) representable
    (~7.1e-218), so a naive s-1 implementation erases it (1 + 7.1e-218 == 1.0 in
    fp64) while direct summation over j != argmax preserves it — this is what
    makes the case discriminating."""
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    rects, r2k = _tables(rs)
    x = _t([25.]); y = _t([50.])
    m, t, am = softmax_stats(x, y, rects, r2k, 2, tau=0.1)   # gap 50 / 0.1 = 500
    assert int(am.item()) == 0
    assert t.item() > 0.0 and np.isfinite(t.item())
    assert t.item() == pytest.approx(np.exp(-500.0), rel=1e-6, abs=0.0)

def test_ell_matches_log1p_in_unsaturated_region():
    rs = make_grid_regions(DIE, 4, 4, lattice=16)
    rects, r2k = _tables(rs)
    rng = np.random.default_rng(2)
    x = _t(rng.uniform(20, 80, 300)); y = _t(rng.uniform(20, 80, 300))
    tau = 30.0
    m, t, am = softmax_stats(x, y, rects, r2k, 16, tau)
    sdf = region_sdf_l1(x, y, rects, r2k, 0, 16)
    p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
    ref = torch.softmax(-sdf / tau, dim=1)
    assert torch.allclose(p, ref, rtol=1e-10, atol=1e-12)
    assert torch.allclose(ell, torch.log1p(-ref), rtol=1e-6, atol=1e-8)

def test_ell_finite_and_bounded_when_saturated():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    rects, r2k = _tables(rs)
    x = _t([5., 10., 45.]); y = _t([50., 50., 50.])
    tau = 1e-4 * 50.0
    m, t, am = softmax_stats(x, y, rects, r2k, 2, tau)
    sdf = region_sdf_l1(x, y, rects, r2k, 0, 2)
    p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
    assert torch.isfinite(ell).all()
    assert float(ell.min()) >= -70.0

def test_argmax_recovers_hard_region_assignment():
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rg = RegionGrid(rs)
    rects, r2k = _tables(rs)
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.6, 99.4, 1000); ys = rng.uniform(0.6, 99.4, 1000)
    m, t, am = softmax_stats(_t(xs), _t(ys), rects, r2k, 16, tau=1e-3)
    assert np.array_equal(am.cpu().numpy(), rg.region_of_points(xs, ys).astype(np.int64))

def test_d_star_equals_min_sdf():
    rs = make_grid_regions(DIE, 4, 4, lattice=16)
    rects, r2k = _tables(rs)
    rng = np.random.default_rng(4)
    x = _t(rng.uniform(0, 100, 200)); y = _t(rng.uniform(0, 100, 200))
    tau = 3.0
    m, t, am = softmax_stats(x, y, rects, r2k, 16, tau)
    d = region_sdf_l1(x, y, rects, r2k, 0, 16)
    assert torch.allclose(d_star_from_m(m, tau), d.min(dim=1).values, rtol=1e-12, atol=1e-9)

def test_sdf_gradient_matches_finite_difference():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    rects, r2k = _tables(rs)
    x = _t([13.0, 71.0]).requires_grad_(True); y = _t([27.0, 62.0]).requires_grad_(True)
    d = region_sdf_l1(x, y, rects, r2k, 0, 4)
    (d.sum()).backward()
    gx = x.grad.clone()
    h = 1e-6
    for i in range(2):
        xp = x.detach().clone(); xp[i] += h
        xm = x.detach().clone(); xm[i] -= h
        fd = (region_sdf_l1(xp, y.detach(), rects, r2k, 0, 4).sum()
              - region_sdf_l1(xm, y.detach(), rects, r2k, 0, 4).sum()) / (2 * h)
        assert float(gx[i]) == pytest.approx(float(fd), abs=1e-5)

@pytest.mark.parametrize("tau", [1e-4 * 50.0, 10.0 * 50.0])
def test_no_nan_at_extreme_tau(tau):
    rs = make_grid_regions(DIE, 4, 4, lattice=16)
    rects, r2k = _tables(rs)
    rng = np.random.default_rng(5)
    x = _t(rng.uniform(0, 100, 400)); y = _t(rng.uniform(0, 100, 400))
    m, t, am = softmax_stats(x, y, rects, r2k, 16, tau)
    sdf = region_sdf_l1(x, y, rects, r2k, 0, 16)
    p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
    assert torch.isfinite(m).all() and torch.isfinite(t).all()
    assert torch.isfinite(p).all() and torch.isfinite(ell).all()
