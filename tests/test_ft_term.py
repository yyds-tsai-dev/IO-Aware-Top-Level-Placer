import numpy as np
import pytest
import torch

from ioplace.ops.soft_assign import rect_table
from ioplace.ops.io_term import IoTerm, IoTermRef, build_net_node_csr
from ioplace.ops.ft_term import FtTerm, FtTermRef
from ioplace.regions import make_grid_regions
from tests.test_io_term import _nl


def _make(k=4, chunk=2, fixed=0, fillers=0, w_mode="unit"):
    rs = make_grid_regions((0., 0., 100., 100.), k, 1, lattice=10)
    nl = _nl([(17., 50.), (53., 50.), (83., 50.)], [[0, 1, 2]],
             n_extra_nodes=fillers)
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    common = dict(csr=csr, rects=rects, rect2region=r2k, K=k,
                  num_movable=3-fixed, num_physical=nl.num_physical,
                  num_nodes=nl.num_physical, device="cpu", w_mode=w_mode)
    prod = IoTerm(**common, chunk_budget=max(1, chunk) * max(1, len(csr.flat_net2node)))
    ref = IoTermRef(**common)
    D = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :]).astype(np.float64)
    return nl, prod, ref, D


def _pos(nl, fillers=0, requires=True):
    n = nl.num_physical
    p = torch.zeros(2*n, dtype=torch.float64)
    p[:nl.num_physical] = torch.as_tensor(nl.node_x)
    p[n:n+nl.num_physical] = torch.as_tensor(nl.node_y)
    return p.requires_grad_(requires)


def test_nonzero_ft_and_ft_only_are_not_zero():
    nl, io, _, D = _make(k=4, chunk=1)
    ft = FtTerm(io, D); ft.set_home([0])
    p = _pos(nl)
    got = ft.ft_only(p, 8.)
    assert float(got.detach()) > 0.0
    expected = ft(p, 8., 1., 1.) - io(p, 8., 1.)
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_production_reference_value_and_gradient(chunk):
    nl, io, ioref, D = _make(k=4, chunk=chunk)
    a, b = FtTerm(io, D), FtTermRef(ioref, D)
    a.set_home([0]); b.set_home([0])
    pa, pb = _pos(nl), _pos(nl)
    va = a(pa, 7., 1.3, .8, .2, .4, 1.1)
    vb = b(pb, 7., 1.3, .8, .2, .4, 1.1)
    va.backward(); vb.backward()
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)


def test_zero_kappa_exact_io_bypass_without_home_and_nonunit_weights():
    nl, io, _, D = _make(k=4, chunk=1, w_mode="inv_deg")
    ft = FtTerm(io, D); p = _pos(nl)
    got = ft(p, 9., 2.5, 0., .7, .3, 2.)
    ref = io(p, 9., 2.5, .7, .3, 2.)
    assert torch.equal(got, ref)
    got.backward(); g = p.grad.clone(); p.grad.zero_(); ref.backward()
    assert torch.equal(g, p.grad)


def test_fixed_and_filler_gradients_zero_but_value_participates():
    nl, io, _, D = _make(k=4, chunk=2, fixed=1, fillers=2)
    ft = FtTerm(io, D); ft.set_home([0]); p = _pos(nl, 2)
    ft(p, 6., 1., 1.).backward()
    n = nl.num_physical
    assert torch.equal(p.grad[2], torch.zeros_like(p.grad[2]))
    assert torch.equal(p.grad[n+3:], torch.zeros_like(p.grad[n+3:]))


def test_home_is_snapshot_between_forward_and_backward():
    nl, io, _, D = _make(k=4, chunk=1)
    ft = FtTerm(io, D); ft.set_home([0]); p = _pos(nl)
    y = ft.ft_only(p, 5.)
    ft.set_home([3])
    y.backward(); g = p.grad.clone()
    ft.set_home([0]); q = _pos(nl); ft.ft_only(q, 5.).backward()
    assert torch.allclose(g, q.grad, atol=1e-10, rtol=1e-10)


def test_ft_is_independent_of_io_weight():
    nl, io1, _, D = _make(k=4, chunk=2, w_mode="unit")
    _, io2, _, _ = _make(k=4, chunk=2, w_mode="inv_deg")
    a, b = FtTerm(io1, D), FtTerm(io2, D); a.set_home([0]); b.set_home([0])
    p, q = _pos(nl), _pos(nl)
    assert torch.equal(a.ft_only(p, 4.), b.ft_only(q, 4.))


def test_gradcheck_nonsingular():
    nl, io, _, D = _make(k=4, chunk=2)
    ft = FtTerm(io, D); ft.set_home([0]); p = _pos(nl)
    assert torch.autograd.gradcheck(lambda z: ft(z, 11., 1., .7), (p,),
                                    eps=1e-6, atol=1e-5, rtol=1e-5)


def test_empty_csr_is_finite():
    rs = make_grid_regions((0., 0., 100., 100.), 4, 1, lattice=10)
    nl = _nl([(10., 50.)], [[0]])
    csr = build_net_node_csr(nl, 100); rects, r2k = rect_table(rs)
    io = IoTerm(csr, rects, r2k, 4, 1, 1, 1, device="cpu", chunk_budget=1)
    ft = FtTerm(io, np.zeros((4, 4))); ft.set_home([])
    assert torch.isfinite(ft(_pos(nl), 5., 0., 1.))


def test_s4a_hard_limit_is_not_true_feedthrough():
    nl, io, _, D = _make(k=3, chunk=1)
    ft = FtTerm(io, D)
    ft.set_home([0])
    pos = _pos(nl)
    # All three regions contain a terminal: true pure FT is zero. The
    # endpoint-rooted S4a cost remains D(0,2)-1 = 1, by its definition.
    value = ft.ft_only(pos, .001)
    assert float(value.detach()) == pytest.approx(1., abs=1e-10)
    value.backward()
    assert torch.isfinite(pos.grad).all()


def test_merged_gradient_is_linear_for_frozen_home():
    nl, io, _, D = _make(k=4, chunk=1)
    ft = FtTerm(io, D)
    ft.set_home([0])
    p = _pos(nl)
    merged, = torch.autograd.grad(ft(p, 9., 1., .4), p)
    i, = torch.autograd.grad(io(p, 9., 1.), p)
    f, = torch.autograd.grad(ft.ft_only(p, 9.), p)
    torch.testing.assert_close(merged, i+.4*f, atol=1e-10, rtol=1e-10)
