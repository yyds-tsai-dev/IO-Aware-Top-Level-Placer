import numpy as np
import pytest
import torch

from ioplace.ops.cap_term import (CAP_CURVATURE_DREF, CapTermRef, cap_curvature,
                                  cap_penalty, cap_penalty_grad,
                                  l1_point_box_dist, l1_point_box_grad,
                                  normalised_overflow, segment_softmax)
from ioplace.ops.io_term import IoTerm, build_net_node_csr
from ioplace.ops.soft_assign import rect_table
from ioplace.region_grid import RegionGrid
from ioplace.region_segments import enumerate_segments, select_candidates
from ioplace.regions import make_grid_regions
from tests.test_io_term import _nl

DIE = (0., 0., 90., 30.)


def _strip(k=3):
    """A 1 x k strip of regions on a lattice-9 grid: region 0 spans x in
    [0,30), region 1 [30,60), region 2 [60,90). Adjacent pairs are (0,1) and
    (1,2), so a net with pins in 0 and 2 feeds through 1."""
    return make_grid_regions(DIE, k, 1, lattice=9)


def _setup(node_xy, nets, k=3, chunk=4):
    rs = _strip(k)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    nl = _nl(node_xy, nets)
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                num_movable=len(node_xy), num_physical=nl.num_physical,
                num_nodes=nl.num_physical, device="cpu",
                chunk_budget=max(1, chunk) * max(1, len(csr.flat_net2node)))
    return nl, rg, table, io, csr


def _pos(nl):
    n = nl.num_physical
    p = torch.zeros(2 * n, dtype=torch.float64)
    p[:n] = torch.as_tensor(nl.node_x)
    p[n:] = torch.as_tensor(nl.node_y)
    return p.requires_grad_(True)


# ---------------------------------------------------------------- penalty --
@pytest.mark.parametrize("d", [-10., -1., -1e-9, 0.])
def test_penalty_is_exactly_zero_below_capacity(d):
    """Not approximately zero: sec 5 rejects softplus precisely because it
    exerts force where nothing is violated."""
    assert cap_penalty(torch.tensor(d, dtype=torch.float64)).item() == 0.0
    assert cap_penalty_grad(torch.tensor(d, dtype=torch.float64)).item() == 0.0
    assert cap_penalty(np.float64(d)) == 0.0


def test_penalty_matches_the_spec_expression_above_capacity():
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    np.testing.assert_allclose(cap_penalty(d).numpy(),
                               [2 * .125 + .25, 2 + 1, 16 + 4])
    np.testing.assert_allclose(cap_penalty_grad(d).numpy(),
                               [6 * .25 + 1., 6 + 2, 24 + 4])


def test_penalty_is_c1_but_not_c2_at_the_knee():
    h = 1e-3
    at = lambda v: float(cap_penalty(torch.tensor(v, dtype=torch.float64)))
    central_first = (at(h) - at(-h)) / (2 * h)
    assert abs(central_first) < h            # -> 0: first derivative continuous
    central_second = (at(h) - 2 * at(0.) + at(-h)) / h ** 2
    # (pen''(0+) + pen''(0-)) / 2 = (2 + 0) / 2 = 1: a real second-derivative
    # jump, deliberately kept (GrandPlan Eq.5's shape).
    assert central_second == pytest.approx(1.0, abs=1e-2)


def test_penalty_gradient_matches_finite_differences():
    for d in (0.3, 1.0, 2.5):
        h = 1e-6
        at = lambda v: float(cap_penalty(torch.tensor(v, dtype=torch.float64)))
        assert cap_penalty_grad(torch.tensor(d)).item() == \
            pytest.approx((at(d + h) - at(d - h)) / (2 * h), rel=1e-6)


def test_declared_curvature_is_pen_second_derivative_at_the_reference():
    assert CAP_CURVATURE_DREF == 1.0
    assert cap_curvature() == pytest.approx(14.0)
    assert cap_curvature(0.0) == pytest.approx(2.0)


# ------------------------------------------------------- zero capacity ----
def test_zero_capacity_segments_keep_a_finite_penalty_with_no_epsilon():
    demand = torch.tensor([0., 2., 3.], dtype=torch.float64)
    capacity = torch.tensor([0., 0., 6.], dtype=torch.float64)
    d, inv = normalised_overflow(demand, capacity)
    np.testing.assert_allclose(d.numpy(), [0., 2., -0.5])
    np.testing.assert_allclose(inv.numpy(), [1., 1., 1. / 6.])
    penalty = cap_penalty(d)
    assert torch.isfinite(penalty).all()
    assert penalty[0].item() == 0.0                     # no demand -> no penalty
    assert penalty[1].item() == pytest.approx(2 * 8 + 4)
    assert penalty[2].item() == 0.0                     # under capacity


def test_zero_capacity_overflow_is_differentiable_and_has_no_nan():
    demand = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
    capacity = torch.tensor([0.], dtype=torch.float64)
    d, _inv = normalised_overflow(demand, capacity)
    cap_penalty(d).sum().backward()
    assert torch.isfinite(demand.grad).all()
    assert demand.grad.item() == pytest.approx(6 * 4 + 2 * 2)


# ------------------------------------------------------------- alpha ------
def test_segment_softmax_normalises_within_each_group():
    u = torch.tensor([0., -1., 5., 5., 5.], dtype=torch.float64)
    group = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    alpha = segment_softmax(u, group, 2)
    assert alpha[:2].sum().item() == pytest.approx(1.0)
    assert alpha[2:].sum().item() == pytest.approx(1.0)
    np.testing.assert_allclose(alpha[2:].numpy(), [1 / 3.] * 3)
    assert alpha[0].item() > alpha[1].item()


def test_segment_softmax_is_shift_invariant_and_overflow_safe():
    u = torch.tensor([1e6, 1e6 - 1.], dtype=torch.float64)
    group = torch.zeros(2, dtype=torch.int64)
    alpha = segment_softmax(u, group, 1)
    assert torch.isfinite(alpha).all()
    small = segment_softmax(u - 1e6, group, 1)
    torch.testing.assert_close(alpha, small)


def test_l1_point_box_distance_and_subgradient():
    box = torch.tensor([[10., 0., 10., 20.]], dtype=torch.float64)  # vertical seg
    for cx, cy, expected in ((10., 5., 0.), (13., 5., 3.), (7., 25., 8.)):
        got = l1_point_box_dist(torch.tensor([cx], dtype=torch.float64),
                                torch.tensor([cy], dtype=torch.float64), box)
        assert got.item() == pytest.approx(expected)
    ddx, ddy = l1_point_box_grad(torch.tensor([13., 7., 10.], dtype=torch.float64),
                                 torch.tensor([5., 25., 5.], dtype=torch.float64),
                                 box.repeat(3, 1))
    np.testing.assert_allclose(ddx.numpy(), [1., -1., 0.])
    np.testing.assert_allclose(ddy.numpy(), [0., 1., 0.])


# ------------------------------------------------- reference term ---------
def test_feed_through_charges_both_the_entry_and_the_exit_segment():
    """Interpretation D-1/D-2: a net with pins in regions 0 and 2 traverses 1,
    so its entry (0|1) and exit (1|2) segments are in different alpha groups
    and each take the full w*q_0*q_2 -- the spec's 'both take demand'."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    assert table.num_segments == 2
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s]))
             for s in range(table.num_segments)]
    assert pairs == [(0, 1), (1, 2)]
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    assert cand.n_groups == 2                       # NOT one softmax over both
    ref = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    demand = ref.demand(_pos(nl), tau=0.5).detach().numpy()
    # tau is small, so q_0 = q_2 = 1 and q_1 = 0; alpha = 1 in each singleton
    # group; w_e = 1.
    np.testing.assert_allclose(demand, [1., 1.], atol=1e-6)


def test_two_alternatives_on_one_boundary_pair_share_one_unit_of_demand():
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]], k=3)
    cand = select_candidates(np.zeros(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64), table)
    ref = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    demand = ref.demand(_pos(nl), tau=0.5).detach().numpy()
    assert demand[0] == pytest.approx(1.0, abs=1e-6)
    assert demand[1] == pytest.approx(0.0, abs=1e-12)


def test_reference_penalty_responds_only_above_capacity():
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    slack = CapTermRef(io, table.box, np.full(2, 10.), tau_b=2. * rg.cell_w)
    tight = CapTermRef(io, table.box, np.full(2, 0.5), tau_b=2. * rg.cell_w)
    slack.set_candidates(cand)
    tight.set_candidates(cand)
    p = _pos(nl)
    assert float(slack(p, 0.5, 1.0).detach()) == 0.0
    assert float(tight(p, 0.5, 1.0).detach()) == pytest.approx(6.0)


def test_reference_gradient_is_nonzero_and_finite():
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    ref = CapTermRef(io, table.box, np.full(2, 0.5), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    p = _pos(nl)
    ref(p, 3.0, 1.0).backward()
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.abs().sum()) > 0.0
