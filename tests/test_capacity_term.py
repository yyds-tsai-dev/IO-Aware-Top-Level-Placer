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
from ioplace.region_segments import Candidates, enumerate_segments, select_candidates
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


def test_normalised_overflow_gradient_matches_inv_for_positive_capacity():
    """Fix round 1 item 4: `inv` was pinned only as a *value* for C > 0, never
    as the actual backward-computed dd/dD -- a CapTerm that dropped `inv` (or
    used 1/C only in the value, not the gradient) would pass every other test
    in this file."""
    demand = torch.tensor([9.], dtype=torch.float64, requires_grad=True)
    capacity = torch.tensor([6.], dtype=torch.float64)
    d, inv = normalised_overflow(demand, capacity)
    d.backward()
    assert demand.grad.item() == pytest.approx(inv.item())
    assert demand.grad.item() == pytest.approx(1.0 / 6.0)


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


def test_segment_softmax_gradient_matches_the_softmax_jacobian():
    """Fix round 1 item 4: `:115-132` never went backward through
    `segment_softmax`, so a CapTerm that dropped the per-group normalisation
    or flipped a sign would pass every test that only checked forward values.
    The group-normalised softmax jacobian is
    d(sum_i c_i*alpha_i)/du_j = alpha_j * (c_j - sum_{i in group(j)} c_i*alpha_i)
    -- a closed form, pinned exactly rather than by finite difference."""
    u = torch.tensor([0., -1., 5., 5., 5.], dtype=torch.float64,
                     requires_grad=True)
    group = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    c = torch.tensor([2., -3., 1., 4., -2.], dtype=torch.float64)
    alpha = segment_softmax(u, group, 2)
    (c * alpha).sum().backward()
    with torch.no_grad():
        weighted = torch.zeros(2, dtype=torch.float64)
        weighted.index_add_(0, group, c * alpha)
        expected = alpha * (c - weighted[group])
    np.testing.assert_allclose(u.grad.numpy(), expected.numpy(), rtol=1e-10, atol=1e-12)


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


def test_l1_point_box_grad_matches_autograd_at_an_endpoint_aligned_centroid():
    """Fix round 1 item 2: torch.clamp(min=...)'s backward passes gradient
    through *at* the boundary (x >= min, not x > min), so l1_point_box_dist's
    own autograd gives +-1 exactly on a face, not 0. The old strict `>`/`<`
    hand gradient disagreed with CapTermRef's autograd by exactly 1 there --
    silent today (CapTermRef is the only consumer) but a hard failure of
    Task 5's CapTerm/CapTermRef `rel <= 1e-10` parity contract once CapTerm's
    hand gradient exists. Uses a non-degenerate box (both axes have extent)
    so each axis's face is tested independently of the other axis's
    degenerate-line cancellation (already covered above)."""
    box = torch.tensor([[0., 0., 20., 10.]], dtype=torch.float64)
    # cx exactly on the right (x) face, cy strictly inside the y-range.
    cx = torch.tensor([20.], dtype=torch.float64, requires_grad=True)
    cy = torch.tensor([5.], dtype=torch.float64, requires_grad=True)
    l1_point_box_dist(cx, cy, box).sum().backward()
    ddx, ddy = l1_point_box_grad(cx.detach(), cy.detach(), box)
    assert ddx.item() == pytest.approx(cx.grad.item())
    assert ddy.item() == pytest.approx(cy.grad.item())
    assert ddx.item() == pytest.approx(1.0)   # not 0 -- exactly the fix
    assert ddy.item() == pytest.approx(0.0)

    # cx strictly inside the x-range, cy exactly on the top (y) face.
    cx2 = torch.tensor([10.], dtype=torch.float64, requires_grad=True)
    cy2 = torch.tensor([10.], dtype=torch.float64, requires_grad=True)
    l1_point_box_dist(cx2, cy2, box).sum().backward()
    ddx2, ddy2 = l1_point_box_grad(cx2.detach(), cy2.detach(), box)
    assert ddx2.item() == pytest.approx(cx2.grad.item())
    assert ddy2.item() == pytest.approx(cy2.grad.item())
    assert ddx2.item() == pytest.approx(0.0)
    assert ddy2.item() == pytest.approx(1.0)  # not 0 -- exactly the fix


def test_alpha_softmax_direction_favours_the_nearer_alternative_segment():
    """Fix round 2: `_CapBase._alpha` softmaxes on `-d1/tau_b` (nearer
    alternative -> larger alpha), never `+d1/tau_b`. Every geometry
    `select_candidates` can build on this grid collapses to singleton
    alpha-groups (ruling D-4: `select_candidates`'s m_seg=2 budget never
    fires because no two segments this table produces share one (net,u,v,
    boundary) group), and a softmax over a one-element group is 1.0
    regardless of sign -- confirmed by mutation testing: flipping the sign at
    `cap_term.py`'s `_alpha` passed all 23 tests in this file before this one
    was added. `test_segment_softmax_gradient_matches_the_softmax_jacobian`
    calls `segment_softmax` directly with hand-built inputs, so it bypasses
    `_alpha`'s real call site (and its sign) entirely.

    Build a `Candidates` fixture directly -- not through `select_candidates`,
    which would collapse to singletons on this table -- with two *real*
    segments (the (0|1) and (1|2) boundaries of the strip fixture, at x=30
    and x=60) forced into one group, and a net centroid close to one and far
    from the other."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]], k=3)
    assert table.num_segments == 2
    cand = Candidates(net=np.array([0, 0], dtype=np.int64),
                      u=np.array([0, 0], dtype=np.int64),
                      v=np.array([2, 2], dtype=np.int64),
                      seg=np.array([0, 1], dtype=np.int64),
                      group=np.array([0, 0], dtype=np.int64),
                      count=np.array([1, 1], dtype=np.int64),
                      n_groups=1)
    ref = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    # x=33 is 3 units from segment 0's boundary line (x=30) and 27 units from
    # segment 1's (x=60) -- unambiguously nearer to segment 0.
    cx = torch.tensor([33.], dtype=torch.float64)
    cy = torch.tensor([15.], dtype=torch.float64)
    alpha = ref._alpha(cx, cy)
    assert alpha.sum().item() == pytest.approx(1.0)
    assert alpha[0].item() > alpha[1].item()


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


def test_candidate_count_scales_demand_linearly():
    """Fix round 1 item 3: D_s must charge per observed crossing
    (`cand.count`), matching capacity.npz's own semantics string ("one net
    crossing consumes one track") and evaluator_ref's accumulation (one
    increment per crossing). A candidate with count=3 must contribute 3x a
    count=1 candidate's demand, all else equal."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    cand1 = select_candidates(np.zeros(2, dtype=np.int64),
                              np.zeros(2, dtype=np.int64),
                              np.full(2, 2, dtype=np.int64),
                              np.array([0, 1], dtype=np.int64),
                              np.ones(2, dtype=np.int64), table)
    cand3 = select_candidates(np.zeros(2, dtype=np.int64),
                              np.zeros(2, dtype=np.int64),
                              np.full(2, 2, dtype=np.int64),
                              np.array([0, 1], dtype=np.int64),
                              np.full(2, 3, dtype=np.int64), table)
    ref1 = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref1.set_candidates(cand1)
    ref3 = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref3.set_candidates(cand3)
    p = _pos(nl)
    demand1 = ref1.demand(p, tau=0.5).detach().numpy()
    demand3 = ref3.demand(p, tau=0.5).detach().numpy()
    np.testing.assert_allclose(demand3, 3.0 * demand1, rtol=1e-9)


def test_center_anchor_changes_demand_not_the_gradient_contract():
    """Fix round 1 item 1: `_demand` must read the sec 7 anchor (the v2 flow's
    default is node_anchor="center" -- run_main_flow.py/run_placement_io.py --
    and the freeze reads membership at the centre too, freeze.py), not
    silently fall back to the cell lower-left the way it did before this fix
    (proof from review: under center, L_io moved 0.823574 -> 0.500000 while
    cap demand stayed bit-identical to the lower_left run -- the bug)."""
    node_xy = [(27., 14.), (61., 14.)]
    nl = _nl(node_xy, [[0, 1]])
    rs = _strip()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    size = np.full(nl.num_physical, 2.0)
    io_ll = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3,
                  num_movable=nl.num_movable, num_physical=nl.num_physical,
                  num_nodes=nl.num_physical, device="cpu",
                  chunk_budget=4 * len(csr.flat_net2node),
                  node_anchor="lower_left")
    io_c = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3,
                 num_movable=nl.num_movable, num_physical=nl.num_physical,
                 num_nodes=nl.num_physical, device="cpu",
                 chunk_budget=4 * len(csr.flat_net2node), node_anchor="center",
                 node_size_x=size, node_size_y=size)
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    ref_ll = CapTermRef(io_ll, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref_ll.set_candidates(cand)
    ref_c = CapTermRef(io_c, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref_c.set_candidates(cand)
    p = _pos(nl)
    demand_ll = ref_ll.demand(p, tau=0.5).detach().numpy()
    demand_c = ref_c.demand(p, tau=0.5).detach().numpy()
    assert not np.allclose(demand_ll, demand_c)
    # a per-node *constant* offset changes no gradient's existence/finiteness
    p2 = _pos(nl)
    ref_c(p2, 3.0, 1.0).backward()
    assert torch.isfinite(p2.grad).all()


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


def test_reference_gradient_matches_finite_differences():
    """Fix round 1 item 4: finiteness/non-zero-ness alone would pass a term
    whose backward is subtly wrong (dropped softmax normalisation, dropped
    `inv`, flipped alpha sign, ...). Pin the analytic gradient against a
    central finite difference instead, at points close enough to the
    boundary (28/62, not 15/75) that both the q-membership and the
    alpha-softmax gradient paths are live rather than saturated."""
    nl, rg, table, io, _csr = _setup([(28., 15.), (62., 15.)], [[0, 1]])
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    ref = CapTermRef(io, table.box, np.full(2, 0.5), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    tau, lam = 3.0, 1.0
    base = np.array([28., 62., 15., 15.], dtype=np.float64)
    p = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    val = ref(p, tau, lam)
    val.backward()
    analytic = p.grad.detach().numpy().copy()

    h = 1e-5
    fd = np.zeros_like(base)
    for i in range(len(base)):
        bp, bm = base.copy(), base.copy()
        bp[i] += h
        bm[i] -= h
        vp = float(ref(torch.tensor(bp, dtype=torch.float64), tau, lam).detach())
        vm = float(ref(torch.tensor(bm, dtype=torch.float64), tau, lam).detach())
        fd[i] = (vp - vm) / (2 * h)

    assert float(val.detach()) > 0.0            # over capacity here
    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)
    assert float(np.abs(analytic).sum()) > 0.0


from ioplace.ops.cap_term import CapNormTerm, CapTerm


def _pair_case(chunk, capacity, k=3):
    """One 3-pin net spanning all three regions of the strip, so the candidate
    list carries two distinct demand pairs and both boundaries."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 20.), (75., 10.)],
                                     [[0, 1, 2]], k=k, chunk=chunk)
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.array([1, 2], dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    caps = np.full(table.num_segments, capacity)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    return nl, production, reference


@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize("capacity", [0.2, 0.6, 5.0, 0.0])
def test_production_matches_the_reference_value_and_gradient(chunk, capacity):
    nl, production, reference = _pair_case(chunk, capacity)
    pa, pb = _pos(nl), _pos(nl)
    va = production(pa, 3.0, 1.7)
    vb = reference(pb, 3.0, 1.7)
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)


def test_production_gradient_matches_autograd_on_a_200_cell_toy():
    """sec 9's 'gradient vs autograd on a 200-cell toy'. The oracle is
    CapTermRef's ordinary autograd; _CapFn's hand-written backward must agree
    on a case big enough to exercise every chunk boundary and several
    candidates per net."""
    rng = np.random.default_rng(7)
    k = 4
    rs = make_grid_regions((0., 0., 120., 40.), k, 1, lattice=12)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 200
    xy = list(zip(rng.uniform(1., 119., n_cells), rng.uniform(1., 39., n_cells)))
    nets = [sorted(rng.choice(n_cells, int(rng.integers(2, 5)), replace=False).tolist())
            for _ in range(60)]
    nl = _nl(xy, nets)
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    common = dict(csr=csr, rects=rects, rect2region=r2k, K=k,
                  num_movable=n_cells, num_physical=nl.num_physical,
                  num_nodes=nl.num_physical, device="cpu")
    io = IoTerm(chunk_budget=2 * len(csr.flat_net2node), **common)
    # every active net gets both boundaries of a random adjacent pair
    n_active = len(csr.net_ids)
    nets_idx, us, vs, segs = [], [], [], []
    for e in range(n_active):
        pair = int(rng.integers(0, k - 1))
        for seg in np.nonzero((table.pair_a == pair) & (table.pair_b == pair + 1))[0]:
            nets_idx.append(e)
            us.append(pair)
            vs.append(pair + 1)
            segs.append(int(seg))
    cand = select_candidates(np.asarray(nets_idx, dtype=np.int64),
                             np.asarray(us, dtype=np.int64),
                             np.asarray(vs, dtype=np.int64),
                             np.asarray(segs, dtype=np.int64),
                             np.ones(len(segs), dtype=np.int64), table)
    caps = np.full(table.num_segments, 3.0)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    pa, pb = _pos(nl), _pos(nl)
    va, vb = production(pa, 4.0, 1.0), reference(pb, 4.0, 1.0)
    assert float(va.detach()) > 0.0
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)
    assert float(pa.grad.abs().sum()) > 0.0


def test_the_q_product_gradient_is_exact_not_linearised():
    """Isolate the q path: freeze alpha by giving every candidate its own
    singleton group, then check d L / d q_u against the analytic q_v cofactor
    by perturbing one node along x and comparing with a central difference of
    the *whole* term. A linearised product (q_u*q_v^frozen) would match the
    value but not this derivative."""
    nl, production, reference = _pair_case(chunk=1, capacity=0.2)
    p = _pos(nl)
    value = production(p, 4.0, 1.0)
    value.backward()
    analytic = p.grad.clone()
    base = p.detach().clone()
    h = 1e-6
    for index in (0, 1, 2):
        probe = base.clone()
        probe[index] += h
        up = float(production(probe.requires_grad_(False), 4.0, 1.0).detach())
        probe = base.clone()
        probe[index] -= h
        down = float(production(probe.requires_grad_(False), 4.0, 1.0).detach())
        assert analytic[index].item() == pytest.approx((up - down) / (2 * h),
                                                       rel=1e-4, abs=1e-9)


def _split_boundary():
    """Region 0 touches region 1 along two disjoint vertical runs (region 2
    plugs the middle), so one net can carry two genuine alternative segments
    on the same boundary pair -- the configuration in which alpha has a
    gradient at all."""
    from ioplace.regions import RegionSet, RegionSpec
    rs = RegionSet(die=DIE, lattice=9, regions=[
        RegionSpec("A", np.array([[0., 0., 30., 30.]])),
        RegionSpec("B", np.array([[30., 0., 60., 10.], [30., 20., 60., 30.]])),
        RegionSpec("C", np.array([[30., 10., 60., 20.], [60., 0., 90., 30.]]))])
    rs.validate()
    return rs


def _split_case(capacity=0.1, chunk=4):
    rs = _split_boundary()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    alternatives = [s for s in range(table.num_segments)
                    if (int(table.pair_a[s]), int(table.pair_b[s])) == (0, 1)]
    assert len(alternatives) == 2
    nl = _nl([(15., 5.), (45., 5.)], [[0, 1]])
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3, num_movable=2,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=chunk * len(csr.flat_net2node))
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.ones(2, dtype=np.int64),
                             np.asarray(alternatives, dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    assert cand.n_groups == 1                    # one group, two alternatives
    caps = np.full(table.num_segments, capacity)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    return nl, production, reference


def test_the_alpha_path_survives_a_saturated_q():
    """After the freeze q is 0/1 and carries no gradient (sec 3); the term must
    still move cells *along* a boundary through alpha. tau is tiny here, so q
    is saturated and every bit of the surviving gradient is the alpha path --
    and it lies purely in y, the coordinate along the boundary."""
    nl, production, reference = _split_case()
    pa, pb = _pos(nl), _pos(nl)
    va, vb = production(pa, 1e-3, 1.0), reference(pb, 1e-3, 1.0)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(pa.grad).all()
    assert float(pa.grad.abs().sum()) > 0.0
    n = nl.num_physical
    assert float(pa.grad[:n].abs().sum()) == pytest.approx(0.0, abs=1e-12)  # x
    assert float(pa.grad[n:].abs().sum()) > 0.0                             # y
    # the group's two alternatives share exactly one unit of demand
    assert float(production.last_demand.sum()) == pytest.approx(1.0, abs=1e-6)


def test_a_singleton_alpha_group_has_no_alpha_gradient():
    """Recorded limitation D-4, pinned so nobody 'fixes' it by accident: a
    softmax over one element is identically 1, so a group with a single
    candidate contributes nothing once q saturates. This is why
    frac_singleton_groups is a reported diagnostic."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]], k=3)
    cand = select_candidates(np.zeros(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64), table)
    production = CapTerm(io, table.box, np.full(table.num_segments, 0.1),
                         tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    p = _pos(nl)
    production(p, 1e-3, 1.0).backward()
    assert float(p.grad.abs().sum()) == pytest.approx(0.0, abs=1e-12)
    assert production.diagnostics(_pos(nl), 1e-3)["frac_singleton_groups"] == 1.0


def test_no_candidates_is_an_exact_zero_detached_from_the_graph():
    """The empty term returns pos.new_zeros(()) -- the same shape IoTerm and
    FtTerm return through term_fn. It has no grad_fn, which is correct:
    DREAMPlace adds it to an objective that does."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]])
    production = CapTerm(io, table.box, np.ones(table.num_segments),
                         tau_b=2. * rg.cell_w)
    value = production(_pos(nl), 3.0, 1.0)
    assert float(value.detach()) == 0.0
    assert value.requires_grad is False
    assert production.diagnostics(_pos(nl), 3.0)["num_candidates"] == 0


def test_fixed_and_filler_nodes_never_receive_gradient():
    nl, rg, table, io_full, _csr = _setup([(15., 15.), (45., 15.), (75., 15.)],
                                          [[0, 1], [1, 2]])
    rs = _strip(3)
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3, num_movable=2,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=4 * len(csr.flat_net2node))
    cand = select_candidates(np.array([0, 1], dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.array([1, 2], dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    production = CapTerm(io, table.box, np.full(table.num_segments, 0.1),
                         tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    p = _pos(nl)
    production(p, 3.0, 1.0).backward()
    n = nl.num_physical
    assert p.grad[2].item() == 0.0            # node 2 is fixed
    assert p.grad[n + 2].item() == 0.0


def test_diagnostics_and_curvature():
    nl, production, _reference = _pair_case(chunk=2, capacity=0.2)
    diag = production.diagnostics(_pos(nl), 3.0)
    assert set(diag) == {"l_cap", "demand_total", "max_d", "num_over_capacity",
                         "num_candidates", "num_groups", "frac_singleton_groups"}
    assert diag["num_candidates"] == 2 and diag["num_groups"] == 2
    assert diag["max_d"] == pytest.approx(3.997371310542147, rel=1e-9)
    assert diag["num_over_capacity"] == 2
    assert diag["frac_singleton_groups"] == 1.0
    assert production.curvature == pytest.approx(14.0)


def test_cap_norm_term_exposes_the_unweighted_value():
    nl, production, _reference = _pair_case(chunk=2, capacity=0.2)
    adapter = CapNormTerm(production)
    p = _pos(nl)
    got = adapter.value(p, {"tau": 3.0, "iteration": 0, "overflow": 0.2,
                            "gamma": 1.0})
    torch.testing.assert_close(got, production(p, 3.0, 1.0))


def test_set_candidates_rejects_malformed_input():
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]])
    production = CapTerm(io, table.box, np.ones(table.num_segments),
                         tau_b=2. * rg.cell_w)
    good = select_candidates(np.zeros(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64), table)
    import dataclasses
    with pytest.raises(ValueError, match="active nets"):
        production.set_candidates(dataclasses.replace(
            good, net=np.array([99], dtype=np.int64)))
    with pytest.raises(ValueError, match="segment id"):
        production.set_candidates(dataclasses.replace(
            good, seg=np.array([999], dtype=np.int64)))
    with pytest.raises(ValueError, match="u < v"):
        production.set_candidates(dataclasses.replace(
            good, u=np.array([1], dtype=np.int64), v=np.array([0], dtype=np.int64)))


# ------------------------------------ live-band parity (Task 5 additions) --
# The flow's capacity band is tau = (0.057 .. 0.03) * L_R with
# L_R = sqrt(die area / K) (run_main_flow.py); on this 90x30 / K=3 fixture
# L_R = 30, so tau in [0.9, 1.71]. The brief's parity cases run at tau 3/4
# (singleton groups only) or 1e-3 (q saturated); these pin parity where the
# term is actually live, on a MULTI-member alpha group, under the centre
# anchor, and with count > 1.
_BAND_TAUS = (0.9, 1.3, 1.71)


def _split_live_case(capacity, chunk, count=1, anchor="lower_left",
                     xy=((28., 4.), (33., 24.))):
    """_split_boundary's two alternatives on the (0,1) pair, pins placed
    within ~1-2 tau of the boundary so q is NOT saturated, and the centroid
    off-centre so alpha is non-uniform: both the q and the alpha channel
    carry gradient."""
    rs = _split_boundary()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    alternatives = [s for s in range(table.num_segments)
                    if (int(table.pair_a[s]), int(table.pair_b[s])) == (0, 1)]
    nl = _nl(list(xy), [[0, 1]])
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    extra = {}
    if anchor == "center":
        size = np.full(nl.num_physical, 2.0)
        extra = dict(node_anchor="center", node_size_x=size, node_size_y=size)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3, num_movable=2,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=chunk * len(csr.flat_net2node),
                **extra)
    assert io.k_chunk == min(chunk, 3)
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.ones(2, dtype=np.int64),
                             np.asarray(alternatives, dtype=np.int64),
                             np.full(2, count, dtype=np.int64), table)
    assert cand.n_groups == 1 and len(cand.net) == 2   # a 2-member group
    caps = np.full(table.num_segments, capacity)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    return nl, production, reference


def _assert_parity(nl, production, reference, tau, lam=1.3):
    pa, pb = _pos(nl), _pos(nl)
    va, vb = production(pa, tau, lam), reference(pb, tau, lam)
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)
    return pa.grad


@pytest.mark.parametrize("tau", _BAND_TAUS)
@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize("capacity", [0.05, 0.3, 0.0])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("anchor", ["lower_left", "center"])
def test_production_parity_on_a_two_member_alpha_group_in_the_live_tau_band(
        tau, chunk, capacity, count, anchor):
    nl, production, reference = _split_live_case(capacity, chunk, count, anchor)
    grad = _assert_parity(nl, production, reference, tau)
    n = nl.num_physical
    # the alpha channel is live: the along-boundary (y) gradient is non-zero
    assert float(grad[n:].abs().sum()) > 0.0
    torch.testing.assert_close(production.last_demand, reference.last_demand,
                               atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("tau", _BAND_TAUS)
def test_production_parity_mixes_over_and_under_capacity_and_group_sizes(tau):
    """Two nets on the split boundary: net 0 carries a 2-member group, net 1
    (a different net, same pair) a singleton on one alternative. Capacities
    differ per segment so one segment is over and one under capacity."""
    rs = _split_boundary()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    alternatives = [s for s in range(table.num_segments)
                    if (int(table.pair_a[s]), int(table.pair_b[s])) == (0, 1)]
    nl = _nl([(28., 4.), (33., 24.), (27., 26.), (32., 27.)], [[0, 1], [2, 3]])
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3, num_movable=4,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=1 * len(csr.flat_net2node))
    cand = select_candidates(np.array([0, 0, 1], dtype=np.int64),
                             np.zeros(3, dtype=np.int64),
                             np.ones(3, dtype=np.int64),
                             np.array([alternatives[0], alternatives[1],
                                       alternatives[1]], dtype=np.int64),
                             np.array([1, 1, 2], dtype=np.int64), table)
    assert cand.n_groups == 2
    caps = np.full(table.num_segments, 5.0)
    caps[alternatives[0]] = 0.1           # over capacity
    caps[alternatives[1]] = 5.0           # under capacity
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    _assert_parity(nl, production, reference, tau)
    d = production.last_d
    assert float(d[alternatives[0]]) > 0.0 > float(d[alternatives[1]])
    assert production.diagnostics(_pos(nl), tau)["frac_singleton_groups"] == 0.5


def test_production_gradient_matches_finite_differences_on_a_two_member_group():
    """The hand backward against the *production* forward's own central
    difference, at a band tau, on a multi-member group -- independent of
    CapTermRef, so a shared mistake in both would still be caught."""
    nl, production, _reference = _split_live_case(0.05, chunk=1)
    tau, lam = 1.3, 1.0
    p = _pos(nl)
    production(p, tau, lam).backward()
    analytic = p.grad.detach().numpy().copy()
    base = p.detach().clone()
    h = 1e-6
    for index in range(base.numel()):
        up, down = base.clone(), base.clone()
        up[index] += h
        down[index] -= h
        fd = (float(production(up, tau, lam)) - float(production(down, tau, lam))) / (2 * h)
        assert analytic[index] == pytest.approx(fd, rel=1e-6, abs=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_production_parity_on_cuda_in_the_live_tau_band():
    nl, production, reference = _split_live_case(0.05, chunk=1, count=2,
                                                  anchor="center")
    production.io_term.cuda()   # shared by both terms
    production.cuda()
    reference.cuda()
    pa = _pos(nl).detach().cuda().requires_grad_(True)
    pb = _pos(nl).detach().cuda().requires_grad_(True)
    va, vb = production(pa, 1.3, 1.0), reference(pb, 1.3, 1.0)
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)


def _l_boundary_case(capacity, chunk):
    """Region B wraps region A on two sides, so the (0,1) pair has one
    VERTICAL and one HORIZONTAL segment. On collinear alternatives (the split
    boundary) the alpha path's across-boundary component cancels exactly
    (equal d d1/d cx, softmax back-prop sums to 0 per group); here it does
    not, so this is the only fixture exercising the x channel of the alpha
    backward."""
    from ioplace.regions import RegionSet, RegionSpec
    rs = RegionSet(die=(0., 0., 60., 60.), lattice=6, regions=[
        RegionSpec("A", np.array([[0., 0., 30., 30.]])),
        RegionSpec("B", np.array([[30., 0., 60., 60.], [0., 30., 30., 60.]]))])
    rs.validate()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    assert table.num_segments == 2
    nl = _nl([(25., 12.), (33., 22.)], [[0, 1]])
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=2, num_movable=2,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=chunk * len(csr.flat_net2node))
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.ones(2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    assert cand.n_groups == 1
    caps = np.full(2, capacity)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    return nl, production, reference


@pytest.mark.parametrize("tau_rel", [0.03, 0.045, 0.057])
@pytest.mark.parametrize("chunk", [1, 2])
@pytest.mark.parametrize("capacity", [0.05, 0.0])
def test_production_parity_on_an_l_shaped_two_member_group(tau_rel, chunk, capacity):
    nl, production, reference = _l_boundary_case(capacity, chunk)
    tau = tau_rel * (60. * 60. / 2) ** 0.5            # L_R = sqrt(area / K)
    _assert_parity(nl, production, reference, tau)
    # saturated q: only the alpha path survives, and it has an x component
    nl, production, reference = _l_boundary_case(capacity, chunk)
    grad = _assert_parity(nl, production, reference, 1e-3)
    assert float(grad[:nl.num_physical].abs().sum()) > 0.0
