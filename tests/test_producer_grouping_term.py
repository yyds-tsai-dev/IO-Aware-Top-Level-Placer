import numpy as np
import pytest
import torch
from ioplace.producer import hull
from ioplace.producer.grouping_term import GroupingTerm

DIE = (0.0, 0.0, 8.0, 8.0)
SQ = np.array([[2., 2.], [6., 2.], [6., 6.], [2., 6.]])
BIG = np.array([[0., 0.], [8., 0.], [8., 8.], [0., 8.]])


def _one_cell_term(device="cpu"):
    """One zero-size movable cell in region 0, one dummy fixed node."""
    t = GroupingTerm(part=np.array([0]), node_size_x=np.zeros(1),
                     node_size_y=np.zeros(1), num_movable=1, num_nodes=2,
                     device=device)
    t.set_tables(hull.anchor_tables([SQ, BIG], DIE, lattice=8, device=device))
    return t


def _pos(x, y, n_nodes=2, dtype=torch.float64):
    p = torch.zeros(2 * n_nodes, dtype=dtype)
    p[0], p[n_nodes] = x, y
    return p


def test_energy_matches_the_closed_form_of_eq1():
    """Cell centre (0.5, 0.5). Pull anchor = nearest point of SQ = (2, 2).
    Push: the one foreign hull BIG contains it; nearest point of dBIG is
    (0.5, 0) (tie with (0, 0.5) broken on the first edge). So
    E = 0.5*((0.5-2)^2 + (0.5-2)^2) + 0.5*1*((0.5-0.5)^2 + (0.5-0)^2) = 2.375."""
    t = _one_cell_term()
    e = float(t(_pos(0.5, 0.5), lam=1.0))
    assert e == pytest.approx(2.375, rel=1e-6)


def test_gradient_matches_the_closed_form_of_eq2():
    t = _one_cell_term()
    p = _pos(0.5, 0.5).requires_grad_(True)
    t(p, lam=1.0).backward()
    # dE/dx = (0.5-2) + 1*(0.5-0.5) = -1.5 ; dE/dy = (0.5-2) + (0.5-0) = -1.0
    assert float(p.grad[0]) == pytest.approx(-1.5, rel=1e-6)
    assert float(p.grad[2]) == pytest.approx(-1.0, rel=1e-6)


def test_cell_inside_its_own_hull_feels_no_pull():
    t = _one_cell_term()
    # centre (3.5, 3.5) is inside SQ; push from BIG is the only contribution
    e = float(t(_pos(3.5, 3.5), lam=1.0))
    proj, _ = hull.nearest_on_polygon_boundary(
        torch.tensor([3.5], dtype=torch.float64),
        torch.tensor([3.5], dtype=torch.float64), BIG)
    expect = 0.5 * float((3.5 - proj[0, 0]) ** 2 + (3.5 - proj[0, 1]) ** 2)
    assert e == pytest.approx(expect, abs=1e-3)


def test_lambda_scales_the_value_and_zero_short_circuits():
    t = _one_cell_term()
    assert float(t(_pos(0.5, 0.5), lam=2.0)) == pytest.approx(2.0 * 2.375, rel=1e-6)
    assert float(t(_pos(0.5, 0.5), lam=0.0)) == 0.0


def test_term_is_zero_before_the_first_rebuild():
    t = GroupingTerm(part=np.array([0]), node_size_x=np.zeros(1),
                     node_size_y=np.zeros(1), num_movable=1, num_nodes=2,
                     device="cpu")
    assert t.n_rebuilds == 0
    assert float(t(_pos(0.5, 0.5), lam=1.0)) == 0.0


def test_cell_centre_anchoring_uses_node_size():
    """spec section 7: the soft anchor is the cell CENTRE, not the lower-left
    corner. A 1x1 cell placed at (0,0) has its centre at (0.5,0.5), so it must
    score exactly the closed form above."""
    t = GroupingTerm(part=np.array([0]), node_size_x=np.ones(1),
                     node_size_y=np.ones(1), num_movable=1, num_nodes=2,
                     device="cpu")
    t.set_tables(hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu"))
    assert float(t(_pos(0.0, 0.0), lam=1.0)) == pytest.approx(2.375, rel=1e-6)


def test_anchors_stay_frozen_between_rebuilds():
    """Moving the cell inside the same bin must not move the anchor: the energy
    must follow the same quadratic, and n_rebuilds must not change."""
    t = _one_cell_term()
    e0 = float(t(_pos(0.5, 0.5), lam=1.0))
    e1 = float(t(_pos(0.6, 0.5), lam=1.0))
    d_pull = (0.6 - 2.0) ** 2 - (0.5 - 2.0) ** 2
    d_push = (0.6 - 0.5) ** 2 - 0.0
    assert e1 - e0 == pytest.approx(0.5 * (d_pull + d_push), rel=1e-6)
    assert t.n_rebuilds == 1


def test_fixed_and_filler_nodes_never_receive_gradient():
    t = GroupingTerm(part=np.array([0, 1]), node_size_x=np.zeros(2),
                     node_size_y=np.zeros(2), num_movable=2, num_nodes=5,
                     device="cpu")
    t.set_tables(hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu"))
    p = torch.full((10,), 0.5, dtype=torch.float64).requires_grad_(True)
    t(p, lam=1.0).backward()
    g = p.grad
    assert torch.equal(g[2:5], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(g[7:10], torch.zeros(3, dtype=torch.float64))


@pytest.mark.gpu
def test_gradient_matches_central_differences_on_a_200_cell_toy():
    """Finite differences are valid here because the anchor is piecewise
    constant per lattice bin: every cell sits at a bin centre plus a jitter
    bounded well inside its bin, and h is five orders of magnitude smaller than
    the bin width, so no probe crosses a bin boundary."""
    dev = "cuda"
    die = (0.0, 0.0, 1024.0, 1024.0)
    lattice, n, h = 64, 200, 1e-3
    bin_w = 1024.0 / lattice                      # 16.0
    rng = np.random.default_rng(3)
    ibin = rng.integers(0, lattice, size=(n, 2))
    jitter = rng.uniform(-0.25 * bin_w, 0.25 * bin_w, size=(n, 2))
    xy = (ibin + 0.5) * bin_w + jitter
    part = rng.integers(0, 3, size=n)
    hulls = [hull.build_hull(xy[part == k] if (part == k).any() else xy,
                             a_max=1024.0 * 1024.0) for k in range(3)]
    t = GroupingTerm(part=part, node_size_x=np.zeros(n), node_size_y=np.zeros(n),
                     num_movable=n, num_nodes=n, device=dev)
    t.set_tables(hull.anchor_tables(hulls, die, lattice=lattice, device=dev))
    p = torch.tensor(np.concatenate([xy[:, 0], xy[:, 1]]), dtype=torch.float64,
                     device=dev).requires_grad_(True)
    t(p, lam=1.0).backward()
    g = p.grad.detach().cpu().numpy()
    base = p.detach()
    for i in rng.choice(2 * n, size=20, replace=False):
        up, dn = base.clone(), base.clone()
        up[i] += h
        dn[i] -= h
        fd = (float(t(up, lam=1.0)) - float(t(dn, lam=1.0))) / (2 * h)
        assert fd == pytest.approx(g[int(i)], rel=1e-5, abs=1e-6)
