import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.netlist import Netlist
from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.ops.soft_assign import rect_table, softmax_stats, d_star_from_m
from ioplace.ops.io_term import (NetCsr, build_net_node_csr, net_mask_io,
                                 margin_penalty, IoTermRef, DEG_BUCKET_LABELS)

DIE = (0., 0., 100., 100.)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- fixtures
def _nl(node_xy, nets, n_extra_nodes=0):
    """node_xy: [(x,y),...]; nets: [[node_idx,...],...] (may repeat a node)."""
    nx_ = np.array([p[0] for p in node_xy] + [0.] * n_extra_nodes, dtype=np.float64)
    ny_ = np.array([p[1] for p in node_xy] + [0.] * n_extra_nodes, dtype=np.float64)
    pins, p2n = [], []
    for e, nodes in enumerate(nets):
        pins += list(nodes); p2n += [e] * len(nodes)
    pins = np.array(pins, np.int32); p2n = np.array(p2n, np.int32)
    start = np.searchsorted(p2n, np.arange(len(nets) + 1)).astype(np.int32)
    n = len(nx_)
    return Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n), node_size_y=np.ones(n),
                   num_movable=n, num_terminals=0, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                   pin2node=pins, pin2net=p2n,
                   flat_net2pin=np.arange(len(pins), dtype=np.int32),
                   flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)

def _pos(nl, device=DEV, n_filler=0):
    n_all = nl.num_physical + n_filler
    v = torch.zeros(2 * n_all, dtype=torch.float64, device=device)
    v[:nl.num_physical] = torch.as_tensor(nl.node_x, device=device)
    v[n_all:n_all + nl.num_physical] = torch.as_tensor(nl.node_y, device=device)
    return v.requires_grad_(True)

def _term(nl, rs, K, w_mode="unit", num_movable=None, n_filler=0, ignore=100):
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, ignore)
    return IoTermRef(csr, rects, r2k, K,
                     num_movable=nl.num_movable if num_movable is None else num_movable,
                     num_physical=nl.num_physical,
                     num_nodes=nl.num_physical + n_filler,
                     device=DEV, w_mode=w_mode)

# ---------------------------------------------------------------- CSR / mask
def test_build_csr_dedups_same_node_pins():
    nl = _nl([(10., 10.), (90., 10.)], [[0, 1, 0, 1, 0]])   # 5 pins, 2 distinct nodes
    csr = build_net_node_csr(nl, 100)
    assert csr.n_nets_total == 1 and len(csr.net_ids) == 1
    assert list(csr.degrees) == [2] and list(csr.pin_degrees) == [5]
    assert sorted(csr.flat_net2node[csr.net2node_start[0]:csr.net2node_start[1]].tolist()) == [0, 1]

def test_build_csr_drops_degree_1_and_large_nets():
    nl = _nl([(10., 10.), (90., 10.), (10., 90.)],
             [[0], [0, 1], [0, 1, 2]])
    csr = build_net_node_csr(nl, ignore_net_degree=3)     # keeps 2 <= pin_deg < 3
    assert list(csr.net_ids) == [1]

def test_build_csr_drops_net_whose_pins_collapse_to_one_node():
    nl = _nl([(10., 10.), (90., 10.)], [[0, 0, 0], [0, 1]])
    csr = build_net_node_csr(nl, 100)
    assert list(csr.net_ids) == [1]

def test_net_mask_io_matches_dreamplace_predicate():
    """DP: net_mask_ignore_large_degrees = (2 <= deg) & (deg < ignore_net_degree)."""
    nl = _nl([(1., 1.)] * 6, [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
    got = net_mask_io(nl, ignore_net_degree=4)
    deg = nl.net_degrees
    assert np.array_equal(got, (deg >= 2) & (deg < 4))

def test_deg_bucket_labels_and_assignment():
    assert DEG_BUCKET_LABELS == ("2", "3", "4-7", "8-15", "16-31", "32-63", "64-99")
    nl = _nl([(1., 1.)] * 70, [list(range(2)), list(range(3)), list(range(5)),
                               list(range(9)), list(range(20)), list(range(40)),
                               list(range(70))])
    csr = build_net_node_csr(nl, 100)
    assert list(csr.deg_bucket) == [0, 1, 2, 3, 4, 5, 6]

# ---------------------------------------------------------------- semantics
def test_two_node_net_value_matches_hand_computation():
    """K=2 vertical split, tau=25, nodes at (25,50) and (75,50).
    d = (-25, +25) and (+25, -25) -> p = (0.88079708, 0.11920292) and mirror.
    S_k = log(0.11920292) + log(0.88079708) = -2.25385602 for both k,
    q_k = 0.89500641, lambda = 1.79001283, L_IO = lambda - 1."""
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.)], [[0, 1]])
    term = _term(nl, rs, 2)
    L = term(_pos(nl), tau=25.0, lambda_io=1.0)
    assert float(L) == pytest.approx(0.7900128292, rel=1e-9)

def test_three_node_net_with_duplicate_position():
    """Adding a third node at the same place as node 0 changes S_0 by one more
    log(1-p_0) term: lambda = 1.8950064146."""
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.), (25., 50.)], [[0, 1, 2]])
    term = _term(nl, rs, 2)
    L = term(_pos(nl), tau=25.0, lambda_io=1.0)
    assert float(L) == pytest.approx(0.8950064146, rel=1e-9)

def test_lambda_io_scales_value_and_gradient_linearly():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.)], [[0, 1]])
    term = _term(nl, rs, 2)
    p1 = _pos(nl); L1 = term(p1, 25.0, 1.0); L1.backward()
    p2 = _pos(nl); L2 = term(p2, 25.0, 3.0); L2.backward()
    assert float(L2) == pytest.approx(3.0 * float(L1), rel=1e-12)
    assert torch.allclose(p2.grad, 3.0 * p1.grad, rtol=1e-10, atol=1e-14)

def test_w_mode_inv_deg_divides_by_degree_minus_one():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.), (25., 50.)], [[0, 1, 2]])
    a = float(_term(nl, rs, 2, "unit")(_pos(nl), 25.0, 1.0))
    b = float(_term(nl, rs, 2, "inv_deg")(_pos(nl), 25.0, 1.0))
    assert b == pytest.approx(a / 2.0, rel=1e-12)     # deg' = 3 -> 1/(3-1)

def test_converges_to_hard_lambda_minus_one_as_tau_shrinks():
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rg = RegionGrid(rs)
    rng = np.random.default_rng(7)
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, 30), rng.uniform(2, 98, 30))]
    nets = [sorted(rng.choice(30, int(rng.integers(2, 6)), replace=False).tolist())
            for _ in range(15)]
    nl = _nl(xy, nets)
    term = _term(nl, rs, 16)
    L = float(term(_pos(nl), tau=1e-3, lambda_io=1.0))
    bm = rg.pin_region_bitmask(nl, nl.node_x, nl.node_y)
    hard = sum(max(bin(int(v)).count("1") - 1, 0) for v in bm)
    assert L == pytest.approx(float(hard), rel=1e-2)

def test_lambda_clamped_at_one_never_negative():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(5., 50.), (6., 50.)], [[0, 1]])       # both deep inside P0
    L = float(_term(nl, rs, 2)(_pos(nl), tau=0.01, lambda_io=1.0))
    assert L >= 0.0 and L < 1e-6

# ---------------------------------------------------------------- gradients
def test_gradient_matches_fp64_autograd_reference():
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rng = np.random.default_rng(8)
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, 20), rng.uniform(2, 98, 20))]
    nets = [sorted(rng.choice(20, int(rng.integers(2, 5)), replace=False).tolist())
            for _ in range(10)]
    nl = _nl(xy, nets)
    term = _term(nl, rs, 16)
    pos = _pos(nl); term(pos, 12.0, 1.0).backward()
    g = pos.grad.clone()
    h = 1e-6
    for i in (0, 3, 7):
        for off in (0, nl.num_physical):
            pp = pos.detach().clone(); pp[i + off] += h
            pm = pos.detach().clone(); pm[i + off] -= h
            fd = (float(term(pp, 12.0, 1.0)) - float(term(pm, 12.0, 1.0))) / (2 * h)
            assert float(g[i + off]) == pytest.approx(fd, abs=2e-5)

def test_no_nan_on_pathological_saturated_configuration():
    """probe1 fixture: several nodes of one net deep inside the same region, tiny tau."""
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(5., 50.), (6., 50.), (7., 50.), (8., 50.)], [[0, 1, 2, 3]])
    term = _term(nl, rs, 2)
    pos = _pos(nl)
    term(pos, tau=1e-3, lambda_io=1.0).backward()
    assert torch.isfinite(pos.grad).all()

def test_fixed_and_filler_nodes_get_zero_gradient():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.), (30., 50.)], [[0, 1], [1, 2]])
    n_filler = 4
    term = _term(nl, rs, 2, num_movable=2, n_filler=n_filler)   # node 2 is a terminal
    n_all = nl.num_physical + n_filler
    pos = _pos(nl, n_filler=n_filler)
    term(pos, 25.0, 1.0).backward()
    g = pos.grad
    assert float(g[2].abs()) == 0.0 and float(g[n_all + 2].abs()) == 0.0
    assert float(g[nl.num_physical:n_all].abs().sum()) == 0.0
    assert float(g[:2].abs().sum()) > 0.0                       # movable ones do move

def test_io_grad_l1_is_unweighted_and_matches_manual_norm():
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rng = np.random.default_rng(9)
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, 25), rng.uniform(2, 98, 25))]
    nets = [sorted(rng.choice(25, 3, replace=False).tolist()) for _ in range(12)]
    nl = _nl(xy, nets)
    term = _term(nl, rs, 16)
    pos = _pos(nl); term(pos, 9.0, 1.0).backward()
    assert term.io_grad_l1(pos.detach(), 9.0) == pytest.approx(float(pos.grad.abs().sum()), rel=1e-9)

def test_diagnostics_shape_and_frac_soft_monotonicity():
    """Clustered fixture (coordinator resolution, see task-2a-report.md Concern 1):
    30 nets of 3 nodes each, with a given net's 3 members co-located at one
    (random) point. For such a net, S_{e,k} = deg'*ell_k (a single shared ell,
    repeated deg' times), so lambda_e = sum_k [1 - (1-p_k)^deg']. f(x)=(1-x)^deg'
    is convex for deg'>=2, so sum_k f(p_k) is Schur-convex in p; raising tau moves
    softmax(p) toward uniform in the majorization order (standard softmax-
    temperature fact), and a Schur-convex sum can only decrease under that move --
    hence lambda_e = K - sum_k(1-p_k)^deg' is monotonically non-decreasing (here,
    strictly increasing for generic non-equidistant positions) in tau. This is a
    theorem for this fixture. frac_soft's direction is separately a theorem for
    ANY fixture: p_max = max_k p_k is Schur-convex too, so it is non-increasing
    in tau regardless of geometry -- which is why frac_soft's assertion already
    held even on the old random-topology fixture.
    The random-topology direction is NOT an invariant: see task-2a-report.md's
    14-point tau sweep, where soft_lambda_sum decreased monotonically the whole
    way because degree-3 nets independent of position already sit near their
    per-net ceiling at the hard limit, leaving smearing nowhere to go but down --
    finite-sample noise from that fixture's topology, not a property of the
    surrogate.
    """
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rng = np.random.default_rng(10)
    n_nets = 30
    cx = rng.uniform(15, 85, n_nets); cy = rng.uniform(15, 85, n_nets)
    xy = [(float(cx[e]), float(cy[e])) for e in range(n_nets) for _ in range(3)]
    nets = [[3 * e, 3 * e + 1, 3 * e + 2] for e in range(n_nets)]
    nl = _nl(xy, nets)
    term = _term(nl, rs, 16)
    pos = _pos(nl).detach()
    taus = (0.25, 2.5, 25.0)
    diags = [term.diagnostics(pos, tau=t) for t in taus]
    for d in diags:
        assert d["grad_share"].shape == (7,)
        assert d["grad_share"].sum() == pytest.approx(1.0, rel=1e-6)
    lam_sums = [d["soft_lambda_sum"] for d in diags]
    fracs = [d["frac_soft"] for d in diags]
    assert lam_sums[0] < lam_sums[1] < lam_sums[2]
    assert fracs[0] <= fracs[1] <= fracs[2]

# ---------------------------------------------------------------- margin (design v2 sec 9.2)
@pytest.mark.parametrize("D,expect_grad", [(0.5, -3.432596e-01), (2.5, -2.924234e-01),
                                           (5.0, -2.000000e-01), (12.5, -1.897035e-02)])
def test_margin_pushes_cells_deeper_inside(D, expect_grad):
    """d_star is NEGATIVE inside a region; D = depth = -d_star.
    dL_margin/dD must be < 0 (force points inward)."""
    d = torch.tensor([-D], dtype=torch.float64, device=DEV, requires_grad=True)
    L = margin_penalty(d, m=5.0, tau_m=2.5)
    L.backward()
    dL_dD = -float(d.grad)          # chain: d_star = -D
    assert dL_dD < 0.0
    assert dL_dD == pytest.approx(expect_grad, rel=1e-6)

def test_margin_is_negligible_far_from_boundary():
    d = torch.tensor([-50.0], dtype=torch.float64, device=DEV, requires_grad=True)
    margin_penalty(d, m=5.0, tau_m=2.5).backward()
    assert abs(float(d.grad)) < 1e-6

def test_v1_wrong_sign_form_points_the_other_way():
    """Regression guard: design v1 used softplus(-d/tau_m), which pushes cells
    TOWARD the boundary. Pin the wrongness so it can never come back."""
    d = torch.tensor([-5.0], dtype=torch.float64, device=DEV, requires_grad=True)
    torch.nn.functional.softplus(-d / 2.5).sum().backward()
    wrong_dL_dD = -float(d.grad)
    d2 = torch.tensor([-5.0], dtype=torch.float64, device=DEV, requires_grad=True)
    margin_penalty(d2, m=5.0, tau_m=2.5).backward()
    right_dL_dD = -float(d2.grad)
    assert wrong_dL_dD > 0.0 and right_dL_dD < 0.0

def test_margin_enters_forward_only_when_enabled():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.)], [[0, 1]])
    term = _term(nl, rs, 2)
    base = float(term(_pos(nl), 25.0, 1.0))
    off = float(term(_pos(nl), 25.0, 1.0, lambda_margin=0.0, margin_m=5.0, margin_tau=2.5))
    on = float(term(_pos(nl), 25.0, 1.0, lambda_margin=1.0, margin_m=5.0, margin_tau=2.5))
    assert off == pytest.approx(base, rel=1e-12)
    assert on > base

# ---------------------------------------------------------------- cross-check vs T0 probe3
def test_probe3_grad_loo_matches_reference():
    """Coordinator follow-up on a T0 finding: commit 1ddcf12 fixed a bug in
    ioplace.diagnostics.probes_m2.probe3_grad's leave-one-out backward
    (grad_loo / stable_p_and_ell). Close the loop by checking that the fixed
    logic agrees with IoTermRef's autograd-based dL_IO/dpos on the same coords.

    probe3_grad is script-style (loads a real ISPD netlist from a hardcoded
    path via load_netlist), so it is not directly callable here. This test
    ports its math (stable_p_and_ell + grad_loo, unchanged) as a minimal shim
    that takes the (rects, pin2node, pin2net) tables directly instead of a
    loaded netlist. The shim's sdf() assumes one rect per region, which holds
    for make_grid_regions (see rect_table: each grid RegionSpec contributes
    exactly one rect, in region order) -- true for every fixture in this file.

    Both paths differentiate through the SAME numerically-stable ell =
    log(1-p) form (soft_assign.chunk_p_ell and probe3_grad.stable_p_and_ell
    compute it with the same two formulas: t/s at the argmax column, (s-e)/s
    elsewhere), just via two different autograd routes (direct autograd
    through ell for IoTermRef; a detached leave-one-out coefficient `c` fed
    through a surrogate `sum(p*c)` for probe3). They should therefore agree
    to near machine precision in float64. rtol/atol chosen to match this
    file's other exact (non finite-difference) gradient-equality checks, e.g.
    test_lambda_io_scales_value_and_gradient_linearly's rtol=1e-10/atol=1e-14.
    """
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(25., 50.), (75., 50.)], [[0, 1]])
    K = 2
    tau = 25.0

    # --- reference: IoTermRef's autograd path ---
    term = _term(nl, rs, K)
    pos = _pos(nl)
    term(pos, tau, lambda_io=1.0).backward()
    ref_gx = pos.grad[:nl.num_physical].clone()
    ref_gy = pos.grad[nl.num_physical:2 * nl.num_physical].clone()

    # --- shim: probe3_grad's fixed grad_loo / stable_p_and_ell, ported to take
    # rects directly instead of a loaded netlist ---
    rects, _ = rect_table(rs)                      # (K,4); one rect per grid region
    rects_t = torch.as_tensor(rects, dtype=torch.float64, device=DEV)
    pin2node = torch.as_tensor(nl.pin2node.astype(np.int64), device=DEV)
    pin2net = torch.as_tensor(nl.pin2net.astype(np.int64), device=DEV)
    x0 = torch.as_tensor(nl.node_x, dtype=torch.float64, device=DEV)
    y0 = torch.as_tensor(nl.node_y, dtype=torch.float64, device=DEV)
    n_nets = nl.num_nets

    def sdf(x, y):
        dx = torch.maximum(rects_t[:, 0][None] - x[:, None], x[:, None] - rects_t[:, 2][None])
        dy = torch.maximum(rects_t[:, 1][None] - y[:, None], y[:, None] - rects_t[:, 3][None])
        return dx.clamp(min=0) + dy.clamp(min=0) + torch.maximum(dx, dy).clamp(max=0)

    def stable_p_and_ell(x, y, tau, floor=1e-30):
        z = -sdf(x, y) / tau
        m, am = z.max(1, keepdim=True)
        e = torch.exp(z - m)                       # (N,K), e[argmax]=1
        onehot = torch.zeros_like(e).scatter_(1, am, 1.0)
        t = (e * (1 - onehot)).sum(1, keepdim=True).clamp_min(floor)   # sum over j != argmax, EXACT
        s = 1.0 + t
        p = e / s
        omp = (s - e) / s                                             # 1-p, cancellation only at argmax
        omp = torch.where(onehot.bool(), (t / s).expand_as(omp), omp).clamp_min(floor)
        return p, torch.log(omp)

    def grad_loo(x, y, tau):
        """log-space leave-one-out backward: dL/dp_{i,k} = exp(S_{e,k} - ell_{i,k})."""
        x = x.detach().clone().requires_grad_(True); y = y.detach().clone().requires_grad_(True)
        p, ell = stable_p_and_ell(x, y, tau)
        ellp = ell[pin2node]
        S = torch.zeros((n_nets, K), dtype=torch.float64, device=DEV).index_add_(0, pin2net, ellp)
        lam = (1.0 - torch.exp(S)).sum(1)
        with torch.no_grad():
            c = torch.exp(S[pin2net] - ellp) * ((lam - 1.0) > 0).double()[pin2net].unsqueeze(1)
        surrog = (p[pin2node] * c).sum()              # coefficient acts on p, not ell
        return torch.autograd.grad(surrog, [x, y])

    shim_gx, shim_gy = grad_loo(x0, y0, tau)

    assert torch.allclose(ref_gx, shim_gx, rtol=1e-10, atol=1e-14)
    assert torch.allclose(ref_gy, shim_gy, rtol=1e-10, atol=1e-14)
