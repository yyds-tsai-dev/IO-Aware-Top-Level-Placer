import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.ops.io_term import IoTerm, IoTermRef, build_net_node_csr
from ioplace.ops.soft_assign import NODE_ANCHORS, anchor_offsets, rect_table
from ioplace.regions import make_grid_regions
from tests.test_io_term import DIE, _nl, _pos

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _straddler():
    """One 2-pin net. Cell 0 sits inside region 0. Cell 1's lower-left corner is
    in region 0 (x=44 < 50, y=44 < 50) but its centre is in region 3, the
    corner diagonally opposite region 0 (44 + 14/2 = 51 on both axes), so the
    anchor alone decides whether this net crosses a boundary.

    The straddle is diagonal, not axis-aligned, on purpose: this fixture used
    to put cell 1's centre in region 1, which is edge-adjacent to region 0
    (hop distance 1 in the D matrix used by test_ft_term_inherits_the_anchor_
    from_its_io_term below). FtTerm's own S4a formula is `max(D[home,k]-1,
    0)`, which ft_term.py's module docstring calls out as "not true pure-
    feedthrough": it is deliberately 0 at hop distance 1, so that fixture made
    ft_only bit-exact 0.0 under *both* anchors -- L_IO differed (0.0 vs 1.0)
    but ft_only could never show the anchor mattering at all, no matter how
    IoTerm/FtTerm anchored their coordinates. Region 3 is hop distance 2 from
    region 0 (D[0,3]=2 in that test's D), so `max(D-1,0)=1` there and ft_only
    becomes genuinely anchor-dependent. This was a defect in the original plan's
    fixture, not in the anchor implementation it was meant to exercise (design
    v2 sec 7 / P-F task 1 controller ruling, 2026-09-20)."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)          # boundaries at x=50, y=50
    nl = _nl([(10., 10.), (44., 44.)], [[0, 1]])
    nl.node_size_x = np.array([4., 14.])
    nl.node_size_y = np.array([4., 14.])
    return rs, nl


def _ref(nl, rs, anchor, cls=IoTermRef, **extra):
    rects, r2k = rect_table(rs)
    kw = dict(csr=build_net_node_csr(nl, 100), rects=rects, rect2region=r2k, K=rs.k,
              num_movable=nl.num_movable, num_physical=nl.num_physical,
              num_nodes=nl.num_physical, device=DEV, node_anchor=anchor)
    if anchor == "center":
        kw.update(node_size_x=nl.node_size_x, node_size_y=nl.node_size_y)
    kw.update(extra)
    return cls(**kw)


def test_node_anchors_tuple_is_the_three_spec_values():
    assert NODE_ANCHORS == ("lower_left", "center", "pin")


def test_anchor_offsets_lower_left_is_none_center_is_half_the_cell():
    assert anchor_offsets("lower_left", None, None, 3, device="cpu") == (None, None)
    dx, dy = anchor_offsets("center", np.array([4., 6., 8.]), np.array([2., 2., 2.]),
                            3, device="cpu")
    assert dx.tolist() == [2., 3., 4.] and dy.tolist() == [1., 1., 1.]
    assert dx.dtype == torch.float64


def test_anchor_offsets_rejects_pin_and_bad_sizes():
    with pytest.raises(ValueError, match="only in IoTermRef"):
        anchor_offsets("pin", np.ones(3), np.ones(3), 3, device="cpu")
    with pytest.raises(ValueError, match="requires node_size_x"):
        anchor_offsets("center", None, None, 3, device="cpu")
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        anchor_offsets("center", np.ones(2), np.ones(3), 3, device="cpu")


def test_center_anchor_makes_a_straddling_cell_cross_the_boundary():
    """The whole point of spec sec 7: at the same positions, the lower-left
    anchor scores this net as fully contained (L_IO = 0) while the centre
    anchor scores the one real crossing it will have after fence LG (L_IO = 1)."""
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    tau = 0.05
    ll = _ref(nl, rs, "lower_left")(pos, tau, 1.0)
    ce = _ref(nl, rs, "center")(pos, tau, 1.0)
    assert float(ll.detach()) == pytest.approx(0.0, abs=1e-9)
    assert float(ce.detach()) == pytest.approx(1.0, abs=1e-9)


def test_center_anchor_leaves_the_gradient_on_the_movable_nodes_only():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    term = _ref(nl, rs, "center")
    term(pos, 0.3, 1.0).backward()
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().sum()) > 0.0


def test_io_term_matches_io_term_ref_under_the_center_anchor():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    for chunk_budget in (10 ** 9, 1):
        prod = _ref(nl, rs, "center", cls=IoTerm, chunk_budget=chunk_budget)
        ref = _ref(nl, rs, "center")
        assert prod.node_anchor == "center"
        a = prod(pos, 0.3, 1.0)
        b = ref(pos, 0.3, 1.0)
        assert float(a.detach()) == pytest.approx(float(b.detach()), rel=1e-12)
        ga, = torch.autograd.grad(a, pos, retain_graph=False)
        gb, = torch.autograd.grad(b, pos, retain_graph=False)
        assert torch.allclose(ga, gb, rtol=1e-10, atol=1e-12)


def test_io_term_diagnostics_use_the_same_anchor_as_forward():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    prod = _ref(nl, rs, "center", cls=IoTerm, chunk_budget=10 ** 9)
    assert prod.diagnostics(pos.detach(), 0.05)["l_io"] == pytest.approx(1.0, abs=1e-9)
    plain = _ref(nl, rs, "lower_left", cls=IoTerm, chunk_budget=10 ** 9)
    assert plain.diagnostics(pos.detach(), 0.05)["l_io"] == pytest.approx(0.0, abs=1e-9)


def test_io_term_rejects_the_pin_anchor():
    rs, nl = _straddler()
    with pytest.raises(ValueError, match="only in IoTermRef"):
        _ref(nl, rs, "pin", cls=IoTerm, chunk_budget=10 ** 9)


def test_center_anchor_requires_sizes_over_all_physical_nodes():
    rs, nl = _straddler()
    rects, r2k = rect_table(rs)
    with pytest.raises(ValueError, match="requires node_size_x"):
        IoTermRef(csr=build_net_node_csr(nl, 100), rects=rects, rect2region=r2k,
                  K=rs.k, num_movable=2, num_physical=2, num_nodes=2, device=DEV,
                  node_anchor="center")


def test_unknown_anchor_is_rejected_by_both_classes():
    rs, nl = _straddler()
    for cls, extra in ((IoTermRef, {}), (IoTerm, {"chunk_budget": 10 ** 9})):
        with pytest.raises(ValueError, match="node_anchor"):
            _ref(nl, rs, "centre", cls=cls, **extra)


def test_ft_term_inherits_the_anchor_from_its_io_term():
    """D[0,3]=2 (hop distance 2, diagonally opposite home region 0) is the
    region _straddler()'s centre anchor now lands cell 1 in -- see
    _straddler()'s docstring for why the original hop-1 crossing (region 0 to
    the edge-adjacent region 1) could never make ft_only anchor-dependent,
    regardless of anchor plumbing. The FtTermRef cross-check is the strongest
    part of this test: it is an independent dense-autograd oracle that never
    touches _IoFn/_FtFn's chunked machinery, so its bit-exact agreement with
    the production FtTerm confirms the nonzero value is real, not an artifact
    of one implementation's anchor handling."""
    from ioplace.ops.ft_term import FtTerm, FtTermRef
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    D = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]], dtype=np.float64)
    values = {}
    for anchor in ("lower_left", "center"):
        io = _ref(nl, rs, anchor, cls=IoTerm, chunk_budget=10 ** 9)
        ft = FtTerm(io, D)
        ft.set_home(torch.zeros(io.n_active, dtype=torch.int64, device=DEV))
        values[anchor] = float(ft.ft_only(pos, 0.05).detach())

        io_ref = _ref(nl, rs, anchor)
        ft_ref = FtTermRef(io_ref, D)
        ft_ref.set_home(torch.zeros(io_ref.n_active, dtype=torch.int64, device=DEV))
        assert float(ft_ref.ft_only(pos, 0.05).detach()) == pytest.approx(
            values[anchor], rel=1e-12)
    assert values["lower_left"] != pytest.approx(values["center"], abs=1e-9)
    assert values["lower_left"] == pytest.approx(0.0, abs=1e-9)
    assert values["center"] == pytest.approx(1.0, abs=1e-9)


def test_io_term_ref_needs_a_pin_csr_for_the_pin_anchor():
    """Task 2 supplies the arm itself; Task 1 only guarantees the constructor
    cannot silently fall back to the node-level accumulation."""
    rs, nl = _straddler()
    with pytest.raises(ValueError, match="pin_csr"):
        _ref(nl, rs, "pin")


def test_run_io_rejects_the_pin_anchor_before_touching_cuda():
    from ioplace.drivers.run_placement_io import run_io
    with pytest.raises(ValueError, match="IoTermRef-only"):
        run_io("nonexistent.json", 16, "grid", 0, "out.json", node_anchor="pin")


def test_run_io_rejects_an_unknown_anchor():
    from ioplace.drivers.run_placement_io import run_io
    with pytest.raises(ValueError, match="node_anchor must be"):
        run_io("nonexistent.json", 16, "grid", 0, "out.json", node_anchor="centre")


def test_cli_defaults_node_anchor_to_center_and_offers_all_three():
    from ioplace.drivers.run_placement import build_parser
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--mode", "io", "--out", "o.json"])
    assert args.node_anchor == "center"
    action, = [a for a in parser._actions if a.dest == "node_anchor"]
    assert tuple(action.choices) == ("lower_left", "center", "pin")


def test_result_fields_carry_the_anchor():
    from ioplace.drivers.run_placement_io import RESULT_FIELDS
    assert "node_anchor" in RESULT_FIELDS


# ---------------------------------------------------------- fix round 1

def test_the_class_default_is_lower_left_while_the_driver_default_is_center():
    """P-F fix round 1, item 1 (review-task-1.md I-1): pins the two defaults
    the plan's own global-constraints.md says are wired together in exactly
    one place per driver -- the CLASS default on IoTermRef/IoTerm stays
    'lower_left' (fourteen existing construction sites supply no node sizes,
    which 'center' requires), while run_io's own keyword default is 'center'
    (the flag's default, spec sec 0/sec 7). Nothing failed before this test
    if either default silently reverted, since the CLI always passes the
    value explicitly."""
    import inspect
    from ioplace.drivers.run_placement_io import run_io
    for cls in (IoTermRef, IoTerm):
        assert inspect.signature(cls.__init__).parameters["node_anchor"].default == "lower_left"
    assert inspect.signature(run_io).parameters["node_anchor"].default == "center"


def test_the_center_offset_is_half_the_cell_not_the_whole_cell():
    """P-F fix round 1, item 2 (review-task-1.md I-2): a doubled offset left
    every other test in this file green (measured: L_IO and ft_only both
    1.0 either way), because the only assertion on the half-size was one
    level above the apply site (anchor_offsets's return value, not what
    _anchor_xy actually adds to x/y). x=40 with w=14: the centre (47) stays
    in region 0, but x+w (54) would cross into region 1 -- so this fails if
    the offset is ever applied twice (or is the whole cell instead of half).
    Verified to FAIL against a temporarily-doubled offset (see task-1-report.md)."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(10., 10.), (40., 40.)], [[0, 1]])
    nl.node_size_x = np.array([4., 14.]); nl.node_size_y = np.array([4., 14.])
    pos = _pos(nl, device=DEV)
    assert float(_ref(nl, rs, "center")(pos, 0.05, 1.0).detach()) == pytest.approx(0.0, abs=1e-9)


def test_run_main_flow_wires_node_anchor_like_the_other_drivers():
    """P-F fix round 1, item 4 (promoted from the reviewer's m-7):
    run_main_flow.py is a driver too, and freeze.py already freezes on cell
    centres, so it must default to 'center' (when a value is actually needed,
    i.e. --phase soft/all) and reject 'pin' before CUDA exactly like
    run_placement.py/run_placement_io.py do.

    The parser default itself is None, not 'center' -- fence-diagnostics fix
    (P-F Task 7 finding): run_main_flow must tell an omitted --node-anchor
    apart from an explicit 'center' so a --phase fence call that omits the
    flag can inherit the anchor freeze.json recorded from the soft phase
    instead of relabelling it 'center' (see test_main_flow_driver.py's
    test_phase_fence_inherits_the_recorded_node_anchor_when_omitted /
    test_phase_fence_raises_when_node_anchor_disagrees_with_freeze).
    run_placement.py/run_placement_io.py have no two-invocation split and
    keep their own 'center' default unchanged."""
    from ioplace.drivers.run_main_flow import build_parser, run_main_flow, run_soft_phase
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.node_anchor is None
    action, = [a for a in parser._actions if a.dest == "node_anchor"]
    assert tuple(action.choices) == ("lower_left", "center", "pin")
    with pytest.raises(ValueError, match="IoTermRef-only"):
        run_main_flow("nonexistent.json", "unused_out_dir", node_anchor="pin")
    with pytest.raises(ValueError, match="node_anchor must be"):
        run_main_flow("nonexistent.json", "unused_out_dir", node_anchor="centre")
    with pytest.raises(ValueError, match="IoTermRef-only"):
        run_soft_phase("nonexistent.json", "unused_out_dir", k=4, rtype="grid", seed=0,
                       node_anchor="pin")


def test_run_main_flow_rejects_node_anchor_under_phase_fence_before_any_side_effect():
    """P-F fix round 2 (review-task-1.md round 2): the test above calls
    run_main_flow with phase='all' (its default), which delegates to
    run_soft_phase -- and run_soft_phase re-validates node_anchor with
    identical messages, so that test would still pass even if run_main_flow's
    OWN checks (run_main_flow.py, right after the phase/init/norm_policy
    checks, before os.makedirs) were moved or deleted; run_soft_phase's
    checks would mask the regression. --phase fence is the one scenario that
    never calls run_soft_phase (IO/FT are off after the freeze -- see
    run_main_flow.py's own comment above its node_anchor check), so it is the
    only case that actually exercises run_main_flow's own validation in
    isolation. Also asserts the rejection happens before any side effect
    (out_dir is never created) -- the property the brief's 'before any CUDA
    allocation' requirement generalizes to."""
    import os
    import tempfile
    from ioplace.drivers.run_main_flow import run_main_flow

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "unused_out_dir")
        with pytest.raises(ValueError, match="IoTermRef-only"):
            run_main_flow("nonexistent.json", out_dir, phase="fence", node_anchor="pin")
        assert not os.path.exists(out_dir), \
            "run_main_flow created out_dir before validating node_anchor"

        with pytest.raises(ValueError, match="node_anchor must be"):
            run_main_flow("nonexistent.json", out_dir, phase="fence", node_anchor="centre")
        assert not os.path.exists(out_dir), \
            "run_main_flow created out_dir before validating node_anchor"


# ---------------------------------------------------------- task 2: pin arm

def _pin_ref(nl, rs):
    from ioplace.ops.io_term import build_net_pin_csr
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    return IoTermRef(csr=csr, rects=rects, rect2region=r2k, K=rs.k,
                     num_movable=nl.num_movable, num_physical=nl.num_physical,
                     num_nodes=nl.num_physical, device=DEV,
                     node_anchor="pin", pin_csr=build_net_pin_csr(nl, csr))


def test_build_net_pin_csr_aligns_with_the_node_csr():
    from ioplace.ops.io_term import build_net_pin_csr
    rs, nl = _straddler()
    csr = build_net_node_csr(nl, 100)
    pc = build_net_pin_csr(nl, csr)
    assert pc.pin_node.shape == pc.net_pos.shape == (len(nl.pin2node),)
    assert pc.net_pos.min() >= 0 and pc.net_pos.max() < len(csr.net_ids)
    assert np.array_equal(csr.net_ids[pc.net_pos], nl.pin2net[: len(pc.net_pos)])


def test_build_net_pin_csr_drops_pins_of_nets_the_node_csr_dropped():
    """Degree-1 nets and nets above ignore_net_degree are absent from NetCsr;
    their pins must be absent here too or net_pos would not index w."""
    nl = _nl([(10., 10.), (90., 10.), (10., 90.)], [[0], [0, 1], [0, 1, 2]])
    nl.node_size_x = np.ones(3)
    nl.node_size_y = np.ones(3)
    from ioplace.ops.io_term import build_net_pin_csr
    csr = build_net_node_csr(nl, ignore_net_degree=3)      # keeps only net 1
    pc = build_net_pin_csr(nl, csr)
    assert len(csr.net_ids) == 1 and csr.net_ids.tolist() == [1]
    assert pc.net_pos.tolist() == [0, 0]
    assert sorted(pc.pin_node.tolist()) == [0, 1]


def test_pin_arm_equals_the_lower_left_node_arm_when_every_node_has_one_pin_at_offset_zero():
    """The strongest available correctness statement: with one zero-offset pin
    per node the pin-level and node-level accumulations are the same sum, so
    the two arms must agree to the last bit of the fp64 accumulator."""
    rng = np.random.default_rng(3)
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, 12),
                                               rng.uniform(2, 98, 12))]
    nets = []
    for d in (2, 3, 2, 3, 2):
        nets.append(sorted(rng.choice(12, d, replace=False).tolist()))
    nl = _nl(xy, nets)
    nl.node_size_x = np.full(12, 3.)
    nl.node_size_y = np.full(12, 2.)
    pos = _pos(nl, device=DEV)
    node_arm = _ref(nl, rs, "lower_left")(pos, 0.5, 1.0)
    pin_arm = _pin_ref(nl, rs)(pos, 0.5, 1.0)
    assert float(pin_arm.detach()) == float(node_arm.detach())
    g_node, = torch.autograd.grad(node_arm, pos, retain_graph=True)
    g_pin, = torch.autograd.grad(pin_arm, pos)
    assert torch.allclose(g_node, g_pin, rtol=1e-12, atol=1e-14)


def test_pin_arm_double_counts_a_node_carrying_two_pins_of_one_net():
    """The defect m2-differentiable-io-design.md:105 rejected, made visible:
    the node arm dedups (net, node), the pin arm does not."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(10., 10.), (90., 10.)], [[0, 1, 1]])   # node 1 carries two pins
    nl.node_size_x = np.ones(2)
    nl.node_size_y = np.ones(2)
    csr = build_net_node_csr(nl, 100)
    assert csr.degrees.tolist() == [2] and csr.pin_degrees.tolist() == [3]
    from ioplace.ops.io_term import build_net_pin_csr
    assert len(build_net_pin_csr(nl, csr).pin_node) == 3


def test_pin_arm_moves_with_the_pin_offsets_not_the_cell_corner():
    """A cell whose lower-left corner and centre are both in region 0 but whose
    one pin sits past the boundary: only the pin arm sees the crossing."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(10., 10.), (44., 10.)], [[0, 1]])
    nl.node_size_x = np.array([2., 2.])
    nl.node_size_y = np.array([2., 2.])
    nl.pin_offset_x = np.array([0., 10.])            # pin of node 1 at x = 54
    pos = _pos(nl, device=DEV)
    assert float(_ref(nl, rs, "lower_left")(pos, 0.05, 1.0).detach()) == pytest.approx(0., abs=1e-9)
    assert float(_ref(nl, rs, "center")(pos, 0.05, 1.0).detach()) == pytest.approx(0., abs=1e-9)
    assert float(_pin_ref(nl, rs)(pos, 0.05, 1.0).detach()) == pytest.approx(1., abs=1e-9)


def test_pin_arm_rejects_a_margin_and_diagnostics():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    term = _pin_ref(nl, rs)
    with pytest.raises(ValueError, match="lambda_margin"):
        term(pos, 0.3, 1.0, lambda_margin=1.0, margin_m=1.0)
    with pytest.raises(NotImplementedError, match="pin"):
        term.diagnostics(pos.detach(), 0.3)


def test_ft_term_ref_rejects_a_pin_anchored_io_term():
    from ioplace.ops.ft_term import FtTermRef
    rs, nl = _straddler()
    with pytest.raises(ValueError, match="pin"):
        FtTermRef(_pin_ref(nl, rs), np.zeros((4, 4)))


# ---------------------------------------------- task 2 fix round 1

def test_pin_arm_forward_and_backward_scatter_correctly_for_a_repeated_pin_index():
    """test_pin_arm_double_counts_a_node_carrying_two_pins_of_one_net only
    checks PinCsr's shape/counts; it never runs the repeated pin_node entry
    through forward+backward. The whole point of the pin arm is that repeat,
    so pin its VALUE and its GRADIENT against an independent ground truth: a
    physical-layout twin where node 1 is split into two separate movable
    nodes at the identical position. The node arm's own (net, node) dedup CSR
    then sees exactly three distinct (position) contributions -- the same
    triple of floating-point values, accumulated in the same order, that the
    pin arm's gather (x[pin_node] with pin_node=[0, 1, 1]) produces directly.
    Forward value must therefore match bit-for-bit (`==`, not `approx`), and
    node 1's single gradient in the pin case must equal the SUM of the two
    split nodes' gradients in the twin case: a wrong scatter target (e.g.
    pin_net_idx swapped for pin_node) or a dropped duplicate (only one of the
    two pins contributing) would break one or both."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    tau = 12.5   # bias table (see test_pin_arm_bias_grows_from_zero_...): nonzero here

    nl_pin = _nl([(10., 10.), (90., 10.)], [[0, 1, 1]])   # node 1 carries two pins
    nl_pin.node_size_x = np.ones(2)
    nl_pin.node_size_y = np.ones(2)
    pos_pin = _pos(nl_pin, device=DEV)
    pin_term = _pin_ref(nl_pin, rs)
    L_pin = pin_term(pos_pin, tau, 1.0)
    g_pin, = torch.autograd.grad(L_pin, pos_pin)

    # ground truth: node 1 "split" into two physically-coincident nodes, one
    # net over all three -- literally the same triple of (position) values.
    nl_dup = _nl([(10., 10.), (90., 10.), (90., 10.)], [[0, 1, 2]])
    nl_dup.node_size_x = np.ones(3)
    nl_dup.node_size_y = np.ones(3)
    pos_dup = _pos(nl_dup, device=DEV)
    dup_term = _ref(nl_dup, rs, "lower_left")
    L_dup = dup_term(pos_dup, tau, 1.0)
    g_dup, = torch.autograd.grad(L_dup, pos_dup)

    assert float(L_pin.detach()) == float(L_dup.detach())
    assert float(L_pin.detach()) > 0.0    # sanity: not a vacuously-zero comparison

    n_pin, n_dup = nl_pin.num_physical, nl_dup.num_physical
    gx_pin, gy_pin = g_pin[:n_pin], g_pin[n_pin:2 * n_pin]
    gx_dup, gy_dup = g_dup[:n_dup], g_dup[n_dup:2 * n_dup]

    # node 0 (one pin, both cases) is untouched by the duplication.
    assert float(gx_pin[0]) == float(gx_dup[0])
    assert float(gy_pin[0]) == float(gy_dup[0])
    # node 1's gradient must be the SUM of the two split nodes' gradients --
    # the multiplicity check a lost duplicate or a wrong scatter target fails.
    assert float(gx_pin[1]) == float(gx_dup[1]) + float(gx_dup[2])
    assert float(gy_pin[1]) == float(gy_dup[1]) + float(gy_dup[2])
    assert float(gx_pin[1]) != 0.0 or float(gy_pin[1]) != 0.0


def test_pin_arm_bias_grows_from_zero_and_stays_above_the_node_arm():
    """Runnable reproduction of the sec 7 bias table (task-2-report.md): the
    pin arm's extra (P,K) row for node 1's duplicate pin can only add mass to
    its net's soft assignment sum, never remove it, so the pin arm's soft
    Sum_k q_k is >= the node arm's at every tau, exactly 0 while the softmax
    stays hard, and grows once tau softens it. Deliberately does not assert
    the exact float value at any tau (that is a property of softmax_stats'
    numerics, not of this task's contract) -- only the structure that
    matters: zero in the hard regime, pin >= node (within the noise band)
    throughout, strictly increasing once the signal clears that band, and
    distinctly nonzero by the largest tau probed.

    tau=2.0 sits right at the float64 noise floor (~4.2e-11 measured on this
    host, DEV-dependent -- a different GPU/CUDA build or a CPU fallback could
    flip its sign) and gets only a loose sanity bound, never an exact-zero,
    ordering or sign claim against its neighbours; tau>=3.0's ~1.7e-7 and up
    is two-plus orders of magnitude clear of that noise and gets the real
    monotonicity check (round 2 fix, review-task-2.md)."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(10., 10.), (90., 10.)], [[0, 1, 1]])   # node 1 carries two pins
    nl.node_size_x = np.ones(2)
    nl.node_size_y = np.ones(2)
    pos = _pos(nl, device=DEV)
    node_term = _ref(nl, rs, "lower_left")
    pin_term = _pin_ref(nl, rs)

    # noise band: ~20x the ~4.2e-11 noise observed at tau=2.0, ~180x below the
    # genuine ~1.7e-7 signal at tau=3.0 -- comfortably separates "zero" from
    # "real" without depending on the sign of a 1e-11 quantity.
    NOISE_TOL = 1e-9

    taus = (0.05, 1.0, 2.0, 3.0, 5.0, 12.5)
    biases = []
    print("\ntau       node_lam       pin_lam        bias(pin-node)")
    for tau in taus:
        x, y = node_term._split_xy(pos)
        _, lam_node, _ = node_term._forward_io(x, y, tau)
        x2, y2 = pin_term._split_xy(pos)
        _, lam_pin, _ = pin_term._forward_io(x2, y2, tau)
        bias = float(lam_pin.detach()) - float(lam_node.detach())
        biases.append(bias)
        print(f"{tau:<9} {float(lam_node.detach()):.8f}   {float(lam_pin.detach()):.8f}   {bias:+.8f}")

    # hard regime (tau 0.05, 1.0): zero within the noise band, not literal ==.
    assert abs(biases[0]) < NOISE_TOL
    assert abs(biases[1]) < NOISE_TOL
    # tau=2.0: at the noise floor -- weak sanity only, no sign/ordering claim.
    assert abs(biases[2]) < 1e-6
    # pin arm never meaningfully below the node arm, at any tau probed.
    assert all(b >= -NOISE_TOL for b in biases)
    # from tau=3.0 on the signal is well clear of noise: strictly increasing,
    # and clearly nonzero by the largest tau.
    tail = biases[3:]
    assert all(a < b for a, b in zip(tail, tail[1:]))
    assert tail[-1] > 1e-3, "expected the bias to be clearly nonzero by the largest tau probed"
