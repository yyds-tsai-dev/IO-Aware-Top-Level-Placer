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
    centres, so it must default to 'center' and reject 'pin' before CUDA
    exactly like run_placement.py/run_placement_io.py do."""
    from ioplace.drivers.run_main_flow import build_parser, run_main_flow, run_soft_phase
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.node_anchor == "center"
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
