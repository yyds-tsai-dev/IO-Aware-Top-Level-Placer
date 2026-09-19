from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ioplace.ops.steiner_wirelength as sw
from ioplace.route_eval.topology import flute_tree


def _netlist(points, *, pin_nodes=None, offsets=None, num_movable=None):
    points = np.asarray(points, dtype=float)
    degree = len(points)
    pin_nodes = np.arange(degree, dtype=np.int64) if pin_nodes is None else np.asarray(pin_nodes)
    offsets = np.zeros_like(points) if offsets is None else np.asarray(offsets, dtype=float)
    return SimpleNamespace(
        flat_net2pin=np.arange(degree, dtype=np.int64),
        flat_net2pin_start=np.array([0, degree], dtype=np.int64),
        pin2node=pin_nodes,
        pin_offset_x=offsets[:, 0],
        pin_offset_y=offsets[:, 1],
        num_nets=1,
        num_movable=degree if num_movable is None else num_movable,
    )


def _pos(points, *, dtype=torch.float64, device="cpu", requires_grad=False):
    points = torch.as_tensor(points, dtype=dtype, device=device)
    return torch.cat((points[:, 0], points[:, 1])).requires_grad_(requires_grad)


def _raw_tree_for_one_interior_pin(_pins, **_kwargs):
    # Terminals 0..4, then Steiner nodes.  Only terminal 4 is strictly inside
    # the terminal bounding box, and its frozen parent is node 5.
    return ({
        "positions": np.array([
            [0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.],
            [1.5, 0.5], [0., 1.], [2., 1.],
        ]),
        "parents": np.array([6, 6, 7, 7, 5, 6, 7, 7]),
        "terminal_nodes": np.arange(5),
    }, {"unique_quantized_pins": 5})


def test_flute_tree_preserves_terminal_identity_quantization_and_zero_branches():
    pins = np.array([[0.00011, 0.00019], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    tree, meta = flute_tree(pins, coordinate_scale=1000.)
    terminal = tree["terminal_nodes"]
    np.testing.assert_allclose(tree["positions"][terminal], tree["snapped_pins"])
    assert np.all(terminal >= 0)
    assert np.all(tree["parents"][terminal] >= meta["unique_quantized_pins"])

    # This cross has a terminal colocated with a Steiner node.  Its raw branch
    # must survive even though the public geometric edge view omits zero edges.
    cross = np.array([[0., 0.], [0., 2.], [1., 1.], [2., 0.], [2., 2.]])
    raw, _ = flute_tree(cross)
    center = raw["terminal_nodes"][2]
    parent = raw["parents"][center]
    np.testing.assert_array_equal(raw["positions"][center], raw["positions"][parent])

    duplicate = np.array([[0., 0.], [10., 10.], [2., 6.], [2., 6.], [7., 4.]])
    raw, _ = flute_tree(duplicate)
    a, b = raw["terminal_nodes"][2:4]
    assert a != b
    np.testing.assert_array_equal(raw["positions"][[a, b]], duplicate[[2, 3]])


def test_eq5_closed_form_value_and_gradient_on_frozen_interior_branch(monkeypatch):
    monkeypatch.setattr(sw, "flute_tree", _raw_tree_for_one_interior_pin)
    points = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    weights = torch.tensor([2.], dtype=torch.float64)
    op = sw.FrozenSteinerWirelength(_netlist(points), 5, net_weights=weights)
    pos = _pos(points, requires_grad=True)
    meta = op.rebuild(pos)
    value = op(pos, torch.tensor(1., dtype=pos.dtype))
    value.backward()

    # 2 net weight * (phi(-.5) + phi(.5)), phi(d)=d*tanh(d/2).
    assert value.item() == pytest.approx(0.48983732480741826)
    expected = 2. * (-0.4799223746052669)
    assert pos.grad[4].item() == pytest.approx(expected)
    assert pos.grad[9].item() == pytest.approx(-expected)
    assert torch.count_nonzero(pos.grad).item() == 2
    assert meta["interior_pins"] == meta["interior_branches"] == 1


def test_rebuild_freezes_topology_and_finite_difference_matches_autograd(monkeypatch):
    monkeypatch.setattr(sw, "flute_tree", _raw_tree_for_one_interior_pin)
    points = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    op = sw.FrozenSteinerWirelength(_netlist(points), 5)
    base = _pos(points)
    op.rebuild(base)
    generation = op.generation
    probe = base.clone().requires_grad_(True)
    grad, = torch.autograd.grad(op(probe, 0.7), probe)
    eps = 1e-6
    plus, minus = base.clone(), base.clone()
    plus[4] += eps
    minus[4] -= eps
    finite = (op(plus, 0.7) - op(minus, 0.7)) / (2 * eps)
    assert finite.item() == pytest.approx(grad[4].item(), rel=2e-6, abs=2e-7)
    assert op.generation == generation


def test_masks_degrees_boundaries_fixed_nodes_and_uses_current_weights(monkeypatch):
    monkeypatch.setattr(sw, "flute_tree", _raw_tree_for_one_interior_pin)
    points = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    weights = torch.tensor([1.])
    op = sw.FrozenSteinerWirelength(_netlist(points, num_movable=4), 5,
                                    net_weights=weights)
    pos = _pos(points, dtype=torch.float32, requires_grad=True)
    op.rebuild(pos)
    weights.mul_(3.)
    value = op(pos, 1.)
    assert value.item() == pytest.approx(3. * 0.2449186624, rel=2e-6)
    assert torch.autograd.grad(value, pos, allow_unused=True)[0].count_nonzero().item() == 0

    for degree in (2, 3):
        small = points[:degree]
        small_op = sw.FrozenSteinerWirelength(_netlist(small), degree)
        meta = small_op.rebuild(_pos(small))
        assert small_op(_pos(small), 1.).item() == 0.
        assert meta["wa_only_degree_le_3_nets"] == 1

    masked = sw.FrozenSteinerWirelength(_netlist(points), 5, net_mask=np.array([False]))
    meta = masked.rebuild(_pos(points))
    assert masked(_pos(points), 1.).item() == 0.
    assert meta["masked_nets"] == 1


def test_actual_flute_classifies_only_strict_interior_terminal_branch():
    points = np.array([[0., 0.], [0., 10.], [4., 4.], [10., 0.], [10., 10.]])
    op = sw.FrozenSteinerWirelength(_netlist(points), 5)
    meta = op.rebuild(_pos(points))
    assert meta["interior_pins"] == meta["interior_branches"] == 1
    assert op(_pos(points), 1.).item() >= 0.


def test_actual_flute_fig2_fixture_has_published_interior_gradients():
    points = np.array([(2., 8.), (0., 5.), (4., 6.), (9., 4.), (2., 2.), (6., 0.)])
    pos = _pos(points, requires_grad=True)
    op = sw.FrozenSteinerWirelength(_netlist(points), 6)
    meta = op.rebuild(pos)
    value = op(pos, 1.)
    gradient, = torch.autograd.grad(value, pos)
    assert meta["interior_pins"] == 2
    assert value.item() == pytest.approx(3.0463766238230594)
    np.testing.assert_allclose(gradient[:6].numpy(), [0., 0., 1.181568497569791, 0., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(gradient[6:].numpy(), [0., 0., 0., 0., -1.181568497569791, 0.], atol=1e-12)


def test_whole_pin_boundary_rule_offsets_and_explicit_over_degree(monkeypatch):
    boundary = np.array([(0., 13.), (11., 1.), (18., 4.), (5., 3.),
                         (6., 3.), (6., 16.), (9., 18.)])
    pos = _pos(boundary, requires_grad=True)
    op = sw.FrozenSteinerWirelength(_netlist(boundary), 7)
    op.rebuild(pos)
    gradient, = torch.autograd.grad(op(pos, 1.), pos)
    # Pin 2 is on xmax: strict whole-pin screening excludes both axes, even
    # though its adjacent raw branch has a nonzero y displacement.
    assert gradient[2].item() == 0.
    assert gradient[7 + 2].item() == 0.

    monkeypatch.setattr(sw, "flute_tree", _raw_tree_for_one_interior_pin)
    physical = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    offsets = np.zeros_like(physical)
    offsets[4] = [.25, .2]
    nodes = physical - offsets
    offset_op = sw.FrozenSteinerWirelength(_netlist(nodes, offsets=offsets), 5)
    shifted = _pos(nodes, requires_grad=True)
    offset_op.rebuild(shifted)
    assert offset_op(shifted, 1.).item() == pytest.approx(0.24491866240370913)

    excluded = sw.FrozenSteinerWirelength(_netlist(physical), 5, max_degree=4)
    meta = excluded.rebuild(_pos(physical))
    assert meta["over_degree_nets"] == 1
    assert excluded(_pos(physical), 1.).item() == 0.


def test_coincident_interior_terminal_has_zero_correction():
    cross = np.array([[0., 0.], [0., 2.], [1., 1.], [2., 0.], [2., 2.]])
    op = sw.FrozenSteinerWirelength(_netlist(cross), 5)
    meta = op.rebuild(_pos(cross))
    assert meta["interior_pins"] == 1
    assert op(_pos(cross), .5).item() == 0.


def test_real_frozen_topology_gradcheck_extreme_ratio_and_gamma_validation():
    points = np.array([(2., 8.), (0., 5.), (4., 6.), (9., 4.), (2., 2.), (6., 0.)])
    pos = _pos(points, requires_grad=True)
    op = sw.FrozenSteinerWirelength(_netlist(points), 6)
    op.rebuild(pos)
    assert torch.autograd.gradcheck(lambda value: op(value, .73), (pos,),
                                    eps=1e-6, atol=2e-5, rtol=2e-4)

    extreme = pos.detach().clone().requires_grad_(True)
    extreme.data[2] = 1e20
    value = op(extreme, 1e-6)
    gradient, = torch.autograd.grad(value, extreme)
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    for gamma in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="gamma"):
            op(pos, gamma)


def test_shared_node_pin_offsets_accumulate_and_fillers_have_zero_gradient(monkeypatch):
    def shared_tree(_pins, **_kwargs):
        positions = np.array([
            [0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.], [1.2, 1.2],
            [1.5, .5], [.5, 1.5], [0., 1.], [2., 1.],
        ])
        return ({"positions": positions,
                 "parents": np.array([8, 8, 9, 9, 6, 7, 8, 9, 9, 9]),
                 "terminal_nodes": np.arange(6)}, {"unique_quantized_pins": 6})

    monkeypatch.setattr(sw, "flute_tree", shared_tree)
    physical = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.], [1.2, 1.2]])
    nodes = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.], [8., 8.], [9., 9.]])
    pin_nodes = np.array([0, 1, 2, 3, 4, 4])
    offsets = np.zeros_like(physical)
    offsets[5] = [.2, .2]
    nl = _netlist(physical, pin_nodes=pin_nodes, offsets=offsets)
    pos = _pos(nodes, requires_grad=True)
    op = sw.FrozenSteinerWirelength(nl, 7)
    op.rebuild(pos)
    gradient, = torch.autograd.grad(op(pos, 1.), pos)
    assert gradient[4].item() == pytest.approx(0.1668511923413813)
    assert gradient[7 + 4].item() == pytest.approx(0.1843623539675381)
    assert torch.count_nonzero(gradient[[5, 6, 12, 13]]).item() == 0


def test_rebuild_rejects_raw_tree_without_terminal_to_steiner_parents(monkeypatch):
    def malformed(_pins, **_kwargs):
        return ({"positions": np.zeros((8, 2)),
                 "parents": np.array([1, 6, 7, 7, 5, 6, 7, 7]),
                 "terminal_nodes": np.arange(5)}, {})

    monkeypatch.setattr(sw, "flute_tree", malformed)
    points = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    op = sw.FrozenSteinerWirelength(_netlist(points), 5)
    with pytest.raises(RuntimeError, match="terminal-to-Steiner"):
        op.rebuild(_pos(points))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_float32_float64_parity(monkeypatch):
    monkeypatch.setattr(sw, "flute_tree", _raw_tree_for_one_interior_pin)
    points = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [1., 1.]])
    for dtype in (torch.float32, torch.float64):
        values = []
        gradients = []
        for device in ("cpu", "cuda"):
            pos = _pos(points, dtype=dtype, device=device, requires_grad=True)
            op = sw.FrozenSteinerWirelength(_netlist(points), 5)
            op.rebuild(pos)
            value = op(pos, torch.tensor(.8, dtype=dtype, device=device))
            values.append(value.detach().cpu())
            gradients.append(torch.autograd.grad(value, pos)[0].cpu())
        torch.testing.assert_close(values[0], values[1], rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-6, atol=2e-7)
