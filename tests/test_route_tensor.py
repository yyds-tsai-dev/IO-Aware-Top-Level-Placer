"""Physical union and sparse smooth crossings, including their smooth pieces."""
import importlib

import pytest
import torch


@pytest.fixture
def api():
    try:
        return importlib.import_module("ioplace.ops.route_tensor")
    except ModuleNotFoundError:
        pytest.fail("the route tensor primitives are not implemented")


@pytest.fixture(params=[("cpu", torch.float32), ("cpu", torch.float64),
                        ("cuda", torch.float32), ("cuda", torch.float64)])
def spec(request):
    device, dtype = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return device, dtype


def tensors(rows, spec):
    device, dtype = spec
    a = torch.tensor(rows, device=device, dtype=dtype).reshape(-1, 5)
    return dict(net_ids=a[:, 0].long(), axes=a[:, 1].long(),
                tracks=a[:, 2], low=a[:, 3], high=a[:, 4])


def test_union_overlaps_reversed_duplicates_and_net_separation(api, spec):
    data = tensors([[0, 0, 2, 3, 0], [0, 0, 2, 2, 5], [0, 0, 2, 3, 0],
                    [0, 0, 2, 8, 7], [0, 0, 4, 0, 3], [1, 0, 2, 1, 4],
                    [0, 1, 2, 1, 3], [0, 0, 2, 9, 9]], spec)
    result = api.union_intervals(**data)
    got = torch.stack([result[k] for k in ("net_ids", "axes", "tracks", "low", "high")], 1)
    want = torch.tensor([[0, 0, 2, 0, 5], [0, 0, 2, 7, 8], [0, 0, 4, 0, 3],
                         [0, 1, 2, 1, 3], [1, 0, 2, 1, 4]], device=spec[0], dtype=spec[1])
    torch.testing.assert_close(got, want)


def test_hard_crossings_clamp_outer_bins_and_use_internal_cuts(api, spec):
    # 3 columns, 2 rows: horizontal IDs 0..3, vertical IDs 4..6.
    data = tensors([[0, 0, -3, -2, 2.8], [1, 1, 5, -2, 2.8]], spec)
    x = torch.tensor([0., 1., 2., 3.], device=spec[0], dtype=spec[1])
    y = x[:3]
    c = api.grid_crossings(api.union_intervals(**data), x, y, .01)
    assert c["edge_ids"].tolist() == [0, 1, 6]
    assert c["net_ids"].tolist() == [0, 0, 1]
    torch.testing.assert_close(c["values"], torch.ones(3, device=spec[0], dtype=spec[1]))


def test_exact_hard_endpoint_cuts_and_transverse_ties(api, spec):
    x = torch.tensor([0., 1., 2., 3.], device=spec[0], dtype=spec[1])
    data = tensors([[0, 0, 1, 1, 2], [1, 1, 2, 1, 2]], spec)
    c = api.grid_crossings(api.union_intervals(**data), x, x, .3, hard=True)
    assert c["edge_ids"].tolist() == [3, 11]
    assert c["values"].tolist() == [1., 1.]


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -float("inf"), 0., -.1])
@pytest.mark.parametrize("hard", [False, True])
def test_crossings_reject_nonfinite_or_nonpositive_tau(api, tau, hard):
    x = torch.tensor([0., 1., 2.])
    union = api.union_intervals(**tensors([[0, 0, .4, .2, 1.8]], ("cpu", torch.float32)))
    with pytest.raises(ValueError, match="finite.*positive"):
        api.grid_crossings(union, x, x, tau, hard=hard)


def test_crossings_accept_zero_dimensional_tensor_tau(api, spec):
    x = torch.tensor([0., 1., 2.], device=spec[0], dtype=spec[1])
    union = api.union_intervals(**tensors([[0, 0, .4, .2, 1.8]], spec))
    tau = torch.tensor(.25, device=spec[0], dtype=spec[1])
    a = api.grid_crossings(union, x, x, tau)
    b = api.grid_crossings(union, x, x, .25)
    for key in a:
        torch.testing.assert_close(a[key], b[key])


@pytest.mark.parametrize("shape", [(1,), (2,), (1, 1)])
def test_crossings_reject_nonscalar_tau(api, shape):
    x = torch.tensor([0., 1., 2.])
    union = api.union_intervals(**tensors([[0, 0, .4, .2, 1.8]], ("cpu", torch.float32)))
    with pytest.raises(ValueError, match="scalar"):
        api.grid_crossings(union, x, x, torch.full(shape, .25))


def test_distinct_tracks_sum_physical_io_but_max_resource_occupancy(api, spec):
    x = torch.tensor([0., 1., 2.], device=spec[0], dtype=spec[1])
    y = torch.tensor([0., 3.], device=spec[0], dtype=spec[1])
    data = tensors([[0, 0, 1, .2, 1.8], [0, 0, 2, .2, 1.8],
                    [0, 0, 1, .2, 1.8]], spec)
    c = api.grid_crossings(api.union_intervals(**data), x, y, .01)
    assert c["values"].sum().item() == 2
    assert api.joint_demand(c, 1).item() == 1
    data = tensors([[0, 0, 1, .2, 1.8], [1, 0, 1, .2, 1.8]], spec)
    c = api.grid_crossings(api.union_intervals(**data), x, y, .01)
    assert api.joint_demand(c, 1).item() == 2


def test_empty_and_zero_length_have_zero_usage_and_gradient(api, spec):
    data = tensors([[0, 0, 1, 1, 1]], spec)
    data["low"].requires_grad_()
    x = torch.tensor([0., 1., 2.], device=spec[0], dtype=spec[1])
    c = api.grid_crossings(api.union_intervals(**data), x, x, .2)
    demand = api.joint_demand(c, 4)
    assert torch.count_nonzero(demand) == 0
    demand.sum().backward()
    torch.testing.assert_close(data["low"].grad, torch.zeros_like(data["low"]))


def test_background_changes_endpoint_congestion_gradient(api, spec):
    x = torch.tensor([0., 1., 2.], device=spec[0], dtype=spec[1])
    y = x[:2]
    gradients = []
    for background in (0., 1.):
        data = tensors([[0, 0, .4, .2, 1.05]], spec)
        data["high"].requires_grad_()
        c = api.grid_crossings(api.union_intervals(**data), x, y, .3)
        d = api.joint_demand(c, 1, torch.full((1,), background, device=spec[0], dtype=spec[1]))
        loss = api.congestion_cost(d, torch.ones_like(d))
        gradients.append(torch.autograd.grad(loss, data["high"])[0].item())
    assert 0 < gradients[0] < gradients[1]


def test_full_coordinate_gradient_away_union_and_kernel_ties(api):
    # Unique tracks and strict partial overlap leave a smooth local piece.
    net = torch.tensor([0, 0, 1])
    axes = torch.tensor([0, 0, 1])
    x = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
    y = torch.tensor([0., 1.4, 2.8], dtype=torch.float64)
    coords = torch.tensor([[1.25, .3, 2.13], [.38, .71, 1.87],
                           [1.14, .43, 1.54]], dtype=torch.float64, requires_grad=True)

    def objective(z):
        union = api.union_intervals(net, axes, z[:, 0], z[:, 1], z[:, 2])
        c = api.grid_crossings(union, x, y, .4)
        d = api.joint_demand(c, 7, torch.linspace(.1, .7, 7, dtype=z.dtype))
        return c["values"].sum() + api.congestion_cost(d, torch.ones_like(d))

    assert torch.autograd.gradcheck(objective, (coords,), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_duplicate_common_coordinate_motion_preserves_gradient(api):
    # Equal tracks are a frozen tie: perturb the shared coordinate together.
    z = torch.tensor([.8, .2, 1.1], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0., 1., 2.], dtype=z.dtype)
    def objective(v, copies):
        c = api.grid_crossings(api.union_intervals(torch.zeros(copies, dtype=torch.long),
            torch.zeros(copies, dtype=torch.long), v[0].expand(copies),
            v[1].expand(copies), v[2].expand(copies)), x, x, .4)
        return c["values"].sum()
    one, two = objective(z, 1), objective(z, 2)
    torch.testing.assert_close(one, two)
    torch.testing.assert_close(torch.autograd.grad(one, z)[0], torch.autograd.grad(two, z)[0])
    assert torch.autograd.gradcheck(lambda v: objective(v, 2), (z,))


def test_nested_union_prefix_max_does_not_leak_across_tracks(api, spec):
    # A long interval contains the next two; track 2 starts at a much smaller
    # high value. Float32 must retain tiny spans even with many large groups.
    data = tensors([[0, 0, 1, -10000, 10000], [0, 0, 1, -2, 2],
                    [0, 0, 1, 3, 4], [0, 0, 2, .001, .002],
                    [1, 0, 1, .003, .004]], spec)
    u = api.union_intervals(**data)
    torch.testing.assert_close(u["low"], torch.tensor([-10000, .001, .003], device=spec[0], dtype=spec[1]))
    torch.testing.assert_close(u["high"], torch.tensor([10000, .002, .004], device=spec[0], dtype=spec[1]))


def test_overlapping_union_shared_track_full_gradient(api, spec):
    device, dtype = spec
    z = torch.tensor([.83, .21, 1.13, .71, 1.82], device=device, dtype=dtype, requires_grad=True)
    x = torch.tensor([0., 1., 2.], device=device, dtype=dtype)
    ids = torch.zeros(2, device=device, dtype=torch.long)
    def loss(v):
        u = api.union_intervals(ids, ids, v[0].expand(2), v[[1, 3]], v[[2, 4]])
        c = api.grid_crossings(u, x, x, .4)
        demand = api.joint_demand(c, 4, torch.full((4,), .7, device=device, dtype=dtype))
        return c["values"].sum() + api.congestion_cost(demand, torch.ones_like(demand)) + (u["high"] - u["low"]).sum()
    actual = torch.autograd.grad(loss(z), z)[0]
    eps = .001 if dtype == torch.float32 else 1e-6
    numeric = []
    for index in range(z.numel()):
        plus, minus = z.detach().clone(), z.detach().clone()
        plus[index] += eps
        minus[index] -= eps
        numeric.append((loss(plus) - loss(minus)) / (2 * eps))
    torch.testing.assert_close(actual, torch.stack(numeric), atol=.004 if dtype == torch.float32 else 2e-5, rtol=.004)
    assert actual[2].item() == 0 and actual[3].item() == 0


def test_hard_resources_match_independent_cpu_evaluator(api, spec):
    import numpy as np
    from ioplace.route_eval.joint import ResourceGrid

    rng = np.random.default_rng(7201)
    rows, segments = [], []
    for index in range(101):
        axis = index % 2
        track, low, high = rng.uniform(-2, 5, 3)
        # Include exact endpoint and transverse ties.
        if index % 7 == 0:
            track, low, high = 1., 0., 2.
        rows.append([index % 5, axis, track, low, high])
        segments.append([[low, track], [high, track]] if axis == 0 else [[track, low], [track, high]])
    data = tensors(rows, spec)
    x = torch.tensor([0., .7, 1., 2., 4.], device=spec[0], dtype=spec[1])
    y = torch.tensor([0., 1., 3.], device=spec[0], dtype=spec[1])
    grid = ResourceGrid(x.cpu().numpy(), y.cpu().numpy())
    crossings = api.grid_crossings(api.union_intervals(**data), x, y, .2, hard=True)
    demand = api.joint_demand(crossings, grid.edge_count)
    expected = np.zeros(grid.edge_count)
    for net in range(5):
        expected[grid.edge_keys(np.asarray(segments)[np.arange(101) % 5 == net])] += 1
    torch.testing.assert_close(demand, torch.tensor(expected, device=spec[0], dtype=spec[1]))
