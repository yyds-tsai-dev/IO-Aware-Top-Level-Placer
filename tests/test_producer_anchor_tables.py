import numpy as np
import pytest
import torch
from ioplace.producer import hull

DIE = (0.0, 0.0, 8.0, 8.0)
SQ = np.array([[2., 2.], [6., 2.], [6., 6.], [2., 6.]])       # CCW inner square
BIG = np.array([[0., 0.], [8., 0.], [8., 8.], [0., 8.]])      # CCW whole die


def _bin(ix, iy, lattice=8):
    return iy * lattice + ix


def test_nearest_on_polygon_boundary_inside_and_outside():
    px = torch.tensor([0.5, 3.5, 3.5], dtype=torch.float64)
    py = torch.tensor([0.5, 2.5, 3.5], dtype=torch.float64)
    proj, inside = hull.nearest_on_polygon_boundary(px, py, SQ)
    assert list(inside.tolist()) == [False, True, True]
    assert proj[0].tolist() == pytest.approx([2.0, 2.0])   # corner of the square
    assert proj[1].tolist() == pytest.approx([3.5, 2.0])   # nearest edge is the bottom
    assert proj[2].tolist() == pytest.approx([3.5, 2.0])   # tie -> first edge in order


def test_pull_table_is_zero_inside_and_the_projection_offset_outside():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu")
    assert t.k == 2 and t.lattice == 8
    b_out = _bin(0, 0)          # bin centre (0.5, 0.5), outside SQ
    b_in = _bin(3, 3)           # bin centre (3.5, 3.5), inside SQ
    assert bool(t.pull_on[0, b_out]) is True
    assert bool(t.pull_on[0, b_in]) is False
    assert t.pull_off[0, b_out].double().tolist() == pytest.approx([1.5, 1.5], abs=1e-2)
    assert t.pull_off[0, b_in].double().tolist() == pytest.approx([0.0, 0.0])
    # every bin centre is inside BIG, so region 1 never pulls
    assert not bool(t.pull_on[1].any())


def test_push_count_and_mean_offset():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu")
    # region 0's foreign hulls = {BIG}, which contains every bin centre
    assert int(t.push_cnt[0].min()) == 1 and int(t.push_cnt[0].max()) == 1
    # region 1's foreign hulls = {SQ}: only the 4x4 central block is inside
    inside = t.push_cnt[1].reshape(8, 8).numpy()
    expect = np.zeros((8, 8), dtype=np.uint8)
    expect[2:6, 2:6] = 1
    assert np.array_equal(inside, expect)
    b = _bin(3, 2)              # centre (3.5, 2.5): nearest point of dSQ is (3.5, 2.0)
    assert t.push_off[1, b].double().tolist() == pytest.approx([0.0, -0.5], abs=1e-2)
    assert t.push_off[1, _bin(0, 0)].double().tolist() == pytest.approx([0.0, 0.0])


def test_reconstructed_anchor_matches_the_exact_projection_within_fp16():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu")
    ix, iy = np.meshgrid(np.arange(8), np.arange(8), indexing="xy")
    cx = (ix.ravel() + 0.5).astype(np.float64)
    cy = (iy.ravel() + 0.5).astype(np.float64)
    proj, inside = hull.nearest_on_polygon_boundary(
        torch.as_tensor(cx), torch.as_tensor(cy), SQ)
    got = np.stack([cx, cy], axis=1) + t.pull_off[0].double().numpy()
    out = ~inside.numpy()
    assert np.abs(got[out] - proj.numpy()[out]).max() < 1e-2


def test_table_shapes_dtypes_and_memory_budget():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=512, device="cpu")
    for name, tensor, dtype, shape in (
            ("pull_off", t.pull_off, torch.float16, (2, 512 * 512, 2)),
            ("pull_on", t.pull_on, torch.bool, (2, 512 * 512)),
            ("push_off", t.push_off, torch.float16, (2, 512 * 512, 2)),
            ("push_cnt", t.push_cnt, torch.uint8, (2, 512 * 512))):
        assert tensor.dtype is dtype, name
        assert tuple(tensor.shape) == shape, name
    # spec section 2's budget: K x 512^2 x 2 fp16 == 16 MiB at K=16.
    per_region = t.pull_off.element_size() * t.pull_off.nelement() // t.k
    assert per_region * 16 == 16 * 1024 * 1024


def test_anchor_tables_is_deterministic():
    a = hull.anchor_tables([SQ, BIG], DIE, lattice=16, device="cpu")
    b = hull.anchor_tables([SQ, BIG], DIE, lattice=16, device="cpu")
    assert torch.equal(a.pull_off, b.pull_off)
    assert torch.equal(a.push_off, b.push_off)
    assert torch.equal(a.push_cnt, b.push_cnt)
