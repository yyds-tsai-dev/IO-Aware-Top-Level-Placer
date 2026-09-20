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


def test_pull_off_fp16_error_is_bounded_on_a_realistic_die_size():
    # Fix round 1: the earlier "negligible against a bin width" docstring
    # claim was never exercised by the 8-unit toy DIE above. Use a real
    # (scaled) die size -- mempool_tile_wrap is 4737.0 x 4737.2 -- and a
    # small hull in the origin corner, so most bins are far outside it and
    # exercise the largest offsets the fp16 store has to carry.
    big_die = (0.0, 0.0, 4737.0, 4737.2)
    sq = np.array([[0., 0.], [94.7, 0.], [94.7, 94.7], [0., 94.7]])
    t = hull.anchor_tables([sq], big_die, lattice=512, device="cpu")
    assert torch.isfinite(t.pull_off).all()
    assert torch.isfinite(t.push_off).all()

    L = 512
    idx = np.arange(L, dtype=np.float64)
    ix, iy = np.meshgrid(idx, idx, indexing="xy")
    cx = (0.0 + (ix.ravel() + 0.5) * (4737.0 / L))
    cy = (0.0 + (iy.ravel() + 0.5) * (4737.2 / L))
    proj, inside = hull.nearest_on_polygon_boundary(
        torch.as_tensor(cx), torch.as_tensor(cy), sq)
    off = proj - torch.stack([torch.as_tensor(cx), torch.as_tensor(cy)], dim=1)
    ref = torch.where(inside.unsqueeze(1), torch.zeros_like(off), off)

    # P-C Task 10 fix round 1 (I3): the tables now store offsets in BIN
    # WIDTHS, so the reference has to be normalised the same way. The bound is
    # the SAME 0.25 bin the earlier adjudication established -- it is one
    # statement in two unit systems -- so it is asserted both normalised and
    # denormalised here.
    bin_w, bin_h = 4737.0 / 512, 4737.2 / 512
    bin_size = torch.tensor([bin_w, bin_h], dtype=torch.float64)
    ref_norm = ref / bin_size
    assert (t.pull_off.double() - ref_norm).abs().max() <= 0.25
    assert ((t.pull_off.double() - ref_norm) * bin_size).abs().max() \
        <= 0.25 * max(bin_w, bin_h)
    # and the normalised magnitudes really are bounded by the lattice, which is
    # what makes the fp16 range independent of die size
    assert t.pull_off.double().abs().max() <= 512.0


def test_a_die_far_past_the_old_fp16_ceiling_is_now_accepted():
    """P-C Task 10 fix round 1 (I3). Storing the offsets in scaled units made
    the fp16 range a hard ceiling on die size: this die -- roughly a 30M-cell
    NanGate45 chip, ~15.4x mempool_tile_wrap's 4168 scaled units linearly --
    used to raise ValueError inside the first hull rebuild. In bin widths the
    largest magnitude is the lattice, whatever the die."""
    huge_die = (0.0, 0.0, 64000.0, 64000.0)
    sq = np.array([[0., 0.], [1000., 0.], [1000., 1000.], [0., 1000.]])
    t = hull.anchor_tables([sq], huge_die, lattice=64, device="cpu")
    assert torch.isfinite(t.pull_off).all() and torch.isfinite(t.push_off).all()
    assert t.pull_off.double().abs().max() <= 64.0        # <= the lattice


def test_anchor_tables_rejects_a_lattice_that_would_overflow_fp16():
    """The guard survives the I3 change, but it now bounds the LATTICE-
    normalised magnitude rather than the die extent (hull.FP16_OFFSET_LIMIT)."""
    with pytest.raises(ValueError, match="lattice"):
        hull.check_anchor_table_range(DIE, int(hull.FP16_OFFSET_LIMIT))
    with pytest.raises(ValueError, match="lattice"):
        hull.check_anchor_table_range(DIE, 0)
    hull.check_anchor_table_range(DIE, 512)               # the production value


def test_anchor_tables_rejects_a_degenerate_die():
    """New precondition introduced by I3: the offsets are divided by the bin
    size, so a zero-extent axis would store nan rather than a wrong number."""
    with pytest.raises(ValueError, match="positive"):
        hull.anchor_tables([BIG], (0.0, 0.0, 0.0, 8.0), lattice=8, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        hull.anchor_tables([BIG], (0.0, 0.0, 8.0, 0.0), lattice=8, device="cpu")
