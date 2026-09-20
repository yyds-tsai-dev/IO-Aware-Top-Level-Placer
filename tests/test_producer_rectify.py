import numpy as np
import pytest
from scipy import ndimage
from ioplace.producer import rectify

DIE = (0.0, 0.0, 1024.0, 1024.0)


def _comb(n=16):
    """Region 1 = a spine row plus 8 teeth (9 maximal-strip rectangles);
    region 0 = everything else."""
    lab = np.zeros((n, n), dtype=np.int16)
    lab[0, :] = 1
    for c in range(0, n, 2):
        lab[1:3, c] = 1
    return lab


def test_mask_to_rects_on_a_solid_rectangle():
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:4] = True
    r = rectify.mask_to_rects(m)
    assert r.tolist() == [[1, 1, 4, 3]]


def test_mask_to_rects_on_an_l_shape_gives_two_rects():
    m = np.zeros((4, 4), dtype=bool)
    m[0:2, 0:4] = True
    m[2:4, 0:2] = True
    r = rectify.mask_to_rects(m)
    assert len(r) == 2
    assert sorted(r.tolist()) == [[0, 0, 4, 2], [0, 2, 2, 4]]


def test_mask_to_rects_reconstructs_the_mask_exactly():
    """Area alone can hide an overlap and a gap of equal size cancelling out
    in the sum; rasterise the returned rects back into a grid and require it
    to be pixel-identical to the input, not just equal in total area."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((12, 12)) < 0.4
        r = rectify.mask_to_rects(m)
        recon = np.zeros_like(m)
        for x0, y0, x1, y1 in r.tolist():
            assert not recon[y0:y1, x0:x1].any(), "rects overlap"
            recon[y0:y1, x0:x1] = True
        assert np.array_equal(recon, m)


def test_region_rect_counts_sees_the_comb():
    counts = rectify.region_rect_counts(_comb(), 2)
    assert counts[1] == 9


def test_enforce_rect_max_reduces_the_comb_and_keeps_a_legal_tiling():
    lab = _comb()
    out = rectify.enforce_rect_max(lab, 2, rect_max=8)
    counts = rectify.region_rect_counts(out, 2)
    assert max(counts) <= 8
    assert out.shape == lab.shape
    assert set(np.unique(out).tolist()) == {0, 1}
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(2):
        m = out == k
        assert m.any()
        assert ndimage.label(m, structure=st)[1] == 1


def test_enforce_rect_max_is_a_noop_when_already_within_budget():
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    out = rectify.enforce_rect_max(lab, 2, rect_max=8)
    assert np.array_equal(out, lab)


def test_enforce_rect_max_is_deterministic():
    lab = _comb()
    a = rectify.enforce_rect_max(lab, 2, rect_max=8)
    b = rectify.enforce_rect_max(lab, 2, rect_max=8)
    assert np.array_equal(a, b)


def test_shed_one_strip_gives_the_smallest_strip_to_the_majority_neighbour():
    """Ruling D3's second move. Region 1 is the right block plus a one-row
    overhang; the overhang is the smaller maximal strip, so it is the one that
    goes, and it goes to region 0 (its only foreign 4-neighbour)."""
    lab = np.zeros((6, 6), dtype=np.int16)
    lab[:, 3:] = 1
    lab[5, 2] = 1
    assert rectify.region_rect_counts(lab, 2)[1] == 2
    out = lab.copy()
    assert rectify._shed_one_strip(out, 1) is True
    assert rectify.region_rect_counts(out, 2)[1] == 1
    assert (out[5, 2:6] == 0).all()
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(2):
        m = out == k
        assert m.any() and ndimage.label(m, structure=st)[1] == 1


def test_repeated_shedding_drops_exactly_one_rect_a_time_and_terminates():
    """The construction argument in the task prose: mask_to_rects' rects are
    whole row runs, so removing one removes exactly those runs and the count
    falls by exactly one. Both regions must stay non-empty and 4-connected at
    every step, and the loop must stop on its own."""
    lab = _comb()
    st = ndimage.generate_binary_structure(2, 1)
    start = rectify.region_rect_counts(lab, 2)[1]
    previous = start
    for _ in range(30):
        if not rectify._shed_one_strip(lab, 1):
            break
        count = rectify.region_rect_counts(lab, 2)[1]
        assert count == previous - 1
        previous = count
        for k in range(2):
            m = lab == k
            assert m.any()
            assert ndimage.label(m, structure=st)[1] == 1
    assert previous < start


def test_enforce_rect_max_sheds_when_absorption_would_empty_the_donor():
    """Integration test through the public enforce_rect_max entry point (not
    the private helpers), covering ruling D3's absorption-then-shedding
    composition end to end. Region 1 is a frame -- top/bottom bands plus a
    left and a right margin column -- that spans the entire grid, so its
    bbox IS the grid and region 0 (a left blob, a one-row neck, and a right
    blob, all one 4-connected piece) is its only bbox-hole component.
    Absorbing that single hole would consume every region-0 cell, which
    _fill_one_notch's feasibility test correctly refuses (a donor must stay
    non-empty); enforce_rect_max must then fall through to
    _shed_one_strip, which gives away region 1's smallest strip -- the
    left margin column, redundant because the top/bottom bands already stay
    connected through the right margin alone -- to region 0."""
    lab = np.ones((9, 17), dtype=np.int16)
    lab[2:7, 1:6] = 0      # left blob
    lab[2:7, 11:16] = 0    # right blob
    lab[4, 6:11] = 0       # one-row neck joining the two blobs
    assert rectify.region_rect_counts(lab, 2) == [5, 6]
    # Absorption is infeasible by construction: confirmed directly before
    # relying on it, so the enforce_rect_max assertions below are known to
    # exercise the shedding branch and not a lucky absorption.
    assert rectify._fill_one_notch(lab.copy(), 1) is False

    out = rectify.enforce_rect_max(lab, 2, rect_max=5)
    counts = rectify.region_rect_counts(out, 2)
    assert max(counts) <= 5

    expected = lab.copy()
    expected[2:7, 0] = 0    # the shed left-margin column, given to region 0
    assert np.array_equal(out, expected)

    st = ndimage.generate_binary_structure(2, 1)
    for k in range(2):
        m = out == k
        assert m.any()
        assert ndimage.label(m, structure=st)[1] == 1


def test_enforce_rect_max_raises_when_neither_move_is_feasible():
    """A solid two-region tiling -- each region already exactly fills its own
    bounding box -- with an impossible rect_max=0 budget: _fill_one_notch has
    no notch to fill (mask already equals bbox) and _shed_one_strip has
    nothing to shed (already a single rect), so enforce_rect_max must raise
    the in-loop RuntimeError (ruling D3: an abort here is legitimate
    behaviour, not a bug) naming the offending region and its rect count."""
    lab = np.zeros((4, 8), dtype=np.int16)
    lab[:, 4:] = 1
    with pytest.raises(RuntimeError, match=r"region 0 has 1 rects \(> 0\)"):
        rectify.enforce_rect_max(lab, 2, rect_max=0)


def test_enforce_rect_max_raises_on_budget_exhaustion_when_the_target_cycles():
    """Ruling D4: there is no monotone potential, so the arg-max target can
    legitimately ping-pong between regions forever. This 3x4 tiling is
    verified to cycle under an impossible rect_max=1: region 0's top row and
    region 1's 2x2 block trade a single row back and forth via shedding, so
    enforce_rect_max must exhaust the k*B^2 pass budget and raise -- naming
    the region and its rect count on the final pass -- rather than loop
    forever."""
    lab = np.array([
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ], dtype=np.int16)
    budget = 2 * lab.size + 1
    with pytest.raises(RuntimeError,
                        match=rf"did not converge within {budget} passes"):
        rectify.enforce_rect_max(lab, 2, rect_max=1)


@pytest.mark.parametrize("bins", [32, 64])
def test_rects_to_regionset_validates_and_snaps_to_the_512_lattice(bins):
    lab = np.zeros((bins, bins), dtype=np.int16)
    lab[:, bins // 2:] = 1
    lab[bins // 2:, :bins // 4] = 2
    rs = rectify.rects_to_regionset(lab, 3, DIE, lattice=512)
    rs.validate()                                   # must not raise
    assert rs.k == 3 and rs.lattice == 512
    pitch = (DIE[2] - DIE[0]) / 512
    for r in rs.regions:
        q = np.asarray(r.rects) / pitch
        assert np.allclose(q, np.round(q), atol=1e-9)
    total = sum(((np.asarray(r.rects)[:, 2] - np.asarray(r.rects)[:, 0]) *
                 (np.asarray(r.rects)[:, 3] - np.asarray(r.rects)[:, 1])).sum()
                for r in rs.regions)
    assert total == pytest.approx(1024.0 * 1024.0)


def test_rects_to_regionset_rejects_a_bin_count_that_does_not_divide_the_lattice():
    lab = np.zeros((10, 10), dtype=np.int16)
    with pytest.raises(AssertionError):
        rectify.rects_to_regionset(lab, 1, DIE, lattice=512)


def test_rects_to_regionset_rejects_an_empty_region():
    lab = np.zeros((32, 32), dtype=np.int16)
    with pytest.raises(AssertionError):
        rectify.rects_to_regionset(lab, 2, DIE, lattice=512)
