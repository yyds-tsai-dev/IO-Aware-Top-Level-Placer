import numpy as np
import pytest
from scipy import ndimage
from ioplace.producer import extract

DIE = (0.0, 0.0, 16.0, 16.0)


def _two_l_shapes(n=16):
    """A 16x16 label grid: region 1 is the top-right quadrant, region 0 is the
    L-shaped remainder. `labels[y, x]`."""
    lab = np.zeros((n, n), dtype=np.int16)
    lab[n // 2:, n // 2:] = 1
    return lab


def _cells_from_labels(lab, die=DIE):
    """One unit-area cell at the centre of every bin, labelled by that bin."""
    n = lab.shape[0]
    xl, yl, xh, yh = die
    cw, ch = (xh - xl) / n, (yh - yl) / n
    iy, ix = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = xl + ix.ravel() * cw
    y = yl + iy.ravel() * ch
    w = np.full(x.size, cw)
    h = np.full(x.size, ch)
    return x, y, w, h, lab.ravel().astype(np.int32)


def test_density_maps_and_argmax_recover_the_input_labels():
    lab = _two_l_shapes()
    x, y, w, h, part = _cells_from_labels(lab)
    dens = extract.density_maps(x, y, w, h, part, 2, DIE, bins=16)
    assert dens.shape == (2, 16, 16)
    assert dens.sum() == pytest.approx(16.0 * 16.0)
    assert np.array_equal(extract.argmax_labels(dens), lab)


def test_argmax_marks_empty_bins():
    dens = np.zeros((2, 4, 4))
    dens[0, 0, 0] = 1.0
    lab = extract.argmax_labels(dens)
    assert lab[0, 0] == 0
    assert (lab[1:] == -1).all()


def test_majority_downsample_2x2_blocks():
    fine = np.array([[0, 0, 1, 1],
                     [0, 1, 1, 1],
                     [-1, -1, 0, 0],
                     [-1, 2, 0, 0]], dtype=np.int16)
    out = extract.majority_downsample(fine, 2, 3)
    # block (0,0): {0,0,0,1} -> 0 ; block (0,1): {1,1,1,1} -> 1
    # block (1,0): {-1,-1,-1,2} -> 2 (empty bins do not vote) ; block (1,1): 0
    assert out.tolist() == [[0, 1], [2, 0]]


def test_majority_downsample_marks_a_fully_empty_block():
    fine = -np.ones((4, 4), dtype=np.int16)
    fine[0, 0] = 1
    assert extract.majority_downsample(fine, 2, 2).tolist() == [[1, -1], [-1, -1]]


def test_opening_does_not_erode_a_solid_rectangle():
    """The 3x3 square structuring element is chosen precisely for this: the
    4-connected cross would strip all four corners of every rectangle."""
    lab = _two_l_shapes(8)
    assert np.array_equal(extract.morph_open_close(lab, 2), lab)


def test_opening_removes_an_isolated_speck():
    lab = _two_l_shapes(8)
    lab[1, 1] = 1                       # one stray bin of region 1 inside region 0
    out = extract.morph_open_close(lab, 2)
    assert out[1, 1] == -1              # opened away, now unclaimed whitespace


def test_morphology_never_deletes_a_whole_partition():
    """Opening removes an isolated speck, but a partition whose only bins were
    specks must not vanish outright -- it is restored from the pre-morphology
    map wherever the post-morphology map is unclaimed."""
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    lab[0, 0] = 2                       # region 2 is a single speck
    out = extract.morph_open_close(lab, 3)
    assert (out == 2).any()
    assert out[0, 0] == 2


def test_largest_component_drops_a_detached_island():
    lab = _two_l_shapes(16)
    lab[1:4, 1:4] = 1                   # a 3x3 island that survives opening
    out = extract.largest_component(lab, 2)
    assert (out[1:4, 1:4] == -1).all()
    assert (out[8:, 8:] == 1).all()


def test_ensure_nonempty_rescues_a_vanished_partition():
    lab = _two_l_shapes(8)              # region 2 has no bins at all
    coarse = np.zeros((3, 8, 8))
    coarse[2, 0, 5] = 9.0               # region 2's best bin, currently region 0's
    out = extract.ensure_nonempty(lab, coarse, 3)
    assert out[0, 5] == 2
    assert int((out == 2).sum()) == 1


def test_fill_whitespace_leaves_no_unlabelled_bin():
    lab = _two_l_shapes(8)
    lab[3:5, 3:5] = -1
    out = extract.fill_whitespace(lab)
    assert (out >= 0).all()
    assert out[0, 0] == 0 and out[7, 7] == 1


def test_fill_whitespace_rejects_a_fully_empty_map():
    with pytest.raises(ValueError):
        extract.fill_whitespace(-np.ones((4, 4), dtype=np.int16))


def test_extract_end_to_end_tiles_the_die_with_both_l_shapes():
    lab = _two_l_shapes(16)
    x, y, w, h, part = _cells_from_labels(lab)
    out = extract.extract(x, y, w, h, part, 2, DIE, out_bins=8, fine_bins=16)
    assert out.shape == (8, 8)
    assert (out >= 0).all()
    assert set(np.unique(out).tolist()) == {0, 1}
    assert (out[4:, 4:] == 1).all()
    assert (out[:4, :] == 0).all()


def test_canonicalise_connectivity_reattaches_a_stray_component():
    """Ruling D7: SA and rectification both require every region to be
    4-connected, and neither ensure_nonempty nor fill_whitespace guarantees it.
    Region 1's stray corner bin has no region-1 4-neighbour, so it must be
    handed to the majority label among its own 4-neighbours (region 0)."""
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    lab[0, 0] = 1
    out = extract.canonicalise_connectivity(lab, 2)
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(2):
        m = out == k
        assert m.any(), k
        assert ndimage.label(m, structure=st)[1] == 1, k
    assert out[0, 0] == 0


def test_extract_asserts_every_region_is_connected():
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    lab[0, 0] = 1                       # a disconnected speck of region 1
    lab[7, 0] = 1
    x, y, w, h, part = _cells_from_labels(lab, die=(0.0, 0.0, 8.0, 8.0))
    out = extract.extract(x, y, w, h, part, 2, (0.0, 0.0, 8.0, 8.0),
                          out_bins=8, fine_bins=8)
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(2):
        m = out == k
        assert m.any(), k
        assert ndimage.label(m, structure=st)[1] == 1, k


def test_extract_is_deterministic():
    lab = _two_l_shapes(16)
    x, y, w, h, part = _cells_from_labels(lab)
    a = extract.extract(x, y, w, h, part, 2, DIE, out_bins=8, fine_bins=16)
    b = extract.extract(x, y, w, h, part, 2, DIE, out_bins=8, fine_bins=16)
    assert np.array_equal(a, b)
