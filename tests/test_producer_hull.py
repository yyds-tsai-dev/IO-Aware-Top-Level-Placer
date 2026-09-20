import numpy as np
import pytest
from ioplace.producer import hull


def test_reduce_candidates_keeps_the_extreme_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, size=(5000, 2))
    pts = np.vstack([pts, [[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]]])
    out = hull.reduce_candidates(pts)
    for corner in ([-5., -5.], [5., -5.], [5., 5.], [-5., 5.]):
        assert (np.abs(out - corner).sum(axis=1) < 1e-12).any(), corner
    # Algorithm 1's own bound, plus ruling D1's one support point per band.
    assert len(out) <= 2 * hull.DIRECTIONS_M * (hull.K_DIR + 1)


def test_reduce_candidates_is_deterministic_and_passes_small_sets_through():
    pts = np.array([[0., 0.], [1., 0.], [0., 1.]])
    assert np.array_equal(hull.reduce_candidates(pts),
                          hull.reduce_candidates(pts))
    assert len(hull.reduce_candidates(pts)) == 3

    # The above input is <= K_DIR and only exercises the pass-through branch
    # (reduce_candidates:31-32). Exercise the actual m-direction reduction
    # loop with an input large enough to skip that branch, and check
    # determinism there too.
    rng = np.random.default_rng(3)
    big = rng.uniform(0.0, 1.0, size=(5000, 2))
    assert len(big) > hull.K_DIR
    out1 = hull.reduce_candidates(big)
    out2 = hull.reduce_candidates(big)
    assert np.array_equal(out1, out2)


def test_convex_hull_of_a_square_cloud_is_the_square_ccw():
    rng = np.random.default_rng(1)
    inner = rng.uniform(0.1, 0.9, size=(200, 2))
    pts = np.vstack([inner, [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    v = hull.convex_hull(pts)
    assert len(v) == 4
    assert hull.polygon_area(v) == pytest.approx(1.0, rel=1e-12)
    # counter-clockwise => positive signed area
    sx, sy = v[:, 0], v[:, 1]
    signed = 0.5 * (np.dot(sx, np.roll(sy, -1)) - np.dot(sy, np.roll(sx, -1)))
    assert signed > 0.0


def test_convex_hull_degenerates_to_a_box_for_collinear_input():
    v = hull.convex_hull(np.array([[0., 0.], [1., 0.], [2., 0.]]))
    assert len(v) >= 3
    assert hull.polygon_area(v) > 0.0
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(2.0)


def _assert_valid_ccw_hull(v):
    assert len(v) >= 3
    assert hull.polygon_area(v) > 0.0
    sx, sy = v[:, 0], v[:, 1]
    signed = 0.5 * (np.dot(sx, np.roll(sy, -1)) - np.dot(sy, np.roll(sx, -1)))
    assert signed > 0.0


def test_convex_hull_fallback_for_a_single_point():
    v = hull.convex_hull(np.array([[3.0, 4.0]]))
    _assert_valid_ccw_hull(v)


def test_convex_hull_fallback_for_a_single_point_far_from_the_origin():
    # eps must scale with coordinate magnitude, not a bare absolute 1e-9,
    # or the inflated box underflows to ~zero area.
    v = hull.convex_hull(np.array([[1e6, 1e6]]))
    _assert_valid_ccw_hull(v)


def test_convex_hull_fallback_for_two_points():
    v = hull.convex_hull(np.array([[0.0, 0.0], [1.0, 1.0]]))
    _assert_valid_ccw_hull(v)


def test_shrink_to_area_matches_the_closed_form_within_the_tolerance():
    """Uniform scaling about a fixed point scales area exactly by s**2, so
    s* = sqrt(a_max/a) is the exact answer; the spec mandates bisection to
    1e-3 relative area, and this pins the bisection against that oracle."""
    v = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])
    a = hull.polygon_area(v)
    a_max = 0.25 * a
    out = hull.shrink_to_area(v, a_max)
    got = hull.polygon_area(out)
    assert got <= a_max + 1e-12
    assert got == pytest.approx(a_max, rel=1e-3)
    # centroid preserved
    assert out.mean(axis=0) == pytest.approx(v.mean(axis=0))


def test_shrink_to_area_is_a_noop_below_the_cap():
    v = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert np.array_equal(hull.shrink_to_area(v, 10.0), v)


def test_shrink_to_area_rejects_non_positive_a_max():
    v = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    with pytest.raises(ValueError):
        hull.shrink_to_area(v, 0.0)
    with pytest.raises(ValueError):
        hull.shrink_to_area(v, -1.0)


def test_macro_pseudo_points_respect_the_cap_and_stay_inside():
    x = np.array([0.0, 100.0])
    y = np.array([0.0, 100.0])
    w = np.array([50.0, 3.0])
    h = np.array([50.0, 3.0])
    p = hull.macro_pseudo_points(x, y, w, h, pitch_x=1.0, pitch_y=1.0)
    # 50x50 at pitch 1 would be 2500 points; the cap forces 8x8.
    assert len(p) == 64 + 9
    first = p[:64]
    assert (first[:, 0] > 0.0).all() and (first[:, 0] < 50.0).all()
    assert (first[:, 1] > 0.0).all() and (first[:, 1] < 50.0).all()


def test_macro_pseudo_points_handles_no_macros():
    e = np.zeros(0)
    assert hull.macro_pseudo_points(e, e, e, e, 1.0, 1.0).shape == (0, 2)


def test_macro_pseudo_points_rejects_non_positive_pitch():
    x = np.array([0.0])
    y = np.array([0.0])
    w = np.array([10.0])
    h = np.array([10.0])
    with pytest.raises(ValueError):
        hull.macro_pseudo_points(x, y, w, h, pitch_x=0.0, pitch_y=1.0)
    with pytest.raises(ValueError):
        hull.macro_pseudo_points(x, y, w, h, pitch_x=1.0, pitch_y=-1.0)


def test_build_hull_applies_reduction_then_cap():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.0, 10.0, size=(20000, 2))
    v = hull.build_hull(pts, a_max=25.0)
    area = hull.polygon_area(v)
    assert area <= 25.0 + 1e-9
    # Ruling D12: the upper bound alone is satisfied by a hull that collapsed
    # onto the dense cluster, which is exactly how D1 slipped through. The cap
    # binds here (the uncapped hull is ~100), so the shrink must land ON it.
    assert area >= 0.9 * 25.0
    assert len(v) >= 3


import torch


@pytest.mark.gpu
def test_reduce_candidates_torch_matches_the_numpy_path():
    """Continuous random data, so exact ties (where torch.topk and numpy's
    stable argsort could disagree) have measure zero."""
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(50_000, 2)) * np.array([3.0, 1.0])
    want = hull.reduce_candidates(pts)
    x = torch.as_tensor(pts[:, 0], device="cuda", dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], device="cuda", dtype=torch.float64)
    got = hull.reduce_candidates_torch(x, y)
    assert got.shape[1] == 2
    assert {tuple(r) for r in got.tolist()} == {tuple(r) for r in want.tolist()}


@pytest.mark.gpu
def test_reduce_candidates_torch_passes_small_sets_through():
    x = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0], device="cuda", dtype=torch.float64)
    got = hull.reduce_candidates_torch(x, y)
    assert sorted(map(tuple, got.tolist())) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]


@pytest.mark.gpu
def test_reduce_candidates_torch_feeds_a_usable_hull():
    rng = np.random.default_rng(12)
    pts = rng.uniform(0.0, 10.0, size=(20_000, 2))
    x = torch.as_tensor(pts[:, 0], device="cuda", dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], device="cuda", dtype=torch.float64)
    v = hull.convex_hull(hull.reduce_candidates_torch(x, y))
    assert hull.polygon_area(v) == pytest.approx(100.0, rel=0.02)


@pytest.mark.gpu
def test_reduce_candidates_torch_matches_the_numpy_path_on_tied_grid_data():
    """Grid-aligned input, unlike the continuous cloud above: many points
    share the exact same projection for axis- and diagonal-aligned directions
    among the m=16, so the quantile band collapses to ss == t and the k_dir
    cutoff's tie-break is actually exercised. Both paths must land on the
    identical point set -- a discrete equality, never a float tolerance --
    which only holds if the GPU's stable sort breaks ties the same way as the
    CPU's np.argsort(kind="stable")."""
    xs, ys = np.meshgrid(np.arange(60, dtype=np.float64),
                         np.arange(60, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    want = hull.reduce_candidates(pts)
    x = torch.as_tensor(pts[:, 0], device="cuda", dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], device="cuda", dtype=torch.float64)
    got = hull.reduce_candidates_torch(x, y)
    assert np.array_equal(got, want)


@pytest.mark.gpu
def test_reduce_candidates_torch_stride_subsamples_the_quantile_above_the_threshold():
    """quantile_subsample makes the "n > QUANTILE_SUBSAMPLE" stride-subsample
    branch reachable without allocating 8e6 points -- that branch is this
    task's own motivation (torch.quantile's input-element limit) and would
    otherwise go untested. With the threshold forced far below n, every
    quantile is estimated from a strided subsample rather than the full
    column; the documented invariants that estimate must still keep are
    determinism (same input, same result) and ruling D1 (every direction's
    true extreme point over the FULL data survives even though the threshold
    itself is only an estimate)."""
    rng = np.random.default_rng(21)
    pts = rng.uniform(0.0, 10.0, size=(5_000, 2))
    x = torch.as_tensor(pts[:, 0], device="cuda", dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], device="cuda", dtype=torch.float64)
    got1 = hull.reduce_candidates_torch(x, y, quantile_subsample=100)
    got2 = hull.reduce_candidates_torch(x, y, quantile_subsample=100)
    assert np.array_equal(got1, got2)
    for j in range(hull.DIRECTIONS_M):
        th = j * (2.0 * np.pi / hull.DIRECTIONS_M)
        s = pts[:, 0] * np.cos(th) + pts[:, 1] * np.sin(th)
        for sign in (1.0, -1.0):
            extreme = pts[np.argmax(sign * s)]
            assert (np.abs(got1 - extreme).sum(axis=1) < 1e-9).any()
