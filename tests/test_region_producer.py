import numpy as np
import pytest
from scipy import ndimage
from ioplace.producer import sa


def _l_map():
    """4x4: region 1 is the 2x2 block at rows 0-1, cols 2-3; region 0 is the
    L-shaped remainder."""
    lab = np.zeros((4, 4), dtype=np.int16)
    lab[0:2, 2:4] = 1
    return lab


def _notch_map():
    lab = np.zeros((4, 4), dtype=np.int16)
    lab[1, 1] = 1
    return lab


def test_e_area_matches_eq5_closed_form():
    lab = np.zeros((10, 20), dtype=np.int16)
    lab.reshape(-1)[:80] = 0
    lab.reshape(-1)[80:] = 1            # counts 80 and 120, bin_area 1
    # d_0 = (0.95*100 - 80)/100 = 0.15 -> 50*0.15^3 + 25*0.15^2 = 0.73125
    # d_1 = (0.95*100 - 120)/100 < 0    -> 0
    got = sa.e_area(lab, 2, ea=np.array([100.0, 100.0]), bin_area=1.0)
    assert got == pytest.approx(0.73125, rel=1e-12)


def test_e_area_is_zero_above_the_slack_threshold():
    lab = np.zeros((10, 10), dtype=np.int16)
    lab[5:] = 1
    assert sa.e_area(lab, 2, ea=np.array([50.0, 50.0]), bin_area=1.0) == 0.0


def test_corner_counts_straight_boundary_is_zero():
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    assert int(sa.corner_counts(lab, 2)[0, 1]) == 0


def test_corner_counts_l_turn_is_one():
    assert int(sa.corner_counts(_l_map(), 2)[0, 1]) == 1


def test_corner_counts_single_bin_notch_is_four():
    assert int(sa.corner_counts(_notch_map(), 2)[0, 1]) == 4


def test_corner_counts_is_symmetric_with_a_zero_diagonal():
    c = sa.corner_counts(_l_map(), 2)
    assert np.array_equal(c, c.T)
    assert int(np.trace(c)) == 0


def test_e_boundary_matches_eq6_and_penalises_straight_boundaries_too():
    """digest section 4.2: Eq.6 is not one-sided, so C_ij = 0 costs 0.1*(0-2)^2."""
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    assert sa.e_boundary(lab, 2) == pytest.approx(0.4, rel=1e-12)
    assert sa.e_boundary(_l_map(), 2) == pytest.approx(0.1 * (1 - 2) ** 2, rel=1e-12)


def test_e_boundary_skips_non_adjacent_pairs():
    lab = np.zeros((6, 6), dtype=np.int16)
    lab[:, 2:4] = 1
    lab[:, 4:] = 2                       # 0-1 and 1-2 adjacent, 0-2 not
    assert sa.shared_edges(lab, 3)[0, 2] == 0
    assert sa.e_boundary(lab, 3) == pytest.approx(0.8, rel=1e-12)


def test_e_compact_matches_eq7_closed_form():
    # region 0: 12 bins in a 4x4 bbox -> rho 0.75 -> 5*((0.8-0.75)/0.8)^2
    # region 1: 4 bins in a 2x2 bbox  -> rho 1.0  -> 0
    assert sa.e_compact(_l_map(), 2) == pytest.approx(5.0 * (0.05 / 0.8) ** 2,
                                                      rel=1e-12)


def test_e_diff_is_the_changed_fraction():
    a = _l_map()
    b = a.copy()
    b[3, 0] = 1
    b[3, 1] = 1
    assert sa.e_diff(b, a) == pytest.approx(2.0 / 16.0, rel=1e-12)


def test_calibration_sets_minmax_and_a_positive_t0():
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 8:] = 1
    cfg = sa.SaConfig(probe_moves=50, seed=1)
    st = sa.SaState(lab, 2, ea=np.array([160.0, 96.0]), bin_area=1.0, cfg=cfg)
    lo, hi, t0 = st.calibrate(np.random.default_rng(cfg.seed))
    assert lo.shape == (4,) and hi.shape == (4,)
    assert (hi >= lo).all()
    assert t0 > 0.0
    assert np.array_equal(st.labels, lab), "calibration must leave the map intact"


def test_a_fragmenting_move_is_rejected():
    """Handing row 2 of region 0's 5x2 block to region 1 splits region 0 into
    two lobes, so try_move must refuse it and roll back."""
    lab = np.zeros((5, 5), dtype=np.int16)
    lab[:, 2:] = 1                       # region 0 is a solid 5x2 block, region 1 5x3
    st = sa.SaState(lab, 2, ea=np.array([10.0, 15.0]), bin_area=1.0,
                    cfg=sa.SaConfig())
    waist = np.array([2 * 5 + 0, 2 * 5 + 1], dtype=np.int64)
    assert st.try_move((waist, 1)) is None
    assert np.array_equal(st.labels, lab), "a rejected move must be rolled back"


def test_anneal_is_deterministic_under_a_fixed_seed():
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 8:] = 1
    ea = np.array([128.0, 128.0])
    cfg = sa.SaConfig(probe_moves=40, levels=10, moves_per_level=10, seed=5)
    a, ra = sa.anneal(lab, 2, ea, 1.0, cfg)
    b, rb = sa.anneal(lab, 2, ea, 1.0, cfg)
    assert np.array_equal(a, b)
    assert ra["e_raw_final"] == rb["e_raw_final"]


def test_anneal_keeps_every_region_non_empty_and_connected():
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:8, :8] = 0
    lab[:8, 8:] = 1
    lab[8:, :8] = 2
    lab[8:, 8:] = 3
    ea = np.array([80.0, 60.0, 60.0, 56.0])
    cfg = sa.SaConfig(probe_moves=60, levels=20, moves_per_level=20, seed=11)
    out, rep = sa.anneal(lab, 4, ea, 1.0, cfg)
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(4):
        m = out == k
        assert m.any(), f"region {k} vanished"
        assert ndimage.label(m, structure=st)[1] == 1, f"region {k} fragmented"
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3}
    assert rep["levels_run"] <= cfg.levels
    assert rep["t0"] > 0.0
    assert len(rep["e_raw_final"]) == 4 and len(rep["beta"]) == 4


def test_anneal_reduces_the_area_imbalance_it_is_given():
    """Region 1 starts far under its target area; area-balancing moves exist,
    so the final E_area must not be worse than the initial one.

    This monotonicity is not a general SA guarantee -- the annealer minimises
    the beta-weighted total, and a move can legitimately trade a little
    E_area for a lot of E_boundary/E_compact. It holds for this fixture
    because the area deficit dominates the beta-weighted total (initial
    E_area 29.4 versus the worst-case final E_area of 5.43 measured across
    seeds 0-499, 0/500 failures at fix-round-1 adjudication). Keep the
    assertion as written; do not weaken it on the strength of this comment
    alone if a future seed disagrees.
    """
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 14:] = 1
    ea = np.array([128.0, 128.0])
    cfg = sa.SaConfig(probe_moves=60, levels=40, moves_per_level=30, seed=3)
    out, rep = sa.anneal(lab, 2, ea, 1.0, cfg)
    assert rep["e_raw_final"][0] <= rep["e_raw_initial"][0] + 1e-9
    assert int((out == 1).sum()) > int((lab == 1).sum())


def test_lo_offset_cancels_at_fixed_span(monkeypatch):
    """Fix-round 2: the property to pin is offset-invariance at FIXED span --
    total() is affine per term, so adding the same constant c_j to both lo_j
    and hi_j leaves span_j = hi_j - lo_j unchanged and shifts only the
    offset, which must cancel exactly in every Delta E.

    (Round 1's version zeroed lo alone on a balanced fixture whose calibrated
    lo was already all zeros -- vacuous, and it also changed span_j, which the
    re-reviewer showed produces genuinely different runs.) This fixture has a
    real area deficit (region 1 far under its target area), so lo[0] != 0
    after calibration; that is asserted below so this test cannot silently go
    vacuous again."""
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 14:] = 1
    ea = np.array([128.0, 128.0])
    cfg = sa.SaConfig(probe_moves=60, levels=40, moves_per_level=30, seed=3)
    out_a, rep_a = sa.anneal(lab, 2, ea, 1.0, cfg)

    lo_a = np.asarray(rep_a["norm_lo"])
    span_a = np.asarray(rep_a["norm_span"])
    assert lo_a[0] != 0.0, "fixture must calibrate to a nonzero lo, or this test is vacuous"

    orig_calibrate = sa.SaState.calibrate
    # A different constant per term, small enough that hi_j + c_j - (lo_j + c_j)
    # stays within float64 rounding of hi_j - lo_j (large offsets like 1e3
    # against spans of O(1) lose bits to cancellation -- that is a float64
    # artefact of the test's own arithmetic, not a property of total()).
    offset = np.array([10.0, -10.0, 10.0, 10.0])

    def _offset_calibrate(self, rng):
        lo, hi, t0 = orig_calibrate(self, rng)
        self.lo = self.lo + offset
        self.hi = self.hi + offset
        return self.lo, self.hi, t0

    monkeypatch.setattr(sa.SaState, "calibrate", _offset_calibrate)
    out_b, rep_b = sa.anneal(lab, 2, ea, 1.0, cfg)

    span_b = np.asarray(rep_b["norm_span"])
    assert np.allclose(span_a, span_b, atol=1e-9), "the perturbation must leave span unchanged"
    assert not np.array_equal(np.asarray(rep_b["norm_lo"]), lo_a), \
        "the perturbation must actually move lo, or this test is vacuous"
    assert np.array_equal(out_a, out_b)
    assert rep_a["e_raw_final"] == rep_b["e_raw_final"]
    assert rep_a["levels_run"] == rep_b["levels_run"]
