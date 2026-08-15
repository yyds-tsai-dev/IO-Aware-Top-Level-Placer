import os
import shutil

import numpy as np
import pytest

from ioplace.bench import glue_gen
from ioplace.bench import tile_bookshelf as tb
from ioplace.bench import verify_bench as vb
from tests.test_bench_tile_bookshelf import (
    write_toy_bookshelf, _TOY_NODES, _TOY_PL, _TOY_SCL, _TOY_NETS, _TOY_WTS, _TOY_AUX,
)


def _clean_2x2(tmp_path, name="ok"):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / name / "arr")
    manifest = tb.tile(src, dst, R=2, C=2, seed=0)
    return src, dst, manifest


def test_v0_passes_on_a_clean_array(tmp_path):
    src, dst, manifest = _clean_2x2(tmp_path)
    res = vb.check_v0_structural(dst, manifest=manifest, source_prefix=src)
    assert res["status"] == "ok", res["errors"]
    assert res["checks"]["aux_files_exist"] is True
    assert res["checks"]["nodes_header_matches_body"] is True
    assert res["checks"]["nets_header_matches_body"] is True
    assert res["checks"]["matches_manifest"] is True
    assert res["checks"]["self_loop_fraction_not_worse_than_source"] is True
    assert res["checks"]["self_loop_fraction"] == pytest.approx(0.0)
    assert res["checks"]["source_self_loop_fraction"] == pytest.approx(0.0)
    assert res["checks"]["all_nodes_referenced_or_declared_unconnected"] is True
    assert res["checks"]["unreferenced_count"] == 0
    assert res["checks"]["source_unreferenced_count"] == 0
    assert res["checks"]["rows_no_overlap_no_gap"] is True


def test_v0_catches_num_nodes_header_mismatch(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    with open(dst + ".nodes") as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumNodes"))
    lines[idx] = "NumNodes : 999999\n"
    with open(dst + ".nodes", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["nodes_header_matches_body"] is False


def test_v0_catches_num_nets_or_pins_header_mismatch(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    with open(dst + ".nets") as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumPins"))
    lines[idx] = "NumPins : 1\n"
    with open(dst + ".nets", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["nets_header_matches_body"] is False


def test_v0_catches_manifest_mismatch(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    bad_manifest = dict(manifest)
    bad_manifest["glue"] = {"n_nets": 5, "n_pins": 10, "kernel": "power_law"}  # claims glue that isn't there
    res = vb.check_v0_structural(dst, manifest=bad_manifest)
    assert res["status"] == "fail"
    assert res["checks"]["matches_manifest"] is False


def test_v0_self_loop_no_worse_than_source_passes(tmp_path):
    """2026-08-15 adjudication sec 4.2 point 1: a source that itself has
    self-loops (like real `mempool_group`, 4.951% of its own nets) must
    pass once tiled -- the tiler only replicates them, it doesn't add any
    -- even though the array's raw self-loop count is nonzero and > 0
    (the pre-2026-08-15 unconditional "must be exactly zero" bar would
    have misfired here exactly as it did on the real 2x2 array)."""
    src = str(tmp_path / "src" / "loopy")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src + ".nodes", "w") as f:
        f.write(_TOY_NODES)
    with open(src + ".pl", "w") as f:
        f.write(_TOY_PL)
    with open(src + ".scl", "w") as f:
        f.write(_TOY_SCL)
    with open(src + ".wts", "w") as f:
        f.write(_TOY_WTS)
    with open(src + ".aux", "w") as f:
        f.write(_TOY_AUX.replace("toy.", "loopy."))
    loopy_nets = _TOY_NETS.replace(
        "NumNets : 3\nNumPins : 7",
        "NumNets : 4\nNumPins : 9",
    ) + "NetDegree : 2 self_loop\n    o0 I : 0 0\n    o0 I : 0 0\n"
    with open(src + ".nets", "w") as f:
        f.write(loopy_nets)

    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=2, C=2, seed=0)
    res = vb.check_v0_structural(dst, manifest=manifest, source_prefix=src)
    assert res["status"] == "ok", res["errors"]
    assert res["checks"]["self_loop_fraction_not_worse_than_source"] is True
    assert res["checks"]["self_loop_fraction"] == pytest.approx(res["checks"]["source_self_loop_fraction"])


def test_v0_catches_self_loop_worse_than_source(tmp_path):
    src, dst, manifest = _clean_2x2(tmp_path)  # source has 0 self-loops
    with open(dst + ".nets") as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumNets"))
    n_nets = int(lines[idx].split(":")[1])
    idx_pins = next(i for i, l in enumerate(lines) if l.strip().startswith("NumPins"))
    n_pins = int(lines[idx_pins].split(":")[1])
    lines[idx] = f"NumNets : {n_nets + 1}\n"
    lines[idx_pins] = f"NumPins : {n_pins + 2}\n"
    lines.append("NetDegree : 2 t0_0/self_loop\n")
    lines.append("    t0_0/o0 I : 0 0\n")
    lines.append("    t0_0/o0 I : 0 0\n")
    with open(dst + ".nets", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst, source_prefix=src)
    assert res["status"] == "fail"
    assert res["checks"]["self_loop_fraction_not_worse_than_source"] is False
    assert res["checks"]["self_loop_fraction"] > res["checks"]["source_self_loop_fraction"]
    assert any("self-loop" in e for e in res["errors"])


def test_v0_no_source_or_manifest_falls_back_to_old_self_loop_free_bar(tmp_path):
    """Without a source to compare against, `check_v0_structural` has
    nothing to be "no worse than" -- no self-loop check is even computed
    (same framework-style fallback as `degree_le1_not_worse_than_source`),
    it is not silently treated as passing or failing."""
    _, dst, manifest = _clean_2x2(tmp_path)
    res = vb.check_v0_structural(dst)
    assert "self_loop_fraction_not_worse_than_source" not in res["checks"]
    assert res["checks"]["self_loop_fraction"] == pytest.approx(0.0)


def test_v0_unreferenced_count_matches_source_count_times_r_times_c(tmp_path):
    """2026-08-15 adjudication sec 4.2 point 3: the real bug this was
    written to catch -- 2x2's 8 unreferenced nodes = the source's own 2 x
    4 (R*C), not a tiler defect. A source with exactly 1 unreferenced node
    must tile to exactly 1*R*C = 4 in a 2x2 array."""
    src = str(tmp_path / "src" / "orphaned")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    orphaned_nodes = _TOY_NODES.replace("NumNodes : 7", "NumNodes : 8") + "orphan 1 1\n"
    with open(src + ".nodes", "w") as f:
        f.write(orphaned_nodes)
    with open(src + ".pl", "w") as f:
        f.write(_TOY_PL + "orphan 0 0 : N\n")
    with open(src + ".nets", "w") as f:
        f.write(_TOY_NETS)
    with open(src + ".scl", "w") as f:
        f.write(_TOY_SCL)
    with open(src + ".wts", "w") as f:
        f.write(_TOY_WTS)
    with open(src + ".aux", "w") as f:
        f.write(_TOY_AUX.replace("toy.", "orphaned."))

    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=2, C=2, seed=0)
    res = vb.check_v0_structural(dst, manifest=manifest, source_prefix=src)
    assert res["status"] == "ok", res["errors"]
    assert res["checks"]["source_unreferenced_count"] == 1
    assert res["checks"]["unreferenced_count"] == 4  # 1 * R(2) * C(2)
    assert res["checks"]["all_nodes_referenced_or_declared_unconnected"] is True

    # one *extra* unreferenced node beyond source_count*R*C must fail
    with open(dst + ".nodes") as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumNodes"))
    n_nodes = int(lines[idx].split(":")[1])
    lines[idx] = f"NumNodes : {n_nodes + 1}\n"
    lines.append("t0_0/extra_orphan 1 1\n")
    with open(dst + ".nodes", "w") as f:
        f.writelines(lines)
    res2 = vb.check_v0_structural(dst, manifest=manifest, source_prefix=src)
    assert res2["status"] == "fail"
    assert res2["checks"]["all_nodes_referenced_or_declared_unconnected"] is False
    assert res2["checks"]["unreferenced_count"] == 5


def test_v0_catches_unreferenced_node_without_source_uses_old_zero_bar(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    with open(dst + ".nodes") as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumNodes"))
    n_nodes = int(lines[idx].split(":")[1])
    lines[idx] = f"NumNodes : {n_nodes + 1}\n"
    lines.append("t0_0/orphan 1 1\n")
    with open(dst + ".nodes", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["all_nodes_referenced_or_declared_unconnected"] is False

    # explicitly declaring it unconnected clears that specific check
    res2 = vb.check_v0_structural(dst, allowed_unconnected={"t0_0/orphan"})
    assert res2["checks"]["all_nodes_referenced_or_declared_unconnected"] is True


def test_v0_allows_row_gap_under_a_macro(tmp_path):
    """2026-08-15 adjudication sec 4.2 point 2: a gap alone (no overlap)
    must no longer fail -- a macro's footprint legitimately has no
    placement row under it (`tile_bookshelf.assert_rows_no_overlap_no_gap`
    already made this call at construction time; `verify_bench` was the
    one place still out of sync with it).

    (The toy fixture's two rows exactly and tightly cover the tile height
    with no slack -- editing an *already-tiled* array's `.scl` to shift one
    row would just collide with the next tile's row instead of opening a
    clean gap, the same trap `tile_bookshelf.py`'s own
    `test_row_gap_is_allowed` sidesteps. So this builds a *source* with
    the gap already in it -- `tile()`'s tile-height `H` is measured from
    the source's own row bounding box, so the gap is carried through
    consistently, not collided into -- then tiles that, same as that
    test.)"""
    src = str(tmp_path / "src" / "gappy")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src + ".nodes", "w") as f:
        f.write(_TOY_NODES)
    with open(src + ".pl", "w") as f:
        f.write(_TOY_PL)
    with open(src + ".nets", "w") as f:
        f.write(_TOY_NETS)
    with open(src + ".wts", "w") as f:
        f.write(_TOY_WTS)
    with open(src + ".aux", "w") as f:
        f.write(_TOY_AUX.replace("toy.", "gappy."))
    # second row starts at y=3 instead of y=2 -> a 1-unit gap (e.g. a macro)
    gappy_scl = _TOY_SCL.replace("Coordinate : 2", "Coordinate : 3")
    with open(src + ".scl", "w") as f:
        f.write(gappy_scl)

    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=1, C=2, seed=0)  # must not raise (tiler already allows gaps)
    res = vb.check_v0_structural(dst, manifest=manifest, source_prefix=src)
    assert res["status"] == "ok", res["errors"]
    assert res["checks"]["rows_no_overlap_no_gap"] is True


def test_v0_still_catches_row_overlap(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    with open(dst + ".scl") as f:
        text = f.read()
    # bump exactly one row's Coordinate *down* by 1 -> a 1-unit overlap
    # with the row above it (which still ends at the same y as before)
    lines = text.splitlines(keepends=True)
    for i, l in enumerate(lines):
        if l.strip() == "Coordinate : 2":
            lines[i] = "\tCoordinate : 1\n"
            break
    with open(dst + ".scl", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["rows_no_overlap_no_gap"] is False
    assert any("overlap" in e for e in res["errors"])


def test_v0_catches_missing_aux_referenced_file(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    os.remove(dst + ".pl")
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["aux_files_exist"] is False


def test_v0_degree_le1_worse_than_source_is_caught(tmp_path):
    src, dst, manifest = _clean_2x2(tmp_path)
    # source has 0 degree<=1 nets; inject one into the tiled output
    with open(dst + ".nets") as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumNets"))
    n_nets = int(lines[idx].split(":")[1])
    idx_pins = next(i for i, l in enumerate(lines) if l.strip().startswith("NumPins"))
    n_pins = int(lines[idx_pins].split(":")[1])
    lines[idx] = f"NumNets : {n_nets + 1}\n"
    lines[idx_pins] = f"NumPins : {n_pins + 1}\n"
    lines.append("NetDegree : 1 t0_0/degenerate\n")
    lines.append("    t0_0/o1 I : 0 0\n")
    with open(dst + ".nets", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst, source_prefix=src)
    assert res["status"] == "fail"
    assert res["checks"]["degree_le1_not_worse_than_source"] is False


def test_v1_is_not_run_without_a_config():
    res = vb.check_v1_dreamplace_readable()
    assert res["status"] == "not_run"


@pytest.mark.parametrize("checker,kwargs", [
    (vb.check_h1_rent_shape, {}),
    (vb.check_h2_cut_histogram, {}),
    (vb.check_h3_degree_ks, {}),
    (vb.check_h4_interface_dist_ks, {}),
    (vb.check_h5_k_grid_lambda_ratio, {}),
])
def test_holdout_checks_are_not_run_without_real_cluster_data(checker, kwargs):
    res = checker(**kwargs)
    assert res["status"] == "not_run"


def test_h1_rent_shape_computes_when_given_both_p_values():
    ok = vb.check_h1_rent_shape(p_synthetic=0.62, p_real=0.65)
    assert ok["status"] == "ok"
    assert ok["band_status"] == "evaluated"
    fail = vb.check_h1_rent_shape(p_synthetic=0.62, p_real=0.90)
    assert fail["status"] == "fail"


def test_h1_rent_shape_band_none_only_evaluates_the_relative_leg(tmp_path):
    """T6B (adjudication sec 4.1's H1/H2 row: `not_applicable` -- the 3x3
    array has no same-scale real reference for an absolute band): a pair
    of p's that would fail the *default* band (both outside [0.55,0.80])
    must still pass with `band=None` as long as they agree with each
    other, and the default band itself must be untouched by this call."""
    outside_band = vb.check_h1_rent_shape(p_synthetic=0.53, p_real=0.5221, band=None)
    assert outside_band["status"] == "ok"
    assert outside_band["band_status"] == "not_applicable"
    assert outside_band["band"] is None

    still_gated_on_diff = vb.check_h1_rent_shape(p_synthetic=0.40, p_real=0.5221, band=None)
    assert still_gated_on_diff["status"] == "fail"

    # default band is unchanged (sec 8-4: "不得改動 default band 的數值")
    default_call = vb.check_h1_rent_shape(p_synthetic=0.53, p_real=0.5221)
    assert default_call["band"] == (0.55, 0.80)
    assert default_call["status"] == "fail"  # both p's outside [0.55, 0.80]


def test_h3_degree_ks_is_zero_for_identical_distributions():
    degs = [2, 2, 3, 3, 4, 5, 5, 5]
    res = vb.check_h3_degree_ks(synthetic_degrees=degs, real_degrees=degs)
    assert res["status"] == "ok"
    assert res["ks"] == pytest.approx(0.0)


def test_h3_degree_ks_fails_for_clearly_different_distributions():
    a = [2] * 100
    b = [50] * 100
    res = vb.check_h3_degree_ks(synthetic_degrees=a, real_degrees=b, ks_threshold=0.05)
    assert res["status"] == "fail"
    assert res["ks"] == pytest.approx(1.0)


def test_h5_k_grid_lambda_ratio_band():
    ok = vb.check_h5_k_grid_lambda_ratio(hard_lambda_sum_synthetic=100, n_nets_synthetic=100,
                                          hard_lambda_sum_real=110, n_nets_real=100)
    assert ok["status"] == "ok"
    fail = vb.check_h5_k_grid_lambda_ratio(hard_lambda_sum_synthetic=1000, n_nets_synthetic=100,
                                            hard_lambda_sum_real=10, n_nets_real=100)
    assert fail["status"] == "fail"


def test_b2_h5_matched_resolution_is_not_run_without_matched_inputs():
    res = vb.check_b2_h5_matched_resolution()
    assert res["status"] == "not_run"
    assert res["metric"] == "B2_H5_k_grid_lambda_ratio_matched_resolution"


def test_b2_h5_matched_resolution_gates_on_the_matched_pair_only():
    ok = vb.check_b2_h5_matched_resolution(
        hard_lambda_sum_synthetic_k16=100, n_nets_synthetic_k16=100,
        hard_lambda_sum_real_k4=110, n_nets_real_k4=100)
    assert ok["status"] == "ok"
    assert ok["ratio"] == pytest.approx(100 / 110)
    assert ok["resolution"]["synthetic"]["k"] == 16
    assert ok["resolution"]["real"]["k"] == 4
    assert "diagnostic_mismatched_resolution" not in ok

    fail = vb.check_b2_h5_matched_resolution(
        hard_lambda_sum_synthetic_k16=1000, n_nets_synthetic_k16=100,
        hard_lambda_sum_real_k4=10, n_nets_real_k4=100)
    assert fail["status"] == "fail"


def test_b2_h5_matched_resolution_reports_mismatched_ratio_as_diagnostic_only():
    """The mismatched (both K=16) comparison must never flip `status` --
    only the matched-resolution pair gates (adjudication sec 4.4)."""
    res = vb.check_b2_h5_matched_resolution(
        hard_lambda_sum_synthetic_k16=100, n_nets_synthetic_k16=100,
        hard_lambda_sum_real_k4=110, n_nets_real_k4=100,
        # deliberately way outside [0.7, 1.4] if it *were* gating
        hard_lambda_sum_real_k16=10000, n_nets_real_k16=100)
    assert res["status"] == "ok"  # unaffected by the mismatched values
    assert res["diagnostic_mismatched_resolution"]["ratio"] == pytest.approx(100 / 10000)
    assert res["diagnostic_mismatched_resolution"]["real_k"] == 16


def test_h5prime_is_not_run_without_fitted_glue(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    res = vb.check_h5prime_self_consistency(manifest)
    assert res["status"] == "not_run"


def test_h5prime_passes_on_deterministic_expected_mode_glue(tmp_path):
    src, dst, manifest = _clean_2x2(tmp_path)
    lambda_0, alpha = 1000.0, 0.7
    counts = glue_gen.sample_glue_net_count(None, lambda_0, alpha, R=2, C=2, mode="expected")
    candidate_nodes = {(i, j): ["o0", "o1", "o2"] for i in range(2) for j in range(2)}
    rng = np.random.default_rng(0)
    nets = glue_gen.sample_glue_nets(rng, counts, candidate_nodes)
    glue_gen.append_glue_nets(dst, nets, lambda_0, alpha, seed=None)

    import json
    with open(dst + ".manifest.json") as f:
        manifest2 = json.load(f)

    res = vb.check_h5prime_self_consistency(manifest2, tol=1e-3)
    assert res["status"] == "ok", res
    assert res["rel_err"] <= 1e-3
    assert 0.0 < res["implied_cross_tile_share"] < 1.0

    # and the resulting design still passes V0 (glue merged in cleanly)
    v0 = vb.check_v0_structural(dst, manifest=manifest2, source_prefix=src)
    assert v0["status"] == "ok", v0["errors"]


def test_h5prime_fails_when_manifest_glue_count_disagrees_with_its_own_formula(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    manifest["glue"] = {"n_nets": 1, "n_pins": 2, "kernel": "power_law", "lambda_0": 1000.0, "alpha": 0.7}
    res = vb.check_h5prime_self_consistency(manifest, tol=1e-3)
    assert res["status"] == "fail"


def test_b1_rent_invariance_did_passes_when_glue_increment_matches_across_shapes():
    """Adjudication sec 4.1/4.3's pre-registered prediction: N2's glue
    increment should be essentially shape-invariant once each array's own
    zero-glue combinatorial baseline is subtracted out."""
    res = vb.check_b1_rent_invariance_did(
        p_3x3_glue=0.530, p_3x3_noglue=0.477,   # Delta = 0.053
        p_2x2_glue=0.5262620, p_2x2_noglue=0.475)  # Delta ~ 0.0513
    assert res["status"] == "ok"
    assert res["did"] == pytest.approx(abs(0.053 - 0.0512620), abs=1e-6)
    assert res["threshold"] == 0.03


def test_b1_rent_invariance_did_fails_when_glue_increment_diverges_across_shapes():
    res = vb.check_b1_rent_invariance_did(
        p_3x3_glue=0.610, p_3x3_noglue=0.477,   # Delta = 0.133 (N1-like blowup)
        p_2x2_glue=0.5262620, p_2x2_noglue=0.475)  # Delta ~ 0.0513
    assert res["status"] == "fail"
    assert res["did"] > 0.03


def test_b1_rent_invariance_did_threshold_is_not_a_parameter():
    import inspect
    sig = inspect.signature(vb.check_b1_rent_invariance_did)
    assert list(sig.parameters) == ["p_3x3_glue", "p_3x3_noglue", "p_2x2_glue", "p_2x2_noglue"]
