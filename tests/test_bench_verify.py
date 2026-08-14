import os
import shutil

import numpy as np
import pytest

from ioplace.bench import glue_gen
from ioplace.bench import tile_bookshelf as tb
from ioplace.bench import verify_bench as vb
from tests.test_bench_tile_bookshelf import write_toy_bookshelf


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
    assert res["checks"]["no_self_loops"] is True
    assert res["checks"]["all_nodes_referenced_or_declared_unconnected"] is True
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


def test_v0_catches_self_loop(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
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
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["no_self_loops"] is False
    assert any("self-loop" in e for e in res["errors"])


def test_v0_catches_unreferenced_node(tmp_path):
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


def test_v0_catches_row_gap(tmp_path):
    _, dst, manifest = _clean_2x2(tmp_path)
    with open(dst + ".scl") as f:
        text = f.read()
    # bump exactly one row's Coordinate by 1 to open a gap in its x-band
    lines = text.splitlines(keepends=True)
    for i, l in enumerate(lines):
        if l.strip() == "Coordinate : 2":
            lines[i] = "\tCoordinate : 3\n"
            break
    with open(dst + ".scl", "w") as f:
        f.writelines(lines)
    res = vb.check_v0_structural(dst)
    assert res["status"] == "fail"
    assert res["checks"]["rows_no_overlap_no_gap"] is False


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
    fail = vb.check_h1_rent_shape(p_synthetic=0.62, p_real=0.90)
    assert fail["status"] == "fail"


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
