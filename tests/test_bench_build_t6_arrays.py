import json
import os

import pytest

from ioplace.bench import build_t6_arrays as bta
from ioplace.bench import glue_gen
from tests.test_bench_tile_bookshelf import write_toy_bookshelf


def _write_cluster_stats(path, lambda_0_tile, alpha):
    with open(path, "w") as f:
        json.dump({"lambda_0_tile": {"value": lambda_0_tile}, "mom": {"alpha": alpha}}, f)


def test_shapes_includes_3x3_alongside_the_original_two():
    assert bta.SHAPES["1x2"] == (1, 2)
    assert bta.SHAPES["2x2"] == (2, 2)
    assert bta.SHAPES["3x3"] == (3, 3)


@pytest.mark.parametrize("shape_name", ["1x2", "2x2"])
def test_expected_pair_counts_n2_still_matches_the_pre_sinkhorn_closed_form(shape_name):
    """Adjudication sec 8-3: switching `_expected_pair_counts`'s n2 branch
    to the Sinkhorn solver must not move the existing 1x2/2x2 manifests'
    expected glue-pair counts -- proves the already-built T6 arrays'
    recorded expectations are unaffected by this generalization."""
    R, C = bta.SHAPES[shape_name]
    lambda_0_tile, alpha = 4387.31377772961, 0.10230633302323285
    legacy = glue_gen.n2_pair_counts(lambda_0_tile, alpha, R, C, budget_pairs=3.0)
    expected, diag = bta._expected_pair_counts("n2", lambda_0_tile, alpha, R, C)
    assert set(expected) == set(legacy)
    for pair in legacy:
        assert expected[pair] == pytest.approx(legacy[pair], rel=1e-6)
    assert diag is not None and diag["n2_iters"] >= 1


def test_expected_pair_counts_n2_3x3_no_longer_raises_not_implemented():
    lambda_0_tile, alpha = 4387.31377772961, 0.10230633302323285
    expected, diag = bta._expected_pair_counts("n2", lambda_0_tile, alpha, 3, 3)
    assert len(expected) == 36  # C(9,2) unordered tile pairs
    assert diag["n2_max_rel_dev"] < 1e-9


def test_expected_pair_counts_n1_reports_no_sinkhorn_diagnostics():
    expected, diag = bta._expected_pair_counts("n1", 1000.0, 0.5, 2, 2)
    assert diag is None


def test_build_shape_n2_manifest_records_sinkhorn_diagnostics(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    candidate_names = bta._movable_node_names(src + ".nodes")
    out_dir = str(tmp_path / "out")
    result = bta.build_shape(src, lambda_0_tile=1000.0, alpha=0.5, shape_name="2x2", R=2, C=2,
                              out_dir=out_dir, seed=0, extra_seeds=[], candidate_names=candidate_names)
    n2_t6 = result["n2"]["t6"]
    assert n2_t6["n2_rule"] == "sinkhorn"
    assert n2_t6["n2_iters"] >= 1
    assert n2_t6["n2_max_rel_dev"] < 1e-9
    # n1 branch has no Sinkhorn to report
    assert "n2_rule" not in result["n1"]["t6"]


def test_no_glue_cli_flag_builds_a_zero_glue_control_array(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    cluster_stats = str(tmp_path / "cluster_stats.json")
    _write_cluster_stats(cluster_stats, 1000.0, 0.5)
    out_dir = str(tmp_path / "out")

    results = bta.main(["--source", src, "--cluster-stats", cluster_stats, "--out-dir", out_dir,
                         "--shapes", "2x2", "--no-glue"])
    manifest = results["2x2"]
    assert manifest["glue"] is None
    assert manifest["no_glue"] is True

    dst_prefix = os.path.join(out_dir, "2x2_noglue", "2x2_noglue")
    with open(dst_prefix + ".manifest.json") as f:
        on_disk = json.load(f)
    assert on_disk["glue"] is None
    assert on_disk["no_glue"] is True

    # base replication only -- no glue nets appended, so NumNets is exactly
    # the manifest's base count (same acceptance bar as the plain tiler)
    with open(dst_prefix + ".nets") as fh:
        lines = fh.readlines()
    num_nets_line = next(l for l in lines if l.strip().startswith("NumNets"))
    assert int(num_nets_line.split(":")[1]) == manifest["base"]["n_nets"]


def test_no_glue_and_normal_build_are_independent_output_directories(tmp_path):
    """--no-glue is a separate invocation from the normal N1/N2 build (not
    a modifier on it) -- its output must not collide with or replace
    `{shape}_n1`/`{shape}_n2`."""
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    cluster_stats = str(tmp_path / "cluster_stats.json")
    _write_cluster_stats(cluster_stats, 1000.0, 0.5)
    out_dir = str(tmp_path / "out")

    bta.main(["--source", src, "--cluster-stats", cluster_stats, "--out-dir", out_dir,
              "--shapes", "1x2", "--extra-seeds"])
    bta.main(["--source", src, "--cluster-stats", cluster_stats, "--out-dir", out_dir,
              "--shapes", "1x2", "--no-glue"])

    assert os.path.exists(os.path.join(out_dir, "1x2_n1", "1x2_n1.manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "1x2_n2", "1x2_n2.manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "1x2_noglue", "1x2_noglue.manifest.json"))
