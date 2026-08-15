import json
import math

import numpy as np
import pytest

from ioplace.bench import glue_gen
from ioplace.bench import tile_bookshelf as tb
from tests.test_bench_tile_bookshelf import write_toy_bookshelf


def test_phi_is_one_at_d_equals_1():
    for alpha in (0.0, 0.3, 1.0, 2.5):
        assert glue_gen.phi(1, alpha) == pytest.approx(1.0)


def test_mom_2x2_recovers_known_lambda0_alpha():
    lambda_0_true, alpha_true = 12.0, 0.65
    lambda_adj = lambda_0_true * glue_gen.phi(1, alpha_true)
    lambda_diag = lambda_0_true * glue_gen.phi(math.sqrt(2), alpha_true)
    res = glue_gen.solve_mom_2x2(lambda_adj, lambda_diag)
    assert res.status == "ok"
    assert res.lambda_0 == pytest.approx(lambda_0_true)
    assert res.alpha == pytest.approx(alpha_true)


def test_mom_1x4_recovers_known_lambda0_alpha_and_holdout_passes_on_exact_power_law():
    lambda_0_true, alpha_true = 7.5, 0.42
    lambda_1 = lambda_0_true * glue_gen.phi(1, alpha_true)
    lambda_2 = lambda_0_true * glue_gen.phi(2, alpha_true)
    lambda_3 = lambda_0_true * glue_gen.phi(3, alpha_true)  # exact power law -> z should be 0
    res = glue_gen.solve_mom_1x4(lambda_1, lambda_2, lambda_3_obs=lambda_3)
    assert res.status == "ok"
    assert res.lambda_0 == pytest.approx(lambda_0_true)
    assert res.alpha == pytest.approx(alpha_true)
    holdout = res.detail["holdout"]
    assert holdout["z"] == pytest.approx(0.0, abs=1e-9)
    assert holdout["pass"] is True


def test_mom_1x4_holdout_fails_when_d3_departs_from_the_fitted_power_law():
    lambda_0_true, alpha_true = 7.5, 0.42
    lambda_1 = lambda_0_true * glue_gen.phi(1, alpha_true)
    lambda_2 = lambda_0_true * glue_gen.phi(2, alpha_true)
    # d=3 observed at 10x the power-law prediction -> should blow the z<=2 gate
    lambda_3_hat = lambda_0_true * glue_gen.phi(3, alpha_true)
    res = glue_gen.solve_mom_1x4(lambda_1, lambda_2, lambda_3_obs=10 * lambda_3_hat)
    holdout = res.detail["holdout"]
    assert abs(holdout["z"]) > 2.0
    assert holdout["pass"] is False


def test_mom_2x2_zero_count_is_truncated_not_guessed():
    res = glue_gen.solve_mom_2x2(lambda_adj=5.0, lambda_diag=0.0)
    assert res.status == "truncated_at_d"
    assert math.isnan(res.alpha)


def test_mom_2x2_negative_alpha_is_kernel_rejected():
    # farther (diagonal) pairs denser than nearer (adjacent) pairs
    res = glue_gen.solve_mom_2x2(lambda_adj=2.0, lambda_diag=5.0)
    assert res.status == "kernel_rejected"
    assert res.alpha < 0


def test_mom_1x4_zero_count_is_truncated_not_guessed():
    res = glue_gen.solve_mom_1x4(lambda_1=4.0, lambda_2=0.0)
    assert res.status == "truncated_at_d"


def test_tile_pair_distances_2x2_has_4_adjacent_2_diagonal():
    dists = glue_gen.tile_pair_distances(2, 2)
    assert len(dists) == 6
    vals = sorted(dists.values())
    assert vals[:4] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert vals[4:] == pytest.approx([math.sqrt(2), math.sqrt(2)])


def test_tile_pair_distances_1x4_has_3_d1_2_d2_1_d3():
    dists = glue_gen.tile_pair_distances(1, 4)
    vals = sorted(dists.values())
    assert vals.count(1.0) == 3
    assert vals.count(2.0) == 2
    assert vals.count(3.0) == 1


def test_expected_glue_total_matches_manual_sum_2x2():
    lambda_0, alpha = 3.0, 0.7
    got = glue_gen.expected_glue_total(lambda_0, alpha, 2, 2)
    expect = 4 * lambda_0 * glue_gen.phi(1, alpha) + 2 * lambda_0 * glue_gen.phi(math.sqrt(2), alpha)
    assert got == pytest.approx(expect)


def test_bootstrap_ci_contains_the_mean_and_shrinks_toward_it():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=0.1, size=200)
    ci = glue_gen.bootstrap_ci(values, n_resamples=500, seed=1)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 1.0  # tight around a low-variance sample


def test_sample_glue_net_count_expected_mode_is_deterministic_and_matches_formula():
    lambda_0, alpha = 1000.0, 0.7
    counts = glue_gen.sample_glue_net_count(None, lambda_0, alpha, 2, 2, mode="expected")
    total = sum(counts.values())
    expected = glue_gen.expected_glue_total(lambda_0, alpha, 2, 2)
    assert abs(total - expected) / expected <= 0.001
    counts2 = glue_gen.sample_glue_net_count(None, lambda_0, alpha, 2, 2, mode="expected")
    assert counts == counts2  # deterministic -- no rng draw in this mode


def test_sample_glue_net_count_rejects_unknown_mode():
    with pytest.raises(ValueError):
        glue_gen.sample_glue_net_count(np.random.default_rng(0), 1.0, 0.5, 2, 2, mode="bogus")


def test_append_glue_nets_rejects_higher_than_degree_2_hist():
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError):
        glue_gen.sample_glue_nets(rng, {}, {}, deg_hist=(3,))


@pytest.mark.parametrize("R,C", [(1, 2), (2, 2)])
def test_n2_sinkhorn_matches_closed_form_bit_for_bit_on_uniform_shapes(R, C):
    """Adjudication doc sec 5.2's decisive property for choosing Sinkhorn:
    on every shape `n2_pair_counts` already supports (uniform per-tile
    neighbor-distance multiset -> all Z_u equal -> all a_u equal), the
    general Sinkhorn solver must reproduce the exact same closed-form
    answer, so the existing 1x2/2x2 T6 arrays are unaffected by this
    generalization."""
    lambda_0, alpha = 4387.31377772961, 0.10230633302323285
    legacy = glue_gen.n2_pair_counts(lambda_0, alpha, R, C, budget_pairs=3.0)
    sinkhorn, diag = glue_gen.n2_pair_counts_sinkhorn(lambda_0, alpha, R, C, budget_pairs=3.0)
    assert set(sinkhorn) == set(legacy)
    for pair in legacy:
        assert sinkhorn[pair] == pytest.approx(legacy[pair], rel=1e-6)
    assert diag["n2_iters"] >= 1
    assert diag["n2_max_rel_dev"] < 1e-9


def test_n2_sinkhorn_3x3_every_tile_terminal_budget_within_tol():
    """3x3 has 3 distinct neighbor-distance multisets (corner/edge/center)
    -- `n2_pair_counts` raises `NotImplementedError` for it; the Sinkhorn
    generalization must still hold *every* tile's summed glue-terminal
    count to within 1e-9 relative of B (adjudication sec 4.1 B-6)."""
    lambda_0, alpha = 4387.31377772961, 0.10230633302323285
    budget_pairs = 3.0
    B = budget_pairs * lambda_0
    counts, diag = glue_gen.n2_pair_counts_sinkhorn(lambda_0, alpha, 3, 3, budget_pairs=budget_pairs)
    per_tile = {}
    for (ta, tb), c in counts.items():
        per_tile[ta] = per_tile.get(ta, 0.0) + c
        per_tile[tb] = per_tile.get(tb, 0.0) + c
    assert len(per_tile) == 9
    for u, total in per_tile.items():
        assert abs(total - B) / B < 1e-9, (u, total, B)
    # cross-check against the returned diagnostic itself
    assert diag["n2_max_rel_dev"] < 1e-9


@pytest.mark.parametrize("kernel,alpha", [
    ("power_law", 0.0),
    ("power_law", 0.10230633302323285),
    ("power_law", 1.915),
    ("exp", 0.08558),
    ("truncated", 0.10230633302323285),
])
def test_n2_sinkhorn_3x3_total_is_m_b_over_2_for_every_kernel(kernel, alpha):
    """Adjudication sec 5.3's structural finding: under N2, the kernel
    shape only *redistributes* glue nets (each tile is pinned to the same
    B by the Sinkhorn constraint), it never changes the array total -- so
    K1/K2/K3 (and K1 at every alpha) all land on the same
    `m * budget_pairs * lambda_0 / 2` = 59,228.74 for the 3x3 T7 array."""
    lambda_0 = 4387.31377772961
    budget_pairs = 3.0
    B = budget_pairs * lambda_0
    counts, diag = glue_gen.n2_pair_counts_sinkhorn(lambda_0, alpha, 3, 3, budget_pairs=budget_pairs,
                                                      kernel=kernel)
    total = sum(counts.values())
    assert total == pytest.approx(9 * B / 2, rel=1e-6)
    assert total == pytest.approx(59228.74, abs=1.0)
    assert diag["n2_iters"] <= 100


def test_n2_sinkhorn_rejects_unknown_kernel():
    with pytest.raises(ValueError):
        glue_gen.n2_pair_counts_sinkhorn(1.0, 0.5, 3, 3, kernel="bogus")


def test_2x2_total_counts_equal_4x_source_plus_actual_nonzero_glue(tmp_path):
    """The T4 acceptance criterion (design draft sec 7.1): a 2x2 toy array's
    node/net/pin counts = 4x source + the manifest's *actual* (non-zero,
    recorded) glue counts."""
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    dst = str(tmp_path / "out" / "arr")
    manifest = tb.tile(src, dst, R=2, C=2, seed=1)
    base = manifest["base"]

    rng = np.random.default_rng(1)
    lambda_0, alpha = 3.0, 0.8
    counts = glue_gen.sample_glue_net_count(rng, lambda_0, alpha, R=2, C=2)
    assert sum(counts.values()) > 0, "toy lambda_0 chosen so this is overwhelmingly likely"

    candidate_nodes = {(i, j): ["o0", "o1", "o2"] for i in range(2) for j in range(2)}
    nets = glue_gen.sample_glue_nets(rng, counts, candidate_nodes)
    n_glue_nets, n_glue_pins = glue_gen.append_glue_nets(dst, nets, lambda_0, alpha, seed=1)

    assert n_glue_nets == sum(counts.values()) == len(nets)
    assert n_glue_pins == 2 * n_glue_nets
    assert n_glue_nets > 0 and n_glue_pins > 0

    with open(dst + ".manifest.json") as f:
        manifest2 = json.load(f)
    assert manifest2["glue"]["n_nets"] == n_glue_nets
    assert manifest2["glue"]["n_pins"] == n_glue_pins
    assert manifest2["glue"]["lambda_0"] == lambda_0
    assert manifest2["glue"]["alpha"] == alpha

    with open(dst + ".nets") as f:
        lines = f.readlines()
    num_nets_line = next(l for l in lines if l.strip().startswith("NumNets"))
    num_pins_line = next(l for l in lines if l.strip().startswith("NumPins"))
    assert int(num_nets_line.split(":")[1]) == base["n_nets"] + n_glue_nets
    assert int(num_pins_line.split(":")[1]) == base["n_pins"] + n_glue_pins
    n_degree_lines = sum(1 for l in lines if "NetDegree" in l)
    assert n_degree_lines == base["n_nets"] + n_glue_nets

    # every glue pin references an existing tile-prefixed node name
    valid_names = {f"t{i}_{j}/{n}" for i in range(2) for j in range(2) for n in ("o0", "o1", "o2", "o3", "o4", "p0", "f0")}
    for l in lines:
        s = l.strip()
        if s and "NetDegree" not in s and not s.startswith(("UCLA", "NumNets", "NumPins")):
            assert s.split()[0] in valid_names
