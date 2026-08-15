"""M4 T6B (2026-08-15 T6 holdout adjudication doc `docs/results/2026-08-
15-m4-t6-holdout-adjudication.md` sec 4.1's B-0..B-6 table): tests for
`ioplace/bench/assemble_t6b.py`.

No test here scans a real multi-GB `.nets` file (3x3_n2's is ~16GB) -- same
discipline as `tests/test_probe_rent_arrays.py`'s module docstring.
Coverage:
  - `scan_degrees_and_tile_glue_endpoints` against a tiny hand-written
    `.nets` fixture (a few net blocks, one of them a glue net);
  - B-0/B-3/B-4 against a genuinely tiny 3x3 array built by
    `build_t6_arrays.build_shape`/`build_shape_no_glue` on the shared toy
    Bookshelf source (same fixture `test_bench_build_t6_arrays.py` uses) --
    real files, real code paths, just toy-scale;
  - B-5/B-6 against small hand-built manifests/lambda_0 values (pure
    arithmetic, no file scanning at all);
  - a dedicated unit test for B-6's per-tile recompute/threshold logic
    with `glue_gen.n2_pair_counts_sinkhorn` monkeypatched to a controlled
    (non-Sinkhorn) fixture, isolating the aggregation logic from whether
    the real solver converges;
  - the fixed-status entries (`pending_measurement`/`not_applicable`/
    `already_evaluated_T6`/`infrastructure_blocked`) and an end-to-end
    `assemble()` schema check on the same toy array.
"""
import json
import os

import numpy as np
import pytest

from ioplace.bench import assemble_t6b as at6b
from ioplace.bench import build_t6_arrays as bta
from ioplace.bench import glue_gen
from tests.test_bench_tile_bookshelf import write_toy_bookshelf


# ---------------------------------------------------------------------------
# scan_degrees_and_tile_glue_endpoints
# ---------------------------------------------------------------------------

_TINY_NETS = """UCLA nets 1.0

NumNets : 3
NumPins : 8

NetDegree : 3 n0
    t0_0/o0 O : -1 -1
    t0_0/o1 I : -1 -1
    t0_0/f0 I : 0 0
NetDegree : 2 n1
    t0_0/o2 O : -1 -1
    t1_0/o3 I : -1 -1
NetDegree : 3 glue0
    t0_0/o4 O : -1 -1
    t1_1/o0 I : -1 -1
    t0_0/o1 I : -1 -1
"""


def test_scan_degrees_and_tile_glue_endpoints_reads_degree_header_only(tmp_path):
    nets_path = str(tmp_path / "toy.nets")
    with open(nets_path, "w") as f:
        f.write(_TINY_NETS)

    degrees, glue_names = at6b.scan_degrees_and_tile_glue_endpoints(nets_path, tile_name="t0_0")
    assert degrees.dtype == np.int32
    assert list(degrees) == [3, 2, 3]
    # glue0 touches t0_0 twice (o4, o1); n0/n1 aren't glue nets so are excluded
    # even though n0 also has t0_0 pins.
    assert sorted(glue_names) == ["o1", "o4"]


def test_scan_degrees_and_tile_glue_endpoints_without_tile_name_skips_glue_scan(tmp_path):
    nets_path = str(tmp_path / "toy.nets")
    with open(nets_path, "w") as f:
        f.write(_TINY_NETS)
    degrees, glue_names = at6b.scan_degrees_and_tile_glue_endpoints(nets_path, tile_name=None)
    assert list(degrees) == [3, 2, 3]
    assert glue_names is None


# ---------------------------------------------------------------------------
# Shared tiny 3x3 fixture (real build_t6_arrays code path, toy scale)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_3x3(tmp_path):
    src = write_toy_bookshelf(str(tmp_path / "src" / "toy"))
    out_dir = str(tmp_path / "arrays")
    candidate_names = bta._movable_node_names(src + ".nodes")
    lambda_0_tile, alpha = 5.0, 0.3
    bta.build_shape(src, lambda_0_tile=lambda_0_tile, alpha=alpha, shape_name="3x3", R=3, C=3,
                     out_dir=out_dir, seed=0, extra_seeds=[], candidate_names=candidate_names)
    bta.build_shape_no_glue(src, shape_name="3x3", R=3, C=3, out_dir=out_dir, seed=0)
    return {"arrays_root": out_dir, "source_prefix": src,
            "lambda_0_tile": lambda_0_tile, "alpha": alpha}


def test_check_b0_passes_on_a_freshly_tiled_clean_array(tiny_3x3):
    res = at6b.check_b0(arrays_root=tiny_3x3["arrays_root"], source_prefix=tiny_3x3["source_prefix"])
    assert res["metric"] == "B0_v0_structural"
    assert set(res["arrays"]) == {"3x3_n2", "3x3_noglue"}
    assert res["status"] == "ok", res["arrays"]
    for name, sub in res["arrays"].items():
        assert sub["status"] == "ok", (name, sub["errors"])
        assert sub["checks"]["matches_manifest"] is True


def test_check_b3_and_b4_share_one_scan_and_have_expected_schema(tiny_3x3):
    prefix_n2 = os.path.join(tiny_3x3["arrays_root"], "3x3_n2", "3x3_n2")
    synth_degrees, glue_names = at6b.scan_degrees_and_tile_glue_endpoints(
        prefix_n2 + ".nets", tile_name="t0_0")
    assert len(synth_degrees) > 0
    assert len(glue_names) > 0  # lambda_0=5.0 should sample at least one glue endpoint at t0_0

    degrees_dir = os.path.join(os.path.dirname(tiny_3x3["arrays_root"]), "degrees")
    b3 = at6b.check_b3(arrays_root=tiny_3x3["arrays_root"], source_prefix=tiny_3x3["source_prefix"],
                        degrees_dir=degrees_dir, synth_degrees=synth_degrees)
    assert b3["metric"] == "H3_degree_ks"
    assert b3["status"] in ("ok", "fail")
    assert b3["n_nets"]["3x3_n2"] == len(synth_degrees)
    assert "ks" in b3 and "per_bucket" in b3

    b4 = at6b.check_b4(arrays_root=tiny_3x3["arrays_root"], source_prefix=tiny_3x3["source_prefix"],
                        seed=0, glue_local_names=glue_names)
    assert b4["metric"] == "H4_interface_dist_ks"
    assert b4["n_glue_endpoints"] == len(glue_names)
    assert b4["status"] in ("ok", "fail")


# ---------------------------------------------------------------------------
# B-5
# ---------------------------------------------------------------------------

def test_check_b5_main_leg_uses_the_real_manifest_and_variants_are_recipe_only(tmp_path):
    arrays_root = str(tmp_path / "arrays")
    prefix = os.path.join(arrays_root, "3x3_n2", "3x3_n2")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    lambda_0_tile, alpha = 10.0, 0.2
    expected = glue_gen.expected_glue_total(lambda_0_tile, alpha, 3, 3)
    manifest = {
        "R": 3, "C": 3,
        "base": {"source_n_nets": 100},
        "glue": {"n_nets": round(expected), "n_pins": round(expected) * 2,
                  "lambda_0": lambda_0_tile, "alpha": alpha},
    }
    with open(prefix + ".manifest.json", "w") as f:
        json.dump(manifest, f)

    res = at6b.check_b5(arrays_root=arrays_root, lambda_0_tile=lambda_0_tile, alpha_main=alpha)
    assert res["metric"] == "B5_h5prime_arithmetic_self_check"
    # main leg's actual == round(expected) by construction -> should pass H5' exactly
    assert res["main_3x3_n2"]["status"] == "ok"
    assert set(res["variants"]) == {"K3_truncated", "K1_alpha0_flat", "K1_alpha1915_optimistic"}
    for name, v in res["variants"].items():
        assert v["variant"] == name
        assert "sinkhorn_diagnostics" in v
        assert "note" in v


# ---------------------------------------------------------------------------
# B-6 -- schema + dedicated recompute-logic unit tests
# ---------------------------------------------------------------------------

def test_check_b6_converges_and_matches_budget_for_all_four_kernels():
    res = at6b.check_b6(lambda_0_tile=10.0, alpha_main=0.3, r=3, c=3)
    assert res["metric"] == "B6_glue_budget_identity"
    assert set(res["kernels"]) == {"K1_main", "K3_truncated", "K1_alpha0_flat", "K1_alpha1915_optimistic"}
    assert res["status"] == "ok"
    for name, k in res["kernels"].items():
        assert k["status"] == "ok"
        assert k["max_rel_err_vs_B"] <= 1e-9
        for tile_sum in k["per_tile_sum"].values():
            assert tile_sum == pytest.approx(res["B"], rel=1e-9)


def test_check_b6_recompute_logic_flags_only_the_kernel_that_violates_budget(monkeypatch):
    """Dedicated unit test for B-6's aggregation logic (2026-08-15
    adjudication sec 4.1 B-6 row's "正式重算一次"): monkeypatches
    `glue_gen.n2_pair_counts_sinkhorn` so the "truncated" kernel returns a
    pair-count map that deliberately violates the per-tile budget while
    every other kernel returns an exactly-on-budget map -- this isolates
    `check_b6`'s own per-tile-sum/relative-error/status logic from whether
    the real Sinkhorn solver converges (already covered by the "converges"
    test above and by `test_bench_glue_gen.py`)."""
    real_fn = glue_gen.n2_pair_counts_sinkhorn

    def fake(lambda_0, alpha, r, c, budget_pairs=3.0, kernel="power_law", **kw):
        if kernel == "truncated":
            # every tile pair gets weight 1 but the total is deliberately
            # scaled to 50% of budget -- every tile's row sum comes out at
            # B/2, a rel_err of 0.5, unambiguously "fail".
            b = budget_pairs * lambda_0
            pairs = glue_gen.tile_pair_distances(r, c)
            n_neighbors = {}
            for (ta, tb_) in pairs:
                n_neighbors[ta] = n_neighbors.get(ta, 0) + 1
                n_neighbors[tb_] = n_neighbors.get(tb_, 0) + 1
            # uniform tile degree for r=c=3's corner/edge/center mix is NOT
            # uniform, so just split B/2 evenly per pair touching a tile --
            # good enough to deliberately break the identity, that's the point.
            counts = {pair: (b / 2.0) / n_neighbors[pair[0]] for pair in pairs}
            return counts, {"n2_iters": 1, "n2_max_rel_dev": 0.5}
        return real_fn(lambda_0, alpha, r, c, budget_pairs=budget_pairs, kernel=kernel, **kw)

    monkeypatch.setattr(glue_gen, "n2_pair_counts_sinkhorn", fake)

    res = at6b.check_b6(lambda_0_tile=10.0, alpha_main=0.3, r=3, c=3)
    assert res["status"] == "fail"
    assert res["kernels"]["K3_truncated"]["status"] == "fail"
    assert res["kernels"]["K3_truncated"]["max_rel_err_vs_B"] > 1e-9
    for name in ("K1_main", "K1_alpha0_flat", "K1_alpha1915_optimistic"):
        assert res["kernels"][name]["status"] == "ok"


# ---------------------------------------------------------------------------
# Fixed-status entries + end-to-end schema
# ---------------------------------------------------------------------------

def test_fixed_status_entries_match_sec_4_1_table():
    entries = at6b._fixed_status_entries()
    assert entries["B1_rent_invariance_did"]["status"] == "pending_measurement"
    assert entries["B2_h5_matched_resolution"]["status"] == "pending_measurement"
    assert entries["H1_rent_shape"]["status"] == "not_applicable"
    assert entries["H2_cut_histogram"]["status"] == "not_applicable"
    assert entries["H1_H2_H3_H4_H6_at_1x2_2x2"]["status"] == "already_evaluated_T6"
    assert entries["V1_dreamplace_readable"]["status"] == "infrastructure_blocked"


def test_assemble_end_to_end_schema_on_the_toy_3x3_fixture(tiny_3x3):
    degrees_dir = os.path.join(os.path.dirname(tiny_3x3["arrays_root"]), "degrees")
    out = at6b.assemble(arrays_root=tiny_3x3["arrays_root"], source_prefix=tiny_3x3["source_prefix"],
                         degrees_dir=degrees_dir)
    assert out["shape"] == "3x3"
    assert out["normalization"] == "n2"
    assert set(out["provenance"]) >= {"repo_commit", "date", "basis_doc", "n2_rule", "lambda_0_tile", "alpha_main"}
    expected_keys = {"B0_v0_structural", "B3_h3prime_degree_ks", "B4_h4prime_interface_self_consistency",
                      "B5_h5prime_arithmetic_self_check", "B6_glue_budget_identity",
                      "B1_rent_invariance_did", "B2_h5_matched_resolution", "H1_rent_shape",
                      "H2_cut_histogram", "H1_H2_H3_H4_H6_at_1x2_2x2", "V1_dreamplace_readable"}
    assert set(out["checks"]) == expected_keys
    assert set(out["gating_check_status"]) == {"B0_v0_structural", "B3_h3prime_degree_ks",
                                                 "B4_h4prime_interface_self_consistency",
                                                 "B5_h5prime_arithmetic_self_check",
                                                 "B6_glue_budget_identity"}
    assert out["scaling_usable"] == all(v == "ok" for v in out["gating_check_status"].values())
    assert out["quality_claims_prohibited"] is True
