import os
from types import SimpleNamespace

import numpy as np
import pytest

from ioplace.drivers.run_main_flow import (EVALUATION_NPZ, FREEZE_JSON,
                                           FROZEN_MEMBERSHIP_NPZ,
                                           PLACEMENT_NPZ, REGIONS_JSON,
                                           RESULT_JSON, SOFT_NPZ,
                                           _flag_dreamplace_density_cap,
                                           _resolve_prior, _resolve_regions,
                                           build_parser, main, run_main_flow,
                                           run_soft_phase)
from ioplace.regions import make_grid_regions


def test_parser_exposes_the_v2_switches():
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.k == 16 and args.rtype == "grid" and args.phase == "all"
    assert args.init == "die_center" and args.norm_policy == "grandplan"
    assert args.every == 50 and args.freeze_window == 50
    assert args.freeze_overflow == 0.15 and args.freeze_tau_rel == 0.05
    assert args.freeze_churn == 0.005
    assert args.density_clamp_lo == 0.25 and args.density_clamp_hi == 4.0
    assert args.remap_blocks == "auto"
    assert args.argmax_chunk == 4 and args.norm_target_share is None
    for bad, choices in (("--phase", "soft fence all"),
                         ("--init", "die_center region_center seed"),
                         ("--norm-policy", "legacy grandplan adaptive")):
        for choice in choices.split():
            parser.parse_args(["--config", "c.json", "--out-dir", "o", bad, choice])
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "c.json", "--out-dir", "o", "--phase", "bogus"])


def test_resolve_regions_rejects_a_region_file_from_another_die(tmp_path):
    path = str(tmp_path / "regions.json")
    make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=8).to_json(path)
    rs, source = _resolve_regions((0., 0., 100., 100.), 4, "grid", 0, path)
    assert rs.k == 4 and source == "file"
    with pytest.raises(ValueError, match="native post-read units"):
        _resolve_regions((0., 0., 200., 100.), 4, "grid", 0, path)


def test_resolve_regions_rejects_a_region_file_with_the_wrong_k(tmp_path):
    """C-6: IoTerm(K=k), argmax_region(..., k), region_centers, freeze_record
    and save_membership all assume rs.k == --k. Arms (b)/ours pass a producer
    regions.json, so a K=16 geometry against a K=4 term must not run."""
    path = str(tmp_path / "regions.json")
    make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=8).to_json(path)
    with pytest.raises(ValueError, match="k=4"):
        _resolve_regions((0., 0., 100., 100.), 16, "grid", 0, path)


def test_resolve_regions_falls_back_to_the_builtin_grid():
    rs, source = _resolve_regions((0., 0., 100., 100.), 4, "grid", 0, None)
    assert rs.k == 4 and source == "builtin"


def test_resolve_prior_remaps_partitioner_block_ids(tmp_path, monkeypatch):
    from ioplace.artifacts import save_membership
    path = str(tmp_path / "membership.npz")
    part = np.array([0, 1, 2, 3], dtype=np.int32)
    save_membership(path, part, source="mtkahypar", k=4, seed=0, epsilon=0.03)
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=8)
    calls = []

    def fake_assign(parts, nl, region_set):
        calls.append(region_set.k)
        return np.asarray(parts, dtype=np.int32)[::-1].copy()

    monkeypatch.setattr("ioplace.drivers.run_placement_two_stage.assign_blocks_to_regions",
                        fake_assign)
    built = []

    def nl_fn():
        built.append(1)
        return None

    got, info = _resolve_prior(path, "auto", nl_fn=nl_fn, rs=rs, num_movable=4)
    assert calls == [4] and info["remapped"] is True
    assert got.tolist() == [3, 2, 1, 0]
    assert built == [1]

    got_off, info_off = _resolve_prior(path, "off", nl_fn=nl_fn, rs=rs,
                                       num_movable=4)
    assert info_off["remapped"] is False and got_off.tolist() == [0, 1, 2, 3]
    # D-7: no remap, no netlist -- netlist_from_placedb copies pin2node and
    # flat_net2pin, which is multiple GB at 10M-30M cells.
    assert built == [1]


def test_fence_phase_alone_requires_the_soft_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="soft.npz"):
        run_main_flow("unused.json", str(tmp_path), phase="fence")


def test_soft_phase_rejects_cadences_and_inputs_it_cannot_honour(tmp_path):
    """Every guard `run_soft_phase` evaluates before it opens DREAMPlace."""
    out = str(tmp_path)
    base = dict(k=4, rtype="grid", seed=0)
    with pytest.raises(ValueError, match="positive multiple"):
        run_soft_phase("c.json", out, every=0, **base)
    with pytest.raises(ValueError, match="positive multiple"):
        run_soft_phase("c.json", out, every=50, freeze_window=75, **base)
    with pytest.raises(ValueError, match="home_period"):
        run_soft_phase("c.json", out, every=50, home_period=75, **base)
    with pytest.raises(ValueError, match="rho-max"):
        run_soft_phase("c.json", out, f_ft_max=0.2, rho_max=0.0, **base)
    with pytest.raises(ValueError, match="requires --membership"):
        run_soft_phase("c.json", out, init="region_center", **base)
    with pytest.raises(ValueError, match="requires --seed-npz"):
        run_soft_phase("c.json", out, init="seed", **base)


def test_run_main_flow_rejects_an_unknown_phase_init_or_policy(tmp_path):
    for kwargs, pattern in ((dict(phase="bogus"), "unknown phase"),
                            (dict(init="bogus"), "unknown init"),
                            (dict(norm_policy="bogus"), "unknown norm policy")):
        with pytest.raises(ValueError, match=pattern):
            run_main_flow("c.json", str(tmp_path), **kwargs)


def test_main_forwards_every_switch_to_run_main_flow(monkeypatch):
    seen = {}

    def fake_run(config_json, out_dir, **options):
        seen.update(options)
        seen["config_json"], seen["out_dir"] = config_json, out_dir
        return {"ok": True}

    monkeypatch.setattr("ioplace.drivers.run_main_flow.run_main_flow", fake_run)
    result = main(["--config", "c.json", "--out-dir", "o", "--k", "8",
                   "--phase", "soft", "--init", "seed", "--seed-npz", "s.npz",
                   "--membership", "m.npz", "--remap-blocks", "off",
                   "--norm-policy", "adaptive", "--norm-target-share", "io=0.4",
                   "--regions", "r.json", "--every", "25", "--f-ft-max", "0.2",
                   "--argmax-chunk", "2", "--density-clamp-hi", "8.0",
                   "--d-max", "1000", "--dp-seed", "7", "--deterministic", "1",
                   "--check-invariant", "--benchmark-kind", "synthetic"])
    assert result == {"ok": True}
    assert seen["config_json"] == "c.json" and seen["out_dir"] == "o"
    assert seen["k"] == 8 and seen["phase"] == "soft"
    assert seen["init"] == "seed" and seen["seed_npz"] == "s.npz"
    assert seen["membership_npz"] == "m.npz" and seen["remap_blocks"] == "off"
    assert seen["norm_policy"] == "adaptive" and seen["norm_target_share"] == "io=0.4"
    assert seen["regions_json"] == "r.json" and seen["every"] == 25
    assert seen["f_ft_max"] == 0.2 and seen["argmax_chunk"] == 2
    assert seen["density_clamp_hi"] == 8.0 and seen["ignore_net_degree"] == 1000
    assert seen["dp_seed"] == 7 and seen["deterministic"] == 1
    assert seen["check_invariant"] is True
    assert seen["benchmark_kind"] == "synthetic"


def test_a_clamp_ceiling_above_dreamplaces_own_cap_is_flagged():
    """Task 4 review: `update_density_weight_op_overflow` clamps the new weight
    to 10 itself (PlaceObj.py:875), so a hi_abs above that can never bind."""
    log = [{"hi_abs": 4.0}, {"hi_abs": 40.0}]
    assert _flag_dreamplace_density_cap(log) is log
    assert log[0]["hi_abs_capped_by_dreamplace"] is False
    assert log[1]["hi_abs_capped_by_dreamplace"] is True


import json
from pathlib import Path


def test_flag_dreamplace_density_cap_against_a_real_clamp_log():
    """M4 (task review): the existing hand-typed-dict test above cannot
    catch a rename of `clamp_density_weight`'s own `hi_abs` key
    (fence_phase.py:59), the only way this helper can actually break in
    production. Build the log from the *real* function instead, on a
    minimal fake model with just the `density_weight` tensor it reads."""
    import torch
    from ioplace.fence_phase import clamp_density_weight

    def clamp_entry(value, *, reference, hi):
        class FakeModel:
            density_weight = torch.tensor([value])
        return clamp_density_weight(FakeModel(), reference=reference,
                                    lo=0.25, hi=hi)

    log = [clamp_entry(1.0, reference=1.0, hi=4.0),
           clamp_entry(1.0, reference=10.0, hi=4.0)]
    assert _flag_dreamplace_density_cap(log) is log
    for entry in log:
        # clamp_density_weight's documented key set (fence_phase.py:59-62);
        # a rename of any of these fails here, not only in the fabricated
        # dict above.
        assert set(entry) >= {"reference", "lo_abs", "hi_abs", "before",
                              "after", "bound"}
    assert log[0]["hi_abs"] == pytest.approx(4.0)
    assert log[0]["hi_abs_capped_by_dreamplace"] is False
    assert log[1]["hi_abs"] == pytest.approx(40.0)
    assert log[1]["hi_abs_capped_by_dreamplace"] is True


def test_run_main_flow_maps_fence_compliance_and_matches_the_result_schema(
        tmp_path, monkeypatch):
    """I1 (task review): `fence_compliance()["lower_left"]`/`["center"]` ->
    `result["fence_compliance"]`/`["fence_compliance_center"]`
    (run_main_flow.py's result assembly) and result.json's exact key set
    (artifacts.MAIN_FLOW_RESULT_FIELDS) are both reached only through a full
    `run_main_flow`, and neither had a fast test. Stubs run_soft_phase and
    run_fence_gp so this needs no real DREAMPlace placement, but builds
    fence_compliance/region_area_balance/the eval metrics/the legalization
    fields from the *live* functions rather than hand-typed dicts, so a
    rename of any of their return keys still fails this test. The geometry
    below is chosen so `lower_left` and `center` disagree (1.0 vs 0.75), so a
    *swapped* mapping fails too (re-review finding: the original all-1.0
    geometry could not detect a swap)."""
    from ioplace.artifacts import (MAIN_FLOW_RESULT_FIELDS, save_freeze,
                                   save_membership, save_positions)
    from ioplace.drivers.run_placement import (_legalization_fields,
                                               _pack_eval_metrics)
    from ioplace.freeze import freeze_record, region_cell_stats
    from ioplace.main_flow_metrics import fence_compliance, region_area_balance
    from ioplace.region_grid import RegionGrid

    out = tmp_path / "run"
    out.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")

    k = 4
    die = (0.0, 0.0, 100.0, 100.0)
    rs = make_grid_regions(die, 2, 2, lattice=8)
    part = np.array([0, 1, 2, 3], dtype=np.int32)
    size_x = np.array([10.0, 10.0, 10.0, 10.0])
    size_y = np.array([10.0, 10.0, 10.0, 10.0])
    # Cell 0's lower-left corner sits in region 0, but at (45, 45) with a
    # 10x10 size its centre (50, 50) straddles the x=y=50 region boundary and
    # lands in region 3 instead -- so lower_left and center disagree (1.0 vs
    # 0.75) and a swapped fence_compliance()["lower_left"]/["center"] mapping
    # fails this test (re-review: the previous all-1.0 geometry could not).
    node_x = np.array([45.0, 60.0, 10.0, 60.0])
    node_y = np.array([45.0, 10.0, 60.0, 60.0])

    regions_path = str(out / REGIONS_JSON)
    rs.to_json(regions_path)
    soft_path = str(out / SOFT_NPZ)
    save_positions(soft_path, node_x, node_y, die=die, shift_factor=(0.0, 0.0),
                   scale_factor=1.0, placedb_sha256="fake", kind="soft")
    membership_path = str(out / FROZEN_MEMBERSHIP_NPZ)
    save_membership(membership_path, part, source="freeze", k=k, seed=0,
                    epsilon=0.0)
    freeze_path = str(out / FREEZE_JSON)
    record = freeze_record(
        iteration=10, reason="gp_end", overflow=0.05, tau=0.1, tau_rel=0.02,
        churn=0.0, k=k, io_soft=5, membership_npz=membership_path,
        soft_npz=soft_path, repaired_empty_regions=[], gp_iterations_soft=11,
        density_weight_soft=1.5,
        stats=region_cell_stats(part, size_x, size_y, rs))
    save_freeze(freeze_path, record)

    def fake_run_soft_phase(config_json, out_dir, *, k, timer=None, **kwargs):
        return {"freeze": record, "part": part, "soft_npz": soft_path,
                "membership_npz": membership_path,
                "regions_json": regions_path, "region_source": "builtin",
                "init": {"mode": "die_center"}, "prior": None,
                "norm_policy": kwargs.get("norm_policy"),
                "trace_path": str(out / "norm_trace.jsonl"),
                "probe_samples": [], "num_probes": 0, "num_refreshes": 0,
                "gp_iterations_soft": 11, "density_weight_soft": 1.5,
                "placedb_sha256": "fake", "die_native": die}

    rg = RegionGrid(rs)
    real_compliance = fence_compliance(rg, node_x, node_y, part, size_x, size_y)
    real_balance = region_area_balance(part, size_x, size_y, rs)
    # R-1 (task 5): run_fence_gp now hands back the raw EvalResult beside its
    # packed metrics dict, so _pack_straddle_metrics has something to read.
    real_eval_result = SimpleNamespace(
        io_count=3, ft_count=0, tree_wl=1.0, hpwl=2.0, large_net_lb=0,
        hard_lambda_sum=0.0, io_rg=0.0, ft_rg=0.0,
        straddle_cells=1, straddle_area_fraction=0.25,
        straddle_pin_split_nets=0, straddle_out_area=0.5,
        straddle_movable_area=2.0, straddle_wide_cells=0)
    real_metrics = _pack_eval_metrics(real_eval_result)
    real_legal_fields = _legalization_fields(True, 0, 1)

    def fake_run_fence_gp(config_json, out_dir, *, region_set, part,
                          positions, reference_density_weight,
                          density_clamp_lo=0.25, density_clamp_hi=4.0,
                          dp_seed=None, deterministic=None, extra_terms=(),
                          timer=None):
        placement_path = str(out / PLACEMENT_NPZ)
        evaluation_path = str(out / EVALUATION_NPZ)
        save_positions(placement_path, node_x, node_y, die=die,
                       shift_factor=(0.0, 0.0), scale_factor=1.0,
                       placedb_sha256="fake", kind="placement")
        np.savez_compressed(evaluation_path, node_x=node_x, node_y=node_y)
        return {
            "metrics": real_metrics, "eval_result": real_eval_result,
            "io_fence_gp": 4,
            "io_fence_gp_source": "legalize_op",
            "hpwl_gp": 2.5, "hpwl_lg": 2.0,
            "fence_compliance": real_compliance,
            "region_area_balance": real_balance,
            "density_weight_clamp": [],
            "escape_cell": 0, "escape_from": 1,
            "legal_fields": real_legal_fields,
            "scale_fields": {"effective_target_density": 0.7,
                            "num_filler_nodes": 0, "num_bins_x": 8,
                            "num_bins_y": 8},
            "final_overflow": 0.05, "stop_overflow_reached": True,
            "gp_iterations_fence": 5, "gp_iteration_budget": 100,
            "placement_npz": placement_path, "evaluation_npz": evaluation_path,
            "params_seed": 7, "deterministic": 1,
        }

    monkeypatch.setattr("ioplace.drivers.run_main_flow.run_soft_phase",
                        fake_run_soft_phase)
    monkeypatch.setattr("ioplace.drivers.run_main_flow.run_fence_gp",
                        fake_run_fence_gp)

    result = run_main_flow(str(config), str(out), k=k, phase="all")

    assert result["fence_compliance"] == real_compliance["lower_left"] == 1.0
    assert result["fence_compliance_center"] == real_compliance["center"] == 0.75
    assert set(result) == set(MAIN_FLOW_RESULT_FIELDS)

    on_disk = json.loads((out / RESULT_JSON).read_text())
    assert set(on_disk) == set(MAIN_FLOW_RESULT_FIELDS)

    # Regression: placement.npz must round-trip through load_positions like
    # every other positions artefact (soft.npz, seed.npz) -- it used to be
    # written with a bare np.savez_compressed that carried none of the
    # schema_version/die/shift_factor/scale_factor/placedb_sha256/kind
    # stamps load_positions requires, and the fake in this very test used to
    # mirror that same bare-savez bug instead of catching it.
    from ioplace.artifacts import load_positions
    loaded = load_positions(str(out / PLACEMENT_NPZ))
    assert np.array_equal(loaded.node_x, node_x)
    assert np.array_equal(loaded.node_y, node_y)
    assert loaded.die == die
    assert loaded.shift_factor == (0.0, 0.0)
    assert loaded.scale_factor == 1.0
    assert loaded.placedb_sha256 == "fake"
    assert loaded.kind == "placement"


@pytest.mark.slow
@pytest.mark.parametrize("norm_policy", ["legacy", "grandplan"])
def test_main_flow_end_to_end_on_gcd_closes_the_io_identity(tmp_path, norm_policy):
    """Design v2 sec 9's small end-to-end case: producer-free grid K=4 on GCD,
    full main flow, asserting the artefacts exist and
    io(final) = io(soft) + io_delta_at_freeze + lg_loss.

    Parametrised over the two shipped policies (Global Constraints ruling
    2026-09-19): `grandplan` is the driver's default and design sec 4's
    coefficient path, `legacy` is the bit-for-bit regression arm. `adaptive`
    is ablation-only and is covered by tests/test_norm_adapter.py, not here."""
    from ioplace.artifacts import MAIN_FLOW_RESULT_FIELDS
    from ioplace.paths import REPO_ROOT
    config = Path(REPO_ROOT) / "results/route_feedback_20260914/gcd.json"
    if not config.exists():
        pytest.skip("GCD benchmark required")
    out = tmp_path / f"gcd_k4_{norm_policy}"
    result = run_main_flow(str(config), str(out), k=4, rtype="grid", seed=0,
                           init="die_center", every=25, freeze_window=50,
                           rho_max=0.05, norm_policy=norm_policy,
                           dp_seed=1000, deterministic=1)
    assert result["norm_policy"] == norm_policy

    # the normalisation trace's file name is the policy's, not a constant:
    # legacy writes legacy_trace.jsonl, grandplan/adaptive norm_trace.jsonl
    # (amendment A-9).
    trace_name = ("legacy_trace.jsonl" if norm_policy == "legacy"
                  else "norm_trace.jsonl")
    for name in ("regions.json", "soft.npz", "freeze.json",
                 "frozen_membership.npz", "placement.npz", "evaluation.npz",
                 trace_name, "result.json"):
        assert (out / name).exists(), name
    on_disk = json.loads((out / "result.json").read_text())
    for field in MAIN_FLOW_RESULT_FIELDS:
        assert field in on_disk, field
    assert on_disk["mode"] == "main_flow" and on_disk["schema_version"] == 1

    # The accounting identity (design v2 sec 7) is algebraic:
    # io_identity_residual is 0 for *any* three inputs, so it is asserted as
    # the schema invariant it is, and the real check comes from provenance
    # (amendment D-1) -- io_fence_gp must be the legalize_op wrapper's exact
    # pre-LG measurement, and io_soft must be the number freeze.json recorded.
    assert result["io_identity_residual"] == 0
    assert result["io_fence_gp_source"] == "legalize_op"
    assert result["io_soft"] == result["freeze"]["io_soft"]
    assert result["io_count"] == result["io_fence_gp"] + result["lg_loss"]

    freeze = result["freeze"]
    assert freeze["reason"] in ("criterion", "gp_end")
    assert freeze["k"] == 4 and len(freeze["region_cell_count"]) == 4
    assert min(freeze["region_cell_count"]) >= 1          # no empty fence region
    assert freeze["density_weight_soft"] > 0

    assert result["fence_compliance_center"] >= 0.9
    assert result["region_area_balance"]["utilization_ratio"] is not None
    assert result["region_area_balance"]["empty_regions"] == []
    assert result["density_weight_clamp"] and "bound" in result["density_weight_clamp"][0]
    assert result["hpwl"] > 0 and result["hpwl_gp"] > 0 and result["hpwl_lg"] > 0
    assert result["gp_iterations_soft"] >= 1 and result["gp_iterations_fence"] >= 1
    assert result["peak_mem_mb"] > 0 and result["t_gp_fence"] > 0
    assert set(result["artifacts"]) >= {"soft_npz", "freeze_json",
                                        "membership_npz", "placement_npz",
                                        "evaluation_npz"}

    # Row assertions branch on the policy (amendment A-9): legacy_trace.jsonl
    # carries publish_atomic's keys plus the driver's per-probe extras;
    # norm_trace.jsonl carries exactly norm_trace.ROW_FIELDS and nothing else,
    # because NormTraceWriter validates every row. The else branch is what the
    # --norm-policy grandplan|adaptive arms of the spec section 4 ablation hit.
    from ioplace.norm_trace import ROW_FIELDS, TERM_FIELDS, read_norm_trace
    rows = read_norm_trace(str(out / trace_name))
    assert rows
    if norm_policy == "legacy":
        for row in rows:
            for key in ("iteration", "overflow", "tau", "tau_rel", "lambda_io",
                        "grad_l1_wl", "grad_l1_io", "obj_version", "policy",
                        "io_count", "ft_count", "churn"):
                assert key in row, key
    else:
        for row in rows:
            assert set(row) == set(ROW_FIELDS)
            assert row["policy"] == norm_policy and "io" in row["terms"]
            assert set(row["terms"]["io"]) == set(TERM_FIELDS)
        # design sec 4's point: the IO coefficient is derived from a measured
        # gradient ratio, and the committed lambda is un-ramped while the
        # applied one carries the activation ramp (review I1).
        assert any(row["terms"]["io"]["ratio_ema"] for row in rows)
        assert any(row["terms"]["io"]["lam"] > 0.0 for row in rows)
        assert all(row["terms"]["io"]["lam_applied"] <= row["terms"]["io"]["lam"]
                   for row in rows)

    # soft-phase provenance survives into result.json (amendment D-6)
    summary = result["soft_summary"]
    assert summary["region_source"] == "builtin"
    assert summary["init"]["mode"] == "die_center" and summary["prior"] is None
    assert summary["num_probes"] == len(summary["probe_samples"])
    # Under legacy the adapter writes each row as the driver hands it over.
    # Under grandplan/adaptive TermNormalizer holds the row until
    # mark_refreshed(), and the freeze raise deliberately skips that last
    # refresh (amendment D-13), so a criterion freeze drops exactly one row --
    # by design: those coefficients never reached the objective.
    dropped = int(norm_policy != "legacy" and freeze["reason"] == "criterion")
    assert len(rows) == summary["num_probes"] - dropped
    assert summary["trace_path"].endswith(trace_name)

    # --phase fence reproduces the fence half from the artefacts alone
    rerun = run_main_flow(str(config), str(out), phase="fence", k=4,
                          dp_seed=1000, deterministic=1)
    assert rerun["io_soft"] == result["io_soft"]
    assert rerun["io_count"] == result["io_count"]
    assert rerun["lg_loss"] == result["lg_loss"]


def test_parser_defaults_the_node_anchor_to_center():
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.node_anchor == "center"


def test_main_flow_result_fields_carry_the_p_f_diagnostics():
    from ioplace.artifacts import MAIN_FLOW_RESULT_FIELDS
    from ioplace.straddle import STRADDLE_SCALARS
    for name in ("node_anchor",) + STRADDLE_SCALARS:
        assert name in MAIN_FLOW_RESULT_FIELDS, name
