import os
import numpy as np
import pytest

from ioplace.drivers.run_main_flow import (_flag_dreamplace_density_cap,
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
