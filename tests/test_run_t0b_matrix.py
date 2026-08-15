"""No-GPU-required tests for M4 T0b's run_t0b_matrix.py: config generation
and --dry-run plan building only (never calls run_point / imports torch).
"""
import json
import os
import subprocess
import sys

from ioplace.diagnostics.probes_m4 import run_t0b_matrix as t0b


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_build_points_is_11_unique_tags_9_plus_2():
    points = t0b.build_points()
    assert len(points) == 11
    tags = [p["tag"] for p in points]
    assert len(set(tags)) == 11

    bins_density = [p for p in points if p["design_point"] == "bins_density"]
    net_drop = [p for p in points if p["design_point"] == "net_drop"]
    assert len(bins_density) == 9
    assert len(net_drop) == 2

    combos = {(p["meta"]["num_bins"], p["meta"]["target_density"]) for p in bins_density}
    expected_combos = {(nb, td) for nb in (1024, 2048, 4096) for td in (0.70, 0.835, 0.90)}
    assert combos == expected_combos

    fracs = sorted(p["meta"]["drop_fraction"] for p in net_drop)
    assert fracs == [0.25, 0.50]


def test_bins_density_config_fixed_protocol_fields():
    for p in t0b.build_points():
        if p["design_point"] != "bins_density":
            continue
        cfg = p["config"]
        assert cfg["legalize_flag"] == 0
        assert cfg["deterministic_flag"] == 1
        assert cfg["random_seed"] == 1000
        assert cfg["detailed_place_flag"] == 0
        assert cfg["global_place_stages"][0]["iteration"] == 2000
        nb = p["meta"]["num_bins"]
        assert cfg["num_bins_x"] == nb
        assert cfg["num_bins_y"] == nb
        assert cfg["global_place_stages"][0]["num_bins_x"] == nb
        assert cfg["global_place_stages"][0]["num_bins_y"] == nb
        assert cfg["target_density"] == p["meta"]["target_density"]
        # design point 1 must not touch the source netlist -- LEF/DEF inputs
        # (and therefore N_pins) stay exactly the group main config's own.
        assert "lef_input" in cfg and "def_input" in cfg


def test_net_drop_config_points_at_derived_prefix_and_group_bins_density():
    with open(t0b.GROUP_CONFIG) as f:
        group_cfg = json.load(f)

    for p in t0b.build_points():
        if p["design_point"] != "net_drop":
            continue
        cfg = p["config"]
        assert cfg["legalize_flag"] == 0
        assert cfg["deterministic_flag"] == 1
        assert cfg["random_seed"] == 1000
        assert cfg["global_place_stages"][0]["iteration"] == 2000
        # design point 2 must not touch bins/density -- only N_pins moves.
        assert cfg["num_bins_x"] == group_cfg["num_bins_x"]
        assert cfg["num_bins_y"] == group_cfg["num_bins_y"]
        assert cfg["target_density"] == group_cfg["target_density"]
        assert "lef_input" not in cfg and "def_input" not in cfg
        assert "sol_file_format" not in cfg

        frac = p["meta"]["drop_fraction"]
        expected_prefix = t0b.net_drop_prefix(frac, p["meta"]["net_drop_seed"])
        assert cfg["aux_input"] == expected_prefix + ".aux"
        assert p["meta"]["net_drop_prefix"] == expected_prefix


def test_build_points_is_deterministic():
    a = [(p["tag"], p["config"]) for p in t0b.build_points()]
    b = [(p["tag"], p["config"]) for p in t0b.build_points()]
    assert a == b


def test_build_plan_writes_11_scratch_configs_and_matches_points(tmp_path):
    scratch_dir = str(tmp_path / "scratch_configs")
    out_dir = str(tmp_path / "t0b_out")
    plan = t0b.build_plan(scratch_dir=scratch_dir, out_dir=out_dir)

    assert len(plan) == 11
    tags = {e["tag"] for e in plan}
    assert tags == {p["tag"] for p in t0b.build_points()}

    for entry in plan:
        assert os.path.exists(entry["config_path"])
        assert entry["config_path"] == os.path.join(scratch_dir, entry["tag"] + ".json")
        assert entry["out_json"] == os.path.join(out_dir, entry["tag"] + ".json")
        with open(entry["config_path"]) as f:
            on_disk = json.load(f)
        assert on_disk["random_seed"] == 1000
        assert on_disk["legalize_flag"] == 0
        # --dry-run only ever generates configs; nothing under out_dir yet.
        assert not os.path.exists(entry["out_json"])


def test_dry_run_cli_writes_plan_and_no_result_json(tmp_path):
    scratch_dir = str(tmp_path / "scratch")
    out_dir = str(tmp_path / "out")
    plan_out = str(tmp_path / "plan.json")

    proc = subprocess.run(
        [sys.executable, "-m", "ioplace.diagnostics.probes_m4.run_t0b_matrix",
         "--dry-run", "--scratch-dir", scratch_dir, "--out-dir", out_dir,
         "--plan-out", plan_out],
        cwd=_repo_root(), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    with open(plan_out) as f:
        plan = json.load(f)
    assert len(plan) == 11
    assert not os.path.isdir(out_dir), "dry-run must not execute any point"
    # every planned config actually landed on disk
    for entry in plan:
        assert os.path.exists(entry["config_path"])


def test_dry_run_only_restricts_to_single_tag(tmp_path):
    scratch_dir = str(tmp_path / "scratch")
    out_dir = str(tmp_path / "out")
    plan_out = str(tmp_path / "plan.json")
    tag = "group__netdrop25__seed0"

    proc = subprocess.run(
        [sys.executable, "-m", "ioplace.diagnostics.probes_m4.run_t0b_matrix",
         "--dry-run", "--only", tag,
         "--scratch-dir", scratch_dir, "--out-dir", out_dir, "--plan-out", plan_out],
        cwd=_repo_root(), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    with open(plan_out) as f:
        plan = json.load(f)
    assert len(plan) == 1
    assert plan[0]["tag"] == tag


def test_only_unknown_tag_errors(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "ioplace.diagnostics.probes_m4.run_t0b_matrix",
         "--dry-run", "--only", "not_a_real_tag",
         "--scratch-dir", str(tmp_path / "scratch"), "--out-dir", str(tmp_path / "out"),
         "--plan-out", str(tmp_path / "plan.json")],
        cwd=_repo_root(), capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "not_a_real_tag" in proc.stderr
