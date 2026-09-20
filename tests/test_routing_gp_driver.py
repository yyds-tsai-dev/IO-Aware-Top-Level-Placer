"""Real GP integration of routing gradients and later measured-router feedback."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from ioplace.paths import REPO_ROOT
from ioplace.route_eval.online_openroad import digest


@pytest.mark.slow
def test_real_gp_route_gradient_and_two_later_router_observations(tmp_path):
    root = Path(REPO_ROOT)
    binary = os.environ.get("OPENROAD_BIN")
    config = root / "results/route_feedback_20260914/gcd.json"
    calibration = root / "results/route_feedback_20260914/gcd_swap32_seed1000/baseline"
    if not binary or not Path(binary).exists() or not config.exists():
        pytest.skip("GCD and OpenROAD runtime required")
    reports = {}
    for mode in ("wa", "joint"):
        output = tmp_path / mode
        calibration_input = calibration
        if mode == "joint":
            # A stale DEF beside a current physical checkpoint must not cause
            # feedback to divide measured IO by a different placement's model.
            calibration_input = tmp_path / "calibration_with_stale_def"
            calibration_input.mkdir()
            for name in ("out.def", "coord.json", "regions.json", "netmap.json", "placement.npz"):
                shutil.copy2(calibration / name, calibration_input / name)
            shutil.copy2(tmp_path / "wa/routed/out.def", calibration_input / "out.def")
        command = [sys.executable, str(root / "src/scripts/run_route_gp.py"),
                   "--config", str(config), "--out", str(output), "--mode", mode,
                   "--iterations", "24", "--start", "5", "--rebuild", "5",
                   "--router-every", "8", "--seed", "1000", "--stop-overflow", "0",
                   "--calibration", str(calibration_input), "--openroad", binary]
        run = subprocess.run(command, capture_output=True, text=True,
                             env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"})
        assert run.returncode == 0, run.stderr[-6000:] + run.stdout[-2000:]
        reports[mode] = json.loads((output / "result.json").read_text())
        protocol = json.loads((output / "protocol.json").read_text())
        assert protocol["input_sha256"] and protocol["tool_sha256"]
        for source in ("src/scripts/run_route_gp.py", "src/ioplace/ops/routing_gp_controller.py"):
            assert protocol["source_sha256"][str(root / source)] == digest(root / source)
        assert (output / "calibration/grt/resources.json").exists()
        assert (output / "routed/grt/receipt.json").exists()
        assert reports[mode]["legal"] and reports[mode]["fixed_unchanged"]
        for oracle in reports[mode]["oracle"]:
            assert oracle["live_position_before"] == oracle["live_position_after"]
            assert oracle["db_arrays_unchanged"] and oracle["export_database_isolated"]
            assert oracle["legal"] and oracle["fixed_unchanged"]
            assert Path(oracle["snapshot"]).exists()
            assert oracle["snapshot_sha256"] == digest(oracle["snapshot"])
            repair = oracle["row_legalization"]
            assert repair["immutable_identity_unchanged"]
            assert Path(repair["receipt_path"]).exists()
            for source, expected_hash in repair["inputs"].items():
                assert digest(source) == expected_hash
            directory = Path(oracle["snapshot"]).parent
            assert (directory / "dreamplace.def").exists()
            assert (directory / "dreamplace.npz").exists()
            assert digest(directory / "out.def") == oracle["placement_sha256"]
            coord = json.loads((directory / "coord.json").read_text())
            with np.load(directory / "row_legalization/coordinates.npz") as repaired, np.load(oracle["snapshot"]) as actual:
                count = len(repaired["x_dbu"])
                np.testing.assert_allclose(actual["node_x"][:count],
                    (repaired["x_dbu"] - coord["shift_factor"][0]) * coord["scale_factor"], rtol=1e-7, atol=1e-5)
                np.testing.assert_allclose(actual["node_y"][:count],
                    (repaired["y_dbu"] - coord["shift_factor"][1]) * coord["scale_factor"], rtol=1e-7, atol=1e-5)
        with np.load(output / "legal.npz") as final, np.load(output / "routed/placement.npz") as routed:
            np.testing.assert_array_equal(final["full_pos"], routed["full_pos"])

    wa, joint = reports["wa"], reports["joint"]
    assert joint["oracle"][0]["actual_io"] == wa["oracle"][0]["actual_io"]
    baseline_repair = wa["oracle"][0]["row_legalization"]
    if not baseline_repair["changed_location_count"] and not baseline_repair["changed_orientation_count"]:
        assert wa["oracle"][0]["actual_io"] == 414
    assert wa["initial_sha256"] == joint["initial_sha256"]
    assert [r["position_sha256"] for r in wa["trace"][:5]] == [r["position_sha256"] for r in joint["trace"][:5]]
    assert joint["gp_sha256"] != wa["gp_sha256"]
    active = [r for r in joint["trace"] if r["active"]]
    assert active and any(r["route_gradient_l1"] > 0 and r["route_lambda"] > 0 for r in active)
    assert any(r["io_gradient_l1"] > 0 for r in active)
    assert any(r["congestion_gradient_l1"] > 0 for r in active)
    assert any(r["wirelength_gradient_l1"] > 0 for r in active)
    assert all(r["forward_generation_min"] == r["forward_generation_max"] == r["generation"] == r["cache_generation"] for r in active)
    audit = joint["gradient_audit"]
    assert audit["added_l1"] > 0 and audit["gradient_sum_max_error"] <= audit["tolerance"]
    online = [r for r in joint["oracle"] if r.get("observation_version")]
    assert len(online) >= 2
    for measured in online:
        later = [r for r in active if r["iteration"] > measured["iteration"]
                 and r["published_observation_version"] >= measured["observation_version"]]
        assert later, "measured router feedback was never consumed by a later GP step"
        assert measured["measured_position_sha256"] != measured["live_position_before"]
    with np.load(tmp_path / "joint/legal.npz") as checkpoint:
        assert np.isfinite(checkpoint["node_x"]).all()


def test_joint_cli_requires_router_before_creating_output(tmp_path):
    root = Path(REPO_ROOT)
    result = subprocess.run([sys.executable, str(root / "src/scripts/run_route_gp.py"),
        "--config", "unused.json", "--out", str(tmp_path / "missing"), "--mode", "joint"],
        capture_output=True, text=True,
        env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"})
    assert result.returncode != 0 and "--openroad" in result.stderr
    assert not (tmp_path / "missing").exists()


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["wa_standard", "paper", "joint"])
def test_gp_modes_and_self_generated_legal_calibration(tmp_path, mode):
    root = Path(REPO_ROOT)
    config = root / "results/route_feedback_20260914/gcd.json"
    binary = os.environ.get("OPENROAD_BIN")
    if not config.exists() or (mode in ("wa_standard", "joint") and (not binary or not Path(binary).exists())):
        pytest.skip("GCD/OpenROAD runtime required")
    output = tmp_path / mode
    command = [sys.executable, str(root / "src/scripts/run_route_gp.py"),
        "--config", str(config), "--out", str(output), "--mode", mode,
        "--iterations", "8", "--start", "2", "--rebuild", "3", "--hot-nets", "0",
        "--stop-overflow", "0", "--router-every", "4"]
    if mode in ("wa_standard", "joint"):
        command += ["--openroad", binary]
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"})
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout[-1200:]
    report = json.loads((output / "result.json").read_text())
    assert report["legal"] and report["fixed_unchanged"] and report["iterations"] == 8
    if mode == "wa_standard":
        assert report["cache_refreshes"] == report["topology_rebuilds"] == 0
        assert all(not row["active"] for row in report["trace"])
        assert not (output / "calibration").exists() and [row["label"] for row in report["oracle"]] == ["routed"]
        assert report["oracle"][0]["row_legalization"]["immutable_identity_unchanged"]
        for source, expected_hash in report["oracle"][0]["row_legalization"]["inputs"].items():
            assert digest(source) == expected_hash
        assert (output / "routed/grt/receipt.json").exists()
    else:
        assert report["cache_refreshes"] == 6 and report["topology_rebuilds"] >= 2
        assert report["gradient_audit"]["added_l1"] > 0
    if mode == "joint":
        assert report["oracle"][0]["label"] == "calibration"
        assert report["oracle"][0]["legal"]
        assert any(row["published_observation_version"] == 2 for row in report["trace"])


def test_publication_report_leaves_early_stop_observation_pending(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    from scripts.run_route_gp import observation_publication
    observations = [dict(observation_version=1, iteration=-1),
                    dict(observation_version=2, iteration=7),
                    dict(observation_version=3, iteration=15)]
    trace = [dict(iteration=5, active=True, published_observation_version=1, generation=1),
             dict(iteration=8, active=True, published_observation_version=2, generation=2),
             dict(iteration=15, active=True, published_observation_version=2, generation=3)]
    result = observation_publication(observations, trace)
    assert result["consumed_observation_versions"] == [1, 2]
    assert result["pending_observation_versions"] == [3]
    assert result["publication"][1]["first_consuming_iteration"] == 8


@pytest.mark.slow
def test_saved_tile_row_gap_checkpoint_repairs_without_running_gp(tmp_path, monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    import torch
    from ioplace.drivers.run_placement import _load_dreamplace
    from scripts.run_route_gp import repair_routed_snapshot
    root = Path(REPO_ROOT)
    checkpoint = root / "results/route_gp_20260914/mempool_tile_wrap/wa_standard"
    config = root / "results/route_gp_20260914/mempool_tile_wrap.json"
    binary = os.environ.get("OPENROAD_BIN")
    if not config.exists() or not (checkpoint / "legal.npz").exists() or not binary:
        pytest.skip("saved tile checkpoint and OpenROAD runtime required")
    params, db = _load_dreamplace(str(config))
    db.initialize(params)
    torch.set_num_threads(params.num_threads)
    import NonLinearPlace
    placer = NonLinearPlace.NonLinearPlace(params, db, None)
    with np.load(checkpoint / "legal.npz") as data:
        snapshot = torch.as_tensor(data["full_pos"], device=placer.pos[0].device).clone()
    original = snapshot.clone()
    assert bool(placer.op_collections.legality_check_op(snapshot))
    directory = tmp_path / "tile_repair"
    directory.mkdir()
    for name in ("out.def", "coord.json", "regions.json", "netmap.json"):
        shutil.copy2(checkpoint / "routed" / name, directory / name)
    names = [v.decode() if isinstance(v, bytes) else str(v) for v in db.node_names[:db.num_movable_nodes]]
    settings = json.loads(config.read_text())
    repaired, receipt, orientations = repair_routed_snapshot(directory, snapshot,
        num_nodes=db.num_nodes, movable_names=names, lefs=settings["lef_input"], binary=binary,
        legality_check=placer.op_collections.legality_check_op)
    assert receipt["immutable_identity_unchanged"] and receipt["changed_location_count"] > 0
    assert receipt["before_check_code"] != 0
    assert len(orientations) == db.num_movable_nodes
    assert bool(placer.op_collections.legality_check_op(repaired))
    torch.testing.assert_close(snapshot, original, rtol=0, atol=0)
    nm, nall = db.num_movable_nodes, db.num_nodes
    torch.testing.assert_close(repaired[nm:nall], original[nm:nall], rtol=0, atol=0)
    torch.testing.assert_close(repaired[nall+nm:], original[nall+nm:], rtol=0, atol=0)
    assert digest(directory / "out.def") != digest(directory / "dreamplace.def")
