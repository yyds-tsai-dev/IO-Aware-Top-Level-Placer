import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import m4_forecast as forecast


REPO = Path(__file__).resolve().parents[1]
CLUSTER = REPO / "results/m4/t8b/cluster__k16__grid__flat.json"
COUNT = REPO / "results/m4/scaling/count_freeze_30m.json"
PROBE = REPO / "results/m4/probes/probe_gp_memory__mempool_cluster_probe2500.json"
CANONICAL = REPO / "results/m4/forecast/h100_prediction.json"


def _inputs():
    return (json.loads(CLUSTER.read_text()), json.loads(COUNT.read_text()),
            json.loads(PROBE.read_text()))


def _sku():
    return {
        "name": "NVIDIA H100 NVL",
        "sm_arch": "sm_90",
        "cuda_version": "12.8",
        "mig": False,
        "persistence_mode": "on",
        "clocks": "default",
        "measurement_protocol": "pre-registered host observation",
        "observed_fields": {"board": "NVL"},
    }


def test_default_is_legacy_h100_and_override_is_copied():
    cluster, count, probe = _inputs()
    override = _sku()
    prediction = forecast.build_prediction(
        cluster, count, probe,
        cluster_flat_path=CLUSTER, count_freeze_path=COUNT,
        probe_gp_path=PROBE, sku=override,
    )
    assert prediction["sku"]["name"] == "NVIDIA H100 NVL"
    assert prediction["sku"]["measurement_protocol"] == override["measurement_protocol"]
    assert prediction["sku"]["observed_fields"] == override["observed_fields"]
    assert prediction["sku"] is not override
    assert "source_sha256" not in prediction["sku"]
    legacy = forecast.build_prediction(
        cluster, count, probe,
        cluster_flat_path=CLUSTER, count_freeze_path=COUNT,
        probe_gp_path=PROBE,
    )
    assert legacy["sku"] == forecast.H100_SKU


@pytest.mark.parametrize("bad", [
    {},
    {"name": "", "sm_arch": "sm_90", "cuda_version": "12.8", "mig": False,
     "persistence_mode": "on", "clocks": "default"},
    {"name": "bad", "sm_arch": "sm_90", "cuda_version": "12.8", "mig": 0,
     "persistence_mode": "on", "clocks": "default"},
])
def test_invalid_sku_rejected(bad):
    with pytest.raises(ValueError):
        forecast._validate_sku(bad)


def test_cli_invalid_sku_rejects_without_output(tmp_path):
    sku = tmp_path / "bad.json"
    sku.write_text(json.dumps({"name": ""}))
    out = tmp_path / "invalid.json"
    result = subprocess.run(
        [sys.executable, "scripts/m4_forecast.py", "--sku-json", str(sku),
         "--out", str(out)], cwd=REPO, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert not out.exists()


def test_cli_alternate_frozen_output_records_sku_and_hash(tmp_path):
    before = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    sku = tmp_path / "nvl.json"
    sku.write_text(json.dumps(_sku(), indent=2))
    out = tmp_path / "frozen.json"
    result = subprocess.run(
        [sys.executable, "scripts/m4_forecast.py", "--sku-json", str(sku),
         "--out", str(out), "--freeze"], cwd=REPO, text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["status"] == "frozen"
    assert data["sku"]["name"] == "NVIDIA H100 NVL"
    assert data["sku"]["measurement_protocol"] == _sku()["measurement_protocol"]
    assert data["sku"]["source_sha256"] == hashlib.sha256(sku.read_bytes()).hexdigest()
    assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == before
