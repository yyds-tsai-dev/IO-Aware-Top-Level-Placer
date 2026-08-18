"""Backfill script (`scripts/m4_backfill_result_gate_fields.py`) tests.

The script edits already-published experiment artifacts, so the properties
worth pinning are the *restrictions*, not the happy path: it adds only the
one missing field plus its note, it refuses to write `workload_status`
without positive evidence that the run finished, and it never overwrites a
value that is already there. Fixtures follow `test_m4_report_lint.py`'s
convention -- small JSON + small markdown built entirely in `tmp_path`, no
real `results/**.json` is read.
"""
import json

from scripts.m4_backfill_result_gate_fields import collect_targets, main, plan_one
from tests.test_m4_report_lint import (
    _quality_report, _scaling_report, _real_record, _synthetic_record, _write_json,
)

QUALITY_ROW = "| bigblue4 | 16 | 1.2e6 | -3.1 | run-real-1 |"
SCALING_ROW = "| synth6.2M | 16 | 1.0e7 | 120.0 | run-syn-1 |"

# what a finished placement run carries once the driver has written its last fields
_COMPLETED = {
    "legalization_status": "success", "num_unplaced_cells": 0,
    "final_overflow": 0.069, "gp_iterations_run": 1251, "hpwl_lg": 1.2e6,
    "runtime_s": 6840.0,
}


def _quality_case(tmp_path, **record_overrides):
    results_root = tmp_path / "results"
    record = _real_record("run-real-1", **record_overrides)
    record.pop("workload_status")
    path = _write_json(results_root, "m4/tables/real1.json", **record)
    report = tmp_path / "report.md"
    report.write_text(_quality_report([QUALITY_ROW]))
    return report, results_root, path


def _scaling_case(tmp_path):
    results_root = tmp_path / "results"
    record = _synthetic_record("run-syn-1")
    record.pop("generator_verified")
    path = _write_json(results_root, "m4/tables/syn1.json", **record)
    report = tmp_path / "report.md"
    report.write_text(_scaling_report([SCALING_ROW], disclosure=True))
    return report, results_root, path


def _run(report, results_root, tmp_path, apply=True):
    argv = [str(report), "--results-root", str(results_root),
            "--audit-log", str(tmp_path / "audit.json")]
    if apply:
        argv.append("--apply")
    return main(argv)


# ---------------------------------------------------------------------------
# target resolution -- the script may only reach what the linter cites
# ---------------------------------------------------------------------------

def test_targets_come_from_the_linters_own_table_resolution(tmp_path):
    report, results_root, path = _quality_case(tmp_path, **_COMPLETED)
    _write_json(results_root, "m4/tables/uncited.json", **_real_record("run-uncited"))

    targets = collect_targets(report, results_root)

    assert [(kind, p.name) for kind, _line, p in targets] == [("quality", "real1.json")]
    assert path.exists()


# ---------------------------------------------------------------------------
# additive-only behaviour
# ---------------------------------------------------------------------------

def test_quality_row_with_completion_evidence_gets_workload_status(tmp_path):
    report, results_root, path = _quality_case(tmp_path, **_COMPLETED)
    before = json.loads(path.read_text())

    assert _run(report, results_root, tmp_path) == 0

    after = json.loads(path.read_text())
    assert after["workload_status"] == "completed"
    assert set(after) - set(before) == {"workload_status", "schema_backfill_note"}
    assert all(after[k] == v for k, v in before.items()), "existing values must be untouched"
    assert after["schema_backfill_note"]["added_fields"] == ["workload_status"]


def test_evaluate_only_row_is_completed_on_its_evaluated_quantities(tmp_path):
    report, results_root, path = _quality_case(
        tmp_path, evaluate_only=True, hpwl=1.2e6, io_count=62055,
        ft_count=3284, runtime_s=266.9)

    assert _run(report, results_root, tmp_path) == 0
    assert json.loads(path.read_text())["workload_status"] == "completed"


def test_scaling_row_gets_generator_verified_false(tmp_path):
    report, results_root, path = _scaling_case(tmp_path)

    assert _run(report, results_root, tmp_path) == 0
    assert json.loads(path.read_text())["generator_verified"] is False


def test_audit_log_records_sha256_before_and_after(tmp_path):
    report, results_root, path = _quality_case(tmp_path, **_COMPLETED)

    assert _run(report, results_root, tmp_path) == 0

    log = json.loads((tmp_path / "audit.json").read_text())
    (rec,) = log["records"]
    assert rec["action"] == "add" and rec["field"] == "workload_status"
    assert rec["sha256_before"] != rec["sha256_after"]
    assert sorted(rec["keys_added"]) == ["schema_backfill_note", "workload_status"]


# ---------------------------------------------------------------------------
# the restrictions
# ---------------------------------------------------------------------------

def test_refuses_a_quality_row_with_no_completion_evidence(tmp_path):
    report, results_root, path = _quality_case(tmp_path)  # no end-of-run fields
    before = path.read_text()

    assert _run(report, results_root, tmp_path) == 1
    assert path.read_text() == before, "a refused run must write nothing"
    assert not (tmp_path / "audit.json").exists()


def test_refuses_when_legalization_did_not_succeed(tmp_path):
    report, results_root, path = _quality_case(
        tmp_path, **{**_COMPLETED, "legalization_status": "failed"})

    assert _run(report, results_root, tmp_path) == 1
    assert "workload_status" not in json.loads(path.read_text())


def test_refuses_when_status_is_not_ok(tmp_path):
    report, results_root, path = _quality_case(
        tmp_path, **{**_COMPLETED, "experiment_status": "contaminated"})

    assert _run(report, results_root, tmp_path) == 1
    assert "workload_status" not in json.loads(path.read_text())


def test_never_overwrites_an_existing_value(tmp_path):
    results_root = tmp_path / "results"
    path = _write_json(results_root, "m4/tables/real1.json",
                       **_real_record("run-real-1", workload_status="oom", **_COMPLETED))
    report = tmp_path / "report.md"
    report.write_text(_quality_report([QUALITY_ROW]))

    action, field, value, _why = plan_one("quality", path)

    assert (action, field, value) == ("skip", "workload_status", "oom")
    assert _run(report, results_root, tmp_path) == 0
    assert json.loads(path.read_text())["workload_status"] == "oom"


def test_dry_run_writes_nothing(tmp_path):
    report, results_root, path = _quality_case(tmp_path, **_COMPLETED)
    before = path.read_text()

    assert _run(report, results_root, tmp_path, apply=False) == 0
    assert path.read_text() == before
    assert not (tmp_path / "audit.json").exists()
