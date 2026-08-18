"""T11 report linter (`scripts/m4_report_lint.py`) tests. Design draft sec
7.1 T11 row's acceptance is literal: "linter 在 pytest 中對故意違規的樣本
表格必須報錯" -- start from one known-good report + results-root pair and
break exactly one thing per test, matching `result_gate.py`'s own testing
convention. Fixtures are built entirely from `tmp_path` (small JSON +
small markdown); no real `results/**.json` is read."""
import json

import pytest

from scripts.m4_report_lint import (
    MANDATORY_DISCLOSURE_SENTENCE, lint_report, main,
)


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _write_json(results_root, rel_path, **fields):
    p = results_root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fields))
    return p


def _real_record(run_id, **overrides):
    r = {
        "run_id": run_id, "repo_commit": "abc123", "dp_commit": "def456",
        "command": "pytest", "hostname": "host1", "gpu_name": "NVIDIA L4",
        "benchmark_kind": "real", "experiment_status": "ok",
        "workload_status": "completed",
    }
    r.update(overrides)
    return r


def _synthetic_record(run_id, generator_verified=False, **overrides):
    r = {
        "run_id": run_id, "repo_commit": "abc123", "dp_commit": "def456",
        "command": "pytest", "hostname": "host1", "gpu_name": "NVIDIA L4",
        "benchmark_kind": "synthetic", "experiment_status": "ok",
        "workload_status": "completed",
        "generator_verified": generator_verified, "t6_verdict": "FAIL",
        "normalization": "n2",
    }
    r.update(overrides)
    return r


QUALITY_HEADER = "| case | K | hpwl | Δio% | run_id |"
QUALITY_SEP =    "|---|---|---|---|---|"
SCALING_HEADER = "| case | K | nets | t_total | run_id |"
SCALING_SEP =    "|---|---|---|---|---|"


def _quality_report(rows, disclosure=False):
    lines = ["# M4 report", "", "## quality_real_cases", "", QUALITY_HEADER, QUALITY_SEP]
    lines.extend(rows)
    if disclosure:
        lines += ["", MANDATORY_DISCLOSURE_SENTENCE]
    return "\n".join(lines) + "\n"


def _scaling_report(rows, disclosure=False):
    lines = ["# M4 report", "", "## scaling_synthetic_cases", "", SCALING_HEADER, SCALING_SEP]
    lines.extend(rows)
    if disclosure:
        lines += ["", MANDATORY_DISCLOSURE_SENTENCE]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# clean samples -- must pass with zero errors
# ---------------------------------------------------------------------------

def test_clean_quality_table_passes(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real1.json", **_real_record("run-real-1"))
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-real-1 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert errors == [], errors


def test_clean_scaling_table_with_generator_verified_true_passes(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn1.json",
                **_synthetic_record("run-syn-1", generator_verified=True))
    report = tmp_path / "report.md"
    report.write_text(_scaling_report(["| synth6.2M | 16 | 1.0e7 | 120.0 | run-syn-1 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert errors == [], errors


def test_clean_scaling_table_with_generator_verified_false_and_disclosure_passes(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn2.json",
                **_synthetic_record("run-syn-2", generator_verified=False))
    report = tmp_path / "report.md"
    report.write_text(_scaling_report(
        ["| synth12.3M | 16 | 2.0e7 | 240.0 | run-syn-2 |"], disclosure=True))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert errors == [], errors


def test_clean_report_with_inline_json_reference_instead_of_run_id_column(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real2.json", **_real_record("run-real-2"))
    report = tmp_path / "report.md"
    lines = ["# M4 report", "", "## quality_real_cases", "",
            "| case | K | hpwl | ref |", "|---|---|---|---|",
            "| bigblue4 | 32 | 1.2e6 | results/m4/tables/real2.json |"]
    report.write_text("\n".join(lines) + "\n")

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert errors == [], errors


def test_unrelated_table_not_classified_and_ignored(tmp_path):
    results_root = tmp_path / "results"
    report = tmp_path / "report.md"
    lines = ["# M4 report", "", "## Some other table", "",
            "| foo | bar |", "|---|---|", "| 1 | 2 |"]
    report.write_text("\n".join(lines) + "\n")

    findings = lint_report(report, results_root)
    assert findings == []


# ---------------------------------------------------------------------------
# rule 1: table parsing -- row missing run_id / json reference
# ---------------------------------------------------------------------------

def test_row_missing_run_id_and_json_reference_errors(tmp_path):
    results_root = tmp_path / "results"
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "table_parsing" for f in errors), findings


def test_row_run_id_not_resolvable_errors(tmp_path):
    results_root = tmp_path / "results"
    (results_root / "m4").mkdir(parents=True)
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-does-not-exist |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "table_parsing" for f in errors), findings


# ---------------------------------------------------------------------------
# rule 2: synthetic isolation
# ---------------------------------------------------------------------------

def test_synthetic_run_in_quality_table_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn3.json",
                **_synthetic_record("run-syn-3", generator_verified=True))
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| synth6.2M | 16 | 1.0e7 | -3.1 | run-syn-3 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "synthetic_isolation" for f in errors), findings


def test_scaling_table_with_quality_column_in_header_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn4.json",
                **_synthetic_record("run-syn-4", generator_verified=True))
    report = tmp_path / "report.md"
    lines = ["# M4 report", "", "## scaling_synthetic_cases", "",
            "| case | K | nets | Δhpwl% | run_id |", "|---|---|---|---|---|",
            "| synth6.2M | 16 | 1.0e7 | -3.1 | run-syn-4 |"]
    report.write_text("\n".join(lines) + "\n")

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "synthetic_isolation" for f in errors), findings


# ---------------------------------------------------------------------------
# rule 3: generator_verified
# ---------------------------------------------------------------------------

def test_scaling_row_missing_generator_verified_field_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn5.json", **_real_record("run-syn-5",
                benchmark_kind="synthetic"))  # no generator_verified key at all
    report = tmp_path / "report.md"
    report.write_text(_scaling_report(["| synth6.2M | 16 | 1.0e7 | 120.0 | run-syn-5 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "generator_verified" for f in errors), findings


def test_generator_verified_false_without_disclosure_sentence_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn6.json",
                **_synthetic_record("run-syn-6", generator_verified=False))
    report = tmp_path / "report.md"
    # no disclosure=True -- mandatory sentence absent.
    report.write_text(_scaling_report(["| synth12.3M | 16 | 2.0e7 | 240.0 | run-syn-6 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "generator_verified" for f in errors), findings


# ---------------------------------------------------------------------------
# rule 4: run_id uniqueness
# ---------------------------------------------------------------------------

def test_duplicate_run_id_in_same_table_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real3.json", **_real_record("run-dup"))
    report = tmp_path / "report.md"
    report.write_text(_quality_report([
        "| bigblue4 | 16 | 1.2e6 | -3.1 | run-dup |",
        "| bigblue4 | 32 | 1.3e6 | -2.9 | run-dup |",
    ]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "run_id_uniqueness" for f in errors), findings


def test_same_run_id_in_different_tables_does_not_error(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real4.json", **_real_record("run-shared"))
    lines = ["# M4 report", "", "## quality_real_cases", "", QUALITY_HEADER, QUALITY_SEP,
            "| bigblue4 | 16 | 1.2e6 | -3.1 | run-shared |",
            "", "## quality_real_cases (appendix, seed B)", "", QUALITY_HEADER, QUALITY_SEP,
            "| bigblue4 | 16 | 1.2e6 | -3.1 | run-shared |"]
    report = tmp_path / "report.md"
    report.write_text("\n".join(lines) + "\n")

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error" and f.rule == "run_id_uniqueness"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# rule 5: RESULT GATE spot check
# ---------------------------------------------------------------------------

def test_experiment_status_not_ok_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real5.json",
                **_real_record("run-bad-status", experiment_status="contaminated"))
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-bad-status |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "result_gate" for f in errors), findings


def test_status_field_name_variant_also_checked(tmp_path):
    results_root = tmp_path / "results"
    rec = _real_record("run-bad-status2")
    del rec["experiment_status"]
    rec["status"] = "crashed"
    _write_json(results_root, "m4/tables/real6.json", **rec)
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-bad-status2 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "result_gate" for f in errors), findings


def test_quality_row_requires_workload_status_completed(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real7.json",
                **_real_record("run-not-completed", workload_status="oom"))
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-not-completed |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "result_gate" for f in errors), findings


def test_scaling_row_workload_status_present_but_not_completed_errors(tmp_path):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn7.json",
                **_synthetic_record("run-syn-7", generator_verified=True,
                                    workload_status="crashed"))
    report = tmp_path / "report.md"
    report.write_text(_scaling_report(["| synth6.2M | 16 | 1.0e7 | 120.0 | run-syn-7 |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    assert any(f.rule == "result_gate" for f in errors), findings


def test_missing_provenance_is_warning_not_error(tmp_path):
    results_root = tmp_path / "results"
    rec = _real_record("run-legacy")
    rec["repo_commit"] = None
    rec["hostname"] = None
    _write_json(results_root, "m4/tables/legacy1.json", **rec)
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-legacy |"]))

    findings = lint_report(report, results_root)
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    assert errors == [], errors
    assert any(f.rule == "provenance" for f in warnings), findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_exits_0_on_clean_report(tmp_path, capsys):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/real8.json", **_real_record("run-cli-clean"))
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-cli-clean |"]))

    rc = main([str(report), "--results-root", str(results_root)])
    assert rc == 0


def test_cli_exits_1_on_violation(tmp_path, capsys):
    results_root = tmp_path / "results"
    _write_json(results_root, "m4/tables/syn8.json",
                **_synthetic_record("run-syn-8", generator_verified=True))
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| synth6.2M | 16 | 1.0e7 | -3.1 | run-syn-8 |"]))

    rc = main([str(report), "--results-root", str(results_root)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "synthetic_isolation" in out


def test_cli_strict_fails_on_warning_only(tmp_path, capsys):
    results_root = tmp_path / "results"
    rec = _real_record("run-cli-warn")
    rec["command"] = None
    _write_json(results_root, "m4/tables/real9.json", **rec)
    report = tmp_path / "report.md"
    report.write_text(_quality_report(["| bigblue4 | 16 | 1.2e6 | -3.1 | run-cli-warn |"]))

    rc_default = main([str(report), "--results-root", str(results_root)])
    assert rc_default == 0

    rc_strict = main([str(report), "--results-root", str(results_root), "--strict"])
    assert rc_strict == 1
