#!/usr/bin/env python3
"""Audited backfill of the two RESULT GATE fields that `run_placement.py`
never wrote (M4-G8 / T11; user ruling A, 2026-08-19).

Background. `scripts/m4_report_lint.py --strict` reports 24 errors against
`docs/results/m4-scale-up-report.md`: the 16 rows of the
`quality_real_cases` table resolve to JSONs with no `workload_status`, and
the 8 rows of the `scaling_synthetic_cases` table resolve to JSONs with no
`generator_verified`. Both fields were *defined after* those runs were
executed (design draft sec 7.0 rule 1, holdout adjudication sec 6.2), and
the drivers were never taught to emit them. The numbers themselves are
fine -- the schema is what's incomplete.

What this script does, and what it refuses to do.

  * It only ever *adds* the two missing fields. If a field is already
    present it is left alone (never overwritten), and the file is reported
    as `skipped`.
  * It never touches a measured value. Every other key is copied through
    byte-identically; the audit log records a sha256 of each file before
    and after, plus the exact key-set delta.
  * `workload_status="completed"` is only written when the JSON carries
    positive evidence that the run reached its end (see
    `_completion_evidence` -- a full placement run must have status ok
    plus the end-of-run fields the driver writes last; an evaluate-only
    run must have status ok plus its evaluated quantities). A file that
    cannot show that evidence is *refused*, not guessed at, and the script
    exits non-zero.
  * `generator_verified=false` is written unconditionally for scaling
    rows, because that is the factual state: no generator certification
    was ever performed (holdout adjudication sec 7-1 -- this is exactly
    the value that keeps the mandatory disclosure sentence obligatory).
  * Every file it writes gets a `schema_backfill_note` recording when,
    by what, under whose ruling, and on what evidence the field was added.

The target set is not hand-listed: it is resolved by importing
`m4_report_lint` and re-running its own table classification and row
resolution, so this script can only reach files the linter itself cites.

CLI:
    m4_backfill_result_gate_fields.py REPORT.md --results-root results/ \
        [--apply] [--audit-log PATH]

Default is a dry run. `--apply` writes the files and the audit log.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m4_report_lint as lint  # noqa: E402

RULING = ("user ruling A, 2026-08-19: approve an audited backfill of the "
          "post-hoc RESULT GATE fields")
NOTE_KEY = "schema_backfill_note"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _completion_evidence(data):
    """Returns (ok, evidence_or_reason). The fields checked are the ones
    the drivers write *after* the workload finishes, so their presence is
    what "the workload completed" means for these records."""
    status = data.get("experiment_status", data.get("status"))
    if status != "ok":
        return False, f"status={status!r} is not 'ok'"

    if data.get("evaluate_only") is True:
        needed = ("hpwl", "io_count", "ft_count", "runtime_s")
        missing = [k for k in needed if k not in data]
        if missing:
            return False, f"evaluate-only run missing end-of-run field(s) {missing}"
        return True, ("status=ok; evaluate_only run carries its evaluated "
                      "quantities (hpwl/io_count/ft_count/runtime_s)")

    needed = ("legalization_status", "num_unplaced_cells", "final_overflow",
              "gp_iterations_run", "hpwl_lg", "runtime_s")
    missing = [k for k in needed if k not in data]
    if missing:
        return False, f"placement run missing end-of-run field(s) {missing}"
    if data.get("legalization_status") != "success":
        return False, f"legalization_status={data.get('legalization_status')!r} is not 'success'"
    return True, ("status=ok; placement run carries the fields the driver "
                  "writes last (legalization_status=success, "
                  "num_unplaced_cells, final_overflow, gp_iterations_run, "
                  "hpwl_lg, runtime_s)")


def collect_targets(report_path, results_root):
    """Re-uses the linter's own parsing so the target set is exactly the
    set of JSONs the linter resolves from the report's classified tables.
    Returns [(kind, line_no, Path)], de-duplicated, in report order."""
    lines = Path(report_path).read_text().splitlines()
    by_run_id = lint.build_results_index(results_root)
    targets, seen = [], set()

    for t in lint.find_tables(lines):
        caption = lint._nearby_caption(lines, t["header_line"] - 1)
        kind = lint.classify_table(caption, t["header"])
        if kind is None:
            continue
        for line_no, cells in t["rows"]:
            _run_id, _ref, path, data = lint.resolve_row(
                cells, t["header"], results_root, by_run_id)
            if path is None or data is None:
                continue
            key = str(Path(path).resolve())
            if key in seen:
                continue
            seen.add(key)
            targets.append((kind, line_no, Path(path)))
    return targets


def plan_one(kind, path):
    """Returns (action, field, value, evidence_or_reason). action is one
    of 'add' / 'skip' / 'refuse'."""
    data = json.loads(path.read_text())

    if kind == "quality":
        field = "workload_status"
        if field in data:
            return "skip", field, data[field], "field already present; left untouched"
        ok, why = _completion_evidence(data)
        if not ok:
            return "refuse", field, None, why
        return "add", field, "completed", why

    field = "generator_verified"
    if field in data:
        return "skip", field, data[field], "field already present; left untouched"
    return "add", field, False, (
        "no generator certification was ever performed for the tiled "
        "synthetic arrays (holdout adjudication sec 7-1); false is the "
        "factual value and keeps the mandatory disclosure sentence required")


def apply_one(path, field, value, evidence, stamp):
    """Adds `field` and the backfill note, preserving every other key and
    its order. Returns (sha_before, sha_after, keys_added)."""
    sha_before = _sha256(path)
    data = json.loads(path.read_text())
    keys_added = [k for k in (field, NOTE_KEY) if k not in data]

    data[field] = value
    note = data.get(NOTE_KEY, {})
    note.update({
        "added_fields": sorted(set(note.get("added_fields", [])) | {field}),
        "added_utc": stamp,
        "added_by": "scripts/m4_backfill_result_gate_fields.py",
        "ruling": RULING,
        "reason": ("field was defined after this run executed and the "
                   "driver never emitted it"),
        "evidence": evidence,
        "invariant": "no measured value was read, recomputed or altered",
    })
    data[NOTE_KEY] = note

    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return sha_before, _sha256(path), keys_added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--results-root", default="results/")
    ap.add_argument("--apply", action="store_true",
                    help="write the files (default is a dry run)")
    ap.add_argument("--audit-log", default="results/m4/backfill/"
                    "2026-08-19-result-gate-backfill.json")
    args = ap.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    targets = collect_targets(args.report, args.results_root)

    records, refused = [], []
    for kind, line_no, path in targets:
        action, field, value, why = plan_one(kind, path)
        rec = {"table": kind, "report_line": line_no,
               "path": str(path), "action": action, "field": field,
               "value": value, "evidence": why}
        if action == "refuse":
            refused.append(rec)
        records.append(rec)
        print(f"{action.upper():7s} {path} :: {field}="
              f"{json.dumps(value, ensure_ascii=False)} -- {why}")

    n_add = sum(1 for r in records if r["action"] == "add")
    print(f"\n{len(targets)} resolved row(s): {n_add} to add, "
          f"{sum(1 for r in records if r['action'] == 'skip')} already present, "
          f"{len(refused)} refused")

    if refused:
        print("REFUSED -- evidence of completion is missing; nothing was "
              "written. Fix or exclude these before re-running.", file=sys.stderr)
        return 1

    if not args.apply:
        print("dry run -- pass --apply to write")
        return 0

    for rec in records:
        if rec["action"] != "add":
            continue
        before, after, added = apply_one(Path(rec["path"]), rec["field"],
                                         rec["value"], rec["evidence"], stamp)
        rec.update({"sha256_before": before, "sha256_after": after,
                    "keys_added": added})

    log_path = Path(args.audit_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "utc": stamp,
        "ruling": RULING,
        "report": str(args.report),
        "results_root": args.results_root,
        "script": "scripts/m4_backfill_result_gate_fields.py",
        "invariant": "only the named field plus schema_backfill_note were "
                     "added; no measured value was altered",
        "records": records,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"audit log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
