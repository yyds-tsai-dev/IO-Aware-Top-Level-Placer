#!/usr/bin/env python3
"""M4 T11 report linter (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.0
"RESULT GATE" / sec 7.1's T11 row / sec 6.2's schema; T6 holdout
adjudication `docs/results/2026-08-15-m4-t6-holdout-adjudication.md`
sec 6.2's `generator_verified` rule and sec 7-1's mandatory disclosure
sentence).

Lints a rendered M4 report (`docs/results/m4-scale-up-report.md`, or any
markdown file following the same table conventions) plus a `results/`
root, so that "the report cites a number" and "the number came from a
RESULT-GATE-valid, correctly-classified run" stop being conflated. Six
rules, matching T11's acceptance ("linter 在 pytest 中對故意違規的樣本表格
必須報錯"):

  1. table parsing -- a markdown table is classified "quality" (caption/
     heading mentions `quality_real_cases`, or its header row carries a
     quality column such as hpwl/io_count/ft_count) or "scaling" (caption/
     heading mentions `scaling_synthetic_cases`); tables matching neither
     are left alone. Every row of a classified table must cite a `run_id`
     (resolved against --results-root by scanning every *.json under it
     for a matching `run_id` field) or an inline `results/**.json`
     reference (resolved by stripping the literal `results/` prefix and
     joining onto --results-root -- design draft sec 6.2's paths are
     always written relative to the results root, not the filesystem).
  2. synthetic isolation -- a row in a quality table whose resolved JSON
     has `benchmark_kind=="synthetic"` is an error; a scaling table's
     header may not carry a quality column (schema ban, design draft
     sec 6.2: "scaling_synthetic_cases.md ... 禁止 Δio%/Δft%/Δhpwl%/
     winner/pareto*").
  3. generator_verified -- every scaling-table row's resolved JSON must
     have a `generator_verified` field; when it is `false`, the report's
     full text must contain the adjudication sec 7-1 mandatory sentence
     verbatim, or it's an error.
  4. run_id uniqueness -- no table may cite the same run_id (or the same
     JSON reference, when a row has no run_id column) twice.
  5. RESULT GATE spot check -- for each resolved JSON: `experiment_status`
     (or, failing that, `status`) must be "ok" if the field is present;
     `workload_status` must be "completed" if present, and is *required*
     to be "completed" outright for quality-table rows (design draft
     sec 7.0 rule 1: only `experiment_status=="ok" and workload_status==
     "completed"` runs may enter a quality/scaling table); a resolved
     JSON missing any of run_id/repo_commit/command/hostname is a
     warning, not an error (old calibration runs are allowed to lack full
     provenance).

CLI:
    m4_report_lint.py REPORT.md --results-root results/ [--strict]

Exit 0 with no errors (or, under --strict, no errors and no warnings);
exit 1 otherwise. Findings print one per line, `path:line: LEVEL [rule]
message`.
"""
import argparse
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

# Holdout adjudication sec 7-1's mandatory sentence, verbatim -- copied
# character-for-character (ASCII comma, full-width 。) from the source doc
# so this never drifts from what's actually required in the report.
MANDATORY_DISCLOSURE_SENTENCE = "30M 的連通性是擬合出來的,不是校準出來的。"

# design draft sec 6.2: quality_real_cases.md's quality columns, banned
# from scaling_synthetic_cases.md's schema. Lower-cased for matching
# (Δ upper-cases to Δ; str.lower() maps it to δ, handled below).
QUALITY_COLUMN_MARKERS = {
    "hpwl", "io_count", "ft_count", "io_mst", "ft_mst", "io_rg", "ft_rg",
    "δio%", "δft%", "δhpwl%", "winner", "pareto",
}

QUALITY_CAPTION_MARKER = "quality_real_cases"
SCALING_CAPTION_MARKER = "scaling_synthetic_cases"

_PROVENANCE_FIELDS = ("run_id", "repo_commit", "command", "hostname")

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_CELL_RE = re.compile(r"^:?-+:?$")
_JSON_REF_RE = re.compile(r"results/[\w./\-]+?\.json")

Finding = namedtuple("Finding", ["line", "level", "rule", "message"])


# ---------------------------------------------------------------------------
# markdown table parsing
# ---------------------------------------------------------------------------

def _split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def find_tables(lines):
    """Returns a list of {"header_line": 1-based line no of the header
    row, "header": [cell, ...], "rows": [(1-based line no, [cell, ...])]}
    for every markdown table (header row + a `---`/`:--:` separator row +
    >=0 data rows) in `lines`."""
    tables = []
    i = 0
    n = len(lines)
    while i < n:
        if _TABLE_ROW_RE.match(lines[i]) and i + 1 < n:
            sep_cells = _split_row(lines[i + 1])
            if sep_cells and all(_SEP_CELL_RE.match(c.strip()) for c in sep_cells):
                header = _split_row(lines[i])
                header_line = i + 1
                j = i + 2
                rows = []
                while j < n and _TABLE_ROW_RE.match(lines[j]):
                    rows.append((j + 1, _split_row(lines[j])))
                    j += 1
                tables.append({"header_line": header_line, "header": header, "rows": rows})
                i = j
                continue
        i += 1
    return tables


def _nearby_caption(lines, table_start_idx, window=6):
    """Text used to classify a table: the nearest heading line above it,
    plus the `window` lines immediately preceding it (captions/labels are
    conventionally written as a heading or a line right before the
    table)."""
    texts = []
    for k in range(table_start_idx - 1, -1, -1):
        if lines[k].lstrip().startswith("#"):
            texts.append(lines[k])
            break
    start = max(0, table_start_idx - window)
    texts.extend(lines[start:table_start_idx])
    return "\n".join(texts)


def classify_table(caption, header):
    """"quality" / "scaling" / None, per rule 1's caption-or-columns
    heuristic."""
    cap = caption.lower()
    if QUALITY_CAPTION_MARKER in cap:
        return "quality"
    if SCALING_CAPTION_MARKER in cap:
        return "scaling"
    header_norm = {h.strip().lower() for h in header}
    if header_norm & QUALITY_COLUMN_MARKERS:
        return "quality"
    return None


# ---------------------------------------------------------------------------
# results-root resolution
# ---------------------------------------------------------------------------

def build_results_index(results_root):
    """Scans every *.json under `results_root` and indexes it by its own
    `run_id` field, for rows that cite a bare run_id with no inline JSON
    path."""
    by_run_id = {}
    for p in sorted(Path(results_root).rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rid = data.get("run_id") if isinstance(data, dict) else None
        if rid:
            by_run_id[rid] = (p, data)
    return by_run_id


def _strip_inline_md(cell):
    s = cell.strip()
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()
    return s


def resolve_row(cells, header, results_root, by_run_id):
    """Returns (run_id_val, json_ref, resolved_path, data) for one table
    row: `run_id_val` is the row's `run_id` column value (if the table has
    one); `json_ref` is a `results/**.json` reference found in any cell;
    `data` is the loaded JSON (None if neither resolves to a real file)."""
    header_lower = [h.strip().lower() for h in header]
    run_id_val = None
    if "run_id" in header_lower:
        idx = header_lower.index("run_id")
        if idx < len(cells):
            run_id_val = _strip_inline_md(cells[idx])
            if run_id_val in ("", "-", "—"):
                run_id_val = None

    json_ref = None
    for c in cells:
        m = _JSON_REF_RE.search(c)
        if m:
            json_ref = m.group(0)
            break

    path = None
    data = None
    if json_ref:
        rel = json_ref[len("results/"):] if json_ref.startswith("results/") else json_ref
        path = Path(results_root) / rel
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
    elif run_id_val:
        hit = by_run_id.get(run_id_val)
        if hit:
            path, data = hit

    return run_id_val, json_ref, path, data


# ---------------------------------------------------------------------------
# lint rules
# ---------------------------------------------------------------------------

def lint_report(report_path, results_root, strict=False):
    """Runs all 5 rules against `report_path`, returning a list of
    `Finding`s (both errors and warnings; caller decides the exit code)."""
    report_path = Path(report_path)
    text = report_path.read_text()
    lines = text.splitlines()
    tables = find_tables(lines)
    by_run_id = build_results_index(results_root)

    findings = []

    for t in tables:
        header = t["header"]
        header_start_idx = t["header_line"] - 1
        caption = _nearby_caption(lines, header_start_idx)
        kind = classify_table(caption, header)
        if kind is None:
            continue

        if kind == "scaling":
            header_norm = {h.strip().lower() for h in header}
            bad_cols = header_norm & QUALITY_COLUMN_MARKERS
            if bad_cols:
                findings.append(Finding(
                    t["header_line"], "error", "synthetic_isolation",
                    "scaling table header carries prohibited quality "
                    f"column(s): {sorted(bad_cols)}"))

        seen = {}
        for line_no, cells in t["rows"]:
            run_id_val, json_ref, path, data = resolve_row(cells, header, results_root, by_run_id)

            if not run_id_val and not json_ref:
                findings.append(Finding(
                    line_no, "error", "table_parsing",
                    "row has neither a run_id nor a results/**.json reference"))
                continue

            ident = run_id_val or json_ref
            if ident in seen:
                findings.append(Finding(
                    line_no, "error", "run_id_uniqueness",
                    f"duplicate run_id/reference {ident!r} (first seen at line {seen[ident]})"))
            else:
                seen[ident] = line_no

            if data is None:
                findings.append(Finding(
                    line_no, "error", "table_parsing",
                    f"could not resolve JSON for row (run_id={run_id_val!r}, "
                    f"ref={json_ref!r}) under results-root {results_root}"))
                continue

            # rule 2: synthetic isolation.
            bk = data.get("benchmark_kind")
            if kind == "quality" and bk == "synthetic":
                findings.append(Finding(
                    line_no, "error", "synthetic_isolation",
                    "quality table row references a synthetic run "
                    f"(benchmark_kind='synthetic', run_id={data.get('run_id')!r})"))

            # rule 3: generator_verified.
            if kind == "scaling":
                if "generator_verified" not in data:
                    findings.append(Finding(
                        line_no, "error", "generator_verified",
                        "scaling table row's JSON is missing 'generator_verified'"))
                elif data.get("generator_verified") is False:
                    if MANDATORY_DISCLOSURE_SENTENCE not in text:
                        findings.append(Finding(
                            line_no, "error", "generator_verified",
                            "generator_verified=false but the report text is "
                            "missing the mandatory disclosure sentence "
                            "(holdout adjudication sec 7-1)"))

            # rule 5: RESULT GATE spot check.
            status_val = data.get("experiment_status", data.get("status"))
            if status_val is not None and status_val != "ok":
                findings.append(Finding(
                    line_no, "error", "result_gate",
                    f"experiment_status/status != 'ok' (got {status_val!r})"))

            ws = data.get("workload_status")
            if kind == "quality":
                if ws != "completed":
                    findings.append(Finding(
                        line_no, "error", "result_gate",
                        f"quality table row requires workload_status=='completed' "
                        f"(got {ws!r})"))
            else:
                if ws is not None and ws != "completed":
                    findings.append(Finding(
                        line_no, "error", "result_gate",
                        f"workload_status present but not 'completed' (got {ws!r})"))

            missing_prov = [f for f in _PROVENANCE_FIELDS if not data.get(f)]
            if missing_prov:
                findings.append(Finding(
                    line_no, "warning", "provenance",
                    f"missing provenance field(s) {missing_prov} "
                    "(allowed for legacy calibration runs)"))

    return findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="M4 T11 report linter (RESULT GATE / "
                                              "synthetic-isolation / generator_verified)")
    ap.add_argument("report", type=Path, help="path to the markdown report")
    ap.add_argument("--results-root", type=Path, required=True,
                    help="root directory to resolve results/**.json references against")
    ap.add_argument("--strict", action="store_true",
                    help="also fail (exit 1) on warnings, not just errors")
    args = ap.parse_args(argv)

    findings = lint_report(args.report, args.results_root, strict=args.strict)

    for f in sorted(findings, key=lambda f: f.line):
        print(f"{args.report}:{f.line}: {f.level.upper()} [{f.rule}] {f.message}")

    n_errors = sum(1 for f in findings if f.level == "error")
    n_warnings = sum(1 for f in findings if f.level == "warning")
    print(f"{args.report}: {n_errors} error(s), {n_warnings} warning(s)", file=sys.stderr)

    if n_errors or (args.strict and n_warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
