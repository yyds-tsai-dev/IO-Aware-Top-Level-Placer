"""Collect Task 7/9/12 result JSON files into a Markdown comparison table
(M0/M1 exit).

Consumes the flat result schema written by ``run_placement.run_flat`` /
``run_two_stage`` / ``run_reweight`` (Tasks 7/9/12): a flat dict with at
least ``mode``, ``config``, ``k``, ``rtype``, ``seed``, ``io_count``,
``ft_count``, ``tree_wl``, ``hpwl``, ``runtime_s``, ``peak_mem_mb``.
``run_reweight`` additionally sets ``num_reweights`` (Task 13); flat/
two_stage rows never set it, so ``_fmt(r.get(c, ""))`` renders that cell
blank for them rather than erroring or showing a placeholder. ``case`` is
derived here from ``config``'s basename (e.g. ``.../adaptec1.json`` ->
``adaptec1``) rather than stored in the JSON, so results from any config
path collect_results() reads normalize uniformly into the same field name
Task 10's brief/spec Step 2 asserts on.
"""
import argparse, glob, json, os

COLS = ["mode", "k", "rtype", "seed", "io_count", "ft_count",
        "tree_wl", "hpwl", "runtime_s", "peak_mem_mb", "num_reweights"]

def collect_results(d):
    """Read every ``*.json`` in ``d`` and tag each with a ``case`` field.

    Returns rows sorted by filename (glob is sorted for reproducible report
    ordering); each row is the parsed JSON dict plus ``case`` derived from
    ``os.path.basename(config).removesuffix(".json")``.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        r["case"] = os.path.basename(r["config"]).replace(".json", "")
        rows.append(r)
    return rows

def _fmt(v):
    """Thousands-separated formatting; bool is checked first since bool is an
    int subclass in Python and would otherwise render as ``1``/``0``."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return str(v)

def render_markdown(rows):
    """Render ``rows`` (as returned by collect_results) into one Markdown
    section per distinct ``case``, each with a table of COLS in the fixed
    order the brief's interface spec mandates."""
    out = ["# Baseline 對照表\n"]
    for case in sorted({r["case"] for r in rows}):
        out.append(f"\n## {case}\n")
        out.append("| " + " | ".join(["case"] + COLS) + " |")
        out.append("|" + "---|" * (len(COLS) + 1))
        for r in [x for x in rows if x["case"] == case]:
            out.append("| " + " | ".join([case] + [_fmt(r.get(c, "")) for c in COLS]) + " |")
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md = render_markdown(collect_results(a.dir))
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)

# ---------------------------------------------------------------------------
# Task 8 Step 3: M2 ablation/report table (design v2 sec 8.2)
# ---------------------------------------------------------------------------

# Column label -> JSON key, in the exact order of design v2 sec 8.2's header
# row. Every key here is looked up with a plain dict .get() against (a) the
# parsed run JSON, for keys the run schema actually writes (checked against
# results/m2/sweep/*.json and results/m2/noise/*.json -- both use the
# run_placement_io.run_io schema, including for their "flat" baselines, which
# T6/T7 generate via rho_max=0 observer-mode `io` runs, not `run_flat`), or
# (b) the row dict this module computes (the "case"/Δ% keys below). A label
# with no matching key anywhere (only "margin" -- the schema only ever has
# "rho_margin"/"margin_m", never a field literally named "margin") always
# renders "—"; this module never guesses a mapping for it.
M2_TABLE_SPEC = [
    ("case", "case"), ("mode", "mode"), ("k", "k"), ("rtype", "rtype"),
    ("det", "det"), ("dp_seed", "dp_seed"), ("io_count", "io_count"),
    ("io_gp", "io_gp"), ("ft_count", "ft_count"),
    ("hard_lambda_sum", "hard_lambda_sum"), ("tree_wl", "tree_wl"), ("hpwl", "hpwl"),
    ("Δio%", "d_io_pct"), ("Δio_gp%", "d_io_gp_pct"), ("Δhpwl%", "d_hpwl_pct"),
    ("lg_loss", "lg_loss"), ("runtime_s", "runtime_s"), ("peak_mem_mb", "peak_mem_mb"),
    ("rho_max", "rho_max"), ("tau_hi", "tau_hi"), ("tau_lo", "tau_lo"),
    ("alpha_io", "alpha_io"), ("w_mode", "w_mode"), ("d_max", "d_max"),
    ("margin", "margin"), ("lambda_io_final", "lambda_io_final"),
    ("spearman_rho", "spearman_rho"),
]

def _delta_pct(value, baseline):
    """(value - baseline) / baseline * 100, or None if either side is missing
    or baseline is 0 (division by a flat baseline of 0 is meaningless, not
    infinite) -- None renders as the table's "—" placeholder, never 0 or NaN."""
    if value is None or baseline is None or baseline == 0:
        return None
    return (value - baseline) / baseline * 100.0

def _fmt_or_dash(v):
    return "—" if v is None else _fmt(v)

def build_m2_table(result_jsons, flat_baselines):
    """Task 8 Step 3 / design v2 sec 8.2: render the ablation/report table.

    Args:
      result_jsons: list of paths to run JSON files (the A0-A7 schema written
        by run_flat/run_reweight/run_io).
      flat_baselines: dict keyed (case, k, rtype) -> {"io_count", "io_gp",
        "hpwl"} holding the flat mean each of Δio%/Δio_gp%/Δhpwl% is computed
        against (design v2 sec 8.2: "同 case/同 k/同 rtype/同 det regime 的
        flat 平均值"; the det-regime split is the caller's responsibility --
        pass a baselines dict already restricted to the run's det regime).
        A missing (case,k,rtype) key, or a missing metric within it, renders
        that Δ% cell "—" rather than guessing a value.

    Returns a single Markdown table (header + one row per result_jsons entry,
    in input order) as a string.
    """
    lines = ["| " + " | ".join(h for h, _ in M2_TABLE_SPEC) + " |",
             "|" + "---|" * len(M2_TABLE_SPEC)]
    for p in result_jsons:
        r = json.load(open(p))
        case = os.path.basename(r["config"]).replace(".json", "")
        base = flat_baselines.get((case, r.get("k"), r.get("rtype")), {})
        row = dict(r)
        row["case"] = case
        row["d_io_pct"] = _delta_pct(r.get("io_count"), base.get("io_count"))
        row["d_io_gp_pct"] = _delta_pct(r.get("io_gp"), base.get("io_gp"))
        row["d_hpwl_pct"] = _delta_pct(r.get("hpwl"), base.get("hpwl"))
        lines.append("| " + " | ".join(_fmt_or_dash(row.get(key))
                                       for _, key in M2_TABLE_SPEC) + " |")
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    main()
