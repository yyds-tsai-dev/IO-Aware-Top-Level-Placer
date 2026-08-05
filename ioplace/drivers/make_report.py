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

if __name__ == "__main__":
    main()
