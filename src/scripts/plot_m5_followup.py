"""Plot M5 follow-up proposal evidence with separate units."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from summarize_m5_followup import collect

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
RUNS = REPO / "results/m5_followup_20260906/runs"
SCALE = REPO / "results/m5_scale_20260906"
OUT = REPO / "docs/results/figs/m5-followup-20260906"


def run(case, mode):
    return json.loads((RUNS / f"{case}__{mode}__seed1000.json").read_text())


def main():
    checked = collect()
    cases = [case for case in ("adaptec1", "mempool_tile_wrap", "bigblue4")
             if all(any(row['design']==case and row['mode']==mode for row in checked['rows'])
                    for mode in ('none','ce','refine','ce_refine'))]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"adaptec1": "#1f77b4", "mempool_tile_wrap": "#d62728", "bigblue4": "#2ca02c"}
    for case in cases:
        base = run(case, "none")
        label = case
        modes = ["ce", "refine", "ce_refine"]
        for mode in modes:
            d = run(case, mode)
            prop = d["discrete_result"]["proposal"]
            discrete = d['discrete_result']
            score = discrete['proposal_score'] - discrete['acceptance_score_threshold']
            x = modes.index(mode) + cases.index(case)*.18
            ax[0, 0].bar(x, score, width=.16, color=colors[case], alpha=.8,
                         label=label if mode == "ce" else None)
            ax[0, 1].bar(x, prop["hpwl"] / base["hpwl"], width=.16, color=colors[case], alpha=.8)
            ax[1, 0].bar(x, prop["io_count"], width=.16, color=colors[case], alpha=.8)
            ax[1, 1].bar(x, prop["ft_count"], width=.16, color=colors[case], alpha=.8)
    names = ["CE", "Refine", "CE+Refine"]
    for a, title, ylabel in zip(ax.flat,
                                ["Full IO+FT score minus acceptance threshold", "Proposal normalized HPWL",
                                 "Post-LG candidate IO", "Post-LG candidate FT"],
                                ["score difference (must be < 0)", "proposal / baseline", "IO count", "FT count"]):
        a.set_title(title); a.set_ylabel(ylabel)
        a.set_xticks([i+.09*(len(cases)-1) for i in range(3)], names)
        a.grid(axis="y", alpha=.25)
    ax[0, 0].axhline(0, color="black", linestyle="--", linewidth=.8)
    ax[0, 1].axhline(1.01, color="black", linestyle="--", linewidth=.8, label="HPWL 1.01 limit")
    ax[0, 0].legend()
    fig.suptitle("M5 follow-up: completed four-arm screening cohorts")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT) + ".png", dpi=180)
    fig.savefig(str(OUT) + ".pdf")


if __name__ == "__main__":
    main()
