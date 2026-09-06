"""Plot frozen component-model holdouts without clipping invalid predictions."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
COMP = REPO / "results/component_models_20260906"
HOST = REPO / "results/host_rss_models_20260906"
OUT = REPO / "docs/results/figs/component-models-20260906"


def main():
    hold = json.loads((COMP / "holdout_check.json").read_text())
    host = json.loads((HOST / "holdout_check.json").read_text())["checks"]
    rows = hold["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"evaluator": "#1f77b4", "io_term": "#d62728"}
    ax = axes[0, 0]
    for component in ("evaluator", "io_term"):
        for target, suffix, marker in (("peak_allocated_gib", "allocated", "o"), ("peak_reserved_gib", "reserved", "s")):
            sub = [r for r in rows if r["component"] == component and r["target"] == target]
            ax.scatter([r["actual"] for r in sub], [r["point"] for r in sub],
                       color=colors[component], marker=marker, alpha=.75,
                       label=f"{component} {suffix}")
    ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=.8)
    ax.set_title("Component memory holdout")
    ax.set_xlabel("Actual (GiB)"); ax.set_ylabel("Predicted point (GiB)")
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = axes[0, 1]
    for component in ("evaluator", "io_term"):
        sub = [r for r in rows if r["component"] == component and r["target"] == "warm_mean_wall_s"]
        ax.scatter([r["actual"] for r in sub], [r["point"] for r in sub],
                   color=colors[component], alpha=.8, label=component)
    ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=.8)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set_title("Component runtime holdout (negative values retained)")
    ax.set_xlabel("Actual (seconds)"); ax.set_ylabel("Predicted point (seconds)")
    ax.legend(); ax.grid(alpha=.25)

    ax = axes[1, 0]
    labels = ["Host RSS"]
    ax.bar([0], [host["host_peak_rss_gib"]["actual"]], width=.35, label="Actual")
    ax.bar([.4], [host["host_peak_rss_gib"]["point"]], width=.35, label="Predicted point")
    ax.set_xticks([.2], labels); ax.set_ylabel("GiB"); ax.set_title("Host RSS holdout")
    ax.legend(); ax.grid(axis="y", alpha=.25)

    ax = axes[1, 1]
    ax.bar([0], [host["read_s"]["actual"]], width=.35, label="Actual")
    ax.bar([.4], [host["read_s"]["point"]], width=.35, label="Predicted point")
    ax.set_xticks([.2], ["Read time"]); ax.set_ylabel("Seconds"); ax.set_title("Host read-time holdout")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.suptitle("2026-09-06 component and host-RSS models")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT) + ".png", dpi=180)
    fig.savefig(str(OUT) + ".pdf")


if __name__ == "__main__":
    main()
