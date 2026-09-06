import json
from pathlib import Path
import matplotlib.pyplot as plt

R = Path("results/ft_followup_20260906")
O = Path("docs/results/figs"); O.mkdir(parents=True, exist_ok=True)
s = json.loads((R / "summary.json").read_text())
arms = s["levers"]
names = list(arms); io = [arms[x]["io_count"] for x in names]
ft = [arms[x]["ft_count"] for x in names]; hp = [arms[x]["hpwl"] / 1e6 for x in names]
flat = s["flat_seed_noise"]["io_count"]["mean"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), dpi=180)
sc = ax[0].scatter(io, ft, c=hp, cmap="viridis", s=75, edgecolor="black")
for n, x, y in zip(names, io, ft): ax[0].annotate(n, (x, y), xytext=(5, 4), textcoords="offset points")
ax[0].axvline(flat, color="tab:red", ls="--", lw=1.2, label=f"IO guard / flat mean ({flat:.0f})")
ax[0].set(xlabel="Final IO count", ylabel="Final FT count", title="FT vs IO by arm")
ax[0].legend(fontsize=8); fig.colorbar(sc, ax=ax[0], label="HPWL (million)")
ax[1].bar(names, io, color="tab:blue", alpha=.75, label="IO")
ax[1].axhline(flat, color="tab:red", ls="--", label="flat mean")
ax[1].set(ylabel="Final IO count", title="IO guard screening")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(O / "ft-followup-20260906.png"); fig.savefig(O / "ft-followup-20260906.pdf")
