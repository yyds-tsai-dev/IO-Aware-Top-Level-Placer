"""Render paired Stage2 calibration values without changing any gate."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fmt(value):
    return "N/A" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default="results/stage2_followup_20260906/calibration/tables.json")
    ap.add_argument("--out", default="docs/results/2026-09-06-stage2-followup.md")
    ap.add_argument("--fig", default="docs/results/figs/stage2-followup-20260906")
    args = ap.parse_args()
    path = Path(args.tables)
    data = json.loads(path.read_text())
    samples = data["valid_samples"]
    designs = sorted({row["design"] for row in samples})
    expected = {f"{design}__{arm}_k{k}" for design in ("mgc_fft_1", "des_perf_1", "mempool_tile_wrap")
                for arm in ("flat", "ours") for k in (16, 32)}
    missing = sorted(expected - {row["sample_id"] for row in samples})
    complete = not missing and not data["unevaluable_sample_table"]["rows"]
    lines = ["# Stage2 routing／逐net校準結果", "",
        f"本次分析{'涵蓋完整註冊cohort' if complete else '仍為partial快照'}：{len(samples)}/12 samples，{len(designs)}/3 designs。",
        "尚缺：" + ("、".join(missing) if missing else "無。"), "",
        "完整執行不等於各項模型／品質門檻PASS。歷史報告保留，本頁僅使用本輪",
        "OpenROAD routed geometry與逐net evaluator evidence；GPU重算、wire oracle、",
        "instance/net/endpoint identity、input/output hashes與固定region/net order均經檢查。", "",
        "## 樣本與可評估範圍", "",
        "正式母體為matched、已route、degree2–256的signal nets，unrouted signal≤2%；",
        "C4再取兩臂共同eligible net-name交集。FT未知值保留為未知，不補零。", "",
        "| Sample | K | Formal nets | Unrouted fraction | FT coverage | DR violations |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in samples:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key in (
            "sample_id", "k_placement", "calibration_nets", "unrouted_signal_fraction",
            "route_ft_coverage", "drc_final_violations_log")) + " |")
    c1 = {row["sample_id"]:row for row in data["c1_model_selection"].get("comparisons", [])}
    ratios = data["table1_total_ratio"]["rows"]
    lines += ["", "## C1／總量與逐net相關", "",
        "各ratio為route_cross_dw(2)/evaluator總量。C1逐sample套用原門檻，pooled相關僅供描述。", "",
        "| Sample | R λ−1 | R RG | R MST | ρ RG | ρ MST | C1 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in ratios:
        corr = c1.get(row["sample_id"], {})
        values = [row["sample_id"], row["R_lam_minus_1"], row["R_io_rg"], row["R_io_mst"],
                  corr.get("rho_rg"), corr.get("rho_mst"), corr.get("verdict", "not_evaluable")]
        lines.append("| " + " | ".join(map(fmt, values)) + " |")
    fit = data["regression"]
    lines += ["", "## C2／C3／C5", "",
        f"NNLS rank={fit['rank']}、residual dof={fit['dof']}、identifiable={fit['identifiable']}。",
        "Net rows在design內相依；沒有IID-net CI，也沒有已驗證的跨design不確定性區間。", "",
        "| Fit | α | β | γ | Rank | Dof |",
        "|---|---:|---:|---:|---:|---:|",
        "| pooled | " + " | ".join(fmt(fit["coefficients"][key]["value"]) for key in ("alpha", "beta", "gamma"))
        + f" | {fit['rank']} | {fit['dof']} |"]
    per_design = data["per_design_regression_informational"]
    for name, row in per_design.items():
        lines.append("| " + " | ".join(map(fmt, [name, row["alpha"], row["beta"], row["gamma"], row["rank"], row["dof"]])) + " |")
    lines += ["", "C2：`" + json.dumps(data["c2_regression_calibration"], ensure_ascii=False) + "`。", "",
        "C3：`" + json.dumps(data["c3_kappa_ft"], ensure_ascii=False) + "`。此估計不改動placement objective或歷史M3權重。", "",
        "C5：`" + json.dumps(data["c5_transfer_cv"], ensure_ascii=False) + "`。至少需要三個可辨識design fits。", "",
        "## C4／同K方向", "",
        f"Formal verdict：**{data['c4_sign_invariance']['formal_verdict']}**。", "",
        "| Design | From → To | Common nets | GP Δλ−1 | Route Δ | Scope | Flip |",
        "|---|---|---:|---:|---:|---|---|"]
    for row in data["c4_sign_invariance"]["rows"]:
        vals = [row["design"], row["from_arm"]+" → "+row["to_arm"], row.get("common_eligible_nets"),
                row.get("pre_route_delta"), row.get("post_route_delta"), row["comparison_scope"], row["sign_flip"]]
        lines.append("| " + " | ".join(map(fmt, vals)) + " |")
    scans = data["g4_raw_vs_dw2_drift"]["delta_scans"]
    lines += ["", "## G4／δ scan", "",
        "所有δ的正式總量使用相同primary eligible mask；括號為whole-design supplementary總量。", "",
        "| Sample | δ0 | δ1 | δ2 | δ4 |",
        "|---|---:|---:|---:|---:|"]
    for row in scans:
        if row.get("scan") is None:
            lines.append(f"| {row['sample_id']} | N/A | N/A | N/A | N/A |")
            continue
        scan = row["scan"]
        formal = {v["delta"]:v["route_cross_dw"] for v in scan["rows"]}
        whole = {v["delta"]:v["route_cross_dw"] for v in scan.get("whole_design_supplementary", [])}
        lines.append("| " + row["sample_id"] + " | " + " | ".join(f"{fmt(formal.get(k))} ({fmt(whole.get(k))})" for k in (0,1,2,4)) + " |")
    boundary = data["table7_boundary_pair_demand"]["rows"]
    lines += ["", "## Boundary demand", "",
        "Raw demand為crossing counts；長度正規化使用evaluator-coordinate units，不標成microns。", "",
        "| Sample | Eval max / p90 / Gini | Route max / p90 / Gini | Eval / route normalized max |",
        "|---|---|---|---|"]
    usable_boundary = [row for row in boundary if not row.get("not_evaluable")]
    for row in usable_boundary:
        ev, rt = row["evaluator_statistics"], row["route_statistics"]
        values = [row["sample_id"], *[" / ".join(fmt(side["boundary_demand"][key]) for key in ("max", "p90", "gini")) for side in (ev,rt)],
                  " / ".join(fmt(side["boundary_demand_per_length"]["max"]) for side in (ev,rt))]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## 限制與證據", "",
        "固定detailed-route迭代上限5（iteration0–5），殘留DRC如上；不是signoff-clean。",
        "本輪為OpenROAD-only，沒有Innovus或commercial-DP ground truth。CPU/GPU共享量測不作獨占效能宣稱。",
        "完整逐net相關、degree／lambda buckets、FT decomposition與boundary資料均保存在tables.json。",
        "GP與routed evaluator的差值也跨越native pin保留／OpenDB endpoint與pin-center慣例，",
        "不能全部歸因於router移動cell；routing前後的OpenDB幾何變化由獨立identity artifact記錄。",
        f"輸入 `{path}`；SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`。",
        "圖表：[PNG](figs/stage2-followup-20260906.png)、[PDF](figs/stage2-followup-20260906.pdf)。"]
    Path(args.out).write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(len(ratios))
    for key, label in (("R_lam_minus_1", "lambda-1"), ("R_io_rg", "RG"), ("R_io_mst", "MST")):
        axes[0,0].plot(x, [v[key] if v[key] is not None else np.nan for v in ratios], "o-", label=label)
    axes[0,0].set_xticks(x, [v["sample_id"] for v in ratios], rotation=65, ha="right", fontsize=7)
    axes[0,0].set(title="Formal route / evaluator totals", ylabel="Ratio"); axes[0,0].legend()
    for row in scans:
        if row.get("scan") is None: continue
        values = {v["delta"]:v["route_cross_dw"] for v in row["scan"]["rows"]}
        if values.get(2):
            axes[0,1].plot([0,1,2,4], [values[k]/values[2] for k in (0,1,2,4)], "o-", label=row["sample_id"])
    axes[0,1].set(title="Same-population delta sensitivity", xlabel="delta", ylabel="Crossings / delta2 crossings")
    axes[0,1].legend(fontsize=6)
    for i, (name, row) in enumerate(per_design.items()):
        axes[1,0].bar(np.arange(3)+i*.22, [row[k] for k in ("alpha", "beta", "gamma")], .22, label=name)
    axes[1,0].set_xticks(np.arange(3)+.11*(len(per_design)-1), ["alpha", "beta", "gamma"])
    axes[1,0].set(title="Per-design NNLS (informational)", ylabel="Coefficient"); axes[1,0].legend(fontsize=7)
    for row in usable_boundary:
        axes[1,1].scatter(row["evaluator_statistics"]["boundary_demand"]["gini"],
            row["route_statistics"]["boundary_demand"]["gini"], label=row["sample_id"])
    axes[1,1].set(title="Boundary demand Gini", xlabel="Evaluator Gini", ylabel="Route Gini")
    axes[1,1].legend(fontsize=6)
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.suptitle(f"Stage2 {'complete cohort' if complete else 'partial snapshot'}: {len(samples)} samples / {len(designs)} designs")
    fig.savefig(args.fig+".png", dpi=180); fig.savefig(args.fig+".pdf")


if __name__ == "__main__":
    main()
