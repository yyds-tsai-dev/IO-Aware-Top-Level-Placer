"""Render paired Stage2 calibration values without changing any gate."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fmt(value):
    return "N/A" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def g4_supplement(data, fig_prefix):
    root = Path(data["s8_dir"])
    raw_path = root / "g4_raw/tables.json"
    distance_root = root / "wire_boundary_distance"
    if not raw_path.exists() or not (distance_root / "execution.json").exists():
        return ["", "G4 raw雙值結論／wire距離診斷尚未完成。"]
    raw = json.loads(raw_path.read_text())
    receipt = json.loads((distance_root / "execution.json").read_text())
    expected = {s["sample_id"] for s in data["valid_samples"]}
    distances = []
    for name, digest in receipt["outputs"].items():
        path = Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("distance diagnostic digest changed")
        distances.append(json.loads(path.read_text()))
    if (receipt["status"] != "completed" or raw["status"] != "completed"
            or {s["sample_id"] for s in distances} != expected or set(raw["inputs"]) != expected):
        raise ValueError("incomplete G4 companion cohort")
    for sample in data["valid_samples"]:
        path = Path(sample["source_paths"]["crossings_matched"]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw["inputs"][sample["sample_id"]].get(str(path)) != digest:
            raise ValueError("raw companion uses different primary evidence")
    lines = ["", "## G4／raw與δ2雙值結論", "",
        f"正式總量raw={data['g4_raw_vs_dw2_drift']['route_cross_raw_total']}、δ2={data['g4_raw_vs_dw2_drift']['route_cross_dw2_total']}；",
        f"差異為{fmt(data['g4_raw_vs_dw2_drift']['drift_pct'])}%，觸發15%門檻。以下raw為補充，δ2仍是主要判定。",
        "兩者正式eligible母體相同。相關／回歸各自套用原有非零active規則；",
        "raw新增active rows及兩個mask hash逐sample保存，相關係數／R²差異不視為完全相同rows的比較。", "",
        "| Sample | Raw R λ−1 / RG / MST | Raw ρ RG / MST | C1 raw / δ2 | Active raw / δ2 |",
        "|---|---|---|---|---|"]
    primary_c1 = {r["sample_id"]:r for r in data["c1_model_selection"]["comparisons"]}
    raw_c1 = {r["sample_id"]:r for r in raw["c1_model_selection"]["comparisons"]}
    active = {r["sample_id"]:r for r in raw["active_population_comparison"]}
    for row in raw["table1_total_ratio"]["rows"]:
        name = row["sample_id"]; c = raw_c1[name]; a = active[name]
        lines.append("| " + " | ".join([name,
            " / ".join(fmt(row[k]) for k in ("R_lam_minus_1", "R_io_rg", "R_io_mst")),
            f"{fmt(c['rho_rg'])} / {fmt(c['rho_mst'])}",
            f"{c['verdict']} / {primary_c1[name]['verdict']}",
            f"{a['n_active_raw']} / {a['n_active_dw2']}"]) + " |")
    def e2_count(values):
        groups = {}
        for row in values["table2_per_net_correlation"]["rows"]:
            if row["model"] in ("io_rg", "io_mst") and row["spearman"] is not None:
                groups.setdefault(row["sample_id"], []).append(row["spearman"])
        return sum(max(v) >= .70 for v in groups.values())
    lines += ["", "### E1–E5與G1–G7判定", "",
        "| Exit | 本輪結果 |", "|---|---|",
        f"| E1 | {len(expected)}個樣本、3 designs×2 K×2臂，identity／≤2% coverage結果見主表；flat的同DEF明示重用。 |",
        f"| E2 | Spearman≥0.70：raw {e2_count(raw)}/{len(expected)}、δ2 {e2_count(data)}/{len(expected)}；未全數通過。 |",
        "| E3 | Raw／δ2 ratio gate及in-sample校準殘差均已交付，數值見下表及JSON。 |",
        "| E4 | Raw／δ2各7張表與C1／C5判定已交付。 |",
        "| E5 | Raw／δ2的C4已完成，方向及差值見下表。 |", "",
        "G1：本輪交付open-source router ground truth，僅相對OpenROAD；未執行本機Innovus license重試。",
        "G2：C1存在reconsider_rg_model結果，RG模型需重審；本輪未據此改動M3 loss。",
        "G3：依主要δ2的C5判定處理，跨benchmark遷移是後续使用者決策。",
        "G4：觸發，雙值報告、δ scan與wire距離診斷已補齊。",
        "G5／G7：本輪未執行Innovus／commercial DP，無法評估。",
        "G6：原tile缺線由繼承stub造成，保留失敗後修正routing輸入；相同placement重繞後全樣本0%缺線。"]
    lines += ["", "| C2／C3／C5 | Raw | δ2 primary |", "|---|---|---|"]
    for label, getter in (
        ("E3 ratio gate", lambda x: x["c2_regression_calibration"]["e3_pass"]),
        ("Pooled α / β / γ", lambda x: " / ".join(fmt(x["regression"]["coefficients"][k]["value"]) for k in ("alpha","beta","gamma"))),
        ("κ FT（估計值）", lambda x: x["c3_kappa_ft"]["kappa_ft"]),
        ("C5 CV α / β / γ", lambda x: " / ".join(fmt(x["c5_transfer_cv"]["cv"][k]) for k in ("alpha","beta","gamma"))),
        ("C5 verdict", lambda x: x["c5_transfer_cv"]["verdict"])):
        lines.append(f"| {label} | {fmt(getter(raw))} | {fmt(getter(data))} |")
    lines += ["", "E3判定如上；需要校準時的in-sample io_calibrated/residual與各design係數均保存於各自tables.json。",
        f"C5 raw={raw['c5_transfer_cv']['verdict']}、主要δ2={data['c5_transfer_cv']['verdict']}。跨設計套用仍依主要δ2判定，估計值不是已驗證objective權重。",
        ("依G3，benchmark／Phase1遷移仍需另行決策；本輪未改benchmark或重寫既有exit verdict。"
         if data["c5_transfer_cv"]["verdict"] == "fail" else "本輪未改benchmark或重寫既有exit verdict。"), "",
        "| Design / K | GP Δλ−1 | Route Δ raw / δ2 | C4 raw / δ2 |", "|---|---:|---:|---|"]
    primary_c4 = {(r["design"],r["from_arm"]):r for r in data["c4_sign_invariance"]["rows"]}
    for row in raw["c4_sign_invariance"]["rows"]:
        p = primary_c4[(row["design"],row["from_arm"])]
        lines.append(f"| {row['design']} / {row['from_arm']} | {row['pre_route_delta']} | {row['post_route_delta']} / {p['post_route_delta']} | {'flip' if row['sign_flip'] else 'same sign'} / {'flip' if p['sign_flip'] else 'same sign'} |")
    lines += ["", f"C4 formal verdict：raw={raw['c4_sign_invariance']['formal_verdict']}、δ2={data['c4_sign_invariance']['formal_verdict']}。",
        "FT與boundary demand以raw visited regions／實際轉換計數，",
        "不受δ過濾影響；已逐sample確認FT、wire length、pair demand完全相等。七張表均提供raw伴隨版本。", "",
        "## G4／wire到邊界距離", "",
        "以下為WIRE長度加權CDF：依垂直方向的lattice cell尺寸正規化距離，裁切到die內，保留decoded segment重複量。",
        "Any為到任一internal grid boundary的最近距離；Parallel只計與wire方向平行的boundary。",
        "這個距離不是δ的run-length閾值。d=0的沿界wire有正長度，單純穿越交點的長度為零。",
        "此分布提供幾何診斷；不能單凭raw／δ2差異就斷言沿界走線是唯一或主要原因。", "",
        "| Sample | Any ≤1 / ≤2 cells | Parallel ≤1 / ≤2 cells | Outside die WIRE DBU |", "|---|---|---|---:|"]
    for row in distances:
        d = row["formal"]; ix = [d["distance_cells"].index(v) for v in (1.,2.)]
        lines.append("| " + " | ".join([row["sample_id"],
            *[" / ".join(f"{100*d[key][i]:.3f}%" for i in ix) for key in ("any_boundary_cdf","parallel_boundary_cdf")],
            fmt(d["outside_die_wire_length_dbu"])]) + " |")
    fig, axes = plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for i, design in enumerate(sorted({s["design"] for s in data["valid_samples"]})):
        for j, key in enumerate(("any_boundary_cdf", "parallel_boundary_cdf")):
            ax=axes[i,j]
            for row in distances:
                if row["sample_id"].startswith(design+"__"):
                    d=row["formal"]; ax.plot(d["distance_cells"],d[key],label=row["sample_id"].split("__")[1])
            ax.set_xscale("symlog",linthresh=.25); ax.set_ylim(0,1.02)
            ax.set_title(design+" / "+("any" if j==0 else "parallel")); ax.set_xlabel("Distance (perpendicular lattice cells)")
            ax.set_ylabel("WIRE length fraction"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    for suffix in ("png","pdf"):
        fig.savefig(str(fig_prefix)+"-distance."+suffix,dpi=160)
    plt.close(fig)
    lines += ["", f"Raw evidence `{raw_path}`；SHA256 `{hashlib.sha256(raw_path.read_bytes()).hexdigest()}`。",
        f"Distance protocol／12份結果／hash receipt：`{distance_root}`。",
        "距離圖：[PNG](figs/stage2-followup-20260906-distance.png)、[PDF](figs/stage2-followup-20260906-distance.pdf)。"]
    return lines


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
    coverage_failed = [s["sample_id"] for s in samples if s.get("unrouted_signal_gate_pass") is False]
    coverage_pass = complete and all(s.get("unrouted_signal_gate_pass") is True for s in samples)
    coverage_verdict = "PASS" if coverage_pass else "FAIL" if coverage_failed else "待完成／不可評估"
    lines = ["# Stage2：open-source router ground truth／逐net校準結果", "",
        f"本次分析{'涵蓋完整註冊cohort' if complete else '仍為partial快照'}：{len(samples)}/12 samples，{len(designs)}/3 designs。",
        "尚缺：" + ("、".join(missing) if missing else "無。"), "",
        f"完整cohort的≤2% signal缺線gate：**{coverage_verdict}**。",
        "失敗樣本：" + ("、".join(coverage_failed) if coverage_failed else "無已觀察失敗；未完成樣本仍待檢查。" if not coverage_pass else "無。"), "",
        "完整執行不等於各項模型／品質門檻PASS。歷史報告保留，本頁僅使用本輪",
        "OpenROAD routed geometry與逐net evaluator evidence；GPU重算、wire oracle、",
        "instance/net/endpoint identity、input/output hashes與固定region/net order均經檢查。", "",
        "## 樣本與可評估範圍", "",
        "正式母體為matched、已route、degree2–256的signal nets，unrouted signal≤2%；",
        "C4再取兩臂共同eligible net-name交集。FT未知值保留為未知，不補零。", "",
        "| Sample | K | Formal nets | Unrouted fraction | FT coverage | Final DR log violations |",
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
    setup_path = Path(data["s8_dir"]) / "protocol.json"
    if setup_path.exists():
        setup = json.loads(setup_path.read_text())
        if setup.get("same_placements_and_cohort"):
            lines += ["", "## 繼承routing的修正與重用範圍", "",
                "tile首次route有4,360條零長度signal ROUTED記錄仍停在舊座標；",
                "GRT將nonnull dbWire視為既有routing而跳過，恰好造成3.2006%缺線。",
                "原始樣本與2% gate失敗完整保留。修正run先移除一般signal dbWire，",
                "保留special／POWER／GROUND routing，再對相同placement執行GR＋DR。",
                "沒有修改density、cohort、placement座標或coverage門檻。",
                "FFT／DES四臂的NETS section各自逐byte相同，且沒有繼承signal routing；",
                "因此保留其有效route與evidence，透過明示case symlink納入本次校準。",
                "tile四臂使用獨立結果目錄；兩個flat K的DEF／netmap／CoordMap／config一致才重用route。",
                f"修正protocol：`{setup_path}`；SHA256 `{hashlib.sha256(setup_path.read_bytes()).hexdigest()}`。"]
    lines += g4_supplement(data, args.fig)
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
