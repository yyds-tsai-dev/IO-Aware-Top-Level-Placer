"""Report the completed frozen NVL run and its unchanged prospective test."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/h100_nvl_followup_20260906")
    args = parser.parse_args()
    root = args.root
    check = json.loads((root / "prediction_check.json").read_text())
    actual = json.loads((root / "full_3x3_flat_k16.json").read_text())
    prediction = json.loads((root / "forecast/h100_nvl_prediction.json").read_text())
    protocol = json.loads((root / "forecast/protocol.json").read_text())
    receipt = json.loads((root / "full_3x3_flat_k16.execution.json").read_text())
    if not check["protocol_compatible"] or receipt["status"] != "completed":
        raise ValueError("full run has not completed compatible adjudication")
    phases = check["phase_checks"]["phases"]
    names = ["read", "gp", "lg", "eval"]
    memory = check["gpu_memory"]
    wall = check["wall_time"]
    verdict = check["mainline_hypothesis"]["confirmed"]
    lines = ["# H100 NVL 27.7M 首跑與凍結預測裁決", "",
        "27,699,021 movable／27,804,699 physical nodes 的 native GP+LG+evaluation 已完成。",
        f"主線時間假設判定：**{'成立' if verdict is True else '不成立' if verdict is False else '不可評估'}**。",
        f"四個 phase 總和 {wall['actual']:.3f}s；事前凍結範圍 [{wall['band_low']:.3f}, {wall['band_high']:.3f}]s。",
        f"Supervisor wall time {receipt['supervisor_wall_s']:.3f}s，另列且不代換預先登錄的 phase sum。", "",
        "## Phase 判定", "",
        "| Phase | 實測 s | 凍結 fast–slow s | s_obs | 宣告 s 範圍 | 相對誤差≤50% |",
        "|---|---:|---:|---:|---:|---|"]
    for name in names:
        p = phases[name]
        f = prediction["wall_time_forecast"]["phases"][name]
        lines.append(f"| {name} | {p['t_h100_actual_s']:.3f} | {f['t_h100_fast_s']:.3f}–{f['t_h100_slow_s']:.3f} | {p['s_p_obs']:.5f} | {p['s_p_lo']}–{p['s_p_hi']} | {'PASS' if p['same_order'] else 'FAIL'} |")
    lines += ["", "這是未驗證硬體假設的敏感度範圍，並非統計prediction interval。",
        "較快或較慢而落在範圍外，都不能把凍結假設改判成立。首跑不產生新的有效PI。", "",
        "## 方法與歸因限制", "",
        "本次使用凍結driver：最終eval先執行完整serial CPU reference，再執行GPU evaluator。",
        "將整個混合phase除以2.5–5倍GPU加速因子，沒有反映其中的CPU工作。",
        "read／LG／GP亦跨越更新後DREAMPlace、CPU、資料格式與共享負載條件；",
        "phase比值不是單獨的GPU硬體因果加速比。上表如實保留原始預測，不事後放寬。",
        "後續開發版本已讓IO driver重用既有GPU最終結果，省去重複CPU reference；",
        "該版本通過實際40-iteration driver regression，但沒有回套本次首跑或FT/M5凍結結果。", "",
        "若需有效新模型，必須另行預註冊多個可比較H100觀測並驗證；",
        "本次元件與raw-read模型已獨立報告辨識失敗，不能補成未量測的PI。", "",
        "## 記憶體與完整性", "",
        f"Process peak allocated：{memory['process_peak_allocated_gib']:.3f} GiB；",
        f"whole-card sampled peak：{memory['whole_card_peak_gib']:.3f} GiB。",
        "共享背景用量不從whole-card峰值任意相減；解析process模型與whole-card量測的",
        "歸因不同，因此GPU memory forecast grade為not_applicable_shared_attribution。",
        f"Host process HWM：{check['host_rss']['actual']:.3f} GiB；沒有凍結有效host RSS區間。",
        "T9A/B各完成4次交錯迭代；84GiB為註冊process預算，H100 NVL實際總量93.584GiB。",
        "使用者允許容量足夠時共享GPU，各次裝置快照與背景程序保留在device history。", "",
        f"Legality：`{json.dumps(check['legality'], ensure_ascii=False)}`。",
        f"GP iterations：{actual.get('gp_iterations_run')} / {actual.get('gp_iteration_budget')}。",
        "GP iteration完成、overflow門檻與physical legality分別報告。此為synthetic scale實驗，",
        "不宣稱placement品質改善。Raw pins108,354,439與canonical pins106,683,202分開保存。", "",
        "## Provenance", "",
        f"GPU UUID：`{protocol['gpu_uuid']}`；DREAMPlace commit：`{receipt['dreamplace_commit']}`。",
        f"Frozen source digest：`{protocol['source_snapshot_digest']}`。",
        f"Prediction SHA256：`{check['prediction_sha256']}`；actual JSON：`{check['actual_sha256']}`。",
        "Checker驗證protocol/config/source與JSON/log/device hashes，並核對最終座標數量及finite值。",
        "座標NPZ雜湊在adjudication時計算；supervisor原先綁定的是JSON/log/device history。",
        "舊H100 SXM預測保留，沒有拿來當作此次NVL的相同SKU驗證。", "",
        "圖表：[PNG](figs/h100-nvl-followup-20260906.png)、[PDF](figs/h100-nvl-followup-20260906.pdf)。"]
    if verdict is not True:
        lines += ["", "任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間",
                  "本報告提供首跑歸因與模型不成立的證據；尚無足夠同條件觀測可產生有效重新校準區間。"]
    (REPO / "docs/results/2026-09-06-h100-nvl-followup.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, name in enumerate(names):
        f = prediction["wall_time_forecast"]["phases"][name]
        axes[0].vlines(i, f["t_h100_fast_s"], f["t_h100_slow_s"], color="tab:blue", linewidth=5)
        axes[0].plot(i, f["t_h100_fast_s"], "_", color="tab:blue")
        axes[0].plot(i, phases[name]["t_h100_actual_s"], "o", color="tab:red")
    axes[0].plot([], [], "-", color="tab:blue", label="Frozen sensitivity band")
    axes[0].plot([], [], "o", color="tab:red", label="Actual")
    axes[0].set_xticks(range(4), names)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Seconds (log scale)")
    axes[0].set_title("Frozen workload: phase times")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
    x = np.arange(4)
    axes[1].bar(x-.18, [actual["phases"][n]["peak_alloc_gb"] for n in names], .36, label="Process allocated")
    axes[1].bar(x+.18, [actual["phases"][n]["peak_reserved_gb"] for n in names], .36, label="Process reserved")
    axes[1].axhline(protocol["process_budget_gib"], color="black", linestyle="--", label="Registered process budget")
    axes[1].set_xticks(x, names); axes[1].set_ylabel("GiB")
    axes[1].set_title("Process CUDA allocator peaks")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("H100 NVL shared run; no exclusive hardware speedup claim")
    out = REPO / "docs/results/figs/h100-nvl-followup-20260906"
    fig.savefig(str(out)+".png", dpi=180); fig.savefig(str(out)+".pdf")


if __name__ == "__main__":
    main()
