"""Validate M5 receipts and report the exact registered acceptance metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
ROOT = REPO / "results/m5_followup_20260906"
DESIGNS = ("adaptec1", "mempool_tile_wrap", "bigblue4")
MODES = ("none", "ce", "refine", "ce_refine")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    manifest = ROOT / "manifest.json"
    rows, pending = [], []
    for design in DESIGNS:
        for mode in MODES:
            name = f"{design}__{mode}__seed1000"
            path = ROOT / "runs" / (name + ".json")
            receipt_path = path.with_suffix(".execution.json")
            if not path.exists():
                pending.append(name)
                continue
            value, receipt = (json.loads(p.read_text()) for p in (path, receipt_path))
            if (receipt.get("returncode") != 0 or receipt["output_sha256"] != sha(path)
                    or receipt["manifest_sha256"] != sha(manifest)):
                raise ValueError(f"execution receipt mismatch: {name}")
            d = value.get("discrete_result")
            coordinates_equal_to_none = None
            if d:
                base, candidate = d["baseline"], d["proposal"]
                score = candidate["io_count"]/max(base["io_count"],1) + candidate["ft_count"]/max(base["ft_count"],1)
                if abs(score-d["proposal_score"]) > 1e-10:
                    raise ValueError(f"acceptance score mismatch: {name}")
                if not d["accepted"] and any(value[k] != base[k] for k in ("io_count", "ft_count", "hpwl")):
                    raise ValueError(f"rejected proposal changed incumbent: {name}")
                if sha(Path(d["sidecar"])) != d["sidecar_sha256"]:
                    raise ValueError(f"discrete sidecar changed: {name}")
                if not d["accepted"]:
                    baseline = ROOT / "runs" / f"{design}__none__seed1000.json.npz"
                    with np.load(baseline, allow_pickle=False) as before, np.load(str(path)+".npz", allow_pickle=False) as after:
                        coordinates_equal_to_none = all(np.array_equal(before[key],after[key]) for key in ("node_x","node_y"))
                    if not coordinates_equal_to_none:
                        raise ValueError(f"rejected final coordinates differ from matched none arm: {name}")
            rows.append(dict(design=design, mode=mode, path=str(path), sha256=sha(path),
                receipt_sha256=sha(receipt_path), coordinates_equal_to_none=coordinates_equal_to_none, result=value))
    return dict(rows=rows, pending=pending, manifest_sha256=sha(manifest))


def main():
    summary = collect()
    (ROOT / "checked_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    rows = summary["rows"]
    lines = ["# 2026-09-06 M5 實測結果", "",
        f"已核對 {len(rows)}/12 個 seed1000 初篩 run 的輸出、退出碼與凍結 manifest。",
        "待完成：" + ("、".join(summary["pending"]) if summary["pending"] else "無。"), "",
        "接受條件使用完整 netlist 的 MST `io_count`、`ft_count`：",
        "`S = IO_candidate / max(IO_incumbent,1) + FT_candidate / max(FT_incumbent,1)`。",
        "候選必須在 legalization 後嚴格優於 incumbent 與額外一次 LG control 的分數，",
        "同時 HPWL ≤ 1.01 × incumbent，通過 legality、fixed-tail 與 control 有效性檢查。",
        "`io_rg`、`ft_rg` 是另外保存的診斷欄位，沒有代入這項接受判定。", "",
        "## 合法基準與額外 LG control", "",
        "| 設計 | incumbent IO / FT | incumbent HPWL | extra LG IO / FT | extra LG HPWL |",
        "|---|---:|---:|---:|---:|"]
    for design in DESIGNS:
        available = [r for r in rows if r["design"] == design and r["mode"] == "ce"]
        if not available:
            continue
        d = available[0]["result"]["discrete_result"]
        b, c = d["baseline"], d["legalizer_control"]
        lines.append(f"| {design} | {b['io_count']} / {b['ft_count']} | {b['hpwl']:.3f} | {c['io_count']} / {c['ft_count']} | {c['hpwl']:.3f} |")
    lines += ["", "再次 LG 會改變座標與品質，因此每個候選都實際執行 matched extra-LG control。", "",
        "## Legalization 後候選", "",
        "| 設計 | 方法 | IO / FT | S | 接受上限（嚴格小於） | HPWL 比值 | 接受 | 拒絕原因 |",
        "|---|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        if row["mode"] == "none":
            continue
        d = row["result"]["discrete_result"]
        p, b = d["proposal"], d["baseline"]
        lines.append(f"| {row['design']} | {row['mode']} | {p['io_count']} / {p['ft_count']} | {d['proposal_score']:.6f} | {d['acceptance_score_threshold']:.6f} | {p['hpwl']/b['hpwl']:.6f} | {'是' if d['accepted'] else '否'} | {', '.join(d['reasons']) or '—'} |")
    lines += ["", "拒絕的候選保留原合法 incumbent；已將各最終座標與對應 none run 逐元素核對一致，",
        "並驗證 discrete sidecar hash。表中的候選分數改善不等於通過完整品質門檻。",
        "只有初篩接受的方法才啟動 seed1001–1003 確認，並逐 seed 配對 none control。",
        "初篩沒有接受的方法，不增加確認 seed 或事後放寬 HPWL 門檻。", "",
        "## 1M / 10M / 30M 元件規模量測", "",
        "| N | active | preparation s | CE s | refine s | host HWM GiB | GPU allocated / reserved GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for path in sorted((REPO / "results/m5_scale_20260906").glob("n*.json"), key=lambda p:int(p.stem[1:])):
        x = json.loads(path.read_text())
        if x["status"] != "completed":
            raise ValueError(f"scale incomplete: {path}")
        lines.append(f"| {x['N']} | {x['preparation']['active_cells']} | {x['preparation']['preparation_s']:.4f} | {x['ce']['decode_s']:.4f} | {x['refine']['refine_s']:.4f} | {x['host_peak_rss_gib']:.4f} | {x['process_peak_allocated_gib']:.4f} / {x['process_peak_reserved_gib']:.4f} |")
    lines += ["", "固定 K16、active 上限65,536，耗時與記憶體仍依賴 active cells 的 incident nets/pins。",
        "這些是共享 H100 NVL 上的 unlegalized synthetic 元件量測，沒有 full GP 或 physical legality 結果；",
        "不能據此宣稱 30M 最終 placement 品質改善或完整流程速度。", "",
        "## 證據", "",
        f"Manifest SHA-256：`{summary['manifest_sha256']}`。內含 source/config/input 與 preregistration hashes。",
        "C++ source SHA-256：`a631070eb306b2836a4aea8bf90fbeb6f40b80862815ccc2ee04f9aadeaf3b1a`。",
        "`checked_summary.json` 保存逐 run 輸出與 receipt hashes。初篩使用 region seed0、DP seed1000、",
        "deterministic=1、rho0、K16 grid、GP+LG、DP 關閉；各 arm 使用同一凍結 source。",
        "後續開發中的 GPU-only 最終評估改動沒有套入本組凍結實驗。", "",
        "圖表：[PNG](figs/m5-followup-20260906.png)、[PDF](figs/m5-followup-20260906.pdf)。"]
    (REPO / "docs/results/2026-09-06-m5-followup.md").write_text("\n".join(lines)+"\n")
    print(f"checked {len(rows)}/12; pending {summary['pending']}")


if __name__ == "__main__":
    main()
