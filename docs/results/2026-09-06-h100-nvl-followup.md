# H100 NVL 27.7M 首跑與凍結預測裁決

27,699,021 movable／27,804,699 physical nodes 的 native GP+LG+evaluation 已完成。
主線時間假設判定：**不成立**。
四個 phase 總和 13042.050s；事前凍結範圍 [24213.928, 25490.045]s。
Supervisor wall time 13330.271s，另列且不代換預先登錄的 phase sum。

## Phase 判定

| Phase | 實測 s | 凍結 fast–slow s | s_obs | 宣告 s 範圍 | 相對誤差≤50% |
|---|---:|---:|---:|---:|---|
| read | 1476.255 | 22731.414–22731.414 | 15.39803 | 1.0–1.0 | FAIL |
| gp | 305.236 | 199.551–399.102 | 3.26880 | 2.5–5.0 | PASS |
| lg | 481.518 | 206.397–206.397 | 0.42864 | 1.0–1.0 | FAIL |
| eval | 10779.041 | 1076.566–2153.132 | 0.49938 | 2.5–5.0 | FAIL |

這是未驗證硬體假設的敏感度範圍，並非統計prediction interval。
較快或較慢而落在範圍外，都不能把凍結假設改判成立。首跑不產生新的有效PI。

## 方法與歸因限制

本次使用凍結driver：最終eval先執行完整serial CPU reference，再執行GPU evaluator。
將整個混合phase除以2.5–5倍GPU加速因子，沒有反映其中的CPU工作。
read／LG／GP亦跨越更新後DREAMPlace、CPU、資料格式與共享負載條件；
phase比值不是單獨的GPU硬體因果加速比。上表如實保留原始預測，不事後放寬。
後續開發版本已讓IO driver重用既有GPU最終結果，省去重複CPU reference；
該版本通過實際40-iteration driver regression，但沒有回套本次首跑或FT/M5凍結結果。

若需有效新模型，必須另行預註冊多個可比較H100觀測並驗證；
本次元件與raw-read模型已獨立報告辨識失敗，不能補成未量測的PI。

## 記憶體與完整性

Process peak allocated：25.308 GiB；
whole-card sampled peak：29.612 GiB。
共享背景用量不從whole-card峰值任意相減；解析process模型與whole-card量測的
歸因不同，因此GPU memory forecast grade為not_applicable_shared_attribution。
Host process HWM：185.612 GiB；沒有凍結有效host RSS區間。
T9A/B各完成4次交錯迭代；84GiB為註冊process預算，H100 NVL實際總量93.584GiB。
使用者允許容量足夠時共享GPU，各次裝置快照與背景程序保留在device history。

Legality：`{"legalization_status": "success", "num_unplaced_cells": 0, "stop_overflow_reached": true, "final_overflow": 0.06630443036556244}`。
GP iterations：1065 / 2000。
GP iteration完成、overflow門檻與physical legality分別報告。此為synthetic scale實驗，
不宣稱placement品質改善。Raw pins108,354,439與canonical pins106,683,202分開保存。

## Provenance

GPU UUID：`GPU-1a75fdd3-2f46-0258-4123-095efa8636e9`；DREAMPlace commit：`6627f3327e6cc17db7782c0b90073a498531ca3c`。
Frozen source digest：`fe3cee71184899133e557b2611a0935e898d385aa611ec711b71b6938597e192`。
Prediction SHA256：`951a05e9c4775c3f5b245436ceebc99f79f806ea8c11521b38008bd1bf9fefbb`；actual JSON：`8cd6973b08a1a7383030629d5cbe42959cce35282df701b24f14ab4e42a58239`。
Checker驗證protocol/config/source與JSON/log/device hashes，並核對最終座標數量及finite值。
座標NPZ雜湊在adjudication時計算；supervisor原先綁定的是JSON/log/device history。
舊H100 SXM預測保留，沒有拿來當作此次NVL的相同SKU驗證。

圖表：[PNG](figs/h100-nvl-followup-20260906.png)、[PDF](figs/h100-nvl-followup-20260906.pdf)。

## 額外cache順序診斷（不改寫首跑）

首次cache重建的GPU核對出現IO−1／FT−2；三次GPU逐net結果一致。
對314,108個候選net執行CPU reference後，CPU(cache)與GPU(cache)逐net一致，
其IO455,889／FT102,943與原始native輸入的IO455,890／FT102,945不同。
差異定位到cache漏掉native createPin的逐output front/back交換；
既有等價測試只比pin multiset，未驗證net遍歷順序。Schema4補足此順序，
schema3及首次失敗檢查完整保留，沒有修改GPU evaluator演算法。
修正cache的全量驗證與27.7M核對尚在執行。

任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間
本報告提供首跑歸因與模型不成立的證據；尚無足夠同條件觀測可產生有效重新校準區間。
