# FLUTE、共享容量與 OpenROAD 閉環驗證

本次把三項功能接入同一個 placement loop：multi-pin FLUTE 成本決定位置候選與接受、每個 net 的共享分支只計一次且不同 nets 聯合占用容量、OpenROAD 觀測改變下一輪成本。論文全文筆記見 [routing-topology reference](../research/2026-09-14-routing-topology-placement-reference.md)。

## 方法與實作範圍

- `src/ioplace/route_eval/joint.py`：實際 bundled FLUTE、rectilinear 分支 union、L/Z/外繞候選、全 net 線長預算、零容量阻擋、聯合 utilization/overflow 成本。先移除受影響 nets，再重建；候選失敗不污染 incumbent。
- `src/ioplace/ops/joint_route_feedback.py`：等尺寸合法 cell swaps。以論文 Fig. 5 / Algorithm 2 的 Steiner / 中點 / 自身 anchor multiset 產生位置候選；Eq. 8 的加權 Manhattan minimizer 也作為搜尋起點。每個候選重新建立所有受影響 multi-pin nets，使用全域 joint objective 接受。沒有把整棵 RSMT 成本重複加到 WA 上。
- `src/ioplace/route_eval/online_openroad.py`：逐 checkpoint 執行真實 OpenROAD，讀取 per-net/per-layer segments 與 OpenDB capacity/usage。觀測先完整驗證、在 fork 上更新並 reroute，成功後才發布 state/history。
- IO 誤差以 per-net EMA 權重回饋；router utilization 以 spatial edge prices 回饋。保留 placement 的 router 觀測另更新 background。拒絕候選仍更新下一輪權重／價格；其位置與 background 不提交。
- 測量需求按每 net／每 layer 去重，跨層保留 multiplicity，再從 usage 扣除支援 nets。候選需求仍為每 net／2-D edge 一份。這是投影模型，沒有宣稱重現 layer assignment 或 detailed routing。

候選成本衡量線長預算內可找到的低 IO 路徑；它不是一般 router 的實際 IO。OpenROAD IO 另行量測，而且必須嚴格下降才保留 placement。HPWL 與 shared-union WL 上限始終相對原始 checkpoint 的 5%，不逐輪放寬；joint overflow 不得因接受候選增加。獨立 router WL 也受原始 5% 預算限制。

這是論文 anchor refinement 的 exact-pin／合法交換改寫；未實作論文的 analytical GP gradient、quadtree density search 或 coarse-bin schedule。CPU 執行 FLUTE 與 OpenROAD；DREAMPlace／GPU evaluator 使用 `CUDA_VISIBLE_DEVICES=1` 的 H100。

## 固定實驗設定

GCD 原始合法 checkpoints seed 1000、1001，各跑三輪。每輪最多 16 active nodes、每個 search seed 取 8 個鄰居，最大位移 32 個 region-grid cells。兩組使用相同 config、source、placement、cohort、工具與接受門檻，差別只有 `--no-learn`。

注意 ablation 的範圍：`--no-learn` 關閉全部 observation-derived net weights、edge prices 與 background refresh；仍讀 baseline capacity、仍執行每次候選的 OpenROAD veto。這個對照沒有單獨隔離「後續 EMA」和「初始 background」的效果。

每個 checkpoint 的 563 個 routed nets 用於實際 IO 比較。FLUTE optimization cohort 為 509 nets，其中 143 個 multi-pin、366 個 two-pin；54 個含 die 外端點的 nets 不在 FLUTE cohort，另 16 個 degree < 2 nets 不參與。88 個連到未建模非平凡 nets 的 movable cells 被凍結，避免其背景需求隨交換失效。因此 FLUTE raw IO 與完整 router IO 不能直接當成相同 cohort 的 prediction error。

## 結果

下表取自保存的 `result.json` 與 `audit.json`；實際 IO 使用完整一致的 routed-net cohort。

| Checkpoint | Router feedback | 實際 IO（前 → 後） | 實際 WL 變化 | HPWL 變化 |
|---|---|---:|---:|---:|
| GCD seed 1000 | 開啟 | 414 → 413 | -0.139% | +0.077% |
| GCD seed 1000 | 關閉 | 414 → 414 | +0.000% | +0.000% |
| GCD seed 1001 | 開啟 | 407 → 407 | +0.000% | +0.000% |
| GCD seed 1001 | 關閉 | 407 → 407 | +0.000% | +0.000% |

Seed 1000：開啟回饋第一輪交換 cells 21／110；FLUTE cohort IO 336 → 335，OpenROAD IO 414 → 413。舊 MST evaluator 卻是 396 → 397；這說明舊 MST gate 會否決這次真正 routed IO 改善。後兩輪 FLUTE 又提出 IO 335 → 334 的交換，但實際 IO 仍為 413，故拒絕。

觀測世代為 1、2、3。第三輪使用第二輪被拒絕候選的 receipt hash，其 incumbent joint objective 由 434.0892 變成 446.8625；更新訊號確實進入後續搜尋。同一交換仍再次被提出，因此目前校準不保證立即排除所有被 router 否決的候選。

Seed 1001 開／關回饋均維持 407；三個候選都沒有實際 IO 改善，保留 baseline。

Seed 1000 關閉回饋時，三輪提出 cells 114／105，FLUTE cohort IO 同樣 336 → 335，但 router IO 一直是 414，全部拒絕。結果只支持這些 checkpoints 上的觀測，不支持大型 benchmarks、signoff 或統計顯著性主張。

## 驗證與追溯

- 新舊 FLUTE／budgeted／feedback／真實 OpenROAD driver：**36 passed**，67.13 秒。
- GPU evaluator／route parser／crossing 非 slow 回歸：**82 passed, 1 skipped, 4 deselected**，10.31 秒。skip 為一般 Python 無法 import OpenDB；本次 OpenROAD subprocess integration 已另通過。
- 更廣的未篩選 GPU suite 曾跑到 **26 passed, 1 failed**，失敗為舊 slow regression 所需 `results/m2/ablation/adaptec1_A0_k16_grid.json.npz` 不存在。後續大型 slow 測試手動中止，改執行上述有界 suite；沒有宣稱完整 slow suite 通過。原始失敗／中止 log 保留在 `gpu_route_regression.log`。
- 四組實驗 audit 通過：16 個實際 OpenROAD invocations（各 baseline + 三候選），paired source/config/checkpoint hashes 一致、learning generations 被下一輪使用、selected placement 與已 routed checkpoint 座標完全相同。

回歸涵蓋同 net 共享分支／跨 net 累加、整 net 移除與回滾、容量競爭的替代路徑、實際 multi-pin-only 位置更新、固定 cells／原始預算／同世代比較、anchor 中點與 multiplicity、跨層 foreground 不殘留、無效觀測與 reroute 失敗不提交、拒絕後繼續學習，以及真正 OpenROAD 驅動的兩輪更新。另一個唯讀 reviewer 複查並重現兩項缺陷後，已確認修正。

Artifacts：`results/joint_route_feedback_20260914/`。每次執行有 protocol、原始／候選／保留 placements、per-net FLUTE 指標、容量／背景／價格陣列、oracle receipt、tool/input/output SHA-256。`source_snapshot/` 保存 131 個執行來源，`source_manifest.json` 另記 paper PDF 與 FLUTE source/LUT hashes。`audit.py` 檢查來源、配對設定、receipt、世代、selected placement 與實際 routed checkpoint 一致；結果寫入 `audit.json`。

初次呼叫因缺少 OpenROAD shared-library search path 失敗，已保留 `gcd_learn_missing_libs/` 和失敗 regression log。有效執行環境列於 `environment.sh`。重現：先在 repo root source 該檔，再執行各 run 的 `protocol.json` 所列 driver arguments；輸出需使用新目錄。
