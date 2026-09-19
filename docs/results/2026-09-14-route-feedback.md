# 外繞 feedback、FLUTE 與獨立 routing 驗證

已完成外繞評估到實際位置更新的閉環、FLUTE 拓撲與共享分支計數，以及 OpenROAD GRT 檢查。**目前沒有證據支持實際 routing IO 改善**：GCD 兩個 seed 的 evaluator 改善未通過 router 驗證，最終均保留 baseline。Bigblue4 有很小的 evaluator 改善，尚無 router 驗證。

## 閉環結果

所有 placement 工作使用 `CUDA_VISIBLE_DEVICES=1`。區域為 K32 slicing、region seed 0、lattice 512。每輪有獨立的額外合法化對照，候選更新不得改變固定節點；完整網表 HPWL 與 cohort 路徑長的 +5% 限制都相對原始 baseline，不能逐輪累積放寬。FT 另列，不抵銷 IO 增加。

| 案例 | 完整網表 legacy IO | 固定 cohort 外繞 IO | HPWL 變化 | OpenROAD GRT IO | 最終解讀 |
|---|---:|---:|---:|---:|---|
| GCD seed1000、32-lattice 交換 | 396 → 392 | 186 → 185 | +0.08070% | 414 → 415 | router 否決，保留 baseline |
| GCD seed1001、相同設定 | 390 → 386 | 185 → 185 | −0.02526% | 407 → 407 | 無 routing IO 改善，保留 baseline |
| Bigblue4 seed1000、8-lattice 交換 | 166,268 → 166,263 | 58,438 → 58,437 | +0.000123% | 未執行 | 僅預測改善，幅度約 0.003% |

GCD 每個 seed 的 feedback cohort 是相同的 366 個 degree-two nets；排除 160 個其他 degree nets 與 53 個基準 pin 位於 core 外的 degree-two nets。排除 nets 仍納入完整網表接受檢查。Bigblue4 使用全部 1,552,439 個 in-die degree-two nets，另有 677,447 個其他 degree nets。Bigblue4 的兩輪閉環耗時 175.3 秒（包含讀取、合法化、完整 GPU 評估、FLUTE diagnostic 與 checkpoint 儲存），只測一個 seed。

先前直接沿轉折移動端點的 `bend` 組，在 GCD 與 Bigblue4 都使合法化後的 IO 變差，沒有接受更新。GCD 的 8-lattice 等尺寸交換組也沒有改善；因該上限不足一個 row 高度，另外明確命名 32-lattice 探索組，再用 seed1001 確認。這是探索式實驗，並非事先凍結來源與參數的 preregistration。

## 更新訊號與防護

`RouteFeedback` 在固定 degree-two cohort 上比較兩種 L、H-V-H／V-H-V 及外繞候選，以 crossing savings 加權端點到轉折的位移。這個位移只是提案方向，不是假造的離散 crossing 梯度。

`cost_delta_swap` 沿提案方向尋找等尺寸 movable cells，以所有 incident nets 的重新計算成本差選擇交換；每個候選重建 MST 與外繞路徑，同一 cell 的多個 pins 一起移動。每輪最多檢查 512 個 active nodes、每個目標 8 個鄰近位置；incident degree >256 的交換直接排除。等尺寸交換保留既有合法矩形，並由 DREAMPlace legality checker 驗證。每輪另跑 extra-LG 對照。接受條件要求完整 legacy IO 優於 incumbent 與對照，且 cohort `(IO, wirelength)` 嚴格改善；若 IO 不變而線長降低，僅稱線長改善。

每個 baseline、對照、候選、接受點都有座標 NPZ、metrics 與 hashes。`result.json` / `selected.npz` 是 evaluator 選擇；獨立驗證後的正式選擇是 `router_selected.json` / `router_selected.npz`。Router gate 只能確認 evaluator 已接受的 checkpoint，不能把先前被拒絕的候選選回來。兩個 GCD 組均已輸出 baseline 作為 router-selected placement。

## FLUTE 與共享分支

使用 repository 內 DREAMPlace 所附的真正 FLUTE C++ 與 POWV9／POST9 LUT，透過小型 ctypes bridge 呼叫；沒有以 MST 或自製 heuristic 冒充 FLUTE。每次建樹做座標平移／量化、int32 範圍檢查、重複 pin 處理，並將量化端點接回原始 pins。來源、LUT、compiler、library hashes 由 backend 提供。CPU diagnostic 支援最多 256 個不同的量化 pins。

每條 net 的共線重疊分支取幾何聯集後再計算線長與 crossings。不同實體 tracks 不合併；OpenROAD 結果還保留不同 layers。外繞的 net union 必須符合相對 baseline union 的 +5% 線長限制，不能只靠逐分支限制推論去重後也符合預算。

| GCD baseline、多 pin cohort | MST union WL | FLUTE union WL | 變化 | MST / FLUTE union IO |
|---|---:|---:|---:|---:|
| seed1000、143 nets | 9,990.789 | 9,498.592 | −4.925% | 159 / 151 |
| seed1001、143 nets | 10,051.145 | 9,542.024 | −5.065% | 157 / 155 |

數字是相同 placement 上的拓撲估計，單位是 DREAMPlace internal coordinates，不是實際 routed wirelength。FLUTE 尚未成為大規模 GPU placement 的梯度項；閉環位置提案仍使用 degree-two cohort，FLUTE 用於 multi-pin evaluator 與 checkpoint 對照。

## 障礙物／容量與獨立驗證的範圍

`RoutingResources` 可對 2-D lattice edges 提供容量與背景需求。容量零代表不可通行，正容量使用額外一單位需求的平方 utilization 成本；候選搜尋納入所有 lattice 中心 tracks。若預算內沒有可行 L/Z 候選，明確失敗，不回傳穿越障礙的假解。這是**凍結背景需求**模型，尚未在 nets 之間保留容量，也沒有 layer／via／DRC 模型。共用樹與此資源模式不能直接混用；目前實際閉環實驗仍用 obstacle-free opportunity estimator。

OpenROAD GRT 完全不接收 region partition：先清除輸入 DEF 遺留的 signal routing，檢查 placement，再執行 50 次 congestion iteration 上限的正常 global routing，未使用 `allow_congestion` 或 infinite capacity。輸出原生 route segments，逐 net／layer 去重後計算 crossings，並以 tool／LEF／DEF／segment hashes 驗證重用。兩個 GCD seed 的比較均固定 563 個共同 routed nets，沒有以較少的 routed nets 假造改善。

placement core 外的 GRT endpoints 有另外標記。全共同 cohort 的 lattice 計數沿用邊界 clamping；另提供所有 checkpoint 都在 core 內的共同子集（seed1000：499 nets；seed1001：500 nets），router gate 要求兩個口徑都改善。seed1000 的 core 內 IO 同樣增加（358 → 359），所以全量結果的失敗並非只由 core 外的 endpoints 造成。GRT 包含 routing/congestion 處理，但這次沒有執行 detailed routing 或 signoff。參考 [OpenROAD GRT 文件](https://openroad.readthedocs.io/en/latest/main/src/grt/README.html) 與 [FLUTE 論文](https://home.engineering.iastate.edu/~cnchu/pubs/c33.pdf)；本次直接存取論文 URL 被站方拒絕，實作以本地 FLUTE 原始碼與測試驗證。

## 重現與驗證

```bash
source src/scripts/env.sh
source src/scripts/openroad_env.sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$IOPLACE_PYTHON" src/scripts/run_route_feedback.py \
  --config results/route_feedback_20260914/gcd.json --seed 1001 \
  --max-displacement-cells 32 --emit-def --out results/my_new_route_feedback
"$IOPLACE_PYTHON" src/scripts/verify_route_feedback_grt.py \
  --run results/my_new_route_feedback --config results/route_feedback_20260914/gcd.json \
  --openroad "$OPENROAD_BIN"
```

既有合法 checkpoint 可加 `--placement PATH.npz`；測試原始轉折位移用 `--proposal-mode bend`。輸出目錄必須是新的。`--reuse-verified` 僅在 router 輸入與輸出 hashes 全部符合時重用既有 routing，重算分析與 gate。

最後合併驗證：**142 passed、1 skipped、5 slow deselected（18.64 秒）**。包含 19 個新增／擴充 targeted tests，以及既有 evaluator CPU/GPU、discrete CE／postprocess、route decode／crossings 與 tradeoff 回歸。此次沒有重新完成整個 repository 的 slow suite。索引 generation 仍為 `2026-09-14T17:14:02Z`，未追蹤 `src/` 搬移後的路徑；已按 coverage 建議直接讀取相關原始碼。

原始 artifact 位於 `results/route_feedback_20260914/`：保留 `gcd_seed1000`、`gcd_swap_seed1000`、`gcd_swap32_seed1000`、`gcd_swap32_seed1001`、`bigblue4_seed1000` 與 `bigblue4_swap_seed1000` 的全部成功／失敗提案。第一個 OpenROAD 嘗試因 Tcl command namespace 錯誤而失敗，log 留在 `grt_failed_command`；修正為 `grt::write_segments` 後重新 routing，正式 receipt 均成功。

最終來源快照：`results/route_feedback_20260914/final_source.tar.gz`；132 個來源檔案與 FLUTE provenance 存於 `final_source_manifest.json`。`artifact_manifest.json` 記錄結果與選定座標 hashes；`validation_tests.log` 保存最後合併測試輸出。來源快照是完成時版本，不追溯宣稱探索組執行期間已凍結。
