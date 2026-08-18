已核對目標檔與 commit `0bd96118450eaae36b8e26c3786beaad11207cce` 完全一致。以下不採信任何 `/tmp` 實測，只審查其推論、程式語義與 task/gate 是否成立。

1. [BLOCKER] 五點 GP 模型不足以支撐 27.7M 的容量與 runtime 裁決

   原文（§1.1、§5.2）：「`GP_peak_bytes ≈ 87·N_total + 73·N_pins + 160·n_bins`（五點，最大誤差 4.3%）」；「27.7M GP ≈ 890 s」。見 [M4 draft §1.1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:41) 與 [§5.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:283)。

   證據：

   - 五點擬合三個係數、無 intercept，只剩兩個 residual degrees of freedom；4.3% 是 training residual，不是 prediction error。
   - `N_total`、pins、bins 都隨 case 規模同步增加，無法識別 87/73/160 各自是否穩定。橫跨很大規模不會消除 multicollinearity。
   - case 同時改變資料格式、filler 比、bin resolution 與設計拓撲；模型未做固定 case、單獨改 bins/target density 的 factorial probe。
   - 890 s 並不是一個已定義的 runtime fit，而近似把 cluster 時間按 pins 放大；沒有 iteration 數、停止條件、FFT/WL kernel scaling 或 prediction interval。

   具體修正：T0 必須輸出 design matrix、單位、condition number、係數不確定度與 leave-one-case-out 殘差。用同一 case 獨立掃 bins、filler/target density；把 6.2M、12.3M 當 holdout，通過後才能外推 27.7M。Runtime 另建 `iterations × per-iteration cost + fixed phases` 模型，不得從 memory fit 或單一比例推出。

2. [BLOCKER] 「不共時但相加」既不能證明可跑，也不能證明跑不了

   原文（§2.2）：「GP 的峰值與 op 的峰值不同時發生……但為安全起見一律用加總」；接著以 14.7 GiB 宣稱 11.3M 可行、31.1 GiB 宣稱 27.7M GPU 也擋。見 [§2.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:102)。

   證據：

   - 獨立 run 的 absolute peaks 相加只是很鬆的上界。`sum > capacity` 不能推出實際 OOM；`sum < capacity` 也不能涵蓋 allocator reserved、碎片或未計入的共存狀態。
   - 真實 driver 在 placement 前同時建好 `GpuEvalContext` 與 `IoTerm`，兩者靜態 buffers 會跨 GP 存活，見 [run_placement_io.py:46](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement_io.py:46)、[line 63](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement_io.py:63)、[line 93](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement_io.py:93)。因此正確帳是「共同常駐 + 各 phase transient 的最大值」，不是三個 standalone absolute peaks 的和。
   - 21.7 GiB cap 與各表「GB」混用；`mem_get_info()` 是裝置 free memory，而 `max_memory_allocated()` 是 PyTorch active tensor peak。若 free 是 CUDA 初始化後取得，再扣 context 會重複扣除；草案沒有定義量測時點。

   具體修正：新增 integrated lifetime probe，記錄每個 buffer 的 create/destroy phase、phase-start allocated/reserved、phase absolute peak及增量。可行性只由 full-lifetime spike 判定；standalone sum 只能列為保守上界，不能當 blocker 或 feasibility proof。

3. [MAJOR] Host RAM 與 IoTerm 的「per-pin 模型」沒有足夠 predictor

   原文：「host RSS ≈ 0.88 KB/pin（Bookshelf）」及「84 B/pin，記憶體純線性於 pin 數」。見 [§1.1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:41) 與 [§1.3](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:67)。

   證據：

   - 表中 Bookshelf host RSS 只有 bigblue4 一個有效點；從一個 ratio 外推到 108M pins 不是模型。
   - PlaceDB host footprint 至少含 nodes、nets、pins、名稱字串、rawdb/parser 結構；3×3 前綴會改變名稱長度，不能只按 pins。
   - IoTerm 的主規模量是 deduplicated `P′`、physical nodes、active nets 及 `k_chunk`，見 [io_term.py:329](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:329)，不是 raw Bookshelf pin 數。
   - [spike_10m.json](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m2/spike/spike_10m.json) 的 target 是 40M pins、實際為 49,019,855；它也沒有 commit/env/input hash 或目前程式會輸出的 `peak_reserved_gb`，不能支撐精確係數。

   具體修正：模型至少用 `N_nodes + N_nets + N_raw_pins + N_dedup_pins + total_name_bytes`。在真實 group、1×2、2×2 產物上量測；Host RAM blocker 必須由 2×2 holdout 殘差與上置信界推出。

4. [BLOCKER] 12.3M/27.7M 的 nets/pins 帳完全漏掉 glue，並使 int32 設計可能溢位

   原文（§2.1）把 27.7M 寫成 31.54M nets、108.2M pins；T4 驗收卻明文要求「`4×來源 + glue`」。見 [§2.1 表](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:90) 與 [T4](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:392)。

   證據：

   - `3,503,992 × 9 = 31,535,928`、`12,026,191 × 9 = 108,235,719`，恰好就是表中數字；因此表只算 replication，glue nets/pins 為零。
   - 同一問題存在於 2×2：14,015,968 nets、48,104,764 pins 也正好是 4×來源。
   - T2 同時提議 `unique(net·64+rid)` 與「索引 int64→int32」，見 [§4.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:228)。signed int32 key 最多容納 33,554,431 nets；3×3 base 只剩 2,018,503 glue-net headroom。任何超過約 6.4% 的 glue 就溢位。

   具體修正：T6 後新增「count freeze」task，以 manifest 實際 glue net/pin 數重算所有 GPU/host/runtime 帳。Composite key 保持 int64，或在建構時 assert `n_nets <= INT32_MAX//64`；不能把所有 index 一概降成 int32。

5. [BLOCKER] `γ_far` 在 2×2 calibration 中完全不可識別，且 3×3 的核未定義完整

   原文（§3.2）：「`φ(1)=1`、`φ(√2)=γ_diag`、`φ(2)=γ_far`」；「`(γ_diag,γ_far)` 由 2×2 的 V2 校準決定」。見 [§3.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:165)。

   證據：

   - 2×2 tile pairs 只有距離 1 與 √2；不存在距離 2。因此 `γ_far` 對所有 2×2 觀測量的導數都是零。
   - 3×3 除 1、√2、2 外還有 √5、√8；草案沒有定義 `φ(√5)` 或 `φ(√8)`。
   - `γ_far` 首次影響的正是被外推的 3×3，卻沒有任何 real target 約束。

   具體修正：改成一個對所有距離定義完整的單參數核，例如 `φ(d;α)`，以 2×2 的 diagonal/adjacent 統計估 α，讓遠距值由公式導出；否則固定 `γ_far=0` 並明確承認 3×3 不模擬 long-range connectivity。

6. [MAJOR] V2/V4 是調參目標，不是外部驗證；目前是 scan-until-pass

   原文：「不是自由參數：調到 2×2 的 Rent p 與 cut 統計對上真實 cluster 為止」；T6：「掃 `(γ_diag,γ_far)` 直到 V2 通過」。見 [§3.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:181) 與 [T6](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:394)。

   證據：

   - 同一 real design、同一 2×2 scale、同一 Rent/cut aggregate 同時用來選參數與宣稱 generator 有效，沒有 held-out evidence。
   - V4 的 `[0.5,2.0]` 容忍範圍過寬，且與 Rent/cut 都主要反映 glue 量，並非獨立辨識方向。
   - glue pin 抽樣有 seed noise，T6 沒要求多 seed、CI 或穩定參數區間。
   - V5 說「跨 tile net 佔總 net 比例成長符合 `(RC)^p`」，見 [V5](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:192)。若跨 tile 數量按 `m^p`、總 nets 按 `m`，比例應涉及 `m^(p-1)`；目前量綱也未定義。

   具體修正：預先拆成 fit metrics 與 holdout metrics。可用 pair-count by distance/degree 作 fit；Rent curve shape、per-level cut histogram、interface-cell spatial distribution作 holdout。固定 search grid、seed 集、bootstrap CI，失敗時不得繼續掃到通過。

7. [BLOCKER] 「cluster = 4 groups」尚未成立，卻已成為整個 synthesis ground truth

   原文（§3.1）：「`11,310,807 / 3,077,669 = 3.67 ⇒ mempool_cluster 就是 4 個 mempool_group + group 間互連`」。見 [§3.1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:142)。

   證據：

   - Cell-count ratio 只能提示可能有四份階層；不能證明四份與 standalone group 同構，也不能排除 synthesis pruning、共享 top-level logic、不同參數化或 duplicated infrastructure。
   - 草案自己在 L9 承認沒有 RTL hierarchy 直接驗證，見 [M4-L9](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:482)。
   - T4 的正式驗收沒有 instance-prefix、每 group node/net/pin 數或 standalone-to-subhierarchy correspondence gate。

   具體修正：新增 blocking T3a：解析 cluster hierarchy，證明四個 group prefixes；逐 group 報 cells/nets/pins、cell-type histogram、interface-net classification，並與 standalone group 比對。只有通過後才能產 `cluster_stats.json` 或啟動 T4/T6。

8. [MAJOR] 「30M 只作 scaling」沒有由 task/artifact 結構強制執行

   原文（§3.4）：「所有 IO/FT 品質宣稱一律限定在三個真實 case」。見 [§3.4](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:205)。

   證據：

   - T8 把 synthetic 6.2M 放入 `{flat, ours}` 全流程矩陣，見 [T8](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:396)。
   - E3 要求它進 `full_scale_comparison.md`；表 schema 又含 `Δio%/Δft%/Δhpwl%`，見 [§6.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:345)。這已是 placement-quality comparison。
   - 沒有 report linter 或 `benchmark_kind` 防止 synthetic rows 被畫進 quality Pareto 圖或 aggregate claim。

   具體修正：拆成 `quality_real_cases.md` 與 `scaling_synthetic_cases.md`。Synthetic artifact 禁止 `Δio/Δft`、winner、Pareto 等欄位，只報 throughput、memory、completion與結構檢查；加入自動 schema gate。

9. [MAJOR] int8 改法數學上可保義，但 M3/M4 分工邊界寫錯

   原文（§4.2）：「兩個累加器 `int64→int8`，風險極低、reduce 方式不變」。見 [§4.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:228)。

   證據：

   - `pin_bits` 目前由 `_one_hot_planes` 產生 int64，見 [evaluator_gpu.py:203](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_gpu.py:203)，再 scatter 到 int64 accumulator，見 [line 351](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_gpu.py:351)。
   - `passed_bit_acc` 同樣接收 int64 `seg_bits`，見 [line 313](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_gpu.py:313)。
   - PyTorch 2.8 實際要求 `self.dtype == src.dtype`；只把 accumulator 改 int8 會報 `scatter(): Expected self.dtype to be equal to src.dtype`。
   - 若 source 也直接生為 int8，0/1 上的 `amax` 確實等價於 OR，且 `_pack_bits` 與 int64 powers 相乘會 promotion 回 int64。因此語義可保留，但改動不只兩個 accumulator。
   - 若先產 int64 再 `.to(int8)`，兩份 tensor 會短暫共存，可能不降 peak；若直接把 `(P,K)` source 改 int8，又已侵入 M4 所稱的 one-hot memory work。

   具體修正：把 M3 scope 寫成「source bit planes + accumulator 成對改 dtype」，並重算記憶體收益。新增 CUDA K={1,32,64}、multi-chunk、空 bucket 與 legacy-field exact regression。

10. [BLOCKER] Edge batching 不可能保證所有輸出逐位元相同

   原文（T2 驗收）：「batch size ∈ {1e6,8e6,全部} 三種設定結果逐位元相同」。見 [T2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:390)。

   證據：

   - `tree_wl` 現在對所有 edges 做單次 float64 reduction，見 [evaluator_gpu.py:394](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_gpu.py:394)。分 batch 後若累加 batch sums，浮點加法順序改變，不能一般性 bit-identical。
   - 檔案自己明說只有 integer outcomes 必須 exact，`tree_wl` 是 sum-order-sensitive，見 [evaluator_gpu.py:8](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_gpu.py:8)。
   - 現有測試對 `tree_wl` 使用 `pytest.approx(rel=1e-5)`，見 [test_evaluator_gpu.py:41](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/tests/test_evaluator_gpu.py:41)。

   具體修正：驗收分欄位：crossings、FT、lambda、pair counts、integer per-net fields 必須 exact；HPWL/tree_wl 使用預先固定的 abs/rel tolerance。若產品真的要求 bit identity，必須使用 batch-size-independent canonical reduction tree，而不是普通 incremental sum。

11. [MAJOR] B1 污染結論成立，但「固定步進證明 cumulative HWM」的因果論證不成立

   原文（§1.4 B1）：「第一臂逐位元相同 + 後續固定步進 ⇒ process 累積 high-water mark」。見 [B1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:76)。

   證據：

   - 兩個 driver 的確沒有 reset，見 [run_placement.py:129](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement.py:129) 與 [run_placement_io.py:202](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement_io.py:202)。
   - Ablation 確實在同一 process 依序跑 arms，見 [run_ablation_m2.py:86](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/diagnostics/run_ablation_m2.py:86)。
   - 但 `max_memory_allocated` 是「歷史最大 active allocation」，本身不是每 run 自動相加。固定增量更像 retained tensors/reference cycles/driver teardown 缺失；只加 reset 不會清除仍存活的 allocations。
   - Runner 會跳過已有檔案，resumed run 的真實執行順序無 provenance，不能只由 filenames 重建。
   - 因此「M2 數字受 process/order 污染、必須作廢」合理；「原因已證明且 reset 足以修好」不合理。

   具體修正：T1 加 controlled A/B probe：fresh subprocess、same-process、reset-only、explicit teardown/GC 四種；記錄 run-start/end allocated/reserved。正式 ablation 每 arm 用 subprocess，或證明 teardown 後 baseline 回到容忍範圍。

12. [MAJOR] B2 的 8 次 backward 對目前兩個 case 成立，但不是完整成本模型；所提 reuse 也不成立

   原文（B2）：「diagnostics 7 趟 + io_grad_l1 1 趟 ⇒ 8×」；T1：「讓 `io_grad_l1` 復用 diagnostics 的那一次 backward」。見 [B2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:81) 與 [T1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:389)。

   證據：

   - 定義有七個 buckets，見 [io_term.py:16](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:16)。
   - diagnostics 對每個非空 bucket 各做一次 full fwd+bwd，見 [io_term.py:386](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:386)。adaptec1、bigblue4 七桶皆非空，所以當前是 7+1。
   - 一般公式應是 `1 + n_nonempty_buckets`，不是固定 8。
   - diagnostics 之前另有 substantial no-grad softmax/chunk pass，見 [io_term.py:364](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:364)，callback 還有 WL backward。因此 wall-time 大於單純 `8 × op fwd+bwd`。
   - Bucket-masked gradients 的 L1 norms 不能還原完整 gradient 的 L1，因跨 bucket 座標梯度可能相消；不存在可直接拿來供 `io_grad_l1` 復用的「那一次 backward」。

   具體修正：先決定 diagnostic 語義。若要 exact bucket gradient share，保留多 backward 但降低頻率；若要一個 backward，需改成可從同一 full gradient/forward attribution取得的新 metric，並承認不是原本的 per-bucket gradient norm。T1 驗收要量實際 callback wall-time，而非只數 backward。

13. [MINOR] B3 修法會把固定錯誤改成可任意調高的 gate

   原文：「把 8 GB 參數化成 `--budget-gb`」。見 [B3](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:82)。目前程式確實寫死 8 GiB，見 [spike_10m.py:116](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/diagnostics/spike_10m.py:116)。

   問題是 T9 只要求輸出 `ok`，沒有固定 27.7M budget 或來源；使用者可傳任意大值讓 gate 綠燈。

   具體修正：結果同時保存 `measured_peak`、`budget`、`budget_source`、`baseline_reserved`；L4/H100 各自的 budget 在執行前寫死。禁止 CLI budget 超過該 run manifest 的硬體契約。

14. [MAJOR] 新 profile 基建仍重複 B1 類型的量測錯誤

   原文（§6.1）：「`ru_maxrss` 在 phase 邊界各取一次」；「每 0.5 s 用 `mem_get_info()` 取 device used」。見 [§6.1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:320)。

   證據：

   - `ru_maxrss` 是 process-lifetime high-water mark，不能 reset；phase 邊界取值只會單調不減，無法得到 per-phase peak。
   - `mem_get_info()` 的 `total-free` 是整張 GPU 的使用量，不是此 process；其他 process 會污染。
   - 0.5 s polling 可能漏掉短 evaluator/op transient。
   - `reset_peak_memory_stats()` 只重設 PyTorch allocator statistics，不解決外部 CUDA allocations、reserved fragmentation 或 host high-water。

   具體修正：正式 memory run 要求 exclusive GPU，記錄 device baseline；每個要獨立歸因的 phase 用 child process，或以明確 liveness instrumentation 分解。Host 以 child-process peak RSS 取得；GPU 同時報 active、reserved、device delta，且 sampler accuracy 要用人工短峰 probe 驗證。

15. [BLOCKER] E5 的容忍帶幾乎不可證偽，頻寬 heuristic 也不是 end-to-end 模型

   原文（E5）：「預測 0.5–1.5 h；落在 `[0.3×,3×]` 外才算模型錯」；推導為 H100/L4 bandwidth ratio × 0.5–0.7。見 [E5](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:361)。

   證據：

   - `[0.3×,3×]` 本身上下界相差 10 倍。若套到原 interval 兩端，實際 pass band 是 0.15–4.5 h，相差 30 倍。
   - 只要「在數小時附近」幾乎都會通過；它無法區分 2×、5×、8× 的性能模型。
   - gather + FP64 atomic 的 scaling 取決於 atomic throughput、coalescing、occupancy、clock、kernel launch與 contention，不等於 peak bandwidth 固定比例。
   - End-to-end 還含 host parse、GP/FFT、evaluator及 I/O，應使用 Amdahl 分解；草案直接把 op speedup 套到整體。
   - 推導指定 H100 SXM 3.35 TB/s，但資源契約只寫 H100 80GB，未釘死 SKU、功耗模式、CUDA build或時鐘。

   具體修正：先用 L4 profiler量每個 phase 比例及 achieved bandwidth，預先登錄 point estimate + 一個硬 prediction interval，例如 median `[0.7,1.3]×`；刪除第二層 0.3–3 tolerance。若目前無 H100，E5 應標為 forecast deliverable，不是 M4 exit；另設實際 H100 validation task。

16. [BLOCKER] DAG 與驗收條件沒有保證宣稱的成果會被真正執行

   原文（§7）：「T2 是一切 ≥5M 前置」及「只有 T8/T11 的 M3 欄位 dependent」。見 [依賴圖](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:403)。

   證據：

   - T8 包含 synthetic 6.2M，必須依賴 T4 tiler；若要使用 calibrated glue 還必須依賴 T6。但 DAG 沒有 T4/T6→T8。
   - `ours` 沒定義是 M2 還是 M3。若是 M3，依賴的不只是兩個 report 欄位，而是 M3 evaluator、objective、driver與選出的 `f_ft`；若是 M2，就不能稱為完整 M3 scale-up。
   - 「per-case `ρ*` calibration」被塞在 T8 內容裡，沒有獨立 task、seed split、winner-bias 控制或 acceptance。
   - E1 要求 `num_unplaced_cells == 0` 與 final overflow，但 §6.2 JSON schema及目前 driver 都沒有這些欄位，見 [E1](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:357)。
   - T8 驗收只是「每格有 JSON」；空、失敗、NaN、錯 commit 或重複引用同一 run 都可能通過。
   - E4 允許 GP 三係數模型只用四點重擬合，幾乎沒有 falsification power。
   - T10 只有 3.1M runbook rehearsal；沒有真正 H100 execution/validation task，因此 E5 可永遠不被檢驗。

   具體修正：重畫 DAG：

   `T0 → hierarchy gate → tiler/model → calibration+holdout → count freeze → T2 integrated memory gate → per-case parameter calibration → full-flow experiments → H100 execution → report audit`

   並讓每個結果 gate 驗證 `status=success`、schema、finite values、input hash、repo/DP commit、command、hardware、seed及唯一 run ID。E1 所需 legalization/overflow 欄位必須在 T1/T8 明確新增。

## Verdict

v1 不能直接升為 v2。至少以下 BLOCKER 必須先改設計，而不是留給實作時補：

- Finding 1–2：廢除目前的 GP 外推裁決與 standalone-peak 加總可行性判斷。
- Finding 4–5：把 glue 實際 counts 納入所有帳，解決 `γ_far` 不可識別及距離核未定義。
- Finding 7：把「cluster = 4 groups」改成 T4 前的 blocking hierarchy gate。
- Finding 10：改成 field-specific exact/tolerance streaming 契約。
- Finding 15：重寫 E5，移除 10×/30× 容忍帶並增加真正 H100 validation task。
- Finding 16：修正 DAG、M3 arm 定義與不可被空 JSON 滿足的 acceptance。

另外 Findings 3、6、8、9、11、12、14 都應在 v2 同步修正文案與 task acceptance；否則即使 blockers 表面修完，T0/T1/T6/T8 仍可產出「全綠但不能支撐結論」的結果。Finding 13 可不阻擋 v2，但 budget 必須在 T9 執行前預先固定。