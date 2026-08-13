# M4 設計草案 v1:規模化(1M → 30M)

- 日期:2026-08-13
- 狀態:**v1 草案(未定稿)**——待 Codex 對抗性審查後修訂為 v2
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §7(benchmark 計畫)、§8(實驗設計)、§9 M4 列、§10 風險 2、D6(硬體)
- 相依草案:`docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`(§7 G6 8GB 契約、§7 R2 evaluator 斷點、T1 分工邊界)、`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md`(§3.2 環境盤點、S5 = NanGate45 語料接入,**本文與其共用同一個 task,不重複設計**)
- 繼承資產:`ioplace/ops/io_term.py` chunked-k 契約、`ioplace/diagnostics/spike_10m.py`、`ioplace/evaluator_gpu.py`、`ioplace/drivers/run_placement{,_io}.py`
- **本文所有標記「實測」的數字,皆為 2026-08-13 在本機 L4 上以 `$DP/.venv312/bin/python` 實跑取得(探針目前在 `/tmp`,T0 遷入 repo 前一律視為「待重現」,沿用 M2/M3 慣例);標記「推估」的一律未驗證。**

---

## 0. 摘要(草案裁決一覽)

| # | 問題 | 草案裁決(v1) | 信心 |
|---|---|---|---|
| Q1 | benchmark 階梯 | **1.3M = ISPD2015 `mgc_superblue12`;3.1M = ISPD2025 `mempool_group`(實測 3,077,669 cells,**不是 spec 寫的 5M**);11.3M = ISPD2025 `mempool_cluster`(實測 11,310,807 cells)。6.2M / 12.3M / 27.7M 由同一個 tiler 從 `mempool_group` 合成。** **實測:`mempool_cluster` 的 DREAMPlace GP+LG 在 L4 上跑得完——362.4 s、GPU 峰值 6,889.6 MiB、host RSS 62.6 GB。**「10M 只能在 H100 跑」的先驗假設**被推翻** | 高(三個規模皆實跑) |
| Q2 | 30M 測資 | **配方 A′:`mempool_group` 的 3×3 陣列(27.70M cells / 108.2M pins)+ 從真實 `mempool_cluster` 量得的 group-間互連統計所生成的 glue nets,在 Bookshelf 域串流合成**(DREAMPlace 原生 `BOOKSHELFALL` writer 可把 LEF/DEF 轉出完整 `.nodes/.nets`,含 per-pin offset)。**關鍵設計:tiler 的驗收不是靠 Rent 自證,而是靠「2×2 陣列(12.31M)vs 真實 4-group 的 `mempool_cluster`(11.31M)」的同尺度對照**——MemPool 的 cluster 就是 4 個 group,所以我們有一個真實的 ground truth 來校準合成器,再外推到 3×3 | 中高(來源實測;glue 模型未驗) |
| Q3 | evaluator 規模斷點 | **M3 draft R2 嚴重低估了。實測 evaluator 記憶體 = (0.85 + 0.061·K) GB 每 M-net ⇒ `mempool_cluster`(12.71M nets, K=32)需要 35.6 GB,L4/40GB-H100 皆 OOM。** R2 點名的兩個 `(E,K)` int64 累加器只佔其中 5.7 GB;**真正最大的單項是 `evaluator_gpu.py:351` 的 `(P,K)` int64 one-hot(11.3 GB)**,其次是完全未分批的 MST edge / segment 陣列。**分工裁決:M3 T1 只做「不改張量形狀語意」的兩項低風險修正(int64→int8 累加器、pair-demand 改 GPU bincount);結構性 streaming 重寫一律歸 M4 的 T2,且 T2 是 M4 全部 ≥5M 實驗的硬前置** | 高(五點線性擬合) |
| Q4 | 10M+ 全流程瓶頸 | **不是 DREAMPlace GP。**GP 峰值 = `87·N_total + 73·P + 160·n_bins` bytes(五個 case 擬合,最大誤差 4.3%)⇒ 11.3M 只要 6.9 GB、27.7M 推估 12.6 GB。三個真瓶頸依序是:**(i) evaluator 記憶體(Q3);(ii) IO op 的 K-pass runtime——實測 30M spike 單次 fwd+bwd 26.7 s,推估單臂 3.3–5.4 h;(iii) host RAM——`PlaceDB.read` 實測 ≈1.4 KB/pin(LEF/DEF)/ 0.88 KB/pin(Bookshelf),27.7M 需要 95–105 GB,超過本機可用的 ~80 GB。** ⇒ **30M 全流程在本機是被 host RAM 擋住,不是被 GPU 擋住** | 高(三項皆實測) |
| Q5 | 計時與 profile | 量 6 個 phase 的牆鐘 + **每 phase 各自 reset 的** GPU 峰值 + host peak RSS + per-iteration CUDA-event 直方圖 + op fwd/bwd 分離計時;**零新依賴**(`torch.cuda.mem_get_info()` 取代 NVML,`resource.getrusage` 取代 psutil)。**exit 改寫為「L4 可驗證的部分 + 一個預先登錄的 H100 預測」**:M4 在 L4 交付 11.3M 全流程對照表與 27.7M 的元件級 spike,並**事先寫死** 30M 在 H100 上的 wall-time 預測區間,H100 一跑就是對這個模型的可證偽檢驗 | 中高 |
| Q6 | task 分解 | **T0 探針遷入 → T1 profile 基建 → T2 evaluator streaming(擋住一切 ≥5M)→ T3 語料接入(= Stage2 S5,共用)→ T4 tiler → T5 Rent/驗收器 → T6 12.3M 校準實驗 → T7 27.7M 生成 → T8 10M 全流程實驗 → T9 30M spike → T10 H100 交接包 → T11 報告**;條件 task T12(numpy-only PlaceDB shim)、T13(fused CUDA op)。**M3-dependent 的只有 T8/T11 的「M3 臂」欄位,其餘全部現在就能做** | — |

**一句話結論:** 今天的實測把 M4 的問題換掉了——**DREAMPlace 在 10M 級沒問題(L4 上 6 分鐘、6.9 GB),我們自己的兩個元件才是規模斷點**:evaluator 在 12.7M nets 要 35.6 GB(必須結構性重寫),IO op 在 30M 要 26.7 s/iteration(K 趟掃 pin 的固有成本)。而 30M 的最終阻擋者是 host RAM(95–105 GB),與 GPU 無關——這正好把「H100 交接」變成一個**可以寫成資源契約與可證偽預測**的東西,而不是一句「等機器」。

---

## 1. 今日實測基準(全部為本次在 L4 上實跑)

環境:`$DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`、`$DP/.venv312/bin/python`、torch 2.8.0+cu128、NVIDIA L4。**注意:`torch.cuda.mem_get_info()` 回報的可用裝置記憶體是 21.95 GiB(不是 23 GB),CUDA context 另佔約 0.2 GiB ⇒ 本文一律以 21.7 GiB 為張量預算上限。**

### 1.1 DREAMPlace GP+LG(det=1、DP 關、fillers 開、**不含 evaluator**)

| case | 格式 | movable | nets | pins | filler | bins | read s | GP+LG s | host RSS | **GPU 峰值 alloc** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 | Bookshelf | 210,904 | 221,142 | 919,701 | 160,067 | 512² | 3.4 | 14.4 | — | **140.8 MiB** |
| mempool_tile_wrap | LEF/DEF | 127,739 | 145,589 | 521,021 | 163,607 | 512² | 2.7 | 7.8 | 1.13 GB | **105.2 MiB** |
| bigblue4 | Bookshelf | 2,169,183 | 2,229,886 | 8,731,365 | 2,885,580 | 2048² | 39.3 | 40.8 | 8.18 GB | **1,670.9 MiB** |
| **mempool_group** | LEF/DEF | 3,077,669 | 3,503,992 | 12,026,191 | 633,128 | 2048² | 73.5 | 92.0 | 16.67 GB | **1,792.6 MiB** |
| **mempool_cluster** | LEF/DEF | 11,310,807 | 12,712,800 | 43,947,793 | 3,889,360 | 4096² | 284.9 | **362.4** | **62.59 GB** | **6,889.6 MiB** |

**擬合模型(五點,最大誤差 4.3%):**

```
GP_peak_bytes ≈ 87·N_total + 73·N_pins + 160·n_bins        (N_total = physical + filler)
host_RSS(PlaceDB.read) ≈ 1.35–1.42 KB/pin (LEF/DEF)  /  0.88 KB/pin (Bookshelf)
```

四個必須寫進報告的次級事實:

1. **mempool 家族沒有 movable macro**(實測 tile_wrap:`n_movable_macro=0`、`max_h_in_rows=1.0`,SRAM 全在 `num_terminals`)⇒ 符合 Phase 1 的 fixed-macro 前提,不需要額外的 macro 凍結步驟。
2. **`mempool_group` 是 3.08M 不是 spec §7 寫的 5M**;`mempool_cluster` 是 11.31M 不是 10M。spec §7 的表要修。
3. **必須 glob 全部 15 個 NanGate45 LEF、tech.lef 在最前**(Stage2 §3.2 已實測;本次三個 config 皆照此產生,零失敗)。
4. GP 的 iteration 預算是 1000(config),`mempool_cluster` 362.4 s ⇒ **每 iteration ≥ 0.36 s**(不含我們的 op)。

### 1.2 evaluator(`evaluator_gpu.GpuEvalContext.evaluate`,合成 netlist、bigblue4 degree 分布、uniform 位置)

| nets | pins | K | **峰值 GB** | eval s |
|---:|---:|---:|---:|---:|
| 0.6M | 2.46M | 32 | 1.72 | 0.78 |
| 1.2M | 4.92M | 32 | 3.38 | 1.02 |
| 2.4M | 9.82M | 32 | 6.74 | 1.73 |
| 2.4M | 9.82M | 16 | 4.39 | 1.60 |
| 4.8M | 19.6M | 32 | 13.46 | 3.15 |

線性完美:**`mem ≈ (0.85 + 0.061·K) GB 每 M-net`**(K-independent 項 = MST edge/segment 陣列與 fp64 座標;K-dependent 項 = 三個 `(·,K)` 張量)。

### 1.3 IoTerm(M2 chunked op,K=32)

| nodes | nets | pins | k_chunk | **峰值 GB** | fwd ms | bwd ms | host RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 12M | 49.02M | 1 | 4.107(既有 `results/m2/spike/spike_10m.json`) | 3,929 | 11,917 | — |
| **30M** | **36M** | **147.1M** | 1 | **12.32**(reserved 15.29) | **5,732** | **20,932** | 11.76 GB |

**兩點的 `峰值 / pins` 完全相同 = 84 B/pin** ⇒ chunked-k 契約在 30M 依然成立,記憶體純線性於 pin 數、與 K 無關。runtime 則是 **181 ns/pin(30M)/ 323 ns/pin(10M)** 每次 fwd+bwd(小 case 的 GPU 未飽和)。

### 1.4 三個必須修的「量測工具本身的 bug」

| # | 問題 | 證據 |
|---|---|---|
| **B1** | `results/m2/**/*.json` 的 `peak_mem_mb` **不是 per-run 峰值,是 process 累積 high-water mark** —— `run_placement.py:129` / `run_placement_io.py:202` 都沒有 `reset_peak_memory_stats()`,而 `run_ablation_m2.py` 在同一個 process 內跑完所有臂 | adaptec1 `A0_k8` = 140.765625 MiB,與本次全新 process 的 GP 峰值**逐位元相同**;其後 k16/k32/slicing = 253.4 / 366.7 / 479.3(定值 +112.6 步進);bigblue4 A2/A3/A4/A6 = 10.58/12.46/14.35/16.23 GB(定值 +1.88 GB 步進)。**M2/M3 報告裡任何引用 `peak_mem_mb` 的敘述都必須撤回或重測** |
| **B2** | driver 每次 callback 呼叫 `io_term.diagnostics()`(`run_placement_io.py:155`),而它在 `io_term.py:386-399` 會**逐 degree bucket 各跑一次完整 chunked fwd+bwd(7 趟)**,再加 `io_grad_l1`(`:145`)1 趟 ⇒ **每個 callback 的 op 成本是 8×**。30M 下這是每 callback 約 213 s | 讀碼 + §1.3 計時 |
| **B3** | `spike_10m.py:116` 的 `ok = peak_gb <= 8.0` 是 M2 針對 10M 定的門檻,30M 下必然 false(實測 12.32)。M4 要把它參數化成 `--budget-gb`,而不是把 30M 的結果當成「失敗」 | 讀碼 + §1.3 |

---

## 2. Q1 — benchmark 階梯與各級的 L4 可行性

### 2.1 定案階梯

| 級 | case | 來源 | cells | nets | pins | **GP 峰值** | **evaluator(現況/修後)K=32** | **IoTerm** | host RAM | L4? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.3M | `mgc_superblue12` | ISPD2015,`$DP/install/test/ispd2015/lefdef/mgc_superblue12.json` 已存在 | 1.287M | ~1.4M(推估) | ~4.5M(推估) | 0.64 GB(推估) | 3.9 / ~0.6 GB | 0.35 GB | ~6 GB | ✅ |
| 2.2M | `bigblue4` | ISPD2005 Bookshelf | 2.169M | 2.230M | 8.731M | **1.63 GB 實測** | 6.2 / ~0.9 GB | 0.68 GB | 8.18 GB 實測 | ✅ |
| 3.1M | **`mempool_group`** | ISPD2025 NanGate45 | 3.078M | 3.504M | 12.026M | **1.75 GB 實測** | 9.8 / ~1.3 GB | 0.94 GB | 16.67 GB 實測 | ✅ |
| 6.2M | group 1×2 陣列 | **本專案 tiler** | 6.16M | 7.01M | 24.05M | 3.07 GB(推估) | 19.6 / ~2.5 GB | 1.88 GB | ~21 GB(推估) | ✅(修後) |
| **11.3M** | **`mempool_cluster`** | ISPD2025 NanGate45 | 11.311M | 12.713M | 43.948M | **6.73 GB 實測** | **35.6 / ~4.5 GB** | 3.44 GB | **62.59 GB 實測** | ✅(**T2 修後**) |
| 12.3M | group 2×2 陣列 | **本專案 tiler**(= cluster 的合成對照組) | 12.31M | 14.02M | 48.10M | 7.3 GB(推估) | 38.8 / ~4.9 GB | 3.77 GB | ~43 GB(推估) | ✅(修後) |
| **27.7M** | group 3×3 陣列 | **本專案 tiler** | 27.70M | 31.54M | 108.2M | **12.6 GB(推估)** | 87.3 / ~10 GB(推估) | 8.47 GB | **95–105 GB(推估)** | ❌ **host RAM 擋** |

`evaluator` 欄「現況」= §1.2 的實測模型直接外推;「修後」= T2 完成後的推估(§4.3)。`IoTerm` 欄 = 84 B/pin × pins(§1.3 的實測係數,K≥8 時 `k_chunk=1` 恆成立)。

### 2.2 每一級的加總帳(L4 上限 21.7 GiB)

GP 的峰值與 op 的峰值**不同時發生**(op 的 chunk 暫存在 backward 結束即釋放,GP 的 density/FFT 峰值在另一個時點),但為安全起見一律用加總:

| 級 | GP + IoTerm + evaluator(修後) | 對 21.7 GiB |
|---|---:|---|
| 3.1M | 1.75 + 0.94 + 1.3 = **4.0 GB** | 餘裕充足 |
| 6.2M | 3.07 + 1.88 + 2.5 = **7.5 GB** | 餘裕充足 |
| 11.3M | 6.73 + 3.44 + 4.5 = **14.7 GB** | **可行,餘裕 32%** |
| 12.3M | 7.3 + 3.77 + 4.9 = **16.0 GB** | 可行,餘裕 26% |
| 27.7M | 12.6 + 8.47 + 10 = **31.1 GB** | **超出 43%,GPU 也擋** |

**⇒ 只有 27.7M 這一級必須上 H100;1.3M–12.3M 全部在 L4 上跑得完(前提:T2 完成)。**

### 2.3 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| **維持 spec §7「5M = MemPool 半 cluster / 3×Group」** | (a) `mempool_group` 實測是 3.08M 不是 5M,「3×Group」= 9.2M 已經是 10M 級;(b)「半 cluster」不是一個可交付的物件(要從 11.3M 的 DEF 切一半,幾何與階層都要重建)。**改用 tiler 產 1×2 = 6.16M**,同一個工具、同一套驗收,零額外設計 |
| **1M 級用 ISPD2005 `bigblue4`(2.18M)取代 `superblue12`** | bigblue4 保留為 **M0–M3 連續性**的 case(所有既有數字都在上面),但它是 Bookshelf、無 metal stack、無法接 Stage 2 校準。`superblue12` 是 spec §7 明列的 1M 主力且 DREAMPlace config 已存在 ⇒ **兩個都留,角色不同** |
| **10M 用合成資料而非 `mempool_cluster`** | 真實 10M 就在本機而且**今天已經跑通**(362 s / 6.9 GB)。用合成的取代它會白白丟掉唯一一個「真實 RTL 互連 + 真實 PDK + 10M」的 case,也丟掉 §3 tiler 的校準靶 |
| **10M 直接上 H100 不在 L4 驗** | 被實測推翻(§1.1)。而且 D6 原本就寫「L4 做全規模功能驗證、H100 跑計時」——現在證據更強:11.3M 連計時都可以在 L4 做 |
| **`bsg_chip`(0.71M)/`NV_NVDLA_partition_c`(0.145M)/`ariane`(0.115M)入階梯** | 規模與 ISPD2015 重疊、不填任何空缺;保留為 Stage 2 校準的多樣性語料(Stage2 §4.1 L2),不進 M4 階梯 |

### 2.4 低信心處

- **M4-L1(中):`superblue12` 的 pins/nets 是推估。** 需要跑一次 `PlaceDB.read` 才有實數(約 30 s)。→ T3 的第一件事。
- **M4-L2(中):`mempool_cluster` 的 GP 品質未驗。** 本次只證明「跑得完、記憶體夠」,沒看 HPWL/overflow 是否收斂到合理值(`stop_overflow=0.07` 是否達到、legalization 是否成功)。**這是 T8 的第一個 gate,不是已完成事項。**
- **M4-L3(高風險):`mempool_cluster` 的 die 面積利用率 / target_density 未調。** `mempool_group` 實測 `area_util=0.704`、`tile_wrap` 只有 0.389;沿用 ISPD2005 的 `target_density=0.835` 在低利用率的 case 上會產生大量 filler(tile_wrap 實測 filler 163,607 > movable 127,739)。→ T3 必須為每個 NanGate45 case 從 `total_movable_node_area / free_area` 反推一個合理的 `target_density`,並在報告揭露。

---

## 3. Q2 — 30M 測資合成

### 3.1 定案:配方 A′ = `mempool_group` 的 3×3 陣列 + 真實 cluster 統計導出的 glue nets,Bookshelf 域串流合成

**為什麼這條路可行(三個實測支撐):**

1. **DREAMPlace 原生就能把 LEF/DEF 轉成完整 Bookshelf。** `PlaceDB.write(params, path, place_io.SolutionFileFormat.BOOKSHELFALL)`(`$DP/dreamplace/PlaceDB.py:1010-1033`)→ `BookShelfWriter::writeAll`(`$DP/dreamplace/ops/place_io/src/BookshelfWriter.cpp:22-51`)寫出 `.nodes/.nets/.wts/.pl/.scl/.aux`;`writeNets`(同檔 `:100-138`)**逐 pin 寫出相對 cell 中心的 offset**,與 ISPD2005 格式一致 ⇒ 轉換無損。`BOOKSHELFALL` 在 pybind 層已導出(`PybindPlaceDB.cpp:60`)。
2. **Bookshelf 比 DEF 省 3 倍以上。** 實測 bigblue4:`.nets` 345.3 MB / 8.73M pins = **39.6 B/pin**;`.nodes` 15.0 B/node;`.pl` 16.5 B/node。⇒ 27.7M cells / 108.2M pins 的 Bookshelf 約 **5.2 GB**(對比 DEF 域要 27 GB)。磁碟 11 TB 可用,無壓力。
3. **MemPool 的階層剛好給我們一個免費的校準靶。** 實測 `11,310,807 / 3,077,669 = 3.67` ⇒ **`mempool_cluster` 就是 4 個 `mempool_group` + group 間互連**。所以我們的 tiler 產出的 **2×2 陣列(12.31M)有一個真實的同尺度對照組**。

### 3.2 合成器規格(`ioplace/bench/`)

```
(a) group_bookshelf/          ← 一次性:PlaceDB.read(mempool_group) → write(BOOKSHELFALL)
                                 host 16.7 GB 實測、約 90 s
(b) cluster_stats.json        ← 一次性:從 mempool_cluster 量「跨 group 的 net 統計」
                                 (階層前綴切 group;per-net 的 (degree, 觸及 group 數, 每 group pin 數) 直方圖)
(c) tile_bookshelf.py         ← 串流 tiler:讀 (a),寫 R×C 陣列,不在記憶體中持有完整 netlist
(d) glue_gen.py               ← 依 (b) 的統計生成跨 tile nets
(e) rent.py                   ← Landman–Russo Rent 量測(Mt-KaHyPar 1.6.2,已安裝)
(f) verify_bench.py           ← §3.3 的驗收
```

**tiler 的硬性規則:**

- 陣列 `R×C`,tile `(i,j)` 的 node/net 名前綴 `t{i}_{j}/`;座標平移 `(i·W, j·H)`,`W,H` = 來源 die 的 `(xh−xl, yh−yl)`。**完全 abutted、無 channel**(對齊 spec §2 的 design style 裁決)。
- `.scl`:來源 row 清單複製 R×C 份並平移;**必須 assert 所有 row 的 y 區間在陣列後不重疊、不留縫**(fully-abutted 的機械檢查)。
- fixed macro(來源的 `num_terminals`)照樣平移並保持 fixed。
- **串流**:逐 tile 逐行寫,峰值 host 記憶體 = O(單 tile 的 node 名表)≈ 1 GB。⇒ **生成 27.7M 不需要 95 GB;只有「用 DREAMPlace 讀回來」才需要。**
- 決定性:單一 `--seed`,輸出附 `manifest.json`(來源 sha256、seed、R、C、glue 參數、輸出各檔 sha256)。

**glue net 模型(本節最脆弱的一環,明確標為設計假設):**

```
從 (b) 取得真實 cluster 的跨 group net 集合 G_real:
    n_pairs_real[a,b]  = 連接 group a,b 的 net 數(a,b ∈ 4 個 group)
    deg_hist_real      = 這些 net 的 degree 分布
    pinshare_real      = 每個 net 在各 group 的 pin 數分布

R×C 陣列上:
    對每一對 tile (u,v),期望 glue net 數 = mean(n_pairs_real) · φ(dist(u,v))
    φ 是距離衰減核,φ(1)=1(相鄰)、φ(√2)=γ_diag、φ(2)=γ_far   ← 三個可掃參數
    每條 glue net 的 degree 與 pin 分配從 deg_hist_real / pinshare_real 抽樣
    pin 的 tile 內落點:從該 tile 內「原本就是跨 group net 的 pin」所在 cell 中抽樣
                       (保留真實的「哪些 cell 是介面 cell」的空間分布)
```

`(γ_diag, γ_far)` 由 §3.3 的 V2 校準決定,**不是自由參數**:調到 2×2 陣列的 Rent p 與 cut 統計對上真實 cluster 為止,再原樣用於 3×3。

### 3.3 驗收條件(機械可判定;`verify_bench.py` 輸出 `results/m4/bench/verify_<name>.json`)

| # | 條件 | 門檻 | 備註 |
|---|---|---|---|
| **V0** 結構完整性 | `.aux` 指到的檔全在;`NumNodes/NumNets/NumPins` 與實際行數一致;無 self-loop;degree ≤ 1 的 net 比例不高於來源;每個 node 至少被一條 net 參照或明列為 unconnected;row 覆蓋無重疊無縫隙 | 全部 true | 純檔案級,不需 DREAMPlace |
| **V1** DREAMPlace 可讀 | `PlaceDB.read` 成功且 `num_movable/num_nets/num_pins` 等於 manifest | 全等 | **27.7M 因 host RAM 只能在 H100/大記憶體機驗;L4 上驗 1×2(6.2M)與 2×2(12.3M)** |
| **V2** Rent 對照(**核心**) | 2×2 陣列(12.31M)的 Rent p vs 真實 `mempool_cluster`(11.31M):`|p_syn − p_real| ≤ 0.05`,且兩者皆落在 `[0.55, 0.80]` | — | 不通過就調 `(γ_diag, γ_far)` 重生成;**這一條沒過就不准生成 3×3** |
| **V3** degree 分布 | 2×2 vs cluster 的 net-degree 分布 KS 統計量 ≤ 0.05;bucket 佔比(用 `io_term.DEG_BUCKET_EDGES` 同一套切法)逐桶相對誤差 ≤ 20% | — | 直接沿用 M2 的 bucket 定義,報表可比 |
| **V4** IO/FT 口徑合理性 | 以 K=16 grid 疊上去,2×2 的 `hard_lambda_sum / n_nets` 與 cluster 的比值落在 `[0.5, 2.0]` | — | 防「naive replication 讓 cut 假性趨零」(spec §7 紀律) |
| **V5** 3×3 外推一致性 | 3×3 的 Rent p 與 2×2 的差 ≤ 0.05;跨 tile net 佔總 net 比例的成長率符合 `(R·C)^p` 的預測 ±20% | — | 只用檔案級統計,不需要讀進 DREAMPlace |

**Rent 量測的實作定案:** Landman–Russo 遞迴二分。用已安裝的 `mtkahypar==1.6.2`(`docs/dev-env.md` 有完整 API 紀錄)遞迴切到 `2^L` 個 block,每層記 `(平均 block 大小 B, 平均外部 terminal 數 T)`,在 `10³ ≤ B ≤ 10⁶` 的層做 `log T = log t + p·log B` 線性回歸,附 bootstrap 95% CI。
**M4-L4(低信心):Mt-KaHyPar 在 27.7M-node / 31.5M-net hypergraph 上的時間與記憶體完全未知。** 緩解:(a) V2/V3 只在 12.3M 上做(那裡 Mt-KaHyPar 較可能撐得住);(b) V5 改用「取樣子超圖(連續 2M cell 視窗)+ 頂層精確計數(跨 tile net 由建構法直接可數)」的混合估計;(c) 若 Mt-KaHyPar 在 12.3M 就 OOM,退回幾何遞迴切分(bisection by median x/y)量 Rent,並在報告註明「幾何 Rent 而非最佳分割 Rent,是上界」。

### 3.4 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| **`mempool_cluster` ×3 幾何複製** | (a) DEF 域:27 GB DEF、讀回需 ~190 GB host RAM,本機與多數 H100 節點都不可能;(b) 即使轉 Bookshelf,33.9M cells / 132M pins ⇒ 116 GB host RAM,比 3×3 group(95–105 GB)更糟;(c) **失去校準靶**——cluster 已經是最大的真實單位,複製它就沒有同尺度的真實對照可比。3×3 group 保留了「2×2 vs 真 cluster」這個驗證迴路 |
| **把 `spike_10m.py` 的合成拓撲升級成完整 Bookshelf/LEF-DEF** | 它的 pin→node 是 `rng.integers` 均勻抽樣(`spike_10m.py:57`),**沒有任何空間/階層 locality ⇒ Rent p→1.0**,直接違反 spec §7 的 `p≈0.6–0.75` 要求。而且今天的 evaluator 探針已經暴露副作用:均勻位置讓 crossing 數暴增(4.8M nets 合成 case 的 `io_count=29.8M`,對比真實 bigblue4 K=16 只有 100,333)⇒ **會系統性高估 evaluator 的成本與記憶體**。保留其原本的角色:**元件級 spike 的合成器**(§6.2 的 T9),不升格為 benchmark |
| **配方 C:ArtNet 直接生成 30M** | 需要 clone GitHub(本機**網路不通**,硬性限制)。列為 future work |
| **配方 B(純 Rent 差額補 glue,不看真實互連)** | 保留為 **fallback**:若 §3.2 (b) 的階層前綴切不出 group 邊界(名稱慣例不符),就退回按 `t·(K·g)^p − K·t·g^p` 掃 `p∈{0.60,0.65,0.70}`。但它有 spec §7 已點名的循環論證風險 ⇒ 只作備案,且報告必須明寫 |
| **把 30M 當品質宣稱的 benchmark** | **明確非目標。** 30M 的跨 tile 連通性——正是 objective 要優化的量——有一部分是我們自己造的。⇒ **30M 只作 scaling/runtime benchmark;所有 IO/FT 品質宣稱一律限定在 `superblue12` / `mempool_group` / `mempool_cluster` 三個真實 case 上。**這一句必須逐字進論文與報告 |

---

## 4. Q3 — evaluator 規模斷點

### 4.1 先修正 M3 draft R2 的量級

M3 draft §7 R2 寫「`(E,K)` int64 累加器在 12M net × K=32 是 3.0 GB 一份」。**這個描述正確但嚴重不完整**:實測(§1.2)整個 evaluator 在 12.71M nets / K=32 需要 **35.6 GB**,而兩個 `(E,K)` 累加器只佔 6.5 GB。逐項拆解(以實測的 2.4M nets / 9.82M pins / K=32 = 6.74 GB 校準,再外推到 `mempool_cluster` 的 12.71M nets / 43.95M pins):

| 項 | 位置 | 形狀 | @2.4M nets | **@12.71M nets(cluster)** |
|---|---|---|---:|---:|
| **pin one-hot** | `evaluator_gpu.py:351` `pin_bits = self._one_hot_planes(pin_rid)` | `(P, K)` int64 | 2.51 GB | **11.25 GB** |
| pin 累加器 | `:352` `pin_bit_acc` | `(E, K)` int64 | 0.61 GB | 3.25 GB |
| passed 累加器 | `:413` `passed_bit_acc` | `(E, K)` int64 | 0.61 GB | 3.25 GB |
| MST edge 陣列 | `:385` `_batch_mst` 的 `torch.cat` + 各 bucket list | 3×`(M,)` int64 ×2 份 | 0.36 GB | 1.9 GB |
| edge 端點座標 | `:392-397` `xa/ya/xb/yb` fp64 + `ax/ay/bx/by` int64 | 8×`(M,)` | 0.47 GB | 2.5 GB |
| segment 描述 | `:400-406` `h_row/h_lo/h_hi/v_col/v_lo/v_hi` | 6×`(M,)` int64 | 0.36 GB | 1.9 GB |
| pair-demand key | `:326` append + `:332` `torch.cat(...).cpu()` | O(總 crossing 數) | 0.12 GB | 位置相依,真實 case 小,合成 case 可爆 |
| segment chunk 暫存 | `:177` `_seg_chunk_budget = 8e6` | 有界 | ~0.45 GB | ~0.45 GB |
| 其他 `(E,)`/`(P,)` | — | — | ~0.25 GB | ~1.3 GB |
| **合計** | | | **~5.7(實測 6.74)** | **~26–36 GB** |

### 4.2 定案:M3 T1 與 M4 T2 的分工邊界

**裁決:兩邊都做,但切在「是否改變張量形狀語意」這條線上。**

| | M3 T1(語意擴充,順手修) | **M4 T2(結構性 streaming,M4 自己做)** |
|---|---|---|
| 範圍 | ① `(E,K)` 兩個累加器 `int64 → int8`(實測 `scatter_reduce_(reduce="amax")` 支援 int8/uint8/int16/int32 on CUDA);② `pair_key_chunks` 改成 per-chunk `torch.bincount(key, minlength=K*64)` 累加,刪掉 `:332` 的 `torch.cat(...).cpu().numpy()` | ③ 消滅 `(P,K)` one-hot(改成 K 分塊 int8,或 `unique(net·64+rid)` 後 `index_add(1<<rid)` 的精確 OR);④ **MST edge / segment 的批次化**(以 edge 為單位分批,增量累加 `per_net_crossings`/`passed`/`pair_demand`);⑤ 索引 int64 → int32;⑥ 座標 fp64 只在需要位元一致的算子上保留 |
| 收益 @12.71M | 6.5 GB → 0.8 GB,外加移除一次 CPU 同步 | 再省 ~20 GB,把 evaluator 壓到 **~4.5 GB(推估)** |
| 風險 | 極低(dtype 與 reduce 方式不變,`amax` 對 0/1 仍是 OR) | 中(改動 MST/segment 主迴圈,必須靠既有等價測試 + 「舊欄位逐位元不變」測試把關) |
| 為什麼這樣切 | M3 的 exit 全在 adaptec1/bigblue4(≤2.23M nets,evaluator ≤6.2 GB),那裡不是瓶頸;把結構性重寫塞進 M3 會同時提高「M3 T1 改壞舊欄位」的風險——而 M3 T1 本來就要動同一段程式碼加 `io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home`。**低風險兩項順手做,高風險四項留給 M4 專門做一輪 TDD** | |

**M4 T2 是 M4 全部 ≥5M 實驗的硬前置**(依賴序見 §7)。

### 4.3 T2 之後的記憶體推估(**推估**,由 T2 的驗收實測取代)

`mempool_cluster` 12.71M nets / 43.95M pins / K=32:`(E,K)` int8 ×2 = 0.81 GB;pin 座標 fp64 = 0.70 GB;`(E,)` 欄位 ~10 個 int32/int64 = 0.9 GB;edge 批次緩衝(batch = 8M edges)= ~1.0 GB;grid/Ph/Pv(lattice 512)= 6 MB ⇒ **≈ 4.5 GB**。27.7M 合成 case(31.5M nets / 108.2M pins)⇒ **≈ 10 GB(推估)**。

### 4.4 Steiner 欄位(M3 新增)在 30M 的成本

| 欄位 | 形狀 | @31.5M nets |
|---|---|---:|
| `per_net_steiner` | `(E,)` int32 | 126 MB |
| `per_net_home` | `(E,)` int8 | 32 MB |
| `D`、`ell`、`ecc_max` | `(K,K)`/`(K,)` 常數 | < 10 KB |
| Λ≤3 closed form | gather,無新張量 | 0 |
| **Λ∈[4,8] Dreyfus–Wagner DP 表** | **`(B, 2^Λ, K)` fp32/int32** | **必須分批** |

**硬性護欄(寫進 T2/M3 T1 的驗收):** DP 表必須滿足 `B · 2^Λ · K · 4 B ≤ 512 MB` ⇒ `Λ=8` 時 `B ≤ 128k`。實測 Λ 分布極度偏斜(M3 draft §2.2:adaptec1 k16 的 Λ≥4 只有 517 / 216,932 = 0.24%),外推到 31.5M nets 約 **76k 個 net**,一批就夠 ⇒ **成本可忽略,但沒有護欄就是一顆定時炸彈**(K=32 且某個 case 的 Λ 分布較厚時會瞬間要 32 GB)。

**⇒ Steiner 欄位在 30M 的淨成本 < 0.2 GB,不是規模問題。真正的問題全在 §4.1 的表。**

### 4.5 被否掉的修法

| 選項 | 否掉理由 |
|---|---|
| **packed int64 bitmask 直接 `scatter_reduce(amax)`** | **數學上錯的**:兩個 packed mask 的 `max` 不等於 `OR`(例:`0b10` vs `0b01`,max=2,OR=3)。只有在「每列只有一個 bit」時才成立——那正是 `pin_bit_acc` 的情形但**不是** `passed_bit_acc` 的情形(一個 segment 會經過多個 region) |
| **K 分塊(重走 K/c 趟 segment)** | 對 `pin_bits` 可行且便宜;但對 `passed_bit_acc` 要重跑 `_process_segments`(整個 lattice walk)K/c 次,計算成本 ×(K/c)。**改用 edge 批次化**:一趟 walk、增量累加,計算不變、記憶體有界 |
| **只降 dtype、不做 edge 批次化** | 只能救 K-dependent 的部分。§1.2 的分解顯示 K-independent 項是 0.85 GB/M-net ⇒ 31.5M nets 仍要 26.8 GB。**必須做 edge 批次化** |
| **evaluator 改跑 CPU** | `evaluator_ref` 在 bigblue4 上實測要 ~450 s 一次(從 `bigblue4_A0` 的 538.5 s 總時間扣掉今天實測的 40.8 s GP 得出);GP 內圈每 50 iteration 呼叫一次 ⇒ 完全不可能 |
| **降 K 到 8/16 迴避** | 治標;而且 spec §8 的實驗矩陣要求 K ∈ {8,16,32}。降 K 只把 35.6 GB 降到 23.2 GB(K=16),L4 仍 OOM |

---

## 5. Q4 — 全流程瓶頸與本機驗證策略

### 5.1 三個真瓶頸的量化

| # | 瓶頸 | 量化 | 是否擋住 11.3M | 是否擋住 27.7M |
|---|---|---|---|---|
| **P1** evaluator 記憶體 | 35.6 GB @12.71M nets K=32(§1.2 實測外推) | **是**(T2 修後解除) | 是(T2 修後 ~10 GB) |
| **P2** IO op runtime | **181 ns/pin 每次 fwd+bwd**(§1.3 實測)⇒ 11.3M = 7.95 s/iter、27.7M = 19.6 s/iter。K 趟掃全部 pin 是結構成本(`io_term.py:331` 在 P ≫ chunk_budget 時 `k_chunk` 恆為 1) | 否(單臂 ~2.6 h,可接受) | 否(單臂 3.3–5.4 h,勉強) |
| **P3** host RAM | `PlaceDB.read` 1.35–1.42 KB/pin(LEF/DEF)/ 0.88 KB/pin(Bookshelf)。**cluster 實測 62.59 GB** | 否(本機 125 GB) | **是**(27.7M Bookshelf 推估 95–105 GB > 可用 ~80 GB) |

**加上 B2 的放大效應:** driver 每個 callback 跑 8 趟 op fwd+bwd(§1.4 B2)⇒ 11.3M 每 callback 額外 64 s、27.7M 額外 157 s。20 次 callback 就是 21 分鐘 / 52 分鐘的純診斷成本。**T1 必須把 diagnostics 改成可取樣、可關閉。**

### 5.2 單臂 wall-time 推估(L4,`ρ` 啟用約佔 600 個 iteration)

| case | GP | op(600 iter) | evaluator(20 次) | diagnostics(取樣後) | **合計** |
|---|---:|---:|---:|---:|---:|
| `mempool_group` 3.1M | 92 s 實測 | 0.36 h | ~1 min | ~2 min | **≈ 0.45 h** |
| `mempool_cluster` 11.3M | 362 s 實測 | 1.33 h | ~4 min | ~7 min | **≈ 1.6 h** |
| 27.7M 合成 | ~890 s 推估 | 3.27 h | ~10 min | ~15 min | **≈ 3.9 h** |

### 5.3 定案:本機驗證策略(三層)

**第 1 層 — L4 全流程(GP + LG + op + evaluator + 報表),T2 之後即可跑:**
`superblue12`(1.3M)、`bigblue4`(2.2M,M0–M3 連續性)、`mempool_group`(3.1M)、group 1×2(6.2M)、**`mempool_cluster`(11.3M)**、group 2×2(12.3M)。
⇒ spec §9 M4 的「10M 全流程」**在 L4 上就交付得出來**,不需要等 H100。

**第 2 層 — L4 元件級 spike(27.7M),不經 `PlaceDB`:**
用 `ioplace/bench/` 的串流 reader 直接建 `ioplace.netlist.Netlist`(host 記憶體 ~3 GB,實測 30M 合成拓撲的 spike host RSS 只有 11.76 GB),然後單獨量:
- IoTerm(+M3 的 S4)fwd+bwd 的峰值與時間 —— **已經有一次實測:12.32 GB / 26.7 s**;
- T2 之後的 evaluator 的峰值與時間;
- 兩者的加總是否 ≤ 21.7 GiB(目前推估 ~18.5 GB,**會過但無餘裕**)。
⇒ 這一層驗證的是「我們自己寫的兩個元件在 30M 能動」,把 DREAMPlace 的 GP 隔離掉。

**第 3 層 — H100 交接(27.7M 全流程):**
資源契約 + 預先登錄的預測(§6.3)。

### 5.4 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| **「10M 只跑 evaluator+op spike、GP 縮到 5M」**(任務書列的候選) | **被實測推翻**:11.3M 的真實 GP 今天就在 L4 上跑完了(362 s / 6.9 GB)。降級成 spike 會白白丟掉 spec §9 M4 的第一條 exit |
| **為了塞進 L4 而在 30M 降 K 到 8 / 降 bins 到 2048²** | 省不了多少(GP 12.6→11.9 GB、op 的 84 B/pin 與 K 無關),而且 host RAM 才是硬牆。**不值得為此污染實驗矩陣** |
| **T12(numpy-only PlaceDB shim)當主線** | 它能把 30M 的 host RAM 從 ~100 GB 降到 ~3 GB,但 GPU 仍要 ~31 GB > 21.7 GiB ⇒ **在 L4 上仍然跑不了全流程**,只是把「被 host 擋」換成「被 GPU 擋」。⇒ **降為條件 task**(觸發條件:H100 節點的 host RAM < 128 GB)。它的實作成本已探明:替換 `PlaceDB.initialize_from_rawdb`(`$DP/dreamplace/PlaceDB.py:499-616`,117 行)所填的欄位,並 stub 掉 `apply`(`:1133`,唯一在 GP 路徑上用到 `rawdb` 的地方;`BasicPlace.py:604/631` 的 `lefUnit/defUnit` 只在 routability 選項下才走到)——**可行且有界,但不是 M4 主線** |
| **等 H100 才開始做規模化** | 今天的實測顯示 80% 的規模化工作(T1–T7、T9)在 L4 上就能完成並驗證;把它們積壓到 H100 只會讓 H100 的時間被除錯吃掉 |

---

## 6. Q5 — 計時、記憶體 profile 與 exit 改寫

### 6.1 量什麼

| 類別 | 欄位 | 方法 |
|---|---|---|
| **Phase 牆鐘** | `t_read`、`t_initialize`、`t_gp`、`t_lg`、`t_eval_total`、`t_op_total`、`t_diag_total`、`t_total` | `time.perf_counter()`,phase 邊界 `torch.cuda.synchronize()` |
| **Per-iteration** | `iter_ms[]` 全陣列(1000 個 float,8 KB,直接存)+ p50/p90/p99/max | `torch.cuda.Event(enable_timing=True)` 成對記錄,**事件 `elapsed_time()` 一律在 run 結束後批次查詢**(熱迴圈內不 sync) |
| **op 分離計時** | `op_fwd_ms[]`、`op_bwd_ms[]`(只在 profile 模式,每 `--time-every` 個 iteration 記一次) | 同上,事件包住 `_IoFn.forward` / `backward` |
| **evaluator** | `eval_ms[]`(每次 callback) | 同上 |
| **GPU 記憶體** | 每個 phase 的 `peak_alloc_gb` / `peak_reserved_gb`,**phase 開頭 `reset_peak_memory_stats()`**(修 §1.4 B1) | `torch.cuda.max_memory_allocated/reserved` |
| **裝置級記憶體** | `device_peak_used_gb`(含 CUDA context 與碎片,`max_memory_reserved` 抓不到的部分) | **背景 thread 每 0.5 s 取 `torch.cuda.mem_get_info()`(free,total)**;`used = total − free`。**零新依賴**(本機無 `pynvml`/`psutil`,且**網路不通無法安裝**;`nvidia-smi --query-compute-apps` 實測回傳空) |
| **Host 記憶體** | `host_peak_rss_gb` | `resource.getrusage(RUSAGE_SELF).ru_maxrss`,phase 邊界各取一次 |
| **規模元資料** | `n_movable/n_physical/n_filler/n_nets/n_pins/K/rtype/lattice/n_bins/k_chunk/n_active` | 直接讀 |
| **導出係數** | `bytes_per_pin_gp`、`ns_per_pin_op`、`gb_per_mnet_eval` | 由上面算,**每次 run 都重新驗證 §1 的模型** |

**torch.profiler 的定位:** 只在 ≤3.1M 的 case、只開一個 **有界視窗** —— `schedule(wait=5, warmup=2, active=3, repeat=1)` + `profile_memory=True, record_shapes=True`,輸出 `trace.json.gz` + top-20 `key_averages` 表。**明確禁止**在 ≥6M 的 run 上開 profiler(trace 會膨脹到數 GB,且 `profile_memory` 本身有可觀的 overhead,會污染我們要量的 per-iteration 時間)。profiler 的用途是**歸因**(哪個 kernel 吃時間),不是**計時**(計時用 CUDA event)。

### 6.2 輸出格式

```
results/m4/profile/<case>__k<K>__<rtype>__<arm>.json      # 上表全部欄位 + env metadata + 輸入 sha256
results/m4/profile/<case>__.../trace.json.gz              # 只有 ≤3.1M 的 case 有
results/m4/scaling/model_fit.json                         # §1 三個模型的最新擬合 + 殘差
results/m4/tables/full_scale_comparison.md                # spec §9 M4 第二條要求的「全規模對照表」
```

`full_scale_comparison.md` 的欄位(每列 = 一個 case × K × rtype × 臂):
`case | cells | nets | pins | K | rtype | arm | io_mst | ft_mst | io_rg | ft_rg | hpwl | Δio% | Δft% | Δhpwl% | t_total | t_gp | t_op | t_eval | gpu_peak | host_peak`
(`io_rg`/`ft_rg` 兩欄 **M3-dependent**;M3 未完成時留空並註明。)

### 6.3 Exit 改寫(**誠實版**)

spec §9 M4 原文:「10M 全流程;30M 測資製作並跑通;H100 計時;記憶體 profile。exit:30M 端到端數小時內完成;全規模對照表」。

原 exit 的問題:「30M 端到端」在只有 L4 的現實下**無法驗證**(§5.1 P3:host RAM 95–105 GB > 可用 ~80 GB)。定案改寫成 **三段式,前兩段今天的機器就能判、第三段是預先登錄的可證偽預測**:

| # | 判準 | 在哪驗 | 具體門檻 |
|---|---|---|---|
| **E1** 10M 全流程 | **L4** | `mempool_cluster`(11.31M)完成 GP+LG+evaluator+報表,`Δhpwl` 相對其自身 flat ≤ +5%、legalization 成功(`num_unplaced_cells == 0`)、GPU 峰值 ≤ 20 GB、單臂 wall-time ≤ 4 h。至少 K ∈ {16, 32} 各一組 flat 與一組 ours |
| **E2** 30M 測資 | **L4** | (a) 27.7M Bookshelf 產出且 V0/V5 全綠、manifest sha256 齊全;(b) 12.3M 的 V1–V4 全綠(**含 V2 的 Rent 對 `mempool_cluster` 校準**);(c) 27.7M 的元件級 spike(IoTerm+S4 與 T2 後的 evaluator)在 L4 上跑完並記錄峰值/時間 |
| **E3** 全規模對照表 | **L4** | `full_scale_comparison.md` 涵蓋 {1.3M, 2.2M, 3.1M, 6.2M, 11.3M} × K{16,32} 的 flat 與 ours,每格可追到一個 `results/m4/profile/*.json` |
| **E4** 記憶體 profile | **L4** | §1 的三個模型(GP / evaluator / op)各自用 ≥4 個實測點重新擬合,`model_fit.json` 記錄係數與最大殘差;**且 §1.4 的 B1/B2/B3 三個量測 bug 全部修掉並在報告揭露** |
| **E5** H100 交接(**預先登錄的預測**) | **交接文件,不在 L4 判** | M4 報告在 H100 跑之前,**寫死**下列預測:27.7M 單臂在 H100(80 GB,host ≥ 192 GB)的 wall-time 落在 **[0.5 h, 1.5 h]**,GPU 峰值落在 **[26 GB, 38 GB]**,host RSS 落在 **[95 GB, 130 GB]**。H100 實跑若落在 `[0.3×, 3×]` 的容忍帶外,**模型錯誤**,必須回頭重擬合並在報告記錄——這比「跑得完/跑不完」有資訊量得多 |

**E5 的推導(必須逐項寫進報告,讓它可被檢驗):** L4 上 27.7M 單臂推估 3.9 h(§5.2);L4 記憶體頻寬 300 GB/s、H100 SXM 3.35 TB/s(11.2×),但我們的 op 是 gather + fp64 atomic 混合負載,實際加速一般落在頻寬比的 0.5–0.7 ⇒ 6–8× ⇒ **0.5–0.65 h**;加上 GP 與 evaluator 的非線性,取 **[0.5 h, 1.5 h]**。

### 6.4 H100 交接契約(T10 的交付物)

```
docs/handover/h100-30m-runbook.md
  ├─ 資源契約:GPU ≥ 40 GB(建議 80 GB);host RAM ≥ 192 GB;磁碟 ≥ 100 GB
  ├─ 環境重建:docs/dev-env.md 的完整 build 流程(Boost/CUDA 無 root 安裝已驗證)
  │            + 兩個必要檢查(CMAKE_CXX_ABI=1、CUDA_ARCH_FLAGS 要加 sm_90)
  ├─ 資料:benchmark 的 manifest + sha256(30M Bookshelf 由 tiler 在目標機**重新生成**,
  │        不搬 5.2 GB 檔案——tiler 是串流的,重生成只要 ~10 min 與 1 GB 記憶體)
  ├─ 單一指令:scripts/m4_run.sh <case> <K> <rtype> <arm>
  └─ 預先登錄的預測(§6.3 E5)與判定腳本 scripts/m4_check_prediction.py
```

**設計原則:H100 上不做任何設計決策。** 所有參數、所有 gate、所有預測都在 L4 上定死,H100 只執行與比對。

---

## 7. Q6 — Task 分解與依賴序

規約沿用 M2 §11 / M3 §8:每 task 一次 TDD 迴圈;測試一律 `$DP/.venv312/bin/python -m pytest`;新程式碼全在 `ioplace/`;DREAMPlace 原始碼不動。

| Task | 類型 | M3-dep? | 內容 | 驗收 |
|---|---|---|---|---|
| **T0** 探針遷入 | 純軟體 | 否 | 把 §1 的 5 個探針遷入 `ioplace/diagnostics/probes_m4/`(`probe_gp_memory.py`、`probe_eval_scaling.py`、`probe_spike_scale.py`、`probe_host_rss.py`、`probe_bookshelf_export.py`),輸出 `results/m4/probes/*.json` + env metadata + 輸入 sha256 | §1 每個數字可追溯;不符即更新本文 |
| **T1** profile 基建 | 純軟體 | 否 | 修 §1.4 的 B1(每 phase `reset_peak_memory_stats`)、B2(`--diag-every N`、`--no-diag`,並讓 `io_grad_l1` 復用 diagnostics 的那一次 backward)、B3(`spike_10m` 的 `--budget-gb`);新增 `ioplace/profile.py`(CUDA-event 計時器 + `mem_get_info` 取樣 thread + §6.2 的 JSON schema) | 小 case 上 phase 峰值互不污染的測試;取樣 thread 不改變結果的決定性測試;schema 完整性測試 |
| **T2** **evaluator streaming** | 純軟體 | **軟相依** | §4.2 的 ③④⑤⑥:消滅 `(P,K)` one-hot、MST edge/segment 批次化、索引 int32、批次大小為建構參數。**若 M3 T1 已 merge 則在其之上做,否則 M4 自己先做 ①②** | **既有 CPU/GPU 等價測試全綠**;新增「批次大小 ∈ {1e6, 8e6, 全部} 三種設定結果逐位元相同」;`bigblue4` K=32 峰值降幅 ≥ 60%;**`mempool_group`(3.5M nets)K=32 實跑峰值 ≤ 2 GB** |
| **T3** 語料接入 | 純軟體 | 否 | **= Stage2 S5,共用同一個 task,不重複實作。** 產 `benchmarks/ispd25/*.json` config(**全 15 LEF、tech 最前**)+ per-case `target_density` 反推(§2.4 M4-L3);補 `superblue12` 的實際 pins/nets | 三個 NanGate45 case + `superblue12` 各有一份 config 與一份 `PlaceDB.read` 統計 JSON |
| **T4** tiler | 純軟體 | 否 | `ioplace/bench/{export_bookshelf,tile_bookshelf,glue_gen}.py`(§3.2) | 玩具 case:2×2 陣列的 node/net/pin 數 = 4× 來源 + glue;座標與 row 無重疊無縫隙;決定性(同 seed 逐位元相同);串流峰值記憶體 ≤ 2 GB(以 3.1M 來源實測) |
| **T5** Rent/驗收器 | 純軟體 | 否 | `ioplace/bench/{rent,verify_bench}.py`(§3.3 V0–V5) | 已知 Rent 的合成 case(規則 mesh p≈0.5、完全隨機 p≈1.0)上量得的 p 落在 ±0.05;V0 對故意破壞的檔案全部抓得到 |
| **T6** **tiler 校準實驗** | **實驗** | 否 | 產 2×2 陣列(12.31M),跑 V1–V4;掃 `(γ_diag, γ_far)` 直到 V2 通過 | V2/V3/V4 全綠;`results/m4/bench/verify_group2x2.json` + 掃描軌跡 |
| **T7** 30M 生成 | 純軟體 | 否 | 用 T6 定出的 `(γ_diag, γ_far)` 產 3×3(27.70M);跑 V0/V5 | Bookshelf 檔 + manifest;V0/V5 綠 |
| **T8** **全流程實驗** | **實驗** | **部分** | §6.3 E1/E3 的矩陣:{1.3M, 2.2M, 3.1M, 6.2M, 11.3M} × K{16,32} × {flat, ours};`ours` 臂的 `ρ*` 需 per-case 校準(沿用 M3 T5 的方法) | E1/E3 的每一格有 JSON;**`io_rg`/`ft_rg` 欄位 M3-dependent,M3 未完成則留空並註明** |
| **T9** 30M spike | **實驗** | **部分** | §5.3 第 2 層:27.7M 的 IoTerm(+S4)與 T2 後 evaluator 的元件級量測 | 峰值、時間、`ok` 判定;**S4 的 0–2 個 `(E,) fp64` 累加器兩種情境都量**(§8 的參數化帳) |
| **T10** H100 交接包 | 文件+軟體 | 否 | §6.4 的 runbook + `scripts/m4_run.sh` + `scripts/m4_check_prediction.py` + 預先登錄的預測 | 在 L4 上以 3.1M case 完整演練一次 runbook(除了規模,每一步都要真的跑過) |
| **T11** 報告 | 文件 | **部分** | `docs/results/m4-scale-up-report.md`:E1–E5 逐條打勾/打叉;三個模型的擬合與殘差;全規模對照表;B1/B2/B3 的揭露與對 M2/M3 舊數字的影響評估 | 每個數字可追到 `results/m4/**/*.json`;FAIL 照 M1/M2 慣例誠實記錄 |
| **T12**(條件) | 純軟體 | 否 | numpy-only PlaceDB shim(§5.4)。**觸發條件:H100 節點 host RAM < 128 GB** | 3.1M case 上 shim 路徑與原生路徑的 GP 結果逐位元相同 |
| **T13**(條件) | 純軟體 | 否 | IoTerm 的 fused CUDA op(spec §4 line 64 已預告)。**觸發條件:T8 的 11.3M 單臂 > 3 h,或 T9 顯示 27.7M 的 op 佔總時間 > 85%** | 與 pure-torch 路徑的 fp64 `gradcheck` 一致;`bigblue4` 上 ≥5× 加速 |

### 依賴序

```
T0 ──┬── T1 ──────────────┬── T8 ── T11
     │                    │
     ├── T2 ──────────────┤        (T2 是一切 >=5M 實驗的硬前置)
     │                    │
     ├── T3 ──┬── T4 ── T5 ── T6 ── T7 ── T9 ──┘
     │        │
     └────────┴── T10 (演練需 T3+T8 的 3.1M 結果)

條件:T12 (host RAM 情報) / T13 (T8 或 T9 的計時)
```

- **T0 必須最先**(M2/M3 的 finding 遷入慣例)。
- **T2 是硬前置**:沒有它,11.3M / 12.3M 的任何 evaluator 呼叫都 OOM。
- **T3/T4/T5 可與 T1/T2 並行**(不同檔案、不同人)。
- **T6 必須在 T7 之前**(§3.3:V2 沒過不准生成 3×3)。
- **T8 中「M3 臂」的欄位擋在 M3 v2 完成之後**;`flat` 與 M2 臂**現在就能跑**。
- **T10 的演練必須在 T8 的 3.1M 結果之後**(runbook 要引用真實的輸出檔名與格式)。

**現在就能開工的(零 M3 相依):T0, T1, T2, T3, T4, T5, T6, T7。**
**擋在 M3 v2 之後的:T8 的 `io_rg`/`ft_rg` 欄位與 `κ_ft` 臂、T9 的 S4 情境、T11 的對應章節。**

---

## 8. 記憶體帳的參數化(因應 M3 v2 的 S4 公式尚未定案)

M3 v2 可能把 S4b 換成別的候選,但都是「同一條 chunked backward 上的係數擴充」。M4 的帳因此寫成兩種情境,**兩種都在 T9 實測**:

| 情境 | 額外常駐狀態 | @12.71M nets | @31.54M nets |
|---|---|---:|---:|
| **A**(0 個額外 `(E,)` fp64) | 只有 `home_e` `(E,)` int8 | +13 MB | +32 MB |
| **B**(2 個額外 `(E,)` fp64,即 M3 v1 的 `N_e`/`M_e`) | `home_e` int8 + 2×`(E,)` fp64 | +216 MB | +537 MB |

**⇒ 兩種情境的差 ≤ 0.54 GB @30M,相對於 op 本體的 8.47 GB 與 GP 的 12.6 GB 是二階量。M4 的所有規模結論對 S4 的最終形式不敏感** —— 這一句要寫進 M4 報告,免得 M3 v2 一改就得重做整份規模分析。

同理,M3 的 `L_cap`(S7)若啟用,額外狀態是 `(K,K)` 的 `Q`/`G` 與 `(K,K,A)` 的 `P` 常數表(K≤32、A≤2K)⇒ **< 1 MB,可忽略**。

---

## 9. 風險與 gate

| ID | 量測 | 門檻 | 觸發後動作 |
|---|---|---|---|
| **M4-G1** T2 沒達標 | `mempool_group` K=32 的 evaluator 峰值 | **> 2 GB** | T2 未完成,不得進 T8 的 ≥6M 臂。先做 edge 批次化的 profiling 找剩餘大戶 |
| **M4-G2** 11.3M 全流程失敗 | E1 的任一條 | 任一 fail | 依失敗模式分流:OOM → 降 K/lattice 並記錄;legalization 失敗 → 調 `target_density`(§2.4 M4-L3);`Δhpwl` 爆 → 降 `ρ_max`(per-case ρ 校準,沿用 M3 T5 方法) |
| **M4-G3** tiler 造假 | V2(Rent)/V4(cut 比) | V2 差 > 0.05 或 V4 出 `[0.5,2.0]` | glue 模型錯 ⇒ 掃 `(γ_diag, γ_far)`;掃不出來就退配方 B(§3.4),並在報告明寫「30M 的連通性是擬合出來的」 |
| **M4-G4** op runtime 失控 | T8 的 11.3M 單臂 wall-time | **> 3 h** | 觸發 T13(fused CUDA op)。**明確不准**用「降低 GP iteration 數」來偽造達標 |
| **M4-G5** host RAM | 任一 `PlaceDB.read` 的 peak RSS | **> 100 GB** | 該 case 在本機作廢,轉 H100;若 H100 也不足 ⇒ 觸發 T12 |
| **M4-G6** 模型失效 | §1 三個模型在新 case 上的殘差 | **> 25%** | 模型不可外推 ⇒ E5 的預測區間必須放寬並在報告說明;不得事後偷偷改預測 |
| **M4-G7** B1 的污染範圍 | M2/M3 報告中引用 `peak_mem_mb` 的段落 | 存在即觸發 | 逐條標註「此數字為 process 累積值,已作廢」,並在 M4 報告給出重測值 |

**額外風險(無獨立 gate,但必須記載):**

| # | 風險 | 證據/狀態 | 緩解 |
|---|---|---|---|
| R1 | `mempool_cluster` 的 GP **品質**未驗(只驗了「跑得完」) | §2.4 M4-L2 | T8 的第一個 gate |
| R2 | ISPD2025 是 `place_opt`/`preCts` 的**已擺置** DEF;DREAMPlace 從它的座標起跑等於 warm start,與 ISPD2005 的 random-center init **不同 regime** | 實測 DEF header(Stage2 §3.2) | T3 必須決定並記錄:是否 `random_center_init_flag=1` 覆蓋掉輸入座標;兩種都跑一次作 sanity |
| R3 | lattice 解析度在大 die 上的敏感度未量 | `run_placement.py:11` 預設 512 | T8 加一組 lattice ∈ {512, 1024, 2048} 的 sensitivity(只在 3.1M 上做) |
| R4 | 合成 case 的 uniform 位置會**系統性高估** evaluator 成本 | §1.2 的合成 `io_count=29.8M` vs 真實 bigblue4 K16 的 100,333 | T9 的 spike 必須用 **tiler 產出的真實 `.pl` 起始座標**,不用 uniform random |
| R5 | 30M 的 pin/net 比是外推(3.91 pins/cell,來自 group 的 12.03M/3.08M) | 推估 | T7 產出後直接數,回填本文 |
| R6 | H100 節點的 CUDA arch 需重 build(`sm_90`) | `docs/dev-env.md` 記錄本機只 build 了 `sm_89` | T10 的 runbook 第一步 |
| R7 | `mempool_cluster` 讀取 285 s + 62.6 GB,**每次 run 都要付一次** | 實測 | T3 評估「一次讀入、多臂共用同一 process」的 driver 模式;但要同時保證 §1.4 B1 的 per-phase reset 正確 |

---

## 10. 低信心段落總表(v2 對抗性審查請優先攻擊這裡)

| ID | 段落 | 不確定的是什麼 | 由誰解決 |
|---|---|---|---|
| M4-L1 | §2.4 | `superblue12` 的 pins/nets 與 GP 峰值是推估 | T3 |
| M4-L2 | §2.4 | `mempool_cluster` 的 GP **品質**(HPWL/overflow/LG)完全未看 | T8 |
| M4-L3 | §2.4 | NanGate45 的 `target_density` 該取多少(利用率 0.39–0.70 差異巨大) | T3 |
| M4-L4 | §3.3 | Mt-KaHyPar 在 12.3M/27.7M hypergraph 上的可行性完全未知 | T5/T6 |
| M4-L5 | §3.2 | glue net 模型(距離衰減核 φ)是**假設**;把 4-group 的互連統計外推到 9-group 沒有理論保證 | T6 的 V2 校準 |
| M4-L6 | §4.3 | T2 之後的 evaluator 記憶體(4.5 GB / 10 GB)是推估 | T2 的驗收 |
| M4-L7 | §5.2 | 單臂 wall-time 推估假設 op 在 600 個 iteration 上啟用;實際啟用點由 `of_on=0.90` 決定,在 10M 級的 overflow 軌跡未知 | T8 |
| M4-L8 | §6.3 E5 | H100 的 6–8× 加速比是**啟發式**(頻寬比 × 0.5–0.7),沒有同型 workload 的實測支撐 | H100 首跑;容忍帶 `[0.3×,3×]` 就是為此而設 |
| M4-L9 | §3.1 | 「cluster = 4 groups」是從 `11,310,807/3,077,669 = 3.67` 與 MemPool 架構常識推出的,**沒有從 RTL 階層名稱直接驗證** | T4 的第一步:抽樣 `mempool_cluster` 的 instance 名稱,確認 group 前綴 |
| M4-L10 | §4.4 | Λ∈[4,8] 的 net 在 NanGate45 語料上的比例未量(只有 ISPD2005 的 0.24%) | T3 順手量 |

---

## 11. 證據附錄(T0 待遷入)

**狀態:下列每一條皆為 2026-08-13 在本機實跑;探針目前在 `/tmp`,T0 完成前一律視為「待重現」。**

| 探針(暫存 → 目標檔名) | 內容 | 支撐 |
|---|---|---|
| `/tmp/probe_m4_mem.py` / `/tmp/probe_m4_full.py` → `probe_gp_memory.py` | 五個 case 的 `PlaceDB.read` + `initialize` + GP+LG,量 phase 時間、host RSS、GPU 峰值 | §1.1 全表、§1.1 的擬合模型 |
| `/tmp/probe_m4_read.py` → `probe_host_rss.py` | 只讀不跑,量 `PlaceDB.read` 的 peak RSS 與 die/利用率 | §1.1 host RSS 欄、§2.4 M4-L3 |
| `/tmp/probe_m4_eval.py` → `probe_eval_scaling.py` | 合成 netlist 上掃 `evaluator_gpu` 的 (nets, K) → 峰值/時間 | §1.2 全表、§4.1 拆解 |
| `/tmp/probe_m4_spike30.py` → `probe_spike_scale.py` | `spike_10m.run()` 參數化到 30M | §1.3、§5.3 第 2 層 |
| `/tmp/probe_m4_macro.py` → `probe_macro_stats.py` | movable macro 統計、平均 degree | §1.1 次級事實 1 |
| (新增)`probe_bookshelf_export.py` | `BOOKSHELFALL` 轉出 + 檔案大小/時間 | §3.1 證據 1、2 |
| (既有)`results/m2/spike/spike_10m.json` | 10M spike 的 4.107 GB / 84 B/pin | §1.3 |
| (讀碼)`ioplace/evaluator_gpu.py:326,332,351,352,385,392-416,413` | §4.1 的逐項定位 | §4.1 |
| (讀碼)`ioplace/ops/io_term.py:331,386-399` | `k_chunk` 規則、diagnostics 的 7 趟 | §1.4 B2、§5.1 P2 |
| (讀碼)`ioplace/drivers/run_placement.py:129`、`run_placement_io.py:145,155,202` | 缺 `reset_peak_memory_stats`、diagnostics 呼叫點 | §1.4 B1/B2 |
| (讀碼)`$DP/dreamplace/PlaceDB.py:1010-1033,499-616,1133`、`ops/place_io/src/BookshelfWriter.cpp:22-51,100-138`、`Params.h:26`、`PybindPlaceDB.cpp:60` | Bookshelf 匯出能力、T12 的實作面 | §3.1、§5.4 |

**與既有資產的關係:** `ioplace/ops/io_term.py` 的 chunked 契約、`ioplace/schedules.py` 的 version invariant、`ioplace/dp_hook.py` 的 secant refresh、`ioplace/drivers/run_placement_io.py` 的 callback 結構,M4 **全部原樣沿用**;M4 的改動集中在 `ioplace/evaluator_gpu.py`(T2)、新目錄 `ioplace/bench/`(T4/T5)與新模組 `ioplace/profile.py`(T1),外加 driver 的三個量測 bug 修正。**沒有任何一項要求改寫 M2 已驗證的數學路徑。**

---

**與其他草案的介面(避免重複設計):**
- **Stage 2 S5 = M4 T3**,同一個 task,同一份 config 產生器;Stage 2 §4.3 的 L-Q1-b(`mempool_cluster` 可讀性)**今天已被本文的實測回答**:可讀,285 s / 62.6 GB。
- **M3 T1 與 M4 T2** 的分工見 §4.2,已明確切分,兩邊的驗收條件互不重疊。
- M3 draft §7 G6 的「10M spike 8 GB 契約」在 M4 改為 `--budget-gb` 參數(§1.4 B3);**M3 自己的 G6 判準不變**(仍是 10M / 8 GB),M4 只是不再把 30M 的 12.32 GB 誤判成失敗。
