# Benchmark 公平性與比較方法論筆記（不進投影片）

2026-08-05。給自己的研究思考，回答三個問題：
(1) 和 DREAMPlace 比 IO number 在其原生 benchmark 上是否合理；
(2) 1M–30M benchmark 是否公平可信；
(3) region 到底該由誰、怎麼決定。

分析由 Opus deep-reasoner 完成（含在本機 M0/M1 實際資料上跑的 probe），
Fable 主線整理定稿。probe 數字尚未進 repo（見文末「驗證狀態」）。

---

## 0. 核心發現：在幾何 region 上，io_count 與 tree wirelength 近乎共線

用 integral geometry（Cauchy–Crofton 的軸對齊版本）：對「與 netlist 無關」的
geometric region（grid / random slicing），期望穿越數

```
E[io] ≈ ρ_v·Σ|Δx| + ρ_h·Σ|Δy|,   ρ = 內部 region 邊界總長 / die 面積
```

在 M0 的實際輸出上驗證（讀 `results/m0/*.npz` 最終座標 + `evaluator_ref` 的 MST）：

| case (k=16, grid) | Crofton 預測 | 實測 λ_geo | 實測 io_count |
|---|---:|---:|---:|
| adaptec1 | 23,765 | 24,163（誤差 +1.7%） | 30,011 |
| bigblue4 | 79.6k–80.0k | 80,925（+1.2~1.6%） | 100,108 |

- `io_count ≈ 1.24 × λ_geo`，兩個 case 同一常數；那 24% 是 MST 拓撲 + L-shape 繞路 overhead。
- k=8→16、grid→slicing 的變化也全由邊界密度解釋（誤差個位數 %）。

**含意：flat DREAMPlace 在最小化 HPWL 時「順便」就把 io 最小化了。**
M1 reweighting 對 io 無效（best −0.06%）不是實作壞掉，而是這個設定下
IO 目標與 WL 目標共線、沒有可利用的 slack。真正的 headroom 要分兩塊量：

| adaptec1 k=16 grid | 值 | 含意 |
|---|---:|---|
| io_count(flat) | 30,011 | 實測 |
| λ_geo(flat 誘導歸屬) | 24,163 | assignment 造成的不可約 crossing |
| io − λ_geo | 5,848（19.5%） | **detour headroom**（M1 的 ft 改善就是打這塊） |
| Mt-KaHyPar min-cut λ−1 | 19,594 | **assignment headroom**（−18.9%） |

注意 bigblue4 的反例：Mt-KaHyPar λ−1 = 98,847 **比 flat 自己誘導的 λ_geo = 80,925 還差 22%**。
M0 報告把「initial_cut_io_lb 低於 flat io」讀成「partition 切得夠好」——那是拿
λ−1（min-cut）去比 io（含繞路），不同量綱；正確對照是 λ−1 vs λ_geo。這句要改。

---

## 1. Q1：在 DREAMPlace 原生 benchmark（ISPD2005/06、DAC2012、ISPD2015/19）上比 IO number？

**判定：有條件合理——但條件很嚴，目前的形式會反過來傷自己。**

### Strawman 風險是「倒過來的」
標準批評是「贏一個對手不知道的指標不算贏」。實況更糟：由第 0 節，DREAMPlace
事實上就在優化這個指標。M1 三個有基準的 case，io Δ = 0.00% / −0.06% / +0.63%，
同時 HPWL 付 +0.9~1.3%——這個資料點目前是**反證**，不是證據。

### 要能站得住，必須共同報告
- 全欄位：HPWL、tree WL、io、pure-FT、boundary-pair demand vs capacity、runtime、peak mem。
- **`io/tree_wl`（IO efficiency）**：沒有它，任何縮線長的方法都「免費贏 IO」。
  順帶：two_stage 的 io/tree_wl 其實比 flat 好 2.35×，輸在 tree_wl 爆 2.9×——單點比較把這事藏掉了。
- **`io = λ_geo + detour` 分解**：可微分歸屬打 λ_geo、reweighting 打 detour，
  M1「ft 大降、io 不動」正是這個結構的直接證據。不分解就說不清哪個機制在動。
- **iso-HPWL 的 Pareto 曲線**（掃 λ_IO），不報單一調好的點。

### 在無 region 的 suite 上加 region 的立場問題

| region 來源 | 偏誤 | 用法 |
|---|---|---|
| 由 partitioner 導出幾何 | **循環論證**——正確答案被 planted，且 planted 的正是我們的 objective | 否決為主選；只可當 sensitivity arm |
| 中性 grid（現行） | 無 planted bias，但就是第 0 節的退化情形 | 保留為 control |
| 固定 seed random slicing | 同上，邊界密度稍高 | control 第二型態 |
| 設計者 fence（ISPD2015 有真 fence：superblue11_a×4、superblue16_a×2、des_perf_b×12） | 最有立場；但 K 小、不鋪滿 die，需一條 fence→全 die 分割的 adaptation 規則 | **headline 的「真實 region」arm** |

鐵律：**region 幾何在任何 run 之前凍結、commit 成 artifact、對所有 arm 位元相同；
region 歸屬永不作為任何一方的輸入。**

### 規模天花板
本機實測：ISPD2005 上限 bigblue4 = 2,177,353；ISPD2006 newblue7 = 2,507,954；
ISPD2015 superblue12 = 1,287,037。**stock suite ≈ 2.5M 封頂，離 10M–30M 差一個數量級。**
所以 stock-suite 的表只能背「方法在真實工業 netlist 上有效 / IO 效率更高」，
**不能背 scaling claim**——scaling 靠 mempool_cluster（ISPD2025, ~11.3M）與 TeraPool。

### 審稿人清單（缺一被打）
1. pre-registered、netlist-independent 的 region 協定 + region 檔隨論文釋出；
2. iso-HPWL 比較 + Pareto；
3. baseline 資訊對等與同等調參 effort；
4. 一條便宜的 region-aware baseline（flat GP + 邊界 cell-swap post-pass）——沒有它，貢獻可能被一個 post-pass 複製；
5. 多 seed + 變異數（現況噪聲 ~1.0% > 效果 0.06%，n=1 等於沒有數字）;
6. 主動揭露退化性（io/tree_wl、λ_geo 分解）——審稿人自己推出 Crofton 關係時就晚了；
7. evaluator 保真度外驗：至少一個 case 過真 global router 或 ICCAD24-D 官方 checker，證明 tree-crossing 與實際 boundary pin demand 單調相關。

---

## 2. Q2：現在的 1M–30M benchmark 公平可信嗎？

**判定：前提不成立——benchmark 還不存在；且現有兩個真實 case 的實驗設計有一個必修 bug。**

### 前提更正
repo 內沒有任何 benchmark 生成程式碼。實際跑過的只有 adaptec1（211K）與
bigblue4（2.18M）；「合成 2M」只是 evaluator 計時 fixture，沒有 quality 數字。
1M/5M/10M/30M 全部還在 spec §7 的紙上階段。

### 四項稽核
1. **Netlist realism**：ISPD2005 真實但老（2005、無 timing、無 LEF/DEF），且
   adaptec1 的 fixed macro 佔 die **56%**；完全 flattened——沒有 module hierarchy，
   就沒有 region 顆粒度的自然 cluster 結構。這正是第 0 節退化的根因。
2. **Planted-partition bias**：現行幾何 region 沒有（乾淨）。但**計畫中的配方 A 是
   教科書級 planted partition**：spec 明寫用 RTL tile/group/cluster 邊界當 region ground
   truth——把 MemPool tile 陣列複製到 30M、region 定在 tile 邊界，正確答案 trivially
   可還原，任何 min-cut partitioner 都能解，贏面取決於 baseline 在 trivial 任務上輸多難看。
   配方 B（Rent-gap glue nets）spec 自己標了循環論證風險。**這是前方最大可信度地雷。**
3. **Baseline 資訊對等**：兩個方向都有事。
   - 刻意不對等（合法）：flat 不給 region → claim 只能是「同 HPWL 下 IO 降 X%」。
   - **必修 bug**：two_stage 的 Mt-KaHyPar 用 unit node weight + ε=0.03 把 cell「數量」
     均分；但 adaptec1 k=16 的 per-region **自由面積**（扣 fixed macro）差 **4.05×**，
     且 16 區中 3 區結構上塞不下均分的 cell 面積 → two_stage 的 HPWL 3× 爆炸與
     legalization overlap 有相當比例是**baseline 被綁手**，不是 fence 的本質代價。
     修法：node weight = cell area、block target ∝ region 自由面積。
4. **統計噪聲**：n=1，run-to-run 噪聲 ~1.0% > 宣稱效果 0.06%。M0/M1 的所有 delta
   目前都不可解讀（M1 報告自己已誠實記錄）。

### 最小修補集（CP 值排序）
1. 修 two_stage 的 capacity model（area-weighted partition + per-region free-area target），重跑 M0 四格。
2. 每張表加 `λ_geo`、`io−λ_geo`、`io/tree_wl` 三欄（λ_geo 用 pin_region_bitmask popcount，幾乎免費）；GPU/CPU parity test 同步鎖住。
3. ≥5 paired seeds（所有 arm 共用 seed；動 placement 的隨機性，不只 region seed），報 mean±std。
4. region 三型態並列：grid（control）+ slicing×3 seeds（control）+ ISPD2015 真 fence（designer arm）。
5. 生成/擴增 case 一律附 netlist-stat 報表（RentCon p、net-degree 分布、cell-size mix、macro 佔比、per-region utilization），對照真實設計。
6. 10M/30M 用真實設計（mempool_cluster / TeraPool）；ArtNet 只當 scalability case（報 runtime/memory，不報 quality）。
7. **反 planted-partition 硬規則**：headline 實驗的 region 刻意與 tile 邊界錯開
   （平移半個 tile、或 K 不整除 tile 數）；tile-aligned 版本另立 arm 標為 favourable case，
   **主動報兩者 gap**——那個 gap 就是「gain 有多少來自 planted structure」的量化答案。

---

## 3. Q3：兩個實驗的 region 該怎麼決定

### Exp A（designer region + per-region Innovus；ours 重指派 block）
**有條件合理，是說服力最高的一條。**
- region 來源：設計者的，一個字都不要動——立場正來自「不是我們給的」。
- 跨 arm 固定：region 幾何 byte-identical、tech/LEF/lib/SDC、Innovus 版本/腳本/effort/seed、
  top-level FT insertion policy、以及**兩 arm 用同一個 evaluator 計數**（工具 report 當 cross-check）。
- **關鍵不變量：per-region utilization**。重指派若讓某區變鬆某區變緊，QoR 差異就被
  容積重分配汙染。要（i）沿用設計者的 per-region utilization cap 當硬約束，
  （ii）兩 arm 都報 utilization 表。
- 使比較失效：改了 region 幾何/容量、其實做了 re-floorplan、兩 arm utilization 不同、
  兩 arm 量 IO 的尺不同、baseline 是未調初版而 ours 調到飽。
- **必加第三條 arm**：IO-agnostic 的 assignment（現成 greedy+2-opt）。不加就分不清
  「贏在 IO-aware」還是「贏在有做任何最佳化」（本質是 QAP）。

### Exp B（ours vs 無視分區的 flat DREAMPlace）
**目前形式不合理；加條件後有條件合理。**
- region：netlist-independent、跑前凍結 commit（grid + 固定 seed slicing）；絕不從 partitioner 來。
- 固定：同 config/版本/iteration 預算/LG-DP 設定/target density/seed/GPU/evaluator 參數。
- 使比較失效：在測試 case 上挑 λ 或 (α, every) 的最佳格當結果（M1 的 sweep 就是這樣，
  所以 −0.06% 不能當獨立結果）；ours 多跑 iteration；不同 HPWL 點上比 IO；
  只跟 region-blind 比（共線性 ⇒ 必加 region-aware 便宜 baseline：flat + 邊界 cell-swap）。
- 呈現：iso-HPWL Pareto，不是單點。「+1.3% HPWL / −0.06% IO」的單點是在替審稿人寫 reject 理由。

---

## 4. Top-5 風險（殺傷力排序）

1. **指標退化（io ≈ 1.24×ρ×tree_wl，誤差 <2%）**——被審稿人自己推導出來，novelty 就變
   「WL 的重新包裝」。緩解：主動報分解；headline 換到有真 hierarchy 且 region 與
   hierarchy 不完全對齊的設計（mempool_cluster / ISPD2015 fence），那裡才有非共線 slack。
2. **Planted partition（配方 A/B）**——region=RTL 邊界 ⇒ 答案被種進題目。緩解：錯開 + 報 gap。
3. **Baseline 被綁手**——unit-weight partition vs 4.05× 自由面積差、3/16 區不可行。緩解：area-weighted 重跑。
4. **統計效力不足**——n=1、噪聲>效果。緩解：≥5 paired seeds；之前不把任何 delta 寫進論文。
5. **規模宣稱無憑據**——stock 天花板 2.5M；10M/30M 生成碼不存在。緩解：mempool_cluster
   載入路徑列為 M4 關鍵路徑；stock 表不背 scaling claim。

---

## 5. 可直接執行的下一步（丟給 fast-worker 的粒度）

1. `ioplace/partition/mtkahypar_runner.py`：`partition_netlist` 加 `node_weights`
   （cell area 整數化）與 `block_weights`（∝ per-region 自由面積，用 RegionGrid 光柵化
   fixed macro 計算）；`run_placement_two_stage.py` 傳入；重跑 M0 四格 two_stage。
2. `evaluator_ref.py` / `evaluator_gpu.py`：`EvalResult` 加 `lam_geo`
   （pin_region_bitmask popcount，deg>1 取 popcount−1 加總）；`make_report.py` 表格加
   `lam_geo`、`io−lam_geo`、`io/tree_wl` 三欄；加 GPU/CPU parity test。
   ——這一步同時把第 0 節的 probe 變成可重現的 repo 內診斷。
3. `run_placement.py` CLI 加 `--place-seed` 接到 DREAMPlace 隨機源；adaptec1
   k∈{8,16} × {flat, reweight} × 5 seeds，報 mean±std。這是判定 M1 −0.06% 是效果
   還是噪聲的唯一實驗。
4. 新增 `run_placement_swap.py`：flat GP+LG 後，對邊界一個 bin 內的 cell 做以
   evaluator 為目標的貪婪 pairwise swap（面積相容）。Exp B 必備的 region-aware baseline。
5. specs 新增「region 定義協定」一節：生成規則、seed 清單、凍結時點、artifact 路徑
   （`configs/regions/`，plan 有列但目前不存在）；明訂 region 歸屬永不作為輸入。

---

## 驗證狀態

- 第 0 節的 Crofton probe 與 λ_geo/自由面積數字出自本 session 的 Opus deep-reasoner
  在 `results/m0/*.npz` 上的實測 probe（腳本在 session scratchpad，未 commit）。
  方向與 M1 報告的獨立觀察（ft 降、io 不動）互相印證，但**數字要等第 5 節第 2 步
  把 `lam_geo` 落進 evaluator 後才算可重現**。
- 其餘引用的 M0/M1 數字皆出自已 commit 的
  `docs/results/m0-baseline-report.md` / `docs/results/m1-reweight-report.md`。
