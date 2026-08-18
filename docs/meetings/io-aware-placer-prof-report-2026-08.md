# IO-Aware Top-Level Placement — Progress Report (2026-08)

對應投影片 [io-aware-placer-prof-report-2026-08.pptx](io-aware-placer-prof-report-2026-08.pptx) 的逐頁內容。
投影片本文為英文，每頁附中文講者備忘（也寫在 pptx 的 speaker notes 裡）。
Benchmark 公平性的研究思考（不進投影片）另見
[2026-08-benchmark-fairness-notes.md](2026-08-benchmark-fairness-notes.md)。

---

## Slide 1 — Title

**IO-Aware Top-Level Placement**
GPU global placement for 10M–30M-cell designs that minimizes cross-region IO and pure feed-throughs

- Progress Report · August 2026
- DREAMPlace-based · dev on NVIDIA L4, target H100 · Phase 1: fixed regions, count-based objectives

> 講者備忘：開場 30 秒講 scope——10M–30M cell 的 top-level placement，region 幾何由設計者給定，
> 目標是最小化跨 region 的 IO 穿越數與 pure feed-through 數（都是「計數」而非線長），在 DREAMPlace 上用 GPU 完成。

---

## Slide 2 — Outline

1. **Previous Works** — 7 papers · feed-through-aware planning line + GPU analytical placement line, one slide each
2. **Benchmark** — the 1M / 5M / 10M / 30M ladder, region overlay, and realism gates
3. **Algorithm Flow** — Phase-1 pipeline + what M0/M1 already built and taught us
4. **Experiments** — two comparison tracks: per-region commercial P&R, and flat DREAMPlace

> 講者備忘：四個部分；Previous Works 含老師指定的 GrandPlan 與 ASP-DAC'24 兩篇。

---

## Slide 3 — Problem Formulation (Phase 1)

**Given**

- Flat netlist at 10M–30M standard cells (macros fixed in Phase 1)
- K = 4–32 exactly-abutted partition regions — geometry is a fixed input (channelless style)

**Optimize placement to minimize**

- **IO count** — crossings of region boundaries by each net's routing tree (MST / RSMT proxy)
- **Pure feed-throughs** — tree passes a region where the net has no pin
- … while preserving wirelength (HPWL) and legality

**Output**

- Placement + induced block→region membership + per-net IO / FT report

標籤：Counts, not wirelength ｜ Regions fixed, membership free ｜ GPU end-to-end

（右側示意圖：2×2 region 的 die；橙點 = tree edge 穿越邊界（IO）；紅色 net 從 R3 走到 R2 途中
穿過沒有 pin 的 R4，該段即 pure feed-through。）

> 講者備忘：用圖解釋兩個指標的定義。強調：優化的是離散計數，region 幾何不動，動的是每顆 cell（亦即 block 歸屬）。

---

## Slide 4 — Previous Works: Two Research Lines, and the Gap We Target

**Line A · Feed-through-aware top-level planning（module granularity, ~10² blocks）**
GrandPlan (ISPD'26) · FTAFP (ASP-DAC'25) · Flora (arXiv'25) · FT insertion (ISQED'25) · HiDaP (TCAD'21)
— decide region geometry / module shapes; minimize cross-partition wirelength or module-level FT

**Line B · GPU analytical placement engines（cell granularity, 10⁶–10⁷ cells）**
DREAMPlace (DAC'19/TCAD'21) · 3.0 fence regions (ICCAD'20) · 4.0 net weighting (TCAD'23) · RTO placement (ASP-DAC'24)
— place cells at scale on GPU; objectives are wirelength / density / timing — no cross-region IO or FT notion

**This work — the intersection neither line covers**
Cell-granularity GPU placement with count-based objectives (boundary-crossing IO #, pure-FT #) under
fixed region geometry at 10M–30M. Our survey found no prior differentiable pure-FT model and no
per-net tree-crossing evaluator at this scale.

> 講者備忘：兩條線——模組粒度 FT-aware 規劃（決定幾何）與 cell 粒度 GPU 引擎（不懂 IO/FT）。
> 我們是交集：幾何固定、優化離散計數、cell 粒度、30M 規模。

---

## Slide 5 — GrandPlan（closest work）

**GrandPlan: Differentiable, Simultaneous Top-Level Floorplanning and Partition-Level Cell Placement for Large-Scale IP-Cores**
Z. Xiong, Y.-C. Lu, D. Z. Pan, H. Ren — ISPD 2026, pp. 64–72（NVIDIA Research + UT Austin）

- **Key idea**：GPU-accelerated、differentiable、end-to-end 同時優化 top-level floorplan 與
  partition-level cell placement（custom CUDA kernels）；產出乾淨的 rectilinear、exactly-abutted 分區邊界。
- **Method（3 coupled stages）**：(1) flat IP-core placement with differentiable grouping objectives；
  (2) boundary refinement via simulated annealing（area / routability constraints）；
  (3) routability-aware fence-region placement。
- **Results**：8 顆工業 IP-core（最大 25M cells）：total WL 最多 −14%、cross-partition（feed-through）WL
  平均 −27%（vs 人工 expert floorplan）；平均 runtime ≈1.2 h。

**vs. Ours**

- Region 幾何是他們的 *output*——我們的設定裡是設計者給定的 *fixed input*
- 他們最小化 cross-partition **wirelength**；我們最小化**離散計數**（IO crossings、pure FT），用 exact per-net tree evaluator
- 無公開程式碼 → 只能當概念對照，不是可跑的 baseline

> 講者備忘：最接近的對手。差異講清楚：幾何是否可動、目標是線長還是計數。
> 風險：題目碰撞 → M1/M2 證據要盡快成型發表。

---

## Slide 6 — Analytical Placement with Routing-Topology Optimization

**An Analytical Placement Algorithm with Routing Topology Optimization**
M. Wei, X. Tong, Z. Cai, P. Zou, Z. Lin, J. Chen — ASP-DAC 2024, pp. 294–299（福州大學）
（期刊擴充版：*Integration (VLSI Journal)*, vol. 100, 2025, “looking-ahead RTO”）

- **Key idea**：HPWL-driven placer 忽略 net 內部 routing topology → 估計與 routed WL 有落差；
  在 GP 內建立以理想 RSMT topology 為基礎的 differentiable wirelength model，
  用 segment screening / tracing 讓內部點也有有效梯度。
- **Method**：GP 內 RSMT-based differentiable WL + post-GP cell refinement（swift density control）。
- **Results**：ICCAD-2015 suite：routed WL −3%、HPWL −0.8%、TNS −23.8%（vs SOTA analytical placer）。

**vs. Ours**

- 配方形狀與我們相同：analytical GP 裡放 tree-topology 項 + 週期性 exact 重評估
- 但目標是 routed wirelength——region-oblivious，無 IO/FT 概念
- 證明 per-net tree objective 能在規模上驅動 GP 梯度

> 講者備忘：老師指定的第二篇。它驗證了「tree 目標放進 GP」的可行性；我們把同樣的迴路接到 IO/FT 計數上。

---

## Slide 7 — FTAFP

**FTAFP: A Feedthrough-Aware Floorplanner for Hierarchical Design of Large-Scale SoCs**
Z. Li, K. Tian, J. Zhai, Z. Li, S. Kai, S. Xu, B. Yu, K. Zhao — ASP-DAC 2025, pp. 886–892（北郵 + 華為 + CUHK）

- **Key idea**：hierarchical SoC 重用下，feed-through 是穿過模組的連線，需要在被重用模組內
  加 buffer + port → 傷 routability、congestion、timing；先前幾乎沒有工作對 FT 建模，FTAFP 在
  floorplan 階段顯式建模並優化。
- **Method**：feed-through-aware floorplanning（module granularity），模組形狀/位置與 FT cost 共同優化。
- **Why it matters to us**：確立 FT 是 2025 學術界的一級目標，且其顆粒度正好在我們上一層。

**vs. Ours**

- 優化 ~10² 個模組的形狀/位置；我們放 10⁷ cells 且 region 幾何固定
- FT 以模組間計；我們是 per-net tree-based 計數 + boundary-pin（IO）demand

> 講者備忘：FT 定義的工業動機來源。層級在我們上方（floorplan），互補不衝突。

---

## Slide 8 — Flora（“One Step Beyond”）

**One Step Beyond: Feedthrough & Placement-Aware Rectilinear Floorplanner**
Z. Xu, J. Wang, S. Xu, Z. Geng, M. Yuan, F. Wu — arXiv:2507.14914, 2025（中科大 + 華為諾亞方舟）

- **Key idea**：floorplanning 與後段 placement 脫節 → placement 次佳、feed-through 過多；Flora 把兩者耦合。
- **Method（3 stages）**：(1) wiremask + position-mask 共同優化 HPWL 與 FT；
  (2) fixed-outline、zero-whitespace 的模組 resize，同步精修 FT 與 placement；
  (3) tree-search 決定模組內 component 位置，再調整邊界。
- **Results**：HPWL −6%、FT-pin −5.2%、FT-module −29.2%、component placement +14%（vs SOTA）。

**vs. Ours**

- 仍是 module-level rectilinear floorplanning（learning / search based）
- 佐證 FT+placement 耦合是 2025 的活題——但沒有 GPU-scale cell placement、沒有 count-exact evaluator

> 講者備忘：FT 系譜第三篇；FTpin/FTmod 的指標切法可對照我們的 per-net 計數。

---

## Slide 9 — Feedthrough Insertion Methodology（industry）

**Die Area Reduction by Decongesting Top Channels Using Novel Feedthrough Insertion Methodology in Hierarchical SoC Designs**
R. Sakariya, S. Aich, V. Joshi, R. Griesmer — ISQED 2025

- **Key idea**：channel-based hierarchical SoC 裡，top-channel congestion 決定 die area；
  在 floorplan 階段插 feed-through 可疏通 top channels → 省 die area。
  人工 / flatten 的 FT insertion 在規模下失效：top-level 連線量巨大、RTL 耦合、
  sub-chip port porting、LEC coherence。
- **Method**：整合進 hierarchical flow 的自動 FT-insertion methodology。
- **Relation to our setting**：channel-based 與 channelless 是 hierarchical 整合的兩種型態——
  我們 Phase 1 用 channelless（fully-abutted regions）cost model，Phase 2 規劃 channel-style cost switch；
  我們的 placement 每少產生一個 feed-through，這類 methodology 就少插一個。

**vs. Ours**

- 下游補救：把設計已經需要的 FT 插好
- 我們在上游：placement 直接決定歸屬，讓需要的 FT 變少
- 產業證詞：FT 數量是一級成本（die area、TAT）

> 講者備忘：用它回答「為什麼 FT 值得優化」——工業界為了它專門做 methodology。

---

## Slide 10 — HiDaP

**Multilevel Dataflow-Driven Macro Placement Guided by RTL Structure and Analytical Methods**
A. Vidal-Obiols, J. Cortadella, J. Petit, M. Galceran-Oms, F. Martorell — IEEE TCAD 40(12), 2021（UPC + eSilicon）

- **Key idea**：RTL hierarchy + dataflow affinity（register latency、flow width）承載設計者意圖，
  flat synthesis 會丟掉——用它們驅動 multilevel macro placement。
- **Method**：hierarchy tree → blocks（macros + std cells）；top-down slicing-structure layout；
  adaptive multi-objective cost（WL、timing、overlap、preferred locations）+ spectral / force-directed hints。
- **Results**：大型工業設計上勝過商用/學術 placer 與 expert 手工 floorplan（WL 與 timing）；post-route 接近 closure。

**vs. Ours**

- Macro/block 粒度的 top-level planning，無顯式 IO/FT 計數目標
- 支持我們 benchmark 的前提：RTL hierarchy ≈ 天然 physical regions（真實設計上的 region ground truth）

> 講者備忘：引它支撐「RTL tile/group/cluster 邊界當 region」的合理性。

---

## Slide 11 — DREAMPlace Lineage（the engine we build on）

1. **DREAMPlace**（DAC 2019 / TCAD 2021）— 把 ePlace/RePlAce 靜電模型寫成 CUDA tensor 上的
   「訓練神經網路」：GP 加速 30×+，10M-cell GP 分鐘級。Lin, Jiang, Gu, Li, Dhar, Ren, Khailany, Pan。
2. **DREAMPlace 3.0**（ICCAD 2020）— multi-electrostatics fence-region placement（region constraints）
   —— 我們 two-stage baseline 的機制。Gu, Jiang, Lin, Pan。
3. **DREAMPlace 4.0**（DATE 2022 / TCAD 2023）— timing-driven momentum-based net weighting
   —— M1 IO-reweighting 借用的 net-weight 機制。Liao, Guo, Guo, Liu, Lin, Yu。

**vs. Ours — engine, not competitor**：系譜全都看不到 cross-region IO/FT。
我們的整合 = 5 行 iteration callback + region/evaluator stack；fork 其餘不動。

> 講者備忘：交代我們對 DREAMPlace 的改動極小（5 行 callback + shapely 相容 patch），其餘都在自己的 repo。

---

## Slide 12 — The 1M → 30M Benchmark Ladder

| Tier     | Design / source                                                              | Scale                       | Status                                                 |
| -------- | ---------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------ |
| 1M-class | ISPD'05/06 bigblue4 · newblue7；ISPD'15 superblue12（LEF/DEF, real fences） | 2.18M / 2.51M / 1.29M cells | In use — M0/M1 numbers on adaptec1 (0.21M) + bigblue4 |
| 5M       | MemPool half-cluster ≈ 2 × mempool_group（NanGate45, OpenROAD flow）       | ≈5.7M（group ≈ 3.1M）     | To assemble — 四階中優先度最低                        |
| 10M      | ISPD 2025 mempool_cluster — full LEF/DEF/LIB/SDC + placed DEF               | 11.31M cells / 12.71M nets  | On disk (46 GB suite) — golden workhorse              |
| 30M      | TeraPool reduced（open RTL；ISPD'24 simplified format reaches ≈50M）        | ~30M target                 | To build — moderate effort                            |

- **Region overlay**：grid {2×2 … 8×4} + seeded random slicing，snap 到 512² lattice（exact counting），
  K ∈ {8, 16, 32}；真實設計後續用 DEF REGION/GROUP + RTL-hierarchy ground truth（tile / group / cluster）
- **Realism gate**：禁止 naive replication（cut 崩到 0、Rent p → 0）——每個擴增 case 用 RentCon 驗證，
  目標 p ≈ 0.6–0.75；fallback：Rent-gap glue nets、ArtNet 合成
- **Experiment matrix**：scale × K × region-type；1M 級 ≥3 seeds；報 IO、pure-FT、HPWL / tree-WL、GP runtime、peak GPU memory

> 講者備忘：誠實講：目前真正跑過的只有 adaptec1 與 bigblue4；10M 的 mempool_cluster 已在硬碟上，30M 要自己組。

---

## Slide 13 — Phase-1 Pipeline

`0 · Inputs & preprocess` → `1 · GPU global placement` → `2 · Decode` → `3 · Refine` → `4 · Output` → ↳ per-region Innovus P&R

0. **Inputs & preprocess**（implemented，我們的 code）：netlist + K partition regions（幾何固定，設計者給）
   → 512² region-id grid + boundary table；Mt-KaHyPar warm start
1. **GPU global placement**（partial，**唯一在 DREAMPlace 內的 stage**——我們的 fork：加目標項 + 5 行 iteration callback）：
   `min WA-WL + λ_D·density + λ_IO·L_IO + λ_FT·L_FT + λ_pin·L_pincap`
2. **Decode**（planned，我們的 code）：soft membership → hard assignment（conditional expectation）
3. **Refine**（planned，我們的 code）：boundary-local trial moves / bin-FM
4. **Output**（implemented，我們的 code）：placement + block→region membership + per-net IO/FT report

**Evaluator loop（stage 1 內，每 N = 50–100 iterations）**——是回饋訊號，不是停止門檻（GP 停止由 DREAMPlace 收斂決定）：
GPU MST per net（degree-bucketed batched Prim）→ boundary crossings via 2-D prefix sums（O(1)/edge）
→ FT via K-bit region masks（popcount(passed & ~pins)）→ feed back：report · λ re-normalization · per-net weights

**↳ Downstream（Exp. 1）**：最終 block→region membership → 各 region 用 Innovus 跑 P&R → 組裝後的 top-level QoR
（routed WL · timing · congestion）

狀態圖例：implemented (M0/M1) ｜ partial — differentiable IO/FT = M2/M3 ｜ planned

> 講者備忘：五格裡只有 stage 1 在 DREAMPlace；0/2/3/4 是我們包在引擎外的 code。GP 產出 softmax 軟歸屬，
> 所以需要 decode（軟→硬）與 refine（邊界清理），這兩步是 M2 之後的規劃。Evaluator 開銷 <2.5%。
> 設計選擇：region 幾何固定 + 位置誘導歸屬 ⇒ 主流程一個 global density field 就夠。

---

## Slide 14 — Status & Evidence: M0 + M1 Complete (13/13 tasks)

**Built & verified（33+ unit tests）**

- Exact GPU evaluator：per-net MST + prefix-sum crossings + bitmask FT——所有計數與 CPU golden bit-match
- Warm evaluate 0.16 s（adaptec1）/ 1.32 s（bigblue4）；2M 合成 case 上 174× vs CPU；佔 placement runtime <2.5%
- Baselines：flat DREAMPlace + post-hoc counting；two-stage Mt-KaHyPar → greedy+2-opt region mapping → fence-constrained placement
- Net-reweighting closed loop（5 行 DREAMPlace callback）：`w = 1 + α·min(crossings, cap)`

**Evidence → next step**

- **M0**：two-stage IO = 1.2–1.8× flat，儘管其 min-cut bound ≤ flat →
  損失在「hypergraph cut → geometric tree crossings」的轉換，這個 gap 就是研究目標
- **M1**：reweighting 把 pure-FT 降 −10% … −49%（HPWL +≈1%），但 IO 最佳只 −0.06%（run noise ±1%）
  → WL 權重能壓縮路徑，翻不動 region 歸屬
- ⇒ **M2**：differentiable soft membership（softmax over region distances、τ annealing），給「哪個 region」一個梯度

重點數字：**174×** evaluator 加速 ｜ **−49%** pure-FT（bigblue4）｜ **±1%** run-to-run 噪聲（→ 需 multi-seed）｜ **<2.5%** evaluator 佔比

> 講者備忘：本頁是核心。負結果誠實講：reweighting 只動得了 FT（路徑緊實度），動不了 IO（離散歸屬）；
> 單 seed 噪聲比效果大一個量級。這直接導出 M2。

---

## Slide 15 — Exp. 1: Designer Regions + Per-Region Commercial P&R（PROPOSED）

**Arm A · Baseline — designer flow**

1. Designer floorplan：region geometry + block→region membership
2. Per-region P&R with Innovus（identical scripts / constraints / effort）
3. Assemble → top-level metrics

**Arm B · Ours — membership re-decided**

1. **同一套 region 幾何；我們的 GPU IO-aware placement 重新決定 block→region membership**
2. Per-region P&R with Innovus（identical scripts / constraints / effort）
3. Assemble → top-level metrics

**Held fixed across arms**：region geometry（byte-identical）· tech/LEF/lib/SDC · Innovus 版本、
scripts、effort、seeds · per-region utilization caps · 兩 arm 用同一個 IO/FT evaluator

**Compare**：routed WL · WNS/TNS · # feed-through nets & ports · top-channel congestion / DRC ·
per-region utilization · turnaround time

*Needs：有真實 designer floorplan 的 testcase + Innovus access。建議加第三條 arm——IO-agnostic
assignment（greedy+2-opt）——把「IO-aware 的功勞」與「任何最佳化的功勞」分開。*

> 講者備忘：這是提案（repo 尚無 Innovus flow）。公平性關鍵：per-region utilization 必須鎖住，
> 否則贏的可能是容積重分配。需要老師協助：testcase 與 license。

---

## Slide 16 — Exp. 2: vs Region-Oblivious Flat DREAMPlace

三條 arm：**Flat DREAMPlace**（不知道 region；post-hoc 計數）｜ **Two-stage**（Mt-KaHyPar → fence placement）｜ **Ours**（IO-aware GP：M1 reweighting → M2 differentiable）

**Preliminary（bigblue4, K=16 grid, single seed — M1 reweighting only）**

| Metric             |    flat | two-stage |       ours (M1) |        Δ vs flat |
| ------------------ | ------: | --------: | --------------: | ----------------: |
| IO count           | 100,108 |   178,594 |         100,736 | +0.6%（≈ noise） |
| Pure feed-throughs |   7,045 |    77,899 | **3,585** | **−49.1%** |
| HPWL               | 748.5 M | 4,459.8 M |         756.2 M |             +1.0% |
| Runtime (s)        |     545 |     3,738 |             535 |              −2% |

**Read with care**：single seed；IO 噪聲 ≈ ±1%；reweighting-only——M2 differentiable membership 才是壓 IO 的機制。

- **Protocol**：scales 1M → 30M × K ∈ {8,16,32} × {grid, rectilinear}；1M 級 ≥3 seeds；
  各 arm 同 config 與 iteration 預算；region 在任何 run 之前凍結
- **Fairness**：HPWL / runtime / memory 一律共同報告（絕不單報 IO）；報完整 λ-sweep Pareto，不報單一調好的點

> 講者備忘：現有數據誠實呈現：FT 大降、IO 持平。強調公平性協定（凍結 region、同預算、Pareto）。

---

## Slide 17 — Summary: Where We Are, and What We Need

- **Related work**：feed-through 優化在 2025–26 很熱（GrandPlan、FTAFP、Flora）——但全是模組粒度或
  wirelength-based；count-based、cell-level、fixed-region 的形式仍是空白
- **Benchmarks**：1M → 30M 階梯已定義；10M 的 ISPD'25 mempool_cluster 已在硬碟；每個擴增 case 過 Rent-exponent realism gate
- **Status**：M0 + M1 完成——exact GPU evaluator（174×、<2.5% overhead）；reweighting 負結果已診斷 → M2 differentiable membership
- **Experiments**：兩條實驗軌已規格化——designer regions + per-region Innovus P&R（只換 membership），
  以及 region-oblivious flat DREAMPlace（凍結 region、multi-seed、Pareto 協定）

**Asks**

1. 有真實 designer floorplan 的 testcase + Innovus access（Experiment 1）
2. M4 scale-up（10M / 30M）的 H100 配額
3. Benchmark 公信力計畫的意見：region 來源、multi-seed 預算

> 講者備忘：若老師問最大風險：與 NVIDIA GrandPlan 的題目碰撞——M1/M2 證據要盡快成型。

---

## 附錄：本次文獻搜尋紀錄（paper-search-mcp + Zotero）

| # | Work                                                        | Venue                                 | 與本研究關係                                 |
| - | ----------------------------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 1 | GrandPlan (Xiong, Lu, Pan, Ren)                             | ISPD 2026                             | 最接近；幾何可動、目標為跨區 WL；無 code     |
| 2 | Analytical Placement w/ Routing Topology Opt. (Wei et al.)  | ASP-DAC 2024（ext. Integration 2025） | GP 內 tree-topology 目標的可行性證明         |
| 3 | FTAFP (Li et al.)                                           | ASP-DAC 2025                          | FT 顯式建模（floorplan 層）                  |
| 4 | Flora / One Step Beyond (Xu et al.)                         | arXiv 2025                            | FT + placement 耦合（floorplan 層）          |
| 5 | FT Insertion Methodology (Sakariya et al.)                  | ISQED 2025                            | 產業動機：FT ↔ die area / TAT               |
| 6 | HiDaP (Vidal-Obiols et al.)                                 | IEEE TCAD 2021                        | RTL hierarchy ≈ region 的前提支撐           |
| 7 | DREAMPlace / 3.0 / 4.0 (Lin et al.; Gu et al.; Liao et al.) | DAC'19/TCAD'21; ICCAD'20; TCAD'23     | 引擎；fence-region 與 net-weighting 機制來源 |

搜尋涵蓋 DBLP / OpenAlex / Semantic Scholar / arXiv / Google Scholar 與本機 Zotero 庫
（GrandPlan、ASP-DAC'24 兩篇在 Zotero 已有完整條目）。DBLP 對 "Piano"（spec 提及的
floorplanning 相關工作）查無結果，未列入。
