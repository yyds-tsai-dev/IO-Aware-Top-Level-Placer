# GPU Placer 基礎設施與快速 WL/IO Evaluator 文獻調查

- 日期:2026-07-30
- 目的:為「IO-aware top-level placement」專案(10M–30M cells flat GPU global placement,objective = wirelength + IO number + pure feed-through number)盤點可用的 placer codebase 與 evaluator 技術。
- 證據等級標註:**[全文]** = 已讀論文 PDF 全文;**[摘要]** = 只讀 abstract;**[README]** = GitHub README/文件;**[網頁]** = 搜尋結果摘要;**[未驗證]** = 依訓練知識或間接來源,未經一手確認。
- 下載之 PDF 暫存於 session scratchpad(`PLACE_DAC2019_Lin.pdf`、`PLACE_TCAD2020_Lin.pdf`、`PLACE_ICCAD2020_Gu.pdf`、`PLACE_DATE2022_Liao.pdf`、`ROUTE_ICCAD2024_Zhao.pdf`、`ROUTE_DATE2022_Liu.pdf`、DG-RePlAce 全文)。

---

## TL;DR

1. **10M cells flat GPU GP 有直接公開證據**(DREAMPlace 在單顆 V100 上完成 10.5M-cell industrial design)**;30M cells 無 placer 直接證據,但 GPU global router 已在單顆 A100 上處理 50M cells / 60M nets**(ISPD 2024 contest / HeLEM-GR / InstantGR),記憶體量級推算顯示 30M cells placement 在單顆 40–80GB GPU 上可行。
2. **最適合加自訂 differentiable objective 的 codebase 是 DREAMPlace**(PyTorch autograd + custom C++/CUDA ops 架構,加一個新 objective term = 加一個 op),Xplace 為次選(同為 PyTorch、較新較快、但 fence/region 支援弱)。
3. 現有 fence region 機制(DREAMPlace 3.0、MORPH)全部假設 **cell→region 歸屬是輸入**;我們「region 幾何固定、歸屬由 placement 決定」的設定在 flat analytical placement 文獻中沒有現成解。**最接近的工作是 NVIDIA 的 GrandPlan(ISPD 2026,25M cells,differentiable grouping + feedthrough WL)**,但其 region 幾何是輸出且未最小化 boundary-pin 數;「per-net MST tree-edge 跨 region boundary 計數作為 IO/feedthrough evaluator」**未找到現成文獻,屬我們的機會點**。

---

## 主題 1:GPU / Analytical Placers

### 1.1 DREAMPlace 系列(1.0 → 4.3)

核心論文與版本演進(版本號依官方 README [README];論文依 dblp/PDF):

| 版本 | 論文 / venue | 新增內容 | 來源 |
|---|---|---|---|
| 1.0 | Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement", **DAC 2019**;期刊版 **TCAD vol.40, pp.748–761, 2021**(DOI 10.1109/TCAD.2020.3003843) | 把 ePlace/RePlAce 類 electrostatics GP 寫成「Python/PyTorch optimizer + C++/CUDA 低階 ops」;GPU 加速 GP+LG | [全文] |
| 2.0 | Lin et al., "DREAMPlace 2.0: Open-Source GPU-Accelerated Global and Detailed Placement...", **ASP-DAC 2020** | GPU detailed placement(ABCDPlace)約 16× vs NTUplace3(million-size benchmarks);movable macros、Tetris-like legalization | [網頁][README] |
| 2.2 | — | routability 最佳化(外掛 NCTUgr) | [README] |
| 3.0 | Gu et al., "DREAMPlace 3.0: Multi-Electrostatics Based Robust VLSI Placement with Region Constraints", **ICCAD 2020**(DOI 10.1145/3400302.3415691) | **fence region 支援**(見 1.3) | [全文] |
| 4.0 | Liao et al., "DREAMPlace 4.0: Timing-driven Global Placement with Momentum-based Net Weighting", **DATE 2022**;期刊版 TCAD vol.42, pp.3374–3387, 2023(加 Lagrangian-based refinement) | timing-driven GP(momentum net weighting + preconditioner 修改),實驗於 **ICCAD 2015 contest benchmarks** | [全文(DATE 版)] |
| 4.1 | — | BB-step、two-stage macro placement flow | [README] |
| 4.2 | Y. Liu et al., "The Power of Graph Signal Processing for Chip Placement Acceleration", **ICCAD 2024**(arXiv 2502.17632) | **GiFt 初始化**:graph signal processing 初始解,免訓練;GiFt-DREAMPlace 迭代數 −33%、總 runtime −46% | [網頁/摘要] |
| 4.3 | — | HeteroSTA 整合(GPU-accelerated timing analysis) | [README] |

**規模 / runtime / 硬體證據(關鍵數字)**[全文,TCAD 2021 版]:

- 實驗平台:40-core Intel E5-2698 v4 + **1× NVIDIA Tesla V100**(Volta;論文未註明 16GB 或 32GB 版)。
- GP speedup vs RePlAce(40 threads):**38×(ISPD 2005)/ 47×(industrial suite)**;整條 placement flow GPU 4.6×。DAC 2019 版數字為 35×/43×、全流程 5×。
- 「**1M cells 約 1 分鐘完成;支援至 10M cells 的 industrial designs**」(兩版論文皆此表述)。
- Industrial benchmarks design1–design6:cell 數 1345K / 1306K / 2265K / 1525K / 1316K / **10504K**;net 數最大 **10747K**(design6 ≈ 10.5M cells、10.7M nets)——這是 DREAMPlace 系列公開的最大單一設計。
- 對照組 RePlAce(CPU)在該 10M design 於 Nesterov 第 6 次迭代 **crash(peak memory 超過 64GB 主記憶體)**;以其每迭代 7.5s 估算 GP 需 ≈10896s。DREAMPlace 順利完成(該設計跑 1000 次迭代;論文以圖表呈現 runtime,未給出單一總秒數數字)。
- 精度:同時支援 float64 / float32;**float32 相對 float64 在 GPU 上再快 1.3×、品質幾乎不變**。
- GP 內部 runtime 拆解:density 相關(FFT/DCT)佔約 73%、wirelength 約 27%(bigblue4)。
- 高扇出處理:預設 `ignore_net_degree = 100`,大於 100 pins 的 net 不進 WL gradient(引自 DG-RePlAce 論文 footnote 對 DREAMPlace 預設值的描述)[全文]。

**Multi-GPU**:官方 repo 與論文皆**無** multi-GPU 單一 placement 支援;DAC 2019 結論將 multi-GPU 列為 future work [全文]。本次搜尋亦**未找到**任何「單一 flat GP 切到多 GPU」的公開論文(AutoDMP 的 DGX 平行是「多組候選解/超參數試驗平行」,不是單一 placement 的分散式計算)[網頁]。→ 30M cells 需以**單卡放得下**為前提規劃(見主題 3)。

**Repo 現況**(github.com/limbo018/DREAMPlace)[README]:
- License:**BSD-3-Clause**。
- 相依:Python 3.5–3.9、**PyTorch 1.6/1.7/1.8/2.0**、CUDA 9.1+(GPU arch ≥ 6.0;無 GPU 時可純 CPU 跑)、GCC 7.5(C++17)、Boost ≥1.55、Bison ≥3.3、Limbo parser 等。
- 版本已到 **4.3.0**;master 824 commits、~1k stars。GitHub 無正式 Releases 頁(以 README 版本表為準),最後 commit 精確日期[未驗證],但 4.2.0(ICCAD 2024 GiFt)與 4.3.0(HeteroSTA)顯示 2024–2025 仍持續開發。
- Benchmark 支援:ISPD 2005、ISPD 2015、ICCAD 2015、MMS。輸入 Bookshelf 與 LEF/DEF。

### 1.2 其他 placers

| 系統 | venue / 年 | 定位 | 規模證據 | 開源 |
|---|---|---|---|---|
| **ePlace / ePlace-MS** | TODAES 2015 / TCAD 2015(Lu et al.)[未驗證-經典引用;內容經 Lu 博士論文摘要佐證[摘要]] | electrostatics(eDensity)+ FFT Poisson + Nesterov 的原型;CPU 單執行緒 | ISPD 2005/2006、MMS(百萬級) | 是(研究碼) |
| **RePlAce** | TCAD 2019(Cheng, Kahng, Kang, Wang;DOI 10.1109/TCAD.2018.2859220) | ePlace 工程強化(local smoothing、dynamic step size、routability);CPU;OpenROAD gpl 的基礎 | ISPD 2005/2006、MMS、DAC-2012/ICCAD-2012;HPWL 全面領先當年 SOTA;**在 10M design 上因 >64GB RAM crash**(DREAMPlace 論文實測) | 是(OpenROAD) |
| **Xplace** | DAC 2022(Liu, Fu, Wong, Young);期刊版 **TCAD vol.43, pp.1872–1885, 2024**;Xplace-Timing 於 ICCAD 2024 | GPU + PyTorch 的輕量高速 placement framework;**每次 GP 迭代約 3× 快於 DREAMPlace**(官方 README);含 GPU detailed placement、cell-inflation routability、**內建 GGR GPU global router**、GPU timer | ISPD 2005/2015 等(~2M cells 級);未見 10M 級公開數字 | 是,cuhk-eda/Xplace,**BSD-3**;deps:CUDA ≥11.3、PyTorch ≥1.12、CMake ≥3.24 [README] |
| **OpenROAD gpl** | — | RePlAce 的 C++ 重寫,CPU;RTL-to-GDS 全流程整合 | 百萬級 | 是 |
| **DG-RePlAce(= OpenROAD gpl2)** | arXiv 2404.13049;TCAD 2024(IEEE 10620224;Kahng & Wang) | OpenROAD 基礎上的 **C++/CUDA(無 PyTorch)** GPU GP,~14K LOC;dataflow-driven(ML accelerator 導向) | 論文 benchmark ≤ **1.06M std cells**;GP 平均 **22.49× vs RePlAce、1.75× vs DREAMPlace**;其 WL gradient kernel 在高扇出 net 上比 DREAMPlace 的演算法快 3.25×;DAC 2024 BOF 投影片宣稱支援 "up to ~10M instances"(投影片宣稱,論文未展示)[網頁] | 是,ABKGroup/DG-RePlAce-AutoDMP;整合入 OpenROAD src/gpl2(主線目前狀態[未驗證]) |
| **AutoDMP** | ISPD 2023(Agnesina et al., NVIDIA;DOI 10.1145/3569052.3578923) | DREAMPlace 之上的 macro+std-cell concurrent placement + multi-objective Bayesian 超參數搜尋 | **2.7M cells + 320 macros,3 小時,一台 DGX Station A100**(該工作站標準配置 4×A100;平行化為多組試驗層級) | 是,NVlabs/AutoDMP,**Apache-2.0**;綁定特定 DREAMPlace commit [README] |
| **MORPH** | ICCAD 2024(Mai, Zhang, Lin, Wang, Huang;DOI 10.1145/3676536.3676745) | region constraint 最新 SOTA:default/fence/guide **三類 hybrid region constraints** 統一 multi-electrostatic formulation + 二階資訊 | ISPD 2015:HPWL 改善 5.6–14.3%、overflow −10~24% vs SOTA region-aware placers | [摘要];開源狀態[未驗證] |
| **GrandPlan** | **ISPD 2026**(Xiong, Lu, Pan, Ren;UT Austin + NVIDIA;DOI 10.1145/3764386.3779591) | **GPU differentiable 同步 top-level floorplanning + partition-level placement**(見 1.4(c)) | **8 個 industrial IP-cores,最大 25M cells**;total WL 最多 −14%、**cross-partition(feedthrough)WL 平均 −27%** vs 人工 baseline;平均 1.2 小時 | [摘要];未見開源 |

另有 FPGA 分支 DREAMPlaceFPGA / DREAMPlaceFPGA-MP(ISPD 2023 / arXiv 2311.08582,支援 FPGA region constraints 與 cascade shapes),對本專案僅具參考價值 [dblp]。

### 1.3 現有 fence/region 機制怎麼做(DREAMPlace 3.0 細節)

[全文] DREAMPlace 3.0(ICCAD 2020):
- 問題定義:K 個 fence regions,每個 region 由一或多個 disjoint 矩形 sub-regions 組成;**「被指派到某 fence region 的 cells」必須放進該 region 內,未指派的 cells 不得進入**;另設 exterior region r_K 收容未指派 cells。
- 機制:**multi-electrostatics** —— 把單一 electrostatic system 拆成 **K+1 個獨立場**;對每個 region 以「virtual blockage insertion(把 region 外的空間切成矩形 virtual blockages)+ field isolation」建場;各 region 的 density/potential 獨立計算(region k 複雜度 O(|v_k| + M² log M)),可全部平行。
- 穩健化:self-adaptive quadratic density penalty、entropy injection;K+1 個 region 同步收斂控制後同時 legalize。
- 成果:在 **ISPD 2015 benchmarks(含 fence regions,例:superblue16_a 有 2 個 fence regions)**上,HPWL 比 region-aware placers Eh?Placer、NTUplace4dr 好 **>13%**、top-5 overflow 好 >11%;robustness 技術在 ICCAD 2014 與 ISPD 2019 suites 上比原版 DREAMPlace HPWL ~1%、runtime ~10% 改善(摘要原文表述)。
- MORPH(ICCAD 2024)延伸到 fence/guide/default 三類混合 region,仍以「歸屬給定」為前提 [摘要]。

### 1.4 三個問題的回答

**(a) 10M / 30M cells flat GP 可行性的公開證據**

- **10M:有直接證據。** DREAMPlace 在 1× V100 上完成 10.5M-cell / 10.7M-net industrial design(float64;1000 iterations)[全文]。GP runtime 論文以 scaling 曲線呈現「近線性」;以「1M ≈ 1 分鐘」線性外插 10M ≈ 10 分鐘級(外插推估,論文未給單一數字)。
- **25M:間接證據(placement 情境)。** GrandPlan 在 industrial IP-cores(最大 25M cells)上以 GPU differentiable flow 平均 1.2 小時完成 floorplanning+placement 全流程 [摘要]。
- **30M:placer 無直接證據;鄰近證據充分。** ISPD 2024 GPU/ML global routing contest 釋出 **最大 ~50M cells / 60M nets** 的 industrial cases;HeLEM-GR(ICCAD 2024)在 **1× A100** 上最大 case ~10 分鐘完成 routing [全文];InstantGR(ICCAD 2024,open source)同級 [網頁/README]。Routing 的 per-net/per-pin 資料結構規模與 placement 相當,說明 30M-cell 級的 netlist 資料結構 + GPU kernels 在單卡上工程可行。結論:**30M flat GP 是「工程外插可行、學術上尚無人展示」的空白區**——本專案可順帶補上這個數據點。

**(b) 要加自訂 differentiable objective(IO / feed-through 項),哪個 codebase 最適合?**

**首選 DREAMPlace**,理由:
1. **架構天然支援**:DAC 2019 論文原文即定位為「Python + PyTorch(optimizer/API)、C++/CUDA(低階 operators)」[全文];每個 objective(weighted-average WL、electric potential/density、RUDY、pin utilization…)是一個註冊為 autograd Function 的 op。加 IO/feed-through 項 = 新增一個 custom CUDA op(forward 算 soft crossing count、backward 回傳對 cell 座標的梯度),掛進 objective 加權和即可,不動核心迭代器。
2. **規模證據最強**(10.5M 實測)且 float32 路徑成熟(對 30M 記憶體重要)。
3. **3.0 的 multi-electrostatics 基礎設施可直接改造**成我們 Phase 1 需要的「per-region 容量/密度控制」(見 (c))。
4. BSD-3、持續維護、AutoDMP/GiFt 等外部專案已示範二次開發模式。
弱點:PyTorch 版本綁定偏舊(≤2.0)、build 系統(CMake+Limbo+Bison)較重。

**次選 Xplace**:同為 PyTorch、程式碼較新較精簡、單迭代 ~3× 快、自帶 GGR global router(Phase 2 router-based evaluator 現成起點)[README]。弱點:fence/region 支援不完整(README 僅見 `ispd2019_no_fence` 類格式處理)、最大公開規模 ~2M cells、社群較小。務實策略:**以 DREAMPlace 為主幹,把 Xplace 的 kernel 技巧與 GGR 當參考/移植來源**。

**不建議當主幹**:DG-RePlAce/gpl2(C++/CUDA 無 autograd,新梯度須手推手寫,迭代成本高;但其高扇出 WL gradient kernel 與 OpenROAD DB 整合值得借鑑)、OpenROAD gpl(CPU)、自建 from scratch(需重造 density/FFT/Nesterov/legalization 全套,至少數人月,且已有 BSD-3 成熟件)。

**(c) 現有 fence/region 機制 vs 我們的設定:本質差異**

- 現有機制(DREAMPlace 3.0、MORPH、NTUplace4dr、Eh?Placer):**membership 是輸入**(DEF REGIONS/GROUPS 由 designer 給),placer 只負責幾何上「關進去/擋在外」;multi-electrostatics 的場切割正是**以 membership 已知**為前提建構的。
- 我們的設定:**region 幾何是輸入,membership 是輸出**——cell 屬於哪個 partition 由其最終座標落點決定,objective(IO number、pure feed-through number)是 membership 的函數。這在數學上把 placement 變成「連續座標 + 隱式離散指派」的耦合問題,與 fence placement 的約束方向相反。
- 兩個重要推論:
  1. 若 Phase 1 對每個 region 有**容量/utilization 上限**,可把 3.0 的 multi-electrostatics 反過來用:membership 用 soft assignment(cell 對每個 region 的隸屬機率,由位置的 smooth indicator 決定),各 region 的 density 場僅對「機率加權面積」收費——機制可重用,語義相反。
  2. 最近鄰工作:**GrandPlan(ISPD 2026)**已做「flat placement + differentiable grouping objectives → 邊界細化 → fence-region placement」三段式,證明此方向工業可行且值得(cross-partition WL −27%)[摘要]。差異:GrandPlan 的 partition 幾何是**輸出**(SA refinement),且其目標是 cross-partition **wirelength**;我們是幾何**固定**(Phase 1 給定)、目標是 **boundary-pin 數(IO number)與 pure feed-through 數**這兩個離散計數——objective 與 evaluator 都不同。另一個結構同型的先例:**ePlace-3D**(ISPD 2016)在連續 placement 中同時決定 cell 的 tier 歸屬並最小化 vertical interconnect(VI)數(−9~10% VI)[摘要]——「跨界計數 + 歸屬由 placement 決定」的垂直版類比,可引為方法論支撐。

---

## 主題 2:快速 Wirelength / IO Evaluator(~30M nets)

### 2.1 RSMT / MST 建構

| 方法 | 出處 | 關鍵事實 | 來源 |
|---|---|---|---|
| **FLUTE** | Chu & Wong, TCAD 27(1), 2008(DOI 10.1109/TCAD.2007.907068) | Lookup table RSMT:**degree ≤ 9 最優**(LUT 直查);degree ≤ 100 仍高準確(net-breaking);速度「幾乎與高效 Prim RMST 實作一樣快」;accuracy/runtime 可調 | [摘要] |
| **GeoSteiner** | Juhl, Warme, Winter, Zachariasen, Math. Prog. Comp. 10(4), 2018 | exact RSMT/ESMT solver;最強品質、指數級最壞情況;適合做 golden 校驗,不適合 30M-net 吞吐 | [標題/引用鏈] |
| **REST** | J. Liu, G. Chen, E.F.Y. Young, **DAC 2021**(DOI 10.1109/DAC18074.2021.9586209)——注意:**是 DAC 2021,非 NeurIPS 2021**(任務描述有誤,已更正) | RL 建 RSMT;≤50 pins 的 net 平均長度誤差 ≤0.36%;每 net <1.9ms(可 GPU batch) | [摘要] |
| **NN-Steiner** | Kahng, Nerem, Wang, Yang, AAAI 2024 | neural + Arora PTAS 混合框架,泛化到大 instance | [摘要] |
| **GAT-Steiner** | Onal et al., arXiv 2407.01440(2024;venue [未驗證]) | GNN 預測 Steiner points,GPU 平行;ISPD19 nets 99.85% 完全正確 | [摘要] |
| **GPU MST(Borůvka)** | Prokopenko, Sao, Lebrun-Grandié, ICPP 2022(DOI 10.1145/3545008.3546185) | single-tree Borůvka EMST on GPU(ArborX/Kokkos);**37M 點 EMST < 0.5s(1× A100)**;比最快多執行緒 CPU 快 4–24× | [摘要] |

註:我們的工作負載是「**數千萬個小 MST**」而非一個大 MST——每 net 獨立、degree 多在 2–10,天然 embarrassingly parallel,比 ICPP 2022 的單棵大樹問題更容易 GPU 化(bucket-by-degree + 每 warp 一個 net 的 O(d²) Prim);上表僅證明 GPU 在此資料量級的 MST 運算力綽綽有餘。

### 2.2 GPU Global Routers(當 evaluator 用)

| Router | venue / 年 | 規模 / 數字 | 開源 |
|---|---|---|---|
| **GAMER** | ICCAD 2021;TCAD 42(2) 2023(DOI 10.1109/TCAD.2022.3184281) | GPU maze routing(multi-source multi-destination shortest path);coarsened maze routing 加速 16× | [網頁];開源狀態[未驗證] |
| **FastGR** | Siting Liu et al., DATE 2022(有 TCAD 延伸版[未驗證]) | CPU-GPU 異質 task graph + GPU pattern routing(batch 10.877× vs CPU 逐 net);整體 2.426× vs SOTA(CUGR);ICCAD 2019 benchmarks;RTX 2080 | [全文] |
| **InstantGR** | Shiju Lin et al., **ICCAD 2024**(DOI 10.1145/3676536.3676787);另有 "Scalable GPU Parallelization for 3-D Global Routing" 更新版[未驗證] | ISPD 2024 contest benchmarks(至 50M cells);contest 優勝級品質與速度 | **是**,cuhk-eda/InstantGR,BSD-3;研究碼(僅 2 commits)[README] |
| **HeLEM-GR** | Zhao, Guo, Wang, Wen, Liang, Lin(PKU), **ICCAD 2024**(DOI 10.1145/3676536.3676650) | linearized exponential multiplier + 異質 routing kernels;**ISPD 2024 全部 cases(至 ~50M cells / 60M nets)於 1× A100;最大 case ~10 分鐘**;品質分數比 top-3 winners 好 4.8–5.8%、快 1.62–2.07×;C++/CUDA/Taskflow/CUB | [全文] |
| (contest) | "GPU/ML-Enhanced Large Scale Global Routing Contest", ISPD 2024(NVIDIA 主辦;DOI 10.1145/3626184.3639693);ISPD 2025 續辦 performance-driven 版 | 釋出 50M cells / 60M nets 級 industrial benchmarks——**本專案 Phase 2 evaluator 的規模對標與現成測資來源** | [網頁/全文引述] |

### 2.3 「MST/RSMT tree-edge 跨 region boundary 估 IO / feedthrough」有現成文獻嗎?

**核心答案:未找到。** 在 arXiv / OpenAlex / Semantic Scholar / dblp / web 多輪檢索中,沒有任何工作把「flat placement 迭代中,用 per-net MST/RSMT 的 tree edges 與固定 partition region boundaries 的交點數」當作 IO(boundary pin)數 / pure feed-through 數的 fast evaluator。**此為本專案的機會點。** 最接近的相鄰工作(皆不等同):

- **GrandPlan(ISPD 2026)**[摘要]:唯一在 flat GPU placement 中最佳化 cross-partition 連線的工作,但 (i) 最佳化的是 cross-partition **wirelength** 而非 boundary-pin/feedthrough **計數**,(ii) partition 幾何是輸出,(iii) 未描述 tree-based crossing evaluator(以摘要所及)。
- **Feedthrough-aware floorplanning 系列**(模組級、非 flat cell placement):**FTAFP**(ASP-DAC 2025,DOI 10.1145/3658617.3697728;feedthrough 建模 + floorplan 最佳化)[摘要];**Flora**(arXiv 2507.14914,2025;FTpin/FTmod 指標,三段式 floorplanner)[摘要];**Piano**(arXiv 2508.13161,2025;pin-assignment-aware,以 graph shortest path 評 feedthrough 與 unplaced pins ——與我們 Phase 3 IO legalization 相關)[摘要]。這些確認了「feedthrough 數量」正成為活躍 objective,但全部在 floorplanning/模組層級,無 per-net tree crossing evaluator、無 10M+ 規模。
- **ePlace-3D**(ISPD 2016)[摘要]:連續 placement 中最小化跨 tier interconnect 數——結構同型(crossing count + 歸屬由 placement 決定),但是 3D/TSV 語境且用 density-based 機制,非 tree-crossing evaluator。
- **Two-Level Rectilinear Steiner Trees**(Held & Kämmerling, arXiv 1501.00933;Comput. Geom. 2017)[摘要]:pin 集合被分群後的兩層 RSMT 理論(2.37-approx / PTAS)——可作為我們「per-net 跨 partition 的樹長下界/近似」的理論參照。
- 傳統 min-cut placement(Capo 世系)與 terminal propagation:歸屬由 partitioning 決定且最小化 cut nets,但 objective 是 cut 數非 boundary-pin 幾何位置,且非 analytical/GPU 世代 [未驗證-背景知識]。

### 2.4 RUDY(congestion 估計,順帶)

RUDY(Spindler & Johannes, DATE 2007,DOI 10.1109/DATE.2007.364463):每 net 以 bounding box 內均勻 wire density 疊加成 congestion map,無需 router、O(#nets),DREAMPlace/Xplace 均內建 GPU 版;我們的 Phase 1 可直接沿用,或把「跨 boundary 的 demand」投影到 boundary 段上做 IO 容量的早期壓力指標 [摘要]。

### 2.5 30M cells 下 per-net MST + crossing count:複雜度與 GPU 實作策略

設 N=30M cells、nets ≈ 30M、pins ≈ 1.2 億(平均 degree ~4)、partitions K ≈ 10–100(矩形或 rectilinear、幾何固定)。

**建樹**:
- 依 degree 分桶:d=2 直接;d=3 有 closed-form L/Z 拓撲;4 ≤ d ≤ 32 每 warp 一 net 跑 O(d²) Prim(Manhattan 距離,on-the-fly 算距離不存矩陣);32 < d ≤ ~256 每 block 一 net;d > `ignore_net_degree`(~100–1000,約佔 net 數 <1%)不建樹,改用「**region-presence 下界**」:IO 下界 = (該 net pins 觸及的 distinct region 數 − 1)——這同時是任何 routing 的**精確下界**,對 clock/reset 類大 net 反而比樹估計更穩。
- 總工作量 Σd² ≈ 30M × ~20 ≈ 6×10⁸ 距離運算——A100 上 kernel 時間為毫秒~數十毫秒級(**推估,未驗證**;對照:37M 點的完整 EMST 都 <0.5s [ICPP 2022 摘要])。
- 品質:RMST ≤ 1.5 × RSMT(Hwang bound,經典結果[未驗證-教科書級]);若要更貼 router,可對 4 ≤ d ≤ 9 上 GPU 版 FLUTE-LUT(POWV 表查詢本質是 per-net 獨立查表,適合 GPU;REST/GAT-Steiner 已示範 per-net batch 神經替代)。

**跨界計數**:
- 前處理:把固定 region 幾何 rasterize 成 die 上的 **region-id lookup grid**(uniform grid;30M cells 規模下 4096²×int16 ≈ 34MB,可忽略),另存每條 boundary 的線段集合。
- 每條 tree edge(全晶片 ≈ pins − nets ≈ 9×10⁷ 條)以 L-shape 內嵌:若 partitions 構成軸對齊 grid,crossing 數 = x-span 內垂直 boundary 線數 + y-span 內水平 boundary 線數(**與 L 的方向選擇無關**);一般 rectilinear partition 則沿兩段線段做 region-id walk。每 edge O(平均跨界數) → 總量 ~10⁸ 級,毫秒級 kernel。
- **pure feed-through 判定**:每 net 先做 pin 的 region bitmask(K ≤ 64 用一個 uint64;K ≤ 256 用 4×uint64);edge walk 經過 region r 且 bitmask[r]=0 ⇒ 該 net 在 r 為 pure feed-through。O(edges × 平均經過 region 數)。
- **整體單次評估 < 0.1–1 s @30M nets(推估,未驗證)**;對照 full GR(HeLEM/InstantGR)在 60M nets ~10 分鐘 [全文] ——快 3 個數量級,適合放進 GP 迭代內圈(每次或每 K 次迭代);full GR 留給 Phase 2 精評與校正。
- **梯度**:計數本身離散;做進 objective 需 smooth surrogate——對每條 edge 每條 boundary 用 sigmoid(位置−boundary)的乘積形式(等價於 soft crossing indicator),或 cell-level soft membership(如 (c) 所述)+ net-level 的 soft「region presence」;以 DREAMPlace 的 autograd op 形式實作,forward 用精確計數(報表)、backward 用 surrogate 梯度(straight-through 亦可)。

**何種精度足夠**:IO/feed-through 是**拓撲計數**,不是長度——當 net span ≪ region 尺寸時(絕大多數 local nets),MST 拓撲與 router 實走對「跨了哪條 boundary」幾乎一致;誤差集中在 (i) 貼著 boundary 的 nets(L 方向歧義,在 grid partition 下已消除大半)、(ii) congestion 繞行(系統性低估,Phase 2 router evaluator 校正)、(iii) 高扇出 nets(用 region-presence 精確下界)。結論:**MST-level evaluator 對 Phase 1 的梯度方向與相對排序足夠;絕對值以 Phase 2 router-based evaluator 定期校正**(此主張為我們的分析推論,非文獻結論)。

---

## 主題 3:30M cells 的 GPU 記憶體足跡推算

**推算假設**:N=30M movable cells、nets=30M、pins=1.2 億(平均 4 pins/net);float32(4B)+ int32 index;density grid 4096²;Nesterov 保留 ~8 份 2N 向量(u/v/g 各當前+前一步等);WA wirelength 採 merged kernel(per-net 累加器 + per-pin 中間值)。結構參照 DREAMPlace 論文對其資料流的描述 [全文] 與其開源 ops 一般結構 [未驗證-程式碼層細節]。

| 類別 | 估算 | float32 佔用 |
|---|---|---|
| per-cell(pos/size/grad + Nesterov 狀態) | ~110 B/cell × 30M | ~3.3 GB |
| per-pin(offset、pin2net/pin2node、pin pos、grad、WA 中間值) | ~50 B/pin × 120M | ~6.0 GB |
| per-net(CSR、weight、mask、WA 累加器) | ~33 B/net × 30M | ~1.0 GB |
| density / FFT(4096² × ~8 面) | | ~0.5 GB |
| PyTorch context、workspace、碎片 | | ~2–4 GB |
| **小計(GP)** | | **~13–15 GB** |
| float64 全程 | ×~1.9 | ~25–30 GB |
| 本專案新增(region-id、per-net bitmask、MST edges、crossing buffers) | | +1–2 GB |

**經驗錨點交叉檢核**:(1) 10.5M-cell design 以 float64 在單顆 V100(16 或 32GB,論文未註明)完成 [全文] ⇒ 每 M cells ≤1.6–3.2 GB(float64 上界),線性外推 30M float32 ≤ 24–48 GB——與 bottom-up 估算(~0.45–0.5 GB/M cells,float32)一致偏保守。(2) HeLEM-GR 在 1× A100 上處理 60M nets 的 routing 資料結構 [全文]。
**結論:30M cells 的 flat GP 在 A100/H100 40GB(float32)可行、80GB 從容(可容 float64 或更肥的 evaluator);V100-32GB 屬邊緣,需 float32 + 精簡狀態。**(整段為推算,非實測;實作後應以 `torch.cuda.max_memory_allocated` 校正。)

---

## 綜合分析

### 基礎設施選項比較

| 選項 | autograd 擴充性 | region 機制現成度 | 規模證據 | 維護/授權 | 對本專案的改造成本 |
|---|---|---|---|---|---|
| **fork DREAMPlace** | ◎(PyTorch + custom CUDA ops,即插 objective) | ◎(3.0 multi-electrostatics 可反轉復用) | ◎(10.5M 實測) | BSD-3;活躍(4.3.0) | **低–中**:新增 MST/crossing ops + soft membership;痛點是舊 PyTorch 與重 build 鏈 |
| fork Xplace | ◎(PyTorch,碼小) | △(fence 不完整) | ○(~2M 實測;kernel 快 3×/iter) | BSD-3;活躍 | 中:要自補 region 基建;GGR 可直接當 Phase 2 evaluator 起點 |
| DG-RePlAce / gpl2 | ✕(純 C++/CUDA,無 autograd) | △ | ○(≤1.1M 論文;10M 為投影片宣稱) | BSD-3(OpenROAD 系) | 高:每個新 objective 手推梯度;優點是 OpenROAD DB/流程整合 |
| OpenROAD gpl(CPU) | ✕ | △ | ✕(30M 不現實) | — | 不適用(規模) |
| 自建 from scratch | ◎(自由) | ✕ | ✕ | — | 最高:重造 density/FFT/Nesterov/LG 全套,無必要 |

### 建議(初步)

1. **主幹:fork DREAMPlace(4.x)**。Phase 1 以 float32 跑 30M;把 IO/feed-through 做成兩個新 ops:`mst_crossing_eval`(forward:batched Prim/FLUTE-LUT + region-id walk,精確計數)與 `soft_io_objective`(backward:sigmoid/soft-membership surrogate 梯度);region 容量控制反用 3.0 的 multi-electrostatics(soft assignment 版)。同時把 MORPH(ICCAD 2024)當 region-constraint 數值穩健化的參考文獻。
2. **Evaluator 對照組**:Phase 1 內圈用自研 MST-crossing(~亞秒級/次,推估);Phase 2 接 GPU global router——優先評估 **InstantGR(BSD-3,開源)** 改造為「crossing-count reporter」,並以 ISPD 2024 contest benchmarks(50M cells/60M nets)做規模壓測;HeLEM-GR 論文的 LEM formulation 是 congestion-aware 版本的參考。
3. **速度優化備援**:若 DREAMPlace kernel 成瓶頸,參考 Xplace 的 per-iteration 優化與 DG-RePlAce 的高扇出 WL gradient kernel(其論文顯示對 >100-pin nets 快 3.25×);初始化可開 GiFt(4.2.0 內建)換 −46% runtime。
4. **定位聲明(寫論文用)**:與 GrandPlan(ISPD 2026)的區隔 = 「region 幾何固定 + boundary-pin/pure-feed-through 計數 objective + per-net tree-crossing fast evaluator + 30M 規模」;與 FTAFP/Flora/Piano 的區隔 = flat cell-level placement(非模組級 floorplanning);與 ePlace-3D 的區隔 = 2D 多 partition 且 evaluator 是 tree-crossing 而非 density 場。**「per-net MST tree-edge 跨固定 region boundary 的 IO/feedthrough evaluator」目前查無先例,是本專案最乾淨的 novelty claim;GrandPlan 的存在同時證明此問題有工業價值、且時機緊迫(NVIDIA 已入場)。**

---

## 參考來源(依主題)

**DREAMPlace 系列**
- Lin et al., DAC 2019, DOI 10.1145/3316781.3317803 [全文 PDF];TCAD 2021 vol.40 pp.748–761, DOI 10.1109/TCAD.2020.3003843 [全文 PDF]
- Gu et al., "DREAMPlace 3.0", ICCAD 2020, DOI 10.1145/3400302.3415691 [全文 PDF]
- Liao et al., "DREAMPlace 4.0", DATE 2022 [全文 PDF];TCAD 2023 vol.42 pp.3374–3387, DOI 10.1109/TCAD.2023.3240132 [dblp]
- Repo: https://github.com/limbo018/DREAMPlace [README]
- GiFt: Y. Liu et al., ICCAD 2024;arXiv 2502.17632 [網頁/摘要]

**其他 placers**
- ePlace/ePlace-MS: J. Lu et al., TODAES 2015 / TCAD 2015 [未驗證-經典];Lu 博士論文(OpenAlex W2402497138)[摘要]
- RePlAce: Cheng et al., TCAD 2019, DOI 10.1109/TCAD.2018.2859220 [摘要]
- Xplace: Liu et al., DAC 2022, DOI 10.1145/3489517.3530485;TCAD 2024 vol.43 pp.1872–1885, DOI 10.1109/TCAD.2023.3346291 [dblp];repo https://github.com/cuhk-eda/Xplace [README]
- DG-RePlAce: Kahng & Wang, arXiv 2404.13049 / TCAD 2024(IEEE 10620224)[全文];OpenROAD BOF 2024 slides(Zhiang Wang)[網頁];repo https://github.com/ABKGroup/DG-RePlAce-AutoDMP
- AutoDMP: Agnesina et al., ISPD 2023, DOI 10.1145/3569052.3578923 [摘要];repo https://github.com/NVlabs/AutoDMP [README]
- MORPH: Mai et al., ICCAD 2024, DOI 10.1145/3676536.3676745 [摘要]
- GrandPlan: Xiong, Lu, Pan, Ren, ISPD 2026, DOI 10.1145/3764386.3779591 [摘要]
- ePlace-3D: Lu et al., ISPD 2016, DOI 10.1145/2872334.2872361 [摘要]

**Evaluators / RSMT / Routers**
- FLUTE: Chu & Wong, TCAD 27(1) 2008, DOI 10.1109/TCAD.2007.907068 [摘要]
- GeoSteiner: Juhl et al., Math. Prog. Comp. 10(4) 2018 [標題級]
- REST: J. Liu et al., DAC 2021, DOI 10.1109/DAC18074.2021.9586209 [摘要]
- NN-Steiner: Kahng et al., AAAI 2024 [摘要];GAT-Steiner: arXiv 2407.01440 [摘要]
- GPU EMST: Prokopenko et al., ICPP 2022, DOI 10.1145/3545008.3546185 [摘要]
- GAMER: Lin et al., ICCAD 2021;TCAD 42(2) 2023, DOI 10.1109/TCAD.2022.3184281 [網頁]
- FastGR: S. Liu et al., DATE 2022 [全文 PDF]
- InstantGR: S. Lin et al., ICCAD 2024, DOI 10.1145/3676536.3676787;repo https://github.com/cuhk-eda/InstantGR(BSD-3)[README]
- HeLEM-GR: Zhao et al., ICCAD 2024, DOI 10.1145/3676536.3676650 [全文 PDF]
- ISPD 2024 GR contest: DOI 10.1145/3626184.3639693;https://liangrj2014.github.io/ISPD24_contest/ [網頁]
- RUDY: Spindler & Johannes, DATE 2007, DOI 10.1109/DATE.2007.364463 [摘要]

**Feedthrough / 相鄰工作**
- FTAFP: Z. Li et al., ASP-DAC 2025, DOI 10.1145/3658617.3697728 [摘要]
- Flora: arXiv 2507.14914(2025)[摘要];Piano: arXiv 2508.13161(2025)[摘要]
- Two-Level RSMT: Held & Kämmerling, arXiv 1501.00933;Comput. Geom. 2017 [摘要]
