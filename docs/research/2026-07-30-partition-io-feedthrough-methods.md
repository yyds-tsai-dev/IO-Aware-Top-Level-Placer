# Partition / IO / Feed-through-Aware Placement 與 Floorplanning 方法調查

- 日期:2026-07-30
- 目的:為「IO-aware top-level placement」專案(flat GPU global placement,10M–30M cells,多 partition regions,objective = wirelength + IO number + pure feed-through number)整理可借用的既有方法。
- 閱讀深度標註:**[精讀全文]** = 已下載/讀完全文;**[僅摘要]** = 只讀到 abstract 與 metadata;**[未驗證]** = 無法確認的推斷,明確標出。
- 已下載 PDF 存放於 scratchpad(`.../scratchpad/papers/`),非 repo 內。

---

## 1. GrandPlan(本專案最核心參考)

**GrandPlan: Differentiable, Simultaneous Top-Level Floorplanning and Partition-Level Cell Placement for Large-Scale IP-Cores** **[僅摘要 + 引文清單;全文 PDF 未能取得]**

- 出處:ISPD 2026(Proceedings of the 2026 International Symposium on Physical Design),pp. 64–72。DOI: [10.1145/3764386.3779591](https://doi.org/10.1145/3764386.3779591)。DBLP: `conf/ispd/XiongLPR26`。發表日期 2026-03-15(ISPD 2026 於德國 Bonn,2026/3/15–18,共 14 篇 accepted papers)。
- 作者:Zhili Xiong(UT Austin, UTDA/Prof. David Z. Pan 學生)、Yi-Chen Lu(NVIDIA)、David Z. Pan(UT Austin)、Haoxing Ren(NVIDIA)。NVIDIA 主導的產業級工作(Yi-Chen Lu 個人頁面列 GrandPlan 為其 mentee Zhili Xiong 的 ISPD 2026 論文)。
- Open access:Unpaywall/OpenAlex 標 gold OA(CC-BY),但 PDF 僅掛在 ACM DL(dl.acm.org 有 Cloudflare 阻擋,本環境無法下載;無 arXiv 版)。**全文層級的公式細節在本報告中標為未驗證。**

### 1.1 問題定義(依 abstract)
- Top-level floorplanning:把 die 切成**恰好相鄰接(exactly abutted)的 regions**、精確配置各 region 面積,支撐 hierarchical place-and-route 與 PPA trade-off。
- 痛點:實務上 floorplan 仍靠人工 + RTL hierarchy,物理資訊有限;商用工具無法在 IP-core 規模做 flat optimization;後期 routability 驅動的 partition resize 會引發鄰接 partition 的連鎖 boundary 變更,拉長 turnaround。
- GrandPlan 的定位:**GPU 加速、可微分、end-to-end**,在單一自動化 loop 內同時決定 top-level floorplan(region 幾何)與 partition-level cell placement。

### 1.2 方法(依 abstract;細節公式未驗證)
三個緊耦合 stages:
1. **Flat IP-core placement with differentiable grouping objectives**:flat 放置 macro + standard cells,同時以「可微 grouping objective」把同一 partition/group 的 cells 拉聚。⇒ cell 的 partition 歸屬是靠 grouping 吸引力在連續空間形成的空間聚集,而不是離散指派變數。
2. **Boundary refinement via simulated annealing**:在 area 與 routability constraints 下,從 stage 1 的聚集結果萃取並精修出**乾淨的 rectilinear partition boundaries**(用 custom CUDA kernels)。其引文包含 morphological image analysis(Serra 等三本形態學文獻)與 quickhull/computational geometry,推測 boundary 萃取採用「density/佔據圖 → 形態學 open/close 清理 → rectilinear 化」的影像處理路線 **[由引文清單推斷,未驗證]**。
3. **Routability-aware fence-region placement**:以確定的 region 幾何為 fence regions 重新做 placement(引文含 DREAMPlace 3.0 multi-electrostatics region constraints,推測用 per-region density field 實作 fence **[推斷,未驗證]**)。

### 1.3 實驗規模與結果(依 abstract)
- 8 個大型工業 IP-cores,**最大 25M cells**。
- 相對 human-expert-crafted baselines:**total wirelength 最多 −14%;cross-partition (feedthrough) wirelength 平均 −27%**;平均 runtime **1.2 小時**。

### 1.4 Code
- 未見開源(工業資料集;搜尋 GitHub/NVIDIA research 頁面無 repo)。**[未驗證是否日後釋出]**

### 1.5 對本專案的意義與差異
- **直接支持我們的 Phase 2 方向**:證明「flat GPU placement + differentiable grouping + boundary 幾何化」在 25M-cell 規模可行、1.2 hr 級 runtime。
- 關鍵差異(= 我們的機會):
  1. GrandPlan 中 **partition region 幾何是輸出**(floorplanning 由工具決定);我們 Phase 1 是 **region 幾何給定、placement 決定 cell 歸屬**——問題相反方向,GrandPlan 的 stage 3(fence-region placement)反而更像我們的「已定幾何」情境,但它此時歸屬已定,不再優化 IO。
  2. 依 abstract,其目標與評估都是 **wirelength 類**(total WL、cross-partition WL);**沒有把 boundary-pin/IO「數量」與 pure feed-through「條數」做成可微 objective 的證據**(abstract 未提;全文未驗證)。我們的 IO-count / pure-feed-through-count objective 是空白區。
  3. 其 reference 清單(經 Semantic Scholar 驗證)以 analytical placement 系譜為主:ePlace(DAC 2014)、DREAMPlace(TCAD 2020)、DREAMPlace 3.0(ICCAD 2020,region constraints)、Xplace(TCAD 2024)、DG-RePlAce(TCAD 2024)、C3PO(ASP-DAC 2026)、RePlAce(TCAD 2019)、routability-driven mixed-size prototyping considering design hierarchy(DAC 2019)、MDP-trees(ASP-DAC 2019)、電荷法 fixed-outline floorplanning(ICCAD 2023)等——即 GrandPlan 是「electrostatics placement 家族 + 形態學 boundary 萃取」的組合,無 graph-partitioning/min-cut 元素。

---

## 2. Feed-through-aware floorplanning / placement(2012–2026)

> 名詞對齊:此領域的 "feedthrough" 指 net 穿越一個「它在其中沒有 pin 的 module/partition」,該 module 需為它開 feed-through port + buffer——與我們的 "pure feed-through" 定義一致。這批工作多為 module-level floorplanning(數十~數百 modules),用 SA + 樹狀拓撲表示;**沒有任何一篇在 million-cell flat placement 中做 feed-through 目標**。

### 2.1 FTAFP: A Feedthrough-Aware Floorplanner for Hierarchical Design of Large-Scale SoCs **[精讀全文]**
- 出處:ASP-DAC 2025, pp. 886–892。DOI: [10.1145/3658617.3697728](https://doi.org/10.1145/3658617.3697728)。作者:Zirui Li, Kanglin Tian, Jianwang Zhai(BUPT)、Shixiong Kai, Siyuan Xu(Huawei Noah's Ark)、Bei Yu(CUHK)、Kang Zhao(BUPT)。
- 做什麼:自稱**第一篇在 floorplanning 階段建模並優化 feedthrough** 的工作。fixed-outline floorplanning(SCB-Tree = CB-Tree + slack computation;two-phase SA)。
- **Feed-through cost model(對我們最有價值的部分)**:
  - 假設:直接相鄰(共邊)的同 net modules 不需 feedthrough;鄰接關係在 net 內可傳遞;隔著 whitespace 不算 feedthrough。
  - **Net simplification**:對每條 net,用 union-find 把「彼此相鄰」的 modules 合併成 sub-nets(連通分量),feedthrough 只發生在 sub-net 之間。
  - **Feedthrough wirelength**:對 sub-net 中心建 MST,邊權 `D_w(A,B) = wl_manh(A,B) × ω(A,B)`,`ω(A,B) = (n_blk(A)+n_blk(B))/2`;`fth_wl(A,B) = [wl_manh(A,B) − wl'(A) − wl'(B)] × ω(A,B)`,其中 `wl'(A) = (w(A)+h(A))/2 × sqrt(area'(A)/area(A))` 是 sub-net 尺寸修正。
  - **Feedthrough module 數 fth_num**:沿 MST 邊在 corner-stitching tile plane 上做貪婪 tile-walk,數走過的 solid tiles(Algorithm 2)。
  - **Common edge length CE_len**:相鄰同-net modules 的共邊長(決定 port 容量),`ce_len` = 區間交集長度。
  - **權重正規化**:推導均勻 √n×√n 網格下各 metric 的期望值(如 `E[HPWL] = 2m(n−1)/(3√n−2)×S`、`E[FTH_num] = 2(n−1)/(3√n−2) × Σ(n_i−1)/2`),以期望值當分母做 auto-normalization(Eq. 6–9),cost `φ = α·Area/E + β·HPWL/E − γ·CE/E + δ·(FTHnum/E + FTHwl/E)`。
- 規模與結果:GSRC/MCNC(10–200 modules、最多 1585 nets)。相對 CB-Tree:FTNUM −12%、FTWL −28%、CEL +25%、HPWL +4%(可接受的 trade-off);無 feedthrough 優化模式下 HPWL 較 Corblivar/SP-FOFP/CB-Tree −23%/−12%/−6%。C++ 單執行緒,RT 數秒~52 秒。
- Code:論文未提供 FTAFP 開源(baseline Corblivar 開源、SP-FOFP 有執行檔)。
- 可借用點:
  - 「**相鄰即免 feedthrough、跨 sub-net 才計數**」的 net simplification 概念,可移植成我們 evaluator 的精確計數規則(把 module 換成 partition region)。
  - 期望值正規化讓多目標權重免調參,對我們 Phase 1 的 objective 權重初始化有直接參考價值。
  - 侷限:全離散(SA + tree),無法微分;規模僅百級 modules。

### 2.2 Flora: One Step Beyond: Feedthrough & Placement-Aware Rectilinear Floorplanner **[精讀全文]**
- 出處:arXiv:[2507.14914](https://arxiv.org/abs/2507.14914)(2025-07;DBLP 列 CoRR。**發表 venue 未驗證**)。作者:Zhexuan Xu, Jie Wang, Zijie Geng, Feng Wu(USTC)、Siyuan Xu, Mingxuan Yuan(Huawei Noah's Ark)。
- 做什麼:三階段 feedthrough & placement-aware rectilinear floorplanner:(1) wiremask/position-mask 的 SA 粗優化 HPWL+feedthrough;(2) fixed-outline 下局部 resize 模組達 **zero-whitespace**(先矩形擴張、再 rectilinear whitespace 分配,分配準則 = 最小化 ΔFTpin);(3) tree search 快速擺 module 內 components(macro+stdcell clusters),再回頭微調 module boundary(cross-stage)。
- **Feed-through cost model**:
  - `FTmod(N_i) = 1/2 · Σ_{M_j ∈ B_i} 1(M_j is a feedthrough module)`,B_i = net bounding box(與 HPWL 對齊的近似:數 bbox 內被穿越的 modules)。
  - `FTpin(M_i,M_j) = max(0, ceil((u·Y_ij − CE_ij)/u))`:u = min pin spacing、Y_ij = 兩模組間需要的 pin 數、CE_ij = 共邊長 ⇒ **共邊容納不下的 pin 數**。這是「boundary pin 容量」的顯式模型(對我們 Phase 3 IO legalization 很有參考性)。
- 規模與結果:MCNC/GSRC(10–300 modules)。vs Corblivar/TOFU/Wiremask-EA:HPWL −5.93%、FTpin −5.16%、FTmod −29.15%、PD(可放置率)+14.24%、zero whitespace;RT 22–442s。也可當 post-processor(對 Corblivar 結果:FTmod 大幅下降)。
- Code:未見開源連結。
- 可借用點:zero-whitespace/exactly-abutted region 的產生流程(與 GrandPlan stage 2 目標相同但方法離散);FTpin 容量式模型可直接當我們 boundary-pin legalization 的容量約束;「把 whitespace 分給能減少 FTpin 的鄰居」= 邊界微調的貪婪準則。

### 2.3 Piano: A Multi-Constraint Pin Assignment-Aware Floorplanner **[精讀全文]**
- 出處:arXiv:[2508.13161](https://arxiv.org/abs/2508.13161)(2025;USTC + Huawei,同 Flora 團隊。**發表 venue 未驗證**)。
- 做什麼:floorplanning 與 **pin assignment 同時優化**(multi-constraint:fixed-outline、zero-whitespace、pre-placed modules)。
- **方法核心(對我們的 IO/feed-through evaluator 最具參考價值)**:
  - `P2PRes(M_i,M_j) = floor(AvailEdge_ij / u)`:相鄰邊上還能放幾根 pin(pin spacing u);指派一根 pin 就把該邊資源 −1。
  - 建 **資源圖 G_Res**(node=module,edge weight=P2PRes),每條 net 在其 mask 圖上跑 **A\***(邊權 = 模組中心歐氏距離)找 feedthrough 路徑;路徑長 >2 個 modules ⇒ 中間者為 feedthrough modules(`ftnum = n−2`,`ftlen = Σ 相鄰中心距`);找不到路徑 ⇒ **Unplacepin**(不可佈 pin)計數。實體 pin 位置用 beam search 選最短實體路徑。
  - Net 先分解:multi-pin net 依 driver/sink 拆成 2-pin nets(1 input pin 假設)。
  - 第三階段 SA 用三個 operators(random exchange / adjacent exchange with boundary queue / P2PRes enhancement——在共邊上開槽增加邊長)。SA 權重 HPWL:FTlen:FTnum:Unplacepin = 1:50:2000:100。
- 規模與結果:MCNC/GSRC(10–300 modules):HPWL −6.81%、FTlen −13.39%、FTnum −16.36%、Unplacepin −21.21%,zero whitespace;RT 1.5–42s。
- Code:未見開源。
- 可借用點:
  - **「feed-through 路徑 = 資源圖上的最短路」**:我們 Phase 2 的 router-based evaluator 可以直接採 A\*/Dijkstra on partition-adjacency graph(edge 容量 = 共邊剩餘 pin 資源),同時得到 IO 數、feed-through 穿越的 partitions、與 unplaceable 判定。
  - P2PRes/Unplacepin 給了 Phase 3 boundary-pin legalization 一個現成的可行性判準。

### 2.4 其他 feed-through / 跨模組連線相關工作(定位式條目)
- **Die Area Reduction by Decongesting Top Channels Using Novel Feedthrough Insertion Methodology in Hierarchical SoC Designs**,ISQED 2025, pp. 1–6,DOI: [10.1109/ISQED65160.2025.11014329](https://doi.org/10.1109/ISQED65160.2025.11014329)(Sakariya, Aich, Joshi, Griesmer)。**[僅 metadata]** 工業(標題顯示為 hierarchical SoC 後段流程的 feedthrough insertion 方法學,以 feedthrough 疏通 top-level channel 換 die area)。作為「feedthrough 是實務痛點」的佐證;方法細節未取得。
- **Channel based SoC feedthrough insertion methodology**,ICCCAS 2022, pp. 125–130(Y. Hong, C. Huang, Y. Gao, C. Li)。**[經 FTAFP 引文確認存在,未讀]** channel-based 的 feedthrough 插入方法學。
- **BOB-RSMT: Reclaiming over-the-IP-block routing resources with buffering-aware rectilinear Steiner minimum tree construction**,ICCAD 2012, pp. 137–143(Y. Zhang, A. Chakraborty, S. Chowdhury, D. Z. Pan)。**[經 FTAFP 引文確認,未讀]** 在 pre-routing/global routing 階段最小化 over-block 連線+buffer 可行性——feed-through 的「後段補救」路線;FTAFP 用它論證「該在 floorplan 期就處理」。
- **TOFU: A Two-Step Floorplan Refinement Framework for Whitespace Reduction**,DATE 2023(S. Kai et al., Huawei)。**[經 Flora/Piano 引文確認,未讀]** 把既有 floorplan 精修成近 zero-whitespace(rectilinear 化)。與「exactly-abutted regions」的幾何生成相關。
- **JigsawPlanner: Jigsaw-like floorplanner for eliminating whitespace and overlap among complex rectilinear modules**,ICCAD 2024, pp. 1–9(X. Du et al.)。**[經 Piano 引文確認,未讀]** zero-whitespace rectilinear floorplan 生成;Piano 批評其邊界過於不規則。
- **Challenges in Floorplanning and Macro Placement for Modern SoCs**(invited),ISPD 2024, pp. 71–72,DOI: [10.1145/3626184.3639695](https://doi.org/10.1145/3626184.3639695)(I-L. Tseng)。**[僅 metadata]** 產業視角問題陳述:floorplan 迭代中 feedthrough nets/ports 會被加入/移除 partitions,迫使重新評估物理可行性——正是我們專案的動機文獻。
- 註:DBLP 上 2018–2026 以 "feedthrough" 為題的 EDA 論文極少(多數 "feedthrough" 命中為 RF/類比/控制領域);EDA 圈系統性處理此問題自 FTAFP(2025)才開始,**這個賽道非常新**。

---

## 3. 「跨界數」作為 placement objective 的同構問題

### 3.1 3D-IC analytical placement 的 inter-tier via / cut 最小化

> 文獻分兩派:**pseudo-3D**(2D placement + 離散 tier partitioning:Compact-2D / Pin-3D / Snap-3D)與 **true-3D analytical**(z 座標連續化 + 3D density:ePlace-3D、CUHK/PKU bistratal 系列)。對「把 partition assignment 做成可微目標」最相關的是後者。

#### 3.1.1 ePlace-3D: Electrostatics based Placement for 3D-ICs【精讀全文】
- ISPD 2016(arXiv:1512.08291,DOI 10.1145/2872334.2872361)。Lu, Zhuang, Kang, Chen, Cheng(UCSD)。
- 目標:`min HPWL(v) + βz·#VI, s.t. density`。**#VI = net 穿透 tier 邊界的次數**(tier1→tier3 算 2)——正是 1-D 版的 feed-through 計數。
- 連續化:z 為連續變數;x/y/z 全用 weighted-average(WA)平滑;#VI 以 z 向 WA net span `βz·We_z` 隱式懲罰,βz 依電容比設定(`βz = #tiers×C_VI/(#rows×C_ROW)`,如 45nm TSV 30fF vs row wire 0.3fF)。
- 密度 = **eDensity-3D**:cell 視為 3D 電荷,解 3D Poisson(Neumann 邊界)+ 3D FFT,O(n log n);tier 離散化不靠 constraint,靠密度均勻化自然把 cell 推向各 tier。
- 結果:IBM-PLACE 12K–210K cells,vs mPL6-3D / NTUplace3-3D:WL −6.44%/−37.15%、VI −9.11%/−10.27%;MMS 最大 2.5M objects。無開源(核心後由 DREAMPlace 家族重實作)。
- 借用點:「穿透次數 = feed-through 計數」定義 + z-span 平滑 proxy;3D Poisson 密度是同時連續化 capacity 與 assignment 的標準骨架。

#### 3.1.2 CUHK/PKU Bistratal 3D placement(★對 Phase 1 最核心)【精讀全文】
- **Analytical Die-to-Die 3D Placement with Bistratal Wirelength Model and GPU Acceleration**,TCAD 2024(arXiv:2310.07424)。Peiyu Liao, Yuxuan Zhao, Dawei Guo, Yibo Lin, Bei Yu(CUHK+PKU)。DREAMPlace C++/CUDA 全 GPU。
- **同時決定 die assignment 與座標**。方法核心:
  1. Partition 變數 δ∈{0,1}ⁿ;**net cut indicator `C_e(δ) = max_{i∈e} δ_i − min_{i∈e} δ_i`**(= 該 net 是否需要一個 hybrid-bonding terminal / vertical IO)。
  2. **連續 z + 每疊代 rounding**:`δ_i = round(2z_i/z_max − 1/2)`;cell 尺寸/pin offset 依 δ 硬切換兩套製程屬性。
  3. **Bistratal wirelength** `W_e,Bi = max{p_e(x), p_e⁺(x)+p_e⁻(x)} + max{p_e(y), p_e⁺(y)+p_e⁻(y)}`(split net 放最佳 HBT 位置後的精確 D2D 線長閉式解;傳統 3D HPWL 只是其下界)。
  4. IO(HBT)數懲罰:總目標 `W = Σ_e W_e,Bi + α·Σ_e p_e(z)`——cut 數用 z 向 net span(WA 平滑)近似。
  5. **z 梯度不存在(W 對 z 不連續)→ 改用 finite difference:`∇_z W_e,Bi := (4/z_max)·[W_e,Bi(z=z_max/2) − W_e,Bi(z=0)]`,= 「把 cell 假想搬到另一 die,線長差多少」直接當梯度**——本質是把 FM gain 連續嵌入一階最佳化。
  6. 密度 = eDensity-3D;z 向密度力把 cell 推離中線、逼成合法二分。
- 結果:ICCAD 2022 contest(2.7K–221K cells):WL 比前三名平均 −4.1%/−5.7%/−7.2%,**HBT 數 −52.3%/−21.1%/−2.0%**,端到端最多 9.8× 加速(RTX 3090Ti)。
- 後續 **DAC 2024 mixed-size 版**(arXiv:2403.09070,Zhao 等):3D mixed-size preconditioner + MILP macro rotation;ICCAD 2023 contest(**官方 score = HPWL + 10·#HBT**,最大 740K cells)比冠軍 quality +5.9%、4.0× 加速。
- 借用點:`C_e = max δ − min δ` 就是我們 IO number 的 2-partition 特例;「rounding + 硬屬性切換 + **FD 假想搬移梯度**」是可直接推廣到 cell→K-partition 指派的完整配方(對 pure feed-through 也可定義 FD:假想搬到鄰 partition 算差分);「WL + α·(cut proxy)」與我們 objective 結構同構。

#### 3.1.3 其他 true-3D 連續化路線【摘要級/二手轉述】
- **NTUplace3-3D / TSV-aware analytical placement**(Hsu, Balabanov, Chang,DAC 2011 / TCAD 2013):WA wirelength 始祖;z 維用 **bell-shaped 函數**做 per-tier local smoothing;**mPL6-3D**(Luo, Shi, Cong,TCAD 2013)用 Helmholtz + **Huber** 平滑——「多井位能」路線雛形,缺點是僅局部平滑、收斂較差(ePlace-3D 之比較)。
- **NTU MTWA sigmoid 系列**(Chen 等,DAC 2023 LBR;DAC 2024,DOI 10.1145/3649329.3657370):**pin offset 等製程屬性寫成 z 的 sigmoid 連續過渡**(annealed sigmoid 內插)——partition 相依屬性可微化的最簡法;A7 指出其未準確反映 D2D 線長縮短效應。
- **ART-3D**(Murali 等,ISPD 2022):NL-3D 目標 `Σ_e (1+γ_e)(WL(e) + α_MIV·MIV(e))` + RL(OpenTuner)自動調 11 個參數(含 **α_MIV∈[1,10]**)。啟示:**via/IO 權重是強影響且 design-dependent 的超參數,值得自動搜尋**(PDP 最多 −43%;調參本身 +12% 品質)。
- **iPL-3D**(Zhao 等,ICCAD 2023,DOI 10.1109/ICCAD57390.2023.10323811):把 partition+座標寫成 **bilevel programming 交替最佳化**,兩個 tier operator 同時優化 WL 與 #terminal;vs ICCAD'22 前三名 WL −4.33%/−4.42%/−5.88%。連續化失敗時的 fallback 架構。
- **DCO-3D**(DAC 2025)與 **DRG-3D**(ASP-DAC 2026,DOI 10.1109/ASP-DAC66049.2026.11420682,NVIDIA+GT):GPU 全可微多目標(congestion、WL、via cost、**F2F-via cost**);vs Pin-3D:overflow −8.37%、TNS −23.99%。**tier assignment 的具體鬆弛形式未能取得原文確認【未驗證】**——與我們方向最接近的最新工作,建議取 IEEE Xplore 原文。

#### 3.1.4 Pseudo-3D(離散指派當後處理)【精讀/部分精讀】
- **Compact-2D**(Ku, Chang, Lim,ISPD 2018;TCAD 2020 Best Paper):2 倍面積 2D 實作(單位 RC 縮 0.707)→ 座標收縮 → **bin-based placement-driven FM min-cut** 決定 tier。**cut 數全由離散 FM 決定;唯一旋鈕是 bin size**(小 bin → via 多;大 bin → cut 少但位移大;LDPC 最佳 40µm)。工業 28nm、77K–312K cells,功耗最多 −26.8%。
- **Pin-3D**(Pentapati 等,ICCAD 2020 / TCAD 2024):輸入已 tier-partitioned;「transparent cells + **pin projection**」讓商用工具在完整 3D context 下 die-by-die 精修——固定 assignment 下跨-partition 精修的工程手法(WL 最多 −9.0%、TNS −88%)。
- **Snap-3D**(Vanna-Iampikul 等,ISPD 2021 / TCAD 2023):cell 高度砍半、**placement row 奇偶交錯標記 tier**,讓一個 2D placer 一次佈兩層——「把離散指派編碼進 legal 幾何空間」而非目標項,完全繞過可微性。WL −5.4%、power −10.1%、TNS −92.3%。
- **TP-GNN**(Lu 等,DAC 2020;TCAD 2022):2D placement 後以無監督 GNN embedding + 分群取代 bin-FM 決定 tier——離散 refinement 的 learning 替代。
- **PPA-aware tier partitioning with ILP**(Jeong, Park, Kim,ASP-DAC 2025,DOI 10.1145/3658617.3697733):分群後 ILP 同時最佳化 **inter-tier cut、重疊、timing-critical path 上的 tier 轉換次數、面積平衡**(power −1.23%、TNS −24.08%)——cut+timing 線性整合的正式表述,可作 cluster 級精修參考。
- **Open3DBench**(arXiv:2503.12946,南大+Huawei):OpenROAD 開源 3D benchmark + DREAMPlace cross-die co-placement 基線(**開源**)——可作我們的公開評測平台。

### 3.2 Chiplet partitioning 最小化 inter-chiplet IO

- **Chiplet Actuary**(Feng & Ma,DAC 2022,DOI 10.1145/3489517.3530428)【精讀全文】:D2D 介面 + NRE 的量化成本模型(良率 `Y = (1+D·S/c)^{−c}`、封裝/KGD/interposer 成本分解;D2D 介面以 chiplet 面積 ~10% 入模)。**開源**:github.com/Yinxiao-Feng/DAC2022。→ 我們 IO 的**實體單價**(PHY 面積/功耗/bump)可由此換算成與 WL 可比的 λ_IO。
- **Chipletizer / Chipletizer 2.0**(Li 等,ASP-DAC 2024,DOI 10.1109/ASP-DAC58780.2024.10473888;TCAD 2026)【僅摘要】:統一 characterization graph + **SA** 搜尋跨產品線共用的 SoC 切割,最小化含 D2D/封裝的總成本;2.0 把切割、floorplan、D2D 規格一起決定。具體改善百分比未取得原文【未驗證】。
- **Floorplet**(Chen 等,TCAD 2024,arXiv:2308.01672)【大部分精讀】:切割(對 RTL 階層樹遞迴切,非 min-cut)+ floorplan(數學規劃,Gurobi:`min β1·wl + β2·warpage + β3·C_2.5D`)+ Gem5 模擬回饋(**以實測 traffic 頻率加權跨界通訊成本**)。8–30 顆 chiplets;inter-chiplet 通訊成本平均 −24.81%。→ 「net 權重 = traffic 頻率」的思想可引入我們的 IO 目標(非齊一計數)。
- **ChipletPart**(Graening, Gupta, Kahng, Pramanik, Wang,TODAES 2026,DOI 10.1145/3796532;arXiv:2507.19819)【大部分精讀】:**GA(genome = 各 chiplet 製程)× FM 式 cost-driven 多路切割 × SA floorplan 驗 IO-reach feasibility** 的閉環;明確論證 ILP 不適用(成本與 IO-feasibility 無法線性化)。vs min-cut partitioners(hMETIS、TritonPart):成本最多 −58%(幾何平均 −20%);vs Floorplet −47%;vs Chipletizer −48%。**開源**(含 cost model)。→ 兩個關鍵教訓:(1) **min-cut ≠ min-cost**——純砍 crossing 數會產生不可行/次優解,IO 目標應掛上實體代價與幾何可行性;(2) reach 約束 ↔ 我們 boundary-pin 的容量/擁塞約束。
- 其他 2026 新作【僅 DBLP 標題驗證】:3D chiplet partitioning + floorplanning with vertical bonding(DATE 2026);timing-aware circuit partitioning with feasibility constraints for multi-chiplet(GLSVLSI 2026);SoC partitioning + global chiplet placement co-optimization(ISQED 2026)。
- 規模註記:chiplet 系工作都在「數百 IP blocks / ≤數十 chiplets」的粒度,與我們 10M-cell flat 不同;可借的是成本模型與 feasibility 迴圈,不是演算法本體。

### 3.3 傳統 min-cut placement 簡史(重點:terminal propagation)

> 本節條目經 DBLP 核實書目(標註【書目】= 未讀原文)。

- **Breuer, "A Class of Min-Cut Placement Algorithms", DAC 1977**【書目】:開山之作,把 placement 形式化為「沿一序列 cut lines 遞迴分割、每步最小化跨越 cut 的 net 數」——以 cut 數作為線長代理。跨 cut 的 net 數 ≈ 我們的 IO number 在單一 cut 上的特例。
- **Dunlop & Kernighan, "A Procedure for Placement of Standard-Cell VLSI Circuits", IEEE TCAD 4(1), 1985**【書目】★:min-cut placement 定型之作(KL/FM 遞迴二分)。關鍵貢獻 **terminal propagation**:分割區域 R 時,連到 R 外部的 net 把外部端點**投影到 R 邊界最近點**,以 zero-area 固定 dummy terminal 加入子問題,讓 FM 把「外部拉力」計入;投影落在靠近中線的模糊帶則忽略(細節以原文為準)。**Propagated terminal 在概念上就是我們的 partition boundary pin**——古典法是「先固定跨界點、再切下一刀」的貪婪序列決策;我們則把跨界數寫進可微目標與 placement 同時鬆弛,可視為 terminal propagation 的 soft/global 版本。
- **Suaris & Kedem(quadrisection), IEEE TCAS 1988 / TCAD 1989**【書目;1988 篇未於 DBLP 驗證,1989 篇已驗證】:一次 2×2 四分,cost 計入四象限兩兩之間的 net 跨越數,terminal propagation 一般化到四象限——「>2-way 一次決策優於序列二分」的最早證據。
- **Capo:Caldwell, Kahng, Markov, "Can Recursive Bisection Alone Produce Routable Placements?", DAC 2000**【書目】:multilevel FM(MLPart)引擎的工業級 min-cut placer;系統化 whitespace 配置與精細 terminal propagation(區分對 cut 有/無資訊量的 nets)。開源(UMpack/Capo;連結存活未驗證)。
- **Fengshui:Yildiz & Madden, DAC 2001;Agnihotri et al., "Fractional Cut", ICCAD 2003**【書目】:動態 cut 順序 + **fractional cut**(cut line 不對齊 row、延後 legalization)大幅改善線長。教訓:「離散幾何約束越晚硬化越好」——支持我們把 partition 邊界處理成 soft(電位/密度)直到後期才 snap。
- **總結**:古典鏈條共同確立「跨 partition 的 net 必須以邊界 fixed pin 的形式出現在子問題中,否則各區域獨立最佳化必然錯」。我們反其道:boundary pin 不再是預先固定的輸入,而是可微 loss 的輸出;初始化時(先用 partitioner 切好)仍可用 Dunlop–Kernighan 式投影生成第一版 boundary-pin anchors。

---

## 4. 可微 / ML graph partitioning 與 hypergraph partitioners

### 4.1 GAP: Generalizable Approximate Graph Partitioning Framework【精讀全文】★
- 出處:**arXiv:1903.00614(2019-03,CoRR)**。經 DBLP 與 Semantic Scholar 雙重查證**僅為 arXiv preprint,未正式發表**(「ICLR workshop」之說未獲資料庫證實)。作者:Azade Nazi, Will Hang, Anna Goldie, Sujith Ravi, Azalia Mirhoseini(Google Research + Stanford)。
- 做什麼:第一批把 balanced min-cut 寫成連續可微 loss、以 GNN + backprop 無監督訓練,並展示跨圖泛化。
- **方法核心(公式取自原文)**:soft assignment `Y ∈ R^{n×g}`(GCN/GraphSAGE + MLP + softmax),`Y_ik = P(v_i ∈ S_k)`:
  - 期望 cut:`E[cut(S_k, S̄_k)] = Σ_{v_i∈S_k} Σ_{v_j∈N(v_i)} Σ_z Y_iz(1−Y_jz)`;矩陣式 `reduce-sum(Y_{:,k}(1−Y_{:,k})ᵀ ⊙ A)`。
  - 期望 volume:`Γ = YᵀD`;**期望 normalized cut** `E[Ncut] = reduce-sum((Y⊘Γ)(1−Y)ᵀ ⊙ A)`。
  - Balance 項:`reduce-sum((1ᵀY − n/g)²)`。總 loss = 兩者相加。
  - 單一 edge 之期望 cut `Σ_k Y_ik(1−Y_jk) = 1 − ⟨Y_i,Y_j⟩`,即未正規化版恰為 `Σ_{(i,j)∈E}(1−Y_i·Y_j)`。
- 規模與結果:最大 27,114 nodes(Inception-v3 computation graph)+ 1k–10k 隨機圖;vs hMETIS cut 相當或更好、balancedness 99%、inference 快 10–100×。**離我們 10M–30M 差 3 個數量級**。
- Code:官方從未釋出(僅第三方重製,品質未驗證)。
- 可借用點:partition-assignment 的模板——每 cell 一個 g 維 softmax;balance 項換成 area 加權 `(areaᵀY − cap)²`;reduce-sum 矩陣寫法可直接映射 GPU sparse ops,與 DREAMPlace CUDA kernel 相容。

### 4.2 MinCutPool / DMoN(可微分群的兩個標準替代品)【節選】
- **MinCutPool**:Bianchi, Grattarola, Alippi, "Spectral Clustering with Graph Neural Networks for Graph Pooling", **ICML 2020**。Cut loss `L_c = −Tr(SᵀÃS)/Tr(SᵀD̃S)`(trace-ratio 形式 normalized cut)+ **orthogonality loss** `L_o = ‖SᵀS/‖SᵀS‖_F − I_K/√K‖_F`(平衡 + 逼近 one-hot 合一,比 GAP 二次項更不易塌縮)。Code:PyTorch Geometric `dense_mincut_pool` / Spektral 內建。
- **DMoN**:Tsitsulin, Palowitch, Perozzi, Müller, "Graph Clustering with Graph Neural Networks", **JMLR 24 (2023)**(arXiv:2006.16904)。Modularity 目標 `L = −(1/2m)Tr(CᵀBC) + (√k/n)‖Σ_i Cᵢᵀ‖_F − 1`(第二項 = **collapse regularization**);計算分解為 sparse SpMM + rank-1 修正 ⇒ **與邊數線性**——此分解技巧對 10M+ cells 關鍵。Code:google-research repo(`graph_embedding/dmon`)。

### 4.3 MedPart: A Multi-Level Evolutionary Differentiable Hypergraph Partitioner【僅摘要】★
- 出處:**ISPD 2024, pp. 3–11**,DOI: 10.1145/3626184.3633319。作者:Rongjian Liang, Anthony Agnesina, **Haoxing Ren(NVIDIA;與 GrandPlan 同資深作者)**。
- 做什麼:multilevel 架構 + fast spectral coarsening,每個 coarsening level 跑 evolutionary **differentiable** 最佳化,借 deep graph learning toolkit 在 GPU 上加速 hypergraph partitioning。
- 結果(摘要宣稱):穩定勝過 hMETIS;部分 benchmark cut 比含 SpecPart 的已發表最佳解**再好至多 30%**;runtime 與 hyperedge 數線性。Code 未公開。**全文在 ACM 付費牆後,可微 loss 精確式未驗證——必讀原文。**
- 意義:直接證明「GPU + differentiable hypergraph cut + multilevel」能打贏 hMETIS/SpecPart;NVIDIA 同團隊脈絡(MedPart → GrandPlan)顯示他們手上已有全部零件,但**尚未把可微 hypergraph cut 接進 placement objective**——我們的空間。

### 4.4 其他(定位)
- **NeuroCUT**(KDD 2024,arXiv:2310.11787):GNN + RL 逐節點決策,**不要求目標可微**、k 可變——若 pure feed-through 最終難寫成好的可微式,RL 是逃生門(但 10M 節點樣本效率存疑)。
- **Erdős Goes Neural**(Karalias & Loukas, NeurIPS 2020,arXiv:2006.10643):期望值框架的理論根據——網路輸出機率分布、對目標取期望當 loss、用 method of conditional expectation 取整並保證解不劣於期望值。**這正是我們 soft assignment → 離散 partition 的 decode 配方。** 後續 Wang & Li(NeurIPS 2022,arXiv:2207.05984,未逐字驗證)證明 entry-wise concave 鬆弛才有好取整性質。
- 命名陷阱:"Deep Multilevel Graph Partitioning"(ESA 2021)的 "deep" 指深層 multilevel 遞迴,**非神經網路方法**,勿誤引。
- **SAAP**(arXiv:2604.16357,2026):對「有 placement 座標、有空間約束」的 netlist 做 analytic k-way partitioning——與我們「partition 有幾何」設定精神相近,值得追蹤。【僅摘要】

### 4.5 Hypergraph「期望被切數 / λ−1」的可微寫法(對應我們的 IO number)
> 調查結論:除 MedPart(公式未能核實)外,**未找到專門給出 hypergraph 版可微 cut loss 的標準論文**;但在獨立性假設下推導是初等的,且與 Erdős-goes-neural 期望值框架一致。設 row-stochastic `Y ∈ [0,1]^{n×g}`,net e 的 pin 集合 `P_e`:

- `P[e 完整落在 part k] = Π_{i∈P_e} Y_ik`
- **期望 cut-net 數**:`E[cut] = Σ_e w_e (1 − Σ_k Π_{i∈P_e} Y_ik)`(各 k 事件互斥可直接相加)。
- **期望 connectivity**:`E[λ_e] = Σ_k (1 − Π_{i∈P_e}(1 − Y_ik))` ⇒ **connectivity−1(km1)可微版** `E[λ_e − 1]`。
- **對應 IO number**:跨 λ 個 partition 的 net 以 spanning-tree 佈線約需 λ−1 條跨界連線(每條兩端各一顆 boundary pin),故 `Σ_e w_e·E[λ_e−1]` 是 IO number 的自然可微代理;若每個被碰到的 partition 各開一 port 則用 `E[λ_e]`。
- **實作注意**(工程經驗,非文獻主張):大 net 連乘用 `exp(Σ log(·))` + clamp 防 underflow;大 net 梯度被稀釋,建議 1/(|e|−1) 加權或截斷超大 net;GAP 的 pairwise 形式等價 clique expansion,會**高估**大 net 的 cut——hypergraph 語意下 product 形式才正確。

### 4.6 平行 / 現代 hypergraph partitioners(初始化與 baseline)

| 工具 | 定位 | 引用(經 DBLP/官方 repo 核實) | 約束 | Code |
|---|---|---|---|---|
| hMETIS | multilevel 經典、20 年 de-facto baseline;serial、closed binary | Karypis et al., DAC 1997 + IEEE TVLSI 1999;k-way: VLSI Design 2000 | fixed vertices、UBfactor | glaros.dtc.umn.edu(binary;連結未驗證) |
| PaToH | 科學計算出身,快、品質略遜 KaHyPar | Çatalyürek & Aykanat, IEEE TPDS 1999 | multi-constraint、fixed vertices(依 manual,未逐字驗證) | gatech 網頁 binary(未驗證) |
| KaHyPar | n-level 高品質 + flow-based refinement;serial 品質標竿 | Schlag et al., ACM JEA 2022 | fixed vertices、individual block weights、cut/**km1** | github.com/kahypar/kahypar ✓ |
| **Mt-KaHyPar** | shared-memory 平行版,品質 ≈ serial、快一個量級;**我們規模初始化首選** | Gottesbüren et al., ACM Trans. Algorithms 2024(arXiv:2303.17679) | fixed vertices、cut/km1、large-k | github.com/kahypar/mt-kahypar ✓(宣稱可處理數十億 pins 級,數字未逐字驗證) |
| BiPart | 平行且 deterministic;速度/確定性換品質 | Maleki et al., PPoPP 2021 | balance;fixed vertices 未確認 | Galois repo ✓ |
| SpecPart / K-SpecPart | supervised spectral **refinement**:拿既有解當 hint,generalized eigenproblem + cut-overlay + ILP;K 版 multi-way 比 hMETIS/KaHyPar 好至多 20%(摘要宣稱) | ICCAD 2022;IEEE TCAD 2024(arXiv:2305.06167) | 2-way / k-way、balance | github.com/TILOS-AI-Institute/HypergraphPartitioning ✓(含 benchmark + 最佳解 leaderboard) |
| **TritonPart**(OpenROAD par) | constraints-driven,**功能面最貼近我們**:multi-dim weights、多重 balance、fixed vertices、group constraints、**embedding-aware(可吃 placement 資訊)**、timing-driven | Bustany et al., ICCAD 2023(官方 README 已核實功能) | 如左 | OpenROAD repo(src/par)✓ |

另:Schlag, Heuer, Schulz, "Invited: Modern Hypergraph Partitioning: KaHyPar, Mt-KaHyPar, and Beyond", **ISPD 2026**(DBLP 已驗證)——最新官方 survey。
**選型建議**:初始 assignment 用 Mt-KaHyPar(規模)或 TritonPart(fixed vertices + 多維 balance + embedding-aware,語意最合);品質上限用 KaHyPar + (K-)SpecPart;TILOS leaderboard 當 cut 品質對照組。

---

## 5. 綜合分析

### 5.1 現有方法的 gap(= 我們的機會)
1. **規模 gap**:feed-through-aware 工作(FTAFP/Flora/Piano)全部是 10–300 modules 的 SA/離散方法;GrandPlan 到 25M cells 但(依 abstract)只優化 wirelength 類指標。「**在 million-cell flat placement 中把 IO 數與 pure feed-through 數當一級可微目標**」目前沒有人做——這正是本專案定位。
2. **目標 gap**:GrandPlan 的 differentiable grouping 讓 cells 聚集,但聚集 ≠ 最小割:相同 grouping 能量的兩個解可以有不同 IO 數。cut-count 需要 per-net 的離散指示函數;可微形式存在於 3D-IC(`C_e = max δ − min δ` + z-span 平滑、FD 梯度,§3.1)與 GAP/hypergraph 期望值(`E[λ_e−1]`,§4.5),**但兩邊都沒被接到 2D 多 partition 的 top-level placement 上**。特別是:
   - 3D 工作只有 2 層(至多數層、1-D 排列),partition 拓撲是「線」;我們是任意 2D 相鄰圖。
   - GAP/MedPart 是純 partitioning,無座標、無幾何;我們的 assignment 與 (x,y) 耦合。
3. **pure feed-through gap**:兩個子調查一致確認——**「pure feed-through number」的可微建模找不到任何直接文獻**。最接近的是 (i) ePlace-3D 的 #VI「穿透次數」(1-D 特例)、(ii) Flora 的 FTmod(bbox 覆蓋計數,離散)、(iii) Piano 的 A\* 路徑(evaluator,離散)。此為明確 novelty 空間。
4. **幾何方向 gap**:GrandPlan 幾何是輸出;我們 Phase 1 幾何是輸入(region 給定、歸屬待定)。給定幾何下的「cell↔partition 歸屬 + IO 最小化」= fence-region assignment(DREAMPlace 3.0)+ min-cut(GAP/MedPart)的混合問題,現有文獻兩邊各做一半。
5. **min-cut ≠ min-cost 的教訓**(ChipletPart, TODAES 2026):純砍 crossing 數會產出幾何不可行或成本次優的解——IO 目標應綁定實體單價(D2D/PHY 面積、buffer/port 成本;可用 Chiplet Actuary 模型換算)與 boundary 容量可行性(Flora FTpin / Piano P2PRes)。

### 5.2 Phase 1 objective 的可微 surrogate 候選清單(附出處依據)
> 設 cell i 位置 (x_i,y_i);regions R_1..R_K 幾何給定。兩條互補路線:**(A) 位置驅動**——歸屬由座標誘導,梯度回流到 (x,y);**(B) 顯式機率**——每 cell 一個 K 維 softmax Y_i,與座標聯合優化。

- **(S1) 幾何軟歸屬(路線 A 的基礎)**:`p_{i,k} = softmax_k(−d_k(x_i,y_i)/τ)`,d_k = 到 region k 的 signed distance(rectilinear region 用每邊 max/min 組合),τ 退火 → one-hot。依據:GrandPlan 以「連續位置 + differentiable grouping」隱式決定歸屬(ISPD 2026);DREAMPlace 3.0 multi-electrostatics 證明 per-region density 場可微可行(ICCAD 2020);3D 系列(§3.1)證明「位置軸 = 指派軸」在 2 層時成立,K 個 2D region 是其一般化。
- **(S2) IO number 主候選 — 期望 connectivity−1**:net-level 聚合 `q_{e,k} = 1 − Π_{i∈e}(1 − p_{i,k})`(net e 碰到 region k 的機率),`E[λ_e] = Σ_k q_{e,k}`,**IO loss = Σ_e w_e·(E[λ_e] − 1)**。依據:hypergraph km1 期望值推導(§4.5,獨立性假設下與 Erdős-goes-neural 期望值框架一致,NeurIPS 2020);GAP 的 expected normalized cut 是其 graph 特例(arXiv:1903.00614 Eq.4–9);ePlace-3D/CUHK 的 z-span 是其 1-D 幾何版。實作:`exp(Σ log)` 防 underflow、大 net 1/(|e|−1) 加權(§4.5 注意事項)。
- **(S3) IO number 替代 — 平滑 span/幾何 proxy**:沿用 3D 慣例,把跨界懲罰寫成幾何量:對每對相鄰 region 的 boundary,以 WA/LSE 平滑的「net 在 boundary 兩側都有質量」指標,或直接 `α·(net 的 cross-boundary WA span)`。依據:ePlace-3D 的 `βz·We_z`(ISPD 2016)、CUHK 的 `α·Σ p_e,WA(z)`(TCAD 2024)。優點:與 WL 梯度同構、幾乎免費;缺點:是 WL 的重加權,對「數量」的控制間接。
- **(S4) pure feed-through 主候選 — 覆蓋×無 pin**:`FT(e) = Σ_k coverage_k(e)·(1 − q_{e,k})`,coverage_k = net 的 route 估計(bbox 或 Steiner/FLUTE 骨架)與 region k 交疊的平滑指標。依據:Flora FTmod(bbox 內被穿越 module 計數,arXiv:2507.14914 Eq.1)的連續化;ePlace-3D #VI「穿透」語意的 2D 推廣。進階:在 partition-adjacency graph 上定義 net 的軟路徑(soft shortest path / flow),feed-through = 路徑經過且 q≈0 的 region——精確版交給 Phase 2 router-based evaluator(Piano A\* 資源圖,arXiv:2508.13161)。
- **(S5) FD(finite-difference)假想搬移梯度(離散量的萬用梯度)**:對每 cell 平行計算「搬到鄰 region 後 WL/IO/FT 的差分」當作該 cell 對 assignment 軸的梯度。依據:CUHK TCAD 2024 式(33)(z 梯度不存在 → Nörlund 差商,= FM gain 嵌入一階法);GPU 上與 net bounding 資訊共用資料結構。**當 S2/S4 的乘積式在超大 net 上梯度稀釋時,FD 是最穩的 fallback。**
- **(S6) 防塌/平衡正則化**:GAP 二次 balance 項的面積加權版 `Σ_k (area^T Y_{:,k} − cap_k)²`;或 MinCutPool orthogonality `‖SᵀS/‖SᵀS‖_F − I/√K‖_F`(平衡+one-hot 合一,ICML 2020);或 DMoN collapse regularization(JMLR 2023)。註:我們 region 幾何給定時,per-region 靜電密度場(DREAMPlace 3.0)本身就是最強的 capacity 約束,S6 只需管 softmax 路線的塌縮。
- **(S7) boundary-pin 容量懲罰(Phase 3 前置)**:Flora `FTpin = max(0, ceil((u·Y−CE)/u))` 鬆弛為 `softplus((u·Y_est − CE)/u)`,Y_est = 由 q_{e,k} 累計的相鄰 region 對需求 pin 數,CE = 共享邊長。依據:Flora(arXiv:2507.14914 Eq.2)、Piano P2PRes(arXiv:2508.13161)。避免把 IO 全擠向短邊界,並為 IO legalization 留可行域。
- **(S8) decode 與校正迴圈**:soft → 離散用 method of conditional expectation 逐 cell 取整(保證不劣於期望值;Erdős Goes Neural, NeurIPS 2020);離散後用精確 evaluator(FTAFP net-simplification+MST+tile-walk;Piano A\*/P2PRes/Unplacepin)量測真 IO/FT,再以 FTAFP 期望值 auto-normalization(ASP-DAC 2025 Eq.6–9)+ ART-3D 式自動調參(RL/BO,ISPD 2022)更新 λ 權重。

### 5.3 Pipeline 建議(文獻依據對應)
1. **初始 assignment**:Mt-KaHyPar(規模;ACM TALG 2024)或 TritonPart(fixed vertices、多維 balance、embedding-aware;ICCAD 2023),得到高品質離散初始解;必要時 K-SpecPart refinement(TCAD 2024)。
2. **Phase 1 global placement**:DREAMPlace 3.0 式 multi-electrostatics(每 region 一個密度場)+ S2/S4 可微 IO/FT loss + S6 正則;初始 boundary-pin anchors 可用 Dunlop–Kernighan terminal propagation 式投影(TCAD 1985)。
3. **精修**:S5 FD 梯度或 bin-level FM(Compact-2D 式)/ILP(ASP-DAC 2025 式)做離散 refinement;fractional-cut 教訓(ICCAD 2003)= 幾何約束越晚硬化越好。
4. **評測**:ICCAD 2022/2023 3D contest(score = HPWL + 10·#HBT)證明「WL + β·IO」是社群接受的官方目標形式;Open3DBench 與 TILOS HypergraphPartitioning repo 提供公開對照。
5. **必讀待補**:GrandPlan 全文(ACM DL)、MedPart 全文(ISPD 2024)、DRG-3D 全文(ASP-DAC 2026)——三篇都與我們最近鄰,且全文皆未能免費取得,建議走學校授權下載後更新本報告。

---

## 附錄:本輪已驗證的支撐文獻(供 §1–2 引用)
- ePlace: Electrostatics Based Placement Using Nesterov's Method,DAC 2014(GrandPlan 引文)。
- DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement,TCAD 2020,DOI 10.1109/TCAD.2020.3003843。
- DREAMPlace 3.0: Multi-Electrostatics Based Robust VLSI Placement with Region Constraints,ICCAD 2020,DOI 10.1145/3400302.3415691(fence-region 的 multi-electrostatics 處理;我們 Phase 1 的基座技術)。
- Xplace(TCAD 2024,DOI 10.1109/TCAD.2023.3346291)、DG-RePlAce(TCAD 2024,DOI 10.1109/TCAD.2024.3436521)、RePlAce(TCAD 2019,DOI 10.1109/TCAD.2018.2859220)——GPU/analytical placement 系譜(GrandPlan 引文)。
- C3PO: Commercial-Quality Global Placement...,ASP-DAC 2026,DOI 10.1109/ASP-DAC66049.2026.11420751(NVIDIA;Best Paper Award,見 Yi-Chen Lu 首頁)。
- Handling Orientation and Aspect Ratio of Modules in Electrostatics-Based Large Scale Fixed-Outline Floorplanning,ICCAD 2023,DOI 10.1109/ICCAD57390.2023.10323841;PeF: Poisson's Equation Based Large-Scale Fixed-Outline Floorplanning,TCAD 2023(arXiv:2210.03293)——電荷法做 floorplan 的證據鏈。
- Routability-driven Mixed-size Placement Prototyping Considering Design Hierarchy and Indirect Connectivity Between Macros,DAC 2019,DOI 10.1145/3316781.3317901(hierarchy-aware flat prototyping;GrandPlan 引文)。
