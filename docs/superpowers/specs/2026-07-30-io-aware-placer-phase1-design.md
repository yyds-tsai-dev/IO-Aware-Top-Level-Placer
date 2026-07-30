# Phase 1 設計:IO-Aware Top-Level Placer(fixed-macro、region 給定)

- 日期:2026-07-30
- 狀態:草案,待審閱
- 文獻依據:`docs/research/2026-07-30-partition-io-feedthrough-methods.md`、`docs/research/2026-07-30-gpu-placers-and-evaluators.md`、`docs/research/2026-07-30-benchmarks-and-iccad24d.md`

## 1. 背景與目標

研究主題為 IO-aware top-level placement:超大規模設計(10M–30M standard cells)做 flat GPU global placement,die 上有 K 個 partition regions,目標是在維持線長品質的同時,最小化跨 region 的連線數(IO)與純穿越(pure feed-through)。三階段路線:

- **Phase 1(本 spec)**:macro 固定、region 幾何給定,objective = wirelength + IO + pure feed-through;建 fast evaluator;benchmark 擴增到 1M/5M/10M/30M。
- Phase 2:mix-sized(macro 可動)、region/partition 方式共同優化、router-based 精確 evaluator。
- Phase 3:IO(boundary pin)legalization。

定位聲明(相對最近鄰工作):與 GrandPlan(ISPD 2026, NVIDIA+UT Austin)的區隔 = region 幾何固定(輸入)、objective 是 boundary-pin/feed-through「計數」而非 cross-partition wirelength、per-net tree-crossing fast evaluator、30M 規模;與 FTAFP/Flora/Piano 的區隔 = flat cell-level placement 而非百級 module 的 floorplanning;與 3D-IC 系列的區隔 = 2D 任意 K-region 相鄰拓撲而非 2 層 tier。「可微 pure feed-through 建模」與「MST tree-edge crossing evaluator」兩者文獻調查均查無先例,為主要 novelty。

## 2. 關鍵定義

- **IO number**:per-net routing tree(Phase 1 用 MST/RSMT 近似)跨 partition region boundary 的 crossing 總數;一個 crossing 對應一對 boundary pins。報表同時附 cut-net 數與 connectivity−1(λ−1)作參考 metrics。
- **Pure feed-through**:tree edge 穿過某 region、且該 net 在該 region 內無任何 pin 的事件(per net、per 被穿越 region 累計)。
- **Design style**:channelless / fully-abutted(六路一手證據一致的業界主流;MediaTek ICCAD24-D spec、Intel FloorSet、GrandPlan、Hong 2022、FTAFP、Flora)。channel 語意留作 Phase 2 的 cost-model 開關。

## 3. 已確認決策(2026-07-30 brainstorming)

| # | 決策 | 內容 |
|---|---|---|
| D1 | 歸屬=輸出 | region 幾何給定且 placement 期間固定;cell 落點決定歸屬;IO/FT 因此是 placement 的可優化函數 |
| D2 | Partition 數 | K ≈ 4–32;O(N×K) 稠密結構在 30M cells 可行 |
| D3 | Region 形狀 | rectilinear 從 Phase 1 即支援,以「矩形集合聯集」表達;實作順序 grid → 單矩形 → rectilinear |
| D4 | IO 計數語意 | boundary crossing 數(見 §2) |
| D5 | Baselines | (a) two-stage:Mt-KaHyPar → fence-constrained DREAMPlace(multi-electrostatics);(b) IO-oblivious flat DREAMPlace + 事後計數;(c) GrandPlan 概念性對照(設定不同、無開源 code,以文獻數字定位) |
| D6 | 硬體 | 開發:單卡 NVIDIA L4(24GB);正式實驗:單卡 H100(申請制,確保可用)。記憶體推算 30M cells float32 約 15–17GB,L4 可做全規模功能驗證,H100 跑計時 |
| D7 | 產出 | 學位論文為主,會議投稿為加分項;里程碑制、不綁死單一 deadline |
| D8 | 技術路線 | 方案 A:可微 IO/FT objective + MST-crossing evaluator 閉環;milestone M1 = 純 evaluator + net-reweighting(方案 B 作為中繼產物) |

## 4. 系統架構

```
輸入: netlist(Bookshelf 或 LEF/DEF)、K 個 rectilinear regions、fixed macros
  │
  ▼
(0) 前處理: region rasterize → region-id grid(4096², ~34MB)+ boundary 段表
            Mt-KaHyPar 初始 assignment → warm-start placement(歸屬不固定)
  │
  ▼
(1) GP 主迴圈(fork DREAMPlace 4.x, float32):
      min  WA-WL + λ_D·density + λ_IO·L_IO + λ_FT·L_FT + λ_pin·L_pincap
      每 N=50–100 迭代呼叫 evaluator → 報表 + λ auto-normalization + per-net weight 校正
  │
  ▼
(2) Decode: soft → hard 歸屬(method of conditional expectation,逐 cell 取整)
  │
  ▼
(3) 離散 refinement: boundary 附近 cells 以 finite-difference 假想搬移 / bin-FM 精修
  │
  ▼
(4) 輸出: placement + assignment + per-net crossing/FT 報表(evaluator 終評)
```

要點:

- **單一全域 density 場即可**:region 幾何固定、歸屬由位置誘導時,region 面積即容量,全域 electrostatics 已隱式滿足 per-region capacity。DREAMPlace 3.0 multi-electrostatics 僅 baseline (a) 的 fence placement 使用。
- **Region 輸入格式**:LEF/DEF 流程用 DEF REGION/GROUP(ISPD 2015 慣例);Bookshelf 流程用 sidecar 檔(矩形清單 + 面積預算)。附 region 產生器:grid 切割與 random slicing-tree(面積比例可控)兩種模式,亦接受外部給定的 region 檔。
- **平台**:fork DREAMPlace 4.x(BSD-3;PyTorch autograd + custom CUDA op 架構,新 objective = 新增一個 op,不動核心迭代器)。Xplace kernel 技巧與其 GGR 作為備援參考;版本以容器鎖定。

## 5. Objective formulation

### 5.1 軟歸屬(S1)

`p_{i,k} = softmax_k(−d_k(x_i, y_i) / τ)`

- `d_k` = cell i 到 region k 的 signed distance(region = 矩形集合:取成員矩形 signed distance 的 min;region 內為負)。
- τ 退火:初期大(模糊、梯度可流動)→ 後期小(趨近 one-hot);schedule 於 M2 以實驗定案。

### 5.2 IO 項(S2,主候選)

- net e 觸及 region k 的機率:`q_{e,k} = 1 − Π_{i∈e}(1 − p_{i,k})`,實作用 `exp(Σ log(1−p+ε))` 防 underflow。
- `L_IO = Σ_e w_e · (Σ_k q_{e,k} − 1)`(期望 connectivity−1;跨 λ 區的 net 以 spanning tree 佈線需 λ−1 個 crossing,與 D4 一致)。
- 大 net 處理:w_e 含 1/(|e|−1) 稀釋補償;degree > D_max(預設 256)的 net 不入此項,改用 region-presence 精確下界(distinct region 數 − 1)入報表、其梯度略過(或走 S5)。

### 5.3 Pure feed-through 項(S4,主要 novelty)

- `L_FT = Σ_e Σ_k cov_k(e) · (1 − q_{e,k})`
- `cov_k(e)` = net e 的骨架與 region k 的重疊平滑指標。Phase 1 預設:`area(softbbox(e) ∩ R_k) / area(softbbox(e))`(soft bbox 由 WA min/max 得,天然可微);M3 ablation 比較「以 region 面積正規化」與「Steiner 骨架取代 bbox」兩個變體。

### 5.4 Boundary-pin 容量項(S7)

- 對每對相鄰 region (a,b),共享邊界長 ℓ_ab、容量 `C_ab = ρ·ℓ_ab`(ρ = 每單位長 pin 數,由製程/metal pitch 設定)。
- soft 需求 `D_ab = Σ_e w_e · q_{e,a} · q_{e,b}`(限相鄰對);`L_pincap = Σ_{(a,b)} softplus((D_ab − C_ab)/C_ab)`。
- 幾何上「crossing 落在哪段邊界」的精確歸屬由 evaluator 給出(per-boundary-pair demand),用以校正此 soft 近似。

### 5.5 權重調度與 fallback

- λ_IO、λ_FT 由小漸增(先讓 WL/density 穩定大局);每 N 迭代以 evaluator 精確計數做 auto-normalization(FTAFP 式期望值正規化)。
- 梯度品質 fallback:**S5 FD 假想搬移梯度**(平行計算「cell 搬到鄰 region 的成本差分」當梯度;CUHK bistratal TCAD 2024 的 K-region 推廣)與 **S3 span proxy**(跨界懲罰寫成 WA span 幾何量)。M2 若乘積式表現不佳即切換。

## 6. Fast evaluator(`mst_crossing_eval`)

- **建樹**(per-net 獨立、embarrassingly parallel):d=2 直連;d=3 closed-form;4≤d≤32 每 warp 一 net 的 O(d²) Manhattan Prim;32<d≤256 每 block 一 net;d>256 不建樹,用 region-presence 精確下界。總工作量 ~6×10⁸ 距離運算 @30M nets。
- **Crossing 計數**:每條 tree edge 以 L-shape 兩段沿 region-id grid walk;grid 型 partition 有 O(1) 特例(x-span 內垂直界線數 + y-span 內水平界線數,與 L 方向無關)。
- **FT 判定**:per-net pin region bitmask(K≤64 → uint64);edge walk 經過 region r 且 bitmask[r]=0 ⇒ pure FT +1。
- **輸出**:總 IO/FT、per-net crossing、per-boundary-pair pin demand(餵 §5.4 校正與 Phase 3)、樹長(WL 估計)。
- **效能推估**:單次 <0.1–1s @30M nets(推估;對照 full GR 在 60M nets ~10 分鐘)。放進 GP 內圈每 50–100 迭代一次;同一把尺評所有 baseline。
- **正確性驗證**:小 case crossing 對 brute-force 全檢一致;樹長對 FLUTE/GeoSteiner 抽樣校驗(RMST ≤ 1.5×RSMT 為理論上界);cost 計分邏輯與 ICCAD24-D 官方 checker 概念對齊。
- **精度立場**:IO/FT 是拓撲計數而非長度 — local nets(span ≪ region 尺寸)的 MST 拓撲與 router 實走對「跨哪條界」幾乎一致;系統性誤差(congestion 繞行)由 Phase 2 router-based evaluator 校正。

## 7. Benchmark 計畫(1M/5M/10M/30M)

主幹 = **配方 A:架構級 replication(跨 region 連線為真實 RTL 互連)**:

| 規模 | 來源 | 狀態 |
|---|---|---|
| 1M | ISPD05/06 bigblue4(2.18M)/newblue7(2.51M)、ISPD15 superblue12(1.29M,LEF/DEF+fence) | 直接可用(DREAMPlace 原生支援) |
| 5M | MemPool 半 cluster / 3×Group 組合(ETH PULP RTL 開源,NanGate45+OpenROAD) | 需自組 |
| 10M | ISPD 2025 mempool_cluster(≈10M cells / 12M nets,完整 LEF/DEF/LIB/SDC + 已擺置 DEF) | **直接可用,黃金主力** |
| 30M | TeraPool 縮減版(RTL 開源自行合成;ISPD24 有 ≈50M 簡化格式參照) | 工程成本中等 |

- **Region 疊加**:DEF REGION/GROUP;歸屬 ground truth 用 RTL 階層前綴(tile/group/cluster 邊界天然為 region 邊界)。
- **紀律**:嚴禁 naive replication(不補跨 tile nets 會使 cut/IO 假性趨零、Rent p→0);每個擴增 case 以 RentCon 驗 Rent exponent(目標 p≈0.6–0.75);MemPool 系 p 偏低 → 混入 superblue/BlackParrot 等異質設計。
- **備案**:配方 B = replication + 依 Rent 差額 `t·(K·g)^p − K·t·g^p` 補跨 tile glue nets(掃 p∈{0.6,0.65,0.7};ISPD18/19 官方先例;有循環論證風險,僅作佐證)。配方 C = ArtNet 直接生成 30M(實測 100M instances/小時級、p 可控;repo 成熟度未驗證)。
- **ICCAD24-D 測資角色**:block 顆粒度(47–191 blocks),不作 placement benchmark;用作 (a) evaluator cost model 對齊參考(官方 checker 可下載)、(b) top-level 連線分布統計樣本。

## 8. 實驗設計

- **主 metrics**:IO 數、pure FT 數、HPWL 與 evaluator 樹長、GP runtime、峰值 GPU 記憶體。參考 metrics:cut-net 數、λ−1、per-boundary-pair demand vs 容量。
- **Protocol**:所有方法(ours + 三組 baseline)最終 placement 以同一個 evaluator 硬計數;報「同線長水準下的 IO/FT 改善」與 cost-tradeoff 曲線(掃 λ_IO)。
- **矩陣**:規模 {1M, 5M, 10M, 30M} × K {8, 16, 32} × region 型態 {grid, random rectilinear};1M 跑全矩陣 + 多 seed(≥3),10M/30M 跑代表性組合。
- **Ablation**:(i) reweighting-only(M1)vs 可微項(M2/M3);(ii) 各 objective 項逐一開關;(iii) τ schedule;(iv) D_max;(v) evaluator 校正頻率 N;(vi) cov_k 變體。
- **公平性**:baseline 同等調參 effort(Mt-KaHyPar imbalance ε sweep、fence placement 參數);全部走相同 legalization/DP 收尾。

## 9. 里程碑(exit criteria 制)

| 里程碑 | 內容 | Exit criteria |
|---|---|---|
| M0 基建 | fork DREAMPlace 跑通 1M 真實測資;region 輸入格式 + region-id grid + 產生器;Mt-KaHyPar 初始化;兩條 baseline 流程 | 1M 級 baseline 對照表(IO/FT/WL)產出 |
| M1 Evaluator + reweighting | `mst_crossing_eval` 完成並驗證(小 case brute-force 全對、樹長 vs FLUTE 統計);net-reweighting 閉環 | evaluator runtime 報告(1M/10M);IO 相對 flat baseline 明顯下降 |
| M2 可微 IO 項 | S1+S2 op(forward 精確計數、backward surrogate 梯度);τ 退火;auto-normalization | 同線長水準下 IO 優於 M1;reweighting vs 可微 ablation 完成 |
| M3 可微 FT + 容量項 | S4+S7;完整 objective | FT 顯著下降且 WL/IO 無明顯劣化;完整 ablation 表 |
| M4 規模化 | 10M 全流程;30M 測資製作並跑通;H100 計時;記憶體 profile | 30M 端到端數小時內完成;全規模對照表 |
| M5 論文素材 | decode+refinement 定稿;全實驗矩陣;圖表與寫作 | 學位論文 Phase 1 章節草稿 |

## 10. 風險與緩解

1. **乘積式梯度稀釋 / 退火敏感** → S5 FD 備援與 S3 span proxy 已入設計;M2 提早 ablation,不行就換路線(結構已預留)。
2. **30M 測資工程量**(TeraPool 合成、LEF/DEF 自製)→ 10M 主力先行;30M 只需 1–2 個 case;備案配方 B/C。
3. **λ 權重調參** → evaluator 閉環 auto-normalization 為主;必要時小規模 BO/sweep(ART-3D 先例:IO 權重 design-dependent、值得自動搜尋)。
4. **DREAMPlace build 鏈老舊**(PyTorch ≤2.0、CMake+Limbo)→ 容器鎖版;改動最小化(僅新增 ops)。
5. **撞題風險**(NVIDIA GrandPlan 系列持續演進)→ 差異點(D1、計數 objective、tree-crossing evaluator、30M)在論文明寫;M1/M2 盡快出可引用結果。
6. **Rectilinear 實作複雜度** → 資料結構第一天即用「矩形集合」介面,但實驗推進順序 grid → 矩形 → rectilinear。
7. **Evaluator 高估/低估爭議** → 對外主張限定「梯度方向與相對排序」;絕對值準確性由 Phase 2 router evaluator 校正(論文如此陳述)。

## 11. 非目標(Phase 1 不做)

- Timing/power objective;macro 位置優化(Phase 2);region 幾何優化(Phase 2);router-based 精確 evaluator(Phase 2);IO pin 實體 legalization(Phase 3);channel-based cost model(Phase 2 開關)。
- Detailed placement / legalization 品質不是貢獻點(沿用 DREAMPlace 既有元件)。
- 不自建 placer 核心(density/FFT/Nesterov 全沿用)。

## 12. 主要參考

- GrandPlan(ISPD 2026, DOI 10.1145/3764386.3779591);CUHK bistratal 3D placement(TCAD 2024, arXiv:2310.07424);GAP(arXiv:1903.00614);Erdős Goes Neural(NeurIPS 2020);Flora(arXiv:2507.14914)、FTAFP(ASP-DAC 2025)、Piano(arXiv:2508.13161);DREAMPlace(DAC'19/TCAD'20)與 3.0(ICCAD 2020);ICCAD 2024 台灣場 Problem D spec/checker;ISPD 2024/2025 contest(MemPool/TeraPool);ArtNet(arXiv:2510.13582);Mt-KaHyPar(ACM TALG 2024)。
- 細節與逐篇引用見 `docs/research/` 三份報告。
