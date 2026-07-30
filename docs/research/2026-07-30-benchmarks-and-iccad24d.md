# 文獻調查:ICCAD 2024 Problem D、Benchmark 盤點、Netlist 擴增、Channel vs Channelless

- 日期:2026-07-30
- 調查目的:支援「IO-aware top-level placement」專案 Phase 1(flat GPU global placement + partition regions,objective = wirelength + IO + pure feed-through;benchmark 需擴增至 1M/5M/10M/30M cells;決定 channel-based vs channelless design style)。
- 方法:官方 contest 網站與 spec/benchmark 一手下載驗證(iccad-contest.org、ispd.cc、contest GitHub)、學術資料庫(DBLP/arXiv/Semantic Scholar)、實際解壓 testcase 統計。所有下載檔存於 session scratchpad(`/tmp/claude-1100/.../scratchpad/`)。
- 標註慣例:查不到一手證據者標「**未驗證**」;由本人實測(解壓、計數)者標「**實測**」。

---

## TL;DR

1. **「Chip Level Global Router」確為 ICCAD 2024 CAD Contest Problem D,由 MediaTek(聯發科技)出題**,但它是 **台灣場(積體電路電腦輔助設計軟體製作競賽)的「推廣題」**,不在國際場(國際場 2024 只有 Problem A/B/C)。官方 spec(v2024-02-02)與 7/29 評分修正、7 個公開 testcase、checker 均可由官網 Google Drive 下載;本報告完整轉錄 cost function(對我們 evaluator 設計直接可用)。該題把 die 抽象成 block/channel/region/tile,以 **feedthroughable / non-feedthroughable block、HMFT、edge pin density(≈ 我們的 IO number)、through_block_net_num(≈ 我們的 pure feed-through 上限)** 為核心約束——與本專案的 cost 結構高度同源。
2. **公開 placement benchmark 天花板**:傳統 Bookshelf 系(ISPD 2005/2006)最大到 newblue7 = 2,507,954 objects;LEF/DEF 系(ISPD 2015)最大到 mgc_superblue12 = 1,286,948 cells。**真正的破口是 ISPD 2024/2025 global routing contest 的 MemPool/TeraPool 家族:mempool_cluster ≈ 10M cells(ISPD 2025 有完整 LEF/DEF/LIB/SDC),TeraPool-Cluster ≈ 50M cells / 60M nets(僅簡化 .cap/.net 格式)**——證明 30M 級的公開 netlist 已存在且可再生(開源 RTL + NanGate45)。
3. **擴增配方首選**:以「**架構級 replication(MemPool/TeraPool 參數化 RTL)為主幹 + ISPD 2005/2015 舊 suite 為異質性補充**」建 1M/5M/10M/30M 階梯;**避免 naive netlist tiling**(不補跨 tile nets 會讓 cut/IO 指標假性趨零,violate Rent's rule);若需合成,用 **ArtNet(2025,可在 ~1 小時內生成 100M-instance netlist、控制 Rent exponent、支援 macro 與階層)**,ANG(開源)僅適用 ≤ ~250K。
4. **Design style 證據一面倒向 channelless(fully-abutted)為現代大型 SoC 主流**,channel 僅保留給不可 feedthrough 的 hard IP 周邊與 top-level 資源;feedthrough insertion(含 buffer/pipeline)是 abutted flow 的標配。Phase 1 建議:**channelless(regions 完全鋪滿 die、共享邊界)為 primary 模型,IO number = 邊界 crossing 數、pure feed-through = 無 pin 穿越區域數;channel 版本作為次要變體**。

---

## 1. ICCAD 2024 CAD Contest Problem D:Chip Level Global Router(MediaTek)

### 1.1 正名與出處(一手驗證)

- **國際場**(CAD Contest at ICCAD 2024,iccad-contest.org/2024/Problems.html)只有三題:A「Reinforcement Logic Optimization for a General Cost Function」(Cadence)、B「Power and Timing Optimization Using Multibit Flip-Flop」(Synopsys)、C「Scalable Logic Gate Sizing Using ML Techniques and GPU Acceleration」(ASU + NVIDIA Research)。**沒有 Problem D**。此與 contest overview 論文一致:Shao-Yun Fang, Yi-Yu Liu, Chung-Kuan Cheng, Tsun-Ming Tseng, "Overview of 2024 CAD Contest at ICCAD," Proc. ICCAD 2024, pp. 195:1–195:3, DOI [10.1145/3676536.3689908](https://doi.org/10.1145/3676536.3689908)(DBLP 驗證)。
- **台灣場**(2024 積體電路電腦輔助設計軟體製作競賽,iccad-contest.org/2024/tw/03_problems.html)在 A/B/C 之外另設兩個「推廣題」:**Problem D「Chip Level Global Router」(MediaTek Inc.)** 與 Problem E「Automatic Analog Layout for Re-generate SDL & Placement Migration」(Novatek)。官方頁面明列 spec、Q&A、case1/case2、`problem_D_testcase_case02_case03_update`、「Modification of Score Calculation Method」(07/29)、`problem_D_checker_20240830` 等下載連結(全部 Google Drive,本次已實際下載驗證)。
- 年度交叉檢查:台灣場 2023 的 Problem D 是「Fixed-Outline Floorplanning with Rectilinear Soft Blocks」(Maxeda),2025 的 Problem D 是「APB Transaction Recognizer」(TESDA)——**「Chip Level Global Router」唯一出現在 2024**。另見 §1.8:MediaTek 在 **2026 台灣場推廣題 Problem E「Early Floorplanning with Global Route」** 推出了後繼題。
- 使用者記憶中的題名「Chip Level Global Router」**完全正確**;唯一需要修正的認知是:它是台灣場推廣題,因此**沒有** ICCAD invited contest paper 條目(國際場 overview paper 只涵蓋 A/B/C),也不會出現在 ICCAD proceedings 的 contest section。

### 1.2 問題定義(spec v2024-02-02,已下載全文)

動機(spec §1,直譯自中文原文):現代實體設計採階層式;floorplan 後晶片分為 chip level 與 block level。**chip level 只能用 block 佔據後剩餘的不規則區域(channel)繞線**;「現今的趨勢會盡量縮減 channel 的面積來降低晶片成本」。繞線區域由五種元件構成:

| 元件 | 定義(spec 原文摘要) |
|---|---|
| **Channel** | chip level 扣除 block 與 tile 的剩餘區域;趨勢是盡量縮小其面積 |
| **Feedthroughable block** | 允許 net 穿過的 block |
| **Non-feedthroughable block** | 一般 net 不可穿過;但 **HMFT(hard macro feedthrough)** 可以穿過(目的:縮減 channel 面積,把某些 net 穿進 non-feedthrough block) |
| **Region** | chip_top 中散落的 standard cells 依 function 分群而成;**沒有固定形狀,無需顧慮 edge pin density** |
| **Tile** | 由 region 和 block 構成、允許 net 穿過;目的是切割 chip_top 以縮短 top-level 執行時間 |

四種成本(spec §2):(1) **Channel overflow** = occupied/available routing tracks(available 由輸入的 tracks/um 給定;每種 net type 佔用不同 track 數,例:signal net 1 條、clock net 3 條);(2) **Wirelength** = 所有 segment 長度總和;(3) **Edge Pin Density** = block 每條邊上 demand/capacity(capacity 由輸入給定,例:`Block A (x0,y0)->(x1,y1) 1000`);(4) **Turn cost** = 每條 net 的轉折數(via 電阻 → timing)。

**與本專案的對映**:此題的 net 是 block-to-block 的「bundle」(每條 net 有 NUM 個 tracks),沒有 standard cell —— 它是 top-level 抽象層;我們的「IO number」對應其 edge pin density / through_block_edge_net_num,「pure feed-through」對應其 feedthrough(through_block_net_num、is_feedthroughable、HMFT);其「region(依 function 分群、無固定形狀)」與我們的 partition region 概念一致。

### 1.3 輸入/輸出格式(spec §3–§5)

輸入每組測資三類檔案:
1. **DEF**:`chip_top.def`(所有 block 的合法擺置;僅含 VERSION/DESIGN/UNITS/DIEAREA statements + COMPONENTS section)與 `block.def`(block 形狀);遵循 LEF/DEF Language Reference。
2. **CFG(JSON)**:每個 block 的屬性:`block_name`、`through_block_net_num`(扣除起迄於此 block 的 net 後,可 feedthrough 此 block 的**繞線數上限**)、`through_block_edge_net_num`(某條邊上「出發/接收 + feedthrough」的繞線數上限,格式 `[(X0,Y0),(X1,Y1),100]`)、`block_port_region`(邊上可放 pin 的矩形區域)、`is_feedthroughable`(True/False)。實際 testcase 中還多了 `is_tile` 欄位(**實測**)。
3. **Connection matrix(JSON)**:每條 net 八個屬性:`ID`、`TX`(起點 block)、`RX`(終點 block list,可多點)、`NUM`(此 net 佔用的 routing tracks 數)、`MUST_THROUGH`(必須穿過的 block 及其兩條邊)、`HMFT_MUST_THROUGH`(必須以 HMFT 穿過的 non-feedthroughable block 及邊)、`TX_COORD` / `RX_COORD`(相對 block 左下角的座標)。

輸出:`caseOO_net.rpt`,每條 net 拆成水平/垂直 segment(`[net id]` 後接 `(x0,y0),(x1,y1)` 列表,分叉點以共用端點表示)。Evaluator 會把 segment 座標 snap 到 gcell 中心:`x_modified = (floor(x/width_gcell)+0.5)*width_gcell`(y 同理)。**gcell edge 寬度由該 case 最大 net NUM 決定**:例 20 tracks/um、max NUM=100 → gcell edge = 5 um(spec §8(9))。

執行介面:C/C++,執行檔名 `CGR`,用法 `./CGR XX(tracks/um) caseOO.def caseOO.cfg.json caseOO.connection_matrix.json`;Linux + gcc/g++;每組測資 2 小時內完成(含 I/O)。

### 1.4 Cost function 精確定義(spec §8 + 2024/07/29「Modification of Score Calculation Method」)

以下公式同時見於 spec PDF 與 7/29 修正圖檔(修正圖中以紅字標出總分式與 illegal 罰項;兩者最終一致,已逐字核對):

```
score = 0.55 * cost_overflowLength
      + 0.35 * cost_edgePinDensity
      + 0.10 * e^(time / (2*60*60))          # time 以秒計,2 小時為分母
      + 0.30 * penalty_#pin_constraint
      + 0.01 * penalty_#net_turn
      + 0.50 * penalty_#illegal
```

各項定義:

```
cost_overflowLength
  = Σ_{net ∈ all nets} [ Σ_{segment,共 netlength/gcell_width 段}
        (1 + indicator( #used_track / capacity_gcell_edge > 0.7 ))   # 超過 0.7 的 gcell edge 記 1,否則 0
        * gcell_width
    ] / HPWL_bbox_net                                                # 每條 net 以其 bounding box HPWL 正規化

HPWL_bbox_net = 該 net bounding box 的 HPWL

cost_edgePinDensity
  = Σ_{block ∈ all blocks} Σ_{edge ∈ block edges} Σ_{segment,共 edgelength/gcell_width 段}
        indicator( #used_track / capacity_gcell_edge > 0.6 )

penalty_#pin_constraint
  = Σ_{constraint ∈ all constraints} e^( r )   其中 r = #used_track / #pin_constraints 若 r > 1,否則 r = -inf
    (即:未超限的 constraint 貢獻 e^{-inf}=0;超限者以指數放大)

penalty_#net_turn
  = Σ_{net ∈ all nets} e^( t )                 其中 t = #net_turn 若 #net_turn > 1,否則 t = -inf
    (轉折數 ≤ 1 的 net 不罰;轉折多者指數放大)

penalty_#illegal
  = Σ_{net ∈ illegal nets} HPWL_bbox_net
```

不予計分(硬性 legality,spec §8(2)–(7)):超時 2 小時、輸出格式錯誤、起迄點錯誤、違反 MUST_THROUGH / HMFT_MUST_THROUGH、穿過 `is_feedthroughable=False` 的 block、繞出 chip_top。**超過 `through_block_net_num` / `through_block_edge_net_num` 則是扣分**(soft,進 penalty 項),不是直接 0 分(spec §8(8))。

註記/歧義(如實記錄):(a) spec §2(1) 把 overflow 分子分母都誤寫成 "number of available routing tracks",依上下文第 ii 項應為 per-type 佔用 track 數;(b) spec §8(1) 寫「總分由高至低排序」,但 score 各項皆為 cost/penalty(愈低愈好),排序方向疑為筆誤,**得獎名單未附分數,無法反推,標未驗證**;(c) `penalty_#pin_constraint` 中 `#pin_constrains` 分母的精確語意(所有 constraint 數或該 constraint 的容量)spec 未明說——實作 evaluator 時建議以官方 checker(`problem_D_checker_20240830`,官網可下載)為準。

**對我們 evaluator 的可借鑑點**:(1) 以「indicator 超過閾值(0.7/0.6)才計」的 soft-overflow 設計,比連續 overflow 更貼近 DRC 風險語意;(2) 每條 net 以自身 HPWL 正規化,避免長 net 支配;(3) 違規用指數罰項而非硬拒絕,利於 optimizer 漸進;(4) runtime 直接入 score(e^{time/limit});(5) IO/feedthrough 以「**per-edge capacity + per-block feedthrough 上限**」建模——這正是我們 Phase 1 IO cost 與 pure feed-through cost 的自然離散版本。

### 1.5 Benchmark(實測統計)

Spec §7 稱提供 5 組公開 + 3 組隱藏測資;實際 2024/06/28 更新包(官網 case2 連結,`problem_D_testcase_case02_case03_update` 同捆)內含 **7 組公開 case(case00–case06)**(**實測**,已解壓統計):

| case | #blocks | feedthroughable | tile | #nets(bundle) | Σ NUM(track demand) | MUST_THROUGH nets | HMFT nets |
|---|---|---|---|---|---|---|---|
| case00 | 47 | 16 | 0 | 2,422 | 122,638 | 200 | 94 |
| case01 | 47 | 16 | 0 | 2,467 | 137,566 | 310 | 137 |
| case02 | 175 | 81 | 0 | 2,754 | 120,589 | 299 | 101 |
| case03 | 175 | 81 | 0 | 2,828 | 129,769 | 507 | 131 |
| case04 | 76 | 49 | 0 | 2,306 | 114,744 | 255 | 41 |
| case05 | 76 | 49 | 0 | 2,239 | 123,004 | 336 | 59 |
| case06 | 191 | 56 | 6 | 2,824 | 124,150 | 325 | 93 |

規模結論:此題在 **block 顆粒度**(47–191 blocks、~2.2k–2.8k net bundles、~115k–138k track demand),不含 standard cells——它驗證的是 top-level 抽象與 cost 定義,**不能直接當作我們 10M–30M cell 的 placement benchmark**,但其 CFG/connection-matrix 格式與 cost 定義非常適合作為我們 Phase 1「region-graph 層」的 evaluator 參考,且測資可直接取用作 top-level 連線分布的統計樣本。

### 1.6 優勝隊伍(官方 2024 台灣場「競賽結果」頁)

Problem D 得獎名單(iccad-contest.org/2024/tw/05_results.html):
- **特優**(兩隊並列):`cadd0011` 呂紹謙、黃名毅、吳苡嘉(元智大學;指導:林榮彬、林佑政教授);`cadd0056` 蔡岳宏、李睿穎、吳育丞、張雅淳(國立中央大學;指導:陳聿廣教授)。
- **優等**:`cadd0003`(台科大,陳勇志)、`cadd0004`(台科大,蘇順豐)、`cadd0013`(清大,黃婷婷)。
- **佳作**:`cadd0058`、`cadd0007`、`cadd0032`、`cadd0019`、`cadd0034`(中央、元智、成大等)。

### 1.7 賽後論文與開源 code

- **得獎隊論文(元智,特優 cadd0011)**:ShaoChien Lu, YuCheng Lin, MingYi Huang, YiChia Wu, RungBin Lin, "**Efficient Chip-Level Global Router: ICCAD 2024 CAD Contest Problem D**," 2025 International VLSI Symposium on Technology, Systems and Applications (**VLSI-TSA 2025**)。摘要:以高效方法決定繞線順序、分解 multi-pin nets、A\* 逐 net 繞線,最小化 overflow 與計算時間;自稱比「另一支第一名隊伍」好約 20.86%([Semantic Scholar 條目](https://www.semanticscholar.org/paper/f7e1713d9821fc9bd0cdc355292204f2601a064e);DOI 未查得,**未驗證**)。
- 中央大學隊(cadd0056)之對應論文:未查得(**未驗證**)。
- 開源 code:GitHub 僅見學生練習性 repo(`zouyuoz/Problem_D`、`Erison88/ICCAD_contest`、`charlottinana23/ICCAD_contest`,均 0–1 star,非官方、非得獎隊)。**官方 checker 可下載**(§1.1 連結),這對我們重建 evaluator 最有價值。

### 1.8 後繼題:2026 台灣場 Problem E「Early Floorplanning with Global Route」(MediaTek,進行中)

2026 台灣場推廣題 Problem E(spec v2026-03-11,16 頁,已下載)把同一套 chip-level 概念上移到 early floorplanning:soft/hard/edge blocks + channel + feedthrough 共同優化 wirelength/area/cost。可直接借鑑的量化規則:**feedthrough 數 = 該 block 因穿越產生的 input/output port 數 ÷ 2**(同一 net 進出同 block N 次即計 N 次);**channel wire density limit = 25 nets/um**;允許 feedthrough 的 block 伴隨額外面積成本;不允許 feedthrough 的 block(hard/edge block)強制走 channel。另 2026 國際場/台灣場 Problem C 為 **Intel「The FloorSet Challenge: Data-Driven SoC Floorplanning」**(見 §2 FloorSet)。這兩題顯示:**top-level 的 channel/feedthrough/IO 建模是 2024–2026 產業界持續出題的活躍方向**,與本專案高度同時代。

---

## 2. Benchmark 盤點

### 2.1 總表

「格式」欄:BS = Bookshelf,LD = LEF/DEF。規模皆為該 suite 內**最大**者;來源均為官方頁面/官方論文/官方 slides(已逐一抓取),或標註。

| Suite(年) | 最大 design | #cells/objects | #nets | macro | region/fence | 格式 | 取得方式 |
|---|---|---|---|---|---|---|---|
| ISPD 2005 placement | bigblue4 | 2,177,353 objects(2,169,183 movable / 8,170 fixed) | 2,229,886 | 有(fixed objects) | 無 | BS | ISPD 2005 contest 網頁/論文(Nam et al., ISPD'05, DOI 10.1145/1055137.1055182);DREAMPlace repo 附載 |
| ISPD 2006 placement | newblue7 | 2,507,954 objects(2,481,372 movable / 26,582 fixed) | 2,636,820 | 有 | 無 | BS | ISPD 2006 contest(官方 slides「The ISPD 2006 Placement Contest and Benchmark Suite」,ispd.cc/slides/2006/7-3.pdf,含全表;>1M objects 者 5 個:newblue5/6/7、adaptec5(0.84M)註:adaptec5 未達 1M) |
| DAC 2012 routability(superblue) | superblue12 | 「>1M movable nodes」(官方 contest 論文敘述) | — | 有 | 無 | BS + LD 資訊 | Viswanathan et al., "The DAC 2012 Routability-Driven Placement Contest and Benchmark Suite," DAC'12(archive.sigda.org 已失聯,可由 UCSD RePlAce/學界鏡像取得) |
| ICCAD 2012 routability | superblue 家族 | 至 ~1.3M objects(同 superblue 家族;逐檔數字**未驗證**) | — | 有 | 無 | BS | ICCAD-2012 contest 論文(N. Viswanathan et al.) |
| ISPD 2015 detailed-routing-driven placement | mgc_superblue12 | 1,286,948 cells(89 macros) | 1,293,413 | 有 | **有 fence regions**(superblue11_a:4 個、superblue16_a:2 個、des_perf_b:12 個等) | **LD**(tech.lef + cells.lef + floorplan.def) | ISPD 2015 官方 benchmark description PDF(ispd.cc/contests/15,全表已核對:superblue11_a = 925,616 cells/1,458 macros;superblue16_a = 680,450/419) |
| ISPD 2018 initial detailed routing | ispd18_test10 | 290,386 std cells | 182,000 | 有(部分 test) | 無 | LD | ispd.cc/contests/18(官方表已核對) |
| ISPD 2019 initial detailed routing | ispd19_test9/10 | 899,341 / 899,404 std cells(16 memory blocks) | 895,253 | 有 | 無 | LD | ispd.cc/contests/19(官方表已核對;**test6–10 為 quad-core 複製生成**,見 §3.2) |
| MLCAD 2023 FPGA macro placement(AMD) | 140 公開 + 198 隱藏 | 每檔 561,632–720,227 instances | 3,721,746–4,730,679 | FPGA macros(BRAM/DSP/URAM cascades) | 有 region/clock 約束 | FPGA bookshelf 變體 | TILOS-AI-Institute/MLCAD-2023-FPGA-Macro-Placement-Contest(GitHub);官方論文(UCSD c398)明載 **Rent exponent 0.65–0.72、以 AMD 內部 netlist generator 生成** |
| TILOS MacroPlacement | MemPool Group | 360,724 cells + 324 macros | — | 有 | 無 | LD + RTL + 完整 SP&R flow(NG45/ASAP7/SKY130HD) | github.com/TILOS-AI-Institute/MacroPlacement(另含 Ariane133/136 ≈19.8k、NVDLA 45,295、BlackParrot 214,441;**全數 <1M**) |
| ISPD 2024 GPU/ML global routing(NVIDIA) | **TeraPool-Cluster** | **≈50M cells** | **≈60M nets** | 有 | 無 | **簡化 .cap/.net**(由 LEF/DEF 抽出) | liangrj2014/ISPD24_contest(GitHub;官方公告原文:"We released a testcase with around 50M cells and 60M nets!";另 mempool_cluster "around 10 million cells") |
| ISPD 2025 performance-driven global routing(NVIDIA) | mempool_cluster | ≈10M cells(承 ISPD24) | 12,047,279(visible)/ 12,168,735(blind);timing endpoints 1,082,397 | 有 | 無 | **industry-standard LEF/DEF/LIB/SDC + 簡化 .cap/.net**;OpenROAD evaluation flow | ispd.cc/contests/25 + liangrj2014/ISPD25_contest(GitHub;NanGate45;優勝:Hippo(北大)、RL-Route、It's MyRoute!!!!!) |
| OpenROAD MegaBoom | MegaBoom(Chipyard BOOM) | ≈2M instances(階層);flatten 後 ≈1M(官方 GitHub 討論 #1710 敘述,**約數**) | — | 有 | 無 | RTL + ORFS flow(ASAP7) | github.com/The-OpenROAD-Project/megaboom |
| OpenTitan Earl Grey | top_earlgrey | cell 數**未驗證**(官方文件未直接列 std-cell count) | — | 有 | 無 | RTL(SystemVerilog) | github.com/lowRISC/opentitan |
| Basilisk(ETH) | Basilisk RV64 SoC | 「multi-million-gate」(論文標題自述;精確數**未驗證**) | — | 有 | 無 | RTL + Yosys/OpenROAD(IHP130) | arXiv 2405.04257 |
| HighTide(UCSC, 2026) | 16 designs | 網站稱 "under 20k to over 1M";逐檔最大列示為 Gemmini ≈570k(**>1M 者未逐檔驗證**) | — | 有 | 無 | RTL + ORFS(ASAP7/NG45/SKY130HD) | vlsida.github.io/HighTide;arXiv 2606.04126 |
| FloorSet(Intel, ICCAD 2024) | FloorSet-Prime/Lite | **floorplan 顆粒度**:每筆 ~120 partitions;1M 訓練 + 100 測試 layouts(合成、抽真實 SoC 統計) | — | — | **fully-abutted rectilinear partitions**、shape/boundary/grouping/MIB/pre-place 約束 | tensor/pickle(PyTorch) | github.com/IntelLabs/FloorSet;arXiv 2405.05480;DOI 10.1145/3676536.3676814 |

### 2.2 重點結論

- **≥1M cells 的公開 flat netlist**:bigblue4(2.18M)、newblue5/6/7(1.23M/1.26M/2.51M)、superblue12(DAC12/ICCAD12/ISPD15 版 1.29M)、ISPD24/25 mempool_cluster(≈10M)、TeraPool-Cluster(≈50M,僅簡化格式)、MegaBoom(≈1–2M,需自行跑 flow)。**公開最大 netlist = TeraPool-Cluster ≈50M cells/60M nets**(簡化 .cap/.net);**最大且帶完整 LEF/DEF 的 = ISPD25 mempool_cluster ≈10M cells/12M nets**。
- **有 region/fence 語意的只有 ISPD 2015**(DEF REGION/fence,最大 0.93M cells + 4 fence regions)與 FloorSet(floorplan 層)。我們的 partition-region 語意需自行在大 netlist 上疊加(見 §3.4)。
- **有無 macro**:兩類都齊全(ISPD05/06 objects 含 fixed macros;ISPD15/TILOS/MemPool 有顯式 macros;ISPD18/19 部分 test 無 macro)。Phase 1「macro 固定、region 幾何給定」與 ISPD15 fence-region 情境最接近。
- DREAMPlace(我們的 GPU placement 基底)原生支援 Bookshelf 與 LEF/DEF,並以 ISPD2005/06、DAC/ICCAD2012、ISPD2015 為標準回歸集——以上 suite 可直接餵入。

---

## 3. Netlist 擴增/合成方法(目標:30M cells 且 IO/cut 指標仍有意義)

### 3.1 Artificial netlist generator 文獻(經典 → 2026)

| 工具/論文 | 年/venue | 方法與可控統計 | 規模上限 | 開源 |
|---|---|---|---|---|
| RMC(Darnauer & Dai, "A Method for Generating Random Circuits and Its Application to Routability Measurement") | FPGA 1996 | top-down 分割、以 **Rent's rule 直接參數化**(給定 p 生成) | 小 | 否 |
| GEN/Circ(Hutton, Rose, Grossman, Corneil, "Characterization and Parameterized Generation of Synthetic Combinational Benchmark Circuits") | TCAD 17(10), 1998 | 抽取真實電路 profile(shape、fanout、reconvergence)再生成 | 中 | 學術散佈 |
| **GNL**(Stroobandt, Verplaetse, Van Campenhout, "Generating Synthetic Benchmark Circuits for Evaluating CAD Tools") | TCAD 19(9), 2000(timing 加強版 ICVLSI 2002) | **bottom-up clustering,嚴格維持 Rent scaling(T = t·g^p)**;缺 logic depth 控制 | ArtNet 實測:3 小時內可到 **50.5M instances**(100M 需 ≥29,000s) | 學術散佈(UGent) |
| Kundarewich & Rose, "Synthetic Circuit Generation Using Clustering and Iteration" | TCAD 23(6), 2004 | clustering + 迭代修正統計 | 中 | 否 |
| **ANG**(D. Kim, H. Kwon, S.-Y. Lee, S. Kim, M. Woo, S. Kang, "Machine Learning Framework for Early Routability Prediction with Artificial Netlist Generator," DATE 2021;期刊版 D. Kim, S.-Y. Lee, K. Min, S. Kang, "Construction of Realistic Place-and-Route Benchmarks for Machine Learning Applications," **TCAD 42(6), 2023, pp. 2030–2042**) | DATE 2021 / TCAD 2023 | 6 個拓撲參數:`num_insts`、`num_primary_ios`、`avg_net_degree`(2.5–4.0)、`avg_net_bbox`、`avg_topo_order`、`comb_ratio`;逐 edge 機率生成;整合 OpenROAD(`artnetgen_*` 命令),輸出可直接進 P&R | **~250K instances(3 小時上限內;500K 需 ≥38,000s,超線性)** | **是**:github.com/daeyeon22/artificial_netlist_generator |
| Kang(invited), "Artificial Netlist Generation for Enhanced Circuit Data Augmentation" | **ISPD 2025**(DOI 10.1145/3698364.3709121) | ANG 路線總結(POSTECH) | — | — |
| Fengler, Chen, Nassif, Schlichtmann, "Enabling Machine Learning for Power Modeling via Artificial Netlist Generation" | ISCAS 2025(DOI 10.1109/ISCAS56072.2025.11043199) | 面向 power model 訓練的 ANG 應用 | — | 未查 |
| **ArtNet**(Kahng, Kang, S. Park, D. Yoon, "ArtNet: Hierarchical Clustering-Based Artificial Netlist Generator for ML and DTCO Applications") | arXiv 2510.13582(v2 2026-02;TCAD 投稿版) | 階層 clustering(GNL 式 bottom-up)+ **Rent exponent p 直接可控(RentCon 量測)**、sequential ratio、logic depth min/max、**支援 macro 插入與 submodule 異質性**、#PI/PO 匹配 | **100M instances 在 980–3,212 秒**(sweep 500K–100M、p∈0.45–0.55 實測);對照:GNL 50.5M/3hr、ANG 250K/3hr | repo `github.com/shypark98/ArtNet`(論文標註 "GitHub for review",成熟度**未驗證**) |
| ML 生成路線:LayerDAG(ICLR 2025)、D-VAE(NeurIPS 2019)、HYGENE(AAAI 2025)、LLM4Netlist(JETCAS 2025) | 2019–2025 | 擴散/自迴歸/LLM 生成 DAG/hypergraph | **小**(LayerDAG ≤400 nodes;HYGENE 不及 multi-million;ArtNet 綜述之評語) | 各異 |
| **AMD 內部 netlist generator**(MLCAD 2023 contest 官方論文) | MLCAD 2023 | 工業界先例:生成 140+198 個 56 萬–72 萬 instance 的 FPGA netlists,**顯式掃 Rent exponent 0.65–0.72**、utilization、clock 數 | ~720K/檔 | 否(產出的 benchmark 開放) |
| FloorSet(Intel) | ICCAD 2024 | **floorplan 層**合成:抽真實 SoC 拓撲統計生成 1M 筆 ~120-partition 的 fully-abutted layouts(含 MIB、grouping、boundary 約束) | partition 顆粒度 | **是**(IntelLabs/FloorSet) |

**要點**:(1) 學界 2020s 共識是以 **Rent exponent 為第一階「connectivity 保真」指標**(ANG/ArtNet/MLCAD 皆以其為控制參數;RentCon 為標準量測工具);(2) 30M 級生成在 2025 之前僅 GNL 勉強可行,**ArtNet 是目前唯一宣稱 100M-instance/小時級的生成器**;(3) ML 生成路線(diffusion/LLM)規模全數不足,可忽略。

### 3.2 簡單 tiling/replication:先例與缺陷

**先例(組織者背書)**:
- ISPD 2018/2019 contest 官方 testcase 就是 replication 產物:官方表格備註明寫 "Quad-core design with double/triple/quadruple the number of standard cells with 16 memory blocks"(ispd18_test7–10、ispd19_test6–10)——把單核設計複製 2×2 並補上共享 memory blocks 與 top 連線。
- ISPD 2024 contest 用 **架構級 replication**:MemPool-Tile(~18K)→ Group(~360K)→ Cluster(~10M)→ TeraPool-Cluster(~50M)是 ETH PULP 平台同一參數化架構的放大(tile→group→cluster 的階層互連為真實 RTL),被 NVIDIA 直接採為 50M-cell contest benchmark。
- MegaBoom 亦是參數化放大(BOOM 多核 config)。

**缺陷(對 cut/IO 指標的致命性)**:
- **數學事實**:把 netlist 複製 K 份而不加跨 tile nets,任何把不同 copy 分開的 partition 其 cut = 0;我們的 IO number 與 pure feed-through 指標會**假性趨零**,optimizer 只要把每個 copy 塞進一個 region 就「作弊成功」。此現象即 Rent's rule 失效:複本集合的 terminal 數不再隨 g^p 成長(p→0)。
- **Rent's rule 理論根據**:Landman & Russo("On a Pin Versus Block Relationship for Partitions of Logic Graphs," IEEE TC-20(12), 1971)給出 T = t·g^p 及大模組端的 **Region II 飽和**;Christie & Stroobandt("The Interpretation and Application of Rent's Rule," IEEE TVLSI 8(6), 2000)系統化說明 p 反映互連複雜度、均質/異質電路差異,以及 wirelength 分布對 p 的敏感性。**cut-based 指標(如我們的 IO)在統計上由 p 決定**:若擴增後的 netlist p 明顯低於真實 SoC(現代設計常見 p ≈ 0.6–0.75;MLCAD 用 0.65–0.72),IO 最佳化的難度與解型態都會失真。
- **實務警訊**:ArtNet 論文對舊 generator 的批評同樣適用於 naive replication——「overly connected 或 under-connected 的設計會 mismatch 真實 net degree 分布與 Rent 參數」;PEKO 系列(Cong et al.;Chang, Cong, Xie "Optimality and Scalability..." 及 arXiv 2305.16413 書章)則證明「統計相似但結構人工」的 benchmark 會系統性扭曲 placer 行為評估,須以「net-degree 分布逐一匹配 + 明知其人工性」的態度使用。

**正確的 replication 姿勢**(由上述先例歸納):複製 K 份後,**必須補跨 tile 互連**,且補的量與分布要能把整體 Rent exponent 拉回目標值——ISPD18/19 用共享 memory blocks + top nets,MemPool/TeraPool 用真實 NoC/hierarchical interconnect,MLCAD/AMD 用 generator 直接控 p。

### 3.3 三個候選配方(擴增到 1M/5M/10M/30M)

**配方 A(首選):架構級 replication —— MemPool/TeraPool 家族為主幹 + 舊 suite 異質補充**
- 作法:1M ≈ 3×MemPool-Group 或 cluster 子集;5M ≈ 半 cluster;10M = mempool_cluster(**ISPD 2025 已附完整 LEF/DEF/LIB/SDC + 已擺置 DEF,可直接用**);30M = TeraPool 縮減版(TeraPool RTL 開源,ETH PULP;或以 ISPD24 .cap/.net 反推 + 自行 synthesis)。Partition region 天然存在:tile/group/cluster 階層邊界即 region 邊界,跨 region nets 為真實互連 → **IO 與 pure feed-through 指標真實**。
- 證據:ISPD 2024/2025 contest 已驗證此家族到 50M 的可行性與工業擁擠度(§2);NanGate45 + OpenROAD flow 全開源。
- 風險:(i) regular manycore 的 p 偏低、連線過於規則,對「異質 SoC」的代表性不足 → 以 BlackParrot/MegaBoom/ISPD15 superblue 混入;(ii) 自跑 synthesis/placement 到 30M 的工程成本(MegaBoom 單次 build ~24 小時等級);(iii) TeraPool 官方只給簡化格式,LEF/DEF 需自製(**風險中**)。

**配方 B(備案 1):replication + 依 Rent's rule 補跨 tile nets(合成 glue)**
- 作法:把 bigblue4/newblue7/superblue12 各複製 K 份鋪成 K 個 region;以 GNL/ArtNet 的 bottom-up 邏輯在「region=超級節點」層生成 top-level netlist(或直接規定跨 region net 數 N_cross = t·(K·g)^p − K·t·g^p 的 Rent 差額),隨機掛到各 copy 的邊界附近 cells;掃 p ∈ {0.6, 0.65, 0.7} 產生難度階梯。
- 證據:ISPD18/19 官方即為「replication+glue」;MLCAD/AMD 證明掃 p 是工業界生成 benchmark 的標準手段;Rent 理論(§3.2)給出補線量的閉式估計。
- 風險:**IO/cut 指標的 ground truth 由我們自己的合成參數決定**(circular);glue nets 的空間分布(掛哪些 cells)會直接影響 feed-through 型態,需 sensitivity study;審稿人可能質疑 realism → 必須同時報告配方 A 的真實設計結果。

**配方 C(備案 2):ArtNet/ANG 直接生成 30M**
- 作法:用 ArtNet 以目標參數(N_inst=30M、p≈0.65、S_ratio、logic depth、macro 插入、**submodule 階層 = 我們的 partition**)一次生成;其 submodule/cluster 機制天然給出 region 歸屬與跨 region 連線。1M 級可用開源 ANG 交叉驗證(ANG 上限 ~250K,僅能當小規模 sanity check)。
- 證據:ArtNet 實測 100M instances / 980–3,212 秒,Rent 誤差小(論文含 RentCon 驗證);ISPD 2025 invited talk 顯示社群接受度上升。
- 風險:ArtNet repo 成熟度未驗證("for review");人工 netlist 無真實 RTL,Phase 2/3(mix-sized、IO legalization)延伸性差;timing/functional 結構人工。

(次要變體:**多設計拼接**——把數個不同 benchmark 各作為一個 region 拼在同一 die,再以配方 B 的 Rent 差額補 top-level nets。適合模擬異質 SoC,但跨 region 連線同樣是合成的,歸類為 B 的特例。)

### 3.4 region 疊加(所有配方通用)

不論 A/B/C,partition region 幾何以 **DEF REGION/GROUP(fence)語法**表達最通用(ISPD 2015 即用此;DREAMPlace 支援 fence region);cell→region 歸屬:配方 A 用 RTL 階層(module 前綴),B 用 copy id,C 用 ArtNet cluster id。如此 Phase 1 的「IO number」可定義為跨 region-boundary 的 net crossing 數、「pure feed-through」為 net bbox/route 穿過無其 pin 之 region 的事件數,兩者皆可在 GPU 上以 region raster + net bbox 快速估計。

---

## 4. Channel-based vs Channelless(abutted)hierarchical design style

### 4.1 一手/工業證據

1. **MediaTek(Problem D spec, 2024)**:「現今的趨勢會盡量縮減 channel 的面積來降低晶片成本」;HMFT 的存在目的即「縮減 channel 面積,把 net 穿進 non-feedthrough block」。→ 出題廠商直接陳述 channel 面積是要被壓縮的成本。
2. **Yi Hong et al., "Channel Based SOC Feedthrough Insertion Methodology," IEEE(2022)**(IEEE Xplore 9825309):摘要開宗明義——「In **fully abutted** SoC designs, feedthrough insertion is often an important step before pin placement in floorplan stage」;當走 channel 的線太多時 channel 面積佔比過大,「IPs are often abutted at the SoC level, and **channels are replaced by SoC feedthrough wires** that run across the IPs」;並處理 Multiple Instantiated Blocks(MIB)的 feedthrough 自動化。→ 直接證言:abutted 是為省面積,feedthrough 取代 channel。
3. **FTAFP(Z. Li, K. Tian, J. Zhai, Z. Li, S. Kai, S. Xu, B. Yu, K. Zhao, "FTAFP: A Feedthrough-Aware Floorplanner for Hierarchical Design of Large-Scale SoCs," ASP-DAC 2025**, DOI 10.1145/3658617.3697728;北郵+華為諾亞+CUHK):feedthrough 定義為 through-module connection,需在 module 內加 **buffers 與 ports**;不及早規劃會導致 timing 劣化、congestion、後期 ECO。以 SCB-tree SA 共同優化 feedthrough module 數與線長。→ 學界已把 feedthrough 成本內建進 floorplanning objective。
4. **Flora(Z. Xu, J. Wang, S. Xu, Z. Geng, M. Yuan, F. Wu, "One Step Beyond: Feedthrough & Placement-Aware Rectilinear Floorplanner," arXiv 2507.14914, 2025**;USTC+華為):目標是 **zero-whitespace(完全 abutted)layout**;定義兩個 feedthrough 指標——**FTmod(每條 net 的 feedthrough module 數)與 FTpin(共享邊上的 feedthrough pin 數)**。→ 與我們的 pure feed-through(=FTmod)與 IO number(=FTpin)幾乎一一對應,是 cost model 的直接文獻錨點。
5. **GrandPlan("GrandPlan: Differentiable, Simultaneous Top-Level Floorplanning and Partition-Level Cell Placement for Large-Scale IP-Cores," ISPD 2026**, DOI 10.1145/3764386.3779591):敘述工業現況——「top-level floorplanning 中 **die 被切成 exactly abutted regions**」、現行實務仍大量人工;GPU differentiable 框架同時做 top-level floorplan + partition 內 cell placement,8 顆工業 IP-core(**至 25M cells**)上,總線長 −14%、**cross-partition feedthrough wirelength 平均 −27%**,runtime 1.2 小時。→ 與本專案最接近的同期工作(abutted region + GPU + feedthrough 指標),亦佐證 exactly-abutted 為工業預設。
6. **FloorSet(Intel, ICCAD 2024)**:自真實 SoC 統計生成的 dataset,其 Prime 變體即「**fully-abutted** rectilinear partitions」,典型大型 SoC ~120 partitions。→ Intel 對「真實 SoC top-level = fully abutted」的背書。
7. **I-Lun Tseng(MediaTek), "Challenges in Floorplanning and Macro Placement for Modern SoCs," ISPD 2024 invited**(DOI 10.1145/3626184.3639695;slides 受密碼保護,內容**未驗證**):MediaTek 對現代 SoC floorplan 挑戰的產業視角。
8. **eInfochips(Arrow)工程實務文**(design-reuse, 2020, "Strategy To Fix Register-to-Register Timing For Large Feedthrough Blocks…"):feedthrough block 的 port 位置/尺寸是 hard-fixed;跨 block timing 以 **pipeline registers、upsizing、buffer insertion** 處理。→ feedthrough 的 timing 成本與 buffer/pipeline 實務。
9. **歷史對照**:Y.-C. 等 "Simultaneous Routing and Feedthrough Algorithm to Decongest Top Channel"(IEEE, 2008)顯示「用 feedthrough 疏解 top channel」的問題意識已存在十餘年;2020s 的變化是 abutted 成為預設而 channel 成為例外。

### 4.2 實務 flow 歸納(供 Phase 1/3 cost model)

- **主流**:fully-abutted(channelless)top-level;partition 區塊直接相鄰,跨 partition 訊號以 **feedthrough(block 內開 port + 內部 buffer/pipeline)** 通過;feedthrough 需求在 floorplan/pin-assignment 階段規劃(FTAFP/Flora/GrandPlan;Hong 2022 的自動化 flow)。
- **channel 保留給**:(i) 不可 feedthrough 的 hard IP(memory/analog/第三方 hard macro——Problem D 的 non-feedthroughable block、2026 Problem E 的 hard/edge block);(ii) top-level glue logic、repeater/buffer bank、clock/power trunk;(iii) 早期規劃時的彈性餘裕(2026 Problem E 仍以 channel + feedthrough 混合建模,channel 有 wire density limit 25 nets/um)。
- **feedthrough 的成本**:block 內面積(buffer/pipeline)、port 資源(edge pin density/through_block_edge_net_num 上限)、timing(多級 pipeline 延遲);對 top-level placer 而言即我們的 IO cost 與 pure feed-through cost。

### 4.3 對 Phase 1 的建議

- **primary:channelless/abutted 模型**——regions 完全鋪滿 die(zero-whitespace,如 Flora),region 邊界零寬;IO number = net 在 region boundary 的 crossing 數(= FTpin/edge pin density),pure feed-through = net 穿過無其 pin 的 region(= FTmod/through_block_net);可另加 per-boundary capacity(Problem D 的 `through_block_edge_net_num`)作為 constraint 版本。
- **secondary(sensitivity)**:channel 版本——region 間留固定寬度 channel,IO 記在 region-channel 介面,channel 容量 25 nets/um 可直接沿用 2026 Problem E 的數字作 default。
- 這樣的雙模型與 2024 Problem D(channel 為主、feedthrough 減 channel)→ 2026 Problem E(channel+feedthrough 混合)→ GrandPlan/Flora(exactly abutted)的演進譜系一致,論文敘事上可引用整條證據鏈。

---

## 5. 綜合分析

### 5.1 Benchmark 擴增建議

**首選(A+ 混合)**:
1. **1M**:ISPD05 bigblue4 / ISPD06 newblue7(直接可用,Bookshelf,DREAMPlace 原生)+ ISPD15 mgc_superblue12(LEF/DEF、有 macros)——真實工業 netlist,無需擴增。
2. **5M/10M**:**ISPD 2025 mempool_group/mempool_cluster**(LEF/DEF/LIB/SDC 齊全、已擺置 DEF 可當初始解/對照;region = group/tile 階層邊界)。
3. **30M**:TeraPool 路線(ETH PULP RTL + NanGate45 自行 synthesis;或先以 ISPD24 TeraPool .cap/.net 做 routing 層實驗)。工程量最大,建議 Phase 1 後期才投入;先以 10M 完成方法學驗證。
4. Region 疊加一律用 DEF fence-region 語法(ISPD15 相容)。

**備案**:若 30M 真實設計工程受阻,採 **配方 B(replication + Rent 差額補跨 tile nets,掃 p=0.6/0.65/0.7)** 快速出 30M 測資,並以配方 C(ArtNet)生成對照組;論文中同時報告「真實(≤10M)+ 合成(30M)」兩組,並附 Rent-exponent 保真驗證(RentCon)以回應 realism 質疑。**絕不使用不補線的 naive replication**(cut/IO 假性偏低,§3.2)。

### 5.2 Design style 證據傾向

**channelless(fully-abutted)為現代大型 SoC top-level 主流**(MediaTek spec、Intel FloorSet、GrandPlan、Hong 2022、FTAFP/Flora 六路互證);channel 僅存於 hard-IP 周邊與 top-level 資源帶。Phase 1 應以 channelless 為預設 cost model(IO=boundary crossing、pure feed-through=FTmod),channel-based 作為可切換變體;evaluator 的 soft-threshold/指數罰項設計可直接移植 Problem D 公式(§1.4)。

---

## 附:主要來源清單

**Contest 官方**
- ICCAD 2024 國際場問題頁:https://www.iccad-contest.org/2024/Problems.html(僅 A/B/C)
- 台灣場 2024 問題頁(Problem D 條目與全部下載連結):http://iccad-contest.org/2024/tw/03_problems.html
- Problem D spec(v2024-02-02,Google Drive id `1Tb-LXfna6RLHOsSSE0nT41kwk_bwqDvT`);評分修正(id `15l-kvSUyTWQZ7w0vLJlA4l-ArosNsn48`);Q&A 78 頁(id `1LlrtVYHo5A9y2c3iAWaIxnqqP2Y5FrcQ`);checker(folder id `1STPTkbTdNgXys-Ah9eCyII0vp4ogYEoC`);testcases(case2 檔 id `1M9uTcldgfoA0V9wxEwEeLQLqiF7k6kL1`,含 case00–06)
- 台灣場 2024 得獎名單:http://iccad-contest.org/2024/tw/05_results.html
- 台灣場 2026 問題頁(Problem E「Early Floorplanning with Global Route」/MediaTek;Problem C FloorSet/Intel):http://iccad-contest.org/2026/tw/03_problems.html;Problem E spec v2026-03-11(Drive id `1wTbrNgeNFQ7s2EyMFAbts10FBV9Qwt-Y`)
- Fang, Liu, Cheng, Tseng, "Overview of 2024 CAD Contest at ICCAD," ICCAD 2024, DOI 10.1145/3676536.3689908
- ISPD 2006 官方 slides(benchmark 全表):http://www.ispd.cc/slides/2006/7-3.pdf
- ISPD 2015 benchmark description:http://www.ispd.cc/contests/15/web/benchmarks/ispd_2015_contest_benchmark_description.pdf
- ISPD 2018/2019 contest 頁(benchmark 表):http://www.ispd.cc/contests/18/、http://www.ispd.cc/contests/19/
- ISPD 2024/2025 global routing contest:https://github.com/liangrj2014/ISPD24_contest、https://github.com/liangrj2014/ISPD25_contest、https://www.ispd.cc/contests/25/
- MLCAD 2023:https://github.com/TILOS-AI-Institute/MLCAD-2023-FPGA-Macro-Placement-Contest;官方論文 https://vlsicad.ucsd.edu/Publications/Conferences/398/c398.pdf

**論文(擇要)**
- Lu, Lin, Huang, Wu, Lin, "Efficient Chip-Level Global Router: ICCAD 2024 CAD Contest Problem D," VLSI-TSA 2025(Semantic Scholar f7e1713d…)
- Nam et al., ISPD 2005 contest, DOI 10.1145/1055137.1055182;Nam, ISPD 2006 contest, DOI 10.1145/1123008.1123042
- Viswanathan et al., DAC 2012 contest, DOI 10.1145/2228360.2228500
- Bustany, Chinnery, Shinnerl, Yutsis, ISPD 2015 contest, DOI 10.1145/2717764.2723572
- Landman & Russo, IEEE TC-20(12), 1971;Christie & Stroobandt, IEEE TVLSI 8(6), 2000
- Darnauer & Dai, FPGA 1996;Hutton et al., TCAD 1998;Stroobandt et al., TCAD 2000(GNL);Kundarewich & Rose, TCAD 2004
- Kim, Kwon, Lee, Kim, Woo, Kang, DATE 2021(ANG);Kim, Lee, Min, Kang, TCAD 42(6) 2023;ANG repo:https://github.com/daeyeon22/artificial_netlist_generator
- Kang, ISPD 2025 invited, DOI 10.1145/3698364.3709121;Fengler et al., ISCAS 2025, DOI 10.1109/ISCAS56072.2025.11043199
- Kahng, Kang, Park, Yoon, "ArtNet," arXiv 2510.13582;repo https://github.com/shypark98/ArtNet
- Hong et al., "Channel Based SOC Feedthrough Insertion Methodology," IEEE 2022(Xplore 9825309)
- Li, Tian, Zhai, Li, Kai, Xu, Yu, Zhao, "FTAFP," ASP-DAC 2025, DOI 10.1145/3658617.3697728(slides:https://www.aspdac.com/aspdac2025/archive/pdf/6B-2.pdf)
- Xu, Wang, Xu, Geng, Yuan, Wu, "Flora," arXiv 2507.14914
- "GrandPlan," ISPD 2026, DOI 10.1145/3764386.3779591
- Intel Labs, "FloorSet – a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs," ICCAD 2024, DOI 10.1145/3676536.3676814, arXiv 2405.05480, https://github.com/IntelLabs/FloorSet(作者名單未逐一核對)
- Tseng(MediaTek), ISPD 2024 invited, DOI 10.1145/3626184.3639695
- TILOS MacroPlacement:https://github.com/TILOS-AI-Institute/MacroPlacement;MegaBoom:https://github.com/The-OpenROAD-Project/megaboom;HighTide:arXiv 2606.04126;Basilisk:arXiv 2405.04257
