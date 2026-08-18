# Stage 2(Phase 2)規劃:commercial-tool ground-truth 校準

- 日期:2026-08-13
- 狀態:**v1 草案(未定稿)**——待 Codex 對抗性審查後修訂為 v2;Innovus 相關段落在 license 打通前一律為**假設**
- 使用者正式指示來源:`docs/research/2026-08-13-routing-aware-evaluator-directions.md` §「方向 B」(L42–L63)
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §1(Phase 2 定位:router-based 精確 evaluator)、§6(fast evaluator)、§7(benchmark)、§9(M4/M5)、§10 風險 7、§11 非目標
- 相依草案:`docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`(方向 A / RG 模型;本文的三值對照直接消費其 `io_rg`/`ft_rg` 欄位)
- 口徑基準:`docs/results/m2-differentiable-io-report.md`、`ioplace/evaluator_ref.py`、`ioplace/evaluator_gpu.py`
- **本文中標記「實測」的每一條環境事實,皆為 2026-08-13 在本機(L4 host)實跑取得,指令與輸出見 §12;標記「假設」的一律未驗證。**

---

## 0. 摘要(草案裁決一覽)

| # | 問題 | 草案裁決(v1) | 信心 |
|---|---|---|---|
| Q1 | 校準語料 | **主語料 = ISPD2015 LEF/DEF(DREAMPlace 原生 config 已存在);規模語料 = ISPD2025 NanGate45(mempool_tile_wrap → group → cluster)。ISPD2005 bookshelf 不直接校準**——DREAMPlace 的 DEF writer 是「輸入 DEF 逐行 passthrough」,bookshelf 流程根本產不出可繞線 DEF(實測,§3.3)。代價是 M0–M3 的所有數字都在 ISPD2005 上,校準必須靠**轉移假設**,故 C5 轉移檢驗列為硬性驗收 | 高 |
| Q2 | region 如何進 DEF | **不進 DEF。**region 幾何只留在 sidecar `regions.json`(`RegionSet.to_json`,`ioplace/regions.py:45`),route 完再用幾何過濾抽 crossing(Option R1)。理由:(a) DEF `GROUPS` 只能用階層 wildcard(實測 ISPD2015 用 `- er0 h0c/* h0a/*`,`mgc_pci_bridge32_a/floorplan.def:101059`),我們是**幾何歸屬**,只能逐 instance 列名 ⇒ 10M cell 會產生 GB 級 GROUPS section;(b) D1 明定「cell 落點決定歸屬」,DP 把 cell 推過界**就是真的換 region**,fence 反而遮蔽這個效應。**小規模對照臂 R2**(≤200k cells)才寫 `REGIONS + TYPE FENCE`,用來量「DP 跨界搬移」的上界 | 高 |
| Q3 | router | **雙軌:F-OR(OpenROAD,今天就能跑)+ F-INV(Innovus,license 通了才跑)。OpenROAD 是演練與第二 router 對照,不是 commercial ground truth 的替代品**;報表一律分欄 `route_or` / `route_inv`。實測:本機有 Innovus 21.19.000 binary,但**三台 license server 的 5280 port 全部 `No route to host`**(§3.1) | 高(license 不通為實測) |
| Q4 | routed DEF 怎麼解析 | **主解析器 = OpenROAD 的 `odb.dbWireDecoder`(`openroad -python`),不自己寫 DEF 文字剖析器。**它讀得懂 Innovus 寫出的 DEF(實測:ISPD2025 的 DEF 就是 Innovus 23.34 `saveDesign` 產物,odb 讀得動),且已含 `PATH/POINT/POINT_EXT/VIA/TECH_VIA/RECT/SHORT/**VWIRE**/JUNCTION` opcode(實測)。輸出 per-net segment 表 `.npz`,再由 `$DP/.venv312` 端消費。自寫文字剖析器只作 1,000-net 抽樣交叉驗證 | 高 |
| Q5 | crossing 怎麼數才對得上 evaluator | **重用 `evaluator_ref._walk_segment`(`ioplace/evaluator_ref.py:47-66`)這把同一把尺**,只是餵真實 wire 而非 MST L 形。同時報 `route_cross_raw`(逐 lattice transition)與 `route_cross_dw(δ)`(run-length < δ 的短暫越界濾掉,δ∈{0,1,2,4}),因為 router 沿界線走會製造大量假 crossing——**用 δ 掃描把這個效應量化,不用猜** | 中高 |
| Q6 | 校準怎麼回饋 | 三分量非負最小平方 `route_e ≈ α(λ_e−1) + β(ST_e−(λ_e−1)) + γ(io_mst_e−ST_e)`,三個係數正好對應「必要 crossing / feed-through / MST 繞路」。回饋三處:(a) evaluator 主報表尺的選擇(C1);(b) M3 的 `κ_ft` 先驗改為 `β̂/α̂`(C3);(c) Phase 1 §9 exit 判準在校準後量上的 **sign-invariance 檢查**(C4) | 中 |
| Q7 | 不需 Innovus 就能先做的 | **S0–S5 全部**(環境固化、DEF 輸出、routed-DEF 解析、crossing 抽取、OpenROAD 端到端演練、NanGate45 語料接入)。license 是 S6 之後才擋 | 高 |
| Q8 | Stage 2 的 exit | E1 ≥12 個 routed DEF 且全部通過「instance/net 集合不變」assert;E2 最佳模型 vs route 的 per-net Spearman ≥ 0.70;E3 總量比 ∈ [0.80, 1.25] 或給出迴歸殘差;E4 三值分解表 + per-bucket 表;E5 校準前後 M2-best vs flat 的方向一致 | 中 |

**一句話結論:** 使用者要的 ground truth 鏈路在本機**軟體上完全打得通、license 上完全打不通**——Innovus 21.19 binary 在 `/usr/cad/cadence/INNOVUS/`,但 `5280@lshc/lstc/lstn` 三台全部 `No route to host`;同時 OpenROAD v2.0 完整可用(GR + DR + odb wire decoder 全部實測通過,ISPD2025/NanGate45 的 detailed route 正在本機跑)。因此本規劃把「校準流程的所有軟體件」與「Innovus 執行」徹底解耦:S0–S5 用 OpenROAD 把整條鏈路做完做熟,Innovus 一通就只是換一個 router 的 Tcl 腳本。

---

## 1. 目標與範圍

### 1.1 目標(校準,不是取代)

Phase 1 §6 的最後一句寫死了立場:「IO/FT 是拓撲計數而非長度……系統性誤差(congestion 繞行)由 Phase 2 router-based evaluator 校正」;§10 風險 7 也寫「對外主張限定『梯度方向與相對排序』;絕對值準確性由 Phase 2 router evaluator 校正」。Stage 2 就是來兌現這張支票。

明確地說,**本階段要產出的是一個「量測誤差模型」,不是一個新的 in-loop evaluator**:

- `mst_crossing_eval` 仍然是 GP 內圈唯一的 evaluator(Phase 1 §6 的 <1s@30M 預算不變);router 只在**最終 placement** 上跑一次,per-arm 一次。
- 校準產物是三個係數 `(α̂, β̂, γ̂)` + per-bucket 修正表 + 一份「哪一把尺最貼近真實 router」的裁定,回頭修**報表口徑、objective 權重先驗、exit 判準的絕對門檻**——不動 GP 內圈的計算。
- 因此本階段**不承諾**任何 placement 品質改善;它承諾的是**所有既有數字的可信度**。

### 1.2 與方向 A(RG 模型)的關係:三值對照的意義

方向 A 已經把「evaluator 該量什麼」拆成兩把尺(M3 draft §2.2):`io_mst`(MST 幾何 + L 形 walk)與 `io_rg = Σ_e ST_e`(region-adjacency Steiner,crossing-minimal)。Stage 2 加入第三個值 `route`(真實繞線)。三個差額各自有明確、互不重疊的意義:

| 差額 | 意義 | 若它很大代表什麼 |
|---|---|---|
| `io_mst − io_rg`(= `mst_excess`,M3 draft 已量測:adaptec1 k16 flat 3,335 / M2 1,927;bigblue4 k16 flat 12,380 / M2 1,166) | **MST 幾何相對 crossing-minimal 的超額** | evaluator 因為用 MST 而系統性高估;且該高估**隨 placement 幾何劇烈變動**(bigblue4 flat→M2 掉了 90%),會污染跨臂比較 |
| `route − io_rg` | **真實 router 相對 crossing-minimal 拓撲的超額**(congestion 繞行 + router 不優化 crossing) | RG 模型太樂觀;`ft_rg` 只是下界這件事(M3 draft L1)有多嚴重 |
| `io_mst − route`(有號) | **evaluator 的淨偏差** | 正 ⇒ 我們高估(Phase 1 §10 風險 7 的假設成立);負 ⇒ 我們低估,對外主張要改寫 |

以及 `(λ−1)` 這個 routing-independent 下界(Phase 1 §5.2 / D4):`route − (λ−1)` 是「無論如何都省不掉的部分之外,router 實際多付了多少」。**四值 `(λ−1) ≤ io_rg ≤ ?route? ≤/≥ io_mst` 的實測排序本身就是一個可發表的結果**——目前沒有任何一篇文獻報過這條鏈。

方向 A 的 formulation 選擇(M3 Q1)因此在 Stage 2 得到**事後裁決**:若實測 `ρ(io_rg, route) > ρ(io_mst, route)` 且總量比更接近 1,M3 的 RG 裁決被證實;反之 M3 Q1 需要重審(§9 的 G2)。這是本文與 M3 的唯一硬耦合點。

### 1.3 額外產出(免費副產品,建議升格為論文賣點)

流程一旦跑通,以下三個量幾乎零額外成本:

1. **`ft_route`**:routed wire 穿過但該 net 無 pin 的 region 數 ⇒ pure feed-through 的**真值**。這是本專案主要 novelty 的第一份 ground-truth 驗證。
2. **`route_pair_demand`**:每對邊界的真實 crossing 數 ⇒ 直接校準 M3 §4 的 S7 容量模型 `C_ab = ρ·ℓ_ab`,把 M3 draft L-Q3「絕對容量門檻在 ISPD2005 上無法定錨」這個公認的洞補上。
3. **`Δio_after_commercial_DP`**:commercial DP 之後我們的 IO 改善還剩多少。M2 已知 GP→LG 會流失(報告 §5.1 的 `lg_loss`);GP→LG→commercial DP 的流失率是審稿人一定會問的問題,現在能直接回答。

### 1.4 非目標(Stage 2 不做)

- 不把 router 放進 GP 內圈(違反 Phase 1 §6 的 runtime 預算)。
- 不做 timing/power 校準(Phase 1 §11)。
- 不做 CTS——**明確禁止**,見 §6.3 的 instance-set 不變約束。
- 不用校準結果反推「更好的 placement」;那是 M4/M5 的事。

---

## 2. 名詞與欄位(Phase 1 §2、M3 draft §1 之外新增)

| 符號 / 欄位 | 意義 | 來源 |
|---|---|---|
| `route_cross_raw[e]` | net e 的 routed wire 沿 region lattice 的 region-id 轉換總數 | 新增,§7 |
| `route_cross_dw[e](δ)` | 同上,但 region-id run-length < δ 的短暫越界先被 run-length filter 吃掉 | 新增,§7.4 |
| `Λ_route[e]` | routed wire 造訪過的 distinct region 數 | 新增 |
| `route_ft[e]` | `|Λ_route(e) 的 region 集合 \ pin-region 集合|`(語意逐字對齊 `evaluator_ref.py:113`) | 新增 |
| `route_pair_demand` | `{(min(a,b), max(a,b)): count}`,語意逐字對齊 `EvalResult.boundary_pair_demand` | 新增 |
| `route_wl[e]` | routed wire 總長(不含 via) | 新增,用於 §10 S2 的驗收 |
| `α̂, β̂, γ̂` | 三分量校準係數(§8.3) | 新增 |
| `F-OR` / `F-INV` | OpenROAD flow / Innovus flow | — |
| `R1` / `R2` | 幾何過濾(不寫 DEF region)/ fence 標註(寫 DEF REGIONS+GROUPS) | §5 |

**硬性分工(延續 M2 §1、M3 §1 的慣例):** `evaluator_ref` / `evaluator_gpu` 的既有欄位語意與數值,Stage 2 **不得改動一個位元**。所有 route 側的量都放在新模組 `ioplace/route_eval/` 的獨立資料結構裡,只在分析腳本裡與 `EvalResult` 併表。

---

## 3. 前置條件盤點(全部本機實測)

### 3.1 EDA 工具

| 項目 | 狀態 | 證據 |
|---|---|---|
| Innovus binary | **在**,21.19.000 | `/usr/cad/cadence/INNOVUS/INNOVUS_21.19.000`(`/usr/cad` → symlink 到 `/nashome/CAD`);`/usr/cad/cadence/INNOVUS/cur/tools/bin/innovus` → `../innovus/bin/innovus` |
| Innovus 環境腳本 | **在** | `/nashome/CAD/cadence/CIC/innovus.cshrc`(tcsh only,需 `source`) |
| Innovus license | **不通(實測)** | `license.cshrc` 設 `LM_LICENSE_FILE=5280@lshc:5280@lstc:5280@lstn`;三台 DNS 都解得出(140.126.24.16 / 140.110.140.29 / 140.110.127.149),但 TCP 5280 全部 `[Errno 113] No route to host` |
| OpenROAD | **在且可用** | `/usr/bin/openroad`,`v2.0-17598-ga008522d8`,`+Charts +GPU +GUI +Python`;`read_lef`/`read_def`/`global_route`/`detailed_route`/`write_def`/`set_thread_count` 全部存在 |
| OpenROAD Python + odb | **在** | `openroad -python` 可 `import openroad, odb`;`odb.dbWireDecoder` 具備 `PATH/POINT/POINT_EXT/VIA/TECH_VIA/RECT/SHORT/VWIRE/JUNCTION/END_DECODE` |

**Innovus 環境檢查清單(本機無法驗證,逐項打勾制;每一項都是 §11 的 A-級假設):**

| # | 檢查項 | 通過判準 |
|---|---|---|
| I1 | license server 可達 | `nc -z lshc 5280` 或 `lmstat -a` 回應;三台任一即可 |
| I2 | Innovus 可啟動 | `source /usr/cad/cadence/CIC/innovus.cshrc; innovus -version` 回版本號且不報 license error |
| I3 | 授權含 route feature | 啟動後 `routeDesign`/`route_design` 不因 feature 缺失被拒(NanoRoute 常為獨立 feature) |
| I4 | **無 .lib 能否 `init_design`** | ISPD2015 **沒有任何 `.lib`**(實測:各 design 目錄只有 `tech.lef`/`cells.lef`/`design.v`/`floorplan.def`/`placement.constraints`)。需驗證 Innovus 21.19 能否在無 MMMC 下 init;不能則此語料只能走 F-OR,或改走有 .lib 的 NanGate45 |
| I5 | 並行度與記憶體 | `setMultiCpuUsage -localCpu N`;1.29M(superblue12)與 10M(mempool_cluster)的 route peak memory 需實測 |
| I6 | DEF 5.7/5.8 相容 | ISPD2015 是 DEF 5.7,ISPD2025 是 Innovus 23.34 寫的 5.8;21.19 讀 23.34 的 DEF 需驗證(向下相容通常沒問題,但**未驗證**) |
| I7 | `defOut -routing` 輸出完整 | 輸出 DEF 的 `NETS` section 含 `+ ROUTED`,且 `COMPONENTS`/`NETS` 數與輸入相同 |
| I8 | batch 可跑、無 GUI | `innovus -no_gui -batch -files flow.tcl` |

### 3.2 Benchmark 盤點

**(a) ISPD2015 LEF/DEF —— 本機齊備、DREAMPlace 原生支援(實測)**

`$DP/benchmarks/ispd2015/` 下 20 個 design,每個含 `tech.lef` / `cells.lef` / `design.v` / `floorplan.def`(未擺置)/ `after_legalized.ntup.fix.def`(已擺置)/ `placement.constraints`;`$DP/install/test/ispd2015/lefdef/*.json` 已有 20 份 DREAMPlace config,且都設 `"sol_file_format": "DEF"`。

| design | COMPONENTS | 有 DEF REGIONS/GROUPS |
|---|---:|---|
| mgc_pci_bridge32_a / _b | 29,521 / 28,920 | **是 / 是** |
| mgc_fft_1 / _2 / _a / _b | 32,281 / 32,281 / 30,631 / 30,631 | 否 |
| mgc_des_perf_1 / _a / _b | 112,644 / 108,292 / 112,644 | 否 / **是** / **是** |
| mgc_edit_dist_a | 127,419 | **是** |
| mgc_matrix_mult_1/_2/_a/_b/_c | 155,325 / 155,325 / 149,655 / 146,442 / 146,442 | 否/否/否/**是**/**是** |
| mgc_superblue19 | 506,383 | 否 |
| mgc_superblue14 | 612,583 | 否 |
| mgc_superblue16_a | 680,869 | **是** |
| mgc_superblue11_a | 927,074 | **是** |
| **mgc_superblue12** | **1,287,037** | 否 |

- 實測 DREAMPlace 讀 `mgc_fft_1`:35,291 physical / 32,281 movable / 3,010 terminal_NI / **33,307 nets**(= DEF 的 `NETS 33307`),`net_names[0] = b'n_9999'`;`sort_nets_by_degree = 0` 且 degree 序列**非單調** ⇒ net index 就是 parser 順序,穩定可重現。
- **`ignore_net_degree` 只是 WL objective 的 mask**(`$DP/dreamplace/BasicPlace.py:163`),**不會從 netlist 移除 net** ⇒ net index ↔ net name 是全滿的雙射,§7.2 的 key 對齊沒有隱藏黑洞。
- **限制 1:沒有 `.lib`** ⇒ Innovus timing-driven flow 不可能(見 I4)。
- **限制 2(實測):ISPD2015 tech.lef 過不了 TritonRoute** —— `detailed_route` 直接 `[ERROR DRT-0338] Duplicated via definition for VIA23_2cut_E`。`global_route` 則可跑(需 `-allow_congestion`,否則 `[ERROR GRT-0116] finished with congestion`),並寫出 per-net、按 net name 索引、含 rect + layer 的 guide 檔(mgc_fft_1 為 5.1MB)。
- **限制 3:metal 層數不一** —— `mgc_pci_bridge32_a/tech.lef` 只有 metal1–metal5;`mgc_superblue12/tech.lef` 有 metal1–metal9。校準表必須分開報,不可混池。

**(b) ISPD2025 NanGate45 —— 本機齊備,是 Phase 1 §7 的 10M 黃金主力(實測)**

`/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/`,`visible/` 與 `blind/` 各 6 個 design,共用 `NanGate45/{lef(15), lib(15), qrc}`;每個 design 含 `.def` / `.v.gz` / `.sdc` / `.cap` / `.net`。

| design | DEF 大小(visible) | 備註 |
|---|---:|---|
| mempool_tile_wrap | 85 MB | 127,759 comps / 145,589 nets;**Stage 2 的首選演練 case** |
| ariane | 69 MB | |
| NV_NVDLA_partition_c | 119 MB | |
| bsg_chip | 573 MB | |
| mempool_group | 2.3 GB | 5M 級 |
| **mempool_cluster** | **9.7 GB** | Phase 1 §7 的 10M 黃金主力 |

- DEF header 實測:`Generated by: Cadence Innovus 23.34-e024_1` / `saveDesign` / `flow_implementation_stage = place_opt` / `flow_timed_design_state = preCts` ⇒ **這批測資本身就是 Innovus 產物**,格式相容性風險最低;signal net 未繞線(mempool_tile_wrap 的 145,589 nets 裡只有 4,360 個 `+ ROUTED`,是 clock/pre-route),正是我們要 router 去繞的狀態。
- 10 層 metal(TRACKS 到 metal10),有 `.lib` ⇒ **Innovus MMMC 可正常建**,I4 的風險在此語料上消失。
- **實測:DREAMPlace 讀得動** —— NanGate45 全 15 個 LEF + `mempool_tile_wrap.def`,**3.1 秒**,得到 127,739 movable / 145,589 nets / die (20140,19880)–(1779920,1778280) / row_height 2800。
- **⚠ 實測踩雷,必須寫進 config 產生器**:若照 contest 自帶的 `lib_setup.tcl` 只給 10 個 LEF 的**子集**,`place_io` 會斷言失敗並 core dump——`PlaceDB.cpp:1469 ... failed to find pin clk in macro AND2_X1`。**必須 glob 全部 15 個 LEF,且 tech.lef 排在最前。**
- **實測:OpenROAD 在此語料上 GR + DR 都動得起來** —— `global_route -allow_congestion` 完成並寫出 39MB guide;`detailed_route` 通過 tech/LEF 檢查、走完 metal1–metal10 的初始化、進入 `[INFO DRT-0195] Start 0th optimization iteration.` 且 `Completing 10% with 0 violations`(本次盤點 time-box 內尚未跑完全程,**「DR 能完整收斂」目前是待驗證項,不是已驗證事實**)。

**(c) ISPD2005 bookshelf —— 不作為校準語料(裁決,理由見 §4.2)**

`$DP/benchmarks/ispd2005/{adaptec1-4, bigblue1-4}`,只有 `.aux/.nodes/.nets/.pl/.scl/.wts`,無 LEF、無 metal stack。M0–M3 的所有實驗都在這上面。

### 3.3 DREAMPlace 的 DEF 讀寫能力(實測 + 讀碼)

- **讀**:LEF/DEF 路徑可用(§3.2 的兩個實測)。`PlaceDB.read()` 之後 `placedb.net_names` / `node_names`(bytes 陣列)可用,index 與 evaluator 的 net index 完全一致。
- **寫**:`PlaceDB.write(params, filename, DEF)`(`$DP/dreamplace/PlaceDB.py:1010-1033`)→ `place_io.PlaceIOFunction.write(rawdb, ..., SolutionFileFormat.DEF, ...)` → `PlaceDB::write`(`ops/place_io/src/PlaceDB.cpp:1368-1379`)→ `DefWriter(*this).write(filename, **userParam().defInput**, ...)`。
- **關鍵性質(讀 `ops/place_io/src/DefWriter.cpp:12-83`):DEF writer 是「輸入 DEF 逐行 passthrough」** —— 它開啟 `defInput`,一行一行複製,只做三件事:(1) 把 `COMPONENTS…END COMPONENTS` 整段換成新座標;(2) 用 db 的 row 資料重寫 `ROW` 行;(3) 其餘**全部原樣輸出**(包含 `REGIONS`/`GROUPS`/`NETS`/`TRACKS`/`VIAS`/`PINS`)。三個直接推論:
  1. **DEF 輸出必須有 DEF 輸入** ⇒ bookshelf(ISPD2005)無論如何產不出可繞線 DEF。
  2. `DEFSIMPLE`(`DefWriter.cpp:84-104`)只吐 `VERSION` + `DESIGN` + `COMPONENTS`,**不可繞線**,不能用。
  3. 若把 `REGIONS`/`GROUPS` 注入**輸入** DEF,會原封不動出現在**輸出** DEF ⇒ §5 的 R2 對照臂在實作上是「改輸入 DEF」,不是「改 writer」。
- **座標系**:`write()` 內先 `unscale_pl(params.shift_factor, params.scale_factor)`(`PlaceDB.py:1030`、`135-147`)。反向映射(解析 DEF 回 evaluator 內部座標)為 `x_internal = (x_def − shift_factor[0]) × scale_factor`(對照 `scale_pl`,`PlaceDB.py:124-133`)。ISPD2015 config 的 `scale_factor` 是 `0.0`(= 未設,由 place_io 自訂),**因此不可硬編 1.0**,必須從 `params` 讀並在 S1 的驗收裡做 round-trip assert。
- **fence region 注入點**:`ioplace/fence_inject.py:6-34` 已有「`read()` 之後、`initialize()` 之前」的注入窗口與 `placedb.regions` / `flat_region_boxes` / `node2fence_region_map` 四欄位的填法——R2 對照臂可直接複用,不需新設計。

### 3.4 OpenROAD 側能力(實測)

- 讀 ISPD2015 LEF/DEF:OK(mgc_fft_1:9 layers / 331 lib cells / 33,307 nets / 32,281 insts)。
- **DEF `REGIONS`/`GROUPS` 完整往返**:讀 `mgc_pci_bridge32_a/floorplan.def` → `getRegions()` 得 4 個 region(`TYPE EXCLUSIVE`,1–5 個 boundary rect)、`getGroups()` 得 4 個 group(3,922 / 1,552 / 3,530 / 118 insts);`write_def` 再吐出 `REGIONS 4 ; … + TYPE FENCE ;`。⇒ R2 對照臂的傳輸介質已驗證。
- `global_route` 產 guide(按 net name 索引,rect + layer)、`detailed_route` 產 routed DEF、`odb.dbWireDecoder` 可逐 net 解 wire path。

---

## 4. Q1 — 校準語料選擇

### 4.1 定案

| 層級 | 語料 | K | 用途 |
|---|---|---|---|
| **L0 演練** | ISPD2015 `mgc_fft_1`(32k)或 `mgc_pci_bridge32_a`(30k,自帶 REGIONS) | 8 | 把 S1–S4 的軟體跑通;**不進校準統計** |
| **L1 主校準** | ISPD2015 `mgc_des_perf_1`(113k)、`mgc_matrix_mult_1`(155k)、`mgc_superblue19`(506k)、**`mgc_superblue12`(1.29M)** | 16, 32 | 校準係數的主要來源;涵蓋 metal5 與 metal9 兩種 stack |
| **L2 規模/平台交叉** | ISPD2025 `mempool_tile_wrap`(128k)、`mempool_group`(5M)、`mempool_cluster`(10M) | 16, 32 | 驗證校準係數在**不同 PDK / 10 層 metal / 10M 規模**下是否成立(C5 轉移檢驗的主力) |
| **不校準** | ISPD2005 adaptec/bigblue | — | 只作為「被校準的對象」;校準係數靠轉移套用,並標記為外插 |

每個 case 至少跑兩臂:`flat`(IO-oblivious baseline)與 `ours`(M2/M3 best arm),因為**校準係數本身可能依 placement 而變**——若 flat 與 ours 的 `(α̂,β̂,γ̂)` 差很多,代表校準不是設計常數而是 placement 的函數,那 §8.4 的回饋方式要整個改寫(G3)。

### 4.2 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| **在 ISPD2005 上直接校準(bookshelf → 合成 LEF/DEF)** | (a) DEF writer 需要輸入 DEF(§3.3 實測),要從零合成 DIEAREA/ROW/TRACKS/PINS/NETS 全套;(b) bookshelf 的 pin offset 是**逐 instance**給的,同尺寸 cell 的 pin 位置不同 ⇒ 要為**每個 node** 造一個 LEF MACRO(adaptec1 = 211,447 個),LEF 檔會爆炸且無任何 design rule;(c) ISPD2005 沒有 metal stack、row 不在真實 track grid 上,繞出來的東西不是「真實 router 的行為」,是我們自己編的 PDK 的行為 ⇒ **ground truth 的信度歸零,失去整件事的意義** |
| **只用 ISPD2025(放棄 ISPD2015)** | (a) `mempool_cluster.def` 9.7GB 的 parse/記憶體完全未驗證,把校準全押在它上面是單點風險;(b) ISPD2015 有 DREAMPlace 原生 config 與 20 個 design 的多樣性,能撐起 C5 的係數散布統計;(c) ISPD2015 的 superblue12 是 Phase 1 §7 明列的 1M 級主力,不能跳過 |
| **只用 ISPD2015(放棄 ISPD2025)** | (a) ISPD2015 **無 `.lib`** ⇒ Innovus 能否 init 完全未知(I4);(b) DR 在 OpenROAD 上被 `DRT-0338` 擋住(實測),需先修 LEF;(c) 沒有 10M case ⇒ 校準無法覆蓋本專案的目標規模 |
| **自己合成一套 30M LEF/DEF 來校準** | Phase 1 §10 風險 2 已把 30M 測資列為工程成本項;在校準這件事上再加一層自製 PDK,等於用「我們自己的假設」去校準「我們自己的 evaluator」,循環論證 |

### 4.3 低信心處

- **L-Q1-a:ISPD2005 → ISPD2015/2025 的校準轉移**。M0–M3 的所有結論都在 ISPD2005 上,而校準只能在 LEF/DEF 語料上做。這是本規劃最大的方法論裂縫。**緩解**:C5 轉移檢驗(§8.5)若失敗,結論是「Phase 1 的主 benchmark 必須搬到 LEF/DEF」,而不是「校準失敗」——這個結論本身就有價值,但代價是 M4 要重跑實驗矩陣。**必須在 v2 前讓使用者知情並確認可接受。**
- **L-Q1-b:mempool_cluster 9.7GB DEF 的可讀性**未驗證(mempool_tile_wrap 的 85MB 用 3.1s,線性外推約 6 分鐘與 ~100GB+ 記憶體,**線性外推極可能不成立**)。S5 的驗收就是把這個問號變成事實。

---

## 5. Q2 — DEF 輸出與 region 標註

### 5.1 定案:R1(幾何過濾),R2 只作小規模對照臂

**R1(主線)**:placer 輸出

```
<run>/out.def          ← placedb.write(params, path, DEF);COMPONENTS 為我們的 GP+LG 結果
<run>/regions.json     ← RegionSet.to_json(ioplace/regions.py:45),die + lattice + K 個 rect 集合
<run>/netmap.json      ← {net_index: net_name},由 placedb.net_names 產出(§3.2 已證雙射)
<run>/coord.json       ← {shift_factor, scale_factor, def_units_per_micron, xl,yl,xh,yh}
```

DEF 裡**沒有任何 region 資訊**。router 看到的就是一個普通的 flat design。crossing 由 §7 的解析器事後用 `regions.json` 幾何過濾算出。

**R2(對照臂,只在 ≤200k cells 的 case)**:把 `REGIONS`(`+ TYPE FENCE`)與 `GROUPS`(逐 instance 列名)注入**輸入** DEF(利用 §3.3 的 passthrough 性質),讓 DP 尊重 fence。用途:量「commercial DP 把多少 cell 推過 region 邊界」的上界。

### 5.2 為什麼是 R1

1. **DEF `GROUPS` 的表達能力不匹配**(實測):ISPD2015 的 GROUPS 用階層 wildcard(`- er0 h0c/* h0a/* h0b/* h0/*`,`mgc_pci_bridge32_a/floorplan.def:101059`)。我們的歸屬是**幾何**的(Phase 1 D1),與 RTL 階層無關,只能逐 instance 列名 ⇒ superblue12(1.29M)的 GROUPS section 約數十 MB,mempool_cluster(10M)約數百 MB,寫得出來但沒有必要。
2. **語意上 R1 才是對的**:D1 說「cell 落點決定歸屬」。若 commercial DP 把 cell 從 region a 推到 region b,那它**就是**變成了 region b 的 cell,IO 數也**就該**跟著變。用 fence 鎖住反而是在偽造一個 placement 從未達到的狀態。
3. **少一個 Innovus 語意假設**:`createInstGroup` / `setInstGroupSoftGuide` / fence 的確切 DP 行為都是 A-級假設;R1 一個都不需要。
4. R2 因此降級為「量測 DP 搬移效應」的診斷臂,而不是主流程。

### 5.3 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| **Innovus `createInstGroup` + `addInstToInstGroup` 走 Tcl** | 與 R2 同樣的規模問題,而且把 region 資訊綁進 Tcl(工具相依),F-OR 無法共用同一份輸入 |
| **`setInstGroupSoftGuide`(soft guide 而非 fence)** | 「軟」的程度是 Innovus 內部啟發式,無法量化 ⇒ 校準會混入一個不可控變因 |
| **用 DEF `BLOCKAGES` 沿邊界擺 placement blockage 模擬 region** | 會同時改變 routing 資源分布,污染 crossing 的量測 |
| **完全不輸出 DEF,改用 guide 檔比對** | GR guide 是 GCell 粒度的矩形(實測 mgc_fft_1 的 guide 格式),一個橫跨邊界的 guide rect **不等於**一次 crossing ⇒ 只能當粗略演練(§10 S4 的中間驗收),不能當 ground truth |

### 5.4 低信心處

- **L-Q2-a:`out.def` 能否被 Innovus 21.19 直接 `defIn`**。DEF writer 是 passthrough,所以輸出 DEF 的格式版本 = 輸入 DEF 的版本(ISPD2015 = 5.7,ISPD2025 = 5.8);風險在於 writer 重寫 `ROW` 行的格式(`DefWriter.cpp:63-73`)是否 100% 合法。**S1 的驗收用 OpenROAD `read_def` 把關;Innovus 端只能等 I6。**

---

## 6. Q3 — Router flow 骨架

### 6.1 F-OR(OpenROAD,今天就能跑)

```tcl
# ---- 讀入(tech.lef 必須最先;ISPD2025 必須 glob 全部 15 個 LEF,見 §3.2 踩雷) ----
foreach f $TECH_LEFS { read_lef $f }
foreach f $CELL_LEFS { read_lef $f }
read_def  $OUR_DEF                      ; # placer 輸出的 out.def

# ---- 不變式:instance / net 集合必須與輸入相同 ----
set block [[[ord::get_db] getChip] getBlock]
puts "ASSERT insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"

set_thread_count 16
# ---- 不做 CTS、不做 optimization、不插 buffer ----
global_route -allow_congestion \
             -congestion_report_file $OUT/congestion.rpt \
             -guide_file $OUT/route.guide
detailed_route -output_drc $OUT/drc.rpt -verbose 1
write_def $OUT/routed.def
```

- `-allow_congestion` 是**必要**的(實測:ISPD2015 mgc_fft_1 不加會 `[ERROR GRT-0116]` 直接中止)。但這代表**繞不完的 net 會留下 congestion**,§11 的 R-5 要處理選擇偏差。
- OpenROAD 不做 detailed placement;F-OR 因此量到的是「我們的 GP+LG 直接進 router」的結果——與 F-INV(含 DP)不同,兩者要分欄報,不可混。若要對齊,可在 F-OR 前先 `detailed_placement`(OpenROAD 的 dpl)作為第三臂。

### 6.2 F-INV(Innovus,骨架;**全部為假設**)

```tcl
# ---- init(ISPD2015 無 .lib ⇒ I4;NanGate45 有 .lib ⇒ 走 MMMC) ----
set init_lef_file  "$TECH_LEF $CELL_LEFS"
set init_def_file  $OUR_DEF
set init_mmmc_file $MMMC          ;# NanGate45:用 contest 的 lib_setup.tcl 產;ISPD2015:見 I4
init_design
setMultiCpuUsage -localCpu 16

# ---- 硬性:不得改變 instance / net 集合 ----
#   不跑 CTS、不跑 optDesign / place_opt_design、不做 scan reorder、不插 buffer
setPlaceMode  -place_global_place_io_pins false
setDesignMode -topRoutingLayer $TOP_LAYER

# ---- (1) detailed placement 只做 legalization/DP,不做 optimization ----
place_detail

# ---- (2) route:關掉 timing/SI driven,讓 crossing 反映純幾何/congestion ----
setNanoRouteMode -routeWithTimingDriven false
setNanoRouteMode -routeWithSiDriven false
routeDesign

# ---- (3) 輸出 ----
defOut -routing $OUT/routed.def
verify_connectivity -type all -report $OUT/conn.rpt
report_route            > $OUT/route.rpt
summaryReport -noHtml -outfile $OUT/summary.rpt
```

**需要的報表**:`routed.def`(唯一必需)、`conn.rpt`(open/short 為 0 才算有效樣本)、`route.rpt`(total wirelength / via count,用於 §10 S2 的交叉驗收)、DRC 數(記錄,不作 gate)。

### 6.3 兩條 flow 共同的硬性不變式(解析器必須 assert)

| # | 不變式 | 檢查 |
|---|---|---|
| V1 | `#COMPONENTS(routed.def) == #COMPONENTS(out.def)` | CTS / buffer insertion 一旦發生就會破 |
| V2 | `#NETS(routed.def) == #NETS(out.def)` 且 net name 集合完全相同 | 同上 |
| V3 | 每個 instance 的 master(cell type)不變 | 防 sizing |
| V4 | 每條 signal net 都有非空 wire(或明確列入「未繞線」清單) | 未繞線的 net 必須從統計中剔除**並回報比例** |
| V5 | routed.def 的 `UNITS DISTANCE MICRONS` 與 out.def 相同 | 座標換算前提 |

**V1/V2 被破 ⇒ 該樣本作廢,不做「盡量對齊」的補救**——net/instance 集合一變,per-net key 對齊就失去意義,勉強對齊只會產生看起來合理的假數字。

---

## 7. Q4/Q5 — routed DEF 解析與 per-net crossing 抽取

### 7.1 架構

```
routed.def ──[openroad -python]──> segments.npz ──[$DP/.venv312]──> route_metrics.json
             dump_segments.py                      route_crossings.py
```

- 新模組 `ioplace/route_eval/dump_segments.py`:**在 OpenROAD 的 Python 下執行**(不是我們的 venv),用 `odb.dbWireDecoder` 逐 net 解 `net.getWire()`,輸出扁平化的 `.npz`:
  `seg_net_id (int32)`、`seg_x0/seg_y0/seg_x1/seg_y1 (int64, DBU)`、`seg_layer (int8)`。
- 新模組 `ioplace/route_eval/route_crossings.py`:在我們的 venv 下執行,把 DBU 座標映回 evaluator 內部座標,**呼叫 `evaluator_ref._walk_segment`(`ioplace/evaluator_ref.py:47-66`)這把同一把尺**,累出 `route_cross_raw` / `Λ_route` / `route_ft` / `route_pair_demand` / `route_wl`。

**為什麼不自己寫 DEF 文字剖析器(定案理由)**:`odb.dbWireDecoder` 已經處理了 `POINT` 續值(`*`)、`POINT_EXT`(端點延伸)、`TECH_VIA`/`VIA`、`RECT` patch、`SHORT`、`JUNCTION`,以及最容易寫錯的 **`VWIRE`(virtual wire,無金屬,絕對不能算成 wire)**;而且它讀得懂 Innovus 寫出的 DEF(ISPD2025 那批就是 Innovus 23.34 產物,實測 odb 讀得動)。自寫剖析器只作 1,000-net 抽樣的交叉驗證器(S2 的驗收之一)。

**已知 binding 缺陷 + workaround(實作 S2 時發現,非合成 fixture 而是對真實 TritonRoute routed DEF 跑出來的)**:這個 OpenROAD build(v2.0-17598)的 Python binding 下,`dbWireDecoder.getPoint()`/`getRect()` 對 `POINT_EXT`/`RECT` 兩個 opcode 不可達——`getPoint()` 永遠解到 2-int `getPoint(int&,int&)` overload,對 `POINT_EXT` 呼叫會 assert `_opcode == POINT` 而 SIGABRT(非可 catch 的 Python exception);`getRect()` 同樣拿不到可用的 overload。Workaround:兩者都改直接讀 `dbWire` 自己的 data array——`POINT_EXT` 用 `wire.getCoord(decoder.getJunctionId())` 取點、`wire.getData(jid+1)`(經 `wire.getOpcode(jid+1) & 0x0F == WOP_OPERAND` 驗證)取延伸量;`RECT` 用 `wire.getData(jid..jid+3)`(經 `wire.getOpcode(jid) & 0x0F == WOP_RECT` 驗證)取四個相對於目前點的 delta。`getJunctionId()`/`getCoord()` 這組 accessor 已用 `dump_segments.py --selfcheck`(預設開啟)對每個一般 `POINT` opcode 交叉驗證過 `wire.getCoord(getJunctionId()) == decoder.getPoint()`:一份真實 routed DEF 上 845,555/845,555 個 POINT 全部相符,才敢把同一組 accessor 用在 `POINT_EXT`/`RECT` 上。

### 7.2 Key 對齊(與 `per_net_crossings` 同 index)

1. `netmap.json` 給 `net_index → net_name`(來自 `placedb.net_names`)。
2. odb 端逐 net 取 `dbNet.getName()`,反查得 `net_index`;**未命中即 fail**(不做 fuzzy match)。
3. 已知的合法落差,必須逐項列在報表裡:
   - **SPECIALNETS(PG)**:只解析 `NETS`,PG 自然不在 `placedb` 的 net 集合裡 ⇒ 不算落差。
   - **degree ≤ 1 的 net**:`evaluator_ref.py:89-90` 直接 `continue`,`per_net_crossings` 為 0;route 側也應為 0(或極短 stub)⇒ 一律排除於相關性統計之外,但列入總量比的分母說明。
   - **`max_degree`(預設 256)以上的 net**:evaluator 走 region-presence 下界(`evaluator_ref.py:96-100`),**不建樹**。這批 net 的 `io_mst` 語意與其他 net 不同,**必須單獨成一個 bucket 報**,不可混入主相關性。
   - **`ignore_net_degree`**:實測只影響 WL objective 的 mask(`BasicPlace.py:163`),不移除 net ⇒ **不造成 key 落差**。
4. 覆蓋率 assert:`|matched| / |placedb nets with degree ≥ 2| ≥ 0.999`,否則整個樣本作廢。

### 7.3 座標映射

```
x_internal = (x_def − shift_factor[0]) × scale_factor        # 對照 PlaceDB.scale_pl (PlaceDB.py:124-133)
y_internal = (y_def − shift_factor[1]) × scale_factor
```
`shift_factor` / `scale_factor` 從 `coord.json` 讀(**不可硬編 1.0**:ISPD2015 config 的 `scale_factor` 是 `0.0` = 由 place_io 自訂)。S1 的 round-trip 驗收:輸出 DEF 再讀回 COMPONENTS,與 `node_x/node_y` 的差 ≤ 1 DBU。

### 7.4 Crossing 的計數約定(**本節是全文最容易出錯的地方**)

真實 router 的 wire 會沿著、貼著、來回穿越 region 邊界;逐 lattice transition 硬數會製造大量「假 crossing」。定案是**同時報三個量,用 δ 掃描把這個效應量化而不是猜**:

| 量 | 定義 | 對應 evaluator 的哪一把尺 |
|---|---|---|
| `route_cross_raw` | 每條 wire segment 沿 region lattice 走,累計相鄰 lattice cell 的 region-id 變化數(**逐字等同 `_walk_segment` 的 `len(diff_pos)`**) | `io_mst`(同約定) |
| `route_cross_dw(δ)` | 同上,但每條 segment 的 region-id 序列先過 run-length filter:長度 < δ 的 run 併入前一個 run,再數 transition。δ ∈ {0,1,2,4}(δ=0 即 raw) | 掃描用,決定「假 crossing」的量級 |
| `Λ_route − 1` | routed wire 造訪的 distinct region 數 − 1 | `io_rg = ST_e` 的可比下界 |

- **via 不算 crossing**(平面上零長度);只有 wire segment 算。
- **`RECT` patch 忽略**(尺寸遠小於 lattice cell,die/512 級)。
- **`VWIRE` 一律跳過**。
- 恆等式測試(S3 驗收):`route_cross_raw ≥ route_cross_dw(δ) ≥ Λ_route − 1 ≥ 0`,任一違反即為 bug。
- 主報表用 `route_cross_dw(2)`;`raw` 與 δ 掃描表附在報告裡。**若 `raw` 與 `dw(2)` 的總量差超過 15%,代表「沿界線走」是主要效應,§8 的所有結論必須同時用兩個值陳述。**
- **`route_wl` 與 `dbWire::getLength()` 的 reconciliation 恆等式**:`route_wl` 是 centerline WIRE-only 的量(`ioplace/route_eval/segments.py` 的 `Segments.wire_length()`);`RECT` 不進 `route_wl`,也不進上面任何一個 crossing 量。但 odb 自己的 `dbWire::getLength()`(`verify_routed_def.py` check 1 用作 native oracle)**會**把每個 `RECT` patch 算成它的長邊,因此兩者之間有個固定、可算的落差:`dbWire::getLength() = route_wl + Σ_RECT(long_side − short_side)`。S2 的驗收(§10)因此比對的是**reconciled** 後的量,不是 `route_wl` 原始值本身;`ioplace/route_eval/segments.py` 的 `rect_reconciliation_dbu(segments)` 是這個 `Σ_RECT(long_side − short_side)` 項的純 numpy 實作。

### 7.5 分層報表

routed wire 分佈在 metal1–metal9/10;高層 metal 的長線可能一口氣跨多個 region。定案:主表用**全層合計**,附表給 per-layer 的 crossing 分解。這張附表本身有價值——它回答「boundary pin 應該開在哪幾層」,直接餵 Phase 3 的 IO legalization。

---

## 8. Q6 — 校準協定

### 8.1 每個樣本(= 一個 design × 一個 K × 一個臂 × 一個 router)產出的四值表

對每條 degree ≥ 2 且成功繞線的 net e:

| 欄位 | 來源 |
|---|---|
| `lam_minus_1[e] = per_net_lambda[e] − 1` | `EvalResult.per_net_lambda`(**在 router 看到的那個 placement 上重跑**) |
| `io_rg[e] = per_net_steiner[e]` | M3 T1 的新欄位(若 M3 未完成,退化為只做兩值對照,並在報告標明) |
| `io_mst[e] = per_net_crossings[e]` | `EvalResult.per_net_crossings` |
| `route[e] = route_cross_dw(2)[e]` | §7 |

**協定硬性要求:evaluator 必須在 router 實際看到的 placement 上重跑**(即 F-INV 的 `place_detail` 之後、F-OR 的 `read_def` 當下的 COMPONENTS 座標),不是在我們 GP+LG 的原始座標上。做法:從 `routed.def` 讀回 COMPONENTS,反算 `node_x/node_y`,呼叫 `evaluate_gpu`。**同時記錄「原始 placement 的 evaluator 值」與「post-DP 的 evaluator 值」之差** = §1.3 的 `Δio_after_commercial_DP`。

### 8.2 統計量

1. **總量比** `R_X = Σ_e route[e] / Σ_e X[e]`,X ∈ {`lam_minus_1`, `io_rg`, `io_mst`}。
2. **per-net 相關**:Spearman ρ 與 Pearson r,樣本限定 `route[e] > 0 或 X[e] > 0`(定義寫死,與 M2 報告 §6.1 的「僅取 `per_net_crossings>0`」慣例明確區分並在報告註明差異)。
3. **per-degree bucket** 偏差結構:沿用 M2 報告 §4.2 的 bucket 切法;每個 bucket 報 `R_X` 與 ρ。
4. **per-Λ bucket** 偏差結構:沿用 M3 draft §2.2 的 Λ 分桶(1 / 2 / 3 / 4–8 / >8)。
5. **三值分解總表**(每個樣本一行):

   | design | K | 臂 | router | `Σ(λ−1)` | `Σio_rg` | `Σroute` | `Σio_mst` | `mst_excess` | `route−io_rg` | `io_mst−route` |
   |---|---|---|---|---|---|---|---|---|---|---|

6. **FT 三值表**:`ft_rg` / `ft_mst` / `route_ft` 的同構表。
7. **boundary-pair demand 相關**:`route_pair_demand` vs `EvalResult.boundary_pair_demand`,per-pair Pearson + 每單位邊界長的需求分布 ⇒ 校準 S7 的 `C_ab = ρ·ℓ_ab`,定出 `ρ` 的實測值。

### 8.3 迴歸校準(C2)

對每個樣本擬合(非負最小平方,**無截距**):

```
route[e] ≈ α·(λ_e − 1) + β·(ST_e − (λ_e − 1)) + γ·(io_mst[e] − ST_e)
```

三個 regressor 恰好是 M3 draft §2.2 恆等式的三個分量(必要 crossing / feed-through / MST 繞路),因此係數有直接物理意義:

- `α̂ ≈ 1` ⇒ 「碰 λ 個 region 至少 λ−1 次 crossing」這個下界在真實 router 上是實打實的(預期成立;若 `α̂` 顯著 > 1,代表 router 連下界都做不到,congestion 效應主導)。
- `β̂` ⇒ 真實 router 實現了多少 region-graph 預測的 feed-through。
- `γ̂` ⇒ MST 幾何的繞路裡,有多少會被真 router 實現(`γ̂ → 0` 支持 M3 的 RG 裁決;`γ̂ → 1` 則 MST 才是對的尺)。

報 per-sample 的 `(α̂, β̂, γ̂)` 與 R²,以及 bootstrap 95% CI。

### 8.4 回饋規則(機械可判定)

| # | 規則 | 動作 |
|---|---|---|
| **C1 模型選擇** | 若 `ρ(io_rg, route) > ρ(io_mst, route) + 0.05` **且** `|R_{io_rg} − 1| < |R_{io_mst} − 1|` | RG 模型成為**主報表尺**,M3 Q1 裁決被證實;否則觸發 G2(M3 Q1 重審) |
| **C2 迴歸校準** | 若 §9 的 E3 不過(總量比出界) | 報表新增 `io_calibrated = α̂(λ−1) + β̂(ST−λ+1) + γ̂(io_mst−ST)` 欄,所有絕對數字改用它陳述,並附殘差分布 |
| **C3 objective 權重** | M3 的 `κ_ft` 預設值 | 從 `1.0` 改為 `β̂/α̂`(真實 router 下 feed-through 相對必要 crossing 的實際成本比)。S7 的 `ρ`(每單位邊界長 pin 數)改用 §8.2-7 的實測值 |
| **C4 exit 判準** | Phase 1 §9 的 M2/M3 exit(相對百分比門檻) | 在校準後的量上重述,並做 **sign-invariance 檢查**:M2-best vs flat、M3-best vs M2-best 的方向是否翻轉。翻轉即為重大發現,必須進報告主文 |
| **C5 轉移檢驗** | ≥3 個 design 各自的 `(α̂,β̂,γ̂)` 的變異係數 CV | CV ≤ 25% ⇒ 校準視為 design-independent,可外插到 ISPD2005;CV > 25% ⇒ **校準不可轉移**,觸發 G3,結論改為「Phase 1 主 benchmark 必須搬到 LEF/DEF」 |

### 8.5 被否掉的校準形式

| 選項 | 否掉理由 |
|---|---|
| **單一純量修正 `route ≈ c · io_mst`** | 丟掉全部結構資訊;而且 M3 draft §2.2 已實測 `mst_excess` 佔 detour 的份額**隨 placement 從 13% 變到 65%**,單一純量必然在不同臂之間失效 |
| **per-net 機器學習校準(GBDT/NN)** | 過擬合風險 + 不可解釋 + 無法回饋成 objective 權重(C3 就沒東西可用了);而且樣本量(design 數)遠小於特徵維度的合理需求 |
| **只比總量、不看 per-net** | 總量對得上但 per-net 全錯的情況完全可能(高估與低估互相抵消),而 placer 的 reweighting 與 exit 判準都依賴 per-net 排序 ⇒ 必須看 Spearman |
| **拿 GR guide 當 ground truth** | guide 是 GCell 粒度的矩形,橫跨邊界的 guide rect ≠ 一次 crossing(§5.3);只能作為 S4 的中途驗收 |

---

## 9. Exit 判準與 gate

### 9.1 Stage 2 自身的 exit(判的是「校準做完了沒」,不是 placement 品質)

| # | 判準 |
|---|---|
| **E1** | ≥ 3 個 design × ≥ 2 個 K × 2 臂(flat / ours)= **≥ 12 個 routed DEF** 成功產出,且每個都通過 §6.3 的 V1–V5;未繞線 net 比例 ≤ 2% 並如實列出 |
| **E2** | 最佳模型(io_rg 或 io_mst)對 route 的 per-net Spearman **≥ 0.70**(在 `route>0 或 X>0` 的 net 上,degree ≤ `max_degree` 的 bucket) |
| **E3** | 總量比 `R_X ∈ [0.80, 1.25]`;不成立則必須交出 C2 的迴歸校準與殘差 |
| **E4** | §8.2 的 7 張表全部產出;C1 與 C5 的判定寫進報告 |
| **E5** | C4 的 sign-invariance 檢查完成,結果(不論通過與否)寫進報告主文 |

### 9.2 Gate(觸發即改設計,不是改數字)

| # | 條件 | 動作 |
|---|---|---|
| **G1** | Innovus license 在 S6 的 time-box 內仍不通 | Stage 2 以 **F-OR 單 router** 交付;報告標題與摘要明寫「open-source router ground truth」,所有對外主張降級為「相對 OpenROAD 的校準」;Innovus 列為 future work。**不得把 OpenROAD 的結果寫成 commercial-tool ground truth** |
| **G2** | C1 判定 MST 尺優於 RG 尺 | M3 Q1(RG 模型)必須重審;通知 M3 分支,`κ_ft` 的定義可能要換 regressor |
| **G3** | C5 的 CV > 25% | 校準不可轉移 ⇒ 提報使用者:Phase 1 的主 benchmark 是否搬到 LEF/DEF。**這是需要使用者決策的分岔,不由執行者自決** |
| **G4** | `route_cross_raw` 與 `route_cross_dw(2)` 總量差 > 15% | 「沿界線走」主導 ⇒ 所有結論雙值陳述,且需補一個「wire 與邊界的距離分布」診斷 |
| **G5** | V1/V2 在 F-INV 上系統性被破(CTS/opt 關不掉) | 改用 Innovus 的 `place_detail` + `routeDesign` 最小子集;仍不行則該 router 只能量 GR 級,或退回 F-OR |
| **G6** | 未繞線 net 比例 > 2%(congestion) | 樣本有選擇偏差 ⇒ 降 target_density 重跑 placement,或改用 utilization 較低的 design;並在報告量化偏差方向 |
| **G7** | `Δio_after_commercial_DP` 顯示 M2/M3 的改善在 DP 後流失 > 50% | 這是 Phase 1 的重大發現,**必須立刻回報**並觸發「decode + refinement(Phase 1 §4 步驟 3)是否需要提前到 M4」的討論 |

---

## 10. Task 分解

規約沿用 M2 §11 / M3 §8:每個 task 一次 TDD 迴圈;測試一律 `$DP/.venv312/bin/python -m pytest`;新程式碼全在 `ioplace/`(OpenROAD 端腳本例外,放 `ioplace/route_eval/or_scripts/`,不 import torch);DP 原始碼不動。

| Task | 需 Innovus? | 內容 | 驗收 |
|---|---|---|---|
| **S0** 環境固化 | 否 | 把 §3 的所有探測寫成 `ioplace/diagnostics/probes_stage2/probe_env.py`,輸出 `results/stage2/env.json`(Innovus 路徑/版本、license port 連通性、OpenROAD 版本與 feature、各 benchmark 的存在性與大小 + sha256) | JSON 產出;§3 的每個數字可追溯;license 不通如實記為 `false` |
| **S1** DEF 輸出 | 否 | `ioplace/export/def_export.py`:`export_def(placedb, params, node_x, node_y, out_dir)` 產 `out.def` / `regions.json` / `netmap.json` / `coord.json`;`run_placement_io.py` 加 `--emit-def DIR` | mgc_fft_1 GP+LG 後輸出;OpenROAD `read_def` 零 error;`#COMPONENTS` 與輸入相同;讀回座標 vs `node_x/node_y` 差 ≤ 1 DBU;`netmap` 對 `placedb.net_names` 逐項相同 |
| **S2** routed-DEF 解析 | 否 | `ioplace/route_eval/or_scripts/dump_segments.py`(odb `dbWireDecoder`)+ `ioplace/route_eval/segments.py`(讀 npz);另寫 `def_text_parser.py` 作抽樣交叉驗證 | 手寫小 DEF 的 golden test(含 `*` 續值、via、`VWIRE`、`RECT`);**reconciled** `Σ route_wl + Σ_RECT(long_side − short_side)` vs `dbWire::getLength()`(`report_wire_length`)誤差 < 1%,raw definitional delta(`route_wl` 未加回 RECT 項)照實揭露、不當 gate;**全部** routed net(不抽樣)odb vs 文字解析逐 net 相同 |
| **S3** crossing 抽取 | 否 | `ioplace/route_eval/route_crossings.py`,**重用 `evaluator_ref._walk_segment`**;產 `route_cross_raw/dw(δ)`、`Λ_route`、`route_ft`、`route_pair_demand`、`route_wl` | 合成 case:一條直線橫跨 k 個 grid region → 精確 k−1;恆等式 `raw ≥ dw(δ) ≥ Λ_route−1 ≥ 0` 隨機測試 1,000 次;把 evaluator 的 MST edge 當成「假 wire」餵進來 ⇒ 逐位元重現 `per_net_crossings` / `per_net_ft` / `boundary_pair_demand`(**這條是最強的口徑對齊證明,必做**) |
| **S4** LEF 清洗 + F-OR 端到端演練 | 否 | 修 ISPD2015 tech.lef 的重複 VIA(DRT-0338)或改走 NanGate45;跑通 1 個 design 的 GP+LG → DEF → OR GR+DR → routed DEF → 四值表 | 端到端一次成功;§8.2 的表 1–5 產出;`results/stage2/rehearsal/*.json` |
| **S5** NanGate45 語料接入 | 否 | 產 `benchmarks/ispd25/*.json` DREAMPlace config(**必須 glob 全 15 個 LEF,tech 在最前**——見 §3.2 踩雷);跑通 mempool_tile_wrap 的 GP+LG;量測 mempool_group / mempool_cluster 的 `PlaceDB.read` 時間與 peak RSS | tile_wrap GP+LG 完成並出 evaluator 報表;group/cluster 的讀取結果**如實記錄(含失敗)**,回填 §4.3 的 L-Q1-b |
| **S6** Innovus 環境驗證 | **是** | §3.1 的 I1–I8 逐項打勾 | checklist JSON;任一項失敗即記錄失敗模式;I1 失敗 ⇒ 觸發 G1 |
| **S7** F-INV flow | **是** | §6.2 的 Tcl 骨架落地為 `ioplace/route_eval/or_scripts/innovus_route.tcl`;V1–V5 assert 腳本 | 1 個小 design 端到端;V1–V5 全綠 |
| **S8** 校準實驗矩陣 | 是/否 | L1 語料 × K{16,32} × 臂{flat, ours} × router{OR, INV};L2 語料至少 tile_wrap 全跑 | ≥12 個有效樣本(E1) |
| **S9** 校準分析 | 否 | `ioplace/diagnostics/stage2_calibration.py`:§8.2 的 7 張表 + §8.3 迴歸 + C1–C5 判定 | 每個數字可追到 `results/stage2/**/*.json` |
| **S10** 報告 | 否 | `docs/results/stage2-calibration-report.md`;E1–E5 逐條打勾/打叉;G1–G7 判定 | 沿用 M1/M2 的誠實慣例——不挑好看的 case |

### 依賴序

```
S0 ── S1 ── S2 ── S3 ── S4 ──┬── S8 ── S9 ── S10
                             │
      S5 ──────────────────  ┤
                             │
      S6 ── S7 ────────────  ┘   (S6/S7 需 Innovus;不通則 S8 只走 F-OR,觸發 G1)
```

- **S0–S5 完全不需要 Innovus**,這是本規劃的核心排程設計:license 是否打通,不影響 80% 的工作量。
- **S3 的最後一條驗收(把 MST edge 當假 wire 餵回去、逐位元重現 evaluator 欄位)是整條鏈路的信任基石**,必須在 S4 之前綠。
- S5 與 S1–S4 可並行(不同人/不同 worktree)。

---

## 11. 風險與未定案(誠實清單)

| # | 風險 | 狀態 | 緩解 |
|---|---|---|---|
| **A1** | Innovus 21.19 能否在**無 `.lib`** 的 ISPD2015 上 `init_design` | **假設,未驗證** | I4;不行就把 ISPD2015 交給 F-OR,Innovus 只跑 NanGate45(有 .lib) |
| **A2** | Innovus `place_detail` 不改變 instance 集合 / 不插 buffer | **假設,未驗證** | V1–V3 assert 硬擋;破了就作廢樣本(G5) |
| **A3** | Innovus 21.19 讀得動 Innovus 23.34 寫的 DEF 5.8(ISPD2025) | **假設,未驗證** | I6;不行則 ISPD2025 只走 F-OR |
| **A4** | `defOut -routing` 的 wire 表達能被 odb 完整解出 | **假設**(odb 讀得動 23.34 的 place_opt DEF 是實測,但那份沒有 signal routing) | S7 的第一個小 case 就驗;不行則啟用自寫文字剖析器 |
| **R1** | **license 三台全部 `No route to host`** | **實測,現況為阻塞** | S0–S5 全部先做;G1 定義了降級交付 |
| **R2** | ISPD2005(M0–M3 的全部數字)無法直接校準,只能靠轉移 | **設計限制,已知** | C5 檢驗 + G3 升級給使用者決策。**這是本規劃最大的方法論裂縫,v2 前必須讓使用者知情** |
| **R3** | `mempool_cluster` 9.7GB DEF 的 parse 時間/記憶體 | **未驗證** | S5 的驗收就是把它變成事實;失敗則 10M 校準退回 mempool_group(5M) |
| **R4** | ISPD2015 tech.lef 過不了 TritonRoute(`DRT-0338` 重複 VIA) | **實測** | S4 修 LEF(dedupe VIA)或改走 NanGate45;修 LEF 屬於「改測資」,必須在報告揭露 |
| **R5** | congestion 導致 route 不完 ⇒ 樣本選擇偏差 | **實測**(mgc_fft_1 不加 `-allow_congestion` 直接 GRT-0116 中止) | G6;同時量化「未繞線 net 的 degree/λ 分布」以判斷偏差方向 |
| **R6** | router 沿邊界走製造假 crossing | **未量化** | §7.4 的 δ 掃描 + G4 |
| **R7** | commercial DP 搬動 cell ⇒ evaluator 必須在 post-DP placement 上重跑 | **設計已處理**(§8.1) | 同時把 `Δio_after_commercial_DP` 升格為獨立 metric(§1.3、G7) |
| **R8** | 兩個 router(OR / INV)給出不同的 crossing ⇒ 「ground truth」不唯一 | 預期會發生 | 分欄報;把 `route_or` vs `route_inv` 的一致性本身當成一個結果(router-dependence 的量化) |
| **R9** | OpenROAD 沒有 detailed placement 步驟 ⇒ F-OR 與 F-INV 的 placement 不同源 | 已知 | F-OR 可選加 OpenROAD `detailed_placement` 作第三臂對齊;或明確聲明 F-OR 量的是「GP+LG 直入 router」 |

### 低信心段落總表(v2 對抗性審查請優先攻擊這裡)

| ID | 段落 | 不確定的是什麼 | 由誰解決 |
|---|---|---|---|
| L-Q1-a | §4.3 | ISPD2005 → LEF/DEF 的校準轉移是否成立 | C5 / S9;失敗則需使用者決策 |
| L-Q1-b | §4.3 | mempool_cluster/group 的可讀性 | S5 |
| L-Q2-a | §5.4 | passthrough DEF writer 的輸出能否被 Innovus `defIn` | S1(OpenROAD 把關)+ I6 |
| L-Q4-a | §7.4 | δ 的取值(預設 2)缺乏第一性原理,是對 lattice 解析度(die/512)的直覺 | S4 的 δ 掃描實測 |
| L-Q4-b | §7.5 | 「全層合計」是否為正確口徑(高層長線可能不該與 metal1 短線等權) | S9 的 per-layer 附表;必要時 v2 改為分層加權 |
| L-Q6-a | §8.3 | 三分量迴歸假設 route 是三個分量的**線性**組合 | S9 的 R² 與殘差結構;R² < 0.5 則此形式作廢 |
| L-Q6-b | §8.4 C3 | `κ_ft ← β̂/α̂` 的推導是啟發式的(把「成本比」等同「計數比」) | 需要 M3 分支確認;或退回 `κ_ft` 掃描 |
| L-全文 | §6.2 | **整個 F-INV Tcl 骨架**都是憑 Innovus 一般用法寫的,一行都沒在本機執行過 | S6/S7 |

---

## 12. 證據附錄(本次盤點的實跑紀錄,S0 待遷入 repo)

**狀態:下列每一條都是 2026-08-13 在本機實跑取得;探測腳本目前在 `/tmp/or_probe/`,S0 完成前一律視為「待重現」**(沿用 M2/M3 的 finding 遷入慣例)。

| # | 探測 | 結果 |
|---|---|---|
| P1 | `socket.connect((lshc/lstc/lstn, 5280))` | 三台 DNS 皆解得出,TCP 全部 `[Errno 113] No route to host` |
| P2 | `ls /usr/cad/cadence/INNOVUS/` | `cur` → `INNOVUS_21.19.000`;`cur/tools/bin/innovus` symlink 存在 |
| P3 | `openroad -version` / `info commands` | `v2.0-17598-ga008522d8`,`+Charts +GPU +GUI +Python`;`read_lef/read_def/global_route/detailed_route/write_def/set_thread_count` 皆存在 |
| P4 | OpenROAD 讀 ISPD2015 `mgc_fft_1` + `global_route` | 讀取 OK(9 layers / 331 cells / 33,307 nets / 32,281 insts);不加 `-allow_congestion` → `[ERROR GRT-0116]`;加了則完成並寫出 5.1MB guide(按 net name,rect + layer) |
| P5 | OpenROAD `detailed_route` on ISPD2015 | **失敗**:`[ERROR DRT-0338] Duplicated via definition for VIA23_2cut_E` |
| P6 | OpenROAD 讀 `mgc_pci_bridge32_a/floorplan.def` 的 REGIONS/GROUPS | 4 regions(EXCLUSIVE,1–5 boundary)+ 4 groups(3,922/1,552/3,530/118 insts);`write_def` 正確重吐 `+ TYPE FENCE` |
| P7 | `openroad -python`:`import odb` | OK;`odb.dbWireDecoder` 具 `PATH/POINT/POINT_EXT/VIA/TECH_VIA/RECT/SHORT/VWIRE/JUNCTION/END_DECODE` |
| P8 | DREAMPlace 讀 ISPD2015 `mgc_fft_1` | 35,291 physical / 32,281 movable / 3,010 NI / 33,307 nets;`net_names[0]=b'n_9999'`;`sort_nets_by_degree=0`,degree 非單調;`scale_factor=0.0`(未設) |
| P9 | DREAMPlace 讀 NanGate45 + `mempool_tile_wrap.def` | **全 15 LEF:3.1 秒成功**(127,739 movable / 145,589 nets);**只給 lib_setup.tcl 的 10 LEF 子集:`PlaceDB.cpp:1469` 斷言失敗 + core dump**(`failed to find pin clk in macro AND2_X1`) |
| P10 | OpenROAD 對 NanGate45 + `mempool_tile_wrap` GR+DR | GR 完成(39MB guide);DR 通過 tech 檢查、走完 metal1–metal10 初始化、進入 `[INFO DRT-0195] Start 0th optimization iteration.`、`Completing 10% with 0 violations`;**本次 time-box 內未跑完全程,「DR 能完整收斂」為待驗證項** |
| P11 | 讀碼:`DefWriter.cpp:12-83` / `PlaceDB.cpp:1377` | DEF writer 為輸入 DEF 逐行 passthrough,只換 COMPONENTS 與重寫 ROW;輸入 DEF 路徑 = `userParam().defInput` |
| P12 | 讀碼:`BasicPlace.py:163` | `ignore_net_degree` 只產生 WL objective 的 net mask,不移除 net |

---

## 13. 主要參考

- Phase 1 design(`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md`)§1/§6/§7/§9/§10/§11
- M3 draft(`docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`)§2.2 的三分解恆等式與實測表
- M2 報告(`docs/results/m2-differentiable-io-report.md`)§4.2 degree bucket、§5.1 `lg_loss`、§6.3 detour
- 使用者指示(`docs/research/2026-08-13-routing-aware-evaluator-directions.md`)方向 B
- ISPD 2025 contest benchmark(`/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/`,含 `openroad_evaluation.tcl` / `GR.tcl` / `lib_setup.tcl` 可作 flow 參照)
- OpenROAD `odb` DEF/wire API;Cadence Innovus 21.19.000(本機安裝,未執行)
