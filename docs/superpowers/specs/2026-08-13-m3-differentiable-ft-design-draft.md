# M3 設計草案 v1:可微 feed-through(S4)+ 邊界容量項(S7)

- 日期:2026-08-13
- 狀態:**v1 草案(未定稿)**——待 Codex 對抗性審查後修訂為 v2,再交 writing-plans 轉成實作計畫
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §2(FT 定義)、§5.3(S4)、§5.4(S7)、§6(evaluator)、§9 M3 列
- 繼承資產:`docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md`(v2)的 S1 L1-SDF 軟歸屬、S2 product-form、§2.5 chunked-k 契約、§4.2/§5.2 schedule、§6.2 params-borne patch、§6.4 secant refresh、§7.2 噪聲 regime——**M3 一律繼承,不重新發明**
- 前置結論:`docs/results/m2-differentiable-io-report.md`(M2 exit PASS:−18.5% io @ +1.31% hpwl;但 **ft_count 反升 +38.5%(adaptec1)/+17.3%(bigblue4)**)
- 新輸入:`docs/research/2026-08-13-routing-aware-evaluator-directions.md`(方向 A)
- **本文所有數值皆為本次在 L4 上以 `$DP/.venv312/bin/python` 對 M2 已存的 `results/m2` 下 `.npz` 實跑量測**(探針腳本目前在 `/tmp`,**必須由 T0 遷入 repo 才能被引用為設計依據**,沿用 M2 的 M3-finding 慣例,見 §10)

---

## 0. 摘要(草案裁決一覽)

| # | 問題 | 草案裁決(v1) | 信心 |
|---|---|---|---|
| Q1 | routing 模型與 FT 定義 | **region-adjacency Steiner(方向 A 候選 1)成為 M3 objective 的參考 routing 模型**;evaluator **只新增** `io_rg`/`ft_rg`/`per_net_steiner` 欄位,MST 既有欄位逐位元不變(M2 硬性分工延續)。核心理由:此模型下 **`io_rg = (Λ−1) + ft_rg` 是恆等式**,S2 與 S4 因此是同一個量的兩個互補分量,不會互相打架;且實測 `ft_rg` 與 `ft_mst` 在 grid 上只差 2.7–3.7%,**不是換尺換出來的改善** | 高 |
| Q2 | S4 可微形式 | **`L_FT = Σ_e w_e·ReLU(R_e − (λ_e − 1))`**,`R_e` = 以 `q_{e,k}` 加權、對 region-graph 距離 `D[home_e,k]` 取的 **WA soft-max(soft 偏心距)**。四候選實測比較(§3.2):此式總量 = 真值的 **0.84–0.96 倍**、Spearman **0.62–0.93**、per-Λ-bucket 份額比 0.11–1.19(其餘三個候選有 1.2–43 倍的 bucket 偏斜)。**與 S2 共用同一條 leave-one-out backward,零額外 chunk pass、只多 3 個 `(E,)` 累加器** | 中高(離線實測支撐;in-loop 行為未驗) |
| Q3 | S7 容量項 | **S7 = 邊界 pin 容量/平衡項(依 spec §5.4),不是 region 面積項。** 面積容量由全域 electrostatics 隱式滿足(spec §4;實測 A0→A2 的 per-region 利用率 max/mean 2.208→2.184,**未惡化**),故**不加面積項**,只加診斷。相對地 M2 **確實惡化了邊界需求的均勻度**(max/median demand-per-length:adaptec1 k16 3.41→4.83、k32 3.23→4.60、bigblue4 2.33→2.90)⇒ S7 以**長度正規化的平衡約束**形式進 objective | 中(需求惡化已實測;絕對容量門檻在 ISPD2005 上無法定錨) |
| Q4 | 完整 objective 排程 | **不引入第三個 ρ。** 只加兩個**無量綱比值** `κ_ft`(預設 1.0)與 `κ_cap`(預設 1.0);`λ_io` 仍由 M2 的 `ρ_io × ratio_ema` 導出,`ratio` 改量**合併項**的梯度。`κ_ft=1` 時 `ReLU(λ−1)+ReLU(R−λ+1)` 恆等於 `max(λ−1, R)` = soft `io_rg` 的下界——即「最小化 RG 模型下的總 crossing」;`κ_ft` 大於 1 = 額外加罰 feed-through。**新增的離散事件只有 `home` 重算與 S7 啟用,前者掛在既有 callback 上,不新增事件點** | 中高 |
| Q5 | 實驗與 exit | exit 改成**對現任者的 Pareto 支配**判準(§6.2):M3 best 臂必須同時 (a) `ft_mst` 不高於 M1 的 ft、(b) `io_mst` 不高於 M2-best 的 io 的 1.02 倍、(c) `Δhpwl ≤ +2%` vs flat、(d) `ft_rg` 不高於 flat 的 `ft_rg` 的 0.85 倍。實測 `σ_seed(ft)/mean = 0.83%`(3σ = 2.5%),門檻遠在噪聲外。**bigblue4 per-case ρ 校準列為 M3 的前置實驗 T5,擋在所有 bigblue4 臂之前** | 中高 |
| Q6 | 風險/fallback | 7 個 gate(G1–G7,§7),最重要的是 **G3「RG 模型被鑽漏洞」**:若 `ft_rg` 下降而 `ft_mst` 上升超過 +3%,代表 objective 在利用 region-graph 的幾何樂觀性 ⇒ 啟動候選 2 的 maze 抽樣校驗並改用 bounded-detour 變體 | 中 |
| Q7 | task 分解 | **T0(探針遷入)→ T1(evaluator RG 擴充)→ T2(S4 op)→ T3(schedules)→ T4(driver)→ T5(bigblue4 ρ 校準,實驗)→ T6(κ 掃描,實驗)→ T7(ablation,實驗)→ T8(報告)**;條件 task T9(S7 實作)、T10(S4 變體深掃)、T11(maze 校驗) | — |

**一句話結論:** M2 用 S2 壓掉了 `Λ−1`,但把壓力轉嫁到「跨越距離」——實測 `ft_rg` 從 2,454 升到 3,383(+37.9%)。region-adjacency Steiner 模型把 crossing 精確拆成 `(Λ−1) + ft`,S2 已覆蓋前者,M3 只需要一個**同樣走 `q_{e,k}` 這條路徑**的第二個係數就能覆蓋後者——這正是 §3 裁決的 soft 偏心距形式,實作上是 M2 op 的一個係數擴充,不是新的運算子。

---

## 1. 新增名詞與符號(M2 §1 之外)

| 符號 | 意義 | 大小/來源 |
|---|---|---|
| `G_R` | **region adjacency graph**:K 個節點,兩 region 在 lattice 上共邊即相鄰 | 實測相鄰對數:adaptec1 k16 grid 24、k32 52、k16 slicing 33 |
| `D[a,b]` | `G_R` 上的 hop 距離(all-pairs,Floyd–Warshall,K≤32 ⇒ 常數表) | (K,K) int8,靜態 |
| `ℓ_ab` | region a,b 共享邊界長度(以 lattice 邊數計) | (K,K) int32,靜態 |
| `Λ_e` | hard 觸及 region 數(M2 已有) | evaluator |
| `ST_e` | **region-graph Steiner cost**:在 `G_R` 上連接 `touched(e)` 的最小樹邊數 | 新增 evaluator 欄位 |
| `io_rg` | `Σ_e ST_e` = crossing-minimal routing 下的 IO 數 | 新增 |
| `ft_rg` | `Σ_e (ST_e − (Λ_e − 1))` = 同模型下的 pure feed-through 數 | 新增(**由恆等式導出,不需獨立計算**) |
| `mst_excess` | `io_mst − io_rg` = MST 幾何相對 crossing-minimal 的超額 | 報表欄位 |
| `home_e` | net e 的「主 region」= 擁有最多 pin 的 region(evaluator 每次 callback 給) | `(E,)` int8 |
| `ecc_e` | `max` over `k ∈ touched(e)` of `D[home_e, k]`(hard 偏心距) | — |
| `R_e` | **soft 偏心距**(§3.1) | `(E,)` fp64 |
| `κ_ft` | `λ_ft/λ_io` 的無量綱比值 | 預設 1.0 |
| `β` | 偏心距 soft-max 的溫度(hop 為單位) | 預設 0.5 |
| `κ_cap` | S7 的梯度量級相對 S2+S4 的比值 | 預設 1.0 |
| `D_ab`/`C_ab` | 邊界 (a,b) 的 soft 需求 / 容量 | §4 |

**硬性分工(延續 M2 §1):** `evaluator_ref`/`evaluator_gpu` 的既有欄位語意與數值 **M3 不得改動**;`io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home` 一律**新增**,由既有等價測試 + 新增的「舊欄位逐位元不變」測試把關。所有 exit 判定與報表數字回 evaluator,`L_FT` 只是 optimizer-facing surrogate。

---

## 2. Q1 — routing 模型與 FT 定義的耦合(**先裁**)

### 2.1 問題

spec §2 定義 pure feed-through =「tree edge 穿過某 region、且該 net 在該 region 內無任何 pin」。「穿過」在哪個 routing 模型下成立,決定了 `ft_count` 的值:現行 evaluator 用 **MST 幾何 + L 形 walk**(`ioplace/evaluator_ref.py:68-72` 固定「先水平段(在 `y0`)再垂直段(在 `x1`)」),這個 L 方向是**任意約定**;方向 A 提出的兩個替代是 region-adjacency Steiner(候選 1)與 boundary-cost maze(候選 2)。M2 的教訓(design v2 §3.2.5/H3)是:**objective 與 evaluator 口徑錯位會讓 alignment 診斷失效**,所以 M3 必須先把模型釘死。

### 2.2 定案

**M3 的參考 routing 模型 = region-adjacency Steiner(候選 1)。** 定義:

```
route(e)     := G_R 上連接 touched(e) 的最小 Steiner tree T_e(邊權 1 = 一次 boundary crossing)
crossings(e) := |E(T_e)| = ST_e
FT(e)        := |V(T_e) 扣掉 touched(e)| = (ST_e + 1) − Λ_e = ST_e − (Λ_e − 1)
```

於是有**恆等式**

```
io_rg  = Σ_e ST_e = Σ_e (Λ_e − 1) + Σ_e FT(e) = hard_lambda_sum + ft_rg
io_mst = io_rg + mst_excess
     ⇒ detour_mst(M2 報告 §6.3 的量) = ft_rg + mst_excess        (三分解)
```

**證據(本次實測,adaptec1/bigblue4 真實 M2 placement;Λ≤3 為精確解,Λ≥4 用 metric-closure MST 上界 ⇒ `io_rg`/`ft_rg` 皆為上界):**

| run | `io_mst` | `hard_λ_sum` | `io_rg` | **`ft_rg`** | `ft_mst` | `mst_excess` | `detour_mst` |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 k16 grid flat | 30,256 | 24,467 | 26,921 | **2,454** | 2,545 | 3,335 | 5,789 |
| adaptec1 k16 grid **M2 best** | 24,647 | 19,337 | 22,720 | **3,383** | 3,525 | 1,927 | 5,310 |
| adaptec1 k32 grid flat | 47,839 | 34,485 | 42,585 | 8,100 | 8,281 | 5,254 | 13,354 |
| adaptec1 k32 grid M2 | 41,272 | 27,985 | 37,405 | 9,420 | 9,600 | 3,867 | 13,287 |
| adaptec1 k16 slicing flat | 36,615 | 25,951 | 31,461 | 5,510 | 7,389 | 5,154 | 10,664 |
| adaptec1 k16 slicing M2 | 33,460 | 22,202 | 28,600 | 6,398 | 8,812 | 4,860 | 11,258 |
| bigblue4 k16 grid flat | 100,333 | 81,205 | 87,953 | 6,748 | 6,935 | 12,380 | 19,128 |
| bigblue4 k16 grid M1 | 100,860 | 85,580 | 88,438 | 2,858 | 2,948 | 12,422 | 15,280 |
| bigblue4 k16 grid **M2** | 63,755 | 54,518 | 62,589 | **8,071** | 8,136 | 1,166 | 9,237 |

四個支持本裁決的實測事實:

1. **兩把 FT 尺在 grid 上幾乎一致**:`ft_rg/ft_mst` = 0.964(adaptec1 k16 flat)、0.960(M2)、0.978(k32)、0.973/0.992(bigblue4)。per-net Spearman ρ(`ft_rg` vs `ft_mst`)= 0.875(k16)、0.974(k32)。**⇒ 換模型不會憑空製造 FT 改善**,「改尺換數字」的指控可用這張表直接反駁。rectilinear slicing 是唯一顯著分歧的組態(0.75/0.73,ρ=0.59–0.60),差額正是幾何被迫繞行的部分——這恰恰是 evaluator 應該與 objective 分開報的東西。
2. **M2 在兩把尺上都讓 FT 變差**(adaptec1 `ft_rg` +37.9% / `ft_mst` +38.5%;bigblue4 +19.6%/+17.3%)⇒ M2 的 FT 退化不是 MST 約定的假象,**M3 有真實的問題要解**。
3. **恆等式把 M2 的「detour 14–34%」拆開了**:adaptec1 flat 的 5,789 detour = 2,454 真 FT(42%)+ 3,335 MST 次佳(58%);bigblue4 M2 的 9,237 = 8,071(87%)+ 1,166(13%)。**S2 已經吃掉了 MST 次佳的部分(bigblue4 `mst_excess` 12,380→1,166),剩下的 detour 幾乎全是真 FT**——M3 的目標函數對象因此非常明確。
4. **成本可負擔**:Λ 分布極度偏斜(adaptec1 k16:Λ=1 有 195,140 net、Λ=2 有 20,036、Λ=3 有 1,239、Λ≥4 只有 517;bigblue4:74,680 / 2,116 / 675)。⇒ `ST_e` 對 99% 以上的非平凡 net 有 closed form(Λ=2 查表 `D[a,b]`、Λ=3 取 `min_v D[v,a]+D[v,b]+D[v,c]`),Λ 在 4–8 走批次 Dreyfus–Wagner(`2^Λ × K` 表),Λ 大於 8 走 metric-closure MST 並在報表標記為上界。**成本與 pin 數無關,只與 Λ≤K 有關**,比 MST walk 更容易外推到 30M net。

### 2.3 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| **維持 MST 幾何為唯一模型** | (a) L 形方向是任意約定,`ft_mst` 對它敏感而 `io_mst` 在 grid 下不敏感(spec §6 的 O(1) 特例),**FT 因此比 IO 更依賴約定**;(b) 沒有 `io = (Λ−1) + ft` 的分解,S2 與 S4 無法證明不重疊;(c) 實測 `mst_excess` 佔 detour 的 13–65%,且**隨 placement 幾何劇烈變動**(bigblue4 flat 65% → M2 13%),會污染跨臂比較 |
| **候選 2(boundary-cost maze)當主 evaluator** | 每條 tree edge 在 4096 平方的 lattice 上跑 A*/Dijkstra,對 12M–36M net 完全不可能塞進 GP 內圈(spec §6 的預算是 1 秒以內 / 30M net)。**保留為抽樣校驗器**(候選 3 混合式):抽 1 萬條 net 跑 crossing-cost maze,量 RG 模型的樂觀誤差 → T11 |
| **完全取代 MST 欄位(不保留舊尺)** | 破壞 M0/M1/M2 的可比性,且違反 M2 §1 的硬性分工。定案是**雙尺並報**,每張表都出 `io_mst` / `io_rg` / `ft_mst` / `ft_rg` / `mst_excess` 五欄 |
| **以 λ−1 為唯一 IO 定義(放棄 FT 概念)** | 直接與 spec §2/D4 的計數語意衝突,且 feed-through 是本題的主要 novelty 主張,不能取消 |

### 2.4 明確標記的低信心處(需 v2 前確認)

- **L1:RG 模型的幾何樂觀性未量化。** region-graph 允許「無限繞路換一次 crossing」,rectilinear 的相鄰邊界可能離 net 很遠。`ft_rg` 只是**下界**。→ **P5 maze 抽樣校驗**(T11)必須在 M3 報告前給出「RG 值 vs 帶長度成本 ε 的 min-crossing 幾何路徑」的偏差分布;若中位偏差超過 20%,v2 應改用 **bounded-detour 變體**(路徑限制在 net soft bbox 膨脹 δ 之內)。
- **L2:Λ≥4 用 metric-closure MST(2-approx 上界)。** 目前它只佔 `ft_rg` 的 1–5%(bigblue4 A0:68 / 6,748),但 adaptec1 k=32 升到 4.5%。→ T1 必須實作 Λ≤8 的精確 Dreyfus–Wagner 並回報「上界與精確值的差」。
- **L3:`ft_mst` 對 L 形約定的敏感度未實測。** → **P1**(T0):把 evaluator 的 walk 改成先垂直後水平重跑一次,量 `ft_mst` 的變動幅度。若變動超過 10%,§2.3 對 MST 的否決理由再加一條硬證據;若低於 2%,本文對「任意約定」的措辭要收斂。

---

## 3. Q2 — S4 可微 FT surrogate

### 3.1 定案公式

沿用 M2 的 `p_{i,k}`(S1)與 `q_{e,k} = 1 − exp(S_{e,k})`(S2),**新增三個 `(E,)` 累加器**:

```
常數(靜態,K≤32):  D[h,k],  ecc_max[h] = max_k D[h,k]
per-callback 更新:  home_e,  取值於 0..K-1(evaluator 給:pin 數最多的 region)

# 與 S2 的 FWD-2 同一趟 k-chunk 迴圈內累加:
a_{e,k} = q_{e,k} · exp( (D[home_e,k] − ecc_max[home_e]) / β )     # 落在 (0,1],建構上不溢位
N_e    += a_{e,k}                                                  # (E,) fp64
M_e    += a_{e,k} · D[home_e,k]                                    # (E,) fp64
λ_e    += q_{e,k}                                                  # (E,) fp64  [M2 既有]

R_e   = M_e / max(N_e, ε)              # soft 偏心距;β 趨近 0 時趨近 max over {k: q>0} of D[home,k]
L_FT  = Σ_e w_e · ReLU( R_e − (λ_e − 1) )
L_IO  = Σ_e w_e · ReLU( λ_e − 1 )      # M2 既有
L_S24 = L_IO + κ_ft · L_FT             # κ_ft = 1 時恆等於 Σ_e w_e · max(λ_e−1, R_e)
```

**backward 與 S2 完全共用**(這是本形式最重要的工程性質)。M2 的 backward 核心是

```
∂L/∂p_{i,k} = c_{e,k} · exp(S_{e,k} − ℓ_{i,k})          # leave-one-out(M2 §3.2.2)
```

M3 只是把 `c_{e,k}` 從 `w_e·1[λ_e 大於 1]` 換成

```
c_{e,k} = w_e · [ 1[λ_e 大於 1] + κ_ft · 1[R_e 大於 λ_e−1] · ( (D[home_e,k] − R_e)·a_{e,k}/(q_{e,k}·N_e) − 1 ) ]
```

其餘(softmax Jacobian 的 `A_i` 耦合項、`(−1/τ)·∂d_k/∂x`、`num_movable` 以上清零、fp64 `index_add`)**逐行沿用 `ioplace/ops/io_term.py::_IoFn` 的四趟結構**。⇒ **零額外 chunk pass、零額外 `(N,K)`/`(P,K)` 張量、常駐狀態只多 `N_e`/`M_e`/`R_e`/`home_e` 四個 `(E,)`**,M2 §2.5 的記憶體契約自動滿足。

### 3.2 為什麼是「偏心距」而不是「星形和」——四候選實測

hard 版四個候選對「精確 `ft_rg`」的逼近品質(adaptec1,M2 存檔 placement;`ratio` = 總量比、`ρ` = per-net Spearman、bucket 欄 = 該 Λ 桶佔 surrogate 總量的份額 除以 佔真值總量的份額,理想 = 1):

| 候選 | 形式 | ratio(A0k16 / A2k16 / A2k32 / A2slic) | ρ | bucket 份額比 L2 / L3 / L4+(A2k16) |
|---|---|---|---|---|
| S4a 星形和(home 根) | `Σ_k 1[k∈T]·(D[home,k]−1)_+` | 1.80 / 1.44 / 1.36 / 1.21 | 0.19 / 0.34 / 0.71 / 0.66 | 0.69 / **2.60** / **12.54** |
| **S4b 偏心距(home 根)【定案】** | `(ecc_home − (Λ−1))_+` | **0.96 / 0.96 / 0.95 / 0.84** | **0.76 / 0.85 / 0.93 / 0.62** | **1.04 / 0.70 / 0.20** |
| S4c 星形和(最佳根) | `min_h Σ_k (D[h,k]−1)_+` | 0.48 / 0.30 / 0.56 / 0.31 | −0.34 / 0.21 / 0.71 / 0.65 | 0.41 / 2.86 / **31.27** |
| S4d 偏心距(最佳根) | `min_h (ecc_h − (Λ−1))_+` | 0.07 / 0.11 / 0.26 / 0.13 | 0.86 / 0.92 / 0.90 / 0.80 | 1.09 / 0.17 / 0.00 |

裁決理由:

1. **量級對**:只有 S4b 的總量落在真值的 0.84–0.96 倍(其餘:S4a 高估 1.2–1.8 倍、S4c 低估 2–3 倍、S4d 低估 4–15 倍)。低估的候選會讓 auto-normalization 把權重放大到病態區,高估的會讓 objective 追一個不存在的量。
2. **不偏斜**:M2 §3.2.1 已經寫死了「任一 bucket 的份額比偏離 1 超過 3 倍即推翻」的機械判準。**S4a/S4c 在 L4+ bucket 是 12.5 倍 / 31.3 倍,直接違反**(原因:星形和對每個 terminal 獨立計費,鏈狀 net `a-b-c` 明明 Steiner=Λ−1、FT=0,星形和仍收 1;而 Λ≥3 的 net 真實 FT 極稀疏——bigblue4 A0 的 2,116 個 Λ=3 net 總共只貢獻 163 的 `ft_rg`,星形和卻算出 1,445)。S4b 的最差 bucket 是 0.11–0.35(**低估** L4+),方向安全。
3. **Λ=2 上精確**:實測 Λ=2 桶上兩者逐位元相同(bigblue4 A0 6,517 = 6,517;A2 7,872 = 7,872)。而 Λ=2 佔 `ft_rg` 的 **87–97%**(bigblue4 A0 96.6%、A2 97.5%;adaptec1 A2 90.3%)。S4b 在 Λ=2 上同樣精確(`ecc − 1 = D[a,b] − 1`)⇒ **兩者在 FT 質量的主體上都對,差別全在稀疏的 Λ≥3 尾巴,而 S4b 的尾巴處理與真值一致(接近 0)**。
4. **Spearman 0.62–0.93 比 M2 的 S2 更好**(M2 報告 §6.1 的 `spearman_rho` = 0.38–0.52,而 M2 已據此判定「目標與指標方向一致」)。同一標準下 S4b 的 alignment 更強。
5. **`κ_ft=1` 的語意乾淨**:`ReLU(λ−1)+ReLU(R−λ+1) = max(λ−1, R)`,而 `max(Λ−1, ecc_home)` 是 `ST_e` 的標準下界(home 是 terminal,樹必須連到最遠 terminal)。⇒ 合併項就是 **soft `io_rg` 的下界**,`κ_ft` 是「額外加罰 feed-through」的唯一旋鈕。

### 3.3 被否掉的替代 S4

| 方案 | 否掉理由 |
|---|---|
| **spec §5.3 原案:soft bbox 覆蓋 `cov_k(e)·(1−q_{e,k})`** | (a) 幾何 bbox 覆蓋與 RG-Steiner 不同構——4x4 grid 上對角 net 的 bbox 覆蓋 16 個 region,Steiner 路徑只經過 7 個,對「對角展開」是乘積式懲罰而非線性;(b) 需要第三個溫度 `τ_b`(soft bbox 的 WA);(c) 需要一組獨立的 per-net soft min/max reduce。**保留為 ablation C5**,而且它有一個真實優點:梯度不經過會飽和的 softmax(見 §7 G1 的 fallback) |
| **soft-Steiner over region graph(可微 MST / 軟根 star `min_h Σ_k q_k D[h,k]`)** | `min_h` 需要 `(E,K)` 累加器(12M x 32 fp32 = 1.5 GB @10M、4.6 GB @30M),且實測(S4c/S4d)品質更差。與 M2 §3.3 否決 soft-MST 的理由一致:Steiner 拓撲是離散的,它是計分尺不是 objective |
| **二次型 `q_e^T W q_e`(pairwise region 交互)** | 每 net `O(K^2)` 且需 `(E,K)`;對大 net 無界;沒有 `ST` 的直接語意 |
| **S3 span proxy 直接當 FT 項(`L_span − L_IO`)** | `span`(觸及 region 集合的 bbox 半周長)是 Steiner 的**弱下界**,對 2x2 區塊 net 給 2 而 Λ−1=3 ⇒ `L_FT` 可為負,clamp 後梯度死。S3 維持 M2 §9.3 的定位:S2 的 F2 fallback,不是 FT 項 |
| **直接對 evaluator 的 `per_net_ft` 做 REINFORCE / 有限差分** | 與 M2 §3.3 同一理由:高變異、需多次 evaluator 呼叫;S5 已是其結構化版本 |

### 3.4 低信心處

- **L4:`β` 的可用窗口未量測。** M2 §4.1 對 `τ` 做過梯度 L1 範數的單峰掃描;`β` 需要同型掃描(`β` 取 {0.25,0.5,1,2} 乘上 `τ_rel` 取 {0.30,0.10,0.03})⇒ **P4(T0)**。`β` 太大 ⇒ `R` 退化成加權平均(懲罰與 WL 共線,失去「相鄰免費」的語意);太小 ⇒ 梯度集中在單一最遠 region,雜訊大。
- **L5:`home_e` 凍結的代價未量測。** 定案是每次 callback(N=50)由 evaluator 重算一次(§5.3)。→ **P2(T0)**:用 M2 的 trajectory 量「相鄰兩次 callback 之間 `home_e` 改變的 net 比例」。若超過 20%,v2 應改為每 iteration 以 chunked argmax 重算(多一個 forward pass)或改用 net pin 質心所在 region。
- **L6:`R_e` 的 fp32/fp64 邊界。** `exp((D−ecc)/β)` 在 `β=0.25`、`D` 跨距 31 hop 時是 `e` 的 −124 次方,fp32 直接歸零(讓只觸及 home 的 net 的 `N_e` underflow)。定案:`N/M` 一律 fp64 累加(與 M2 的 `S` 同規格),並在 T2 加 `β ≥ 0.1` 的 assert。

---

## 4. Q3 — S7 邊界容量項

### 4.1 先裁:S7 是**邊界 pin 容量**,不是 region 面積容量

任務書把 S7 描述為「region capacity(面積/utilization)」,但 spec §5.4 的 S7 定義是 boundary-pin 容量。**本文採 spec 的定義**,理由是實測的:

- **面積容量已被全域 density 場隱式滿足**(spec §4 line 62 的主張)。實測 per-region「movable cell 面積 / region 面積」:adaptec1 k16 flat 的 max/mean = **2.208**,M2 best = **2.184**(−1.1%);k32 M2 = 2.627、slicing M2 = 2.151。**可微 IO 項沒有惡化面積分布**——分布的不均勻來自固定 macro 佔據 region 面積(adaptec1 fixed area 6.41e7),不是 placement 決策。⇒ **不加面積懲罰項**(加了只會與 electrostatic force 對打),改為每次 callback 記錄 `region_util[k]` 進 trajectory 作為佐證。**(低信心 L7:分母應為 region 的 free area 而非 total area;T0-P6 用 free area 重算一次,若 max/mean 在 M2 下惡化超過 10%,本裁決需重審。)**
- **邊界需求的均勻度確實被 M2 惡化了**(evaluator 的 `boundary_pair_demand` 除以共享邊界 lattice 邊數):

| run | 相鄰對數 | median demand/len | max demand/len | **max/median** |
|---|---:|---:|---:|---:|
| adaptec1 k16 grid flat | 24 | 8.21 | 27.97 | **3.41** |
| adaptec1 k16 grid M2 best | 24 | 4.89 | 23.62 | **4.83(+42%)** |
| adaptec1 k32 grid flat | 52 | 8.67 | 27.97 | 3.23 |
| adaptec1 k32 grid M2 | 52 | 5.34 | 24.58 | 4.60(+43%) |
| adaptec1 k16 slicing flat | 33 | 9.20 | 23.36 | 2.54 |
| adaptec1 k16 slicing M2 | 33 | 7.66 | 21.41 | 2.79(+10%) |
| bigblue4 k16 grid flat | 24 | 30.11 | 70.09 | 2.33 |
| bigblue4 k16 grid M1 | 24 | 32.20 | 65.70 | 2.04 |
| bigblue4 k16 grid M2 | 24 | 19.21 | 55.63 | 2.90(+24%) |

M2 把總量壓下來的同時,把 crossing **集中**到少數邊界(max/median +10% 到 +43%)。這正是 Phase 3(IO legalization)會踩到的東西,也是 S7 存在的理由。

### 4.2 S7 的定案形式(平衡式,非絕對容量)

```
容量:  C_ab = ν · ℓ_ab · ( 全部相鄰對的 D_ab 總和 / 全部相鄰對的 ℓ_ab 總和 )      # ν 預設 1.5
懲罰:  L_cap = Σ over 相鄰對 (a,b) of softplus( (D_ab − C_ab) / C_ab )
```

**soft 需求 `D_ab` 的算法(與 §2.2 的 routing 模型一致,且 chunk-safe):**

```
靜態預算: P[h,k,(a,b)] = 從 h 到 k 的所有最短路徑中,使用邊界 (a,b) 的比例   # (K,K,A) 常數,K≤32、A≤2K
per k-chunk: Q[h,k] += Σ over {e: home_e = h} of w_e · q_{e,k}    # (K,K) 累加器(index_add),記憶體可忽略
D_ab = Σ over (h,k) of Q[h,k] · P[h,k,(a,b)]
```

backward:`∂L_cap/∂Q[h,k]` = 對所有相鄰對加總 `σ((D_ab−C_ab)/C_ab)/C_ab · P[h,k,ab]`,是一個 `(K,K)` 矩陣 `G`,於是
`∂L_cap/∂q_{e,k} = w_e · G[home_e, k]` ——**又是同一個「(e,k) 對應一個係數」的形狀**,直接併進 §3.1 的 `c_{e,k}`,共用同一條 backward。

**為何不用 spec §5.4 的字面式 `D_ab = Σ_e w_e q_{e,a} q_{e,b}`:** (a) 它只計「相鄰且都被觸及」的對,對「跨兩跳」的 net 完全不記需求,與 §2.2 的 routing 模型不一致;(b) 對三個互相相鄰的 region 會重複計數;(c) 兩個 q 欄同時在手需要 `(E,K)` 或 K 趟 netlist 掃描。**保留為 ablation。**

**為何用相對(平衡)而非絕對容量:** ISPD2005 bookshelf 沒有 metal pitch/track 資訊,`ρ`(每單位長 pin 數)無法定錨;實測 bigblue4 max = 70.1 crossing 每 lattice 邊,換算後遠低於任何合理的多層 track 容量 ⇒ **絕對式在 Phase 1 benchmark 上是恆不觸發的死項**。相對式量的是 M2 確實惡化的東西。LEF/DEF case(ISPD2015 / mempool,M4)有真實 pitch 時,把 `C_ab = ρ·ℓ_ab` 當成 `ν` 的另一個 config 分支即可,公式不變。

### 4.3 S7 的納入條件(gate,寫死在 T6 判準)

S7 **預設 `κ_cap = 0`(關)**,只有在 T6 的 `κ_ft` 掃描結束後、best 臂滿足下列任一條時才啟用並跑 T9:

- best 臂的 `max/median demand-per-length` 相對同 case flat 惡化超過 **25%**(M2 best 已達 +42%,預期會觸發),或
- LEF/DEF case 上存在 `D_ab` 超過 `ρ·ℓ_ab` 的邊界對。

若兩者皆不成立,S7 以**診斷欄位**形式結案(誠實記錄為 negative result,沿用 M1/M2 報告慣例),不進 objective。

---

## 5. Q4 — 完整 objective 的排程整合

### 5.1 目標函數

```
min  WA-WL + μ·density + λ_io·[ L_IO + κ_ft·L_FT ] + λ_cap·L_cap + λ_margin·L_margin
                                └──── 同一個 IoFtTerm op,共用 backward ────┘
```

### 5.2 權重與排程(**繼承 M2 §4.2/§5.2,只列差異**)

| 量 | 規則 | 與 M2 的差異 |
|---|---|---|
| `τ` | `τ_rel(of)` log-linear 0.30 到 0.03,每 iteration 連續漂移 | **不變** |
| `ρ_io` | `ρ_max·clip((of_on−of)/(of_on−of_full))·ramp`,每 iteration | **不變** |
| `ratio_ema` | WL 梯度 L1 除以**合併項**梯度 L1,每 N=50 iter | **量合併項的梯度**(不是只量 `L_IO`);否則 `κ_ft` 會被正規化抵銷掉 |
| `λ_io` | `ρ_io · ratio_ema`,Lipschitz 護欄 | 護欄改為 `λ_io ≤ c_lip·τ²/γ ÷ max(1, g_comb/g_io)`,其中 `g_comb/g_io` 是同一次 callback 免費得到的梯度比——合併項曲率比 `L_IO` 大多少,護欄就收緊多少 |
| `κ_ft` | **常數**(掃描旋鈕),預設 1.0 | 新增;`κ_ft=0` 逐位元退化成 M2(天然的 ablation 控制組) |
| `β` | **常數**,預設 0.5 | 新增;ablation 掃 {0.25,0.5,1,2} 與「隨 τ 一起退火 2.0 到 0.5」 |
| `λ_cap` | 每 callback 解出使 `L_cap` 的梯度 L1 等於 `κ_cap` 乘上合併項的梯度 L1,上限 `ρ_max` 乘 WL 梯度 L1;當 `L_cap` 梯度為 0 時**保持不變**(項本來就惰性) | 新增。**明確禁止**用「WL 梯度除以 cap 梯度」做 M2 式 ratio 正規化:`L_cap` 是 hinge,啟用初期梯度恆 0,除法會爆炸(§7 R6) |
| `home_e` | 每次 callback 由 evaluator 重算 | 新增 |
| 啟用時序 | IO 與 FT 同時於 `of ≤ of_on = 0.90` 啟用(共用同一個 20-iter ramp);S7 於 `of ≤ of_cap = 0.25` 啟用 | FT 不新增啟用事件;S7 新增一個 |

### 5.3 離散事件集合與 secant refresh(**M2 §6.4 的擴充**)

| 事件 | M2 有 | M3 新增 | 備註 |
|---|---|---|---|
| IO/FT 項啟用(ramp 起點) | 有 | | |
| `ratio_ema` 更新 | 有 | | |
| `w_e` 校準(`α_io` 大於 0) | 有 | | |
| margin 項開關 | 有 | | |
| **`home_e` 重算** | | 新增 | **掛在既有 callback 上,與 `ratio` 同一個事件點 ⇒ 不新增事件點,只新增順序約束** |
| **S7 啟用 / `C_ab` 參考值更新** | | 新增 | 新增一個事件點(`of_cap` 跨越時) |

**強制順序(T4 必須有測試):** `home_e` 由 evaluator 更新 → `D_ab`/`C_ab` 更新 → `ratio_ema` 更新 → (選配 `w_e`) → `refresh_nesterov_secant` → `mark_refreshed()`。M2 的 `obj_version`/`refreshed_version` invariant(`ioplace/schedules.py:92-96`)原樣沿用,`home_e` 與 `C_ab` 的變更都必須讓 `obj_version` 遞增。

**line-search 內常數性(M2 §6.3 坑 3)延伸:** `home_e`、`D`/`ecc_max` 查表、`C_ab` 在一個 iteration 內**只讀**;`N_e`/`M_e`/`R_e` 是 forward 的中間量,每次 `obj_fn` 重算(無狀態,合法)。T4 加測試:同一 iteration 內呼叫 `obj_fn` 三次,`home_e`/`κ_ft`/`λ_cap` 逐位元相同。

**為何 `home_e` 一定要凍結:** 若在 forward 內即時取 `argmax_k q_{e,k}`,`L_FT` 在 home 翻轉處**不連續**(兩個候選 home 的 `D` 加權和一般不相等),等於在 line search 內部注入跳變——正是 M2 R12 花了整節處理的病症。凍結後 `L_FT` 在 iteration 內是 `q` 的光滑函數,跨 callback 的跳變由既有的 secant refresh 吸收。

---

## 6. Q5 — 實驗設計與 exit

### 6.1 量測 regime

沿用 M2 §7.2 的裁決:**`deterministic_flag=1`**,`σ_rep = 0`。**M3 新增:FT 的 seed 噪聲底線**(M2 只報了 io 的):由 `results/m2/noise/flat_seed100x.json` 重算得 `ft_count` = {2545, 2571, 2519, 2532, 2523},mean = 2,538.0,**`σ_seed(ft) = 20.98`(0.83%),3σ = 2.5%**。所有 FT 宣稱一律以 `3σ_seed` 為噪聲帶(**不用 `σ_rep=0`**——那是決定性重跑的退化值,不是科學上的不確定度)。

### 6.2 Exit 判準(**對現任者的 Pareto 支配**,可機械判定)

M3 best 臂(adaptec1 k16 grid,det=1)必須**同時**滿足:

| # | 條件 | 具體數值(adaptec1 k16) | 來源 |
|---|---|---|---|
| E1 | `ft_mst` 不高於 M1 reweight 的 `ft_count` | **不超過 2,430** | `results/m2/ablation/adaptec1_A1_k16_grid.json` |
| E2 | `ft_rg` 不高於 flat 的 `ft_rg` 的 0.85 倍 | **不超過 2,086** | §2.2 表(flat 2,454) |
| E3 | `io_mst` 不高於 M2 best 的 `io_count` 的 1.02 倍 | **不超過 25,140** | M2 報告 §2(24,647) |
| E4 | `Δhpwl ≤ +2.0%` vs flat | **不超過 75,402,146** | M2 報告 §1 |
| E5 | E1/E2 的改善幅度超過 `3σ_seed(ft) = 2.5%` | — | §6.1 |
| E6 | 完整 ablation 表(§6.3)每格有對應 JSON | — | spec §9 M3 第二條 |

同一組判準在 bigblue4 k16 上以該 case 自己的現任者代入(M1 `ft=2,948`、M2 `io=63,755`、flat `ft_rg=6,748`);**bigblue4 的 E3/E4 以 T5 校準後的 per-case `ρ*` 為基準**,不沿用 adaptec1 的 `ρ=0.40`(M2 遺留問題,見 §6.4)。

**為何不用「Δft 低於 −X% vs flat」的固定門檻:** M1 純 reweight 在 bigblue4 已達 `Δft = −57.5%`,固定門檻要嘛對 adaptec1 太鬆、要嘛對 bigblue4 太緊。對現任者的支配式判準直接回答 spec §9 M3 的原話「FT 顯著下降且 WL/IO 無明顯劣化」,而且**不可能靠換尺達成**(E1 用的是 M2 未改的 MST 尺)。

### 6.3 Ablation 矩陣

| 臂 | 設定 | case | 目的 |
|---|---|---|---|
| C0 | flat | 全部(沿用 M2 A0) | 基準 |
| C1 | M2 best(`κ_ft=0`) | adaptec1 k16、bigblue4 k16 | 現任者 / 控制組 |
| C2 | `κ_ft` 取 {0.5, 1, 2, 4} | adaptec1 k16 | **主掃描**(4 臂) |
| C3 | FT-only(`L_IO` 關,只留 `κ_ft·L_FT`) | adaptec1 k16 | 隔離 FT 項的貢獻 |
| C4 | S4a 星形和變體 | adaptec1 k16 | 驗證 §3.2 的離線裁決在 in-loop 也成立 |
| C5 | S4c bbox/cov 變體(spec §5.3 原案) | adaptec1 k16 | spec §8 要求的 `cov_k` 變體比較 |
| C6 | `β` 取 {0.25, 1.0} 加上 `β` 退火 | adaptec1 k16 | L4 |
| C7 | `home` 重算頻率 {50, 200} | adaptec1 k16 | L5 |
| C8 | 加上 S7(`κ_cap` 取 {0.5, 1, 2}) | adaptec1 k16 | 條件臂(§4.3 gate) |
| C9 | 最佳配置 乘上 {k=8, k=32, slicing} | adaptec1 | 泛化 |
| C10 | 最佳配置 加 bigblue4 `ρ*` | bigblue4 k16 | 規模 |

報表欄位 = M2 §8.2 的欄位 **加上 `io_rg`、`ft_rg`、`mst_excess`、`kappa_ft`、`beta`、`kappa_cap`、`demand_max_over_med`、`region_util_max_over_mean`、`spearman_ft`**(per-net `ft_rg` vs soft `ReLU(R−λ+1)`)。

時間預算:adaptec1 單臂約 2 分鐘(M2 實測 110–140 秒;M3 的額外成本只有 3 個 `(E,)` 累加器與 evaluator 的 Steiner 計算)⇒ C2 到 C9 約 20 臂約 45 分鐘;bigblue4 ρ 校準 4 臂加主臂約 1.5 小時。**全部低於 3 小時 GPU。**

### 6.4 bigblue4 per-case ρ(M2 遺留,列為 T5 前置實驗)

M2 報告附註記載:bigblue4 直接套 adaptec1 的 `ρ*=0.40` 得 `Δio=−36.5%` 但 `Δhpwl=+5.41%`,超出 iso-WL 帶。**T5 = 在 bigblue4 k16 grid 上掃 `ρ_max` 取 {0.05, 0.10, 0.15, 0.20}(annealed τ、`κ_ft=0`),取滿足 `Δhpwl ≤ +2%` 的最大 `ρ`。** 先驗預估 `ρ*` 約 0.10 到 0.15:M2 §5.2 的 λ 錨點(adaptec1 1.52e3、bigblue4 2.79e3)顯示同 ρ 下 bigblue4 的 `λ_io` 大 1.83 倍,而 `Δhpwl` 在 ρ=0.40 時是 adaptec1 的 4.1 倍。**所有 bigblue4 的 M3 臂都必須跑在 `ρ*` 上**,否則 E3/E4 沒有意義。

---

## 7. Q6 — 風險與 fallback(gate 制,仿 M2 §9.1)

| ID | 量測 | 門檻 | 觸發後動作 |
|---|---|---|---|
| **G1** FT 梯度死區 | 退火末端 `L_FT` 梯度 L1 除以 `L_IO` 梯度 L1(每 callback) | **低於 0.05** | `β` 補掃(C6);若仍低於 0.05 → 切 **S4c bbox 變體**(梯度不經飽和 softmax,走 WA soft bbox,是 G1 的天然 fallback) |
| **G2** alignment | GP 末端 Spearman(per-net `ft_rg` vs soft `ReLU(R−λ+1)`) | **低於 0.30** | 目標與指標脫鉤 → 切 S4c;若 S4c 也低於 0.30 → 回退到「只報 FT、不優化 FT」並誠實記錄 |
| **G3** **RG 模型被鑽漏洞** | `ft_rg` 下降 10% 以上但 `ft_mst` 上升超過 +3%(超過 3σ_seed) | 任一臂 | objective 在吃 region-graph 的幾何樂觀性 ⇒ 啟動 **T11 maze 抽樣校驗**;若校驗確認,改用 bounded-detour 變體(把 `D` 換成「限制在 net bbox 膨脹 δ 內」的距離) |
| **G4** IO 與 FT 互相打架 | `κ_ft` 增大時 `io_mst` 單調惡化,且在 `κ_ft=0.5` 就超過 E3 | — | 兩項不相容 ⇒ 退回 `κ_ft` 不超過 0.5 的小權重區,把 M3 定位成「在 M2 的 Pareto 前緣上多一個維度」而非取代 |
| **G5** `home` 抖動 | 相鄰 callback 間 `home_e` 改變的 net 比例 | **超過 20%** | 提高重算頻率(C7)或改用 pin 質心 region;同時檢查 backtrack 中位數是否達到 5(secant 汙染) |
| **G6** 規模契約 | 10M 合成拓撲的 chunked fwd+bwd 峰值(**沿用 M2 T2 Step 6 的 spike,加 FT 項重跑**) | **超過 8 GB** 或 OOM | 不得凍結 T2 介面;先砍 `(E,)` fp64 累加器為 fp32(需重測數值)或加大 chunk 數 |
| **G7** 穩定性 | 排程 iteration 的 `obj_eval_count` 增量中位數達到 5,或 `Δhpwl` 超過 +10%,或 `check_divergence` | 任一 | 檢查 §5.3 的事件順序與 Lipschitz 護欄;降 `κ_ft`/`ρ_max` |

**額外風險(無獨立 gate,但必須在報告記載):**

| # | 風險 | 證據 | 緩解 |
|---|---|---|---|
| R1 | S4b 系統性**低估** Λ≥4 net 的 FT(bucket 份額比 0.11–0.35) | §3.2 表 | 這些 net 只佔真實 `ft_rg` 的 1.2–4.5%;報表逐 bucket 追蹤,若 k=32 下 L4+ 的真實份額升到 10% 以上則重審 |
| R2 | **evaluator 的 `(E,K)` int64 累加器**(`ioplace/evaluator_gpu.py:352` 的 `pin_bit_acc`、`:413` 的 `passed_bit_acc`)在 12M net 乘 K=32 是 **3.0 GB 各一份** | 讀碼 | 這是 **M2 就存在**的規模斷點,M3 的 Steiner 擴充會再加一份 terminal 列表 ⇒ T1 必須改成直接累加 packed int64 bitmask、或按 net 分塊。**列為 T1 的驗收條件之一** |
| R3 | `ft` 是小整數計數(adaptec1 約 2.5k),相對噪聲比 io 大 | `σ_seed(ft)/mean = 0.83%` vs io 的 0.59% | exit 門檻 E5 用 3σ;最佳臂重跑 3 個 seed |
| R4 | 雙溫度(`τ` 與 `β`)交互 | 未量 | C6 掃描加 P4 探針;預設 `β` 不退火,先固定一個變因 |
| R5 | `κ_ft` 讓 objective 偏離 M2 已驗證的良好區 | M2 的 `ρ*=0.40` 是在 `κ_ft=0` 下調出來的 | T6 的 C2 掃描固定 `ρ_max=ρ*` 只動 `κ_ft`;若最佳 `κ_ft` 落在掃描邊界,再做 `ρ` 乘 `κ` 的 2D 補掃(9 臂) |
| R6 | S7 的 hinge 在啟用初期梯度恆 0,ratio 正規化會除以 0 | 結構性 | §5.2 已禁止 M2 式正規化,改用「相對 IO 項梯度」的解法,並在 cap 梯度為 0 時凍結 `λ_cap` |
| R7 | slicing(rectilinear)下兩把 FT 尺分歧最大(`ft_rg/ft_mst` = 0.73–0.75、ρ=0.59) | §2.2 表 | slicing 的 exit 只用 `ft_mst`(保守尺);`ft_rg` 只作診斷,並在報告明寫 rectilinear 是 G3 最可能觸發的組態 |

---

## 8. Q7 — Task 分解草案

規約沿用 M2 §11:每個 task 一次 TDD 迴圈(先寫失敗測試,再實作,跑綠,commit);測試一律用 `$DP/.venv312/bin/python -m pytest`;新程式碼全在 `ioplace/`;DP 改動只允許既有的 python-only patch(**M3 不需要新 patch**)。

| Task | 類型 | 內容 | 驗收 |
|---|---|---|---|
| **T0** 探針遷入 | 純軟體 | 把本文 6 個探針遷入 `ioplace/diagnostics/probes_m3/`(P1 L 形約定敏感度、P2 `home` 抖動率、P3 邊界需求/容量、P4 `β` 與 `τ` 的梯度掃描、P5 RG vs maze 抽樣、P6 free-area 利用率),輸出 `results/m3/probes/*.json` 加 env metadata 加輸入 sha256;`tests/test_probes_m3_regression.py` 用小型合成 case 鎖住恆等式 `ST_e − (Λ_e−1) = FT_e ≥ 0` 與 `Λ_e−1 ≤ ST_e ≤ io_mst` | 6 個 JSON 產出;本文 §2.2/§3.2/§4.1 的每個數字可追溯;不符即更新本文 |
| **T1** evaluator RG 擴充 | 純軟體 | `EvalResult` 新增 `io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home`;`evaluator_ref`(brute-force 參考)與 `evaluator_gpu`(Λ≤3 closed form、Λ 在 4 到 8 批次 Dreyfus–Wagner、Λ 大於 8 metric-closure MST 加上界標記)同步;`region_graph(rg)` 回傳 `(adj, D, ell)`,放 `ioplace/region_graph.py`;**順帶修 R2 的 `(E,K)` int64 累加器** | 舊欄位逐位元不變(既有等價測試);新欄位 CPU/GPU 等價;小 case 對 brute-force Steiner 全對;恆等式測試;bigblue4 上 evaluator runtime 增量低於 +30% |
| **T2** S4 op | 純軟體 | 把 `ioplace/ops/io_term.py::IoTerm`/`_IoFn` 擴成 `IoFtTerm`:FWD-2 內多累加 `N_e`/`M_e`,backward 的 `c_{e,k}` 併入 FT 係數;`home`/`D`/`ecc_max` 為 buffer;**參考實作 `IoTermRef` 同步擴充**作為等價測試對照 | 玩具 case 手算 `R`/`L_FT`;fp64 `gradcheck`;chunked 與 unchunked 一致(fp64 1e−10);**`κ_ft=0` 時與 M2 逐位元相同(回歸鎖)**;`τ` 與 `β` 同時趨近 0 時 `L_FT` 收斂到 hard `ft_rg`(相對誤差低於 5%);`pos[num_movable:]` 梯度恆 0;`β` 小於 0.1 時 raise;**10M spike 重跑(G6)** |
| **T3** schedules 擴充 | 純軟體 | `ScheduleState` 新增 `kappa_ft`/`beta`/`kappa_cap`/`of_cap`/`home_version`;合併項的 `update_ratio`;多項 Lipschitz 護欄;`λ_cap` 的解法與零梯度凍結規則 | 純函數測試:端點、單調性、`κ_ft=0` 退化、`obj_version` 只在離散事件遞增、`λ_cap` 在零梯度時不變 |
| **T4** driver | 純軟體 | `ioplace/drivers/run_placement_io.py` 新增 `--kappa-ft --beta --kappa-cap --of-cap`;callback 內加 `home` 更新與 §5.3 的強制順序;trajectory 新增 `io_rg`/`ft_rg`/`ft_grad_l1`/`demand_max_over_med`/`region_util` | JSON schema 齊全;`κ_ft=0` 的 run 與 M2 存檔逐位元相同;事件順序測試;同 iteration 內 `obj_fn` 三次常數性;version invariant 全程零違反 |
| **T5** bigblue4 ρ 校準 | **實驗** | §6.4 的 4 臂 ρ 掃描(`κ_ft=0`),定出 `ρ*` | 存在 `Δhpwl ≤ +2%` 的臂;寫進 M3 報告第一節 |
| **T6** κ_ft 掃描 | **實驗** | C2(4 臂)加 C3 加 C1 對照;機械判定 G1 到 G5、G7 與 §4.3 的 S7 gate | Pareto 表加圖(x 軸 Δhpwl%、y 軸 Δft%,標 flat/M1/M2/M3 前緣與 3σ_seed 帶);選出最佳 `κ_ft` |
| **T7** ablation | **實驗** | C4/C5/C6/C7/C9/C10 | 每格有 JSON;per-Λ-bucket 診斷表 |
| **T8** 報告 | 文件 | `docs/results/m3-differentiable-ft-report.md`:regime,ρ 校準,κ 掃描,四方對照(flat/M1/M2/M3),ablation,雙尺 FT 加三分解表,G1 到 G7 判定,**對 spec §9 M3 兩個 exit 條件逐條打勾或打叉** | 每個數字可追到 `results/m3/*.json`;FAIL 就照 M1/M2 慣例誠實記錄 |
| **T9**(條件:§4.3 gate) | 純軟體加實驗 | S7 實作(`P[h,k,ab]` 預算、`(K,K)` 累加器、fused backward)加 C8 掃描 | 需求均勻度改善且 `ft`/`io`/`hpwl` 不退出 exit 帶 |
| **T10**(條件:G1/G2) | 純軟體加實驗 | S4c bbox/cov 變體 op 加掃描 | 與 S4b 在同一張 Pareto 圖比較 |
| **T11**(條件:G3,或 v2 審查要求) | 純軟體加實驗 | maze 抽樣校驗器(1 萬條 net,region-id lattice 上 crossing=1、length=ε 的 Dijkstra),量 RG 模型的樂觀誤差分布 | 中位偏差報告;超過 20% 則觸發 bounded-detour 變體設計 |

### 依賴序

```
T0 -- T1 -- T2 --+-- T4 -- T5 -- T6 --+-- T7 -- T8
                 |                    |
                 +-- T3 --+           +-- T9   (條件:§4.3 gate)
                                      +-- T10  (條件:G1/G2)
                                      +-- T11  (條件:G3)
```

- **T0 必須最先**(M2 的 M3-finding 慣例:數值依據要先在 repo 內可重現)。
- **T1 必須在 T2 之前**:`home_e` 由 evaluator 供給、`D` 表由 `region_graph` 供給,而且 `ft_rg` 是 T2 收斂測試的對照真值。
- **T2 的 10M spike(G6)是 T4 的前置門檻**(沿用 M2 的介面凍結規則)。
- **T5 必須在任何 bigblue4 臂之前**(否則 E3/E4 無意義)。
- T3 與 T4 可並行,兩者皆完成才進 T5。

---

## 9. 低信心段落總表(v2 對抗性審查請優先攻擊這裡)

| ID | 段落 | 不確定的是什麼 | 由誰解決 |
|---|---|---|---|
| L1 | §2.4 | RG 模型的幾何樂觀性(允許無成本繞路) | P5 / T11 |
| L2 | §2.4 | Λ≥4 的 metric-closure MST 上界鬆緊度 | T1 的精確 Dreyfus–Wagner |
| L3 | §2.4 | `ft_mst` 對 L 形約定的敏感度(§2.3 對 MST 的否決理由之一目前是**推理**不是實測) | P1 / T0 |
| L4 | §3.4 | `β` 的可用窗口、與 `τ` 的交互 | P4 / C6 |
| L5 | §3.4 | `home_e` 凍結 50 iter 的代價 | P2 / C7 |
| L6 | §3.4 | `N`/`M` 累加的數值下限 | T2 的 `β` assert 測試 |
| L7 | §4.1 | region 利用率的分母應為 free area | P6 / T0 |
| L8 | §5.2 | 多項 Lipschitz 護欄的形式(除以 `max(1, g_comb/g_io)`)是**啟發式**,與 M2 的 `τ²/γ` 同一等級的猜測 | T6 的 G7 監控 |
| L9 | §6.2 | E2 的 0.85 倍係數缺乏第一性原理,是對照 M1 在 adaptec1 只做到 −4.5% 而設 | v2 可依 T5/T6 的實測前緣調整 |
| L10 | §4.2 | `P[h,k,ab]`(最短路徑的邊界佔用比例)在多條最短路徑時取均分,是**假設**不是模型結論 | T9 開工前用 evaluator 的實際 `boundary_pair_demand` 校驗 |
| L11 | 全文 | S4b 只在**離線的最終 placement** 上驗過;GP 過程中(overflow 高、cell 尚未展開)`home_e` 與 `q` 的品質未知 | T6 的 trajectory 診斷 |

---

## 10. 證據附錄(T0 待遷入)

**狀態:下表數字皆為本次實跑,但腳本目前在 `/tmp`。在 T0 完成之前一律視為「待重現」。**

| 探針(暫存路徑,箭頭後為目標檔名) | 內容 | 支撐 |
|---|---|---|
| `/tmp/probe_m3_rg.py` → `probe_rg_steiner.py` | 對 M2 存檔 placement 算 region graph、all-pairs `D`、per-net Steiner(Λ≤3 精確、Λ≥4 metric-closure MST)、`io_rg`/`ft_rg`、`ft_rg` 對 `ft_mst` 的 Spearman、boundary demand 除以長度 | §2.2 全表、§4.1 需求表 |
| `/tmp/probe_m3_bb.py` → `probe_rg_bigblue4.py` | 同上,跑 bigblue4,加 per-Λ-bucket 分解 | §2.2 bigblue4 三列、§3.2 的「Λ=2 佔 87–97%」 |
| `/tmp/probe_m3_surrogate.py` → `probe_ft_surrogate.py` | S4a/S4b/S4c/S4d 四候選對精確 `ft_rg`:總量比、Spearman、per-Λ-bucket 份額比 | §3.2 全表(**本文最關鍵的裁決依據**) |
| `/tmp/probe_m3_util.py` → `probe_region_util.py` | per-region 面積利用率(A0 對 A2) | §4.1 的「面積容量不需要項」 |
| (新增)`probe_l_convention.py` | P1:L 形反向後的 `ft_mst` 變動 | L3 |
| (新增)`probe_home_churn.py` | P2:`home_e` 在 callback 間的改變率 | L5 |
| (新增)`probe_beta_tau.py` | P4:`L_FT` 梯度 L1 對 `(β, τ)` 的曲面 | L4 |
| (新增)`probe_maze_sample.py` | P5:RG 對 maze 的樂觀誤差 | L1 / G3 |
| (既有)`results/m2/noise/flat_seed100x.json` | `σ_seed(ft) = 20.98`(0.83%) | §6.1 |

**與 M2 資產的關係(給 v2 審查者的快速索引):** S1(`ioplace/ops/soft_assign.py`)、chunked 四趟結構(`ioplace/ops/io_term.py::_IoFn`)、schedule 與 version invariant(`ioplace/schedules.py`)、secant refresh(`ioplace/dp_hook.py`)、driver callback(`ioplace/drivers/run_placement_io.py:102-172`)全部原樣沿用;M3 的所有新機制都是這五個檔案的**加法**,沒有任何一項要求改寫 M2 已驗證的路徑。
