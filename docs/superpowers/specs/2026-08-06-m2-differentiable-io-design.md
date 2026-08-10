# M2 設計:可微 IO 項(S1 軟歸屬 + S2 乘積式 crossing surrogate)

- 日期:2026-08-06
- 狀態:**v2(Codex 對抗性審查後修訂)**——供 writing-plans 轉成實作計畫
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §5.1/§5.2/§5.5、§9 M2 列、§10
- 前置結論:`docs/results/m1-reweight-report.md`(M1 exit 條件 2 FAIL:ft −9.9%~−49.1% 但 io 持平)
- 本文所有數值證據皆為在 L4 上以 `$DP/.venv312/bin/python` 實跑量測,非推估;探針腳本由 **T0** 提交進 repo 後方可被引用為設計依據(見 §13)。

---

## 修訂紀錄(v2)

Codex(GPT 家族)對抗性審查回報 5 HIGH + 3 medium,scheduler 逐條裁決。以下為 v1 → v2 的逐條落點。

| 發現 ID | 裁決 | 修訂內容 | 落點小節 |
|---|---|---|---|
| **H1** objective 排程汙染 Nesterov secant | 接受 | (a) τ/ρ 改為**每 iteration 連續漂移**(與 DP 自身 γ/μ 同級的小漂移);(b) 任何**離散**變更(啟用、`ratio` 重算、`w_e` 校準)後**強制**呼叫 `refresh_nesterov_secant()` 讓 `g_k`/`g_k_1`/`alpha_k` 在新 objective 下重算;(c) 新增 patch 暴露 `self.optimizer`/`self.model`;(d) T4 加「無跨 objective secant pair」的 RED/GREEN 測試 + 全程 invariant assert。**新增 `probe8_secant.py` 實測:不刷新時步長塌陷 11,000×** | **新 §6.4**、§4.2、§5.2、§11 T4 |
| **H2** degree skew 無安全論證 | 部分接受 | (a) `D_max` **預設改為 `2 ≤ deg < params.ignore_net_degree`(=100)**,與 DP 的 `net_mask_ignore_large_degrees` 逐位元同義——徹底消除「有 IO 力卻無 WL 反作用力」的 net;`D_max=256` 降為 ablation A6。(b) `w_e=1` 維持預設,但 **A4 升為必跑且 adaptec1 + bigblue4 都跑**;T7 軌跡新增 per-degree-bucket 的梯度佔比與精確 io 貢獻診斷 | §3.1、**§3.2.1 改寫**、§8.1、§11 T2/T7 |
| **H3** F4 被退火混淆 / 74–84% 表述過強 | 全盤接受 | (a) 每次 callback 由 evaluator 算 **hard λ**(`hard_lambda_sum`/`per_net_lambda`,新欄位),F4 改定義在 **hard-λ delta vs 精確 io delta**;(b) soft 診斷一律在**固定參考 τ** 評估;(c) GP 末端加 per-net rank correlation;(d) F2 措辭改為「所有 ρ×τ 掃描臂(目前 10 組)」;(e) 74–84% 改稱「**份額(share)**」並註明非解釋變異、非因果 | **§3.2.5 改寫**、**§9.1 改寫**、§11 T5/T7 |
| **H4** margin 符號反了 + LG 排序 | 全盤接受(**v1 是實錯**) | (a) margin 改為 `softplus((d+m)/τ_m)`(d 為 signed distance,內部為負);(b) **T2 就實作** margin 項(flag 預設關、ε=0)+ 有限差分方向測試;(c) T7 加**第 11 臂 margin pilot**,LG-retention 證據隨掃描一起出;(d) F2 的 GP 判定改用 `io_gp`,LG 流失獨立為 **F5**;(e) T11 降為「pilot 有效才做的深掃」 | **§9.2 新增**、§10 R4、§8.1、§11 T2/T7/T11 |
| **H5** 規模契約與目標規模不相容 | 接受 | T2 的 `autograd.Function` **第一天就走 chunked-k**,forward/backward 各兩趟、**絕不 materialize `(N,K)`/`(P,K)`**,常駐狀態只有 `m(N,)`、`s(N,)`、`λ(E,)`;chunked vs unchunked 等價測試;新增 **10M 級合成拓撲 feasibility spike**(T2 Step 6) | **新 §2.5**、§2.3 改寫、§10 R5、§11 T2 |
| **M1** 依賴未言明的 optimizer lock | 接受 | (a) T5 driver **assert** `optimizer=="nesterov"`、`use_bb==0`、`Lsub_iteration==1`;(b) T4 加「obj 值加常數擾動 ⇒ iteration 軌跡完全不變」的回歸測試;(c) §3.2.4 的「STE 無定義」措辭**收回**——`hard.detach()+soft−soft.detach()` 是標準構造,真正理由是成本 + 鎖定組態下無效 | **§3.2.4 改寫**、§11 T4/T5 |
| **M2** class-level monkey-patch 汙染 | 接受(直接跳到 patch) | **完全放棄 monkey-patch**。改為 `params` 承載的 per-instance patch:`PlaceObj.__init__` 讀 `params._extra_obj_terms`、`obj_fn` 尾端累加;python-only patch + `install/` 同步(M1 precedent,無 cmake rebuild)。`dp_hook.py` 降為 helper | **§6 全節改寫** |
| **M3** 關鍵數值證據不可重現 | 接受 | 新增 **T0**(M2 branch 第一個 task):8 個探針(**含修好的 probe3 + 新增的 probe8**)+ runner + env metadata + 輸入 hash + raw JSON 提交進 repo;從提交版重跑校驗本文數字 | **新 §11 T0**、§13 改寫 |
| *scheduler 補充 1* | — | T6 的 σ_rep 在 `deterministic_flag ∈ {0,1}` 兩個 regime 各量 5 次;若 flag=1 把 σ_rep 壓到可忽略且 runtime 代價 <20%,M2 全臂改用 flag=1 並在同 regime 重測 flat 基準 | **§7.2 改寫**、§11 T6 |
| *scheduler 補充 2* | — | exit =(`Δio ≤ −5%` ∧ `>3σ_rep` ∧ `Δhpwl ≤ +2%`);**F2 只在無任何臂於 `io_gp` 基準達 `Δio ≤ −3%` 時觸發**;−3%~−5% 灰帶 = 調參問題(τ 補掃 + margin 深掃),不換路線 | §7.1、**§9.1 改寫** |

**未修訂(刻意保留)**:§2.3 / §4.1 / §3.2.5 的實測數值表逐字保留;§2.4 對「grid 專用可分離 fast path」的否決維持。

---

## 0. 摘要(定案一覽)

| # | 問題 | 定案(v2) |
|---|---|---|
| Q1 | S1 數學形式 | `p_{i,k} = softmax_k(−d_k/τ)`;`d_k` = **L1(Manhattan)box signed distance**,rectilinear 取成員矩形 hard-min。以 **node**(非 pin)為單位。τ 以 `L_R = sqrt(A_die/K)` 正規化。**運算子契約自第一天即為 chunked-k,不暴露 `(N,K)`/`(P,K)`**(§2.5) |
| Q2 | S2 精確形式 | `q_{e,k} = 1 − exp(Σ_{i∈e} log(1−p_{i,k}))`、`L_IO = Σ_e w_e·max(Σ_k q_{e,k} − 1, 0)`;**`w_e` 預設 = 1**(明示偏離 spec,A4 必跑裁決);**net 過濾與 DP 的 `net_mask_ignore_large_degrees` 逐位元同義(`2 ≤ deg < ignore_net_degree` = 100)**;per-net node 去重;手寫 backward 用 log-space leave-one-out `exp(S−ℓ_i)`——無 guard 的 naive autograd 在真實座標上產生 **193,798 個 NaN** |
| Q3 | τ 退火 | 不綁 DP 的 γ,但沿用其函數形式:τ 由 **overflow** 驅動、log-linear 從 `τ_rel=0.30` 退到 `0.03`,且**每 iteration 連續更新**(H1)。`‖∇L_IO‖₁` 實測對 τ 單峰,峰在 τ_rel≈0.1 |
| Q4 | auto-normalization | `λ_io = ρ_io(overflow)·‖∇WL‖₁/‖∇L_IO‖₁`,`ratio` 每 N=50 iter 重算(離散事件 → 必須配 secant refresh)、`ρ_io(of)` 每 iteration 連續漂移;EMA 阻尼 + Lipschitz 護欄 `λ_io ≤ τ²/γ`;啟用時 20 iter 線性 ramp-in。錨點:ρ=1 → adaptec1 k16 λ_io≈1.5e3、bigblue4 k16 λ_io≈2.8e3 |
| Q5 | DP 整合點 | **`params` 承載的 per-instance patch**(`params._extra_obj_terms`),python-only、無全域變異、無需 uninstall、天然 exception-safe;**monkey-patch 方案已完全放棄**。另加一行 patch 暴露 `self.optimizer`/`self.model`(H1 需要)。選項 B(M1 callback 加梯度)結構上不可行 |
| Q6 | iso-WL 協定 | 掃 `ρ_max` 產生 (Δhpwl%, Δio%) Pareto 前緣;exit =(`Δio ≤ −5%` ∧ `>3σ_rep` ∧ `Δhpwl ≤ +2%`);**T6 先量 σ_rep(deterministic_flag 0/1 兩 regime)**;既有 `--seed` 在 grid 模式對 placement 無作用 → 必須加 `--dp-seed` |
| Q7 | ablation | A0 flat / A1 reweight-only / A2 可微-only / A3 疊加 / **A4 `w_e` 變體(必跑,兩 case)** / A5 τ 固定 / A6 `D_max=256` / **A7 margin pilot**;表格新增 `io_gp`、`hard_lambda_sum`、per-degree-bucket 診斷 |
| Q8 | fallback 觸發 | **F1** 梯度覆蓋率 <5%;**F2** 所有 ρ×τ 臂在 **`io_gp`** 基準皆未達 `Δio ≤ −3%` → 換 surrogate(S3);**F3** 不穩定;**F4** hard-λ 降 ≥10% 但精確 io 降 <3%(或 `io_gp` 反升 >3σ_rep);**F5** LG 流失惡化 >2% → **獨立治理(margin 深掃),不換 surrogate** |
| Q9 | 風險 | 梯度死區(τ_rel=0.02 時 45.8% cell 無 IO 梯度)、退火敏感、**Nesterov secant 汙染**、記憶體(chunked 契約後 10M 級 ≤6GB 目標,由 spike 驗證)、**LG 抹平 GP 收益(實測 +6.8%~+9.9%)**、fixed macro 必須零梯度、`params` 序列化污染 |
| Q10 | task 分解 | **10 個核心 SDD task(T0→T9)+ 2 個條件 task(T10/T11)**,依賴序見 §12 |

**一句話結論:** M1 診斷正確——WL 加權對「離散 region 歸屬」沒有槓桿;S1+S2 提供的正是那個缺失的槓桿,而且實測顯示 **hard `Σ(λ−1)` 佔了實際 `io_count` 的 74–84% 份額**(§3.2.5),所以 S2 在原理上「打得到」M2 的 exit 指標。v2 之後,主要不確定性收斂為 **(a) 退火窗口是否夠寬、(b) legalization 會不會把 GP 的收益吃掉(已升格為 T7 內的第 11 臂,不再排在掃描之後)**。

---

## 1. 名詞與符號

| 符號 | 意義 | 來源/大小 |
|---|---|---|
| `N` | movable + fixed 的 physical node 數 | adaptec1 211,447(movable 210,904);bigblue4 2,177,353 |
| `P` | pin 數 | adaptec1 919,701;bigblue4 8,731,365 |
| `E` | net 數 | adaptec1 221,142;bigblue4 2,229,886 |
| `K` | region 數 | 8/16/32(≤64,evaluator bitmask 限制) |
| `R_k` | region k = 軸對齊矩形的聯集(`RegionSpec.rects`,`ioplace/regions.py:6`) | grid/slicing 皆單矩形 |
| `d_k(x,y)` | 點到 `R_k` 的 **L1 signed distance**(內部為負) | §2.1 |
| `d*_i` | `min_k d_k(x_i,y_i)`,到「自己 region」邊界的 signed distance | margin 項用 |
| `τ` | 軟歸屬溫度(長度單位) | `τ = τ_rel · L_R` |
| `L_R` | region 特徵尺度 `sqrt(A_die/K)` | adaptec1 k16 = 2,671.5;bigblue4 k16 = 8,064.7 |
| `p_{i,k}` | node i 屬於 region k 的軟機率 | 概念上 `(N,K)`,實作**不 materialize** |
| `ℓ_{i,k}` | `log(1 − p_{i,k})` | 同上 |
| `S_{e,k}` | `Σ_{i∈e} ℓ_{i,k}`(去重後的 node) | 概念上 `(E,K)`,實作只留 `λ (E,)` |
| `q_{e,k}` | `1 − exp(S_{e,k})`,net e 觸及 region k 的機率 | — |
| `λ_e` | `Σ_k q_{e,k}`,**soft** 期望觸及 region 數 | `(E,)` |
| `Λ_e` | **hard** 觸及 region 數 = `popcount(pin_bm_e)`,由 evaluator 給 | `(E,)`,H3 新增 |
| `L_IO` | `Σ_e w_e·max(λ_e − 1, 0)` | scalar |
| `λ_io` | objective 中 `L_IO` 的權重 | auto-normalized |
| `ρ_io` | **無量綱** IO/WL 梯度比目標值(真正的可掃旋鈕) | 0.02–0.4 |
| `γ` | DP 的 WA-WL smoothing gamma | `10·params.gamma·(bin_x+bin_y)` 起始,隨 overflow 退火 |
| `of` | DP 的 density overflow | 1.0 → `stop_overflow`=0.07 |
| `io_gp` | **GP 最後一次 callback**(legalization 之前)的精確 `io_count` | H4 新增欄位 |
| `io_count`/`ft_count` | **ground truth**,一律取 `evaluator_ref.evaluate` | 不動語意 |

**硬性分工(不可違反):** `ioplace/evaluator_ref.py` 與 `ioplace/evaluator_gpu.py` 是唯一的計分尺,M2 **不改其既有輸出的語意或數值**;H3 只**新增** `hard_lambda_sum`/`per_net_lambda` 兩個輸出欄位,既有欄位必須逐位元不變(由既有等價測試把關)。`L_IO` 只是 optimizer-facing surrogate,任何 exit 判定、報表數字都回 `evaluator_ref`。

---

## 2. Q1 — S1 軟歸屬的精確數學形式

### 2.1 定案

```
# 單一矩形 R = [xl, xh] × [yl, yh] 的 L1 signed distance
dx = max(xl − x, x − xh)          # >0 表 x 在區間外
dy = max(yl − y, y − yh)
d_rect(x,y) = max(dx,0) + max(dy,0) + min(max(dx,dy), 0)
#             \_____ 外部 L1 距離 ____/   \__ 內部(負)= −到最近邊的距離 __/

# region = 矩形集合
d_k(x,y) = min_{r ∈ rects(k)} d_rect_r(x,y)        # hard min(次微分,a.e. 可微)

# 軟歸屬
z_{i,k} = − d_k(x_i, y_i) / τ
p_{i,·} = softmax_k(z_{i,·})
```

**為何用 L1 而不是 Euclidean:** 全案(HPWL、MST、evaluator 的 L-shape walk)都是 Manhattan 幾何;L1 box distance 的梯度是 ±1/0 的分段常數,與 DP 的 WL 梯度同一個非光滑等級,不引入新的病態。Euclidean 的 `sqrt` 在角落附近有 `1/r` 奇異點,無好處。

**為何 hard-min 而非 soft-min:** rectilinear region 的成員矩形彼此相鄰(同一 region 的矩形聯集是連通的),`min` 的不可微集合落在矩形內部的「等距線」上,離 region **邊界**很遠——而梯度只在邊界附近才有量級(見 §2.3),所以 hard-min 的 kink 實際上永遠不在作用區。grid/slicing(單矩形)下 hard-min 是恆等式。

**單位是 node 不是 pin:** `p` 只依賴 node 座標(pin offset 對 region 歸屬的影響 ≪ lattice cell)。用 node 有三個好處:(a) 張量從 `(P,K)` 降到 `(N,K)`;(b) 自然解決 DP 警告的「同 node 多 pin」問題(adaptec1 20,285 nets / bigblue4 135,260 nets 有 same-node pins),否則 `Π_{i∈e}(1−p_i)` 會把同一顆 cell 算兩次;(c) 梯度直接落在 `pos` 上,不需要經過 `pin_pos` op。

### 2.2 數值安全形式(**這是必須照抄的實作細節**)

直接 `torch.log1p(-torch.softmax(...))` 在 fp32 下會產生 `−inf`,再與 `exp(S)=0` 相乘產生 **NaN**。定案的穩定寫法(逐 k-chunk 執行,見 §2.5):

```python
z = -sdf / tau                                   # (N, c) — c = chunk 內的 region 數
m, am = z_full.max(dim=1)                        # 全 K 的 max(pass 1 得到,(N,))
e = torch.exp(z - m.unsqueeze(1))                # e[argmax] == 1.0
# s = 1 + t,  t = Σ_{j≠argmax} e_j  由 pass 1 以 k-chunk 累加得到,**無抵消誤差**
p   = e / s.unsqueeze(1)
omp = torch.where(is_argmax, (t / s).unsqueeze(1).expand_as(e), (s.unsqueeze(1) - e) / s.unsqueeze(1)).clamp_min(1e-30)
ell = torch.log(omp)                             # = log(1 - p),下限 −69
```

關鍵在 `t`:對 `k = argmax`,`1 − p_k = t/(1+t)` 是**直接算出來的**,不是 `s − e_k` 的災難性相減;對 `k ≠ argmax`,`(s − e_k)` 只在兩個 logit 幾乎打平時才有相減,而那時 `1−p_k ≈ 0.5`,沒有精度問題。`t` 必須以「排除 argmax 的 k-chunked 加總」求得,**不可**寫成 `s − 1`。

### 2.3 記憶體與梯度成本(實測 — **v1 的 materialized 原型,保留為對照基準**)

bigblue4 真實 netlist + 真實最終座標,**純 torch materialized 原型**(即 §2.5 要取代的寫法)前向+反向一次:

| K | fwd | bwd | peak GPU mem |
|---:|---:|---:|---:|
| 8 | 124.6 ms | 132.0 ms | 2,431 MB |
| 16 | 53.7 ms | 108.2 ms | 4,696 MB |
| 32 | 101.7 ms | 204.3 ms | 9,229 MB |

adaptec1 K=16:fwd 3.8 ms / bwd 8.7 ms / peak 409 MB。

- 峰值幾乎全部來自 autograd 保存的 `(P,K)` 中間張量。**這正是 H5 指出的規模斷點:同一寫法外推到 N=10M / P≈40M / K=32 是 ~42 GB,在 placer + evaluator + line-search 暫存之前就已超過 L4、逼近 H100。**
- **v2 的處置:不把 materialized 版本當交付形態。** T2 從第一天就實作 §2.5 的 chunked-k 契約,上表僅作為 (a) 等價測試的參考實作(小 case)與 (b) chunked 版效能/記憶體的對照基準。
- 每 iteration 成本:`obj_and_grad_fn` 在 `step_nobb` 的 backtracking line search 內被呼叫 **1–10 次**(`NesterovAcceleratedGradientOptimizer.py:128`;ISPD2005 無 movable macro → `use_bb` 由 `"auto"` 解析為 **0**,`PlaceDB.py:837`)。以平均 2 次估:adaptec1 GP 約 +25 ms/iter(≈ +1.8× GP 時間),bigblue4 約 +324 ms/iter。chunked 版因為要重算 SDF,forward+backward 的 FLOP 約 ×2,但仍是 memory-bound,預期 wall-time 增幅 <30%。

**chunked 版(T2b Step 5 實測,`IoTerm`,`chunk_budget` 預設 8e6,`results/m2/probes/chunked_perf.json`):**

bigblue4 真實 netlist + 真實最終座標,`IoTerm` 前向+反向一次:

| K | fwd | bwd | peak GPU mem | k_chunk |
|---:|---:|---:|---:|---:|
| 8 | 38.1 ms | 132.9 ms | 796.8 MB | 1 |
| 16 | 74.7 ms | 263.8 ms | 797.1 MB | 1 |
| 32 | 146.9 ms | 524.5 ms | 797.5 MB | 1 |

adaptec1(`k_chunk=8` for all three K, i.e. only K=8 is fully materialized):
K=8 fwd 187.6 ms / bwd 766.7 ms / peak 364.7 MB; K=16 fwd 40.9 ms / bwd 70.7 ms
/ peak 393.2 MB; K=32 fwd 79.7 ms / bwd 138.0 ms / peak 393.3 MB.

- **bigblue4 K=32 peak 797.5 MB vs materialized 9,229 MB — 11.6x 下降,遠低於
  `< 4,000 MB` 的驗收門檻。** `k_chunk=1` for every bigblue4 K (its
  `n_pins_dedup ≈ 9e6` dominates `chunk_budget // max(N, n_pins_dedup)`),
  i.e. bigblue4 is walking the *most* chunked (slowest, most memory-safe)
  regime available at the default budget, and the peak is still <1 GB.
- **wall-time**: chunked bigblue4 K=32 bwd is *slower* than the materialized
  prototype's bwd (524.5 vs 204.3 ms, +157%); fwd is also slower (146.9 vs
  101.7 ms, +44%). adaptec1 (small enough that `k_chunk` only drops to 8,
  not 1) shows the same direction at smaller magnitude. The regression is
  bigger than the "<30%" predicted above, especially for bwd: BWD-1 and
  BWD-2 each recompute `region_sdf_l1` independently (plus FWD-2's own
  pass), so it's three SDF passes per K-chunk, not the one/two implied by a
  flat "FLOP ×2" estimate. The memory win is the point of §2.5, not
  wall-time; M4's fused kernel is where wall-time gets addressed.
- **10M-scale feasibility spike (T2b Step 5, gates T4/interface freeze,
  `results/m2/spike/spike_10m.json`):** synthetic N=10,000,000 /
  n_nets=12,000,000 / n_pins≈49,019,855 (bigblue4-bucket-ratio-resampled
  degree; the nominal 40e6 target undercounts bigblue4's own ~4.1 average
  degree — see `ioplace/diagnostics/spike_10m.py`), K=32, grid 8x4,
  `chunk_budget` default (`k_chunk=1`): **peak 4.11 GB, fwd 3.93 s, bwd
  11.92 s, `ok=true`** (budget: `peak_gb <= 8.0`). Confirms the ~42 GB naive
  extrapolation above is exactly what §2.5's chunking was for -- the actual
  chunked-k peak at this scale is ~10x under the 8 GB gate, with headroom
  before the placer/evaluator/line-search working set is even added.

### 2.4 被否掉的替代方案

| 方案 | 否掉理由 |
|---|---|
| grid 專用的可分離軟歸屬 `p_{i,(a,b)} = softmax_a(−\|x−c_a\|/τ)·softmax_b(−\|y−c_b\|/τ)` | 只需 `(N, nx+ny)` 而非 `(N,K)`,更省;但**違反 D3(Phase 1 即支援 rectilinear)**,且會養出一條只在 grid 有效的程式路徑。**v2 維持否決**——§2.5 的 chunked 契約已經解決記憶體問題,不需要用泛用性去換 |
| 用 `RegionGrid` 的 lattice 做 bilinear soft-lookup | region id 是離散標籤,插值無意義;且會把 512×512 lattice 的解析度變成梯度的人工尺度 |
| Euclidean SDF | 與全案 Manhattan 幾何不一致,角落 `1/r` 奇異 |
| 學一個 assignment 參數 `a_{i,k}`(GAP/Erdős 式)再與位置解耦 | 違反 D1(歸屬必須是位置的函數,才是 placement 的可優化量);且多 `N×K` 個可訓練變數 |

### 2.5 【新增,H5】Chunked-k 運算子契約(第一天即為交付形態)

**契約(T2 的驗收條件之一):`IoTerm` 的任何 forward/backward 路徑,都不得配置任何一個維度為完整 `K` 且另一維度為 `N` 或 `P` 的張量。**

前向兩趟 + 反向兩趟,每趟對 region 以 chunk `C ⊂ {0..K−1}`(`|C| = c`)迴圈,SDF 每趟重算(每個 `(i,k)` 只有 ~8 FLOP,遠比搬記憶體便宜):

```
FWD-1 (reduce):     對每個 chunk 算 z[:,C];維護 m(N,) = running max、以及 argmax 索引
                    第二遍 chunk 迴圈累加 t(N,) = Σ_{j≠argmax} exp(z_j − m)        # 排除 argmax,無抵消
FWD-2 (accumulate): 對每個 chunk 重算 z[:,C] → p,ell;
                    S_chunk(E,c) = index_add(ell[node_of_pin])
                    λ(E,) += Σ_{k∈C} (1 − exp(S_chunk))      # 只保留 (E,) 的 running sum
                    L_IO = Σ_e w_e·max(λ_e − 1, 0)
BWD-1 (reduce):     對每個 chunk 重算 p、S_chunk;
                    c_{i,k} = w_e·1[λ_e>1]·exp(S_{e,k} − ℓ_{i,k})
                    A(N,) += Σ_{k∈C} c_{i,k}·p_{i,k}                     # softmax Jacobian 的耦合項
BWD-2 (scatter):    對每個 chunk 再重算一次;
                    dL/dz_{i,k} = p_{i,k}·(c_{i,k} − A_i)
                    dL/dx_i += dL/dz_{i,k}·(−1/τ)·∂d_k/∂x_i              # 累加進 (N,) 的梯度
```

**forward → backward 之間的常駐狀態只有 `m(N,)`、`t(N,)`、`argmax(N,)`、`λ(E,)`**(加上靜態拓撲)。峰值 = `O(N + E + P) + O((N + P_chunk)·c)`。

- **累加精度:** `S_chunk` 的 `index_add` 用 **fp64 累加器**(`ℓ ∈ [−69,0]`,degree ≤ 99 ⇒ `|S| ≤ 6.8e3`;fp32 在此量級的絕對誤差 ~1e-3,會直接汙染 `exp(S−ℓ)`)。fp64 累加同時大幅降低 atomic 加總順序造成的 run-to-run 抖動(與 R9 的噪聲議題相關)。
- **chunk 大小:** 由可用 VRAM 自動決定(`c = clamp(budget // max(N, P_chunk), 1, K)`),預設 budget 8e6 元素(沿用 `evaluator_gpu._mst_chunk_budget` 的同一心智模型)。`c = K` 時退化為 materialized 路徑,**這正是等價測試的對照組**。
- **10M 級預估:** N=10M、E≈12M、P≈40M、K=32、c=4 → 常駐 `(N,)`×3 + `(E,)` ≈ 0.5 GB;per-chunk `(P,4)` fp32 = 0.64 GB、`(E,4)` fp64 = 0.38 GB → 峰值 **≤ 6 GB**(不含 placer/evaluator)。**由 T2 Step 6 的 feasibility spike 在 L4 上以合成拓撲實測驗證;spike 不通過就不得凍結 T2 介面。**
- **與 M4 的關係:** fused CUDA kernel 是**之後的效能優化**,不是可行性前提——chunked-python 版已經滿足 10M 的記憶體契約,M4 只需把同一個四趟結構融進 kernel,語意與測試完全沿用。

---

## 3. Q2 — S2 IO 項的精確形式

### 3.1 定案公式

```
去重:  對每個 net e,取其 pin 所在 node 的 unique 集合 V_e(|V_e| = deg'_e)
過濾:  net_mask_io[e] = (2 <= deg_e) AND (deg_e < params.ignore_net_degree)
       # 逐位元同義於 DP 的 net_mask_ignore_large_degrees(BasicPlace.py:161-163)
       # adaptec1/bigblue4 的 config 皆為 ignore_net_degree = 100
S_{e,k} = Σ_{i ∈ V_e} ℓ_{i,k}
q_{e,k} = 1 − exp(S_{e,k})
λ_e     = Σ_k q_{e,k}
L_IO    = Σ_e w_e · max(λ_e − 1, 0)               # clamp(min=0) 消掉 λ<1 的數值殘差
```

**`D_max` 的變更(H2(a)):** v1 用 `D_max = 256`(對齊 evaluator 的 MST 分支)。v2 改為**直接沿用 DP 的 WL mask 述詞**,理由是 Codex 指出的硬傷:`ignore_net_degree=100` 讓 degree ≥ 100 的 net **完全沒有 WL 梯度**,若 IO 項對它們施力,這些 net 上只有 IO 力、沒有任何反作用力,幾何上可以被無限拉扯而不付出線長代價。實務影響量級:adaptec1 只有 2 個 net 落在 (100, 256];bigblue4 需在 T2 Step 1 統計後記錄。`D_max=256` 保留為 **ablation A6**,並要求 A6 同時報告「該區間 net 的 WL 是否被 mask 掉」以確認風險已理解。

`max(·,0)` 不是裝飾:`λ_e` 在 τ 大時可能因浮點誤差略小於 1,不 clamp 會在報表出現負的「期望 crossing」。clamp 的 mask 同時進 backward。

### 3.2 關鍵設計決定

#### 3.2.1 【改寫,H2(b)】`w_e` 預設為 1(**偏離 spec §5.2 的 1/(|e|−1),且必須被證據裁決**)

spec 要求 `w_e` 含 `1/(|e|−1)` 稀釋補償。**本設計以 `w_e = 1` 為預設**,論證如下——但 Codex 正確指出「與純量指標對齊」不等於「per-node 梯度組成正確」,因此本節同時規定**必跑的反證實驗**。

支持 `w_e = 1` 的論證:

1. **與指標對齊。** evaluator 的 `io_count` 對大 net **不打折**地累加 crossing(`evaluator_ref.py:108`);把大 net 的 objective 權重除以 `|e|−1` 會系統性地讓 optimizer 忽略貢獻最大的 net。
2. **乘積式自帶的稀釋是語意正確的。** `∂q_{e,k}/∂p_{i,k} = Π_{j≠i}(1−p_{j,k})`——當 region k 內已經有別的 pin,把這顆 pin 搬走**本來就不該**降低 `q_{e,k}`;梯度只在「該 region 的最後一顆 pin」上是 O(1)。再乘 `1/(|e|−1)` 是重複打折。
3. **`D_max` 收緊後,最極端的 degree 已不在項內**(deg < 100),skew 的動態範圍大幅縮小。

**Codex 的反論(記錄在案,未被論證推翻):** 依 τ 與佔用狀態,高 degree net 可能在**許多**被佔 region 上同時貢獻梯度,也可能因 `q` 飽和而幾乎不貢獻;**全域 L1 正規化(§5)只能調整總量級,無法修正組成**。這個反論只能用實測反駁,不能用推理。

**因此(v2 的硬性要求):**

- **A4 升為必跑,且 adaptec1 與 bigblue4 都跑**(v1 只排 adaptec1,而 adaptec1 只有 2 個 net 超過 degree 100,根本測不出 skew)。
- **T7 的每次 callback 都要輸出 per-degree-bucket 診斷**,bucket = `{2, 3, 4–7, 8–15, 16–31, 32–63, 64–99}`:
  - `grad_share[b]` = 該 bucket 的 node 收到的 `‖∇L_IO‖₁` 佔全體比例;
  - `io_share[b]` = 該 bucket 的 net 在 evaluator 精確 `per_net_crossings` 中的佔比;
  - `lambda_share[b]` = 該 bucket 在 `Σ(Λ_e−1)` 中的佔比。
- **推翻條件(寫死在 T9 報告的判準):** 若任一 bucket 的 `grad_share[b] / io_share[b]` 偏離 1 超過 3×,或 A4(`w_e=1/(deg'−1)`)在任一 case 的 Pareto 前緣上支配預設,則**預設改為 `1/(deg'−1)`** 並更新本節與 spec 偏離註記。

#### 3.2.2 必須手寫 backward(理由已用實測校正)

真實 adaptec1 座標、K=16,對「無 clamp 的 naive autograd」與「log-space leave-one-out」逐項比對:

| τ_rel | naive **無 guard** 的 NaN 數(共 422,894 項) | clamp 版 vs 穩定版的相對 L1 差 | fp32 穩定版 vs fp64 參考 |
|---:|---:|---:|---:|
| 0.20 | 0 | 1.65e−07 | 1.38e−07 |
| 0.05 | **14,674** | 1.89e−07 | 1.64e−07 |
| 0.02 | **193,798(45.8%)** | 2.79e−07 | 2.26e−07 |

誠實結論:

- **無 guard 的 naive autograd 是致命的**:`softmax` 在 fp32 下對「深處於某 region 的 cell」直接回傳 `1.0`,`log1p(-1.0) = −inf`,`exp(−inf)=0`,鏈式相乘得 `0 × inf = NaN`;τ_rel=0.02 時**近半個梯度向量是 NaN**——一次就毀掉整個 placement。
- **但只要加上 `p.clamp(max=1−1e−7)`,naive autograd 在 fp32 下與嚴謹的 log-space leave-one-out 一致到 round-off(相對 L1 ~2e−7)。** 所以手寫 backward 的理由**不是數值正確性**,而是:(a) **記憶體與規模契約**(§2.5——autograd 不可能自行做四趟 chunked 重算);(b) 速度;(c) 把 clamp/floor 變成顯式、可測試的契約;(d) 提供 `‖∇L_IO‖₁` 的免費輸出(§5 需要)。
- 手寫 backward 的核心式(**不要寫成 `exp(S)/(1−p)`**):
  ```
  ∂L/∂p_{i,k} = w_e · 1[λ_e>1] · exp(S_{e,k} − ℓ_{i,k})        # leave-one-out,永遠 ∈ (0,1]
  ∂L/∂z_{i,k} = p_{i,k}·(∂L/∂p_{i,k} − A_i),  A_i = Σ_k (∂L/∂p_{i,k})·p_{i,k}
  ∂L/∂x_i     = Σ_k (∂L/∂z_{i,k})·(−1/τ)·∂d_k/∂x_i
  ```
  `exp(S − ℓ)` 的指數在**相減之後**才取,結構上不可能出現 `0×inf`。

#### 3.2.3 fixed / filler node 的梯度必須清零

`pos` 佈局為 `[x_0..x_{n_all−1} | y_0..y_{n_all−1}]`(`n_all = placedb.num_nodes` 含 filler;見 `run_placement_reweight.py:33-35`)。terminal(`num_movable ≤ i < num_physical`)有 pin、必須參與 `S_{e,k}` 的**前向**(fixed IO pad 在哪個 region 是真實資訊),但**反向必須零梯度**——Nesterov 的 `u_{k+1} = v_k − α·g_k` 會更新整個 `pos` 向量,給 fixed macro 非零梯度等於把它搬走,直接違反 Phase 1「macro 固定」的前提。filler 無 pin,自然不在任何 `V_e` 中。

#### 3.2.4 【改寫,M1】「forward 精確計數 / backward surrogate 梯度」的落地與 optimizer lock

spec §9 M2 行寫的是 straight-through 混合。**逐 iteration 的字面 straight-through 被否掉,但 v1 的理由要修正一條:**

- ~~「STE 在這裡沒有良好定義」~~ — **收回**。Codex 正確指出 `L = hard.detach() + soft − soft.detach()` 是標準構造,在這裡完全可寫。v1 的措辭是錯的。
- **成立的理由 1(成本):** evaluator 單次 warm = 0.159 s(adaptec1)/1.318 s(bigblue4);DP 自身一次 `obj_and_grad_fn` 約 15 ms / 450 ms,而 line search 每 iteration 呼叫 1–10 次。逐次精確計數 = **2.5–3.7× 的總 runtime**。
- **成立的理由 2(鎖定組態下無行為差異,且此鎖定必須被 assert):** 在 `optimizer=nesterov` + `use_bb=0` + `Lsub_iteration=1` 下,objective **值** 證實無法影響軌跡:
  - `step_nobb` 的 backtracking 判準是 `alpha_kp1 > 0.95*alpha_k`,而 `alpha_kp1 = ‖v_{k+1}−v_k‖/‖g_{k+1}−g_k‖`(`NesterovAcceleratedGradientOptimizer.py:129`)——**純梯度量**;`f_kp1` 只存進 `cur_metric.objective` 記 log。
  - `Lsub_stop_criterion`(`NonLinearPlace.py:358-372`)**確實**消費 `metrics[..].objective`,但它在 `for Lsub_step in range(model.Lsub_iteration)` 內被呼叫(`:702`);`Lsub_iteration == 1` 時該迴圈本來就只跑一次,回傳值無論真假都不改變控制流。
  - `check_divergence` 用的是 hpwl 與 overflow,不是 objective。
  - **⇒ T5 必須 assert 這三個條件;T4 必須有回歸測試(對 obj 加常數擾動,iteration 軌跡逐位元不變)。** 若未來組態漂移(CG optimizer 走值域 line search、或 `Lsub_iteration>1`),本節結論失效,屆時必須改採真 STE。

**定案的等價落地(把「精確」放在它真正有價值的地方):**

- **(a) 報表與 exit 一律用精確值。** 所有 `io_count`/`ft_count`/`hard_lambda_sum` 取 `evaluator_ref`。
- **(b) 每 N=50 iter 的 evaluator callback 產生四件事:** ① 精確 `io_count`/`ft_count`/`hard_lambda_sum`/`per_net_lambda` 進軌跡 log(GP 期間 + **GP 末端 `io_gp`** + LG 後各一次);② `λ_io` 的 `ratio` 重算(§5);③ 選配的 per-net 校準權重 `w_e ← 1 + α_io·min(c_e, cap)`;④ **`refresh_nesterov_secant()`**(§6.4,只要 ②/③ 之一實際改變了 objective)。
- **(c) `α_io` 預設 0**(主結果不被 M1 機制汙染),`α_io > 0` 是 ablation A3。這也讓 M1 的 reweight 與 M2 的可微項共用**同一個** evaluator 訊號,只是施力點不同。
- **(d) 記得 M1 的 cap 飽和教訓**(M1 報告假設 B)。若啟用 (c),cap 預設放寬到 64 並在報表記錄「被夾限的 net 比例」。

#### 3.2.5 【改寫,H3】S2 的目標(soft `λ−1`)與真實指標(crossing 數)的關係 — 實測**份額**

`L_IO` 最小化的是「期望觸及 region 數 − 1」,而 `io_count` 數的是 MST edge 跨界次數。用 M1 的最終 placement 實測 **hard** 版本 `Σ_e(Λ_e−1)`:

| case | `io_count` | `Σ_e (Λ_e−1)`(hard) | 比值 | 超出下界的 crossing |
|---|---:|---:|---:|---:|
| adaptec1 k=16 | 29,992 | 24,222 | **1.238** | 5,770(19.2%) |
| adaptec1 k=32 | 49,123 | 36,186 | **1.358** | 12,937(26.3%) |
| bigblue4 k=16 | 100,736 | 84,665 | **1.190** | 16,071(16.0%) |

**表述的三個限制(v2 明訂,取代 v1 過強的「解釋了 74–84%」):**

1. 這是 `Σ(Λ−1)` 佔 `io_count` 的**份額(share)**,**不是**解釋變異(R²),**不是** rank correlation,**更不是**「降低 λ 就會降低 io」的因果證據。
2. 樣本只有 **3 個最終 placement**,且全部來自 flat-like 軌跡;它證明的是「在這類幾何下,crossing 主要由 region-presence 決定,繞路只佔 16–26%」,不保證優化過程中該比例穩定。
3. 對照 M0 報告的 two_stage `io/lb = 1.80×–2.48×`,可見這個比例**會**因幾何品質而劣化——這正是 F4 要監控的東西。

**因此 v2 新增三項必測診斷(T5 實作、T7 記錄):**

- **hard-λ 軌跡**:每次 callback 由 evaluator 回報 `hard_lambda_sum`,與精確 `io_count` 同步記錄。
- **固定參考 τ 的 soft 診斷**:若要看 soft `λ` 的優化進展,一律在 `τ_ref = 0.05·L_R`(與退火無關的固定值)重算一次,避免 §4.1 那種「座標不動、L_IO 從 243,943 掉到 33,503」的溫度假象。
- **GP 末端 rank correlation**:Spearman ρ(`per_net_crossings`, `per_net_lambda − 1`),只取 `per_net_crossings > 0` 的 net;連同 detour 量 `io_count − Σ(Λ−1)` 一併報告。

### 3.3 被否掉的替代 IO surrogate

| 方案 | 否掉理由 |
|---|---|
| `L = Σ_e Σ_k q_{e,k}(1−q_{e,k})`(熵式/割式) | 最小值在 q∈{0,1},但不區分「觸及 1 個 region」與「觸及 K 個」,無法表達 λ−1 |
| soft-min-cut(`Σ_{e} Σ_{k<k'} q_{e,k} q_{e,k'}`) | 是 λ−1 的二階近似,對大 net 爆炸(O(K²) 項且無界),且沒有 λ−1 的直接語意 |
| 直接對 evaluator 的 `per_net_crossings` 做 REINFORCE / 有限差分 | 高變異、需要多次 evaluator 呼叫;S5 已經是它的結構化版本,留作 fallback |
| 對 MST 拓撲本身可微化(soft-MST) | Prim 的 argmin 不可微且拓撲離散;evaluator 的 MST 是計分尺不是 objective,不該綁進梯度路徑 |

---

## 4. Q3 — τ 退火 schedule

### 4.1 實測:梯度強度對 τ 是單峰的(**表格保留不動**)

adaptec1 k=16 真實座標,`‖∇L_IO‖₁`(正確的 leave-one-out 梯度):

| τ_rel | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L_IO` | 344,304 | 243,943 | 168,236 | 88,015 | 52,500 | 33,503 | 28,233 |
| `‖∇L_IO‖₁` | 66.8 | 143.4 | 206.8 | **236.2** | 229.1 | 209.7 | 190.9 |
| `‖∇WL‖₁/‖∇L_IO‖₁` | 5,376 | 2,505 | 1,736 | **1,520** | 1,568 | 1,713 | 1,881 |

bigblue4 k=16 同形狀(峰值同樣落在 τ_rel≈0.1,`‖∇L_IO‖₁` 從 213(0.5)升到 670(0.1)再降到 610(0.01))。

物理解讀:τ 大 → `p` 趨近均勻,`q` 對位置不敏感;τ 小 → 只有距邊界 ~τ/2 內的 cell 有梯度(**τ_rel=0.02 時 45.8% 的 cell `p_max > 1−1e−7`,IO 梯度恆 0**)。兩端都稀釋,中間有窗口——這就是 spec §10.1 風險的量化版本。

**⚠ 這張表同時是 H3 的直接證據:座標完全不動,只改 τ,`L_IO` 就從 243,943 掉到 33,503(−86%)。任何以 soft `L_IO` 或 soft `λ` 的下降幅度做的診斷,在退火進行中都是無效的。**

### 4.2 【改寫,H1】定案 schedule — 連續漂移

```
τ_rel(of) = clip( τ_lo · (τ_hi/τ_lo) ^ ((of − of_end)/(of_on − of_end)), τ_lo, τ_hi )
  τ_hi = 0.30, τ_lo = 0.03
  of_on  = 0.90   (IO 項啟用時的 overflow)
  of_end = stop_overflow = 0.07
τ = τ_rel(of) · L_R,  L_R = sqrt(A_die / K)
```

**更新頻率(v2 的關鍵改動):** τ 與 `ρ_io` **每 iteration** 依當前 overflow 重算(`of` 由暴露出來的 `model.overflow` 直接讀,零成本),而**不是**只在每 50 iter 的 callback 更新。理由是 H1:每 iteration 的 O(dτ) 連續漂移與 DP 自身的 γ/`density_weight` 漂移同級,Nesterov 的 secant 估計對它有容忍度(DP 本來就這樣跑了七年);真正危險的是**離散跳變**(啟用瞬間、`ratio` 重算、`w_e` 校準),那三者由 §6.4 的 secant refresh 處理。

三個設計理由(不變):

1. **與 DP 既有機制同構。** `update_gamma`(`PlaceObj.py:901-914`)就是 `γ = base_γ · 10^((of−0.1)·20/9 − 1)`——同一個「以 overflow 為時鐘、log-linear」的家族。
2. **自動與 density weight μ 的演化耦合,但不直接綁。** 以 overflow 為共同時鐘則兩者天然同步、互不放大。
3. **不直接綁 γ。** γ 的軌跡實測是 adaptec1 1,670 → 14.3,換算成 `τ_rel` 是 **0.63 → 0.0054**——上端太糊、下端遠低於可用的 0.02 下限(γ 的自然尺度是 bin size 41.7,τ 的自然尺度是 region size 2,671,差 64×)。**函數形式借用、參數不共用。**

### 4.3 掃描範圍(T7)

- 主掃:`τ-schedule ∈ { annealed(0.30→0.03), fixed(0.10) }` × `ρ_max ∈ {0.02, 0.05, 0.10, 0.20, 0.40}` = **10 臂**(adaptec1 k=16,單臂 ~2 分鐘)+ **第 11 臂 = margin pilot**(§9.2)。
- 第二階段(僅在 §9.1 的灰帶或 F1/F3 觸發時):`τ_hi ∈ {0.5, 0.3, 0.2}` × `τ_lo ∈ {0.05, 0.03, 0.015}` 的對角 3 組;`of_on ∈ {0.9, 0.5}` 的 2 組。

---

## 5. Q4 — auto-normalization(λ_io 配平)

### 5.1 借鑑對象:DP 的 `initialize_density_weight`

`PlaceObj.py:767-825` 的做法是:對 WL 與 density 各做一次獨立 backward,取 **L1 norm 比**,再乘一個無量綱係數:

```python
grad_norm_ratio = wirelength_grad_norm / density_grad_norm      # PlaceObj.py:819
self.density_weight = params.density_weight * grad_norm_ratio   # 8e-5 · ratio
```

注意兩件事:(a) 用 **L1** 而非 L2;(b) 在 **precondition 之前**取 norm。M2 完全鏡射這兩點。

### 5.2 【改寫,H1】定案 — 連續量與離散事件分離

```
# --- 每 iteration(連續,無 secant 風險)---
tau   = tau_rel_from_overflow(of) * L_R
rho   = rho_max * clip((of_on − of)/(of_on − of_full), 0, 1) * ramp(iteration)
λ_io  = rho * ratio_ema                      # ratio_ema 只在離散事件更新

ramp(it) = clip((it − it_activate) / 20, 0, 1)     # 啟用後 20 iter 線性 ramp-in

# --- 每 N=50 iter 的 callback(離散事件)---
g_wl      ← backward(wirelength_op(pos))      # 一次額外 backward,precondition 之前取 L1
g_io_l1   ← 本 op 最近一次 backward 自留的 ‖∇L_IO‖₁(免費)
ratio_new ← ‖g_wl‖₁ / max(g_io_l1, ε)
ratio_ema ← 0.5·ratio_ema + 0.5·ratio_new
λ_io      ← min(rho * ratio_ema, c_lip · τ² / γ)      # Lipschitz 平價護欄,c_lip=1.0
if (ratio_ema 改變) or (w_e 改變) or (剛啟用):
    refresh_nesterov_secant(optimizer)                # §6.4 —— 強制,不可省
```

- **`ρ_io` 才是可掃的旋鈕**(無量綱、跨 case 可移植);`λ_io` 是導出量。實測錨點:ρ=1 時 λ_io ≈ 1,520(adaptec1 k16, τ_rel=0.1)、2,787(bigblue4 k16)。因此 `ρ_max ∈ [0.02, 0.4]` ⇒ adaptec1 的 λ_io ∈ [30, 610]。
- **啟用 ramp-in(v2 新增):** 啟用瞬間 `λ_io: 0 → λ*` 是所有離散事件中最大的一次跳變;20 iter 的線性 ramp 把它攤成連續漂移,secant refresh 只需在 ramp 起點做一次。
- **Lipschitz 平價護欄的來歷(標示為啟發式):** WL 項的曲率尺度 ~ `1/γ`,IO 項的曲率尺度 ~ `λ_io/τ²`。`step_nobb` 的步長估計 `α = ‖Δv‖/‖Δg‖` 是全域 Lipschitz 的倒數,IO 項若在小 τ 下主導曲率,步長會被壓垮、backtracking 次數暴增。以 adaptec1 退火末端代入:τ=53.4、γ=14.3 → 上限 λ_io ≈ 200,對應 ρ ≈ 0.13。
- 額外 backward 的成本:每 50 iter 一次 ≈ 15 ms(adaptec1)/450 ms(bigblue4);加上 secant refresh 的 2 次 obj+grad,合計 **< 4% runtime**。
- 為何**不**每 iteration 重算 `ratio`:那會變成自我抵消的回饋(IO 梯度一旦變小就自動加權放大);DP 也只在初始化時算一次比值。

### 5.3 被否掉的替代

| 方案 | 否掉理由 |
|---|---|
| 固定 `λ_io` 常數 | 量級跨 case 差 2×(1,520 vs 2,787),且隨 τ 退火漂移,無法做 iso-WL 比較 |
| 用 evaluator 的 `io_count` 直接做期望值正規化(FTAFP 式) | 兩者量綱不同(計數 vs 梯度),且 `io_count` 每 50 iter 才有一次 |
| 綁 `μ`(density weight)成比例 | μ 是幾何成長的乘法更新,綁上去會讓 λ_io 在後期爆炸 |
| L2 norm 比 | 與 DP 既有慣例不一致;L1 對「少數邊界 cell 有大梯度」的分布更合適 |

---

## 6. 【全節改寫,M2 + H1】Q5 — DREAMPlace 整合點

### 6.1 選項比較與定案

| | 選項 A:正式 `dreamplace/ops/ioaware/` op | 選項 B:M1 iteration callback 加梯度 | 選項 B′:class-level monkey-patch `PlaceObj.obj_fn` | **選項 C(定案):`params` 承載的 per-instance patch** |
|---|---|---|---|---|
| 侵入性 | 高:新增 op 目錄 + `CMakeLists` + import,每次改都要 `cmake --build && --target install`(2–4 分鐘) | 0 行 | 0 行 DP 改動 | **~8 行 python-only patch,3 個檔案,無 cmake rebuild(與 M1 `iteration-callback.patch` 同一 precedent)** |
| 可行性 | 可行 | **結構上不可行** | 可行但**不安全** | 可行 |
| 隔離性 | per-instance | — | **process 全域變異**;例外/巢狀安裝/忘記 uninstall/同 process 連跑 flat 與 io ⇒ 重複加項、汙染對照組、還原錯誤的 wrapper、GPU 記憶體滯留 | **per-run params → per-instance**,無全域狀態、無需 uninstall、天然 exception-safe |
| 可測試性 | 差(pytest 需先 rebuild DP) | — | 中 | **好(pure python patch,pytest 直接跑)** |
| 與 M1 共存 | 可 | — | 可 | 可(兩者正交:M1 寫 `data_collections.net_weights`,M2 加 objective 項) |

**選項 B 被硬事實否掉:** M1 的 `iteration_callback` 掛在 `NonLinearPlace.py:517-519`,而 `make_parameter_update()`(即 `optimizer.step()`)在 `:450/:458`——**callback 在參數已經更新之後才觸發**,不可能對「本次 step」貢獻梯度。它只能做事後的位置修正(那是 S5 的形狀)。

**選項 B′ 被 Codex 的 M2 否掉並由 scheduler 裁決放棄:** v1 曾提議「先 monkey-patch、凍結後再 patch」的兩階段。v2 **取消階段 1**,直接落 patch——理由是 monkey-patch 的風險(尤其「同 process 先跑 flat 對照組再跑 io」正是 T6/T7 的實際用法)遠大於它省下的一次 patch 撰寫成本,而 patch 本身是 python-only、無編譯成本。

### 6.2 Patch 內容(`ioplace/dp_patch/m2-extra-obj-terms.patch`,一個檔、四個 hunk)

1. **`dreamplace/Params.py::toJson`** — 排除底線開頭的 key:
   ```python
   if key != 'params_dict' and not key.startswith('_'):
   ```
   *必要性:* `Params.dump()` 會 `json.dump(self.toJson())`(`Params.py:131`),把一串 python callable 放進 `params.__dict__` 會讓任何序列化路徑爆掉。用底線前綴 + 排除規則一次解決,且對未來所有 runtime-only 附掛都有效。
2. **`dreamplace/PlaceObj.py::PlaceObj.__init__`**(尾端)—
   ```python
   # io-aware: runtime-attached extra objective terms (see IO-Aware-Top-Level-Placer)
   self.extra_obj_terms = list(getattr(params, "_extra_obj_terms", []))
   ```
3. **`dreamplace/PlaceObj.py::PlaceObj.obj_fn`**(`return result` 之前)—
   ```python
   for term in self.extra_obj_terms:
       result = result + term(pos)
   ```
4. **`dreamplace/NonLinearPlace.py`**(`logging.info("use %s optimizer" ...)`,約 `:231` 之後)—
   ```python
   # io-aware: expose optimizer/model so the iteration callback can refresh
   # objective-dependent optimizer state (see design §6.4)
   self.optimizer = optimizer
   self.model = model
   ```

同步規則沿用 M1:在 `$DP` 的 `io-aware` branch commit,`install/` 一併更新(python-only,不需 rebuild),diff 存本 repo `ioplace/dp_patch/`。

**為何 per-instance 一定成立:** `construct_model()`(`NonLinearPlace.py:113-124`)在 `__call__` 內建立 `PlaceObj`,`closure()`(`:434`)可能再建一次;兩者都吃同一個 `params`,所以兩個 instance 都會拿到同一份 term list,而**不同 run 用不同 `params` 物件**(我們的 driver 每次 `_load_dreamplace` 都新建 `Params()`),天然隔離。

### 6.3 三個整合期的坑(必須寫進實作)

1. **precondition 會除到我們的梯度。** `obj.backward()` 之後 `precondition_op(pos.grad, ...)`(`PlaceObj.py:409`)把整個 `pos.grad` 除以 `precond = Σpin_weights + α·density_weight·node_area`(clamp≥1,`PlaceObj.py:69-101`)。**預設接受**(IO 力量級本來就與 pin 數同階),但 auto-normalization 的 norm 必須在 precondition **之前**量(與 DP 一致)。若 T7 出現「多 pin cell 動不了」,ablation 可加「IO 梯度不預條件」的旁路。
2. **`obj_fn` 的回傳值會進 `cur_metric.objective` 的 log。** 加了 `λ_io·L_IO` 之後這個數字不再可與 M0/M1 的 log 直接比較;報表用 hpwl/io_count,不用 objective。§3.2.4 的 lock 保證它不影響行為。
3. **line search 會呼叫多次 ⇒ 同一 iteration 內 `τ`/`λ_io`/`w_e` 必須是常數。** 實作上:schedule 在 `iteration_callback` 與 iteration 開頭各更新一次,term 物件在 `obj_fn` 內**只讀**;禁止在 `obj_fn` 內部依呼叫次數改變任何狀態。T4 有測試(呼叫 3 次 `obj_fn`,斷言 τ/λ 相同)。**但這只解決 iteration 內部的一致性,跨 iteration 的 secant 問題見 §6.4。**

### 6.4 【新增,H1】Nesterov secant 失效機制(**強制**)

#### 6.4.1 失效模式(Codex H1,已由原始碼確認)

`step_nobb`(`NesterovAcceleratedGradientOptimizer.py:65-166`)在 group state 中**跨 iteration 快取** `g_k`(上一步的梯度)、`obj_k`、`g_k_1`、`obj_k_1`、`alpha_k`。步長來自 secant/Lipschitz 估計:

```
alpha_kp1 = sqrt( Σ(v_kp1 − v_k)² / Σ(g_kp1 − g_k)² )        # :129
while True:
    ...
    if alpha_kp1 > 0.95*alpha_k or backtrack_cnt >= 10: break # :141
```

若 `g_k` 在 objective `O_t` 下算出、`g_kp1` 在 `O_{t+1}` 下算出,則 `g_kp1 − g_k` 混入了**objective 變動量**而非只有位置變動量。後果比「估計失準」更糟:

- objective 跳變讓 `‖g_kp1 − g_k‖` 膨脹 ⇒ `alpha_kp1` 塌陷 ⇒ `alpha_kp1 > 0.95*alpha_k` 判定失敗 ⇒ **backtracking 迴圈重跑**,而重跑時 `g_k` 仍是舊的 ⇒ 膨脹持續 ⇒ **一路 backtrack 到上限 10 次**,燒掉 10 次 obj+grad,並以極小的 `alpha_k` 收場。
- 該極小 `alpha_k` 會被帶進下一 iteration(`alpha_k` 是持續狀態),污染範圍不只一步。

也就是說,**每 50 iter 的一次 λ/w 更新,可能造成一次 10× 的計算浪費加上一次步長崩潰**——正好發生在掃描的每個排程點上,足以讓 ρ 掃描的結果失去意義。

**實測確認(`probe8_secant.py`,對**真的** `NesterovAcceleratedGradientOptimizer` 跑 2 維二次問題,第 5 步後切換 objective):**

| | `‖g_k − ∇O_new(v_k)‖∞` | `‖g_k − ∇O_old(v_k)‖∞` | `alpha_k`(切換後 → 下一步) | 下一步的 obj 求值次數 |
|---|---:|---:|---:|---:|
| **不刷新(RED)** | 5.230e+02 | **0.000e+00** | 4.4636e−01 → **4.0364e−05** | 3 |
| **刷新(GREEN)** | **0.000e+00** | 5.230e+02 | 9.8030e−03 → 9.8073e−03 | 1 |

不刷新時,快取的 `g_k` **精確地就是舊 objective 的梯度**(誤差 0),下一步的步長塌陷 **11,000×**(4.46e−1 → 4.04e−5)並多燒 3 倍的 obj 求值;刷新後 `g_k` 精確等於新 objective 的梯度,步長只變動 +0.04%。**這比 v1 預估的「估計失準」嚴重得多,H1 是必修項而非防禦性設計。**

#### 6.4.2 定案機制

**(1) 把大部分變動連續化**(§4.2、§5.2):τ、`ρ_io`、ramp 每 iteration 微幅更新,量級與 DP 自身的 γ/`density_weight` 漂移同級。

**(2) 對剩下的離散事件強制失效重算。** 事件清單:① IO 項啟用(ramp 起點);② `ratio_ema` 更新;③ `w_e` 校準(`α_io>0` 時);④ margin 項開/關。

```python
# ioplace/dp_hook.py
def refresh_nesterov_secant(optimizer):
    """在 objective 離散變更後,讓 Nesterov 的梯度快取在新 objective 下重算。
    前提(由 T5 assert):optimizer 為 NesterovAcceleratedGradientOptimizer、
    use_bb == 0(走 step_nobb)、單一 param group 且單一 param。"""
    g = optimizer.param_groups[0]
    if not g["g_k"]:                       # 尚未跑過第一步,無快取可汙染
        return
    f = optimizer.obj_and_grad_fn          # 已綁定的 model.obj_and_grad_fn
    obj_k, grad_k = f(g["v_k"][0])         # v_k 就是 pos 本身
    g["g_k"][0].copy_(grad_k.data)
    g["obj_k"][0].copy_(obj_k.data)
    if g["g_k_1"]:
        obj_k1, grad_k1 = f(g["v_k_1"][0])
        g["g_k_1"][0].copy_(grad_k1.data)  # 注意 g_k_1 可能別名 v_k_1.grad,self-copy 安全
        g["obj_k_1"][0].copy_(obj_k1.data)
        dv = (g["v_k"][0].data - g["v_k_1"][0].data).norm(p=2)
        dg = (g["g_k"][0] - g["g_k_1"][0]).norm(p=2)
        if dv > 0 and dg > 0:
            g["alpha_k"][0].copy_(dv / dg)  # 以同一 objective 的一致 secant 重估步長
```

呼叫時機:**在新的 τ/λ/w 生效之後、`iteration_callback` 返回之前**(callback 本身就在 `optimizer.step()` 之後,`v_k` 已等於當前 `pos`,正是下一步要用的參考點)。成本:2 次 obj+grad / 事件 ≈ 每 50 iter 加 2 次 ⇒ ~2% runtime。

**(3) 版本號 invariant(測試用,可在 production 關閉)。** schedule state 持有單調遞增的 `obj_version`,每次離散變更 +1;`refresh_nesterov_secant` 記錄 `refreshed_version`。在測試模式下包裝 `obj_and_grad_fn`,於每次呼叫斷言 `obj_version == refreshed_version`——亦即**不存在任何一次梯度求值發生在「已變更但未刷新」的狀態下**。

#### 6.4.3 T4 的驗收測試(RED/GREEN 明確)

- **單元(不需 DP):** 用**真的** `NesterovAcceleratedGradientOptimizer` 跑一個 2 維二次問題。`O_A(x)=½‖x‖²`;跑 5 步後切到 `O_B(x)=½‖x‖²+c‖x−a‖²`。
  - **RED(不呼叫 refresh):** 斷言 `g_k` 等於 `∇O_A(v_k)` 而非 `∇O_B(v_k)`(差異 > 1e−6),且切換後第一步的 `backtrack_cnt` 顯著上升 / `alpha_k` 塌陷。
  - **GREEN(呼叫 refresh):** 斷言 `g_k == ∇O_B(v_k)`(1e−10)、`g_k_1 == ∇O_B(v_k_1)`、`alpha_k` 等於用新 objective 重算的 secant。
- **整合(`simple`,`@pytest.mark.slow`):** 啟用版本號 invariant,跑完全程斷言零次違反;另斷言 backtracking 總次數不因排程事件出現尖峰(每個排程 iteration 的 `obj_eval_count` 增量 ≤ 非排程 iteration 的中位數 × 2)。

---

## 7. Q6 — iso-WL 比較協定

### 7.1 產生可比點的方式:ρ sweep → Pareto 前緣(不是單點配平)

- **軸:** x = `hpwl`(取自 `evaluator_ref`)、y = `io_count`,兩者相對同 case 的 **flat baseline 平均值**取百分比。
- **可比點來源:** §4.3 的 10 個 ρ×τ 臂 + 第 11 臂 margin pilot。M1 側取其唯一非劣點 `(α,every)=(0.2,100)` 與 flat。
- **M2 exit 判準:** 存在一個 M2 配置同時滿足
  - `Δhpwl ≤ +2.0%`(vs flat;M1 最佳點是 +1.30%),且
  - `Δio ≤ −5.0%`(vs flat)且 `|Δio| > 3σ_rep`,且
  - `Δio` 明顯優於 M1 同 case 的 `Δio`(adaptec1 k16 = −0.06%,bigblue4 k16 = +0.63%)。
- **灰帶處置(scheduler 裁決):** 若最佳臂落在 `−5% < Δio ≤ −3%`,判為**調參問題**——走 §4.3 第二階段 τ 補掃 + margin 深掃(T11),**不**觸發 F2 換 surrogate。
- 每個 run **必須**同時報 `io_gp`(GP 末端、LG 之前)與 `io_count`(LG 之後)。

### 7.2 【改寫】噪聲底線與 `deterministic_flag` 協定(T6,擋在 T7 前面)

M1 報告已警告:同設定重跑的 `io_count` 差 303(~1.0%),而它量到的「改善」只有 0.06%。T6 定義三個量:

- `σ_rep(det=0)`:**完全相同設定**、`deterministic_flag=0`(M0/M1 既有協定)重跑 5 次的 `io_count` 標準差。
- `σ_rep(det=1)`:同上但 `deterministic_flag=1`,重跑 5 次。
- `σ_seed`:`deterministic_flag` 取上面選定的值,改 `params.random_seed ∈ {1000..1004}` 各 1 次。

**regime 選擇規則(寫死在 T6 的驗收):**

- 若 `σ_rep(det=1)/mean < 0.2%` **且** `det=1` 的 runtime 相對 `det=0` 增幅 `< 20%` ⇒ **M2 全部 arm 改用 `deterministic_flag=1`**,並在同一 regime **重測 flat 基準**(所有 Δ% 的分母必須來自同 regime,否則不可比)。
- 否則維持 `deterministic_flag=0` + `3σ_rep` 噪聲帶。
- 兩種情況都要在 M2 報告記載選擇與依據;若 `σ_rep(det=1)` 仍 > 1%,那是獨立的可重現性問題(config 已設 `deterministic_flag: 1` 卻仍不可重現),**停下來 root-cause**,不要帶著它跑掃描。

**既有 driver 缺口(必須在 T5 修):** `run_flat(config, k, rtype, seed, out)` 的 `seed` **只餵給 `get_regions_for`**(`run_placement.py:102`),而 `rtype="grid"` 時 `make_grid_regions` 根本不用 seed——**現有的 `--seed` 在 grid 模式下完全不改變 placement**。必須新增 `--dp-seed` 直接寫 `params.random_seed`,否則 M2 的「多 seed」是假的。

**init_pos 缺口(T4 診斷發現,已修):** driver 繞過 `Placer.place()`,而那是唯一呼叫 `np.random.seed(params.random_seed)`(`Placer.py:36`)的地方;`BasicPlace.__init__` 只重設 torch(`:265`),init_pos 的中心噪聲與 filler 初始位置(`:272-289`/`:352-362`)卻吃 **numpy 全域 RNG** ⇒ 修正前同 process 內每次 placement 的 init_pos 都是不受控 draw(實測 adaptec1 k16 grid、det=1:`io_count` run-to-run ~0.8%、`ft_count` ~1.4%、hpwl ~0.01%)。`_place()` 現在自行 `np.random.seed(params.random_seed)`,因此 **`--dp-seed` 同時控制 init_pos 與 torch 側 `gp_noise`**。兩個後果:(a) M0/M1 的既有 results 檔帶著這個不受控噪聲產出,與修正後的 run **不 bit-可比**(M1 reweight 的 Δio=−0.06% 在該噪聲帶內,本來就非 signal;ft −9.9% 在帶外,結論不變);(b) T6 的 σ_rep/σ_seed **必須在此修正之後量測**,且 flat 基準照 §7.2 規則在選定 regime 下重測,否則 σ_rep 會把 init draw 的變異混進 GPU 非決定性。

### 7.3 Case 與規模

| 角色 | case | k | rtype | 次數 |
|---|---|---|---|---|
| 主戰場(掃描 + ablation) | adaptec1 | 16 | grid | 每臂 1 次;最佳臂 3 次 |
| 噪聲底線 | adaptec1 (flat) | 16 | grid | 5(det=0)+ 5(det=1)+ 5(seed) |
| 規模確認 / degree skew | bigblue4 | 16 | grid | flat / M1 / A2 / **A4** 各 1 次 |
| k 敏感度 | adaptec1 | 8, 32 | grid | flat / A2 各 1 次(補齊 M1 報告缺的 k=32 flat 基準) |
| region 形狀 | adaptec1 | 16 | slicing | flat / A2 各 1 次 |

時間預算:adaptec1 單次 ~60 s(flat)/~110–140 s(含 IO 項);bigblue4 ~545 s / ~1,100–1,400 s。全部合計 < 4 小時 GPU。

### 7.4 mtkahypar 非重現性 — 定案

`ioplace/partition/mtkahypar_runner.py:20` 用 `PresetType.DEFAULT` + `threads=8`,M0 報告已記錄它不 bit-reproducible。**但 M2 完全不用 mtkahypar**(可微項建在 flat 之上),所以這不是 M2 的阻塞項。若 M2 報表要重列 two_stage 對照欄:**用 `PresetType.DETERMINISTIC` 而不是 `threads=1`**(已確認存在於安裝的 `mtkahypar==1.6.2`),多執行緒下仍可重現。加 `--mtk-preset` 旗標、預設維持 `DEFAULT`,以免動到 M0/M1 已發表的數字。

---

## 8. Q7 — Ablation 設計

### 8.1 【改寫,H2/H4】配置

| 臂 | 名稱 | 設定 | 跑哪些 case | 目的 |
|---|---|---|---|---|
| A0 | flat | 無 IO 項、無 reweight | 全部 | 基準(M0 已有,T6 補噪聲與 regime) |
| A1 | reweight-only | M1 最佳:α=0.2, every=100 | adaptec1 k16、bigblue4 k16 | M1 對照(已有) |
| A2 | **diff-only** | `ρ_max=ρ*`, τ annealed, `α_io=0`, `D_max=99` | adaptec1 k∈{8,16,32}+slicing、bigblue4 k16 | M2 主結果 |
| A3 | diff + reweight | A2 + `α_io=0.2` | adaptec1 k16、bigblue4 k16 | 兩者是否互補 |
| **A4** | **`w_e` 變體(必跑)** | A2 但 `w_e = 1/(deg'−1)` | **adaptec1 k16 + bigblue4 k16** | 裁決 §3.2.1 對 spec 的偏離(v1 只跑 adaptec1,而 adaptec1 幾乎沒有大 net,測不出 skew) |
| A5 | τ 固定 | A2 但 τ_rel ≡ 0.10 | adaptec1 k16 | 退火是否真的必要 |
| A6 | `D_max` | A2 但 `D_max = 256` | adaptec1 k16、**bigblue4 k16** | 量化「有 IO 力但無 WL 反作用力」的 net 的影響 |
| **A7** | **margin pilot** | A2 + margin 項開(`ρ_margin=0.05`, `m=2·row_height`, 末 15% GP) | adaptec1 k16(T7 第 11 臂) | LG-retention;`io_count − io_gp` 的流失是否縮小 |

### 8.2 輸出表格式(沿用 `make_report.py` 的 markdown 風格)

```
| case | mode | k | rtype | det | dp_seed | io_count | io_gp | ft_count | hard_lambda_sum | tree_wl | hpwl |
  Δio% | Δio_gp% | Δhpwl% | lg_loss | runtime_s | peak_mem_mb | rho_max | tau_hi | tau_lo | alpha_io |
  w_mode | d_max | margin | lambda_io_final | spearman_rho |
```

- `io_gp` = GP 最後一次 callback 的精確 `io_count`(LG 前);`lg_loss = io_count − io_gp`。
- `hard_lambda_sum` = `Σ_e max(Λ_e−1, 0)`(H3 新增的 evaluator 欄位)。
- `spearman_rho` = GP 末端 `per_net_crossings` vs `per_net_lambda − 1` 的 rank correlation。
- `Δio%`/`Δhpwl%` 一律相對**同 case/同 k/同 rtype/同 det regime 的 flat 平均值**(T6 的 5 次重跑平均)。
- 另出 per-degree-bucket 診斷表(§3.2.1)與 Pareto 圖(`docs/results/figs/m2-pareto-adaptec1-k16.png`;x=Δhpwl%、y=Δio%,標出 flat 原點、M1 點、M2 前緣、±3σ_rep 噪聲帶)。

---

## 9. Q8 — Fallback 觸發準則與備援的最小形狀

### 9.1 【改寫,H3/H4/scheduler-10】觸發準則(全部可在 T7 結束時機械判定)

| ID | 量測 | 門檻 | 觸發後動作 |
|---|---|---|---|
| **F1** 梯度覆蓋率 | 退火末端(τ=τ_lo)時 `p_max < 1−1e−3` 的 movable cell 比例(每次 callback 記錄) | **< 5%** | 調參:§4.3 第二階段 τ 補掃;若補掃後仍 <5% 且 F2 亦觸發 → S5 |
| **F2** surrogate 失效 | **所有 ρ×τ 掃描臂(目前 10 組,數字隨臂數走)** 在 **`io_gp`** 基準上皆未達 `Δio_gp ≤ −3%` | 全 miss | **換 surrogate → S3(T10)** |
| **F2b** 灰帶 | 有臂達 `Δio_gp ≤ −3%` 但無臂達 exit 的 `Δio ≤ −5%` | — | **調參**:τ 補掃 + margin 深掃(T11);**不**換 surrogate |
| **F3** 不穩定 | ≥2 臂觸發 DP 的 `check_divergence`,或 `Δhpwl > +10%`,或排程 iteration 的 `obj_eval_count` 增量中位數 ≥ 5 | 任一 | 檢查 §5.2 的 Lipschitz 護欄與 §6.4 的 secant refresh 是否生效;降 `ρ_max` |
| **F4a** alignment(hard) | 同一 run 內,從啟用到 GP 末端 `Δ(hard_lambda_sum) ≤ −10%` **但** `Δ(io_gp) > −3%` | — | 目標-指標脫鉤 → **S3(T10)** |
| **F4b** exact-IO 反向 | 任一臂的 `io_gp` 相對其自身啟用時的值**上升** `> 3σ_rep`(**不看 λ,獨立觸發**) | — | 立即停該臂;檢查 secant/NaN/梯度符號 |
| **F5** LG 流失 | `lg_loss(M2) − lg_loss(flat)` 相對 flat 的 `io_count` **> +2%** | — | **獨立治理:margin 深掃(T11);不觸發換 surrogate** |

**F4 的關鍵修正(H3):** v1 把 F4 定義在 soft `last_lambda_sum` 上,但 §4.1 的表格證明**座標完全不動、只改 τ,soft `L_IO` 就下降 86%**——退火期間的 soft λ 軌跡完全無法當 alignment 診斷。v2 一律改用 evaluator 的 **hard** `hard_lambda_sum`;若確實需要 soft 診斷(例如看優化器是否在動),一律在**固定參考 τ = 0.05·L_R** 下重算。

**F2 的關鍵修正(H4 + scheduler-10):** v1 用 LG 後的 `io_count` 判 F2,會把「GP surrogate 成功但被 legalization 抹平」誤判為「surrogate 失效」而錯誤地把工作導向 S3。v2 的 F2 一律用 `io_gp`(GP 末端),LG 流失由 F5 獨立治理。

### 9.2 【新增,H4】Boundary keep-out margin 項 — **v1 的符號是錯的**

**v1 的錯誤:** v1 寫 `ε·Σ_i softplus(−d_{k*(i)}/τ_m)`。因為 `d*` 在 region 內部為**負**,`−d*/τ_m` 對「越深處」的 cell 越大 ⇒ 罰得越重 ⇒ 梯度把 cell **推向邊界**,與 keep-out 完全相反。

**v2 的正確形式(採 Codex 建議):**

```
d*_i     = min_k d_k(x_i, y_i)                       # signed,內部為負;已由 §2.5 FWD-1 的 max 免費得到
L_margin = Σ_{i ∈ movable} softplus( (d*_i + m) / τ_m )
```

- 深處 cell(`d* = −D`,`D ≫ m`):`(d*+m)/τ_m` 極負 ⇒ `softplus ≈ 0` ⇒ 不罰。
- 邊界帶內(`|d*| < m`):`softplus > 0` ⇒ 罰,梯度把 `d*` 推得更負 = **推離邊界、推向 region 內部**。✓
- 落在自己 region 外(`d* > 0`,理論上不該發生):重罰。
- 預設:`m = 2 × row_height`(bookshelf 的 `node_size_y` 中位數;約 `0.007·L_R` @adaptec1)、`τ_m = m/2`、權重 `ε` 用與 λ_io 相同的 auto-normalization(`ρ_margin` 預設 0.05)、只在 `of < of_margin = 0.15`(GP 末 ~15%)啟用。
- **開關本身是離散事件 ⇒ 必須配 §6.4 的 secant refresh。**

**T2 就實作(不是 T11)**,flag 預設關(`ρ_margin = 0`);T7 用第 11 臂(A7)做 pilot。驗收測試:

1. **有限差分方向測試:** 單 cell 置於距邊界 `D ∈ {0.1m, 0.5m, m, 3m, 10m}`,斷言 `∂L_margin/∂D < 0`(力指向內部)且 `D=10m` 時 `|∂L/∂D| < 1e−6`(遠處無作用)。
2. **符號回歸測試:** 斷言 v1 的錯誤形式 `softplus(−d/τ_m)` 在同一 fixture 下方向**相反**(把錯誤釘在測試裡,避免回歸)。
3. **GP→LG retention 測試(整合,`simple` 或 adaptec1 小跑):** 開 margin 後 `lg_loss` 不增加。

### 9.3 S3 span proxy 的最小形狀(條件 task T10)

grid / lattice-aligned rectilinear 的 region 邊界是一組軸對齊直線 `X = {X_1..X_{nx−1}}`、`Y = {Y_1..Y_{ny−1}}`(adaptec1 k=16 只有 3+3=6 條)。用 DP 既有的 WA soft min/max:

```
L_S3 = Σ_e w_e · [ Σ_j σ((xmax_e − X_j)/τ_s)·σ((X_j − xmin_e)/τ_s)
                 + Σ_j σ((ymax_e − Y_j)/τ_s)·σ((Y_j − ymin_e)/τ_s) ]
```

即「net 的 soft bbox 跨過第 j 條界線」的機率和。**成本 O(E × (nx+ny))**,沒有 `(P,K)` 張量,記憶體只有 S2 的 ~1/K,而且它直接近似 **crossing 數**(不是 λ−1)——這正是 F2/F4a 觸發時需要的東西。最小實作 = `ioplace/ops/span_term.py` + 沿用同一個 patch/driver/報表管線。

### 9.4 S5 FD 假想搬移的最小形狀(最後手段)

在既有的 evaluator callback 內(**不需要任何梯度路徑**):
1. 從 `RegionGrid` 取每顆 movable cell 的當前 region 與距最近邊界的距離,選出邊界帶內的 cell 集合 B。
2. 對 B 中每顆 cell、每個相鄰 region r,用 evaluator 的 per-net bitmask 增量算 `Δ(io + ft)`。
3. 把 `−Δcost` 轉成一個朝該 region 的位移量,**直接寫進 `pos`**(投影式更新,非梯度),幅度受 density 限制。

因為它完全活在 callback 裡(callback 在 step 之後,正好適合做位置修正),S5 是**唯一能沿用 M1 現有 patch 而不需要新掛鉤**的備援;但它是 refinement 而非 objective,**且會使 `pos` 與 optimizer 的 `u_k`/`g_k` 快取脫節,啟用時同樣必須呼叫 §6.4 的 refresh**。

---

## 10. Q9 — 風險清單

| # | 風險 | 證據/量化 | 緩解 |
|---|---|---|---|
| R1 | **梯度稀釋(兩端)** | `‖∇L_IO‖₁` 在 τ_rel=0.5 只有峰值的 28%,在 0.01 是 81% | 退火窗口鎖在 [0.03, 0.30];F1 監控;T7 掃描 |
| R2 | **深處 cell 的梯度死區** | τ_rel=0.02 時 45.8% 的 cell `p_max>1−1e−7`,IO 梯度恆 0 | 這是**物理真實**不是 bug;但代表 S2 只能做邊界附近的微調,大搬移要靠早期(大 τ)。F1 觸發 → S5 |
| R3 | **NaN** | 無 guard 的 naive autograd 在 τ_rel=0.02 產生 193,798 個 NaN(45.8% 的梯度向量) | 手寫 backward(§3.2.2)+ `t`/`omp` 的 `clamp_min(1e-30)` + T2 的 NaN 回歸測試 |
| **R4** | **【改寫,H4】LG 抹平 GP 收益 + margin 符號** | M1 軌跡實測:iter 600 → LG 後,io 從 27,557→30,295(**+9.9%**),控制組 28,124→30,044(+6.8%);M1 的 GP 期優勢(−567)在 LG 後變成劣勢(+251)。**v1 的 margin 公式符號寫反,會把 cell 推向邊界、放大而非縮小流失** | (a) 每個 run 記 `io_gp`/`io_count`/`lg_loss` 三欄;(b) **F2 改用 `io_gp` 判定**,LG 流失獨立為 F5,避免把 GP 成功誤判為 surrogate 失效;(c) margin 項改為 `softplus((d*+m)/τ_m)` 並在 **T2 就實作**、T7 第 11 臂 pilot;(d) 符號有限差分測試 + 把 v1 錯誤形式釘進回歸測試 |
| **R5** | **【改寫,H5】規模契約** | materialized 原型:bigblue4 K=32 = 9.23 GB;外推 10M/K=32 = **~42 GB**,在 placer+evaluator 之前就爆 | **不交付 materialized 版本。** T2 第一天即實作 §2.5 的 chunked-k 四趟契約(常駐只有 `(N,)`×3 + `(E,)`),chunked vs unchunked 等價測試;**T2 Step 6 的 10M 合成拓撲 spike 是凍結介面的前置條件**(目標峰值 ≤6 GB,含 placer+evaluator 的合併量測)。bigblue4 K=32 若合併峰值 >18 GB,callback 結束即 `del GpuEvalContext` |
| R6 | **runtime** | line search 每 iteration 呼叫 `obj_and_grad_fn` 1–10 次;IO 項 adaptec1 12.5 ms、bigblue4 162 ms/次(materialized);chunked 版 FLOP ×2 | 接受 ~1.8–2.3× GP 時間;選配「net activity mask」(只對至少一顆 pin 落在邊界 `m·τ` 帶內的 net 建項,理論等價到 `exp(−m)` 精度)—— T2 選配 step,預設關閉並用等價測試把關 |
| R7 | **退火敏感 / 步長不穩** | IO 項曲率 ~`λ_io/τ²`,小 τ + 大 ρ 時可壓垮 `α = ‖Δv‖/‖Δg‖` | Lipschitz 平價護欄 `λ_io ≤ τ²/γ`(§5.2);F3 監控 backtrack 次數 |
| R8 | **fixed macro 被搬動** | Nesterov 更新整個 `pos` 向量;WL op 靠 `pin_mask_ignore_fixed_macros` 保護,我們沒有 | backward 明確把 `idx ≥ num_movable` 的梯度清零;T2 單元測試鎖死 |
| R9 | **測不出訊號(噪聲)** | M1 實測同設定重跑 io 差 1.0%,與其宣稱的改善同量級 | T6 先量 `σ_rep(det=0/1)`/`σ_seed`;exit 門檻 `3σ_rep` 且 ≥5%;最佳臂跑 3 次;`S` 用 fp64 累加降低 atomic 順序抖動 |
| R10 | **detailed placement 慣例** | M0/M1 一律 `detailed_place_flag=0`(`run_placement.py:43-44`) | M2 沿用,不引入新變數 |
| R11 | **`Σ(Λ−1)` 與 `io_count` 脫鉤** | 目前份額 1.19–1.36(良好),但只有 3 個樣本、且會因幾何品質劣化(two_stage 是 1.80–2.48) | F4a 監控 hard-λ vs 精確 io;GP 末端 rank correlation;必要時轉 S3 |
| **R12** | **【新增,H1】Nesterov secant 跨 objective 汙染** | `g_k` 跨 iteration 快取;objective 跳變 ⇒ `alpha` 塌陷 ⇒ backtrack 增加(`NesterovAcceleratedGradientOptimizer.py:129/141`)。**實測(§6.4.1):步長塌陷 11,000×、obj 求值 ×3;刷新後步長變動僅 +0.04%** | τ/ρ 連續化 + λ ramp-in(§4.2/§5.2);離散事件強制 `refresh_nesterov_secant`(§6.4);版本號 invariant + T4 RED/GREEN 測試 |
| **R13** | **【新增,M2】`params` 序列化污染** | `Params.toJson()` 迭代 `__dict__`,`dump()` 會 `json.dump`(`Params.py:109-131`);掛 callable 上去會炸 | 屬性名用 `_extra_obj_terms`,patch 讓 `toJson` 跳過底線開頭的 key;T4 測試 `params.toJson()`/`dump()` 在掛載後仍可序列化 |

---

## 11. Q10 — Task 分解草圖(10 核心 + 2 條件)

規約:每個 task = 一次 TDD 迴圈(先寫失敗測試 → 實作 → 跑綠 → commit);測試一律 `$DP/.venv312/bin/python -m pytest`;單元測試用手工合成 case,不依賴大 benchmark;新程式碼全部在本 repo `ioplace/`,DP 改動只允許 §6.2 的 python-only patch。

### T0 — 【新增,M3】把 M2 的數值證據變成可重現資產

- **檔案:** 建 `ioplace/diagnostics/probes_m2/{__init__.py, probe1_numerics.py … probe8_secant.py, run_all.py}`;輸出 `results/m2/probes/*.json`
- **內容:**
  - 8 個探針的**修正版**——特別是 **probe3 的 `grad_loo` 實作有誤(多了一個 `1/(1−p)` 因子,且 `clamp_min` 阻斷梯度),必須修好**;修好後重跑,若梯度數字與本文 §3.2.2 記載不一致,**以重跑值為準並更新本文**(NaN 計數預期不變,那部分與該 bug 無關)。
  - `run_all.py` 記錄 env metadata:`torch.__version__`、CUDA runtime、GPU 名稱、`$DP` 的 git commit、`ioplace` 的 git commit、每個輸入 `.npz` 的 sha256。
  - 決定性檢查降為 pytest 回歸 fixture:`tests/test_probes_regression.py` 用**小型合成 case**(非大 benchmark)鎖住 (a) 無 guard naive autograd 產生 NaN、(b) 穩定式不產生 NaN、(c) `Σ(Λ−1) ≤ io_count`。
- **驗證:** `run_all.py` 跑完產出 8 個 JSON;`pytest tests/test_probes_regression.py -v` 全綠;本文 §2.3/§3.2.2/§3.2.5/§4.1 的每個數字都能在 JSON 中找到對應項(不符即更新本文並在 commit message 註明)

### T1 — S1 軟歸屬模組
- **檔案:** 建 `ioplace/ops/__init__.py`, `ioplace/ops/soft_assign.py`;測 `tests/test_soft_assign.py`
- **介面:** `rect_table(region_set) -> (rects (R,4), rect2region (R,))`;`region_sdf_l1(x, y, rects, rect2region, k_slice) -> (N,c)`(**只算指定的 region chunk**);`softmax_stats(x, y, ..., tau) -> (m (N,), t (N,), argmax (N,))`(FWD-1);`chunk_p_ell(x, y, ..., m, t, k_slice, tau) -> (p (N,c), ell (N,c))`(§2.2)
- **測試:** ① 單矩形內/外/角落的解析值;② 兩矩形 L 形 region 的 hard-min;③ τ→0 時 `argmax_k p` 逐點等於 `RegionGrid.region_of_points`(隨機 1000 點,grid 與 slicing 各一);④ `ell` 與 `torch.log1p(-p)` 在非飽和區一致到 1e−6、飽和區有限且 ≥ −69;⑤ **chunked 的 `(m,t,argmax)` 與一次算完全 K 的版本逐位元相同**;⑥ 有限差分梯度檢查(fp64);⑦ 極端 τ(1e−4·L_R, 10·L_R)無 NaN/Inf
- **驗證:** `pytest tests/test_soft_assign.py -v` 全綠

### T2 — IO surrogate 項(核心;chunked 契約 + margin 項 + 規模 spike)
- **檔案:** 建 `ioplace/ops/io_term.py`;測 `tests/test_io_term.py`
- **Step 1 — 拓撲建構:** `build_net_node_csr(nl, ignore_net_degree) -> (flat_net2node, net2node_start, net_ids, degrees, deg_bucket)`,**向量化去重**(對 `net*num_nodes + node` 打包後 `np.unique`,禁用 per-net python 迴圈);同時輸出 bigblue4 的 degree 分布統計(§3.1 需要)
- **Step 2 — chunked forward/backward:** `class IoTerm(torch.nn.Module)` + `class _IoFn(torch.autograd.Function)`,嚴格照 §2.5 的四趟結構;`S` 用 fp64 累加;`num_movable` 以上梯度清零;暴露 `last_grad_l1`、`last_soft_lambda_sum`、`grad_share_by_degree_bucket`
- **Step 3 — margin 項:** `margin_penalty(d_star, m, tau_m)`(§9.2),flag 預設 `rho_margin=0`
- **Step 4 — 選配 net activity mask**(R6),預設關閉
- **Step 5 — 效能量測:** adaptec1/bigblue4 × K∈{8,16,32} 的 chunked fwd/bwd 時間與峰值,回填 §2.3
- **Step 6 — 【H5 前置門檻】10M feasibility spike:** 合成 N=10M / E≈12M / P≈40M / K=32(degree 分布仿 bigblue4)的拓撲,在 L4 上實測 chunked 峰值與單次 fwd+bwd 時間;**峰值 > 8 GB 或 OOM 即視為未通過,不得凍結介面**
- **測試:** ① 3-node/2-net 玩具,`q`/`λ`/`L` 手算比對;② 對 fp64 autograd 參考版做 `gradcheck`;③ **chunked(c=1,2,3) vs unchunked(c=K)在小 case 上 `L` 與梯度一致到 1e−10(fp64)/1e−6(fp32)**;④ **NaN 回歸**:用 probe1 的病態組態斷言無 NaN,且 naive 參考路徑產生 NaN(RED 態存證);⑤ τ→0 時 `L_IO` 收斂到 `RegionGrid` 硬算的 `Σ(Λ−1)`(相對誤差 <1%);⑥ `pos[num_movable:]` 梯度恆 0;⑦ same-node 多 pin 只計一次;⑧ net mask 與 DP 的 `net_mask_ignore_large_degrees` **逐位元相同**;⑨ **margin 有限差分方向測試 + v1 錯誤形式的反向回歸**(§9.2);⑩ activity mask 開/關等價
- **驗證:** `pytest tests/test_io_term.py -v` 全綠;Step 6 spike 通過;adaptec1 真實座標 smoke 對照 §4.1 表(±5%)

### T3 — 排程與 auto-normalization
- **檔案:** 建 `ioplace/schedules.py`;測 `tests/test_schedules.py`
- **介面:** `tau_rel_from_overflow`、`rho_from_overflow`(含 `ramp`)、`autonorm_ratio`(EMA)、`lipschitz_cap`;`@dataclass ScheduleState`(含 `obj_version`、`refreshed_version`、`active`、`tau`、`lambda_io`、`rho_margin`、`w`)
- **測試:** 端點值、單調性、clip、EMA 遞推、ramp 邊界、`lip_cap` 生效、除零保護、**`obj_version` 只在離散事件遞增(連續的 τ/ρ 更新不遞增)**
- **驗證:** 純函數測試全綠

### T4 — DP patch + secant 失效機制 + optimizer lock
- **檔案:** 建 `ioplace/dp_hook.py`, `ioplace/dp_patch/m2-extra-obj-terms.patch`;測 `tests/test_dp_hook.py`
- **介面:** `attach_terms(params, terms)`(寫 `params._extra_obj_terms`)、`assert_optimizer_lock(params, placedb)`、`refresh_nesterov_secant(optimizer)`(§6.4.2)、`install_version_invariant(optimizer, state)`(測試用)
- **Step 1:** 套 patch 到 `$DP` 的 `io-aware` branch + 同步 `install/`,diff 存本 repo
- **測試:**
  - ① **secant 單元(§6.4.3)**:真 optimizer + 2 維二次問題,RED(不 refresh)/GREEN(refresh)兩態都跑。**原型已驗證可行**(T0 的 `probe8_secant.py`),直接改寫成測試即可;注意 `NAG.step()` 開頭有 `if p.grad is None: continue`,測試必須先呼叫一次 `obj_and_grad_fn(p)` 把 `p.grad` 灌好(DP 主迴圈也是這樣做的),否則 optimizer state 永遠不會初始化
  - ② **版本號 invariant**:切換 objective 但不 refresh ⇒ invariant 觸發 AssertionError;refresh 後 ⇒ 不觸發
  - ③ **optimizer lock**:`Lsub_iteration=2` 或 `use_bb=1` 或非 nesterov ⇒ `assert_optimizer_lock` raise
  - ④ **obj 值不影響行為(M1)**:同一 seed 跑兩次 `simple`,其中一次額外加常數項 `+1e6` 到 objective,斷言 iteration 數與最終 `pos` 逐位元相同
  - ⑤ **序列化(R13)**:掛載 terms 後 `params.toJson()`/`params.dump(tmpfile)` 不拋例外,且 dump 出的 JSON 不含 terms
  - ⑥ **隔離性**:同一 process 先跑 flat(未掛 terms)再跑 io(掛 terms),斷言 flat 的 `obj_fn` 未被加項;順序反過來再測一次
  - ⑦ `@pytest.mark.slow` 的 `simple` 整合:`obj_fn` 被呼叫 >0 次、`pos.grad` 有限、跑完不 crash
- **驗證:** `pytest tests/test_dp_hook.py -v`(含 slow)

### T5 — evaluator 擴充 + driver `--mode io`
- **Step 1(獨立 TDD 迴圈)— evaluator 擴充(H3):** `EvalResult` 新增 `hard_lambda_sum: int`、`per_net_lambda: np.ndarray(int32)`;`evaluator_ref` 與 `evaluator_gpu` 同步實作。**既有欄位必須逐位元不變**——用既有的 GPU/CPU 等價測試 + 新增的「新欄位等價 + 舊欄位不變」測試把關
- **Step 2 — driver:** 建 `ioplace/drivers/run_placement_io.py`;改 `ioplace/drivers/run_placement.py`(新增 mode 與旗標);測 `tests/test_driver_io.py`
  - `run_io(config, k, rtype, seed, out_json, *, rho_max, tau_hi, tau_lo, of_on, of_end, alpha_io, ignore_net_degree_override, rho_margin, every, dp_seed, deterministic)`
  - CLI:`--mode io --rho-max --tau-hi --tau-lo --alpha-io --rho-margin --d-max --dp-seed --deterministic`
  - **必做的既有缺口修補:** `--dp-seed` 直接寫 `params.random_seed`(§7.2);`--deterministic` 寫 `params.deterministic_flag`
  - 啟動時呼叫 `assert_optimizer_lock`
  - **每 iteration**:更新 τ/ρ(連續);**每 N iter callback**:evaluator 精確計數(`io_count`/`ft_count`/`hard_lambda_sum`/`per_net_lambda`)→ `ratio` 重算 →(選配)`w_e` 校準 → **`refresh_nesterov_secant`** → 記軌跡;GP 結束後另記一次 `io_gp`,LG 後記 `io_count`
- **測試:** ① evaluator 新欄位的 CPU/GPU 等價 + 舊欄位不變;② JSON schema(§8.2 欄位齊全);③ `simple` slow smoke;④ `--dp-seed` 確實改變最終座標;⑤ callback 觸發次數 = `iterations/every`;⑥ 版本號 invariant 全程零違反
- **驗證:** `pytest tests/test_driver_io.py tests/test_evaluator_ref.py tests/test_evaluator_gpu.py -v`(含 slow)

### T6 — 噪聲底線與 `deterministic_flag` regime 決定(擋在掃描之前)
- **檔案:** 建 `ioplace/diagnostics/measure_noise_floor.py`;輸出 `results/m2/noise/*.json`
- **內容:** adaptec1 k=16 grid flat × 5(det=0)+ × 5(det=1)+ × 5 個 `--dp-seed`;報 `mean/std/min/max` 的 `io_count`/`ft_count`/`hpwl`/`runtime_s`
- **驗證:** 依 §7.2 的規則機械決定 regime 並寫進 M2 報告第一節;若 `σ_rep(det=1)/mean > 1%`,**停下來 root-cause**,不得繼續 T7

### T7 — ρ × τ 掃描 + margin pilot → Pareto 前緣
- **內容:** adaptec1 k=16 grid,`ρ_max ∈ {0.02,0.05,0.10,0.20,0.40}` × `τ-schedule ∈ {annealed, fixed 0.10}` = 10 臂,**加第 11 臂 A7(margin pilot)**;每臂記完整軌跡:`io`/`io_gp`/`hard_lambda_sum`/固定參考 τ 的 soft λ/`‖∇L_IO‖₁`/`λ_io`/`τ`/backtrack 次數/F1 的可動 cell 比例/**per-degree-bucket 的 `grad_share`、`io_share`、`lambda_share`**;GP 末端算 Spearman ρ
- **驗證:** Pareto 表 + 圖;機械判定 F1/F2/F2b/F3/F4a/F4b/F5;選出 `(ρ*, τ-schedule*)`;最佳臂重跑 3 次

### T8 — Ablation + 規模/形狀確認
- **內容:** §8.1 的 A0–A7 依「跑哪些 case」欄執行;**A4 與 A6 必須含 bigblue4**
- **驗證:** `make_report.py` 產出的表格欄位齊全、每格都有對應 JSON;per-degree-bucket 診斷表產出

### T9 — M2 報告與 exit 檢核
- **檔案:** 建 `docs/results/m2-differentiable-io-report.md`
- **內容:** regime 與噪聲底線 → Pareto 前緣 → 三方對照(flat / M1 / M2)→ ablation 表(含 A4 對 §3.2.1 預設的裁決)→ `io_gp` vs `io_count`(LG 影響 + A7 pilot)→ hard-λ alignment 與 Spearman ρ → F1–F5 判定 → **對照 spec §9 M2 的兩個 exit 條件逐條打勾/打叉**,FAIL 就照 M1 報告的誠實慣例如實記錄 + 診斷
- **驗證:** 報告中每個數字都能追到 `results/m2/*.json`;若 A4 推翻 `w_e=1` 預設,**同時更新本設計文件 §3.2.1 與 spec 偏離註記**

### T10(條件:F2 或 F4a 觸發)— S3 span proxy 臂
- **檔案:** `ioplace/ops/span_term.py` + `tests/test_span_term.py`;沿用 T4/T5 管線
- **內容:** §9.3 的公式;掃 `ρ_max` 3 組 + `τ_s` 2 組
- **驗證:** 與 S2 在同一張 Pareto 圖上比較

### T11(條件:F5 觸發,或 A7 pilot 顯示 `lg_loss` 明顯改善)— margin 深掃
- **內容:** `m ∈ {1, 2, 4} × row_height` × `ρ_margin ∈ {0.02, 0.05, 0.15}` × `of_margin ∈ {0.30, 0.15}` 的部分因子設計(6–9 組)
- **驗證:** `lg_loss` 相對 flat 的惡化幅度(baseline +6.8%~+9.9%)顯著縮小,且 `Δhpwl` 不超過 exit 上限

---

## 12. 依賴序

```
T0 ──> T1 ──> T2 ──┬─> T4 ──> T5 ──> T6 ──> T7 ──┬─> T8 ──> T9
                   └─> T3 ──┘                     ├─> T10 (條件 F2 / F4a)
                                                  └─> T11 (條件 F5 / A7 有效)
```

- **T0 必須最先**:所有數值依據在被實作採信之前,要先在 repo 內可重現(M3)。
- **T2 Step 6(10M spike)是 T4 的前置門檻**:介面未證明可擴展就不凍結。
- **T6 必須在 T7 之前**:沒有噪聲底線就無法判讀掃描結果(M1 的教訓)。

---

## 13. 證據附錄

**狀態:v2 起,以下探針由 T0 提交進 `ioplace/diagnostics/probes_m2/`,原始輸出落 `results/m2/probes/*.json`,含 env metadata 與輸入 hash。在 T0 完成之前,下表數字視為「待重現」而非既定事實。**

| 探針 | 內容 | 支撐的結論 | T0 待辦 |
|---|---|---|---|
| `probe1_numerics.py` | 合成病態組態(深處同 region 的多 pin net) | naive autograd 一律 NaN;leave-one-out 係數為 O(1) → §3.2.2 | 原樣提交 + 轉成 pytest fixture |
| `probe2_scale.py` | adaptec1 真實最終座標,τ 掃描,fwd/bwd 時間與峰值記憶體 | §2.3 的 adaptec1 數字、死區統計 → R2 | 原樣提交 |
| `probe3_grad.py` | 無 clamp 的 naive autograd 的 NaN 計數 | 14,674 / 193,798 個 NaN → R3 | **必須修好 `grad_loo`**(多餘的 `1/(1−p)` 因子 + `clamp_min` 阻斷梯度);修好後重跑,梯度數字若與本文不符即更新本文(NaN 計數與該 bug 無關,預期不變) |
| `probe4.py` | 正確的 leave-one-out 梯度 + WA-WL 梯度 L1 norm,τ 掃描 | §4.1 的表、§5.2 的 λ_io 錨點 | 原樣提交 |
| `probe5.py` | clamp 版 naive vs 穩定版 vs fp64 參考,逐元素 | §3.2.2 的「加了 clamp 之後一致到 2e−7」 | 原樣提交 |
| `probe6_mem.py` | bigblue4 K∈{8,16,32} 的時間/記憶體 | §2.3 的 bigblue4 表、R5 | 原樣提交;T2 Step 5 補 chunked 版對照 |
| `probe7_gap.py` | `io_count` vs `Σ(Λ−1)` | §3.2.5 的 1.19–1.36 **份額** | 補上 k=8 這格(v1 該格未輸出);補 Spearman ρ |
| `probe8_secant.py` | 真 `NesterovAcceleratedGradientOptimizer` + objective 跳變,刷新/不刷新兩態 | §6.4.1 的步長塌陷 11,000× → R12、T4 測試 ① 的直接原型 | 原樣提交;**直接升級成 `tests/test_dp_hook.py` 的 RED/GREEN 單元測試**(T4 測試 ①) |

其餘引用來自原始碼閱讀:`$DP/dreamplace/PlaceObj.py`(:69-101 precondition、:206 gamma 初值、:293-320 `obj_fn`、:394-411 `obj_and_grad_fn`、:767-825 `initialize_density_weight`、:901-914 `update_gamma`)、`$DP/dreamplace/NonLinearPlace.py`(:113-124 `construct_model`、:173 optimizer 綁定、:231 patch 錨點、:358-372/:702 `Lsub_stop_criterion`、:450/:458 step、:517-519 M1 callback)、`$DP/dreamplace/NesterovAcceleratedGradientOptimizer.py`(:23/:50/:60 `use_bb`、:65-166 `step_nobb`、:128-142 line search 與 secant)、`$DP/dreamplace/PlaceDB.py`(:819-837 `use_bb="auto"` 的解析)、`$DP/dreamplace/BasicPlace.py`(:159-163 `net_mask_ignore_large_degrees`)、`$DP/dreamplace/Params.py`(:109-131 `toJson`/`dump`)、本 repo `ioplace/evaluator_ref.py`、`ioplace/drivers/run_placement.py:43-44,102`、`ioplace/drivers/run_placement_reweight.py:31-40`、`ioplace/reweight.py:4-14`。
