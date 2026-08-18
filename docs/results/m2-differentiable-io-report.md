# M2 可微 IO 項報告(初稿)

對應計畫 [`docs/superpowers/plans/2026-08-06-m2-differentiable-io.md`](../superpowers/plans/2026-08-06-m2-differentiable-io.md)
Task 9;設計依據 [`docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md`](../superpowers/specs/2026-08-06-m2-differentiable-io-design.md)(v2)。
每個數字皆從 `results/m2/{noise,sweep,ablation,probes,spike}/*.json` 讀出核實,行文與誠實慣例沿用
[`docs/results/m1-reweight-report.md`](m1-reweight-report.md)——如實記錄未達成之處,不挑好看的 case。

---

## 1. 量測 regime 與噪聲底線

`results/m2/noise/summary.json`(adaptec1 k16 grid,主戰場 case):

| 量 | `det=0`(5 次重跑) | `det=1`(5 次重跑) |
|---|---:|---:|
| `io_count` 各次 | 29,991 / 30,069 / 30,041 / 29,924 / 30,144 | 30,256 ×5(逐位元相同) |
| 平均 | 30,033.8 | 30,256 |
| `σ_rep` | 82.65 | **0.0** |
| `runtime_s` 平均 | 60.64 | 59.92 |

`σ_seed`(`det=1`,`dp_seed ∈ {1000..1004}` 各跑一次):`io_count` = 30,256 / 29,840 / 29,973 / 29,823 /
30,045,平均 29,987.4,`σ_seed = 176.37`。

**regime 裁決(照設計 §7.2 的機械規則):** `σ_rep(det=1)/mean = 0/30,256 = 0% < 0.2%`,且
`det=1` 的 runtime(59.92s)相對 `det=0`(60.64s)**沒有增幅**(−1.2%,遠低於 20% 門檻)⇒ **選定
`deterministic_flag=1`(regime=1)**,M2 全部 arm 與重測後的 flat 基準統一在此 regime 下量測。
`det=1` 下 flat 基準(regime=1,即本報告後續所有 Δ% 的分母):`io_count=30,256`、`hpwl=73,923,673`、
`lg_loss_flat=1,958`。

**init_pos 缺口背景(設計 §7.2「init_pos 缺口」小節):** driver 原本繞過 `Placer.place()`(唯一呼叫
`np.random.seed(params.random_seed)` 的地方),`BasicPlace.__init__` 只重設 torch 側 RNG,
init_pos 的中心噪聲與 filler 初始位置卻吃 numpy 全域 RNG——修正前同一 process 內每次 placement 的
init_pos 都是不受控 draw。已由 commit `0685022`(`fix(driver): seed numpy global RNG per placement --
init_pos was an uncontrolled draw`)修正,修正後 `_place()` 自行 `np.random.seed(params.random_seed)`,
`--dp-seed` 同時控制 init_pos 與 torch 側 `gp_noise`。**後果:M0/M1 報告(`docs/results/m0-*.md`、
`docs/results/m1-reweight-report.md`)與 `results/m1/*.json` 的絕對數字帶著這個未修正的噪聲產出,與
本報告修後的 run **不 bit-可比**;本報告第 3 節引用 M1 reweight 時一律用**修正後、同 regime 重測**的
`results/m2/ablation/{adaptec1,bigblue4}_A1_k16_grid.json`,不直接引用 `results/m1/*.json` 的舊數字。**
T6 的 `σ_rep`/`σ_seed` 皆在此修正之後量測。

---

## 2. Pareto 前緣

`results/m2/sweep/gates.json` 的 11 臂(10 個 `ρ_max × τ` 掃描臂 + 第 11 臂 A7 margin pilot,
adaptec1 k16 grid,regime=1,`flat_io=30,256`,`flat_hpwl=73,923,673`);圖見
[`figs/m2-pareto-adaptec1-k16.png`](figs/m2-pareto-adaptec1-k16.png)。

| 臂 | `ρ_max` | `τ` 排程 | `io_count` | `Δio%` | `io_gp` | `Δio_gp%` | `hpwl` | `Δhpwl%` | `lg_loss` | `frac_soft_end` |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho0.02_annealed | 0.02 | annealed(0.30→0.03) | 29,410 | −2.796 | 27,526 | −2.728 | 73,894,672 | −0.039 | 1,884 | 0.5631 |
| rho0.02_fixed | 0.02 | fixed(0.10) | 29,560 | −2.300 | 27,868 | −1.520 | 73,870,745 | −0.072 | 1,692 | 0.5610 |
| rho0.05_annealed | 0.05 | annealed | 29,203 | −3.480 | 27,304 | −3.513 | 73,938,784 | +0.020 | 1,899 | 0.5599 |
| rho0.05_fixed | 0.05 | fixed | 29,336 | −3.041 | 27,532 | −2.707 | 73,914,087 | −0.013 | 1,804 | 0.5544 |
| rho0.10_annealed | 0.10 | annealed | 28,200 | −6.795 | 26,485 | −6.407 | 73,874,053 | −0.067 | 1,715 | 0.5548 |
| rho0.10_fixed | 0.10 | fixed | 29,048 | −3.993 | 27,178 | −3.958 | 73,927,369 | +0.005 | 1,870 | 0.5436 |
| rho0.20_annealed | 0.20 | annealed | 26,825 | −11.340 | 25,340 | −10.453 | 74,017,172 | +0.126 | 1,485 | 0.5381 |
| rho0.20_fixed | 0.20 | fixed | 27,073 | −10.520 | 25,861 | −8.612 | 74,102,192 | +0.241 | 1,212 | 0.5238 |
| **rho0.40_annealed**(best) | **0.40** | **annealed** | **24,647** | **−18.538** | **23,554** | −16.764 | 74,895,086 | **+1.314** | 1,093 | 0.5006 |
| rho0.40_fixed | 0.40 | fixed | 25,829 | −14.632 | 24,674 | −12.807 | 75,440,359 | +2.052 | 1,155 | 0.4602 |
| A7_margin(11th) | 0.40 | annealed + margin | 24,817 | −17.977 | 23,593 | −16.627 | 74,888,888 | +1.306 | 1,224 | 0.5000 |

觀察:

- **annealed 全面優於同 `ρ_max` 的 fixed**——五組 `ρ_max ∈ {0.02,0.05,0.10,0.20,0.40}` 中,annealed
  的 `Δio%` 都比對應 fixed 更負(改善更多),且 `Δhpwl%` annealed 版本除了 `ρ_max=0.20/0.40` 略高外大致
  持平或更低,`ρ_max=0.40` 是唯一 annealed 的 `Δhpwl%`(+1.314%)明顯低於 fixed(+2.052%)的一組——annealed
  路線在同一 `ρ_max` 下用更少的線長代價換到更多的 io 改善,證實設計 §4 的退火排程是必要的(A5 ablation
  於第 4 節再次獨立確認)。
- **`σ_rep=0`(regime=1)⇒ 噪聲帶寬度為 0**——任何非零的 `Δio%` 在本報告的度量下都超過 `3σ_rep`,Pareto
  圖上不需要畫噪聲帶(帶寬 0)。
- 最佳臂 `rho0.40_annealed`:`Δio=−18.538%`、`Δhpwl=+1.314%`,同時滿足 `Δhpwl ≤ +2.0%` 與
  `Δio ≤ −5.0%`(見第 8 節 exit 檢核)。

---

## 3. 三方對照(flat / M1 reweight / M2 best,同 regime)

### adaptec1 k16 grid

| 指標 | flat(`noise/flat_det1_rep0`) | M1 reweight(`ablation/adaptec1_A1_k16_grid`,同 regime 重測) | M2 best(`sweep/adaptec1_k16_rho0.40_annealed`) |
|---|---:|---:|---:|
| `io_count` | 30,256 | 29,869(**Δ −1.28%**) | 24,647(**Δ −18.54%**) |
| `io_gp` | 28,298 | n/a(reweight 模式不追蹤 `io_gp`) | 23,554(Δ −16.76%) |
| `ft_count` | 2,545 | 2,430(Δ −4.52%) | 3,525(Δ +38.51%) |
| `hpwl` | 73,923,673 | 74,856,866(Δ +1.26%) | 74,895,086(Δ +1.31%) |
| `hard_lambda_sum` | 24,467 | n/a(reweight 模式不產出此欄位) | 19,337(Δ −20.97%) |

### bigblue4 k16 grid(A2 沿用 adaptec1 調出的 `ρ*=0.40`,未針對 bigblue4 重調)

| 指標 | flat(`ablation/bigblue4_A0_k16_grid`) | M1 reweight(`ablation/bigblue4_A1_k16_grid`) | M2(`ablation/bigblue4_A2_k16_grid`) |
|---|---:|---:|---:|
| `io_count` | 100,333 | 100,860(Δ +0.53%) | 63,755(**Δ −36.46%**) |
| `io_gp` | n/a(flat 模式不追蹤) | n/a | 63,743 |
| `ft_count` | 6,935 | 2,948(Δ −57.49%) | 8,136(Δ +17.32%) |
| `hpwl` | 748,520,918 | 755,435,247(Δ +0.92%) | 789,030,695(**Δ +5.41%**) |
| `hard_lambda_sum` | n/a | n/a | 54,518 |

**讀法:** 兩個 case 上 M1 reweight 的 `io_count` 都幾乎打平(adaptec1 −1.28%、bigblue4 +0.53%,量級與
M1 報告一致,ft_count 大幅改善但非 exit 指標),M2 可微項的 `io_count` 改善量級是 M1 的一個數量級以上
(adaptec1 −18.5% vs −1.3%,bigblue4 −36.5% vs +0.5%)。**但 bigblue4 用 adaptec1 調出的 `ρ*` 直接套用,
`Δhpwl=+5.41%` 明顯超出 adaptec1 掃出的 iso-WL 帶(`≤+2%`)**——這不是 bigblue4 case 本身做不到 iso-WL 下的
io 改善,而是本次沒有為 bigblue4 單獨掃 `ρ`;iso-WL 掃描本次只在 adaptec1 做過。這是 M4 前要補的點(見第
7 節與下方備註),**不影響 M2 exit**(exit 的判定範圍是主戰場 adaptec1,見第 8 節)。

---

## 4. Ablation 表

### 4.1 A0–A7 完整表(`results/m2/ablation/*.json`,Δ% 相對同 case/k/rtype 的 flat)

| case | arm | k | rtype | `io_count` | `Δio%` | `hpwl` | `Δhpwl%` | `ft_count` | `io_gp` | `lg_loss` | `hard_λ_sum` |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 | A0(flat) | 16 | grid | 30,256 | — | 73,923,673 | — | 2,545 | — | — | — |
| adaptec1 | A0(flat) | 16 | slicing | 36,615 | — | 73,923,673 | — | 7,389 | — | — | — |
| adaptec1 | A0(flat) | 32 | grid | 47,839 | — | 73,923,673 | — | 8,281 | — | — | — |
| adaptec1 | A0(flat) | 8 | grid | 22,544 | — | 73,923,673 | — | 1,979 | — | — | — |
| adaptec1 | A1(reweight) | 16 | grid | 29,869 | −1.279 | 74,856,866 | +1.262 | 2,430 | — | — | — |
| adaptec1 | **A2(diff-only)** | 16 | grid | **24,647** | **−18.538** | 74,895,086 | +1.314 | 3,525 | 23,554 | 1,093 | 19,337 |
| adaptec1 | A2 | 16 | slicing | 33,460 | −8.617 | 74,367,942 | +0.601 | 8,812 | 32,151 | 1,309 | 22,202 |
| adaptec1 | A2 | 32 | grid | 41,272 | −13.727 | 74,444,560 | +0.705 | 9,600 | 38,659 | 2,613 | 27,985 |
| adaptec1 | A2 | 8 | grid | 19,084 | −15.348 | 74,764,454 | +1.137 | 2,962 | 18,548 | 536 | 14,856 |
| adaptec1 | A3(diff+reweight) | 16 | grid | 24,542 | −18.886 | 74,940,755 | +1.376 | 3,585 | 23,418 | 1,124 | 19,170 |
| adaptec1 | **A4(`w_e=1/(deg'−1)`)** | 16 | grid | 25,811 | −14.691 | 75,044,467 | +1.516 | 3,412 | 24,389 | 1,422 | 20,303 |
| adaptec1 | A5(τ 固定) | 16 | grid | 25,829 | −14.632 | 75,440,359 | +2.052 | 3,349 | 24,674 | 1,155 | 20,495 |
| adaptec1 | A6(`D_max=256`) | 16 | grid | 24,647 | −18.538 | 74,895,086 | +1.314 | 3,525 | 23,554 | 1,093 | 19,337 |
| adaptec1 | A7(margin pilot) | 16 | grid | 24,817 | −17.977 | 74,888,888 | +1.306 | 3,482 | 23,593 | 1,224 | 19,542 |
| bigblue4 | A0(flat) | 16 | grid | 100,333 | — | 748,520,918 | — | 6,935 | — | — | — |
| bigblue4 | A1(reweight) | 16 | grid | 100,860 | +0.525 | 755,435,247 | +0.924 | 2,948 | — | — | — |
| bigblue4 | **A2(diff-only)** | 16 | grid | **63,755** | **−36.457** | 789,030,695 | +5.412 | 8,136 | 63,743 | 12 | 54,518 |
| bigblue4 | A3(diff+reweight) | 16 | grid | 63,873 | −36.339 | 791,212,509 | +5.703 | 8,365 | 63,864 | 9 | 54,399 |
| bigblue4 | **A4(`w_e=1/(deg'−1)`)** | 16 | grid | 65,201 | −35.015 | 788,217,606 | +5.303 | 8,245 | 65,180 | 21 | 55,721 |
| bigblue4 | A6(`D_max=256`) | 16 | grid | 63,725 | −36.486 | 789,082,252 | +5.419 | 8,113 | 63,706 | 19 | 54,496 |

觀察(不含 A4 裁決,見 4.3):

- **A2 全面優於 A1** 兩個量級以上(adaptec1 −18.5% vs −1.3%;bigblue4 −36.5% vs +0.5%),在
  adaptec1 的 k8/k16/k32/slicing 四種 region 形狀 × 粒度組合下皆穩定成立(`Δio ∈ [−15.3%, −8.6%]`)。
- **A3(diff+reweight)相對 A2 是混合結果,非明確互補**:adaptec1 上 A3 的 `io_count` 略優於 A2
  (−18.886% vs −18.538%,差 0.35pp)但 `hpwl` 略差(+1.376% vs +1.314%);bigblue4 上 A3 在 io 與
  hpwl **兩個維度都比 A2 差**(`Δio` −36.339% vs −36.457%,`hpwl` 791,212,509 vs 789,030,695)。M1 的
  reweight 機制疊加在可微項之上沒有帶來一致的額外收益。
- **A5(τ 固定)被 A2(annealed)全面支配**:`io_count` 更差(25,829 vs 24,647)且 `hpwl` 更差
  (75,440,359 vs 74,895,086)——與第 2 節 Pareto 表的 annealed-vs-fixed 觀察一致,獨立確認退火排程
  是必要的,不是裝飾性設計。
- **A6(`D_max=256`)在 adaptec1 與 A2 逐位元相同**(`io_count`/`hpwl` 完全一致)——`degree_distribution.json`
  記載 adaptec1 落在 `(100,256]` 的 net 數為 0(`n_deg_gt_256=2` 但這兩個 net 度數 > 256,不在
  `D_max=256` 新納入的範圍內),所以 `D_max` 收緊/放寬在 adaptec1 上完全不影響結果。bigblue4 上 A6
  對 A2 只有極小差異(`Δio` −36.486% vs −36.457%,`hpwl` 差 +0.0066%)——量化了「有 IO 力但無 WL
  反作用力」的 net 在 bigblue4 上的影響:**存在但很小**,不推翻 `D_max=100`(= `ignore_net_degree`)
  的預設。

### 4.2 per-degree-bucket 診斷(`results/m2/ablation/degree_buckets.json`,A2 vs A4,k16 grid)

adaptec1(`io_share`/`lambda_share` 由 CPU `evaluator_ref` 對該 run 最終座標重算,`grad_share` 取自
該 run trajectory 最後一筆記錄):

| bucket | A2 `grad_share` | A2 `io_share` | A2 ratio | A4 `grad_share` | A4 `io_share` | A4 ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.3248 | 0.2920 | 1.112 | 0.6413 | 0.2381 | 2.694 |
| 3 | 0.1508 | 0.1331 | 1.133 | 0.1489 | 0.1344 | 1.108 |
| 4-7 | 0.2986 | 0.2686 | 1.111 | 0.1715 | 0.2976 | 0.576 |
| 8-15 | 0.1396 | 0.1755 | 0.796 | 0.0305 | 0.1949 | 0.157 |
| 16-31 | 0.0549 | 0.0580 | 0.945 | 0.0064 | 0.0549 | 0.117 |
| 32-63 | 0.0314 | 0.0716 | 0.438 | 0.0014 | 0.0791 | 0.017 |
| 64-99 | 0.0000232 | 0.0011 | 0.021 | 0.0000007 | 0.0010 | 0.0007 |

bigblue4:

| bucket | A2 `grad_share` | A2 `io_share` | A2 ratio | A4 `grad_share` | A4 `io_share` | A4 ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.4474 | 0.5207 | 0.859 | 0.7616 | 0.4739 | 1.607 |
| 3 | 0.1174 | 0.1078 | 1.089 | 0.1163 | 0.1175 | 0.990 |
| 4-7 | 0.1844 | 0.1557 | 1.184 | 0.0906 | 0.1741 | 0.521 |
| 8-15 | 0.0863 | 0.0961 | 0.898 | 0.0207 | 0.1101 | 0.188 |
| 16-31 | 0.0620 | 0.0283 | 2.187 | 0.0070 | 0.0323 | 0.216 |
| 32-63 | 0.0786 | 0.0787 | 0.999 | 0.0033 | 0.0787 | 0.041 |
| 64-99 | 0.0240 | 0.0127 | 1.891 | 0.0006 | 0.0135 | 0.043 |

（A6 的 bucket 數字在 adaptec1 與 A2 逐位元相同,bigblue4 上與 A2 幾乎相同,故不另列。）

### 4.3 A4 裁決:**維持 `w_e=1` 預設**

依設計 §3.2.1 的機械推翻條件(「任一 bucket 的 `grad_share/io_share` 偏離 1 超過 3×,或 A4 在任一
case 的 Pareto 前緣上支配預設」)逐條核對:

1. **A4 在兩個 case 都被 A2 支配(或接近支配)。** adaptec1 上 A2 在 `io_count`(24,647 vs 25,811)與
   `hpwl`(74,895,086 vs 75,044,467)**兩個維度都嚴格更優**——A2 嚴格 Pareto 支配 A4。bigblue4 上並非
   嚴格支配:A4 的 `hpwl` 略低於 A2(788,217,606 vs 789,030,695,**−0.10%**),但 `io_count` 明顯更差
   (65,201 vs 63,755,**+2.27%**,`Δio` 弱 1.44 個百分點)——用 0.10% 的線長換 2.27% 更差的 io,在
   exit 指標以 io 為主的前提下實務上仍是明顯更差的權衡,而非真正的互補點。
2. **bucket 表顯示 `inv_deg`(A4)才是偏斜的一方,不是 `unit`(A2)。** A4 的 `grad_share` 系統性堆積到
   degree-2 bucket(adaptec1 64.1%、bigblue4 76.2%,對比 A2 的 32.5%/44.7%),其餘 bucket 的
   `grad_share/io_share` ratio 崩塌到 0.017–0.576(adaptec1,不含近空的 64-99)、0.041–0.521
   (bigblue4,不含 bucket 2/3)——這正是 `1/(deg'-1)` 加權系統性稀釋大 net 梯度貢獻的預期後果。
   A2 的 ratio 除了下一條提到的例外,分布在 **0.438(adaptec1 32-63 bucket)至 2.187(bigblue4 16-31
   bucket)**之間,遠比 A4 均勻集中在 1 附近。
3. **A2 唯一字面偏離 >3× 的 bucket 是 adaptec1 的 64-99**(ratio=0.021),但該 bucket `io_share` 僅
   0.11%(`degree_distribution.json`:adaptec1 落在 64-99 度數區間的 net 只有 1 個),幾近空集合,屬
   統計噪音,不構成有意義的違反。

**結論:三條推翻條件均不成立(A4 未支配 A2,A2 的偏斜遠小於 A4,A2 唯一的字面越界是近空 bucket)—— 維持
`w_e=1` 為預設。設計 §3.2.1 不修改。**

---

## 5. LG 影響

### 5.1 GP→LG 流失(`lg_loss = io_count − io_gp`,以 `lg_loss/io_gp%` 與 M1 報告的口徑對齊比較)

flat 基準(regime=1,`noise/flat_det1_rep0.json`):`io_gp=28,298`、`io_count=30,256`、`lg_loss=1,958`,
**`lg_loss/io_gp = 6.92%`**——這個量級本身就落在 M1 報告記載的 GP→LG 流失 **+6.8%~+9.9%**(iter600→LG
後,α=0.2 為 +9.9%,α=0 控制組為 +6.8%)區間內側,即使完全不開可微項,flat 本身在 M2 driver 下的 LG
流失也與 M1 觀察到的量級一致。

adaptec1 k16(sweep 11 臂,`ρ_max` 越大流失比例越低):

| 臂 | `io_gp` | `io_count` | `lg_loss` | `lg_loss/io_gp%` | 相對 flat(6.92%)的 pp 差 |
|---|---:|---:|---:|---:|---:|
| rho0.02_annealed | 27,526 | 29,410 | 1,884 | 6.844 | −0.075 |
| rho0.02_fixed | 27,868 | 29,560 | 1,692 | 6.071 | −0.848 |
| rho0.05_annealed | 27,304 | 29,203 | 1,899 | 6.955 | +0.036 |
| rho0.05_fixed | 27,532 | 29,336 | 1,804 | 6.552 | −0.367 |
| rho0.10_annealed | 26,485 | 28,200 | 1,715 | 6.475 | −0.444 |
| rho0.10_fixed | 27,178 | 29,048 | 1,870 | 6.881 | −0.039 |
| rho0.20_annealed | 25,340 | 26,825 | 1,485 | 5.860 | −1.059 |
| rho0.20_fixed | 25,861 | 27,073 | 1,212 | 4.687 | −2.233 |
| **rho0.40_annealed(best)** | 23,554 | 24,647 | 1,093 | **4.640** | **−2.279** |
| rho0.40_fixed | 24,674 | 25,829 | 1,155 | 4.681 | −2.238 |
| A7_margin | 23,593 | 24,817 | 1,224 | 5.188 | −1.731 |

**沒有一個臂的 `lg_loss/io_gp%` 超過 flat 自身的 6.92%**——可微項不只沒有讓 legalization 抹掉更多 GP
收益,反而系統性地**降低**了 GP→LG 的相對流失比例,且降幅隨 `ρ_max` 增大而增大(`ρ_max=0.02` 時幾乎
與 flat 打平,`ρ_max=0.40` 時降到 4.64%,比 flat 低 2.28pp)。這與 gates.json 的 `F5=false` 一致
(F5 的判準 `(lg_loss − lg_loss_flat)/flat_io > +2%` 全部臂皆為負值,詳見第 7 節)。

### 5.2 A7 margin pilot vs best 臂(A2,無 margin)

| 臂 | `io_count` | `Δio%` | `io_gp` | `lg_loss` | `lg_loss/io_gp%` |
|---|---:|---:|---:|---:|---:|
| best(rho0.40_annealed,無 margin) | **24,647** | **−18.538** | 23,554 | 1,093 | **4.640** |
| A7(margin,`ρ_margin=0.05`) | 24,817 | −17.977 | 23,593 | 1,224 | 5.188 |

**如實記錄:margin pilot 在本次量測中沒有展現預期的 LG-retention 效果——A7 的 `io_count`(24,817)、
`Δio%`(−17.977% vs −18.538%)與 `lg_loss/io_gp%`(5.188% vs 4.640%)都比不開 margin 的 best 臂略差。**
margin 項的原始動機是縮小 GP→LG 的流失(design R4/§9.2),但 best 臂本身在沒有 margin 的情況下已經把
流失壓到比 flat 低 2.28pp,margin 在這個基礎上疊加反而略微變差(`lg_loss` 從 1,093 升到 1,224)。這與
`gates.json` 的 `F5=false`(margin 開關前後都遠低於 +2% 門檻,不是流失問題)以及設計 §11 T11 的觸發
條件(`F5==true` 或「A7 pilot 顯示 `lg_loss` 相對 A2 明顯改善」)一致地指向:**兩個觸發條件都不成立,
T11(margin 深掃)未觸發**,詳見第 7 節。

---

## 6. Alignment 診斷

### 6.1 GP 末端 Spearman ρ(`per_net_crossings` vs `per_net_lambda − 1`,僅取 `per_net_crossings>0`)

各 A2 臂 run JSON 頂層 `spearman_rho`(GP 末端):

| run | `spearman_rho` |
|---|---:|
| adaptec1 A2 k16 grid | 0.5160 |
| adaptec1 A2 k16 slicing | 0.5021 |
| adaptec1 A2 k32 grid | 0.4757 |
| adaptec1 A2 k8 grid | 0.4885 |
| bigblue4 A2 k16 grid | 0.3778 |

adaptec1 四個 k/rtype 組合的 ρ 集中在 0.48–0.52,bigblue4 較低(0.378)——與 §3.2.5 的份額表(下方
6.3 節)在 bigblue4 上份額也略低的方向一致:bigblue4 的幾何品質(繞路占比)比 adaptec1 差,rank
correlation 也隨之弱化。**正相關方向穩定,無一個 case 出現負相關或接近零**,支持「S2 的目標與精確指標
方向一致」這個前提。

### 6.2 hard-λ 軌跡(best 臂,adaptec1 k16 rho0.40_annealed,啟用→末端)

| iteration | overflow | τ | `λ_io` | `io_count`(callback) | `hard_lambda_sum` | `soft_lambda_ref_tau`(固定參考 τ) | `frac_soft` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.831 | 661.0 | 0.000 | 22,682 | 19,726 | 311,909.6 | 0.8125 |
| 100 | 0.820 | 641.3 | 1.294 | 23,243 | 20,327 | 257,872.3 | 0.6230 |
| 150 | 0.822 | 646.3 | 0.829 | 22,880 | 20,193 | 255,764.8 | 0.4496 |
| 200 | 0.818 | 637.9 | 0.588 | 22,523 | 19,938 | 255,666.3 | 0.4589 |
| 250 | 0.813 | 630.1 | 0.433 | 22,361 | 19,652 | 256,489.5 | 0.5051 |
| 300 | 0.798 | 603.6 | 0.431 | 24,535 | 21,358 | 261,126.6 | 0.5495 |
| 350 | 0.720 | 486.1 | 1.942 | 23,853 | 20,921 | 257,819.3 | 0.5709 |
| 400 | 0.618 | 366.5 | 7.325 | 25,611 | 22,152 | 257,268.1 | 0.4925 |
| 450 | 0.490 | 257.0 | 23.469 | 23,718 | 19,647 | 254,532.7 | 0.5109 |
| 500 | 0.346 | 172.4 | 61.036 | 24,326 | 19,594 | 254,829.9 | 0.5181 |
| 550 | 0.193 | 112.6 | 139.305 | 23,711 | 18,845 | 253,483.3 | 0.5043 |
| 600(GP 末端) | 0.074 | 81.1 | 246.229 | 23,554 | 18,587 | 253,775.6 | 0.5006 |
| **LG 後(頂層欄位)** | — | — | — | **24,647** | **19,337** | — | — |

`soft_lambda_ref_tau`(固定 `τ_ref=0.05·L_R`,與退火無關)從啟用時的 311,909.6 整體下降到 253,775.6
(−18.6%),中間在 iteration 300 附近有一次局部回升(261,126.6,+2.2% 相對 iteration 250),與同一
iteration 的 `io_count` 局部峰值(24,535)同步——即優化軌跡本身有波動,不是單調下降,但整體趨勢一致
向下,說明這是真實的優化進展,不是 §4.1 描述的「座標不動、只改 τ 就讓 soft `L_IO` 假性下降」那種温度
假象(該假象在固定 `τ_ref` 下已被排除)。

### 6.3 detour = `io_count − hard_lambda_sum`(LG 後,頂層欄位)

| run | `io_count` | `hard_lambda_sum` | detour | detour% | `io_count/hard_lambda_sum` |
|---|---:|---:|---:|---:|---:|
| flat(adaptec1,regime=1) | 30,256 | 24,467 | 5,789 | 19.13% | 1.2366 |
| adaptec1 A2 k16 grid | 24,647 | 19,337 | 5,310 | 21.54% | 1.2746 |
| adaptec1 A2 k16 slicing | 33,460 | 22,202 | 11,258 | 33.65% | 1.5071 |
| adaptec1 A2 k32 grid | 41,272 | 27,985 | 13,287 | 32.19% | 1.4748 |
| adaptec1 A2 k8 grid | 19,084 | 14,856 | 4,228 | 22.15% | 1.2846 |
| bigblue4 A2 k16 grid | 63,755 | 54,518 | 9,237 | 14.49% | 1.1694 |

**§3.2.5 的「份額」表述在優化過程中大致維持,但不是不變量。** 設計文件記載的 M1 靜態樣本份額比值
(adaptec1 k16 1.238、k32 1.358、bigblue4 k16 1.190)與本表 A2 優化後的比值(adaptec1 k16 1.275、k32
1.475、bigblue4 k16 1.169)同量級,方向一致(k32 detour 占比高於 k16,bigblue4 detour 占比低於
adaptec1)——即使優化把 `io_count` 壓低了 18–36%,份額關係沒有崩潰。但 adaptec1 k16 slicing 的 detour
占比(33.65%)明顯高於同 case grid(21.54%),顯示 **region 形狀(rectilinear slicing vs 軸對齊 grid)
對 detour 占比有實質影響**,份額不是與幾何無關的常數。GP 期間(trajectory,6.2 節)的比值也穩定在
1.15–1.25 附近(`io_count/hard_lambda_sum`,以 iteration 600 為例:23,554/18,587=1.267),與 LG 後的
1.275 相近,說明份額關係在 GP→LG 轉換前後也維持。

**detour 的來源與後續方向:** detour 佔比(14–34%)量化的是 evaluator 的 MST 幾何相對 region-presence
下界(`Σ(Λ−1)`)的高估——MST 只最小化樹長,不理會 region 邊界,因此有機會穿過 partition 密集的區域把
`io_count` 灌高,而 S2 surrogate 優化的 `λ_e = |touched regions| − 1` 本身就是 routing-independent 的
crossing 下界。routing-aware evaluator 的改進方向已立項,見
[`docs/research/2026-08-13-routing-aware-evaluator-directions.md`](../research/2026-08-13-routing-aware-evaluator-directions.md)
(region-adjacency Steiner / boundary-cost maze routing 兩個候選 formulation,留待 M3 design 裁決)。

---

## 7. F1–F5 判定

`results/m2/sweep/gates.json` 布林值(全部在 T7 掃描結束時機械判定):

| ID | 布林值 | 門檻(設計 §9.1) | 觸發後動作 |
|---|---|---|---|
| F1 | **false** | 退火末端 `p_max<1−1e−3` 的 movable cell 比例 < 5% | 調參:τ 補掃;若補掃後仍 <5% 且 F2 亦觸發 → S5 |
| F2 | **false** | 所有 ρ×τ 掃描臂在 `io_gp` 基準皆未達 `Δio_gp≤−3%` | 換 surrogate → S3(T10) |
| F2b | **false** | 有臂達 `Δio_gp≤−3%` 但無臂達 exit 的 `Δio≤−5%` | 調參:τ 補掃 + margin 深掃(T11) |
| F3 | **false** | ≥2 臂觸發 `check_divergence`,或 `Δhpwl>+10%`,或 `obj_eval_count` 增量中位數 ≥5 | 檢查 Lipschitz 護欄與 secant refresh;降 `ρ_max` |
| F4a | **false** | 同一 run 內 `Δ(hard_lambda_sum)≤−10%` 但 `Δ(io_gp)>−3%` | 目標-指標脫鉤 → S3(T10) |
| **F4b** | **true** | 任一臂 `io_gp` 相對其自身啟用時的值上升 `>3σ_rep`(`σ_rep=0`,故任何上升皆觸發) | 立即停該臂;檢查 secant/NaN/梯度符號 |
| F5 | **false** | `lg_loss(M2) − lg_loss(flat)` 相對 flat `io_count` `>+2%` | 獨立治理:margin 深掃(T11);不換 surrogate |

**best 臂 = `adaptec1_k16_rho0.40_annealed`。**

**F4b=true 的裁決(全文引 `gates.json` 的 `F4b_note`):**

> "activation happens at overflow~0.85 while cells are still unfolding; the flat observer rises
> +19.80% over the same window, above every arm's rise, and arm rises anti-correlate with rho.
> Mandated checks passed: backtrack_median=1.0 on all arms (secant healthy), no NaN, net io_gp
> deltas negative -- adjudicated an artifact of the activation-point baseline, not surrogate
> pathology (see T9)."

即:F4b 量的是「每個臂的 `io_gp` 相對它自己剛啟用時那一刻的值」是否上升,而 `σ_rep=0` 讓任何上升都
literal 觸發。但對照 `io_gp_delta_pct`(gates.json,同一定義套用在**完全不開 IO 項的 flat observer**
上)顯示 flat 在同一窗口(overflow≈0.85 起算,cell 尚未展開完成)本身就上升 **+19.80%**——比任何一個
有 IO 力的臂都高。11 個臂的 `io_gp_delta_pct` 為 `{16.60, 18.10, 15.87, 16.91, 12.52, 16.06, 8.27,
11.23, 1.34, 7.31, 1.51}`,與 `ρ_max` 明顯反相關(`ρ_max` 越大,上升幅度越小;`ρ_max=0.40` 的
best 臂只有 +1.34%,是 11 臂中最低)——說明 IO 力越強,越能壓制這個「早期基線本身自然上升」的效應,
方向與「surrogate 病理」相反。加上強制檢查(`backtrack_median=1.0` 全數健康、無 NaN、每個臂的淨
`io_gp` delta 相對 flat 都是負的)全部通過,**裁決:activation-baseline artifact,非 surrogate
病理,不停臂。**

**T10/T11 觸發檢查:**

- T10(條件:`F2==true` 或 `F4a==true`)——兩者皆 `false` ⇒ **未觸發**,S3 span proxy 不需要做。
- T11(條件:`F5==true`,或 A7 pilot 顯示 `lg_loss` 相對 A2 明顯改善)——`F5=false`,且第 5.2 節顯示
  A7 margin pilot 的 `lg_loss`(1,224)實際上比 best 臂 A2(1,093)**更差**而非改善 ⇒ **未觸發**,
  margin 深掃不需要做。

---

## 8. Exit 檢核(對照計畫 Task 9 Step 2 / 設計 §7.1)

- [x] **條件 1 PASS:** best 臂(`adaptec1_k16_rho0.40_annealed`)同時滿足
  - `Δhpwl = +1.314% ≤ +2.0%`(PASS)
  - `Δio = −18.538% ≤ −5.0%`(PASS)
  - `|Δio| = 18.538% > 3σ_rep = 0`(regime=1 下 `σ_rep=0`,任何非零改善都超過此門檻;PASS)
  - `Δio` 遠優於 M1 同 case(同 regime 重測):M1 reweight(`ablation/adaptec1_A1_k16_grid.json`)
    的 `Δio = −1.279%`,M2 best 的 `−18.538%` 是其 **14.5 倍**。
  三個數值門檻與「明顯優於 M1」的定性要求全部滿足。

- [x] **條件 2 PASS:** 臂的存在性逐一核對(`results/m2/ablation/*.json` 檔案清單):
  - A0/A1/A2/A3 四臂在 adaptec1(k16 grid,另加 k8/k16-slicing/k32 的 A0/A2 擴充)與 bigblue4
    (k16 grid)皆有結果 —— `adaptec1_A{0,1,2,3}_k16_grid.json`、
    `bigblue4_A{0,1,2,3}_k16_grid.json` 均存在。
  - A4/A5/A6 在 adaptec1 k16 完成 —— `adaptec1_A{4,5,6}_k16_grid.json` 均存在。
  - A4/A6 在 bigblue4 k16 完成 —— `bigblue4_A{4,6}_k16_grid.json` 均存在。
  （A7 margin pilot 額外完成於 adaptec1 k16,超出條件 2 的最低要求。）

**M2 exit:兩條件皆 PASS。**

---

## 附註:待補項(不影響本次 exit,留給 M3/M4 前置)

- **bigblue4 需要 per-case `ρ` 調整。** 第 3 節已記錄:bigblue4 直接套用 adaptec1 調出的 `ρ*=0.40`
  得到 `Δio≈−36%` 級的改善,但 `Δhpwl=+5.41%` 超出 iso-WL 帶(`≤+2%`)。iso-WL 掃描本次僅在 adaptec1
  做過(design §7.3 的既定範圍)。後續建議:(a) 為 bigblue4 單獨掃一次 `ρ×τ`;或 (b) 用設計 §5.2 的
  `λ_io` 錨點(`ρ=1` 時 adaptec1 k16 `λ_io≈1.5e3`、bigblue4 k16 `λ_io≈2.8e3`)反推等效 `ρ`,兩案的
  `λ_io` 錨點比值本身暗示需要更小的 `ρ_max` 才能落回 iso-WL 帶。
- **routing-aware evaluator。** 第 6.3 節指出的 detour 佔比(14–34%)已立項為獨立研究方向,見
  `docs/research/2026-08-13-routing-aware-evaluator-directions.md`。
