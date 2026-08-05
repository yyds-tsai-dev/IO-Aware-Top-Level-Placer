# Baseline 對照表


## adaptec1

| case | mode | k | rtype | seed | io_count | ft_count | tree_wl | hpwl | runtime_s | peak_mem_mb | num_reweights |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptec1 | flat | 16 | grid | 0 | 30,011 | 2,582 | 84,655,386.0 | 73,928,933.0 | 58.3 | 140.8 |  |
| adaptec1 | flat | 16 | slicing | 0 | 36,600 | 7,308 | 84,637,857.0 | 73,890,377.0 | 58.8 | 140.8 |  |
| adaptec1 | flat | 8 | grid | 0 | 22,405 | 1,981 | 84,680,077.0 | 73,955,613.0 | 60.7 | 140.8 |  |
| adaptec1 | reweight | 16 | grid | 0 | 29,992 | 2,326 | 85,034,661.0 | 74,892,624.0 | 60.8 | 606.6 | 6 |
| adaptec1 | reweight | 32 | grid | 0 | 49,123 | 7,514 | 86,065,697.0 | 75,822,539.0 | 58.9 | 870.3 | 6 |
| adaptec1 | reweight | 8 | grid | 0 | 22,405 | 1,576 | 84,849,402.0 | 74,631,933.0 | 60.3 | 474.8 | 6 |
| adaptec1 | two_stage | 16 | grid | 0 | 36,783 | 16,494 | 243,933,433.2 | 226,599,966.2 | 106.5 | 501.2 |  |
| adaptec1 | two_stage | 16 | slicing | 0 | 50,089 | 26,201 | 260,501,940.3 | 240,057,731.6 | 111.1 | 503.4 |  |
| adaptec1 | two_stage | 8 | grid | 0 | 28,266 | 12,282 | 240,420,189.9 | 224,371,075.9 | 89.5 | 369.8 |  |

## bigblue4

| case | mode | k | rtype | seed | io_count | ft_count | tree_wl | hpwl | runtime_s | peak_mem_mb | num_reweights |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bigblue4 | flat | 16 | grid | 0 | 100,108 | 7,045 | 857,859,395.0 | 748,503,270.0 | 545.5 | 1,670.9 |  |
| bigblue4 | reweight | 16 | grid | 0 | 100,736 | 3,585 | 864,957,551.0 | 756,168,955.0 | 535.3 | 6,280.5 | 8 |
| bigblue4 | two_stage | 16 | grid | 0 | 178,594 | 77,899 | 4,693,332,230.1 | 4,459,824,739.1 | 3,738.2 | 6,686.1 |  |

---

## GPU evaluator runtime

### 合成 2M case(摘自 Task 11 Step 4,含 review-fix 後的 chunking,現行 production 程式碼 commit `07542fe`)

2,000,000 cells / 2,000,000 nets / 10,244,871 pins,K=16(4×4),lattice=512,die 100,000×100,000。

| | reference(`evaluate`,CPU numpy) | GPU(`GpuEvalContext.evaluate`,L4) |
|---|---:|---:|
| construction | n/a | 0.633 s |
| cold `evaluate()` | n/a | 3.508 s |
| **median warm `evaluate()`(5 之中位數)** | **599.383 s** | **3.436 s** |
| speedup | — | **~174×** |
| peak GPU memory | — | 4.618 GB(review-fix 前為 15.00 GB,`_seg_chunk_budget` 分塊後 −69%) |

### 本次真實 case 實測(adaptec1 / bigblue4,k=16 grid,reweight 跑出的最終擺放位置)

沿用 Task 11 §3 的方法論(ctor 一次,`evaluate()` 跑 1 次 cold + 10 次 warm,取 warm 中位數),量測腳本已整理
提交至 `ioplace/diagnostics/measure_real_case_eval_time.py`,讀取
`results/m1/{adaptec1,bigblue4}_reweight_k16_grid.json.npz` 的最終 `node_x/node_y`:

| | adaptec1(k=16) | bigblue4(k=16) |
|---|---:|---:|
| n_nets / n_pins / n_nodes | 221,142 / 919,701 / 211,447 | 2,229,886 / 8,731,365 / 2,177,353 |
| construction | 0.184 s | 0.635 s |
| cold `evaluate()` | 0.282 s | 2.720 s |
| **median warm `evaluate()`** | **0.159 s** | **1.318 s** |
| peak GPU memory(獨立量測,僅 evaluator) | 491.5 MB | 4,830.5 MB |

Task 11 brief 原定的 adaptec1 `<0.5s` / bigblue4 `<5s` 目標(該報告因當時無真實 placement 結果只能用合成
case 驗證),本次用真實 case 驗證:**兩者皆 PASS**(0.159s / 1.318s)。

**證據留存與重跑校驗(review Finding 1 跟進,2026-08-05):** 上表數字原始的 console 輸出當時未落盤(僅腳本
留在 scratchpad、未 commit),為補齊證據鏈,已將腳本整理提交(路徑見上),並用相同方法論、相同 npz 座標重跑
一次,原始輸出存於 `results/m1/diagnostics/{adaptec1,bigblue4}_eval_time.json`。重跑校驗值:adaptec1
construction 0.184s(與原記載一致)、cold `evaluate()` 0.388s(原 0.282s)、**median warm 0.158s**(原
0.159s)、peak GPU memory 491.5MB(與原記載完全一致)、io_count/ft_count 30,004/2,326(與下段「GPU/CPU
差異回顧與修復」記載的、C1 修復前的分歧數字完全重現——該分歧已在本次修復後消除,見下段);bigblue4
construction 0.551s(原 0.635s)、cold `evaluate()` 2.343s(原 2.720s)、
**median warm 1.260s**(原 1.318s)、peak GPU memory 4,830.5MB(與原記載完全一致)、io_count/ft_count
100,736/3,585(與原記載完全一致)。結論不變:兩案例的 median warm 皆仍遠優於 brief 的 `<0.5s`/`<5s` 門檻;
peak memory 與 io/ft_count 兩類非計時型指標精確重現。三類計時數字中,**median warm**(唯一被拿來對照
`<0.5s`/`<5s` 門檻、也是下方「evaluator 開銷佔比」表格輸入的數字)差距最小、最穩健(adaptec1 −0.4%、
bigblue4 −4.4%);`construction`/`cold evaluate()` 這兩個單次量測本身噪聲較大,bigblue4 兩者分別差
−13.2%/−13.9%,adaptec1 的 `cold evaluate()` 差距最大(+37.7%,0.282s→0.388s)——adaptec1 是本報告中
最小的 case,cold-start 的一次性 CUDA kernel 暖機/編譯開銷相對其本身 ~0.3s 的量級佔比最大,故相對噪聲也
最大,但兩次量測的絕對值都遠低於下方表格與門檻會用到的任何數字量級,不影響本節或「evaluator 開銷佔比」
表格的判讀。

**GPU/CPU 差異回顧與修復(Critical C1,whole-branch review 追蹤,修復於 2026-08-05):** 本節先前版本
記錄 adaptec1 用本量測腳本(GPU evaluator)算出的 `io_count=30,004` 與該次 run 主表中(CPU
`evaluator_ref`)記錄的 `29,992` 相差 12(168 個 net 分歧),並把 root cause 歸因於 `torch.argmin`
與 numpy 在 exact-tie 情況下 first-occurrence 判定路徑不同、真實 placement 座標比合成隨機座標更容易
撞上 Manhattan 距離 tie。**該解釋是錯的**——whole-branch review 用刻意構造的 tie-saturated 合成掃描
(大量座標製造等長 Manhattan 距離的簡併 MST 選邊)逐一比對 GPU/CPU 的 first-occurrence tie-break
結果,並未發現任何分歧,證偽了這個假設。

真正 root cause 是 `ioplace/evaluator_gpu.py` 的 `GpuEvalContext._to_idx`(修復前)用 python float
`self.cell_w`/`self.cell_h` 當除數:CUDA 把 `tensor / python_float` 編譯成 reciprocal-multiply
(先算 `1/cell_w` 再相乘),不是正確捨入的除法。例如 adaptec1 的 `cell_w = 10692/512 = 20.8828125`,
`2673.0 / 20.8828125` 的精確值是 128.0,但 reciprocal-multiply 算出 `127.99999999999999`,
`.to(torch.int64)` 向下截斷得 127;numpy(reference)的除法正確捨入直接得 128。座標恰好落在 lattice
邊界時,GPU 因此把該點分到相鄰的錯誤 region,少算一次 crossing——這正是造成 adaptec1 168 個 net
分歧的機制,與 MST tie-breaking 無關。

**修復**:`GpuEvalContext.__init__` 新增 `self._cell_w_t`/`self._cell_h_t`(0-dim float64 tensor,
由 `self.cell_w`/`cell_h` 建構),`_to_idx` 改除以這兩個 tensor(`tensor / tensor`,即使是 0-dim,
也是正確捨入的)取代原本除以 python float。新增回歸測試
`tests/test_evaluator_gpu.py::test_gpu_matches_reference_on_lattice_boundaries`(non-dyadic
lattice + 座標恰壓在 region 邊界上),**修復前 FAIL**(`per_net_crossings` ref `[1,1,3]` vs gpu
`[0,0,3]`、`boundary_pair_demand` 分歧),**修復後 PASS**——RED→GREEN 兩態都已用 pytest 實跑確認。

**adaptec1 實測(k=16 grid,本次 reweight run 的最終 placement 座標,即
`results/m1/adaptec1_reweight_k16_grid.json.npz`):** 修復後 `evaluate_gpu` 與 `evaluate`(CPU
reference)**精確一致**——`io_count` 29,992 == 29,992(先前的 168-net 分歧完全消失)、`ft_count`
2,326 == 2,326、`per_net_crossings`/`per_net_ft` 逐 net 全等(0 個 net 分歧)、`boundary_pair_demand`
相等、`tree_wl` 相對差 0.0(85,034,661.0 == 85,034,661.0)、`large_net_lb` 相等。主表中所有
`io_count`/`ft_count` 數字仍一律取自 `evaluator_ref`(CPU),與 M0 報告既有慣例一致(此慣例不受本次
修復影響,純屬既有的量測協定)。

**MST argmin tie-breaking(縮記):** Task 11 docstring 記錄的「`torch.argmin` first-occurrence
tie-break 與 numpy 一致」的經驗觀察未被推翻,但仍只是特定 torch/CUDA 版本上的經驗結果而非規格保證
——torch 升級時應重驗 tie 行為。

### reweight 閉環的 evaluator 開銷佔比(num_reweights × 單次時間 vs 總 runtime)

| case(最佳設定) | num_reweights | median warm evaluate() | 簡單估計(num_reweights × warm) | 精算估計(ctor + 1×cold + (n−1)×warm) | 總 runtime_s | 開銷佔比(簡單/精算) |
|---|---:|---:|---:|---:|---:|---:|
| adaptec1 k=16(α=0.2,every=100) | 6 | 0.159 s | 0.954 s | 1.260 s | 60.77 s | 1.57% / 2.07% |
| bigblue4 k=16(α=0.2,every=100) | 8 | 1.318 s | 10.541 s | 12.579 s | 535.29 s | 1.97% / 2.35% |

（`簡單估計`/`精算估計` 兩欄用未四捨五入的完整精度 median warm/cold/ctor 值算出,並非直接拿上面顯示的
3 位小數相乘——例如 1.318 s × 8 = 10.544 s,與表中 10.541 s 相差 0.003 s,屬顯示精度的四捨五入,非計算
錯誤。）

兩個 real case 的 evaluator 開銷都 **< 2.5%** 的總 runtime——reweight 閉環的時間成本主要來自 DREAMPlace
自身的 GP/LG 迭代,不是我們的 evaluator;§5.3 節「evaluator 效能」层面判定 PASS。GPU 記憶體方面,
reweight 模式的 `peak_mem_mb`(來自實際 run 的 JSON,含 placer + `GpuEvalContext` 全程共存的合計峰值)
比對應 flat 高出 adaptec1 +331%(140.8→606.6 MB,精確值 330.92%)、bigblue4 +276%(1,670.9→6,280.5 MB)
——量級符合預期
(`GpuEvalContext` 的靜態拓撲快取 + 全程與 placer 共駐),bigblue4 的絕對峰值(6.28 GB)距 L4 的 23 GB
仍有充足餘裕。

---

## M1 觀察

### 敏感度掃描(adaptec1 k=16 grid,seed=0;flat baseline io_count = 30,011)

| α | every | num_reweights | io_count | Δio vs flat | ft_count | hpwl | runtime_s | net_weights_max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 50 | 14 | 30,101 | +0.30% | 1,909 | 74,775,887 | 68.4 | 3.0 |
| **0.2** | **100** | **6** | **29,992** | **−0.06%** | 2,326 | 74,892,624 | 60.8 | 3.0 |
| 0.5 | 50 | 15 | 32,455 | +8.14% | 2,449 | 80,026,906 | 60.9 | 6.0 |
| 0.5 | 100 | 7 | 30,980 | +3.23% | 2,305 | 76,912,995 | 60.7 | 6.0 |
| 1.0 | 50 | 15 | 35,760 | +19.16% | 3,136 | 89,165,527 | 62.5 | 11.0 |
| 1.0 | 100 | 7 | 34,784 | +15.90% | 2,806 | 86,118,145 | 60.0 | 11.0 |

六組掃描在 α 與 every 兩個維度上**都單調**:α 越大 io_count 越差(0.2→0.5→1.0 在兩個 every 下皆遞增),
every 越小(reweight 越頻繁)io_count 也越差(50 在三個 α 下皆比對應的 100 差)。**只有最溫和的設定
(α=0.2, every=100)略優於 flat,其餘五組全部更差**,最差(α=1.0, every=50)比 flat 高 19.16%。選定
**(α\*, every\*) = (0.2, 100)** 作為主矩陣設定——但如診斷段所述,這個「勝出」的 −0.06% 本身與單次重跑觀察
到的 run-to-run 差異(~1%,n=1)同量級,不宜解讀為確認的改善;多 seed 分佈量測留待 M5。

### 三方對照(reweight[最佳設定] vs flat vs two_stage)

| case | 指標 | flat | two_stage | reweight | Δ(reweight vs flat) | Δ(reweight vs two_stage) |
|---|---|---:|---:|---:|---:|---:|
| adaptec1 k=8 grid | io_count | 22,405 | 28,266 | 22,405 | **+0.00%** | −20.74% |
| | ft_count | 1,981 | 12,282 | 1,576 | **−20.44%** | −87.17% |
| | hpwl | 73,955,613 | 224,371,075.9 | 74,631,933 | +0.91% | −66.74% |
| adaptec1 k=16 grid | io_count | 30,011 | 36,783 | 29,992 | **−0.06%** | −18.46% |
| | ft_count | 2,582 | 16,494 | 2,326 | **−9.91%** | −85.90% |
| | hpwl | 73,928,933 | 226,599,966.2 | 74,892,624 | +1.30% | −66.95% |
| bigblue4 k=16 grid | io_count | 100,108 | 178,594 | 100,736 | **+0.63%** | −43.59% |
| | ft_count | 7,045 | 77,899 | 3,585 | **−49.11%** | −95.40% |
| | hpwl | 748,503,270 | 4,459,824,739.1 | 756,168,955 | +1.02% | −83.04% |

adaptec1 k=32 grid(reweight only,M0 未跑此 k 的 flat/two_stage,無基準可比,如實記錄缺口而非硬湊):
io_count=49,123、ft_count=7,514、hpwl=75,822,539、runtime_s=58.9、num_reweights=6。

**io_count 方向:三個有基準的 case 中,兩個持平(k=8 完全打平、k=16 −0.06%,皆落在下方診斷段量到的
run-to-run 噪聲範圍內),bigblue4 k=16 略差(+0.63%)。没有任何一個 case 顯示「明顯下降」。**(k=8 的
flat 與 reweight 是兩次獨立 run——ft_count(1,981 vs 1,576)、hpwl(73,955,613 vs 74,631,933)、tree_wl
(84,680,077.0 vs 84,849,402.0)皆不同,io_count 精確打平純屬巧合,並非重複資料或誤植。)

**ft_count 方向則相反:三個 case 全部明顯下降(−20.44% / −9.91% / −49.11%),且降幅隨 case 規模擴大而
擴大**——這是本次量到最一致、效果量最大、且不落在噪聲範圍內的正面結果,但它不是 M1 exit 條件 2 要求的
指標(io_count)。與 two_stage 相比,reweight 在兩個指標上都全面勝出(io −18~−44%,ft −85~−95%),
方向與 M0 報告「two_stage 的 io_count 其實高於 flat」的既有發現一致——reweight 建立在 flat 之上(無
fence 約束),自然繼承 flat 遠低於 two_stage 的 io_count 基準。

### io/ft 軌跡診斷(adaptec1 k=16, α=0.2/every=100,對照 α=0 no-op control)

為了直接回應「reweight 有沒有用」而非只看終值,用診斷腳本(不改動已 commit 的
`run_placement_reweight.py`,已整理提交至 `ioplace/diagnostics/run_reweight_trajectory.py`)重跑同一設定,
額外記錄每次 callback 當下的 `ctx.evaluate()` 結果(`io_count`/`ft_count` 是迴圈本來就會算出的值,純記錄不
增加額外開銷),並用 **α=0(net_weights 恆為 1.0,真正的 no-op)** 重跑一次同樣腳本作控制組,隔離「reweight
事件本身」與「GP+LG 本身的自然軌跡」。兩次跑的原始逐-iteration io/ft 序列(即下表出處,非重跑校驗值,是
當時量測的原始資料)已提交於 `results/m1/diagnostics/adaptec1_reweight_k16_trajectory.json`(α=0.2)與
`results/m1/diagnostics/adaptec1_control_alpha0_k16_trajectory.json`(α=0 控制組):

| iteration | io(α=0.2) | io(α=0,control) | ft(α=0.2) | ft(α=0,control) | net_weights_max(α=0.2,更新後) |
|---:|---:|---:|---:|---:|---:|
| 100 | 23,565 | 23,626 | 1,214 | 1,214 | 3.0 |
| 200 | 22,088 | 22,375 | 1,049 | 1,222 | 3.0 |
| 300 | 25,811 | 25,389 | 1,402 | 1,375 | 3.0 |
| 400 | 26,134 | 26,698 | 1,925 | 1,868 | 3.0 |
| 500 | 28,103 | 28,936 | 2,717 | 2,441 | 3.0 |
| 600 | 27,557 | 28,124 | 2,787 | 2,486 | 3.0 |
| **final(LG 後,`evaluator_ref`)** | **30,295** | **30,044** | 2,420 | 2,509 | 3.0 |

(此診斷重跑與主表中「官方」的 (α=0.2, every=100) run 是同設定、同 seed 的獨立第二次執行,終值
29,992 vs 30,295 相差 303/1.0%——即下方第 4 點的噪聲觀察。)

### 診斷(數據支持的假設,非事後諸葛)

1. **假設 A(最有力,直接證據):WL 加權只對「路徑緊緻度」有槓桿,對「離散 region 歸屬」沒有直接槓桿。**
   `update_net_weights` 只對已跨 region 的 net 加 WL 梯度壓力,把該 net 的 pin 拉近成更緊的 Steiner 路徑
   ——這會壓低「路過但沒 pin 的 region」數(ft),但不會直接改變某顆 cell 究竟落在哪個 region(離散決策,
   WL 梯度只能透過 density/legalize 的間接效應影響它)。三個 case 中 ft 全面明顯改善(−9.9%~−49.1%)而
   io 幾乎不動,正是這個假設預測的確切模式——是本次診斷中最直接、最一致的證據,呼應 brief 列出的候選
   假設「reweight 只影響 WL 項無法表達『數量』目標」。
2. **假設 B(直接證據,因果未證實):cap 飽和可能稀釋了對高槓桿 net 的鑑別力。** 六組敏感度掃描中,
   `net_weights_max` 在每個 α 都精確落在 `1+α×cap`(cap=10 預設值)——0.2→3.0、0.5→6.0、1.0→11.0,
   代表**每個 α 設定下都至少有一個 net 的真實 crossing 數 ≥ cap**,即 cap 從最溫和的 α=0.2 開始就已經
   在夾限。crossing=10 的 net 與 crossing=50 的 net 會拿到相同權重,即使後者對 io_count 的貢獻大得多
   ——機制在最需要鑑別力的高槓桿 net 上恰好失去了鑑別力。未做 cap 消融實驗,不宣稱因果,但這是後續
   便宜可測的具體項目(調高/移除 cap;記錄 per-net crossing 分布看有多少 net 被夾限)。
3. **假設 C(直接證據,細化版「晚期擾動」):效果會在 GP→LG 交界處流失,而非被 reweight 事件本身
   破壞。** 軌跡表中,iteration 200/400/500/600 這四個點 α=0.2 都低於(或大致持平)α=0 control——即
   reweight 在 GP 過程本身有溫和的正面效果;但 legalize 之後,α=0.2 反而變得比 control 差(30,295 vs
   30,044,+0.84%)。Legalization 把 cell 對齊到合法 row/site 造成的位移,足以抹掉、甚至逆轉 GP 階段
   累積的 crossing 改善。
4. **控制組直接排除「reweight 事件本身造成晚期劣化」這個更簡單的假設。** α=0(net_weights 全程固定
   1.0,真正 no-op)的控制組軌跡呈現**同樣的**「中段下降、後段回升」形狀(iter 200 的 22,375 → iter 600
   的 28,936),與 α=0.2 幾乎同型——證明這個形狀是 **DREAMPlace 自身 GP+LG 動態的內生現象**(推測是
   density penalty 隨迭代增強、在 legality 與 wirelength/locality 之間權衡轉移所致),**不是** reweight
   造成的。這排除了「reweight 在後期擾動優化軌跡」作為主因,把假設 C 的範圍收窄到 GP→LG 交界本身。
5. **run-to-run 噪聲量級不可忽略,且與量到的「改善」同量級。** 兩次完全相同設定
   (α=0.2, every=100, k=16, seed=0,程式碼唯一差異是診斷腳本多讀了幾個唯讀欄位、不影響計算圖)的
   io_count 相差 303(~1.0%)——比六組敏感度掃描中唯一「勝出」設定相對 flat 的優勢(−0.06%,19 個
   單位)還大一個量級。合理推測來自 DREAMPlace 自身 GPU kernel(density/wirelength 的 scatter/atomic-add
   類操作)在不同 process 之間並非 bit-deterministic——與 Global Constraints 已知的 mtkahypar
   非完全可重現性同一類問題,但這是首次在 DREAMPlace 自身的 GP+LG 路徑上觀察到,未進一步 root-cause
   (超出本 task 範圍)。**實務含意:本里程碑的 single-seed 數字無法區分「真實的 ~1% io_count 效果」
   與「單純的 run-to-run 噪聲」,任何這個量級的改善宣稱都需要多 seed 驗證才能當真。**
6. **k 粒度:本次缺口,如實記錄。** M0 從未跑過 adaptec1 k=32 的 flat/two_stage,本 task 也未補跑(不在
   brief 授權範圍內),所以 k=32 只有 reweight 的絕對數字、無法算出 delta——留白比硬湊一個不可比的基準
   更誠實。
7. **這正是 M2 可微化項的 motivation。** ft_count 在當前「只加 WL 壓力」的方案下有乾淨、一致、量級隨
   case 增大的改善,說明梯度式壓力在這條路徑上**結構上是有效的**——它只是還沒被接到 M1 真正在意的指標
   (io_count / 離散 region crossing)上。下一步需要能直接對「哪個 region」這個離散決策提供梯度訊號的
   東西(例如 region 歸屬的可微鬆弛,用溫度退火的 soft-assignment,讓 cell 質量真正往降低 crossing 的
   region 方向移動),而不只是在已經固定跨 region 的 net 上把既有的兩端拉近。

---

## M1 exit 檢核(對照 spec §9)

- [x] **條件 1:`mst_crossing_eval` 等價驗證與 runtime 報告。** Task 11 的 7/7 GPU/CPU 等價測試(含
  large-net 分支與 k=64 邊界案例)全綠;本報告「GPU evaluator runtime」節補上了 brief 要求、Task 11
  當時缺的兩塊:(a) adaptec1/bigblue4 真實 case 的 evaluate() 單次時間實測(0.159s / 1.318s,雙雙優於
  brief 原定的 <0.5s / <5s 目標);(b) reweight 閉環的 evaluator 開銷佔比實測(1.6%~2.4% 的總
  runtime,非瓶頸)。**PASS。**
- [ ] **條件 2:reweight 的 io_count 相對 flat 明顯下降。** **未達成,如實記錄,不挑好看的 case。**
  三個有 flat 基準的 case 中,最好的結果是 k=16 的 −0.06%(29,992 vs 30,011)與 k=8 的完全打平(0 delta)
  ,bigblue4 k=16 則是 +0.63%(略差);六組敏感度掃描中五組明顯更差(最差 +19.16%)。−0.06% 本身與單次
  重跑觀察到的 run-to-run 差異(~1.0%,n=1)同量級,不能視為確認的改善;多 seed 分佈量測留待 M5。
  **Sanity 判讀:FAIL(方向未達「明顯下降」的門檻)。** 診斷段(上)給出三個數據支持的假設(WL 項對離散 region 歸屬缺乏直接槓桿 / cap 飽和 /
  GP→LG 交界流失改善)並用控制組實驗排除了「reweight 事件本身造成晚期擾動」這個更簡單的競爭假設,同時
  記錄了 ft_count 方向一致轉好(−9.9%~−49.1%)、run-to-run 噪聲不可忽略兩個旁證。此負面結果與診斷是
  M2(可微化 region 歸屬)工作的直接 motivation,見診斷第 7 點。
