# M4 規模化(1M → 30M)報告(草稿,T11 進行中)

- 日期:2026-08-16;分支 `m2-differentiable-io`,報告基準 HEAD `283e1a5`
- 環境:`docs/dev-env.md`(`DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`、`$DP/.venv312/bin/python`、torch 2.8.0+cu128、CUDA 12.8、NVIDIA L4、`dp_commit d971880a15ef684c4a90bccd2c65d62ae7e33297`,近期跑於 hostname `NVL5`)
- 對應設計:`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`(v2.3)
- 對應裁決:`docs/results/2026-08-14-m4-t6-adjudication.md`(T6 協定 v2、G-D 撤回、Bug A/B)、`docs/results/2026-08-15-m4-t6-holdout-adjudication.md`(T6 holdout FAIL 維持、N2 選定、T6B 協定 B-0…B-6、B5、附錄 A 的 B-5 重算)
- 前置報告:`docs/results/m3-differentiable-ft-report.md`(M3 exit FAIL——`ours@M3` 臂本報告一律不納入任何表格,只有 `flat`/`ours@M2`)
- **狀態:草稿。本報告自身尚未走完 T11。** 已齊章節照 M0–M3 慣例寫實;未齊段落一律用 `<!-- PENDING: ... --> `標記其等待的 task/artifact,禁止以推測數字填補。

**品質宣稱紅線(design draft §3.4/§6.2;T6 holdout 裁決 §7-1):** 本報告中任何 `benchmark_kind="synthetic"` 的 case(6.2M/12.3M/27.7M 合成陣列)一律不得出現 HPWL/IO/FT 品質欄位或 winner/pareto 宣稱。凡合成表格涉及生成器未認證(`generator_verified=false`),依裁決 §7-1 必須逐字附上:

> 30M 的連通性是擬合出來的,不是校準出來的。

---

## 1. 摘要與 verdict 總表

M4 的核心命題(v1→v2.3 一路存活):**DREAMPlace GP 在 10M 級不是瓶頸,evaluator 記憶體與 IO op 的 K-pass runtime 才是**;而 v1 的推論方式(五點擬合外推、峰值相加判可行、cell-count ratio 當階層證據、10× 容忍帶當預測)全部被 Codex 兩輪對抗性審查與 T3a/T6 的實測推翻,v2.3 把每一個裁決換成「一個能證偽它的實驗 + 一條在證據不足時停下來的規則」。本報告是這條規則的第一次結案:**一個實質的 FAIL→PASS 轉變已經發生(E1 的 overflow 判準),T6 的生成器驗證維持 FAIL 且已誠實揭露其失敗機制,其餘四個 E 判準都卡在尚未執行的 task 上,不以任何理由代填。**

| # | 判準 | 狀態 | 一句話 |
|---|---|---|---|
| **E1** 10M 全流程 | **部分可判** | seed A 上,`mempool_cluster` 的 overflow 判準已從 FAIL(1000-iter 上限,final_overflow=0.327)轉為 PASS(2000-iter,final_overflow=0.0698);完整 §6.3 矩陣(K∈{16,32}×{flat,ours@M2}、seed B)尚未交付,見 §3、§7 |
| **E2** 30M 測資 | **PENDING** | (a) 27.7M Bookshelf 產出 + V0/H5′ 綠已達成(T6B);(b) 12.3M 的 H1–H5 holdout **未**全綠(T6 holdout FAIL 維持,非懸置而是已判定的 FAIL);(c) 27.7M 元件級 spike(T9)三態終結尚未執行 → 待 T9 |
| **E3** 全規模對照表 | **PENDING** | `quality_real_cases.md`/`scaling_synthetic_cases.md` 需要 seed B 正式 run;現有只有 seed A 校準/篩選資料(§5)→ 待 T8b 之後的 seed-B T8 |
| **E4** 記憶體 profile | **PENDING** | B1/B2/B3 三個量測工具 bug 已處置揭露(§2);但 T0b 的 factorial 擬合(`model_fit.json`)尚未執行,6.2M/12.3M holdout 殘差無從談起 → 待 T0b |
| **E5** H100 forecast | **PENDING(凍結)** | 草稿預測已產出(`results/m4/forecast/h100_prediction.draft.json`,`status="draft"`、`unvalidated=true`),含具體 point estimate;spec 要求的**凍結**版(`h100_prediction.json` + sha256 登錄)尚未執行 → 待 T10 收尾 |

<!-- PENDING: 本表在 seed-B 全部 report run(T8)+ T0b + T9 + T10 凍結完成後需整表重判,屆時 E1 亦轉為完整判定。 -->

**一句話結論(暫定,待上述 task 補齊後可能改判):** M4 v2.3 的誠實紀律本身站得住——iteration 預算調整是一次性的、方向不利於「ours」的全語料修正(§3),T6 的 FAIL 不因任何事後論證被翻案(§4),T6B 的自洽性驗收全綠(§4);但「30M 全流程數小時內完成」的原始 exit 主張本報告尚不能簽署,因為它依賴的 T0b/T9/T10 三個量測 task 都還沒跑。

---

## 2. 量測方法與 bug 揭露

沿用 spec §1.4 的四條(v2 修正其中兩條的論證)加上 T3a/T6 過程中新發現的第五條。**全部適用於本報告引用的每一個數字。**

### B1 — `peak_mem_mb` 同 process 污染(M2/M3 數字全部作廢)

`results/m2/**/*.json` 的 `peak_mem_mb` 受同 process 內先前臂污染:adaptec1 `A0_k8`=140.765625 MiB 與全新 process 的 GP 峰值逐位元相同,其後 k16/k32/slicing = 253.4/366.7/479.3 MiB(每步 +112.6 MiB 定值步進);bigblue4 A2/A3/A4/A6 = 10.58/12.46/14.35/16.23 GB(每步 +1.88 GB 定值步進)。`max_memory_allocated()` 是「歷史最大 active allocation」不會自動相加,v1 的「process 累積 high-water mark」是過度推論;更可能是每臂結束後仍存活的張量(retained tensors / 缺 teardown),**單靠 reset 不會清掉仍存活的 allocation**。

**處置:結論(污染成立、M2/M3 的 `peak_mem_mb` 一律作廢)保留;成因改標「未證」。** 正式的成因裁決(T1b 四臂 A/B probe)本報告範圍內尚未執行。<!-- PENDING: T1b 四臂 subprocess A/B probe(fresh subprocess / same-process / reset-only / explicit teardown+GC)——本報告目前只複述 spec §1.4 B1 的結論段,未新增實測。 -->本報告的 GPU 峰值數字(§5、§6、§7)一律取自**每次 run 各自獨立 process** 產出的 `device_used_gb`/`peak_mem_mb`(schema v3,`peak_mem_mb_reset_semantics=true`),不引用任何 M2/M3 的舊 `peak_mem_mb`。

### B2 — diagnostics 每 callback 成本是 `1 + n_nonempty_buckets`,不是固定 8×

`io_term.diagnostics()` 對每個**非空** degree bucket 各跑一次完整 chunked fwd+bwd(`DEG_BUCKET_LABELS` 共 7 桶),再加 `io_grad_l1` 一趟——正確式是 `1 + n_nonempty_buckets`,adaptec1/bigblue4 七桶皆非空才「剛好」等於 8。**v1 提議的 `io_grad_l1` 復用同一次 backward 已撤回**:bucket-masked 梯度的 L1 範數在數學上不能還原完整梯度的 L1(跨 bucket 座標梯度會相消,只有 `Σ_b‖g_b‖₁ ≥ ‖Σ_b g_b‖₁`)。T1 已落地 `--diag-every N`/`--no-diag`;本報告的 T8b/T8 run 一律 `diag_every=1`(schema v3 欄位可查,如 `results/m4/t8b/cluster__k16__grid__flat.json` 的 `"diag_every": 1`)——驗收改**實測 callback wall-time**,不數 backward 次數。

### B3 — spike 門檻參數化 + 硬體契約 assert

`spike_10m.py:116` 原 `ok = peak_gb <= 8.0` 是 M2 對 10M 定的門檻,30M 下必然 false。已參數化為 `--budget-gb`,同時記錄 `measured_peak_gb`/`budget_gb`/`budget_source`/`baseline_reserved_gb`;`budget_gb` 上限由 run manifest 的硬體契約 assert:**L4 = 19.5 GB(= 0.9×21.7 GiB)、H100 = 72 GB、M2 G6 的 10M 契約維持 8 GB**,CLI 不得超過。§6 引用的 27.7M 元件級 spike(T9)即依此契約判定;本報告尚未有該 spike 的實測(見 E2/§7)。

### B4 — T3a hierarchy gate 兩個 bug(2026-08-14 裁決)

`results/m4/corpus/hierarchy_gate.json`(commit `309e5d7`)的 G-D 結論被撤回,原因是兩個獨立的量測 bug:

- **Bug A(bbox 質心)**:原判準用 bbox 與 bbox 中心——那是 2.8M 點的極值統計,由每群幾顆散落 cell 決定,對「真實 placement 不按階層分區」給出與事實相反的答案。改用**質量統計**後,真實 `mempool_cluster` 的四個 group **確實構成一個(模糊的)2×2**:最佳象限指派可解釋 **68.0%**(固定中線 66.0%)的 cell 質量,1×4 只有 48.8%、4×1 只有 46.5%;真實質心兩兩距離 **0.12–0.55 die extent**(不是 gate 原記的 0.00032);指派 g1=UL、g3=UR、g2=LL、g0=RD,左右純度 0.869、上下純度 0.765。新判準 `PASS_2x2 = score_2x2 ≥ 0.50 且領先 ≥ 0.10`,實測 **0.6797 vs 0.4879,margin 0.192** → **PASS_2x2**(commit `ea91dc2`)。
- **Bug B(DEF `+` 截斷)**:DEF `NETS` 的 pin tuple 掃描沒有在 `+` 子句處停止,把 `+ ROUTED ... (x y)` 繞線座標與 `(* pin)` 萬用 tuple 一併當成 pin,導致 `cluster_stats` 要用的跨 group 量**高估 7.2 倍**。修正後 pin 數逐一對上 DREAMPlace 的 `PlaceDB.read` 統計(cluster 43,944,352+3,441=43,947,793;group 12,014,771+11,420=12,026,191);頂層殘量 pins 由 6,668,755 修正為 **24,276**;每 group 跨界 net 由 78,214–91,128 修正為 **11,653–12,585**;group 端子 Rent p 由 0.759–0.769 修正為 **0.6305–0.6352**。

修正後的真實跨 group 統計(T6/T7 的 glue 生成參數即由此導出):多 group net 22,399 條(0.176%);六 pair 計數 **4,002/4,227/4,034/4,566/3,794/4,019**;觸及 2/3/4 群 = 21,552/517/329(85% 為 degree-2);跨界總 pin 1,313,534,其中 986,903 集中在 2 條全域 net(扣掉後平均 degree 14.6)。⇒ T6 的陣列形狀回到 v2.1 主線 **2×2**,1×4 分支作廢。

### B5 — H1 絕對區間換尺缺陷(2026-08-15 揭露,逐字保留)

> **B5(量測工具缺陷,2026-08-15 揭露)**:H1 的絕對區間 `[0.55, 0.80]`(design draft §3.3)是依「group 端子計數 Rent p ≈ 0.633」訂的,而 H1 實際使用的是 **mtkahypar 遞迴二分(Landman–Russo)**估計器。兩把尺在本設計族上系統性相差約 0.11:同一份 `mempool_cluster` 用遞迴二分量得 **p = 0.5221 [0.5001, 0.5459]**,`mempool_group` 量得 **0.4747 [0.4481, 0.4928]**。**真實參照自己就落在預註冊區間之外**,因此 H1 的絕對腿在本次評測中對「生成器是否忠實」不具任何鑑別力——一個與真實 cluster 完全一致(`|Δp| = 0`)的合成品同樣會被判 fail。依預註冊紀律,門檻**不得事後修改**,判定維持 **FAIL**;上一輪裁決文件中「真實端預期 p ≈ 0.633」的預測據此記為**已被否證**。H1 的相對腿(`|p_syn − p_real| ≤ 0.05`)**通過**:N1 = 0.0102、N2 = 0.0041。此估計器混用未外溢到 glue 參數:`λ₀_tile` 的尺度換算若改用遞迴二分的 p,由 4,387.3 變為 4,328.2(差 1.36%)。

**分辨規則(附錄 A.1,通用):** 一個事後修改屬於**可執行的判準修正**,當且僅當三條全部成立——(1) 被比較的量與預註冊文字指名的量不同(算錯了東西,不是接受域不合意);(2) 修正依據是量測之前就已寫定的文字;(3) 修正後的判準不含任何新數值。H1 的區間第 (1) 條不成立(H1 兩邊都算對了,錯的是接受域)⇒ **FAIL 永久維持**;B-5(§4)三條全部成立 ⇒ 可重算。

---

## 3. iteration 預算決策

### 3.1 診斷紀事

原 GP config(承 M2/M3)把 `global_place_stages[0].iteration` 定死在 **1000**。在 `mempool_cluster`(11.31M cells)上,這個上限先於自然收斂被打斷:

| run(1000-iter 上限,已作廢) | K | `final_overflow` | `experiment_status` |
|---|---:|---:|---|
| `results/m4/profile/superseded_iter1000/mempool_cluster__k16__grid__flat.json`(run_id `9dec7b71-af7c-4d62-b3cd-cc76aba244df`) | 16 | **0.3270** | `superseded_iteration_capped` |
| `results/m4/profile/superseded_iter1000/mempool_cluster__k32__grid__flat.json`(run_id `ce76c6ad-055e-4e13-8ef4-3e9f75a56385`) | 32 | **0.3270** | `superseded_iteration_capped` |

`0.327 ≫ 0.07`(`stop_overflow`)——這是 E1 overflow 判準的一個乾淨 FAIL,不是量測噪聲。診斷探針(`results/m4/probes/probe_gp_memory__mempool_cluster_probe2500.json`,把上限暫時放到 2500 只為觀察軌跡)顯示:同一組真實 trajectory 若不被 1000 打斷,會在 **iteration 1251** 自然早停,`final_overflow=0.06976862251758575`,`ok=true`,無發散跡象。

### 3.2 規則與其誠實屬性

commit `a10c02d`(2026-08-15)定案:

```
budget = max(ceil(1251 * 1.25 / 100) * 100, 2000) = 2000
```

三個屬性使這是一次乾淨的規則變更,不是為了讓某個結果變綠而挑的數字:

1. **前 1000 步逐位元相同**——這是純預算延伸,不是重新調參;早停條件(`stop_overflow`)本身完全未動。
2. **對已經在 1000 步內收斂的 run 是零代價**——只有真的撞到舊上限的 run 才會受影響。
3. **一次調整、套用全語料**——`a10c02d` 同時改了 `superblue12`/`bigblue4`/`mempool_{cluster,group,tile_wrap}`/`synthetic_{1x2,2x2}_n2` 全部 8 個 config 的 `iteration: 1000→2000`,不分 `flat` 或 `ours@M2` 臂、不分哪個 case 對哪個結論有利。**這個規則對「ours」不必然有利**:把預算延伸到自然收斂點,受益的是每一個 arm(包括 `flat` 參照本身),`flat` 的品質參照因此也變得更完整、更難被「ours」相對超越——這與挑一個只讓 `ours` 看起來更好的預算方向相反。

受此規則影響的舊 artifact 已被明確標記淘汰而非靜默覆蓋(commit `b458bdc`,`experiment_status="superseded_iteration_capped"`,涵蓋 `mempool_cluster__{k16,k32}__grid__flat` 與 `group__k16__grid__{flat,rho0.05,rho0.10,rho0.15,rho0.20}` 共 7 個檔案,全部搬移至各自目錄下的 `superseded_iter1000/` 子目錄留存)。

### 3.3 重跑結果:E1 overflow 判準的 FAIL → PASS

在 2000-iter 預算下重跑(seed A,K=16):

| run | `gp_iterations_run` | `final_overflow` | 對 E1(`≤0.07`) |
|---|---:|---:|---|
| flat(run_id `b60324cc-c06c-4cd5-9253-db590454c800`) | 1252 | **0.06977** | PASS |
| `ρ_max=0.05`(run_id `09bc5d56-89ea-4c80-9727-b940789b3db6`) | 1281 | **0.06943** | PASS |

兩者皆早停於 `stop_overflow_reached=true`,`num_unplaced_cells=0`,`legalization_status="success"`,`device_used_gb`(12.85/13.47 GB)遠低於 L4 契約 19.5 GB,wall-time(11,974.7 s / 6,690.8 s ≈ 3.33 h / 1.86 h)在 E1 的 ≤4 h 之內、也未觸發 M4-G4 的 3 h/T13 門檻。這就是摘要與 §7 引用的「E1 部分可判」故事本身。

### 3.4 `target_density = area_util + 0.01` 的副作用揭露

T3(spec M4-L3)把三個 NanGate45 case 的 `target_density` 從 ISPD2005 沿用的固定值改為**逐 case 反推**:`suggested_target_density = total_movable_node_area / free_area + 0.01`,直接寫入各 benchmark config(取代原本會製造大量 filler 的固定值):

| case | `area_util`(`results/m4/corpus/*.json`) | 實際 config `target_density` | filler slack `0.01/area_util` |
|---|---:|---:|---:|
| `mempool_group` | 0.704283 | 0.714 | 1.42% |
| `mempool_cluster` | 0.639169 | 0.649 | **1.56%(≈1.6%)** |
| `mempool_tile_wrap` | 0.389114 | 0.399 | 2.57% |

(`superblue12`=0.65、`bigblue4`=1.0 兩個非 NanGate45 case 不套用此規則,分別沿用 ISPD2015 既有值與 ISPD2005 的全填滿慣例。)

**副作用需要揭露:** 這個固定 +0.01 的絕對邊際(而非按比例放寬)使 GP 只有很薄的「鋪散」餘裕就達到目標密度,尤其在 `mempool_cluster` 上(1.6% 邊際)——density-driven spreading force 的可用空間被壓縮,絕對 HPWL/tree_wl 數字因此系統性偏向較擁擠、較悲觀的一端(相對於一個更寬鬆的 `target_density` 會給出的鋪散結果)。但**這個效應對每個 case 的所有 arm(`flat` 與 `ρ_max>0`)一視同仁**——`target_density` 是 case 級常數,不隨 arm 變動——所以 §5 與 §7 引用的 **Δ 指標(Δhpwl%、Δio%)不受此影響**,只有絕對 HPWL/tree_wl 的量級需要在跨 case 比較時留意這條揭露。

---

## 4. T6 / T6B / T7 紀事

### 4.1 T6(2×2)holdout:FAIL 維持,N2 選定

依 B4(§2)修正後,T6 協定 v2 在真實 `mempool_cluster` 導出的六 pair 統計上,以**閉式 method-of-moments** 解距離核 `φ(d;α)=d^(−α)`:`λ₀=4155.25、α=0.10230633302323285`(2×2 排列下 2 觀測/2 參數 ⇒ 恆等識別、0 lack-of-fit DoF——核形狀本身在 2×2 上不可驗證,見 §4.4)。以此參數各產出 N1(每 pair 常數)與 N2(每 tile 端子預算固定,對稱 Sinkhorn 解)兩種正規化的 1×2/2×2 合成陣列,跑 H1–H6:

| gate | N1 | N2 | 門檻 |
|---|---|---|---|
| H1(Rent 絕對區間) | fail | fail | `[0.55,0.80]`(見 B5,鑑別力為 0) |
| H1(Rent 相對 `|Δp|≤0.05`) | 0.0102 pass | **0.0041 pass** | ≤0.05 |
| H2(per-level cut) | ok | **ok(最大 rel err 9.2%)** | ≤25% |
| H3(全網 degree KS) | ok | ok | ≤0.05 |
| H4(interface-cell 自洽) | ok | ok | ≤0.10 |
| H6(Rent 不變性,1×1/1×2/2×2) | **fail(0.0576)** | fail(0.0515) | `|Δp|≤0.03` |
| **verdict** | **FAIL** | **FAIL** | — |

`t6_summary.json`:`n1_verdict="FAIL"`、`n2_verdict="FAIL"`、`t7_gate="FAIL"`——**FAIL 依預註冊紀律維持,一個字不改**(2026-08-15 裁決)。但 H6 的失敗機制本身是本輪最實質的發現,必須逐字進報告(裁決 §7-3):

> **H6(Rent 不變性,{1×1, 1×2, 2×2},`|Δp| ≤ 0.03`)兩種正規化皆判 FAIL,但兩者的失敗機制不同,必須分開陳述。** 含 `1×1` 的兩條腿把兩件事混在一起:(i)換陣列形狀時每 tile 端子是否守恆,(ii)加入 glue 本身是否改變 p。後者在真實世界裡並非 0——真實 `mempool_group` standalone 的 p 是 0.4747,真實 `mempool_cluster`(四個 group 加上層互連)是 0.5221,真實的階層躍升就是 **+0.0474**;合成 N2 的躍升是 +0.0515、N1 是 +0.0576。唯一不含此混淆的一條腿(`1×2 vs 2×2`)給出了明確的鑑別:**N2 = 0.0042(形狀不變)、N1 = 0.0412(隨形狀漂移)**,方向與幅度都與「N1 讓每 tile 端子隨陣列大小成長、違反 Rent」的先驗預測一致。**H6 因此確實完成了它被設計的用途——證偽 N1;30M 採用 N2。** 依預註冊紀律,`1×1` 那兩條腿的 null 設為 0 是 gate 設計缺陷,但**不得事後改判**,verdict 維持 FAIL。

⇒ **`selected_normalization = "n2"`**(理由是 §3.2(d) 的先驗 Rent 論證 + H6 形狀不變腿的實測,不是「H6 通過」);T7 的配方由此定案,recipe B 不啟用(見 §4.3)。

### 4.2 正面證據(強度分級陳述,裁決 §7-4)

> T6 的綠燈必須依證據強度分級陳述。**強(結構性、非建構恆等式)**:H2 per-level cut(N2 逐層相對誤差最大 9.2%;若把 glue 拿掉,window 頂端三層會退到 25% / 16% / 14%,恰好卡在 25% 門檻上——**跨 tile 連通性的「量」是真實資料所要求的,而校準出來的量是對的**;此比較兩邊的 block 大小差 9.24%,以 `T ∝ B^0.522` 修正後 level 4 的誤差進一步降到 4.3%);H1 相對腿(0.0041);H6 的形狀不變腿(0.0042)。**弱(近乎建構恆等式)**:H3 全網 degree KS = 0.013——合成陣列與真實 cluster 共用同一個 `mempool_group` 拓撲,此結果幾乎必然;H4 KS = 0.011——依 2026-08-14 裁決已降級為對「來源 group 自身」的自洽性檢查,量的是抽樣程序與其自身建構假設是否一致。

真實 cluster 最大 net degree 為 1,081,189(兩條全域 net 合計 986,903 pins);合成陣列為 342,429(來源 group 的最大 net,四份各自獨立、未合併);Rent 與分割兩側一律套用 `max_net_degree=100` 上限。

### 4.3 T7:配方定案(recipe A′,不切 recipe B)

四個理由(裁決 §5.1):(1) 切到 recipe B 無法把任何一個紅燈變綠——H1 絕對腿的問題是真實參照本身在區間外,與配方無關;(2) recipe B 規定的 `p∈{0.60,0.65,0.70}` 與本設計族實測的 p(0.4747/0.5221)矛盾;(3) A′ 的 λ₀ 來自實測六 pair 計數,已被 §4.2 證明是 H2 頂層誤差所需要的量;(4) 兩種配方在 30M scaling 帳上的差異在千分位以下。

```
normalization   = n2(對稱 Sinkhorn 推廣,§4.4)
kernel(主線)    = K1 power_law, α = 0.10230633302323285
λ₀_tile         = 4387.31377772961
per-tile budget = B = 3λ₀_tile = 13,161.9413(N2 定義,每 tile 端子與位置無關)
primary seed    = 0(全檔落地);seeds 1–4 只存 recipe sha256
```

3×3 的 N2 在角/邊/中心三種鄰居形狀上不再有閉式解(`glue_gen.n2_pair_counts` 對 3×3 原明文 `raise NotImplementedError`)——裁決選擇**對稱 Sinkhorn 縮放**(而非兩端點平均的閉式近似,理由:後者在 α=1.915 時每 tile 端子偏差達 ±21%,破壞 N2 唯一的正當性「每 tile 端子與位置無關」)。Gauss–Seidel 迭代 `a_u ← B / Σ_{v≠u} a_v w_uv`,收斂判準 `max_u|Σ_v c(u,v)−B|/B < 1e-9`;在 1×C/R×1/2×2 上與舊公式逐位元等價(已建陣列不受影響)。

**Sinkhorn 初始化揭露(commit `1b4af33`):** 初始化 `a_u=1.0` 是裁決文件未指定的實作選擇。四個核變體的收斂步數分別為 K1 主線(α=0.10230633302323285)16 步、K1(α=1.915,樂觀端)15 步、K1(α=0,平坦)16 步、K3(截斷)56 步——與裁決文件原本探針記下的 21/19/21/72 步不同,但兩者在每一個受 gate 的量上一致(每 tile 端子預算誤差 <1e-9、總數 `mB/2`、d=1 pair 佔比),差距 ≤0.3 個百分點;已建陣列(K1 主線、K3)不受此差異影響。

**K1 + K3 陣列參數表(3×3,已落地,`results/m4/bench/arrays/{3x3_n2,3x3_n2_k3}/*.manifest.json`):**

| 核 | kernel | α | 落地形式 | 實際 `n_nets`/`n_pins` | N2 期望值(對所有核恆等) | d=1 pair 佔比 |
|---|---|---:|---|---:|---:|---:|
| K1(主線) | power_law | 0.10230633302323285 | 全陣列落地(sha256) | 59,360 / 118,720 | 59,228.74 | 34.6% |
| K3(截斷) | truncated(φ=0 for d>√2) | 0.10230633302323285 | 全陣列落地(明示 locality 下界) | 59,280 / 118,560 | 59,228.74 | 74.3% |
| K1(α=0,平坦) | power_law | 0 | recipe-only(sha256+pair 表) | 59,220(遞迴驗算) | 59,228.74 | 33.3% |
| K1(α=1.915,樂觀端) | power_law | 1.915 | recipe-only | 59,228(遞迴驗算) | 59,228.74 | 58.9% |
| K2 | exp | 0.08558 | recipe-only | — | 59,228.74 | 34.8% |

**結構性發現(必須寫入報告,裁決 §5.3):在 N2 之下,核的選擇完全不改變 glue 總數**(每 tile 預算固定,核只重新分配),只改變空間局部性——d=1 佔比 33.3%→74.3%(>2×)。⇒ spec §3.2 line 304「三核總數全距 > 20% 才附 K3 變體」在 N2 下恆為 0%,已退化為無資訊觸發條件;K3 因此**照產不因此條件而省略**。

### 4.4 T6B(3×3 自洽性驗收):B-0/B-3/B-4/B-6 全綠,B-5 重算後全綠

`results/m4/bench/verify_group3x3_t6b.json` + `results/m4/bench/b5_recompute.json`:

| # | 指標 | 結果 | 狀態 |
|---|---|---|---|
| B-0 | V0′ 結構完整性(來源相對判準) | self-loop 0.0494 vs 來源 0.0495(不劣於來源)、`unreferenced_count=18=2×9`(=來源 2 個 ×R×C) | **ok** |
| B-3 | H3′ 全網 degree KS(vs 來源 group) | KS = **0.000626** | **ok**(門檻 0.05) |
| B-4 | H4′ interface-cell 自洽 | KS = **0.0103** | **ok**(門檻 0.10) |
| B-5a | 配方算術(N2/Sinkhorn 總數 vs 建構公式) | rel_err = **1.24e-5** | **ok**(門檻 1e-3) |
| B-5b | 建檔忠實性 | `glue.n_nets=59,360 == Σ manifest.sampled_pair_counts` | **ok**(精確相等) |
| B-5c | 抽樣噪聲(揭露,不判定) | z = 0.539σ | 揭露 |
| B-6 | 每 tile 端子預算恆等式(Sinkhorn) | 四個核 `max_rel_err_vs_B` 皆 ≤ 8.3e-10 | **ok**(門檻 1e-9) |

**B-5 的判定演變本身值得記錄:** 最初一版(commit `daf434b`)B-5 用單一「N1 formula 但套 N2 陣列」的期望值誤判為 fail(rel_err=60.8%)。依附錄 A.1 的三條分辨規則(被比較的量算錯了、依據早於量測、不含新數值)全部成立 ⇒ **判定為可執行的判準修正,不是事後放寬**,拆成 B-5a(配方算術)/B-5b(建檔忠實性)/B-5c(抽樣噪聲揭露)三支重算(commit `9760f36`),結果全綠。**核敏感度預先登錄的預測全數命中**(`kernel_sensitivity.budget_check` 四個變體 `hit=true`、`order_of_magnitude_miss=false`):K1(α=0.102)預測 rel_err≈1.24e-5、K1(α=0)預測≈1.48e-4、K1(α=1.915)預測≈1.24e-5、K3 預測≈8.0e-5——實測與預先登錄值逐位吻合。

**T6(2×2)自身的 H5′ 事實修正(裁決附錄 A.4,必須逐字進報告):**

> T6 的 2×2 N2 陣列的 H5′ 自 commit 1b3671c 起即記為 `fail`(rel_err = 1.24e-2),位於診斷不判定區塊而未被察覺;其成因與 T6B 的 B-5 相同——`check_h5prime_self_consistency` 的期望值公式寫於 N2 正規化存在之前,對 N2 陣列套用了 N1 的 `Σλ₀φ(d)`。2×2 N1 那格的 `ok`(rel_err = 6.07e-4)公式正確,但它是單一 Poisson 抽樣的實現值與其期望值的比較,`z = +0.098σ`;依該檢查函式自身的文件,單一 Poisson 種子本就不應被 1e-3 容差判定。同樣的計算若當初施於 1×2 N2 陣列會得到 rel_err = 3.42e-3,同樣不過。

這是一項**事實修正,不改判 T6 的整體 verdict**——§4.1 的 FAIL 由 H1 絕對腿與 H6 決定,H5′ 本身在 T6 屬診斷不判定區塊,從未進入 verdict 計算;此處記錄純為誠實揭露一個先前未被察覺、與 B-5 同源的量測工具缺陷。對應數字見 `results/m4/bench/b5_recompute.json` 的 `a4_verification` 欄:`1x2_n2_correct_formula_poisson_draw` z=0.393、`2x2_n1_correct_formula` z=0.098、`2x2_n2_buggy_n1_formula`(即原 T6 的 bug,逐位元保留)rel_err=1.24e-2。

**per-pair 抽樣的同號偏誤(裁決附錄 A.5,揭露不判定,不採取行動):** `b5_recompute.json` 的 `a5_per_pair_signs` 欄與各陣列的 `b5c_sampling_noise.z` 顯示,五個已建陣列(1×2_n1、1×2_n2、2×2_n1、2×2_n2、3×3_n2)的 Poisson 總數偏差 z 值**全部同號為正**:+0.388σ / +0.393σ / +0.098σ / +0.106σ / +0.539σ;**5/5 同號,單尾符號檢定 p ≈ 0.03(= 0.5⁵)**。依附錄 A.5 政策,此為揭露事項,**不觸發任何判準調整或動作**;`a5_per_pair_signs` 欄同時對每個陣列的 per-pair(`min_expected≥42`)偏差各自跑了符號檢定與 Poisson 適合度檢定,結果(§4.7 已引用 3×3_n2 的 `chi2_p=0.554`)不支持系統性生成偏誤——5 陣列層級的同號現象與 per-pair 層級的無顯著偏誤是兩個不同粒度的觀察,皆如實記錄,不相互抵銷或加強對方的判定力。

**B-1(3×3 Rent 形狀不變性 DiD)與 B-2(等解析度 K-grid λ)皆為 `pending_measurement`**——尚未在本報告範圍內執行(B-1 需要 `rent.measure_rent` 在 `3x3_n2`/`3x3_noglue`/`2x2_noglue` 三個陣列上各跑一次,預估 45–60 min/run;B-2 是 T8/T9 產物)。<!-- PENDING: B-1(Rent 形狀不變性 DiD,3×3 vs 2×2 vs 無 glue 對照)——`verify_group3x3_t6b.json` 現況 `status="pending_measurement"`,預先登錄預測 `p(3×3,N2,seed 0)=0.530±0.008`,尚待 `rent.measure_rent` 在三個陣列上執行。B-2(H5 K-grid,等 per-tile 解析度)同樣 `pending_measurement`,待 T8/T9。 -->

依裁決 §4.1 的驗收規則(**B-0 與 B-3…B-6 全綠 ⇒ 3×3 bench 可用於 scaling**),`verify_group3x3_t6b.json` 記 **`scaling_usable=true`**,`generator_verified=false`,`quality_claims_prohibited=true`。

### 4.5 V0′ 判準修正(§7-5,逐字)

> `verify_bench.check_v0_structural` 對真實衍生網表有三條必然誤判的子檢查,2×2 的 1,338 條 error 全部出自它們,**沒有一條源自 tiler**:(a)「self-loop」693,952 條 = 來源 `mempool_group` 自身 173,488 條(占其 nets 的 4.951%)× 4,一條 net 接到同一 cell 的兩隻腳在真實網表中合法;(b) row「縫隙」出現在 macro 覆蓋處,含 macro 的 floorplan 在該處本就不該有 placement row(`tile_bookshelf.py:259-276` 已於建構期記載此重新詮釋,tiler 自身的 assert 早已改為只查 overlap);(c) 8 個未被任何 net 參照的 node = 來源 2 個 × 4。三條均改為「不劣於來源」的相對判準後重跑。V0 依 2026-08-14 裁決為診斷不判定項。

### 4.6 核不可識別性(§7-6,逐字)

> 距離核的形狀在 2×2 上有 0 個 lack-of-fit 自由度(§3.2),因此 3×3 的核是一個**宣告的假設**。在 N2 正規化之下,核的選擇**完全不改變 glue 總數**(每 tile 端子預算固定,核只重新分配),只改變空間局部性:d=1 pair 所佔比例在 K1(α=0.102)為 34.6%、K2(指數,α′=0.0856)34.8%、K1(α=1.915)58.9%、K3(截斷,不連 d>√2)74.3%。因此 §3.2 line 304 的「三核總數全距 > 20% 才附 K3 變體」在 N2 下恆為 0%,已退化為無資訊的觸發條件;本報告改以距離層級分佈報告系統性不確定度,並無條件附上 K3 全陣列作為「不模擬 long-range 連通性」的明示下界。

### 4.7 per-pair 抽樣的 Poisson 適合度(揭露,不判定,不採取行動)

`results/m4/bench/glue_poisson_gof.json`(附錄 A.5):彙總 5 個已建陣列、50 對符合 `min_expected≥42` 的 pair,實現值相對期望值 **33 正 / 17 負 / 0 平**,雙尾精確二項符號檢定 **p=0.0328**(弱顯著);但 Poisson 適合度檢定(逐 pair 變異數結構)**卡方=52.34,df=50,p=0.3834**——不支持存在系統性生成偏誤。3×3_n2 單獨一項:36 對中 21 正/15 負,`chi2_p=0.554`。**結論:僅揭露、不判定、不採取行動**(依附錄 A.5 政策,此發現不影響 §4.4 的任何綠燈);觸發的後續調查(`sample_glue_net_count_from_expected` 的 per-pair 偏正複查)留給下一輪。

---

## 5. T8b 校準(seed A)——**calibration only, not report arms**

**這是校準表,不是品質表**:下列格點用於**依預先登錄規則選出 `ρ*`**,不是本報告任何 E3 判準所引用的正式結果;報告用的正式 run 依 T8b 協定一律跑 **seed B**(§9),此處 seed A 的 `run_id` 不得出現在任何最終品質表中。

**選取規則(逐字取自 `results/m4/calib/rho_*.json` 的 `selection_rule`)**:取「`Δhpwl ≤ +5.0%` 前提下最大的 `ρ`」(固定格點 `{0.05, 0.10, 0.15, 0.20}`),以約束選、不以 `Δio` 選(winner-bias 控制);**格點不得事後擴充(M4-G2)**。

### 5.1 三案完整格點(K=16, grid, 2000-iter 預算)

| case(規模) | `ρ_max` | `d_hpwl_pct` | `d_io_pct` | `gp_iter` | `final_overflow` | `run_id` |
|---|---:|---:|---:|---:|---:|---|
| **mempool_group(3.1M)** | 0(flat) | 0 | 0 | 1009 | 0.0697 | `32ba3cd0-4514-4910-9a2b-263b415b6a87` |
| | 0.05 | +0.237% | −2.93% | 995 | 0.0700 | `583385d5-b9df-4fbb-934a-511138afcab8` |
| | 0.10 | +0.287% | −6.24% | 1005 | 0.0697 | `d73d353e-38b1-4e50-98a2-6a58527c48eb` |
| | 0.15 | +0.712% | −7.53% | 1001 | 0.0700 | `b1c2dc4a-7eef-43d7-beaf-bbf8eb8d8513` |
| | **0.20(=ρ\*)** | +1.086% | −8.96% | 1003 | 0.0700 | `27a4877b-eb50-4049-848d-8c119fde81e7` |
| **superblue12(1.3M)** | 0(flat) | 0 | 0 | 1052 | 0.0639 | `51193258-7731-4f07-a1cc-61432f8134fe` |
| | 0.05 | +0.711% | −10.74% | 1048 | 0.0639 | `d7dd3ed5-2184-4f82-8130-48ee50e3b01c` |
| | 0.10 | +1.320% | −16.91% | 1052 | 0.0637 | `affb37e6-846d-405c-86b1-80f4fadbf537` |
| | 0.15 | +1.900% | −21.08% | 1054 | 0.0648 | `73a75e96-229f-4f2d-844c-22beba132702` |
| | **0.20(=ρ\*)** | +2.843% | −25.81% | 1060 | 0.0657 | `244fead5-dce9-4dc6-9ebf-9f3801a57802` |
| **bigblue4(2.2M)** | 0(flat) | 0 | 0 | 846 | 0.0637 | `3728496e-7de1-4682-a01d-412401b17166` |
| | 0.05 | +0.191% | −10.66% | 843 | 0.0658 | `89df200b-e07f-4f7d-800b-d71d29f2a95f` |
| | 0.10 | +1.215% | −21.99% | 839 | 0.0697 | `7aff161c-3268-42b5-8cbe-191f57205c13` |
| | 0.15 | +3.252% | −34.00% | 927 | 0.0696 | `5917bf9a-14da-481b-ad16-530eb5112d5b` |
| | **0.20(=ρ\*)** | +4.616% | −35.31% | 944 | 0.0692 | `0d7dd6f7-1eb2-4b01-9e2c-971fa1f1c06c` |

**格點邊界註記(三案一致):`ρ_star_at_grid_boundary=true`——0.20 是四點格點中最大的候選值,三案的 `Δhpwl` 在 0.20 時都還沒碰到 +5% 的約束(group 1.09%、sb12 2.84%、bb4 4.62%,最貼近的 bb4 距上限仍有 0.38 個百分點),意即真正的最大可行 `ρ` 可能高於 0.20。依 M4-G2「不得臨時擴充格點」,本輪**不**外推;此格點邊界效應留待下一輪校準協定明確納入更大候選值時處理。**

### 5.2 mempool_cluster(11.3M)格點——K=16 全格點已完成(seed A),K=32 獨立臂仍缺

`mempool_cluster` 已走完與 §5.1 同款的完整四點格點(`results/m4/calib/rho_cluster.json`,commit `6587682`),適用同一條**選取規則**(取 `Δhpwl ≤ +5.0%` 前提下最大的 `ρ`,固定格點 `{0.05, 0.10, 0.15, 0.20}`,以約束選、不以 `Δio` 選):

| `ρ_max` | K | `d_hpwl_pct` | `d_io_pct` | `final_overflow` | `gp_iter`/備註 | `run_id` |
|---|---:|---:|---:|---:|---|---|
| 0(flat) | 16 | 0 | 0 | 0.0698 | 1252(GP+LG 完整跑) | `b60324cc-c06c-4cd5-9253-db590454c800` |
| 0.05 | 16 | −0.005% | −2.56% | 0.0694 | 1281(GP+LG 完整跑) | `09bc5d56-89ea-4c80-9727-b940789b3db6` |
| 0.10 | 16 | +0.284% | −6.11% | 0.0699 | 1277(GP+LG 完整跑) | `e965846e-f2c9-4d67-8c4a-4bd6cf9c0ffd` |
| 0.15 | 16 | +0.791% | −9.45% | 0.0697 | 1293(GP+LG 完整跑) | `9c8a5ba8-48d9-4089-b6ad-a4369aaacef2` |
| **0.20(=ρ\*)** | 16 | +1.545% | −13.06% | 0.0693 | 1304(GP+LG 完整跑) | `8a81eeff-d6e7-4176-a3fb-2cc8b59dee67` |
| 0(flat) | 32 | 0(與 K16 flat 逐位元同) | — | — | evaluate-only,重用 K16 flat 的 npz(`source_npz_sha256=db1df9…`) | `4b6002fd-7eee-478d-a136-922b97efd377` |

**格點邊界(與 §5.1 三案一致,四案合計全部 ρ\*=0.20 且全在格點邊界):** 0.20 是四點格點中最大的候選值;`d_hpwl_pct` 在 0.20 時為 +1.545%,介於 §5.1 的 group(+1.09%)與 sb12(+2.84%)之間,同樣離 +5% 約束有相當距離(3.46 個百分點)——真正的最大可行 `ρ` 可能更高。依 M4-G2「不得臨時擴充格點」,本輪**不**外推,理由與 §5.1 相同。

**`seed_B` 欄位歷史(引用時說明):** 本表與 §5.1 引用的 `seed_B=2000` 均為修正後的值。`group`/`sb12`/`bb4` 三案的 calib 檔在 commit `85c1c59` 一度把 `seed_B` 誤記為 **1001**,依附錄 A.3 的跨 session 協議由 commit `9830ff6` 修正為 **2000**;`rho_cluster.json` 建立於 `9830ff6` 之後(commit `6587682`),自始即為 2000,未經過 1001 這個中間值。

**仍未完成的部分:** (i) K=32 目前只有對 K16 flat placement 的 evaluate-only 重新計分(上表最後一列),不是獨立收斂的 K=32 臂,且完全沒有 K=32 的 `ρ_max>0` 臂;(ii) 上表全部五個 K=16 格點與 K=32 flat 都是 **seed A**(=1000)的校準/篩選性質資料,不是 T8b 協定要求的正式 seed-B report run——§5.3 已有 sb12/bb4/group 三案的正式 seed-B 矩陣,cluster 尚未排入。

<!-- PENDING: mempool_cluster 的 K=32 獨立 GP+LG `ours@M2` run(目前只有對 K16 flat 的 evaluate-only 重新計分)、以及 cluster 的正式 seed-B report run(K∈{16,32}×{flat,ours@M2} 共 4 格,見 §5.3)——待排入 seed-B 校準佇列。 -->

### 5.3 Seed-B 品質矩陣(sb12/bb4/group,部分)

`results/m4/profile/{case}__k{16,32}__grid__{flat,oursM2}.json`(`seed=2000`,即附錄 A.3 鎖定的正式 seed B):三個小型真實 case(superblue12/bigblue4/mempool_group)在 K∈{16,32}×{flat, ours@M2(ρ_max=ρ\*=0.20)} 下的完整矩陣已落地,`ours@M2` 一律取各案 §5.1 選定的 ρ\*=0.20。**mempool_cluster 的對應四格尚未排入 seed-B 佇列,標 PENDING(見表後)。**

| case | K | arm | io_abs | ft_abs | hpwl_abs | d_io_pct | d_ft_pct | d_hpwl_pct | num_unplaced_cells | final_overflow | t_total_s | t_gp_s | gpu_peak_gb | host_hwm_gb | run_id |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sb12(1.3M) | 16 | flat | 62,055 | 3,284 | 260,669,447 | 0 | 0 | 0 | 0 | 0.0643 | 340.2 | 75.2 | 3.09 | 6.53 | `2e311be1-dd06-4377-b170-a07ec3eb9550` |
| sb12(1.3M) | 16 | ours@M2 | 45,125 | 4,190 | 267,797,173 | −27.28% | +27.59% | +2.73% | 0 | 0.0664 | 514.1 | 256.3 | 2.80 | 6.54 | `f4f861e8-6f80-45a6-b9a3-ef2635374e52` |
| sb12(1.3M) | 32 | flat† | 106,059 | 9,324 | 260,669,447 | — | — | — | — | — | 266.9 | — | — | — | `1a1388d4-bec1-40b5-955c-51d3ae76fa5b` |
| sb12(1.3M) | 32 | ours@M2 | 80,696 | 11,013 | 265,249,308 | −23.91%‡ | +18.11%‡ | +1.76%‡ | 0 | 0.0631 | 765.5 | 491.5 | 4.43 | 6.55 | `94f20b94-8e89-48c1-bde5-0ddd0d456c79` |
| bb4(2.2M) | 16 | flat | 99,912 | 6,969 | 748,432,433 | 0 | 0 | 0 | 0 | 0.0646 | 572.4 | 56.2 | 4.06 | 10.22 | `a77e59b8-1f06-4edd-a6c6-1d9c224271f9` |
| bb4(2.2M) | 16 | ours@M2 | 64,753 | 8,873 | 782,633,935 | −35.19% | +27.32% | +4.57% | 0 | 0.0697 | 950.0 | 437.1 | 4.06 | 10.25 | `ddcf9e53-4844-45fa-8b3c-2dbea195b95e` |
| bb4(2.2M) | 32 | flat† | 159,947 | 19,193 | 748,432,433 | — | — | — | — | — | 524.2 | — | — | — | `3286a58f-0e8e-4ecb-b882-95f315b17b8c` |
| bb4(2.2M) | 32 | ours@M2 | 118,373 | 21,842 | 763,011,690 | −25.99%‡ | +13.80%‡ | +1.95%‡ | 0 | 0.0693 | 1262.0 | 739.8 | 4.73 | 10.24 | `95e62992-6aca-41bf-b519-66ce0bf1533a` |
| group(3.1M) | 16 | flat | 97,803 | 9,170 | 485,777,937 | 0 | 0 | 0 | 0 | 0.0698 | 767.2 | 78.9 | 4.01 | 18.51 | `3f9b9799-0d19-4b67-bdcf-7a18e9438926` |
| group(3.1M) | 16 | ours@M2 | 89,634 | 10,843 | 491,191,451 | −8.35% | +18.24% | +1.11% | 0 | 0.0699 | 1363.9 | 686.0 | 4.01 | 18.55 | `e5bbb626-1e74-4e7d-890b-7c5e43dc587a` |
| group(3.1M) | 32 | flat† | 162,479 | 25,472 | 485,777,937 | — | — | — | — | — | 661.3 | — | — | — | `8e117c64-a5cd-40af-90cd-5469a7d0dbe5` |
| group(3.1M) | 32 | ours@M2 | 143,783 | 27,126 | 489,615,430 | −11.51%‡ | +6.49%‡ | +0.79%‡ | 0 | 0.0700 | 1982.9 | 1286.3 | 6.85 | 18.61 | `7c17b74e-3d4d-4a4c-ad65-ad518c9be2b9` |

†:K=32 的 `flat` 是對同案 K=16 `flat` placement 的 **evaluate-only** 重新計分(`mode="evaluate_only"`,重用 K16 flat 的 `.npz`),不是獨立收斂的 K=32 GP+LG 臂——`hpwl_abs` 因此與同案 K=16 `flat` 逐位元相同;`final_overflow`/`t_gp_s`/`gpu_peak_gb`/`host_hwm_gb` 留空,因為 evaluate-only run 沒有這些欄位。
‡:K=32 的 `d_io_pct`/`d_ft_pct`/`d_hpwl_pct` 是相對於同案**同 K** 的 `flat†`(evaluate-only 重新計分基準),不是相對 K=16 flat;K=32 `ours@M2` 本身是獨立收斂的 GP+LG 臂(`final_overflow`/`gpu_peak_gb` 等欄位齊全,非 evaluate-only)。

`io_abs`/`ft_abs` = evaluator 主判準(JSON 的 `io_count`/`ft_count` 欄位,即 spec §6.2 的 `io_mst`/`ft_mst`);`io_rg`/`ft_rg` 欄位省略——本表六個 `ours@M2` run 皆無此欄位(`ours@M3` 專屬,依前言紅線本報告不納入任何 `ours@M3` 臂)。`ours@M2` 一律 `ρ_max=0.20`(= §5.1 選定的各案 ρ\*)。

**此表尚未走過 §7.0 RESULT GATE 的正式收錄流程,不是定稿條目**——資料尚未過 report linter 全套規則前不宜視為已定稿。
**PENDING(定稿硬性步驟,不得跳過)**:cluster 四格補齊後,本表 caption 與欄名必須改掛正式的 `quality_real_cases` 格式(spec §6.2 欄位原名),使 `scripts/m4_report_lint.py --strict` 的 RESULT GATE / 唯一 run_id / 合成越界規則**真正咬住本表**——草稿期避開 linter 標記是既定慣例(交接暗規則 #1),但定稿時若忘記換 caption,linter 將永遠不檢查本表,即構成實質規避。

<!-- PENDING: mempool_cluster 的 seed-B K∈{16,32}×{flat,ours@M2} 四格(見 §5.2);合成 case(6.2M/12.3M)的 scaling 對照表;本表正式收錄並過 report linter 全套規則。 -->

---

## 6. 記憶體與 scaling 帳

### 6.1 count freeze(小陣列,T6b)

`results/m4/scaling/count_freeze_small.json`(N2,實際 manifest 計數):

| 級 | `base_n_nets`/`base_n_pins` | `g_n2`/`gp_n2`(實際 glue) | glue 佔比(nets/pins) | int32 headroom(nets/pins) |
|---|---:|---:|---:|---:|
| 1×2(6.2M) | 7,007,976 / 24,052,382 | 13,207 / 26,414 | 0.188% / 0.110% | 305× / 88× |
| 2×2(12.3M) | 14,015,952 / 48,104,764 | 26,341 / 52,682 | 0.188% / 0.109% | 152× / 44× |

N1 的計數(rejected alternative,僅供讀者檢核正規化選擇的量級後果):1×2 = 4,413 nets / 8,826 pins;2×2 = 26,034 nets / 52,068 pins。⇒ **M4-L12 解除**——spec §2.1 原把合成級的 glue 當 0,修正量 < 0.2%,對下游記憶體/runtime 帳影響在第三位小數。

<!-- PENDING: §1.1 三係數 host RSS 模型(fitting 集 adaptec1/bigblue4/group-BS/1×2/net-drop 變體,2×2 只作 holdout)——`count_freeze_small.json` 明記 `host_rss_three_coefficient_model.status="pending_t0b"`,尚未執行。 -->

### 6.2 count freeze(27.7M,T7b)

`results/m4/scaling/count_freeze_30m.json`:`total_n_nets=31,595,252`、`total_n_pins=108,354,439`(base 31,535,892/108,235,719 + glue `g9=59,360`/`gp9=118,720`,佔比 0.188%/0.110%,與 T6b 一致)。

**int32 headroom 揭露(附錄 A.6,無回退):** composite key(`net_id*64+region_id`)欄位在 27.7M 顯示 `warning_headroom_lt_20pct=true`(headroom 僅 6.2%)。但這是一個**反事實**——composite key 從最初設計就是 int64(`evaluator_gpu.py` 已有 `assert n_nets*64 < 2**63`),6.2% 只是「若當初選 int32」會有多緊,並非現行程式路徑的真實風險;**`action_required=false`**。真正在用的 int32 index 欄位(非 composite key)在 27.7M 的實際 headroom:nets 67×、pins 18.8×,全部 `> 20%`。

### 6.3 evaluator streaming(T2)——已完成,acceptance 全過

T2 消滅 `(P,K)` 全量物化、批次化 MST edge/segment、逐欄位 assert 後降 int32(composite key 保持 int64)。本報告當場重跑 `tests/test_evaluator_gpu.py` 的兩條 T2 acceptance 測試(`-m slow`,獨立 process 量測):

| 測試 | 結果 | 門檻 |
|---|---|---|
| `test_gpu_evaluator_memory_bigblue4_k32_reduction` | peak_alloc=**1.2579 GB**;相對 pre-M3-T1(8.3548 GB)降 **84.9%**;相對 pre-T2 HEAD(6.2445 GB)降 **79.9%** | 兩者皆 ≥60% |
| `test_gpu_evaluator_memory_mempool_group_k32_under_2gb` | mempool_group(3,503,992 nets)K=32 peak_alloc=**1.8263 GB** | ≤2 GB |

兩條測試皆 PASS(`2 passed in 151.36s` 全套 / `1 passed in 48.73s` 單獨重跑 bigblue4 那條)。**間接交叉驗證**:`mempool_cluster`(43.9M pins)K=16 的完整 GP+LG+evaluator+IoTerm 端到端 run 總 `device_used_gb=12.85 GB`(§3);若 evaluator 仍是未經 T2 改造前的版本,單獨這一項在 K=16、12.7M nets 規模下依 spec §1.2 舊模型 `(0.85+0.061·K)GB/M-net` 估算就要 **≈23 GB**——遠超這次實測的全端到端總量。T2 對 cluster 能在 L4 上完整跑完是**必要條件**,不只是理論改善。

### 6.4 27.7M read-phase 瓶頸(forecast 草稿的核心發現)

`results/m4/forecast/h100_prediction.draft.json` 用 seed-A `mempool_cluster` flat run(§3)的 phase 拆分做 count-scale 外推(node_ratio=2.457×,由 `count_freeze_30m.json` 的 27.7M 實數 ÷ 探針的 11.31M 實數):

| phase | `t_l4_measured_s`(11.3M) | 外推至 27.7M(`s_p=1.0` 或帶) | 假設 |
|---|---:|---:|---|
| read/parse | **9,250.9** | **22,731.4** | `s_p=1.0`(單執行緒 CPU parser,GPU 換代不受益) |
| GP | 406.1 | 199.6–399.1 | `s_p∈[2.5,5.0]`(BW/FP32 比,未驗證假設) |
| LG | 84.0 | 206.4 | `s_p=1.0` |
| evaluator | 2,183.2 | 1,076.6–2,153.1 | `s_p∈[2.5,5.0]`(同上) |
| **總計** | 11,924.2 | **24,213.9–25,490.0**(6.73–7.08 h) | — |

**核心發現:read/parse 一個 phase 就佔外推總量的 89–94%**,且其 `s_p=1.0`(CPU 單執行緒,不因換到 H100 而加速)——27.7M 這一級的關鍵路徑因此**不是 GPU 算力/頻寬受限,而是 host 端序列讀取受限**;H100 相對 L4 的算力/頻寬優勢在這個規模下幾乎不改變總 wall-time。這是本輪唯一被明確標記為「forecast 的關鍵結論」的發現,詳見 §8。

<!-- PENDING: 這條 read-phase 發現目前只由 T7b 的 count-scale 外推支撐(`unvalidated: true`,無 H100 實測),T14 首跑才能證實或推翻——見 §8。 -->

### 6.5 T0b factorial 擬合:GP 記憶體/runtime 模型(誠實負結果,M4-G6 觸發)

`results/m4/scaling/model_fit.json`:預註冊的 16 點設計(9 點 `mempool_group` 的 `bins × target_density` factorial + 2 點 net-drop 變體 + 5 點 case 級 probe——`adaptec1`、`mempool_cluster_probe2500`、`sb12`/`group`/`cluster` 各自的 `flat_k16`)**全部執行完畢**(`n_points_used=16`、`skipped=[]`,無一跳過),對 design draft §1.1 的其中兩個模型做 OLS 擬合:

| 模型 | 係數(值) | κ(condition number) | κ 門檻 | CI gate(相對半寬 ≤0.25) | `identifiable` |
|---|---|---:|---:|---|---|
| GP 記憶體(`gp_peak_bytes ~ intercept + n_bins + N_pins + N_total`) | intercept 5.952e8、n_bins 531.46、**N_pins −598.44**、N_total 2297.25 | **15.09** | ≤30(過) | **不過**——intercept 相對半寬 **3.04** | **false** |
| GP runtime(`t_gp_s ~ intercept + n_iter·N_pins + n_iter·N_total + n_iter·n_bins·log(n_bins)`) | intercept −11.85、**n_iter·N_pins −1.61e-9**、n_iter·N_total 2.88e-8、n_iter·n_bins·log(n_bins) 1.46e-10 | **23.62** | ≤30(過) | **不過**——`n_iter·N_pins` 相對半寬 **7.76** | **false** |

兩個模型的 κ_std(15.09/23.62)都在 ≤30 的共線性門檻之內——**不是共線性問題**,是樣本量不足以把每個係數的 95% CI 收窄到 ≤0.25 相對半寬的門檻內(記憶體模型的 intercept、runtime 模型的 `n_iter·N_pins` 都超標數倍)。更嚴重的是**記憶體模型的 `N_pins` 係數為負(−598.44 bytes/pin)**——每多一個 pin 反而預測記憶體下降,物理上不合理,是過度參數化下的擬合假影,不是真實的省記憶體機制。⇒ 頂層 `identifiable=false`。

依 **M4-G6**(design draft §9,`identifiable=false` 或殘差超出 95% PI ⇒ 模型不可外推)觸發後,三個直接後果:

1. spec §2.1 定案階梯表 27.7M 列的 host RAM 欄(原「95–105 GB(單點外推,判定見 §5.1)」)在本報告的任何引用中一律改記**「未定」**,不得沿用該單點外推值。
2. E5 的 host RSS 預測必須是**解析估計、不得帶 95% PI**——這正是 §8 現有文字已經採取的立場(點估計 168.9 GB、`interval=null`);此處確認該立場是 M4-G6 觸發後的正確處置,不是尚待處置的缺口。
3. `count_freeze_small.json` 的 `host_rss_three_coefficient_model` 欄**維持 `pending_t0b`**——**需要澄清一個容易混淆的範圍界線**:本輪 T0b(`model_fit.json`)擬合的是 design draft §1.1 的 GP 記憶體/runtime 兩個模型,**不包含**同一節提到的第三個模型(`PlaceDB.read` host RSS 三係數模型,fitting 集 adaptec1/bigblue4/group-BS/1×2/net-drop 變體、2×2 純 holdout)——那個模型完全沒有被這輪 16 點設計覆蓋,`count_freeze_small.json` 對它的 `status="pending_t0b"` 描述依然精確成立,本報告在此明確**不回填**任何 host RSS 係數數字。

**誠實負結果的語氣(比照 M1/M2 慣例):** 這不是「T0b 沒做」,是「T0b 做了,而且做出一個誠實的『不能用』結論」——16 個設計點全部收斂執行、無一跳過,兩個模型的係數估計本身算出來了,只是置信區間寬到不能外推、且其中一個係數方向物理不合理。把這個結果藏起來或悄悄放寬 CI 門檻,比不做 T0b 更糟;M4-G6 存在的目的正是防止這種情況下的沉默外推。

---

## 7. E1–E5 逐條

### E1(10M 全流程)—— 部分可判

**已判定的部分**:`mempool_cluster` overflow 判準的 FAIL→PASS 轉變(§3)——`final_overflow` 由 1000-iter 上限下的 0.327(`experiment_status="superseded_iteration_capped"`,run_id `9dec7b71-af7c-4d62-b3cd-cc76aba244df`)轉為 2000-iter 下的 0.0698(run_id `b60324cc-c06c-4cd5-9253-db590454c800`)。以下欄位在 seed-A flat/rho0.05 兩點皆已核對:

| arm | K | `hpwl_abs` | `final_overflow` | `legalization_status` | `num_unplaced_cells` | `device_used_gb` | `runtime_h` | `run_id` |
|---|---:|---:|---:|---|---:|---:|---:|---|
| flat | 16 | 2,095,445,937 | 0.0698 | success | 0 | 12.85 | 3.33 | `b60324cc-c06c-4cd5-9253-db590454c800` |
| `ρ_max=0.05` | 16 | 2,095,339,300 | 0.0694 | success | 0 | 13.47 | 1.86 | `09bc5d56-89ea-4c80-9727-b940789b3db6` |
| flat(evaluate-only) | 32 | 2,095,445,937(同上,逐位元) | — | — | — | — | — | `4b6002fd-7eee-478d-a136-922b97efd377` |

`Δhpwl` 相對自身 flat ≤ +5%(rho0.05 實為 −0.005%)、`num_unplaced_cells==0`、`legalization_status=="success"`、`device_used_gb ≤ 20 GB`、單臂 wall-time ≤ 4h——六個 E1 分項在這兩點上**全部滿足**。

**尚未判定的部分**:(i) 這是 seed A 的篩選/校準性質資料,不是 T8b 協定要求的正式 seed-B report run;(ii) K=32 目前只有對 K16 flat 的 evaluate-only 重新計分,不是獨立 GP+LG 收斂的 K=32 臂,且完全沒有 K=32 的 `ours@M2`(`ρ_max>0`)臂;(iii) `quality_real_cases.md` 的正式表格條目未建立。⇒ **E1 = 部分可判**,`<!-- PENDING: seed-B 正式 run + K=32 獨立 ours@M2 臂,見 §5.2 -->`。

### E2(30M 測資)—— PENDING

三個子條件逐一交代:

- (a) 27.7M Bookshelf 產出、manifest sha256、實際 glue 計數:**已達成**(§4.4、§6.2,`count_freeze_30m.json`)。
- (b) 12.3M 的 V0/V1 + H1–H5 holdout 全綠:**未達成,已判定為 FAIL 而非懸置**(§4.1,T6 holdout FAIL 維持,H1 絕對腿與 H6 兩腿不過)——依 M4-G3,已轉入 T6B 的替代自洽性驗收路徑(§4.4 全綠),但 spec §6.3 E2(b) 字面要求的「holdout 全綠」本身確實沒有發生,此處誠實記錄不代填。
- (c) 27.7M 元件級 spike(T9)在 §2.2 三態規則下的有效終態:**未執行**。

<!-- PENDING: E2(c) 待 T9(27.7M IoTerm+evaluator 共常駐 spike,`budget_gb=19.5`,三態終結規則)。因 (c) 未完成,E2 整體標 PENDING;(b) 的 FAIL 已定案不會因 T9 結果改變。 -->

### E3(全規模對照表)—— PENDING

`quality_real_cases.md`(真實 case × K{16,32} × {flat, ours@M2})與 `scaling_synthetic_cases.md`(合成 case × K{16,32})兩份正式報表皆需要 T8b 之後的 **seed-B** T8 run;目前只有 §5 的 seed-A 校準資料,依協定不得作為報表引用的 `run_id`。

<!-- PENDING: 待 T8(seed B)完成後,依 §7.0 RESULT GATE 建表,並過 `scripts/m4_report_lint.py`。 -->

### E4(記憶體 profile)—— PENDING

B1/B2/B3 三個量測工具 bug 已處置揭露(§2);T2 evaluator streaming acceptance 已過(§6.3)。但 E4 要求的是**三個模型**(GP/evaluator/IoTerm)以 T0b 的 factorial 設計重新擬合,並附 6.2M/12.3M holdout 殘差——T0b 的 11 個設計點(9 個 bins×density + 2 個 net-drop 變體)已規劃並產出 scratch config(`results/m4/scaling/t0b/_scratch_configs/`、`dry_run_plan.json`),但**尚無一個實際執行**,`results/m4/scaling/model_fit.json` 不存在。

<!-- PENDING: T0b 全部 11 個設計點的實際執行 + OLS 擬合(design matrix、condition number κ≤30、係數 95% CI 相對半寬≤25%)+ 6.2M/12.3M holdout 殘差核對。 -->

### E5(H100 forecast)—— PENDING(凍結步驟未執行)

草稿預測已產出且結構完整(`results/m4/forecast/h100_prediction.draft.json`,§6.4、§8),含 wall-time point estimate(S-BW 主線 6.73–7.08 h 敏感度帶)、GPU 記憶體解析上下界(26.1–45.2 GB)、host RSS 點估計(168.9 GB,`interval=null`,依規則因 host RSS 模型未 fit 只給點估計)。**spec 的 exit 條件是「預測已登錄且輸入可稽核」的凍結版**(`h100_prediction.json` + sha256 登錄進報告)——草稿檔的檔名本身(`.draft.json`)與 `"status": "draft"`/`"unvalidated": true` 欄位標明尚未跨過這一步。

<!-- PENDING: T10 的凍結動作(拿掉 `.draft`、鎖定 sha256、寫入本報告)。証偽本身(T14,H100 執行)排在 M4 主線之外,不計入本次 PENDING。 -->

---

## 8. H100 交接(T10)

**Scripts 清單(已存在,`283e1a5` 已提交):**

| script | 角色 |
|---|---|
| `scripts/m4_run.sh` | 單一指令(§6.4);POSIX sh,`--dry-run` 有煙霧測試(`tests/test_m4_handoff.py::test_m4_run_sh_dry_run_smoke`);OOM/crash 走 §2.2 三態規則而非靜默視為 bug |
| `scripts/m4_forecast.py` | 產生 `h100_prediction*.json`(wall-time/GPU/host RSS 三段預測、`--freeze` 旗標、拒絕無 `--force` 覆寫已凍結檔) |
| `scripts/m4_check_prediction.py` | T14 首跑比對器;`MANDATORY_REFIT_DISCLOSURE_SENTENCE`(「任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間」)逐字內嵌,防止事後放寬 |

**runbook 文件的落地形式與 spec 略有出入,誠實記錄:** spec §6.4 指名的獨立檔案 `docs/handover/h100-30m-runbook.md` 目前**不存在**;等價內容(資源契約、環境重建兩項檢查、單一指令用法、三態規則)以詳盡註解的形式寫在 `scripts/m4_run.sh` 檔頭,而非獨立 markdown。`tests/test_m4_handoff.py` 涵蓋 `m4_run.sh` 本身的 CLI 行為(`--dry-run`、`--help`、未知旗標),但**沒有**直接證據顯示 `m4_run.sh`(這個 wrapper 本身,而非其呼叫的底層 driver)已被非 dry-run 地在 3.1M case 上完整跑過一次——§5.1 的 `mempool_group` 各點是透過 `ioplace.drivers.run_placement` 直接呼叫產出,不確定是否經過 `m4_run.sh` 這層。

<!-- PENDING: (1) 補一份獨立 `docs/handover/h100-30m-runbook.md`(或明確決定維持腳本內嵌註解為最終形式並在 spec 同步);(2) 用 `m4_run.sh`(非 dry-run)在 3.1M case 上完整演練一次,產出可核對的 JSON。 -->

**草稿 forecast 關鍵數(`results/m4/forecast/h100_prediction.draft.json`,`status="draft"`,凍結前狀態):**

- SKU 釘死:H100 SXM5 80GB HBM3、3.35 TB/s peak、sm_90、CUDA 12.8、persistence mode on、預設時脈——若實際機器不同則預測**無效**,須重登錄。
- wall-time:point estimate(S-BW 主線)6.73 h,S-SM/S-FP64 敏感度帶延伸到 7.08 h;**read/parse phase 佔外推總量 89–94%,是唯一支配項**(§6.4)。
- GPU 記憶體:解析下界 26.1 GB(eval 分量主導)、上界 45.2 GB(含 allocator reserve + CUDA context),兩者皆在 H100 的 80 GB 契約內、也遠低於部件級 72 GB 契約。
- host RSS:點估計 168.9 GB(僅由 node_ratio 代理外推,無 95% PI,因 §1.1 的三係數模型尚未 fit),**逼近但未超過** `h100_min_host_ram_gb_contract=192 GB` 的 T10 資源契約——需要在 T0b host RSS 模型完成後重新核算,目前僅供參考、不作為契約通過/未通過的判定。

**凍結條件(尚未滿足,見 E5 的 PENDING):** 需 (1) 從草稿改為正式 `h100_prediction.json`;(2) sha256 寫入本報告;(3) T0b 完成後 host RSS 分項理想上應補上 95% PI(若 T0b 判定 `identifiable=false` 則依 M4-G6 改用更保守方法,不得沉默維持點估計)。T14(H100 實際執行驗證)在 M4 主線之外,不是本報告的交付範圍,但依 spec 要求必須明寫「若永不驗證」的狀態——目前 T14 尚未有任何執行紀錄,屬於「E5 若永不驗證」的現況。

---

## 9. 附錄

### 附錄 A.1 判準修正 vs 事後放寬的分辨規則(通用,預先登錄;§2 B5、§4.4 B-5 已引用)

一個事後修改屬於**可執行的判準修正**,當且僅當三條全部成立:(1) 被比較的量與預註冊文字指名的量不同(算錯了東西,不是接受域不合意);(2) 修正依據是**量測之前**就已寫定的文字(spec/docstring/先前裁決);(3) 修正後的判準**不含任何新數值**,容差逐字沿用。H1 的 `[0.55,0.80]`:第 (1) 條不成立 ⇒ FAIL 永久維持。B-5:三條全部成立 ⇒ 重算。

### 附錄 A.2 Artifact 清單(路徑 + 對應 commit)

| Artifact | 路徑 | commit |
|---|---|---|
| T3a hierarchy gate(修正後) | `results/m4/corpus/hierarchy_gate.json` | `ea91dc2` |
| T6 holdout(N1/N2) | `results/m4/bench/verify_group2x2_n{1,2}.json`、`t6_summary.json` | `1b3671c` |
| T6 holdout 裁決(2026-08-14) | `docs/results/2026-08-14-m4-t6-adjudication.md` | — |
| T6 holdout 裁決(2026-08-15,含附錄 A) | `docs/results/2026-08-15-m4-t6-holdout-adjudication.md` | — |
| T7 30M 陣列(K1 主線) | `results/m4/bench/arrays/3x3_n2/*.manifest.json` | `ceca6b6` |
| T7 30M 陣列(K3 截斷) | `results/m4/bench/arrays/3x3_n2_k3/*.manifest.json` | `717c6c4` |
| T6B B-5 重算 | `results/m4/bench/b5_recompute.json`、`verify_group3x3_t6b.json` | `9760f36` |
| glue Poisson GOF 揭露 | `results/m4/bench/glue_poisson_gof.json` | `43a38cd` |
| iteration 預算規則 | commit message `a10c02d` | `a10c02d` |
| iteration-capped 舊 run 淘汰 | `results/m4/**/superseded_iter1000/*.json` | `b458bdc` |
| overflow 診斷探針 | `results/m4/probes/probe_gp_memory__mempool_cluster_probe2500.json` | `a10c02d` |
| T8b 三案完整格點(seed A) | `results/m4/calib/rho_{group,sb12,bb4}.json`,`results/m4/t8b/{group,sb12,bb4}__k16__grid__*.json` | `85c1c59` |
| T8b cluster 部分格點(seed A) | `results/m4/t8b/cluster__k{16,32}__grid__*.json` | `f00b5e5` |
| count freeze(≤12.3M) | `results/m4/scaling/count_freeze_small.json` | `daf434b` |
| count freeze(27.7M) | `results/m4/scaling/count_freeze_30m.json` | `daf434b` |
| T2 evaluator streaming | `ioplace/evaluator_gpu.py`,`tests/test_evaluator_gpu.py` | `d1c57ee` |
| H100 forecast(草稿) | `results/m4/forecast/h100_prediction.draft.json` | `283e1a5` |
| T10 handover scripts | `scripts/m4_run.sh`、`scripts/m4_forecast.py`、`scripts/m4_check_prediction.py` | `283e1a5` |
| T11 report linter | `scripts/m4_report_lint.py`、`tests/test_m4_report_lint.py` | `eeb7922` |

### 附錄 A.3 跨 session 分工紀錄(2026-08-16,seed B=2000 協定)

commit `9830ff6`(「fix(m4-t8b): fix seed B = 2000 in calib records (cross-session agreement, before any seed-B run)」)把三份校準檔(`rho_group.json`/`rho_sb12.json`/`rho_bb4.json`)的 `seed_B` 欄位鎖定為 **2000**。三份檔案的 `seed_B_note` 逐字記錄了此協定的性質:

> seed B never numerically preregistered; fixed to 2000 by cross-session agreement 2026-08-16 BEFORE any seed-B report run started; requirement satisfied: fixed, documented, not in calibration set

即:`seed_B=2000` 本身不是 T8b 協定原文預先寫定的數字(協定只要求「固定一個 seed B,且不得是校準集合本身用過的 seed」),而是**在任何 seed-B 正式 run 開始執行之前**、由跨 session 的協作決定並寫入 artifact 的一次性選擇。三個必要性質(固定、有文件記錄、不在校準集合內)在 commit 時點已全部滿足;`seed_a=1000`(校準集合用的 seed)與 `seed_B=2000` 因此互斥,不違反 T8b「校準集合與報告集合不得共用同一顆 seed」的紅線。

<!-- PENDING: 本附錄記錄的是「seed B 已鎖定為 2000」這個事實本身,不代表 seed-B 的任何正式 run 已經執行——那些 run 仍是 E1/E3 的 PENDING 項(§7)。 -->
