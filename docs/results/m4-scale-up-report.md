# M4 規模化(1M → 30M)報告(T11 完成;M4-G8 已了結,見 §9 與附錄 A.4)

- 日期:2026-08-18(內容定稿 pass);分支 `m2-differentiable-io`,報告基準 HEAD `dc9d7de`
- 環境:`docs/dev-env.md`(`DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`、`$DP/.venv312/bin/python`、torch 2.8.0+cu128、CUDA 12.8、NVIDIA L4、`dp_commit d971880a15ef684c4a90bccd2c65d62ae7e33297`,近期跑於 hostname `NVL5`)
- 對應設計:`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`(v2.3)
- 對應裁決:`docs/results/2026-08-14-m4-t6-adjudication.md`(T6 協定 v2、G-D 撤回、Bug A/B)、`docs/results/2026-08-15-m4-t6-holdout-adjudication.md`(T6 holdout FAIL 維持、N2 選定、T6B 協定 B-0…B-6、B5、附錄 A 的 B-5 重算)
- 平行交付:Stage 2(commercial-tool 校準)另立報告 `docs/results/stage2-calibration-report.md` — G1 觸發(Innovus license 不通)降級為 OpenROAD-only ground truth;主要新發現:OpenROAD DR 在高利用率 ISPD2015 上的規模天花板 ~150k cells;總量級校準 α̂=0.652 / β̂=0.966(κ_ft 建議 1.482),per-net 級校準待 per-net 陣列落檔後補
- 前置報告:`docs/results/m3-differentiable-ft-report.md`(M3 exit FAIL——`ours@M3` 臂本報告一律不納入任何表格,只有 `flat`/`ours@M2`)
- **狀態:內容已定稿——E1/E3/E5 PASS,E2 PASS(依 T6B 替代路徑,非字面 holdout 全綠),E4 部分判定(GP 子模型已重擬合並誠實揭露 `identifiable=false`;evaluator/op 子模型未曾納入 T0b 範圍,見 §7)。全部章節已依現有 artifact 寫實。`scripts/m4_report_lint.py --strict` 於 **2026-08-19 實跑 `0 error(s), 0 warning(s)`**:原先的 24 error(§5.3/§5.4 引用的 24 個 profile run 缺 `workload_status`/`generator_verified`,driver 輸出 schema 缺口,非表格措辭問題)已依**使用者裁決 A(2026-08-19)**以可稽核的 backfill 補齊——只新增這兩個欄位與一則 `schema_backfill_note`,不讀取、不重算、不更動任何量測值(腳本 `scripts/m4_backfill_result_gate_fields.py`,逐檔 sha256 稽核紀錄 `results/m4/backfill/2026-08-19-result-gate-backfill.json`;附錄 A.4)。**M4-G8 因此了結,T11 標記完成。** 仍有極少數不影響內容判定的收尾項標 `<!-- PENDING: ... -->`,禁止以推測數字填補。

**品質宣稱紅線(design draft §3.4/§6.2;T6 holdout 裁決 §7-1):** 本報告中任何 `benchmark_kind="synthetic"` 的 case(6.2M/12.3M/27.7M 合成陣列)一律不得出現 HPWL/IO/FT 品質欄位或 winner/pareto 宣稱。凡合成表格涉及生成器未認證(`generator_verified=false`),依裁決 §7-1 必須逐字附上:

> 30M 的連通性是擬合出來的,不是校準出來的。

---

## 1. 摘要與 verdict 總表

M4 的核心命題(v1→v2.3 一路存活):**DREAMPlace GP 在 10M 級不是瓶頸,evaluator 記憶體與 IO op 的 K-pass runtime 才是**;而 v1 的推論方式(五點擬合外推、峰值相加判可行、cell-count ratio 當階層證據、10× 容忍帶當預測)全部被 Codex 兩輪對抗性審查與 T3a/T6 的實測推翻,v2.3 把每一個裁決換成「一個能證偽它的實驗 + 一條在證據不足時停下來的規則」。本報告是這條規則的結案 pass:**E1/E3/E5 三個判準已全綠;E2 綠燈但走的是 T6B 替代自洽性路徑,不是字面上的 holdout 全綠;E4 是部分判定——GP 記憶體/runtime 子模型已重擬合並誠實得出 `identifiable=false`,但 exit 文字要求的 evaluator/op 子模型從未被 T0b 的 factorial 設計觸及,此缺口逐字揭露而非代填。**

| # | 判準 | 狀態 | 一句話 |
|---|---|---|---|
| **E1** 10M 全流程 | **PASS** | `mempool_cluster` seed-B(seed=2000)六門檻全過:`Δhpwl` +1.44%≤5%、`num_unplaced_cells=0`、`final_overflow=0.0694≤0.07`、`legalization_status=success`、`device_used_gb=13.47≤20GB`、單臂 1.9h≤4h(`results/m4/profile/mempool_cluster__k16__grid__oursM2.json`);K∈{16,32}×{flat,ours@M2} 四格皆已落地,見 §3、§7 |
| **E2** 30M 測資 | **PASS(依 T6B 替代路徑)** | (a) 27.7M Bookshelf 產出 + T7 3×3 陣列 sha + B-5 重算全綠(T6B);(b) 12.3M 的 H1–H5 holdout 本身**仍 FAIL**(T6 裁決維持,非懸置)——依 M4-G3 轉入 T6B 的 B-0/B-3…B-6 自洽性驗收,全部綠燈,`scaling_usable=true`;(c) 27.7M 元件級 spike(T9)兩情境皆 `feasible_l4_contract`。三支子條件依裁決語意合併判 PASS,**不等於原始 H1–H5 全綠** |
| **E3** 全規模對照表 | **PASS** | `quality_real_cases` 格式的真實 case 表 16/16 格(4 case×K{16,32}×{flat,ours@M2})、`scaling_synthetic_cases` 格式的合成表 8/8 格(2 case×K{16,32}×{flat,ours@M2})皆已落地,seed B=2000,見 §5;`m4_report_lint.py --strict` 曾對這 24 格實跑 24 error(driver 輸出缺 `workload_status`/`generator_verified` 欄位,非表格寫法問題),已於 2026-08-19 依使用者裁決 A 以可稽核 backfill 補欄位後轉為 `0 error, 0 warning`,見 §9/附錄 A.4 |
| **E4** 記憶體 profile | **部分判定** | B1/B2/B3 三個量測工具 bug 已處置揭露(§2,B1 由 T1b 四臂 A/B probe 定案成因);T0b 16 點 factorial 設計**全部執行**,GP 記憶體/runtime 兩個子模型重擬合完畢、`identifiable=false`(M4-G6 觸發,誠實負結果);但 exit 文字要求的 evaluator/op 子模型**從未被 T0b 觸及**(`model_fit.json` 只有 `gp_memory_model`/`gp_runtime_model` 兩個 key),此缺口逐字記錄,不代填 → 見 §7 E4 |
| **E5** H100 forecast | **PASS(凍結)** | `results/m4/forecast/h100_prediction.json` 已凍結(`status="frozen"`),sha256 `4eeea5f9132e1a06585ae57e38c6e45b5523ed31ae5454826ff4cff56e045284` 登錄於 §8;T14 首跑比對器(`scripts/m4_check_prediction.py`)已含逐 phase `s_p_obs` 檢定邏輯 |

**一句話結論:** M4 v2.3 的誠實紀律站得住到底——iteration 預算調整是一次性的、方向不利於「ours」的全語料修正(§3),T6 的 FAIL 不因任何事後論證被翻案(§4),T6B 的自洽性驗收全綠(§4),E1/E3/E5 三個 exit 判準的**內容**已用 seed-B 正式資料補完(§5、§7)。**未完全兌現的缺口本報告逐字揭露,不以任何理由代填:**(1) E4 的 evaluator/op 子模型——T0b 的 factorial 重擬合只覆蓋了 GP 記憶體/runtime,exit 原文點名的另外兩個子模型從未被這輪 T0b 設計納入(§7 E4);(2)(**已於 2026-08-19 了結**)M4-G8——§5.3/§5.4 的 24 個正式 run 內容完整、規模覆蓋齊全,但其來源 JSON 缺少 linter 要求的 `workload_status`/`generator_verified` 欄位,`m4_report_lint.py --strict` 因此判 24 error;依使用者裁決 A 補記這兩個「後定義」欄位(純新增、不動量測值、逐檔 sha256 存證)後,linter 轉為 `0 error, 0 warning`,T11 標記完成(§9、附錄 A.4)。**此一了結不改變本報告任何一個數字或判定。**

---

## 2. 量測方法與 bug 揭露

沿用 spec §1.4 的四條(v2 修正其中兩條的論證)加上 T3a/T6 過程中新發現的第五條。**全部適用於本報告引用的每一個數字。**

### B1 — `peak_mem_mb` 同 process 污染(M2/M3 數字全部作廢)

`results/m2/**/*.json` 的 `peak_mem_mb` 受同 process 內先前臂污染:adaptec1 `A0_k8`=140.765625 MiB 與全新 process 的 GP 峰值逐位元相同,其後 k16/k32/slicing = 253.4/366.7/479.3 MiB(每步 +112.6 MiB 定值步進);bigblue4 A2/A3/A4/A6 = 10.58/12.46/14.35/16.23 GB(每步 +1.88 GB 定值步進)。`max_memory_allocated()` 是「歷史最大 active allocation」不會自動相加,v1 的「process 累積 high-water mark」是過度推論;更可能是每臂結束後仍存活的張量(retained tensors / 缺 teardown),**單靠 reset 不會清掉仍存活的 allocation**。

**處置:結論(污染成立、M2/M3 的 `peak_mem_mb` 一律作廢)保留;成因由 T1b 四臂 A/B probe 定案。** `results/m4/profile/t1b_arms.json`(adaptec1、100 iterations,四種 protocol 依序在同一 A/B 骨架上跑 arm1→arm2 兩臂):

| protocol | arm2 相對 arm1 baseline 的殘留(`arm2_baseline_delta_gb`) |
|---|---:|
| `fresh_subprocess`(每臂各自獨立 process) | **0.0**(無殘留) |
| `same_process`(同 process 依序跑兩臂) | **0.1106 GB** |
| `reset_only`(同 process,phase 開頭僅 `reset_peak_memory_stats()`) | **0.1106 GB**(與 same_process 逐位元相同——`reset_only_fixes_baseline=false`) |
| `teardown_gc`(同 process,顯式釋放張量 + `gc.collect()`) | **0.0**(`teardown_within_2pct_tolerance=true`) |

`judgment.cause = "retained_tensors"`:`reset_only` 不能讓殘留歸零(排除「純粹 high-water mark 累積」的解釋),但顯式 teardown+GC **可以**——這確認 B1 的污染成因是**臂與臂之間仍存活的張量**(而非 allocator 統計本身的性質),與 §1.4 B1 原文「未證」的猜測方向一致,現正式定案。本報告的 GPU 峰值數字(§5、§6、§7)一律取自**每次 run 各自獨立 process** 產出的 `device_used_gb`/`peak_mem_mb`(schema v3,`peak_mem_mb_reset_semantics=true`),不引用任何 M2/M3 的舊 `peak_mem_mb`——這正是 `fresh_subprocess` 這一臂本身已經證明乾淨的量測協定。

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

**3×3 N1 = `not_run`(誠實記錄,§4.1 已否證的配方不予落地):** `build_summary.json` 的 `"3x3"."n1"` 欄位記錄了一組**計算出的** glue 統計(`n_nets=151,491`、`n_pins=302,982`,以 N1 的 `Σλ₀φ(d)` 公式算出),但 `results/m4/bench/arrays/` 下**沒有 `3x3_n1` 目錄**——這組統計從未被實際建構成陣列檔案。理由與 §4.1 一致:N1 已被 H6 的形狀不變腿(3×3 vs 2×2)證偽(`|Δp|` 隨陣列規模漂移),T7 的定案配方(上表)自始就只落地 N2 的 K1/K3;`build_summary.json` 裡的這組數字是規劃階段的副產物,不代表 N1 有任何 3×3 級的實測或已建構產物,本報告不引用它做任何比較。

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

**B-1(3×3 Rent 形狀不變性 DiD)——`ok`,已執行:** `rent.measure_rent` 在 `3x3_n2`/`3x3_noglue`/`2x2_noglue` 三個陣列(+ 沿用 T6 既有的 2×2 N2 量測)上各跑一次,`3x3_n2`/`3x3_noglue`/`2x2_noglue` 分別耗時 2843.0 s / 2882.0 s / 1183.6 s(全部遠低於預先登錄的 4h wall / 100GB RSS 中止上限,未觸發任何中止條件)。DiD 定義為兩層「加 glue 前後的 Rent p 變化量」之差:`Δp(3×3) = p(3x3_n2)−p(3x3_noglue) = 0.53909−0.47995 = 0.05914`;`Δp(2×2) = p(2x2_n2)−p(2x2_noglue) = 0.52626−0.47052 = 0.05575`;**`DiD = |0.05914−0.05575| = 0.003398 ≤ 0.03`(門檻)⇒ B-1 綠燈**——3×3 加 glue 造成的 Rent 躍升幅度與 2×2 幾乎相同,形狀不隨陣列規模系統性漂移。**揭露(不影響判定)**:`p(3x3_n2)=0.53909` 本身落在預先登錄的預測帶 `[0.522, 0.538]` 之外,超出上緣 **0.0011**;此為一個登錄在先的 forecast 落空,不是 B-1 判準(`DiD≤0.03`)本身,依 2026-08-15 裁決 §4.3 逐字揭露、不改判。

**B-2(H5 等解析度 K-grid λ)——`fail`,已執行,published/non-gating:** 合成 2×2(K=16,`synthetic_2x2_n2__k16__grid__flat.json`,seed B)`hard_lambda_sum=197,793`(`n_nets=14,042,293`)⇒ `λ_synthetic=0.014086`;真實 `mempool_group` 在**等解析度**(K=4,2×2 grid,`group__k4__grid__flat_eval_b2.json`,對 seed-A flat npz 的 evaluate-only 重新計分)`hard_lambda_sum=31,191`(`n_nets=3,503,988`)⇒ `λ_real=0.008902`;**`ratio = λ_synthetic/λ_real = 1.5824`,落在 `[0.7,1.4]` 門檻之外 ⇒ FAIL**,依裁決發表但**不 gate** scaling 用途(`adjudication_note`:「fail is published and disclosed, does NOT gate scaling usage」)。**反向診斷(揭露,不判定)**:若改用兩邊皆 K=16(解析度不等,合成的 per-tile 解析度只有真實的 1/4)算出的 `ratio=0.6360`,方向由 >1.4 翻轉為 <0.7——與 §4.4 節「解析度效應」的預先登錄預測方向一致,佐證 B-2 fail 的根源是「等解析度」定義本身的敏感度,而非生成器隨機失準。

依裁決 §4.1 的驗收規則(**B-0 與 B-3…B-6 全綠 ⇒ 3×3 bench 可用於 scaling**),`verify_group3x3_t6b.json` 記 **`scaling_usable=true`**,`generator_verified=false`,`quality_claims_prohibited=true`。B-1/B-2 屬**非 gating 的附加揭露項**(裁決 §4.1 的驗收規則本身只點名 B-0/B-3…B-6),兩者皆已執行完畢,不再是 `pending_measurement`;B-1 綠燈進一步支持 3×3 陣列的結構相似性,B-2 fail 已如實發表且不推翻 `scaling_usable=true`。

### 4.5 V0′ 判準修正(§7-5,逐字)

> `verify_bench.check_v0_structural` 對真實衍生網表有三條必然誤判的子檢查,2×2 的 1,338 條 error 全部出自它們,**沒有一條源自 tiler**:(a)「self-loop」693,952 條 = 來源 `mempool_group` 自身 173,488 條(占其 nets 的 4.951%)× 4,一條 net 接到同一 cell 的兩隻腳在真實網表中合法;(b) row「縫隙」出現在 macro 覆蓋處,含 macro 的 floorplan 在該處本就不該有 placement row(`tile_bookshelf.py:259-276` 已於建構期記載此重新詮釋,tiler 自身的 assert 早已改為只查 overlap);(c) 8 個未被任何 net 參照的 node = 來源 2 個 × 4。三條均改為「不劣於來源」的相對判準後重跑。V0 依 2026-08-14 裁決為診斷不判定項。

### 4.6 核不可識別性(§7-6,逐字)

> 距離核的形狀在 2×2 上有 0 個 lack-of-fit 自由度(§3.2),因此 3×3 的核是一個**宣告的假設**。在 N2 正規化之下,核的選擇**完全不改變 glue 總數**(每 tile 端子預算固定,核只重新分配),只改變空間局部性:d=1 pair 所佔比例在 K1(α=0.102)為 34.6%、K2(指數,α′=0.0856)34.8%、K1(α=1.915)58.9%、K3(截斷,不連 d>√2)74.3%。因此 §3.2 line 304 的「三核總數全距 > 20% 才附 K3 變體」在 N2 下恆為 0%,已退化為無資訊的觸發條件;本報告改以距離層級分佈報告系統性不確定度,並無條件附上 K3 全陣列作為「不模擬 long-range 連通性」的明示下界。

### 4.7 per-pair 抽樣的 Poisson 適合度(揭露,不判定,不採取行動)

`results/m4/bench/glue_poisson_gof.json`(附錄 A.5):彙總 5 個已建陣列、50 對符合 `min_expected≥42` 的 pair,實現值相對期望值 **33 正 / 17 負 / 0 平**,雙尾精確二項符號檢定 **p=0.0328**(弱顯著);但 Poisson 適合度檢定(逐 pair 變異數結構)**卡方=52.34,df=50,p=0.3834**——不支持存在系統性生成偏誤。3×3_n2 單獨一項:36 對中 21 正/15 負,`chi2_p=0.554`。**結論:僅揭露、不判定、不採取行動**(依附錄 A.5 政策,此發現不影響 §4.4 的任何綠燈);觸發的後續調查(`sample_glue_net_count_from_expected` 的 per-pair 偏正複查)留給下一輪。

---

## 5. T8b 校準與 seed-B 正式報表

§5.1–§5.2 是 **T8b 校準表(seed A),不是品質表**:下列格點用於**依預先登錄規則選出 `ρ*`**,不是本報告任何 E3 判準所引用的正式結果;報告用的正式 run 依 T8b 協定一律跑 **seed B**(附錄 A.3),此處 seed A 的 `run_id` 不得出現在任何最終品質表中——最終品質/合成表見 §5.3/§5.4(seed B,已落地)。

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

**已完成:** K=32 的獨立 GP+LG `ours@M2` 臂(`results/m4/profile/mempool_cluster__k32__grid__oursM2.json`,`final_overflow=0.0697`、`device_used_gb=14.13GB`,非 evaluate-only)與 cluster 的正式 seed-B(seed=2000)report 矩陣(K∈{16,32}×{flat,ours@M2} 四格)已補齊,見 §5.3 正式表格。上表(本節)維持原樣、不回填——它是 **seed A**(seed=1000)的校準/篩選性質資料,`ρ*` 選取的紀錄本身不需要、也不應該用 seed-B 資料覆寫。

### 5.3 品質對照表(`quality_real_cases`,seed B,16/16 格)

`results/m4/profile/{case}__k{16,32}__grid__{flat,oursM2}.json`(`seed=2000`,附錄 A.3 鎖定的正式 seed B):四個真實 case(superblue12/bigblue4/mempool_group/mempool_cluster)在 K∈{16,32}×{flat, ours@M2(ρ_max=ρ\*=0.20)} 下的完整矩陣已全數落地(16/16 格),對應 E3(§7)。欄名採 spec §6.2 `quality_real_cases.md` 原欄名(`case | cells | nets | pins | K | rtype | arm | io_mst | ft_mst | io_rg | ft_rg | hpwl | Δio% | Δft% | Δhpwl% | num_unplaced_cells | final_overflow | t_total | t_gp | t_op | t_eval | gpu_peak | host_hwm | run_id`)。

| case | cells | nets | pins | K | rtype | arm | io_mst | ft_mst | io_rg | ft_rg | hpwl | Δio% | Δft% | Δhpwl% | num_unplaced_cells | final_overflow | t_total | t_gp | t_op | t_eval | gpu_peak | host_hwm | run_id |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| sb12(1.3M) | 1,286,948 | 1,293,413 | 4,772,990 | 16 | grid | flat | 62,055 | 3,284 | — | — | 260,669,447 | — | — | — | 0 | 0.0643 | 340.2 | 75.2 | — | 236.0 | 3.09 | 6.53 | `results/m4/profile/sb12__k16__grid__flat.json` |
| sb12(1.3M) | 1,286,948 | 1,293,413 | 4,772,990 | 16 | grid | ours@M2 | 45,125 | 4,190 | — | — | 267,797,173 | −27.28% | +27.59% | +2.73% | 0 | 0.0664 | 514.1 | 256.3 | — | 228.9 | 2.80 | 6.54 | `results/m4/profile/sb12__k16__grid__oursM2.json` |
| sb12(1.3M) | 1,286,948 | 1,293,413 | 4,772,990 | 32 | grid | flat† | 106,059 | 9,324 | — | — | 260,669,447 | — | — | — | — | — | 266.9 | — | — | — | — | — | `results/m4/profile/sb12__k32__grid__flat.json` |
| sb12(1.3M) | 1,286,948 | 1,293,413 | 4,772,990 | 32 | grid | ours@M2 | 80,696 | 11,013 | — | — | 265,249,308 | −23.91%‡ | +18.11%‡ | +1.76%‡ | 0 | 0.0631 | 765.5 | 491.5 | — | 245.9 | 4.43 | 6.55 | `results/m4/profile/sb12__k32__grid__oursM2.json` |
| bb4(2.2M) | 2,169,183 | 2,229,886 | 8,900,078 | 16 | grid | flat | 99,912 | 6,969 | — | — | 748,432,433 | — | — | — | 0 | 0.0646 | 572.4 | 56.2 | — | 466.5 | 4.06 | 10.22 | `results/m4/profile/bb4__k16__grid__flat.json` |
| bb4(2.2M) | 2,169,183 | 2,229,886 | 8,900,078 | 16 | grid | ours@M2 | 64,753 | 8,873 | — | — | 782,633,935 | −35.19% | +27.32% | +4.57% | 0 | 0.0697 | 950.0 | 437.1 | — | 464.6 | 4.06 | 10.25 | `results/m4/profile/bb4__k16__grid__oursM2.json` |
| bb4(2.2M) | 2,169,183 | 2,229,886 | 8,900,078 | 32 | grid | flat† | 159,947 | 19,193 | — | — | 748,432,433 | — | — | — | — | — | 524.2 | — | — | — | — | — | `results/m4/profile/bb4__k32__grid__flat.json` |
| bb4(2.2M) | 2,169,183 | 2,229,886 | 8,900,078 | 32 | grid | ours@M2 | 118,373 | 21,842 | — | — | 763,011,690 | −25.99%‡ | +13.80%‡ | +1.95%‡ | 0 | 0.0693 | 1262.0 | 739.8 | — | 472.3 | 4.73 | 10.24 | `results/m4/profile/bb4__k32__grid__oursM2.json` |
| group(3.1M) | 3,077,669 | 3,503,992 | 12,026,191 | 16 | grid | flat | 97,803 | 9,170 | — | — | 485,777,937 | — | — | — | 0 | 0.0698 | 767.2 | 78.9 | — | 585.7 | 4.01 | 18.51 | `results/m4/profile/group__k16__grid__flat.json` |
| group(3.1M) | 3,077,669 | 3,503,992 | 12,026,191 | 16 | grid | ours@M2 | 89,634 | 10,843 | — | — | 491,191,451 | −8.35% | +18.24% | +1.11% | 0 | 0.0699 | 1363.9 | 686.0 | — | 583.5 | 4.01 | 18.55 | `results/m4/profile/group__k16__grid__oursM2.json` |
| group(3.1M) | 3,077,669 | 3,503,992 | 12,026,191 | 32 | grid | flat† | 162,479 | 25,472 | — | — | 485,777,937 | — | — | — | — | — | 661.3 | — | — | — | — | — | `results/m4/profile/group__k32__grid__flat.json` |
| group(3.1M) | 3,077,669 | 3,503,992 | 12,026,191 | 32 | grid | ours@M2 | 143,783 | 27,126 | — | — | 489,615,430 | −11.51%‡ | +6.49%‡ | +0.79%‡ | 0 | 0.0700 | 1982.9 | 1286.3 | — | 598.4 | 6.85 | 18.61 | `results/m4/profile/group__k32__grid__oursM2.json` |
| cluster(11.3M) | 11,310,807 | 12,712,800 | 43,947,793 | 16 | grid | flat | 221,472 | 11,220 | — | — | 2,095,126,495 | — | — | — | 0 | 0.0698 | 2989.2 | 374.3 | — | 2155.0 | 12.23 | 68.46 | `results/m4/profile/mempool_cluster__k16__grid__flat.json` |
| cluster(11.3M) | 11,310,807 | 12,712,800 | 43,947,793 | 16 | grid | ours@M2 | 189,663 | 17,910 | — | — | 2,125,310,176 | −14.36% | +59.63% | +1.44% | 0 | 0.0694 | 6808.9 | 4260.1 | — | 2123.7 | 13.47 | 68.61 | `results/m4/profile/mempool_cluster__k16__grid__oursM2.json` |
| cluster(11.3M) | 11,310,807 | 12,712,800 | 43,947,793 | 32 | grid | flat† | 355,720 | 37,053 | — | — | 2,095,126,495 | — | — | — | — | — | 2508.2 | — | — | — | — | — | `results/m4/profile/mempool_cluster__k32__grid__flat.json` |
| cluster(11.3M) | 11,310,807 | 12,712,800 | 43,947,793 | 32 | grid | ours@M2 | 314,949 | 46,627 | — | — | 2,118,946,353 | −11.46%‡ | +25.84%‡ | +1.14%‡ | 0 | 0.0697 | 10597.7 | 8045.1 | — | 2133.8 | 14.13 | 68.49 | `results/m4/profile/mempool_cluster__k32__grid__oursM2.json` |

`cells`=corpus `num_movable`;`nets`/`pins`=corpus `num_nets`/`num_pins`(`results/m4/corpus/*.json`,bb4 取自 DREAMPlace 內建 `bigblue4.nodes`/`bigblue4.nets` 的 `NumNodes−NumTerminals`/`NumNets`/`NumPins`)。`io_mst`/`ft_mst` = evaluator 主判準(JSON 的 `io_count`/`ft_count`);`io_rg`/`ft_rg` 全表留空——本表全部 16 個 run 都是 `flat`/`ours@M2`,`ours@M3` 專屬欄位依前言紅線不納入任何一列。`t_op` 留空:本 schema 版本沒有把 IoTerm fwd/bwd 從 `t_gp`(GP 迴圈內呼叫)獨立計時出來,`t_gp` 已含 op 成本。

†:K=32 的 `flat` 是對同案 K=16 `flat` placement 的 **evaluate-only** 重新計分(`mode="evaluate_only"`,重用 K16 flat 的 `.npz`),不是獨立收斂的 K=32 GP+LG 臂——`hpwl` 因此與同案 K=16 `flat` 逐位元相同;`final_overflow`/`t_gp`/`t_eval`/`gpu_peak`/`host_hwm` 留空,因為 evaluate-only run 沒有這些欄位。
‡:K=32 的 `Δio%`/`Δft%`/`Δhpwl%` 是相對於同案**同 K** 的 `flat†`(evaluate-only 重新計分基準),不是相對 K=16 flat;K=32 `ours@M2` 本身是獨立收斂的 GP+LG 臂(`final_overflow`/`gpu_peak` 等欄位齊全,非 evaluate-only)。

`ours@M2` 一律 `ρ_max=0.20`(= §5.1/§5.2 選定的各案 ρ\*)。本表已走過 §7.0 RESULT GATE / linter 的正式收錄流程(附錄 A.4 記錄實跑結果)。

### 5.4 Scaling 對照表(`scaling_synthetic_cases`,seed B,8/8 格)

`results/m4/profile/synthetic_{1x2,2x2}_n2__k{16,32}__grid__{flat,oursM2}.json`(`seed=2000`):兩個合成 case(1×2=6.2M、2×2=12.3M)在 K∈{16,32}×{flat, ours@M2(ρ_max=0.20)} 下的完整矩陣已全數落地(8/8 格),對應 E3(§7)。欄名採 spec §6.2 `scaling_synthetic_cases.md` 原欄名(`case | cells | nets(+glue) | pins(+glue) | K | arm | status | t_total | t_gp | t_op | t_eval | iter_ms_p50 | gpu_peak | host_hwm | run_id`)——**schema 明文禁止品質欄位:本表無 `Δio%`/`Δft%`/`Δhpwl%`/`winner`/`pareto*`,亦不含任何 `hpwl`/`io_count`/`ft_count` 欄位。**

| case | cells | nets(+glue) | pins(+glue) | K | arm | status | t_total | t_gp | t_op | t_eval | iter_ms_p50 | gpu_peak | host_hwm | run_id |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|---:|---:|---|
| 1×2(6.2M) | 6,155,338 | 7,021,183 | 24,078,796 | 16 | flat | ok | 1506.5 | 139.5 | — | 1161.3 | — | 7.35 | 40.31 | `results/m4/profile/synthetic_1x2_n2__k16__grid__flat.json` |
| 1×2(6.2M) | 6,155,338 | 7,021,183 | 24,078,796 | 16 | ours@M2 | ok | 3326.1 | 1911.2 | — | 1197.9 | — | 7.35 | 40.47 | `results/m4/profile/synthetic_1x2_n2__k16__grid__oursM2.json` |
| 1×2(6.2M) | 6,155,338 | 7,021,183 | 24,078,796 | 32 | flat† | ok(evaluate_only) | 1351.6 | — | — | — | — | — | — | `results/m4/profile/synthetic_1x2_n2__k32__grid__flat.json` |
| 1×2(6.2M) | 6,155,338 | 7,021,183 | 24,078,796 | 32 | ours@M2 | ok | 5145.6 | 3681.8 | — | 1244.5 | — | 10.44 | 40.36 | `results/m4/profile/synthetic_1x2_n2__k32__grid__oursM2.json` |
| 2×2(12.3M) | 12,310,676 | 14,042,293 | 48,157,446 | 16 | flat | ok | 3143.6 | 296.8 | — | 2322.8 | — | 13.37 | 81.62 | `results/m4/profile/synthetic_2x2_n2__k16__grid__flat.json` |
| 2×2(12.3M) | 12,310,676 | 14,042,293 | 48,157,446 | 16 | ours@M2 | ok | 7592.0 | 4809.8 | — | 2297.1 | — | 14.38 | 81.77 | `results/m4/profile/synthetic_2x2_n2__k16__grid__oursM2.json` |
| 2×2(12.3M) | 12,310,676 | 14,042,293 | 48,157,446 | 32 | flat† | ok(evaluate_only) | 2716.1 | — | — | — | — | — | — | `results/m4/profile/synthetic_2x2_n2__k32__grid__flat.json` |
| 2×2(12.3M) | 12,310,676 | 14,042,293 | 48,157,446 | 32 | ours@M2 | ok | 11947.5 | 9157.6 | — | 2304.6 | — | 15.90 | 81.81 | `results/m4/profile/synthetic_2x2_n2__k32__grid__oursM2.json` |

`cells`/`nets(+glue)`/`pins(+glue)` 取自 `results/m4/scaling/count_freeze_small.json`(N2 實際 manifest 計數,§6.1):`cells` = 對應 group 數(1×2=2、2×2=4)× `mempool_group` 的 `num_movable`;`nets(+glue)`/`pins(+glue)` = `base_n_nets`/`base_n_pins` + N2 的實際 glue `g_n2`/`gp_n2`。`t_op` 留空,理由同 §5.3(schema 未把 op 從 `t_gp` 拆出)。`iter_ms_p50` 留空:此 schema 版本只在每 50 iteration 記一次 checkpoint(`trajectory[]`,含 overflow/io_count/ft_count 等),不輸出 `iter_ms[]` 逐 iteration 陣列,故沒有可據以算 p50 的原始樣本;`t_read`(GP 之前的 Bookshelf 讀取階段,不在 spec §6.2 schema 內,但 T7b 的核心發現引用它)另行記錄:1×2 四格的 `t_read` 為 171.7/168.9/—(evaluate-only 無此欄位)/171.2 s;2×2 四格為 411.4/368.0/—/370.0 s。

†:K=32 的 `flat` 是對同案 K=16 `flat` 的 evaluate-only 重新計分(`mode="evaluate_only"`),語意同 §5.3 的 flat†。

<!-- 揭露(2026-08-19 更新):§5.4 的 8 個 profile JSON 已補上 `generator_verified=false`(裁決 A 的 backfill,附錄 A.4);但其 `benchmark_kind` 欄位實際值**仍是** `"real"` 而非 `"synthetic"`(driver 呼叫時未帶 `--benchmark-kind synthetic` 旗標),本次 backfill **不修改既有欄位值**故未動它——這是尚未了結的 artifact-schema 缺陷,已知且不 gating(linter rule 2 只單向禁止 quality 表引用 synthetic run)。本表本身不因此欄位缺陷而改變其「不含品質欄位」的誠實承諾。 -->

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

### 6.6 T2b:元件級記憶體生命週期(3.1M 完成、11.3M 誠實負結果)

**3.1M(`mempool_group`,K=32,ours@M2)——完整落地:** `results/m4/profile/lifetime_group__k32__ours_m2.json` 對每個張量緩衝的生命週期做逐 phase 追蹤,`totals` 欄彙總:

| 量 | 值(GB) |
|---|---:|
| `peak_alloc_gb_run`(整個 run 的 PyTorch allocator 峰值) | 4.0118 |
| `peak_reserved_gb_run` | 6.0391 |
| `device_used_gb_run`(`mem_get_info` 量到的整卡峰值) | 6.0781 |
| `resident_max_gb`(逐 buffer 加總的常駐上界) | 1.2962 |
| `max_phase_transient_gb`(單一 phase 內的暫態峰值) | 2.8279 |
| `identity_residual_gb`(`resident_max` 與獨立算出的 `standalone_upper_bound` 之差) | −0.1123 |
| `standalone_upper_bound_gb` | 1.2962 |
| `upper_bound_over_measured`(上界相對 `peak_alloc_gb_run` 的鬆緊比) | 0.3231 |

`identity_residual_gb` 接近 0(−0.11 GB,相對 `peak_alloc_gb_run` 4.01 GB 約 2.8%)——兩種獨立算法(逐 buffer 累加 vs 解析上界公式)在 3.1M 這一級互相印證,不是巧合對齊。

**11.3M(`mempool_cluster`,K=32)——`not_evaluable_scale`,誠實負結果:** `results/m4/profile/lifetime_cluster__k32__ours_m2.json` 記 `experiment_status="not_evaluable_scale"`、`workload_status="timeout_or_oom_killed"`、`feasibility_verdict="not_applicable_see_alternatives"`。六次嘗試逐一記錄:

| # | recipe | 結果 |
|---|---|---|
| 1 | 300 iters / diag-every 2(比照 group 的成功配方) | 4h timeout;根因 1:`scan_cuda_tensors` 對 `PlaceDB` name2id 字典的 O(n log n) 掃描(已於 `236e91d` 修) |
| 2 | 60/10,修前舊碼 | 過期程式碼跑,被排程器殺掉 |
| 3 | 60/10,經 chain 呼叫 | 被 gate race 污染(CPU 讀取階段時 GPU-mem gate 誤開);T9 preflight 攔下 |
| 4 | 60/10,scan 修好、4h | 卡在 LG phase 逾時——GP + 6 個 mid eval 已完成,尾段耗盡預算 |
| 5 | 60/10,6h | 被 host OOM killer SIGKILL(並發 route shard 排程失誤,非本 workload 之過) |
| 6 | 60/10,6h,RAM 已釋放(sb12 shard 暫停) | LG 之後的 final_eval 逾時——同一尾段瓶頸 |

瓶頸診斷(`partial_evidence`):GP phase 與 LG phase 皆已完成;真正的瓶頸是 streaming evaluator 在 12.7M nets、K=32 下的單執行緒 CPU 段(~35 min/call)×(6 次 mid eval + 1 次 final)+ 讀取 ~330s + GP,`host_rss_observed_gb=67.7`。依 T6B/M4-L4 的預先登錄中止語意(§4.4 引用的同一條規則):**超出每 run 上限記為 `not_evaluable_scale`,不得改判為 fail,也不得因此降級估計器**——3.1M 的逐 buffer 記錄已滿足 T2b「per-buffer attribution」的目標,11.3M 這一級的可行性判定不落在這支探針身上。

**§2.2 可行性歸屬(design draft §2.2「加總帳=保守上界,不是可行性判定」的具體落實):** `lifetime_cluster__k32__ours_m2.json` 的 `sec22_feasibility_not_blocked_by_this` 欄明寫其判定路徑——11.3M 的實際可行性由另外兩個獨立來源共同決定,不依賴這支未完成的探針:(1) **整合 run 證據**:§3/§5.3 的 `mempool_cluster__k16__grid__oursM2.json`(E1 seed-B 六門檻全過,`device_used_gb=13.47GB≤20GB`,完整 GP+LG+eval 生命週期);(2) **元件共常駐證據**:§6.7 的 T9 27.7M spike(`feasible_l4_contract`,峰值 18.19/18.63 GB)。T2b 探針本身只負責「逐 buffer 歸因」這個更精細的問題,3.1M 已回答;11.3M 的粗粒度可行性問題已由 (1)(2) 兩條獨立路徑各自作答完畢。

### 6.7 T9:27.7M 元件級共常駐 spike——兩情境皆 `feasible_l4_contract`

`results/m4/bench/t9_cache_3x3` 快取的 3×3 陣列上,`spike_30m.py` 讓 IoTerm(K=32)與 evaluator 在同一 process 內共常駐,依 §2.2 三態規則判定:

| scenario | 內容 | `measured_peak_gb` | `budget_gb` | `resident_gb` | `max_phase_transient_gb` | `feasibility_verdict` | `workload_status` |
|---|---|---:|---:|---:|---:|---|---|
| A | 基準情境 | 18.188 | 19.5 | 3.102 | 15.086 | `feasible_l4_contract` | `completed` |
| B(`B_emulated_ballast`) | 模擬額外 ballast 佔用的情境 | 18.630 | 19.5 | 3.542 | 15.089 | `feasible_l4_contract` | `completed` |

兩情境 `measured_peak_gb` 皆 < 19.5 GB(L4 契約)、`oom_repeats=0`、`exclusivity_evidence="compute_apps_pid_confirmed"`(獨佔 GPU 已驗證)。**A/B 差距 0.44 GB(18.630−18.188)**——支持 §8 forecast「情境敏感度對 27.7M 這一級記憶體結論不敏感」的宣稱:即便刻意加大情境間的 ballast/佈局差異,峰值變化仍在半個 GB 量級,遠小於 19.5 GB 契約與 26.1–45.2 GB 的 H100 解析界之間的裕度。

**gate-race invalid attempt 揭露:** `results/m4/profile/invalid_attempts/spike30m__{A,B}__k32.attempt1_contaminated_by_gate_race.json` 記錄兩情境各自的第一次嘗試——`experiment_status="contaminated"`、`feasibility_verdict="invalid_measurement"`、`workload_status="crashed"`。成因與 §6.6 T2b 第 3 次嘗試相同:CPU 讀取階段時 GPU-mem gate 被提早打開,量到的不是預期的共常駐峰值。依三態規則,`invalid_measurement` 允許重試(上限 2 次);兩次重試(本節主表的 A/B)皆乾淨完成,不需要再進一步重試或降級為 `blocked_external`。

---

## 7. E1–E5 逐條

### E1(10M 全流程)—— **PASS**

`mempool_cluster` overflow 判準的 FAIL→PASS 轉變(§3)——`final_overflow` 由 1000-iter 上限下的 0.327(`experiment_status="superseded_iteration_capped"`,run_id `9dec7b71-af7c-4d62-b3cd-cc76aba244df`)轉為 2000-iter 下的 0.0698。**六門檻的正式驗收改用 seed-B(seed=2000)的 §5.3 矩陣**,以 `ours@M2`(K=16、ρ_max=0.20)一列作為代表:

| arm | K | `hpwl` | `final_overflow` | `legalization_status` | `num_unplaced_cells` | `device_used_gb` | `runtime_h` | `run_id` |
|---|---:|---:|---:|---|---:|---:|---:|---|
| flat | 16 | 2,095,126,495 | 0.0698 | success | 0 | 12.23 | 0.83 | `d9951c9b-afac-45af-829d-588359a15929` |
| **ours@M2(ρ_max=0.20)** | 16 | 2,125,310,176 | **0.0694** | **success** | **0** | **13.47** | **1.89** | `31a22bc6-193b-4cfa-a222-a368cfb897a2` |
| flat(evaluate-only) | 32 | 2,095,126,495(同上,逐位元) | — | — | — | — | — | `0d95be56-de16-4714-b3c8-5732c8a833b9` |
| ours@M2(ρ_max=0.20) | 32 | 2,118,946,353 | 0.0697 | success | 0 | 14.13 | 2.94 | `924dec18-f33c-471f-ae17-7ead73fb1c10` |

六個 E1 分項在 `ours@M2`(K=16)這一格上**全部滿足**:`Δhpwl` 相對自身 flat = **+1.44% ≤ 5%**;`num_unplaced_cells = 0`;`final_overflow = 0.0694 ≤ 0.07`;`legalization_status = "success"`;`device_used_gb = 13.47 ≤ 20 GB`;單臂 wall-time = **1.89 h ≤ 4 h**。K=32 的 `ours@M2` 亦獨立收斂且六項全過(`final_overflow=0.0697`、`device_used_gb=14.13GB`、2.94h)。spec §6.3 E1 要求「K∈{16,32} 各一組 `flat` 與一組 `ours@M2`」——四格已全部落地(§5.3),且全部經過 §7.0 RESULT GATE(`status="ok"`,provenance 齊全)。`quality_real_cases` 正式表格條目已建立(§5.3)。⇒ **E1 = PASS**。

### E2(30M 測資)—— **PASS(依 T6B 替代路徑,非字面 holdout 全綠)**

三個子條件逐一交代:

- (a) 27.7M Bookshelf 產出、manifest sha256、實際 glue 計數:**已達成**(§4.4、§6.2,`count_freeze_30m.json`;T7 3×3 陣列 `results/m4/bench/arrays/3x3_n2/*.manifest.json` 落地,B-5 重算全綠,§4.4)。
- (b) 12.3M 的 V0/V1 + H1–H5 holdout 全綠:**字面上未達成,已判定為 FAIL 而非懸置**(§4.1,T6 holdout FAIL 維持,H1 絕對腿與 H6 兩腿不過)——依 M4-G3,已轉入 T6B 的替代自洽性驗收路徑,**B-0/B-3/B-4/B-5/B-6 全部綠燈**,`verify_group3x3_t6b.json` 記 `scaling_usable=true`(§4.4);spec §6.3 E2(b) 字面要求的「holdout 全綠」本身確實沒有發生,此處誠實記錄、不代填、不改判 T6 的 FAIL。
- (c) 27.7M 元件級 spike(T9)在 §2.2 三態規則下的有效終態:**已執行**,兩情境皆 `feasible_l4_contract`(峰值 18.19/18.63 GB,§6.7),含 `resident/transient/device_used` 分解。

⇒ (a)(c) 字面達成,(b) 字面 FAIL 但依 M4-G3 的裁決語意轉入的替代路徑全綠。**依 T6B 裁決文件(`docs/results/2026-08-15-m4-t6-holdout-adjudication.md`)的驗收規則,「生成器未通過驗證 ⇒ 轉 T6B ⇒ B-0/B-3…B-6 全綠 ⇒ scaling 可用」本身構成一條完整的替代驗收鏈**,三支子條件合併判 **E2 = PASS**——但這是**經由裁決語意合併的 PASS,不是原始 spec §6.3 字面「H1–H5 holdout 全綠」的 PASS**,兩者不可混淆引用。

### E3(全規模對照表)—— **PASS**

`quality_real_cases`(真實 case × K{16,32} × {flat, ours@M2},§5.3)16/16 格、`scaling_synthetic_cases`(合成 case × K{16,32} × {flat, ours@M2},§5.4)8/8 格,兩份正式報表皆已用 T8b 之後的 **seed-B**(seed=2000)T8 run 落地,每格皆可追到唯一 `run_id`,對應 spec §6.3 E3「{1.3M, 2.2M, 3.1M, 11.3M} × K{16,32} × {flat, ours@M2}」與「{6.2M, 12.3M} × K{16,32}」的規模覆蓋要求。**Linter 執行結果見附錄 A.4**——E3 判定本身以資料完整性與規模覆蓋為準,linter 曾揭露的欄位缺口已於 2026-08-19 依裁決 A 的可稽核 backfill 補齊,`--strict` 轉為 `0 error, 0 warning`,M4-G8 了結(§9、附錄 A.4)。

### E4(記憶體 profile)—— **部分判定**

B1/B2/B3 三個量測工具 bug 已處置揭露(§2:B1 由 T1b 四臂 A/B probe 定案成因為 `retained_tensors`;B2 的 `--diag-every`/`--no-diag` 已落地並改為實測 callback wall-time 驗收;B3 的 `--budget-gb` 參數化 + `result_gate.assert_budget()` 硬體契約 assert 已落地,§6.7/§9 的 T9 spike 即依此契約判定);T2 evaluator streaming acceptance 已過(§6.3)。

但 E4 exit 文字要求的是**三個模型**(GP、evaluator、op/IoTerm)各自以 T0b 的 factorial 設計重新擬合,附 `model_fit.json` 的 design matrix、condition number、係數 95% CI、LOO 殘差、以及 6.2M/12.3M holdout 殘差。T0b 的 16 點設計**全部執行完畢、無一跳過**(§6.5),但這 16 點的 factorial 設計範圍**只涵蓋 GP 記憶體與 GP runtime 兩個子模型**(`results/m4/scaling/model_fit.json` 的頂層 key 只有 `gp_memory_model`/`gp_runtime_model`)——兩者皆重擬合完畢,`identifiable=false`(觸發 M4-G6,§6.5 已完整記錄其誠實負結果與三條後果)。**exit 文字點名的另外兩個子模型(evaluator、op)從未被這輪 T0b 設計觸及**:`§1.2` 的 evaluator 記憶體模型(`mem ≈ (0.85+0.061K) GB/M-net`)與 `§1.3` 的 IoTerm 模型都停留在 spec 原文的舊擬合點上,沒有 T0b 式的 factorial 重擬合、CI 或 holdout 殘差;T2 的 acceptance 測試(§6.3)驗證的是「streaming 改造後峰值符合門檻」,不是「模型係數以新設計重新估計」,兩者性質不同,不能互相替代。

⇒ **E4 = 部分判定**:GP 子模型的重擬合 + 揭露義務已完整履行(誠實負結果,依 M4-G6 處置);evaluator/op 子模型的重擬合義務**未履行,亦未被本輪 T0b 設計規劃涵蓋**,此缺口逐字記錄,不以「T2 acceptance 已過」代為滿足,也不因為 GP 子模型已誠實地判定不可辨識就推定其餘子模型同樣可以省略。

### E5(H100 forecast)—— **PASS(凍結)**

`results/m4/forecast/h100_prediction.json` 已從草稿凍結:`status="frozen"`(草稿版 `status="draft"`/`unvalidated=true` 已被取代),sha256 `4eeea5f9132e1a06585ae57e38c6e45b5523ed31ae5454826ff4cff56e045284` 登錄於 §8。凍結版與草稿版除 `status`/`generated_at`/`generator_commit`/`repo_commit` 外,其餘 wall-time/記憶體/host RSS 數字逐位元相同——凍結動作沒有偷改任何 point estimate。內容含 wall-time point estimate(S-BW 主線 6.73–7.08 h 敏感度帶)、GPU 記憶體解析上下界(26.1–45.2 GB)、host RSS 點估計(168.9 GB,`interval=null`,依 M4-G6 規則因 host RSS 模型未 fit 只給點估計)。**T14 首跑比對器**(`scripts/m4_check_prediction.py`)已含逐 phase `s_p_obs = t_l4_extrapolated_27m_s / T_H100,p` 的檢定邏輯,`MANDATORY_REFIT_DISCLOSURE_SENTENCE` 逐字內嵌。⇒ spec 的 exit 條件(「預測已登錄且輸入可稽核」)已滿足,**E5 = PASS**。T14 本身(H100 實測驗證)在 M4 主線之外,不計入本次判定(§8)。

---

## 8. H100 交接(T10)

**Scripts 清單:**

| script | 角色 |
|---|---|
| `scripts/m4_run.sh` | 單一指令(§6.4);POSIX sh,`--dry-run` 有煙霧測試(`tests/test_m4_handoff.py::test_m4_run_sh_dry_run_smoke`);OOM/crash 走 §2.2 三態規則而非靜默視為 bug |
| `scripts/m4_forecast.py` | 產生 `h100_prediction*.json`(wall-time/GPU/host RSS 三段預測、`--freeze` 旗標、拒絕無 `--force` 覆寫已凍結檔) |
| `scripts/m4_check_prediction.py` | T14 首跑比對器;逐 phase `s_p_obs = t_l4_extrapolated_27m_s / T_H100,p` 檢定邏輯已落地;`MANDATORY_REFIT_DISCLOSURE_SENTENCE`(「任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間」)逐字內嵌,防止事後放寬 |

**runbook:** `docs/handover/h100-30m-runbook.md`(spec §6.4 指名的獨立檔案,已補齊)涵蓋資源契約(§1)、環境重建(§2,含 `CMAKE_CXX_ABI=1`/`sm_90` 兩項必要檢查)、27.7M 陣列在目標機上的重新生成(§3)、單一指令 `scripts/m4_run.sh`(§4)、登錄預測與比對工具(§5)、T14 執行 checklist(§6)、已知風險(§7)。

**T10 rehearsal(3.1M,ours@M2 K16,`results/m4/t10_rehearsal.log`):** 用 `m4_run.sh` 在 `mempool_group` 上非 dry-run 完整演練一次(seed=3000,`results/m4/profile/rehearsal_group__k16__grid__oursM2_seed3000.json`,run_id `d34cc686-7db8-4d2c-b4b2-05d577483a5d`)——**status=`ok`,runtime 1331.5 s ≈ 22.2 min**。**演練過程本身即證實了一條交接暗規則(絕對路徑教訓)**:前兩次嘗試(v1/v2)用相對路徑 `--config benchmarks/ispd25/mempool_group.json --out results/m4/profile/...` 直接失敗——`FileNotFoundError: [Errno 2] No such file or directory: 'benchmarks/ispd25/mempool_group.json'`(driver 對相對路徑求值時所在的 cwd 與呼叫者預期的 repo root 不一致);第三次嘗試(v3)把 `--config`/`--out` 都換成絕對路徑後才成功完成(`03:31:27 T10_REHEARSAL_V3_DONE`)。**H100 交接時必須用絕對路徑呼叫 `m4_run.sh` 的 `--config`/`--out`,不能依賴呼叫時的 cwd**——這正是 runbook §4 已經記載、本次演練實測驗證過的教訓。

**Forecast 凍結(`results/m4/forecast/h100_prediction.json`):**

- **`status="frozen"`**,sha256 **`4eeea5f9132e1a06585ae57e38c6e45b5523ed31ae5454826ff4cff56e045284`**(草稿版 `status="draft"`/`unvalidated=true` 已被此凍結版取代;二者除 `status`/`generated_at`/`generator_commit`/`repo_commit` 外其餘數字逐位元相同)。
- SKU 釘死:H100 SXM5 80GB HBM3、3.35 TB/s peak、sm_90、CUDA 12.8、persistence mode on、預設時脈——若實際機器不同則預測**無效**,須重登錄。T10 rehearsal 本身跑在 L4(22.0 GB)上,已在 `t10_rehearsal.log` 明確記錄「GPU 名稱與凍結 SKU 契約不符,任何對照 `h100_prediction.json` 的比較在此硬體上無效」的告警,不誤用本機結果驗證 H100 預測。
- wall-time:point estimate(S-BW 主線)6.73 h,S-SM/S-FP64 敏感度帶延伸到 7.08 h;**read/parse phase 佔外推總量 89–94%,是唯一支配項**(§6.4)。
- GPU 記憶體:解析下界 26.1 GB(eval 分量主導)、上界 45.2 GB(含 allocator reserve + CUDA context),兩者皆在 H100 的 80 GB 契約內、也遠低於部件級 72 GB 契約。
- host RSS:點估計 168.9 GB(僅由 node_ratio 代理外推,無 95% PI,因 §1.1 的三係數模型尚未 fit,依 M4-G6 規則不得補上 PI)——**逼近但未超過** `h100_min_host_ram_gb_contract=192 GB` 的 T10 資源契約;T10 rehearsal 亦記錄本機 host RAM 125 GB < 128 GB 觸發 spec T12 row(27.7M 這一級需要 numpy-only `PlaceDB` shim,`docs/handover/h100-30m-runbook.md` §7 已列為已知風險)。

**T14(H100 實際執行驗證)在 M4 主線之外,不是本報告的交付範圍**——凍結預測已登錄且輸入可稽核,滿足 E5 的 exit 條件(§7);T14 本身目前尚未有任何執行紀錄,屬於「E5 若永不驗證」的現況,依 spec 要求在此明寫。

---

## 9. Gate 判定總表(G1–G8,spec §9)

逐條對照 design draft §9「風險與 gate」——狀態欄只記「觸發/未觸發」,不是「好/壞」:多數 gate 是安全網,未觸發本身就是預期結果。

| ID | 量測 | 門檻 | 狀態 | 依據 |
|---|---|---|---|---|
| **M4-G1** T2 沒達標 | `mempool_group` K=32 evaluator 峰值 | >2 GB | **未觸發** | §6.3:`test_gpu_evaluator_memory_mempool_group_k32_under_2gb` PASS,peak_alloc=1.8263 GB ≤2 GB |
| **M4-G2** 11.3M 全流程失敗 | E1 任一條 | 任一 fail | **未觸發** | §7 E1 = PASS,六門檻全過(§5.3);§5.1/§5.2 的 `ρ*` 格點選取全數落在格點邊界內部(未達 +5% 上限),未曾因 `Δhpwl` 爆表而需要臨時擴充格點或降 `ρ_max` |
| **M4-G3** 生成器未通過驗證 | §3.3 的 H1–H5 | 任一不綠 | **已觸發**(v2.3,2026-08-15) | §4.1:T6 的 2×2 holdout(N1、N2 皆)FAIL,`t7_gate=FAIL`;依規則轉 T6B(§4.4),報告已逐字載入「30M 的連通性是擬合出來的,不是校準出來的」(標題頁) |
| **M4-G3a** 階層前提 | T3a 的 G-A…G-E | 任一不過 | **未觸發** | `results/m4/corpus/hierarchy_gate.json`:`verdict="PASS_2x2"`,g_a/g_b/g_c/g_d 皆 `pass_=true`(§2 B4) |
| **M4-G4** op runtime 失控 | T8 的 11.3M 單臂 wall-time | >3h | **未觸發** | §7 E1 六門檻表:K16/K32 `ours@M2` 分別 1.89h/2.94h,皆 <3h;iteration 預算未曾被人為降低來偽造達標(§3 的 2000-iter 規則對全語料一視同仁) |
| **M4-G5** host RAM | 任一 `PlaceDB.read` peak RSS | >100 GB | **未觸發** | §5.3/§5.4 全部 24 格的 `host_hwm`(≤81.8 GB)與 §6.6 的 `lifetime_cluster` 觀測(67.7 GB)皆 <100 GB;T10 rehearsal 記錄的「本機 125GB<128GB」是**目標機資源契約**的告警,不是任一 run 的 peak RSS 超標(§8) |
| **M4-G6** 模型失效 | T0b `identifiable`,3 模型 holdout 殘差 | `identifiable=false` 或殘差超出 95% PI | **已觸發** | §6.5:GP 記憶體/runtime 兩子模型 `identifiable=false`;三條後果已處置(27.7M host RAM 欄改「未定」、E5 host RSS 只給點估計不帶 PI、host RSS 三係數模型維持 `pending_t0b` 不回填) |
| **M4-G7** B1 污染範圍 | M2/M3 報告引用 `peak_mem_mb` 的段落 | 存在即觸發 | **已觸發**(既有事實,已處置) | §2 B1:結論(污染成立、M2/M3 `peak_mem_mb` 全部作廢)保留,成因由 T1b 四臂 A/B probe(`t1b_arms.json`)定案為 `retained_tensors`;本報告全部 GPU 峰值數字改引 M4 各自獨立 process 產出的 `device_used_gb` |
| **M4-G8** 報表越界 | `scripts/m4_report_lint.py` | 任一違規 | **已觸發(2026-08-18)→ 已了結(2026-08-19)** | `--strict` 於 2026-08-18 實跑 `24 error(s), 0 warning(s)`(附錄 A.4)——§5.3 全 16 列 `result_gate`(`workload_status` 缺失)、§5.4 全 8 列 `generator_verified` 缺失,根因是 DREAMPlace driver 輸出 schema 從未寫出這兩個欄位,不是表格 caption/欄名寫錯。**依 spec 字面觸發後果「⇒ T11 不得標記完成」**,報告曾據此拒絕自我標記完成;2026-08-19 依使用者裁決 A 執行可稽核 backfill(只補這兩個後定義欄位、不動量測值),`--strict` 重跑得 `0 error(s), 0 warning(s)`,**gate 了結、T11 標記完成**,見附錄 A.4 |

---

## 10. 附錄

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
| H100 forecast(凍結) | `results/m4/forecast/h100_prediction.json`(sha256 `4eeea5f9...`,§8) | `254fea6` |
| T10 handover scripts | `scripts/m4_run.sh`、`scripts/m4_forecast.py`、`scripts/m4_check_prediction.py` | `6a73ae8` |
| T10 runbook | `docs/handover/h100-30m-runbook.md` | `2a2a112` |
| T10 rehearsal(3.1M) | `results/m4/t10_rehearsal.log`、`results/m4/profile/rehearsal_group__k16__grid__oursM2_seed3000.json` | `a1e629c` |
| T11 report linter | `scripts/m4_report_lint.py`、`tests/test_m4_report_lint.py` | `eeb7922` |
| T1b 四臂 A/B probe(B1 成因定案) | `results/m4/profile/t1b_arms.json` | `1668b48` |
| T6B B-1(Rent 形狀不變性 DiD) | `results/m4/bench/verify_group3x3_t6b.json` 的 `B1_rent_invariance_did` 塊 | `25ead25` |
| T6B B-2(H5 等解析度 K-grid) | `results/m4/bench/verify_group3x3_t6b.json` 的 `B2_h5_matched_resolution` 塊 | `5e20576` |
| T9 27.7M 共常駐 spike(A/B) | `results/m4/profile/spike30m__{A,B}__k32.json`;invalid attempt:`results/m4/profile/invalid_attempts/spike30m__{A,B}__k32.attempt1_contaminated_by_gate_race.json` | `254fea6` |
| T2b 元件級生命週期(3.1M 完成、11.3M not_evaluable_scale) | `results/m4/profile/lifetime_group__k32__ours_m2.json`、`results/m4/profile/lifetime_cluster__k32__ours_m2.json` | `634d8bc` / `dc9d7de` |
| E1 seed-B 六門檻(cluster K16/K32×flat/oursM2) | `results/m4/profile/mempool_cluster__k{16,32}__grid__{flat,oursM2}.json` | `fbe123e` |
| E3 seed-B 品質/合成矩陣(16+8 格) | `results/m4/profile/{sb12,bb4,group,mempool_cluster,synthetic_1x2_n2,synthetic_2x2_n2}__k{16,32}__grid__{flat,oursM2}.json` | `e70ce3b` / `634d8bc` |

### 附錄 A.3 跨 session 分工紀錄(2026-08-16,seed B=2000 協定)

commit `9830ff6`(「fix(m4-t8b): fix seed B = 2000 in calib records (cross-session agreement, before any seed-B run)」)把三份校準檔(`rho_group.json`/`rho_sb12.json`/`rho_bb4.json`)的 `seed_B` 欄位鎖定為 **2000**。三份檔案的 `seed_B_note` 逐字記錄了此協定的性質:

> seed B never numerically preregistered; fixed to 2000 by cross-session agreement 2026-08-16 BEFORE any seed-B report run started; requirement satisfied: fixed, documented, not in calibration set

即:`seed_B=2000` 本身不是 T8b 協定原文預先寫定的數字(協定只要求「固定一個 seed B,且不得是校準集合本身用過的 seed」),而是**在任何 seed-B 正式 run 開始執行之前**、由跨 session 的協作決定並寫入 artifact 的一次性選擇。三個必要性質(固定、有文件記錄、不在校準集合內)在 commit 時點已全部滿足;`seed_a=1000`(校準集合用的 seed)與 `seed_B=2000` 因此互斥,不違反 T8b「校準集合與報告集合不得共用同一顆 seed」的紅線。本附錄記錄的鎖定事實其後已被兌現:§5.3/§5.4 引用的全部 24 格 seed-B 正式 run(`seed=2000`)皆已執行完畢(E1/E3,§7)。

### 附錄 A.4 Linter 執行結果(M4-G8)與根因揭露

**執行指令(2026-08-18):**

```
PYTHONPATH=. $DP/.venv312/bin/python scripts/m4_report_lint.py --results-root results docs/results/m4-scale-up-report.md --strict
```

**結果:`24 error(s), 0 warning(s)`——不是 `0 error 0 warning`,M4-G8 已觸發。** 全部 24 筆錯誤逐一對應 §5.3/§5.4 的每一列(16+8=24),分兩類:

- **`result_gate`(16 筆,§5.3 全表)**:「quality table row requires `workload_status=='completed'`(got `None`)」。
- **`generator_verified`(8 筆,§5.4 全表)**:「scaling table row's JSON is missing `'generator_verified'`」。

**根因(已核對,非表格措辭問題):** 這不是本報告表格 caption/欄名沒改對——§5.3/§5.4 的 caption 與欄名已依 spec §6.2 原名寫定,linter 也確實「咬住」了兩張表(這本身是設計上的正確行為,§5.3 舊版草稿曾特意用非正式 caption 迴避 linter,定稿時已改正)。**24 筆錯誤全部源自被引用的 artifact 本身缺少 linter 要求的欄位**,不是表格內容或欄名寫錯:

1. `results/m4/profile/{sb12,bb4,group,mempool_cluster,synthetic_1x2_n2,synthetic_2x2_n2}__k{16,32}__grid__{flat,oursM2}.json`(§5.3/§5.4 全部 24 個 run 的來源,由 `ioplace/drivers/run_placement.py`/`run_placement_io.py` 產生)**沒有一個帶 `workload_status` 欄位**——`workload_status` 只存在於較新的 T2b(`lifetime_*.json`)與 T9(`spike30m_*.json`)兩支 harness 的輸出中,DREAMPlace 主 driver 從未寫出這個欄位。RESULT GATE 規則 5 對 `workload_status` 缺失(`None`)與其他任何非 `"completed"` 值一視同仁判 error,quality 表因此逐列全部落網。
2. 同一批 24 個 run 也**沒有一個帶 `generator_verified` 欄位**——這個欄位目前只出現在 `count_freeze_{small,30m}.json` 與 `verify_group3x3_t6b.json` 三個 bench/count-freeze 層級的 artifact,不在任何逐 run 的 profile JSON 裡;§5.4 的 8 個合成 case run 因此全部落網。附帶一提(§5.4 已揭露):這 8 個 run 的 `benchmark_kind` 欄位值其實是 `"real"` 而非 `"synthetic"`(driver 呼叫時未帶 `--benchmark-kind synthetic`)——這**不是**本次 24 筆錯誤的成因(rule 2 的 synthetic-isolation 檢查只單向禁止 quality 表引用 `benchmark_kind=="synthetic"` 的 run,不會因為 scaling 表引用 `benchmark_kind=="real"` 的 run 而報錯),但與 `generator_verified` 缺失是同一組 artifact-schema 缺口的兩個面向。

**處置範圍評估(2026-08-18 當下的誠實記錄,當時未執行):** 修正這 24 筆錯誤需要在 `ioplace/drivers/run_placement.py`/`run_placement_io.py`(或其共用的輸出 schema 模組)補上 `workload_status`/`generator_verified` 兩個欄位的寫出邏輯,並**對已落地的 24 個 run 重新執行**(或針對既有 JSON 做欄位回填)——兩者都超出本次 T11 報告 pass 的唯讀範圍(僅可讀 `results/`、不可修改)與「表格措辭」修正範圍;逕自在 `results/` 下回填欄位值也會構成憑空生成未經量測的 provenance,不可取。**依 M4-G8 的字面觸發後果(design draft §9:「⇒ T11 不得標記完成」),本報告在此誠實記錄:M4-G8 已觸發且尚未處置,T11 因此不能依 spec 字面規則標記為完成**——即便 §5.3/§5.4 的資料內容本身(E1/E3 的規模覆蓋、run_id 唯一性、`status="ok"`)在實質上是完整且正確的。此缺口的了結方式(修 driver + 重新落地 24 個 run,或另立一條類似 B-2/B-5 的裁決把「舊 schema run 免除 `workload_status`/`generator_verified` 硬性要求」正式記錄為非 gating 揭露)不是本報告執行者可以片面決定的設計選擇,留待下一輪處置。

**了結(2026-08-19,使用者裁決 A):** 使用者裁決採 **A(批准可稽核的 backfill)**,上一段所稱「不是報告執行者可以片面決定」的設計選擇因此由使用者作出。執行方式刻意受限,以免把 schema 補記變成憑空生成 provenance:

```
PYTHONPATH=. $DP/.venv312/bin/python scripts/m4_backfill_result_gate_fields.py \
    docs/results/m4-scale-up-report.md --results-root results/ --apply
```

- **目標集合不是手寫的**:腳本 import `m4_report_lint`,用 linter 自己的表格分類與列解析取得要處理的 JSON,因此只能碰到 linter 本身引用的檔案(實跑解析出 24 筆,與 24 個 error 一對一)。
- **只新增、不覆寫**:欄位已存在者一律 `skip`;每個檔案除了新增的欄位與一則 `schema_backfill_note`(記錄補記時間、腳本、裁決、理由、證據、「未讀取/重算/更動任何量測值」的不變式)之外,其餘 key 與值逐一保持不變。事後以 `git show HEAD:<path>` 對照驗證:24 個檔案**無任何 key 被刪除、無任何既有值被改變**。
- **`workload_status="completed"` 需要正面證據才寫**:完整 placement run 必須同時具備 `status="ok"` 與 driver 最後才寫出的收尾欄位(`legalization_status="success"`、`num_unplaced_cells`、`final_overflow`、`gp_iterations_run`、`hpwl_lg`、`runtime_s`);evaluate-only run 必須具備 `status="ok"` 與其評估量(`hpwl`/`io_count`/`ft_count`/`runtime_s`)。拿不出證據的檔案會被 `refuse` 且腳本整體非零退出,不猜、不填。實跑結果:24 筆全部具備證據,0 筆 refuse。
- **`generator_verified=false`**:這是事實值——生成器從未被認證過。寫 `false` 同時讓裁決 §7-1 的強制揭露句在 linter 規則 3 下**持續為必要**,不是繞過它。
- **稽核紀錄**:`results/m4/backfill/2026-08-19-result-gate-backfill.json` 逐檔記錄 table 類別、報告列號、欄位、值、證據字串,以及寫入前後的 sha256。

**重跑結果:`0 error(s), 0 warning(s)`,M4-G8 了結,T11 依 spec 字面規則標記完成。**本報告的任何數字、任何 E/G 判定都未因此改變——backfill 補的是 schema 欄位,不是結論。**仍未了結的同源缺陷:** §5.4 那 8 個 run 的 `benchmark_kind` 值仍是 `"real"`(應為 `"synthetic"`),本次 backfill 因「不覆寫既有值」的自我限制而未動它;根治方式是在 driver 端補上欄位寫出邏輯(`--benchmark-kind` 正確帶入),留待下一輪 driver 修改處置。
