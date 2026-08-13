# M4 設計草案 v2:規模化(1M → 30M)

- 日期:2026-08-13
- 狀態:**v2 草案**——依 `docs/reviews/2026-08-13-m4-draft-v1-adversarial-codex.md`(16 findings / 8 BLOCKER)逐條修訂 v1(commit `0bd9611`)
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §7(benchmark 計畫)、§8(實驗設計)、§9 M4 列、§10 風險 2、D6(硬體)
- 相依草案:`docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`(§7 G6 8GB 契約、§7 R2 evaluator 斷點、T1 分工邊界)、`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md`(§3.2 環境盤點、S5 = NanGate45 語料接入,**本文與其共用同一個 task,不重複設計**)
- 繼承資產:`ioplace/ops/io_term.py` chunked-k 契約、`ioplace/diagnostics/spike_10m.py`、`ioplace/evaluator_gpu.py`、`ioplace/drivers/run_placement{,_io}.py`、`ioplace/profile.py`(T1 已落地保守版)

**兩條誠實原則(v2 新增,適用全文):**

1. **所有 v1 標「實測」的數字一律保留但降級為「待 T0 重現」**——探針仍在 `/tmp`,未進 repo 前不得作為裁決依據。
2. **被 Codex 指出「論證不成立」的段落,v2 改的是論證本身,不是只把結論軟化。** 具體是 §1.4 B1(因果)、§2.2(峰值相加)、§3.1(cluster = 4 groups)、§3.2(距離核可識別性)、§6.3 E5(頻寬啟發式)、§7(DAG)。

---

## 0.1 v1 → v2 變更摘要(逐條回應 Codex 16 findings)

| # | 嚴重度 | Codex 主張(摘要) | **v2 處置** | 影響段落 / task |
|---|---|---|---|---|
| 1 | BLOCKER | 五點 GP 模型是 training residual,共線、無 DoF,890 s 不是 runtime 模型 | **接受。** GP 記憶體模型降級為「保守上界 + 未驗證預測」;新增 **T0b** factorial probe(固定 case 獨立掃 bins / target_density / filler)輸出 design matrix、單位、condition number、係數 95% CI、leave-one-case-out 殘差;runtime 另建 `n_iter × per-iteration + fixed phases` 模型,禁止由記憶體擬合或單一 pins 比例推出;6.2M / 12.3M 為 holdout,通過才准外推 27.7M | §1.1、§2.1、§5.2、T0b、M4-G6 |
| 2 | BLOCKER | standalone 峰值相加既不能證明可行也不能證明不可行;driver 中三個元件是共同常駐 | **接受。** §2.2 改標題為「保守上界」,並補正確帳式 `peak ≈ resident(t) + max_phase_transient`;新增 **T2b integrated lifetime probe**(逐 buffer create/destroy phase、phase-start allocated/reserved、phase peak);**可行性只由 full-lifetime spike 判定**,上界不得單獨當 feasibility proof 或 blocker | §2.2、§5.3、T2b、E1/E2 |
| 3 | MAJOR | host RSS / IoTerm 的 per-pin 模型 predictor 不足(Bookshelf 只有一點;IoTerm 真正規模量是 dedup pins) | **接受。** 兩個模型改多預測子:`N_nodes, N_nets, N_raw_pins, N_dedup_pins, total_name_bytes`;在 group / 1×2 / 2×2 三點量測,27.7M 的 host 判定改由 **2×2 holdout 殘差 + 95% 上/下置信界**的三分規則(見 §5.1) | §1.1、§1.3、§5.1、T0b、T6b |
| 4 | BLOCKER | 12.3M / 27.7M 的 nets/pins 帳只有 replication,glue = 0;int32 composite key 只剩 6.4% headroom | **接受。** §2.1 / §8 全部帳改成 `base + g`(`g_net`、`g_pin` 為符號);新增 **T6b count freeze** 於 T7 後以 manifest 實數回填;**composite key 一律 int64**,並在 `GpuEvalContext.__init__` 加 `assert n_nets * 64 < 2**63`;index 降 int32 只在逐欄位 assert `max < 2**31-1` 後個別為之,不得一概而論 | §2.1、§4.2、§8、T6b |
| 5 | BLOCKER | `γ_far` 在 2×2 完全不可識別;3×3 的 `φ(√5)`、`φ(√8)` 未定義 | **接受並改設計。** 距離核改單參數冪律 `φ(d;α)=d^(−α)`(對所有 d 有定義),`(λ₀, α)` 由 2×2 的鄰接/對角兩個統計**閉式**(method-of-moments)解出,**不再有 grid search**;明寫可識別性帳:2 觀測 / 2 參數 ⇒ 0 residual DoF ⇒ **2×2 無法驗證核形狀**,3×3 必須發表三種核(冪律 / 指數 / `γ_far=0` 截斷)的**系統性不確定度帶** | §3.2、§3.3、T6 |
| 6 | MAJOR | V2/V4 同時當調參目標與驗證,是 scan-until-pass;V5 量綱錯 | **接受。** 驗收拆成 **fit metrics(不判定,只報)** 與 **holdout metrics(預先登錄門檻,只評一次)**;fit 用 pair-count-by-distance,holdout 用 Rent 曲線形狀 + per-level cut 直方圖 + 全網 degree KS + interface-cell 空間分布 + K-grid λ 比;固定 seed 集(≥5)與 bootstrap CI;**失敗不得重掃**。V5 降級為建構法的算術自檢 + 診斷報告(比例定義改 `C(m)/(m·N_tile+C(m))`,不再宣稱 `(RC)^p`) | §3.3、T5、T6 |
| 7 | BLOCKER | 「cluster = 4 groups」只有 cell-count ratio,卻已是整個合成的 ground truth | **接受。** 新增 **T3a(blocking hierarchy gate)**:解析 cluster instance 前綴,逐 group 報 cells/nets/pins/cell-type 直方圖/bounding box,與 standalone group 對照;四道 gate 全過才准產 `cluster_stats.json` 或啟動 T4/T6;不過 ⇒ **退配方 B** 並在報告明寫循環論證風險。另補一個 v1 漏掉的事實:`11,310,807/4 = 2,827,702`,比 standalone group **小 8.1%** ⇒ glue 統計必須**按 tile 規模/邊界長度正規化**後才可外推 | §3.1、§3.2、T3a、M4-L9 |
| 8 | MAJOR | 「30M 只作 scaling」沒有被 artifact 結構強制 | **接受。** 報表拆 `quality_real_cases.md` / `scaling_synthetic_cases.md`;每筆 profile JSON 帶 `benchmark_kind ∈ {real, synthetic}`;synthetic 表的 schema **禁止** `Δio/Δft/Δhpwl/winner/pareto` 欄位;新增 `scripts/m4_report_lint.py` + pytest schema gate | §3.4、§6.2、T8a、T11 |
| 9 | MAJOR | int8 改法要 source + accumulator 成對改,M3/M4 分工寫錯 | **接受。** M3 T1 的 scope 改寫為「`_one_hot_planes`/`_bit_planes` 產生 int8 + 兩個 accumulator int8」**成對**;收益重算:`(P,K)` 11.25→1.41 GB、`2×(E,K)` 6.5→0.81 GB ⇒ M3 T1 省 **≈15.5 GB**(35.6→≈20 GB,**仍 OOM,T2 仍是硬前置**);實測補上 dtype 契約(見 §4.2)。**修正 Codex 的 K=64 測試建議**:`_pack_bits` 用有號 int64,`1<<63` 溢位為 INT64_MIN,K=64 本來就在契約外 ⇒ 加 `assert k <= 32`,測 K ∈ {1, 8, 32} | §4.2、T2、M3 介面 |
| 10 | BLOCKER | edge batching 不可能保證所有輸出逐位元相同(`tree_wl` 是 float64 sum) | **接受。** T2 驗收改 field-specific:整數欄位(crossings/FT/λ/pair counts/per-net 欄位/edge 計數)**逐位元 exact**;`tree_wl`/`hpwl` 改**預先固定容差 rel ≤ 1e-12**(今日反例:7.65M 項 heavy-tail float64 求和,batch=1e6 與全量差 rel 1.15e-16,batch=3e6/8e6 恰好相同 ⇒ 逐位元相同**不可靠**,但誤差在 ulp 級);另加「edge 總數與 per-net edge 數逐位元相同」以防容差掩蓋掉丟邊 | §4.2、T2 |
| 11 | MAJOR | B1 的污染成立,但「固定步進 ⇒ cumulative HWM、reset 足以修好」不成立 | **接受。** §1.4 B1 論證改寫為「污染已證、成因未證」;T1 已落地的 reset 保留但文件明記**非充分**;新增 **T1b**:fresh-subprocess / same-process / reset-only / explicit-teardown+GC 四臂 A/B probe,記錄 run-start/end allocated+reserved;正式 ablation 改**每臂一個 subprocess**,除非 A/B 證明 teardown 後 baseline 回到容忍範圍 | §1.4 B1、T1、T1b |
| 12 | MAJOR | B2 是 `1 + n_nonempty_buckets` 不是固定 8;reuse 不成立 | **接受。** B2 改寫為 `1 + n_nonempty_buckets`(`DEG_BUCKET_LABELS` 7 桶,adaptec1/bigblue4 全非空 ⇒ 現況 8);**撤回 io_grad_l1 reuse**(bucket-masked 梯度的 L1 不能還原完整梯度 L1,跨桶會相消);T1 已落地版只加 `--diag-every N` / `--no-diag`;驗收改**實測 callback wall-time**(含 io_term.py:364 的 no-grad softmax pass 與 callback 內的 WL backward),不數 backward 次數 | §1.4 B2、§5.1、T1 |
| 13 | MINOR | `--budget-gb` 把固定錯誤換成可任意調高的 gate | **接受(T9 前固定)。** spike 結果同時存 `measured_peak_gb / budget_gb / budget_source / baseline_reserved_gb`;budget 由 manifest 的硬體契約決定,CLI 傳超過契約值 ⇒ assert 失敗。預先寫死:L4 元件級 spike = **19.5 GB**(0.9 × 21.7 GiB)、H100 = 72 GB、M2 G6 的 10M 契約維持 8 GB | §1.4 B3、T1b、T9 |
| 14 | MAJOR | 新 profile 基建重複 B1 類錯誤(`ru_maxrss` 不可 reset、`mem_get_info` 是全卡) | **接受。** `ioplace/profile.py` 欄位語意已更名為 `host_rss_hwm_at_phase_end`(process-lifetime HWM 的取樣,**不是** per-phase peak)與 `device_used_gb`(**整卡**用量,含他人 process);**T1b** 補:正式 memory run 要求 **exclusive GPU** 並記錄 device baseline、需獨立歸因的 phase 走 child process 取 peak RSS、sampler 精度用人工短峰 probe 驗證(0.5 s 取樣會漏 transient) | §6.1、T1、T1b |
| 15 | BLOCKER | E5 的 `[0.3×,3×]` 幾乎不可證偽;頻寬啟發式不是 end-to-end 模型 | **接受並重寫。** E5 改為 **phase 級 Amdahl 分解**`T_H100 = Σ_p T_L4,p / s_p`,每個 `s_p` 由 L4 實測 achieved bandwidth 決定 regime(頻寬受限 vs 計算/atomic 受限),host parse 明定 `s=1`;交付 **point estimate + 單一硬 prediction interval**(wall-time median `[0.7,1.3]×`、GPU peak `[0.85,1.15]×`、host RSS 用多預測子模型的 95% PI);**刪除第二層容忍帶**;E5 降為 **forecast deliverable**(M4 exit = 預測已登錄且輸入可稽核),真正的證偽由新增的 **T14(H100 execution & validation)** 執行;SKU/時脈/CUDA build 一併釘死 | §6.3 E5、§6.4、T10、T14 |
| 16 | BLOCKER | DAG 不保證成果被執行;`ours` 未定義;acceptance 可被空 JSON 滿足 | **接受。** 重畫 DAG(T4/T6b→T8、T3a hierarchy gate、T6b count freeze、T8b per-case ρ*、T11 報表 audit);臂改稱 **`flat` / `ours@M2` / `ours@M3`**(M2 臂為主線、M3 臂為 M3-dependent 擴充);E1 需要的 `num_unplaced_cells` / `final_overflow` / `legalization_status` 欄位寫進 **T8a**(schema 在 T1);所有實驗 task 的 acceptance 統一引用 §7.0 的 **RESULT GATE**(status / schema / finite / provenance hash / run_id 唯一性 / 非空) | §6.2、§6.3、§7 全節 |

---

## 0. 摘要(v2 裁決一覽)

| # | 問題 | 裁決(v2) | 信心 |
|---|---|---|---|
| Q1 | benchmark 階梯 | **1.3M = ISPD2015 `mgc_superblue12`;3.1M = ISPD2025 `mempool_group`(3,077,669 cells,**不是 spec 寫的 5M**);11.3M = ISPD2025 `mempool_cluster`(11,310,807 cells);6.2M / 12.3M / 27.7M 由同一個 tiler 從 `mempool_group` 合成。** `mempool_cluster` 的 DREAMPlace GP+LG 在 L4 上跑得完(362.4 s / 6,889.6 MiB GPU / 62.6 GB host,**待 T0 重現**)⇒「10M 只能在 H100 跑」的先驗假設被推翻 | 高(三個規模皆實跑,重現待 T0) |
| Q2 | 30M 測資 | **配方 A′:`mempool_group` 3×3 陣列(27.70M cells + glue)+ 從真實 cluster 量得的跨 group 互連統計,在 Bookshelf 域串流合成。** 校準靶是「2×2 陣列 vs 真實 `mempool_cluster`」——**但這個靶的前提(cluster 由 4 個 group 構成)在 T3a 通過前不成立**;距離核改單參數冪律,參數閉式求解,驗證只用 holdout 指標 | 中(來源實測;階層前提待 T3a;核形狀無法在 2×2 驗證) |
| Q3 | evaluator 規模斷點 | **M3 draft R2 嚴重低估。** evaluator 現況 ≈ `(0.85 + 0.061·K) GB / M-net`(五點擬合,**待 T0 重現**)⇒ cluster(12.71M nets, K=32)需 35.6 GB。最大單項是 `evaluator_gpu.py:351` 的 `(P,K)` int64 one-hot(11.25 GB)。**分工:M3 T1 做「dtype 成對降級」(source + accumulator,省 ≈15.5 GB,仍 OOM);M4 T2 做結構性 streaming(消滅 `(P,K)` 物化、edge/segment 批次化),T2 是 M4 全部 ≥5M 實驗的硬前置** | 高(斷點成立);中(修後估值待 T2 實測) |
| Q4 | 10M+ 全流程瓶頸 | **不是 DREAMPlace GP。** 三個候選瓶頸:(i) evaluator 記憶體(Q3,已確立);(ii) IO op 的 K-pass runtime(30M spike 單次 fwd+bwd 26.7 s,**待 T0 重現**);(iii) host RAM(27.7M 推估 95–105 GB,**但這是單點外推,判定改由 §5.1 的三分規則決定,T6b/T8 才定案**) | 中高(i);中(ii);**低→待 holdout**(iii) |
| Q5 | 計時與 profile | 6 phase 牆鐘 + per-phase reset 的 GPU 峰值 + `host_rss_hwm_at_phase_end` + `device_used_gb` + per-iteration CUDA-event 直方圖 + op fwd/bwd 分離計時,**零新依賴**;**per-phase 歸因的正確性由 T1b 的 subprocess A/B 與 exclusive-GPU 協定保證,不是由 reset 保證**。exit 改為「L4 可驗證的四條 + 一個登錄制 forecast(E5)+ 一個 H100 驗證 task(T14)」 | 中高 |
| Q6 | task 分解 | `T0 → T0b → {T1→T1b, T2→T2b, T3→T3a} → T4 → T5 → T6 → T6b → T7 → T9`;實驗線 `T8a → T8b → T8 → T11`;交接線 `T10 → T14`;條件 task T12/T13。**M3-dependent 的只有 `ours@M3` 臂與其報表欄位** | — |

**一句話結論:** v1 的核心觀察(DREAMPlace GP 在 10M 級不是瓶頸,我們自己的 evaluator 與 IO op 才是)**存活**;v1 的推論方式(五點擬合外推、峰值相加判可行、cell-count ratio 當階層證據、10× 容忍帶當預測)**不存活**。v2 把每一個「靠外推得到的裁決」換成「一個能證偽它的實驗 + 一條在證據不足時停下來的規則」。

---

## 1. 量測基準(**全部待 T0 重現**)

環境:`$DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`、`$DP/.venv312/bin/python`、torch 2.8.0+cu128、NVIDIA L4。`torch.cuda.mem_get_info()` 回報可用裝置記憶體 21.95 GiB,CUDA context 另佔約 0.2 GiB ⇒ 張量預算上限一律以 **21.7 GiB** 計。**量測時點定義(v2 新增,補 Codex #2 的缺口):`mem_get_info()` 一律在 CUDA context 初始化後、任何 M4 張量配置前取一次作為 `device_baseline_gb`,之後的 `device_used_gb` 一律減去它再與預算比較,不再另外扣 context。**

### 1.1 DREAMPlace GP+LG(det=1、DP 關、fillers 開、**不含 evaluator**)

| case | 格式 | movable | nets | pins | filler | bins | read s | GP+LG s | host RSS | GPU 峰值 alloc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 | Bookshelf | 210,904 | 221,142 | 919,701 | 160,067 | 512² | 3.4 | 14.4 | — | 140.8 MiB |
| mempool_tile_wrap | LEF/DEF | 127,739 | 145,589 | 521,021 | 163,607 | 512² | 2.7 | 7.8 | 1.13 GB | 105.2 MiB |
| bigblue4 | Bookshelf | 2,169,183 | 2,229,886 | 8,731,365 | 2,885,580 | 2048² | 39.3 | 40.8 | 8.18 GB | 1,670.9 MiB |
| **mempool_group** | LEF/DEF | 3,077,669 | 3,503,992 | 12,026,191 | 633,128 | 2048² | 73.5 | 92.0 | 16.67 GB | 1,792.6 MiB |
| **mempool_cluster** | LEF/DEF | 11,310,807 | 12,712,800 | 43,947,793 | 3,889,360 | 4096² | 284.9 | **362.4** | **62.59 GB** | **6,889.6 MiB** |

**v1 的擬合模型與其地位(v2 降級):**

```
GP_peak_bytes ≈ 87·N_total + 73·N_pins + 160·n_bins        (N_total = physical + filler)
```

- 這是**五個 case 級觀測、三個係數、無截距**的擬合,只剩 2 個 residual DoF;4.3% 是 **training residual,不是 prediction error**。
- 五個 case 同時改變格式、filler 比、bin 解析度與拓撲,`N_total`/`N_pins`/`n_bins` 高度共線 ⇒ **87/73/160 三個係數各自是否穩定,無法從這五點識別**。
- ⇒ **在 T0b 產出 design matrix、condition number、係數 CI、LOO 殘差,並在 6.2M/12.3M 的 holdout 上通過之前,本式只作為「同量級的量階指示」,不得用來裁決 27.7M 的可行性。** 所有引用它的表格數字一律標「上界/推估(未驗證)」。
- **runtime 不得由此式或 pins 比例推出。** v1 的「27.7M GP ≈ 890 s」**作廢**;改由 T0b/T1 建立 `T_gp = n_iter × (a·N_total + b·N_pins + c·n_bins·log n_bins) + T_fixed`,其中 per-iteration 成本由 T1 的 CUDA-event 直方圖直接量,`n_iter` 由 config 與停止條件決定。

**host RSS 模型(v2 改多預測子,Codex #3):** v1 的 `1.35–1.42 KB/pin (LEF/DEF) / 0.88 KB/pin (Bookshelf)` 中,Bookshelf 只有 bigblue4 **一個**有效點 ⇒ 不是模型。改為

```
host_RSS_read ≈ β0 + β1·N_nodes + β2·N_nets + β3·N_raw_pins + β4·N_dedup_pins + β5·total_name_bytes
```

在 group(3.1M)/ 1×2(6.2M)/ 2×2(12.3M)三個 Bookshelf 產物 + cluster(LEF/DEF)上量測(T6b);`total_name_bytes` 必須顯式計入,因為 tiler 的 `t{i}_{j}/` 前綴會改變名稱長度(27.7M nodes × ~6 B ≈ 0.17 GB,不大但不可默默省略)。

四個必須寫進報告的次級事實(不變):

1. **mempool 家族沒有 movable macro**(tile_wrap:`n_movable_macro=0`、`max_h_in_rows=1.0`,SRAM 全在 `num_terminals`)⇒ 符合 Phase 1 的 fixed-macro 前提。
2. **`mempool_group` 是 3.08M 不是 spec §7 寫的 5M**;`mempool_cluster` 是 11.31M 不是 10M。spec §7 的表要修。
3. **必須 glob 全部 15 個 NanGate45 LEF、tech.lef 在最前**(Stage2 §3.2 已實測)。
4. GP 的 iteration 預算是 1000(config),`mempool_cluster` 362.4 s ⇒ 每 iteration ≥ 0.36 s(不含我們的 op)。

### 1.2 evaluator(`evaluator_gpu.GpuEvalContext.evaluate`,合成 netlist、bigblue4 degree 分布、uniform 位置)

| nets | pins | K | 峰值 GB | eval s |
|---:|---:|---:|---:|---:|
| 0.6M | 2.46M | 32 | 1.72 | 0.78 |
| 1.2M | 4.92M | 32 | 3.38 | 1.02 |
| 2.4M | 9.82M | 32 | 6.74 | 1.73 |
| 2.4M | 9.82M | 16 | 4.39 | 1.60 |
| 4.8M | 19.6M | 32 | 13.46 | 3.15 |

`mem ≈ (0.85 + 0.061·K) GB / M-net`。**這條的外推風險遠低於 GP 模型**——因為 §4.1 的逐項張量帳是**由程式碼直接算出來的**(形狀 × dtype),擬合只是交叉驗證;但仍受一個已知偏差影響:合成 case 的 uniform 位置會系統性高估 crossing 數與 pair-demand(§9 R4),故 K-independent 項偏保守。**T0 重現時必須同時輸出「解析帳 vs 實測」兩欄。**

### 1.3 IoTerm(M2 chunked op,K=32)

| nodes | nets | pins(raw) | k_chunk | 峰值 GB | fwd ms | bwd ms | host RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 12M | 49.02M | 1 | 4.107(`results/m2/spike/spike_10m.json`) | 3,929 | 11,917 | — |
| **30M** | **36M** | **147.1M** | 1 | **12.32**(reserved 15.29) | **5,732** | **20,932** | 11.76 GB |

v1 由這兩點得出「84 B/pin,記憶體純線性於 raw pin 數」。**v2 修正(Codex #3):IoTerm 的真正規模量是 dedup 後的 `P′ = self._n_pins_dedup`、`num_physical` 與 `k_chunk`**(`io_term.py:329-332`:`denom = max(num_physical, P′)`、`k_chunk = max(1, min(K, chunk_budget // denom))`、`last_peak_chunk_elems = k_chunk · denom`),不是 raw Bookshelf pin 數。兩個 spike 點的 dedup 率恰好接近,才讓 84 B/pin 看起來線性。⇒ **模型改寫為 `peak ≈ c1·k_chunk·max(num_physical, P′) + c2·P′ + c3·n_active`,係數在 T0/T9 用真實 tiler 產物重新擬合**;`spike_10m.json` 也缺 commit/env/input hash 與 `peak_reserved_gb`,T0 重跑時一併補齊(target 40M / 實際 49,019,855 pins 的差異亦須記錄)。

### 1.4 三個量測工具本身的 bug(v2 修正其中兩條的論證)

| # | 現象(成立) | 成因(v2 修正) | v2 處置 |
|---|---|---|---|
| **B1** | `results/m2/**/*.json` 的 `peak_mem_mb` **受同 process 內先前的臂污染**:adaptec1 `A0_k8` = 140.765625 MiB 與全新 process 的 GP 峰值逐位元相同,其後 k16/k32/slicing = 253.4/366.7/479.3(+112.6 定值步進);bigblue4 A2/A3/A4/A6 = 10.58/12.46/14.35/16.23 GB(+1.88 GB 定值步進)。兩個 driver 都沒有 reset(`run_placement.py:129`、`run_placement_io.py:202`),`run_ablation_m2.py:86` 在同一 process 依序跑所有臂 | **v1 寫「⇒ process 累積 high-water mark」是過度推論。** `max_memory_allocated()` 是「歷史最大 **active** allocation」,不會自動相加;固定步進**更像**是每臂結束後仍存活的張量(retained tensors / 參照環 / 缺 teardown),而**單靠 reset 不會清掉仍存活的 allocation**。此外 runner 會跳過已存在的檔案,resumed run 的真實執行順序無 provenance,不能只從檔名重建 | **結論(污染成立、M2/M3 的 `peak_mem_mb` 一律作廢)保留;成因改標為「未證」。** T1 已落地的 per-phase reset 保留,但文件明記其**非充分**;**T1b** 以四臂 A/B probe 定案成因,正式 ablation 改每臂一個 subprocess |
| **B2** | driver 每次 callback 呼叫 `io_term.diagnostics()`(`run_placement_io.py:155`),它對**每個非空 degree bucket** 各跑一次完整 chunked fwd+bwd(`io_term.py:386-399`),再加 `io_grad_l1`(`:145`)1 趟 | **v1 的「固定 8×」不是通式。** 正確式是 **`1 + n_nonempty_buckets`**(`DEG_BUCKET_LABELS` 共 7 桶,`io_term.py:17`;adaptec1/bigblue4 七桶皆非空 ⇒ 現況剛好 8)。而且 diagnostics 之前另有一趟 no-grad softmax/chunk pass(`io_term.py:364-376`),callback 內還有 WL backward(`run_placement_io.py:141`)⇒ **實際 wall-time > 8 × op fwd+bwd** | 撤回 v1 的「讓 `io_grad_l1` 復用那一次 backward」——**數學上不可能**:bucket-masked 梯度的 L1 範數不能還原完整梯度的 L1(跨 bucket 的座標梯度會相消,只有 `Σ_b‖g_b‖₁ ≥ ‖Σ_b g_b‖₁`)。T1 只加 `--diag-every N` / `--no-diag`;**驗收改實測 callback wall-time**,不數 backward |
| **B3** | `spike_10m.py:116` 的 `ok = peak_gb <= 8.0` 是 M2 對 10M 定的門檻,30M 下必然 false | 成立 | 參數化為 `--budget-gb`,**但**(Codex #13)結果必須同時存 `measured_peak_gb / budget_gb / budget_source / baseline_reserved_gb`,且 `budget_gb` 上限由該 run manifest 的硬體契約 assert:**L4 = 19.5 GB、H100 = 72 GB、M2 G6 的 10M 契約維持 8 GB**,CLI 不得超過 |

---

## 2. Q1 — benchmark 階梯與各級的 L4 可行性

### 2.1 定案階梯(所有合成級的 nets/pins 皆為 **replication-only base**,glue 以 `g` 表示)

| 級 | case | 來源 | cells | nets | pins | GP 峰值 | evaluator(現況/M3 T1 後/T2 後)K=32 | IoTerm | host RAM | L4? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.3M | `mgc_superblue12` | ISPD2015 config 已存在 | 1.287M | ~1.4M(推估) | ~4.5M(推估) | 0.64 GB(推估) | 3.9 / ~2.2 / ~0.6 GB | 0.35 GB | ~6 GB | ✅ |
| 2.2M | `bigblue4` | ISPD2005 Bookshelf | 2.169M | 2.230M | 8.731M | 1.63 GB(待重現) | 6.2 / ~3.5 / ~0.9 GB | 0.68 GB | 8.18 GB(待重現) | ✅ |
| 3.1M | **`mempool_group`** | ISPD2025 NanGate45 | 3.078M | 3.504M | 12.026M | 1.75 GB(待重現) | 9.8 / ~5.5 / ~1.3 GB | 0.94 GB | 16.67 GB(待重現) | ✅ |
| 6.2M | group 1×2 | **本專案 tiler** | 6.155M | 7.008M **+ g₂** | 24.05M **+ gp₂** | 3.07 GB(推估) | 19.6 / ~11 / ~2.5 GB | 1.88 GB | ~21 GB(推估) | ✅(T2 後) |
| **11.3M** | **`mempool_cluster`** | ISPD2025 NanGate45 | 11.311M | 12.713M | 43.948M | 6.73 GB(待重現) | **35.6 / ~20 / ~4.5 GB** | 3.44 GB | 62.59 GB(待重現) | ✅(**T2 後**) |
| 12.3M | group 2×2(= cluster 的合成對照組) | **本專案 tiler** | 12.311M | 14.016M **+ g₄** | 48.10M **+ gp₄** | 7.3 GB(推估) | 38.8 / ~22 / ~4.9 GB | 3.77 GB | ~43 GB(推估) | ✅(T2 後) |
| **27.7M** | group 3×3 | **本專案 tiler** | 27.699M | 31.536M **+ g₉** | 108.24M **+ gp₉** | 12.6 GB(**上界,未驗證**) | 87.3 / ~49 / ~10 GB | 8.47 GB | **95–105 GB(單點外推,判定見 §5.1)** | **待 T6b/T8 定案** |

- **glue 佔位符**:`g₂/g₄/g₉` = 1×2 / 2×2 / 3×3 的 glue net 數,`gp₂/gp₄/gp₉` = 對應 pin 數。v1 的表把它們當成 0(`3,503,992×9 = 31,535,928`、`12,026,191×9 = 108,235,719` 正好是純複製)⇒ **所有下游記憶體/runtime 帳在 T6b count freeze 之前都是低估**。
- **int32 headroom 警戒線(Codex #4)**:若採 `unique(net·64 + rid)` 這類 composite key 且降為 signed int32,則 `n_nets ≤ floor((2^31−1)/64) = 33,554,431`;3×3 base 已用掉 31,535,928,**只剩 2,018,503(6.40%)給 glue**。⇒ **裁決:composite key 一律保持 int64**(它是逐 chunk 的暫存,代價有界),並在 `GpuEvalContext.__init__` 加 `assert self.n_nets * 64 < 2**63`;個別 index 陣列要降 int32 必須逐一 assert `max_value < 2**31 - 1`。
- evaluator 欄新增「M3 T1 後」一欄(§4.2 的成對 dtype 降級),用來說明**為什麼 M3 T1 不足以解除 11.3M 的封鎖**。

### 2.2 加總帳 = **保守上界**,不是可行性判定(Codex #2)

v1 在此把三個 standalone 峰值相加後,一邊用 `sum < 21.7` 宣告 11.3M 可行、一邊用 `sum > 21.7` 宣告 27.7M 被 GPU 擋。**兩個方向都不成立**:

- **相加是很鬆的上界。** 三個峰值出現在不同時點,實際峰值只會更低 ⇒ `sum > capacity` **不能**推出 OOM。
- **相加也可能低估。** 真實 driver 在 placement 前就同時建好 `GpuEvalContext`(`run_placement_io.py:50`)與 `IoTerm`(`:63`),兩者的靜態 buffer 跨整個 GP 存活(`:93` 才建 placer),再加上 allocator 的 reserved 與碎片 ⇒ `sum < capacity` **不能**推出可行。
- **正確帳式:`peak_device ≈ resident(t) + max_p transient_p`**,其中 `resident(t)` 是該時點所有仍存活的靜態 buffer,`transient_p` 是 phase p 的暫存峰值。

| 級 | 保守上界(GP + IoTerm + evaluator,T2 後) | 對 21.7 GiB | **判定** |
|---|---:|---|---|
| 3.1M | 1.75 + 0.94 + 1.3 = **4.0 GB** | 餘裕充足 | 上界即已通過 ⇒ 可行 |
| 6.2M | 3.07 + 1.88 + 2.5 = **7.5 GB** | 餘裕充足 | 上界即已通過 ⇒ 可行 |
| 11.3M | 6.73 + 3.44 + 4.5 = **14.7 GB** | 上界餘裕 32% | 上界通過 ⇒ 可行(仍須 T2b 記錄實際 resident/transient 分解) |
| 12.3M | 7.3 + 3.77 + 4.9 = **16.0 GB** | 上界餘裕 26% | 上界通過 ⇒ 可行 |
| 27.7M | 12.6 + 8.47 + 10 = **31.1 GB** | 上界超出 43% | **未定。上界超標不構成 blocker;由 T2b/T9 的 full-lifetime spike 判定** |

**判定規則(v2 定案,取代 v1 的相加裁決):**

1. **可行**:full-lifetime run(或全元件共常駐的 spike)完成且 `device_used_gb − device_baseline_gb ≤ 0.9 × 21.7 GiB`。
2. **不可行**:full-lifetime run 實際 OOM,或 T2b 分解顯示 `resident + min_p transient_p > 21.7 GiB`(即連最小 phase 都塞不下)。
3. **其餘一律「未定」**,不得寫進裁決表;27.7M 目前在此區。

### 2.3 被否掉的選項(不變)

| 選項 | 否掉理由 |
|---|---|
| 維持 spec §7「5M = MemPool 半 cluster / 3×Group」 | (a) `mempool_group` 實測 3.08M 不是 5M,「3×Group」= 9.2M 已是 10M 級;(b)「半 cluster」不是可交付物件。改用 tiler 產 1×2 = 6.16M |
| 1M 級用 `bigblue4` 取代 `superblue12` | bigblue4 保留為 M0–M3 連續性 case,但無 metal stack、無法接 Stage 2。**兩個都留,角色不同** |
| 10M 用合成資料取代 `mempool_cluster` | 真實 10M 已在本機跑通,且它是 tiler 唯一的校準靶(前提:T3a 通過) |
| 10M 直接上 H100 不在 L4 驗 | 被 §1.1 推翻;D6 原本就寫「L4 做全規模功能驗證」 |
| `bsg_chip` / `NV_NVDLA_partition_c` / `ariane` 入階梯 | 規模與 ISPD2015 重疊,不填空缺;保留為 Stage 2 校準語料 |

### 2.4 低信心處

- **M4-L1(中)**:`superblue12` 的 pins/nets 是推估 ⇒ T3 第一件事。
- **M4-L2(中)**:`mempool_cluster` 的 GP **品質**未驗(只證明跑得完)⇒ T8 的第一個 gate。
- **M4-L3(高風險)**:`mempool_cluster` 的 die 利用率 / `target_density` 未調(`mempool_group` 實測 `area_util=0.704`、`tile_wrap` 0.389;沿用 ISPD2005 的 0.835 會產生大量 filler)⇒ T3 必須為每個 NanGate45 case 從 `total_movable_node_area / free_area` 反推並揭露。
- **M4-L11(新增,中)**:§1.1 三個模型的係數不確定度與外推有效範圍完全未知 ⇒ T0b。

---

## 3. Q2 — 30M 測資合成

### 3.1 定案:配方 A′ = `mempool_group` 3×3 + 真實 cluster 統計導出的 glue nets,Bookshelf 域串流合成

**三個支撐(第 3 點在 v2 被降級為待證假設):**

1. **DREAMPlace 原生能把 LEF/DEF 轉成完整 Bookshelf。** `PlaceDB.write(params, path, place_io.SolutionFileFormat.BOOKSHELFALL)`(`$DP/dreamplace/PlaceDB.py:1010-1033`)→ `BookShelfWriter::writeAll`(`$DP/dreamplace/ops/place_io/src/BookshelfWriter.cpp:22-51`)寫出 `.nodes/.nets/.wts/.pl/.scl/.aux`;`writeNets`(`:100-138`)逐 pin 寫出相對 cell 中心的 offset ⇒ 轉換無損。`BOOKSHELFALL` 已在 pybind 導出(`PybindPlaceDB.cpp:60`)。
2. **Bookshelf 比 DEF 省 3 倍以上。** bigblue4 實測 `.nets` 345.3 MB / 8.73M pins = 39.6 B/pin;`.nodes` 15.0 B/node;`.pl` 16.5 B/node ⇒ 27.7M cells / 108.2M pins 約 5.2 GB(DEF 域要 27 GB)。磁碟 11 TB,無壓力。
3. ~~「`11,310,807 / 3,077,669 = 3.67` ⇒ cluster 就是 4 個 group」~~ **v2 撤回這個推論(Codex #7)。** cell-count ratio 只能提示「可能有四份階層」,不能證明四份與 standalone group 同構,也排除不了 synthesis pruning、共用 top-level 邏輯、不同參數化或重複基礎設施。**而且這個比值本身就在反駁 v1**:若真是 4 份,每份平均 `11,310,807/4 = 2,827,702` cells,比 standalone group **小 8.1%**。⇒ 改為 **T3a blocking hierarchy gate**(§3.1a),通過前不得產 `cluster_stats.json`、不得啟動 T4/T6。

### 3.1a T3a — hierarchy gate(**blocking**)

輸入:`mempool_cluster` DEF 的 instance 名稱與座標、`mempool_group` 的 standalone 統計。輸出 `results/m4/bench/hierarchy_gate.json`。

| gate | 條件 | 門檻 | 不過的後果 |
|---|---|---|---|
| **G-A** 前綴覆蓋 | 存在恰好 4 個 instance 前綴,覆蓋 cluster ≥ 95% 的 cells | ≥95% | 階層假設死 ⇒ 退配方 B |
| **G-B** 規模一致 | 每個 group 的 cells 與 standalone group(3,077,669)相差 ≤ 15%,且 `Σ4 groups + n_top = cluster` 恆等 | ≤15% + 恆等 | 同上 |
| **G-C** 組成一致 | 每 group 的 cell-type(LEF macro 名)直方圖與 standalone 的 cosine similarity ≥ 0.95 | ≥0.95 | 同上 |
| **G-D** 幾何排列 | 報出 4 個 group 的 bounding box 與相對排列(2×2 / 1×4 / 其他),並據此定義「鄰接對」與「對角對」 | 必須可判定 | 距離核無法錨定 ⇒ 退配方 B |
| **G-E** 頂層殘量 | 顯式報 `n_top`(不屬任何 group 的 cells/nets/pins),並計入 30M 的帳 | 必須有數 | — |

**兩個由 T3a 產生的設計約束(v1 沒有):**

- **正規化**:若 G-B 顯示真實 group 比我們的 tile 小 8%,則跨 group 的 net 數不可直接搬用,必須按**每 1M cells** 或**每單位邊界長度**正規化後再乘回 tile 規模,否則 3×3 的 glue 量會系統性偏低。正規化基準寫進 `cluster_stats.json`,並在報告揭露。
- **排列相依的可識別性**:G-D 若判定是 **2×2**,則只有 d=1、d=√2 兩種距離(見 §3.2 的 DoF 帳);若是 **1×4**,則有 d=1,2,3 三種距離 ⇒ 多出 1 個 residual DoF,核形狀首次可被弱檢驗。**這是 T3a 的附加價值,報告必須說明實際落在哪一種。**

### 3.2 合成器規格(`ioplace/bench/`)

```
(a) export_bookshelf.py   ← 一次性:PlaceDB.read(mempool_group) → write(BOOKSHELFALL)
(b) cluster_stats.py      ← T3a 通過後:從 cluster 量跨 group net 統計(正規化後)
(c) tile_bookshelf.py     ← 串流 tiler:讀 (a),寫 R×C 陣列,不在記憶體中持有完整 netlist
(d) glue_gen.py           ← 依 (b) 的統計 + §3.2 的距離核生成跨 tile nets
(e) rent.py               ← Landman–Russo Rent 量測(Mt-KaHyPar 1.6.2)
(f) verify_bench.py       ← §3.3 的 fit/holdout 指標
```

**tiler 的硬性規則(不變):** 陣列 `R×C`,tile `(i,j)` 前綴 `t{i}_{j}/`,座標平移 `(i·W, j·H)`;完全 abutted、無 channel;`.scl` row 複製並 assert 無重疊無縫隙;fixed macro 照樣平移並保持 fixed;逐 tile 逐行串流寫出(峰值 host ≈ 單 tile 名表 ≈ 1 GB);單一 `--seed` + `manifest.json`(來源 sha256、seed、R、C、核參數、**實際 glue net/pin 數**、輸出各檔 sha256)。

**glue net 模型(v2 重寫,Codex #5):**

```
從 (b) 取得(正規化後的)真實跨 group 統計:
    lambda_adj   = 每一對「鄰接」group 之間的 net 數(按 tile 規模正規化)
    lambda_diag  = 每一對「對角」group 之間的 net 數(同上)
    deg_hist     = 這些 net 的 degree 分布
    pinshare     = 每條 net 在各 group 的 pin 數分布
    iface_dist   = 跨 group pin 所在 cell 的「到 group 邊界距離」分布

距離核(對所有 d 有定義,單參數):
    phi(d; alpha) = d^(-alpha),  phi(1) = 1
    E[glue nets between tiles u,v] = lambda_0 * phi(dist(u,v); alpha)

參數由閉式 method-of-moments 解出(**沒有 grid search**):
    alpha   = log(lambda_adj / lambda_diag) / log(sqrt(2))     (2x2 排列)
    lambda_0 = lambda_adj                                       (phi(1)=1)

每條 glue net 的 degree 與 pin 分配從 deg_hist / pinshare 抽樣;
pin 的 tile 內落點從該 tile 內「原本就是跨 group net 的 pin」所在 cell 抽樣(保留 iface_dist)。
```

**可識別性帳(必須逐字進報告):**

- 2×2 陣列只有 d=1(4 對)與 d=√2(2 對)兩種距離;真實 cluster(若 G-D 判定為 2×2)同樣只給 `lambda_adj`、`lambda_diag` **兩個**獨立觀測量。
- 模型有 `lambda_0`、`alpha` **兩個**參數 ⇒ **恰好識別,0 residual DoF ⇒ 2×2 無法驗證核的形狀,只能複現它自己被餵進去的兩個數**。
- v1 的 `γ_far` 更糟:它對所有 2×2 觀測量的導數為 0(2×2 不存在 d=2),卻只在被外推的 3×3 才首次生效 ⇒ **完全不可識別**,v2 刪除。
- ⇒ **3×3 的核形狀是一個宣告的假設,不是校準結果。** 補償措施:T7 必須用三種核各產一次跨 tile net 計數並發表**系統性不確定度帶**:
  - K1 冪律 `d^(-alpha)`(主線),
  - K2 指數 `exp(-alpha'(d-1))`,`alpha'` 由同一組 `lambda_adj/lambda_diag` 匹配,
  - K3 截斷 `phi(d)=0 for d>sqrt(2)`(= 明示不模擬 long-range connectivity)。
  三者的 3×3 glue net 總數若相對全距 > 20%,則**主線 bench 額外附一份 K3 變體**作為明示下界,報告同時引用。

### 3.3 驗收:fit 指標 vs holdout 指標(Codex #6)

`verify_bench.py` 輸出 `results/m4/bench/verify_<name>.json`。**fit 指標不判定通過與否(它們的值是被構造出來的);判定只由 holdout 指標決定,且只評一次。**

**結構檢查(硬性,兩類 case 都要過):**

| # | 條件 | 門檻 |
|---|---|---|
| **V0** 結構完整性 | `.aux` 指到的檔全在;`NumNodes/NumNets/NumPins` 與實際行數一致且**等於 manifest(含 glue)**;無 self-loop;degree ≤ 1 的 net 比例不高於來源;每個 node 至少被一條 net 參照或明列 unconnected;row 覆蓋無重疊無縫隙 | 全部 true |
| **V1** DREAMPlace 可讀 | `PlaceDB.read` 成功且 `num_movable/num_nets/num_pins` 等於 manifest | 全等(27.7M 因 host RAM 只在 H100/大記憶體機驗;L4 驗 1×2 與 2×2) |

**fit 指標(報告用,不判定):** `lambda_0`、`alpha`(含 ≥5 seeds 的 bootstrap 95% CI)、glue degree 直方圖、pinshare 分布、iface_dist ——這些量按建構法就會對上真實統計。

**holdout 指標(預先登錄門檻,只評一次,不得因失敗重掃):**

| # | 指標 | 門檻 | 為什麼是 holdout |
|---|---|---|---|
| **H1** Rent 曲線形狀 | 2×2 與真實 cluster 的 `p`:`|p_syn − p_real| ≤ 0.05`,且兩者皆落在 `[0.55, 0.80]` | `p` 由**遞迴二分的整體結構**決定,不是被餵進去的 pair count |
| **H2** per-level cut 直方圖 | 各層(`10³ ≤ B ≤ 10⁶`)的平均 cut 相對誤差 ≤ 25% | 同上,且比單一 `p` 更難被兩個參數湊出 |
| **H3** 全網 degree 分布 | 2×2 vs cluster 的 net-degree KS ≤ 0.05;用 `io_term.DEG_BUCKET_EDGES` 同一套切法,逐桶相對誤差 ≤ 20% | 只有 glue net 的 degree 被擬合,**全網**分布受複製結構支配 |
| **H4** interface-cell 空間分布 | 跨 tile pin 的「到 tile 邊界距離」分布 KS ≤ 0.10 | 抽樣保留的是**來源 group 的**分布,跨 tile 幾何是新的 |
| **H5** K-grid λ 口徑 | K=16 grid 下 2×2 的 `hard_lambda_sum / n_nets` 與 cluster 的比值落在 `[0.7, 1.4]` | v1 的 `[0.5,2.0]` 過寬,收緊 |

**H5′(原 V5,降級):** 3×3 的跨 tile net 佔比定義為 `C(m) / (m·N_tile + C(m))`,其中 `C(m) = Σ_{u<v} lambda_0·phi(dist(u,v))`。這是**建構法的算術恆等式**,`verify_bench.py` 只做自檢(生成結果與公式相符,誤差 ≤ 0.1%)並**報告**隱含的 Rent 樣指數;v1 宣稱的「符合 `(R·C)^p` ±20%」量綱不成立(cut 若隨 `m^p`、總 nets 隨 `m`,比例走的是 `m^(p−1)`),**刪除該門檻**。

**Rent 量測的實作定案:** Landman–Russo 遞迴二分,用 `mtkahypar==1.6.2` 遞迴切到 `2^L` 個 block,每層記 `(平均 block 大小 B, 平均外部 terminal 數 T)`,在 `10³ ≤ B ≤ 10⁶` 的層做 `log T = log t + p·log B` 線性回歸,附 bootstrap 95% CI。
**M4-L4(低信心):Mt-KaHyPar 在 12.3M/27.7M hypergraph 上的時間與記憶體未知。** 緩解:(a) H1/H2/H3 只在 12.3M 上做;(b) 27.7M 改用「取樣子超圖(連續 2M cell 視窗)+ 頂層精確計數」的混合估計;(c) 若 12.3M 就 OOM,退回幾何遞迴切分(median x/y bisection)量 Rent,並在報告註明「幾何 Rent 是上界,非最佳分割 Rent」。

### 3.4 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| `mempool_cluster` ×3 幾何複製 | (a) DEF 域 27 GB、讀回 ~190 GB host;(b) 轉 Bookshelf 後 33.9M cells / 132M pins ⇒ 比 3×3 group 更糟;(c) 失去校準靶 |
| 把 `spike_10m.py` 的合成拓撲升格為 benchmark | pin→node 是 `rng.integers` 均勻抽樣(`spike_10m.py:57`),無空間/階層 locality ⇒ Rent p→1.0,違反 spec §7 的 `p≈0.6–0.75`;且均勻位置讓 crossing 數暴增(4.8M nets 合成 case `io_count=29.8M` vs 真實 bigblue4 K=16 的 100,333)⇒ 系統性高估 evaluator 成本。保留其**元件級 spike 合成器**的角色(T9) |
| 配方 C:ArtNet 直接生成 30M | 需 clone GitHub,本機網路不通。future work |
| 配方 B(純 Rent 差額補 glue) | 保留為 **fallback**:T3a 任一 gate 不過,或 §3.2 的前綴切不出 group 邊界時啟用;掃 `p∈{0.60,0.65,0.70}` 補足 `t·(K·g)^p − K·t·g^p`。有 spec §7 點名的循環論證風險 ⇒ 報告必須明寫「30M 的連通性是擬合出來的」 |
| **把 30M(或任何合成級)當品質宣稱的 benchmark** | **明確非目標,且 v2 由 artifact 結構強制**:每筆結果帶 `benchmark_kind`;`quality_real_cases.md` 只收 `superblue12`/`bigblue4`/`mempool_group`/`mempool_cluster`;`scaling_synthetic_cases.md` 的 schema **禁止** `Δio/Δft/Δhpwl/winner/pareto` 欄位;`scripts/m4_report_lint.py` 在 T11 的 pytest 中把違規變成紅燈 |

---

## 4. Q3 — evaluator 規模斷點

### 4.1 逐項張量帳(解析,非擬合)

以 `mempool_cluster` 的 12.71M nets / 43.95M pins / K=32 計:

| 項 | 位置 | 形狀 / dtype | @2.4M nets | @12.71M nets |
|---|---|---|---:|---:|
| **pin one-hot** | `evaluator_gpu.py:351` `pin_bits = self._one_hot_planes(pin_rid)` | `(P,K)` int64 | 2.51 GB | **11.25 GB** |
| pin 累加器 | `:352` `pin_bit_acc` | `(E,K)` int64 | 0.61 GB | 3.25 GB |
| passed 累加器 | `:413` `passed_bit_acc` | `(E,K)` int64 | 0.61 GB | 3.25 GB |
| MST edge 陣列 | `:385` `_batch_mst` 的 `torch.cat` + bucket lists | 3×`(M,)` int64 ×2 份 | 0.36 GB | 1.9 GB |
| edge 端點座標 | `:392-397` `xa/ya/xb/yb` fp64 + `ax/ay/bx/by` int64 | 8×`(M,)` | 0.47 GB | 2.5 GB |
| segment 描述 | `:400-406` `h_row/h_lo/h_hi/v_col/v_lo/v_hi` | 6×`(M,)` int64 | 0.36 GB | 1.9 GB |
| pair-demand key | `:326` append + `:332` `torch.cat(...).cpu()` | O(總 crossing 數) | 0.12 GB | 位置相依(真實 case 小、合成 case 可爆) |
| segment chunk 暫存 | `:177` `_seg_chunk_budget = 8e6` | 有界 | ~0.45 GB | ~0.45 GB |
| 其他 `(E,)`/`(P,)` | — | — | ~0.25 GB | ~1.3 GB |
| **合計** | | | ~5.7(實測 6.74) | **~26–36 GB** |

M3 draft §7 R2 只點名兩個 `(E,K)` 累加器(6.5 GB),**漏掉最大的 `(P,K)` one-hot 與完全未分批的 MST/segment 陣列**。

### 4.2 M3 T1 與 M4 T2 的分工邊界(v2 依 Codex #9/#10 改寫)

**切線改為:「只改 dtype、不改張量形狀與迴圈結構」vs「改形狀/迴圈結構」。**

| | **M3 T1(dtype 成對降級)** | **M4 T2(結構性 streaming)** |
|---|---|---|
| 範圍 | ① `_one_hot_planes`(`:205-208`)與 `_bit_planes`(`:200-201`)加 `dtype` 參數,**直接產生 int8**;② `pin_bit_acc`(`:352`)、`passed_bit_acc`(`:413`)改 int8 —— ①② **必須成對**,PyTorch 2.8 的 `scatter_reduce_` 要求 `self.dtype == src.dtype`(實測錯誤訊息:`scatter(): Expected self.dtype to be equal to src.dtype`);③ `pair_key_chunks` 改 per-chunk `torch.bincount(key, minlength=K*64)` 累加,刪掉 `:332` 的 `torch.cat(...).cpu().numpy()` | ④ 消滅 `(P,K)` 的**全量物化**(K 分塊,或 `unique(net·64+rid)` 後 `index_add(1<<rid)` 的精確 OR);⑤ **MST edge / segment 的批次化**(以 edge 為單位分批,增量累加 `per_net_crossings`/`passed`/`pair_demand`);⑥ 逐欄位 assert 後的 int32 索引;⑦ 批次大小為建構參數 |
| 收益 @12.71M | `(P,K)` 11.25→**1.41 GB**、2×`(E,K)` 6.5→**0.81 GB** ⇒ 省 **≈15.5 GB**,evaluator 35.6→**≈20 GB**,外加移除一次 CPU 同步 | 再省 ~15 GB ⇒ **≈4.5 GB(推估,待 T2 實測 + T6b count freeze 重算)** |
| 風險 | 低,但**不是 v1 說的「極低、只動兩個累加器」**:改動同時觸及 source 與 accumulator;若先產 int64 再 `.to(int8)`,兩份張量短暫共存、峰值不降 ⇒ **必須直接產 int8** | 中(改動 MST/segment 主迴圈) |
| **關鍵結論** | **M3 T1 做完仍是 ≈20 GB > 21.7 GiB 扣掉 GP/op 後的餘裕 ⇒ 11.3M 依然被擋。T2 仍是 M4 全部 ≥5M 實驗的硬前置。** | |

**已實測的 dtype 契約(2026-08-13,L4 / torch 2.8.0+cu128,`/tmp/probe_v2_dtype.py`,T0 遷入為 `probe_scatter_dtype.py`):**

- int8 source + int8 accumulator 的 `scatter_reduce_(reduce="amax")` 在 CUDA 上**可用**,結果與 int64 版**逐位元相同**(0/1 上 `amax` ≡ OR)。
- int8 accumulator + int64 source **直接報錯**(見上)⇒ 成對修改是必要條件,不是風格選擇。
- `_pack_bits` 的 `bit_planes * self._pow2_k` 在 int8 × int64 下**自動升位為 int64**,packed 值與原版相同 ⇒ `:215` 不需改。
- **K 的硬上限**:`_pow2_k` 用有號 int64,`1<<63 = INT64_MIN` ⇒ **K=64 在契約外**。⇒ 加 `assert self.k <= 32`(spec §8 的實驗矩陣是 K ∈ {8,16,32}),測試覆蓋 **K ∈ {1, 8, 32}** + multi-chunk + 空 bucket + legacy 欄位 exact regression。**修正 Codex #9 建議的 K=64 測試**——那不是應該通過的情形。

**T2 的驗收改 field-specific(Codex #10):**

| 欄位類別 | 具體欄位 | 跨 batch ∈ {1e6, 8e6, 全部} 的要求 |
|---|---|---|
| 整數(語意精確) | `io_count`、`ft_count`、`hard_lambda_sum`、`large_lb`、`per_net_crossings`、`per_net_ft`、`per_net_lambda`、`pair_demand` | **逐位元 exact** |
| 結構自檢 | MST edge 總數、per-net edge 數 | **逐位元 exact**(防容差掩蓋丟邊) |
| 浮點(順序敏感) | `tree_wl`(`:394` 單次 float64 `sum`)、`hpwl` | **預先固定 `rel ≤ 1e-12`** |
| 對 `evaluator_ref` | 既有契約 | 不變(`rel ≤ 1e-5`,`tests/test_evaluator_gpu.py:41`) |

**證據(`/tmp/probe_v2_treewl2.py`,T0 遷入為 `probe_reduction_order.py`):** 7,654,321 個 heavy-tail float64 相加,batch=3e6 與 8e6 **恰好**與全量相同,batch=1e6 與 999,983 **不同**(rel 1.148e-16)⇒ **「三種 batch 逐位元相同」在 v1 是碰巧會過的驗收,不能寫成契約**;1e-12 相對誤差比 ulp 級漂移(~1e-16)寬 4 個數量級,又比「掉一條 edge」(4×10⁷ 條時約 1e-8)嚴 4 個數量級,有真正的鑑別力。

### 4.3 T2 之後的記憶體推估(**推估,由 T2 驗收 + T6b count freeze 取代**)

`mempool_cluster` 12.71M nets / 43.95M pins / K=32:`(E,K)` int8 ×2 = 0.81 GB;pin 座標 fp64 = 0.70 GB;`(E,)` 欄位 ~10 個 = 0.9 GB;edge 批次緩衝(8M edges)≈ 1.0 GB;grid/Ph/Pv(lattice 512)= 6 MB ⇒ **≈ 4.5 GB**。27.7M 合成 case(31.54M + g₉ nets / 108.24M + gp₉ pins)⇒ **≈ 10 GB(推估,g₉ 未定)**。

### 4.4 Steiner 欄位(M3 新增)在 30M 的成本

| 欄位 | 形狀 | @31.54M nets |
|---|---|---:|
| `per_net_steiner` | `(E,)` int32 | 126 MB |
| `per_net_home` | `(E,)` int8 | 32 MB |
| `D`、`ell`、`ecc_max` | `(K,K)`/`(K,)` 常數 | < 10 KB |
| Λ≤3 closed form | gather,無新張量 | 0 |
| **Λ∈[4,8] Dreyfus–Wagner DP 表** | `(B, 2^Λ, K)` | **必須分批** |

**硬性護欄:** `B · 2^Λ · K · 4 B ≤ 512 MB` ⇒ `Λ=8` 時 `B ≤ 128k`。M3 draft §2.2 實測 Λ 分布極偏(adaptec1 k16 的 Λ≥4 只有 517/216,932 = 0.24%),外推到 31.5M nets 約 76k nets,一批就夠 ⇒ 成本可忽略,**但沒有護欄就是定時炸彈**(K=32 且 Λ 分布較厚時瞬間要 32 GB)。**NanGate45 語料的 Λ 分布未量(M4-L10)⇒ T3 順手量。**

### 4.5 被否掉的修法

| 選項 | 否掉理由 |
|---|---|
| packed int64 bitmask 直接 `scatter_reduce(amax)` | **數學錯誤**:兩個 packed mask 的 `max` ≠ `OR`(`0b10` vs `0b01`:max=2、OR=3)。只有「每列僅一個 bit」才成立——那是 `pin_bit_acc` 的情形但**不是** `passed_bit_acc`(一個 segment 會經過多個 region) |
| K 分塊(重走 K/c 趟 segment) | 對 `pin_bits` 可行且便宜;對 `passed_bit_acc` 要重跑 `_process_segments`(整個 lattice walk)K/c 次,計算 ×(K/c)。改用 **edge 批次化**:一趟 walk、增量累加 |
| 只降 dtype、不做 edge 批次化 | 只救 K-dependent 部分。K-independent 項 0.85 GB/M-net ⇒ 31.5M nets 仍要 26.8 GB(§4.2 的 M3 T1 後 ≈20 GB @12.7M 亦同源)|
| evaluator 改跑 CPU | `evaluator_ref` 在 bigblue4 上約 450 s 一次;GP 內圈每 50 iteration 呼叫一次 ⇒ 不可能 |
| 降 K 到 8/16 迴避 | 治標,且 spec §8 要求 K ∈ {8,16,32};K=16 只把 35.6 降到 23.2 GB,L4 仍 OOM |

---

## 5. Q4 — 全流程瓶頸與本機驗證策略

### 5.1 三個候選瓶頸與各自的判定規則

| # | 瓶頸 | 量化 | 11.3M | 27.7M | **判定規則** |
|---|---|---|---|---|---|
| **P1** evaluator 記憶體 | 35.6 GB @12.71M nets K=32(解析帳 + 五點擬合一致) | **擋**(T2 後解除) | 擋(T2 後 ~10 GB) | 已由 §4.1 的解析帳確立,不依賴外推 |
| **P2** IO op runtime | 30M spike 26.7 s / fwd+bwd(待 T0 重現);`io_term.py:331` 在 `P′ ≫ chunk_budget` 時 `k_chunk` 恆為 1 ⇒ K 趟掃 pin 是結構成本 | 否(單臂推估 ~2.6 h) | 否(單臂推估 3.3–5.4 h) | 推估值由 §1.3 的**多預測子**模型重算(T0/T9),不再用 84 B/pin 與 181 ns/pin |
| **P3** host RAM | v1 由 `0.88 KB/pin`(**Bookshelf 僅一點**)外推 27.7M 得 95–105 GB | 否(cluster 實測 62.59 GB < 125 GB) | **未定** | **三分規則**:多預測子模型在 2×2(12.3M)holdout 的 95% PI —— 下界 > 可用 RAM ⇒ 判定「擋」;上界 < 可用 RAM ⇒ 判定「可行」;**其餘為「未定」,只能由實測解決** |

**B2 的放大效應(修正後):** 每個 callback 的 op 成本是 `1 + n_nonempty_buckets` 趟 fwd+bwd(現況 8),**外加**一趟 no-grad softmax pass 與 callback 內的 WL backward ⇒ **實際 wall-time 大於 8×**,精確值由 T1 直接量測。11.3M/27.7M 的診斷成本因此至少是 v1 估的 64 s / 157 s 每 callback。**T1 已落地 `--diag-every N` / `--no-diag`。**

### 5.2 單臂 wall-time 推估(**全部標為未驗證**,`ρ` 啟用約佔 600 iteration)

| case | GP | op(600 iter) | evaluator(20 次) | diagnostics(取樣後) | 合計 |
|---|---:|---:|---:|---:|---:|
| `mempool_group` 3.1M | 92 s(待重現) | 0.36 h | ~1 min | ~2 min | ≈ 0.45 h |
| `mempool_cluster` 11.3M | 362 s(待重現) | 1.33 h | ~4 min | ~7 min | ≈ 1.6 h |
| 27.7M 合成 | **未定(v1 的 890 s 已作廢)** | 3.27 h | ~10 min | ~15 min | **未定** |

⇒ 27.7M 的 GP 時間必須由 §1.1 的 runtime 模型(T0b 建立、T8 的 12.3M 為 holdout)給出,附 prediction interval;在那之前 27.7M 的總時間欄位一律留「未定」。

### 5.3 本機驗證策略(三層)

**第 1 層 — L4 全流程(GP + LG + op + evaluator + 報表),T2 之後可跑:** `superblue12`(1.3M)、`bigblue4`(2.2M)、`mempool_group`(3.1M)、group 1×2(6.2M)、**`mempool_cluster`(11.3M)**、group 2×2(12.3M)。⇒ spec §9 M4 的「10M 全流程」在 L4 交付得出來。

**第 2 層 — L4 元件級 spike(27.7M),不經 `PlaceDB`:** 用 `ioplace/bench/` 的串流 reader 直接建 `ioplace.netlist.Netlist`,量 IoTerm(+M3 的 S4)與 T2 後的 evaluator。**v2 強制要求(Codex #2):兩個元件必須共常駐並交錯執行 ≥3 個 iteration**,量的是 full-lifetime 峰值,不是兩次 standalone 峰值相加;並記錄 `device_baseline_gb`、`resident_gb`、`max_phase_transient_gb`。

**第 3 層 — H100 交接(27.7M 全流程):** 資源契約 + 登錄制預測(§6.3 E5)+ **T14 的實際執行與驗證**。

### 5.4 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| 「10M 只跑 spike、GP 縮到 5M」 | 被實測推翻:11.3M 的真實 GP 已在 L4 跑完(362 s / 6.9 GB,待 T0 重現) |
| 為了塞進 L4 而在 30M 降 K 到 8 / 降 bins 到 2048² | 省不了多少(op 峰值與 K 無關,受 `k_chunk·max(num_physical,P′)` 支配),且 host RAM 是另一道牆。不值得污染實驗矩陣 |
| T12(numpy-only PlaceDB shim)當主線 | 能把 30M 的 host RAM 降到 ~3 GB,但 GPU 仍要 ~31 GB(上界)⇒ 只是把「被 host 擋」換成「未定」。降為條件 task(觸發:H100 節點 host RAM < 128 GB)。實作面已探明:替換 `PlaceDB.initialize_from_rawdb`(`$DP/dreamplace/PlaceDB.py:499-616`)所填欄位並 stub `apply`(`:1133`;`BasicPlace.py:604/631` 的 `lefUnit/defUnit` 只在 routability 選項下走到) |
| 等 H100 才開始規模化 | 80% 的工作(T0–T7、T9)在 L4 就能完成並驗證 |

---

## 6. Q5 — 計時、記憶體 profile 與 exit 改寫

### 6.1 量什麼(欄位語意已按 Codex #14 更正)

| 類別 | 欄位 | 方法 | **已知侷限(必須寫進報告)** |
|---|---|---|---|
| Phase 牆鐘 | `t_read/t_initialize/t_gp/t_lg/t_eval_total/t_op_total/t_diag_total/t_total` | `time.perf_counter()`,phase 邊界 `torch.cuda.synchronize()` | — |
| Per-iteration | `iter_ms[]`(1000 floats)+ p50/p90/p99/max | 成對 `torch.cuda.Event`,`elapsed_time()` **run 結束後批次查詢** | — |
| op 分離計時 | `op_fwd_ms[]`、`op_bwd_ms[]`(profile 模式,每 `--time-every` 次記一次) | 同上,事件包住 `_IoFn.forward/backward` | — |
| evaluator | `eval_ms[]` | 同上 | — |
| GPU 記憶體 | 每 phase `peak_alloc_gb` / `peak_reserved_gb`,**phase 開頭 `reset_peak_memory_stats()`** | `max_memory_allocated/reserved` | **只重設 PyTorch allocator 統計**;不處理外部 CUDA 配置、reserved 碎片,也不證明前一 phase 的張量已釋放(§1.4 B1) |
| 裝置級記憶體 | **`device_used_gb`**(v1 誤稱 `device_peak_used_gb`) | 背景 thread 每 0.5 s 取 `mem_get_info()`,`used = total − free − device_baseline` | **是整張卡的用量,含他人 process** ⇒ 正式 memory run 必須 **exclusive GPU**;0.5 s 取樣**會漏短 transient**(T1b 用人工短峰 probe 量其漏檢率) |
| Host 記憶體 | **`host_rss_hwm_at_phase_end`**(v1 誤稱 `host_peak_rss_gb`) | `resource.getrusage(RUSAGE_SELF).ru_maxrss`,phase 邊界各取一次 | **`ru_maxrss` 是 process-lifetime HWM,不可 reset** ⇒ 序列單調不減,**不是 per-phase peak**;需要 per-phase 歸因時,該 phase 必須跑在 **child process**(T1b) |
| 規模元資料 | `n_movable/n_physical/n_filler/n_nets/n_pins/K/rtype/lattice/n_bins/k_chunk/n_active/n_dedup_pins/total_name_bytes` | 直接讀 | — |
| 導出係數 | `bytes_per_pin_gp`、`ns_per_pin_op`、`gb_per_mnet_eval` | 由上面算 | **只是單點比值,不是模型**;模型係數一律在 `model_fit.json`,附 CI 與殘差 |

**torch.profiler 的定位:** 只在 ≤3.1M 的 case、只開有界視窗 `schedule(wait=5, warmup=2, active=3, repeat=1)` + `profile_memory=True, record_shapes=True`,輸出 `trace.json.gz` + top-20 `key_averages`。**禁止**在 ≥6M 的 run 上開(trace 數 GB,且 `profile_memory` 會污染 per-iteration 計時)。profiler 的用途是**歸因**與 **E5 需要的 achieved-bandwidth 量測**,不是計時。

### 6.2 輸出格式與 schema gate

```
results/m4/profile/<case>__k<K>__<rtype>__<arm>.json    # §6.1 全欄位 + provenance + benchmark_kind
results/m4/profile/<case>__.../trace.json.gz            # 只有 <=3.1M
results/m4/scaling/model_fit.json                       # 三個模型的係數 + CI + LOO 殘差 + condition number
results/m4/tables/quality_real_cases.md                 # 只收真實 case,可含 Δio/Δft/Δhpwl/winner
results/m4/tables/scaling_synthetic_cases.md            # 只收合成 case,禁止品質欄位
```

**每筆 JSON 的必備 provenance(RESULT GATE,§7.0):** `run_id`(uuid4)、`status`、`repo_commit`、`dp_commit`、`command`、`hostname`、`gpu_name`、`seed`、輸入檔 `sha256`、`benchmark_kind`、`device_baseline_gb`。

`quality_real_cases.md` 欄位:`case | cells | nets | pins | K | rtype | arm | io_mst | ft_mst | io_rg | ft_rg | hpwl | Δio% | Δft% | Δhpwl% | num_unplaced_cells | final_overflow | t_total | t_gp | t_op | t_eval | gpu_peak | host_hwm | run_id`(`io_rg`/`ft_rg` 為 `ours@M3` 專屬,其餘臂留空並註明)。
`scaling_synthetic_cases.md` 欄位:`case | cells | nets(+glue) | pins(+glue) | K | arm | status | t_total | t_gp | t_op | t_eval | iter_ms_p50 | gpu_peak | host_hwm | run_id` —— **schema 明文禁止** `Δio%`、`Δft%`、`Δhpwl%`、`winner`、`pareto*`。

### 6.3 Exit 改寫

spec §9 M4 原文:「10M 全流程;30M 測資製作並跑通;H100 計時;記憶體 profile。exit:30M 端到端數小時內完成;全規模對照表」。原 exit 的問題:「30M 端到端」在只有 L4 的現實下無法驗證。

| # | 判準 | 在哪驗 | 具體門檻 |
|---|---|---|---|
| **E1** 10M 全流程 | **L4** | `mempool_cluster`(11.31M)完成 GP+LG+evaluator+報表,且:`Δhpwl` 相對其自身 flat ≤ +5%;`num_unplaced_cells == 0`;`final_overflow ≤ params.stop_overflow`;`legalization_status == "success"`;GPU `device_used_gb ≤ 20 GB`;單臂 wall-time ≤ 4 h。至少 K ∈ {16,32} 各一組 `flat` 與一組 `ours@M2`。**每格通過 §7.0 RESULT GATE** |
| **E2** 30M 測資 | **L4** | (a) 27.7M Bookshelf 產出、V0 全綠、manifest sha256 與 **實際 glue 計數** 齊全、H5′ 自檢綠;(b) 12.3M 的 V0/V1 + **H1–H5 holdout 全綠**(且 T3a 五道 gate 已通過);(c) 27.7M 的**共常駐** full-lifetime 元件級 spike 在 L4 完成並記錄 `resident/transient/device_used` 分解 |
| **E3** 全規模對照表 | **L4** | `quality_real_cases.md` 涵蓋 {1.3M, 2.2M, 3.1M, 11.3M} × K{16,32} × {flat, ours@M2};`scaling_synthetic_cases.md` 涵蓋 {6.2M, 12.3M} × K{16,32};每格可追到唯一 `run_id`,report linter 綠 |
| **E4** 記憶體 profile | **L4** | GP / evaluator / op 三個模型各以 **T0b 的 factorial 設計**重新擬合,`model_fit.json` 記錄 design matrix、condition number、係數 95% CI、LOO 殘差與 6.2M/12.3M **holdout 殘差**;**且 §1.4 的 B1/B2/B3 三個量測問題全部處置並在報告揭露**(B1 附 T1b 的四臂 A/B 結果) |
| **E5** H100 forecast(**登錄制,非 L4 判定**) | 交接文件 | 見下 |

**E5 重寫(Codex #15):**

E5 **不是**「跑得完」的門檻,而是一個**在取得 H100 之前就凍結、之後可被單次實驗證偽的預測**。M4 的 exit 條件是「預測已登錄且其輸入可稽核」;**證偽由 T14 執行**。

```
模型:   T_H100 = sum_p ( T_L4,p / s_p )        (phase 級 Amdahl 分解)
phases: read/parse(CPU) | initialize | GP | IO-op fwd+bwd | evaluator | LG   (diagnostics 關閉)

每個 s_p 的來源必須逐項登錄:
  - read/parse:  s = 1.0(純 CPU、單執行緒 parser;除非目標機 CPU 型號不同,另行宣告)
  - GP / op / evaluator:
      在 L4 用 torch.profiler 量該 phase 的 achieved bandwidth
        AB = (解析計數的 bytes moved) / (kernel time)
      若 AB >= 0.6 * peak_L4(300 GB/s)  -> 判為頻寬受限, s = BW_H100 / BW_L4
      否則                                -> 判為計算/atomic/latency 受限,
                                             s = 明確宣告的 FP64 吞吐比或 SM 數比,並註明理由
```

**登錄內容(T10 凍結,附 sha256):**

| 項 | 內容 |
|---|---|
| 硬體 SKU | **H100 SXM5 80GB HBM3**,3.35 TB/s peak,sm_90,CUDA 12.8,persistence mode on,預設時脈(非 MIG、非降頻);若實際機器不同 ⇒ 預測**無效**,須重登錄而非事後放寬 |
| 預測 1 | 27.7M 單臂 wall-time 的 **point estimate `T̂`**,由上式計算(輸入表逐 phase 列出) |
| 預測 2 | GPU `device_used_gb` 的 point estimate |
| 預測 3 | host RSS 的 point estimate(§1.1 多預測子模型) |
| 判定 | wall-time 落在 **`[0.7, 1.3] × T̂`** 內 ⇒ 模型成立;GPU peak `[0.85, 1.15]×`;host RSS 用模型的 95% PI。**任一項落外 ⇒ 模型被證偽**,必須發表重擬合後的模型與差異歸因。**沒有第二層容忍帶**(v1 的 `[0.3×,3×]` 刪除:它的實際通過帶是 0.15–4.5 h,30 倍寬,無法區分 2× 與 8× 的性能模型) |

**v1 的 `[0.5 h, 1.5 h]` 數字一併作廢**:它建立在已作廢的「27.7M GP ≈ 890 s」與「頻寬比 × 0.5–0.7」之上。**v2 不在草案階段寫死任何數字**;`T̂` 由 `scripts/m4_forecast.py` 從 T8/T9 的實測 phase 表計算,在 T10 凍結。

### 6.4 H100 交接契約(T10 的交付物)

```
docs/handover/h100-30m-runbook.md
  |- 資源契約:GPU = H100 SXM5 80GB(SKU 釘死);host RAM >= 192 GB;磁碟 >= 100 GB;exclusive GPU
  |- 環境重建:docs/dev-env.md 的完整 build 流程 + 兩個必要檢查(CMAKE_CXX_ABI=1、CUDA_ARCH_FLAGS 加 sm_90)
  |- 資料:manifest + sha256(30M Bookshelf 在目標機**重新生成**,不搬 5.2 GB;tiler 串流,重生成約 10 min / 1 GB)
  |- 單一指令:scripts/m4_run.sh <case> <K> <rtype> <arm>
  |- 登錄預測:results/m4/forecast/h100_prediction.json(sha256 凍結)+ scripts/m4_check_prediction.py
```

**設計原則:H100 上不做任何設計決策。** 所有參數、gate、預測都在 L4 定死,H100 只執行與比對(T14)。

---

## 7. Q6 — Task 分解與依賴序

規約沿用 M2 §11 / M3 §8:每 task 一次 TDD 迴圈;測試一律 `$DP/.venv312/bin/python -m pytest`;新程式碼全在 `ioplace/`;DREAMPlace 原始碼不動。

### 7.0 RESULT GATE(所有實驗 task 的共同驗收前提,Codex #16)

任何「產出 JSON 即算過」的驗收一律改為引用本節。實作於 `ioplace/bench/result_gate.py`,並有對應 pytest(對故意損壞的樣本必須全部攔下):

1. **status**:`status == "success"`(失敗 run 必須留檔但不得計入表格);
2. **schema**:`PROFILE_SCHEMA_FIELDS` 全數存在且型別正確,`benchmark_kind ∈ {real, synthetic}`;
3. **finite**:所有數值欄位非 NaN/Inf(可空欄位須明列於 schema);
4. **provenance**:`run_id`、`repo_commit`、`dp_commit`、`command`、`hostname`、`gpu_name`、`seed`、輸入 `sha256`、`device_baseline_gb` 齊全,且 `repo_commit` 與當前 HEAD 的關係在報告中可解釋;
5. **唯一性**:同一份表格中沒有兩格引用同一個 `run_id`;
6. **非空**:`len(iter_ms) >= 0.9 × n_iter_expected`、`n_nets > 0`、`t_total > 0`。

### 7.1 Task 表

| Task | 類型 | M3-dep? | 內容 | 驗收 |
|---|---|---|---|---|
| **T0** 探針遷入 | 純軟體 | 否 | 把 §1/§4.2 的探針遷入 `ioplace/diagnostics/probes_m4/`(`probe_gp_memory.py`、`probe_eval_scaling.py`、`probe_spike_scale.py`、`probe_host_rss.py`、`probe_bookshelf_export.py`、`probe_macro_stats.py`、**`probe_scatter_dtype.py`**、**`probe_reduction_order.py`**),輸出 `results/m4/probes/*.json` | §1 每個數字可追溯且**重現誤差 ≤ 5%**;不符即回頭改本文;每份輸出過 RESULT GATE |
| **T0b** **模型紀律** | 純軟體+實驗 | 否 | 固定 case(`mempool_group`)factorial 掃描:bins ∈ {1024²,2048²,4096²} × target_density ∈ {0.70,0.835,0.90}(9 runs,GP-only,det=1),**讓 `N_total`/`N_pins`/`n_bins` 去共線**;輸出 design matrix(含單位)、condition number、OLS 係數 + 95% CI、leave-one-case-out 殘差;另建 runtime 模型 `n_iter × per-iteration + fixed`(per-iteration 由 T1 的 CUDA-event 直方圖來) | `results/m4/scaling/model_fit.json` 含上述全部欄位;**condition number 與係數 CI 必須顯示於 M4 報告**;模型在 6.2M/12.3M 的 holdout 檢驗排在 T8/T9(未通過前 §2.1 的推估欄一律標「未驗證」) |
| **T1** profile 基建(**已落地保守版**) | 純軟體 | 否 | `ioplace/profile.py`:CUDA-event 計時器、`PhaseTimer`(phase 開頭 `reset_peak_memory_stats`)、`DeviceMemSampler`(`mem_get_info`,0.5 s)、§6.2 JSON schema(含 `benchmark_kind`、provenance 欄位)。driver 端:B2 **只加** `--diag-every N` / `--no-diag`(**不做 reuse**),B3 `--budget-gb`。**欄位語意以 `host_rss_hwm_at_phase_end` / `device_used_gb` 為準,並在 docstring 明記其侷限** | 小 case 上 phase 峰值互不污染;取樣 thread 不改變結果的決定性測試;schema 完整性測試;**文件明記「reset 非充分,per-phase/per-run 歸因由 T1b 定案」** |
| **T1b** **量測有效性** | 純軟體+實驗 | 否 | (a) 四臂 A/B probe:fresh subprocess / same-process / reset-only / explicit teardown+GC,記錄每臂 run-start/end 的 allocated+reserved,判定 B1 成因;(b) 正式 ablation 改**每臂一個 subprocess**(或以 (a) 證明 teardown 後 baseline 回到 ±2% 容忍);(c) 需獨立歸因的 phase 用 **child process** 取 peak RSS;(d) **exclusive-GPU 協定**:run 前檢查全卡 `used − baseline < 0.5 GB`,否則標 `status="contaminated"`;(e) sampler 精度:人工造 0.05/0.1/0.5 s 短峰,量 0.5 s 取樣的漏檢率並寫進報告;(f) §1.4 B3 的 `budget_source`/`baseline_reserved_gb` 欄位與硬體契約 assert | (a) 的四臂數據齊全且能區分「cumulative HWM」與「retained tensors」;(d) 對污染情境的測試;(e) 漏檢率有數;(f) 傳超過契約的 `--budget-gb` 必須 assert 失敗 |
| **T2** **evaluator streaming** | 純軟體 | 軟相依 | §4.2 的 ④⑤⑥⑦:消滅 `(P,K)` 全量物化、MST edge/segment 批次化、逐欄位 assert 後的 int32、批次大小為建構參數;**composite key 保持 int64 + `assert n_nets*64 < 2**63`**;加 `assert k <= 32`。若 M3 T1 已 merge 則在其之上做,否則 M4 自己先做 ①②③ | 既有 CPU/GPU 等價測試全綠;**§4.2 的 field-specific 表**(整數 exact / `tree_wl`·`hpwl` rel ≤ 1e-12 / edge 計數 exact);`bigblue4` K=32 峰值降幅 ≥ 60%;**`mempool_group`(3.5M nets)K=32 實跑峰值 ≤ 2 GB**;K ∈ {1,8,32} × multi-chunk × 空 bucket × legacy 欄位 regression |
| **T2b** **integrated memory gate** | 實驗 | 否 | full-lifetime 記憶體剖析:在 `run_placement_io` 路徑上登記每個常駐 buffer 的 (name, dtype, shape, bytes, create_phase, destroy_phase),輸出每 phase 的 `resident_gb` / `transient_gb` / `device_used_gb`;在 3.1M 與 11.3M 各跑一次 | `results/m4/profile/lifetime_<case>.json`;**§2.2 的可行性三分規則據此判定**,並回頭修正 §2.1/§2.2 的表;過 RESULT GATE |
| **T3** 語料接入 | 純軟體 | 否 | **= Stage2 S5,共用同一 task。** 產 `benchmarks/ispd25/*.json` config(全 15 LEF、tech 最前)+ per-case `target_density` 反推(M4-L3);補 `superblue12` 實際 pins/nets;順手量 NanGate45 的 Λ 分布(M4-L10) | 三個 NanGate45 case + `superblue12` 各一份 config 與一份 `PlaceDB.read` 統計 JSON |
| **T3a** **hierarchy gate(blocking)** | 實驗 | 否 | §3.1a 的 G-A…G-E:解析 cluster instance 前綴、逐 group 報 cells/nets/pins/cell-type 直方圖/bbox,與 standalone group 對照;定義鄰接/對角對;量正規化基準 | `results/m4/bench/hierarchy_gate.json`;**五道 gate 全過才准產 `cluster_stats.json`、才准啟動 T4/T6**;任一不過 ⇒ 立即切配方 B 並在報告明寫循環論證風險 |
| **T4** tiler | 純軟體 | 否 | `ioplace/bench/{export_bookshelf,tile_bookshelf,glue_gen}.py`(§3.2) | 玩具 case:2×2 的 node/net/pin 數 = **4×來源 + manifest 記載的 glue 實數**(不得為 0 或未記);座標與 row 無重疊無縫隙;同 seed 逐位元相同;串流峰值 host ≤ 2 GB(以 3.1M 來源實測) |
| **T5** Rent/驗收器 | 純軟體 | 否 | `ioplace/bench/{rent,verify_bench}.py`:V0/V1 + fit 指標 + **H1–H5 holdout 指標**(§3.3),含 bootstrap CI 與多 seed 聚合 | 已知 Rent 的合成 case(規則 mesh p≈0.5、完全隨機 p≈1.0)量得 p 落在 ±0.05;V0 對故意破壞的檔案全抓得到;**fit 與 holdout 指標在程式介面上分屬兩個函式,holdout 不接受任何參數搜尋介面** |
| **T6** **tiler 校準** | 實驗 | 否 | 產 2×2(12.31M);以 §3.2 的**閉式 MoM** 解 `(lambda_0, alpha)`(≥5 seeds,報 bootstrap CI);跑 V0/V1 + H1–H5;產出三種核 K1/K2/K3 的 3×3 glue 計數預估帶 | H1–H5 全綠 ⇒ 生成器**通過驗證**;任一不綠 ⇒ **不得重掃**,直接記為「未通過」並切配方 B(報告明寫);`results/m4/bench/verify_group2x2.json` + 種子軌跡 + 核敏感度帶 |
| **T6b** **count freeze** | 純軟體 | 否 | 由 T4/T6/T7 的 manifest 取**實際** `g₂/g₄/g₉`、`gp₂/gp₄/gp₉`,重算 §2.1 / §4.3 / §8 的**所有** GPU/host/runtime 帳;重擬合 §1.1 的 host 多預測子模型(group / 1×2 / 2×2 三點);檢查所有 int32 assert 的 headroom | `results/m4/scaling/count_freeze.json`;本文 §2.1/§4.3/§8 的表被實數取代;任何 headroom < 20% 的 int32 決策必須回退 int64 |
| **T7** 30M 生成 | 純軟體 | 否 | 用 T6 定出的 `(lambda_0, alpha)` 產 3×3(27.70M + g₉);跑 V0 + H5′ 自檢;另產 K3 變體(若核敏感度帶 > 20%) | Bookshelf 檔 + manifest(含實際 glue 計數);V0 綠、H5′ 自檢誤差 ≤ 0.1% |
| **T8a** driver 儀表化 | 純軟體 | 否 | 把 T1 的 profile 記錄接進 `run_placement{,_io}.py`,輸出 §6.2 的 JSON;**新增 E1 需要的欄位**:`num_unplaced_cells`、`final_overflow`、`legalization_status`、`hpwl_gp`、`hpwl_lg`、`stop_overflow_reached`;加 `benchmark_kind` 與全部 provenance | 3.1M case 上產出的 JSON 過 §7.0 RESULT GATE;新欄位有對應單元測試(含「legalization 失敗」的人工情境) |
| **T8b** **per-case ρ\*** 校準 | 實驗 | 否 | 每個 case 用**固定的候選格點**(沿用 M3 T5 的 ρ_max 集合)在 **seed A** 上校準,選取規則**預先登錄**:取「`Δhpwl ≤ +5%` 前提下最大的 ρ」——**以約束選,不以被報告的 Δio 選**(winner-bias 控制);報告用的正式 run 一律跑 **seed B** | `results/m4/calib/rho_<case>.json` 記錄整個格點的結果與選取過程;報告表格引用的 run_id **不得**出現在校準集合中 |
| **T8** **全流程實驗** | 實驗 | **部分** | E1/E3 的矩陣:真實 case {1.3M, 2.2M, 3.1M, 11.3M} × K{16,32} × {`flat`, `ours@M2`} 進 `quality_real_cases.md`;合成 case {6.2M, 12.3M} × K{16,32} 進 `scaling_synthetic_cases.md`;`ours@M3` 臂**僅在 M3 v2 已 merge 時**追加 | 每格過 RESULT GATE;E1 的六個門檻逐條判定;report linter 綠;**同時作為 T0b 模型的 holdout**:6.2M/12.3M 的實測必須落在模型 95% PI 內,否則觸發 M4-G6 |
| **T9** 30M 元件級 spike | 實驗 | **部分** | §5.3 第 2 層:27.7M 的 IoTerm(+S4)與 T2 後 evaluator **共常駐、交錯 ≥3 iteration**;`budget_gb=19.5`(L4 契約,不可由 CLI 提高);S4 的 0 / 2 個 `(E,) fp64` 累加器兩種情境都量(§8) | `measured_peak_gb / budget_gb / budget_source / baseline_reserved_gb / resident_gb / max_phase_transient_gb` 齊全;過 RESULT GATE;**27.7M 的 GPU 可行性由此判定,不由 §2.2 的加總判定** |
| **T10** H100 交接包 | 文件+軟體 | 否 | §6.4 的 runbook + `scripts/m4_run.sh` + `scripts/m4_forecast.py` + `scripts/m4_check_prediction.py`;**凍結 `results/m4/forecast/h100_prediction.json`**(含逐 phase 的 `T_L4,p`、`s_p`、判定依據、SKU、sha256) | 在 L4 以 3.1M case 完整演練一次 runbook(除規模外每步都真的跑過);`h100_prediction.json` 的 sha256 記入 M4 報告 |
| **T11** 報告 + audit | 文件+軟體 | **部分** | `docs/results/m4-scale-up-report.md`:E1–E5 逐條打勾/打叉;三個模型的擬合、CI 與 holdout 殘差;兩份對照表;B1/B2/B3 的揭露與對 M2/M3 舊數字的影響評估;**`scripts/m4_report_lint.py`**(合成 case 不得出現在 quality 表、不得有品質欄位;每格 run_id 唯一且過 RESULT GATE) | 每個數字可追到 `results/m4/**/*.json`;linter 在 pytest 中對故意違規的樣本表格必須報錯;FAIL 照 M1/M2 慣例誠實記錄 |
| **T12**(條件) | 純軟體 | 否 | numpy-only PlaceDB shim(§5.4)。**觸發:H100 節點 host RAM < 128 GB** | 3.1M case 上 shim 路徑與原生路徑的 GP 結果逐位元相同 |
| **T13**(條件) | 純軟體 | 否 | IoTerm 的 fused CUDA op。**觸發:T8 的 11.3M 單臂 > 3 h,或 T9 顯示 op 佔總時間 > 85%** | 與 pure-torch 路徑的 fp64 `gradcheck` 一致;`bigblue4` 上 ≥5× 加速 |
| **T14**(新增)**H100 執行與驗證** | 實驗 | 否 | 在取得 H100 SXM5 80GB 後,依 T10 的 runbook 跑 27.7M 單臂,執行 `m4_check_prediction.py` 對照凍結的預測 | wall-time / GPU peak / host RSS 三項對照 §6.3 E5 的區間;**任一落外 ⇒ 在報告發表重擬合模型與歸因**,不得事後放寬區間。**排在 M4 主線之外(硬體相依),但 E5 若永不驗證必須在報告明寫** |

### 7.2 依賴序(v2 重畫)

```
T0 ─┬─ T0b ──────────────────────────────────────────────┐
    │                                                     │
    ├─ T1 ── T1b ──────────────────────┐                  │
    │                                   │                  │
    ├─ T2 ── T2b ──────────────────────┤                  │
    │                                   │                  │
    └─ T3 ── T3a(gate)─ T4 ─ T5 ─ T6 ─ T6b ─ T7 ─ T9 ──┤
                                          │               │
                        T8a ─ T8b ────────┴──── T8 ───────┴─ T11
                                                   │
                                                 T10 ─ T14(硬體相依)

條件:T12(host RAM 情報)/ T13(T8 或 T9 的計時)
```

- **T0 最先**(M2/M3 的 finding 遷入慣例);**T0b 緊隨**,因為 §2 的所有推估欄位都靠它決定「能不能寫成數字」。
- **T3a 是 blocking gate**:不過就沒有 `cluster_stats.json`,T4 的 glue 模型無來源 ⇒ 直接切配方 B。
- **T2 是硬前置**:沒有它,11.3M/12.3M 的任何 evaluator 呼叫都 OOM;**T2b 是可行性判定的唯一合法來源**(§2.2)。
- **T6b(count freeze)必須在 T8/T9 之前**:否則 ≥6M 的所有記憶體/時間帳都建立在 glue = 0 的假設上。
- **T8 依賴 T4/T6b**(合成 6.2M/12.3M 的產物與計數)、**T8a**(欄位)、**T8b**(ρ\*)、**T2b**(記憶體 gate)。v1 的 DAG 漏了前三條。
- **T10 的演練必須在 T8 的 3.1M 結果之後**;**T14 排在 T10 之後且硬體相依**。
- **臂的定義**:`flat` = 無 IO 項;**`ours@M2`** = M2 已驗證的 IoTerm(product-form soft-λ + margin),per-case ρ\*,**零 M3 相依,是 M4 的主線臂**;**`ours@M3`** = 前者 + M3 的 f_ft/S4 與選定的 `κ_ft`,需要 M3 v2 的 evaluator 欄位(`io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home`)、objective 與 driver 全部 merge。**報告中不得出現無限定詞的「ours」。**

**現在就能開工(零 M3 相依):T0, T0b, T1, T1b, T2, T2b, T3, T3a, T4, T5, T6, T6b, T7, T8a, T8b, T8(flat + ours@M2), T9(情境 A), T10。**
**擋在 M3 v2 之後:`ours@M3` 臂與其 `io_rg`/`ft_rg` 欄位、T9 的 S4 情境 B、T11 的對應章節。**

---

## 8. 記憶體帳的參數化(S4 形式未定 + glue 未定)

M3 v2 可能把 S4b 換成別的候選,但都是「同一條 chunked backward 上的係數擴充」。M4 的帳因此寫成兩種情境,**兩種都在 T9 實測**;net 數同時帶 glue 佔位符:

| 情境 | 額外常駐狀態 | @12.71M nets | @(31.54M + g₉) nets |
|---|---|---:|---:|
| **A**(0 個額外 `(E,)` fp64) | 只有 `home_e` `(E,)` int8 | +13 MB | +32 MB × (1 + g₉/31.54M) |
| **B**(2 個額外 `(E,)` fp64) | `home_e` int8 + 2×`(E,)` fp64 | +216 MB | +537 MB × (1 + g₉/31.54M) |

**⇒ 兩種情境的差 ≤ 0.54 GB @30M,相對於 op 本體(§1.3 模型重估後)與 GP 是二階量;M4 的所有規模結論對 S4 的最終形式不敏感** —— 這句要寫進 M4 報告,免得 M3 v2 一改就得重做整份規模分析。**但注意:這個「不敏感」的結論本身依賴 g₉ 不會大到改變量級,由 T6b 確認。**

同理,M3 的 `L_cap`(S7)若啟用,額外狀態是 `(K,K)` 的 `Q`/`G` 與 `(K,K,A)` 的 `P` 常數表(K≤32、A≤2K)⇒ < 1 MB,可忽略。

---

## 9. 風險與 gate

| ID | 量測 | 門檻 | 觸發後動作 |
|---|---|---|---|
| **M4-G1** T2 沒達標 | `mempool_group` K=32 的 evaluator 峰值 | > 2 GB | T2 未完成,不得進 T8 的 ≥6M 臂;先做 edge 批次化 profiling 找剩餘大戶 |
| **M4-G2** 11.3M 全流程失敗 | E1 的任一條 | 任一 fail | OOM → 降 K/lattice 並記錄;legalization 失敗 → 調 `target_density`(M4-L3);`Δhpwl` 爆 → 降 `ρ_max`(T8b 的格點內選,**不得**臨時擴充格點) |
| **M4-G3** 生成器未通過驗證 | §3.3 的 H1–H5 | 任一不綠 | **不得重掃參數**(閉式 MoM 本來就沒有掃描空間)⇒ 記為「生成器未通過」,切配方 B,報告明寫「30M 的連通性是擬合出來的」 |
| **M4-G3a**(新增)階層前提 | T3a 的 G-A…G-E | 任一不過 | 立即切配方 B;`cluster_stats.json` 不得產出;§3.1 第 3 點在報告中標為已否證 |
| **M4-G4** op runtime 失控 | T8 的 11.3M 單臂 wall-time | > 3 h | 觸發 T13。**明確不准**用「降低 GP iteration 數」偽造達標 |
| **M4-G5** host RAM | 任一 `PlaceDB.read` 的 peak RSS | > 100 GB | 該 case 在本機作廢轉 H100;H100 也不足 ⇒ 觸發 T12。**判定用 §5.1 的三分規則,不用單點外推** |
| **M4-G6** 模型失效 | T0b 三個模型在 6.2M/12.3M holdout 的殘差 | 超出 95% PI | 模型不可外推 ⇒ **§2.1 的 27.7M 欄位全部改「未定」**,E5 的 point estimate 必須改由更保守的方法產生並在報告說明;**不得事後偷改預測** |
| **M4-G7** B1 的污染範圍 | M2/M3 報告中引用 `peak_mem_mb` 的段落 | 存在即觸發 | 逐條標註「此數字受同 process 前序臂污染,已作廢」,並在 M4 報告給出 T1b 重測值 |
| **M4-G8**(新增)報表越界 | `scripts/m4_report_lint.py` | 任一違規 | 合成 case 出現在 quality 表、或 synthetic 表出現品質欄位、或 run_id 重複/未過 RESULT GATE ⇒ T11 不得標記完成 |

**額外風險(無獨立 gate,必須記載):**

| # | 風險 | 證據/狀態 | 緩解 |
|---|---|---|---|
| R1 | `mempool_cluster` 的 GP **品質**未驗 | M4-L2 | T8 的第一個 gate(E1) |
| R2 | ISPD2025 是 `place_opt`/`preCts` 的**已擺置** DEF;從其座標起跑等於 warm start,與 ISPD2005 的 random-center init 不同 regime | DEF header(Stage2 §3.2) | T3 必須決定並記錄是否 `random_center_init_flag=1`;兩種都跑一次 sanity |
| R3 | lattice 解析度在大 die 上的敏感度未量 | `run_placement.py:11` 預設 512 | T8 加一組 lattice ∈ {512,1024,2048} sensitivity(只在 3.1M) |
| R4 | 合成 case 的 uniform 位置**系統性高估** evaluator 成本 | §1.2 的合成 `io_count=29.8M` vs 真實 bigblue4 K16 的 100,333 | T9 的 spike 必須用 **tiler 產出的真實 `.pl` 起始座標**,不用 uniform random |
| R5 | 30M 的 pin/net 比是外推 | 推估 | T7 產出後直接數,由 T6b 回填 |
| R6 | H100 節點的 CUDA arch 需重 build(`sm_90`) | `docs/dev-env.md` 記錄本機只 build `sm_89` | T10 runbook 第一步;T14 執行前檢查 |
| R7 | `mempool_cluster` 讀取 285 s + 62.6 GB,每次 run 都付一次 | 待重現 | T3 評估「一次讀入、多臂共用同一 process」;**但與 T1b 的「每臂一個 subprocess」衝突** ⇒ 由 T1b 的 A/B 結果仲裁,兩者不可同時成立時以量測正確性優先 |
| R8(新增) | 距離核形狀在 2×2 不可驗證(§3.2 的 0 DoF) | 結構性 | T7 發表三核敏感度帶;若 G-D 判定 1×4 排列則可獲得 1 個 DoF,報告須說明實際情形 |

---

## 10. 低信心段落總表(v3 對抗性審查請優先攻擊這裡)

| ID | 段落 | 不確定的是什麼 | 由誰解決 |
|---|---|---|---|
| M4-L1 | §2.4 | `superblue12` 的 pins/nets 與 GP 峰值是推估 | T3 |
| M4-L2 | §2.4 | `mempool_cluster` 的 GP **品質**完全未看 | T8(E1) |
| M4-L3 | §2.4 | NanGate45 的 `target_density` 該取多少 | T3 |
| M4-L4 | §3.3 | Mt-KaHyPar 在 12.3M/27.7M hypergraph 上的可行性未知 | T5/T6 |
| M4-L5 | §3.2 | glue 距離核的**形狀**在 2×2 結構上不可驗證(0 residual DoF) | T7 的三核敏感度帶;無法根本解決 |
| M4-L6 | §4.3 | T2 後的 evaluator 記憶體(4.5 / 10 GB)是推估 | T2 驗收 + T6b |
| M4-L7 | §5.2 | 單臂 wall-time 假設 op 在 600 iteration 上啟用;實際啟用點由 `of_on=0.90` 決定,10M 級的 overflow 軌跡未知 | T8 |
| M4-L8 | §6.3 E5 | 各 phase 的 `s_p` 是由 L4 achieved bandwidth 推的**單機外推**,沒有同型 workload 的 H100 實測 | T14(唯一能解決的方式) |
| M4-L9 | §3.1 | 「cluster = 4 groups」**已降級為待證假設**,且 3.675 的比值已顯示與 4× 不符 | T3a(blocking) |
| M4-L10 | §4.4 | Λ∈[4,8] 的 net 在 NanGate45 語料上的比例未量 | T3 |
| M4-L11 | §1.1 | 三個模型的係數不確定度與外推有效範圍 | T0b + T8/T9 holdout |
| M4-L12 | §2.1 | 所有合成級的 glue 計數 `g*` 未知 ⇒ 下游帳全為低估 | T6b |

---

## 11. 證據附錄(T0 待遷入)

**狀態:下列每一條皆為 2026-08-13 在本機實跑;探針在 `/tmp`,T0 完成前一律「待重現」。**

| 探針(暫存 → 目標檔名) | 內容 | 支撐 |
|---|---|---|
| `/tmp/probe_m4_mem.py` / `/tmp/probe_m4_full.py` → `probe_gp_memory.py` | 五 case 的 `PlaceDB.read` + `initialize` + GP+LG,量 phase 時間、host RSS、GPU 峰值 | §1.1 全表(**擬合模型由 T0b 重建**) |
| `/tmp/probe_m4_read.py` → `probe_host_rss.py` | 只讀不跑,量 `PlaceDB.read` 的 peak RSS 與 die/利用率 | §1.1 host 欄、M4-L3 |
| `/tmp/probe_m4_eval.py` → `probe_eval_scaling.py` | 合成 netlist 上掃 `evaluator_gpu` 的 (nets, K) → 峰值/時間 | §1.2、§4.1(需加「解析帳 vs 實測」兩欄) |
| `/tmp/probe_m4_spike30.py` → `probe_spike_scale.py` | `spike_10m.run()` 參數化到 30M | §1.3、§5.3 第 2 層 |
| `/tmp/probe_m4_macro.py` → `probe_macro_stats.py` | movable macro 統計、平均 degree | §1.1 次級事實 1 |
| (新增)`probe_bookshelf_export.py` | `BOOKSHELFALL` 轉出 + 檔案大小/時間 | §3.1 證據 1、2 |
| **(v2 新增)`/tmp/probe_v2_dtype.py` → `probe_scatter_dtype.py`** | int8/int8 `scatter_reduce_(amax)` 在 CUDA 可用且與 int64 逐位元相同;int8←int64 直接報錯;`_pack_bits` 的 int8×int64 升位正確;`1<<63` 溢位 ⇒ K≤32 | §4.2 的 dtype 契約、Codex #9 |
| **(v2 新增)`/tmp/probe_v2_treewl2.py` → `probe_reduction_order.py`** | 7.65M 項 heavy-tail float64 求和:batch=3e6/8e6 恰好逐位元相同、batch=1e6/999983 差 rel 1.15e-16 | §4.2 的 field-specific 驗收、Codex #10 |
| (既有)`results/m2/spike/spike_10m.json` | 10M spike 峰值(**缺 commit/env/input hash 與 `peak_reserved_gb`,T0 補齊**) | §1.3 |
| (讀碼)`ioplace/evaluator_gpu.py:200-215,314,326,332,351,352,385,392-416,413` | §4.1 的逐項定位、dtype 路徑 | §4.1、§4.2 |
| (讀碼)`ioplace/ops/io_term.py:16-17,329-332,364-376,386-399` | 7 個 degree bucket、`k_chunk` 規則(dedup pins)、diagnostics 的 `1+n_nonempty` 趟 | §1.3、§1.4 B2、§5.1 |
| (讀碼)`ioplace/drivers/run_placement.py:129`、`run_placement_io.py:50,63,93,141,145,155,202` | 缺 reset、三個元件的共常駐時點、diagnostics/WL backward 呼叫點 | §1.4 B1/B2、§2.2 |
| (讀碼)`$DP/dreamplace/PlaceDB.py:1010-1033,499-616,1133`、`ops/place_io/src/BookshelfWriter.cpp:22-51,100-138`、`PybindPlaceDB.cpp:60` | Bookshelf 匯出能力、T12 的實作面 | §3.1、§5.4 |

**與既有資產的關係:** `ioplace/ops/io_term.py` 的 chunked 契約、`ioplace/schedules.py` 的 version invariant、`ioplace/dp_hook.py` 的 secant refresh、`ioplace/drivers/run_placement_io.py` 的 callback 結構,M4 **全部原樣沿用**;M4 的改動集中在 `ioplace/evaluator_gpu.py`(T2)、新目錄 `ioplace/bench/`(T3a/T4/T5)、`ioplace/profile.py`(T1,已落地)與 driver 的儀表化(T8a)。**沒有任何一項要求改寫 M2 已驗證的數學路徑。**

---

**與其他草案的介面(避免重複設計):**

- **Stage 2 S5 = M4 T3**,同一個 task、同一份 config 產生器;Stage 2 §4.3 的 L-Q1-b(`mempool_cluster` 可讀性)由 §1.1 回答:可讀,285 s / 62.6 GB(待 T0 重現)。
- **M3 T1 與 M4 T2** 的分工見 §4.2,**v2 已依 Codex #9 改寫**:M3 T1 = source + accumulator 的**成對** dtype 降級(省 ≈15.5 GB,**仍不足以解封 11.3M**),M4 T2 = 結構性 streaming。兩邊驗收互不重疊,但 M3 T1 的測試必須覆蓋 K ∈ {1,8,32} × multi-chunk × 空 bucket × legacy 欄位 exact regression。
  **需要 M3 側同步修正的一點:** M3 v3 §2 的 T1 註記(該檔 line 134)仍把「改 packed bitmask 或按 net 分塊」列為 R2 的解法;本文 §4.5 已證明 **packed bitmask + `scatter_reduce(amax)` 對 `passed_bit_acc` 數學上錯誤**(一個 segment 會經過多個 region ⇒ 該列有多個 bit ⇒ `max ≠ OR`),只有 `pin_bit_acc`(每列單一 bit)成立。⇒ M3 T1 應改採 §4.2 的**成對 int8 dtype 降級**,並在 M3 草案中更正該行。
- M3 draft §7 G6 的「10M spike 8 GB 契約」在 M4 改為 `--budget-gb` + `budget_source` + 硬體契約 assert(§1.4 B3);**M3 自己的 G6 判準不變**(仍是 10M / 8 GB),M4 只是不再把 30M 的 12.32 GB 誤判成失敗。
