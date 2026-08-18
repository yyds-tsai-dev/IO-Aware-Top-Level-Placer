# Stage 2(Phase 2)校準報告 —— open-source router(OpenROAD)ground truth 校準

- 日期:2026-08-18
- 分支:`m2-differentiable-io`
- 對應規劃:`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md`(§9 exit 判準與 G1–G7 gate、§10 task 分解)
- 前置報告:`docs/results/m2-differentiable-io-report.md`(evaluator 口徑基準)、`docs/results/m3-differentiable-ft-report.md`(RG 模型、`κ_ft` 先驗、§7 對 Stage2 的移交)
- **標題與摘要規範遵循 G1(見下):本報告的 ground truth 是 OpenROAD(開源 router),不是 commercial-tool(Innovus)ground truth。所有對外主張限定為「相對 OpenROAD 的校準」。**

---

## 0. 一句話結論

**Stage 2 以 F-OR(OpenROAD)單 router 交付**——Innovus license 三台伺服器全數 `No route to host`(`results/stage2/env.json`,G1 已觸發),因此本報告的「真實繞線」全部來自 OpenROAD global route + detailed route,**不是 commercial-tool ground truth**。在這個限定下:總量級校準係數 `α̂=0.6518`、`β̂=0.9661`、`γ̂=0`(R²=0.9946,但僅 n=6、3 殘差自由度,信度有限)可用;**per-net 級對照因 S8 從未落盤 evaluator 的 per-net 陣列而全數 `not_evaluable`**,這是本輪最大的方法論缺口。**規模天花板是本輪最重要的新發現**:OpenROAD detailed route 在 ISPD2015 這類高利用率語料上的可行上限約在 15 萬 cell 上下(`matrix_mult_1` 155k congestion plateau、`superblue19` 506k 第 0 iteration 逾時、`superblue12` 1.29M 未及嘗試),15 個排定的臂裡只有 6 個產出有效樣本,9 個因 congestion/timeout 落入不可評樣本表。E1/E2/E3/C1/C5 依規格如實判 FAIL/partial/not_evaluable;E4/E5/C3/C4 可判並已完成。

---

## 1. 方法:S1–S5 鏈路與信任基石

Stage 2 的軟體鏈路(§10 S0–S5,全部不需要 Innovus)按依賴序執行:

```
S0(環境固化) → S1(DEF 輸出) → S2(routed-DEF 解析) → S3(crossing 抽取) → S4(F-OR 端到端演練) ──┬── S8(校準實驗矩陣) → S9(校準分析) → S10(本報告)
                                                                                          │
                                          S5(NanGate45 語料接入) ─────────────────────────┘
                                          S6/S7(Innovus 環境驗證/F-INV flow)── 因 G1 擱置
```

### S0 — 環境固化

`results/stage2/env.json`(provenance:`repo_commit=685b4227ffa34be71b5b481f7e9e68b7a428f0d8`,`dp_commit=d971880a15ef684c4a90bccd2c65d62ae7e33297`,`hostname=NVL5`,`2026-08-15T08:52:51Z`)。關鍵欄位:

| 項目 | 結果 |
|---|---|
| Innovus binary | 在(`/usr/cad/cadence/INNOVUS/INNOVUS_21.19.000`) |
| Innovus license 三台(`lshc`/`lstc`/`lstn`,port 5280) | **全部 `tcp_connect_ok=false`,`OSError(113, 'No route to host')`**(DNS 皆解得出)⇒ **G1 觸發** |
| OpenROAD | `v2.0-17598-ga008522d8`,`+Charts +GPU +GUI +Python`,`odb_import_ok=true`,`dbWireDecoder` 具 `PATH/POINT/POINT_EXT/VIA/TECH_VIA/RECT/SHORT/VWIRE/JUNCTION/END_DECODE` 全部 opcode |

### S1 — DEF 輸出

`ioplace/export/def_export.py` 產 `out.def`/`regions.json`/`netmap.json`/`coord.json`(commit `be22375`)。驗收綠:OpenROAD `read_def` 零 error、`#COMPONENTS` 與輸入相同、讀回座標與 `node_x/node_y` 差 ≤1 DBU、`netmap` 對 `placedb.net_names` 逐項相同。

### S2 — routed-DEF 解析

`ioplace/route_eval/or_scripts/dump_segments.py`(odb `dbWireDecoder`)。**已知 binding 缺陷 + workaround**:這個 OpenROAD build 的 Python binding 下 `getPoint()`/`getRect()` 對 `POINT_EXT`/`RECT` 不可達(SIGABRT),改直接讀 `dbWire` 的 data array,並用 845,555/845,555 個一般 `POINT` opcode 逐一交叉驗證 accessor 語意後才敢用在 `POINT_EXT`/`RECT` 上(§7.1 規格記錄)。

`results/stage2/rehearsal/mgc_fft_1/verify_s2.json` 驗收:

| 檢查 | 結果 |
|---|---|
| check1(reconciled wire length,`route_wl + Σ_RECT(long_side-short_side)` vs `dbWire::getLength()`) | `err_vs_dbwire_native_recon_pct = 0.0499%`(< 1% 門檻通過);raw definitional delta(未加回 RECT 項)`1.867%` 如實揭露,不作為 gate |
| check2(odb vs 文字解析全量比對) | `n_checked=33307`、`n_mismatch=0`(全語料 0/33,307,非抽樣) |
| `junction_unvalidated` | `false`(此樣本未出現 JUNCTION opcode,故未觸發「含 JUNCTION 的 net 判為 unvalidated」的揭露規則;規則本身已寫入 `verify_routed_def.py`,對其他樣本仍生效) |
| `overall_pass` | `true` |

RECT 口徑:採 patch metal(`RECT` 為 delta-encoded patch,不進 `route_wl` 也不進任何 crossing 量,§7.4)。

### S3 — crossing 抽取

`ioplace/route_eval/route_crossings.py`,重用 `evaluator_ref._walk_segment`(commit `f9c1a77`)。三項驗收全綠:

1. 恆等式違反的 root-cause 與 clamp 修法(`route_cross_raw ≥ route_cross_dw(δ) ≥ Λ_route−1 ≥ 0`)。
2. 10 萬次隨機壓測,0 違反。
3. **MST-as-fake-wire 逐位元重現**:把 evaluator 的 MST edge 當成假 wire 餵回 `route_crossings.py`,`per_net_crossings`/`per_net_ft`/`boundary_pair_demand` 與 `evaluator_ref` 原生輸出逐位元相同——這是整條鏈路的信任基石(規格 §10 明定),已在 S4 之前綠。

### S4 — F-OR 端到端演練(mgc_fft_1)

`results/stage2/rehearsal/mgc_fft_1/`。DR 產出 57.7k violations、不收斂,**如實揭露**(不隱藏、不重跑到收斂);V1/V2 兩個不變式(`#COMPONENTS`/`#NETS` 一致)過。原規劃設想的「LEF 清洗前提」(修 ISPD2015 tech.lef 的重複 VIA `DRT-0338`)**不成立**——真正阻擋 detailed route 收斂的是 DEF-side 的 VIA 重複定義,修法是 DEF-side VIA dedupe,不是清洗 LEF。

### S5 — NanGate45 語料接入

`benchmarks/ispd25/*.json` DREAMPlace config(15 個 LEF 全 glob、tech.lef 排最前,踩雷點見規格 §3.2 P9)+ read stats。`mempool_tile_wrap` GP+LG 跑通並產出 evaluator 報表。

### S6/S7 — Innovus 環境驗證 / F-INV flow

**因 G1(license 不通)擱置**,I1–I8 checklist 與 F-INV Tcl 骨架均未執行;§6.2 的整個 F-INV 流程停留在「憑一般用法寫的骨架,一行未在本機執行過」(規格 L-全文)。列入 §7 future work。

---

## 2. 樣本表

### 2.1 有效樣本(6 個,全部 F-OR、ISPD2015/ISPD2025、K∈{16,32}、臂∈{flat, ours})

| sample | K(placement) | route_cross_dw(matched K) | DR 最終 violations | drc.rpt 行數 | route elapsed | verify_s2/V1V2 |
|---|---:|---:|---:|---:|---:|---|
| des_perf_1__flat | 16 | 6,016 | 206,867 | 23,671 | 9,555s(2:39:15) | pass |
| des_perf_1__ours_k16 | 16 | 5,314 | 202,875 | 23,773 | 9,554s | pass |
| des_perf_1__ours_k32 | 32 | 8,737 | 209,940 | 23,726 | 9,668s | pass |
| mempool_tile_wrap__flat | 16 | 10,169 | 41,387 | 20,041 | 4,566s(1:16:06) | pass |
| mempool_tile_wrap__ours_k16 | 16 | 8,908 | 43,651 | 20,055 | 4,649s | pass |
| mempool_tile_wrap__ours_k32 | 32 | 15,911 | 42,750 | 20,062 | 4,681s | pass |

來源:`results/stage2/calibration/tables.json:sample_list_table`、`results/stage2/s8/{des_perf_1,mempool_tile_wrap}__*/`。DR violations 數量都非常大(20k–24k drc.rpt 行、20 萬+/4 萬+ 最終 violation 計數)——這些 route **完全不是 DRC-clean 的**,依規格 §6.2 的口徑決策,只要求完整 wire geometry(V1/V2/verify_s2 全綠即 instance/net 集合守恆),DRC 數只記錄、不作 gate。

**DR capped 5 iters 的口徑決策**:route.tcl 用 `-droute_end_iter 5` 限制 detailed route 的最佳化迭代數,理由是 crossing 抽取只需要完整繞線(有 wire geometry 即可,不需要 DRC-clean),把預算花在讓更多 case 拿到繞線結果而非讓少數 case 收斂。此決策的代價是 violations 數字偏高(如上表),已在樣本表逐項揭露。

### 2.2 不可評樣本(9 個)—— OpenROAD DR 的規模天花板

| sample | 失敗模式 | 證據摘要 |
|---|---|---|
| matrix_mult_1__flat | congestion_plateau_timeout | `-droute_end_iter 5` 全跑完,violations 卡在 ~544,931;4h wall-timeout 前最後記錄 40% 完成,elapsed 26:47 |
| matrix_mult_1__ours_k16 | congestion_plateau_timeout | 同上;最後 528,897 violations,50% 完成,elapsed 30:59 |
| matrix_mult_1__ours_k32 | congestion_plateau_timeout | 同上;最後 532,973 violations,50% 完成,elapsed 30:17 |
| superblue19__flat | congestion_plateau_timeout | 0th iteration 本身跑超過 3h,violations 從 124,751(10%)飆到 2,199,855(50%),elapsed 3:06:13 被 4h timeout 殺掉 |
| superblue19__ours_k16 | congestion_plateau_timeout | 同上;2,317,033 violations @ 50%,elapsed 3:08:34 |
| superblue19__ours_k32 | dr_0th_iteration_timeout | log 停在「Start 0th optimization iteration」,連一條 `Completing X%` 進度行都沒有——4h 預算在第一個 checkpoint 之前就耗盡 |
| superblue12__ours_k16 | dr_0th_iteration_timeout | 同上,同樣停在 0th iteration 起點、無任何進度行 |
| superblue12__ours_k32 | killed_during_global_route_setup(證據較弱) | log 僅 2.8KB,停在 `GRT-0300` warning 之後、global_route 進度或 detailed_route 起點之前;較像外層批次腳本被中斷,而非 route.tcl 乾淨跑完並失敗 |
| superblue12__flat | route_not_attempted_queue_exhausted | 無 `or_run/` 目錄;`out.def`(169,029,018 bytes)是全 S8 語料裡最大的檔案,批次腳本以「最小檔案優先、concurrency=1」排程,不是 route 失敗而是批次時間預算沒排到它 |

來源:`results/stage2/calibration/tables.json:unevaluable_sample_table`。

**規模天花板的實測結論(本輪最重要的新發現)**:15 個排定的臂(matrix_mult_1/superblue19/superblue12 各 3 臂 + des_perf_1/mempool_tile_wrap 各 3 臂)全數完成 place(GP+LG),但 route 只在 6/15 上成功;`mgc_fft_1`(32k)10 分鐘完成、`des_perf_1`(113k)約 3h/臂完成、`matrix_mult_1`(155k,`target_density=0.802`)congestion plateau 卡在單一 DR iteration、`superblue19`(506k)DR 第 0 iteration 本身逾 4h、`superblue12`(1.29M)未及嘗試。**在 ISPD2015 這類高利用率語料上,本機 OpenROAD detailed route 的可行天花板約在 15 萬 cell 上下**;NanGate45 `mempool_tile_wrap`(128k,10 層 metal,較低利用率)三臂全部 route 成功,顯示天花板同時是 utilization 與 cell 數的函數,不是單純規模函數。

---

## 3. 校準結果

完整機器可讀表見 `results/stage2/calibration/tables.json`;人讀版見 `results/stage2/calibration/stage2_calibration_summary.md`(下方為濃縮引用,數字逐一可追)。

### 3.1 七張表狀態

| # | 表 | 狀態 | 說明 |
|---|---|---|---|
| ① 總量比 + per-net 對照 | **部分可評** | 總量比 `R_X=Σroute/ΣX` 可算;per-net Pearson/Spearman `not_evaluable` |
| ② per-net 相關 | **not_evaluable** | S8 從未落盤 evaluator 的 per-net 陣列(`per_net_crossings`/`per_net_steiner`/`per_net_lambda`);`metrics.json` 只有純量總量,`metrics.json.npz` 只有 `node_x`/`node_y`。要拿到 per-net 陣列須在 routed placement 上重跑 `evaluate_gpu`(需 GPU),本輪明確禁止重跑 GP/GPU |
| ③ per-degree bucket | **not_evaluable** | net degree 未落盤於任何 S8 artifact |
| ④ per-Λ bucket | **降級可評** | 只能用 route 側自身的 `per_net_lambda_route` 分桶(非 route-vs-evaluator 對照);絕大多數 net 落在 Λ_route=1(單樣本 mean 介於 0.96–1.08) |
| ⑤ 三值分解總表 | **可評** | design×K×arm×router,6 列,見下 §3.2 |
| ⑥ FT 三值表 | **部分可評** | `ft_rg`/`ft_mst` 可評;`route_ft` 整欄 `-1` sentinel `not_evaluable`(S8 抽取呼叫從未傳 `pin_regions` 引數) |
| ⑦ boundary-pair demand 相關 | **not_evaluable** | `boundary_pair_demand` 從未落盤,同 ② 根因 |

**這一輪最重要的兩個限制**(逐字保留自 S9 摘要):

1. **S8 從未把 evaluator 的 per-net 陣列落盤**——後果:所有 per-net 對照(表②③⑦、C1 的 ρ 半)一律 `not_evaluable`,只有總量級比值可算(表①⑤⑥)。
2. **`per_net_route_ft` 在全部 6 個樣本裡都是 `-1` sentinel**——後果:表⑥的 `route_ft` 欄整欄 `not_evaluable`,`ft_rg`/`ft_mst`(evaluator 總量)不受影響。

### 3.2 總量比(表①)與三值分解(表⑤)

`R_X = Σ route_cross_dw(δ=2) / Σ X`,X ∈ {`lam_minus_1`, `io_rg`, `io_mst`}(matched-K):

| sample | R_lam_minus_1 | R_io_rg | R_io_mst |
|---|---:|---:|---:|
| des_perf_1__flat | 0.7484 | 0.6997 | 0.6480 |
| des_perf_1__ours_k16 | 0.7951 | 0.7172 | 0.6719 |
| des_perf_1__ours_k32 | 0.8857 | 0.7221 | 0.6742 |
| mempool_tile_wrap__flat | 0.6784 | 0.6511 | 0.5432 |
| mempool_tile_wrap__ours_k16 | 0.7331 | 0.6949 | 0.5782 |
| mempool_tile_wrap__ours_k32 | 0.7568 | 0.6741 | 0.5695 |
| **pooled** | **0.7569** | **0.6869** | **0.5970** |

route 實測值全面**低於** evaluator 的三把尺(所有比值 <1),`io_mst`(MST 幾何)高估最嚴重、`io_rg`(RG/Steiner)次之、`lam_minus_1`(routing-independent 下界)最接近 1。**三者全部落在 E3 的 [0.80,1.25] 門檻之外**(E3 FAIL,見 §4)。

三值分解總表(`route−io_rg` 全部為負、`io_mst−route` 全部為正):

| design | K | arm | Σ(λ−1) | Σio_rg | Σroute(dw2) | Σio_mst | mst_excess | route−io_rg | io_mst−route |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| des_perf_1 | 16 | flat | 8,038 | 8,599 | 6,016 | 9,286 | 687 | −2,583 | 3,270 |
| des_perf_1 | 16 | ours_k16 | 6,682 | 7,410 | 5,314 | 7,909 | 499 | −2,096 | 2,595 |
| des_perf_1 | 32 | ours_k32 | 9,866 | 12,099 | 8,737 | 12,958 | 859 | −3,362 | 4,221 |
| mempool_tile_wrap | 16 | flat | 14,990 | 15,617 | 10,169 | 18,720 | 3,103 | −5,448 | 8,551 |
| mempool_tile_wrap | 16 | ours_k16 | 12,151 | 12,823 | 8,908 | 15,407 | 2,584 | −3,915 | 6,499 |
| mempool_tile_wrap | 32 | ours_k32 | 21,014 | 23,604 | 15,911 | 27,941 | 4,337 | −7,693 | 12,030 |

**觀察(如實記錄,未深究成因)**:本輪實測排序是 `route < io_rg`,即 route 甚至低於 RG 模型的 Steiner 下界——與 §1.2 原本預期的 `(λ−1) ≤ io_rg ≤ route ≤ io_mst` **不一致**。候選解釋(未驗證):(a) F-OR 沒有 detailed placement,GP+LG 直入 router 的幾何可能比 RG 假設更緊湊;(b) run-length filter(δ=2)把大量邊界貼線 crossing 濾掉(見下方 G4)。

### 3.3 FT 三值表(表⑥)

| sample | ft_rg | ft_mst | route_ft |
|---|---:|---:|---|
| des_perf_1__flat | 561 | 622 | not_evaluable |
| des_perf_1__ours_k16 | 728 | 795 | not_evaluable |
| des_perf_1__ours_k32 | 2,233 | 2,360 | not_evaluable |
| mempool_tile_wrap__flat | 627 | 733 | not_evaluable |
| mempool_tile_wrap__ours_k16 | 672 | 719 | not_evaluable |
| mempool_tile_wrap__ours_k32 | 2,590 | 2,678 | not_evaluable |

`ft_rg < ft_mst` 在全部 6 樣本上成立,方向與 M3 draft 的 RG 模型設計預期一致,但缺乏 route 側真值,三值排序無法完成驗證。

### 3.4 迴歸校準(§8.3,C2 的輸入)

per-net 形式不可用(限制 1),改在**樣本總量級**擬合(6 個樣本各貢獻一列,非負最小平方 NNLS、無截距):

| 係數 | 值 | bootstrap 95% CI(6 列重抽 2000 次,1958 次成功) |
|---|---:|---:|
| α̂ | **0.6518** | [0.3247, 0.7227] |
| β̂ | **0.9661** | [0.3053, 2.8755] |
| γ̂ | **0.0000**(NNLS 夾到邊界) | [0.0000, 0.9701] |

R² = **0.9946**,per-sample 相對殘差 −2.0%〜+4.8%。

**誠實揭露 n=6 的限制**:6 個觀測、3 個非負係數 = 僅 3 個殘差自由度;R²=0.995 在這個 dof 下幾乎不能拒絕任何合理的線性形式,不代表模型被強烈驗證。β̂ 的 bootstrap CI 寬達 [0.31, 2.88],本身說明估計不穩定。

per-design 分開擬合(僅供參考,3 樣本 3 係數 = 0 殘差自由度,精確擬合、無穩定性資訊):

| design | α̂ | β̂ | γ̂ |
|---|---:|---:|---:|
| des_perf_1 | 0.6935 | 0.8532 | 0.0000 |
| mempool_tile_wrap | 0.0000 | 0.8891 | 3.1416 |

兩個 design 的係數差異巨大——這正是 C5 本該量化的變異,但因是精確擬合,不構成正式 CV 輸入,只能當作「即使有數字,也不該被信任」的佐證。

### 3.5 C1–C5 判定

| # | 判定 | 結果 | 理由 |
|---|---|---|---|
| **C1**(模型選擇) | **not_evaluable** | 需要 per-net ρ(io_rg,route) vs ρ(io_mst,route),限制 1 擋住。補充證據(非判定):pooled 總量比 `|R_io_rg−1|=0.3131 < |R_io_mst−1|=0.4030`,方向偏向 RG 模型,但單靠總量比不滿足 C1 的 AND 條件 |
| **C2**(迴歸校準) | **觸發**(因 E3 FAIL) | 三個 pooled 總量比(0.757/0.687/0.597)全部落在 [0.80,1.25] 之外;已新增 `io_calibrated` 欄(= α̂(λ−1)+β̂(ST−(λ−1))+γ̂(io_mst−ST)),殘差見 §3.4,寫入 `tables.json:c2_regression_calibration.io_calibrated` |
| **C3**(objective 權重) | **可判** | `κ_ft ← β̂/α̂ = 0.9661/0.6518 = 1.4821`(取代 M3 預設 `κ_ft=1.0`);因 α̂/β̂ 建立在 n=6 迴歸上,κ_ft 的可信度繼承同樣限制 |
| **C4**(sign-invariance) | **可判:sign_invariant** | flat→ours_k16→ours_k32 在 `hard_lambda_sum`(pre-route)與 `route_cross_dw`(post-route)兩把尺上,兩個 design 的方向全部一致(4 組比較,0 次翻轉):des_perf_1 flat→k16 兩者皆降(−1,356/−702),k16→k32 兩者皆升(+3,184/+3,423);mempool_tile_wrap 同型態。M2/M3 既有 exit 判準在校準後的量上維持原結論 |
| **C5**(轉移檢驗) | **not_evaluable** | 需要 ≥3 個 design 才能算 CV;本輪只有 2 個 design(des_perf_1、mempool_tile_wrap)有完整路由樣本,matrix_mult_1/superblue19/superblue12 三個 design 全數落在不可評樣本表,無法補足第三個 design |

### 3.6 G4(raw vs dw(2) 漂移檢查)

`Σroute_cross_raw = 84,954`、`Σroute_cross_dw(2) = 55,055`,漂移 = **+54.3%**(遠超 15% 門檻,**觸發 G4**)。「沿邊界貼線走」是主要效應。**δ∈{1,4} 的完整掃描本輪未做**(S8 只抽取了 δ∈{0(raw),2};補完需要重跑 `route_eval.route_crossings`,超出本輪「只讀既有 artifact」的範圍)——L-Q4-a 仍未解決。

---

## 4. E1–E5 逐條判定

| # | 判準(規格 §9.1) | 判定 | 理由 |
|---|---|---|---|
| **E1** | ≥3 個 design × ≥2 個 K × 2 臂(flat/ours)= ≥12 個 routed DEF 成功產出,且全部通過 V1–V5;未繞線 net 比例 ≤2% 並如實列出 | **FAIL** | 只有 **2 個 design**(des_perf_1、mempool_tile_wrap)、**6 個**有效樣本產出並通過 V1/V2/verify_s2(<12);matrix_mult_1/superblue19/superblue12 三個 design 的 9 個排定臂全數因 congestion/timeout 落入不可評樣本表(§2.2)。design 數與樣本數雙雙不達標。**如實 FAIL,不挑好看的 case** |
| **E2** | 最佳模型(io_rg 或 io_mst)對 route 的 per-net Spearman ≥0.70(在 `route>0 或 X>0` 的 net 上,degree ≤`max_degree` bucket) | **not_evaluable** | 限制 1(per-net evaluator 陣列未落盤)直接擋住;無法計算任何 per-net Spearman/Pearson |
| **E3** | 總量比 `R_X ∈ [0.80,1.25]`;不成立則交出 C2 迴歸校準與殘差 | **FAIL**(依規格觸發 C2 補救) | pooled `R_lam_minus_1=0.7569`、`R_io_rg=0.6869`、`R_io_mst=0.5970` 三者全落在區間之外;已依規則交出 C2 的 NNLS 迴歸(α̂/β̂/γ̂,R²=0.9946)與 per-sample 殘差(§3.4) |
| **E4** | §8.2 的 7 張表全部產出;C1 與 C5 的判定寫進報告 | **部分判定** | 7 張表**全部產出**(狀態逐一標記:①部分可評/②not_evaluable/③not_evaluable/④降級可評/⑤可評/⑥部分可評/⑦not_evaluable,§3.1);C1/C5 判定已寫入報告主文(C1 `not_evaluable`、C5 `not_evaluable`,§3.5)。「表全部產出」達標,但多數表的內容本身是 `not_evaluable`,不代表對照關係已建立 |
| **E5** | C4 的 sign-invariance 檢查完成,結果(不論通過與否)寫進報告主文 | **PASS** | C4 已完成:4 組 flat/k16/k32 比較,pre-route(`hard_lambda_sum`)與 post-route(`route_cross_dw`)方向全部一致,0 次翻轉,判定 `sign_invariant`(§3.5) |

**E1–E5 總覽:E1 FAIL、E2 not_evaluable、E3 FAIL(依規則轉入 C2)、E4 部分判定、E5 PASS。** Stage 2 的軟體鏈路(S0–S5、S8 的 6 個有效樣本內)已驗證可信(§1 的信任基石),但校準的**統計功效**(design 數、per-net 對照)不足規格原定門檻。

---

## 5. G1–G7 判定表

| # | 條件(規格 §9.2) | 判定 | 說明 |
|---|---|---|---|
| **G1** | Innovus license 在 S6 time-box 內仍不通 | **已觸發** | 三台 license server(`lshc`/`lstc`/`lstn`,5280)全部 `No route to host`(`results/stage2/env.json`)。**依規則:Stage 2 以 F-OR 單 router 交付,本報告標題/摘要已明寫 open-source router ground truth,所有對外主張降級為「相對 OpenROAD 的校準」,不得把 OpenROAD 結果寫成 commercial-tool ground truth**(已遵循,見文件開頭)。Innovus 列入 §6 future work |
| **G2** | C1 判定 MST 尺優於 RG 尺 | **未觸發(C1 not_evaluable,無法判定)** | C1 本身 `not_evaluable`(限制 1);補充的總量比證據方向偏向 RG(`\|R_io_rg-1\|<\|R_io_mst-1\|`),但不構成正式 C1 裁決,因此 G2 的觸發條件也無法被正式評估。M3 Q1(RG 模型)**未被本輪證偽,也未被正式證實** |
| **G3** | C5 的 CV >25% | **未觸發(C5 not_evaluable,無法判定)** | 只有 2 個 design 有完整路由樣本,不足規格要求的 ≥3 個 design 才能算 CV。**這本身是需要使用者決策的分岔**(規格 §9.2 原文),本報告不代為裁決「Phase 1 主 benchmark 是否搬到 LEF/DEF」 |
| **G4** | `route_cross_raw` 與 `route_cross_dw(2)` 總量差 >15% | **已觸發** | 漂移 **+54.3%**(84,954 → 55,055),遠超門檻。依規則:所有結論理論上該雙值陳述——本報告主表用 dw(2)(規格指定的主報表值),raw 值列於 §3.2 表①/⑤下方對照。δ∈{1,4} 掃描與「wire-to-boundary 距離分佈」診斷未做,留 §6 |
| **G5** | V1/V2 在 F-INV 上系統性被破(CTS/opt 關不掉) | **未觸發(F-INV 未執行)** | 因 G1,S7 F-INV flow 從未跑起,G5 的觸發條件無法評估。F-OR 側的 V1/V2 在全部 6 個有效樣本上皆通過 |
| **G6** | 未繞線 net 比例 >2%(congestion) | **未直接評估(選擇偏差已定性揭露)** | 6 個有效樣本各自的未繞線 net 比例未在 S9 落盤為獨立欄位(限制同 E1 的 V4 揭露缺口);但**樣本層級的選擇偏差已充分定性記錄**——9/15 個排定臂因 congestion/timeout 完全無法產出 routed DEF(§2.2),偏差方向明確偏向「規模較大、利用率較高」的 case 被排除,這比單一樣本內的 net 級未繞線比例更嚴重 |
| **G7** | `Δio_after_commercial_DP` 顯示 M2/M3 改善在 DP 後流失 >50% | **未觸發(需要 F-INV,未執行)** | `Δio_after_commercial_DP`(§1.3)依賴 commercial DP(Innovus `place_detail`)重新擺放後的 evaluator 值;因 G1,本輪完全沒有 commercial DP 步驟,此量從未產生。**注意**:F-OR 沒有 detailed placement(規格 R9),即使 Innovus 打通後補跑,F-OR 側的 `Δio_after_commercial_DP` 仍然結構性缺失 |

**Gate 總覽:G1、G4 已觸發並依規則處置;G2/G3/G5/G6/G7 因上游判準(C1/C5)或前置步驟(F-INV)`not_evaluable`/未執行而無法正式評估,不代為裁決。**

---

## 6. 限制與 future work

1. **Innovus(G1)**——license 三台全不通,S6/S7/F-INV 全數未執行。所有「commercial-tool ground truth」的主張都尚未有數據支撐;打通後最優先要做的是拿同一批 6+9 個排定樣本重跑,對照 F-OR/F-INV 的 `route_or` vs `route_inv` 一致性(規格 R8)。
2. **per-net 落檔(限制 1 的根因)**——S8 的 `run_placement.py` 從未把 evaluator 的 per-net 陣列(`per_net_crossings`/`per_net_steiner`/`per_net_lambda`)寫入 `metrics.json`/`metrics.json.npz`。這是表②③⑦、C1、E2 全部 `not_evaluable` 的單一共同根因。修法需要在 routed placement 上重跑 `evaluate_gpu`(需 GPU),下一輪應把這個落盤動作補進 S8 的抽取流程,一次性解決最多個 `not_evaluable` 判定。
3. **`route_ft`(限制 2)**——S3 抽取呼叫從未傳 `pin_regions` 引數,全部樣本的 `per_net_route_ft` 是 `-1` sentinel。修法依賴限制 2 的根因(per-net pin-region 資料),與限制 1 同源。
4. **δ-scan 不完整(G4 相關)**——只有 δ∈{0(raw),2},規格要求的 {1,4} 未抽取;`route_cross_dw` 對 δ 的敏感度未量化,而 G4 已顯示 raw→dw(2) 本身漂移 54.3%,這個效應的形狀(單調遞減 vs 存在 plateau)仍未知。
5. **C5 需要 ≥3 個 design**——目前只有 2 個(des_perf_1、mempool_tile_wrap)。matrix_mult_1/superblue19/superblue12 三個候選全數卡在 OpenROAD DR 的規模天花板(§2.2)。下一輪要嘛換更低利用率/更小規模的第三個 ISPD2015 design(如 `mgc_edit_dist_a`,127k,自帶 REGIONS),要嘛投入更長 wall-clock 預算硬跑 matrix_mult_1。
6. **大 case router(規模天花板,本輪新發現)**——OpenROAD detailed route 在 ISPD2015 高利用率語料上的可行天花板約 15 萬 cell;superblue12(1.29M)、mempool_group(5M)、mempool_cluster(10M,Phase 1 §7 的黃金主力)這些 Phase 1 目標規模的 case,**用 F-OR 幾乎不可行**。這對整個 Stage 2 的規劃前提(§4.1 定案的 L1/L2 語料表)是一個實測層級的修正:L2(mempool_group/mempool_cluster)的 route 側校準,在 F-OR 單軌下不現實,只能等 Innovus(commercial router 通常有更成熟的 congestion-driven 演算法與更高並行度)或投入 GR-only 的替代量測(§5.3 已否掉但可重新評估「拿 GR guide 當粗略對照」作為規模天花板以上 case 的權宜方案)。
7. **未列入本報告但規格點名的殘留項**:V4(未繞線 net 比例)的逐樣本量化未落盤為獨立欄位;E1 的「未繞線 net 比例 ≤2% 並如實列出」因此無法逐樣本核對(§4 E1 判定已把這個缺口計入 FAIL 的理由之一)。
