# Stage 2 S9 -- 校準分析(人讀摘要)

- 規格:`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md` §8(七張表、§8.3 迴歸、§8.4 C1-C5)、§10 S9 列
- 產生腳本:`ioplace/diagnostics/stage2_calibration.py`(純 CPU;不跑 GPU/route,只讀 `results/stage2/s8/**` 既有 artifact)
- 機器可讀全量:`results/stage2/calibration/tables.json`
- 樣本語料:`results/stage2/s8/{des_perf_1,mempool_tile_wrap}__{flat,ours_k16,ours_k32}/`(6 個有效樣本);L0 演練樣本 `results/stage2/rehearsal/mgc_fft_1/` 不計入統計(依裁決)

**這一輪最重要的兩個限制,先講在前面(細節見各表):**

1. **S8 從未把 evaluator 的 per-net 陣列(`per_net_crossings`/`per_net_steiner`/`per_net_lambda`)落盤** —— `metrics.json` 只有純量總量(`io_count`/`ft_count`/`hard_lambda_sum`/`io_rg`/`ft_rg`/`tree_wl`/`hpwl`),`metrics.json.npz` 只有 `node_x`/`node_y`(`ioplace/drivers/run_placement.py:408`)。要拿到 per-net evaluator 陣列必須在 routed placement 上重跑 `evaluate_gpu`,那需要 GPU,本輪任務明確禁止。**後果:所有 per-net 對照(表②、表③、表⑦、C1 的 ρ 半)一律 `not_evaluable`,只有總量級比值可算(表①、表⑤、表⑥)。**
2. **`per_net_route_ft` 在全部 6 個樣本裡都是 `-1`(未計算) sentinel**(`ioplace/route_eval/route_crossings.py:373`:`evaluate_route` 沒收到 `pin_regions` 引數時,每個 net 的 `route_ft` 都填 `-1`;S8 的抽取呼叫從未傳這個引數)。**後果:表⑥的 `route_ft` 欄位整欄 `not_evaluable`,`ft_rg`/`ft_mst`(evaluator 總量)不受影響。**

---

## 七張表產出情況

| # | 表 | 狀態 | 說明 |
|---|---|---|---|
| ① | evaluator vs route 總量比 + per-net 對照 | **部分可評** | 總量比(`R_X = Σroute/ΣX`)可算;per-net Pearson/Spearman `not_evaluable`(限制 1) |
| ② | per-net 相關(Spearman/Pearson) | **not_evaluable** | 限制 1 |
| ③ | per-degree bucket | **not_evaluable** | net degree 未落盤於任何 S8 artifact,需重讀 placedb(超出「不重跑 pipeline」範圍) |
| ④ | per-Λ bucket | **降級可評** | 只能用 route 側自身的 `per_net_lambda_route` 分桶(非 route-vs-evaluator 對照) |
| ⑤ | 三值分解總表 | **可評** | design × K × arm × router,6 列 |
| ⑥ | FT 三值表 | **部分可評** | `ft_rg`/`ft_mst` 可評;`route_ft` `not_evaluable`(限制 2) |
| ⑦ | boundary-pair demand 相關 | **not_evaluable** | evaluator 的 `boundary_pair_demand` 從未落盤(同限制 1 根因) |

外加規格要求的:樣本清單表(§8.2-4)、不可評樣本表(§8.2-5,9 個樣本)、§8.3 迴歸、C1-C5 判定、G4 檢查——全部產出,見下。

---

## 表①:evaluator vs route 總量比

`R_X = Σ route_cross_dw(δ=2) / Σ X`,X ∈ {`lam_minus_1`, `io_rg`, `io_mst`}(matched-K,即該樣本 placement/evaluator 實際使用的 K)。

| sample | R_lam_minus_1 | R_io_rg | R_io_mst |
|---|---:|---:|---:|
| des_perf_1__flat | 0.7484 | 0.6997 | 0.6480 |
| des_perf_1__ours_k16 | 0.7951 | 0.7172 | 0.6719 |
| des_perf_1__ours_k32 | 0.8857 | 0.7221 | 0.6742 |
| mempool_tile_wrap__flat | 0.6784 | 0.6511 | 0.5432 |
| mempool_tile_wrap__ours_k16 | 0.7331 | 0.6949 | 0.5782 |
| mempool_tile_wrap__ours_k32 | 0.7568 | 0.6741 | 0.5695 |
| **pooled(Σroute/ΣX,跨全部 6 樣本)** | **0.7569** | **0.6869** | **0.5970** |

**讀法**:route 實測值全面**低於** evaluator 的三把尺(所有比值 < 1),且 `io_mst`(MST 幾何模型)高估最嚴重(pooled 0.597,即 route 只有 io_mst 預測的 59.7%),`io_rg`(RG/Steiner 模型)次之(0.687),`lam_minus_1`(routing-independent 下界)最接近 1(0.757)——三者的**相對順序**與 E3 的「總量比落在 [0.80,1.25]」門檻對照:**三者全部落在區間之外**(E3 fail),因此觸發 C2(見下)。per-net Pearson/Spearman 因限制 1 未計算,無法確認這個總量偏差是系統性平移還是掩蓋了 per-net 排序錯亂。

來源:`results/stage2/s8/<design>__<arm>/metrics.json`(`hard_lambda_sum`/`io_rg`/`io_count`)+ `crossings_k{K}.json`(`total_route_cross_dw`,matched K)。

## 表②/③/⑦:not_evaluable(逐項見上方限制)

## 表④:per-Λ_route 自身分桶(降級,非 evaluator 對照)

桶界沿用 M3 draft §2.2 的 Λ 分桶慣例(1 / 2 / 3 / 4-8 / >8),但這裡的 Λ 是**route 側自己的** `per_net_lambda_route`,不是與 evaluator 匹配的量測。跨 6 樣本 pooled 淨數與平均 `route_cross_dw`(僅列非空桶,完整表見 `tables.json` 的 `table4_lambda_route_bucket`):

| Λ_route bucket | 說明 |
|---|---|
| 1 | 絕大多數 net 落在這裡(單樣本 mean λ_route 介於 0.96-1.08,見下方「樣本清單」的 λ_route 統計)—— 與 evaluator 的 degree≤1 濾除慣例(`evaluator_ref.py:104`)方向一致但**不是同一個量**,不可直接類比 |
| 2/3/4-8/>8 | 隨 K 增大(K=32 樣本)人數增加,`des_perf_1__ours_k32` 的 λ_route max=32,`mempool_tile_wrap__ours_k32` max=31 |

（完整分桶明細 —— 每桶 n_nets / sum_route_cross_dw / mean_route_cross_dw / sum_route_wl —— 在 `tables.json:table4_lambda_route_bucket.rows`。）

## 表⑤:三值分解總表(design × K × arm × router)

| design | K | arm | router | Σ(λ-1) | Σio_rg | Σroute(dw2) | Σio_mst | mst_excess | route−io_rg | io_mst−route |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| des_perf_1 | 16 | flat | OR | 8038 | 8599 | 6016 | 9286 | 687 | -2583 | 3270 |
| des_perf_1 | 16 | ours_k16 | OR | 6682 | 7410 | 5314 | 7909 | 499 | -2096 | 2595 |
| des_perf_1 | 32 | ours_k32 | OR | 9866 | 12099 | 8737 | 12958 | 859 | -3362 | 4221 |
| mempool_tile_wrap | 16 | flat | OR | 14990 | 15617 | 10169 | 18720 | 3103 | -5448 | 8551 |
| mempool_tile_wrap | 16 | ours_k16 | OR | 12151 | 12823 | 8908 | 15407 | 2584 | -3915 | 6499 |
| mempool_tile_wrap | 32 | ours_k32 | OR | 21014 | 23604 | 15911 | 27941 | 4337 | -7693 | 12030 |

**觀察**:`route − io_rg` 全部為負(route 實測比 RG 模型的 Steiner 下界還低),`io_mst − route` 全部為正且量級可觀(2.6k-12k)——這與方向 A 的 `mst_excess` 診斷一致:MST 幾何系統性高估,RG 模型雖然貼近但**仍然高估**(route 比兩者都低)。這與 §1.2 原本預期的排序 `(λ-1) ≤ io_rg ≤ route ≤ io_mst` **不一致**——本輪實測是 `route < io_rg`,即 route 甚至低於 RG 下界。可能原因(未驗證,留給後續):(a) F-OR 只是 GR+DR 演練,§6.1 R9 已知 OpenROAD 無 detailed placement,GP+LG 直入 router 的幾何可能比 RG 假設更緊湊;(b) run-length filter(δ=2)把大量邊界貼線 crossing 濾掉,見下方 G4。

## 表⑥:FT 三值表

| sample | ft_rg | ft_mst | route_ft |
|---|---:|---:|---|
| des_perf_1__flat | 561 | 622 | not_evaluable |
| des_perf_1__ours_k16 | 728 | 795 | not_evaluable |
| des_perf_1__ours_k32 | 2233 | 2360 | not_evaluable |
| mempool_tile_wrap__flat | 627 | 733 | not_evaluable |
| mempool_tile_wrap__ours_k16 | 672 | 719 | not_evaluable |
| mempool_tile_wrap__ours_k32 | 2590 | 2678 | not_evaluable |

`route_ft` 整欄 `-1` sentinel(限制 2)。`ft_rg < ft_mst` 在全部 6 樣本上成立,方向與 M3 draft 的 RG 模型設計預期一致(RG 是 crossing-minimal,MST 幾何的 feed-through 應該 ≥ RG),但因為缺乏 route 側的真值,無法完成三值排序的驗證。

---

## 樣本清單表(§8.2-4)

| sample | K(placement) | route_cross_dw(matched K) | other K | route_cross_dw(other K) | DR 最終 violations(log) | drc.rpt 行數 | route elapsed | verify_s2 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| des_perf_1__flat | 16 | 6016 | 32 | 10236 | 206867 | 23671 | 9555s(2:39:15) | true |
| des_perf_1__ours_k16 | 16 | 5314 | 32 | 9599 | 202875 | 23773 | 9554s | true |
| des_perf_1__ours_k32 | 32 | 8737 | 16 | 5379 | 209940 | 23726 | 9668s | true |
| mempool_tile_wrap__flat | 16 | 10169 | 32 | 16597 | 41387 | 20041 | 4566s(1:16:06) | true |
| mempool_tile_wrap__ours_k16 | 16 | 8908 | 32 | 15615 | 43651 | 20055 | 4649s | true |
| mempool_tile_wrap__ours_k32 | 32 | 15911 | 16 | 9946 | 42750 | 20062 | 4681s | true |

**注**:`route_cross_dw(other K)` 是用**不匹配**的 K 網格重抽 crossing 的結果(僅供靈敏度參考,不能配對 evaluator 的固定-K 總量,因為 evaluator 從未在那個 K 上重跑);兩欄都來自各樣本自己的 `crossings_k16.json`/`crossings_k32.json`。DR violations 數量都非常大(20k-24k drc.rpt 行、20 萬+ 最終 violation 計數)——這些 route 完全不是 DRC-clean 的,只是拿到了完整的 wire geometry(V1/V2/verify_s2 全綠,即 instance/net 集合守恆),符合 spec §6.2「DRC 數記錄、不作 gate」的設計。route elapsed 取自 `or_run/openroad_route.log` 最後一次 `DRT-0267` 行(緊接 `DONE_ROUTE` 前)。

## 不可評樣本表(§8.2-5,9 個)

| sample | 失敗模式 | 證據摘要 |
|---|---|---|
| matrix_mult_1__flat | congestion_plateau_timeout | `-droute_end_iter 5` 全跑完後 violations 仍卡在 ~544,931;4h(14400s)wall-timeout 前最後記錄 40% 完成、elapsed 26:47 |
| matrix_mult_1__ours_k16 | congestion_plateau_timeout | 同上;最後 528,897 violations,50% 完成,elapsed 30:59 |
| matrix_mult_1__ours_k32 | congestion_plateau_timeout | 同上;最後 532,973 violations,50% 完成,elapsed 30:17 |
| superblue19__flat | congestion_plateau_timeout | 0th iteration 本身跑超過 3h,violations 從 124,751(10%)飆到 2,199,855(50%),elapsed 3:06:13 時被 4h timeout 殺掉 |
| superblue19__ours_k16 | congestion_plateau_timeout | 同上;2,317,033 violations @ 50%,elapsed 3:08:34 |
| superblue19__ours_k32 | dr_0th_iteration_timeout | log 停在「Start 0th optimization iteration」,**連一條 `Completing X%` 進度行都沒有**——4h 預算在第一個 checkpoint 之前就耗盡 |
| superblue12__ours_k16 | dr_0th_iteration_timeout | 同上,同樣停在 0th iteration 起點、無任何進度行 |
| superblue12__ours_k32 | killed_during_global_route_setup(證據較弱) | log 僅 2.8KB,停在 `GRT-0300` warning 之後、global_route 進度或 detailed_route 起點之前;`route_chain_sb12.log` 從未為這臂寫過 `[FAIL]` 行,較像外層批次腳本本身被中斷,而非 route.tcl 乾淨地跑完並失敗 |
| superblue12__flat | route_not_attempted_queue_exhausted | 無 `or_run/` 目錄;`out.def`(169,029,018 bytes)是全 S8 語料裡最大的檔案,`stage2_s8_route.sh` 以「最小檔案優先、concurrency=1」排程,`extract_chain.log` 證實:`[FAIL] superblue12__flat: no routed.def (route not run yet?)`——不是 route 失敗,是批次時間預算沒排到它 |

（每列的 log 路徑見 `tables.json:unevaluable_sample_table.rows[].openroad_route_log` / `.route_chain_logs`。）

---

## §8.3 迴歸(C2 的輸入)

`route[e] ≈ α·(λ_e−1) + β·(ST_e−(λ_e−1)) + γ·(io_mst_e−ST_e)`——**per-net 形式不可用(限制 1),改在每個樣本的總量級擬合**:6 個樣本各貢獻一列 `(x1=Σ(λ-1), x2=Σft_rg=ΣST-Σ(λ-1), x3=Σmst_excess=Σio_mst-ΣST) -> y=Σroute_cross_dw(δ=2)`,非負最小平方(NNLS,無截距)。

| 係數 | 值 | bootstrap 95% CI(over 6 rows,2000 次重抽) |
|---|---:|---:|
| α̂ | **0.6518** | [0.3247, 0.7227] |
| β̂ | **0.9661** | [0.3053, 2.8755] |
| γ̂ | **0.0000**(NNLS 夾到邊界) | [0.0000, 0.9701] |

R² = **0.9946**,殘差絕對值 148-339(相對殘差 -2.0% 到 +4.8%)。

**誠實揭露 n=6 的限制(依指示)**:6 個觀測、3 個非負係數 = 僅 3 個殘差自由度;R²=0.995 在這個 dof 下**幾乎不能拒絕任何合理的線性形式**,不代表模型被強烈驗證。bootstrap CI 本身也只是對 6 列重抽,樣本量太小,CI 寬度(尤其 β̂ 的 [0.31, 2.88])本身就說明這個估計不穩定。

**per-design 分開擬合(僅供參考,不是判定)**——3 樣本 3 係數,0 殘差自由度(精確擬合,無穩定性資訊):

| design | α̂ | β̂ | γ̂ |
|---|---:|---:|---:|
| des_perf_1 | 0.6935 | 0.8532 | 0.0000 |
| mempool_tile_wrap | 0.0000 | 0.8891 | 3.1416 |

兩個 design 的 `(α̂,β̂,γ̂)` 差異巨大(des_perf_1 的 γ̂=0 vs mempool 的 γ̂=3.14,des_perf_1 的 α̂=0.69 vs mempool 的 α̂=0)——這正是 C5(轉移檢驗)本來要量化的變異,但 3-樣本精確擬合本身沒有自由度可信,只能當作「即使有數字,現在也不該被信任」的佐證,而不是 CV 的正式輸入。

---

## C1-C5 判定

| # | 判定 | 結果 |
|---|---|---|
| **C1**(模型選擇) | **not_evaluable** | 需要 per-net ρ(io_rg,route) vs ρ(io_mst,route),限制 1 擋住。**補充證據(非判定)**:pooled 總量比 `|R_io_rg-1|=0.313` < `|R_io_mst-1|=0.403`,方向上偏向 RG 模型,但單靠總量比不滿足 C1 的 AND 條件,不構成正式裁決 |
| **C2**(迴歸校準) | **觸發**(E3 fail) | pooled 總量比全部落在 [0.80,1.25] 之外(0.757/0.687/0.597),依規則新增 `io_calibrated` 欄(= α̂(λ-1)+β̂(ST-(λ-1))+γ̂(io_mst-ST)),見上方迴歸表的殘差欄,已寫入 `tables.json:c2_regression_calibration.io_calibrated` |
| **C3**(objective 權重) | **可判** | `κ_ft ← β̂/α̂ = 0.9661/0.6518 = 1.4821`(取代 M3 預設的 `κ_ft=1.0`);由於 α̂/β̂ 本身建立在 n=6 的迴歸上,這個 κ_ft 的可信度繼承同樣的限制 |
| **C4**(sign-invariance) | **可判:sign_invariant** | flat→ours_k16→ours_k32 在 `hard_lambda_sum`(pre-route)與 `route_cross_dw`(post-route)兩把尺上,兩個 design 的方向**全部一致**(4 組比較,0 次翻轉)——des_perf_1 的 flat→k16 兩者皆降(-1356 / -702),k16→k32 兩者皆升(+3184 / +3423);mempool_tile_wrap 同型態。**沒有發生方向翻轉**,M2/M3 既有 exit 判準在校準後的量上維持原結論 |
| **C5**(轉移檢驗) | **not_evaluable** | 需要 ≥3 個 design 才能算 CV;本輪只有 2 個 design 有完整路由樣本(des_perf_1、mempool_tile_wrap)。matrix_mult_1/superblue19/superblue12 三個 design 全數落在不可評樣本表,無法補足第三個 design |

## G4(raw vs dw(2) 漂移檢查,§9.2)

`Σroute_cross_raw = 84,954`,`Σroute_cross_dw(2) = 55,055`,漂移 = **+54.3%**(遠超 15% 門檻,**觸發 G4**)。「沿邊界貼線走」是主要效應,表①/⑤的所有結論理論上都該雙值陳述——本報告的主表用 dw(2)(spec 指定的主報表值),raw 值列在表①(`route_cross_raw` 欄)供對照。**δ∈{0,1,4} 的完整掃描本輪未做**(S8 只抽取了 δ=2;要補 δ=1/4 需要重跑 `route_eval.route_crossings`,超出本輪「只讀既有 artifact」的範圍),這是 L-Q4-a 仍未解決的部分。

---

## 限制清單(彙總)

1. **per-net evaluator 陣列未落盤**——影響表②③⑦、C1 的 ρ 半、迴歸只能做總量級而非 per-net。補救需要 GPU 重跑 `evaluate_gpu`,本輪範圍外。
2. **`route_ft` 全樣本 sentinel(-1)**——影響表⑥的 route 欄。補救需要重跑 S3 抽取並傳入 `pin_regions`(這本身又依賴 per-net evaluator 資料,回到限制 1)。
3. **C5 只有 2 個 design 可評,不足規格要求的 ≥3**——matrix_mult_1/superblue19/superblue12 三個 design 全數落在不可評樣本表(見上)。
4. **迴歸 n=6、3 殘差自由度**——R²=0.995 在這個 dof 下資訊量有限;bootstrap CI 對 6 列重抽同樣脆弱(β̂ 的 CI 寬達 [0.31,2.88])。
5. **δ 掃描不完整**——只有 δ∈{0(raw),2} 的資料,{1,4} 未抽取。
6. **G4 已觸發(漂移 54.3%)**——「沿邊界走」效應顯著,主表值(dw2)可能仍系統性偏低估真正的「有意義」crossing 數;本輪未做 wire-to-boundary 距離分佈診斷(spec G4 建議項)。
7. **F-OR only(G1)**——Innovus license 全程不通(規劃書 §3.1 實測),本輪全部樣本是 OpenROAD ground truth,不是 commercial-tool ground truth;所有結論依 spec 標題規範降級為「相對 OpenROAD 的校準」。
8. **route 幾乎全部非 DRC-clean**(20k-24k drc.rpt 違規行、20萬+ 最終 violation 計數)——crossing 抽取用的是「有完整 wire geometry 但未收斂」的 routed DEF,不是乾淨的 signoff 結果,spec §6.2 明訂此為可接受設計(DRC 記錄、不作 gate),但讀者需知道這不是收斂 route。
9. **表⑤觀察到 `route < io_rg`(route 甚至低於 RG Steiner 下界)**,與 §1.2 原設計預期的 `(λ-1) ≤ io_rg ≤ route ≤ io_mst` 排序不符——本報告只如實記錄,未深究成因(候選解釋見表⑤下方注記),留待後續(可能需要與 G4 的 δ 掃描一起看)。
