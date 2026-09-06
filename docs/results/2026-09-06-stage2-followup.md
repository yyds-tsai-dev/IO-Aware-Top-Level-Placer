# Stage2 routing／逐net校準結果

本次分析仍為partial快照：4/12 samples，1/3 designs。
尚缺：des_perf_1__flat_k16、des_perf_1__flat_k32、des_perf_1__ours_k16、des_perf_1__ours_k32、mempool_tile_wrap__flat_k16、mempool_tile_wrap__flat_k32、mempool_tile_wrap__ours_k16、mempool_tile_wrap__ours_k32

完整執行不等於各項模型／品質門檻PASS。歷史報告保留，本頁僅使用本輪
OpenROAD routed geometry與逐net evaluator evidence；GPU重算、wire oracle、
instance/net/endpoint identity、input/output hashes與固定region/net order均經檢查。

## 樣本與可評估範圍

正式母體為matched、已route、degree2–256的signal nets，unrouted signal≤2%；
C4再取兩臂共同eligible net-name交集。FT未知值保留為未知，不補零。

| Sample | K | Formal nets | Unrouted fraction | FT coverage | DR violations |
|---|---:|---:|---:|---:|---:|
| mgc_fft_1__flat_k16 | 16 | 33306 | 0 | 1 | 61186 |
| mgc_fft_1__ours_k16 | 16 | 33306 | 0 | 1 | 59383 |
| mgc_fft_1__flat_k32 | 32 | 33306 | 0 | 1 | 61186 |
| mgc_fft_1__ours_k32 | 32 | 33306 | 0 | 1 | 59173 |

## C1／總量與逐net相關

各ratio為route_cross_dw(2)/evaluator總量。C1逐sample套用原門檻，pooled相關僅供描述。

| Sample | R λ−1 | R RG | R MST | ρ RG | ρ MST | C1 |
|---|---:|---:|---:|---:|---:|---|
| mgc_fft_1__flat_k16 | 1.03252 | 0.910002 | 0.831045 | 0.673667 | 0.639073 | reconsider_rg_model |
| mgc_fft_1__ours_k16 | 1.09113 | 0.942446 | 0.853257 | 0.67164 | 0.650045 | reconsider_rg_model |
| mgc_fft_1__flat_k32 | 1.1823 | 0.901368 | 0.832378 | 0.73358 | 0.692733 | reconsider_rg_model |
| mgc_fft_1__ours_k32 | 1.22229 | 0.912416 | 0.841227 | 0.73821 | 0.696887 | reconsider_rg_model |

## C2／C3／C5

NNLS rank=3、residual dof=16122、identifiable=True。
Net rows在design內相依；沒有IID-net CI，也沒有已驗證的跨design不確定性區間。

| Fit | α | β | γ | Rank | Dof |
|---|---:|---:|---:|---:|---:|
| pooled | 0.913072 | 0.988628 | 0.457172 | 3 | 16122 |
| mgc_fft_1 | 0.913072 | 0.988628 | 0.457172 | 3 | 16122 |

C2：`{"action": "E3 passed -- no io_calibrated column needed", "e3_pass": true, "pooled_ratios": {"R_io_mst": 0.8386237058803431, "R_io_rg": 0.913673081934259, "R_lam_minus_1": 1.1459520029881407}}`。

C3：`{"alpha_hat": 0.9130720900123415, "beta_hat": 0.9886280069829632, "kappa_ft": 1.08274912550399, "note": "kappa_ft <- beta_hat/alpha_hat per spec sec 8.4 C3 -- estimated real-router feed-through-to-necessary-crossing cost ratio; no placement objective or historical M3 weight is changed by this analysis.", "uncertainty": "clustered-design interval unavailable; estimate is not a validated objective weight"}`。此估計不改動placement objective或歷史M3權重。

C5：`{"n_designs_available": 1, "n_designs_required": 3, "reason": "fewer than 3 identifiable design fits", "verdict": "not_evaluable"}`。至少需要三個可辨識design fits。

## C4／同K方向

Formal verdict：**sign_invariant**。

| Design | From → To | Common nets | GP Δλ−1 | Route Δ | Scope | Flip |
|---|---|---:|---:|---:|---|---|
| mgc_fft_1 | flat_k16 → ours_k16 | 33306 | -315 | -86 | matched_K | False |
| mgc_fft_1 | flat_k32 → ours_k32 | 33306 | -325 | -132 | matched_K | False |

## G4／δ scan

所有δ的正式總量使用相同primary eligible mask；括號為whole-design supplementary總量。

| Sample | δ0 | δ1 | δ2 | δ4 |
|---|---:|---:|---:|---:|
| mgc_fft_1__flat_k16 | 5913 (5996) | 5913 (5996) | 4540 (4586) | 3017 (3036) |
| mgc_fft_1__ours_k16 | 5734 (5860) | 5734 (5860) | 4454 (4501) | 2989 (3003) |
| mgc_fft_1__flat_k32 | 9997 (10139) | 9997 (10139) | 7841 (7926) | 5399 (5438) |
| mgc_fft_1__ours_k32 | 9820 (10005) | 9820 (10005) | 7709 (7811) | 5366 (5406) |

## Boundary demand

Raw demand為crossing counts；長度正規化使用evaluator-coordinate units，不標成microns。

| Sample | Eval max / p90 / Gini | Route max / p90 / Gini | Eval / route normalized max |
|---|---|---|---|
| mgc_fft_1__flat_k16 | 336 / 283.8 / 0.128051 | 309 / 295 / 0.0876247 | 1.01434 / 0.93283 |
| mgc_fft_1__ours_k16 | 309 / 277.9 / 0.127219 | 310 / 277.5 / 0.0817783 | 0.93283 / 0.935849 |
| mgc_fft_1__flat_k32 | 336 / 285.5 / 0.25828 | 308 / 286.7 / 0.21447 | 1.35245 / 1.01434 |
| mgc_fft_1__ours_k32 | 326 / 278.2 / 0.26014 | 303 / 279.9 / 0.218318 | 1.29208 / 1.01434 |

## 限制與證據

固定detailed-route迭代上限5（iteration0–5），殘留DRC如上；不是signoff-clean。
本輪為OpenROAD-only，沒有Innovus或commercial-DP ground truth。CPU/GPU共享量測不作獨占效能宣稱。
完整逐net相關、degree／lambda buckets、FT decomposition與boundary資料均保存在tables.json。
輸入 `results/stage2_followup_20260906/calibration_partial/tables.json`；SHA256 `588d4d55cb6d8b4c209a569ea52a5c35bcd5cf7ee2cbf87341ee3c87c31683b0`。
圖表：[PNG](figs/stage2-followup-20260906.png)、[PDF](figs/stage2-followup-20260906.pdf)。
