# Stage2：open-source router ground truth／逐net校準結果

本次分析涵蓋完整註冊cohort：12/12 samples，3/3 designs。
尚缺：無。

完整cohort的≤2% signal缺線gate：**PASS**。
失敗樣本：無。

完整執行不等於各項模型／品質門檻PASS。歷史報告保留，本頁僅使用本輪
OpenROAD routed geometry與逐net evaluator evidence；GPU重算、wire oracle、
instance/net/endpoint identity、input/output hashes與固定region/net order均經檢查。

## 樣本與可評估範圍

正式母體為matched、已route、degree2–256的signal nets，unrouted signal≤2%；
C4再取兩臂共同eligible net-name交集。FT未知值保留為未知，不補零。

| Sample | K | Formal nets | Unrouted fraction | FT coverage | Final DR log violations |
|---|---:|---:|---:|---:|---:|
| mgc_fft_1__flat_k16 | 16 | 33306 | 0 | 1 | 61186 |
| mgc_fft_1__ours_k16 | 16 | 33306 | 0 | 1 | 59383 |
| mgc_fft_1__flat_k32 | 32 | 33306 | 0 | 1 | 61186 |
| mgc_fft_1__ours_k32 | 32 | 33306 | 0 | 1 | 59173 |
| des_perf_1__flat_k16 | 16 | 112877 | 0 | 1 | 206969 |
| des_perf_1__ours_k16 | 16 | 112877 | 0 | 1 | 203447 |
| des_perf_1__flat_k32 | 32 | 112877 | 0 | 1 | 206969 |
| des_perf_1__ours_k32 | 32 | 112877 | 0 | 1 | 208081 |
| mempool_tile_wrap__flat_k16 | 16 | 136208 | 0 | 1 | 29923 |
| mempool_tile_wrap__ours_k16 | 16 | 136208 | 0 | 1 | 46152 |
| mempool_tile_wrap__flat_k32 | 32 | 136208 | 0 | 1 | 29923 |
| mempool_tile_wrap__ours_k32 | 32 | 136208 | 0 | 1 | 43788 |

## C1／總量與逐net相關

各ratio為route_cross_dw(2)/evaluator總量。C1逐sample套用原門檻，pooled相關僅供描述。

| Sample | R λ−1 | R RG | R MST | ρ RG | ρ MST | C1 |
|---|---:|---:|---:|---:|---:|---|
| mgc_fft_1__flat_k16 | 1.03252 | 0.910002 | 0.831045 | 0.673667 | 0.639073 | reconsider_rg_model |
| mgc_fft_1__ours_k16 | 1.09113 | 0.942446 | 0.853257 | 0.67164 | 0.650045 | reconsider_rg_model |
| mgc_fft_1__flat_k32 | 1.1823 | 0.901368 | 0.832378 | 0.73358 | 0.692733 | reconsider_rg_model |
| mgc_fft_1__ours_k32 | 1.22229 | 0.912416 | 0.841227 | 0.73821 | 0.696887 | reconsider_rg_model |
| des_perf_1__flat_k16 | 0.742871 | 0.696129 | 0.646722 | 0.483378 | 0.432711 | rg_selected |
| des_perf_1__ours_k16 | 0.782868 | 0.70959 | 0.664303 | 0.527495 | 0.498264 | reconsider_rg_model |
| des_perf_1__flat_k32 | 0.832549 | 0.718927 | 0.662601 | 0.586301 | 0.524869 | rg_selected |
| des_perf_1__ours_k32 | 0.880522 | 0.715204 | 0.665244 | 0.649489 | 0.586038 | rg_selected |
| mempool_tile_wrap__flat_k16 | 0.814319 | 0.781732 | 0.652442 | 0.549126 | 0.507453 | reconsider_rg_model |
| mempool_tile_wrap__ours_k16 | 0.870152 | 0.824967 | 0.681131 | 0.514707 | 0.507748 | reconsider_rg_model |
| mempool_tile_wrap__flat_k32 | 0.850772 | 0.775303 | 0.657912 | 0.646945 | 0.604903 | reconsider_rg_model |
| mempool_tile_wrap__ours_k32 | 0.912867 | 0.813416 | 0.683719 | 0.659927 | 0.621408 | reconsider_rg_model |

## C2／C3／C5

NNLS rank=3、residual dof=109135、identifiable=True。
Net rows在design內相依；沒有IID-net CI，也沒有已驗證的跨design不確定性區間。

| Fit | α | β | γ | Rank | Dof |
|---|---:|---:|---:|---:|---:|
| pooled | 0.77198 | 1.06507 | 0.293774 | 3 | 109135 |
| des_perf_1 | 0.671547 | 1.03453 | 0.220747 | 3 | 32801 |
| mempool_tile_wrap | 0.748111 | 1.07044 | 0.277934 | 3 | 60206 |
| mgc_fft_1 | 0.913072 | 0.988628 | 0.457172 | 3 | 16122 |

C2：`{"action": "E3 failed -- io_calibrated column added", "e3_pass": false, "io_calibrated": [{"io_calibrated": 4164.169816420552, "residual": 375.8301835794482, "route_actual": 4540, "sample_id": "mgc_fft_1__flat_k16"}, {"io_calibrated": 3982.2552929448348, "residual": 471.74470705516524, "route_actual": 4454, "sample_id": "mgc_fft_1__ours_k16"}, {"io_calibrated": 7533.091138193677, "residual": 307.90886180632333, "route_actual": 7841, "sample_id": "mgc_fft_1__flat_k32"}, {"io_calibrated": 7360.315375524804, "residual": 348.68462447519596, "route_actual": 7709, "sample_id": "mgc_fft_1__ours_k32"}, {"io_calibrated": 7118.260887646789, "residual": -1022.2608876467893, "route_actual": 6096, "sample_id": "des_perf_1__flat_k16"}, {"io_calibrated": 6058.1330959366005, "residual": -812.1330959366005, "route_actual": 5246, "sample_id": "des_perf_1__ours_k16"}, {"io_calibrated": 11952.531033979558, "residual": -1685.5310339795578, "route_actual": 10267, "sample_id": "des_perf_1__flat_k32"}, {"io_calibrated": 10175.268130757806, "residual": -1604.268130757806, "route_actual": 8571, "sample_id": "des_perf_1__ours_k32"}, {"io_calibrated": 13107.079472891377, "residual": -937.079472891377, "route_actual": 12170, "sample_id": "mempool_tile_wrap__flat_k16"}, {"io_calibrated": 10809.936547079446, "residual": -308.9365470794455, "route_actual": 10501, "sample_id": "mempool_tile_wrap__ours_k16"}, {"io_calibrated": 21761.669744288876, "residual": -1921.6697442888762, "route_actual": 19840, "sample_id": "mempool_tile_wrap__flat_k32"}, {"io_calibrated": 20184.29928388649, "residual": -1085.2992838864884, "route_actual": 19099, "sample_id": "mempool_tile_wrap__ours_k32"}], "pooled_ratios": {"R_io_mst": 0.6960730932393541, "R_io_rg": 0.793303556207167, "R_lam_minus_1": 0.8973203955386205}}`。

C3：`{"alpha_hat": 0.7719803539235923, "beta_hat": 1.0650732754896188, "kappa_ft": 1.3796637052696754, "note": "kappa_ft <- beta_hat/alpha_hat per spec sec 8.4 C3 -- estimated real-router feed-through-to-necessary-crossing cost ratio; no placement objective or historical M3 weight is changed by this analysis.", "uncertainty": "clustered-design interval unavailable; estimate is not a validated objective weight"}`。此估計不改動placement objective或歷史M3權重。

C5：`{"cv": {"alpha": 0.1587358851062338, "beta": 0.03976627175314801, "gamma": 0.3871442131407153}, "n_designs_available": 3, "verdict": "fail"}`。至少需要三個可辨識design fits。

## C4／同K方向

Formal verdict：**sign_invariant**。

| Design | From → To | Common nets | GP Δλ−1 | Route Δ | Scope | Flip |
|---|---|---:|---:|---:|---|---|
| mgc_fft_1 | flat_k16 → ours_k16 | 33306 | -315 | -86 | matched_K | False |
| mgc_fft_1 | flat_k32 → ours_k32 | 33306 | -325 | -132 | matched_K | False |
| des_perf_1 | flat_k16 → ours_k16 | 112877 | -1505 | -850 | matched_K | False |
| des_perf_1 | flat_k32 → ours_k32 | 112877 | -2598 | -1696 | matched_K | False |
| mempool_tile_wrap | flat_k16 → ours_k16 | 136208 | -2805 | -1669 | matched_K | False |
| mempool_tile_wrap | flat_k32 → ours_k32 | 136208 | -2333 | -741 | matched_K | False |

## G4／δ scan

所有δ的正式總量使用相同primary eligible mask；括號為whole-design supplementary總量。

| Sample | δ0 | δ1 | δ2 | δ4 |
|---|---:|---:|---:|---:|
| mgc_fft_1__flat_k16 | 5913 (5996) | 5913 (5996) | 4540 (4586) | 3017 (3036) |
| mgc_fft_1__ours_k16 | 5734 (5860) | 5734 (5860) | 4454 (4501) | 2989 (3003) |
| mgc_fft_1__flat_k32 | 9997 (10139) | 9997 (10139) | 7841 (7926) | 5399 (5438) |
| mgc_fft_1__ours_k32 | 9820 (10005) | 9820 (10005) | 7709 (7811) | 5366 (5406) |
| des_perf_1__flat_k16 | 9813 (10000) | 9813 (10000) | 6096 (6156) | 4195 (4210) |
| des_perf_1__ours_k16 | 8307 (8717) | 8307 (8717) | 5246 (5306) | 3482 (3493) |
| des_perf_1__flat_k32 | 16164 (16450) | 16164 (16450) | 10267 (10368) | 6768 (6799) |
| des_perf_1__ours_k32 | 13534 (14111) | 13534 (14111) | 8571 (8680) | 5721 (5737) |
| mempool_tile_wrap__flat_k16 | 18951 (19372) | 18951 (19372) | 12170 (12338) | 6721 (6733) |
| mempool_tile_wrap__ours_k16 | 15729 (16242) | 15729 (16242) | 10501 (10707) | 6072 (6083) |
| mempool_tile_wrap__flat_k32 | 30600 (31227) | 30600 (31227) | 19840 (20085) | 10998 (11019) |
| mempool_tile_wrap__ours_k32 | 28398 (29114) | 28398 (29114) | 19099 (19373) | 11062 (11082) |

## Boundary demand

Raw demand為crossing counts；長度正規化使用evaluator-coordinate units，不標成microns。

| Sample | Eval max / p90 / Gini | Route max / p90 / Gini | Eval / route normalized max |
|---|---|---|---|
| mgc_fft_1__flat_k16 | 336 / 283.8 / 0.128051 | 309 / 295 / 0.0876247 | 1.01434 / 0.93283 |
| mgc_fft_1__ours_k16 | 309 / 277.9 / 0.127219 | 310 / 277.5 / 0.0817783 | 0.93283 / 0.935849 |
| mgc_fft_1__flat_k32 | 336 / 285.5 / 0.25828 | 308 / 286.7 / 0.21447 | 1.35245 / 1.01434 |
| mgc_fft_1__ours_k32 | 326 / 278.2 / 0.26014 | 303 / 279.9 / 0.218318 | 1.29208 / 1.01434 |
| des_perf_1__flat_k16 | 688 / 565.6 / 0.160195 | 740 / 630 / 0.181201 | 1.23964 / 1.33333 |
| des_perf_1__ours_k16 | 623 / 539.4 / 0.219909 | 642 / 574 / 0.226972 | 1.12252 / 1.15676 |
| des_perf_1__flat_k32 | 688 / 499.8 / 0.268356 | 740 / 514.7 / 0.272717 | 1.33393 / 1.33333 |
| des_perf_1__ours_k32 | 597 / 462 / 0.301988 | 650 / 470.2 / 0.306496 | 1.32315 / 1.17573 |
| mempool_tile_wrap__flat_k16 | 1676 / 1065.2 / 0.179625 | 1762 / 1094.6 / 0.185586 | 1.44877 / 1.52311 |
| mempool_tile_wrap__ours_k16 | 1616 / 852.9 / 0.227949 | 1678 / 913.9 / 0.230082 | 1.39691 / 1.4505 |
| mempool_tile_wrap__flat_k32 | 1676 / 949.7 / 0.294434 | 1762 / 969.8 / 0.301263 | 1.44877 / 1.52311 |
| mempool_tile_wrap__ours_k32 | 1603 / 903.3 / 0.296195 | 1702 / 934.8 / 0.304477 | 1.38567 / 1.47125 |

## 繼承routing的修正與重用範圍

tile首次route有4,360條零長度signal ROUTED記錄仍停在舊座標；
GRT將nonnull dbWire視為既有routing而跳過，恰好造成3.2006%缺線。
原始樣本與2% gate失敗完整保留。修正run先移除一般signal dbWire，
保留special／POWER／GROUND routing，再對相同placement執行GR＋DR。
沒有修改density、cohort、placement座標或coverage門檻。
FFT／DES四臂的NETS section各自逐byte相同，且沒有繼承signal routing；
因此保留其有效route與evidence，透過明示case symlink納入本次校準。
tile四臂使用獨立結果目錄；兩個flat K的DEF／netmap／CoordMap／config一致才重用route。
修正protocol：`/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/results/stage2_routing_input_fix_20260906/protocol.json`；SHA256 `297ef3fa2a349bc3efffb54893910203a26eab9ac575e167cd0697f775a6e627`。

## G4／raw與δ2雙值結論

正式總量raw=172960、δ2=116334；
差異為48.6754%，觸發15%門檻。以下raw為補充，δ2仍是主要判定。
兩者正式eligible母體相同。相關／回歸各自套用原有非零active規則；
raw新增active rows及兩個mask hash逐sample保存，相關係數／R²差異不視為完全相同rows的比較。

| Sample | Raw R λ−1 / RG / MST | Raw ρ RG / MST | C1 raw / δ2 | Active raw / δ2 |
|---|---|---|---|---|
| mgc_fft_1__flat_k16 | 1.34478 / 1.18521 / 1.08237 | 0.701543 / 0.722987 | reconsider_rg_model / reconsider_rg_model | 3390 / 3344 |
| mgc_fft_1__ours_k16 | 1.4047 / 1.21329 / 1.09847 | 0.695137 / 0.732867 | reconsider_rg_model / reconsider_rg_model | 3149 / 3112 |
| mgc_fft_1__flat_k32 | 1.50739 / 1.14921 / 1.06125 | 0.782992 / 0.791947 | reconsider_rg_model / reconsider_rg_model | 5001 / 4947 |
| mgc_fft_1__ours_k32 | 1.557 / 1.16227 / 1.07158 | 0.765784 / 0.782806 | reconsider_rg_model / reconsider_rg_model | 4790 / 4722 |
| des_perf_1__flat_k16 | 1.19583 / 1.12059 / 1.04106 | 0.551656 / 0.610856 | reconsider_rg_model / rg_selected | 7655 / 7438 |
| des_perf_1__ours_k16 | 1.23967 / 1.12363 / 1.05192 | 0.583863 / 0.617094 | reconsider_rg_model / reconsider_rg_model | 6186 / 6031 |
| des_perf_1__flat_k32 | 1.31074 / 1.13185 / 1.04318 | 0.668892 / 0.726274 | reconsider_rg_model / rg_selected | 11085 / 10838 |
| des_perf_1__ours_k32 | 1.39038 / 1.12934 / 1.05045 | 0.702248 / 0.73854 | reconsider_rg_model / rg_selected | 8696 / 8497 |
| mempool_tile_wrap__flat_k16 | 1.26805 / 1.2173 / 1.01598 | 0.676573 / 0.818177 | reconsider_rg_model / reconsider_rg_model | 13037 / 12915 |
| mempool_tile_wrap__ours_k16 | 1.30336 / 1.23568 / 1.02024 | 0.589083 / 0.804678 | reconsider_rg_model / reconsider_rg_model | 11012 / 10891 |
| mempool_tile_wrap__flat_k32 | 1.31218 / 1.19578 / 1.01472 | 0.775832 / 0.869186 | reconsider_rg_model / reconsider_rg_model | 19200 / 19036 |
| mempool_tile_wrap__ours_k32 | 1.35733 / 1.20945 / 1.01661 | 0.77652 / 0.879711 | reconsider_rg_model / reconsider_rg_model | 17519 / 17367 |

### E1–E5與G1–G7判定

| Exit | 本輪結果 |
|---|---|
| E1 | 12個樣本、3 designs×2 K×2臂，identity／≤2% coverage結果見主表；flat的同DEF明示重用。 |
| E2 | Spearman≥0.70：raw 10/12、δ2 2/12；未全數通過。 |
| E3 | Raw／δ2 ratio gate及in-sample校準殘差均已交付，數值見下表及JSON。 |
| E4 | Raw／δ2各7張表與C1／C5判定已交付。 |
| E5 | Raw／δ2的C4已完成，方向及差值見下表。 |

G1：本輪交付open-source router ground truth，僅相對OpenROAD；未執行本機Innovus license重試。
G2：C1存在reconsider_rg_model結果，RG模型需重審；本輪未據此改動M3 loss。
G3：依主要δ2的C5判定處理，跨benchmark遷移是後续使用者決策。
G4：觸發，雙值報告、δ scan與wire距離診斷已補齊。
G5／G7：本輪未執行Innovus／commercial DP，無法評估。
G6：原tile缺線由繼承stub造成，保留失敗後修正routing輸入；相同placement重繞後全樣本0%缺線。

| C2／C3／C5 | Raw | δ2 primary |
|---|---|---|
| E3 ratio gate | False | False |
| Pooled α / β / γ | 1.06367 / 1.00862 / 0.786664 | 0.77198 / 1.06507 / 0.293774 |
| κ FT（估計值） | 0.948245 | 1.37966 |
| C5 CV α / β / γ | 0.0385565 / 0.0430215 / 0.0974158 | 0.158736 / 0.0397663 / 0.387144 |
| C5 verdict | pass | fail |

E3判定如上；需要校準時的in-sample io_calibrated/residual與各design係數均保存於各自tables.json。
C5 raw=pass、主要δ2=fail。跨設計套用仍依主要δ2判定，估計值不是已驗證objective權重。
依G3，benchmark／Phase1遷移仍需另行決策；本輪未改benchmark或重寫既有exit verdict。

| Design / K | GP Δλ−1 | Route Δ raw / δ2 | C4 raw / δ2 |
|---|---:|---:|---|
| mgc_fft_1 / flat_k16 | -315 | -179 / -86 | same sign / same sign |
| mgc_fft_1 / flat_k32 | -325 | -177 / -132 | same sign / same sign |
| des_perf_1 / flat_k16 | -1505 | -1506 / -850 | same sign / same sign |
| des_perf_1 / flat_k32 | -2598 | -2630 / -1696 | same sign / same sign |
| mempool_tile_wrap / flat_k16 | -2805 | -3222 / -1669 | same sign / same sign |
| mempool_tile_wrap / flat_k32 | -2333 | -2202 / -741 | same sign / same sign |

C4 formal verdict：raw=sign_invariant、δ2=sign_invariant。
FT與boundary demand以raw visited regions／實際轉換計數，
不受δ過濾影響；已逐sample確認FT、wire length、pair demand完全相等。七張表均提供raw伴隨版本。

## G4／wire到邊界距離

以下為WIRE長度加權CDF：依垂直方向的lattice cell尺寸正規化距離，裁切到die內，保留decoded segment重複量。
Any為到任一internal grid boundary的最近距離；Parallel只計與wire方向平行的boundary。
這個距離不是δ的run-length閾值。d=0的沿界wire有正長度，單純穿越交點的長度為零。
此分布提供幾何診斷；不能單凭raw／δ2差異就斷言沿界走線是唯一或主要原因。

| Sample | Any ≤1 / ≤2 cells | Parallel ≤1 / ≤2 cells | Outside die WIRE DBU |
|---|---|---|---:|
| mgc_fft_1__flat_k16 | 2.120% / 4.586% | 0.961% / 2.364% | 1.6124e+06 |
| mgc_fft_1__ours_k16 | 2.111% / 4.605% | 0.985% / 2.444% | 1.4818e+06 |
| mgc_fft_1__flat_k32 | 3.575% / 7.510% | 1.632% / 3.750% | 1.6124e+06 |
| mgc_fft_1__ours_k32 | 3.648% / 7.581% | 1.740% / 3.905% | 1.5172e+06 |
| des_perf_1__flat_k16 | 2.989% / 5.802% | 1.699% / 3.265% | 426000 |
| des_perf_1__ours_k16 | 2.765% / 5.446% | 1.674% / 3.278% | 598600 |
| des_perf_1__flat_k32 | 4.605% / 8.975% | 2.516% / 4.894% | 426000 |
| des_perf_1__ours_k32 | 4.269% / 8.412% | 2.517% / 4.941% | 341600 |
| mempool_tile_wrap__flat_k16 | 2.690% / 5.460% | 1.382% / 2.881% | 6.36172e+07 |
| mempool_tile_wrap__ours_k16 | 2.248% / 4.598% | 1.179% / 2.479% | 6.10106e+07 |
| mempool_tile_wrap__flat_k32 | 4.169% / 8.439% | 2.042% / 4.259% | 6.36172e+07 |
| mempool_tile_wrap__ours_k32 | 3.887% / 7.847% | 1.953% / 4.034% | 6.32026e+07 |

Raw evidence `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/results/stage2_routing_input_fix_20260906/g4_raw/tables.json`；SHA256 `f40190383928676302276e8e3794839ec5ef9b9a943b0d18b143cabc354441da`。
Distance protocol／12份結果／hash receipt：`/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/results/stage2_routing_input_fix_20260906/wire_boundary_distance`。
距離圖：[PNG](figs/stage2-followup-20260906-distance.png)、[PDF](figs/stage2-followup-20260906-distance.pdf)。

## 限制與證據

固定detailed-route迭代上限5（iteration0–5），殘留DRC如上；不是signoff-clean。
本輪為OpenROAD-only，沒有Innovus或commercial-DP ground truth。CPU/GPU共享量測不作獨占效能宣稱。
完整逐net相關、degree／lambda buckets、FT decomposition與boundary資料均保存在tables.json。
GP與routed evaluator的差值也跨越native pin保留／OpenDB endpoint與pin-center慣例，
不能全部歸因於router移動cell；routing前後的OpenDB幾何變化由獨立identity artifact記錄。
輸入 `results/stage2_routing_input_fix_20260906/calibration/tables.json`；SHA256 `d1174aecd3ee56c20c63b1581225bc39d616cf60af5659d3d26927e356eba15f`。
圖表：[PNG](figs/stage2-followup-20260906.png)、[PDF](figs/stage2-followup-20260906.pdf)。
