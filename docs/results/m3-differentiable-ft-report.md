# M3 可微 feed-through 報告(最終,T8)

- 日期:2026-08-15;分支 `m2-differentiable-io`,報告基準 HEAD `4064db8`
- 環境:`docs/dev-env.md`(`DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`、`$DP/.venv312/bin/python`、torch 2.8.0+cu128、NVIDIA L4、`dp_commit d971880`)
- 對應設計:`docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`(v3.1)
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §9 M3 列(line 138)
- 正式輸入裁決:`docs/results/2026-08-14-m3-s4-adjudication.md`(commit `6d46c1e`)
- 前置報告:`docs/results/m2-differentiable-io-report.md`(M2 exit PASS,但 `ft_count` 反升)
- 慣例:沿用 M0/M1/M2 的誠實記錄慣例——**FAIL 不粉飾、negative result 逐條寫出、事後判斷一律標明**。

---

## 0. 摘要

### 0.1 三句話版本(照裁決書,不修改)

1. 依 §3.4.3(line 345)/§8 T0-c(line 662)的預註冊規則,**7 個 S4 候選全部淘汰,M3 不把可微 FT 項放進 objective**。
2. 淘汰的機制可陳述:`ReLU(soft-crossings − detached-baseline)` 家族在梯度上恆等於 IO 項本身(S4e `cos(∇L_IO, ∇L_FT) = 0.947`,直接量測);唯一有實質 FT 方向的 **S4a-star**(低 τ trajectory snapshot 上 T1a **17/18**、效應 4–13σ)在支撐檢定(t2c **3/26**)與 HPWL 附帶代價上不合格。
3. 三個判準缺陷(T1-P3 結構性不可通過、T1e 連現任者都違反、T1c 帶寬過窄)已記錄;**全部修復後 S4a 仍 19/30 = 63% < 80%**,故不重跑、不改判準。

### 0.2 本報告追加的兩句

4. **預註冊 fallback 已執行且是 null result**:§3.4.3 的建設性條款「M1 式離散 reweight + FT 診斷」以 `ft_rg` 為 per-net 訊號實作(commit `fe5b474`),adaptec1 k16 grid 四臂(`results/m3/reweight/`)相對 `R0`(= M2-best,逐位元復現)的 `Δft_mst` = **+7 / +26 / −16**、`Δft_rg` = **+15 / +32 / +6**,以 `3σ_seed(ft_mst) = 62.9`、`3σ_seed(ft_rg) = 55.13` 為尺,**全部 |Δ| < 0.6σ**。
5. **T11 maze 校驗通過,`ft_rg` 報表可信**:20,000 條抽樣 net 上 RG 相對 lattice 最短路的相對偏差 **中位 0、p90 0**、無負偏差,`bounded_detour_needed = false`(`results/m3/probes/probe_maze_sample.json`)。唯一例外是 M2-best 的 Λ≥4 尾巴(中位 0.25、p90 0.667),已列入低信心。

### 0.3 Exit 一行結論

**phase-1 spec §9 M3 第一條(「FT 顯著下降且 WL/IO 無明顯劣化」)FAIL**;第二條(「完整 ablation 表」)以縮編形式 PASS。⇒ **M3 exit FAIL**。v3.1 §6.2 的 E1/E5/E6 FAIL、E2/E3/E4 N/A、E7 PASS;§6.3 的 C2/C3/C4/C5/C9 臂 VOID。詳見第 6 節。

---

## 1. 動機:M2 留下的 FT 回歸

M3 的存在理由是 M2 報告 §3 的這張表(`docs/results/m2-differentiable-io-report.md:79-100`):

| case | 指標 | flat | M1 reweight | M2 best |
|---|---|---:|---:|---:|
| adaptec1 k16 grid | `io_count` | 30,256 | 29,869(−1.28%) | **24,647(−18.54%)** |
| | `ft_count` | 2,545 | 2,430(−4.52%) | **3,525(+38.51%)** |
| | `hpwl` | 73,923,673 | 74,856,866(+1.26%) | 74,895,086(+1.31%) |
| bigblue4 k16 grid | `io_count` | 100,333 | 100,860(+0.53%) | **63,755(−36.46%)** |
| | `ft_count` | 6,935 | 2,948(−57.49%) | **8,136(+17.32%)** |

也就是說:**可微 IO 項把 IO 交叉壓下去的同時,把 feed-through 推上去**;而**離散 reweight(M1)在沒有 IO 項時反而把 FT 壓下去**。M3 的問題是:能不能把 FT 直接寫進 objective,拿回這個回歸而不還回 IO 的收益。

答案(本報告):**在預註冊的判準下不能**;而且退回離散 reweight 的路線,在 IO 項已開啟的組態下也沒有效果。

---

## 2. 設計迭代史(v1 → v3.1,四輪審查)

這一節本身是方法論貢獻:M3 的主要產出不是一個 op,而是**一套把「看起來有效的 surrogate」逐層否證掉的協定**。四輪審查文件都在 `docs/reviews/`(該目錄共 6 份,其中 2 份屬 M4)。

| 輪次 | 審查文件 | 被推翻的東西 | 結果 |
|---|---|---|---|
| **v1 → v2** | `2026-08-13-m3-draft-v1-adversarial-opus.md`(F1–F15)+ `2026-08-13-m3-draft-v1-adversarial-codex.md`(1–16) | (a) v1 選定的 **S4b 全部證據都在 hard limit**:soft regime 下總量高估 5.14×、83.0% 的懲罰質量落在真 FT=0 的 net 上(Opus F1 實測);(b) **S7 的立論前提是統計假象**(v1 的「邊界需求 +42% 惡化」是 median 假象);(c) Codex #1:`Λ≥4` 用 metric-closure MST 卻宣稱是精確 Steiner/FT 恆等式 | 新增 P0 探針;S7 進入清算;`st_exact/ft_exact` 與 `st_ub/ft_ub` 分離 |
| **P0 實測** | `results/m3/probes/probe_ft_surrogate_soft.json` | v1/v2 的**整個候選排序**:13 個候選 × 18 格,`ratio_vs_true ∈ [0.7,1.5]` 且 `mass_on_ft0 < 0.20` 的**全過者 0 個**;最佳 `S4b-gated_beta{0.5,1.0}` 各 **4/18**;Opus F2 提的 hardbase 修法 **0/18** | 「先驗選型」路線宣告失敗 |
| **v2 → v3** | `2026-08-13-m3-v2-adversarial-codex.md`(6 BLOCKER + 2 MAJOR + 1 MINOR) | (a) v2 用來翻案的純量效用 **`U = 17.4` 實際是 `87/5`**——分母只有 5 條 net(base `io_rg` 的 0.022%);噪聲底只有單一隨機向量;RMS 正規化與 scheduler 的 L1 錯配;缺 WL/density/Nesterov/`ft_mst` ⇒ **`U` 廢除**;(b) F3 的 worked example 證明 **S4a 結構性偏好「星」勝過「鏈」**,而 v2 的 L12 診斷偵測不到;(c) F4:callback 契約沒有真正 apply,M2 driver 的 refresh 看到舊係數;(d) F5:ramp 與其「晚窗口」論述自相矛盾 | **v3 不選 winner**,改交付預註冊的 P0b;新增 overlap-aware 的 `S4e-union`;新增 G9 鏈轉星 gate;callback 改原子化七步 |
| **v3 → v3.1** | `2026-08-13-m3-v3-verify-codex.md`(9 項中 7 RESOLVED、F1/F5 PARTIALLY、2 新 BLOCKER) | (a) **P0b 未機械凍結**(候選寫 6 實列 7、狀態寫 16 實為 20、隨機方向在 IO-anchored 家族下的尺度未定義);(b) **schedule 係數歧義**(`ρ_io` 已含 ramp、`λ_io` 又乘一次);(c) ramp 校準表的 0.255/0.700 是三位小數 `τ_rel` 的產物,正確值 0.2523/0.7082 | 新增 §3.4.0「凍結的基數」表(**1,684 格**);ramp 全 schedule 只乘一次;七步交易明確掛在 `iteration % 50 == 0` 的 evaluator 分支 |

**每一輪都推翻了前一輪的「答案」,而三次推翻用的都是同一種手法:把宣稱搬到它實際會被使用的 regime 去量。** v1 的證據在 hard limit、v2 的證據在 5 條 net 的分母、v3 的基數在文件裡自相矛盾。最終 P0b 之所以能給出可信的否定答案,是因為它把 regime(真實 trajectory 中段 snapshot)、正規化(L1,與 scheduler 一致)、噪聲(8 個隨機方向的分布)、主判準(未動過的 `ft_mst`)四件事同時釘死。

**代價也要誠實記:** 這四輪把 M3 的實驗預算幾乎全部花在「選型」上,`f_ft` in-loop 掃描(T6)一格都沒跑。P0b 的單步性質(L13)因此仍是**最大殘餘風險**——見第 8 節。

---

## 3. P0b:預註冊的 S4 選型實驗

### 3.1 協定(v3.1 §3.4,執行前已 commit,`preregistration.doc_sha256` 寫進 JSON)

- **臂**:7 個候選 + `IO-only` 參照 = 8 個方向臂(`S4a-star`、`S4b-gated-detach_b{0.5,1.0}`、`S4b-gated_b0.5`、`S4e-union-{detach,hardbar}`、`S4g-bboxcov`)。
- **狀態**:20 個 `(state, τ_rel)` 格 = 12 個真實 trajectory snapshot(`results/m3/snapshots/`,T0-a 產出,`manifest.json` 記 `bit_exact_vs_m2_x/y = true`,`τ_rel` 從 0.2260 到 0.0420)+ 4 個最終 placement × 2 個溫度;另有 bigblue4 縮減子矩陣。
- **步長**:三個家族全用 L1 正規化。判定家族 = `merged_l1`;`io_anchored_l1` 只報不判;`full_objective`(含 `g_wl_density`)為 confirmatory。`η ∈ {0.005, 0.01, 0.02}`,`f = 0.25`。
- **噪聲帶**:每個 `(state, η)` 8 個高斯方向,`3·sd_rand` 為帶寬(§3.3.5 證明其 L1 大小與 `IO-only` 參照臂相同,故與家族無關)。
- **判準**:tier-1(T1a–T1e 五條,聚合成 T1-P1 ≥80% / T1-P2 高溫零違反 / T1-P3 full-objective ≥60%)為必要;tier-2(T2a–T2d 支撐安全)有否決權。
- **執行**:commit `58c3b7e`(探針)+ `da818f4`(完整矩陣),輸出 `results/m3/probes/probe_p0b.json`。

### 3.2 結果:7/7 淘汰

`summary.per_candidate[*]`(逐字):

| 候選 | T1-P1 通過率(低 τ) | T1-P2 高溫違反 | T1-P3(full-obj) | tier-2 | verdict |
|---|---:|---:|---:|---|---|
| `S4a-star` | **0.0625** | 2 | 0.0 | false | **淘汰** |
| `S4b-gated-detach_b0.5` | 0.0 | 18 | 0.0 | false | 淘汰 |
| `S4b-gated-detach_b1.0` | 0.0 | 18 | 0.0 | false | 淘汰 |
| `S4b-gated_b0.5` | 0.0 | 18 | 0.0 | false | 淘汰 |
| `S4e-union-detach` | 0.0 | **0** | 0.0 | false | 淘汰 |
| `S4e-union-hardbar` | 0.0 | **0** | 0.0 | false | 淘汰 |
| `S4g-bboxcov` | 0.0 | **0** | 0.0 | false | 淘汰 |

門檻是 T1-P1 ≥ 0.80。最高者 0.0625(= 2/32)。**沒有任何候選接近。**

補充一個容易誤讀的數字:`S4a-star` 在判定家族的**全部 62 格**中有 **17 格**通過完整的 tier-1 五條,但其分布是「低 τ 2 格、中 τ 8 格、高 τ 7 格」——**判定用的低溫區正是它最弱的地方**,而 T1-P1 只數低 τ 格。這不是判準挑格,而是 §3.4.3 預先登記的 regime 劃分(低 τ = GP 末端 = 真正要動 FT 的地方)。

執行規模:`cells` **1,090** 格 + `random_bands` **372** 筆(= 62 個 `(state, η)` × 6 個指標,對應 496 個隨機方向評估)+ `failed_cells` **2** 筆(皆為 `S4g-bboxcov` 在 bigblue4 上的 CUDA OOM,依 §3.4.0 計為未通過,S4g 分母 60)。

### 3.3 機制診斷:為什麼全滅(裁決書 §B,本報告逐項重算確認)

以下所有數字為**判定家族 `merged_l1`、`primary` tie-break、低 τ(`τ_rel ≤ 0.10`)、adaptec1、n = 30 格**,由 `probe_p0b.json` 重算。

#### B1 單步 FT 效用**存在**,而且很大(這是 P0b 最重要的正面發現)

| 候選 | T1a 通過 | 中位 `Δft_mst − Δft_mst(IO-only)` | 以 `3·sd_rand` 為單位 | 中位 `cos(∇L_IO, ∇L_FT)` |
|---|---:|---:|---:|---:|
| **S4a-star** | **23/30** | **−199 counts** | **−4.34**(帶寬 = 3σ ⇒ **−13.0σ**) | 0.399 |
| `S4b-gated_b0.5` | 2/30 | −58 | −1.29 | 0.085 |
| `S4e-union-detach` | 1/30 | −48 | −0.96 | **0.947** |
| `S4e-union-hardbar` | 1/30 | −53 | −1.12 | **0.944** |
| `S4b-gated-detach_b1.0` | 3/30 | −23.5 | −0.39 | 0.025 |
| `S4b-gated-detach_b0.5` | 0/30 | −10 | −0.22 | 0.152 |
| `S4g-bboxcov` | 0/30 | −12 | −0.31 | −0.001 |

只看 12 個真實 trajectory snapshot 的低 τ 18 格:**S4a 的 T1a 17/18、T1b 18/18、T1d 18/18**(T1c 僅 6/18、T1e 僅 7/18——見 B4)。

**隨機帶沒有吞掉效果**:低 τ 的 30 格中,`3·sd_rand(Δft_mst)` 的中位僅為 `IO-only` 自身 FT 效應的 **37.6%**;逐格帶寬 20.4–134.1(含 bigblue4 則 5.3–134.1),而 S4a 的負超額落在 **−66 到 −470**(26/30 格為負)。

**同時,`IO-only` 參照臂在 30/30 個低 τ 格都讓 `ft_mst` 上升(+43 … +1,054)**——M2 報告的 `ft +38.5%` 在單步解析度上被完整重現。⇒ 資料說的不是「FT 與 IO 不能同時改善」,而是:**純 IO 方向確實在製造 FT,而按 hop 距離加權的 S4a 能把它扳回來一部分。**

#### B2 6/7 是機制性死亡,不是「測不到」

- **S4e(兩個變體)`cos = 0.947 / 0.944` ⇒ 它就是 IO 項本身。** 機制:`L_FT = ReLU(L_cross − base)`,`base` 是 detach 的(v3.1 §3.3.3 line 225)。detach 後在 active set 上 `∇L_FT = ∇L_cross`,而 `L_cross_e = Σ_a U_{e,a}` 正是 soft crossing 計數 = M2 的 IO surrogate。**FT = crossings − Steiner;把第二項 detach 掉,梯度裡就沒有任何 FT-specific 資訊。** S4e 是無害的同義反覆。這同時結案 L15(tie-break 敏感度對一個恆等於 IO 的項沒有意義)。
- **S4g-bboxcov**:`cos ≈ −0.001`、`‖∇L_FT‖₁/‖∇L_IO‖₁ ≈ 658`(⇒ `κ ≈ 3.8e−4`),行為是一個弱 WL 力(中位 HPWL 超額 **−2.22 pp**,即比 IO-only 更省線長),FT 效用 0(T1a 0/30)。⇒ §6.3 的 C5 臂依 §3.3.4 line 247 直接移除。
- **S4b 家族**:§3.2 的「0 次齊次 + gate 梯度盲區」診斷成立。`cos` 0.025–0.152、FT 超額 ≤1.3 倍帶寬、中位 HPWL 超額 **+1.77 … +2.18 pp**。

#### B3 「tier-2 近全過、tier-1 全零」不是弔詭,而是兩者正交

tier-2 適用 regime 為 `τ_rel ≤ 0.0775`(n = 26 格):

| 候選 | t2a `mass_on_ft0 ≤ 0.5` | t2b Λ=2 桶保真 | t2c 稀疏桶 ≤5.0 | 中位 `mass_on_ft0` |
|---|---:|---:|---:|---:|
| `S4b-gated-detach_b1.0` | 25/26 | 26/26 | **26/26** | 0.194 |
| `S4e-union-detach` | 16/26 | 26/26 | 20/26 | 0.378 |
| **`S4a-star`** | 16/26 | 26/26 | **3/26** | 0.327 |

`S4a` 的 `bucket_share_ratio["4+"]` 中位 **13.43**(門檻 5.0)⇒ **方向有用,但支撐嚴重偏向大/遠 net**(已知的 star 偏誤)。`S4b-gated-detach_b1.0` 則相反:**支撐放對了,但那個方向沒用**(T1a 3/30)。tier-1 量「往那個方向走有沒有用」,tier-2 量「懲罰落在哪些 net 上」,P0b 證明兩者獨立;§3.4.4 把 tier-1 排在前面是對的。

#### B4 三個判準缺陷(記錄;不改變結論)

1. **T1-P3 結構性不可通過(客觀,可逐位元驗證)。** 12 個 snapshot 中有 6 個是 `flat`(`λ_io = 0`),full-objective 方向 `g_full = g_wl_density + λ_io·C` 在這 6 格對**所有 8 個臂完全相同**——本報告直接從 JSON 驗證:`flat_it0300/0350/0400/0450/0500/0550` 各自 8 個臂的 `(d_ft_mst, d_io_mst, d_hpwl)` **distinct = 1**。⇒ T1-P3 的 ≥60%(≥8/12)理論上限只有 6/12。此外全 objective 單步在鋪散階段必然讓 FT 惡化:這 6 格的 `Δft_mst` = **+1,304 / +1,449 / +1,781 / +2,828 / +6,418 / +7,395**,T1a 的絕對條款 `Δft_mst < 0` 不可能成立(S4a 在 full-objective 家族 T1a = **0/12**)。**報告不得引用 M3 家族結果作為候選優劣的證據。**
2. **T1e 連現任者都不過(客觀)。** `IO-only` 在 30 個低 τ 格中有 **6 格**違反 `sign(Δio_rg) = sign(Δio_mst)`。若把 T1e 限縮成只看 FT 兩尺,S4a 的 T1e 從 **14/30 提升到 24/30**(整格 tier-1 通過數 2 → 3)。
3. **T1c 的帶寬是抽樣噪聲尺度而非無差異尺度(判斷)。** HPWL 的 `3·sd_rand` 僅為 `IO-only` 自身 HPWL 代價的 **1.24%**,T1c 因此實際讀作「FT 力必須免費」,與本專案接受 HPWL 代價的立場(M1 +1.26%、M2 +1.31%、E3 談的是收斂後)不一致。S4a 的中位 HPWL 超額只有 **+0.399 pp**。

#### B5 修好判準也翻不了案(反事實重算,n = 30,本報告獨立重算確認)

| 修法 | S4a tier-1 通過數 |
|---|---:|
| 原判準(T1a–T1e 全要) | **2/30** |
| T1e 只看 FT 兩尺 | 3/30 |
| HPWL 容忍 +0.5 pp | 7/30 |
| 兩者都改 | 12/30 |
| **T1c、T1e 整條刪掉** | **19/30 = 63.3%** |
| 再限 `η ∈ {0.005, 0.01}` | 14/20 = 70% |

要摸到 80% 還必須再丟掉 4 個最終 placement 狀態(S4a 在 slicing-final 的 T1a 僅 **1/6**)。**三個事後動作才能救活一個候選 = 沒救活。判準修訂 + 重跑不做。** 這也讓「重開 P0b」這條路正式關閉。

#### B6 協定偏差揭露(必記)

1. **T2d 從未裁決**:JSON 無 `t2d` 鍵,`random_bands` 也沒有 chain→star 這個量的帶 ⇒ 無法事後補判。verdict 由 tier-1 決定,不影響結論,但必須記。
2. **chain→star = 19,653 是全部 1,090 格 × 8 臂的總和**;逐臂 `S4a` 2,945、`IO-only` 本身 2,049(`S4b-gated-detach_b1.0` 最高,3,033)。低 τ 判定家族的**淨**轉換(chain→star 減 star→chain)對 S4a 是 **−152**(n=32,含 bigblue4;僅 adaptec1 n=30 則為 −202),**是負的** ⇒ Codex F3 的 star 偏誤在 placement 層級**未被證實**,只在 surrogate 值域成立(即 t2c)。
3. `failed_cells = 2`(S4g bigblue4 OOM),依 §3.4.0 計為未通過。
4. **格數帳**:§3.4.7 的預算表列 1,684 格,其中把 M3 家族的隨機帶另計 96 格;但 §3.3.5 與 §3.4.0 已規定隨機方向與家族無關、每 `(state, η)` 只產生一次。實作照後者 ⇒ 實際執行 1,092 個候選格(1,090 成功 + 2 失敗)+ 496 個隨機方向。**這是預註冊文件內部的重複計數,不是執行偏差**,但依 §3.4 的揭露義務記於此。

---

## 4. Phase B:離散 ft-reweight(預註冊 fallback)

### 4.1 為什麼是這條路,而不是「純 negative result 收工」

§3.4.3 line 345 的 fallback 是**建設性條款**:「沒有候選通過 tier-1 ⇒ M3 不做可微 FT 項,改走**M1 式離散 reweight + FT 診斷**的縮減範圍」。裁決書 §C 據此把 Phase B 重新定義為五個預註冊內動作(reweight 主線、T11、`ft_rg` 的 `σ_seed`、S4 臂作廢、T8 逐條判 exit)。**in-loop pilot 被預註冊明文禁止**(§3.4.2 scope 表 line 304:「P0b 未通過的候選連 in-loop 都進不去」;L13 line 437),因此 S4a 的 in-loop pilot 只能是 post-registration 的新實驗(見 7.3)。

### 4.2 實作(commit `fe5b474`,係數改動,不新增梯度路徑)

- `run_io()` 新增 `--ft-reweight {off,on}` 與 `--alpha-ft`(命名沿用 M1 的 `--alpha-io` 慣例)。
- 在既有 `every = 50` 的 evaluator 分支內、與既有 `alpha_io` 區塊同一事件順序,計算 per-net `FT_rg_e = ST_e − max(Λ_e − 1, 0)`(`ioplace/drivers/run_placement_io.py:266-271`),`ST_e`/`Λ_e` 直接取自該次已經做過的 `evaluate()`,**不新增 evaluator pass**。
- 權重公式**原封不動重用** `ioplace/reweight.py:4-14` 的 `update_net_weights`:`w = 1 + α·min(signal, cap)`。
- `ft_reweight='on'` 與 `alpha_io>0` 互斥(兩者都寫同一個 `w` buffer),在進 placement 迴圈前 fail fast(`run_placement_io.py:150-159`)。
- 每次更新後 `obj_version += 1`,維持 §5.5 的版本 invariant。

### 4.3 R0 的逐位元鎖

`R0`(`--ft-reweight off`)與 M2-best 存檔(`results/m2/sweep/best_x1.json[.npz]` = `adaptec1_k16_rho0.40_annealed`)**逐位元相同**:每個純量指標與 `node_x`/`node_y` 全等(`results/m3/reweight/summary.json` 的 `r0_bit_identical_to_m2_best_archive: true`,commit `4064db8` 訊息載明驗證方式)。單元測試另鎖 `--ft-reweight off` 與完全不給旗標逐位元相同、且 `alpha_ft` 非預設時仍是 no-op(`tests/test_ft_reweight.py:152`)。⇒ **R1–R3 相對 R0 的差異只可能來自 reweight 本身。**

### 4.4 結果表(adaptec1 k16 grid、`det=1`、`dp_seed=1000`、`ρ_max=0.40` annealed、`every=50`)

原值(`results/m3/reweight/summary.json` 的 `raw`,各臂 JSON 在同目錄):

| 臂 | `alpha_ft` | `io_mst` | `ft_mst` | `hpwl` | `io_rg` | `ft_rg` | `λ_io_final` | `spearman_rho` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat(`m2/noise/flat_seed1000`) | — | 30,256 | 2,545 | 73,923,673 | 26,921 | 2,454 | — | — |
| **R0** `off`(= M2-best) | — | 24,647 | 3,525 | 74,895,086 | 22,719 | 3,382 | 340.684 | 0.5160 |
| R1 `on` | 0.25 | 24,748 | 3,532 | 74,947,730 | 22,808 | 3,397 | 336.971 | 0.5129 |
| R2 `on` | 0.50 | 24,659 | 3,551 | 74,925,259 | 22,784 | 3,414 | 332.387 | 0.5064 |
| R3 `on` | 1.00 | 24,810 | 3,509 | 74,901,305 | 22,848 | 3,388 | 331.666 | 0.5029 |

相對 R0 的差(`delta_vs_R0`;正號 = 更差),以 `results/m3/probes/ft_rg_seed_noise.json` 的 5 顆 flat seed 噪聲為尺(`3σ(ft_mst) = 62.93`、`3σ(ft_rg) = 55.13`):

| 臂 | `Δft_mst` | 佔 `3σ` | `Δft_rg` | 佔 `3σ` | `Δio_mst` | ≤ +492 | `Δhpwl` | ≤ +0.5 pp |
|---|---:|---:|---:|---:|---:|:--:|---:|:--:|
| R1 | **+7** | 0.111 | +15 | 0.272 | +101 | yes | +0.070% | yes |
| R2 | **+26** | 0.413 | +32 | 0.580 | +12 | yes | +0.040% | yes |
| R3 | **−16** | −0.254 | +6 | 0.109 | +163 | yes | +0.008% | yes |

四臂 runtime 85.3–88.7 s、`peak_mem_mb` 全部 601.146(四次獨立 process,且 driver 已帶 `peak_mem_mb_reset_semantics: true`,詳見 5.5)、`num_callbacks = 12`、`backtrack_median = 1.0`(G8 穩定性無異常)。

### 4.5 Null result 陳述(正式)

> **在 adaptec1 k16 grid、IO 項開啟(`ρ_max = 0.40` annealed)的組態下,以 `ft_rg` 為 per-net 訊號、寫入 IO 項 per-net 權重的離散 reweight,在 `α_ft ∈ {0.25, 0.5, 1.0}` 三個強度上,對 `ft_mst` 與 `ft_rg` 都沒有超出 seed 噪聲的效果(全部 |Δ| < 0.6σ,最佳臂 R3 為 −0.25σ)。IO 與 HPWL 亦無顯著變化(皆在 screening 門檻內)。**

尺度對照:同一把 `σ_seed` 尺下,**M1 的效果是 −108 counts = −5.1σ**(`ft_count` 2,430 vs flat 5 顆 seed 的 mean 2,538.0,`ft_rg_seed_noise.json`);M3 Phase B 最佳臂是 **−0.25σ**。兩者差 20 倍。

### 4.6 「IO 項啟用後 ft reweight 失效」——發現、機制假說、與識別性警告

這是本里程碑的第二個實質發現,但**必須連同它的識別性缺陷一起陳述**。

**發現(事實層):** M1(無 IO 項、reweight 開)`ft_count` 2,430;M2/M3-R0(IO 項開、無 reweight)3,525;M3 Phase B(IO 項開、reweight 開)3,509–3,551。**加上 reweight 沒有把 3,525 拉回 2,430 的方向,連 1σ 都沒動。**

**機制假說(標為假說,未經實驗識別):**

- **H1「支撐太窄」**:`ft_rg` 訊號的支撐只有 **2,942 條 net**(`probe_m3_rg.json` 的 `A2_best.n_ft_pos`,即 R0 的 placement),佔 216,930 條 active net 的 **1.36%**;而 M1 用的 crossings 訊號支撐是 λ≥2 的 **17,533 條(8.08%)**,寬約 6 倍。不過這些 net 至少貢獻 `hard_lambda_sum = 19,337` 中的 2,942(≥15.2%),所以支撐窄**不足以單獨解釋**完全沒有效果。
- **H2「槓桿不同,而且是零和」**:**M1 寫的是 DREAMPlace 的 `net_weights`(線長權重,`ioplace/drivers/run_placement_reweight.py:48-49`,實測 `net_weights_max = 3.0`),M3 Phase B 寫的是 `io_term.w`(IO surrogate 的 per-net 權重,`run_placement_io.py:270`,進入 `L_io = Σ w_e (λ_e − 1)⁺`,`ioplace/ops/io_term.py:120,147`)。** 前者**新增**一股真實的線長拉力;後者只在一個**被 auto-normalization 固定住總量**的力裡重分配——`λ_io` 依 §5.2 由力量佔比 `ρ` 反解,`w` 一放大,`λ_io` 就等比縮回去。**這個補償在資料上直接看得到**:`λ_io_final` 隨 `α_ft` 單調下降 340.684 → 336.971 → 332.387 → 331.666(α=1.0 時 −2.65%)。同時 `spearman_rho`(IO 項 per-net 貢獻 vs evaluator)單調變差 0.5160 → 0.5029 ⇒ reweight 有效地**扭曲了 IO 項的對齊**卻沒換到 FT。
- **H3「這個槓桿在 M2 已經被驗過一次」**:M2 的 A4 臂(改 `io_term.w` 為 `inv_deg`)在 adaptec1 上被 A2 嚴格 Pareto 支配(`docs/results/m2-differentiable-io-report.md:182-206`)。`io_term.w` 作為旋鈕的既有紀錄就是「動它沒有好處」。

**識別性警告(必讀):** M1 與 M3 Phase B **同時差三件事**——(i) 訊號(crossings vs `ft_rg`)、(ii) 槓桿(WL `net_weights` vs `io_term.w`)、(iii) IO 項在不在場。**因此「IO 項啟用後 ft reweight 失效」這個標題敘述在目前的資料下並未被識別**;能被資料支持的只有更窄的敘述:「**以 `ft_rg` 訊號改寫 IO 項的 per-net 權重,在 IO 項開啟時無效**」。要把因果講清楚,需要下面這個 2×2(見 7.4),M3 沒有跑。

---

## 5. 交付資產(縮編後仍然成立的部分)

### 5.1 evaluator 的 RG 雙尺報表(T1,commit `0a2e32c`)

新增 `ioplace/region_graph.py`:`region_graph(rg) -> (adj, D, ell)`、`next_hop_table`、分層 `steiner_tree_stats`(Λ≤1 trivial、Λ=2 最短路、**Λ∈[3,8] 精確 Dreyfus–Wagner 含完整回溯**、Λ>8 metric-closure MST),`evaluator_ref`(CPU)與 `evaluator_gpu`(GPU,Λ≤3 向量化 closed form、Λ≥4 委派同一函式)**共用同一份程式碼**。`EvalResult` 新增 `io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home`;舊欄位逐位元不變(既有 CPU/GPU 等價測試全綠 + 對存檔 flat JSON 的新回歸)。

依 Opus F11,Λ≥4 的解**一律展開成 `G_R` 上的實際子樹**(union-find 去重最短路邊),`ST` 取邊數、`ft_rg` 取非 terminal 頂點數 ⇒ **`FT = ST + 1 − Λ` 每條 net 精確成立**,CPU/GPU 依構造逐位元一致。`tests/test_region_graph.py` 12 個測試鎖住:F3 worked example、Λ=2 最短路、隨機小圖恆等式、對 brute-force 的精確比對、真實 k32 grid 上 MST 層仍是可行樹。

**雙尺報表**(`results/m3/probes/probe_m3_rg.json` / `probe_m3_bb.json`,`mst_excess = io_mst − io_rg`):

| run | `io_mst` | `hard_λ_sum` | `io_rg` | `ft_rg` | `ft_mst` | `mst_excess` | `detour_mst` |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 k16 grid flat | 30,256 | 24,467 | 26,921 | 2,454 | 2,545 | 3,335 | 5,789 |
| adaptec1 k16 grid M2 best | 24,647 | 19,337 | 22,720 | 3,383 | 3,525 | 1,927 | 5,310 |
| adaptec1 k32 grid flat | 47,839 | 34,485 | 42,585 | 8,100 | 8,281 | 5,254 | 13,354 |
| adaptec1 k32 grid M2 | 41,272 | 27,985 | 37,405 | 9,420 | 9,600 | 3,867 | 13,287 |
| adaptec1 k16 slicing flat | 36,615 | 25,951 | 31,461 | 5,510 | 7,389 | 5,154 | 10,664 |
| adaptec1 k16 slicing M2 | 33,460 | 22,202 | 28,600 | 6,398 | 8,812 | 4,860 | 11,258 |
| bigblue4 k16 grid flat | 100,333 | 81,205 | 87,953 | 6,748 | 6,935 | 12,380 | 19,128 |
| bigblue4 k16 grid M1 | 100,860 | 85,580 | 88,438 | 2,858 | 2,948 | 12,422 | 15,280 |
| bigblue4 k16 grid M2 | 63,755 | 54,518 | 62,589 | 8,071 | 8,136 | 1,166 | 9,237 |

讀法:(1) **grid 上兩把 FT 尺幾乎逐條相同**(總量差 0.8–4.0%),真正的分歧只在 slicing(`ft_rg/ft_mst` = 0.746 / 0.726);(2) 恆等式把 detour 拆開——adaptec1 flat 的 5,789 = 2,454 真 FT + 3,335 MST 次佳,bigblue4 M2 的 9,237 = 8,071 + 1,166 ⇒ **S2 已經吃掉 MST 次佳的部分**;(3) `mst_excess` 佔 detour 13–65% 且隨幾何劇烈漂移,這正是當初否掉「只用 MST 幾何」的主要理由。

**一個必須揭露的 1-count 差異**:上表 `A2_best` 列的 `io_rg`/`ft_rg` 是**探針自帶實作**的值(22,720 / 3,383);**T1 evaluator 對同一個 placement 給 22,719 / 3,382**(`results/m3/reweight/adaptec1_k16_R0_off.json`,R0 與該存檔逐位元相同)。差 1 count(`io_rg` 0.004%、`ft_rg` 0.03%),方向與成因都符合預期:探針在 Λ≥4 用 metric-closure MST 的**權重**(Codex v1 審查 #1 指出的缺陷),T1 改為展開成實際子樹並去重共用邊 ⇒ `ST_T1 ≤ ST_probe`;該 placement 上 Λ≥4 的 net 有 359 條。**以 T1 evaluator 為準;flat 列(26,921 / 2,454)兩者一致。**

**記憶體/runtime**:同 commit 把 `pin_bit_acc`/`passed_bit_acc` 從 `(E,K)` int64 改為與 source bit plane **成對 int8**(`scatter_reduce_` 要求 dtype 相同;0/1 上 `amax ≡ OR`),`K` 上界收緊為 ≤32。commit 訊息記載 bigblue4 k16 evaluator 峰值 **−21.3%**、steady-state runtime **+23%**(在 <+30% 預算內)。**這兩個數字目前只存在於 commit 訊息,沒有 JSON artifact**(列入 8. 未決)。

### 5.2 T11:maze 抽樣校驗(commit `a956790`,`ft_rg` 可信度的唯一機制)

在 `RegionGrid.grid` 這個**字面上的 512×512 region-id lattice**(evaluator 幾何查詢用的同一個陣列)上跑 crossing=1、length=ε 的 Dijkstra,對 A0_flat 與 A2_best 各抽 10,000 條 Λ≥2 的 net,量 RG 的 `per_net_steiner`(取自 T1 `evaluator_ref`)相對可實現路徑的樂觀誤差。

| 量 | 值 |
|---|---|
| 樣本數 | 20,000(2 tag × 10,000,`n_terminal_mismatch_skipped = 0`) |
| 相對偏差中位 / p90 | **0 / 0** |
| 絕對偏差平均 | 0.0386 |
| 負偏差(RG 比可實現路徑還長) | **0** 條 |
| 偏差直方圖 | 0:19,421(97.1%)、1:458、2:98、≥3:23 |
| `bounded_detour_needed`(門檻 0.2) | **false** |
| 分 Λ:Λ=2(n=18,472) | 中位 0 / p90 0 |
| 分 Λ:Λ=3(n=1,073) | 中位 0 / p90 0.5 |
| 分 Λ:Λ≥4(n=455) | 中位 0.2 / p90 0.5;**A2_best 子集 中位 0.25 / p90 0.667,`bounded_detour_needed = true`** |

**裁決:L1 結案。** 設計 §2.4 的降級條款是「中位偏差 > 20% ⇒ 所有 `ft_rg` 敘述降級為拓撲下界」;實測中位 **0%**,遠低於門檻,故**本報告的 `ft_rg` 數字不降級**。結構性理由:grid regions 是矩形且相鄰 = 共邊,Λ=2 的最短 hop 路徑在 lattice 上恆可實現,而 Λ=2 承載絕大部分 FT 質量(bigblue4 flat:6,517/6,748 = 96.6% 的 `ft_rg` 來自 Λ=2,`probe_m3_bb.json`)。**唯一例外是 Λ≥4 的尾巴**(佔樣本 2.3%),在 M2-best 上 RG 樂觀約 25%——這正是 Λ>8 用 metric-closure MST 的已知鬆弛(L2),對總量影響 1–5%。

### 5.3 S7:artifact 診斷,定案不進 objective

v1 主張加邊界容量項的兩個前提**都被實測推翻**:

**(a) 面積容量沒有惡化**(`probe_free_area_util.json`,`l7_triggered = false`、`n_nonpositive_free_area_regions = 0`):

| case | 口徑 | flat max/mean | M2-best max/mean | 變化 |
|---|---|---:|---:|---:|
| adaptec1 k16 grid | free area | 1.9271 | 1.8948 | −1.67% |
| adaptec1 k16 grid | total area | 2.2081 | 2.1843 | −1.08% |
| bigblue4 k16 grid | free area | 1.8725 | 1.8033 | −3.70% |
| bigblue4 k16 grid | total area | 1.9825 | 2.1654 | +9.23% |

**(b) 「邊界需求 +42% 惡化」是 median 假象**——絕對峰值 `max demand/len` 在 5/5 組態下**下降 8.3–20.6%**(adaptec1 k16 27.969→23.617、k32 27.969→24.578、k16 slicing 23.358→21.412、bigblue4 k16 70.094→55.625);上升的是 `max/median` 比值,因為 median 掉得更快。

加上 v1 的 `L_cap` 公式本身有四個獨立結構缺陷(星形流幾何與 §2.1 不一致、`C_ab` 0 次齊次、`softplus` 不是 hinge 且 `D≥0 ⇒ σ(z)≥0.269` 使「零梯度凍結」成為死碼、`λ_cap` 量綱不符且破壞 `obj_version` invariant)⇒ **`κ_cap` 不存在,S7 降為診斷欄位,M4 條件重開**(重開前置條件見 7.2)。

### 5.4 基建與兩道逐位元鎖

| 資產 | commit | 內容 | 鎖 |
|---|---|---|---|
| **T0-a** snapshot 基建 | `157a198` | `--snapshot-iters/--snapshot-dir`,在 N=50 的 evaluator 分支存 `node_x/node_y/iteration/overflow/tau/gamma/density_weight/ratio_ema/lambda_io` + `g_wl_density`;產出 12 個 snapshot(`results/m3/snapshots/manifest.json`) | **鎖 1**:不給 snapshot 參數時與現行 run 逐位元相同(`tests/test_driver_io_snapshot.py:25`);兩條重跑軌跡的最終 npz 對 `results/m2/` 存檔 `bit_exact_vs_m2_x/y = true`;梯度一致性 `g_wl_density + λ_io·g_io` vs `obj_and_grad_fn` 相對誤差 < 1e−6(`:94`) |
| **T3** schedules | `01181a7` | `f_ft_max`/`tau_start`/`tau_full`/`ft_ramp_mode`/`kappa_ft`/`kappa_max`/`eps_rel`/`Cmax`/`home_version`;`τ_rel` 兩點 ramp(sample-and-hold);§5.5 原子化七步交易 | `tests/test_schedules.py` 32 個純函數測試(ramp 端點與四個中間 golden 值、`constant` 模式無分母、`f_ft_max=0` 關閉、`obj_version` 一次 callback 恰好 +1) |
| **T0-b** 密封重發 | `ed9c5bb` | 8 支既有探針改 repo 相對路徑(`__file__` 推導)、原子寫入、統一 provenance schema;移除 `exec(open("/tmp/..."))` 與 import-time 副作用 | `tests/test_probes_m3_schema.py` **46 個測試,本報告當日重跑全綠**;重發後整數欄位與重發前**完全相同**,浮點欄位差在 float64 ULP(已確認是既有 CUDA `index_add_` 非決定性,非回歸) |
| **Phase B** ft-reweight | `fe5b474` | 見 4.2 | **鎖 2**:`--ft-reweight off` 與不給旗標逐位元相同(`tests/test_ft_reweight.py:152`);`on` 時 `obj_version` invariant 零違反(`:172`);R0 對 M2-best 存檔逐位元相同 |
| **T0-c** P0b | `58c3b7e` + `da818f4` | 預註冊實驗本體,1,090 格 + 372 隨機帶 + 2 failed | JSON 帶 `preregistration.doc_sha256`;`failed_cells` 不得靜默略過 |

### 5.5 記憶體數字的處置(M4 B1)

M4 設計草案 §1.4 B1 判定:**M2/M3 早期經 `run_ablation_m2.py` 產生的 `peak_mem_mb` 一律作廢**(同 process 內多臂污染,且 reset 本身不充分)。本報告因此:

- **不引用**任何 `results/m2/**` 的 `peak_mem_mb`;
- Phase B 四臂的 `peak_mem_mb = 601.146 MB` **可用**——四臂各自獨立 process(檔案時間戳 03:09/03:11/03:13/03:15、`runtime_s` 各約 88 s),且 driver 已帶 `peak_mem_mb_reset_semantics: true`(`ioplace/drivers/run_placement_io.py:81`);
- T1 的 evaluator 峰值 −21.3% 是 `measure_real_case_eval_time.py` 路徑(單次 process、進入前 reset)的量測,不受 B1 影響,但**無 JSON artifact**,列入未決。

---

## 6. Exit 判定

### 6.1 對 phase-1 spec §9 M3 列(line 138)

> 「M3 可微 FT + 容量項 | S4+S7;完整 objective | **FT 顯著下降且 WL/IO 無明顯劣化;完整 ablation 表**」

| # | 條件 | 判定 | 依據 |
|---|---|---|---|
| 1 | **FT 顯著下降且 WL/IO 無明顯劣化** | **FAIL** | 沒有任何 M3 臂讓 FT 下降。縮編後最好的臂 R3 `ft_mst = 3,509`,相對 flat 的 5-seed mean 2,538.0(σ 20.98)是 **+46σ**;相對 M2-best(R0)是 **−0.25σ**,在噪聲內。WL/IO 確實沒有明顯劣化(`Δio_mst ≤ +163`、`Δhpwl ≤ +0.07%`),但那是因為**什麼都沒發生**,不構成通過。 |
| 2 | **完整 ablation 表** | **PASS(縮編形式)** | 交付的矩陣是 P0b 的 **1,090 格**(`merged_l1` 主矩陣 480 + `io_anchored_l1` 480 + `full_objective` 96 + S4e tie-break 子矩陣 20 + bigblue4 子矩陣 14,另有 2 個 failed cell 使 bigblue4 子矩陣為 16)+ Phase B 的 **4 臂**(R0–R3,各有 JSON + npz + `summary.json`)+ 9 支密封探針(schema 測試 46 項全綠)。**原計畫的 S4 臂 C2/C3/C4/C5/C9 一律 void,理由逐條列於 6.3。** |

**⇒ M3 exit:第一條 FAIL ⇒ 里程碑 FAIL。** 這與 M0/M1/M2 的慣例一致:負面結果照實登記,不改判準、不換尺、不挑臂。

### 6.2 對 v3.1 §6.2 的 E1–E7(adaptec1 k16 grid,det=1)

判定統計原設計為 confirmation seeds {1001,1002,1003} 的 paired 單側 95% 信賴界;**Phase B 縮編後只跑了 screening seed 1000,confirmation 未執行**——這是 E2/E3/E4 記 N/A 的主因,逐條標明。

| # | 類型 | 條件 | 判定 | 依據與實測值 |
|---|---|---|---|---|
| **E1** | 必要(主) | `ft_mst(M3) − ft_mst(flat)` 上信賴界 < 0 | **FAIL** | 無 M3 objective 臂;縮編臂 `ft_mst` = 3,509 / 3,532 / 3,551 對 flat mean 2,538.0 ⇒ 點估計就是 +38.3%(+46σ),不需 CI 即可判 FAIL |
| **E2** | 必要 | `io_mst(M3) − io_mst(M2best)` 上信賴界 ≤ 492 | **N/A** | 無合格 M3 臂,且 confirmation 未跑。**參考**:縮編臂 screening 點估計 `Δio_mst` = +101 / +12 / +163,皆 ≤ 492 |
| **E3** | 必要 | `hpwl(M3) − hpwl(M2best)` 上信賴界 ≤ 0(真支配) | **N/A** | 同上。**參考**:screening 點估計 `Δhpwl` = +0.070% / +0.040% / +0.008%,**全為正** ⇒ 即使硬把縮編臂當 M3 臂,也不滿足「真支配」 |
| **E4** | 必要(一致性) | `ft_rg(M3) ≤ ft_rg(flat)` 且與 `ft_mst` 同號 | **N/A** | 同上。**參考**:縮編臂 `ft_rg` = 3,388–3,414 vs flat 2,454 ⇒ 亦不滿足;兩尺同號(都變差)這一半成立 |
| **E5** | stretch | `ft_mst` 3-seed mean ≤ 2,157 | **FAIL** | 最佳 3,509 |
| **E6** | stretch | `ft_mst` 3-seed mean ≤ M1 的 2,430 | **FAIL** | 最佳 3,509;**M1 的 2,430 至今仍是 adaptec1 k16 grid 上最好的 FT 數字** |
| **E7** | 必要 | ablation 每格有 JSON 且通過 `tests/test_probes_m3_schema.py` | **PASS** | 46 個測試當日重跑全綠;P0b JSON 含 `preregistration.doc_sha256` 與 `failed_cells`;Phase B 四臂各有 JSON + npz + summary。（schema 測試的涵蓋範圍是 9 支探針 JSON;driver 產出的臂 JSON 由 `run_placement_io.RESULT_FIELDS` 與 `tests/test_ft_reweight.py:30` 把關。） |
| — | — | **bigblue4 k16 的 E1/E2/E3 對應條款**(§6.2 line 601) | **VOID** | 前置的 T5(`ρ*` 校準 + bigblue4 `σ_seed`)在 Phase B 縮編中未執行;§6.4 明訂「所有 bigblue4 的 M3 臂都必須跑在 `ρ*` 上」,故不得用 `ρ=0.40` 的舊臂替代 |

### 6.3 §6.3 ablation 矩陣的逐臂處置

| 臂 | 原設定 | 狀態 | 理由 |
|---|---|---|---|
| C0 | flat | **DONE** | `results/m2/noise/flat_seed1000`(+ 4 顆 seed 的噪聲底) |
| C1 | M2 best(`f_ft_max=0`,legacy) | **DONE(等價形式)** | R0 與 M2-best 存檔逐位元相同 |
| C1b | `f_ft_max=0`,atomic callback | **VOID** | T4 driver 未實作 atomic 順序(Phase B 縮編);T3 的七步交易已交付但未接上 driver |
| **C2** | `f_ft ∈ {0.1,0.25,0.5,1.0}` 主掃描 | **VOID** | 無候選解鎖 T2 ⇒ 無 `f_ft` 可掃 |
| C3 | FT-only(`L_IO` 關) | **VOID** | 同上 |
| C4 | P0b 次佳候選 | **VOID** | P0b 7/7 淘汰,`summary.ranking` 為空,無排名依據 |
| C5 | `S4g-bboxcov` | **VOID** | P0b 淘汰,且機制上已證實是弱 WL 力(`cos ≈ −0.001`、`κ ≈ 3.8e−4`、T1a 0/30),依 §3.3.4 line 247 移除 |
| C6 | `home` 重算頻率 {50,200} | **VOID** | 縮編後無任何組件使用 `home_e`(ft-reweight 只用 `ft_rg`);L5 仍開放 |
| C7 | k8 / k32 / slicing 泛化 | **未執行** | 縮編;但 P0b 已在 k32 grid 與 k16 slicing 的 4 個最終 placement × 2 溫度上取得離線證據 |
| C8 | bigblue4 `ρ*` | **未執行** | T5 未跑(見 7.5 coverage 缺口) |
| C9 | ramp window vs constant | **VOID** | 無 FT 項可 ramp;L14 仍開放 |
| **新增** | P0b 選型矩陣 | **DONE** | 1,090 格 + 372 隨機帶 + 2 failed |
| **新增** | Phase B ft-reweight R0–R3 | **DONE** | 見第 4 節 |

### 6.4 §7 的 G1–G9 gate 結算

| gate | 狀態 | 說明 |
|---|---|---|
| G1 FT 力量異常 | **N/A** | 無 FT 項 |
| G2 alignment 診斷 | **N/A(記錄替代值)** | 無 `L_FT`;IO 側的 per-net Spearman 隨 `α_ft` 單調下降 0.5160 → 0.5029 |
| G3′ in-loop 無效 | **N/A(其 fallback 已實現)** | 無 `f_ft` 臂;G3′ 的觸發動作(「誠實記錄 negative result,M3 退回只報 FT、不優化 FT」)正是本報告 |
| G4 RG 被鑽漏洞 | **未觸發** | 需 `ft_rg` 降 ≥10% 且 `ft_mst` 升 >3%;實測 `ft_rg` 微升 |
| G5 IO/FT 打架 | **N/A** | 無 `f_ft` |
| G6 `home` 抖動 | **未量測** | P2 `probe_home_churn` 未交付(見 8. N3) |
| G7 規模契約 | **N/A** | M3 未新增 op;10M spike 契約沿用 M2(`tests/test_spike_10m.py`) |
| G8 穩定性 | **通過** | 四臂 `backtrack_median = 1.0`、`Δhpwl` vs R0 ≤ +0.07%、無 divergence |
| G9 鏈轉星 | **僅離線量測** | P0b 低 τ 判定家族的淨 chain→star 對 S4a 為 **−152**(負);in-loop 從未量 |

---

## 7. 對 M4 / Stage2 / Phase 3 的移交

### 7.1 RG 欄位已成標配(**這是 M3 最耐久的產出**)

`io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home` 與拓撲五分類已進入 `evaluator_ref` 與 `evaluator_gpu` 的常規輸出,並被下游採用:M4 設計草案 §2.1 的 evaluator 記憶體階梯表已含「M3 T1 後」一欄;M4 T2 的 streaming evaluator(commit `d1c57ee`)建立在 T1 的成對 dtype 降級之上。**任何後續里程碑報 FT 數字時,雙尺(`ft_mst` 主 / `ft_rg` 次)+ `mst_excess` 應為預設欄位**;T11 已證明 grid 上 `ft_rg` 不需降級為拓撲下界。

### 7.2 S7 重開條件(存檔,M4)

重開 gate(§4.3):任一臂的**絕對 max demand-per-length** 相對同 case flat 上升超過 3σ 帶,或 LEF/DEF case 上出現 `D_ab > ρ·ℓ_ab`。前置條件(§4.4):(a) `D_ab` 與 §2.1 routing 模型一致或誠實改名 star-demand;(b) `C_ab` 固定為 flat 實測值 × ν 並全程凍結;(c) 改真 hinge;(d) `λ_cap` 有雙上限、`g_cap=0` ⇒ `λ_cap=0`、每次變更遞增 `obj_version`;(e) 定義零需求/無相鄰對/圖不連通/零長度邊界的行為。

**但目前這個 gate 不可判定**,原因有二,M4 必須先補:(i) 3σ 帶需要 T5 的 bigblue4 `σ_seed`,**從未量測**;(ii) 設計要求的 `boundary_demand_{total,max,p90,mean,gini}` 欄位**未落地**——`probe_m3_rg.json` 的 `demand_per_len_stats` 目前只有 `n/min/med/max/max_over_med`,`total`/`p90`/`mean`/`Gini` 四個欄位在 repo 內任何 JSON 都不存在(T0-P3 未交付,見 8. N2)。

### 7.3 P0c pilot 規格(post-registration,**未執行**)

裁決書 §D 已把規格寫死,若日後執行**必須先獨立預註冊並 commit,且不得改變 P0b 的 verdict**:

- **唯一候選 `S4a-star`**(其餘六個機制性死亡,無排名依據)。case:adaptec1 k16 grid,`det=1`、`seed=1000`、`ρ_max=0.40`、annealed τ;操作限制**借用**「受限合格」配置(須註明是借用,S4a 並未取得該資格):`--ft-ramp-mode window`、`τ_start=0.12`/`τ_full=0.05`、G9 開。
- **臂**:P0 `--f-ft-max 0 --callback-order atomic`(G3′ 對照)、P1 0.10、P2 0.25(P0b 唯一實測的 `f`)、P3 0.50(out-of-envelope,須標明);P3 相對 P2 單調改善且未破 PG2 時追加 P4 1.0。約 4 臂 × 2 分 GPU。
- **Gate(機械判定)**:**PG1** = 存在 `f>0` 臂使 `ft_mst ≤ ft_mst(P0) − 62.9`;**PG2** = 同臂 `io_mst ≤ io_mst(P0) + 492` 且 `hpwl ≤ hpwl(P0) × 1.005`;**PG3** = G9 淨轉換報告(PG1 過則 inert;PG1 敗且淨轉換為正 ⇒ 「S4a 把鏈擠成星」結案,**不得改切 S4e**——S4e 已被證明恆等於 IO 項);**PG4** = G8 穩定性 + `cancellation_ratio ≥ 0.3`。
- **停止規則**:全部 `f>0` 臂 PG1 失敗 ⇒ M3/S4 線路正式關閉。PG1∧PG2 通過 ⇒ 升級 confirmation(seeds 1001–1003、paired 單側 95%、`t = 2.920`、同 seed M2-best 對照)。
- **執行前的測試鎖**:`κ_ft=0` 逐位元短路、fp64 gradcheck、chunked 與 unchunked < 1e−10、filler 梯度恆 0、`τ→0` 收斂到 S4a 的 hard 極限、`legacy + f_ft_max=0` 與 M2 存檔逐位元相同。

### 7.4 建議的決定性實驗:槓桿 2×2(回答 4.6 的識別性缺口)

M3 的 null result 無法區分「訊號」「槓桿」「IO 項在場」三個因素。**最小可識別設計**是同 case/同 seed 的 2×2 加對照(每格約 2 分鐘,總計 < 20 分鐘 GPU):

| | 訊號 = crossings | 訊號 = `ft_rg` |
|---|---|---|
| **槓桿 = WL `net_weights`** | (已有:M1 的 A1,但**IO 項關**) ⇒ 需補「IO 項開」版 | **關鍵缺格**:IO 項開 + `ft_rg` → WL 拉力 |
| **槓桿 = `io_term.w`** | 已有:`--alpha-io`(M2 已支援) | 已有:R1–R3 |

其中「IO 項開 + `ft_rg` → WL `net_weights`」是**唯一同時具備(a) 真實新增線長拉力、(b) FT 專屬訊號**的組合,也是 M1 那個 −5.1σ 效應最直接的移植。實作成本:`run_placement_io.py` 的 ft-reweight 區塊改寫 `placer.data_collections.net_weights` 而非 `io_term.w`(需注意 M1 用 `α=0.2, cap=10`,不是本次的 `cap=64`)。**這是本報告對 M4/Stage2 的第一順位建議。**

### 7.5 coverage 缺口(誠實清單)

1. **bigblue4 完全沒有 reweight 臂。** Phase B 只在 adaptec1 k16 grid 跑。這是最痛的缺口,因為 **bigblue4 上 M1 reweight 的 FT 效應是 −57.5%(2,948 vs 6,935),比 adaptec1 的 −4.5% 大一個數量級**——「reweight 對 FT 有沒有用」這個問題在大 case 上的答案很可能與 adaptec1 不同,而我們沒測。
2. **bigblue4 的 `ρ*` 從未校準**(T5 未執行)。M2 沿用 adaptec1 的 `ρ=0.40` 造成 `Δhpwl = +5.41%`,超出 iso-WL 帶。任何 bigblue4 的 M3/M4 結論都必須先做這件事。
3. **bigblue4 的 `σ_seed` 未量**(`ft_mst` 與 `ft_rg` 都是)⇒ bigblue4 上任何顯著性宣稱目前都不成立。
4. **confirmation seeds 1001–1003 從未跑過任何 M3 臂**(只有 flat 有)。
5. **`α_ft` 只掃了 3 個點且全在同一個槓桿上**;`every`(reweight 頻率)固定 50,未掃。

---

## 8. 低信心與未決總表

### 8.1 設計 §9 的 L1–L15 結算

| ID | 內容 | 結算 | 依據 |
|---|---|---|---|
| L1 | RG 的幾何樂觀性 | **結案** | T11:中位偏差 0%、p90 0%、負偏差 0 條 ⇒ 不降級 |
| L2 | Λ≥4 上界鬆緊度 | **部分結案,仍開放** | T1 把 Λ≤8 改精確 Dreyfus–Wagner,只剩 Λ>8 用 MST;T11 量到 Λ≥4 尾巴(樣本 2.3%)中位樂觀 0.2–0.25 ⇒ 對總量影響 1–5% |
| L3 | L 形 walk 約定敏感度 | **結案** | P1:`ft_mst` 相對差 0.0010 / 0.0028 / 0.0092 / 0.0173(4/4 全 <2%),`io_mst` 兩方向逐位元相同 4/4 |
| L4 | `β` 窗口 | **結案(moot)** | S4b 家族全滅,無實作 |
| L5 | `home_e` 凍結代價 | **開放(但目前 moot)** | 縮編後無組件使用 `home_e`;P2 探針未交付 |
| L6 | 數值下限 | **結案** | P4:`n_underflow_fp32 = 0`(24 格全部);風險只在 K≥32 或 β<0.25 |
| L7 | region 利用率分母 | **結案** | P6:`l7_triggered = false`,free-area 口徑下 M2 亦改善 |
| L8 | 步長上限形式 | **moot** | 無 FT 項;T3 已交付但未接上 driver |
| L9/L10/L11/L12 | — | **結案 / 作廢** | L11 由 12 個中段 snapshot 解決;L12 由拓撲轉換矩陣取代 |
| **L13** | **P0b 是單步實驗,不蘊含多步收斂** | **開放,且已成永久缺口** | 原本的檢疫機制是 T6 的 in-loop G3′,而 T6 從未執行 ⇒ **「S4a 的 4–13σ 單步 FT 效用在多步下是否存活」在 M3 內沒有答案**,只能由 7.3 的 P0c 回答。**這是 M3 最大的殘餘風險。** |
| L14 | `τ_start/τ_full` 非因果證明 | **開放** | C9 臂 void |
| L15 | `S4e` tie-break 約定 / `(E,A)` 30M 記憶體 | **部分結案** | tie-break 對一個梯度上恆等於 IO 項的候選沒有意義(B2);`(E,A)` 記憶體未驗但已無實作需求 |

### 8.2 本報告新增的未決項

| ID | 內容 | 影響 | 誰解決 |
|---|---|---|---|
| **N1** | 「IO 項啟用後 ft reweight 失效」**未被識別**:M1 與 Phase B 同時差訊號、槓桿、IO 項在場三件事 | 4.6 的標題敘述只能寫成弱版本 | 7.4 的槓桿 2×2(<20 分鐘 GPU) |
| **N2** | T0-P3 的 `boundary_demand_{total,p90,mean,gini}` 從未落地(repo 內任何 JSON 都查無此欄) | **S7 的 M4 重開 gate 目前不可判定** | M4 補欄位 + T5 的 `σ_seed` |
| **N3** | P2 `probe_home_churn` 未交付 | G6 / L5 無資料 | 任何重啟 `home`-based surrogate 的里程碑 |
| **N4** | T1 的 evaluator 峰值 −21.3% / runtime +23% 只在 commit 訊息,無 JSON artifact | 可重現性缺口(數字本身方法正確,不受 M4 B1 影響) | M4 的 evaluator profile |
| **N5** | 探針自帶實作與 T1 evaluator 在 adaptec1 k16 M2-best 上差 1 count(`io_rg` 22,720 vs 22,719) | 設計 §2.2 表未同步為 T1 值 | 引用時以 T1 evaluator 為準(本報告已標注) |
| **N6** | P0b 的隨機帶對 `full_objective` 家族是「同 L1 大小的純隨機位移」帶,而非「兩個 full-objective 步之差」的帶 | T1-P3 的帶寬語意較弱 | 不影響結論(T1-P3 結構性不可通過,B4-1) |
| **N7** | T2d(拓撲轉換 gate)從未裁決 | 協定偏差,verdict 不受影響 | 若 P0c 執行需補 |
| **N8** | bigblue4 全線未測(reweight / `ρ*` / `σ_seed`) | 見 7.5 | M4 |

---

## 附錄:證據索引

| 主張 | 檔案 |
|---|---|
| P0b 全表、verdict、tier-1/2 明細、隨機帶、拓撲轉換 | `results/m3/probes/probe_p0b.json`(+ `.checkpoint.jsonl`) |
| P0(先驗選型)13 候選 × 18 格全滅 | `results/m3/probes/probe_ft_surrogate_soft.json` |
| Phase B 四臂原值、Δ、門檻旗標、R0 逐位元宣告 | `results/m3/reweight/summary.json` + `adaptec1_k16_R{0,1,2,3}_*.json[.npz]` |
| `ft_mst`/`ft_rg` 的 `σ_seed`(5 顆 flat seed 重評) | `results/m3/probes/ft_rg_seed_noise.json` |
| T11 maze 校驗 | `results/m3/probes/probe_maze_sample.json` |
| 雙尺報表、Λ 分布、`n_adj_pairs`、邊界需求 | `results/m3/probes/probe_m3_rg.json`、`probe_m3_bb.json`、`probe_m3_surrogate.json` |
| L 形約定 / β-τ 掃描 / free-area 利用率 | `probe_l_convention.json`、`probe_beta_tau.json`、`probe_free_area_util.json`、`probe_m3_util.json` |
| 12 個 trajectory snapshot 與逐位元宣告 | `results/m3/snapshots/manifest.json` |
| 選型裁決(預註冊內不可再議) | `docs/results/2026-08-14-m3-s4-adjudication.md` |
| 四輪對抗性審查 | `docs/reviews/2026-08-13-m3-{draft-v1-adversarial-opus,draft-v1-adversarial-codex,v2-adversarial-codex,v3-verify-codex}.md` |
| M2 對照數字 | `docs/results/m2-differentiable-io-report.md` |

**一句話收尾:** M3 交出的不是一個可微 FT 項,而是**一份足以否證它的證據**——外加一把可信的雙尺 FT 量尺(T1 + T11)、一組把「看起來有效」擋在門外的預註冊協定,以及兩個誠實的負面結果(7 候選全滅;以 `ft_rg` 改寫 IO 項 per-net 權重的離散 reweight,在 IO 項開啟時無效)。**下一步不是再挑一個 surrogate,而是先把 7.4 的槓桿 2×2 跑掉**——因為目前唯一在 FT 上真正有效過的東西(M1 的 −5.1σ),用的是我們在 M3 裡從來沒有試過的那個槓桿。
