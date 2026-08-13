# M3 設計草案 v3:可微 feed-through(S4)+ 邊界容量項(S7)

- 日期:2026-08-13
- 狀態:**v3(2026-08-13,第二輪對抗性審查後修訂)**
  - **Phase A(可立即進 writing-plans / 實作)**:§2 的 routing 模型與 evaluator 擴充(T1)、§4 的 S7 裁決、§5 的 schedule 契約(T3)、§6 的 exit 預註冊、§8 的 T0(探針重發 + **P0b**)。這些章節**與 S4 的形式選擇無關**。
  - **Phase B(封鎖中)**:**S4 的可微形式尚未定案**,§3 只交付「候選集 + P0b 預註冊協定」。T2(S4 op)與 T4–T8 **不得進 writing-plans,直到 P0b 完成且依 §3.4 的判準選出候選**。
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §2(FT 定義)、§5.3(S4)、§5.4(S7)、§6(evaluator)、§9 M3 列
- 繼承資產:`docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md`(v2)的 S1 L1-SDF 軟歸屬、S2 product-form、§2.5 chunked-k 契約、§4.2/§5.2 schedule、§6.2 params-borne patch、§6.4 secant refresh、§7.2 噪聲 regime
- 前置結論:`docs/results/m2-differentiable-io-report.md`(M2 exit PASS:−18.5% io @ +1.31% hpwl;但 **ft_count 反升 +38.5%(adaptec1)/+17.3%(bigblue4)**)
- 對抗性審查:第一輪 `docs/reviews/2026-08-13-m3-draft-v1-adversarial-opus.md`(F1–F15)與 `docs/reviews/2026-08-13-m3-draft-v1-adversarial-codex.md`(1–16)→ v2;第二輪 `docs/reviews/2026-08-13-m3-v2-adversarial-codex.md`(1–9,6 BLOCKER + 2 MAJOR + 1 MINOR)→ **本版**
- **證據狀態**:探針在 `ioplace/diagnostics/probes_m3/`、結果在 `results/m3/probes/`。**v2 宣稱「P0/P1/P4/P6 已 hermetic」是錯的**(路徑硬編碼、env 缺 command/hostname/timestamp、無 `exactness` 旗標),v3 撤回該宣稱,六支探針全部進 T0 的重發清單(§8 T0-b)。v2 §3.3 引用的 ad-hoc 下降實驗**降級為「探索性,不作為裁決依據」**,其角色由 P0b 取代。

---

## 0. 摘要(v3 裁決一覽)

**兩個必須先講的事實:**

1. **P0 的誠實結論(不變):** 依第一輪預註冊的量級判準(`ratio_vs_true ∈ [0.7,1.5]` 且 `mass_on_ft0 < 0.20`),**13 個候選 × 18 格裡沒有任何一個全過**;最佳者 `S4b-gated_beta{0.5,1.0}` 各 4/18,合格格全部集中在 `τ_rel = 0.03`;**`τ_rel = 0.30` 全滅**;Opus F2 的 hardbase 修法比 λ-baseline **更差**。來源:`results/m3/probes/probe_ft_surrogate_soft.json` → `summary.per_candidate[*]`。
2. **v2 用來翻案的證據不夠格(本輪新增):** v2 §3.3 以純量效用 `U` 選出 S4a-soft。Codex 指出 `U = 17.4` 實際是 `87/5`——**分母只有 5 條 net(base `io_rg` 的 0.022%)**,比值不穩定、沒有定義可接受象限、噪聲底來自**單一**隨機向量、步長用 RMS 正規化而 scheduler 用 L1、且缺 WL/density 力、缺 Nesterov 狀態、缺主判準 `ft_mst`。**`U` 廢除。**

**結論:證據不足以在候選之間定案。v3 不選 winner。** 取而代之交付一份**預註冊的 P0b 協定**(§3.4):Pareto 向量判準 + 信賴帶 + 真實 trajectory 中段 snapshot + L1 正規化步 + 拓撲轉換 gate,並把候選集擴充到含**重疊感知(overlap-aware)的邊union 形式 S4e**——後者在 Codex F3 的 worked example(鏈 `h—a—b` vs 星)上**兩者給出相同的值**,結構性地消除 S4a 的「鏈轉星」偏誤。

| # | 問題 | v3 裁決 | Phase | 信心 |
|---|---|---|---|---|
| Q1 | routing 模型與 FT 定義 | **定案(維持)**:region-adjacency Steiner 為參考 routing 模型;evaluator 只新增欄位並**分離 exact / upper-bound**;**`ft_mst` 是 exit 主判準,`ft_rg` 是次要一致性欄位**,幾何樂觀性由 T11 在報告前無條件補上 | **A** | 高 |
| Q2 | **S4 可微形式** | **未定案,延後至 P0b。** §3.3 給 6 個候選(含新增的 overlap-aware `S4e-union`),§3.4 給預註冊的兩層判準與完整實驗協定。**在 P0b 產出並判定之前,任何 S4 形式都不得寫進實作計畫** | **B** | — |
| Q3 | S7 容量項 | **定案:不進 objective,降為診斷欄位。** free-area 與 total-area 兩種分母下面積利用率都沒被 M2 惡化(P6 `l7_triggered=false`);邊界需求的絕對峰值在 5/5 組態下降 8.3–20.6%(v1 的 +42% 是 median 假象);且 v1 的 `L_cap` 有四個獨立結構缺陷。M4 條件重開 | **A** | 高 |
| Q4 | objective 排程 | **契約定案,參數待 P0b。** 主旋鈕 = 力量佔比 `f_ft`;`κ_ft` 每 callback 導出;ramp 改用 **`τ_rel` 的兩點定義(`τ_start=0.12` → `τ_full=0.05`)** 並明確為 **sample-and-hold 階梯**;callback 契約改**原子化**(量測 → κ → 合併範數/EMA → `Cmax`/`λ_io` → 版本一次遞增 → refresh),順帶修掉 M2 driver 遺留的「refresh 看到舊係數」缺陷 | **A** | 中高 |
| Q5 | 實驗與 exit | **定案**:screening seed 1000、confirmation **獨立** seeds {1001,1002,1003};判定用 **paired difference 的單側 95% 信賴界**(vs 同 seed 的 flat / M2-best 對照);E1 主判準用未動過的 `ft_mst` | **A** | 中高 |
| Q6 | 風險/gate | 9 個 gate(G1–G9);新增 **G9「鏈轉星」**(拓撲類別轉換)因為 per-Λ `ft_rg` 表**偵測不到**該偏誤 | **A** | 中 |
| Q7 | task 分解 | **Phase A**:T0(探針重發 + snapshot 工具 + **P0b**)、T1(evaluator RG)、T3(schedules);**Phase B**(P0b 後):T2(S4 op)、T4(driver)、T5–T8(實驗與報告);T11(maze)無條件、可與 Phase A 並行 | — | — |

**一句話結論(v3):** 精確的 FT surrogate 在 GP 的軟 regime 會量級崩壞(P0:最壞 295×),修量級的 membership gate 會殺掉 11–81% 的洩漏梯度(§3.2);v2 提出的替代(home-rooted 星形和)方向上有效但**結構上偏好「星」勝過「鏈」**(Codex worked example:兩者的 `ST`/`Λ`/`FT` 完全相同,S4a 卻給 1 vs 0),而 v2 選它用的純量判準分母只有 5 條 net。v3 因此把選擇權交給一個**預註冊、可否證、有信賴帶**的實驗(P0b),並補上一個在該 worked example 上結構正確的候選 `S4e-union`;同時把所有與 surrogate 無關的決定(routing 模型、S7、schedule 契約、exit)全部定案,讓 Phase A 立刻可以開工。

---

## 0.1 v2 → v3 變更摘要(逐條回應第二輪 Codex 審查)

| 審查 finding | 裁決 | v3 的處置 |
|---|---|---|
| **F1** [BLOCKER] `U` 數學上不穩定(17.4 = 87/5)、不定義可接受象限、噪聲底是單一隨機向量、RMS≠L1、缺 WL/density/Nesterov/`ft_mst` | **完全成立** | **廢除純量 `U`。** §3.4 的 P0b 判準改為 **Pareto 向量** `(Δft_mst, Δft_rg, Δio_mst, Δio_rg, Δhpwl, Δutil)` + **由 ≥8 個隨機方向構成的分布**所定義的信賴帶;步長改 **L1 正規化**(與 scheduler 一致)且提供 **merged-L1 / IO-anchored-L1 兩個家族**;3 個 η;**真實 trajectory 中段 snapshot**(取得方式見 §3.4.1);接受判準改為「FT 絕對改善且 IO/HPWL 不出實測噪聲帶」。v2 §3.3 的表降級為探索性,不再是裁決依據 |
| **F2** [BLOCKER] 「絕對值會被除掉」只在 unclamped/uncapped 的瞬時極限成立;`mass_on_ft0` 量的是支撐錯置,直接改變方向 | **成立** | §3.2.1 把宣稱收斂為「**global 乘法尺度**只在 `κ` 未 clamp、步長上限未綁定的**瞬時**極限下抵銷;per-net/bucket 的支撐扭曲不會抵銷」。`mass_on_ft0` 與 bucket 份額**升格為 tier-2 方向安全 gate**(§3.4.4),不再只是「估計量資格」。§5.4/§6.3 新增必記欄位:`f_effective`、`kappa_clamp_active`、`ratio_inst` 與 `ratio_ema`、`applied_force_l1` |
| **F3** [BLOCKER] S4a 的鏈轉星偏誤有 worked example,且 L12 的 per-Λ `ft_rg` 診斷偵測不到 | **成立** | (a) 候選集新增 **`S4e-union`(邊union soft crossing − 基準)**,在該 worked example 上鏈與星**同值**(§3.3.3 逐項驗算);(b) P0b 新增**拓撲分類**(`chain` / `star` / `true-FT` / `trivial`)與**類別間轉換矩陣**量測;(c) 新增 **G9 鏈轉星 gate**(§7);(d) L12 改寫為「per-topology-class 轉換」而非 per-Λ |
| **F4** [BLOCKER] callback 契約沒有真正 apply/refresh 新係數;M2 driver 先算 `lambda_io` 再更新 `ratio_ema`,refresh 看到舊係數;`g_ft=0` 時 `κ` clip 到 100 會讓 `Cmax` 憑空懲罰 IO 步 | **成立** | §5.5 改為**原子化 callback 契約**(7 步,版本只遞增一次);明文記載這是**修掉 M2 遺留缺陷**,因此 `f_ft_max=0` 與 M2 不再逐位元相同 ⇒ 用 `--callback-order legacy` 保留逐位元回歸鎖(§5.5 註 2);零梯度分支改 **`g_ft ≤ ε_rel·g_io`(`ε_rel = 1e-3`)⇒ `κ_ft = 0`、`Cmax = 1`**,絕不產生虛構懲罰 |
| **F5** [BLOCKER] ramp 與其「晚窗口」論述矛盾(在宣稱的 regression 邊界已有 70.5% 力量);C9 的 `of_ft_full = of_on` 除以零;每 50 iter 更新是階梯不是連續 | **成立** | §5.2 的 ramp 改用 **`τ_rel` 的兩點對數插值**(`τ_start = 0.12` 起,`τ_full = 0.05` 滿),依 M2 trajectory 重新校準:it400(τ_rel 0.137)**= 0.000**、it450 = 0.255、it500 = 0.700、it550+ = 1.000;明文寫成 **sample-and-hold(每 50 iter 階梯)**,並說明階梯是 line-search 常數性的**必要條件**而非缺陷;C9 的全程臂改為顯式 `--ft-ramp-mode constant`(無分母) |
| **F6** [BLOCKER] 兩層判準沒寫進正文、無法否證 S4a;狀態同時說「可進 writing-plans」又說「沒有 P0b 就沒有裁決基礎」 | **成立** | §3.4.3/§3.4.4 把 tier-1 / tier-2 **完整寫進正文**(門檻、適用 regime、不確定度處理、不通過的 fallback 動作);tier-2 具**否決權**(`mass_on_ft0 > 0.50` 於 `f ≥ 0.5·f_max` 的 regime ⇒ 只能進「受限模式」或被拒);頂部狀態改為 **Phase A / Phase B 兩段**,S4 章節明確封鎖 |
| **F7** [MAJOR] screening seed 1000 又出現在 confirmation 集合裡;mean±range 不是 paired confidence bound | **成立** | §6.2 改:screening = seed 1000;**confirmation = 獨立 seeds {1001, 1002, 1003}**;判定改 **paired difference 的單側 95% 信賴界**(n=3,`t_{0.95,2} = 2.920`),對照組為**同 seed** 的 flat 與 M2-best(flat 的 1001–1003 已存在於 `results/m2/noise/`;M2-best 需補跑 3 顆,計入 T6 預算) |
| **F8** [MAJOR] 「已 hermetic」的宣稱與檔案不符 | **成立** | **撤回宣稱**。§8 T0-b 的重發清單明確含 **P0、P1、P4、P6 + 舊四支(rg/bb/surrogate/util)= 7 支**;統一 schema(`command`、`python_executable`、`python_version`、`hostname`、`utc_timestamp`、`repo_commit`、`dp_commit`、`input_sha256`、`exactness`)+ 路徑改 repo 相對(由 `__file__` 推導,可用 `--repo-root` 覆寫)+ 原子寫入 + `tests/test_probes_m3_schema.py` |
| **F9** [MINOR] 「2–30×」「20–40×」誤述 | **成立** | 兩處**刪除**。v2 §3.3 的 ad-hoc 數字改以逐格原值呈現於 §10.2,並標明「探索性、無信賴區間、不得作為裁決依據」 |

## 0.2 v1 → v2 的處置(保留追溯,已結案者不再展開)

Opus F1/F2(S4b 軟 regime 崩壞、負係數)→ P0 已跑並記錄於 §3.2,**形式選擇改由 P0b 決定**;F3/F4 + Codex 9/10/11(S7)→ §4 定案不做;F5/F13 + Codex 3/16(數字誤述)→ 已更正(bucket 份額 **0.01–0.35**、Λ=2 份額 **78.7–97.5%**、grid 兩尺差 **0.8–4.0%**);F6 + Codex 14(exit)→ §6.2 重寫,本版再依 F7 修正 seed 設計;F7 + Codex 13(Lipschitz)→ §5.3 降級為經驗步長上限;F9/F10/F15 + Codex 4/5/6(S4b 的數值陷阱與極限)→ 移入 §3.3 各候選的「數值契約」欄,由 P0b 一併驗;F11/F12 + Codex 1(exact/ub 語意、per-net 一致率)→ §2.2/§2.5 定案;F14(G2 支撐)→ §7 G2;Codex 2(RG 幾何 gap)→ `ft_rg` 降為次要 + T11 無條件;Codex 7(hinge 邊界)→ §8 T2 驗收;Codex 8(callback pass 數)→ §5.4;Codex 12 + F8(正規化語意)→ §5.2 + 本版 F2 的收斂;Codex 15(provenance)→ **本版 F8 撤回完成宣稱並擴大重發清單**。

---

## 1. 名詞與符號

| 符號 | 意義 | 大小/來源 |
|---|---|---|
| `G_R` | region adjacency graph(共邊即相鄰) | 相鄰對數:adaptec1 k16 grid 24、k32 52、k16 slicing 33(`probe_m3_rg.json:n_adj_pairs`) |
| `D[a,b]` | `G_R` 上的 hop 距離(Floyd–Warshall) | (K,K) uint8,靜態 |
| `A` | `G_R` 的邊數(= 相鄰對數) | ≤ 52(實測) |
| `path(h,k)` | 從 h 到 k 的**規範最短路**(tie 規則見 §3.3.3) | 靜態,存成 (K,K) 的 A-bit mask |
| `ecc_max[h]` | `max_k D[h,k]` | (K,) uint8;adaptec1 k16 grid = 6 |
| `Λ_e` / `Λ̄_e` | hard 觸及 region 數 / 其凍結快照 | evaluator |
| `ST_e` | `G_R` 上連接 `touched(e)` 的最小 Steiner 樹邊數 | 新增欄位,**分 exact / ub** |
| `io_rg` / `ft_rg` | `Σ_e ST_e` / `Σ_e (ST_e − (Λ_e−1))` | 新增,**次要欄位** |
| `home_e` | pin 數最多的 region,平手取最小 index | `(E,)` uint8,每 callback 重算後**凍結** |
| `q_{e,k}` / `S_{e,k}` | M2 的 soft 觸及機率 / 其對數(`q = −expm1(S)`,故 `S = log(1−q)`) | `ioplace/ops/io_term.py` |
| `f_ft` | **FT 力量佔比** `= κ_ft‖∇L_FT‖₁ / ‖∇L_IO‖₁` | 主旋鈕,`τ_rel` 兩點 ramp,sample-and-hold |
| `κ_ft` | 由 `f_ft` 導出的實作權重 | 每 callback 更新,clamp `[0, κ_max]` |
| `Cmax` | per-(e,k) 合併係數的上界 | 步長上限用,每 callback 重算 |
| **拓撲類別** | 每條 net 依 `(Λ, ST, FT)` 分成 `trivial`(Λ≤1)、`adjacent`(FT=0 且 Λ=2)、`chain`(FT=0、Λ≥3、`ST=Λ−1`)、`star`(FT=0、Λ≥3、與 chain 的分支結構不同,見 §3.4.5)、`true-FT`(FT≥1) | P0b / trajectory 診斷 |

**硬性分工(延續 M2 §1):** evaluator 既有欄位語意與數值 M3 不得改動;新欄位一律新增,由「舊欄位逐位元不變」測試把關。**所有 FT 的數值宣稱回 evaluator(`ft_mst` 主、`ft_rg` 次);`L_FT` 是 optimizer-facing 的方向,不是 FT 的估計量**(但依 F2,它的**支撐分布**仍受 tier-2 gate 約束)。

---

## 2. Q1 — routing 模型與 FT 定義(**Phase A,定案**)

### 2.1 定案

```
route(e)     := G_R 上連接 touched(e) 的最小 Steiner tree T_e(邊權 1 = 一次 boundary crossing)
crossings(e) := |E(T_e)| = ST_e
FT(e)        := |V(T_e) \ touched(e)| = (ST_e + 1) − Λ_e        # 對一棵**實際的樹**成立
io_rg = Σ_e ST_e = hard_lambda_sum + ft_rg;   io_mst = io_rg + mst_excess
```

### 2.2 證據(`results/m3/probes/probe_m3_rg.json` / `probe_m3_bb.json`)

| run | `io_mst` | `hard_λ_sum` | `io_rg` | `ft_rg` | `ft_mst` | `mst_excess` | `detour_mst` |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 k16 grid flat | 30,256 | 24,467 | 26,921 | 2,454 | 2,545 | 3,335 | 5,789 |
| adaptec1 k16 grid **M2 best** | 24,647 | 19,337 | 22,720 | 3,383 | 3,525 | 1,927 | 5,310 |
| adaptec1 k32 grid flat | 47,839 | 34,485 | 42,585 | 8,100 | 8,281 | 5,254 | 13,354 |
| adaptec1 k32 grid M2 | 41,272 | 27,985 | 37,405 | 9,420 | 9,600 | 3,867 | 13,287 |
| adaptec1 k16 slicing flat | 36,615 | 25,951 | 31,461 | 5,510 | 7,389 | 5,154 | 10,664 |
| adaptec1 k16 slicing M2 | 33,460 | 22,202 | 28,600 | 6,398 | 8,812 | 4,860 | 11,258 |
| bigblue4 k16 grid flat | 100,333 | 81,205 | 87,953 | 6,748 | 6,935 | 12,380 | 19,128 |
| bigblue4 k16 grid M1 | 100,860 | 85,580 | 88,438 | 2,858 | 2,948 | 12,422 | 15,280 |
| bigblue4 k16 grid **M2** | 63,755 | 54,518 | 62,589 | 8,071 | 8,136 | 1,166 | 9,237 |

**語意標註:** `Λ≤3` 精確、`Λ≥4` 是 metric-closure MST 上界 ⇒ (a) 相對 RG 內的精確 Steiner 是**上界**;(b) 相對真實可實現幾何,`ft_rg` 是**下界**(L1);(c) `mst_excess = io_mst − io_rg_ub` 只是真實 excess 的**下界**,三分解在 Λ≥4 部分不是恆等式。T1 修掉(§2.5)。

支持裁決的四個事實:

1. **grid 上兩把 FT 尺 per-net 幾乎逐條相同**:adaptec1 k16 全 216,930 條 active net 中 `io_rg > io_mst` **0 條**、`ft_rg > ft_mst` **0 條**、`ft_rg < ft_mst` 只有 141(A2)/91(A0)條且每條差恰好 1;總量差 **0.8–4.0%**。⇒ 換模型不會憑空製造 FT 改善,但在 grid 上兩尺也**幾乎不帶額外資訊**,真正的分歧只在 slicing(`ft_rg/ft_mst` = 0.73–0.75)。
2. **M2 在兩把尺上都讓 FT 變差**(adaptec1 +37.9%/+38.5%;bigblue4 +19.6%/+17.3%)。
3. **恆等式把 detour 拆開**:adaptec1 flat 5,789 = 2,454 真 FT + 3,335 MST 次佳;bigblue4 M2 的 9,237 = 8,071 + 1,166 ⇒ S2 已吃掉 MST 次佳的部分。
4. **成本與 pin 數無關**:Λ 分布極偏(adaptec1 k16:Λ=1 195,140 / Λ=2 20,036 / Λ=3 1,239 / Λ≥4 517)。

### 2.3 被否掉的選項

| 選項 | 否掉理由 |
|---|---|
| 維持 MST 幾何為唯一模型 | (a) 無 `io = (Λ−1) + ft` 分解,S2/S4 無法證明不重疊;(b) **主要理由**:`mst_excess` 佔 detour 13–65% 且隨幾何劇烈漂移(bigblue4 flat 65% → M2 13%),污染跨臂比較;(c) 次要:L 形 walk 是任意約定——**P1 實測影響輕微**(`ft_mst` 相對差 0.10–1.73%,`io_mst` 兩方向逐位元相同,4/4 run),v1 把它寫成強證據是過度主張,已收回 |
| boundary-cost maze 當主 evaluator | 每條 tree edge 跑 Dijkstra,12M–36M net 塞不進 GP 內圈。**保留為抽樣校驗器 → T11(報告前無條件)** |
| 完全取代 MST 欄位 | 破壞 M0/M1/M2 可比性;定案是雙尺並報,**exit 主判準用未動過的 `ft_mst`** |
| 以 λ−1 為唯一 IO 定義 | 與 spec §2/D4 的計數語意衝突 |

### 2.4 低信心

- **L1:RG 的幾何樂觀性未量化** → **T11 maze 校驗,M3 報告前無條件執行**;中位偏差 >20% 則報告中所有 `ft_rg` 敘述降級為「拓撲下界」。
- **L2:Λ≥4 上界鬆緊度未知**(目前佔 `ft_rg` 的 1–5%)→ T1 的 Λ≤8 精確 Dreyfus–Wagner。
- ~~L3(L 形敏感度)~~ **結案**(P1)。

### 2.5 T1 的硬性要求

1. `EvalResult` 分離 `st_exact/ft_exact`(Λ≤8)與 `st_ub/ft_ub`(Λ>8),提供 `n_nets_ub`;聚合值以 **[下界, 上界] 區間**報。
2. Λ≥4 的解必須**展開成 `G_R` 上的實際子樹**,`ST` 取邊數、`ft_rg` 取**非 terminal 頂點數**(與 `evaluator_ref.py:113` 的 distinct 語意同構),使 `FT = ST + 1 − Λ` 每條 net 精確成立。
3. 順帶修 R2:`evaluator_gpu.py:352` 的 `pin_bit_acc` 與 `:413` 的 `passed_bit_acc` 是 `(E,K)` int64(12M×32 各 3.0 GB),改 packed bitmask 或按 net 分塊。**列為 T1 驗收條件。**
4. 新增 `per_net_topology_class`(§1 的五分類),供 P0b 與 G9 使用。

---

## 3. Q2 — S4 可微形式:**候選集 + 預註冊的 P0b 協定(Phase B 封鎖中)**

### 3.1 為什麼延後

三件事合起來使得現有證據不足以定案:

- **F1**:v2 用來翻案的純量 `U` 在主打數字上是 `87/5`——分母 5 條 net,占 base `io_rg` 的 0.022%。任何邊界附近的少數 cell 移動都能讓它變號或發散;而且它沒有定義可接受象限(兩個指標都改善時分母為負)。
- **F3**:S4a 對 Codex 的 worked example(`h—a—b` 鏈,`ST=2, Λ=3, FT=0`)給 **1**,對「兩個相鄰終端的星」(`ST=2, Λ=3, FT=0`)給 **0**——**兩者的 evaluator 指標完全相同,S4a 卻嚴格偏好星**。而 v2 提出的 L12 診斷(per-Λ 的 `ft_rg` 表)在轉換前後看到的都是 `Λ=3, FT=0`,**完全偵測不到**。
- **F6**:兩層判準沒寫進正文,tier-2 被定義成「只影響能不能當估計量」⇒ 永遠無法否決一個方向。

因此:**v3 不選 winner**,改交付可否證的實驗協定。

### 3.2 已知事實(P0,不變)

| 候選 | 合格格數(/18) | 最壞總量比 | 最壞 `mass_on_ft0` |
|---|---:|---:|---:|
| S4a-soft | 2 | 33.47 | 0.961 |
| S4b-current β0.25 / 0.5 / 1.0 | 0 / 0 / 0 | 207.6 / 79.5 / 31.3 | 0.991 / 0.987 / 0.980 |
| S4b-hardbase β0.25 / 0.5 / 1.0 | 0 / 0 / 0 | 294.5 / 159.6 / 102.3 | 0.992 / 0.991 / 0.991 |
| S4b-gated β0.25 / 0.5 / 1.0 | 0 / **4** / **4** | 3.85 / 1.71 / 0.354 | 0.808 / 0.594 / 0.380 |
| S4b-gated-hardbase β0.25 / 0.5 / 1.0 | 0 / 0 / 0 | 34.1 / 25.9 / 18.6 | 0.980 / 0.976 / 0.975 |

兩個機制性結論(P0 + ad-hoc 探索,§10.2):

1. **`R_e = M_e/N_e` 對 `q` 是 0 次齊次**,而 `λ_e − 1` 是 1 次齊次 ⇒ Λ=1 的 net 只要對 D=2 的 region 漏 `q ≈ 0.02` 就被課約 1.02 的 FT ⇒ 83% 的質量落在真 FT = 0 的 net 上。
2. **修正量級的 membership gate 會殺掉梯度**:`∂q̃/∂q = 0` 當 `q < 0.5`,而遠區(`D ≥ 2`)的 `q` 質量落在 gate 以下的比例是 **78.7 / 60.4 / 22.4%**(A0 k16 @ τ_rel 0.30/0.10/0.03)與 **80.8 / 57.3 / 10.8%**(A2 k16)。τ=0.30 時 gate 以下有 757,841 個 (e,k) 對、以上只有 19,296 個。

#### 3.2.1 「絕對值會被除掉」的正確版本(F2)

設 `I = ∇L_IO`、`F = ∇L_FT`、`W = ∇WL`,實際施加的力是
`κ = clip(f·‖I‖₁/max(‖F‖₁, ε), 0, κ_max)`、`C = I + κF`、`G = min(ρ·a·R_EMA, c_lip τ²/(γ C_max))·C`。

因此:

- `‖λκF‖₁/‖λI‖₁ = f` **只在 clamp 未啟動時**成立;
- `‖G‖₁ = ρ·a·‖W‖₁` **只在瞬時 ratio 且步長上限未綁定時**成立(用 `ratio_ema` 與 §5.3 的上限就不成立);
- **把整個 surrogate 乘上一個正純量,只在 `κ` 未 clamp 時抵銷**;
- **per-net / per-bucket 的支撐扭曲不會抵銷。`mass_on_ft0` 量的是支撐錯置,它直接改變方向。**

實測佐證:在主 placement(adaptec1 A2 k16 grid)上,S4a 仍把 **68.5%**(τ_rel=0.10)與 **29.7%**(τ_rel=0.03)的值放在真 FT = 0 的 net 上(`probe_ft_surrogate_soft.json` 的對應 cell)。⇒ **tier-2 保留否決權**(§3.4.4)。

### 3.3 P0b 候選集(6 個 + 對照)

所有候選共用 M2 的 `p_{i,k}`(S1)與 `q_{e,k} = −expm1(S_{e,k})`(S2);`w_e ≡ 1` 於 P0b;`home_e` 為 pin-count argmax(平手取最小 index)。

#### 3.3.1 `S4a-star`(v2 的暫定形式,現降為候選之一)

```
L = Σ_e Σ_k q_{e,k} · (D[home_e,k] − 1)_+
∂L/∂q_{e,k} = (D[home_e,k] − 1)_+          # 常數係數
```
**優點**:零新增常駐狀態;無 `β`、無除法、無 hinge;`τ→0` 極限唯一且解析。
**已知缺陷**:F3 的鏈轉星偏誤;`mass_on_ft0` 高。
**數值契約**:無(係數是靜態整數)。

#### 3.3.2 `S4b-gated-detach β∈{0.5,1.0}` 與 `S4b-gated β0.5`(P0 的 tier-2 最佳者)

```
q̃ = (q − 0.5)_+ / 0.5 ;  g_k = exp((D[home,k] − ecc_max[home])/β)
N = Σ_k q̃_k g_k ;  M = Σ_k q̃_k g_k D_k ;  R = M/N if N>0 else 0
L = ReLU( R − base ) ,  base = (λ−1)  或  (λ−1).detach()
```
**數值契約(第一輪 F10/Codex 5/6 的修法,P0b 必須照做)**:`ε = 0` + `where(N>0, M/N, 0)`;`exp` 一律 fp64;**禁止** `a_k/q_k` 的寫法(0/0),用 `g_k(D_k−R)/N`;`β ≥ 0.25`(P0 顯示 0.25 全滅,列入僅為對照)。
**已知缺陷**:§3.2 的 0 次齊次 + gate 梯度盲區。

#### 3.3.3 `S4e-union-detach` / `S4e-union-hardbar`(**新增,overlap-aware**)

**動機(直接回應 F3)**:星形和的病灶是「每個 terminal 各自從 home 拉一條路,共用前綴被重複計費」。改成**對邊取聯集**即可根治,而且在 product-form 下**與 S2 同構**:

```
靜態:  path(h,k) ⊆ E(G_R) 的 A-bit mask(規範最短路,tie 規則見下)
per-net 累加(與 S2 的 FWD-2 同一趟 k-chunk):
        S^edge_{e,a} = Σ_{k : a ∈ path(home_e,k)} S_{e,k}          # 注意 S_{e,k} = log(1 − q_{e,k}),已在手
        U_{e,a}      = 1 − exp(S^edge_{e,a})                        # 「net e 用到邊 a」的 soft-OR
L_cross_e = Σ_a U_{e,a}                                             # 聯集樹的 soft 邊數
L_FT_e    = ReLU( L_cross_e − base_e ) ,  base_e = (λ_e − 1).detach()  或  (Λ̄_e − 1)
```

**worked example 驗算(Codex F3 的兩個案例):**

| 拓撲 | touched | `path` 聯集 | `L_cross` | `base` | **`L_FT`** | 真 FT |
|---|---|---|---:|---:|---:|---:|
| 鏈 `h—a—b` | {h,a,b} | {e_ha} ∪ {e_ha,e_ab} = {e_ha,e_ab} | 2 | 2 | **0** | 0 |
| 星(a,c 皆鄰 h) | {h,a,c} | {e_ha} ∪ {e_hc} | 2 | 2 | **0** | 0 |
| 真 FT(h 與 b,D=2,a 未觸及) | {h,b} | {e_ha,e_ab} | 2 | 1 | **1** | 1 |

⇒ **鏈與星同值,不再偏好星**;且在「聯集樹 = Steiner 樹」的 net 上 `L_FT` 在 hard 極限下**精確等於 `ft_rg`**。

**梯度**:`∂U_{e,a}/∂S_{e,k} = −exp(S^edge_{e,a})`(當 `a ∈ path(home_e,k)`),故
`∂L_cross/∂S_{e,k} = −Σ_{a ∈ path(home_e,k)} exp(S^edge_{e,a})`,再乘 `∂S/∂ℓ` 走 M2 既有的 leave-one-out 路徑。**數值上乾淨**:不需要 `log(1−q)`(它就是 `S` 本身),沒有 0/0。

**tie 規則(必須預註冊)**:`path(h,k)` 取 BFS 樹上「region-id 序列字典序最小」的最短路;P0b 另跑一個 **reverse tie-break** 子臂量敏感度(比照 P1 對 L 形約定做的事)。

**已知工程風險(必須寫進決策規則)**:需要 `(E,A)` 累加器(A ≤ 52)。10M-cell spike 下 E≈12M ⇒ fp64 24 邊 = **2.3 GB**、fp32 = 1.15 GB;K=32 的 52 邊 fp32 = 2.5 GB。**決策規則**:若 S4e 在 P0b 勝出,T2 必須先通過 10M spike(G7,8 GB 上限),否則依序退到 (a) fp32 累加 + 數值等價測試、(b) 邊分塊(a-chunk × k-chunk 雙層迴圈,forward 成本 ×A/c_a)、(c) 只在 K ≤ 16 支援 S4e、大 K 退回次佳候選。**P0b 只需離線可算(216k net × 52 邊為小量),不受此限。**

#### 3.3.4 `S4g-bboxcov`(spec §5.3 原案)

`L = Σ_e Σ_k cov_k(e)·(1 − q_{e,k})`,`cov_k(e)` = net soft bbox 對 region k 的覆蓋率(需第三個溫度 `τ_b`)。spec §8 明文要求做這個比較,故列入 P0b,**若在此淘汰則 C5 臂可從 §6.3 移除**。

#### 3.3.5 對照組

- `IO-only`(κ=0):現任方向(M2)。
- `RANDOM ×8`:8 個獨立高斯方向(seed 0–7),同步長 ⇒ 每個指標的噪聲帶。

### 3.4 **P0b 預註冊協定**(fast-worker 可照做)

> 本節在 P0b 執行前不得修改。任何修改都必須在 git 歷史中先於執行 commit,並在報告中揭露。

#### 3.4.1 輸入:狀態集合(16 個)

**(a) 真實 trajectory 中段 snapshot(12 個,主力)**
先由 **T0-a** 為 `ioplace/drivers/run_placement_io.py` 加上 `--snapshot-iters "300,400,450,500,550,600"` 與 `--snapshot-dir`,在指定 iteration 於 callback 內把 `node_x/node_y/iteration/overflow/tau/lambda_io` 存成 `results/m3/snapshots/<tag>_it<NNNN>.npz`;然後 `det=1, seed=1000` 重跑兩個 adaptec1 k16 grid 臂:

| 臂 | 參數 | 對應既有結果(必須逐位元復現最終 npz,否則 P0b 中止) |
|---|---|---|
| `flat` | observer mode,`ρ_max=0` | `results/m2/noise/flat_det1_rep0.json[.npz]` |
| `m2best` | `ρ_max=0.40`,annealed τ | `results/m2/sweep/adaptec1_k16_rho0.40_annealed.json[.npz]` |

每個 snapshot 的 `τ_rel` 由其記錄的 `overflow` 經 `tau_rel_from_overflow` 導出(`schedules.py:7`),涵蓋 **τ_rel ≈ 0.226 / 0.137 / 0.096 / 0.065 / 0.042 / 0.030**——**這解決了 v2「只在收斂後的最終 placement 上量」的承重牆(舊 L11)**。

**(b) 最終 placement(4 個,泛化)**
`adaptec1_{A0,A2}_k32_grid`、`adaptec1_{A0,A2}_k16_slicing`(`results/m2/ablation/*.npz`),各以其最終 overflow 導出的 `τ_rel` 計算,另外**強制加測 `τ_rel = 0.30`** 一格以取得 all-τ 證據(F6 指出 v2 的 ad-hoc 探測缺這一塊)。

**(c) 規模抽查**:`bigblue4_{A0,A2}_k16_grid` 最終 placement,只跑 `η = 0.01`、merged-L1 家族。

#### 3.4.2 步長家族與 η(取代 v2 的 RMS 步)

對候選 `c`,合併梯度 `C = I + κ F`,`κ = f·‖I‖₁/‖F‖₁`(P0b 固定 `f = 0.25`,並對勝出者補掃 `f ∈ {0.1, 0.5, 1.0}`)。位移:

| 家族 | 定義 | 對應的 scheduler 語意 |
|---|---|---|
| **M1(主)merged-L1** | `Δpos = −η·L_R·C / (‖C‖₁/(2·num_movable))` ⇒ **平均每座標位移 = η·L_R** | §5.2 的 merged 正規化(總力固定) |
| **M2(次)IO-anchored-L1** | `Δpos = −α·C`,`α` 由 `‖α·I‖₁ = η·L_R·2·num_movable` 決定(**與候選無關的 α**) | 若改用 IO 錨定正規化(敏感度分析) |

`η ∈ {0.005, 0.01, 0.02}`。**兩個家族都用 L1 正規化**(修正 F1 指出的「probe 用 RMS、scheduler 用 L1」錯配)。只移動 `[:num_movable]`;位移後 clamp 進 die。

#### 3.4.3 tier-1(方向安全,**必要**):Pareto 向量 + 信賴帶

每格(state × candidate × η × family)輸出向量
`Δ = (Δft_mst, Δft_rg, Δio_mst, Δio_rg, Δhpwl, Δutil_max_over_mean)`,全部以**計數/絕對值**與**百分比**兩種形式記錄。`ft_mst`/`io_mst` 由 **evaluator_gpu** 算(主判準必須是未動過的尺);`ft_rg`/`io_rg` 由 §2.1 的 RG 路徑算;`util` 用 P6 的 free-area 口徑。

由 `RANDOM ×8` 得每個指標的 `sd_rand`(同 state、同 η、同 family)。**候選 c 在某一格通過 tier-1,當且僅當同時滿足:**

| 條件 | 式子 |
|---|---|
| T1a **FT 絕對改善** | `Δft_mst(c) < 0` 且 `Δft_mst(c) ≤ Δft_mst(IO-only) − 3·sd_rand(Δft_mst)` |
| T1b **IO 不退化** | `Δio_mst(c) ≤ Δio_mst(IO-only) + 3·sd_rand(Δio_mst)` |
| T1c **HPWL 不退化** | `Δhpwl(c) ≤ Δhpwl(IO-only) + 3·sd_rand(Δhpwl)` |
| T1d **密度不退化** | `Δutil(c) ≤ Δutil(IO-only) + 3·sd_rand(Δutil)` |
| T1e **雙尺一致** | `sign(Δft_rg(c)) = sign(Δft_mst(c))` 且 `sign(Δio_rg(c)) = sign(Δio_mst(c))` |

**候選 c 通過 tier-1(整體)** ⇔ 在 `τ_rel ≤ 0.10` 的格中通過率 **≥ 80%**(家族 M1、全部 η),**且**在 `τ_rel ≥ 0.20` 的格中 **沒有任何一格違反 T1b/T1c/T1d**(高溫區可以無效,但不得有害)。

**不確定度處理**:`sd_rand` 由 8 個方向的樣本標準差計算;若某指標的 8 個隨機方向全同號且 `sd_rand = 0`(離散計數可能發生),以 `sd_rand := max(sd_rand, 1 count)` 保底並在 JSON 標記 `sd_floor_applied`。

**fallback 動作**:沒有候選通過 tier-1 ⇒ **M3 不做可微 FT 項**,改走「M1 式離散 reweight + FT 診斷」的縮減範圍,並在報告誠實記錄 negative result(沿用 M1/M2 慣例)。

#### 3.4.4 tier-2(支撐安全,**有否決權**)

依 F2,支撐錯置**不會**被正規化抵銷,故 tier-2 是 gate 不是註腳。門檻**預先登記**:

| 條件 | 門檻 | 適用 regime | 理由 |
|---|---|---|---|
| T2a `mass_on_ft0` | **≤ 0.50** | 力量達 `f ≥ 0.5·f_max` 的 regime,由 §5.2 的 ramp 解出 = **`τ_rel ≤ 0.0775`** | 超過一半的懲罰落在真 FT = 0 的 net 上時,該項的多數不是在做 FT |
| T2b 主質量桶保真 | Λ=2 桶的 surrogate 份額比 ∈ **[0.5, 2.0]** | 同上 | Λ=2 承載 **78.7–97.5%** 的真 `ft_rg` |
| T2c 稀疏桶不得暴衝 | 任一桶份額比 **≤ 5.0**(非對稱:低估到 0.2 只記錄不否決) | 同上 | 高估稀疏桶會把力氣導離 FT 質量;低估只是弱化 |
| T2d 拓撲轉換 | `chain → star` 的淨轉換數 ≤ `3·sd_rand(該轉換數)` | τ_rel ≤ 0.10 | **F3 的直接量測**(§3.4.5) |

**判定與 fallback:**
- tier-1 通過 **且** tier-2 全過 ⇒ **合格**,可直接進 T2 實作。
- tier-1 通過 **但** tier-2 失敗 ⇒ **受限合格**:只能用 `f_ft_max ≤ 0.25`、`ft-ramp-mode = window`(§5.2),且 in-loop 必須開 **G9**;報告必須列出失敗的 tier-2 條目。
- tier-1 失敗 ⇒ **淘汰**,不論 tier-2。
- 多個候選合格 ⇒ 依序比較:(1) `τ_rel ≤ 0.10` 的 `Δft_mst` 中位數;(2) tier-2 全過者優先;(3) 工程成本(常駐記憶體、chunk pass 數)。**比較順序預先登記,不得事後調換。**

#### 3.4.5 拓撲分類與轉換矩陣(F3 的可偵測性)

每條 net 依 `(Λ, ST, FT)` 與 `home` 的分支結構分成 5 類:`trivial`(Λ≤1)、`adjacent`(Λ=2, FT=0)、`chain`(FT=0, Λ≥3, 且 `touched` 在 `home` 的**單一鄰居分支**內)、`star`(FT=0, Λ≥3, 跨 ≥2 個分支)、`true-FT`(FT≥1)。P0b 對每格輸出 **5×5 轉換矩陣**(步前類別 → 步後類別)與 per-class 的 `Δhpwl`、`hard_S4x − ft_rg`。

**這是 v2 的 L12 診斷做不到的事**:per-Λ 的 `ft_rg` 表在 chain→star 轉換前後都看到 `Λ=3, FT=0`。

#### 3.4.6 輸出 schema(`results/m3/probes/probe_ft_direction.json`)

```
{ "probe": "probe_ft_direction",
  "env": {<統一 provenance schema,見 §8 T0-b>},
  "preregistration": {"doc": "docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md",
                      "section": "3.4", "doc_sha256": "<執行時的檔案 sha256>"},
  "config": {"states": [...], "candidates": [...], "etas": [...], "families": ["merged_l1","io_anchored_l1"],
             "f": 0.25, "n_random": 8, "tier1": {...門檻...}, "tier2": {...門檻...}},
  "cells": [{"state","tau_rel","candidate","eta","family",
             "d_ft_mst","d_ft_rg","d_io_mst","d_io_rg","d_hpwl","d_util",
             "d_*_pct", "grad_l1_io","grad_l1_ft","kappa","cos_io_ft",
             "mass_on_ft0","bucket_share_ratio":{"1","2","3","4+"},
             "topology_transitions": [[5x5]],
             "tier1_pass":bool, "tier1_detail":{...}, "sd_floor_applied":bool}],
  "random_bands": [{"state","tau_rel","eta","family","metric","mean","sd","n":8}],
  "summary": {"per_candidate": [{"candidate","tier1_pass_rate_lowtau","tier1_violations_hightau",
                                 "tier2_pass","verdict":"合格|受限合格|淘汰"}],
              "ranking": [...按 §3.4.4 的預註冊順序...]} }
```

#### 3.4.7 規模與預算

狀態 16 + bigblue4 2;候選 6 + IO-only = 7;η 3;家族 2;隨機 8 方向。
主格數 = `16 × 7 × 3 × 2 = 672`;隨機帶 = `16 × 8 × 3 × 2 = 768`;bigblue4 = `2 × (7+8) × 1 × 1 = 30`。
每格成本 = 1 次 fwd/bwd(< 0.5 s)+ evaluator(`ft_mst`/`io_mst`)+ RG Steiner + hpwl ≈ 3 s(adaptec1)/ 12 s(bigblue4)。
⇒ **adaptec1 約 1.2 小時,bigblue4 約 6 分鐘,加 snapshot 重跑 2 臂約 5 分鐘,加載入/編譯開銷 ⇒ 預估 2–3 GPU 小時。**

### 3.5 不論選誰都成立的工程契約(Phase A 可先凍結)

1. **`home_e` 每 callback 由 evaluator 重算後凍結**;forward 內即時 argmax 會讓 `L_FT` 在 home 翻轉處不連續,等於在 line search 裡注入跳變(M2 R12 的病症)。
2. **係數形狀**:所有候選的 backward 都可寫成「per-(pin, k-chunk) 的係數張量 `(P', c)` 乘上 M2 既有的 `exp(S_c − ℓ)` leave-one-out 因子」;差別只在係數怎麼算(S4a 靜態;S4b 需 `(E,)` 的 `N/R`;S4e 需 `(E,A)` 的 `S^edge`)。⇒ **T2 的介面對三者相同**,可先寫介面測試。
3. **記憶體決策規則**:任何候選在 T2 都必須先過 10M-cell spike(G7,8 GB);超標時的退路依 §3.3.3 的三段式。
4. **`κ_ft = 0` 必須讓 FT 路徑完全短路**(比照 `lambda_margin == 0.0`,`io_term.py:203`)。
5. **hinge 邊界**:`IoTermRef` 改用 `torch.relu` 或顯式 strict mask,並加 `λ_e == 1` / `L_cross == base` 的精確邊界等價測試(Codex 7)。

### 3.6 被否掉的替代(不進 P0b)

| 方案 | 否掉理由 |
|---|---|
| soft-Steiner / 可微 MST / `min_h Σ_k q_k D[h,k]` | `min_h` 需 `(E,K)` 累加器(12M×32 fp32 = 1.5 GB @10M、4.6 GB @30M);且第一輪 hard 實測(S4c/S4d)品質最差(總量比 0.07–0.56) |
| 二次型 `q_e^T W q_e` | 每 net `O(K²)` 且需 `(E,K)`;對大 net 無界 |
| S3 span proxy 當 FT 項 | `span` 是 Steiner 的弱下界,`L_FT` 可為負,clamp 後梯度死;S3 維持 M2 §9.3 的 F2 fallback 定位 |
| REINFORCE / 有限差分打 `per_net_ft` | 高變異、需多次 evaluator 呼叫 |
| S4b-hardbase 族 | P0 實測比 λ-baseline 更差(最壞比 102–295 vs 31–208);機制:`Λ̄=1` 的 net 在 hard 基準 0 下 `ReLU(R)=R` 恆正 |

### 3.7 低信心(S4 相關)

- **L4:`β` 窗口** — 只對 S4b 分支有意義。P4 實測 24 格:G1 死區(<0.05)從未觸發,β=2.0 最低 0.152;異常在**上尾**(β=0.25 + 小 τ 到 **2.371**)。P0b 的 β 集合固定 {0.5, 1.0}。
- **L5:`home_e` 凍結 50 iter 的代價** → **P2(T0)**;S4a/S4e 對 home 誤差的敏感度低於 S4b(係數每差一 hop 只變 1,不是軟 max 比值)。
- **L6:數值下限** — 對 S4a 不存在;對 S4b 由 §3.3.2 的數值契約管;對 S4e 是 `(E,A)` 的 dtype(fp32 累加需等價測試)。P4 實測 K=16 下 fp32 underflow 計數 **0/24 格**;原則性風險只在 `K ≥ 32` 或 `β < 0.25`。
- **L13(新):P0b 本身仍是「單步」實驗。** 單步方向良好不蘊含多步收斂良好(路徑相依、Nesterov 動量、與 WL/density 的交互都沒有進去)。緩解:P0b 涵蓋 6 個真實 trajectory 中段點 + 3 個 η + 2 個步長家族;**最終判定仍在 T6 的 in-loop(G3′)**,P0b 只負責「不該進 in-loop 的候選先淘汰」。

---

## 4. Q3 — S7:**不進 objective,降為診斷欄位(Phase A,定案)**

### 4.1 兩個立論前提都被實測推翻

**(a) 面積容量:兩種分母都沒有惡化**(`probe_free_area_util.json`,L7 結案)

| case | 口徑 | flat max/mean | M2-best max/mean | 變化 |
|---|---|---:|---:|---:|
| adaptec1 k16 grid | free area | 1.9271 | 1.8948 | **−1.67%** |
| adaptec1 k16 grid | total area | 2.2081 | 2.1843 | −1.08% |
| bigblue4 k16 grid | free area | 1.8725 | 1.8033 | **−3.70%** |
| bigblue4 k16 grid | total area | 1.9825 | 2.1654 | +9.23% |

`l7_triggered = false`;free-area 利用率最大 0.978,**無 region 超載**;`n_nonpositive_free_area_regions = 0`。⇒ **不加面積懲罰項**,改記錄 `region_util_free[k]`。

**(b) 邊界需求:v1 的「+42% 惡化」是 median 假象**

| run | 相鄰對 | 絕對 max demand/len(flat → M2) | 變化 | median | max/median |
|---|---:|---:|---:|---:|---:|
| adaptec1 k16 grid | 24 | 27.969 → 23.617 | **−15.6%** | 8.211 → 4.891 | 3.41 → 4.83 |
| adaptec1 k32 grid | 52 | 27.969 → 24.578 | **−12.1%** | 8.672 → 5.340 | 3.23 → 4.60 |
| adaptec1 k16 slicing | 33 | 23.358 → 21.412 | **−8.3%** | 9.204 → 7.663 | 2.54 → 2.79 |
| bigblue4 k16 grid | 24 | 70.094 → 55.625 | **−20.6%** | 30.106 → 19.207 | 2.33 → 2.90 |

(來源 `demand_per_len_stats.{max,med}`。Opus F3 另行重算 adaptec1 k16 的 `max/mean` = 2.842 → 2.947、p90 = 18.38 → 17.80;**這兩個量目前只存在於審查文件,T0-P3 必須把 total/max/p90/mean/Gini 全部寫進 JSON**。)

### 4.2 v1 的 `L_cap` 公式本身也不成立

(1) `Q[h,k]·P[h,k,(a,b)]` 是 home-rooted **星形流**,正是 §3.6 否掉的幾何;校驗用的 `boundary_pair_demand` 又是第三套(MST + L 形 walk)。(2) `C_ab = ν·ℓ_ab·(ΣD/Σℓ)` **0 次齊次**,對總需求下降無感,且可用灌冷邊界放寬所有邊界。(3) `softplus((D−C)/C)` **不是 hinge**:`D ≥ 0 ⇒ σ(z) ≥ 0.269`,「零梯度凍結」是死碼;`D` 全零時 `C=0` 除零。(4) `λ_cap` 上限量綱不符;`C_ab` 每 callback 重算破壞 `obj_version` invariant。

### 4.3 定案

**S7 不進 M3 objective(`κ_cap` 不存在)。** M3 只做:

- **診斷欄位**:每 callback 記 `boundary_demand_{total,max,p90,mean,gini}`(per lattice edge 正規化)與 `region_util_free_max_over_mean`。
- **M4 重開 gate(預先登記)**:任一 M3 臂的**絕對 max demand-per-length** 相對同 case flat 上升超過該量的 3σ 帶(σ 由 T5 的 3 顆 flat seed 量出),或 LEF/DEF case 上出現 `D_ab > ρ·ℓ_ab`。

### 4.4 M4 重開時的前置條件(存檔)

(a) `D_ab` 與 §2.1 的 routing 模型一致(從 Steiner 樹邊映到邊界對),或誠實改名 star-demand;(b) `C_ab` 固定為 flat baseline 實測值 × ν,`ν ≥ 該 case flat 的 max/mean`,全程凍結;(c) 懲罰改真 hinge(`ReLU(·)²` 或 `softplus(z/θ)`,θ≤0.1);(d) `λ_cap = min(κ_cap·g_comb/g_cap, ρ_max·g_WL/g_cap)`,`g_cap = 0` 時 `λ_cap = 0`,每次變更遞增 `obj_version`;(e) 定義零總需求 / 無相鄰對 / 圖不連通 / 零長度邊界的行為。

---

## 5. Q4 — objective 的排程整合(**Phase A,契約定案**)

### 5.1 目標函數

```
min  WA-WL + μ·density + λ_io·[ L_IO + κ_ft·L_FT ] + λ_margin·L_margin
                          └──── 同一個 op,共用 backward ────┘
```

### 5.2 權重與排程

| 量 | 規則 | 與 M2 的差異 |
|---|---|---|
| `τ` | `τ_rel(of)` log-linear 0.30 → 0.03(`schedules.py:7-12`) | 不變 |
| `ρ_io` | `ρ_max·clip((of_on−of)/(of_on−of_full))·ramp` | 不變 |
| `ratio_ema` | `‖∇WL‖₁ / ‖∇(L_IO + κ_ft L_FT)‖₁`,每 N=50 iter | **量合併項**。語意 = **iso-total-force**:合併項施加的總梯度範數 ≈ `ρ_io·‖∇WL‖₁`(**僅在 §3.2.1 的條件下**),`f_ft` 只決定 IO/FT 的分配 |
| **`f_ft`** | **以 `τ_rel` 兩點對數插值**:`f_ft = f_ft_max · clip( ln(τ_start/τ_rel) / ln(τ_start/τ_full), 0, 1 )`,**`τ_start = 0.12`、`τ_full = 0.05`** | 新增,主旋鈕。預設 `f_ft_max` 待 P0b(暫定 0.25),掃描 {0.1, 0.25, 0.5, 1.0};`f_ft_max = 0` 關閉整條 FT 路徑 |
| `κ_ft` | 每 callback:若 `g_ft ≤ ε_rel·g_io`(`ε_rel = 1e-3`)則 **`κ_ft = 0`**;否則 `κ_ft = clip(f_ft·g_io/g_ft, 0, κ_max=100)` 並記 `kappa_clamp_active` | 新增,**導出量不是旋鈕** |
| `Cmax` | `1 + κ_ft·(max_h ecc_max[h] − 1)_+`(S4a/S4e 的閉式上界;S4b 需 backward 內順手 reduce)。**`κ_ft = 0` 時 `Cmax = 1`** | 新增 |
| `λ_io` | `min(ρ_io·ramp·ratio_ema, c_lip·τ²/(γ·Cmax))`,**用同一次 callback 更新後的 `ratio_ema` 與 `Cmax`** | 上限公式改;**計算時機改**(§5.5) |
| `home_e` | 每 callback 由 evaluator 重算 | 新增 |
| `κ_cap` / S7 | **不存在** | v1 的第三個權重取消 |
| ramp 模式 | `--ft-ramp-mode {window, constant}`;`window` = 上式;`constant` = 自 `of ≤ of_on` 起 `f_ft ≡ f_ft_max` | **修掉 v2 的 `of_ft_full = of_on` 除零**(Codex F5) |

**校準對照(直接回應 Codex F5 的表):** 依 `results/m2/sweep/adaptec1_k16_rho0.40_annealed.json` 與 `results/m2/noise/flat_det1_rep0.json` 的 trajectory:

| iteration | `τ_rel`(M2 臂) | flat `ft` | M2 `ft` | 差額 | **v2 舊 ramp `f/f_max`** | **v3 新 ramp `f/f_max`** |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.226 | 1,391 | 1,383 | −8 | 0.255 | **0.000** |
| 350 | 0.182 | 1,342 | 1,293 | −49 | 0.451 | **0.000** |
| 400 | 0.137 | 1,872 | 1,826 | −46 | 0.705 | **0.000** |
| 450 | 0.096 | 2,412 | 2,668 | **+256** | 1.000 | **0.255** |
| 500 | 0.065 | 2,461 | 3,135 | **+674** | 1.000 | **0.700** |
| 550 | 0.042 | 2,529 | 3,427 | **+898** | 1.000 | **1.000** |
| 600 | 0.030 | 2,526 | 3,531 | **+1,005** | 1.000 | **1.000** |

⇒ 新 ramp 在**退化尚未出現的三個 checkpoint 上力量精確為 0**,在退化開始的 it450 才給 25.5%,與論述一致(v2 的舊 ramp 在 it400 已有 70.5%,與其自身論述矛盾)。

**必須明講的兩個限定(Codex F5):**
1. **相同 iteration 的快照對照不構成因果證明**——早期的力仍可能造成延遲的拓撲改變。因此 `τ_start/τ_full` 是**可掃描的設計參數**,C9 臂(`window` vs `constant`)就是這個假設的直接檢定。
2. **施加的力是 sample-and-hold 的階梯**:`κ_ft`(以及 `λ_io`)只在 callback(每 50 iter)更新並在其間凍結。**這不是缺陷而是必要條件**——iteration 內係數必須是常數,否則 line search 會看到跳變的 objective(M2 §6.3 坑 3)。`f_ft(τ_rel)` 在 callback 當下取樣。

### 5.3 步長上限(不是 Lipschitz 界)

```
λ_io ≤ c_lip · τ² / (γ · Cmax)
```
**明文聲明**(Codex 13):這界的是 per-(e,k) **係數量級**對梯度 Lipschitz 常數的線性放大,**不是 Hessian 界**;真正的安全機制是 G8 的 backtrack 中位數監控。`Cmax`、`κ_ft`、`kappa_clamp_active` 全部進 trajectory。

### 5.4 callback 的成本與記憶體帳

- callback 內做 **3 次獨立 fwd+bwd**:WL(既有)、IO-only(既有 `io_term.py:339`)、**FT-only(新增)**;合併範數由**已在手的兩個座標梯度向量**組出,**不需第 4 次**。
- 額外暫存:2 個 `(2N,)` fp32 向量(30M cell ⇒ 2×240 MB),**登記進 G7**。
- 頻率 N=50 ⇒ adaptec1 每 run 多 12 次 backward,總開銷 < 2%。
- **「零額外 chunk pass」的宣稱範圍限於主 objective 的 forward/backward。**
- **必記欄位(F2)**:`f_effective = κ_ft·g_ft/g_io`(clamp 後的實際佔比)、`kappa_clamp_active`、`ratio_inst`、`ratio_ema`、`applied_force_l1 = λ_io·‖I + κ_ft F‖₁`、`cancellation_ratio = ‖I + κF‖₁/(‖I‖₁ + κ‖F‖₁)`。

### 5.5 **原子化 callback 契約**(修掉 M2 遺留缺陷,Codex F4)

M2 的 driver 先算 `lambda_io`、再 `update_ratio`、再 refresh(`run_placement_io.py:102-166`),因此 **refresh 看到的是舊係數,下一個 iteration 卻用新係數在跑,且沒有對應的版本遞增**。加上 `κ_ft` 後這個錯位會放大。v3 的契約:

```
1. evaluator 呼叫 → hard 指標、home_e(新值先落地)
2. 在當前 pos 量三個梯度範數:g_wl, g_io, g_ft(順序固定,互不影響)
3. κ_ft ← (g_ft ≤ ε_rel·g_io) ? 0 : clip(f_ft(τ_rel)·g_io/g_ft, 0, κ_max)
4. 合併範數 ← ‖I + κ_ft·F‖₁(由步驟 2 的兩個座標梯度組出);ratio_inst、ratio_ema 更新
5. Cmax ← 1 + κ_ft·(max_h ecc_max[h] − 1)_+      # κ_ft = 0 ⇒ Cmax = 1,絕不虛構懲罰
6. λ_io ← min(ρ_io·ramp·ratio_ema, c_lip·τ²/(γ·Cmax))      # 用步驟 4/5 的新值
7. obj_version += 1(**整個 callback 只遞增一次**) → refresh_nesterov_secant() → mark_refreshed()
```

**T3/T4 必須有的測試:**
- 契約測試:callback 結束後 `λ_io` 等於「用當次更新後的 `ratio_ema` 與 `Cmax` 重算」的值(即 refresh 看到的就是下一 iteration 會用的係數)。
- `obj_version` 在一次 callback 內**恰好 +1**;`needs_refresh()` 在 callback 後為 False。
- `g_ft = 0` 時 `κ_ft = 0` 且 `Cmax = 1`(**不得**出現 `Cmax = 1 + 100·(ecc−1)`)。
- 同一 iteration 內呼叫 `obj_fn` 三次,`home_e`/`κ_ft`/`λ_io` 逐位元相同。

**註 1:** `home_e` 與 `κ_ft` 掛在既有的 callback 事件點上,**不新增事件點**。
**註 2(重要):** 步驟 6 的重排**改變了 M2 的行為**,因此 `f_ft_max = 0` **不再與 M2 逐位元相同**。逐位元回歸鎖改為 `--callback-order legacy`(保留 M2 順序)+ `f_ft_max = 0`;**預設是 `atomic`**,並在 T4 加一個「legacy vs atomic 在 `f_ft_max=0` 下的 hpwl/io/ft 差異」的量測表,寫進 M3 報告(這是一個獨立的、可能有益的修正,不能偷渡)。

---

## 6. Q5 — 實驗設計與 exit(**Phase A,預先登記**)

### 6.1 噪聲底線

`deterministic_flag=1`;`σ_rep = 0`。adaptec1 k16 grid flat 的 seed 噪聲(`results/m2/noise/flat_seed100*.json`):`ft_count` = {2545, 2571, 2519, 2532, 2523} ⇒ **mean 2,538.0、σ 20.98(0.83%)、3σ 62.9**;`io_count` mean 29,987.4。

**未量測(T5 補)**:bigblue4 的 `σ_seed`;`ft_rg` 的 `σ_seed`。在補測前,任何 `ft_rg` 的顯著性宣稱不成立。

### 6.2 Exit 判準(adaptec1 k16 grid,det=1)

**seed 設計(Codex F7)**:**screening = seed 1000**;**confirmation = 獨立 seeds {1001, 1002, 1003}**,screening 用過的 1000 **不進** confirmation。對照組(flat、M2-best)必須跑**同一組 confirmation seeds**——flat 的 1001/1002/1003 已存在於 `results/m2/noise/`;**M2-best 需補跑 3 顆,計入 T6 預算**。

**判定統計**:對每個指標取 **paired difference** `d_s = metric(M3, seed s) − metric(control, seed s)`,以 `n = 3` 的單側 95% 信賴界 `d̄ + t_{0.95,2}·s_d/√3`(`t = 2.920`)判定;**必須是信賴界而不是點估計通過門檻**。

| # | 類型 | 條件(對 confirmation 的 paired 信賴界) | 對照 / 具體值 |
|---|---|---|---|
| **E1** | 必要(主) | `ft_mst(M3) − ft_mst(flat)` 的**上信賴界 < 0** | 同 seed 的 flat;flat mean 2,538.0 |
| **E2** | 必要 | `io_mst(M3) − io_mst(M2best)` 的上信賴界 ≤ `0.02 × 24,647 = 492`(即 ≤ 25,139 的等價 paired 形式) | 同 seed 的 M2-best |
| **E3** | 必要 | `hpwl(M3) − hpwl(M2best)` 的上信賴界 ≤ 0(**真支配**) | 同 seed 的 M2-best(seed 1000 值 74,895,086) |
| **E4** | 必要(一致性) | `ft_rg(M3) ≤ ft_rg(flat)` 且與 `ft_mst` 同號 | flat `ft_rg` = 2,454;**次尺,不得單獨宣稱改善** |
| **E5** | stretch | `ft_mst` 的 3-seed mean ≤ `0.85 × 2538.0 = 2157.3` ⇒ ≤ 2,157 | — |
| **E6** | stretch | `ft_mst` 的 3-seed mean ≤ M1 reweight 的 2,430 | M1 花 +1.26% hpwl 且 io 僅 29,869 ⇒ 同時達成 E6+E2 才是真 Pareto 前進 |
| **E7** | 必要 | ablation 每格有 JSON 且通過 `tests/test_probes_m3_schema.py` | spec §9 M3 第二條 + Codex 15 |

**bigblue4 k16**:同結構,現任者換成自己的——E1 用 flat `ft_count` 6,935;**E2/E3 的現任者 = T5 掃出的 `ρ*` 臂**(**不得**沿用 `ρ=0.40` 的 io 63,755 / hpwl 789,030,695,那是 `Δhpwl = +5.41%` 的臂);E6 用 M1 的 2,948。

**為何主判準用 `ft_mst`**:它是 M0/M1/M2 都用過、M3 一個位元都不改的尺;`ft_rg` 的幾何樂觀性要等 T11 才有界。

### 6.3 Ablation 矩陣(Phase B)

| 臂 | 設定 | case | 目的 |
|---|---|---|---|
| C0 | flat | 全部 | 基準 |
| C1 | M2 best(`f_ft_max = 0`,`--callback-order legacy`) | adaptec1 k16、bigblue4 k16 | 現任者 + 逐位元回歸鎖 |
| C1b | `f_ft_max = 0`,`--callback-order atomic` | adaptec1 k16 | 隔離 §5.5 註 2 的行為改變 |
| **C2** | `f_ft_max ∈ {0.1, 0.25, 0.5, 1.0}` | adaptec1 k16 | 主掃描 |
| C3 | FT-only(`L_IO` 關) | adaptec1 k16 | 隔離 FT 項;預期失敗 E2 |
| C4 | P0b 的次佳候選 | adaptec1 k16 | 驗證離線裁決在 in-loop 成立 |
| C5 | `S4g-bboxcov`(若未在 P0b 淘汰) | adaptec1 k16 | spec §8 明文要求 |
| C6 | `home` 重算頻率 {50, 200} | adaptec1 k16 | L5 |
| C7 | 最佳配置 × {k=8, k=32, slicing} | adaptec1 | 泛化;slicing 是 G4 最可能觸發的組態 |
| C8 | 最佳配置 + bigblue4 `ρ*` | bigblue4 k16 | 規模 |
| C9 | `--ft-ramp-mode {window, constant}` | adaptec1 k16 | 直接檢定 §5.2 的晚窗口假設 |

報表欄位 = M2 §8.2 + `io_rg`/`ft_rg`(exact/ub 區間)/`mst_excess`/`f_ft`/`f_effective`/`kappa_ft`/`kappa_clamp_active`/`Cmax`/`ratio_inst`/`ratio_ema`/`applied_force_l1`/`cancellation_ratio`/`ft_grad_l1`/`boundary_demand_{total,max,p90,mean,gini}`/`region_util_free_max_over_mean`/`home_churn`/`topology_class_counts`/`spearman_ft`。

時間預算:adaptec1 單臂約 2 分鐘 ⇒ screening 約 20 臂 ≈ 45 分;confirmation 3 臂 × 3 seed + M2-best × 3 seed ≈ 25 分;bigblue4 `ρ*` 校準 4 臂 + 3 顆 flat seed + 主臂 ≈ 2 小時。**Phase B 總計 < 4 GPU 小時**(不含 P0b 的 2–3 小時)。

### 6.4 bigblue4 per-case `ρ`(T5)

掃 `ρ_max ∈ {0.05, 0.10, 0.15, 0.20}`(annealed τ、`f_ft_max = 0`),取滿足 `Δhpwl ≤ +2%` 的最大 `ρ`;同時跑 3 顆 flat seed 取得該 case 的 `σ_seed`。先驗預估 `ρ*` 在 0.10–0.15。**所有 bigblue4 的 M3 臂都必須跑在 `ρ*` 上。**

---

## 7. Q6 — 風險與 gate

| ID | 量測 | 門檻 | 觸發後動作 |
|---|---|---|---|
| **G1** FT 力量異常(雙側) | `g_ft/g_io`(原始比)與 `kappa_clamp_active` | 下側 **< 0.02**;上側 `κ_ft` 撞 clamp | 下側 ⇒ FT 項在該 regime 無力,查 `home_e` 是否退化;上側 ⇒ 凍結 `κ_ft` 前值並在報告標記。`f_ft` 正規化已把 P4 的「比值上衝 2.371」變成 no-op |
| **G2** alignment(診斷) | GP 末端 per-net Spearman(evaluator `ft_rg` vs `L_FT` 的 per-net 貢獻),**支撐釘死為 `ft_rg > 0 ∪ surrogate ≥ 其 90 百分位`** | 報告值,參考線 0.20 | 只記錄(§3.5:`L_FT` 是方向不是估計量) |
| **G3′** **in-loop 無效**(真 gate) | 最佳 `f_ft` 臂的 `ft_mst` vs 同 `ρ` 的 `f_ft = 0` 對照 | 4 個 `f_ft` 臂都沒讓 `ft_mst` 下降超過 3σ(62.9) | 依序試 P0b 的次佳候選、`S4g-bboxcov`;皆敗則誠實記錄 negative result,M3 退回「只報 FT、不優化 FT」 |
| **G4** RG 被鑽漏洞 | `ft_rg` 下降 ≥10% 但 `ft_mst` 上升 > +3% | 任一臂 | 以 T11 的 maze 結果判定,必要時改 bounded-detour |
| **G5** IO/FT 打架 | `f_ft` 增大時 `io_mst` 單調惡化且在 `f_ft = 0.25` 就超過 E2 | — | 限制 `f_ft ≤ 0.1`,把 M3 定位成 Pareto 前緣上多一維 |
| **G6** `home` 抖動 | 相鄰 callback 間 `home_e` 改變比例 | > 20% | 提高重算頻率(C6)或改用 pin 質心 region |
| **G7** 規模契約 | 10M 合成拓撲的 chunked fwd+bwd 峰值(含 §5.4 的 callback 暫存與候選自己的累加器) | > 8 GB 或 OOM | 不得凍結 T2 介面;依 §3.3.3 的三段式退路 |
| **G8** 穩定性 | `obj_eval_count` 增量中位數 ≥ 5,或 `Δhpwl > +10%`,或 `check_divergence` | 任一 | 檢查 §5.5 契約與 §5.3 上限;降 `f_ft`/`ρ_max` |
| **G9** **鏈轉星**(新增,F3) | trajectory 的拓撲類別轉換:`chain → star` 的淨數量,對照同 `ρ` 的 `f_ft = 0` 臂 | 淨轉換 > `3σ_seed(該量)`,且同時 `ft_mst` 未改善 | 該候選在把「鏈」擠成「星」而非真的減少 FT ⇒ 切到 overlap-aware 候選(`S4e`);若已是 `S4e` 則降 `f_ft` |

**額外風險(報告必須記載):** R1 `L_FT` 的值與真實 FT 無定量關係(P0 總量比 1.26–33.5);R2 evaluator 的 `(E,K)` int64 累加器(T1 修);R3 `ft` 是小整數計數(E1 用 paired 信賴界);R4 候選的拓撲偏誤(G9);R5 `f_ft` 讓 objective 偏離 M2 已驗證的良好區(C2 固定 `ρ*` 只動 `f_ft`);R6 梯度抵銷讓 `ratio_ema` 分母塌陷(記 `cancellation_ratio`,< 0.3 凍結 `ratio_ema`);R7 slicing 下兩尺分歧最大(exit 只用 `ft_mst`)。

---

## 8. Q7 — Task 分解(**Phase A / Phase B**)

規約沿用 M2 §11:每個 task 一次 TDD 迴圈;測試一律 `$DP/.venv312/bin/python -m pytest`;新程式碼全在 `ioplace/`;**M3 不需要新的 DREAMPlace patch**。

### Phase A —— 與 S4 形式無關,可立即進 writing-plans

| Task | 類型 | 內容 | 驗收 |
|---|---|---|---|
| **T0-a** snapshot 工具 | 純軟體 | `run_placement_io.py` 加 `--snapshot-iters`/`--snapshot-dir`,在 callback 內寫 `results/m3/snapshots/<tag>_it<NNNN>.npz`(`node_x,node_y,iteration,overflow,tau,lambda_io`) | 不影響既有數值:**未指定 snapshot 時與現行 run 逐位元相同**;`flat`/`m2best` 重跑的最終 npz 與 `results/m2/` 既有檔逐位元相同(det=1) |
| **T0-b** 探針重發 | 純軟體 | **7 支全部**(P0 `probe_ft_surrogate_soft`、P1 `probe_l_convention`、P4 `probe_beta_tau`、P6 `probe_free_area_util`、舊四支中的 `probe_m3_rg`/`probe_m3_bb`/`probe_m3_surrogate`/`probe_m3_util`)改成:路徑由 `__file__` 推導(可 `--repo-root` 覆寫)、原子寫入、**統一 env schema**(`command`、`python_executable`、`python_version`、`hostname`、`utc_timestamp`、`repo_commit`、`dp_commit`、`input_sha256`、`exactness`)。新增 **P2 `probe_home_churn`** 與 **P3**(`demand_per_len` 的 total/max/p90/mean/Gini) | 新增 `tests/test_probes_m3_schema.py` 驗每支 JSON 的 schema 與旗標;`tests/test_probes_m3_regression.py` 用小型合成 case 鎖 `ST_e −(Λ_e−1) = FT_e ≥ 0` 與 `Λ_e−1 ≤ ST_e ≤ io_mst`;**重發後 §2.2/§3.2/§4.1 的每個數字必須不變,不符即更新本文**。註:P1 單次約 **35 分鐘** |
| **T1** evaluator RG 擴充 | 純軟體 | `EvalResult` 新增 `io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home`/`per_net_topology_class` 與 **exact/ub 分離**;`region_graph(rg) -> (adj, D, ell, path_mask)` 放 `ioplace/region_graph.py`(**`path_mask` 供 S4e 用,tie 規則見 §3.3.3**);`evaluator_ref` brute-force、`evaluator_gpu` 走 Λ≤3 closed form / 4–8 批次 Dreyfus–Wagner / >8 metric-closure MST **並展開成實際子樹**;順帶修 R2 | 舊欄位逐位元不變;新欄位 CPU/GPU 等價;小 case 對 brute-force Steiner 全對;**`FT = ST + 1 − Λ` 每條 net 精確成立**;聚合值以區間報且 `n_nets_ub` 有值;拓撲分類有單元測試(§1 的五類各一個玩具 case);bigblue4 evaluator runtime 增量 < +30% |
| **T3** schedules | 純軟體 | `ScheduleState` 新增 `f_ft_max`/`tau_start`/`tau_full`/`ft_ramp_mode`/`kappa_ft`/`kappa_max`/`eps_rel`/`Cmax`/`home_version`;§5.2 的 `τ_rel` 兩點 ramp;§5.5 的原子化更新;`ratio_inst`/`cancellation_ratio` | 純函數測試:ramp 端點(`τ_rel ≥ 0.12 ⇒ 0`、`≤ 0.05 ⇒ 1`)與 §5.2 表的四個中間值;`constant` 模式無分母;`f_ft_max = 0` 關閉;**§5.5 的四個契約測試**;`obj_version` 一次 callback 恰好 +1 |
| **T11** maze 校驗 | 純軟體 + 實驗 | region-id lattice 上 crossing=1、length=ε 的 Dijkstra,抽 1 萬條 net,量 RG 相對可實現路徑的樂觀誤差 | **M3 報告前無條件完成**;中位偏差 > 20% ⇒ 報告中所有 `ft_rg` 敘述降級為「拓撲下界」。可與 T0/T1/T3 並行 |
| **T0-c(P0b)** | **實驗**(Phase A 的最後一個,**Phase B 的解鎖條件**) | 依 §3.4 實作 `ioplace/diagnostics/probes_m3/probe_ft_direction.py` 並執行 | 產出 §3.4.6 的 JSON(含 `preregistration.doc_sha256`);`summary.per_candidate[*].verdict` 對每個候選給出「合格/受限合格/淘汰」;**若全部淘汰 ⇒ 走 §3.4.3 的 fallback,M3 範圍縮減** |

### Phase B —— P0b 判定後才解鎖

| Task | 類型 | 內容 | 驗收 |
|---|---|---|---|
| **T2** S4 op | 純軟體 | 依 P0b 選出的候選擴充 `io_term.py`(介面契約見 §3.5-2);`IoTermRef` 同步 | 玩具 case 手算;fp64 `gradcheck`;chunked 與 unchunked 一致(1e−10);`κ_ft = 0` 完全短路;**`τ → 0` 收斂到該候選的解析 hard 極限**(相對誤差 < 1e−6),另記 hard 值 / `ft_rg` 的比值但不設門檻;`pos[num_movable:]` 梯度恆 0;hinge 精確邊界等價(Codex 7);**10M spike(G7)** |
| **T4** driver | 純軟體 | `--f-ft-max --tau-start --tau-full --ft-ramp-mode --home-period --callback-order`;§5.5 的原子契約落地;trajectory 加 §6.3 欄位 | JSON schema 齊全;`legacy + f_ft_max=0` 與 M2 存檔逐位元相同;`atomic vs legacy` 差異表;version invariant 全程零違反 |
| **T5** bigblue4 `ρ*` + 噪聲 | 實驗 | §6.4 | 存在 `Δhpwl ≤ +2%` 的臂;bigblue4 `σ_seed` 有值 |
| **T6** `f_ft` 掃描 | 實驗 | C2 + C3 + C9 + C1/C1b;機械判定 G1–G9 | Pareto 表 + 圖;選出最佳 `f_ft`;screening 全表存檔 |
| **T7** ablation + confirmation | 實驗 | C4/C5/C6/C7/C8 + 前 3 名在 **seeds 1001–1003** 的 confirmation(含同 seed 的 M2-best 對照) | 每格有 JSON;拓撲轉換表;paired 信賴界 |
| **T8** 報告 | 文件 | `docs/results/m3-differentiable-ft-report.md` | 每個數字可追到 `results/m3/*.json`;對 spec §9 M3 兩個 exit 條件逐條打勾/打叉;FAIL 誠實記錄 |

### 依賴序

```
Phase A:  T0-a ─┬─ T0-b ─┐
                └────────┴─ T0-c (P0b) ══╗   (解鎖 Phase B)
          T1 ────────────────────────────╣
          T3 ────────────────────────────╣
          T11 ───────────────────────────╝
Phase B:  T2 ── T4 ── T5 ── T6 ── T7 ── T8
```

- **T1 必須在 T0-c 之前**(P0b 需要 `path_mask` 與拓撲分類;若 T1 未完成,P0b 可用探針內的暫時實作,但必須在 T1 完成後**重跑一次**確認一致)。
- **T0-a 必須在 T0-c 之前**(P0b 的主力狀態是 snapshot)。
- **T5 必須在任何 bigblue4 臂之前。**
- **T11 必須在 T8 之前。**

---

## 9. 低信心總表

| ID | 段落 | 內容 | 由誰解決 | 狀態 |
|---|---|---|---|---|
| L1 | §2.4 | RG 的幾何樂觀性 | T11(無條件) | 開放 |
| L2 | §2.4 | Λ≥4 上界鬆緊度 | T1 | 開放 |
| ~~L3~~ | §2.3 | L 形約定敏感度 | P1 | **結案**:0.10–1.73%,`io_mst` 逐位元不變 |
| L4 | §3.7 | `β` 窗口(僅 S4b 分支) | P4 + P0b | 部分結案 |
| L5 | §3.7 | `home_e` 凍結代價 | P2 / C6 / G6 | 開放 |
| ~~L6~~ | §3.7 | 數值下限 | P4 | **結案**:K=16 下 fp32 underflow 0/24;風險只在 K≥32 或 β<0.25 |
| ~~L7~~ | §4.1 | region 利用率分母 | P6 | **結案**:free-area 口徑下 M2 亦改善 |
| L8 | §5.3 | 步長上限形式 | G8 | 已降級為經驗上限,不再宣稱是 Lipschitz 界 |
| ~~L9~~ | §6.2 | E2 的 0.85 係數 | — | **結案**:降為 stretch E5 |
| ~~L10~~ | §4.2 | `P[h,k,ab]` 均分假設 | — | **結案**:S7 退出 |
| ~~L11~~ | 全文 | 只在最終 placement 上驗過 | P0b 的 snapshot | **由 §3.4.1(a) 的 12 個中段 snapshot 解決** |
| ~~L12~~ | §3.7 | per-Λ `ft_rg` 診斷 | — | **作廢**:Codex F3 證明它偵測不到鏈轉星,由 §3.4.5 的拓撲轉換矩陣 + G9 取代 |
| **L13** | §3.7 | P0b 仍是單步實驗,不蘊含多步收斂 | T6 的 in-loop(G3′) | 開放(**主要殘餘風險**) |
| **L14** | §5.2 | `τ_start/τ_full` 的選擇建立在「相同 iteration 快照對照」上,**不是因果證明** | C9 臂(window vs constant) | 開放 |
| **L15** | §3.3.3 | `S4e` 的規範最短路 tie 規則是**約定**;`(E,A)` 記憶體在 30M 尺度未驗 | P0b 的 reverse tie-break 子臂 + T2 的 10M spike | 開放 |

---

## 10. 證據附錄

### 10.1 repo 內、可重現(但 provenance 待 T0-b 統一)

| 探針 | JSON | 支撐 |
|---|---|---|
| P0 `probe_ft_surrogate_soft.py`(151e426) | `probe_ft_surrogate_soft.json` | §0 的「無人全過」、§3.2 全表、§3.2.1 的 68.5%/29.7%、G2 的 `spearman_ftpos` |
| P1 `probe_l_convention.py`(58bb09b) | `probe_l_convention.json` | §2.3 的 `rel_diff` 0.0010/0.0028/0.0092/0.0173 與 `io_h == io_v`(4/4) |
| P4 `probe_beta_tau.py`(58bb09b) | `probe_beta_tau.json` | §3.7/§7 G1 的 0.152–2.371、`n_underflow_fp32 = 0` |
| P6 `probe_free_area_util.py`(58bb09b) | `probe_free_area_util.json` | §4.1(a) 全表、`l7_triggered = false` |
| `probe_m3_rg.py` / `probe_m3_bb.py` | `probe_m3_rg.json` / `probe_m3_bb.json` | §2.2 全表、§4.1(b)、`n_adj_pairs` |
| `probe_m3_surrogate.py` | `probe_m3_surrogate.json` | §3.2 的 bucket 份額(0.01–0.35)與 Λ=2 份額(78.7–97.5%) |
| `probe_m3_util.py` | `probe_m3_util.json` | total-area 利用率(已被 P6 取代為次要口徑) |
| `results/m2/noise/flat_seed100*.json` | — | §6.1 的 σ_seed |
| `results/m2/sweep/adaptec1_k16_rho0.40_annealed.json`、`results/m2/noise/flat_det1_rep0.json` 的 `trajectory` | — | §5.2 的校準表 |

**全部七支探針都要在 T0-b 重發**(F8);重發後數字若有變動,本文對應段落必須同步更新。

### 10.2 探索性(**不作為裁決依據**,由 P0b 取代)

v2 §3.3 的固定 RMS 下降實驗(腳本 `/tmp/m3probe/probe_ft_{grad_detach,combined_step,step_hpwl}.py`、`probe_gate_mass.py`)。其缺陷見 Codex F1:純量 `U` 的分母可小至 5 條 net、噪聲底來自單一隨機向量、RMS 而非 L1 正規化、缺 WL/density/Nesterov/`ft_mst`。**保留的只有兩項機制性觀察**(已寫進 §3.2):(a) detach 把 `FT-only` 步的 `Δhard_λ_sum` 從 +6.48% 降到 +0.81%,證實 F2 的「λ 基準獎勵多觸 region」機制;(b) membership gate 以下的遠區 `q` 質量佔 10.8–80.8%。**其餘數字(含所有 `U` 值與倍數宣稱)一律撤回。**

**與 M2 資產的關係:** S1(`ioplace/ops/soft_assign.py`)、chunked 四趟結構(`ioplace/ops/io_term.py::_IoFn`)、schedule 與 version invariant(`ioplace/schedules.py`)、secant refresh(`ioplace/dp_hook.py`)、driver callback(`ioplace/drivers/run_placement_io.py:102-172`)全部沿用;M3 的新機制都是加法,唯一的**行為改變**是 §5.5 註 2 的 callback 重排(有 `legacy` 旗標保留回歸鎖)。
