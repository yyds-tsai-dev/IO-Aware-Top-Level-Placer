# M3 設計草案 v2:可微 feed-through(S4)+ 邊界容量項(S7)

- 日期:2026-08-13
- 狀態:**v2(2026-08-13,對抗性審查後修訂)**——v1 的 §3(S4 形式)與 §4(S7 立論)被實測推翻,本版重寫該兩節並逐條回應兩份審查;可交 writing-plans 轉成實作計畫
- 對應 spec:`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` §2(FT 定義)、§5.3(S4)、§5.4(S7)、§6(evaluator)、§9 M3 列
- 繼承資產:`docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md`(v2)的 S1 L1-SDF 軟歸屬、S2 product-form、§2.5 chunked-k 契約、§4.2/§5.2 schedule、§6.2 params-borne patch、§6.4 secant refresh、§7.2 噪聲 regime——**M3 一律繼承,不重新發明**
- 前置結論:`docs/results/m2-differentiable-io-report.md`(M2 exit PASS:−18.5% io @ +1.31% hpwl;但 **ft_count 反升 +38.5%(adaptec1)/+17.3%(bigblue4)**)
- 對抗性審查(v2 逐條回應,見 §0.1):`docs/reviews/2026-08-13-m3-draft-v1-adversarial-opus.md`(F1–F15)、`docs/reviews/2026-08-13-m3-draft-v1-adversarial-codex.md`(findings 1–16)
- **證據狀態**:v1 的探針已遷入 `ioplace/diagnostics/probes_m3/`,結果在 `results/m3/probes/*.json`;P0(commit 151e426)、P1/P4/P6(commit 58bb09b)已跑。本版另有**兩個 ad-hoc 補測**(§3.3、§3.4),腳本尚在 `/tmp/m3probe/`,**必須由 T0 以 P0b 遷入 repo 才能被引用為定案依據**——引用處一律標記 `[ad-hoc,T0-P0b 待遷入]`。

---

## 0. 摘要(v2 裁決一覽)

**必須先講的壞消息(P0 的誠實結論):** 依 Opus F1 預先登記的合格判準(總量比 `ratio_vs_true ∈ [0.7,1.5]` 且 `mass_on_ft0 < 0.20`),**13 個候選 × 18 個格(6 placement × 3 τ_rel)裡沒有任何一個候選全過**。最佳者 `S4b-gated_beta0.5` 與 `S4b-gated_beta1.0` 各只有 **4/18**,而且合格的格全部集中在 `τ_rel = 0.03`(k32 有兩格 0.10);**在 `τ_rel = 0.30` 全部 13 個候選、全部 6 個 placement 一格都不合格**。Opus F2 提出的「hard `Λ̄` 凍結基準」族(`S4b-hardbase*`)比對應的 λ-baseline 版本**更差**(worst ratio 102–295 vs 31–208)。原始 `S4b-current`(v1 的定案式)在 18 格中 **0 格合格**,最壞總量比 **207.6**(β=0.25)。來源:`results/m3/probes/probe_ft_surrogate_soft.json` → `summary.per_candidate[*].n_eligible / worst_ratio_cell / worst_mass_on_ft0_cell`。

**v1 的 S4b 定案因此撤回。** 但 v2 的結論不是「M3 做不了」,而是**判準本身選錯了量**:`λ_io` 由梯度範數自動正規化(`ratio_ema = ‖∇WL‖₁/‖∇(合併項)‖₁`,`ioplace/schedules.py:86-90`),surrogate 的**絕對數值在 objective 裡會被除掉**,活下來的只有**方向**。以方向為判準重測(§3.3 的固定步長下降實驗),排序完全顛倒:**P0 合格率最低的 `S4a-soft`(星形和)在「每犧牲 1 單位 `io_rg` 換得的 `ft_rg` 下降量」上贏過所有 P0「合格」候選 2–30 倍**,而且 HPWL 代價不高於純 IO 項。

| # | 問題 | v2 裁決 | 與 v1 的差異 | 信心 |
|---|---|---|---|---|
| Q1 | routing 模型與 FT 定義 | **維持**:region-adjacency Steiner 為 M3 的參考 routing 模型;evaluator 只**新增** `io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home` 欄位並**分離 exact / upper-bound 兩套**(Codex 1、Opus F11);**`ft_mst` 是 exit 主判準,`ft_rg` 降為次要一致性欄位**,其幾何樂觀性由 T11 maze 校驗在報告前無條件補上(Codex 2) | 上/下界語意分參照物;per-net 一致率取代 Spearman;L 形敏感度已由 P1 實測(0.10–1.73%)⇒ 措辭收斂 | 高 |
| Q2 | S4 可微形式 | **改判:`L_FT = Σ_e w_e · Σ_k q_{e,k} · (D[home_e,k] − 1)_+`(S4a-soft,星形和)。** 無 `β`、無 `R_e`、無 `N_e/M_e`、**無 hinge**;`∂L_FT/∂q_{e,k}` 是**與 `q` 無關的常數係數** ⇒ 併進 M2 backward 的 `w_pins` 即可,零新增 `(E,)` 常駐狀態。**F2 的負係數問題被結構性消滅**(合併係數 `1[λ>1] + κ_ft·(D−1)_+ ≥ 0` 對任意 `κ_ft ≥ 0` 成立) | v1 的 S4b 偏心距全族(含 gated / hardbase / detach 變體)降為 ablation | 中高(離線方向實驗支撐;in-loop 未驗) |
| Q3 | S7 容量項 | **改判:S7 不進 M3 objective,降為診斷欄位。** 面積容量在 total-area 與 **free-area 兩種分母下都沒有被 M2 惡化**(P6:`l7_triggered=false`);邊界需求的「+42% 惡化」是 median 假象(Opus F3),**絕對峰值 demand-per-length 在全部 5 個組態都下降 8.3–20.6%**;且 v1 的 `L_cap` 有三套互斥 routing 模型、0 次齊次容量、永不熄火的 softplus(Opus F4、Codex 9/10/11)。S7 重新設計延到 M4(有 LEF/DEF pitch 可定錨絕對容量時) | v1 是「條件啟用 + T9」,v2 是「不做 + 診斷欄位 + M4 條件重開」 | 高 |
| Q4 | 完整 objective 排程 | **不引入第三個 ρ,也不引入獨立的 `κ_ft` 旋鈕。** 主旋鈕改為**無量綱力量佔比** `f_ft`(FT 項施加的梯度 L1 佔 IO 項的比例),`κ_ft` 於每次 callback 由 `f_ft·g_io/g_ft` 導出並 clamp;`f_ft` 隨 overflow **連續 ramp**(`of_on=0.90` 起 0,`of_ft_full=0.50` 起滿),**不新增離散事件**。正規化語意寫明是 **iso-total-force**(Opus F8、Codex 12) | v1 的 `κ_ft` 常數旋鈕語意被證明會被正規化抵銷;啟用時序改成連續 ramp | 中高 |
| Q5 | 實驗與 exit | **重寫**:E1 主判準改用**未動過的 MST 尺** `ft_mst`,門檻 = flat 的 3-seed mean 減 `3σ_seed`;E3 改 `hpwl ≤ M2-best`(真支配,Opus F6.1);全部判準以**預先登記的 3-seed 固定種子集**的 mean 判定(Codex 14);bigblue4 的現任者由 T5 的 `ρ*` 臂定義;stretch 判準與必要判準分離 | E2 的 0.85 係數降為 stretch;E1 的 M1 比較降為 stretch;整數門檻修正 | 中高 |
| Q6 | 風險/fallback | 8 個 gate(G1–G8,§7)。**G1 改雙側**(P4 實測梯度比從未低於 0.05,真正的風險是**上尾** 2.37);**G3′「in-loop 無效」**取代 v1 的 Spearman gate 成為真 gate;G4 保留 RG 鑽漏洞監控 | G1/G2 重寫;新增 G3′ | 中 |
| Q7 | task 分解 | T0(探針,含 **P0b 方向效用探針**)→ T1(evaluator RG,exact/ub 分離)→ T2(S4 係數擴充)→ T3(schedules)/T4(driver)→ T5(bigblue4 `ρ*` + 噪聲底線)→ T6(`f_ft` 掃描)→ T7(ablation)→ T8(報告);條件 task T10(S4 變體)、T11(maze,**報告前無條件**);**T9(S7)移出 M3** | 少一個 task(S7),多一個探針(P0b) | — |

**一句話結論(v2):** 精確的 FT surrogate(偏心距/soft Steiner 下界)在 GP 實際會看到的**軟 regime**裡不是「稍微偏差」而是**量級崩壞**(最壞 295 倍),而修正量級的 membership gate 會把 **11–81% 的遠區洩漏質量的梯度直接歸零**(§3.4)——兩條路互斥。M3 因此放棄「可微地估計 FT」,改成「**可微地施加一個能降低 FT 的方向**」:`Σ_k q_{e,k}(D[home_e,k]−1)_+` 是一個 **hop-距離加權的軟 crossing 罰**,它不估計 FT(值不可信),但在固定步長的下降實驗裡以 2.68–17.4 的效率把 `ft_rg` 換下來(噪聲底 0.05),而 FT 的**計數與宣稱一律回 evaluator**。

---

## 0.1 v1 → v2 變更摘要(逐條回應審查)

| 審查 finding | 裁決 | v2 的處置(在哪一節) |
|---|---|---|
| **Opus F1** [BLOCKER] S4b 的證據全在 hard limit;soft 版高估 5.14×、83% 質量落在 FT=0 的 net | **成立,採納** | P0 已跑(`probe_ft_surrogate_soft.json`,sanity cell 復現 total=17388.8 / ratio=5.140 / mass_on_ft0=0.830,與 F1 的 17389/5.14/0.83 相符)。§3.1 改判 S4a-soft;§3.2 記錄 P0 全表;**F1 的合格判準本身被 §3.5 改判**(降為 tier-2「值可否當估計量」,tier-1 改成方向效用) |
| **Opus F2** [BLOCKER] `κ_ft>1` 時 `(λ−1)` 係數為負;建議改用凍結 hard `Λ̄` | **機制成立,修法否決;問題被新形式消滅** | P0 實測 hardbase 族**比 λ-baseline 更差**(§3.2 表)。ad-hoc 實測 detach-λ 確實消除「獎勵多觸 region」的誘因(`FT-only` 步的 `Δhard_λ_sum` 從 +6.48% 降到 +0.81%),但同時吃掉大部分 FT 效果(`Δft_rg` 從 −11.6% 掉到 −2.1%)。**v2 的 S4a 形式裡根本沒有 `(λ−1)` 項**,合併係數對任意 `κ_ft ≥ 0` 恆非負 ⇒ F2 不可能發生(§3.1、§3.4) |
| **Opus F3** [BLOCKER] S7 立論的「+42% 惡化」是 median 假象 | **成立,採納** | §4.1 改報 total / 絕對 max / max-over-mean,並明寫 M2 在**絕對峰值上改善 8.3–20.6%**;S7 退出 objective(§4.3) |
| **Opus F4** [MAJOR] S7 內部三套 routing 模型、`C_ab` 0 次齊次、softplus 永不熄火 | **成立** | S7 退出 ⇒ v1 §4.2 的公式整段刪除;修法要求原文轉載到 §4.4 作為 M4 重開時的前置條件 |
| **Opus F5** [MAJOR] 借來的「3× bucket 機械判準」被不對稱套用且數字漏報 slicing 的 0.01 | **成立** | §3.2 撤回該判準(它是 M2 的 **degree-bucket 梯度佔比**判準,`2026-08-06-m2-differentiable-io-design.md:259`,不是 Λ-bucket 數值份額);bucket 份額改列為診斷,S4b 的 L4+ 份額比更正為 **0.01–0.35** |
| **Opus F6** [MAJOR] E1–E6 不是 Pareto 支配,四個洞 | **成立,採納** | §6.2 全部重寫:E3 改 `hpwl ≤ 74,895,086`(M2-best 本身);E1 主判準改 `ft_mst`;3-seed mean;bigblue4 現任者 = T5 的 `ρ*` 臂;0.85 係數降為 stretch |
| **Opus F7** [MAJOR] Lipschitz 護欄除以會反向的量 | **成立,但修法改良** | §5.3:S4a 的 `max_{e,k} c_{e,k}/w_e` **有閉式解**(`1 + κ_ft·(ecc_max_global − 1)`),不需要 backward 內的 reduce;護欄改為 `λ_io ≤ c_lip·τ²/(γ·Cmax)`,並依 Codex 13 明確降級為**經驗步長上限,不是 Lipschitz 界** |
| **Opus F8** / **Codex 12** `ratio` 量合併項的理由寫反 | **成立** | §5.2 明寫合併正規化 = **iso-total-force**,`f_ft` 只改 IO/FT 的分配比例;並新增 cancellation ratio 監控欄位 |
| **Opus F9** / **Codex 4** T2 的「τ,β→0 收斂到 hard `ft_rg`」不可能通過 | **成立;新形式使其良定** | S4a 無 `β`,`τ→0` 的極限是唯一且解析的 `Σ_e w_e Σ_{k∈touched(e)}(D[home_e,k]−1)_+`。§8 T2 驗收改成收斂到**該解析值**;它與 `ft_rg` 的差距(hard S4a / ft_rg 比值)只記錄不設門檻 |
| **Opus F10** / **Codex 5** `a/(q·N)` 是 0/0、`max(N,ε)` 會靜默壓掉 R | **成立;新形式使其消失** | S4a 沒有 `N`/`R`/除法。P0 探針已按 F10 實作(`probe_ft_surrogate_soft.py:311-319`:ε=0 + `where(N>0,·,0)`),該寫法連同 β assert 一併移交給 T10 的 S4b ablation 分支 |
| **Opus F11** / **Codex 1** 上/下界語意混用;Λ≥4 的 `ft_rg` 不是 distinct-region 數 | **成立,採納** | §2.2 分別標參照物;§2.5 要求 T1 把 Λ≥4 的解**展開成 `G_R` 上的實際子樹**後數非 terminal 頂點,並分離 `st_exact/ft_exact` 與 `st_ub/ft_ub`,聚合值以區間報 |
| **Opus F12** §2.2 理由 1 低估自己的證據且 Spearman 是錯的統計量 | **成立** | §2.2 理由 1 改報 per-net 一致率(grid 上 `io_rg > io_mst` 與 `ft_rg > ft_mst` 各 0 條;`ft_rg < ft_mst` 僅 141/91 條且每條差 1),ρ 降為附註 |
| **Opus F13** / **Codex 3** 「Λ=2 佔 87–97%」錯 | **成立** | §3.2 更正為 **78.7–97.5%**(`probe_m3_surrogate.json` 的 A2_k32=0.851、A2_slic=0.787) |
| **Opus F14** G2 沒定義支撐 | **成立** | §7 G2 釘死支撐為 `ft_rg > 0 ∪ surrogate ≥ 其 90 百分位`,門檻用 P0 的 `spearman_ftpos` 重校為 0.20,且 G2 降為診斷(真 gate 是 G3′) |
| **Opus F15** β 掃描集合自相矛盾 | **成立;新形式使其消失** | S4a 無 β;β 只出現在 T10 的 S4b ablation,掃描集合統一為 {0.5, 1.0}(P0 顯示 0.25 全滅、2.0 未測但量級判準先驗排除) |
| **Codex 1** [BLOCKER] metric-closure 估計被當成真值報 | **成立** | 見 F11 列;另 §10 要求所有既有探針 JSON 補 `exactness` 旗標後重發 |
| **Codex 2** [BLOCKER] RG 模型的幾何 gap 未量測就宣稱物理意義 | **成立,採兩者兼施** | `ft_rg` **降為次要欄位**(exit 主判準改 `ft_mst`),且 T11 maze 校驗改為 **M3 報告前無條件執行**(不再掛在 G4 條件上) |
| **Codex 3** [BLOCKER] S4b 違反自己的 bucket 規則 | **成立** | 見 F5 列;S4b 已不是定案 |
| **Codex 4** [BLOCKER] S4b 不收斂到 hard `ft_rg`,T2 有不可能的驗收 | **成立** | 見 F9 列 |
| **Codex 5** [BLOCKER] 梯度式的數值安全性 | **成立** | 見 F10 列 |
| **Codex 6** [MAJOR] L6 的 underflow 診斷與修法都錯 | **成立,且被 P4 部分推翻** | P4 實測 K=16 下 **fp32 underflow 計數 = 0**(`probe_beta_tau.json` 每格 `n_underflow_fp32`),`N_min_fp64` 與閉式 `exp(−ecc_max/β)` 逐位元吻合。§3.6 把 L6 改寫成「原則性風險:`K≥32` 或 `β<0.25` 時 `exp` 必須在 fp64 求值;K=16/β≥0.25 已實測不觸發」 |
| **Codex 7** [MAJOR] ReLU 邊界語意在 ref 與 custom backward 不一致 | **成立** | §8 T2 驗收新增:`IoTermRef` 改用 `torch.relu`(或顯式 strict mask)並加 `λ_e == 1` 的精確邊界等價測試。S4a 無 hinge,只剩 `L_IO` 這一個 |
| **Codex 8** [MAJOR] 「零額外 chunk pass」對 schedule 診斷不成立 | **成立** | §5.4 明寫 callback 內是 **3 次獨立 backward**(WL / IO-only / FT-only),合併範數由兩個座標梯度向量直接組出(不需第 4 次);並登記 2 個 `(2N,)` 暫存的記憶體帳 |
| **Codex 9/10/11** [BLOCKER] S7 的 demand 是星形流、`L_cap` 不是 hinge、容量參考漂移破壞 version invariant | **成立** | S7 退出 objective(§4.3);三條修法轉為 M4 重開的前置條件(§4.4) |
| **Codex 13** [BLOCKER] L8 不是 Lipschitz 護欄 | **成立** | §5.3 明確改名為「係數感知的經驗步長上限」,並說明它界的是**係數量級**不是 Hessian |
| **Codex 14** [BLOCKER] E1–E5 不是支配、可事後挑選、整數算錯 | **成立** | §6.2 全部重寫並**預先登記**;整數修正 `0.85×2454 = 2085.9 ⇒ ≤2085`、`1.02×24647 = 25139.9 ⇒ ≤25139`;3-seed 固定種子集;screening/confirmation 兩段式並承認 winner's bias |
| **Codex 15** [MAJOR] 探針 provenance 可被舊檔滿足 | **成立** | P0/P1/P4/P6 已 hermetic(repo 相對路徑、env metadata、input sha256、commit);§8 T0 要求**舊四個探針**(rg/bb/surrogate/util)以同一 provenance 規格重發,並新增 `tests/test_probes_m3_schema.py` 驗 schema/status/hash/exactness |
| **Codex 16** [MINOR] §2.2 的「只差 2.7–3.7%」推導錯 | **成立** | §2.2 更正為 **0.8–4.0%**(adaptec1 M2 4.028%、bigblue4 M2 0.799%),並標明每個相關係數用的是哪幾列 |
| **新輸入 P1(L3)** L 形約定敏感度已實測 | — | §2.4 L3 結案:相對差 0.10–1.73%(4 個 run 全低於 2% 門檻),且 `io_mst` 在兩種 walk 方向下**逐位元相同**(證實 spec §6 的 O(1) 特例);§2.3 對 MST 的否決理由**收斂措辭**——L 形敏感度是輕微效應,真正的否決理由是 `mst_excess` 佔比隨幾何在 13–65% 之間漂移 |
| **新輸入 P4(L4)** β×τ 梯度曲面 | — | §7 G1 改雙側:P4 的 24 格**從未觸發** `<0.05` 死區,反而 β=0.25 + 小 τ 時 FT/IO 梯度比高到 **2.371**;`f_ft` 正規化把這個無上界比值變成 no-op,殘餘風險改為 `κ_ft → ∞`(當 `g_ft → 0`)並以 clamp 處理 |
| **新輸入 P6(L7)** free-area 分母 | — | §4.1 L7 結案:free-area 口徑下 max/mean 亦**改善**(adaptec1 1.9271→1.8948、bigblue4 1.8725→1.8033),`l7_triggered=false`,無任何 region 的 free-area 利用率超過 0.978 ⇒「不加面積項」的裁決在兩種口徑下都成立 |

---

## 1. 新增名詞與符號(M2 §1 之外)

| 符號 | 意義 | 大小/來源 |
|---|---|---|
| `G_R` | **region adjacency graph**:K 個節點,兩 region 在 lattice 上共邊即相鄰 | 實測相鄰對數:adaptec1 k16 grid 24、k32 52、k16 slicing 33(`probe_m3_rg.json` `n_adj_pairs`) |
| `D[a,b]` | `G_R` 上的 hop 距離(all-pairs,Floyd–Warshall,K≤32) | (K,K) uint8,靜態 |
| `ecc_max[h]` | `max_k D[h,k]` | (K,) uint8,靜態;adaptec1 k16 grid = 6 |
| `Λ_e` | hard 觸及 region 數(M2 已有) | evaluator |
| `ST_e` | **region-graph Steiner cost**:`G_R` 上連接 `touched(e)` 的最小樹邊數 | 新增 evaluator 欄位,**分 exact / ub 兩套** |
| `io_rg` / `ft_rg` | `Σ_e ST_e` / `Σ_e (ST_e − (Λ_e − 1))` | 新增,**次要欄位**(exit 主判準是 `ft_mst`) |
| `mst_excess` | `io_mst − io_rg` | 報表欄位(Λ≥4 用 ub 時本身是下界,見 §2.5) |
| `home_e` | net e 的「主 region」= pin 數最多的 region,平手取最小 index | `(E,)` uint8,**每次 callback 由 evaluator 重算並凍結** |
| `c^FT_{e,k}` | **FT 係數** `= (D[home_e,k] − 1)_+` | 靜態查表,`(E,c)` uint8 per chunk |
| `f_ft` | **FT 力量佔比** `= κ_ft·‖∇L_FT‖₁ / ‖∇L_IO‖₁` | **主旋鈕**,隨 overflow ramp |
| `κ_ft` | `f_ft` 導出的實作權重 | 每 callback 更新,clamp 至 `[0, 100]` |
| `U` | **方向效用** = (相對 IO-only 多降的 `ft_rg`) ÷ (相對 IO-only 少降的 `io_rg`) | §3.3 的 P0b 判準;隨機方向的噪聲底 ≈ 0.05 |

**硬性分工(延續 M2 §1):** `evaluator_ref`/`evaluator_gpu` 的既有欄位語意與數值 **M3 不得改動**;新欄位一律**新增**,由既有等價測試 + 新增的「舊欄位逐位元不變」測試把關。所有 exit 判定與報表 FT 數字回 evaluator,**`L_FT` 只是 optimizer-facing 的方向,不是 FT 的估計量**(§3.5)。

---

## 2. Q1 — routing 模型與 FT 定義的耦合(維持 v1 裁決,修正語意與界)

### 2.1 定案

**M3 的參考 routing 模型 = region-adjacency Steiner。**

```
route(e)     := G_R 上連接 touched(e) 的最小 Steiner tree T_e(邊權 1 = 一次 boundary crossing)
crossings(e) := |E(T_e)| = ST_e
FT(e)        := |V(T_e) \ touched(e)| = (ST_e + 1) − Λ_e        # 對一棵**實際的樹**成立
io_rg = Σ_e ST_e = hard_lambda_sum + ft_rg;   io_mst = io_rg + mst_excess
```

### 2.2 證據(全部可追溯到 `results/m3/probes/probe_m3_rg.json` / `probe_m3_bb.json`)

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

**語意標註(Opus F11、Codex 1):** 表中 `io_rg`/`ft_rg` 對 `Λ≤3` 是**精確 Steiner**,對 `Λ≥4` 是 **metric-closure MST 上界**;因此 (a) 相對「RG 模型內的精確 Steiner」它們是**上界**,(b) 相對「真實幾何可實現的 crossing 數」`ft_rg` 是**下界**(RG 允許無成本繞路,見 L1),(c) `mst_excess = io_mst − io_rg_ub` 因此只是真實 excess 的**下界**,三分解在 Λ≥4 的部分不是恆等式。T1 必須修掉這一點(§2.5)。

四個支持本裁決的事實(數字已依審查更正):

1. **兩把 FT 尺在 grid 上 per-net 幾乎逐條相同**(取代 v1 用 Spearman 的說法,Opus F12):adaptec1 k16 全 216,930 條 active net 中,`io_rg > io_mst` **0 條**、`ft_rg > ft_mst` **0 條**、`ft_rg < ft_mst` 只有 **141 條(A2)/ 91 條(A0)且每條差恰好 1**。總量差 **0.8–4.0%**(Codex 16:adaptec1 M2 4.028%、bigblue4 M2 0.799%,由七個 grid 列重算)。⇒ **換模型不會憑空製造 FT 改善**;同時也意味著在 grid 上兩把尺**幾乎不帶額外資訊**,真正的分歧只在 rectilinear slicing(`ft_rg/ft_mst` = 0.73–0.75)。
2. **M2 在兩把尺上都讓 FT 變差**(adaptec1 `ft_rg` +37.9% / `ft_mst` +38.5%;bigblue4 +19.6%/+17.3%)⇒ M3 有真實的問題要解。
3. **恆等式把 detour 拆開了**:adaptec1 flat 的 5,789 = 2,454 真 FT(42%)+ 3,335 MST 次佳(58%);bigblue4 M2 的 9,237 = 8,071(87%)+ 1,166(13%)。**S2 已經吃掉 MST 次佳的部分**,剩下的幾乎全是真 FT。
4. **成本可負擔**:Λ 分布極度偏斜(adaptec1 k16:Λ=1 有 195,140、Λ=2 有 20,036、Λ=3 有 1,239、Λ≥4 只有 517)⇒ Λ=2 查表、Λ=3 取 `min_v Σ D`、Λ 在 4–8 走批次 Dreyfus–Wagner,成本只與 Λ≤K 有關,與 pin 數無關。

### 2.3 被否掉的選項(措辭依 P1 收斂)

| 選項 | 否掉理由 |
|---|---|
| **維持 MST 幾何為唯一模型** | (a) 沒有 `io = (Λ−1) + ft` 的分解,S2 與 S4 無法證明不重疊;(b) **主要理由**:`mst_excess` 佔 detour 的 13–65% 且**隨 placement 幾何劇烈漂移**(bigblue4 flat 65% → M2 13%),會污染跨臂比較;(c) 次要理由:L 形 walk 方向是任意約定——**但 P1 實測其影響輕微**:`ft_mst` 相對差 adaptec1 flat 1.73% / M2 0.28% / bigblue4 flat 0.92% / M2 0.10%(全部 <2%),且 `io_mst` 在兩個方向下**逐位元相同**(4/4 run)。v1 把這條寫成強證據是**過度主張**,v2 收回 |
| **候選 2(boundary-cost maze)當主 evaluator** | 每條 tree edge 在 lattice 上跑 A*/Dijkstra,對 12M–36M net 塞不進 GP 內圈。**保留為抽樣校驗器 → T11(v2 改為報告前無條件執行,Codex 2)** |
| **完全取代 MST 欄位** | 破壞 M0/M1/M2 可比性。定案是**雙尺並報**,且 **exit 主判準用未動過的 `ft_mst`** |
| **以 λ−1 為唯一 IO 定義** | 與 spec §2/D4 的計數語意衝突 |

### 2.4 低信心處(更新)

- **L1(維持):RG 模型的幾何樂觀性未量化。** `ft_rg` 是可實現 crossing 數的下界。→ **T11 maze 抽樣校驗,M3 報告前無條件執行**;若中位偏差 >20%,報告中所有 `ft_rg` 敘述必須降級為「拓撲下界」,並在 M4 考慮 bounded-detour 變體。
- **L2(維持):Λ≥4 的上界鬆緊度未知。** 目前僅佔 `ft_rg` 的 1–5%。→ T1 實作 Λ≤8 精確 Dreyfus–Wagner 並回報 exact/ub 差。
- **L3(結案):** P1 已實測,見 §2.3。**移出低信心表。**

### 2.5 T1 的硬性要求(由 Opus F11 + Codex 1 導出)

1. `EvalResult` 分離 `st_exact/ft_exact`(Λ≤8)與 `st_ub/ft_ub`(Λ>8),並提供 `n_nets_ub`;聚合值以 **[exact 下界, ub 上界] 區間**報,不得把 ub 混進單一 `io_rg`/`ft_rg` 標量後宣稱恆等式。
2. Λ≥4 的解必須**展開成 `G_R` 上的實際子樹**,`ST` 取該樹邊數、`ft_rg` 取該樹的**非 terminal 頂點數**(與 `evaluator_ref.py:113` 的 `len(passed − pin_regions)` distinct 語意同構),使 `FT = ST + 1 − Λ` 在每條 net 上**精確成立**。
3. 順帶修 R2:`evaluator_gpu.py:352` 的 `pin_bit_acc` 與 `:413` 的 `passed_bit_acc` 是 `(E,K)` int64(12M×32 各 3.0 GB),改成 packed int64 bitmask 直接累加或按 net 分塊。**列為 T1 驗收條件。**

---

## 3. Q2 — S4 可微 FT 項(**v2 改判**)

### 3.1 定案公式

```
常數(靜態,K≤32):  D[h,k]  (K,K) uint8
per-callback 更新:  home_e ∈ 0..K−1(evaluator 給:pin 數最多的 region,平手取最小 index)

L_IO = Σ_e w_e · ReLU(λ_e − 1)                                  # M2 既有,一字不改
L_FT = Σ_e w_e · Σ_k q_{e,k} · c^FT_{e,k},  c^FT_{e,k} = (D[home_e,k] − 1)_+
L_S24 = L_IO + κ_ft · L_FT
```

**梯度(這是本形式唯一重要的工程性質):**

```
∂L_S24/∂q_{e,k} = w_e · [ 1[λ_e > 1] + κ_ft · (D[home_e,k] − 1)_+ ]      # 與 q 無關的常數
```

M2 backward 的核心是 `∂L/∂p_{i,k} = c_{e,k}·exp(S_{e,k} − ℓ_{i,k})`(leave-one-out,`ioplace/ops/io_term.py:225-232` 的 `w_pins` × `exp(S_c − ell_pins)`)。M3 **只把 `w_pins` 從 `(E,)` 提升成 `(P',c)`**:

```
w_pins_c = w[net_idx].unsqueeze(1) * ( (lam > 1).double()[net_idx].unsqueeze(1)
                                       + kappa_ft * coef_c[net_idx] )       # (P', c)
coef_c   = (D_table[home_e][:, lo:hi] − 1).clamp(min=0)                     # (E, c) uint8
```

其餘(`A_i` 耦合項、`(−1/τ)·∂sdf/∂x`、`num_movable` 以上清零、fp64 `index_add`)**逐行沿用 `_IoFn` 的四趟結構**。

**記憶體帳(對照 M2 §2.5 契約):**
- 新增常駐:`home_e` `(E,)` uint8(10M-cell spike 下 E≈12M ⇒ **12 MB**)、`D_table` `(K,K)` uint8(可忽略)。
- 新增暫存:每個 chunk 一個 `coef_c` `(E,c)` uint8(c=16 ⇒ **192 MB**);對照既有的 `S_c` `(E,c)` fp64(**1.5 GB**)是 +12.8%,在 G7 的 8 GB 內。
- **不需要 `N_e`/`M_e`/`R_e`**;`ctx` 不多存任何張量。
- `κ_ft = 0` 時整段短路(比照 `lambda_margin == 0.0` 的既有寫法,`io_term.py:203`)⇒ **與 M2 逐位元相同**(回歸鎖)。

**極限(取代 F9/Codex 4 那條不可能的驗收):** `τ → 0` 時 `q_{e,k} → 1[k ∈ touched(e)]`,故
`L_FT → Σ_e w_e Σ_{k ∈ touched(e)} (D[home_e,k] − 1)_+`(hard S4a)。這是**唯一、路徑無關、可解析算出**的極限,沒有第二個溫度。

### 3.2 為什麼推翻 v1 的 S4b:P0 的軟 regime 實測

`results/m3/probes/probe_ft_surrogate_soft.json`,13 候選 × 6 placement × 3 `τ_rel` = 234 格。合格判準(Opus F1 預先登記):`ratio_vs_true ∈ [0.7,1.5]` 且 `mass_on_ft0 < 0.20`。

| 候選 | 合格格數 | 最壞總量比(格) | 最壞 `mass_on_ft0` |
|---|---:|---:|---:|
| S4a-soft | 2/18 | 33.47(A0 k16 grid, τ=0.30) | 0.961 |
| S4b-current β0.25 / 0.5 / 1.0(**v1 定案**) | 0 / 0 / 0 | 207.6 / 79.5 / 31.3 | 0.991 / 0.987 / 0.980 |
| S4b-hardbase β0.25 / 0.5 / 1.0(**Opus F2 的修法**) | 0 / 0 / 0 | 294.5 / 159.6 / 102.3 | 0.992 / 0.991 / 0.991 |
| **S4b-gated β0.25 / 0.5 / 1.0** | 0 / **4** / **4** | 3.85 / 1.71 / 0.354 | 0.808 / 0.594 / 0.380 |
| S4b-gated-hardbase β0.25 / 0.5 / 1.0 | 0 / 0 / 0 | 34.1 / 25.9 / 18.6 | 0.980 / 0.976 / 0.975 |

三個必須記錄的事實:

1. **沒有任何候選全過**;最佳者 4/18,且合格格全部落在 `τ_rel = 0.03`(k32 有兩格 0.10);**`τ_rel = 0.30` 全部 234 格中的對應 78 格無一合格**。
2. **F2 的 hardbase 修法讓事情更糟**(每個 β 上 hardbase 的最壞比值都比對應的 λ-baseline 大 1.5–3.7 倍)。機制:`Λ̄ = 1` 的 net 在 hard 基準 0 下 `ReLU(R) = R` 恆正,失去了 soft `λ−1` 與洩漏的部分抵銷。**F2 的診斷(負係數)成立,但它開的藥更毒**——v2 用「FT 項裡完全沒有 `(λ−1)`」來解這題,而不是換基準。
3. **v1 §3.2 的四候選比較(ratio 0.84–0.96、ρ 0.62–0.93)量的是 hard touched set,不是 op 會算的量**;同一組 placement 在軟 regime 下 S4b-current(β=0.5)是 **2.87–79.5 倍**。v1 的表移到 §10 存檔,不再作為裁決依據。

**同時更正的數字**(Opus F5/F13、Codex 3):S4b 的 L4+ bucket 份額比是 **0.01–0.35**(不是 v1 寫的 0.11–0.35;`probe_m3_surrogate.json` A2_slic = 0.01);Λ=2 佔 `ft_rg` 的 **78.7–97.5%**(不是 87–97%)。v1 §3.2 理由 2 借用的「3× bucket 機械判準」**撤回**:M2 的原判準量的是 **degree bucket 的梯度佔比**,不是 Λ bucket 的數值份額。

### 3.3 為什麼是 S4a:方向效用實驗(**ad-hoc,T0-P0b 待遷入**)

**設計:** 在 adaptec1 A2 k16 grid(M2 best placement,base `ft_rg` 3,383 / `io_rg` 22,693 / hpwl 74,895,086)上,對 movable node 走一步
`Δpos = −η·L_R·g/rms(g)`(η=0.02 ⇒ RMS 位移 53.4 DBU,所有臂等距),`g = ∇L_IO + κ·∇L_FT`,`κ` 取成 `f·‖∇L_IO‖₁/‖∇L_FT‖₁`(即 `f_ft = f`),然後**用 hard 的 region-graph 指標重新量** `ft_rg` / `io_rg`,並量 HPWL。

| τ_rel | 臂 | Δ`ft_rg` | Δ`io_rg` | Δhpwl | **U** |
|---|---|---:|---:|---:|---:|
| 0.03 | IO-only(= M2,`f_ft`=0) | **+1.45%** | −1.67% | +3.55% | — |
| 0.03 | **S4a-soft, f=0.25** | **−6.33%** | −1.24% | +3.39% | **2.68** |
| 0.03 | S4a-soft, f=1.0 | −9.73% | −0.42% | +2.93% | 1.33 |
| 0.03 | S4b-gated-detach β1.0, f=0.25 | −2.63% | −1.02% | +1.75% | 0.93 |
| 0.03 | S4b-gated-detach β0.5, f=0.25 | +0.24% | −1.48% | +3.13% | 0.93 |
| 0.03 | S4b-gated-λbase β0.5, f=0.25 | −1.95% | −1.30% | — | 1.35 |
| 0.10 | IO-only | +0.65% | −0.85% | +3.86% | — |
| 0.10 | **S4a-soft, f=0.25** | **−1.92%** | −0.83% | +3.85% | **17.4** |
| 0.10 | S4a-soft, f=1.0 | −9.10% | +0.00% | +3.96% | 1.69 |
| 0.10 | S4b-gated-detach β1.0, f=0.25 | −3.25% | +0.61% | +2.60% | 0.40 |
| 0.10 | S4b-gated-detach β0.5, f=0.25 | −2.01% | −0.02% | +3.69% | 0.47 |
| — | **隨機方向控制組(同 RMS)** | −4.70% | **+16.55%** | — | **0.05** |

`U` = (相對 IO-only 多降的 `ft_rg` 計數) ÷ (相對 IO-only 少降的 `io_rg` 計數)。隨機控制組給出噪聲底 **0.05**;注意隨機抖動本身會讓 `ft_rg` 下降 4.7%(因為它把 Λ 灌大 20%,而 `ft = ST − (Λ−1)`)——**這正是為什麼 `Δft` 不能單獨看,必須配 `Δio_rg` 一起判**,也是 v2 把 `U` 定為判準的理由。

**五個結論:**

1. **IO-only 一步就複製了 M2 的病灶**(`ft_rg` +1.45% / +0.65%,`io_rg` −1.67% / −0.85%)——M3 要打的東西在單步尺度上就看得見。
2. **S4a-soft 在 `f=0.25` 上支配所有 S4b 變體**:τ=0.03 時 U=2.68 vs 0.93–1.35;τ=0.10 時 U=17.4 vs 0.40–0.61(20–40 倍)。
3. **S4a 的 HPWL 代價不高於純 IO 項**(τ=0.03:+3.39% vs +3.55%;τ=0.10:+3.85% vs +3.86%)。P0 擔心的「83% 質量落在 FT=0 的 net 上會白費力氣」**沒有轉化成 WL 代價**。(注意:這些 Δhpwl 之所以都很大,是因為這是**沒有 WL/density 項**的純 IO/FT 下降步;只有臂與臂之間的比較有意義。)
4. **F2 的機制被獨立證實**:`FT-only` 步下,gated-λbase 的 `Δhard_λ_sum` 是 **+6.48%**(全部候選最差),detach 後降到 **+0.81%**——`−(λ−1)` 的梯度確實在**獎勵多觸 region**。但 detach 同時把 `Δft_rg` 從 −11.6% 砍到 −2.1%,淨效用沒有提升(U 1.35 → 0.93)。
5. `f=1.0` 買到更多 FT 但把 `io_rg` 的改善幾乎全部吐回(τ=0.10 時 Δ`io_rg` = 0.00%)⇒ **預設 `f_ft = 0.25`,掃描 {0.1, 0.25, 0.5, 1.0}**。

**已知侷限(必須寫進 T0-P0b 與報告):** 單一 placement、單一 η、單步、無 WL/density 項、`ft_rg` 在 Λ≥4 用上界。P0b 遷入時必須擴成 ≥2 placement(A0/A2)× ≥2 η(0.01/0.02)× 3 τ_rel,並加上 bigblue4 一格。

### 3.4 為什麼 S4b 在軟 regime 必敗(機制,不只是數據)

1. **`R_e = M_e/N_e` 對 `q` 是 0 次齊次**(比值),而 `λ_e − 1` 是 1 次齊次(和)。一條 Λ=1 的 net 只要對 D=2 的 region 漏 `q ≈ 0.02`,就得到 `R ≈ 1.04` 而 `λ−1 ≈ 0.02` ⇒ 被課 ~1.02 的 FT。這就是 83% 質量落在 `ft_true == 0` 的直接原因。
2. **修正量級的 membership gate(`q̃ = (q−0.5)_+/0.5`)會把梯度一起殺掉。** `∂q̃/∂q = 0` 當 `q < 0.5` ⇒ 對**正要開始洩漏**的 (e,k) 對梯度恆為 0。實測遠區(`D[home,k] ≥ 2`)的 `q` 質量落在 gate 以下的比例(**ad-hoc,T0-P0b 待遷入**):

   | placement | τ_rel=0.30 | 0.10 | 0.03 |
   |---|---:|---:|---:|
   | adaptec1 A0 k16 grid | 78.7% | 60.4% | 22.4% |
   | adaptec1 A2 k16 grid | **80.8%** | 57.3% | 10.8% |

   τ=0.30 時 gate 以下有 757,841 個 (e,k) 對、gate 以上只有 19,296 個。**gate 用「看不見 78–81% 的初期洩漏」換「總量看起來對」**——這正是 §3.3 裡 gated 系列 U 值只有 0.40–0.93 的原因。
3. **S4a 沒有這兩個問題**:係數是常數,`∂L_FT/∂q_{e,k}` 對任何 `q > 0` 的洩漏都給滿額壓力,而且大小正比於 hop 距離。

### 3.5 誠實定位:`L_FT` 是方向,不是估計量

`L_S24 = Σ_e w_e Σ_k q_{e,k}·[1[λ_e>1]/(觸及數) 的等價分攤 + κ_ft(D[home_e,k]−1)_+]`,實質上是**以 hop 距離加權的軟 crossing 罰**:相鄰 region(D=1)不額外收費(由 `L_IO` 負責),D≥2 的 region 按 `(D−1)` 加價。它**不是 Steiner FT 的估計量**(星形和對鏈狀 net 重複計費;P0 的總量比 1.26–33.5 就是這個誤差)。

因此 v2 明文規定:

- **所有 FT 的數值宣稱一律來自 evaluator 的 `ft_mst`(主)/ `ft_rg`(次)**;`L_FT` 的值只作為 trajectory 診斷,**報告中不得寫成「估計的 feed-through 數」**。
- 論文/報告的 novelty 敘述必須寫成「以 region-graph hop metric 加權的可微 crossing 項,經驗上降低 feed-through 計數」,**不得寫成「可微 Steiner feed-through」**。這是 v1 過度主張的地方。

### 3.6 被否掉的替代 S4

| 方案 | 否掉理由 |
|---|---|
| **S4b 偏心距全族(v1 定案 + hardbase + gated + detach)** | §3.2 的量級崩壞 + §3.4 的 gate 梯度盲區 + §3.3 的 U 值落後 2–40 倍;工程上還要背 `N/M` 累加器、`1/N`、`β`、underflow 守衛、路徑相依極限。**保留為 T10 的 ablation 臂 C4 = S4b-gated-detach β1.0**(該族在 U 上最好的一個) |
| **spec §5.3 原案:soft bbox 覆蓋 `cov_k(e)·(1−q_{e,k})`** | 幾何 bbox 覆蓋與 RG-Steiner 不同構;需第三個溫度;需獨立的 per-net soft min/max reduce。**保留為 ablation C5**(spec §8 明文要求做這個比較) |
| **soft-Steiner / 可微 MST / `min_h Σ_k q_k D[h,k]`** | `min_h` 需 `(E,K)` 累加器(12M×32 fp32 = 1.5 GB @10M、4.6 GB @30M);且 v1 §3.2 的 hard 實測(S4c/S4d)品質更差 |
| **二次型 `q_e^T W q_e`** | 每 net `O(K²)` 且需 `(E,K)`;對大 net 無界 |
| **S3 span proxy 當 FT 項** | `span` 是 Steiner 的弱下界,`L_FT` 可為負,clamp 後梯度死。S3 維持 M2 §9.3 的定位 |
| **REINFORCE / 有限差分打 evaluator 的 `per_net_ft`** | 高變異、需多次 evaluator 呼叫 |

### 3.7 低信心處(更新)

- **L4(改寫):`β` 的窗口** — S4a 無 β,本項只在 T10 的 S4b ablation 分支存在。P4 已實測 24 格的梯度曲面(`probe_beta_tau.json`):G1 死區(比值 <0.05)**從未觸發**,β=2.0 的最低比值 0.152;真正的異常是 β=0.25 + 小 τ 時比值上衝到 **2.371**。ablation 的 β 集合定為 **{0.5, 1.0}**。
- **L5(維持):`home_e` 凍結的代價未量測。** → **P2(T0)**:用 M2 trajectory 量相鄰 callback 間 `home_e` 改變的 net 比例。S4a 對 home 誤差的敏感度低於 S4b(係數每差一 hop 只變 1,不像 `R` 是軟 max 比值),但若 churn >20% 仍走 G6。
- **L6(改寫,Codex 6):** 「fp32 exp underflow」對 S4a **不存在**(沒有 exp)。對 T10 的 S4b 分支:P4 實測 K=16 下 `n_underflow_fp32 = 0`(24/24 格),`N_min_fp64` 與閉式 `exp(−ecc_max/β)` 逐位元吻合;**原則性風險只在 `K ≥ 32` 或 `β < 0.25`**(`exp` 必須在 fp64 求值,且 `ε=0` + `where(N>0,·,0)`)。
- **L12(新增):S4a 的星形重複計費對「鏈狀 net」的偏誤未在 in-loop 量測。** hard 極限下 S4a 對 Λ=3 鏈狀 net 高估 1(真 FT=0)。→ T6 的 trajectory 加報 per-Λ-bucket 的 `ft_rg` 變化,若 Λ=3 桶的 `ft_rg` 反升而 Λ=2 桶下降,代表項在把 net 從「鏈」擠成「星」,需在 T10 補做「以 `min(D[home,k], D 沿路徑)` 折扣」的變體。

---

## 4. Q3 — S7:**不進 objective,降為診斷欄位**

### 4.1 兩個立論前提都被實測推翻

**(a) 面積容量:兩種分母都沒有惡化。** `results/m3/probes/probe_free_area_util.json`(P6,L7 結案):

| case | 口徑 | flat max/mean | M2-best max/mean | 變化 |
|---|---|---:|---:|---:|
| adaptec1 k16 grid | free area(`movable / (region − fixed overlap)`) | 1.9271 | 1.8948 | **−1.67%** |
| adaptec1 k16 grid | total area(v1 用的口徑) | 2.2081 | 2.1843 | −1.08% |
| bigblue4 k16 grid | free area | 1.8725 | 1.8033 | **−3.70%** |
| bigblue4 k16 grid | total area | 1.9825 | 2.1654 | +9.23% |

`l7_triggered = false`(門檻 +10%);free-area 利用率最大值 0.978(adaptec1 A2),**無任何 region 超載**;`n_nonpositive_free_area_regions = 0`。⇒ **面積懲罰項不做**(加了只會與 electrostatic force 對打),改為每 callback 記錄 `region_util_free[k]` 進 trajectory。

**(b) 邊界需求:v1 的「+42% 惡化」是 median 假象**(Opus F3)。改報容量相關的統計量:

| run | 相鄰對數 | 絕對 max demand/len(flat → M2) | 變化 | median(flat → M2) | max/median |
|---|---:|---:|---:|---:|---:|
| adaptec1 k16 grid | 24 | 27.969 → 23.617 | **−15.6%** | 8.211 → 4.891 | 3.41 → 4.83 |
| adaptec1 k32 grid | 52 | 27.969 → 24.578 | **−12.1%** | 8.672 → 5.340 | 3.23 → 4.60 |
| adaptec1 k16 slicing | 33 | 23.358 → 21.412 | **−8.3%** | 9.204 → 7.663 | 2.54 → 2.79 |
| bigblue4 k16 grid | 24 | 70.094 → 55.625 | **−20.6%** | 30.106 → 19.207 | 2.33 → 2.90 |

(來源:`probe_m3_rg.json` / `probe_m3_bb.json` 的 `demand_per_len_stats.{max,med}`。Opus F3 另行重算 adaptec1 k16 的 `max/mean` = 2.842 → 2.947(**+3.7%**)與 p90 = 18.38 → 17.80(−3.2%);**這兩個量目前只存在於審查文件裡,T0-P3 必須把 total/max/p90/mean/Gini 全部寫進 JSON**。)

**五個組態的絕對峰值全部下降 8.3–20.6%。** v1 的 S7 存在理由與 25% 啟用 gate 都建立在 median 掉得比 max 快這件事上。

### 4.2 v1 的 `L_cap` 公式本身也不成立

即使前提成立,v1 §4.2 的實作有四個獨立缺陷(Opus F4、Codex 9/10/11),v2 全部接受:

1. **模型自相矛盾**:`Q[h,k]·P[h,k,(a,b)]` 是 home-rooted **星形流**——正是 v1 §3.2 花整節否掉的幾何;而校驗用的 `boundary_pair_demand` 又是第三套(MST + L 形 walk,`evaluator_ref.py:105`)。
2. **`C_ab = ν·ℓ_ab·(ΣD/Σℓ)` 是 0 次齊次**:對總需求下降完全無感,且對冷邊界灌需求可以放寬所有邊界。
3. **`softplus((D−C)/C)` 不是 hinge**:`D ≥ 0` ⇒ 參數 ≥ −1 ⇒ `σ(z) ≥ 0.269`,**梯度下界是最大值的 27%**;v1 §5.2 的「零梯度凍結」是死碼,R6 的風險描述是錯的;`D` 全零時 `C = 0` 還會除以零。
4. **`λ_cap` 上限量綱不符**、`C_ab` 每 callback 重算會破壞 `obj_version` invariant。

### 4.3 定案

**S7 不進 M3 objective(`κ_cap` 不存在,不是預設 0)。** M3 只做兩件事:

- **診斷欄位**:每次 callback 記錄 `boundary_demand_total / max / p90 / mean / gini`(per lattice edge 正規化)與 `region_util_free_max_over_mean`,寫進 trajectory 與最終 JSON。
- **M4 重開 gate(預先登記)**:若任一 M3 臂的**絕對 max demand-per-length** 相對同 case flat **上升**(3-seed mean,超過該量的 3σ 帶——`σ` 由 T5 的 3 顆 flat seed 同時量出),或 LEF/DEF case 上出現 `D_ab > ρ·ℓ_ab`,才在 M4 重開 S7。

**這使 v1 的 T9 移出 M3**,並消掉四個 blocker finding(Opus F4、Codex 9/10/11)。

### 4.4 M4 重開 S7 時的前置條件(存檔,勿遺失)

(a) `D_ab` 必須與 §2.1 的 routing 模型一致(從 Steiner 樹邊映到邊界對),或誠實改名為 star-demand 並撤回 §3.6 對星形的否決理由;(b) `C_ab` 固定為 **flat baseline 的實測值 × ν**,`ν ≥ 該 case flat 的 max/mean`,一次算完全程凍結;(c) 懲罰改真 hinge(`ReLU(·)²` 或 `softplus(z/θ)`,θ≤0.1);(d) `λ_cap = min(κ_cap·g_comb/g_cap, ρ_max·g_WL/g_cap)`,`g_cap = 0` 時 `λ_cap = 0`,每次變更遞增 `obj_version`;(e) 定義零總需求 / 無相鄰對 / 圖不連通 / 零長度邊界的行為。

---

## 5. Q4 — 完整 objective 的排程整合

### 5.1 目標函數

```
min  WA-WL + μ·density + λ_io·[ L_IO + κ_ft·L_FT ] + λ_margin·L_margin
                          └──── 同一個 IoFtTerm op,共用 backward ────┘
```

### 5.2 權重與排程(繼承 M2 §4.2/§5.2,只列差異)

| 量 | 規則 | 與 M2 的差異 |
|---|---|---|
| `τ` | `τ_rel(of)` log-linear 0.30 → 0.03(`schedules.py:7-12`) | **不變** |
| `ρ_io` | `ρ_max·clip((of_on−of)/(of_on−of_full))·ramp` | **不變** |
| `ratio_ema` | `‖∇WL‖₁ / ‖∇(L_IO + κ_ft L_FT)‖₁`,每 N=50 iter | **量合併項**。語意(Opus F8、Codex 12):這是 **iso-total-force** 正規化——合併項施加的總梯度範數 ≈ `ρ_io·‖∇WL‖₁`,**與 `f_ft` 無關**;`f_ft` 只決定這股力量在 IO 與 FT 之間怎麼分。v1 說「否則 `κ_ft` 會被抵銷掉」是**寫反了**:正是這個選擇把 `κ_ft` 的量級效應抵銷掉,而這正是我們要的(掃描才有可比性) |
| **`f_ft`** | `f_ft(of) = f_ft_max · clip((of_on − of)/(of_on − of_ft_full))`,`of_on=0.90`、`of_ft_full=0.50`;**連續函數,不新增離散事件** | **新增,主旋鈕**。預設 `f_ft_max = 0.25`(§3.3),掃描 {0.1, 0.25, 0.5, 1.0};`f_ft_max = 0` 逐位元退化成 M2 |
| `κ_ft` | 每 callback:`κ_ft = clip(f_ft(of)·g_io_l1/max(g_ft_l1, EPS), 0, 100)` | **新增,導出量不是旋鈕**。與 `ratio_ema` 同一個事件點更新 ⇒ 不新增事件點。上 clamp 處理 `g_ft → 0`(P4 的無上界比值風險反向) |
| `λ_io` | `ρ_io · ratio_ema`,步長上限見 §5.3 | 上限公式改(F7/Codex 13) |
| `home_e` | 每次 callback 由 evaluator 重算 | **新增**,與 `ratio_ema` 同事件點 |
| `κ_cap` / S7 | **不存在** | v1 的第三個權重取消 |
| 啟用時序 | IO 與 FT 共用 `of ≤ of_on = 0.90` 的同一個 20-iter ramp;FT 另外被 `f_ft(of)` 的連續 ramp 壓在 `of > 0.50` 時接近 0 | **不新增啟用事件**(v1 的 S7 事件消失) |

**`of_ft_full = 0.50` 的依據(這是「啟用時序」問題的答案):** `of = 0.504` 恰好對應 `τ_rel = 0.10`(由 `tau_rel_from_overflow` 反解:`0.03·10^((of−0.07)/0.83) = 0.10`)。而 M2 的 FT 退化**全部發生在這個點之後**——比對 `results/m2/sweep/adaptec1_k16_rho0.40_annealed.json` 與 `results/m2/noise/flat_det1_rep0.json` 的 trajectory:

| iteration | `of`(M2 臂) | `τ_rel` | flat `ft_count` | M2 `ft_count` | 差額 |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.798 | 0.226 | 1,391 | 1,383 | −8 |
| 400 | 0.618 | 0.137 | 1,872 | 1,826 | −46 |
| 450 | 0.490 | 0.096 | 2,412 | 2,668 | **+256** |
| 500 | 0.346 | 0.065 | 2,461 | 3,135 | **+674** |
| 550 | 0.193 | 0.042 | 2,529 | 3,427 | **+898** |
| 600 | 0.074 | 0.030 | 2,526 | 3,531 | **+1,005** |

⇒ **iteration 400 之前 M2 的 FT 與 flat 沒有差別;+980 的最終退化 100% 在 `τ_rel ≤ 0.137` 之後產生,75% 在 `τ_rel ≤ 0.096` 之後產生。** 把 FT 壓力集中在 `τ_rel ≤ 0.10` 的窗口:(a) 完全覆蓋要修的損害;(b) 避開 P0 顯示「所有候選全滅」的 `τ_rel = 0.30` 區;(c) adaptec1 的窗口約 iteration 445→GP 結束(≈600–650)= **155–200 iteration / 4 個 callback**,bigblue4 約 iteration 700→999 = **300 iteration / 6 個 callback**,足夠做功。

**為何用連續 ramp 而不是硬啟用門檻:** 硬門檻會新增一個離散事件(要遞增 `obj_version` + 觸發 secant refresh + 在 line search 內製造跳變);連續 ramp 與既有的 `ρ_io` 同型,零新增事件。`κ_ft` 只在 callback 更新(既有事件點),所以 objective 在 iteration 內仍是常數係數的光滑函數。

### 5.3 步長上限(取代 v1 的「多項 Lipschitz 護欄」)

Codex 13 成立:梯度範數比值不界定 Hessian。v2 **改名並降級**為「係數感知的經驗步長上限」:

```
Cmax = 1 + κ_ft · (max_h ecc_max[h] − 1)_+          # S4a 的閉式解,零計算成本
λ_io ≤ c_lip · τ² / (γ · Cmax)
```

理由:S4a 的 per-(e,k) 係數上界**有閉式解**(adaptec1 k16 grid:`max_h ecc_max[h] = 6` ⇒ `Cmax = 1 + 5κ_ft`),不需要 Opus F7 提的 backward 內 reduce。**明文聲明:這界的是係數量級對梯度 Lipschitz 常數的線性放大,不是曲率界;真正的安全機制是 G8 的 backtrack 中位數監控。** `Cmax` 與 `κ_ft` 進 trajectory。

### 5.4 callback 的成本與記憶體帳(Codex 8)

`f_ft → κ_ft` 的導出需要**分開的** `‖∇L_IO‖₁` 與 `‖∇L_FT‖₁`,無法從合併範數還原。定案:

- callback 內做 **3 次獨立 fwd+bwd**:WL(既有)、IO-only(既有 `io_term.py:339` 的 `io_grad_l1`)、**FT-only(新增)**。
- 合併範數 `‖g_io + κ_ft·g_ft‖₁` 由**已在手的兩個座標梯度向量**直接組出,**不需第 4 次**。
- 額外常駐:callback 期間 2 個 `(2N,)` fp32 向量(30M cell ⇒ 2×240 MB 暫存)。**登記進 §5.5 的記憶體帳與 G7。**
- 頻率 N=50 ⇒ adaptec1 每 run 多 12 次 backward,實測單次 backward 遠小於 1 秒,總開銷 <2%。
- **「零額外 chunk pass」的宣稱範圍收窄為「主 objective 的 forward/backward」**(該宣稱對照 `io_term.py:181-279` 成立)。

### 5.5 離散事件集合與強制順序

| 事件 | M2 有 | M3 新增 |
|---|---|---|
| IO/FT 項啟用(ramp 起點) | 有 | |
| `ratio_ema` 更新 | 有 | |
| `w_e` 校準(`α_io > 0`) | 有 | |
| margin 項開關 | 有 | |
| **`home_e` 重算 + `κ_ft` 更新** | | **掛在既有 `ratio_ema` 事件點上,不新增事件點** |

**強制順序(T4 必須有測試):** `home_e` 由 evaluator 更新 → 3 次梯度量測(WL / IO-only / FT-only)→ `κ_ft = f_ft(of)·g_io/g_ft` → `ratio_ema` 更新(用合併範數)→(選配 `w_e`)→ `refresh_nesterov_secant` → `mark_refreshed()`。`obj_version`/`refreshed_version` invariant(`ioplace/schedules.py:92-96`)原樣沿用;`home_e` 與 `κ_ft` 的變更都必須讓 `obj_version` 遞增。

**line-search 內常數性(M2 §6.3 坑 3)延伸:** `home_e`、`D` 查表、`κ_ft` 在一個 iteration 內**只讀**。T4 加測試:同一 iteration 內呼叫 `obj_fn` 三次,`home_e`/`κ_ft`/`λ_io` 逐位元相同。

**為何 `home_e` 一定要凍結:** 若在 forward 內即時取 `argmax_k`,`L_FT` 在 home 翻轉處**不連續**(兩個候選 home 的 `D` 行不同),等於在 line search 內部注入跳變。凍結後 `L_FT` 在 iteration 內是 `q` 的光滑線性函數。

---

## 6. Q5 — 實驗設計與 exit(**預先登記**)

### 6.1 量測 regime 與噪聲底線

沿用 M2 §7.2:`deterministic_flag=1`,`σ_rep = 0`。FT 的 seed 噪聲底線(adaptec1 k16 grid,flat):由 `results/m2/noise/flat_seed100*.json` 得 `ft_count` = {2545, 2571, 2519, 2532, 2523},**mean = 2,538.0、`σ_seed` = 20.98(0.83%)、`3σ` = 62.9 counts(2.48%)**。`io_count` mean = 29,987.4。

**未量測(T5 必須補):** bigblue4 的 `σ_seed`(3 顆 flat seed);`ft_rg` 的 `σ_seed`(Codex 14 指出 v1 把 `ft_mst` 的方差套到 `ft_rg` 上)。在補測之前,**任何 `ft_rg` 的顯著性宣稱一律不成立**。

### 6.2 Exit 判準(adaptec1 k16 grid,det=1;**本節即預先登記,T6 開跑前不得修改**)

**判定方式:** screening 階段每臂 1 顆 seed(1000);screening 的前 3 名(依 `ft_mst`,且滿足 E2/E3)進 confirmation,以**固定 3 顆 seed {1000, 1001, 1002}** 重跑,exit 用 **3-seed mean** 判。**承認 winner's bias**:confirmation 臂數(3)與 seed 集合於本節預先固定,報告必須同時列出 screening 的完整 20 臂表與 confirmation 的 3 臂 mean±range。

| # | 類型 | 條件 | 具體數值 | 來源 |
|---|---|---|---|---|
| **E1** | **必要(主)** | 3-seed mean `ft_mst` ≤ flat mean − 3σ_seed | **≤ 2,475** | `2538.0 − 62.9 = 2475.1` ⇒ 取 2,475(§6.1) |
| **E2** | **必要** | 3-seed mean `io_mst` ≤ 1.02 × M2-best `io_count` | **≤ 25,139** | `1.02 × 24,647 = 25,139.9`(Codex 14 的整數修正) |
| **E3** | **必要** | 3-seed mean `hpwl` ≤ M2-best `hpwl`(**真支配**,不是 +2% 盒子) | **≤ 74,895,086** | `results/m2/ablation/adaptec1_A2_k16_grid.json`(Opus F6.1) |
| **E4** | **必要(一致性)** | `ft_rg` ≤ flat `ft_rg`,且與 `ft_mst` 同號變化 | **≤ 2,454** | §2.2 表;**次要尺,不得單獨用來宣稱改善**(Codex 2) |
| **E5** | stretch | `ft_mst` ≤ 0.85 × flat mean | ≤ 2,157 | `0.85 × 2538.0 = 2157.3`;v1 的 E2 野心值降級 |
| **E6** | stretch | `ft_mst` ≤ M1 reweight 的 `ft_count` | ≤ 2,430 | `results/m2/ablation/adaptec1_A1_k16_grid.json`;M1 花了 +1.26% hpwl 且 io 只有 29,869 ⇒ M3 若同時達成 E6+E2 就是**真正的 Pareto 前進** |
| **E7** | 必要 | ablation 矩陣每格有對應 JSON,且通過 `tests/test_probes_m3_schema.py` 的 provenance/exactness 驗證 | — | spec §9 M3 第二條 + Codex 15 |

**bigblue4 k16:** 同結構,現任者換成該 case 自己的:E1 用 flat `ft_count` = 6,935 減 T5 量出的 3σ;**E2/E3 的現任者 = T5 掃描中滿足 `Δhpwl ≤ +2%` 的最佳 `ρ*` 臂**(**不得**沿用 `ρ=0.40` 的 io=63,755 / hpwl=789,030,695,那是 `Δhpwl = +5.41%` 的臂,Opus F6.4);E6 用 M1 的 2,948。

**為何主判準用 `ft_mst` 而不是 `ft_rg`:** `ft_mst` 是 M0/M1/M2 都用過、M3 一個位元都不改的尺(Codex 2 的「換尺自肥」防線);`ft_rg` 的幾何樂觀性要等 T11 才有界。

### 6.3 Ablation 矩陣

| 臂 | 設定 | case | 目的 |
|---|---|---|---|
| C0 | flat | 全部(沿用 M2 A0) | 基準 |
| C1 | M2 best(`f_ft_max = 0`) | adaptec1 k16、bigblue4 k16 | 現任者 / 逐位元回歸鎖 |
| **C2** | `f_ft_max ∈ {0.1, 0.25, 0.5, 1.0}` | adaptec1 k16 | **主掃描**(4 臂) |
| C3 | FT-only(`L_IO` 關) | adaptec1 k16 | 隔離 FT 項;§3.3 預期它會失敗 E2(`Δio_rg` 為正),用來確認 IO 項不可移除 |
| C4 | S4b-gated-detach β1.0(§3.3 中 S4b 族最好的一個) | adaptec1 k16 | 驗證離線的方向裁決在 in-loop 也成立 |
| C5 | spec §5.3 的 bbox/`cov_k` 變體 | adaptec1 k16 | spec §8 明文要求 |
| C6 | `home` 重算頻率 {50, 200} | adaptec1 k16 | L5 |
| C7 | 最佳配置 × {k=8, k=32, slicing} | adaptec1 | 泛化;slicing 是 G4 最可能觸發的組態 |
| C8 | 最佳配置 + bigblue4 `ρ*` | bigblue4 k16 | 規模 |
| C9 | `of_ft_full ∈ {0.50, 0.90}`(即「窗口 vs 全程」) | adaptec1 k16 | 驗證 §5.2 的啟用時序裁決 |

報表欄位 = M2 §8.2 加上 `io_rg`/`ft_rg`(含 exact/ub 區間)/`mst_excess`/`f_ft`/`kappa_ft`/`Cmax`/`ft_grad_l1`/`cancellation_ratio`/`boundary_demand_{total,max,p90,mean,gini}`/`region_util_free_max_over_mean`/`home_churn`/`spearman_ft`。

時間預算:adaptec1 單臂約 2 分鐘(M2 實測 110–140 秒;M3 多 12 次 callback backward 與 evaluator 的 Steiner 計算)⇒ screening 約 20 臂 ≈ 45 分鐘、confirmation 3 臂 × 3 seed ≈ 20 分鐘;bigblue4 的 `ρ*` 校準 4 臂 + 3 顆 flat seed + 主臂 ≈ 2 小時。**總計 < 3.5 小時 GPU。**

### 6.4 bigblue4 per-case `ρ`(T5 前置實驗)

M2 報告附註:bigblue4 直接套 adaptec1 的 `ρ*=0.40` 得 `Δio = −36.5%` 但 `Δhpwl = +5.41%`。**T5 = 在 bigblue4 k16 grid 上掃 `ρ_max ∈ {0.05, 0.10, 0.15, 0.20}`(annealed τ、`f_ft_max = 0`),取滿足 `Δhpwl ≤ +2%` 的最大 `ρ`**;同時跑 **3 顆 flat seed** 取得該 case 的 `σ_seed(ft_count)`。先驗預估 `ρ*` 在 0.10–0.15(M2 §5.2 的 λ 錨點顯示同 ρ 下 bigblue4 的 `λ_io` 大 1.83 倍)。**所有 bigblue4 的 M3 臂都必須跑在 `ρ*` 上。**

---

## 7. Q6 — 風險與 gate

| ID | 量測 | 門檻 | 觸發後動作 |
|---|---|---|---|
| **G1** FT 力量異常(**雙側**) | 每 callback 的 `g_ft_l1/g_io_l1`(原始比,未乘 `κ_ft`)與導出的 `κ_ft` | 下側 **< 0.02**(P4 24 格從未低於 0.15;S4a 實測 0.089–0.75);上側 **`κ_ft` 撞到 clamp 100** | 下側 ⇒ FT 項在該 regime 無力,記錄並檢查 `home_e` 是否退化;上側 ⇒ `g_ft` 崩塌,凍結 `κ_ft` 為前一值並在報告標記。**`f_ft` 正規化已把 P4 觀察到的「比值上衝到 2.371」變成 no-op** |
| **G2** alignment(**診斷,非 gate**) | GP 末端 per-net Spearman(evaluator `ft_rg` vs `L_FT` 的 per-net 貢獻),**支撐釘死為 `ft_rg > 0 ∪ surrogate ≥ 其 90 百分位`** | 報告值,參考線 0.20(P0 的 `spearman_ftpos`:S4a 在 τ=0.03 為 0.464–0.795) | 只記錄。v1 用它當 gate 是錯的——S4a 明文是方向不是估計量(§3.5) |
| **G3′** **in-loop 無效**(真 gate) | 最佳 `f_ft` 臂的 `ft_mst` vs 同 `ρ` 的 `f_ft = 0` 對照臂 | 全部 4 個 `f_ft` 臂都**沒有**讓 `ft_mst` 下降超過 `3σ_seed`(62.9) | S4a 在 in-loop 無效 ⇒ 依序試 C4(S4b-gated-detach)、C5(bbox 變體);兩者皆敗則誠實記錄 negative result,M3 退回「只報 FT、不優化 FT」 |
| **G4** RG 模型被鑽漏洞 | `ft_rg` 下降 ≥10% 但 `ft_mst` 上升 > +3%(> 3σ) | 任一臂 | objective 在吃 region-graph 的幾何樂觀性 ⇒ 以 T11 的 maze 結果判定,必要時改 bounded-detour |
| **G5** IO 與 FT 互相打架 | `f_ft` 增大時 `io_mst` 單調惡化,且在 `f_ft = 0.25` 就超過 E2 | — | 限制在 `f_ft ≤ 0.1`,把 M3 定位成「在 M2 的 Pareto 前緣上多一個維度」 |
| **G6** `home` 抖動 | 相鄰 callback 間 `home_e` 改變的 net 比例(P2 / trajectory) | **> 20%** | 提高重算頻率(C6)或改用 pin 質心 region;同時檢查 backtrack 中位數 |
| **G7** 規模契約 | 10M 合成拓撲的 chunked fwd+bwd 峰值(沿用 M2 T2 Step 6 的 spike,加 FT 係數與 §5.4 的 callback 暫存重跑) | **> 8 GB** 或 OOM | 不得凍結 T2 介面;先砍 `coef_c` 到 packed 4-bit 或加大 chunk 數 |
| **G8** 穩定性 | `obj_eval_count` 增量中位數 ≥ 5,或 `Δhpwl > +10%`,或 `check_divergence` | 任一 | 檢查 §5.5 的事件順序與 §5.3 的步長上限;降 `f_ft`/`ρ_max` |

**額外風險(無獨立 gate,但必須在報告記載):**

| # | 風險 | 證據 | 緩解 |
|---|---|---|---|
| R1 | **`L_FT` 的值與真實 FT 無定量關係**(P0 總量比 1.26–33.5、最大 96% 質量落在 FT=0 的 net) | `probe_ft_surrogate_soft.json` | §3.5 的明文規定:所有 FT 數字回 evaluator;`L_FT` 只進 trajectory 診斷欄 |
| R2 | evaluator 的 `(E,K)` int64 累加器在 12M net × K=32 是 3.0 GB 各一份 | 讀碼(`evaluator_gpu.py:352,413`) | T1 的驗收條件(§2.5-3) |
| R3 | `ft` 是小整數計數,相對噪聲比 io 大 | `σ_seed(ft)/mean = 0.83%` vs io 的 0.59% | E1 用 3σ;confirmation 3 seed |
| R4 | S4a 的星形重複計費把鏈狀 net 擠成星狀(L12) | hard 極限的結構性偏誤 | T6 加 per-Λ-bucket 的 `ft_rg` 變化表 |
| R5 | `f_ft` 讓 objective 偏離 M2 已驗證的良好區 | M2 的 `ρ*=0.40` 是在無 FT 項下調出來的 | C2 固定 `ρ_max = ρ*` 只動 `f_ft`;若最佳 `f_ft` 落在掃描邊界,再做 `ρ × f_ft` 的 2D 補掃(9 臂) |
| R6 | 梯度抵銷讓 `ratio_ema` 的分母塌陷 | 結構性(Codex 12) | trajectory 記 `cancellation_ratio = ‖g_io + κ g_ft‖₁/(‖g_io‖₁ + κ‖g_ft‖₁)`;< 0.3 時凍結 `ratio_ema`。S4a 的 `cos(∇L_FT, ∇L_IO)` 實測為 **+0.23 到 +0.86**(全正),抵銷風險低 |
| R7 | slicing 下兩把 FT 尺分歧最大(`ft_rg/ft_mst` = 0.73–0.75) | §2.2 表 | slicing 的 exit 只用 `ft_mst`;`ft_rg` 只作診斷 |

---

## 8. Q7 — Task 分解

規約沿用 M2 §11:每個 task 一次 TDD 迴圈;測試一律 `$DP/.venv312/bin/python -m pytest`;新程式碼全在 `ioplace/`;**M3 不需要新的 DREAMPlace patch**。

| Task | 類型 | 內容 | 驗收 |
|---|---|---|---|
| **T0** 探針 | 純軟體 | (a) **已完成**:P0 `probe_ft_surrogate_soft.py`(151e426)、P1 `probe_l_convention.py` / P4 `probe_beta_tau.py` / P6 `probe_free_area_util.py`(58bb09b);(b) **新增 P0b `probe_ft_direction.py`**:§3.3 的固定步長方向效用實驗(≥2 placement × 2 η × 3 τ_rel × 候選集合 {IO-only, S4a-soft, S4b-gated-detach β{0.5,1.0}, S4b-gated-λbase β0.5, RANDOM 控制},輸出 `U`、`Δft_rg`、`Δio_rg`、`Δhpwl`、`cos(∇L_FT,∇L_IO)`、combined 係數負質量比),外加 §3.4 的 gate 質量表;(c) **P2 `probe_home_churn.py`**;(d) **P3**:把 `demand_per_len` 的 total/max/p90/mean/Gini 全部寫進 JSON;(e) 舊四支探針(rg/bb/surrogate/util)以 P0 的 provenance 規格重發並加 `exactness` 旗標 | 每支輸出 `results/m3/probes/*.json` 含 env metadata + repo/dp commit + command + input sha256 + exactness;**新增 `tests/test_probes_m3_schema.py`** 驗 schema 與旗標(Codex 15);`tests/test_probes_m3_regression.py` 用小型合成 case 鎖 `ST_e −(Λ_e−1) = FT_e ≥ 0` 與 `Λ_e−1 ≤ ST_e ≤ io_mst`;**本文 §3.3/§3.4 的每個數字可追溯**,不符即更新本文。註:P1 單次執行約 **35 分鐘**(bigblue4 全量 evaluator 跑兩遍) |
| **T1** evaluator RG 擴充 | 純軟體 | `EvalResult` 新增 `io_rg`/`ft_rg`/`per_net_steiner`/`per_net_home` 與 **exact/ub 分離**;`region_graph(rg) -> (adj, D, ell)` 放 `ioplace/region_graph.py`;`evaluator_ref` brute-force 參考、`evaluator_gpu` 走 Λ≤3 closed form / Λ 4–8 批次 Dreyfus–Wagner / Λ>8 metric-closure MST **並展開成實際子樹**;順帶修 R2 | 舊欄位逐位元不變;新欄位 CPU/GPU 等價;小 case 對 brute-force Steiner 全對;**`FT = ST + 1 − Λ` 在每條 net 上精確成立**(F11);聚合值以區間報且 `n_nets_ub` 有值;bigblue4 evaluator runtime 增量 < +30% |
| **T2** S4 係數擴充 | 純軟體 | `ioplace/ops/io_term.py` 的 `IoTerm`/`_IoFn` → `IoFtTerm`/`_IoFtFn`:`w_pins` 由 `(P',)` 提升為 `(P',c)`,加 `home`/`D_table` buffer 與 `kappa_ft` 參數;`IoTermRef` 同步擴充 | 玩具 case 手算 `L_FT`;fp64 `gradcheck`;chunked 與 unchunked 一致(fp64 1e−10);**`κ_ft = 0` 與 M2 逐位元相同(回歸鎖)**;**`τ → 0` 時 `L_FT` 收斂到解析的 hard S4a(相對誤差 < 1e−6),並另外記錄 hard S4a / `ft_rg` 的比值不設門檻**(F9/Codex 4);`pos[num_movable:]` 梯度恆 0;**`λ_e == 1` 的精確邊界上 ref 與 production 的次梯度一致**(Codex 7,`IoTermRef` 改用 `torch.relu`);**10M spike 重跑含 §5.4 的 callback 暫存(G7)** |
| **T3** schedules 擴充 | 純軟體 | `ScheduleState` 新增 `f_ft_max`/`of_ft_full`/`kappa_ft`/`home_version`/`Cmax`;`f_ft(of)` 連續 ramp;`κ_ft` 導出 + clamp;合併範數的 `update_ratio`;§5.3 的步長上限;`cancellation_ratio` | 純函數測試:端點、單調性、`f_ft_max = 0` 退化、`obj_version` **只在既有事件點**遞增(新增 `home`/`κ_ft` 不新增事件點)、`κ_ft` clamp 邊界、`g_ft = 0` 時不除零 |
| **T4** driver | 純軟體 | `run_placement_io.py` 新增 `--f-ft-max --of-ft-full --home-period`;callback 內 3 次梯度量測與 §5.5 的強制順序;trajectory 新增 §6.3 的欄位 | JSON schema 齊全;`f_ft_max = 0` 的 run 與 M2 存檔逐位元相同;事件順序測試;同 iteration 內 `obj_fn` 三次常數性;version invariant 全程零違反 |
| **T5** bigblue4 `ρ*` + 噪聲 | **實驗** | §6.4 的 4 臂 ρ 掃描 + 3 顆 flat seed | 存在 `Δhpwl ≤ +2%` 的臂;bigblue4 的 `σ_seed(ft_count)` 有值;寫進 M3 報告第一節 |
| **T6** `f_ft` 掃描 | **實驗** | C2(4 臂)+ C3 + C9 + C1 對照;機械判定 G1–G8 | Pareto 表 + 圖(x = Δhpwl%、y = Δft_mst%,標 flat/M1/M2/M3 前緣與 3σ 帶);選出最佳 `f_ft`;screening 全表存檔 |
| **T7** ablation + confirmation | **實驗** | C4/C5/C6/C7/C8 + 前 3 名的 3-seed confirmation | 每格有 JSON;per-Λ-bucket 診斷表;confirmation 的 mean±range |
| **T8** 報告 | 文件 | `docs/results/m3-differentiable-ft-report.md`:regime、`ρ*` 校準、`f_ft` 掃描、四方對照(flat/M1/M2/M3)、ablation、雙尺 FT + 三分解表、T11 的 maze 校驗結果、G1–G8 判定、**對 spec §9 M3 兩個 exit 條件逐條打勾或打叉** | 每個數字可追到 `results/m3/*.json`;FAIL 就照 M1/M2 慣例誠實記錄 |
| **T11** maze 校驗 | 純軟體 + 實驗 | region-id lattice 上 crossing=1、length=ε 的 Dijkstra,抽 1 萬條 net,量 RG 模型相對可實現路徑的樂觀誤差分布 | **M3 報告前無條件完成**(Codex 2);中位偏差報告;>20% 則報告中所有 `ft_rg` 敘述降級為「拓撲下界」 |
| **T10**(條件:G3′) | 純軟體 + 實驗 | S4b-gated-detach 與 bbox/cov 變體的深掃(含 β ∈ {0.5,1.0}、F10 的數值守衛) | 與 S4a 在同一張 Pareto 圖比較 |
| ~~T9 S7~~ | — | **移出 M3**(§4.3);M4 條件重開,前置條件見 §4.4 | — |

### 依賴序

```
T0 -- T1 -- T2 --+-- T4 -- T5 -- T6 -- T7 -- T8
                 |                      |
                 +-- T3 --+             +-- T10 (條件:G3')
                 |
                 +-- T11 ---------------------+   (無條件,擋 T8)
```

- **T0 必須最先**;P0b 是 §3 定案的唯一直接依據,**沒有 P0b 就沒有 S4a 的裁決基礎**。
- **T1 必須在 T2 之前**(`home_e` 由 evaluator 供給、`D` 由 `region_graph` 供給、`ft_rg` 是 T2 的對照真值)。
- **T2 的 10M spike(G7)是 T4 的前置門檻**。
- **T5 必須在任何 bigblue4 臂之前**。
- **T11 可與 T2–T7 並行,但必須在 T8 之前完成**。

---

## 9. 低信心段落總表(v2 更新)

| ID | 段落 | 不確定的是什麼 | 由誰解決 | 狀態 |
|---|---|---|---|---|
| L1 | §2.4 | RG 模型的幾何樂觀性 | T11(**無條件**) | 開放 |
| L2 | §2.4 | Λ≥4 的 metric-closure 上界鬆緊度 | T1 的 Dreyfus–Wagner | 開放 |
| ~~L3~~ | §2.3 | L 形約定敏感度 | P1 | **結案**:0.10–1.73%,`io_mst` 逐位元不變 |
| L4 | §3.7 | `β` 窗口(僅 T10 的 S4b 分支) | P4 + T10 | 部分結案:G1 死區從未觸發,上尾 2.371 |
| L5 | §3.7 | `home_e` 凍結 50 iter 的代價 | P2 / C6 / G6 | 開放 |
| ~~L6~~ | §3.7 | `N`/`M` 累加的數值下限 | P4 | **結案(對 S4a 不適用)**:K=16 下 fp32 underflow 計數 0;K≥32 或 β<0.25 才是風險 |
| ~~L7~~ | §4.1 | region 利用率的分母 | P6 | **結案**:free-area 口徑下 M2 亦改善,`l7_triggered=false` |
| L8 | §5.3 | 步長上限的形式 | G8 監控 | 降級為經驗上限(Codex 13),不再宣稱是 Lipschitz 界 |
| ~~L9~~ | §6.2 | E2 的 0.85 係數 | — | **結案**:降為 stretch E5;主判準改 `flat mean − 3σ` |
| ~~L10~~ | §4.2 | `P[h,k,ab]` 的均分假設 | — | **結案**:S7 退出 objective |
| **L11** | 全文 | S4a 只在**離線的最終 placement** 上驗過;GP 過程中(overflow 高、cell 尚未展開)`home_e` 與 `q` 的品質未知 | T6 的 trajectory;`of_ft_full = 0.50` 的 ramp 是對此的結構性緩解 | 開放(**承重牆**) |
| **L12** | §3.7 | S4a 的星形重複計費是否把鏈狀 net 擠成星狀 | T6 的 per-Λ-bucket 表 | 新增 |
| **L13** | §3.3 | P0b 的方向效用只量了單步、單一 placement、無 WL/density 項 | T0-P0b 擴充 + T6 的 in-loop 驗證 | 新增(**S4a 裁決的主要風險**) |

---

## 10. 證據附錄

### 10.1 已在 repo 內、可重現的探針

| 探針 | JSON | 支撐本文的哪些數字 |
|---|---|---|
| `probes_m3/probe_ft_surrogate_soft.py`(P0,commit 151e426) | `results/m3/probes/probe_ft_surrogate_soft.json` | §0 的「無人全過」、§3.2 全表(`summary.per_candidate[*].{n_eligible,worst_ratio_cell,worst_mass_on_ft0_cell}`)、§3.5 的總量比範圍、G2 的 `spearman_ftpos`(`cells[*].spearman_ftpos`) |
| `probes_m3/probe_l_convention.py`(P1,commit 58bb09b) | `results/m3/probes/probe_l_convention.json` | §2.3 的 `rel_diff` 0.0010/0.0028/0.0092/0.0173 與 `io_h == io_v`(4/4 run)。runtime ≈ 35 min |
| `probes_m3/probe_beta_tau.py`(P4,commit 58bb09b) | `results/m3/probes/probe_beta_tau.json` | §3.7/§7 G1 的梯度比範圍(`runs[*].grid[*].ratio_ft_over_io`,0.152–2.371)、`n_underflow_fp32 = 0`、`N_min_fp64` |
| `probes_m3/probe_free_area_util.py`(P6,commit 58bb09b) | `results/m3/probes/probe_free_area_util.json` | §4.1(a) 全表(`cases[*].runs[*].{max_over_mean,total_area_max_over_mean,max,n_nonpositive_free_area_regions}`)、`l7_triggered = false` |
| `probes_m3/probe_m3_rg.py` / `probe_m3_bb.py` | `probe_m3_rg.json` / `probe_m3_bb.json` | §2.2 全表、§4.1(b) 的 `demand_per_len_stats.{max,med}`、`n_adj_pairs`。**T0 必須補 provenance 與 exactness 旗標後重發** |
| `probes_m3/probe_m3_surrogate.py` | `probe_m3_surrogate.json` | §3.2 更正的 bucket 份額(0.01–0.35)與 Λ=2 份額(0.787–0.975)。**hard-limit 的四候選表已不作為裁決依據,僅存檔** |
| `probes_m3/probe_m3_util.py` | `probe_m3_util.json` | v1 §4.1 的 total-area 利用率(已被 P6 取代為次要口徑) |
| (既有)`results/m2/noise/flat_seed100*.json` | — | §6.1 的 `σ_seed(ft) = 20.98`(0.83%)、mean 2,538.0 |
| (既有)`results/m2/sweep/adaptec1_k16_rho0.40_annealed.json`、`results/m2/noise/flat_det1_rep0.json` 的 `trajectory` | — | §5.2 的 FT 退化時序表(`trajectory[*].{iteration,overflow,tau,ft_count}`) |

### 10.2 ad-hoc,**T0-P0b 必須遷入後才可引用為定案依據**

| 腳本(暫存) | 內容 | 支撐 |
|---|---|---|
| `/tmp/m3probe/probe_ft_grad_detach.py` | 5 候選 × 2 placement × 3 τ_rel 的 `‖∇L_FT‖₁/‖∇L_IO‖₁`、`cos(∇L_FT,∇L_IO)`、combined 係數負質量比(κ ∈ {0.5,1,2,4} × detach/λbase)、FT-only 固定 RMS 下降步的 `Δft_rg`/`Δhard_λ_sum`、隨機方向控制組 | §3.3 的 F2 驗證(detach 把 `Δhard_λ_sum` 從 +6.48% 降到 +0.81%)、§0.1 F2 列 |
| `/tmp/m3probe/probe_ft_combined_step.py` | 合併目標(`∇L_IO + κ∇L_FT`,`f ∈ {0.25,1.0}`)的固定 RMS 下降步 → hard `ft_rg` / `io_rg` | **§3.3 主表與 `U` 值(S4a 裁決的核心依據)** |
| `/tmp/m3probe/probe_step_hpwl.py` | 同上步長下的 HPWL 變化 | §3.3 的 Δhpwl 欄 |
| `/tmp/m3probe/probe_gate_mass.py` | 遠區(`D ≥ 2`)的 `q` 質量落在 membership gate 以下的比例與 (e,k) 對數 | §3.4 的表(78.7/60.4/22.4%、80.8/57.3/10.8%) |

**與 M2 資產的關係:** S1(`ioplace/ops/soft_assign.py`)、chunked 四趟結構(`ioplace/ops/io_term.py::_IoFn`)、schedule 與 version invariant(`ioplace/schedules.py`)、secant refresh(`ioplace/dp_hook.py`)、driver callback(`ioplace/drivers/run_placement_io.py:102-172`)全部原樣沿用;M3 的所有新機制都是這五個檔案的**加法**——而且 v2 的 S4a 形式把「加法」縮到**只有一個 `(P',c)` 係數張量**,沒有任何新的常駐累加器。
