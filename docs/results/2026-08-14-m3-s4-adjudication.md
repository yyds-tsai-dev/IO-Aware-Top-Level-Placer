# M3 / S4 選擇裁決書(2026-08-14,依 P0b 預註冊判準)

- 依據:`results/m3/probes/probe_p0b.json`(commit da818f4)、v3.1 design §3.4.3/§3.4.4/§8
- 裁決者:deep-reasoner(Opus),scheduler 採納
- 地位:本文件是 T8 報告的正式輸入;預註冊內的裁決不可再議,post-registration 判斷已逐項標明

**結論(一句話)**:P0b 的「7 個全滅」是**預註冊明文涵蓋的分支**(§3.4.3 line 345 + §8 T0-c line 662),照做即可——M3 不做可微 FT 項;但 P0b 的機械 verdict 掩蓋了一個實質發現:**S4a-star 的單步 FT 效用不在噪聲帶內,它是 4–13σ 的真效應**,把它淘汰的是 T1c(近乎零容忍的 HPWL 帶)與 T1e(連現任者自己都過不了的雙尺檢定)。即便把這兩條完全刪掉,S4a 仍只有 19/30 < 80%,**所以任何判準修訂都翻不了案**——這一點讓 (3) 不必做,也讓 (1) 成為誠實且穩健的主線。

---

## A. 預註冊的字面規則:全滅情境有規定,且已排除 in-loop pilot

**§3.4.4(lines 358–362)只有三個分支**,逐字:

> - tier-1 通過 **且** tier-2 全過 ⇒ **合格**,可直接進 T2 實作。
> - tier-1 通過 **但** tier-2 失敗 ⇒ **受限合格**:只能用 `f_ft_max ≤ 0.25`、`ft-ramp-mode = window`(§5.2),且 in-loop 必須開 **G9**;報告必須列出失敗的 tier-2 條目。
> - tier-1 失敗 ⇒ **淘汰**,不論 tier-2。

§3.4.4 **沒有**「全部淘汰」分支——因為那條規則被寫在別處,而且寫得很清楚。**§3.4.3 line 345**:

> **fallback 動作**:沒有候選通過 tier-1 ⇒ **M3 不做可微 FT 項**,改走「M1 式離散 reweight + FT 診斷」的縮減範圍,並在報告誠實記錄 negative result(沿用 M1/M2 慣例)。

**§8 T0-c 驗收(line 662)**再確認一次:

> **若全部淘汰 ⇒ 走 §3.4.3 的 fallback,M3 範圍縮減**

**這不是預註冊漏洞。** 執行動作 = 走 §3.4.3 fallback。

且**in-loop pilot 被預註冊明文禁止**,不是「未規定」:

- §3.4.2 scope 表(line 304):「檢疫機制 = T6 的 G3′ in-loop gate……且 **P0b 未通過的候選連 in-loop 都進不去**」
- L13(line 437):「P0b 只負責『不該進 in-loop 的候選先淘汰』」

§3.4.4 的「受限合格」是唯一的放寬階梯,而它以 **tier-1 通過**為前提。因此:**任何 in-loop pilot 都不能被描述成 §3.4.4 fallback ladder 的延續**;要做只能宣告為 post-registration 的新實驗(P0c,見 D)。

---

## B. 「tier-2 近全過、tier-1 全零」的診斷

### B1 首先推翻預設:單步 FT 效用**存在**,而且很大

判定家族 M1、低 τ、adaptec1(n=30/候選):

| 候選 | T1a 通過 | 中位 `Δft_mst − Δft_mst(IO-only)` | 以 `3·sd_rand` 為單位 | cos(∇L_IO,∇L_FT) |
|---|---:|---:|---:|---:|
| **S4a-star** | **23/30** | **−199 counts** | **−4.34**(≈13σ) | 0.399 |
| S4b-gated_b0.5 | 2/30 | −58 | −1.29 | 0.085 |
| S4e-union-detach | 1/30 | −48 | −0.96 | **0.947** |
| S4e-union-hardbar | 1/30 | −53 | −1.12 | **0.944** |
| S4b-gated-detach_b1.0 | 3/30 | −23.5 | −0.39 | 0.025 |
| S4g-bboxcov | 0/30 | −12 | −0.31 | −0.001 |

只看 §3.4.1(a) 的 12 個真實 trajectory snapshot(低 τ 的 18 格):**S4a 的 T1a 17/18、T1b 18/18、T1d 18/18**。單步實得 FT 增益 = base `ft_mst` 的 **2.7–8.6%**(如 `m2best_it0550` η=0.01:gain 294 / base 3,427 = 8.58%)。

**隨機帶並沒有吞掉效果**:低 τ 的 30 個 (state,η) 中,`3·sd_rand(Δft_mst)` 中位僅為 IO-only 自身效果的 **37.6%**,而 S4a 的超額是該帶的 4.34 倍。逐格帶在 13–134 之間,S4a 的超額在 66–470。

同時,**IO-only 參照臂在每一個低 τ 格都讓 ft 上升**(+43…+1,054)——M2 報告的 ft +38.5% 在單步解析度上被完整重現。資料說的不是「FT 與 IO 不能同時改善」,而是:**純 IO 方向確實在製造 FT,而按 hop 距離加權的變體(S4a)能把它扳回來。**

### B2 6/7 候選是**機制性死亡**,不是「測不到」

- **S4e(兩個變體)cos = 0.947/0.944 ⇒ 它就是 IO 項本身。** 機制:`L_FT = ReLU(L_cross − base)`,`base` 是 detach 的(§3.3.3 line 225)。detach 後 `∇L_FT = ∇L_cross`(active set 上),而 `L_cross_e = Σ_a U_{e,a}` 正是 soft crossing 計數 = M2 的 IO surrogate。**FT = crossings − Steiner;把第二項 detach 掉,梯度裡就沒有任何 FT-specific 資訊。** S4e 是「無害的同義反覆」。這也連帶結案 L15(tie-break 敏感度對一個 ≡ IO 的項沒有意義)。
- **S4g:cos ≈ −0.001、`‖∇L_FT‖₁/‖∇L_IO‖₁ = 658` ⇒ κ ≈ 3.8e−4**,行為是一個弱 WL 力(中位 HPWL 超額 **−2.22%**),FT 效用 0(T1a 0/30)。⇒ §6.3 的 C5 臂可依 §3.3.4 line 247 直接移除。
- **S4b 家族**:§3.2 的 0 次齊次 + gate 梯度盲區診斷成立。cos 0.025–0.152,FT 超額 ≤1.3 倍帶寬,HPWL 超額 +1.77…+2.18%。

### B3 「tier-2 全過、tier-1 全零」不是弔詭,而是兩者正交

- **S4b-gated-detach_b1.0**:tier-2 最佳(t2a 25/26、t2b 26/26、t2c 26/26,中位 `mass_on_ft0` = 0.194)⇒ **支撐放對了,但那個方向沒用**。
- **S4a-star**:tier-2 最差之一(t2c 僅 3/26;`bucket_share_ratio["4+"] = 7.96`)⇒ **方向有用,但支撐偏向大/遠 net**(已知的 star 偏誤)。

tier-2 量「懲罰落在哪些 net 上」,tier-1 量「往那個方向走有沒有用」。P0b 證明兩者獨立——§3.4.4 把 tier-1 排在前面是對的。

### B4 判準的三個缺陷(記錄;不改變結論)

1. **T1-P3 結構性不可通過(客觀)**:12 個 snapshot 有 6 個是 flat(λ_io=0),M3 家族方向 `g_full = g_wl_density + λ_io·C` 在這 6 格所有臂逐位元相同(JSON 可驗)⇒ T1-P3 的 ≥60%(≥8/12)理論上限 6/12。且 T1a 的絕對條款 `Δft_mst < 0` 在鋪散階段的全 objective 單步必然失敗。**報告不得引用 M3 家族結果當證據。**
2. **T1e 連現任者都不過(客觀)**:IO-only 在 30 個低 τ M1 格中 6 格違反 `sign(Δio_rg)=sign(Δio_mst)`。限縮成 FT 兩尺後 S4a 的 T1e 從 14/30 → 24/30。
3. **T1c 的帶寬是抽樣噪聲尺度而非無差異尺度(判斷)**:HPWL 的 `3·sd_rand` 僅為 IO-only 自身 HPWL 代價的 1.24%,T1c 實際讀作「FT 力必須免費」,與專案接受 HPWL 代價的立場(M1 +1.26%、E3 談收斂後)不一致。

### B5 修好判準也翻不了案(反事實重算,n=30)

| 修法 | S4a 通過數 |
|---|---:|
| 原判準 | 2/30 |
| T1e 只看 FT 兩尺 | 3/30 |
| HPWL 容忍 +0.5 pp | 7/30 |
| 兩者都改 | 12/30 |
| **T1c、T1e 整條刪掉** | **19/30 = 63%** |
| 再限 η∈{0.005,0.01} | 14/20 = 70% |

要到 80% 須再丟 4 個最終 placement 狀態(S4a 在 slicing-final T1a 僅 1/6)——三個事後動作才能救活一個候選 = 沒救活。**判準修訂+重跑不做。**

### B6 協定偏差揭露

1. **T2d 從未裁決**:JSON 無 `t2d` 鍵,`random_bands` 無 chain→star 帶 ⇒ 無法事後補判(verdict 由 tier-1 決定,不影響結論,報告必須記)。
2. **chain→star = 19,653 是全部 1,090 格 × 8 臂總和**;逐臂 S4a 2,945、IO-only 本身 2,049;低 τ M1 的**淨**轉換對 S4a 是 **−152(負)** ⇒ F3 的 star 偏誤在 placement 層級未被證實(只在 surrogate 值域成立,即 t2c)。
3. `failed_cells = 2`(S4g bigblue4 OOM)依 §3.4.0 計未通過;S4g 分母 60。

---

## C. Phase B 重新定義

### 預註冊內(執行,不需新設計決定)

1. **七候選淘汰為最終判定;T2(S4 op)不解鎖**;§6.3 的 C2/C3/C4/C5/C9 臂作廢,T8 表中列 void 附原因。
2. **執行 §3.4.3 建設性條款:「M1 式離散 reweight + FT 診斷」**——把既有 IO 項的 `w_e` 從靜態改為每 50-iter callback 由 evaluator 刷新的 per-net 向量,權重來源 = T1 已交付的 `ft_rg`/topology。`io_term.py` 的 `IoTermRef`/`IoTerm` 已帶 per-net `w`;改成可刷新 buffer 是**係數改動**:不新增梯度路徑、不新增常駐狀態、無 τ→0 極限問題、無支撐疑慮。既有機具:`ioplace/reweight.py`、`run_placement_reweight.py`。
3. **T11(maze 校驗)無條件在 T8 之前**——縮編後更依賴 `ft_rg` 報表,T11 是其可信度的唯一機制。
4. **補 `ft_rg` 的 σ_seed**(§6.1):用 T1 evaluator 重評 `results/m2/noise/flat_seed100*.npz` 五顆,不需新 placement。
5. **T8 對 phase-1 §9 M3 exit 逐條判**:「FT 顯著下降」在 reweight 交出結果前記 FAIL;「完整 ablation 表」以縮編形式 PASS(S4 臂 void 附原因)。E1/E5/E6 FAIL、E2/E3/E4 N/A、E7 PASS,照 M1/M2 慣例誠實記錄。

### Post-registration(標明;預算受限時不做)

6. **可選 P0c:S4a-star in-loop pilot**(規格見 D)。條件:獨立預註冊先 commit;不改變 P0b verdict;排在第 2 項之後(共用 T4 driver 工作)。**若預算只夠一條路:做第 2 項**(預註冊內、有 M1 實證、直接打 M2 的 ft 回歸)。

**為何不是純 negative result**:那會把「6 個機制性死亡 + 1 個有 13σ 效應但被兩條有瑕疵附帶條件擋下」寫成「可微 FT 沒用」——不準確;且 §3.4.3 fallback 自己規定要繼續做離散 reweight。

---

## D. P0c pilot 規格(若執行;post-registration)

**唯一候選 S4a-star**(其餘六個機制性死亡,無排名依據)。case:adaptec1 k16 grid,det=1、seed=1000、ρ_max=0.40、annealed τ。操作限制借用「受限合格」配置(報告註明是借用):`--ft-ramp-mode window`、τ_start=0.12/τ_full=0.05、G9 開。

臂集合(4 臂 ×~2 分 GPU):P0 `--f-ft-max 0 --callback-order atomic`(G3′ 對照)、P1 0.10、P2 0.25(P0b 唯一實測 f)、P3 0.50(out-of-envelope,標明);P3 相對 P2 單調改善且未破 PG2 時追加 P4 1.0。

Gate(機械):**PG1** = 存在 f>0 臂使 `ft_mst ≤ ft_mst(P0) − 62.9`(3σ_seed);**PG2** = 同臂 `io_mst ≤ io_mst(P0)+492` 且 `hpwl ≤ hpwl(P0)×1.005`(screening 形式,不蘊含 E3);**PG3** = G9 淨轉換報告(PG1 過則 inert;PG1 敗且淨轉換正 ⇒「S4a 把鏈擠成星」結案,不得切 S4e);**PG4** = G8 穩定性 + `cancellation_ratio ≥ 0.3`。

停止規則:全部 f>0 臂 PG1 失敗 ⇒ M3 內不再有 S4 工作。PG1∧PG2 通過 ⇒ 升級 confirmation(seeds 1001–1003、paired 單側 95%、t=2.920、同 seed M2-best 對照)。

成本:S4a 是最便宜的 T2(靜態整數係數、零常駐狀態、不新增 chunk pass);T3 已交付;T4 與 reweight 路線共用。執行前測試鎖:κ_ft=0 逐位元短路、fp64 gradcheck、chunked=unchunked<1e−10、filler 梯度恆 0、τ→0 收斂 S4a hard 極限、legacy+f_ft_max=0 與 M2 存檔逐位元同。

---

## 給 T8 報告的三句話版本

1. 依 §3.4.3/§8 的預註冊規則,7 個 S4 候選全部淘汰,M3 不把可微 FT 項放進 objective。
2. 淘汰的機制可陳述:`ReLU(soft-crossings − detached-baseline)` 家族在梯度上恆等於 IO 項本身(S4e cos=0.947 直接量測);唯一有實質 FT 方向的 S4a-star(低 τ 17/18、4–13σ)在支撐檢定(t2c 3/26)與 HPWL 附帶代價上不合格。
3. 三個判準缺陷(T1-P3 結構性不可通過、T1e 連現任者都違反、T1c 帶寬過窄)已記錄;全部修復後 S4a 仍 63% < 80%,故不重跑。
