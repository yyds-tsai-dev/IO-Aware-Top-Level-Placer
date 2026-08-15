# M4 T6 holdout FAIL 裁決(2026-08-15)——維持 FAIL、選定 N2、T6B 可執行協定與 T7 配方

- 依據:`results/m4/bench/verify_group2x2_n1.json`、`verify_group2x2_n2.json`、`t6_summary.json`、`cluster_stats.json`(commit 1b3671c);T6 協定 v2 = `docs/results/2026-08-14-m4-t6-adjudication.md`;M4 design v2.1 §3.2/§3.3/§7.1/§9
- 裁決者:deep-reasoner(Opus),scheduler 採納
- 地位:預註冊內的裁決不可再議;**本文不修改任何既有門檻、不重掃任何參數、不重評任何已評過的指標**。所有新門檻只作用於**尚未量測**的 artifact,並在量測前登錄
- 探針(本輪新做,待遷入 `ioplace/diagnostics/probes_m4/`):N2 於非齊次陣列的兩種推廣、核敏感度、Sinkhorn 收斂、來源 self-loop 掃描、per-level (B,T) 對照(源自 `/tmp/rent_synthetic_result.json`、`/tmp/rent_cluster_result.json`,已搶救至 `results/m4/bench/rescue/`,待正式遷入,見 §8-1)

## 結論(前置)

**FAIL 維持,一個字都不改。** 但 T6 的資料裡有一個被 verdict 掩蓋的實質結果:**H6 已經完成了它被設計來做的事——它證偽了 N1、留下了 N2**,只是它的 `1×1` 那條腿把「換形狀時每 tile 端子是否守恆」跟「加了 glue 之後 p 會不會變」混在一起,而後者**在真實世界裡本來就不是 0**:真實 `mempool_group` standalone p=0.4747、真實 `mempool_cluster` p=0.5221,真實的階層躍升就是 **+0.0474**。合成 N2 的躍升是 +0.0515(差 0.0041),N1 是 +0.0576(差 0.0102);而唯一乾淨的那條腿(1×2 vs 2×2)N2 給 **0.0042**、N1 給 **0.0412**(門檻 0.03)。

同時,H1 的 fail 完全來自 `[0.55,0.80]` 這個區間——它是從**另一把尺**(group 端子計數 Rent p≈0.633)搬到 **mtkahypar 遞迴二分**這把尺上的,而真實參照自己就是 0.5221,落在區間外。⇒ 這條腿對「生成器好不好」的鑑別力是 **0**:任何與真實完全吻合的合成品都會 fail。這是 gate 設計缺陷,**必須揭露、不得據以翻案**。

三個決定:**(1) N2 為 30M 唯一正規化**;**(2) T7 走 recipe A′(MoM,α=0.102,N2),不切 recipe B**——切過去不會把任何一個紅燈變綠,只會把一個有實測支撐的 glue 換成一個純代數猜測;**(3) T6B 落成一份只評「尚未量過的 artifact」的清單**,其 Rent 條目改為來源相對的自洽性判準(T6B 全篇本來就是「對照來源 group 而非 cluster」,只有 Rent 那一格漏改)。

## 裁決一覽表

| # | 問題 | 裁決 | 一句話理由 |
|---|---|---|---|
| a-1 | T6 verdict | **FAIL 維持;`t7_gate=FAIL`、`selected_normalization` 由 null 改記 `n2`(見 d)不改 verdict** | 預註冊的門檻是門檻;`undecidable` 也不是綠,M4-G3 一樣觸發,標 FAIL 更保守 |
| a-2 | H1 區間是否可重判 | **不可。**記 FAIL,另立 **B5** 條目揭露「區間換尺」缺陷 | 改的是**決策規則**不是**量測值**,與 Bug A/B(算錯了量)不同類;且改動方向恰好對己方有利 |
| a-3 | 是否算「量測方法問題」 | **是,可揭露;但揭露不改結論** | 兩者可以並存,而且必須並存 |
| b | T6B 協定 | **落成 §4 的 B-0…B-6 七條**;Rent 改「來源相對 + 形狀不變性 + 零 glue 對照」的 DiD,沿用 H6 原門檻 0.03 逐字 | T6B 全篇的設計原則就是對照來源 group;`[0.55,0.80]` 那格是漏改,且對來源自己(0.4747)也會誤判 |
| c | T7 glue 配方 | **只走 A′ + N2 + α=0.102 + Sinkhorn 推廣**;另產 K3 全陣列作明示 locality 下界;α∈{0,1.915} 只出 recipe/sha256。**T4B 記 `not_executed`(觸發條件未成立)** | recipe B 修不了任何一個紅燈,且其 p∈{0.60,0.65,0.70} 與本設計實測 p(0.47–0.52)矛盾 |
| d | N1 vs N2 | **N2**,且證據比 T6 前更強 | §2(d) 的 Rent 論證存活,並被 H6 唯一乾淨的那條腿實測背書(0.0042 vs 0.0412) |
| e-1 | T6b count freeze | **照常進行**,用 N2 臂;N1 計數併列作 rejected-alternative | manifest 計數是檔案事實,`matches_manifest=true`,不受 verdict 影響 |
| e-2 | T8 合成臂 | **照常跑**,但每筆結果加 `generator_verified=false` 等 provenance 欄位,並由 linter 強制 | 合成表本來就禁品質欄位;現在再加一層「未認證」標記 |

---

## 1. (a) verdict 成立——H1 的失敗機制與正確處置

### 1.1 事實

`verify_bench.py:188-198` 的 H1 是兩條腿的 AND:

- 相對腿 `|p_syn − p_real| ≤ 0.05`:**n1 = 0.0102、n2 = 0.0041,兩邊都過**;
- 絕對腿 `p_syn ∈ [0.55,0.80] 且 p_real ∈ [0.55,0.80]`(`verify_bench.py:195` 的 `band[0] <= p_real <= band[1]`):**p_real = 0.5221 就在區間外**,p_syn = 0.5323/0.5263 也在區間外。

區間出處是 spec §3.3 H1 列(design draft line 323),其依據是 T6 協定 v2 line 19 量到的 **group 端子 Rent p = 0.6305–0.6352**——那是**四個 group 的端子數 vs 規模**的四點擬合,不是 `rent.py` 的 Landman–Russo 遞迴二分。同一份協定 v2 的 line 36 自己寫了「同報告同估計器硬規則」,line 32 又預測「真實端預期 p≈0.633」。**預測錯了(實測 0.5221),而區間是照著錯預測畫的。**

### 1.2 裁決:FAIL 維持

三個理由,缺一不可:

1. **改的東西的性質不同。** T3a 的 Bug A/B 是「算錯了被量的量」——修好後同一個預註冊統計量得到不同的值,門檻沒動。H1 的兩個 p 都算對了(同一把尺、`max_net_degree=100` 兩邊同時套用,`t6_summary.json:mtkahypar_feasibility.cap_applied_both_sides=true`),錯的是**接受域**。看到結果之後改接受域,正是預註冊存在的唯一理由所要禁止的事,而且此處的修改方向**恰好**把 fail 變成 pass——任何具有這個性質的事後修訂一律不受理。
2. **改判為 `undecidable`/`void` 沒有任何好處。** M4-G3 的觸發條件是「任一不綠」(design draft line 698),`undecidable` 同樣不綠,fallback 一樣觸發;而「看到結果後宣告某個 gate 無效」是一個永遠對實驗者有利的裁量權。資訊完全由 §7 的揭露文字保存,標 FAIL 是嚴格更保守的選項。
3. **裁決者是同一人。** 區間是上一輪的裁決搬過去的,預測 0.633 也是同一輪寫的。把自己的錯誤預測的後果抹掉,和把別人的抹掉一樣不可接受。

### 1.3 但必須揭露的機制(這屬於「量測方法問題」,可揭露)

**H1 絕對腿的鑑別力為零。** 真實參照自己就在區間外 ⇒ 一個與真實 cluster 完全一致(`|Δp|=0`)的合成品也會 fail;一個與真實差 0.049 的合成品也是 fail。這條腿在本次評測中不含任何關於生成器的資訊。這句話必須逐字進報告(§7-2)。

**且此缺陷不外溢。** `cluster_stats.json:lambda_0_tile` 的尺度換算用了 `rent_p_used = 0.6326`(同一把「錯尺」)把 λ₀=4155.25 換成 λ₀_tile=4387.31。若改用 mtkahypar 的 0.4747,λ₀_tile = 4155.25×1.0897^0.4747 = **4328.2**,差 **1.36%** ⇒ 估計器混用對 glue 量的影響是二階量,不必也不得回頭重算(重算即等於重掃)。這一點要寫進報告,免得讀者以為整條校準鏈都被污染。

---

## 2. H6 的失敗機制——本輪最實質的發現

### 2.1 數字

| 陣列 | glue nets | glue 端點/tile | p | 95% CI | Δp vs 1×1 |
|---|---|---|---|---|---|
| 1×1 source(無 glue) | 0 | 0 | 0.474749 | [0.4481, 0.4928] | — |
| 1×2 N1 | 4,413 | 4,413 | 0.491107 | [0.4783, 0.5059] | 0.01636 |
| 1×2 N2 | 13,207 | 13,207 | 0.530429 | [0.5175, 0.5412] | 0.05568 |
| 2×2 N1 | 26,034 | 13,017 | 0.532342 | [0.5247, 0.5401] | 0.05759 |
| 2×2 N2 | 26,341 | 13,171 | 0.526262 | [0.5161, 0.5339] | 0.05151 |
| **真實 cluster** | — | (真實跨 group ≈12,321/group) | **0.522124** | [0.5001, 0.5459] | **0.04737** |

H6 逐對:N1 = {1×1:1×2 = 0.0164, 1×1:2×2 = **0.0576**, 1×2:2×2 = **0.0412**};N2 = {1×1:1×2 = **0.0557**, 1×1:2×2 = **0.0515**, 1×2:2×2 = **0.0042**}。

### 2.2 機制(已用 per-level (B,T) 表直接驗證,不是推論)

glue 是**均勻隨機取端點的 degree-2 長程 net**(`build_t6_arrays.py` 的 `known_simplification`:"uniform ... candidate interface-cell sampling")。一個大小 B 的 block 抓到 glue 端點的機率是 `B/N_tile`,而它的對端幾乎必然在 block 外 ⇒ glue 對 block 端子數的貢獻是 **ΔT(B) = ε·B**(指數 1),而不是 base 的 `B^0.47`。實測(相同 B 逐層對照,B 完全對齊):

| B | T(1×1 無 glue) | T(2×2 N2) | 實測 ΔT | 預測 ε·B |
|---|---|---|---|---|
| 772,353 | 6,610 | 9,654 | 3,043 | 3,293 |
| 96,544 | 2,219 | 2,571 | 352 | 412 |
| 12,068 | 827 | 866 | 39 | 51 |
| 1,509 | 349 | 356 | 6 | 6 |

比值 0.7–1.1 橫跨三個數量級,四個陣列全部成立。因此 fit window 頂端端子被墊高 46%、底端只墊高 2% ⇒ OLS 斜率必然上移 ≈ ln(1.46/1.02)/ln(772353/1509) = **0.058**,與實測 0.0515 相符。**Δp 只是 glue 端點數/tile 的函數**:四點的 Δp/端點數 = 3.71/4.22/4.42/3.91 ×10⁻⁶(均值 4.06,±9%,這個 ±9% 同時就是 mtkahypar 種子 + 網表差異的噪聲底,約 ±0.003 的 p)。

### 2.3 H6 證偽了什麼

- **`1×2 vs 2×2`(唯一乾淨的腿)**:兩者的每 tile glue 端點在 N2 下相同(13,207 vs 13,171)、在 N1 下差 3 倍(4,413 vs 13,017)。實測 N2 = 0.0042(過)、N1 = 0.0412(不過)。**這正是 H6 被設計來測的東西(協定 v2 line 37:「直接證偽 glue 正規化選擇」),而它成功地證偽了 N1。**
- **含 `1×1` 的兩條腿**:1×1 是**零 glue** 的設計,而陣列有 glue;真實世界對同一個躍升給的答案不是 0 而是 **+0.0474**(group→cluster)。以這個實測的 null 來看,N2 的偏差 0.0041、N1 的 0.0102,兩者都 ≤ 0.03。⇒ **這兩條腿在 null = 0 的設定下對兩種正規化一視同仁地誤判**,不含鑑別 N1/N2 的資訊。

**紀律聲明:以上分析不改變 T6 的 verdict,只用於 (i) 報告揭露文字,(ii) 為尚未量測的 artifact 設計判準。**

### 2.4 順帶得到的最強正面證據(必須進報告,因為它不是套套邏輯)

把 glue 拿掉之後,合成陣列與真實 cluster 的 per-level cut 相對誤差在 window 頂端三層是 **25% / 16% / 14%**(level 4/5/6)——正好卡在 H2 的 25% 門檻上;加上 N2 的校準 glue 之後降到 **9.2% / 5.9% / 5.3%**(N1 為 12.6% / 9.6% / 8.3%)。而 H2 比較的兩邊 B 差 9.24%(合成 772,353 vs 真實 707,006,因為真實 group 比 standalone 小 8%);以 `T ∝ B^0.522` 修正尺度後,level 4 的誤差進一步降到 **4.3%**。

⇒ **glue 的「量」是被真實資料需要的,而且校準出來的量是對的。** 這是 T6 資料集裡唯一一個既非建構恆等式、又非來源繼承的正面結果,證據強度排序必須誠實地寫成:

- **強**(結構性、非套套邏輯):H2 per-level cut(N2 最大 9.2%)、H1 相對腿(0.0041)、H6 的 `1×2 vs 2×2` 腿(0.0042);
- **弱**(近乎建構恆等式):H3(合成與真實共用同一個 `mempool_group` 拓撲,KS=0.013 幾乎必然)、H4(協定 v2 line 33 已降級為對「來源 group 自身」的自洽性,KS=0.011 是抽樣方式的自我確認)。

---

## 3. (d) N1 vs N2——選 N2,論證比 T6 前更強

協定 v2 §2(d) 的先驗論證(N1 讓每 tile 端子隨 m 成長、違反 Rent)**完全存活**,因為它是關於**建構法**的論證,不是關於 T6 資料的論證;T6 的失敗沒有攻擊到它的任何前提。而現在它多了三份實測背書:

1. H6 唯一乾淨的腿:N2 形狀不變(0.0042),N1 隨形狀漂移(0.0412),方向與幅度都與先驗預測一致;
2. H1 相對腿:N2 (0.0041) 比 N1 (0.0102) 更貼近真實 cluster;H2:N2 (9.2%) 優於 N1 (12.6%),逐層全面較優;
3. 外推發散:3×3 下 N1 = 151,259 nets(每 tile 端點 33,613)、N2 = 59,229(每 tile 13,162)。以 §2.2 的機制外推,N1 的 3×3 會把 p 推到 ≈ **0.61**,遠離任何真實參照;N2 停在 ≈ 0.53。

**⇒ `selected_normalization = "n2"`,理由記為「T6 未通過 ⇒ 選擇依據為 §2(d) 的先驗 Rent 論證 + H6 形狀不變腿的實測」,而不是「H6 通過」。**

---

## 4. (b) T6B 可執行清單

### 4.0 T6B 的重新界定(這是 spec 沒寫的 case,在此定案)

spec 把 T4B/T6B 綁成一組 fallback,觸發條件是 **T3a 不過 ⇒ 沒有 ground truth**。實際情形是 **T3a `PASS_2x2`(commit ea91dc2)、ground truth 存在且已用上,但 holdout 未全綠**。兩者的失效模式不同:

- T4B 的存在理由是「拿不到 cluster 統計」⇒ **觸發條件未成立,T4B 記 `not_executed`**(理由與證據寫進報告,見 §5.3);
- T6B 的存在理由是「生成器未取得認證,仍需一條可用的驗收路徑」⇒ **觸發條件成立,執行**。

**T6B 的唯一評測原則(預先登錄):一個預註冊指標,每個 artifact 只評一次。** T6 已在 1×2/2×2 上評過 H1/H2/H3/H4/H6,**一律不得重評**(重評 = 第二次機會 = 與重掃同罪)。T6B 只評:(i) 從未評過的指標(H5),(ii) 在**新 artifact**(3×3、零 glue 對照陣列)上的指標。

### 4.1 清單

所有項目一律:normalization = **N2**、kernel = K1 α=0.10230633302323285、λ₀_tile = 4387.31377772961、primary seed = 0、Rent 一律 `backend="mtkahypar"`, `threads=16`, `b_lo=1e3`, `b_hi=1e6`, `n_bootstrap=200`, `max_net_degree=100`(兩邊同套)。

| # | 指標 | 對照對象 | 門檻(全部沿用既有預註冊數字,無新數) | artifact | 不過的後果 |
|---|---|---|---|---|---|
| **B-0** | V0′ 結構完整性 | 來源 `mempool_group` 自身 | 見 §4.2 的重述版,**全部 true** | 3×3 + 3×3-noglue | 建構 bug ⇒ 修 tiler 後重建(這是 bug 修復,不是重掃) |
| **B-1** | Rent 形狀不變性(DiD) | 自身零 glue 對照 | `\|Δ_glue(3×3) − Δ_glue(2×2)\| ≤ 0.03`,其中 `Δ_glue(s) = p(s, N2 glue) − p(s, 無 glue)` | 3×3、3×3-noglue、2×2-noglue(2×2-glue 用已在檔的 0.5262620) | 見 §4.3 |
| **B-2** | H5 K-grid λ 口徑 | 來源 group,**等 per-tile 解析度** | `ratio ∈ [0.7, 1.4]`(spec §3.3 H5 原值) | 2×2(K=16 vs 來源 K=4) | 標 fail 並揭露;不阻擋 scaling 用途 |
| **B-3** | H3′ 全網 degree KS | 來源 group | KS ≤ 0.05 且逐桶 ≤ 20%(原值) | 3×3 | 建構 bug 才會不過 |
| **B-4** | H4′ interface-cell 自洽 | 3×3 自身的均勻參照樣本 | KS ≤ 0.10(原值) | 3×3 | 同上 |
| **B-5** | H5′ 算術自檢 | 建構公式 | rel_err ≤ 1e-3(原值) | 3×3 + 三個變體 | 生成器 bug |
| **B-6** | 每 tile 端子預算恆等式 | N2 定義 | 每個 tile 的 glue 端點期望值與 `B = 3λ₀_tile = 13,161.94` 的相對差 ≤ 1e-9 | 3×3 全部四個核 | Sinkhorn 未收斂 ⇒ 實作 bug |
| — | H1 / H2 | — | **`not_applicable`(27.7M 無同尺度真實對照)**,不得標綠 | 3×3 | — |
| — | H1 / H2 / H3 / H4 / H6 於 1×2、2×2 | — | **`already_evaluated_T6`,結果凍結,不得重評** | — | — |
| — | 「三個 p 並列取中位數」 | — | **`not_applicable`(recipe B 未產出)** | — | — |
| — | V1 DREAMPlace 可讀 | — | **`infrastructure_blocked`**(`rent.py:23-44`:`PlaceDB.read_pl` 的 `\w+` regex 無法匹配階層名,在 `mempool_group` 自身上即已重現,與 tiler 無關) | — | — |

**驗收:B-0 與 B-3…B-6 全綠 ⇒ 3×3 bench 可用於 scaling(且永久不得用於品質宣稱)。B-1、B-2 的結果一律發表,其判定寫入報告但不阻擋 scaling 用途(理由見 §4.3)。**

### 4.2 B-0:V0′ 的重述(這是修正已知的判準 bug,不是放寬門檻)

`verify_bench.check_v0_structural` 現行版對本專案的真實衍生網表有三條**必然誤判**的子檢查,2×2 的 1,338 條 error 全部出自它們:

1. **`no_self_loops`**:2×2 報 693,952 條「self-loop」。實測來源 `mempool_group.nets` 自身就有 **173,488** 條(4.951%),`693,952 = 4 × 173,488` **完全等於複製數** ⇒ tiler 一條都沒新增。「一條 net 接到同一個 cell 的兩隻腳」在真實網表裡合法且常見。⇒ 改為 **`self_loop_fraction_not_worse_than_source`**,並把兩邊的數字都寫進輸出。
2. **`rows_no_overlap_no_gap`**:`tile_bookshelf.py:259-276` 在 T6 建構期間就已經查明並記錄:含 macro 的 floorplan 在 macro 覆蓋處**合法地沒有 row**,spec §3.2 的字面「無縫隙」對任何含 macro 的設計都是不可滿足的,tiler 自己的 assert 因此已改成**只查 overlap**(`tile_bookshelf.py:285`)。`verify_bench.py:151-164` 沒同步。⇒ 改為**只查 overlap**,與 tiler 同語意。**這是把兩處已經分歧的實作對齊到先於量測就已寫定的語意,不是事後放寬。**
3. **`all_nodes_referenced_or_declared_unconnected`**:2×2 報 8 個 ⇒ 來源 2 個 × 4。⇒ 改為 **`unreferenced_count == source_count × R × C`**。

另有一條**規模阻擋**:`check_v0_structural` 用 `nets = list(_iter_nets(...))` 把整份網表(3×3 = 108M pins)持有在 Python list 中,2×2(48M pins)勉強跑完,27.7M 會吃掉數十至上百 GB host RAM。⇒ **B-0 執行前必須先把 `check_v0_structural` 改成單趟串流**(與 `append_glue_nets` 的 M4 T6 scale fix 同形)。

### 4.3 B-1 的設計理由、預先登錄的預測與失敗後果

**為什麼要零 glue 對照:** 3×3 的 9 個 tile 無法被二分法對半切成整數個 tile,前幾層必然把 tile 切開,這會系統性墊高低層的端子數——一個與 glue 無關的組合學假象。用 DiD 把它消掉。(補充:3×3 的 fit window 從 level 5 起(B=869k),前四層都在 `b_hi=1e6` 之外,所以這個假象只影響被排除的層;DiD 是保險,不是必需——但保險很便宜。)

**預先登錄的可證偽預測(在量測之前寫下):** N2 使 3×3 的每 tile glue 端點恰為 13,162,與 2×2 的 13,171 相同 ⇒ 由 §2.2 的機制,

> `p(3×3, N2, seed 0) = 0.530 ± 0.008`,`|p(3×3) − p(2×2)| ≈ 0.003`,`Δ_glue(3×3) ≈ 0.053`。
> 若當初選 N1,對應值會是 `p ≈ 0.61`、`|Δ| ≈ 0.08`。

若實測落在 0.530±0.008 之外,§2.2 的機制模型即被證偽,報告必須發表這件事並重新檢討 N2 的選擇。

**可行性與預先登錄的中止條件:** 2×2(12.36M nodes / 14.04M nets)的 `measure_rent` 實測 **1,210 s**、真實 cluster **1,220 s**;3×3 為 2.25× ⇒ 預估 45–60 min/run,四個 run(3×3 glue、3×3 noglue、2×2 noglue、+ 1 個備援)約 3 h。本機 125 GB RAM(可用 110 GB)。**預先登錄:單 run 上限 4 h wall / 100 GB RSS;超出即記 `not_evaluable_scale`,不得改判為 fail,也不得降級估計器**(spec §3.3 M4-L4 的取樣子超圖 fallback 會換掉估計器,與 2×2 的 p 不可比,禁止用於 B-1)。

**B-1 不過的後果(預先登錄):** 3×3 **仍可用於 scaling**(nets/pins/記憶體/runtime 帳與 Rent 無關),但報告必須加記「3×3 在 27.7M 尺度上與真實設計的結構相似性未獲驗證,Rent 形狀不變性檢定值 = X」。品質宣稱本來就已永久禁止,不因此再變。

### 4.4 B-2 的解析度匹配(在量測之前定義,因此屬於合法的預先登錄)

H5 從未被評測過(`verify_group2x2_*.json` 裡沒有這個欄位;協定 v2 line 35 把它降為診斷不判定)。因此本文有完全的自由**在量測前**把比較口徑定死:

- `hard_lambda_sum / n_nets` 對格點解析度極度敏感。同樣用 K=16 去量「一個 tile 的設計」和「四個 tile 的陣列」,後者的每 tile 解析度只有前者的 1/4 ⇒ 必然 `syn < real`(協定 v2 line 34 已預告)。
- ⇒ **判定用等 per-tile 解析度**:2×2 陣列 `make_grid_regions(die, 4, 4)`(K=16,每 tile 2×2 region)vs 來源 group `make_grid_regions(die, 2, 2)`(K=4,每 tile 2×2 region)。門檻沿用 `[0.7,1.4]` 逐字。
- 另**併列報告**未匹配口徑(兩邊都 K=16)作為診斷,以記錄該效應的大小。
- **3×3 記 `not_applicable_resolution`**:等解析度需要 `K = 9 × k_tile ≥ 36`,而 `evaluator_gpu.py:141` 與 `region_graph.py:60` 都有 `assert k <= 32` 的硬上限。理由是算術事實,不是選擇。

---

## 5. (c) T7 的確切生成參數與 provenance

### 5.1 裁決:只走 A′,不切 recipe B

四個理由:

1. **切過去修不了任何一個紅燈。** H1 的絕對腿是被**真實參照**弄壞的,換 glue 配方完全無關(而且在 T6B 下 H1 本來就 `not_applicable`);H6 的 1×1 腿是「加了 glue」造成的,recipe B 也加 glue(3×3 約 25.5k–30.9k nets,每 tile 端點 5.7k–6.9k ⇒ 依 §2.2 的機制 Δp ≈ 0.023–0.028,**依然貼著 0.03 門檻**)。換配方只是換一個一樣紅的燈。
2. **recipe B 的 `p` 與本設計實測矛盾。** 它規定 `p ∈ {0.60,0.65,0.70}`,而這族設計用同一把 mtkahypar 尺量到的 p 是 **0.4747(group)/ 0.5221(cluster)**。用一個與被建模對象差 0.13–0.23 的指數去算跨 tile net 數,是把一個已知錯的假設寫進 benchmark。(參考數字:以 `T(g)·(m − m^p)/2`、`T(g) = 11,742` 計,3×3 得 30,898 / 28,350 / 25,506;若改用實測 p=0.4747 則為 36,178。)
3. **它會嚴格降低 bench 的保真度。** A′ 的 λ₀ 來自實測六個 pair 計數(3,794–4,566,Bug B 修正後),而 §2.4 已證明**這個量是真實 per-level cut 所需要的**:拿掉 glue,H2 在 window 頂端就退到 25%/16%/14%。recipe B 的 25.5k–30.9k 只有 A′ 的 43–52%,會把已經對上的那一項弄壞。
4. **它對 30M bench 的唯一用途(scaling)毫無影響。** 全部候選的 glue 都在 31.5M base nets 的 **0.08%–0.19%** 之間,pin 側 0.05%–0.11%。差異在記憶體/runtime 帳上是第三位小數。

⇒ **T4B 記 `not_executed`**,理由與上述四點併入報告;recipe B 的三個數字以**解析式**列在報告的敏感度段落(不產檔案)。

### 5.2 N2 在 3×3 的推廣(必須裁決,因為現行實作在 3×3 直接拋例外)

`glue_gen.n2_pair_counts`(`glue_gen.py:170-203`)對每 tile 鄰居距離多重集不同的形狀(3×3 的角/邊/中心)**明文 `raise NotImplementedError`**(`glue_gen.py:196-200`)。兩個候選:

- **(A) 兩端點平均的閉式**:`c(u,v) = (B/2)(φ/Z_u + φ/Z_v)`。總數恰為 `m·B/2`,但每 tile 端子不再恰為 B:α=0.102 時角/邊/中心 = 13,073 / 13,203 / 13,355(±1.5%);**α=1.915 時 11,739 / 13,898 / 15,909(±21%)**。
- **(B) 對稱 Sinkhorn 縮放**:求 `a_u > 0` 使 `c(u,v) = a_u a_v φ(d_uv)` 滿足 `Σ_{v≠u} c(u,v) = B` 對每個 u 成立。每 tile 端子**恰為 B**,總數同為 `m·B/2`。

**裁決:(B)。** 決定性理由:N2 相對 N1 的**全部**正當性就是「每 tile 端子是 tile 的性質、與形狀/鄰居數無關」;(A) 自己破壞這個性質,而且破壞量隨 α 增大——而 α∈{0,0.102,1.0,1.915} 的敏感度掃描正是 T7 的交付物,用 (A) 會讓「核形狀」與「每 tile 預算被違反」再度混淆,那正是 H6 剛剛懲罰過的錯誤。(B) 在 1×C/R×1/2×2 上與現行公式**逐位元等價**(所有 Z_u 相等 ⇒ 所有 a_u 相等 ⇒ `c = Bφ/Z`),因此已建好的 1×2/2×2 陣列與新規則一致——這一條要寫成單元測試。

**執行細節(全部預先登錄):** Gauss–Seidel 就地迭代 `a_u ← B / Σ_{v≠u} a_v w_uv`,tile 以 row-major 固定順序;收斂判準 `max_u |Σ_v c(u,v) − B| / B < 1e-9`,上限 1000 次。實測收斂次數:K1(α=0.102)**21**、α=1.915 **19**、α=0 **21**、K3 **72**。manifest 記錄 `n2_rule="sinkhorn"`、`n2_iters`、`n2_max_rel_dev`。

### 5.3 T7 的確切參數

```
source          = results/m4/bench/mempool_group_export/mempool_group
                  (sha256: nodes 36613f1a…, nets 42c5a9d6…, pl 82dd06e3…,
                   scl 08b84fe8…, wts bd900238… — 逐字取自現有 manifest)
shape           = 3x3   (R=3, C=3)
base counts     = nodes 27,804,699 / nets 31,535,892 / pins 108,235,719
                  / rows 290,151 / terminals 105,678        (= 9 × 來源)
normalization   = n2  (rule = symmetric Sinkhorn, budget_pairs = 3.0,
                       B = 3 × λ₀_tile = 13,161.9413)
λ₀_tile         = 4387.31377772961          α = 0.10230633302323285
kernel(主線)    = K1 power_law φ(d) = d^(−α)
期望 glue       = 59,228.74 nets → 118,457 pins  (= m·B/2,對所有核恆等)
sampling        = per-pair Poisson,primary seed 0(全檔落地)
                  + seeds 1–4 只存 recipe sha256(沿用 T6 的
                  `seed_reproduction_trajectory_sha256` 慣例)
合計(期望)     = nets 31,595,121 / pins 108,354,177
                  glue 佔比 nets 0.188% / pins 0.109%
```

**變體(§3.2 R8 的核不可識別性補償):**

| 變體 | 產出形式 | 3×3 glue 總數 | d=1 佔比 |
|---|---|---|---|
| **K1 α=0.1023(主線)** | **全陣列落地** | 59,228.7 | 34.6% |
| **K3 截斷(φ=0 for d>√2)** | **全陣列落地**(明示「不模擬 long-range」的 locality 下界) | 59,228.7 | 74.3% |
| K1 α=0(平坦) | recipe + sha256 + pair 表 | 59,228.7 | 33.3% |
| K1 α=1.915(樂觀端) | recipe + sha256 + pair 表 | 59,228.7 | 58.9% |
| K2 exp(α′=0.08558) | recipe + pair 表 | 59,228.7 | 34.8% |

**必須寫進報告的結構性發現:** 在 N2 之下,**核的選擇完全不改變 glue 總數**(每 tile 預算固定,核只重新分配),只改變**空間局部性**。⇒ spec §3.2 line 304 的「三核總數全距 > 20% 才附 K3 變體」在 N2 下**恆為 0%,已退化為無資訊觸發條件**;敏感度必須改報在**距離層級分佈**上(d=1 佔比 33.3%→74.3%,>2×)。**K3 全陣列照產**(增加交付物永遠可以,減少不行)。

### 5.4 Provenance 標籤(每個 T7 產物的 manifest 與每筆下游結果都要帶)

```
benchmark_kind              = "synthetic"
glue_provenance             = "recipe_A_prime_uncertified"
generator_verified          = false
t6_verdict                  = "FAIL"
t6_fail_mechanism           = ["H1_band_estimator_mismatch",
                               "H6_1x1_leg_zero_glue_baseline"]
normalization               = "n2"
n2_rule                     = "sinkhorn"
quality_claims_prohibited   = true
```

刻意**不用** `"recipe_B_fitted"`(那個標籤的語意是「連通性是擬合出來的」,與事實不符,且 `verify_bench.py` 會據它關閉所有 cluster 對照——我們的 cluster 對照是真的做過的,只是沒全綠)。新標籤 `recipe_A_prime_uncertified` 的語意是「來自真實 cluster 統計,但生成器未通過預註冊驗收」。

---

## 6. (e) 對 T6b(count freeze)與 T8 合成臂的影響

### 6.1 T6b:照常,用 N2 臂

verdict 不影響計數:manifest 的 `matches_manifest` 為 true,header 與 body 逐條對上(`verify_group2x2_*.json:diagnostics_not_gating.V0_structural.checks`)。填入 §2.1:

| 級 | 陣列 | g(nets) | gp(pins) | 佔 base |
|---|---|---|---|---|
| 6.2M | 1×2 N2 | **13,207** | **26,414** | 0.188% / 0.110% |
| 12.3M | 2×2 N2 | **26,341** | **52,682** | 0.188% / 0.110% |
| 27.7M | 3×3 N2 | 59,229(期望,T7b 回填實數) | 118,457 | 0.188% / 0.109% |

⇒ **M4-L12 解除,修正量 < 0.2%**;§8 的 int32 headroom 判定不受影響(nets 從 31.54M 到 31.60M)。N1 的計數(4,413 / 26,034 / 151,259)以 `rejected_alternative` 欄位併列,供讀者檢核正規化選擇的量級後果。host 三係數模型不受 verdict 影響(它量的是 `PlaceDB.read` 的 host RSS,與 glue 認證無關);2×2 維持**只當 holdout、永不進 fitting**。

### 6.2 T8 合成臂:照常跑,加標記

6.2M / 12.3M × K{16,32} × {`flat`, `ours@M2`} 照原計畫進 `scaling_synthetic_cases.md`。追加要求:

1. 每筆結果的 JSON 帶 §5.4 的七個 provenance 欄位(至少 `benchmark_kind` / `generator_verified` / `t6_verdict` / `normalization`);
2. `scripts/m4_report_lint.py` 新增一條規則:**合成表的每一列都必須有 `generator_verified` 欄位;若其值為 false 而該列所在檔案沒有出現 §7-1 的逐字揭露段落,即報錯**;
3. 表格 caption 逐字印出 §7-1;
4. T8 兼作 T0b 模型 holdout 的角色不變(那是記憶體/runtime 的預測檢驗,與生成器認證無關)。

---

## 7. 必須逐字寫進 M4 報告的揭露文字

### 7-1(spec 強制句 + 事實補充;強制句一字不改,補充只增不減)

> 30M 的連通性是擬合出來的,不是校準出來的。
>
> (補充說明,不取代上一句)本 benchmark 的跨 tile 連通性**參數**確實取自真實 `mempool_cluster` 的實測跨 group net 統計(六組 pair 計數 3,794–4,566,經 DEF parser bug 修正後),不是純代數假設;未被校準的是**生成器整體**——它在 2×2 尺度上的預註冊 holdout 驗收(T6)判定為 **FAIL**,詳見下段。因此本 benchmark **僅供 scaling 量測(cell/net/pin 數、記憶體、runtime)使用,永久不得用於任何品質宣稱**(HPWL / IO crossing / feed-through / winner / pareto)。

### 7-2(H1 的 gate 設計缺陷,B5)

> **B5(量測工具缺陷,2026-08-15 揭露)**:H1 的絕對區間 `[0.55, 0.80]`(design draft §3.3)是依「group 端子計數 Rent p ≈ 0.633」訂的,而 H1 實際使用的是 **mtkahypar 遞迴二分(Landman–Russo)**估計器。兩把尺在本設計族上系統性相差約 0.11:同一份 `mempool_cluster` 用遞迴二分量得 **p = 0.5221 [0.5001, 0.5459]**,`mempool_group` 量得 **0.4747 [0.4481, 0.4928]**。**真實參照自己就落在預註冊區間之外**,因此 H1 的絕對腿在本次評測中對「生成器是否忠實」不具任何鑑別力——一個與真實 cluster 完全一致(`|Δp| = 0`)的合成品同樣會被判 fail。依預註冊紀律,門檻**不得事後修改**,判定維持 **FAIL**;上一輪裁決文件中「真實端預期 p ≈ 0.633」的預測據此記為**已被否證**。H1 的相對腿(`|p_syn − p_real| ≤ 0.05`)**通過**:N1 = 0.0102、N2 = 0.0041。此估計器混用未外溢到 glue 參數:`λ₀_tile` 的尺度換算若改用遞迴二分的 p,由 4,387.3 變為 4,328.2(差 1.36%)。

### 7-3(H6 的機制與它實際證偽了什麼)

> **H6(Rent 不變性,{1×1, 1×2, 2×2},`|Δp| ≤ 0.03`)兩種正規化皆判 FAIL,但兩者的失敗機制不同,必須分開陳述。** 含 `1×1` 的兩條腿把兩件事混在一起:(i)換陣列形狀時每 tile 端子是否守恆,(ii)加入 glue 本身是否改變 p。後者在真實世界裡並非 0——真實 `mempool_group` standalone 的 p 是 0.4747,真實 `mempool_cluster`(四個 group 加上層互連)是 0.5221,真實的階層躍升就是 **+0.0474**;合成 N2 的躍升是 +0.0515、N1 是 +0.0576。唯一不含此混淆的一條腿(`1×2 vs 2×2`)給出了明確的鑑別:**N2 = 0.0042(形狀不變)、N1 = 0.0412(隨形狀漂移)**,方向與幅度都與「N1 讓每 tile 端子隨陣列大小成長、違反 Rent」的先驗預測一致。**H6 因此確實完成了它被設計的用途——證偽 N1;30M 採用 N2。** 依預註冊紀律,`1×1` 那兩條腿的 null 設為 0 是 gate 設計缺陷,但**不得事後改判**,verdict 維持 FAIL。
>
> 偏離的機制已被逐層驗證且完全歸因於一項**已登錄的建構簡化**:glue net 的端點在 tile 內是**均勻隨機**抽樣的(非 interface-cell 偏置),因此一個大小 B 的 block 對 glue 端子的貢獻是 `ε·B`(指數 1)而非 base 的 `B^0.47`;實測在三個數量級上 `ΔT(B)/(ε·B) = 0.7–1.1`(B = 772,353 時 T 由 6,610 升到 9,654)。`Δp` 因此只是「每 tile glue 端點數」的函數(四個陣列的 `Δp/端點數` = 3.7–4.4 ×10⁻⁶),與陣列形狀無關。

### 7-4(正面證據與其強度分級——避免過度宣稱)

> T6 的綠燈必須依證據強度分級陳述。**強(結構性、非建構恆等式)**:H2 per-level cut(N2 逐層相對誤差最大 9.2%;若把 glue 拿掉,window 頂端三層會退到 25% / 16% / 14%,恰好卡在 25% 門檻上——**跨 tile 連通性的「量」是真實資料所要求的,而校準出來的量是對的**;此比較兩邊的 block 大小差 9.24%,以 `T ∝ B^0.522` 修正後 level 4 的誤差進一步降到 4.3%);H1 相對腿(0.0041);H6 的形狀不變腿(0.0042)。**弱(近乎建構恆等式)**:H3 全網 degree KS = 0.013——合成陣列與真實 cluster 共用同一個 `mempool_group` 拓撲,此結果幾乎必然;H4 KS = 0.011——依 2026-08-14 裁決已降級為對「來源 group 自身」的自洽性檢查,量的是抽樣程序與其自身建構假設是否一致。
>
> 另揭露:真實 cluster 的最大 net degree 為 1,081,189(兩條全域 net 合計 986,903 pins),合成陣列為 342,429(來源 group 的最大 net,四份各自獨立,未合併);Rent 與分割在兩側一律套用 `max_net_degree = 100` 上限。

### 7-5(V0 的判準修正)

> `verify_bench.check_v0_structural` 對真實衍生網表有三條必然誤判的子檢查,2×2 的 1,338 條 error 全部出自它們,**沒有一條源自 tiler**:(a)「self-loop」693,952 條 = 來源 `mempool_group` 自身 173,488 條(占其 nets 的 4.951%)× 4,一條 net 接到同一 cell 的兩隻腳在真實網表中合法;(b) row「縫隙」出現在 macro 覆蓋處,含 macro 的 floorplan 在該處本就不該有 placement row(`tile_bookshelf.py:259-276` 已於建構期記載此重新詮釋,tiler 自身的 assert 早已改為只查 overlap);(c) 8 個未被任何 net 參照的 node = 來源 2 個 × 4。三條均改為「不劣於來源」的相對判準後重跑。V0 依 2026-08-14 裁決為診斷不判定項。

### 7-6(核形狀的不可識別性,在 N2 之下的正確表述)

> 距離核的形狀在 2×2 上有 0 個 lack-of-fit 自由度(§3.2),因此 3×3 的核是一個**宣告的假設**。在 N2 正規化之下,核的選擇**完全不改變 glue 總數**(每 tile 端子預算固定,核只重新分配),只改變空間局部性:d=1 pair 所佔比例在 K1(α=0.102)為 34.6%、K2(指數,α′=0.0856)34.8%、K1(α=1.915)58.9%、K3(截斷,不連 d>√2)74.3%。因此 §3.2 line 304 的「三核總數全距 > 20% 才附 K3 變體」在 N2 下恆為 0%,已退化為無資訊的觸發條件;本報告改以距離層級分佈報告系統性不確定度,並無條件附上 K3 全陣列作為「不模擬 long-range 連通性」的明示下界。

---

## 8. 給執行者的實作缺口清單(全部是機械工作,無設計決定)

1. **【阻塞、且有資料遺失風險——已於 2026-08-15 搶救至 `results/m4/bench/rescue/`】** `/tmp/rent_synthetic_result.json`、`/tmp/rent_cluster_result.json`、`/tmp/assemble_t6.py`、`/tmp/measure_rent_{cluster,synthetic}.py` 原不在 repo 內,而 H1/H2/H6 的逐層 `(B, T)` 表與 verdict 組裝邏輯全在裡面 ⇒ T6 的 holdout 判定此前**無法從 repo 重現**。依 T0 慣例遷入 `ioplace/diagnostics/probes_m4/probe_rent_arrays.py` 與 `ioplace/bench/assemble_t6.py`,逐層表存成 `results/m4/bench/rent_levels_{synthetic,cluster}.json`,並把 `verify_group2x2_*.json` 的 `rent_p` 補上 CI 與 `levels_used`。
2. `ioplace/bench/glue_gen.py:170-203` — 新增 `n2_pair_counts_sinkhorn(lambda_0, alpha, R, C, budget_pairs=3.0, tol=1e-9, max_iter=1000)`,規格見 §5.2;保留舊函式,並加測試「1×2 / 2×2 兩者逐位元等價」「3×3 每 tile 端子與 B 的相對差 < 1e-9」「K1/K2/K3/α∈{0,1.915} 總數皆為 `m·B/2`」。
3. `ioplace/bench/build_t6_arrays.py:98-104` — `_expected_pair_counts` 的 `n2` 分支改呼叫 Sinkhorn 版;`SHAPES` 加 `3x3: (3,3)`;加 `--no-glue` 旗標(產 B-1 需要的零 glue 對照陣列)。
4. `ioplace/bench/verify_bench.py:69-166` — (i) 改單趟串流(見 §4.2 末);(ii) 三條子檢查改「不劣於來源」語意;(iii) `check_h1_rent_shape` 加 `band=None` 時只評相對腿並回 `band_status="not_applicable"`(供 T6B 使用,**不得**改動 default band 的數值)。
5. `ioplace/bench/verify_bench.py` — 新增 `check_b1_rent_invariance_did(...)`(門檻常數 0.03 寫死、無參數搜尋介面)與 `check_h5_k_grid_lambda_ratio` 的等解析度呼叫端(用 `ioplace.regions.make_grid_regions(die, nx, ny)`,2×2→(4,4)、來源→(2,2))。
6. `scripts/m4_report_lint.py` — 加 §6.2 第 2 條規則(linter 於 T11 建立時一併實作)。
7. 文件同步(spec v2.3):§3.3 H1 列加註 B5;§7.1 的 T6 列加「holdout FAIL,轉 T6B」、T4B 列標 `not_executed(觸發條件未成立)`、T6B 列改寫為 §4.1 的表、T7 列改為 §5.3 的參數;§9 加 **M4-G3 已觸發** 的紀錄;§1.4 加 **B5**;§10 的 M4-L4 加上 3×3 Rent 的時間/記憶體實測預估與中止條件。

## 9. 建議 Codex 對抗性複核的兩點

1. **§1.2 的「不得重判」是否過嚴**——是否存在一個原則,能同時容許「修正估計器不匹配的接受域」又不打開「看到結果再改門檻」的門?(立場:不存在,因為修改方向可判斷地對己方有利;但這值得一個不同模型家族來攻。)
2. **§5.2 選 Sinkhorn 而非閉式平均**——(A) 在主線 α=0.102 下只差 1.5%,是否值得引入一個迭代步驟;以及「每 tile 端子預算與位置無關」這個 Rent 前提在角落 tile 上是否真的該成立(真實晶片的角落區塊介面確實較少)。
