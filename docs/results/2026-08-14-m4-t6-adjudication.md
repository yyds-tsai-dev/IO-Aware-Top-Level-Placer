# M4 T6 校準協定裁決(2026-08-14)——G-D 結論撤回與 T6 協定 v2

- 依據:`results/m4/corpus/hierarchy_gate.json`(commit 309e5d7,**其 G-D 數字已被本文作廢**)、M4 design v2.1 §3
- 裁決者:deep-reasoner(Opus),scheduler 採納
- 探針:`/tmp/p1_clean_cross.py`、`/tmp/p3b_mass_cluster.py`、`/tmp/p4_tilepairs.py`、`/tmp/p5_arrange.py`(待依 T0 慣例遷入 `ioplace/diagnostics/probes_m4/`)

## 裁決:T6 協定不需要為 G-D 改寫——G-D 的結論本身是量測假象,必須先撤回

**結論(前置)**:「真實 placement 不按階層分區」不成立。`hierarchy_gate.py` 的 G-D 用 **bbox 與 bbox 中心**判定幾何,那是 2.8M 點的極值統計,由每群幾顆散落 cell 決定。改用質量統計後,真實 `mempool_cluster` 的四個 group **確實是一個(模糊的)2×2**:最佳象限指派可解釋 **66.0%**(自由切割線 cx=0.54/cy=0.52 下 **68.0%**)的 cell 質量,而 1×4 只有 48.8%、4×1 只有 46.5%;真實質心兩兩距離 **0.12–0.55 die extent**,不是 gate 記的 0.00032。

同時發現**第二個更嚴重的 bug**:DEF `NETS` 的 pin tuple 掃描沒有在 `+` 子句處停止,把 `+ ROUTED ... ( x y )` 繞線座標與 `( * pin )` 萬用 tuple 當成 pin。後果:`cluster_stats` 要用的跨 group 量**高估 7.2 倍**。

⇒ T6 的陣列形狀回到 v2.1 主線 **2×2**,1×4 分支作廢;但 T3a 必須先修兩個 bug 並重跑。

## 證據摘要

**Bug A(bbox 質心)**:質量統計實測四 group 質心兩兩距離 0.1216–0.5541 die extent;64×64 bin 質量加權最大佔比 0.9685(均勻混合=0.25);「單一 group 佔比>0.9」的 bin 涵蓋 91.46% 質量;最佳 2×2 指派 0.660(固定中線)/0.680(擬合切割線)vs 1×4 0.4879 / 4×1 0.4649。指派:g1=UL、g3=UR、g2=LL、g0=RD;左右半邊純度 0.869、上下 0.765——真實但「軟」的 2×2。group 內 16 tile 層更乾淨(象限純度 93–98%),bbox 判準在兩個階層都給出與事實相反的答案。

**Bug B(`+` 截斷)**:修正後 parser 的 pin 數逐一對上 DREAMPlace(cluster 43,944,352+3,441=43,947,793;group 12,014,771+11,420=12,026,191)。頂層殘量 pins 6,668,755→**24,276**;每 group 跨界 net 78,214–91,128→**11,653–12,585**(高估 7.2×);group 端子 Rent p 0.759–0.769→**0.6305–0.6352**(四群一致到第三位,落在 spec §7 的 [0.55,0.80])。

**真實跨 group 統計(修正後)**:多 group net 22,399 條(0.176%);六 pair 計數 4002/4227/4034/4566/3794/4019;觸及 2/3/4 群 = 21,552/517/329;85% 是 degree-2;跨界總 pin 1,313,534,其中 986,903 集中在 2 條全域 net(扣掉後平均 degree 14.6)。

## 1. V/H 指標裁決

分類原則:指標兩端是否都是拓撲量。

| 指標 | 裁決 |
|---|---|
| V0/V1 | 存活不變 |
| H1 Rent(mtkahypar)| 存活,門檻不變(真實端預期 p≈0.633)|
| H2 per-level cut | 存活不變 |
| H3 degree KS | 存活 + 加 tail 揭露(全域 net 政策)|
| H4 interface-cell 距離 | **降級自洽 gate**:參照改「來源 group 自身」(真實邊界是軟的,跨設計 KS 必敗且與生成器無關)|
| H5(V4)K-grid λ 比 | **降級診斷不判定**,註明預期 `syn < real` |
| H5′ 算術自檢 | 存活不變 |
| 幾何 Rent | 保留 OOM fallback + 同報告同估計器硬規則 |
| **H6(新)Rent 不變性** | {1×1,1×2,2×2} 上 `|Δp| ≤ 0.03`——直接證偽 glue 正規化選擇 |

## 2. glue 統計與距離核

- **(a)** 直方圖類統計為拓撲量、不受 G-D 影響,但現行 artifact 因 Bug B 全錯,必須重量。
- **(b)** 2×2 下閉式 MoM 適用:`λ_adj=4155.2`、`λ_diag=4010.5` ⇒ **α=0.1023、λ₀=4155.2**;但 `var/mean=16.9>3` 觸發 v2.1 §3.2(c) ⇒ 所有 Poisson 檢定標不可靠;可發表陳述:**|α| ≤ 0.2,主線 α=0.10,「距離響應未被統計解析」**。架構解釋:group 間是 K4 全連接等寬(`gen_remote_interco`),λ 平坦是 RTL 事實。
- **(c)** 下一層(16 tile,120 pair)實測 **α=1.915、corr=−0.785** ⇒ α 階層相依。敏感度掃描 **α ∈ {0, 0.102, 1.0, 1.915}**(1.915 標「樂觀端:placement-induced,因果反向,對本方法過於友善」)。
- **(d)** **正規化採 N2(每 tile 端子預算 B=3λ₀,按 φ(d) 分配)**,否決 N1(每 pair 常數;會讓每 tile 端子隨 m 成長、違反 Rent)。3×3 glue:N2 59,231 nets vs N1 151,283。尺度換算 Rent 一致式 `λ₀×(3.0780/2.8246)^0.633=4,387`。**T6 兩種都產,由 H6 判;不得因失敗重掃。**
- **(e)** **全域 net 政策:`per_tile` 獨立**(不合併;合併會造 4.4M pin net 塌進 evaluator degree bucket 成 artifact)。manifest 記錄;T6 並列報 syn/real 最大 net degree。
- **(f)** glue 量級:3×3 N2 下 +0.19% nets/+0.80% pins ⇒ v2.1 §2.1 的 g 佔位帳誤差 <1%,M4-L12 降級已解。

## 3. 30M 有效性主張

**設計性選擇,且 G-D 修正後更強**:真實商用流程產生的正是同一個 2×2 的模糊版;合成 benchmark 是其銳利版 = partition-driven 頂層流程的輸入條件。報告措辭(v2.1 紅線不變:僅 scaling、不作品質宣稱):

> 30M 合成陣列採用 abutted tiles + 距離衰減 glue,這是對應本專案問題設定的設計性選擇(無 channel 的分區式頂層放置),不是對商用 flat placement 結果的模仿。此選擇有實測支撐:真實 `mempool_cluster` 的四個 RTL group 在 Innovus flat placement 中確實構成 2×2 排列(最佳象限指派 68.0% 質量,對照 1×4 的 48.8%),但邊界是軟的——13.1% 的 cell 越過垂直邊界、23.5% 越過水平邊界,每個 group 的 bbox 橫跨幾乎整個 die。合成 benchmark 取同一排列的硬邊界版本。

## 4. T6 可執行協定 v2

**前置(阻塞,屬 T3a)**:
1. `hierarchy_gate.py:236-255`:net record 遇第一個 `+` 即停收 pin tuple;加 `n_wildcard_tuples`/`n_post_plus_tuples_skipped` 欄。驗收:pin 數對上 §1.1 表(cluster 43,947,793;group 12,026,191)。
2. `hierarchy_gate.py:113-154`:質心改**質量質心**(sum_x/sum_y/n_placed);bbox 改名 `bbox_outlier_extent` 註明極值統計。
3. 刪距離分層分類器,改**假設指派評分**:`score_2x2`(cx,cy∈[0.20,0.80] step 0.02 搜最佳切割線 × 4! 指派)vs `score_1x4`/`score_4x1`;**PASS_2x2 = score_2x2 ≥ 0.50 且領先 ≥ 0.10**(實測 0.6797 vs 0.4879,margin 0.192)。輸出 assignment、三 score、bin dominance、質心矩陣、純度。
4. 重跑取代 `hierarchy_gate.json`(預期 `PASS_2x2`);M4 報告記 **B4** 條目(309e5d7 的 gate 數字作廢)。

**T6 本體**:陣列 2×2(12.31M);鄰接由 assignment 給定;參數 λ₀=4155.2、α=0.1023、λ₀_tile=4387;不確定度 net-level bootstrap ≥1000 + ≥5 seeds;正規化 N1/N2 都產由 H6 判;核敏感度 α∈{0,0.102,1.0,1.915};fit 指標(只報):λ₀/α+CI、glue degree 直方圖、pinshare、iface_dist、六 pair 原始計數與 overdispersion;holdout(評一次):H1 |Δp|≤0.05 且 ∈[0.55,0.80]、H2 ≤25%、H3 KS≤0.05+逐桶≤20%+最大 degree 揭露、H4 自洽版 KS≤0.10、H6 |Δp|≤0.03;診斷不判定:H5、H5′、V0/V1。全綠 ⇒ T7 用選定正規化+α=0.102 產 3×3(另產 α 變體與 K3);不過 ⇒ 不得重掃,轉 T6B。

**文件同步(spec v2.2)**:§3.1a G-D 列、§3.2 1×4 段、§7.1 T3a/T6 列、§9 M4-G3a、R8、§10 M4-L5(補「α ≤0.2 已量,有跨階層錨點」)、M4-L12(降級已解)、§1.4 加 B4。

**建議 Codex 複核的兩個高風險選擇**:N1 vs N2 的 Rent 論證;α 主線取 0.102 而非 1.915 的方向性選擇(關係 benchmark 是否對本方法過於友善)。

## 5. 動機性陳述(修正版,可直接進論文)

> 我們在 ISPD2025 `mempool_cluster`(11.31M cells)上量測商用流程(Innovus, place_opt/preCTS)flat placement 與 RTL 階層的關係。四個 RTL group 確實被大致排成 2×2:最佳象限指派涵蓋 68.0% 的 cell 質量(隨機=25%,1×4 僅 48.8%),64×64 bin 質量加權最大佔比 0.9685。但這個分區完全沒有被強制:13.1% 的 cell 越過垂直邊界、23.5% 越過水平邊界;8.5% 位於無任何 group 佔優的混合 bin;每個 group 的 bbox 橫跨 ≥99.98% 的 die。更關鍵地,group 間互連在 RTL 上是全連接等寬的(六組 pair 的 net 數 3,794–4,566、平均 4,107、α=0.10±0.08 且 var/mean=16.9 使其統計上無法與 0 區分)。純線長目標的 flat placer 因此沒有任何訊號去偏好把相鄰 group 的邏輯放在一起。這正是頂層放置需要顯式 region/IO 目標的原因——越界 cell 直接轉成 IO crossing 與 feed-through,而線長目標對它們幾乎不敏感。相同量測在下一層(group 內 16 tile)呈現強距離結構(α=1.92、corr=−0.79),說明「無距離紀律」是頂層特有的,正是本工作鎖定的層級。
