# IO-aware placement 階段進度（2026-09-08 UTC）

本輪原定工作 **1～5 全部完成並完成對應驗證**。工作 3 的 12/12 routing、校準、G4 診斷與全量文字解析交叉驗證均已完成。以下把「完成實驗」與「品質門檻通過」分開；失敗結果不改寫成 PASS。

| Stage | 已完成／已測試 | 進行中／未測試或仍未通過 | 待辦／未開始 |
|---|---|---|---|
| M0：baseline、資料與 native identity | Schema4 修正 native node／pin／output-front 順序；真實 visible 1×2 嚴格順序等價通過（537.66s），三個 cache 全量 ordered 驗證通過，27.7M 整數 parity 通過。 | 歷史 two-stage IO 方向 FAIL 與 escape-cell legality 缺口仍未解決。 | Two-stage escape-cell 修正。 |
| M1：GPU evaluator、reweight | CPU/GPU parity、chunking、閉環 reweight；本輪 144 組元件量測＋36 組 holdout。 | 六個元件模型均未通過 coefficient-CI 門檻，沒有有效外推模型。 | 新模型辨識、外部驗證與跨設計 reweight。 |
| M2：可微 IO placement | 歷史 exit PASS；adaptec1 最佳臂 IO −18.538%、HPWL +1.314%；本輪 atomic callback 與真實 integration 通過。 | FT 上升問題仍不能由 IO 改善推定已解決。 | 更廣泛 design／region regime 泛化。 |
| M3：FT 訊號與可微 loss | 21 組註冊實驗、三組配對確認、21 組 boundary probes。A2 對 A0 的 IO／FT 下降。 | 歷史 exit FAIL 保留；A2 的 FT 仍高於 flat。P0c 三個正權重候選皆失敗於 IO guard。 | P0c confirmation／P4 因 gate 未過而不啟動；跨設計 FT objective 與 bigblue4 校準仍待做。 |
| M4：10M–30M scale-up | 27.7M H100 NVL GP＋LG＋evaluation 完成，0 unplaced、overflow 0.06630、phase 合計 3.623h、CUDA allocated 峰值 25.308GiB。修正 cache 後三次 GPU 整數 parity 全過，warm evaluation 3.216s。 | 凍結時間預測 FAIL；host 18 train＋3 holdout 的兩個模型未通過辨識。單次首跑不構成有效 prediction interval。 | 另立模型校準與多次外部驗證。 |
| Stage2：placement→route→校準 | 三設計 × K16/32 × flat/ours 共 12/12 樣本完成，缺線率全為 0%。逐 net evidence、identity、wire oracle、δ0/1/2/4、7 張表、C1–C5、G4 raw 雙值與距離分布完成；全量文字解析交叉驗證 9/9 獨立 route 通過（對應 12/12 cases）。 | 主要 δ2 的 E2 僅 2/12 達 Spearman≥0.70；raw 為 10/12。C5：δ2 FAIL（γ CV 0.387），raw PASS。DRC 仍存在，非 signoff-clean。 | Innovus／commercial DP 未啟動；benchmark 遷移與 RG 模型重審需另行決策。 |
| M5：CE decode＋boundary refinement | Compiled CE、逐 move/swap 重算、legal acceptance 與 extra-LG control；1M/10M/30M 元件量測完成。三設計四臂初篩 12/12 完成，9 個候選全部拒絕，最終座標與 none 對照完全相同。 | 尚無通過完整品質門檻的方法；部分候選改善 IO＋FT 綜合分數，仍失敗於 HPWL guard。 | 沒有符合條件的 seed1001–1003 confirmation，故不啟動。 |

## Stage2 子階段

| 子階段 | 狀態與實際範圍 |
|---|---|
| S0 環境、S1 DEF 輸出 | 本輪 H100／DREAMPlace／OpenROAD provenance、12 組 placement 與 DEF／netmap／CoordMap 已保存，native export／identity 測試通過。 |
| S2 routed-DEF 解析 | 全設計 wire-length oracle 與所有 routed nets 的 canonical geometry-set 比對通過：9 個獨立 routes、847,230 次逐 net 比對，零 mismatch／missing；不比較 wire width 或重複 multiplicity。 |
| S3 crossings／FT | δ0/1/2/4、per-net FT／lambda／boundary demand 與合成／native 回歸完成；FT 未知值不補零。 |
| S4 OpenROAD flow、S5 NanGate45 | FFT／DES／tile 的 GP＋LG→GR＋DR→evidence 跑通；tile 繼承 stub 問題修正後 0% 缺線。可見資料／15 LEFs 與 group 規模資料驗證完成，歷史失敗資料另存。 |
| S6 Innovus 環境、S7 F-INV | 本輪未執行；本次僅交付相對 OpenROAD 的 open-source router ground truth。 |
| S8 校準矩陣 | 12/12 完成。相同輸入的兩個 flat K 明示重用 route；不是 12 次獨立隨機 routing。 |
| S9 校準 | Raw／δ2 各 7 張表與 C1–C5 完成。六組同 K 的 IO 改善方向均一致；raw／δ2 的 C5 判定不同。 |
| S10 報告 | 主報告、raw 雙值、距離圖、E1–E5／G1–G7 與全量 S2 supplement 全部完成。 |

## 本輪工作 1～5

1. **Native replication correctness：完成。** Schema4、真實 1×2 ordered 等價、三個 cache 全量驗證、27.7M parity 全部完成。
2. **H100 27.7M 首跑：完成。** GP＋LG＋evaluation、資源紀錄、凍結預測裁決與額外 GPU parity 完成；時間假設不成立。
3. **Stage2 逐 net routing 校準：完成。** 12/12 evidence、raw／δ2 校準、G4 距離診斷與全量文字解析 supplement 完成。
4. **FT factorial／P0c／boundary noise：完成。** 保留 A2 正面確認與 P0c 負面結果，不更改歷史 M3 FAIL。
5. **元件模型＋M5：完成。** 模型辨識失敗與候選拒絕均是完整實测結果，不宣稱外推或最終品質成功。

最新完整 non-slow：**1015 passed、1 skipped、39 deselected**（40.22s）。
真實 driver integration 35 passed、DEF export/reweight 12 passed、visible 1×2 ordered equivalence 1 passed。
各群組有重疊，不相加成唯一 test 數。MtKaHyPar 使用明示單執行緒緩解措施；原生 parallel wheel 問題未宣稱已修復。

使用既有 H100 NVL 與更新後 DREAMPlace；共享 GPU 有餘裕即可跑，保留 UUID／背景用量。
Subagent 依新規則：Astra/high 唯讀診斷、Luna/low 執行限定工作，主 agent 整合並核對。

本輪沒有仍在執行的實驗；表中的未通過門檻與後續待辦屬下一輪研究，不作成功宣稱。

詳細證據：[Stage2](../results/2026-09-06-stage2-followup.md)、[執行清單](2026-09-06-execution-plan.md)、
[輸入](2026-09-06-input-inventory.md)、[FT](../results/2026-09-06-ft-followup.md)、
[模型](../results/2026-09-06-component-models.md)、[M5](../results/2026-09-06-m5-followup.md)、
[H100](../results/2026-09-06-h100-nvl-followup.md)、[native runtime](../results/2026-09-06-native-runtime.md)。
