# IO-aware placement 階段進度（2026-09-06，執行中）

以下區分「工作已完成且結果已測試」與「品質／模型門檻通過」。失敗的實驗
仍是已完成的工作；不能因此將品質門檻改成 PASS。

| Stage | 已完成／已測試 | 進行中／尚待完整驗證 | 尚未啟動或後續工作 |
|---|---|---|---|
| M0：baseline、資料與 evaluator 基礎 | 歷史 flat/two-stage baseline 與幾何指派修正已有結果。本輪 schema3 native node/pin identity 修正完成；schema3真實visible1×2 endpoint集合等價、三cache全量驗證通過。後续schema4補足output-front遍歷，真實1×2嚴格順序等價537.66s通過；33項新回歸通過。 | 歷史 two-stage 的 IO 方向 FAIL、escape-cell legality 缺口，不能由本輪 cache 修正推定已解決。 | 完整處理 two-stage escape-cell 問題。 |
| M1：GPU evaluator、reweight | 歷史 CPU/GPU parity、chunking、閉環 reweight 與真實設計效能測試完成。本輪補足 evaluator／IoTerm 144組 primary、36組 holdout。 | 六個新元件模型均未通過辨識門檻；已有量測，尚無有效外推模型。 | 依失敗資料另立新模型與外部驗證；擴大 reweight 跨設計證據。 |
| M2：可微 IO placement | 歷史 exit PASS；adaptec1 最佳臂 IO −18.538%、HPWL +1.314%。本輪保留 legacy 路徑並加入 opt-in atomic callback；真實40-iteration integration 通過。 | 歷史 FT 上升問題由 M3 新實驗處理，不能把 M2 的 IO 改善等同 FT 改善。 | 更廣泛的設計／region regime 泛化。 |
| M3：FT 訊號與可微 FT loss | 歷史 exit FAIL 保留。本輪21組預註冊實驗、三組配對確認與21組 boundary probes 完成。A2 對 A0 在 adaptec1 上 IO/FT 下降。P0c 三個正權重候選皆因 IO guard 失敗。 | 尚無跨設計 FT objective 成功證據；A2 的 FT 仍高於 flat。 | P0c confirmation/P4 因 gate 未通過而不啟動；bigblue4 rho 校準等廣義 M3 未決項仍屬後續。 |
| M4：10M–30M scale-up | 歷史10M full flow、T6B替代路徑與T11完成。本輪恢復歷史相同27.7M輸入、凍結H100 NVL共享預測；T9A/B各4輪通過。Host18train+3holdout完成，兩個 fit 均未通過辨識。 | 27.7M GP+LG+evaluation全部完成，0 unplaced、overflow .06630，phase總和3.623h，CUDA allocated峰值25.308GiB。凍結時間假設FAIL；額外核對定位並修正cache順序，三次GPU核對精確重現native IO／FT，warm eval3.216s。 | 依T14結果決定是否另立校準實驗；單次首跑不產生有效PI。 |
| Stage2：placement→route→校準 | 12組新placement完成。OpenROAD GRT與uint64 binding修復；wire oracle、orientation/multiport、identity tests通過。Per-net落盤、routed-geometry重算、FT、δ0/1/2/4、boundary、C1–C5程式完成；52項校準測試與四個真實FFT case完整鏈通過。 | tile首次route因4,360條繼承signal stub被GRT跳過，coverage gate失敗。已保留失敗樣本並修正輸入清除，在新目錄重繞相同placement。尚在完成全部routing與三設計完整校準；目前FFT四臂已產生配對C4結果，只有一設計，C5不可評估。DR為固定迭代上限，仍有DRC，不是signoff-clean。 | 本輪沒有啟動Innovus或commercial-DP對照；大於本cohort的routing coverage仍屬後續。 |
| M5：離散 CE decode＋boundary refinement | 實作compiled CE、逐move/swap重算、完整legal acceptance與extra-LG control；14項核心／integration tests通過。1M/10M/30M、active≤65,536元件量測完成。三設計四臂初篩12/12完成，9個候選全部拒絕；最終座標與none對照逐元素相同。 | 初篩完成，但沒有通過完整品質門檻的方法。部分adaptec1/tile候選改善IO+FT綜合分數，仍因HPWL guard失敗。 | 沒有符合條件的seed1001–1003確認；不宣稱最終品質改善。 |

## 本輪工作1～5

1. **Native replication correctness：已修正並通過真實測試。** schema4真實1×2嚴格ordered等價通過；歷史manifest與array一致，三cache全量ordered驗證均通過；27.7M整數parity通過。
2. **H100 27.7M 首跑：完成。** GP+LG+評估、完整性檢查及凍結預測裁決均完成；時間假設不成立。額外GPU-only整數parity已通過。
3. **Stage2逐net routing校準：進行中。** 程式、回歸與FFT四臂完整證據完成，
   全cohort routing及三設計C1–C5結果未完成。
4. **IO-enabled FT factorial、P0c與boundary/noise：完成。** 如實保留正面
   A2確認與負面P0c結果，不改寫歷史M3 FAIL。
5. **元件模型＋M5：完成並已測試。** 模型
   辨識失敗與M5候選拒絕都是實測結果，不宣稱外推或品質成功。

目前採既有H100 NVL與更新後DREAMPlace，共享GPU有餘裕即可執行；
記錄UUID、背景用量與共享狀態。Subagent套用專案新規則：Astra/high負責
唯讀診斷，Luna/low負責明確指定的機械工作，主agent整合並核對產物。

最新完整non-slow：991 passed、1 skipped；driver integration 35 passed、DEF export/reweight 12 passed。明確使用MtKaHyPar單執行緒緩解措施；原生parallel缺陷沒有宣稱已修復。各test group有重疊，不相加成唯一test數。

詳細證據：[執行清單](2026-09-06-execution-plan.md)、
[輸入與工具](2026-09-06-input-inventory.md)、
[FT](../results/2026-09-06-ft-followup.md)、
[模型](../results/2026-09-06-component-models.md)、
[M5](../results/2026-09-06-m5-followup.md)、
[H100首跑](../results/2026-09-06-h100-nvl-followup.md)、
[native runtime](../results/2026-09-06-native-runtime.md)。
