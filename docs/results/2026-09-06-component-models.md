# 2026-09-06 元件模型證據報告

本報告只整理已記錄的量測與預註冊規則，不重新擬合、不刪除失敗係數，也不把未通過辨識門檻的點估計當成可交付預測區間。資料來源是 `results/component_models_20260906/`、`results/host_rss_models_20260906/` 與 `docs/experiments/2026-09-06-component-model-preregistration.md`。

## 結論

主要矩陣的 evaluator 與 io-term 量測均完成：共 144/144 個 process 成功，holdout 共 36/36 個 process 成功，沒有靜默刪除或失敗配方遺漏。六個元件模型（evaluator/io-term × allocated/reserved/warm runtime）全部未通過係數相對信賴區間 gate；因此沒有任何 validated prediction interval，也不允許把模型外推當成可行性或容量保證。Evaluator holdout 的 warm runtime 點估計全部為負值（實測為正），這是明確的模型失效訊號。

Host RSS 是獨立的 Bookshelf-read 實驗。6 個訓練配方與 3 次重複完成；host RSS 與 read-time 兩個模型都未通過辨識 gate。可見 2×2 holdout 的 host RSS 實測為 76.1969388 GiB，未驗證點估計為 76.3492153 GiB；read 實測 685.7673374 s，點估計 609.2690785 s。這些數字是觀察與未驗證外推的並列，不能形成 PI。

## 元件模型 gate 與 holdout

預註冊矩陣固定使用 `[1,N,P,E,E×K]`（evaluator）與 `[1,c×max(N,P′),P′,E_active]`（io-term），採 scaled QR/SVD、完整宣告迴歸項與係數 95% t 區間。條件數均低於 30，但所有六個 fit 的係數相對半寬至少有一項超過 25%，故 `identifiable=false`：

| 模型 | 配方數 | rank | standardized condition | 結果 |
|---|---:|---:|---:|---|
| evaluator peak allocated GiB | 16 | 5 | 2.1748 | FAIL |
| evaluator peak reserved GiB | 16 | 5 | 2.1748 | FAIL |
| evaluator warm runtime s | 16 | 5 | 2.1748 | FAIL |
| io-term peak allocated GiB | 32 | 4 | 1.2152 | FAIL |
| io-term peak reserved GiB | 32 | 4 | 1.2152 | FAIL |
| io-term warm runtime s | 32 | 4 | 1.2152 | FAIL |

36 個 holdout process 的實測/點估計範圍如下；圖中保留負 runtime 預測，沒有把它裁成零：

| 元件/目標 | holdout 實測範圍 | 未驗證點估計範圍 |
|---|---:|---:|
| evaluator allocated GiB | 2.6944–7.2126 | 1.6709–7.1373 |
| evaluator reserved GiB | 3.3418–8.1094 | 1.1210–10.9432 |
| evaluator warm runtime s | 1.5788–6.0439 | **−0.4879–−0.1810** |
| io-term allocated GiB | 2.5711–5.1112 | 0.2743–0.4430 |
| io-term reserved GiB | 3.0176–5.9453 | 0.8553–1.0818 |
| io-term warm runtime s | 0.1999–0.7325 | 0.5375–1.0591 |

Holdout JSON 報告 `pi95=null`、`identified_model=false`、`extrapolation_allowed=false`；這是預註冊 gate 的直接結果。Historical L4 fit 與本次 H100 NVL controlled component measurements 分開保存，不能互相替代。

## Host RSS 與 read-time

Host RSS 訓練使用 6 個配方（adaptec1、bigblue4、group、groupDrop25、groupDrop50、array1x2），3 次 process replicate；2×2 是 holdout。兩模型均 rank=3、standardized condition=6.0075，小於條件數上限 30，但係數相對區間 gate 失敗。Host RSS 模型訓練實測範圍為 1.2656–38.0424 GiB，read-time 為 5.7662–299.8047 s；這是 host 記憶體/CPU read evidence，與 GPU peak 或舊 L4 evaluator 模型分開。

## Provenance 與硬體限制

- component matrix SHA-256：`54be5595fedf832a5ff2750816cbabe15dc3337ea0c69828c5d84b33fc5447cd`
- frozen component fits SHA-256：`a58e1850c8e517e5eeb795764c3e22a52a26993b8414a437079cafc8ecdb87d0`
- host matrix SHA-256：`a9774d0dcd3b515d5f183fb01712894b617d5ec97230f545d9200bf3f9fc19dd`
- host frozen fits SHA-256：`e59aac9593c9f6e580d2830ef798c5d2d94319fb815d820b6de7584e95274172`

圖表由 `scripts/plot_component_models.py` 直接讀取上述 frozen fits/holdout JSON 產生：[PNG](figs/component-models-20260906.png)、[PDF](figs/component-models-20260906.pdf)。圖中的負 runtime 預測是原始值。

量測快照 manifest 也保存了原始輸入與程式來源摘要：component matrix 的 `source_sha256` 與 `registration_sha256` 位於 `results/component_models_20260906/matrix.json`；host-RSS matrix 的 `input_sha256`、`source_sha256` 與 `registration_sha256` 位於 `results/host_rss_models_20260906/matrix.json`。因此本報告的 fits SHA 與輸入快照 digest 可共同稽核，沒有只記錄模型輸出而遺失實驗來源。

本報告沒有 H100 NVL 的端到端計時或 T14 驗證。共享 H100 NVL 的裝置負載、CUDA context、時脈與其他程序會影響實際量測；這些 component/host 模型結果不能宣稱 GPU 容量、wall-time PI 或正式 H100 可行性。

補充輸入順序範圍：本輪元件／holdout沿用凍結schema3 cache；後續查出其
endpoint集合正確，但沒有native output-front net遍歷順序。原始資源量測與
failed辨識裁決保留，不拿來宣稱ordered native品質等價；schema4重建與
27.7M數值核對另存於H100後續報告。
