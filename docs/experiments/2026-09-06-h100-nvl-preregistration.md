# H100 NVL：27.7M 執行與預測登錄

本登錄在本次 27.7M GP 結果產生前凍結。原 SXM5 預測保留為歷史文件，
不適用於目前 NVL 的硬體判定。

可機器核對的設定與預測位於
`results/h100_nvl_followup_20260906/forecast/{sku,protocol,h100_nvl_prediction}.json`。
`protocol.json` 登錄完整 config、來源 snapshot、輸入 manifest 的 SHA256，
以及下列條件：

- NVIDIA H100 NVL，93.58 GiB 可見容量；process 預算 84 GiB，啟動時另留 4 GiB。
- 使用者允許共享 GPU。全卡使用量僅作容量觀測；Torch allocated/reserved
  作 process 記憶體觀測，不以相減方式虛構獨占 NVML 用量。
- 原 visible corpus 的 3×3：27,699,021 movable、27,804,699 physical nodes、
  31,595,252 nets；108,354,439 raw pins、106,683,202 canonical pins。
  原 raw count freeze 保留；native 執行採去重後的拓撲。
- T9：K32、四次 IO forward/backward 與 evaluator 交錯執行；A、B 各新程序，
  B 明確為兩個 fp64 net buffer 的 ballast，不宣稱執行 FtTerm；每次上限六小時。
- T14：原生 PlaceDB 讀檔與 fillers，4096² bins、2000 iteration 上限、
  target density .714、GP+LG、無 DP；K16 flat observer、seed1000、rho0、
  legacy callback、每50次 callback、無診斷；程序上限24小時。
- T9 cache 必須完成 full streaming verification，另有實際 native 1×2 等價證據。
  T9 成功僅說明元件共存，不等於完整 GP 可行。

預測沿用 read/LG 1×、GP/eval 2.5–5× 的**未驗證假設帶**，沒有以本次結果
校準速度倍率。原始參考與本次 DREAMPlace／evaluator 實作存在差異，這是
預測可能失準的原因，不能在結果出來後改寫區間。

依既定規則逐 phase 檢查速度倍率相對宣告帶的誤差是否不超過50%，並檢查
phase 總時間是否位於凍結總時間帶。另報 supervisor wall time。
共享情境的全卡用量與 process 解析界比較標記
`not_applicable_shared_attribution`；process peak、全卡 peak 與資源餘裕分別報告。
Host RSS 模型未通過辨識門檻前，不產生假的驗收區間。

本合成 benchmark 的既有 generator verification FAIL 保持不變；
大型執行提供容量／時間證據，不提供真實設計品質提升的結論。
