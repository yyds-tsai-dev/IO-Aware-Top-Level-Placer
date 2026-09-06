# 輸入與工具清單（2026-09-06，持續更新）

## 已恢復且已驗證

- 使用現有 `/ldaphome/yyds-tsai-dev/DREAMPlace` 與 `.venv312/bin/python`，
  Python3.12、Torch2.8.0+cu128、CUDA12.8、sm90；不重建已更新的 DREAMPlace。
- 官方 ISPD2025 ZIP：
  `/ldaphome/yyds-tsai-dev/benchmarks/ispd25/archive/ISPD2025_benchmarks.zip`，
  3,269,285,853 bytes，來源 `https://www.ispd.cc/contests/25/ISPD2025_benchmarks.zip`。
- 正確 visible group：同目錄 `extracted/ISPD2025_benchmarks/visible/mempool_group/mempool_group.def`；
  SHA256 `e5266220eda7591c614ff17c815cb0f9e962256c676f0663fe8dd80928fa8d5b`，
  與歷史 corpus 完全一致。Visible tile 也已從同一 archive 解出。
- NanGate45 的15個 LEFs 位於
  `/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef`，全部符合歷史雜湊。
- ISPD2015 官方 archive 已下載並解至
  `/ldaphome/yyds-tsai-dev/benchmarks/ispd2015/`；來源
  `https://ispd.cc/contests/15/web/benchmarks/ispd_2015_contest_benchmark.tgz`，
  129,132,340 bytes。FFT、DES 等 LEF/DEF/Verilog 已可用。
  DREAMPlace內建test configs所需的 `install/benchmarks/ispd2015` 另以symlink
  指向此目錄；原先缺少這個相對路徑，native reader會在golden export test中abort。
- 正確 group export、1×2／2×2／3×3 arrays 與 schema3 caches 位於
  `results/recovery_visible_20260906/`。三種 array 的 source/output hashes、base
  counts、glue counts 均與歷史 manifest 一致，見
  `historical_manifest_equivalence.json`。三個 cache 的 full streaming
  verification 均通過；當時真實1×2 native測試比對的是endpoint multiset，
  未涵蓋output-front net遍歷順序。後續schema4補足此缺口，另存於
  `cache_schema4/`；舊cache與量測保持原樣，新ordered全量驗證三cache全部通過；27.7M三次整數核對也通過。

3×3 有27,699,021 movable、27,804,699 physical nodes、31,595,252 nets；
raw pins108,354,439，native canonical pins106,683,202。不得混用兩種 pin count。

## 必須區分的歷史資料

- 初次從可變動的 Google Drive 連結下載到
  `/ldaphome/yyds-tsai-dev/benchmarks/ispd25/visible/` 的部分檔案實際是 **blind**
  corpus；目錄名不能作為 visible 身分證據。相關 export 在
  `results/recovery_20260906/`，與正確 namespace 分開保留。
- 歷史 `results/m4/bench/arrays/` 與 `results/stage2/s8/` 主要是輕量 manifest／
  metrics，原 bulk payload 不在該路徑。本次重新生成結果使用新 namespace，
  不把舊 metrics 配到不同工具版本產生的新座標。
- 舊 H100 SXM 預測仍保留；本次 NVL 預測／共享執行條件另於
  `results/h100_nvl_followup_20260906/forecast/` 凍結。

## OpenROAD

原 contest fork `1a29dadda2c06e2cb2f5b09b8f7425778dd88b47` 是 net/capacity exporter，
不能作正常 GRT。已將完整 `src/grt/` 恢復到最近 upstream base
`c90cd95a8723efb4de67cd8c6efc69f715fc9052`，另保留 Main.cc 的 readline portability guard。

- Exporter binary 保留於 `/ldaphome/yyds-tsai-dev/tools/openroad/prefix/bin/openroad`。
- 可正常 GRT 的 binary 位於 `prefix-upstream-grt/bin/openroad`；
  SHA256 `86b6fc0f0416b742d9ad178422a3e6fbf2fc120fa8c559e58c3a30e784de39d6`。
- GRT restoration diff SHA256：
  `408bd361a91848378a0079f3dff2bbf9dec8fc7fb68a51a8977fb6552fe0e534`。
- Readline guard SHA256：
  `4f826ed1e5f0fb787e1410e1212bb76546e3f641ec0ce256225c960f9dbf358d`。
- `source scripts/env.sh; source scripts/openroad_env.sh` 載入現有工具及本地相依庫。
- 已驗證非空 GRT guide。第一組完整 capped DR 產生559,886個 geometry rows，
  33,307/33,307 signal nets 有 wire；special nets vss/vdd 另列，不當成漏接 signal。
  DRC 仍有殘留，不能標示 signoff-clean。
- Pin geometry adapter 使用 transformed union bbox center，已用八種 orientation、
  asymmetric multi-rectangle pin、長名稱的 native OpenDB fixture 驗證。
- 已修復 Python `dbWire.getLength()` 的 `uint64_t` SWIG typemap；新工具位於
  `prefix-upstream-grt-uint64/bin/openroad`，SHA256
  `8fb20d30744ebcc2b4061434a8c382a80e82a0a16659d08569abf6cf71f48e10`。
  unsigned64 大於2^63的測試保留精確值9223372036854775931。實際 FFT 的
  native wirelength oracle 與1000筆 text-parser 抽樣通過，總長差異0.0587%。
- Native OpenDB 測試也涵蓋一個 BTerm 的多個 asymmetric BPin PORTs：
  使用跨 PORT union bbox center，且僅產生一個 net endpoint。

Stage2 新12組 placement 都已完成，詳細繞線與後續證據位於
`results/stage2_followup_20260906/`。進度及尚未完成項目見 execution-plan。
