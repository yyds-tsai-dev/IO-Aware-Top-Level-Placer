# MtKaHyPar native runtime 與本次驗證條件

安裝的 `mtkahypar==1.6.2` 在多執行緒 recursive Rent partitioning 間歇性
SIGSEGV。移除 OpenROAD 的 LD_LIBRARY_PATH、保留 initialize 回傳物件、
限制 OpenBLAS 單執行緒都無法排除崩潰；不能將它歸因於這三項因素。

GDB 捕捉的 native stack 是 private `libtbbmalloc` → scalable_malloc →
TBB allocate_memory → ConcurrentBucketMap reserve → StaticHypergraph contract
→ multilevel coarsening。這定位到原生 coarsening/allocation 路徑，
但不足以區分 allocator defect 或較早發生的 memory corruption。

[v1.6.2 Python binding](https://github.com/kahypar/mt-kahypar/blob/v1.6.2/python/module.cpp)
的 initialize 本來就使用 atomic once guard，回傳空 token；單純快取 owner
並非本次崩潰的修復。Wheel extension SHA256：
`20f567077dd66b8fa24658efdd7927177bb634dd80ee2ad83f18aee12b5e2c24`。

## 已測試的緩解措施

明確指定 `IOPLACE_MTKAHYPAR_THREADS=1`。Native 單執行緒已通過8個 Rent
tests、額外20次 torus 與20次1024-node/64-degree random-graph recursive
measurements，正常退出。多執行緒偶爾也會通過，單次 green run 不構成修復證據。

兩個 production callers（partition runner 與 Rent）現在共用 runtime helper。
它記錄當次 requested threads、override、first requested 與實際生效值；
upstream 只能初始化一次，因此初始化後更改不同的 explicit override 會報錯。
沒有改用 geometric backend，也沒有改 preset、KM1 objective 或 seeds。

```bash
source scripts/env.sh
source scripts/openroad_env.sh
ulimit -c 0
IOPLACE_MTKAHYPAR_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$IOPLACE_PYTHON" -m pytest -q -m 'not slow'
```

最初完整結果為968 passed、1 skipped、38 deselected；最新整合後結果為
978 passed、1 skipped、39 deselected，兩者皆exit0。另有 fresh subprocess
regression 先以 requested8 執行 partition，再以 requested4 執行三次 Rent，
確認 effective1、合法 assignment 與原本 exponent assertions。該組 targeted
測試共18 passed。Logs 與 hashes 保存在 `results/validation_20260906/`。

這是可重現的緩解措施；原生多執行緒缺陷尚未完成根因修復。歷史多執行緒
結果保留原始provenance。另一次driver suite exit134定位為DREAMPlace內建
ISPD2015相對路徑缺少benchmark alias；補上既有資料的symlink後，DEF export／
reweight尾段12 passed正常退出。Driver lifecycle integration亦35 passed。
此資料路徑問題與MtKaHyPar多執行緒缺陷分開記錄；測試群組有重疊。
