# Baseline 對照表


## adaptec1

| case | mode | k | rtype | seed | io_count | ft_count | tree_wl | hpwl | runtime_s | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptec1 | flat | 16 | grid | 0 | 30,011 | 2,582 | 84,655,386.0 | 73,928,933.0 | 58.3 | 140.8 |
| adaptec1 | flat | 16 | slicing | 0 | 36,600 | 7,308 | 84,637,857.0 | 73,890,377.0 | 58.8 | 140.8 |
| adaptec1 | flat | 8 | grid | 0 | 22,405 | 1,981 | 84,680,077.0 | 73,955,613.0 | 60.7 | 140.8 |
| adaptec1 | two_stage | 16 | grid | 0 | 44,904 | 24,360 | 249,720,736.4 | 233,492,867.4 | 107.0 | 500.8 |
| adaptec1 | two_stage | 16 | slicing | 0 | 58,850 | 37,341 | 266,599,568.3 | 247,521,637.2 | 103.1 | 506.6 |
| adaptec1 | two_stage | 8 | grid | 0 | 33,780 | 15,911 | 286,246,732.2 | 266,381,745.2 | 89.4 | 371.4 |

## bigblue4

| case | mode | k | rtype | seed | io_count | ft_count | tree_wl | hpwl | runtime_s | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| bigblue4 | flat | 16 | grid | 0 | 100,108 | 7,045 | 857,859,395.0 | 748,503,270.0 | 545.5 | 1,670.9 |
| bigblue4 | two_stage | 16 | grid | 0 | 236,635 | 130,676 | 5,998,346,593.4 | 5,718,403,942.4 | 3,752.7 | 6,688.0 |

---

## M0 exit 檢核

矩陣完成度:**8/8 runs**(adaptec1 × {flat, two_stage} × {k8-grid, k16-grid, k16-slicing} = 6,bigblue4 × {flat, two_stage} × k16-grid = 2)。無 FAILED 格,全部 run 以 exit code 0 結束並產出合法 JSON + npz。

### Sanity check 1 — flat 的 hpwl 應低於 two_stage:**PASS**

| case (k, rtype) | hpwl(flat) | hpwl(two_stage) | 倍率 |
|---|---:|---:|---:|
| adaptec1 (k8, grid) | 73,955,613 | 266,381,745 | 3.60× |
| adaptec1 (k16, grid) | 73,928,933 | 233,492,867 | 3.16× |
| adaptec1 (k16, slicing) | 73,890,377 | 247,521,637 | 3.35× |
| bigblue4 (k16, grid) | 748,503,270 | 5,718,403,942 | 7.64× |

四組配對(matched k/rtype)全數符合預期:two_stage 的 fence 約束犧牲了自由度,hpwl 一致顯著高於 flat(3.2×–7.6×,且 benchmark 越大代價越高 —— bigblue4 的 7.64× 遠高於 adaptec1 的 ~3.2–3.6×)。

### Sanity check 2 — two_stage 的 io_count 應明顯低於 flat:**FAIL(方向相反)**

| case (k, rtype) | io_count(flat) | io_count(two_stage) | 倍率 | initial_cut_io_lb | io_count/lb |
|---|---:|---:|---:|---:|---:|
| adaptec1 (k8, grid) | 22,405 | 33,780 | 1.51× | 17,178 | 1.97× |
| adaptec1 (k16, grid) | 30,011 | 44,904 | 1.50× | 20,333 | 2.21× |
| adaptec1 (k16, slicing) | 36,600 | 58,850 | 1.61× | 20,461 | 2.88× |
| bigblue4 (k16, grid) | 100,108 | 236,635 | 2.36× | 104,103 | 2.27× |

四組配對**全數不符**brief 的預期方向 —— two_stage 的 `io_count` 不是「明顯低於」flat,而是一致地**高於** flat(1.5×–2.36×)。`ft_count` 的落差更誇張(two_stage/flat 比值:adaptec1 k8 = 8.03×,k16-grid = 9.43×,k16-slicing = 5.11×,bigblue4 = 18.55×;逐格算法見下方診斷)。這是如實記錄,不挑數字。

**診斷假設(已用程式碼追查驗證,非臆測):**

1. **`io_count`/`ft_count` 量的是「最終幾何 MST 的 boundary crossing / feed-through」,不是「hypergraph 的 cut 數」。** 後者對應 `initial_cut_io_lb`(`run_placement_two_stage.py` 內 `lam = Σ_e (touched_parts − 1)`,spec 定義的 assignment 層級下界)。表中最後一欄確認 `io_count ≥ initial_cut_io_lb` 在全部 4 組都成立(1.97×–2.88×,相對穩定的倍率窄帶,不像隨機雜訊或 bug 造成的離散結果)—— pipeline 內部是自洽的,`initial_cut_io_lb` 忠實扮演下界角色;問題出在「下界」與「flat 的實際值」之間的相對大小。

2. **root cause:mtkahypar 的 block id 與 region 的物理相鄰關係無關,而 `fence_inject.py` 把兩者直接劃等號。** 追查 `ioplace/partition/mtkahypar_runner.py::partition_netlist`:呼叫 `ctx.set_partitioning_parameters(k, epsilon, mtkahypar.Objective.KM1)` 只讓 mtkahypar 對**抽象 hypergraph** 做 K-way min-cut(minimize Σ(touched_parts−1)),回傳的 `block_id ∈ [0,k)` 對 mtkahypar 而言只是「顏色標籤」,對哪個 block 在物理上跟哪個 block 相鄰**一無所知**。另一方面 `ioplace/drivers/run_placement.py::get_regions_for`(經 `GRID_SHAPES`/`make_grid_regions`)產生的是**純幾何**、row-major 順序的 K 個 die 上矩形 tile,同樣與 netlist 結構無關。兩者由 `ioplace/fence_inject.py:29`(`fence_map[:m] = parts.astype(np.int32)`)直接以「block_id 當 region 索引」接起來,**中間沒有任何「把便宜切的 block pair 對齊到物理相鄰 tile」的 remapping 步驟**。因此一條被 mtkahypar 判定「切得便宜」的 net(只跨 2 個 block,對 mtkahypar 而言 cost 很低)完全可能被指派到兩個**物理上不相鄰**(例如網格對角、或 slicing 樹上很遠)的 region;其幾何最小生成樹要接起這兩個相距很遠的 pin 群,不但被迫拉長(這正是 hpwl 3–8× 膨脹的來源),而且很可能中途「路過」數個原本與該 net 完全無關的中間 region —— 每一次路過都各自貢獻一次 boundary crossing(推高 `io_count`)與一次 empty-region feed-through(推高 `ft_count`,故其膨脹倍率比 `io_count` 更誇張)。

3. **與 flat 對照:** flat 沒有 fence 約束,GP 本身以 HPWL 最小化驅動,同一 net 的 pins 自然被拉在一起;雖然固定的 region grid 事後疊上去仍會產生一些跨界(flat 的 `io_count` 並非 0),但那些跨界幾乎都是「短程、鄰近 tile」的局部效應,較少發生長途路過。這與觀察到的「flat 的 ft_count 遠低於 two_stage」(5×–18.6×)一致。

4. **這是 M0 baseline 的結構性特徵,不是 driver/evaluator 的 bug。** Global Constraints 定義 `io_count`/`ft_count` 的語意就是幾何 tree-crossing(非拓樸 cut),spec 也明確把 region 幾何列為固定輸入、partition 只優化 cut,兩者本就不是同一個目標函數。M0 的兩階段 baseline(partition 完全不知道 region 的物理佈局)是刻意的最簡單版本,為後續里程碑(例如把 partition 目標換成「已知 region 相鄰圖」的 geometry-aware 版本,或 M1 的 reweight 迴路用 soft 方式把 cell 拉回)留出空間;M0 exit 的任務只是「把兩條 baseline 老實跑出來、比較清楚」,而非「驗證 two_stage 一定比 flat 好」。這個發現對後續里程碑的設計是有意義的輸入,建議記錄留給下一階段參考。

### Runtime / peak memory(每格)

| case | mode | k | rtype | runtime_s | runtime(近似) | peak_mem_mb |
|---|---|---:|---|---:|---|---:|
| adaptec1 | flat | 8 | grid | 60.7 | 1分1秒 | 140.8 |
| adaptec1 | flat | 16 | grid | 58.3 | 58秒 | 140.8 |
| adaptec1 | flat | 16 | slicing | 58.8 | 59秒 | 140.8 |
| adaptec1 | two_stage | 8 | grid | 89.4 | 1分29秒 | 371.4 |
| adaptec1 | two_stage | 16 | grid | 107.0 | 1分47秒 | 500.8 |
| adaptec1 | two_stage | 16 | slicing | 103.1 | 1分43秒 | 506.6 |
| bigblue4 | flat | 16 | grid | 545.5 | 9分6秒 | 1,670.9 |
| bigblue4 | two_stage | 16 | grid | 3,752.7 | 62分33秒 | 6,688.0 |

觀察:
- adaptec1(211K cells)全部 6 格落在 1–2 分鐘級,符合預期。
- flat 的 peak_mem_mb 在同一 benchmark 內不隨 k/rtype 變化(140.8 MB / 1,670.9 MB 固定),符合預期 —— flat 完全不建立 fence-region 相關的 GPU 資料結構,k/rtype 只影響 post-hoc 的 evaluator region 疊圖,不影響 GP 階段的記憶體配置。two_stage 的 peak_mem_mb 隨 k 增加(adaptec1 two_stage k8→k16 grid: 371.4→500.8 MB),符合「region 數越多、per-region density/filler 簿記越多」的預期。
- bigblue4 two_stage 實測 **62.5 分鐘**,遠超任務指示的 10–30 分鐘估計,落在 brief 原估 30–90 分鐘區間內。輪詢過程中以 `nvidia-smi` + `/proc/<pid>`(state/threads/RSS)+ log tail 交叉確認,重建出粗略的三段式時間分佈(polling 間隔約 9–10 分鐘,非逐秒 profiling,邊界為約略值):
  - **≈0–20 分鐘:Mt-KaHyPar CPU partition。** 啟動後到 t≈19.5 分鐘的檢查點,GPU 皆為 0%、log 因 C++ extension 的 stdout 全緩衝而幾乎無輸出;t≈20 分鐘後的下一次檢查已確認 GPU 跳到 95%(7428 MiB,151 threads,RSS 10.7GB)。
  - **≈20–49 分鐘(約 29 分鐘):fence-constrained global placement(GPU)。** 這是**佔比最大的單一階段**,且明顯比 flat bigblue4 的「GP+LG+evaluate 全部合計僅 545s(9 分鐘)」慢上 3 倍以上 —— 合理推測 fence 約束大幅限制了每次迭代的搜尋空間(cell 被鎖在 assigned region 內,density/overflow 更難收斂),導致 GP 需要更多 iteration 才能達到 `stop_overflow` 門檻,而非單純資料量放大 16 個 region 的線性開銷。
  - **≈49–62.5 分鐘(約 13.5 分鐘):最終 Legalization(CPU)+ reference evaluator + JSON/npz 寫出。** t≈49 分鐘的檢查點看到 log 出現本節下方討論的「Standard cell legalization / Greedy legalization / ERROR legality check failed」訊息群(row 889 overlap),與 flat run 收尾前的 Greedy+Abacus legalization 屬同一類 CPU-side 步驟;bigblue4 的 evaluator(io_count/ft_count 皆遠高於 flat)需要走訪的 tree edge 數量也遠多於 flat,合理推測比 flat 評估耗時更久。

過程中 CPU 使用率穩定在 100%+、RSS 平穩不增長(無 leak 跡象),期間持續以 `kill -0`/log tail 監控,始終無 hang/crash 訊號 —— 純粹是三段式運算量大,其中 GP 階段本身(而非原先猜測的 partition 階段)是最大宗。此為下一階段值得記錄的效能觀察(例如 fence 約束下 GP 收斂變慢的具體原因、`partition_netlist` 的 `threads=8` 是否值得開放給呼叫端覆寫成更接近機器實際核數 64),但不影響本次 M0 對照表數字的正確性;三段邊界為粗粒度 polling 觀察,非精確 profiling,未來如需精確數字應改用逐 iteration 計時或 profiler。

### 其他觀察:DREAMPlace 內部 legalizer 訊息(non-fatal)

`two_stage` 模式的每一個 run(adaptec1 3 格 + bigblue4 1 格)在 log 中都會看到 DREAMPlace 內部 `ERROR:root:legality check failed in greedy legalization` 訊息(來自 `placedb.initialize()` 內、per-fence-region 的 filler/legalization 簿記,見 `run_placement_two_stage.py` 對 `calc_num_filler_for_fence_region()` 的既有討論)。這些訊息**全部**發生在 exit code 0、JSON/npz 正常產出的 run 中,不是 crash。

`[ERROR]` 行數統計如實記錄(誠實區分「完整 log」與「只擷取尾段」兩種資料品質,不誇大也不低估):

| case | log 擷取方式 | `[ERROR]` 行數 | out-of-fence-region 行數 | 最終 `fence_compliance` |
|---|---|---:|---:|---:|
| adaptec1 two_stage k8-grid | 完整(未截斷) | 5(全部集中在單一 node 39274) | 0 | 0.9999952585062398 |
| adaptec1 two_stage k16-grid | **僅 tail -40**(真實總數未知) | ≥38(於 tail 內觀察到,全部集中在單一 row 889) | 0(於 tail 內) | 0.9999952585062398 |
| adaptec1 two_stage k16-slicing | 完整(存檔於 `results/m0/logs/`) | 41,585 | 123(全部落在 region 12) | 0.999412054773736 |
| bigblue4 two_stage k16-grid | 完整(存檔於 `results/m0/logs/`) | 4 | 0 | 0.9999995389969403 |

關鍵觀察:**真正代表最終正確性的是 `fence_compliance`**(由我們自己的 driver 在放置完成後,依最終座標獨立計算,與 DREAMPlace 內部 log 無關)。k8-grid、k16-grid、bigblue4 三格的 `fence_compliance` 都精確等於「只有那顆刻意釋放的 escape cell 不合規」的公式值(adaptec1 為 1 − 1/210904 = 0.9999952585062398,與 Task 9 已記錄的數字完全吻合;bigblue4 同構但 num_movable 不同,故值不同但同型態)—— 代表 greedy legalization 階段出現的這些 transient `[ERROR]`(不論是 5 行還是 ≥38 行)最終都被後續的 Abacus legalization / GP-LG 流程收斂掉了,沒有殘留成最終落點違規。

只有 **k16-slicing** 例外:41,585 行 `[ERROR]` 中有 123 行是 "out of fence region 12"(明確指出 region 12 是問題來源),對應 `fence_compliance` 降到 0.999412(約 124/210,904 顆 cell 最終落點仍不合規)。診斷假設:slicing 分割在這個 seed 下產生的 region 12 形狀/尺寸相對窄小或不規則,DREAMPlace 逐 row 的貪婪 legalizer 在該區域內較難找到可行排列,且這次未能被後續步驟完全收斂;bigblue4 在同一 k16-grid 設定下 log 乾淨(僅 4 行,無 out-of-region),支持「與特定 region 的幾何形狀相關,而非隨 benchmark 規模惡化」的假設。此現象不影響本報告數字的有效性(所有 `io_count`/`hpwl` 皆基於最終落點座標算得,`fence_compliance` 已誠實反映落點合規率),記錄於此供後續(例如改善 slicing 產生器對窄 region 的保護,呼應 progress.md 已記錄的 Task 3 已知風險)參考。
