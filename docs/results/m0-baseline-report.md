# Baseline 對照表


## adaptec1

| case | mode | k | rtype | seed | io_count | ft_count | tree_wl | hpwl | runtime_s | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptec1 | flat | 16 | grid | 0 | 30,011 | 2,582 | 84,655,386.0 | 73,928,933.0 | 58.3 | 140.8 |
| adaptec1 | flat | 16 | slicing | 0 | 36,600 | 7,308 | 84,637,857.0 | 73,890,377.0 | 58.8 | 140.8 |
| adaptec1 | flat | 8 | grid | 0 | 22,405 | 1,981 | 84,680,077.0 | 73,955,613.0 | 60.7 | 140.8 |
| adaptec1 | two_stage | 16 | grid | 0 | 36,783 | 16,494 | 243,933,433.2 | 226,599,966.2 | 106.5 | 501.2 |
| adaptec1 | two_stage | 16 | slicing | 0 | 50,089 | 26,201 | 260,501,940.3 | 240,057,731.6 | 111.1 | 503.4 |
| adaptec1 | two_stage | 8 | grid | 0 | 28,266 | 12,282 | 240,420,189.9 | 224,371,075.9 | 89.5 | 369.8 |

## bigblue4

| case | mode | k | rtype | seed | io_count | ft_count | tree_wl | hpwl | runtime_s | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| bigblue4 | flat | 16 | grid | 0 | 100,108 | 7,045 | 857,859,395.0 | 748,503,270.0 | 545.5 | 1,670.9 |
| bigblue4 | two_stage | 16 | grid | 0 | 178,594 | 77,899 | 4,693,332,230.1 | 4,459,824,739.1 | 3,738.2 | 6,686.1 |

---

## M0 exit 檢核

### baseline 修正

修正內容:two-stage baseline 原本把 mtkahypar 回傳的 partition block id 直接當作幾何 region id 注入 fence(`fence_inject.py` 內 `fence_map[:m] = parts`),block id 與 region 的物理相鄰關係毫無對應;現改由新增的 `assign_blocks_to_regions()`(`ioplace/drivers/run_placement_two_stage.py`,貪婪初始指派 + 2-opt pairwise region swap,依 block-pair connectivity `C` 與 region 中心 Manhattan 距離 `D` 做指派)先把高連通度的 block 對齊到幾何相鄰的 region,再重跑受影響的 4 個 two_stage 格。

修正前後對照(舊值取自 `git show e8f0824:docs/results/m0-baseline-report.md`;新值取自本次重跑的 JSON,時間戳 17:05–18:12):

| case (k, rtype) | io_count 舊→新 | ft_count 舊→新 | hpwl 舊→新 | runtime_s 舊→新 |
|---|---:|---:|---:|---:|
| adaptec1 (k8, grid) | 33,780 → 28,266(0.84×) | 15,911 → 12,282(0.77×) | 266,381,745.2 → 224,371,075.9(0.84×) | 89.4 → 89.5(1.00×) |
| adaptec1 (k16, grid) | 44,904 → 36,783(0.82×) | 24,360 → 16,494(0.68×) | 233,492,867.4 → 226,599,966.2(0.97×) | 107.0 → 106.5(1.00×) |
| adaptec1 (k16, slicing) | 58,850 → 50,089(0.85×) | 37,341 → 26,201(0.70×) | 247,521,637.2 → 240,057,731.6(0.97×) | 103.1 → 111.1(1.08×) |
| bigblue4 (k16, grid) | 236,635 → 178,594(0.75×) | 130,676 → 77,899(0.60×) | 5,718,403,942.4 → 4,459,824,739.1(0.78×) | 3,752.7 → 3,738.2(1.00×) |

四格全數下降(io 0.75×–0.85×,ft 0.60×–0.77×,hpwl 0.78×–0.97×),方向與「幾何相鄰對齊減少不必要繞路」的修正意圖一致,bigblue4/adaptec1-k16-grid 的 ft_count 降幅最大(0.60×/0.68×)。runtime 三格持平(1.00×),k16-slicing 上升到 1.08×,兩者皆未進一步歸因根因。

### sanity 重評

**Sanity 1 — hpwl(flat) < hpwl(two_stage):仍 PASS。**

| case (k, rtype) | hpwl(flat) | hpwl(two_stage) 新 | 倍率(新) | 倍率(舊) |
|---|---:|---:|---:|---:|
| adaptec1 (k8, grid) | 73,955,613 | 224,371,075.9 | 3.03× | 3.60× |
| adaptec1 (k16, grid) | 73,928,933 | 226,599,966.2 | 3.07× | 3.16× |
| adaptec1 (k16, slicing) | 73,890,377 | 240,057,731.6 | 3.25× | 3.35× |
| bigblue4 (k16, grid) | 748,503,270 | 4,459,824,739.1 | 5.96× | 7.64× |

四格 flat 的 hpwl 仍全部低於 two_stage,倍率全數縮小(修正減少了幾何繞路造成的額外膨脹),但 fence 約束犧牲自由度的結構性代價依然存在,方向未變。

**Sanity 2 — io_count(two_stage) 應明顯低於 flat:仍 FAIL(方向相反),如實記錄。**

| case (k, rtype) | io_count(flat) | io_count(two_stage) 新 | 倍率(新) | 倍率(舊) | initial_cut_io_lb 新 | lb/io(flat) |
|---|---:|---:|---:|---:|---:|---:|
| adaptec1 (k8, grid) | 22,405 | 28,266 | 1.26× | 1.51× | 15,736 | 0.70× |
| adaptec1 (k16, grid) | 30,011 | 36,783 | 1.23× | 1.50× | 19,594 | 0.65× |
| adaptec1 (k16, slicing) | 36,600 | 50,089 | 1.37× | 1.61× | 20,214 | 0.55× |
| bigblue4 (k16, grid) | 100,108 | 178,594 | 1.78× | 2.36× | 98,847 | 0.99× |

四格 two_stage 的 io_count 依然高於 flat(方向不變,如實記錄,不挑數字),但倍率全數從 1.50×–2.36× 收斂到 1.23×–1.78×,與 baseline 修正後 ft_count/hpwl 同步改善的方向一致。

觀察(min-cut 的 λ−1 下界 vs 幾何 crossing 的關係):四格修正後的 `initial_cut_io_lb`(assignment 本身 Σ_e(touched_parts−1) 的下界,語意上是 hypergraph cut 量,與 block↔region 的幾何重排無關——`assign_blocks_to_regions` 對 `parts` 做的是 block id → region id 的 bijective relabeling,任一 net 觸及的相異 id 數在 bijection 下不變,理論上不可能改變這個下界)本身在四格**全部低於** flat 的 io_count(lb/io(flat) = 0.55×–0.99×,bigblue4 最接近打平,僅差 1.3%)——代表如果幾何 MST 能完美貼合這個下界,two_stage 理論上該打平甚至贏過 flat。但實際 io_count/lb 比值仍有 1.80×–2.48×(雖然已比修正前的 1.97×–2.88× 收斂;修正前後的逐格 io/lb 倍率:adaptec1 k8 1.97×→1.80×,k16-grid 2.21×→1.88×,k16-slicing 2.88×→2.48×,bigblue4 2.27×→1.81×),說明目前擋住 two_stage 追平 flat 的主因不是 partition 本身切得不夠好(它的下界已經比 flat 低),而是「把 hypergraph cut 結果轉成幾何 MST crossing」這一步仍有相當損耗——即使 block 已對齊到相鄰 region,region 內部的實際落點與 MST 路徑仍可能繞路。

**不確定處明說:** 四格修正後的 `initial_cut_io_lb` 本身也比修正前的舊值低(例如 bigblue4 從 104,103 降到 98,847,adaptec1 三格同向下降)。但如上所述,`assign_blocks_to_regions` 的 bijective relabeling 數學上不可能改變 Σ(touched_parts−1),所以這個下界的變動不可能來自這次修正本身,只能代表兩次 run 底層 mtkahypar 回傳的 partition(block 切法)並不相同。`ioplace/partition/mtkahypar_runner.py::partition_netlist` 呼叫 `mtkahypar.set_seed(seed)` 固定 seed=0,但 `threads=8` 走 TBB 平行的 multilevel coarsening/refinement——平行 hypergraph partitioner(含 Mt-KaHyPar 的 default preset)在多執行緒下普遍不保證 bit-reproducible,即使 seed 固定,這是已知特性但本次未獨立重跑 mtkahypar 隔離驗證。也就是說,上面兩張表「修正前 vs 修正後」的數字差異,除了 `assign_blocks_to_regions` 本身的效果外,還混入了「重跑時 mtkahypar 恰好切出不同 partition」的變異,兩者的相對貢獻本次未拆分——如需精確歸因,需在同一份固定的 mtkahypar 輸出上分別跑「修正前 vs 修正後」的 region 指派做控制實驗,本次沒有做這個隔離。

### legality 判讀

`results/m0/logs/bigblue4_two_stage_k16_grid.rerun.log` 尾端的 4 行 `[ERROR]` 都指向同一對 node 的 row-overlap:`grep -o "overlap node [0-9]*" results/m0/logs/bigblue4_two_stage_k16_grid.rerun.log | sort -u` 只得到單一結果 `overlap node 2169386`,且每一行的 `with node` 都是 `node 1`——即 movable node 1 與 fixed macro node 2169386 重疊(2169386 > num_movable_nodes,落在 terminal/macro 範圍,與 fixed macro 的定位一致)。(a) `grep -c "overlap node" results/m0/logs/bigblue4_flat_k16_grid.log` 回傳 0(exit 1,無匹配)——flat 的合法化過程完全沒有這類 overlap,代表這**不是**「benchmark 本身 fixed macro 互相重疊」的通用特性,而是 two_stage/fence 路徑特有的現象。(b) 涉及的 node id 就只有這一對,如上。(c) 該 JSON 的 `fence_compliance = 0.9999995389969403`,驗證後這個值在 float64 精度下 bit-exact 等於 `1 − 1/2169183`(即 2,169,183 顆 movable cell 中恰好 1 顆不合規)——與 `run_placement_two_stage.py::_pick_escape_cell` 刻意釋放單一 escape-valve cell 使其不受 fence 約束的機制完全吻合,是相當有力的旁證。額外交叉檢查(非任務要求但免費且加強證據):同一對 node id(movable 1 / fixed 2169386)也出現在修正前、較早的 `results/m0/logs/bigblue4_two_stage_k16_grid.log`(16:35,同一種 overlap、相近 row 範圍),即這個 escape cell 的身份在兩次 run(底層 mtkahypar partition 疑似不同,見上一節)之間保持一致,與「node 1 的面積在 movable cell 中接近全域最小、幾乎必然落在非 singleton block 內,因此 `_pick_escape_cell` 的 argmin-area 選擇每次都會挑到它」的假設相符;但本次沒有在程式碼中插入 print 直接確認 `escape_idx == 1`,所以「node 1 就是 escape cell」仍是高度支持、但未經直接程式碼驗證的推論,在此明確標注為不確定。
