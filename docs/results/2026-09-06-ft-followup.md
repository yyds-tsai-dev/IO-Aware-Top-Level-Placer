# FT follow-up（2026-09-06）

已完成21組預先登錄的實驗，以及全部最終座標的 boundary-demand 重算。
登錄提交為 `6c0938a`；設定與來源快照位於
`results/ft_followup_20260906/manifest.json`。歷史 M3 **FAIL 不變**。

## Signal／lever 篩選

所有組別使用相同 adaptec1、K16/grid、seed1000 與 atomic callback；
A1–A4 構成 signal×lever 比較，A0/A5 是額外控制組。

| Arm | IO term | Reweight signal → target | IO | FT | HPWL |
| --- | --- | --- | ---: | ---: | ---: |
| A0 | 開 | 無 | 24885 | 3577 | 74924025 |
| A1 | 開 | crossings → WL | 22671 | 3043 | 75853268 |
| A2 | 開 | FT_rg → WL | 23564 | 2985 | 75018274 |
| A3 | 開 | crossings → IO weights | 24739 | 3611 | 74942672 |
| A4 | 開 | FT_rg → IO weights | 24787 | 3578 | 74978812 |
| A5 | 關 | crossings → WL | 29954 | 1890 | 74718953 |

A2−A0 的三組配對確認種子為1001–1003；負值表示 A2 較低。

| 指標 | 三組配對差值 | 平均差 | 單側95%上界 |
| --- | --- | ---: | ---: |
| IO | −1171、−1027、−1105 | −1101 | −979.478 |
| FT | −604、−599、−600 | −601 | −596.540 |
| HPWL | +200600、+83717、+157368 | +147228.333 | +246858.175 |

這支持 adaptec1 上相對 IO-only 的 IO／FT 改善；HPWL 平均上升，
FT 仍高於 flat baseline，且尚無跨設計泛化證據。五個 flat seeds 的
IO 平均30024.6、標準差77.99；FT 平均2545、標準差26.47；
HPWL 平均73930872.8、標準差49830.53。

## P0c pilot

| Pilot | f_ft_max | IO | FT | HPWL | PG2 |
| --- | ---: | ---: | ---: | ---: | --- |
| P0 | 0 | 24885 | 3577 | 74924025 | 控制組 |
| P1 | .1 | 25867 | 3223 | 74637807 | FAIL |
| P2 | .25 | 26618 | 2947 | 74394124 | FAIL |
| P3 | .5 | 27237 | 2814 | 74255113 | FAIL |

P1/P2/P3 的 IO 增量分別為982、1733、2352，均超過預先登錄的492上限。
因此沒有符合條件的 P0c confirmation，也不執行 P4；這是 pilot 的負面
結果，與 A2 的 signal／lever 確認實驗分開判讀。PG4 報告的是最後 HPWL
變化及已觀察到的收斂／rollback 記錄，不宣稱取得逐 iteration HPWL trace。

## Boundary demand 與 seed noise

[Boundary probes](../../results/ft_followup_20260906/boundary_probes.json)
在21組凍結最終座標上重算，IO／FT／HPWL 與原紀錄一致。A2−A0 的結果：

| 指標 | 平均差 | 單側95%上界 |
| --- | ---: | ---: |
| Boundary demand 總量 | −1101 | −979.478 |
| Boundary 最大值 | −194 | −96.453 |
| Boundary p90 | −42.133 | −8.051 |
| Boundary Gini | +.007801 | +.013793 |
| 長度正規化最大值 | −.072659 | −.036125 |

總量與尖峰下降，但 Gini 上升，不能宣稱負載更均勻。五個 flat seeds 的
boundary max／Gini 標準差為14.1669／.00124909；長度正規化後為
.00530594／.00124424。這些是診斷資料，沒有依觀察結果新增驗收門檻。

[完整統計](../../results/ft_followup_20260906/summary.json)、
[圖表 PNG](figs/ft-followup-20260906.png)、[圖表 PDF](figs/ft-followup-20260906.pdf)。
