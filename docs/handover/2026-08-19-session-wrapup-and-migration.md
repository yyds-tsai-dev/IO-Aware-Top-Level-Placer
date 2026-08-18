# Session 收尾與 H100 搬遷交接(2026-08-19)

- 日期:2026-08-19
- 分支:`m2-differentiable-io`(收尾時 HEAD `51e8eef`),本次收尾後 merge 進 `main`
- 動機:**開發機從 L4(hostname `NVL5`)搬到具 H100 的伺服器**。所有 L4 側能做完的事都已做完;剩下的主線工作(27.7M 全流程、E5/T14 假設檢定)在契約上就必須在 H100 上跑。
- 測試狀態(收尾時實測):`pytest -m "not slow"` **813 passed / 2 skipped**;slow 套件 **35/36 passed**,唯一 FAIL 見 §3.C-0(既有問題,非本次迴歸)。
- 本文件用途:**單一入口**。把三條並行 session 的狀態收斂成一份「現況 + 未完成事項 + 搬遷清單」,不重述已有報告的內容,只指路。

執行細節在既有文件裡,本文件不複製:

| 主題 | 文件 |
|---|---|
| H100 執行契約、rebuild、資料重生成、凍結預測比對 | `docs/handover/h100-30m-runbook.md` |
| M4 全部判定與揭露 | `docs/results/m4-scale-up-report.md` |
| Stage 2 校準與規模天花板 | `docs/results/stage2-calibration-report.md` |
| M3(exit FAIL)與其未決項 N1–N8 | `docs/results/m3-differentiable-ft-report.md` |
| 環境/venv/DREAMPlace commit | `docs/dev-env.md` |

---

## 1. 里程碑收案總表

| 里程碑 | 狀態 | 一句話 | 報告 |
|---|---|---|---|
| M0 / M1 | 完成(已在 `main`) | foundations、evaluator、reweighting 閉環 | `m0-baseline-report.md`、`m1-reweight-report.md` |
| M2 | 完成 | 可微分 IO 項;`ours@M2` 是後續所有臂的基準 | `m2-differentiable-io-report.md` |
| M3 | **收案為 exit FAIL(誠實紀錄)** | 可微分 FT:E1/E5/E6 FAIL、E7 PASS;Phase B ft-reweight 為 null result(<0.6σ)。**`ours@M3` 臂不得進入任何 M4 表格** | `m3-differentiable-ft-report.md` |
| M4 | **T11 完成(2026-08-19);E1/E3/E5 PASS、E2 PASS(T6B 替代路徑)、E4 部分判定;M4-G8 已了結(§3.C)** | 命題成立:GP 在 10M 級不是瓶頸,evaluator 記憶體與 IO op 的 K-pass runtime 才是 | `m4-scale-up-report.md` |
| Stage 2 | **完成,但降級為 OpenROAD-only ground truth** | G1 觸發(Innovus license 三台皆 `No route to host`);頭條發現是 **OpenROAD DR 的規模天花板 ~150k cells** | `stage2-calibration-report.md` |

關鍵數字(細節見各報告,勿在此處引用以外的數字):

- **M4 E1**:`mempool_cluster` 11.3M、seed B=2000,六門檻全過(Δhpwl +1.44%、`device_used_gb` 13.47、單臂 1.9h)。
- **M4 E2**:12.3M 的 H1–H5 holdout **仍 FAIL 且不翻案**,改走 T6B 自洽性驗收(B-0/B-3…B-6 全綠,`scaling_usable=true`;B-1 DiD 0.0034 綠、B-2 1.582 fail 已揭露為非 gating);T9 27.7M 元件級 spike 兩情境皆 `feasible_l4_contract`(18.19 / 18.63 GB)。
- **M4 E4**:T0b 16 點預註冊 factorial 全跑完 → `identifiable=false`(M4-G6 觸發,誠實負結果);exit 文字要求的 evaluator/op 子模型**從未被這輪 T0b 設計觸及**,缺口逐字揭露不代填。
- **M4 E5**:`results/m4/forecast/h100_prediction.json` 已凍結,sha256 `4eeea5f9132e1a06585ae57e38c6e45b5523ed31ae5454826ff4cff56e045284`。
- **Stage 2**:α̂=0.652、β̂=0.966、κ_ft←1.482、R²=0.995(**n=6、3 殘差自由度,信度有限**);15 個排定臂只有 6 個有效,9 個因 congestion/timeout 不可評;per-net 級校準全數 `not_evaluable`(S8 未落盤 evaluator per-net 陣列)。

---

## 2. 三條 session 的收尾狀態(2026-08-19 確認)

| Session | 範圍 | 收尾狀態 |
|---|---|---|
| `add feedthrough objective` | M3 全生命週期、M4 design 與 T0–T6/T8a、Stage 2 規劃 v1(`66d0934..1b3671c`,35 commits) | 已停機待命。工作樹 **零未提交修改**、**零 background process** |
| `comple M4 & stage2` | M4 T6 二次裁決之後全部、T8b/T9/T10/T11、Stage 2 S0–S10 | 已停機待命。未提交的只有實驗產出檔(本次一併處理)、**零 background process**;原懸案 backfill 裁決已由使用者裁 A 並執行(見 §3.C) |
| 本 session(收尾線) | gitignore 整理、產出檔歸檔、本文件、裁決 A 的 backfill、commit + merge | 本次完成;`main` = `5aa98dd`(`--no-ff` 里程碑 merge) |

**Background 全停確認**:`ps` 實測,repo 相關的 route / lifetime / T9 / extract chain 全部結束,無 `run_placement` / `openroad` / probe / rent 行程存活,無 cron、無排程 gate 待觸發。機器上殘留的 `wandb` / `vscode` / MCP 行程屬其他專案。

**Worktree**:`m2-differentiable-io-bg`、`wt-s1-def-export`、`wt-s2-route-parse`、`wt-t8a-instr` 四個分支**全部已是 `m2-differentiable-io` 的祖先**(內容已併入),工作樹本身無獨有的已提交內容。搬遷時可直接 `git worktree remove` + 刪分支,不會丟東西。

---

## 3. 未完成事項(搬遷後的待辦)

### A. 必須在 H100 上做(= 搬遷的動機)

1. **T14 / T10 首跑**:27.7M(3×3、N2)全流程單臂,照 `h100-30m-runbook.md` 走。
   - 前置:CUDA `sm_90` rebuild(configure log 已含,到機驗證即可)、tiler 在目標機重生成 3×3 陣列、T9 tiled-netlist cache 本地重建。
   - 硬體契約:H100 SXM5 80GB、**host RAM ≥ 192 GB**(host RSS 點估 168.9 GB,貼線)、GPU 獨佔、非 MIG。
   - 判定:`scripts/m4_check_prediction.py` 對照凍結 sha `4eeea5f9…`。
   - **紅線**:落在預測帶外時,規則是「**重新登錄預測(發表重擬合)**」,不是事後放寬區間;SKU 與凍結 `sku` 區塊不符時預測直接失效,必須重跑 `scripts/m4_forecast.py` 重凍結。
2. **Stage 2 大 case 繞線重試(需大量 CPU,不需 GPU)**:`matrix_mult_1`(155k congestion plateau)、`superblue19`(506k,第 0 iteration >4h)、`superblue12`(1.29M,未及嘗試)。目標機 CPU 核數更多才值得重試,**每臂預算需 >12h**,且要先接受 §2.2 的規模天花板結論可能不變。

### B. 不需 H100(小工作,隨時可做)

1. **Stage 2 per-net 校準**(本輪最大的方法論缺口):改 driver 落盤 evaluator 的 per-net 陣列(`per_net_crossings`),才能補 C1/E2 的 per-net 對照。
2. **Stage 2 δ-scan**、以及把有效 designs 補到 **≥3**(G3 是使用者決策點:是否為了 C5 再投入)。
3. **Innovus(F-INV)**:license 一通就能重開 S6/S7,把 ground truth 從 OpenROAD-only 升級。
4. **M3 第一順位補格實驗**:「IO 項開 + `ft_rg`→WL `net_weights`」決定性實驗,規格在 M3 報告 §4,`adaptec1` 單臂約 20 分鐘 GPU。
5. **M3 P0c pilot**(S4a-star in-loop,post-registration):規格在 S4 裁決書 §D 與報告 7.3,4 臂 `adaptec1`。
6. **M3 N1–N8 未決項**:P2 home-churn 探針、T0-P3 `boundary_demand` 欄位(S7 重開 gate 目前不可判)、`bigblue4` reweight 臂缺口等,已在 M3 報告逐項列出。

### C-0. **新浮現的未決項:T9 replication 與 PlaceDB 的 node-ID 對不齊(2026-08-19 發現)**

`tests/test_bench_bookshelf_netlist.py::test_replication_equals_placedb_read_on_real_1x2_array_movable_first` 在合併後的樹上**仍然 FAIL**,但**失敗原因已經換了一個**,這才是重點:

- 收尾時跑 slow 套件,此測試是唯一的 FAIL,錯誤是 `FileNotFoundError: benchmarks/ispd25/synthetic_1x2_n2.json`。真因是 `ioplace/netlist.py` 的 `load_netlist` 為了解析 config **內部**的相對路徑而 chdir 到 `$DP/install`,卻在 chdir **之後**才 `params.load(config_json)`,於是呼叫端自己傳的 repo 相對路徑也被拿去 `$DP/install` 下解析。其他呼叫端一律傳絕對路徑,所以從沒踩到。已修:commit `4c9bb46`(chdir 前先 `abspath`)。
- **修掉之後這個測試才第一次真正跑到它要驗的那個斷言,然後掛在斷言上**:`num_movable` / `num_physical` / `num_nets` 三個計數都相等,但逐 net 的 node-ID 集合在第 464,944 個 net 出現差異——`frozenset({6155340})` vs `frozenset({6155980})`,兩邊都是**單 pin net**,且 ID 都 > `num_movable`(6,155,338),也就是**落在 terminal/fixed 區塊**。徵狀指向:replication constructor 與 PlaceDB 對 movable 之後那一段(terminal / terminal_NI / fixed macro)的排序不一致,而測試是用 node ID 比對的。
- **這個不一致在此之前一直被路徑 bug 遮住**,不是本次收尾造成的迴歸。T9 cache 另有一條獨立驗證(對 Bookshelf source 全量比對 0 mismatch,commit `00f2902`),所以**這不必然是資料錯誤,也可能是測試對 ID 順序的要求超出契約**——兩種可能都沒有被排除。
- **下一步**(不需 H100,單機 ~8 分鐘可重現):判定到底是 (a) replication 的 terminal 段排序真的錯了(那會影響 T9/T14 的 27.7M 路徑),還是 (b) 測試該改成以 node **name** 而非 ID 比對。**在 H100 上跑 T14 之前應該先結掉這題**,因為 T14 用的正是同一條 replication 路徑。

### C. M4-G8 / linter backfill —— **已了結(2026-08-19,使用者裁決 A)**

`scripts/m4_report_lint.py --strict` 原本實跑 24 errors:§5.3/§5.4 引用的 24 個 profile JSON 缺少「後定義」的欄位(16 個 quality run 缺 `workload_status`、8 個 synthetic run 缺 `generator_verified`)。**使用者裁決採 A(批准可稽核的 backfill)**,已於 commit `2fd3ed4` 執行:

- `scripts/m4_backfill_result_gate_fields.py` 的目標集合**不是手寫的** —— 它 import `m4_report_lint`,用 linter 自己的表格分類與列解析取檔,因此只碰得到 linter 引用的檔案(解析出 24 筆,與 24 個 error 一對一)。
- **只新增不覆寫**;`workload_status="completed"` 需要正面完成證據(`status=ok` + driver 最後才寫出的收尾欄位;evaluate-only run 用其評估量),拿不出證據就 `refuse` 並整體非零退出。`generator_verified=false` 是事實值,寫 `false` 讓裁決 §7-1 的強制揭露句在 linter rule 3 下持續為必要。
- 事後以 `git show HEAD:<path>` 逐鍵驗證 24 個檔案:**無 key 被刪除、無既有值被改變**;稽核檔 `results/m4/backfill/2026-08-19-result-gate-backfill.json` 記錄逐檔 sha256 前後值。
- 結果:linter `--strict` 轉為 `0 error, 0 warning`,**M4-G8 了結、T11 標記完成**。報告 §9 與附錄 A.4 同時保留 08-18 觸發與 08-19 了結的逐字紀錄。**任何數字、任何 E/G 判定都未改變。**

**遺留(同源缺陷,轉入 §3.B)**:那 8 個 synthetic run 的 `benchmark_kind` 值仍是 `"real"` 而非 `"synthetic"`(driver 呼叫時未帶 `--benchmark-kind synthetic`)。backfill 的「不覆寫既有值」規則讓它沒被動到;根治要在 driver 端補欄位寫出邏輯。不 gating(linter rule 2 只單向禁止 quality 表引用 synthetic run)。

## 4. 搬遷清單

**必須帶走(不可重建 / 重建代價極高)**

- 整個 git repo(含本次 merge 後的 `main`)。
- `results/m2/`、`results/m3/` 的 `.npz`(M2/M3 重現基準,被 `.gitignore` 蓋住,不在 git 內)。
- `results/stage2/s8/` 的繞線產物:6 個有效臂的 route 結果代表 **>12h/臂** 的 OpenROAD 運算。小檔(`metrics.json`、`crossings_k*.json`、`verify_s2.json`、`regions.json`、`coord.json`、`congestion.rpt`)本次已進 git;**大檔(`out.def`、`route.guide`、`netmap.json`、`segments.json`、`drc.rpt`、`verify_wire_length.rpt`)只在磁碟上,要重跑校準就得一起搬**。
- `results/m4/bench/rescue/`(從 `/tmp` 搶救的 T6 重現依據)、`results/m4/bench/verify_group3x3_t6b.json.bak`(B-5 原始 fail 證據)。
- `results/m4/forecast/h100_prediction.json`(凍結預測,T14 的比對基準)。

**不必帶走(目標機重建)**

- `results/m4/bench/t9_cache_3x3/`(4 GB `.npy` cache)——H100 端用 `bookshelf_netlist` 重建,runbook §3.2 有步驟。
- `results/m4/bench/arrays/`、`mempool_group_export/`——tiler 在目標機重生成(runbook §3.1)。
- `.claude/worktrees/`(已全部併入主分支)、`__pycache__`、venv。

**環境**

- DREAMPlace:`DP=…/DREAMPlace`、`dp_commit d971880a15ef684c4a90bccd2c65d62ae7e33297`、`$DP/.venv312/bin/python`(torch 2.8.0+cu128)。目標機要重 build,必查 `CMAKE_CXX_ABI=1` 與 `CUDA_ARCH_FLAGS` 含 `sm_90`(runbook §2)。
- benchmark 語料位置會變:`bb4` 用的是 **repo-local** `benchmarks/ispd2005_bigblue4_m4.json`(內含**絕對** aux 路徑),搬機後必須改路徑。同理檢查所有 `benchmarks/*.json` 的絕對路徑。

---

## 5. 已知暗坑(搬遷後仍然有效)

**實驗/資料**

1. `superseded_iter1000/` 目錄(t8b 與 profile 各一)內的 JSON 帶 `experiment_status=superseded_iteration_capped` — **報告與 linter 一律不得引用**。
2. DREAMPlace 的 aux lexer **拒絕數字開頭的檔名** — 合成 config 一律用字母前綴 alias(3×3 的 `synC` alias 已建)。
3. calib 檔的 `seed_B` 先 1001 後改 2000(跨 session 協議),引用時要說明。
4. `3x3_n1` 建過又刪(N1 `not_run`;seed 0 可位元重現),只記在 commit `ceca6b6` 訊息裡。
5. T6b `count_freeze_small.json` 的三係數模型仍標 `pending_t0b` — T0b 已 fit 完,這是唯一「要嘛回改、要嘛在報告揭露不改」的凍結檔。
6. Sinkhorn `init a_u=1.0` 是裁決未指定時的實作選擇;收斂步數與裁決探針不同但 gated 量全一致(commit `1b4af33` 已揭露,報告 §4 保留該句)。

**工具行為**

7. `openroad -python` 對 `sys.exit(0)` 會印 traceback 且回傳非零 — **判成功要看 artifact 是否存在,不能看 exit code**。
8. `pgrep -f` / `pkill -f` 的 pattern 會匹配到自己 bash 的命令列(自殺 / 死鎖,已中招三次)— background chain 的 gate 改用 `nvidia-smi` 記憶體或 log 標記,且**標記要唯一**(grep 到舊行會提早開 gate)。
9. GPU-mem gate 在目標行程還在 CPU 段時會誤放行(T9 gate race)— 需要 preflight 補救。
10. usage-limit 暫停會**殺掉 background bash chain** — 所有 chain 必須冪等(檔案存在即 skip)才能重啟。
11. 同 repo 兩個 session 並行會互踩(cluster OOM 互殺、verify JSON 寫入衝突、重工)— 開工前先 `ListAgents` 查 peer session,協調檔案分界與 GPU 單一佇列。

---

## 6. 到 H100 後的第一步

1. `git log --oneline -1`(應在本次 merge commit)、`git worktree list` 清掉舊 worktree。
2. 照 `h100-30m-runbook.md` §2 重 build,跑 `$DP/.venv312/bin/python -m pytest`(先 `-m "not slow"`,再全跑)確認移植無損。
3. 修 `benchmarks/*.json` 的絕對路徑。
4. runbook §3 重生成 27.7M 陣列與 T9 cache,**逐步驗證 sha**。
5. 向使用者取得 §3.C 的 backfill 裁決(A 或 B),再決定 T11 是否標記完成。
6. 執行 runbook §4 的單一命令 → §5 的 `m4_check_prediction.py` 比對。**在 H100 上不做任何設計決策。**
