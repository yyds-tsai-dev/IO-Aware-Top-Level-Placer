# Routing-aware evaluator:兩個新方向(2026-08-13,使用者指示)

M2 收尾時使用者提出兩個規劃項,記錄於此作為 M3 design 與 Phase 2 規劃的正式輸入。

## 方向 A(Stage 1 / Phase 1,M3 design 輸入):MST detour 問題

**問題陳述。** `evaluator_ref`/`evaluator_gpu` 量測的是 per-net **MST 幾何段**跨
region boundary 的次數。MST 只最小化樹長,不理會 region 邊界,因此有機會穿過
partition region 密集的區域,把 io_count 灌高。但本專案是 IO-aware placement:
既然 objective 願意犧牲 wirelength 換更少 crossing,「正確」的 routing 模型也
應該允許 route longer path 繞開 region 密集區 —— evaluator 應量測
**crossing-minimal routing** 下的 crossing 數,而非幾何 MST 的 crossing 數。

**既有證據。** M2 alignment 診斷的 detour(`io_count − Σ(Λ−1)`)正是這個效應的
量化:adaptec1 k16 約 26%、bigblue4 k16 約 16% 的實測 io_count 來自「繞路/幾何」
而非 region-presence 下界(design §3.2.5 的份額表)。也就是說 evaluator 目前
系統性高估了可實現的 IO count,而且高估幅度隨 placement 幾何變動 —— 這會污染
alignment 診斷(Spearman、F4a)與跨臂比較。

**與 placer objective 的關係。** S2 surrogate 的 `λ_e = |touched regions| − 1`
本來就是 routing-independent 的 crossing 下界(net 碰到 k 個 region,無論怎麼
繞至少 k−1 次 crossing)。所以 placer 已在優化「對的東西」;要修的是 evaluator
的量測模型。修正後預期 exit 指標的絕對數字下降、alignment 變乾淨。

**候選 formulation(M3 design 由 deep-reasoner 正式裁決):**

1. **Region-adjacency Steiner**:net 的 crossing-minimal 值 = 在 region adjacency
   graph(K 個節點,K≈4–32 的小圖)上連接該 net touched regions 的 Steiner tree
   邊數;不相鄰的 region 之間要穿過中間 region,自然計入。per-net 是 O(K) 級
   小圖問題,10M-30M net 規模可 GPU 批次化。下界緊,但忽略幾何容量(所有 net
   都繞同一個 gap 是否可行是 Phase 2 congestion 的事)。
2. **Boundary-cost maze routing**:對每條 tree edge 在 cost map(crossing=1、
   length=ε)上跑 A*/maze,量測「幾何可實現」的 min-crossing path。更精確、
   更貴;可作為抽樣校驗而非全量 evaluator。
3. **混合**:全量用 (1) 的 region-graph 值 + 抽樣用 (2) 校驗 (1) 的樂觀誤差。

**注意 FT 的耦合。** pure feed-through(net 穿過一個它沒有 pin 的 region)的
計數同樣依賴 routing 模型:crossing-minimal routing 會改變哪些 region 被穿過。
M3 本來就要做可微 FT 項(S4+S7),evaluator 的 routing 模型必須在 M3 design
一併定案,否則 FT 的 objective 與 evaluator 又會出現 M2 修掉的那種口徑錯位。

## 方向 B(Stage 2 / Phase 2):commercial-tool ground truth 校準

**新 objective(使用者 2026-08-13 指示)。** 我們的 placer 跑完 global placement
+ legalization 後,用 Innovus 等 commercial EDA tool 跑 detailed placement +
routing,從真實 route 結果抽取 partition-boundary crossing 數,與我們的
evaluator 對照,把 evaluator 校準(calibrate)到更精確。

**具體工作項(Phase 2 規劃,M4 之後):**

1. 輸出流程:placer 結果 → DEF(含 region/fence 標註),LEF/DEF 與 Innovus
   flow 的銜接。
2. Innovus 跑 DP + route(需要 license 與環境;benchmark 用 ISPD 系列的
   LEF/DEF 版本或 MemPool/TeraPool 家族)。
3. crossing 抽取:解析 routed DEF/route DB,計 per-net wire 跨 region boundary
   次數(與 evaluator 的 per_net_crossings 同 key 對齊)。
4. 對照與校準:per-net 相關性(Spearman/Pearson)、總量偏差、依 degree bucket
   的偏差結構;以此決定 evaluator 的修正(例如方向 A 的模型選擇、或迴歸校準)。
5. 校準後的 evaluator 回饋 Phase 1 的 objective 權重與 exit 判準。

**依賴。** 方向 A 的 formulation 選擇會影響對照的解讀(MST vs region-Steiner vs
真實 route 三者的差距各有意義:MST−route = detour 高估;route−lower bound =
router 的 crossing 品質)。建議 Phase 2 對照時三個值都記。
