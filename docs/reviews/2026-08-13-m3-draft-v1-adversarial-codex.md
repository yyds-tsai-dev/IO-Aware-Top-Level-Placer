1. [BLOCKER] Metric-closure estimates are being reported as actual Steiner/FT values.

   Claim attacked (§2.2): “`Λ≥4 用 metric-closure MST 上界 ⇒ io_rg/ft_rg 皆為上界`,” while defining `ST_e` as the minimum Steiner-tree cost and `FT(e)=ST_e-(Λ_e-1)` ([draft lines 61–77](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:61)).

   Evidence: For an exact tree, the identity is correct and matches the spec’s unique “passed region without a pin” count because \(|V(T)|=|E(T)|+1\); the current evaluator also unions passed regions before counting FT ([evaluator_ref.py:102](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_ref.py:102)). But the probe substitutes terminal metric-MST cost for `ST_e` at every \(\Lambda\ge4\) ([probe_m3_rg.py:77](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/diagnostics/probes_m3/probe_m3_rg.py:77)). On a four-leaf star, exact \(ST=4,FT=1\), while terminal metric-MST cost is 6 and the formula reports \(FT_{ub}=6-3=3\). The algebraic identity still holds, but it is now an identity between upper estimates—not a route-level FT identity. Per-net upper bounds sum to an aggregate upper bound, but `mst_excess = io_mst − io_rg_ub` is only a lower bound on the true excess, not the claimed exact decomposition.

   Fix: Separate `st_exact/ft_exact` from `st_ub/ft_ub` and expose `per_net_steiner_exact`. Do not mix upper bounds into `io_rg`, `ft_rg`, or the three-way decomposition. Either compute exact values for all reported nets or publish lower/upper aggregate intervals.

2. [BLOCKER] The RG model is declared physically meaningful before its admitted geometry gap is measured.

   Claim attacked (§2.2): “不是換尺換出來的改善”; §2.4: “`ft_rg` 只是下界,” with maze validation deferred to T11 ([draft lines 91–109](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:91), [lines 376–378](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:376)).

   Evidence: Agreement with one arbitrary MST L-walk does not establish agreement with a realizable bounded-detour route; both can share bias. T11 is conditional on G3 even though L1 says the validation is mandatory. Using E2 as a primary exit metric before that check lets the optimizer exploit arbitrarily long within-region detours without failing.

   Fix: Either make maze/bounded-detour validation unconditional before v2, or explicitly define RG as a purely topological reference lower bound and demote `ft_rg` from exit criterion to diagnostic until calibrated.

3. [BLOCKER] S4b violates the draft’s own bucket-rejection rule.

   Claim attacked (§3.2): “任一 bucket 的份額比偏離 1 超過 3 倍即推翻,” followed by “S4b … 最差 bucket 是 0.11–0.35（低估），方向安全” ([draft lines 162–166](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:162)).

   Evidence: The probe gives S4b L4+ share ratios 0.11, 0.20, 0.35, and **0.01**, not 0.11–0.35 ([probe JSON:42](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_surrogate.json:42), [line 303](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_surrogate.json:303)). Ratios below \(1/3\) deviate by more than 3×, so three of four configurations fail the stated symmetric rule. Slicing also has only 78.7% of FT mass in \(\Lambda=2\), contradicting the generalized “87–97%” justification ([probe JSON:293](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_surrogate.json:293)).

   Fix: Apply the predeclared rule symmetrically and reject S4b, or revise the rule before selecting a winner with a justified asymmetric loss function. At minimum, require per-bucket recall for L3/L4+, not aggregate ratio alone.

4. [BLOCKER] S4b does not converge to hard `ft_rg`; T2 contains an impossible acceptance test.

   Claim attacked (§8 T2): “`τ` 與 `β` 同時趨近 0 時 `L_FT` 收斂到 hard `ft_rg`（相對誤差低於 5%）” ([draft line 369](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:369)).

   Evidence: In the hard limit, S4b is
   \[
   [ecc_{home}-(\Lambda-1)]_+ \le ST-(\Lambda-1)=FT,
   \]
   only a lower bound. On the four-leaf star above, \(FT=1\) but \(ecc_{home}=2,\Lambda-1=3\), so S4b is 0. The observed aggregate ratios 0.842–0.962 already disprove a universal 5% limit ([probe JSON:29](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_surrogate.json:29), [line 290](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_surrogate.json:290)). The simultaneous \(\tau,\beta\to0\) limit is also path-dependent: exponentially small nonterminal \(q_k\) can be outweighed by \(e^{D/\beta}\).

   Fix: Replace the test with convergence to the hard S4b expression, then separately measure its gap to exact FT. Specify a sequential limit—\(\tau\to0\) first, then \(\beta\to0\)—or a required relationship between the schedules.

5. [BLOCKER] The displayed gradient is correct only on an unstated interior domain; its implemented form would be numerically unsafe.

   Claim attacked (§3.1):  
   “`(D−R)·a/(q·N) − 1`” ([draft lines 137–147](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:137)).

   Evidence: Let \(b_k=e^{(D_k-ecc_{\max})/\beta}\), \(a_k=q_kb_k\), \(N=\sum q_kb_k\), \(M=\sum q_kb_kD_k\). For \(q_k>0,N>\epsilon\),
   \[
   \frac{\partial R}{\partial q_k}
   =\frac{b_k(D_k-R)}N
   =\frac{(D_k-R)a_k}{q_kN}.
   \]
   Thus the `−1` term from \(-\lambda\) is necessary, and the claimed coefficient is correct only there. At \(q_k=0\), `a/q` is \(0/0\). If `max(N,ε)` binds, the derivative is \(b_kD_k/\epsilon\), not \(b_k(D_k-R)/N\).

   Fix: Implement the cancellation-safe form `b_k*(D_k-R)/N`, never `a_k/q_k`. Define the \(N\le\epsilon\) branch and test it by finite differences.

6. [MAJOR] L6’s underflow diagnosis and remedy are wrong.

   Claim attacked (§3.4): fp32 exponent underflow “讓只觸及 home 的 net 的 `N_e` underflow”; fp64 accumulation plus `β ≥ 0.1` is declared sufficient ([draft lines 180–184](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:180)).

   Evidence: At \(k=home\), \(D=0=ecc\) only when eccentricity max? More generally the exponent shift gives a nonzero maximal term; `N` is not lost merely because distant terms underflow. More importantly, fp64 accumulation cannot recover a value already rounded to zero when the exponent was evaluated in fp32: \(e^{-124}\) is 0 in fp32 but \(1.404\times10^{-54}\) in fp64. Lowering β to 0.1 worsens this.

   Fix: Compute exponentials and `q*b` in fp64 before accumulation, or use a per-net log-sum-exp normalization. Derive β’s minimum from the chosen dtype and required gradient dynamic range; do not use `β≥0.1` as an underflow proof.

7. [MAJOR] ReLU boundary semantics will disagree between the reference and custom backward.

   Claim attacked (§3.1): strict indicators `1[λ>1]` and `1[R>λ−1]`, while T2 requires a synchronized autograd reference ([draft lines 143–149](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:143), [line 369](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:369)).

   Evidence: Strict `>` correctly chooses the zero ReLU subgradient at the boundary. But the existing reference uses `.clamp(min=0)` ([io_term.py:120](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:120)), while production hard-codes `(lam > 1)` ([io_term.py:225](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:225)); PyTorch clamp takes derivative 1 at exact equality.

   Fix: Use `torch.relu` in the reference or a custom zero-at-boundary mask, and add exact-boundary equivalence tests for both hinges.

8. [MAJOR] “Zero extra chunk pass” is true for the objective backward, but false for the promised schedule diagnostics.

   Claim attacked (§3.1/§5.2): “零額外 chunk pass” and `g_comb/g_io` is “同一次 callback 免費得到” ([draft line 149](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:149), [line 260](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:260)).

   Evidence: Saving global \(N,R,\lambda\) after forward does let both existing backward chunk sweeps consume the FT coefficient, so no third objective pass is needed. But separate L1 norms of \(g_{comb}\) and \(g_{io}\) cannot be recovered from one combined L1 norm. Current `io_grad_l1` explicitly launches an independent full forward/backward ([io_term.py:339](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/ops/io_term.py:339), [driver:145](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement_io.py:145)).

   Fix: Narrow the claim to the main objective pass. Either budget two callback backward calls or accumulate IO-only and combined coordinate gradients in parallel, documenting the additional \(O(N)\) buffers.

9. [BLOCKER] S7’s demand is a home-rooted star flow, not Steiner-tree demand.

   Claim attacked (§4.2): “與 §2.2 的 routing 模型一致” for \(P[h,k,(a,b)]\) and \(Q[h,k]\) ([draft lines 218–229](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:218)).

   Evidence: For a chain \(h-a-b\) with all three regions touched, the exact Steiner tree uses `h-a` once and `a-b` once. The proposed sum routes one unit to \(a\) and another to \(b\), charging `h-a` twice. This is the same independent-terminal overcounting used to reject S4a. Further, the evidence in §4.1 is legacy MST L-walk `boundary_pair_demand` ([evaluator_ref.py:105](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/evaluator_ref.py:105)), not the proposed shortest-path flow.

   Fix: Add a route-consistent `boundary_pair_demand_rg` with an explicit Steiner/tree tie policy, or label S7 as a star-flow heuristic and validate it against that new evaluator before selecting it.

10. [BLOCKER] `L_cap` is not a hinge and is never inert below capacity.

   Claim attacked (§4.2/R6):  
   `softplus((D_ab−C_ab)/C_ab)` and “hinge 在啟用初期梯度恆 0” ([draft lines 213–216](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:213), [line 356](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:356)).

   Evidence: At \(D=0,C>0\), the argument is only \(-1\), so loss is 0.313 and \(\partial L/\partial D=0.269/C\), not zero. The “freeze when gradient is zero” branch is effectively dead. If total demand is zero, the capacity formula gives \(C=0\) and division by zero. The claimed `A≤2K` is also already contradicted by the draft’s K=16 slicing case with 33 adjacent pairs.

   Fix: Use an actual ReLU/squared hinge or a sharp shifted softplus with a stated sharpness. Define behavior for zero total demand, no adjacent pairs, disconnected graphs, zero-length boundaries, and zero median demand.

11. [BLOCKER] Capacity reference drift and `λ_cap` updates are underspecified and break the inherited objective-version invariant.

   Claim attacked (§5.2–§5.3): update `D_ab/C_ab` at callbacks, “`λ_cap` … cap 梯度為 0 時保持不變,” but only S7 activation/C update is listed as an event ([draft lines 263–280](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:263)).

   Evidence: If \(C(D)\) is recomputed inside the differentiable expression, §4.2’s derivative omits the global \(\partial C/\partial D\) term. If it is detached and refreshed at every callback, the target drifts and every C/`λ_cap` change is a discrete objective change requiring version increment and secant refresh. Holding a stale nonzero `λ_cap` when `g_cap=0` is unsafe. The stated scalar upper bound “`ρ_max` times WL gradient L1” is also dimensionally incomplete; it must be divided by `g_cap`.

   Fix: Freeze one capacity reference per run—preferably from the matched flat baseline—or precisely define detached snapshot cadence. Use
   \[
   \lambda_{cap}=\min\!\left(\kappa_{cap}\frac{\|g_{comb}\|_1}{\|g_{cap}\|_1},
   \rho_{\max}\frac{\|g_{WL}\|_1}{\|g_{cap}\|_1}\right),
   \]
   with `λ_cap=0` when the denominator is zero, and version every change.

12. [MAJOR] The merged-gradient normalization explanation is backwards and ignores cancellation.

   Claim attacked (§5.2): measuring the merged term is required “否則 `κ_ft` 會被正規化抵銷掉” ([draft line 259](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:259)).

   Evidence: With
   \[
   \lambda_{io}=\rho\|g_{WL}\|_1/\|g_{IO}+\kappa g_{FT}\|_1,
   \]
   the merged gradient’s total magnitude is normalized to \(\rho\|g_{WL}\|_1\); that is precisely what normalizes away κ’s overall scale. κ still changes direction and IO/FT composition, but not total force. Worse, IO and FT coordinate gradients can cancel, making the denominator small and amplifying both.

   Fix: State κ’s intended semantics explicitly. Log a cancellation ratio  
   \(\|g_{IO}+\kappa g_{FT}\|_1/(\|g_{IO}\|_1+\kappa\|g_{FT}\|_1)\), and cap or reject normalization when it is small.

13. [BLOCKER] L8 is not a Lipschitz guard.

   Claim attacked (§5.2/L8): divide the M2 cap by `max(1,g_comb/g_io)` because it measures how much larger the merged curvature is ([draft line 260](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:260), [line 409](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:409)).

   Evidence: A gradient-norm ratio does not bound a Hessian or Lipschitz constant. S4 adds rational \(M/N\) curvature controlled by the distribution of \(N\); the proposed ratio contains none of that information and can decrease under cancellation exactly when local curvature is large.

   Fix: Demote it to an empirical step cap, not a safety guard. Calibrate using local secants/Hessian-vector probes over \((τ,β)\), or retain the conservative M2 cap and rely on measured backtracking until a bound exists.

14. [BLOCKER] E1–E5 are not Pareto dominance and can be selected/gamed after the sweep.

   Claim attacked (§6.2/L9): “對現任者的 Pareto 支配,” E2=`0.85×flat`, with the coefficient allowed to be adjusted using T5/T6 results ([draft lines 292–307](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:292), [line 410](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:410)).

   Evidence:

   - Allowing +2% IO and +2% HPWL is an acceptance envelope, not dominance over any single incumbent.
   - The 0.85 factor is safely beyond the measured 2.48% three-sigma band, but 15% itself has no defensible derivation. Adjusting it after observing the frontier is outcome-dependent thresholding.
   - E5 applies flat `ft_mst` seed variance to M1 comparisons and to `ft_rg`, whose seed variance was never measured.
   - E1 permits equality with M1 while E5 demands significant improvement; the combination is ambiguous.
   - Selecting the best of many arms on one seed and rerunning only the winner creates winner’s bias.
   - Integer arithmetic is slightly wrong: \(0.85·2454=2085.9\), so strict E2 is ≤2085, not 2086; E3 similarly gives ≤25139, not 25140.

   Fix: Pre-register thresholds before T6; call them an acceptance envelope. Run the same fixed seed set for every candidate entering the frontier, report paired confidence intervals per metric/case, and require the confidence bound—not a single best run—to satisfy E1–E4.

15. [MAJOR] E6 and the probe provenance contract are currently satisfiable by stale files.

   Claim attacked (§6.2/T0): “每格有對應 JSON” and every number is traceable with environment metadata/input SHA ([draft line 303](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:303), [lines 365–368](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:365)).

   Evidence: The migrated probes still import a hard-coded obsolete worktree, execute code from `/tmp`, and write results back to `/tmp` ([probe_m3_rg.py:3](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/diagnostics/probes_m3/probe_m3_rg.py:3), [probe_m3_surrogate.py:4](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/diagnostics/probes_m3/probe_m3_surrogate.py:4)). The checked-in JSON lacks the promised commit, command, environment, input SHA, and exact/upper-bound flags. File existence alone does not demonstrate a completed arm.

   Fix: Make probes hermetic and repo-relative, write atomically to `results/m3/probes`, and validate schema, status, input/output hashes, commit, command, seed, and exactness—not merely file presence.

16. [MINOR] The scalar tables mostly match the JSON, but §2.2’s derived range is wrong.

   Claim attacked: grid `ft_rg` and `ft_mst` “只差 2.7–3.7%” ([draft line 93](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:93)).

   Evidence: Fresh recomputation from the seven grid rows gives differences of 0.799%–4.028%, including adaptec1 M2 at 4.028% ([probe RG:71](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_rg.json:71)) and bigblue4 M2 at 0.799% ([probe BB:30](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_bb.json:30)). The individual §2.2 table values and §3.2 candidate totals do otherwise match the checked-in JSON.

   Fix: Report the actual 0.8%–4.0% range and state which rows each quoted correlation uses.

Low-confidence disposition: L1, L2, L6, L7, L8, L9, and L10 are true v2 blockers under the current claims. L3, L4, L5, and L11 can be deferred to tasks if v2 labels their defaults provisional and installs predeclared go/no-go gates. L1 becomes deferrable only if RG is explicitly demoted to a topological diagnostic; L7 requires the free-area probe before retaining the “no area term” decision.

Verdict: v2 must change Findings **1–5, 9–11, and 13–15**. Findings **6–8 and 12** must be corrected in the engineering/runtime claims before implementation planning. Finding **16** is editorial but should be repaired with the probe refresh.