The revised v2 is not ready for implementation planning. The JSON numbers largely reproduce, but the new S4a selection argument, force schedule, and eligibility logic have six blocking defects.

1. [BLOCKER] `U` is mathematically unstable and does not reproduce the applied optimization regime.

   Claim (§3.3): “`U` = …多降的 `ft_rg` ÷ …少降的 `io_rg`,” with random-direction noise floor `0.05` ([v2 §3.3](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:209)).

   Evidence:

   - At `τ_rel=0.10`, the advertised `U=17.4` is exactly
     \[
     \frac{22-(-65)}{-189-(-194)}=\frac{87}{5}=17.4.
     \]
     Thus the headline result divides by a five-count IO difference—0.022% of base `io_rg=22,693` ([raw output](/tmp/m3probe/ft_combined_step.json:1)). A few boundary-crossing nodes can make `U` infinite, change sign, or reverse rankings.
   - `U` does not define acceptable quadrants. A candidate that improves both metrics yields a negative denominator; one that worsens FT and improves IO can also produce a positive ratio.
   - “Noise floor 0.05” is one random vector from `manual_seed(0)`, not a distribution ([probe](/tmp/m3probe/probe_ft_grad_detach.py:167)).
   - The probe equalizes combined-gradient RMS ([probe](/tmp/m3probe/probe_ft_combined_step.py:61)), whereas the proposed scheduler normalizes L1 force. Candidates with different gradient sparsity therefore receive a different effective experiment.
   - It omits WL/density forces, Nesterov state, the coefficient cap, and the primary exit metric `ft_mst`; v2 itself admits the missing forces and single-step limitation ([v2 §3.3](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:231)).

   Concrete fix: replace scalar `U` with a preregistered Pareto vector `(Δft_mst, Δft_rg, Δio_mst, Δio_rg, Δhpwl, density)` and confidence bands. Require absolute FT improvement and no IO/HPWL regression outside a measured band. Test several small `η`, multiple random controls, checkpoints from actual trajectories, and the actual L1-normalized/full-objective step.

2. [BLOCKER] Gradient normalization removes only global multiplicative gradient scale—not the failed magnitude/support evidence.

   Claim (§0): surrogate “絕對數值在 objective 裡會被除掉,活下來的只有方向” ([v2 §0](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:15)).

   Let \(I=\nabla L_{IO}\), \(F=\nabla L_{FT}\), \(W=\nabla WL\). The actual proposed force is

   \[
   \kappa=\operatorname{clip}\!\left(f\frac{\|I\|_1}{\max(\|F\|_1,\epsilon)},0,100\right),\quad
   C=I+\kappa F,
   \]
   \[
   G_{\rm applied}=
   \min\!\left(\rho a\,R_{\rm EMA},
        \frac{c_{\rm lip}\tau^2}{\gamma C_{\max}}\right)C.
   \]

   Consequently:

   - \(\|\lambda\kappa F\|_1/\|\lambda I\|_1=f\) only while the clamp is inactive.
   - \(\|G_{\rm applied}\|_1=\rho a\|W\|_1\) only with an instantaneous ratio and a non-binding cap—not with `ratio_ema` and the §5.3 cap.
   - Rescaling the entire surrogate by a positive scalar cancels under unclamped \(\kappa\). Per-net/bucket distortion does not. `mass_on_ft0` measures support misallocation, which directly changes direction; it is not an irrelevant global scale.
   - On the main A2 k16 placement, S4a still puts 68.5% of its value on FT-zero nets at `τ=0.10` and 29.7% at `τ=0.03` ([P0 JSON](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_ft_surrogate_soft.json:1614)).

   Concrete fix: narrow the claim to “global gradient scaling cancels in the unclamped, uncapped instantaneous limit.” Preserve support/bucket diagnostics as direction-safety gates and log `f_effective`, cap activity, instantaneous/EMA ratios, and applied combined-force norm.

3. [BLOCKER] S4a’s chain-to-star bias is real, and the proposed L12 diagnostic cannot detect it.

   Claim (§3.5/L12): S4a is a hop-weighted direction despite “星形和對鏈狀 net 重複計費”; per-Λ `ft_rg` changes will detect trouble ([v2 §3.5](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:254), [L12](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:274)).

   Evidence: for the path `h—a—b` with all three regions terminal,

   \[
   ST=2,\quad \Lambda=3,\quad FT=0,\quad
   S4a=(0-1)_+ +(1-1)_+ +(2-1)_+=1.
   \]

   For a star with two terminals adjacent to `h`, `ST=2`, `Λ=3`, `FT=0`, but S4a is zero. S4a therefore strictly prefers the star although both evaluator metrics are identical. A per-Λ `ft_rg` table sees `Λ=3, FT=0` before and after and misses the conversion completely.

   The old bucket evidence remains directionally alarming: on A2 k16, S4a overweights L3 by 2.6× and L4+ by 12.54×, while 90.3% of true FT is in L2 ([probe JSON](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_m3_surrogate.json:90)). The aggregate one-step result is therefore dominated by L2 and does not reconcile the complex-topology contradiction.

   Concrete fix: before selecting S4a, classify nets by topology and measure transitions among chain/star/true-FT classes, plus `hard_S4a−ft_rg` and HPWL per class. Install a gate on chain→star conversions. Otherwise use an overlap-aware edge-union/tree surrogate rather than the home-rooted star sum.

4. [BLOCKER] The callback contract does not yet apply or refresh the force claimed by §5.2.

   Claim (§5.5): update home → measure gradients → update `κ_ft` → update `ratio_ema` → refresh, while preserving the version invariant ([v2 §5.5](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:394)).

   Evidence: the inherited M2 driver computes `lambda_io` first, then updates `ratio_ema`, then refreshes ([driver lines 102–166](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/drivers/run_placement_io.py:102)). It does not recompute `lambda_io` after the new ratio. The refresh therefore sees the old coefficient; the next iteration applies the new coefficient without a corresponding version increment/refresh. Adding `κ_ft` makes this mismatch larger.

   The zero-gradient branch is also wrong as specified: when `g_ft=0` and `f>0`, the formula clips `κ_ft` to 100, then `Cmax=1+100(ecc−1)` can crush the IO step even though the FT force is exactly zero. “No division by zero” is insufficient.

   Concrete fix: make the callback update atomic: measure → derive `κ` → derive combined norm/EMA → recompute `Cmax` and `lambda_io` → bump the objective version once → refresh. Define `g_ft≤ε_rel·g_io` as `κ=0` or an explicitly frozen prior value, with no fictitious Cmax penalty.

5. [BLOCKER] The activation ramp contradicts its late-window rationale and contains an undefined ablation arm.

   Claim (§5.2): FT is “壓在 `of > 0.50` 時接近 0,” concentrating pressure at `τ_rel≤0.10`, without discrete events ([v2 §5.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:344)).

   Evidence from the cited M2 trajectory:

   | Iteration | `τ_rel` | `f/f_max` under the proposed ramp |
   |---:|---:|---:|
   | 300 | 0.226 | 0.255 |
   | 350 | 0.182 | 0.451 |
   | 400 | 0.137 | 0.705 |
   | 450 | 0.096 | 1.000 |

   Thus 70.5% of FT force is already active at the claimed pre-regression boundary, not “near zero.” The trajectory counts themselves reproduce ([M2 trajectory](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m2/sweep/adaptec1_k16_rho0.40_annealed.json:197), [flat trajectory](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m2/noise/flat_det1_rep0.json:141)), but equal-iteration snapshots do not causally prove that earlier forces did not create delayed topology changes.

   Moreover, C9 sets `of_ft_full=0.90` while `of_on=0.90` ([v2 C9](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:438)); the formula divides by `of_on−of_ft_full=0`.

   Finally, if `κ_ft` changes only every 50 iterations, the applied force is a staircase, not a continuous ramp.

   Concrete fix: define explicit start/full points in `τ_rel` or overflow—e.g. zero through the chosen cutoff, then ramp to full—and specify sample-and-hold semantics. Give the “full-course” C9 arm an explicit constant-active mode rather than a zero denominator.

6. [BLOCKER] The two-tier eligibility rule is absent and cannot falsify S4a.

   Claim (§0.1): F1 is downgraded to tier 2 and tier 1 becomes direction utility ([v2 change table](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:33)).

   Evidence:

   - No `U≥1`/`U>=1` threshold, all-τ rule, uncertainty rule, or failure action appears in the body.
   - S4a fails the original magnitude criterion in 16/18 cells and also fails on the main A2 k16 placement at both `τ=0.10` and `0.03`. Calling tier 2 “only estimator status” means it can never reject the optimizer direction.
   - The current ad-hoc combined-step probe tests only A2, one `η`, and `τ={0.10,0.03}` ([probe constants](/tmp/m3probe/probe_ft_combined_step.py:31)); it contains no `τ=0.30` all-τ evidence.
   - The draft simultaneously says it is ready for `writing-plans` ([v2 status](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:3)) and “沒有 P0b 就沒有 S4a 的裁決基礎” ([dependency rule](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:519)).

   Concrete fix: preregister both tiers in the body, with explicit thresholds, regimes, uncertainty, and rejection/fallback actions. Complete P0b before selecting S4a or starting implementation planning. Tier 2 must either constrain activation/selection or be honestly labeled diagnostic, not “eligibility.”

7. [MAJOR] The change table overstates closure of the winner-bias blocker.

   Claim (§0.1): Codex 14 is resolved by fixed three-seed confirmation ([v2 change table](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:61)).

   Evidence: screening selects on seed 1000; confirmation then uses `{1000,1001,1002}` ([v2 §6.2](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:420)). One-third of the confirmation mean therefore contains the value used for selection. Reporting mean±range is not the requested paired confidence bound.

   Concrete fix: confirm on independent seeds—e.g. screen on 1000, confirm on 1001–1003—or run the fixed seed set for every frontier candidate. Compare against matched-seed flat/M2 controls and use a confidence bound on paired differences.

8. [MAJOR] The provenance blocker is claimed complete but remains observable in the cited artifacts.

   Claim (§0.1): “P0/P1/P4/P6 已 hermetic(repo 相對路徑、env metadata、input sha256、commit)” ([v2 change table](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:62)).

   Evidence:

   - P0 hard-codes absolute repository and DREAMPlace paths ([probe source](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/diagnostics/probes_m3/probe_ft_surrogate_soft.py:71)); P1/P4/P6 do likewise.
   - P1/P4/P6 JSON `env` objects lack `command`, Python executable/version, hostname, and timestamp; for example [probe_beta_tau.json](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m3/probes/probe_beta_tau.json:1).
   - None of the four JSONs contains an `exactness` flag.
   - T0 says every probe will have these fields, but only schedules re-emission of the “old four” rg/bb/surrogate/util probes, not P0/P1/P4/P6 ([v2 T0](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:494)).

   Concrete fix: stop claiming completion; make all probe paths repository-relative/configurable, write atomically, apply one schema to every M3 JSON, and re-run P0/P1/P4/P6 as part of T0.

9. [MINOR] The direction-win multiplier is misstated.

   Claim: S4a wins “2–30×” globally and “20–40×” at `τ=0.10` ([v2 §0](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:17), [§3.3](/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:231)).

   Evidence: using the stated U values, the range is approximately 1.99×–43.5×: `2.68/1.35=1.99`, `17.4/0.61=28.5`, and `17.4/0.40=43.5`.

   Concrete fix: report the exact range or remove ratio-of-ratios rhetoric until P0b supplies confidence intervals.

The JSON-backed scalar citations themselves check out: P0 eligibility/worst cells, P4’s `0.152–2.371` range and zero underflows, P6 free-area ratios and `l7_triggered=false`, P1 L-convention differences, and all quoted M2 trajectory checkpoint counts. The ad-hoc table also matches the local `/tmp/m3probe` outputs; its interpretation and provenance are the problem.

Verdict: Findings 1–6 must be resolved in v2, and P0b must be completed under a corrected criterion, before implementation planning starts. Finding 7 also needs a specification change before the experiment plan is frozen. Finding 8 may be executed in T0, but v2 must stop claiming it is already addressed and explicitly add P0/P1/P4/P6 to the reissue set.