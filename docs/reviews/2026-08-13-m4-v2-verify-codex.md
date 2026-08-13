## Verdict

No blanket go for M4 v2 as written.

At commit `3c2ad5d`, the closure count is:

- **RESOLVED: 8**
- **PARTIALLY RESOLVED: 8**
- **NOT RESOLVED: 0**

The author responded to all 16 findings, but did not fully close all 16. I verified that both current spec files are byte-identical to their `3c2ad5d` blobs and that the prior review contains exactly 16 findings.

## A. Findings 1–16

| # | Judgment | v2 quote and verification |
|---|---|---|
| 1 | **PARTIALLY** | §1.1 correctly says the old fit “**不得用來裁決 27.7M 的可行性**” and invalidates the 890 s estimate ([lines 74–77](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:74>)). But T0b claims a fixed-case density/bin sweep will “讓 `N_total`/`N_pins`/`n_bins` 去共線” ([line 526](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:526>)). `N_pins` is constant for a fixed netlist, so the pin coefficient remains unidentified unless scale/topology is independently varied. There is also no condition-number/CI rejection threshold. |
| 2 | **PARTIALLY** | The correct lifetime formula is now stated: “`peak_device ≈ resident(t) + max_p transient_p`” ([line 149](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:149>)). But the same table still says “**上界通過 ⇒ 可行**” for 11.3M/12.3M ([lines 155–156](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:155>)), contradicting the claimed full-lifetime-only rule. The `未定` state can also deadlock E2; see B2. |
| 3 | **PARTIALLY** | The predictor set is improved to `N_nodes, N_nets, N_raw_pins, N_dedup_pins, total_name_bytes` ([lines 79–85](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:79>)). But this is a six-coefficient model with only three Bookshelf scale points, and §5.1 calls 2×2 a holdout while T6b includes it in refitting ([line 383](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:383>), [line 536](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:536>)). |
| 4 | **PARTIALLY** | Glue placeholders and int64 composite keys are correctly specified: “**composite key 一律保持 int64**” ([lines 139–140](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:139>)). But T6b consumes T7’s manifest while the DAG orders T6b before T7 ([line 536](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:536>), [line 557](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:557>)). The count-freeze remedy is therefore not executable as ordered. |
| 5 | **PARTIALLY** | The power-law kernel is complete for all distances and 2×2’s zero residual DoF is honestly stated ([lines 234–255](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:234>)). However, the allowed 1×4 branch has no estimator, test statistic, threshold, or zero-count handling; §3.2 contains only the 2×2 `√2` formula. See B1. |
| 6 | **RESOLVED** | §3.3 explicitly separates fit metrics from one-shot holdout metrics: “**fit 指標不判定…判定只由 holdout 指標決定，且只評一次**,” prohibits rescanning, and corrects V5’s dimensional error ([lines 257–280](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:257>)). |
| 7 | **RESOLVED** | The cell-count inference is withdrawn, and T3a requires prefix coverage, group size/composition, geometry, and top-level residuals before producing `cluster_stats.json` ([lines 192–209](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:192>)). |
| 8 | **RESOLVED** | Real-quality and synthetic-scaling artifacts are separated, with forbidden synthetic quality columns and a report linter ([lines 434–447](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:434>)). |
| 9 | **RESOLVED** | §4.2 now requires source bit planes and accumulators to change dtype together, recalculates memory, and correctly excludes K=64 ([lines 318–334](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:318>)). |
| 10 | **RESOLVED** | Integer/structural fields are exact; `tree_wl` and `hpwl` use a fixed `rel ≤ 1e-12` contract, with an explicit reduction-order counterexample ([lines 336–345](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:336>)). |
| 11 | **RESOLVED** | B1 now says “污染成立、成因未證,” explicitly calls reset insufficient, and assigns the four-arm subprocess/teardown experiment to T1b ([line 119](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:119>)). |
| 12 | **RESOLVED** | The formula is corrected to `1 + n_nonempty_buckets`; the invalid gradient reuse is withdrawn; callback wall-time becomes the acceptance measure ([line 120](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:120>)). |
| 13 | **RESOLVED** | Budgets, sources, baseline reserved memory, and hardware-contract assertions are fixed in advance; the CLI cannot raise the contract limit ([line 121](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:121>)). |
| 14 | **PARTIALLY** | Field semantics, child-process RSS, exclusive-GPU requirements, and short-peak tests are added ([lines 418–430](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:418>)). But T1b’s enforcement checks `used − baseline < 0.5 GB` immediately after baseline capture ([line 528](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:528>)); that masks GPU memory already occupied by another process and is effectively tautological. |
| 15 | **PARTIALLY** | E5 is correctly downgraded to a forecast, uses phase decomposition, removes the second tolerance band, and adds T14 ([lines 461–489](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:461>)). But `AB ≥ 0.6×peak` and `[0.7,1.3]×T̂` have no empirical or probabilistic calibration. See B3. |
| 16 | **PARTIALLY** | Arm definitions, required fields, RESULT GATE, T8b, and most dependencies are materially improved ([lines 510–572](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:510>)). Remaining defects include T6b/T7 inversion, no T10→T11 edge, T0b’s undeclared T1 dependency, and RESULT GATE’s inability to accept an expected OOM as a valid completed experiment. |

## B. Three self-flagged weak points

### B1. §3.2 identifiability

The 2×2 claim is directionally correct but imprecisely worded.

A 2×2 array has six unordered tile-pair counts:

- four pairs at `d=1`;
- two pairs at `d=√2`.

Thus it has six raw counts but only **two distinct distance levels**. Once pooled into `lambda_adj` and `lambda_diag`, there are exactly two moment equations for two parameters. That gives parameter identifiability, but zero lack-of-fit degrees of freedom for the distance-response shape.

They are also not necessarily statistically independent: a multi-group net can contribute to several pair counts. The report should say “two distinct moment equations,” not “two independent observations.”

For 1×4, the distance classes are `d={1,2,3}`. It can provide one scalar lack-of-fit check:

1. Fit `λ₀, α` from `d=1,2`.
2. Hold out `d=3`.
3. Test `λ̂₃ = λ₁·3^{-α}`, where `α=log(λ₁/λ₂)/log 2`.

So the distance-2 observation alone does not validate shape; the held-out distance-3 observation provides the single residual. Even then it is only a weak falsification because there is one `d=3` pair and no defined count-noise model.

Required patch:

- define the 1×4 estimator and held-out statistic;
- retain all pair-specific counts to report anisotropy/overdispersion;
- define handling for zero counts and `α<0`;
- if the real cluster is 1×4, validate against a synthetic 1×4, not the currently mandated synthetic 2×2.

### B2. Three-way feasibility and E2(c)

Yes, `27.7M` can remain `未定` forever.

A clean spike might complete at 20.0 GiB:

- it fails the 19.5 GiB safety-budget definition of feasible;
- it does not OOM;
- it does not prove `resident + min transient > 21.7`;
- therefore it remains `未定`.

Worse, an actual OOM is evidence for “infeasible” under §2.2, but RESULT GATE requires `status=="success"`, while E2(c) requires the spike to “完成.” An expected OOM cannot close T9/E2 as written.

Use this decidable closure rule:

- Separate `experiment_status` from `workload_status`.
- A parent process must always write a valid diagnostic artifact, even if the isolated child OOMs.
- Terminal verdicts:

  - `feasible_l4_contract`: clean run completes and peak ≤19.5 GiB.
  - `infeasible_l4_contract`: clean run completes above 19.5 GiB, or clean OOM is reproduced twice, or the analytical lower bound exceeds capacity.
  - `invalid_measurement`: contamination/instrumentation/input failure; retry at most twice, then close as `blocked_external` and trigger the named fallback.

- E2(c) should require a valid terminal verdict, not `workload_status=success`.

This makes the experiment decidable while preserving an honest FAIL outcome.

### B3. E5 forecast form

The honest form is indeed:

- **Memory:** interval/bounds.
- **Wall time:** point estimate plus explicit assumptions and sensitivity scenarios.

`AB ≥ 0.6×peak` is an arbitrary binary classifier. Low achieved bandwidth can mean latency, occupancy, atomics, cache behavior, launch overhead, or computation; it does not select FP64-ratio versus SM-ratio scaling reliably.

Likewise, `[0.7,1.3]×T̂` is not a prediction interval without H100 residual data. It is merely an acceptance band.

Recommended E5 form:

- GPU memory: analytical lower/upper bounds from resident tensors, phase transients, allocator reserve, and context overhead.
- Host memory: a 95% PI only after the host model becomes identifiable.
- Wall time: freeze `T̂=ΣT_L4,p/s_p` with each `s_p` explicitly labelled as an assumption; report BW/SM/FP64 scenario sensitivity, not a statistical interval.
- T14’s first H100 run tests the assumptions. Only subsequent H100 observations can support a calibrated prediction interval.

## C. M3 cross-fix

The actual cross-fix is now consistent.

[M3 line 134](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md:134>) requires:

> “source bit planes 與 accumulator 成對 int64→int8”

and forbids packed-mask `amax` for `passed_bit_acc`. This matches [M4 §4.5](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:363>).

However, [M4 line 670](</nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:670>) still says M3 line 134 remains stale and must be corrected. That note is itself stale and should be deleted.

## D. New v2 defects

- **[BLOCKER] T9/E2 is not decidable.** Expected OOM and successful runs above the safety margin cannot satisfy the success-only RESULT GATE.
- **[BLOCKER] T6b/T7 dependency inversion.** T6b consumes T7 counts but precedes T7. Use `T6 → T7 → T6b → T9`, or split 2×2 and 3×3 count freezes.
- **[BLOCKER, conditional] 1×4 is treated as a passing hierarchy outcome, but T6 still compares a synthetic 2×2 with the real cluster.** That comparison is geometrically confounded.
- **[MAJOR] T0b cannot independently vary `N_pins`.** Add scale/subgraph/replication points and a numerical rejection gate.
- **[MAJOR] Host RSS regression is underdetermined and leaks its stated 2×2 holdout into fitting.**
- **[MAJOR] §2.2 simultaneously calls the standalone sum an upper bound and says it may underestimate; its table again infers feasibility from that sum.**
- **[MAJOR] T3a failure has no executable fallback branch.** It both forbids T4/T6 and says to switch to recipe B, but recipe B has no replacement tasks or acceptance path.
- **[MAJOR] E5’s threshold and intervals are uncalibrated.**
- **[MAJOR] The exclusive-GPU gate subtracts a baseline that already includes pre-existing external usage.**
- **[MAJOR] The revised DAG still omits required edges**, including T1→T0b, T1→T8a, and T10→T11.
- **[MINOR] M4 line 670 falsely reports that the M3 line-134 fix remains outstanding.**

## Phase authorization

A limited start is reasonable, but M4 as a whole is not design-approved.

| Task | Decision |
|---|---|
| T0 | **GO** |
| T0b | **HOLD** until the identifiable design matrix and rejection thresholds are specified |
| T1b | **HOLD acceptance** until the exclusive-GPU check is corrected |
| T2 | **GO** |
| T2b | **GO implementation**; do not accept memory results until the T1b clean-run contract is corrected |
| T3 | **GO** |
| T3a | **GO evidence collection**; a 1×4 result must stop before T4/T6 interpretation |
| T4 | **PARTIAL GO** for replication/streaming infrastructure; hold glue/fallback behavior |
| T5 | **GO** for generic validation infrastructure |
| T6 | **HOLD** until the 1×4 protocol, fallback branch, and uncertainty method are specified |

Final verdict: **M4 Phase may not proceed wholesale on v2. It may proceed only with the bounded subset above while a v2.1 design patch closes the blockers.**