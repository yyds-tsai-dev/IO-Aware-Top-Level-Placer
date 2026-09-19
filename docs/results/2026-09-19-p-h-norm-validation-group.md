# P-H group-scale normalisation acceptance (`mempool_group`)

## Scope

Task 9 of subproject P-H (2026-09-19 v2 normalisation redesign,
`docs/superpowers/plans/2026-09-19-v2-p-h-normalisation.md`) requires
group-scale evidence that the new `grandplan` normalisation policy tracks the
retired `legacy` λ path within 2x at matched iterations, at the `mempool_group`
benchmark (k=16, grid). Three arms were run on GPU 3 from
`runs/norm-validation/mempool_group.json` (derived from
`results/recovery_visible_20260906/configs/mempool_group.json` with
`gpu=1, num_threads=16, plot_flag=0, detailed_place_flag=0`):
`--norm-policy legacy --rho-max .40`, `--norm-policy grandplan --norm-wt-max
.40` (matched weight ceiling per the brief's rationale), and `--norm-policy
adaptive --norm-target-share io=0.3,ft=0.1`. Full commands are in
`.superpowers/sdd/2026-09-19-v2-p-h-normalisation/task-9-brief.md` Steps 3-6.

## Run receipts

| Arm | Wall clock (end-start epoch) | Peak GPU mem (mem.csv max, MiB) | `norm_policy` | `norm_wt_max` | `rho_max` | Final `io_count` | Final `ft_count` | Final `hpwl` |
|---|---|---|---|---|---|---|---|---|
| legacy    | 317 s | 26709 | legacy    | 1.0 | 0.4 | 85481  | 8191  | 491369294.97 |
| grandplan | 312 s | 26709 | grandplan | 0.4 | 0.4 | 109306 | 7336  | 526237478.72 |
| adaptive  | 313 s | 26709 | adaptive  | 1.0 | 0.4 | 105110 | 10904 | 526423312.86 |

(`norm_wt_max` is echoed from each run's own JSON; the grandplan arm was
launched with `--norm-wt-max .40` to match legacy's `rho_max` per the brief's
"Why `--norm-wt-max 0.4`" note — the two occupy the same slot in
`λ = weight · ratio_ema`. `norm_p=1`, `norm_ramp_period=100`,
`norm_probe_every=50` on both non-legacy arms.)

## Acceptance test outcome

```
IOPLACE_NORM_VALIDATION_DIR=runs/norm-validation \
  "$IOPLACE_PYTHON" -m pytest tests/test_norm_group_validation.py -v
```

**FAILED** at the matched-iteration precondition, before any ratio is computed:

```
AssertionError: only 14 matched active iterations
assert 14 >= 20
```

Legacy's active iterations (`lambda_io > 0`) are
`[100, 150, ..., 1000]` (19 points, every 50); grandplan's active iterations
are `[150, 200, ..., 800]` (14 points) — grandplan's λ_io reads 0 at iteration
1000 (see below) because IO's contribution has been driven to zero by that
point in the run, so the intersection is exactly grandplan's 14 points, one
short of the test's `>= 20` floor. No worst-ratio/iteration is printed because
the test raises at that assertion.

Informational only (not gating, computed separately, not printed by the
test): over the 14 matched iterations, using the same tail-half /
worst-ratio logic the test would apply, the worst ratio is **8.950x at
iteration 500** (legacy λ_io=5.25561, grandplan λ_io=47.0374) — this would
also fail the `<= 2.0` bound had the count reached 20.

### Adaptive arm (informational; the test has no hook for it)

Adaptive has 16 iterations matching legacy (`[150, 200, ..., 900]`). λ_io at a
few of them, compared to legacy:

| iteration | legacy λ_io | adaptive λ_io | ratio |
|---|---|---|---|
| 150 | 4.07429 | 36.3294 | 8.917 |
| 400 | 2.04801 | 48.6109 | 23.736 |
| 650 | 19.889  | 62.0362 | 3.119 |
| 750 | 65.4041 | 82.2175 | 1.257 |
| 900 | 148.338 | 131.151 | 1.131 |

The ratio is far outside 2x through the middle of the run and only closes to
near-parity near the end (iterations 750-900), consistent with adaptive's
target-share controller converging its λ_io toward legacy's late-run
magnitude only late in the schedule.

## Full suite (P-H gate, `full_suite_v3.log`)

```
2 failed, 1321 passed, 11 skipped, 2 deselected, 1 warning in 2177.93s (0:36:17)
```

Failures (verbatim, first lines from the log):

- `tests/test_evaluator_gpu.py::test_legacy_fields_bit_exact_regression_adaptec1_k16_grid_flat`
  — `FileNotFoundError: [Errno 2] No such file or directory:
  '/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/results/m2/ablation/adaptec1_A0_k16_grid.json.npz'`
- `tests/test_profile.py::test_phase_peaks_do_not_cross_contaminate`
  — `assert timer.phases["b"]["peak_alloc_gb"] < 0.05` /
  `AssertionError: assert 0.05078125 < 0.05`

These two are being triaged separately (see
`.superpowers/sdd/2026-09-19-v2-p-h-normalisation/progress.md`, Task 9 entry
at 16:56 UTC); this document records them as observed, not as classified.

Deselected (2, pre-existing, not part of this failure set): per the earlier
Task 9 triage ruling recorded in `progress.md`, the P-H gate excludes
`tests/test_evaluator_gpu.py::test_gpu_evaluator_memory_mempool_group_k32_under_2gb`
and
`tests/test_bench_bookshelf_netlist.py::test_replication_equals_placedb_read_on_real_1x2_array_movable_first`
by name — both fail from stale `/nashome/NVL4` paths baked into
`benchmarks/ispd25/mempool_group.json` and
`results/m4/bench/arrays/1x2_n2/1x2_n2.manifest.json`, with no P-H-range
commit touching those files.

## Artefacts

- `runs/norm-validation/{legacy,grandplan,adaptive}.json` (+ `.npz`)
- `runs/norm-validation/{grandplan,adaptive}.json.norm_trace.jsonl`
- `runs/norm-validation/logs/<arm>.{log,start_epoch,end_epoch,mem.csv,pid}`
- `runs/norm-validation/logs/full_suite_v3.log` (the valid full-suite run)
