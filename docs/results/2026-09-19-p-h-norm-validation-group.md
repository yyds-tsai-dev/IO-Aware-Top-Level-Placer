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

---

# Rerun after the fix wave (`runs/norm-validation-r2/`)

Controller ruling F3: after the P-H final-review fix wave (code commit
`16b8ca1`) the three arms were rerun with the *identical* Steps 3–6 commands,
into `runs/norm-validation-r2/`, and the acceptance was re-run in its redefined
form (ruling F2). The first attempt's tables above are kept on record.

## Run receipts

All three sequential on GPU 3 (`CUDA_VISIBLE_DEVICES=3`), same config
(`runs/norm-validation-r2/mempool_group.json`, byte-identical recipe to the
first attempt), `repo_commit 16b8ca1`.

| Arm | Wall (end−start epoch) | `runtime_s` | Torch peak (`peak_mem_mb`) | Device used / baseline (GB) | `gp_iterations_run` | Final overflow | `stop_overflow_reached` | Final `io_count` | Final `ft_count` | Final `hpwl` |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy    | 453 s | 427 s | 3140 | 26.20 / 22.33 | 1028 | 0.0697 | **True**  | 85481  | 8191  | 4.914e+08 |
| grandplan | 563 s | 531 s | 3041 | 26.20 / 22.33 |  856 | 0.7142 | False | 277602 | 15137 | 3.716e+09 |
| adaptive  | 583 s | 560 s | 3035 | 27.71 / 22.33 |  955 | 0.7286 | False | 374860 | 23770 | 4.459e+09 |

(The `logs/<arm>.mem.csv` samplers in this rerun were launched with
`nvidia-smi -i 0`, which ignores `CUDA_VISIBLE_DEVICES` and therefore sampled
*physical GPU 0*, a foreign device — those three files are meaningless and are
not used here. The memory columns above come from each run's own JSON:
`peak_mem_mb` is the torch allocator peak, `device_used_gb` the whole-device
reading whose 22.33 GB baseline is foreign traffic on GPU 3.)

## Legacy bit-exactness across the fix wave (new receipt)

`runs/norm-validation/legacy.json` vs `runs/norm-validation-r2/legacy.json`:
**0 differences** over all 20 trajectory events on
`{iteration, overflow, tau, lambda_io, kappa_ft, ratio_ema, obj_version,
io_count, ft_count}`, and identical `io_count` (85481), `ft_count` (8191),
`hpwl` (491369294.9676937), `gp_iterations_run` (1028), `final_overflow`
(0.06973785907030106) and `lambda_io_final` (92.65009553306824). The
`--norm-policy legacy` guarantee holds at group scale across the whole fix
wave, not only in the unit tests.

## Acceptance outcome (redefined, ruling F2): **both gates FAIL**

```
IOPLACE_NORM_VALIDATION_DIR=runs/norm-validation-r2 \
  "$IOPLACE_PYTHON" -m pytest tests/test_norm_group_validation.py -v -s
```

```
AssertionError: only 17 matched iterations with a ratio_ema on both arms
assert 17 >= 20
```

- **Gate (a) — measured normalisation parity: FAILS on the count, passes on the
  criterion.** 17 matched iterations (50…850), 20 required. Over those 17, the
  worst `ratio_ema` ratio is **1.780× at iteration 600** (legacy 181.752,
  grandplan 323.567) and **17 of 17 are within 2×**. The shortfall is entirely
  that grandplan's GP ran 856 iterations against legacy's 1028, so it only
  produced 17 `every=50` callbacks.
- **Gate (b) — no premature collapse: FAILS as written.** Grandplan's last
  iteration with `λ_io > 0` is **850**; legacy's is **1000**; the gate needs
  ≥ 900. But λ_io is positive at **16 of grandplan's 17 callbacks, including
  its last one** — nothing collapsed inside the run; the run itself is shorter.
- **Informational — worst λ ratio: 16.794× at iteration 850** (legacy 224.641,
  grandplan 3772.66) over 16 matched active iterations. Every earlier matched
  iteration is far closer; the outlier is entirely the F1 saturation event
  described below.
- **Informational — FT force share** (λ_FT·‖∇FT‖ / total) at iterations 50 /
  450 / 850: legacy 0.00000 / 0.00000 / 0.02351, grandplan 0.00000 / 0.00000 /
  0.00000. FT never activates on the non-legacy arms because its gate is
  overflow ≤ 0.30 and neither arm's overflow got below 0.64. **I4's per-term FT
  weight ceiling (`wt_max = f_ft_max × norm_wt_max = 0.1`, visible in the trace)
  is therefore not exercised at group scale by this rerun** — only by the unit
  tests.

## What the rerun actually shows: the `eps_rel` guard misfires in late GP

The rerun did not merely miss the gates; the two non-legacy arms **got much
worse than in the first attempt** — overflow stalls at 0.71/0.73 instead of
reaching `stop_overflow`, HPWL is 7.1×/8.5× legacy's, and `io_count` is 3.2×/4.4×
legacy's. The cause is a single mechanism, and it is not the dependency rule,
the activation clock or the per-term weights:

`update_grad_norms`/`_compute` classify a term as signal-less when
`‖∇T‖ ≤ eps_rel · ‖∇WL‖` (`eps_rel = 1e-3`). That threshold is *relative*, and
on this benchmark ‖∇WL‖ grows by ~6× over GP while ‖∇IO‖ stays flat:

| iteration | ‖∇IO‖ | `1e-3·‖∇WL‖` | classified dead? | λ_io (pre-fix, r1) | λ_io (post-fix, r2) | cap (r2) |
|---|---|---|---|---|---|---|
| 500 | 1417 | 304  | no  | 47.04 | 39.22 | 4131 |
| 800 | 1162 | 893  | no  | 292.1 | 266.9 | 3881 |
| 850 | 1140 | 1184 | **yes** | **0** | **3772.66** | 3773 |
| 900 (r1) | 1304 | 2237 | yes | 0 | — | — |
| 1100 (r1) | 1141 | 1937 | yes | 0 | — | — |

The IO gradient is perfectly healthy at iteration 850 (1140, the same magnitude
it held all run) and `wl/‖∇IO‖ = 1039` is a perfectly usable ratio. What
crossed is ‖∇WL‖. So:

- **Pre-fix (r1):** every probe from 850 on was called dead, λ_io was forced to
  0, and the IO penalty was simply **switched off for the last 25% of GP**
  (285 iterations). That — not good normalisation — is why r1's non-legacy arms
  looked healthy (HPWL 5.26e8, overflow 0.0697): they finished as near-flat
  placements. It is also the "premature collapse" gate (b) was written against.
- **Post-fix (r2, ruling F1):** the same misclassification now hands the term
  the Lipschitz cap's entire remaining headroom, **3772.66 — 14.1× the last
  healthy value (266.9)** — in a single transaction. The placement never
  recovers: overflow rises, GP terminates at 856 iterations on a bad
  configuration.

Both behaviours follow from the same misclassification; F1 changed which way it
fails. Legacy is immune because its coefficient is `min(ρ·ramp·ratio_ema, cap)`
with an EMA that never vanishes — it degrades continuously (λ_io 224.6 → 148.3
→ 114.1 → 93.2 over iterations 850–1000) instead of jumping to either 0 or the
cap.

Reported, not tuned, per ruling F3. The obvious candidate ruling for the
controller — *not applied here* — is to make the guard behave like legacy: keep
using the (stale but finite) `ratio_ema` and let the existing cap bound it, i.e.
`λ_t = min(policy(ratio_ema), cap)`, reserving `λ = 0` for `‖∇T‖ == 0` exactly.
On this run that would give λ_io = 266.9 at iteration 850 (continuous with
iteration 800) rather than 0 or 3772.66. A second, independent question it
raises: whether `eps_rel · ‖∇WL‖` is the right deadness test at all, given that
what it actually tracked here was WL's growth.

## Artefacts

- `runs/norm-validation-r2/{legacy,grandplan,adaptive}.json` (+ `.npz`)
- `runs/norm-validation-r2/{grandplan,adaptive}.json.norm_trace.jsonl`
- `runs/norm-validation-r2/logs/<arm>.{log,start_epoch,end_epoch,exit,pid}`
  (`<arm>.mem.csv` is invalid, see above)

---

# r3 — rerun after ruling F1' (`runs/norm-validation-r3/`)

Controller ruling F1' (fix wave round 2) removed the relative `eps_rel·‖∇WL‖`
deadness classification from the non-legacy arm entirely: every probe now
measures `ratio_inst = ‖∇WL‖/max(‖∇T‖, EPS)` and updates the EMA, λ is
`min(policy(ratio_ema), the term's share of the Lipschitz cap)`, and only an
exactly-zero gradient yields λ = 0. Ruling F2' added the length-relative
liveness gate and a placement-quality gate. Code commit `5e07cdf`; the r1 and
r2 tables above are kept on record.

Same Task 9 Step 4–6 commands, same config (`mempool_group.json` byte-identical
to r2's), sequential on GPU 3. The memory sampler was corrected to
`nvidia-smi -i 3` (r2's sampled physical GPU 0 and was discarded).

## Run receipts

| Arm | Wall | `runtime_s` | Torch peak (MB) | Device max (`mem.csv`, MiB; 22.3 GB foreign baseline) | GP iters | Final overflow | `stop_overflow_reached` | `io_count` | `ft_count` | `hpwl` | λ_io final | λ_ft final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy    | 284 s | 267 s | 3140 | 26747 | 1028 | 0.0697 | True | 85481 | 8191 | 4.91369e+08 | 92.650 | 576.22 |
| grandplan | 318 s | 300 s | 3116 | 26747 | 1199 | 0.0695 | True | 95368 | 9024 | 5.41741e+08 | 114.30 | 483.88 |
| adaptive  | 334 s | 315 s | 3140 | 26747 | 1301 | 0.0696 | True | 97264 | 9990 | 5.51461e+08 | 63.856 | 485.68 |

All three now reach `stop_overflow` (r2: only legacy did). Legacy's r3 run is
**bit-identical to both r1 and r2** — 0 trajectory differences on
`{iteration, overflow, tau, lambda_io, kappa_ft, ratio_ema, obj_version,
io_count, ft_count}`, same `hpwl` and `io_count` — so the legacy guarantee has
now survived three group-scale reruns across both fix-wave rounds.

## Acceptance outcome (ruling F2'): **PASS**

```
IOPLACE_NORM_VALIDATION_DIR=runs/norm-validation-r3 \
  "$IOPLACE_PYTHON" -m pytest tests/test_norm_group_validation.py -v -s
-> 1 passed
```

- **Gate (a) — measured normalisation parity: PASS.** 20 matched iterations,
  16 required (`max(15, 0.8 × min(20 legacy, 23 grandplan))`). Worst
  `ratio_ema` ratio **1.780× at iteration 600** (legacy 181.752, grandplan
  323.567); **20/20 within 2×**.
- **Gate (b) — no premature collapse: PASS.** Legacy's last `λ_io > 0` is
  iteration 1000 of 1028 = **0.9728** of its own run; grandplan's is 1150 of
  1199 = **0.9591**; threshold 0.8755.
- **Gate (c) — placement quality: PASS.** Final overflow
  0.0695 / 0.0697 = **0.9961×** (limit 1.5×); final HPWL
  5.41741e+08 / 4.91369e+08 = **1.1025×** (limit 1.2×).
- **Informational — worst λ ratio: 7.463× at iteration 600** (legacy 13.0072,
  grandplan 97.07) over 19 matched active iterations. This is the
  wt-schedule-vs-ρ-schedule difference the gates were redefined to stop
  treating as a normalisation signal; both arms end within 1.24× of each other
  (114.30 vs 92.65).
- **Informational — adaptive arm** (no gate covers it): worst `ratio_ema` ratio
  2.774× over 20 matched, active fraction 0.999 (legacy 0.973), final overflow
  0.999× legacy's, final HPWL 1.122× legacy's. It would pass (b) and (c) and
  miss (a) by 0.77×.

## FT activation and the I4 ceiling — now exercised

Unlike r2 (where neither non-legacy arm's overflow got below 0.64), both arms
converge far enough for FT's overflow ≤ 0.30 gate to fire:

| Arm | first active iteration | `wt` → | `wt_max` | λ_ft at the end | FT force share (last callback) |
|---|---|---|---|---|---|
| legacy    | it 800 (κ_FT > 0) | — | — | 576.22 | 0.01009 at it 1000 |
| grandplan | it 1050 (of 0.2725) | 0.05 → 0.10 | **0.10 = f_ft_max × norm_wt_max** | 483.88 | 0.00977 at it 1150 |
| adaptive  | it 1150 (of 0.2882) | 0.10 (target share) | 1.0 (no override) | 485.68 | 0.00731 at it 1300 |

Grandplan's FT weight steps 0.05 → 0.10 and **saturates exactly at the I4
ceiling** `f_ft_max × norm_wt_max = 0.25 × 0.40 = 0.10`, and the realised FT
force share (0.00977) lands within 3% of legacy's (0.01009) — which is what I4
was for: without the per-term ceiling FT's share would have converged to IO's.
`kappa_clamped` is False throughout, so M2's bound never had to act.

## Comparison across the three attempts (grandplan arm)

| | r1 (pre-fix: λ := 0 on a "dead" probe) | r2 (ruling F1: λ := cap headroom) | r3 (ruling F1': measure and cap) |
|---|---|---|---|
| final overflow | 0.0697 | 0.7142 | **0.0695** |
| `stop_overflow_reached` | True | False | **True** |
| final HPWL | 5.262e+08 | 3.716e+09 | **5.417e+08** |
| `io_count` | 109306 | 277602 | **95368** |
| λ_io late-GP behaviour | 0 for the last 285 iterations | jumps to 3772.66 (14.1× the last healthy value) | continuous, 679 → 311 → 257 → 114 |
| acceptance | fail (count) | fail (a)+(b) | **pass (a)+(b)+(c)** |

r1's apparently-healthy overflow/HPWL came from the IO penalty being switched
*off* for the last quarter of GP; r3 keeps it on for 96% of the run and still
converges, with a *better* `io_count` than r1 (95368 vs 109306).

## Artefacts

- `runs/norm-validation-r3/{legacy,grandplan,adaptive}.json` (+ `.npz`)
- `runs/norm-validation-r3/{grandplan,adaptive}.json.norm_trace.jsonl`
- `runs/norm-validation-r3/logs/<arm>.{log,start_epoch,end_epoch,exit,mem.csv}`
  (`mem.csv` is valid in r3: sampler bound to `-i 3`)
