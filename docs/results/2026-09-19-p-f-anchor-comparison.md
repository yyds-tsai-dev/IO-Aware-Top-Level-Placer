# P-F — Soft-assign anchor comparison (design v2 §7)

**Status:** filled from `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json`,
produced 2026-09-20 on the H100 NVL host (GPU 3).
**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §7,
"Verification experiment".
**Plan:** `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` Task 7.

## Question

From one soft solution, does the cell-centre anchor predict post-fence-LG IO
better than the node lower-left or the pin anchor? Spec §7 expects centre
closest. **If `pin` is closest, LG displacement exceeds half a cell and the
legaliser must be examined** — that is a finding about LG, not about the anchor.

## Answer

**Yes — centre is closest, as design v2 §7 expected.** `pin` is not closest
(it is in fact the *worst* of the three, matching Task 2's measured pin bias),
so the LG-displacement investigation §7 would trigger on a pin win is not
triggered. `--node-anchor center` stays the default.

## Protocol

| item | value |
|---|---|
| case | `mempool_tile_wrap` |
| config | `results/route_gp_20260914/mempool_tile_wrap.json` (ruling R-3). `benchmarks/ispd25/h100/mempool_tile_wrap.json` landed on this branch during this task (commit `4f88774`) and is **byte-identical** (`sha256` `d81e0faf3191…de09aec` for both) — the R-3 choice is recorded for traceability, not because the two differ. |
| K / geometry | 16 / grid 4×4 |
| soft solution | `run_main_flow --phase soft --node-anchor center --dp-seed 1000 --init die_center` → `soft.npz`, `freeze.json` (froze at GP iteration 648, reason `gp_end`, τ=35.679, τ_rel=0.0308, overflow=0.0798) |
| fence LG | `run_main_flow --phase fence --dp-seed 1000` → `placement.npz`, `result.json` |
| τ | `freeze.json["tau"]` = 35.67911205963874 (the τ the soft solution actually stopped at) |
| surrogate | `L_IO = Σ_e max(λ_e − 1, 0)`, `w_mode=unit`, `λ_io=1`, no margin |
| truth | `result.json["hard_lambda_sum"]` on the post-fence-LG placement |
| script | `"$IOPLACE_PYTHON" src/scripts/run_anchor_comparison.py --config results/route_gp_20260914/mempool_tile_wrap.json --out-dir runs/anchor_cmp/mempool_tile_wrap --k 16` |
| host | H100 NVL, `CUDA_VISIBLE_DEVICES=3` (GPUs 0–2 foreign/busy at the time; GPU 3 idle before and during the run) |

## Result

| anchor | l_io_soft | lambda_sum_soft | io_lb_final | abs_err | rel_err | io_count_final | closest |
|---|---|---|---|---|---|---|---|
| lower_left | 16360.9 | 152520.9 | 11658 | 4702.9 | 0.4034 | 14908 | |
| center | 16294.4 | 152454.4 | 11658 | 4636.4 | 0.3977 | 14908 | yes |
| pin | 16529.3 | 152689.3 | 11658 | 4871.3 | 0.4178 | 14908 | |

(`n_active` = 136160 nets for every row; all three anchors read off the *same*
soft solution and the *same* τ, so the spread above is anchor bias alone.)

## Straddle context (from the same `result.json`)

| metric | value |
|---|---|
| `straddle_cells` | 774 |
| `straddle_area_fraction` | 7.964488404639223e-08 |
| `straddle_pin_split_nets` | 0 |
| `straddle_wide_cells` | 15446 |
| `fence_compliance` (lower_left) | 1.0 |
| `fence_compliance_center` | 1.0 |
| `io_soft` | 14758 |
| `io_fence_gp` | 67001 |
| `io_count` (post-fence-LG, MST geometry) | 14908 |
| `io_delta_at_freeze` | 52243 |
| `lg_loss` | -52093 |
| identity residual (`io_identity.verify_io_identity`) | 0 (`14908 = 14758 + 52243 + (-52093)`) |

`fence_compliance == fence_compliance_center == 1.0`: every movable cell ends
up in the fence region its centre was assigned to, under both the lower-left
and centre bookkeeping — so the residual anchor-bias measured above is not
being masked by any cells that actually violated their fence.

## Reading

**Centre closest (expected).** §7's fix is confirmed: the centre is the best
available predictor of post-LG cell ownership among the three anchors, and it
coincides with the freeze rule and fence ownership convention (`fence_compliance_center
= 1.0`). Keep `--node-anchor center` as the default. `abs_err(lower_left) −
abs_err(center) = 4702.93 − 4636.38 = 66.55` is the systematic bias the anchor
change removes (≈0.57% of `io_lb_final`); `abs_err(pin) − abs_err(center) =
234.89` (≈2.0% of `io_lb_final`) is consistent in direction and rough
magnitude with Task 2's double-pin bias measurement (the pin arm over-counts
multi-pin cells and so pushes the surrogate further from the hard truth, not
closer).

All three anchors sit roughly 40% above `hard_lambda_sum` in relative terms —
this is expected and not a defect of the anchor choice: the freeze fired at
`gp_end` with τ still fairly soft (τ_rel≈0.031, not the fully-hardened
`tau_full=0.05` floor because `gp_end` — not a τ-driven freeze condition — was
the reason recorded), so a large share of the soft assignment's mass is still
smeared over more than one region at freeze time; §7 asks which anchor is
*closest*, not how small the absolute gap is, and centre wins that comparison
cleanly and in the expected direction.

**Caveat — `legalization_status: failed`.** The fence-LG run's own DREAMPlace
legality check (`legality_check_op`) reported `legal=False` on the final
placement (`num_unplaced_cells=0`, so no cell left the die or went
non-finite, but at least one cell failed row/site alignment during greedy
legalization — the log shows a single recurring row-232 misalignment on the
order of a few sites, well inside `precision=0.005`-scale rounding, not a
gross overlap). `hard_lambda_sum`/`io_count` are still read off this
placement per the protocol above (no re-run was authorized to chase a clean
legalization, and the accounting identity above closes at residual 0 against
this same placement, so the numbers are internally consistent). This is
recorded as a finding for whoever next investigates LG robustness on
`mempool_tile_wrap`, not something this task's script/doc silently smoothed
over.

## Artefacts

| file | sha256 |
|---|---|
| `runs/anchor_cmp/mempool_tile_wrap/soft.npz` | `ec1ad8ac2e8f698620e7f4054408c1cd3f80137594dee1b3dc08b189892ec0f1` |
| `runs/anchor_cmp/mempool_tile_wrap/placement.npz` | `e0ea766dd1f14ad7745a4f6e5faff711ed311e897a604fe2b5306fcdc2c98c09` |
| `runs/anchor_cmp/mempool_tile_wrap/result.json` | `9b9ec40f21ab1d5a4b986910b1e7967842b0e6ae38e7fa8d27f2427aa2065490` |
| `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json` | `35a1aa80ec6a0017cf0097bdbeb24bdfd7eaf1405183f7be4a39c26ba4c6bc68` |
