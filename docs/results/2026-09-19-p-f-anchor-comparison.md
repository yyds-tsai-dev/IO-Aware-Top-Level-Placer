# P-F — Soft-assign anchor comparison (design v2 §7)

**Status:** filled from `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json`,
produced 2026-09-20 on the H100 NVL host (GPU 3). **Revised 2026-09-21 (fix
round 2)** after an adversarial review found the original scalar comparison
degenerate — see "Fix round 2" below before reading "Answer"/"Result".
**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §7,
"Verification experiment".
**Plan:** `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` Task 7.

## Question

From one soft solution, does the cell-centre anchor predict post-fence-LG IO
better than the node lower-left or the pin anchor? Spec §7 expects centre
closest. **If `pin` is closest, LG displacement exceeds half a cell and the
legaliser must be examined** — that is a finding about LG, not about the anchor.

## Fix round 2: the original scalar metric is degenerate — read this first

The first pass of this doc scored each anchor by
`argmin_anchor |l_io_soft(anchor) - hard_lambda_sum|` — one shared scalar,
`hard_lambda_sum`, compared against each anchor's total `l_io_soft`. An
adversarial review found two problems with that:

1. **It degenerates to "smallest number wins."** All three anchors'
   `l_io_soft` land *above* `hard_lambda_sum` (see the Result table below —
   every `abs_err` is positive), so `argmin |l_io_soft - hard_lambda_sum|` is
   *identical* to `argmin l_io_soft`. Over- and under-counting across ~136k
   nets cancels into a single total on each side before scoring ever
   happens, so the comparison cannot tell "predicts accurately" apart from
   "happens to report a smaller aggregate."
2. **The truth is itself pin-anchored, not lower-left.** `result.json`'s
   `hard_lambda_sum` comes from `evaluator_gpu.py`'s `per_net_lambda`, a
   popcount of the *actual pin coordinates'* region membership
   (`_pin_positions` at `evaluator_gpu.py:341-344`, feeding `pin_bm`'s
   popcount at `evaluator_gpu.py:738-740`). This is a different array from
   `node_region_convention: lower_left_position` in `evaluation.npz`'s
   metadata (`export/evaluation.py:121`), which governs only the exported
   `node_region` array (one entry per node, for route-side FT bookkeeping),
   not λ. Scoring the `pin` arm against a pin-anchored truth with a magnitude
   metric can penalise `pin` for a systematic bias (Task 2's measured
   double-counting) that has nothing to do with how close it is to the
   truth's own convention as τ→0 — the first pass's reading ("pin over-counts
   so it is further from truth") mixed up the sign of that limit statement.

**The fix: score per net, not in aggregate.** `net_l1_report()` (added to
`src/scripts/run_anchor_comparison.py` in this fix round) computes, for each
anchor, `Σ_e |λ_soft_e - λ_hard_e|` **per net**, restricted to
`2 <= net_degree < 100` (the same band `build_net_node_csr` already uses),
then sums the *absolute* per-net errors. Over- and under-counting no longer
cancel across nets, because each net's soft λ is compared to *that net's own*
hard λ before anything is summed. `λ_hard` is `evaluation.npz`'s
`per_net_lambda`, loaded through `load_evaluation(..., net_names=placedb.net_names)`,
which raises if the net order doesn't match — the net-order identity is
checked (and confirmed, see below) before any per-net pairing is trusted.

**Result of the rescoring: centre still wins, and now with a metric that
actually measures prediction, not magnitude.** See "Result" and "Per-net L1
rescoring" below. The original headline ("centre wins") survives; the
original *reasoning* about why (the pin-bias explanation) does not and is
withdrawn.

## Answer

**Centre wins on the per-net L1 metric, with a paired bootstrap 95% CI that
excludes zero (mean per-net advantage −0.00076, CI [−0.00086, −0.00066]).**
This is real evidence for §7's expectation, not an artefact of the
degenerate scalar. `pin` is *not* closest by either metric, so §7's
"examine LG" branch does not trigger. `--node-anchor center` stays the
default — on this evidence, not just prior design grounds.

One nuance the rescoring surfaced and the old scalar hid: `lower_left`
(L1 = 7168.94) and `pin` (L1 = 7168.67) are essentially **tied** for
second place (a 0.27-wide gap on totals of ~7169, i.e. ~0.004% apart) — the
opposite of the old scalar table, which had `pin` clearly worst
(`abs_err` 4871.3 vs `lower_left`'s 4702.9). That ordering was an artefact
of the scalar's cancellation, not a real finding about `pin`'s bias
direction, and is withdrawn. No bootstrap was run on the `lower_left`-vs-`pin`
gap specifically (only `center`-vs-`lower_left` was asked for); given how
small that gap is relative to `center`'s confirmed margin, treat `lower_left`
and `pin` as statistically indistinguishable here, not ranked.

## Protocol

| item | value |
|---|---|
| case | `mempool_tile_wrap` |
| config | `results/route_gp_20260914/mempool_tile_wrap.json` (ruling R-3). `benchmarks/ispd25/h100/mempool_tile_wrap.json` landed on this branch during this task (commit `4f88774`) and is **byte-identical** (`sha256` `d81e0faf3191…de09aec` for both) — the R-3 choice is recorded for traceability, not because the two differ. |
| K / geometry | 16 / grid 4×4 |
| soft solution | `run_main_flow --phase soft --node-anchor center --dp-seed 1000 --init die_center` → `soft.npz`, `freeze.json` (froze at GP iteration 648, reason `gp_end`, τ=35.679, τ_rel=0.0308, overflow=0.0798) |
| fence LG | `run_main_flow --phase fence --dp-seed 1000` → `placement.npz`, `result.json`, `evaluation.npz` |
| τ | `freeze.json["tau"]` = 35.67911205963874 (the τ the soft solution actually stopped at; same τ used for every anchor, including the fix-round-2 per-net rescoring) |
| surrogate | `L_IO = Σ_e max(λ_e − 1, 0)`, `w_mode=unit`, `λ_io=1`, no margin |
| scalar truth (degenerate, kept for provenance) | `result.json["hard_lambda_sum"]` on the post-fence-LG placement |
| per-net truth (fix round 2) | `evaluation.npz["per_net_lambda"]`, pin-anchored (see above), paired by net id after `load_evaluation`'s net-order check |
| net-order identity check | `evaluation.npz`'s `metadata.net_order_sha256` = `e594705775c0622e556356712dc0b0bfa132cc849eacd5ba0d1f6a6201dcab5f`, confirmed equal to `array_digest(normalized_names(placedb.net_names))` recomputed from a fresh `load_netlist(config)` in this same session — **PASSED**, so the per-net pairing below is meaningful |
| script | `"$IOPLACE_PYTHON" src/scripts/run_anchor_comparison.py --config results/route_gp_20260914/mempool_tile_wrap.json --out-dir runs/anchor_cmp/mempool_tile_wrap --k 16` (now also runs `net_l1_report` whenever `evaluation.npz` is present) |
| host | H100 NVL, `CUDA_VISIBLE_DEVICES=3` (GPUs 0–2 foreign/busy at the time; GPU 3 idle before and during every run) |

## Result — the scalar table (degenerate; kept for provenance, not the answer)

| anchor | l_io_soft | lambda_sum_soft | io_lb_final | abs_err | rel_err | io_count_final | closest |
|---|---|---|---|---|---|---|---|
| lower_left | 16360.9 | 152520.9 | 11658 | 4702.9 | 0.4034 | 14908 | |
| center | 16294.4 | 152454.4 | 11658 | 4636.4 | 0.3977 | 14908 | yes |
| pin | 16529.3 | 152689.3 | 11658 | 4871.3 | 0.4178 | 14908 | |

(`n_active` = 136160 nets for every row; all three anchors read off the *same*
soft solution and the *same* τ.) **`closest` here is `argmin l_io_soft` in
disguise (see "Fix round 2" above) — do not read this table as the sec 7
answer on its own.** It agrees with the per-net L1 rescoring on the winner
(`center`) but disagrees on the ranking of the other two, which the L1
rescoring shows are actually tied.

## Per-net L1 rescoring (the actual metric, fix round 2)

`Σ_e |λ_soft_e - λ_hard_e|`, restricted to `2 <= net_degree < 100`
(136,208 nets in band, out of 145,589 total):

| anchor | per-net L1 |
|---|---|
| lower_left | 7168.94 |
| **center** | **7065.71** |
| pin | 7168.67 |

Paired bootstrap (10,000 resamples, seed 1000) over per-net
`d_i = |err_center_i| - |err_lower_left_i|` (negative ⇒ centre's per-net
error is smaller):

| statistic | value |
|---|---|
| mean(d) | −0.00075789 |
| sum(d) | −103.23 |
| 95% CI on mean(d) | [−0.00085660, −0.00065955] |
| excludes zero | **yes** |

Centre's per-net advantage over `lower_left` is small in relative terms
(≈1.4% of `lower_left`'s own L1) but the bootstrap CI is entirely on the
"centre is better" side of zero — over 136k paired observations, a small but
consistent per-net effect adds up to a CI that does not touch zero. This is
the evidence the original scalar table could not provide.

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
and centre bookkeeping.

**On the ±52k terms and the negative `lg_loss`, cross-checked in this fix
round.** `io_delta_at_freeze = +52243` and `lg_loss = -52093` are credible
measurements from the same evaluator in the same coordinate frame (the
identity residual closes at 0 against them, and both come from re-evaluating
the *same* placement snapshots with the *same* `evaluator_gpu` code path used
throughout P-F). The large negative `lg_loss` — post-fence-LG IO *dropping*
relative to the fence GP's own IO, rather than the usual small positive cost
LG adds — is **fence enforcement, not row-snapping**: `inject_fence_regions`
(`fence_phase.py`) sets `node2fence_region_map` so the fence GP only *softly*
pulls cells toward their assigned region (a density-objective bias), while
the fence-aware legalizer supplies the actual hard guarantee
(`fence_compliance = 1.0` is enforced at LG, not at GP). So a fence GP that
hasn't fully pulled every cell home by the time LG runs is expected to show a
large IO drop at the LG step, because LG is where containment actually
becomes exact. Cross-checked against `runs/norm-validation{,-r2,-r3}/
{legacy,grandplan,adaptive}.json` (non-fence norm-policy validation runs,
no fence LG involved): every one of those reports a small **positive**
`lg_loss` (993 to 2942, with `hpwl_lg > hpwl_gp`), except two runs under
`-r2` (`grandplan`/`adaptive`) that diverged during GP (`lg_loss` in the
hundred-thousands, a divergence artifact, not a sign flip). This run's
sign-inverted, much larger `lg_loss` is therefore symptomatic of a
comparatively weak fence GP on `mempool_tile_wrap` at this K/seed (it leaves
more IO for the fence-aware LG to fix than a non-fence run ever has to) —
worth a follow-up on fence GP strength on this benchmark, but **irrelevant to
the anchor ranking above**, which never touches `io_delta_at_freeze` or
`lg_loss`.

## Reading

**Centre wins, on real evidence.** The per-net L1 rescoring (fix round 2)
confirms design v2 §7's expectation with a bootstrap CI that excludes zero:
centre's soft assignment is a measurably better per-net predictor of the
post-fence-LG hard λ than `lower_left`. `pin` is not closer than `center` by
either metric, so the LG-displacement investigation §7 would trigger on a
`pin` win is not triggered. Keep `--node-anchor center` as the default.

**Withdrawn: the pin-bias explanation from the first pass.** The original
doc read `pin`'s worse scalar `abs_err` as "the pin arm over-counts
multi-pin cells and so pushes the surrogate further from the hard truth."
That direction-of-bias claim doesn't survive the per-net rescoring — `pin`
and `lower_left` are statistically indistinguishable in L1 (see above), and
in any case the truth `pin` would be compared against as τ→0 is *itself*
pin-anchored (see "Fix round 2"), so a magnitude-based "further from truth"
reading of `pin`'s scalar total was measuring the wrong thing. Task 2's
double-pin bias measurement stands on its own (it did not depend on this
experiment), but it should not be cited as "explaining" the old scalar
ranking here.

All three anchors sit far above `hard_lambda_sum` in the scalar table's
relative terms (~40%) — this remains expected and not a defect of the anchor
choice: the freeze fired at `gp_end` with τ still fairly soft (τ_rel≈0.031,
not the fully-hardened `tau_full=0.05` floor, because `gp_end` — not a
τ-driven freeze condition — was the reason recorded), so a large share of the
soft assignment's mass is still smeared over more than one region at freeze
time.

**Caveat — `legalization_status: failed`, and why it cannot flip this
ranking.** The fence-LG run's own DREAMPlace legality check
(`legality_check_op`) reported `legal=False` on the final placement
(`num_unplaced_cells=0`, so no cell left the die or went non-finite, but at
least one cell failed row/site alignment during greedy legalization — the
log shows a recurring row-232 misalignment on the order of a couple of
sites, gap ~6.5 native units at `precision=0.005`, not a gross overlap).
Quantitatively this cannot be responsible for `center` winning: a few-site
misalignment perturbs the affected pin's coordinates by order 10² native
units, against this benchmark's own fence-region width of 439,945 native
units (`regions.json`'s die is `[20140, 19880, 1779920, 1778280]`, region 0
is `439945 × 439600`) — flipping the per-net L1 ranking would require a
swing on the order of the `center`-vs-`lower_left` gap (Δ≈4636 in scalar
terms, or the bootstrap's confirmed per-net margin), several orders of
magnitude larger than what one misaligned cell could contribute.
`hard_lambda_sum`/`per_net_lambda` are still read off this placement (no
re-run was authorized to chase a clean legalization, and the accounting
identity closes at residual 0 against it), so the numbers used above are
internally consistent even though the placement isn't strictly legal. Still
recorded as a finding for whoever next investigates LG robustness on
`mempool_tile_wrap`.

**Also cheaply confirmed this fix round — `final_overflow: 0.99996` is very
likely a false-negative `stop_overflow_reached`, not evidence of poor fence
GP convergence.** `result.json`'s `escape_cell` is `{"from": 4, "index": 21}`
— `fence_phase.py`'s `_pick_escape_cell` moves exactly one cell into
DREAMPlace's implicit "no fence" bucket (`node2fence_region_map[escape] = k`,
the (K+1)-th bucket) because the real K regions tile the whole die and
`PlaceDB.py`'s own filler-count code divides by zero on an empty implicit
bucket otherwise. `final_overflow = float(placer.model.overflow.max())`
(`run_main_flow.py:556`) takes the max over all K+1 per-fence-region overflow
entries, and DREAMPlace's own `NonLinearPlace.py` (lines 300-311) explicitly
singles out the *last* entry as "the outer cell overflow" that decides
stopping "for fence region" placements — i.e. the same escape bucket. A
bucket holding exactly one cell cannot satisfy a target-density-based
overflow metric (there's nothing to fill the rest of an arbitrarily large
leftover region with), so its overflow saturates near 1.0 regardless of how
well the K *real* fence regions converged, and `stop_overflow_reached =
(final_overflow <= stop_overflow)` (`run_placement.py:209`,
`0.99996 <= 0.07` here) reads that as "did not converge" even when the real
regions are fine. This is corroborating source-level cross-reference, not a
new instrumented run (none was authorized this fix round to confirm it
directly by capturing the per-region overflow vector) — flagged here as a
likely false negative worth fixing in a follow-up, and, like the LG-legality
caveat above, irrelevant to the anchor ranking.

## Artefacts

| file | sha256 |
|---|---|
| `runs/anchor_cmp/mempool_tile_wrap/soft.npz` | `ec1ad8ac2e8f698620e7f4054408c1cd3f80137594dee1b3dc08b189892ec0f1` |
| `runs/anchor_cmp/mempool_tile_wrap/placement.npz` | `e0ea766dd1f14ad7745a4f6e5faff711ed311e897a604fe2b5306fcdc2c98c09` |
| `runs/anchor_cmp/mempool_tile_wrap/result.json` | `9b9ec40f21ab1d5a4b986910b1e7967842b0e6ae38e7fa8d27f2427aa2065490` |
| `runs/anchor_cmp/mempool_tile_wrap/evaluation.npz` | `afbc3da6dedf37894e17f404de3e26cfaa4a4b991680049329b5b25d2a6201a3` |
| `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json` (fix round 2, includes `l1`) | `44d070ab55b969b7f04028d6eba9825e26914159720aa76ee4ee8a678cdd0a7b` |
| `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.md` (fix round 2) | `fb4174aa36d1e03799edaf46cf3c0b2098533d5727e2387133e8fe3bbd01e971` |
