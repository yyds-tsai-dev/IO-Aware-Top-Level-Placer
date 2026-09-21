# P-F — Soft-assign anchor comparison (design v2 §7)

**Status:** filled from `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json`,
produced 2026-09-20 on the H100 NVL host (GPU 3). **Revised 2026-09-21 (fix
round 2)** after an adversarial review found the original scalar comparison
degenerate. **Revised again 2026-09-21 (fix round 3)**: the per-net L1
rescoring itself was verified bit-exact and accepted, but a further review
found what the result actually *means* was overclaimed, and added a second
placement (`--node-anchor lower_left`, same config/seed) to turn this from a
convention check into an actual placement-quality comparison. Read
"Fix round 3" below before "Answer."
**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §7,
"Verification experiment".
**Plan:** `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` Task 7.

## Question

From one soft solution, does the cell-centre anchor predict post-fence-LG IO
better than the node lower-left or the pin anchor? Spec §7 expects centre
closest. **If `pin` is closest, LG displacement exceeds half a cell and the
legaliser must be examined** — that is a finding about LG, not about the anchor.

## Fix round 3: what the per-net L1 result can and cannot mean

Fix round 2 replaced the degenerate scalar comparison with a per-net paired
L1 and found centre closest with a bootstrap CI excluding zero. That
rescoring was independently verified bit-exact against the production
`IoTerm` forward and is not in question. What fix round 3 corrects is the
**interpretation**.

**The truth's ownership convention IS the cell centre, by construction.**
`src/ioplace/freeze.py:9` defines membership as "the argmax region of the
**cell centre**." `src/ioplace/drivers/run_main_flow.py`'s `centre_argmax`
(used by the freeze monitor to decide membership) calls `cell_centers`
**unconditionally, regardless of `--node-anchor`** — the flag only changes
which anchor the *IO surrogate* is evaluated at during optimisation, never
which anchor the *freeze* uses to assign membership. That membership is then
hard-enforced at fence-aware legalisation (`fence_compliance_center = 1.0`
throughout this experiment). So a centre-anchored surrogate on the same soft
positions is a smooth relaxation of the *very membership the fence flow
already enforces* — it is not an independent predictor being tested against
an unrelated truth. **§7's experiment, run this way, cannot falsify the
centre choice**: centre must win as τ→0, by the flow's own construction, not
because it happens to model something the other two anchors don't.

**A τ sweep on the existing artefacts confirms this directly** (no new
placement needed — same soft solution, same evaluation.npz, just a smaller τ
fed to the same surrogate):

| τ | lower_left L1 | center L1 | pin L1 |
|---|---|---|---|
| τ₀ = 35.679 | 7168.94 | 7065.71 | 7168.67 |
| τ₀/4 = 8.920 | 2017.32 | 1707.45 | 1781.63 |
| τ₀/16 = 2.230 | 1229.33 | 515.84 | 685.04 |
| τ₀/100 = 0.357 | 1185.12 | **159.18** | 427.88 |

Centre's advantage grows sharply as τ hardens (roughly 1.4% of `lower_left`'s
L1 at τ₀, to a large multiple by τ₀/100) — exactly the behaviour expected of
a quantity that *is* the enforced convention, becoming exact as the
relaxation tightens.

**This also refutes a hypothesis this doc was about to record: that `pin`
converges to the truth as τ→0, since the truth is itself pin-anchored
(`per_net_lambda`, `evaluator_gpu.py`).** It does not. At τ₀/100, `pin`'s L1
(427.88) is 2.7× *worse* than centre's (159.18) — because the truth is
**post-fence-LG** pin positions under **centre-argmax enforced ownership**
(`straddle_pin_split_nets = 0` — no net's pins are ever split across the
fence boundary from a different cell's pin position than its owning cell),
not raw pre-LG pin geometry. The rule is testable and has real power (the
arms separate 3–7× as τ hardens); it simply never fires in `pin`'s favour on
this benchmark. Do not read "the rule is untestable at this τ" out of this
sweep — it is not; it has power and consistently favours centre.

**The honest, earned claim:** the centre anchor is the surrogate convention
*consistent with* the freeze/fence ownership the flow already enforces,
confirmed quantitatively on this one placement and increasingly so as τ
hardens. This justifies keeping `--node-anchor center` as the flow's
ownership **convention** — it is not a claim that centre *improves placement
quality*. That is a separate question, addressed for the first time in
"The 2×3 experiment" below (authorised and run this round).

## Answer

**Centre is the internally-consistent anchor, confirmed by construction and
by the τ sweep — not "the better predictor" in a sense that could have come
out otherwise.** `pin` is never closest at any τ tested, so §7's "examine
LG" branch does not trigger. `--node-anchor center` stays the default, on
consistency grounds. Whether centre also yields a *better final placement*
than `lower_left` is answered (with a genuine trade-off, not a clean win) in
"The 2×3 experiment" below — run this round because the coordinator judged
it worth the ~90 s of GPU time on this benchmark.

See "What this experiment cannot show" for the scope this all sits inside.

## Effect size, honestly stated

Two numbers the earlier passes did not report, and should have:

- **Win rate.** At τ₀, centre's per-net absolute error is smaller than
  `lower_left`'s on 73,550 of 136,208 band nets, larger on 62,591, tied on
  67 — a **54.0%** win rate. Sign test (normal approximation, ties excluded):
  z ≈ 29.7 (i.e. this win rate is not chance at this sample size, but 54%
  is a modest majority, not a landslide).
- **Trivial baseline.** Scoring the trivial predictor `λ≡1` for every net
  (the "no information" baseline: one region, no crossing) against the same
  truth gives L1 = 11,601. Centre recovers `11,601 − 7,065.71 = 4,535.29` of
  that baseline gap; `lower_left` recovers `11,601 − 7,168.94 = 4,432.06`.
  Centre's *extra* recovered signal over `lower_left` is `4,535.29 −
  4,432.06 = 103.23` — **≈2.3% of centre's own recovered signal**
  (`103.23 / 4,535.29`). This is the honest effect size: real, bootstrap-
  confirmed, and small.

Two further bootstrap CIs the earlier passes should have reported (same
10,000-resample paired methodology as `center`-vs-`lower_left`):

| comparison | mean(d) | 95% CI | verdict |
|---|---|---|---|
| center − lower_left | −0.00076 | [−0.00086, −0.00066] | centre better, confirmed |
| pin − lower_left | −0.0000020 | [−1.90e-4, +1.91e-4] | **tied, confirmed** (CI straddles 0) |
| pin − center | +0.00076 | [+5.91e-4, +9.24e-4] | centre beats pin too, confirmed |

`lower_left` and `pin` are a **confirmed** statistical tie, not merely "not
tested" as fix round 2 said — this round ran the bootstrap fix round 2
skipped.

## What this experiment cannot show

- **One soft solution, one placement per anchor tested.** No variance
  across seeds is measured; the CIs above are net-to-net *within* one
  placement, not run-to-run. A different seed could shift the win rate.
- **Scale.** 129,033 nodes on `mempool_tile_wrap` — about 1/100th of the
  10M–30M-cell target this flow is built for. Nothing here says the effect
  holds, grows, or vanishes at target scale.
- **K=16, one geometry (4×4 grid), one τ trajectory** (this soft run's own
  freeze history). Not swept.
- **No placement-quality claim in the L1 result itself.** The L1 metric
  scores a surrogate against a truth that is defined by the same convention
  the surrogate being tested happens to share (see "Fix round 3" above) — it
  is a consistency check, not evidence that centre-anchored optimisation
  produces a better chip. The 2×3 experiment below is the first thing in
  this doc that speaks to placement quality at all, and even that is one
  placement pair, not a swept claim.

State plainly: this justifies keeping `--node-anchor center` as the flow's
ownership **convention** (it is the anchor consistent with what freeze/fence
already enforce), not a claim that centre **improves placement quality**.

## The 2×3 experiment (authorised this round): does the anchor used to *run*
optimisation change the final placement?

Everything above scores three surrogate anchors against the truth of a
single soft solution that was itself produced with `--node-anchor center`.
This section reruns the **entire** soft→freeze→fence→LG flow a second time
with `--node-anchor lower_left` (same config, same `--dp-seed 1000`, same
K/geometry) — a different anchor is now driving the actual GP gradient, not
just the diagnostic scoring — and compares final placement quality.

**Row 1 — flow run with `--node-anchor center`** (`runs/anchor_cmp/mempool_tile_wrap/`):

| surrogate anchor | scalar abs_err | per-net L1 |
|---|---|---|
| lower_left | 4702.9 | 7168.94 |
| **center** | 4636.4 | **7065.71** |
| pin | 4871.3 | 7168.67 |

Final placement: `hard_lambda_sum=11,658`, `io_count=14,908`,
`hpwl=49,183,571.8`, `hpwl_gp=89,658,064`, `gp_iterations_fence=205/2000`.

**Row 2 — flow run with `--node-anchor lower_left`** (`runs/anchor_cmp/mempool_tile_wrap_ll/`):

| surrogate anchor | scalar abs_err | per-net L1 |
|---|---|---|
| lower_left | 4718.4 | 7151.8 |
| **center** | 4682.7 | **7087.7** |
| pin | 4916.6 | 7189.0 |

Final placement: `hard_lambda_sum=11,758`, `io_count=15,236`,
`hpwl=30,058,489.7`, `hpwl_gp=52,350,104`, `gp_iterations_fence=2000/2000`.

Paired bootstrap, row 2's center-vs-lower_left: mean(d) = −0.00047, 95% CI
[−0.00057, −0.00037] — **centre wins the per-net L1 comparison in both rows**,
regardless of which anchor actually drove the optimisation. This is further
confirmation of "Fix round 3"'s reading: centre's advantage in this metric
tracks the enforced convention, not the anchor that happened to run GP.

### Final-placement comparison: is `lower_left`'s placement better, equal, or worse?

**Neither anchor produces an unambiguously better final placement — it is a
real trade-off, and it comes with a confound that has to be named plainly.**

| metric | center run | lower_left run | which is "better" |
|---|---|---|---|
| `hard_lambda_sum` (excess region crossings) | 11,658 | 11,758 (+0.86%) | **center** (P-F's own target metric) |
| `io_count` (MST crossings) | 14,908 | 15,236 (+2.2%) | **center** |
| `hpwl` (final wirelength) | 49,183,572 | 30,058,490 (**−38.9%**) | **lower_left**, by a large margin |
| `hpwl_gp` (wirelength at fence-GP end, pre-LG) | 89,658,064 | 52,350,104 (−41.6%) | **lower_left** |
| `gp_iterations_fence` | 205 / 2000 | 2000 / 2000 | — (see confound below) |

**The confound: the two fence GPs did not run for the same length of time.**
The center run's fence GP self-terminated at iteration 205 of a 2000
budget; DREAMPlace's own stopping heuristic
(`NonLinearPlace.py` `Lgamma_stop_criterion`, `len(placedb.regions) > 0 and
model.update_mask.sum() == 0` — "all regions stop updating, finish global
placement") fired because every fence region's own convergence mask had
already zeroed out. The `lower_left` run's fence GP never hit that
condition and used the entire budget. Wirelength is exactly the kind of
quantity more GP iterations reduce, and `hpwl_gp` is already ~42% lower for
`lower_left` *before LG even runs* — so a large share of the final `hpwl`
gap traces to **how far each fence GP got**, which itself differs because
the two soft phases (running with different anchors) handed the fence GP
different starting cell distributions and region-occupancy targets, not
because `lower_left` is an inherently better GP anchor. This experiment
cannot separate "different anchor → different soft-phase cell distribution →
different fence-GP stopping time → different wirelength" from "different
anchor → directly better wirelength outcome" — that would need a controlled
follow-up (e.g. forcing both runs to the same fence-GP iteration count), not
authorised this round.

**Reading:** on the metric P-F exists to reduce (IO crossings —
`hard_lambda_sum`, `io_count`), the `center`-anchor run's final placement is
mildly better (≈1–2%). On wirelength, `lower_left`'s final placement is
substantially better, but that gap is largely explained by an early-stopping
artefact in the center run's fence GP rather than demonstrated to be an
inherent property of the anchor choice. Neither observation moves "which
anchor to keep as the flow's default convention" — that question is settled
by the freeze/fence coupling in "Fix round 3," independent of which anchor
happens to win on wirelength in one confounded run. If wirelength under
`--node-anchor center` turns out to be a real, anchor-caused regression
(not just this run's early fence-GP stop), that is a separate, follow-up-
worthy finding about the fence GP's stopping heuristic under fences, not
about which anchor the diagnostic surrogate should use.

**Provenance footnote — a driver quirk found while producing row 2.**
`result.json["node_anchor"]` reads `"center"` for **both** rows above,
including the `lower_left` run. This is because `--phase fence` (run as a
separate invocation here, per the preferred two-call protocol) accepts its
own `--node-anchor` argument and stamps `result.json`'s top-level field from
*that* call — defaulting to `"center"` since none was passed the second
time — even though the flag is a genuine no-op for `--phase fence`'s own
computation (IO/FT terms are off after freeze). The soft phase itself
**did** run with `node_anchor="lower_left"` (passed explicitly, and
confirmed by the substantially different `io_soft`/`io_fence_gp`/`tree_wl`
values between the two rows, which a no-op couldn't produce) — so this
experiment's row 2 is not compromised. But anyone reading `result.json` in
isolation would be misled about which anchor a split soft/fence run
actually used. Worth a follow-up fix (persist `node_anchor` into
`freeze.json` at the soft phase and have `--phase fence` read it from there
rather than accepting a fresh CLI default) — not fixed here, out of this
task's scope.

## Protocol

| item | value |
|---|---|
| case | `mempool_tile_wrap` |
| config | `results/route_gp_20260914/mempool_tile_wrap.json` (ruling R-3). `benchmarks/ispd25/h100/mempool_tile_wrap.json` landed on this branch during this task (commit `4f88774`) and is **byte-identical** (`sha256` `d81e0faf3191…de09aec` for both) — the R-3 choice is recorded for traceability, not because the two differ. |
| K / geometry | 16 / grid 4×4 |
| soft solution (row 1) | `run_main_flow --phase soft --node-anchor center --dp-seed 1000 --init die_center` → `soft.npz`, `freeze.json` (froze at GP iteration 648, reason `gp_end`, τ=35.679, τ_rel=0.0308, overflow=0.0798) |
| soft solution (row 2) | same, `--node-anchor lower_left` → `runs/anchor_cmp/mempool_tile_wrap_ll/` |
| fence LG | `run_main_flow --phase fence --dp-seed 1000` → `placement.npz`, `result.json`, `evaluation.npz` (both rows) |
| τ | `freeze.json["tau"]` = 35.67911205963874 for row 1 (the τ sweep in "Fix round 3" divides this by 4/16/100); row 2 has its own freeze τ, used for its own 3-anchor table |
| surrogate | `L_IO = Σ_e max(λ_e − 1, 0)`, `w_mode=unit`, `λ_io=1`, no margin |
| scalar truth (degenerate, kept for provenance) | `result.json["hard_lambda_sum"]` on the post-fence-LG placement |
| per-net truth (fix round 2) | `evaluation.npz["per_net_lambda"]`, pin-anchored (see "Fix round 2" note below), paired by net id after `load_evaluation`'s net-order check |
| net-order identity check | `evaluation.npz`'s `metadata.net_order_sha256` = `e594705775c0622e556356712dc0b0bfa132cc849eacd5ba0d1f6a6201dcab5f` (row 1), confirmed equal to `array_digest(normalized_names(placedb.net_names))` recomputed from a fresh `load_netlist(config)` — **PASSED**, so the per-net pairing is meaningful |
| script | `"$IOPLACE_PYTHON" src/scripts/run_anchor_comparison.py --config <cfg> --out-dir <out-dir> --k 16` (runs `net_l1_report` whenever `evaluation.npz` is present) |
| host | H100 NVL, `CUDA_VISIBLE_DEVICES=3` (GPUs 0–2 foreign/busy at the time; GPU 3 idle before and during every run) |

## Result — the scalar table (degenerate; kept for provenance, not the answer)

Row 1 (`--node-anchor center` flow):

| anchor | l_io_soft | lambda_sum_soft | io_lb_final | abs_err | rel_err | io_count_final | closest |
|---|---|---|---|---|---|---|---|
| lower_left | 16360.9 | 152520.9 | 11658 | 4702.9 | 0.4034 | 14908 | |
| center | 16294.4 | 152454.4 | 11658 | 4636.4 | 0.3977 | 14908 | yes |
| pin | 16529.3 | 152689.3 | 11658 | 4871.3 | 0.4178 | 14908 | |

(`n_active` = 136160 nets for every row.) **`closest` here is `argmin
l_io_soft` in disguise (see "Fix round 2") — do not read this table as the
sec 7 answer on its own.** It agrees with the per-net L1 rescoring on the
winner (`center`) but disagrees on the ranking of the other two, which the
L1 rescoring (and the further bootstraps this round) shows are a confirmed
tie.

## Per-net L1 rescoring (row 1, the primary run)

`Σ_e |λ_soft_e - λ_hard_e|`, restricted to `2 <= net_degree < 100`
(136,208 nets in band, out of 145,589 total), at τ₀ = 35.679:

| anchor | per-net L1 |
|---|---|
| lower_left | 7168.94 |
| **center** | **7065.71** |
| pin | 7168.67 |

See "Effect size, honestly stated" above for the win rate, trivial baseline,
and the three pairwise bootstrap CIs (center-vs-lower_left,
pin-vs-lower_left, pin-vs-center). See "Fix round 3" for the τ sweep.

## Straddle context (row 1's `result.json`)

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

**On the ±52k terms and the negative `lg_loss`.** `io_delta_at_freeze =
+52243` and `lg_loss = -52093` are credible measurements from the same
evaluator in the same coordinate frame (the identity residual closes at 0
against them). The large negative `lg_loss` — post-fence-LG IO *dropping*
relative to the fence GP's own IO — is **fence enforcement, not
row-snapping**: `inject_fence_regions` (`fence_phase.py`) sets
`node2fence_region_map` so the fence GP only *softly* pulls cells toward
their assigned region, while the fence-aware legalizer supplies the actual
hard guarantee. Cross-checked against `runs/norm-validation{,-r2,-r3}/
{legacy,grandplan,adaptive}.json` (non-fence norm-policy validation runs, no
fence LG involved): every one of those reports a small **positive**
`lg_loss` (993 to 2942, `hpwl_lg > hpwl_gp`), except two `-r2` runs that
diverged during GP (hundred-thousands, a divergence artifact). This run's
sign-inverted, much larger `lg_loss` is symptomatic of a comparatively weak
fence GP on `mempool_tile_wrap` at this K/seed — and "The 2×3 experiment"
above now has a concrete mechanism for *why* (the `update_mask.sum()==0`
early stop at iteration 205/2000) — but it remains irrelevant to the anchor
ranking, which never reads `io_delta_at_freeze` or `lg_loss`.

## Reading

**Centre is the internally-consistent choice; it is not shown to be a
better predictor in a sense that could have failed.** See "Fix round 3."
`pin` is never closest at any τ, so the LG-displacement investigation §7
would trigger on a `pin` win never fires. Keep `--node-anchor center` as the
default, on consistency grounds (confirmed by construction and by the τ
sweep), not on an unearned "better predictor" claim.

**Withdrawn (fix round 2): the pin-bias explanation from the first pass.**
The original doc read `pin`'s worse scalar `abs_err` as "the pin arm
over-counts multi-pin cells and so pushes the surrogate further from the
hard truth." `pin` and `lower_left` are now a **confirmed** statistical tie
(bootstrap CI [−1.90e-4, +1.91e-4], straddles zero) — the old scalar's
"pin worst" ordering was an artefact of magnitude cancellation, not a real
finding.

**Caveat — `legalization_status: failed` is the escape cell, not general LG
robustness on `mempool_tile_wrap`.** The *only* legality errors in the
entire fence-LG log (four lines, both runs) are all on **node 21**, and
`result.json["escape_cell"]` is `{"from": 4, "index": 21}` in both rows —
the same cell `fence_phase.py`'s `_pick_escape_cell` moved into
DREAMPlace's implicit "no fence" bucket (`node2fence_region_map[escape] =
k`) because the real K regions tile the whole die and DREAMPlace's own
filler-count code divides by zero on an empty implicit bucket otherwise.
The legalization log's own island-level trace shows exactly why: the escape
bucket's own greedy-legalization pass reports `#bin_objs = 0` and
`num_unplaced_cells = 1` — a zero-bin region being asked to legalize one
cell, a direct, structural consequence of `fence_phase.py:134-142`'s
escape-bucket mechanism, not a row-alignment robustness problem with this
benchmark generally.

**Reporting discrepancy, flagged rather than smoothed over:** the fence-LG
log's own island trace says `num_unplaced_cells = 1` for the escape bucket
at the point greedy legalization runs on it, but `result.json`'s
`num_unplaced_cells` field reads **0**. These are different definitions —
the log's count is DREAMPlace's own per-island greedy-legalization counter
at that pass, while `result.json`'s field (`_legalization_diagnostics`,
`run_placement.py`) checks only whether the *final* position is finite and
inside the die box, which node 21 satisfies (it ends up somewhere in the
die, just misaligned to its row/site) — hence `legal=False` from
`legality_check_op` alongside `num_unplaced_cells=0` from the narrower
check. The earlier pass of this doc leaned on the `0` as if it meant "no
cell is problematic"; it does not, and this discrepancy should have been
named at the time.

**`final_overflow: 0.99996` / `stop_overflow_reached: False` — CONFIRMED
false negative, not "very likely."** The full chain, all directly readable
from the existing artefacts (no new run needed):
`result.json["escape_cell"] = {"from": 4, "index": 21}` confirms the escape
bucket holds exactly one cell; `final_overflow = float(placer.model.overflow.max())`
(`run_main_flow.py:556`) takes the max over all K+1 per-fence-region
overflow entries; DREAMPlace's own `NonLinearPlace.py:300-311` explicitly
reads only the *last* entry ("the outer cell overflow") to decide fence-
region GP stopping, i.e. the same escape bucket, never the max; a bucket
holding one cell in an arbitrarily large leftover region cannot satisfy a
target-density overflow metric and saturates near 1.0 regardless of how
well the K real regions converged; `_stop_overflow_reached`
(`run_placement.py:203-209`) then compares this saturated 0.99996 against
`stop_overflow=0.07` and reports `False`. Corroborated by
`gp_iterations_fence = 205/2000` (the run stopped on the *unrelated*
`update_mask.sum()==0` "all regions" criterion, not on overflow, which
never came close to triggering) and by `hpwl_lg=49.18M` against
`hpwl_gp=89.66M` (LG's usual cleanup, not evidence the K real regions were
still diverging). **This is now stated as confirmed, sourced entirely to
existing code and the existing run's own artefacts** — not something a new
instrumented run was needed to establish. Per the coordinator's ruling, the
fix (record the full overflow vector and feed `overflow[-1]` — or an
equivalent that excludes the escape bucket — to `_stop_overflow_reached`
when `len(placedb.regions) > 0`) is out of this task's scope and is routed
separately; it is *not* fixed here.

## Artefacts

| file | sha256 |
|---|---|
| `runs/anchor_cmp/mempool_tile_wrap/soft.npz` | `ec1ad8ac2e8f698620e7f4054408c1cd3f80137594dee1b3dc08b189892ec0f1` |
| `runs/anchor_cmp/mempool_tile_wrap/placement.npz` | `e0ea766dd1f14ad7745a4f6e5faff711ed311e897a604fe2b5306fcdc2c98c09` |
| `runs/anchor_cmp/mempool_tile_wrap/result.json` | `9b9ec40f21ab1d5a4b986910b1e7967842b0e6ae38e7fa8d27f2427aa2065490` |
| `runs/anchor_cmp/mempool_tile_wrap/evaluation.npz` | `afbc3da6dedf37894e17f404de3e26cfaa4a4b991680049329b5b25d2a6201a3` |
| `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json` | `44d070ab55b969b7f04028d6eba9825e26914159720aa76ee4ee8a678cdd0a7b` |
| `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.md` | `fb4174aa36d1e03799edaf46cf3c0b2098533d5727e2387133e8fe3bbd01e971` |
| `runs/anchor_cmp/mempool_tile_wrap_ll/soft.npz` (row 2, `--node-anchor lower_left`) | `3f1e2d95e2152e383415d3064f4612c4a05cac5f0389b367efabd9466502cf46` |
| `runs/anchor_cmp/mempool_tile_wrap_ll/placement.npz` | `93fd416bccfa71fccf6accbe5232d17d712c82f8c67c978847ef7a932462daa7` |
| `runs/anchor_cmp/mempool_tile_wrap_ll/result.json` | `502222562e354a70785ac293ada6909c8411e87f6ec57daba1826a936ed5f58c` |
| `runs/anchor_cmp/mempool_tile_wrap_ll/evaluation.npz` | `3c2d4818788ef4971ab5154d8fe64b8aadbbb0125cd8993185ef8adf907c76f4` |
| `runs/anchor_cmp/mempool_tile_wrap_ll/anchor_comparison.json` | `e97479fcb29b9fb55b5a21106c03b4f719a758594fa20e95d4a8d449d16b7c56` |
| `runs/anchor_cmp/mempool_tile_wrap_ll/anchor_comparison.md` | `4e019b1c68dc16e9806a0886117a840706d7d7a07ae58e9675d2a88611727783` |
