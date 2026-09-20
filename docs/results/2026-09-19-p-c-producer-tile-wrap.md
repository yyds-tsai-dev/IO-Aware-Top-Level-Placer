# P-C region producer: `mempool_tile_wrap` acceptance run (K=16, 64² and 32²)

Task 11 of `.superpowers/sdd/2026-09-19-v2-p-c-region-producer/`. Real GPU
campaign on `mempool_tile_wrap` (127,739 movable / 129,033 physical cells,
145,589 nets), the last task of the P-C plan.

Host: shared H100 NVL, `CUDA_VISIBLE_DEVICES=3`. `IOPLACE_PYTHON =
/ldaphome/yyds-tsai-dev/DREAMPlace/.venv312/bin/python`.

## Commands

Config promotion (Steps 1/1b/1c):

```bash
source src/scripts/env.sh
mkdir -p benchmarks/ispd25/h100
cp results/route_gp_20260914/mempool_tile_wrap.json \
   benchmarks/ispd25/h100/mempool_tile_wrap.json
# mempool_group.json: copied from results/recovery_visible_20260906/configs/mempool_group.json
# with "gpu" forced from 0 to 1.
# mempool_cluster.json: copied from mempool_group.json with def_input swapped to
# .../benchmarks/ispd25/visible/mempool_cluster.def (UNVALIDATED, no run in this plan).
```

64² arm:

```bash
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m ioplace.drivers.run_region_producer \
  --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
  --k 16 --membership mtkahypar --extract-bins 64 --rect-max 8 \
  --dp-seed 1000 --deterministic 1 \
  --out results/p_c_producer_20260919/tile_wrap_k16_b64
```

32² arm (faithful GrandPlan, spec §8 arm (e)):

```bash
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m ioplace.drivers.run_region_producer \
  --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
  --k 16 --membership mtkahypar --extract-bins 32 --rect-max 8 \
  --dp-seed 1000 --deterministic 1 \
  --out results/p_c_producer_20260919/tile_wrap_k16_b32
```

Matched flat baseline (same config, same seed):

```bash
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m ioplace.drivers.run_placement --mode flat \
  --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
  --dp-seed 1000 --deterministic 1 \
  --out results/p_c_producer_20260919/tile_wrap_flat.json
```

Both producer runs exited 0 and wrote all four artefacts
(`regions.json`, `seed.npz`, `membership.npz`, `producer.json`) to their
respective directories. The flat run exited 0 and wrote
`tile_wrap_flat.json` + `tile_wrap_flat.json.npz` (final positions,
scaled units).

## Step 4: verification

`RegionSet.validate()` and `RegionGrid(rs)` (gapless, non-overlapping
tiling) pass on both arms; `rs.k == 16`, `rs.lattice == 512`,
`rect_max_observed <= 8` on both; `membership.npz`'s `source ==
"mtkahypar"` on both.

**The seed die-span assertion needed correcting from the brief's literal
form, and the correction itself is informative.** The brief's script
checks `seed.node_x`/`node_y` over the full `num_physical` range against
`die_native`. Run literally, this assertion **fails**: 1,274 of the 1,294
fixed (non-movable) nodes fall outside the die box (e.g. `node_x` as low
as ‑2400.0 against `die_native` x range `[20140, 1779920]`). This is not
a `*scale`/`/scale` transposition — cross-checked against the DEF parse
log, these are exactly the 1,274 primary-IO `terminal_NI` pins (`num_terminals
20, numFixed 20, num_terminal_NIs 1274`), which ISPD DEFs place at the chip
boundary, outside the placement core box `placedb` reports as `xl/yl/xh/yh`.
That is standard layout, not a producer bug: fixed terminals are placed by
the DEF, not by the GP/LG the coordinate contract governs.

Scoped to the **movable** prefix (the actual population the coordinate
contract concerns), the assertion passes cleanly on both arms:

```
tile_wrap_k16_b64: movable x in [20140.0, 1779160.0] die x [20140.0, 1779920.0];
  movable y in [19880.0, 1775480.0] die y [19880.0, 1778280.0]; 1274/1294 fixed
  nodes outside die (primary-IO terminal_NIs)
tile_wrap_k16_b32: movable x in [20140.0, 1779160.0] die x [20140.0, 1779920.0];
  movable y in [19880.0, 1775480.0] die y [19880.0, 1778280.0]; 1274/1294 fixed
  nodes outside die (primary-IO terminal_NIs)
```

Every movable cell's native coordinate is inside `die_native` on both
arms, with headroom below the die's upper edge (cells are legalized
standard-cell footprints, so `max < xh/yh` is expected). This is the
assertion that would have caught a `*scale`/`/scale` transposition on a
run where `scale_factor != 1`, and it holds.

**Coordinate contract values actually observed** (both arms, same config
so identical): `scale_factor = 0.002631578947368421` (= 1/380, i.e. a
380-unit site width — **not** 1.0, unlike `simple.json`), `shift_factor =
(20140.0, 19880.0)`, `die_native = (20140.0, 19880.0, 1779920.0,
1778280.0)`, `die_scaled = (0.0, 0.0, 4631.0, 4627.368...)`.

### Runtime table (Step 4)

| run | bins | rebuilds | max_rects | util_max/min | gp_s | lg_s | extract_s | sa_s | rectify_s | total_s |
|---|---|---|---|---|---|---|---|---|---|---|
| tile_wrap_k16_b64 | 64 | 15 | 8 | 11.053 | 13.2 | 0.5 | 0.6 | 7.9 | 0.2 | 36.3 |
| tile_wrap_k16_b32 | 32 | 15 | 8 | 5.701 | 14.0 | 0.5 | 0.6 | 2.7 | 0.0 | 31.2 |

SA time: **7.9 s (64²) / 2.7 s (32²), both well under the "SA < 60 s CPU"
budget** (spec §2).

## Per-region rects and utilisation

64² arm — `rects_per_region`: `[6, 5, 8, 7, 4, 6, 7, 4, 4, 4, 1, 2, 6, 5,
5, 6]` (max 8, at the cap, on region 2 only); `region_bins`: `[281, 264,
269, 177, 176, 192, 223, 146, 148, 59, 462, 363, 722, 236, 232, 146]`
(min 59, no region at 1 bin); `region_utilisation`: `[0.255, 0.345,
0.304, 0.416, 0.487, 0.383, 0.325, 0.516, 0.509, 1.198, 0.159, 0.196,
0.108, 0.309, 0.311, 0.471]` (region 9 is over-utilised at 1.198 —
`area_balance.max_over_min = 11.05`).

32² arm — `rects_per_region`: `[5, 6, 7, 5, 3, 2, 4, 7, 8, 3, 5, 5, 8, 6,
8, 5]` (max 8, at the cap, on regions 8 and 12); `region_bins`: `[65, 57,
59, 44, 51, 46, 57, 65, 63, 30, 178, 41, 106, 75, 50, 37]` (min 30, no
region at 1 bin); `region_utilisation`: `[0.276, 0.399, 0.346, 0.418,
0.42, 0.399, 0.318, 0.29, 0.299, 0.589, 0.103, 0.434, 0.185, 0.243,
0.361, 0.465]` (`area_balance.max_over_min = 5.70`, notably better
balanced than 64²).

`n_hull_rebuilds = 15` on both arms (701 GP iterations / `t_hull=50` →
14 scheduled rebuilds at iterations 50..700 plus the mandatory
iteration-0 rebuild = 15). `rect_max_path = "direct"` on both — see
below.

## Spec §10 risk 6: did K=16 at 64² bins hold up?

Yes, on the proxy signal the brief specifies (no region left at 1 bin
after the full pipeline): the minimum `region_bins` count was **59** at
64² and **30** at 32² — both well clear of 1. Note this is a proxy, not
direct instrumentation: `extract.ensure_nonempty` does not log when it
fires, and SA's area-balancing moves could in principle grow a
formerly-empty, `ensure_nonempty`-rescued single-bin region well past 1
bin by the time the run finishes, so a minimum of 59/30 bins does not by
itself prove `ensure_nonempty` never fired — only that no region ended up
looking like it had failed to recover. Adding direct instrumentation was
out of scope for this task. Under that caveat, density-argmax +
largest-CC extraction held up at K=16/64² on this design: no region
collapsed, `rect_max=8` was reachable directly (no fallback), and the 32²
map is a cleaner, better-balanced tiling (`max_over_min` 5.70 vs 11.05)
at the cost of a coarser membership grid.

## Spec §10 risk 1: the rect-budget path

Both arms report `sa.rect_max_path = "direct"` — **neither arm needed the
`--extract-bins 32` fallback**, cleaner than the plan's own measured
expectation (23/80 → 6/80 aborts at 64² over 20 seeds, without
elimination). `sa.rect_max_fallback_reason` is `None` on both. The 64²
`producer.json`'s own `extract_bins` field is 64 (not 32), confirming the
direct path actually held at the harder bin count on this design/seed.

## The λ activation ramp — data, and a note on the brief's stale assumption

**The brief's Step 5 text (and the plan's ruling D5/D6 as originally
written) assumes `TermNormalizer.register`'s default `n_ramp=20`, under
which λ_group would be exactly 0 on the activating transaction and the
grouping term would be inert for the first `t_hull=50` iterations.** That
assumption predates a same-day controller ruling recorded directly in
`run_region_producer.py:53-65` (`GROUP_N_RAMP = 0`): this driver
registers the group term with **`n_ramp=0`**, not 20, because
`TermNormalizer.lambdas` is read directly as the *committed* coefficient
elsewhere in the driver (the `_GroupAdapter`/`attach_terms` closure), while
the ramp lives only in `lam_applied` — an `n_ramp=20` would have applied
the un-ramped λ to the optimizer from iteration 0 while the trace recorded
a ramped 0, a silent trace-vs-force disagreement. With `n_ramp=0`,
`activation_ramp` is the step function `1{iteration >= it_activate}`, so
`lambda_group` is governed purely by Eq.3's own `wt` schedule (0.05,
stepping +0.05 every 100 iterations) from the very first probe.

The actual data confirms this: `lambda_group` is **nonzero at iteration
0** on both arms, not 0:

```
iter 0:   wt=0.05  lambda_group=2.8146e-05  grad_l1_group=1.40e7  grad_l1_wl=7903.4   ratio_ema=5.629e-04
iter 50:  wt=0.05  lambda_group=1.7213e-05  grad_l1_group=2.51e7  grad_l1_wl=3148.5   ratio_ema=3.443e-04
iter 100: wt=0.10  lambda_group=2.3196e-05  grad_l1_group=2.09e7  grad_l1_wl=2500.3   ratio_ema=2.320e-04
iter 150: wt=0.10  lambda_group=2.3508e-05  grad_l1_group=2.08e7  grad_l1_wl=4957.8   ratio_ema=2.351e-04
...
iter 650: wt=0.35  lambda_group=4.0456e-03  grad_l1_group=2.35e7  grad_l1_wl=352899.3 ratio_ema=1.156e-02
iter 700: wt=0.40  lambda_group=5.4841e-03  grad_l1_group=2.40e7  grad_l1_wl=379942.0 ratio_ema=1.371e-02
```

(Identical on both arms — extraction bin count does not affect the GP
run.) `wt_final=0.4`, `lambda_group_final=0.005484`,
`ratio_ema_final=0.01371`. The soft start visible here is Eq.3's `wt`
ramp (0.05 → 0.4 over the 701-iteration run), not a second activation
ramp on top of it — exactly the documented rationale for
`GROUP_N_RAMP=0`.

## Macro enrichment

`mempool_tile_wrap` has 0 movable macros (confirmed: `Macro legalization:
regard 0 cells as dummy fixed (movable macros)` in the DP log, and the
driver's `macro_idx` is computed from `is_macro = size_y > 2*row_h` over
the **movable** prefix only). `macro_pseudo_points` never ran. The `≤64
points/macro` spec line is **not exercised** by this task.

## Producer overhead vs. a matched flat run — the GP+LG comparison is the headline

Total-vs-total is **not** a valid comparison here and is reported only to
show why: the flat driver's `runtime_s` includes a full `eval` phase (the
GPU evaluator: crossings, feed-through, congestion proxies) that the
producer driver has no counterpart for at all. Measured: flat
`t_eval = 34.85 s` against a flat `t_gp + t_lg = 5.74 s` — the eval phase
alone is 6x the flat run's placement time. Naively comparing
`producer.total` (36.3 s / 31.2 s) against `flat.runtime_s` (48.7 s) would
report a **negative** "overhead" (‑25.5% / ‑35.9%), which is nonsense —
an artefact of the eval-phase asymmetry, not evidence the producer is
faster.

**Original figure below is STALE — it predates a hull optimisation and has
been superseded by the re-measurement in the next subsection.** The two
runs this original table is built from were taken before
`src/ioplace/producer/hull.py` was optimised (commits `2004981`, `d5206ff`,
`05ffc45`, 2026-09-19/20): `reduce_candidates_torch` was found to be
82-84% of the producer's GP overhead because `keep[idx] = True` with a
Python `bool` forced a blocking pageable host→device copy on every call;
it is now batched across all `2m` directions in one pass, measured at
roughly 11x faster per region. Kept here for the historical record only —
**do not cite the numbers in this subsection as current.**

| arm | producer gp+lg | producer iters | flat gp+lg | flat iters | overhead |
|---|---|---|---|---|---|
| b64 | 13.704 s (gp 13.211, lg 0.493) | 701 | 5.740 s (gp 5.445, lg 0.295) | 649 | +138.7% (stale) |
| b32 | 14.434 s (gp 13.966, lg 0.468) | 701 | 5.740 s (gp 5.445, lg 0.295) | 649 | +151.5% (stale) |

### Re-measurement after the hull optimisation (2026-09-20)

The matched pair (producer 64² arm + flat baseline, same config, same
`--dp-seed 1000 --deterministic 1`) was re-run back-to-back in one
contention window on `CUDA_VISIBLE_DEVICES=3`, so the comparison is
internally consistent against the same host state. `nvidia-smi` before and
after the window showed GPU 3 holding 21,702 MiB from other processes'
resident allocations but **0% utilisation** throughout — i.e. memory
pressure from co-tenants but no compute contention observed during this
specific window. (The brief notes this host has shown ±25% run-to-run
variance elsewhere; this is a single matched-pair measurement, not an
average over repeated runs, so some of that variance band should be
assumed around the number below.)

Outputs were written to a scratch location, not into
`results/p_c_producer_20260919/`, to avoid disturbing the already-verified
committed artefacts from Steps 2-4 above; the re-measurement is a timing
check only, not a new deliverable run. `n_hull_rebuilds = 15`,
`rect_max_path = "direct"`, `extract_bins = 64` on the re-measured
producer run — identical to the original 64² arm, confirming this is the
same workload, just re-timed post-optimisation.

| arm | producer gp+lg | producer iters | flat gp+lg | flat iters | overhead | ratio |
|---|---|---|---|---|---|---|
| b64 (re-measured) | 10.534 s (gp 10.088, lg 0.446) | 701 | 5.527 s (gp 5.217, lg 0.310) | 649 | **+90.6%** | **1.906x** |

Per-iteration GP cost: producer 14.39 ms/iter vs. flat 8.04 ms/iter — a
1.79x per-iteration slowdown (down from the stale run's 2.25x-2.37x).

**This is a real, substantial improvement from the hull optimisation —
overhead dropped from +138.7% to +90.6%, roughly a 1.5x reduction in the
GP+LG delta — but the result still misses the recorded gate.** The gate
(a documented spec deviation replacing spec §2's unreachable "<3%"):
producer GP+LG ≤ **1.35x** a matched flat GP+LG, and ≤10% of end-to-end
flow wall time. **The measured ratio is 1.906x, which is a clear miss of
the 1.35x gate** — the post-hull-fix projection of ≈1.27x chained two
unmeasured steps and turned out to be well outside this host's noise
band; this measurement, not the projection, is authoritative. No
adjustment has been made to try to make this pass; it is reported as
measured.

The second half of the gate ("≤10% of end-to-end flow wall time") cannot
be assessed from this task's instrumentation: the producer driver has no
counterpart to a full downstream flow (detailed placement, legalization,
routing), so there is no "end-to-end flow wall time" figure to divide
into. As a weaker proxy, the absolute GP+LG delta (10.534 − 5.527 =
5.007 s) is 16.2% of the producer's own `total` (31.02 s) — itself not
the quantity the gate names, and offered only as a data point, not a
gate verdict.

**Residual non-reduction overhead.** After the reduction fix, 1.79x
per-iteration slowdown remains, and this task's instrumentation still
cannot decompose it into hull-rebuild wall-clock,
`refresh_nesterov_secant`'s extra objective evaluations, and the
`GroupingTerm` forward+backward, because no per-phase-within-GP timer
exists below the single `gp` phase total — the same limitation the stale
measurement had. One documented, non-reducible contributor is
`anchor_tables`: it is a **fixed `K × 512²` cost**, independent of cell
count N, computed once per hull rebuild (15 times over 701 iterations
here). `mempool_tile_wrap` is only 127K movable cells, so this fixed cost
is a much larger fraction of its GP time than it would be on a
multi-million-cell design where per-iteration GP cost scales with N while
the anchor-table rebuild cost does not — **this design's percentage
overhead should be read as an upper bound that a larger design would not
be expected to reproduce**, not a scale-invariant figure.

`peak_mem_mb = 233.1 MB` on both arms (identical GP run). For scale
against the plan's carried number ("~80 MB per million cells" for the
anchor-table lookup transient specifically, "~2.4 GB extrapolated to
30M"): this design has 0.1277M movable cells, so 233.1 MB / 0.1277M ≈
1825 MB per million cells — but `peak_mem_mb` here is the whole GP
phase's peak allocation (all DREAMPlace tensors plus the anchor tables),
not the anchor-table transient in isolation, and most of a GP phase's
memory scales with N while the K×512² anchor tables do not. **This
number is not a like-for-like check against the carried 80 MB/M figure**
and should not be extrapolated linearly to 30M cells without isolating
the anchor-table-only allocation, which this run's instrumentation does
not do.

## Does the grouping force move the placement? — the central question, answered

**Yes, dramatically — this is not a no-op.** Same config, same
`--dp-seed 1000 --deterministic 1` (so the same `np.random.seed`-derived
initial centre/filler positions in both runs, seeded immediately before
`NonLinearPlace` construction in both drivers), same DREAMPlace version:
the only difference between the flat run and the producer run is the
grouping term's presence.

**HPWL** (both `hpwl_gp`/`hpwl_lg` are SCALED, `native_hpwl/site_width`,
comparable as-is per the global-constraints coordinate-contract note):

| | hpwl_gp | hpwl_lg |
|---|---|---|
| flat | 20,583,080 | 20,709,318 |
| producer (b64/b32, identical GP) | 23,733,912 | 23,780,930 |

Producer's post-LG HPWL is **+14.83%** higher than flat's — expected and
consistent with the grouping term's design intent (it trades wirelength
for K-way spatial compactness), and itself proof the term is exerting
real force on the optimisation, not sitting at zero gradient the way
`simple.json`'s frozen case does.

**Per-cell displacement** (native units, movable cells only, producer's
post-LG `seed.npz` positions vs. flat's post-LG positions converted from
scaled to native with the *same* `scale_factor`/`shift_factor` — both
runs derive these purely from the DEF's site width/die box, independent
of run mode, so they are identical across the two runs):

| | b64 | b32 |
|---|---|---|
| mean displacement | 318,246.6 | 318,246.6 |
| median | 256,985.0 | 256,985.0 |
| p95 | 846,433.4 | 846,433.4 |
| p99 | 1,110,250.7 | 1,110,250.7 |
| max | 1,358,215.9 | 1,358,215.9 |
| cells moved > 1.0 native unit | 127,732 / 127,739 (99.99%) | same |
| bit-identical cells | 7 / 127,739 | same |

(Identical between b64/b32 because both share the same GP+LG run — only
extraction differs downstream.)

Die width is 1,759,780 native units (x) / 1,758,400 (y). A mean
displacement of 318,247 units is **≈18.1% of the die's width** — cells
are not nudged, they are relocated across a substantial fraction of the
chip. 99.99% of movable cells moved by more than 1 native unit; the 7
bit-identical cells are a negligible remainder (plausibly degenerate
zero-size or otherwise-constrained cells, not investigated further).

**Conclusion: the grouping force is unambiguously live and moves the
placement by a large margin on this design — the opposite of
`simple.json`'s frozen GP, where `sum|x|` was bit-identical across all
200 callbacks.** This is the first real evidence in the plan that the
term does what it is designed to do, at the cost of a substantial (and,
per the overhead section above, more expensive than budgeted) wirelength
and runtime penalty.

## Full suite

```
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```

See the test summary in the task report / handback for the pass/fail
count actually observed.

## Files

- `benchmarks/ispd25/h100/{mempool_tile_wrap,mempool_group,mempool_cluster}.json`
- `results/p_c_producer_20260919/tile_wrap_k16_b64/{regions.json,seed.npz,membership.npz,producer.json}`
- `results/p_c_producer_20260919/tile_wrap_k16_b32/{regions.json,seed.npz,membership.npz,producer.json}`
- `results/p_c_producer_20260919/tile_wrap_flat.json` + `.npz` (matched flat baseline, not part of the brief's original artefact list but required for the overhead/displacement comparison above)
