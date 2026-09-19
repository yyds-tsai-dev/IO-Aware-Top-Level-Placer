# v2 IO-Aware Top-Level Placer — Redesign Design
Date: 2026-09-19. Status: approved design (brainstorming session 2026-09-18/19). Supersedes the round-feedback line as the main flow.

## 0. Decisions (fixed)

| Decision | Value |
|---|---|
| Problem input | netlist + partition geometry (rectilinear RegionSet); cell→partition membership decided by placement (soft-assign), not given |
| Drivers | two independent drivers (option A): region producer (GrandPlan replication) and main flow; communicate only through files |
| Main flow | soft-assign GP → freeze membership (overflow ≤0.15, τ_rel ≤0.05, membership churn ≤0.5% over 50 iters; membership = argmax at cell centre) → DREAMPlace fence-region GP warm-started from the soft solution → fence LG (hard containment) → fast evaluator. Two-stage first; continuous soft-membership multi-electrostatic is an upgrade path only |
| Region representation | coarse-grid bin union on the 512 lattice (RegionSet multi-rect); producer extraction 2048² → 64² majority vote → SA on 64²; arm (e) uses faithful 32²; rect_max = 8 per region |
| Terms after freeze | IO off, FT off, capacity on, pseudo-FT on |
| Freeze mechanism | hard DREAMPlace fence (no MORPH guide regions; DREAMPlace 4.3.1 has no guide-region op); fallback if fence GP too expensive: per-cell quadratic spring to assigned rect on the flat single-field GP |
| Straddling | no overlap/straddle term; fence LG guarantees containment; soft-assign anchor = cell centre (flag --node-anchor {lower_left,center,pin}, default center; pin only in IoTermRef); evaluator stays pin-based; diagnostics straddle_cells / straddle_area_fraction / straddle_pin_split_nets / io_delta_at_freeze / fence_compliance; identity io(final) = io(soft) + io_delta_at_freeze + lg_loss; anchor-comparison experiment on mempool_tile (three anchors vs post-fence-LG truth) |
| IO capacity | per adjacent-pair boundary SEGMENT; feed-through nets consume capacity on entry and exit segments; capacity from a one-time pre-GP OpenROAD extraction (fallback ρ·ℓ from tech LEF); penalty L_cap = Σ_s (2[d_s]_+³ + [d_s]_+²), d_s=(D_s−C_s)/C_s, activated at overflow ≤0.30, stays on after freeze; evaluator hard check with per-segment utilisation |
| Pseudo-point FT | one movable pseudo point per 2-pin cross-partition net, injected as filler-class node at the tail after placedb.initialize(), non-fence bucket, tiny area (1 site × 1 row); cost ω(r(p))·[WA(u,v,p)−WA(u,v)] with ω_k=(util_k/target_density)² frozen per home_period; hinge anti-collapse at ρ_min=0.1‖u−v‖₁; GP-internal only; n_pseudo_max=2e6; multi-pin deferred; fallback softmin over fixed candidates |
| Normalisation | one module src/ioplace/norm.py (TermNormalizer) replacing the three legacy paths; policy A grandplan (Eq.3 ratio, wt 0.05 → 1.0 step +0.05 every 100 iters) and policy B adaptive (force-share target with DREAMPlace-4.0 momentum 0.75/0.25), both as ablations; norm order L1 default with L2 switch; Lipschitz cap kept with per-term curvature; one obj_version bump per transaction; norm_trace.jsonl |
| Routing | OpenROAD global_route once at the very end of every arm; all GR-in-loop code behind IOPLACE_ENABLE_GR_IN_LOOP=1, unmaintained; .worktrees/round-feedback archived by tag, not merged |
| Arms | (a) grid 4×4 + region-centre init; (b) producer geometry + centre init; (c) grid + flat seed; ours = producer + flat seed; (e) faithful GrandPlan (fixed Mt-KaHyPar membership, grouping, fence GP, WL/density/grouping only); (f) fence-from-start (run_placement_two_stage.py with all terms minus the soft phase). All on DREAMPlace, same LG. K=16 only; slicing and K=32 retired. Every arm reports region area balance |
| Benchmarks | Tier 1 MemPool tile/group/cluster (ISPD2025 NanGate45); Tier 2 ISPD2015 superblue11_a/12/16_a + ISPD2024/2025 NanGate45 bp_quad and NVDLA; Tier 3 (later) ASAP7: NV_NVDLA_partition_{a,c,m,p} from NVlabs/CircuitOps (fallback VLSIDA/HighTide), Ariane/bp_quad/nvdla(c) from TILOS MacroPlacement. Bookshelf ISPD2005 → unit tests only. Develop on mempool_group, validate on mempool_cluster, synthetic 3×3 27.7M once |
| Subproject order | P-A literature (done) → P-H normalisation → P-B+P-C (main flow + producer, bound; acceptance = the 2×2 on mempool_group) → P-F straddling → P-D capacity → P-E pseudo-FT → P-G final GRT matrix |
| Schedule | no deadline pressure; DAC 2027 (Nov 2026) is one candidate venue |
| Claim | IO-aware simultaneous region production + fence placement, under boundary IO capacity, beats grid (a) and pure GrandPlan (e) on IO / FT / final GRT overflow; never claim "first feed-through-aware floorplanner" (Flora exists) |

## 1. Architecture and data contracts

Coordinate contract: **native post-read PlaceDB units** — the die box right
after `placedb.read()`, before `placedb.initialize()`
(`src/ioplace/drivers/run_placement_two_stage.py:181` already uses it, because
`PlaceDB.scale()` inside `initialize()` rescales `regions`,
`flat_region_boxes` and `node_x` together, `PlaceDB.py:151-196`). Evaluator
artefacts stay in scaled evaluator units and record `shift_factor`/`scale_factor`.

Two drivers, separate processes, communicating only through files.

**Driver 1 — region producer**, `src/ioplace/drivers/run_region_producer.py`
(new). In: DREAMPlace config, `K`, membership source. Out: `regions.json`,
`seed.npz`, `membership.npz`, `producer.json`.

**Driver 2 — main flow**, `src/ioplace/drivers/run_main_flow.py` (new, forked
from `run_placement_io.py`, which stays as the single-phase legacy driver). In:
config, `regions.json`, optional `seed.npz` and `capacity.npz`. Out: `soft.npz`,
`freeze.json`, `placement.npz`, `evaluation.npz`, `norm_trace.jsonl`,
`result.json`.

Artefacts, one directory per arm (`runs/<case>/<arm>/`):

| File | Payload | Producer / reader |
|---|---|---|
| `regions.json` | `RegionSet` (die, lattice, per-region rect list) | `src/ioplace/regions.py:44` `to_json`/`from_json`; the format `run_route_gp.py` already reads |
| `seed.npz` | `node_x`,`node_y` over `num_physical`, native units, `die`, `shift_factor`, `scale_factor`, `placedb_sha256` | producer → warm start |
| `membership.npz` | `part` int32 per movable node, `{source,k,seed,epsilon}` | producer → arm (e) fences, soft-phase prior |
| `capacity.npz` | segment table (§5), `capacity`, `capacity_source`, OpenROAD receipt hash | extractor → GP + evaluator |
| `evaluation.npz` | existing `save_evaluation` schema (`export/evaluation.py:66`) + per-segment arrays | evaluator → reports |
| `norm_trace.jsonl` | one row per normalisation probe (§4) | main flow |

**Reused unchanged:** `regions.py`, `region_grid.py`, `region_graph.py`,
`fence_inject.py`, `ops/soft_assign.py`, `ops/io_term.py`, `ops/ft_term.py`,
`evaluator_ref.py`/`evaluator_gpu.py`, `export/evaluation.py`,
`partition/mtkahypar_runner.py`, `dp_hook.py`, and both DREAMPlace patches.
`m2-extra-obj-terms.patch` already adds extra terms *after* the fence branch of
`PlaceObj.obj_fn` (verified, `PlaceObj.py:298-328`), so **no new DREAMPlace
patch is needed** for any v2 term; `iteration-callback.patch` supplies the
hull-rebuild and probe hook.

**Retired:** `make_slicing_regions` (`regions.py:75`) and `K=32`;
`src/scripts/run_route_gp.py`, `ops/routing_gp_controller.py`, `ops/route_gp.py`,
`ops/joint_route_feedback.py`, `ops/route_feedback.py` go behind
`IOPLACE_ENABLE_GR_IN_LOOP=1`, unmaintained; `.worktrees/round-feedback` is
archived by tag, not merged. `run_placement_two_stage.py` survives as arm (f).

---

## 2. Region producer (P-C)

Membership prior → flat GP with grouping → flat LG → density-argmax extraction
→ open/close → largest CC → SA → `RegionSet`.

**Prior.** `partition_netlist` (`partition/mtkahypar_runner.py:14`), `K=16`,
`epsilon=0.03`, or RTL hierarchy prefixes; substitutes for GrandPlan's given RTL
partitions and runs before the flat GP. No block→region matching
(`run_placement_two_stage.py:113-172` unused): hulls give geometry, not a
permutation.

**Grouping loss.** Attached via `dp_hook.attach_terms` — no patch. GrandPlan
Eq.1/2: quadratic springs to a frozen anchor; pull when outside own hull, push
when inside a foreign hull; `α_pull=α_push=1`.

**Cost control at 3M–11M cells** (the paper's silent hotspot is `Σ_{s≠k}`
point-in-polygon over all cells). Per rebuild: (i) Algorithm-1 candidate
reduction on GPU — `m=16` directions, quantile `q=0.90`, band `α=0.25`,
`K_dir=64`, so quickhull sees ≤1024 points/region; (ii) **rasterise** each hull
onto the `512²` lattice and precompute per lattice cell per region the anchor
offset (`K×512²×2` fp16 = 16 MB at K=16). The per-iteration loss is then two
table lookups and a spring per cell, O(N), independent of hull complexity.
Rebuild every `T_hull=50` iterations, anchors frozen in between (the paper's own
semantics). Macro enrichment: pseudo points at mean std-cell pitch, ≤64/macro.
Area cap `A_max=EA_k`, centroid shrink by bisection to 1e-3 relative area.

**SA.** Energies exactly Eqs.4-8 (`θ=0.05`, `50[d]³+25[d]²`; `0.1(C_ij−2)²`;
`5((0.8−ρ_fill)/0.8)²_+`; `D_diff/N_bins`), each min-max normalised over the
first 200 samples before `β`. Defaults for the unspecified values:
`β=(1.0,0.3,0.5,0.2)`; `T_0` = mean |ΔE| of 200 probe moves; geometric cooling
0.92; 50 moves/level; 150 levels; stop after 3 levels with no accept. Moves:
area-balancing and corner-filling windows of 2–9 bins, equal probability while
any area violation exists, corner-filling only afterwards; reject any move that
fragments a region.

**Extraction resolution.** `2048²` density bins → majority vote to **`64²`** →
SA on `64²` → rects on the `512` lattice (64 | 512, so `validate()` passes);
`64²` gives 256 bins/region at K=16 instead of 64, i.e. usable rectilinear
detail, and SA over 4096 bins is still negligible. Rejected: faithful `32²`
extraction, used only by arm (e).

**Rect cap (hard requirement, §10 risk 1).** Decompose each region into maximal
horizontal strips, merge, and enforce `rect_max=8` by filling the smallest
notches; re-run `validate()`.

**Runtime budget.** Anchors: bigblue4 2.18M flat GP+LG 545 s
(`results/m0/bigblue4_flat_k16_grid.json`); cluster 11.3M flat GP+LG 2989 s
(`results/m4/profile/mempool_cluster__k16__grid__flat.json`). Producer ≈ flat
GP+LG +<3% + SA <60 s CPU. Target: group ≤25 min, cluster ≤60 min.

---

## 3. Main flow (P-B)

Four phases, two DREAMPlace instances. The split is forced: fence data may only
be injected between `read()` and `initialize()` (`fence_inject.py:9-16`), so the
fence GP cannot reuse phase 1's PlaceDB; each phase becomes independently
restartable and cacheable.

**Phase 1 — soft GP.** Existing machinery unchanged: `IoTerm`/`FtTerm` over
`softmax(−SDF/τ)`, τ/ρ from `schedules.py`, λ from §4. Warm start: write
`seed.npz` into `placedb.node_x/node_y` right after `read()` and set
`random_center_init_flag=0`, so `BasicPlace.py:269-288` consumes it and
`initialize()` scales it. Centre-init arms leave the flag at 1.

**Phase 2 — freeze.** Trigger when all three hold: `overflow ≤ 0.15`;
`τ_rel ≤ 0.05` (the FT full-ramp point, `schedules.py:124`); membership churn —
the fraction of movable cells whose argmax region changed over the last 50
iterations, free from `softmax_stats`' `am` (`ops/soft_assign.py:40-58`) —
`≤0.5%`. Membership = argmax region of the cell **centre**. Emit `freeze.json`
(iteration, overflow, τ, churn, per-region cell count and area vs region area)
and `soft.npz`; report `io_delta_at_freeze`.

**Phase 3 — fence GP.** Rebuild PlaceDB, `inject_fence_regions`
(`fence_inject.py:6`), keep the escape-cell workaround
(`run_placement_two_stage.py:192-233`), warm start by the same `node_x` write.
Density weight is *not* reset by hand — `PlaceObj.initialize_density_weight`
(`PlaceObj.py:776-835`) recomputes the per-region vector from the grad-norm
ratio *at the seed*. That is a hazard: at a near-legal seed the density gradient
is small, so `params.density_weight·‖∇WL‖₁/‖∇D‖₁` can over-weight density and
blow the seed apart. Mandate: log phase-1's final `density_weight`, clamp
phase-3's initial value to `[0.25×,4×]` of it, log whether the clamp bound.
Verify with the open question (iii) experiment before the first group run. The
`obj_version`/`refresh_nesterov_secant` discipline (`dp_hook.py:36-59`) carries
over verbatim.

**Terms after the freeze.** IO and FT **off** (membership is fixed and cells
confined, so `q_{e,k}` carries no usable gradient; the residual is a
boundary-margin push, measured neutral-to-worse); capacity **on** (per-segment
demand still depends on where along a boundary cells sit, §5); pseudo-FT **on**
(pseudo nodes are unconfined, §6). Rejected: all terms on, which adds an extra
probe per event and a term fighting the fence density field.

**Freeze mechanism.** The hard DREAMPlace fence, with a surrogate in reserve if
fence GP is too expensive (6.9× runtime, 4× memory, §10 risk 2): a per-cell
quadratic spring to its assigned region on the flat single-field GP — the
grouping Pull term with the hull replaced by the assigned rect, available today
through `attach_terms`. Rejected: MORPH guide regions — DREAMPlace 4.3.1 has
**no** guide-region op (only `dreamplace/ops/fence_region`), so MORPH would
mean new CUDA work.

**Phase 4 — fence LG** = `build_multi_fence_region_legalization`, selected
automatically once `placedb.regions` is non-empty (`BasicPlace.py:412-417`),
then the GPU evaluator, then stop.

### 3b. Continuous upgrade path (≤120 words, not designed here)

One run in which every cell
carries a soft membership vector and region `k`'s electrostatic field
accumulates `p_{i,k}·area_i` instead of a 0/1 assignment, so membership and
position co-optimise and the fence emerges by annealing τ. Needs weighted
per-field density accumulation in the CUDA op, a per-field overflow definition
under fractional area, a per-region Lagrangian that stays stable while `p`
moves, and an LG fed the final argmax. It subsumes the freeze event and removes
the two-instance split, at the price of K density fields for the whole run.
Revisit only after the two-stage flow shows a measured win.

---

## 4. Normalisation module (P-H)

New `src/ioplace/norm.py`, one class `TermNormalizer`, replacing three ad-hoc
paths: λ_IO EMA + Lipschitz cap (`schedules.py:94-104,159-176`), κ_FT force
share (`schedules.py:66-81,178-237`), and the one-shot route λ
(`ops/routing_gp_controller.py:64-78`).

Interface:

- `register(name, term, curvature)` for `io`, `ft`, `cap`, `pseudo_ft`,
  `group`; `term` exposes `value(pos, ctx)` and `grad_l1(pos)`.
- `probe(iteration, pos, obj_and_grad)` every `probe_every=50` iterations: one
  WL-only backward plus one isolated backward per registered term — the pattern
  already in `run_placement_io.py:465-481`. Five terms ≈ +12% backward cost at
  N=50; `probe_terms` can subset it.
- `weights(iteration, overflow, tau, gamma) -> {name: λ}`.
- `transaction(...)`: computes all λ, bumps `obj_version` exactly once, returns
  a flag telling the driver to call `refresh_nesterov_secant` then
  `mark_refreshed()` — `apply_ft_transaction`'s seven-step atomic discipline
  generalised to N terms.

Policy A `grandplan` (Eq.3): `λ_t = wt_t·‖∇WL‖_p/‖∇T_t‖_p`, `wt_t` starting at
0.05, stepped +0.05 every `ramp_period=100` iterations to `wt_max=1.0`, gated on
the activation overflow threshold. Policy B `adaptive`: target force shares
`f_t`; `λ_t ← λ_t·(f_t·G/(λ_t‖∇T_t‖))^0.5` with DREAMPlace-4.0-style momentum
`λ_t ← 0.75λ_t^prev + 0.25λ_t^new`. Both are ablation arms and share the EMA
(`ema=0.5`), the activation ramps and the cap.

**Lipschitz cap survives in form:** `lipschitz_cap(tau, gamma, c_lip, Cmax)`
(`schedules.py:94`) with `Cmax = 1 + Σ_t κ_t(curv_t−1)_+`, generalising
`derive_cmax` (`schedules.py:84`). Declared curvatures: IO 1, FT `ecc_max`,
capacity `max_s pen''`, pseudo-FT 1. λ's are clipped so their *sum* respects the
cap; which term bound the cap is logged.

**Norm order.** GrandPlan Eq.3 uses L2. Default `norm_p=1` with L2 as a switch:
every calibrated constant (`ratio_ema`, `κ_ft`, `c_lip`) was fitted under L1,
and the ratio is self-consistent within a policy.

Logged per probe to `norm_trace.jsonl`: iteration, overflow, τ, per-term
`grad_l1`, `ratio_inst`, `ratio_ema`, `wt`/`f`, `λ`, realised share
`λ‖∇T‖/Σ`, `cap_binding`, `cancellation_ratio`, `obj_version`,
`refreshed_version`.

---

## 5. Capacity term (P-D)

**Segment enumeration** (new `src/ioplace/region_segments.py`). On
`RegionGrid.grid` (512², `region_grid.py:5-24`): take `grid[:,:-1]!=grid[:,1:]`,
key each boundary unit edge by its unordered pair, run-length-encode along the
perpendicular index per column; each maximal run is a segment. Same for
horizontal. O(L²)≈2.6e5, pure numpy. Outputs the segment table
(orientation, line index, `lo`/`hi`, pair, physical length), a **unit-edge →
segment-id** raster (2×512×511 int32 ≈ 2 MB), and per-row/column CSR lists of
boundary positions with segment ids. `region_graph`'s `ell[a,b]`
(`region_graph.py:62-71`) is the sum of a pair's segment lengths and remains the
pair-level aggregate.

**Capacity extraction, once, pre-GP.** Reuse
`route_eval/or_scripts/dump_online_route.py:90-114` and
`route_eval/online_openroad.py:44-159` verbatim on the **input DEF** with
minimal congestion iterations and `-allow_congestion`; keep only
`horizontal_capacity`/`vertical_capacity` and the GCell edges; discard `usage`
(it reflects the input placement). Offline conversion: for a vertical segment at
`x=X` over `[y0,y1]`, `C_seg = Σ_rows overlap(row,[y0,y1])/row_h ·
hcap[row][col(X)]` — horizontal-preferred-direction layers only, blockages
already folded in by OpenDB. Fallback without layer data: `C_seg = ρ·ℓ`,
`ρ = Σ_{layers ⊥ seg} 1/pitch` from the tech LEF. Record `capacity_source`, the
receipt SHA-256, and `capacity_semantics = "usable tracks crossing the segment;
one net crossing consumes one track"`.

**Unit compliance** (`.../2026-09-15-round-feedback-design.md:169-196`): the
GCell grid is touched once, offline, by the extractor; its output is a scalar
keyed by segment id. Inside GP nothing indexes one grid with the other's ids and
no IO-lattice load is divided by a GCell capacity. Zero-capacity segments stay
hard-blocked with the same finite penalty; no epsilon substitution.

**Differentiable demand.** Every `home_period` iterations the evaluator emits,
per net, the (pair, segment) items it actually crossed, capped at `m_pairs=4`,
`m_seg=2` — the frozen-discrete/continuous split already used for `home`
(`ops/ft_term.py:40`, `run_placement_io.py:452-457`). Between refreshes,
`D_s = Σ_{e∈cand(s)} w_e·q_{e,a(s)}·q_{e,b(s)}·α_{e,s}(c_e)`,
`α_{e,s} = softmax_{s'∈cand(e,pair)}(−d₁(c_e,s')/τ_b)`, `c_e` the net's soft pin
centroid (one `index_add`), `τ_b = 2·cell_w`. Feed-through attribution is
automatic: a traversing net crossed entry and exit segments, so both are in its
candidate list and both take demand. `q_aq_b` is differentiated exactly, not
linearised — both cofactors are cached per candidate during the forward chunk
loop into an `(E_cand,m)` array, so no `(E,K)` tensor appears.

**Penalty.** `d_s=(D_s−C_s)/C_s`, `L_cap = Σ_s (2[d_s]_+³ + [d_s]_+²)` —
GrandPlan Eq.5's shape, `C¹` at the knee, steep beyond. Rejected: phase-1 §5.4's
`softplus((D−C)/C)` (`.../2026-07-30-io-aware-placer-phase1-design.md:86-90`),
because `softplus(−1)=0.31` exerts force at half capacity, distorting WL where
nothing is violated. λ_cap from §4, activated at `overflow ≤ 0.30`, stays on
after the freeze.

**Evaluator hard check.** Each unit crossing maps to a segment id through the
raster; the per-row CSR turns an L-shape leg into a contiguous slice, so work
equals total crossings, and the existing `Ph`/`Pv` prefix-sum counts
(`evaluator_gpu.py:182-190`) become a free assertion on slice lengths. New
fields: `segment_demand`, `segment_capacity`, `segment_util`,
`num_over_capacity`, `max_util`, `p99_util`, stored beside
`boundary_pairs`/`boundary_demand`/`boundary_length`
(`export/evaluation.py:90-91`).

---

## 6. Pseudo-point FT term (P-E)

**Injection.** Pseudo points are appended as **extra filler-class nodes at the
tail of the node arrays, between `placedb.initialize(params)` and the
`NonLinearPlace(...)` construction**. Fillers already have every needed
property: no pins (invisible to WA wirelength and HPWL), density participation,
presence in `pos` (`BasicPlace.py:375`), and exclusion from the reported
placement and the evaluator (which reads only `num_physical`). Tail injection
needs **no index remapping** — every pin index stays `< num_physical`.
Increment `num_nodes`/`num_filler_nodes`, extend `filler_start_map`, and put
them in the implicit non-fence bucket (`region_id = K`) in fence mode so they
stay unconfined.

**Rejected alternatives.** (a) Appending *movable* nodes before
`initialize()` requires `pin2node_map[pin2node>=num_movable] += n_pseudo` plus
edits to `node_names`, `node_orient`, `node2orig_node_map`, `node2pin_map`,
`flat_node2pin_*`, `node2fence_region_map`, `movable_macro_mask` — mechanical,
large surface. (b) Splitting the net into two 2-pin edges breaks exact net
identity (round-feedback spec) and the GR net audit. (c) Fallback if the filler
tail destabilises: drop the DOF, use a softmin envelope over 8–16 fixed
candidates per net — same cost, no injection, no true DOF.

**Cost.** For a 2-pin cross-partition net with pins `u,v` and pseudo point `p`:
`L_ft = Σ_e w_e·ω(r(p_e))·[WA_γ(u,v,p_e) − WA_γ(u,v)] + reg`, where `ω_k =
(util_k/target_density)²` is the traversed region's utilisation weight from its
density-field overflow, refreshed on the `home_period` cadence and frozen inside
the gradient (the `home` discipline). The bracket is ≥0 and measures the detour
of routing through `p`; minimising `ω·detour` pushes `p` out of congested
regions. Gradient w.r.t. `p` is the existing WA derivative times frozen `ω`;
w.r.t. `u,v` the usual WA term. `r(p)` is read as a smoothed
`softmax(−SDF/τ_p)`, so `ω` is a soft mixture with no discontinuity at a
boundary.

**Anti-collapse.** `reg = μ·Σ_e[(ρ_min−‖p−u‖₁)_+² + (ρ_min−‖p−v‖₁)_+²]`,
`ρ_min = 0.1‖u−v‖₁`, μ from §4. A pure detour cost is minimised at `p=u` or
`p=v`, so a hinge floor on the pin distances is the minimal fix; a midpoint
spring would bias the solution instead.

**Density / filler / LG.** Area = one site × one row height (the "tiny area"
decision), added to the filler budget so movable+filler area is unchanged.
Fillers are excluded from greedy/abacus LG in the flat flow, but
`build_multi_fence_region_legalization` *does* hand a region's fillers to its
legaliser (`BasicPlace.py:817-859`) — hence the non-fence bucket, and post-LG
pseudo coordinates are never read. "Removed before LG" = the FT term is detached
at the `legalize_op` boundary (existing wrapper, `run_placement_io.py:312-323`),
pseudo coordinates dumped to `pseudo.npz` before LG, nothing downstream indexes
beyond `num_physical`.

**Memory at 30M nets.** ≈48 B/node (2 floats in `pos` plus ~5 Nesterov/
preconditioner copies) ⇒ ≈1.4 GB at 30M, plus ≈0.6 GB sizes/areas — feasible,
but default `n_pseudo_max = 2e6`, the top nets by measured per-net FT. Multi-pin
nets (FLUTE set, `route_eval/topology.py`) deferred.

---

## 7. Straddling (P-F)

**The inconsistency.** The differentiable term's soft assignment uses the node
*lower-left* corner (`ops/io_term.py:101-110`, `ops/soft_assign.py:19-31`), the
evaluator counts at *pin* positions (`evaluator_ref.py:97,113`;
`evaluator_gpu.py:303-306`), and fence assignment owns the *whole* cell
(`fence_inject.py:28-34`). A cell spanning a boundary is judged differently by
all three, so GP optimises a quantity the evaluator does not score.

**The fix.** Anchor the soft assignment at the **cell centre**
(`x+0.5·node_size_x`, `y+0.5·node_size_y`) behind
`--node-anchor {lower_left,center,pin}`, default `center`. One-line change at
`region_sdf_l1`'s caller; keeps the `(N,K)` shape and the no-double-counting
property that motivated the original decision
(`docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md:105`); removes
a systematic half-cell bias; makes the freeze consistent with whole-cell fence
ownership. The `pin` arm is implemented only in `IoTermRef`, so the bias is
quantified once at small scale rather than paid for at 11M cells. Rejected:
moving the production term to pins, which reintroduces `(P,K)` cost and
multi-pin double counting. The evaluator stays on pins; the residual gap
becomes a *measured* bias, not a silent one.

**Why the cell centre.** The final metric is post-fence-LG IO, where pin-count
equals cell-count because no cell straddles; the GP surrogate must therefore
predict post-LG cell ownership. The centre is its best predictor and coincides
with the freeze rule and fence ownership. Pin anchoring is exact only for a
pre-LG straddling state that v2 eliminates and lets one cell be fractionally
present in two regions, which no legal outcome realises. Lower-left is a
biased centre.

**Verification experiment.** On `mempool_tile`, from the same soft solution,
compute the GP surrogate IO with all three anchors and compare against the
evaluator after fence LG; expect centre closest. If pin is closer, LG
displacement exceeds half a cell and LG must be examined.

**No overlap/straddle penalty**, per decision. Containment comes from the fence
LG; straddling is instrumented.

**Diagnostics to add** (evaluator side, into `evaluation.npz` and
`result.json`): (1) `straddle_cells` — movable cells whose box
`[x,x+w]×[y,y+h]` meets >1 region, from the four corners' region ids via
`RegionGrid.region_of_points`; (2) `straddle_area_fraction` — out-of-owner area
over total movable area; (3) `straddle_pin_split_nets` — nets whose crossing
count would drop if every straddling cell's pins were re-attributed to the
cell's owner, i.e. the direct attribution of `lg_loss`; (4)
`io_delta_at_freeze` (§3) alongside the existing `lg_loss`
(`run_placement_io.py:691-692`); (5) `fence_compliance` after fence LG, reusing
`run_placement_two_stage.py:252-255`.

Together these close an accounting identity: `io(final) = io(soft, last GP) +
io_delta_at_freeze + lg_loss`, with the straddle metrics explaining the last two
terms.

---

## 8. Final GRT protocol and experiment matrix (P-G)

**Per-arm recipe**, identical for every arm, one GR call at the very end: final
`placement.npz` → `route_eval/placement_openroad.py:45-127` `legalize_export`
for DEF row repair (mandatory: it changed 648,869 locations on group,
`docs/results/2026-09-15-route-gp-completion-audit.md:110-121`) →
`online_openroad.run_openroad` with `set_routing_layers metal2-metal10`,
`-congestion_iterations 50`, `-allow_congestion`, 4 GRT threads,
`grt::write_segments` → `route_eval/common_grt.py:61-172` decode → fast
evaluator on the same row-repaired, oriented coordinates → `result.json` with
per-file SHA-256.

**Metrics.** Primary: **actual per-segment crossings vs capacity** —
`num_over_capacity`, `max_util`, `p99_util`, total overflow tracks — from routed
segments mapped through §5's unit-edge raster. Secondary: native total overflow
(`online_openroad.parse_native_congestion`), routed WL, via count, HPWL(GP),
HPWL(LG), `io_count`/`ft_count`/`hard_lambda_sum`/`io_rg`/`ft_rg`, the §7
straddle set, per-phase runtime and peak GPU memory, and region area balance
per arm (max/min region utilisation and cell-count deviation) — required
because arms (e)/(f) carry the partitioner's ε=0.03 balance constraint while
soft-assign arms do not.

**Runtime expectations**, measured, not estimated: tile routed all signals at 5
congestion iterations, metal2–metal10, zero overflow, in **221.9 s**, while
unbounded congestion removal cost **~9–10 h** on tile/group before the protocol
change (`docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43`).
Hence bounded iterations plus `-allow_congestion` are mandatory; GR budget ≈4
min (tile) and 1–3 h (group) per arm, once. Placement side: group flat GP+LG
660–2000 s, cluster 2989 s flat / 6809 s with the IO term; fence GP measured
6.9× flat at bigblue4 (§10).

**Matrix.** Develop on `mempool_group`, validate on `mempool_cluster`, one
synthetic 3×3 27.7M run. `K=16` only.

| Arm | Geometry | Init | Terms |
|---|---|---|---|
| (a) | grid 4×4 | region centres | main flow |
| (b) | producer | region centres | main flow |
| (c) | grid 4×4 | flat seed | main flow |
| ours | producer | flat seed | main flow |
| (e) | producer | flat seed | faithful GrandPlan: fixed Mt-KaHyPar membership, grouping, fence GP, WL/density/grouping only |
| (f) | producer | flat seed | fence from start (`run_placement_two_stage.py`), all terms minus the soft-assign IO phase |

(a)–(c) plus ours is the Table-3 2×2. One table per case, normalised to
`ours = 1.000`, plus §4's policy-A-vs-B ablation and capacity / pseudo-FT
on-off as separate rows.

---

## 9. Testing and verification

Protocol: `source src/scripts/env.sh`, then `"$IOPLACE_PYTHON" -m pytest`;
`-m "not slow"` while iterating, full suite before declaring done.

**Unit tests (new).** `tests/test_region_segments.py`: enumeration on
hand-built grids (grid K=4 → 4 segments; an L-shaped region → one pair split
into 2 disjoint segments; a 1-bin notch), unit-edge raster round-trip vs brute
force, and `Σ_s seg_len == ell[a,b]` for every pair (`region_graph.py:62-71`).
`tests/test_norm.py`: policy-A ramp, policy-B momentum, cap binding, exactly one
`obj_version` bump per transaction, refresh required before the next step
(reuse `dp_hook.install_version_invariant`). `tests/test_capacity_term.py`:
penalty exactly 0 below capacity, `C¹` at the knee by finite difference,
gradient vs autograd on a 200-cell toy, exact `q_aq_b` product gradient.
`tests/test_pseudo_ft.py`: collapse regularisation activates, `ω` stays frozen,
filler-tail injection leaves `pin2node` and HPWL bit-identical, pseudo nodes
absent from `evaluation.npz`. `tests/test_region_producer.py`: SA energies vs
the closed forms of Eqs.5-8, moves never fragment a region, output passes
`validate()` and `rect_max`, determinism under a fixed seed.
`tests/test_freeze.py`: churn criterion, `freeze.json` schema, warm-start
write-back equals the seed after `initialize()` scaling.

**Parity.** Extend `tests/test_evaluator_gpu.py`'s contract (integer fields
bit-exact, `tree_wl`/`hpwl` rel ≤1e-12) to `segment_demand`, `segment_util`,
`straddle_cells`, `straddle_pin_split_nets`: **bit-exact** ref-vs-GPU across
batch sizes and chunk budgets, on adaptec1 and bigblue4 K=16 grid plus one
multi-rect producer geometry. Add a `slow` regression pinning per-segment demand
for one frozen placement.

**End-to-end small case.** GCD (`results/route_feedback_20260914/gcd.json`) and
`mempool_tile_wrap`: producer → main flow → fence LG → evaluator → one GRT,
asserting §7's accounting identity, a non-empty `capacity.npz`, and a
reproducible `num_over_capacity` across two same-seed runs.

**"Done" per subproject.** H: both policies implemented, `norm_trace.jsonl`
emitted, the three legacy paths deleted or reduced to adapters, and a
group-scale run whose λ is within 2× of the retired path's λ at matched
iterations. B+C: the 2×2 on `mempool_group` complete, all four arms finishing
GP+LG+evaluator, with a shape-vs-seed decomposition reported (acceptance is
completion, not a quality threshold). F: five diagnostics reported and the
identity closing within ±1 crossing. D: per-segment utilisation reported from
both the GP surrogate and the router, with their rank correlation stated. E: FT
strictly reduced on `mempool_group` at HPWL ≤1.01× the no-pseudo arm, or an
honest negative result. G: full matrix tabled with hashes.

---

## 10. Risks, open questions, Flora differentiation

**Risks, by expected damage.**

1. **`region_sdf_l1` explodes with rectilinear regions.**
   `ops/soft_assign.py:19-31` builds `(N,r)` temporaries with `r` = rects per
   k-chunk. Today `r=K`; at 8–64 rects/region cost and memory scale with it
   (11M × 1024 fp32 ≈ 45 GB — infeasible). Mitigations in order: `rect_max=8`
   (§2); chunk over *rects* not regions; else replace the per-rect min with a
   per-region `512²` distance transform read by bilinear interpolation
   (differentiable, 8 MB at K=16, O(1) per cell-region). Decide before P-C
   lands — it sets the producer's rect budget.
2. **Fence GP cost.** bigblue4 2.18M K=16: flat 545 s / 1.67 GB vs two-stage
   fence 3738 s / 6.69 GB (`results/m0/bigblue4_{flat,two_stage}_k16_grid.json`)
   — 6.9× runtime, 4× memory. Extrapolated to cluster (flat 2989 s / 11.1 GB):
   ≈5.7 h, ≈45 GB, on a host where all four H100 NVLs are at 100% utilisation
   with 50–70 GB resident. Reserve a GPU; keep §3's soft guide as fallback.
3. **Fence flows can be worse.** Same records: two-stage `io_count` 178,594 vs
   flat 100,108. Arms (e)/(f) may lose badly — say so up front. This gap is
   partly structural (fixed membership plus balance constraint), so it is
   reported, not tuned away.
4. **The shape axis may be worthless.** GrandPlan Table 3: shapes alone 1.098 vs
   centre-init 1.099, seeding 1.048. If reproduced, the headline must be the
   capacity and pseudo-FT terms, not the geometry.
5. **Capacity semantics.** "One crossing = one track" ignores multi-wire nets
   and vias; OpenDB `getCapacity` is uint8-clamped (tile 7,871,705 vs native
   7,872,067, `docs/results/2026-09-15-ggr-trial.md:39-40`);
   `benchmarks/ispd25/visible/*.cap` is blind data mislabelled visible
   (`docs/results/2026-09-15-benchmark-router-diagnosis.md:58-69`) — not a
   capacity source.
6. **K=16 at 64² bins** leaves ~256 bins/region; density-argmax + largest-CC
   extraction is unvalidated at that ratio (GrandPlan ran K≤8).

**Open questions, each with its deciding experiment.** (i) Does `D_s`
rank-correlate with routed per-segment crossings? — Spearman on group from one
GRT; below 0.5, fall back to pair-level `D_ab`/`C_ab`. (ii) Does the pseudo
point move, or does frozen `ω` starve it? — displacement percentiles per 100
iterations on tile. (iii) Does phase-3 density-weight re-initialisation destroy
the seed? — HPWL at fence-GP iteration 50 with and without the clamp. (iv) Is
the centre anchor enough for wide macros? — one `IoTermRef` pin-anchor run at
tile. (v) Do capacity and pseudo-FT still help once cells are confined? —
on/off ablation at group.

### 10b. Flora differentiation (≤150 words)

Flora (arXiv 2507.14914) is the nearest
published objective and does carry a common-edge pin capacity,
`FTpin(M_i,M_j)=max(0,(u·Y_ij − CE_ij)/u)` (Eq.2: `Y_ij` the pin count the
netlist demands, `u` minimum pin spacing, `CE_ij` shared-edge length), plus
`FTmod`, the module count inside a net's bounding box. Four differences to
state. (1) Granularity: module-pair level on MCNC/GSRC, 10–300 modules with
*synthesised* internals, versus our cell-level 3M–30M instances on real
NanGate45 LEF/DEF. (2) Capacity source: their geometric proxy `CE_ij/u` with
`Y_ij` fixed by the netlist, versus our per-**segment** capacity from OpenROAD
per-layer GCell tracks with blockages and a placement-dependent demand. (3)
Optimisation: Flora is discrete throughout (wiremask/position-mask SA, greedy
whitespace reallocation, tree-search macro packing); ours is a differentiable
penalty inside GPU global placement with gradient-norm normalisation. (4)
Direction: Flora resizes module shapes; our geometry is an input and cells move.
Claim accordingly — never "first feed-through-aware floorplanner".

---

## 11. Research artefacts

- `docs/research/2026-09-18-grandplan-digest.md` — implementation-oriented close reading of the GrandPlan (ISPD'26) paper, mapping each of its mechanisms onto what v2 already has, must invent, or is strictly ahead of.
- `docs/research/2026-09-18-literature-survey-v2.md` — ~75-query literature sweep across 11 topics (fence regions, GPU GP, routability, feed-through, IO pin assignment, rectilinear floorplanning, hierarchical placement, gradient balancing, pseudo-pins, boundary legalization, GR-as-evaluator) with a top-10 reading list.
- `docs/research/2026-09-18-benchmark-sources-nvdla-asap7.md` — fact-finding report on where `NV_NVDLA_partition_*` and the TILOS Ariane/NVDLA/BlackParrot trio can be sourced with ready-made DEF/netlist/tech LEF, for the Tier 3 ASAP7 benchmark plan.
