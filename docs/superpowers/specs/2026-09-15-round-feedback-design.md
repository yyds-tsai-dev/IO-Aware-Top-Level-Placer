# Feedback between complete GP rounds

Status: architecture decisions confirmed; detailed defaults pending written review.

## 1. Objective and scope

Run complete global-placement (GP) rounds, evaluate each result outside GP with
a FLUTE-backed `GpuEvalContext`, and use the results to guide the next round.
Keep provisional search state separate from the most recent placement accepted
by global routing (GR). Use detailed routing (DR) only for representative final
cases after GR validation.

This design supersedes the iteration-level router-feedback scheduling in
`2026-09-14-paper-gp-gradient-design.md` and the local-swap outer search in
`2026-09-14-flute-joint-feedback-design.md` for the new round-based driver.
Their numerical correctness, shared-route accounting, placement identity, and
honest reporting requirements still apply. Existing experimental entry points
and historical results remain identifiable by algorithm/backend version.

Confirmed decisions are recorded in
`../../adr/0001-evaluate-between-gp-rounds.md`; vocabulary is in
`../../../CONTEXT.md`. No implementation or measured improvement is claimed.

## 2. Current implementation and intended boundary

Current `src/scripts/run_route_gp.py` calls the placer once. Its iteration
callback can run GR and assimilate observations. `GpuEvalContext` runs only
after GP and final routing; it does not drive another complete GP round.
`GpuEvalContext.evaluate` still builds MST edges. Existing
`topology_batch.batch_flute_trees` supplies native CPU/OpenMP FLUTE construction
for nets of degree 2 through 256.

The new controller owns a full GP invocation, evaluation, feedback publication,
GR checkpoints, and recovery. GP iterations evaluate differentiable losses
using frozen feedback coefficients; they do not call `GpuEvalContext`, GR, or
DR. A GP round is a full configured GP run to its convergence condition or
iteration cap, not a fragment of one long optimizer run.

Differentiable geometry and topology refreshes required by the existing GP
objective may continue at optimizer-step boundaries. They must not update
feedback coefficients, invoke the fast evaluator, or change a line-search
objective generation midway through a step. Existing wirelength, density,
paper-GP terms, fixed-node handling, and preconditioning remain operative.
Feedback-related global scaling is configured or calibrated before publishing
the round payload. Intrinsic GP density schedules remain separate from that
payload; they are not evaluator feedback updates.

## 3. State and round lifecycle

Maintain three distinct records:

- **Initial baseline:** immutable legal placement, initial GR metrics, baseline
  fast evaluation, routing policy, net cohort, and resource geometry.
- **Accepted state:** latest GR-accepted legal placement, orientations, GR
  measurements, placement-associated background/resource provenance, and
  accepted feedback seed. This is the recovery point.
- **Provisional state:** current candidate positions, fast evaluations, frozen
  round feedback, and progress since the last checkpoint. This is not a
  publishable accepted result.

Initialize from a supplied valid placement, legalize it if necessary, and run
baseline GR. If no usable placement is supplied, run the existing GP with
feedback disabled to obtain the baseline. Record this initialization cost
separately from candidate-round cost. A failed baseline GR prevents candidate
acceptance; do not fabricate baseline metrics.

For each candidate round:

1. Publish one validated feedback generation from the current placement.
2. Start a fresh GP optimizer warm-started from that placement. Regenerate
   optimizer caches and filler state; do not carry momentum across rounds.
3. Complete GP, then run FLUTE-backed fast evaluation on its physical positions.
4. At a non-checkpoint round, save the provisional placement and derive feedback
   for the next round.
5. At a checkpoint, legalize a candidate copy. If legalization changes physical
   placement, evaluate the legalized copy again. Run GR on that exact candidate
   and apply the acceptance predicate in Section 7.
6. On acceptance, atomically replace accepted state and clear the rejection
   streak. On rejection, restore accepted state as specified in Section 8.

Check every three candidate rounds by default. Stop at twelve candidate rounds
or two consecutive rejected checkpoints, whichever occurs first. Both limits
and the interval are configurable positive integers. A last partial interval
must be checked before its candidate can be published. A previously checked,
unchanged placement can reuse its GR result only when all identity and routing
policy keys match. Do not count reused results as another rejection.

Publish the most recent accepted placement. If no candidate was accepted,
publish the baseline and report that no measured improvement was found.

## 4. FLUTE-backed fast evaluation

Reuse `GpuEvalContext` for persistent net/region tensors and GPU aggregation.
Add an explicit FLUTE backend that emits geometric segments with net ownership
and Steiner vertices. MST pin-index pairs cannot represent this interface.
Native FLUTE construction runs on CPU; the design does not claim GPU FLUTE or
an end-to-end speedup before measurement.

Use the existing native wrapper with accuracy 3 and coordinate scale 1000 as
initial configurable defaults. Preserve exact pin identities and offsets.
Quantization, coincident pins, zero-length branches, and reconnection to original
pin coordinates follow one deterministic, tested geometry convention. Use the
existing deterministic rectilinear branch expansion consistently in reference
and GPU paths. No silent MST fallback is allowed.

Merge overlapping collinear segments within each net before measuring physical
wirelength and IO crossings. Count shared same-net resource occupancy once per
resource edge; different nets accumulate. Feed-through counts refer to distinct
traversed regions without terminals, using the same region-boundary convention
as the reference evaluator. A point touching a boundary without crossing it
must not become an extra crossing.

Preserve the current supported FLUTE cohort of degree 2..256 nets, fixed at run
initialization. Degree 0/1 nets contribute zero routed-tree cost. Higher-degree
nets are explicitly unmodeled by this fast-tree backend; report their count,
pin count, IDs, and separate region-presence lower bounds. Do not blend those
lower bounds into a total advertised as full FLUTE cost. They remain included
in full-net GR acceptance. GR coverage and fast-model coverage are separate
reported quantities.

Evaluate in bounded batches with deterministic net ordering. Do not materialize
an unbounded net-by-grid tensor. Release GP working buffers before fast
evaluation where possible. Record FLUTE construction, transfer, aggregation,
total wall time, host memory, and peak GPU memory separately.

The fast result supplies per-net IO/FT, tree wirelength, HPWL, region presence,
spatially resolved IO-boundary loads, and predicted occupancy on the GR resource
grid. Existing aggregate `boundary_pair_demand` alone is insufficient for
localized boundary penalties. New outputs include backend, cohort, geometry,
placement, and resource-grid identities.

## 5. Feedback mapping and proposed defaults

Feedback is a validated payload consumed by differentiable GP penalties, not a
raw `EvalResult` or an optimizer gradient. Publish it once per round. Include
per-net IO/FT multipliers, IO-boundary multipliers, routing-resource prices,
capacity/background provenance, source placement hash, and generation ID.
Keep fast-model adjustments separate from GR observations so one cannot
silently overwrite the other.

The following dimensionless update rules are initial engineering defaults for
review and later calibration, not empirically established optimal settings.
Let `positive(x) = max(x, 0)` and `clip(x, a, b)` limit a value to `[a, b]`.
Use damping `eta = 0.25`, gain `alpha = 1`, and multiplier cap 8 by default.

### Per-net penalties

For modeled net `e`, let `I_e` be estimated IO crossings, `F_e` estimated FT,
and `L_e = max(number_of_terminal_regions - 1, 0)` its region-presence lower
bound. Define `E_e = positive(I_e - L_e)`. This excess is a prioritization
signal, not a claim that every excess crossing can be removed.

Use immutable baseline scales `sI_e = max(E_e_baseline, 1)` and
`sF_e = max(F_e_baseline, 1)`. Targets are:

```text
target_io[e] = clip(1 + alpha * E_e / sI_e, 1, 8)
target_ft[e] = clip(1 + alpha * F_e / sF_e, 1, 8)
next_multiplier = (1 - eta) * previous_multiplier + eta * target
```

Apply these multipliers to differentiable IO/FT penalties, not merely to the
wirelength objective's net weights. The feedback payload defaults to ones
before the first baseline-derived update. FT must have an explicit per-net
differentiable contribution; adding an FT metric to logs is not integration.

### IO-boundary spatial penalties

IO region boundaries and GR routing resources have different grids and units.
Keep them separate. On each fixed IO-lattice boundary element `b`, use the
localized same-net-deduplicated crossing load `D_b` and baseline `D0_b`:

```text
relative_growth[b] = positive(D_b - D0_b) / max(D0_b, 1)
target_boundary[b] = clip(1 + alpha * relative_growth[b], 1, 8)
next_boundary = (1 - eta) * previous_boundary + eta * target_boundary
```

Apply these coefficients to the corresponding differentiable IO crossing
events while preserving the Boolean physical boundary mask. Baseline
normalization, including the zero-baseline rule, does not assert a physical
capacity for a region boundary.

### Routing-grid congestion prices

Use GR's actual, potentially nonuniform GCell resource grid and preferred-layer
aggregated track capacities. Compute FLUTE occupancy on that grid; do not divide
IO-lattice loads by GCell capacities or index one grid with the other's IDs.

For positive-capacity resource `r`, use
`u_fast[r] = positive((background[r] + predicted_demand[r]) / capacity[r] - 1)`.
Update a fast-price component by a damped target `clip(u_fast, 0, 8)`. At a GR
checkpoint, independently update a measured-price component from the GR
observation's nonnegative excess-utilization signal using the same damping and
cap. Add these price components in the differentiable resource-price term;
keep their identities distinct in logs and ablations.

Zero-capacity resources remain explicitly blocked in hard geometry checks and
retain the existing finite differentiable blocked-resource penalty. Never
replace zero capacity with an epsilon and advertise a physical utilization.

Capacity, blockage demand, and measured wire usage must remain distinguishable.
Subtract modeled foreground occupancy before constructing cached background;
never add full measured usage on top of the same modeled foreground. Background
from unmodeled nets is a cached approximation between GR checkpoints, carries
its source placement hash, and is never reported as a new measurement. Reject
invalid or saturated resource extraction rather than inventing unit capacity.

## 6. Module boundaries

- **Fast evaluator:** existing `evaluator_gpu.py` delegates FLUTE geometry to a
  focused backend adapter and performs batched GPU metric aggregation. It
  returns measurements and does not modify placement or policy.
- **Feedback policy:** a new focused module maps measurements and retained GR
  observations into bounded, immutable round payloads. It owns normalization,
  damping, and provenance validation.
- **GP objective adapter:** extend existing `FrozenJointRouteCost` and IO/FT
  terms to consume explicit payload coefficients. Preserve geometric masks,
  separate grids, fixed-node behavior, and line-search consistency. Existing
  `assimilate` is not reused unchanged as a fast-feedback setter.
- **Round controller:** a new focused module owns initialization, GP calls,
  checkpoints, acceptance, recovery, counters, and publication. It does not
  implement metric geometry or differentiable kernels.
- **Driver/reporting:** wire the controller into the active routing-GP workflow
  with explicit configuration. Write round and checkpoint manifests. Keep
  representative DR selection downstream of GR results.

Exact function signatures and file-level implementation tasks belong in the
implementation plan after this design is reviewed.

## 7. GR acceptance predicate

Let `P0` be the initial baseline, `Pa` the accepted placement, and `Pc` the legal
checkpoint candidate. Accept `Pc` only if all conditions hold:

1. Placement legality, fixed-node/orientation constraints, router completion,
   expected routed-net coverage, and metric finiteness pass.
2. Measured routed IO is strictly less than that of `Pa`.
3. Measured routed wirelength is at most `1.01 * WL(P0)`.
4. `native_congestion.total_overflow(Pc)` is no greater than that of `Pa`.

IO and native overflow are integer comparisons without tolerance. Derive
wirelength from the original integer router DBU segments and compare
`100 * WL(Pc) <= 101 * WL(P0)` to avoid a floating boundary ambiguity. Preserve
the DBU scale in the manifest. Missing or inconsistent length units invalidate
the comparison; do not add another tolerance to the user's 1% allowance.

Native total overflow is the sum of layer overflows. The current fields named
`max_horizontal_overflow` and `max_vertical_overflow` aggregate per-layer maxima;
they are not a global maximum-edge guard and do not define acceptance here.
Retain them as labeled diagnostics. Aggregated 2-D resource overflow is also
diagnostic and cannot substitute for native overflow.

Baseline, incumbent, and candidate comparisons require identical benchmark,
net identities, layer policy, router build, seed, capacity policy, and GR effort.
Use one declared GR effort for acceptance runs. The driver may retain a
separate higher-effort final diagnostic mode, but compare baseline and accepted
placement at that same effort; it cannot silently replace checkpoint metrics or
publish a result that fails the final declared gate. Fall back to the baseline
if no previously accepted placement passes that final comparison.

Missing metrics, partial routing, nonfinite outputs, or legalization failure
reject the candidate. Preserve the accepted result and record the reason.
Infrastructure failure is reported separately from measured quality regression;
after two consecutive failed checkpoints the same stop limit applies, but the
run must not be reported as a completed quality comparison.

## 8. Recovery and observations

Keep rejected GR observations as observations of their exact measured candidate.
They can update retained congestion-price information. They do not authorize
publishing the candidate or replacing incumbent-associated background with
candidate-derived background.

Restore accepted physical coordinates and orientations, discard candidate
optimizer state, restore accepted background/resource provenance, and evaluate
the restored placement. Start from the accepted feedback seed when damping the
fresh per-net/boundary update. Combine it with retained GR congestion prices in
a new generation. This avoids carrying unvalidated candidate per-net penalties
or background into the recovered placement while retaining failure information.

Write observations, checkpoints, and the accepted-state pointer atomically.
Persist placement/config/source hashes, counters, metric values, decision
reasons, source paths, elapsed times, process exit codes, and final artifact
paths. An interrupted GR result cannot be treated as accepted on resume.

## 9. GR-first validation and representative DR

Stage one runs matched feedback-enabled and feedback-disabled experiments on
the declared benchmark manifest with identical initialization, seeds, GP limits,
GR policies, and net cohorts. GCD is an integration smoke case; it does not
establish behavior or runtime at the intended multi-million-cell scale. Report
baseline and selected routed IO, wirelength, native overflow, coverage, all
rejections, GP/evaluation/GR time, memory, and source identities. Preserve
negative results and the cost of baseline initialization.

Stage two selects at most three distinct representative benchmark/seed pairs
from completed stage-one results: highest initial native overflow, median
physical-cell count, and worst selected relative IO improvement, defined as
`(IO_baseline - IO_selected) / max(IO_baseline, 1)`. Break ties by benchmark
ID then seed; select the lower median for an even-sized set and deduplicate
overlapping choices. Include a fallback/no-improvement case if present via the
worst-improvement rule. Persist the selection before launching DR.

Run matched baseline/selected DR for those representatives only, reusing a run
when both placements are identical. Report routed wirelength, IO, DRC, completion,
coverage, and runtime under the same technology/routing policy. Unsupported
technology inputs or unfinished DR remain explicit validation gaps; they do not
trigger DR on every benchmark or make GR results into DR evidence.

## 10. Required verification

- Hand-check FLUTE geometry with Steiner vertices, coincident pins, duplicate
  pins, shared/reversed branches, zero-length links, and boundary touches.
  Independently verify union wirelength, IO/FT, and resource occupancy; cover
  degree limits and declared unmodeled nets. Check CPU/GPU and batch invariance.
- Check feedback zero-baseline behavior, caps, damping, separate grid shapes,
  zero capacities, and stale/mismatched provenance. Test two placements with
  different per-net or boundary pressure producing different next-round payloads.
- Verify finite differences for the weighted differentiable terms, fixed-node
  gradients, CPU/CUDA parity, and frozen feedback throughout each GP round.
  A topology refresh must not replace the feedback generation.
- Exercise the complete round controller: checkpoints at 3/6/9/12, a partial
  final interval, strict IO ties, native overflow regression, the exact 1% bound,
  nonaccumulating wirelength allowance, two consecutive rejects, accepted-streak
  reset, routing failures, and reuse of an identical checked placement.
- Check recovery of coordinates/orientations and background provenance, fresh
  optimizer state, retained rejected-observation prices, re-evaluation of the
  incumbent, atomic publication, and resume after an incomplete checkpoint.
- Demonstrate in real GP that next-round payload changes alter the objective's
  gradient and later placement, using a feedback-disabled control. Verify the
  evaluator and routers are not called from inside GP iteration callbacks.
- Complete matched real-case GR tests before claiming measured improvement;
  verify representative DR selection and report its results separately.

## 11. Limits and review focus

The CPU FLUTE stage and transfers may dominate fast evaluation; batching is not
a measured speed claim. Fast resource prices use a 2-D layer aggregation and
cannot certify layer routing or DR feasibility. Degree-above-256 nets remain
outside fast-tree modeling, although full-net GR acceptance covers them.

The written-review decisions added here are the exact feedback formulas and
defaults, supported fast cohort, native-total-overflow predicate, failure
semantics, initialization accounting, and deterministic representative DR
selection. Review these before implementation planning. The confirmed
checkpoint interval, wirelength bound, stop limits, and recovery policy remain
unchanged.
