# Paper GP gradient and existing online routing loop

Implement Sec. III-A Eqs. 5–7 of the supplied ASP-DAC 2024 paper inside actual
DREAMPlace global placement. Existing FLUTE joint cost, shared demand, and
online router-feedback placement loop remain operative, and the new GP output
must be exercised through that loop rather than only a standalone toy optimizer.

The GP objective is DREAMPlace weighted-average wirelength plus smooth interior
FLUTE branch lengths and its existing density penalty. Steiner coordinates and
branch classification are detached/frozen for each optimizer step; refreshes
occur at iteration boundaries, never in line-search objective evaluations.
Two/three-pin nets contribute only WA. Boundary contributions stay in WA.
Implement stable value and mathematically consistent gradients, including pin
positions/offsets, fixed nodes, trunk constants, tied/coincident points, weighted
nets, and periodic topology rebuilding. No gradient through FLUTE construction.

Validate frozen-objective finite differences, independently derived analytical
derivatives, CPU/CUDA parity, density/preconditioner integration, optimizer cache
refresh, actual GP trajectory changes and legality after LG. Compare WA against
paper GP with identical seeds/config, then run OpenROAD and the existing online
loop on the legal results. Preserve negative outcomes. Full GP means all
published GP model components, not merely the post-GP anchor approximation;
underspecified paper choices must be explicit, not misrepresented as exact
unpublished author code. No implementation of quadtree post-GP is claimed.

## Explicit paper interpretation

The paper does not disambiguate per-axis boundary masks. We use strict whole-pin
interior: both coordinates strictly inside the bounding rectangle, matching its
text about interior/boundary pins. Figure 2 supports adjacent original Steiner
points as virtual anchors; no unpublished remote tracing algorithm is invented.
Raw FLUTE terminal/Steiner identities and zero-length links must be preserved;
quantization stubs are not substitutes for physical branches. Duplicate pins
retain separate terminal slots. Total physical degree is capped at 256 with
explicit metadata; experiments must verify no otherwise-enabled net is omitted.

GP uses one Nesterov non-BB stage and at least one WA warmup step. Default
activation follows the paper's experiment: iteration 200 after 200 WA steps.
Topology/gamma are published before the optimizer step, both secant endpoints
refreshed, and all line-search evaluations share one generation. Existing
density weighting/preconditioning stay in DREAMPlace's real objective.

## User scope extension (2026-09-14, mandatory)

Also connect detour / shared-FLUTE IO, wirelength and capacity costs to actual GP
iteration gradients. Post-GP candidate swaps alone do not meet this new clause.
After the complete combined implementation passes numerical/integration checks,
validate large physical benchmarks and detailed routing, including actual routed
wirelength/IO/DRC outcomes. GCD and global-routing-only evidence cannot substitute
for this final validation. The previous GP-only plan is now a prerequisite,
not the completion boundary. Preserve all original joint-feedback requirements.

## Acceptance replacement authorized 2026-09-15

The user replaced the previous large full-DR completion gate with two stages.
Stage one compares matched WA/joint global-routing IO, wirelength and native
congestion, and verifies that measured feedback changes later placement.
Representative detailed routing is deferred to stage two. Earlier full-DR
requirements above describe the prior scope and must not force automatic
relaunch of the intentionally cancelled DR pipeline.

Intermediate GRT may use five or ten congestion iterations with remaining
congestion permitted; final GRT retains the declared full effort. Both arms
must share inputs, seed, net cohort, layer policy and final routing effort.
Record actual overflow and preserve negative outcomes. A successful feedback
implementation does not imply better routed quality. See the bounded-GRT plan
and results report dated 2026-09-15 for evidence and outstanding large cases.
