# FLUTE, joint resource accounting, and online router feedback

The authorized objective is: FLUTE multi-pin costs must affect position updates
and acceptance; shared trees and capacity demand must be counted jointly; router
results must affect subsequent placement updates. A diagnostic-only FLUTE sample,
frozen independent edge costs, or a final-only router veto does not satisfy it.

## Design

Use a transactional CPU route state for a fixed supported-net cohort (all in-die
nets with degree 2..256, including every eligible multi-pin net). Build actual
FLUTE trees; use collinear segment union for physical wirelength/IO and one unit
per net per resource-grid edge for joint 2-D demand. Different nets accumulate.
Removing/replacing a net must remove its old demand exactly; failed candidates
must leave positions, routes, and demand unchanged. Global resource penalties
include background demand, zero-capacity blocking, utilization, and overflow.

Candidate L/Z paths are scored against the union of the rest of their net and
all other committed nets. Whole-net union wirelength is budgeted. Placement
proposals use equal-size movable-cell swaps, actual multi-pin FLUTE cost deltas,
and incident-net transactional rerouting. The full joint objective, original
HPWL/route-length budgets, fixed nodes, and legality gate final acceptance.
The legacy MST metric remains a diagnostic rather than an unrelated mandatory
objective that can erase a genuine multi-pin improvement.

Run OpenROAD on baseline and periodically proposed legal placements. Extract
per-net/layer route segments and resource capacity/usage from OpenDB. Preserve
exact placement/net identity and report projection limitations. Feed measured
per-net prediction errors and spatial demand back into weights/background costs,
then reroute/re-score the retained incumbent before generating the next round.
A rejected router candidate must still influence future proposals. Never stop
solely because one surrogate-improving placement was vetoed. Compare weights
only within the same observation generation. Keep router-validated selection
separate from exploratory search state, and retain baseline if none improves.

## Verification

Hand-checked shared/reversed branches count once per net, twice across two nets;
transaction removal/replacement/rollback equals full recomputation. Blocked-edge
and competing-net routes exercise whole-tree candidate selection. A multi-pin-only
placement must generate moves and its FLUTE cost must control acceptance. A
router observation must change subsequent candidate scores/order, including
after rejection, with immutable compared cohorts. Run at least two online rounds
on real GCD checkpoints, log actual OpenROAD invocations and consumed generation
hashes, compare feedback-enabled and feedback-disabled behavior, and preserve
negative outcomes. Passing unit tests alone cannot establish routed improvement.

## Constraints

Keep existing experiments and unrelated dirty work intact. Use the current src
layout and real bundled FLUTE. No silent MST fallback, no fake router evidence,
no arbitrary claims of multi-layer or signoff feasibility. Run GPU work on the
user-authorized H100 when available. Work stays on a new branch in the current
workspace so the uncommitted src migration remains authoritative.

## Full-text paper refinement

The supplied ASP-DAC 2024 paper was read in full. Candidate generation now uses
FLUTE branch anchors with pin-to-pin midpoints and self-anchor multiplicity
(Fig. 5 / Algorithm 2), plus an Eq. 8 weighted-median search seed. This is an
exact-pin, legal-swap adaptation; analytical GP gradients and the quadtree
density search are outside this implementation. See
`docs/research/2026-09-14-routing-topology-placement-reference.md`.

Observation transactions validate and reroute on an isolated wrapper before
publishing either state or history. Measured foreground subtraction preserves
per-layer multiplicities; 2-D surrogate demand remains unique per net/edge.
