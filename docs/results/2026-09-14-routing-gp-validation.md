# Routing-aware GP validation (in progress)

The routing objective now contributes actual position derivatives inside GP:
WA + density + the paper's strict-interior smooth Steiner correction + a frozen
coefficient times smooth physical IO, shared capacity penalty, union wirelength,
and learned resource prices. Batch FLUTE keeps terminal identity and zero-length
access links; topology, path choices, calibration and gamma remain fixed per step.
Both Nesterov secant endpoints are refreshed before a changed objective is used.

Physical IO sums crossings of distinct union segments; routing demand counts a
net once per resource edge and sums different nets. Hot L/Z candidates are scored
against all modeled nets plus background demand, with a whole-net length budget.
CPU candidates use the exact boundaries and tracks representable by the GPU dtype.
The final GPU demand must match the demand maintained during CPU candidate search.
Selection prioritizes blocked net-edge uses, then scalar objective, then length;
a lower blocked count can therefore accompany a larger scalar objective.

Router observations are evaluated against the actually routed, independently
legalized placement. The original GP tensor and database remain unchanged.
Observations queue new IO calibration, capacities, background and edge prices;
only a subsequent GP rebuild publishes them. A supplied calibration tensor is
re-exported independently so a stale adjacent DEF cannot calibrate another placement.
Pending observations after early termination must not count as applied feedback.

## Evidence so far

- Core regression set: 107 passed (`results/route_gp_20260914/combined_core.log`).
  Includes native batch FLUTE, CPU/CUDA route derivatives, shared geometry,
  background coupling, terminal identities, float32 boundaries, whole-net budgets,
  and invalid-observation atomicity.
- Real paired 24-step GCD integration demonstrates changed GP coordinates,
  component gradients, real PlaceObj gradient addition, and two later consumed
  online-router observations. This short run is an integration check, not evidence
  of improved quality. Final frozen-source suite: 6 integration tests passed on GPU0. The paired
  smoke result is WA593 versus joint630 actual IO; no quality gain. Observation
  versions1/2/3 were consumed at GP iterations5/8/16; none pending. Artifacts
  copied to `results/route_gp_20260914/gcd_integration`.
- Full default GCD detailed routing (`results/detailed_route_adapter_gcd_20260914_v3`):
  404 detailed-wire IO crossings, 0 DRC, 0 unwired nontrivial nets. The 16 unwired
  regular nets all have at most one terminal. Placement, connectivity, placement
  status and pin geometry are unchanged. Decoded length 12,301,130 DBU versus
  native 12,293,570 DBU (0.0615% difference).
- Earlier paper-only 250-step GCD matrix did **not** improve over matched-refresh
  WA: seed1000 IO479→494 and seed1001 IO498→507. These results are retained under
  `results/paper_gp_20260914` and must not be attributed to the new joint GP term.

## Required large validation

Inputs are SHA-256 verified in `results/route_gp_20260914/input_inventory.json`:
`mempool_tile_wrap` 127,759 components and authentic `mempool_group` 3,077,989.
Use identical inputs/seeds and compare standard WA, matched-refresh WA, paper,
and joint routing GP. Record legal/fixed checks, actual GRT IO/WL, gradient and
publication evidence, then full detailed routing on baseline and joint outputs.
Full detailed routing uses the tool default iteration limit; no shortened cap.

The 127,759-component four-mode matrix has started after feature validation.
Large runs have not yet completed. No large-design improvement or detailed-route
signoff claim is made. GPU 1 became occupied by another job; subsequent work uses
idle H100 GPU 0 with the selection recorded in each run's protocol.

## Large-design validation exposed a prerequisite bug

The initial tile run is invalid: DREAMPlace's regular-core legality check passed,
but OpenROAD rejected 10,405 instances outside the fragmented DEF ROW intervals.
For example, a cell at (348840,647080) DBU lies between valid x intervals
[20140,299060) and [440420,751640). The export retained the correct ROWs.
An older OpenROAD Python binding also swallowed the Tcl placement-check error
and continued routing. That boundary now fails the process immediately.

The two initial benchmark processes and the invalid detailed-route attempt were
stopped; their artifacts remain under the original design directories. They are
not valid comparisons. A standalone OpenROAD detailed-placement repair honoring
the original rows passes its placement check, with 1.7um mean displacement and
4% reported HPWL increase. This repair is being integrated identically for every
arm and every routed snapshot, with repaired coordinates imported before feedback.
Large validation must be rerun after the new integration tests pass.

Measured snapshots now carry movable orientations. Calibration builds a separate
netlist with N/FS/FN/S pin offsets transformed from DREAMPlace's canonical-N
movable offsets; fixed offsets remain untouched. GP continues to optimize its
explicit fixed-offset proxy. Unsupported rotations fail rather than silently
using the wrong offsets. Six offset/atomicity regressions pass alongside ten
route-core tests. Independent repaired-OpenDB comparison passes for 24 asymmetric pin bbox
centers across R0/MX, with maximum error 0.000153 DBU; see
`results/route_gp_20260914/row_gap_probe/pin_offset_validation.json`.

The OpenROAD checked-Tcl bridge has a real fail-fast reproduction: the known
invalid tile placement exits 1 in about 3 seconds, with no routing continuation;
a valid GCD check exits 0. Routing receipts also reject logged tool errors.

After fail-fast changes, full-default GCD detailed routing passes again at
`results/route_gp_20260914/gcd_detailed_final`: IO404, DRC0, no unwired
nontrivial nets, and unchanged identity. Persisted tile repair at
`results/placement_repair_tile_20260914_v3` passes separate fresh-process
verification, with 14,111 changed locations and 86,572 changed orientations;
all 145,589 original NETS entries are retained by COMPONENTS-only replacement.

## Corrected large runs restarted

All 7 driver tests passed after row repair and immutable provenance changes;
97 core tests passed. Frozen source snapshot: `source_snapshot_rows` (166files).
The corrected four-arm tile/group matrices run under the `_rows` directories.
The tile standard-WA repaired export was SHA-checked against its successful
row-repair receipt before starting full detailed routing. Large results remain
pending; the original invalid attempts stay separate.

## First corrected large results

Tile matched WA: 647 GP steps, GRT IO31,574 and per-layer union length
25,793,603.684 internal units. Paper term with rebuild interval25:670steps,
470 active steps/19 rebuilds, IO39,110 (+23.87%) and union length
30,647,632.105 (+18.82%). This is a regression. Both share identical initial
positions and the first200step trajectory. The paper model's frozen-Steiner
refresh interval differs from the per-step setting tested in the original
small paper-only driver; a controlled cadence diagnostic is under review.
No positive joint-GP or detailed-routing conclusion is available yet.
