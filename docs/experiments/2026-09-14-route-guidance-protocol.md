# Route-guided placement follow-up

Status: implemented and evaluated in a subsequent authorized follow-up. The
original bend proposal below failed on GCD and Bigblue4. A bounded cost-difference
swap refinement and FLUTE diagnostics were then added; see
[results and independent OpenROAD veto](../results/2026-09-14-route-feedback.md).
The settings below remain the original proposal, not a claim that every planned
real-design/seed run was completed. Actual runs and deviations are in that report.

## Signal

For each current MST branch, search the budgeted route family. When the chosen
route reduces crossings, its first/last bend defines a target for the respective
pin. Treat selected topology as fixed during one update. The signal is the
negative gradient of a weighted squared distance to those bend targets, with
crossing savings as weights. Aggregate incident-pin forces by movable node,
normalize by total incident weight, and cap each node's displacement. Fixed
nodes receive no update. Recompute topology after an accepted update.

## Initial experiment settings

- Real designs: mempool_tile_wrap and bigblue4; K32 slicing, region seed 0.
- Placement seeds: 1000 screening, 1001 confirmation with the same policy.
- GP+LG only, rho=0, deterministic=1, atomic callbacks, existing per-design
  density/bin/iteration settings; no detailed placement.
- Route proposal allowance: +5% per MST branch.
- Maximum per-round node displacement: 8 lattice cells; step fractions
  0.25, 0.5 and 1.0; at most 2 accepted rounds.
- Final full-netlist HPWL must remain within +5% of the original legal GP+LG
  baseline, even after multiple rounds.
- Every round includes a matched extra-legalization-only control. Invalid
  geometry, changed fixed nodes, or failed legality rejects a proposal.

## Assessment

Keep both legacy MST crossings and budgeted-route crossings/length. Compare the
same evaluator and eligible-net cohort before and after moving cells. Require
legacy IO improvement and a strict improvement over both incumbent and
extra-LG control in the route pair (crossings first, wirelength second); no
increase of route crossings is accepted. If crossings decrease, a routed-length
increase is permitted only within +5% of the original baseline. Lower length
with unchanged route IO is a placement improvement but is explicitly reported
as such, not as a new IO reduction. FT is a diagnostic, not silently folded
into the IO objective. All rejected proposals retain the last legal incumbent.

Report screening/confirmation outcomes, accepted steps, coordinates/hashes,
active nodes, legality, full legacy IO/FT/HPWL, and the common-cohort route
crossings/length. Include a dense-region construction for signal direction and
a uniform-grid no-op control. The routing evaluator remains obstacle-free;
these tests do not establish OpenROAD detailed-route or signoff quality.
