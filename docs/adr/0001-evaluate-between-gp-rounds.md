---
status: accepted
---

# Evaluate between complete GP rounds and accept placements at GR checkpoints

The placement search needs fast feedback between complete GP runs while
bounding the cost of routing validation. The user confirmed the following
decisions during the design discussion on 2026-09-15. This record captures those
decisions; it does not claim that the implementation already follows them or
that the remaining detailed design has been approved.

## Evaluation and feedback

Run fast evaluation through a FLUTE-backed `GpuEvalContext` after each complete
GP round, outside the GP iteration loop. FLUTE replaces the MST backend for
this evaluation. Convert evaluation results into both per-net penalties and
boundary spatial penalties for the next GP round. Feedback parameters remain
fixed within a round.

Raw metrics are inputs to the feedback update rule, not feedback parameters by
themselves. Normalize IO-boundary load changes against the initial baseline on
the IO lattice. Separately evaluate routing-resource demand against GR capacity
on the actual routing grid and calibrate it at GR checkpoints. These grids and
their units must not be mixed.

## GR checkpoints and acceptance

Run a GR checkpoint every three GP rounds, with a mandatory check of the final
candidate. Make the interval configurable. Intermediate placements are
provisional; they do not replace the GR-accepted recovery point.

Accept limited routed wirelength growth in exchange for measured routed IO
improvement. Compare IO against the most recent GR-accepted placement. Require
GR overflow not to increase, and require routed wirelength to remain at or below
1.01 times the initial GR baseline wirelength. The wirelength allowance does
not accumulate across checkpoints. Compared GR results must use the same net
cohort and routing effort.

On rejection, restore the most recent GR-accepted placement and discard the
failed candidate's optimizer momentum. Retain the congestion information
revealed by the rejected GR candidate. Re-evaluate the restored placement and
derive fresh feedback before continuing, rather than blindly reusing feedback
computed for the rejected position.

Stop after at most twelve GP rounds or after two consecutive rejected GR
checkpoints, whichever comes first. Make both limits configurable. An accepted
checkpoint resets the rejection streak. Publish the most recent GR-accepted
placement; if no candidate improved on the initial baseline, retain the
baseline. A mandatory final check of an already checked, unchanged placement
may reuse the matching GR result rather than rerouting identical inputs.

## Final diagnostics

Use GR as the first diagnostic stage. Run DR only for representative cases or
benchmarks in the second stage to limit validation time.

## Trade-offs and remaining design

Checking every three rounds reduces GR calls compared with checking every
round, but a rejection can discard several rounds of provisional placement
progress. Retaining observations allows the next attempt to use information
learned from the failure while keeping the published placement GR-accepted.

The [detailed design](../superpowers/specs/2026-09-15-round-feedback-design.md)
proposes the feedback mapping, overflow measurements, numerical conventions,
routing failure handling, supported FLUTE cohort, and representative DR
selection. Those detailed defaults require written review before implementation
planning.
