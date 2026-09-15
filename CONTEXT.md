# IO-Aware Top-Level Placement

This glossary defines the placement and routing terms used in the round-based
optimization design.

## Language

**GP round**:
A complete global-placement optimization run that produces a candidate
placement. A round contains multiple optimizer iterations.
_Avoid_: Using "round" to mean a single optimizer iteration.

**Provisional placement**:
A candidate placement that has not yet passed the global-routing acceptance
check. It can seed further exploration without becoming the accepted result.
_Avoid_: Calling an unchecked candidate the accepted placement.

**GR-accepted placement**:
The most recent placement that passed the global-routing acceptance criteria.
It is the recovery point when a later candidate is rejected.

**GR checkpoint**:
A global-routing assessment at which a provisional placement is accepted or
rejected using measured routing quality.

**Initial GR baseline**:
The initial reference placement and its global-routing measurements. This
reference remains fixed when evaluating the allowed wirelength increase.

**Placement feedback**:
Information from placement evaluation that guides a subsequent GP round,
including per-net penalties and spatial penalties associated with boundaries.
_Avoid_: Using "feedback" as a synonym for either raw evaluation metrics or an
optimizer gradient.
