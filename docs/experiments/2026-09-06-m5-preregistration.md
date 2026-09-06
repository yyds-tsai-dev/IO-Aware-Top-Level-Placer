# M5 conditional-expectation decode and boundary refinement

Implement and evaluate a bounded boundary-cell optimization, with at most65536
active movable cells. This is not physical-capacity-constrained partitioning.
The existing legal placement is the incumbent and remains available throughout.

Candidates consist of the unchanged cell location and the nearest lower-left
anchor location in adjacent regions. Select active cells closest to a region
boundary, breaking ties by original node ID. Use temperature .03 times the
characteristic region length. Candidate probabilities are proportional to
exp(-L1 projection displacement/temperature); compute the all-region normalizer
for active cells so discarded mass from the neighbor restriction is measurable.
Membership uses the actual lower-left anchor convention; pin membership uses
each pin's actual offset at each candidate location.

The CE surrogate is expected touched-region crossings divided by
max(incumbent IO,1), plus frozen-MST-passed-region absence divided by
max(incumbent FT,1), plus .01 mean L1 displacement/region length, plus .1 expected
squared deviation of per-region anchor-assigned movable area from the incumbent.
Normalize each region's squared area deviation by its geometric area squared.
These are anchor-load balance targets, not physical free-area capacities.
Inactive nodes are deterministic. Multiple pins on one cell are one categorical
variable: union their candidate region memberships before taking absence
products. Use log-products plus exact-zero counts; do not divide zero factors.

Only degree2..256 incident nets participate in the CE/refinement cost, and this
limited population is reported. Frozen-tree FT is a surrogate, not the moved-pin
MST FT. All candidates, weights, normalization denominators, and tree region
sets remain frozen during a CE pass. Selecting the lowest conditional expected
cost must not increase that surrogate beyond floating-point tolerance.

The exact refinement stage recomputes affected pin-aware MST crossings, FT and
HPWL using compiled loops. Explore adjacent moves and opposite-direction swaps;
unequal-area swaps use their actual area changes. Keep objective gains and
rollback state consistent for nets shared by multiple moved cells. The accepted
result is judged on full evaluation after legalization, not only supported-net
local estimates. Excluded high-degree nets remain included in the final HPWL
check.

For each proposed placement, run the same legalizer as the incumbent. Accept
only legal geometry with unchanged fixed nodes, normalized full IO+FT lower
than the incumbent, and full HPWL≤1.01×incumbent. Otherwise retain the incumbent
and report the rejection. Report IO and FT separately: a lower combined score
does not establish that both improved. Recompute final region IDs from accepted
geometry, since legalization may move cells across region boundaries.

Compare GP+LG, CE+LG, exact-refinement+LG, and CE+refinement+LG on adaptec1 and
bigblue4, then the recovered visible mempool tile when available. Record seeds,
temperature, active/support counts, source/input hashes, CE objective trace,
candidate/accepted/rejected moves and swaps, legalizer outcome, wall time,
memory and final evaluator output. Initial comparison seed1000; confirm any
accepted quality improvement with seeds1001–1003 without tuning frozen weights.
Separately measure bounded active-work memory/time on exact synthetic1M/10M/30M
node cases; a toy-only implementation is not a scale result.

Required tests include exhaustive categorical expectations and conditioning,
zero/one factors, multiple pins per cell, unequal areas, inactive loads, ties,
incremental versus full metrics for moves/swaps/rollback, and final rejection
when legalization or an excluded high-degree net violates the acceptance guard.

Execution detail before the first real M5 comparison: use K16/grid and the
rho0 flat-observer GP driver for all four arms. In addition to the original
GP+LG incumbent, evaluate an extra-LG-only control. A proposal must beat the
normalized IO+FT score of both legal baselines, preventing an extra legalizer
call from being credited to CE/refinement. This tightens acceptance; the
original incumbent is retained on rejection. Record whether the repeated
legalizer is position-idempotent. Initial GP configs keep the existing design
defaults, with DP disabled and8 CPU threads; freeze exact config/source hashes
before measurements.
