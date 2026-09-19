# Reduce IO by avoiding dense region boundaries

With fixed pins, HPWL is fixed. A Manhattan MST minimizes its sum of pairwise
Manhattan edge lengths; HPWL is a separate bounding-box lower bound. Neither
chooses a physical route that minimizes region crossings. Even equally short
L/Z paths may have different crossings on a nonuniform partition.

`BudgetedRouter` keeps the pins and MST connectivity fixed, searches both L
orientations and H-V-H / V-H-V paths, and can place the middle segment outside
the pin bounding box. Candidate tracks lie half a lattice cell inside both
sides of region-band boundaries. Selection is lexicographic: fewer crossings,
then shorter wirelength. Every branch must satisfy
`route_length <= Manhattan_length * (1 + wirelength_budget)`; applying the
same bound to every branch also bounds their total length. Zero-crossing
Manhattan branches are already optimal and skip the search.

```python
from ioplace.evaluator_ref import evaluate

baseline = evaluate(netlist, node_x, node_y, region_grid)
detour = evaluate(netlist, node_x, node_y, region_grid,
                  route_wirelength_budget=0.05)
assert detour.hpwl == baseline.hpwl
assert detour.tree_wl <= baseline.tree_wl * 1.05
```

The option defaults to `None`, preserving the historical horizontal-first
MST evaluator. `0.0` searches alternative shortest paths without adding length.
The optional mode recomputes crossings, distinct feed-through regions and
boundary-pair demand by walking the selected segments. The independent
adjacency-only `io_rg` / `ft_rg` diagnostics do not change. Nets above
`max_degree` retain the existing lower-bound treatment and are not routed.
Budgeted endpoints must be finite and within the die; they are never silently
clamped. For geometry, use `route_net(router, pin_x, pin_y, ...)` or the batched
`router.route_edges(starts, ends, ...)` API in `ioplace.route_eval.budgeted`.

This is an **obstacle-free geometric evaluator**, not an OpenROAD routing
replacement. Its paths are connected, orthogonal and remain inside the die,
but are not checked against layer availability, obstacles, congestion, vias,
timing or DRC. Crossings and lengths count MST branch multiplicity, just as
the original evaluator does; shared branch segments are not deduplicated.
The bounded candidate family is not a global minimum-crossing router, and
FT is measured rather than included in its objective. The GPU evaluator still
uses its original geometry and is not interchangeable with this opt-in mode.

## Tests and measurements

The dense-region fixture has two pins at `(5,50)` and `(95,50)`, with ten
narrow regions in between. The straight path has HPWL/length 90 and 11
crossings. A 5% budget permits length 93 with zero crossings. Multi-pin
coverage checks that the evaluator also recomputes FT and boundary demand.
Random slicing tests independently walk every selected segment and check
endpoints, orthogonality, budgets and monotonic crossing counts. Uniform-grid
controls and chunk-invariance tests protect against false gains.

`scripts/benchmark_route_detours.py` streams all in-die degree-two nets from a
verified schema4 native cache. It checks the fixed-node tail against the exact
float32 DREAMPlace coordinate transform, keeps the higher-degree/outside-die
exclusions explicit, and saves the first 128 improved routes per budget for
independent geometry recounting. It does not claim full-netlist routing.

See [2026-09-14 measurements](results/2026-09-14-io-tradeoff.md) for the
dense testcase, 27.7M-cell source cohort and separate placement experiments.

## Placement tradeoff

`scripts/run_io_tradeoff.py` independently runs matched GP+LG baselines and IO
objective weights. This changes placement and therefore can change HPWL.
For example, after sourcing `scripts/env.sh` and selecting a GPU:

```bash
"$IOPLACE_PYTHON" scripts/run_io_tradeoff.py \
  --config /absolute/path/to/design.json --out results/my_new_sweep \
  --rhos 0.1 0.2 0.4 --seeds 1000 1001 --hpwl-budget 0.05
```

The output directory must be new. `selected.json` points to the chosen actual
placement, retaining the baseline if no legal, converged candidate improves
IO within budget. Comparisons require the same config, K, partition and seeds;
FT improvement cannot compensate for increased IO. Selection tables include
0/2/5/10% HPWL budgets. These are new experiments, not a regrading of the
historical M5 1% guard. Actual routed wirelength needs separate router evidence;
the HPWL selection alone makes no claim about it.


## Closed-loop feedback and FLUTE follow-up

The new opt-in `route_net(..., topology="flute")` backend uses actual FLUTE and
returns shared-segment accounting under `result["union"]`. MST can use the same
accounting with `shared_branches=True`; legacy branch-multiplicity fields remain
available. Net-union budgets are checked separately from branch budgets.
`BudgetedRouter(rg, resources=RoutingResources(...))` also supports blocked edges
and frozen capacity/demand costs for edge candidates; it does not allocate
shared multi-net capacity or model layers.

`src/scripts/run_route_feedback.py` turns route savings into bounded position
proposals and checks fresh full-net/cohort metrics. The default cost-difference
mode uses legal equal-size swaps; the original `bend` mode is retained for
ablation. `src/scripts/verify_route_feedback_grt.py` independently routes exported
checkpoints and can veto the evaluator selection. In the first two GCD seeds,
**the evaluator improved while GRT IO increased or stayed unchanged**, so the
router-selected placement is the baseline. See the
[complete results and limits](results/2026-09-14-route-feedback.md).
