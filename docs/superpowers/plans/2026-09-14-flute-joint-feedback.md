# FLUTE Joint Feedback Implementation Plan

> **For agentic workers:** Execute inline with test-first steps and review checkpoints. Preserve the current workspace's uncommitted migration; no parallel agents are required.

**Goal:** Put FLUTE multi-pin cost, shared-tree capacity demand, and measured router feedback into one functioning placement loop.

**Architecture:** A transactional joint route state owns net geometry and demand. A multi-pin feedback optimizer changes legal placements using that state. An OpenROAD adapter supplies measured routes/resources to the next optimization round.

**Tech Stack:** Python, NumPy/SciPy, bundled FLUTE C++/ctypes, DREAMPlace, OpenROAD Tcl/OpenDB.

**Spec:** `docs/superpowers/specs/2026-09-14-flute-joint-feedback-design.md`

## Global Constraints

- Preserve existing experiments and unrelated dirty work.
- Real FLUTE, degree 2..256; unsupported cohort exclusions explicit and immutable.
- Original baseline HPWL and shared-union wirelength budgets; legal/fixed checks.
- Joint 2-D resource model is not a claim of detailed-route/layer feasibility.
- Router observations need real tool/input/output hashes and immutable net identity.

## Task 1: Joint capacity ledger and shared FLUTE routing

Files: create `src/ioplace/route_eval/joint.py`, `tests/test_joint_routing.py`.
Interfaces: `ResourceGrid`, `JointRoutingState`, `state.replace(net_id, pins)`,
`state.fork()`, `state.metrics()`, `JointRouteFeedback.observe(...)`.

- [x] Write hand-checked tests: net A has overlapping segments [0,3] and [1,4], consumes each edge once; net B [2,4] adds a second unit on two edges. Removal/rollback reproduces exact prior demand.
- [x] Run `pytest -q tests/test_joint_routing.py` and record the missing-behavior failures.
- [x] Implement sparse per-net occupancy, joint demand, union geometry, capacity penalty, blocked routes, and atomic replacement.
- [x] Add real FLUTE candidate routing and a capacity-conflict fixture requiring another L/Z path; whole-net union length bound applies.
- [x] Run tests plus `tests/test_flute_topology.py tests/test_budgeted_routing.py`.

## Task 2: Multi-pin placement feedback

Files: create `src/ioplace/ops/joint_route_feedback.py`, `tests/test_joint_route_feedback.py`.
Interfaces: fixed supported-net cohort; full joint score; equal-size swap proposals;
transactional incident-net rebuild; legality and original-budget acceptance.

- [x] Add a multi-pin-only netlist whose legal swap changes FLUTE cost and is selected.
- [x] Add tests for failure rollback, unchanged fixed nodes, cumulative budgets, and shared capacity affecting route ranking and observation-derived placement costs.
- [x] Implement bounded move generation, route-state deltas, and generation-consistent acceptance.
- [x] Run the new tests and existing route feedback regressions.

## Task 3: Online OpenROAD observation and subsequent updates

Files: create `src/ioplace/route_eval/online_openroad.py`, extend/create an online driver under `src/scripts/`, add `tests/test_online_router_feedback.py`.
Interfaces: observation contains exact net identity, per-net measured IO, projected
resource occupancy/background, capacities, provenance, and placement hash.

- [x] Test observation-driven candidate score/order changes and continued search after a veto.
- [x] Extract real OpenDB resources and segments, validate shapes/identity/units, and feed the retained placement's next generation.
- [x] Integrate periodic routing, calibrated net/spatial costs, router validation, and checkpoint receipts into the driver.
- [x] Run GCD for at least two online generations with and without feedback, keeping settings identical.

## Task 4: Final audit and evidence

- [x] Run relevant CPU/GPU, route geometry, resource, feedback, and OpenROAD integration checks.
- [x] Review each requirement against current source and runtime traces; fix uncovered behavior.
- [x] Record source snapshot, result hashes, actual improvements or failures, cohort exclusions, and runtime limits.
- [x] Update reports only after verification. Mark goal complete only when all three objective clauses are evidenced end-to-end.

## Completion evidence

Implemented and verified in `docs/results/2026-09-14-joint-flute-feedback.md`.
Two GCD checkpoints, three rounds each, paired learning/no-learning runs; one
checkpoint improves actual routed IO by 1, the other is unchanged. Source
snapshot and 16 real-router receipts pass `audit.py`. Bounded test suites total
118 passed, 1 skipped, 4 deselected. The unfiltered slow-suite missing-fixture
failure and interruption are disclosed in the report, not counted as passes.

The supplied ASP-DAC paper was read in full. Anchor rules were adapted; the
analytical GP gradient and quadtree refinement were not claimed as implemented.
Reviewer findings (layer multiplicities and observation transaction ordering)
were fixed with regressions and independently rechecked.
