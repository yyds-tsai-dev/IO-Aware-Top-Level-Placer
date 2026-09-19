# Paper GP Gradient Implementation Plan

> Execute with test-first steps and bounded independent reviews. Preserve all unrelated dirty work.

**Goal:** Implement and test the paper's full GP objective/gradient, integrated with the existing FLUTE/resource/router-feedback loop.
**Architecture:** Frozen FLUTE topology and differentiable Torch branch objective; DREAMPlace iteration-boundary attachment; paired real GP+LG+OpenROAD experiments.
**Tech Stack:** Python/Torch CPU/CUDA, bundled FLUTE, DREAMPlace, OpenROAD.
**Spec:** `docs/superpowers/specs/2026-09-14-paper-gp-gradient-design.md`

## Global constraints

Exact Eq. 7 structure; fixed Steiner coordinates per step; no silent MST fallback;
explicit degree support/exclusions; unchanged fixed cells; existing source migration
and prior result snapshots preserved; GPU 1 if available; no falsely positive routing claims.

## Tasks

- [x] 1. Verify paper branch/interior definition and stable derivative. Create `src/ioplace/ops/steiner_wirelength.py` with frozen topology, differentiable interior term, independent full WA reference, and focused CPU/GPU tests.
- [x] 2. Attach the term to real DREAMPlace GP, reusing its density/preconditioner and refreshing Nesterov state at topology changes. Add `src/scripts/run_paper_gp.py` and real integration tests; record independent term/gradient contributions and topology generations.
- [x] 3. Paired WA/paper GP runs from identical inputs on two seeds; real legal output, OpenROAD validation and at least two subsequent online feedback generations from paper GP placement.
- [ ] 4. Independent math/integration review, regressions, source/tool/result audit, report. Goal complete only after both full GP implementation and existing feedback-loop requirements are evidenced.

## Rulings

- Existing feature branch is retained; do not stage or commit unrelated migration.
- Work is already authorized by the extended explicit goal; no duplicate design approval.
- Paper-only GP and online-IO extensions have separate reported metrics; measured IO improvement is tested, never presumed.

## Progress / review rulings

- Task 1 implemented with raw FLUTE terminal identities; scalar/finite-difference,
  real Fig.2-pattern fixture and CUDA parity pass. Additional edge cases underway.
- Task 2 implemented and actual GP integration passes. First activation checks
  `grad(WA+density+B) = grad(WA+density) + grad(B)` in real PlaceObj.
- Reviewer corrections applied: distinguish pre-step overflow from post-step
  position metrics; reject fence/dynamic-pin modes; compare standard WA, matched
  cache-refresh WA, and paper GP to avoid attributing solver refresh effects to B.
- Frozen trunk constants are omitted with identical coordinate gradients; do not
  claim B scalar is total smooth StWL. Exact whole-pin boundary interpretation
  and paper ambiguity are recorded in docs/research/2026-09-14-paper-gp-math.md.

## Extended required work (user steering)

- [x] Add routed FLUTE L/Z/detour IO, shared resource and wirelength costs to GP gradients with exact geometric union and sparse GPU crossings.
- [x] Route selection uses complete-net union and all-net demand; batched native FLUTE removes Python-per-net construction bottleneck.
- [ ] Integrate the route objective and measured router calibration into actual GP, then verify effective coordinate updates and full-objective gradients.
- [ ] After all features pass, run large physical benchmarks (mempool tile 127,759 components and group 3,077,989 components) and detailed routing. Keep full DRC, IO, WL, legal/fixed/cohort checks and negative results.

Current prerequisite evidence: paper-only 250-step paired GCD runs (standard WA,
matched-refresh WA, paper) and two online generations from each paper placement
have completed. They are preliminary small-case evidence, not final scope completion.
Native batch FLUTE and GPU interval/resource primitives are separately tested;
full-default GCD detailed routing adapter returns DRC0 and DR-wire IO404.

Review: float32 candidate/grid precision fixed and demand checked against full GPU rebuild. 107 core tests pass. Full GCD DR has zero DRC and zero unwired nontrivial nets. GPU1 became busy; use idle GPU0 for subsequent runs. Final integrated-driver tests and large validation remain.

Large validation found a real prerequisite: fragmented DEF ROW gaps are absent
from DREAMPlace's regular-core legalizer. Initial tile check failed for10,405cells;
initial large runs invalidated and stopped. Add common isolated OpenROAD
placement repair to each routed snapshot/final across allarms, use repaired
coordinates plus oriented pin-offset copies for calibration/finalevaluation,
and fail-fast on Tcl/logged errors. Preserve original non-COMPONENTS DEF text
so empty nets/properties are not lost by the writer; verify persisted output
in a fresh tool process. Rerun small/tilechecks before restartinglargecases.
