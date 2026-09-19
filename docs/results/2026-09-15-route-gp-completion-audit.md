# FLUTE and evaluator-feedback GP: completion audit

## Status

The implementation and small-case closed cycle are present and verified. The
full requested goal is **not complete**: the large tile/group runs and full
detailed-routing comparisons are being recovered after interruption. No large-case improvement
is claimed. This report continues the scope in
[the GP specification](../superpowers/specs/2026-09-14-paper-gp-gradient-design.md),
including its mandatory routing-gradient and large-benchmark extension.

At approximately 04:13 UTC on September 15, all eight watched original pipeline,
queue, and router process handles exited. Their logs ended mid-routing without
completion receipts; the cause and signal/exit codes are unavailable. These
interrupted attempts are not counted as completed routing. Their artifacts
remain in the original `*_rows` directories. Recovery uses new
`*_recovery_20260915` output directories, unchanged GP/routing parameters, and
GPU 3, which was idle when selected. Validated tile standard-WA, matched-WA,
and paper outputs are reused. The supervisor
`results/route_gp_20260914/recover_large_20260915.py` records actual subprocess
PIDs and exit codes under `recovery_20260915/`.

The prior post-GP swap results remain a separate experiment, documented in
[the joint FLUTE report](2026-09-14-joint-flute-feedback.md).

## Fresh verification on 2026-09-15

- FLUTE, shared routing, route GP, paper derivatives, online feedback, and repair
  adapter unit regressions: **68 passed, 1 skipped, 6 deselected** (32.16 seconds).
- Real DREAMPlace/OpenROAD integration and saved tile row-gap repair:
  **5 passed, 2 deselected** (170.24 seconds). These include paired WA/joint GP,
  nonzero IO/capacity/wirelength gradients, full objective-gradient addition,
  subsequent consumption of router observations, source placement isolation,
  fixed coordinates, and independent verification of repaired DEF files.
- Geometry, evaluator, and paper-driver regressions: **141 passed, 5
  deselected** (16.70 seconds). Live online-routing and joint-driver checks:
  **7 passed** (87.60 seconds), including the test skipped in the first suite
  when `OPENROAD_BIN` was not set.
- Completed experiment artifact audit: **2,281 SHA-256 checks, no errors**,
  including the new paired feedback-causality runs.
  It verifies protocol sources/tools/inputs, saved outputs, router and repair
  receipts, paired initial placement hashes, gradient generations, and measured
  observation publication. This is bounded evidence for completed reports,
  not evidence that pending experiments finished.

Logs and the repeatable artifact audit are under
`results/route_gp_20260914/`; the audit writes `recovery_audit.json` and lists
missing deliverables explicitly. Independent numerical/integration review found
no blocking implementation defect. Its report is preserved as
`results/route_gp_20260914/independent_review_20260915.md`.

The older archived GCD integration reports retain absolute paths into a pytest
temporary directory that subsequent tests cleaned up. The audit explicitly maps
those paths to their preserved archive copies (and the original unchanged
calibration placement), checks every original expected hash, and records the
mapping in `relocated_archive_paths`. It does not rewrite the old receipts or
pretend the temporary files still exist.

The exact collinear-union objective is only piecewise differentiable. Separating
independently movable coincident tracks can produce a finite value jump for an
arbitrarily small displacement. Autograd differentiates the selected smooth
piece; frozen FLUTE generation does not freeze coordinate-dependent union
membership across forward evaluations. CPU review probes confirmed this
limitation. Neither numerical tests nor this report claim global smoothness or
attribute the observed quality changes to these ties. A globally smooth union
surrogate would be a different objective requiring separate design and validation.

## Measured results available so far

All rows below use the repaired legal placement actually passed to OpenROAD.
Wirelength is in the normalized placement coordinate units recorded by each
run's `coord.json`, not detailed-route DBU.

| Benchmark | Mode | GP steps | GRT IO | GRT wirelength |
| --- | --- | ---: | ---: | ---: |
| GCD integration | matched WA | 24 | 593 | 39,568.421 |
| GCD integration | joint | 24 | 634 | 41,845.263 |
| mempool tile | standard WA | 649 | 31,798 | 25,752,963.158 |
| mempool tile | matched WA | 647 | 31,574 | 25,793,603.684 |
| mempool tile | paper | 670 | 39,110 | 30,647,632.105 |

The short GCD run demonstrates functioning integration, not converged quality.
The tile paper result worsens GRT IO by 23.87% and wirelength by 18.82% relative
to matched WA. Preserve this negative result. Joint tile and all group final
comparisons are pending.

GCD's saved joint report records observation versions 1, 2, and 3 first consumed
at iterations 5, 8, and 16. Its full objective-gradient sum error is
`5.960464477539063e-08`, below the recorded tolerance. This establishes real
later-step consumption; it does not establish improved routing quality.

## Isolated feedback-causality experiment

A fresh paired GCD experiment uses joint GP in both arms, the same initial
measured calibration, and topology rebuilding every step. The only treatment
change is subsequent router observations every eight steps versus none.
Trajectories match through iteration 7 and first differ at iteration 8. Both
arms have identical topology-refresh iteration lists; static calibration
consumes version 1, while online calibration consumes versions 1, 2, and 3.
This isolates an effective coordinate-update effect of subsequent observations.

After 24 steps, static calibration has GRT IO **571** and wirelength
**39,756.316**; online calibration has IO **603** and wirelength **40,043.684**.
Both outputs are legal and preserve fixed coordinates. This is a negative
quality result, not evidence of improvement. Artifacts and a runnable paired
experiment script are under `results/route_gp_20260914/gcd_causality_20260915/`
and `results/route_gp_20260914/run_causality_20260915.py`.

## Physical legality and scope

DREAMPlace's regular-core legality check did not represent fragmented DEF ROW
gaps. Every routed snapshot now uses isolated OpenROAD detailed placement,
followed by a fresh-process read/check of the persisted DEF. The common repair
preserves fixed placement, masters, connectivity, and BPin geometry. Actual
repaired orientations are used for measured prediction and final evaluation;
active GP retains its canonical static pin offsets.

For tile standard WA (127,759 components), repair changed 14,111 cell locations
and 86,572 orientations. For group standard WA (3,077,989 components), repair
changed 648,869 locations and 2,120,322 orientations. Therefore comparisons
include this common repair stage, whose effect must remain explicit.

Tile paper's final topology reports 136,208 supported nets, including 46,869
multi-pin nets, zero degree exclusions among enabled nets, and 9,381 masked
nets. These are distinct from the full router cohort. Full detailed routing
must report identity, DRC, unwired nontrivial nets, decoded wirelength, and native
wirelength reconciliation before any final physical-quality conclusion.

## Remaining completion gates

1. Independent review, relevant regressions, and the paired feedback-causality
   experiment have completed; retain their stated limitations in the final audit.
2. Finish tile joint GP and group standard WA, matched WA, paper, and joint GP.
3. Finish full detailed routing for matched WA and joint on both large designs.
4. Audit completed artifacts and compare actual IO, wirelength, legality,
   identity/cohorts, DRC, and routing completeness, retaining negative outcomes.
5. Publish the final report and mark the goal complete only when every required
   deliverable has authoritative evidence.
