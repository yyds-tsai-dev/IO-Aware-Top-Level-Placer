# IO-aware placement continuation (2026-09-06)

This work continues the five items requested by the user. Completion means
implementation **and** the corresponding experiments/checks below; a harness,
plan, or passing toy test alone does not complete a real-data requirement.

## Execution conditions

- Use the existing H100 NVL DREAMPlace installation documented in `docs/dev-env.md`.
- Apply `.codex/agents/deep-reasoner.toml` (Astra/high, read-only diagnosis)
  and `fast-worker.toml` (Luna/low, bounded execution).
- Preserve the pre-existing environment and agent-configuration edits.
- The user explicitly permits shared GPUs when capacity/load allows it.
  Record GPU UUID, baseline memory, utilization, and shared status. Do not
  represent shared measurements as exclusive measurements.
- The old SXM5 forecast remains historical. Register an NVL forecast before
  observing the 27.7M result; preserve the original prediction and thresholds.

## Requirements and evidence

| Item | Required work and evidence | Current state |
| --- | --- | --- |
| 1 | Correct native node ordering/counts/pin coordinates; native small-case regression; full real 1x2 equivalence including net identity and pin multiplicity | Schema3 implemented. Native real visible1x2 equivalence passed518s. All three recovered arrays match historical hashes; 3x3/1x2/2x2 full streaming verification passed1344/376/776s. |
| 2 | Recover/regenerate 27.7M inputs/cache, register actual NVL hardware, execute full GP+LG+evaluation, adjudicate the frozen prediction with resource provenance | NVL protocol/forecast frozen in49dccd4. T9 A/B each completed4 interleaved iterations within84GiB process budget. Full native GP+LG+eval completed, legality success/0 unplaced; phase sum13042.050s, process CUDA allocated peak25.308GiB. Frozen timing hypothesis false. Separate GPU-only check found IO−1/FT−2 versus historical CPU metrics; per-net diagnosis active. |
| 3 | Persist aligned per-net evaluator/degree/pin-region/boundary data; route FT and delta 0/1/2/4; >=3 routed designs and complete calibration tables/gates | All12 placements completed. Restored actual upstream GRT and launched capped DR. OpenDB pin snapshots, provenance, eligibility masks and paired calibration implemented; FFT four-arm complete paired evidence/calibration available; DES/tile routes and final three-design cohort still pending. |
| 4 | Run the IO-enabled signal/lever experiment with matched controls; pre-register and implement P0c if pursued; report confirmation and demand/noise probes without changing prior verdicts |21 registered runs completed; A2-A0 confirmation improvesFT/IO onadaptec1. All positive P0c arms failIOguard; noqualified confirmation/P4.21 final-coordinate boundary probes verified. Historical M3 FAIL unchanged; report in docs/results/2026-09-06-ft-followup.md. |
| 5 | Complete evaluator/IoTerm/host-RSS model evidence and output metadata; implement real conditional-expectation decode and boundary refinement; verify legal final geometry and run comparisons |144 primary +36 holdout component processes completed; all six fits fail coefficient-CI gate. Host18train+3holdout completed, both fits fail identification. M5 CE/refinement/legal acceptance implemented and14 focused tests passed; 1M/10M/30M bounded scale runs completed. Real screening12/12 completed. All9 proposals rejected by registered guards, final coordinates bitwise equal matched none controls. |

## Data dependencies

See `2026-09-06-input-inventory.md`. Missing historical payloads must be
retrieved or regenerated with fresh provenance; old metrics must not be
silently paired with placements from a different DREAMPlace build.

## Completion audit

Keep this checklist open until each real-data deliverable has an artifact,
command, input hashes, exit status, and a checked verdict. Report negative
experimental outcomes as outcomes, never as successful quality improvement.
Run the applicable unit/native/integration suites and the full suite after
integration; distinguish missing prerequisites from test regressions.

## Current experiment roots

- `results/recovery_visible_20260906/`: exact historical visible source/arrays,
  schema3 caches and full verification; blind downloads are a separate namespace.
- `results/h100_nvl_followup_20260906/`: frozen prediction, source snapshot,
  T9 A/B, full-run receipt/log/device history. Full run uses physical GPU1 UUID.
- `results/stage2_followup_20260906/`: three designs × flat/ours × K16/32,
  capped detailed routing and prospective paired evidence pipeline.
- `results/ft_followup_20260906/`: registration/source snapshot,21 runs,
  confirmation/pilot summary and actual boundary-demand probes.
- `results/component_models_20260906/`:144 primary measurements, frozen failed
  identification fits,36 external holdouts on physical GPU3.
- `results/host_rss_models_20260906/`: CPU native raw-read measurements and
  deterministic group net-drop inputs; fit freeze precedes2x2 holdout.

The five-item goal is still active. Passing a component test or freezing a
protocol does not complete the full-run, routing, or M5 comparison requirements.

M5 update19:45 UTC: all12 screening runs completed, all9 proposals rejected.
Final coordinates match the corresponding none arm bit-for-bit; sidecar and
execution hashes verified. No conditional confirmation is eligible. Item5's
model and M5 experiment work is complete; negative model/quality findings remain.

Validation update20:49 UTC: latest full nonslow978 passed,1 skipped,39 deselected;
driver lifecycle integration35 passed (1390s), DEF export/reweight12 passed (473s).
Both snapshot/lifetime normal behavior and exception cleanup are tested. The
missing built-in ISPD2015 benchmark alias was repaired; its native-read abort is
separate from the earlier MtKaHyPar parallel-wheel issue. Group counts overlap.
See results/validation_20260906/summary.json and hashed logs.

## 19:35 UTC checkpoint

- Latest full nonslow suite:968 passed,1 skipped,38 deselected, exit0,
  with explicit `IOPLACE_MTKAHYPAR_THREADS=1`; parallel native wheel failure
  is mitigated, not claimed fixed. Driver integration suite is running.
- Full27.7M GP+LG completed. Frozen final evaluation runs the serial CPU
  reference over31.6M nets before its GPU pass, and is still active. Preserve
  the frozen experiment. Future live IO driver now reuses GPU final metrics;
  a real40-iteration test prohibits any reference call and passed.
- Stage2 identity + complete3-design serialized-cohort tests pass52 checks;
  native OpenDB orientation/multiport checks pass2 tests. First actual FFT
  flatK16 paired calibration completed; C5 remains unavailable with1design.
- Frozen evidence source manifest:
  `results/stage2_followup_20260906/evidence_source/source_manifest.json`,
  SHA256 `b31a577918ee4ccff63c0c5893fce4b22e17c75bc5bf29f56f30c597abe871f0`.
  Its watcher processes completed routes and waits for all12cases.
- Routing scheduler `unique_route_scheduler.json` preserves two current FFT
  ours workers. Original dispatcher PID84136 is deliberately stopped until
  the unique-work scheduler finishes six remaining cases and exact flat-K
  reuse. Do not resume it early or launch duplicate routing work.

## 21:45 UTC checkpoint

H100 T14 adjudication and report are complete. A separate posthoc engineering
check retains an exact IO−1/FT−2 discrepancy; three GPU repeats have identical
per-net arrays. CPU-reference localization runs314108 selected nets including
224 boundary-risk nets. Native/cache pin-order equality is under additional
audit because the earlier real1x2 test compared endpoint multisets. Existing
first-run payloads and source archives remain immutable.

Schema4 strictnative1x2 equivalence passed537.66s; 33focusedtests and latest
fullnonslow990passed/1skip/39deselected55.75s. Neworderedcachefullverification
and27.7Mhistoricalaggregateparity remain active; originalschema3artifacts preserved.
