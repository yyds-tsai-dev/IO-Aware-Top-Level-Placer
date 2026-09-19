# Bounded GRT Feedback Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Complete stage-one WA/joint GRT comparison and feedback causality with affordable intermediate observations. Defer representative detailed routing to stage two, per the user's 2026-09-15 scope change.

**Architecture:** Expose separate intermediate and final GRT policies in the existing GP driver. Use limited 5/10 congestion iterations with remaining congestion permitted for intermediate observations; preserve full 50-iteration final evaluation and record remaining overflow. Keep the same signal-net cohort, seed, placement initialization, final routing effort, and legality checks across compared arms.

**Tech Stack:** Python, DREAMPlace/CUDA, OpenROAD Tcl/Python, pytest.

**Spec:** The user's two-stage acceptance and five acceleration methods in this session.

## Constraints

- No fixed wall-time or speedup promise from iteration limits.
- Do not delete genuine signal nets or silently sample different cohorts.
- Never report summed-layer resource overflow as equivalent to native per-layer overflow.
- Keep actual later-step observation consumption and placement divergence evidence.
- Existing full-DR jobs are superseded by the new staged protocol; preserve artifacts and record intentional cancellation before stopping them.
- Use completion-driven background notification; no model polling loops.

## Task 1: Routing policy and evidence

Files: `src/ioplace/route_eval/online_openroad.py`, `src/ioplace/route_eval/or_scripts/dump_online_route.py`, `src/scripts/run_route_gp.py`, `tests/test_bounded_grt_feedback.py`.

- [x] Add failing tests for iteration/thread validation, `allow_congestion` command construction, native final-overflow parsing, and distinct feedback/final policy selection.
- [x] Extend `run_openroad(..., congestion_iterations=50, allow_congestion=False, threads=4)` with keyword-only settings, elapsed time, hashed settings and measured congestion evidence.
- [x] Add stage timing and net degree/type/connectivity cohort audit to the one-shot router. Record all nets; omit none from the routing request.
- [x] Add `--feedback-grt-iterations`, `--feedback-allow-congestion`, `--final-grt-iterations`, `--final-allow-congestion`, and `--grt-threads` to the GP driver. Publish policy, timing, and overflow in oracle/final reports.
- [x] Run targeted unit and real GP/OpenROAD integration tests.

## Task 2: Small-case experiments and profiling

Files: `results/grt_fast_20260915/` reproducible scripts and reports.

- [x] Record runtime `global_route` help/body, tool hash, local supported options, and live profiler restrictions. Do not infer current upstream options exist locally.
- [x] Route the same small WA/joint exports at 5, 10, and 50 iterations; compare IO, wirelength, overflow, route cohort, and stage/total seconds.
- [x] Run paired small GP experiments with fast observations and a static-feedback control; compare 8-step and 16-step router cadence with unchanged GPU objective/rebuild cadence and full final GRT.
- [x] Verify identical initialization, identical pre-feedback trajectories, subsequent consumed observations, and changed later placement. Retain negative quality outcomes.
- [x] Probe incremental routing in one live database, with a moved legal instance and unchanged connectivity. Compare with a fresh full route of the same changed placement; leave production one-shot routing intact unless evidence supports a safe persistent design.

## Task 3: Large stage-one validation

- [x] After small-case evidence, launch matched WA/joint tile and group arms in new directories with selected bounded intermediate GRT, fewer checkpoints, full final GRT, and actual congestion/cohort/timing receipts.
- [x] Keep full-GRT final evaluation costs explicit. Save GP/legality/feedback evidence before final routing so delayed final routing does not hide completed placement work.
- [ ] Publish stage-one acceptance with IO/WL/overflow/legality/identity and later-update causality; document stage-two representative DR selection separately.
