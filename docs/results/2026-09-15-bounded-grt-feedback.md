# Two-stage routing feedback validation

## Scope

The user changed acceptance on 2026-09-15: stage one compares WA/joint global
routing IO, wirelength, congestion and later placement feedback; stage two
uses representative detailed routing to assess physical wire quality. The
previous full-DR recovery pipeline was intentionally superseded. Its artifacts,
checkpoints, actual termination receipts and cancellation reason are preserved.
The original full-DR goal is not marked complete.

## Implementation and verification

Intermediate and final GRT settings are independent. The driver accepts
`--feedback-grt-iterations`, `--feedback-allow-congestion`,
`--final-grt-iterations`, `--final-allow-congestion`, `--grt-threads`, and
`--grt-signal-layers`. Existing defaults remain 50 iterations and strict
congestion handling. New large trials explicitly request intermediate 5,
final 50, allowed remaining congestion, four threads and metal2–metal10.

Each successful observation includes native final overflow, separately named
preferred-layer and aggregated resource overflow, exact connectivity and routed
net-name cohort hashes, high-degree/type audit, hashed settings, native log and
artifacts, and timings for loading, net audit, GRT and extraction. Strict
failures retain any available native congestion report and their real failure
status. Completed GP/feedback evidence is saved in `gp_completed.json` before
the potentially long final GRT.

Independent review found no blocker; its three evidence-hardening findings
were fixed with regression tests. Final targeted tests: **26 passed**, one live
case deselected there. Separate real GRT/GP integration: **2 passed**. A real
5-iteration metal2–metal10 smoke route completed with 563 routed nets and zero
overflow. The earlier 24-thread full GCD DR also completed with zero DRC and
zero unrouted nontrivial nets; it is a separate small stage-two artifact.

## Small-case findings

Identical legal WA/joint exports produced identical IO/WL/cohorts at 5, 10 and
50 iterations. All had zero overflow. In a 24-step GP experiment, 5/10/50
intermediate policies also produced identical final GP positions under the
8-step feedback cadence. This supports fidelity on this small uncongested
case; it does not establish fidelity on a congested large design.

All rows below use the same seed, initial placement, rebuild cadence and final
50-iteration GRT. Wirelength is in the normalized placement coordinate units
used by `load_observation`, not DBU.

| Arm | Final GRT IO | Final GRT WL | Overflow | Router calls |
| --- | ---: | ---: | ---: | ---: |
| Matched WA, feedback calibration 5 | 593 | 39568.421 | 0 | 2 |
| Joint, initial observation only | 571 | 39756.316 | 0 | 2 |
| Joint, 5-iteration feedback every 8 steps | 603 | 40043.684 | 0 | 4 |
| Joint, 5-iteration feedback every 16 steps | 586 | 40209.474 | 0 | 3 |
| Joint, 10-iteration feedback every 8 steps | 603 | 40043.684 | 0 | 4 |
| Joint, 50-iteration feedback every 8 steps | 603 | 40043.684 | 0 | 4 |

The static and online joint trajectories are identical before the first new
observation. They first diverge at iteration 8 or 16 respectively, with
identical topology-refresh schedules. Online observations are consumed by
subsequent GP steps: versions `[1,2,3]` and `[1,2]`. This establishes feedback
causality. The table retains negative IO/WL outcomes; it does not establish
consistent optimization improvement. The small artifacts passed **1,877 hash
checks**, resolving changed source paths against the preserved v1 snapshot.

An explicitly artificial GCD congestion test reduced capacity by 90%. With
identical placement/cohort/resources, 5 iterations retained overflow 449;
10 retained 428. The routing command took 0.296 versus 0.506 seconds in single
observations. Strict mode returned GRT-0116 for the 5-iteration route. This
validates the overflow/effort tradeoff and allowed-congestion behavior, not
large-case quality or speedup.

## Incremental and profiling probes

An isolated persistent-DB experiment legally swapped two identical-master
cells. Incremental GRT rerouted four incident nets; its canonical routes
matched fresh routing of the same changed DEF. Measured routing-command time
was 0.0376 versus 0.1388 seconds, excluding loading and production state
synchronization. No production incremental service was enabled: global GP
moves many more cells, and the current one-shot process loses router state.

Live native function sampling was attempted but ptrace attachment was denied;
`perf` was unavailable with kernel restrictions. Old logs locate the long
phase in extra congestion removal. New command/stage timings quantify GRT
versus parsing/audit/extraction, but do not identify an exact C++ hotspot.

## Large trials and completion

`results/grt_fast_20260915/run_large_stage1.py` gates launch on completed small
causality and benchmark identity audits. It runs matched WA/joint for tile and
group using five intermediate iterations, a 500-step router interval instead
of 200, and full 50-iteration final GRT. All signal nets remain included;
remaining overflow is reported. No fixed duration or speedup is promised.

The background supervisor records commands, PIDs, logs and actual exit codes
under `large_status`. A supported same-session completion/error notification
continues final auditing without model polling. Stage-one large results remain
pending until complete receipts and comparisons exist. Representative DR is
deferred; neither stage-one success nor a zero router exit proves DRC closure.

Evidence: `results/grt_fast_20260915/{small_comparison.json,fidelity.json,
small_artifact_audit.json,code_review.md,incremental/}` and the benchmark/router
diagnosis report in this directory.

## Completed tile stage-one result

The final tile arms completed and passed 815 artifact/source/tool/receipt hash
checks with no errors. Both cover the same 136,225 routed nets, preserve legal
placement/fixed identities, and use full 50-iteration metal2–metal10 GRT.

| Arm | GP steps | Final IO | Final WL, normalized | Native overflow | End-to-end seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched WA | 647 | 31560 | 25789094.211 | 0 | 745.609 |
| Joint | 668 | 39961 | 31132046.842 | 0 | 2545.297 |

Joint increased IO by 26.62% and wirelength by 20.72%; these are negative quality
results, not an optimization win. Its 5-iteration feedback at step000500 took
147.517 seconds and retained native overflow 4,764. Observation version2 was
first consumed by optimizer iteration500 after measurement iteration499. The
gradient-sum error was2.384e-7 against tolerance2.393e-4. Final overflow0 does
not erase the intermediate congestion or imply detailed-routing closure.

Group WA/joint remain pending. The group WA calibration is a live five-iteration
GRT call; the limit is not a fixed wall-time bound. The same-session notification
transport has since failed in both agent and root calls with an HTTP error.
The background process monitor remains live, but automatic wakeup must not be
claimed as currently verified. Durable status and the goal remain active.

The user additionally authorized an actual GGR/GAMER trial. Its isolated build
and evaluation plan are recorded in `2026-09-15-ggr-trial.md`; this does not
replace or silently alter the ongoing OpenROAD pair.
