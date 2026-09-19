# Benchmark validity and alternative global routers

## Which cases were slow?

Two physical experiments entered expensive FastRoute congestion removal:

| Case | Placement being routed | Cells | Macros | Largest signal net | Standard-cell / row area |
| --- | --- | ---: | ---: | --- | ---: |
| mempool_tile_wrap | Joint GP step 200 | 127759 | 20 | clk_i, degree 12657 | 42.50% |
| mempool_group | Standard-WA final placement | 3077989 | 320 | clk_i, degree 342429 | 77.53% |

The observed routes ran for approximately ten and nine hours respectively
before the user-authorized protocol replacement. The tile log reached
`Extra Run for hard benchmark`; group reached extra overflow iterations.
These logs identify the stage, not an exact slow net or C++ function.

Both raw DEFs are legitimate pre-CTS, place-optimized ISPD2025 inputs. Original
and stalled exports retain the same component/master counts and connectivity
in the completed audits. The group is substantially denser. High fanout and
congestion do not establish that a benchmark is malformed.

The raw DEF labels `clk_i` as SIGNAL; the current one-shot router loads no
Liberty/SDC timing context and reports zero clock nets. These very large
unbuffered pre-CTS clocks are plausible cost contributors, but correlation is
not proof. A diagnostic ablation excludes only clk_i in explicitly non-acceptance
runs; production acceptance continues to include all signal nets. Clock or
reset nets must not be deleted merely to improve a score. Representative
post-CTS physical signoff would need a separately specified common CTS flow.

## Evidence against rejecting the tile benchmark

The original tile placement completed **all-signal** GRT at five iterations,
metal2–metal10, with **zero native overflow** in **221.934 seconds**. Native
wirelength was 4,332,029 microns. This supports original-layout routability
under this global-route policy; it does not prove detailed routability.
All four same-policy comparisons are now complete:

| Placement | Clock in routing request | Total seconds | Native overflow | Native WL, microns |
| --- | --- | ---: | ---: | ---: |
| Original | Yes | 221.934 | 0 | 4332029 |
| Original | No, diagnostic only | 61.031 | 0 | 4262193 |
| GP step 200 | Yes | 602.985 | 475827 | 12760414 |
| GP step 200 | No, diagnostic only | 628.430 | 469920 | 12695799 |

Removing the clock reduces stalled-placement overflow by only 1.24%; the GRT
command itself takes 586.556 versus 589.755 seconds. Clock processing has a
measurable cost in the original placement but does not resolve the intermediate
placement's congestion. The much worse placement geometry is therefore the
stronger explanation for the stalled tile case. This comparison does not
identify a unique failing net, prove infeasibility, or establish the same cause
for group. All receipt output hashes were recomputed successfully.

The filtered cases remain diagnostic, not full-design acceptance results. Their
remaining routed cohort is identical, with only `clk_i` removed and no added
nets. Full evidence is in `results/grt_fast_20260915/clock_diagnostic/report.md`.

## Confirmed setup and provenance problems

1. **Release mismatch in cached external-router data.** The local directory
   called `benchmarks/ispd25/visible` contains blind-release data: the tile DEF
   header names the blind design, its `.cap` grid is 10×386×386, and `.net`
   contains 135814 nets; group `.net` contains 3218496 nets with blind PPA
   weights. These match the official blind table. The physical experiments use
   archived visible DEFs (tile is the approximately 900-micron original), not
   those neighboring files. The existing physical flow does not consume these
   `.cap`/`.net` files, so this mismatch is a trap for a new adapter, not a
   demonstrated cause of the current FastRoute delay. Source paths/releases
   are pinned in the benchmark manifest; these cached files are rejected as
   inputs for the current changed placement.
2. **Layer policy differed from the contest.** Previous routing permitted
   metal1. ISPD2025 specifies that metal1 is not used for net routing. The
   adapter now supports an explicit signal-layer range; new trials use
   metal2–metal10 for both arms. Old results remain labeled with their original
   policy. This correction is not claimed as a speedup; removing a routing
   layer can increase congestion.
3. **A contest-style external adapter must regenerate physical data.** The
   official update specifies 4200×4200 GCells from the matching `.cap` data,
   rather than arbitrary DEF GCELLGRID. After GP changes coordinates, original
   `.net` pin access points are stale even if net names match. Re-export and
   validate grid, units, layers, blockages and the routed cohort. Fixing the
   layer range alone does not make this experiment the official contest
   evaluator. [Official ISPD2025 specification](https://github.com/liangrj2014/ISPD25_contest/blob/main/index.md).

No official source DEF was overwritten, no artificial capacity was applied to
the large acceptance cases, and no real signal net was removed. The artificial
small congestion probe is separately labeled.

## Alternative algorithms and OpenROAD connection

The full paper-search evidence, authors/years/DOIs, source/license checks, five
downloaded papers, and six-candidate comparison are in
`results/grt_fast_20260915/literature/report.md`.

- **Experimental native CUGR:** Current upstream OpenROAD implements
  `global_route -use_cugr`; the installed binary lacks it. Upstream explicitly
  says it is not ready for production. An isolated new build is the shortest
  native-interface experiment; it must retain explicit overflow validation.
  [Pinned official documentation](https://github.com/The-OpenROAD-Project/OpenROAD/blob/c751cdc78e74d062afbc70e1ffb0ed68553805e7/src/grt/README.md).
- **GGR/GAMER in Xplace:** GPU maze routing directly addresses sequential
  search. A documented standalone mode reads LEF/DEF, writes guides and exposes
  capacity/demand maps useful for feedback. Build and tensor/geometry semantics
  still need validation. [Author implementation](https://github.com/cuhk-eda/Xplace/blob/main/cpp_to_py/gpugr/README.md).
- **CUGR2 / EDGE:** CPU DAG routing with joint path/layer choices is a useful
  independent physical-router reference. Reads LEF/DEF and emits guides.
  [Paper](https://doi.org/10.1109/DAC56929.2023.10247702), [code](https://github.com/cuhk-eda/cu-gr-2).
- **DGR + modified CUGR2:** Differentiable routing pattern/tree selection is a
  congestion-quality experiment, with a multi-stage adapter rather than a
  direct executable replacement. [Paper](https://doi.org/10.1145/3649329.3656530), [code](https://github.com/NVlabs/Differentiable-Global-Router).
- **InstantGR:** Scalable GPU routing is relevant to feedback, but its released
  ISPD2024 `.cap`/`.net` parser is not a drop-in reader for ISPD2025's extensions
  or current LEF/DEF. [Paper](https://doi.org/10.1145/3676536.3676787), [code](https://github.com/cuhk-eda/InstantGR).

Two OpenROAD bridges are verified at the interface level: installed
`grt::read_segments` accepts physical route segments for stage-one evaluation;
`read_guides` supplies regions to detailed routing in stage two. Guides are not
exact wires or demand maps. An external feedback adapter needs actual segments
or per-edge demand/capacity plus placement/cohort/provenance checks. No candidate
was built or benchmarked on the target designs; integration feasibility is not
evidence that an algorithm will solve their congestion.
