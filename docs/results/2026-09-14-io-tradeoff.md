# 2026-09-14 IO / wirelength tradeoff

## Fixed-placement routing geometry

The 27,699,021-movable / 27,804,699-physical-node source placement is the verified September 6 synthetic 3x3 array. This run evaluates **all 19,972,730 in-die degree-two nets**, not the full netlist. It excludes 11,590,524 other-degree nets and 31,998 degree-two nets with outside-die pins. Pins, placement and HPWL remain fixed. K32 slicing uses region seed 0 and lattice 512.

| Per-net length budget | IO crossings | IO change vs horizontal-first | Total wirelength change | Nets using extra length with lower IO |
| --- | ---: | ---: | ---: | ---: |
| +0% | 406,354 | -12.6047% | +0.00000% | 0 |
| +2% | 405,979 | -12.6854% | +0.00388% | 315 |
| +5% | 405,408 | -12.8082% | +0.02137% | 784 |
| +10% | 404,637 | -12.9740% | +0.06950% | 1,406 |

Baseline: 464,961 crossings. At a 5% budget, 405,408 remain; HPWL change is exactly 0. Most savings already occur at zero length allowance (406,354 crossings), so **the paid-detour improvement is a further 946 crossings**, not the entire 12.81%. Each individual route respects its budget (maximum observed ratio 1.049954). The first 128 improved paths per budget were independently walked and saved, rather than selecting the best-looking examples.

The four-budget streaming routing phase took 36.923s with host HWM 2.921 GiB on the shared host. These numbers exclude input hashing/loading and do not measure native GP or OpenROAD routing.

The same 19,972,730 nets under uniform K32 grid regions are a negative control: both 0% and 5% budgets produce zero IO/length change. This illustrates that the opportunity depends on partition geometry.

Input arrays and final coordinates were additionally hashed against the previous ordered GPU validation protocol; every checked hash matched. See `results/io_tradeoff_20260914/validation/route_input_identity.json`.

## Dense-region testcase

Two fixed pins have HPWL 90. The straight path has length 90 and 11 crossings; a +5% budget permits an outer path of length 93 (+3.33%) with zero crossings. The explicit segments and region geometry are saved in `results/io_tradeoff_20260914/dense_case/`.

![Dense region detour](../../results/io_tradeoff_20260914/dense_case/dense_detour.png)

These are obstacle-free geometry proposals. They do not establish OpenROAD routing, capacity feasibility, timing or DRC cleanliness. The bounded L/Z/U candidate family is not a global route optimizer. Shared MST branches retain the original evaluator multiplicity convention.

## Placement experiments and validation

Bigblue4: all 8 matched GP+LG runs completed, including two seeds and rho 0/0.1/0.2/0.4. Every run is legal, converged and has zero unplaced cells. Both seeds select rho=0.2 under the 5% HPWL budget; rho=0.4 is rejected at 5% and becomes eligible at 10%.

| Seed | rho | IO | IO delta | HPWL delta | FT | Within 5% HPWL |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 0.1 | 77,594 | -22.588% | +1.252% | 7,855 | yes |
| 1000 | 0.2 | 64,812 | -35.340% | +4.761% | 8,854 | yes |
| 1000 | 0.4 | 63,633 | -36.516% | +5.431% | 8,138 | no |
| 1001 | 0.1 | 77,548 | -22.387% | +1.273% | 7,898 | yes |
| 1001 | 0.2 | 64,637 | -35.309% | +4.520% | 8,786 | yes |
| 1001 | 0.4 | 63,631 | -36.316% | +5.460% | 8,124 | no |

Seed1000 baseline: IO 100,235 / FT 6,936 / HPWL 748,513,576. Seed1001 baseline: IO 99,916 / FT 6,934 / HPWL 748,437,693. FT increases in these IO-focused arms and is not claimed as an improvement. Saved selections reference actual placement NPZ files and their SHA-256 values.

The 12.3M synthetic GP+LG pair and full regression are still completing. The native 1x2 ordered-equivalence test passed on the current corpus (501.86s); its initial full-suite attempt used an unavailable historical corpus path. Historical M5 results retain their original 1% guard and verdicts.

These exploratory placement sweeps were not source-frozen: optional CPU detour code/tests were added during execution, while default GPU placement/evaluation behavior was retained. Original protocol hashes and a separate pre-migration source archive remain available under `results/io_tradeoff_20260914/validation/`. This is not represented as a frozen-source preregistered comparison.


Implementation and API: [route wirelength tradeoff](../route-wirelength-tradeoff.md).
