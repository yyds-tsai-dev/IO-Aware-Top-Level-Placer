# FT follow-up on H100 NVL (registered before execution)

This is a new experiment using the current DREAMPlace installation. The
2026-08 M3/P0b FAIL remains unchanged. Old measurements are not paired with
new-build runs. Each arm runs in a fresh process on a shared H100 NVL, as
authorized by the user; record exact commits, dirty-tree digest, input hashes,
GPU UUID/baseline/load, arguments, completion status, and final legality.

## A. Signal / lever experiment

Use adaptec1, K=16 grid, deterministic=1, seed=1000, rho_max=.40,
tau_hi=.30, tau_lo=.03, every=50, no margin, unit initial IO weights.
Use one common atomic callback order for all arms: evaluate; update weights;
measure independent WL and IO gradients; update normalization; refresh both
Nesterov secant points under the published objective. This avoids the legacy
gradient-buffer and one-event-old normalization confound.

All reweight arms use alpha=.2, cap=10, as specified by the M3 report sec7.4.
The weights are replaced by 1+alpha*min(signal,cap), not accumulated.

| Arm | IO objective | Signal | Updated buffer |
| --- | --- | --- | --- |
| A0 | on | none | none |
| A1 | on | MST crossings | WL net_weights |
| A2 | on | ft_rg = ST-max(lambda-1,0) | WL net_weights |
| A3 | on | MST crossings | io_term.w |
| A4 | on | ft_rg | io_term.w |
| A5 | off | MST crossings | WL net_weights |

Report all IO/FT/HPWL/legality/runtime results and contrasts A2-A0, A2-A1,
A2-A4, A1-A5. Run five same-build flat seeds1000..1004 to estimate the noise
floor. Confirmation seeds1001..1003 repeat A0 and A2 regardless of the
screening direction; report paired differences and one-sided95% bounds
(t=2.920, n=3). Successful placement or completed statistics do not imply
FT quality improvement. Preserve null and adverse outcomes.

## B. P0c S4a-star pilot

Execute after A. S4a uses exactly sum_e,k q[e,k]*max(D[home_e,k]-1,0), with
original-pin-count evaluator homes and deduplicated-node soft membership.
The FT term is unweighted. It is not equal to true pure-FT in the hard limit.

Same case/geometry/seed/rho/tau settings as A. No reweight or margin.
Atomic callback; home refresh every50; window ramp tau_rel .12 to .05;
kappa_max100, eps_rel .001, EMA .5. Use existing apply_ft_transaction once
per evaluator event. Freeze home/kappa through each line search. The
legacy callback plus f_ft_max0 remains a separate bit-exact regression lock.

| Arm | f_ft_max |
| --- | --- |
| P0 | 0 |
| P1 | .10 |
| P2 | .25 |
| P3 | .50 (outside P0b's tested envelope) |

- PG1: a positive arm's final ft_mst <= P0.ft_mst-62.9.
- PG2: the same arm's io_mst <= P0.io_mst+492 and HPWL <=1.005*P0.HPWL.
- PG3: record actual five-class topology transitions and cumulative
  chain-to-star minus star-to-chain, including difference from P0.
- PG4: finite/converged/legal placement, objective-evaluation median<5,
  no >10% HPWL excursion, no kappa clamp, and cancellation>=.3 when applied
  FT force is nonzero. Zero-force cancellation is not evaluable, not failure.
  Clamp/cancellation failure is reported; no adaptive rescue rule is added.

All positive arms fail PG1: stop this S4 line. PG1 and PG2 pass: repeat the
selected smallest-f qualifying arm on seeds1001..1003 against same-seed
M2-best controls and report paired one-sided95% bounds. P4=1.0 is permitted
only if P3 improves FT over P2 while satisfying PG2. No change to any gate
after seeing results.

## Required implementation checks before B

Native fp64 gradcheck; production/reference and region-chunk parity;
zero-kappa exact IoTerm dispatch; zero fixed/filler gradients with nonzero
fixed contributions; empty/saturated inputs; saved-home stability; independent
FT weights; atomic transaction/version/refresh checks; hard-limit counterexample;
actual G9 transition tests; legacy zero-FT placement identity.

## Scope of conclusions

This registration does not promise a successful FT hypothesis. Completion
requires implemented and checked mechanisms, executed arms required by the
gates, and an evidence-backed outcome. New-build results cannot retroactively
change the old M3 exit verdict.
