# H100 NVL component model measurements

Register the following matrix before collecting its measurements. Historical
L4 measurements and rejected GP fits remain separate.

The primary synthetic population has exact N ∈ {500000,2000000},
E ∈ {250000,1000000}, P ∈ {2000000,8000000}, K ∈ {8,32}: 16 factorial recipes.
Net degree is floor(P/E) plus one for the first P mod E nets. Each net selects
a seeded starting node and consecutive node IDs modulo N, guaranteeing distinct
endpoints. Node positions form a row-major square lattice over [0,100000]²;
all nodes are movable, size1, pin offsets0, fp64 coordinates. This is a
controlled component benchmark, with no placement-quality claim.

Evaluator measurements use these16 recipes. IoTerm measurements cross them
with chunk budgets4000000 and16000000, yielding32 recipes. Each recipe runs
in three fresh processes, with order randomized using seed20260906. Each
process measures construction and three evaluations or forward/backward
iterations; first-iteration and subsequent timing samples remain separate.
The three process means are aggregated by recipe before fitting. Warm timings
are not counted as independent experimental replicates. Per-process timeout
is1800s. Failed/OOM/timeout recipes remain in the inventory; no silent deletion.

All measurements use the current existing H100 NVL software environment and
an available shared GPU. Record device UUID/load/free memory, source hashes,
input recipe and realized counts, dtype, layout, chunk size, active nets and
deduplicated pins, construction/runtime CUDA events and synchronized wall
times, Torch allocated/reserved peaks, and process host RSS HWM. Device-wide
memory is capacity evidence, never an isolated process estimate.

Fit memory and runtime separately. Evaluator regressors are
[1,N,P,E,E×K]; IoTerm regressors are [1,c×max(N,P′),P′,E_active].
Use scaled full-rank QR/SVD least squares, record original-unit coefficients,
covariance, residual variance, coefficient95% t intervals, recipe-grouped
leave-one-out residuals, and standardized condition number. Identification
requires condition≤30 and every coefficient interval's relative halfwidth≤25%.
Zero/near-zero coefficients do not receive an automatic pass. Preserve the
declared regressors when a gate fails.

Freeze fits before real holdouts. Evaluate the recovered visible1×2 and2×2
at K16/32 as external validation; these differ in graph/locality from the
controlled synthetic population, so out-of-population error is reported.
Prediction intervals use residual variance plus x·Cov(beta)·x and a t critical
value only when the identification gate passes. Otherwise report unvalidated
point extrapolations without a validated interval or feasibility claim.

Host RSS is a separate fresh-process native Bookshelf-read experiment:
adaptec1, bigblue4, visible group export, deterministic group net-drop25%/50%,
and visible1×2 train [1,N,P_raw]; visible2×2 is held out. Count and report
prefix/name bytes separately. Raw read HWM is not a driver-phase RSS delta.
Use three process replicates and the same identification gate; freeze before
the held-out read. Extra uniform-position stress probes are supplementary
and may not replace failed primary recipes or tune a frozen model.
