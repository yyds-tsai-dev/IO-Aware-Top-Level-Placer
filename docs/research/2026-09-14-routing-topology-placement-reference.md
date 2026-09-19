# Routing-topology-aware placement: verified full-text reference

Min Wei, Xingyu Tong, Zhijie Cai, Peng Zou, Zhifeng Lin, and Jianli Chen,
**An Analytical Placement Algorithm with Routing topology Optimization**,
ASP-DAC 2024, pp. 294–299,
[DOI: 10.1109/ASP-DAC58780.2024.10473827](https://doi.org/10.1109/ASP-DAC58780.2024.10473827).

Evidence: the user supplied the six-page PDF in this directory. All six pages
were extracted; equations and Fig. 5 / Algorithm 2 were also inspected in rendered
pages. Extracted text and initial paper-search metadata are preserved under
`results/joint_route_feedback_20260914/literature/`. Earlier online PDF lookup
failed; that limitation was resolved by the supplied PDF. The initial read-error
files remain as search history, not as a description of current evidence.

## What the paper actually does

**Analytical placement, Sec. II-C / III-A, Eqs. 5–7 (pp. 295–297).** FLUTE
constructs an RSMT. Branches incident to physical pins contribute position
updates; trunks between Steiner points are constant while a topology is held
fixed. Rebuilding topology changes that partition. The hybrid objective is
WA plus smooth Steiner-segment length for the interior part, not WA plus the
entire RSMT. This avoids adding duplicate boundary-pulling terms; two- and
three-pin nets reduce to WA in the paper's model. For a two-point segment,
its weighted-average smooth length can equivalently be written
`delta * tanh(delta / (2 gamma))` on each axis. This is an algebraic restatement,
not notation used by the authors. Steiner coordinates stay fixed during each
position differentiation.

**Refinement, Sec. III-B, Eq. 8 / Fig. 5 / Algorithms 1–2 (pp. 297–298).**
The paper rebuilds FLUTE at each bin granularity and searches available area
through a quadtree. A cell's cost is weighted Manhattan distance to a multiset
of anchors, normalized by its number of anchors in each net. Anchors are adjacent
Steiner points, or midpoints for directly connected physical pins. Overlapping
cells need self anchors to preserve potential zero-length connections: one at
degree one, two at degree two; degree-three junctions get one self anchor even
without a second physical cell. Multiplicity matters. The plotted explanation
resolves Algorithm 2's directed-edge shorthand: a pin-to-Steiner branch must
keep the adjacent Steiner anchor; a direct pin-to-pin connection uses a midpoint.

The authors report lower routed wirelength with Innovus earlyGlobalRoute and
ICCAD2015 / industrial benchmarks. They do not measure this project's region IO
crossings. Their reported RWL improvement is not evidence that an IO detour
surrogate predicts an ordinary router's decisions.

## What this implementation adopts

`rsmt_anchors` implements the branch/midpoint/self-anchor rules at exact pin
coordinates. `topology_anchors` converts them into cell-origin coordinates,
normalizes the anchor multiset per cell/net, and adds its weighted Manhattan
minimizer as a search seed. The user's IO calibration weights additionally
weight each net. Candidate swaps search nearby legal, equal-size cells. All
affected multi-pin nets are then rebuilt with FLUTE; the exact joint objective
and original placement budgets determine acceptance.

This is a post-placement adaptation of the paper's anchor idea. It does not
implement their quadtree density search, bin-coarsening schedule, or analytical
GP gradient (Eqs. 5–7). Exact pin coordinates preserve physical pin offsets;
co-location means equal pin coordinates rather than equal coarse-bin indices.
The retained topology is refreshed after every router observation and every
candidate swap rebuilds all affected trees.

Shared geometric branches count once per net. Other nets contribute additional
resource demand. L/Z/detour alternatives use a 2-D capacity projection. Measured
OpenROAD occupancy subtracts each supported net's **per-layer multiplicity**
before computing background; proposal occupancy remains one unit per net/2-D
edge. Observed IO errors and resource pressure influence the next generation's
costs, including after a rejected placement. These are project extensions,
not claims from the paper; the projection does not reproduce layer assignment.

Prediction and opportunity remain distinct: optimized FLUTE paths estimate
available low-IO routes within the budget. Actual OpenROAD IO is measured
separately and must strictly improve before a placement is retained.
