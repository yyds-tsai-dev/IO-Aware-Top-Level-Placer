# GrandPlan (ISPD'26) — implementation-oriented digest

Date: 2026-09-18. Produced by an agent during the 2026-09-18 brainstorming
session as a close reading of the GrandPlan PDF for implementation planning;
copied verbatim into the repository on 2026-09-19. Caveats the report itself
states: it is a single reviewer's digest of one paper, not an independent
reproduction — the paper ships no public benchmark and no released code, so
several numeric knobs (`beta_j`, `alpha_1`, `wt`, `slow_max`, `m`, `q`,
`alpha`, `K_dir`, the SA temperature schedule, the hull-rebuild period, and
stopping criteria) are flagged in §9 as unspecified reproduction gaps; the
"headline caveat" (§ top) that GrandPlan is *not* a differentiable
floorplanner despite its title is called out explicitly because it drives
which parts of the paper are safe to adopt.

Source: `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/docs/research/GrandPlan-Differentiable, Simultaneous Top-Level Floorplanning and Partition-Level Cell Placement for Large-Scale IP-Cores.pdf`
Xiong, Lu, Pan, Ren (UT Austin + Nvidia), ISPD'26, 9 pages, pp. 64-72. DOI 10.1145/3764386.3779591.
Built on **C3PO** [21] (Lu et al., ASP-DAC'26) + **DREAMPlace 3.0** fence regions [6].

**Headline caveat up front:** despite the title, GrandPlan is *not* a differentiable
floorplanner. There are **no shape variables anywhere**. The only optimization variables
are cell coordinates `{x, y}` (Fig. 3). Partition shapes are produced by a *discrete*
pipeline: flat placement -> per-partition density-bin argmax -> downsample to 32x32 ->
morphological cleanup -> simulated annealing on bin labels. "Simultaneous" means
*cell-level granularity during floorplanning*, not joint continuous optimization of
shapes and cells. This is the single most important fact for deciding what to adopt.

---

## 1. Problem formulation (§2, §4.1)

**Inputs** (Fig. 1 "common inputs in LEF/DEF format"):
- Chiplet/IP-core netlist with **RTL partitions given** ("a chiplet netlist with RTL
  partitions as input", §2). The partition *membership* of every instance is an input;
  GrandPlan never re-partitions the netlist. `K` = 3..8 partitions in the experiments.
- Fixed die outline ("die boundary"), "die IO port"s, netlist hierarchy.
- Optional inputs (Fig. 1): **(1) feedthrough weights, (2) max notches per block**.
- Target density / cell utilization (0.3-0.8, §4.1), row height `h_row`, macros.

**Outputs** (§2): "non-overlapping floorplan shapes within a fixed outline, together
with a legal placement solution in which all instances are placed within their
designated regions." Shapes are **rectilinear** (union of bins on a 32x32 grid),
exactly abutted, **zero whitespace**: "This process yields non-overlapping,
zero-whitespace floorplan shapes that conform to the fixed chiplet outline" (§3.2).

**Partitions are neither soft nor hard blocks** in the packing sense. They are *label
sets over a bin grid*. Consequently:
- **Non-overlap and exact tiling are structural, not constraints**: every bin carries
  exactly one partition ID, so overlap is impossible and whitespace is zero by
  construction. There is *no* overlap penalty and *no* outline penalty in the paper.
- **Area** is a soft constraint: `E_area` (Eq. 5) penalizes only *shortfall* below
  `A_min,i = (1-theta) EA_i`, `theta = 0.05`. Surplus area is unpenalized (whitespace
  is zero so surplus is bounded implicitly).
- **Aspect ratio: no constraint at all.** The word never appears except in ref [8].
  Shape regularity is controlled indirectly by the corner count `C_max = 2` (Eq. 6) and
  the bounding-box fill ratio `rho_target = 0.8` (Eq. 7).
- **IO/pin constraints: none.** No pin capacity, no per-boundary IO budget. I/O pinning
  is only cited as *motivation* for smoothing ("To better support routing, I/O pinning,
  and power grid design", §3.3) and as future work ("extending the framework to I/O pin
  assignment", §4.4 and §5).

---

## 2. The "floorplanning model" (§3.1.1, §3.2, §3.3)

Three representations in sequence; only the first is differentiable and it is a
*clustering* proxy, not a shape.

### 2.1 Directional-extrema convex hull per partition (§3.1.1, Algorithm 1)
Per-partition point set `X_k`:
- standard cells: their coordinates;
- **macros: pseudo points.** "For macros, whose large footprints would otherwise be
  underrepresented, we enrich `X_k` by interpolating additional points on a regular grid
  within the macro area using the average standard-cell dimensions as spacing. The final
  point set is the union of standard-cell locations and interpolated macro points."

Algorithm 1: sample `m` unit directions `U = {(cos th_j, sin th_j)}_{j=0}^{m-1}` equally
spaced on `[0, 2pi)`; project `s_i = u_x x_i + u_y y_i`; keep a quantile band
`B_hi = { i : t_hi <= s_i <= t_hi + alpha (s_max - t_hi) }` with `t_hi = quantile(s, q)`,
`q in (0.5, 1)`, band factor `alpha in (0,1)`; cap `|B_hi| <= K_dir` keeping the `K_dir`
closest to `t_hi`; repeat symmetrically with `t_lo = quantile(s, 1-q)`; dedup; run
quickhull [1,24]. Then **area cap**: `A_max = EA_k` (estimated partition area from total
cell area and target density); if `Area(H) > A_max`, "the hull vertices are uniformly
scaled inward toward the centroid until the area constraint is satisfied."

Rationale for hulls over bin clustering or net reweighting (§3.1.1): (i) early
iterations have heavily overlapping partitions so "bin-based cluster boundary detection
is unreliable"; (ii) "unlike aggressive net reweighting, our approach applies forces only
to spatial outliers, thereby preserving overall wirelength quality."

The hull is **recomputed from the current cells each time and then held constant inside
the gradient** — that is the whole "differentiability" story. No shape DOFs.

### 2.2 Bin-map extraction (§3.2, Fig. 5)
Runs **after flat *legalized* placement** ("After flat legalized placement, coherent
clusters naturally emerge"). Build `K` per-partition density maps; assign each bin to the
argmax-density partition -> `1024 x 1024` map; downsample to **`32 x 32` by majority
voting**; morphological **opening then closing** [25,26]; keep only the largest connected
component per partition; assign remaining whitespace bins to the nearest partition.

### 2.3 SA refinement of the bin map (§3.3)
Discrete simulated annealing over bin labels, moves restricted to relabeling a bin to an
*adjacent* partition. "Unlike prior floorplanning approaches that swap partition
locations, our simulated annealing operates directly on the extracted bin map by flipping
bins between adjacent partitions." Every accepted move must not create fragmented regions.

---

## 3. Cell-level placement inside partitions (§3.4)

**Yes, it is fence-region placement, and it is DREAMPlace 3.0's.** "We largely follow the
fence-region placement formulation of [6], where density is modeled using multiple
independent electrostatic systems [5,9,17,20]."

Cells are constrained by a **per-region density field with virtual blockages** (not by
projection, not by a distance penalty):

    b_k = slicing_rectangles(A \ (r_k ∪ m)),                                  (9)

"where `slicing_rectangles(·)` decomposes rectilinear regions into rectangular blockages.
When an instance overlaps with `b_k`, density overflow generates a repulsive force pushing
it back into its legal region `r_k`." (`A` = placement area, `r_k` = region k; `m` most
plausibly the fixed-macro/blockage set already modeled — the paper does not define `m`.)

**Coupling direction (critical): one-way, stage-serial.** Cell gradients do *not* flow
into region shape — there are no shape variables in stages 2/3, and stage 3 explicitly
states: "This stage does not modify partition shapes or geometry; instead, it provides
physical feedback by refining cell placement within the fixed fence regions to improve
routability." Cells influence shapes only through the *discrete* density-argmax
extraction of §3.2.

### Macro preconditioning (§3.4.1, Eq. 10) — the new numeric trick
Problem: "this formulation degrades with large partitions containing many macros, where
strong density forces drive macros toward boundaries and create halos along abutting
regions." Fix (per region `k`, macro `m`; `rho_k` = macro area fraction in region `k`,
`TD` = target density, `h_row` = row height, `a_bar_k` = mean cell area in region `k`,
`slow_max` = max slowdown factor):

    p_k  = h_row^2 · min( min( (1/TD)^2 , 4.0 ) · rho_k^2 , 1 ),
    s_m  = 1 + 1.5 · min( max( a_m / a_bar_k − 1 , 0 ) , slow_max − 1 ),      (10)
    a'_m = a_m · s_m + p_k.

"These adjusted macro areas attenuate excessively large density gradients (used in
gradient scaling for optimizer inputs)" — i.e. `a'_m` enters the **preconditioner**, not
the density energy itself.

---

## 4. Objective function — every term

### 4.1 Stage 1, flat placement (§3.1)
Objective = C3PO defaults (**wirelength + density + congestion**, Fig. 3) **+ grouping**.
The paper never writes stage-1's full objective; only the grouping terms are given.

**Grouping loss** (§3.1.2, Eq. 1). `Pi_{H_k}(x_c)` = projection of cell `c_k` at `x_c`
onto the positive edge of its own hull; `Pi_{∂H_s}(x_c)` = projection onto the negative
edge of another hull; `Omega_k` = closed interior of `H_k` (including boundary):

    Pull(c_k) = (alpha_pull / 2) ‖x_c − Pi_{H_k}(x_c)‖_2^2 · 1{x_c ∉ Omega_k}
    Push(c_k) = (alpha_push / 2) Σ_{s≠k} ‖x_c − Pi_{∂H_s}(x_c)‖_2^2 · 1{x_c ∈ Omega_s}   (1)

    ∇_{x_c} Pull = alpha_pull ( x_c − Pi_{H_k}(x_c) ) 1{x_c ∉ Omega_k}
    ∇_{x_c} Push = alpha_push Σ_{s≠k} ( x_c − Pi_{∂H_s}(x_c) ) 1{x_c ∈ Omega_s}          (2)

Semantics (Fig. 4): "the quadratic pushing force is applied when a cell lies within the
intersection area of an incorrect hull: in this case, anchor points are inserted on the
corresponding negative edge by projecting the cell to hull edges"; the pulling force
"is applied when a cell lies outside its own hull: here, anchor points are inserted on the
positive edge of `H_k` by projecting the cell to the edge". So both are **quadratic
springs to an anchor**, anchor held fixed within the step. `alpha_pull = alpha_push = 1`.
Note Eq. (2) is the *exact* gradient for a fixed convex set (Moreau envelope: the
d(Pi)/dx term vanishes); the only non-smoothness is hull *reconstruction*, plus the
indicator switching at hull boundaries.

**Weight normalization / scheduling** (Eq. 3) — this is the piece worth copying:

    alpha_{2,3} = wt · ‖∇W_e(x,y)‖_2 / ‖∇Push(x,y) + ∇Pull(x,y)‖_2                       (3)

"A common Lagrange multiplier ... is further scaled relative to the wirelength gradient",
`wt` a scaling factor. Fig. 3 labels this "integrated multiplier scaling". §4.3 shows the
schedule: `wt` is **ramped by a fixed increment every 100 or 200 iterations**; constant
`wt` is strictly worse at reducing hull-intersection area than either ramp.

**Inter-partition / feedthrough net weighting** (§3.1.3) — the entire feedthrough model:
"we apply a constant weighting to inter-partition nets to emphasize their importance. The
weight is determined by the ratio `#total_net / #inter_partition_net`". Motivation:
"previous floorplanning methods, due to the lack of partition-level cell placement
information, can only formulate the wirelength of inter-partition/feedthrough nets as
**loose signals** ... without accounting for pin positions or cell placement. In contrast,
our proposed flow enables direct optimization of the criticality of the **actual signals**."

### 4.2 Stage 2, SA energy (§3.3.1, Eqs. 4-8)

    E_total = beta_0 E_area + beta_1 E_boundary + beta_2 E_compact + beta_3 E_diff        (4)

"with nonnegative weights `beta_j`. The following penalty terms are **normalized into
similar scale** before we use the weights `beta_j`." (Numeric `beta_j` are never given.)

Area, with `d_i = (A_min,i − A_c,i) / EA_i`, `A_min,i = (1 − theta) EA_i`, `theta = 0.05`:

    E_area = Σ_{i∈P} E_area,i ,   E_area,i = 50 [d_i]_+^3 + 25 [d_i]_+^2                  (5)

Boundary (corner count `C_ij` on the shared boundary of pair `(i,j) ∈ PP`, `C_max = 2`):

    E_boundary = Σ_{(i,j)∈PP} E_boundary,i,j ,   E_boundary,i,j = 0.1 ( C_ij − C_max )^2  (6)

Compactness (`rho_fill,i = A_c,i / A_bb,i`, `rho_target = 0.8`):

    E_compact = Σ_{i∈P} E_compact,i ,
    E_compact,i = 5.0 ( (rho_target − rho_fill,i) / rho_target )_+^2                      (7)

Deviation from the extracted initial map:

    D_diff = Σ_{b∈B} 1( M_b ≠ M_b^(0) ) ,  N_bins = |f| ,  E_diff = D_diff / N_bins       (8)

Note Eq. (6) is *not* one-sided: `C_ij < C_max` is also penalized, so a perfectly straight
shared boundary (`C_ij = 0`) costs `0.1·4`. Probably intentional slack, possibly a typo.

### 4.3 Stage 3, fence-region placement (§3.4.2, Eqs. 11-16)

**Region-wise illegal RUDY** — a differentiable congestion term restricted to
*intra-partition* nets whose bbox spills into *illegal* bins. Motivation: "although we
encourage partition shapes with fill ratios close to 1, rectilinear boundaries are
preserved. This can cause intra-partition nets to extend into bins outside their legal
regions (especially at notches), leading to routability issues."

For partition `k ∈ P` with intra-partition nets `E_k ⊆ E`, `B_k` its illegal bins
("analogous to virtual blockages `b_k`, but at bin granularity"), net `n ∈ E_k`:

    x_span,n = (x_max,n − x_min,n) + eps ,   y_span,n = (y_max,n − y_min,n) + eps         (11)

    ov_n(i,j) = max(0, min(x_max,n, x_i^h) − max(x_min,n, x_i^l))
              × max(0, min(y_max,n, y_j^h) − max(y_min,n, y_j^l))                        (12)

    H_n(i,j) = w_n ov_n / y_span,n ,        V_n(i,j) = w_n ov_n / x_span,n                (13)

    RUDY^illegal_n(i,j) = H_n(i,j) + V_n(i,j),   (i,j) ∈ B_k,  n ∈ E_k                    (14)

    ∇_{x_p} RUDY^illegal = (w_n / y_span,n) ∂ov_n/∂x_p
        + (w_n / x_span,n^2) ( x_span,n ∂ov_n/∂x_p − ov_n ∂x_span,n/∂x_p )                (15)

"where `∂ov_n/∂x_p = ± y_span,n` if moving `x_p` shifts the left (−) or right (+) edge of
the bounding box, and 0 otherwise; while `∂x_span,n/∂x_p = ±1` for the respective edges."

Total stage-3 objective:

    min_{x,y} ( Σ_{e∈E} W_e(x,y) + ⟨ lambda · D(x_k, y_k, k) ⟩
                + alpha_1 Σ_{k∈P} RUDY^illegal_k(x_k, y_k) )                              (16)

`⟨·⟩` is undefined in the paper; by context it is the DREAMPlace-3.0 multi-electrostatic
density sum with per-region multipliers (inner product over regions `k`). `alpha_1` is
never given a value or a scaling rule — unlike Eq. (3), the routability weight has **no
stated normalization schedule**. That is a reproduction gap.

---

## 5. Optimization schedule (§2, §3, §4.3, §4.7)

Three **sequential** stages in one automated loop (Fig. 2), *not* alternating, *not*
simultaneous in the variables:

1. **Flat GP** (no region constraints at all) with grouping + inter-partition net
   weighting + C3PO WL/density/congestion, then **flat legalization**.
   Initialization: "a flat placement initialized at the **chiplet center**" (§4.2).
   Global placement runs to roughly **iteration 1800** (§4.3); hulls are rebuilt during
   GP; `wt` ramps every 100/200 iterations.
2. **Boundary extraction** (§3.2) + **SA refinement** (§3.3): "At each temperature level,
   50 candidate moves are explored ... if it decreases, the move is accepted; otherwise,
   it is accepted with a probability that decays exponentially with the energy increase
   and the current temperature. When area imbalances are present, the neighbor-generation
   routine selects between the two move types with equal probability; if no area
   violations exist, only corner-filling moves are attempted."
   Moves: **area-balancing** (target/donor along shared boundary, candidates scored
   `min(#excess_bins, #bins_needed, #continuous_boundary_length)`, expand from highest
   score until deficit resolved) and **corner-filling** (random rectangular windows of
   **2-9 bins** along boundaries; if a corner is detected in the window, "all bins in the
   window are reassigned to the majority partition, thereby eliminating sharp notches and
   protrusions"). Converges in ~140 iterations (Fig. 10).
3. **Fence-region placement** (§3.4) **seeded from the flat placement**, then fence-region
   legalization.

**Solver: not stated.** Only "update cell locations via gradient descent" (§3.1). No
Nesterov/Adam claim in the text (Nesterov appears only in ref [20]); it inherits C3PO's
optimizer, which is in the ePlace/DREAMPlace Nesterov lineage. **Stopping criteria: not
stated** for any stage.

Runtime split (§4.7, Fig. 11): IO 11%, flat GP 47%, flat LG 7%, extraction+SA 5%,
fence-region GP 28%, fence-region LG 2%. "Seeding fence-region placement with flat
placement significantly accelerates convergence, while tailored move types keep simulated
annealing overhead negligible."

---

## 6. Legalization (§3.2, §4.7)

Not described. Only observable facts: a flat LG runs *before* extraction ("After flat
legalized placement, coherent clusters naturally emerge", §3.2) and costs 7% of runtime;
a fence-region LG runs at the end and costs 2%. Presumably C3PO/DREAMPlace LG+DP.
**Rectilinear regions are never "finalized" geometrically** — they are already exact
unions of 32x32 bins, and the final DEF is produced by adjusting sites/rows: "the
placement sites and rows in the DEF files are adjusted according to the generated
floorplan shapes" (§4.1). Bins are snapped to nothing finer than 1/32 of the die; there is
no site/row-pitch snapping discussion, which at 3nm is a real concern the paper skips.

---

## 7. Feedthrough / IO count / pin capacity / pseudo points / Steiner / straddling — exact status

| Concept | Status in GrandPlan |
|---|---|
| **Feedthrough** | Only as **constant net weight** `#total_net / #inter_partition_net` on inter-partition nets during flat placement (§3.1.3). Feedthrough == "inter-partition net"; the two words are used interchangeably ("cross-partition (feedthrough) wirelength", Abstract). Measured as `HPWL_inter`. **No count of traversed regions, no hop model, no per-boundary accounting.** "feedthrough weights" is listed as an *optional user input* in Fig. 1. |
| **IO count per boundary** | **Absent.** Closest proxies: corner count `C_ij` with `C_max = 2` (Eq. 6), and "max notches per block" as an optional input (Fig. 1). |
| **Pin capacity** | **Absent.** Partition-level routing evaluation explicitly drops it: "inter-partition nets and pinning are omitted" (§4.1). Future work: "extending the framework to I/O pin assignment" (§4.4, §5). |
| **Pseudo points** | **Yes, one use only:** macro footprint enrichment for hull construction — "we enrich `X_k` by interpolating additional points on a regular grid within the macro area using the average standard-cell dimensions as spacing" (§3.1.1). Not optimization variables; derived from the macro's position each time the hull is rebuilt. |
| **Anchor points** | Yes — the projections `Pi_{H_k}(x_c)`, `Pi_{∂H_s}(x_c)` of Eq. (1) are called anchors and are inserted on hull edges (§3.1.2, Fig. 4). Quadratic springs to those anchors. |
| **Steiner points** | **Absent.** The word does not occur. Wirelength is HPWL; congestion is RUDY over net bounding boxes (Eqs. 11-14). |
| **Cells straddling a boundary** | Never handled as such. In stage 1 cells are unconstrained (push/pull only). In stage 3 confinement is soft via the blockage density field (Eq. 9) and hard only after fence LG. Macros are the acknowledged pain point and get preconditioning (Eq. 10) because "strong density forces drive macros toward boundaries and create halos along abutting regions" (§3.4.1) — i.e. the paper concedes near-boundary macros are the weak spot, and damps them rather than modeling straddling. |

---

## 8. Benchmarks, metrics, baselines, results (§4)

**Setup** (§4.1): 8 **industrial/proprietary** benchmarks, commercial **3 nm**, "2M-21M
instances, 3-8 partitions, and cell utilizations ranging from 0.3 to 0.8". (Abstract says
"up to 25M cells"; Table 1's largest is 20.4M — inconsistent.) Hardware: "a Linux server
equipped with eight NVIDIA A100 GPUs (100 GB each) and an AMD EPYC 7742 64-core CPU".
Implemented in Python, CUDA, C++ on top of **C3PO in wirelength-driven mode**.
**No public benchmark, no released code.**

**Metrics:** scaled `HPWL`; `HPWL_inter` (HPWL of inter-partition nets); `RouteOVFL` from
the heuristic RUDY estimator [27]; **partition-level routed wirelength** from a
"commercial-quality router" on per-partition DEFs with inter-partition nets and pinning
omitted.

**Baselines:** expert-crafted rectangular and rectilinear floorplans, same underlying
placer, "initializing partition instances at the **centers** of their assigned regions"
(§4.2); plus unconstrained flat placement as a lower-bound reference. Explicitly **no
comparison to prior floorplanners**: "We do not compare against prior floorplanning
methods, as existing approaches are packing-based and do not produce exactly abutted
partitions or optimize wirelength at cell-level granularity" (§4.2).

**Table 1 geomeans** (expert = 1.000): flat `0.928 / 0.235 / 0.880`;
**ours `0.958 (−4.2%) HPWL`, `0.728 (−27.2%) HPWL_inter`, `1.024 RouteOVFL`**.
Per-case HPWL wins up to 13.5% (testcase1 `3974.2 -> 3439.1`; testcase7 `3733.1 -> 3287.5` = −11.9%); **losses on testcase5
(+5.2%) and testcase6 (+8.7%)**, and `HPWL_inter` is *worse* than expert on testcases
2/3/4 (e.g. testcase3 `107.8 -> 116.5`). `RouteOVFL` is on average **2.4% worse** than
expert. Runtimes 1537-13239 s (avg 1.2 h; 20.4M/6 case = 13239 s = 3.7 h).

**Ablations:**
- Grouping (§4.3, Fig. 9): hull-intersection area drops monotonically with stronger `wt`;
  ramping every 100/200 iters beats constant; without grouping "the blue partition splits
  into disconnected clusters due to connectivity, making boundary extraction unreliable."
- Net weighting (§4.4, Table 2): removing it costs **+45.1% `HPWL_inter`** (geomean 1.451)
  at `HPWL` geomean 1.002 — i.e. essentially free. "We also observed that inter-partition
  weighting sometimes alters the floorplan produced during flat placement, which may be
  the primary reason for the improved `HPWL_inter`."
- Router (§4.5, Fig. 8): coarse (unrefined) boundaries **+1.5%** routed WL vs refined;
  expert fence-region placement **+10.8%**. testcase6: expert has lower placement HPWL yet
  worse routed WL, "likely due to higher partition-level congestion".
- **Shapes vs seeding (§4.6, Table 3) — the most decision-relevant table.** HPWL ratios
  (ours = 1.000): expert `1.099`; **ours floorplan-only, center init `1.098`**;
  **expert regions + flat-placement seed `1.048`**; ours `1.000`.
  "The results show that **floorplan shapes alone provide little benefit**, whereas
  seeding with flat placement improves HPWL by about 5% even under expert-crafted regions."
  So of the ~9.9% gain: ~0.1% from shapes alone, ~5% from seeding, ~4.8% from the
  interaction.

---

## 9. Limitations

**Stated:** "this work does not explicitly optimize power or timing" (§5); I/O pin
assignment is future work (§4.4, §5); timing for critical cross-partition paths is future
work (§5).

**Observed (not stated):**
1. **Not differentiable w.r.t. shapes.** Area/corner/compactness/deviation objectives are
   reachable only by discrete SA on a 32x32 grid. A partition's shape granularity is
   1/32 of the die edge — at a 2 mm die that is 62.5 um quanta. Area is only controllable
   to ~1 bin = 0.1% of die area, and the `theta = 0.05` slack is doing real work.
2. **RouteOVFL regresses (1.024)** and `HPWL_inter` regresses on 3/8 cases. The claimed
   27.2% geomean `HPWL_inter` win is dominated by three cases where the expert baseline
   was terrible (testcase6 `567.4`, testcase8 `571.1`, testcase7 `513.5`); on the cases
   where the expert was already good, GrandPlan loses. A geomean over baselines of such
   uneven quality oversells the method.
3. **Partition-level routing omits inter-partition nets and pinning** (§4.1) — precisely
   the quantity the method claims to improve. The routed-WL win is therefore about
   intra-partition congestion only.
4. **Scale (relevant to our 10M-30M target):** the largest case is 20.4M cells at 3.7 h
   wall clock; the flat GP alone is 47% of that. Two full GP runs (flat + fence) are
   inherent to the flow, so cost is ~2x a single GP. Convex hulls are rebuilt during GP
   over tens of millions of points plus interpolated macro points; Algorithm 1 is
   `O(m · N)` projections per rebuild and the paper gives neither rebuild frequency nor
   `m`, `q`, `alpha`, `K_dir`. At 30M cells and `K = 8` the `Push` term's `Σ_{s≠k}`
   point-in-polygon tests over all other hulls is the obvious hotspot; the paper says
   nothing about how it is bounded (hull vertex counts are unreported).
5. **Rectilinear regions:** shapes are bin-unions, so they can be arbitrarily rectilinear;
   `E_boundary`/`E_compact` only *discourage* complexity, they do not bound the number of
   rectangles after slicing. `slicing_rectangles` (Eq. 9) cost, the resulting blockage
   count, and its effect on DREAMPlace-3.0 multi-electrostatic memory (one field per
   region) are unreported. With `K = 8` and 1024x1024 base resolution this is where
   30M-cell memory would bite.
6. **Reproduction gaps:** `beta_j`, `alpha_1`, `wt`, `slow_max`, `m`, `q`, `alpha`,
   `K_dir`, SA temperature schedule, hull rebuild period, stopping criteria, and the
   definition of `m` in Eq. (9) are all unspecified.
7. **K is small (3-8).** Nothing validates the density-argmax + largest-connected-
   component extraction at larger `K`, where partitions are smaller than a few bins.
8. **No legality guarantee on area.** `E_area` is a soft shortfall penalty with 5% slack;
   a partition can end under-area and simply lose.

---

## 10. What maps onto our items

Repo context used: `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/src/ioplace/regions.py:5-56`
(`RegionSpec.rects` is already an `(R,4)` multi-rect, lattice-snapped, validated to tile
the die with no overlap and no gaps), `src/ioplace/region_grid.py:5-24` (regions -> label
grid, i.e. exactly GrandPlan's bin map), `src/ioplace/fence_inject.py:6-34` (we already
inject DREAMPlace-3.0 fence regions with rectilinear `flat_region_boxes`),
`src/ioplace/ops/io_term.py`, `src/ioplace/ops/ft_term.py` (our softmax-SDF region
presence `q[e,k]` and `sum(q[e,k]·max(D[home[e],k]−1,0))` feedthrough surrogate).

**(a) GrandPlan as a producer of rectilinear regions + initial placement, vs. cells-at-partition-centre init — ALREADY DONE, and it is the paper's main measured result.**
Table 3 (§4.6) is exactly this experiment and the answer is unambiguous: shapes alone
`1.098` vs center-init expert `1.099` (**no benefit**), flat-placement seeding `1.048`
(**~5%**), both `1.000`. Nothing to invent for the *comparison*; what we must build is the
producer. Mapping is clean because our `RegionSet` is already a bin-union: run a flat
(region-free) GP with the grouping loss, legalize, extract `1024x1024` per-partition
density argmax -> `32x32` majority vote -> open+close+largest-CC -> SA, emit `RegionSet`
JSON via `RegionSet.to_json`. Our lattice validation (`regions.py:24-42`) will accept it
as-is provided the bin grid divides the die. **To invent:** nothing conceptual; the only
judgement calls are (i) our `lattice` vs their 32, (ii) whether our partitions come from
Mt-KaHyPar rather than RTL hierarchy — GrandPlan *assumes partition membership is given*,
so our hMETIS/Mt-KaHyPar step substitutes for their RTL input and must run **before** the
flat GP. Also note their extraction needs *legalized* flat placement.

**(b) Replace our GP/legalization with GrandPlan-like fence-region placement — MOSTLY ALREADY OURS; only two deltas are new.**
GrandPlan §3.4 *is* DREAMPlace 3.0 [6] fence regions, which `fence_inject.py` already
wires up. The genuinely new content is: **(i) macro preconditioning Eq. (10)** — a
closed-form, drop-in modification of the per-node area used in gradient preconditioning,
worth adopting verbatim if we have macros in fence regions with the reported
boundary-halo pathology; **(ii) region-wise illegal RUDY Eqs. (11)-(15)** with hand-written
gradient. Both are mechanical. **To invent:** `alpha_1`'s scale (the paper gives none) —
apply their own Eq. (3) trick, `alpha_1 = wt · ‖∇W_e‖_2 / ‖∇RUDY^illegal‖_2`; and the
`B_k` illegal-bin tensor, which we can derive directly from `RegionGrid.grid` (`grid != k`
is `B_k`) — that is a one-liner against existing code.

**(c) IO-capacity constraint per partition boundary — NOT DONE. Fully ours to invent.**
The paper has no pin/IO capacity anywhere and names it future work (§4.4, §5). Only
transferable ideas: (i) their normalization discipline (Eq. 3, and "penalty terms are
normalized into similar scale" before `beta_j`) — apply it to an IO-capacity term so it
does not fight WL; (ii) their one-sided-penalty pattern `[d]_+^3 + [d]_+^2` (Eq. 5) is a
good template for a *capacity* violation (penalize only overflow above the boundary's pin
budget, cubic+quadratic so it is `C^1` at the knee and steep beyond). Our `io_term.py`
already produces a per-net per-region presence `q[e,k]`; a per-boundary crossing count is
a contraction of `q` over adjacent region pairs, so the differentiable machinery exists.
**To invent:** the boundary-length -> pin-budget model (pins per micron of shared
boundary per layer), attributing a net's crossings to a *specific* shared boundary rather
than to a region pair (with rectilinear regions a pair can share several disjoint
boundary segments), and the schedule. This is real design work with no paper support.

**(d) Feedthrough term aware of partition density using pseudo points moved by GP — NOT DONE; we are strictly ahead of the paper here.**
GrandPlan's entire feedthrough model is a *scalar constant net weight*
`#total_net/#inter_partition_net` (§3.1.3), with no notion of which or how many regions a
net traverses, and no density awareness. Our `ft_term.py`'s
`sum_k q[e,k] · max(D[home[e],k] − 1, 0)` already encodes traversal depth. The paper's
only transferable pieces: (i) **pseudo points exist, but only as macro-footprint fillers
for hull construction** (§3.1.1) — *not* as GP variables, so "pseudo points moved by GP"
is **not** in the paper; (ii) the ablation evidence (Table 2: `+45.1% HPWL_inter` for
`HPWL` `1.002`) says cheap emphasis on inter-partition nets is nearly free in total HPWL,
which supports being aggressive with our FT weight; (iii) their honest observation that
net weighting works largely by *changing the floorplan*, not by shortening a fixed
floorplan's nets — for us that means an FT term is most valuable in the round that
*produces* regions, not after they are frozen. **To invent:** everything about making
pseudo points first-class GP variables (their gradient coupling, count, initialization,
and how they interact with the density field). Note Eq. (9)/(16) offer no mechanism for
extra movable non-cell points; DREAMPlace-3.0 fence regions would treat them as zero-area
movable nodes, which needs care so they do not perturb density/filler accounting.

**(e) Cells straddling boundaries — NOT ADDRESSED; only a damping workaround exists.**
The paper has no straddling model. Its one relevant admission is §3.4.1: strong per-region
density forces "drive macros toward boundaries and create halos along abutting regions",
answered with Eq. (10) preconditioning, i.e. slow the macro down rather than model the
straddle. Their fill-ratio / illegal-RUDY machinery (Fig. 6, Eqs. 11-15) is about *net
bounding boxes* spilling into illegal bins, not *cells* spanning a boundary — a different
object, though our feedthrough intuition is closer to theirs than to a cell-geometry one.
**To invent:** everything. If we want a cell to legitimately straddle (e.g. a wide macro
shared by two abutting partitions), we need a partial-area ownership model in the
multi-electrostatic density (a cell's area split across two fields by overlap fraction)
plus an LG that admits it. The paper's structural assumption — every instance lands
strictly inside its designated region (§2) — is incompatible with straddling, so nothing
carries over.

**Two cross-cutting things worth stealing regardless of the above:**
1. **Eq. (3) gradient-norm-ratio multiplier + ramp every 100/200 iterations.** A
   dimensionless, self-normalizing way to weight a structural term against WL, with the
   ablation (Fig. 9) showing ramping beats a constant. Directly applicable to our IO and
   FT terms, whose magnitudes differ from WL by orders of magnitude.
2. **The stage-2 SA-on-a-bin-map formulation (Eqs. 4-8) as a region post-processor.**
   Given `RegionGrid.grid`, `E_area`/`E_boundary`/`E_compact`/`E_diff` plus the two move
   types are ~200 lines of CPU code, cost 5% of runtime, and buy 1.5% routed WL (§4.5).
   It is the cheapest way to turn *any* of our region proposals (including Mt-KaHyPar +
   density-argmax) into designer-acceptable rectilinear shapes, and `E_diff` is exactly
   the knob that keeps the refined shapes close enough that the flat placement remains a
   good seed.

**The decisive experiment for us, if we adopt only one thing:** replicate Table 3's
2x2 (our grid/Mt-KaHyPar regions vs GrandPlan-extracted regions) x (center init vs
flat-placement seed) on one 10M-class case. The paper's own numbers predict the seed axis
dominates the shape axis by ~50x, which would mean our effort belongs in
*flat-placement-seeded fence-region GP*, not in better region shapes.
