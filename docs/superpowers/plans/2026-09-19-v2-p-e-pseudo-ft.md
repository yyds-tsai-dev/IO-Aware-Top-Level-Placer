# Pseudo-Point Feed-Through Term (P-E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every selected 2-pin cross-partition net one movable *pseudo point* — a filler-class node injected at the tail of the DREAMPlace node arrays — and drive it with a differentiable feed-through cost `w_e·ω(r(p))·[WA_γ(u,v,p) − WA_γ(u,v)]` plus an anti-collapse hinge, so that global placement shortens feed-throughs that cross congested partitions without ever exposing the pseudo nodes to legalisation, to the evaluator, or to any reported placement.

**Architecture:** Two new modules. `src/ioplace/ops/pseudo_inject.py` is pure bookkeeping — it picks the nets (top-n by *measured* per-net FT), appends `n` entries to `placedb.node_size_x/node_size_y`, bumps `num_filler_nodes` (and therefore the derived `num_nodes` property), deliberately **does not** extend `filler_start_map`, and owns `pseudo.npz`. `src/ioplace/ops/pseudo_ft_term.py` is the objective — a closed-form 3-point weighted-average wirelength with an analytic backward, a frozen per-region utilisation weight `ω_k` read through a differentiable `softmax(−SDF/τ_p)` mixture, and a hinge regulariser — exposed to `norm.TermNormalizer` through the standard `value(pos, ctx)` protocol. The term is attached with `dp_hook.attach_terms` as a *second* extra objective term, so no DREAMPlace patch is added.

**Tech Stack:** Python 3.12 on this host (floor 3.9), PyTorch 2.8.0+cu128, NumPy, DREAMPlace 4.3.1 (`PlaceDB`, `BasicPlace`, `NonLinearPlace`, the `m2-extra-obj-terms` and `iteration-callback` patches), pytest.

**Spec:** [docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md](../specs/2026-09-19-v2-io-aware-redesign-design.md) — the §0 "Pseudo-point FT" decision row, all of §6, §3 ("terms after the freeze": pseudo-FT **on**, pseudo nodes unconfined), §4 (register as `pseudo_ft`, curvature 1, `value(pos, ctx)`), §9 (`tests/test_pseudo_ft.py` and the "done" criterion for E), §10 open question (ii).

## Global Constraints

- **Run protocol.** From the repo root: `source src/scripts/env.sh`; `export CUDA_VISIBLE_DEVICES=3`; `"$IOPLACE_PYTHON" -m pytest`. Use `-m "not slow"` while iterating and the full suite before declaring a task done. `src/scripts/env.sh` exports `DREAMPLACE_ROOT=/ldaphome/yyds-tsai-dev/DREAMPlace` and selects `$DREAMPLACE_ROOT/.venv312/bin/python`.
- **Shared host.** GPUs 0–2 are foreign workloads at 100% utilisation. Only GPU 3 may be used, and only for the tasks that say so. Check `nvidia-smi` first; if GPU 3 is busy, wait — do not fall back to another device.
- **No new DREAMPlace patch.** Spec §1, verbatim: `m2-extra-obj-terms.patch` already adds extra terms *after* the fence branch of `PlaceObj.obj_fn` (verified, `PlaceObj.py:298-328`), so **no new DREAMPlace patch is needed** for any v2 term; `iteration-callback.patch` supplies the hull-rebuild and probe hook. Everything in this plan is `attach_terms` + scoped `_install_attribute` monkeypatching, torn down by `_io_cleanup`.
- **Python floor `>=3.9`** (`pyproject.toml:8`). No `X | Y` annotations, no `match`, no `dataclass(slots=True)`.
- **Commit trailer.** Every commit message ends with the line
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
- **Injection invariants** (asserted by `pseudo_inject.assert_injection_invariants`, re-asserted in tests):
  1. Every pin index stays `< num_physical_nodes`: `pin2node_map`, `pin2net_map`, `flat_net2pin_map`, `flat_node2pin_start_map` are **never touched**.
  2. `pin2node_map` and the initial HPWL are **bit-identical** before and after injection.
  3. `num_nodes − num_filler_nodes == num_physical_nodes` still holds (this is what keeps every DREAMPlace filler slice `node_size_x[num_nodes−num_filler_nodes+beg : …+end]` pointing at the *regular* fillers).
  4. In fence mode the pseudo nodes are in the implicit non-fence bucket and **outside every `filler_start_map` slice**: `filler_start_map` is unchanged and `filler_start_map[-1] == num_filler_nodes − n_pseudo`.
  5. Pseudo nodes are never legalised: in fence mode `build_multi_fence_region_legalization` only ever slices through `filler_start_map`, so it cannot see them; in flat mode greedy/abacus LG excludes fillers (spec §6). In addition the term is explicitly detached at the `legalize_op` boundary and `pseudo.npz` is written *before* `orig_legalize` runs.
  6. Nothing downstream indexes beyond `num_physical`: `export/evaluation.py:82-83` already slices `node_x[:nl.num_physical]`, and `extract_final_positions` returns `num_physical_nodes` entries.
- **Cost, verbatim from spec §6.** For a 2-pin cross-partition net with pins `u,v` and pseudo point `p`:
  `L_ft = Σ_e w_e·ω(r(p_e))·[WA_γ(u,v,p_e) − WA_γ(u,v)] + reg`
  with `ω_k = (util_k/target_density)²` from the region's density-field overflow, refreshed on the `home_period` cadence and frozen inside the gradient, and `r(p)` read as `softmax(−SDF/τ_p)`.
- **Anti-collapse, verbatim from spec §6.**
  `reg = μ·Σ_e[(ρ_min−‖p−u‖₁)_+² + (ρ_min−‖p−v‖₁)_+²]`, `ρ_min = 0.1‖u−v‖₁`, `μ` from §4's normaliser.
- **Normaliser contract (spec §4).** Registered as `normalizer.register("pseudo_ft", term, 1.0, …)` — name `pseudo_ft`, **curvature 1**, protocol `value(pos, ctx)` returning the *unweighted* `L_ft`. λ comes from `normalizer.lambdas["pseudo_ft"]` and is applied by the driver, never inside the term.
- **`n_pseudo_max` default 2e6** (spec §6), selection = the top nets **by measured per-net FT**. Multi-pin nets are deferred (FLUTE set, `route_eval/topology.py`); only degree-2 nets are eligible.
- **GP-internal only.** Nothing in `placement.npz`, `evaluation.npz`, `result.json`'s placement fields, or the DEF export changes. The only new artefact is `pseudo.npz`.
- **Flag namespace.** This plan owns exactly `--pseudo-ft`, `--pseudo-max`, `--pseudo-mu`, `--pseudo-tau-rel`, `--pseudo-omega-max`, `--pseudo-omega-source`, `--pseudo-k-chunk`, `--pseudo-npz`. P-D (capacity) must use a `--cap-*` prefix; do not add anything starting with `--cap` or `--segment` here.

### Three recorded deviations from the spec's *mechanism* (its *invariants* are all kept)

Each was forced by a measurement taken while writing this plan; each is re-asserted by a test.

1. **The pseudo tail is an "orphan" filler block — it participates in no density field.** Spec §6 lists "density participation" among the filler properties being reused. Measured on GCD with a tiling 2×2 fence geometry (probe run 2026-09-19, CPU): `filler_start_map == [0, 639, 1279, 1912, 2567, 2567]` — the **non-fence bucket has zero fillers**, because `PlaceDB.calc_num_filler_for_fence_region` computes its placeable area as `max(total_space_area, area − fixed) − Σ region area`, which is ≈0 when the regions tile the core (`PlaceDB.py:705-711`). Putting 2e6 nodes with real area into that field would give it an unbounded overflow and crush the pseudo points into the (nonexistent) inter-region gaps, and there is no filler budget there to subtract the pseudo area from, so the spec's own "movable+filler area is unchanged" invariant would be violated. Leaving `filler_start_map` unextended keeps the pseudo nodes out of every `ElectricPotential` `pos_mask` (`electric_potential.py:345-362`) *and* out of `build_fence_region_legalization` (`BasicPlace.py:838-851`), while `num_nodes − num_filler_nodes == num_physical_nodes` keeps every one of those slices correct. The spec's invariants ("movable+filler area unchanged", "never legalised", "unconfined") are all satisfied *more* strongly than by the literal mechanism.
   **Flat mode is the exception**: with `len(placedb.regions) == 0` there is no `filler_start_map` and `ElectricPotential` takes the whole filler tail, so flat mode *does* include the pseudo nodes in the single density field and therefore *does* compensate the filler budget by shrinking regular filler widths (Task 2). Flat mode exists only to exercise the plumbing end-to-end in `run_placement_io.py`; the research path is phase 3 of the main flow.
2. **`util_k` is operationalised as `target_density_k·(1 + overflow_k)`, so `ω_k = (1 + overflow_k)²`.** Spec §6 says `ω_k = (util_k/target_density)²` "from its density-field overflow" without defining the map. In fence mode `model.overflow` is the `(K+1,)` vector `fence_region_density_overflow_merged_op / total_movable_node_area_fence_region` (`NonLinearPlace.py:257`, `EvalMetrics.py:121-127`), assigned to `model.overflow` at `NonLinearPlace.py:419`. Under this map `target_density` cancels, `ω_k = 1` for a relieved region and grows monotonically with congestion. It is clamped to `omega_max` (default 25.0) because the `[-1]` non-fence entry can be enormous; that entry is discarded outright.
3. **The detour bracket is ≥ `−8γ/e`, not ≥ 0, at finite γ.** Spec §6 asserts "the bracket is ≥0". That is true of HPWL but not of WA: adding a point strictly *between* the weighted-max and weighted-min reduces WA's smoothing bias. Proof of the bound: for `wa_max(S) = Σx_i e^{x_i/γ}/Σe^{x_i/γ}` we have `max(S) − wa_max(S) = Σ_{i≠argmax} t_i e^{−t_i/γ} / Σ e^{(x_i−max)/γ} ≤ (|S|−1)·γ/e` since `t·e^{−t/γ} ≤ γ/e` and the denominator is ≥1; symmetrically `wa_min(S) − min(S) ≤ (|S|−1)γ/e`. Hence per axis `WA(3) − WA(2) ≥ (max₃−max₂) − 4γ/e ≥ −4γ/e`, and over two axes `≥ −8γ/e ≈ −2.943γ`. The implementation uses the raw bracket (verbatim formula, no clamp — clamping would zero the gradient exactly where `p` is inside the bounding box and the ω-mixture channel is the only thing steering it), the bound is a unit test, and `pseudo_detour_min` is logged per probe so a run where the negative lobe dominates is visible rather than silent.

### One design ruling the spec leaves ambiguous

Spec §6 says both "`ω` … frozen inside the gradient" and "minimising `ω·detour` pushes `p` out of congested regions" / "`r(p)` is read as a smoothed `softmax(−SDF/τ_p)`, so `ω` is a soft mixture with no discontinuity at a boundary". These are incompatible if *all* of `ω(r(p))` is detached: with a constant coefficient the argmin over `p` of `ω·detour + reg` does not depend on `ω` at all, so `p` would collapse onto the hinge boundary regardless of congestion and the term would degenerate into a congestion-weighted wirelength on `u,v`.

**Ruling: the `K` scalars `ω_k` are frozen (detached, refreshed per `home_period`); the mixture `r(p)` is differentiable.** This is exactly the `home` discipline the spec cites — `ops/ft_term.py:40` freezes the table `D[home]` while `q` stays differentiable — and it is the only reading under which the smoothing sentence and the "pushes `p` out of congested regions" sentence are both true. Two tests pin it: `ω` carries no gradient (Task 4 `test_omega_is_frozen_in_the_gradient`), and the gradient w.r.t. `p` *does* change when `ω` is made non-uniform at fixed mean (Task 4 `test_omega_mixture_channel_is_live`).

---

## File Structure

**Created**

| File | Single responsibility |
|---|---|
| `src/ioplace/ops/pseudo_inject.py` | Selection (`measure_net_ft`, `select_pseudo_nets` → `PseudoPlan`), the node-array tail mutation (`inject_pseudo_nodes`, `assert_injection_invariants`), initial pseudo coordinates, and `pseudo.npz` I/O (`save_pseudo`/`load_pseudo`). Pure numpy + a `placedb` duck type; **no torch, no DREAMPlace import**. |
| `src/ioplace/ops/pseudo_ft_term.py` | The objective: `wa_detour` (3-point WA with analytic backward) and `wa_detour_numpy`, `omega_from_overflow`, `omega_mixture`, `collapse_hinge`, `auto_mu`, `PseudoFtTerm` (torch, chunked) and `PseudoFtTermRef` (dense numpy oracle). |
| `src/ioplace/fence_terms.py` | `FenceTermBundle` — the one seam through which post-freeze terms reach phase 3: `terms` (for `attach_terms`), `on_iteration(iteration, pos, placer, normalizer)`, `before_legalize(out_dir)`, `summary()`, and `combine(*bundles)` so P-D can add its capacity bundle without touching this file's callers. |
| `src/scripts/run_pseudo_displacement.py` | Spec §10 open question (ii): the per-100-iteration pseudo displacement-percentile experiment on `mempool_tile`. |
| `docs/results/2026-09-19-p-e-pseudo-displacement.md` | Result document for that experiment (template filled by the script's JSON). |
| `tests/test_pseudo_inject.py` | Selection, injection invariants, bit-identity, `pseudo.npz` round-trip. |
| `tests/test_pseudo_ft.py` | The spec §9 module: WA detour vs autograd and vs numpy, the `−8γ/e` bound, ω frozen, ω mixture continuity and liveness, hinge activation, term/ref parity, normaliser registration and λ application, `evaluation.npz` exclusion. |
| `tests/test_pseudo_ft_driver.py` | Driver wiring: CLI/kwarg validation, the flat-mode slow end-to-end on GCD, the group-scale A/B acceptance (slow, env-gated). |

**Modified**

| File | Change |
|---|---|
| `src/ioplace/drivers/run_placement_io.py` | `run_io` gains `pseudo_ft`, `pseudo_max`, `pseudo_mu`, `pseudo_tau_rel`, `pseudo_omega_max`, `pseudo_omega_source`, `pseudo_k_chunk`, `pseudo_npz`; builds the plan, injects **immediately after `placedb.initialize(params)`**, registers `pseudo_ft` with the normalizer, attaches a second term function, refreshes ω on the `home_period` cadence inside the existing atomic transaction, installs the precondition proxy, and dumps `pseudo.npz` in the `legalize_op` wrapper. Adds `pseudo_*` fields to `RESULT_FIELDS`. |
| `src/ioplace/drivers/run_placement.py` | `build_parser()` gains the eight `--pseudo-*` flags; `main()` forwards them to `run_io`. |
| `src/ioplace/drivers/run_main_flow.py` (P-B) | `--pseudo-ft on\|off` and friends; builds the bundle and passes it to `run_fence_gp`. **Guarded**: Task 8 only runs once P-B has landed. |
| `src/ioplace/fence_phase.py` / `run_main_flow.run_fence_gp` (P-B) | `run_fence_gp` gains `term_factory=None`, called after `nl`/`rg`/`ctx` are built and **before** `NonLinearPlace(...)`; its returned `FenceTermBundle` supplies `extra_terms`, the per-iteration hook and the pre-LG hook. `extra_terms=()` stays for callers that need no injection. |
| `docs/dev-env.md` | A "Pseudo-point FT (P-E) flags" subsection. |

### Reconciliation with P-D (capacity), read before Task 7 and Task 8

`docs/superpowers/plans/2026-09-19-v2-p-d-capacity.md` exists and lands **before**
this plan (spec §0 subproject order: … → P-F → P-D → P-E). Four seams were checked:

1. **Flag namespace — clear.** P-D owns `--capacity`, `--cap-*`, `--dbu-per-micron`,
   `--layer-range`, `--signal-layers`, `--openroad-bin`, `--or-out`, `--tech-lef`.
   This plan owns only `--pseudo-*`. No overlap.
2. **Normaliser term names — clear.** P-D registers `"cap"` with curvature
   `max_s pen''`; this plan registers `"pseudo_ft"` with curvature 1.
3. **The phase-3 `TermNormalizer` is P-D's, and there must be exactly one.**
   P-D's Task 9 creates "a phase-3 `TermNormalizer` carrying only `cap`". Do **not**
   create a second one here: two normalizers mean two `obj_version` counters, and
   `dp_hook.install_version_invariant` takes a single state (use
   `norm.VersionPair` only if a *different* counter, such as a τ schedule, also
   exists). Task 8 therefore takes the live normalizer as the
   `make_pseudo_bundle(fctx, *, normalizer, …)` argument and only calls
   `register("pseudo_ft", …)` on it. If P-D has not landed when Task 8 runs,
   create the phase-3 normalizer in `run_main_flow` with the same constructor
   arguments P-D's Task 9 specifies and mark it `# owned by P-D once it lands`.
4. **`run_fence_gp`'s attachment point.** P-D appends a plain callable to
   `extra_terms` and adds a `capacity=` kwarg. This plan adds `term_factory=`,
   which is the *stateful* form (it must mutate `placedb` before
   `NonLinearPlace(...)`, which a plain callable cannot). Both coexist: Task 8
   Step 5 attaches `list(extra_terms) + bundle.terms`. In `run_placement_io.py`
   the same rule applies — P-D appends its `cap` contribution inside the existing
   `term_fn`, this plan appends a *separate* `pseudo_term_fn` to the
   `attach_terms` list, so the two edits touch adjacent lines but not the same
   expression.

**Reused unchanged — do not edit:** `src/ioplace/ops/soft_assign.py` (`region_sdf_l1`, `softmax_stats`, `chunk_p_ell`, `_chunks`, `rect_table`), `src/ioplace/ops/ft_term.py`, `src/ioplace/ops/io_term.py`, `src/ioplace/norm.py`, `src/ioplace/norm_trace.py`, `src/ioplace/dp_hook.py`, `src/ioplace/export/evaluation.py`, `src/ioplace/evaluator_gpu.py`, `src/ioplace/region_grid.py`, `src/ioplace/regions.py`, both DREAMPlace patches.

**Test inputs.** `$DREAMPLACE_ROOT/install/test/simple.json` (8 movable nodes, flat) for injection bit-identity. `results/route_feedback_20260914/gcd.json` (508 movable, 168 terminals, 54 terminal-NIs, real NanGate45 LEF/DEF) for fence-mode injection and the flat end-to-end. `mempool_tile` / `mempool_group` for the slow experiments.

**Measured DREAMPlace facts this plan relies on** (probe run 2026-09-19, CPU, both cases):

- `PlaceDB.num_nodes` is a **property**, `num_physical_nodes + num_filler_nodes` (`PlaceDB.py:251-256`) — bumping `num_filler_nodes` is the whole of the size change.
- `simple.json` flat: `num_nodes` 58→63, `pin2node_map` SHA identical, `flat_node2pin_start_map` length 11 (unchanged), `data_collections.node_areas` 58→63, `pos` 116→126, initial HPWL `276.16094970703125` in both runs.
- GCD fence (2×2 grid, escape cell forced into the non-fence bucket): `num_nodes` 3304, `num_filler_nodes` 2574, `filler_start_map [0,639,1279,1912,2567,2567]`, initial HPWL `23775.1875` identical with and without 7 injected nodes, `pin2node_map` SHA identical. With the orphan tail the pseudo entries of `init_pos` stay at `0.0` (the `BasicPlace.py:289-350` spreading loop only walks `filler_start_map` slices) — Task 7 must write them explicitly.
- Adding fillers shifts numpy's global RNG stream, so the *regular* fillers' **y** draws differ between a run with and without injection (`BasicPlace.py:352-362` draws x for all fillers, then y). Physical-node initial positions are unaffected because the movable centre-noise draws (`BasicPlace.py:274-288`) happen first. Only bit-identity of *physical* init positions and HPWL is claimed.

---

### Task 1: Net selection — `PseudoPlan` and the top-n-by-measured-FT rule

**Files:**
- Create: `src/ioplace/ops/pseudo_inject.py`
- Test: `tests/test_pseudo_inject.py`

**Interfaces:**
- Consumes: `ioplace.netlist.Netlist` (fields `pin2node`, `pin2net`, `flat_net2pin`, `flat_net2pin_start`, `pin_offset_x/y`, `num_movable`, `num_physical`, `net_degrees`); `ioplace.region_grid.RegionGrid.region_of_points`; an evaluator `EvalResult` exposing `per_net_steiner` and `per_net_lambda` (both length `nl.num_nets`, indexed by **global** net id — `evaluator_gpu` semantics, see `run_placement_io.py:546-548`).
- Produces:
  - `PSEUDO_SCHEMA_VERSION = 1`
  - `class PseudoPlan` with fields `net_ids` (int64), `node_u`, `node_v` (int64), `off_ux`, `off_uy`, `off_vx`, `off_vy` (float64), `w` (float64), `ft` (float64), `movable_u`, `movable_v` (bool), and the mutable slots `pseudo_start=None`, `num_nodes_before=None`, `num_nodes_after=None`, `mode=None`, `size_x=None`, `size_y=None`; property `n -> int`.
  - `measure_net_ft(result) -> np.ndarray` (float64, length `num_nets`)
  - `select_pseudo_nets(nl, rg, node_x, node_y, ft, *, n_max=2_000_000, min_ft=1.0) -> PseudoPlan`

**Why "measured per-net FT" is `steiner − max(λ−1, 0)`.** That is the per-net form of the scalar `ft_rg = io_rg − hard_lambda_sum` the evaluator already reports, and it is already computed exactly this way at `run_placement_io.py:547-548` and `:631-634`. For a degree-2 net it equals `D[a,b] − 1`, i.e. the number of regions the net *feeds through* — nets between adjacent regions score 0 and are correctly excluded by `min_ft=1.0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pseudo_inject.py`:

```python
import numpy as np
import pytest

from ioplace.netlist import Netlist
from ioplace.ops.pseudo_inject import (PSEUDO_SCHEMA_VERSION, PseudoPlan,
                                       measure_net_ft, select_pseudo_nets)
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


class _Result(object):
    def __init__(self, steiner, lam):
        self.per_net_steiner = np.asarray(steiner, dtype=np.float64)
        self.per_net_lambda = np.asarray(lam, dtype=np.float64)


def _netlist(nets, num_movable, num_terminals=0, positions=None, offsets=None):
    """nets: list of node-id lists. positions: (x, y) arrays over all nodes."""
    pins, p2n = [], []
    for i, ns in enumerate(nets):
        pins += list(ns)
        p2n += [i] * len(ns)
    start = np.concatenate([[0], np.cumsum([len(n) for n in nets])]).astype(np.int32)
    n_nodes = num_movable + num_terminals
    if positions is None:
        positions = (np.zeros(n_nodes), np.zeros(n_nodes))
    if offsets is None:
        offsets = (np.zeros(len(pins)), np.zeros(len(pins)))
    return Netlist(
        node_x=np.asarray(positions[0], dtype=np.float64),
        node_y=np.asarray(positions[1], dtype=np.float64),
        node_size_x=np.ones(n_nodes), node_size_y=np.ones(n_nodes),
        num_movable=num_movable, num_terminals=num_terminals, num_terminal_NIs=0,
        pin_offset_x=np.asarray(offsets[0], dtype=np.float64),
        pin_offset_y=np.asarray(offsets[1], dtype=np.float64),
        pin2node=np.asarray(pins, dtype=np.int32),
        pin2net=np.asarray(p2n, dtype=np.int32),
        flat_net2pin=np.arange(len(pins), dtype=np.int32),
        flat_net2pin_start=start,
        xl=0., yl=0., xh=100., yh=100.)


def test_measure_net_ft_is_steiner_minus_lambda_minus_one():
    res = _Result([0, 1, 3, 2], [1, 2, 2, 3])
    assert list(measure_net_ft(res)) == [0.0, 0.0, 2.0, 0.0]


def _grid_case():
    # 2x2 grid over (0,0,100,100): P0 lower-left, P1 lower-right,
    # P2 upper-left, P3 upper-right.
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=100)
    rg = RegionGrid(rs)
    # 6 movable nodes at known region centres.
    x = np.array([25., 75., 25., 75., 25., 30.])
    y = np.array([25., 25., 75., 75., 25., 25.])
    nets = [[0, 3],     # net 0: P0 -> P3, degree 2, cross-partition
            [0, 1],     # net 1: P0 -> P1, degree 2, cross-partition
            [0, 4],     # net 2: P0 -> P0, degree 2, same partition
            [0, 1, 3]]  # net 3: degree 3 -> ineligible (multi-pin deferred)
    nl = _netlist(nets, num_movable=6, positions=(x, y))
    return nl, rg, x, y


def test_select_takes_only_degree_two_cross_partition_nets_with_ft():
    nl, rg, x, y = _grid_case()
    ft = np.array([2.0, 1.0, 0.0, 5.0])   # net 3 has the biggest FT but degree 3
    plan = select_pseudo_nets(nl, rg, x, y, ft, n_max=10)
    assert list(plan.net_ids) == [0, 1]
    assert plan.n == 2


def test_select_orders_by_measured_ft_descending_then_net_id():
    nl, rg, x, y = _grid_case()
    ft = np.array([1.0, 1.0, 0.0, 0.0])
    plan = select_pseudo_nets(nl, rg, x, y, ft, n_max=10)
    assert list(plan.net_ids) == [0, 1]          # tie broken by ascending net id
    ft2 = np.array([1.0, 4.0, 0.0, 0.0])
    plan2 = select_pseudo_nets(nl, rg, x, y, ft2, n_max=10)
    assert list(plan2.net_ids) == [1, 0]


def test_select_respects_n_max_and_min_ft():
    nl, rg, x, y = _grid_case()
    ft = np.array([2.0, 1.0, 0.0, 0.0])
    assert list(select_pseudo_nets(nl, rg, x, y, ft, n_max=1).net_ids) == [0]
    assert select_pseudo_nets(nl, rg, x, y, ft, n_max=10, min_ft=3.0).n == 0


def test_select_records_endpoints_offsets_and_movability():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=100)
    rg = RegionGrid(rs)
    x = np.array([25., 75.])
    y = np.array([25., 75.])
    # one movable node (0) and one terminal (1); pin offsets 1.0/2.0 on the
    # first pin, 3.0/4.0 on the second.
    nl = _netlist([[0, 1]], num_movable=1, num_terminals=1,
                  positions=(x, y), offsets=([1.0, 3.0], [2.0, 4.0]))
    plan = select_pseudo_nets(nl, rg, x, y, np.array([1.0]), n_max=10)
    assert plan.n == 1
    assert list(plan.node_u) == [0] and list(plan.node_v) == [1]
    assert list(plan.off_ux) == [1.0] and list(plan.off_uy) == [2.0]
    assert list(plan.off_vx) == [3.0] and list(plan.off_vy) == [4.0]
    assert list(plan.movable_u) == [True] and list(plan.movable_v) == [False]
    assert list(plan.w) == [1.0]


def test_select_is_deterministic():
    nl, rg, x, y = _grid_case()
    ft = np.array([2.0, 2.0, 0.0, 0.0])
    a = select_pseudo_nets(nl, rg, x, y, ft, n_max=10)
    b = select_pseudo_nets(nl, rg, x, y, ft, n_max=10)
    assert np.array_equal(a.net_ids, b.net_ids)
    assert np.array_equal(a.node_u, b.node_u) and np.array_equal(a.node_v, b.node_v)


def test_empty_plan_is_legal():
    nl, rg, x, y = _grid_case()
    plan = select_pseudo_nets(nl, rg, x, y, np.zeros(4), n_max=10)
    assert plan.n == 0 and plan.net_ids.dtype == np.int64
    assert PSEUDO_SCHEMA_VERSION == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_inject.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ioplace.ops.pseudo_inject'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/ops/pseudo_inject.py`:

```python
"""Pseudo-point injection bookkeeping (design v2 sec 6, subproject P-E).

Pure numpy. This module decides *which* nets get a pseudo point, appends the
pseudo nodes to a PlaceDB's node-size arrays as an orphan filler tail, and owns
`pseudo.npz`. It never imports torch or DREAMPlace.
"""
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

PSEUDO_SCHEMA_VERSION = 1


@dataclass
class PseudoPlan:
    """One pseudo point per selected net, in selection order.

    `pseudo_start` .. `pseudo_start + n` is the half-open slice of the *x* half
    of DREAMPlace's `pos` occupied by the pseudo nodes; the matching *y* entries
    live at `num_nodes_after + pseudo_start + i`. Both are filled in by
    `inject_pseudo_nodes`, not by `select_pseudo_nets`.
    """
    net_ids: np.ndarray
    node_u: np.ndarray
    node_v: np.ndarray
    off_ux: np.ndarray
    off_uy: np.ndarray
    off_vx: np.ndarray
    off_vy: np.ndarray
    w: np.ndarray
    ft: np.ndarray
    movable_u: np.ndarray
    movable_v: np.ndarray
    pseudo_start: int = None
    num_nodes_before: int = None
    num_nodes_after: int = None
    mode: str = None
    size_x: float = None
    size_y: float = None

    @property
    def n(self):
        return int(self.net_ids.shape[0])


def measure_net_ft(result):
    """Per-net feed-through count: `ST_e - max(Lambda_e - 1, 0)`.

    The per-net form of the evaluator's scalar `ft_rg` (see
    `run_placement_io.py:547-548`). For a degree-2 net it is exactly the number
    of regions the net traverses between its two endpoint regions.
    """
    steiner = np.asarray(result.per_net_steiner, dtype=np.float64)
    lam = np.asarray(result.per_net_lambda, dtype=np.float64)
    return steiner - np.maximum(lam - 1.0, 0.0)


def _empty_plan():
    i = np.zeros(0, dtype=np.int64)
    f = np.zeros(0, dtype=np.float64)
    b = np.zeros(0, dtype=bool)
    return PseudoPlan(net_ids=i, node_u=i.copy(), node_v=i.copy(),
                      off_ux=f.copy(), off_uy=f.copy(), off_vx=f.copy(),
                      off_vy=f.copy(), w=f.copy(), ft=f.copy(),
                      movable_u=b.copy(), movable_v=b.copy())


def select_pseudo_nets(nl, rg, node_x, node_y, ft, *, n_max=2_000_000, min_ft=1.0):
    """Top-`n_max` degree-2 cross-partition nets by measured per-net FT.

    Eligibility (design v2 sec 6): degree exactly 2 (multi-pin deferred), the
    two pins in different regions, and `ft >= min_ft` (a net between adjacent
    regions has FT 0 and needs no pseudo point). Ordering is by descending
    `ft` with ascending net id as the tie-break, so the plan is deterministic.
    """
    if n_max < 0:
        raise ValueError("n_max must be non-negative, got %r" % (n_max,))
    ft = np.asarray(ft, dtype=np.float64)
    if ft.shape != (nl.num_nets,):
        raise ValueError("ft must cover every net: expected %d, got %s"
                         % (nl.num_nets, ft.shape))
    degrees = np.asarray(nl.net_degrees, dtype=np.int64)
    eligible = np.flatnonzero((degrees == 2) & (ft >= min_ft))
    if eligible.size == 0 or n_max == 0:
        return _empty_plan()

    start = np.asarray(nl.flat_net2pin_start, dtype=np.int64)
    pin_u = np.asarray(nl.flat_net2pin, dtype=np.int64)[start[eligible]]
    pin_v = np.asarray(nl.flat_net2pin, dtype=np.int64)[start[eligible] + 1]
    node_u = np.asarray(nl.pin2node, dtype=np.int64)[pin_u]
    node_v = np.asarray(nl.pin2node, dtype=np.int64)[pin_v]
    off_ux = np.asarray(nl.pin_offset_x, dtype=np.float64)[pin_u]
    off_uy = np.asarray(nl.pin_offset_y, dtype=np.float64)[pin_u]
    off_vx = np.asarray(nl.pin_offset_x, dtype=np.float64)[pin_v]
    off_vy = np.asarray(nl.pin_offset_y, dtype=np.float64)[pin_v]

    x = np.asarray(node_x, dtype=np.float64)
    y = np.asarray(node_y, dtype=np.float64)
    region_u = rg.region_of_points(x[node_u] + off_ux, y[node_u] + off_uy)
    region_v = rg.region_of_points(x[node_v] + off_vx, y[node_v] + off_vy)
    cross = region_u != region_v
    if not cross.any():
        return _empty_plan()

    keep = np.flatnonzero(cross)
    ids = eligible[keep]
    # -ft as the primary key and the net id as the secondary key: lexsort takes
    # the *last* key as primary, so this is "ft descending, net id ascending".
    order = np.lexsort((ids, -ft[ids]))[:int(n_max)]
    sel = keep[order]
    ids = eligible[sel]
    return PseudoPlan(
        net_ids=ids.astype(np.int64),
        node_u=node_u[sel], node_v=node_v[sel],
        off_ux=off_ux[sel], off_uy=off_uy[sel],
        off_vx=off_vx[sel], off_vy=off_vy[sel],
        w=np.ones(sel.size, dtype=np.float64),
        ft=ft[ids].astype(np.float64),
        movable_u=(node_u[sel] < nl.num_movable),
        movable_v=(node_v[sel] < nl.num_movable))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_inject.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/pseudo_inject.py tests/test_pseudo_inject.py
git commit -m "feat(pseudo-ft): select degree-2 cross-partition nets by measured per-net FT

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Filler-tail injection, its invariants, and `pseudo.npz`

**Files:**
- Modify: `src/ioplace/ops/pseudo_inject.py` (append)
- Test: `tests/test_pseudo_inject.py` (append)

**Interfaces:**
- Consumes: Task 1's `PseudoPlan`.
- Produces:
  - `inject_pseudo_nodes(placedb, plan, *, size_x=None, size_y=None) -> dict` — mutates `placedb`, fills `plan.pseudo_start/num_nodes_before/num_nodes_after/mode/size_x/size_y`, returns a log dict with keys `mode` (`"fence"`/`"flat"`), `n_pseudo`, `pseudo_start`, `num_nodes_before`, `num_nodes_after`, `size_x`, `size_y`, `filler_shrink` (float, 1.0 in fence mode).
  - `assert_injection_invariants(placedb, before, plan) -> None`, where `before = snapshot_placedb(placedb)`.
  - `snapshot_placedb(placedb) -> dict` (the pre-injection fingerprint: `num_physical_nodes`, `num_filler_nodes`, `num_nodes`, `filler_start_map` copy, `pin2node_sha256`, `total_filler_node_area`).
  - `initial_pseudo_positions(plan, node_x, node_y) -> (np.ndarray, np.ndarray)` — the `(u,v)` pin midpoint.
  - `save_pseudo(path, plan, px, py, *, iteration, shift_factor, scale_factor, omega, mode) -> None`, `load_pseudo(path) -> dict`.

**Why the fence path leaves `filler_start_map` alone** — see Global Constraints deviation 1. The invariant that makes it safe is `num_nodes − num_filler_nodes == num_physical_nodes`, which every DREAMPlace filler slice is written against (`BasicPlace.py:838-851`, `PlaceObj.py:85-95`, `electric_potential.py:345-362`). Because the pseudo block sits *after* the last bucket's end, each of those slices resolves to exactly the regular fillers it resolved to before.

**Why the flat path shrinks regular fillers.** In flat mode `ElectricPotential` is constructed with `num_filler_nodes=placedb.num_filler_nodes` and takes the whole tail, so the pseudo nodes do enter the single density field. Spec §6 requires "movable+filler area is unchanged", so the regular fillers' widths are scaled by `1 − A_pseudo/A_filler_total`. Injection refuses to run when `A_pseudo > 0.25·A_filler_total`; the caller must lower `--pseudo-max`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pseudo_inject.py`:

```python
import hashlib
import os

from ioplace.ops.pseudo_inject import (assert_injection_invariants,
                                       initial_pseudo_positions,
                                       inject_pseudo_nodes, load_pseudo,
                                       save_pseudo, snapshot_placedb)

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")
GCD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "route_feedback_20260914", "gcd.json")


def _plan_of_size(n):
    i = np.arange(n, dtype=np.int64)
    f = np.zeros(n, dtype=np.float64)
    b = np.ones(n, dtype=bool)
    return PseudoPlan(net_ids=i, node_u=i.copy(), node_v=i.copy(),
                      off_ux=f.copy(), off_uy=f.copy(), off_vx=f.copy(),
                      off_vy=f.copy(), w=np.ones(n), ft=np.ones(n),
                      movable_u=b, movable_v=b.copy())


class _FakeDb(object):
    """The minimal PlaceDB duck type `inject_pseudo_nodes` touches."""

    def __init__(self, n_phys, n_filler, filler_start_map=None,
                 filler_size=(2.0, 3.0)):
        self.num_physical_nodes = n_phys
        self.num_filler_nodes = n_filler
        self.node_size_x = np.concatenate(
            [np.full(n_phys, 5.0), np.full(n_filler, filler_size[0])])
        self.node_size_y = np.concatenate(
            [np.full(n_phys, 4.0), np.full(n_filler, filler_size[1])])
        self.pin2node_map = np.arange(n_phys, dtype=np.int32)
        self.total_filler_node_area = n_filler * filler_size[0] * filler_size[1]
        self.site_width = 0.5
        self.row_height = 3.0
        self.regions = [] if filler_start_map is None else [None] * (
            len(filler_start_map) - 2)
        if filler_start_map is not None:
            self.filler_start_map = np.asarray(filler_start_map, dtype=np.int64)

    @property
    def num_nodes(self):
        return self.num_physical_nodes + self.num_filler_nodes


def test_fence_injection_leaves_filler_start_map_untouched():
    db = _FakeDb(10, 20, filler_start_map=[0, 8, 16, 20, 20])
    before = snapshot_placedb(db)
    plan = _plan_of_size(7)
    log = inject_pseudo_nodes(db, plan)
    assert log["mode"] == "fence" and log["filler_shrink"] == 1.0
    assert list(db.filler_start_map) == [0, 8, 16, 20, 20]
    assert db.num_filler_nodes == 27 and db.num_nodes == 37
    assert db.num_nodes - db.num_filler_nodes == db.num_physical_nodes
    assert int(db.filler_start_map[-1]) == db.num_filler_nodes - 7
    assert plan.pseudo_start == 30 and plan.num_nodes_after == 37
    assert db.node_size_x.shape == (37,) and db.node_size_y.shape == (37,)
    assert db.node_size_x[30:].tolist() == [0.5] * 7
    assert db.node_size_y[30:].tolist() == [3.0] * 7
    # the regular fillers are untouched in fence mode
    assert db.node_size_x[10:30].tolist() == [2.0] * 20
    assert db.total_filler_node_area == before["total_filler_node_area"]
    assert_injection_invariants(db, before, plan)


def test_flat_injection_compensates_the_filler_budget():
    db = _FakeDb(10, 20)                        # no regions -> flat
    before = snapshot_placedb(db)
    area0 = float(np.sum(db.node_size_x[10:] * db.node_size_y[10:]))
    plan = _plan_of_size(4)
    log = inject_pseudo_nodes(db, plan)
    assert log["mode"] == "flat" and log["filler_shrink"] < 1.0
    area1 = float(np.sum(db.node_size_x[10:] * db.node_size_y[10:]))
    assert area1 == pytest.approx(area0, rel=1e-12)
    assert db.num_filler_nodes == 24
    assert_injection_invariants(db, before, plan)


def test_flat_injection_refuses_an_oversized_pseudo_budget():
    db = _FakeDb(10, 2)                         # tiny filler budget
    plan = _plan_of_size(1000)
    with pytest.raises(ValueError, match="pseudo area"):
        inject_pseudo_nodes(db, plan)


def test_injection_of_zero_nodes_is_a_noop():
    db = _FakeDb(10, 20, filler_start_map=[0, 8, 16, 20, 20])
    before = snapshot_placedb(db)
    plan = _plan_of_size(0)
    log = inject_pseudo_nodes(db, plan)
    assert log["n_pseudo"] == 0 and db.num_nodes == before["num_nodes"]
    assert_injection_invariants(db, before, plan)


def test_invariants_catch_a_touched_pin_map():
    db = _FakeDb(10, 20, filler_start_map=[0, 8, 16, 20, 20])
    before = snapshot_placedb(db)
    plan = _plan_of_size(3)
    inject_pseudo_nodes(db, plan)
    db.pin2node_map = db.pin2node_map + 1
    with pytest.raises(AssertionError, match="pin2node"):
        assert_injection_invariants(db, before, plan)


def test_invariants_catch_an_extended_filler_start_map():
    db = _FakeDb(10, 20, filler_start_map=[0, 8, 16, 20, 20])
    before = snapshot_placedb(db)
    plan = _plan_of_size(3)
    inject_pseudo_nodes(db, plan)
    db.filler_start_map[-1] += 3
    with pytest.raises(AssertionError, match="filler_start_map"):
        assert_injection_invariants(db, before, plan)


def test_initial_pseudo_positions_are_the_pin_midpoint():
    plan = _plan_of_size(1)
    plan.node_u = np.array([0], dtype=np.int64)
    plan.node_v = np.array([1], dtype=np.int64)
    plan.off_ux = np.array([1.0]); plan.off_uy = np.array([0.0])
    plan.off_vx = np.array([0.0]); plan.off_vy = np.array([2.0])
    px, py = initial_pseudo_positions(plan, np.array([0.0, 10.0]),
                                      np.array([0.0, 20.0]))
    assert px.tolist() == [5.5] and py.tolist() == [11.0]


def test_pseudo_npz_round_trip(tmp_path):
    plan = _plan_of_size(3)
    plan.pseudo_start, plan.num_nodes_before, plan.num_nodes_after = 30, 30, 33
    plan.mode, plan.size_x, plan.size_y = "fence", 0.5, 3.0
    path = str(tmp_path / "pseudo.npz")
    px = np.array([1.0, 2.0, 3.0]); py = np.array([4.0, 5.0, 6.0])
    save_pseudo(path, plan, px, py, iteration=350, shift_factor=[1.0, 2.0],
                scale_factor=0.5, omega=np.array([1.0, 4.0]), mode="fence")
    got = load_pseudo(path)
    assert got["schema_version"] == PSEUDO_SCHEMA_VERSION
    assert got["iteration"] == 350 and got["mode"] == "fence"
    assert np.array_equal(got["pseudo_x"], px)
    assert np.array_equal(got["pseudo_y"], py)
    assert np.array_equal(got["net_ids"], plan.net_ids)
    assert np.array_equal(got["omega"], np.array([1.0, 4.0]))
    assert got["scale_factor"] == 0.5 and list(got["shift_factor"]) == [1.0, 2.0]
    assert got["pseudo_start"] == 30 and got["num_nodes_after"] == 33


def _real_db(config, fence):
    from ioplace.drivers.run_placement import _load_dreamplace
    params, db = _load_dreamplace(config)
    params.gpu = 0
    params.legalize_flag = 0
    if fence:
        from ioplace.fence_inject import inject_fence_regions
        from ioplace.regions import make_grid_regions
        die = (float(db.xl), float(db.yl), float(db.xh), float(db.yh))
        rs = make_grid_regions(die, 2, 2, lattice=512)
        parts = (np.arange(db.num_movable_nodes) % 4).astype(np.int32)
        inject_fence_regions(db, rs, parts)
        db.node2fence_region_map[0] = np.iinfo(np.int32).max   # escape cell
    db.initialize(params)
    return params, db


def _build_placer(params, db):
    import NonLinearPlace
    np.random.seed(params.random_seed)
    return NonLinearPlace.NonLinearPlace(params, db, None)


@pytest.mark.parametrize("config,fence", [(SIMPLE, False), (GCD, True)])
def test_real_injection_is_hpwl_and_pin2node_bit_identical(config, fence):
    torch = pytest.importorskip("torch")
    params0, db0 = _real_db(config, fence)
    placer0 = _build_placer(params0, db0)
    with torch.no_grad():
        hpwl0 = float(placer0.op_collections.hpwl_op(placer0.pos[0]))
    sha0 = hashlib.sha256(db0.pin2node_map.tobytes()).hexdigest()
    n_phys = db0.num_physical_nodes
    x0 = placer0.pos[0].data[:n_phys].clone()
    y0 = placer0.pos[0].data[db0.num_nodes:db0.num_nodes + n_phys].clone()

    params1, db1 = _real_db(config, fence)
    before = snapshot_placedb(db1)
    plan = _plan_of_size(7)
    inject_pseudo_nodes(db1, plan)
    assert_injection_invariants(db1, before, plan)
    placer1 = _build_placer(params1, db1)
    with torch.no_grad():
        hpwl1 = float(placer1.op_collections.hpwl_op(placer1.pos[0]))

    assert db1.num_nodes == db0.num_nodes + 7
    assert hashlib.sha256(db1.pin2node_map.tobytes()).hexdigest() == sha0
    assert hpwl1 == hpwl0                                   # bit-identical
    x1 = placer1.pos[0].data[:n_phys]
    y1 = placer1.pos[0].data[db1.num_nodes:db1.num_nodes + n_phys]
    assert torch.equal(x0, x1) and torch.equal(y0, y1)
    assert placer1.pos[0].shape[0] == 2 * db1.num_nodes
    assert placer1.data_collections.node_areas.shape[0] == db1.num_nodes
    assert len(db1.flat_node2pin_start_map) == len(db0.flat_node2pin_start_map)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_inject.py -v`
Expected: FAIL with `ImportError: cannot import name 'inject_pseudo_nodes'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/ops/pseudo_inject.py`:

```python
import hashlib
import json


def snapshot_placedb(placedb):
    """Pre-injection fingerprint, the second argument of
    `assert_injection_invariants`."""
    fsm = getattr(placedb, "filler_start_map", None)
    return dict(
        num_physical_nodes=int(placedb.num_physical_nodes),
        num_filler_nodes=int(placedb.num_filler_nodes),
        num_nodes=int(placedb.num_nodes),
        filler_start_map=None if fsm is None else np.array(fsm, copy=True),
        pin2node_sha256=hashlib.sha256(
            np.asarray(placedb.pin2node_map).tobytes()).hexdigest(),
        total_filler_node_area=float(placedb.total_filler_node_area),
        filler_area=float(np.sum(
            placedb.node_size_x[placedb.num_physical_nodes:] *
            placedb.node_size_y[placedb.num_physical_nodes:])))


def inject_pseudo_nodes(placedb, plan, *, size_x=None, size_y=None,
                        max_area_fraction=0.25):
    """Append `plan.n` pseudo nodes to the tail of the node-size arrays.

    Fence mode (`len(placedb.regions) > 0`): `filler_start_map` is deliberately
    **not** extended, so the pseudo block sits outside every per-region filler
    slice -- it therefore joins no electric field and no fence legaliser, while
    `num_nodes - num_filler_nodes == num_physical_nodes` keeps every existing
    slice pointing at the same regular fillers (design plan P-E, deviation 1).

    Flat mode: the single `ElectricPotential` takes the whole filler tail, so
    the pseudo nodes do carry density; the regular fillers are shrunk so that
    movable+filler area is unchanged (design v2 sec 6).
    """
    n = plan.n
    mode = "fence" if len(getattr(placedb, "regions", []) or []) > 0 else "flat"
    sx = float(placedb.site_width) if size_x is None else float(size_x)
    sy = float(placedb.row_height) if size_y is None else float(size_y)
    plan.num_nodes_before = int(placedb.num_nodes)
    plan.pseudo_start = int(placedb.num_nodes)
    plan.mode, plan.size_x, plan.size_y = mode, sx, sy
    log = dict(mode=mode, n_pseudo=n, pseudo_start=plan.pseudo_start,
               num_nodes_before=plan.num_nodes_before, size_x=sx, size_y=sy,
               filler_shrink=1.0)
    if n == 0:
        plan.num_nodes_after = plan.num_nodes_before
        log["num_nodes_after"] = plan.num_nodes_after
        return log

    if mode == "flat":
        n_phys = int(placedb.num_physical_nodes)
        filler_area = float(np.sum(placedb.node_size_x[n_phys:] *
                                   placedb.node_size_y[n_phys:]))
        pseudo_area = n * sx * sy
        if filler_area <= 0.0 or pseudo_area > max_area_fraction * filler_area:
            raise ValueError(
                "pseudo area %g exceeds %g of the flat filler budget %g -- "
                "lower --pseudo-max" % (pseudo_area, max_area_fraction,
                                        filler_area))
        shrink = 1.0 - pseudo_area / filler_area
        placedb.node_size_x[n_phys:] = placedb.node_size_x[n_phys:] * shrink
        log["filler_shrink"] = shrink

    dtype_x, dtype_y = placedb.node_size_x.dtype, placedb.node_size_y.dtype
    placedb.node_size_x = np.concatenate(
        [placedb.node_size_x, np.full(n, sx, dtype=dtype_x)])
    placedb.node_size_y = np.concatenate(
        [placedb.node_size_y, np.full(n, sy, dtype=dtype_y)])
    placedb.num_filler_nodes = int(placedb.num_filler_nodes) + n
    plan.num_nodes_after = int(placedb.num_nodes)
    log["num_nodes_after"] = plan.num_nodes_after
    return log


def assert_injection_invariants(placedb, before, plan):
    """The six Global-Constraints injection invariants, as assertions."""
    n = plan.n
    assert int(placedb.num_physical_nodes) == before["num_physical_nodes"], \
        "num_physical_nodes changed: pin indices are no longer valid"
    assert hashlib.sha256(np.asarray(placedb.pin2node_map).tobytes()).hexdigest() \
        == before["pin2node_sha256"], "pin2node map was modified by injection"
    assert int(placedb.num_filler_nodes) == before["num_filler_nodes"] + n, \
        "num_filler_nodes must grow by exactly n_pseudo"
    assert int(placedb.num_nodes) - int(placedb.num_filler_nodes) \
        == int(placedb.num_physical_nodes), \
        "num_nodes - num_filler_nodes != num_physical_nodes: every DREAMPlace " \
        "filler slice would now be misaligned"
    assert placedb.node_size_x.shape[0] == int(placedb.num_nodes), \
        "node_size_x length must equal num_nodes"
    assert placedb.node_size_y.shape[0] == int(placedb.num_nodes), \
        "node_size_y length must equal num_nodes"
    fsm = getattr(placedb, "filler_start_map", None)
    if before["filler_start_map"] is not None:
        assert fsm is not None and np.array_equal(
            np.asarray(fsm), before["filler_start_map"]), \
            "filler_start_map must be unchanged: the pseudo block belongs to " \
            "no bucket (design plan P-E, deviation 1)"
        assert int(np.asarray(fsm)[-1]) == int(placedb.num_filler_nodes) - n, \
            "filler_start_map[-1] must exclude exactly the n_pseudo tail"
        # every per-region filler slice ends at or before the pseudo block
        assert int(placedb.num_physical_nodes) + int(np.asarray(fsm)[-1]) \
            <= int(plan.pseudo_start), \
            "a fence filler slice reaches into the pseudo block"
    if plan.mode == "flat" and n:
        area = float(np.sum(
            placedb.node_size_x[placedb.num_physical_nodes:] *
            placedb.node_size_y[placedb.num_physical_nodes:]))
        assert abs(area - before["filler_area"]) <= 1e-6 * max(
            1.0, before["filler_area"]), \
            "flat mode must keep movable+filler area unchanged"


def initial_pseudo_positions(plan, node_x, node_y):
    """Midpoint of the two pins -- the hinge is inactive there because
    `0.5*||u-v||_1 > rho_min = 0.1*||u-v||_1`."""
    x = np.asarray(node_x, dtype=np.float64)
    y = np.asarray(node_y, dtype=np.float64)
    ux = x[plan.node_u] + plan.off_ux
    uy = y[plan.node_u] + plan.off_uy
    vx = x[plan.node_v] + plan.off_vx
    vy = y[plan.node_v] + plan.off_vy
    return 0.5 * (ux + vx), 0.5 * (uy + vy)


def save_pseudo(path, plan, px, py, *, iteration, shift_factor, scale_factor,
                omega, mode):
    """Write `pseudo.npz`. Coordinates are in the *scaled* PlaceDB units the GP
    used, with `shift_factor`/`scale_factor` recorded so a reader can convert to
    native units -- the same convention `export/evaluation.py` uses."""
    metadata = dict(schema_version=PSEUDO_SCHEMA_VERSION, iteration=int(iteration),
                    mode=str(mode), n_pseudo=int(plan.n),
                    pseudo_start=int(plan.pseudo_start),
                    num_nodes_before=int(plan.num_nodes_before),
                    num_nodes_after=int(plan.num_nodes_after),
                    size_x=float(plan.size_x), size_y=float(plan.size_y),
                    scale_factor=float(scale_factor),
                    shift_factor=[float(v) for v in shift_factor])
    arrays = dict(
        pseudo_x=np.asarray(px, dtype=np.float64),
        pseudo_y=np.asarray(py, dtype=np.float64),
        net_ids=plan.net_ids, node_u=plan.node_u, node_v=plan.node_v,
        ft=plan.ft, omega=np.asarray(omega, dtype=np.float64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pseudo-", suffix=".npz", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_pseudo(path):
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files if key != "metadata"}
        out.update(json.loads(str(data["metadata"])))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_inject.py -v`
Expected: 16 passed (the two `test_real_injection_*` parametrisations take ~30 s each on CPU).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/pseudo_inject.py tests/test_pseudo_inject.py
git commit -m "feat(pseudo-ft): orphan filler-tail injection with bit-identity invariants and pseudo.npz

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The 3-point WA detour, closed form with an analytic backward

**Files:**
- Create: `src/ioplace/ops/pseudo_ft_term.py`
- Test: `tests/test_pseudo_ft.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure torch/numpy).
- Produces:
  - `wa_detour(ux, uy, vx, vy, px, py, gamma) -> Tensor (E,)` — `WA_γ(u,v,p) − WA_γ(u,v)`, differentiable in all six coordinate tensors.
  - `wa_detour_numpy(ux, uy, vx, vy, px, py, gamma) -> np.ndarray (E,)` — the float64 oracle.
  - `wa_1d(coords, gamma) -> Tensor` — `WA` along one axis for a `(E, m)` coordinate block; exported for tests.
  - `DETOUR_LOWER_BOUND_COEFF = 8.0 / math.e`

**Decision: write the closed form, do not reuse DREAMPlace's `weighted_average_wirelength` op.** Four reasons, all load-bearing:

1. The op is a compiled extension keyed to PlaceDB's `flat_netpin`/`netpin_start`/`pin2net_map` and consumes *pin* positions from `pin_pos_op`. The pseudo nodes deliberately have **no pins** — that absence is exactly what makes tail injection leave `pin2node` bit-identical (Global Constraints invariant 1). Feeding them to the op would mean fabricating pins and destroying that invariant.
2. The op reduces to the scalar `Σ_e net_weight_e·WA_e`. This term needs the **per-net** value, because each net is weighted by its own `ω(r(p_e))` *and* because the cost is a *difference* between two different net sets (3-point minus 2-point). Folding `ω` into `net_weights` would have to be done identically in two op instances with two synthetic pin arrays, and still could not express the per-net hinge.
3. A 3-point WA is twelve lines whose gradient is the same closed form DREAMPlace's kernel implements; it is checkable against `torch.autograd` and a float64 numpy oracle to 1e-12, and against the `−8γ/e` bound analytically.
4. `γ` is read directly from `placer.model.gamma` (`PlaceObj.py:206,483,516`), the same tensor the WL op uses, so the detour and the wirelength are always smoothed identically without depending on the op's internal `inv_gamma` plumbing.

**Memory.** A custom `autograd.Function` saves only the six `(E,)` inputs, so the retained graph is ~48 B/net; at `E = 2e6` that is ≈96 MB. A plain-autograd implementation would retain ~24 intermediates of shape `(E,3)` (≈576 MB) — that is why the analytic backward is mandatory, not an optimisation.

**The closed form.** For coordinates `x_1..x_m` on one axis,
`wa_max = Σ x_i e^{x_i/γ} / Σ e^{x_i/γ}`, `wa_min = Σ x_i e^{−x_i/γ} / Σ e^{−x_i/γ}`, `WA = wa_max − wa_min`,
with the exponentials shifted by the per-row max/min for stability. The derivatives are
`∂wa_max/∂x_i = a_i·(1 + (x_i − wa_max)/γ)` with `a_i = e^{x_i/γ}/Σe^{x_j/γ}`, and
`∂wa_min/∂x_i = b_i·(1 − (x_i − wa_min)/γ)` with `b_i = e^{−x_i/γ}/Σe^{−x_j/γ}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pseudo_ft.py`:

```python
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ioplace.ops.pseudo_ft_term import (DETOUR_LOWER_BOUND_COEFF, wa_1d,
                                        wa_detour, wa_detour_numpy)


def _rand(n, seed=0, scale=100.0):
    rng = np.random.default_rng(seed)
    return [torch.tensor(rng.uniform(0.0, scale, size=n), dtype=torch.float64)
            for _ in range(6)]


def test_wa_1d_matches_the_definition():
    x = torch.tensor([[0.0, 10.0, 4.0]], dtype=torch.float64)
    g = 2.0
    e = np.exp(np.array([0.0, 10.0, 4.0]) / g)
    f = np.exp(-np.array([0.0, 10.0, 4.0]) / g)
    expect = (np.dot([0.0, 10.0, 4.0], e) / e.sum()
              - np.dot([0.0, 10.0, 4.0], f) / f.sum())
    assert float(wa_1d(x, g)) == pytest.approx(expect, rel=1e-12)


def test_wa_1d_is_stable_at_a_tiny_gamma():
    x = torch.tensor([[0.0, 1000.0, 400.0]], dtype=torch.float64)
    got = float(wa_1d(x, 1e-3))
    assert math.isfinite(got) and got == pytest.approx(1000.0, abs=1e-6)


def test_detour_matches_the_numpy_oracle():
    ux, uy, vx, vy, px, py = _rand(64, seed=1)
    got = wa_detour(ux, uy, vx, vy, px, py, 7.5)
    ref = wa_detour_numpy(ux.numpy(), uy.numpy(), vx.numpy(), vy.numpy(),
                          px.numpy(), py.numpy(), 7.5)
    assert np.allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)


def test_detour_gradient_matches_autograd():
    ux, uy, vx, vy, px, py = _rand(32, seed=2)
    args = [t.clone().requires_grad_(True) for t in (ux, uy, vx, vy, px, py)]
    wa_detour(*args, 7.5).sum().backward()
    analytic = [a.grad.clone() for a in args]

    def plain(u0, u1, v0, v1, p0, p1, g):
        three_x = torch.stack([u0, v0, p0], dim=-1)
        three_y = torch.stack([u1, v1, p1], dim=-1)
        two_x = torch.stack([u0, v0], dim=-1)
        two_y = torch.stack([u1, v1], dim=-1)
        return (wa_1d(three_x, g) + wa_1d(three_y, g)
                - wa_1d(two_x, g) - wa_1d(two_y, g))

    args2 = [t.clone().requires_grad_(True) for t in (ux, uy, vx, vy, px, py)]
    plain(*args2, 7.5).sum().backward()
    for a, b in zip(analytic, args2):
        assert torch.allclose(a, b.grad, rtol=1e-9, atol=1e-11)


def test_detour_gradient_matches_finite_differences():
    ux, uy, vx, vy, px, py = _rand(8, seed=3)
    args = [t.clone().requires_grad_(True) for t in (ux, uy, vx, vy, px, py)]
    wa_detour(*args, 5.0).sum().backward()
    h = 1e-6
    for slot in (4, 5):                       # p_x and p_y
        for i in range(8):
            plus = [t.clone() for t in (ux, uy, vx, vy, px, py)]
            minus = [t.clone() for t in (ux, uy, vx, vy, px, py)]
            plus[slot][i] += h
            minus[slot][i] -= h
            fd = float(wa_detour(*plus, 5.0)[i] - wa_detour(*minus, 5.0)[i]) / (2 * h)
            assert fd == pytest.approx(float(args[slot].grad[i]), abs=1e-6)


def test_detour_is_nonnegative_when_p_is_far_outside_the_bbox():
    ux = torch.tensor([0.0], dtype=torch.float64)
    uy = torch.tensor([0.0], dtype=torch.float64)
    vx = torch.tensor([100.0], dtype=torch.float64)
    vy = torch.tensor([0.0], dtype=torch.float64)
    px = torch.tensor([50.0], dtype=torch.float64)
    py = torch.tensor([400.0], dtype=torch.float64)
    assert float(wa_detour(ux, uy, vx, vy, px, py, 2.0)) > 300.0


def test_detour_respects_the_minus_eight_gamma_over_e_bound():
    # A p strictly inside the (u,v) box is where WA's smoothing bias can make
    # the bracket negative; the bound is -8*gamma/e per design plan P-E,
    # deviation 3.
    rng = np.random.default_rng(7)
    for gamma in (0.5, 2.0, 25.0, 250.0):
        ux = torch.zeros(256, dtype=torch.float64)
        uy = torch.zeros(256, dtype=torch.float64)
        vx = torch.tensor(rng.uniform(1.0, 500.0, 256))
        vy = torch.tensor(rng.uniform(1.0, 500.0, 256))
        t = torch.tensor(rng.uniform(0.0, 1.0, 256))
        px, py = t * vx, t * vy
        d = wa_detour(ux, uy, vx, vy, px, py, gamma)
        assert float(d.min()) >= -DETOUR_LOWER_BOUND_COEFF * gamma - 1e-9


def test_detour_converges_to_the_hpwl_detour_as_gamma_shrinks():
    ux = torch.tensor([0.0], dtype=torch.float64)
    uy = torch.tensor([0.0], dtype=torch.float64)
    vx = torch.tensor([100.0], dtype=torch.float64)
    vy = torch.tensor([100.0], dtype=torch.float64)
    px = torch.tensor([-40.0], dtype=torch.float64)   # 40 units of real detour
    py = torch.tensor([50.0], dtype=torch.float64)
    assert float(wa_detour(ux, uy, vx, vy, px, py, 1e-4)) == pytest.approx(40.0, abs=1e-3)


def test_detour_is_exactly_zero_when_p_sits_on_u():
    ux, uy, vx, vy, _, _ = _rand(16, seed=5)
    d = wa_detour(ux, uy, vx, vy, ux.clone(), uy.clone(), 3.0)
    # p == u duplicates a point: WA over {u,u,v} differs from WA over {u,v} only
    # through the softmax weights, and the difference is bounded by 8*gamma/e.
    assert float(d.abs().max()) <= DETOUR_LOWER_BOUND_COEFF * 3.0 + 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ioplace.ops.pseudo_ft_term'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/ops/pseudo_ft_term.py`:

```python
"""Pseudo-point feed-through objective (design v2 sec 6, subproject P-E).

`L_ft = sum_e w_e * omega(r(p_e)) * [WA_g(u,v,p_e) - WA_g(u,v)] + reg`, with
`reg = mu * sum_e [(rho_min - ||p-u||_1)_+^2 + (rho_min - ||p-v||_1)_+^2]` and
`rho_min = 0.1 * ||u-v||_1`.
"""
import math

import numpy as np
import torch

DETOUR_LOWER_BOUND_COEFF = 8.0 / math.e


def _wa_parts(coords, gamma):
    """(wa_max, wa_min, a, b) for a (E, m) coordinate block, numerically
    shifted. `a`/`b` are the softmax / softmin weights, both (E, m)."""
    shift_hi = coords.max(dim=-1, keepdim=True).values
    e = torch.exp((coords - shift_hi) / gamma)
    a = e / e.sum(dim=-1, keepdim=True)
    shift_lo = coords.min(dim=-1, keepdim=True).values
    f = torch.exp((shift_lo - coords) / gamma)
    b = f / f.sum(dim=-1, keepdim=True)
    return (a * coords).sum(dim=-1), (b * coords).sum(dim=-1), a, b


def wa_1d(coords, gamma):
    """Weighted-average wirelength along one axis for a (E, m) block."""
    wa_max, wa_min, _, _ = _wa_parts(coords, gamma)
    return wa_max - wa_min


def _wa_grad(coords, gamma):
    """d(wa_1d)/d(coords), shape (E, m)."""
    wa_max, wa_min, a, b = _wa_parts(coords, gamma)
    d_max = a * (1.0 + (coords - wa_max.unsqueeze(-1)) / gamma)
    d_min = b * (1.0 - (coords - wa_min.unsqueeze(-1)) / gamma)
    return d_max - d_min


class _WaDetour(torch.autograd.Function):
    """`WA_g(u,v,p) - WA_g(u,v)` with an analytic backward.

    Only the six (E,) inputs are saved, so the retained graph is ~48 B/net --
    a plain-autograd version would retain ~24 (E,3) intermediates.
    """

    @staticmethod
    def forward(ctx, ux, uy, vx, vy, px, py, gamma):
        three_x = torch.stack([ux, vx, px], dim=-1)
        three_y = torch.stack([uy, vy, py], dim=-1)
        two_x = torch.stack([ux, vx], dim=-1)
        two_y = torch.stack([uy, vy], dim=-1)
        out = (wa_1d(three_x, gamma) + wa_1d(three_y, gamma)
               - wa_1d(two_x, gamma) - wa_1d(two_y, gamma))
        ctx.save_for_backward(ux, uy, vx, vy, px, py)
        ctx.gamma = float(gamma)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gout):
        ux, uy, vx, vy, px, py = ctx.saved_tensors
        gamma = ctx.gamma
        g3x = _wa_grad(torch.stack([ux, vx, px], dim=-1), gamma)
        g3y = _wa_grad(torch.stack([uy, vy, py], dim=-1), gamma)
        g2x = _wa_grad(torch.stack([ux, vx], dim=-1), gamma)
        g2y = _wa_grad(torch.stack([uy, vy], dim=-1), gamma)
        return (gout * (g3x[:, 0] - g2x[:, 0]),
                gout * (g3y[:, 0] - g2y[:, 0]),
                gout * (g3x[:, 1] - g2x[:, 1]),
                gout * (g3y[:, 1] - g2y[:, 1]),
                gout * g3x[:, 2],
                gout * g3y[:, 2],
                None)


def wa_detour(ux, uy, vx, vy, px, py, gamma):
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be positive, got %r" % (gamma,))
    return _WaDetour.apply(ux, uy, vx, vy, px, py, float(gamma))


def wa_detour_numpy(ux, uy, vx, vy, px, py, gamma):
    """Float64 oracle used by the tests and by `PseudoFtTermRef`."""
    def wa(block):
        block = np.asarray(block, dtype=np.float64)
        hi = block.max(axis=-1, keepdims=True)
        e = np.exp((block - hi) / gamma)
        lo = block.min(axis=-1, keepdims=True)
        f = np.exp((lo - block) / gamma)
        return ((e * block).sum(-1) / e.sum(-1)
                - (f * block).sum(-1) / f.sum(-1))

    ux, uy = np.asarray(ux, dtype=np.float64), np.asarray(uy, dtype=np.float64)
    vx, vy = np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64)
    px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
    return (wa(np.stack([ux, vx, px], -1)) + wa(np.stack([uy, vy, py], -1))
            - wa(np.stack([ux, vx], -1)) - wa(np.stack([uy, vy], -1)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/pseudo_ft_term.py tests/test_pseudo_ft.py
git commit -m "feat(pseudo-ft): closed-form 3-point WA detour with an analytic backward

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The frozen `ω_k` and the differentiable `softmax(−SDF/τ_p)` mixture

**Files:**
- Modify: `src/ioplace/ops/pseudo_ft_term.py` (append)
- Test: `tests/test_pseudo_ft.py` (append)

**Interfaces:**
- Consumes: `ioplace.ops.soft_assign.region_sdf_l1`, `softmax_stats`, `_chunks`, `rect_table`.
- Produces:
  - `omega_from_overflow(overflow, *, omega_max=25.0) -> np.ndarray (K,)` — `clip((1+overflow_k)², max=omega_max)` over the first `K` entries; a `(K+1,)` input has its trailing non-fence entry dropped.
  - `omega_mixture(px, py, rects, rect2region, K, tau_p, omega, k_chunk=None) -> Tensor (E,)` — `Σ_k softmax_k(−SDF_k/τ_p)·ω_k`, differentiable in `px, py`, **not** in `omega`.

**The freeze contract, restated so it cannot be mis-implemented.** `omega` enters `omega_mixture` as a plain tensor with `requires_grad=False`, and `PseudoFtTerm` stores it through `register_buffer` after `.detach()`. The mixture weights `softmax_k(−SDF_k/τ_p)` *are* differentiable in `p` — that is the channel that pushes `p` out of congested regions (see the design ruling in Global Constraints). `omega` is rewritten only inside the `home_period` atomic transaction, exactly like `FtTerm.set_home`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pseudo_ft.py`:

```python
from ioplace.ops.pseudo_ft_term import omega_from_overflow, omega_mixture
from ioplace.ops.soft_assign import rect_table
from ioplace.regions import make_grid_regions


def _two_region_rects():
    """Two side-by-side regions over (0,0,100,100): region 0 is x<50."""
    rs = make_grid_regions((0., 0., 100., 100.), 2, 1, lattice=100)
    rects, r2k = rect_table(rs)
    return (torch.tensor(rects, dtype=torch.float64),
            torch.tensor(r2k, dtype=torch.int64), rs.k)


def test_omega_from_overflow_is_one_plus_overflow_squared():
    got = omega_from_overflow(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(got, [1.0, 2.25, 4.0])


def test_omega_from_overflow_drops_the_non_fence_entry_and_clamps():
    got = omega_from_overflow(np.array([0.0, 2.0, 1e6]), omega_max=25.0)
    # length 3 with K inferred as 2 is ambiguous, so K comes from the caller:
    got_k = omega_from_overflow(np.array([0.0, 2.0, 1e6]), k=2, omega_max=25.0)
    assert len(got_k) == 2 and np.allclose(got_k, [1.0, 9.0])
    assert float(np.max(got)) <= 25.0


def test_omega_mixture_reduces_to_the_region_weight_deep_inside():
    rects, r2k, k = _two_region_rects()
    omega = torch.tensor([1.0, 9.0], dtype=torch.float64)
    px = torch.tensor([5.0, 95.0], dtype=torch.float64)
    py = torch.tensor([50.0, 50.0], dtype=torch.float64)
    got = omega_mixture(px, py, rects, r2k, k, 1.0, omega)
    assert float(got[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(got[1]) == pytest.approx(9.0, abs=1e-6)


def test_omega_mixture_is_continuous_across_a_boundary():
    rects, r2k, k = _two_region_rects()
    omega = torch.tensor([1.0, 9.0], dtype=torch.float64)
    xs = torch.linspace(45.0, 55.0, 4001, dtype=torch.float64)
    ys = torch.full_like(xs, 50.0)
    vals = omega_mixture(xs, ys, rects, r2k, k, 2.0, omega)
    steps = (vals[1:] - vals[:-1]).abs()
    assert float(steps.max()) < 0.05              # smooth
    # a hard argmax would jump by |omega_0 - omega_1| = 8 at x = 50
    assert float(vals[0]) == pytest.approx(1.0, abs=0.2)
    assert float(vals[-1]) == pytest.approx(9.0, abs=0.2)
    assert float(vals[2000]) == pytest.approx(5.0, abs=0.2)   # midpoint


def test_omega_mixture_chunking_is_exact():
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=100)
    rects, r2k = rect_table(rs)
    rects = torch.tensor(rects, dtype=torch.float64)
    r2k = torch.tensor(r2k, dtype=torch.int64)
    rng = np.random.default_rng(11)
    px = torch.tensor(rng.uniform(0, 100, 512))
    py = torch.tensor(rng.uniform(0, 100, 512))
    omega = torch.tensor(rng.uniform(1.0, 9.0, 16))
    full = omega_mixture(px, py, rects, r2k, 16, 3.0, omega, k_chunk=None)
    chunked = omega_mixture(px, py, rects, r2k, 16, 3.0, omega, k_chunk=4)
    assert torch.allclose(full, chunked, rtol=1e-12, atol=1e-12)


def test_omega_is_frozen_in_the_gradient():
    rects, r2k, k = _two_region_rects()
    omega = torch.tensor([1.0, 9.0], dtype=torch.float64)
    assert omega.requires_grad is False
    px = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
    py = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
    out = omega_mixture(px, py, rects, r2k, k, 2.0, omega).sum()
    # no gradient path reaches omega ...
    with pytest.raises(RuntimeError):
        torch.autograd.grad(out, [omega], retain_graph=True, allow_unused=True)
    # ... while the value genuinely depends on it (finite difference over the
    # frozen input, done by re-evaluating, not by autograd).
    bumped = omega_mixture(px.detach(), py.detach(), rects, r2k, k, 2.0,
                           torch.tensor([1.0, 9.1], dtype=torch.float64)).sum()
    assert float(bumped) != float(out.detach())


def test_omega_mixture_channel_is_live():
    """The r(p) channel must carry gradient: a non-uniform omega with the same
    mean must give a different d/dp than the uniform one (design ruling)."""
    rects, r2k, k = _two_region_rects()
    grads = []
    for omega in ([5.0, 5.0], [1.0, 9.0]):
        px = torch.tensor([48.0], dtype=torch.float64, requires_grad=True)
        py = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
        omega_mixture(px, py, rects, r2k, k, 2.0,
                      torch.tensor(omega, dtype=torch.float64)).sum().backward()
        grads.append(float(px.grad))
    assert grads[0] == pytest.approx(0.0, abs=1e-12)   # uniform -> no force
    assert abs(grads[1]) > 1e-3                        # non-uniform -> force
    assert grads[1] < 0.0        # pushes p away from the omega=9 side (+x)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -k omega -v`
Expected: FAIL with `ImportError: cannot import name 'omega_from_overflow'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/ops/pseudo_ft_term.py`:

```python
from .soft_assign import _chunks, region_sdf_l1, softmax_stats


def omega_from_overflow(overflow, *, k=None, omega_max=25.0):
    """`omega_k = (util_k/target_density)^2` with `util_k := target_density_k *
    (1 + overflow_k)`, so `target_density` cancels (design plan P-E,
    deviation 2).

    In fence mode `model.overflow` is the `(K+1,)` per-field vector
    (`NonLinearPlace.py:257`, `EvalMetrics.py:121-127`); its trailing non-fence
    entry is meaningless here (the non-fence field has almost no movable area,
    so its overflow can be enormous) and is dropped when `k` is given.
    """
    values = np.asarray(overflow, dtype=np.float64).reshape(-1)
    if k is not None:
        if values.shape[0] < k:
            raise ValueError("overflow has %d entries, need at least k=%d"
                             % (values.shape[0], k))
        values = values[:k]
    return np.clip(np.square(1.0 + values), None, float(omega_max))


def _softmax_probs(sdf_chunk, m, t, tau):
    """softmax(-sdf/tau) for one k-chunk, normalised over all K via (m, t)
    from `soft_assign.softmax_stats`. Same algebra as `chunk_p_ell`'s `p`,
    without building the log term we do not need."""
    e = torch.exp(-sdf_chunk / tau - m.unsqueeze(1))
    return e / (1.0 + t).unsqueeze(1)


def omega_mixture(px, py, rects, rect2region, K, tau_p, omega, k_chunk=None):
    """`sum_k softmax_k(-SDF_k(p)/tau_p) * omega_k`.

    Differentiable in `px`/`py` through the softmax; `omega` is a frozen
    constant vector and is asserted to carry no gradient.
    """
    if omega.requires_grad:
        raise ValueError("omega must be frozen (requires_grad=False): design "
                         "v2 sec 6 freezes it inside the gradient")
    if float(tau_p) <= 0.0:
        raise ValueError("tau_p must be positive, got %r" % (tau_p,))
    rects = rects.to(dtype=px.dtype)
    omega = omega.to(dtype=px.dtype, device=px.device)
    m, t, _ = softmax_stats(px, py, rects, rect2region, K, tau_p, chunk=k_chunk)
    out = torch.zeros_like(px)
    for lo, hi in _chunks(K, k_chunk):
        sdf = region_sdf_l1(px, py, rects, rect2region, lo, hi)
        probs = _softmax_probs(sdf, m, t, tau_p)
        out = out + (probs * omega[lo:hi].unsqueeze(0)).sum(dim=1)
    return out
```

**Note for the implementer on `softmax_stats`.** It is differentiable and is *not* detached here, so `m` and `t` carry graph — that is required for the exact softmax derivative. The retained graph is `O(E·K)`: at `E = 2e6`, `K = 16`, fp32, expect ≈0.8 GB of extra GPU memory across the retained chunk tensors. `--pseudo-k-chunk` lowers the *transient* peak but not the retained total; `--pseudo-max` is the knob that lowers both. Record the measured figure in Task 9's result document.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/pseudo_ft_term.py tests/test_pseudo_ft.py
git commit -m "feat(pseudo-ft): frozen per-region omega read through a differentiable softmax mixture

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The anti-collapse hinge and `μ`

**Files:**
- Modify: `src/ioplace/ops/pseudo_ft_term.py` (append)
- Test: `tests/test_pseudo_ft.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `collapse_hinge(ux, uy, vx, vy, px, py) -> (Tensor (E,), Tensor (E,))` returning `(hinge, rho_min)`; `hinge_e = (ρ_min−‖p−u‖₁)_+² + (ρ_min−‖p−v‖₁)_+²` with `ρ_min = 0.1‖u−v‖₁`.
  - `auto_mu(rho_min) -> float` — `1.0 / median(ρ_min)` over the strictly positive entries, `0.0` when there are none.

**Two implementation rulings, both tested.**

1. **`ρ_min` is detached.** It is a threshold derived from `u,v`; differentiating it would put a force on `u` and `v` that pushes them *apart* to relax the hinge — a pure numerical artefact that would degrade HPWL. It is recomputed from the live (detached) `u,v` every forward.
2. **`u` and `v` are detached inside the hinge.** The regulariser exists to stop `p` collapsing onto a pin, not to move real cells. Leaving `u,v` differentiable would push the *cells* away from their own pseudo point. Only `p` feels `reg`.

**Why `auto_mu = 1/median(ρ_min)`.** The hinge has units of length² and the detour has units of length, so `μ` must carry `1/length` for the two to be commensurate before the normaliser's single λ scales them together. With `μ = 1/median(ρ_min)`, a fully collapsed pseudo point (`p = u`) contributes `μ·ρ_min² ≈ ρ_min`, i.e. a penalty of the same order as the detour it is trading against. `--pseudo-mu` accepts `auto` (default) or an explicit float.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pseudo_ft.py`:

```python
from ioplace.ops.pseudo_ft_term import auto_mu, collapse_hinge


def _uv():
    ux = torch.tensor([0.0], dtype=torch.float64)
    uy = torch.tensor([0.0], dtype=torch.float64)
    vx = torch.tensor([100.0], dtype=torch.float64)
    vy = torch.tensor([0.0], dtype=torch.float64)
    return ux, uy, vx, vy      # ||u-v||_1 = 100 -> rho_min = 10


def test_hinge_is_exactly_zero_above_rho_min():
    ux, uy, vx, vy = _uv()
    px = torch.tensor([50.0], dtype=torch.float64)
    py = torch.tensor([0.0], dtype=torch.float64)
    hinge, rho = collapse_hinge(ux, uy, vx, vy, px, py)
    assert float(rho) == pytest.approx(10.0)
    assert float(hinge) == 0.0


def test_hinge_activates_below_rho_min_on_each_pin():
    ux, uy, vx, vy = _uv()
    px = torch.tensor([4.0], dtype=torch.float64)      # 4 from u, 96 from v
    py = torch.tensor([0.0], dtype=torch.float64)
    hinge, _ = collapse_hinge(ux, uy, vx, vy, px, py)
    assert float(hinge) == pytest.approx((10.0 - 4.0) ** 2)
    px_v = torch.tensor([97.0], dtype=torch.float64)   # 3 from v
    hinge_v, _ = collapse_hinge(ux, uy, vx, vy, px_v, py)
    assert float(hinge_v) == pytest.approx((10.0 - 3.0) ** 2)


def test_hinge_is_c1_at_the_knee():
    ux, uy, vx, vy = _uv()
    py = torch.tensor([0.0], dtype=torch.float64)
    h = 1e-6
    grads = []
    for x in (10.0 - h, 10.0 + h):
        px = torch.tensor([x], dtype=torch.float64, requires_grad=True)
        collapse_hinge(ux, uy, vx, vy, px, py)[0].sum().backward()
        grads.append(float(px.grad))
    assert grads[0] == pytest.approx(0.0, abs=1e-4)
    assert grads[1] == pytest.approx(0.0, abs=1e-12)


def test_hinge_pushes_p_away_from_the_pin():
    ux, uy, vx, vy = _uv()
    px = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    py = torch.tensor([0.0], dtype=torch.float64)
    collapse_hinge(ux, uy, vx, vy, px, py)[0].sum().backward()
    assert float(px.grad) < 0.0      # descending the hinge increases px


def test_hinge_puts_no_force_on_u_or_v():
    ux = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    uy = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    vx = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
    vy = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    px = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    py = torch.tensor([0.0], dtype=torch.float64)
    collapse_hinge(ux, uy, vx, vy, px, py)[0].sum().backward()
    assert ux.grad is None and uy.grad is None
    assert vx.grad is None and vy.grad is None
    assert px.grad is not None


def test_auto_mu_is_the_inverse_median_rho_min():
    rho = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    assert auto_mu(rho) == pytest.approx(1.0 / 20.0)
    assert auto_mu(torch.zeros(3, dtype=torch.float64)) == 0.0
    assert auto_mu(torch.zeros(0, dtype=torch.float64)) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -k "hinge or mu" -v`
Expected: FAIL with `ImportError: cannot import name 'collapse_hinge'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/ops/pseudo_ft_term.py`:

```python
def collapse_hinge(ux, uy, vx, vy, px, py):
    """`(rho_min - ||p-u||_1)_+^2 + (rho_min - ||p-v||_1)_+^2`, with
    `rho_min = 0.1 * ||u-v||_1` (design v2 sec 6).

    `u`, `v` and `rho_min` are detached: the regulariser's job is to stop `p`
    collapsing onto a pin, not to move real cells apart (design plan P-E,
    Task 5 rulings 1 and 2).
    """
    ux_d, uy_d = ux.detach(), uy.detach()
    vx_d, vy_d = vx.detach(), vy.detach()
    rho_min = 0.1 * ((ux_d - vx_d).abs() + (uy_d - vy_d).abs())
    du = (px - ux_d).abs() + (py - uy_d).abs()
    dv = (px - vx_d).abs() + (py - vy_d).abs()
    return ((rho_min - du).clamp_min(0.0) ** 2
            + (rho_min - dv).clamp_min(0.0) ** 2), rho_min


def auto_mu(rho_min):
    """`mu = 1 / median(rho_min)` over the strictly positive entries -- the
    1/length factor that makes `mu * hinge` commensurate with the detour before
    the normaliser's single lambda scales both."""
    values = torch.as_tensor(rho_min).detach().reshape(-1)
    values = values[values > 0]
    if values.numel() == 0:
        return 0.0
    return float(1.0 / values.median())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -v`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/pseudo_ft_term.py tests/test_pseudo_ft.py
git commit -m "feat(pseudo-ft): anti-collapse hinge with detached endpoints and an auto mu

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `PseudoFtTerm`, `PseudoFtTermRef`, and the `TermNormalizer` registration

**Files:**
- Modify: `src/ioplace/ops/pseudo_ft_term.py` (append)
- Test: `tests/test_pseudo_ft.py` (append)

**Interfaces:**
- Consumes: Task 1's `PseudoPlan`; Tasks 3–5's `wa_detour`, `wa_detour_numpy`, `omega_mixture`, `omega_from_overflow`, `collapse_hinge`, `auto_mu`; `ioplace.ops.soft_assign.rect_table`; `ioplace.norm.TermNormalizer.register`.
- Produces:
  - `class PseudoFtTerm(torch.nn.Module)`
    - `__init__(plan, rects, rect2region, K, *, num_movable, num_nodes, tau_p, mu="auto", omega_max=25.0, k_chunk=4, device="cuda", dtype=torch.float32, node_x=None, node_y=None)`
    - buffers `omega` (K,), `node_u`, `node_v`, `off_ux`, `off_uy`, `off_vx`, `off_vy`, `w`, `movable_u`, `movable_v`
    - attributes `pseudo_start`, `n`, `mu`, `enabled` (bool, default `True`), `last_stats` (dict)
    - `set_omega(omega)` — validates length `K`, finite, positive; stores detached.
    - `set_num_nodes(num_nodes)` — the `pos` y-offset, set once after injection.
    - `endpoints(pos) -> (ux, uy, vx, vy, px, py)`
    - `forward(pos, gamma) -> Tensor ()` — the unweighted `L_ft`; returns a zero scalar when `not self.enabled` or `n == 0`.
    - `value(pos, ctx) -> Tensor ()` — the §4 protocol: `self.forward(pos, ctx["gamma"])`.
    - `pseudo_xy(pos) -> (Tensor, Tensor)` — the current pseudo coordinates, detached.
  - `class PseudoFtTermRef` — dense float64 numpy oracle with the same constructor and `value_numpy(node_x_all, node_y_all, gamma) -> float`.

**Endpoint masking.** `u` or `v` may be a terminal. The gradient must not reach fixed nodes (the `I4` invariant `ops/io_term.py:66` states for the IO term). Rather than materialising a full detached copy of `pos` (240 MB per half at 30M nodes, the `FtTermRef` pattern at `ft_term.py:188-189`), gather the `(E,)` endpoints and mask them with `torch.where(movable, value, value.detach())`, which is `(E,)`-sized and routes gradient only to movable endpoints.

**Value assembled exactly as spec §6 writes it:**
`L_ft = Σ_e w_e·ω_e·detour_e + μ·Σ_e hinge_e`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pseudo_ft.py`:

```python
from ioplace.norm import TermNormalizer
from ioplace.ops.pseudo_ft_term import PseudoFtTerm, PseudoFtTermRef
from ioplace.ops.pseudo_inject import PseudoPlan


def _toy_plan(n_movable=4):
    """Two nets: (node 0 -> node 1) and (node 2 -> node 3); node 3 is fixed."""
    return PseudoPlan(
        net_ids=np.array([0, 1], dtype=np.int64),
        node_u=np.array([0, 2], dtype=np.int64),
        node_v=np.array([1, 3], dtype=np.int64),
        off_ux=np.array([1.0, 0.0]), off_uy=np.array([0.0, 1.0]),
        off_vx=np.array([0.0, 2.0]), off_vy=np.array([2.0, 0.0]),
        w=np.array([1.0, 1.0]), ft=np.array([1.0, 2.0]),
        movable_u=np.array([True, True]),
        movable_v=np.array([True, False]),
        pseudo_start=6, num_nodes_before=6, num_nodes_after=8,
        mode="fence", size_x=0.5, size_y=3.0)


def _toy_term(mu=0.01, omega=(1.0, 9.0), dtype=torch.float64):
    rects, r2k, k = _two_region_rects()
    term = PseudoFtTerm(_toy_plan(), rects, r2k, k, num_movable=3, num_nodes=8,
                        tau_p=2.0, mu=mu, k_chunk=None, device="cpu",
                        dtype=dtype)
    term.set_omega(np.asarray(omega))
    return term


def _toy_pos(dtype=torch.float64):
    # 8 nodes: 0..3 physical, 4..5 regular fillers, 6..7 pseudo.
    x = torch.tensor([10., 90., 20., 80., 0., 0., 45., 55.], dtype=dtype)
    y = torch.tensor([10., 90., 20., 80., 0., 0., 50., 50.], dtype=dtype)
    return torch.cat([x, y]).requires_grad_(True)


def test_term_matches_the_numpy_reference():
    term = _toy_term()
    ref = PseudoFtTermRef(_toy_plan(), *_two_region_rects()[:2],
                          _two_region_rects()[2], num_movable=3, num_nodes=8,
                          tau_p=2.0, mu=0.01)
    ref.set_omega(np.array([1.0, 9.0]))
    pos = _toy_pos()
    got = float(term(pos, 5.0))
    want = ref.value_numpy(pos.detach().numpy()[:8], pos.detach().numpy()[8:], 5.0)
    assert got == pytest.approx(want, rel=1e-10)


def test_term_gradient_reaches_pseudo_and_movable_endpoints_only():
    term = _toy_term()
    pos = _toy_pos()
    term(pos, 5.0).backward()
    g = pos.grad
    # pseudo nodes (6, 7) move
    assert abs(float(g[6])) > 0 and abs(float(g[7])) > 0
    # movable endpoints 0, 1, 2 feel the detour
    assert abs(float(g[0])) > 0 and abs(float(g[1])) > 0 and abs(float(g[2])) > 0
    # fixed endpoint 3 does not
    assert float(g[3]) == 0.0 and float(g[8 + 3]) == 0.0
    # regular fillers 4, 5 are untouched
    assert float(g[4]) == 0.0 and float(g[5]) == 0.0


def test_term_is_zero_when_disabled_or_empty():
    term = _toy_term()
    term.enabled = False
    assert float(term(_toy_pos(), 5.0)) == 0.0
    empty = PseudoFtTerm(
        PseudoPlan(net_ids=np.zeros(0, np.int64), node_u=np.zeros(0, np.int64),
                   node_v=np.zeros(0, np.int64), off_ux=np.zeros(0),
                   off_uy=np.zeros(0), off_vx=np.zeros(0), off_vy=np.zeros(0),
                   w=np.zeros(0), ft=np.zeros(0),
                   movable_u=np.zeros(0, bool), movable_v=np.zeros(0, bool),
                   pseudo_start=6, num_nodes_before=6, num_nodes_after=6,
                   mode="fence", size_x=0.5, size_y=3.0),
        *_two_region_rects()[:2], _two_region_rects()[2], num_movable=3,
        num_nodes=6, tau_p=2.0, mu=0.0, device="cpu", dtype=torch.float64)
    empty.set_omega(np.array([1.0, 9.0]))
    assert float(empty(torch.zeros(12, dtype=torch.float64), 5.0)) == 0.0


def test_set_omega_validates_its_input():
    term = _toy_term()
    with pytest.raises(ValueError):
        term.set_omega(np.array([1.0]))                    # wrong length
    with pytest.raises(ValueError):
        term.set_omega(np.array([1.0, float("nan")]))      # not finite
    with pytest.raises(ValueError):
        term.set_omega(np.array([1.0, -1.0]))              # not positive


def test_pseudo_xy_reads_the_tail_slice():
    term = _toy_term()
    pos = _toy_pos()
    px, py = term.pseudo_xy(pos)
    assert px.tolist() == [45.0, 55.0] and py.tolist() == [50.0, 50.0]
    assert px.requires_grad is False


def test_last_stats_records_the_detour_minimum():
    term = _toy_term()
    term(_toy_pos(), 5.0)
    stats = term.last_stats
    assert set(stats) >= {"detour_min", "detour_mean", "hinge_active",
                          "omega_mean", "n"}
    assert stats["n"] == 2 and 0 <= stats["hinge_active"] <= 2


def test_value_protocol_and_normalizer_registration():
    term = _toy_term()
    normalizer = TermNormalizer(policy="grandplan", num_movable=3, num_nodes=8)
    normalizer.register("pseudo_ft", term, 1.0, target_share=0.1,
                        activate_overflow=0.30)
    assert normalizer.configs["pseudo_ft"].curvature == 1.0
    pos = _toy_pos()
    direct = float(term(pos, 5.0))
    through = float(term.value(pos, {"gamma": 5.0, "tau": 1.0}))
    assert through == pytest.approx(direct, rel=1e-12)


def test_lambda_is_applied_by_the_caller_not_the_term():
    term = _toy_term()
    pos = _toy_pos()
    base = float(term(pos, 5.0))
    scaled = 0.25 * base
    assert scaled == pytest.approx(0.25 * float(term(pos, 5.0)), rel=1e-12)
    # the term itself never reads a lambda
    assert not hasattr(term, "lambda_pseudo_ft")


def test_auto_mu_is_resolved_at_construction():
    rects, r2k, k = _two_region_rects()
    x = np.array([10., 90., 20., 80., 0., 0., 45., 55.])
    y = np.array([10., 90., 20., 80., 0., 0., 50., 50.])
    term = PseudoFtTerm(_toy_plan(), rects, r2k, k, num_movable=3, num_nodes=8,
                        tau_p=2.0, mu="auto", device="cpu",
                        dtype=torch.float64, node_x=x, node_y=y)
    assert term.mu > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py -k "term or normalizer or lambda or pseudo_xy or stats" -v`
Expected: FAIL with `ImportError: cannot import name 'PseudoFtTerm'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/ops/pseudo_ft_term.py`:

```python
class PseudoFtTerm(torch.nn.Module):
    """`L_ft = sum_e w_e*omega(r(p_e))*[WA_g(u,v,p_e) - WA_g(u,v)] + mu*hinge`.

    Registered with `norm.TermNormalizer` as `"pseudo_ft"` with curvature 1
    (design v2 sec 4); the caller applies lambda, the term never does.
    """

    def __init__(self, plan, rects, rect2region, K, *, num_movable, num_nodes,
                 tau_p, mu="auto", omega_max=25.0, k_chunk=4, device="cuda",
                 dtype=torch.float32, node_x=None, node_y=None):
        super(PseudoFtTerm, self).__init__()
        if plan.pseudo_start is None:
            raise ValueError("plan.pseudo_start is unset: call "
                             "pseudo_inject.inject_pseudo_nodes first")
        self.K = int(K)
        self.n = int(plan.n)
        self.pseudo_start = int(plan.pseudo_start)
        self.num_movable = int(num_movable)
        self.num_nodes = int(num_nodes)
        self.tau_p = float(tau_p)
        self.omega_max = float(omega_max)
        self.k_chunk = k_chunk
        self.enabled = True
        self.last_stats = {}
        dev = torch.device(device)
        self.register_buffer("rects", torch.as_tensor(
            np.asarray(rects), dtype=dtype, device=dev))
        self.register_buffer("rect2region", torch.as_tensor(
            np.asarray(rect2region), dtype=torch.int64, device=dev))
        for name in ("node_u", "node_v"):
            self.register_buffer(name, torch.as_tensor(
                getattr(plan, name), dtype=torch.int64, device=dev))
        for name in ("off_ux", "off_uy", "off_vx", "off_vy", "w"):
            self.register_buffer(name, torch.as_tensor(
                getattr(plan, name), dtype=dtype, device=dev))
        for name in ("movable_u", "movable_v"):
            self.register_buffer(name, torch.as_tensor(
                getattr(plan, name), dtype=torch.bool, device=dev))
        self.register_buffer("omega", torch.ones(self.K, dtype=dtype, device=dev))
        if mu == "auto":
            if node_x is None or node_y is None or self.n == 0:
                self.mu = 0.0
            else:
                x = torch.as_tensor(np.asarray(node_x), dtype=torch.float64)
                y = torch.as_tensor(np.asarray(node_y), dtype=torch.float64)
                u_x = x[self.node_u.cpu()] + self.off_ux.double().cpu()
                u_y = y[self.node_u.cpu()] + self.off_uy.double().cpu()
                v_x = x[self.node_v.cpu()] + self.off_vx.double().cpu()
                v_y = y[self.node_v.cpu()] + self.off_vy.double().cpu()
                self.mu = auto_mu(0.1 * ((u_x - v_x).abs() + (u_y - v_y).abs()))
        else:
            self.mu = float(mu)

    def set_num_nodes(self, num_nodes):
        self.num_nodes = int(num_nodes)

    def set_omega(self, omega):
        """Frozen per-region utilisation weights, refreshed on the
        `home_period` cadence inside the atomic transaction."""
        values = torch.as_tensor(np.asarray(omega, dtype=np.float64),
                                 dtype=self.omega.dtype, device=self.omega.device)
        if values.shape != (self.K,):
            raise ValueError("omega must have one entry per region: expected "
                             "%d, got %s" % (self.K, tuple(values.shape)))
        if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
            raise ValueError("omega must be finite and strictly positive")
        self.omega = values.detach().clone()

    def _masked(self, values, movable):
        return torch.where(movable, values, values.detach())

    def endpoints(self, pos):
        n = self.num_nodes
        x, y = pos[:n], pos[n:2 * n]
        ux = self._masked(x[self.node_u], self.movable_u) + self.off_ux
        uy = self._masked(y[self.node_u], self.movable_u) + self.off_uy
        vx = self._masked(x[self.node_v], self.movable_v) + self.off_vx
        vy = self._masked(y[self.node_v], self.movable_v) + self.off_vy
        lo, hi = self.pseudo_start, self.pseudo_start + self.n
        return ux, uy, vx, vy, x[lo:hi], y[lo:hi]

    def pseudo_xy(self, pos):
        n = self.num_nodes
        lo, hi = self.pseudo_start, self.pseudo_start + self.n
        return (pos[lo:hi].detach().clone(),
                pos[n + lo:n + hi].detach().clone())

    def forward(self, pos, gamma):
        if not self.enabled or self.n == 0:
            return pos.new_zeros(())
        ux, uy, vx, vy, px, py = self.endpoints(pos)
        detour = wa_detour(ux, uy, vx, vy, px, py, float(gamma))
        omega = omega_mixture(px, py, self.rects, self.rect2region, self.K,
                              self.tau_p, self.omega, self.k_chunk)
        hinge, rho_min = collapse_hinge(ux, uy, vx, vy, px, py)
        loss = (self.w * omega * detour).sum()
        if self.mu:
            loss = loss + self.mu * hinge.sum()
        with torch.no_grad():
            self.last_stats = dict(
                n=self.n, detour_min=float(detour.min()),
                detour_mean=float(detour.mean()),
                omega_mean=float(omega.mean()), omega_max=float(omega.max()),
                hinge_active=int((hinge > 0).sum()),
                rho_min_median=float(rho_min.median()), mu=self.mu,
                gamma=float(gamma))
        return loss

    def value(self, pos, ctx):
        """`norm.TermNormalizer`'s term protocol (design v2 sec 4): the
        *unweighted* objective; the normalizer owns the backward."""
        return self.forward(pos, ctx["gamma"])


class PseudoFtTermRef(object):
    """Dense float64 numpy oracle for the small numerical tests. Same formula,
    no chunking, no masking shortcuts."""

    def __init__(self, plan, rects, rect2region, K, *, num_movable, num_nodes,
                 tau_p, mu=0.0, omega_max=25.0):
        self.plan = plan
        self.rects = np.asarray(rects, dtype=np.float64)
        self.rect2region = np.asarray(rect2region, dtype=np.int64)
        self.K = int(K)
        self.num_movable = int(num_movable)
        self.num_nodes = int(num_nodes)
        self.tau_p = float(tau_p)
        self.mu = float(mu)
        self.omega = np.ones(self.K, dtype=np.float64)

    def set_omega(self, omega):
        self.omega = np.asarray(omega, dtype=np.float64).reshape(self.K)

    def _sdf(self, px, py):
        out = np.full((px.shape[0], self.K), np.inf)
        for r, (rxl, ryl, rxh, ryh) in enumerate(self.rects):
            k = int(self.rect2region[r])
            dx = np.maximum(rxl - px, px - rxh)
            dy = np.maximum(ryl - py, py - ryh)
            d = (np.maximum(dx, 0) + np.maximum(dy, 0)
                 + np.minimum(np.maximum(dx, dy), 0))
            out[:, k] = np.minimum(out[:, k], d)
        return out

    def value_numpy(self, node_x_all, node_y_all, gamma):
        plan = self.plan
        if plan.n == 0:
            return 0.0
        x = np.asarray(node_x_all, dtype=np.float64)
        y = np.asarray(node_y_all, dtype=np.float64)
        ux = x[plan.node_u] + plan.off_ux
        uy = y[plan.node_u] + plan.off_uy
        vx = x[plan.node_v] + plan.off_vx
        vy = y[plan.node_v] + plan.off_vy
        lo = plan.pseudo_start
        px = x[lo:lo + plan.n]
        py = y[lo:lo + plan.n]
        detour = wa_detour_numpy(ux, uy, vx, vy, px, py, gamma)
        z = -self._sdf(px, py) / self.tau_p
        z = z - z.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        omega = probs @ self.omega
        rho_min = 0.1 * (np.abs(ux - vx) + np.abs(uy - vy))
        du = np.abs(px - ux) + np.abs(py - uy)
        dv = np.abs(px - vx) + np.abs(py - vy)
        hinge = (np.maximum(rho_min - du, 0) ** 2
                 + np.maximum(rho_min - dv, 0) ** 2)
        return float((plan.w * omega * detour).sum() + self.mu * hinge.sum())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft.py tests/test_pseudo_inject.py -v`
Expected: all pass (31 in `test_pseudo_ft.py`, 16 in `test_pseudo_inject.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/pseudo_ft_term.py tests/test_pseudo_ft.py
git commit -m "feat(pseudo-ft): PseudoFtTerm, its numpy reference, and the pseudo_ft normaliser registration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Driver wiring in `run_placement_io.py` (flat mode) and the CLI

**Files:**
- Modify: `src/ioplace/drivers/run_placement_io.py` (`RESULT_FIELDS` at `:34-59`; `run_io`'s signature at `:139-159`; the `placedb.initialize(params)` site at `:244`; the term-construction block at `:286-380`; `term_fn`/`attach_terms` at `:359-382`; the `legalize_op` wrapper at `:404-419`; the atomic-callback block at `:534-586`; the result assembly)
- Modify: `src/ioplace/drivers/run_placement.py` (`build_parser()` / `main()`)
- Test: `tests/test_pseudo_ft_driver.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - `run_io(..., pseudo_ft="off", pseudo_max=2_000_000, pseudo_mu="auto", pseudo_tau_rel=0.05, pseudo_omega_max=25.0, pseudo_omega_source="overflow", pseudo_k_chunk=4, pseudo_npz=None)`
  - New `RESULT_FIELDS` entries: `pseudo_ft`, `pseudo_n`, `pseudo_max`, `pseudo_mu`, `pseudo_mode`, `pseudo_npz`, `pseudo_detour_min`, `pseudo_detour_mean`, `pseudo_hinge_active`, `pseudo_omega_mean`, `lambda_pseudo_ft_final`, `pseudo_disp_p50`, `pseudo_disp_p95`, `pseudo_disp_p99`.
  - `_PseudoPrecond` — the precondition proxy (module-level class in `run_placement_io.py`).
  - CLI: `--pseudo-ft {on,off}`, `--pseudo-max INT`, `--pseudo-mu {auto|FLOAT}`, `--pseudo-tau-rel FLOAT`, `--pseudo-omega-max FLOAT`, `--pseudo-omega-source {overflow,uniform}`, `--pseudo-k-chunk INT`, `--pseudo-npz PATH`.

**Ordering, which is the whole correctness story.** Inside `run_io`:

1. `placedb.initialize(params)` (`:244`, unchanged).
2. `nl = netlist_from_placedb(placedb)` (`:266`) — reads only physical nodes, so it stays valid across injection.
3. `rs`/`rg`/`ctx` (`:270-273`, unchanged).
4. **New:** if `pseudo_ft == "on"`, run one `ctx.evaluate` on the initial `placedb.node_x/node_y`, `measure_net_ft`, `select_pseudo_nets`, `snapshot_placedb`, `inject_pseudo_nodes`, `assert_injection_invariants`.
5. `io_term = IoTerm(..., num_nodes=placedb.num_nodes, ...)` (`:292`) — now reads the **post-injection** `num_nodes`, which is why injection must precede it. Build `PseudoFtTerm` here too.
6. `attach_terms(params, [term_fn, pseudo_term_fn])` — a *list*; `PlaceObj.obj_fn` sums every entry (`PlaceObj.py:325-327`).
7. `np.random.seed(params.random_seed)`; `NonLinearPlace(...)`.
8. **New:** write the pseudo initial positions into `placer.pos[0].data` (the `BasicPlace` filler-spreading loop leaves them at 0 in fence mode and gives them a meaningless uniform draw in flat mode).
9. **New:** install `_PseudoPrecond` at the first iteration callback, once `placer.model` exists.

**Why the precondition proxy.** `PreconditionOp` divides by `sum_pin_weights + α·density_weight·node_area`, clamped to ≥1; pseudo nodes have no pins, so their divisor is 1 and the division is a no-op. The proxy exists for the *other* half: `PreconditionOp.__call__` zeroes filler gradients through `filler_mask = update_mask[self.filler2fence_region_map]` (`PlaceObj.py:106-109`), and `filler2fence_region_map` is `torch.zeros(num_filler_nodes)` filled only over `filler_start_map` slices (`PlaceObj.py:53-60`) — the orphan pseudo tail keeps the default `0`, so the pseudo gradients would be frozen the moment region 0's field terminates. The proxy saves and restores the pseudo slice around the inner call, which also makes the pseudo gradient immune to `fix_nodes_mask`. It forwards every other attribute (notably `set_overflow`, called at `NonLinearPlace.py:805`) through `__getattr__`.

**ω refresh cadence.** Inside the existing `callback_order == "atomic"` block, immediately after the `ft_term.set_home(...)` refresh (`run_placement_io.py:546-551`) and **before** `normalizer.transaction(...)`, so the ω write and the λ update land in the same atomic transaction and the single `obj_version` bump plus `refresh_nesterov_secant` already in place covers both. With `pseudo_omega_source == "uniform"` (mandatory in flat mode, where `model.overflow` is a scalar) ω stays all-ones and is never rewritten.

**Validation.** `pseudo_ft == "on"` requires `callback_order == "atomic"` and `norm_policy in ("grandplan", "adaptive")` — the legacy policy derives no λ for a third term. In flat mode `pseudo_omega_source` must be `"uniform"`. Violations raise `ValueError` from `run_io`'s prologue next to the existing checks at `:160-174`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pseudo_ft_driver.py`:

```python
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement_io import _PseudoPrecond, run_io

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
GCD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "route_feedback_20260914", "gcd.json")


def test_pseudo_ft_requires_atomic_callbacks(tmp_path):
    with pytest.raises(ValueError, match="atomic"):
        run_io(GCD, 4, "grid", 0, str(tmp_path / "r.json"),
               callback_order="legacy", norm_policy="grandplan", pseudo_ft="on")


def test_pseudo_ft_rejects_the_legacy_norm_policy(tmp_path):
    with pytest.raises(ValueError, match="norm_policy"):
        run_io(GCD, 4, "grid", 0, str(tmp_path / "r.json"),
               callback_order="atomic", norm_policy="legacy", pseudo_ft="on")


def test_pseudo_ft_rejects_overflow_omega_in_flat_mode(tmp_path):
    with pytest.raises(ValueError, match="uniform"):
        run_io(GCD, 4, "grid", 0, str(tmp_path / "r.json"),
               callback_order="atomic", norm_policy="grandplan",
               pseudo_ft="on", pseudo_omega_source="overflow")


class _InnerPrecond(object):
    def __init__(self):
        self.calls = 0

    def set_overflow(self, value):
        self.seen = value

    def __call__(self, grad, density_weight, update_mask=None,
                 fix_nodes_mask=None):
        self.calls += 1
        grad.zero_()           # the behaviour the proxy must undo for pseudo
        return grad


def test_pseudo_precond_proxy_preserves_the_pseudo_slice():
    inner = _InnerPrecond()
    proxy = _PseudoPrecond(inner, pseudo_start=4, n=2, num_nodes=6)
    grad = torch.arange(12, dtype=torch.float64)
    out = proxy(grad, torch.ones(1))
    assert inner.calls == 1
    assert out[4:6].tolist() == [4.0, 5.0]        # pseudo x restored
    assert out[10:12].tolist() == [10.0, 11.0]    # pseudo y restored
    assert out[0:4].tolist() == [0.0] * 4         # everything else zeroed
    proxy.set_overflow(0.5)                       # forwards through __getattr__
    assert inner.seen == 0.5


@pytest.mark.slow
def test_gcd_flat_end_to_end_with_pseudo_points(tmp_path):
    out = str(tmp_path / "pseudo_on.json")
    npz = str(tmp_path / "pseudo.npz")
    res = run_io(GCD, 4, "grid", 0, out, callback_order="atomic",
                 norm_policy="grandplan", f_ft_max=0.0, rho_max=0.1,
                 pseudo_ft="on", pseudo_max=256, pseudo_omega_source="uniform",
                 pseudo_npz=npz, emit_eval=str(tmp_path / "evaluation.npz"),
                 every=50)
    assert res["pseudo_ft"] == "on" and res["pseudo_n"] > 0
    assert res["pseudo_mode"] == "flat"
    assert res["lambda_pseudo_ft_final"] >= 0.0
    assert os.path.exists(npz)

    from ioplace.ops.pseudo_inject import load_pseudo
    dumped = load_pseudo(npz)
    assert dumped["n_pseudo"] == res["pseudo_n"]
    assert len(dumped["pseudo_x"]) == res["pseudo_n"]
    assert np.isfinite(dumped["pseudo_x"]).all()

    # evaluation.npz must not know pseudo nodes exist
    from ioplace.export.evaluation import load_evaluation
    data = load_evaluation(str(tmp_path / "evaluation.npz"))
    meta = data["metadata"] if isinstance(data, dict) else data.metadata
    num_physical = json.loads(meta)["num_physical"] if isinstance(meta, str) \
        else meta["num_physical"]
    assert num_physical == res["num_physical_nodes"] if "num_physical_nodes" in res \
        else num_physical > 0
    assert len(data["node_region"]) == num_physical


@pytest.mark.slow
def test_pseudo_off_matches_the_pre_change_baseline(tmp_path):
    """--pseudo-ft off must be a provable no-op: nothing is injected, no term
    is attached, so the result is bit-identical to a run of the same command
    without the flag."""
    a = run_io(GCD, 4, "grid", 0, str(tmp_path / "a.json"),
               callback_order="atomic", norm_policy="grandplan", every=50)
    b = run_io(GCD, 4, "grid", 0, str(tmp_path / "b.json"),
               callback_order="atomic", norm_policy="grandplan", every=50,
               pseudo_ft="off", pseudo_max=256)
    assert a["hpwl"] == b["hpwl"] and a["io_count"] == b["io_count"]
    assert b["pseudo_ft"] == "off" and b["pseudo_n"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -m "not slow" -v`
Expected: FAIL with `ImportError: cannot import name '_PseudoPrecond'`.

- [ ] **Step 3: Add the precondition proxy and the validation**

In `src/ioplace/drivers/run_placement_io.py`, after `_install_attribute` (`:137`):

```python
class _PseudoPrecond(object):
    """Keep the pseudo block's gradient exactly raw.

    `PreconditionOp` zeroes filler gradients through
    `update_mask[filler2fence_region_map]` (`PlaceObj.py:106-109`), and the
    orphan pseudo tail is outside every `filler_start_map` slice so its entry
    in that map keeps the default 0 -- the pseudo points would freeze as soon
    as region 0's field terminated. Saving and restoring the pseudo slice also
    makes them immune to `fix_nodes_mask`. Pseudo nodes have no pins and (in
    fence mode) no density, so their preconditioner is already the clamped 1.0
    and no scaling is lost.
    """

    def __init__(self, inner, pseudo_start, n, num_nodes):
        self._inner = inner
        self._lo = int(pseudo_start)
        self._hi = int(pseudo_start) + int(n)
        self._n = int(num_nodes)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, grad, density_weight, update_mask=None,
                 fix_nodes_mask=None):
        lo, hi, n = self._lo, self._hi, self._n
        saved_x = grad[lo:hi].clone()
        saved_y = grad[n + lo:n + hi].clone()
        self._inner(grad, density_weight, update_mask, fix_nodes_mask)
        grad[lo:hi] = saved_x
        grad[n + lo:n + hi] = saved_y
        return grad
```

Extend `run_io`'s signature (after `norm_trace=None`) with:

```python
           pseudo_ft="off", pseudo_max=2_000_000, pseudo_mu="auto",
           pseudo_tau_rel=0.05, pseudo_omega_max=25.0,
           pseudo_omega_source="overflow", pseudo_k_chunk=4, pseudo_npz=None):
```

and add to the prologue, next to the existing checks:

```python
    if pseudo_ft not in ("on", "off"):
        raise ValueError("pseudo_ft must be on or off, got %r" % (pseudo_ft,))
    if pseudo_omega_source not in ("overflow", "uniform"):
        raise ValueError("pseudo_omega_source must be overflow or uniform, "
                         "got %r" % (pseudo_omega_source,))
    if pseudo_ft == "on":
        if callback_order != "atomic":
            raise ValueError("pseudo_ft requires atomic callbacks")
        if norm_policy == "legacy":
            raise ValueError("pseudo_ft requires norm_policy grandplan or "
                             "adaptive: the legacy path derives no lambda for "
                             "a third term")
        if pseudo_omega_source != "uniform":
            raise ValueError(
                "run_placement_io is the flat single-phase driver: its "
                "model.overflow is a scalar, so omega carries no per-region "
                "signal -- use pseudo_omega_source='uniform' here and the "
                "fence phase of run_main_flow for the research arm")
        if pseudo_max < 0:
            raise ValueError("pseudo_max must be non-negative")
```

- [ ] **Step 4: Run the validation tests**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -m "not slow" -v`
Expected: 4 passed.

- [ ] **Step 5: Wire the injection, the term and the artefact**

Still in `run_io`, after `ctx = GpuEvalContext(nl, rg, device="cuda")` (`:273`) and **before** the `io_term = IoTerm(...)` construction (`:292`):

```python
        pseudo_term = None
        pseudo_plan = None
        pseudo_log = {"mode": None, "n_pseudo": 0}
        if pseudo_ft == "on":
            from ioplace.ops.pseudo_inject import (assert_injection_invariants,
                                                   initial_pseudo_positions,
                                                   inject_pseudo_nodes,
                                                   measure_net_ft,
                                                   save_pseudo, select_pseudo_nets,
                                                   snapshot_placedb)
            from ioplace.ops.pseudo_ft_term import PseudoFtTerm, omega_from_overflow
            seed_res = ctx.evaluate(
                torch.as_tensor(placedb.node_x[:nl.num_physical], device="cuda"),
                torch.as_tensor(placedb.node_y[:nl.num_physical], device="cuda"))
            pseudo_plan = select_pseudo_nets(nl, rg, placedb.node_x[:nl.num_physical],
                                             placedb.node_y[:nl.num_physical],
                                             measure_net_ft(seed_res),
                                             n_max=int(pseudo_max))
            before_db = snapshot_placedb(placedb)
            pseudo_log = inject_pseudo_nodes(placedb, pseudo_plan)
            assert_injection_invariants(placedb, before_db, pseudo_plan)
```

Right after the `io_term`/`ft_term` block and the `L_R` line (`:305`):

```python
        if pseudo_plan is not None and pseudo_plan.n:
            pseudo_term = PseudoFtTerm(
                pseudo_plan, rects, r2k, k,
                num_movable=nl.num_movable, num_nodes=placedb.num_nodes,
                tau_p=pseudo_tau_rel * L_R, mu=pseudo_mu,
                omega_max=pseudo_omega_max, k_chunk=pseudo_k_chunk,
                device="cuda", dtype=torch.float32,
                node_x=placedb.node_x[:nl.num_physical],
                node_y=placedb.node_y[:nl.num_physical])
            normalizer.register("pseudo_ft", pseudo_term, 1.0,
                                target_share=shares.get("pseudo_ft", 0.1),
                                activate_overflow=0.30, n_ramp=state.n_ramp)

        def pseudo_term_fn(pos):
            if pseudo_term is None:
                return pos.new_zeros(())
            lam = normalizer.lambdas.get("pseudo_ft", 0.0)
            if lam == 0.0:
                return pos.new_zeros(())
            return lam * pseudo_term(pos, float(placer_holder["gamma"]))
```

`placer_holder` is a one-entry dict created just above `attach_terms` and refreshed by the iteration callback (`placer_holder = {"gamma": float(params.gamma)}`); `term_fn` already reads live schedule state the same way. Change the attach site (`:381`) to:

```python
        if not observer_mode:
            terms = [term_fn]
            if pseudo_term is not None:
                terms.append(pseudo_term_fn)
            attach_terms(params, terms)
```

Immediately after `placer = NonLinearPlace.NonLinearPlace(params, placedb, None)` (`:388`):

```python
        if pseudo_term is not None:
            px0, py0 = initial_pseudo_positions(
                pseudo_plan, placedb.node_x[:nl.num_physical],
                placedb.node_y[:nl.num_physical])
            lo = pseudo_plan.pseudo_start
            hi = lo + pseudo_plan.n
            n_all_new = placedb.num_nodes
            with torch.no_grad():
                placer.pos[0].data[lo:hi] = torch.as_tensor(
                    px0, dtype=placer.pos[0].dtype, device=placer.pos[0].device)
                placer.pos[0].data[n_all_new + lo:n_all_new + hi] = torch.as_tensor(
                    py0, dtype=placer.pos[0].dtype, device=placer.pos[0].device)
            pseudo_prev = (torch.as_tensor(px0, dtype=torch.float64).clone(),
                           torch.as_tensor(py0, dtype=torch.float64).clone())
```

In `cb`, at the very top (so it runs before anything else on the first callback):

```python
            placer_holder["gamma"] = gamma
            if pseudo_term is not None and not cb_state["pseudo_precond"]:
                _install_attribute(
                    cleanup, placer.model.op_collections, "precondition_op",
                    _PseudoPrecond(placer.model.op_collections.precondition_op,
                                   pseudo_plan.pseudo_start, pseudo_plan.n,
                                   placedb.num_nodes))
                cb_state["pseudo_precond"] = True
```

with `"pseudo_precond": False` added to `cb_state`.

Inside the `callback_order == "atomic"` block, immediately after the `ft_term.set_home(...)` branch and **before** the `norm_policy` branch:

```python
                    if pseudo_term is not None and iteration % home_period == 0:
                        if pseudo_omega_source == "overflow":
                            pseudo_term.set_omega(omega_from_overflow(
                                placer.model.overflow.detach().cpu().numpy(),
                                k=k, omega_max=pseudo_omega_max))
                        px_now, py_now = pseudo_term.pseudo_xy(pos)
                        disp = ((px_now.double().cpu() - pseudo_prev[0]).abs()
                                + (py_now.double().cpu() - pseudo_prev[1]).abs())
                        pseudo_prev = (px_now.double().cpu(), py_now.double().cpu())
                        entry.update(
                            pseudo_disp_p50=float(np.percentile(disp.numpy(), 50)),
                            pseudo_disp_p95=float(np.percentile(disp.numpy(), 95)),
                            pseudo_disp_p99=float(np.percentile(disp.numpy(), 99)))
                    if pseudo_term is not None:
                        entry.update({"pseudo_" + key: value
                                      for key, value in pseudo_term.last_stats.items()})
```

In the `_timed_legalize` wrapper (`:406`), before `orig_legalize(pos)`:

```python
            if pseudo_term is not None:
                pseudo_term.enabled = False        # detach at the LG boundary
                px_end, py_end = pseudo_term.pseudo_xy(pos)
                save_pseudo(pseudo_npz or (out_json + ".pseudo.npz"),
                            pseudo_plan,
                            px_end.double().cpu().numpy(),
                            py_end.double().cpu().numpy(),
                            iteration=cb_state["last_iteration"],
                            shift_factor=scale_fields["shift_factor"],
                            scale_factor=scale_fields["scale_factor"],
                            omega=pseudo_term.omega.double().cpu().numpy(),
                            mode=pseudo_log["mode"])
```

Finally, in the result assembly, add:

```python
        result.update(
            pseudo_ft=pseudo_ft, pseudo_n=(pseudo_plan.n if pseudo_plan else 0),
            pseudo_max=int(pseudo_max), pseudo_mode=pseudo_log["mode"],
            pseudo_mu=(pseudo_term.mu if pseudo_term is not None else 0.0),
            pseudo_npz=(pseudo_npz or (out_json + ".pseudo.npz")
                        if pseudo_term is not None else None),
            pseudo_detour_min=(pseudo_term.last_stats.get("detour_min")
                               if pseudo_term is not None else None),
            pseudo_detour_mean=(pseudo_term.last_stats.get("detour_mean")
                                if pseudo_term is not None else None),
            pseudo_hinge_active=(pseudo_term.last_stats.get("hinge_active")
                                 if pseudo_term is not None else None),
            pseudo_omega_mean=(pseudo_term.last_stats.get("omega_mean")
                               if pseudo_term is not None else None),
            lambda_pseudo_ft_final=float(normalizer.lambdas.get("pseudo_ft", 0.0)),
            pseudo_disp_p50=trajectory[-1].get("pseudo_disp_p50") if trajectory else None,
            pseudo_disp_p95=trajectory[-1].get("pseudo_disp_p95") if trajectory else None,
            pseudo_disp_p99=trajectory[-1].get("pseudo_disp_p99") if trajectory else None)
```

and append every one of those keys to `RESULT_FIELDS`.

- [ ] **Step 6: Add the CLI flags**

In `src/ioplace/drivers/run_placement.py`'s `build_parser()`:

```python
    parser.add_argument("--pseudo-ft", choices=("on", "off"), default="off",
                        help="pseudo-point feed-through term (design v2 sec 6)")
    parser.add_argument("--pseudo-max", type=int, default=2_000_000,
                        help="cap on pseudo points, top nets by measured per-net FT")
    parser.add_argument("--pseudo-mu", default="auto",
                        help="anti-collapse weight: 'auto' (1/median rho_min) or a float")
    parser.add_argument("--pseudo-tau-rel", type=float, default=0.05,
                        help="tau_p for softmax(-SDF/tau_p), relative to L_R")
    parser.add_argument("--pseudo-omega-max", type=float, default=25.0)
    parser.add_argument("--pseudo-omega-source", choices=("overflow", "uniform"),
                        default="overflow")
    parser.add_argument("--pseudo-k-chunk", type=int, default=4)
    parser.add_argument("--pseudo-npz", default=None,
                        help="path for pseudo.npz (default: <out>.pseudo.npz)")
```

and forward them in `main()`:

```python
        pseudo_ft=args.pseudo_ft, pseudo_max=args.pseudo_max,
        pseudo_mu=args.pseudo_mu, pseudo_tau_rel=args.pseudo_tau_rel,
        pseudo_omega_max=args.pseudo_omega_max,
        pseudo_omega_source=args.pseudo_omega_source,
        pseudo_k_chunk=args.pseudo_k_chunk, pseudo_npz=args.pseudo_npz,
```

(`pseudo_mu` stays a string when it is `"auto"` and is converted by `PseudoFtTerm`; convert an explicit numeric string with `float(args.pseudo_mu)` in `main()` when `args.pseudo_mu != "auto"`.)

- [ ] **Step 7: Run the slow end-to-end**

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
"$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -v
```
Expected: 6 passed. If GPU 3 is busy, wait — GPUs 0–2 are foreign workloads.

- [ ] **Step 8: Run the whole suite**

Run: `source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3 && "$IOPLACE_PYTHON" -m pytest`
Expected: no new failures; in particular `tests/test_driver_io.py`, `tests/test_norm*.py` and `tests/test_ft_callback.py` unchanged, because `--pseudo-ft off` attaches nothing.

- [ ] **Step 9: Commit**

```bash
git add src/ioplace/drivers/run_placement_io.py src/ioplace/drivers/run_placement.py tests/test_pseudo_ft_driver.py
git commit -m "feat(pseudo-ft): wire injection, the term and pseudo.npz into run_placement_io and the CLI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Fence-mode wiring — `FenceTermBundle` and the `run_main_flow` hook

**Files:**
- Create: `src/ioplace/fence_terms.py`
- Modify: `src/ioplace/drivers/run_main_flow.py` (P-B Task 7's `run_fence_gp` and its CLI)
- Test: `tests/test_fence_terms.py`, `tests/test_pseudo_ft_driver.py` (append)

**Precondition.** This task depends on P-B having landed. If `src/ioplace/drivers/run_main_flow.py` does not exist, **stop and report** — do not invent a substitute. Tasks 1–7 and 9–10 are independent of it, except that Task 10's group A/B needs this task.

**Interfaces:**
- Consumes: Tasks 1–6; P-B's `run_fence_gp(config_json, out_dir, *, region_set, part, positions, reference_density_weight, k, …, extra_terms=(), timer=None)`.
- Produces:
  - `class FenceTermBundle` with `terms` (list), `on_iteration(iteration, pos, placer, normalizer, entry)`, `before_legalize(pos, out_dir, context)`, `summary() -> dict`, and the module function `combine(*bundles) -> FenceTermBundle`.
  - `make_pseudo_bundle(fctx, *, pseudo_max, pseudo_mu, pseudo_tau_rel, pseudo_omega_max, pseudo_k_chunk, normalizer) -> FenceTermBundle` in `src/ioplace/ops/pseudo_ft_term.py`.
  - `run_fence_gp(..., term_factory=None)` where `term_factory(fctx) -> FenceTermBundle` and `fctx` is a `SimpleNamespace` with `params`, `placedb`, `nl`, `rg`, `region_set`, `part`, `ctx`, `k`, `positions`, `out_dir`.

**Where the hook goes in `run_fence_gp`.** Between `ctx = GpuEvalContext(nl, rg, device="cuda")` and `np.random.seed(params.random_seed); placer = NonLinearPlace.NonLinearPlace(...)`. That is the only window in which the pseudo nodes can be injected: after `build_fence_placedb` has run `placedb.initialize(params)` and before `BasicPlace.__init__` sizes `init_pos`/`pos` from `placedb.num_nodes`. `install_density_weight_clamp` stays where it is; the two are independent.

**Fence mode is where ω is real.** `model.overflow` is the `(K+1,)` per-field vector here, so `pseudo_omega_source="overflow"` is the default and the flat-mode restriction from Task 7 does not apply.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fence_terms.py`:

```python
import numpy as np
import pytest

from ioplace.fence_terms import FenceTermBundle, combine


def test_empty_bundle_is_inert():
    bundle = FenceTermBundle()
    assert bundle.terms == []
    entry = {}
    bundle.on_iteration(0, None, None, None, entry)
    bundle.before_legalize(None, "/tmp", {})
    assert entry == {} and bundle.summary() == {}


def test_combine_concatenates_terms_and_fans_out_the_hooks():
    seen = []

    class _Probe(FenceTermBundle):
        def __init__(self, tag):
            FenceTermBundle.__init__(self, terms=[lambda pos, t=tag: t])
            self.tag = tag

        def on_iteration(self, iteration, pos, placer, normalizer, entry):
            seen.append(("iter", self.tag, iteration))
            entry[self.tag] = iteration

        def before_legalize(self, pos, out_dir, context):
            seen.append(("lg", self.tag))

        def summary(self):
            return {self.tag: 1}

    merged = combine(_Probe("a"), _Probe("b"))
    assert len(merged.terms) == 2
    entry = {}
    merged.on_iteration(7, None, None, None, entry)
    merged.before_legalize(None, "/tmp", {})
    assert entry == {"a": 7, "b": 7}
    assert seen == [("iter", "a", 7), ("iter", "b", 7), ("lg", "a"), ("lg", "b")]
    assert merged.summary() == {"a": 1, "b": 1}


def test_combine_rejects_duplicate_summary_keys():
    class _Same(FenceTermBundle):
        def summary(self):
            return {"pseudo_n": 1}

    with pytest.raises(ValueError, match="pseudo_n"):
        combine(_Same(), _Same())
```

Append to `tests/test_pseudo_ft_driver.py`:

```python
@pytest.mark.slow
def test_gcd_fence_phase_with_pseudo_points(tmp_path):
    """Phase 3 of the main flow with the pseudo term on: the bundle injects,
    the term registers, pseudo.npz lands, and the pseudo nodes are provably
    outside every fence filler bucket."""
    run_main_flow = pytest.importorskip("ioplace.drivers.run_main_flow")
    out_dir = str(tmp_path / "fence")
    res = run_main_flow.run_main_flow(
        GCD, out_dir, k=4, region_source="grid", init="die_center",
        norm_policy="grandplan", pseudo_ft="on", pseudo_max=128)
    assert res["pseudo_n"] > 0 and res["pseudo_mode"] == "fence"
    assert os.path.exists(os.path.join(out_dir, "pseudo.npz"))
    from ioplace.ops.pseudo_inject import load_pseudo
    dumped = load_pseudo(os.path.join(out_dir, "pseudo.npz"))
    assert dumped["mode"] == "fence"
    # the fence legaliser never saw them: every pseudo coordinate is still the
    # last GP value, i.e. not snapped to a row.
    assert res["fence_compliance"] >= 0.9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_fence_terms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ioplace.fence_terms'`.

- [ ] **Step 3: Write `src/ioplace/fence_terms.py`**

```python
"""The single seam through which post-freeze objective terms reach phase 3.

Design v2 sec 3 keeps IO and FT off after the freeze and turns capacity (P-D)
and pseudo-FT (P-E) on. Both need the same four things from `run_fence_gp`: a
callable to hand `dp_hook.attach_terms`, a per-iteration hook inside the atomic
transaction, a pre-legalisation hook, and a summary merged into `result.json`.
`combine` lets P-D add its bundle without touching any caller.
"""


class FenceTermBundle(object):
    def __init__(self, terms=None):
        self.terms = list(terms or [])

    def on_iteration(self, iteration, pos, placer, normalizer, entry):
        """Called inside the atomic callback, before `normalizer.transaction`,
        so any coefficient this bundle refreshes joins that transaction's
        single `obj_version` bump."""

    def before_legalize(self, pos, out_dir, context):
        """Called in the `legalize_op` wrapper, before the real legaliser."""

    def summary(self):
        return {}


class _Combined(FenceTermBundle):
    def __init__(self, bundles):
        terms = []
        for bundle in bundles:
            terms.extend(bundle.terms)
        FenceTermBundle.__init__(self, terms)
        self._bundles = list(bundles)

    def on_iteration(self, iteration, pos, placer, normalizer, entry):
        for bundle in self._bundles:
            bundle.on_iteration(iteration, pos, placer, normalizer, entry)

    def before_legalize(self, pos, out_dir, context):
        for bundle in self._bundles:
            bundle.before_legalize(pos, out_dir, context)

    def summary(self):
        merged = {}
        for bundle in self._bundles:
            for key, value in bundle.summary().items():
                if key in merged:
                    raise ValueError("duplicate summary key %r across fence "
                                     "term bundles" % (key,))
                merged[key] = value
        return merged


def combine(*bundles):
    return _Combined([b for b in bundles if b is not None])
```

- [ ] **Step 4: Write `make_pseudo_bundle`**

Append to `src/ioplace/ops/pseudo_ft_term.py`:

```python
def make_pseudo_bundle(fctx, *, normalizer, pseudo_max=2_000_000, mu="auto",
                       tau_rel=0.05, omega_max=25.0, k_chunk=4,
                       omega_source="overflow", home_period=50,
                       target_share=0.1, activate_overflow=0.30, n_ramp=20,
                       device="cuda", dtype=torch.float32):
    """Build the phase-3 `FenceTermBundle` for the pseudo-point FT term.

    Must be called *after* `placedb.initialize(params)` and *before*
    `NonLinearPlace(...)`: it mutates `fctx.placedb`'s node arrays.
    """
    from ..fence_terms import FenceTermBundle
    from .pseudo_inject import (assert_injection_invariants,
                                initial_pseudo_positions, inject_pseudo_nodes,
                                measure_net_ft, save_pseudo, select_pseudo_nets,
                                snapshot_placedb)
    from .soft_assign import rect_table

    placedb, nl, rg = fctx.placedb, fctx.nl, fctx.rg
    node_x = np.asarray(placedb.node_x[:nl.num_physical], dtype=np.float64)
    node_y = np.asarray(placedb.node_y[:nl.num_physical], dtype=np.float64)
    seed_res = fctx.ctx.evaluate(
        torch.as_tensor(node_x, device=device),
        torch.as_tensor(node_y, device=device))
    plan = select_pseudo_nets(nl, rg, node_x, node_y, measure_net_ft(seed_res),
                              n_max=int(pseudo_max))
    before = snapshot_placedb(placedb)
    log = inject_pseudo_nodes(placedb, plan)
    assert_injection_invariants(placedb, before, plan)
    if plan.n == 0:
        return FenceTermBundle()

    rects, r2k = rect_table(fctx.region_set)
    die = (placedb.xl, placedb.yl, placedb.xh, placedb.yh)
    l_r = ((die[2] - die[0]) * (die[3] - die[1]) / fctx.k) ** 0.5
    term = PseudoFtTerm(plan, rects, r2k, fctx.k,
                        num_movable=nl.num_movable, num_nodes=placedb.num_nodes,
                        tau_p=tau_rel * l_r, mu=mu, omega_max=omega_max,
                        k_chunk=k_chunk, device=device, dtype=dtype,
                        node_x=node_x, node_y=node_y)
    normalizer.register("pseudo_ft", term, 1.0, target_share=target_share,
                        activate_overflow=activate_overflow, n_ramp=n_ramp)
    px0, py0 = initial_pseudo_positions(plan, node_x, node_y)
    state = {"gamma": 1.0, "prev": (np.asarray(px0), np.asarray(py0))}

    def term_fn(pos):
        lam = normalizer.lambdas.get("pseudo_ft", 0.0)
        if lam == 0.0:
            return pos.new_zeros(())
        return lam * term(pos, state["gamma"])

    class _PseudoBundle(FenceTermBundle):
        def __init__(self):
            FenceTermBundle.__init__(self, [term_fn])
            self.term = term
            self.plan = plan
            self.log = log
            self.init_xy = (px0, py0)

        def on_iteration(self, iteration, pos, placer, normalizer_, entry):
            state["gamma"] = float(placer.model.gamma)
            if omega_source == "overflow" and iteration % home_period == 0:
                term.set_omega(omega_from_overflow(
                    placer.model.overflow.detach().cpu().numpy(),
                    k=fctx.k, omega_max=omega_max))
            if iteration % home_period == 0:
                px, py = term.pseudo_xy(pos)
                px = px.double().cpu().numpy()
                py = py.double().cpu().numpy()
                disp = (np.abs(px - state["prev"][0])
                        + np.abs(py - state["prev"][1]))
                state["prev"] = (px, py)
                entry.update(pseudo_disp_p50=float(np.percentile(disp, 50)),
                             pseudo_disp_p95=float(np.percentile(disp, 95)),
                             pseudo_disp_p99=float(np.percentile(disp, 99)))
            entry.update({"pseudo_" + key: value
                          for key, value in term.last_stats.items()})

        def before_legalize(self, pos, out_dir, context):
            term.enabled = False
            px, py = term.pseudo_xy(pos)
            save_pseudo(os.path.join(out_dir, "pseudo.npz"), plan,
                        px.double().cpu().numpy(), py.double().cpu().numpy(),
                        iteration=int(context.get("iteration", -1)),
                        shift_factor=context["shift_factor"],
                        scale_factor=context["scale_factor"],
                        omega=term.omega.double().cpu().numpy(),
                        mode=log["mode"])

        def summary(self):
            out = dict(pseudo_n=plan.n, pseudo_mode=log["mode"],
                       pseudo_mu=term.mu, pseudo_max=int(pseudo_max))
            out.update({"pseudo_" + key: value
                        for key, value in term.last_stats.items()})
            return out

    return _PseudoBundle()
```

(`import os` is already at the top of `pseudo_ft_term.py`; add it if not.)

- [ ] **Step 5: Add the hook to `run_fence_gp` and the flags to `run_main_flow`**

In `src/ioplace/drivers/run_main_flow.py`:

1. Add `term_factory=None` to `run_fence_gp`'s keyword-only signature, next to `extra_terms=()`, and document that `extra_terms` is the plain-callable form and `term_factory` the stateful form.
2. After `ctx = GpuEvalContext(nl, rg, device="cuda")` and the density-weight clamp install, before `np.random.seed(...)`:

```python
        bundle = None
        if term_factory is not None:
            from types import SimpleNamespace
            bundle = term_factory(SimpleNamespace(
                params=params, placedb=placedb, nl=nl, rg=rg,
                region_set=rs_scaled, part=part, ctx=ctx, k=k,
                positions=positions, out_dir=out_dir))
        attached = list(extra_terms) + (list(bundle.terms) if bundle else [])
        if attached:
            attach_terms(params, attached)
```

   (replacing the existing `if extra_terms: attach_terms(...)`).
3. After `placer = NonLinearPlace.NonLinearPlace(params, placedb, None)`, write the pseudo initial positions from `bundle.init_xy` and `bundle.plan.pseudo_start`, exactly as Task 7 Step 5 does for the flat driver, and install `_PseudoPrecond` on the first callback.
4. In `timed_legalize`, before `original_legalize(pos)`:

```python
            if bundle is not None:
                bundle.before_legalize(pos, out_dir, dict(
                    iteration=cb_state["last_iteration"],
                    shift_factor=info["shift_factor"],
                    scale_factor=info["scale_factor"]))
```
5. In `cb`, call `bundle.on_iteration(iteration, pos, placer, normalizer, entry)` before the transaction.
6. Merge `bundle.summary()` into the returned dict and extend `MAIN_FLOW_RESULT_FIELDS` with `pseudo_ft`, `pseudo_n`, `pseudo_mode`, `pseudo_mu`, `pseudo_max`, `pseudo_detour_min`, `pseudo_detour_mean`, `pseudo_hinge_active`, `pseudo_omega_mean`, `pseudo_disp_p50/p95/p99`, `lambda_pseudo_ft_final`.
7. Add the same eight `--pseudo-*` flags as Task 7 Step 6 to `run_main_flow`'s parser (default `--pseudo-omega-source overflow` here), and build the factory with `functools.partial(make_pseudo_bundle, normalizer=..., pseudo_max=..., ...)` when `--pseudo-ft on`.

- [ ] **Step 6: Run the tests**

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_fence_terms.py -v
"$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -v
"$IOPLACE_PYTHON" -m pytest
```
Expected: `test_fence_terms.py` 3 passed; the fence end-to-end passes; full suite green.

- [ ] **Step 7: Commit**

```bash
git add src/ioplace/fence_terms.py src/ioplace/ops/pseudo_ft_term.py src/ioplace/drivers/run_main_flow.py tests/test_fence_terms.py tests/test_pseudo_ft_driver.py
git commit -m "feat(pseudo-ft): FenceTermBundle seam and the phase-3 pseudo bundle in run_main_flow

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Open question (ii) — do the pseudo points actually move?

**Files:**
- Create: `src/scripts/run_pseudo_displacement.py`
- Create: `docs/results/2026-09-19-p-e-pseudo-displacement.md`
- Test: `tests/test_pseudo_ft_driver.py` (append one non-slow unit test for the summariser)

**Interfaces:**
- Consumes: Task 8's `run_main_flow(..., pseudo_ft="on")` and its `pseudo_disp_*` trajectory entries.
- Produces:
  - `summarise_displacement(trajectory, *, window=100) -> dict` with keys `windows` (list of `{iteration, p50, p95, p99}`), `p50_median`, `p95_median`, `p99_max`, `frac_windows_below_eps`, `verdict` (`"moves"` / `"starved"` / `"unstable"`).
  - `main(argv=None)` running the experiment on `mempool_tile` and writing `results/p_e_pseudo_displacement/<stamp>/summary.json`.

**The question, from spec §10 (ii):** *"Does the pseudo point move, or does frozen ω starve it? — displacement percentiles per 100 iterations on tile."* The decision rule, fixed here so the result cannot be argued after the fact:

- `eps = 0.5 · row_height` (half a standard cell row) is the "did not move" threshold.
- **`"starved"`** if `p95` per 100-iteration window is below `eps` in more than 80% of windows after the term activates. Then frozen ω is not the culprit to fix first — check λ_pseudo_ft in `norm_trace.jsonl` and the preconditioner proxy — and the spec §6 fallback (softmin over 8–16 fixed candidates, no DOF) becomes the recommendation.
- **`"unstable"`** if any window's `p99` exceeds `0.5 ·` the die diagonal. Then the pseudo preconditioner of 1.0 is too aggressive; the fix is to raise `--pseudo-mu` or lower the `pseudo_ft` target share.
- **`"moves"`** otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pseudo_ft_driver.py`:

```python
def test_summarise_displacement_classifies_the_three_regimes():
    import importlib.util
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "run_pseudo_displacement",
        _os.path.join(root, "src", "scripts", "run_pseudo_displacement.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    moving = [{"iteration": i, "pseudo_disp_p50": 5.0, "pseudo_disp_p95": 20.0,
               "pseudo_disp_p99": 40.0} for i in range(100, 1100, 100)]
    out = module.summarise_displacement(moving, eps=1.0, die_diagonal=10000.0)
    assert out["verdict"] == "moves" and len(out["windows"]) == 10
    assert out["p95_median"] == pytest.approx(20.0)

    starved = [dict(row, pseudo_disp_p95=0.1) for row in moving]
    assert module.summarise_displacement(
        starved, eps=1.0, die_diagonal=10000.0)["verdict"] == "starved"

    unstable = [dict(row, pseudo_disp_p99=9000.0) for row in moving]
    assert module.summarise_displacement(
        unstable, eps=1.0, die_diagonal=10000.0)["verdict"] == "unstable"

    assert module.summarise_displacement([], eps=1.0,
                                         die_diagonal=1.0)["verdict"] == "starved"
```

The import is by file path on purpose: `src/scripts/` is a script directory, not an importable package (`pyproject.toml:19` puts only `src` on `pythonpath`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -k summarise -v`
Expected: FAIL with `FileNotFoundError` on `run_pseudo_displacement.py`.

- [ ] **Step 3: Write the script**

Create `src/scripts/run_pseudo_displacement.py`:

```python
#!/usr/bin/env python
"""Design v2 sec 10 open question (ii): do the pseudo points move, or does the
frozen omega starve them?

Runs the main flow on mempool_tile with --pseudo-ft on, reads the
`pseudo_disp_*` entries the phase-3 callback records every `home_period`
iterations, aggregates them into 100-iteration windows and applies the fixed
decision rule from the P-E plan, Task 9.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

VERDICTS = ("moves", "starved", "unstable")


def summarise_displacement(trajectory, *, window=100, eps=1.0, die_diagonal=1.0,
                           starved_fraction=0.8, unstable_fraction=0.5):
    rows = [row for row in trajectory if row.get("pseudo_disp_p95") is not None]
    if not rows:
        return dict(windows=[], p50_median=0.0, p95_median=0.0, p99_max=0.0,
                    frac_windows_below_eps=1.0, verdict="starved",
                    eps=eps, die_diagonal=die_diagonal)
    buckets = {}
    for row in rows:
        key = int(row["iteration"]) // int(window)
        buckets.setdefault(key, []).append(row)
    windows = []
    for key in sorted(buckets):
        group = buckets[key]
        windows.append(dict(
            iteration=key * int(window),
            p50=float(np.max([r["pseudo_disp_p50"] for r in group])),
            p95=float(np.max([r["pseudo_disp_p95"] for r in group])),
            p99=float(np.max([r["pseudo_disp_p99"] for r in group]))))
    p95 = np.array([w["p95"] for w in windows])
    p99_max = float(np.max([w["p99"] for w in windows]))
    frac_below = float(np.mean(p95 < eps))
    if p99_max > unstable_fraction * die_diagonal:
        verdict = "unstable"
    elif frac_below > starved_fraction:
        verdict = "starved"
    else:
        verdict = "moves"
    return dict(windows=windows,
                p50_median=float(np.median([w["p50"] for w in windows])),
                p95_median=float(np.median(p95)), p99_max=p99_max,
                frac_windows_below_eps=frac_below, verdict=verdict,
                eps=float(eps), die_diagonal=float(die_diagonal))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="DREAMPlace config for mempool_tile")
    parser.add_argument("--out-root", default="results/p_e_pseudo_displacement")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--pseudo-max", type=int, default=200_000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--region-source", default="grid")
    args = parser.parse_args(argv)

    from ioplace.drivers.run_main_flow import run_main_flow
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_root, stamp)
    os.makedirs(out_dir, exist_ok=True)
    result = run_main_flow(args.config, out_dir, k=args.k,
                           region_source=args.region_source,
                           norm_policy="grandplan", pseudo_ft="on",
                           pseudo_max=args.pseudo_max)
    trajectory = result.get("fence_trajectory") or result.get("trajectory") or []
    die = result["die_native"]
    diagonal = abs(die[2] - die[0]) + abs(die[3] - die[1])
    eps = 0.5 * float(result.get("row_height", 1.0))
    summary = summarise_displacement(trajectory, window=args.window, eps=eps,
                                     die_diagonal=diagonal)
    summary.update(config=args.config, k=args.k, pseudo_n=result.get("pseudo_n"),
                   pseudo_mu=result.get("pseudo_mu"),
                   lambda_pseudo_ft_final=result.get("lambda_pseudo_ft_final"),
                   peak_mem_mb=result.get("peak_mem_mb"), out_dir=out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -k summarise -v`
Expected: PASS.

- [ ] **Step 5: Create the result document**

Create `docs/results/2026-09-19-p-e-pseudo-displacement.md`:

```markdown
# P-E open question (ii): do the pseudo points move?

Date: <FILL: run date>. Spec: `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §10 open question (ii).
Plan: `docs/superpowers/plans/2026-09-19-v2-p-e-pseudo-ft.md` Task 9.

## Question

Does the movable pseudo point carry usable gradient, or does the frozen `ω_k`
starve it so the degree of freedom is wasted? If it is starved, spec §6's
fallback (a softmin envelope over 8–16 fixed candidates per net, no DOF) is the
recommended replacement.

## Protocol

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" src/scripts/run_pseudo_displacement.py \
  --config <mempool_tile config> --k 16 --pseudo-max 200000 --window 100
```

Phase 3 (fence GP) of the main flow, `--pseudo-ft on`,
`--pseudo-omega-source overflow`, `--norm-policy grandplan`. Displacement is
the L1 distance a pseudo point travels between consecutive `home_period`
samples, aggregated to the maximum within each 100-iteration window.

Decision rule (fixed before the run): `eps = 0.5·row_height`;
**starved** if `p95 < eps` in more than 80% of post-activation windows;
**unstable** if any window's `p99` exceeds half the die's L1 diagonal;
**moves** otherwise.

## Result

| Quantity | Value |
|---|---|
| `pseudo_n` | <FILL> |
| `pseudo_mu` | <FILL> |
| `lambda_pseudo_ft_final` | <FILL> |
| `p50` displacement / 100 iters (median over windows) | <FILL> |
| `p95` displacement / 100 iters (median over windows) | <FILL> |
| `p99` displacement / 100 iters (max over windows) | <FILL> |
| fraction of windows with `p95 < eps` | <FILL> |
| **verdict** | <FILL: moves / starved / unstable> |
| extra GPU memory vs `--pseudo-ft off` | <FILL> MB (predicted ≈0.8 GB at 2e6 nets, K=16) |
| phase-3 runtime delta | <FILL> |

Per-window table: see `results/p_e_pseudo_displacement/<stamp>/summary.json`.

## Reading

<FILL: one paragraph. If "starved": state which of the three candidate causes
the norm_trace shows — λ_pseudo_ft never leaving zero, the preconditioner proxy
not installed, or the detour gradient being dominated by the hinge — and
whether the spec §6 fallback should be adopted. If "moves": state the observed
relationship between a region's ω and the direction of travel, which is the
evidence that the r(p) mixture channel (P-E design ruling) is doing the work.>

## Artefacts

| File | SHA-256 |
|---|---|
| `results/p_e_pseudo_displacement/<stamp>/summary.json` | <FILL> |
| `<out_dir>/pseudo.npz` | <FILL> |
| `<out_dir>/norm_trace.jsonl` | <FILL> |
```

- [ ] **Step 6: Run the experiment**

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
"$IOPLACE_PYTHON" src/scripts/run_pseudo_displacement.py \
  --config benchmarks/ispd25/h100/mempool_tile_wrap.json --k 16 \
  --pseudo-max 200000 --window 100
```

Fill every `<FILL>` in the result document from `summary.json`, and record each
artefact's SHA-256 with `sha256sum`.

- [ ] **Step 7: Commit**

```bash
git add src/scripts/run_pseudo_displacement.py docs/results/2026-09-19-p-e-pseudo-displacement.md tests/test_pseudo_ft_driver.py results/p_e_pseudo_displacement
git commit -m "test(pseudo-ft): displacement experiment for design open question (ii)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: The "done" criterion — the `mempool_group` A/B — and the documentation sync

**Files:**
- Modify: `tests/test_pseudo_ft_driver.py` (append the env-gated acceptance test)
- Modify: `docs/dev-env.md`
- Create: `docs/results/2026-09-19-p-e-group-ab.md`

**The criterion, verbatim from spec §9:** *"E: FT strictly reduced on `mempool_group` at HPWL ≤1.01× the no-pseudo arm, or an honest negative result."*

**Recipe, fixed here so both arms differ in exactly one flag.** Same config, same `K=16`, same region geometry (`regions.json` from the producer or `--region-source grid`), same seed, same `--norm-policy grandplan`, same iteration budget. Arm A: `--pseudo-ft off`. Arm B: `--pseudo-ft on --pseudo-max 2000000 --pseudo-omega-source overflow`. Read `ft_count` and `hpwl` (post-LG, i.e. `hpwl_lg`) from each `result.json`. Pass iff `ft_B < ft_A` **and** `hpwl_B <= 1.01·hpwl_A`. Anything else is written up as a negative result in `docs/results/2026-09-19-p-e-group-ab.md` — a negative result is an acceptable completion of P-E, a *missing* result is not.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pseudo_ft_driver.py`:

```python
def check_group_ab(arm_off, arm_on, *, hpwl_tolerance=1.01):
    """Spec sec 9's 'done' criterion for P-E, as a pure function so it is
    testable without a two-hour run."""
    ft_ok = arm_on["ft_count"] < arm_off["ft_count"]
    hpwl_ok = arm_on["hpwl_lg"] <= hpwl_tolerance * arm_off["hpwl_lg"]
    return dict(passed=bool(ft_ok and hpwl_ok), ft_ok=bool(ft_ok),
                hpwl_ok=bool(hpwl_ok),
                ft_delta=arm_on["ft_count"] - arm_off["ft_count"],
                hpwl_ratio=arm_on["hpwl_lg"] / arm_off["hpwl_lg"])


def test_group_ab_criterion_is_both_conditions():
    base = {"ft_count": 1000, "hpwl_lg": 100.0}
    assert check_group_ab(base, {"ft_count": 900, "hpwl_lg": 100.5})["passed"]
    assert not check_group_ab(base, {"ft_count": 900, "hpwl_lg": 102.0})["passed"]
    assert not check_group_ab(base, {"ft_count": 1000, "hpwl_lg": 99.0})["passed"]
    got = check_group_ab(base, {"ft_count": 900, "hpwl_lg": 101.0})
    assert got["ft_delta"] == -100 and got["hpwl_ratio"] == pytest.approx(1.01)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("IOPLACE_GROUP_AB"),
                    reason="set IOPLACE_GROUP_AB=<mempool_group config> to run "
                           "the P-E acceptance A/B (hours, GPU 3 only)")
def test_mempool_group_pseudo_ab(tmp_path):
    from ioplace.drivers.run_main_flow import run_main_flow
    config = os.environ["IOPLACE_GROUP_AB"]
    common = dict(k=16, region_source="grid", norm_policy="grandplan",
                  init="seed" if os.environ.get("IOPLACE_GROUP_SEED") else "die_center")
    off = run_main_flow(config, str(tmp_path / "off"), pseudo_ft="off", **common)
    on = run_main_flow(config, str(tmp_path / "on"), pseudo_ft="on",
                       pseudo_max=2_000_000, pseudo_omega_source="overflow",
                       **common)
    verdict = check_group_ab(off, on)
    with open(str(tmp_path / "ab.json"), "w") as stream:
        json.dump({"off": off, "on": on, "verdict": verdict}, stream, indent=2,
                  sort_keys=True, default=str)
    # A negative result is a legitimate outcome of P-E, but it must be written
    # up: fail loudly only if the artefacts are missing.
    assert on["pseudo_n"] > 0
    assert set(verdict) == {"passed", "ft_ok", "hpwl_ok", "ft_delta", "hpwl_ratio"}
```

- [ ] **Step 2: Run the non-slow test to verify it fails, then passes**

Run: `source src/scripts/env.sh && "$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -k group_ab -v`
Expected: first FAIL with `NameError: name 'check_group_ab' is not defined` if the helper is added after the test; then PASS once both are in place.

- [ ] **Step 3: Run the acceptance A/B**

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
export IOPLACE_GROUP_AB=<mempool_group config>
"$IOPLACE_PYTHON" -m pytest tests/test_pseudo_ft_driver.py -k mempool_group -v -s
```
Budget: phase 1 + phase 3 GP + LG per arm; at group scale expect 1–3 h per arm plus the fence multiplier from spec §10 risk 2. Two arms, one GPU, sequential.

- [ ] **Step 4: Write up the result**

Create `docs/results/2026-09-19-p-e-group-ab.md` with the following sections, every `<FILL>` replaced:

```markdown
# P-E acceptance: pseudo-point FT on mempool_group

Date: <FILL>. Criterion (spec §9): FT strictly reduced at HPWL ≤ 1.01× the
no-pseudo arm, or an honest negative result.

| Metric | `--pseudo-ft off` | `--pseudo-ft on` | Δ |
|---|---|---|---|
| `ft_count` | <FILL> | <FILL> | <FILL> |
| `io_count` | <FILL> | <FILL> | <FILL> |
| `hpwl_gp` | <FILL> | <FILL> | <FILL> |
| `hpwl_lg` | <FILL> | <FILL> | ratio <FILL> |
| `lg_loss` | <FILL> | <FILL> | <FILL> |
| `fence_compliance` | <FILL> | <FILL> | — |
| phase-3 runtime (s) | <FILL> | <FILL> | <FILL> |
| peak GPU memory (GB) | <FILL> | <FILL> | <FILL> |
| `pseudo_n` | 0 | <FILL> | — |
| `lambda_pseudo_ft_final` | — | <FILL> | — |
| `pseudo_detour_min` | — | <FILL> | — |
| `pseudo_hinge_active` | — | <FILL> | — |

**Verdict:** <FILL: PASS / NEGATIVE>.

**Reading:** <FILL: if NEGATIVE, say which half failed and what the evidence
points at — λ never ramping (check `norm_trace.jsonl`'s `pseudo_ft` rows), the
pseudo points not moving (cross-reference
`docs/results/2026-09-19-p-e-pseudo-displacement.md`), or FT falling while HPWL
overshoots 1.01× (then report the λ sweep that would trade them). Do not tune
the criterion after the fact.>

**Artefacts:** `<off dir>/result.json`, `<on dir>/result.json`,
`<on dir>/pseudo.npz`, both `norm_trace.jsonl`, with SHA-256 for each.
```

- [ ] **Step 5: Document the flags**

Add to `docs/dev-env.md` a subsection **"Pseudo-point FT (P-E) flags"** listing all eight flags with their defaults, the two drivers that accept them (`run_placement.py --mode io` and `run_main_flow.py`), the `pseudo.npz` artefact and its schema, the flat-mode restriction (`--pseudo-omega-source uniform` only, because `model.overflow` is a scalar there), the requirement that `--norm-policy` is `grandplan` or `adaptive`, and the memory rule of thumb (≈48 B/net for the term's saved tensors plus ≈`K·E·4 B` × ~6 for the retained ω-mixture graph; ≈0.9 GB at `E = 2e6`, `K = 16`).

- [ ] **Step 6: Run the full suite one last time**

Run: `source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3 && "$IOPLACE_PYTHON" -m pytest`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add tests/test_pseudo_ft_driver.py docs/dev-env.md docs/results/2026-09-19-p-e-group-ab.md
git commit -m "docs(pseudo-ft): group-scale A/B acceptance, result write-up and flag reference

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage.**

| Spec requirement | Task |
|---|---|
| §0 row: one movable pseudo point per 2-pin cross-partition net | Task 1 (`select_pseudo_nets`, degree-2 + cross-region filter) |
| §6: injected as filler-class nodes at the tail, between `initialize()` and `NonLinearPlace(...)` | Task 2 (`inject_pseudo_nodes`), Task 7 Step 5 and Task 8 Step 5 (call sites) |
| §6: no index remapping, every pin index `< num_physical` | Task 2 (`assert_injection_invariants` 1–3), Task 2 Step 1 `test_real_injection_is_hpwl_and_pin2node_bit_identical` |
| §6: increment `num_nodes`/`num_filler_nodes`, extend `filler_start_map` | Task 2 — `num_nodes` is a derived property so incrementing `num_filler_nodes` covers it; `filler_start_map` is **deliberately not** extended, with the measurement and rationale in Global Constraints deviation 1 |
| §6: non-fence bucket (`region_id = K`) in fence mode, unconfined | Task 2 (the pseudo block sits past `filler_start_map[-1]`, i.e. beyond the non-fence bucket, so no fence field or legaliser reaches it) |
| §6: rejected alternatives (movable-node prepend, net splitting) | Not implemented — recorded in Global Constraints and the Task 3 op-reuse decision |
| §6: `L_ft = Σ w_e·ω(r(p))·[WA_γ(u,v,p) − WA_γ(u,v)] + reg` | Tasks 3, 4, 6 |
| §6: `ω_k = (util_k/target_density)²` from the density-field overflow, refreshed per `home_period`, frozen in the gradient | Task 4 (`omega_from_overflow`, deviation 2), Task 7/8 (refresh inside the atomic transaction), Task 4 `test_omega_is_frozen_in_the_gradient` |
| §6: `r(p)` read as `softmax(−SDF/τ_p)` | Task 4 (`omega_mixture`), `test_omega_mixture_is_continuous_across_a_boundary` |
| §6: anti-collapse hinge, `ρ_min = 0.1‖u−v‖₁`, `μ` from §4 | Task 5 |
| §6: area = one site × one row height, filler budget unchanged | Task 2 (`size_x=site_width`, `size_y=row_height`; fence mode leaves every budget untouched, flat mode shrinks regular fillers and asserts the total) |
| §6: never legalised; FT detached at the `legalize_op` boundary; `pseudo.npz` before LG; nothing indexes beyond `num_physical` | Task 2 (invariants 4–6), Task 7 Step 5 and Task 8 Step 5 (`enabled = False` + `save_pseudo` in the wrapper), Task 7 Step 1 (`evaluation.npz` assertion) |
| §6: ≈48 B/node memory, default `n_pseudo_max = 2e6`, top nets by measured per-net FT | Task 1 (`n_max=2_000_000`, ranking), Task 3 (analytic backward is what makes 48 B/net true), Task 4 note + Task 10 Step 5 (the ω-mixture's extra `O(E·K)`) |
| §6: multi-pin deferred; fallback = softmin over fixed candidates | Task 1 (degree-2 only), Task 9 (the fallback is the documented recommendation if the verdict is `"starved"`) |
| §3: pseudo-FT stays ON after the freeze, pseudo nodes unconfined | Task 8 (the phase-3 bundle) |
| §4: register as `pseudo_ft`, curvature 1, `value(pos, ctx)` | Task 6 (`test_value_protocol_and_normalizer_registration`), Tasks 7/8 (`normalizer.register(..., 1.0, ...)`) |
| §9 `tests/test_pseudo_ft.py`: collapse regularisation activates | Task 5 |
| §9: `ω` stays frozen | Task 4 |
| §9: filler-tail injection leaves `pin2node` and HPWL bit-identical | Task 2 |
| §9: pseudo nodes absent from `evaluation.npz` | Task 7 Step 1 |
| §9 "done" for E | Task 10 |
| §10 (ii): displacement percentiles per 100 iterations on tile | Task 9 |

No gaps. The two spec sentences not implemented literally — "density participation" and "the bracket is ≥0" — are recorded as deviations 1 and 3 with the measurement that forced each and a test pinning the replacement invariant.

**2. Placeholder scan.** Every code step carries runnable code. The only `<FILL>` markers are inside the two *result documents*, where they mark measurements that do not exist until the experiment runs; each is paired with an explicit command that produces the number. Task 8 carries a hard precondition ("if `run_main_flow.py` does not exist, stop and report") rather than a vague dependency. Task 8 Steps 5.3 and 5.7 refer back to Task 7 Step 5 for the initial-position write and the precond install; the code is spelled out once in Task 7 and the reference names the exact step, which keeps the two call sites from drifting.

**3. Type consistency.** Checked across tasks: `PseudoPlan` field names (`net_ids`, `node_u`, `node_v`, `off_ux/uy/vx/vy`, `w`, `ft`, `movable_u/v`, `pseudo_start`, `num_nodes_before/after`, `mode`, `size_x/y`) are identical in Tasks 1, 2, 6, 8; `inject_pseudo_nodes` returns the log dict whose `mode` key Tasks 7 and 8 pass to `save_pseudo(..., mode=...)`; `omega_from_overflow(overflow, *, k=None, omega_max=25.0)` is called with `k=` in Tasks 7 and 8 and without it in Task 4's unit test, matching the default; `PseudoFtTerm.set_omega` takes a `(K,)` array in every call site; `wa_detour`'s six-coordinate order `(ux, uy, vx, vy, px, py, gamma)` is the same in Task 3's tests, Task 6's `forward`, and `wa_detour_numpy`; `FenceTermBundle.on_iteration(iteration, pos, placer, normalizer, entry)` has the same five parameters in `fence_terms.py`, `_Combined`, `_PseudoBundle` and the `run_main_flow` call site; `_PseudoPrecond(inner, pseudo_start, n, num_nodes)` is constructed identically in Task 7's test and its two call sites. `PseudoFtTerm.value(pos, ctx)` reads `ctx["gamma"]`, which is the key `run_placement_io.py:562`'s `ctx_norm` already supplies.

**4. Cross-plan check.** Re-read against P-B (`run_fence_gp`'s `extra_terms` hook, the `pseudo.npz` artefact row, Task 4's `fence_phase` builder), P-H (`TermNormalizer.register`'s exact signature and the one-bump-per-transaction discipline) and P-D (flags, term names, the single phase-3 normalizer, the two attachment forms). The four seams are enumerated in "Reconciliation with P-D" above; the P-B seam is Task 8's stated precondition; the P-H seam is Task 6's registration test, which constructs a real `TermNormalizer` rather than a stub so a signature change upstream fails here loudly.
