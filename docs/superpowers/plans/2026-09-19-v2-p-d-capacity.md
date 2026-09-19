# v2 Subproject P-D — Per-Segment Boundary IO Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the placer a per-boundary-*segment* IO capacity constraint — segments enumerated once from the region lattice, capacities extracted once from OpenROAD (with a tech-LEF pitch fallback), a differentiable demand and penalty term inside global placement, and an exact evaluator hard check that reports per-segment utilisation.

**Architecture:** Four seams that only meet through plain arrays. (1) `src/ioplace/region_segments.py` turns a `RegionGrid` into a frozen *segment table* plus a unit-edge→segment-id raster and per-row/column CSR indexes; every other component keys off the segment id and nothing else. (2) `src/ioplace/capacity/` runs OpenROAD once, offline, on the input DEF, converts GCell track capacities into one scalar per segment id, and writes `capacity.npz` with a receipt hash — the GCell grid is never touched again. (3) `src/ioplace/ops/cap_term.py` adds a chunked autograd term whose demand is `D_s = Σ_e w_e·q_{e,u}·q_{e,v}·α_{e,s}` and whose penalty is GrandPlan Eq.5's shape; it registers with the P-H `TermNormalizer` like `io`/`ft` do. (4) The reference and GPU evaluators map every unit crossing through the raster and report `segment_demand`/`segment_util`, bit-exact against each other.

**Tech Stack:** Python 3.12 (`$DREAMPLACE_ROOT/.venv312/bin/python`), torch 2.8.0+cu128, numpy, pytest, DREAMPlace 4.3.1 (`$DREAMPLACE_ROOT/install`), OpenROAD (`$OPENROAD_BIN`, optional/env-gated). No new DREAMPlace patch.

**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` — §0 "IO capacity" row, all of §5, §3 ("terms after the freeze: capacity on"), §4 (`register("cap", term, curvature=<max_s pen''>, activate_overflow=0.30)`; term protocol `value(pos, ctx)`), §9 (the D "done" criterion, the parity contract, `tests/test_region_segments.py` and `tests/test_capacity_term.py`), §10 risk 5 and open question (i). Read the spec before starting; this plan argues from it and the two travel together.

**Dependency plans — read, do not duplicate:**

- `docs/superpowers/plans/2026-09-19-v2-p-h-normalisation.md` — P-H owns `src/ioplace/norm.py` (`TermNormalizer.register/probe/transaction`, the `value(pos, ctx)` term protocol, `src/ioplace/ops/norm_terms.py`) and the `--norm-*` flags in `src/ioplace/drivers/run_placement.build_parser`. **Landed.** P-D adds one registration and one adapter class in the same style; it changes nothing in `norm.py`.
- `docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md` — P-B owns `src/ioplace/artifacts.py`, `src/ioplace/freeze.py`, `src/ioplace/main_flow_metrics.py`, `src/ioplace/drivers/run_main_flow.py`, and inside it `run_fence_gp(..., extra_terms=(), timer=None)` — the documented attachment point for this term — plus the `capacity.npz` row in the artefact table (§1). P-D does **not** re-implement any of them; Task 9 adds a `capacity=` kwarg to `run_fence_gp`/`run_main_flow` and appends its own `term_fn` to `extra_terms`.
- `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` — P-F owns `src/ioplace/straddle.py`, `--node-anchor`, and the `evaluation.npz` schema-2 conventions (`SCHEMA_VERSION`, `SUPPORTED_SCHEMA_VERSIONS`, the `_pack_<x>_metrics` packer pattern). P-D reuses those conventions verbatim for its own block — see the **schema-version rule** in Global Constraints.

**What P-D owns:** `src/ioplace/region_segments.py`; `src/ioplace/capacity/`; `src/ioplace/ops/cap_term.py`; the `segment_*` fields in `EvalResult`, `evaluation.npz` and `result.json`; the `--capacity` flag in both drivers; the surrogate-vs-router rank-correlation experiment and its result document.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Run protocol.** From the repo root `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer`, branch `v2/redesign`:

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```

`src/scripts/env.sh` exports `DREAMPLACE_ROOT=/ldaphome/yyds-tsai-dev/DREAMPlace` and `IOPLACE_PYTHON=$DREAMPLACE_ROOT/.venv312/bin/python`. Use `-m "not slow"` while iterating; run the full suite before declaring a task done. This is a shared H100 NVL host: run `nvidia-smi` before any GPU work and honour `CUDA_VISIBLE_DEVICES`. At the time this plan was written GPUs 0–2 were foreign and at 100% utilisation and GPU 3 was carrying a long acceptance run (22 GB resident, 0% util) — check again; if GPU 3 is still busy, run `-m "not slow and not gpu"` and defer the GPU steps of Tasks 5, 7, 8 and 9.

**OpenROAD protocol.** `source src/scripts/openroad_env.sh` after `env.sh`; it exports `OPENROAD_BIN` (at writing: `/ldaphome/yyds-tsai-dev/tools/openroad/prefix-upstream-grt-uint64/bin/openroad`, present). **Every OpenROAD-touching test is `@pytest.mark.slow` and skipped unless `IOPLACE_OPENROAD_TESTS=1` *and* `$OPENROAD_BIN` is executable.** The extraction code path must be fully testable without OpenROAD, through the recorded `resources.json` fixture Task 3 checks in.

**No new DREAMPlace patch.** DREAMPlace source is off-limits. `src/ioplace/dp_patch/m2-extra-obj-terms.patch`, `iteration-callback.patch` and `shapely2-compat.patch` are the only modifications and none of them changes. `m2-extra-obj-terms.patch` already adds extra terms after the fence branch of `PlaceObj.obj_fn`, so the capacity term attaches through `dp_hook.attach_terms` alone (spec §1).

**Python >= 3.9.** No `match`, no PEP-604 `X | Y` annotations, no PEP-585 `tuple[int, ...]` annotations in runtime code (the interpreter is 3.12, but the floor is `pyproject.toml`'s `requires-python = ">=3.9"` and the rest of `src/ioplace` honours it).

**Unit rule (verbatim, `docs/superpowers/specs/2026-09-15-round-feedback-design.md:169-196` as quoted by spec §5).**

> the GCell grid is touched once, offline, by the extractor; its output is a scalar keyed by segment id. Inside GP nothing indexes one grid with the other's ids and no IO-lattice load is divided by a GCell capacity. Zero-capacity segments stay hard-blocked with the same finite penalty; no epsilon substitution.

**Penalty form (verbatim, spec §5).**

> `d_s=(D_s−C_s)/C_s`, `L_cap = Σ_s (2[d_s]_+³ + [d_s]_+²)` — GrandPlan Eq.5's shape, `C¹` at the knee, steep beyond. Rejected: phase-1 §5.4's `softplus((D−C)/C)` (`docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md:86-90`), because `softplus(−1)=0.31` exerts force at half capacity, distorting WL where nothing is violated.

No other penalty shape may appear anywhere in this subproject, including in diagnostics.

**Capacity semantics string (verbatim, spec §5).** `capacity.npz`'s metadata must carry this exact string as `capacity_semantics`:

```
usable tracks crossing the segment; one net crossing consumes one track
```

**Parity contract.** Extending `tests/test_evaluator_gpu.py`'s existing contract (module docstring `evaluator_gpu.py:16-26`; `_assert_batch_invariant_fields`, `tests/test_evaluator_gpu.py:426-445`): the **integer** capacity fields `segment_demand`, `num_over_capacity`, `num_zero_capacity_segments`, `zero_capacity_demand`, `segment_demand_total`, and the candidate arrays `cand_net`/`cand_u`/`cand_v`/`cand_seg`/`cand_count` must be **bit-exact** between `evaluator_ref` and `evaluator_gpu` and across the construction parameters `mst_chunk_budget`, `seg_chunk_budget`, `edge_batch_size`. `segment_util`, `max_util` and `p99_util` are **also bit-exact**, because both evaluators compute them by calling the single numpy helper `region_segments.segment_utilisation` on the (bit-exact) `segment_demand` — they are not independently reduced. The *gradient* of the capacity term carries no bit-exactness claim: `index_put_(accumulate=True)` on CUDA is atomic, exactly like `_IoFn`'s existing `index_add_`, so gradient tests use `rel <= 1e-10`.

**Activation and lifetime (spec §0/§3/§5).** `λ_cap` comes from `TermNormalizer`, activated at `overflow ≤ 0.30`, and **stays on after the freeze**: phase 3 registers it again on the fence GP's own normalizer. IO and FT are off there; the capacity gradient in phase 3 flows through `α` (where along a boundary a net sits), not through `q` (which is saturated once membership is frozen and cells are fenced). Any change that disables the `α` path when `q` is binary breaks the post-freeze term.

**`evaluation.npz` schema-version rule.** P-D adds a capacity block to `evaluation.npz` using P-F's conventions (`SCHEMA_VERSION`, `SUPPORTED_SCHEMA_VERSIONS`, `_pack_<x>_metrics`). Which number to use is decided by inspection at execution time, in Task 8 Step 0:

```bash
"$IOPLACE_PYTHON" - <<'PY'
from ioplace.export import evaluation as e
print("SCHEMA_VERSION", e.SCHEMA_VERSION,
      "SUPPORTED", getattr(e, "SUPPORTED_SCHEMA_VERSIONS", None))
PY
```

- Prints `SCHEMA_VERSION 2 SUPPORTED (1, 2)` → **P-F has landed.** P-D sets `SCHEMA_VERSION = 3` and `SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)`.
- Prints `SCHEMA_VERSION 1 SUPPORTED None` → **P-F has not landed.** P-D introduces `SCHEMA_VERSION = 2` and `SUPPORTED_SCHEMA_VERSIONS = (1, 2)` itself, with the same comment P-F's Task 5 Step 3 uses, and the executor **must** append a note to `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` Task 5 saying that P-D already took version 2 and P-F must bump to 3 with `SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)`.
- Anything else → stop and escalate; the two plans have drifted.

Write the chosen number into the Task 8 commit message so the next reader can tell which branch was taken.

**Subproject preconditions.** The spec's order is P-H → P-B+P-C → P-F → P-D, so P-H, P-B, P-C and P-F are all expected to have landed. Only two things actually gate this plan: `src/ioplace/norm.py` (P-H, already on `v2/redesign`) and, for Task 9 only, `src/ioplace/drivers/run_main_flow.py` (P-B). If `run_main_flow.py` is absent, Task 9's tests `pytest.importorskip` out and the task is recorded as blocked rather than skipped silently; Tasks 1–8 and 10 are independent of it.

**Commit trailer.** Every commit message in this plan ends with:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

**"Done" for D (spec §9).** *"per-segment utilisation reported from both the GP surrogate and the router with their rank correlation stated."* Task 10 produces that number and writes it into `docs/results/2026-09-19-p-d-capacity-rank-correlation.md`. Spec §10 open question (i) pre-registers the decision rule: Spearman **below 0.5 → fall back to pair-level `D_ab`/`C_ab`**. The result document must state the number and apply the rule explicitly.

---

## Recorded interpretations

Two places where §5's prose is under-determined and the plan fixes a reading. Both are load-bearing; an implementer who reads only the spec will get them wrong.

**D-1: the demand pair is the net's *terminal* pair, not the segment's boundary pair.** §5 writes `D_s = Σ_e w_e·q_{e,a(s)}·q_{e,b(s)}·α_{e,s}` with `a(s)`, `b(s)` the segment's own region pair, and separately claims *"Feed-through attribution is automatic: a traversing net crossed entry and exit segments, so both are in its candidate list and both take demand."* Those two statements contradict each other. `q_{e,k} = 1 − Π_i(1−p_{i,k})` is a *pin-presence* indicator (`ops/io_term.py:117-120`). For a 2-pin net whose pins sit in regions 0 and 2 and whose route feeds through region 1, `q_{e,1} = 0`, so the entry segment's product `q_{e,0}·q_{e,1}` is zero and the exit segment's `q_{e,1}·q_{e,2}` is zero: under the literal formula a feed-through net contributes **no** capacity demand at all, which is the opposite of the stated intent.

The reading this plan implements: each candidate carries a **demand pair** `(u, v)` — the region ids of the two MST-edge endpoints whose leg produced the crossing — and a **segment** `s`. Demand is

```
D_s = Σ_{c ∈ cand(s)} w_{e(c)} · q_{e(c),u(c)} · q_{e(c),v(c)} · α_c
```

For a leg between adjacent regions, `(u, v) == (a(s), b(s))` and this is §5's formula character for character. For a feed-through, `(u, v) = (0, 2)` and both the entry and the exit segment take `w·q_0·q_2`, which is what §5 says must happen. Legs with `u == v` (an L-route that detours out of and back into one region) are **dropped**: they carry no `q`-expressible signal. The dropped fraction of crossings is reported as `cap_cand_dropped_frac` so the surrogate's coverage is visible rather than silent.

**D-2: the α softmax groups by (net, demand pair, *boundary* pair).** §5 says `α_{e,s} = softmax_{s'∈cand(e,pair)}(−d₁(c_e,s')/τ_b)`, i.e. a softmax over "the pair's" candidate segments, which sums to 1 per group. Alternatives must share a group; *sequential* segments must not. Two segments on the same boundary pair `(a,b)` are alternative places the same crossing could land, so softmaxing between them is right. A feed-through's entry segment (boundary pair `(0,1)`) and exit segment (boundary pair `(1,2)`) are both *required*; putting them in one softmax would halve each. So the group key is `(net, u, v, pair(s))`, and each group's candidates share one full `w·q_u·q_v`. This is again §5's formula exactly whenever a net's candidates all lie on one boundary pair.

The caps keep §5's budget: **`m_pairs = 4`** groups per net (ranked by total crossing count, ties by group key) and **`m_seg = 2`** segments per group (ranked by count, ties by segment id) — at most 8 candidates per net, as §5 intends.

---

## File Structure

**New modules**

| File | Responsibility |
|---|---|
| `src/ioplace/region_segments.py` | Pure numpy. `SegmentTable` (the frozen segment table), `enumerate_segments(rg)`, the unit-edge→segment-id raster, per-row/column CSR, `segments_digest`, leg lookup (`segment_ids_on_row/col`, `edge_segment_ids`), `segment_utilisation`, `CAPACITY_SCALARS`, `select_candidates` + `Candidates`. No torch, no DREAMPlace, no OpenROAD. Everything else in P-D keys off this module's segment ids. |
| `src/ioplace/capacity/__init__.py` | Re-exports `CAPACITY_SEMANTICS`, `save_capacity`, `load_capacity`, `extract_capacity`. |
| `src/ioplace/capacity/extract.py` | The one-time offline extraction: tech-LEF parsing (`parse_tech_lef_layers`, `track_density`), the `ρ·ℓ` fallback (`lef_capacity`), the OpenROAD GCell conversion (`gcell_capacity`), the thin `run_openroad` wrapper (`run_extraction`), `capacity.npz` I/O with receipt hash (`save_capacity`/`load_capacity`), and a `python -m ioplace.capacity.extract` CLI. |
| `src/ioplace/ops/cap_term.py` | torch. `cap_penalty`/`cap_penalty_grad`/`normalised_overflow`, `segment_softmax`, `l1_point_box_dist`, `CapTermRef` (dense autograd oracle, tests only), `_CapFn` + `CapTerm` (chunked production term), `CapNormTerm` (the `value(pos, ctx)` adapter for `TermNormalizer`). |
| `src/scripts/run_cap_rank_correlation.py` | The slow surrogate-vs-router experiment for spec §9's D criterion and §10 open question (i). |
| `docs/results/2026-09-19-p-d-capacity-rank-correlation.md` | Result document for that experiment (template written by Task 10, filled by the run). |

**Modified**

| File | Change |
|---|---|
| `src/ioplace/evaluator_ref.py` | `evaluate(..., segments=None, segment_capacity=None, capacity_candidates=False)`; new `EvalResult` capacity fields; per-leg slice-length assertion. |
| `src/ioplace/evaluator_gpu.py` | `GpuEvalContext(..., segments=None, segment_capacity=None)`, `evaluate(..., capacity_candidates=False)`; segment ids collected in `_process_segments`; the `Ph`/`Pv` total assertion. |
| `src/ioplace/export/evaluation.py` | Schema bump per the rule above; `segment_demand`/`segment_capacity`/`segment_util` arrays; `metadata["capacity"]`; load-side validation. |
| `src/ioplace/drivers/run_placement.py` | `_pack_capacity_metrics`; `--capacity` and the three `--cap-*` knobs in `build_parser`; pass-through in `main`. |
| `src/ioplace/drivers/run_placement_io.py` | `capacity=None` kwarg, term construction, `normalizer.register("cap", ...)`, `term_fn` addition, candidate refresh on the `home_period` cadence, trajectory + `RESULT_FIELDS`. |
| `src/ioplace/drivers/run_main_flow.py` (guarded) | `capacity=None` on `run_main_flow` and `run_fence_gp`; a phase-3 `TermNormalizer` carrying only `cap`. |

**Tests**

`tests/test_region_segments.py` (new), `tests/test_capacity_extract.py` (new), `tests/test_capacity_term.py` (new), `tests/test_evaluator_capacity.py` (new), `tests/test_evaluator_gpu.py` (append), `tests/test_evaluation_export.py` (append), `tests/test_driver_capacity.py` (new), `tests/test_cap_rank_correlation.py` (new). Fixture: `tests/data/capacity/gcd_resources.json` (recorded OpenROAD `resources.json`, hand-trimmed).

---

### Task 1: Segment enumeration — `SegmentTable`, raster, CSR

**Files:**
- Create: `src/ioplace/region_segments.py`
- Test: `tests/test_region_segments.py`

**Interfaces:**
- Consumes: `ioplace.region_grid.RegionGrid` (`.grid` (ny,nx) int16, `.k`, `.die`, `.cell_w`, `.cell_h`, `.nx`, `.ny`); `ioplace.regions.RegionSet`/`RegionSpec`.
- Produces:
  - `SEGMENT_SCHEMA_VERSION = 1`, `ORIENT_V = 0`, `ORIENT_H = 1`
  - `class SegmentTable` with attributes `orient` (S,) int8, `line` (S,) int32, `lo` (S,) int32, `hi` (S,) int32, `pair_a` (S,) int16, `pair_b` (S,) int16, `length_units` (S,) int32, `length` (S,) float64, `box` (S,4) float64, `edge_seg_v` (ny,nx-1) int32, `edge_seg_h` (ny-1,nx) int32, `row_ptr` (ny+1,) int64, `row_col` (Ev,) int32, `row_seg` (Ev,) int32, `col_ptr` (nx+1,) int64, `col_row` (Eh,) int32, `col_seg` (Eh,) int32, `k` int, `lattice` int, `die` tuple; properties `num_segments` and methods `pair_key()`, `grid_shape()`
  - `enumerate_segments(rg) -> SegmentTable`
  - `segments_digest(table) -> str` (sha256 hex)

**Conventions this task fixes for every downstream consumer.** A *vertical* segment (`ORIENT_V`) comes from the `grid[:, :-1] != grid[:, 1:]` test: it lies on the boundary line `x = xl + (line+1)·cell_w`, spans rows `[lo, hi)` and is crossed by wires travelling **horizontally**; one unit edge is `cell_h` long. A *horizontal* segment (`ORIENT_H`) comes from `grid[:-1, :] != grid[1:, :]`: line `y = yl + (line+1)·cell_h`, columns `[lo, hi)`, crossed **vertically**, unit edge `cell_w`. Segment ids run V-first, ordered by `(line, lo)`, then H, ordered by `(line, lo)` — deterministic, so ref and GPU and the extractor cannot disagree.

- [ ] **Step 1: Write the failing test**

Create `tests/test_region_segments.py`:

```python
import numpy as np
import pytest

from ioplace.region_grid import RegionGrid
from ioplace.region_graph import region_graph
from ioplace.region_segments import (ORIENT_H, ORIENT_V, enumerate_segments,
                                     segments_digest)
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions

DIE = (0., 0., 4., 4.)


def _rs(*specs):
    """A 4x4-lattice RegionSet over DIE from (name, rects) pairs."""
    return RegionSet(die=DIE, lattice=4,
                     regions=[RegionSpec(n, np.asarray(r, dtype=np.float64))
                              for n, r in specs])


def l_shape():
    """Region 0 is a genuine L (left column + bottom row); region 1 is the
    3x3 block it wraps. The pair (0,1) is therefore split into two disjoint
    segments -- one vertical, one horizontal."""
    return _rs(("L", [[0., 0., 1., 4.], [1., 0., 4., 1.]]),
               ("B", [[1., 1., 4., 4.]]))


def plug():
    """Region 2 plugs the middle of the 0|1 boundary, so the pair (0,1) is
    split into two disjoint segments *on the same boundary line* -- the case
    that actually exercises the run-length encoding."""
    return _rs(("A", [[0., 0., 2., 4.]]),
               ("B", [[2., 0., 4., 1.], [2., 3., 4., 4.]]),
               ("C", [[2., 1., 4., 3.]]))


def notch():
    """A single-bin notch: region 0 pokes one lattice cell into region 1."""
    return _rs(("A", [[0., 0., 2., 4.], [2., 1., 3., 2.]]),
               ("B", [[2., 0., 4., 1.], [3., 1., 4., 2.], [2., 2., 4., 4.]]))


def _as_tuples(table):
    return [(int(table.orient[s]), int(table.line[s]), int(table.lo[s]),
             int(table.hi[s]), int(table.pair_a[s]), int(table.pair_b[s]))
            for s in range(table.num_segments)]


def test_grid_k4_has_exactly_four_segments():
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=4))
    table = enumerate_segments(rg)
    assert table.num_segments == 4
    # V first (line, lo), then H (line, lo). make_grid_regions numbers
    # regions row-major in y: 0=SW, 1=SE, 2=NW, 3=NE.
    assert _as_tuples(table) == [
        (ORIENT_V, 1, 0, 2, 0, 1),
        (ORIENT_V, 1, 2, 4, 2, 3),
        (ORIENT_H, 1, 0, 2, 0, 2),
        (ORIENT_H, 1, 2, 4, 1, 3),
    ]
    assert np.array_equal(table.length_units, [2, 2, 2, 2])
    # cell_w == cell_h == 1.0 here, so physical length equals the unit count.
    np.testing.assert_allclose(table.length, [2., 2., 2., 2.])


def test_l_shaped_region_splits_one_pair_into_two_segments():
    table = enumerate_segments(RegionGrid(l_shape()))
    assert _as_tuples(table) == [(ORIENT_V, 0, 1, 4, 0, 1),
                                 (ORIENT_H, 0, 1, 4, 0, 1)]
    assert table.num_segments == 2


def test_a_plug_splits_one_pair_on_the_same_boundary_line():
    table = enumerate_segments(RegionGrid(plug()))
    assert _as_tuples(table) == [
        (ORIENT_V, 1, 0, 1, 0, 1),
        (ORIENT_V, 1, 1, 3, 0, 2),
        (ORIENT_V, 1, 3, 4, 0, 1),
        (ORIENT_H, 0, 2, 4, 1, 2),
        (ORIENT_H, 2, 2, 4, 1, 2),
    ]
    zero_one = [s for s in range(table.num_segments)
                if (table.pair_a[s], table.pair_b[s]) == (0, 1)]
    assert len(zero_one) == 2
    assert {int(table.line[s]) for s in zero_one} == {1}   # same boundary line
    assert sorted((int(table.lo[s]), int(table.hi[s])) for s in zero_one) == \
        [(0, 1), (3, 4)]


def test_a_one_bin_notch_produces_unit_length_segments():
    table = enumerate_segments(RegionGrid(notch()))
    assert _as_tuples(table) == [
        (ORIENT_V, 1, 0, 1, 0, 1),
        (ORIENT_V, 1, 2, 4, 0, 1),
        (ORIENT_V, 2, 1, 2, 0, 1),
        (ORIENT_H, 0, 2, 3, 0, 1),
        (ORIENT_H, 1, 2, 3, 0, 1),
    ]
    assert int((table.length_units == 1).sum()) == 3


@pytest.mark.parametrize("factory", [
    lambda: make_grid_regions(DIE, 2, 2, lattice=4),
    l_shape, plug, notch,
    lambda: make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20),
])
def test_segment_lengths_sum_to_the_pair_aggregate_ell(factory):
    """spec sec 5: `region_graph`'s ell[a,b] is the sum of a pair's segment
    lengths and remains the pair-level aggregate."""
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    _adj, _D, ell = region_graph(rg)
    totals = np.zeros((rg.k, rg.k), dtype=np.int64)
    for s in range(table.num_segments):
        a, b = int(table.pair_a[s]), int(table.pair_b[s])
        totals[a, b] += int(table.length_units[s])
        totals[b, a] += int(table.length_units[s])
    np.testing.assert_array_equal(totals, ell)
    assert (table.pair_a < table.pair_b).all()


@pytest.mark.parametrize("factory", [l_shape, plug, notch,
                                     lambda: make_grid_regions(DIE, 2, 2, lattice=4)])
def test_raster_round_trips_against_brute_force(factory):
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    grid = rg.grid.astype(np.int64)
    ny, nx = grid.shape
    for i in range(ny):
        for j in range(nx - 1):
            sid = int(table.edge_seg_v[i, j])
            if grid[i, j] == grid[i, j + 1]:
                assert sid == -1
                continue
            assert sid >= 0 and table.orient[sid] == ORIENT_V
            assert int(table.line[sid]) == j
            assert int(table.lo[sid]) <= i < int(table.hi[sid])
            assert (int(table.pair_a[sid]), int(table.pair_b[sid])) == \
                (min(grid[i, j], grid[i, j + 1]), max(grid[i, j], grid[i, j + 1]))
    for i in range(ny - 1):
        for j in range(nx):
            sid = int(table.edge_seg_h[i, j])
            if grid[i, j] == grid[i + 1, j]:
                assert sid == -1
                continue
            assert sid >= 0 and table.orient[sid] == ORIENT_H
            assert int(table.line[sid]) == i
            assert int(table.lo[sid]) <= j < int(table.hi[sid])


def test_csr_matches_the_raster_and_is_sorted():
    rg = RegionGrid(plug())
    table = enumerate_segments(rg)
    ny, nx = rg.grid.shape
    for i in range(ny):
        s0, s1 = int(table.row_ptr[i]), int(table.row_ptr[i + 1])
        cols = table.row_col[s0:s1]
        assert list(cols) == sorted(set(cols))
        expected = np.nonzero(table.edge_seg_v[i] >= 0)[0]
        np.testing.assert_array_equal(cols, expected)
        np.testing.assert_array_equal(table.row_seg[s0:s1],
                                      table.edge_seg_v[i][expected])
    for j in range(nx):
        s0, s1 = int(table.col_ptr[j]), int(table.col_ptr[j + 1])
        rows = table.col_row[s0:s1]
        assert list(rows) == sorted(set(rows))
        expected = np.nonzero(table.edge_seg_h[:, j] >= 0)[0]
        np.testing.assert_array_equal(rows, expected)
        np.testing.assert_array_equal(table.col_seg[s0:s1],
                                      table.edge_seg_h[:, j][expected])


def test_enumeration_is_deterministic_and_digested():
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    a, b = enumerate_segments(rg), enumerate_segments(rg)
    assert segments_digest(a) == segments_digest(b)
    other = enumerate_segments(RegionGrid(plug()))
    assert segments_digest(other) != segments_digest(a)


def test_a_512_lattice_stays_cheap():
    """spec sec 5: O(L^2) ~ 2.6e5, pure numpy. Guard against a python
    per-cell loop creeping back in."""
    import time
    rg = RegionGrid(make_grid_regions((0., 0., 1000., 1000.), 4, 4, lattice=512))
    started = time.perf_counter()
    table = enumerate_segments(rg)
    assert time.perf_counter() - started < 2.0
    # 4x4 regions: 3 internal vertical lines x 4 row bands = 12 V segments,
    # symmetrically 12 H segments.
    assert table.num_segments == 24
    assert table.edge_seg_v.shape == (512, 511)
    assert table.edge_seg_h.shape == (511, 512)
```

Every expected value above was verified against a working prototype of Step 3's code before this plan was written (4×4 grid → 4 segments; L → 2; plug → 5 with `(0,1)` split on line 1 into rows `[0,1)` and `[3,4)`; notch → 5 with three unit-length segments; 512² lattice with 4×4 regions → 24 segments in 0.03 s). They are assertions about real behaviour, not guesses.

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_region_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.region_segments'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/region_segments.py`:

```python
"""Boundary segments of a RegionGrid (v2 design sec 5, subproject P-D).

A *segment* is one maximal run of lattice boundary unit edges that separates
the same unordered region pair along the same boundary line. Everything in
P-D -- the capacity extractor, the differentiable term, both evaluators --
keys off the integer segment ids this module hands out, and off nothing else
(the unit rule, round-feedback spec:169-196: the GCell grid is touched once,
offline, and its output is a scalar keyed by segment id).

Orientation convention, fixed here once:

  ORIENT_V  from grid[:, :-1] != grid[:, 1:].  Boundary line x = xl +
            (line+1)*cell_w, spanning rows [lo, hi).  Wires cross it
            *horizontally*.  One unit edge is cell_h long.
  ORIENT_H  from grid[:-1, :] != grid[1:, :].  Boundary line y = yl +
            (line+1)*cell_h, spanning columns [lo, hi).  Wires cross it
            *vertically*.  One unit edge is cell_w long.

Segment ids are assigned V first, ordered by (line, lo), then H, ordered by
(line, lo).  The order is a contract: evaluator_ref, evaluator_gpu and
capacity.npz all index the same table and a reordering would silently
mismatch them.  `segments_digest` fingerprints it.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

SEGMENT_SCHEMA_VERSION = 1

ORIENT_V = 0
ORIENT_H = 1

# The scalar capacity fields every driver's result.json carries (the
# _pack_<x>_metrics convention P-F introduced for the straddle block).
CAPACITY_SCALARS = (
    "num_segments", "segment_demand_total", "num_over_capacity",
    "num_zero_capacity_segments", "zero_capacity_demand",
    "max_util", "p99_util",
)


@dataclass
class SegmentTable:
    """Frozen per-segment geometry plus the two lookup structures sec 5 asks
    for: a unit-edge -> segment-id raster (2 x L x (L-1) int32, ~2 MB at
    L=512) and per-row/column CSR lists of boundary positions."""

    orient: np.ndarray        # (S,) int8, ORIENT_V / ORIENT_H
    line: np.ndarray          # (S,) int32, boundary line index
    lo: np.ndarray            # (S,) int32, inclusive start along the run axis
    hi: np.ndarray            # (S,) int32, exclusive end
    pair_a: np.ndarray        # (S,) int16, min region id
    pair_b: np.ndarray        # (S,) int16, max region id
    length_units: np.ndarray  # (S,) int32 == hi - lo
    length: np.ndarray        # (S,) float64, physical length in die units
    box: np.ndarray           # (S,4) float64 [x0,y0,x1,y1]; degenerate in one axis
    edge_seg_v: np.ndarray    # (ny, nx-1) int32, -1 where there is no boundary
    edge_seg_h: np.ndarray    # (ny-1, nx) int32
    row_ptr: np.ndarray       # (ny+1,) int64
    row_col: np.ndarray       # (Ev,) int32, ascending within a row
    row_seg: np.ndarray       # (Ev,) int32
    col_ptr: np.ndarray       # (nx+1,) int64
    col_row: np.ndarray       # (Eh,) int32, ascending within a column
    col_seg: np.ndarray       # (Eh,) int32
    k: int
    lattice: int
    die: tuple

    @property
    def num_segments(self):
        return int(self.orient.shape[0])

    def grid_shape(self):
        return (self.edge_seg_h.shape[0] + 1, self.edge_seg_v.shape[1] + 1)

    def pair_key(self):
        """(S,) int64 = pair_a*k + pair_b -- the boundary-pair identity used
        to group alternative segments (interpretation D-2)."""
        return self.pair_a.astype(np.int64) * self.k + self.pair_b.astype(np.int64)


def _runs(key, run_len, other_len, transposed):
    """Run-length-encode a 2-D `key` array (0 == no boundary) along its run
    axis. `key` is already laid out so that a C-order ravel walks the run
    axis fastest. Returns (starts, ends, flat_ids) where flat_ids is -1 off
    the boundary and a dense 0-based run index on it."""
    flat = key.ravel()
    prev = np.zeros_like(flat)
    prev[1:] = flat[:-1]
    first_in_line = (np.arange(flat.size) % run_len) == 0
    start = (flat != 0) & (first_in_line | (prev != flat))
    nxt = np.zeros_like(flat)
    nxt[:-1] = flat[1:]
    last_in_line = (np.arange(flat.size) % run_len) == (run_len - 1)
    end = (flat != 0) & (last_in_line | (nxt != flat))
    starts = np.nonzero(start)[0]
    ends = np.nonzero(end)[0]
    assert starts.shape == ends.shape, "run-length encoding lost a run"
    ids = np.cumsum(start) - 1
    ids = np.where(flat != 0, ids, -1).astype(np.int64)
    return starts, ends, ids


def _boundary_key(a, b, k):
    """0 where a == b, else min*k + max + 1 (so 0 is never a valid pair)."""
    diff = a != b
    return np.where(diff, np.minimum(a, b) * k + np.maximum(a, b) + 1, 0)


def enumerate_segments(rg):
    """Enumerate the boundary segments of a RegionGrid. Pure numpy, O(L^2)."""
    k = int(rg.k)
    grid = rg.grid.astype(np.int64)
    ny, nx = grid.shape
    xl, yl, _xh, _yh = rg.die
    cw, ch = float(rg.cell_w), float(rg.cell_h)

    # ---- vertical segments: runs down the rows of each column boundary ----
    key_v = _boundary_key(grid[:, :-1], grid[:, 1:], k)          # (ny, nx-1)
    # transpose so a C-order ravel walks rows fastest within one column
    starts_v, ends_v, ids_v = _runs(key_v.T.copy(), ny, nx - 1, True)
    line_v = (starts_v // ny).astype(np.int32)
    lo_v = (starts_v % ny).astype(np.int32)
    hi_v = (ends_v % ny + 1).astype(np.int32)
    pk_v = key_v.T.ravel()[starts_v] - 1
    n_v = int(starts_v.size)
    edge_seg_v = np.where(ids_v >= 0, ids_v, -1).reshape(nx - 1, ny).T.astype(np.int32)

    # ---- horizontal segments: runs along the columns of each row boundary ----
    key_h = _boundary_key(grid[:-1, :], grid[1:, :], k)          # (ny-1, nx)
    starts_h, ends_h, ids_h = _runs(key_h, nx, ny - 1, False)
    line_h = (starts_h // nx).astype(np.int32)
    lo_h = (starts_h % nx).astype(np.int32)
    hi_h = (ends_h % nx + 1).astype(np.int32)
    pk_h = key_h.ravel()[starts_h] - 1
    edge_seg_h = np.where(ids_h >= 0, ids_h + n_v, -1).reshape(ny - 1, nx).astype(np.int32)

    orient = np.concatenate([np.full(n_v, ORIENT_V, dtype=np.int8),
                             np.full(starts_h.size, ORIENT_H, dtype=np.int8)])
    line = np.concatenate([line_v, line_h]).astype(np.int32)
    lo = np.concatenate([lo_v, lo_h]).astype(np.int32)
    hi = np.concatenate([hi_v, hi_h]).astype(np.int32)
    pk = np.concatenate([pk_v, pk_h]).astype(np.int64)
    pair_a = (pk // k).astype(np.int16)
    pair_b = (pk % k).astype(np.int16)
    length_units = (hi - lo).astype(np.int32)

    unit = np.where(orient == ORIENT_V, ch, cw)
    length = length_units.astype(np.float64) * unit

    box = np.empty((orient.size, 4), dtype=np.float64)
    is_v = orient == ORIENT_V
    bx = xl + (line.astype(np.float64) + 1.0) * cw
    by = yl + (line.astype(np.float64) + 1.0) * ch
    box[:, 0] = np.where(is_v, bx, xl + lo.astype(np.float64) * cw)
    box[:, 2] = np.where(is_v, bx, xl + hi.astype(np.float64) * cw)
    box[:, 1] = np.where(is_v, yl + lo.astype(np.float64) * ch, by)
    box[:, 3] = np.where(is_v, yl + hi.astype(np.float64) * ch, by)

    # ---- CSR indexes ----
    ii, jj = np.nonzero(edge_seg_v >= 0)                # row-major: (i, then j)
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(ii, minlength=ny))]).astype(np.int64)
    row_col = jj.astype(np.int32)
    row_seg = edge_seg_v[ii, jj].astype(np.int32)

    hi_i, hi_j = np.nonzero(edge_seg_h >= 0)
    order = np.lexsort((hi_i, hi_j))                     # (j, then i)
    hi_i, hi_j = hi_i[order], hi_j[order]
    col_ptr = np.concatenate([[0], np.cumsum(np.bincount(hi_j, minlength=nx))]).astype(np.int64)
    col_row = hi_i.astype(np.int32)
    col_seg = edge_seg_h[hi_i, hi_j].astype(np.int32)

    return SegmentTable(orient=orient, line=line, lo=lo, hi=hi,
                        pair_a=pair_a, pair_b=pair_b,
                        length_units=length_units, length=length, box=box,
                        edge_seg_v=edge_seg_v, edge_seg_h=edge_seg_h,
                        row_ptr=row_ptr, row_col=row_col, row_seg=row_seg,
                        col_ptr=col_ptr, col_row=col_row, col_seg=col_seg,
                        k=k, lattice=int(rg.nx), die=tuple(float(v) for v in rg.die))


def segments_digest(table):
    """sha256 over the defining arrays plus the lattice identity. Written into
    capacity.npz and evaluation.npz so a capacity file can never be paired
    with a different geometry."""
    digest = hashlib.sha256()
    digest.update(str(SEGMENT_SCHEMA_VERSION).encode())
    digest.update(str((table.k, table.lattice, table.die)).encode())
    for arr in (table.orient, table.line, table.lo, table.hi,
                table.pair_a, table.pair_b):
        contiguous = np.ascontiguousarray(arr)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_region_segments.py -v`
Expected: PASS (11 tests). If `test_a_512_lattice_stays_cheap` reports a different segment count, read the number, confirm it against the 4×4 arithmetic above, and pin it.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/region_segments.py tests/test_region_segments.py
git commit -m "$(cat <<'MSG'
feat(capacity): enumerate boundary segments on the region lattice

Run-length-encode the lattice boundary unit edges into maximal same-pair
segments, with a unit-edge->segment-id raster and per-row/column CSR
indexes (v2 design sec 5, P-D task 1). Segment ids are the only currency
the rest of P-D uses; segments_digest pins the ordering contract.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: Leg lookup, utilisation, candidate selection

**Files:**
- Modify: `src/ioplace/region_segments.py` (append; do not touch Task 1's code)
- Test: `tests/test_region_segments.py` (append)

**Interfaces:**
- Consumes: Task 1's `SegmentTable`, `CAPACITY_SCALARS`.
- Produces:
  - `segment_ids_on_row(table, row, lo, hi) -> np.ndarray` (int32, the vertical segments a horizontal leg in grid row `row` from column `lo` to column `hi` inclusive crosses; a CSR slice, so its cost is the number of crossings)
  - `segment_ids_on_col(table, col, lo, hi) -> np.ndarray`
  - `edge_segment_ids(rg, table, x0, y0, x1, y1) -> np.ndarray` (an axis-aligned leg in *coordinates*)
  - `segment_utilisation(demand, capacity) -> (util, scalars)` with `tuple(scalars) == CAPACITY_SCALARS`
  - `class Candidates` with `net`/`u`/`v`/`seg`/`group`/`count` (all (M,) int64) and `n_groups` int; methods `remap_nets(mapping) -> Candidates`, `is_empty()`
  - `select_candidates(net, u, v, seg, count, table, m_pairs=4, m_seg=2) -> Candidates`

**Why the CSR and not the raster here.** Spec §5: *"the per-row CSR turns an L-shape leg into a contiguous slice, so work equals total crossings, and the existing `Ph`/`Pv` prefix-sum counts (`evaluator_gpu.py:182-190`) become a free assertion on slice lengths."* Scanning `edge_seg_v[row, lo:hi]` costs the leg's *length*; slicing the CSR costs its number of *crossings*, which is what the reference evaluator needs at 11M-cell scale. The GPU evaluator keeps using the raster instead (Task 7) because it is already doing a padded gather over the whole leg for the pair-demand pass and a CSR `searchsorted` would add work rather than remove it.

**Candidate capping (interpretation D-2).** `select_candidates` groups by `(net, u, v, pair(seg))`, keeps the `m_pairs` heaviest groups per net (ties by group key) and the `m_seg` heaviest segments per group (ties by segment id), and renumbers groups densely in ascending `(net, u, v, pair, ...)` order. Every tie-break is total, so the output is a pure function of the input arrays — which is what makes the ref/GPU parity contract achievable at all.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_region_segments.py`:

```python
from ioplace.region_segments import (Candidates, edge_segment_ids,
                                     segment_ids_on_col, segment_ids_on_row,
                                     segment_utilisation, select_candidates,
                                     CAPACITY_SCALARS)


def _brute_row(table, row, lo, hi):
    """Every vertical unit edge a horizontal leg in `row` from column `lo` to
    column `hi` (inclusive) crosses, straight off the raster."""
    a, b = min(lo, hi), max(lo, hi)
    ids = table.edge_seg_v[row, a:b]
    return ids[ids >= 0]


def _brute_col(table, col, lo, hi):
    a, b = min(lo, hi), max(lo, hi)
    ids = table.edge_seg_h[a:b, col]
    return ids[ids >= 0]


@pytest.mark.parametrize("factory", [l_shape, plug, notch])
def test_csr_leg_lookup_matches_the_raster_everywhere(factory):
    rg = RegionGrid(factory())
    table = enumerate_segments(rg)
    ny, nx = rg.grid.shape
    for row in range(ny):
        for lo in range(nx):
            for hi in range(nx):
                np.testing.assert_array_equal(
                    segment_ids_on_row(table, row, lo, hi),
                    _brute_row(table, row, lo, hi))
    for col in range(nx):
        for lo in range(ny):
            for hi in range(ny):
                np.testing.assert_array_equal(
                    segment_ids_on_col(table, col, lo, hi),
                    _brute_col(table, col, lo, hi))


def test_leg_lookup_length_equals_the_prefix_sum_crossing_count():
    """The Ph/Pv identity sec 5 calls a 'free assertion on slice lengths'."""
    rg = RegionGrid(plug())
    table = enumerate_segments(rg)
    grid = rg.grid
    ny, nx = grid.shape
    hdiff = (grid[:, :-1] != grid[:, 1:]).astype(np.int64)
    ph = np.zeros((ny, nx), dtype=np.int64)
    ph[:, 1:] = np.cumsum(hdiff, axis=1)
    for row in range(ny):
        for lo in range(nx):
            for hi in range(lo, nx):
                assert len(segment_ids_on_row(table, row, lo, hi)) == \
                    int(ph[row, hi] - ph[row, lo])


def test_edge_segment_ids_walks_an_l_route_in_coordinates():
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=4))
    table = enumerate_segments(rg)
    # (10,10) is in region 0 (SW); (90,90) is in region 3 (NE).
    ids = edge_segment_ids(rg, table, 10., 10., 90., 90.)
    # The L-route goes horizontally at y=10 (crossing the 0|1 boundary) then
    # vertically at x=90 (crossing the 1|3 boundary).
    assert len(ids) == 2
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s])) for s in ids]
    assert pairs == [(0, 1), (1, 3)]


def test_segment_utilisation_reports_the_capacity_scalars():
    demand = np.array([0, 5, 12, 3, 1], dtype=np.int64)
    capacity = np.array([10., 10., 10., 0., 0.], dtype=np.float64)
    util, scalars = segment_utilisation(demand, capacity)
    np.testing.assert_allclose(util[:3], [0., 0.5, 1.2])
    assert util[3] == np.inf and util[4] == np.inf
    assert tuple(scalars) == CAPACITY_SCALARS
    assert scalars["num_segments"] == 5
    assert scalars["segment_demand_total"] == 21
    # 12 > 10, and both zero-capacity segments carry demand
    assert scalars["num_over_capacity"] == 3
    assert scalars["num_zero_capacity_segments"] == 2
    assert scalars["zero_capacity_demand"] == 4
    assert scalars["max_util"] == pytest.approx(1.2)
    assert scalars["p99_util"] == pytest.approx(np.percentile([0., 0.5, 1.2], 99))


def test_segment_utilisation_never_divides_by_a_zero_capacity():
    """Unit rule: zero-capacity segments stay blocked with a finite penalty and
    no epsilon is substituted -- so no NaN, no inf/inf, no 1e-9 anywhere."""
    util, scalars = segment_utilisation(np.zeros(3, dtype=np.int64),
                                        np.zeros(3, dtype=np.float64))
    assert np.array_equal(util, np.zeros(3))
    assert scalars["num_over_capacity"] == 0
    assert scalars["max_util"] == 0. and scalars["p99_util"] == 0.


def test_select_candidates_caps_groups_and_segments():
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20))
    table = enumerate_segments(rg)
    # Six distinct (u, v, pair(seg)) groups for net 0, one segment each,
    # with descending counts; m_pairs=4 must keep the four heaviest.
    segs, seen = [], set()
    for s in range(table.num_segments):
        key = (int(table.pair_a[s]), int(table.pair_b[s]))
        if key not in seen:
            seen.add(key)
            segs.append(s)
        if len(segs) == 6:
            break
    net = np.zeros(6, dtype=np.int64)
    u = np.array([int(table.pair_a[s]) for s in segs], dtype=np.int64)
    v = np.array([int(table.pair_b[s]) for s in segs], dtype=np.int64)
    seg = np.asarray(segs, dtype=np.int64)
    count = np.array([60, 50, 40, 30, 20, 10], dtype=np.int64)
    cand = select_candidates(net, u, v, seg, count, table, m_pairs=4, m_seg=2)
    assert cand.n_groups == 4
    assert sorted(cand.count.tolist()) == [30, 40, 50, 60]


def test_select_candidates_keeps_two_alternatives_on_one_boundary_pair():
    """Three segments on the same boundary pair for the same net: one group,
    the two heaviest kept (m_seg=2), ties broken by segment id."""
    table = enumerate_segments(RegionGrid(notch()))
    same = [s for s in range(table.num_segments)
            if (int(table.pair_a[s]), int(table.pair_b[s])) == (0, 1)
            and int(table.orient[s]) == ORIENT_V]
    assert len(same) == 3
    net = np.zeros(3, dtype=np.int64)
    u = np.zeros(3, dtype=np.int64)
    v = np.ones(3, dtype=np.int64)
    seg = np.asarray(same, dtype=np.int64)
    count = np.array([1, 1, 5], dtype=np.int64)
    cand = select_candidates(net, u, v, seg, count, table)
    assert cand.n_groups == 1
    # the 5-count one, plus the lower-id of the two tied 1-counts
    assert sorted(cand.seg.tolist()) == sorted([same[2], min(same[0], same[1])])
    assert np.array_equal(cand.group, np.zeros(2, dtype=np.int64))


def test_select_candidates_is_order_independent():
    table = enumerate_segments(RegionGrid(plug()))
    net = np.array([0, 0, 1, 1], dtype=np.int64)
    u = np.array([0, 0, 1, 0], dtype=np.int64)
    v = np.array([1, 2, 2, 2], dtype=np.int64)
    seg = np.array([0, 1, 3, 1], dtype=np.int64)
    count = np.array([3, 7, 2, 9], dtype=np.int64)
    a = select_candidates(net, u, v, seg, count, table)
    order = np.array([2, 0, 3, 1])
    b = select_candidates(net[order], u[order], v[order], seg[order],
                          count[order], table)
    for name in ("net", "u", "v", "seg", "group", "count"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.n_groups == b.n_groups


def test_remap_nets_drops_inactive_nets_and_recompacts_groups():
    table = enumerate_segments(RegionGrid(plug()))
    cand = select_candidates(np.array([0, 5, 5], dtype=np.int64),
                             np.array([0, 0, 1], dtype=np.int64),
                             np.array([1, 2, 2], dtype=np.int64),
                             np.array([0, 1, 3], dtype=np.int64),
                             np.array([1, 1, 1], dtype=np.int64), table)
    mapping = np.full(6, -1, dtype=np.int64)
    mapping[5] = 0                       # only global net 5 is active
    remapped = cand.remap_nets(mapping)
    assert np.array_equal(remapped.net, np.zeros(2, dtype=np.int64))
    assert remapped.n_groups == 2
    assert np.array_equal(remapped.group, np.array([0, 1]))
    assert select_candidates(np.zeros(0, dtype=np.int64), *(
        np.zeros(0, dtype=np.int64) for _ in range(4)), table).is_empty()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_region_segments.py -v -k "leg or util or candidate or remap"`
Expected: FAIL — `ImportError: cannot import name 'segment_ids_on_row' from 'ioplace.region_segments'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/region_segments.py`:

```python
def segment_ids_on_row(table, row, lo, hi):
    """Vertical segments crossed by a horizontal leg in grid row `row` running
    between grid columns `lo` and `hi` (inclusive, either order). The leg
    crosses the column boundaries j with min(lo,hi) <= j < max(lo,hi), so this
    is one contiguous CSR slice and costs the number of crossings, not the
    length of the leg (sec 5)."""
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    s0, s1 = int(table.row_ptr[row]), int(table.row_ptr[row + 1])
    cols = table.row_col[s0:s1]
    i0 = int(np.searchsorted(cols, a, side="left"))
    i1 = int(np.searchsorted(cols, b, side="left"))
    return table.row_seg[s0 + i0:s0 + i1]


def segment_ids_on_col(table, col, lo, hi):
    """Horizontal segments crossed by a vertical leg in grid column `col`."""
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    s0, s1 = int(table.col_ptr[col]), int(table.col_ptr[col + 1])
    rows = table.col_row[s0:s1]
    i0 = int(np.searchsorted(rows, a, side="left"))
    i1 = int(np.searchsorted(rows, b, side="left"))
    return table.col_seg[s0 + i0:s0 + i1]


def edge_segment_ids(rg, table, x0, y0, x1, y1):
    """Segments crossed by the L-route (x0,y0) -> (x1,y0) -> (x1,y1), in that
    order -- the same geometry `evaluator_ref.edge_regions_and_crossings`
    walks, so `len(...)` equals that function's crossing count."""
    ax, ay = rg.to_idx(np.asarray([x0], dtype=np.float64),
                       np.asarray([y0], dtype=np.float64))
    bx, by = rg.to_idx(np.asarray([x1], dtype=np.float64),
                       np.asarray([y1], dtype=np.float64))
    horizontal = segment_ids_on_row(table, int(ay[0]), int(ax[0]), int(bx[0]))
    vertical = segment_ids_on_col(table, int(bx[0]), int(ay[0]), int(by[0]))
    return np.concatenate([horizontal, vertical])


def segment_utilisation(demand, capacity):
    """(util, scalars) from per-segment demand and capacity.

    Unit rule (round-feedback spec:169-196): a zero-capacity segment is
    blocked, not scaled -- its utilisation is reported as `inf` when it
    carries demand and 0.0 when it does not, and it is *excluded* from
    max_util/p99_util (which are the statistics of the routable segments) but
    *counted* in num_over_capacity and reported separately. No epsilon is
    substituted anywhere.

    Both evaluators call this one function on the same bit-exact
    `segment_demand`, which is what makes segment_util/max_util/p99_util
    bit-exact across CPU and GPU rather than merely close."""
    demand = np.asarray(demand, dtype=np.int64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if demand.shape != capacity.shape:
        raise ValueError("segment demand and capacity must have the same shape")
    positive = capacity > 0.0
    util = np.zeros(demand.shape, dtype=np.float64)
    np.divide(demand, capacity, out=util, where=positive)
    util[(~positive) & (demand > 0)] = np.inf
    routable = util[positive]
    scalars = {
        "num_segments": int(demand.size),
        "segment_demand_total": int(demand.sum()),
        "num_over_capacity": int((demand > capacity).sum()),
        "num_zero_capacity_segments": int((~positive).sum()),
        "zero_capacity_demand": int(demand[~positive].sum()),
        "max_util": float(routable.max()) if routable.size else 0.0,
        "p99_util": float(np.percentile(routable, 99)) if routable.size else 0.0,
    }
    assert tuple(scalars) == CAPACITY_SCALARS, "CAPACITY_SCALARS drifted"
    return util, scalars


@dataclass
class Candidates:
    """The capped per-net candidate list the capacity term differentiates.

    One row per (net, demand pair (u,v), segment). `group` is the dense id of
    the alpha-softmax group (net, u, v, boundary pair of seg) -- see
    interpretation D-2 in the plan: alternatives on one boundary pair share a
    group and a single w*q_u*q_v; a feed-through's entry and exit segments sit
    in different groups and each take the full product."""

    net: np.ndarray       # (M,) int64
    u: np.ndarray         # (M,) int64, demand-pair low region
    v: np.ndarray         # (M,) int64, demand-pair high region
    seg: np.ndarray       # (M,) int64
    group: np.ndarray     # (M,) int64, dense and non-decreasing
    count: np.ndarray     # (M,) int64, observed crossings (diagnostics only)
    n_groups: int

    def is_empty(self):
        return int(self.net.shape[0]) == 0

    def remap_nets(self, mapping):
        """Translate global net ids through `mapping` (an (n_nets,) int64 array
        with -1 for nets the differentiable term does not carry -- the IO CSR
        drops nets above `ignore_net_degree` and nets collapsing to one node,
        `ops/io_term.build_net_node_csr`), dropping the rest and recompacting
        the group ids."""
        mapping = np.asarray(mapping, dtype=np.int64)
        new_net = mapping[self.net]
        keep = new_net >= 0
        group = self.group[keep]
        _uniq, dense = np.unique(group, return_inverse=True)
        return Candidates(net=new_net[keep], u=self.u[keep], v=self.v[keep],
                          seg=self.seg[keep], group=dense.astype(np.int64),
                          count=self.count[keep], n_groups=int(_uniq.size))


def _rank_within(owner, order):
    """Position of each element within its `owner` run, for an array already
    sorted so that equal owners are contiguous and the desired ranking order
    holds inside each run. `order` is that sort permutation."""
    owner_sorted = owner[order]
    n = owner_sorted.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    new_run[1:] = owner_sorted[1:] != owner_sorted[:-1]
    positions = np.arange(n, dtype=np.int64)
    run_start = np.maximum.accumulate(np.where(new_run, positions, 0))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = positions - run_start
    return rank


def select_candidates(net, u, v, seg, count, table, m_pairs=4, m_seg=2):
    """Cap a net's crossed (demand pair, segment) items at `m_pairs` groups and
    `m_seg` segments per group (spec sec 5's m_pairs=4 / m_seg=2 budget, read
    through interpretation D-2).

    Inputs are the evaluator's per-(net, u, v, seg) crossing counts with u < v
    and u != v; both evaluators produce them in the same ascending composite-key
    order, so this function's output is bit-identical between them. Every
    tie-break is total: groups by (-total count, group key), segments within a
    group by (-count, segment id)."""
    net = np.asarray(net, dtype=np.int64)
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    seg = np.asarray(seg, dtype=np.int64)
    count = np.asarray(count, dtype=np.int64)
    if net.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return Candidates(net=z, u=z, v=z, seg=z, group=z, count=z, n_groups=0)
    if (u >= v).any():
        raise ValueError("candidate demand pairs must satisfy u < v "
                         "(same-region legs are dropped by the caller)")
    k = np.int64(table.k)
    spair = table.pair_key()[seg]
    # (net, u, v, boundary pair) -> dense group id, ascending by that tuple
    gkey = ((net * k + u) * k + v) * (k * k) + spair
    uniq_g, gid = np.unique(gkey, return_inverse=True)
    gid = gid.astype(np.int64)
    gtot = np.bincount(gid, weights=count.astype(np.float64),
                       minlength=uniq_g.size).astype(np.int64)
    gnet = np.zeros(uniq_g.size, dtype=np.int64)
    gnet[gid] = net

    # ---- cap the number of groups per net ----
    g_order = np.lexsort((uniq_g, -gtot, gnet))
    g_rank = _rank_within(gnet, g_order)
    group_keep = g_rank < m_pairs

    # ---- cap the number of segments per kept group ----
    keep = group_keep[gid]
    i_order = np.lexsort((seg[keep], -count[keep], gid[keep]))
    i_rank = _rank_within(gid[keep], i_order)
    item_keep = i_rank < m_seg

    idx = np.nonzero(keep)[0][item_keep]
    idx = idx[np.lexsort((seg[idx], gid[idx]))]          # (group, seg) ascending
    _uniq, dense = np.unique(gid[idx], return_inverse=True)
    return Candidates(net=net[idx], u=u[idx], v=v[idx], seg=seg[idx],
                      group=dense.astype(np.int64), count=count[idx],
                      n_groups=int(_uniq.size))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_region_segments.py -v`
Expected: PASS (all of Task 1's plus the nine new ones).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/region_segments.py tests/test_region_segments.py
git commit -m "$(cat <<'MSG'
feat(capacity): CSR leg lookup, utilisation scalars, candidate capping

segment_ids_on_row/col slice the per-row/column CSR so an L-route leg costs
its crossing count, not its length (v2 design sec 5). segment_utilisation is
the single helper both evaluators call, which is what makes segment_util /
max_util / p99_util bit-exact across CPU and GPU. select_candidates caps a
net at m_pairs=4 alpha-groups x m_seg=2 segments with total tie-breaks.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: One-time capacity extraction and `capacity.npz`

**Files:**
- Create: `src/ioplace/capacity/__init__.py`, `src/ioplace/capacity/extract.py`
- Create: `tests/data/capacity/tiny_resources.json` (hand-written GCell fixture)
- Test: `tests/test_capacity_extract.py`

**Interfaces:**
- Consumes: Task 1's `SegmentTable`/`ORIENT_V`/`ORIENT_H`/`segments_digest`; `ioplace.route_eval.online_openroad.run_openroad(def_path, lefs, out, binary, *, congestion_iterations, allow_congestion, threads, signal_layers)` and `digest(path)`; `ioplace.regions.RegionSet`; `ioplace.region_grid.RegionGrid`.
- Produces:
  - `CAPACITY_SEMANTICS` (the verbatim string), `CAPACITY_SCHEMA_VERSION = 1`
  - `parse_tech_lef_layers(path) -> list` of `{"name", "pitch", "direction", "index"}` for `TYPE ROUTING` layers, in LEF order
  - `track_density(layers, direction, layer_range=None) -> float` (tracks per micron)
  - `lef_capacity(table, layers, *, scale_factor, dbu_per_micron, layer_range=("metal2", "metal10")) -> (S,) float64`
  - `gcell_capacity(table, resources, *, shift_factor=(0., 0.), scale_factor=1., chunk=1_000_000) -> (S,) float64`
  - `run_extraction(def_path, lefs, out_dir, binary, *, congestion_iterations=5, signal_layers="metal2-metal10", threads=4) -> (resources_dict, receipt_sha256)`
  - `save_capacity(path, table, capacity, *, source, receipt_sha256=None, extra=None) -> metadata_dict`
  - `load_capacity(path, table=None) -> dict` with keys `capacity`, `metadata`, and the stored segment arrays
  - `main(argv=None)` behind `python -m ioplace.capacity.extract`

**Why the GCell grid is read exactly once.** Unit rule, verbatim in Global Constraints. `run_extraction` is the only code in P-D that imports `route_eval`, it runs before global placement, and its entire output downstream is one float per segment id. Nothing in `ops/cap_term.py` or either evaluator ever sees `x_edges_dbu`.

**Why the conversion is an overlap-weighted row sum.** Spec §5: *"for a vertical segment at `x=X` over `[y0,y1]`, `C_seg = Σ_rows overlap(row,[y0,y1])/row_h · hcap[row][col(X)]` — horizontal-preferred-direction layers only, blockages already folded in by OpenDB."* `dump_online_route.py:96-108` builds exactly `hcap[y][x]` (shape `(ny, nx-1)`, horizontal-preferred layers summed) and `vcap[y][x]` (shape `(ny-1, nx)`, vertical), so the two arrays are already the per-GCell preferred-direction track counts a cut consumes. `col(X)` is the GCell **column containing X**, clipped into the edge-index range: `clip(searchsorted(x_edges, X, "right") - 1, 0, nx-2)`. Rejected: taking the min of the two adjacent GCell edges (pessimistic and non-monotone in X) and interpolating between them (invents fractional tracks the router cannot use).

**Known inaccuracy to record, not fix (spec §10 risk 5).** OpenDB `getCapacity` is uint8-clamped — tile measured 7,871,705 against a native 7,872,067 (`docs/results/2026-09-15-ggr-trial.md:39-40`). `dump_online_route.py` already reports `uint8_backend`; `save_capacity` copies it into the metadata so every downstream number carries the caveat. `benchmarks/ispd25/visible/*.cap` is **not** a capacity source (`docs/results/2026-09-15-benchmark-router-diagnosis.md:58-69`) and must not appear anywhere in this task.

- [ ] **Step 1: Write the fixture and the failing tests**

Create `tests/data/capacity/tiny_resources.json` — a 3×3-GCell grid over a 3000×3000 DBU die, shaped exactly like `dump_online_route.py`'s output:

```json
{
  "x_edges_dbu": [0, 1000, 2000, 3000],
  "y_edges_dbu": [0, 1000, 2000, 3000],
  "horizontal_capacity": [[10, 20], [30, 40], [50, 60]],
  "vertical_capacity": [[7, 8, 9], [11, 12, 13]],
  "horizontal_usage": [[1, 1], [1, 1], [1, 1]],
  "vertical_usage": [[1, 1, 1], [1, 1, 1]],
  "layers": [{"name": "metal2", "direction": "VERTICAL"},
             {"name": "metal3", "direction": "HORIZONTAL"}],
  "capacity_semantics": "OpenDB total track capacity; usage includes blockages and wires; preferred-direction layers summed",
  "uint8_backend": true,
  "preferred_layer_overflow": 0
}
```

Create `tests/test_capacity_extract.py`:

```python
import json
import os

import numpy as np
import pytest

from ioplace.capacity.extract import (CAPACITY_SEMANTICS, gcell_capacity,
                                      lef_capacity, load_capacity,
                                      parse_tech_lef_layers, save_capacity,
                                      track_density)
from ioplace.region_grid import RegionGrid
from ioplace.region_segments import enumerate_segments, segments_digest
from ioplace.regions import make_grid_regions

DIE = (0., 0., 3000., 3000.)
TECH_LEF = ("/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/"
            "NangateOpenCellLibrary.tech.lef")
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "capacity",
                       "tiny_resources.json")


def _table():
    """2x2 regions on a lattice-4 grid over a 3000x3000 die: four segments,
    V at x=1500 over y in [0,1500] and [1500,3000], H at y=1500 over
    x in [0,1500] and [1500,3000]."""
    return enumerate_segments(RegionGrid(make_grid_regions(DIE, 2, 2, lattice=4)))


def test_gcell_capacity_is_the_overlap_weighted_row_sum():
    table = _table()
    with open(FIXTURE) as stream:
        resources = json.load(stream)
    capacity = gcell_capacity(table, resources, shift_factor=(0., 0.),
                              scale_factor=1.)
    # V at x=1500 -> GCell column 1 -> hcap[:,1] = [20,40,60].
    #   y in [0,1500]:    1.0*20 + 0.5*40           = 40
    #   y in [1500,3000]: 0.5*40 + 1.0*60           = 80
    # H at y=1500 -> GCell row 1 -> vcap[1,:] = [11,12,13].
    #   x in [0,1500]:    1.0*11 + 0.5*12           = 17
    #   x in [1500,3000]: 0.5*12 + 1.0*13           = 19
    np.testing.assert_allclose(capacity, [40., 80., 17., 19.])


def test_gcell_capacity_rejects_a_mismatched_resource_grid():
    table = _table()
    with open(FIXTURE) as stream:
        resources = json.load(stream)
    resources["horizontal_capacity"] = [[1, 2, 3]]
    with pytest.raises(ValueError, match="resource dimensions"):
        gcell_capacity(table, resources)


@pytest.mark.skipif(not os.path.exists(TECH_LEF), reason="NanGate45 tech LEF absent")
def test_nangate45_track_densities_come_from_the_real_pitches():
    layers = parse_tech_lef_layers(TECH_LEF)
    assert [layer["name"] for layer in layers] == \
        ["metal%d" % i for i in range(1, 11)]
    assert layers[0]["pitch"] == pytest.approx(0.14)
    assert layers[0]["direction"] == "HORIZONTAL"
    assert layers[9]["pitch"] == pytest.approx(1.6)
    assert layers[9]["direction"] == "VERTICAL"
    # metal2-metal10 (the sec 8 GRT routing-layer range):
    #   horizontal metal3/5/7/9 = 1/.14 + 1/.28 + 1/.8 + 1/1.6
    #   vertical  metal2/4/6/8/10 = 1/.19 + 1/.28 + 1/.28 + 1/.8 + 1/1.6
    assert track_density(layers, "HORIZONTAL", ("metal2", "metal10")) == \
        pytest.approx(12.589285714285714)
    assert track_density(layers, "VERTICAL", ("metal2", "metal10")) == \
        pytest.approx(14.280015037593985)
    # unrestricted, metal1 joins the horizontal set
    assert track_density(layers, "HORIZONTAL") == pytest.approx(19.732142857142854)
    assert track_density(layers, "VERTICAL") == pytest.approx(14.280015037593985)


@pytest.mark.skipif(not os.path.exists(TECH_LEF), reason="NanGate45 tech LEF absent")
def test_lef_fallback_is_rho_times_length_in_microns():
    table = _table()
    layers = parse_tech_lef_layers(TECH_LEF)
    capacity = lef_capacity(table, layers, scale_factor=1., dbu_per_micron=2000.,
                            layer_range=("metal2", "metal10"))
    # each segment is 1500 dbu = 0.75 um long; a vertical segment is crossed
    # by horizontal layers and vice versa (sec 5's "layers perpendicular to
    # the segment").
    np.testing.assert_allclose(capacity[:2], [12.589285714285714 * 0.75] * 2)
    np.testing.assert_allclose(capacity[2:], [14.280015037593985 * 0.75] * 2)


def test_capacity_npz_round_trips_with_a_receipt_hash(tmp_path):
    table = _table()
    capacity = np.array([40., 80., 17., 0.])
    path = tmp_path / "capacity.npz"
    metadata = save_capacity(path, table, capacity, source="openroad",
                             receipt_sha256="a" * 64,
                             extra={"uint8_backend": True,
                                    "congestion_iterations": 5})
    assert metadata["capacity_semantics"] == CAPACITY_SEMANTICS
    assert metadata["capacity_semantics"] == (
        "usable tracks crossing the segment; "
        "one net crossing consumes one track")
    assert metadata["capacity_source"] == "openroad"
    assert metadata["receipt_sha256"] == "a" * 64
    assert metadata["segments_sha256"] == segments_digest(table)
    assert metadata["zero_capacity_segments"] == 1
    assert metadata["uint8_backend"] is True

    loaded = load_capacity(path, table=table)
    np.testing.assert_array_equal(loaded["capacity"], capacity)
    assert loaded["metadata"] == metadata


def test_loading_against_a_different_geometry_fails_loudly(tmp_path):
    table = _table()
    path = tmp_path / "capacity.npz"
    save_capacity(path, table, np.ones(table.num_segments), source="lef_pitch")
    other = enumerate_segments(RegionGrid(make_grid_regions(DIE, 4, 4, lattice=4)))
    with pytest.raises(ValueError, match="segment table"):
        load_capacity(path, table=other)


def test_save_capacity_rejects_a_negative_or_nonfinite_capacity(tmp_path):
    table = _table()
    for bad in (np.array([1., -1., 1., 1.]), np.array([1., np.inf, 1., 1.])):
        with pytest.raises(ValueError, match="finite nonnegative"):
            save_capacity(tmp_path / "bad.npz", table, bad, source="lef_pitch")


def test_zero_capacity_is_preserved_not_floored(tmp_path):
    """Unit rule: zero-capacity segments stay blocked; no epsilon substitution
    anywhere in the extractor."""
    table = _table()
    path = tmp_path / "capacity.npz"
    save_capacity(path, table, np.zeros(table.num_segments), source="lef_pitch")
    loaded = load_capacity(path, table=table)
    assert (loaded["capacity"] == 0.).all()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("IOPLACE_OPENROAD_TESTS") != "1"
    or not os.access(os.environ.get("OPENROAD_BIN", ""), os.X_OK),
    reason="needs IOPLACE_OPENROAD_TESTS=1 and an executable $OPENROAD_BIN")
def test_real_openroad_extraction_on_gcd(tmp_path):
    """The one test that actually runs the router. Everything the extractor
    computes is already covered by the recorded fixture above; this checks the
    plumbing -- that run_openroad's resources.json still has the shape
    gcell_capacity expects, and that a real extraction produces finite,
    mostly-positive capacities."""
    import json as _json
    from ioplace.capacity.extract import run_extraction
    from ioplace.regions import RegionSet
    case = _json.load(open("results/route_feedback_20260914/gcd.json"))
    lefs = case["lef_input"]
    def_path = case["def_input"]
    resources, receipt_sha = run_extraction(
        def_path, lefs, tmp_path / "or", os.environ["OPENROAD_BIN"],
        congestion_iterations=5, signal_layers="metal2-metal10", threads=4)
    assert len(receipt_sha) == 64
    for key in ("x_edges_dbu", "y_edges_dbu", "horizontal_capacity",
                "vertical_capacity"):
        assert key in resources
    die = (0., 0., float(resources["x_edges_dbu"][-1]),
           float(resources["y_edges_dbu"][-1]))
    table = enumerate_segments(RegionGrid(
        __import__("ioplace.regions", fromlist=["make_grid_regions"])
        .make_grid_regions(die, 2, 2, lattice=8)))
    capacity = gcell_capacity(table, resources)
    assert np.isfinite(capacity).all() and (capacity >= 0).all()
    assert (capacity > 0).sum() >= table.num_segments // 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_capacity_extract.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.capacity'`.

- [ ] **Step 3: Write `src/ioplace/capacity/__init__.py`**

```python
"""One-time, offline boundary-capacity extraction (v2 design sec 5, P-D)."""
from ioplace.capacity.extract import (CAPACITY_SCHEMA_VERSION,
                                      CAPACITY_SEMANTICS, extract_capacity,
                                      load_capacity, save_capacity)

__all__ = ["CAPACITY_SCHEMA_VERSION", "CAPACITY_SEMANTICS", "extract_capacity",
           "load_capacity", "save_capacity"]
```

- [ ] **Step 4: Write `src/ioplace/capacity/extract.py`**

```python
"""Per-segment boundary capacity, extracted once before global placement.

Two sources (v2 design sec 5):

  openroad   one `global_route` on the *input* DEF with minimal congestion
             iterations and -allow_congestion, reusing
             `route_eval/or_scripts/dump_online_route.py:90-114` and
             `route_eval/online_openroad.py:44-159` verbatim. Only
             horizontal_capacity / vertical_capacity and the GCell edges are
             kept; `usage` is discarded, because it reflects the input
             placement and this run exists only to read resources.
  lef_pitch  the fallback C_seg = rho*l, rho = sum over the layers
             perpendicular to the segment of 1/pitch, from the tech LEF.

The unit rule (round-feedback spec:169-196) lives here: this module is the
only place in P-D that touches the GCell grid, it runs once, offline, and its
whole output is one scalar per segment id. Zero-capacity segments are written
out as zero and stay blocked downstream; no epsilon is substituted.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile

import numpy as np

from ioplace.region_segments import (ORIENT_V, SegmentTable, segments_digest)

CAPACITY_SCHEMA_VERSION = 1

# Spec sec 5, verbatim. Do not reword: downstream readers compare the string.
CAPACITY_SEMANTICS = ("usable tracks crossing the segment; "
                      "one net crossing consumes one track")

_SEGMENT_ARRAYS = ("orient", "line", "lo", "hi", "pair_a", "pair_b",
                   "length_units", "length")


# ---------------------------------------------------------------- tech LEF --
def parse_tech_lef_layers(path):
    """Routing layers of a tech LEF, in file order: name, PITCH (microns),
    DIRECTION. Only `TYPE ROUTING` layers are returned; MASTERSLICE, CUT and
    OVERLAP layers are skipped."""
    layers = []
    current = None
    with open(path) as stream:
        for raw in stream:
            line = raw.strip()
            match = re.match(r"^LAYER\s+(\S+)", line)
            if match is not None:
                current = {"name": match.group(1)}
                continue
            if current is None:
                continue
            if line.startswith("TYPE"):
                current["type"] = line.split()[1].rstrip(";").strip()
            elif line.startswith("PITCH"):
                current["pitch"] = float(line.split()[1])
            elif line.startswith("DIRECTION"):
                current["direction"] = line.split()[1].rstrip(";").strip()
            elif line.startswith("END "):
                if current.get("type") == "ROUTING":
                    if "pitch" not in current or "direction" not in current:
                        raise ValueError(
                            "routing layer %r has no PITCH/DIRECTION" % (current["name"],))
                    current["index"] = len(layers)
                    layers.append(current)
                current = None
    if not layers:
        raise ValueError("no TYPE ROUTING layers in %s" % (path,))
    return layers


def track_density(layers, direction, layer_range=None):
    """Tracks per micron available in `direction`, summed over the routing
    layers whose preferred DIRECTION matches. `layer_range` is an inclusive
    (first, last) layer-name pair -- pass ("metal2", "metal10") to match the
    sec 8 GRT protocol's `set_routing_layers metal2-metal10`."""
    selected = layers
    if layer_range is not None:
        names = [layer["name"] for layer in layers]
        try:
            lo, hi = names.index(layer_range[0]), names.index(layer_range[1])
        except ValueError:
            raise ValueError("layer range %r not in this LEF (%r)"
                             % (layer_range, names))
        if lo > hi:
            raise ValueError("layer range %r is inverted" % (layer_range,))
        selected = layers[lo:hi + 1]
    return float(sum(1.0 / layer["pitch"] for layer in selected
                     if layer["direction"] == direction))


def lef_capacity(table, layers, *, scale_factor, dbu_per_micron,
                 layer_range=("metal2", "metal10")):
    """Fallback C_seg = rho * l. A vertical segment is crossed by wires running
    horizontally, so its rho is the HORIZONTAL track density, and vice versa
    (sec 5: "rho = sum over layers perpendicular to seg of 1/pitch").

    `table.length` is in the RegionGrid's own coordinate frame, which is DEF
    DBU scaled by `scale_factor` (`route_eval/online_openroad.load_observation`
    uses the same (dbu - shift) * scale convention), so microns are
    length / (scale_factor * dbu_per_micron)."""
    if scale_factor <= 0 or dbu_per_micron <= 0:
        raise ValueError("positive scale_factor and dbu_per_micron required")
    rho_h = track_density(layers, "HORIZONTAL", layer_range)
    rho_v = track_density(layers, "VERTICAL", layer_range)
    microns = table.length / (float(scale_factor) * float(dbu_per_micron))
    rho = np.where(table.orient == ORIENT_V, rho_h, rho_v)
    return (rho * microns).astype(np.float64)


# ------------------------------------------------------------- GCell grid --
def gcell_capacity(table, resources, *, shift_factor=(0.0, 0.0),
                   scale_factor=1.0, chunk=1_000_000):
    """Convert one `dump_online_route.py` resources payload into a per-segment
    capacity (sec 5's overlap-weighted row/column sum). `usage` is ignored by
    construction -- it is never read here."""
    shift = np.asarray(shift_factor, dtype=np.float64).reshape(2)
    scale = float(scale_factor)
    x_edges = (np.asarray(resources["x_edges_dbu"], dtype=np.float64) - shift[0]) * scale
    y_edges = (np.asarray(resources["y_edges_dbu"], dtype=np.float64) - shift[1]) * scale
    hcap = np.asarray(resources["horizontal_capacity"], dtype=np.float64)
    vcap = np.asarray(resources["vertical_capacity"], dtype=np.float64)
    nx, ny = x_edges.size - 1, y_edges.size - 1
    if nx < 2 or ny < 2:
        raise ValueError("invalid OpenDB resource dimensions: need >= 2 GCells "
                         "per axis, got %dx%d" % (nx, ny))
    if hcap.shape != (ny, nx - 1) or vcap.shape != (ny - 1, nx):
        raise ValueError("invalid OpenDB resource dimensions: expected hcap "
                         "%r / vcap %r, got %r / %r"
                         % ((ny, nx - 1), (ny - 1, nx), hcap.shape, vcap.shape))
    if not (np.isfinite(hcap).all() and np.isfinite(vcap).all()):
        raise ValueError("nonfinite OpenDB capacity")
    row_lo, row_hi = y_edges[:-1], y_edges[1:]
    col_lo, col_hi = x_edges[:-1], x_edges[1:]
    out = np.zeros(table.num_segments, dtype=np.float64)

    v_idx = np.nonzero(table.orient == ORIENT_V)[0]
    step = max(1, chunk // max(ny, 1))
    for start in range(0, v_idx.size, step):
        sub = v_idx[start:start + step]
        x = table.box[sub, 0]
        y0, y1 = table.box[sub, 1], table.box[sub, 3]
        col = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 2)
        overlap = np.clip(np.minimum(row_hi[None, :], y1[:, None])
                          - np.maximum(row_lo[None, :], y0[:, None]), 0.0, None)
        out[sub] = (overlap / (row_hi - row_lo)[None, :] * hcap[:, col].T).sum(axis=1)

    h_idx = np.nonzero(table.orient != ORIENT_V)[0]
    step = max(1, chunk // max(nx, 1))
    for start in range(0, h_idx.size, step):
        sub = h_idx[start:start + step]
        y = table.box[sub, 1]
        x0, x1 = table.box[sub, 0], table.box[sub, 2]
        row = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 2)
        overlap = np.clip(np.minimum(col_hi[None, :], x1[:, None])
                          - np.maximum(col_lo[None, :], x0[:, None]), 0.0, None)
        out[sub] = (overlap / (col_hi - col_lo)[None, :] * vcap[row, :]).sum(axis=1)

    return out


def run_extraction(def_path, lefs, out_dir, binary, *, congestion_iterations=5,
                   signal_layers="metal2-metal10", threads=4):
    """One OpenROAD process on the input DEF. `-allow_congestion` is mandatory
    and the iteration count is deliberately small: unbounded congestion removal
    cost ~9-10 h on tile/group, while 5 iterations routed tile in 221.9 s
    (`docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43`). We are
    reading resources, not producing a route, so a congested result is fine.

    Returns (resources_payload, sha256 of the receipt) -- the receipt is the
    provenance record save_capacity stores."""
    from ioplace.route_eval.online_openroad import digest, run_openroad
    out_dir = str(out_dir)
    run_openroad(str(def_path), [str(p) for p in lefs], out_dir, str(binary),
                 congestion_iterations=int(congestion_iterations),
                 allow_congestion=True, threads=int(threads),
                 signal_layers=signal_layers)
    with open(os.path.join(out_dir, "resources.json")) as stream:
        resources = json.load(stream)
    return resources, digest(os.path.join(out_dir, "receipt.json"))


# ------------------------------------------------------------------- I/O ---
def save_capacity(path, table, capacity, *, source, receipt_sha256=None,
                  extra=None):
    """Write capacity.npz: the segment table's defining arrays (so the file is
    self-describing), the per-segment capacity, and the metadata sec 5 asks
    for -- capacity_source, the OpenROAD receipt SHA-256, and the verbatim
    semantics string."""
    if source not in ("openroad", "lef_pitch"):
        raise ValueError("capacity_source must be 'openroad' or 'lef_pitch', "
                         "got %r" % (source,))
    capacity = np.asarray(capacity, dtype=np.float64)
    if capacity.shape != (table.num_segments,):
        raise ValueError("capacity must carry one value per segment (%d), got %r"
                         % (table.num_segments, capacity.shape))
    if not np.isfinite(capacity).all() or (capacity < 0).any():
        raise ValueError("finite nonnegative capacity required")
    metadata = dict(
        schema_version=CAPACITY_SCHEMA_VERSION,
        segment_schema_version=int(__import__(
            "ioplace.region_segments", fromlist=["SEGMENT_SCHEMA_VERSION"]
        ).SEGMENT_SCHEMA_VERSION),
        capacity_source=source,
        capacity_semantics=CAPACITY_SEMANTICS,
        receipt_sha256=receipt_sha256,
        segments_sha256=segments_digest(table),
        num_segments=int(table.num_segments),
        zero_capacity_segments=int((capacity == 0.0).sum()),
        k=int(table.k), lattice=int(table.lattice), die=list(table.die),
    )
    metadata.update(extra or {})
    arrays = {"seg_" + name: np.ascontiguousarray(getattr(table, name))
              for name in _SEGMENT_ARRAYS}
    arrays["capacity"] = capacity
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    parent = os.path.dirname(os.path.abspath(str(path)))
    os.makedirs(parent, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".capacity-", suffix=".npz",
                                         dir=parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def load_capacity(path, table=None):
    """Read capacity.npz. With `table`, verify the segment fingerprint -- a
    capacity file paired with a different geometry is the single most damaging
    silent failure in P-D, so it is a hard error."""
    with np.load(str(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data.pop("metadata")))
    if metadata.get("schema_version") != CAPACITY_SCHEMA_VERSION:
        raise ValueError("unsupported capacity schema %r"
                         % (metadata.get("schema_version"),))
    if metadata.get("capacity_semantics") != CAPACITY_SEMANTICS:
        raise ValueError("capacity_semantics does not match this build")
    capacity = np.asarray(data["capacity"], dtype=np.float64)
    if capacity.shape != (metadata["num_segments"],):
        raise ValueError("capacity array does not match num_segments")
    if table is not None and segments_digest(table) != metadata["segments_sha256"]:
        raise ValueError("capacity.npz was built for a different segment table")
    data["capacity"] = capacity
    data["metadata"] = metadata
    return data


def extract_capacity(table, *, source="auto", def_path=None, lefs=(),
                     openroad_bin=None, or_out=None, tech_lef=None,
                     dbu_per_micron=2000.0, layer_range=("metal2", "metal10"),
                     shift_factor=(0.0, 0.0), scale_factor=1.0,
                     congestion_iterations=5, signal_layers="metal2-metal10",
                     threads=4):
    """Dispatch. `auto` prefers OpenROAD and records why it fell back."""
    if source not in ("auto", "openroad", "lef_pitch"):
        raise ValueError("source must be auto|openroad|lef_pitch")
    can_route = bool(def_path and lefs and openroad_bin and or_out)
    if source in ("auto", "openroad") and can_route:
        try:
            resources, receipt = run_extraction(
                def_path, lefs, or_out, openroad_bin,
                congestion_iterations=congestion_iterations,
                signal_layers=signal_layers, threads=threads)
            capacity = gcell_capacity(table, resources,
                                      shift_factor=shift_factor,
                                      scale_factor=scale_factor)
            extra = {"uint8_backend": bool(resources.get("uint8_backend", False)),
                     "opendb_capacity_semantics": resources.get("capacity_semantics"),
                     "congestion_iterations": int(congestion_iterations),
                     "signal_layers": signal_layers}
            return capacity, "openroad", receipt, extra
        except Exception as error:
            if source == "openroad":
                raise
            fallback_reason = "openroad extraction failed: %s" % (error,)
    elif source == "openroad":
        raise ValueError("source='openroad' needs def_path, lefs, openroad_bin "
                         "and or_out")
    else:
        fallback_reason = "no OpenROAD inputs supplied"
    if tech_lef is None:
        raise ValueError("the lef_pitch fallback needs --tech-lef")
    layers = parse_tech_lef_layers(tech_lef)
    capacity = lef_capacity(table, layers, scale_factor=scale_factor,
                            dbu_per_micron=dbu_per_micron,
                            layer_range=layer_range)
    extra = {"fallback_reason": fallback_reason,
             "tech_lef": os.path.abspath(tech_lef),
             "dbu_per_micron": float(dbu_per_micron),
             "layer_range": list(layer_range),
             "rho_h": track_density(layers, "HORIZONTAL", layer_range),
             "rho_v": track_density(layers, "VERTICAL", layer_range)}
    return capacity, "lef_pitch", None, extra


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m ioplace.capacity.extract",
        description="Extract per-segment boundary IO capacity (v2 design sec 5)")
    parser.add_argument("--regions", required=True, help="regions.json")
    parser.add_argument("--out", required=True, help="capacity.npz to write")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "openroad", "lef_pitch"])
    parser.add_argument("--def", dest="def_path", default=None)
    parser.add_argument("--lef", action="append", default=[])
    parser.add_argument("--openroad-bin", default=os.environ.get("OPENROAD_BIN"))
    parser.add_argument("--or-out", default=None,
                        help="fresh directory for the OpenROAD run (must not exist)")
    parser.add_argument("--congestion-iterations", type=int, default=5)
    parser.add_argument("--signal-layers", default="metal2-metal10")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tech-lef", default=None)
    parser.add_argument("--dbu-per-micron", type=float, default=2000.)
    parser.add_argument("--layer-range", default="metal2-metal10")
    parser.add_argument("--shift-factor", default="0,0")
    parser.add_argument("--scale-factor", type=float, default=1.)
    return parser


def main(argv=None):
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import RegionSet
    args = build_parser().parse_args(argv)
    region_set = RegionSet.from_json(args.regions)
    region_set.validate()
    table = enumerate_segments(RegionGrid(region_set))
    shift = tuple(float(v) for v in args.shift_factor.split(","))
    first, _, last = args.layer_range.partition("-")
    capacity, source, receipt, extra = extract_capacity(
        table, source=args.source, def_path=args.def_path, lefs=args.lef,
        openroad_bin=args.openroad_bin, or_out=args.or_out,
        tech_lef=args.tech_lef, dbu_per_micron=args.dbu_per_micron,
        layer_range=(first, last), shift_factor=shift,
        scale_factor=args.scale_factor,
        congestion_iterations=args.congestion_iterations,
        signal_layers=args.signal_layers, threads=args.threads)
    metadata = save_capacity(args.out, table, capacity, source=source,
                             receipt_sha256=receipt, extra=extra)
    print(json.dumps(metadata, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Replace the awkward `__import__("ioplace.region_segments", ...)` in `save_capacity` with a plain module-level `from ioplace.region_segments import SEGMENT_SCHEMA_VERSION` added to the existing import line, and use `int(SEGMENT_SCHEMA_VERSION)`; the `__import__` form above is only there to keep the import list in one place while reading. Likewise drop the unused `SegmentTable` and `hashlib` imports if flake8 is run.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_capacity_extract.py -v -m "not slow"`
Expected: PASS (8 tests).

- [ ] **Step 6: Run the real extraction once, by hand, and record the receipt**

```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
export IOPLACE_OPENROAD_TESTS=1
"$IOPLACE_PYTHON" -m pytest tests/test_capacity_extract.py -v -m slow
```

If `$OPENROAD_BIN` is unset or not executable the test skips and that is an acceptable outcome for this task — but say so in the commit body, because Task 10's experiment needs a real extraction and will have to do it then.

- [ ] **Step 7: Commit**

```bash
git add src/ioplace/capacity tests/test_capacity_extract.py tests/data/capacity
git commit -m "$(cat <<'MSG'
feat(capacity): one-time OpenROAD / LEF-pitch capacity extraction

Reuse dump_online_route.py's resource dump on the input DEF with minimal
congestion iterations and -allow_congestion, keep only the H/V GCell
capacities, and convert them into one scalar per segment id by the
overlap-weighted row sum (v2 design sec 5). rho*l from the tech LEF is the
fallback. capacity.npz records capacity_source, the receipt SHA-256 and the
verbatim capacity_semantics string; zero capacities are preserved, never
floored. The GCell grid is touched here and nowhere else (unit rule).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Penalty, α softmax, and the dense reference term

**Files:**
- Create: `src/ioplace/ops/cap_term.py`
- Test: `tests/test_capacity_term.py`

**Interfaces:**
- Consumes: Task 1/2's `SegmentTable`, `Candidates`, `enumerate_segments`, `select_candidates`; `ioplace.ops.io_term.IoTerm`/`IoTermRef`/`build_net_node_csr`; `ioplace.ops.soft_assign.{rect_table, softmax_stats, region_sdf_l1, chunk_p_ell, _chunks}`.
- Produces:
  - `cap_penalty(d)`, `cap_penalty_grad(d)` — elementwise, torch or numpy
  - `normalised_overflow(demand, capacity) -> (d, inv)` where `inv = ∂d/∂D`
  - `segment_softmax(u, group, n_groups) -> alpha`
  - `l1_point_box_dist(cx, cy, box) -> d1` and `l1_point_box_grad(cx, cy, box) -> (ddx, ddy)`
  - `class CapTermRef(torch.nn.Module)` with `__init__(io_term, seg_box, seg_capacity, *, tau_b)`, `set_candidates(cand)`, `forward(pos, tau, lambda_cap)`, `demand(pos, tau)`
  - `CAP_CURVATURE_DREF = 1.0` and `cap_curvature(d_ref=CAP_CURVATURE_DREF) -> float`

**The penalty, verbatim (spec §5).** `pen(d) = 2[d]_+³ + [d]_+²`, `d_s = (D_s − C_s)/C_s`. `pen(d) = 0` exactly for `d ≤ 0` — not "approximately zero", not softplus; that is the whole reason §5 rejects phase-1 §5.4's `softplus((D−C)/C)`, whose `softplus(−1) = 0.31` exerts force at half capacity. `pen'(d) = 6[d]_+² + 2[d]_+`, so `pen'(0) = 0` from both sides: **C¹**. `pen''(0⁺) = 2` and `pen''(0⁻) = 0`, so it is deliberately *not* C² — the tests pin both facts.

**Zero-capacity segments (unit rule).** `d_s = (D_s − C_s)/C_s` is undefined at `C_s = 0`, and the unit rule forbids an epsilon. The reading this plan implements, recorded as **interpretation D-3**: when `C_s = 0`, `d_s := D_s` and `∂d_s/∂D_s := 1`. The penalty is then `2D_s³ + D_s²` — finite, C¹, exactly zero at zero demand, monotone, and needing no substitution. It is the only choice that keeps the penalty expression itself unchanged. Rejected: `C_s ← max(C_s, ε)` (an epsilon by another name, and it makes the penalty scale with `1/ε³`); `C_s ← 1` (a silent capacity grant); dropping the segment (removes the block the unit rule demands).

**The declared curvature (spec §4: `curvature=<max_s pen''>`).** `pen''(d) = 12d + 2` is unbounded above, so there is no true `max_s pen''`. `cap_curvature(d_ref)` returns `12·d_ref + 2` with `d_ref = 1.0` (100% over capacity — the overflow level the design is willing to tolerate before the cubic takes over), i.e. **14.0**. `TermConfig.curvature` only feeds `cmax_from_curvatures`' λ-weighted *mean* over active terms (`norm.py:75-89`), so a representative value is what the cap wants, not a supremum. The driver logs the largest observed `d_s` every probe (`cap_max_d`, Task 8) so the choice is auditable against data rather than assumed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capacity_term.py`:

```python
import numpy as np
import pytest
import torch

from ioplace.ops.cap_term import (CAP_CURVATURE_DREF, CapTermRef, cap_curvature,
                                  cap_penalty, cap_penalty_grad,
                                  l1_point_box_dist, l1_point_box_grad,
                                  normalised_overflow, segment_softmax)
from ioplace.ops.io_term import IoTerm, build_net_node_csr
from ioplace.ops.soft_assign import rect_table
from ioplace.region_grid import RegionGrid
from ioplace.region_segments import enumerate_segments, select_candidates
from ioplace.regions import make_grid_regions
from tests.test_io_term import _nl

DIE = (0., 0., 90., 30.)


def _strip(k=3):
    """A 1 x k strip of regions on a lattice-9 grid: region 0 spans x in
    [0,30), region 1 [30,60), region 2 [60,90). Adjacent pairs are (0,1) and
    (1,2), so a net with pins in 0 and 2 feeds through 1."""
    return make_grid_regions(DIE, k, 1, lattice=9)


def _setup(node_xy, nets, k=3, chunk=4):
    rs = _strip(k)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    nl = _nl(node_xy, nets)
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                num_movable=len(node_xy), num_physical=nl.num_physical,
                num_nodes=nl.num_physical, device="cpu",
                chunk_budget=max(1, chunk) * max(1, len(csr.flat_net2node)))
    return nl, rg, table, io, csr


def _pos(nl):
    n = nl.num_physical
    p = torch.zeros(2 * n, dtype=torch.float64)
    p[:n] = torch.as_tensor(nl.node_x)
    p[n:] = torch.as_tensor(nl.node_y)
    return p.requires_grad_(True)


# ---------------------------------------------------------------- penalty --
@pytest.mark.parametrize("d", [-10., -1., -1e-9, 0.])
def test_penalty_is_exactly_zero_below_capacity(d):
    """Not approximately zero: sec 5 rejects softplus precisely because it
    exerts force where nothing is violated."""
    assert cap_penalty(torch.tensor(d, dtype=torch.float64)).item() == 0.0
    assert cap_penalty_grad(torch.tensor(d, dtype=torch.float64)).item() == 0.0
    assert cap_penalty(np.float64(d)) == 0.0


def test_penalty_matches_the_spec_expression_above_capacity():
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    np.testing.assert_allclose(cap_penalty(d).numpy(),
                               [2 * .125 + .25, 2 + 1, 16 + 4])
    np.testing.assert_allclose(cap_penalty_grad(d).numpy(),
                               [6 * .25 + 1., 6 + 2, 24 + 4])


def test_penalty_is_c1_but_not_c2_at_the_knee():
    h = 1e-3
    at = lambda v: float(cap_penalty(torch.tensor(v, dtype=torch.float64)))
    central_first = (at(h) - at(-h)) / (2 * h)
    assert abs(central_first) < h            # -> 0: first derivative continuous
    central_second = (at(h) - 2 * at(0.) + at(-h)) / h ** 2
    # (pen''(0+) + pen''(0-)) / 2 = (2 + 0) / 2 = 1: a real second-derivative
    # jump, deliberately kept (GrandPlan Eq.5's shape).
    assert central_second == pytest.approx(1.0, abs=1e-2)


def test_penalty_gradient_matches_finite_differences():
    for d in (0.3, 1.0, 2.5):
        h = 1e-6
        at = lambda v: float(cap_penalty(torch.tensor(v, dtype=torch.float64)))
        assert cap_penalty_grad(torch.tensor(d)).item() == \
            pytest.approx((at(d + h) - at(d - h)) / (2 * h), rel=1e-6)


def test_declared_curvature_is_pen_second_derivative_at_the_reference():
    assert CAP_CURVATURE_DREF == 1.0
    assert cap_curvature() == pytest.approx(14.0)
    assert cap_curvature(0.0) == pytest.approx(2.0)


# ------------------------------------------------------- zero capacity ----
def test_zero_capacity_segments_keep_a_finite_penalty_with_no_epsilon():
    demand = torch.tensor([0., 2., 3.], dtype=torch.float64)
    capacity = torch.tensor([0., 0., 6.], dtype=torch.float64)
    d, inv = normalised_overflow(demand, capacity)
    np.testing.assert_allclose(d.numpy(), [0., 2., -0.5])
    np.testing.assert_allclose(inv.numpy(), [1., 1., 1. / 6.])
    penalty = cap_penalty(d)
    assert torch.isfinite(penalty).all()
    assert penalty[0].item() == 0.0                     # no demand -> no penalty
    assert penalty[1].item() == pytest.approx(2 * 8 + 4)
    assert penalty[2].item() == 0.0                     # under capacity


def test_zero_capacity_overflow_is_differentiable_and_has_no_nan():
    demand = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
    capacity = torch.tensor([0.], dtype=torch.float64)
    d, _inv = normalised_overflow(demand, capacity)
    cap_penalty(d).sum().backward()
    assert torch.isfinite(demand.grad).all()
    assert demand.grad.item() == pytest.approx(6 * 4 + 2 * 2)


# ------------------------------------------------------------- alpha ------
def test_segment_softmax_normalises_within_each_group():
    u = torch.tensor([0., -1., 5., 5., 5.], dtype=torch.float64)
    group = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    alpha = segment_softmax(u, group, 2)
    assert alpha[:2].sum().item() == pytest.approx(1.0)
    assert alpha[2:].sum().item() == pytest.approx(1.0)
    np.testing.assert_allclose(alpha[2:].numpy(), [1 / 3.] * 3)
    assert alpha[0].item() > alpha[1].item()


def test_segment_softmax_is_shift_invariant_and_overflow_safe():
    u = torch.tensor([1e6, 1e6 - 1.], dtype=torch.float64)
    group = torch.zeros(2, dtype=torch.int64)
    alpha = segment_softmax(u, group, 1)
    assert torch.isfinite(alpha).all()
    small = segment_softmax(u - 1e6, group, 1)
    torch.testing.assert_close(alpha, small)


def test_l1_point_box_distance_and_subgradient():
    box = torch.tensor([[10., 0., 10., 20.]], dtype=torch.float64)  # vertical seg
    for cx, cy, expected in ((10., 5., 0.), (13., 5., 3.), (7., 25., 8.)):
        got = l1_point_box_dist(torch.tensor([cx], dtype=torch.float64),
                                torch.tensor([cy], dtype=torch.float64), box)
        assert got.item() == pytest.approx(expected)
    ddx, ddy = l1_point_box_grad(torch.tensor([13., 7., 10.], dtype=torch.float64),
                                 torch.tensor([5., 25., 5.], dtype=torch.float64),
                                 box.repeat(3, 1))
    np.testing.assert_allclose(ddx.numpy(), [1., -1., 0.])
    np.testing.assert_allclose(ddy.numpy(), [0., 1., 0.])


# ------------------------------------------------- reference term ---------
def test_feed_through_charges_both_the_entry_and_the_exit_segment():
    """Interpretation D-1/D-2: a net with pins in regions 0 and 2 traverses 1,
    so its entry (0|1) and exit (1|2) segments are in different alpha groups
    and each take the full w*q_0*q_2 -- the spec's 'both take demand'."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    assert table.num_segments == 2
    pairs = [(int(table.pair_a[s]), int(table.pair_b[s]))
             for s in range(table.num_segments)]
    assert pairs == [(0, 1), (1, 2)]
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    assert cand.n_groups == 2                       # NOT one softmax over both
    ref = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    demand = ref.demand(_pos(nl), tau=0.5).detach().numpy()
    # tau is small, so q_0 = q_2 = 1 and q_1 = 0; alpha = 1 in each singleton
    # group; w_e = 1.
    np.testing.assert_allclose(demand, [1., 1.], atol=1e-6)


def test_two_alternatives_on_one_boundary_pair_share_one_unit_of_demand():
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]], k=3)
    cand = select_candidates(np.zeros(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64), table)
    ref = CapTermRef(io, table.box, np.full(2, 100.), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    demand = ref.demand(_pos(nl), tau=0.5).detach().numpy()
    assert demand[0] == pytest.approx(1.0, abs=1e-6)
    assert demand[1] == pytest.approx(0.0, abs=1e-12)


def test_reference_penalty_responds_only_above_capacity():
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    slack = CapTermRef(io, table.box, np.full(2, 10.), tau_b=2. * rg.cell_w)
    tight = CapTermRef(io, table.box, np.full(2, 0.5), tau_b=2. * rg.cell_w)
    slack.set_candidates(cand)
    tight.set_candidates(cand)
    p = _pos(nl)
    assert float(slack(p, 0.5, 1.0).detach()) == 0.0
    assert float(tight(p, 0.5, 1.0).detach()) == pytest.approx(6.0)


def test_reference_gradient_is_nonzero_and_finite():
    nl, rg, table, io, _csr = _setup([(15., 15.), (75., 15.)], [[0, 1]])
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.full(2, 2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    ref = CapTermRef(io, table.box, np.full(2, 0.5), tau_b=2. * rg.cell_w)
    ref.set_candidates(cand)
    p = _pos(nl)
    ref(p, 3.0, 1.0).backward()
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.abs().sum()) > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_capacity_term.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.ops.cap_term'`.

- [ ] **Step 3: Write `src/ioplace/ops/cap_term.py` (pure functions + reference term)**

```python
"""Differentiable per-segment boundary capacity term (v2 design sec 5, P-D).

Demand, as this plan reads sec 5 (interpretations D-1/D-2 in
docs/superpowers/plans/2026-09-19-v2-p-d-capacity.md):

    D_s = sum over candidates c on segment s of
              w_{e(c)} * q_{e(c),u(c)} * q_{e(c),v(c)} * alpha_c

  * (u, v) is the candidate's *demand pair* -- the regions of the two MST-edge
    endpoints whose leg produced the crossing, not the segment's own boundary
    pair. For an adjacent-pair leg they coincide and this is sec 5's formula
    verbatim; for a feed-through they do not, and that is exactly what makes
    sec 5's "a traversing net crossed entry and exit segments, so both take
    demand" true (with the segment's own pair, q_{e,middle} == 0 kills both).
  * alpha is a softmax over the candidates of one (net, u, v, boundary pair)
    group, by -L1(net soft pin centroid, segment)/tau_b. Alternatives on one
    boundary share a group and one unit of demand; a feed-through's entry and
    exit lie on different boundaries, so they are different groups and each
    takes the full product.

Penalty, verbatim from sec 5:

    d_s = (D_s - C_s)/C_s,   L_cap = sum_s (2[d_s]_+^3 + [d_s]_+^2)

C1 at the knee, exactly zero below capacity. Phase-1 sec 5.4's
softplus((D-C)/C) is rejected there and must not reappear. For C_s == 0 the
unit rule forbids an epsilon, so d_s := D_s (interpretation D-3): finite,
zero at zero demand, and the penalty expression itself is unchanged.
"""
import numpy as np
import torch

from ioplace.ops.soft_assign import (_chunks, chunk_p_ell, region_sdf_l1,
                                     softmax_stats)

# pen''(d) = 12d + 2 is unbounded, so sec 4's "max_s pen''" has no supremum.
# The declared curvature is pen'' at the overflow level the design tolerates.
CAP_CURVATURE_DREF = 1.0


def cap_curvature(d_ref=CAP_CURVATURE_DREF):
    """pen''(d_ref) = 12*d_ref + 2 -- the value handed to
    `TermNormalizer.register(..., curvature=...)`. Cmax is a lambda-weighted
    *mean* over active terms (`norm.cmax_from_curvatures`), so a representative
    curvature is what the Lipschitz cap wants."""
    return 12.0 * float(d_ref) + 2.0


def cap_penalty(d):
    """2[d]_+^3 + [d]_+^2, elementwise. Exactly 0 for d <= 0."""
    if torch.is_tensor(d):
        r = d.clamp(min=0.0)
    else:
        r = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
    return 2.0 * r ** 3 + r ** 2


def cap_penalty_grad(d):
    """pen'(d) = 6[d]_+^2 + 2[d]_+. pen'(0) == 0 from both sides (C1)."""
    if torch.is_tensor(d):
        r = d.clamp(min=0.0)
    else:
        r = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
    return 6.0 * r ** 2 + 2.0 * r


def normalised_overflow(demand, capacity):
    """(d, inv) with d = (D - C)/C and inv = dd/dD.

    Interpretation D-3 (unit rule: zero-capacity segments stay blocked with a
    finite penalty, no epsilon substitution): where C == 0, d := D and
    inv := 1. `safe` keeps the *unused* branch of torch.where finite, because
    a NaN/inf there still poisons the backward pass."""
    positive = capacity > 0
    safe = torch.where(positive, capacity, torch.ones_like(capacity))
    d = torch.where(positive, (demand - capacity) / safe, demand)
    inv = torch.where(positive, 1.0 / safe, torch.ones_like(safe))
    return d, inv


def segment_softmax(u, group, n_groups):
    """softmax of `u` within each `group`. Shift-invariant (per-group max
    subtracted) so a large |u| cannot overflow."""
    peak = torch.full((n_groups,), -float("inf"), dtype=u.dtype, device=u.device)
    peak.scatter_reduce_(0, group, u, reduce="amax", include_self=True)
    e = torch.exp(u - peak[group])
    total = torch.zeros(n_groups, dtype=u.dtype, device=u.device)
    total.index_add_(0, group, e)
    return e / total[group]


def l1_point_box_dist(cx, cy, box):
    """L1 distance from (cx, cy) to the axis-aligned box [x0,y0,x1,y1]; zero
    inside. A segment's box is degenerate in one axis (a vertical segment has
    x0 == x1), so this is |cx - X| plus the clamped y distance -- one
    expression covering both orientations."""
    dx = (box[:, 0] - cx).clamp(min=0.0) + (cx - box[:, 2]).clamp(min=0.0)
    dy = (box[:, 1] - cy).clamp(min=0.0) + (cy - box[:, 3]).clamp(min=0.0)
    return dx + dy


def l1_point_box_grad(cx, cy, box):
    """(d d1/d cx, d d1/d cy). The subgradient at the kink (on the box, or
    exactly on a degenerate axis) is 0, matching torch.sign's convention and
    `region_sdf_l1`'s own clamped form."""
    ddx = (cx > box[:, 2]).to(cx.dtype) - (cx < box[:, 0]).to(cx.dtype)
    ddy = (cy > box[:, 3]).to(cy.dtype) - (cy < box[:, 1]).to(cy.dtype)
    return ddx, ddy


class _CapBase(torch.nn.Module):
    """Shared candidate/segment bookkeeping for CapTermRef and CapTerm."""

    def __init__(self, io_term, seg_box, seg_capacity, *, tau_b):
        super().__init__()
        self.io_term = io_term
        device = io_term.rects.device
        box = torch.as_tensor(np.asarray(seg_box, dtype=np.float64),
                              dtype=torch.float64, device=device)
        capacity = torch.as_tensor(np.asarray(seg_capacity, dtype=np.float64),
                                   dtype=torch.float64, device=device)
        if box.ndim != 2 or box.shape[1] != 4:
            raise ValueError("seg_box must be (S,4)")
        if capacity.shape != (box.shape[0],):
            raise ValueError("seg_capacity must carry one value per segment")
        if not bool(torch.isfinite(capacity).all()) or bool((capacity < 0).any()):
            raise ValueError("finite nonnegative capacity required")
        if not (tau_b > 0):
            raise ValueError("tau_b must be positive")
        self.tau_b = float(tau_b)
        self.register_buffer("seg_box", box)
        self.register_buffer("C", capacity)
        self.register_buffer("deg", torch.bincount(
            io_term.net_idx, minlength=io_term.n_active).to(torch.float64))
        self.n_groups = 0
        self.last_demand = None
        self.last_d = None
        for name in ("cand_net", "cand_u", "cand_v", "cand_seg", "cand_group"):
            self.register_buffer(name,
                                 torch.zeros(0, dtype=torch.int64, device=device))

    @property
    def num_segments(self):
        return int(self.C.numel())

    def set_candidates(self, cand):
        """Install a `region_segments.Candidates` (already remapped into the IO
        term's active-net index space). Called by the driver on the
        `home_period` cadence, exactly like `FtTerm.set_home`."""
        device = self.C.device
        meta = self.io_term
        arrays = {}
        for name, values in (("cand_net", cand.net), ("cand_u", cand.u),
                             ("cand_v", cand.v), ("cand_seg", cand.seg),
                             ("cand_group", cand.group)):
            arrays[name] = torch.as_tensor(np.asarray(values, dtype=np.int64),
                                           dtype=torch.int64, device=device)
        size = arrays["cand_net"].numel()
        if any(t.numel() != size for t in arrays.values()):
            raise ValueError("candidate arrays must be the same length")
        if size:
            if bool(((arrays["cand_net"] < 0)
                     | (arrays["cand_net"] >= meta.n_active)).any()):
                raise ValueError("candidate net index outside the active nets")
            for key in ("cand_u", "cand_v"):
                if bool(((arrays[key] < 0) | (arrays[key] >= meta.K)).any()):
                    raise ValueError("candidate region id outside [0,K)")
            if bool((arrays["cand_u"] >= arrays["cand_v"]).any()):
                raise ValueError("candidate demand pairs must satisfy u < v")
            if bool(((arrays["cand_seg"] < 0)
                     | (arrays["cand_seg"] >= self.num_segments)).any()):
                raise ValueError("candidate segment id outside the table")
            groups = arrays["cand_group"]
            if int(groups.min()) < 0 or int(groups.max()) >= int(cand.n_groups):
                raise ValueError("candidate group ids are not dense")
        for name, tensor in arrays.items():
            setattr(self, name, tensor)
        self.n_groups = int(cand.n_groups)

    def w_cand(self):
        """w_e per candidate, read live from the IO term so the reweighting
        paths (`--alpha-io`, `--ft-reweight`) are picked up."""
        return self.io_term.w[self.cand_net]

    def _centroids(self, x, y):
        """The net's soft pin centroid c_e: one index_add over the IO CSR's
        deduplicated node list (sec 5), in the same anchored coordinates the
        IO term uses, so P-F's --node-anchor choice carries through."""
        meta = self.io_term
        cx = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
        cy = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
        cx.index_add_(0, meta.net_idx, x[meta.node_idx].double())
        cy.index_add_(0, meta.net_idx, y[meta.node_idx].double())
        return cx / self.deg, cy / self.deg

    def _alpha(self, cx, cy):
        d1 = l1_point_box_dist(cx[self.cand_net], cy[self.cand_net],
                               self.seg_box[self.cand_seg])
        return segment_softmax(-d1 / self.tau_b, self.cand_group, self.n_groups)


class CapTermRef(_CapBase):
    """Dense autograd oracle -- materialises an (E,K) `q`, so it is for tests
    and small-scale probes only, exactly like `IoTermRef`/`FtTermRef`. The
    production term is `CapTerm` in this module and is bound to this class by
    an equivalence test."""

    def _demand(self, pos, tau):
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        x = torch.cat((x[:meta.num_movable], x[meta.num_movable:].detach()))
        y = torch.cat((y[:meta.num_movable], y[meta.num_movable:].detach()))
        rects = meta.rects.to(dtype=x.dtype)
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau)
        sdf = region_sdf_l1(x, y, rects, meta.rect2region, 0, meta.K)
        _p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
        S = torch.zeros((meta.n_active, meta.K), dtype=torch.float64,
                        device=x.device)
        S = S.index_add(0, meta.net_idx, ell[meta.node_idx].double())
        q = -torch.expm1(S)
        cx, cy = self._centroids(x, y)
        alpha = self._alpha(cx, cy)
        product = (self.w_cand() * q[self.cand_net, self.cand_u]
                   * q[self.cand_net, self.cand_v] * alpha)
        demand = torch.zeros(self.num_segments, dtype=torch.float64,
                             device=x.device)
        return demand.index_add(0, self.cand_seg, product)

    def demand(self, pos, tau):
        return self._demand(pos, tau)

    def forward(self, pos, tau, lambda_cap):
        if lambda_cap == 0.0 or self.cand_net.numel() == 0:
            return pos.new_zeros(())
        demand = self._demand(pos, tau)
        d, _inv = normalised_overflow(demand, self.C)
        self.last_demand, self.last_d = demand.detach(), d.detach()
        return lambda_cap * cap_penalty(d).sum()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_capacity_term.py -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/ops/cap_term.py tests/test_capacity_term.py
git commit -m "$(cat <<'MSG'
feat(capacity): penalty, alpha softmax and the dense reference term

L_cap = sum_s (2[d]_+^3 + [d]_+^2) verbatim from v2 design sec 5 -- exactly
zero below capacity, C1 at the knee, C2-discontinuous by design; phase-1's
softplus form stays rejected. Zero-capacity segments take d := D (no epsilon,
unit rule). CapTermRef is the dense autograd oracle the production term is
bound to; a feed-through's entry and exit segments land in different alpha
groups and each take the full w*q_u*q_v.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: The production chunked `CapTerm`

**Files:**
- Modify: `src/ioplace/ops/cap_term.py` (append; `_CapBase`, `CapTermRef` and the pure functions from Task 4 are unchanged)
- Test: `tests/test_capacity_term.py` (append)

**Interfaces:**
- Consumes: Task 4's `_CapBase`, `normalised_overflow`, `cap_penalty`, `cap_penalty_grad`, `l1_point_box_grad`, `cap_curvature`; `ops/soft_assign._chunks/chunk_p_ell/region_sdf_l1/softmax_stats`.
- Produces:
  - `class _CapFn(torch.autograd.Function)`
  - `class CapTerm(_CapBase)` with `forward(pos, tau, lambda_cap)`, `demand(pos, tau)`, `diagnostics(pos, tau) -> dict`, property `curvature`
  - `class CapNormTerm` with `value(pos, ctx)` — the `TermNormalizer` adapter, alongside `ops/norm_terms.IoNormTerm`/`FtNormTerm`

**Memory contract (spec §5, §10 risk 1).** *"`q_aq_b` is differentiated exactly, not linearised — both cofactors are cached per candidate during the forward chunk loop into an `(E_cand,m)` array, so no `(E,K)` tensor appears."* `_CapFn` mirrors `_FtFn`'s four-pass structure exactly (`ops/ft_term.py:20-127`): every per-chunk tensor is `(N,c)`, `(P',c)` or `(E,c)` with `c == meta.k_chunk`, and the only candidate-sized arrays are the two `(M,)` cofactor caches `qa`/`qb`. Nothing of shape `(N,K)`, `(P,K)` or `(E,K)` is ever allocated.

**The exact product gradient.** `∂(q_u·q_v)/∂q_u = q_v` and vice versa, so each candidate contributes two entries into the per-chunk coefficient tensor: `coeff[e, u] += g_s·w_e·q_v·α` and `coeff[e, v] += g_s·w_e·q_u·α`, with `g_s = pen'(d_s)·∂d_s/∂D_s`. From there the chain rule to positions is *character for character* `_IoFn`'s: `∂q_{e,k}/∂p_{i,k} = exp(S_{e,k} − ℓ_{i,k})` (the leave-one-out product), the softmax Jacobian `∂L/∂z_{i,k} = p_{i,k}(c_nodes[i,k] − A_i)` with `A_i = Σ_k c_nodes[i,k]·p_{i,k}` needing a full pass over all K, and then a local `autograd.grad` through `region_sdf_l1` so tie-breaks and boundary subgradients match the reference exactly. No linearisation, no `detach()` on either cofactor.

**The α path.** Closed form and cheap — only `(M,)` and `(E,)` tensors: `∂L/∂α_c = g_s·w_e·q_u·q_v`, the softmax back-prop `∂L/∂u_c = α_c(∂L/∂α_c − Σ_{c'∈group} α_{c'}∂L/∂α_{c'})`, then `∂u/∂c_e = −∇d₁/τ_b` and `∂c_e/∂(x_i,y_i) = 1/deg_e`. **This path is what keeps the term alive after the freeze**, when `q` is saturated at 0/1 and carries no gradient (spec §3) — do not gate it on `q`.

**Recorded limitation D-4: a singleton α-group has no gradient.** A softmax over one element is identically 1, so `∂α/∂c_e = 0`. Measured on a prototype of this exact code: a net whose group holds one candidate segment gives `‖∇L_cap‖₁ = 0` once `q` saturates, while the same net with two alternative segments on the same boundary pair gives `‖∇L_cap‖₁ = 19.5`, entirely in the coordinate along the boundary. So the post-freeze usefulness of this term is **conditional on the evaluator actually supplying ≥2 alternatives per group** (`m_seg = 2` is the cap, not a guarantee): it does so for nets whose MST legs cross one boundary pair in more than one place, and not for a simple two-pin net crossing one short boundary. Before the freeze the `q` path carries signal regardless, so this only bites in phase 3. `CapTerm.diagnostics` therefore reports `frac_singleton_groups`, the driver logs it, and Task 10's result document must state it next to the rank correlation — if it is near 1.0 on `mempool_group`, spec §3's "capacity on after the freeze" is buying nothing and that is a finding to report, not to hide. Do **not** "fix" this by dropping the softmax normalisation: that would break the `Σ_{s∈group} α = 1` property that makes `D_ab = Σ_{s∈(a,b)} D_s` equal phase-1's pair-level `w_e·q_a·q_b`, which is the fallback §10 open question (i) pre-registers.

**Determinism note.** `index_put_(accumulate=True)` on CUDA is atomic, so the coefficient tensor's float64 reduction order is not fixed — the same property `_IoFn`'s existing `index_add_` already has. At most `m_pairs·m_seg = 8` candidates per net collide on one `(net, region)` slot, so the discrepancy is ~1 ulp. The gradient tests therefore use `rel <= 1e-10`, matching `tests/test_ft_term.py`'s existing tolerance; the *bit-exact* contract in this subproject belongs to the evaluator's integer fields (Task 7), not to gradients.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capacity_term.py`:

```python
from ioplace.ops.cap_term import CapNormTerm, CapTerm


def _pair_case(chunk, capacity, k=3):
    """One 3-pin net spanning all three regions of the strip, so the candidate
    list carries two distinct demand pairs and both boundaries."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 20.), (75., 10.)],
                                     [[0, 1, 2]], k=k, chunk=chunk)
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.array([1, 2], dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    caps = np.full(table.num_segments, capacity)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    return nl, production, reference


@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize("capacity", [0.2, 0.6, 5.0, 0.0])
def test_production_matches_the_reference_value_and_gradient(chunk, capacity):
    nl, production, reference = _pair_case(chunk, capacity)
    pa, pb = _pos(nl), _pos(nl)
    va = production(pa, 3.0, 1.7)
    vb = reference(pb, 3.0, 1.7)
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)


def test_production_gradient_matches_autograd_on_a_200_cell_toy():
    """sec 9's 'gradient vs autograd on a 200-cell toy'. The oracle is
    CapTermRef's ordinary autograd; _CapFn's hand-written backward must agree
    on a case big enough to exercise every chunk boundary and several
    candidates per net."""
    rng = np.random.default_rng(7)
    k = 4
    rs = make_grid_regions((0., 0., 120., 40.), k, 1, lattice=12)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 200
    xy = list(zip(rng.uniform(1., 119., n_cells), rng.uniform(1., 39., n_cells)))
    nets = [sorted(rng.choice(n_cells, int(rng.integers(2, 5)), replace=False).tolist())
            for _ in range(60)]
    nl = _nl(xy, nets)
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    common = dict(csr=csr, rects=rects, rect2region=r2k, K=k,
                  num_movable=n_cells, num_physical=nl.num_physical,
                  num_nodes=nl.num_physical, device="cpu")
    io = IoTerm(chunk_budget=2 * len(csr.flat_net2node), **common)
    # every active net gets both boundaries of a random adjacent pair
    n_active = len(csr.net_ids)
    nets_idx, us, vs, segs = [], [], [], []
    for e in range(n_active):
        pair = int(rng.integers(0, k - 1))
        for seg in np.nonzero((table.pair_a == pair) & (table.pair_b == pair + 1))[0]:
            nets_idx.append(e)
            us.append(pair)
            vs.append(pair + 1)
            segs.append(int(seg))
    cand = select_candidates(np.asarray(nets_idx, dtype=np.int64),
                             np.asarray(us, dtype=np.int64),
                             np.asarray(vs, dtype=np.int64),
                             np.asarray(segs, dtype=np.int64),
                             np.ones(len(segs), dtype=np.int64), table)
    caps = np.full(table.num_segments, 3.0)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    pa, pb = _pos(nl), _pos(nl)
    va, vb = production(pa, 4.0, 1.0), reference(pb, 4.0, 1.0)
    assert float(va.detach()) > 0.0
    assert torch.allclose(va, vb, atol=1e-10, rtol=1e-10)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)
    assert float(pa.grad.abs().sum()) > 0.0


def test_the_q_product_gradient_is_exact_not_linearised():
    """Isolate the q path: freeze alpha by giving every candidate its own
    singleton group, then check d L / d q_u against the analytic q_v cofactor
    by perturbing one node along x and comparing with a central difference of
    the *whole* term. A linearised product (q_u*q_v^frozen) would match the
    value but not this derivative."""
    nl, production, reference = _pair_case(chunk=1, capacity=0.2)
    p = _pos(nl)
    value = production(p, 4.0, 1.0)
    value.backward()
    analytic = p.grad.clone()
    base = p.detach().clone()
    h = 1e-6
    for index in (0, 1, 2):
        probe = base.clone()
        probe[index] += h
        up = float(production(probe.requires_grad_(False), 4.0, 1.0).detach())
        probe = base.clone()
        probe[index] -= h
        down = float(production(probe.requires_grad_(False), 4.0, 1.0).detach())
        assert analytic[index].item() == pytest.approx((up - down) / (2 * h),
                                                       rel=1e-4, abs=1e-9)


def _split_boundary():
    """Region 0 touches region 1 along two disjoint vertical runs (region 2
    plugs the middle), so one net can carry two genuine alternative segments
    on the same boundary pair -- the configuration in which alpha has a
    gradient at all."""
    from ioplace.regions import RegionSet, RegionSpec
    rs = RegionSet(die=DIE, lattice=9, regions=[
        RegionSpec("A", np.array([[0., 0., 30., 30.]])),
        RegionSpec("B", np.array([[30., 0., 60., 10.], [30., 20., 60., 30.]])),
        RegionSpec("C", np.array([[30., 10., 60., 20.], [60., 0., 90., 30.]]))])
    rs.validate()
    return rs


def _split_case(capacity=0.1, chunk=4):
    rs = _split_boundary()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    alternatives = [s for s in range(table.num_segments)
                    if (int(table.pair_a[s]), int(table.pair_b[s])) == (0, 1)]
    assert len(alternatives) == 2
    nl = _nl([(15., 5.), (45., 5.)], [[0, 1]])
    csr = build_net_node_csr(nl, 100)
    rects, r2k = rect_table(rs)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3, num_movable=2,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=chunk * len(csr.flat_net2node))
    cand = select_candidates(np.zeros(2, dtype=np.int64),
                             np.zeros(2, dtype=np.int64),
                             np.ones(2, dtype=np.int64),
                             np.asarray(alternatives, dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    assert cand.n_groups == 1                    # one group, two alternatives
    caps = np.full(table.num_segments, capacity)
    production = CapTerm(io, table.box, caps, tau_b=2. * rg.cell_w)
    reference = CapTermRef(io, table.box, caps, tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    reference.set_candidates(cand)
    return nl, production, reference


def test_the_alpha_path_survives_a_saturated_q():
    """After the freeze q is 0/1 and carries no gradient (sec 3); the term must
    still move cells *along* a boundary through alpha. tau is tiny here, so q
    is saturated and every bit of the surviving gradient is the alpha path --
    and it lies purely in y, the coordinate along the boundary."""
    nl, production, reference = _split_case()
    pa, pb = _pos(nl), _pos(nl)
    va, vb = production(pa, 1e-3, 1.0), reference(pb, 1e-3, 1.0)
    va.backward()
    vb.backward()
    assert torch.allclose(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(pa.grad).all()
    assert float(pa.grad.abs().sum()) > 0.0
    n = nl.num_physical
    assert float(pa.grad[:n].abs().sum()) == pytest.approx(0.0, abs=1e-12)  # x
    assert float(pa.grad[n:].abs().sum()) > 0.0                             # y
    # the group's two alternatives share exactly one unit of demand
    assert float(production.last_demand.sum()) == pytest.approx(1.0, abs=1e-6)


def test_a_singleton_alpha_group_has_no_alpha_gradient():
    """Recorded limitation D-4, pinned so nobody 'fixes' it by accident: a
    softmax over one element is identically 1, so a group with a single
    candidate contributes nothing once q saturates. This is why
    frac_singleton_groups is a reported diagnostic."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]], k=3)
    cand = select_candidates(np.zeros(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64), table)
    production = CapTerm(io, table.box, np.full(table.num_segments, 0.1),
                         tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    p = _pos(nl)
    production(p, 1e-3, 1.0).backward()
    assert float(p.grad.abs().sum()) == pytest.approx(0.0, abs=1e-12)
    assert production.diagnostics(_pos(nl), 1e-3)["frac_singleton_groups"] == 1.0


def test_no_candidates_is_an_exact_zero_detached_from_the_graph():
    """The empty term returns pos.new_zeros(()) -- the same shape IoTerm and
    FtTerm return through term_fn. It has no grad_fn, which is correct:
    DREAMPlace adds it to an objective that does."""
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]])
    production = CapTerm(io, table.box, np.ones(table.num_segments),
                         tau_b=2. * rg.cell_w)
    value = production(_pos(nl), 3.0, 1.0)
    assert float(value.detach()) == 0.0
    assert value.requires_grad is False
    assert production.diagnostics(_pos(nl), 3.0)["num_candidates"] == 0


def test_fixed_and_filler_nodes_never_receive_gradient():
    nl, rg, table, io_full, _csr = _setup([(15., 15.), (45., 15.), (75., 15.)],
                                          [[0, 1], [1, 2]])
    rs = _strip(3)
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    io = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=3, num_movable=2,
                num_physical=nl.num_physical, num_nodes=nl.num_physical,
                device="cpu", chunk_budget=4 * len(csr.flat_net2node))
    cand = select_candidates(np.array([0, 1], dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.array([1, 2], dtype=np.int64),
                             np.array([0, 1], dtype=np.int64),
                             np.ones(2, dtype=np.int64), table)
    production = CapTerm(io, table.box, np.full(table.num_segments, 0.1),
                         tau_b=2. * rg.cell_w)
    production.set_candidates(cand)
    p = _pos(nl)
    production(p, 3.0, 1.0).backward()
    n = nl.num_physical
    assert p.grad[2].item() == 0.0            # node 2 is fixed
    assert p.grad[n + 2].item() == 0.0


def test_diagnostics_and_curvature():
    nl, production, _reference = _pair_case(chunk=2, capacity=0.2)
    diag = production.diagnostics(_pos(nl), 3.0)
    assert set(diag) == {"l_cap", "demand_total", "max_d", "num_over_capacity",
                         "num_candidates", "num_groups", "frac_singleton_groups"}
    assert diag["num_candidates"] == 2 and diag["num_groups"] == 2
    assert diag["max_d"] == pytest.approx(3.997371310542147, rel=1e-9)
    assert diag["num_over_capacity"] == 2
    assert diag["frac_singleton_groups"] == 1.0
    assert production.curvature == pytest.approx(14.0)


def test_cap_norm_term_exposes_the_unweighted_value():
    nl, production, _reference = _pair_case(chunk=2, capacity=0.2)
    adapter = CapNormTerm(production)
    p = _pos(nl)
    got = adapter.value(p, {"tau": 3.0, "iteration": 0, "overflow": 0.2,
                            "gamma": 1.0})
    torch.testing.assert_close(got, production(p, 3.0, 1.0))


def test_set_candidates_rejects_malformed_input():
    nl, rg, table, io, _csr = _setup([(15., 15.), (45., 15.)], [[0, 1]])
    production = CapTerm(io, table.box, np.ones(table.num_segments),
                         tau_b=2. * rg.cell_w)
    good = select_candidates(np.zeros(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64),
                             np.ones(1, dtype=np.int64), table)
    import dataclasses
    with pytest.raises(ValueError, match="active nets"):
        production.set_candidates(dataclasses.replace(
            good, net=np.array([99], dtype=np.int64)))
    with pytest.raises(ValueError, match="segment id"):
        production.set_candidates(dataclasses.replace(
            good, seg=np.array([999], dtype=np.int64)))
    with pytest.raises(ValueError, match="u < v"):
        production.set_candidates(dataclasses.replace(
            good, u=np.array([1], dtype=np.int64), v=np.array([0], dtype=np.int64)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_capacity_term.py -v -k "production or alpha_path or q_product or diagnostics or norm_term or malformed or no_candidates or filler"`
Expected: FAIL — `ImportError: cannot import name 'CapTerm' from 'ioplace.ops.cap_term'`.

- [ ] **Step 3: Append the production term to `src/ioplace/ops/cap_term.py`**

```python
class _CapFn(torch.autograd.Function):
    """Chunked-k forward/backward, structurally identical to `_FtFn`
    (ops/ft_term.py:20-127): FWD-1 reduce (m/t/argmax over all K), FWD-2
    accumulate (per chunk q -> the per-candidate cofactor caches), BWD-1
    reduce (A_i over all K), BWD-2 scatter (dL/dz -> positions via a local
    autograd.grad on region_sdf_l1). Nothing of shape (N,K)/(P,K)/(E,K) is
    ever allocated; the only candidate-sized state is qa/qb, two (M,)
    float64 arrays -- sec 5's "both cofactors are cached per candidate ...
    so no (E,K) tensor appears"."""

    @staticmethod
    def forward(ctx, x, y, meta, cap, tau, lambda_cap):
        rects = meta.rects.to(dtype=x.dtype)
        m, t, am = softmax_stats(x, y, rects, meta.rect2region, meta.K, tau,
                                 chunk=meta.k_chunk)
        size = cap.cand_net.numel()
        qa = torch.zeros(size, dtype=torch.float64, device=x.device)
        qb = torch.zeros(size, dtype=torch.float64, device=x.device)
        for lo, hi in _chunks(meta.K, meta.k_chunk):
            sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)
            _p_c, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
            S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                              device=x.device).index_add_(
                                  0, meta.net_idx, ell_c[meta.node_idx].double())
            q_c = -torch.expm1(S_c)
            sel = (cap.cand_u >= lo) & (cap.cand_u < hi)
            if bool(sel.any()):
                qa[sel] = q_c[cap.cand_net[sel], cap.cand_u[sel] - lo]
            sel = (cap.cand_v >= lo) & (cap.cand_v < hi)
            if bool(sel.any()):
                qb[sel] = q_c[cap.cand_net[sel], cap.cand_v[sel] - lo]

        cx, cy = cap._centroids(x, y)
        alpha = cap._alpha(cx, cy)
        demand = torch.zeros(cap.num_segments, dtype=torch.float64,
                             device=x.device).index_add_(
                                 0, cap.cand_seg, cap.w_cand() * qa * qb * alpha)
        d, inv = normalised_overflow(demand, cap.C)
        value = cap_penalty(d).sum()

        cap.last_demand, cap.last_d = demand.detach(), d.detach()
        meta.last_peak_chunk_elems = meta.k_chunk * max(meta.num_physical,
                                                        meta._n_pins_dedup)
        ctx.save_for_backward(x, y, m, t, am, qa, qb, alpha, d, inv, cx, cy,
                              cap.w_cand().detach().clone())
        ctx.meta, ctx.cap, ctx.tau, ctx.lambda_cap = meta, cap, tau, lambda_cap
        return lambda_cap * value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gout):
        x, y, m, t, am, qa, qb, alpha, d, inv, cx, cy, w_c = ctx.saved_tensors
        meta, cap, tau, lambda_cap = ctx.meta, ctx.cap, ctx.tau, ctx.lambda_cap
        rects = meta.rects.to(dtype=x.dtype)
        n = x.shape[0]

        with torch.no_grad():
            # dL/dD_s, held fixed while D_s is differentiated (chain rule).
            g_seg = cap_penalty_grad(d) * inv
            g_cand = g_seg[cap.cand_seg]

            # ---- alpha path: closed form, only (M,)/(E,) tensors ----
            product = w_c * qa * qb
            dl_dalpha = g_cand * product
            pooled = torch.zeros(cap.n_groups, dtype=torch.float64,
                                 device=x.device)
            pooled.index_add_(0, cap.cand_group, alpha * dl_dalpha)
            dl_du = alpha * (dl_dalpha - pooled[cap.cand_group])
            dl_dd1 = -dl_du / cap.tau_b
            ddx, ddy = l1_point_box_grad(cx[cap.cand_net], cy[cap.cand_net],
                                         cap.seg_box[cap.cand_seg])
            gcx = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
            gcy = torch.zeros(meta.n_active, dtype=torch.float64, device=x.device)
            gcx.index_add_(0, cap.cand_net, dl_dd1 * ddx)
            gcy.index_add_(0, cap.cand_net, dl_dd1 * ddy)
            gx = torch.zeros(n, dtype=torch.float64, device=x.device)
            gy = torch.zeros(n, dtype=torch.float64, device=x.device)
            gx.index_add_(0, meta.node_idx, (gcx / cap.deg)[meta.net_idx])
            gy.index_add_(0, meta.node_idx, (gcy / cap.deg)[meta.net_idx])

            # ---- q path: the exact product derivative, both cofactors ----
            coeff_u = g_cand * w_c * qb * alpha        # dL/dq_{e,u}
            coeff_v = g_cand * w_c * qa * alpha        # dL/dq_{e,v}

            def coeff_chunk(lo, hi):
                out = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device)
                sel = (cap.cand_u >= lo) & (cap.cand_u < hi)
                if bool(sel.any()):
                    out.index_put_((cap.cand_net[sel], cap.cand_u[sel] - lo),
                                   coeff_u[sel], accumulate=True)
                sel = (cap.cand_v >= lo) & (cap.cand_v < hi)
                if bool(sel.any()):
                    out.index_put_((cap.cand_net[sel], cap.cand_v[sel] - lo),
                                   coeff_v[sel], accumulate=True)
                return out

            # BWD-1 (reduce): A_i = sum over ALL k of c_{i,k} * p_{i,k}.
            acc = torch.zeros(n, dtype=torch.float64, device=x.device)
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                sdf_c = region_sdf_l1(x, y, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx,
                                                              ell_pins)
                c_pins = (coeff_chunk(lo, hi)[meta.net_idx]
                          * torch.exp(S_c[meta.net_idx] - ell_pins))
                c_nodes = torch.zeros((n, hi - lo), dtype=torch.float64,
                                      device=x.device).index_add_(
                                          0, meta.node_idx, c_pins)
                acc = acc + (c_nodes * p_c.double()).sum(dim=1)

            # BWD-2 (scatter): dL/dz = p*(c - A); dL/dx += dL/dz * (-1/tau) *
            # d(sdf)/dx, through a *local* autograd.grad on the very same
            # region_sdf_l1 the reference differentiates, so ties and boundary
            # subgradients match CapTermRef exactly.
            for lo, hi in _chunks(meta.K, meta.k_chunk):
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    yg = y.detach().requires_grad_(True)
                    sdf_c = region_sdf_l1(xg, yg, rects, meta.rect2region, lo, hi)
                p_c, ell_c = chunk_p_ell(sdf_c.detach(), m, t, am, lo, tau)
                ell_pins = ell_c[meta.node_idx].double()
                S_c = torch.zeros((meta.n_active, hi - lo), dtype=torch.float64,
                                  device=x.device).index_add_(0, meta.net_idx,
                                                              ell_pins)
                c_pins = (coeff_chunk(lo, hi)[meta.net_idx]
                          * torch.exp(S_c[meta.net_idx] - ell_pins))
                c_nodes = torch.zeros((n, hi - lo), dtype=torch.float64,
                                      device=x.device).index_add_(
                                          0, meta.node_idx, c_pins)
                dl_dz = p_c.double() * (c_nodes - acc.unsqueeze(1))
                w_out = dl_dz * (-1.0 / tau)
                gxc, gyc = torch.autograd.grad(sdf_c, [xg, yg],
                                               grad_outputs=w_out.to(sdf_c.dtype))
                gx = gx + gxc.double()
                gy = gy + gyc.double()

            scale = lambda_cap * gout.double()
            gx = gx * scale
            gy = gy * scale
            gx[meta.num_movable:] = 0.0       # fixed + filler never move
            gy[meta.num_movable:] = 0.0
        return gx.to(x.dtype), gy.to(y.dtype), None, None, None, None


class CapTerm(_CapBase):
    """Chunked production capacity term. Same value/gradient contract as
    `CapTermRef`, routed through `_CapFn` so no forward or backward path ever
    allocates an (N,K)/(P,K)/(E,K) tensor. **This is the only class a driver
    may use.**"""

    @property
    def curvature(self):
        """Handed to `TermNormalizer.register(..., curvature=...)` (sec 4)."""
        return cap_curvature(getattr(self, "curvature_dref", CAP_CURVATURE_DREF))

    def forward(self, pos, tau, lambda_cap):
        if lambda_cap == 0.0 or self.cand_net.numel() == 0:
            return pos.new_zeros(())
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        return _CapFn.apply(x, y, meta, self, tau, lambda_cap)

    def demand(self, pos, tau):
        """Per-segment soft demand D_s, no grad -- the GP surrogate side of the
        sec 9 rank-correlation experiment."""
        with torch.no_grad():
            if self.cand_net.numel() == 0:
                return torch.zeros(self.num_segments, dtype=torch.float64,
                                   device=self.C.device)
            self.forward(pos.detach(), tau, 1.0)
            return self.last_demand.clone()

    def diagnostics(self, pos, tau):
        """Reporting only. The driver logs `max_d` every probe so the declared
        curvature (pen'' at d_ref = 1.0) stays auditable against real data."""
        with torch.no_grad():
            if self.cand_net.numel() == 0:
                return {"l_cap": 0.0, "demand_total": 0.0, "max_d": 0.0,
                        "num_over_capacity": 0, "num_candidates": 0,
                        "num_groups": 0, "frac_singleton_groups": 0.0}
            value = float(self.forward(pos.detach(), tau, 1.0))
            sizes = torch.bincount(self.cand_group, minlength=self.n_groups)
            return {"l_cap": value,
                    "demand_total": float(self.last_demand.sum()),
                    "max_d": float(self.last_d.max()),
                    "num_over_capacity": int((self.last_d > 0).sum()),
                    "num_candidates": int(self.cand_net.numel()),
                    "num_groups": int(self.n_groups),
                    # limitation D-4: a group of size 1 has no alpha gradient,
                    # so this is the fraction of the term that goes silent once
                    # q saturates after the freeze.
                    "frac_singleton_groups": float((sizes == 1).double().mean())}


class CapNormTerm(object):
    """`TermNormalizer` adapter, alongside `ops/norm_terms.IoNormTerm` and
    `FtNormTerm`: `value(pos, ctx)` returns the *unweighted* L_cap at the
    schedule's live tau, and the normalizer owns the backward, the
    fixed/filler masking and the norm order (design sec 4)."""

    def __init__(self, cap_term):
        self.cap_term = cap_term

    def value(self, pos, ctx):
        return self.cap_term(pos, ctx["tau"], 1.0)
```

Also add `curvature_dref` to `_CapBase.__init__`'s signature as a keyword with default `CAP_CURVATURE_DREF`, stored as `self.curvature_dref = float(curvature_dref)`, so `--cap-curvature-dref` (Task 8) can move it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_capacity_term.py -v`
Expected: PASS (all of Task 4's, plus 12 parametrised production cases and 8 more).

- [ ] **Step 5: Check the memory contract by hand**

Run:

```bash
"$IOPLACE_PYTHON" - <<'PY'
import torch
from ioplace.ops.cap_term import CapTerm
import inspect, ioplace.ops.cap_term as m
src = inspect.getsource(m._CapFn)
assert "meta.K)" not in src.replace("meta.K, tau", ""), "a full-K tensor crept in"
print("chunk shapes:", [l.strip() for l in src.splitlines()
                        if "hi - lo" in l][:4])
PY
```

Expected: every allocation in `_CapFn` is sized `hi - lo`, never `meta.K`. This is a reading aid, not a test; the enforceable version is the `k_chunk`-parametrised equivalence test in Step 1.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/ops/cap_term.py tests/test_capacity_term.py
git commit -m "$(cat <<'MSG'
feat(capacity): chunked production CapTerm with the exact product gradient

_CapFn mirrors _FtFn's four-pass structure; q_u*q_v is differentiated
exactly, both cofactors cached per candidate in two (M,) arrays, so no
(E,K) tensor appears (v2 design sec 5). The alpha path is closed-form and
survives a saturated q, which is what keeps the term useful after the
freeze (sec 3). CapNormTerm exposes value(pos, ctx) for TermNormalizer.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: Evaluator hard check — reference side

**Files:**
- Modify: `src/ioplace/evaluator_ref.py:6-25` (`EvalResult`), `:83-96` (signature/docstring), `:109-113` (accumulators), `:134-159` (the per-edge loop), `:163-172` (the return)
- Test: `tests/test_evaluator_capacity.py`

**Interfaces:**
- Consumes: Task 1/2's `enumerate_segments`, `edge_segment_ids`, `segment_utilisation`, `select_candidates`.
- Produces (all default `None`, so every existing caller is unaffected):
  - `evaluate(nl, node_x, node_y, rg, max_degree=256, *, route_wirelength_budget=None, segments=None, segment_capacity=None, capacity_candidates=False)`
  - `EvalResult.segment_demand` (S,) int64, `.segment_capacity` (S,) float64, `.segment_util` (S,) float64, `.num_over_capacity` int, `.num_zero_capacity_segments` int, `.zero_capacity_demand` int, `.max_util` float, `.p99_util` float
  - `EvalResult.cand_net` / `.cand_u` / `.cand_v` / `.cand_seg` / `.cand_count`, each (U,) int64, **ascending by the composite key `((net·K + u)·K + v)·S + seg`**
  - `EvalResult.cand_dropped` int — crossings whose leg had `u == v` (interpretation D-1)

**The hard check (spec §5).** *"Each unit crossing maps to a segment id through the raster; the per-row CSR turns an L-shape leg into a contiguous slice, so work equals total crossings, and the existing `Ph`/`Pv` prefix-sum counts become a free assertion on slice lengths."* On the reference side the assertion is per leg — `len(edge_segment_ids(...)) == nc`, where `nc` is the count `edge_regions_and_crossings` independently produced by walking the lattice — which is strictly stronger than the global identity the GPU can afford. A global `segment_demand.sum() == io_count − large_net_lb` check closes it (nets above `max_degree` take the presence lower bound and never enter the edge loop, `evaluator_ref.py:129-133`).

**Why the candidate key is built here and not in the caller.** The composite key `((net·K + u)·K + v)·S + seg` and `np.unique`'s ascending order are the *entire* mechanism by which the reference and GPU candidate arrays are bit-exact (`torch.unique` sorts the same way). Both evaluators must build the same key with the same `K` and `S`; the capping is then the shared numpy `select_candidates`, called by the driver.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluator_capacity.py`:

```python
import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate
from ioplace.region_grid import RegionGrid
from ioplace.region_segments import (edge_segment_ids, enumerate_segments,
                                     segment_utilisation)
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions
from tests.test_io_term import _nl

DIE = (0., 0., 90., 30.)


def _strip_case():
    """Regions 0|1|2 left to right; one 2-pin net from region 0 to region 2,
    which feeds through region 1 and therefore crosses both boundaries."""
    rs = make_grid_regions(DIE, 3, 1, lattice=9)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    nl = _nl([(15., 15.), (75., 15.)], [[0, 1]])
    return nl, rg, table


def test_segment_demand_counts_every_crossing_once():
    nl, rg, table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table)
    np.testing.assert_array_equal(res.segment_demand, [1, 1])
    assert int(res.segment_demand.sum()) == res.io_count
    assert res.io_count == 2


def test_segment_demand_matches_a_brute_force_walk():
    rng = np.random.default_rng(3)
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 40
    xy = list(zip(rng.uniform(1., 99., n_cells), rng.uniform(1., 99., n_cells)))
    nets = [sorted(rng.choice(n_cells, int(rng.integers(2, 6)),
                              replace=False).tolist()) for _ in range(30)]
    nl = _nl(xy, nets)
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table)
    # independent recount straight off the raster, via the reference MST
    from ioplace.evaluator_ref import net_mst_edges
    from ioplace.netlist import pin_positions
    px, py = pin_positions(nl, nl.node_x, nl.node_y)
    expected = np.zeros(table.num_segments, dtype=np.int64)
    start = nl.flat_net2pin_start
    for net in range(nl.num_nets):
        s, e = start[net], start[net + 1]
        if e - s <= 1:
            continue
        idx = nl.flat_net2pin[s:e]
        nx_, ny_ = px[idx], py[idx]
        for a, b in net_mst_edges(nx_, ny_):
            for sid in edge_segment_ids(rg, table, nx_[a], ny_[a],
                                        nx_[b], ny_[b]):
                expected[sid] += 1
    np.testing.assert_array_equal(res.segment_demand, expected)
    assert int(res.segment_demand.sum()) == res.io_count - res.large_net_lb


def test_segment_demand_sums_to_the_pair_demand_per_pair():
    """The two accountings must agree: sec 5 keeps ell[a,b] as the pair-level
    aggregate and segment demand as its refinement."""
    rng = np.random.default_rng(11)
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 30
    xy = list(zip(rng.uniform(1., 99., n_cells), rng.uniform(1., 99., n_cells)))
    nets = [sorted(rng.choice(n_cells, 3, replace=False).tolist())
            for _ in range(25)]
    nl = _nl(xy, nets)
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table)
    rolled = {}
    for s in range(table.num_segments):
        key = (int(table.pair_a[s]), int(table.pair_b[s]))
        rolled[key] = rolled.get(key, 0) + int(res.segment_demand[s])
    assert {k: v for k, v in rolled.items() if v} == \
        {k: v for k, v in res.boundary_pair_demand.items() if v}


def test_capacity_fields_come_from_the_shared_helper():
    nl, rg, table = _strip_case()
    capacity = np.array([0.5, 4.0])
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   segment_capacity=capacity)
    util, scalars = segment_utilisation(res.segment_demand, capacity)
    np.testing.assert_array_equal(res.segment_util, util)
    np.testing.assert_array_equal(res.segment_capacity, capacity)
    assert res.num_over_capacity == scalars["num_over_capacity"] == 1
    assert res.max_util == scalars["max_util"] == pytest.approx(2.0)
    assert res.p99_util == scalars["p99_util"]
    assert res.num_zero_capacity_segments == 0
    assert res.zero_capacity_demand == 0


def test_a_zero_capacity_segment_counts_as_over_capacity():
    nl, rg, table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   segment_capacity=np.array([0.0, 4.0]))
    assert res.num_over_capacity == 1
    assert res.num_zero_capacity_segments == 1
    assert res.zero_capacity_demand == 1
    assert res.segment_util[0] == np.inf
    assert res.max_util == pytest.approx(0.25)     # zero-capacity excluded


def test_candidates_carry_the_terminal_pair_for_a_feed_through():
    """Interpretation D-1: the demand pair is the MST-edge endpoints' regions
    (0 and 2), not the segments' own boundary pairs."""
    nl, rg, table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   capacity_candidates=True)
    np.testing.assert_array_equal(res.cand_net, [0, 0])
    np.testing.assert_array_equal(res.cand_u, [0, 0])
    np.testing.assert_array_equal(res.cand_v, [2, 2])
    np.testing.assert_array_equal(res.cand_seg, [0, 1])
    np.testing.assert_array_equal(res.cand_count, [1, 1])
    assert res.cand_dropped == 0


def test_candidates_are_sorted_by_the_composite_key():
    rng = np.random.default_rng(5)
    rs = make_grid_regions((0., 0., 100., 100.), 4, 4, lattice=20)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    n_cells = 30
    xy = list(zip(rng.uniform(1., 99., n_cells), rng.uniform(1., 99., n_cells)))
    nets = [sorted(rng.choice(n_cells, 3, replace=False).tolist())
            for _ in range(20)]
    nl = _nl(xy, nets)
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   capacity_candidates=True)
    k, size = rg.k, table.num_segments
    key = ((res.cand_net * k + res.cand_u) * k + res.cand_v) * size + res.cand_seg
    assert (np.diff(key) > 0).all()
    assert (res.cand_u < res.cand_v).all()
    assert int(res.cand_count.sum()) + res.cand_dropped == \
        int(res.segment_demand.sum())


def test_same_region_legs_are_dropped_and_counted():
    """An L-route between two pins of the same region can still detour through
    a neighbour. Such crossings carry no q-expressible signal (u == v), so they
    are dropped from the candidate list -- and the drop is reported."""
    rs = RegionSet(die=DIE, lattice=9, regions=[
        RegionSpec("A", np.array([[0., 0., 30., 30.], [60., 0., 90., 30.]])),
        RegionSpec("B", np.array([[30., 0., 60., 30.]]))])
    rs.validate()
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    nl = _nl([(15., 15.), (75., 15.)], [[0, 1]])     # both pins in region 0
    res = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   capacity_candidates=True)
    assert int(res.segment_demand.sum()) == 2
    assert res.cand_net.size == 0
    assert res.cand_dropped == 2


def test_segments_from_a_different_grid_are_rejected():
    nl, rg, _table = _strip_case()
    other = enumerate_segments(RegionGrid(make_grid_regions(DIE, 2, 1, lattice=9)))
    with pytest.raises(ValueError, match="segment table"):
        evaluate(nl, nl.node_x, nl.node_y, rg, segments=other)


def test_capacity_fields_stay_none_when_no_segments_are_supplied():
    nl, rg, _table = _strip_case()
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    for name in ("segment_demand", "segment_capacity", "segment_util",
                 "num_over_capacity", "max_util", "p99_util", "cand_net"):
        assert getattr(res, name) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_capacity.py -v`
Expected: FAIL — `TypeError: evaluate() got an unexpected keyword argument 'segments'`.

- [ ] **Step 3: Extend `EvalResult`**

In `src/ioplace/evaluator_ref.py`, append to the `EvalResult` dataclass (after `per_net_home`, line 25):

```python
    # v2 P-D (design sec 5): the per-segment hard check. All None unless
    # `evaluate(..., segments=...)` was asked for, so "not measured" and
    # "measured as zero" stay distinguishable.
    segment_demand: np.ndarray = None      # (S,) int64
    segment_capacity: np.ndarray = None    # (S,) float64
    segment_util: np.ndarray = None        # (S,) float64, inf where C == 0 < D
    num_over_capacity: int = None
    num_zero_capacity_segments: int = None
    zero_capacity_demand: int = None
    max_util: float = None
    p99_util: float = None
    # capped-candidate source arrays, ascending by ((net*K+u)*K+v)*S+seg --
    # the ordering that makes ref/GPU parity bit-exact.
    cand_net: np.ndarray = None
    cand_u: np.ndarray = None
    cand_v: np.ndarray = None
    cand_seg: np.ndarray = None
    cand_count: np.ndarray = None
    cand_dropped: int = None               # crossings on legs with u == v
```

- [ ] **Step 4: Thread the segments through `evaluate`**

Replace the signature and docstring tail (`evaluator_ref.py:83-96`):

```python
def evaluate(nl, node_x, node_y, rg, max_degree=256, *,
             route_wirelength_budget=None, segments=None,
             segment_capacity=None, capacity_candidates=False):
    """Evaluate legacy MST L-routes or opt into budgeted detour geometry.

    A budget of .05 allows +5% per MST branch; pin HPWL and connectivity stay
    fixed. The opt-in mode recomputes crossings, FT and pair demand from the
    selected segments. It requires in-die pins and models no routing obstacles.
    Nets above max_degree retain the legacy lower-bound treatment.

    v2 P-D (design sec 5): with `segments` (a region_segments.SegmentTable
    built from this same `rg`) every unit crossing is additionally mapped to
    its segment id and accumulated into `segment_demand`; with
    `segment_capacity` the utilisation fields are filled from the shared
    `region_segments.segment_utilisation` helper; with `capacity_candidates`
    the per-(net, demand pair, segment) crossing counts the capacity term's
    candidate refresh consumes are returned too.
    """
```

Immediately after the docstring, before `router = None`:

```python
    if segments is not None:
        if (segments.k != rg.k or segments.grid_shape() != rg.grid.shape
                or segments.die != tuple(float(v) for v in rg.die)):
            raise ValueError("segment table was built for a different region grid")
    elif segment_capacity is not None or capacity_candidates:
        raise ValueError("segment_capacity/capacity_candidates need segments=")
```

After the `pair_demand = {}` line (`:109`):

```python
    segment_demand = (np.zeros(segments.num_segments, dtype=np.int64)
                      if segments is not None else None)
    candidate_counts = {} if capacity_candidates else None
    candidates_dropped = 0
```

Inside the per-edge loop, right after `tree_wl += length` (`:157`), add:

```python
            if segments is not None:
                if routes is None:
                    seg_ids = edge_segment_ids(rg, segments, nx_[a], ny_[a],
                                               nx_[b], ny_[b])
                else:
                    seg_ids = np.concatenate(
                        [edge_segment_ids(rg, segments, p[0], p[1], q[0], q[1])
                         for p, q in zip(points[:-1], points[1:])]
                        or [np.zeros(0, dtype=np.int32)])
                # the sec 5 "free assertion on slice lengths": the CSR slice
                # and the independent lattice walk must agree exactly.
                assert len(seg_ids) == nc, (
                    "segment lookup disagrees with the lattice walk: "
                    "%d vs %d" % (len(seg_ids), nc))
                np.add.at(segment_demand, seg_ids, 1)
                if candidate_counts is not None:
                    ua, ub = int(net_pin_rids[a]), int(net_pin_rids[b])
                    if ua == ub:
                        candidates_dropped += len(seg_ids)
                    else:
                        lo_r, hi_r = min(ua, ub), max(ua, ub)
                        for sid in seg_ids:
                            key = (net, lo_r, hi_r, int(sid))
                            candidate_counts[key] = candidate_counts.get(key, 0) + 1
```

`nc` is already the local crossing count in both branches (`:144-152`), and `net_pin_rids` is already computed at `:122`.

Before the `return EvalResult(...)`, add:

```python
    capacity_fields = {}
    if segments is not None:
        capacity_fields["segment_demand"] = segment_demand
        assert int(segment_demand.sum()) == int(per_net_crossings.sum()) - large_lb, \
            "per-segment demand does not reconcile with the Ph/Pv crossing count"
        if segment_capacity is not None:
            capacity = np.asarray(segment_capacity, dtype=np.float64)
            util, scalars = segment_utilisation(segment_demand, capacity)
            capacity_fields["segment_capacity"] = capacity
            capacity_fields["segment_util"] = util
            for name in ("num_over_capacity", "num_zero_capacity_segments",
                         "zero_capacity_demand", "max_util", "p99_util"):
                capacity_fields[name] = scalars[name]
    if candidate_counts is not None:
        size = segments.num_segments
        keys = np.array(sorted(candidate_counts), dtype=np.int64).reshape(-1, 4)
        counts = np.array([candidate_counts[tuple(int(v) for v in row)]
                           for row in keys], dtype=np.int64)
        # sort by the composite key torch.unique will produce on the GPU side
        composite = (((keys[:, 0] * rg.k + keys[:, 1]) * rg.k + keys[:, 2]) * size
                     + keys[:, 3]) if keys.size else np.zeros(0, dtype=np.int64)
        order = np.argsort(composite, kind="stable")
        capacity_fields.update(
            cand_net=keys[order, 0] if keys.size else np.zeros(0, dtype=np.int64),
            cand_u=keys[order, 1] if keys.size else np.zeros(0, dtype=np.int64),
            cand_v=keys[order, 2] if keys.size else np.zeros(0, dtype=np.int64),
            cand_seg=keys[order, 3] if keys.size else np.zeros(0, dtype=np.int64),
            cand_count=counts[order] if keys.size else np.zeros(0, dtype=np.int64),
            cand_dropped=int(candidates_dropped))
```

and spread `**capacity_fields` into the `EvalResult(...)` call. Add to the imports at the top of the file:

```python
from ioplace.region_segments import edge_segment_ids, segment_utilisation
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_capacity.py tests/test_evaluator_ref.py -v`
Expected: PASS. `test_evaluator_ref.py` must be untouched — every new argument is keyword-only with a `None` default.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/evaluator_ref.py tests/test_evaluator_capacity.py
git commit -m "$(cat <<'MSG'
feat(evaluator): per-segment demand and utilisation in the reference

Map every unit crossing through the segment raster, assert the CSR slice
length against the independent lattice walk per leg, and reconcile the total
against the crossing count (v2 design sec 5). Utilisation comes from the one
shared region_segments.segment_utilisation helper. Candidate source arrays
carry the *terminal* region pair, sorted by the composite key the GPU side
will reproduce.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 7: Evaluator hard check — GPU side and bit-exact parity

**Files:**
- Modify: `src/ioplace/evaluator_gpu.py:~100-118` (`GpuEvalContext.__init__` signature and the new device tensors), `:~490-520` (`_process_segments`), `:~560-585` (`evaluate`'s `pin_rid` lifetime), `:~690-760` (the edge-batch loop and the return)
- Test: `tests/test_evaluator_gpu.py` (append)

**Interfaces:**
- Consumes: Task 6's `EvalResult` capacity fields and their exact semantics; Task 1/2's `SegmentTable`, `segment_utilisation`.
- Produces: `GpuEvalContext(nl, rg, device="cuda", max_degree=256, mst_chunk_budget=..., seg_chunk_budget=..., edge_batch_size=..., segments=None, segment_capacity=None)` and `ctx.evaluate(node_x, node_y, capacity_candidates=False)`; `evaluate_gpu(..., segments=None, segment_capacity=None, capacity_candidates=False)`.

**Why the raster and not the CSR on the GPU.** `_process_segments` already performs a padded gather of every lattice cell along each leg in order to compute the visited-region bitmask and the pair-demand transitions (`evaluator_gpu.py:~495-520`). The segment id of a transition at position `t` of that gather is one more indexed read into the raster at the coordinates it already holds — free. A CSR `searchsorted` would add work the GPU path does not otherwise do. The reference keeps the CSR because its cost model is the opposite (Task 2). Both produce identical numbers; that is the parity test's job to prove.

**Where the transition sits.** Within the *unpadded* span, `cand[:, t+1] == cand[:, t] + 1`, so a transition between gather positions `t` and `t+1` is the unit edge whose boundary index is `cand[:, t]`. For the horizontal pass (`is_vert=False`, a leg along a grid row `f`) that edge is `edge_seg_v[f, cand[:, t]]`; for the vertical pass it is `edge_seg_h[cand[:, t], f]`. The padding clamps `cand` to `h0`, so padded positions repeat the last cell, `diff` is `False` there, and no spurious transition is produced — the same property the existing pair-demand code relies on.

**Candidate keys across batches.** The pair-demand accumulator can be a fixed `64*64` bincount; the candidate key space `n_nets·K·K·S` cannot. Per edge batch, `torch.unique(key, return_counts=True)` collapses the batch to its *distinct* `(net, u, v, seg)` tuples — which is what the result returns anyway — and the small per-batch arrays are concatenated and uniqued once at the end. `torch.unique` returns ascending order, matching `np.unique` on the reference side, so the two candidate lists are bit-identical by construction rather than by coincidence.

**Parity contract addition (spec §9).** Extend `_assert_batch_invariant_fields` with `segment_demand`, `cand_*`, `cand_dropped`, the four capacity scalars, and — because they are computed by the same numpy helper on the same integers — `segment_util`, `max_util`, `p99_util` as **bit-exact**, not `approx`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluator_gpu.py`:

```python
# ---------------------------------------------------------------------------
# v2 P-D (design sec 5 / sec 9): per-segment demand and utilisation, bit-exact
# ref vs GPU across batch sizes and chunk budgets.
# ---------------------------------------------------------------------------
from ioplace.region_segments import enumerate_segments, segment_utilisation


def _multi_rect_regions(die):
    """One non-rectangular ("producer-like") geometry, so parity is not only
    tested on axis-aligned grid partitions: region 0 is an L, region 1 the
    block it wraps, region 2 and 3 split the right half."""
    from ioplace.regions import RegionSet, RegionSpec
    xl, yl, xh, yh = die
    mx, my = (xl + xh) / 2., (yl + yh) / 2.
    qx = xl + (xh - xl) / 4.
    rs = RegionSet(die=die, lattice=20, regions=[
        RegionSpec("L", np.array([[xl, yl, qx, yh], [qx, yl, mx, my]])),
        RegionSpec("B", np.array([[qx, my, mx, yh]])),
        RegionSpec("R0", np.array([[mx, yl, xh, my]])),
        RegionSpec("R1", np.array([[mx, my, xh, yh]])),
    ])
    rs.validate()
    return rs


def _capacity_for(table, rng):
    """A capacity vector with slack, tight and blocked segments, so all three
    branches of segment_utilisation are exercised."""
    capacity = rng.uniform(0.5, 6.0, table.num_segments)
    capacity[::7] = 0.0
    return capacity


def _assert_capacity_fields_bit_exact(a, b):
    np.testing.assert_array_equal(a.segment_demand, b.segment_demand)
    np.testing.assert_array_equal(a.segment_util, b.segment_util)
    np.testing.assert_array_equal(a.segment_capacity, b.segment_capacity)
    assert a.num_over_capacity == b.num_over_capacity
    assert a.num_zero_capacity_segments == b.num_zero_capacity_segments
    assert a.zero_capacity_demand == b.zero_capacity_demand
    # max_util / p99_util come from the same numpy helper on the same integers,
    # so they are bit-exact, not merely close (parity contract).
    assert a.max_util == b.max_util
    assert a.p99_util == b.p99_util


def _assert_candidates_bit_exact(a, b):
    for name in ("cand_net", "cand_u", "cand_v", "cand_seg", "cand_count"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.cand_dropped == b.cand_dropped


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("multi_rect", [False, True])
def test_gpu_segment_demand_matches_the_reference(seed, multi_rect):
    rng = np.random.default_rng(seed)
    rs = (_multi_rect_regions(DIE) if multi_rect
          else make_grid_regions(DIE, 4, 4, lattice=20))
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    capacity = _capacity_for(table, rng)
    nl = _random_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg, segments=table,
                   segment_capacity=capacity, capacity_candidates=True)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg, segments=table,
                       segment_capacity=capacity, capacity_candidates=True)
    _assert_capacity_fields_bit_exact(gpu, ref)
    _assert_candidates_bit_exact(gpu, ref)
    assert int(ref.segment_demand.sum()) == ref.io_count - ref.large_net_lb


def test_gpu_segment_demand_is_batch_invariant():
    """T2's field-specific acceptance, extended: construction parameters that
    force many tiny chunks must give bit-exact capacity fields against a
    single-huge-batch reference run."""
    rs = _multi_rect_regions(DIE)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    rng = np.random.default_rng(123)
    capacity = _capacity_for(table, rng)
    nl = _lambda_ge4_netlist(rng, rg.k)

    from ioplace.evaluator_gpu import GpuEvalContext
    baseline = GpuEvalContext(nl, rg, device="cuda", mst_chunk_budget=10 ** 9,
                              seg_chunk_budget=10 ** 9, edge_batch_size=10 ** 9,
                              segments=table, segment_capacity=capacity)
    reference = baseline.evaluate(nl.node_x, nl.node_y, capacity_candidates=True)
    for mst_b, seg_b, edge_b in [(3, 3, 3), (7, 11, 5), (1, 4, 2)]:
        ctx = GpuEvalContext(nl, rg, device="cuda", mst_chunk_budget=mst_b,
                             seg_chunk_budget=seg_b, edge_batch_size=edge_b,
                             segments=table, segment_capacity=capacity)
        got = ctx.evaluate(nl.node_x, nl.node_y, capacity_candidates=True)
        _assert_batch_invariant_fields(reference, got)
        _assert_capacity_fields_bit_exact(reference, got)
        _assert_candidates_bit_exact(reference, got)


def test_gpu_capacity_fields_are_absent_without_segments():
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(np.random.default_rng(0))
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    for name in ("segment_demand", "segment_util", "num_over_capacity",
                 "cand_net"):
        assert getattr(gpu, name) is None


def test_gpu_large_nets_never_contribute_segment_demand():
    """Nets above max_degree take the presence lower bound and skip the MST /
    segment pipeline entirely, on both sides."""
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    table = enumerate_segments(rg)
    rng = np.random.default_rng(11)
    nl = _random_case(rng, n_cells=60, n_nets=8, max_d=30)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg, max_degree=8, segments=table)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg, max_degree=8,
                       segments=table)
    assert ref.large_net_lb > 0
    np.testing.assert_array_equal(gpu.segment_demand, ref.segment_demand)
    assert int(ref.segment_demand.sum()) == ref.io_count - ref.large_net_lb


@pytest.mark.slow
def test_segment_demand_regression_on_a_frozen_placement(tmp_path):
    """sec 9: 'Add a slow regression pinning per-segment demand for one frozen
    placement.' The first run writes the baseline next to the existing
    evaluator regression fixtures; later runs compare against it."""
    import json
    import os
    rs = _multi_rect_regions(DIE)
    rg = RegionGrid(rs)
    table = enumerate_segments(rg)
    rng = np.random.default_rng(2026)
    nl = _random_case(rng, n_cells=400, n_nets=300, max_d=10)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg, segments=table)
    baseline = os.path.join(os.path.dirname(__file__), "data",
                            "segment_demand_multirect.json")
    payload = {"segments_sha256": __import__(
        "ioplace.region_segments", fromlist=["segments_digest"]
    ).segments_digest(table),
        "segment_demand": gpu.segment_demand.tolist(),
        "io_count": int(gpu.io_count), "large_net_lb": int(gpu.large_net_lb)}
    if not os.path.exists(baseline):
        os.makedirs(os.path.dirname(baseline), exist_ok=True)
        with open(baseline, "w") as stream:
            json.dump(payload, stream, indent=1)
        pytest.skip("wrote the segment-demand baseline; rerun to compare")
    with open(baseline) as stream:
        saved = json.load(stream)
    assert saved["segments_sha256"] == payload["segments_sha256"]
    assert saved["segment_demand"] == payload["segment_demand"]
    assert saved["io_count"] == payload["io_count"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py -v -k "segment or capacity" -m "not slow"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'segments'`.

- [ ] **Step 3: Accept and validate the segment table in `GpuEvalContext.__init__`**

Add `segments=None, segment_capacity=None` to the keyword-only tail of `__init__`, and after the `self._seg_bounds_t = ...` line:

```python
        # ---- v2 P-D (design sec 5): the unit-edge -> segment-id raster ----
        # The GPU path reads the raster directly rather than the per-row CSR:
        # _process_segments already gathers every lattice cell along a leg for
        # the bitmask and pair-demand passes, so the segment id of a transition
        # is one more indexed read at coordinates it already holds. The CSR is
        # the reference's structure, where the cost model is the opposite.
        self.segments = segments
        self.segment_capacity = None
        self.edge_seg_v_t = None
        self.edge_seg_h_t = None
        if segments is not None:
            if (segments.k != rg.k or segments.grid_shape() != rg.grid.shape
                    or segments.die != tuple(float(v) for v in rg.die)):
                raise ValueError("segment table was built for a different region grid")
            self.num_segments = int(segments.num_segments)
            assert self.n_nets * self.k * self.k * max(self.num_segments, 1) < 2 ** 63, (
                "no int64 headroom for the (net,u,v,seg) composite candidate key")
            self.edge_seg_v_t = torch.from_numpy(
                np.ascontiguousarray(segments.edge_seg_v)).to(self.device)
            self.edge_seg_h_t = torch.from_numpy(
                np.ascontiguousarray(segments.edge_seg_h)).to(self.device)
            if segment_capacity is not None:
                self.segment_capacity = np.asarray(segment_capacity,
                                                   dtype=np.float64)
                if self.segment_capacity.shape != (self.num_segments,):
                    raise ValueError("segment_capacity must carry one value "
                                     "per segment")
        elif segment_capacity is not None:
            raise ValueError("segment_capacity needs segments=")
```

- [ ] **Step 4: Collect segment ids in `_process_segments`**

Change its signature to

```python
    def _process_segments(self, fixed_idx, lo, hi, net_id, is_vert,
                          passed_bit_acc, pair_count_acc,
                          segment_demand_acc=None, edge_u=None, edge_v=None,
                          candidate_keys=None):
```

and, inside the innermost chunk loop, replace the pair-demand block

```python
                if L > 1:
                    diff = ids[:, 1:] != ids[:, :-1]
                    if diff.any():
                        a = ids[:, :-1][diff]
                        b = ids[:, 1:][diff]
                        lo_ab = torch.minimum(a, b)
                        hi_ab = torch.maximum(a, b)
                        keys = lo_ab * 64 + hi_ab
                        pair_count_acc += torch.bincount(keys, minlength=pair_count_acc.numel())
```

with

```python
                if L > 1:
                    diff = ids[:, 1:] != ids[:, :-1]
                    if diff.any():
                        a = ids[:, :-1][diff]
                        b = ids[:, 1:][diff]
                        lo_ab = torch.minimum(a, b)
                        hi_ab = torch.maximum(a, b)
                        keys = lo_ab * 64 + hi_ab
                        pair_count_acc += torch.bincount(keys, minlength=pair_count_acc.numel())

                        # v2 P-D: the same transitions, keyed by segment id.
                        # Within the unpadded span cand[:,t+1] == cand[:,t]+1,
                        # so the transition at position t is the unit edge whose
                        # boundary index is cand[:,t]. Padded positions repeat
                        # the last cell, so `diff` is False there -- the same
                        # property the pair-demand pass above relies on.
                        if segment_demand_acc is not None:
                            base = cand[:, :-1]
                            fixed_exp = f.unsqueeze(1).expand(-1, L - 1)
                            if is_vert:
                                seg_ids = self.edge_seg_h_t[base, fixed_exp][diff]
                            else:
                                seg_ids = self.edge_seg_v_t[fixed_exp, base][diff]
                            seg_ids = seg_ids.to(torch.int64)
                            assert bool((seg_ids >= 0).all()), \
                                "a lattice transition mapped to no segment"
                            segment_demand_acc += torch.bincount(
                                seg_ids, minlength=segment_demand_acc.numel())
                            if candidate_keys is not None:
                                nid_exp = nid.unsqueeze(1).expand(-1, L - 1)[diff]
                                u_exp = edge_u[sel][start:end].unsqueeze(1).expand(-1, L - 1)[diff]
                                v_exp = edge_v[sel][start:end].unsqueeze(1).expand(-1, L - 1)[diff]
                                keep = u_exp != v_exp
                                candidate_keys["dropped"] += int((~keep).sum())
                                if bool(keep.any()):
                                    net_k = nid_exp[keep].to(torch.int64)
                                    lo_uv = torch.minimum(u_exp[keep], v_exp[keep]).to(torch.int64)
                                    hi_uv = torch.maximum(u_exp[keep], v_exp[keep]).to(torch.int64)
                                    key = (((net_k * self.k + lo_uv) * self.k + hi_uv)
                                           * self.num_segments + seg_ids[keep])
                                    uniq, counts = torch.unique(key, return_counts=True)
                                    candidate_keys["keys"].append(uniq)
                                    candidate_keys["counts"].append(counts)
```

`edge_u`/`edge_v` are per-selected-segment arrays in the same order as `fixed_idx`, so they are sliced with the same `sel` and `start:end` the other per-row arrays use.

- [ ] **Step 5: Wire `evaluate`**

Change the signature to `def evaluate(self, node_x, node_y, capacity_candidates=False):`. Keep `pin_rid` alive when segments are enabled — replace

```python
        del pin_ix, pin_iy, pin_rid  # only needed to build net_region_key above
```

with

```python
        del pin_ix, pin_iy
        if self.segments is None:
            del pin_rid          # only needed to build net_region_key above
```

Before the edge-batch loop, next to `pair_count_acc`:

```python
            segment_demand_acc = None
            candidate_keys = None
            if self.segments is not None:
                segment_demand_acc = torch.zeros(self.num_segments,
                                                 dtype=torch.int64, device=dev)
                if capacity_candidates:
                    candidate_keys = {"keys": [], "counts": [], "dropped": 0}
```

Inside the batch loop, after `bx, by = self._to_idx(xb, yb)`:

```python
                edge_u = edge_v = None
                if candidate_keys is not None:
                    edge_u = pin_rid[b_pin_a]
                    edge_v = pin_rid[b_pin_b]
```

and extend the two `_process_segments` calls:

```python
                self._process_segments(h_row, h_lo, h_hi, b_net_id, False,
                                       passed_bit_acc, pair_count_acc,
                                       segment_demand_acc, edge_u, edge_v,
                                       candidate_keys)
                self._process_segments(v_col, v_lo, v_hi, b_net_id, True,
                                       passed_bit_acc, pair_count_acc,
                                       segment_demand_acc, edge_u, edge_v,
                                       candidate_keys)
```

After the loop, next to `pair_demand = self._reduce_pair_demand(pair_count_acc)`:

```python
            if segment_demand_acc is not None:
                capacity_fields["segment_demand"] = \
                    segment_demand_acc.cpu().numpy().astype(np.int64)
            if candidate_keys is not None:
                capacity_fields.update(
                    self._reduce_candidates(candidate_keys))
```

with `capacity_fields = {}` initialised just before `M = edge_net_id.numel()`, and the post-loop finishing block placed right before the `return EvalResult(...)`:

```python
        if self.segments is not None:
            demand = capacity_fields.setdefault(
                "segment_demand", np.zeros(self.num_segments, dtype=np.int64))
            # sec 5: the Ph/Pv prefix-sum counts become a free assertion. One
            # device sync on two scalars, not per edge.
            assert int(demand.sum()) == int(per_net_crossings.sum().item()) - large_lb, \
                "per-segment demand does not reconcile with the Ph/Pv crossing count"
            if self.segment_capacity is not None:
                util, scalars = segment_utilisation(demand, self.segment_capacity)
                capacity_fields["segment_capacity"] = self.segment_capacity
                capacity_fields["segment_util"] = util
                capacity_fields.update(
                    (name, scalars[name]) for name in
                    ("num_over_capacity", "num_zero_capacity_segments",
                     "zero_capacity_demand", "max_util", "p99_util"))
            if capacity_candidates and "cand_net" not in capacity_fields:
                empty = np.zeros(0, dtype=np.int64)
                capacity_fields.update(cand_net=empty, cand_u=empty, cand_v=empty,
                                       cand_seg=empty, cand_count=empty,
                                       cand_dropped=0)
```

and `**capacity_fields` spread into `EvalResult(...)`. Add the reducer as a method:

```python
    def _reduce_candidates(self, candidate_keys):
        """Collapse the per-batch unique (net,u,v,seg) keys into one ascending
        list. torch.unique sorts, and the reference builds the same composite
        key with np.unique -- which is what makes the two candidate lists
        bit-identical rather than merely equivalent as sets."""
        empty = np.zeros(0, dtype=np.int64)
        if not candidate_keys["keys"]:
            return dict(cand_net=empty, cand_u=empty, cand_v=empty,
                        cand_seg=empty, cand_count=empty,
                        cand_dropped=int(candidate_keys["dropped"]))
        keys = torch.cat(candidate_keys["keys"])
        counts = torch.cat(candidate_keys["counts"])
        uniq, inverse = torch.unique(keys, return_inverse=True)
        totals = torch.zeros(uniq.numel(), dtype=torch.int64, device=uniq.device)
        totals.index_add_(0, inverse, counts.to(torch.int64))
        uniq = uniq.cpu().numpy().astype(np.int64)
        size = self.num_segments
        seg = uniq % size
        rest = uniq // size
        v = rest % self.k
        rest = rest // self.k
        u = rest % self.k
        net = rest // self.k
        return dict(cand_net=net, cand_u=u, cand_v=v, cand_seg=seg,
                    cand_count=totals.cpu().numpy().astype(np.int64),
                    cand_dropped=int(candidate_keys["dropped"]))
```

Add `from ioplace.region_segments import segment_utilisation` to the module imports, and widen `evaluate_gpu`:

```python
def evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda",
                 segments=None, segment_capacity=None,
                 capacity_candidates=False):
    ctx = GpuEvalContext(nl, rg, device=device, max_degree=max_degree,
                         segments=segments, segment_capacity=segment_capacity)
    return ctx.evaluate(node_x, node_y, capacity_candidates=capacity_candidates)
```

- [ ] **Step 6: Extend the batch-invariance contract**

In `tests/test_evaluator_gpu.py`, append to `_assert_batch_invariant_fields` (after the `boundary_pair_demand` line):

```python
    # v2 P-D: integer capacity fields are bit-exact; segment_util/max_util/
    # p99_util are too, because both come from region_segments.
    # segment_utilisation applied to the same bit-exact segment_demand.
    if a.segment_demand is not None or b.segment_demand is not None:
        np.testing.assert_array_equal(a.segment_demand, b.segment_demand)
```

- [ ] **Step 7: Run the tests**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py tests/test_evaluator_capacity.py -v -m "not slow"
"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py -v -m slow -k segment_demand_regression
```

Expected: PASS, including every pre-existing test in the file. If GPU 3 is still occupied by the acceptance run, this task's tests are the ones to defer — say so rather than running them on a foreign GPU.

- [ ] **Step 8: Commit**

```bash
git add src/ioplace/evaluator_gpu.py tests/test_evaluator_gpu.py tests/data/segment_demand_multirect.json
git commit -m "$(cat <<'MSG'
feat(evaluator): per-segment demand on the GPU, bit-exact against the reference

Read the segment id of each lattice transition straight out of the raster in
_process_segments (the padded gather already holds the coordinates), reconcile
the total against the Ph/Pv crossing count, and collapse per-batch unique
(net,u,v,seg) keys into one ascending candidate list -- the same composite key
and ordering np.unique produces on the reference side. Capacity scalars come
from the shared numpy helper, so segment_util / max_util / p99_util are
bit-exact too, not merely close.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 8: Persistence and the `run_placement_io` driver

**Files:**
- Modify: `src/ioplace/export/evaluation.py:14-18`, `:68-117` (`save_evaluation`), `:120-141` (`load_evaluation`)
- Modify: `src/ioplace/drivers/run_placement.py` (after `_pack_eval_metrics`, ~line 119; `build_parser` ~line 505-534; `main`'s `run_io(...)` call ~line 552-570)
- Modify: `src/ioplace/drivers/run_placement_io.py:33-58` (`RESULT_FIELDS`), `:139-159` (signature), `:255-300` (term construction), `:321-354` (normalizer registration), `:355-383` (`term_fn`), `:520-560` (the callback), `:830-870` (`result`)
- Test: `tests/test_evaluation_export.py` (append), `tests/test_driver_capacity.py` (new)

**Interfaces:**
- Consumes: Tasks 1–7 in full.
- Produces:
  - `export/evaluation.SCHEMA_VERSION` and `SUPPORTED_SCHEMA_VERSIONS` per the **schema-version rule** in Global Constraints
  - `evaluation.npz` arrays `segment_demand` (S,) int64, `segment_capacity` (S,) float64, `segment_util` (S,) float64, plus `metadata["capacity"]` = the seven `CAPACITY_SCALARS` + `"segments_sha256"` + `"capacity_source"` + `"capacity_semantics"`, or `None`
  - `run_placement._pack_capacity_metrics(res) -> dict` keyed exactly by `CAPACITY_SCALARS`
  - CLI `--capacity PATH`, `--cap-tau-b-cells 2.0`, `--cap-m-pairs 4`, `--cap-m-seg 2`, `--cap-curvature-dref 1.0`
  - `run_io(..., capacity=None, cap_tau_b_cells=2.0, cap_m_pairs=4, cap_m_seg=2, cap_curvature_dref=1.0)`
  - `RESULT_FIELDS` gains `capacity`, `capacity_source`, `lambda_cap_final`, `cap_frac_singleton_groups`, and the seven `CAPACITY_SCALARS`

- [ ] **Step 0: Decide the schema version**

Run the inspection command from Global Constraints and write the chosen number into every place below that says `<SCHEMA>`. Do not skip this: picking the wrong branch either orphans P-F's block or silently reuses its version number.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation_export.py`:

```python
def _capacity_evidence(tmp_path, with_capacity=True):
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from ioplace.region_segments import enumerate_segments
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, table = _strip_case()
    capacity = np.array([0.5, 4.0]) if with_capacity else None
    result = evaluate_ref(nl, nl.node_x, nl.node_y, rg, segments=table,
                          segment_capacity=capacity)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y, ["n0"],
                    segments=table)
    return path, nl, rg, result, table


def test_capacity_block_round_trips(tmp_path):
    path, _nl, rg, result, table = _capacity_evidence(tmp_path)
    data = load_evaluation(path, rg=rg)
    block = data["metadata"]["capacity"]
    from ioplace.region_segments import CAPACITY_SCALARS, segments_digest
    for name in CAPACITY_SCALARS:
        assert name in block
    assert block["num_over_capacity"] == 1
    assert block["segment_demand_total"] == 2
    assert block["segments_sha256"] == segments_digest(table)
    assert block["capacity_semantics"] == (
        "usable tracks crossing the segment; "
        "one net crossing consumes one track")
    np.testing.assert_array_equal(data["segment_demand"], result.segment_demand)
    np.testing.assert_array_equal(data["segment_capacity"], [0.5, 4.0])
    np.testing.assert_array_equal(data["segment_util"], result.segment_util)


def test_evidence_without_capacity_records_that_fact(tmp_path):
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, _table = _strip_case()
    result = evaluate_ref(nl, nl.node_x, nl.node_y, rg)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y, ["n0"])
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["capacity"] is None
    assert "segment_demand" not in data


def test_a_corrupted_segment_demand_fails_total_validation(tmp_path):
    import json
    path, _nl, _rg, _result, _table = _capacity_evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["segment_demand"][0] += 5
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="total mismatch: segment_demand"):
        load_evaluation(path)


def test_older_schema_archives_still_load(tmp_path):
    """The SUPPORTED_SCHEMA_VERSIONS convention P-F introduced: results/ holds
    historical evidence the route-calibration tooling still pairs against."""
    import json
    from ioplace.export.evaluation import SUPPORTED_SCHEMA_VERSIONS
    path, _nl, _rg, _result, _table = _capacity_evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema_version"] = SUPPORTED_SCHEMA_VERSIONS[0]
    metadata.pop("capacity")
    for key in ("segment_demand", "segment_capacity", "segment_util"):
        arrays.pop(key)
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    data = load_evaluation(path)
    assert data["metadata"]["schema_version"] == SUPPORTED_SCHEMA_VERSIONS[0]
    assert data["metadata"].get("capacity") is None
```

Create `tests/test_driver_capacity.py`:

```python
import json
import os

import numpy as np
import pytest

from ioplace.region_segments import CAPACITY_SCALARS


def test_pack_capacity_metrics_has_exactly_the_scalar_names():
    from ioplace.drivers.run_placement import _pack_capacity_metrics
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, table = _strip_case()
    res = evaluate_ref(nl, nl.node_x, nl.node_y, rg, segments=table,
                       segment_capacity=np.array([0.5, 4.0]))
    packed = _pack_capacity_metrics(res)
    assert tuple(packed) == CAPACITY_SCALARS
    assert isinstance(packed["num_over_capacity"], int)
    assert isinstance(packed["max_util"], float)
    assert packed["num_over_capacity"] == 1


def test_pack_capacity_metrics_is_all_none_without_a_measurement():
    from ioplace.drivers.run_placement import _pack_capacity_metrics
    from ioplace.evaluator_ref import evaluate as evaluate_ref
    from tests.test_evaluator_capacity import _strip_case
    nl, rg, _table = _strip_case()
    packed = _pack_capacity_metrics(evaluate_ref(nl, nl.node_x, nl.node_y, rg))
    assert tuple(packed) == CAPACITY_SCALARS
    assert set(packed.values()) == {None}


def test_result_fields_carry_every_capacity_scalar():
    from ioplace.drivers.run_placement_io import RESULT_FIELDS
    for name in CAPACITY_SCALARS:
        assert name in RESULT_FIELDS, name
    for name in ("capacity", "capacity_source", "lambda_cap_final",
                 "cap_frac_singleton_groups"):
        assert name in RESULT_FIELDS, name


def test_the_cli_exposes_the_capacity_flags():
    from ioplace.drivers.run_placement import build_parser
    args = build_parser().parse_args(
        ["--mode", "io", "--config", "c.json", "--out", "o.json",
         "--capacity", "cap.npz", "--cap-tau-b-cells", "3",
         "--cap-m-pairs", "2", "--cap-m-seg", "1",
         "--cap-curvature-dref", "0.5"])
    assert args.capacity == "cap.npz"
    assert args.cap_tau_b_cells == 3.0
    assert args.cap_m_pairs == 2 and args.cap_m_seg == 1
    assert args.cap_curvature_dref == 0.5
    default = build_parser().parse_args(
        ["--mode", "io", "--config", "c.json", "--out", "o.json"])
    assert default.capacity is None
    assert default.cap_tau_b_cells == 2.0
    assert (default.cap_m_pairs, default.cap_m_seg) == (4, 2)


def test_capacity_needs_a_matching_geometry(tmp_path):
    """The driver must refuse a capacity.npz built for different regions
    before it touches CUDA."""
    from ioplace.capacity.extract import save_capacity
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import make_grid_regions
    from ioplace.drivers.run_placement_io import load_capacity_for_grid
    die = (0., 0., 100., 100.)
    table = enumerate_segments(RegionGrid(make_grid_regions(die, 2, 2, lattice=20)))
    path = tmp_path / "capacity.npz"
    save_capacity(path, table, np.ones(table.num_segments), source="lef_pitch")
    good = RegionGrid(make_grid_regions(die, 2, 2, lattice=20))
    assert load_capacity_for_grid(str(path), good)[1].shape == \
        (table.num_segments,)
    bad = RegionGrid(make_grid_regions(die, 4, 4, lattice=20))
    with pytest.raises(ValueError, match="different segment table"):
        load_capacity_for_grid(str(path), bad)


@pytest.mark.slow
@pytest.mark.gpu
def test_gcd_end_to_end_with_capacity(tmp_path):
    """sec 9's small end-to-end: build a LEF-fallback capacity.npz for GCD,
    run the IO driver with --capacity, and check that the capacity term was
    actually live -- registered, normalised, and reported."""
    import subprocess
    import sys
    from ioplace.capacity.extract import save_capacity, lef_capacity, \
        parse_tech_lef_layers
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import make_grid_regions
    case_path = "results/route_feedback_20260914/gcd.json"
    if not os.path.exists(case_path):
        pytest.skip("the GCD case description is absent")
    tech = ("/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/"
            "NangateOpenCellLibrary.tech.lef")
    if not os.path.exists(tech):
        pytest.skip("NanGate45 tech LEF absent")

    # The die comes from the same place the driver reads it, so the segment
    # fingerprints match: run one throwaway read first.
    from ioplace.dreamplace_env import setup_dreamplace
    setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    params.load(case_path)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    die = (float(placedb.xl), float(placedb.yl),
           float(placedb.xh), float(placedb.yh))
    scale = float(getattr(placedb, "scale_factor", 1.0) or 1.0)

    rs = make_grid_regions(die, 2, 2, lattice=64)
    table = enumerate_segments(RegionGrid(rs))
    layers = parse_tech_lef_layers(tech)
    capacity = lef_capacity(table, layers, scale_factor=scale,
                            dbu_per_micron=2000., layer_range=("metal2", "metal10"))
    # scale it down so the run actually violates something and the term fires
    capacity = np.maximum(capacity * 0.02, 0.0)
    cap_path = tmp_path / "capacity.npz"
    save_capacity(cap_path, table, capacity, source="lef_pitch")

    out = tmp_path / "result.json"
    subprocess.run(
        [sys.executable, "-m", "ioplace.drivers.run_placement",
         "--mode", "io", "--config", case_path, "--k", "4", "--rtype", "grid",
         "--out", str(out), "--every", "50", "--rho-max", "0.1",
         "--norm-policy", "grandplan", "--callback-order", "atomic",
         "--capacity", str(cap_path)],
        check=True)
    result = json.load(open(out))
    assert result["capacity"] == os.path.abspath(str(cap_path))
    assert result["capacity_source"] == "lef_pitch"
    assert result["num_segments"] == table.num_segments
    assert result["segment_demand_total"] >= 0
    assert result["max_util"] is not None
    assert result["lambda_cap_final"] is not None
    trace = [json.loads(line) for line
             in open(result["norm_trace"]).read().splitlines() if line.strip()]
    assert trace, "no norm_trace rows"
    assert any("cap" in row["terms"] for row in trace)
    assert any(row["terms"]["cap"]["active"] for row in trace)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_driver_capacity.py tests/test_evaluation_export.py -v -m "not slow"`
Expected: FAIL — `ImportError: cannot import name '_pack_capacity_metrics'`.

- [ ] **Step 3: Bump the `evaluation.npz` schema**

In `src/ioplace/export/evaluation.py`, replace lines 14-18 with the branch the Step-0 inspection selected. If P-F has landed:

```python
SCHEMA_VERSION = 3
# v2 P-D: schema 3 adds the sec 5 per-segment capacity block next to P-F's
# schema-2 straddle block. Older archives stay readable -- results/ holds
# historical evidence the route-calibration tooling still pairs against.
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
```

If it has not:

```python
SCHEMA_VERSION = 2
# v2 P-D: schema 2 adds the sec 5 per-segment capacity block. Schema 1
# archives stay readable -- results/ holds historical evidence the
# route-calibration tooling still pairs against. NOTE for P-F: this plan took
# version 2 first; P-F's straddle block must bump to 3 and extend
# SUPPORTED_SCHEMA_VERSIONS to (1, 2, 3).
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
```

Add `from ioplace.region_segments import CAPACITY_SCALARS, segments_digest` to the imports.

Widen `save_evaluation`'s signature to `def save_evaluation(path, nl, rg, result, node_x, node_y, net_names, *, max_degree=256, provenance=None, segments=None, capacity_metadata=None):` and insert, after the `arrays.update(...)` block and before `metadata = dict(...)`:

```python
    # v2 P-D (design sec 5). Present iff the evaluator actually computed the
    # per-segment fields, so "not measured" and "measured as zero" stay
    # distinguishable -- the same convention as P-F's straddle block.
    capacity = None
    if getattr(result, "segment_demand", None) is not None:
        demand = np.asarray(result.segment_demand, dtype=np.int64)
        arrays["segment_demand"] = demand
        capacity = {name: getattr(result, name) for name in CAPACITY_SCALARS}
        capacity = {name: (None if value is None else
                           (int(value) if isinstance(value, (int, np.integer))
                            else float(value)))
                    for name, value in capacity.items()}
        capacity["num_segments"] = int(demand.size)
        capacity["segment_demand_total"] = int(demand.sum())
        if segments is not None:
            capacity["segments_sha256"] = segments_digest(segments)
        if getattr(result, "segment_capacity", None) is not None:
            arrays["segment_capacity"] = np.asarray(result.segment_capacity,
                                                    dtype=np.float64)
            arrays["segment_util"] = np.asarray(result.segment_util,
                                                dtype=np.float64)
        capacity.update(capacity_metadata or {})
```

and add `capacity=capacity,` to the `metadata = dict(...)` literal.

In `load_evaluation`, replace the version check with

```python
    if metadata.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported evaluator evidence schema")
```

and insert, after the existing per-net total checks:

```python
    capacity = metadata.get("capacity")
    if capacity is not None:
        if "segment_demand" not in data:
            raise ValueError("invalid evaluator evidence array: segment_demand")
        demand = data["segment_demand"]
        if demand.shape != (capacity["num_segments"],):
            raise ValueError("invalid evaluator evidence array: segment_demand")
        if int(demand.sum(dtype=np.int64)) != capacity["segment_demand_total"]:
            raise ValueError("evaluator total mismatch: segment_demand")
        if "segment_capacity" in data:
            if data["segment_capacity"].shape != demand.shape or \
                    data["segment_util"].shape != demand.shape:
                raise ValueError("invalid evaluator evidence array: segment_capacity")
            over = int((demand > data["segment_capacity"]).sum())
            if over != capacity["num_over_capacity"]:
                raise ValueError("evaluator total mismatch: num_over_capacity")
```

- [ ] **Step 4: Add the metric packer and the CLI flags**

In `src/ioplace/drivers/run_placement.py`, after `_pack_eval_metrics`:

```python
def _pack_capacity_metrics(res):
    """v2 P-D (design sec 5). Deliberately separate from _pack_eval_metrics,
    which also feeds run_flat / run_two_stage / run_reweight -- none of those
    has a RESULT_FIELDS gate, so widening it would silently change three other
    drivers' result.json. Every value is None when the run carried no
    capacity.npz, so a missing measurement is visible rather than zero."""
    from ioplace.region_segments import CAPACITY_SCALARS
    out = {}
    for name in CAPACITY_SCALARS:
        value = getattr(res, name, None)
        if value is None:
            out[name] = None
        elif isinstance(value, (int, np.integer)):
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out
```

In `build_parser`, after the `--norm-trace` argument:

```python
    # v2 P-D (design sec 5): per-segment boundary IO capacity. mode=io only.
    ap.add_argument("--capacity", default=None,
                    help="capacity.npz from ioplace.capacity.extract; enables "
                         "the per-segment capacity term and the evaluator's "
                         "per-segment hard check")
    ap.add_argument("--cap-tau-b-cells", type=float, default=2.,
                    help="alpha softmax temperature in lattice cell widths "
                         "(design sec 5: tau_b = 2*cell_w)")
    ap.add_argument("--cap-m-pairs", type=int, default=4,
                    help="candidate alpha-groups kept per net (design sec 5)")
    ap.add_argument("--cap-m-seg", type=int, default=2,
                    help="candidate segments kept per group (design sec 5)")
    ap.add_argument("--cap-curvature-dref", type=float, default=1.,
                    help="declared curvature reference: the term registers "
                         "pen''(d_ref) = 12*d_ref+2 with the TermNormalizer")
```

and in `main`'s `run_io(...)` call add

```python
              capacity=args.capacity, cap_tau_b_cells=args.cap_tau_b_cells,
              cap_m_pairs=args.cap_m_pairs, cap_m_seg=args.cap_m_seg,
              cap_curvature_dref=args.cap_curvature_dref,
```

- [ ] **Step 5: Wire `run_placement_io.run_io`**

**5a. Signature and validation.** Add to `run_io`'s keyword tail (`:157-159`): `capacity=None, cap_tau_b_cells=2.0, cap_m_pairs=4, cap_m_seg=2, cap_curvature_dref=1.0,` and, next to the other validations:

```python
    if cap_m_pairs < 1 or cap_m_seg < 1 or cap_tau_b_cells <= 0:
        raise ValueError("cap_m_pairs/cap_m_seg must be >= 1 and "
                         "cap_tau_b_cells > 0")
    if capacity is not None and callback_order != "atomic":
        raise ValueError("the capacity term needs atomic callbacks "
                         "(--callback-order atomic)")
```

**5b. A module-level loader** (so the geometry check is unit-testable without CUDA) — put it next to `RESULT_FIELDS`:

```python
def load_capacity_for_grid(path, rg):
    """(SegmentTable, capacity, metadata) for this run's region grid. Raises
    before any CUDA allocation if capacity.npz was built for a different
    geometry -- the most damaging silent failure in P-D."""
    from ioplace.capacity.extract import load_capacity
    from ioplace.region_segments import enumerate_segments
    table = enumerate_segments(rg)
    data = load_capacity(path, table=table)
    return table, data["capacity"], data["metadata"]
```

and re-raise `load_capacity`'s message unchanged, which already reads `capacity.npz was built for a different segment table`.

**5c. Build the table and the term.** Right after `rg = RegionGrid(rs)` (`:272`), before `ctx = GpuEvalContext(...)`:

```python
        segment_table = cap_meta = None
        segment_capacity = None
        if capacity is not None:
            segment_table, segment_capacity, cap_meta = \
                load_capacity_for_grid(capacity, rg)
        ctx = GpuEvalContext(nl, rg, device="cuda", segments=segment_table,
                             segment_capacity=segment_capacity)
```

(replacing the existing `ctx = GpuEvalContext(nl, rg, device="cuda")`).

After `io_term` is built (`:296`):

```python
        cap_term = None
        net_to_active = None
        if segment_table is not None:
            from ioplace.ops.cap_term import CapNormTerm, CapTerm
            cap_term = CapTerm(io_term, segment_table.box, segment_capacity,
                               tau_b=cap_tau_b_cells * float(rg.cell_w),
                               curvature_dref=cap_curvature_dref)
            # global net id -> IO-CSR active net index; the IO CSR drops nets
            # above ignore_net_degree and nets collapsing to one node, and the
            # capacity term only carries the nets it can differentiate.
            net_to_active = np.full(nl.num_nets, -1, dtype=np.int64)
            net_to_active[csr.net_ids] = np.arange(len(csr.net_ids),
                                                    dtype=np.int64)
```

**5d. Observer mode and registration.** Widen the `observer_mode` predicate (`:319`) to `and capacity is None`, and after the `ft` registration (`:351-354`):

```python
        if cap_term is not None:
            # design sec 4/5: curvature = pen''(d_ref), activation at
            # overflow <= 0.30, and -- unlike io/ft -- the term stays on after
            # the freeze (sec 3), which is phase 3's business in run_main_flow.
            normalizer.register("cap", CapNormTerm(cap_term),
                                cap_term.curvature,
                                target_share=shares.get("cap", 0.1),
                                activate_overflow=0.30, n_ramp=state.n_ramp)
```

**5e. `term_fn`.** Restructure the tail of `term_fn` (`:373-382`) so the capacity term is not gated by the IO term's activation — `state.active` is the IO schedule's latch, and capacity has its own:

```python
            lam_cap = (0.0 if cap_term is None
                       else normalizer.lambdas.get("cap", 0.0))
            io_off = not state.active or (lam_io == 0.0
                                          and state.lambda_margin == 0.0)
            if io_off and lam_cap == 0.0:
                return pos.new_zeros(())
            if io_off:
                base = pos.new_zeros(())
            elif ft_term is not None:
                base = ft_term(pos, state.tau, lam_io, kappa,
                               state.lambda_margin, state.margin_m,
                               state.margin_tau)
            else:
                base = io_term(pos, state.tau, lam_io, state.lambda_margin,
                               state.margin_m, state.margin_tau)
            if lam_cap != 0.0:
                base = base + cap_term(pos, state.tau, lam_cap)
            return base
```

**5f. Candidate refresh, on the `home_period` cadence.** In the atomic branch of the callback, right after the `ft_term.set_home(...)` block (`:545-550`):

```python
                    if cap_term is not None and (
                            cap_term.cand_net.numel() == 0
                            or iteration % home_period == 0):
                        # sec 5: "Every home_period iterations the evaluator
                        # emits, per net, the (pair, segment) items it actually
                        # crossed, capped at m_pairs=4, m_seg=2" -- the same
                        # frozen-discrete/continuous split `home` already uses.
                        from ioplace.region_segments import select_candidates
                        raw = select_candidates(res.cand_net, res.cand_u,
                                                res.cand_v, res.cand_seg,
                                                res.cand_count, segment_table,
                                                m_pairs=cap_m_pairs,
                                                m_seg=cap_m_seg)
                        cap_term.set_candidates(raw.remap_nets(net_to_active))
                        cap_diag = cap_term.diagnostics(pos.detach(), state.tau)
                        entry.update(
                            cap_num_candidates=cap_diag["num_candidates"],
                            cap_num_groups=cap_diag["num_groups"],
                            cap_frac_singleton_groups=cap_diag["frac_singleton_groups"],
                            cap_max_d=cap_diag["max_d"],
                            cap_dropped=int(res.cand_dropped))
```

and make the periodic evaluator call ask for candidates on that cadence — replace `res = ctx.evaluate(node_x, node_y)` (`:520`) with

```python
                want_candidates = (cap_term is not None
                                   and (cap_term.cand_net.numel() == 0
                                        or iteration % home_period == 0))
                res = ctx.evaluate(node_x, node_y,
                                   capacity_candidates=want_candidates)
```

and add `lambda_cap` to the entry next to `lambda_ft`:

```python
                    entry["lambda_cap"] = txn.lambdas.get("cap", 0.)
```

**5g. Result fields.** Add to `RESULT_FIELDS`:

```python
                 # v2 P-D (design sec 5): per-segment boundary IO capacity.
                 "capacity", "capacity_source", "lambda_cap_final",
                 "cap_frac_singleton_groups", "cap_tau_b_cells",
                 "cap_m_pairs", "cap_m_seg", "cap_curvature_dref",
                 "num_segments", "segment_demand_total", "num_over_capacity",
                 "num_zero_capacity_segments", "zero_capacity_demand",
                 "max_util", "p99_util",
```

and to the `result = {...}` literal:

```python
            "capacity": os.path.abspath(capacity) if capacity else None,
            "capacity_source": (cap_meta or {}).get("capacity_source"),
            "lambda_cap_final": (None if cap_term is None
                                 else float(normalizer.lambdas.get("cap", 0.))),
            "cap_frac_singleton_groups": cb_state["cap_frac_singleton_groups"],
            "cap_tau_b_cells": cap_tau_b_cells, "cap_m_pairs": cap_m_pairs,
            "cap_m_seg": cap_m_seg, "cap_curvature_dref": cap_curvature_dref,
            **_pack_capacity_metrics(res),
```

`res` is the final `EvalResult` from the `eval` phase (`run_placement_io.py:784`), and because `ctx` was constructed with `segments=segment_table` it already carries the per-segment fields — no second evaluator call. `cap_frac_singleton_groups` is carried forward from the callback rather than recomputed: add `"cap_frac_singleton_groups": None` to the `cb_state` dict literal (`:468-479`) and write it in the refresh block of Step 5f:

```python
                        cb_state["cap_frac_singleton_groups"] = \
                            cap_diag["frac_singleton_groups"]
```

Recomputing it at the end would need a `2*num_nodes` `pos` tensor that no longer exists after `placer` returns; the callback already has one.

Finally, pass the segment table into the evidence writer (`:806-810`), replacing the existing call with:

```python
                save_evaluation(emit_eval, netlist_from_placedb(placedb), rg,
                                res, node_x, node_y, placedb.net_names,
                                provenance={"config": os.path.abspath(config_json),
                                            "placement_stage": "gp_lg",
                                            "def_directory": os.path.abspath(emit_def) if emit_def else None},
                                segments=segment_table,
                                capacity_metadata=(
                                    None if cap_meta is None else
                                    {"capacity_source": cap_meta["capacity_source"],
                                     "capacity_semantics": cap_meta["capacity_semantics"],
                                     "segments_sha256": cap_meta["segments_sha256"]}))
```

- [ ] **Step 6: Run the fast tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_driver_capacity.py tests/test_evaluation_export.py tests/test_driver_io.py -v -m "not slow"`
Expected: PASS, including the pre-existing `test_driver_io.py` (every new argument defaults to off).

- [ ] **Step 7: Run the slow GCD end-to-end**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_driver_capacity.py -v -m slow
```

Expected: PASS. If it reports `cap` never activating, check the overflow trajectory reached 0.30 — with `stop_overflow=0.1` on GCD it must. If `max_util` is `0.0` the capacity scaling in the test is too generous; the assertion set deliberately does not require a violation, only that the term was live.

- [ ] **Step 8: Run the whole suite and commit**

```bash
"$IOPLACE_PYTHON" -m pytest
git add src/ioplace/export/evaluation.py src/ioplace/drivers/run_placement.py \
        src/ioplace/drivers/run_placement_io.py tests/test_driver_capacity.py \
        tests/test_evaluation_export.py
git commit -m "$(cat <<'MSG'
feat(driver): --capacity end to end in run_placement_io

evaluation.npz grows a schema-<SCHEMA> capacity block (segment_demand /
segment_capacity / segment_util plus the seven scalars), _pack_capacity_metrics
mirrors P-F's packer convention, and run_io builds the CapTerm, registers it
with the TermNormalizer at curvature pen''(d_ref) and activate_overflow=0.30,
refreshes its candidates from the evaluator every home_period, and applies
lambda_cap in term_fn independently of the IO term's own activation latch.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 9 (guarded on P-B): capacity in the main flow's fence phase

**Precondition.** `src/ioplace/drivers/run_main_flow.py` must exist (P-B Task 7). Check first:

```bash
test -f src/ioplace/drivers/run_main_flow.py && echo "P-B landed" || echo "BLOCKED"
```

If it prints `BLOCKED`, **stop this task**, record it as blocked (not done, not skipped) in the task tracker with the exact reason, and move to Task 10. Tasks 1–8 and 10 do not depend on it.

**Files:**
- Modify: `src/ioplace/drivers/run_main_flow.py` — `run_fence_gp` (signature, phase-3 body) and `run_main_flow` (signature, pass-through, CLI)
- Modify: `src/ioplace/artifacts.py` — `MAIN_FLOW_RESULT_FIELDS`
- Test: `tests/test_main_flow_capacity.py` (new)

**Interfaces:**
- Consumes: Task 8's `load_capacity_for_grid`, `_pack_capacity_metrics`; Tasks 1–5.
- Produces: `run_fence_gp(..., capacity=None, cap_tau_b_cells=2.0, cap_m_pairs=4, cap_m_seg=2, cap_curvature_dref=1.0, norm_policy="grandplan", norm_probe_every=50)` and the same five on `run_main_flow`; `--capacity` and the four `--cap-*` flags on the main-flow CLI; `MAIN_FLOW_RESULT_FIELDS` gains the same names Task 8 added to `RESULT_FIELDS`.

**Why a `capacity=` kwarg and not just `extra_terms`.** P-B documents `run_fence_gp(..., extra_terms=())` as *"the documented attachment point for P-D's capacity term and P-E's pseudo-FT term"*, and `extra_terms` is a list of `term_fn(pos)` callables. But the capacity term cannot be built by the caller: it needs phase 3's own `nl`, `rg` and `IoTerm`, none of which exist until `run_fence_gp` has rebuilt the PlaceDB (`fence_inject.py:9-16` forces the two-instance split). So P-D adds a `capacity=` path kwarg, builds the term *inside* `run_fence_gp` after `rg = RegionGrid(rs_scaled)`, and **appends** its `term_fn` to whatever `extra_terms` the caller passed. The hook is honoured, not bypassed, and P-E can do the same for pseudo-FT without either plan editing the other's code.

**Why phase 3 needs its own normalizer.** Spec §3: after the freeze IO and FT are **off** and capacity is **on**; spec §4: `cap` registers with the `TermNormalizer`. P-B's phase 3 has no normalizer at all because it attaches no terms. So this task instantiates one carrying only `cap`, probing the fence GP's own WL op on the same `probe_every` cadence — a direct mirror of `run_placement_io.py:560-590`'s non-legacy branch, not a new mechanism. Policy `legacy` is rejected here: it exists only to reproduce the retired λ_IO/κ_FT path, and there is no IO term in phase 3 for it to reproduce.

**Where the post-freeze gradient comes from.** Read limitation D-4 in Task 5 before writing this. With membership frozen and cells fenced, `q` is saturated and the entire useful gradient is the `α` path, which only exists for α-groups holding ≥2 alternatives. `run_fence_gp` must therefore log `cap_frac_singleton_groups` into its returned dict, and Task 10's result document must quote it — if it is ~1.0, this task's term is inert and that is the finding.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_flow_capacity.py`:

```python
import inspect
import os

import numpy as np
import pytest

pytest.importorskip("ioplace.drivers.run_main_flow",
                    reason="P-B has not landed")


def test_run_fence_gp_accepts_a_capacity_path():
    from ioplace.drivers.run_main_flow import run_fence_gp
    signature = inspect.signature(run_fence_gp)
    for name, default in (("capacity", None), ("cap_tau_b_cells", 2.0),
                          ("cap_m_pairs", 4), ("cap_m_seg", 2),
                          ("cap_curvature_dref", 1.0),
                          ("norm_policy", "grandplan"),
                          ("norm_probe_every", 50)):
        assert name in signature.parameters, name
        assert signature.parameters[name].default == default
    assert "extra_terms" in signature.parameters


def test_run_main_flow_passes_capacity_through():
    from ioplace.drivers.run_main_flow import run_main_flow
    signature = inspect.signature(run_main_flow)
    for name in ("capacity", "cap_tau_b_cells", "cap_m_pairs", "cap_m_seg",
                 "cap_curvature_dref"):
        assert name in signature.parameters, name
    source = inspect.getsource(run_main_flow)
    assert "capacity=capacity" in source


def test_main_flow_result_fields_carry_the_capacity_scalars():
    from ioplace.artifacts import MAIN_FLOW_RESULT_FIELDS
    from ioplace.region_segments import CAPACITY_SCALARS
    for name in CAPACITY_SCALARS:
        assert name in MAIN_FLOW_RESULT_FIELDS, name
    for name in ("capacity", "capacity_source", "lambda_cap_final",
                 "cap_frac_singleton_groups"):
        assert name in MAIN_FLOW_RESULT_FIELDS, name


def test_the_fence_phase_rejects_the_legacy_norm_policy_with_capacity(tmp_path):
    """Policy 'legacy' reproduces the retired lambda_IO/kappa_FT path; phase 3
    has no IO term, so it has nothing to reproduce."""
    from ioplace.drivers.run_main_flow import run_fence_gp
    with pytest.raises(ValueError, match="legacy"):
        run_fence_gp("c.json", str(tmp_path), region_set=None, part=None,
                     positions=None, reference_density_weight=1.0, k=4,
                     capacity="cap.npz", norm_policy="legacy")


def test_the_cli_exposes_capacity():
    from ioplace.drivers.run_main_flow import build_parser
    args = build_parser().parse_args(
        ["--config", "c.json", "--out-dir", "o", "--capacity", "cap.npz"])
    assert args.capacity == "cap.npz"
    assert args.cap_m_pairs == 4 and args.cap_m_seg == 2


@pytest.mark.slow
@pytest.mark.gpu
def test_gcd_main_flow_with_capacity_reports_the_block(tmp_path):
    """The same GCD end-to-end as Task 8, through the two-phase main flow, so
    the post-freeze branch is actually exercised."""
    import json
    from tests.test_driver_capacity import test_gcd_end_to_end_with_capacity \
        as _fast_path            # reuse its capacity.npz construction comments
    pytest.importorskip("ioplace.drivers.run_main_flow")
    from ioplace.drivers.run_main_flow import run_main_flow
    case_path = "results/route_feedback_20260914/gcd.json"
    if not os.path.exists(case_path):
        pytest.skip("the GCD case description is absent")
    cap_path = os.environ.get("IOPLACE_GCD_CAPACITY_NPZ")
    if not cap_path or not os.path.exists(cap_path):
        pytest.skip("set IOPLACE_GCD_CAPACITY_NPZ to a capacity.npz for GCD "
                    "(Task 8 Step 7 writes one)")
    result = run_main_flow(case_path, str(tmp_path), k=4, rtype="grid",
                           norm_policy="grandplan", capacity=cap_path)
    assert result["capacity"] == os.path.abspath(cap_path)
    assert result["num_segments"] > 0
    assert result["cap_frac_singleton_groups"] is not None
    assert result["lambda_cap_final"] is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_capacity.py -v -m "not slow"`
Expected: FAIL on the signature assertions (or a clean module-level skip if P-B has not landed, in which case this task is blocked — see the precondition).

- [ ] **Step 3: Extend `run_fence_gp`**

Add to its keyword-only tail: `capacity=None, cap_tau_b_cells=2.0, cap_m_pairs=4, cap_m_seg=2, cap_curvature_dref=1.0, norm_policy="grandplan", norm_probe_every=50,` and at the top of the body:

```python
    if capacity is not None and norm_policy == "legacy":
        raise ValueError("the capacity term cannot use norm_policy 'legacy': "
                         "that policy exists to reproduce the retired "
                         "lambda_IO/kappa_FT path, and phase 3 has no IO term")
```

Immediately after `rg = RegionGrid(rs_scaled)` (before `ctx = GpuEvalContext(nl, rg, device="cuda")`), replace that context construction with:

```python
        segment_table = cap_meta = segment_capacity = None
        if capacity is not None:
            from ioplace.drivers.run_placement_io import load_capacity_for_grid
            segment_table, segment_capacity, cap_meta = \
                load_capacity_for_grid(capacity, rg)
        ctx = GpuEvalContext(nl, rg, device="cuda", segments=segment_table,
                             segment_capacity=segment_capacity)
```

After `install_density_weight_clamp(...)` and **before** the `if extra_terms:` line, build the term and its normalizer:

```python
        cap_state = {"term": None, "normalizer": None, "net_to_active": None,
                     "table": segment_table, "meta": cap_meta,
                     "frac_singleton_groups": None}
        terms = list(extra_terms)
        if segment_table is not None:
            # design sec 3: after the freeze IO and FT are OFF and capacity is
            # ON. The term is built here, not by the caller, because it needs
            # phase 3's own netlist/grid/IoTerm -- none of which exist until
            # this PlaceDB has been rebuilt (fence_inject.py:9-16).
            from ioplace.norm import TermNormalizer
            from ioplace.norm_trace import NormTraceWriter
            from ioplace.ops.cap_term import CapNormTerm, CapTerm
            from ioplace.ops.io_term import IoTerm, build_net_node_csr
            from ioplace.ops.soft_assign import rect_table
            rects, r2k = rect_table(rs_scaled)
            csr = build_net_node_csr(nl, int(params.ignore_net_degree))
            io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, k_or_K=None) \
                if False else IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                                     num_movable=nl.num_movable,
                                     num_physical=nl.num_physical,
                                     num_nodes=placedb.num_nodes, device="cuda")
            cap_term = CapTerm(io_term, segment_table.box, segment_capacity,
                               tau_b=cap_tau_b_cells * float(rg.cell_w),
                               curvature_dref=cap_curvature_dref)
            net_to_active = np.full(nl.num_nets, -1, dtype=np.int64)
            net_to_active[csr.net_ids] = np.arange(len(csr.net_ids),
                                                    dtype=np.int64)
            trace = NormTraceWriter(os.path.join(out_dir, "norm_trace.jsonl"))
            cleanup.callback(trace.close)
            normalizer = TermNormalizer(
                policy=norm_policy, norm_p=1, probe_every=norm_probe_every,
                num_movable=placedb.num_movable_nodes,
                num_nodes=placedb.num_nodes,
                track_cancellation=(placedb.num_nodes <= 5_000_000),
                trace=trace)
            normalizer.register("cap", CapNormTerm(cap_term),
                                cap_term.curvature, target_share=0.1,
                                activate_overflow=0.30)
            cap_state.update(term=cap_term, normalizer=normalizer,
                             net_to_active=net_to_active)

            def cap_term_fn(pos):
                lam = normalizer.lambdas.get("cap", 0.0)
                if lam == 0.0:
                    return pos.new_zeros(())
                # tau is irrelevant to the post-freeze alpha path but the
                # signature needs one; the fence GP has no tau schedule, so use
                # the smallest value the soft phase ever reached, which is what
                # "membership is frozen" means numerically.
                return cap_term(pos, cap_state["tau"], lam)

            cap_state["tau"] = 0.05 * ((rg.die[2] - rg.die[0])
                                       * (rg.die[3] - rg.die[1]) / k) ** 0.5
            terms.append(cap_term_fn)
        if terms:
            attach_terms(params, terms)
```

(delete the original `if extra_terms: attach_terms(params, list(extra_terms))`).

Replace P-B's minimal callback with one that drives the normalizer and the candidate refresh:

```python
        def cb(iteration, pos):
            cb_state["last_iteration"] = iteration
            cap_term = cap_state["term"]
            if cap_term is None:
                return
            normalizer = cap_state["normalizer"]
            overflow = float(placer.model.overflow.max())
            gamma = float(placer.model.gamma)
            if iteration % norm_probe_every:
                return
            n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
            res = ctx.evaluate(pos.data[:n_phys],
                               pos.data[n_all:n_all + n_phys],
                               capacity_candidates=True)
            raw = select_candidates(res.cand_net, res.cand_u, res.cand_v,
                                    res.cand_seg, res.cand_count,
                                    cap_state["table"], m_pairs=cap_m_pairs,
                                    m_seg=cap_m_seg)
            cap_term.set_candidates(raw.remap_nets(cap_state["net_to_active"]))
            ctx_norm = {"iteration": iteration, "overflow": overflow,
                        "tau": cap_state["tau"], "gamma": gamma}
            normalizer.probe(iteration, pos,
                             placer.model.op_collections.wirelength_op, ctx_norm)
            normalizer.transaction(iteration, overflow, cap_state["tau"], gamma)
            if normalizer.needs_refresh():
                refresh_nesterov_secant(placer.optimizer)
                normalizer.mark_refreshed()
            cap_state["frac_singleton_groups"] = cap_term.diagnostics(
                pos.detach(), cap_state["tau"])["frac_singleton_groups"]
```

with `from ioplace.dp_hook import refresh_nesterov_secant` and `from ioplace.region_segments import select_candidates` added to the module imports. Add to the returned dict:

```python
            "capacity": os.path.abspath(capacity) if capacity else None,
            "capacity_source": (cap_meta or {}).get("capacity_source"),
            "lambda_cap_final": (None if cap_state["normalizer"] is None else
                                 float(cap_state["normalizer"].lambdas.get("cap", 0.))),
            "cap_frac_singleton_groups": cap_state["frac_singleton_groups"],
            "cap_tau_b_cells": cap_tau_b_cells, "cap_m_pairs": cap_m_pairs,
            "cap_m_seg": cap_m_seg, "cap_curvature_dref": cap_curvature_dref,
            **_pack_capacity_metrics(res),
```

using the same final `res` the `eval` phase already produces for `_pack_eval_metrics`, and pass `segments=segment_table` plus the `capacity_metadata` dict into that phase's `save_evaluation(...)` call exactly as Task 8 Step 5g does for `run_placement_io`.

Clean up the deliberately ugly `IoTerm(...) if False else IoTerm(...)` above into a single call — it is written that way here only to make the full keyword list visible in one place.

- [ ] **Step 4: Thread `run_main_flow` and `artifacts`**

Add `capacity=None, cap_tau_b_cells=2.0, cap_m_pairs=4, cap_m_seg=2, cap_curvature_dref=1.0,` to `run_main_flow`'s keyword tail, forward all five into the `run_fence_gp(...)` call as `capacity=capacity, cap_tau_b_cells=cap_tau_b_cells, ...` plus `norm_policy=("grandplan" if norm_policy == "legacy" else norm_policy), norm_probe_every=50`, and add the matching CLI flags to the main-flow `build_parser` with the same help strings Task 8 used. Extend `artifacts.MAIN_FLOW_RESULT_FIELDS` with the same names Task 8 added to `RESULT_FIELDS`, and add the `capacity.npz` entry to the artefact dict the driver writes:

```python
                     ("capacity_npz", capacity),
```

inside the existing `artefacts = {...}` comprehension's source tuple, guarded so a `None` path is skipped.

- [ ] **Step 5: Run the tests**

```bash
"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_capacity.py tests/test_main_flow_driver.py tests/test_artifacts.py -v -m "not slow"
```

Expected: PASS. Then, with a GCD `capacity.npz` from Task 8 and a free GPU:

```bash
export CUDA_VISIBLE_DEVICES=3 IOPLACE_GCD_CAPACITY_NPZ=/path/to/capacity.npz
"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_capacity.py -v -m slow
```

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/drivers/run_main_flow.py src/ioplace/artifacts.py \
        tests/test_main_flow_capacity.py
git commit -m "$(cat <<'MSG'
feat(main-flow): capacity term stays on after the freeze

run_fence_gp takes a capacity.npz path, builds the CapTerm against phase 3's
own PlaceDB (the caller cannot: the fence split forces a rebuild), appends its
term_fn to extra_terms, and drives it with a TermNormalizer carrying only
`cap` -- design sec 3's "IO off, FT off, capacity on". Reports
cap_frac_singleton_groups so the post-freeze usefulness of the term is a
measured number rather than an assumption (limitation D-4).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 10: The surrogate-vs-router rank correlation — D's "done" criterion

**Files:**
- Create: `src/scripts/run_cap_rank_correlation.py`
- Create: `docs/results/2026-09-19-p-d-capacity-rank-correlation.md`
- Test: `tests/test_cap_rank_correlation.py`

**Interfaces:**
- Consumes: Tasks 1–8 in full; `ioplace.route_eval.online_openroad.{run_openroad, load_observation}`; `ioplace.route_eval.topology.segment_union`; `ioplace.regions.RegionSet`; `ioplace.capacity.extract.load_capacity`.
- Produces:
  - `spearman(a, b) -> float` (numpy, average ranks for ties)
  - `route_segment_demand(rg, table, polylines) -> (S,) int64`
  - `surrogate_demand(...) -> (S,) float64` — `CapTerm.demand` at the final GP positions
  - `main(argv=None)` writing `rank_correlation.json`

**What §9 asks for.** *"D: per-segment utilisation reported from both the GP surrogate and the router, with their rank correlation stated."* And §10 open question (i) pre-registers the decision: *"Does `D_s` rank-correlate with routed per-segment crossings? — Spearman on group from one GRT; below 0.5, fall back to pair-level `D_ab`/`C_ab`."* So the script must emit three series over the same segment ids and two correlations:

1. `surrogate` — the differentiable `D_s` from `CapTerm.demand` at the final GP positions.
2. `evaluator` — the exact `segment_demand` from the GPU evaluator's MST-L-route proxy on the same positions.
3. `router` — actual per-segment crossings from one `global_route`, decoded through `grt::write_segments` and mapped through the §5 raster.

Reporting both `spearman(surrogate, router)` (the number §10 (i) judges) and `spearman(evaluator, router)` separates two failure modes that otherwise look identical: a bad *surrogate* (the soft `q·α` model) and a bad *proxy* (MST L-routes vs real GR topology). Only the first would justify the pair-level fallback.

**Scope.** `mempool_tile_wrap` or `mempool_group` per §9; `results/route_feedback_20260914/gcd.json` is the fast smoke path the unit test uses. This is a `slow`, GPU- and OpenROAD-gated script, run by hand, not in CI.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cap_rank_correlation.py`:

```python
import numpy as np
import pytest

from ioplace.region_grid import RegionGrid
from ioplace.region_segments import enumerate_segments
from ioplace.regions import make_grid_regions

DIE = (0., 0., 100., 100.)


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr
    from src.scripts.run_cap_rank_correlation import spearman
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=50)
        b = 0.7 * a + rng.normal(size=50)
        assert spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic),
                                               rel=1e-10)


def test_spearman_handles_ties_with_average_ranks():
    from scipy.stats import spearmanr
    from src.scripts.run_cap_rank_correlation import spearman
    a = np.array([1., 1., 1., 2., 3.])
    b = np.array([5., 4., 4., 2., 1.])
    assert spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic),
                                           rel=1e-10)


def test_spearman_of_a_constant_series_is_nan_not_a_crash():
    from src.scripts.run_cap_rank_correlation import spearman
    assert np.isnan(spearman(np.ones(10), np.arange(10.)))


def test_route_segment_demand_counts_polyline_crossings():
    from src.scripts.run_cap_rank_correlation import route_segment_demand
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=4))
    table = enumerate_segments(rg)
    # one horizontal wire at y=10 crossing the vertical 0|1 boundary at x=50,
    # and one vertical wire at x=90 crossing the horizontal 1|3 boundary.
    polylines = np.array([[[10., 10.], [90., 10.]],
                          [[90., 10.], [90., 90.]]], dtype=np.float64)
    demand = route_segment_demand(rg, table, polylines)
    assert int(demand.sum()) == 2
    crossed = np.nonzero(demand)[0]
    pairs = sorted((int(table.pair_a[s]), int(table.pair_b[s])) for s in crossed)
    assert pairs == [(0, 1), (1, 3)]


def test_route_segment_demand_deduplicates_within_a_net():
    """segment_union collapses overlapping collinear GRT segments, so one net's
    doubled-back wire is not counted twice -- the same convention
    route_eval.topology.union_metrics uses for crossings."""
    from src.scripts.run_cap_rank_correlation import route_segment_demand
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=4))
    table = enumerate_segments(rg)
    polylines = np.array([[[10., 10.], [90., 10.]],
                          [[20., 10.], [80., 10.]]], dtype=np.float64)
    assert int(route_segment_demand(rg, table, polylines).sum()) == 1


def test_the_result_document_template_exists_and_names_the_decision_rule():
    import os
    path = "docs/results/2026-09-19-p-d-capacity-rank-correlation.md"
    assert os.path.exists(path)
    text = open(path).read()
    assert "spearman_surrogate_router" in text
    assert "0.5" in text and "pair-level" in text
    assert "frac_singleton_groups" in text
    assert "uint8" in text          # sec 10 risk 5 caveat must be carried
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_cap_rank_correlation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.scripts.run_cap_rank_correlation'`.

- [ ] **Step 3: Write `src/scripts/run_cap_rank_correlation.py`**

```python
"""Does the differentiable per-segment demand predict the router's?

v2 design sec 9's "done" criterion for subproject D -- "per-segment
utilisation reported from both the GP surrogate and the router, with their
rank correlation stated" -- and sec 10 open question (i)'s deciding
experiment: "Spearman on group from one GRT; below 0.5, fall back to
pair-level D_ab/C_ab".

Three series over the same segment ids:

  surrogate  CapTerm.demand at the final GP positions -- the soft
             sum w_e q_u q_v alpha the objective actually minimises.
  evaluator  the GPU evaluator's exact segment_demand on the same positions --
             MST L-routes, no router.
  router     actual crossings from one OpenROAD global_route, decoded through
             grt::write_segments and mapped through the sec 5 raster.

Reporting spearman(surrogate, router) *and* spearman(evaluator, router)
separates a bad surrogate from a bad routing proxy. Only the first justifies
the pair-level fallback sec 10 (i) pre-registers.

Slow, GPU- and OpenROAD-gated. Run by hand:

  source src/scripts/env.sh && source src/scripts/openroad_env.sh
  export CUDA_VISIBLE_DEVICES=3
  "$IOPLACE_PYTHON" src/scripts/run_cap_rank_correlation.py \
      --run-dir runs/mempool_group/ours --config <config.json> \
      --regions runs/.../regions.json --capacity runs/.../capacity.npz \
      --out docs/results/2026-09-19-p-d-capacity-rank-correlation.json
"""
import argparse
import json
import os

import numpy as np


def _rankdata(values):
    """Average ranks, ties included -- scipy.stats.rankdata's default,
    reimplemented so this script does not pull scipy into the runtime path."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < values.size:
        stop = index + 1
        while stop < values.size and sorted_values[stop] == sorted_values[index]:
            stop += 1
        ranks[order[index:stop]] = 0.5 * (index + stop - 1) + 1.0
        index = stop
    return ranks


def spearman(a, b):
    """Spearman rank correlation. NaN when either series is constant (its rank
    variance is zero and the coefficient is undefined) -- returned rather than
    raised, because a constant router series is a real outcome on a small case
    and the caller should report it, not crash."""
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denominator)


def route_segment_demand(rg, table, polylines):
    """Per-segment crossing counts from routed geometry. `polylines` is an
    (N,2,2) array of axis-aligned segments in evaluator coordinates, as
    `route_eval.online_openroad.load_observation` produces. Collinear overlaps
    are collapsed by `topology.segment_union` first, the same convention
    `topology.union_metrics` uses for the crossing count, so a doubled-back
    wire is not charged twice."""
    from ioplace.region_segments import edge_segment_ids
    from ioplace.route_eval.topology import segment_union
    demand = np.zeros(table.num_segments, dtype=np.int64)
    geometry = np.asarray(polylines, dtype=np.float64).reshape(-1, 2, 2)
    if geometry.size == 0:
        return demand
    for a, b in segment_union(geometry):
        ids = edge_segment_ids(rg, table, float(a[0]), float(a[1]),
                               float(b[0]), float(b[1]))
        if len(ids):
            np.add.at(demand, np.asarray(ids, dtype=np.int64), 1)
    return demand


def surrogate_demand(config_json, regions_json, capacity_npz, node_x, node_y,
                     *, tau, m_pairs=4, m_seg=2, tau_b_cells=2.0, k=None):
    """CapTerm.demand at the given positions. Rebuilds the netlist and the IO
    CSR from the same PlaceDB the placement came from, so the active-net index
    space matches."""
    import torch

    from ioplace.capacity.extract import load_capacity
    from ioplace.dreamplace_env import setup_dreamplace
    from ioplace.evaluator_gpu import GpuEvalContext
    from ioplace.netlist import netlist_from_placedb
    from ioplace.ops.cap_term import CapTerm
    from ioplace.ops.io_term import IoTerm, build_net_node_csr
    from ioplace.ops.soft_assign import rect_table
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments, select_candidates
    from ioplace.regions import RegionSet

    setup_dreamplace()
    import Params
    import PlaceDB
    params = Params.Params()
    params.load(config_json)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    nl = netlist_from_placedb(placedb)
    region_set = RegionSet.from_json(regions_json)
    region_set.validate()
    rg = RegionGrid(region_set)
    table = enumerate_segments(rg)
    capacity = load_capacity(capacity_npz, table=table)["capacity"]

    rects, r2k = rect_table(region_set)
    csr = build_net_node_csr(nl, int(params.ignore_net_degree))
    io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k,
                     K=region_set.k, num_movable=nl.num_movable,
                     num_physical=nl.num_physical, num_nodes=placedb.num_nodes,
                     device="cuda")
    cap_term = CapTerm(io_term, table.box, capacity,
                       tau_b=tau_b_cells * float(rg.cell_w))

    ctx = GpuEvalContext(nl, rg, device="cuda", segments=table,
                         segment_capacity=capacity)
    res = ctx.evaluate(node_x, node_y, capacity_candidates=True)
    raw = select_candidates(res.cand_net, res.cand_u, res.cand_v, res.cand_seg,
                            res.cand_count, table, m_pairs=m_pairs,
                            m_seg=m_seg)
    mapping = np.full(nl.num_nets, -1, dtype=np.int64)
    mapping[csr.net_ids] = np.arange(len(csr.net_ids), dtype=np.int64)
    cap_term.set_candidates(raw.remap_nets(mapping))

    pos = torch.zeros(2 * placedb.num_nodes, dtype=torch.float64, device="cuda")
    pos[:nl.num_physical] = torch.as_tensor(node_x[:nl.num_physical],
                                            dtype=torch.float64, device="cuda")
    pos[placedb.num_nodes:placedb.num_nodes + nl.num_physical] = \
        torch.as_tensor(node_y[:nl.num_physical], dtype=torch.float64,
                        device="cuda")
    demand = cap_term.demand(pos, tau).cpu().numpy()
    diag = cap_term.diagnostics(pos, tau)
    return rg, table, capacity, demand, res, diag


def build_parser():
    parser = argparse.ArgumentParser(
        description="Spearman(GP surrogate D_s, routed per-segment crossings)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--placement", required=True,
                        help="placement.npz with node_x/node_y in evaluator units")
    parser.add_argument("--route-dir", required=True,
                        help="an existing online_openroad output directory "
                             "(receipt.json + segments.txt), i.e. the sec 8 "
                             "final GRT for this arm")
    parser.add_argument("--coord", required=True,
                        help="json with shift_factor/scale_factor")
    parser.add_argument("--netmap", required=True,
                        help="json mapping net index -> DEF net name")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--tau-b-cells", type=float, default=2.)
    parser.add_argument("--m-pairs", type=int, default=4)
    parser.add_argument("--m-seg", type=int, default=2)
    parser.add_argument("--out", required=True)
    return parser


def main(argv=None):
    from ioplace.route_eval.online_openroad import load_observation
    args = build_parser().parse_args(argv)
    placement = np.load(args.placement)
    node_x, node_y = placement["node_x"], placement["node_y"]

    rg, table, capacity, surrogate, res, diag = surrogate_demand(
        args.config, args.regions, args.capacity, node_x, node_y,
        tau=args.tau, m_pairs=args.m_pairs, m_seg=args.m_seg,
        tau_b_cells=args.tau_b_cells)

    observation = load_observation(args.route_dir, args.coord, args.regions,
                                   args.netmap)
    routed = np.zeros(table.num_segments, dtype=np.int64)
    for net, keys in observation["net_keys"].items():
        pass  # net_keys are ResourceGrid edge keys; the geometry is below
    # load_observation already unioned each net's per-layer geometry into
    # metric["segments"]; recover them per net from net_wirelength's source by
    # re-walking the decoded polylines it stored in net_keys' companion.
    polylines = observation.get("net_polylines")
    if polylines is None:
        raise SystemExit(
            "this route directory predates net_polylines; re-run "
            "load_observation from a build that returns per-net geometry, or "
            "decode segments.txt directly with route_eval/common_grt.py")
    for net, geometry in polylines.items():
        routed += route_segment_demand(rg, table, geometry)

    evaluator = np.asarray(res.segment_demand, dtype=np.int64)
    routable = capacity > 0
    record = {
        "num_segments": int(table.num_segments),
        "num_routable_segments": int(routable.sum()),
        "spearman_surrogate_router": spearman(surrogate[routable],
                                              routed[routable]),
        "spearman_evaluator_router": spearman(evaluator[routable],
                                              routed[routable]),
        "spearman_surrogate_evaluator": spearman(surrogate[routable],
                                                 evaluator[routable]),
        "surrogate_total": float(surrogate.sum()),
        "evaluator_total": int(evaluator.sum()),
        "router_total": int(routed.sum()),
        "num_over_capacity_evaluator": int((evaluator > capacity).sum()),
        "num_over_capacity_router": int((routed > capacity).sum()),
        "max_util_router": float((routed[routable] / capacity[routable]).max())
        if routable.any() else 0.0,
        "frac_singleton_groups": diag["frac_singleton_groups"],
        "cand_dropped": int(res.cand_dropped),
        "router_provenance": observation["provenance"],
        "placement_sha256": observation["placement_sha256"],
    }
    record["decision"] = (
        "pair-level fallback (sec 10 open question (i): below 0.5)"
        if not (record["spearman_surrogate_router"] >= 0.5)
        else "keep per-segment demand")
    with open(args.out, "w") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Two things the implementer must resolve while writing this, and must not paper over:

1. `load_observation` (`route_eval/online_openroad.py:100-159`) currently returns `net_keys` (ResourceGrid edge keys) but not the decoded polylines. Either add a `net_polylines` key there — a one-line `net_polylines[net] = np.asarray(segments)` next to the existing `net_keys[net] = ...`, which is additive and breaks no caller — or decode `segments.txt` directly with `route_eval/common_grt.py:61-172`. Pick one, do it in this task's commit, and delete the dead `for net, keys in ...: pass` loop above.
2. `--tau` is the soft temperature at which the surrogate is evaluated. Use the value the run's own `result.json` trajectory reports at its last iteration, not the default — record which value was used in the output JSON.

- [ ] **Step 4: Write the result-document template**

Create `docs/results/2026-09-19-p-d-capacity-rank-correlation.md`:

```markdown
# P-D: does the differentiable per-segment demand predict the router's?

Date: <fill on run>. Status: template — fill from `rank_correlation.json`.
Spec: `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §9
("done" for D) and §10 open question (i).
Plan: `docs/superpowers/plans/2026-09-19-v2-p-d-capacity.md` Task 10.

## Decision rule, pre-registered

§10 (i): *"Does `D_s` rank-correlate with routed per-segment crossings? —
Spearman on group from one GRT; below 0.5, fall back to pair-level
`D_ab`/`C_ab`."* This document states `spearman_surrogate_router` and applies
that rule. It is the deciding experiment, not a sanity check.

## Setup

| Field | Value |
|---|---|
| Case | |
| Arm | |
| `regions.json` sha256 | |
| `capacity.npz` sha256 / `capacity_source` | |
| `segments_sha256` | |
| OpenROAD receipt sha256 | |
| GRT policy | `set_routing_layers metal2-metal10`, `-congestion_iterations`, `-allow_congestion` |
| τ used for the surrogate | |

## Results

| Quantity | Value |
|---|---|
| segments / routable segments | |
| `spearman_surrogate_router` | |
| `spearman_evaluator_router` | |
| `spearman_surrogate_evaluator` | |
| surrogate total / evaluator total / router total | |
| `num_over_capacity` (evaluator / router) | |
| `max_util_router` | |
| `frac_singleton_groups` | |
| `cand_dropped` (u == v legs, interpretation D-1) | |

## Reading

Three numbers, three different failure modes:

- **`spearman_surrogate_router` < 0.5** → apply §10 (i): fall back to
  pair-level `D_ab`/`C_ab`. Note that `Σ_{s∈(a,b)} D_s` already equals the
  pair-level demand by construction (the α softmax sums to 1 per group), so
  the fallback is a reduction of the same term, not a rewrite.
- **`spearman_evaluator_router` also low** → the problem is the *proxy*, not
  the surrogate: MST L-routes are not predicting GR topology, and no amount
  of work on `q·α` will fix it.
- **`spearman_surrogate_evaluator` high but both router numbers low** → the
  differentiable model faithfully reproduces the evaluator, and the evaluator
  is the thing that is wrong.

`frac_singleton_groups` bounds how much of this term can survive the freeze
(plan limitation D-4: a one-element α softmax has zero gradient). Near 1.0
means spec §3's "capacity on after the freeze" buys nothing on this case and
must be reported as such.

## Caveat carried from §10 risk 5

`capacity` is OpenDB's uint8-clamped `getCapacity` (tile measured 7,871,705
against a native 7,872,067, `docs/results/2026-09-15-ggr-trial.md:39-40`), and
"one crossing = one track" ignores multi-wire nets and vias. Utilisation
numbers here are therefore accurate to a few tenths of a percent at best, and
`num_over_capacity` near the knee should not be over-read.
```

- [ ] **Step 5: Run the tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_cap_rank_correlation.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Run the experiment and fill the document**

```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" src/scripts/run_cap_rank_correlation.py \
    --config <case config> --regions <regions.json> --capacity <capacity.npz> \
    --placement <placement.npz> --route-dir <sec 8 GRT output dir> \
    --coord <coord.json> --netmap <netmap.json> --tau <final tau> \
    --out docs/results/2026-09-19-p-d-capacity-rank-correlation.json
```

Fill the date line and every empty table cell in the markdown from that JSON, then write the one-sentence decision the rule produces. A document with empty cells is not a completed task.

- [ ] **Step 7: Commit**

```bash
git add src/scripts/run_cap_rank_correlation.py tests/test_cap_rank_correlation.py \
        docs/results/2026-09-19-p-d-capacity-rank-correlation.md \
        docs/results/2026-09-19-p-d-capacity-rank-correlation.json \
        src/ioplace/route_eval/online_openroad.py
git commit -m "$(cat <<'MSG'
feat(capacity): surrogate-vs-router rank correlation, D's done criterion

Report per-segment utilisation from the GP surrogate, the evaluator's exact
MST proxy and one real GRT over the same segment ids, and state the two
Spearman coefficients (v2 design sec 9). sec 10 open question (i)'s rule --
below 0.5, fall back to pair-level D_ab/C_ab -- is applied explicitly in the
result document, alongside frac_singleton_groups and the uint8 capacity
caveat from sec 10 risk 5.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

## Self-Review

Run against the spec with fresh eyes after writing the plan. Issues found were fixed inline; what follows is the record, not a to-do list.

### 1. Spec coverage

| Spec requirement | Task |
|---|---|
| §0 "IO capacity": per adjacent-pair boundary **segment** | 1 |
| §0: feed-through nets consume capacity on **entry and exit** segments | 4 (`test_feed_through_charges_both_the_entry_and_the_exit_segment`), interpretations D-1/D-2 |
| §0: capacity from a one-time pre-GP OpenROAD extraction | 3 |
| §0: fallback `ρ·ℓ` from tech LEF | 3 (`test_lef_fallback_is_rho_times_length_in_microns`, real NanGate45 numbers) |
| §0/§5: penalty `L_cap = Σ_s (2[d]_+³ + [d]_+²)`, `d=(D−C)/C` | 4, verbatim in Global Constraints |
| §0/§5: activated at overflow ≤ 0.30 | 8 (`activate_overflow=0.30`), 9 |
| §0/§3/§5: stays on after the freeze | 9 |
| §0/§5: evaluator hard check with per-segment utilisation | 6, 7 |
| §5: enumeration on `RegionGrid.grid` 512² with run-length encoding | 1 |
| §5: segment table (orientation, line index, lo/hi, pair, physical length) | 1 |
| §5: unit-edge→segment-id raster (2×512×511 int32 ≈ 2 MB) | 1 (`edge_seg_v` (512,511) + `edge_seg_h` (511,512) = 523,264 int32) |
| §5: per-row/column CSR lists of boundary positions with segment ids | 1, 2 |
| §5: `ell[a,b]` is the sum of a pair's segment lengths, still the pair aggregate | 1 (`test_segment_lengths_sum_to_the_pair_aggregate_ell`), 6 (`test_segment_demand_sums_to_the_pair_demand_per_pair`) |
| §5: reuse `dump_online_route.py:90-114` / `online_openroad.py:44-159` on the INPUT DEF | 3 (`run_extraction`) |
| §5: minimal congestion iterations + `-allow_congestion` | 3 (`congestion_iterations=5`, `allow_congestion=True`, with the measured justification) |
| §5: keep only H/V GCell capacities, discard `usage` | 3 (`gcell_capacity` never reads `usage`) |
| §5: `C_seg = Σ_rows overlap/row_h · hcap` for a vertical segment, symmetric for horizontal | 3 (`test_gcell_capacity_is_the_overlap_weighted_row_sum`, hand-checked 40/80/17/19) |
| §5: `capacity.npz` with `capacity_source` / receipt SHA-256 / semantics string | 3 |
| §5 unit rule: GCell grid touched once offline; scalar per segment id; zero-capacity keeps a finite penalty; no epsilon | 3, 4 (interpretation D-3), verbatim in Global Constraints |
| §5: `D_s = Σ_e w_e·q·q·α_{e,s}` | 4, 5 |
| §5: `α` softmax over the pair's candidate segments by L1 distance from the soft pin centroid | 4 (`segment_softmax`, `l1_point_box_dist`, `_centroids`) |
| §5: `τ_b = 2·cell_w` | 8 (`--cap-tau-b-cells` default 2.0, multiplied by `rg.cell_w`) |
| §5: candidates refreshed from the evaluator every `home_period`, caps `m_pairs=4`, `m_seg=2` | 2, 8 Step 5f, 9 |
| §5: exact `q_a·q_b` gradient, cofactors cached per candidate, no `(E,K)` tensor | 5 |
| §5: rejected `softplus((D−C)/C)` | Global Constraints (verbatim), 4 |
| §5: new evaluator fields `segment_demand`/`segment_capacity`/`segment_util`/`num_over_capacity`/`max_util`/`p99_util` beside `boundary_*` | 6, 7, 8 |
| §5: `Ph`/`Pv` prefix sums as a free assertion | 2 (`test_leg_lookup_length_equals_the_prefix_sum_crossing_count`), 6 (per leg), 7 (global) |
| §4: `register("cap", term, curvature=<max_s pen''>, activate_overflow=0.30)` | 8, 9; `cap_curvature` in 4 with the recorded reasoning |
| §4: term protocol `value(pos, ctx)` | 5 (`CapNormTerm`) |
| §9: `tests/test_region_segments.py` — grid K=4 → 4 segments, L-shape, 1-bin notch, raster round-trip, `Σ seg_len == ell` | 1, 2 |
| §9: `tests/test_capacity_term.py` — penalty exactly 0 below capacity, C¹ at the knee by finite difference, gradient vs autograd on a 200-cell toy, exact `q_aq_b` product gradient | 4, 5 |
| §9 parity: bit-exact ref vs GPU across batch sizes / chunk budgets, on grid and one multi-rect geometry | 7 |
| §9: a `slow` regression pinning per-segment demand for one frozen placement | 7 |
| §9: end-to-end small case (GCD) | 8 |
| §9 "done" for D: rank correlation stated | 10 |
| §10 risk 5: uint8 clamp, "one crossing = one track", `.cap` is not a source | 3 (metadata + prose), 10 (result-document caveat) |
| §10 open question (i): Spearman, fall back below 0.5 | 10 |
| §1 artefact table: `capacity.npz` produced by the extractor, read by GP + evaluator | 3, 8, 9 |
| P-B `extra_terms=()` hook | 9 (appended to, not bypassed) |
| P-F schema conventions | 8, with the explicit two-branch version rule in Global Constraints |

**No gaps found.** Two items the spec mentions that this plan deliberately does *not* own: the pair-level `D_ab`/`C_ab` fallback (§10 (i) only mandates it *if* the measurement says so — Task 10 states the rule and the number; implementing the fallback is a follow-up conditioned on that result), and `route_eval/online_openroad.py`'s `net_polylines` addition, which Task 10 Step 3 calls out explicitly as work to do in that commit rather than leaving implicit.

### 2. Placeholder scan

Searched for `TBD`, `TODO`, `implement later`, `fill in details`, `Similar to Task N`, `add appropriate error handling`, `add validation`, `handle edge cases`, `Write tests for the above`, `FIXME`, `XXX`. Zero hits remain. Three near-misses were fixed:

- Task 1's `test_a_512_lattice_stays_cheap` originally carried a guessed segment count with a "replace it if the first run disagrees" note. Every expected value in Tasks 1, 2, 3 and 4 was instead **executed against a working prototype** of the plan's own implementation code before the plan was finalised (4×4 grid → 4 segments; L → 2; plug → 5 with `(0,1)` split on line 1; notch → 5 with three unit segments; 512² → 24 in 0.03 s; GCell fixture → 40/80/17/19; NanGate45 ρ → 12.589285714285714 / 14.280015037593985; `_CapFn` vs `CapTermRef` agreeing to 1e-13 across three chunk sizes and four capacities including zero). The note is gone and the numbers are pinned.
- Task 8 Step 5g hedged "use the same variable name that call uses" and offered two ways to get `frac_singleton_groups`. Both are now exact: the variable is `res` (`run_placement_io.py:784`) and the fraction is carried in `cb_state` from the callback, because no `pos` tensor survives past `placer(...)`.
- Task 10's result template used the literal token `TBD` for its date. Replaced with `<fill on run>`, with an explicit step that an unfilled document is not a completed task.

Two code blocks intentionally carry a "clean this up" instruction rather than being written twice: `save_capacity`'s `__import__("ioplace.region_segments", ...)` (Task 3 Step 4) and `run_fence_gp`'s `IoTerm(...) if False else IoTerm(...)` (Task 9 Step 3). Both say exactly what the final form must be. They exist to keep the full keyword list visible at the point of use; they are instructions, not gaps.

### 3. Type consistency

Checked every name that crosses a task boundary against its definition:

- `SegmentTable` attribute names (`orient`/`line`/`lo`/`hi`/`pair_a`/`pair_b`/`length_units`/`length`/`box`/`edge_seg_v`/`edge_seg_h`/`row_ptr`/`row_col`/`row_seg`/`col_ptr`/`col_row`/`col_seg`/`k`/`lattice`/`die`) and methods (`num_segments`, `grid_shape()`, `pair_key()`) are defined in Task 1 and used identically in Tasks 2, 3, 4, 5, 6, 7, 8, 10.
- `Candidates` fields (`net`/`u`/`v`/`seg`/`group`/`count`/`n_groups`) and methods (`remap_nets`, `is_empty`) are defined in Task 2 and consumed by `_CapBase.set_candidates` (Task 4) and both drivers (Tasks 8, 9) with the same names. `set_candidates` reads `cand.n_groups`, which `remap_nets` recomputes — checked.
- `CAPACITY_SCALARS` is defined once, in Task 1, and `segment_utilisation` asserts its own dict matches it. Tasks 6, 7, 8, 9 all key off that one tuple; `_pack_capacity_metrics` iterates it rather than restating the names.
- `CapTerm`/`CapTermRef` share `_CapBase`, so `set_candidates`, `w_cand`, `_centroids`, `_alpha`, `num_segments`, `seg_box`, `C`, `deg`, `tau_b`, `curvature_dref`, `last_demand`, `last_d` have one definition. `forward(pos, tau, lambda_cap)` has the same signature on both — the Task 5 equivalence test depends on that and would fail loudly otherwise.
- `CapNormTerm.value(pos, ctx)` matches `norm.TermNormalizer.probe`'s call `term.value(p, ctx)` (`norm.py:425`) and the `IoNormTerm`/`FtNormTerm` precedent (`ops/norm_terms.py`).
- `EvalResult`'s new fields are added in Task 6 and read by name in Tasks 7 (`_assert_capacity_fields_bit_exact`), 8 (`save_evaluation`, `_pack_capacity_metrics`) and 10. `cand_dropped` is an `int`, not an array — used consistently.
- `evaluate(..., segments=, segment_capacity=, capacity_candidates=)` on the reference and `GpuEvalContext(..., segments=, segment_capacity=)` + `evaluate(..., capacity_candidates=)` on the GPU: the asymmetry is deliberate (the GPU context is built once and reused across iterations, so the static tables belong in `__init__`) and `evaluate_gpu`'s wrapper reconciles the two for the parity tests.
- `load_capacity_for_grid(path, rg) -> (table, capacity, metadata)` is defined in Task 8 and reused verbatim by Task 9.
- `save_capacity(path, table, capacity, *, source, receipt_sha256=None, extra=None)` — Task 3's definition matches every call site in Tasks 3, 8 and 9.
- `cap_curvature(d_ref)` returns a float; `CapTerm.curvature` is a property returning it; `TermNormalizer.register(name, term, curvature, ...)` takes it positionally third, matching `norm.py:266`.

### 4. Known limitations recorded rather than hidden

Four, all with a test that pins the behaviour so nobody "fixes" them by accident:

1. **D-1** — the spec's literal `q_{e,a(s)}·q_{e,b(s)}` makes feed-through demand identically zero. Resolved by using the terminal pair.
2. **D-2** — a single softmax over a feed-through's entry and exit segments would halve each. Resolved by grouping on the boundary pair.
3. **D-3** — `d_s = (D_s−C_s)/C_s` is undefined at `C_s = 0` and the unit rule bans an epsilon. Resolved by `d_s := D_s`.
4. **D-4** — a singleton α-group has zero gradient, so this term's post-freeze value is conditional on the evaluator supplying real alternatives. Measured, reported as `frac_singleton_groups`, and quoted in Task 10's result document. **This is the one that could invalidate spec §3's "capacity on after the freeze"**, and it is the first thing to look at if Task 9's fence phase shows no effect.

### 5. Risk to flag to the scheduler before execution

Task 9 has the largest specification surface per line of any task here: it introduces a `TermNormalizer` into a phase that P-B built without one, and it does so against a file this plan's author could not read (P-B has not landed). Its step text is written from P-B's plan document, not from the landed code. The executor must diff P-B's final `run_fence_gp` against the excerpt quoted in Task 9's rationale **before** editing, and escalate rather than improvise if the callback structure has changed.
