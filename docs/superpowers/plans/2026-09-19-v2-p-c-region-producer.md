# P-C Region Producer (GrandPlan Replication) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v2 region producer — a standalone driver that runs a flat DREAMPlace GP+LG with a GrandPlan grouping loss, extracts rectilinear partition shapes from the resulting density maps, refines them by simulated annealing, and emits `regions.json` + `seed.npz` + `membership.npz` + `producer.json` for the main flow.

**Architecture:** One new package `src/ioplace/producer/` with five pure, separately testable modules (hull geometry + rasterised anchor tables, the grouping objective term, bin-map extraction, SA refinement, rectification) plus one new driver `src/ioplace/drivers/run_region_producer.py` that sequences them. The file contract itself, `src/ioplace/artifacts.py`, is **not** created here: it is owned by P-B Task 1 and consumed by this plan (see the reconciliation note under File Structure). Everything talks to DREAMPlace only through the two extension points that already exist (`dp_hook.attach_terms` and `placer.iteration_callback`) — no new DREAMPlace patch.

**Tech Stack:** Python 3.12, numpy 1.26.4, scipy 1.17.1 (`scipy.spatial.ConvexHull` = Qhull/quickhull, `scipy.ndimage` morphology/labelling/EDT), torch 2.8.0+cu128 (CUDA 12.8, H100 NVL), DREAMPlace 4.3.1 at `$DREAMPLACE_ROOT`, mtkahypar 1.6.2, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` (binding: §0 decisions, §1 data contracts, §2 producer design, §8 arm (e), §9 tests, §10 risks 1 and 6). Equations come from `docs/research/2026-09-18-grandplan-digest.md` §2, §3.1, §4.1–4.2, §5 (Eq.1–8, Algorithm 1).

## Global Constraints

- **Run protocol.** From the repo root: `source src/scripts/env.sh`, then `export CUDA_VISIBLE_DEVICES=3`, then `"$IOPLACE_PYTHON" -m pytest`. `env.sh` already defaults `IOPLACE_MTKAHYPAR_THREADS=1`. Use `-m "not slow"` while iterating; run the full suite before declaring a task done. This is a shared host — check `nvidia-smi` and pick an idle device before GPU work.
- **No new DREAMPlace patch.** `m2-extra-obj-terms.patch` already adds extra terms after the fence branch of `PlaceObj.obj_fn`, and `iteration-callback.patch` already supplies the per-iteration hook. Everything in this plan goes through `dp_hook.attach_terms(params, [...])` and `placer.iteration_callback`. Do not edit anything under `$DREAMPLACE_ROOT`.
- **Coordinate contract: native post-read PlaceDB units.** Every emitted artefact (`regions.json`, `seed.npz`) is in the die box captured right after `placedb.read(params)` and **before** `placedb.initialize(params)`, because `PlaceDB.scale()` inside `initialize()` rescales `node_x`, `regions` and `flat_region_boxes` together (`PlaceDB.py:151-196`, called from `initialize` which sets `params.shift_factor = (xl, yl)` and `params.scale_factor = 1/site_width`). Conversion back from the scaled system is `x_native = x_scaled / scale_factor + shift_factor[0]`. **Inside** the GP (grouping term, anchor tables) everything is in the scaled system — the tables are built from the post-`initialize()` die. This is the same contract as P-B's: the main flow reads `seed.npz`/`regions.json` in native units and writes them into `placedb.node_x`/`node_y` after `read()` and before `initialize()` (P-B Tasks 2 and 4), which is where `PlaceDB.scale()` converts them; `fence_phase.build_fence_placedb` additionally rejects a `regions.json` whose die is not the native post-read die. Checked 2026-09-19: the two plans agree, no change was needed on either side.
- **Artefact schema, verbatim from spec §1.** `regions.json` = `RegionSet` (die, lattice, per-region rect list) via `ioplace.regions.RegionSet.to_json`/`from_json`. `seed.npz` = `node_x`, `node_y` over `num_physical`, native units, `die`, `shift_factor`, `scale_factor`, `placedb_sha256`. `membership.npz` = `part` int32 per movable node, `{source, k, seed, epsilon}`. `producer.json` = the producer's own run record (field list `artifacts.PRODUCER_FIELDS`). **`src/ioplace/artifacts.py` is owned by P-B Task 1** (`docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md`), which lands first and implements every reader/writer with exactly these field names; this plan consumes them and defines none of its own. The canonical names are `placedb_identity_sha256`, `save_positions`/`load_positions` (with `kind="seed"`), `save_membership`/`load_membership` and `save_producer_json`/`load_producer_json` — see the rename table in Task 1.
- **`rect_max = 8`** rectangles per region, hard (spec §2 and §10 risk 1 — `ops/soft_assign.py:19-31` builds `(N, r)` temporaries with `r` = rects per k-chunk, so the budget is a memory contract, not cosmetics).
- **Lattice 512** for the emitted `RegionSet`. Extraction runs at `2048²` fine bins, majority-votes to `--extract-bins` ∈ {64, 32}, and 512 % 64 == 512 % 32 == 0, so every rect edge lands exactly on the lattice and `RegionSet.validate()` passes.
- **Fixed producer knobs (spec §2, do not re-derive):** Algorithm 1 `m=16`, `q=0.90`, `α=0.25`, `K_dir=64`; area cap `A_max = EA_k` with centroid shrink by bisection to `1e-3` relative area; macro pseudo points at mean std-cell pitch, `≤64/macro`; hull rebuild period `T_hull = 50` with anchors frozen in between; `α_pull = α_push = 1`; SA `θ=0.05`, `C_max=2`, `ρ_target=0.8`, `β=(1.0, 0.3, 0.5, 0.2)`, min-max normalisation over the first 200 samples, `T_0` = mean `|ΔE|` of 200 probe moves, geometric cooling `0.92`, 50 moves/level, 150 levels, stop after 3 levels with no accept, moves = area-balancing + corner-filling windows of 2–9 bins (equal probability while any area violation exists, corner-filling only afterwards), reject any move that fragments a region.
- **Determinism.** Every stochastic step takes an explicit seed and uses `numpy.random.default_rng(seed)`. `np.random.seed(params.random_seed)` must be called immediately before `NonLinearPlace(...)` is constructed (BasicPlace draws centre-noise and filler init from numpy's *global* RNG — see `run_placement._place`).
- **Commit messages** end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **Test tiers.** Pure-CPU modules get fast unit tests with hand-built inputs (no marker). Tests that need a CUDA device get `@pytest.mark.gpu` (still selected by `-m "not slow"`). Tests that run a real DREAMPlace placement get `@pytest.mark.slow`. The smallest real case the existing fast/slow tests use is `$DREAMPLACE_ROOT/install/test/simple.json` (8 movable nodes, 10 physical, 8 nets — `tests/test_driver.py:18-26`, `tests/test_dp_hook.py:167`); use it for the slow end-to-end tests.

---

## File Structure

**Reconciliation note (2026-09-19).** P-B and P-C were written in parallel and
both defined `src/ioplace/artifacts.py`. **P-B Task 1 is the single owner; this
plan consumes it.** Renamed here: `placedb_fingerprint` →
`placedb_identity_sha256`, `save_seed`/`load_seed` →
`save_positions`/`load_positions` (`kind="seed"`, and `load_positions` returns a
`Positions` dataclass, not a dict), positional `save_membership`/`load_membership`
→ P-B's keyword-only pair returning a `Membership` dataclass, and
`save_producer_json` moves to P-B Task 1 with a `PRODUCER_FIELDS` contract and a
new `load_producer_json`. Task 1 below is now a verification step and
`tests/test_artifacts.py` belongs to P-B. Two further seams: the
`IOPLACE_ENABLE_GR_IN_LOOP` gate is entirely P-H's and P-B's business (nothing
in this plan touches it), and **Task 11 is the single owner of the dead
`/nashome/NVL4` benchmark-config repair** — P-B stays on GCD and creates nothing
under `benchmarks/`. The coordinate contract in Task 10 rule 1 was checked
against P-B's warm-start write and needed no change.

**New files**

| File | Responsibility |
|---|---|
| `src/ioplace/producer/__init__.py` | Empty package marker. |
| `src/ioplace/producer/hull.py` | Algorithm-1 candidate reduction, quickhull via `scipy.spatial.ConvexHull`, area cap by centroid bisection, macro pseudo points, and the rasterised `512²` anchor tables (Eq.1's two anchor fields). Pure geometry — no DREAMPlace, no netlist. |
| `src/ioplace/producer/grouping_term.py` | `GroupingTerm` (Eq.1/Eq.2 quadratic springs to frozen, table-read anchors; a `torch.nn.Module` callable as a `dp_hook.attach_terms` term) and `GroupingWeight` (Eq.3 gradient-norm ratio with §4.3's ramp). |
| `src/ioplace/producer/extract.py` | Density maps → per-bin argmax → majority downsample → morphological open/close → largest connected component → non-empty guard → whitespace to nearest. Pure numpy/scipy on a label grid. |
| `src/ioplace/producer/sa.py` | Eq.4–8 energies, min-max normalisation, the two move types, fragmentation rejection, and the annealing schedule. Pure numpy/scipy. |
| `src/ioplace/producer/rectify.py` | Label grid → maximal-horizontal-strip rectangles → `rect_max=8` enforcement by smallest-notch filling → `RegionSet` on the 512 lattice + `validate()`. Pure numpy/scipy. |
| `src/ioplace/producer/membership.py` | The membership prior: `mtkahypar` (via `partition/mtkahypar_runner.partition_netlist`) or `hierarchy` (RTL name prefixes). Returns one `int32` label per movable node. |
| `src/ioplace/drivers/run_region_producer.py` | The CLI and the only place that touches DREAMPlace: read → prior → flat GP (grouping attached, tables rebuilt every `T_hull` from the iteration callback) → flat LG → extract → SA → rectify → write the four artefacts. |
| `benchmarks/ispd25/h100/mempool_tile_wrap.json` | Host-local DREAMPlace config for the acceptance run (the checked-in `benchmarks/ispd25/mempool_tile_wrap.json` still points at the retired `/nashome/NVL4/...` paths). |

**New tests**

`tests/test_producer_hull.py`, `tests/test_producer_anchor_tables.py`, `tests/test_producer_grouping_term.py`, `tests/test_producer_extract.py`, `tests/test_region_producer.py` (SA, per spec §9's named file), `tests/test_producer_rectify.py`, `tests/test_producer_membership.py`, `tests/test_run_region_producer.py`.

**Files read but never modified:** `src/ioplace/artifacts.py` (P-B Task 1 — the file contract; imported by Tasks 10 and 11, never edited), `src/ioplace/regions.py`, `src/ioplace/region_grid.py`, `src/ioplace/region_graph.py`, `src/ioplace/dp_hook.py`, `src/ioplace/netlist.py`, `src/ioplace/paths.py`, `src/ioplace/profile.py`, `src/ioplace/norm.py` (P-H; `ema_update` and `grandplan_weight` are imported by Task 4), `src/ioplace/partition/mtkahypar_runner.py`, `src/ioplace/drivers/run_placement.py`.

**Boundary rule:** `hull.py`/`extract.py`/`sa.py`/`rectify.py` take arrays and return arrays. Only `run_region_producer.py` knows what a `PlaceDB` is. That is what makes every geometric step testable in milliseconds with a hand-built input.

---

### Task 1: Verify the artefact contract (`artifacts.py`, owned by P-B)

`src/ioplace/artifacts.py` is owned by **P-B Task 1**
(`docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md`), which lands first and
already implements every name this plan needs, under the spec §1 field names,
alongside the `freeze.json`/`result.json` I/O only the main flow uses. An earlier
draft of this plan defined a second, incompatible copy of those names; that
duplication is now merged away. **This task creates no file.** It verifies the
contract once, before eight tasks build on it.

Merged names — use the right-hand column everywhere in Tasks 2–11:

| Old P-C name | Canonical name (P-B Task 1) |
|---|---|
| `placedb_fingerprint(placedb)` | `placedb_identity_sha256(placedb)` — the digest also covers `node_size_x`/`node_size_y` and `pin2node_map`/`pin2net_map`, so it must be taken after `read()` and **before** `initialize()` (node sizes are multiplied by `scale_factor` inside it, `PlaceDB.py:160-161`). Task 10 already hashes inside its `read` phase. |
| `save_seed(path, x, y, die, shift_factor, scale_factor, placedb_sha256)` | `save_positions(path, x, y, *, die, shift_factor, scale_factor, placedb_sha256, kind="seed")` — keyword-only, raises `ValueError` (not `AssertionError`) on a bad write |
| `load_seed(path) -> dict` | `load_positions(path, *, expect_num_physical=None, expect_sha256=None) -> Positions` — a **dataclass**: `seed.node_x`, not `seed["node_x"]` |
| `save_membership(path, part, source, k, seed, epsilon)` | `save_membership(path, part, *, source, k, seed=0, epsilon=0.0)` — keyword-only |
| `load_membership(path) -> dict` | `load_membership(path, *, expect_num_movable=None, expect_k=None, require_nonempty=False) -> Membership` — a dataclass: `mem.part`, `mem.k`, `mem.source` |
| `save_producer_json(path, payload)` | same name and same stamping behaviour, defined in P-B Task 1; it now validates `artifacts.PRODUCER_FIELDS` (the 48 keys Task 10's payload sets) and has a matching `load_producer_json(path) -> dict` |
| `SEED_SCHEMA_VERSION` | `POSITIONS_SCHEMA_VERSION` |
| `MEMBERSHIP_SCHEMA_VERSION`, `PRODUCER_SCHEMA_VERSION` | unchanged |

`tests/test_artifacts.py` belongs to P-B; the two fingerprint tests this plan
originally carried were moved into it.

**Files:**
- Create: nothing.
- Read: `src/ioplace/artifacts.py`.

**Interfaces:**
- Consumes: `src/ioplace/artifacts.py` from P-B Task 1.
- Produces: nothing. Tasks 10 and 11 import
  `artifacts.{placedb_identity_sha256, save_positions, load_positions,
  save_membership, load_membership, save_producer_json, load_producer_json}`.

- [ ] **Step 1: Check that P-B Task 1 has landed**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" - <<'PY'
from ioplace.artifacts import (MEMBERSHIP_SCHEMA_VERSION,
                               POSITIONS_SCHEMA_VERSION, PRODUCER_FIELDS,
                               PRODUCER_SCHEMA_VERSION, Membership, Positions,
                               load_membership, load_positions,
                               load_producer_json, placedb_identity_sha256,
                               save_membership, save_positions,
                               save_producer_json)

required = ("config", "out_dir", "placedb_sha256", "k", "membership_source",
            "membership_seed", "epsilon", "hierarchy_depth", "extract_bins",
            "fine_bins", "lattice", "rect_max", "t_hull", "probe_every",
            "alpha_pull", "alpha_push", "sa_seed", "die_native", "die_scaled",
            "shift_factor", "scale_factor", "num_movable", "num_physical",
            "num_nodes", "num_nets", "target_density", "n_hull_rebuilds",
            "wt_final", "lambda_group_final", "ratio_ema_final", "probes",
            "gp_iterations_run", "final_overflow", "hpwl_gp", "hpwl_lg", "sa",
            "rects_per_region", "rect_max_observed", "region_bins",
            "region_area", "region_cell_area", "region_utilisation",
            "area_balance", "runtime_s", "peak_mem_mb", "command", "hostname",
            "env")
missing = [name for name in required if name not in PRODUCER_FIELDS]
extra = [name for name in PRODUCER_FIELDS if name not in required]
assert not missing and not extra, (missing, extra)
print("artifacts contract ok", POSITIONS_SCHEMA_VERSION,
      MEMBERSHIP_SCHEMA_VERSION, PRODUCER_SCHEMA_VERSION, len(PRODUCER_FIELDS))
PY
```
Expected: `artifacts contract ok 1 1 1 48`.

If the import raises `ModuleNotFoundError` or `ImportError`, **stop and report**:
P-B Task 1 has not landed yet. Do **not** create `src/ioplace/artifacts.py`
here — a second definition of these names is exactly the conflict the
2026-09-19 reconciliation removed. If the `PRODUCER_FIELDS` assertion fires, the
two plans have drifted: reconcile the field list with Task 10's payload before
continuing, and fix it in P-B's plan, not by shadowing the module.

- [ ] **Step 2: Run P-B's artefact tests**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_artifacts.py -v
```
Expected: PASS — 11 passed.

- [ ] **Step 3: No commit**

This task produces no diff. Record
`git log -1 --format=%h -- src/ioplace/artifacts.py` and quote it in Task 10's
commit message, so the contract revision the producer was built against is
traceable.

---

### Task 2: Hull geometry (`producer/hull.py`, part 1)

GrandPlan Algorithm 1 (digest §2.1): sample `m` directions, keep a quantile band per direction, cap at `K_dir` per band, dedup, quickhull, then shrink toward the centroid until `Area ≤ A_max = EA_k`. Plus the macro enrichment of §3.1.1 ("interpolating additional points on a regular grid within the macro area using the average standard-cell dimensions as spacing"), capped at 64 points/macro by spec §2.

**Files:**
- Create: `src/ioplace/producer/__init__.py` (empty), `src/ioplace/producer/hull.py`
- Test: `tests/test_producer_hull.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 3 and 9):
  - Constants `DIRECTIONS_M = 16`, `QUANTILE_Q = 0.90`, `BAND_ALPHA = 0.25`, `K_DIR = 64`, `MACRO_MAX_POINTS = 64`
  - `reduce_candidates(pts: np.ndarray, m=16, q=0.90, alpha=0.25, k_dir=64) -> np.ndarray` — `(N,2)` float64 in, `(M,2)` float64 out
  - `convex_hull(pts: np.ndarray) -> np.ndarray` — `(V,2)` float64, counter-clockwise, always ≥3 vertices
  - `polygon_area(verts: np.ndarray) -> float`
  - `shrink_to_area(verts: np.ndarray, a_max: float, rel_tol=1e-3, max_iter=60) -> np.ndarray`
  - `macro_pseudo_points(x, y, w, h, pitch_x, pitch_y, max_per_macro=64) -> np.ndarray` — `(P,2)` float64
  - `build_hull(pts, a_max, **kw) -> np.ndarray` — the composed pipeline

- [ ] **Step 1: Write the failing test**

Create `tests/test_producer_hull.py`:

```python
import numpy as np
import pytest
from ioplace.producer import hull


def test_reduce_candidates_keeps_the_extreme_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, size=(5000, 2))
    pts = np.vstack([pts, [[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]]])
    out = hull.reduce_candidates(pts)
    for corner in ([-5., -5.], [5., -5.], [5., 5.], [-5., 5.]):
        assert (np.abs(out - corner).sum(axis=1) < 1e-12).any(), corner
    # Algorithm 1's own bound: 2 bands x m directions x K_dir, before dedup.
    assert len(out) <= 2 * hull.DIRECTIONS_M * hull.K_DIR


def test_reduce_candidates_is_deterministic_and_passes_small_sets_through():
    pts = np.array([[0., 0.], [1., 0.], [0., 1.]])
    assert np.array_equal(hull.reduce_candidates(pts),
                          hull.reduce_candidates(pts))
    assert len(hull.reduce_candidates(pts)) == 3


def test_convex_hull_of_a_square_cloud_is_the_square_ccw():
    rng = np.random.default_rng(1)
    inner = rng.uniform(0.1, 0.9, size=(200, 2))
    pts = np.vstack([inner, [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    v = hull.convex_hull(pts)
    assert len(v) == 4
    assert hull.polygon_area(v) == pytest.approx(1.0, rel=1e-12)
    # counter-clockwise => positive signed area
    sx, sy = v[:, 0], v[:, 1]
    signed = 0.5 * (np.dot(sx, np.roll(sy, -1)) - np.dot(sy, np.roll(sx, -1)))
    assert signed > 0.0


def test_convex_hull_degenerates_to_a_box_for_collinear_input():
    v = hull.convex_hull(np.array([[0., 0.], [1., 0.], [2., 0.]]))
    assert len(v) >= 3
    assert hull.polygon_area(v) > 0.0
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(2.0)


def test_shrink_to_area_matches_the_closed_form_within_the_tolerance():
    """Uniform scaling about a fixed point scales area exactly by s**2, so
    s* = sqrt(a_max/a) is the exact answer; the spec mandates bisection to
    1e-3 relative area, and this pins the bisection against that oracle."""
    v = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])
    a = hull.polygon_area(v)
    a_max = 0.25 * a
    out = hull.shrink_to_area(v, a_max)
    got = hull.polygon_area(out)
    assert got <= a_max + 1e-12
    assert got == pytest.approx(a_max, rel=1e-3)
    # centroid preserved
    assert out.mean(axis=0) == pytest.approx(v.mean(axis=0))


def test_shrink_to_area_is_a_noop_below_the_cap():
    v = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert np.array_equal(hull.shrink_to_area(v, 10.0), v)


def test_macro_pseudo_points_respect_the_cap_and_stay_inside():
    x = np.array([0.0, 100.0])
    y = np.array([0.0, 100.0])
    w = np.array([50.0, 3.0])
    h = np.array([50.0, 3.0])
    p = hull.macro_pseudo_points(x, y, w, h, pitch_x=1.0, pitch_y=1.0)
    # 50x50 at pitch 1 would be 2500 points; the cap forces 8x8.
    assert len(p) == 64 + 9
    first = p[:64]
    assert (first[:, 0] > 0.0).all() and (first[:, 0] < 50.0).all()
    assert (first[:, 1] > 0.0).all() and (first[:, 1] < 50.0).all()


def test_macro_pseudo_points_handles_no_macros():
    e = np.zeros(0)
    assert hull.macro_pseudo_points(e, e, e, e, 1.0, 1.0).shape == (0, 2)


def test_build_hull_applies_reduction_then_cap():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.0, 10.0, size=(20000, 2))
    v = hull.build_hull(pts, a_max=25.0)
    assert hull.polygon_area(v) <= 25.0 + 1e-9
    assert len(v) >= 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_hull.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.producer'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/producer/__init__.py` as an empty file. Create `src/ioplace/producer/hull.py`:

```python
"""Per-partition convex hulls for the GrandPlan grouping loss.

Algorithm 1 (directional-extrema candidate reduction) and the area cap of
docs/research/2026-09-18-grandplan-digest.md section 2.1, with the constants
fixed by the v2 spec section 2: m=16, q=0.90, alpha=0.25, K_dir=64, so quickhull
never sees more than ~1024 points per region even at 11M cells.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

DIRECTIONS_M = 16
QUANTILE_Q = 0.90
BAND_ALPHA = 0.25
K_DIR = 64
MACRO_MAX_POINTS = 64


def reduce_candidates(pts, m=DIRECTIONS_M, q=QUANTILE_Q, alpha=BAND_ALPHA,
                      k_dir=K_DIR):
    """Algorithm 1. For each of m equally spaced directions, keep the quantile
    band [t, t + alpha*(s_max - t)] with t = quantile(s, q), capped at the k_dir
    projections closest to t; repeat with the mirrored direction (which is the
    paper's t_lo = quantile(s, 1-q) branch); dedup.

    The output bound is 2*m*k_dir before dedup; antipodal direction pairs select
    largely the same points, which is why the spec quotes "<=1024 points/region"
    at these defaults. Fully deterministic: no RNG, and np.unique sorts.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= k_dir:
        return np.unique(pts, axis=0)
    keep = np.zeros(len(pts), dtype=bool)
    for j in range(m):
        th = j * (2.0 * np.pi / m)
        s = pts[:, 0] * np.cos(th) + pts[:, 1] * np.sin(th)
        for sign in (1.0, -1.0):
            ss = sign * s
            t = float(np.quantile(ss, q))
            hi = float(ss.max())
            band = np.nonzero((ss >= t) & (ss <= t + alpha * (hi - t)))[0]
            if len(band) > k_dir:
                order = np.argsort(ss[band] - t, kind="stable")
                band = band[order[:k_dir]]
            keep[band] = True
    return np.unique(pts[keep], axis=0)


def polygon_area(verts):
    """Shoelace absolute area."""
    v = np.asarray(verts, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ccw(verts):
    v = np.asarray(verts, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return v if signed > 0 else v[::-1].copy()


def convex_hull(pts):
    """Quickhull (scipy.spatial.ConvexHull is Qhull) over the deduplicated
    points, returned counter-clockwise. Fewer than three distinct points, or a
    collinear set (QhullError), falls back to the axis-aligned bounding box
    inflated to a non-degenerate rectangle -- the anchor tables downstream need
    a polygon with positive area and at least three vertices."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    uniq = np.unique(pts, axis=0)
    if len(uniq) >= 3:
        try:
            return _ccw(uniq[ConvexHull(uniq).vertices])
        except QhullError:
            pass
    if len(uniq) == 0:
        raise ValueError("convex_hull needs at least one point")
    lo, hi = uniq.min(axis=0), uniq.max(axis=0)
    eps = np.maximum((hi - lo) * 1e-6, 1e-9)
    hi = np.maximum(hi, lo + eps)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def shrink_to_area(verts, a_max, rel_tol=1e-3, max_iter=60):
    """Area cap: uniformly scale the hull vertices toward their centroid until
    Area(H) <= A_max (digest section 2.1). Spec section 2 mandates bisection to
    1e-3 relative area; the closed form s = sqrt(a_max/a) is the oracle the unit
    test checks this against."""
    v = np.asarray(verts, dtype=np.float64)
    a = polygon_area(v)
    if a <= a_max or a <= 0.0:
        return v
    c = v.mean(axis=0)
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        s = 0.5 * (lo + hi)
        a_s = polygon_area(c + s * (v - c))
        if a_s > a_max:
            hi = s
        else:
            lo = s
            if (a_max - a_s) <= rel_tol * a_max:
                break
    return c + lo * (v - c)


def macro_pseudo_points(x, y, w, h, pitch_x, pitch_y,
                        max_per_macro=MACRO_MAX_POINTS):
    """Macro enrichment (digest section 2.1): a regular grid of points inside
    each macro footprint at the mean standard-cell pitch, capped at
    max_per_macro per macro (spec section 2). x, y are lower-left corners.
    Points sit at sub-cell centres so they are strictly inside the footprint."""
    cap_side = max(1, int(np.floor(np.sqrt(max_per_macro))))
    out = []
    for xi, yi, wi, hi in zip(np.asarray(x, dtype=np.float64),
                              np.asarray(y, dtype=np.float64),
                              np.asarray(w, dtype=np.float64),
                              np.asarray(h, dtype=np.float64)):
        nx = int(min(max(1, round(wi / pitch_x)), cap_side))
        ny = int(min(max(1, round(hi / pitch_y)), cap_side))
        gx = xi + (np.arange(nx) + 0.5) * (wi / nx)
        gy = yi + (np.arange(ny) + 0.5) * (hi / ny)
        out.append(np.stack(np.meshgrid(gx, gy, indexing="ij"),
                            axis=-1).reshape(-1, 2))
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(out, axis=0)


def build_hull(pts, a_max, m=DIRECTIONS_M, q=QUANTILE_Q, alpha=BAND_ALPHA,
               k_dir=K_DIR):
    """Algorithm 1 -> quickhull -> area cap, the per-rebuild pipeline."""
    return shrink_to_area(
        convex_hull(reduce_candidates(pts, m=m, q=q, alpha=alpha, k_dir=k_dir)),
        a_max)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_hull.py -v
```
Expected: PASS — 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/__init__.py src/ioplace/producer/hull.py tests/test_producer_hull.py
git commit -m "feat(producer): GrandPlan Algorithm-1 hull reduction, area cap, macro points

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Rasterised anchor tables (`producer/hull.py`, part 2)

Spec §2's cost control: rasterise each hull onto the `512²` lattice and precompute, per lattice bin per region, the anchor offset, so the per-iteration loss is table lookups plus a spring per cell — `O(N)`, independent of hull complexity.

Two subtleties this task locks in, both load-bearing:

1. **Offsets, not absolute coordinates.** fp16 has ~11 mantissa bits, so an absolute die coordinate would quantise to ~0.1 % of the die. The table stores `anchor − bin_centre`; the consumer reconstructs `anchor = bin_centre(bin(x)) + offset`. The anchor is therefore frozen *and* quantised to the bin the cell is read from, which is exactly the "anchor held fixed within the step" semantics of digest §2.1 — the spring itself still uses the cell's continuous position, so the gradient is the exact Eq.2 gradient for that frozen anchor.
2. **Push collapses to a mean plus a count.** Eq.1's push is `Σ_{s≠k} ‖x − a_s‖²`. Since `Σ_s ‖x − a_s‖² = n‖x − ā‖² + (Σ_s‖a_s‖² − n‖ā‖²)` with `ā = (1/n)Σ_s a_s`, storing `(ā, n)` reproduces Eq.2's push **gradient exactly** and Eq.1's push **value up to a frozen constant**. Without this, the `Σ_{s≠k}` in the spec's "the paper's silent hotspot" would be back.

**Files:**
- Modify: `src/ioplace/producer/hull.py` (append)
- Test: `tests/test_producer_anchor_tables.py`

**Interfaces:**
- Consumes: `hull.convex_hull`, `hull.build_hull` (Task 2).
- Produces (used by Tasks 4 and 9):
  - `class AnchorTables` — dataclass with fields `lattice: int`, `die: tuple`, `pull_off: torch.Tensor (K, L*L, 2) float16`, `pull_on: torch.Tensor (K, L*L) bool`, `push_off: torch.Tensor (K, L*L, 2) float16`, `push_cnt: torch.Tensor (K, L*L) uint8`, and a property `k -> int`
  - `nearest_on_polygon_boundary(px, py, verts, chunk=16384) -> (proj (B,2) float64, inside (B,) bool)`
  - `anchor_tables(hulls, die, lattice, device="cuda", bin_chunk=16384) -> AnchorTables`

- [ ] **Step 1: Write the failing test**

Create `tests/test_producer_anchor_tables.py`:

```python
import numpy as np
import pytest
import torch
from ioplace.producer import hull

DIE = (0.0, 0.0, 8.0, 8.0)
SQ = np.array([[2., 2.], [6., 2.], [6., 6.], [2., 6.]])       # CCW inner square
BIG = np.array([[0., 0.], [8., 0.], [8., 8.], [0., 8.]])      # CCW whole die


def _bin(ix, iy, lattice=8):
    return iy * lattice + ix


def test_nearest_on_polygon_boundary_inside_and_outside():
    px = torch.tensor([0.5, 3.5, 3.5], dtype=torch.float64)
    py = torch.tensor([0.5, 2.5, 3.5], dtype=torch.float64)
    proj, inside = hull.nearest_on_polygon_boundary(px, py, SQ)
    assert list(inside.tolist()) == [False, True, True]
    assert proj[0].tolist() == pytest.approx([2.0, 2.0])   # corner of the square
    assert proj[1].tolist() == pytest.approx([3.5, 2.0])   # nearest edge is the bottom
    assert proj[2].tolist() == pytest.approx([3.5, 2.0])   # tie -> first edge in order


def test_pull_table_is_zero_inside_and_the_projection_offset_outside():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu")
    assert t.k == 2 and t.lattice == 8
    b_out = _bin(0, 0)          # bin centre (0.5, 0.5), outside SQ
    b_in = _bin(3, 3)           # bin centre (3.5, 3.5), inside SQ
    assert bool(t.pull_on[0, b_out]) is True
    assert bool(t.pull_on[0, b_in]) is False
    assert t.pull_off[0, b_out].double().tolist() == pytest.approx([1.5, 1.5], abs=1e-2)
    assert t.pull_off[0, b_in].double().tolist() == pytest.approx([0.0, 0.0])
    # every bin centre is inside BIG, so region 1 never pulls
    assert not bool(t.pull_on[1].any())


def test_push_count_and_mean_offset():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu")
    # region 0's foreign hulls = {BIG}, which contains every bin centre
    assert int(t.push_cnt[0].min()) == 1 and int(t.push_cnt[0].max()) == 1
    # region 1's foreign hulls = {SQ}: only the 4x4 central block is inside
    inside = t.push_cnt[1].reshape(8, 8).numpy()
    expect = np.zeros((8, 8), dtype=np.uint8)
    expect[2:6, 2:6] = 1
    assert np.array_equal(inside, expect)
    b = _bin(3, 2)              # centre (3.5, 2.5): nearest point of dSQ is (3.5, 2.0)
    assert t.push_off[1, b].double().tolist() == pytest.approx([0.0, -0.5], abs=1e-2)
    assert t.push_off[1, _bin(0, 0)].double().tolist() == pytest.approx([0.0, 0.0])


def test_reconstructed_anchor_matches_the_exact_projection_within_fp16():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu")
    ix, iy = np.meshgrid(np.arange(8), np.arange(8), indexing="xy")
    cx = (ix.ravel() + 0.5).astype(np.float64)
    cy = (iy.ravel() + 0.5).astype(np.float64)
    proj, inside = hull.nearest_on_polygon_boundary(
        torch.as_tensor(cx), torch.as_tensor(cy), SQ)
    got = np.stack([cx, cy], axis=1) + t.pull_off[0].double().numpy()
    out = ~inside.numpy()
    assert np.abs(got[out] - proj.numpy()[out]).max() < 1e-2


def test_table_shapes_dtypes_and_memory_budget():
    t = hull.anchor_tables([SQ, BIG], DIE, lattice=512, device="cpu")
    for name, tensor, dtype, shape in (
            ("pull_off", t.pull_off, torch.float16, (2, 512 * 512, 2)),
            ("pull_on", t.pull_on, torch.bool, (2, 512 * 512)),
            ("push_off", t.push_off, torch.float16, (2, 512 * 512, 2)),
            ("push_cnt", t.push_cnt, torch.uint8, (2, 512 * 512))):
        assert tensor.dtype is dtype, name
        assert tuple(tensor.shape) == shape, name
    # spec section 2's budget: K x 512^2 x 2 fp16 == 16 MiB at K=16.
    per_region = t.pull_off.element_size() * t.pull_off.nelement() // t.k
    assert per_region * 16 == 16 * 1024 * 1024


def test_anchor_tables_is_deterministic():
    a = hull.anchor_tables([SQ, BIG], DIE, lattice=16, device="cpu")
    b = hull.anchor_tables([SQ, BIG], DIE, lattice=16, device="cpu")
    assert torch.equal(a.pull_off, b.pull_off)
    assert torch.equal(a.push_off, b.push_off)
    assert torch.equal(a.push_cnt, b.push_cnt)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_anchor_tables.py -v
```
Expected: FAIL — `AttributeError: module 'ioplace.producer.hull' has no attribute 'nearest_on_polygon_boundary'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/producer/hull.py`:

```python
from dataclasses import dataclass

import torch


@dataclass
class AnchorTables:
    """Eq.1's two anchor fields, rasterised onto a lattice x lattice bin grid.

    pull_off (K, L*L, 2) fp16 : Pi_{H_k}(centre(b)) - centre(b), zero inside H_k.
    pull_on  (K, L*L)   bool  : centre(b) is OUTSIDE H_k (Eq.1's 1{x not in Omega_k}).
    push_off (K, L*L, 2) fp16 : mean over foreign hulls s != k containing b of
                                Pi_{dH_s}(centre(b)), minus centre(b).
    push_cnt (K, L*L)   uint8 : how many foreign hulls contain centre(b).

    Offsets rather than absolute coordinates because fp16 has ~11 mantissa bits:
    an absolute die coordinate would quantise to ~0.1% of the die, while the
    offset's fp16 error is negligible against a bin width. The consumer
    reconstructs anchor = centre(bin(x)) + offset.

    (mean, count) rather than one anchor per foreign hull because
    sum_s ||x - a_s||^2 = n*||x - abar||^2 + (sum_s ||a_s||^2 - n*||abar||^2):
    the bracket does not depend on x, so this reproduces Eq.2's push gradient
    exactly and Eq.1's push value up to a frozen constant, while keeping the
    per-cell cost O(1) instead of O(K) -- the hotspot spec section 2 calls out.
    """
    lattice: int
    die: tuple
    pull_off: torch.Tensor
    pull_on: torch.Tensor
    push_off: torch.Tensor
    push_cnt: torch.Tensor

    @property
    def k(self):
        return int(self.pull_off.shape[0])


def nearest_on_polygon_boundary(px, py, verts, chunk=16384):
    """Nearest point on a CONVEX polygon's boundary, plus an inside test.

    For a convex H this single routine yields both of Eq.1's anchors: the
    nearest boundary point is Pi_{H}(x) when x is outside H and Pi_{dH}(x) when
    x is inside it. Ties between equidistant edges break on the first edge in
    vertex order (torch.argmin's own rule) -- deterministic, which is all the
    frozen-anchor semantics require.

    px, py: (B,) float64 tensors. verts: (V,2) counter-clockwise.
    Returns (proj (B,2) float64, inside (B,) bool) on px's device.
    """
    dev = px.device
    v = torch.as_tensor(np.asarray(verts, dtype=np.float64), device=dev)
    a = v
    e = torch.roll(v, -1, dims=0) - v                       # (V,2) edge vectors
    ee = (e * e).sum(dim=1).clamp(min=1e-30)                # (V,)
    nrm = torch.stack([e[:, 1], -e[:, 0]], dim=1)           # outward normal (CCW)
    n = int(px.shape[0])
    proj = torch.empty((n, 2), dtype=torch.float64, device=dev)
    inside = torch.empty((n,), dtype=torch.bool, device=dev)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        qx, qy = px[lo:hi], py[lo:hi]
        wx = qx.unsqueeze(1) - a[:, 0].unsqueeze(0)         # (c,V)
        wy = qy.unsqueeze(1) - a[:, 1].unsqueeze(0)
        t = ((wx * e[:, 0] + wy * e[:, 1]) / ee).clamp(0.0, 1.0)
        cxv = a[:, 0] + t * e[:, 0]
        cyv = a[:, 1] + t * e[:, 1]
        d2 = (qx.unsqueeze(1) - cxv) ** 2 + (qy.unsqueeze(1) - cyv) ** 2
        j = d2.argmin(dim=1, keepdim=True)
        proj[lo:hi, 0] = cxv.gather(1, j).squeeze(1)
        proj[lo:hi, 1] = cyv.gather(1, j).squeeze(1)
        inside[lo:hi] = ((wx * nrm[:, 0] + wy * nrm[:, 1]) <= 0.0).all(dim=1)
    return proj, inside


def anchor_tables(hulls, die, lattice, device="cuda", bin_chunk=16384):
    """Rasterise Eq.1's anchors for every hull onto the lattice bin grid.

    hulls: list of (V,2) counter-clockwise vertex arrays, one per region, in
    region-id order. die: (xl, yl, xh, yh) in the SAME coordinate system the
    consumer's positions are in (inside GP that is the scaled post-initialize()
    system, not the native one).
    """
    xl, yl, xh, yh = (float(v) for v in die)
    L = int(lattice)
    cw, ch = (xh - xl) / L, (yh - yl) / L
    idx = torch.arange(L, dtype=torch.float64, device=device)
    gx = (xl + (idx + 0.5) * cw).repeat(L)                  # bin b = iy*L + ix
    gy = (yl + (idx + 0.5) * ch).repeat_interleave(L)
    centre = torch.stack([gx, gy], dim=1)
    K, B = len(hulls), L * L

    pull_off = torch.zeros((K, B, 2), dtype=torch.float16, device=device)
    pull_on = torch.zeros((K, B), dtype=torch.bool, device=device)
    own_off = torch.zeros((K, B, 2), dtype=torch.float32, device=device)
    own_in = torch.zeros((K, B), dtype=torch.bool, device=device)
    sum_off = torch.zeros((B, 2), dtype=torch.float64, device=device)
    sum_n = torch.zeros((B,), dtype=torch.int32, device=device)

    zero2 = torch.zeros((B, 2), dtype=torch.float64, device=device)
    for k, verts in enumerate(hulls):
        proj, inside = nearest_on_polygon_boundary(gx, gy, verts, chunk=bin_chunk)
        off = proj - centre
        pull_off[k] = torch.where(inside.unsqueeze(1), zero2, off).half()
        pull_on[k] = ~inside
        own_off[k] = off.float()
        own_in[k] = inside
        sum_off += torch.where(inside.unsqueeze(1), off, zero2)
        sum_n += inside.to(torch.int32)

    push_off = torch.zeros((K, B, 2), dtype=torch.float16, device=device)
    push_cnt = torch.zeros((K, B), dtype=torch.uint8, device=device)
    for k in range(K):
        n = (sum_n - own_in[k].to(torch.int32)).clamp(min=0)
        s = sum_off - torch.where(own_in[k].unsqueeze(1),
                                  own_off[k].double(), zero2)
        mean = s / n.clamp(min=1).unsqueeze(1).double()
        push_off[k] = torch.where((n > 0).unsqueeze(1), mean, zero2).half()
        push_cnt[k] = n.clamp(max=255).to(torch.uint8)

    return AnchorTables(lattice=L, die=(xl, yl, xh, yh), pull_off=pull_off,
                        pull_on=pull_on, push_off=push_off, push_cnt=push_cnt)
```

Move `from dataclasses import dataclass` and `import torch` up into the module's import block alongside the Task-2 imports; the code above shows them where they are first needed only for readability.

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_anchor_tables.py tests/test_producer_hull.py -v
```
Expected: PASS — 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/hull.py tests/test_producer_anchor_tables.py
git commit -m "feat(producer): rasterised 512^2 anchor tables for the grouping loss

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Grouping objective term (`producer/grouping_term.py`)

GrandPlan Eq.1/Eq.2 (digest §4.1): quadratic springs to a frozen anchor — pull when the cell is outside its own hull, push when it is inside a foreign hull, `α_pull = α_push = 1`. Anchors come from Task 3's tables and are rebuilt every `T_hull = 50` iterations by the driver; between rebuilds they are constants, which is what makes Eq.2 the exact gradient (digest §4.1: "Eq. (2) is the *exact* gradient for a fixed convex set (Moreau envelope)"). Weighting is Eq.3 with §4.3's ramp.

The soft-assign anchor is the **cell centre** per spec §7 (`x + 0.5·node_size_x`, `y + 0.5·node_size_y`), not the lower-left corner.

**Dependency on P-H.** `src/ioplace/norm.py` (spec §4, subproject P-H) already exists on this branch and exports the pure helpers `ema_update(prev, inst, ema=0.5)` and `grandplan_weight(iteration, it_activate, wt0, wt_step, ramp_period, wt_max)`. `GroupingWeight` **delegates** to those two rather than re-deriving Eq.3's ramp and EMA; it keeps only the `lam = wt · ratio_ema` composition locally, so P-C does not have to wait for P-H's `TermNormalizer` class. If `src/ioplace/norm.py` is missing when you start this task, stop and report it — do not reimplement the helpers.

**Files:**
- Create: `src/ioplace/producer/grouping_term.py`
- Test: `tests/test_producer_grouping_term.py`

**Interfaces:**
- Consumes: `hull.AnchorTables`, `hull.anchor_tables` (Task 3).
- Produces (used by Task 9):
  - `class GroupingTerm(torch.nn.Module)` with
    `__init__(part, node_size_x, node_size_y, num_movable, num_physical, num_nodes, alpha_pull=1.0, alpha_push=1.0, device="cuda")`,
    `set_tables(tables: AnchorTables) -> None`,
    `forward(pos: torch.Tensor, lam: float) -> torch.Tensor` (0-d),
    `grad_l1(pos: torch.Tensor) -> float`,
    attribute `n_rebuilds: int`
  - `class GroupingWeight` with
    `__init__(wt0=0.05, wt_step=0.05, wt_max=1.0, ramp_period=100, ema=0.5)`,
    `wt(iteration) -> float`, `update(iteration, g_wl_l1, g_group_l1) -> float`,
    attributes `lam: float`, `ratio_ema: float | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_producer_grouping_term.py`:

```python
import numpy as np
import pytest
import torch
from ioplace.producer import hull
from ioplace.producer.grouping_term import GroupingTerm, GroupingWeight

DIE = (0.0, 0.0, 8.0, 8.0)
SQ = np.array([[2., 2.], [6., 2.], [6., 6.], [2., 6.]])
BIG = np.array([[0., 0.], [8., 0.], [8., 8.], [0., 8.]])


def _one_cell_term(device="cpu"):
    """One zero-size movable cell in region 0, one dummy fixed node."""
    t = GroupingTerm(part=np.array([0]), node_size_x=np.zeros(1),
                     node_size_y=np.zeros(1), num_movable=1, num_physical=2,
                     num_nodes=2, device=device)
    t.set_tables(hull.anchor_tables([SQ, BIG], DIE, lattice=8, device=device))
    return t


def _pos(x, y, n_nodes=2, dtype=torch.float64):
    p = torch.zeros(2 * n_nodes, dtype=dtype)
    p[0], p[n_nodes] = x, y
    return p


def test_energy_matches_the_closed_form_of_eq1():
    """Cell centre (0.5, 0.5). Pull anchor = nearest point of SQ = (2, 2).
    Push: the one foreign hull BIG contains it; nearest point of dBIG is
    (0.5, 0) (tie with (0, 0.5) broken on the first edge). So
    E = 0.5*((0.5-2)^2 + (0.5-2)^2) + 0.5*1*((0.5-0.5)^2 + (0.5-0)^2) = 2.375."""
    t = _one_cell_term()
    e = float(t(_pos(0.5, 0.5), lam=1.0))
    assert e == pytest.approx(2.375, rel=1e-6)


def test_gradient_matches_the_closed_form_of_eq2():
    t = _one_cell_term()
    p = _pos(0.5, 0.5).requires_grad_(True)
    t(p, lam=1.0).backward()
    # dE/dx = (0.5-2) + 1*(0.5-0.5) = -1.5 ; dE/dy = (0.5-2) + (0.5-0) = -1.0
    assert float(p.grad[0]) == pytest.approx(-1.5, rel=1e-6)
    assert float(p.grad[2]) == pytest.approx(-1.0, rel=1e-6)


def test_cell_inside_its_own_hull_feels_no_pull():
    t = _one_cell_term()
    # centre (3.5, 3.5) is inside SQ; push from BIG is the only contribution
    e = float(t(_pos(3.5, 3.5), lam=1.0))
    proj, _ = hull.nearest_on_polygon_boundary(
        torch.tensor([3.5], dtype=torch.float64),
        torch.tensor([3.5], dtype=torch.float64), BIG)
    expect = 0.5 * float((3.5 - proj[0, 0]) ** 2 + (3.5 - proj[0, 1]) ** 2)
    assert e == pytest.approx(expect, abs=1e-3)


def test_lambda_scales_the_value_and_zero_short_circuits():
    t = _one_cell_term()
    assert float(t(_pos(0.5, 0.5), lam=2.0)) == pytest.approx(2.0 * 2.375, rel=1e-6)
    assert float(t(_pos(0.5, 0.5), lam=0.0)) == 0.0


def test_term_is_zero_before_the_first_rebuild():
    t = GroupingTerm(part=np.array([0]), node_size_x=np.zeros(1),
                     node_size_y=np.zeros(1), num_movable=1, num_physical=2,
                     num_nodes=2, device="cpu")
    assert t.n_rebuilds == 0
    assert float(t(_pos(0.5, 0.5), lam=1.0)) == 0.0


def test_cell_centre_anchoring_uses_node_size():
    """spec section 7: the soft anchor is the cell CENTRE, not the lower-left
    corner. A 1x1 cell placed at (0,0) has its centre at (0.5,0.5), so it must
    score exactly the closed form above."""
    t = GroupingTerm(part=np.array([0]), node_size_x=np.ones(1),
                     node_size_y=np.ones(1), num_movable=1, num_physical=2,
                     num_nodes=2, device="cpu")
    t.set_tables(hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu"))
    assert float(t(_pos(0.0, 0.0), lam=1.0)) == pytest.approx(2.375, rel=1e-6)


def test_anchors_stay_frozen_between_rebuilds():
    """Moving the cell inside the same bin must not move the anchor: the energy
    must follow the same quadratic, and n_rebuilds must not change."""
    t = _one_cell_term()
    e0 = float(t(_pos(0.5, 0.5), lam=1.0))
    e1 = float(t(_pos(0.6, 0.5), lam=1.0))
    d_pull = (0.6 - 2.0) ** 2 - (0.5 - 2.0) ** 2
    d_push = (0.6 - 0.5) ** 2 - 0.0
    assert e1 - e0 == pytest.approx(0.5 * (d_pull + d_push), rel=1e-6)
    assert t.n_rebuilds == 1


def test_fixed_and_filler_nodes_never_receive_gradient():
    t = GroupingTerm(part=np.array([0, 1]), node_size_x=np.zeros(2),
                     node_size_y=np.zeros(2), num_movable=2, num_physical=3,
                     num_nodes=5, device="cpu")
    t.set_tables(hull.anchor_tables([SQ, BIG], DIE, lattice=8, device="cpu"))
    p = torch.full((10,), 0.5, dtype=torch.float64).requires_grad_(True)
    t(p, lam=1.0).backward()
    g = p.grad
    assert torch.equal(g[2:5], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(g[7:10], torch.zeros(3, dtype=torch.float64))


@pytest.mark.gpu
def test_gradient_matches_central_differences_on_a_200_cell_toy():
    """Finite differences are valid here because the anchor is piecewise
    constant per lattice bin: every cell sits at a bin centre plus a jitter
    bounded well inside its bin, and h is five orders of magnitude smaller than
    the bin width, so no probe crosses a bin boundary."""
    dev = "cuda"
    die = (0.0, 0.0, 1024.0, 1024.0)
    lattice, n, h = 64, 200, 1e-3
    bin_w = 1024.0 / lattice                      # 16.0
    rng = np.random.default_rng(3)
    ibin = rng.integers(0, lattice, size=(n, 2))
    jitter = rng.uniform(-0.25 * bin_w, 0.25 * bin_w, size=(n, 2))
    xy = (ibin + 0.5) * bin_w + jitter
    part = rng.integers(0, 3, size=n)
    hulls = [hull.build_hull(xy[part == k] if (part == k).any() else xy,
                             a_max=1024.0 * 1024.0) for k in range(3)]
    t = GroupingTerm(part=part, node_size_x=np.zeros(n), node_size_y=np.zeros(n),
                     num_movable=n, num_physical=n, num_nodes=n, device=dev)
    t.set_tables(hull.anchor_tables(hulls, die, lattice=lattice, device=dev))
    p = torch.tensor(np.concatenate([xy[:, 0], xy[:, 1]]), dtype=torch.float64,
                     device=dev).requires_grad_(True)
    t(p, lam=1.0).backward()
    g = p.grad.detach().cpu().numpy()
    base = p.detach()
    for i in rng.choice(2 * n, size=20, replace=False):
        up, dn = base.clone(), base.clone()
        up[i] += h
        dn[i] -= h
        fd = (float(t(up, lam=1.0)) - float(t(dn, lam=1.0))) / (2 * h)
        assert fd == pytest.approx(g[int(i)], rel=1e-5, abs=1e-6)


def test_grouping_weight_ramp_and_ratio():
    w = GroupingWeight()
    assert w.wt(0) == pytest.approx(0.05)
    assert w.wt(99) == pytest.approx(0.05)
    assert w.wt(100) == pytest.approx(0.10)
    assert w.wt(1_000_000) == pytest.approx(1.0)
    # first update: ratio_ema == the instantaneous ratio
    lam = w.update(0, g_wl_l1=100.0, g_group_l1=10.0)
    assert w.ratio_ema == pytest.approx(10.0)
    assert lam == pytest.approx(0.05 * 10.0)
    # second: ema = 0.5*10 + 0.5*20 = 15
    lam = w.update(100, g_wl_l1=100.0, g_group_l1=5.0)
    assert w.ratio_ema == pytest.approx(15.0)
    assert lam == pytest.approx(0.10 * 15.0)


def test_grouping_weight_ignores_a_zero_group_gradient():
    w = GroupingWeight()
    w.update(0, 100.0, 10.0)
    before = w.lam
    assert w.update(0, 100.0, 0.0) == before
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_grouping_term.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.producer.grouping_term'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/producer/grouping_term.py`:

```python
"""GrandPlan Eq.1/Eq.2 grouping loss (digest section 4.1), evaluated through the
rasterised anchor tables of producer/hull.py.

Attached to DREAMPlace with dp_hook.attach_terms -- no new patch. The tables are
rebuilt every T_hull iterations by run_region_producer's iteration callback and
are FROZEN in between, which is the paper's own semantics (digest section 2.1:
the hull is "recomputed from the current cells each time and then held constant
inside the gradient"). With a frozen anchor Eq.2 is the exact gradient, so the
implementation is literally a quadratic spring and autograd gets it right.
"""
import numpy as np
import torch

from ioplace import norm


class GroupingTerm(torch.nn.Module):
    """Callable as `term(pos, lam)`; the driver wraps it in a one-argument
    closure for dp_hook.attach_terms.

    Positions are the cell CENTRE (spec section 7's --node-anchor center):
    x + 0.5*node_size_x, y + 0.5*node_size_y. Only the movable prefix is read,
    so fixed nodes and fillers structurally receive no gradient.
    """

    def __init__(self, part, node_size_x, node_size_y, num_movable,
                 num_physical, num_nodes, alpha_pull=1.0, alpha_push=1.0,
                 device="cuda"):
        super().__init__()
        m = int(num_movable)
        assert len(part) == m, f"part has {len(part)} entries, expected {m}"
        self.num_movable = m
        self.num_physical = int(num_physical)
        self.num_nodes = int(num_nodes)
        self.alpha_pull = float(alpha_pull)
        self.alpha_push = float(alpha_push)
        self.n_rebuilds = 0
        self.tables = None
        self.register_buffer("part", torch.as_tensor(
            np.asarray(part, dtype=np.int64), dtype=torch.int64, device=device))
        self.register_buffer("half_w", torch.as_tensor(
            0.5 * np.asarray(node_size_x, dtype=np.float64)[:m],
            dtype=torch.float64, device=device))
        self.register_buffer("half_h", torch.as_tensor(
            0.5 * np.asarray(node_size_y, dtype=np.float64)[:m],
            dtype=torch.float64, device=device))

    def set_tables(self, tables):
        """Install a freshly rasterised AnchorTables. Discretely changes the
        objective: the caller MUST bump its obj_version and call
        dp_hook.refresh_nesterov_secant before the next optimizer step."""
        assert tables.k > int(self.part.max()), \
            "anchor tables must cover every region id in `part`"
        self.tables = tables
        self.n_rebuilds += 1

    def _centres(self, pos):
        m = self.num_movable
        x = pos[:m].double() + self.half_w
        y = pos[self.num_nodes:self.num_nodes + m].double() + self.half_h
        return x, y

    def _lookup(self, x, y):
        """-> (bin_centre_x, bin_centre_y, pull_off, pull_on, push_off, push_cnt),
        all detached: the anchor is frozen inside the gradient."""
        t = self.tables
        xl, yl, xh, yh = t.die
        L = t.lattice
        cw, ch = (xh - xl) / L, (yh - yl) / L
        with torch.no_grad():
            ix = ((x - xl) / cw).floor().long().clamp_(0, L - 1)
            iy = ((y - yl) / ch).floor().long().clamp_(0, L - 1)
            b = iy * L + ix
            k = self.part
            cx = xl + (ix.double() + 0.5) * cw
            cy = yl + (iy.double() + 0.5) * ch
            return (cx, cy, t.pull_off[k, b].double(), t.pull_on[k, b],
                    t.push_off[k, b].double(), t.push_cnt[k, b].double())

    def forward(self, pos, lam):
        if self.tables is None or lam == 0.0:
            return pos.new_zeros(())
        x, y = self._centres(pos)
        cx, cy, pull_off, pull_on, push_off, push_n = self._lookup(x, y)
        dxp = x - (cx + pull_off[:, 0])
        dyp = y - (cy + pull_off[:, 1])
        pull = (0.5 * self.alpha_pull) * (
            pull_on.double() * (dxp * dxp + dyp * dyp)).sum()
        dxs = x - (cx + push_off[:, 0])
        dys = y - (cy + push_off[:, 1])
        push = (0.5 * self.alpha_push) * (
            push_n * (dxs * dxs + dys * dys)).sum()
        return (float(lam) * (pull + push)).to(pos.dtype)

    def grad_l1(self, pos):
        """||grad Group||_1 at lam=1, for GroupingWeight's Eq.3 ratio. Runs an
        independent fwd+bwd on a detached clone, exactly like
        IoTerm.io_grad_l1."""
        if self.tables is None:
            return 0.0
        p = pos.detach().clone().requires_grad_(True)
        self.forward(p, lam=1.0).backward()
        return float(p.grad.abs().sum())


class GroupingWeight:
    """GrandPlan Eq.3 with the schedule of digest section 4.1:
    lam = wt * ||grad WL||_p / ||grad Group||_p, wt starting at 0.05 and stepped
    +0.05 every 100 iterations up to 1.0.

    The ramp and the EMA are P-H's (src/ioplace/norm.py, spec section 4); this
    class only composes them and holds the state, so there is exactly one
    implementation of Eq.3's schedule in the repo. When P-H's TermNormalizer
    class lands, register the term with
    TermNormalizer.register("group", term, curvature=1) and delete this class --
    the driver reads only `.lam`, `.wt(...)` and `.update(...)`.

    norm_p = 1, matching spec section 4's "Norm order" decision (every
    calibrated constant in this repo was fitted under L1) and the existing
    io_grad_l1/g_wl_l1 probe in run_placement_io.py:472-481.
    """

    def __init__(self, wt0=0.05, wt_step=0.05, wt_max=1.0, ramp_period=100,
                 ema=0.5):
        self.wt0 = float(wt0)
        self.wt_step = float(wt_step)
        self.wt_max = float(wt_max)
        self.ramp_period = int(ramp_period)
        self.ema = float(ema)
        self.ratio_ema = None
        self.lam = 0.0

    def wt(self, iteration):
        # it_activate=0: the grouping term is on from iteration 0 (unlike IO/FT,
        # it has no overflow activation gate -- digest section 5 runs it for the
        # whole flat GP).
        return norm.grandplan_weight(int(iteration), 0, wt0=self.wt0,
                                     wt_step=self.wt_step,
                                     ramp_period=self.ramp_period,
                                     wt_max=self.wt_max)

    def update(self, iteration, g_wl_l1, g_group_l1):
        if not (g_group_l1 > 0.0) or not np.isfinite(g_wl_l1):
            return self.lam
        r = float(g_wl_l1) / float(g_group_l1)
        self.ratio_ema = norm.ema_update(self.ratio_ema, r, ema=self.ema)
        self.lam = self.wt(iteration) * self.ratio_ema
        return self.lam
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_grouping_term.py -v
```
Expected: PASS — 12 passed (the `gpu`-marked finite-difference test runs on CUDA device 3).

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/grouping_term.py tests/test_producer_grouping_term.py
git commit -m "feat(producer): GrandPlan Eq.1-3 grouping term with frozen table anchors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Bin-map extraction (`producer/extract.py`)

Digest §2.2 / spec §2: `2048²` per-partition density maps → per-bin argmax → majority vote to `--extract-bins` (64 default, 32 for arm (e)) → morphological opening then closing → largest connected component per partition → whitespace bins to the nearest partition.

Two decisions this task locks in:

- **Structuring element = the full `3×3` square** (`scipy.ndimage.generate_binary_structure(2, 2)`), not the 4-connected cross. Opening a solid rectangle with the cross erodes its four corners (erosion leaves the interior, dilation cannot put the corners back), manufacturing exactly the one-bin notches that `E_boundary` and the `rect_max=8` budget exist to avoid. Connected components and region adjacency stay 4-connected, matching `region_graph.region_graph`'s unit-edge adjacency (`region_graph.py:62-71`).
- **A non-empty guard** after the largest-CC step. Spec §10 risk 6: at K=16 on `64²` bins a partition can lose every bin, which would produce an empty `RegionSpec` and fail `RegionSet.validate()`. Each empty partition gets back the single bin where its own coarse density is highest among bins whose current owner still has ≥2 bins.

**Files:**
- Create: `src/ioplace/producer/extract.py`
- Test: `tests/test_producer_extract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 6, 7, 9):
  - `FINE_BINS = 2048`
  - `density_maps(node_x, node_y, node_w, node_h, part, k, die, bins=FINE_BINS) -> (k, bins, bins) float64`
  - `block_sum(dens, out_bins) -> (k, out_bins, out_bins) float64`
  - `argmax_labels(dens) -> (bins, bins) int16` (`-1` = no density)
  - `majority_downsample(fine, out_bins, k) -> (out_bins, out_bins) int16`
  - `morph_open_close(labels, k) -> (B, B) int16`
  - `largest_component(labels, k) -> (B, B) int16`
  - `ensure_nonempty(labels, coarse_dens, k) -> (B, B) int16`
  - `fill_whitespace(labels) -> (B, B) int16`
  - `extract(node_x, node_y, node_w, node_h, part, k, die, out_bins, fine_bins=FINE_BINS) -> (out_bins, out_bins) int16`

- [ ] **Step 1: Write the failing test**

Create `tests/test_producer_extract.py`:

```python
import numpy as np
import pytest
from ioplace.producer import extract

DIE = (0.0, 0.0, 16.0, 16.0)


def _two_l_shapes(n=16):
    """A 16x16 label grid: region 1 is the top-right quadrant, region 0 is the
    L-shaped remainder. `labels[y, x]`."""
    lab = np.zeros((n, n), dtype=np.int16)
    lab[n // 2:, n // 2:] = 1
    return lab


def _cells_from_labels(lab, die=DIE):
    """One unit-area cell at the centre of every bin, labelled by that bin."""
    n = lab.shape[0]
    xl, yl, xh, yh = die
    cw, ch = (xh - xl) / n, (yh - yl) / n
    iy, ix = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = xl + ix.ravel() * cw
    y = yl + iy.ravel() * ch
    w = np.full(x.size, cw)
    h = np.full(x.size, ch)
    return x, y, w, h, lab.ravel().astype(np.int32)


def test_density_maps_and_argmax_recover_the_input_labels():
    lab = _two_l_shapes()
    x, y, w, h, part = _cells_from_labels(lab)
    dens = extract.density_maps(x, y, w, h, part, 2, DIE, bins=16)
    assert dens.shape == (2, 16, 16)
    assert dens.sum() == pytest.approx(16.0 * 16.0)
    assert np.array_equal(extract.argmax_labels(dens), lab)


def test_argmax_marks_empty_bins():
    dens = np.zeros((2, 4, 4))
    dens[0, 0, 0] = 1.0
    lab = extract.argmax_labels(dens)
    assert lab[0, 0] == 0
    assert (lab[1:] == -1).all()


def test_majority_downsample_2x2_blocks():
    fine = np.array([[0, 0, 1, 1],
                     [0, 1, 1, 1],
                     [-1, -1, 0, 0],
                     [-1, 2, 0, 0]], dtype=np.int16)
    out = extract.majority_downsample(fine, 2, 3)
    # block (0,0): {0,0,0,1} -> 0 ; block (0,1): {1,1,1,1} -> 1
    # block (1,0): {-1,-1,-1,2} -> 2 (empty bins do not vote) ; block (1,1): 0
    assert out.tolist() == [[0, 1], [2, 0]]


def test_majority_downsample_marks_a_fully_empty_block():
    fine = -np.ones((4, 4), dtype=np.int16)
    fine[0, 0] = 1
    assert extract.majority_downsample(fine, 2, 2).tolist() == [[1, -1], [-1, -1]]


def test_opening_does_not_erode_a_solid_rectangle():
    """The 3x3 square structuring element is chosen precisely for this: the
    4-connected cross would strip all four corners of every rectangle."""
    lab = _two_l_shapes(8)
    assert np.array_equal(extract.morph_open_close(lab, 2), lab)


def test_opening_removes_an_isolated_speck():
    lab = _two_l_shapes(8)
    lab[1, 1] = 1                       # one stray bin of region 1 inside region 0
    out = extract.morph_open_close(lab, 2)
    assert out[1, 1] == -1              # opened away, now unclaimed whitespace


def test_morphology_never_deletes_a_whole_partition():
    """Opening removes an isolated speck, but a partition whose only bins were
    specks must not vanish outright -- it is restored from the pre-morphology
    map wherever the post-morphology map is unclaimed."""
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    lab[0, 0] = 2                       # region 2 is a single speck
    out = extract.morph_open_close(lab, 3)
    assert (out == 2).any()
    assert out[0, 0] == 2


def test_largest_component_drops_a_detached_island():
    lab = _two_l_shapes(16)
    lab[1:4, 1:4] = 1                   # a 3x3 island that survives opening
    out = extract.largest_component(lab, 2)
    assert (out[1:4, 1:4] == -1).all()
    assert (out[8:, 8:] == 1).all()


def test_ensure_nonempty_rescues_a_vanished_partition():
    lab = _two_l_shapes(8)              # region 2 has no bins at all
    coarse = np.zeros((3, 8, 8))
    coarse[2, 0, 5] = 9.0               # region 2's best bin, currently region 0's
    out = extract.ensure_nonempty(lab, coarse, 3)
    assert out[0, 5] == 2
    assert int((out == 2).sum()) == 1


def test_fill_whitespace_leaves_no_unlabelled_bin():
    lab = _two_l_shapes(8)
    lab[3:5, 3:5] = -1
    out = extract.fill_whitespace(lab)
    assert (out >= 0).all()
    assert out[0, 0] == 0 and out[7, 7] == 1


def test_fill_whitespace_rejects_a_fully_empty_map():
    with pytest.raises(ValueError):
        extract.fill_whitespace(-np.ones((4, 4), dtype=np.int16))


def test_extract_end_to_end_tiles_the_die_with_both_l_shapes():
    lab = _two_l_shapes(16)
    x, y, w, h, part = _cells_from_labels(lab)
    out = extract.extract(x, y, w, h, part, 2, DIE, out_bins=8, fine_bins=16)
    assert out.shape == (8, 8)
    assert (out >= 0).all()
    assert set(np.unique(out).tolist()) == {0, 1}
    assert (out[4:, 4:] == 1).all()
    assert (out[:4, :] == 0).all()


def test_extract_is_deterministic():
    lab = _two_l_shapes(16)
    x, y, w, h, part = _cells_from_labels(lab)
    a = extract.extract(x, y, w, h, part, 2, DIE, out_bins=8, fine_bins=16)
    b = extract.extract(x, y, w, h, part, 2, DIE, out_bins=8, fine_bins=16)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_extract.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.producer.extract'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/producer/extract.py`:

```python
"""Boundary extraction: legalized flat placement -> rectilinear bin map.

digest section 2.2 / spec section 2: per-partition density maps at 2048^2, per
bin argmax, majority vote down to 64^2 (32^2 for arm (e)), morphological opening
then closing, largest connected component per partition, whitespace bins to the
nearest partition. Pure numpy/scipy; the caller supplies coordinates in whatever
system it wants the output interpreted in.
"""
import numpy as np
from scipy import ndimage

FINE_BINS = 2048

# Full 3x3 square. The 4-connected cross erodes every rectangle's corners during
# opening (dilation cannot restore them), manufacturing exactly the one-bin
# notches E_boundary and the rect_max budget exist to remove.
_SE = ndimage.generate_binary_structure(2, 2)
# Connected components and adjacency stay 4-connected, matching
# region_graph.region_graph's unit-edge adjacency (region_graph.py:62-71).
_CC = ndimage.generate_binary_structure(2, 1)


def density_maps(node_x, node_y, node_w, node_h, part, k, die, bins=FINE_BINS):
    """(k, bins, bins) cell-area density, accumulated at the cell CENTRE (spec
    section 7's anchor decision). node_x/node_y are lower-left corners."""
    xl, yl, xh, yh = (float(v) for v in die)
    cw, ch = (xh - xl) / bins, (yh - yl) / bins
    cx = np.asarray(node_x, dtype=np.float64) + 0.5 * np.asarray(node_w, dtype=np.float64)
    cy = np.asarray(node_y, dtype=np.float64) + 0.5 * np.asarray(node_h, dtype=np.float64)
    ix = np.clip(((cx - xl) / cw).astype(np.int64), 0, bins - 1)
    iy = np.clip(((cy - yl) / ch).astype(np.int64), 0, bins - 1)
    p = np.asarray(part, dtype=np.int64)
    area = np.asarray(node_w, dtype=np.float64) * np.asarray(node_h, dtype=np.float64)
    flat = (p * bins + iy) * bins + ix
    return np.bincount(flat, weights=area,
                       minlength=k * bins * bins).reshape(k, bins, bins)


def block_sum(dens, out_bins):
    """Aggregate a (k, F, F) map into (k, out_bins, out_bins)."""
    k, f, _ = dens.shape
    assert f % out_bins == 0, f"{f} fine bins do not divide into {out_bins}"
    s = f // out_bins
    return dens.reshape(k, out_bins, s, out_bins, s).sum(axis=(2, 4))


def argmax_labels(dens):
    """Per-bin argmax partition; bins with no density become -1. Ties break on
    the lowest partition id (np.argmax's own rule)."""
    lab = dens.argmax(axis=0).astype(np.int16)
    lab[dens.sum(axis=0) <= 0.0] = -1
    return lab


def majority_downsample(fine, out_bins, k):
    """Majority vote of `fine` (F x F, -1 = empty) into out_bins x out_bins.
    Empty fine bins do not vote; a block with no votes becomes -1. Ties break on
    the lowest partition id."""
    f = fine.shape[0]
    assert f % out_bins == 0, f"{f} fine bins do not divide into {out_bins}"
    s = f // out_bins
    blocks = fine.reshape(out_bins, s, out_bins, s).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(out_bins * out_bins, s * s)
    counts = np.stack([(blocks == kk).sum(axis=1) for kk in range(k)], axis=1)
    out = np.full(out_bins * out_bins, -1, dtype=np.int16)
    has = counts.sum(axis=1) > 0
    out[has] = counts[has].argmax(axis=1).astype(np.int16)
    return out.reshape(out_bins, out_bins)


def morph_open_close(labels, k):
    """Per-partition binary opening then closing (digest section 2.2). Bins
    claimed by more than one partition afterwards go to the partition with the
    most 4-neighbours in the pre-morphology map (tie -> lowest id); bins claimed
    by none become -1."""
    masks = np.stack([
        ndimage.binary_closing(
            ndimage.binary_opening(labels == kk, structure=_SE), structure=_SE)
        for kk in range(k)])
    n_claim = masks.sum(axis=0)
    out = np.full(labels.shape, -1, dtype=np.int16)
    single = n_claim == 1
    if single.any():
        out[single] = masks[:, single].argmax(axis=0).astype(np.int16)
    multi = np.nonzero(n_claim > 1)
    if len(multi[0]):
        cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        nb = np.stack([ndimage.convolve((labels == kk).astype(np.int32), cross,
                                        mode="constant", cval=0)
                       for kk in range(k)])
        cand = np.where(masks[:, multi[0], multi[1]],
                        nb[:, multi[0], multi[1]], -1)
        out[multi] = cand.argmax(axis=0).astype(np.int16)
    # Opening is allowed to shave a partition, never to delete it: a partition
    # made entirely of specks would otherwise vanish here and leave
    # ensure_nonempty with no donor at all on small designs.
    for kk in range(k):
        had = labels == kk
        if had.any() and not (out == kk).any():
            out[had & (out < 0)] = np.int16(kk)
            if not (out == kk).any():
                out[np.unravel_index(int(np.argmax(had)), out.shape)] = np.int16(kk)
    return out


def largest_component(labels, k):
    """Keep only each partition's largest 4-connected component; the rest
    becomes -1. ndimage.label numbers components in raster order and np.argmax
    returns the lowest index on ties, so equal-sized components break on the one
    whose first bin comes first in raster order."""
    out = np.full(labels.shape, -1, dtype=np.int16)
    for kk in range(k):
        m = labels == kk
        if not m.any():
            continue
        cc, n = ndimage.label(m, structure=_CC)
        sizes = np.bincount(cc.ravel(), minlength=n + 1)
        sizes[0] = 0
        out[cc == int(np.argmax(sizes))] = kk
    return out


def ensure_nonempty(labels, coarse_dens, k):
    """spec section 10 risk 6: at K=16 on 64^2 bins the argmax + largest-CC
    pipeline can leave a partition with no bins at all, which would build an
    empty RegionSpec and fail RegionSet.validate(). Give every such partition
    the single bin where its own coarse density is highest among bins whose
    current owner still has at least 2 bins. Deterministic; moves at most one
    bin per empty partition."""
    out = labels.copy()
    for kk in range(k):
        if (out == kk).any():
            continue
        counts = np.bincount(out[out >= 0].ravel(), minlength=k)
        donor_ok = np.zeros(out.shape, dtype=bool)
        owned = out >= 0
        donor_ok[owned] = counts[out[owned]] >= 2
        if not donor_ok.any():
            raise ValueError(
                f"partition {kk} vanished and no donor bin is available; "
                "reduce --extract-bins or K")
        score = np.where(donor_ok, coarse_dens[kk], -np.inf)
        out[np.unravel_index(int(np.argmax(score)), out.shape)] = kk
    return out


def fill_whitespace(labels):
    """Assign every -1 bin to the nearest labelled bin (Euclidean distance
    transform; scipy's return_indices tie-break is deterministic)."""
    unl = labels < 0
    if not unl.any():
        return labels
    if unl.all():
        raise ValueError("every bin is unlabelled; extraction produced no regions")
    _, idx = ndimage.distance_transform_edt(unl, return_indices=True)
    return labels[idx[0], idx[1]].astype(np.int16)


def extract(node_x, node_y, node_w, node_h, part, k, die, out_bins,
            fine_bins=FINE_BINS):
    """The whole digest section 2.2 pipeline. Returns an (out_bins, out_bins)
    int16 label grid with every bin owned by exactly one partition, so the
    result tiles the die with zero whitespace by construction."""
    dens = density_maps(node_x, node_y, node_w, node_h, part, k, die,
                        bins=fine_bins)
    coarse_dens = block_sum(dens, out_bins)
    lab = majority_downsample(argmax_labels(dens), out_bins, k)
    lab = morph_open_close(lab, k)
    lab = largest_component(lab, k)
    lab = ensure_nonempty(lab, coarse_dens, k)
    lab = fill_whitespace(lab)
    assert (lab >= 0).all()
    return lab
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_extract.py -v
```
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/extract.py tests/test_producer_extract.py
git commit -m "feat(producer): density-argmax bin-map extraction with non-empty guard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: SA refinement (`producer/sa.py`)

Digest §4.2 Eq.4–8 and §5's move set, with spec §2's defaults for everything the paper leaves unspecified: `β=(1.0, 0.3, 0.5, 0.2)`, min-max normalisation over the first 200 samples, `T_0` = mean `|ΔE|` of those 200 probe moves, cooling `0.92`, 50 moves/level, 150 levels, stop after 3 idle levels.

Two things to be explicit about:

- **Corner definition on a bin grid.** Eq.6 needs `C_ij`, a corner count, which the paper never defines discretely. Here: at each interior lattice vertex, look at the four surrounding bins and the four incident unit edges; pair `(i,j)` has a corner at that vertex iff exactly two of those edges separate `i` from `j` **and** they are perpendicular (one horizontal, one vertical). A straight shared boundary scores 0 (and Eq.6 then charges `0.1·(0−2)² = 0.4`, which digest §4.2 flags as the paper's own intentional slack), an L-turn scores 1, a one-bin notch scores 4.
- **Corner-filling's trigger.** Digest §5 says "if a corner is detected in the window, all bins in the window are reassigned to the majority partition". The operational test here is that the window carries ≥2 distinct labels, i.e. it straddles a boundary; reassigning to the majority then removes the protrusion. This is a superset of a strict corner test (it also flattens straight boundaries, which is a no-op for the energy), and it is the only test that is well-defined for a 1×2 window. `E_boundary` does the actual corner accounting.

**Files:**
- Create: `src/ioplace/producer/sa.py`
- Test: `tests/test_region_producer.py` (spec §9 names this file for the SA tests)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 7 and 9):
  - `@dataclass SaConfig` with `beta=(1.0,0.3,0.5,0.2)`, `theta=0.05`, `c_max=2`, `rho_target=0.8`, `probe_moves=200`, `cooling=0.92`, `moves_per_level=50`, `levels=150`, `idle_levels_stop=3`, `corner_tries=20`, `seed=0`
  - `e_area(labels, k, ea, bin_area, theta=0.05) -> float`
  - `shared_edges(labels, k) -> (k,k) int64`
  - `corner_counts(labels, k) -> (k,k) int64`
  - `e_boundary(labels, k, c_max=2) -> float`
  - `e_compact(labels, k, rho_target=0.8) -> float`
  - `e_diff(labels, labels0) -> float`
  - `class SaState` with `raw()`, `total(raw)`, `calibrate(rng)`, `propose(rng)`, `try_move(mv)`, `commit()`, `rollback()`
  - `anneal(labels0, k, ea, bin_area, cfg=None) -> (labels int16, report dict)` (`cfg=None` means `SaConfig()`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_region_producer.py`:

```python
import numpy as np
import pytest
from scipy import ndimage
from ioplace.producer import sa


def _l_map():
    """4x4: region 1 is the 2x2 block at rows 0-1, cols 2-3; region 0 is the
    L-shaped remainder."""
    lab = np.zeros((4, 4), dtype=np.int16)
    lab[0:2, 2:4] = 1
    return lab


def _notch_map():
    lab = np.zeros((4, 4), dtype=np.int16)
    lab[1, 1] = 1
    return lab


def test_e_area_matches_eq5_closed_form():
    lab = np.zeros((10, 20), dtype=np.int16)
    lab.reshape(-1)[:80] = 0
    lab.reshape(-1)[80:] = 1            # counts 80 and 120, bin_area 1
    # d_0 = (0.95*100 - 80)/100 = 0.15 -> 50*0.15^3 + 25*0.15^2 = 0.73125
    # d_1 = (0.95*100 - 120)/100 < 0    -> 0
    got = sa.e_area(lab, 2, ea=np.array([100.0, 100.0]), bin_area=1.0)
    assert got == pytest.approx(0.73125, rel=1e-12)


def test_e_area_is_zero_above_the_slack_threshold():
    lab = np.zeros((10, 10), dtype=np.int16)
    lab[5:] = 1
    assert sa.e_area(lab, 2, ea=np.array([50.0, 50.0]), bin_area=1.0) == 0.0


def test_corner_counts_straight_boundary_is_zero():
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    assert int(sa.corner_counts(lab, 2)[0, 1]) == 0


def test_corner_counts_l_turn_is_one():
    assert int(sa.corner_counts(_l_map(), 2)[0, 1]) == 1


def test_corner_counts_single_bin_notch_is_four():
    assert int(sa.corner_counts(_notch_map(), 2)[0, 1]) == 4


def test_corner_counts_is_symmetric_with_a_zero_diagonal():
    c = sa.corner_counts(_l_map(), 2)
    assert np.array_equal(c, c.T)
    assert int(np.trace(c)) == 0


def test_e_boundary_matches_eq6_and_penalises_straight_boundaries_too():
    """digest section 4.2: Eq.6 is not one-sided, so C_ij = 0 costs 0.1*(0-2)^2."""
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    assert sa.e_boundary(lab, 2) == pytest.approx(0.4, rel=1e-12)
    assert sa.e_boundary(_l_map(), 2) == pytest.approx(0.1 * (1 - 2) ** 2, rel=1e-12)


def test_e_boundary_skips_non_adjacent_pairs():
    lab = np.zeros((6, 6), dtype=np.int16)
    lab[:, 2:4] = 1
    lab[:, 4:] = 2                       # 0-1 and 1-2 adjacent, 0-2 not
    assert sa.shared_edges(lab, 3)[0, 2] == 0
    assert sa.e_boundary(lab, 3) == pytest.approx(0.8, rel=1e-12)


def test_e_compact_matches_eq7_closed_form():
    # region 0: 12 bins in a 4x4 bbox -> rho 0.75 -> 5*((0.8-0.75)/0.8)^2
    # region 1: 4 bins in a 2x2 bbox  -> rho 1.0  -> 0
    assert sa.e_compact(_l_map(), 2) == pytest.approx(5.0 * (0.05 / 0.8) ** 2,
                                                      rel=1e-12)


def test_e_diff_is_the_changed_fraction():
    a = _l_map()
    b = a.copy()
    b[3, 0] = 1
    b[3, 1] = 1
    assert sa.e_diff(b, a) == pytest.approx(2.0 / 16.0, rel=1e-12)


def test_calibration_sets_minmax_and_a_positive_t0():
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 8:] = 1
    cfg = sa.SaConfig(probe_moves=50, seed=1)
    st = sa.SaState(lab, 2, ea=np.array([160.0, 96.0]), bin_area=1.0, cfg=cfg)
    lo, hi, t0 = st.calibrate(np.random.default_rng(cfg.seed))
    assert lo.shape == (4,) and hi.shape == (4,)
    assert (hi >= lo).all()
    assert t0 > 0.0
    assert np.array_equal(st.labels, lab), "calibration must leave the map intact"


def test_a_fragmenting_move_is_rejected():
    """Handing row 2 of region 0's 5x2 block to region 1 splits region 0 into
    two lobes, so try_move must refuse it and roll back."""
    lab = np.zeros((5, 5), dtype=np.int16)
    lab[:, 2:] = 1                       # region 0 is a solid 5x2 block, region 1 5x3
    st = sa.SaState(lab, 2, ea=np.array([10.0, 15.0]), bin_area=1.0,
                    cfg=sa.SaConfig())
    waist = np.array([2 * 5 + 0, 2 * 5 + 1], dtype=np.int64)
    assert st.try_move((waist, 1)) is None
    assert np.array_equal(st.labels, lab), "a rejected move must be rolled back"


def test_anneal_is_deterministic_under_a_fixed_seed():
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 8:] = 1
    ea = np.array([128.0, 128.0])
    cfg = sa.SaConfig(probe_moves=40, levels=10, moves_per_level=10, seed=5)
    a, ra = sa.anneal(lab, 2, ea, 1.0, cfg)
    b, rb = sa.anneal(lab, 2, ea, 1.0, cfg)
    assert np.array_equal(a, b)
    assert ra["e_raw_final"] == rb["e_raw_final"]


def test_anneal_keeps_every_region_non_empty_and_connected():
    rng = np.random.default_rng(7)
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:8, :8] = 0
    lab[:8, 8:] = 1
    lab[8:, :8] = 2
    lab[8:, 8:] = 3
    ea = np.array([80.0, 60.0, 60.0, 56.0])
    cfg = sa.SaConfig(probe_moves=60, levels=20, moves_per_level=20, seed=11)
    out, rep = sa.anneal(lab, 4, ea, 1.0, cfg)
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(4):
        m = out == k
        assert m.any(), f"region {k} vanished"
        assert ndimage.label(m, structure=st)[1] == 1, f"region {k} fragmented"
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3}
    assert rep["levels_run"] <= cfg.levels
    assert rep["t0"] > 0.0
    assert len(rep["e_raw_final"]) == 4 and len(rep["beta"]) == 4
    _ = rng     # the map above is fixed on purpose; no randomness in the input


def test_anneal_reduces_the_area_imbalance_it_is_given():
    """Region 1 starts far under its target area; area-balancing moves exist,
    so the final E_area must not be worse than the initial one."""
    lab = np.zeros((16, 16), dtype=np.int16)
    lab[:, 14:] = 1
    ea = np.array([128.0, 128.0])
    cfg = sa.SaConfig(probe_moves=60, levels=40, moves_per_level=30, seed=3)
    out, rep = sa.anneal(lab, 2, ea, 1.0, cfg)
    assert rep["e_raw_final"][0] <= rep["e_raw_initial"][0] + 1e-9
    assert int((out == 1).sum()) > int((lab == 1).sum())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_region_producer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.producer.sa'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/producer/sa.py`:

```python
"""Simulated-annealing refinement of the extracted bin map.

Energies are digest section 4.2's Eq.4-8 verbatim; the schedule and move set are
digest section 5 with spec section 2's defaults for every value the paper leaves
unspecified. Pure numpy/scipy, deterministic given `SaConfig.seed`.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_CC = ndimage.generate_binary_structure(2, 1)      # 4-connected, as region_graph
# Every axis-aligned window shape with 2..9 bins (digest section 5).
_WINDOWS = tuple((w, h) for w in range(1, 10) for h in range(1, 10)
                 if 2 <= w * h <= 9)


@dataclass
class SaConfig:
    beta: tuple = (1.0, 0.3, 0.5, 0.2)
    theta: float = 0.05
    c_max: int = 2
    rho_target: float = 0.8
    probe_moves: int = 200
    cooling: float = 0.92
    moves_per_level: int = 50
    levels: int = 150
    idle_levels_stop: int = 3
    corner_tries: int = 20
    seed: int = 0


# ------------------------------------------------------------------ energies

def e_area(labels, k, ea, bin_area, theta=0.05):
    """Eq.5: d_i = (A_min,i - A_c,i)/EA_i with A_min,i = (1-theta)*EA_i;
    E_area,i = 50[d_i]_+^3 + 25[d_i]_+^2. Only shortfall is penalised."""
    cnt = np.bincount(np.asarray(labels).ravel(), minlength=k)[:k].astype(np.float64)
    ea = np.asarray(ea, dtype=np.float64)
    d = ((1.0 - theta) * ea - cnt * float(bin_area)) / np.where(ea > 0.0, ea, 1.0)
    dp = np.maximum(d, 0.0)
    return float((50.0 * dp ** 3 + 25.0 * dp ** 2).sum())


def shared_edges(labels, k):
    """(k,k) count of unit lattice edges separating each pair -- the same
    quantity region_graph.region_graph calls `ell` (region_graph.py:62-71),
    computed here on a bin map instead of a RegionGrid."""
    a = np.asarray(labels, dtype=np.int64)
    ell = np.zeros((k, k), dtype=np.int64)
    for p, q in ((a[:, :-1], a[:, 1:]), (a[:-1, :], a[1:, :])):
        m = p != q
        np.add.at(ell, (p[m], q[m]), 1)
        np.add.at(ell, (q[m], p[m]), 1)
    return ell


def corner_counts(labels, k):
    """C_ij for Eq.6. At each interior lattice vertex the four surrounding bins
    define four incident unit edges (S, N, W, E). Pair (i,j) has a corner there
    iff exactly two of those edges separate i from j and they are perpendicular
    -- one of {S,N} and one of {W,E}. Straight boundary -> 0, L-turn -> 1,
    one-bin notch -> 4."""
    a = np.asarray(labels, dtype=np.int64)
    A, Bb, C, D = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]

    def key(p, q):
        return np.where(p == q, -1, np.minimum(p, q) * k + np.maximum(p, q))

    kS, kN, kW, kE = key(A, Bb), key(C, D), key(A, C), key(Bb, D)
    flat = np.zeros(k * k, dtype=np.int64)
    for p, q, o1, o2 in ((kS, kW, kN, kE), (kS, kE, kN, kW),
                         (kN, kW, kS, kE), (kN, kE, kS, kW)):
        m = (p >= 0) & (p == q) & (o1 != p) & (o2 != p)
        if m.any():
            np.add.at(flat, p[m], 1)
    upper = flat.reshape(k, k)
    return upper + upper.T


def e_boundary(labels, k, c_max=2):
    """Eq.6 over adjacent pairs only: sum 0.1*(C_ij - C_max)^2. Not one-sided
    (digest section 4.2), so a perfectly straight shared boundary still costs
    0.1*C_max^2."""
    ell = shared_edges(labels, k)
    cc = corner_counts(labels, k)
    iu = np.triu_indices(k, 1)
    adj = ell[iu] > 0
    return float((0.1 * (cc[iu][adj].astype(np.float64) - c_max) ** 2).sum())


def e_compact(labels, k, rho_target=0.8):
    """Eq.7: 5.0 * ((rho_target - rho_fill,i)/rho_target)_+^2 with
    rho_fill,i = A_c,i / A_bb,i on the bin grid."""
    lab = np.asarray(labels)
    total = 0.0
    for kk in range(k):
        m = lab == kk
        n = int(m.sum())
        if n == 0:
            continue
        ys, xs = np.nonzero(m)
        abb = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        d = max((rho_target - n / abb) / rho_target, 0.0)
        total += 5.0 * d * d
    return float(total)


def e_diff(labels, labels0):
    """Eq.8: D_diff / N_bins."""
    lab = np.asarray(labels)
    return float(np.count_nonzero(lab != np.asarray(labels0))) / float(lab.size)


# ------------------------------------------------------------------- SA state

def _halo(mask):
    """4-neighbourhood of a boolean mask (the mask itself excluded by caller)."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


class SaState:
    """Holds the map, the frozen min-max normalisation and one pending move.

    Move protocol: `mv = propose(rng)`; `d = try_move(mv)` applies it, rejects
    (returns None, already rolled back) if it fragments or empties a region,
    otherwise returns the normalised energy delta; then exactly one of
    `commit()` / `rollback()`.
    """

    def __init__(self, labels0, k, ea, bin_area, cfg):
        self.cfg = cfg
        self.k = int(k)
        self.ea = np.asarray(ea, dtype=np.float64)
        self.bin_area = float(bin_area)
        self.labels0 = np.asarray(labels0, dtype=np.int16).copy()
        self.labels = self.labels0.copy()
        self._flat = self.labels.reshape(-1)
        self.lo = np.zeros(4)
        self.hi = np.ones(4)
        self._pending = None
        self._raw = self.raw()

    def raw(self):
        c = self.cfg
        return np.array([
            e_area(self.labels, self.k, self.ea, self.bin_area, c.theta),
            e_boundary(self.labels, self.k, c.c_max),
            e_compact(self.labels, self.k, c.rho_target),
            e_diff(self.labels, self.labels0)], dtype=np.float64)

    def total(self, raw):
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        norm = (np.asarray(raw, dtype=np.float64) - self.lo) / span
        return float(np.dot(np.asarray(self.cfg.beta, dtype=np.float64), norm)), norm

    # -- move application ---------------------------------------------------
    def _apply(self, mv):
        flat, new = mv
        old = self._flat[flat].copy()
        self._flat[flat] = np.int16(new)
        self._pending = (flat, old, int(new))

    def rollback(self):
        flat, old, _ = self._pending
        self._flat[flat] = old
        self._pending = None

    def commit(self):
        self._raw = self.raw()
        self._pending = None

    def _breaks_a_region(self):
        flat, old, new = self._pending
        for kk in set(int(v) for v in np.unique(old)) | {new}:
            m = self.labels == kk
            if not m.any():
                return True
            if ndimage.label(m, structure=_CC)[1] != 1:
                return True
        return False

    def try_move(self, mv):
        if mv is None:
            return None
        self._apply(mv)
        if self._breaks_a_region():
            self.rollback()
            return None
        return self.total(self.raw())[0] - self.total(self._raw)[0]

    # -- move generation ----------------------------------------------------
    def _deficits(self):
        cnt = np.bincount(self._flat, minlength=self.k)[:self.k].astype(np.float64)
        a_min = (1.0 - self.cfg.theta) * self.ea
        need = np.maximum(np.ceil((a_min - cnt * self.bin_area) / self.bin_area), 0)
        excess = np.maximum(np.floor((cnt * self.bin_area - a_min) / self.bin_area), 0)
        return need.astype(np.int64), excess.astype(np.int64)

    def has_area_violation(self):
        return bool((self._deficits()[0] > 0).any())

    def _area_balancing(self, rng):
        """digest section 5: pick a deficit region, score every contiguous run
        of a surplus neighbour's boundary bins by
        min(#excess_bins, #bins_needed, run length), and take the best run's
        bins in raster order."""
        need, excess = self._deficits()
        deficit = np.nonzero(need > 0)[0]
        if not len(deficit):
            return None
        t = int(deficit[int(rng.integers(len(deficit)))])
        tmask = self.labels == t
        halo = _halo(tmask) & ~tmask
        if not halo.any():
            return None
        best = None
        for d in np.unique(self.labels[halo]):
            d = int(d)
            if excess[d] <= 0:
                continue
            cc, n = ndimage.label(halo & (self.labels == d), structure=_CC)
            ccf = cc.reshape(-1)
            for c in range(1, n + 1):
                run = np.flatnonzero(ccf == c)
                score = int(min(excess[d], need[t], len(run)))
                if score <= 0:
                    continue
                key = (-score, d, int(run[0]))
                if best is None or key < best[0]:
                    best = (key, run[:score])
        if best is None:
            return None
        return best[1], t

    def _corner_filling(self, rng):
        """digest section 5: a random 2..9-bin window on a boundary; every bin
        in it goes to the window's majority label. "On a boundary" is the
        operational corner test -- the only one defined for a 1x2 window; Eq.6
        does the actual corner accounting."""
        b = self.labels.shape[0]
        for _ in range(self.cfg.corner_tries):
            w, h = _WINDOWS[int(rng.integers(len(_WINDOWS)))]
            if w > b or h > b:
                continue
            x0 = int(rng.integers(0, b - w + 1))
            y0 = int(rng.integers(0, b - h + 1))
            win = self.labels[y0:y0 + h, x0:x0 + w]
            vals, cnts = np.unique(win, return_counts=True)
            if len(vals) < 2:
                continue
            maj = int(vals[int(np.argmax(cnts))])       # np.unique sorts -> low id wins
            idx = np.nonzero(win.reshape(-1) != maj)[0]
            ys, xs = np.divmod(idx, w)
            return ((y0 + ys) * b + (x0 + xs)).astype(np.int64), maj
        return None

    def propose(self, rng, force_both=False):
        """Equal probability between the two move types while any area
        violation exists; corner-filling only afterwards (digest section 5)."""
        if (force_both or self.has_area_violation()) and rng.random() < 0.5:
            mv = self._area_balancing(rng)
            if mv is not None:
                return mv
        return self._corner_filling(rng)

    # -- calibration --------------------------------------------------------
    def calibrate(self, rng):
        """Min-max normalise each term over the first `probe_moves` samples
        (digest section 4.2: "normalized into similar scale before we use the
        weights beta_j"), then T_0 = mean |Delta E| over the same probes (spec
        section 2). Leaves the map exactly as it found it."""
        base = self._raw.copy()
        samples = [base]
        probes = []
        for _ in range(self.cfg.probe_moves):
            mv = self.propose(rng, force_both=True)
            if mv is None:
                continue
            self._apply(mv)
            if self._breaks_a_region():
                self.rollback()
                continue
            r = self.raw()
            samples.append(r)
            probes.append(r)
            self.rollback()
        arr = np.stack(samples)
        self.lo, self.hi = arr.min(axis=0), arr.max(axis=0)
        base_tot = self.total(base)[0]
        deltas = [abs(self.total(r)[0] - base_tot) for r in probes]
        t0 = float(np.mean(deltas)) if deltas else 1.0
        return self.lo, self.hi, max(t0, 1e-9)


def anneal(labels0, k, ea, bin_area, cfg=None):
    """Run the full schedule. Returns (labels int16, report dict)."""
    cfg = cfg or SaConfig()
    st = SaState(labels0, k, ea, bin_area, cfg)
    rng = np.random.default_rng(cfg.seed)
    _, _, t0 = st.calibrate(rng)
    raw_initial = st.raw()
    st._raw = raw_initial.copy()
    temp = t0
    idle = 0
    accepted = proposed = 0
    level = 0
    for level in range(1, cfg.levels + 1):
        n_acc = 0
        for _ in range(cfg.moves_per_level):
            mv = st.propose(rng)
            if mv is None:
                continue
            proposed += 1
            d = st.try_move(mv)
            if d is None:
                continue
            if d <= 0.0 or rng.random() < np.exp(-d / max(temp, 1e-12)):
                st.commit()
                n_acc += 1
            else:
                st.rollback()
        accepted += n_acc
        idle = idle + 1 if n_acc == 0 else 0
        if idle >= cfg.idle_levels_stop:
            break
        temp *= cfg.cooling
    raw_final = st.raw()
    report = {
        "t0": float(t0), "levels_run": int(level),
        "moves_proposed": int(proposed), "moves_accepted": int(accepted),
        "beta": list(cfg.beta),
        "norm_lo": st.lo.tolist(), "norm_hi": st.hi.tolist(),
        "e_raw_initial": raw_initial.tolist(), "e_raw_final": raw_final.tolist(),
        "e_norm_initial": st.total(raw_initial)[1].tolist(),
        "e_norm_final": st.total(raw_final)[1].tolist(),
        "e_total_initial": st.total(raw_initial)[0],
        "e_total_final": st.total(raw_final)[0]}
    return st.labels, report
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_region_producer.py -v
```
Expected: PASS — 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/sa.py tests/test_region_producer.py
git commit -m "feat(producer): GrandPlan Eq.4-8 SA refinement of the extracted bin map

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Rectification and `rect_max` enforcement (`producer/rectify.py`)

Spec §2's hard requirement and §10 risk 1: "decompose each region into maximal horizontal strips, merge, and enforce `rect_max=8` by filling the smallest notches; re-run `validate()`". Risk 1 is a memory contract, not cosmetics — `ops/soft_assign.py:19-31` builds `(N, r)` temporaries with `r` = rects per k-chunk, and 11M × 1024 fp32 is ≈45 GB.

**Termination argument (must hold, it is why the loop is safe):** an accepted notch fill adds bins that lie strictly inside the target region's own bounding box and removes none from it, so `|bbox(r) \ r|` strictly decreases while `bbox(r)` is unchanged; every other region only shrinks, so its bbox only shrinks. Therefore `Σ_r |bbox(r) \ r|` strictly decreases on every pass and is bounded by `k·B²`. A candidate is **feasible** only if every donor region stays non-empty and 4-connected *and* the target stays 4-connected; if a region still exceeds `rect_max` and has no feasible candidate, the function raises with the region id and count rather than emitting geometry that would blow up `region_sdf_l1`.

**Files:**
- Create: `src/ioplace/producer/rectify.py`
- Test: `tests/test_producer_rectify.py`

**Interfaces:**
- Consumes: nothing from earlier tasks; imports `ioplace.regions.RegionSet`/`RegionSpec`.
- Produces (used by Task 9):
  - `mask_to_rects(mask) -> (R,4) int64` half-open `[x0, y0, x1, y1)` in bin coordinates, raster order
  - `region_rect_counts(labels, k) -> list[int]`
  - `enforce_rect_max(labels, k, rect_max=8) -> (B,B) int16`
  - `rects_to_regionset(labels, k, die, lattice=512, name_fmt="P{}") -> RegionSet` (already `validate()`d)

- [ ] **Step 1: Write the failing test**

Create `tests/test_producer_rectify.py`:

```python
import numpy as np
import pytest
from scipy import ndimage
from ioplace.producer import rectify

DIE = (0.0, 0.0, 1024.0, 1024.0)


def _comb(n=16):
    """Region 1 = a spine row plus 8 teeth (9 maximal-strip rectangles);
    region 0 = everything else."""
    lab = np.zeros((n, n), dtype=np.int16)
    lab[0, :] = 1
    for c in range(0, n, 2):
        lab[1:3, c] = 1
    return lab


def test_mask_to_rects_on_a_solid_rectangle():
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:4] = True
    r = rectify.mask_to_rects(m)
    assert r.tolist() == [[1, 1, 4, 3]]


def test_mask_to_rects_on_an_l_shape_gives_two_rects():
    m = np.zeros((4, 4), dtype=bool)
    m[0:2, 0:4] = True
    m[2:4, 0:2] = True
    r = rectify.mask_to_rects(m)
    assert len(r) == 2
    assert sorted(r.tolist()) == [[0, 0, 4, 2], [0, 2, 2, 4]]


def test_mask_to_rects_area_always_matches_the_mask():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((12, 12)) < 0.4
        r = rectify.mask_to_rects(m)
        area = int(((r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])).sum()) if len(r) else 0
        assert area == int(m.sum())


def test_region_rect_counts_sees_the_comb():
    counts = rectify.region_rect_counts(_comb(), 2)
    assert counts[1] == 9


def test_enforce_rect_max_reduces_the_comb_and_keeps_a_legal_tiling():
    lab = _comb()
    out = rectify.enforce_rect_max(lab, 2, rect_max=8)
    counts = rectify.region_rect_counts(out, 2)
    assert max(counts) <= 8
    assert out.shape == lab.shape
    assert set(np.unique(out).tolist()) == {0, 1}
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(2):
        m = out == k
        assert m.any()
        assert ndimage.label(m, structure=st)[1] == 1


def test_enforce_rect_max_is_a_noop_when_already_within_budget():
    lab = np.zeros((8, 8), dtype=np.int16)
    lab[:, 4:] = 1
    out = rectify.enforce_rect_max(lab, 2, rect_max=8)
    assert np.array_equal(out, lab)


def test_enforce_rect_max_is_deterministic():
    lab = _comb()
    a = rectify.enforce_rect_max(lab, 2, rect_max=8)
    b = rectify.enforce_rect_max(lab, 2, rect_max=8)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bins", [32, 64])
def test_rects_to_regionset_validates_and_snaps_to_the_512_lattice(bins):
    lab = np.zeros((bins, bins), dtype=np.int16)
    lab[:, bins // 2:] = 1
    lab[bins // 2:, :bins // 4] = 2
    rs = rectify.rects_to_regionset(lab, 3, DIE, lattice=512)
    rs.validate()                                   # must not raise
    assert rs.k == 3 and rs.lattice == 512
    pitch = (DIE[2] - DIE[0]) / 512
    for r in rs.regions:
        q = np.asarray(r.rects) / pitch
        assert np.allclose(q, np.round(q), atol=1e-9)
    total = sum(((np.asarray(r.rects)[:, 2] - np.asarray(r.rects)[:, 0]) *
                 (np.asarray(r.rects)[:, 3] - np.asarray(r.rects)[:, 1])).sum()
                for r in rs.regions)
    assert total == pytest.approx(1024.0 * 1024.0)


def test_rects_to_regionset_rejects_a_bin_count_that_does_not_divide_the_lattice():
    lab = np.zeros((10, 10), dtype=np.int16)
    with pytest.raises(AssertionError):
        rectify.rects_to_regionset(lab, 1, DIE, lattice=512)


def test_rects_to_regionset_rejects_an_empty_region():
    lab = np.zeros((32, 32), dtype=np.int16)
    with pytest.raises(AssertionError):
        rectify.rects_to_regionset(lab, 2, DIE, lattice=512)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_rectify.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.producer.rectify'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/producer/rectify.py`:

```python
"""Bin map -> rectilinear RegionSet with a hard rect_max budget.

spec section 2 / section 10 risk 1: maximal horizontal strips merged vertically,
then rect_max=8 enforced by filling the smallest notches, then
RegionSet.validate(). The budget is a memory contract -- ops/soft_assign.py's
region_sdf_l1 builds (N, rects-per-chunk) temporaries.
"""
import numpy as np
from scipy import ndimage

from ioplace.regions import RegionSet, RegionSpec

_CC = ndimage.generate_binary_structure(2, 1)      # 4-connected


def _row_runs(row):
    """Maximal True runs of a 1-D boolean row as (x0, x1) half-open pairs."""
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def mask_to_rects(mask):
    """Maximal horizontal strips merged vertically: identical [x0,x1) runs in
    consecutive rows become one rectangle. Returns (R,4) int64 half-open
    [x0, y0, x1, y1) in bin coordinates, sorted in raster order."""
    mask = np.asarray(mask, dtype=bool)
    ny = mask.shape[0]
    open_runs = {}
    rects = []
    for y in range(ny + 1):
        runs = set() if y == ny else set(_row_runs(mask[y]))
        for key in [k for k in open_runs if k not in runs]:
            rects.append((key[0], open_runs.pop(key), key[1], y))
        for key in runs:
            open_runs.setdefault(key, y)
    rects.sort(key=lambda r: (r[1], r[0]))
    return np.array(rects, dtype=np.int64).reshape(-1, 4)


def region_rect_counts(labels, k):
    lab = np.asarray(labels)
    return [len(mask_to_rects(lab == kk)) for kk in range(k)]


def _connected(mask):
    return bool(mask.any()) and ndimage.label(mask, structure=_CC)[1] == 1


def _feasible(lab, r, sel):
    """A candidate notch fill is feasible iff every donor stays non-empty and
    4-connected and the target stays 4-connected."""
    if not _connected((lab == r) | sel):
        return False
    for d in np.unique(lab[sel]):
        if int(d) == r:
            continue
        if not _connected((lab == int(d)) & ~sel):
            return False
    return True


def _fill_one_notch(lab, r):
    """Fill one notch of region r, preferring the smallest candidate that
    strictly reduces r's rectangle count and falling back to the smallest
    feasible candidate otherwise. Returns True if the map changed."""
    mask = lab == r
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    hole = np.zeros_like(mask)
    hole[y0:y1, x0:x1] = ~mask[y0:y1, x0:x1]
    if not hole.any():
        return False                                   # already equal to its bbox
    cc, n = ndimage.label(hole, structure=_CC)
    comps = []
    for c in range(1, n + 1):
        sel = cc == c
        comps.append((int(sel.sum()), int(np.flatnonzero(sel.reshape(-1))[0]), sel))
    comps.sort(key=lambda t: (t[0], t[1]))             # smallest notch first
    before = len(mask_to_rects(mask))
    fallback = None
    for _, _, sel in comps:
        if not _feasible(lab, r, sel):
            continue
        if fallback is None:
            fallback = sel
        if len(mask_to_rects(mask | sel)) < before:
            lab[sel] = np.int16(r)
            return True
    if fallback is None:
        return False
    lab[fallback] = np.int16(r)
    return True


def enforce_rect_max(labels, k, rect_max=8):
    """spec section 2's hard rect cap. Repeatedly fill the smallest notch of
    the region with the most rectangles.

    Termination: an accepted fill adds bins strictly inside the target's own
    bounding box and removes none, so |bbox(r) \\ r| strictly decreases while
    bbox(r) is unchanged, and every other region only shrinks -- so
    sum_r |bbox(r) \\ r| strictly decreases every pass, bounded by k*B^2.
    """
    lab = np.asarray(labels, dtype=np.int16).copy()
    budget = k * lab.size + 1
    for _ in range(budget):
        counts = region_rect_counts(lab, k)
        worst = int(np.argmax(counts))
        if counts[worst] <= rect_max:
            return lab
        if not _fill_one_notch(lab, worst):
            raise RuntimeError(
                f"region {worst} has {counts[worst]} rects (> {rect_max}) and no "
                "feasible notch fill remains; lower --extract-bins or K")
    raise RuntimeError(
        f"enforce_rect_max did not converge within {budget} passes")


def rects_to_regionset(labels, k, die, lattice=512, name_fmt="P{}"):
    """Bin-coordinate rectangles -> a RegionSet on `lattice`. Requires
    lattice % bins == 0 so every edge lands exactly on the lattice (64 | 512 and
    32 | 512, spec section 2), which is what makes validate() pass."""
    lab = np.asarray(labels)
    b = lab.shape[0]
    assert lab.shape[0] == lab.shape[1], "label map must be square"
    assert lattice % b == 0, \
        f"lattice {lattice} must be a multiple of the bin count {b}"
    xl, yl, xh, yh = (float(v) for v in die)
    cw, ch = (xh - xl) / b, (yh - yl) / b
    regions = []
    for kk in range(k):
        r = mask_to_rects(lab == kk).astype(np.float64)
        assert len(r), f"region {kk} is empty; cannot build a RegionSpec"
        regions.append(RegionSpec(name_fmt.format(kk), np.stack([
            xl + r[:, 0] * cw, yl + r[:, 1] * ch,
            xl + r[:, 2] * cw, yl + r[:, 3] * ch], axis=1)))
    rs = RegionSet(die=(xl, yl, xh, yh), lattice=int(lattice), regions=regions)
    rs.validate()
    return rs
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_rectify.py -v
```
Expected: PASS — 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/rectify.py tests/test_producer_rectify.py
git commit -m "feat(producer): strip decomposition and hard rect_max=8 notch filling

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Membership prior (`producer/membership.py`)

Spec §2: "`partition_netlist` (`partition/mtkahypar_runner.py:14`), `K=16`, `epsilon=0.03`, or RTL hierarchy prefixes; substitutes for GrandPlan's given RTL partitions and runs before the flat GP." GrandPlan takes partition membership as an **input** and never re-partitions (digest §1); this module is that input. The membership is fixed for the whole producer run — the grouping loss, the extraction and `membership.npz` all use the same labels, which is also what arm (e) needs (spec §8: "fixed Mt-KaHyPar membership").

**No block→region matching.** Spec §2 is explicit: `run_placement_two_stage.assign_blocks_to_regions` is unused here, because hulls give geometry, not a permutation.

**Files:**
- Create: `src/ioplace/producer/membership.py`
- Test: `tests/test_producer_membership.py`

**Interfaces:**
- Consumes: `ioplace.partition.mtkahypar_runner.partition_netlist`.
- Produces (used by Task 9):
  - `mtkahypar_membership(nl, k, epsilon=0.03, seed=0, threads=8) -> (num_movable,) int32`
  - `hierarchy_membership(node_names, num_movable, k, depth=1) -> (num_movable,) int32`
  - `build_membership(source, *, nl, node_names, num_movable, k, epsilon=0.03, seed=0, depth=1) -> (num_movable,) int32`

- [ ] **Step 1: Write the failing test**

Create `tests/test_producer_membership.py`:

```python
import numpy as np
import pytest
from ioplace.netlist import Netlist
from ioplace.producer import membership


def _names(spec):
    out = []
    for prefix, n in spec:
        out.extend([f"{prefix}{i}".encode() for i in range(n)])
    return np.array(out)


def test_hierarchy_groups_share_a_label_and_are_balanced():
    """LPT packing: a(10)->bucket0, b(6)->bucket1, c(4)->bucket1 (now 10),
    d(1)->bucket0 on the tie. Two buckets of 11 and 10."""
    names = _names([("a/x", 10), ("b/y", 6), ("c/z", 4), ("d", 1)])
    part = membership.hierarchy_membership(names, 21, 2)
    assert part.dtype == np.int32 and part.shape == (21,)
    assert len(set(part[:10].tolist())) == 1
    assert len(set(part[10:16].tolist())) == 1
    assert part[0] != part[10]
    assert sorted(np.bincount(part, minlength=2).tolist()) == [10, 11]


def test_hierarchy_is_deterministic():
    names = _names([("a/x", 7), ("b/y", 5), ("c/z", 3)])
    a = membership.hierarchy_membership(names, 15, 3)
    b = membership.hierarchy_membership(names, 15, 3)
    assert np.array_equal(a, b)


def test_hierarchy_depth_splits_deeper_prefixes():
    names = np.array([b"top/a/x0", b"top/a/x1", b"top/b/y0", b"top/b/y1"])
    with pytest.raises(ValueError):
        membership.hierarchy_membership(names, 4, 2, depth=1)   # one prefix only
    part = membership.hierarchy_membership(names, 4, 2, depth=2)
    assert part[0] == part[1] and part[2] == part[3] and part[0] != part[2]


def test_hierarchy_rejects_fewer_prefix_groups_than_k():
    names = _names([("a/x", 4), ("b/y", 4)])
    with pytest.raises(ValueError, match="prefix groups"):
        membership.hierarchy_membership(names, 8, 4)


def test_hierarchy_accepts_str_names_too():
    names = ["a/x0", "a/x1", "b/y0", "b/y1"]
    part = membership.hierarchy_membership(names, 4, 2)
    assert part[0] == part[1] and part[0] != part[2]


def _two_clique_netlist(n_per=20):
    """Two 20-node cliques joined by a single net -- a hypergraph any
    partitioner must cut in exactly one place."""
    nets = []
    for base in (0, n_per):
        for i in range(n_per - 1):
            nets.append([base + i, base + i + 1])
        nets.append(list(range(base, base + n_per)))
    nets.append([0, n_per])
    pins, p2n, p2e, start = [], [], [], [0]
    for e, nodes in enumerate(nets):
        pins.extend(nodes)
        p2n.extend(nodes)
        p2e.extend([e] * len(nodes))
        start.append(len(p2n))
    n = 2 * n_per
    return Netlist(
        node_x=np.zeros(n), node_y=np.zeros(n),
        node_size_x=np.ones(n), node_size_y=np.ones(n),
        num_movable=n, num_terminals=0, num_terminal_NIs=0,
        pin_offset_x=np.zeros(len(p2n)), pin_offset_y=np.zeros(len(p2n)),
        pin2node=np.array(p2n, dtype=np.int32),
        pin2net=np.array(p2e, dtype=np.int32),
        flat_net2pin=np.arange(len(p2n), dtype=np.int32),
        flat_net2pin_start=np.array(start, dtype=np.int32),
        xl=0.0, yl=0.0, xh=100.0, yh=100.0)


def test_mtkahypar_membership_shape_range_and_determinism():
    pytest.importorskip("mtkahypar")
    nl = _two_clique_netlist()
    a = membership.mtkahypar_membership(nl, 2, seed=0)
    assert a.dtype == np.int32 and a.shape == (nl.num_movable,)
    assert int(a.min()) >= 0 and int(a.max()) < 2
    b = membership.mtkahypar_membership(nl, 2, seed=0)
    assert np.array_equal(a, b)


def test_build_membership_dispatches_and_rejects_unknown_sources():
    names = _names([("a/x", 4), ("b/y", 4)])
    part = membership.build_membership(
        "hierarchy", nl=None, node_names=names, num_movable=8, k=2)
    assert part.shape == (8,)
    with pytest.raises(ValueError, match="membership source"):
        membership.build_membership(
            "magic", nl=None, node_names=names, num_movable=8, k=2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_membership.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.producer.membership'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/producer/membership.py`:

```python
"""The producer's membership prior.

GrandPlan takes RTL partition membership as an input and never re-partitions
(digest section 1). spec section 2 substitutes Mt-KaHyPar (K=16, epsilon=0.03)
or RTL hierarchy prefixes, run BEFORE the flat GP. The labels are fixed for the
whole producer run: the grouping loss, the extraction and membership.npz all
read the same array, which is also what arm (e) needs.

No block->region matching: spec section 2 explicitly rules out
run_placement_two_stage.assign_blocks_to_regions here, because the hulls supply
geometry rather than a permutation.
"""
import numpy as np

from ioplace.partition.mtkahypar_runner import partition_netlist


def mtkahypar_membership(nl, k, epsilon=0.03, seed=0, threads=8):
    """Mt-KaHyPar KM1 partition, truncated to the movable prefix.

    `threads` is only a request: mtkahypar_runtime honours
    IOPLACE_MTKAHYPAR_THREADS (src/scripts/env.sh defaults it to 1 because the
    installed 1.6.2 wheel crashes in parallel coarsening).
    """
    part = partition_netlist(nl, int(k), epsilon=float(epsilon), seed=int(seed),
                             threads=int(threads))
    return np.asarray(part[:nl.num_movable], dtype=np.int32)


def _prefix(name, depth):
    s = name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
    parts = s.split("/")
    return "/".join(parts[:depth]) if len(parts) > depth else (
        "/".join(parts[:-1]) if len(parts) > 1 else "")


def hierarchy_membership(node_names, num_movable, k, depth=1):
    """Group movable cells by the first `depth` slash-separated name
    components, then pack the groups into k buckets longest-processing-time
    first (largest group into the currently lightest bucket; ties on the lowest
    bucket index). Fully deterministic."""
    m = int(num_movable)
    names = list(node_names)[:m]
    keys = [_prefix(n, depth) for n in names]
    uniq, inv = np.unique(np.array(keys, dtype=object), return_inverse=True)
    if len(uniq) < k:
        raise ValueError(
            f"hierarchy membership found {len(uniq)} prefix groups at depth "
            f"{depth}, fewer than k={k}; raise --hierarchy-depth or use "
            "--membership mtkahypar")
    sizes = np.bincount(inv, minlength=len(uniq))
    order = sorted(range(len(uniq)), key=lambda g: (-int(sizes[g]), str(uniq[g])))
    load = np.zeros(k, dtype=np.int64)
    group_bucket = np.zeros(len(uniq), dtype=np.int32)
    for g in order:
        b = int(np.argmin(load))            # np.argmin -> lowest index on ties
        group_bucket[g] = b
        load[b] += int(sizes[g])
    return group_bucket[inv].astype(np.int32)


def build_membership(source, *, nl, node_names, num_movable, k, epsilon=0.03,
                     seed=0, depth=1, threads=8):
    """Dispatch for the driver's --membership {mtkahypar,hierarchy}."""
    if source == "mtkahypar":
        return mtkahypar_membership(nl, k, epsilon=epsilon, seed=seed,
                                    threads=threads)
    if source == "hierarchy":
        return hierarchy_membership(node_names, num_movable, k, depth=depth)
    raise ValueError(
        f"unknown membership source {source!r}; expected 'mtkahypar' or 'hierarchy'")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_membership.py -v
```
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/membership.py tests/test_producer_membership.py
git commit -m "feat(producer): Mt-KaHyPar and RTL-hierarchy membership priors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: GPU candidate reduction (`producer/hull.py`, part 3)

Spec §2 says the candidate reduction runs **on GPU**, and the runtime budget makes that mandatory rather than decorative: at 11M cells and K=16 the numpy path does ~360M projections per rebuild, which at `T_hull=50` over a 2989 s flat GP would cost far more than the "+<3%" the spec allows. Moving the projections and the quantile onto the device keeps the rebuild in the tens of milliseconds and leaves quickhull with ≤~1024 points.

**Files:**
- Modify: `src/ioplace/producer/hull.py` (append)
- Modify: `tests/test_producer_hull.py` (append)

**Interfaces:**
- Consumes: `hull.reduce_candidates` (Task 2), for the equivalence oracle.
- Produces (used by Task 10):
  - `QUANTILE_SUBSAMPLE = 8_000_000`
  - `reduce_candidates_torch(x, y, m=16, q=0.90, alpha=0.25, k_dir=64) -> np.ndarray` — `(N,)` torch tensors in, `(M,2)` float64 numpy out

- [ ] **Step 1: Write the failing test**

Append to `tests/test_producer_hull.py`:

```python
import torch


@pytest.mark.gpu
def test_reduce_candidates_torch_matches_the_numpy_path():
    """Continuous random data, so exact ties (where torch.topk and numpy's
    stable argsort could disagree) have measure zero."""
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(50_000, 2)) * np.array([3.0, 1.0])
    want = hull.reduce_candidates(pts)
    x = torch.as_tensor(pts[:, 0], device="cuda", dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], device="cuda", dtype=torch.float64)
    got = hull.reduce_candidates_torch(x, y)
    assert got.shape[1] == 2
    assert {tuple(r) for r in got.tolist()} == {tuple(r) for r in want.tolist()}


@pytest.mark.gpu
def test_reduce_candidates_torch_passes_small_sets_through():
    x = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0], device="cuda", dtype=torch.float64)
    got = hull.reduce_candidates_torch(x, y)
    assert sorted(map(tuple, got.tolist())) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]


@pytest.mark.gpu
def test_reduce_candidates_torch_feeds_a_usable_hull():
    rng = np.random.default_rng(12)
    pts = rng.uniform(0.0, 10.0, size=(20_000, 2))
    x = torch.as_tensor(pts[:, 0], device="cuda", dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], device="cuda", dtype=torch.float64)
    v = hull.convex_hull(hull.reduce_candidates_torch(x, y))
    assert hull.polygon_area(v) == pytest.approx(100.0, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_hull.py -v
```
Expected: FAIL — `AttributeError: module 'ioplace.producer.hull' has no attribute 'reduce_candidates_torch'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ioplace/producer/hull.py` (add `import math` at the top):

```python
QUANTILE_SUBSAMPLE = 8_000_000


def reduce_candidates_torch(x, y, m=DIRECTIONS_M, q=QUANTILE_Q,
                            alpha=BAND_ALPHA, k_dir=K_DIR):
    """Algorithm 1 on the device (spec section 2: "Algorithm-1 candidate
    reduction on GPU"). Same semantics as reduce_candidates; only the
    projections, the quantile and the top-k live on the GPU, and only the
    surviving <=2*m*k_dir candidates come back to the host for quickhull.

    torch.quantile has an input-element limit, so above QUANTILE_SUBSAMPLE the
    threshold is estimated from a deterministic stride subsample -- a threshold
    estimate, not a filter: the band test still runs over every point.
    """
    n = int(x.numel())
    if n <= k_dir:
        return np.unique(torch.stack([x, y], dim=1).double().cpu().numpy(), axis=0)
    keep = torch.zeros(n, dtype=torch.bool, device=x.device)
    stride = max(1, n // QUANTILE_SUBSAMPLE + (1 if n % QUANTILE_SUBSAMPLE else 0))
    for j in range(m):
        th = j * (2.0 * math.pi / m)
        s = x.double() * math.cos(th) + y.double() * math.sin(th)
        for sign in (1.0, -1.0):
            ss = sign * s
            t = torch.quantile(ss[::stride].contiguous(), q)
            hi = ss.max()
            idx = ((ss >= t) & (ss <= t + alpha * (hi - t))).nonzero(as_tuple=True)[0]
            if idx.numel() > k_dir:
                sel = torch.topk(ss[idx] - t, k_dir, largest=False, sorted=True).indices
                idx = idx[sel]
            keep[idx] = True
    return np.unique(
        torch.stack([x[keep], y[keep]], dim=1).double().cpu().numpy(), axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_producer_hull.py -v
```
Expected: PASS — 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/producer/hull.py tests/test_producer_hull.py
git commit -m "feat(producer): GPU Algorithm-1 candidate reduction for hull rebuilds

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: The producer driver (`drivers/run_region_producer.py`)

Sequences everything: read → prior → flat GP with the grouping term attached and the hull tables rebuilt every `T_hull` from the iteration callback → flat LG → extraction → SA → rectification → four artefacts. This is the only file that touches DREAMPlace.

Three integration rules that must be respected exactly:

1. **The coordinate flip.** Everything inside the GP (hull tables, `ea_scaled`, the die passed to `anchor_tables`) is in the **scaled** post-`initialize()` system. Everything written out (`regions.json`, `seed.npz`) and everything from extraction onward (`extract`, SA `bin_area`/`ea`, `rects_to_regionset`) is in the **native** post-`read()` system: `x_native = x_scaled / scale_factor + shift_factor[0]`, `size_native = size_scaled / scale_factor`. This is what P-B consumes: its `init_pos.apply_init(..., "seed")` writes these native values straight into `placedb.node_x`/`node_y` after `read()` and before `initialize()`, and `fence_phase.build_fence_placedb` rejects a `regions.json` whose die is not the native post-read die. The fingerprint is likewise taken in the `read` phase below, before `initialize()` rescales the node sizes it digests.
2. **The obj_version discipline** (`dp_hook.py:36-59`, design v2 §6.4). A hull rebuild and a λ update are both discrete objective changes. Let the new tables and λ take effect first, *then* call `refresh_nesterov_secant(placer.optimizer)` and `mark_refreshed()` — the same order `run_placement_io.py:584-588` uses.
3. **Centre init, no fence.** GrandPlan §4.2 initialises the flat placement at the chiplet centre, so `random_center_init_flag` is left at its configured value and no fence data is injected. `np.random.seed(params.random_seed)` immediately before `NonLinearPlace(...)`, because BasicPlace draws centre noise and filler positions from numpy's *global* RNG (`run_placement._place`'s comment).

**Files:**
- Create: `src/ioplace/drivers/run_region_producer.py`
- Test: `tests/test_run_region_producer.py`

**Interfaces:**
- Consumes: `artifacts.{placedb_identity_sha256,save_positions,save_membership,save_producer_json}` (P-B Task 1, verified in Task 1; this task's tests also use `artifacts.{load_positions,load_membership,load_producer_json}`); `hull.{reduce_candidates_torch,macro_pseudo_points,build_hull,anchor_tables}` (Tasks 2, 3, 9); `grouping_term.{GroupingTerm,GroupingWeight}` (Task 4); `extract.extract` (Task 5); `sa.{SaConfig,anneal}` (Task 6); `rectify.{enforce_rect_max,rects_to_regionset,region_rect_counts}` (Task 7); `membership.build_membership` (Task 8); `run_placement.{_load_dreamplace,extract_final_positions}`; `dp_hook.{attach_terms,assert_optimizer_lock,refresh_nesterov_secant}`; `profile.PhaseTimer`.
- Produces:
  - `LATTICE = 512`
  - `class VersionState` with `obj_version`, `refreshed_version`, `bump()`, `needs_refresh()`, `mark_refreshed()`
  - `run_producer(config_json, out_dir, *, k=16, membership_source="mtkahypar", extract_bins=64, rect_max=8, seed=0, epsilon=0.03, t_hull=50, probe_every=50, hierarchy_depth=1, sa_seed=0, alpha_pull=1.0, alpha_push=1.0, skip_sa=False, dp_seed=None, deterministic=None) -> dict`
  - `main()` CLI

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_region_producer.py`:

```python
import json
import os
import numpy as np
import pytest
from ioplace import artifacts
from ioplace.regions import RegionSet

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
CFG = os.path.join(DP, "install", "test", "simple.json")


def test_version_state_tracks_the_refresh_contract():
    from ioplace.drivers.run_region_producer import VersionState
    v = VersionState()
    assert not v.needs_refresh()
    v.bump()
    assert v.needs_refresh()
    v.mark_refreshed()
    assert not v.needs_refresh()


@pytest.mark.slow
def test_producer_emits_all_four_artefacts_on_simple(tmp_path):
    from ioplace.drivers.run_region_producer import run_producer
    out = str(tmp_path / "run")
    res = run_producer(CFG, out, k=2, membership_source="mtkahypar",
                       extract_bins=32, rect_max=8, seed=0, t_hull=20,
                       probe_every=20, sa_seed=0)
    for name in ("regions.json", "seed.npz", "membership.npz", "producer.json"):
        assert os.path.exists(os.path.join(out, name)), name

    rs = RegionSet.from_json(os.path.join(out, "regions.json"))
    rs.validate()
    assert rs.k == 2 and rs.lattice == 512
    for r in rs.regions:
        assert 1 <= len(np.asarray(r.rects)) <= 8

    seed = artifacts.load_positions(os.path.join(out, "seed.npz"),
                                    expect_num_physical=res["num_physical"],
                                    expect_sha256=res["placedb_sha256"])
    assert seed.kind == "seed"
    assert seed.node_x.shape == seed.node_y.shape
    assert seed.die == tuple(res["die_native"])

    mem = artifacts.load_membership(os.path.join(out, "membership.npz"),
                                    expect_num_movable=res["num_movable"],
                                    expect_k=2)
    assert mem.source == "mtkahypar"

    doc = artifacts.load_producer_json(os.path.join(out, "producer.json"))
    assert doc["k"] == 2 and doc["extract_bins"] == 32 and doc["rect_max"] == 8
    assert doc["n_hull_rebuilds"] >= 1
    assert max(doc["rects_per_region"]) <= 8
    assert doc["runtime_s"]["total"] > 0.0
    assert set(doc["runtime_s"]) >= {"read", "prior", "gp", "extract", "sa",
                                     "rectify", "total"}


@pytest.mark.slow
def test_seed_is_in_native_units_and_inside_the_die(tmp_path):
    """The coordinate contract: seed.npz must be in the post-read die box, not
    the post-initialize scaled one."""
    from ioplace.drivers.run_region_producer import run_producer
    out = str(tmp_path / "run")
    res = run_producer(CFG, out, k=2, extract_bins=32, t_hull=20, probe_every=20)
    xl, yl, xh, yh = res["die_native"]
    assert (xl, yl) != (0.0, 0.0), "simple.json's die does not start at the origin"
    s = artifacts.load_positions(os.path.join(out, "seed.npz"))
    assert s.shift_factor == (xl, yl)
    assert (s.node_x >= xl - 1e-6).all() and (s.node_x <= xh + 1e-6).all()
    assert (s.node_y >= yl - 1e-6).all() and (s.node_y <= yh + 1e-6).all()


@pytest.mark.slow
def test_extract_bins_32_and_64_both_produce_valid_geometry(tmp_path):
    """spec section 8: arm (e) needs faithful 32^2 extraction, so the producer
    must expose --extract-bins {32,64} and both must validate."""
    from ioplace.drivers.run_region_producer import run_producer
    for bins in (32, 64):
        out = str(tmp_path / f"b{bins}")
        res = run_producer(CFG, out, k=2, extract_bins=bins, t_hull=20,
                           probe_every=20)
        rs = RegionSet.from_json(os.path.join(out, "regions.json"))
        rs.validate()
        assert res["extract_bins"] == bins
        assert rs.lattice == 512


@pytest.mark.slow
def test_producer_is_reproducible_with_the_same_seeds(tmp_path):
    from ioplace.drivers.run_region_producer import run_producer
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    for out in (a, b):
        run_producer(CFG, out, k=2, extract_bins=32, seed=0, sa_seed=0,
                     t_hull=20, probe_every=20, dp_seed=1000, deterministic=1)
    pa = artifacts.load_membership(os.path.join(a, "membership.npz")).part
    pb = artifacts.load_membership(os.path.join(b, "membership.npz")).part
    assert np.array_equal(pa, pb)
    ja = json.load(open(os.path.join(a, "regions.json")))
    jb = json.load(open(os.path.join(b, "regions.json")))
    assert ja["regions"] == jb["regions"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_run_region_producer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.drivers.run_region_producer'`.

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/drivers/run_region_producer.py`:

```python
"""Driver 1 of the v2 flow: the region producer (spec section 1 / section 2).

In:  a DREAMPlace config, K, and a membership source.
Out: regions.json, seed.npz, membership.npz, producer.json in one directory.

Pipeline (digest section 5's stages 1-2): membership prior -> flat GP with the
GrandPlan grouping loss (hull tables rebuilt every T_hull from the iteration
callback) -> flat LG -> density-argmax extraction -> SA -> rect_max -> RegionSet.

Coordinate contract: everything inside the GP is in the SCALED post-initialize()
system; everything written out is in the NATIVE post-read() system
(x_native = x_scaled / scale_factor + shift_factor[0]).
"""
import argparse
import os
import socket
import sys
import time

import numpy as np

from ioplace.artifacts import (placedb_identity_sha256, save_membership,
                               save_positions, save_producer_json)
from ioplace.dp_hook import (assert_optimizer_lock, attach_terms,
                             refresh_nesterov_secant)
from ioplace.drivers.run_placement import (_load_dreamplace,
                                           extract_final_positions)
from ioplace.netlist import netlist_from_placedb
from ioplace.producer import extract as extract_mod
from ioplace.producer import hull as hull_mod
from ioplace.producer import rectify as rectify_mod
from ioplace.producer import sa as sa_mod
from ioplace.producer.grouping_term import GroupingTerm, GroupingWeight
from ioplace.producer.membership import build_membership
from ioplace.paths import REPO_ROOT
from ioplace.profile import PhaseTimer, env_metadata

LATTICE = 512


class VersionState:
    """dp_hook's obj_version / refreshed_version discipline (design v2 sec 6.4):
    a hull rebuild or a lambda change is a DISCRETE objective change, so the
    Nesterov secant cache must be recomputed before the next optimizer step."""

    def __init__(self):
        self.obj_version = 0
        self.refreshed_version = 0

    def bump(self):
        self.obj_version += 1

    def needs_refresh(self):
        return self.obj_version != self.refreshed_version

    def mark_refreshed(self):
        self.refreshed_version = self.obj_version


def run_producer(config_json, out_dir, *, k=16, membership_source="mtkahypar",
                 extract_bins=64, rect_max=8, seed=0, epsilon=0.03, t_hull=50,
                 probe_every=50, hierarchy_depth=1, sa_seed=0, alpha_pull=1.0,
                 alpha_push=1.0, skip_sa=False, dp_seed=None,
                 deterministic=None):
    import torch

    t_start = time.time()
    timer = PhaseTimer()
    os.makedirs(out_dir, exist_ok=True)

    # ---- read: native post-read coordinate system ------------------------
    with timer.phase("read"):
        params, placedb = _load_dreamplace(config_json)
        if dp_seed is not None:
            params.random_seed = dp_seed
        if deterministic is not None:
            params.deterministic_flag = deterministic
        die_native = (float(placedb.xl), float(placedb.yl),
                      float(placedb.xh), float(placedb.yh))
        nl0 = netlist_from_placedb(placedb)
        # Before initialize(): the digest covers node sizes, which PlaceDB.scale()
        # multiplies by scale_factor (PlaceDB.py:160-161). Hashing here is what
        # makes the main flow's expect_sha256 check agree across processes.
        fingerprint = placedb_identity_sha256(placedb)
        node_names = getattr(placedb, "node_names", None)

    import NonLinearPlace   # importable only after _load_dreamplace's setup

    with timer.phase("prior"):
        part = build_membership(membership_source, nl=nl0, node_names=node_names,
                                num_movable=int(placedb.num_movable_nodes), k=k,
                                epsilon=epsilon, seed=seed, depth=hierarchy_depth)

    # ---- initialize: scaled coordinate system ----------------------------
    with timer.phase("initialize"):
        placedb.initialize(params)
        assert_optimizer_lock(params)
        die_scaled = (float(placedb.xl), float(placedb.yl),
                      float(placedb.xh), float(placedb.yh))
        scale = float(params.scale_factor)
        shift = (float(params.shift_factor[0]), float(params.shift_factor[1]))
        m = int(placedb.num_movable_nodes)
        n_all = int(placedb.num_nodes)
        n_phys = int(placedb.num_physical_nodes)
        size_x = np.asarray(placedb.node_size_x[:m], dtype=np.float64)
        size_y = np.asarray(placedb.node_size_y[:m], dtype=np.float64)
        cell_area = size_x * size_y
        target_density = float(params.target_density)
        ea_scaled = np.array(
            [cell_area[part == kk].sum() / target_density for kk in range(k)])
        row_h = float(placedb.row_height)
        is_macro = size_y > 2.0 * row_h
        is_std = ~is_macro
        pitch_x = float(size_x[is_std].mean()) if is_std.any() else float(size_x.mean())
        pitch_y = float(size_y[is_std].mean()) if is_std.any() else float(size_y.mean())
        macro_idx = np.nonzero(is_macro)[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    term = GroupingTerm(part=part, node_size_x=size_x, node_size_y=size_y,
                        num_movable=m, num_physical=n_phys, num_nodes=n_all,
                        alpha_pull=alpha_pull, alpha_push=alpha_push,
                        device=device)
    weight = GroupingWeight()
    version = VersionState()
    attach_terms(params, [lambda pos: term(pos, weight.lam)])

    part_t = torch.as_tensor(part.astype(np.int64), device=device)
    half_x = torch.as_tensor(0.5 * size_x, dtype=torch.float64, device=device)
    half_y = torch.as_tensor(0.5 * size_y, dtype=torch.float64, device=device)
    die_box = np.array([[die_scaled[0], die_scaled[1]], [die_scaled[2], die_scaled[1]],
                        [die_scaled[2], die_scaled[3]], [die_scaled[0], die_scaled[3]]])

    def rebuild(pos):
        """Recompute every partition's hull from the current cell centres and
        re-rasterise the anchor tables. Frozen until the next rebuild."""
        with torch.no_grad():
            cx = pos[:m].double() + half_x
            cy = pos[n_all:n_all + m].double() + half_y
        hulls = []
        for kk in range(k):
            sel = (part_t == kk).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                hulls.append(die_box)
                continue
            if cx.is_cuda:
                pts = hull_mod.reduce_candidates_torch(cx[sel], cy[sel])
            else:
                pts = hull_mod.reduce_candidates(
                    torch.stack([cx[sel], cy[sel]], dim=1).numpy())
            mk = macro_idx[part[macro_idx] == kk]
            if len(mk):
                mx = cx[mk].cpu().numpy() - 0.5 * size_x[mk]
                my = cy[mk].cpu().numpy() - 0.5 * size_y[mk]
                pts = np.vstack([pts, hull_mod.macro_pseudo_points(
                    mx, my, size_x[mk], size_y[mk], pitch_x, pitch_y)])
            hulls.append(hull_mod.build_hull(pts, a_max=float(ea_scaled[kk])))
        term.set_tables(hull_mod.anchor_tables(hulls, die_scaled, LATTICE,
                                               device=device))
        version.bump()

    probes = []
    cb_state = {"last_iteration": -1}

    def cb(iteration, pos):
        cb_state["last_iteration"] = iteration
        if iteration % t_hull == 0:
            rebuild(pos)
        if iteration % probe_every == 0 and term.tables is not None:
            placer.model.op_collections.wirelength_op(pos).backward()
            g_wl = float(pos.grad.abs().sum())
            pos.grad.zero_()
            g_group = term.grad_l1(pos.detach())
            weight.update(iteration, g_wl, g_group)
            version.bump()
            probes.append({"iteration": int(iteration), "grad_l1_wl": g_wl,
                           "grad_l1_group": g_group,
                           "wt": weight.wt(iteration),
                           "lambda_group": weight.lam})
        # Order is fixed: the new tables / lambda take effect first, then the
        # secant cache is rebuilt under the new objective (run_placement_io.py:584).
        if version.needs_refresh():
            refresh_nesterov_secant(placer.optimizer)
            version.mark_refreshed()

    np.random.seed(params.random_seed)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

    orig_legalize = placer.op_collections.legalize_op
    hpwl = {}
    gp_phase = timer.phase("gp")
    gp_phase.__enter__()

    def _timed_legalize(p):
        gp_phase.__exit__(None, None, None)
        with torch.no_grad():
            hpwl["hpwl_gp"] = float(placer.op_collections.hpwl_op(p))
        with timer.phase("lg"):
            out = orig_legalize(p)
        with torch.no_grad():
            hpwl["hpwl_lg"] = float(placer.op_collections.hpwl_op(out))
        return out

    placer.op_collections.legalize_op = _timed_legalize
    placer.iteration_callback = cb
    placer(params, placedb, params.global_place_stages[0]["learning_rate"])
    if "gp" not in timer.phases:
        gp_phase.__exit__(None, None, None)
    final_overflow = float(placer.model.overflow.max())

    # ---- back to native units -------------------------------------------
    with timer.phase("extract"):
        x_s, y_s = extract_final_positions(placer, placedb)
        x_n = x_s / scale + shift[0]
        y_n = y_s / scale + shift[1]
        w_n = np.asarray(placedb.node_size_x[:n_phys], dtype=np.float64) / scale
        h_n = np.asarray(placedb.node_size_y[:n_phys], dtype=np.float64) / scale
        labels0 = extract_mod.extract(x_n[:m], y_n[:m], w_n[:m], h_n[:m],
                                      part, k, die_native, out_bins=extract_bins)

    bin_area = ((die_native[2] - die_native[0]) / extract_bins) * \
               ((die_native[3] - die_native[1]) / extract_bins)
    ea_native = ea_scaled / (scale * scale)
    with timer.phase("sa"):
        if skip_sa:
            labels, sa_report = labels0.copy(), {"skipped": True}
        else:
            labels, sa_report = sa_mod.anneal(
                labels0, k, ea_native, bin_area, sa_mod.SaConfig(seed=sa_seed))

    with timer.phase("rectify"):
        labels = rectify_mod.enforce_rect_max(labels, k, rect_max=rect_max)
        rs = rectify_mod.rects_to_regionset(labels, k, die_native, lattice=LATTICE)

    # ---- artefacts -------------------------------------------------------
    rs.to_json(os.path.join(out_dir, "regions.json"))
    save_positions(os.path.join(out_dir, "seed.npz"), x_n, y_n, die=die_native,
                   shift_factor=shift, scale_factor=scale,
                   placedb_sha256=fingerprint, kind="seed")
    save_membership(os.path.join(out_dir, "membership.npz"), part,
                    source=membership_source, k=k, seed=seed, epsilon=epsilon)

    counts = rectify_mod.region_rect_counts(labels, k)
    bins_per_region = np.bincount(labels.ravel(), minlength=k)[:k]
    region_area = bins_per_region.astype(np.float64) * bin_area
    cell_area_native = cell_area / (scale * scale)
    region_cell_area = np.array(
        [cell_area_native[part == kk].sum() for kk in range(k)])
    util = np.divide(region_cell_area, region_area,
                     out=np.zeros(k), where=region_area > 0)
    runtime = {name: float(p["t_s"]) for name, p in timer.phases.items()}
    runtime["total"] = time.time() - t_start

    payload = {
        "config": os.path.abspath(config_json), "out_dir": os.path.abspath(out_dir),
        "k": int(k), "membership_source": membership_source,
        "membership_seed": int(seed), "epsilon": float(epsilon),
        "hierarchy_depth": int(hierarchy_depth),
        "extract_bins": int(extract_bins), "fine_bins": int(extract_mod.FINE_BINS),
        "lattice": LATTICE, "rect_max": int(rect_max),
        "t_hull": int(t_hull), "probe_every": int(probe_every),
        "n_hull_rebuilds": int(term.n_rebuilds),
        "alpha_pull": float(alpha_pull), "alpha_push": float(alpha_push),
        "wt_final": weight.wt(cb_state["last_iteration"]),
        "lambda_group_final": float(weight.lam),
        "ratio_ema_final": (None if weight.ratio_ema is None
                            else float(weight.ratio_ema)),
        "die_native": list(die_native), "die_scaled": list(die_scaled),
        "shift_factor": list(shift), "scale_factor": scale,
        "placedb_sha256": fingerprint,
        "num_movable": m, "num_physical": n_phys, "num_nodes": n_all,
        "num_nets": int(nl0.num_nets), "target_density": target_density,
        "gp_iterations_run": int(cb_state["last_iteration"]) + 1,
        "final_overflow": final_overflow,
        "hpwl_gp": hpwl.get("hpwl_gp"), "hpwl_lg": hpwl.get("hpwl_lg"),
        "sa": sa_report, "sa_seed": int(sa_seed),
        "rects_per_region": [int(c) for c in counts],
        "rect_max_observed": int(max(counts)),
        "region_bins": [int(b) for b in bins_per_region],
        "region_area": region_area.tolist(),
        "region_cell_area": region_cell_area.tolist(),
        "region_utilisation": util.tolist(),
        "area_balance": {"max_util": float(util.max()),
                         "min_util": float(util.min()),
                         "max_over_min": (float(util.max() / util.min())
                                          if util.min() > 0 else None)},
        "probes": probes,
        "runtime_s": runtime,
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                        if torch.cuda.is_available() else 0.0),
        "command": " ".join(sys.argv), "hostname": socket.gethostname(),
        "env": env_metadata(str(REPO_ROOT), os.environ.get("DREAMPLACE_ROOT", ""),
                            input_paths=(config_json,)),
    }
    save_producer_json(os.path.join(out_dir, "producer.json"), payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="v2 region producer (P-C)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--membership", choices=["mtkahypar", "hierarchy"],
                    default="mtkahypar")
    ap.add_argument("--extract-bins", type=int, choices=[32, 64], default=64,
                    help="64 = spec default; 32 = faithful GrandPlan for arm (e)")
    ap.add_argument("--rect-max", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--t-hull", type=int, default=50)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--hierarchy-depth", type=int, default=1)
    ap.add_argument("--sa-seed", type=int, default=0)
    ap.add_argument("--alpha-pull", type=float, default=1.0)
    ap.add_argument("--alpha-push", type=float, default=1.0)
    ap.add_argument("--no-sa", action="store_true")
    ap.add_argument("--dp-seed", type=int, default=None)
    ap.add_argument("--deterministic", type=int, default=None)
    a = ap.parse_args()
    run_producer(a.config, a.out, k=a.k, membership_source=a.membership,
                 extract_bins=a.extract_bins, rect_max=a.rect_max, seed=a.seed,
                 epsilon=a.epsilon, t_hull=a.t_hull, probe_every=a.probe_every,
                 hierarchy_depth=a.hierarchy_depth, sa_seed=a.sa_seed,
                 alpha_pull=a.alpha_pull, alpha_push=a.alpha_push,
                 skip_sa=a.no_sa, dp_seed=a.dp_seed,
                 deterministic=a.deterministic)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_run_region_producer.py -v
```
Expected: PASS — 5 passed (1 fast, 4 `slow`; each `simple.json` run takes tens of seconds).

- [ ] **Step 5: Run the whole suite**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```
Expected: no new failures relative to the pre-task baseline. Record the baseline with `git stash && "$IOPLACE_PYTHON" -m pytest -q | tail -3 && git stash pop` if any pre-existing failure is in doubt.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/drivers/run_region_producer.py tests/test_run_region_producer.py
git commit -m "feat(producer): run_region_producer driver emitting the four v2 artefacts

Writes regions.json, seed.npz, membership.npz and producer.json through
src/ioplace/artifacts.py (P-B Task 1, contract revision <sha from P-C Task 1
Step 3>); this plan defines no artefact I/O of its own.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Acceptance run on `mempool_tile_wrap` (K=16, 64² and 32²)

Spec §2's runtime budget and §10 risk 6 are both empirical claims; this task measures them. `mempool_tile_wrap` is 127,453 components — small enough to run twice in a review cycle, large enough that `K=16` on `64²` bins really does give ~256 bins/region, which is exactly the ratio risk 6 says is unvalidated.

The checked-in `benchmarks/ispd25/mempool_tile_wrap.json` still points at the retired `/nashome/NVL4/...` paths. A host-local copy with `gpu: 1` already exists at `results/route_gp_20260914/mempool_tile_wrap.json`; promote it into `benchmarks/` so the acceptance run has a stable, checked-in input.

**This task is the single owner of that repair** (2026-09-19 reconciliation). P-B names the same dead config in its "Small test input" note but only to forbid its use — P-B runs on GCD and creates nothing under `benchmarks/`. Any later plan that needs a host-local ISPD-25 config adds it to `benchmarks/ispd25/h100/` alongside this one rather than re-deriving it.

**Files:**
- Create: `benchmarks/ispd25/h100/mempool_tile_wrap.json`
- Create: `results/p_c_producer_20260919/tile_wrap_k16_b64/{regions.json,seed.npz,membership.npz,producer.json}` (run output)
- Create: `results/p_c_producer_20260919/tile_wrap_k16_b32/{regions.json,seed.npz,membership.npz,producer.json}` (run output)
- Create: `docs/results/2026-09-19-p-c-producer-tile-wrap.md`

**Interfaces:**
- Consumes: `run_region_producer.main()`'s CLI (Task 10).
- Produces: the two artefact directories P-B's arms (b), ours, (e) will consume, and the measured runtime numbers spec §2's budget is checked against.

- [ ] **Step 1: Create the host-local config**

```bash
source src/scripts/env.sh
mkdir -p benchmarks/ispd25/h100
cp results/route_gp_20260914/mempool_tile_wrap.json \
   benchmarks/ispd25/h100/mempool_tile_wrap.json
"$IOPLACE_PYTHON" - <<'PY'
import json, os
p = "benchmarks/ispd25/h100/mempool_tile_wrap.json"
d = json.load(open(p))
assert d["gpu"] == 1, d["gpu"]
for f in d["lef_input"] + [d["def_input"]]:
    assert os.path.exists(f), f
print("ok", d["def_input"], d["global_place_stages"][0]["iteration"])
PY
```
Expected: `ok /ldaphome/yyds-tsai-dev/benchmarks/ispd25/visible/mempool_tile_wrap.def 2000`.

- [ ] **Step 2: Check the GPU is free, then run the 64² arm**

```bash
source src/scripts/env.sh
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=3
mkdir -p results/p_c_producer_20260919
"$IOPLACE_PYTHON" -m ioplace.drivers.run_region_producer \
  --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
  --k 16 --membership mtkahypar --extract-bins 64 --rect-max 8 \
  --dp-seed 1000 --deterministic 1 \
  --out results/p_c_producer_20260919/tile_wrap_k16_b64 \
  2>&1 | tee results/p_c_producer_20260919/tile_wrap_k16_b64.log
```
Expected: exit 0; the four artefacts written. If the run aborts inside `enforce_rect_max` with "no feasible notch fill remains", that is spec §10 risk 1 materialising — record the region id and rect count in the results doc and re-run with `--extract-bins 32` before escalating.

- [ ] **Step 3: Run the 32² arm (faithful GrandPlan, spec §8 arm (e))**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m ioplace.drivers.run_region_producer \
  --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
  --k 16 --membership mtkahypar --extract-bins 32 --rect-max 8 \
  --dp-seed 1000 --deterministic 1 \
  --out results/p_c_producer_20260919/tile_wrap_k16_b32 \
  2>&1 | tee results/p_c_producer_20260919/tile_wrap_k16_b32.log
```
Expected: exit 0; four artefacts written.

- [ ] **Step 4: Verify both outputs and print the runtime table**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" - <<'PY'
import os
from ioplace import artifacts
from ioplace.regions import RegionSet
from ioplace.region_grid import RegionGrid

root = "results/p_c_producer_20260919"
rows = []
for name in ("tile_wrap_k16_b64", "tile_wrap_k16_b32"):
    d = os.path.join(root, name)
    doc = artifacts.load_producer_json(os.path.join(d, "producer.json"))
    rs = RegionSet.from_json(os.path.join(d, "regions.json"))
    rs.validate()
    RegionGrid(rs)                        # also asserts a full, gapless tiling
    assert rs.k == 16 and rs.lattice == 512
    assert doc["rect_max_observed"] <= 8, doc["rects_per_region"]
    seed = artifacts.load_positions(os.path.join(d, "seed.npz"),
                                    expect_num_physical=doc["num_physical"],
                                    expect_sha256=doc["placedb_sha256"])
    mem = artifacts.load_membership(os.path.join(d, "membership.npz"),
                                    expect_num_movable=doc["num_movable"],
                                    expect_k=16)
    assert mem.source == "mtkahypar"
    xl, yl, xh, yh = seed.die
    assert seed.node_x.min() >= xl - 1e-6 and seed.node_x.max() <= xh + 1e-6
    assert seed.node_y.min() >= yl - 1e-6 and seed.node_y.max() <= yh + 1e-6
    r = doc["runtime_s"]
    rows.append((name, doc["extract_bins"], doc["n_hull_rebuilds"],
                 doc["rect_max_observed"], round(doc["area_balance"]["max_over_min"], 3),
                 round(r["gp"], 1), round(r.get("lg", 0.0), 1),
                 round(r["extract"], 1), round(r["sa"], 1),
                 round(r["rectify"], 1), round(r["total"], 1)))
hdr = ("run", "bins", "rebuilds", "max_rects", "util_max/min",
       "gp_s", "lg_s", "extract_s", "sa_s", "rectify_s", "total_s")
print(" | ".join(hdr))
for row in rows:
    print(" | ".join(str(v) for v in row))
PY
```
Expected: both rows print, no assertion fires, `max_rects ≤ 8` on both, and `sa_s < 60` (spec §2's "SA <60 s CPU").

- [ ] **Step 5: Write the results note**

Create `docs/results/2026-09-19-p-c-producer-tile-wrap.md` containing: the exact two commands, the runtime table printed in Step 4, the per-region rect counts and utilisation from both `producer.json` files, the observed `n_hull_rebuilds` and final `lambda_group`, and one explicit paragraph on spec §10 risk 6 — whether density-argmax + largest-CC extraction held up at K=16 with `64²` bins (did `ensure_nonempty` fire? check `region_bins` for any region at 1 bin) and how the `32²` map compares. State the SA time against the "<60 s CPU" budget and the producer total against "flat GP+LG +<3%" using the `gp`/`lg` phases as the flat baseline.

- [ ] **Step 6: Run the full suite one more time**

```bash
source src/scripts/env.sh && export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/ispd25/h100/mempool_tile_wrap.json \
        results/p_c_producer_20260919 \
        docs/results/2026-09-19-p-c-producer-tile-wrap.md
git commit -m "feat(producer): mempool_tile_wrap K=16 acceptance runs at 64^2 and 32^2

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement (§2 unless noted) | Task |
|---|---|
| §1 artefact table verbatim (`regions.json`, `seed.npz`, `membership.npz`, `producer.json`) | P-B Task 1 owns the I/O; verified here in 1, written in 10 |
| §1 coordinate contract, native post-read units | P-B Task 1 (docstring), 10 (the flip), test in 10 |
| Prior: `partition_netlist`, K=16, ε=0.03, or RTL hierarchy prefixes, run before the flat GP | 8, 10 |
| No block→region matching | 8 (documented and not implemented) |
| Grouping loss attached via `dp_hook.attach_terms`, Eq.1/Eq.2, `α_pull=α_push=1` | 4, 10 |
| Algorithm-1 candidate reduction on GPU, `m=16`, `q=0.90`, `α=0.25`, `K_dir=64`, ≤1024 pts/region | 2, 9 |
| Rasterised `512²` anchor tables, `K×512²×2` fp16 ≈ 16 MB at K=16 | 3 (memory assertion in the test) |
| Per-iteration cost O(N), independent of hull complexity | 4 (table lookups only) |
| Rebuild every `T_hull=50`, anchors frozen in between | 4, 10 |
| Macro pseudo points at mean std-cell pitch, ≤64/macro | 2, 10 |
| Area cap `A_max = EA_k`, centroid shrink by bisection to 1e-3 relative area | 2 |
| SA Eq.5–8 with `θ=0.05`, `50[d]³+25[d]²`, `0.1(C_ij−2)²`, `5((0.8−ρ)/0.8)²₊`, `D_diff/N_bins` | 6 |
| Min-max normalisation over the first 200 samples before `β=(1.0,0.3,0.5,0.2)` | 6 |
| `T_0` = mean `|ΔE|` of 200 probe moves, cooling 0.92, 50 moves/level, 150 levels, 3 idle levels | 6 |
| Moves: area-balancing + corner-filling 2–9 bins, equal probability under violation, corner-filling only afterwards | 6 |
| Reject any fragmenting move | 6 |
| Extraction `2048²` → `64²` majority → SA on `64²` → rects on the 512 lattice | 5, 6, 7, 10 |
| Rect cap `rect_max=8` by filling the smallest notches, re-run `validate()` (§10 risk 1) | 7 |
| §8 arm (e): faithful `32²` extraction and fixed membership; `--extract-bins {32,64}` | 8 (fixed membership), 10 (CLI), 11 (both runs) |
| §9 `tests/test_region_producer.py`: SA energies vs closed forms, moves never fragment, output passes `validate()` and `rect_max`, determinism under a fixed seed | 6 (energies, fragmentation, determinism), 7 (`validate()`, `rect_max`), 10/11 (end-to-end) |
| §10 risk 6: K=16 at `64²` bins unvalidated | 5 (`morph_open_close` restore + `ensure_nonempty`), 11 Step 5 (reported) |
| No new DREAMPlace patch | 4, 10 (`attach_terms` + `iteration_callback` only) |

**Deliberate deviations, all named in the task that makes them:**

1. **"Two table lookups"** (§2) is four in Task 4: `pull_off`, `pull_on`, `push_off`, `push_cnt`. The property the spec is buying — O(1) per cell per iteration, independent of hull vertex count — holds; splitting the indicator out of the offset avoids inferring "outside" from a zero offset, which is ambiguous for a bin exactly on the hull boundary.
2. **Push is stored as a mean plus a count**, so Eq.1's push *value* differs from the paper by a frozen additive constant while Eq.2's push *gradient* is exact (derivation in Task 3's docstring). Without this the `Σ_{s≠k}` the spec calls "the paper's silent hotspot" would be back at O(K) per cell.
3. **The discrete corner definition for `C_ij`** (Task 6) is ours; the paper defines corners only pictorially. The rule is pinned by three unit tests (straight → 0, L-turn → 1, one-bin notch → 4).
4. **Corner-filling's "corner detected" test** is "the window carries ≥2 labels" (Task 6). A strict corner test is undefined for a 1×2 window, which digest §5 explicitly allows; `E_boundary` does the real corner accounting and rejects moves that make things worse.
5. **Morphology uses the full `3×3` square structuring element**, not the 4-connected cross (Task 5, verified on this host: the cross strips every rectangle's corners during opening; the square does not).
6. **λ for the grouping term lives in a local `GroupingWeight`**, but its ramp and EMA are P-H's `norm.grandplan_weight` / `norm.ema_update` (spec §4), so Eq.3's schedule has exactly one implementation. P-C therefore depends on `src/ioplace/norm.py` existing — it does on this branch, and as of 2026-09-19 it is committed, with `ema_update(prev, inst, ema=0.5)` and `grandplan_weight(iteration, it_activate, wt0, wt_step, ramp_period, wt_max)` matching P-H Task 1's Interfaces block character-for-character, but `TermNormalizer` not yet. Task 4 says to stop and report rather than reimplement if it is absent, and states the one-line migration to `TermNormalizer.register("group", ...)`.

**Known gap, deliberate:** spec §2's runtime targets ("group ≤25 min, cluster ≤60 min") are not exercised here. Task 11 measures `mempool_tile_wrap` only; the group-scale numbers are produced by the joint P-B+P-C acceptance (§9 "Done": "the 2×2 on `mempool_group`"), which is P-B's plan. Task 11 Step 5 states the tile-wrap SA time against the "<60 s CPU" budget and the producer overhead against "flat GP+LG +<3%" so the extrapolation is at least anchored.

### 2. Placeholder scan

Searched the plan for `TBD`, `TODO`, `implement later`, `fill in details`, `Similar to Task`, `add appropriate`, `handle edge cases`, and bare "write tests for the above": no hits. Every test step carries runnable test code and every implementation step carries the full module or the exact block to append.

### 3. Type consistency

Cross-checked every name a later task uses against the task that defines it:

- `hull.reduce_candidates` / `reduce_candidates_torch` / `convex_hull` / `polygon_area` / `shrink_to_area` / `macro_pseudo_points` / `build_hull` / `nearest_on_polygon_boundary` / `anchor_tables` / `AnchorTables` — defined in Tasks 2, 3, 9; used in Tasks 3, 4, 9, 10.
- `AnchorTables` fields `lattice`, `die`, `pull_off`, `pull_on`, `push_off`, `push_cnt`, property `k` — defined in Task 3, read in Task 4's `_lookup` and Task 4's `set_tables` assert.
- `GroupingTerm.set_tables` / `.tables` / `.grad_l1` / `.n_rebuilds`; `GroupingWeight.wt` / `.update` / `.lam` / `.ratio_ema` — defined in Task 4, used in Task 10.
- `extract.extract(..., out_bins=)` and `extract.FINE_BINS` — Task 5, used in Task 10.
- `sa.SaConfig(seed=)` and `sa.anneal(labels0, k, ea, bin_area, cfg)` returning `(labels, report)` with keys `t0`, `levels_run`, `e_raw_initial`, `e_raw_final`, `beta` — Task 6, used in Tasks 6's tests and 10.
- `rectify.enforce_rect_max(labels, k, rect_max=)`, `rectify.rects_to_regionset(labels, k, die, lattice=)`, `rectify.region_rect_counts(labels, k)`, `rectify.mask_to_rects(mask)` — Task 7, used in Task 10 and Task 11's verifier.
- `membership.build_membership(source, *, nl, node_names, num_movable, k, epsilon, seed, depth, threads)` — Task 8, called in Task 10 without `threads` (default 8, itself overridden by `IOPLACE_MTKAHYPAR_THREADS=1`).
- `artifacts.save_positions(path, node_x, node_y, *, die, shift_factor, scale_factor, placedb_sha256, kind)`, `artifacts.save_membership(path, part, *, source, k, seed, epsilon)`, `artifacts.placedb_identity_sha256(placedb)` and `artifacts.save_producer_json(path, payload)` — all **P-B Task 1**, all called by keyword in Task 10; `load_positions`/`load_membership` return the `Positions`/`Membership` dataclasses Task 11's verifier reads by attribute, and `load_producer_json` validates the same `PRODUCER_FIELDS` that Task 10's payload fills (checked key-for-key by Task 1 Step 1).
- Existing-code borrowings verified against source: `run_placement._load_dreamplace`, `run_placement.extract_final_positions`, `dp_hook.attach_terms` / `assert_optimizer_lock` / `refresh_nesterov_secant`, `profile.PhaseTimer` (`timer.phases[name]["t_s"]`, as `run_placement._phase_summary` reads it), `profile.env_metadata(repo_root, dp_root, input_paths=...)`, `ioplace.paths.REPO_ROOT`, `regions.RegionSet`/`RegionSpec`/`validate`/`to_json`/`from_json`, `region_grid.RegionGrid`, `partition.mtkahypar_runner.partition_netlist(nl, k, epsilon, seed, threads)`.
- Host API assumptions verified by probe on 2026-09-19: `scipy.spatial.QhullError` importable (scipy 1.17.1); `ndimage.generate_binary_structure(2,2)` is the full `3×3` and opening a rectangle with it is a no-op while the cross is not; `torch.quantile` on a strided slice matches `np.quantile`; `params.shift_factor`/`params.scale_factor` are set by `PlaceDB.initialize` (`simple.json`: die `(459,459,555,555)` → `(0,0,96,96)`, shift `(459,459)`, scale `1.0`); `placedb.node_names` is an array of `np.bytes_`.

