# Round Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FLUTE evaluation between complete GP rounds produce feedback that measurably improves final GR-accepted placement, with checkpoint rollback and evidence-driven feedback revision when it does not.

**Architecture:** Separate geometric measurement, immutable feedback policy, differentiable GP terms, and the outer round controller. GR owns acceptance; provisional search and retained observations cannot overwrite the accepted placement. Representative DR follows the final GR campaign.

**Tech Stack:** Python/NumPy, PyTorch CPU/CUDA, native C++/OpenMP FLUTE, DREAMPlace, OpenROAD, pytest.

**Spec:** [Approved design](../specs/2026-09-15-round-feedback-design.md), including the added long-horizon outcome requirement in Section 12.

## Global Constraints

- A round is a complete configured GP invocation; no evaluator/GR/DR call inside its iteration loop.
- Feedback coefficients are frozen within a round; topology refreshes remain optimizer-step-boundary operations.
- GR interval 3; maximum GP rounds 12; stop after 2 consecutive rejected checkpoints. All are configurable positive integers.
- Candidate IO must strictly improve on the accepted placement; native total overflow must not increase.
- `100 * candidate_wirelength_dbu <= 101 * initial_wirelength_dbu`; no accumulating allowance or extra tolerance.
- Preserve coordinates, orientations, pin identity/offsets, fixed nodes, original baseline, and accepted resource provenance.
- Discard candidate optimizer state on recovery; retain valid rejected GR price observations, then re-evaluate the restored placement.
- Native FLUTE, accuracy 3, scale 1000, modeled degree 2..256. No silent MST fallback or undisclosed higher-degree omission.
- Feedback defaults: damping 0.25, gain 1, multiplier cap 8. These are revisable policies; acceptance gates are not.
- IO-boundary loads are baseline-normalized on the IO lattice. Capacity-normalized prices use the actual GR resource lattice.
- A changed gradient is an integration gate, not the outcome gate. Long-horizon paired GR improvement is required; revise feedback when it fails.
- Run GR first; run baseline/selected DR only for at most 3 final representative benchmark/seed pairs.
- Keep the package compatible with its declared Python >=3.9; use the existing H100 runtime described in `docs/dev-env.md` for verification.
- Preserve existing dirty source migration, experimental files, and index entries. Never commit unrelated staged work.

## Execution preparation and evidence

The working tree, not HEAD alone, contains the current driver, FLUTE adapter,
and source-layout migration. Before execution, apply using-git-worktrees: detect
existing isolation and preserve this working source snapshot when isolating.
A worktree created only from HEAD is not an equivalent execution baseline.
Record a binary patch of staged and unstaged tracked files plus an explicit
manifest/archive of required untracked source/test files before transferring
the snapshot. Do not copy result corpora, virtual environments, or running jobs.

Parent graph evidence: Verify Tier 2; project
`ldaphome-yyds-tsai-dev-IO-Aware-Top-Level-Placer`, generation
`2026-09-15T17:27:31Z`. Exact `project.src.*` and `project.tests.*` queries were
fully paginated for the driver, route/IO/FT terms, FLUTE batch, tensor geometry,
and selected tests. Coverage checks recorded no gaps for the cited code paths.
Some earlier graph import traces resolved archived result snapshots, and line
metadata differed from current numbered source. Use exact working source for
wiring and recheck coverage/freshness before editing or delegation.

Known integration facts:

- `run_route_gp.main` currently invokes one placer; its nested `online` callback routes and assimilates during GP.
- `FrozenJointRouteCost.components` already has per-net IO weights, congestion, resource prices, and wirelength.
- `FrozenJointRouteCost.rebuild` recreates the Boolean IO-boundary mask and publishes observation arrays; coefficients need their own persistent payload.
- `_FtFn` is the bounded S4a-star surrogate, not exact physical feed-through. Its forward and both backward coefficient passes must receive the same per-net FT weights.
- `oriented_netlist` supplies ordinary row-flip pin transforms; it must always transform canonical offsets, never already transformed offsets.
- GR directional `max_*` fields are sums of layer maxima; only native `total_overflow` defines the overflow acceptance gate.
- Checked usable configs: `results/route_feedback_20260914/gcd.json`, `results/route_gp_20260914/mempool_tile_wrap.json`, and `results/route_gp_20260914/mempool_group.json` had existing DEF/LEF inputs during planning. Revalidate at execution.
- Legacy `benchmarks/ispd25/*.json` have stale absolute paths. `benchmarks/ispd2005_bigblue4_m4.json` is Bookshelf-only and cannot substitute for physical GR evidence without an independently validated technology conversion.

All shell test commands below run after:

```bash
source src/scripts/env.sh
```

Before CUDA/real placement checks, inspect `nvidia-smi`, preserve any existing
`CUDA_VISIBLE_DEVICES`, and use an available authorized H100. Do not select a
busy device merely because an old report used it. Capture the baseline suite
once before edits; record pre-existing failures separately.

## Files and task dependencies

| Task | Responsibility | Depends on |
| --- | --- | --- |
| 1 | Immutable records and exact GR gate | none |
| 2 | Streamed FLUTE segment adapter | 1 |
| 3 | GPU FLUTE measurements and two-grid outputs | 1, 2 |
| 4 | Pure feedback updates and generation validation | 1 |
| 5 | Frozen GP IO/FT/spatial penalties | 1, 4 |
| 6 | Exact GR receipt, resources, and placement identity | 1 |
| 7 | Round controller, checkpoint persistence, and rollback | 1, 4, 6 |
| 8 | Real GP runner and driver wiring | 2, 3, 5, 7 |
| 9 | Campaign reporting and representative DR selection | 1, 7 |
| 10 | Numerical and real closed-loop integration gates | 3, 5, 6, 8, 9 |
| 11 | Long-horizon outcomes, feedback revision, final DR | 10 |

Tasks share one cohesive round-feedback feature; the pure modules can be
reviewed independently, but runtime integration follows their contracts.
Subagents, if chosen for execution, own only their assigned task files and
must not revert others' edits. Review shared-file changes sequentially.

### Task 1: Immutable records and exact GR acceptance

**Files:** Create `src/ioplace/feedback/__init__.py`, `types.py`, `gate.py`;
create `tests/test_round_gate.py`, `tests/test_round_types.py`.

**Interfaces:** `types.py` defines the following records for all later tasks.
Array-valued constructors defensively copy, validate, then mark NumPy arrays
read-only. Frozen dataclasses alone are insufficient. Hashes include shapes,
dtypes, ordered IDs, and bytes; reject NaN/infinite coordinates before hashing.

| Record | Required fields |
| --- | --- |
| `RunKey` | string hashes `design`, `net_cohort`, `regions`, `resource_grid`, `router`, `routing_policy`, `capacity_policy`; integers `seed`, `dbu_per_micron` |
| `PlacementSnapshot` | physical `node_x`, `node_y`; tuple of movable `orientations`; canonical identity hash; oriented `pin_offset_x`, `pin_offset_y`; computed `sha256` |
| `ResourceState` | `capacity`, `background`; `grid_sha256`, `cohort_sha256`, `source_placement_sha256`, `source_receipt_sha256`; `approximate_background: bool` |
| `FastMetrics` | full-net integer arrays `io`, `ft`, `terminal_regions`, `home`; Boolean `modeled`; float arrays `boundary_load`, `resource_demand`; scalar `tree_wl`, `hpwl`; `placement_sha256`, `cohort_sha256`, `io_grid_sha256`, `resource_grid_sha256`, `backend`; `omissions`, `timings` mappings |
| `RoundFeedback` | integer `generation`; full-net integer `home`; float arrays `io_weights`, `ft_weights`, `boundary_weights`, `fast_prices`, `gr_prices`; scalar `route_lambda`, `ft_strength`; `resources: ResourceState`; `placement_sha256`, `cohort_sha256`, `io_grid_sha256`; `policy_version` |
| `GRMetrics` | optional integer `io`, `wirelength_dbu`, `total_overflow` (all required when complete); `key: RunKey`; `placement_sha256`; Boolean `complete`, `legal`, `coverage_ok`; optional `failure_reason` |
| `GRCheckpoint` | `placement: PlacementSnapshot`, `metrics: GRMetrics`, `resources: ResourceState`; `gr_price_signal` array; immutable `receipt_path` |
| `GateDecision` | Boolean `accepted`; ordered tuple `reasons`; Boolean `quality_comparison_complete` |

`gate.accept_candidate(baseline: GRMetrics, incumbent: GRMetrics,
candidate: GRMetrics) -> GateDecision` checks identity/validity before quality.
Reasons are stable strings: `incomplete`, `illegal`, `coverage`, `identity`,
`io_not_improved`, `wirelength_budget`, `overflow_regression`.
Incomplete records carry `None` for unmeasured quantities, never invented zero
measurements. Validity checks must return before arithmetic on those records.

- [ ] Add an independently testable gate fixture and failing predicate tests:

```python
from dataclasses import replace
from ioplace.feedback.types import GRMetrics, RunKey
from ioplace.feedback.gate import accept_candidate

def record(io, wl, overflow):
    key = RunKey("design", "nets", "regions", "grid", "router", "policy",
                 "capacity", 1000, 2000)
    return GRMetrics(io, wl, overflow, key, "placement", True, True, True, None)

def test_allowance_is_anchored_to_initial_baseline():
    base = record(100, 10000, 5)
    incumbent = record(90, 10100, 5)
    assert accept_candidate(base, incumbent, record(89, 10100, 5)).accepted
    assert not accept_candidate(base, incumbent, record(89, 10101, 5)).accepted
    assert not accept_candidate(base, incumbent, record(90, 10000, 5)).accepted
    assert not accept_candidate(base, incumbent, record(89, 10000, 6)).accepted
    changed = replace(record(89, 10000, 5), key=replace(base.key, seed=1001))
    assert "identity" in accept_candidate(base, incumbent, changed).reasons
```

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_round_gate.py tests/test_round_types.py`; confirm the missing API is the failure.
- [ ] Implement records and the integer-only quality core after validity checks:

```python
reasons = []
if candidate.io >= incumbent.io:
    reasons.append("io_not_improved")
if 100 * candidate.wirelength_dbu > 101 * baseline.wirelength_dbu:
    reasons.append("wirelength_budget")
if candidate.total_overflow > incumbent.total_overflow:
    reasons.append("overflow_regression")
return GateDecision(not reasons, tuple(reasons), True)
```

- [ ] Reject negative counts, booleans used as counts, nonintegral counts, bad units, and wrong array lengths before the core. Add mutation tests: changing a source array after record construction does not alter the record; writes through the record fail. Include counts above `2**53` to prove no float budget comparison.
- [ ] Run the targeted tests to green. Commit only this task's files with `feat: define round feedback records and exact GR gate` after reviewing their diff.

### Task 2: Streamed geometric FLUTE backend

**Files:** Create `src/ioplace/evaluator_flute.py`, `tests/test_evaluator_flute.py`;
reuse `src/ioplace/route_eval/topology_batch.py`, `topology.py` without copying FLUTE.

**Interfaces:** `SegmentBatch(net_ids, axes, tracks, low, high)` stores equal-length
one-dimensional arrays, `axes` 0 for horizontal and 1 for vertical.
`iter_flute_segments(pins, starts, net_ids, *, pin_budget=1000000,
accuracy=3, coordinate_scale=1000., threads=8)` yields complete-net batches.
Inputs are packed physical pins in CSR order and global net IDs of degree 2..256.
`segments_from_raw(tree, pins, starts, net_ids) -> SegmentBatch` converts one raw
batch. `segment_union_reference(batch) -> SegmentBatch` is a scalar CPU oracle.

- [ ] Add a real three-pin Steiner test; its known length distinguishes it from MST:

```python
import numpy as np
from ioplace.evaluator_flute import iter_flute_segments, segment_union_reference

def test_real_steiner_tree_has_length_four():
    pins = np.array([[1., 1.], [1., 3.], [3., 2.]])
    batches = list(iter_flute_segments(pins, np.array([0, 3]), np.array([7])))
    assert len(batches) == 1
    union = segment_union_reference(batches[0])
    assert float((union.high - union.low).sum()) == 4.
    assert set(union.net_ids.tolist()) == {7}
```

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_evaluator_flute.py`; confirm red.
- [ ] Build batches by cumulative pin count without splitting a net. Call `batch_flute_trees` once per batch; convert global parents and terminal IDs without losing zero-length identities. The nonzero branch expansion is:

```python
a = positions[branch_ids]
b = positions[parents[branch_ids]]
elbow = np.column_stack((b[:, 0], a[:, 1]))
horizontal_low = np.minimum(a[:, 0], elbow[:, 0])
horizontal_high = np.maximum(a[:, 0], elbow[:, 0])
vertical_low = np.minimum(elbow[:, 1], b[:, 1])
vertical_high = np.maximum(elbow[:, 1], b[:, 1])
```

Emit axis/track rows for both legs, omit zero-length rows, and append exact-pin
to snapped-terminal stubs using the same H-then-V convention. Preserve terminal
mapping separately from physical segment rows. Parents must belong to their
net's branch range. Fail with the global net ID on native or geometry errors.

- [ ] Implement scalar union grouped by `(net_id, axis, track)`: sort intervals by low/high, merge overlapping or touching intervals, preserve different nets. Test reversed duplicate branches and two nets sharing the same geometry.
- [ ] Add duplicate/coincident pins, half-quantization ties, negative coordinates after translation, degree 256/257, int32 FLUTE limits, empty input, and a pin budget smaller than one eligible net. The last must yield that one net and report the exceptional batch size, not loop forever.
- [ ] Run new tests plus `tests/test_flute_batch.py` and `tests/test_flute_topology.py`. Verify serial/parallel and batching produce identical union geometry. Commit `feat: stream FLUTE geometry for placement evaluation`.

### Task 3: GPU FLUTE metrics and separate IO/resource grids

**Files:** Modify `src/ioplace/evaluator_gpu.py`; create
`src/ioplace/evaluator_flute_gpu.py`, `tests/test_evaluator_flute_gpu.py`;
extend `tests/test_evaluator_gpu.py` only for legacy-backend regression coverage.

**Interfaces:** Extend `GpuEvalContext(..., backend="mst", resource_grid=None,
flute_accuracy=3, flute_scale=1000., flute_threads=8)` without changing legacy
`evaluate(x, y)` callers. The new driver explicitly requires `backend="flute"`.
Add `evaluate_feedback(snapshot: PlacementSnapshot) -> FastMetrics`; fail if
the context backend is not FLUTE. `set_pin_offsets(x, y)` refreshes only oriented
offset tensors and identity before measurement. Put FLUTE work arrays in the
helper module rather than adding another monolithic method.

- [ ] Copy the small `Netlist` construction pattern from `tests/test_route_gp.py::fixture` into a local three-pin fixture using Task 2's coordinates. Use two x-regions over `(0,0,4,4)` and a distinct nonuniform GR grid.

Concrete `flute_case` fixture in the new test module:

```python
import numpy as np
import pytest
from ioplace.netlist import Netlist
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from ioplace.route_eval.joint import ResourceGrid
from ioplace.feedback.types import PlacementSnapshot
from ioplace.evaluator_gpu import GpuEvalContext

@pytest.fixture
def flute_case():
    points = np.array([[1., 1.], [1., 3.], [3., 2.]])
    ids = np.arange(3, dtype=np.int64)
    nl = Netlist(points[:, 0], points[:, 1], np.full(3, .1), np.full(3, .1),
                 3, 0, 0, np.zeros(3), np.zeros(3), ids, np.zeros(3, dtype=np.int64),
                 ids, np.array([0, 3]), 0, 0, 4, 4)
    rg = RegionGrid(make_grid_regions((0, 0, 4, 4), 2, 1, lattice=4))
    grid = ResourceGrid(np.array([0., .5, 2.5, 4.]), np.array([0., 2., 4.]))
    snapshot = PlacementSnapshot(points[:, 0], points[:, 1], ("N", "N", "N"),
                                 "tiny-canonical-netlist", np.zeros(3), np.zeros(3))
    return nl, rg, grid, snapshot
```
- [ ] Add a failing test against hand-computed metrics, including omission metadata:

```python
def test_flute_measurement_has_separate_spatial_grids(flute_case):
    nl, rg, grid, snapshot = flute_case
    context = GpuEvalContext(nl, rg, device="cpu", backend="flute", resource_grid=grid)
    result = context.evaluate_feedback(snapshot)
    assert result.backend == "flute"
    assert result.tree_wl == 4.
    assert result.io.tolist() == [1]
    assert result.ft.tolist() == [0]
    assert result.terminal_regions.tolist() == [2]
    assert result.boundary_load.shape != result.resource_demand.shape
    assert result.omissions["higher_degree_net_count"] == 0
```

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_evaluator_flute_gpu.py`; confirm red.
- [ ] On each complete-net batch, move segment arrays to the selected device and use `route_tensor.union_intervals`. Use hard `grid_crossings` for each grid independently. Keep IO crossing events at distinct physical tracks; derive boundary load by unique `(net, IO-edge)` and resource demand by unique `(net, resource-edge)`:

```python
key = events["net_ids"].to(torch.int64) * edge_count + events["edge_ids"]
unique_key = torch.unique(key)
edge_ids = unique_key.remainder(edge_count)
load.index_add_(0, edge_ids, torch.ones_like(edge_ids, dtype=load.dtype))
```

Validate packed-key overflow before multiplication. Because batches contain
complete nets, integer accumulators can be added without cross-batch duplicate
leakage. Mask non-boundary IO edges only on the IO grid. Preserve physical IO
crossing counts separately from deduplicated boundary/resource occupancy.

- [ ] Reuse bounded visited-region processing for FT, including geometry entirely inside one region. Compute `visited_regions AND NOT terminal_regions`, then popcount. Preserve deterministic home-region ties. Keep modeled tree totals separate from higher-degree lower bounds; include exact omitted global IDs.
- [ ] Add an independent scalar reference that enumerates regions, crossings, and resource keys from Task 2's union. Cover boundary endpoint ownership (`low < cut <= high`), zero-length contact, two tracks of one net on one coarse resource, shared nets, fillers, fixed pins, and orientation flips.
- [ ] Run CPU reference parity first, then CUDA parity and batch invariance. Integer metrics must match exactly; compare float64 tree length with relative tolerance `1e-12`. Measure transfer/build/aggregation times and peak memory on the existing bigblue4 evaluator-only data if available; mark it evaluator evidence, not GR evidence.
- [ ] Bound incidence expansion as well as pin count: estimate hard crossing
  counts before allocation and cap emitted incidences per chunk at 1000000 by
  default. If one net needs multiple chunks, retain its unique occupancy keys
  until that net is complete, then accumulate once. Test a long two-pin net on
  a fine grid and a many-net batch; changing the event budget must not change
  integer metrics or hide a full-size intermediate allocation.
- [ ] Run `tests/test_evaluator_gpu.py`, new evaluator tests, and `tests/test_route_tensor.py`. Commit `feat: aggregate FLUTE IO and resource feedback on GPU`.

### Task 4: Pure, versioned feedback updates

**Files:** Create `src/ioplace/feedback/policy.py`, `tests/test_round_policy.py`.

**Interfaces:** `FeedbackPolicy(eta=.25, alpha=1., cap=8., version="ratio-v1")`;
`update(baseline: FastMetrics, current: FastMetrics, previous: RoundFeedback,
resources: ResourceState, gr_prices, *, generation: int) -> RoundFeedback`;
`observe_gr(previous_prices, signal) -> np.ndarray` updates measured prices
only. `neutral_feedback(current, resources, *, generation, route_lambda,
ft_strength) -> RoundFeedback` creates identity multipliers and zero prices.
Keep `route_lambda`/`ft_strength` fixed from the supplied previous payload.

- [ ] Add a local `metrics(io, ft, regions, boundary, demand)` fixture builder
  returning full valid `FastMetrics` with consistent synthetic identity hashes.
  Use it to test zero baselines, omitted-net masking, and exact damping:

```python
import numpy as np
import pytest
from ioplace.feedback.types import FastMetrics, ResourceState
from ioplace.feedback.policy import FeedbackPolicy, neutral_feedback

def metrics(io, ft, regions, boundary, demand, placement="base"):
    count = len(io)
    return FastMetrics(
        io=np.array(io, dtype=np.int64), ft=np.array(ft, dtype=np.int64),
        terminal_regions=np.array(regions, dtype=np.int64),
        home=np.zeros(count, dtype=np.int64), modeled=np.ones(count, dtype=bool),
        boundary_load=np.array(boundary, dtype=float),
        resource_demand=np.array(demand, dtype=float), tree_wl=0., hpwl=0.,
        placement_sha256=placement, cohort_sha256="nets", io_grid_sha256="io-grid",
        resource_grid_sha256="resource-grid", backend="flute", omissions={}, timings={})

@pytest.fixture
def policy_case():
    base = metrics([1, 0], [0, 0], [2, 1], [0, 4], [0, 0, 0])
    current = metrics([3, 0], [1, 0], [2, 1], [2, 4], [2, 0, 4], "candidate")
    resources = ResourceState(np.array([1., 0., 2.]), np.zeros(3), "resource-grid",
                              "nets", "base", "baseline-receipt", True)
    previous = neutral_feedback(base, resources, generation=0,
                                route_lambda=.1, ft_strength=1.)
    return base, current, previous, resources
```

```python
def test_zero_boundary_baseline_is_finite_and_damped(policy_case):
    base, current, previous, resources = policy_case
    result = FeedbackPolicy().update(base, current, previous, resources,
                                     previous.gr_prices, generation=1)
    # Fixture: baseline boundary=[0,4], current=[2,4], previous=[1,1].
    np.testing.assert_allclose(result.boundary_weights, [1.5, 1.])
    assert result.generation == 1
    assert result.placement_sha256 == current.placement_sha256
    assert np.isfinite(result.fast_prices).all()
```

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_round_policy.py`; confirm red.
- [ ] Implement a shared bounded multiplier update and the approved formulas:

```python
def damped_multiplier(previous, signal, *, eta, alpha, cap):
    target = np.clip(1. + alpha * signal, 1., cap)
    return (1. - eta) * previous + eta * target

excess = np.maximum(current.io - np.maximum(current.terminal_regions - 1, 0), 0)
base_excess = np.maximum(baseline.io - np.maximum(baseline.terminal_regions - 1, 0), 0)
io_signal = excess / np.maximum(base_excess, 1)
ft_signal = current.ft / np.maximum(baseline.ft, 1)
boundary_signal = np.maximum(current.boundary_load - baseline.boundary_load, 0) / np.maximum(baseline.boundary_load, 1)
```

- [ ] Compute fast prices only where capacity is positive; blocked edges carry
  a separate mask through the resource contract. Their utilization entry is
  absent from ratios, not divided by epsilon. Apply prices as damped targets
  clipped to `[0,8]`; do not add the multiplier's leading one to a price.
- [ ] Reject invalid eta/gain/cap, mismatched cohorts/grids, non-monotonic generations, NaN/negative demand, and current metrics from the wrong placement. Do not require cached background's source placement to equal current placement; require it to be explicitly marked approximate and linked to an accepted GR receipt.
- [ ] Test per-net and boundary signals independently, and prove updating fast prices leaves supplied GR prices byte-identical. Run targeted tests and commit `feat: derive immutable per-round feedback from measurements`.

### Task 5: Consume frozen feedback in actual GP gradients

**Files:** Modify `src/ioplace/ops/route_gp.py`, `io_term.py`, `ft_term.py`,
`routing_gp_controller.py`; create `src/ioplace/ops/round_objective.py`,
`tests/test_round_objective.py`; extend existing FT reference tests.

**Interfaces:** Add `FrozenJointRouteCost.set_round_feedback(payload)` callable
only before a round is sealed. Add `seal_round()`/`finish_round()` to reject
in-round mutation. Add `FtTerm.set_net_weights(weights)` and the same method to
`FtTermRef`; weights have shape `(io_term.n_active,)` in CSR active-net order.
`RoundObjective(route_term, ft_term, *, ft_strength)` exposes
`components(pos, *, tau, gamma)` and `__call__(pos, *, tau, gamma)`; delegate
`paper`, `rebuild`, and generation metadata to the wrapped route term for the
existing controller. Add optional `fixed_route_lambda` to
`RoutingGPController`; legacy callers retain their existing calibration path.

- [ ] Add weighted-gradient and mutation tests using the existing small route fixture. Define the new `weighted_case` fixture locally: construct `IoTerm` with two rectangular regions, `build_net_node_csr(nl, 257)`, and the matching region distance matrix; wrap it in `FtTerm` and then `RoundObjective`. Use at least three regions for a nonzero S4a-star FT contribution.

```python
def test_payload_is_immutable_through_topology_rebuild(weighted_case):
    objective, payload, pos = weighted_case
    objective.route_term.set_round_feedback(payload)
    objective.route_term.seal_round()
    before = objective.route_term.feedback_sha256
    objective.rebuild(pos)
    objective.rebuild(pos.detach() + 0.001)
    assert objective.route_term.feedback_sha256 == before
    with pytest.raises(RuntimeError, match="sealed"):
        objective.route_term.set_round_feedback(payload)
```

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_round_objective.py`; confirm missing APIs fail.
- [ ] Keep physical masks and payload weights separate in route components:

```python
event_net = io_events["net_ids"]
event_edge = io_events["edge_ids"]
physical_io = io_events["values"] * self.boundaries[event_edge]
io = (physical_io * self.io_weights[event_net]
      * self.round_io_weights[event_net]
      * self.round_boundary_weights[event_edge]).sum()
price = ((self.round_fast_prices + self.round_gr_prices)
         * (demand - self.background)).sum()
```

`io_raw` remains unweighted. In round mode, do not also add legacy assimilated
edge prices/calibration on top of the same payload. Preserve payload arrays
through `rebuild`, which may change geometry only. Use explicit all-degree-2..256
net IDs for route feedback rather than DREAMPlace's `ignore_net_degree=100`
wirelength mask. Report each term's actual active-net coverage.

- [ ] Extend `_FtFn` with a separately cloned `ft_weights` tensor. In forward,
  accumulate `(ft_weights[:, None] * a * q).sum()`. In **both** backward passes,
  use the same weighted coefficient:

```python
coeff = io_coeff.unsqueeze(1) + ctx.ft_scale * ft_weights.unsqueeze(1) * a
```

Save the FT weights used in forward for backward, just as IO weights are saved.
Return two coordinate gradients plus exactly the number of `None` entries
required by `len(ctx.needs_input_grad) - 2`. Add identical mathematical weighting
to the independent dense `FtTermRef`; do not call production code from it.
With FT weights all one, reproduce existing output/gradient. With
`kappa_ft == 0`, preserve direct `IoTerm` dispatch.

- [ ] Compose `route_term(...) + ft_strength * ft_term.ft_only(pos, tau)`;
  do not call `FtTerm.forward` and double-count its soft IO term. Gather
  `payload.ft_weights[csr.net_ids]`; the CSR's deduplication of pins to cells
  makes S4a-star a documented surrogate, not a physical FT identity. Refresh
  its `home` from `payload.home` before publication, then freeze it. Register
  `IoTerm.active_net_ids` from `csr.net_ids` as a device int64 buffer so the
  composite objective can perform this gather without assuming contiguous
  global net IDs. `RoundObjective` owns the route/FT seal lifecycle, and exposes
  the active payload's `feedback_sha256` for runtime assertions; sealed FT
  weight/home setters must reject mutation too.
- [ ] In fixed-lambda round mode, `_step` must not call `_calibrate`. Preserve
  `fixed_route_lambda` across every topology rebuild. Set `router_every=0` and
  reject a non-null router callback in this mode. Include FT value/gradient,
  feedback generation/hash, and fixed lambda in trace/audit output.
- [ ] Test finite differences away from geometry ties, IO-only and FT-only
  weighting, zeroed fixed/filler gradients, source-array mutation after forward,
  CPU/CUDA parity, and unchanged coefficients across a line search. Run
  `tests/test_route_gp.py`, `tests/test_route_tensor.py`, new tests, and the
  `tests/test_ft_term.py` and `tests/test_ft_callback.py`. Commit
  `feat: apply frozen round feedback to IO and FT gradients`.

### Task 6: Exact GR measurements and trustworthy resource snapshots

**Files:** Modify `src/ioplace/route_eval/online_openroad.py`,
`or_scripts/dump_online_route.py`; create `src/ioplace/feedback/gr_adapter.py`,
`tests/test_round_gr_adapter.py`; extend `tests/test_bounded_grt_feedback.py`.

**Interfaces:** `checkpoint_from_observation(observation, snapshot, key,
modeled_net_ids) -> GRCheckpoint`. Extend observation with integer
`actual_wirelength_dbu`, `dbu_per_micron`, capacity extraction diagnostics, and
per-layer resources/usage sufficient to audit aggregation. Preserve legacy
float `actual_wirelength` as a report field. Invalid observations return a
failed checkpoint only when a valid incumbent/key exists; baseline failure
raises a typed `BaselineRoutingError` defined in `gr_adapter.py`.

- [ ] Add integer segment-union tests independent of coordinate scaling:

```python
from ioplace.feedback.gr_adapter import layer_union_length_dbu

def test_wirelength_unions_within_net_and_layer_only():
    segments = [(0, "M2", 0, 0, 100, 0), (0, "M2", 50, 0, 150, 0),
                (0, "M3", 0, 0, 100, 0), (1, "M2", 0, 0, 100, 0)]
    assert layer_union_length_dbu(segments) == 350
```

`layer_union_length_dbu(segments) -> int` accepts tuples shown above and uses
Python integer interval unions grouped by net/layer/axis/track. Ignore
zero-lateral vias for planar wirelength; reject non-Manhattan wire segments.

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_round_gr_adapter.py`; confirm red.
- [ ] Parse raw integer DBU segments before converting to placement coordinates.
  Require receipt exit code 0, hashed output integrity, immutable net names,
  full expected routed-net audit, and native `total_overflow`. Derive expected
  nets from the router's signal-net manifest, not `len(nets)` alone. Include
  same-location and high-degree signal nets according to that audit contract.
- [ ] Preserve per-layer capacity/usage extraction before 2-D aggregation.
  Record when the current uint8 API is used, and count per-layer values at the
  representational maximum **before** summing layers. Conservatively reject
  saturated/ambiguous extraction for feedback; never infer saturation from a
  perfectly valid aggregate above 255. Add stub-grid tests with a single layer
  at 255 and two clean layers whose sum exceeds 255.
- [ ] Build background only by subtracting measured modeled-net occupancy with
  layer multiplicities from measured usage. Validate compatible semantics and
  nonnegative residuals before aggregation. Do not subtract 2-D deduplicated
  FLUTE demand from a layer-multiplicity GR measurement. Keep raw observations
  and derived background in separate immutable records.
- [ ] A rejected checkpoint can contribute a measured congestion-price signal,
  but its derived background is never installed on the incumbent. Test a case
  where an unmodeled high-degree net moves in the rejected candidate: the
  restored background must retain the incumbent receipt/hash.
- [ ] Run new tests plus bounded-GR tests. Preserve legacy directional-max names
  with explicit semantics, never use them in acceptance. Commit
  `feat: validate exact GR checkpoints and resource provenance`.

### Task 7: Outer round state machine and durable recovery

**Files:** Create `src/ioplace/feedback/controller.py`, `store.py`,
`tests/test_round_controller.py`, `tests/test_round_store.py`.

**Interfaces:** `RoundConfig(interval=3, max_rounds=12, max_rejections=2)`;
`RoundServices` is a dataclass of callables:

- `prepare(snapshot) -> PlacementSnapshot`: construct/normalize the next GP
  model input without running GP; expose any dtype conversion in its hash.
- `run_gp(snapshot, payload, round_index) -> PlacementSnapshot`: a complete GP
  call with fresh optimizer and no routing/evaluator callbacks.
- `evaluate(snapshot) -> FastMetrics`.
- `legalize(snapshot, checkpoint_index) -> PlacementSnapshot`.
- `route(snapshot, checkpoint_index) -> GRCheckpoint`.

`AcceptedState(checkpoint, metrics, feedback_seed)` and
`RunState(baseline, round_index, rejection_streak, accepted, provisional, feedback,
gr_prices, history, quality_complete)` are defined in `controller.py`.
`baseline` is the immutable initial `AcceptedState`, including both its GR
checkpoint and fast metrics. It is never reassigned when a candidate passes.
`RoundController(config, services, policy, store).run(initial_state) -> RunState`.
`CheckpointStore(root).save(state)`, `.load(expected_key)`, and
`.record_event(event)` provide durable state; serialization is explicit JSON/NPZ
with `allow_pickle=False`, never arbitrary pickled Python objects.

- [ ] Define a fake `RoundServices` in the new test module. Each function logs
  its phase; `run_gp` returns a new known coordinate array, and `route` consumes
  a scripted list of real `GRCheckpoint` records from Task 1. Assert:

```python
def test_two_rejections_restore_incumbent_with_new_feedback(scripted_run):
    controller, initial, calls = scripted_run(rejected_checkpoints=2)
    result = controller.run(initial)
    assert result.round_index == 6
    assert result.rejection_streak == 2
    assert result.accepted.checkpoint.placement.sha256 == initial.accepted.checkpoint.placement.sha256
    assert [row["round"] for row in result.history if row["kind"] == "checkpoint"] == [3, 6]
    assert any(row["kind"] == "restore_evaluation" for row in result.history)
    assert result.feedback.resources.source_receipt_sha256 == initial.accepted.feedback_seed.resources.source_receipt_sha256
```

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_round_controller.py tests/test_round_store.py`; confirm red.
- [ ] Implement the loop with an explicit last-round checkpoint:

```python
checkpoint_due = (round_index % config.interval == 0
                  or round_index == config.max_rounds)
if checkpoint_due:
    legal = services.legalize(provisional, round_index)
    legal_metrics = services.evaluate(legal)
    checkpoint = services.route(legal, round_index)
    decision = accept_candidate(baseline.checkpoint.metrics, accepted.checkpoint.metrics,
                                checkpoint.metrics)
```

If legalization leaves the exact snapshot unchanged, reuse its fast result;
otherwise evaluate again. A failed legalization is a failed checkpoint and
never calls GR with an invalid snapshot. Catch declared infrastructure errors,
record failure metadata, and retain the incumbent; do not swallow arbitrary
programming exceptions as quality rejection.

- [ ] On acceptance, install candidate checkpoint/metrics/feedback seed and
  reset streak. On rejection, update retained prices only from valid complete
  observations, restore incumbent geometry/resource state, prepare it, and
  evaluate that actual next-round model input before deriving feedback from the
  **accepted** seed. Increment generation even when the placement is restored.
  Reset optimizer via `prepare`, not by copying only its position tensor.
- [ ] Cache GR by placement/orientation/pin hashes plus full `RunKey` and effort.
  Reusing the accepted placement is not a fresh rejection. A newly routed
  candidate that fails strict IO improvement is a rejection even if its score
  ties. Stop at the configured horizon independently of rejection streak.
- [ ] Implement atomic store publication: write uniquely named JSON/NPZ
  artifacts, flush and fsync, calculate hashes, then replace a manifest pointer
  with `os.replace`. Fsync the parent directory. Keep the previous accepted
  manifest until the new one is durable. Record launch identity, command/log
  paths, process/session ID, exit code and timing for external jobs.
- [ ] Inject interruption before each publication stage. Resume only verified
  artifacts with a matching run key; an incomplete GR receipt stays incomplete.
  Test 3/6/9/12 scheduling, max_rounds=4 final checkpoint, streak reset, unchanged
  final cache hit, missing metrics, and orientation/background recovery. Commit
  `feat: orchestrate GP rounds with atomic GR rollback`.

### Task 8: Real GP lifecycle, orientation-correct warm starts, and CLI

**Files:** Create `src/ioplace/feedback/gp_runner.py`; modify
`src/scripts/run_route_gp.py`, `tests/test_routing_gp_driver.py` for explicit
legacy selection, and `src/ioplace/ops/placement_offsets.py` only if
the existing row-flip helper requires validation changes; create
`tests/test_round_gp_runner.py`, `tests/test_round_gp_driver.py`.

**Interfaces:** `DreamPlaceRoundRunner(config_path, output_root, *, seed,
gp_iterations, topology_rebuild, round_route_lambda=.1, ft_strength=1.)`
provides `prepare`, `run_gp`, `legalize`, `route`, and `evaluate` matching Task 7.
It also provides `initialize(initial_placement=None) -> RunState` and `close()`.
Bind OpenROAD binary, LEFs, region grid, and routing policy through validated
runner configuration, not globals inside the controller.

- [ ] Add subprocess CLI tests that reject missing OpenROAD, nonpositive
  round limits, non-FLUTE backend in round mode, and mismatched checkpoint/final
  effort before creating an output directory. Include a fake runner test that
  checks no optional router callback is supplied to the GP controller.
- [ ] Extract only the required loading, snapshot export/repair, and full-GP
  lifecycle from the existing driver. Reuse `_load_dreamplace`, `export_def`,
  `repair_routed_snapshot`, `oriented_netlist`, and Task 6's adapter. Keep the
  export database separate from the GP database. Do not turn the new controller
  into another closure-heavy driver.
- [ ] In `prepare`, use fresh params/placer/optimizer. Preserve fixed nodes;
  regenerate fillers; disable random-center reinitialization and added position
  noise for a warm-started candidate. Transform canonical pin offsets once with
  the supplied ordinary-row orientations before constructing GP pin operators.
  Update both GP and route/FT model offsets consistently. Unsupported rotations
  fail explicitly. Record dtype-converted starting geometry and re-evaluate it
  outside GP before publishing its payload; keep the original accepted GR
  snapshot separate from this model representation.
- [ ] Implement one full `run_gp` with scoped term attachment and cleanup:

```python
attach_terms(params, [controller])
placer.iteration_callback = controller.callback
try:
    placer(params, db, params.global_place_stages[0]["learning_rate"])
finally:
    placer.iteration_callback = None
    detach_terms(params)
    if controller.original_step is not None:
        controller.optimizer.step = controller.original_step
```

Here `controller` is the Task 5 fixed-lambda controller wrapping
`RoundObjective`; the outer controller is a different object. Suppress automatic
LG inside candidate GP rounds and run LG only through the checkpoint service.
Capture output physical positions before destroying GP buffers. Do not call
`GpuEvalContext.evaluate_feedback` from `legalize_op` or any iteration callback.

- [ ] Add `--workflow {round-feedback,legacy}` to the existing command; default
  new joint runs to round-feedback, and preserve explicit legacy behavior for
  historical reproduction. Add `--gr-interval 3 --max-rounds 12
  --max-rejections 2 --feedback-policy ratio-v1 --feedback-eta .25
  --feedback-alpha 1 --feedback-cap 8 --round-route-lambda .1
  --ft-strength 1 --feedback {on,off} --resume --initial-state PATH`.
  Keep existing `--iterations` as iterations per complete GP round, record
  effective convergence settings, and use one declared GR acceptance effort.
  `--mode` still selects the GP objective; require an explicit legacy workflow
  for old in-loop `--router-every` behavior.
- [ ] Feedback-off holds the baseline frozen payload and measured prices fixed;
  it still runs the same GP objective, GR schedule, acceptance and stop rules.
  Use identical initial baseline artifacts for paired arms. Do not accidentally
  compare feedback-on joint GP to an unrelated WA-only objective.
- [ ] Initialization without supplied placement runs feedback-disabled GP once,
  then LG/GR and baseline FLUTE evaluation. Count this cost separately. With a
  supplied baseline, verify paired source/config/technology identity rather than
  trusting its filename. Require genuine GR resources; prohibit the legacy
  `np.ones(grid.edge_count)` fallback in round mode.
- [ ] Test a row-flipped cell with an asymmetric pin: accepted DEF pin geometry,
  evaluator pin geometry, and the next GP starting pin positions must agree
  under the recorded coordinate transform. Test fresh optimizer IDs per round,
  released GPU buffers, fixed fillers, and a callback that raises if evaluation
  or routing occurs inside the GP phase. Run legacy tests explicitly with
  `--workflow legacy`; commit `feat: run full GP rounds with external feedback`.

### Task 9: Trajectory reports and representative DR selection

**Files:** Create `src/ioplace/feedback/reporting.py`,
`src/scripts/run_round_feedback_campaign.py`,
`src/scripts/run_round_feedback_dr.py`,
`tests/test_round_reporting.py`, `tests/test_round_dr_selection.py`.

**Interfaces:** `CampaignRow` fields: `benchmark_id`, `seed`, `physical_cells`,
`arm`, `policy_version`, `split`, `complete`, `baseline: GRMetrics`,
`selected: GRMetrics`, `result_dir`, `costs`, `trajectory`.
`select_dr_cases(rows: Sequence[CampaignRow]) -> list[CampaignRow]` considers
only complete feedback-on rows for the final frozen policy, then selects
highest initial overflow, lower-median physical size, and worst relative IO
gain. Stable tie order is benchmark ID then seed; deduplicate pairs.
`paired_quality(rows) -> dict` returns paired gains and gate status without
silently dropping unsuccessful, early-stopped, or incomplete runs.

- [ ] Add deterministic selection tests, including overlapping categories:

```python
def test_selection_deduplicates_and_retains_no_improvement(campaign_rows):
    selected = select_dr_cases(campaign_rows)
    keys = [(r.benchmark_id, r.seed) for r in selected]
    assert keys == [("congested", 1002), ("median", 1002), ("no_gain", 1002)]
    assert len(keys) == len(set(keys)) <= 3
```

The fixture has physical sizes 10/20/30, highest overflow on `congested`, and
zero relative improvement only on `no_gain`; assign `median` size 20. Also test
all roles selecting one case and exact tie ordering.

- [ ] Run `"$IOPLACE_PYTHON" -m pytest -q tests/test_round_reporting.py tests/test_round_dr_selection.py`; confirm red.
- [ ] Implement ranking with integer/rational gains where possible. Emit
  per-round raw GP and legal checkpoint hashes, fast IO/FT/WL, native GR metrics,
  decision reasons, payload hashes, rejection streak, and GP/evaluation/GR
  durations. Report early-stop outcomes rather than extrapolating missing later
  metrics as new measurements. Produce CSV/JSON plus static trajectory figures
  from matplotlib; no AI-generated scientific charts.

Core selector implementation:

```python
from fractions import Fraction

def select_dr_cases(rows):
    rows = sorted((r for r in rows if r.complete and r.arm == "on" and r.split == "validation"),
                  key=lambda r: (r.benchmark_id, r.seed))
    if not rows:
        return []
    highest = min(rows, key=lambda r: (-r.baseline.total_overflow,
                                      r.benchmark_id, r.seed))
    by_size = sorted(rows, key=lambda r: (r.physical_cells, r.benchmark_id, r.seed))
    median = by_size[(len(by_size) - 1) // 2]
    worst = min(rows, key=lambda r: (
        Fraction(r.baseline.io - r.selected.io, max(r.baseline.io, 1)),
        r.benchmark_id, r.seed))
    selected = {}
    for row in (highest, median, worst):
        selected.setdefault((row.benchmark_id, row.seed), row)
    return list(selected.values())
```

The caller first validates that all candidate rows belong to the same frozen
final policy; mixed-policy input fails instead of cherry-picking a policy per
case. The final selection uses completed validation rows, not calibration or
smoke cases from exploratory runs.

- [ ] Campaign CLI accepts `--manifest PATH --split calibration|validation
  --policy PATH --out PATH --openroad PATH`; manifest rows define config paths,
  seeds, resource requirements, and immutable input hashes. Dispatch paired arms
  with identical baseline/config/effort and bounded concurrency. Snapshot
  generated effective configs instead of editing source benchmark configs.
- [ ] DR CLI accepts `--selection PATH --out PATH --openroad PATH`, validates
  selection hashes, and calls existing `run_detailed_route(export_dir, lefs,
  out, binary, threads=8)` for each matched baseline/selected pair. Reuse
  identical placements under identical DR policy. It must not run the entire
  GR manifest. Preserve DRC, unrouted-net and native-WL diagnostics separately
  from process exit status. Commit `feat: report round trajectories and bounded DR validation`.

### Task 10: Numerical and real closed-loop integration verification

**Files:** Create `tests/test_round_integration.py`; extend
`tests/test_round_objective.py`, `tests/test_round_gp_driver.py`;
record `docs/reports/2026-09-15-round-feedback-integration.md` and immutable
artifacts under `results/round_feedback_20260915/integration/`.

**Interfaces:** No new policy APIs. Validate Task 1–9 contracts together using
the real driver and actual FLUTE/GP/GR adapters. Reports must distinguish unit
tests, mocked controller tests, and real process measurements.

- [ ] Add a finite-difference helper to the objective tests. Freeze topology
  and payload before perturbations; use float64 and points away from geometric
  ties. Compare each IO/FT/boundary/resource component separately:

```python
def central_difference(value, position, index, epsilon=1e-5):
    plus = position.detach().clone()
    minus = position.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    return (float(value(plus)) - float(value(minus))) / (2 * epsilon)

def test_weighted_component_derivative(weighted_component):
    value, position, movable_index = weighted_component
    derivative, = torch.autograd.grad(value(position), position)
    expected = central_difference(value, position, movable_index)
    assert float(derivative[movable_index]) == pytest.approx(expected, rel=1e-4, abs=1e-6)
```

Define `weighted_component` from Task 5's three-region fixture, parameterized
over each named component; do not use a component whose gradient is identically
zero at the chosen test point. Add explicit zero-gradient fixtures separately.

- [ ] Run focused CPU tests, then corresponding CUDA tests. Exercise actual
  `PlaceObj` gradient summation and Nesterov secant-cache refresh after topology
  changes with a constant payload. Test no cross-round graph reuse or retained
  references to the previous optimizer.
- [ ] Run real GCD integration with small iteration counts solely to prove
  wiring, outside-loop scheduling, and artifact identity:

```bash
"$IOPLACE_PYTHON" src/scripts/run_route_gp.py \
  --workflow round-feedback --mode joint \
  --config results/route_feedback_20260914/gcd.json \
  --out results/round_feedback_20260915/integration/on \
  --openroad "$OPENROAD_BIN" --iterations 8 --start 2 --rebuild 3 \
  --gr-interval 3 --max-rounds 6 --max-rejections 2 --feedback on \
  --feedback-grt-iterations 50 --final-grt-iterations 50
```

Run the paired off arm with `--feedback off` and a distinct output directory,
sharing the saved initial baseline. Task 8 adds `--initial-state PATH` to load
that verified baseline-only manifest, not the on-arm's selected state. Check
real call phase ordering, generation hashes, orientation-aware measurement,
GR at rounds 3 and 6 unless the recorded policy stops earlier, and no router
or evaluator call inside a GP iteration.

- [ ] Add a controlled GP fixture where a known different payload produces a
  different derivative and later placement. A real case with no displacement
  change fails this integration gate unless its expected zero-signal condition
  is explicitly tested separately. Do not require GCD's routed outcome to
  improve; empirical improvement belongs to Task 11.
- [ ] Run required regression checks with the configured environment:

```bash
"$IOPLACE_PYTHON" -m pytest -q tests/test_round_types.py tests/test_round_gate.py \
  tests/test_evaluator_flute.py tests/test_evaluator_flute_gpu.py \
  tests/test_round_policy.py tests/test_round_objective.py \
  tests/test_round_gr_adapter.py tests/test_round_controller.py \
  tests/test_round_store.py tests/test_round_reporting.py \
  tests/test_round_dr_selection.py tests/test_round_gp_runner.py
"$IOPLACE_PYTHON" -m pytest -q tests/test_round_gp_driver.py tests/test_round_integration.py
"$IOPLACE_PYTHON" -m pytest -q
```

Inspect exit codes and skipped-test reasons. Missing physical inputs or an
unavailable GPU do not count as a passed real integration gate. Compare any
failures with the recorded pre-edit baseline; do not revert unrelated work to
make the suite green.
- [ ] Record exact commands, source/config hashes, test results, peak memory,
  timing breakdown, and remaining gaps. Commit tests/report after inspection
  with `test: verify full GP round feedback and recovery`.

### Task 11: Long-horizon outcome campaign and feedback revision

**Files:** Create `benchmarks/round_feedback_campaign.json`,
`benchmarks/round_feedback_policies/ratio-v1.json`,
`docs/reports/2026-09-15-round-feedback-outcomes.md`;
extend `src/ioplace/feedback/policy.py` and its tests when diagnostics justify
revisions. Store immutable artifacts under `results/round_feedback_20260915/`.

**Interfaces:** Task 9's manifest schema is concrete JSON. Each case has
`benchmark_id`, `config`, `split`, `seeds`, and `gp_iterations`. Each policy JSON
has `version`, `method`, `eta`, `alpha`, `cap`, `round_route_lambda`, and
`ft_strength`. `method="ratio"` selects Task 4's initial update. Add a policy
factory `load_policy(path) -> FeedbackPolicy` that validates method/version;
an unknown method is an error, not fallback to ratio.

- [ ] Freeze this initial manifest after validating and hashing all referenced
  inputs. Preserve effective configs inside each run; do not edit historical
  result configs in place:

```json
{
  "gr_interval": 3,
  "max_rounds": 12,
  "max_rejections": 2,
  "gr_iterations": 50,
  "cases": [
    {"benchmark_id": "gcd", "config": "results/route_feedback_20260914/gcd.json", "split": "calibration", "seeds": [1000, 1001], "gp_iterations": 500},
    {"benchmark_id": "mempool_tile_wrap", "config": "results/route_gp_20260914/mempool_tile_wrap.json", "split": "calibration", "seeds": [1000, 1001], "gp_iterations": 2000},
    {"benchmark_id": "mempool_tile_wrap", "config": "results/route_gp_20260914/mempool_tile_wrap.json", "split": "validation", "seeds": [1002, 1003, 1004], "gp_iterations": 2000},
    {"benchmark_id": "mempool_group", "config": "results/route_gp_20260914/mempool_group.json", "split": "validation", "seeds": [1002, 1003, 1004], "gp_iterations": 2000}
  ]
}
```

The validation split holds out seeds on both physical cases and initially
holds out the group design. It does not claim both designs are unseen. Record
actual physical-cell counts; an assumed benchmark name is not scale evidence.
If referenced inputs disappear, retain the missing-case gate and resolve the
data source rather than silently substituting GCD.

- [ ] Freeze the initial policy file:

```json
{"version":"ratio-v1","method":"ratio","eta":0.25,"alpha":1.0,"cap":8.0,"round_route_lambda":0.1,"ft_strength":1.0}
```

Before seeing validation results, add report tests for this **predeclared
outcome gate**: all 6 validation pairs are complete; every selected placement
satisfies its GR acceptance constraints; each design's median paired IO gain
is positive; at least 4 of 6 pairs have lower selected IO with feedback on
than off. Paired gain is `(IO_off - IO_on) / max(IO_off, 1)`. This is evidence
on the tested cases, not a statistical claim of universal improvement. Baseline
fallback is a valid safe output, but ties/fallbacks do not count as improvement.

Add a reporting regression test whose six pairs both improve on their common
baseline by the same amount. It must still fail the feedback benefit gate:

```python
def test_equal_on_off_gain_is_not_feedback_improvement(equal_paired_rows):
    report = paired_quality(equal_paired_rows)
    assert report["complete_pairs"] == 6
    assert report["positive_pairs"] == 0
    assert not report["outcome_passed"]
```

The local fixture constructs 12 `CampaignRow` records: tile/group, seeds
1002/1003/1004, on/off. Each has baseline IO 100 and selected IO 90, wirelength
10000 DBU and zero overflow. Each pair shares a `RunKey`; case/seed keys remain
distinct. Also test an incomplete pair and a baseline fallback: neither may be
deleted from the denominator to improve the outcome verdict.

- [ ] Run calibration first, inspect full trajectories, and freeze the chosen
  policy before validation. Launch with Task 9's CLI:

```bash
"$IOPLACE_PYTHON" src/scripts/run_round_feedback_campaign.py \
  --manifest benchmarks/round_feedback_campaign.json --split calibration \
  --policy benchmarks/round_feedback_policies/ratio-v1.json \
  --out results/round_feedback_20260915/ratio-v1/calibration \
  --openroad "$OPENROAD_BIN"
"$IOPLACE_PYTHON" src/scripts/run_round_feedback_campaign.py \
  --manifest benchmarks/round_feedback_campaign.json --split validation \
  --policy benchmarks/round_feedback_policies/ratio-v1.json \
  --out results/round_feedback_20260915/ratio-v1/validation \
  --openroad "$OPENROAD_BIN"
```

These are potentially long jobs. Use supported completion notification to the
same session when available, and verify it can resume the session before
claiming it is configured. Otherwise retain one durable handoff with actual
process/session IDs, logs, job manifest, and outstanding gates. Do not create
an LLM poller, repeatedly wait on unchanged logs, or replace a healthy run.

- [ ] Compare accepted trajectories at rounds 3/6/9/12 where measured, final
  selected states, actual GP iterations, and wall-time cost. Keep early stops
  as outcomes. Show raw candidates separately so rollback does not hide
  predictor failure. Compare fast IO/FT/demand to GR observations on matched
  positions/cohorts; do not correlate different snapshots.
- [ ] If the outcome gate fails, classify the failure before changing policy:

| Evidence | Authorized response |
| --- | --- |
| Payload changes but its gradient contribution is absent | Fix adapter/coverage/publication; rerun Task 10 before more campaigns |
| Fast improvements repeatedly become GR regressions | Calibrate fast signals with matched GR residuals; keep fast and measured state separate |
| Weight/price oscillation and alternating rejection | Reduce damping/gain or use a bounded trust step; retain rejection information |
| Penalties saturate while placement stagnates | Replace ratio targets with bounded projected-dual updates; inspect relative IO/FT/resource gradient magnitudes |
| Rollback restores stale geometry/background | Fix recovery before tuning formulas |
| Quality gain disappears at matched cost | Report the cost trade-off and revise computation; coefficient changes alone are not success |

For a projected-dual candidate, add `ProjectedDualPolicy` with the same
`update`/`observe_gr` interface and versioned `method="projected-dual"` JSON.
An initial bounded update is:

```python
def projected_update(previous, violation, *, eta, decay, lower, upper):
    return np.clip(previous + eta * violation - decay * (previous - lower),
                   lower, upper)
```

Use lower 1 for multipliers and 0 for prices; retain the cap and independent
grids. Define violations from Task 4's normalized per-net/spatial signals.
Add zero-signal, persistent-violation, saturation, decay, and recovery tests
before evaluating the new policy. This is an alternative to investigate when
diagnostics support it, not a prediction that it will outperform ratio feedback.

- [ ] For every revision, record hypothesis, changed formula/parameters,
  triggering cases, tests, and source hash. Run calibration, freeze, then use
  the next unused seed triplet (1005–1007, then 1008–1010) for validation.
  Previously inspected group results are no longer an unseen-design holdout;
  label later evidence as fresh-seed validation on known designs. Keep all
  attempted policies and negative results. Never weaken the 1% budget, native
  overflow gate, or strict incumbent IO criterion to manufacture success.
- [ ] Continue diagnosis/revision while authorized resources allow meaningful
  progress. If a resource limit prevents a full campaign, leave the outcome
  gate unmet and record its state; do not declare placement optimization
  achieved. Failed initial feedback is not a terminal deliverable when another
  justified revision remains possible.
- [ ] Once the final frozen policy passes the GR outcome gate, persist Task 9's
  representative selection from that campaign and run matched DR pairs. Report
  IO/WL/DRC/completeness without routing every exploratory policy. Negative DR
  remains a reported limitation, not evidence to suppress or signoff feasibility.
- [ ] Re-run affected tests after revisions and the required suite before
  completion. Write baseline/on/off trajectories, all policy versions, resource
  costs, GR verdict, selected DR results, and the scope of evidence into the
  outcome report. Commit final policy/tests/report only after inspecting actual
  results. Writing this plan does not complete any empirical task checkbox.

## Coverage and completion checklist

| Spec requirement | Implementation/verification |
| --- | --- |
| Complete GP rounds; evaluator outside iterations | Tasks 7, 8, 10 |
| Actual FLUTE, terminal identity, shared geometry | Tasks 2, 3, 10 |
| Bounded host/GPU intermediates and separate grids | Tasks 2, 3, 6 |
| Per-net IO/FT plus boundary/resource feedback | Tasks 4, 5 |
| Frozen coefficients with GP topology refresh | Tasks 5, 8, 10 |
| Exact GR IO/WL/overflow and identity | Tasks 1, 6 |
| Baseline initialization and separate costs | Tasks 7, 8, 9 |
| Checkpoints, final checks, stop limits, cache reuse | Tasks 1, 7, 10 |
| Orientation/offset and recovery provenance | Tasks 1, 6, 7, 8 |
| Retain rejected observations without stale background | Tasks 4, 6, 7 |
| Durable resume and background work handoff | Tasks 7, 9, 11 |
| Numerical, CPU/CUDA, and real GP verification | Tasks 3, 5, 10 |
| Long-horizon improvement; revise ineffective feedback | Task 11 |
| GR-first diagnostics and representative DR | Tasks 9, 11 |

Software correctness, demonstrated GR improvement, and representative DR
results are separate completion gates. Report each explicitly. This plan does
not claim any execution gate has already passed.
