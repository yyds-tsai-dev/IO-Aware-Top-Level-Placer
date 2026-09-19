# v2 Subproject P-B — Main Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v2 main-flow driver `src/ioplace/drivers/run_main_flow.py` — soft-assign GP → freeze → fence-region GP → fence LG → evaluator — as two DREAMPlace instances communicating through validated on-disk artefacts, emitting a `result.json` whose IO accounting identity closes.

**Architecture:** Four phases, two DREAMPlace instances. The split is forced: fence data may only be injected between `placedb.read()` and `placedb.initialize()` (`src/ioplace/fence_inject.py:9-16`), so the fence GP cannot reuse phase 1's `PlaceDB`. Phase 1 runs the existing soft `IoTerm`/`FtTerm` machinery until a freeze criterion fires, writes `soft.npz` + `freeze.json` + `frozen_membership.npz`, and stops. Phase 3 rebuilds a fresh `PlaceDB`, injects the frozen membership as hard fences, warm-starts from `soft.npz`, clamps the re-derived density weight, and runs GP with the IO and FT terms **off**. Phase 4 is DREAMPlace's automatic multi-fence legalisation, then the GPU evaluator. Every cross-phase array is written in native post-read PlaceDB units, so any phase can be re-run from artefacts alone (`--phase {all,soft,fence}`).

**Tech Stack:** Python 3.12, torch 2.8.0+cu128, DREAMPlace 4.3.1 (`$DREAMPLACE_ROOT/install`), numpy, pytest. No new DREAMPlace patch — `m2-extra-obj-terms.patch` and `iteration-callback.patch` already supply every hook this plan needs.

**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` (§1 data contracts, §3 main flow, §7 driver-side straddling diagnostics `io_delta_at_freeze` / `fence_compliance`, §8 arm definitions, §9 tests and the B+C "done" criterion). Read the spec before starting; this plan argues from it and the two travel together.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Run protocol.** From the repo root `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer`, branch `v2/redesign`:

```bash
source src/scripts/env.sh
export IOPLACE_MTKAHYPAR_THREADS=1 CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```

`src/scripts/env.sh` exports `DREAMPLACE_ROOT=/ldaphome/yyds-tsai-dev/DREAMPlace` and `IOPLACE_PYTHON=$DREAMPLACE_ROOT/.venv312/bin/python`. Use `-m "not slow"` while iterating; run the full suite before declaring a task done. This is a shared H100 NVL host — check `nvidia-smi` and honour `CUDA_VISIBLE_DEVICES` before any GPU work.

**No new DREAMPlace patch.** DREAMPlace source is off-limits. The two existing patches (`src/ioplace/dp_patch/m2-extra-obj-terms.patch`, `src/ioplace/dp_patch/iteration-callback.patch`) plus `src/ioplace/dp_patch/shapely2-compat.patch` are the only modifications. Everything else is driver-side: `params`-borne term attachment (`src/ioplace/dp_hook.py:7-10`), the per-iteration callback (`NonLinearPlace.py:521-523`), instance-attribute monkeypatches installed and restored through `_install_attribute` (`src/ioplace/drivers/run_placement_io.py:110-119`).

**Coordinate contract: native post-read PlaceDB units.** Every cross-phase array artefact (`seed.npz`, `soft.npz`, `regions.json`, `membership.npz`) is expressed in the coordinate system of `placedb` **after `read(params)` and before `initialize(params)`**. `initialize()` calls `PlaceDB.scale()` (`$DREAMPLACE_ROOT/install/dreamplace/PlaceDB.py:151-196`), which rescales `node_x`/`node_y`, `node_size_x`/`node_size_y`, the die box, `regions` and `flat_region_boxes` together, with `params.shift_factor = (xl, yl)` and `params.scale_factor = 1/site_width` fixed at `PlaceDB.py:759-767`. Conversions: `scaled = (native - shift_factor) * scale_factor`, `native = scaled / scale_factor + shift_factor`. Evaluator artefacts (`evaluation.npz`) stay in scaled evaluator units and record `shift_factor`/`scale_factor`, exactly as `src/ioplace/export/evaluation.py:68-117` already does. This is the same contract P-C Task 10 states as its "coordinate flip": the producer works in scaled units *inside* the GP and converts back to native before writing `regions.json`/`seed.npz`, which is exactly the frame `init_pos.apply_init` (Task 2) and `fence_phase.build_fence_placedb` (Task 4) write into `placedb.node_x`/`node_y` after `read()` and before `initialize()`.

**Artefact schema (spec §1 table, verbatim):**

| File | Payload | Producer / reader |
|---|---|---|
| `regions.json` | `RegionSet` (die, lattice, per-region rect list) | `src/ioplace/regions.py:44` `to_json`/`from_json`; the format `run_route_gp.py` already reads |
| `seed.npz` | `node_x`,`node_y` over `num_physical`, native units, `die`, `shift_factor`, `scale_factor`, `placedb_sha256` | producer → warm start |
| `membership.npz` | `part` int32 per movable node, `{source,k,seed,epsilon}` | producer → arm (e) fences, soft-phase prior |
| `capacity.npz` | segment table (§5), `capacity`, `capacity_source`, OpenROAD receipt hash | extractor → GP + evaluator |
| `evaluation.npz` | existing `save_evaluation` schema (`export/evaluation.py:66`) + per-segment arrays | evaluator → reports |
| `norm_trace.jsonl` | one row per normalisation probe (§4) | main flow |

P-B produces `soft.npz` (a `seed.npz`-schema file with `kind="soft"`), `frozen_membership.npz` (a `membership.npz`-schema file with `source="freeze"`), `freeze.json`, `placement.npz`, `evaluation.npz`, `norm_trace.jsonl` and `result.json`. `capacity.npz` is P-D's; do not implement it.

**Terms after the freeze are OFF.** Phase 3 runs WL + density + fence only. Capacity (P-D) and pseudo-FT (P-E) attach at the documented hook point `run_fence_gp(..., extra_terms=())`; do **not** implement them here.

**Two independent drivers.** `src/ioplace/drivers/run_placement_io.py` stays as the legacy single-phase driver and must not be modified. `run_main_flow.py` is a fork that imports the small shared helpers rather than duplicating them.

**Retired-path gating.** `IOPLACE_ENABLE_GR_IN_LOOP=1` is required to import `src/scripts/run_route_gp.py` or `src/ioplace/ops/routing_gp_controller.py` (Task 8). **Stated deviation from spec §1 (controller ruling C-8, 2026-09-19).** Spec §1 lists five retired modules; this plan gates only those two. `ops/route_gp.py`, `ops/joint_route_feedback.py` and `ops/route_feedback.py` stay ungated: they are import-side-effect-free libraries that live unit tests exercise directly, so an import-time gate there would fail those tests at collection for no safety gain. Accepted and recorded; revisit only if a v2 driver ever imports them.

**`schema_version` asymmetry (controller ruling C-2, 2026-09-19).** `save_producer_json` stamps `PRODUCER_SCHEMA_VERSION` itself and `PRODUCER_FIELDS` does not list it; `save_freeze`/`save_result` require the *caller* to supply `schema_version`, and `FREEZE_FIELDS`/`MAIN_FLOW_RESULT_FIELDS` do list it. This is deliberate, not an oversight: the producer (P-C Task 10) builds its whole payload as one dict literal that it also returns to its caller, whereas the main flow's `result` dict is assembled by spreading `_t8a_provenance`, which carries `run_placement`'s own `RESULT_SCHEMA_VERSION` and would clobber any writer-side stamp (see Task 7's `result["schema_version"] = ...` restore). Cross-plan ruling 3 was revised to match (`.superpowers/sdd/v2-cross-plan-rulings.md`); do not "unify" this without editing P-C Task 1's 48-name `PRODUCER_FIELDS` assertion in the same commit.

**Commit trailer.** Every commit message in this plan ends with:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

**Acceptance (B+C exit, spec §9).** The 2×2 arm matrix on `mempool_group` completes — all four arms finishing GP+LG+evaluator with a shape-vs-seed decomposition reported. Acceptance is *completion*, not a quality threshold. That campaign is run after this plan lands; the automatable part of it is Task 9's end-to-end test.

---

## File Structure

**Reconciliation note (2026-09-19).** P-B and P-C were written in parallel and
both defined `src/ioplace/artifacts.py`. **P-B Task 1 is the single owner; P-C
consumes it.** Merged into Task 1 below: P-C's `placedb_fingerprint` becomes
`placedb_identity_sha256` (its digest now also covers `pin2node_map` and
`pin2net_map`, so it is strictly stronger than either original); P-C's
`save_seed`/`load_seed` become `save_positions`/`load_positions` with
`kind="seed"`; P-C's positional `save_membership`/`load_membership` become this
plan's keyword-only pair; and P-C's `save_producer_json` moves here, gaining
`PRODUCER_SCHEMA_VERSION`, a `PRODUCER_FIELDS` contract and a
`load_producer_json` reader. P-C's Task 1 shrinks to a verification step and
every artefact test lives in `tests/test_artifacts.py` here. Two further seams
were reconciled: **Task 8** *relocates* P-H Task 8's `IOPLACE_ENABLE_GR_IN_LOOP`
check into `src/ioplace/gr_in_loop.py` instead of adding a second gate, and the
dead `/nashome/NVL4` benchmark config is repaired exactly once, by **P-C Task
11** (`benchmarks/ispd25/h100/mempool_tile_wrap.json`) — P-B stays on GCD and
does not create or edit anything under `benchmarks/`. The coordinate contract
below was checked against P-C Task 10's scaled-inside/native-outside flip and
needed no change.

**Pre-flight amendments (2026-09-19).** The pre-flight conflict scan
(`.superpowers/sdd/2026-09-19-v2-p-b-main-flow/preflight.md`, tables A–E) and the
controller rulings recorded at the end of
`.superpowers/sdd/2026-09-19-v2-p-b-main-flow/progress.md` changed this plan in the
places listed below; task numbering is unchanged and nothing else moved.

- **Task 5 (B-1 / A-10 / A-11)** now ships *both* `LegacyNormAdapter` and
  `TermNormalizerAdapter` in `src/ioplace/norm_adapter.py`. The old text made
  `--norm-policy grandplan|adaptive` a permanent `NotImplementedError` and
  attributed `TermNormalizerAdapter` to P-H's plan, which never mentions
  `norm_adapter.py`; cross-plan ruling 2's literal form ("build on
  `TermNormalizer` directly, no adapter") is unimplementable because
  `TermNormalizer` owns no τ schedule and no activation trigger. Ruling 2 was
  revised to match. Without this, spec §4's policy-A/B ablation is unreachable
  from `run_main_flow`.
- **Tasks 5 / 7 / 9 (A-9)** — `norm_trace.jsonl` now carries exactly one schema,
  P-H's `norm_trace.ROW_FIELDS`, written only through
  `norm_trace.NormTraceWriter` (rows emitted by `TermNormalizer.mark_refreshed`).
  The legacy path writes `legacy_trace.jsonl` with `publish_atomic`'s own keys.
  Task 9's row assertions branch on the policy.
- **Task 7 (C-6)** — `_resolve_regions` rejects a `regions.json` whose `k`
  disagrees with `--k`.
- **Tasks 6 / 7 / 9 (D-1)** — `run_fence_gp` returns `io_fence_gp_source`
  (`"legalize_op"` | `"fallback"`), `MAIN_FLOW_RESULT_FIELDS` carries it, and
  Task 9 asserts it instead of two tautologies: `io_identity_residual` is 0 by
  construction and can never detect a stale measurement.
- **Task 8 (A-12)** — the claim that P-H Task 8 modifies no existing test module
  is false; its Files list includes `tests/test_routing_gp_driver.py:41`.
- **Tasks 1 / 3 / 4 / 9 (E-3)** — an empty real region crashes `initialize()`
  with `IndexError` at `PlaceDB.py:687` (numpy 1.26.4 `np.percentile` on an
  empty slice), not `ValueError` at `:729`.
- **Smaller amendments** — D-2 (one region-stats implementation, owned by
  `freeze.py`, called with native-unit sizes from both sites), D-3 (dead
  `sampler` parameters dropped), D-5 (`gp_iteration_budget` reaches
  `result.json`), D-6 (`soft_summary` provenance), D-7 (the prior-remap netlist
  is built through a zero-arg factory, only when it is used), D-8
  (`--argmax-chunk`, default 4), D-9, D-10, D-12, D-13, D-14, D-4's docstring
  note, and Task 3's Interfaces line.
- **Recorded in Global Constraints** — C-8's stated deviation from spec §1's
  five-module retirement list, and C-2's deliberate `schema_version` asymmetry.

Deferred to implementation review by ruling, *not* gaps: C-4 (the clamp band
against phase 3's `(K+1,)` density-weight vector — spec §3's own open question
(iii)), D-11, A-24, C-13.

**New modules**

| File | Responsibility |
|---|---|
| `src/ioplace/artifacts.py` | Artefact I/O and the coordinate contract for **both** v2 drivers: `seed.npz`/`soft.npz`, `membership.npz`, `freeze.json`, `producer.json`, `result.json` readers/writers with schema validation; `placedb_identity_sha256`; native↔scaled `RegionSet` conversion. No DREAMPlace import. P-C (region producer) imports these names and adds nothing of its own. |
| `src/ioplace/init_pos.py` | The three initial-position modes (`die_center`, `region_center`, `seed`) written into `placedb.node_x/node_y` between `read()` and `initialize()`. |
| `src/ioplace/freeze.py` | Cell-centre region argmax, membership-churn tracking, the three-part freeze criterion, empty-region repair, per-region cell/area statistics, the `freeze.json` record. |
| `src/ioplace/fence_phase.py` | Phase-3 PlaceDB construction: fence injection, escape-cell workaround, warm start, and the density-weight clamp (including its scoped `PlaceObj.initialize_density_weight` wrapper). |
| `src/ioplace/norm_adapter.py` | The single seam between the driver and normalisation policy, with one protocol and two implementations: `LegacyNormAdapter` (`schedules.ScheduleState` + `ops/ft_callback.publish_atomic`, writing `legacy_trace.jsonl`) and `TermNormalizerAdapter` (P-H's `norm.TermNormalizer` for `grandplan`/`adaptive`, writing `norm_trace.jsonl` through `norm_trace.NormTraceWriter`). The driver never imports `schedules.py` or `norm.py`. |
| `src/ioplace/main_flow_metrics.py` | Pure metric functions: IO accounting identity, region area balance, fence compliance, phase summary. No torch, no DREAMPlace. |
| `src/ioplace/drivers/run_main_flow.py` | The driver: phase orchestration, CLI, artefact wiring, `result.json`. |

**New tests**

`tests/test_artifacts.py`, `tests/test_init_pos.py`, `tests/test_freeze.py`, `tests/test_fence_phase.py`, `tests/test_norm_adapter.py`, `tests/test_main_flow_metrics.py`, `tests/test_main_flow_driver.py`.

**Modified**

`src/scripts/run_route_gp.py` (import-time guard); `src/ioplace/ops/routing_gp_controller.py` (P-H Task 8's inline constructor gate *relocated* into `gr_in_loop.py` — not a second gate, see Task 8); `tests/test_routing_gp_driver.py`, `tests/test_bounded_grt_feedback.py` (set the guard env var); `docs/dev-env.md` (driver table).

**Reused unchanged** — do not edit: `regions.py`, `region_grid.py`, `region_graph.py`, `fence_inject.py`, `ops/soft_assign.py`, `ops/io_term.py`, `ops/ft_term.py`, `ops/ft_callback.py`, `schedules.py`, `dp_hook.py`, `export/evaluation.py`, `evaluator_gpu.py`, `drivers/run_placement.py`, `drivers/run_placement_io.py`, `drivers/run_placement_two_stage.py`.

**Small test input.** The smallest real LEF/DEF case available on this host is GCD: config `results/route_feedback_20260914/gcd.json`, LEF/DEF under `third_party/OpenROAD/src/grt/test/` (both present). 508 movable nodes, 168 terminals, 54 terminal-NIs, 579 nets, native die `(20140, 22400, 180500, 179200)`. `benchmarks/ispd25/mempool_tile_wrap.json` points at `/nashome/NVL4/...`, which does not exist on this host — do not use it, and **do not repair it here**: P-C Task 11 is the single owner of that repair and promotes a host-local copy to `benchmarks/ispd25/h100/mempool_tile_wrap.json`. No task in this plan creates or edits anything under `benchmarks/`. `$DREAMPLACE_ROOT/install/test/simple.json` (8 cells) is too small for multi-fence legalisation (see the note at `tests/test_fence_inject.py:127-149`) and is only used for non-fence unit work.

---

### Task 1: Artefact I/O and the coordinate contract

**Files:**
- Create: `src/ioplace/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `ioplace.regions.RegionSet`/`RegionSpec` (`src/ioplace/regions.py:5-58`).
- Consumed by: this plan's Tasks 2-9 **and all of P-C** (`drivers/run_region_producer.py` and its acceptance verifier). P-C Task 1 is a verification step over these names; it adds nothing. Do not rename anything here without editing `docs/superpowers/plans/2026-09-19-v2-p-c-region-producer.md` in the same commit.
- Produces:
  - `POSITIONS_SCHEMA_VERSION = 1`, `MEMBERSHIP_SCHEMA_VERSION = 1`, `FREEZE_SCHEMA_VERSION = 1`, `MAIN_FLOW_RESULT_SCHEMA_VERSION = 1`, `PRODUCER_SCHEMA_VERSION = 1`
  - `MAIN_FLOW_RESULT_FIELDS: tuple[str, ...]`, `FREEZE_FIELDS: tuple[str, ...]`, `PRODUCER_FIELDS: tuple[str, ...]`
  - `@dataclass Positions(node_x, node_y, die, shift_factor, scale_factor, placedb_sha256, kind)` with `.num_physical`
  - `@dataclass Membership(part, source, k, seed, epsilon)` with `.num_movable`
  - `placedb_identity_sha256(placedb) -> str`
  - `save_positions(path, node_x, node_y, *, die, shift_factor, scale_factor, placedb_sha256, kind) -> None`
  - `load_positions(path, *, expect_num_physical=None, expect_sha256=None) -> Positions`
  - `save_membership(path, part, *, source, k, seed=0, epsilon=0.0) -> None`
  - `load_membership(path, *, expect_num_movable=None, expect_k=None, require_nonempty=False) -> Membership`
  - `save_freeze(path, record) -> None`, `load_freeze(path) -> dict`
  - `save_result(path, record) -> None`
  - `save_producer_json(path, payload) -> None`, `load_producer_json(path) -> dict` (P-C's `producer.json`; the only writer that stamps `schema_version` itself, because the producer builds its whole payload in one place)
  - `scaled_region_set(rs, shift_factor, scale_factor) -> RegionSet`
  - `file_sha256(path) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_artifacts.py`:

```python
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ioplace.artifacts import (
    FREEZE_FIELDS, MAIN_FLOW_RESULT_FIELDS, MAIN_FLOW_RESULT_SCHEMA_VERSION,
    PRODUCER_FIELDS, PRODUCER_SCHEMA_VERSION, Membership, Positions,
    file_sha256, load_freeze, load_membership, load_positions,
    load_producer_json, placedb_identity_sha256, save_freeze, save_membership,
    save_positions, save_producer_json, save_result, scaled_region_set)
from ioplace.regions import make_grid_regions


def _fake_placedb(**override):
    base = dict(num_movable_nodes=3, num_physical_nodes=4, num_nets=2,
                node_size_x=np.array([1., 1., 1., 2.]),
                node_size_y=np.array([1., 1., 1., 2.]),
                pin2node_map=np.array([0, 1, 2, 3], dtype=np.int32),
                pin2net_map=np.array([0, 0, 1, 1], dtype=np.int32),
                flat_net2pin_start_map=np.array([0, 2, 4], dtype=np.int32))
    base.update(override)
    return SimpleNamespace(**base)


def test_placedb_identity_is_stable_and_connectivity_sensitive():
    first = placedb_identity_sha256(_fake_placedb())
    assert first == placedb_identity_sha256(_fake_placedb())
    assert len(first) == 64
    assert first != placedb_identity_sha256(
        _fake_placedb(pin2node_map=np.array([0, 1, 3, 2], dtype=np.int32)))
    assert first != placedb_identity_sha256(
        _fake_placedb(node_size_x=np.array([1., 1., 1., 3.])))


def test_placedb_identity_ignores_node_positions():
    """Counts, node sizes and connectivity only -- never a coordinate. Node
    sizes do change under PlaceDB.scale(), which is why the docstring pins the
    fingerprint to the pre-initialize() point of the lifecycle; the producer
    (P-C Task 10) hashes in its `read` phase for exactly that reason."""
    db = _fake_placedb()
    before = placedb_identity_sha256(db)
    db.node_x = np.array([1., 2., 3., 4.])
    db.xl, db.yl, db.xh, db.yh = 0., 0., 10., 10.
    assert placedb_identity_sha256(db) == before


def _positions(tmp_path, **override):
    kwargs = dict(die=(0., 0., 100., 200.), shift_factor=(0., 0.), scale_factor=2.,
                  placedb_sha256="abc", kind="seed")
    kwargs.update(override)
    path = str(tmp_path / "seed.npz")
    save_positions(path, np.array([1., 2., 3.]), np.array([4., 5., 6.]), **kwargs)
    return path


def test_positions_round_trip_preserves_every_field(tmp_path):
    path = _positions(tmp_path)
    got = load_positions(path)
    assert isinstance(got, Positions)
    assert np.array_equal(got.node_x, [1., 2., 3.])
    assert np.array_equal(got.node_y, [4., 5., 6.])
    assert got.die == (0., 0., 100., 200.)
    assert got.shift_factor == (0., 0.)
    assert got.scale_factor == 2.
    assert got.placedb_sha256 == "abc" and got.kind == "seed"
    assert got.num_physical == 3


def test_positions_reject_bad_writes(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        save_positions(str(tmp_path / "a.npz"), [1.], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=1., placedb_sha256="a",
                       kind="bogus")
    with pytest.raises(ValueError, match="same length"):
        save_positions(str(tmp_path / "b.npz"), [1., 2.], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=1., placedb_sha256="a",
                       kind="seed")
    with pytest.raises(ValueError, match="finite"):
        save_positions(str(tmp_path / "c.npz"), [np.nan], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=1., placedb_sha256="a",
                       kind="seed")
    with pytest.raises(ValueError, match="scale_factor"):
        save_positions(str(tmp_path / "d.npz"), [1.], [2.], die=(0., 0., 1., 1.),
                       shift_factor=(0., 0.), scale_factor=0., placedb_sha256="a",
                       kind="seed")


def test_positions_load_enforces_identity_and_size(tmp_path):
    path = _positions(tmp_path)
    load_positions(path, expect_num_physical=3, expect_sha256="abc")
    with pytest.raises(ValueError, match="num_physical"):
        load_positions(path, expect_num_physical=4)
    with pytest.raises(ValueError, match="placedb fingerprint"):
        load_positions(path, expect_sha256="other")


def test_membership_round_trip_and_validation(tmp_path):
    path = str(tmp_path / "m.npz")
    save_membership(path, np.array([0, 1, 1, 3]), source="mtkahypar", k=4,
                    seed=7, epsilon=0.03)
    got = load_membership(path, expect_num_movable=4, expect_k=4)
    assert isinstance(got, Membership)
    assert got.part.dtype == np.int32 and got.part.tolist() == [0, 1, 1, 3]
    assert got.source == "mtkahypar" and got.k == 4 and got.seed == 7
    assert got.epsilon == 0.03 and got.num_movable == 4
    with pytest.raises(ValueError, match="empty regions"):
        load_membership(path, require_nonempty=True)
    with pytest.raises(ValueError, match="num_movable"):
        load_membership(path, expect_num_movable=5)
    with pytest.raises(ValueError, match="out of range"):
        save_membership(str(tmp_path / "bad.npz"), np.array([0, 4]), source="x", k=4)


def test_freeze_record_requires_every_declared_field(tmp_path):
    record = {name: 0 for name in FREEZE_FIELDS}
    record["schema_version"] = 1
    path = str(tmp_path / "freeze.json")
    save_freeze(path, record)
    assert load_freeze(path)["schema_version"] == 1
    del record["churn"]
    with pytest.raises(ValueError, match="churn"):
        save_freeze(str(tmp_path / "bad.json"), record)


def test_result_requires_every_declared_field(tmp_path):
    record = {name: 0 for name in MAIN_FLOW_RESULT_FIELDS}
    record["schema_version"] = MAIN_FLOW_RESULT_SCHEMA_VERSION
    path = str(tmp_path / "result.json")
    save_result(path, record)
    assert json.load(open(path))["schema_version"] == MAIN_FLOW_RESULT_SCHEMA_VERSION
    del record["lg_loss"]
    with pytest.raises(ValueError, match="lg_loss"):
        save_result(str(tmp_path / "bad.json"), record)


def test_producer_json_requires_every_field_and_stamps_the_version(tmp_path):
    record = {name: 0 for name in PRODUCER_FIELDS}
    record["k"] = 16
    record["extract_bins"] = 64
    path = str(tmp_path / "producer.json")
    save_producer_json(path, record)
    on_disk = json.load(open(path))
    assert on_disk["k"] == 16 and on_disk["extract_bins"] == 64
    assert on_disk["schema_version"] == PRODUCER_SCHEMA_VERSION
    assert load_producer_json(path)["rects_per_region"] == 0
    del record["rects_per_region"]
    with pytest.raises(ValueError, match="rects_per_region"):
        save_producer_json(str(tmp_path / "bad.json"), record)


def test_scaled_region_set_matches_placedb_scale_and_still_validates():
    native = make_grid_regions((10., 20., 110., 220.), 2, 2, lattice=8)
    scaled = scaled_region_set(native, (10., 20.), 2.)
    scaled.validate()
    assert scaled.die == (0., 0., 200., 400.)
    assert scaled.lattice == native.lattice and scaled.k == native.k
    for a, b in zip(native.regions, scaled.regions):
        expected = (np.asarray(a.rects) - np.array([10., 20., 10., 20.])) * 2.
        assert np.allclose(np.asarray(b.rects), expected)
        assert a.name == b.name


def test_file_sha256_is_stable_and_content_sensitive(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    assert file_sha256(str(first)) == file_sha256(str(second))
    second.write_bytes(b"hellp")
    assert file_sha256(str(first)) != file_sha256(str(second))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.artifacts'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/artifacts.py`:

```python
"""v2 artefact I/O and the native/scaled coordinate contract (design v2 sec 1).

Shared by BOTH v2 drivers: the main flow (P-B, drivers/run_main_flow.py) and the
region producer (P-C, drivers/run_region_producer.py). Every name the producer
needs -- placedb_identity_sha256, save/load_positions, save/load_membership,
save/load_producer_json -- lives here; P-C adds nothing of its own.

Every array artefact here is written in **native post-read PlaceDB units**:
the coordinate frame of `placedb` after `placedb.read(params)` and before
`placedb.initialize(params)`. `initialize()` calls `PlaceDB.scale()`
($DREAMPLACE_ROOT/install/dreamplace/PlaceDB.py:151-196), which rescales
node positions, node sizes, the die box and `regions`/`flat_region_boxes`
together with `shift_factor = (xl, yl)` and `scale_factor = 1/site_width`
fixed at PlaceDB.py:759-767 -- native units are therefore the only frame in
which a seed, a region set and a membership vector produced by different
processes can be combined.

This module deliberately imports neither torch nor DREAMPlace: it must stay
loadable in a bare CPU process (report tooling, tests).
"""
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from ioplace.regions import RegionSet, RegionSpec

POSITIONS_SCHEMA_VERSION = 1
MEMBERSHIP_SCHEMA_VERSION = 1
FREEZE_SCHEMA_VERSION = 1
MAIN_FLOW_RESULT_SCHEMA_VERSION = 1
PRODUCER_SCHEMA_VERSION = 1

POSITION_KINDS = ("seed", "soft")

FREEZE_FIELDS = (
    "schema_version", "iteration", "reason", "overflow", "tau", "tau_rel",
    "churn", "k", "region_cell_count", "region_cell_area", "region_area",
    "region_utilization", "io_soft", "membership_npz", "soft_npz",
    "repaired_empty_regions", "gp_iterations_soft", "density_weight_soft",
)

MAIN_FLOW_RESULT_FIELDS = (
    # identity / provenance
    "mode", "schema_version", "config", "k", "rtype", "seed", "init",
    "norm_policy", "phase", "regions_json", "run_id", "status", "repo_commit",
    "input_sha256", "env", "command", "hostname", "benchmark_kind",
    "device_baseline_gb", "dp_seed", "det", "runtime_s",
    # IO accounting (design sec 7 diagnostics 4/5 + the closing identity)
    "io_soft", "io_fence_gp", "io_count", "io_delta_at_freeze", "lg_loss",
    "io_identity_residual", "io_fence_gp_source",
    # evaluator metrics
    "ft_count", "hard_lambda_sum", "tree_wl", "hpwl", "io_rg", "ft_rg",
    "large_net_lb", "hpwl_gp", "hpwl_lg",
    # fence + geometry diagnostics
    "fence_compliance", "fence_compliance_center", "region_area_balance",
    "freeze", "density_weight_clamp", "escape_cell",
    # runtime / memory / legalization
    "phases", "t_read_soft", "t_gp_soft", "t_freeze", "t_read_fence",
    "t_gp_fence", "t_lg", "t_eval", "peak_mem_mb", "peak_mem_mb_by_phase",
    "device_used_gb", "host_peak_rss_gb",
    "gp_iterations_soft", "gp_iterations_fence", "gp_iteration_budget",
    "final_overflow", "stop_overflow_reached", "legalization_status",
    "num_unplaced_cells", "effective_target_density", "num_filler_nodes",
    "num_bins_x", "num_bins_y",
    # soft-phase provenance (the run_soft_phase record minus its arrays; None
    # for a --phase fence run, which never opens the soft phase)
    "soft_summary",
    # artefacts
    "artifacts",
)

# producer.json -- the region producer's run record (P-C Task 10 builds it).
PRODUCER_FIELDS = (
    # identity / provenance
    "config", "out_dir", "placedb_sha256", "command", "hostname", "env",
    # knobs
    "k", "membership_source", "membership_seed", "epsilon", "hierarchy_depth",
    "extract_bins", "fine_bins", "lattice", "rect_max", "t_hull",
    "probe_every", "alpha_pull", "alpha_push", "sa_seed",
    # geometry and the coordinate contract
    "die_native", "die_scaled", "shift_factor", "scale_factor",
    "num_movable", "num_physical", "num_nodes", "num_nets", "target_density",
    # grouping-term telemetry
    "n_hull_rebuilds", "wt_final", "lambda_group_final", "ratio_ema_final",
    "probes",
    # placement outcome
    "gp_iterations_run", "final_overflow", "hpwl_gp", "hpwl_lg",
    # shapes
    "sa", "rects_per_region", "rect_max_observed", "region_bins",
    "region_area", "region_cell_area", "region_utilisation", "area_balance",
    # runtime
    "runtime_s", "peak_mem_mb",
)


@dataclass
class Positions:
    node_x: np.ndarray
    node_y: np.ndarray
    die: tuple
    shift_factor: tuple
    scale_factor: float
    placedb_sha256: str
    kind: str

    @property
    def num_physical(self):
        return len(self.node_x)


@dataclass
class Membership:
    part: np.ndarray
    source: str
    k: int
    seed: int
    epsilon: float

    @property
    def num_movable(self):
        return len(self.part)


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def placedb_identity_sha256(placedb):
    """Fingerprint of the netlist structure a seed/membership must match.

    Must be computed after read() and BEFORE initialize(): node sizes are
    multiplied by scale_factor inside initialize() (PlaceDB.py:160-161), so
    the same design would otherwise fingerprint differently in phase 1 and
    phase 3 and every cross-phase check would spuriously fail. The region
    producer (P-C) hashes in its own `read` phase for the same reason.

    Covers counts, node sizes and full pin connectivity -- never a coordinate.
    pin2node_map/pin2net_map come from P-C's `placedb_fingerprint`, which this
    function replaces: the start map alone does not notice a permutation of the
    pins inside a net, which is exactly the "seed written for a different
    design" case the guard exists for.
    """
    return _digest_arrays(
        np.asarray([placedb.num_movable_nodes, placedb.num_physical_nodes,
                    placedb.num_nets, len(placedb.pin2node_map)], dtype=np.int64),
        np.asarray(placedb.node_size_x[:placedb.num_physical_nodes], dtype=np.float64),
        np.asarray(placedb.node_size_y[:placedb.num_physical_nodes], dtype=np.float64),
        np.asarray(placedb.pin2node_map, dtype=np.int64),
        np.asarray(placedb.pin2net_map, dtype=np.int64),
        np.asarray(placedb.flat_net2pin_start_map, dtype=np.int64))


def _atomic_savez(path, **arrays):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", suffix=".npz", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=1, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_positions(path, node_x, node_y, *, die, shift_factor, scale_factor,
                   placedb_sha256, kind):
    if kind not in POSITION_KINDS:
        raise ValueError(f"kind must be one of {POSITION_KINDS}, got {kind!r}")
    x = np.asarray(node_x, dtype=np.float64).reshape(-1)
    y = np.asarray(node_y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("node_x and node_y must have the same length")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("positions must be finite")
    die = tuple(float(v) for v in die)
    shift_factor = tuple(float(v) for v in shift_factor)
    if len(die) != 4:
        raise ValueError("die must be (xl, yl, xh, yh)")
    if len(shift_factor) != 2:
        raise ValueError("shift_factor must be (dx, dy)")
    if not float(scale_factor) > 0.0:
        raise ValueError("scale_factor must be positive")
    _atomic_savez(path, node_x=x, node_y=y,
                  die=np.asarray(die, dtype=np.float64),
                  shift_factor=np.asarray(shift_factor, dtype=np.float64),
                  scale_factor=np.asarray(float(scale_factor), dtype=np.float64),
                  placedb_sha256=np.asarray(str(placedb_sha256)),
                  kind=np.asarray(str(kind)),
                  schema_version=np.asarray(POSITIONS_SCHEMA_VERSION, dtype=np.int64))


def load_positions(path, *, expect_num_physical=None, expect_sha256=None):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if int(data["schema_version"]) != POSITIONS_SCHEMA_VERSION:
        raise ValueError(f"unsupported positions schema in {path}")
    out = Positions(
        node_x=np.asarray(data["node_x"], dtype=np.float64),
        node_y=np.asarray(data["node_y"], dtype=np.float64),
        die=tuple(float(v) for v in data["die"]),
        shift_factor=tuple(float(v) for v in data["shift_factor"]),
        scale_factor=float(data["scale_factor"]),
        placedb_sha256=str(data["placedb_sha256"]),
        kind=str(data["kind"]))
    if expect_num_physical is not None and out.num_physical != int(expect_num_physical):
        raise ValueError(f"{path}: num_physical {out.num_physical} != {expect_num_physical}")
    if expect_sha256 is not None and out.placedb_sha256 != expect_sha256:
        raise ValueError(f"{path}: placedb fingerprint mismatch "
                         f"({out.placedb_sha256} != {expect_sha256})")
    return out


def save_membership(path, part, *, source, k, seed=0, epsilon=0.0):
    arr = np.asarray(part).reshape(-1).astype(np.int32)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    if arr.size and (arr.min() < 0 or arr.max() >= k):
        raise ValueError(f"membership out of range [0, {k})")
    _atomic_savez(path, part=arr,
                  source=np.asarray(str(source)),
                  k=np.asarray(k, dtype=np.int64),
                  seed=np.asarray(int(seed), dtype=np.int64),
                  epsilon=np.asarray(float(epsilon), dtype=np.float64),
                  schema_version=np.asarray(MEMBERSHIP_SCHEMA_VERSION, dtype=np.int64))


def load_membership(path, *, expect_num_movable=None, expect_k=None,
                    require_nonempty=False):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if int(data["schema_version"]) != MEMBERSHIP_SCHEMA_VERSION:
        raise ValueError(f"unsupported membership schema in {path}")
    out = Membership(part=np.asarray(data["part"], dtype=np.int32),
                     source=str(data["source"]), k=int(data["k"]),
                     seed=int(data["seed"]), epsilon=float(data["epsilon"]))
    if expect_num_movable is not None and out.num_movable != int(expect_num_movable):
        raise ValueError(f"{path}: num_movable {out.num_movable} != {expect_num_movable}")
    if expect_k is not None and out.k != int(expect_k):
        raise ValueError(f"{path}: k {out.k} != {expect_k}")
    if require_nonempty:
        counts = np.bincount(out.part, minlength=out.k)
        empty = np.flatnonzero(counts == 0).tolist()
        if empty:
            # PlaceDB.calc_num_filler_for_fence_region takes np.percentile of
            # an empty movable-size slice (PlaceDB.py:687); under the installed
            # numpy (1.26.4) that raises `IndexError: index -1 is out of bounds
            # for axis 0 with size 0` right there, inside initialize() -- it
            # never reaches the int(round(nan)) at PlaceDB.py:728. Verified on
            # this host 2026-09-19 (pre-flight E-3); the older "silently yields
            # NaN then ValueError at :729" wording came from
            # run_placement_two_stage.py's comment block and is wrong. Refuse
            # the membership here with a readable message either way.
            raise ValueError(f"{path}: empty regions {empty} would crash "
                             "PlaceDB.calc_num_filler_for_fence_region")
    return out


def _require_fields(record, fields, what):
    missing = [name for name in fields if name not in record]
    if missing:
        raise ValueError(f"{what} is missing required field(s): {', '.join(missing)}")


def save_freeze(path, record):
    _require_fields(record, FREEZE_FIELDS, "freeze.json")
    _atomic_write_json(path, record)


def load_freeze(path):
    with open(path) as stream:
        record = json.load(stream)
    _require_fields(record, FREEZE_FIELDS, f"{path}")
    if int(record["schema_version"]) != FREEZE_SCHEMA_VERSION:
        raise ValueError(f"unsupported freeze schema in {path}")
    return record


def save_result(path, record):
    _require_fields(record, MAIN_FLOW_RESULT_FIELDS, "result.json")
    _atomic_write_json(path, record)


def save_producer_json(path, payload):
    """producer.json -- the region producer's own run record (P-C Task 10).

    Unlike save_freeze/save_result this writer stamps schema_version itself:
    run_region_producer.run_producer builds the payload as one dict literal and
    also returns it to its caller, so the version belongs to the writer.
    """
    _require_fields(payload, PRODUCER_FIELDS, "producer.json")
    record = dict(payload)
    record["schema_version"] = PRODUCER_SCHEMA_VERSION
    _atomic_write_json(path, record)


def load_producer_json(path):
    with open(path) as stream:
        record = json.load(stream)
    _require_fields(record, PRODUCER_FIELDS, f"{path}")
    if int(record["schema_version"]) != PRODUCER_SCHEMA_VERSION:
        raise ValueError(f"unsupported producer schema in {path}")
    return record


def scaled_region_set(rs, shift_factor, scale_factor):
    """Native -> scaled RegionSet, using PlaceDB.scale()'s own transform
    (PlaceDB.py:184-196: subtract the shift, multiply by the scale, applied to
    both corners of every rect). The lattice count is unchanged, so an input
    that passes validate() still passes it afterwards."""
    shift = np.asarray([shift_factor[0], shift_factor[1],
                        shift_factor[0], shift_factor[1]], dtype=np.float64)
    scale = float(scale_factor)
    regions = [RegionSpec(r.name,
                          (np.asarray(r.rects, dtype=np.float64).reshape(-1, 4) - shift) * scale)
               for r in rs.regions]
    xl, yl, xh, yh = rs.die
    die = ((xl - shift_factor[0]) * scale, (yl - shift_factor[1]) * scale,
           (xh - shift_factor[0]) * scale, (yh - shift_factor[1]) * scale)
    return RegionSet(die=die, lattice=rs.lattice, regions=regions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_artifacts.py -v`
Expected: PASS — 11 passed.

- [ ] **Step 5: Run the fast suite for regressions**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow" -q`
Expected: PASS — no new failures relative to the pre-task baseline.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/artifacts.py tests/test_artifacts.py
git commit -m "feat(artifacts): v2 artefact I/O with schema validation

Adds seed/soft positions, membership, freeze.json, producer.json and
result.json readers and writers in native post-read PlaceDB units, plus
placedb_identity_sha256 and the native->scaled RegionSet conversion. This is
the single owner of the producer (P-C) <-> main-flow (P-B) file contract.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Initial-position modes (`die_center`, `region_center`, `seed`)

**Files:**
- Create: `src/ioplace/init_pos.py`
- Test: `tests/test_init_pos.py`

**Interfaces:**
- Consumes: `ioplace.artifacts.Positions` (Task 1); `ioplace.regions.RegionSet`.
- Produces:
  - `INIT_MODES = ("die_center", "region_center", "seed")`
  - `region_centers(rs) -> np.ndarray` shape `(k, 2)`, area-weighted rect centres, native units
  - `apply_init(placedb, params, mode, *, region_set=None, part=None, positions=None, rng_seed=0) -> dict`

**Why this design.** `BasicPlace.__init__` copies `placedb.node_x` into `init_pos` and, only when `params.random_center_init_flag` is set, overwrites the movable slice with `np.random.normal(loc=die centre, scale=0.001*(xh-xl))` (`$DREAMPLACE_ROOT/install/dreamplace/BasicPlace.py:269-288`). So a warm start is "write `placedb.node_x`/`node_y` before `initialize()` and set `random_center_init_flag = 0`"; `initialize()`'s `scale()` then converts it (spec §3 phase 1). `region_center` is *not* DREAMPlace's die-centre path with a different `loc` — DREAMPlace writes the **lower-left** at the centre, which for a wide macro puts its *centre* in the neighbouring region and disagrees with the freeze membership (argmax at the cell centre, spec §3 phase 2). This module therefore places the **cell centre** at the region centre and reuses DREAMPlace's own noise scale verbatim. The draw comes from a dedicated `np.random.default_rng(rng_seed)` so it cannot perturb the global numpy stream that `BasicPlace`'s filler initialisation consumes (`run_placement.py:66-71`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_init_pos.py`:

```python
import numpy as np
import pytest
from types import SimpleNamespace

from ioplace.init_pos import INIT_MODES, apply_init, region_centers
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def _placedb():
    return SimpleNamespace(
        num_movable_nodes=4, num_terminals=1, num_terminal_NIs=0,
        num_physical_nodes=5,
        node_x=np.array([1., 2., 3., 4., 90.]),
        node_y=np.array([5., 6., 7., 8., 90.]),
        node_size_x=np.array([2., 2., 40., 2., 1.]),
        node_size_y=np.array([2., 2., 2., 2., 1.]),
        xl=0., yl=0., xh=100., yh=100.)


def _params():
    return SimpleNamespace(random_center_init_flag=1, random_seed=1000)


def test_modes_are_declared():
    assert INIT_MODES == ("die_center", "region_center", "seed")


def test_region_centers_are_area_weighted():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    assert np.allclose(region_centers(rs),
                       [[25., 25.], [75., 25.], [25., 75.], [75., 75.]])


def test_die_center_leaves_positions_to_dreamplace():
    db, params = _placedb(), _params()
    before_x = db.node_x.copy()
    info = apply_init(db, params, "die_center")
    assert params.random_center_init_flag == 1
    assert np.array_equal(db.node_x, before_x)
    assert info["mode"] == "die_center"


def test_region_center_puts_every_cell_centre_in_its_own_region():
    db, params = _placedb(), _params()
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    part = np.array([0, 1, 2, 3], dtype=np.int32)
    info = apply_init(db, params, "region_center", region_set=rs, part=part,
                      rng_seed=1000)
    assert params.random_center_init_flag == 0
    m = db.num_movable_nodes
    cx = db.node_x[:m] + db.node_size_x[:m] / 2.
    cy = db.node_y[:m] + db.node_size_y[:m] / 2.
    assert np.array_equal(RegionGrid(rs).region_of_points(cx, cy), part)
    # the 40-wide cell 2 must have its centre, not its lower-left, on the centre
    assert db.node_x[2] < 25.
    assert np.array_equal(db.node_x[m:], [90.])      # fixed nodes untouched
    assert info["mode"] == "region_center" and info["noise_scale_x"] == 0.1


def test_region_center_is_deterministic_per_seed():
    def run(seed):
        db, params = _placedb(), _params()
        rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
        apply_init(db, params, "region_center", region_set=rs,
                   part=np.array([0, 1, 2, 3], dtype=np.int32), rng_seed=seed)
        return db.node_x.copy()
    assert np.array_equal(run(1000), run(1000))
    assert not np.array_equal(run(1000), run(2000))


def test_seed_mode_writes_movable_slice_and_validates_fixed_slice():
    from ioplace.artifacts import Positions
    db, params = _placedb(), _params()
    good = Positions(node_x=np.array([10., 11., 12., 13., 90.]),
                     node_y=np.array([20., 21., 22., 23., 90.]),
                     die=(0., 0., 100., 100.), shift_factor=(0., 0.),
                     scale_factor=1., placedb_sha256="x", kind="seed")
    apply_init(db, params, "seed", positions=good)
    assert params.random_center_init_flag == 0
    assert np.array_equal(db.node_x, [10., 11., 12., 13., 90.])
    assert np.array_equal(db.node_y, [20., 21., 22., 23., 90.])

    db2 = _placedb()
    moved = Positions(node_x=np.array([10., 11., 12., 13., 91.]),
                      node_y=np.array([20., 21., 22., 23., 90.]),
                      die=(0., 0., 100., 100.), shift_factor=(0., 0.),
                      scale_factor=1., placedb_sha256="x", kind="seed")
    with pytest.raises(ValueError, match="fixed nodes"):
        apply_init(db2, _params(), "seed", positions=moved)

    db3 = _placedb()
    short = Positions(node_x=np.array([1., 2.]), node_y=np.array([1., 2.]),
                      die=(0., 0., 100., 100.), shift_factor=(0., 0.),
                      scale_factor=1., placedb_sha256="x", kind="seed")
    with pytest.raises(ValueError, match="num_physical"):
        apply_init(db3, _params(), "seed", positions=short)


def test_unknown_mode_and_missing_inputs_are_rejected():
    db, params = _placedb(), _params()
    with pytest.raises(ValueError, match="mode"):
        apply_init(db, params, "warp")
    with pytest.raises(ValueError, match="region_set and part"):
        apply_init(db, params, "region_center")
    with pytest.raises(ValueError, match="positions"):
        apply_init(db, params, "seed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_init_pos.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.init_pos'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/init_pos.py`:

```python
"""Initial movable-cell positions for the v2 main flow (design v2 sec 3, arms (a)-(d)).

All three modes write `placedb.node_x`/`node_y` in **native post-read units**
and must run after `placedb.read(params)` and before
`placedb.initialize(params)`: `initialize()`'s `scale()` converts whatever is
in those arrays into the internal frame (PlaceDB.py:151-196), and
`BasicPlace.__init__` copies them into `init_pos` (BasicPlace.py:269-288).

`random_center_init_flag` is the switch that decides whether DREAMPlace
overwrites the movable slice with its own die-centre Gaussian
(BasicPlace.py:272-277/283-288): `die_center` leaves it at 1, the two
explicit modes set it to 0.
"""
import numpy as np

INIT_MODES = ("die_center", "region_center", "seed")

# BasicPlace.py:276/287 -- DREAMPlace's own centre-init noise, as a fraction of
# the die span. Reused verbatim so `region_center` differs from `die_center`
# only in the mean, never in the dispersion.
NOISE_FRACTION = 0.001


def region_centers(rs):
    """(k, 2) area-weighted centre of each region's rects, native units.

    Same formula as run_placement_two_stage.assign_blocks_to_regions'
    geometry step (run_placement_two_stage.py:112-119) -- a RegionSpec may
    hold several rects once the producer (P-C) emits rectilinear regions.
    """
    centers = np.zeros((rs.k, 2), dtype=np.float64)
    for rid, region in enumerate(rs.regions):
        rects = np.asarray(region.rects, dtype=np.float64).reshape(-1, 4)
        area = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        cx = (rects[:, 0] + rects[:, 2]) * 0.5
        cy = (rects[:, 1] + rects[:, 3]) * 0.5
        total = area.sum()
        centers[rid] = (np.sum(cx * area) / total, np.sum(cy * area) / total)
    return centers


def apply_init(placedb, params, mode, *, region_set=None, part=None,
               positions=None, rng_seed=0):
    """Install the initial positions for `mode`; return a JSON-able record."""
    if mode not in INIT_MODES:
        raise ValueError(f"mode must be one of {INIT_MODES}, got {mode!r}")
    m = placedb.num_movable_nodes
    n_phys = placedb.num_physical_nodes

    if mode == "die_center":
        params.random_center_init_flag = 1
        return {"mode": mode}

    params.random_center_init_flag = 0

    if mode == "region_center":
        if region_set is None or part is None:
            raise ValueError("region_center needs region_set and part")
        part = np.asarray(part, dtype=np.int64).reshape(-1)
        if len(part) != m:
            raise ValueError(f"part has {len(part)} entries, expected {m}")
        if part.min() < 0 or part.max() >= region_set.k:
            raise ValueError(f"part out of range [0, {region_set.k})")
        centers = region_centers(region_set)
        width = float(placedb.xh) - float(placedb.xl)
        height = float(placedb.yh) - float(placedb.yl)
        scale_x, scale_y = width * NOISE_FRACTION, height * NOISE_FRACTION
        rng = np.random.default_rng(int(rng_seed))
        size_x = np.asarray(placedb.node_size_x[:m], dtype=np.float64)
        size_y = np.asarray(placedb.node_size_y[:m], dtype=np.float64)
        # The *cell centre* lands on the region centre: the freeze membership
        # is the argmax at the cell centre (design v2 sec 3 phase 2), so
        # initialising the lower-left there -- what BasicPlace.py:272-277 does
        # for the die centre -- would start wide cells in a foreign region.
        x = centers[part, 0] - 0.5 * size_x + rng.normal(0.0, scale_x, size=m)
        y = centers[part, 1] - 0.5 * size_y + rng.normal(0.0, scale_y, size=m)
        x = np.clip(x, float(placedb.xl), float(placedb.xh) - size_x)
        y = np.clip(y, float(placedb.yl), float(placedb.yh) - size_y)
        placedb.node_x[:m] = x.astype(placedb.node_x.dtype)
        placedb.node_y[:m] = y.astype(placedb.node_y.dtype)
        return {"mode": mode, "rng_seed": int(rng_seed),
                "noise_scale_x": scale_x, "noise_scale_y": scale_y}

    if positions is None:
        raise ValueError("seed mode needs positions")
    if positions.num_physical != n_phys:
        raise ValueError(f"seed num_physical {positions.num_physical} != {n_phys}")
    fixed_x = np.asarray(placedb.node_x[m:n_phys], dtype=np.float64)
    fixed_y = np.asarray(placedb.node_y[m:n_phys], dtype=np.float64)
    span = max(float(placedb.xh) - float(placedb.xl),
               float(placedb.yh) - float(placedb.yl))
    tol = 1e-6 * span
    if (not np.allclose(positions.node_x[m:n_phys], fixed_x, atol=tol, rtol=0.0)
            or not np.allclose(positions.node_y[m:n_phys], fixed_y, atol=tol, rtol=0.0)):
        raise ValueError("seed disagrees with this design's fixed nodes "
                         "(terminals/macros must not move between phases)")
    placedb.node_x[:m] = np.asarray(positions.node_x[:m]).astype(placedb.node_x.dtype)
    placedb.node_y[:m] = np.asarray(positions.node_y[:m]).astype(placedb.node_y.dtype)
    return {"mode": mode, "placedb_sha256": positions.placedb_sha256,
            "source_kind": positions.kind}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_init_pos.py -v`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/init_pos.py tests/test_init_pos.py
git commit -m "feat(init): die_center/region_center/seed initial-position modes

region_center places each cell's centre (not its lower-left) on its prior
region's area-weighted centre with DREAMPlace's own 0.001*span Gaussian,
drawn from a private RNG so the global numpy stream BasicPlace uses for
filler init is untouched.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Freeze monitor — cell-centre argmax, churn, criterion

**Files:**
- Create: `src/ioplace/freeze.py`
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `ioplace.ops.soft_assign.region_sdf_l1` (`src/ioplace/ops/soft_assign.py:19-31`); `ioplace.artifacts.FREEZE_SCHEMA_VERSION` (Task 1). It does **not** consume `ioplace.init_pos.region_centers` — only `tests/test_freeze.py` imports that, to build the `centers` argument `ensure_nonempty_regions` takes as a plain array (pre-flight amendment, Task 3 Interfaces line).
- Consumed by: `ioplace.main_flow_metrics.region_area_balance` (Task 6) imports `region_cell_stats` from here — one implementation of the per-region count/area/utilisation arithmetic, two consumers (pre-flight amendment D-2). The dependency points this way, not the other, because Task 3 lands first and `main_flow_metrics` must stay torch-free at import time (`freeze.py` pulls in `ops/soft_assign`, which imports torch), so Task 6 takes it as a local import inside the function.
- Produces:
  - `argmax_region(x, y, rects, rect2region, K, chunk=None) -> torch.Tensor` (int64, shape `(N,)`)
  - `cell_centers(x, y, size_x, size_y) -> (torch.Tensor, torch.Tensor)`
  - `class FreezeMonitor(window=50, overflow_max=0.15, tau_rel_max=0.05, churn_max=0.005)` with `observe(iteration, argmax) -> float | None`, `.churn`, `.last_iteration`, `should_freeze(overflow, tau_rel) -> bool`, `reasons(overflow, tau_rel) -> dict`
  - `ensure_nonempty_regions(part, k, cx, cy, centers) -> (np.ndarray, list[dict])`
  - `region_cell_stats(part, size_x, size_y, rs) -> dict` with keys `region_cell_count`, `region_cell_area`, `region_area`, `region_utilization`
  - `freeze_record(**fields) -> dict`

**Why the argmax is at the cell centre.** `region_sdf_l1` returns a *signed* L1 distance — negative inside the owning rect (`soft_assign.py:27`: `max(dx,dy).clamp(max=0)`), non-negative elsewhere — so `argmax_k(-SDF_k)` is exactly the region containing the point, with no dependence on τ. Evaluating it at `(x + w/2, y + h/2)` is what spec §3 phase 2 mandates ("Membership = argmax region of the cell **centre**"); the production `IoTerm` still anchors at the lower-left until P-F changes it, so P-B computes this argmax explicitly rather than reusing the term's.

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze.py`:

```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.freeze import (FreezeMonitor, argmax_region, cell_centers,
                            ensure_nonempty_regions, freeze_record,
                            region_cell_stats)
from ioplace.init_pos import region_centers
from ioplace.ops.soft_assign import rect_table
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def _grid():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    rects, r2k = rect_table(rs)
    return rs, torch.as_tensor(rects), torch.as_tensor(r2k)


def test_argmax_region_matches_region_of_points_at_cell_centres():
    rs, rects, r2k = _grid()
    x = torch.tensor([10., 60., 10., 60.], dtype=torch.float64)
    y = torch.tensor([10., 10., 60., 60.], dtype=torch.float64)
    got = argmax_region(x, y, rects, r2k, rs.k)
    assert got.tolist() == RegionGrid(rs).region_of_points(x.numpy(), y.numpy()).tolist()
    assert argmax_region(x, y, rects, r2k, rs.k, chunk=1).tolist() == got.tolist()


def test_cell_centre_anchor_differs_from_lower_left_for_a_straddling_cell():
    rs, rects, r2k = _grid()
    x = torch.tensor([40.], dtype=torch.float64)       # lower-left in region 0
    y = torch.tensor([10.], dtype=torch.float64)
    size_x = torch.tensor([30.], dtype=torch.float64)  # centre at x=55 -> region 1
    size_y = torch.tensor([2.], dtype=torch.float64)
    assert argmax_region(x, y, rects, r2k, rs.k).tolist() == [0]
    cx, cy = cell_centers(x, y, size_x, size_y)
    assert argmax_region(cx, cy, rects, r2k, rs.k).tolist() == [1]


def test_monitor_churn_uses_the_sample_one_window_back():
    monitor = FreezeMonitor(window=50)
    base = np.array([0, 1, 2, 3], dtype=np.int64)
    assert monitor.observe(0, base) is None            # nothing to compare to yet
    assert monitor.observe(50, base.copy()) == 0.0
    changed = base.copy(); changed[0] = 3
    assert monitor.observe(100, changed) == 0.25
    assert monitor.churn == 0.25 and monitor.last_iteration == 100


def test_criterion_needs_all_three_conditions():
    monitor = FreezeMonitor(window=50, overflow_max=.15, tau_rel_max=.05,
                            churn_max=.005)
    base = np.zeros(1000, dtype=np.int64)
    monitor.observe(0, base)
    monitor.observe(50, base.copy())                   # churn 0.0
    assert monitor.should_freeze(overflow=.10, tau_rel=.04)
    assert not monitor.should_freeze(overflow=.16, tau_rel=.04)
    assert not monitor.should_freeze(overflow=.10, tau_rel=.06)
    churned = base.copy(); churned[:10] = 1            # churn 1% > 0.5%
    monitor.observe(100, churned)
    assert not monitor.should_freeze(overflow=.10, tau_rel=.04)
    flags = monitor.reasons(overflow=.10, tau_rel=.04)
    assert flags == {"overflow_ok": True, "tau_ok": True, "churn_ok": False}


def test_criterion_never_fires_before_a_full_window():
    monitor = FreezeMonitor(window=50)
    monitor.observe(10, np.zeros(4, dtype=np.int64))
    assert monitor.churn is None
    assert not monitor.should_freeze(overflow=.0, tau_rel=.0)


def test_ensure_nonempty_regions_moves_the_nearest_cell_only():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    centers = region_centers(rs)
    part = np.array([0, 0, 0, 1], dtype=np.int32)      # regions 2 and 3 empty
    cx = np.array([10., 20., 30., 60.])
    cy = np.array([10., 10., 70., 10.])
    repaired, moves = ensure_nonempty_regions(part, 4, cx, cy, centers)
    assert np.bincount(repaired, minlength=4).min() >= 1
    assert len(moves) == 2
    assert moves[0]["region"] == 2 and moves[0]["cell"] == 2   # (30,70) is nearest
    assert int(repaired[2]) == 2


def test_region_cell_stats_reports_area_and_utilisation():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    part = np.array([0, 0, 1], dtype=np.int32)
    size_x = np.array([10., 10., 5.])
    size_y = np.array([10., 10., 4.])
    stats = region_cell_stats(part, size_x, size_y, rs)
    assert stats["region_cell_count"] == [2, 1, 0, 0]
    assert stats["region_cell_area"] == [200., 20., 0., 0.]
    assert stats["region_area"] == [2500.] * 4
    assert stats["region_utilization"] == [0.08, 0.008, 0., 0.]


def test_freeze_record_is_schema_complete(tmp_path):
    from ioplace.artifacts import save_freeze
    record = freeze_record(
        iteration=300, reason="criterion", overflow=.12, tau=1.5, tau_rel=.04,
        churn=.001, k=4, io_soft=1234, membership_npz="frozen_membership.npz",
        soft_npz="soft.npz", repaired_empty_regions=[], gp_iterations_soft=301,
        density_weight_soft=1.5e-5,
        stats={"region_cell_count": [1, 1, 1, 1], "region_cell_area": [1.] * 4,
               "region_area": [4.] * 4, "region_utilization": [.25] * 4})
    save_freeze(str(tmp_path / "freeze.json"), record)      # must not raise
    assert record["schema_version"] == 1 and record["reason"] == "criterion"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_freeze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.freeze'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/freeze.py`:

```python
"""Phase-2 freeze: cell-centre membership, churn, and the stop criterion
(design v2 sec 3 phase 2).

Freeze when all three hold:
  * `overflow <= 0.15`
  * `tau_rel <= 0.05` (the FT full-ramp point, schedules.py:125)
  * membership churn over the last 50 iterations `<= 0.5%`

Membership is the argmax region of the **cell centre**. `region_sdf_l1`
returns a signed distance -- negative inside the owning rect
(soft_assign.py:27's `max(dx,dy).clamp(max=0)`), non-negative outside -- so
`argmax_k(-SDF_k)` is the containing region independently of tau, and this
module needs only the first of `softmax_stats`' two passes
(soft_assign.py:40-58).
"""
from collections import deque

import numpy as np

from ioplace.artifacts import FREEZE_SCHEMA_VERSION
from ioplace.ops.soft_assign import region_sdf_l1


def _chunk_bounds(K, chunk):
    if chunk is None or chunk >= K:
        return [(0, K)]
    return [(lo, min(lo + chunk, K)) for lo in range(0, K, chunk)]


def argmax_region(x, y, rects, rect2region, K, chunk=None):
    """(N,) int64 owning-region index, chunked over regions like softmax_stats."""
    import torch
    best = torch.full((x.shape[0],), -float("inf"), dtype=x.dtype, device=x.device)
    out = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    for lo, hi in _chunk_bounds(K, chunk):
        z = -region_sdf_l1(x, y, rects, rect2region, lo, hi)
        cm, ci = z.max(dim=1)
        upd = cm > best                       # strict > keeps first-occurrence ties
        best = torch.where(upd, cm, best)
        out = torch.where(upd, ci + lo, out)
    return out


def cell_centers(x, y, size_x, size_y):
    return x + 0.5 * size_x, y + 0.5 * size_y


class FreezeMonitor:
    """Tracks membership churn over a fixed iteration window and evaluates the
    three-part freeze criterion. Observation cadence is the caller's
    (the driver's `--every` evaluator-gated callback); churn is always taken
    against the newest retained sample at least `window` iterations old, so a
    cadence change cannot silently change the meaning of the number."""

    def __init__(self, *, window=50, overflow_max=0.15, tau_rel_max=0.05,
                 churn_max=0.005):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self.overflow_max = float(overflow_max)
        self.tau_rel_max = float(tau_rel_max)
        self.churn_max = float(churn_max)
        self.churn = None
        self.last_iteration = None
        self._samples = deque()               # (iteration, argmax int64 numpy)

    def observe(self, iteration, argmax):
        arr = np.asarray(argmax.cpu() if hasattr(argmax, "cpu") else argmax,
                         dtype=np.int64)
        reference = None
        for it, part in self._samples:
            if it <= iteration - self.window:
                reference = (it, part)        # newest sample a full window back
        self.churn = (float(np.mean(arr != reference[1]))
                      if reference is not None else None)
        self._samples.append((int(iteration), arr))
        # keep only what a future comparison can still need: the newest sample
        # older than the window, plus everything after it.
        while len(self._samples) > 2 and self._samples[1][0] <= iteration - self.window:
            self._samples.popleft()
        self.last_iteration = int(iteration)
        return self.churn

    def reasons(self, overflow, tau_rel):
        return {"overflow_ok": bool(overflow <= self.overflow_max),
                "tau_ok": bool(tau_rel <= self.tau_rel_max),
                "churn_ok": bool(self.churn is not None
                                 and self.churn <= self.churn_max)}

    def should_freeze(self, overflow, tau_rel):
        return all(self.reasons(overflow, tau_rel).values())


def ensure_nonempty_regions(part, k, cx, cy, centers):
    """Guarantee every region owns at least one cell.

    An empty *real* region makes `PlaceDB.calc_num_filler_for_fence_region`
    take `np.percentile` of an empty movable-size slice (PlaceDB.py:687).
    Under the installed numpy (1.26.4) that raises `IndexError: index -1 is out
    of bounds for axis 0 with size 0` on the spot, inside `initialize()` --
    verified on this host 2026-09-19 (pre-flight E-3). It never reaches the
    `int(round(nan))` at PlaceDB.py:728, so do not go looking for a
    `ValueError`. The escape-cell workaround
    (run_placement_two_stage.py:192-233) only covers the *implicit* bucket, so
    the freeze must repair real regions itself: move the single cell whose
    centre is closest (L1) to the empty region's centre, preferring cells from
    regions that own more than one.
    """
    part = np.asarray(part, dtype=np.int32).copy()
    centers = np.asarray(centers, dtype=np.float64)
    moves = []
    for region in range(int(k)):
        counts = np.bincount(part, minlength=int(k))
        if counts[region] > 0:
            continue
        donor_ok = counts[part] >= 2
        pool = np.flatnonzero(donor_ok) if donor_ok.any() else np.arange(len(part))
        distance = (np.abs(cx[pool] - centers[region, 0])
                    + np.abs(cy[pool] - centers[region, 1]))
        cell = int(pool[int(np.argmin(distance))])
        moves.append({"region": region, "cell": cell, "from": int(part[cell]),
                      "distance": float(distance.min())})
        part[cell] = region
    return part, moves


def region_cell_stats(part, size_x, size_y, rs):
    """The four per-region arrays `freeze.json` carries.

    This is the **single** implementation of the per-region count / cell-area /
    region-area / utilisation arithmetic in the v2 flow:
    `main_flow_metrics.region_area_balance` (Task 6) calls it and only adds the
    max/min/ratio/deviation summaries that belong to `result.json` (pre-flight
    amendment D-2). It lives here rather than there because Task 3 lands first
    and `main_flow_metrics` must stay importable without torch -- this module
    pulls in `ops/soft_assign`, which imports torch -- so Task 6 takes it as a
    local import inside the function.

    `region_cell_area` and `region_area` are **not** scale-invariant; only
    `region_utilization` is (both of its terms carry `scale_factor**2`). Every
    caller must therefore pass sizes and a `RegionSet` in the *same* frame, and
    both call sites in `run_main_flow` pass **native-unit** sizes with the
    native `RegionSet`, because `freeze.json` is a native-unit artefact and
    `result.json` must quote the same numbers.
    """
    part = np.asarray(part, dtype=np.int64)
    area = np.asarray(size_x, dtype=np.float64) * np.asarray(size_y, dtype=np.float64)
    counts = np.bincount(part, minlength=rs.k)[:rs.k]
    cell_area = np.bincount(part, weights=area, minlength=rs.k)[:rs.k]
    region_area = np.empty(rs.k, dtype=np.float64)
    for rid, region in enumerate(rs.regions):
        rects = np.asarray(region.rects, dtype=np.float64).reshape(-1, 4)
        region_area[rid] = float(np.sum((rects[:, 2] - rects[:, 0])
                                        * (rects[:, 3] - rects[:, 1])))
    return {"region_cell_count": counts.astype(int).tolist(),
            "region_cell_area": cell_area.tolist(),
            "region_area": region_area.tolist(),
            "region_utilization": (cell_area / region_area).tolist()}


def freeze_record(*, iteration, reason, overflow, tau, tau_rel, churn, k,
                  io_soft, membership_npz, soft_npz, repaired_empty_regions,
                  gp_iterations_soft, density_weight_soft, stats):
    record = {"schema_version": FREEZE_SCHEMA_VERSION,
              "iteration": int(iteration), "reason": str(reason),
              "overflow": float(overflow), "tau": float(tau),
              "tau_rel": float(tau_rel),
              "churn": None if churn is None else float(churn),
              "k": int(k), "io_soft": int(io_soft),
              "membership_npz": str(membership_npz), "soft_npz": str(soft_npz),
              "repaired_empty_regions": list(repaired_empty_regions),
              "gp_iterations_soft": int(gp_iterations_soft),
              "density_weight_soft": float(density_weight_soft)}
    record.update(stats)
    return record
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_freeze.py -v`
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/freeze.py tests/test_freeze.py
git commit -m "feat(freeze): cell-centre membership, churn window and freeze criterion

argmax over -region_sdf_l1 at the cell centre, a windowed churn tracker, the
overflow/tau_rel/churn triple gate, empty-region repair (which would
otherwise crash PlaceDB.calc_num_filler_for_fence_region) and the freeze.json
record builder.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Phase-3 builder — fences, escape cell, warm start, density-weight clamp

**Files:**
- Create: `src/ioplace/fence_phase.py`
- Test: `tests/test_fence_phase.py`

**Interfaces:**
- Consumes: `ioplace.fence_inject.inject_fence_regions`; `ioplace.drivers.run_placement_two_stage._pick_escape_cell`; `ioplace.drivers.run_placement._load_dreamplace`; `ioplace.drivers.run_placement_io._install_attribute`; `ioplace.init_pos.apply_init` (Task 2); `ioplace.artifacts.placedb_identity_sha256` (Task 1).
- Produces:
  - `clamp_density_weight(model, reference, lo=0.25, hi=4.0) -> dict`
  - `install_density_weight_clamp(cleanup, reference, log, *, lo=0.25, hi=4.0) -> None`
  - `build_fence_placedb(config_json, region_set, part, positions, *, dp_seed=None, deterministic=None) -> (params, placedb, info)` where `info` has keys `escape_cell`, `escape_from`, `placedb_sha256`, `shift_factor`, `scale_factor`, `die_native`, `die_scaled`, `init`

**Why the clamp is installed on the class, not the instance.** `PlaceObj.initialize_density_weight` recomputes the density weight from the grad-norm ratio *at the seed* (`PlaceObj.py:776-834`); at a near-legal warm start the density gradient is small, so `params.density_weight * ||∇WL||₁ / ||∇D||₁` can over-weight density and blow the seed apart (spec §3 phase 3). It is called lazily from `NonLinearPlace.py:399-400`, and `placer.model` does not exist until `NonLinearPlace.__call__` assigns it at `NonLinearPlace.py:236` — so there is no instance to wrap beforehand. The iteration callback is not an option either: it fires at `NonLinearPlace.py:521`, *after* `make_parameter_update()` at `NonLinearPlace.py:454/462`, i.e. one full step would already have been taken at the unclamped weight. The only hook that runs before that step is a scoped patch of the class method, installed through the same `_install_attribute` save/restore used everywhere else in the drivers and torn down by `_io_cleanup`.

In fence mode `density_weight` is a `(K+1,)` vector `u * s` and the overflow-based update recomputes it from `density_weight_u` (`PlaceObj.py:862-875`), so clamping the product alone would be undone at the first update: the clamp scales `density_weight_u` by the same elementwise ratio and recomputes `density_weight_step_size` exactly as `PlaceObj.py:817` does.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fence_phase.py`:

```python
import os
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.fence_phase import build_fence_placedb, clamp_density_weight

GCD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results/route_feedback_20260914/gcd.json")


class _FlatModel:
    def __init__(self, value):
        self.density_weight = torch.tensor([value], dtype=torch.float64)


class _FenceModel:
    def __init__(self, values):
        self.density_weight = torch.tensor(values, dtype=torch.float64)
        self.density_weight_u = torch.tensor(values, dtype=torch.float64) * 2.0
        self.density_weight_step_size_inc_low = 1.03
        self.density_weight_step_size = 0.0


def test_clamp_is_a_noop_inside_the_band():
    model = _FlatModel(2.0)
    log = clamp_density_weight(model, reference=1.0)
    assert model.density_weight.tolist() == [2.0]
    assert log["bound"] is False and log["lo_abs"] == 0.25 and log["hi_abs"] == 4.0
    # lo_abs/hi_abs are absolute bounds, not the [0.25, 4] multipliers: they
    # only coincide with them when reference == 1 (amendment D-14).
    scaled = clamp_density_weight(_FlatModel(2.0), reference=2.0)
    assert scaled["lo_abs"] == 0.5 and scaled["hi_abs"] == 8.0
    assert scaled["bound"] is False


def test_clamp_binds_above_and_below():
    high = _FlatModel(40.0)
    assert clamp_density_weight(high, reference=1.0)["bound"] is True
    assert high.density_weight.tolist() == [4.0]
    low = _FlatModel(0.01)
    assert clamp_density_weight(low, reference=1.0)["bound"] is True
    assert low.density_weight.tolist() == [0.25]


def test_clamp_keeps_the_fence_subgradient_state_consistent():
    model = _FenceModel([0.5, 8.0, 1.0])
    log = clamp_density_weight(model, reference=1.0)
    assert model.density_weight.tolist() == [0.5, 4.0, 1.0]
    # u is scaled by the same elementwise ratio: [1.0, 16.0, 2.0] * [1, .5, 1]
    assert model.density_weight_u.tolist() == [1.0, 8.0, 2.0]
    expected = 0.03 * float(torch.tensor([1.0, 8.0, 2.0]).norm(p=2))
    assert model.density_weight_step_size == pytest.approx(expected)
    assert log["before"] == [0.5, 8.0, 1.0] and log["after"] == [0.5, 4.0, 1.0]


def test_clamp_rejects_a_nonpositive_reference():
    with pytest.raises(ValueError, match="reference"):
        clamp_density_weight(_FlatModel(1.0), reference=0.0)


@pytest.mark.slow
def test_build_fence_placedb_injects_fences_and_scales_the_warm_start(tmp_path):
    from ioplace.artifacts import Positions, placedb_identity_sha256
    from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
    if not os.path.exists(GCD):
        pytest.skip("GCD benchmark required")
    params0, db0 = _load_dreamplace(GCD)
    m, n_phys = db0.num_movable_nodes, db0.num_physical_nodes
    die = (float(db0.xl), float(db0.yl), float(db0.xh), float(db0.yh))
    rs = get_regions_for(die, 4, "grid", 0)
    sha = placedb_identity_sha256(db0)
    from ioplace.region_grid import RegionGrid
    cx = np.asarray(db0.node_x[:m], np.float64) + np.asarray(db0.node_size_x[:m], np.float64) / 2
    cy = np.asarray(db0.node_y[:m], np.float64) + np.asarray(db0.node_size_y[:m], np.float64) / 2
    part = RegionGrid(rs).region_of_points(cx, cy).astype(np.int32)
    seed = Positions(node_x=np.asarray(db0.node_x[:n_phys], np.float64),
                     node_y=np.asarray(db0.node_y[:n_phys], np.float64),
                     die=die, shift_factor=(0., 0.), scale_factor=1.,
                     placedb_sha256=sha, kind="soft")

    params, db, info = build_fence_placedb(GCD, rs, part, seed)
    assert len(db.regions) == 4
    assert info["placedb_sha256"] == sha
    escape = info["escape_cell"]
    assert db.node2fence_region_map[escape] == 4          # implicit no-fence bucket
    mask = np.ones(m, dtype=bool); mask[escape] = False
    assert np.array_equal(db.node2fence_region_map[:m][mask], part[mask])
    # warm start survives initialize()'s scale(): native -> (v - shift) * scale
    shift, scale = info["shift_factor"], info["scale_factor"]
    expected = (seed.node_x[:m] - shift[0]) * scale
    assert np.allclose(np.asarray(db.node_x[:m], np.float64), expected, atol=1e-3)
    assert params.random_center_init_flag == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_fence_phase.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.fence_phase'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/fence_phase.py`:

```python
"""Phase-3 (fence GP) PlaceDB construction and density-weight clamp
(design v2 sec 3 phase 3).

Fence data may only be injected between `read()` and `initialize()`
(fence_inject.py:9-16), so phase 3 always builds a *fresh* PlaceDB; that is
what forces the two-instance split in the first place.
"""
import numpy as np

from ioplace.artifacts import placedb_identity_sha256
from ioplace.drivers.run_placement import _load_dreamplace
from ioplace.drivers.run_placement_io import _install_attribute
from ioplace.drivers.run_placement_two_stage import _pick_escape_cell
from ioplace.fence_inject import inject_fence_regions
from ioplace.init_pos import apply_init

DENSITY_CLAMP_LO = 0.25
DENSITY_CLAMP_HI = 4.0


def clamp_density_weight(model, reference, lo=DENSITY_CLAMP_LO, hi=DENSITY_CLAMP_HI):
    """Clamp `model.density_weight` to `[lo*reference, hi*reference]`.

    In fence mode `density_weight == density_weight_u * s` and the
    overflow-based update recomputes it from `density_weight_u`
    (PlaceObj.py:862-875), so the clamp scales `u` by the same elementwise
    ratio and recomputes the step size exactly as PlaceObj.py:817 does --
    otherwise the very first update would undo the clamp.

    The returned `lo_abs`/`hi_abs` are the **absolute** bounds
    (`lo*reference`, `hi*reference`), not the multipliers; they coincide only
    when `reference == 1` (pre-flight amendment D-14).
    """
    import torch
    reference = float(reference)
    if not reference > 0.0:
        raise ValueError(f"reference density weight must be positive, got {reference}")
    low, high = lo * reference, hi * reference
    with torch.no_grad():
        before = model.density_weight.detach().clone()
        after = before.clamp(min=low, max=high)
        # dtype-safe floor: GCD's placedb.dtype is float32, where 1e-300
        # underflows to 0.0 and the guard would rest entirely on torch.where's
        # mask (pre-flight amendment D-10).
        tiny = torch.finfo(before.dtype).tiny
        ratio = torch.where(before > 0, after / before.clamp_min(tiny),
                            torch.ones_like(before))
        model.density_weight.copy_(after)
        u = getattr(model, "density_weight_u", None)
        if u is not None:
            u.mul_(ratio)
            model.density_weight_step_size = (
                model.density_weight_step_size_inc_low - 1.0) * float(u.norm(p=2))
    return {"reference": reference, "lo_abs": float(low), "hi_abs": float(high),
            "before": [float(v) for v in before.reshape(-1)],
            "after": [float(v) for v in after.reshape(-1)],
            "bound": bool(torch.any(after != before))}


def install_density_weight_clamp(cleanup, reference, log, *, lo=DENSITY_CLAMP_LO,
                                 hi=DENSITY_CLAMP_HI):
    """Scope a clamp around every `PlaceObj.initialize_density_weight` call.

    The class, not the instance, is patched: `placer.model` only exists once
    `NonLinearPlace.__call__` assigns it (NonLinearPlace.py:236), the density
    weight is initialised at NonLinearPlace.py:399-400, and the iteration
    callback only fires at NonLinearPlace.py:521 -- after
    `make_parameter_update()` (NonLinearPlace.py:454/462) has already taken a
    step at the unclamped weight. `_install_attribute` restores the original
    function when `cleanup` closes, including on an exception.

    The patch is **process-global**: every `PlaceObj` built in this interpreter
    sees it while it is installed, so it must only ever be installed inside an
    `_io_cleanup()` scope (which is what guarantees the restore) and never
    around code that runs a second, unrelated placement concurrently.
    `initialize_density_weight` has two call sites (NonLinearPlace.py:400 and
    :783), so `log` may collect more than one entry per run (pre-flight
    amendment D-4).
    """
    import PlaceObj
    original = PlaceObj.PlaceObj.initialize_density_weight

    def wrapped(self, params, placedb):
        weight = original(self, params, placedb)
        log.append(clamp_density_weight(self, reference, lo, hi))
        return weight

    _install_attribute(cleanup, PlaceObj.PlaceObj, "initialize_density_weight", wrapped)


def build_fence_placedb(config_json, region_set, part, positions, *,
                        dp_seed=None, deterministic=None):
    """Read a fresh PlaceDB, inject fences from the frozen membership, apply the
    escape-cell workaround, warm start from `positions`, then initialize()."""
    params, placedb = _load_dreamplace(config_json)
    assert params.enable_fillers == 1, "fence mode requires enable_fillers"
    if dp_seed is not None:
        params.random_seed = dp_seed
    if deterministic is not None:
        params.deterministic_flag = deterministic

    k = region_set.k
    m = placedb.num_movable_nodes
    part = np.asarray(part, dtype=np.int32)
    if len(part) != m:
        raise ValueError(f"membership has {len(part)} entries, expected {m}")
    die_native = (float(placedb.xl), float(placedb.yl),
                  float(placedb.xh), float(placedb.yh))
    span = max(die_native[2] - die_native[0], die_native[3] - die_native[1])
    if not np.allclose(region_set.die, die_native, atol=1e-6 * span, rtol=0.0):
        raise ValueError(f"region set die {region_set.die} != placedb die {die_native}; "
                         "regions.json must be in native post-read units")
    sha = placedb_identity_sha256(placedb)
    if positions.placedb_sha256 != sha:
        raise ValueError("warm-start positions were produced for a different netlist "
                         f"({positions.placedb_sha256} != {sha})")

    inject_fence_regions(placedb, region_set, part)
    # DREAMPlace always allocates an implicit "no fence" bucket (region_id == k)
    # and calls calc_num_filler_for_fence_region on it; our regions tile the whole
    # die, so that bucket is empty and np.percentile of the empty slice raises
    # IndexError at PlaceDB.py:687 under numpy 1.26.4 (pre-flight E-3; the older
    # "int(round(nan)) -> ValueError at :729" wording is wrong). Same workaround,
    # same helper, as run_placement_two_stage.py:192-233.
    escape = _pick_escape_cell(placedb.node2fence_region_map, part,
                               placedb.node_size_x, placedb.node_size_y, k)
    escape_from = int(placedb.node2fence_region_map[escape])
    placedb.node2fence_region_map[escape] = k

    init = apply_init(placedb, params, "seed", positions=positions)
    placedb.initialize(params)

    info = {"escape_cell": int(escape), "escape_from": escape_from,
            "placedb_sha256": sha,
            "shift_factor": (float(params.shift_factor[0]), float(params.shift_factor[1])),
            "scale_factor": float(params.scale_factor),
            "die_native": die_native,
            "die_scaled": (float(placedb.xl), float(placedb.yl),
                           float(placedb.xh), float(placedb.yh)),
            "init": init}
    return params, placedb, info
```

- [ ] **Step 4: Run the fast tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_fence_phase.py -v -m "not slow"`
Expected: PASS — 4 passed, 1 deselected.

- [ ] **Step 5: Run the slow test on GCD**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_fence_phase.py -v -m slow`
Expected: PASS — 1 passed. (Takes ~30 s; the run reads the GCD LEF/DEF and calls `initialize()` but no GP.)

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/fence_phase.py tests/test_fence_phase.py
git commit -m "feat(fence-phase): phase-3 PlaceDB builder and density-weight clamp

Fresh read -> inject_fence_regions -> escape-cell workaround -> warm start ->
initialize(). The density-weight clamp scopes PlaceObj.initialize_density_weight
(placer.model does not exist before NonLinearPlace.__call__, and the iteration
callback fires only after the first optimizer step) and keeps density_weight_u
and the step size consistent so the fence-mode update cannot undo it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Normalisation adapter (the only seam to P-H)

**Files:**
- Create: `src/ioplace/norm_adapter.py`
- Test: `tests/test_norm_adapter.py`

**Interfaces:**
- Consumes: `ioplace.schedules.ScheduleState`; `ioplace.ops.ft_callback.publish_atomic` (`src/ioplace/ops/ft_callback.py:18-49`); P-H's `ioplace.norm.TermNormalizer` / `VersionPair` / `parse_target_shares` (`src/ioplace/norm.py:169-188,190-630,108-119`), `ioplace.norm_trace.NormTraceWriter` (`src/ioplace/norm_trace.py:27-54`), `ioplace.ops.norm_terms.IoNormTerm` / `FtNormTerm` (`src/ioplace/ops/norm_terms.py:10-28`).
- Produces:
  - `NORM_POLICIES = TermNormalizer.POLICIES` — `("legacy", "grandplan", "adaptive")`
  - `LEGACY_TRACE_NAME = "legacy_trace.jsonl"`, `NORM_TRACE_NAME = "norm_trace.jsonl"`
  - `make_norm_adapter(policy, **config) -> LegacyNormAdapter | TermNormalizerAdapter`
  - `class LegacyNormAdapter`, `class TermNormalizerAdapter` — both implementing the adapter protocol below.

**Adapter protocol** — the driver uses only these members, so `--norm-policy` is a one-word change with no other edit to `run_main_flow.py`:

| Member | Meaning |
|---|---|
| `policy: str` | name recorded in `result.json` and in every trace row |
| `active: bool` | the term has been switched on (the `ScheduleState` latch, in both adapters) |
| `tau: float`, `tau_rel: float` | soft-assign temperature, absolute and relative to `L_R` |
| `lambda_io: float`, `kappa_ft: float` | coefficients the driver feeds to `IoTerm`/`FtTerm`; `FtTerm.forward` multiplies the FT part by `lambda_io * kappa_ft`, so `kappa_ft` is `lambda_ft / lambda_io` |
| `obj_version: int`, `refreshed_version: int` | objective version and the version the Nesterov cache was last refreshed at; `dp_hook.install_version_invariant(optimizer, adapter)` reads both off the adapter |
| `begin_iteration(iteration, overflow, gamma) -> bool` | continuous update; `True` when a discrete change happened |
| `probe(iteration, pos, *, io_term, ft_term, wirelength_op, ecc_max, gamma) -> dict` | one isolated backward per term, one atomic coefficient publication, one `obj_version` bump; returns the policy's trace row |
| `write_trace_row(row, extra) -> dict | None` | the adapter, not the driver, owns the trace file; `extra` is the driver's per-probe `{io_count, ft_count, churn, …}` |
| `trace_path: str | None`, `close() -> None` | where this policy's trace went, and its teardown |
| `needs_refresh() -> bool`, `mark_refreshed() -> None` | Nesterov-secant discipline (`dp_hook.py:36-59`) |

**Why two classes, not one (pre-flight B-1/A-10/A-11, cross-plan ruling 2 as revised).** The original text deferred `TermNormalizerAdapter` to P-H and raised `NotImplementedError` for `grandplan`/`adaptive`. Both halves of that were wrong: P-H's plan never touches `norm_adapter.py` (its Task 7 modifies `run_placement_io.py`, `run_placement.py`, `docs/dev-env.md` and `tests/test_norm_driver.py`), so the deferral pointed at nobody, and spec §4's policy-A/B ablation would have been permanently unreachable from `run_main_flow`. Ruling 2's literal form — "build on `TermNormalizer` directly, no adapter" — is not implementable either: the live `TermNormalizer` (`src/ioplace/norm.py`) owns coefficients and nothing else. It has no τ schedule, no `L_R`, no activation trigger, no `lambda_io`/`kappa_ft` names, and its `probe(iteration, pos, wl_fn, ctx, probe_terms=None)` signature differs from the driver's. Something must still map overflow onto τ and latch activation, and that something is `ScheduleState`. So `TermNormalizerAdapter` keeps a `ScheduleState` **solely** for τ/ρ/activation (`update_continuous`) and hands every coefficient to a `TermNormalizer`.

**Division of labour inside `TermNormalizerAdapter`.**

- `ScheduleState` — τ from `tau_rel_from_overflow` (`schedules.py:9-14`), ρ, and the single activation latch. Its own `lambda_io`/`kappa_ft`/`Cmax` are never read.
- `TermNormalizer` — every coefficient, the Lipschitz cap, `Cmax`, the cancellation ratio, and `norm_trace.jsonl`. Rows are emitted by `mark_refreshed()`, i.e. only once the objective version is live in the optimizer's cache.
- `VersionPair(state, normalizer)` (`norm.py:169-188`) — one version pair covering both, so a single `install_version_invariant(optimizer, adapter)` sees an activation bump from `update_continuous` *and* a coefficient bump from `transaction()`. Stacking two invariant wrappers would not work: `refresh_nesterov_secant` unwraps exactly one `__wrapped__` level.
- Activation is synchronised **down** from the schedule every iteration. `TermNormalizer._activate` only runs inside `weights()`/`transaction()`, i.e. on the probe cadence, so a term left to latch itself would set `it_activate` tens of iterations late and restart policy A's `activation_ramp` (and policy B's ramped share) from the first probe instead of from the schedule's own activation.
- Terms are registered on the **first probe**, which is the first moment the driver hands over `io_term`/`ft_term` and the FT curvature `ecc_max`. Declared curvatures (design sec 4): IO 1, FT `ecc_max`.
- Target shares (policy B) default to the legacy knobs: `io <- rho_max`, `ft <- rho_max * f_ft_max`. `TermNormalizer`'s shares are fractions of the *total* force `G = ‖∇WL‖ + Σ λ_t‖∇T_t‖`, and `rho_max` is already "IO force as a fraction of the WL force", while legacy's `f_ft_max` is the FT force as a fraction of the *IO* force — hence the product. `--norm-target-share io=…,ft=…` (`norm.parse_target_shares`) overrides either.
- The τ_rel-driven FT window (`--tau-start`/`--tau-full`, `schedules.ft_activation_ramp`) applies to `--norm-policy legacy` only. Policies A and B replace it with the unified iteration ramp plus the target share, which is the whole point of design sec 4.

**One trace schema per file name (pre-flight A-9).** `norm_trace.jsonl` is spec §1's artefact and P-H's `NormTraceWriter` validates every row against `norm_trace.ROW_FIELDS`, rejecting extra or missing keys. The legacy path's row is `publish_atomic`'s dict, which shares almost none of those keys, so it goes to `legacy_trace.jsonl` instead. The driver never formats a row: it calls `adapter.write_trace_row(row, sample)` and the adapter decides. Under `grandplan`/`adaptive` that call is a no-op (the normalizer already wrote the validated row) and the driver's per-probe extras reach `result.json` through `soft_summary["probe_samples"]` instead.

**Checked against P-H (2026-09-19, at `da83162`).** `NORM_POLICIES` is bound to `TermNormalizer.POLICIES` rather than re-spelled, so it cannot drift. Under `legacy` the normalizer owns no coefficient maths and delegates `obj_version`/`refreshed_version` to a `ScheduleState` (`norm.py:566-622`) — exactly what `LegacyNormAdapter` does here — so the two legacy paths agree by construction and `LegacyNormAdapter` keeps `publish_atomic` as its single source of truth (`tests/test_norm_legacy_adapter.py`'s golden trajectory is the regression lock). P-C's `GroupingWeight` (its Task 4) uses `norm.py`'s pure helpers `grandplan_weight` and `ema_update` at a different level; no name clash. P-H Task 7 (the `run_placement_io.py` wiring) had **not** landed when this task was written, so the wiring below is derived from `norm.py` itself; if it has landed by the time you implement, cross-check `probe`/`transaction`/`mark_refreshed` ordering against that driver and report any difference rather than silently diverging.

- [ ] **Step 1: Write the failing test**

Create `tests/test_norm_adapter.py`:

```python
import json

import pytest
torch = pytest.importorskip("torch")

from ioplace.norm_adapter import (NORM_POLICIES, LegacyNormAdapter,
                                  TermNormalizerAdapter, make_norm_adapter)
from ioplace.norm_trace import ROW_FIELDS, read_norm_trace


class _FakeIoTerm:
    num_movable, num_nodes = 2, 3

    def __call__(self, pos, tau, lambda_io, *args):
        return lambda_io * (pos ** 2).sum()


def _wirelength(pos):
    return (3.0 * pos).abs().sum()


def _pos():
    # x = pos[:3], y = pos[3:]; entries 2 and 5 are the fixed/filler slots the
    # normalizer's mask zeroes, so ||grad WL||_1 = 4*3 = 12 and
    # ||grad IO||_1 = |2*[1,2,4,5]| = 24 for every probe below.
    return torch.arange(6, dtype=torch.float64) + 1.0


def _config(**override):
    config = dict(L_R=100.0, rho_max=0.1, tau_hi=0.30, tau_lo=0.03, of_on=2.0,
                  of_end=0.5, of_full=1.0, f_ft_max=0.0, ft_ramp_mode="window",
                  tau_start=0.12, tau_full=0.05)
    config.update(override)
    return config


def _probe(adapter, iteration):
    return adapter.probe(iteration, _pos(), io_term=_FakeIoTerm(), ft_term=None,
                         wirelength_op=_wirelength, ecc_max=0.0, gamma=1.0)


def _activated(policy, **override):
    """Activate at iteration 0 so policy A's 20-iteration activation ramp is
    complete by the first probe at 50. Without the adapter's activation sync
    the term would latch at the probe itself and ramp to exactly 0."""
    adapter = make_norm_adapter(policy, **_config(**override))
    adapter.begin_iteration(0, overflow=1.5, gamma=1.0)
    adapter.mark_refreshed()
    adapter.begin_iteration(50, overflow=1.5, gamma=1.0)
    return adapter


def test_policies_are_declared():
    assert NORM_POLICIES == ("legacy", "grandplan", "adaptive")


def test_legacy_adapter_activates_once_and_reports_tau():
    adapter = make_norm_adapter("legacy", **_config())
    assert adapter.policy == "legacy"
    assert adapter.begin_iteration(0, overflow=3.0, gamma=1.0) is False
    assert adapter.active is False and adapter.lambda_io == 0.0
    assert adapter.begin_iteration(10, overflow=1.5, gamma=1.0) is True
    assert adapter.active is True
    assert adapter.begin_iteration(20, overflow=1.5, gamma=1.0) is False
    assert adapter.tau == pytest.approx(adapter.tau_rel * 100.0)


def test_probe_bumps_obj_version_exactly_once_and_needs_a_refresh():
    adapter = _activated("legacy")
    before = adapter.obj_version
    row = _probe(adapter, 50)
    assert adapter.obj_version == before + 1
    assert adapter.needs_refresh() is True
    assert adapter.refreshed_version != adapter.obj_version
    adapter.mark_refreshed()
    assert adapter.needs_refresh() is False
    assert adapter.refreshed_version == adapter.obj_version
    row = adapter.write_trace_row(row, {"iteration": 50, "policy": "legacy"})
    for key in ("grad_l1_wl", "grad_l1_io", "grad_l1_ft", "ratio_ema",
                "lambda_io", "obj_version", "tau", "tau_rel", "kappa_ft"):
        assert key in row
    assert row["grad_l1_wl"] > 0 and row["grad_l1_io"] > 0
    assert adapter.lambda_io > 0


def test_unknown_policy_and_unknown_knob_are_rejected():
    with pytest.raises(ValueError, match="norm policy"):
        make_norm_adapter("bogus", **_config())
    with pytest.raises(TypeError, match="unexpected keyword"):
        make_norm_adapter("legacy", bogus_knob=1, **_config())
    with pytest.raises(ValueError, match="LegacyNormAdapter"):
        TermNormalizerAdapter("legacy", **_config())


def test_grandplan_adapter_normalises_a_positive_lambda_io(tmp_path):
    """Policy A end to end: one probe of a term with a real positive gradient
    must leave a non-zero lambda_io and one schema-valid norm_trace row."""
    adapter = _activated("grandplan", out_dir=str(tmp_path))
    assert isinstance(adapter, TermNormalizerAdapter)
    row = _probe(adapter, 50)
    assert adapter.lambda_io > 0.0
    assert adapter.kappa_ft == 0.0                      # no FT term registered
    assert row["policy"] == "grandplan" and set(row) == set(ROW_FIELDS)
    assert row["grad_l1_wl"] == pytest.approx(12.0)
    assert row["terms"]["io"]["grad_l1"] == pytest.approx(24.0)
    assert row["terms"]["io"]["ratio_ema"] == pytest.approx(0.5)
    assert row["terms"]["io"]["lam"] == pytest.approx(adapter.lambda_io)
    # wt = activation_ramp(50, 0, 20) * grandplan_weight(...) = 1.0 * 0.05, so
    # lambda_io = wt * ratio_ema = 0.025, well under the Lipschitz cap.
    assert adapter.lambda_io == pytest.approx(0.025)
    # the row reaches disk only once the Nesterov cache has been refreshed
    assert not (tmp_path / "norm_trace.jsonl").read_text()
    adapter.mark_refreshed()
    adapter.close()
    rows = read_norm_trace(str(tmp_path / "norm_trace.jsonl"))
    assert len(rows) == 1
    assert rows[0]["obj_version"] == rows[0]["refreshed_version"]


def test_term_normalizer_adapter_bumps_obj_version_once_per_probe(tmp_path):
    adapter = _activated("grandplan", out_dir=str(tmp_path))
    assert adapter.needs_refresh() is False
    before = adapter.obj_version
    _probe(adapter, 50)
    # VersionPair sums the ScheduleState's counter and the normalizer's, so one
    # transaction is one bump even though two objects carry versions.
    assert adapter.obj_version == before + 1
    assert adapter.needs_refresh() is True
    adapter.mark_refreshed()
    assert adapter.needs_refresh() is False
    assert adapter.refreshed_version == adapter.obj_version
    adapter.begin_iteration(100, overflow=1.5, gamma=1.0)
    _probe(adapter, 100)
    assert adapter.obj_version == before + 2
    adapter.mark_refreshed()
    adapter.close()
    assert len(read_norm_trace(str(tmp_path / "norm_trace.jsonl"))) == 2


def test_adaptive_policy_constructs_and_bootstraps_its_coefficient(tmp_path):
    """Policy B: target shares default to the legacy knobs (io <- rho_max,
    ft <- rho_max*f_ft_max) and the first update takes adaptive_lambda's
    bootstrap branch, so one probe leaves a positive coefficient."""
    adapter = _activated("adaptive", out_dir=str(tmp_path))
    assert adapter.policy == "adaptive" and adapter.normalizer.policy == "adaptive"
    assert adapter.target_shares == {"io": 0.1, "ft": 0.0}
    _probe(adapter, 50)
    # bootstrap: target_share * ||grad WL|| / ||grad T|| = 0.1 * 12 / 24
    assert adapter.lambda_io == pytest.approx(0.05)
    adapter.mark_refreshed()
    adapter.close()
    override = make_norm_adapter("adaptive", target_shares="io=0.3,ft=0.05",
                                 **_config())
    assert override.target_shares == {"io": 0.3, "ft": 0.05}
    assert override.trace_path is None


def test_each_policy_writes_its_own_trace_file(tmp_path):
    """A-9: one schema per file name. norm_trace.jsonl is the design sec 4
    schema NormTraceWriter validates; the legacy row is publish_atomic's own
    dict and goes to legacy_trace.jsonl."""
    legacy_dir, norm_dir = str(tmp_path / "legacy"), str(tmp_path / "grandplan")
    legacy = _activated("legacy", out_dir=legacy_dir)
    assert legacy.trace_path.endswith("legacy_trace.jsonl")
    written = legacy.write_trace_row(_probe(legacy, 50),
                                     {"io_count": 7, "churn": 0.0})
    legacy.mark_refreshed()
    legacy.close()
    lines = [json.loads(line) for line
             in open(legacy.trace_path).read().splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0]["io_count"] == 7
    assert lines[0]["grad_l1_io"] > 0 and written["policy"] == "legacy"
    assert not (tmp_path / "legacy" / "norm_trace.jsonl").exists()

    grandplan = _activated("grandplan", out_dir=norm_dir)
    assert grandplan.trace_path.endswith("norm_trace.jsonl")
    # the driver's extras must not contaminate the validated schema
    assert grandplan.write_trace_row(_probe(grandplan, 50),
                                     {"io_count": 7, "churn": 0.0}) is None
    grandplan.mark_refreshed()
    grandplan.close()
    rows = read_norm_trace(grandplan.trace_path)
    assert len(rows) == 1 and set(rows[0]) == set(ROW_FIELDS)
    assert not (tmp_path / "grandplan" / "legacy_trace.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_norm_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.norm_adapter'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/norm_adapter.py`:

```python
"""The single seam between the v2 main flow and term normalisation.

`run_main_flow.py` must never import `schedules.py` or `norm.py` directly:
subproject P-H replaced the three ad-hoc normalisation paths with
`src/ioplace/norm.py`'s `TermNormalizer` (design v2 sec 4), and this module is
what lets one driver drive either path with a one-word `--norm-policy` change.

Two adapters, one protocol (the plan's Task 5 table):

* `LegacyNormAdapter` -- `schedules.ScheduleState` plus
  `ops/ft_callback.publish_atomic`: the recorded pre-v2 coefficients,
  bit-for-bit. Writes `legacy_trace.jsonl`.
* `TermNormalizerAdapter` -- policies `grandplan` (design sec 4 policy A) and
  `adaptive` (policy B) over the live `norm.TermNormalizer`. A `ScheduleState`
  survives here too, but *only* as the tau / rho / activation schedule:
  `TermNormalizer` owns no temperature and no activation trigger, so something
  still has to map overflow onto tau. Every coefficient comes from the
  normalizer. Writes `norm_trace.jsonl` through `norm_trace.NormTraceWriter`.
"""
import json
import os

from ioplace.norm import TermNormalizer, VersionPair, parse_target_shares
from ioplace.norm_trace import NormTraceWriter
from ioplace.ops.ft_callback import publish_atomic
from ioplace.ops.norm_terms import FtNormTerm, IoNormTerm
from ioplace.schedules import ScheduleState

#: Bound to P-H's tuple rather than re-spelled, so the two cannot drift.
NORM_POLICIES = TermNormalizer.POLICIES

LEGACY_TRACE_NAME = "legacy_trace.jsonl"
NORM_TRACE_NAME = "norm_trace.jsonl"

#: Keys `_ScheduleBacked` -- and therefore both adapters -- understands.
_SCHEDULE_KEYS = ("L_R", "rho_max", "tau_hi", "tau_lo", "of_on", "of_end",
                  "of_full", "f_ft_max", "ft_ramp_mode", "tau_start",
                  "tau_full", "ema", "c_lip", "n_ramp", "kappa_max", "eps_rel",
                  "out_dir")

#: Keys only `TermNormalizerAdapter` understands. The driver builds one kwarg
#: set for every policy, so under `legacy` these are accepted and dropped
#: rather than raising -- `--norm-policy` must stay a one-word change.
_NORMALIZER_KEYS = ("num_movable", "num_nodes", "norm_p", "probe_every", "wt0",
                    "wt_step", "ramp_period", "wt_max", "momentum",
                    "target_shares", "track_cancellation")


class _ScheduleBacked(object):
    """Shared tau / rho / activation plumbing and trace-file placement.

    Both adapters keep a `ScheduleState`: it is the only thing in the tree that
    maps overflow onto the soft-assign temperature
    (`schedules.tau_rel_from_overflow`) and latches activation, and
    `TermNormalizer` deliberately owns neither.
    """

    TRACE_NAME = None

    def __init__(self, *, L_R, rho_max, tau_hi, tau_lo, of_on, of_end, of_full,
                 f_ft_max, ft_ramp_mode, tau_start, tau_full, ema=0.5,
                 c_lip=1.0, n_ramp=20, kappa_max=100.0, eps_rel=1e-3,
                 out_dir=None):
        if not L_R > 0:
            raise ValueError("L_R must be positive")
        self.L_R = float(L_R)
        self.state = ScheduleState(
            rho_max=rho_max, tau_hi=tau_hi, tau_lo=tau_lo, of_on=of_on,
            of_end=of_end, of_full=of_full, n_ramp=n_ramp, c_lip=c_lip,
            f_ft_max=f_ft_max, ft_ramp_mode=ft_ramp_mode, tau_start=tau_start,
            tau_full=tau_full, kappa_max=kappa_max, eps_rel=eps_rel, ema=ema)
        self.out_dir = out_dir
        if out_dir is None:
            self.trace_path = None
        else:
            os.makedirs(out_dir, exist_ok=True)
            self.trace_path = os.path.join(out_dir, self.TRACE_NAME)

    @property
    def active(self):
        return bool(self.state.active)

    @property
    def tau(self):
        return float(self.state.tau)

    @property
    def tau_rel(self):
        return float(self.state.tau) / self.L_R


class LegacyNormAdapter(_ScheduleBacked):
    """The pre-v2 path: `ScheduleState` plus the seven-step atomic FT
    transaction (schedules.py:178-237, ops/ft_callback.py:18-49). Always uses
    the atomic discipline -- with `ft_term=None` `publish_atomic` degenerates
    to the IO-only ratio update while still bumping `obj_version` exactly
    once, which is what the Nesterov invariant
    (dp_hook.install_version_invariant) requires."""

    policy = "legacy"
    TRACE_NAME = LEGACY_TRACE_NAME

    def __init__(self, **config):
        super().__init__(**config)
        self._trace = None if self.trace_path is None else open(self.trace_path, "w")

    @property
    def lambda_io(self):
        return float(self.state.lambda_io)

    @property
    def kappa_ft(self):
        return float(self.state.kappa_ft)

    @property
    def obj_version(self):
        return int(self.state.obj_version)

    @property
    def refreshed_version(self):
        # dp_hook.install_version_invariant reads obj_version and
        # refreshed_version off whatever object it is handed, so the adapter
        # exposes both and the driver never reaches through to `.state`.
        return int(self.state.refreshed_version)

    def begin_iteration(self, iteration, overflow, gamma):
        return bool(self.state.update_continuous(iteration, overflow, self.L_R, gamma))

    def probe(self, iteration, pos, *, io_term, ft_term, wirelength_op,
              ecc_max, gamma):
        row = publish_atomic(self.state, io_term, ft_term, wirelength_op, pos,
                             iteration, self.tau_rel, ecc_max, gamma)
        row.update(policy=self.policy, iteration=int(iteration), tau=self.tau,
                   tau_rel=self.tau_rel,
                   refreshed_version=int(self.state.refreshed_version))
        return row

    def write_trace_row(self, row, extra):
        """One `legacy_trace.jsonl` line per probe: `publish_atomic`'s keys plus
        whatever the driver measured in the same callback. This file is
        deliberately *not* `norm_trace.jsonl` -- that name belongs to design
        sec 4's schema, which `norm_trace.NormTraceWriter` validates and this
        row does not satisfy (pre-flight amendment A-9)."""
        record = dict(row)
        record.update(extra)
        if self._trace is not None:
            self._trace.write(json.dumps(record) + "\n")
            self._trace.flush()
        return record

    def close(self):
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def needs_refresh(self):
        return bool(self.state.needs_refresh())

    def mark_refreshed(self):
        self.state.mark_refreshed()


class TermNormalizerAdapter(_ScheduleBacked):
    """Policies `grandplan` (A) and `adaptive` (B) over `norm.TermNormalizer`.

    `ScheduleState` supplies tau, rho and the single activation latch; the
    normalizer supplies every coefficient, the Lipschitz cap, the cancellation
    ratio and `norm_trace.jsonl`. `VersionPair` presents both version counters
    as one, so a single `dp_hook.install_version_invariant(optimizer, adapter)`
    covers an activation bump from `update_continuous` *and* a coefficient bump
    from `transaction()` -- stacking two invariant wrappers would not work,
    because `refresh_nesterov_secant` unwraps exactly one `__wrapped__` level.
    """

    TRACE_NAME = NORM_TRACE_NAME

    def __init__(self, policy, *, num_movable=None, num_nodes=None, norm_p=1,
                 probe_every=50, wt0=0.05, wt_step=0.05, ramp_period=100,
                 wt_max=1.0, momentum=0.75, target_shares=None,
                 track_cancellation=True, **config):
        if policy not in ("grandplan", "adaptive"):
            raise ValueError(
                "TermNormalizerAdapter serves 'grandplan' and 'adaptive'; "
                "'legacy' is LegacyNormAdapter's (got %r)" % (policy,))
        super().__init__(**config)
        self.policy = policy
        # Policy B's shares are fractions of the *total* force
        # G = ||grad WL|| + sum_t lam_t ||grad T_t||, so the legacy knobs map
        # across directly: rho_max is already "IO force as a fraction of the WL
        # force", and legacy's f_ft_max is the FT force as a fraction of the
        # *IO* force, i.e. rho_max * f_ft_max of the whole.
        self.target_shares = {"io": float(self.state.rho_max),
                              "ft": float(self.state.rho_max) * float(self.state.f_ft_max)}
        self.target_shares.update(parse_target_shares(target_shares))
        self.normalizer = TermNormalizer(
            policy=policy, norm_p=norm_p, ema=self.state.ema,
            probe_every=probe_every, wt0=wt0, wt_step=wt_step,
            ramp_period=ramp_period, wt_max=wt_max, momentum=momentum,
            c_lip=self.state.c_lip, eps_rel=self.state.eps_rel,
            num_movable=num_movable, num_nodes=num_nodes,
            track_cancellation=track_cancellation,
            trace=(None if self.trace_path is None
                   else NormTraceWriter(self.trace_path)))
        self._versions = VersionPair(self.state, self.normalizer)
        self._overflow = float("nan")
        self._registered = False

    @property
    def lambda_io(self):
        return float(self.normalizer.lambdas.get("io", 0.0))

    @property
    def kappa_ft(self):
        """`lambda_ft / lambda_io` -- the ratio `ops/ft_term.FtTerm.forward`
        wants, since it scales the FT part by `lambda_io * kappa_ft`. 0.0 when
        `lambda_io == 0`: there is no IO coefficient to divide by, and the
        driver's `term_fn` is gated on `lambda_io != 0` anyway."""
        lambda_io = float(self.normalizer.lambdas.get("io", 0.0))
        if lambda_io == 0.0:
            return 0.0
        return float(self.normalizer.lambdas.get("ft", 0.0)) / lambda_io

    @property
    def obj_version(self):
        return int(self._versions.obj_version)

    @property
    def refreshed_version(self):
        return int(self._versions.refreshed_version)

    def begin_iteration(self, iteration, overflow, gamma):
        self._overflow = float(overflow)
        discrete = bool(self.state.update_continuous(iteration, overflow,
                                                     self.L_R, gamma))
        self._sync_activation()
        return discrete

    def _sync_activation(self):
        """`ScheduleState` is the single activation authority.

        `TermNormalizer._activate` latches a term the first time it *sees* an
        overflow at or below the threshold, and it only runs inside
        `weights()`/`transaction()` -- i.e. on the probe cadence, tens of
        iterations after the schedule activated. Left alone, policy A's
        `activation_ramp` would restart from that probe (and policy B's ramped
        share with it). Latching here, every iteration, keeps one
        `it_activate` for the whole adapter; the normalizer's own latch is
        monotone and idempotent, so it becomes a no-op afterwards.
        """
        if not self.state.active:
            return
        for term_state in self.normalizer.states.values():
            if not term_state.active:
                term_state.active = True
                term_state.it_activate = int(self.state.it_activate)

    def _register(self, io_term, ft_term, ecc_max):
        """Register the production terms on the first probe -- the first moment
        the driver hands over the terms and the FT curvature.

        Curvatures are design sec 4's declared values: 1 for IO, `ecc_max` for
        FT (floored at 1, the curvature of a term with no eccentricity spread,
        matching `schedules.derive_cmax`'s `max(ecc_max - 1, 0)`).
        `num_movable`/`num_nodes` fall back to the IO term's, which is where
        `ops/ft_callback.publish_atomic` reads them from too; without them
        `TermNormalizer.probe` refuses to run, because the fixed/filler mask
        would silently become a no-op.
        """
        if self.normalizer.num_movable is None:
            self.normalizer.num_movable = int(io_term.num_movable)
        if self.normalizer.num_nodes is None:
            self.normalizer.num_nodes = int(io_term.num_nodes)
        self.normalizer.register(
            "io", IoNormTerm(io_term), 1.0,
            target_share=self.target_shares.get("io", 0.0),
            activate_overflow=self.state.of_on, n_ramp=self.state.n_ramp)
        if ft_term is not None:
            self.normalizer.register(
                "ft", FtNormTerm(ft_term), max(float(ecc_max), 1.0),
                target_share=self.target_shares.get("ft", 0.0),
                activate_overflow=self.state.of_on, n_ramp=self.state.n_ramp)
        self._registered = True
        self._sync_activation()

    def probe(self, iteration, pos, *, io_term, ft_term, wirelength_op,
              ecc_max, gamma):
        """One WL backward plus one isolated backward per term, then one atomic
        coefficient transaction. Returns the pending `norm_trace.jsonl` row;
        `mark_refreshed()` is what actually writes it."""
        if not self._registered:
            self._register(io_term, ft_term, ecc_max)
        ctx = {"iteration": int(iteration), "overflow": self._overflow,
               "tau": self.tau, "gamma": float(gamma)}
        self.normalizer.probe(iteration, pos, wirelength_op, ctx)
        transaction = self.normalizer.transaction(iteration, self._overflow,
                                                  self.tau, gamma)
        return transaction.row

    def write_trace_row(self, row, extra):
        """No-op. `norm_trace.jsonl` is written by the normalizer's own
        `NormTraceWriter` at `mark_refreshed()` time, and that writer rejects
        any row whose keys are not exactly `norm_trace.ROW_FIELDS` (pre-flight
        amendment A-9: one schema per file name). The driver's per-probe
        extras -- `io_count`, `ft_count`, `churn` -- reach `result.json`
        through `soft_summary["probe_samples"]` instead."""
        return None

    def close(self):
        if self.normalizer.trace is not None:
            self.normalizer.trace.close()
            self.normalizer.trace = None

    def needs_refresh(self):
        return self.obj_version != self.refreshed_version

    def mark_refreshed(self):
        self.state.mark_refreshed()
        self.normalizer.mark_refreshed()


def make_norm_adapter(policy, **config):
    if policy not in NORM_POLICIES:
        raise ValueError(f"unknown norm policy {policy!r}; expected one of {NORM_POLICIES}")
    unknown = sorted(set(config) - set(_SCHEDULE_KEYS) - set(_NORMALIZER_KEYS))
    if unknown:
        raise TypeError(f"make_norm_adapter got unexpected keyword(s): {unknown}")
    schedule = {key: value for key, value in config.items() if key in _SCHEDULE_KEYS}
    if policy == "legacy":
        return LegacyNormAdapter(**schedule)
    extra = {key: value for key, value in config.items() if key in _NORMALIZER_KEYS}
    return TermNormalizerAdapter(policy, **schedule, **extra)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_norm_adapter.py -v`
Expected: PASS — 8 passed.

- [ ] **Step 5: Run P-H's own normalisation tests for regressions**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_norm.py tests/test_norm_trace.py tests/test_norm_legacy_adapter.py -q`
Expected: PASS — unchanged. This task adds a consumer of `norm.py`/`norm_trace.py`; it must not edit either.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/norm_adapter.py tests/test_norm_adapter.py
git commit -m "feat(norm-adapter): legacy and TermNormalizer adapters behind one protocol

LegacyNormAdapter wraps ScheduleState + publish_atomic; TermNormalizerAdapter
drives P-H's TermNormalizer for --norm-policy grandplan|adaptive, keeping a
ScheduleState only for tau/rho/activation and exposing lambda_io, kappa_ft and
a VersionPair over both version counters so the Nesterov invariant still
holds. Each policy writes its own trace file: norm_trace.jsonl through
NormTraceWriter, legacy_trace.jsonl for the retired row schema.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Main-flow metrics (IO accounting, area balance, fence compliance, phase summary)

**Files:**
- Create: `src/ioplace/main_flow_metrics.py`
- Test: `tests/test_main_flow_metrics.py`

**Interfaces:**
- Consumes: `ioplace.region_grid.RegionGrid`; `ioplace.profile.host_rss_gb`; `ioplace.freeze.region_cell_stats` (Task 3 — the single implementation of the per-region count/area/utilisation arithmetic, amendment D-2). The last two are **local** imports inside the functions that need them: this module must stay importable without torch, and `freeze` pulls in `ops/soft_assign`, which imports torch.
- Produces:
  - `io_accounting(io_soft, io_fence_gp, io_final) -> dict` with `io_soft`, `io_fence_gp`, `io_count`, `io_delta_at_freeze`, `lg_loss`, `io_identity_residual`
  - `region_area_balance(part, node_size_x, node_size_y, rs) -> dict`
  - `fence_compliance(rg, node_x, node_y, part, node_size_x=None, node_size_y=None) -> dict` with `lower_left`, `center`
  - `MAIN_FLOW_PHASES = ("read_soft", "gp_soft", "freeze", "read_fence", "gp_fence", "lg", "eval")`
  - `phase_summary(timer, sampler, *, names=MAIN_FLOW_PHASES, host_rss=None) -> dict` — always emits `t_<name>` for every name in `names` (0.0 when the phase did not run), so `save_result`'s field contract holds for a `--phase fence` run too

**Definitions** (spec §7 diagnostics 4/5 and the closing identity). The three IO measurements are taken with the same GPU evaluator on the same netlist ordering: `io_soft` at the freeze iteration, `io_fence_gp` on the phase-3 GP positions immediately before `legalize_op`, `io_count` on the final post-LG placement. Then `io_delta_at_freeze = io_fence_gp - io_soft` (everything the fence phase cost or saved) and `lg_loss = io_count - io_fence_gp` (the same definition `run_placement_io.py:691-692` already uses), so `io(final) = io(soft) + io_delta_at_freeze + lg_loss` closes with `io_identity_residual == 0` by construction. `io_identity_residual` stays in the schema as the written-down form of that identity, but it is **not** a check: it is algebraically zero for *any* three inputs and can never detect a stale or mismatched measurement (pre-flight amendment D-1). What can is provenance — `run_fence_gp` reports `io_fence_gp_source`, `"legalize_op"` when the wrapper actually measured the GP positions handed to LG and `"fallback"` when it did not, and Task 9 asserts the former plus `result["io_soft"] == freeze["io_soft"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_flow_metrics.py`:

```python
import numpy as np
import pytest

from ioplace.main_flow_metrics import (fence_compliance, io_accounting,
                                       phase_summary, region_area_balance)
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def test_io_accounting_closes_the_identity():
    out = io_accounting(io_soft=1000, io_fence_gp=1120, io_final=1155)
    assert out["io_delta_at_freeze"] == 120
    assert out["lg_loss"] == 35
    assert out["io_identity_residual"] == 0
    assert out["io_soft"] == 1000 and out["io_fence_gp"] == 1120
    assert out["io_count"] == 1155


def test_region_area_balance_reports_utilisation_and_count_deviation():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)   # 2500 each
    part = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    size_x = np.array([10., 10., 10., 5., 5.])
    size_y = np.array([10., 10., 10., 10., 10.])
    out = region_area_balance(part, size_x, size_y, rs)
    assert out["region_cell_count"] == [2, 1, 1, 1]
    assert out["region_utilization"] == [0.08, 0.04, 0.02, 0.02]
    assert out["utilization_max"] == 0.08 and out["utilization_min"] == 0.02
    assert out["utilization_ratio"] == 4.0
    assert out["cell_count_max"] == 2 and out["cell_count_min"] == 1
    assert out["cell_count_deviation"] == pytest.approx(0.6)   # |2-1.25|/1.25


def test_region_area_balance_survives_an_empty_region():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    out = region_area_balance(np.array([0, 0], dtype=np.int32),
                              np.array([1., 1.]), np.array([1., 1.]), rs)
    assert out["utilization_min"] == 0.0
    assert out["utilization_ratio"] is None
    assert out["empty_regions"] == [1, 2, 3]


def test_fence_compliance_distinguishes_lower_left_from_centre():
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    rg = RegionGrid(rs)
    node_x = np.array([10., 40.])
    node_y = np.array([10., 10.])
    size_x = np.array([2., 30.])          # cell 1 centre at x=55 -> region 1
    size_y = np.array([2., 2.])
    part = np.array([0, 1], dtype=np.int32)
    out = fence_compliance(rg, node_x, node_y, part, size_x, size_y)
    assert out["lower_left"] == 0.5
    assert out["center"] == 1.0
    assert fence_compliance(rg, node_x, node_y, part)["center"] is None


def test_phase_summary_takes_the_max_peak_and_skips_unmeasured_phases():
    class _Timer:
        phases = {"gp_soft": {"t_s": 12.5, "peak_alloc_gb": 1.5,
                              "host_rss_hwm_at_phase_end": 3.0},
                  "gp_fence": {"t_s": 30.0, "peak_alloc_gb": 4.0,
                               "host_rss_hwm_at_phase_end": 5.0},
                  "lg": {"t_s": 1.0, "peak_alloc_gb": None,
                         "host_rss_hwm_at_phase_end": 5.0}}

    class _Sampler:
        device_used_gb = 7.25

    out = phase_summary(_Timer(), _Sampler(), host_rss=6.0)
    assert out["t_gp_soft"] == 12.5 and out["t_gp_fence"] == 30.0 and out["t_lg"] == 1.0
    # every declared phase gets a key, so result.json's contract cannot break
    # when only half the flow ran
    assert out["t_read_soft"] == 0.0 and out["t_freeze"] == 0.0
    assert out["t_read_fence"] == 0.0 and out["t_eval"] == 0.0
    assert out["peak_mem_mb"] == 4.0 * 1024.0
    assert out["peak_mem_mb_by_phase"]["gp_fence"] == 4.0 * 1024.0
    assert out["peak_mem_mb_by_phase"]["lg"] is None
    assert out["device_used_gb"] == 7.25
    assert out["host_peak_rss_gb"] == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.main_flow_metrics'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/main_flow_metrics.py`:

```python
"""Pure metric functions for the v2 main flow's result.json (design v2 sec 7).

No torch, no DREAMPlace: every function here takes numpy arrays or plain
records so it can be unit-tested and re-run over saved artefacts.
"""
import numpy as np

# Every phase the v2 main flow can open, in run order. phase_summary emits a
# `t_<name>` key for each one whether or not it ran, so result.json's field
# contract (artifacts.MAIN_FLOW_RESULT_FIELDS) holds for `--phase fence` too.
MAIN_FLOW_PHASES = ("read_soft", "gp_soft", "freeze", "read_fence", "gp_fence",
                    "lg", "eval")


def io_accounting(io_soft, io_fence_gp, io_final):
    """The design's closing identity:
    `io(final) = io(soft, last GP) + io_delta_at_freeze + lg_loss`.

    `lg_loss` keeps run_placement_io.py:691-692's definition (post-LG minus
    the last GP evaluation); `io_delta_at_freeze` is everything the fence
    phase changed, measured between the freeze evaluation and the last
    fence-GP evaluation.

    `io_identity_residual` is `io_final - (io_soft + delta + lg_loss)`, which
    is 0 for *any* three inputs -- both summands were just defined as
    differences of them. It is kept as a result.json field because the schema
    documents the identity, but it detects nothing (pre-flight amendment D-1);
    the field that does is `io_fence_gp_source`, which run_fence_gp sets from
    whether the legalize_op wrapper actually ran.
    """
    io_soft, io_fence_gp, io_final = int(io_soft), int(io_fence_gp), int(io_final)
    delta = io_fence_gp - io_soft
    lg_loss = io_final - io_fence_gp
    return {"io_soft": io_soft, "io_fence_gp": io_fence_gp, "io_count": io_final,
            "io_delta_at_freeze": delta, "lg_loss": lg_loss,
            "io_identity_residual": io_final - (io_soft + delta + lg_loss)}


def region_area_balance(part, node_size_x, node_size_y, rs):
    """Per-region cell count/area/utilisation plus the max-min summaries.

    The four per-region arrays come from `freeze.region_cell_stats` -- one
    implementation, two consumers (pre-flight amendment D-2); this function
    adds only the summaries result.json quotes. The import is local because
    `freeze` pulls in `ops/soft_assign`, which imports torch, and this module
    must stay loadable in a bare CPU report process.

    Utilisation is `cell area / region area`, which is invariant under
    PlaceDB's shift+scale (both terms carry `scale_factor**2`).
    `region_cell_area`/`region_area` are **not**, so call this with
    native-unit sizes and the native `RegionSet`: that is the frame
    `freeze.json` is written in, and result.json must quote the same numbers.
    """
    from ioplace.freeze import region_cell_stats
    stats = region_cell_stats(part, node_size_x, node_size_y, rs)
    counts = np.asarray(stats["region_cell_count"], dtype=np.float64)
    utilization = np.asarray(stats["region_utilization"], dtype=np.float64)
    k = rs.k
    mean_count = counts.mean() if k else 0.0
    out = dict(stats)
    out.update({
        "k": int(k),
        "utilization_max": float(utilization.max()),
        "utilization_min": float(utilization.min()),
        "utilization_ratio": (float(utilization.max() / utilization.min())
                              if utilization.min() > 0 else None),
        "cell_count_max": int(counts.max()), "cell_count_min": int(counts.min()),
        "cell_count_deviation": (float(np.max(np.abs(counts - mean_count)) / mean_count)
                                 if mean_count > 0 else None),
        "empty_regions": np.flatnonzero(counts == 0).astype(int).tolist()})
    return out


def fence_compliance(rg, node_x, node_y, part, node_size_x=None, node_size_y=None):
    """Fraction of movable cells that landed in their assigned region.

    `lower_left` is run_placement_two_stage.py:252-255's definition, kept so
    the number stays comparable with the legacy two-stage arm. `center` uses
    the same anchor as the freeze membership (design v2 sec 3 phase 2) and is
    the one to quote for the v2 flow; it is None when sizes are not supplied.
    """
    part = np.asarray(part, dtype=np.int64)
    m = len(part)
    x = np.asarray(node_x, dtype=np.float64)[:m]
    y = np.asarray(node_y, dtype=np.float64)[:m]
    out = {"lower_left": float((rg.region_of_points(x, y) == part).mean()),
           "center": None}
    if node_size_x is not None and node_size_y is not None:
        cx = x + 0.5 * np.asarray(node_size_x, dtype=np.float64)[:m]
        cy = y + 0.5 * np.asarray(node_size_y, dtype=np.float64)[:m]
        out["center"] = float((rg.region_of_points(cx, cy) == part).mean())
    return out


def phase_summary(timer, sampler, *, names=MAIN_FLOW_PHASES, host_rss=None):
    """Assemble the per-phase runtime/GPU-peak block from a `profile.PhaseTimer`
    and a `profile.DeviceMemSampler`.

    `peak_mem_mb` is the max over phase peaks, not a single end-of-run read:
    each phase calls `reset_peak_memory_stats()` on entry, so the global
    counter holds only the last phase's peak by the end (same reasoning as
    run_placement._phase_summary). Phases recorded with `peak_alloc_gb=None`
    (PhaseTimer(reset_peak=False)) are skipped rather than counted as zero.
    """
    phases = dict(timer.phases)
    out = {"phases": phases, "peak_mem_mb_by_phase": {}}
    for name in names:
        out[f"t_{name}"] = 0.0
    for name, record in phases.items():
        out[f"t_{name}"] = float(record.get("t_s", 0.0))
        peak = record.get("peak_alloc_gb")
        out["peak_mem_mb_by_phase"][name] = None if peak is None else peak * 1024.0
    measured = [value for value in out["peak_mem_mb_by_phase"].values()
                if value is not None]
    out["peak_mem_mb"] = max(measured) if measured else 0.0
    if host_rss is None:
        from ioplace.profile import host_rss_gb
        host_rss = host_rss_gb()
    host_peaks = [record.get("host_rss_hwm_at_phase_end") or 0.0
                  for record in phases.values()] + [host_rss]
    out["device_used_gb"] = sampler.device_used_gb
    out["host_peak_rss_gb"] = max(host_peaks)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_metrics.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/main_flow_metrics.py tests/test_main_flow_metrics.py
git commit -m "feat(metrics): IO accounting identity, area balance, fence compliance

io_delta_at_freeze = io(fence GP) - io(soft), lg_loss keeps run_placement_io's
definition, so io(final) = io(soft) + io_delta_at_freeze + lg_loss closes with a
zero residual. Fence compliance is reported at both the lower-left (comparable
with the two-stage arm) and the cell centre (the freeze anchor).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The driver — `run_main_flow.py`

**Files:**
- Create: `src/ioplace/drivers/run_main_flow.py`
- Test: `tests/test_main_flow_driver.py` (fast tests only; the end-to-end run is Task 9)

**Interfaces:**
- Consumes: everything from Tasks 1–6, plus `run_placement._load_dreamplace` / `get_regions_for` / `extract_final_positions` / `_pack_eval_metrics` / `_legalization_diagnostics` / `_effective_scale_fields` / `_stop_overflow_reached` / `_gp_iteration_budget` / `_t8a_provenance`, `run_placement_io._io_cleanup` / `_install_attribute` / `_cleanup_once`, `run_placement_two_stage.assign_blocks_to_regions`, `dp_hook`, `ops/io_term`, `ops/ft_term`, `ops/soft_assign.rect_table`, `evaluator_gpu.GpuEvalContext`, `export/evaluation.save_evaluation`.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `run_soft_phase(config_json, out_dir, *, k, rtype, seed, regions_json=None, init="die_center", seed_npz=None, membership_npz=None, remap_blocks="auto", norm_policy="legacy", norm_target_share=None, every=50, home_period=None, rho_max=0.1, f_ft_max=0.0, tau_hi=0.30, tau_lo=0.03, of_on=0.90, of_end=None, of_full=0.20, ft_ramp_mode="window", tau_start=0.12, tau_full=0.05, freeze_window=50, freeze_overflow=0.15, freeze_tau_rel=0.05, freeze_churn=0.005, argmax_chunk=4, ignore_net_degree=None, w_mode="unit", dp_seed=None, deterministic=None, check_invariant=False, timer=None) -> dict` returning `{"freeze", "part", "soft_npz", "membership_npz", "regions_json", "region_source", "init", "prior", "norm_policy", "trace_path", "probe_samples", "num_probes", "num_refreshes", "gp_iterations_soft", "density_weight_soft", "placedb_sha256", "die_native"}`
  - `run_fence_gp(config_json, out_dir, *, region_set, part, positions, reference_density_weight, k, density_clamp_lo, density_clamp_hi, dp_seed, deterministic, extra_terms=(), timer) -> dict`

  `sampler` is gone from both: neither ever read it, only `run_main_flow`'s own sampler reaches `phase_summary` (pre-flight amendment D-3).
  - `run_main_flow(config_json, out_dir, **options) -> dict`
  - `main(argv=None)`

**Key behaviours**

1. **Phase 1 stops at the freeze.** `NonLinearPlace` offers no "stop now" return value (`NonLinearPlace.py:521-523` ignores the callback's result), so the monitor raises `_FreezeReached` from inside the callback and the driver catches it around `placer(params, placedb, lr)`. Phase 1 runs with `params.legalize_flag = 0` — the soft placement is never legalised; LG belongs to phase 4 on the fence instance. If the GP loop ends before the criterion fires (budget exhausted, or `Lgamma_stop_criterion` at `stop_overflow`), the driver takes the final `placer.pos[0]` and records `reason = "gp_end"`.
2. **Two instances, artefacts in between.** Phase 3 never touches phase 1's `PlaceDB`; it goes through `soft.npz` / `frozen_membership.npz` / `freeze.json`, so `--phase fence` reproduces it from disk.
3. **IO and FT are off after the freeze.** `run_fence_gp` attaches nothing unless the caller passes `extra_terms` — the hook point where P-D's capacity term and P-E's pseudo-FT term will attach.
4. **`io_fence_gp` is measured exactly.** The `legalize_op` wrapper evaluates the GP positions immediately before legalisation, so `lg_loss` is never contaminated by a stale periodic callback.
5. **Soft-assign geometry lives in scaled units.** `IoTerm`, the freeze argmax and the evaluator all work on the optimizer's coordinates, so the native `RegionSet` is converted once with `artifacts.scaled_region_set(rs, params.shift_factor, params.scale_factor)`. Everything written to disk is converted back. The two *area* statistics are the exception: `region_cell_stats`/`region_area_balance` are called with **native-unit** sizes and the native `RegionSet` at both call sites, because `region_area`/`region_cell_area` are not scale-invariant and `freeze.json` and `result.json` must quote the same numbers (amendment D-2).
6. **One normalisation trace per policy, written by the adapter.** `--norm-policy legacy` writes `legacy_trace.jsonl` (`publish_atomic`'s keys plus the driver's per-probe extras); `grandplan`/`adaptive` write `norm_trace.jsonl` strictly through `norm_trace.NormTraceWriter`, whose rows `TermNormalizer` emits at `mark_refreshed()` time. The driver never formats or opens a trace file: it hands `adapter.write_trace_row(row, sample)` the extras and lets the adapter decide, and registers `adapter.close` on the cleanup stack. The per-probe `io_count`/`ft_count`/`churn` samples always reach `result.json` through `soft_summary["probe_samples"]`, whichever policy ran (amendment A-9).

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_flow_driver.py`:

```python
import os
import numpy as np
import pytest

from ioplace.drivers.run_main_flow import (_resolve_prior, _resolve_regions,
                                           build_parser, run_main_flow)
from ioplace.regions import make_grid_regions


def test_parser_exposes_the_v2_switches():
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.k == 16 and args.rtype == "grid" and args.phase == "all"
    assert args.init == "die_center" and args.norm_policy == "legacy"
    assert args.every == 50 and args.freeze_window == 50
    assert args.freeze_overflow == 0.15 and args.freeze_tau_rel == 0.05
    assert args.freeze_churn == 0.005
    assert args.density_clamp_lo == 0.25 and args.density_clamp_hi == 4.0
    assert args.remap_blocks == "auto"
    assert args.argmax_chunk == 4 and args.norm_target_share is None
    for bad, choices in (("--phase", "soft fence all"),
                         ("--init", "die_center region_center seed"),
                         ("--norm-policy", "legacy grandplan adaptive")):
        for choice in choices.split():
            parser.parse_args(["--config", "c.json", "--out-dir", "o", bad, choice])
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "c.json", "--out-dir", "o", "--phase", "bogus"])


def test_resolve_regions_rejects_a_region_file_from_another_die(tmp_path):
    path = str(tmp_path / "regions.json")
    make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=8).to_json(path)
    rs, source = _resolve_regions((0., 0., 100., 100.), 4, "grid", 0, path)
    assert rs.k == 4 and source == "file"
    with pytest.raises(ValueError, match="native post-read units"):
        _resolve_regions((0., 0., 200., 100.), 4, "grid", 0, path)


def test_resolve_regions_rejects_a_region_file_with_the_wrong_k(tmp_path):
    """C-6: IoTerm(K=k), argmax_region(..., k), region_centers, freeze_record
    and save_membership all assume rs.k == --k. Arms (b)/ours pass a producer
    regions.json, so a K=16 geometry against a K=4 term must not run."""
    path = str(tmp_path / "regions.json")
    make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=8).to_json(path)
    with pytest.raises(ValueError, match="k=4"):
        _resolve_regions((0., 0., 100., 100.), 16, "grid", 0, path)


def test_resolve_regions_falls_back_to_the_builtin_grid():
    rs, source = _resolve_regions((0., 0., 100., 100.), 4, "grid", 0, None)
    assert rs.k == 4 and source == "builtin"


def test_resolve_prior_remaps_partitioner_block_ids(tmp_path, monkeypatch):
    from ioplace.artifacts import save_membership
    path = str(tmp_path / "membership.npz")
    part = np.array([0, 1, 2, 3], dtype=np.int32)
    save_membership(path, part, source="mtkahypar", k=4, seed=0, epsilon=0.03)
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=8)
    calls = []

    def fake_assign(parts, nl, region_set):
        calls.append(region_set.k)
        return np.asarray(parts, dtype=np.int32)[::-1].copy()

    monkeypatch.setattr("ioplace.drivers.run_placement_two_stage.assign_blocks_to_regions",
                        fake_assign)
    built = []

    def nl_fn():
        built.append(1)
        return None

    got, info = _resolve_prior(path, "auto", nl_fn=nl_fn, rs=rs, num_movable=4)
    assert calls == [4] and info["remapped"] is True
    assert got.tolist() == [3, 2, 1, 0]
    assert built == [1]

    got_off, info_off = _resolve_prior(path, "off", nl_fn=nl_fn, rs=rs,
                                       num_movable=4)
    assert info_off["remapped"] is False and got_off.tolist() == [0, 1, 2, 3]
    # D-7: no remap, no netlist -- netlist_from_placedb copies pin2node and
    # flat_net2pin, which is multiple GB at 10M-30M cells.
    assert built == [1]


def test_fence_phase_alone_requires_the_soft_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="soft.npz"):
        run_main_flow("unused.json", str(tmp_path), phase="fence")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_driver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.drivers.run_main_flow'`

- [ ] **Step 3: Write the implementation**

Create `src/ioplace/drivers/run_main_flow.py`:

```python
"""v2 main flow: soft-assign GP -> freeze -> fence GP -> fence LG -> evaluator.

Design v2 sec 3. Forked from run_placement_io.py, which stays as the
single-phase legacy driver and is imported here only for its resource-cleanup
helpers. The four phases live in two DREAMPlace instances because fence data
may only be injected between read() and initialize() (fence_inject.py:9-16);
they communicate through validated artefacts, so `--phase {all,soft,fence}`
can re-run either half from disk.
"""
import argparse
import json
import os
import time

import numpy as np

from ioplace.artifacts import (MAIN_FLOW_RESULT_SCHEMA_VERSION, file_sha256,
                               load_freeze, load_membership, load_positions,
                               placedb_identity_sha256, save_freeze,
                               save_membership, save_positions, save_result,
                               scaled_region_set)
from ioplace.dp_hook import (assert_optimizer_lock, attach_terms, detach_terms,
                             install_version_invariant, refresh_nesterov_secant)
from ioplace.drivers.run_placement import (_effective_scale_fields,
                                           _gp_iteration_budget,
                                           _legalization_diagnostics,
                                           _pack_eval_metrics,
                                           _stop_overflow_reached,
                                           _t8a_provenance, _load_dreamplace,
                                           extract_final_positions,
                                           get_regions_for)
from ioplace.drivers.run_placement_io import (_cleanup_once, _install_attribute,
                                              _io_cleanup)
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.export.evaluation import save_evaluation
from ioplace.fence_phase import build_fence_placedb, install_density_weight_clamp
from ioplace.freeze import (FreezeMonitor, argmax_region, cell_centers,
                            ensure_nonempty_regions, freeze_record,
                            region_cell_stats)
from ioplace.init_pos import INIT_MODES, apply_init, region_centers
from ioplace.main_flow_metrics import (fence_compliance, io_accounting,
                                       phase_summary, region_area_balance)
from ioplace.netlist import netlist_from_placedb
from ioplace.norm_adapter import NORM_POLICIES, make_norm_adapter
from ioplace.ops.io_term import IoTerm, build_net_node_csr
from ioplace.ops.soft_assign import rect_table
from ioplace.profile import DeviceMemSampler, PhaseTimer
from ioplace.region_grid import RegionGrid
from ioplace.regions import RegionSet

SOFT_NPZ = "soft.npz"
FREEZE_JSON = "freeze.json"
FROZEN_MEMBERSHIP_NPZ = "frozen_membership.npz"
PLACEMENT_NPZ = "placement.npz"
EVALUATION_NPZ = "evaluation.npz"
NORM_TRACE = "norm_trace.jsonl"
LEGACY_TRACE = "legacy_trace.jsonl"
REGIONS_JSON = "regions.json"
RESULT_JSON = "result.json"
SOFT_RESULT_JSON = "soft_result.json"


class _FreezeReached(Exception):
    """Raised from the phase-1 iteration callback to end the soft GP.

    NonLinearPlace ignores the callback's return value
    (NonLinearPlace.py:521-523), so an exception is the only way to stop the
    loop from the driver without touching DREAMPlace source. Phase 1 runs with
    legalize_flag=0, so nothing *this driver reads* is skipped by unwinding:
    the tail it skips is placedb.apply() (NonLinearPlace.py:942), plotting and
    the legalize/DP branches, and the driver takes its positions from
    placer.pos[0] on both the freeze and the gp_end path. placedb.node_x does
    stay at its pre-GP values here, unlike on the gp_end path -- nothing below
    reads it (amendment D-12).
    """

    def __init__(self, snapshot):
        super().__init__("freeze criterion reached")
        self.snapshot = snapshot


def _die_of(placedb):
    return (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))


def _resolve_regions(die_native, k, rtype, seed, regions_json):
    if regions_json:
        rs = RegionSet.from_json(regions_json)
        rs.validate()
        # IoTerm(K=k), argmax_region(..., k), region_centers(rs), freeze_record
        # and save_membership all assume the file's k IS --k; arms (b)/ours pass
        # a producer regions.json and would otherwise silently run a K=16
        # geometry against a K=4 term (pre-flight amendment C-6).
        if rs.k != k:
            raise ValueError(f"{regions_json} has k={rs.k}, --k is {k}")
        span = max(die_native[2] - die_native[0], die_native[3] - die_native[1])
        if not np.allclose(rs.die, die_native, atol=1e-6 * span, rtol=0.0):
            raise ValueError(f"{regions_json} die {tuple(rs.die)} != placedb die "
                             f"{die_native}; regions.json must be in native "
                             "post-read units")
        return rs, "file"
    return get_regions_for(die_native, k, rtype, seed), "builtin"


def _resolve_prior(membership_npz, remap_blocks, *, nl_fn, rs, num_movable):
    """Load the soft-phase membership prior and, when it carries partitioner
    block ids, remap them onto geometric regions.

    mtkahypar block ids have no geometric meaning (M0 finding, see
    run_placement_two_stage.assign_blocks_to_regions' docstring), so feeding
    them straight in as region ids can throw two heavily connected blocks onto
    opposite die corners. `auto` decides from the artefact's own `source`.

    `nl_fn` is a zero-arg factory, not a netlist: `netlist_from_placedb` copies
    pin2node/pin2net/flat_net2pin, several GB at 10M-30M cells, and the remap
    is the only consumer -- with `remap_blocks="off"` or a non-mtkahypar
    artefact it must never be built (pre-flight amendment D-7). It cannot be
    hoisted and shared with the caller's own netlist either: that one is built
    after `placedb.initialize()`, and `netlist_from_placedb` captures node
    positions and sizes, which `scale()` has rewritten by then.
    """
    mem = load_membership(membership_npz, expect_num_movable=num_movable,
                          expect_k=rs.k)
    if remap_blocks == "auto":
        remap = mem.source.startswith("mtkahypar")
    elif remap_blocks == "on":
        remap = True
    elif remap_blocks == "off":
        remap = False
    else:
        raise ValueError(f"unknown remap_blocks {remap_blocks!r}")
    part = np.asarray(mem.part, dtype=np.int32)
    if remap:
        from ioplace.drivers import run_placement_two_stage
        part = run_placement_two_stage.assign_blocks_to_regions(part, nl_fn(), rs)
    return np.asarray(part, dtype=np.int32), {
        "source": mem.source, "k": mem.k, "seed": mem.seed,
        "epsilon": mem.epsilon, "remapped": bool(remap)}


def _to_native(values, shift, scale):
    return np.asarray(values, dtype=np.float64) / scale + shift


def run_soft_phase(config_json, out_dir, *, k, rtype, seed, regions_json=None,
                   init="die_center", seed_npz=None, membership_npz=None,
                   remap_blocks="auto", norm_policy="legacy",
                   norm_target_share=None, every=50,
                   home_period=None, rho_max=0.1, f_ft_max=0.0, tau_hi=0.30,
                   tau_lo=0.03, of_on=0.90, of_end=None, of_full=0.20,
                   ft_ramp_mode="window", tau_start=0.12, tau_full=0.05,
                   freeze_window=50, freeze_overflow=0.15, freeze_tau_rel=0.05,
                   freeze_churn=0.005, argmax_chunk=4, ignore_net_degree=None,
                   w_mode="unit", dp_seed=None, deterministic=None,
                   check_invariant=False, timer=None):
    """Phase 1 + phase 2. Writes soft.npz, frozen_membership.npz, freeze.json
    and the active policy's normalisation trace (norm_trace.jsonl, or
    legacy_trace.jsonl under --norm-policy legacy); returns the record the
    caller folds into result.json as `soft_summary`."""
    import torch
    if every <= 0 or freeze_window % every:
        raise ValueError("freeze_window must be a positive multiple of --every "
                         f"(got window={freeze_window}, every={every})")
    home_period = every if home_period is None else home_period
    if home_period % every:
        raise ValueError("home_period must be a multiple of --every")
    if f_ft_max > 0 and rho_max <= 0:
        raise ValueError("--f-ft-max > 0 needs an enabled IO term (--rho-max > 0): "
                         "kappa_ft multiplies lambda_io")
    if init == "region_center" and not membership_npz:
        raise ValueError("--init region_center requires --membership (the prior "
                         "each cell is centred on)")
    if init == "seed" and not seed_npz:
        raise ValueError("--init seed requires --seed-npz")
    os.makedirs(out_dir, exist_ok=True)
    timer = PhaseTimer() if timer is None else timer

    with _io_cleanup() as cleanup:
        with timer.phase("read_soft"):
            params, placedb = _load_dreamplace(config_json)
            cleanup.callback(detach_terms, params)
            if dp_seed is not None:
                params.random_seed = dp_seed
            if deterministic is not None:
                params.deterministic_flag = deterministic
            # Phase 1 never legalises: LG is phase 4, on the fence instance.
            params.legalize_flag = 0
            die_native = _die_of(placedb)
            sha = placedb_identity_sha256(placedb)
            rs_native, region_source = _resolve_regions(die_native, k, rtype, seed,
                                                        regions_json)
            rs_native.to_json(os.path.join(out_dir, REGIONS_JSON))
            m = placedb.num_movable_nodes
            prior_part, prior_info = (None, None)
            if membership_npz:
                prior_part, prior_info = _resolve_prior(
                    membership_npz, remap_blocks,
                    nl_fn=lambda: netlist_from_placedb(placedb),
                    rs=rs_native, num_movable=m)
            seed_positions = None
            if seed_npz:
                seed_positions = load_positions(
                    seed_npz, expect_num_physical=placedb.num_physical_nodes,
                    expect_sha256=sha)
            init_info = apply_init(placedb, params, init, region_set=rs_native,
                                   part=prior_part, positions=seed_positions,
                                   rng_seed=int(params.random_seed))
            placedb.initialize(params)
            shift = (float(params.shift_factor[0]), float(params.shift_factor[1]))
            scale = float(params.scale_factor)
        assert_optimizer_lock(params)
        import NonLinearPlace

        gp_phase = timer.phase("gp_soft")
        gp_phase.__enter__()
        nl = netlist_from_placedb(placedb)
        rs_scaled = scaled_region_set(rs_native, shift, scale)
        rg = RegionGrid(rs_scaled)
        ctx = GpuEvalContext(nl, rg, device="cuda")
        rects, r2k = rect_table(rs_scaled)
        if of_end is None:
            of_end = float(params.stop_overflow)
        if ignore_net_degree is None:
            ignore_net_degree = int(params.ignore_net_degree)
        die_scaled = _die_of(placedb)
        L_R = ((die_scaled[2] - die_scaled[0]) * (die_scaled[3] - die_scaled[1]) / k) ** 0.5

        csr = build_net_node_csr(nl, ignore_net_degree)
        io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                         num_movable=nl.num_movable, num_physical=nl.num_physical,
                         num_nodes=placedb.num_nodes, device="cuda", w_mode=w_mode)
        ft_term, distance = None, None
        if f_ft_max > 0:
            from ioplace.region_graph import region_graph
            from ioplace.ops.ft_term import FtTerm
            _, distance, _ = region_graph(rg)
            ft_term = FtTerm(io_term, distance)
        ecc_max = float(distance.max()) if distance is not None else 0.0

        # The adapter owns its own trace file (legacy_trace.jsonl or
        # norm_trace.jsonl, amendment A-9) and, under grandplan/adaptive, the
        # TermNormalizer that needs num_movable/num_nodes for its fixed/filler
        # gradient mask. `out_dir` is where the trace lands; unknown-to-legacy
        # keys are dropped by make_norm_adapter, so this one call serves every
        # policy.
        adapter = make_norm_adapter(norm_policy, L_R=L_R, rho_max=rho_max,
                                    tau_hi=tau_hi, tau_lo=tau_lo, of_on=of_on,
                                    of_end=of_end, of_full=of_full,
                                    f_ft_max=f_ft_max, ft_ramp_mode=ft_ramp_mode,
                                    tau_start=tau_start, tau_full=tau_full,
                                    out_dir=out_dir, probe_every=every,
                                    target_shares=norm_target_share,
                                    num_movable=nl.num_movable,
                                    num_nodes=placedb.num_nodes)
        _cleanup_once(cleanup, adapter.close)
        monitor = FreezeMonitor(window=freeze_window, overflow_max=freeze_overflow,
                                tau_rel_max=freeze_tau_rel, churn_max=freeze_churn)

        def term_fn(pos):
            if not adapter.active or adapter.lambda_io == 0.0:
                return pos.new_zeros(())
            if ft_term is not None:
                return ft_term(pos, adapter.tau, adapter.lambda_io, adapter.kappa_ft)
            return io_term(pos, adapter.tau, adapter.lambda_io)

        attach_terms(params, [term_fn])
        np.random.seed(params.random_seed)
        placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

        n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
        total_iterations = params.global_place_stages[0]["iteration"]
        rects_t = torch.as_tensor(rects, device="cuda")
        r2k_t = torch.as_tensor(r2k, device="cuda")
        size_x = torch.as_tensor(np.asarray(placedb.node_size_x[:m], dtype=np.float64),
                                 device="cuda")
        size_y = torch.as_tensor(np.asarray(placedb.node_size_y[:m], dtype=np.float64),
                                 device="cuda")
        cb_state = {"last_iteration": -1, "num_refreshes": 0, "installed": False,
                    "io_gp": 0, "overflow": float("nan"), "probes": 0,
                    "probe_samples": []}
        previous_home = None

        def snapshot(iteration, pos, overflow, reason, io_count, argmax):
            return {"iteration": int(iteration), "reason": reason,
                    "overflow": float(overflow), "tau": adapter.tau,
                    "tau_rel": adapter.tau_rel, "churn": monitor.churn,
                    "io_count": int(io_count),
                    "argmax": np.asarray(argmax.cpu(), dtype=np.int32),
                    "node_x": pos.data[:n_phys].detach().cpu().numpy().astype(np.float64),
                    "node_y": pos.data[n_all:n_all + n_phys].detach().cpu().numpy().astype(np.float64)}

        def centre_argmax(pos):
            x = pos.data[:m].double()
            y = pos.data[n_all:n_all + m].double()
            cx, cy = cell_centers(x, y, size_x, size_y)
            # Chunk over regions: region_sdf_l1 materialises (N, R) and this
            # function a further (N, K). At 30M cells x K=16 in float64 that is
            # ~7.7 GB per gated callback (amendment D-8).
            return argmax_region(cx, cy, rects_t.double(), r2k_t, k,
                                 chunk=argmax_chunk)

        def cb(iteration, pos):
            nonlocal previous_home
            cb_state["last_iteration"] = iteration
            overflow = float(placer.model.overflow.max())
            cb_state["overflow"] = overflow
            gamma = float(placer.model.gamma)
            discrete = adapter.begin_iteration(iteration, overflow, gamma)
            if (iteration > 0 and iteration % every == 0) or iteration == total_iterations - 1:
                res = ctx.evaluate(pos.data[:n_phys], pos.data[n_all:n_all + n_phys])
                cb_state["io_gp"] = res.io_count
                if ft_term is not None and (ft_term.home is None
                                            or iteration % home_period == 0):
                    homes = res.per_net_home[csr.net_ids]
                    ft_term.set_home(homes)
                    previous_home = homes.copy()
                row = adapter.probe(iteration, pos, io_term=io_term, ft_term=ft_term,
                                    wirelength_op=placer.model.op_collections.wirelength_op,
                                    ecc_max=ecc_max, gamma=gamma)
                argmax = centre_argmax(pos)
                sample = {"iteration": int(iteration), "overflow": overflow,
                          "io_count": int(res.io_count),
                          "ft_count": int(res.ft_count),
                          "churn": monitor.observe(iteration, argmax)}
                cb_state["probe_samples"].append(sample)
                # The adapter owns the trace: under legacy this writes
                # publish_atomic's row plus `sample` to legacy_trace.jsonl;
                # under grandplan/adaptive it is a no-op, because
                # TermNormalizer already wrote the design sec 4 row to
                # norm_trace.jsonl at mark_refreshed() time and NormTraceWriter
                # rejects any extra key (amendment A-9). `sample` reaches
                # result.json through soft_summary["probe_samples"] either way.
                adapter.write_trace_row(row, sample)
                cb_state["probes"] += 1
                if monitor.should_freeze(overflow, adapter.tau_rel):
                    # This skips the refresh block below, leaving the adapter in
                    # needs_refresh(). Deliberate: the GP loop is over, nothing
                    # evaluates the objective again, and the freeze path reads
                    # positions rather than gradients (amendment D-13).
                    raise _FreezeReached(snapshot(iteration, pos, overflow,
                                                  "criterion", res.io_count, argmax))
            if discrete or adapter.needs_refresh():
                refresh_nesterov_secant(placer.optimizer)
                adapter.mark_refreshed()
                cb_state["num_refreshes"] += 1
            if check_invariant and not cb_state["installed"]:
                cleanup.callback(install_version_invariant(placer.optimizer, adapter))
                cb_state["installed"] = True

        _install_attribute(cleanup, placer, "iteration_callback", cb)
        lr = params.global_place_stages[0]["learning_rate"]
        try:
            placer(params, placedb, lr)
            pos = placer.pos[0]
            argmax = centre_argmax(pos)
            res = ctx.evaluate(pos.data[:n_phys], pos.data[n_all:n_all + n_phys])
            monitor.observe(cb_state["last_iteration"] + 1, argmax)
            shot = snapshot(cb_state["last_iteration"], pos, cb_state["overflow"],
                            "gp_end", res.io_count, argmax)
        except _FreezeReached as event:
            shot = event.snapshot
        density_weight_soft = float(placer.model.density_weight.detach().max())
        gp_phase.__exit__(None, None, None)

        with timer.phase("freeze"):
            centers_scaled = region_centers(rs_scaled)
            cx = shot["node_x"][:m] + np.asarray(placedb.node_size_x[:m], dtype=np.float64) / 2.0
            cy = shot["node_y"][:m] + np.asarray(placedb.node_size_y[:m], dtype=np.float64) / 2.0
            part, repaired = ensure_nonempty_regions(shot["argmax"], k, cx, cy,
                                                     centers_scaled)
            native_x = _to_native(shot["node_x"], shift[0], scale)
            native_y = _to_native(shot["node_y"], shift[1], scale)
            size_x_native = np.asarray(placedb.node_size_x[:m], dtype=np.float64) / scale
            size_y_native = np.asarray(placedb.node_size_y[:m], dtype=np.float64) / scale
            soft_path = os.path.join(out_dir, SOFT_NPZ)
            membership_path = os.path.join(out_dir, FROZEN_MEMBERSHIP_NPZ)
            save_positions(soft_path, native_x, native_y, die=die_native,
                           shift_factor=shift, scale_factor=scale,
                           placedb_sha256=sha, kind="soft")
            save_membership(membership_path, part, source="freeze", k=k,
                            seed=seed, epsilon=0.0)
            record = freeze_record(
                iteration=shot["iteration"], reason=shot["reason"],
                overflow=shot["overflow"], tau=shot["tau"], tau_rel=shot["tau_rel"],
                churn=shot["churn"], k=k, io_soft=shot["io_count"],
                membership_npz=os.path.abspath(membership_path),
                soft_npz=os.path.abspath(soft_path),
                repaired_empty_regions=repaired,
                gp_iterations_soft=cb_state["last_iteration"] + 1,
                density_weight_soft=density_weight_soft,
                stats=region_cell_stats(part, size_x_native, size_y_native, rs_native))
            save_freeze(os.path.join(out_dir, FREEZE_JSON), record)
        detach_terms(params)

    return {"freeze": record, "part": part, "soft_npz": soft_path,
            "membership_npz": membership_path, "regions_json":
            os.path.join(out_dir, REGIONS_JSON), "region_source": region_source,
            "init": init_info, "prior": prior_info,
            "norm_policy": norm_policy, "trace_path": adapter.trace_path,
            "probe_samples": cb_state["probe_samples"],
            "num_probes": cb_state["probes"],
            "num_refreshes": cb_state["num_refreshes"],
            "gp_iterations_soft": cb_state["last_iteration"] + 1,
            "density_weight_soft": density_weight_soft,
            "placedb_sha256": sha, "die_native": die_native}


def run_fence_gp(config_json, out_dir, *, region_set, part, positions,
                 reference_density_weight, k, density_clamp_lo=0.25,
                 density_clamp_hi=4.0, dp_seed=None, deterministic=None,
                 extra_terms=(), timer=None):
    """Phase 3 + phase 4 + evaluator.

    `extra_terms` is the documented attachment point for P-D's capacity term
    and P-E's pseudo-FT term. It is empty by default: after the freeze the IO
    and FT terms are OFF (design v2 sec 3, "terms after the freeze").
    """
    import torch
    timer = PhaseTimer() if timer is None else timer
    clamp_log = []

    with _io_cleanup() as cleanup:
        with timer.phase("read_fence"):
            params, placedb, info = build_fence_placedb(
                config_json, region_set, part, positions, dp_seed=dp_seed,
                deterministic=deterministic)
            cleanup.callback(detach_terms, params)
            scale_fields = _effective_scale_fields(params, placedb)
        assert_optimizer_lock(params)
        import NonLinearPlace

        gp_phase = timer.phase("gp_fence")
        gp_phase.__enter__()
        nl = netlist_from_placedb(placedb)
        rs_scaled = scaled_region_set(region_set, info["shift_factor"],
                                      info["scale_factor"])
        rg = RegionGrid(rs_scaled)
        ctx = GpuEvalContext(nl, rg, device="cuda")
        install_density_weight_clamp(cleanup, reference_density_weight, clamp_log,
                                     lo=density_clamp_lo, hi=density_clamp_hi)
        if extra_terms:
            attach_terms(params, list(extra_terms))
        np.random.seed(params.random_seed)
        placer = NonLinearPlace.NonLinearPlace(params, placedb, None)

        n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
        holder = {}
        original_legalize = placer.op_collections.legalize_op

        def timed_legalize(pos):
            gp_phase.__exit__(None, None, None)
            with torch.no_grad():
                holder["hpwl_gp"] = float(placer.op_collections.hpwl_op(pos))
            # exact io(fence GP): measured on the GP positions handed to LG, so
            # lg_loss can never be contaminated by a stale periodic callback.
            holder["io_fence_gp"] = int(ctx.evaluate(
                pos.data[:n_phys], pos.data[n_all:n_all + n_phys]).io_count)
            with timer.phase("lg"):
                out = original_legalize(pos)
            with torch.no_grad():
                holder["hpwl_lg"] = float(placer.op_collections.hpwl_op(out))
            return out

        _install_attribute(cleanup, placer.op_collections, "legalize_op", timed_legalize)
        cb_state = {"last_iteration": -1}

        def cb(iteration, pos):
            cb_state["last_iteration"] = iteration

        _install_attribute(cleanup, placer, "iteration_callback", cb)
        lr = params.global_place_stages[0]["learning_rate"]
        placer(params, placedb, lr)
        # Key the guard off *this* phase's own name: `timer` is shared across
        # all four phases here, so testing for "lg" would silently stop closing
        # gp_fence the day any other phase is named lg (amendment D-9).
        if "gp_fence" not in timer.phases:
            gp_phase.__exit__(None, None, None)

        with timer.phase("eval"):
            node_x, node_y = extract_final_positions(placer, placedb)
            legal_fields = _legalization_diagnostics(placer, placedb, params,
                                                     node_x, node_y)
            res = ctx.evaluate(node_x, node_y)
            metrics = _pack_eval_metrics(res)
            evaluation_path = os.path.join(out_dir, EVALUATION_NPZ)
            save_evaluation(evaluation_path, nl, rg, res, node_x, node_y,
                            placedb.net_names,
                            provenance={"config": os.path.abspath(config_json),
                                        "placement_stage": "fence_gp_lg",
                                        "shift_factor": list(info["shift_factor"]),
                                        "scale_factor": info["scale_factor"]})
            placement_path = os.path.join(out_dir, PLACEMENT_NPZ)
            np.savez_compressed(placement_path, node_x=node_x, node_y=node_y)
            final_overflow = float(placer.model.overflow.max())
            m = placedb.num_movable_nodes
            compliance = fence_compliance(
                rg, node_x, node_y, part,
                np.asarray(placedb.node_size_x[:m], dtype=np.float64),
                np.asarray(placedb.node_size_y[:m], dtype=np.float64))
            # Native units, native RegionSet: region_area/region_cell_area are
            # not scale-invariant, and freeze.json already carries them in the
            # native frame via region_cell_stats -- result.json must quote the
            # same numbers (amendment D-2). fence_compliance above stays in the
            # scaled frame because it is pure geometry against `rg`.
            size_x_native = (np.asarray(placedb.node_size_x[:m], dtype=np.float64)
                             / info["scale_factor"])
            size_y_native = (np.asarray(placedb.node_size_y[:m], dtype=np.float64)
                             / info["scale_factor"])
            balance = region_area_balance(part, size_x_native, size_y_native,
                                          region_set)
        detach_terms(params)

    io_fence_gp = holder.get("io_fence_gp")
    return {"metrics": metrics,
            "io_fence_gp": metrics["io_count"] if io_fence_gp is None else io_fence_gp,
            # Where io_fence_gp actually came from. io_identity_residual is 0
            # for any three inputs and can never detect the fallback; this can
            # (amendment D-1). "fallback" means legalize_op never fired -- the
            # config had legalize_flag=0 -- so io_fence_gp is the post-"LG"
            # number and lg_loss is 0 by definition rather than by measurement.
            "io_fence_gp_source": "fallback" if io_fence_gp is None else "legalize_op",
            "hpwl_gp": holder.get("hpwl_gp"), "hpwl_lg": holder.get("hpwl_lg"),
            "fence_compliance": compliance, "region_area_balance": balance,
            "density_weight_clamp": clamp_log, "escape_cell": info["escape_cell"],
            "escape_from": info["escape_from"], "legal_fields": legal_fields,
            "scale_fields": scale_fields, "final_overflow": final_overflow,
            "stop_overflow_reached": _stop_overflow_reached(final_overflow,
                                                            params.stop_overflow),
            "gp_iterations_fence": cb_state["last_iteration"] + 1,
            "gp_iteration_budget": _gp_iteration_budget(params),
            "placement_npz": placement_path, "evaluation_npz": evaluation_path,
            "params_seed": int(params.random_seed),
            "deterministic": int(params.deterministic_flag)}


def run_main_flow(config_json, out_dir, *, k=16, rtype="grid", seed=0,
                  regions_json=None, phase="all", init="die_center",
                  seed_npz=None, membership_npz=None, remap_blocks="auto",
                  norm_policy="legacy", norm_target_share=None, every=50,
                  home_period=None, rho_max=0.1,
                  f_ft_max=0.0, tau_hi=0.30, tau_lo=0.03, of_on=0.90,
                  of_end=None, of_full=0.20, ft_ramp_mode="window",
                  tau_start=0.12, tau_full=0.05, freeze_window=50,
                  freeze_overflow=0.15, freeze_tau_rel=0.05, freeze_churn=0.005,
                  argmax_chunk=4, density_clamp_lo=0.25, density_clamp_hi=4.0,
                  ignore_net_degree=None, w_mode="unit", dp_seed=None,
                  deterministic=None, check_invariant=False,
                  benchmark_kind="real", extra_terms=()):
    import torch
    if phase not in ("all", "soft", "fence"):
        raise ValueError(f"unknown phase {phase!r}")
    if init not in INIT_MODES:
        raise ValueError(f"unknown init {init!r}")
    if norm_policy not in NORM_POLICIES:
        raise ValueError(f"unknown norm policy {norm_policy!r}")
    os.makedirs(out_dir, exist_ok=True)
    device_baseline_gb = 0.0
    if torch.cuda.is_available():
        free0, total0 = torch.cuda.mem_get_info()
        device_baseline_gb = (total0 - free0) / 2 ** 30
    t0 = time.time()
    timer = PhaseTimer()
    sampler = DeviceMemSampler()
    sampler.start()
    try:
        soft, soft_summary = None, None
        if phase in ("all", "soft"):
            soft = run_soft_phase(
                config_json, out_dir, k=k, rtype=rtype, seed=seed,
                regions_json=regions_json, init=init, seed_npz=seed_npz,
                membership_npz=membership_npz, remap_blocks=remap_blocks,
                norm_policy=norm_policy, norm_target_share=norm_target_share,
                every=every, home_period=home_period,
                rho_max=rho_max, f_ft_max=f_ft_max, tau_hi=tau_hi, tau_lo=tau_lo,
                of_on=of_on, of_end=of_end, of_full=of_full,
                ft_ramp_mode=ft_ramp_mode, tau_start=tau_start, tau_full=tau_full,
                freeze_window=freeze_window, freeze_overflow=freeze_overflow,
                freeze_tau_rel=freeze_tau_rel, freeze_churn=freeze_churn,
                argmax_chunk=argmax_chunk,
                ignore_net_degree=ignore_net_degree, w_mode=w_mode,
                dp_seed=dp_seed, deterministic=deterministic,
                check_invariant=check_invariant, timer=timer)
            # Everything the soft phase decided that result.json would otherwise
            # lose: region_source, the resolved init record, the prior
            # (including `remapped`, which is the --remap-blocks auto decision
            # a reviewer of arms (a)/(b) will ask about), the probe samples and
            # which trace file the policy wrote (amendment D-6). `part` is a
            # numpy array and `freeze` is already result["freeze"].
            soft_summary = {key: value for key, value in soft.items()
                            if key not in ("part", "freeze")}
            torch.cuda.empty_cache()
            if phase == "soft":
                with open(os.path.join(out_dir, SOFT_RESULT_JSON), "w") as stream:
                    json.dump({key: value for key, value in soft.items()
                               if key != "part"}, stream, indent=1, default=str)
                return soft

        soft_path = os.path.join(out_dir, SOFT_NPZ)
        freeze_path = os.path.join(out_dir, FREEZE_JSON)
        membership_path = os.path.join(out_dir, FROZEN_MEMBERSHIP_NPZ)
        regions_path = regions_json or os.path.join(out_dir, REGIONS_JSON)
        for required in (soft_path, freeze_path, membership_path, regions_path):
            if not os.path.exists(required):
                raise FileNotFoundError(
                    f"--phase fence needs {os.path.basename(required)} in {out_dir}; "
                    "run --phase soft (or --phase all) first")
        record = load_freeze(freeze_path)
        positions = load_positions(soft_path)
        region_set = RegionSet.from_json(regions_path)
        region_set.validate()
        part = load_membership(membership_path, expect_k=region_set.k,
                               require_nonempty=True).part

        fence = run_fence_gp(
            config_json, out_dir, region_set=region_set, part=part,
            positions=positions,
            reference_density_weight=record["density_weight_soft"],
            k=region_set.k, density_clamp_lo=density_clamp_lo,
            density_clamp_hi=density_clamp_hi, dp_seed=dp_seed,
            deterministic=deterministic, extra_terms=extra_terms, timer=timer)
    finally:
        sampler.stop()

    accounting = io_accounting(record["io_soft"], fence["io_fence_gp"],
                               fence["metrics"]["io_count"])
    artefacts = {name: {"path": os.path.abspath(path),
                        "sha256": file_sha256(path)}
                 for name, path in (
                     ("regions_json", regions_path), ("soft_npz", soft_path),
                     ("freeze_json", freeze_path),
                     ("membership_npz", membership_path),
                     ("placement_npz", fence["placement_npz"]),
                     ("evaluation_npz", fence["evaluation_npz"]),
                     # whichever the policy wrote (amendment A-9); the
                     # os.path.exists filter drops the other
                     ("norm_trace", os.path.join(out_dir, NORM_TRACE)),
                     ("legacy_trace", os.path.join(out_dir, LEGACY_TRACE)))
                 if os.path.exists(path)}
    result = {
        **fence["metrics"], **accounting,
        "mode": "main_flow", "schema_version": MAIN_FLOW_RESULT_SCHEMA_VERSION,
        "config": os.path.abspath(config_json), "k": int(region_set.k),
        "rtype": rtype, "seed": seed, "init": init, "norm_policy": norm_policy,
        "phase": phase, "regions_json": os.path.abspath(regions_path),
        "dp_seed": fence["params_seed"], "det": fence["deterministic"],
        "runtime_s": time.time() - t0,
        "hpwl_gp": fence["hpwl_gp"], "hpwl_lg": fence["hpwl_lg"],
        "fence_compliance": fence["fence_compliance"]["lower_left"],
        "fence_compliance_center": fence["fence_compliance"]["center"],
        "region_area_balance": fence["region_area_balance"],
        "freeze": record, "density_weight_clamp": fence["density_weight_clamp"],
        "escape_cell": {"index": fence["escape_cell"], "from": fence["escape_from"]},
        "gp_iterations_soft": record["gp_iterations_soft"],
        "gp_iterations_fence": fence["gp_iterations_fence"],
        "gp_iteration_budget": fence["gp_iteration_budget"],
        "io_fence_gp_source": fence["io_fence_gp_source"],
        "soft_summary": soft_summary,
        "final_overflow": fence["final_overflow"],
        "stop_overflow_reached": fence["stop_overflow_reached"],
        "artifacts": artefacts,
        **fence["legal_fields"],
        **{key: fence["scale_fields"][key] for key in
           ("effective_target_density", "num_filler_nodes", "num_bins_x", "num_bins_y")},
        **phase_summary(timer, sampler),
        **_t8a_provenance(config_json, benchmark_kind=benchmark_kind,
                          device_baseline_gb=device_baseline_gb),
    }
    # _t8a_provenance carries run_placement's own RESULT_SCHEMA_VERSION (3);
    # it is spread last, so restore this driver's schema version afterwards.
    result["schema_version"] = MAIN_FLOW_RESULT_SCHEMA_VERSION
    save_result(os.path.join(out_dir, RESULT_JSON), result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="v2 main flow: soft GP -> freeze -> fence GP -> fence LG -> evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase", choices=["all", "soft", "fence"], default="all")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regions", default=None,
                        help="producer regions.json (native post-read units); "
                             "omit to use the built-in grid")
    parser.add_argument("--init", choices=list(INIT_MODES), default="die_center")
    parser.add_argument("--seed-npz", default=None)
    parser.add_argument("--membership", default=None)
    parser.add_argument("--remap-blocks", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--norm-policy", choices=list(NORM_POLICIES), default="legacy")
    parser.add_argument("--norm-target-share", default=None,
                        help="policy B force shares as fractions of the total "
                             "force, e.g. 'io=0.3,ft=0.1'; default io=--rho-max, "
                             "ft=--rho-max*--f-ft-max. Ignored by --norm-policy "
                             "legacy")
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--home-period", type=int, default=None)
    parser.add_argument("--rho-max", type=float, default=0.1)
    parser.add_argument("--f-ft-max", type=float, default=0.0)
    parser.add_argument("--tau-hi", type=float, default=0.30)
    parser.add_argument("--tau-lo", type=float, default=0.03)
    parser.add_argument("--of-on", type=float, default=0.90)
    parser.add_argument("--of-end", type=float, default=None)
    parser.add_argument("--of-full", type=float, default=0.20)
    parser.add_argument("--ft-ramp-mode", choices=["window", "constant"], default="window")
    parser.add_argument("--tau-start", type=float, default=0.12)
    parser.add_argument("--tau-full", type=float, default=0.05)
    parser.add_argument("--freeze-window", type=int, default=50)
    parser.add_argument("--freeze-overflow", type=float, default=0.15)
    parser.add_argument("--freeze-tau-rel", type=float, default=0.05)
    parser.add_argument("--freeze-churn", type=float, default=0.005)
    parser.add_argument("--argmax-chunk", type=int, default=4,
                        help="regions per chunk in the freeze argmax; the "
                             "unchunked (N, R) + (N, K) intermediates are "
                             "~7.7 GB at 30M cells x K=16 in float64")
    parser.add_argument("--density-clamp-lo", type=float, default=0.25)
    parser.add_argument("--density-clamp-hi", type=float, default=4.0)
    parser.add_argument("--d-max", type=int, default=None, dest="ignore_net_degree")
    parser.add_argument("--w-mode", default="unit", choices=["unit", "inv_deg"])
    parser.add_argument("--dp-seed", type=int, default=None)
    parser.add_argument("--deterministic", type=int, default=None)
    parser.add_argument("--check-invariant", action="store_true")
    parser.add_argument("--benchmark-kind", default="real", choices=["real", "synthetic"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_main_flow(
        args.config, args.out_dir, k=args.k, rtype=args.rtype, seed=args.seed,
        regions_json=args.regions, phase=args.phase, init=args.init,
        seed_npz=args.seed_npz, membership_npz=args.membership,
        remap_blocks=args.remap_blocks, norm_policy=args.norm_policy,
        norm_target_share=args.norm_target_share,
        every=args.every, home_period=args.home_period, rho_max=args.rho_max,
        f_ft_max=args.f_ft_max, tau_hi=args.tau_hi, tau_lo=args.tau_lo,
        of_on=args.of_on, of_end=args.of_end, of_full=args.of_full,
        ft_ramp_mode=args.ft_ramp_mode, tau_start=args.tau_start,
        tau_full=args.tau_full, freeze_window=args.freeze_window,
        freeze_overflow=args.freeze_overflow, freeze_tau_rel=args.freeze_tau_rel,
        freeze_churn=args.freeze_churn, argmax_chunk=args.argmax_chunk,
        density_clamp_lo=args.density_clamp_lo,
        density_clamp_hi=args.density_clamp_hi,
        ignore_net_degree=args.ignore_net_degree, w_mode=args.w_mode,
        dp_seed=args.dp_seed, deterministic=args.deterministic,
        check_invariant=args.check_invariant, benchmark_kind=args.benchmark_kind)
    return result


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_driver.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 5: Verify the CLI help renders (no import-time errors)**

Run: `"$IOPLACE_PYTHON" -m ioplace.drivers.run_main_flow --help`
Expected: the argparse help text, exit code 0.

- [ ] **Step 6: Run the fast suite**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow" -q`
Expected: PASS — no new failures.

- [ ] **Step 7: Commit**

```bash
git add src/ioplace/drivers/run_main_flow.py tests/test_main_flow_driver.py
git commit -m "feat(driver): v2 main flow run_main_flow.py

Two DREAMPlace instances, four phases, artefacts in between. Phase 1 runs with
legalize_flag=0 and stops by raising _FreezeReached from the iteration callback
(NonLinearPlace ignores the callback's return value); phase 3 rebuilds the
PlaceDB, injects the frozen membership, warm starts from soft.npz, clamps the
re-derived density weight and runs WL+density+fence only. io(fence GP) is
measured inside the legalize_op wrapper so lg_loss is exact.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Gate the retired GR-in-loop paths behind `IOPLACE_ENABLE_GR_IN_LOOP`

**Files:**
- Create: `src/ioplace/gr_in_loop.py`
- Modify: `src/ioplace/ops/routing_gp_controller.py` (**replace** P-H Task 8's inline `os.environ` check in `__init__` with a call to `require_gr_in_loop()`; no import-time gate)
- Modify: `src/scripts/run_route_gp.py` (add the import-time guard after the module docstring)
- Modify: `tests/test_routing_gp_driver.py:41`, `:105`, `:127`, `:149`, `:166`
- Modify: `tests/test_bounded_grt_feedback.py:48-49`
- Test: `tests/test_gr_in_loop_gate.py`
- Do **not** modify: `tests/test_routing_gp_retirement.py` (P-H Task 8 owns it)

**Interfaces:**
- Produces: `GR_IN_LOOP_ENV = "IOPLACE_ENABLE_GR_IN_LOOP"`, `require_gr_in_loop() -> None`
- Consumed by: `ioplace.ops.routing_gp_controller.RoutingGPController.__init__`, `src/scripts/run_route_gp.py`.

Spec §1 "Retired" puts `src/scripts/run_route_gp.py`, `ops/routing_gp_controller.py`, `ops/route_gp.py`, `ops/joint_route_feedback.py` and `ops/route_feedback.py` behind `IOPLACE_ENABLE_GR_IN_LOOP=1` as unmaintained. This task gates the two modules named in the P-B scope; the remaining three stay importable because live unit tests exercise them directly as libraries.

**Reconciliation with P-H Task 8 (2026-09-19) — relocate, do not duplicate.** P-H lands before P-B and its Task 8 already installs an `IOPLACE_ENABLE_GR_IN_LOOP` check, but *inside* `RoutingGPController.__init__`, not at import time. Two consequences:

1. **The controller keeps a constructor gate, not an import gate.** P-H ships `tests/test_routing_gp_retirement.py`, whose module-level `from ioplace.ops.routing_gp_controller import RoutingGPController` would fail at collection if this task added an import-time guard to that module. So this task *moves* P-H's four-line `if os.environ.get(...) != "1": raise RuntimeError(...)` block out of `__init__` and into `gr_in_loop.require_gr_in_loop()`, leaving a one-line call at the same place (before the `mode` check). Same env var, **same message text, copied verbatim from P-H**, so `test_controller_is_retired_unless_the_escape_hatch_is_set` keeps passing untouched. The import-time guard is added only to `src/scripts/run_route_gp.py`, whose module docstring is the only thing P-H Task 8 changes there.
2. **P-H Task 8 already edits `tests/test_routing_gp_driver.py:41` — check before applying that one line.** P-H Task 8's Files list (`docs/superpowers/plans/2026-09-19-v2-p-h-normalisation.md:2242`) reads `Modify: tests/test_routing_gp_driver.py:41 (subprocess env= for the now-gated construction)`, and P-H's ledger ruling E-1 repeats it. An earlier draft of this paragraph claimed P-H modifies no existing test module; that was false and is corrected here (pre-flight amendment A-12). So before applying Step 5's line-41 edit, **read the line**: if the `env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"}` argument is already present, leave it alone and say so in the task report; if it is not — P-H Task 8 had not landed when this text was written, and line 41 was then still the un-envved `subprocess.run(command, capture_output=True, text=True)` — apply it. The other four sites in that module and the one in `tests/test_bounded_grt_feedback.py` are this task's alone: neither test constructs a `RoutingGPController`, so P-H's constructor gate never fires there and they break only because of the `run_route_gp.py` import guard added in Step 4. What this task must still *skip* is duplicating P-H's constructor-gate coverage — `tests/test_gr_in_loop_gate.py` asserts only that the controller *delegates* to `require_gr_in_loop`, never re-testing the `RuntimeError` message.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gr_in_loop_gate.py`:

```python
import importlib
import os
import subprocess
import sys
import pytest

from ioplace.paths import REPO_ROOT


def test_require_gr_in_loop_rejects_an_unset_env(monkeypatch):
    from ioplace.gr_in_loop import GR_IN_LOOP_ENV, require_gr_in_loop
    assert GR_IN_LOOP_ENV == "IOPLACE_ENABLE_GR_IN_LOOP"
    monkeypatch.delenv(GR_IN_LOOP_ENV, raising=False)
    with pytest.raises(RuntimeError, match="IOPLACE_ENABLE_GR_IN_LOOP"):
        require_gr_in_loop()
    monkeypatch.setenv(GR_IN_LOOP_ENV, "1")
    require_gr_in_loop()


def test_the_controller_gate_routes_through_require_gr_in_loop(monkeypatch):
    """P-H Task 8 owns the controller's RuntimeError and its message
    (tests/test_routing_gp_retirement.py). This only pins that the controller
    delegates to gr_in_loop instead of re-reading os.environ, so the repo has
    exactly one gate with exactly one message."""
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    module = importlib.import_module("ioplace.ops.routing_gp_controller")
    calls = []
    monkeypatch.setattr(module, "require_gr_in_loop", lambda: calls.append(1))
    module.RoutingGPController(object(), object())
    assert calls == [1]


def test_run_route_gp_cli_is_gated(tmp_path):
    script = os.path.join(str(REPO_ROOT), "src/scripts/run_route_gp.py")
    env = {key: value for key, value in os.environ.items()
           if key != "IOPLACE_ENABLE_GR_IN_LOOP"}
    blocked = subprocess.run([sys.executable, script, "--help"],
                             capture_output=True, text=True, env=env)
    assert blocked.returncode != 0
    assert "IOPLACE_ENABLE_GR_IN_LOOP" in blocked.stderr
    env["IOPLACE_ENABLE_GR_IN_LOOP"] = "1"
    allowed = subprocess.run([sys.executable, script, "--help"],
                             capture_output=True, text=True, env=env)
    assert allowed.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_gr_in_loop_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.gr_in_loop'`

- [ ] **Step 3: Write the guard module**

Create `src/ioplace/gr_in_loop.py`:

```python
"""The one gate for the retired GR-in-loop paths (design v2 sec 1, "Retired").

`src/scripts/run_route_gp.py` and `ioplace/ops/routing_gp_controller.py` are
not part of the v2 flow and are unmaintained. They stay in the tree (the
round-feedback results reference them) but must be opted into explicitly so
no v2 driver picks them up by accident.

The message below is P-H Task 8's, moved here verbatim from
`RoutingGPController.__init__`: one env var, one message, two call sites (the
controller at construction time, the CLI script at import time).
"""
import os

GR_IN_LOOP_ENV = "IOPLACE_ENABLE_GR_IN_LOOP"


def require_gr_in_loop():
    if os.environ.get(GR_IN_LOOP_ENV) != "1":
        raise RuntimeError(
            "in-loop GR is retired by the v2 design (sec 1): the final GRT "
            "protocol runs once, after placement. Set "
            "IOPLACE_ENABLE_GR_IN_LOOP=1 to use this unmaintained path.")
```

- [ ] **Step 4: Relocate P-H's check, then guard the script**

In `src/ioplace/ops/routing_gp_controller.py`, add `from ioplace.gr_in_loop import require_gr_in_loop` to the project imports and **delete** the block P-H Task 8 put at the top of `__init__`:

```python
        if os.environ.get("IOPLACE_ENABLE_GR_IN_LOOP") != "1":
            raise RuntimeError(
                "in-loop GR is retired by the v2 design (sec 1): the final GRT "
                "protocol runs once, after placement. Set "
                "IOPLACE_ENABLE_GR_IN_LOOP=1 to use this unmaintained path.")
```

replacing it, at the same position (before the `mode` check), with:

```python
        require_gr_in_loop()
```

Drop the now-unused `import os` if nothing else in the module uses it. Do **not** add an import-time guard to this module: `tests/test_routing_gp_retirement.py` (P-H) imports `RoutingGPController` at module level and would fail at collection.

In `src/scripts/run_route_gp.py`, insert immediately after the module docstring (the one P-H Task 8 rewrote) and before `import argparse`:

```python
from ioplace.gr_in_loop import require_gr_in_loop

require_gr_in_loop()
```

- [ ] **Step 5: Update the tests that legitimately use the retired paths**

In `tests/test_routing_gp_driver.py`:
- line 41 — **check first**: P-H Task 8's Files list already claims this edit (see the reconciliation note above). If the `env=` argument is there, skip this bullet. Otherwise change `run = subprocess.run(command, capture_output=True, text=True)` to
  `run = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"})`
- line 105 — add `env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"},` to that `subprocess.run(...)` call
- line 127 — change `result = subprocess.run(command, capture_output=True, text=True)` to
  `result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "IOPLACE_ENABLE_GR_IN_LOOP": "1"})`
- line 148 — change the signature to
  `def test_publication_report_leaves_early_stop_observation_pending(monkeypatch):` and insert
  `monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")` before `from scripts.run_route_gp import observation_publication`
- line 163 — change the signature to
  `def test_saved_tile_row_gap_checkpoint_repairs_without_running_gp(tmp_path, monkeypatch):` and insert
  `monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")` before `from scripts.run_route_gp import repair_routed_snapshot`

In `tests/test_bounded_grt_feedback.py`:
- line 48 — change the signature to
  `def test_final_routing_does_not_inherit_fast_feedback_policy(monkeypatch):` and insert
  `monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")` before `from scripts.run_route_gp import routing_policy`

- [ ] **Step 6: Run the gate test and the two touched test modules**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_gr_in_loop_gate.py tests/test_routing_gp_retirement.py tests/test_routing_gp_driver.py tests/test_bounded_grt_feedback.py -v -m "not slow"`
Expected: PASS — the gate's 3 tests pass; P-H's 5 retirement tests still pass **unmodified** (that is the check that the relocation preserved the message); the previously passing fast tests in the two driver modules still pass (slow ones deselected).

- [ ] **Step 7: Run the fast suite**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow" -q`
Expected: PASS — no new failures.

- [ ] **Step 8: Commit**

```bash
git add src/ioplace/gr_in_loop.py src/ioplace/ops/routing_gp_controller.py \
        src/scripts/run_route_gp.py tests/test_gr_in_loop_gate.py \
        tests/test_routing_gp_driver.py tests/test_bounded_grt_feedback.py
git commit -m "chore(gr-in-loop): one gate for the retired routing-GP paths

Moves P-H's inline IOPLACE_ENABLE_GR_IN_LOOP check out of
RoutingGPController.__init__ into ioplace.gr_in_loop.require_gr_in_loop, so
there is a single env var with a single message, and adds the same guard at
import time to run_route_gp.py (design v2 sec 1 'Retired'). The tests that
still exercise the script set the flag explicitly.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end main flow on GCD (slow)

**Files:**
- Modify: `tests/test_main_flow_driver.py` (append the slow test)

**Interfaces:**
- Consumes: `run_main_flow` (Task 7); `artifacts.MAIN_FLOW_RESULT_FIELDS` (Task 1).

**Case.** GCD at `K=4` grid, `results/route_feedback_20260914/gcd.json`. 508 movable cells, 168 terminals, 54 terminal-NIs — note that `placedb.node2fence_region_map` has length `num_movable + num_terminals = 676` after `read()`, exactly what `inject_fence_regions` writes, so the terminal-NIs are not an issue. A probe of the fence flow on this case (grid K=4, 120 GP iterations) reached `fence_compliance` 0.998 with a per-region cell split of `[147, 138, 81, 142]` and a `(K+1,)` density-weight vector, so the thresholds below are comfortable.

**Do not assert `legalization_status == "success"`.** The single escape cell parked in the implicit no-fence bucket has no fence legaliser of its own; on the probe run the greedy legaliser reported `num_unplaced_cells = 1` for that bucket and the legality check failed on one node. That is a property of the DREAMPlace workaround, not of this flow, and `num_unplaced_cells` in `result.json` counts only cells outside the die (`run_placement._legalization_diagnostics`), which stays 0.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_main_flow_driver.py`:

```python
import json
from pathlib import Path


@pytest.mark.slow
def test_main_flow_end_to_end_on_gcd_closes_the_io_identity(tmp_path):
    """Design v2 sec 9's small end-to-end case: producer-free grid K=4 on GCD,
    full main flow, asserting the artefacts exist and
    io(final) = io(soft) + io_delta_at_freeze + lg_loss."""
    from ioplace.artifacts import MAIN_FLOW_RESULT_FIELDS
    from ioplace.paths import REPO_ROOT
    config = Path(REPO_ROOT) / "results/route_feedback_20260914/gcd.json"
    if not config.exists():
        pytest.skip("GCD benchmark required")
    out = tmp_path / "gcd_k4"
    result = run_main_flow(str(config), str(out), k=4, rtype="grid", seed=0,
                           init="die_center", every=25, freeze_window=50,
                           rho_max=0.05, dp_seed=1000, deterministic=1)

    # the normalisation trace's file name is the policy's, not a constant:
    # legacy writes legacy_trace.jsonl, grandplan/adaptive norm_trace.jsonl
    # (amendment A-9). This run is legacy, the driver's default.
    trace_name = ("legacy_trace.jsonl" if result["norm_policy"] == "legacy"
                  else "norm_trace.jsonl")
    for name in ("regions.json", "soft.npz", "freeze.json",
                 "frozen_membership.npz", "placement.npz", "evaluation.npz",
                 trace_name, "result.json"):
        assert (out / name).exists(), name
    on_disk = json.loads((out / "result.json").read_text())
    for field in MAIN_FLOW_RESULT_FIELDS:
        assert field in on_disk, field
    assert on_disk["mode"] == "main_flow" and on_disk["schema_version"] == 1

    # The accounting identity (design v2 sec 7) is algebraic:
    # io_identity_residual is 0 for *any* three inputs, so it is asserted as
    # the schema invariant it is, and the real check comes from provenance
    # (amendment D-1) -- io_fence_gp must be the legalize_op wrapper's exact
    # pre-LG measurement, and io_soft must be the number freeze.json recorded.
    assert result["io_identity_residual"] == 0
    assert result["io_fence_gp_source"] == "legalize_op"
    assert result["io_soft"] == result["freeze"]["io_soft"]
    assert result["io_count"] == result["io_fence_gp"] + result["lg_loss"]

    freeze = result["freeze"]
    assert freeze["reason"] in ("criterion", "gp_end")
    assert freeze["k"] == 4 and len(freeze["region_cell_count"]) == 4
    assert min(freeze["region_cell_count"]) >= 1          # no empty fence region
    assert freeze["density_weight_soft"] > 0

    assert result["fence_compliance_center"] >= 0.9
    assert result["region_area_balance"]["utilization_ratio"] is not None
    assert result["region_area_balance"]["empty_regions"] == []
    assert result["density_weight_clamp"] and "bound" in result["density_weight_clamp"][0]
    assert result["hpwl"] > 0 and result["hpwl_gp"] > 0 and result["hpwl_lg"] > 0
    assert result["gp_iterations_soft"] >= 1 and result["gp_iterations_fence"] >= 1
    assert result["peak_mem_mb"] > 0 and result["t_gp_fence"] > 0
    assert set(result["artifacts"]) >= {"soft_npz", "freeze_json",
                                        "membership_npz", "placement_npz",
                                        "evaluation_npz"}

    # Row assertions branch on the policy (amendment A-9): legacy_trace.jsonl
    # carries publish_atomic's keys plus the driver's per-probe extras;
    # norm_trace.jsonl carries exactly norm_trace.ROW_FIELDS and nothing else,
    # because NormTraceWriter validates every row. The else branch is what the
    # --norm-policy grandplan|adaptive arms of the spec section 4 ablation hit.
    from ioplace.norm_trace import ROW_FIELDS, read_norm_trace
    rows = read_norm_trace(str(out / trace_name))
    assert rows
    if result["norm_policy"] == "legacy":
        for row in rows:
            for key in ("iteration", "overflow", "tau", "tau_rel", "lambda_io",
                        "grad_l1_wl", "grad_l1_io", "obj_version", "policy",
                        "io_count", "ft_count", "churn"):
                assert key in row, key
    else:
        for row in rows:
            assert set(row) == set(ROW_FIELDS)
            assert row["policy"] == result["norm_policy"] and "io" in row["terms"]

    # soft-phase provenance survives into result.json (amendment D-6)
    summary = result["soft_summary"]
    assert summary["region_source"] == "builtin"
    assert summary["init"]["mode"] == "die_center" and summary["prior"] is None
    assert summary["num_probes"] == len(rows) == len(summary["probe_samples"])
    assert summary["trace_path"].endswith(trace_name)

    # --phase fence reproduces the fence half from the artefacts alone
    rerun = run_main_flow(str(config), str(out), phase="fence", k=4,
                          dp_seed=1000, deterministic=1)
    assert rerun["io_soft"] == result["io_soft"]
    assert rerun["io_count"] == result["io_count"]
    assert rerun["lg_loss"] == result["lg_loss"]
```

- [ ] **Step 2: Run the test to verify it fails or passes for the right reason**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_driver.py::test_main_flow_end_to_end_on_gcd_closes_the_io_identity -v`
Expected on first run: it exercises the real flow. If it fails, the failure must be a real defect in Tasks 1–7 — fix that code, not the assertions. The two failure modes to expect and how to handle them:
- `IndexError: index -1 is out of bounds for axis 0 with size 0`, raised by `np.percentile` inside `PlaceDB.calc_num_filler_for_fence_region` at `PlaceDB.py:687` → an empty region survived; `ensure_nonempty_regions` (Task 3) is not being applied to the membership that reaches `save_membership`. Do **not** go looking for a `ValueError` at `:729`: under the installed numpy (1.26.4) the percentile call raises before `int(round(nan))` is ever reached (pre-flight amendment E-3).
- `result["io_fence_gp_source"] == "fallback"` → `io_fence_gp` was not captured in the `legalize_op` wrapper, so it fell back to the post-LG count and `lg_loss` is 0 by definition rather than by measurement. The cause is `legalize_flag = 0` in phase 3; check the config. (`io_identity_residual` cannot report this — it is 0 either way.)

- [ ] **Step 3: Run the whole main-flow test module**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_driver.py -v`
Expected: PASS — 7 passed (6 fast + 1 slow). Wall time dominated by two GCD placements, roughly 2–5 minutes.

- [ ] **Step 4: Run the full suite**

Run: `"$IOPLACE_PYTHON" -m pytest -q`
Expected: PASS — no new failures relative to the pre-plan baseline. Record the baseline before starting if you have not already (`git stash` is not needed; just run the suite on `HEAD` once).

- [ ] **Step 5: Commit**

```bash
git add tests/test_main_flow_driver.py
git commit -m "test(main-flow): end-to-end GCD K=4 run closing the IO identity

Runs the full soft GP -> freeze -> fence GP -> fence LG -> evaluator flow on
the smallest real LEF/DEF case, asserts every artefact exists, that
io(final) = io(soft) + io_delta_at_freeze + lg_loss within +/-1, that no fence
region is empty, and that --phase fence reproduces the fence half from disk.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Documentation sync

**Files:**
- Modify: `docs/dev-env.md` (insert a `## Drivers` section immediately before `## Installed toolchain`)

- [ ] **Step 1: Insert the driver table**

Insert this section into `docs/dev-env.md` between the end of `## Run project commands` and the `## Installed toolchain` heading:

````markdown
## Drivers

| Driver | Entry point | What it runs | Status |
| --- | --- | --- | --- |
| Flat baseline | `ioplace.drivers.run_placement --mode flat` | one-shot GP+LG, no region terms | current |
| Legacy IO driver | `ioplace.drivers.run_placement --mode io` (`run_placement_io.py`) | single-phase GP with the soft IO/FT terms | legacy, frozen |
| Fence-from-start | `ioplace.drivers.run_placement --mode two_stage` | Mt-KaHyPar partition -> fences before GP; v2 arm (f) | current |
| **v2 main flow** | `python -m ioplace.drivers.run_main_flow` | soft GP -> freeze -> fence GP -> fence LG -> evaluator, two DREAMPlace instances, artefacts in between | current |
| GR-in-loop | `src/scripts/run_route_gp.py` | routing-gradient GP with OpenROAD feedback | retired; requires `IOPLACE_ENABLE_GR_IN_LOOP=1` |

The v2 main flow writes one directory per arm containing `regions.json`,
`soft.npz`, `freeze.json`, `frozen_membership.npz`, `placement.npz`,
`evaluation.npz`, `norm_trace.jsonl` and `result.json`. All cross-phase arrays
are in native post-read PlaceDB units (before `placedb.initialize()`'s
`scale()`); `evaluation.npz` stays in scaled evaluator units and records
`shift_factor`/`scale_factor`. `--phase {all,soft,fence}` re-runs either half
from the artefacts. Example:

```bash
source src/scripts/env.sh
export IOPLACE_MTKAHYPAR_THREADS=1 CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m ioplace.drivers.run_main_flow \
  --config results/route_feedback_20260914/gcd.json \
  --out-dir runs/gcd/ours --k 4 --rtype grid --init die_center \
  --norm-policy legacy
```
````

- [ ] **Step 2: Check the document still renders and the links are intact**

Run: `grep -n "^## " docs/dev-env.md`
Expected: the heading list now contains `## Drivers` between `## Run project commands` and `## Installed toolchain`.

- [ ] **Step 3: Commit**

```bash
git add docs/dev-env.md
git commit -m "docs(dev-env): add the driver table and the v2 main-flow invocation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Appendix: arm → CLI mapping (spec §8)

`K=16` only for the real matrix; the end-to-end test uses `K=4` for speed.

| Arm | Geometry | Init | Command fragment |
|---|---|---|---|
| (a) | grid 4×4 | region centres | `--k 16 --rtype grid --init region_center --membership <producer>/membership.npz` |
| (b) | producer | region centres | `--regions <producer>/regions.json --init region_center --membership <producer>/membership.npz` |
| (c) | grid 4×4 | flat seed | `--k 16 --rtype grid --init seed --seed-npz <producer>/seed.npz` |
| ours | producer | flat seed | `--regions <producer>/regions.json --init seed --seed-npz <producer>/seed.npz` |
| (e) | producer | flat seed | not this driver — faithful GrandPlan (fixed Mt-KaHyPar membership, grouping, fence GP) |
| (f) | producer | flat seed | not this driver — `run_placement_two_stage.py`, fence from the start |

For (a) and (b) the prior is the producer's Mt-KaHyPar membership, whose block
ids carry no geometry; `--remap-blocks auto` detects `source == "mtkahypar..."`
and routes it through `assign_blocks_to_regions` before use.

---

## Self-Review

**1. Spec coverage.**

| Spec item | Task |
|---|---|
| §1 artefact table, coordinate contract | Global Constraints + Task 1 |
| §1 `producer.json` I/O (payload built by P-C Task 10) | Task 1 (`PRODUCER_FIELDS`, `save_producer_json`/`load_producer_json`) |
| §1 two drivers, `run_placement_io.py` stays legacy | Global Constraints + Task 7 |
| §1 retired GR-in-loop behind `IOPLACE_ENABLE_GR_IN_LOOP` | Task 8 |
| §3 phase 1 warm start (`seed.npz` → `node_x`, flag 0) | Task 2 + Task 7 |
| §3 centre-init arms; region-centre init | Task 2 |
| §3 phase 2 freeze criterion (overflow ≤ 0.15, τ_rel ≤ 0.05, churn ≤ 0.5%/50 it) | Task 3 |
| §3 membership = argmax at the cell centre | Task 3 (`argmax_region` + `cell_centers`) |
| §3 `freeze.json` and `soft.npz` | Tasks 1, 3, 7 |
| §3 phase 3 rebuild + `inject_fence_regions` + escape cell + warm start | Task 4 |
| §3 density-weight clamp to [0.25×, 4×] with logging | Task 4 |
| §3 `obj_version`/`refresh_nesterov_secant` discipline | Task 5 + Task 7 |
| §3 IO and FT off after the freeze; capacity/pseudo-FT hook points | Task 7 (`extra_terms=()`) |
| §3 phase 4 automatic multi-fence LG, then evaluator | Task 7 |
| §4 normalisation via a module, not `schedules.py` directly | Task 5 |
| §7 `io_delta_at_freeze`, `fence_compliance`, closing identity | Tasks 6, 7, 9 |
| §8 arm definitions | Appendix + Task 7 CLI |
| §9 freeze tests, warm-start-after-scaling test, small end-to-end case | Tasks 3, 4, 9 |
| §9 B+C "done" (2×2 on `mempool_group`) | Global Constraints → Acceptance; campaign, not a task |

Out of scope by design and *not* gaps: `capacity.npz` and the capacity term (P-D), pseudo-FT (P-E), the remaining three straddle diagnostics and the soft-assign anchor change (P-F), the final GRT protocol (P-G), `TermNormalizer` itself (P-H), the region producer's own modules (P-C `producer/*` and `drivers/run_region_producer.py`), §3b's continuous upgrade path. Note the one exception carved out by the 2026-09-19 reconciliation: `src/ioplace/artifacts.py` is owned here and serves P-C too, so Task 1's `PRODUCER_FIELDS`/`save_producer_json` are in scope even though nothing in this plan writes a `producer.json`.

**2. Placeholder scan.** No `TBD`, no "add error handling", no "similar to Task N", no test described without code, no reference to an undefined symbol. Every code step is complete, runnable source.

**3. Type consistency.** Cross-task symbol audit performed and three inconsistencies were found and fixed inline before this section was written:
- `phase_summary` emitted `t_<phase name>` keys while `MAIN_FLOW_RESULT_FIELDS` listed `t_soft_gp`/`t_fence_gp`; the field list now uses the real phase names (`t_read_soft`, `t_gp_soft`, `t_freeze`, `t_read_fence`, `t_gp_fence`, `t_lg`, `t_eval`) and `phase_summary` emits the full fixed set (0.0 for a phase that did not run), so `save_result` cannot fail on a `--phase fence` run.
- `install_version_invariant` needs `obj_version` **and** `refreshed_version`; `LegacyNormAdapter` exposed only the first, and the driver reached through `adapter.state`. Both were fixed: the adapter protocol now carries `refreshed_version` and the driver passes `adapter` itself.
- `_t8a_provenance` carries `run_placement`'s own `RESULT_SCHEMA_VERSION = 3` and was spread last into `result`, silently overwriting `MAIN_FLOW_RESULT_SCHEMA_VERSION`; the driver now restores it after the spread.

Three more were found by the 2026-09-19 pre-flight scan and fixed in the amendment pass:
- `norm_trace.jsonl` had two incompatible schemas under one spec §1 artefact name (the driver's row and P-H's validated `ROW_FIELDS`). The legacy row moved to `legacy_trace.jsonl` and the adapter, not the driver, owns the file.
- `freeze.region_cell_stats` and `main_flow_metrics.region_area_balance` were two implementations of the same arithmetic evaluated in *different coordinate frames*, so `result.json` carried `region_area` in scaled units and `freeze.json` the same quantity in native units. One implementation now, called in native units from both sites.
- `_resolve_prior` took a materialised netlist and the driver built it unconditionally, a multi-GB copy at 10M–30M cells even with `--remap-blocks off`; it now takes a zero-arg factory.

A second residual risk, flagged rather than hidden: the end-to-end test runs `--norm-policy legacy` only, because a second GCD flow would double its wall time. `TermNormalizerAdapter` is therefore covered by Task 5's unit tests (real gradients, real `NormTraceWriter` rows, the version-pair discipline) and by Task 9's policy branch, but the first full `grandplan` GP run happens in the spec §4 ablation campaign. If that run surprises, the first two things to check are the activation sync (`_sync_activation`, which keeps `it_activate` on the schedule's iteration rather than the first probe's) and the default target shares.

One residual risk, flagged rather than hidden: Task 9's `--phase fence` re-run asserts bit-identical IO counts across two processes-in-one-process runs. It relies on `deterministic_flag = 1` plus the `np.random.seed(params.random_seed)` guard, the same determinism the existing `tests/test_driver_io.py::test_lifetime_out_is_bit_exact_with_run_without_it` depends on. If it proves flaky on this host, weaken that one assertion to `abs(rerun["io_count"] - result["io_count"]) <= 1` and record the observed spread — do not weaken the identity assertions, which must hold exactly.
