# v2 Subproject P-G — Final Global-Routing Protocol and the Experiment Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the six v2 arms plus three ablation rows into one reproducible, resumable campaign that ends in exactly one OpenROAD `global_route` per arm and a matrix table normalised to `ours = 1.000`, whose primary metric is per-segment routed crossings against the P-D capacity.

**Architecture:** Four small modules behind one CLI. `campaign/arms.py` is a pure registry that turns an arm id into the exact argument list for the region producer, `run_main_flow`, or a fence-only staged run — no arm logic lives anywhere else. `campaign/final_grt.py` is the spec §8 recipe as one function: rebuild the PlaceDB, export a DEF from `placement.npz`, repair DEF rows through `legalize_export`, run one bounded `global_route`, decode `segments.txt`, map routed geometry through P-D's unit-edge raster, and write `route.json` + `route_segments.npz` with SHA-256 receipts. `campaign/table.py` merges each arm's `result.json` and `route.json` into a normalised Markdown/CSV matrix. `src/scripts/run_v2_campaign.py` sequences producer → placement → GRT → table per arm, resumably, holding a lock so two global routes can never overlap.

**Tech Stack:** Python 3.12 (`$DREAMPLACE_ROOT/.venv312/bin/python`), numpy, pytest, DREAMPlace 4.3.1 (`$DREAMPLACE_ROOT/install`), OpenROAD (`$OPENROAD_BIN`). No torch in `campaign/arms.py` or `campaign/table.py`. No new DREAMPlace patch, no new OpenROAD Tcl/Python script.

**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` — §0 rows "Routing", "Arms", "Benchmarks", "Claim"; all of §8; §9 ("done" for G: *full matrix tabled with hashes*, and the GCD/tile end-to-end small case); §1 (retired GR-in-loop code stays behind `IOPLACE_ENABLE_GR_IN_LOOP`); §10 risk 3 (arms (e)/(f) may lose; report area balance). Read the spec before starting; this plan argues from it and the two travel together.

**Dependency plans — read, do not duplicate.** Every interface this plan consumes from them comes from a *plan document*, not from landed code. Each task's **Interfaces** block marks such a dependency with `— from plan X Task N, verify at pre-flight`; Task 0 is the single pre-flight that checks all of them at once.

- `docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md` — owns `src/ioplace/artifacts.py`, `src/ioplace/freeze.py`, `src/ioplace/main_flow_metrics.py`, `src/ioplace/drivers/run_main_flow.py`, the `result.json` contract `MAIN_FLOW_RESULT_FIELDS`, and the arm → CLI appendix this plan's registry hard-codes.
- `docs/superpowers/plans/2026-09-19-v2-p-c-region-producer.md` — owns `src/ioplace/drivers/run_region_producer.py`, the four producer artefacts, the `--extract-bins 32` faithful mode used by arm (e), and the promoted host-local configs `benchmarks/ispd25/h100/{mempool_tile_wrap,mempool_group,mempool_cluster}.json` (its Task 11).
- `docs/superpowers/plans/2026-09-19-v2-p-d-capacity.md` — owns `src/ioplace/region_segments.py` (`enumerate_segments`, `edge_segment_ids`, `segment_utilisation`, `segments_digest`, `CAPACITY_SCALARS`), `src/ioplace/capacity/` (`capacity.npz`, `load_capacity_for_grid`), the `--capacity` flag, and **`src/scripts/run_cap_rank_correlation.route_segment_demand`** — the router-side per-segment crossing extraction this plan **imports rather than re-implements**.
- `docs/superpowers/plans/2026-09-19-v2-p-e-pseudo-ft.md` — owns the `--pseudo-*` flags on `run_main_flow`.
- `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` — owns the `straddle_*` fields this plan tables.

**What P-G owns:** `src/ioplace/campaign/`; `src/scripts/run_v2_campaign.py`; the `timeout=` kwarg and the `net_polylines` key on `route_eval/online_openroad.py`; `tests/fixtures/grt_gcd_tiny/`; the result documents `docs/results/2026-09-19-v2-matrix-<case>.md`; the campaign schedule. It owns **no** placement code and **no** objective term.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Run protocol.** From the repo root `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer`, branch `v2/redesign`:

```bash
source src/scripts/env.sh
source src/scripts/openroad_env.sh
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```

`src/scripts/env.sh` exports `DREAMPLACE_ROOT=/ldaphome/yyds-tsai-dev/DREAMPlace` and `IOPLACE_PYTHON=$DREAMPLACE_ROOT/.venv312/bin/python`; `src/scripts/openroad_env.sh` exports `OPENROAD_BIN` (at writing `/ldaphome/yyds-tsai-dev/tools/openroad/prefix-upstream-grt-uint64/bin/openroad`, present and executable) and the `LD_LIBRARY_PATH` its ortools/boost/spdlog dependencies need. Use `-m "not slow"` while iterating; run the full suite before declaring a task done. Shared H100 NVL host: run `nvidia-smi` before any GPU work and honour `CUDA_VISIBLE_DEVICES`. Measured 2026-09-19: GPUs 0–2 foreign at 100% utilisation (51–62 GB resident), GPU 3 at 0% with 22 GB resident (a P-H rerun). Check again; nothing in Tasks 0–7 needs a GPU.

**GRT runs once per arm, at the very end, never inside GP (spec §0 "Routing", §8).** No module in this plan may be imported from a placement inner loop. The retired GR-in-loop code (`src/scripts/run_route_gp.py`, `ops/routing_gp_controller.py`) stays behind `IOPLACE_ENABLE_GR_IN_LOOP=1` (spec §1, gated by P-B Task 8); this plan must not import either, and must not un-gate them. `run_v2_campaign.py` calls `final_grt.run_final_grt` exactly once per arm, after `placement.npz` exists.

**Bounded congestion iterations plus `-allow_congestion` are mandatory (spec §8).** Every `global_route` in this plan runs with `set_routing_layers -signal metal2-metal10`, `-congestion_iterations 50`, `-allow_congestion`, and 4 GRT threads. Measured justification, quoted verbatim from spec §8: *"tile routed all signals at 5 congestion iterations, metal2–metal10, zero overflow, in 221.9 s, while unbounded congestion removal cost ~9–10 h on tile/group before the protocol change"* (`docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43`). These four values are defaults in `final_grt.py` and are recorded in every `route.json`; an arm routed with different values is not comparable and the table marks it so.

**DEF row repair is mandatory (spec §8).** Every arm's placement passes through `route_eval/placement_openroad.legalize_export` before routing. It is not optional and not a fallback: it changed **648,869** cell locations and 2,120,322 orientations on `mempool_group` (`docs/results/2026-09-15-route-gp-completion-audit.md:110-121`). The evaluator is then re-run on the **repaired** coordinates, so placement metrics and routing metrics describe the same physical layout.

**Same legalisation for every arm.** All arms legalise through DREAMPlace's automatic multi-fence legaliser inside `run_main_flow` phase 4; the OpenROAD detailed-placement repair above is applied identically afterwards to every arm. No arm may use a different legaliser.

**`K=16` only.** Slicing regions and `K=32` are retired (spec §0 "Region representation", §1). `arms.py` rejects any `k` other than 16 for a real campaign case; the GCD end-to-end test is the single exception and passes `k=4` explicitly through `allow_small_k=True`.

**Every arm reports region area balance (spec §8, §10 risk 3).** `utilization_max`, `utilization_min`, `utilization_ratio`, `cell_count_deviation` from `main_flow_metrics.region_area_balance` appear in every table, for every arm, including the arms that lose. Spec §10 risk 3 is pre-registered: *"two-stage `io_count` 178,594 vs flat 100,108. Arms (e)/(f) may lose badly — say so up front. This gap is partly structural (fixed membership plus balance constraint), so it is reported, not tuned away."* The result document must carry that sentence.

**SHA-256 receipts.** Every artefact a number is read from is hashed into `route.json["sha256"]` and every table carries the hash of each row's `result.json` and `route.json`. `artifacts.file_sha256` (P-B Task 1) is the one hashing helper; `online_openroad.digest` is its equivalent inside the route_eval package. A table generated from a file whose hash does not match its recorded value is an error, not a warning.

**Python >= 3.9.** No `match`, no PEP-604 `X | Y` annotations, no PEP-585 `tuple[int, ...]` annotations in runtime code. The interpreter is 3.12; the floor is `pyproject.toml`'s `requires-python = ">=3.9"`.

**No new OpenROAD script.** `route_eval/or_scripts/dump_online_route.py` and `legalize_placement.py` are used unchanged. The only edits to existing route_eval code are the two additive ones in Task 3.

**Commit trailer.** Every commit message in this plan ends with:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

**"Done" for G (spec §9).** *"full matrix tabled with hashes"*, plus spec §9's end-to-end small case: *"GCD ... and `mempool_tile_wrap`: producer → main flow → fence LG → evaluator → one GRT"*. Task 8 automates GCD; Task 10 schedules the real matrix.

---

## Recorded interpretations and one stated deviation

Five places where §8's prose is under-determined or where following it literally would produce worse evidence. All are load-bearing; an implementer who reads only the spec will get them wrong.

**G-1: arm (f) runs through `run_main_flow --phase fence`, not through `run_placement_two_stage.py`. STATED DEVIATION from spec §8.**

Spec §8 names `run_placement_two_stage.py` as arm (f)'s vehicle, "with all terms minus the soft phase". The landed driver cannot express that arm: it builds its own grid/slicing geometry from `get_regions_for(die, k, rtype, seed)` (`run_placement_two_stage.py:246-247`), runs its own fresh Mt-KaHyPar partition rather than consuming the producer's, has no warm-start path for `seed.npz`, attaches no objective term at all, and writes a `result` dict (`run_placement_two_stage.py:257-266`) that shares almost no field with `artifacts.MAIN_FLOW_RESULT_FIELDS`. Making it arm (f) means adding `--regions`, `--seed-npz`, `--membership`, `--capacity`, `--pseudo-*`, the density-weight clamp, the straddle diagnostics and the v2 result schema to a *second* driver — i.e. a duplicate of `run_main_flow`'s phase 3.

That duplication is not merely expensive, it is *confounding*: the matrix would then compare arm (f) against ours across two different fence-GP implementations, so any difference could come from the driver rather than from the treatment. Spec §8's own intent for (f) is "fence from start, all terms, no soft-assign phase", and `run_main_flow --phase fence` is exactly that — a single GP whose fences exist from iteration 0.

So: **(f) = producer 64² geometry + producer Mt-KaHyPar membership staged as the frozen membership + `seed.npz` as the warm start + capacity on + pseudo-FT on, run as `run_main_flow --phase fence`.** `run_placement_two_stage.py` is **not modified** and remains the pre-v2 baseline driver it is today. The arm then isolates exactly one variable against ours: *who chose the membership* — the partitioner (f) or the IO/FT-driven soft phase (ours) — with geometry, terms, fence machinery, legaliser and evaluator all held fixed. That is a stronger experiment than the spec's literal reading, not a weaker one.

Consequence to report: (f)'s `io_count` is expected to be poor for the structural reason spec §10 risk 3 names (fixed membership plus the partitioner's ε=0.03 balance constraint). The table reports it; nobody tunes it away.

**G-2: "decode" means `common_grt.read_nets` plus `online_openroad.load_observation`, not `common_grt.evaluate`.**

Spec §8 writes "→ `route_eval/common_grt.py:61-172` decode". Lines 61–172 are `resource_arrays` + `evaluate`, and `resource_arrays` (`common_grt.py:61-84`) requires a **per-layer** capacity array of shape `(n_layers, ny, nx)`. The online extraction this protocol uses does not produce one: `or_scripts/dump_online_route.py:98-114` sums the preferred-direction layers into two arrays `hcap[ny][nx-1]` and `vcap[ny-1][nx]` before writing `resources.json`. Feeding `evaluate` would require a second OpenROAD pass with a different dumper.

So P-G decodes with `common_grt.read_nets` (`common_grt.py:29-58`, the *same* parser `evaluate` uses, and the one the line range was reaching for) for via count and DBU wirelength, and with `online_openroad.load_observation` (`online_openroad.py:97-159`) for the region-aware per-net geometry, native congestion and the aggregated resource overflow. Nothing in §8's metric list is lost: `common_wire_edge_overflow` was never in it, and the spec's named secondary "native total overflow" comes from `parse_native_congestion`, which `load_observation` already surfaces.

**G-3: router per-segment demand is counted on the 2-D projection of a net, not per layer.**

`load_observation` sums `union_metrics(...)['crossings']` over layers (`online_openroad.py:138-145`), so its `actual_io` charges a net twice for crossing one boundary on metal2 and again on metal4. The capacity semantics P-D pins is *"usable tracks crossing the segment; one net crossing consumes one track"*, and the evaluator's `segment_demand` is layer-free. The comparable quantity is therefore the **2-D projection**: concatenate a net's per-layer geometry, union it once with `topology.segment_union` (which merges collinear overlaps and *preserves distinct tracks*, `topology.py:163-188`), then count. That is precisely what P-D's `route_segment_demand(rg, table, polylines)` does when handed the concatenated geometry, which is why this plan calls it rather than writing its own counter.

Both numbers are reported: `segment_demand_router_total` (2-D projection, the primary) and `actual_io` (layer-summed, reconciles with `load_observation`). A large gap between them means heavy layer stacking over boundaries and is itself evidence.

**G-4: segment ids are lattice-topological, so the scaled and native grids must agree.**

`enumerate_segments` reads `RegionGrid.grid`, the 512² integer label lattice; only the `length` and `box` fields carry units. Routed geometry arrives in evaluator (scaled) units, so P-G enumerates on the **scaled** `RegionGrid` and loads capacity with `load_capacity_for_grid(capacity_npz, rg_scaled)`. Task 0 asserts `segments_digest` agrees between the native and the scaled enumeration. If it does not, `segments_digest` is covering units it should not, which is a P-D defect to fix in P-D — **escalate, do not work around it here**, because a silent mismatch would pair every arm's routed demand with the wrong capacity.

**G-5: the synthetic 3×3 27.7M case produces no routed row.**

Spec §8 puts "one synthetic 3×3 27.7M run" in the matrix section, but that case is a **Bookshelf** array streamed from `mempool_group` by `ioplace.bench.tile_bookshelf` (`.aux/.nodes/.nets/.wts/.pl/.scl`, `tile_bookshelf.py:309-391`; spec `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:73` Q2). There is no LEF and no DEF, so `legalize_export` and `global_route` cannot run on it at all. That run is therefore a **scale-feasibility row**, reported with `grt_status = "unavailable: bookshelf input has no LEF/DEF"` and every routing cell blank. It contributes runtime, peak memory, HPWL, `io_count`/`ft_count`, evaluator per-segment demand and area balance — and nothing else. `--no-grt` is the flag that says so explicitly, and `table.py` renders a blank routing block rather than a zero.

---

## File Structure

**New package `src/ioplace/campaign/`**

| File | Responsibility |
|---|---|
| `src/ioplace/campaign/__init__.py` | Empty package marker. |
| `src/ioplace/campaign/arms.py` | Pure data + argv construction. `ArmSpec`, `ARMS` (the frozen registry for `a`/`b`/`c`/`ours`/`e`/`f` and the three ablation suffixes), `producer_argv`, `main_flow_argv`, `stage_fence_only_arm`. No torch, no DREAMPlace, no OpenROAD, no subprocess. This is the only file that knows what an arm *is*. |
| `src/ioplace/campaign/final_grt.py` | The spec §8 per-arm recipe as one function. `export_arm_def`, `repair_def_rows`, `decode_segments`, `router_segment_table`, `run_final_grt`. Imports DREAMPlace (to rebuild the PlaceDB for DEF export) and OpenROAD adapters; imports P-D for the raster. |
| `src/ioplace/campaign/table.py` | `Metric`, `METRICS`, `collect_arm`, `normalise`, `render_markdown`, `render_csv`. Pure numpy/stdlib — no torch, no DREAMPlace. Must stay importable in a bare CPU report process. |
| `src/scripts/run_v2_campaign.py` | The CLI: `--case --arms --gpu --out`, per-arm state machine, resumability, the global-route lock, `--dry-run`. |

**New tests**

`tests/test_campaign_arms.py`, `tests/test_campaign_final_grt.py`, `tests/test_campaign_table.py`, `tests/test_run_v2_campaign.py`, `tests/test_online_openroad_additions.py`, and the `slow` end-to-end `tests/test_v2_campaign_gcd.py`.

**New fixture** `tests/fixtures/grt_gcd_tiny/` — a complete, hand-authored `online_openroad` output directory (`grt/{settings,receipt,resources,timings,net_audit}.json`, `grt/{segments.txt,net_manifest.jsonl,run.log}`) plus its `coord.json`, `netmap.json`, `regions.json` sidecars, written by a checked-in generator that needs **no OpenROAD**. It makes the whole decode → per-segment table → `route.json` → matrix path unit-testable offline.

**New result documents** `docs/results/2026-09-19-v2-matrix-mempool_group.md`, `-mempool_cluster.md`, `-synthetic_3x3.md`, and the campaign log `docs/results/2026-09-19-v2-campaign-schedule.md`.

**Modified**

`src/ioplace/route_eval/online_openroad.py` — two additive changes only (Task 3): a `timeout=None` kwarg on `run_openroad`, and a `net_polylines` key on `load_observation`'s return. Both are shared seams with P-D Task 10, which asks for the second one in its own words.

**Read but never modified:** `src/ioplace/route_eval/placement_openroad.py`, `common_grt.py`, `topology.py`, `joint.py`, `or_scripts/*`, `src/ioplace/export/def_export.py`, `src/ioplace/drivers/run_placement.py`, `run_placement_two_stage.py`, `run_main_flow.py`, `run_region_producer.py`, `src/ioplace/artifacts.py`, `region_segments.py`, `capacity/`, `main_flow_metrics.py`.

**Boundary rule:** `arms.py` and `table.py` take and return plain Python data and are testable in milliseconds. `final_grt.py` is the only file that touches a PlaceDB or a subprocess. `run_v2_campaign.py` is the only file that touches process scheduling, locks and the filesystem layout.

**Run directory layout** (one per case, created by `run_v2_campaign.py`):

```
<out>/                                # e.g. results/v2_matrix_20260919/mempool_group
  campaign.json                       # case, config, commit, env, arm list, timestamps
  capacity.npz -> ...                 # symlink to the P-D extraction for this case
  producer_b64/                       # regions.json seed.npz membership.npz producer.json
  producer_b32/                       # the arm (e) faithful 32^2 producer
  arms/<arm_id>/
    state.json                        # resumability: completed steps + input hashes
    result.json evaluation.npz placement.npz freeze.json ...   # run_main_flow output
    export/                           # out.def regions.json netmap.json coord.json
    repair/                           # legalize_export output (legalized.def, receipt.json)
    grt/                              # run_openroad output (segments.txt, resources.json, receipt.json)
    route.json route_segments.npz     # this plan's output
  matrix.md matrix.csv                # table.py output
  .grt.lock                           # held only while a global_route is running
```

---

### Task 0: Pre-flight — verify every cross-plan interface before writing code

Nothing in this plan can be written against imagination. P-B, P-C, P-D, P-E and P-F all landed (or did not) after this plan was written, so the first task is a single script that checks every name this plan imports and every flag it passes. It writes a report; it changes no product code.

**Files:**
- Create: `src/scripts/check_v2_campaign_preflight.py`
- Test: `tests/test_run_v2_campaign.py` (first test only; the rest arrive in Task 7)

**Interfaces:**
- Consumes: nothing. This task is the one that finds out what exists.
- Produces: `check(verbose=False) -> dict` mapping a requirement id to `True`/a string explaining the failure; `main(argv=None)` printing the report and returning `0` only when every requirement holds.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_v2_campaign.py`:

```python
import importlib
import json

import pytest


def test_preflight_reports_every_required_interface():
    """The pre-flight must enumerate every cross-plan name this subproject
    imports, so a missing dependency is a named failure and not an
    ImportError three tasks later."""
    module = importlib.import_module("src.scripts.check_v2_campaign_preflight")
    report = module.check()
    required = {
        "artifacts.save_positions", "artifacts.save_membership",
        "artifacts.save_freeze", "artifacts.file_sha256",
        "artifacts.FREEZE_FIELDS", "artifacts.MAIN_FLOW_RESULT_FIELDS",
        "artifacts.scaled_region_set",
        "run_main_flow.build_parser", "run_main_flow.--phase",
        "run_main_flow.--regions", "run_main_flow.--seed-npz",
        "run_main_flow.--membership", "run_main_flow.--remap-blocks",
        "run_main_flow.--init", "run_main_flow.--norm-policy",
        "run_main_flow.--capacity", "run_main_flow.--pseudo-ft",
        "run_region_producer.main", "run_region_producer.--extract-bins",
        "region_segments.enumerate_segments", "region_segments.edge_segment_ids",
        "region_segments.segment_utilisation", "region_segments.segments_digest",
        "region_segments.CAPACITY_SCALARS",
        "capacity.load_capacity_for_grid",
        "run_cap_rank_correlation.route_segment_demand",
        "main_flow_metrics.region_area_balance",
        "straddle.STRADDLE_SCALARS",
        "openroad_bin", "segments_digest_scale_invariant",
    }
    assert set(report) == required


def test_preflight_main_returns_nonzero_when_something_is_missing(monkeypatch):
    module = importlib.import_module("src.scripts.check_v2_campaign_preflight")
    monkeypatch.setattr(module, "check", lambda verbose=False: {"x": "missing"})
    assert module.main([]) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_run_v2_campaign.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.scripts.check_v2_campaign_preflight'`.

- [ ] **Step 3: Write `src/scripts/check_v2_campaign_preflight.py`**

```python
"""Does every interface subproject P-G depends on actually exist?

P-G (v2 design sec 8) is written against five sibling *plans*, not against
landed code. This script turns "the plan said so" into a checked fact, once,
before any campaign module is written. Every id below is a name or a CLI flag
that src/ioplace/campaign/* or src/scripts/run_v2_campaign.py imports or
passes.

  source src/scripts/env.sh && source src/scripts/openroad_env.sh
  "$IOPLACE_PYTHON" src/scripts/check_v2_campaign_preflight.py -v
"""
import argparse
import importlib
import json
import os


def _has_attr(module_name, attribute):
    try:
        module = importlib.import_module(module_name)
    except Exception as error:                     # noqa: BLE001 - report it
        return "import %s failed: %s" % (module_name, error)
    return True if hasattr(module, attribute) else "%s.%s missing" % (
        module_name, attribute)


def _has_flag(module_name, builder, flag):
    try:
        module = importlib.import_module(module_name)
        parser = getattr(module, builder)()
    except Exception as error:                     # noqa: BLE001 - report it
        return "%s.%s() failed: %s" % (module_name, builder, error)
    options = set()
    for action in parser._actions:                 # argparse public enough here
        options.update(action.option_strings)
    return True if flag in options else "%s has no %s" % (module_name, flag)


def _segments_digest_is_scale_invariant():
    """G-4. Segment ids come from the 512-lattice label grid, so the native
    and the scaled RegionGrid must enumerate to the same digest; only the
    length/box fields carry units. If this fails, segments_digest is hashing
    coordinates and P-D must fix it -- P-G must not work around it."""
    try:
        from ioplace.artifacts import scaled_region_set
        from ioplace.region_grid import RegionGrid
        from ioplace.region_segments import enumerate_segments, segments_digest
        from ioplace.regions import make_grid_regions
    except Exception as error:                     # noqa: BLE001 - report it
        return "import failed: %s" % (error,)
    native = make_grid_regions((100.0, 200.0, 1124.0, 1224.0), 4, 4, lattice=512)
    scaled = scaled_region_set(native, (100.0, 200.0), 0.5)
    a = segments_digest(enumerate_segments(RegionGrid(native)))
    b = segments_digest(enumerate_segments(RegionGrid(scaled)))
    return True if a == b else (
        "segments_digest differs between the native (%s) and the scaled (%s) "
        "grid; escalate to P-D, do not work around" % (a[:12], b[:12]))


def _openroad_binary():
    binary = os.environ.get("OPENROAD_BIN")
    if not binary:
        return "OPENROAD_BIN unset; source src/scripts/openroad_env.sh"
    if not os.access(binary, os.X_OK):
        return "%s is not executable" % (binary,)
    return True


def check(verbose=False):
    report = {}
    for attribute in ("save_positions", "save_membership", "save_freeze",
                      "file_sha256", "FREEZE_FIELDS",
                      "MAIN_FLOW_RESULT_FIELDS", "scaled_region_set"):
        report["artifacts." + attribute] = _has_attr("ioplace.artifacts",
                                                     attribute)
    flow = "ioplace.drivers.run_main_flow"
    report["run_main_flow.build_parser"] = _has_attr(flow, "build_parser")
    for flag in ("--phase", "--regions", "--seed-npz", "--membership",
                 "--remap-blocks", "--init", "--norm-policy", "--capacity",
                 "--pseudo-ft"):
        report["run_main_flow." + flag] = _has_flag(flow, "build_parser", flag)
    producer = "ioplace.drivers.run_region_producer"
    report["run_region_producer.main"] = _has_attr(producer, "main")
    report["run_region_producer.--extract-bins"] = _has_flag(
        producer, "build_parser", "--extract-bins")
    for attribute in ("enumerate_segments", "edge_segment_ids",
                      "segment_utilisation", "segments_digest",
                      "CAPACITY_SCALARS"):
        report["region_segments." + attribute] = _has_attr(
            "ioplace.region_segments", attribute)
    report["capacity.load_capacity_for_grid"] = _has_attr(
        "ioplace.drivers.run_placement_io", "load_capacity_for_grid")
    report["run_cap_rank_correlation.route_segment_demand"] = _has_attr(
        "src.scripts.run_cap_rank_correlation", "route_segment_demand")
    report["main_flow_metrics.region_area_balance"] = _has_attr(
        "ioplace.main_flow_metrics", "region_area_balance")
    report["straddle.STRADDLE_SCALARS"] = _has_attr("ioplace.straddle",
                                                    "STRADDLE_SCALARS")
    report["openroad_bin"] = _openroad_binary()
    report["segments_digest_scale_invariant"] = _segments_digest_is_scale_invariant()
    if verbose:
        print(json.dumps(report, indent=1, sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    report = check(verbose=True)
    failures = {k: v for k, v in report.items() if v is not True}
    if failures:
        print("PREFLIGHT FAILED: %d of %d" % (len(failures), len(report)))
        return 1
    print("PREFLIGHT OK: %d requirements" % (len(report),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Two notes the implementer must act on rather than paper over:

1. `load_capacity_for_grid` is defined by P-D Task 8 inside `src/ioplace/drivers/run_placement_io.py`. If P-D moved it to `ioplace.capacity.extract`, change the `_has_attr` call and the Task 4 import in the same commit — do not duplicate the function.
2. `run_cap_rank_correlation` lives at `src/scripts/run_cap_rank_correlation.py` and is imported as `src.scripts.run_cap_rank_correlation`, the same import style P-D's own `tests/test_cap_rank_correlation.py` uses. If that import fails because `src/scripts/` has no `__init__.py`, add an empty one in this commit and say so in the message.

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_run_v2_campaign.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the pre-flight itself and record the result**

```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
"$IOPLACE_PYTHON" src/scripts/check_v2_campaign_preflight.py -v \
  | tee /tmp/p_g_preflight.json
```
Expected: `PREFLIGHT OK: 30 requirements`.

If any requirement fails, **stop and report which one**. A failure in a `run_main_flow.--*` flag means P-B/P-D/P-E have not landed that flag; Tasks 1–7 can still be written (they are pure), but Tasks 8 and 10 are blocked. A failure in `segments_digest_scale_invariant` is a P-D defect (G-4) and blocks Task 4.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/check_v2_campaign_preflight.py tests/test_run_v2_campaign.py
git commit -m "$(cat <<'MSG'
test(campaign): pre-flight every cross-plan interface P-G depends on

P-G is written against the P-B/P-C/P-D/P-E/P-F plans, not against landed
code. One script turns each of those assumptions into a named, checkable
requirement, including v2 design G-4's scale-invariance of segments_digest,
so a missing dependency fails here instead of three tasks later.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 1: The arm registry — `campaign/arms.py`

One frozen table turns an arm id into the exact argument list for the region producer and for `run_main_flow`, plus the staging function that lets arms (e) and (f) run `--phase fence` with no soft phase. Nothing else in the repo may decide what an arm is.

**Files:**
- Create: `src/ioplace/campaign/__init__.py`
- Create: `src/ioplace/campaign/arms.py`
- Test: `tests/test_campaign_arms.py`

**Interfaces:**
- Consumes: `ioplace.artifacts.{load_positions, save_positions, load_membership, save_membership, save_freeze, FREEZE_FIELDS, FREEZE_SCHEMA_VERSION}` — from plan P-B Task 1, verify at pre-flight (Task 0). `ioplace.regions.RegionSet` (landed, `src/ioplace/regions.py:44`).
- Produces:
  - `ARM_IDS = ("a", "b", "c", "ours", "e", "f")`
  - `ABLATION_IDS = ("ours__polB", "ours__nocap", "ours__nopseudo")`
  - `ALL_ARM_IDS = ARM_IDS + ABLATION_IDS`
  - `@dataclass(frozen=True) ArmSpec(arm_id, label, geometry, init, driver, norm_policy, capacity, pseudo_ft, remap_blocks)` where `geometry in ("grid", "producer_b64", "producer_b32")`, `init in ("region_center", "seed")`, `driver in ("main_flow", "fence_only")`
  - `ARMS: dict` mapping every id in `ALL_ARM_IDS` to its `ArmSpec`
  - `producer_argv(config, out_dir, *, extract_bins, k=16, seed=0, epsilon=0.03) -> list`
  - `main_flow_argv(spec, *, config, out_dir, producer_dirs, capacity_npz=None, k=16, seed=0, allow_small_k=False) -> list`
  - `stage_fence_only_arm(out_dir, producer_dir, *, k) -> dict`

**The six arms and the three ablation rows** (spec §8's table, plus §4's policy ablation and §10 (v)'s on/off ablation):

| id | geometry | init | driver | norm policy | capacity | pseudo-FT |
|---|---|---|---|---|---|---|
| `a` | grid 4×4 | region centres | main_flow | grandplan | on | on |
| `b` | producer 64² | region centres | main_flow | grandplan | on | on |
| `c` | grid 4×4 | flat seed | main_flow | grandplan | on | on |
| `ours` | producer 64² | flat seed | main_flow | grandplan | on | on |
| `e` | producer 32² | flat seed | fence_only | grandplan | **off** | **off** |
| `f` | producer 64² | flat seed | fence_only | grandplan | on | on |
| `ours__polB` | producer 64² | flat seed | main_flow | **adaptive** | on | on |
| `ours__nocap` | producer 64² | flat seed | main_flow | grandplan | **off** | on |
| `ours__nopseudo` | producer 64² | flat seed | main_flow | grandplan | on | **off** |

Policy A `grandplan` is the matrix default because it is GrandPlan Eq.3 and spec §4's primary form; `ours__polB` is the policy-B row spec §8 asks for. `legacy` is P-B's default and is deliberately **not** in the matrix: it is the retired path, kept only so P-B's own tests keep passing.

**Why `remap_blocks` differs between (a) and (b).** P-B's appendix routes a `source == "mtkahypar"` membership through `assign_blocks_to_regions` under `--remap-blocks auto`, because partitioner block ids carry no geometry. That is right for (a), whose geometry is an arbitrary 4×4 grid. It is **wrong** for (b): the producer extracts region `k` from partition `k`'s own density map (P-C Task 5), so block id already *is* region id and remapping would scramble the prior it was built from. So `a` uses `auto` and `b` uses `off`. `c`, `ours` and the ablations pass no membership at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_campaign_arms.py`:

```python
import json
import os

import numpy as np
import pytest

from ioplace.campaign.arms import (ABLATION_IDS, ALL_ARM_IDS, ARM_IDS, ARMS,
                                   main_flow_argv, producer_argv,
                                   stage_fence_only_arm)


def _producer_dirs(root):
    return {"producer_b64": os.path.join(root, "producer_b64"),
            "producer_b32": os.path.join(root, "producer_b32")}


def test_the_registry_is_exactly_the_spec_8_arms_plus_three_ablations():
    assert ARM_IDS == ("a", "b", "c", "ours", "e", "f")
    assert ABLATION_IDS == ("ours__polB", "ours__nocap", "ours__nopseudo")
    assert ALL_ARM_IDS == ARM_IDS + ABLATION_IDS
    assert set(ARMS) == set(ALL_ARM_IDS)
    for arm_id, spec in ARMS.items():
        assert spec.arm_id == arm_id
        assert spec.geometry in ("grid", "producer_b64", "producer_b32")
        assert spec.init in ("region_center", "seed")
        assert spec.driver in ("main_flow", "fence_only")
        assert spec.norm_policy in ("grandplan", "adaptive")


def test_the_2x2_varies_only_geometry_and_init():
    """spec sec 8: (a)-(c) plus ours is the Table-3 2x2 -- grid vs producer
    crossed with region-centre vs flat-seed init, everything else equal."""
    cells = {(ARMS[i].geometry, ARMS[i].init) for i in ("a", "b", "c", "ours")}
    assert cells == {("grid", "region_center"), ("producer_b64", "region_center"),
                     ("grid", "seed"), ("producer_b64", "seed")}
    for arm_id in ("a", "b", "c", "ours"):
        spec = ARMS[arm_id]
        assert spec.driver == "main_flow"
        assert spec.capacity is True and spec.pseudo_ft is True
        assert spec.norm_policy == "grandplan"


def test_arm_e_is_faithful_grandplan_and_carries_no_v2_term():
    """spec sec 8: (e) is WL/density/grouping only, on the faithful 32^2
    extraction. Grouping lives in the producer's flat GP, so the fence phase
    must attach nothing."""
    spec = ARMS["e"]
    assert spec.geometry == "producer_b32"
    assert spec.driver == "fence_only"
    assert spec.capacity is False and spec.pseudo_ft is False


def test_arm_f_is_fence_from_start_with_every_term(tmp_path):
    """Recorded deviation G-1: (f) runs run_main_flow --phase fence, not
    run_placement_two_stage.py, so it differs from ours in exactly one
    variable -- who chose the membership."""
    spec = ARMS["f"]
    assert spec.driver == "fence_only"
    assert spec.geometry == ARMS["ours"].geometry
    assert spec.capacity is ARMS["ours"].capacity
    assert spec.pseudo_ft is ARMS["ours"].pseudo_ft


def test_ablations_differ_from_ours_in_exactly_one_field():
    ours = ARMS["ours"]
    assert ARMS["ours__polB"].norm_policy == "adaptive"
    assert ARMS["ours__polB"].capacity is ours.capacity
    assert ARMS["ours__polB"].pseudo_ft is ours.pseudo_ft
    assert ARMS["ours__nocap"].capacity is False
    assert ARMS["ours__nocap"].pseudo_ft is ours.pseudo_ft
    assert ARMS["ours__nocap"].norm_policy == ours.norm_policy
    assert ARMS["ours__nopseudo"].pseudo_ft is False
    assert ARMS["ours__nopseudo"].capacity is ours.capacity


def test_producer_argv_is_the_p_c_cli(tmp_path):
    argv = producer_argv("case.json", str(tmp_path / "p64"), extract_bins=64)
    assert argv[:2] == ["--config", "case.json"]
    assert "--extract-bins" in argv and argv[argv.index("--extract-bins") + 1] == "64"
    assert argv[argv.index("--k") + 1] == "16"
    assert argv[argv.index("--rect-max") + 1] == "8"


def test_main_flow_argv_for_arm_a_uses_grid_region_centres_and_remaps(tmp_path):
    dirs = _producer_dirs(str(tmp_path))
    argv = main_flow_argv(ARMS["a"], config="case.json",
                          out_dir=str(tmp_path / "arms" / "a"),
                          producer_dirs=dirs, capacity_npz="cap.npz")
    assert "--regions" not in argv
    assert argv[argv.index("--rtype") + 1] == "grid"
    assert argv[argv.index("--init") + 1] == "region_center"
    assert argv[argv.index("--membership") + 1] == os.path.join(
        dirs["producer_b64"], "membership.npz")
    assert argv[argv.index("--remap-blocks") + 1] == "auto"
    assert argv[argv.index("--capacity") + 1] == "cap.npz"
    assert argv[argv.index("--pseudo-ft") + 1] == "on"
    assert argv[argv.index("--norm-policy") + 1] == "grandplan"
    assert argv[argv.index("--k") + 1] == "16"
    assert "--phase" not in argv


def test_main_flow_argv_for_arm_b_does_not_remap_producer_block_ids(tmp_path):
    """The producer extracts region k from partition k's density map, so block
    id already is region id; assign_blocks_to_regions would scramble it."""
    dirs = _producer_dirs(str(tmp_path))
    argv = main_flow_argv(ARMS["b"], config="case.json", out_dir="o",
                          producer_dirs=dirs, capacity_npz="cap.npz")
    assert argv[argv.index("--regions") + 1] == os.path.join(
        dirs["producer_b64"], "regions.json")
    assert argv[argv.index("--remap-blocks") + 1] == "off"


def test_main_flow_argv_for_ours_warm_starts_from_the_flat_seed(tmp_path):
    dirs = _producer_dirs(str(tmp_path))
    argv = main_flow_argv(ARMS["ours"], config="case.json", out_dir="o",
                          producer_dirs=dirs, capacity_npz="cap.npz")
    assert argv[argv.index("--init") + 1] == "seed"
    assert argv[argv.index("--seed-npz") + 1] == os.path.join(
        dirs["producer_b64"], "seed.npz")
    assert "--membership" not in argv


def test_main_flow_argv_for_fence_only_arms_sets_phase_fence(tmp_path):
    dirs = _producer_dirs(str(tmp_path))
    argv_e = main_flow_argv(ARMS["e"], config="case.json", out_dir="o",
                            producer_dirs=dirs, capacity_npz="cap.npz")
    assert argv_e[argv_e.index("--phase") + 1] == "fence"
    assert argv_e[argv_e.index("--regions") + 1] == os.path.join(
        dirs["producer_b32"], "regions.json")
    assert "--capacity" not in argv_e
    assert argv_e[argv_e.index("--pseudo-ft") + 1] == "off"
    argv_f = main_flow_argv(ARMS["f"], config="case.json", out_dir="o",
                            producer_dirs=dirs, capacity_npz="cap.npz")
    assert argv_f[argv_f.index("--phase") + 1] == "fence"
    assert argv_f[argv_f.index("--capacity") + 1] == "cap.npz"
    assert argv_f[argv_f.index("--pseudo-ft") + 1] == "on"


def test_capacity_arms_refuse_to_run_without_a_capacity_file(tmp_path):
    dirs = _producer_dirs(str(tmp_path))
    with pytest.raises(ValueError, match="capacity.npz required"):
        main_flow_argv(ARMS["ours"], config="c.json", out_dir="o",
                       producer_dirs=dirs, capacity_npz=None)
    # arm (e) carries no capacity term, so it is happy without one.
    main_flow_argv(ARMS["e"], config="c.json", out_dir="o",
                   producer_dirs=dirs, capacity_npz=None)


def test_k_other_than_16_is_rejected_unless_explicitly_allowed(tmp_path):
    """spec sec 0: K=16 only; slicing and K=32 are retired. The GCD
    end-to-end test is the single exception."""
    dirs = _producer_dirs(str(tmp_path))
    with pytest.raises(ValueError, match="K=16"):
        main_flow_argv(ARMS["ours"], config="c.json", out_dir="o",
                       producer_dirs=dirs, capacity_npz="cap.npz", k=32)
    argv = main_flow_argv(ARMS["ours"], config="c.json", out_dir="o",
                          producer_dirs=dirs, capacity_npz="cap.npz", k=4,
                          allow_small_k=True)
    assert argv[argv.index("--k") + 1] == "4"


def test_stage_fence_only_arm_writes_the_three_artefacts_phase_fence_reads(tmp_path):
    from ioplace.artifacts import (load_freeze, load_membership,
                                   load_positions, save_membership,
                                   save_positions)
    producer = tmp_path / "producer_b64"
    producer.mkdir()
    node_x = np.arange(6, dtype=np.float64)
    node_y = np.arange(6, dtype=np.float64) * 2.0
    save_positions(str(producer / "seed.npz"), node_x, node_y,
                   die=(0.0, 0.0, 100.0, 100.0), shift_factor=(0.0, 0.0),
                   scale_factor=1.0, placedb_sha256="deadbeef", kind="seed")
    save_membership(str(producer / "membership.npz"),
                    np.array([0, 1, 2, 3], dtype=np.int32),
                    source="mtkahypar", k=4, seed=0, epsilon=0.03)
    out = tmp_path / "arms" / "f"
    out.mkdir(parents=True)
    record = stage_fence_only_arm(str(out), str(producer), k=4)
    soft = load_positions(str(out / "soft.npz"))
    assert soft.kind == "soft"
    assert soft.node_x.tolist() == node_x.tolist()
    frozen = load_membership(str(out / "frozen_membership.npz"))
    assert frozen.source == "freeze" and frozen.k == 4
    assert frozen.part.tolist() == [0, 1, 2, 3]
    freeze = load_freeze(str(out / "freeze.json"))
    assert freeze["reason"] == "fence_from_start" and freeze["iteration"] == 0
    assert record["staged"] == ["soft.npz", "frozen_membership.npz",
                                "freeze.json"]


def test_stage_fence_only_arm_rejects_a_membership_that_leaves_a_region_empty(tmp_path):
    """An empty real region crashes placedb.initialize() with IndexError at
    PlaceDB.py:687 (P-B amendment E-3). Catch it before the GPU is touched."""
    from ioplace.artifacts import save_membership, save_positions
    producer = tmp_path / "producer_b64"
    producer.mkdir()
    save_positions(str(producer / "seed.npz"), np.zeros(4), np.zeros(4),
                   die=(0.0, 0.0, 10.0, 10.0), shift_factor=(0.0, 0.0),
                   scale_factor=1.0, placedb_sha256="d", kind="seed")
    save_membership(str(producer / "membership.npz"),
                    np.array([0, 0, 1, 1], dtype=np.int32),
                    source="mtkahypar", k=4, seed=0, epsilon=0.03)
    out = tmp_path / "arms" / "e"
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="empty region"):
        stage_fence_only_arm(str(out), str(producer), k=4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_arms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.campaign'`.

- [ ] **Step 3: Create the package marker**

```bash
mkdir -p src/ioplace/campaign
: > src/ioplace/campaign/__init__.py
```

- [ ] **Step 4: Write `src/ioplace/campaign/arms.py`**

```python
"""What an arm *is*, in one table.

v2 design sec 8 fixes six arms plus three ablation rows. Every other module in
this subproject takes an ArmSpec and never re-derives one, so a change to the
matrix is a change to this file alone.

Recorded deviation G-1 (see the P-G plan): arm (f) runs
`run_main_flow --phase fence`, not `run_placement_two_stage.py`. The landed
two-stage driver builds its own grid geometry and its own partition, attaches
no objective term, and writes a different result schema; adapting it would
duplicate run_main_flow's phase 3 and would confound the comparison across two
fence-GP implementations. Running (f) through the same driver isolates exactly
one variable against `ours`: who chose the membership.
"""
import os

from dataclasses import dataclass

GEOMETRIES = ("grid", "producer_b64", "producer_b32")
INITS = ("region_center", "seed")
DRIVERS = ("main_flow", "fence_only")
NORM_POLICIES = ("grandplan", "adaptive")

#: Region count. spec sec 0: "K=16 only; slicing and K=32 retired."
DEFAULT_K = 16
#: Producer knobs the campaign never varies (spec sec 2, P-C Task 10 defaults).
DEFAULT_RECT_MAX = 8
DEFAULT_EPSILON = 0.03


@dataclass(frozen=True)
class ArmSpec(object):
    arm_id: str
    label: str
    geometry: str
    init: str
    driver: str
    norm_policy: str
    capacity: bool
    pseudo_ft: bool
    remap_blocks: str


def _spec(arm_id, label, geometry, init, driver, norm_policy, capacity,
          pseudo_ft, remap_blocks):
    if geometry not in GEOMETRIES or init not in INITS or driver not in DRIVERS:
        raise ValueError("invalid arm definition %r" % (arm_id,))
    if norm_policy not in NORM_POLICIES or remap_blocks not in ("auto", "off"):
        raise ValueError("invalid arm definition %r" % (arm_id,))
    return ArmSpec(arm_id, label, geometry, init, driver, norm_policy,
                   bool(capacity), bool(pseudo_ft), remap_blocks)


ARM_IDS = ("a", "b", "c", "ours", "e", "f")
ABLATION_IDS = ("ours__polB", "ours__nocap", "ours__nopseudo")
ALL_ARM_IDS = ARM_IDS + ABLATION_IDS

ARMS = {spec.arm_id: spec for spec in (
    _spec("a", "(a) grid 4x4 + region centres", "grid", "region_center",
          "main_flow", "grandplan", True, True, "auto"),
    _spec("b", "(b) producer + region centres", "producer_b64",
          "region_center", "main_flow", "grandplan", True, True, "off"),
    _spec("c", "(c) grid 4x4 + flat seed", "grid", "seed", "main_flow",
          "grandplan", True, True, "off"),
    _spec("ours", "ours: producer + flat seed", "producer_b64", "seed",
          "main_flow", "grandplan", True, True, "off"),
    _spec("e", "(e) faithful GrandPlan", "producer_b32", "seed",
          "fence_only", "grandplan", False, False, "off"),
    _spec("f", "(f) fence from start", "producer_b64", "seed", "fence_only",
          "grandplan", True, True, "off"),
    _spec("ours__polB", "ours, policy B (adaptive)", "producer_b64", "seed",
          "main_flow", "adaptive", True, True, "off"),
    _spec("ours__nocap", "ours, capacity off", "producer_b64", "seed",
          "main_flow", "grandplan", False, True, "off"),
    _spec("ours__nopseudo", "ours, pseudo-FT off", "producer_b64", "seed",
          "main_flow", "grandplan", True, False, "off"),
)}

#: Which producer directory each geometry reads. "grid" reads the 64-bin
#: producer too, for its Mt-KaHyPar prior and its flat seed -- the geometry
#: comes from --rtype grid, the prior and the seed still come from a producer
#: run, exactly as spec sec 8's 2x2 requires.
GEOMETRY_PRODUCER = {"grid": "producer_b64", "producer_b64": "producer_b64",
                     "producer_b32": "producer_b32"}

#: Bin count per producer directory (spec sec 2: 64 default, 32 for arm (e)).
PRODUCER_BINS = {"producer_b64": 64, "producer_b32": 32}


def producer_argv(config, out_dir, *, extract_bins, k=DEFAULT_K, seed=0,
                  epsilon=DEFAULT_EPSILON):
    """Argument list for `python -m ioplace.drivers.run_region_producer`."""
    if extract_bins not in (32, 64):
        raise ValueError("extract_bins must be 32 or 64, got %r" % (extract_bins,))
    return ["--config", str(config), "--out-dir", str(out_dir),
            "--k", str(int(k)), "--extract-bins", str(int(extract_bins)),
            "--rect-max", str(DEFAULT_RECT_MAX), "--seed", str(int(seed)),
            "--epsilon", repr(float(epsilon))]


def main_flow_argv(spec, *, config, out_dir, producer_dirs, capacity_npz=None,
                   k=DEFAULT_K, seed=0, allow_small_k=False):
    """Argument list for `python -m ioplace.drivers.run_main_flow`.

    `producer_dirs` maps "producer_b64"/"producer_b32" to a directory holding
    that producer run's regions.json / seed.npz / membership.npz.
    """
    k = int(k)
    if k != DEFAULT_K and not allow_small_k:
        raise ValueError("spec sec 0 fixes K=16 only; got k=%d (pass "
                         "allow_small_k=True for the GCD end-to-end test)" % (k,))
    if spec.capacity and not capacity_npz:
        raise ValueError("capacity.npz required for arm %r" % (spec.arm_id,))
    producer = producer_dirs[GEOMETRY_PRODUCER[spec.geometry]]
    argv = ["--config", str(config), "--out-dir", str(out_dir),
            "--k", str(k), "--seed", str(int(seed)),
            "--norm-policy", spec.norm_policy]
    if spec.geometry == "grid":
        argv += ["--rtype", "grid"]
    else:
        argv += ["--regions", os.path.join(producer, "regions.json")]
    if spec.driver == "fence_only":
        # G-1: no soft phase at all. stage_fence_only_arm() has already
        # written soft.npz / frozen_membership.npz / freeze.json into out_dir,
        # which is exactly what --phase fence reads (P-B Task 7 behaviour 2).
        argv += ["--phase", "fence"]
    else:
        argv += ["--init", spec.init]
        if spec.init == "seed":
            argv += ["--seed-npz", os.path.join(producer, "seed.npz")]
        else:
            argv += ["--membership", os.path.join(producer, "membership.npz"),
                     "--remap-blocks", spec.remap_blocks]
    if spec.capacity:
        argv += ["--capacity", str(capacity_npz)]
    argv += ["--pseudo-ft", "on" if spec.pseudo_ft else "off"]
    return argv


def stage_fence_only_arm(out_dir, producer_dir, *, k):
    """Write the three artefacts `run_main_flow --phase fence` reads.

    Arms (e) and (f) have no soft phase, so nothing produced soft.npz,
    frozen_membership.npz or freeze.json. Their fence membership is the
    producer's Mt-KaHyPar labelling and their warm start is the producer's
    flat seed. Region id == partition id by construction: P-C's extraction
    builds region k from partition k's own density map, so no block->region
    permutation is applied here (and must not be).
    """
    from ioplace.artifacts import (FREEZE_FIELDS, FREEZE_SCHEMA_VERSION,
                                   load_membership, load_positions,
                                   save_freeze, save_membership,
                                   save_positions)
    import numpy as np

    k = int(k)
    seed = load_positions(os.path.join(producer_dir, "seed.npz"))
    membership = load_membership(os.path.join(producer_dir, "membership.npz"),
                                 expect_k=k)
    part = np.asarray(membership.part, dtype=np.int32)
    counts = np.bincount(part, minlength=k)
    if counts.size != k or int(counts.min()) == 0:
        raise ValueError(
            "membership leaves an empty region (counts=%s); placedb."
            "initialize() raises IndexError at PlaceDB.py:687 on an empty "
            "real region" % (counts.tolist(),))
    soft_path = os.path.join(out_dir, "soft.npz")
    save_positions(soft_path, seed.node_x, seed.node_y, die=seed.die,
                   shift_factor=seed.shift_factor,
                   scale_factor=seed.scale_factor,
                   placedb_sha256=seed.placedb_sha256, kind="soft")
    membership_path = os.path.join(out_dir, "frozen_membership.npz")
    save_membership(membership_path, part, source="freeze", k=k,
                    seed=int(membership.seed), epsilon=float(membership.epsilon))
    # Fields the freeze never measured stay None rather than 0.0: a
    # fence-from-start arm has no overflow, tau or churn at the freeze,
    # and reporting zeros would put fiction in the table. --phase fence
    # reads only `part` and the positions; the region statistics are
    # recomputed after the fence GP by run_fence_gp.
    record = {name: None for name in FREEZE_FIELDS}
    record.update({
        "schema_version": FREEZE_SCHEMA_VERSION, "iteration": 0,
        "reason": "fence_from_start", "k": k,
        "membership_npz": os.path.abspath(membership_path),
        "soft_npz": os.path.abspath(soft_path),
        "repaired_empty_regions": [], "gp_iterations_soft": 0,
    })
    freeze_path = os.path.join(out_dir, "freeze.json")
    save_freeze(freeze_path, record)
    return {"staged": ["soft.npz", "frozen_membership.npz", "freeze.json"],
            "k": k, "num_movable": int(part.size),
            "region_cell_count": counts.tolist(),
            "producer_dir": os.path.abspath(producer_dir)}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_arms.py -v`
Expected: PASS (13 tests).

If `test_stage_fence_only_arm_writes_the_three_artefacts_phase_fence_reads` fails inside `save_freeze` because `_require_fields` rejects `None`, that means P-B's freeze writer type-checks its fields. In that case replace the `None` defaults with the sentinel P-B's `load_freeze` accepts, record which one in the commit message, and add an assertion in Task 8 that the staged `freeze.json` round-trips.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/campaign/__init__.py src/ioplace/campaign/arms.py \
        tests/test_campaign_arms.py
git commit -m "$(cat <<'MSG'
feat(campaign): the v2 arm registry and fence-only staging

One frozen table for v2 design sec 8's six arms plus sec 4's policy-B and
sec 10 (v)'s capacity / pseudo-FT on-off ablation rows, turning an arm id
into the exact producer and run_main_flow argument lists.

Recorded deviation G-1: arm (f) runs run_main_flow --phase fence rather than
run_placement_two_stage.py, so geometry, terms, fence machinery, legaliser
and evaluator are held fixed and (f) differs from ours in exactly one
variable -- who chose the membership. stage_fence_only_arm writes the
soft.npz / frozen_membership.npz / freeze.json triple that --phase fence
reads, refusing any membership that would leave a region empty.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: The recorded tiny GRT fixture

A complete `online_openroad` output directory, written by a generator that needs no OpenROAD, so every later task can test decode → per-segment table → `route.json` → matrix offline. Its geometry is GCD's four-region 2×2 grid shrunk to round numbers, which keeps every expected value hand-checkable.

**Files:**
- Create: `tests/fixtures/grt_gcd_tiny/make_fixture.py`
- Create (generated, checked in): `tests/fixtures/grt_gcd_tiny/{regions.json,coord.json,netmap.json}` and `tests/fixtures/grt_gcd_tiny/grt/{settings.json,segments.txt,resources.json,timings.json,net_audit.json,net_manifest.jsonl,run.log,receipt.json}`
- Test: `tests/test_campaign_final_grt.py` (fixture tests only; the recipe arrives in Tasks 4–5)

**Interfaces:**
- Consumes: `ioplace.regions.make_grid_regions` (landed, `src/ioplace/regions.py`); `ioplace.route_eval.online_openroad.digest` (landed, `online_openroad.py:16`).
- Produces:
  - `FIXTURE_DIR` — the directory path, as a module constant in the test
  - `build(out_dir) -> dict` in `make_fixture.py`, writing every file and returning the expected decode results

**Why hand-authored and not recorded from a real run.** A recorded GCD directory would be ~30 MB, would pin one OpenROAD build's exact output, and could not be regenerated on a host without the router. This fixture is byte-stable, 6 kB, regenerable by anyone, and exercises every branch the decoder has: a multi-layer net, a via, a doubled-back wire, a net with no geometry, and a crossing on each of the two boundary orientations.

**Geometry.** Die `(0, 0, 100, 100)` in DBU, `shift_factor = [0, 0]`, `scale_factor = 1.0` — so DBU and evaluator units coincide and every expected number is readable off the coordinates. `make_grid_regions(die, 2, 2, lattice=8)` gives K=4 with region `iy*2 + ix`: `0` bottom-left, `1` bottom-right, `2` top-left, `3` top-right. GCell edges `[0, 25, 50, 75, 100]` on both axes, so `ResourceGrid.nx == ny == 4`, `horizontal_capacity` is `(4, 3)`, `vertical_capacity` is `(3, 4)`, and `edge_count == 4*3 + 3*4 == 24` — the shape `load_observation` validates (`online_openroad.py:107-109`).

**Nets.**

| net | geometry | what it exercises |
|---|---|---|
| `n0` | metal2 horizontal `y=25`, `x=10→90`; plus metal2 `x=20→80` at the same `y` | crosses the vertical `0|1` boundary at `x=50` once; the second wire is a doubled-back overlap that `segment_union` must collapse |
| `n1` | metal3 vertical `x=90`, `y=10→90`; a via `(90,10)` metal2↔metal3 | crosses the horizontal `1|3` boundary at `y=50`; one via record |
| `n2` | metal2 horizontal `y=75`, `x=10→40`; metal4 horizontal `y=75`, `x=30→90` | two layers, one 2-D projected crossing of the vertical `2|3` boundary at `x=50` (G-3: layer-summed counting would say one, projected counting also says one, and the *union* is what makes them agree) |
| `n3` | no geometry | a routed net the router emitted empty |

Expected 2-D projected per-segment demand: exactly one crossing on each of the `0|1`, `1|3` and `2|3` boundary segments, three in total. Expected via count 1. Expected DBU wirelength `80 + 80 + 30 + 60 = 250`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_campaign_final_grt.py`:

```python
import json
import os

import numpy as np
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "grt_gcd_tiny")


def test_the_fixture_is_checked_in_and_complete():
    for name in ("regions.json", "coord.json", "netmap.json",
                 "grt/settings.json", "grt/segments.txt", "grt/resources.json",
                 "grt/timings.json", "grt/net_audit.json",
                 "grt/net_manifest.jsonl", "grt/run.log", "grt/receipt.json"):
        assert os.path.isfile(os.path.join(FIXTURE_DIR, name)), name


def test_the_fixture_receipt_hashes_match_its_files():
    """load_observation refuses a directory whose recorded output digests do
    not match (online_openroad.py:100-101); a stale fixture must fail loudly
    here rather than inside every later test."""
    from ioplace.route_eval.online_openroad import digest
    receipt = json.load(open(os.path.join(FIXTURE_DIR, "grt", "receipt.json")))
    assert receipt["returncode"] == 0
    for name, sha in receipt["outputs"].items():
        assert digest(os.path.join(FIXTURE_DIR, "grt", name)) == sha, name


def test_the_fixture_regenerates_byte_identically(tmp_path):
    """The generator is the fixture's source of truth; drift between them
    would make the checked-in files unmaintainable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_fixture", os.path.join(FIXTURE_DIR, "make_fixture.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(str(tmp_path))
    for root, _dirs, files in os.walk(FIXTURE_DIR):
        for name in files:
            if name == "make_fixture.py" or name.endswith(".pyc"):
                continue
            relative = os.path.relpath(os.path.join(root, name), FIXTURE_DIR)
            got = os.path.join(str(tmp_path), relative)
            assert os.path.isfile(got), relative
            assert open(got, "rb").read() == open(
                os.path.join(FIXTURE_DIR, relative), "rb").read(), relative


def test_the_fixture_native_congestion_parses():
    from ioplace.route_eval.online_openroad import parse_native_congestion
    text = open(os.path.join(FIXTURE_DIR, "grt", "run.log")).read()
    native = parse_native_congestion(text)
    assert native["total_overflow"] == 0
    assert native["resource"] == 4800 and native["demand"] == 250
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v`
Expected: FAIL — `AssertionError: regions.json` (the fixture does not exist).

- [ ] **Step 3: Write the generator**

Create `tests/fixtures/grt_gcd_tiny/make_fixture.py`:

```python
"""Regenerate the tiny GRT fixture. Needs no OpenROAD.

A complete `route_eval.online_openroad.run_openroad` output directory plus the
three sidecars `load_observation` reads, on a 100x100 DBU die with a K=4 grid
and a 4x4 GCell grid, so every expected number is readable off the geometry:

  n0  metal2 y=25, x=10..90, plus an overlapping x=20..80  -> one 0|1 crossing
  n1  metal3 x=90, y=10..90, plus one metal2<->metal3 via  -> one 1|3 crossing
  n2  metal2 y=75 x=10..40 and metal4 y=75 x=30..90        -> one 2|3 crossing
  n3  routed but empty

  via records 1, decoded wirelength 250 DBU, projected per-segment demand 3.

Run:  "$IOPLACE_PYTHON" tests/fixtures/grt_gcd_tiny/make_fixture.py
"""
import hashlib
import json
import os
import sys

DIE = (0.0, 0.0, 100.0, 100.0)
EDGES = [0, 25, 50, 75, 100]
CAPACITY_PER_EDGE = 200

SEGMENTS = """\
n0
(
10 25 metal2 90 25 metal2
20 25 metal2 80 25 metal2
)
n1
(
90 10 metal2 90 10 metal3
90 10 metal3 90 90 metal3
)
n2
(
10 75 metal2 40 75 metal2
30 75 metal4 90 75 metal4
)
n3
(
)
"""

RUN_LOG = """\
[INFO GRT-0020] Min routing layer: metal2
[INFO GRT-0021] Max routing layer: metal10
[INFO GRT-0001] Running extra iterations to remove overflow.
Final congestion report:
Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Overflow
metal2            2400           150            6.25%           0 /  0 /  0
metal3            2400           100            4.17%           0 /  0 /  0
Total             4800           250            5.21%           0 /  0 /  0
"""


def _digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _write(path, text):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as stream:
        stream.write(text)
    return path


def build(out_dir):
    from ioplace.regions import make_grid_regions

    out_dir = os.path.abspath(out_dir)
    grt = os.path.join(out_dir, "grt")
    os.makedirs(grt, exist_ok=True)

    make_grid_regions(DIE, 2, 2, lattice=8).to_json(
        os.path.join(out_dir, "regions.json"))
    _write(os.path.join(out_dir, "coord.json"),
           json.dumps({"shift_factor": [0.0, 0.0], "scale_factor": 1.0,
                       "def_units_per_micron": 2000,
                       "xl": DIE[0], "yl": DIE[1], "xh": DIE[2], "yh": DIE[3]},
                      indent=1) + "\n")
    _write(os.path.join(out_dir, "netmap.json"),
           json.dumps({"0": "n0", "1": "n1", "2": "n2", "3": "n3"}, indent=1)
           + "\n")

    _write(os.path.join(grt, "segments.txt"), SEGMENTS)
    _write(os.path.join(grt, "run.log"), RUN_LOG)
    nx = ny = len(EDGES) - 1
    resources = {
        "x_edges_dbu": EDGES, "y_edges_dbu": EDGES,
        "horizontal_capacity": [[CAPACITY_PER_EDGE] * (nx - 1) for _ in range(ny)],
        "vertical_capacity": [[CAPACITY_PER_EDGE] * nx for _ in range(ny - 1)],
        "horizontal_usage": [[0] * (nx - 1) for _ in range(ny)],
        "vertical_usage": [[0] * nx for _ in range(ny - 1)],
        "layers": [{"name": "metal2", "direction": "HORIZONTAL"},
                   {"name": "metal3", "direction": "VERTICAL"},
                   {"name": "metal4", "direction": "HORIZONTAL"}],
        "capacity_semantics": ("OpenDB total track capacity; usage includes "
                               "blockages and wires; preferred-direction "
                               "layers summed"),
        "uint8_backend": True, "preferred_layer_overflow": 0,
    }
    _write(os.path.join(grt, "resources.json"),
           json.dumps(resources, indent=1) + "\n")
    _write(os.path.join(grt, "timings.json"),
           json.dumps({"completed": [{"command": "global_route "
                                      "-congestion_iterations 50 "
                                      "-allow_congestion", "elapsed_s": 1.5}],
                       "running": None, "read_lef_def_s": 0.25,
                       "net_audit_s": 0.05, "resource_dump_s": 0.1,
                       "total_s": 2.0}, indent=1) + "\n")
    _write(os.path.join(grt, "net_audit.json"),
           json.dumps({"net_count": 4, "special_net_count": 0,
                       "nontrivial_nonspecial_net_count": 4,
                       "signal_type_counts": {"SIGNAL": 4},
                       "degree_above_counts": {}, "requested_exclusions": [],
                       "sampling": False,
                       "connectivity_sha256": "0" * 64,
                       "largest_nets": []}, indent=1) + "\n")
    _write(os.path.join(grt, "net_manifest.jsonl"),
           "".join(json.dumps({"degree": 2, "endpoints": [], "name": name,
                               "signal_type": "SIGNAL", "special": False},
                              sort_keys=True, separators=(",", ":")) + "\n"
                   for name in ("n0", "n1", "n2", "n3")))

    def_path = os.path.join(out_dir, "tiny.def")
    _write(def_path, "VERSION 5.8 ;\nDESIGN tiny ;\nEND DESIGN\n")
    settings = {
        "def": def_path, "lefs": [], "clear_script": "clear_signal_routing.tcl",
        "cleared": os.path.join(grt, "cleared.txt"),
        "segments": os.path.join(grt, "segments.txt"),
        "resources": os.path.join(grt, "resources.json"),
        "congestion_iterations": 50, "allow_congestion": True, "threads": 4,
        "timings": os.path.join(grt, "timings.json"),
        "net_audit": os.path.join(grt, "net_audit.json"),
        "net_manifest": os.path.join(grt, "net_manifest.jsonl"),
        "signal_layers": "metal2-metal10",
    }
    _write(os.path.join(grt, "settings.json"), json.dumps(settings, indent=2))

    outputs = {name: _digest(os.path.join(grt, name)) for name in
               ("run.log", "segments.txt", "resources.json", "timings.json",
                "net_audit.json", "net_manifest.jsonl")}
    receipt = {
        "command": ["openroad", "-exit", "-python", "dump_online_route.py",
                    os.path.join(grt, "settings.json")],
        "tool_returncode": 0, "returncode": 0, "logged_errors": [],
        "inputs": {def_path: _digest(def_path)},
        "elapsed_s": 2.0,
        "routing_policy": {"congestion_iterations": 50,
                           "allow_congestion": True, "threads": 4,
                           "signal_layers": "metal2-metal10"},
        "outputs": outputs,
        "native_congestion": {"resource": 4800, "demand": 250,
                              "usage_percent": 5.21,
                              "max_horizontal_overflow": 0,
                              "max_vertical_overflow": 0, "total_overflow": 0},
    }
    _write(os.path.join(grt, "receipt.json"), json.dumps(receipt, indent=2))
    return {"out_dir": out_dir, "expected_via_records": 1,
            "expected_wirelength_dbu": 250.0,
            "expected_projected_crossings": 3}


if __name__ == "__main__":
    sys.exit(0 if build(os.path.dirname(os.path.abspath(__file__))) else 1)
```

- [ ] **Step 4: Generate the fixture**

```bash
source src/scripts/env.sh
"$IOPLACE_PYTHON" tests/fixtures/grt_gcd_tiny/make_fixture.py
ls -R tests/fixtures/grt_gcd_tiny
```
Expected: `regions.json coord.json netmap.json tiny.def make_fixture.py` at the top level and eight files under `grt/`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v`
Expected: PASS (4 tests).

`test_the_fixture_regenerates_byte_identically` also regenerates into a tmp dir; because `settings.json` and `receipt.json` embed absolute paths, the generator writes them from `out_dir`, so the two copies differ in exactly those paths. If that test fails on the path fields, change it to compare every file **except** `grt/settings.json` and `grt/receipt.json` byte for byte, and for those two compare the JSON with every value that starts with the fixture directory rewritten to a placeholder. Do that in this commit; do not skip the test.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/grt_gcd_tiny tests/test_campaign_final_grt.py
git commit -m "$(cat <<'MSG'
test(campaign): hand-authored tiny GRT fixture, no OpenROAD needed

A complete online_openroad output directory plus its coord/netmap/regions
sidecars on a 100x100 die with a K=4 grid and a 4x4 GCell grid, so the whole
decode -> per-segment table -> route.json -> matrix path is unit-testable
offline. Exercises a doubled-back wire, a via, a multi-layer net and an empty
routed net, with hand-checkable expectations: 1 via, 250 DBU wirelength, 3
projected boundary crossings.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Two additive changes to `online_openroad.py` — a timeout and `net_polylines`

The protocol needs a wall-clock guard (unbounded congestion removal cost 9–10 h before the protocol change) and the decoded per-net geometry (P-D Task 10 asks for the same key, in the same words). Both changes are additive and break no caller.

**Files:**
- Modify: `src/ioplace/route_eval/online_openroad.py:44-94` (`run_openroad`) and `:137-159` (`load_observation`)
- Test: `tests/test_online_openroad_additions.py`

**Interfaces:**
- Consumes: the Task 2 fixture.
- Produces:
  - `run_openroad(def_path, lefs, out, binary, *, congestion_iterations=50, allow_congestion=False, threads=4, signal_layers=None, timeout=None) -> dict` — the receipt gains `routing_policy["timeout_s"]`, and on expiry writes `returncode=124`, `timeout_expired=True` before raising `RuntimeError`
  - `load_observation(...)` — the returned dict gains `net_polylines: dict(int -> (M,2,2) float64 ndarray)`, the 2-D geometry of each routed net in evaluator units

**Shared seam with P-D.** P-D Task 10 Step 3 note 1 says, of `load_observation`: *"Either add a `net_polylines` key there — a one-line `net_polylines[net] = np.asarray(segments)` next to the existing `net_keys[net] = ...`, which is additive and breaks no caller — or decode `segments.txt` directly."* Whichever subproject lands first adds it; the second checks it is there and does nothing. Task 0's pre-flight does not check this key (it is added here), so the implementer must `grep -n "net_polylines" src/ioplace/route_eval/online_openroad.py` before editing and skip Step 4 if P-D already added it — recording that in the commit message.

- [ ] **Step 1: Write the failing test**

Create `tests/test_online_openroad_additions.py`:

```python
import json
import os
import stat
import subprocess

import numpy as np
import pytest

from ioplace.route_eval import online_openroad as module

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "grt_gcd_tiny")


def _fake_binary(tmp_path, body):
    path = tmp_path / "openroad"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_run_openroad_rejects_a_nonpositive_timeout(tmp_path):
    with pytest.raises(ValueError, match="positive number timeout"):
        module.run_openroad(tmp_path / "x.def", [], tmp_path / "out",
                            _fake_binary(tmp_path, "exit 0"), timeout=0)


def test_run_openroad_records_the_timeout_in_the_routing_policy(tmp_path):
    """Bounded iterations are mandatory (spec sec 8); the wall-clock guard is
    the second line of defence and must be visible in the receipt."""
    def_path = tmp_path / "x.def"
    def_path.write_text("VERSION 5.8 ;\n")
    out = tmp_path / "out"
    binary = _fake_binary(tmp_path, "sleep 5")
    with pytest.raises(RuntimeError, match="timed out"):
        module.run_openroad(def_path, [], out, binary, timeout=0.5)
    receipt = json.load(open(out / "receipt.json"))
    assert receipt["returncode"] == 124
    assert receipt["timeout_expired"] is True
    assert receipt["routing_policy"]["timeout_s"] == 0.5


def test_load_observation_returns_per_net_polylines():
    """G-3 and P-D Task 10 both need the decoded 2-D geometry per net."""
    observation = module.load_observation(
        os.path.join(FIXTURE_DIR, "grt"),
        os.path.join(FIXTURE_DIR, "coord.json"),
        os.path.join(FIXTURE_DIR, "regions.json"),
        os.path.join(FIXTURE_DIR, "netmap.json"))
    polylines = observation["net_polylines"]
    assert set(polylines) == {0, 1, 2}          # n3 was routed but empty
    for geometry in polylines.values():
        assert geometry.ndim == 3 and geometry.shape[1:] == (2, 2)
    # n0's two overlapping metal2 wires collapse to one 10..90 run.
    assert polylines[0].shape[0] == 1
    assert polylines[0][0].tolist() == [[10.0, 25.0], [90.0, 25.0]]
    # n2 keeps two collinear-but-overlapping runs from two layers; the union
    # inside route_segment_demand merges them into 10..90 at y=75.
    assert sorted(float(a[0]) for a, _b in polylines[2]) == [10.0, 30.0]


def test_load_observation_polylines_are_in_evaluator_units():
    observation = module.load_observation(
        os.path.join(FIXTURE_DIR, "grt"),
        os.path.join(FIXTURE_DIR, "coord.json"),
        os.path.join(FIXTURE_DIR, "regions.json"),
        os.path.join(FIXTURE_DIR, "netmap.json"))
    coord = json.load(open(os.path.join(FIXTURE_DIR, "coord.json")))
    assert coord["shift_factor"] == [0.0, 0.0] and coord["scale_factor"] == 1.0
    assert observation["actual_io"] == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_online_openroad_additions.py -v`
Expected: FAIL — `TypeError: run_openroad() got an unexpected keyword argument 'timeout'` and `KeyError: 'net_polylines'`.

- [ ] **Step 3: Add the timeout to `run_openroad`**

In `src/ioplace/route_eval/online_openroad.py`, change the signature at line 44 and the validation and subprocess block:

```python
def run_openroad(def_path,lefs,out,binary,*,congestion_iterations=50,allow_congestion=False,threads=4,signal_layers=None,timeout=None):
    for name, value in (('congestion_iterations', congestion_iterations), ('threads', threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f'positive integer {name} required')
    if timeout is not None and (not isinstance(timeout,(int,float)) or isinstance(timeout,bool) or timeout<=0):
        raise ValueError('positive number timeout required')
```

and, replacing the `subprocess.run` block at lines 68-76:

```python
    started=time.perf_counter()
    expired=False
    with (out/'run.log').open('w') as log:
        try:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        except subprocess.TimeoutExpired:
            # The bounded -congestion_iterations policy is the real guard
            # (spec sec 8); this is the backstop for the 9-10 h unbounded
            # behaviour recorded in the 2026-09-15 router diagnosis.
            expired=True
            result=None
    with (out/'run.log').open(errors='replace') as log:
        errors=[line.strip() for line in log if '[ERROR ' in line]
    returncode=124 if expired else (result.returncode or (1 if errors else 0))
    receipt=dict(command=command,tool_returncode=None if expired else result.returncode,
        returncode=returncode,timeout_expired=expired,
        logged_errors=errors,inputs=input_hashes,elapsed_s=time.perf_counter()-started,
        routing_policy=dict(congestion_iterations=congestion_iterations,allow_congestion=allow_congestion,
                            threads=threads,timeout_s=timeout))
```

and change the final raise at line 93 so a timeout says so:

```python
    if returncode:
        raise RuntimeError(f'OpenROAD online route {"timed out" if expired else "failed"}; see {out}/run.log')
```

- [ ] **Step 4: Add `net_polylines` to `load_observation`**

First check whether P-D already did it:

```bash
grep -n "net_polylines" src/ioplace/route_eval/online_openroad.py
```

If it prints nothing, edit lines 137-149. Add `net_polylines={}` to the initialisers on line 137 and one line inside the per-net loop next to `net_keys[net] = ...`:

```python
    net_io={};net_keys={};net_usage={};net_wirelength={};outside={};net_polylines={}
    for net,layers in nets.items():
        ...
        net_keys[net]=grid.edge_keys(np.asarray(segments).reshape(-1,2,2)).tolist()
        # 2-D projection of the net across layers, in evaluator units. P-G G-3
        # and P-D Task 10 both count per-segment crossings on the projection:
        # "one net crossing consumes one track" is layer-free, unlike the
        # layer-summed `actual_io` above.
        net_polylines[net]=np.asarray(segments,dtype=float).reshape(-1,2,2)
        net_usage[net]=layer_resource_usage(grid,layers)
```

and add `net_polylines=net_polylines,` to the returned dict at line 148.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_online_openroad_additions.py tests/test_campaign_final_grt.py -v`
Expected: PASS (9 tests).

- [ ] **Step 6: Run every existing route_eval test to prove the changes are additive**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_joint_online_driver.py tests/test_ggr_common_eval.py tests/test_placement_openroad.py tests/test_bounded_grt_feedback.py tests/test_corrected_route_publication.py -v`
Expected: PASS, same counts as before the edit. If any of them constructs a receipt dict by hand and asserts its exact key set, add `timeout_expired` / `timeout_s` there in this commit.

- [ ] **Step 7: Commit**

```bash
git add src/ioplace/route_eval/online_openroad.py \
        tests/test_online_openroad_additions.py
git commit -m "$(cat <<'MSG'
feat(route_eval): optional GRT wall-clock timeout and per-net polylines

Two additive changes the v2 final-GRT protocol needs. run_openroad gains
timeout=, recorded in routing_policy and reported as returncode 124 with
timeout_expired -- the backstop behind spec sec 8's mandatory bounded
-congestion_iterations, against the 9-10 h unbounded behaviour measured in
docs/results/2026-09-15-benchmark-router-diagnosis.md. load_observation gains
net_polylines, the 2-D projection of each routed net in evaluator units,
which P-G's per-segment crossing table and P-D Task 10's rank correlation
both consume.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Decode and the per-segment crossing table — `final_grt.py`, part 1

The half of the recipe that needs no OpenROAD and no GPU: read `segments.txt`, project each net to 2-D, map it through P-D's unit-edge raster, and reduce to the four primary numbers spec §8 names.

**Files:**
- Create: `src/ioplace/campaign/final_grt.py`
- Test: `tests/test_campaign_final_grt.py` (append)

**Interfaces:**
- Consumes:
  - `ioplace.route_eval.common_grt.read_nets(path)` (landed, `common_grt.py:29-58`) — yields `(name, [(x0,y0,l0,x1,y1,l1), ...])`
  - `ioplace.route_eval.online_openroad.load_observation(out, coord, regions, netmap)` with the Task 3 `net_polylines` key
  - `ioplace.region_segments.{enumerate_segments, segment_utilisation, segments_digest, CAPACITY_SCALARS}` — from plan P-D Task 1, verify at pre-flight
  - `src.scripts.run_cap_rank_correlation.route_segment_demand(rg, table, polylines) -> (S,) int64` — from plan P-D Task 10, verify at pre-flight. **Imported, not re-implemented**, so the router-side crossing convention has exactly one definition in the repo.
  - `ioplace.drivers.run_placement_io.load_capacity_for_grid(path, rg) -> (table, capacity, metadata)` — from plan P-D Task 8, verify at pre-flight
  - `ioplace.region_grid.RegionGrid`, `ioplace.regions.RegionSet` (landed)
- Produces:
  - `GRT_POLICY = {"congestion_iterations": 50, "allow_congestion": True, "threads": 4, "signal_layers": "metal2-metal10"}`
  - `ROUTE_SCHEMA_VERSION = 1`
  - `decode_segments(segments_path) -> dict` with `via_records`, `wire_records`, `nonrectilinear_records`, `wirelength_dbu`, `routed_net_count`, `nets_with_geometry`, `net_names_sha256`
  - `router_segment_table(rg, table, net_polylines) -> dict` with `demand` `(S,) int64`, `demand_layered` `(S,) int64`
  - `segment_metrics(demand, capacity, table) -> dict` carrying every name in `CAPACITY_SCALARS` plus `total_overflow_tracks`
  - `build_route_record(...) -> dict` — the `route.json` payload

**Why two demand arrays.** G-3: `demand` is the 2-D projection (one net crossing = one track, the semantics P-D's capacity is expressed in) and is the primary. `demand_layered` sums the per-layer crossings, reconciling with `load_observation`'s `actual_io`. A large gap means heavy layer stacking over boundaries, which is evidence, not noise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_campaign_final_grt.py`:

```python
def test_decode_counts_vias_wires_and_dbu_wirelength():
    from ioplace.campaign.final_grt import decode_segments
    decoded = decode_segments(os.path.join(FIXTURE_DIR, "grt", "segments.txt"))
    assert decoded["via_records"] == 1
    assert decoded["wire_records"] == 5
    assert decoded["routed_net_count"] == 4
    assert decoded["nets_with_geometry"] == 3
    assert decoded["wirelength_dbu"] == pytest.approx(80 + 60 + 80 + 30 + 60)
    assert len(decoded["net_names_sha256"]) == 64


def test_router_segment_table_projects_layers_before_counting():
    """G-3: one net crossing consumes one track regardless of how many layers
    it used. n2 crosses the 2|3 boundary on metal2-and-metal4 overlapping
    geometry and must be charged once."""
    from ioplace.campaign.final_grt import router_segment_table
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import RegionSet
    from ioplace.route_eval.online_openroad import load_observation

    rg = RegionGrid(RegionSet.from_json(os.path.join(FIXTURE_DIR,
                                                     "regions.json")))
    table = enumerate_segments(rg)
    observation = load_observation(
        os.path.join(FIXTURE_DIR, "grt"),
        os.path.join(FIXTURE_DIR, "coord.json"),
        os.path.join(FIXTURE_DIR, "regions.json"),
        os.path.join(FIXTURE_DIR, "netmap.json"))
    result = router_segment_table(rg, table, observation["net_polylines"])
    assert int(result["demand"].sum()) == 3
    crossed = np.nonzero(result["demand"])[0]
    pairs = sorted((int(table.pair_a[s]), int(table.pair_b[s])) for s in crossed)
    assert pairs == [(0, 1), (1, 3), (2, 3)]
    assert result["demand"].dtype == np.int64


def test_segment_metrics_report_every_capacity_scalar_plus_overflow_tracks():
    from ioplace.campaign.final_grt import segment_metrics
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import CAPACITY_SCALARS, enumerate_segments
    from ioplace.regions import RegionSet

    rg = RegionGrid(RegionSet.from_json(os.path.join(FIXTURE_DIR,
                                                     "regions.json")))
    table = enumerate_segments(rg)
    demand = np.zeros(table.num_segments, dtype=np.int64)
    capacity = np.full(table.num_segments, 4.0)
    demand[0] = 6            # 1.5x over
    demand[1] = 2            # under
    capacity[2] = 0.0        # a blocked segment ...
    demand[2] = 3            # ... that the router used anyway
    metrics = segment_metrics(demand, capacity, table)
    for name in CAPACITY_SCALARS:
        assert name in metrics, name
    assert metrics["num_over_capacity"] == 1
    assert metrics["num_zero_capacity_segments"] == 1
    assert metrics["zero_capacity_demand"] == 3
    assert metrics["max_util"] == pytest.approx(1.5)
    assert metrics["total_overflow_tracks"] == pytest.approx(2.0)
    assert metrics["segment_demand_total"] == 11


def test_segment_metrics_are_zero_when_nothing_is_over_capacity():
    from ioplace.campaign.final_grt import segment_metrics
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import RegionSet
    rg = RegionGrid(RegionSet.from_json(os.path.join(FIXTURE_DIR,
                                                     "regions.json")))
    table = enumerate_segments(rg)
    demand = np.ones(table.num_segments, dtype=np.int64)
    metrics = segment_metrics(demand, np.full(table.num_segments, 10.0), table)
    assert metrics["num_over_capacity"] == 0
    assert metrics["total_overflow_tracks"] == 0.0
    assert metrics["max_util"] == pytest.approx(0.1)


def test_build_route_record_carries_the_policy_and_every_hash(tmp_path):
    from ioplace.campaign.final_grt import (GRT_POLICY, ROUTE_SCHEMA_VERSION,
                                            build_route_record)
    record = build_route_record(
        case="gcd", arm="ours", status="ok",
        segment_metrics_out={"num_over_capacity": 0, "max_util": 0.1,
                             "p99_util": 0.1, "total_overflow_tracks": 0.0,
                             "segment_demand_total": 3,
                             "num_zero_capacity_segments": 0,
                             "zero_capacity_demand": 0, "num_segments": 8},
        decoded={"via_records": 1, "wire_records": 5, "wirelength_dbu": 310.0,
                 "routed_net_count": 4, "nets_with_geometry": 3,
                 "net_names_sha256": "a" * 64, "nonrectilinear_records": 0},
        observation={"actual_io": 3, "actual_wirelength": 310.0,
                     "aggregated_resource_overflow": 0.0,
                     "preferred_layer_overflow": 0,
                     "routed_net_count": 4,
                     "routed_net_names_sha256": "b" * 64,
                     "native_congestion": {"total_overflow": 0,
                                           "max_horizontal_overflow": 0,
                                           "max_vertical_overflow": 0,
                                           "usage_percent": 5.21},
                     "router_elapsed_s": 2.0},
        demand_layered_total=3,
        repair={"changed_location_count": 12, "changed_orientation_count": 4,
                "maximum_displacement_dbu": 5.0, "receipt_sha256": "c" * 64},
        timings={"t_export_s": 1.0, "t_repair_s": 2.0, "t_grt_s": 3.0,
                 "t_decode_s": 0.5, "t_eval_s": 0.25},
        hashes={"placement.npz": "d" * 64},
        segments_digest_value="e" * 64, capacity_source="openroad_gcell")
    assert record["schema_version"] == ROUTE_SCHEMA_VERSION
    assert record["routing_policy"] == GRT_POLICY
    assert record["case"] == "gcd" and record["arm"] == "ours"
    assert record["num_over_capacity"] == 0
    assert record["segment_demand_router_total"] == 3
    assert record["segment_demand_router_layered_total"] == 3
    assert record["native_total_overflow"] == 0
    assert record["via_count"] == 1
    assert record["routed_wirelength_dbu"] == 310.0
    assert record["sha256"]["placement.npz"] == "d" * 64
    assert record["segments_sha256"] == "e" * 64
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.campaign.final_grt'`.

- [ ] **Step 3: Write the decode half of `src/ioplace/campaign/final_grt.py`**

```python
"""The v2 design sec 8 per-arm recipe: one global route, at the very end.

placement.npz -> DEF export -> OpenROAD DEF row repair (mandatory) -> one
bounded global_route -> decode -> per-segment crossings against the P-D
capacity -> route.json with SHA-256 receipts.

Three conventions this module fixes, recorded in the P-G plan:

  G-2  "decode" is common_grt.read_nets plus online_openroad.load_observation.
       common_grt.evaluate needs a per-layer capacity array that the online
       resources.json does not carry (dump_online_route.py:98-114 sums the
       preferred-direction layers before writing).
  G-3  Per-segment demand is counted on a net's 2-D projection across layers,
       because the capacity semantics is "one net crossing consumes one
       track". The layer-summed count is reported alongside, not instead.
  G-4  Segment ids come from the 512-lattice label grid, so enumeration on
       the scaled RegionGrid matches the native one; the capacity loader
       validates that.

Never import this module from a placement inner loop. spec sec 0: the router
runs once, at the very end of every arm.
"""
import hashlib
import json
import os
import time

import numpy as np

ROUTE_SCHEMA_VERSION = 1

#: spec sec 8, verbatim: set_routing_layers metal2-metal10,
#: -congestion_iterations 50, -allow_congestion, 4 GRT threads.
GRT_POLICY = {"congestion_iterations": 50, "allow_congestion": True,
              "threads": 4, "signal_layers": "metal2-metal10"}

#: Wall-clock backstop behind the bounded iteration count. Measured anchors
#: (docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43): tile
#: 221.9 s at 5 iterations; unbounded congestion removal 9-10 h on tile and
#: group. spec sec 8's own budget is ~4 min (tile) and 1-3 h (group).
DEFAULT_GRT_TIMEOUT_S = 6 * 3600.0


def decode_segments(segments_path):
    """Via count, wire count and DBU wirelength from grt::write_segments.

    Uses common_grt.read_nets -- the same streaming parser common_grt.evaluate
    uses -- rather than evaluate itself (G-2). Wirelength is the raw sum over
    wire records in DBU, before any per-net union, so it is comparable with
    the router's own native wirelength; the unioned, region-aware length is
    load_observation's `actual_wirelength`.
    """
    from ioplace.route_eval.common_grt import read_nets
    via = wire = nonrectilinear = 0
    length = 0.0
    names, with_geometry = [], 0
    for name, records in read_nets(segments_path):
        names.append(name)
        if records:
            with_geometry += 1
        for x0, y0, l0, x1, y1, l1 in records:
            if l0 != l1:
                via += 1
                continue
            if x0 != x1 and y0 != y1:
                nonrectilinear += 1
                continue
            if x0 == x1 and y0 == y1:
                continue
            wire += 1
            length += float(abs(x1 - x0) + abs(y1 - y0))
    return {"via_records": int(via), "wire_records": int(wire),
            "nonrectilinear_records": int(nonrectilinear),
            "wirelength_dbu": float(length),
            "routed_net_count": len(names),
            "nets_with_geometry": int(with_geometry),
            "net_names_sha256": hashlib.sha256(json.dumps(
                sorted(names), separators=(",", ":")).encode()).hexdigest()}


def router_segment_table(rg, table, net_polylines):
    """Per-segment routed crossings, projected and layer-summed.

    `net_polylines` is load_observation's per-net geometry in evaluator units.
    `demand` unions each net's whole geometry once before counting, so a
    crossing made on three layers costs one track (G-3). `demand_layered`
    counts each layer separately and reconciles with `actual_io`.

    The counting itself is P-D's route_segment_demand -- imported, never
    re-implemented, so the router-side convention has one definition.
    """
    from src.scripts.run_cap_rank_correlation import route_segment_demand
    demand = np.zeros(int(table.num_segments), dtype=np.int64)
    layered = np.zeros(int(table.num_segments), dtype=np.int64)
    for geometry in net_polylines.values():
        geometry = np.asarray(geometry, dtype=np.float64).reshape(-1, 2, 2)
        if geometry.size == 0:
            continue
        demand += route_segment_demand(rg, table, geometry)
        for single in geometry:
            layered += route_segment_demand(rg, table, single.reshape(1, 2, 2))
    return {"demand": demand, "demand_layered": layered}


def segment_metrics(demand, capacity, table):
    """spec sec 8's primary metric block, on the same names P-D's evaluator
    uses (region_segments.CAPACITY_SCALARS), plus the total overflow tracks.

    Zero-capacity segments are counted and their demand reported separately;
    they are never given an epsilon capacity (P-D's unit rule: "zero-capacity
    segments stay hard-blocked ... no epsilon substitution") and never enter
    the utilisation statistics.
    """
    from ioplace.region_segments import segment_utilisation
    demand = np.asarray(demand, dtype=np.int64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if demand.shape != (int(table.num_segments),) or capacity.shape != demand.shape:
        raise ValueError("demand and capacity must be one value per segment")
    metrics = dict(segment_utilisation(demand, capacity))
    routable = capacity > 0
    overflow = np.maximum(demand[routable] - capacity[routable], 0.0)
    metrics["total_overflow_tracks"] = float(overflow.sum())
    metrics["segment_demand_total"] = int(demand.sum())
    metrics["num_segments"] = int(table.num_segments)
    return metrics


def build_route_record(*, case, arm, status, segment_metrics_out, decoded,
                       observation, demand_layered_total, repair, timings,
                       hashes, segments_digest_value, capacity_source,
                       routing_policy=None, notes=None):
    """The route.json payload. Primary metrics first, then spec sec 8's
    secondary list, then provenance."""
    native = observation.get("native_congestion") or {}
    record = {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "case": case, "arm": arm, "status": status,
        # --- primary: per-segment crossings vs capacity (spec sec 8) ---
        "num_over_capacity": int(segment_metrics_out["num_over_capacity"]),
        "max_util": float(segment_metrics_out["max_util"]),
        "p99_util": float(segment_metrics_out["p99_util"]),
        "total_overflow_tracks": float(segment_metrics_out["total_overflow_tracks"]),
        "segment_demand_router_total": int(segment_metrics_out["segment_demand_total"]),
        "segment_demand_router_layered_total": int(demand_layered_total),
        "num_segments": int(segment_metrics_out["num_segments"]),
        "num_zero_capacity_segments": int(segment_metrics_out["num_zero_capacity_segments"]),
        "zero_capacity_demand": int(segment_metrics_out["zero_capacity_demand"]),
        # --- secondary (spec sec 8) ---
        "native_total_overflow": native.get("total_overflow"),
        "native_max_horizontal_overflow": native.get("max_horizontal_overflow"),
        "native_max_vertical_overflow": native.get("max_vertical_overflow"),
        "native_usage_percent": native.get("usage_percent"),
        "routed_wirelength_dbu": float(decoded["wirelength_dbu"]),
        "via_count": int(decoded["via_records"]),
        "nonrectilinear_records": int(decoded["nonrectilinear_records"]),
        "actual_io": observation.get("actual_io"),
        "actual_wirelength": observation.get("actual_wirelength"),
        "aggregated_resource_overflow": observation.get("aggregated_resource_overflow"),
        "preferred_layer_overflow": observation.get("preferred_layer_overflow"),
        "routed_net_count": int(decoded["routed_net_count"]),
        "nets_with_geometry": int(decoded["nets_with_geometry"]),
        "decoded_net_names_sha256": decoded["net_names_sha256"],
        "routed_net_names_sha256": observation.get("routed_net_names_sha256"),
        # --- provenance ---
        "routing_policy": dict(routing_policy or GRT_POLICY),
        "grt_elapsed_s": observation.get("router_elapsed_s"),
        "row_repair": dict(repair),
        "segments_sha256": segments_digest_value,
        "capacity_source": capacity_source,
        "sha256": dict(hashes),
        "notes": list(notes or []),
    }
    record.update({key: float(value) for key, value in timings.items()})
    return record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v`
Expected: PASS (9 tests).

If `segment_utilisation` returns a different set of names than `CAPACITY_SCALARS`, do **not** rename anything here: P-D owns those names and the evaluator reports them; fix the test's expectations to the real names and record the mismatch in the commit message so the table in Task 6 uses the same ones.

- [ ] **Step 5: Commit**

```bash
git add src/ioplace/campaign/final_grt.py tests/test_campaign_final_grt.py
git commit -m "$(cat <<'MSG'
feat(campaign): decode routed segments into per-segment crossings

The offline half of the v2 design sec 8 recipe: read grt::write_segments
with common_grt.read_nets (G-2), project each net's geometry across layers
before counting so one net crossing costs one track (G-3), map it through
P-D's unit-edge raster by calling P-D's own route_segment_demand rather than
re-implementing it, and reduce to num_over_capacity / max_util / p99_util /
total overflow tracks on region_segments.CAPACITY_SCALARS' names. Zero-
capacity segments are reported, never given an epsilon.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: DEF export, row repair, one route — `final_grt.py`, part 2

The half that rebuilds a PlaceDB, writes a DEF from `placement.npz`, repairs its rows in OpenROAD, runs exactly one `global_route`, and glues the whole recipe into `run_final_grt`.

**Files:**
- Modify: `src/ioplace/campaign/final_grt.py` (append)
- Test: `tests/test_campaign_final_grt.py` (append)

**Interfaces:**
- Consumes:
  - `ioplace.drivers.run_placement._load_dreamplace(config) -> (params, placedb)` (landed, used identically at `src/scripts/run_route_gp.py:171-172`)
  - `ioplace.export.def_export.export_def(placedb, params, node_x, node_y, out_dir, region_set) -> dict` (landed, `def_export.py:43`) — writes `out.def`, `regions.json`, `netmap.json`, `coord.json`
  - `ioplace.route_eval.placement_openroad.legalize_export(export_dir, lefs, out, binary, movable_names, threads=16) -> dict` (landed, `placement_openroad.py:45`)
  - `ioplace.route_eval.online_openroad.{run_openroad, load_observation}` with Task 3's additions
  - `ioplace.artifacts.{scaled_region_set, file_sha256}` — from plan P-B Task 1, verify at pre-flight
  - `ioplace.drivers.run_placement_io.load_capacity_for_grid` — from plan P-D Task 8, verify at pre-flight
- Produces:
  - `export_arm_def(config, placement_npz, regions_json, out_dir) -> dict` with `out_def`, `coord_json`, `netmap_json`, `regions_json`, `movable_names`, `lefs`, `scaled_die`, `shift_factor`, `scale_factor`
  - `repair_def_rows(export_dir, lefs, out_dir, binary, movable_names, threads=16) -> dict`
  - `run_final_grt(case, arm, arm_dir, *, config, regions_json, capacity_npz, binary, timeout=DEFAULT_GRT_TIMEOUT_S, threads=4, congestion_iterations=50, repair_threads=16) -> dict`

**Order, and why it is not negotiable.** Export → repair → route → decode → evaluate, all on the *same* coordinates. The repair moves cells (648,869 of them on group, `docs/results/2026-09-15-route-gp-completion-audit.md:110-121`), so any metric measured before it describes a different layout than the one that was routed. `run_final_grt` therefore re-runs nothing on the pre-repair placement and records the repaired coordinates it routed in `repaired_placement.npz`.

**Row repair returns coordinates in DBU, ordered by `movable_names`.** `legalize_export` gives `x_dbu`/`y_dbu`/`orientations` in that exact order (`placement_openroad.py:109-118`), and the sidecar `coord.json` gives `shift_factor`/`scale_factor`, so evaluator units are `(dbu - shift) * scale`. That conversion is the same one `run_route_gp.repair_routed_snapshot` performs at `src/scripts/run_route_gp.py:66-68`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_campaign_final_grt.py`:

```python
def test_run_final_grt_sequences_the_recipe_in_order(monkeypatch, tmp_path):
    """spec sec 8: export -> mandatory row repair -> one bounded global_route
    -> decode -> evaluate, all on the same repaired coordinates, and exactly
    one route call."""
    from ioplace.campaign import final_grt as module

    calls = []
    arm_dir = tmp_path / "arms" / "ours"
    (arm_dir).mkdir(parents=True)
    np.savez_compressed(arm_dir / "placement.npz",
                        node_x=np.zeros(4), node_y=np.zeros(4))

    def fake_export(config, placement_npz, regions_json, out_dir):
        calls.append("export")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "out.def"), "w").write("VERSION 5.8 ;\n")
        return {"out_def": os.path.join(out_dir, "out.def"),
                "coord_json": os.path.join(FIXTURE_DIR, "coord.json"),
                "netmap_json": os.path.join(FIXTURE_DIR, "netmap.json"),
                "regions_json": os.path.join(FIXTURE_DIR, "regions.json"),
                "movable_names": ["u0", "u1"], "lefs": [],
                "scaled_die": (0.0, 0.0, 100.0, 100.0),
                "shift_factor": (0.0, 0.0), "scale_factor": 1.0,
                "num_movable": 2}

    def fake_repair(export_dir, lefs, out_dir, binary, movable_names,
                    threads=16):
        calls.append("repair")
        os.makedirs(out_dir, exist_ok=True)
        legal = os.path.join(out_dir, "legalized.def")
        open(legal, "w").write("VERSION 5.8 ;\n")
        return {"legal_def_path": legal, "x_dbu": np.zeros(2),
                "y_dbu": np.zeros(2), "orientations": np.array(["R0", "R0"]),
                "receipt": {"changed_location_count": 7,
                            "changed_orientation_count": 3,
                            "maximum_displacement_dbu": 11.0},
                "receipt_path": os.path.join(out_dir, "receipt.json")}

    def fake_run_openroad(def_path, lefs, out, binary, **kwargs):
        calls.append(("route", kwargs["congestion_iterations"],
                      kwargs["allow_congestion"], kwargs["threads"],
                      kwargs["signal_layers"], kwargs["timeout"]))
        import shutil
        shutil.copytree(os.path.join(FIXTURE_DIR, "grt"), str(out))
        return json.load(open(os.path.join(str(out), "receipt.json")))

    monkeypatch.setattr(module, "export_arm_def", fake_export)
    monkeypatch.setattr(module, "repair_def_rows", fake_repair)
    monkeypatch.setattr(module, "_run_openroad", fake_run_openroad)
    monkeypatch.setattr(module, "_load_capacity", lambda path, rg, table:
                        (np.full(table.num_segments, 8.0), "lef_pitch"))
    monkeypatch.setattr(module, "_reevaluate", lambda *a, **k: {})
    open(tmp_path / "cap.npz", "w").write("")
    open(tmp_path / "openroad", "w").write("")

    record = module.run_final_grt(
        "gcd", "ours", str(arm_dir), config="c.json",
        regions_json=os.path.join(FIXTURE_DIR, "regions.json"),
        capacity_npz=str(tmp_path / "cap.npz"),
        binary=str(tmp_path / "openroad"), timeout=123.0)

    assert [c if isinstance(c, str) else c[0] for c in calls] == [
        "export", "repair", "route"]
    assert calls[2] == ("route", 50, True, 4, "metal2-metal10", 123.0)
    assert record["status"] == "ok"
    assert record["num_over_capacity"] == 0
    assert record["segment_demand_router_total"] == 3
    assert record["row_repair"]["changed_location_count"] == 7
    assert os.path.isfile(arm_dir / "route.json")
    assert os.path.isfile(arm_dir / "route_segments.npz")


def test_run_final_grt_refuses_to_route_twice(monkeypatch, tmp_path):
    """The router runs once per arm, at the very end (spec sec 0)."""
    from ioplace.campaign import final_grt as module
    arm_dir = tmp_path / "arms" / "ours"
    arm_dir.mkdir(parents=True)
    np.savez_compressed(arm_dir / "placement.npz",
                        node_x=np.zeros(2), node_y=np.zeros(2))
    (arm_dir / "grt").mkdir()
    with pytest.raises(RuntimeError, match="already routed"):
        module.run_final_grt("gcd", "ours", str(arm_dir), config="c.json",
                             regions_json="r.json", capacity_npz="c.npz",
                             binary="openroad")


def test_run_final_grt_requires_a_placement(tmp_path):
    from ioplace.campaign import final_grt as module
    arm_dir = tmp_path / "arms" / "ours"
    arm_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="placement.npz"):
        module.run_final_grt("gcd", "ours", str(arm_dir), config="c.json",
                             regions_json="r.json", capacity_npz="c.npz",
                             binary="openroad")


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("IOPLACE_OPENROAD_TESTS") != "1"
                    or not os.access(os.environ.get("OPENROAD_BIN", ""), os.X_OK),
                    reason="needs IOPLACE_OPENROAD_TESTS=1 and OPENROAD_BIN")
def test_export_and_repair_round_trip_on_gcd(tmp_path):
    """The real export/repair pair on the smallest LEF/DEF case on this host."""
    from ioplace.campaign.final_grt import export_arm_def, repair_def_rows
    from ioplace.regions import make_grid_regions
    from ioplace.drivers.run_placement import _load_dreamplace, extract_final_positions

    config = "results/route_feedback_20260914/gcd.json"
    params, placedb = _load_dreamplace(config)
    placedb.initialize(params)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh),
           float(placedb.yh))
    regions = str(tmp_path / "regions.json")
    # native units: the export helper converts to scaled itself
    native = (die[0] / params.scale_factor + params.shift_factor[0],
              die[1] / params.scale_factor + params.shift_factor[1],
              die[2] / params.scale_factor + params.shift_factor[0],
              die[3] / params.scale_factor + params.shift_factor[1])
    make_grid_regions(native, 2, 2, lattice=512).to_json(regions)
    placement = str(tmp_path / "placement.npz")
    np.savez_compressed(placement,
                        node_x=np.asarray(placedb.node_x, dtype=np.float64),
                        node_y=np.asarray(placedb.node_y, dtype=np.float64))
    exported = export_arm_def(config, placement, regions,
                              str(tmp_path / "export"))
    assert os.path.isfile(exported["out_def"])
    repaired = repair_def_rows(str(tmp_path / "export"), exported["lefs"],
                               str(tmp_path / "repair"),
                               os.environ["OPENROAD_BIN"],
                               exported["movable_names"], threads=4)
    assert os.path.isfile(repaired["legal_def_path"])
    assert len(repaired["x_dbu"]) == len(exported["movable_names"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v -m "not slow"`
Expected: FAIL — `AttributeError: module 'ioplace.campaign.final_grt' has no attribute 'export_arm_def'`.

- [ ] **Step 3: Append the export, repair and recipe code to `src/ioplace/campaign/final_grt.py`**

```python
def _run_openroad(def_path, lefs, out, binary, **kwargs):
    """Indirection so tests can substitute the router without touching the
    real adapter's validation."""
    from ioplace.route_eval.online_openroad import run_openroad
    return run_openroad(def_path, lefs, out, binary, **kwargs)


def _load_capacity(capacity_npz, rg, table):
    """(capacity, capacity_source) for this arm's segment table.

    load_capacity_for_grid re-enumerates and compares segments_digest, so a
    capacity.npz built for another geometry raises before anything is routed
    -- the most damaging silent failure in this protocol.
    """
    from ioplace.drivers.run_placement_io import load_capacity_for_grid
    loaded_table, capacity, metadata = load_capacity_for_grid(capacity_npz, rg)
    from ioplace.region_segments import segments_digest
    if segments_digest(loaded_table) != segments_digest(table):
        raise ValueError("capacity.npz segment table does not match this "
                         "arm's region geometry")
    return np.asarray(capacity, dtype=np.float64), metadata.get(
        "capacity_source", "unknown")


def export_arm_def(config, placement_npz, regions_json, out_dir):
    """Rebuild the PlaceDB and write out.def + the three routing sidecars.

    `placement.npz` holds node_x/node_y in *scaled* evaluator units -- that is
    what run_main_flow saves (extract_final_positions' frame) and what
    export_def documents as its input (def_export.py:52-58). `regions.json` is
    in *native* post-read units (the v2 coordinate contract), so it is
    converted here: load_observation later transforms routed DBU geometry by
    (v - shift) * scale and indexes the RegionGrid with the result, so the
    regions.json written next to the DEF must be the scaled one.
    """
    from ioplace.artifacts import scaled_region_set
    from ioplace.drivers.run_placement import _load_dreamplace
    from ioplace.export.def_export import export_def
    from ioplace.regions import RegionSet

    placement_npz = os.path.abspath(placement_npz)
    if not os.path.isfile(placement_npz):
        raise FileNotFoundError(placement_npz)
    params, placedb = _load_dreamplace(str(config))
    placedb.initialize(params)
    shift = (float(params.shift_factor[0]), float(params.shift_factor[1]))
    scale = float(params.scale_factor)
    native = RegionSet.from_json(str(regions_json))
    native.validate()
    scaled = scaled_region_set(native, shift, scale)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh),
           float(placedb.yh))
    if not np.allclose(np.asarray(scaled.die, dtype=np.float64),
                       np.asarray(die, dtype=np.float64), rtol=0, atol=1e-6):
        raise ValueError("regions.json die %s does not match the scaled "
                         "PlaceDB die %s" % (scaled.die, die))
    with np.load(placement_npz, allow_pickle=False) as data:
        node_x = np.asarray(data["node_x"], dtype=np.float64)
        node_y = np.asarray(data["node_y"], dtype=np.float64)
    movable = int(placedb.num_movable_nodes)
    if node_x.shape != node_y.shape or node_x.size < movable:
        raise ValueError("placement.npz holds %d positions, fewer than the "
                         "%d movable nodes" % (node_x.size, movable))
    os.makedirs(out_dir, exist_ok=True)
    paths = export_def(placedb, params, node_x, node_y, out_dir, scaled)
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in placedb.node_names[:movable]]
    lefs = [str(path) for path in params.lef_input]
    return {"out_def": paths["out_def"], "coord_json": paths["coord_json"],
            "netmap_json": paths["netmap_json"],
            "regions_json": paths["regions_json"],
            "movable_names": names, "lefs": lefs, "scaled_die": die,
            "shift_factor": shift, "scale_factor": scale,
            "num_movable": movable}


def repair_def_rows(export_dir, lefs, out_dir, binary, movable_names,
                    threads=16):
    """Mandatory OpenROAD DEF row repair (spec sec 8).

    Not optional and not a fallback: it changed 648,869 cell locations and
    2,120,322 orientations on mempool_group
    (docs/results/2026-09-15-route-gp-completion-audit.md:110-121). Everything
    downstream -- the route, the decode and the re-run evaluator -- uses the
    coordinates this returns.
    """
    from ioplace.route_eval.placement_openroad import legalize_export
    return legalize_export(export_dir, lefs, out_dir, binary, movable_names,
                           threads=threads)


def _reevaluate(config, arm_dir, regions_json, repaired_x, repaired_y,
                capacity_npz):
    """Re-run the fast evaluator on the repaired coordinates.

    spec sec 8 requires the evaluator to run "on the same row-repaired,
    oriented coordinates", so the placement numbers in the matrix describe the
    layout that was actually routed. Returns the same metric dict shape
    run_placement._pack_eval_metrics produces, prefixed `repaired_`.
    """
    from ioplace.drivers.run_placement import _evaluate_and_pack  # noqa: F401
    from ioplace.evaluator_gpu import GpuEvalContext
    from ioplace.netlist import netlist_from_placedb
    from ioplace.drivers.run_placement import _load_dreamplace, _pack_eval_metrics
    from ioplace.region_grid import RegionGrid
    from ioplace.regions import RegionSet
    from ioplace.artifacts import scaled_region_set

    params, placedb = _load_dreamplace(str(config))
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    scaled = scaled_region_set(RegionSet.from_json(str(regions_json)),
                               (float(params.shift_factor[0]),
                                float(params.shift_factor[1])),
                               float(params.scale_factor))
    rg = RegionGrid(scaled)
    node_x = np.asarray(placedb.node_x, dtype=np.float64).copy()
    node_y = np.asarray(placedb.node_y, dtype=np.float64).copy()
    movable = int(placedb.num_movable_nodes)
    node_x[:movable] = repaired_x
    node_y[:movable] = repaired_y
    ctx = GpuEvalContext(nl, rg, device="cuda")
    metrics = _pack_eval_metrics(ctx.evaluate(node_x, node_y))
    return {"repaired_" + key: value for key, value in metrics.items()}


def run_final_grt(case, arm, arm_dir, *, config, regions_json, capacity_npz,
                  binary, timeout=DEFAULT_GRT_TIMEOUT_S, threads=4,
                  congestion_iterations=50, repair_threads=16,
                  reevaluate=True):
    """The whole spec sec 8 recipe for one arm. Called exactly once per arm."""
    from ioplace.artifacts import file_sha256
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments, segments_digest
    from ioplace.regions import RegionSet
    from ioplace.route_eval.online_openroad import load_observation

    arm_dir = os.path.abspath(arm_dir)
    placement_npz = os.path.join(arm_dir, "placement.npz")
    if not os.path.isfile(placement_npz):
        raise FileNotFoundError("%s: placement.npz is required; run the "
                                "placement stage first" % (arm_dir,))
    grt_dir = os.path.join(arm_dir, "grt")
    if os.path.exists(grt_dir):
        raise RuntimeError("%s already routed; the router runs once per arm "
                           "(spec sec 0). Delete %s to redo it."
                           % (arm, grt_dir))

    started = time.perf_counter()
    export_dir = os.path.join(arm_dir, "export")
    exported = export_arm_def(config, placement_npz, regions_json, export_dir)
    t_export = time.perf_counter() - started

    started = time.perf_counter()
    repaired = repair_def_rows(export_dir, exported["lefs"],
                               os.path.join(arm_dir, "repair"), binary,
                               exported["movable_names"],
                               threads=repair_threads)
    t_repair = time.perf_counter() - started

    policy = {"congestion_iterations": int(congestion_iterations),
              "allow_congestion": True, "threads": int(threads),
              "signal_layers": GRT_POLICY["signal_layers"]}
    started = time.perf_counter()
    _run_openroad(repaired["legal_def_path"], exported["lefs"], grt_dir,
                  binary, timeout=timeout, **policy)
    t_grt = time.perf_counter() - started

    started = time.perf_counter()
    observation = load_observation(grt_dir, exported["coord_json"],
                                   exported["regions_json"],
                                   exported["netmap_json"])
    decoded = decode_segments(os.path.join(grt_dir, "segments.txt"))
    rg = RegionGrid(RegionSet.from_json(exported["regions_json"]))
    table = enumerate_segments(rg)
    capacity, capacity_source = _load_capacity(capacity_npz, rg, table)
    router = router_segment_table(rg, table, observation["net_polylines"])
    metrics = segment_metrics(router["demand"], capacity, table)
    t_decode = time.perf_counter() - started

    started = time.perf_counter()
    shift = exported["shift_factor"]
    scale = exported["scale_factor"]
    repaired_x = (np.asarray(repaired["x_dbu"], dtype=np.float64) - shift[0]) * scale
    repaired_y = (np.asarray(repaired["y_dbu"], dtype=np.float64) - shift[1]) * scale
    np.savez_compressed(os.path.join(arm_dir, "repaired_placement.npz"),
                        x_dbu=np.asarray(repaired["x_dbu"]),
                        y_dbu=np.asarray(repaired["y_dbu"]),
                        orientations=np.asarray(repaired["orientations"]),
                        node_x=repaired_x, node_y=repaired_y,
                        shift_factor=np.asarray(shift, dtype=np.float64),
                        scale_factor=np.float64(scale))
    repaired_metrics = (_reevaluate(config, arm_dir, regions_json, repaired_x,
                                    repaired_y, capacity_npz)
                        if reevaluate else {})
    t_eval = time.perf_counter() - started

    np.savez_compressed(os.path.join(arm_dir, "route_segments.npz"),
                        segment_demand_router=router["demand"],
                        segment_demand_router_layered=router["demand_layered"],
                        segment_capacity=capacity,
                        segment_util_router=np.where(
                            capacity > 0, router["demand"] /
                            np.where(capacity > 0, capacity, 1.0), 0.0))
    receipt = repaired["receipt"]
    hashes = {name: file_sha256(path) for name, path in (
        ("placement.npz", placement_npz),
        ("regions.json", os.path.abspath(str(regions_json))),
        ("capacity.npz", os.path.abspath(str(capacity_npz))),
        ("out.def", exported["out_def"]),
        ("legalized.def", repaired["legal_def_path"]),
        ("segments.txt", os.path.join(grt_dir, "segments.txt")),
        ("resources.json", os.path.join(grt_dir, "resources.json")),
        ("grt_receipt.json", os.path.join(grt_dir, "receipt.json")),
        ("repair_receipt.json", repaired["receipt_path"]))}
    record = build_route_record(
        case=case, arm=arm, status="ok", segment_metrics_out=metrics,
        decoded=decoded, observation=observation,
        demand_layered_total=int(router["demand_layered"].sum()),
        repair={"changed_location_count": receipt.get("changed_location_count"),
                "changed_orientation_count": receipt.get("changed_orientation_count"),
                "maximum_displacement_dbu": receipt.get("maximum_displacement_dbu"),
                "receipt_sha256": hashes["repair_receipt.json"]},
        timings={"t_export_s": t_export, "t_repair_s": t_repair,
                 "t_grt_s": t_grt, "t_decode_s": t_decode, "t_eval_s": t_eval},
        hashes=hashes, segments_digest_value=segments_digest(table),
        capacity_source=capacity_source, routing_policy=policy)
    record.update(repaired_metrics)
    with open(os.path.join(arm_dir, "route.json"), "w") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
    return record
```

- [ ] **Step 4: Run the fast tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v -m "not slow"`
Expected: PASS (12 tests).

- [ ] **Step 5: Run the OpenROAD round-trip on GCD**

```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
export IOPLACE_OPENROAD_TESTS=1
"$IOPLACE_PYTHON" -m pytest tests/test_campaign_final_grt.py -v -m slow
```
Expected: PASS (1 test), under two minutes. GCD is 508 movable nodes; if the repair step reports a nonzero `before_check_code`, that is normal — `legalize_export` only fails on a changed identity digest or a logged `[ERROR`.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/campaign/final_grt.py tests/test_campaign_final_grt.py
git commit -m "$(cat <<'MSG'
feat(campaign): the spec sec 8 per-arm recipe, one global route at the end

export_arm_def rebuilds the PlaceDB and writes out.def plus the three routing
sidecars from placement.npz, converting the native regions.json into the
scaled frame load_observation indexes. repair_def_rows is the mandatory
OpenROAD DEF row repair. run_final_grt sequences export -> repair -> one
bounded global_route -> decode -> per-segment crossings -> evaluator on the
repaired coordinates, refuses to route an arm twice, and writes route.json
and route_segments.npz with a SHA-256 for every file a number came from.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: The matrix table — `campaign/table.py`

One table per case, normalised to `ours = 1.000`, in Markdown and CSV, with every row carrying the hashes of the two files it was read from.

**Files:**
- Create: `src/ioplace/campaign/table.py`
- Test: `tests/test_campaign_table.py`

**Interfaces:**
- Consumes: `ioplace.campaign.arms.{ARMS, ALL_ARM_IDS}` (Task 1); `ioplace.artifacts.file_sha256` — from plan P-B Task 1, verify at pre-flight. Reads `result.json` (P-B `MAIN_FLOW_RESULT_FIELDS`), `route.json` (Task 4), and the `straddle_*` names from `ioplace.straddle.STRADDLE_SCALARS` — from plan P-F Task 3, verify at pre-flight.
- Produces:
  - `@dataclass(frozen=True) Metric(key, label, group, direction, fmt, source)` where `group in ("primary", "routing", "placement", "geometry", "cost")`, `direction in ("lower", "higher", "none")`, `source` is a dotted path such as `"route.num_over_capacity"` or `"result.region_area_balance.utilization_ratio"`
  - `METRICS: tuple` — the ordered metric list
  - `collect_arm(arm_dir, arm_id) -> dict` with keys `arm`, `label`, `values`, `hashes`, `status`
  - `normalise(rows, base="ours") -> rows` adding a `ratios` dict per row
  - `render_markdown(case, rows, *, notes=()) -> str`
  - `render_csv(rows) -> str`
  - `build(case, arms_dir, arm_ids, *, base="ours") -> (markdown, csv, rows)`

**Normalisation rules, decided here so no reader has to guess.**

1. The base row is `ours`. Every ratio is `value / base_value`.
2. `base_value == 0` → the ratio is `None` and renders as `n/a`, with a footnote naming the metric and stating that ours scored zero. A zero `num_over_capacity` for ours is a *good* outcome, and printing `inf` or `0.000` for the others would misrepresent it — the absolute table carries the comparison instead.
3. A missing value (an arm that never routed, the synthetic case's whole routing block) renders as an em dash in both tables and is excluded from the footnotes.
4. Direction is a property of the metric, printed once in the column header (`↓` = lower is better), never per cell. No cell is colour-coded, starred or annotated as a "win": spec §10 risk 3 requires arms (e)/(f) to be reported straight.
5. Every table is emitted twice: the normalised matrix and an absolute-value table below it. The normalised one is for reading; the absolute one is the record.

- [ ] **Step 1: Write the failing test**

Create `tests/test_campaign_table.py`:

```python
import json
import os

import numpy as np
import pytest

from ioplace.campaign.table import (METRICS, build, collect_arm, normalise,
                                    render_csv, render_markdown)


def _write_arm(root, arm_id, *, over=2, hpwl=100.0, util_ratio=1.4,
               route=True):
    directory = os.path.join(root, arm_id)
    os.makedirs(directory, exist_ok=True)
    result = {
        "mode": "main_flow", "schema_version": 1, "io_count": 1000,
        "ft_count": 200, "hard_lambda_sum": 900, "io_rg": 10.0, "ft_rg": 2.0,
        "hpwl_gp": hpwl, "hpwl_lg": hpwl * 1.01, "hpwl": hpwl * 1.01,
        "tree_wl": hpwl * 1.2, "runtime_s": 3600.0, "peak_mem_mb": 8192.0,
        "straddle_cells": 5, "straddle_area_fraction": 0.01,
        "straddle_pin_split_nets": 3,
        "region_area_balance": {"utilization_max": 0.7,
                                "utilization_min": 0.7 / util_ratio,
                                "utilization_ratio": util_ratio,
                                "cell_count_deviation": 0.2},
        "t_gp_soft": 600.0, "t_gp_fence": 2400.0, "t_lg": 300.0,
        "t_eval": 60.0, "fence_compliance": 1.0,
    }
    with open(os.path.join(directory, "result.json"), "w") as stream:
        json.dump(result, stream)
    if route:
        with open(os.path.join(directory, "route.json"), "w") as stream:
            json.dump({"schema_version": 1, "arm": arm_id, "status": "ok",
                       "num_over_capacity": over, "max_util": 1.0 + 0.1 * over,
                       "p99_util": 0.9, "total_overflow_tracks": 3.0 * over,
                       "segment_demand_router_total": 5000,
                       "native_total_overflow": 100 * over,
                       "routed_wirelength_dbu": 1.0e6, "via_count": 50000,
                       "grt_elapsed_s": 900.0,
                       "routing_policy": {"congestion_iterations": 50,
                                          "allow_congestion": True,
                                          "threads": 4,
                                          "signal_layers": "metal2-metal10"},
                       "sha256": {}}, stream)
    return directory


def test_every_metric_is_well_formed_and_the_primaries_come_first():
    keys = [m.key for m in METRICS]
    assert len(keys) == len(set(keys))
    primary = [m.key for m in METRICS if m.group == "primary"]
    assert primary == ["num_over_capacity", "total_overflow_tracks",
                       "max_util", "p99_util"]
    assert [m.group for m in METRICS][:4] == ["primary"] * 4
    for metric in METRICS:
        assert metric.direction in ("lower", "higher", "none")
        assert metric.source.split(".")[0] in ("route", "result")


def test_area_balance_is_reported_for_every_arm():
    """spec sec 8 and sec 10 risk 3: every arm reports region area balance."""
    keys = {m.key for m in METRICS}
    assert {"utilization_ratio", "cell_count_deviation"} <= keys


def test_collect_arm_reads_both_files_and_hashes_them(tmp_path):
    directory = _write_arm(str(tmp_path), "ours")
    row = collect_arm(directory, "ours")
    assert row["arm"] == "ours" and row["status"] == "ok"
    assert row["values"]["num_over_capacity"] == 2
    assert row["values"]["hpwl_lg"] == pytest.approx(101.0)
    assert row["values"]["utilization_ratio"] == pytest.approx(1.4)
    assert set(row["hashes"]) == {"result.json", "route.json"}
    assert all(len(v) == 64 for v in row["hashes"].values())


def test_collect_arm_without_a_route_marks_routing_missing(tmp_path):
    directory = _write_arm(str(tmp_path), "ours", route=False)
    row = collect_arm(directory, "ours")
    assert row["status"] == "placed_not_routed"
    assert row["values"]["num_over_capacity"] is None
    assert set(row["hashes"]) == {"result.json"}


def test_normalise_puts_ours_at_exactly_one(tmp_path):
    rows = [collect_arm(_write_arm(str(tmp_path), "ours", over=2, hpwl=100.0),
                        "ours"),
            collect_arm(_write_arm(str(tmp_path), "a", over=4, hpwl=110.0),
                        "a")]
    rows = normalise(rows, base="ours")
    ours = next(r for r in rows if r["arm"] == "ours")
    other = next(r for r in rows if r["arm"] == "a")
    assert ours["ratios"]["num_over_capacity"] == pytest.approx(1.0)
    assert ours["ratios"]["hpwl_gp"] == pytest.approx(1.0)
    assert other["ratios"]["num_over_capacity"] == pytest.approx(2.0)
    assert other["ratios"]["hpwl_gp"] == pytest.approx(1.1)


def test_normalise_reports_na_when_ours_scored_zero(tmp_path):
    rows = [collect_arm(_write_arm(str(tmp_path), "ours", over=0), "ours"),
            collect_arm(_write_arm(str(tmp_path), "a", over=4), "a")]
    rows = normalise(rows, base="ours")
    for row in rows:
        assert row["ratios"]["num_over_capacity"] is None
    assert "num_over_capacity" in rows[0]["zero_base_metrics"]


def test_normalise_rejects_a_missing_base(tmp_path):
    rows = [collect_arm(_write_arm(str(tmp_path), "a"), "a")]
    with pytest.raises(ValueError, match="base arm 'ours'"):
        normalise(rows, base="ours")


def test_markdown_carries_the_hashes_the_policy_and_the_risk_3_sentence(tmp_path):
    rows = normalise([collect_arm(_write_arm(str(tmp_path), "ours"), "ours"),
                      collect_arm(_write_arm(str(tmp_path), "e", over=9), "e")],
                     base="ours")
    text = render_markdown("mempool_group", rows)
    assert "ours = 1.000" in text
    assert "num_over_capacity" in text and "1.000" in text
    assert "metal2-metal10" in text and "-congestion_iterations 50" in text
    assert "-allow_congestion" in text
    assert "reported, not tuned away" in text
    assert "utilization_ratio" in text
    assert "## Absolute values" in text
    assert "## Provenance" in text
    for row in rows:
        for sha in row["hashes"].values():
            assert sha[:16] in text


def test_csv_has_one_row_per_arm_and_absolute_plus_ratio_columns(tmp_path):
    rows = normalise([collect_arm(_write_arm(str(tmp_path), "ours"), "ours"),
                      collect_arm(_write_arm(str(tmp_path), "a"), "a")],
                     base="ours")
    csv = render_csv(rows)
    header = csv.splitlines()[0].split(",")
    assert header[0] == "arm" and "num_over_capacity" in header
    assert "num_over_capacity_rel" in header
    assert len(csv.splitlines()) == 3


def test_build_reads_a_whole_case_directory(tmp_path):
    for arm_id in ("ours", "a", "e"):
        _write_arm(str(tmp_path), arm_id)
    markdown, csv, rows = build("mempool_group", str(tmp_path),
                                ["a", "ours", "e"])
    assert [r["arm"] for r in rows] == ["a", "ours", "e"]
    assert "mempool_group" in markdown and len(csv.splitlines()) == 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_table.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.campaign.table'`.

- [ ] **Step 3: Write `src/ioplace/campaign/table.py`**

```python
"""One table per case, normalised to ours = 1.000 (v2 design sec 8).

Reads each arm's result.json (P-B) and route.json (P-G Task 4) and emits a
Markdown document and a CSV. Pure stdlib plus numpy for the percentile-free
arithmetic: this module must stay importable in a bare CPU report process, so
it imports neither torch nor DREAMPlace.

Reporting rules, fixed here so no reader has to guess:
  * ours is the base; a missing base is an error, not a silent reordering.
  * a base value of 0 yields `n/a`, never `inf` and never 0.000, and the
    metric is named in a footnote. num_over_capacity = 0 for ours is a good
    outcome and must not be rendered as a divide-by-zero artefact.
  * a missing value is an em dash (an arm that did not route; the synthetic
    Bookshelf case, which has no LEF/DEF and therefore no routing block).
  * direction is in the column header, once. No cell is starred, coloured or
    annotated as a win: spec sec 10 risk 3 requires arms (e)/(f) to be
    reported straight.
"""
import csv as _csv
import io
import json
import os

from dataclasses import dataclass

#: spec sec 10 risk 3, verbatim. Every matrix document carries it.
RISK_3 = ("Arms (e)/(f) may lose badly. Same records: two-stage `io_count` "
          "178,594 vs flat 100,108. This gap is partly structural (fixed "
          "membership plus the partitioner's balance constraint), so it is "
          "reported, not tuned away.")


@dataclass(frozen=True)
class Metric(object):
    key: str
    label: str
    group: str
    direction: str
    fmt: str
    source: str


def _m(key, label, group, direction, fmt, source):
    if group not in ("primary", "routing", "placement", "geometry", "cost"):
        raise ValueError(group)
    if direction not in ("lower", "higher", "none"):
        raise ValueError(direction)
    return Metric(key, label, group, direction, fmt, source)


METRICS = (
    # --- primary: per-segment crossings vs capacity (spec sec 8) ---
    _m("num_over_capacity", "segments over capacity", "primary", "lower",
       "{:.0f}", "route.num_over_capacity"),
    _m("total_overflow_tracks", "overflow tracks", "primary", "lower",
       "{:.1f}", "route.total_overflow_tracks"),
    _m("max_util", "max segment utilisation", "primary", "lower", "{:.3f}",
       "route.max_util"),
    _m("p99_util", "p99 segment utilisation", "primary", "lower", "{:.3f}",
       "route.p99_util"),
    # --- secondary routing ---
    _m("native_total_overflow", "native total overflow", "routing", "lower",
       "{:.0f}", "route.native_total_overflow"),
    _m("routed_wirelength_dbu", "routed WL (DBU)", "routing", "lower",
       "{:.4g}", "route.routed_wirelength_dbu"),
    _m("via_count", "vias", "routing", "lower", "{:.0f}", "route.via_count"),
    _m("segment_demand_router_total", "routed boundary crossings", "routing",
       "lower", "{:.0f}", "route.segment_demand_router_total"),
    # --- placement quality ---
    _m("hpwl_gp", "HPWL (GP)", "placement", "lower", "{:.6g}",
       "result.hpwl_gp"),
    _m("hpwl_lg", "HPWL (LG)", "placement", "lower", "{:.6g}",
       "result.hpwl_lg"),
    _m("io_count", "io_count", "placement", "lower", "{:.0f}",
       "result.io_count"),
    _m("ft_count", "ft_count", "placement", "lower", "{:.0f}",
       "result.ft_count"),
    _m("hard_lambda_sum", "hard_lambda_sum", "placement", "lower", "{:.0f}",
       "result.hard_lambda_sum"),
    _m("io_rg", "io_rg", "placement", "lower", "{:.4g}", "result.io_rg"),
    _m("ft_rg", "ft_rg", "placement", "lower", "{:.4g}", "result.ft_rg"),
    # --- straddling (spec sec 7 / P-F) ---
    _m("straddle_cells", "straddle_cells", "geometry", "lower", "{:.0f}",
       "result.straddle_cells"),
    _m("straddle_area_fraction", "straddle_area_fraction", "geometry",
       "lower", "{:.5f}", "result.straddle_area_fraction"),
    _m("straddle_pin_split_nets", "straddle_pin_split_nets", "geometry",
       "lower", "{:.0f}", "result.straddle_pin_split_nets"),
    # --- region area balance: required for every arm (spec sec 8, sec 10 risk 3)
    _m("utilization_ratio", "region util max/min", "geometry", "lower",
       "{:.3f}", "result.region_area_balance.utilization_ratio"),
    _m("cell_count_deviation", "cell-count deviation", "geometry", "lower",
       "{:.4f}", "result.region_area_balance.cell_count_deviation"),
    _m("fence_compliance", "fence_compliance", "geometry", "higher",
       "{:.6f}", "result.fence_compliance"),
    # --- cost ---
    _m("runtime_s", "placement runtime (s)", "cost", "lower", "{:.0f}",
       "result.runtime_s"),
    _m("grt_elapsed_s", "GRT runtime (s)", "cost", "lower", "{:.0f}",
       "route.grt_elapsed_s"),
    _m("peak_mem_mb", "peak GPU memory (MB)", "cost", "lower", "{:.0f}",
       "result.peak_mem_mb"),
)

GROUP_TITLES = (("primary", "Primary: per-segment crossings vs capacity"),
                ("routing", "Routing (secondary)"),
                ("placement", "Placement quality"),
                ("geometry", "Geometry, straddling and area balance"),
                ("cost", "Cost"))


def _dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def collect_arm(arm_dir, arm_id):
    """Merge one arm's result.json and route.json into a row."""
    from ioplace.artifacts import file_sha256
    from ioplace.campaign.arms import ARMS

    arm_dir = os.path.abspath(arm_dir)
    result_path = os.path.join(arm_dir, "result.json")
    route_path = os.path.join(arm_dir, "route.json")
    if not os.path.isfile(result_path):
        return {"arm": arm_id, "label": ARMS[arm_id].label if arm_id in ARMS
                else arm_id, "values": {m.key: None for m in METRICS},
                "hashes": {}, "status": "missing", "dir": arm_dir}
    payload = {"result": json.load(open(result_path))}
    hashes = {"result.json": file_sha256(result_path)}
    status = "placed_not_routed"
    if os.path.isfile(route_path):
        payload["route"] = json.load(open(route_path))
        hashes["route.json"] = file_sha256(route_path)
        status = payload["route"].get("status", "ok")
    values = {}
    for metric in METRICS:
        head = metric.source.split(".", 1)[0]
        values[metric.key] = (_dig(payload, metric.source)
                              if head in payload else None)
    return {"arm": arm_id,
            "label": ARMS[arm_id].label if arm_id in ARMS else arm_id,
            "values": values, "hashes": hashes, "status": status,
            "dir": arm_dir,
            "routing_policy": _dig(payload, "route.routing_policy")}


def normalise(rows, base="ours"):
    """Add a `ratios` dict per row, with the base arm at exactly 1.000."""
    base_row = next((row for row in rows if row["arm"] == base), None)
    if base_row is None:
        raise ValueError("base arm %r is not in this matrix; ours is the "
                         "normalisation base (spec sec 8)" % (base,))
    zero_base = [metric.key for metric in METRICS
                 if isinstance(base_row["values"].get(metric.key),
                               (int, float))
                 and float(base_row["values"][metric.key]) == 0.0]
    for row in rows:
        ratios = {}
        for metric in METRICS:
            mine = row["values"].get(metric.key)
            theirs = base_row["values"].get(metric.key)
            if (mine is None or theirs is None
                    or float(theirs) == 0.0):
                ratios[metric.key] = None
            else:
                ratios[metric.key] = float(mine) / float(theirs)
        row["ratios"] = ratios
        row["zero_base_metrics"] = list(zero_base)
        row["base"] = base
    return rows


def _cell(value, fmt):
    if value is None:
        return "—"
    return fmt.format(float(value))


def _ratio_cell(value):
    return "n/a" if value is None else "%.3f" % (value,)


def render_markdown(case, rows, *, notes=()):
    base = rows[0].get("base", "ours") if rows else "ours"
    policies = {json.dumps(row.get("routing_policy"), sort_keys=True)
                for row in rows if row.get("routing_policy")}
    out = []
    out.append("# v2 matrix — %s" % (case,))
    out.append("")
    out.append("Normalised to `%s = 1.000`. Lower is better where the header "
               "says `↓`, higher where it says `↑`. `n/a` means the base arm "
               "scored exactly zero on that metric — read the absolute table "
               "instead. `—` means the value was never measured." % (base,))
    out.append("")
    out.append("Routing policy (spec §8, identical for every routed arm): "
               "`set_routing_layers -signal metal2-metal10`, "
               "`-congestion_iterations 50`, `-allow_congestion`, 4 GRT "
               "threads, one call per arm at the very end.")
    if len(policies) > 1:
        out.append("")
        out.append("**WARNING: the arms below were not all routed under the "
                   "same policy; they are not comparable.**")
    out.append("")
    out.append("> %s" % (RISK_3,))
    out.append("")
    for group, title in GROUP_TITLES:
        metrics = [m for m in METRICS if m.group == group]
        if not metrics:
            continue
        out.append("## %s" % (title,))
        out.append("")
        arrows = {"lower": " ↓", "higher": " ↑", "none": ""}
        out.append("| arm | " + " | ".join(
            m.label + arrows[m.direction] for m in metrics) + " |")
        out.append("|---|" + "---|" * len(metrics))
        for row in rows:
            out.append("| `%s` | " % (row["arm"],) + " | ".join(
                _ratio_cell(row["ratios"].get(m.key)) for m in metrics) + " |")
        out.append("")
    zero_base = rows[0].get("zero_base_metrics", []) if rows else []
    if zero_base:
        out.append("Footnote: `%s` scored exactly 0 on %s, so those columns "
                   "are `n/a`; a zero there is a good outcome for the base "
                   "arm, not a missing measurement."
                   % (base, ", ".join("`%s`" % (k,) for k in zero_base)))
        out.append("")
    out.append("## Absolute values")
    out.append("")
    out.append("| arm | label | status | " + " | ".join(
        m.label for m in METRICS) + " |")
    out.append("|---|---|---|" + "---|" * len(METRICS))
    for row in rows:
        out.append("| `%s` | %s | %s | " % (row["arm"], row["label"],
                                            row["status"]) + " | ".join(
            _cell(row["values"].get(m.key), m.fmt) for m in METRICS) + " |")
    out.append("")
    out.append("## Provenance")
    out.append("")
    out.append("| arm | file | sha256 |")
    out.append("|---|---|---|")
    for row in rows:
        for name, sha in sorted(row["hashes"].items()):
            out.append("| `%s` | `%s` | `%s` |" % (row["arm"], name, sha))
    out.append("")
    for note in notes:
        out.append("- %s" % (note,))
    if notes:
        out.append("")
    return "\n".join(out)


def render_csv(rows):
    buffer = io.StringIO()
    writer = _csv.writer(buffer, lineterminator="\n")
    header = ["arm", "label", "status"]
    for metric in METRICS:
        header += [metric.key, metric.key + "_rel"]
    writer.writerow(header)
    for row in rows:
        line = [row["arm"], row["label"], row["status"]]
        for metric in METRICS:
            value = row["values"].get(metric.key)
            ratio = row.get("ratios", {}).get(metric.key)
            line += ["" if value is None else value,
                     "" if ratio is None else "%.6f" % (ratio,)]
        writer.writerow(line)
    return buffer.getvalue()


def build(case, arms_dir, arm_ids, *, base="ours", notes=()):
    rows = [collect_arm(os.path.join(arms_dir, arm_id), arm_id)
            for arm_id in arm_ids]
    rows = normalise(rows, base=base)
    return render_markdown(case, rows, notes=notes), render_csv(rows), rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_table.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Check the `straddle_*` and `region_area_balance` sources against the landed schemas**

```bash
source src/scripts/env.sh
"$IOPLACE_PYTHON" - <<'PY'
from ioplace.artifacts import MAIN_FLOW_RESULT_FIELDS
from ioplace.campaign.table import METRICS
fields = set(MAIN_FLOW_RESULT_FIELDS)
missing = [m.source for m in METRICS
           if m.source.startswith("result.")
           and m.source.split(".")[1] not in fields]
print("missing from MAIN_FLOW_RESULT_FIELDS:", missing)
PY
```
Expected: `missing from MAIN_FLOW_RESULT_FIELDS: []`. If `straddle_cells`, `straddle_area_fraction` or `straddle_pin_split_nets` are reported missing, P-F has not yet added them to `result.json`; leave the metrics in `METRICS` (they render as `—`) and note the gap in the commit message — the table must not silently drop a spec §8 metric.

- [ ] **Step 6: Commit**

```bash
git add src/ioplace/campaign/table.py tests/test_campaign_table.py
git commit -m "$(cat <<'MSG'
feat(campaign): the v2 matrix table, normalised to ours = 1.000

Merges each arm's result.json and route.json into one Markdown + CSV matrix,
primary per-segment crossings first, then the spec sec 8 secondary list, the
P-F straddle set, region area balance for every arm, and cost. A zero base
value renders n/a with a named footnote rather than inf; a missing value
renders as an em dash; direction lives in the header, never in a cell. Every
document carries the routing policy, a sha256 per source file and spec sec 10
risk 3's "reported, not tuned away" sentence verbatim.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 7: The campaign runner — `src/scripts/run_v2_campaign.py`

Sequences producer → placement → GRT → table for a list of arms, resumably, on one GPU, never running two global routes at once.

**Files:**
- Create: `src/scripts/run_v2_campaign.py`
- Test: `tests/test_run_v2_campaign.py` (append)

**Interfaces:**
- Consumes: `ioplace.campaign.arms.{ARMS, ALL_ARM_IDS, producer_argv, main_flow_argv, stage_fence_only_arm}` (Task 1); `ioplace.campaign.final_grt.{run_final_grt, DEFAULT_GRT_TIMEOUT_S}` (Tasks 4–5); `ioplace.campaign.table.build` (Task 6).
- Produces:
  - `STEPS = ("producer", "place", "grt")`
  - `grt_lock(path)` — a context manager holding an exclusive lock file
  - `plan_case(args) -> list` of `{"step", "arm", "argv", "cwd", "env"}` records — what `--dry-run` prints and what the tests assert on
  - `load_state(arm_dir) / record_step(arm_dir, step, payload)`
  - `main(argv=None) -> int`

**Resumability.** Each arm directory carries `state.json`: `{"steps": {"<step>": {"finished_at": ..., "inputs_sha256": {...}}}}`. A step is skipped when it is recorded **and** every recorded input hash still matches. `--force <step>` (repeatable) clears that step and everything after it. This matters because a group-scale arm is a 3–8 hour unit; a crash in arm 7 of 9 must not re-run arms 1–6.

**One route at a time.** `global_route` with 4 threads plus a concurrent DREAMPlace run is fine for the GPU but ruins the routing runtime numbers, which are a reported metric. `grt_lock` takes an `O_CREAT|O_EXCL` lock file holding the pid; a lock whose pid is dead is reclaimed with a logged warning. The runner is a sequential loop, so the lock is a guard against a *second operator*, not against itself.

**GPU hygiene.** `--gpu N` sets `CUDA_VISIBLE_DEVICES=N` in every child's environment. `--require-free-gb` (default 30) queries `nvidia-smi` and refuses to start if the chosen GPU has less free memory than that, because a group fence GP is ~4× the flat placement's footprint (spec §10 risk 2) and an OOM 90 minutes in wastes the whole arm.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_v2_campaign.py`:

```python
import json
import os

import pytest


def _module():
    import importlib
    return importlib.import_module("src.scripts.run_v2_campaign")


def _args(tmp_path, **overrides):
    module = _module()
    argv = ["--case", "mempool_group",
            "--config", "benchmarks/ispd25/h100/mempool_group.json",
            "--arms", overrides.pop("arms", "a,b,c,ours,e,f"),
            "--gpu", "3", "--out", str(tmp_path / "run"),
            "--capacity", str(tmp_path / "cap.npz")]
    for key, value in overrides.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return module.build_parser().parse_args(argv)


def test_parser_defaults_match_the_protocol(tmp_path):
    args = _args(tmp_path)
    assert args.k == 16
    assert args.congestion_iterations == 50
    assert args.grt_threads == 4
    assert args.grt_timeout_s == 21600.0
    assert args.require_free_gb == 30.0
    assert args.no_grt is False
    assert args.steps == "all"


def test_parser_rejects_an_unknown_arm(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, arms="a,nope")


def test_plan_runs_the_producers_once_then_every_arm_in_order(tmp_path):
    module = _module()
    plan = module.plan_case(_args(tmp_path))
    steps = [(p["step"], p["arm"]) for p in plan]
    assert steps[:2] == [("producer", "producer_b64"),
                         ("producer", "producer_b32")]
    # each arm's place is immediately followed by its own grt
    body = steps[2:]
    assert body == [("place", "a"), ("grt", "a"), ("place", "b"), ("grt", "b"),
                    ("place", "c"), ("grt", "c"), ("place", "ours"),
                    ("grt", "ours"), ("place", "e"), ("grt", "e"),
                    ("place", "f"), ("grt", "f")]


def test_plan_skips_the_32_bin_producer_when_no_arm_needs_it(tmp_path):
    module = _module()
    plan = module.plan_case(_args(tmp_path, arms="a,ours"))
    assert [p["arm"] for p in plan if p["step"] == "producer"] == [
        "producer_b64"]


def test_plan_puts_cuda_visible_devices_in_every_child_env(tmp_path):
    module = _module()
    for entry in module.plan_case(_args(tmp_path)):
        assert entry["env"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_plan_stages_fence_only_arms_before_their_placement(tmp_path):
    module = _module()
    plan = module.plan_case(_args(tmp_path, arms="e,f"))
    for arm in ("e", "f"):
        place = next(p for p in plan if p["step"] == "place" and p["arm"] == arm)
        assert place["stage_fence_only"] is True
        assert "--phase" in place["argv"]
        assert place["argv"][place["argv"].index("--phase") + 1] == "fence"
    place_ours = [p for p in plan if p["step"] == "place" and p["arm"] == "a"]
    assert place_ours == []


def test_no_grt_drops_every_route_step(tmp_path):
    module = _module()
    args = _args(tmp_path, arms="ours")
    args.no_grt = True
    plan = module.plan_case(args)
    assert [p["step"] for p in plan] == ["producer", "place"]


def test_state_round_trips_and_skips_a_finished_step(tmp_path):
    module = _module()
    arm_dir = tmp_path / "arms" / "ours"
    arm_dir.mkdir(parents=True)
    probe = arm_dir / "in.txt"
    probe.write_text("x")
    module.record_step(str(arm_dir), "place", {"in.txt": module.sha256(str(probe))})
    assert module.step_is_current(str(arm_dir), "place") is True
    probe.write_text("y")
    assert module.step_is_current(str(arm_dir), "place") is False


def test_grt_lock_is_exclusive(tmp_path):
    module = _module()
    path = str(tmp_path / ".grt.lock")
    with module.grt_lock(path):
        with pytest.raises(RuntimeError, match="another global route"):
            with module.grt_lock(path):
                pass
    # released on exit
    with module.grt_lock(path):
        pass


def test_grt_lock_reclaims_a_dead_pid(tmp_path, capsys):
    module = _module()
    path = str(tmp_path / ".grt.lock")
    with open(path, "w") as stream:
        json.dump({"pid": 999999999, "started_at": 0.0}, stream)
    with module.grt_lock(path):
        pass
    assert "stale" in capsys.readouterr().out


def test_dry_run_prints_every_command_and_touches_nothing(tmp_path, capsys):
    module = _module()
    out = tmp_path / "run"
    code = module.main(["--case", "mempool_group", "--config", "c.json",
                        "--arms", "ours", "--gpu", "3", "--out", str(out),
                        "--capacity", "cap.npz", "--dry-run"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "run_region_producer" in printed
    assert "run_main_flow" in printed
    assert "final_grt" in printed
    assert not out.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_run_v2_campaign.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.scripts.run_v2_campaign'`.

- [ ] **Step 3: Write `src/scripts/run_v2_campaign.py`**

```python
"""Run the v2 experiment matrix for one case: producer, arms, one GRT each.

v2 design sec 8. Sequential by construction, resumable per arm per step, and
holding a lock so two global routes can never overlap -- routing runtime is a
reported metric and a concurrent router would corrupt it.

  source src/scripts/env.sh && source src/scripts/openroad_env.sh
  "$IOPLACE_PYTHON" src/scripts/run_v2_campaign.py \
      --case mempool_group \
      --config benchmarks/ispd25/h100/mempool_group.json \
      --arms a,b,c,ours,e,f --gpu 3 \
      --capacity results/v2_capacity/mempool_group/capacity.npz \
      --out results/v2_matrix_20260919/mempool_group

Add --dry-run first, always: it prints every command it would run and writes
nothing.
"""
import argparse
import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

STEPS = ("producer", "place", "grt")
PRODUCER_DIRS = {"producer_b64": 64, "producer_b32": 32}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(arm_dir):
    path = os.path.join(arm_dir, "state.json")
    if not os.path.isfile(path):
        return {"steps": {}}
    with open(path) as stream:
        return json.load(stream)


def record_step(arm_dir, step, inputs_sha256):
    state = load_state(arm_dir)
    state.setdefault("steps", {})[step] = {
        "finished_at": time.time(), "inputs_sha256": dict(inputs_sha256)}
    os.makedirs(arm_dir, exist_ok=True)
    with open(os.path.join(arm_dir, "state.json"), "w") as stream:
        json.dump(state, stream, indent=1, sort_keys=True)


def step_is_current(arm_dir, step):
    """True when the step finished and every recorded input still hashes the
    same. A changed config, regions.json or capacity.npz invalidates it."""
    entry = load_state(arm_dir).get("steps", {}).get(step)
    if entry is None:
        return False
    for path, recorded in entry.get("inputs_sha256", {}).items():
        if not os.path.isfile(path) or sha256(path) != recorded:
            return False
    return True


def clear_step(arm_dir, step):
    state = load_state(arm_dir)
    order = list(STEPS)
    if step in order:
        for later in order[order.index(step):]:
            state.get("steps", {}).pop(later, None)
    path = os.path.join(arm_dir, "state.json")
    if os.path.isdir(arm_dir):
        with open(path, "w") as stream:
            json.dump(state, stream, indent=1, sort_keys=True)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


@contextlib.contextmanager
def grt_lock(path):
    """Exclusive lock around one global_route. Reclaims a dead holder."""
    if os.path.exists(path):
        try:
            with open(path) as stream:
                holder = json.load(stream)
        except (ValueError, OSError):
            holder = {}
        if _pid_alive(holder.get("pid", -1)):
            raise RuntimeError("another global route is running (pid %s, "
                               "since %s); one at a time"
                               % (holder.get("pid"), holder.get("started_at")))
        print("reclaiming stale GRT lock held by pid %s" % (holder.get("pid"),))
        os.unlink(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(handle, json.dumps({"pid": os.getpid(),
                                     "started_at": time.time()}).encode())
        os.close(handle)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _child_env(gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # mtkahypar's own thread pool fights the GPU process for cores and is not
    # deterministic above one thread; the producer is the only consumer.
    env.setdefault("IOPLACE_MTKAHYPAR_THREADS", "1")
    return env


def plan_case(args):
    """Every command this campaign would run, in order. Pure -- no side
    effects, no filesystem writes -- so --dry-run and the tests share it."""
    from ioplace.campaign.arms import (ARMS, GEOMETRY_PRODUCER, PRODUCER_BINS,
                                       main_flow_argv, producer_argv)

    out = os.path.abspath(args.out)
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    env = _child_env(args.gpu)
    producer_dirs = {name: os.path.join(out, name) for name in PRODUCER_DIRS}
    needed = []
    for arm_id in arms:
        name = GEOMETRY_PRODUCER[ARMS[arm_id].geometry]
        if name not in needed:
            needed.append(name)
    needed.sort(key=lambda name: 0 if name == "producer_b64" else 1)

    plan = []
    for name in needed:
        plan.append({
            "step": "producer", "arm": name,
            "argv": [sys.executable, "-m",
                     "ioplace.drivers.run_region_producer"]
                    + producer_argv(args.config, producer_dirs[name],
                                    extract_bins=PRODUCER_BINS[name],
                                    k=args.k, seed=args.seed),
            "cwd": os.getcwd(), "env": env, "out_dir": producer_dirs[name],
            "stage_fence_only": False})
    for arm_id in arms:
        spec = ARMS[arm_id]
        arm_dir = os.path.join(out, "arms", arm_id)
        plan.append({
            "step": "place", "arm": arm_id,
            "argv": [sys.executable, "-m", "ioplace.drivers.run_main_flow"]
                    + main_flow_argv(spec, config=args.config,
                                     out_dir=arm_dir,
                                     producer_dirs=producer_dirs,
                                     capacity_npz=args.capacity, k=args.k,
                                     seed=args.seed,
                                     allow_small_k=args.allow_small_k),
            "cwd": os.getcwd(), "env": env, "out_dir": arm_dir,
            "stage_fence_only": spec.driver == "fence_only",
            "producer_dir": producer_dirs[GEOMETRY_PRODUCER[spec.geometry]]})
        if args.no_grt:
            continue
        plan.append({
            "step": "grt", "arm": arm_id,
            "argv": ["<in-process>", "ioplace.campaign.final_grt."
                     "run_final_grt", args.case, arm_id, arm_dir],
            "cwd": os.getcwd(), "env": env, "out_dir": arm_dir,
            "stage_fence_only": False,
            "regions_json": (os.path.join(arm_dir, "regions.json")
                             if spec.geometry != "grid" else None)})
    return plan


def _check_gpu(gpu, require_free_gb):
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        print("could not query nvidia-smi (%s); continuing" % (error,))
        return
    for line in output.strip().splitlines():
        index, used, total = [part.strip() for part in line.split(",")]
        if int(index) != int(gpu):
            continue
        free = (int(total) - int(used)) / 1024.0
        print("GPU %s: %.1f GiB free of %.1f GiB"
              % (gpu, free, int(total) / 1024.0))
        if free < require_free_gb:
            raise RuntimeError(
                "GPU %s has %.1f GiB free, below --require-free-gb %.1f; a "
                "group fence GP is about 4x the flat footprint (spec sec 10 "
                "risk 2)" % (gpu, free, require_free_gb))
        return
    raise RuntimeError("GPU %s not visible to nvidia-smi" % (gpu,))


def _run_child(entry):
    print("$ " + " ".join(shlex.quote(part) for part in entry["argv"]))
    completed = subprocess.run(entry["argv"], cwd=entry["cwd"],
                               env=entry["env"])
    if completed.returncode:
        raise RuntimeError("%s/%s failed with %d"
                           % (entry["step"], entry["arm"],
                              completed.returncode))


def _regions_for(entry, args, out):
    """The regions.json the GRT step must use: the arm's own copy if the
    driver wrote one, else the producer's, else the builtin grid the driver
    materialised into the arm directory."""
    candidate = os.path.join(entry["out_dir"], "regions.json")
    if os.path.isfile(candidate):
        return candidate
    if entry.get("regions_json") and os.path.isfile(entry["regions_json"]):
        return entry["regions_json"]
    raise FileNotFoundError(
        "%s has no regions.json; run_main_flow must persist the geometry it "
        "placed against before the router can map crossings onto it"
        % (entry["out_dir"],))


def build_parser():
    from ioplace.campaign.arms import ALL_ARM_IDS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default=",".join(ALL_ARM_IDS))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--capacity", default=None,
                        help="capacity.npz from ioplace.capacity.extract; "
                             "required by every arm with the capacity term")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-small-k", action="store_true",
                        help="only the GCD end-to-end test; spec sec 0 fixes K=16")
    parser.add_argument("--steps", default="all",
                        choices=("all", "producer", "place", "grt", "table"))
    parser.add_argument("--force", action="append", default=[],
                        choices=list(STEPS))
    parser.add_argument("--no-grt", action="store_true",
                        help="placement only; the synthetic Bookshelf case "
                             "has no LEF/DEF and cannot be routed (G-5)")
    parser.add_argument("--congestion-iterations", type=int, default=50)
    parser.add_argument("--grt-threads", type=int, default=4)
    parser.add_argument("--grt-timeout-s", type=float, default=21600.0)
    parser.add_argument("--repair-threads", type=int, default=16)
    parser.add_argument("--require-free-gb", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    from ioplace.campaign.arms import ALL_ARM_IDS, stage_fence_only_arm
    from ioplace.campaign.final_grt import run_final_grt
    from ioplace.campaign.table import build as build_table

    parser = build_parser()
    args = parser.parse_args(argv)
    unknown = [name for name in args.arms.split(",")
               if name.strip() and name.strip() not in ALL_ARM_IDS]
    if unknown:
        parser.error("unknown arm(s): %s" % (", ".join(unknown),))
    plan = plan_case(args)
    if args.dry_run:
        for entry in plan:
            print("[%s %s] %s" % (entry["step"], entry["arm"],
                                  " ".join(shlex.quote(p)
                                           for p in entry["argv"])))
        return 0

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    _check_gpu(args.gpu, args.require_free_gb)
    binary = os.environ.get("OPENROAD_BIN")
    if not args.no_grt and (not binary or not os.access(binary, os.X_OK)):
        parser.error("OPENROAD_BIN is unset or not executable; "
                     "source src/scripts/openroad_env.sh")
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    with open(os.path.join(out, "campaign.json"), "w") as stream:
        json.dump({"case": args.case, "config": os.path.abspath(args.config),
                   "arms": arms, "k": args.k, "seed": args.seed,
                   "gpu": args.gpu, "no_grt": bool(args.no_grt),
                   "capacity": (os.path.abspath(args.capacity)
                                if args.capacity else None),
                   "congestion_iterations": args.congestion_iterations,
                   "grt_threads": args.grt_threads,
                   "grt_timeout_s": args.grt_timeout_s,
                   "started_at": time.time(),
                   "command": sys.argv}, stream, indent=1, sort_keys=True)

    for step in args.force:
        for entry in plan:
            clear_step(entry["out_dir"], step)

    for entry in plan:
        if args.steps not in ("all",) and args.steps != entry["step"]:
            continue
        directory = entry["out_dir"]
        if step_is_current(directory, entry["step"]):
            print("skip %s/%s (already current)" % (entry["step"],
                                                    entry["arm"]))
            continue
        os.makedirs(directory, exist_ok=True)
        if entry["step"] == "producer":
            _run_child(entry)
            record_step(directory, "producer",
                        {os.path.abspath(args.config): sha256(args.config)})
        elif entry["step"] == "place":
            if entry["stage_fence_only"]:
                staged = stage_fence_only_arm(directory,
                                              entry["producer_dir"], k=args.k)
                print("staged fence-only arm %s: %s"
                      % (entry["arm"], staged["staged"]))
            _run_child(entry)
            inputs = {os.path.abspath(args.config): sha256(args.config)}
            if args.capacity and os.path.isfile(args.capacity):
                inputs[os.path.abspath(args.capacity)] = sha256(args.capacity)
            record_step(directory, "place", inputs)
        else:
            regions = _regions_for(entry, args, out)
            with grt_lock(os.path.join(out, ".grt.lock")):
                record = run_final_grt(
                    args.case, entry["arm"], directory, config=args.config,
                    regions_json=regions, capacity_npz=args.capacity,
                    binary=binary, timeout=args.grt_timeout_s,
                    threads=args.grt_threads,
                    congestion_iterations=args.congestion_iterations,
                    repair_threads=args.repair_threads)
            print("routed %s: num_over_capacity=%s max_util=%.3f in %.0f s"
                  % (entry["arm"], record["num_over_capacity"],
                     record["max_util"], record["t_grt_s"]))
            record_step(directory, "grt",
                        {os.path.join(directory, "placement.npz"):
                         sha256(os.path.join(directory, "placement.npz"))})

    markdown, csv, _rows = build_table(args.case, os.path.join(out, "arms"),
                                       arms)
    with open(os.path.join(out, "matrix.md"), "w") as stream:
        stream.write(markdown)
    with open(os.path.join(out, "matrix.csv"), "w") as stream:
        stream.write(csv)
    print("wrote %s and %s" % (os.path.join(out, "matrix.md"),
                               os.path.join(out, "matrix.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Two things the implementer must resolve rather than paper over:

1. `_regions_for` assumes `run_main_flow` persists the geometry it placed against as `<arm>/regions.json`. P-B's artefact list (Global Constraints, §1 table) names `regions.json` as an arm-directory artefact, so it should be there for every arm including the grid ones. Verify it with `ls results/.../arms/a/regions.json` after the first placement; if grid arms do not get one, add a two-line copy in `main()` that writes `get_regions_for(die, k, "grid", seed)` into the arm directory *before* the GRT step, using the die recorded in the arm's `result.json` — and say so in the commit message. Do not let the router map crossings onto a geometry the placer did not use.
2. `run_final_grt` runs in-process, so a router crash takes the campaign down with it. That is deliberate: the per-arm `state.json` makes a restart cheap, and a subprocess wrapper would hide the Python traceback that says *why* the decode failed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_run_v2_campaign.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Dry-run the real group campaign**

```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
"$IOPLACE_PYTHON" src/scripts/run_v2_campaign.py \
  --case mempool_group \
  --config benchmarks/ispd25/h100/mempool_group.json \
  --arms a,b,c,ours,e,f,ours__polB,ours__nocap,ours__nopseudo \
  --gpu 3 --capacity results/v2_capacity/mempool_group/capacity.npz \
  --out results/v2_matrix_20260919/mempool_group --dry-run
```
Expected: 2 producer lines followed by 18 alternating place/grt lines, each `run_main_flow` line carrying the flags Task 1's table prescribes for that arm. Read them; a wrong flag here is a wasted day of GPU later.

- [ ] **Step 6: Run the whole fast suite**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow" -q`
Expected: PASS, with the pre-existing count plus this plan's new tests.

- [ ] **Step 7: Commit**

```bash
git add src/scripts/run_v2_campaign.py tests/test_run_v2_campaign.py
git commit -m "$(cat <<'MSG'
feat(campaign): the v2 matrix runner -- producer, arms, one GRT each

Sequences producer -> placement -> one global route -> table for a list of
arms on one GPU. Resumable per arm per step through state.json with input
hashing, so a crash in arm 7 of 9 does not re-run arms 1-6. An exclusive lock
guarantees two global routes never overlap, because routing runtime is a
reported metric. --dry-run prints every command and writes nothing; --no-grt
covers the synthetic Bookshelf case, which has no LEF/DEF to route (G-5).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 8: End-to-end on GCD (slow) — spec §9's small case

The whole protocol on the smallest real LEF/DEF case on this host: producer → main flow → fence LG → evaluator → one GRT → table, for three arms, at K=4.

**Files:**
- Create: `tests/test_v2_campaign_gcd.py`
- Test: itself

**Interfaces:**
- Consumes: everything from Tasks 1–7; `results/route_feedback_20260914/gcd.json` (508 movable nodes, 168 terminals, 579 nets, LEF/DEF under `third_party/OpenROAD/src/grt/test/`); `ioplace.capacity.extract` for a GCD `capacity.npz` — from plan P-D Task 3, verify at pre-flight.
- Produces: no product code. This task's deliverable is the passing test and the artefacts it leaves under `results/v2_matrix_gcd_smoke/`.

**Why K=4 and three arms.** GCD has 508 movable cells; K=16 would give ~32 cells per region and the producer's density-argmax extraction would be meaningless. P-B's own end-to-end test uses K=4 for the same reason. Three arms — `c` (grid + seed), `ours` (producer + seed), `e` (fence-only, 32²) — cover all three placement paths this plan drives: the grid geometry branch, the producer geometry branch and the fence-only staging branch. Running all nine would triple the runtime for no extra coverage of *this* plan's code.

**Gating.** `@pytest.mark.slow`, plus a skip unless `IOPLACE_OPENROAD_TESTS=1`, `$OPENROAD_BIN` is executable and CUDA is available. It is run by hand, not in CI.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v2_campaign_gcd.py`:

```python
"""spec sec 9's end-to-end small case, on GCD at K=4.

producer -> main flow -> fence LG -> evaluator -> one GRT -> matrix, for the
grid, producer and fence-only paths. Run by hand:

  source src/scripts/env.sh && source src/scripts/openroad_env.sh
  export IOPLACE_OPENROAD_TESTS=1 CUDA_VISIBLE_DEVICES=3
  "$IOPLACE_PYTHON" -m pytest tests/test_v2_campaign_gcd.py -v -s
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

CONFIG = "results/route_feedback_20260914/gcd.json"
ARMS = "c,ours,e"


def _cuda():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("IOPLACE_OPENROAD_TESTS") != "1",
                       reason="needs IOPLACE_OPENROAD_TESTS=1"),
    pytest.mark.skipif(not os.access(os.environ.get("OPENROAD_BIN", ""),
                                     os.X_OK),
                       reason="needs an executable OPENROAD_BIN"),
    pytest.mark.skipif(not _cuda(), reason="needs CUDA"),
]


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2_gcd")
    capacity = str(out / "capacity.npz")
    # One capacity extraction for the case, exactly as the real campaign does.
    subprocess.run([sys.executable, "-m", "ioplace.capacity.extract",
                    "--config", CONFIG, "--k", "4", "--rtype", "grid",
                    "--out", capacity, "--openroad",
                    os.environ["OPENROAD_BIN"]], check=True)
    code = subprocess.run(
        [sys.executable, "src/scripts/run_v2_campaign.py",
         "--case", "gcd", "--config", CONFIG, "--arms", ARMS,
         "--gpu", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
         "--out", str(out / "run"), "--capacity", capacity,
         "--k", "4", "--allow-small-k", "--require-free-gb", "1",
         "--grt-timeout-s", "1800"], check=False).returncode
    assert code == 0
    return out / "run"


def test_every_arm_produced_a_placement_and_exactly_one_route(campaign):
    for arm in ARMS.split(","):
        arm_dir = campaign / "arms" / arm
        for name in ("result.json", "placement.npz", "evaluation.npz",
                     "route.json", "route_segments.npz",
                     "repaired_placement.npz"):
            assert (arm_dir / name).is_file(), "%s/%s" % (arm, name)
        # exactly one GRT directory, and it holds one segments.txt
        assert (arm_dir / "grt" / "segments.txt").is_file()
        assert len(list((arm_dir / "grt").glob("segments*.txt"))) == 1


def test_the_routing_policy_is_identical_for_every_arm(campaign):
    policies = []
    for arm in ARMS.split(","):
        route = json.load(open(campaign / "arms" / arm / "route.json"))
        policies.append(route["routing_policy"])
    assert all(p == policies[0] for p in policies)
    assert policies[0]["congestion_iterations"] == 50
    assert policies[0]["allow_congestion"] is True
    assert policies[0]["threads"] == 4
    assert policies[0]["signal_layers"] == "metal2-metal10"


def test_row_repair_ran_and_its_receipt_is_hashed(campaign):
    """spec sec 8: DEF row repair is mandatory, not a fallback."""
    for arm in ARMS.split(","):
        route = json.load(open(campaign / "arms" / arm / "route.json"))
        assert route["row_repair"]["changed_location_count"] is not None
        assert len(route["row_repair"]["receipt_sha256"]) == 64
        assert (campaign / "arms" / arm / "repair" / "legalized.def").is_file()


def test_the_routed_geometry_maps_onto_real_segments(campaign):
    for arm in ARMS.split(","):
        with np.load(campaign / "arms" / arm / "route_segments.npz") as data:
            demand = data["segment_demand_router"]
            layered = data["segment_demand_router_layered"]
            capacity = data["segment_capacity"]
        assert demand.shape == capacity.shape
        assert demand.sum() > 0, "no routed crossing on %s" % (arm,)
        # G-3: the 2-D projection can never exceed the layer-summed count.
        assert int(demand.sum()) <= int(layered.sum())


def test_every_arm_reports_region_area_balance(campaign):
    """spec sec 8 and sec 10 risk 3."""
    for arm in ARMS.split(","):
        result = json.load(open(campaign / "arms" / arm / "result.json"))
        balance = result["region_area_balance"]
        for key in ("utilization_max", "utilization_min", "utilization_ratio",
                    "cell_count_deviation"):
            assert key in balance, "%s missing %s" % (arm, key)


def test_the_matrix_is_written_and_ours_is_exactly_one(campaign):
    text = open(campaign / "matrix.md").read()
    assert "ours = 1.000" in text and "reported, not tuned away" in text
    import csv as _csv
    rows = list(_csv.DictReader(open(campaign / "matrix.csv")))
    ours = next(row for row in rows if row["arm"] == "ours")
    for key, value in ours.items():
        if key.endswith("_rel") and value:
            assert abs(float(value) - 1.0) < 1e-9, key


def test_rerunning_the_campaign_is_a_no_op(campaign):
    """Resumability: every step is already current, so nothing re-runs and
    the route directories are untouched (the router runs once per arm)."""
    before = {arm: json.load(open(campaign / "arms" / arm / "route.json"))
              for arm in ARMS.split(",")}
    completed = subprocess.run(
        [sys.executable, "src/scripts/run_v2_campaign.py",
         "--case", "gcd", "--config", CONFIG, "--arms", ARMS,
         "--gpu", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
         "--out", str(campaign),
         "--capacity", str(campaign.parent / "capacity.npz"),
         "--k", "4", "--allow-small-k", "--require-free-gb", "1"],
        check=False, capture_output=True, text=True)
    assert completed.returncode == 0
    assert completed.stdout.count("already current") >= 2 * len(
        ARMS.split(","))
    for arm, route in before.items():
        assert json.load(open(campaign / "arms" / arm / "route.json")) == route


def test_the_fence_only_arm_was_staged_from_the_producer(campaign):
    """G-1: (e) has no soft phase; its fence membership is the producer's."""
    from ioplace.artifacts import load_freeze, load_membership
    arm_dir = campaign / "arms" / "e"
    freeze = load_freeze(str(arm_dir / "freeze.json"))
    assert freeze["reason"] == "fence_from_start"
    staged = load_membership(str(arm_dir / "frozen_membership.npz"))
    produced = load_membership(str(campaign / "producer_b32" /
                                   "membership.npz"))
    assert staged.part.tolist() == produced.part.tolist()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
export IOPLACE_OPENROAD_TESTS=1 CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_v2_campaign_gcd.py -v
```
Expected: FAIL in the `campaign` fixture. The first failure you should see is the capacity extraction or the producer; both are dependency plans' code, not this plan's.

- [ ] **Step 3: Make it pass by fixing integration, not by weakening assertions**

No new product code belongs in this task. Everything the test exercises was written in Tasks 1–7. The expected failures and their correct fixes:

| Symptom | Fix |
|---|---|
| `ioplace.capacity.extract` CLI has different flag names | change the fixture's argv to the landed names; do not add a wrapper |
| the producer aborts `rect_max` at K=4 on 508 cells | P-C Task 10's `rectify_with_fallback` should retry at 32 bins automatically; if it does not, record the failure and mark this test `xfail` with the exact P-C ruling it violates |
| `run_main_flow --phase fence` rejects the staged `freeze.json` | fix `stage_fence_only_arm`'s record in `arms.py` (Task 1 Step 5 already flags this) |
| `_regions_for` cannot find `<arm>/regions.json` for arm `c` | apply Task 7 Step 3 note 1 |
| `demand.sum() == 0` | the scaled/native frame is wrong somewhere; check `export_arm_def`'s die assertion first, then G-4's digest check. **Do not relax the assertion** — zero routed crossings on a four-region GCD means the raster mapping is broken |

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
source src/scripts/env.sh && source src/scripts/openroad_env.sh
export IOPLACE_OPENROAD_TESTS=1 CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest tests/test_v2_campaign_gcd.py -v -s
```
Expected: PASS (8 tests), total wall clock under 30 minutes. GCD's GRT is seconds; the producer's flat GP and three fence GPs dominate.

- [ ] **Step 5: Run the whole suite**

Run: `"$IOPLACE_PYTHON" -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_v2_campaign_gcd.py
git commit -m "$(cat <<'MSG'
test(campaign): spec sec 9's end-to-end small case on GCD

producer -> main flow -> fence LG -> evaluator -> one GRT -> matrix at K=4
across the grid, producer and fence-only paths. Asserts exactly one route per
arm, an identical routing policy across arms, that the mandatory DEF row
repair ran and is hashed, that routed geometry maps onto real segments with
the 2-D projection never exceeding the layer-summed count, that every arm
reports region area balance, that ours is exactly 1.000 in the matrix, and
that a rerun is a no-op.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 9: The result-document templates

Three documents, one per case, with every cell named so a filled document is machine-checkable and an unfilled one is obviously unfinished.

**Files:**
- Create: `docs/results/2026-09-19-v2-matrix-mempool_group.md`
- Create: `docs/results/2026-09-19-v2-matrix-mempool_cluster.md`
- Create: `docs/results/2026-09-19-v2-matrix-synthetic_3x3.md`
- Test: `tests/test_campaign_table.py` (append)

**Interfaces:**
- Consumes: `ioplace.campaign.table.RISK_3` (Task 6).
- Produces: no code. The test pins what each template must contain.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_campaign_table.py`:

```python
import glob


def test_every_matrix_template_exists_and_carries_the_required_statements():
    from ioplace.campaign.table import RISK_3
    paths = sorted(glob.glob("docs/results/2026-09-19-v2-matrix-*.md"))
    assert [os.path.basename(p) for p in paths] == [
        "2026-09-19-v2-matrix-mempool_cluster.md",
        "2026-09-19-v2-matrix-mempool_group.md",
        "2026-09-19-v2-matrix-synthetic_3x3.md"]
    for path in paths:
        text = open(path).read()
        assert "ours = 1.000" in text, path
        assert "metal2-metal10" in text, path
        assert "-congestion_iterations 50" in text, path
        assert "-allow_congestion" in text, path
        assert "648,869" in text, path              # the row-repair evidence
        assert RISK_3.split(".")[0] in text, path   # spec sec 10 risk 3
        assert "sha256" in text, path
        assert "<fill on run>" in text, path        # unfilled is obvious


def test_the_synthetic_template_says_why_it_has_no_routing_row():
    text = open("docs/results/2026-09-19-v2-matrix-synthetic_3x3.md").read()
    assert "Bookshelf" in text and "no LEF/DEF" in text
    assert "--no-grt" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_table.py -k template -v`
Expected: FAIL — the glob returns an empty list.

- [ ] **Step 3: Write `docs/results/2026-09-19-v2-matrix-mempool_group.md`**

````markdown
# v2 experiment matrix — mempool_group

Date: <fill on run>. Status: template — fill from `matrix.md` / `matrix.csv`.
Spec: `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §8
(protocol and matrix), §9 ("done" for G), §10 risk 3.
Plan: `docs/superpowers/plans/2026-09-19-v2-p-g-final-grt-matrix.md`.

This is the **develop** case (spec §0 "Benchmarks"). All nine rows run here.

## Protocol, identical for every arm

One OpenROAD `global_route` per arm, at the very end, never inside global
placement. `set_routing_layers -signal metal2-metal10`,
`-congestion_iterations 50`, `-allow_congestion`, 4 GRT threads. Bounded
iterations are mandatory: tile routed all signals at 5 iterations with zero
overflow in 221.9 s, while unbounded congestion removal cost 9–10 h
(`docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43`).

Every placement passes through OpenROAD DEF row repair before routing, and the
evaluator is re-run on the repaired coordinates. The repair is not cosmetic: it
changed **648,869** cell locations and 2,120,322 orientations on this case
(`docs/results/2026-09-15-route-gp-completion-audit.md:110-121`).

`K=16`. Same DREAMPlace multi-fence legalisation for every arm.

## Setup

| Field | Value |
|---|---|
| Case / config | `mempool_group` / `benchmarks/ispd25/h100/mempool_group.json` |
| Config sha256 | <fill on run> |
| Cells / nets | <fill on run> |
| Producer 64² dir / `regions.json` sha256 | <fill on run> |
| Producer 32² dir / `regions.json` sha256 | <fill on run> |
| `capacity.npz` sha256 / `capacity_source` | <fill on run> |
| `segments_sha256` | <fill on run> |
| OpenROAD binary | `/ldaphome/yyds-tsai-dev/tools/openroad/prefix-upstream-grt-uint64/bin/openroad` |
| GPU / host | <fill on run> |
| Repo commit | <fill on run> |

## Arms

| id | geometry | init | membership from | terms after freeze | norm policy |
|---|---|---|---|---|---|
| `a` | grid 4×4 | region centres | soft phase (prior: producer Mt-KaHyPar, remapped) | capacity + pseudo-FT | grandplan |
| `b` | producer 64² | region centres | soft phase (prior: producer Mt-KaHyPar, not remapped) | capacity + pseudo-FT | grandplan |
| `c` | grid 4×4 | flat seed | soft phase | capacity + pseudo-FT | grandplan |
| `ours` | producer 64² | flat seed | soft phase | capacity + pseudo-FT | grandplan |
| `e` | producer 32² | flat seed | producer Mt-KaHyPar (fixed) | none | — |
| `f` | producer 64² | flat seed | producer Mt-KaHyPar (fixed) | capacity + pseudo-FT | grandplan |
| `ours__polB` | producer 64² | flat seed | soft phase | capacity + pseudo-FT | adaptive |
| `ours__nocap` | producer 64² | flat seed | soft phase | pseudo-FT only | grandplan |
| `ours__nopseudo` | producer 64² | flat seed | soft phase | capacity only | grandplan |

Arms (e) and (f) run `run_main_flow --phase fence` with the producer's
membership staged as the frozen membership — the plan's recorded deviation
G-1. `run_placement_two_stage.py` is unchanged and unused.

## Results, normalised to `ours = 1.000`

<paste the four group tables from `matrix.md` here — primary, routing,
placement, geometry/cost>

## Absolute values

<paste the absolute table from `matrix.md` here>

## Provenance

<paste the per-arm `result.json` / `route.json` sha256 table from `matrix.md`
here>

## Reading

1. **The claim (spec §0).** *"IO-aware simultaneous region production + fence
   placement, under boundary IO capacity, beats grid (a) and pure GrandPlan
   (e) on IO / FT / final GRT overflow."* State whether `ours` beats `a` and
   `e` on `num_over_capacity`, `total_overflow_tracks`, `io_count` and
   `ft_count`, with the numbers, and say so plainly if it does not.
2. **The 2×2 (spec §8).** `(a)`, `(b)`, `(c)`, `ours` decompose the effect into
   a shape axis and a seed axis. Report both marginals. Spec §10 risk 4
   pre-registers the negative outcome: GrandPlan's own Table 3 has shapes alone
   at 1.098 against centre-init 1.099 and seeding at 1.048. *If the shape axis
   reproduces as worthless here, the headline is the capacity and pseudo-FT
   terms, not the geometry* — say that, do not bury it.
3. **Arms (e) and (f).** <the RISK_3 sentence, verbatim> Quote each arm's
   `utilization_ratio` and `cell_count_deviation` alongside its losses: the
   partitioner's ε=0.03 balance constraint is part of the explanation and must
   be visible.
4. **Ablations (spec §10 open question (v)).** *"Do capacity and pseudo-FT
   still help once cells are confined?"* Compare `ours__nocap` and
   `ours__nopseudo` against `ours` on the primary block and on HPWL. P-E's own
   "done" bar is FT strictly reduced at HPWL ≤ 1.01× the no-pseudo arm, *or an
   honest negative result*.
5. **Policy A vs B (spec §4).** `ours__polB` against `ours`. If they differ by
   less than run-to-run noise, say so and stop reporting the axis.

## Caveats carried forward

- **§10 risk 5.** `capacity` is OpenDB's uint8-clamped `getCapacity` (tile
  measured 7,871,705 against a native 7,872,067,
  `docs/results/2026-09-15-ggr-trial.md:39-40`), and "one crossing = one track"
  ignores multi-wire nets and vias. `num_over_capacity` near the knee must not
  be over-read.
- **G-3.** `segment_demand_router_total` counts a net's 2-D projection;
  `actual_io` counts per layer. A large gap means layer stacking over
  boundaries.
- **Global routing is not detailed routing.** Nothing here establishes DRC
  clean detailed routability.
````

- [ ] **Step 4: Write `docs/results/2026-09-19-v2-matrix-mempool_cluster.md`**

Copy the group template with these three changes, and nothing else:

- The heading and the "Setup" case row say `mempool_cluster` /
  `benchmarks/ispd25/h100/mempool_cluster.json`, and the setup table gains a
  row **"Config validated"** — that config was authored, never exercised (P-C
  Task 11 Step 1c, *"UNVALIDATED - no run in this plan"*).
- Under "Arms", keep only `ours`, `a` and `e`, and add: *"This is the
  **validation** case (spec §0 "Benchmarks": develop on `mempool_group`,
  validate on `mempool_cluster`). Three arms only: `ours` against the grid
  baseline (a) and the pure-GrandPlan baseline (e). The full nine-row matrix
  stays on the develop case."*
- Add to "Caveats carried forward": *"§10 risk 2: fence GP at cluster scale is
  extrapolated at ≈5.7 h and ≈45 GB from bigblue4's measured 6.9× runtime and
  4× memory over flat. If an arm OOMs or exceeds its budget, record the failure
  here rather than silently reducing the arm set."*

- [ ] **Step 5: Write `docs/results/2026-09-19-v2-matrix-synthetic_3x3.md`**

````markdown
# v2 scale row — synthetic 3×3 (27.7M cells)

Date: <fill on run>. Status: template — fill from `matrix.md` / `matrix.csv`.
Spec: `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §8
("one synthetic 3×3 27.7M run"), §0 "Benchmarks".
Plan: `docs/superpowers/plans/2026-09-19-v2-p-g-final-grt-matrix.md`, recorded
interpretation G-5.

## There is no routing row here, and why

The synthetic 3×3 array is a **Bookshelf** design streamed from `mempool_group`
by `ioplace.bench.tile_bookshelf` (`.aux/.nodes/.nets/.wts/.pl/.scl`,
`tile_bookshelf.py:309-391`; recipe A′ in
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:73`). It has **no
LEF/DEF**, so `route_eval/placement_openroad.legalize_export` and OpenROAD
`global_route` cannot run on it at all. This run is therefore a
**scale-feasibility row**, executed with `--no-grt`, and every routing cell is
blank rather than zero.

For the record, the protocol that *would* apply if a LEF/DEF version ever
exists: one `global_route` per arm at the very end,
`set_routing_layers -signal metal2-metal10`, `-congestion_iterations 50`,
`-allow_congestion`, 4 GRT threads, after the mandatory DEF row repair that
changed 648,869 cell locations on `mempool_group`
(`docs/results/2026-09-15-route-gp-completion-audit.md:110-121`).

## Preconditions

| Field | Value |
|---|---|
| Bookshelf array `.aux` path | <fill on run> |
| Array built by | `ioplace.bench.tile_bookshelf` 3×3 from `mempool_group` |
| Cells / nets / pins (from the T6b count freeze) | <fill on run> |
| DREAMPlace config | <fill on run> |
| Config sha256 | <fill on run> |
| Host RAM at peak (GB) | <fill on run> |
| GPU / peak GPU memory (GB) | <fill on run> |
| Repo commit | <fill on run> |

If the array does not exist on this host, this document records **blocked**,
names the command that would build it, and stops. Do not substitute a smaller
case and call it the 27.7M row.

## Result, `ours` only

| Metric | Value |
|---|---|
| status | <fill on run> |
| HPWL (GP) / HPWL (LG) | <fill on run> |
| `io_count` / `ft_count` / `hard_lambda_sum` | <fill on run> |
| evaluator `num_over_capacity` / `max_util` / `p99_util` | <fill on run> |
| `straddle_cells` / `straddle_area_fraction` / `straddle_pin_split_nets` | <fill on run> |
| region `utilization_ratio` / `cell_count_deviation` | <fill on run> |
| runtime per phase (s) | <fill on run> |
| peak GPU memory (MB) / host RSS (GB) | <fill on run> |
| routing | n/a — Bookshelf input, no LEF/DEF (`--no-grt`) |
| `result.json` sha256 | <fill on run> |

Normalisation is `ours = 1.000` by construction: there is one row.

## Reading

This row answers one question only — does the v2 main flow complete at 27.7M
cells within the host's memory, and at what cost. It is not evidence about
placement quality and must not be quoted as such. Spec §10 risk 3's standing
instruction still applies to every arm this campaign reports elsewhere:
<the RISK_3 sentence, verbatim>.
````

- [ ] **Step 6: Run the tests to verify they pass**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_table.py -v`
Expected: PASS (12 tests).

- [ ] **Step 7: Commit**

```bash
git add docs/results/2026-09-19-v2-matrix-*.md tests/test_campaign_table.py
git commit -m "$(cat <<'MSG'
docs(results): v2 matrix templates for group, cluster and the synthetic row

One template per case, with every cell named so an unfilled document is
obviously unfinished, the routing protocol and the 648,869-location row-repair
evidence stated up front, spec sec 10 risk 3 carried verbatim, and a reading
section that pre-registers what each comparison decides -- including risk 4's
"if the shape axis is worthless, the headline is the terms". The synthetic 3x3
template states G-5 plainly: Bookshelf input, no LEF/DEF, no routing row.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
)"
```

---

### Task 10: The campaign schedule — order, budgets, exact commands

Tasks 1–9 build the machine. This task says exactly what to run, in what order, and what it will cost, so nobody starts a five-day GPU booking by accident and nobody starts the cluster before the group matrix is readable.

**Files:**
- Create: `docs/results/2026-09-19-v2-campaign-schedule.md`
- Test: `tests/test_campaign_table.py` (append one test)

**Interfaces:**
- Consumes: Tasks 1–9 in full, plus `benchmarks/ispd25/h100/{mempool_group,mempool_cluster}.json` — from plan P-C Task 11, verify at pre-flight; `ioplace.capacity.extract` CLI — from plan P-D Task 3.
- Produces: no code.

**Where the numbers come from.** Every figure below is a measurement or an explicitly-labelled extrapolation from one. Measured: bigblue4 2.18M flat GP+LG **545 s / 1.67 GB** and two-stage fence **3738 s / 6.69 GB** (`results/m0/bigblue4_{flat,two_stage}_k16_grid.json`, i.e. **6.9× runtime, 4× memory**); `mempool_group` flat GP+LG **660–2000 s**; `mempool_cluster` flat **2989 s / 11.1 GB**, with the IO term **6809 s** (spec §8, §10 risk 2); tile GRT all-signal 5 iterations **221.9 s**, unbounded **9–10 h**; group DEF row repair changed **648,869** locations. Extrapolated: group fence GP = 6.9 × group flat; cluster GRT = group GRT × the 3.67 cell ratio; cluster fence GP ≈ 5.7 h / 45 GB (spec §10 risk 2's own extrapolation).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_campaign_table.py`:

```python
def test_the_campaign_schedule_states_order_budgets_and_the_gpu_reservation():
    path = "docs/results/2026-09-19-v2-campaign-schedule.md"
    assert os.path.exists(path)
    text = open(path).read()
    for token in ("mempool_group", "mempool_cluster", "synthetic",
                  "run_v2_campaign.py", "--dry-run", "--no-grt",
                  "6.9x", "648,869", "221.9", "capacity.npz",
                  "a,b,c,ours,e,f,ours__polB,ours__nocap,ours__nopseudo",
                  "ours,a,e"):
        assert token in text, token
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_table.py -k schedule -v`
Expected: FAIL — the file does not exist.

- [ ] **Step 3: Write `docs/results/2026-09-19-v2-campaign-schedule.md`**

````markdown
# v2 campaign schedule — order, budgets, commands

Date: 2026-09-19. Status: schedule. Spec:
`docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §0
"Benchmarks", §8. Plan:
`docs/superpowers/plans/2026-09-19-v2-p-g-final-grt-matrix.md` Task 10.

Spec §0: **develop on `mempool_group`, validate on `mempool_cluster`,
synthetic 3×3 27.7M once.** That is the order below, and it is not negotiable:
the cluster is ~40–60 GPU-hours and must not start before the group matrix has
been read.

## Standing rules

- `K=16`. One `global_route` per arm, at the very end,
  `set_routing_layers -signal metal2-metal10`, `-congestion_iterations 50`,
  `-allow_congestion`, 4 GRT threads. Bounded iterations are mandatory: tile
  routed all signals at 5 iterations in **221.9 s**, unbounded congestion
  removal cost **9–10 h**
  (`docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43`).
- Mandatory OpenROAD DEF row repair before every route; it changed **648,869**
  cell locations on `mempool_group`
  (`docs/results/2026-09-15-route-gp-completion-audit.md:110-121`).
- Shared H100 NVL host. Reserve **one** GPU and announce it. Measured
  2026-09-19: GPUs 0–2 foreign at 100 % utilisation, GPU 3 at 0 % with 22 GB
  resident. Check `nvidia-smi` again before each stage.
- Always `--dry-run` first and read the printed commands.
- Never run two campaigns at once. `run_v2_campaign.py` holds `.grt.lock`
  within a case; across cases, that is the operator's job.

## Cost model

| Quantity | Value | Source |
|---|---|---|
| fence GP vs flat GP | **6.9x** runtime, 4× memory | bigblue4 2.18M: 545 s / 1.67 GB vs 3738 s / 6.69 GB |
| `mempool_group` flat GP+LG | 660–2000 s | spec §8 |
| `mempool_cluster` flat GP+LG | 2989 s / 11.1 GB; 6809 s with the IO term | spec §8 |
| producer | flat GP+LG + <3 % + <60 s CPU SA | spec §2 |
| group GRT | 1–3 h per arm | spec §8 |
| cluster GRT | 4–11 h per arm (group × 3.67 cell ratio, **extrapolated**) | this document |
| cluster fence GP | ≈5.7 h, ≈45 GB (**extrapolated**) | spec §10 risk 2 |

## Stage 0 — inputs and capacity (CPU + OpenROAD, ~2 h, no GPU)

```bash
cd /ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer
source src/scripts/env.sh && source src/scripts/openroad_env.sh
"$IOPLACE_PYTHON" src/scripts/check_v2_campaign_preflight.py -v
```
Expected: `PREFLIGHT OK`. Then one capacity extraction per case — once, on the
**input** DEF, at minimal congestion iterations with `-allow_congestion`
(spec §5):

```bash
for case in mempool_group mempool_cluster; do
  mkdir -p results/v2_capacity/$case
  "$IOPLACE_PYTHON" -m ioplace.capacity.extract \
      --config benchmarks/ispd25/h100/$case.json --k 16 --rtype grid \
      --out results/v2_capacity/$case/capacity.npz \
      --openroad "$OPENROAD_BIN" 2>&1 | tee results/v2_capacity/$case/extract.log
done
```
Budget: group ≤1 h, cluster ≤2 h. If either exceeds twice that, stop — an
extraction that slow means the bounded-iteration setting did not take.

Record both `capacity.npz` sha256 values in the matrix documents.

## Stage 1 — develop on `mempool_group`: all nine arms (GPU, 1.5–3.5 days)

```bash
export CUDA_VISIBLE_DEVICES=3
ARMS=a,b,c,ours,e,f,ours__polB,ours__nocap,ours__nopseudo
OUT=results/v2_matrix_20260919/mempool_group
"$IOPLACE_PYTHON" src/scripts/run_v2_campaign.py \
  --case mempool_group --config benchmarks/ispd25/h100/mempool_group.json \
  --arms "$ARMS" --gpu 3 \
  --capacity results/v2_capacity/mempool_group/capacity.npz \
  --out "$OUT" --dry-run
```
Read the 2 producer + 18 place/route lines, then drop `--dry-run` and run it
under `nohup`/`tmux`. Per-arm budget:

| Phase | Budget | Note |
|---|---|---|
| producer 64² / 32² | ≤25 min each, once | spec §2 target |
| soft GP (arms a/b/c/ours and the three ablations) | 0.4–1.3 h | flat 660–2000 s × the 2.3× IO-term factor measured at cluster |
| fence GP | 1.3–3.8 h | 6.9× flat |
| fence LG + evaluator | ≤0.2 h | |
| DEF export + row repair | ≤0.5 h | 3.08M components |
| GRT | 1–3 h | spec §8 |
| **per arm** | **3.0–8.8 h** | fence-only arms (e)/(f) save the soft GP |
| **nine arms** | **31–79 h** | sequential |

Peak GPU memory: fence GP ≈ 4× the flat footprint. Keep `--require-free-gb 30`.

Stop conditions, decided in advance:
- an arm whose GRT hits `--grt-timeout-s` (default 6 h) records
  `returncode 124` and the campaign continues to the next arm; the row is
  reported as `grt_timeout`, never silently dropped;
- an arm that OOMs is recorded, not retried at a smaller `K` — `K=16` is fixed;
- if `ours` itself fails, stop the campaign. A matrix without its base row has
  no normalisation.

Then read `${OUT}/matrix.md` and fill
`docs/results/2026-09-19-v2-matrix-mempool_group.md`. **Gate: do not start
Stage 2 until that document is filled and its "Reading" section answers all
five questions.**

## Stage 2 — validate on `mempool_cluster`: `ours`, `(a)`, `(e)` (GPU, 2–2.5 days)

Three arms only (spec §0: validate, do not re-run the whole matrix).

```bash
export CUDA_VISIBLE_DEVICES=3
OUT=results/v2_matrix_20260919/mempool_cluster
"$IOPLACE_PYTHON" src/scripts/run_v2_campaign.py \
  --case mempool_cluster \
  --config benchmarks/ispd25/h100/mempool_cluster.json \
  --arms ours,a,e --gpu 3 \
  --capacity results/v2_capacity/mempool_cluster/capacity.npz \
  --out "$OUT" --require-free-gb 60 --grt-timeout-s 43200 --dry-run
```

| Phase | Budget |
|---|---|
| producer 64² / 32² | ≤60 min each, once |
| soft GP (`ours`, `a`) | ≈1.9 h |
| fence GP | ≈5.7 h, ≈45 GB (**extrapolated**) |
| fence LG + evaluator | ≤1 h |
| DEF export + row repair | ≤1.5 h |
| GRT | 4–11 h (**extrapolated**; timeout 12 h) |
| **per arm** | **13–20 h** |
| **three arms** | **39–60 h** |

`benchmarks/ispd25/h100/mempool_cluster.json` was authored, never exercised
(P-C Task 11 Step 1c). Validate it first:

```bash
"$IOPLACE_PYTHON" - <<'PY'
import json, os
d = json.load(open("benchmarks/ispd25/h100/mempool_cluster.json"))
assert d["gpu"] == 1
for f in d["lef_input"] + [d["def_input"]]:
    assert os.path.exists(f), f
print("ok", d["def_input"])
PY
```

Fill `docs/results/2026-09-19-v2-matrix-mempool_cluster.md`. If an arm exceeds
its budget or OOMs, record that in the document; do not reduce the arm set
silently.

## Stage 3 — synthetic 3×3 27.7M, `ours` only, no routing (GPU, ≤24 h)

Recorded interpretation G-5: the array is Bookshelf, has no LEF/DEF, and
therefore produces **no routing row**. Check the array exists before booking
the GPU:

```bash
ls -l <bookshelf 3x3 prefix>.aux <bookshelf 3x3 prefix>.nodes
```
If it does not exist, record **blocked** in
`docs/results/2026-09-19-v2-matrix-synthetic_3x3.md`, name
`ioplace.bench.tile_bookshelf` as the builder, and stop. Do not substitute a
smaller case.

```bash
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" src/scripts/run_v2_campaign.py \
  --case synthetic_3x3 --config <synthetic config.json> \
  --arms ours --gpu 3 --out results/v2_matrix_20260919/synthetic_3x3 \
  --no-grt --require-free-gb 60 --dry-run
```
Budget: one run, hard stop at 24 h. Host RSS at 27.7M was estimated at 95–105
GB and that estimate is **unvalidated**
(`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md:105`); watch
`host_peak_rss_gb` and abort rather than swap the host.

## Stage 4 — read the matrix (CPU, hours)

Regenerate any table without re-running anything:

```bash
"$IOPLACE_PYTHON" - <<'PY'
from ioplace.campaign.table import build
md, csv, rows = build("mempool_group",
                      "results/v2_matrix_20260919/mempool_group/arms",
                      ["a", "b", "c", "ours", "e", "f",
                       "ours__polB", "ours__nocap", "ours__nopseudo"])
open("results/v2_matrix_20260919/mempool_group/matrix.md", "w").write(md)
open("results/v2_matrix_20260919/mempool_group/matrix.csv", "w").write(csv)
print(md[:2000])
PY
```

G's "done" (spec §9) is reached when all three result documents are filled with
the hash tables non-empty.

## Total

| Stage | Wall clock | Resource |
|---|---|---|
| 0 inputs + capacity | ~2 h | CPU + OpenROAD |
| 1 group, nine arms | 31–79 h | one GPU, exclusive |
| 2 cluster, three arms | 39–60 h | one GPU, exclusive, ≥60 GB free |
| 3 synthetic, one arm | ≤24 h | one GPU, exclusive, high host RAM |
| 4 reading | hours | CPU |
| **total** | **≈4–7 days of one reserved GPU** | |
````

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_campaign_table.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Run the whole fast suite once more**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/results/2026-09-19-v2-campaign-schedule.md \
        tests/test_campaign_table.py
git commit -m "$(cat <<'MSG'
docs(results): the v2 campaign schedule, order, budgets and commands

Develop on mempool_group (all nine arms), validate on mempool_cluster (ours,
a, e), synthetic 3x3 once with --no-grt because its Bookshelf input has no
LEF/DEF. Every budget is a measurement or a labelled extrapolation from one:
the 6.9x fence-GP factor from bigblue4, group flat 660-2000 s, cluster flat
2989 s / IO 6809 s, tile GRT 221.9 s bounded against 9-10 h unbounded, and
the 648,869-location row repair. Stop conditions and the gate between stages
are decided in advance, not mid-run.

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
| §0 "Routing": `global_route` once at the very end of every arm | Global Constraints; 5 (`run_final_grt` refuses a second route); 7 (one `grt` step per arm); 8 (asserted) |
| §0 "Routing": GR-in-loop stays behind `IOPLACE_ENABLE_GR_IN_LOOP`, unmaintained | Global Constraints (this plan imports neither module and must not un-gate them) |
| §0 "Arms": the six arms, K=16 only, every arm reports area balance | 1 (registry, `K=16` guard); 6 (`utilization_ratio`, `cell_count_deviation` in `METRICS`); 8 (asserted per arm) |
| §0 "Benchmarks": develop group, validate cluster, synthetic 3×3 once | 10 |
| §0 "Claim": beats grid (a) and pure GrandPlan (e) on IO / FT / final GRT overflow | 9 (the "Reading" section states the claim and requires the numbers, including a negative) |
| §8 recipe step 1: `placement.npz` → DEF | 5 (`export_arm_def`) |
| §8 recipe step 2: `legalize_export` DEF row repair, **mandatory** | 5 (`repair_def_rows`); Global Constraints; 8 (asserted); 9 (evidence quoted) |
| §8 recipe step 3: `run_openroad` with `set_routing_layers metal2-metal10`, `-congestion_iterations 50`, `-allow_congestion`, 4 threads, `grt::write_segments` | 4 (`GRT_POLICY`); 5; 7 (CLI defaults); 8 (identical across arms, asserted) |
| §8 recipe step 4: `common_grt.py` decode | 4 (`decode_segments` via `read_nets`), with interpretation G-2 recording why `evaluate` cannot be used |
| §8 recipe step 5: fast evaluator on the same repaired coordinates | 5 (`_reevaluate`, `repaired_*` fields) |
| §8 recipe step 6: `result.json` with per-file SHA-256 | 4 (`build_route_record`'s `sha256` block); 5; 6 (provenance table) |
| §8 primary: per-segment crossings vs capacity — `num_over_capacity`, `max_util`, `p99_util`, total overflow tracks, through §5's unit-edge raster | 4 (`router_segment_table` calling P-D's `route_segment_demand`, `segment_metrics` on `CAPACITY_SCALARS`); 6 (primary group first) |
| §8 secondary: native overflow, routed WL, via count, HPWL(GP/LG), io/ft/hard_lambda_sum/io_rg/ft_rg, the §7 straddle set, per-phase runtime and peak GPU memory, region area balance | 4 (route side); 6 (`METRICS` covers all of them) |
| §8 runtime expectations and the bounded-iterations mandate | Global Constraints; 3 (the timeout backstop); 10 (the cost model) |
| §8 matrix table, one per case, normalised `ours = 1.000` | 6; 9 |
| §8 policy-A-vs-B ablation and capacity / pseudo-FT on-off as separate rows | 1 (`ABLATION_IDS`); 9 (the reading rules); 10 (the arm list) |
| §9 "done" for G: full matrix tabled with hashes | 6, 9, 10 |
| §9 end-to-end small case: GCD, producer → main flow → fence LG → evaluator → one GRT | 8 |
| §10 risk 3: arms (e)/(f) may lose; report area balance | 6 (`RISK_3` in every document); 9; Global Constraints |
| §10 risk 4: if the shape axis is worthless, the headline is the terms | 9 (pre-registered in the "Reading" section) |
| §10 risk 5: uint8-clamped capacity, "one crossing = one track" | 9 (carried as a caveat in every template) |
| §1 retired-path gating | Global Constraints |

**Gaps closed while reviewing.** Three, all now tasks: (i) nothing owned the fact that `run_placement_two_stage.py` cannot express arm (f) — now interpretation G-1 and Task 1; (ii) nothing owned the fact that the synthetic 3×3 case cannot be routed — now G-5, Task 7's `--no-grt`, and Task 9's third template; (iii) `load_observation` returned no decoded geometry, so the per-segment table had nothing to count — now Task 3, shared with P-D Task 10.

**Deliberately out of scope and *not* gaps:** the capacity extraction itself (P-D Task 3 — this plan calls its CLI), the surrogate-vs-router rank correlation (P-D Task 10 — a different question about the same `route.json`), the anchor-comparison experiment (P-F Task 7), the pseudo-FT displacement study (P-E Task 9), and Tier 2/Tier 3 benchmarks (spec §0 lists them; §8's matrix is Tier 1 only).

### 2. Placeholder scan

Searched for `TBD`, `TODO`, `implement later`, `fill in details`, `Similar to Task N`, `add appropriate error handling`, `add validation`, `handle edge cases`, `Write tests for the above`, `FIXME`, `XXX`. Zero remain. Four near-misses fixed:

- Task 5's `_reevaluate` originally said "re-run the evaluator" without code. It is now written out, including the movable-prefix overwrite and the scaled `RegionSet` conversion, because the *point* of the step is that it uses repaired coordinates and a hand-wave would lose that.
- Task 7's `_regions_for` had a "figure out where regions.json is" comment. It now has a three-branch implementation and an explicit note telling the implementer what to do if grid arms do not persist one.
- Task 8 listed "fix whatever breaks". It now carries a symptom → fix table with five named failure modes and an explicit instruction not to relax the `demand.sum() > 0` assertion.
- The result templates used `TBD` for the date; they use `<fill on run>`, and Task 9's test asserts that token is present so an unfilled document is detectable.

Two places deliberately carry an instruction rather than final code, and both say exactly what the final form must be: Task 3 Step 4 (`grep` first, because P-D may have added `net_polylines` already) and Task 0 Step 3 note 1 (`load_capacity_for_grid`'s module may have moved). They are decision points about *which sibling plan landed first*, not gaps.

### 3. Type consistency

Every name crossing a task boundary was checked against its definition:

- `ArmSpec`'s nine fields (`arm_id`, `label`, `geometry`, `init`, `driver`, `norm_policy`, `capacity`, `pseudo_ft`, `remap_blocks`) are defined in Task 1 and read by name in Tasks 6 (`ARMS[arm_id].label`) and 7 (`spec.driver`, `spec.geometry`). `GEOMETRY_PRODUCER` and `PRODUCER_BINS` are defined in Task 1 and imported by Task 7's `plan_case` — both were missing from Task 1's first draft and were added.
- `stage_fence_only_arm(out_dir, producer_dir, *, k) -> dict` — Task 1's signature matches Task 7's call and Task 8's assertion on `staged["staged"]`.
- `run_final_grt(case, arm, arm_dir, *, config, regions_json, capacity_npz, binary, timeout, threads, congestion_iterations, repair_threads, reevaluate)` — Task 5's signature matches Task 7's call exactly; `reevaluate` defaults to `True` and Task 5's own monkeypatched test overrides `_reevaluate`, not the flag.
- `build_route_record`'s keyword list in Task 4 matches every call in Task 5 character for character, including `segment_metrics_out` (deliberately not `segment_metrics`, which is the *function* in the same module).
- `segment_metrics` returns `CAPACITY_SCALARS`' names plus `total_overflow_tracks`, `segment_demand_total` and `num_segments`; `build_route_record` reads exactly those, and Task 6's `METRICS` sources `route.num_over_capacity`, `route.total_overflow_tracks`, `route.max_util`, `route.p99_util` — all present in the record.
- `router_segment_table` returns `{"demand", "demand_layered"}`; Task 5 reads both, Task 8 asserts on the `route_segments.npz` array names `segment_demand_router` / `segment_demand_router_layered` that Task 5 writes.
- `Metric.source` dotted paths resolve against the two-key payload `{"result": ..., "route": ...}` that `collect_arm` builds; Task 6 Step 5 checks every `result.*` leaf against `MAIN_FLOW_RESULT_FIELDS` at execution time rather than trusting the plan.
- `sha256`, `load_state`, `record_step`, `step_is_current`, `clear_step`, `grt_lock`, `plan_case`, `build_parser`, `main` — Task 7 defines all nine and its tests call all nine by those names.
- `GRT_POLICY` (Task 4) vs the per-run `policy` dict (Task 5): the latter is built from the CLI and passed as `routing_policy=`, so an arm routed at non-default settings records what it actually used, and Task 6's markdown warns when the arms disagree. `GRT_POLICY` is the default and the documentation anchor, never silently substituted for the real one.

### 4. Known limitations recorded rather than hidden

1. **G-1's deviation is a design decision, not a shortcut**, and it changes what arm (f) measures. Flagged to the scheduler below.
2. **`_reevaluate` rebuilds a PlaceDB and re-runs the GPU evaluator**, so the GRT step needs the GPU briefly. On a host where the GPU is released between stages, pass `reevaluate=False` and lose the repaired-coordinate metrics — the plan does not expose that as a CLI flag precisely so nobody does it by accident.
3. **Cluster GRT runtime is extrapolated**, not measured; the 12 h timeout in Stage 2 is a guess bounded by the measured group range times the cell ratio. The first cluster arm's `t_grt_s` replaces the estimate.
4. **`num_over_capacity` inherits §10 risk 5's uint8 clamp.** Every template says so; no conclusion may rest on a difference of a few tenths of a percent in utilisation.

### 5. Risks to flag to the scheduler before execution

- **G-1 needs a ruling.** Spec §8 names `run_placement_two_stage.py` for arm (f); this plan runs (f) through `run_main_flow --phase fence` and leaves the two-stage driver untouched. The reasoning is in "Recorded interpretations": the landed driver cannot express the arm, and adapting it would compare `ours` against a *different fence-GP implementation*. If the scheduler prefers literal spec compliance, Task 1 changes and a new task must add six flags plus the v2 result schema to `run_placement_two_stage.py` — budget a full task and accept the confound.
- **Task 5 has the largest specification surface per line here** and is written against `run_main_flow`'s and P-D's plan text, not their landed code. Run Task 0's pre-flight *first* and diff `load_capacity_for_grid`'s real signature and `placement.npz`'s real contents before writing it.
- **The campaign is ~4–7 days of one reserved GPU** (Task 10). That is a scheduling decision, not an implementation detail. Stages 1 and 2 are separated by a hard gate precisely so the cluster booking can be cancelled if the group matrix says the shape axis is worthless (spec §10 risk 4).
