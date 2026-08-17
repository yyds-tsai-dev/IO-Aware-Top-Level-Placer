# H100 30M Handover Runbook (M4 T10)

Deliverable for M4 Task T10 (spec `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`
§6.4 "H100 交接契約"). This document is the single artifact a new operator on the target H100
machine needs to reproduce the 27.7M (3×3) synthetic array, rebuild the environment, run the
frozen single command, and adjudicate the result against the pre-registered forecast.

**Design principle (spec §6.4, verbatim): 在 H100 上不做任何設計決策.** Every parameter below
(K, ρ, seed, region type, iteration budget, gate thresholds, the forecast itself) was already
fixed on the L4 side. The H100 machine's job is to *execute* T10's single command and *compare*
against the frozen prediction (T14) — nothing here should be re-tuned in response to what the
H100 run produces.

---

## 1. Resource contract

| Resource | Requirement | Source |
|---|---|---|
| GPU | **H100 SXM5 80GB HBM3** — SKU pinned exactly: 3.35 TB/s peak memory bandwidth, `sm_90`, CUDA 12.8, persistence mode **on**, default (not locked/boosted) clocks, **not** MIG-partitioned, **not** underclocked | `results/m4/forecast/h100_prediction.json` → `sku` block (frozen, sha256 below) |
| Host RAM | **≥ 192 GB** | `results/m4/forecast/h100_prediction.json` → `host_rss_forecast.h100_min_host_ram_gb_contract == 192.0` |
| Disk | **≥ 100 GB** free (the 27.7M array's 5 Bookshelf files + T9 tiled-netlist-cache `.npy`s, regenerated locally — see §3) | spec §6.4 |
| GPU exclusivity | **Exclusive** — no other CUDA-active process on the device for the duration of the run | `ioplace/diagnostics/probes_m4/gpu_exclusivity.py` (preflight/postflight protocol, below) |

**If the actual machine's SKU differs from the frozen `sku` block in any of the above fields
(different GPU model, clock policy, CUDA build, MIG on) the prediction is invalid** — do not run
T14's comparison against it; re-run `scripts/m4_forecast.py` and re-freeze a new
`h100_prediction.json` instead of stretching the existing one (spec §6.3 E5, `sku.note`: "若實際
機器不同 ⇒ 預測無效,須重登錄而非事後放寬").

### GPU exclusivity — preflight/postflight protocol

`ioplace/diagnostics/probes_m4/gpu_exclusivity.py` is the nvidia-smi-only implementation every
"formal memory run needs exclusive GPU" check in this repo is built from (shared by
`ioplace/bench/spike_30m.py` and this task's T1b probes). It never builds a CUDA context itself
— only `nvidia-smi` subprocess calls.

- **`preflight_exclusivity(self_test_pid=...)`**: `device_baseline_gb` (absolute, not a delta)
  must be `< CONTAMINATION_BASELINE_GIB == 0.5` GiB, and `nvidia-smi --query-compute-apps` must
  list no compute-active PID other than our own. The compute-apps query is self-tested first
  (`self_test_own_pid_visible`) — this host class has been observed to return an *empty*
  compute-apps list even while a real workload is running, so an empty reading is only trusted as
  real evidence of exclusivity (`exclusivity_evidence == "compute_apps_pid_confirmed"`) after the
  self-test confirms the query can see our own PID while it is CUDA-active; otherwise evidence
  downgrades to `"baseline_only"`.
- **`postflight_exclusivity(baseline_before_gb)`**: `device_used_gb` must return to the preflight
  baseline within `POSTFLIGHT_TOLERANCE_GIB == 0.2` GiB, and no compute-apps PID may still be
  lingering.
- Any preflight/postflight failure is the caller's cue to record `experiment_status="contaminated"`
  — this module only reports pass/fail + reasons, it does not itself decide `experiment_status` or
  write artifacts.

`scripts/m4_run.sh` itself does **not** call this module directly (see §4) — it only does a
cheaper torch/CUDA-availability + GPU-name check. Run the preflight/postflight functions above
by hand (or via whatever T9/T14 driver wraps them) around the T10 invocation if a
`experiment_status`-grade exclusivity record is required for the run's provenance.

---

## 2. Environment rebuild

Full build procedure is `docs/dev-env.md`; `DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`,
`PY=$DP/.venv312/bin/python`. Reproduce verbatim on the H100 node (this repo's Python/CUDA/Boost
toolchain gaps were resolved without root, self-contained under `$DP/deps/` — see dev-env.md for
the Boost 1.74.0 `.deb`-extraction and CUDA 12.8 runfile-install procedures):

```bash
DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace
PY=$DP/.venv312/bin/python
export PATH=$DP/deps/cuda-12.8/bin:$PATH
export CUDACXX=$DP/deps/cuda-12.8/bin/nvcc
export CUDA_HOME=$DP/deps/cuda-12.8

cd $DP && cmake -B build312 \
  -DPython_EXECUTABLE=$PY \
  -DCMAKE_INSTALL_PREFIX=$DP/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_ABI=1 \
  -DBOOST_ROOT=$DP/deps/boost/usr \
  -DBOOST_INCLUDEDIR=$DP/deps/boost/usr/include \
  -DBOOST_LIBRARYDIR=$DP/deps/boost/usr/lib/x86_64-linux-gnu \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=$DP/deps/cuda-12.8

cmake --build build312 -j $(nproc)
cmake --build build312 --target install
```

### Required check 1 — `CMAKE_CXX_ABI=1`

torch 2.8.0's official PyPI wheel is built with the new C++11 `std::string` ABI
(`torch.compiled_with_cxx11_abi() == True`); `$DP/CMakeLists.txt` defaults `CMAKE_CXX_ABI`
(i.e. `_GLIBCXX_USE_CXX11_ABI`) to **0** unless overridden. Omitting `-DCMAKE_CXX_ABI=1` produces
a cmake-successful build that fails at **runtime** on `import PlaceDB` with an undefined-symbol
error (`c10::detail::torchCheckFail(...)` demangled with old-ABI `std::string` mangling) —
dev-env.md documents this exact failure and fix. **Confirm in the configure log**:
`CMAKE_CXX_ABI: _GLIBCXX_USE_CXX11_ABI=1`.

### Required check 2 — `CUDA_ARCH_FLAGS` includes `sm_90`

**Verified discrepancy vs. this task's brief:** the brief describes the local (L4) build as
"本機只 build sm_89" and asks for a check that adds `sm_90` to `CUDA_ARCH_FLAGS`. The actual
local configure log (`$DP/build312/configure2.log:42`) already includes **both**:

```
CUDA_ARCH_FLAGS: ...;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_90,code=sm_90
```

This is because `$DP/CMakeLists.txt`'s architecture-selection block (`if (NOT
CMAKE_CUDA_ARCHITECTURES) ... if (CUDA_VERSION_MAJOR >= 12) list(APPEND
CMAKE_CUDA_ARCHITECTURES 8.9 9.0)`) unconditionally appends **both** `8.9` (sm_89, this host's
L4/Ada) and `9.0` (sm_90, H100/Hopper) whenever the CUDA toolchain is major version ≥12 — which
this repo's CUDA 12.8 install is — and no `-DCMAKE_CUDA_ARCHITECTURES` override was passed in
§2's build command. **No extra flag is actually needed to target sm_90**; the existing
`.so`s are already a fat binary covering both architectures. The "check" that matters is
therefore **verification, not addition**: after configure, grep the log —

```bash
grep "CUDA_ARCH_FLAGS" $DP/build312/configure2.log
```

— and confirm `arch=compute_90,code=sm_90` is present. If it is *not* present (e.g. a different
CUDA version, or someone passes an explicit `-DCMAKE_CUDA_ARCHITECTURES` that omits 9.0), add
`-DCMAKE_CUDA_ARCHITECTURES="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"` explicitly to the cmake command
in §2 and rebuild.

---

## 3. Data — regenerate the 27.7M array on-node

**Do not copy the array's Bookshelf files between machines.** Regenerate them from the
`mempool_group` source export + calibrated glue statistics already checked into this repo.

### 3.1 Tiler + glue (3×3, N2)

`ioplace/bench/tile_bookshelf.py` and `ioplace/bench/glue_gen.py` are library modules with no
standalone CLI of their own; the reproducible entry point is `ioplace/bench/build_t6_arrays.py`,
which was the actual invocation used to build `results/m4/bench/arrays/3x3_n2/`:

```bash
PYTHONPATH=. $PY -m ioplace.bench.build_t6_arrays \
    --source results/m4/bench/mempool_group_export/mempool_group \
    --cluster-stats results/m4/bench/cluster_stats.json \
    --out-dir results/m4/bench/arrays --seed 0 \
    --shapes 3x3 --norms n2
```

(Omitting `--shapes`/`--norms` also builds `1x2`/`2x2` × {N1, N2}, matching the docstring's
general-usage example — the flags above scope it down to reproducing exactly `3x3_n2`.) This
tiles the single-tile `mempool_group` source R×C=3×3 (streaming pass, coordinate translation
`(i*W, j*H)`, no channel), then appends N2-normalized cross-tile glue nets (`lambda_0_tile`,
`alpha=0.1023`, `n2_pair_counts_sinkhorn`, `seed=0`).

**Expected output** (`results/m4/bench/arrays/3x3_n2/3x3_n2.manifest.json`, already checked into
this repo — compare against it after regeneration):

| field | value |
|---|---|
| `base.n_nodes` | 27,804,699 |
| `base.n_nets` | 31,535,892 |
| `base.n_pins` | 108,235,719 |
| `glue.n_nets` | 59,360 |
| `glue.n_pins` | 118,720 |
| total nets | 31,595,252 |
| total pins | 108,354,439 |
| `output_sha256.nodes` | `bd3c3fb80b4b4cdd2cf1aa71d8298b8de265f32fcf004b075ef2b22f4a632d85` |
| `output_sha256.nets` | `a1181e73587bba9b8d531ed4fbc3e3ebc319d283052ea678d2768227a63bfaa9` |
| `output_sha256.pl` | `0e836f2e2e07e4d86b2d708f09f9ea8fc9816c08b49cd2165e9e875233fe75b5` |
| `output_sha256.scl` | `b0d7e4c143e1c10cf9b4e6bb0f1e213e37fb479c0207c75f75d5d2262fd0e3d4` |
| `output_sha256.wts` | `bd900238f1c8e315a9a55c04ab4e7bd11195be474df75e486902d6a047a7b562` |

`tile_bookshelf.tile`'s output is deterministic and does **not** depend on `seed` (a pure
streaming replication — `seed` is recorded but only consumed by the glue step); only the glue
tail is seed-dependent. `n_nets`/`n_nodes`/`n_pins` and the `.nodes`/`.pl`/`.scl`/`.wts` sha256s
above should reproduce exactly on the target machine given a byte-identical `mempool_group`
source (its own `source_sha256` block is also in the manifest); `.nets`' sha256 should also match
since the glue seed (0) is fixed.

### 3.2 T9 tiled-netlist cache + full verification

`ioplace/bench/bookshelf_netlist.py` has no CLI wrapper either — call its two functions directly
(same pattern `tests/test_bench_bookshelf_netlist.py` uses):

```bash
PYTHONPATH=. $PY -c "
from ioplace.bench.bookshelf_netlist import build_tiled_netlist_cache, verify_against_bookshelf
import json
meta = build_tiled_netlist_cache(
    'results/m4/bench/arrays/3x3_n2/3x3_n2.manifest.json',
    'results/m4/bench/t9_cache_3x3')
result = verify_against_bookshelf(
    'results/m4/bench/t9_cache_3x3',
    'results/m4/bench/arrays/3x3_n2/3x3_n2',
    mode='full')
print(json.dumps(result, indent=1))
"
```

`build_tiled_netlist_cache` parses the source tile once, replicates via numpy tile/translate
(never re-reading the R×C array's own multi-GB files), applies the movable-first stable
permutation, and writes `node_x/node_y/node_size_x/node_size_y/pin_offset_x/pin_offset_y/
pin2node/pin2net/flat_net2pin/flat_net2pin_start/inv_perm` `.npy`s + `meta.json` to
`results/m4/bench/t9_cache_3x3/`. `verify_against_bookshelf(..., mode="full")` independently
re-derives every record by streaming the on-disk Bookshelf array (never trusts the replication
arithmetic alone) — this is the "~5 min CPU" full verification the design draft's sec 3.2
describes.

**0-mismatch precedent** (`results/m4/bench/verify_tiled_netlist_3x3.json`, already checked in —
this run should reproduce the same zero-mismatch counts against the regenerated array):

```json
{
 "mode": "full", "ok": true, "errors": [],
 "n_checked_nodes": 27804699, "n_mismatched_nodes": 0,
 "n_checked_pl": 27804699, "n_mismatched_pl": 0,
 "n_checked_nets": 31595252, "n_mismatched_nets": 0,
 "n_checked_pins": 108354439, "n_mismatched_pins": 0,
 "n_checked_glue_nets": 59360, "n_mismatched_glue_nets": 0,
 "elapsed_s": 324.5166075229645
}
```

`ioplace/bench/spike_30m.py` (T9's own spike driver) expects the cache at
`--cache-dir results/m4/bench/t9_cache_3x3` — this is the exact path convention already used by
the L4-side spike runs (`results/m4/profile/spike30m__A__k32.json` /
`spike30m__B__k32.json`'s own recorded `command` field).

---

## 4. Single command — `scripts/m4_run.sh`

**Discrepancy vs. this task's brief:** the brief and spec §6.4 describe the interface as
`scripts/m4_run.sh <case> <K> <rtype> <arm>` — four positional arguments. **The actual script
takes no positional arguments; it is entirely flag-based** (`--config`, `--k`, `--rho`, `--seed`,
`--dp-seed`, `--rtype`, `--deterministic`, `--benchmark-kind`, `--out`, `--dry-run`). There is no
`<case>` selector at all — the "case" is implicit in whichever `--config` path is passed, and
there is no separate `<arm>` flag: flat vs. `ours@M2` is selected via `--rho`/`--rho-max`
(`0.0` = flat, `>0.0` = ours@M2 — the script's own comment cites this as "the established T8b
convention", matching `results/m4/t8b/cluster__k16__grid__flat.json`'s `"mode":"io","rho_max":0.0`
fields: flat has never been a distinct driver mode, it is `io` mode with the IO term weighted to
zero). Actual usage (`bash scripts/m4_run.sh --help`):

```
Usage: m4_run.sh [options]

  --config PATH       DREAMPlace params JSON (default: benchmarks/ispd25/synthetic_3x3_n2.json)
  --k K                region count (default: 16)
  --rho RHO            IO term weight (--rho-max passthrough); 0.0 = flat arm,
                        >0.0 = ours@M2 arm (default: 0.0)
  --seed SEED           placement seed (default: 1000)
  --dp-seed SEED        DREAMPlace RNG seed (default: same as --seed)
  --rtype RTYPE          grid|slicing (default: grid)
  --deterministic 0|1     (default: 1)
  --benchmark-kind KIND   real|synthetic (default: synthetic -- the 27.7M
                           3x3 array is always synthetic per T7's provenance)
  --out PATH             output profile JSON path
  --dry-run               print the env check + the command that would run,
                           but do not execute the placement itself
  -h, --help
```

The script's default `--config` is `benchmarks/ispd25/synthetic_3x3_n2.json` — **this file does
not exist in this repo yet** (confirmed: no `synthetic_3x3_n2.json` under `benchmarks/ispd25/`).
It must be created on the H100 node once §3's array is regenerated, following the same shape as
the existing `synthetic_1x2_n2.json`/`synthetic_2x2_n2.json` configs (`aux_input` pointing at
`results/m4/bench/arrays/3x3_n2/3x3_n2.aux`).

**Second discrepancy, found while inspecting the sibling configs (relevant to T10 setup):**
`synthetic_1x2_n2.json` and `synthetic_2x2_n2.json` were both patched (commit `9b662a2`,
2026-08-17) to point `aux_input` at a letter-prefixed symlink alias (`.../alias/synA.aux`,
`.../alias/synB.aux`) rather than the array's own digit-leading `.aux` — DREAMPlace's Bookshelf
lexer parses a digit-leading filename like `1x2_n2.aux` as an integer token and fails
(`1x2_n2.aux:1.21 syntax error`; the commit message ties this directly to the T6B
`V1_dreamplace_readable == "infrastructure_blocked"` finding, `results/m4/bench/
verify_group3x3_t6b.json`'s `checks.V1_dreamplace_readable`). **`3x3_n2` has the same digit-leading
name (`3x3_n2.aux`) and no alias directory exists for it yet** (`results/m4/bench/arrays/3x3_n2/`
has no `alias/` subdirectory, unlike `1x2_n2/alias/` and `2x2_n2/alias/`). Whoever writes
`benchmarks/ispd25/synthetic_3x3_n2.json` on the H100 node should create the same
letter-prefixed alias (e.g. `results/m4/bench/arrays/3x3_n2/alias/synC.aux` symlinking to the
verified array files, content untouched, sha256 chain preserved via the manifest) and point
`aux_input` at that alias — otherwise DREAMPlace's own `PlaceDB.read` will hit the same lexer
failure this run is trying to avoid. This is separate from — and does not fix — §7's
`V1_dreamplace_readable` hierarchical-node-name regex issue, which is about instance *names*
inside the files, not the aux *filename*.

`m4_run.sh` itself does three things: (1) an environment check (venv python present, torch/CUDA
importable, GPU name/memory reported — warns, does not hard-fail, if the GPU name doesn't
contain `"H100"`; host RAM ≥128 GB or a T12-fallback warning — note this 128 GB threshold is a
different, lower bar than this runbook's §1 contract of ≥192 GB, since it is T12's numpy-only
`PlaceDB` shim trigger, not T10's own resource contract); (2) prints, and unless `--dry-run` runs,
the full `ioplace.drivers.run_placement` command (schema v3, `--mode io`, `--no-diag` — matching
the phase model `scripts/m4_forecast.py` assumes, "diagnostics 關閉"); (3) on non-zero exit,
prints the §2.2/§7.0 three-state recording guidance (`feasible_l4_contract` /
`infeasible_l4_contract` / `invalid_measurement`) instead of treating a crash or OOM as just a
bug — an OOM at this scale is a legitimate, informative result.

Example T10 invocation once the config exists (**use ABSOLUTE paths for
--config/--out** — the driver resolves them against its own cwd; the L4
rehearsal failed twice on relative paths before this was pinned down):

```bash
bash scripts/m4_run.sh --config "$REPO/benchmarks/ispd25/synthetic_3x3_n2.json" \
    --k 16 --rho 0.0 --rtype grid --out "$REPO/results/m4/t10/h100_3x3__k16__grid__flat.json"
```

L4 rehearsal record (2026-08-18): the full runbook flow was exercised on the
3.1M `mempool_group` case (`--k 16 --rho 0.20 --seed 3000`) — environment
check printed the expected SKU + host-RAM warnings, run completed in 22.2 min,
`status=ok`, `legalization_status=success`, artifact
`results/m4/profile/rehearsal_group__k16__grid__oursM2_seed3000.json`.

---

## 5. Registered prediction — `results/m4/forecast/h100_prediction.json`

Frozen forecast, `status: "frozen"`, generator `scripts/m4_forecast.py`:

- **sha256**: `4eeea5f9132e1a06585ae57e38c6e45b5523ed31ae5454826ff4cff56e045284`

Do not edit this file. If any of §6.3 E5's frozen `sku` fields don't match the actual H100
machine, re-run `scripts/m4_forecast.py` and freeze a **new** file rather than editing this one
in place (spec: "須重登錄而非事後放寬").

### Comparison tool — `scripts/m4_check_prediction.py`

```bash
$PY scripts/m4_check_prediction.py ACTUAL.json \
    --prediction results/m4/forecast/h100_prediction.json \
    --out results/m4/t14/prediction_check.json
```

`ACTUAL.json` is the schema-v3 profile JSON `m4_run.sh` writes (`--out` path). The script
extracts a wall-time total (`t_total`, or `sum(phases[*].t_s)`, or `sum(t_read/t_gp/t_lg/t_eval)`,
whichever is present), compares it against `wall_time_forecast.total_fast_s`/`total_slow_s`,
compares `device_used_gb` against `gpu_memory_forecast.lower_bound_gb`/`upper_bound_gb`, and
reports `host_rss_forecast` as `not_applicable` (no band was frozen — see §7). Exit 0 if every
item with a real band is `"in"`; exit 1 if any is `"out"` — on any `"out"` verdict it also prints
the design draft's verbatim mandatory-disclosure sentence to stderr:

> 任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間

**This is a hypothesis test, not an interval refit** (spec §6.3: "檢定假設,不是檢定區間"). A
result outside the band is not a bug in the script — it is a finding that must be disclosed
verbatim per the rule above, never used to widen the band after the fact.

**Gap found and since closed:** an earlier draft of this runbook flagged that
`scripts/m4_check_prediction.py` only compared the **total** wall-time against
`[total_fast_s, total_slow_s]` and did not implement the per-phase `s_p^obs` protocol that
`h100_prediction.json`'s own `t14_check_protocol` field and spec §6.3's T14 row describe. **This
is now fixed** — the checker reads the frozen prediction's `wall_time_forecast.phases.*` (each
phase's already-scaled `t_l4_extrapolated_27m_s` and declared `[s_p_lo, s_p_hi]` band — the
count-ratio scaling is read back verbatim, never recomputed) and the actual run's T8a-schema
`phases.{read,gp,lg,eval}.t_s`, computes `s_p_obs = t_l4_extrapolated_27m_s / T_H100,p` per phase
(`check_phase_s_p`/`check_all_phases`), and derives a `mainline_hypothesis.confirmed` verdict
(`build_mainline_hypothesis_verdict`) that is `True` only when every phase's `s_p_obs` is within
50% relative error of its declared band **and** the total falls inside
`[total_fast_s, total_slow_s]` — matching this repo's actual (non-3-scenario) frozen forecast
structure, since `m4_forecast.py` never implemented spec §6.3's S-BW/S-SM/S-FP64 3-scenario table
either (see §5's `wall_time_forecast` fields above: one `[s_p_lo, s_p_hi]` band per phase, not
three named scenarios) — `[total_fast_s, total_slow_s]` *is* this repo's `[min, max]`. Memory stays
a separate check (`items.gpu_peak`, unchanged), per spec §6.3's own text keeping it out of the
wall-time mainline hypothesis. `check_prediction`'s output JSON now carries a `phase_checks` block
(per-phase `s_p_obs`/`relative_error`/`same_order`), a `mainline_hypothesis` block
(`confirmed`/`phases_same_order`/`total_time_in_band`/`reason`), and a
`no_new_prediction_interval_note` (spec §6.3(ii): "首跑不產生 prediction interval"); a `confirmed:
false` verdict now also drives `any_out_of_band`/the mandatory-disclosure sentence, same as the
existing total-time/GPU-peak items. A `confirmed: null` (missing per-phase `t_s` in the actual
run) is reported as indeterminate, never silently treated as a pass. Covered by
`tests/test_m4_handoff.py` (`test_scenario_all_pass_mainline_confirmed`,
`test_scenario_one_phase_out_of_order_disconfirms_mainline`,
`test_scenario_total_wall_time_out_of_band_disconfirms_mainline`,
`test_scenario_gpu_memory_out_of_band_independent_of_mainline`, plus the phase-level and
missing-data unit tests) — T14 can now run `scripts/m4_check_prediction.py` as the single source
of truth for the full protocol; no by-hand per-phase arithmetic is required.

---

## 6. T14 execution checklist

1. **No design decisions on the H100 machine** — every K/ρ/seed/gate value below is fixed from
   the L4 side; the H100 run only executes §4's command and records results.
2. Run §4's single command with the frozen K/ρ/seed/rtype combination(s) required by the M4 task
   list (`flat` = `--rho 0.0`, `ours@M2` = `--rho >0.0`).
3. **Per-phase comparison** (spec §6.3 T14 row, §5 above): `scripts/m4_check_prediction.py` now
   computes this automatically (`s_p_obs = t_l4_extrapolated_27m_s / T_H100,p` for each of
   `read`/`gp`/`lg`/`eval`, compared against the declared `[s_p_lo, s_p_hi]` in
   `h100_prediction.json`'s `wall_time_forecast.phases.<phase>`) — no manual arithmetic needed;
   the actual run's schema-v3 profile JSON just needs its `phases.<phase>.t_s` fields populated.
4. **Memory comparison**: compare `device_used_gb` against the analytic
   `gpu_memory_forecast.lower_bound_gb`/`upper_bound_gb` (not a statistical interval — a resolved
   analytic range).
5. **First-run `cuda_context` calibration**: `gpu_memory_forecast.cuda_context_gb` in the frozen
   prediction is currently a **placeholder** — the L4-measured `device_baseline_gb` (0.385 GB)
   substituted for the (unmeasured) H100 constant. T14's first run should measure the actual
   H100 `cuda_context` baseline and record it for any subsequent (re-)forecast; it is explicitly
   *not* refit into the already-frozen `h100_prediction.json`.
6. Run `scripts/m4_check_prediction.py` (§5) — it now emits the total-time/GPU-peak band
   verdicts, the full per-phase `s_p_obs` table (`phase_checks`), and the combined
   `mainline_hypothesis.confirmed` verdict for item 3 in a single pass; nothing further needs to
   be computed by hand.
7. Any out-of-band verdict (total time, GPU peak, a per-phase `s_p_obs`, or an overall
   `mainline_hypothesis.confirmed == false`) triggers the mandatory disclosure sentence in the
   report and a re-fit — never a retroactive interval widening.
8. **No new prediction interval is produced on the first run** — a calibrated interval needs at
   least a second H100 observation (spec: recorded as future work).

---

## 7. Known risks / open items going into T10

- **T0b host-RSS model is not identifiable — no prediction interval exists for host RSS.**
  `results/m4/scaling/model_fit.json`'s top-level `"identifiable": false`, with the explicit note:
  `"identifiable=false: this model must not be used for any extrapolation (design draft sec 7.1
  T0b acceptance)"`. Both sub-models it contains (`gp_memory_model`, `gp_runtime_model`) were
  rejected on the same condition-number/CI-width criterion (e.g. `gp_memory_model.
  rejection_reasons`: `"coefficient 'intercept' 95% CI relative half-width 3.037 >
  threshold 0.25"`). Consequently `h100_prediction.json`'s `host_rss_forecast` carries
  `model_status: "unfit"`, `interval: null`, and only a `point_estimate_gb` (168.85 GB, from a
  simple node-count scale-ratio proxy, not a validated model) — per spec §6.3 E5(iii)'s "否則"
  branch: analytic estimate only, no 95% PI. `m4_check_prediction.py`'s `check_host_rss` reports
  this item as `verdict: "not_applicable"` rather than grading it against a band that was never
  frozen.
- **27.7M host RAM is unmeasured.** No full DREAMPlace `PlaceDB.read` of the 3×3 array has ever
  been run to completion: `results/m4/bench/verify_group3x3_t6b.json`'s
  `checks.V1_dreamplace_readable` records `status: "infrastructure_blocked"` (DREAMPlace's
  `PlaceDB.read_pl` hardcoded `\w+` node-name regex cannot match this project's hierarchical
  (`./[]/`-containing) instance names — a structural incompatibility confirmed on `mempool_group`
  itself, independent of the tiler). A **separate** fix exists (commit `9b662a2`,
  letter-prefixed `.aux` alias symlinks) for a *different* lexer problem (digit-leading
  filenames), applied and read-smoke-tested on the 1×2/2×2 arrays only — see §4's discrepancy note
  above; it does **not** touch the hierarchical-name regex issue, and no `synthetic_3x3_n2.json`
  config or alias has been built/tested at all yet. Net effect: **the 27.7M case's actual host
  RSS under a real DREAMPlace read has never been measured**, on either fix.
- **L4 T9 spike peak, for reference/sanity-check on the H100 side**: the component-level
  full-lifetime spike (`ioplace/bench/spike_30m.py`, bypasses DREAMPlace's own `PlaceDB.read` via
  the T9 tiled-netlist cache — so this number is *not* subject to the V1 blocker above) measured
  `measured_peak_gb = 18.188034` GB (scenario A, K=32,
  `results/m4/profile/spike30m__A__k32.json`) and `measured_peak_gb = 18.630152702331543` GB
  (scenario B, K=32, `results/m4/profile/spike30m__B__k32.json`), both
  `feasibility_verdict: "feasible_l4_contract"` under the L4 19.5 GB budget contract. These are
  GPU device-memory peaks, not host RSS, but are the closest existing empirical data point for
  the 27.7M case's memory footprint until a real H100 run exists.

---

## Verification performed while writing this runbook

`bash scripts/m4_run.sh --help` (no positional args; environment/GPU checks not exercised, no
placement/route started) — confirmed the usage block reproduced verbatim in §4 above, exit 0.
