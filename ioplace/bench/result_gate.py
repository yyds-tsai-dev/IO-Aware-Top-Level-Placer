"""M4 sec 7.0 RESULT GATE (design draft `docs/superpowers/specs/2026-08-13-
m4-scale-up-design-draft.md` sec 7.0/2.2, Codex finding #16): the common
acceptance gate every M4 experiment task's artifact is checked against, so
"produced a JSON" stops being conflated with "produced a *valid*" one.
Also T9's home for the sec 2.2 three-state feasibility rule (`feasibility_
verdict`) and sec 1.4 B3's hardware-contract budget assert
(`assert_budget`) -- both are used by T2b's `probe_lifetime_gate.py` today
and by T9's 30M spike later, so they live here rather than duplicated per
probe.

Schema is parametrized (sec 7.0 rule 2/"schema"): `check(record, schema=
...)` accepts `PROFILE_SCHEMA_FIELDS` (profile.py's existing per-run
profile schema), or this module's own `LIFETIME_SCHEMA_FIELDS` (T2b's
`lifetime_<case>__k<K>__<arm>.json`) / `SPIKE_SCHEMA_FIELDS` (T9's 30M
component-level spike). Rule 6 ("non-empty") is schema-dependent -- a
lifetime/spike record has no `iter_ms` list to check the length of, so it
maps onto the semantically closest field each of those schemas actually
has (`n_callbacks_with_active >= 3` / `n_interleaved_iters >= 3`,
respectively -- both stand in for "the workload interleaved enough
iterations under memory pressure to be a meaningful measurement", the same
thing rule 6 was checking for a profile record's `iter_ms`).
"""

import math

# sec 1: `mem_get_info()` reports 21.95 GiB free on this L4; budgets are
# always compared against 21.7 GiB (21.95 GiB minus ~0.2 GiB CUDA context).
L4_DEVICE_GIB = 21.7

# sec 1.4 B3 (Codex #13): the 8.0 GB M2 10M/8GB contract, parametrized so a
# 30M-scale spike isn't misjudged against a threshold sized for 10M -- but
# a CLI `--budget-gb` may never exceed what the run's own hardware contract
# permits. `_m2_10m_contract` is keyed independently of GPU name since it's
# a fixed *task* contract (M2's 10M/8GB), not a hardware ceiling.
HW_BUDGET_GB = {
    "NVIDIA L4": 19.5,                 # 0.9 * 21.7 GiB (sec 2.2's L4 contract)
    "NVIDIA H100 80GB HBM3": 72.0,     # sec 1.4 B3
    # NVL host reports 95,830 MiB = 93.58 GiB. Round 90% down to 84 GiB;
    # this is a separate contract, not a change to the historical SXM/L4 caps.
    "NVIDIA H100 NVL": 84.0,
    "_m2_10m_contract": 8.0,           # sec 1.4 B3 / spike_10m.py's original default
}

# design draft sec 6.2 (profile.py's existing per-run schema).
from ioplace.profile import PROFILE_SCHEMA_FIELDS  # noqa: E402  (re-exported for convenience)

# T2b's lifetime record schema (design draft sec 7.1 T2b row +
# `probe_lifetime_gate.py`'s output-schema list): `record_kind` lets the
# report linter (T11's `m4_report_lint.py`) exclude these from any
# quality/scaling table by construction.
LIFETIME_SCHEMA_FIELDS = (
    "record_kind", "case", "K", "rtype", "arm", "benchmark_kind",
    "budget_gb", "budget_source",
    "experiment_status", "workload_status", "feasibility_verdict",
    "device_baseline_gb", "exclusivity_evidence", "host_mem_available_gb_at_start",
    "buffers", "phases", "totals",
    "n_callbacks_with_active", "n_evals_while_active", "of_on", "gp_iterations_run",
    "run_id", "repo_commit", "dp_commit", "command", "hostname", "gpu_name",
    "seed", "input_sha256",
)

# T9's 30M component-level spike record schema (design draft T9 row's
# explicit field list: "experiment_status/workload_status/
# feasibility_verdict/measured_peak_gb/budget_gb/budget_source/
# baseline_reserved_gb/resident_gb/resident_lower_bound_gb/
# max_phase_transient_gb/retry_count").
SPIKE_SCHEMA_FIELDS = (
    "record_kind", "case", "K", "benchmark_kind",
    "experiment_status", "workload_status", "feasibility_verdict",
    "measured_peak_gb", "budget_gb", "budget_source", "baseline_reserved_gb",
    "resident_gb", "resident_lower_bound_gb", "max_phase_transient_gb",
    "retry_count", "n_interleaved_iters",
    "run_id", "repo_commit", "dp_commit", "command", "hostname", "gpu_name",
    "seed", "input_sha256", "device_baseline_gb",
)

_PROVENANCE_FIELDS = ("run_id", "repo_commit", "dp_commit", "command", "hostname",
                      "gpu_name", "seed", "input_sha256", "device_baseline_gb")

_VALID_WORKLOAD_STATUS = (None, "completed", "oom", "crashed")
_VALID_BENCHMARK_KIND = (None, "real", "synthetic")


def assert_budget(gpu_name, budget_gb, source):
    """sec 1.4 B3: `budget_gb` may never exceed the hardware-contract
    table -- an assert failure, not a soft warning, when a CLI
    `--budget-gb` flag tries to raise the L4/H100/M2-10M budget past what
    the manifest's own contract permits. `gpu_name` is carried through
    into the failure message only (provenance context); the pass/fail test
    is purely `budget_gb <= HW_BUDGET_GB[source]`."""
    assert source in HW_BUDGET_GB, (
        f"assert_budget: unknown budget_source {source!r}, must be one of "
        f"{sorted(HW_BUDGET_GB)}")
    cap = HW_BUDGET_GB[source]
    assert budget_gb <= cap, (
        f"assert_budget: budget_gb={budget_gb} exceeds the {source!r} "
        f"hardware contract ({cap} GB) on gpu_name={gpu_name!r} -- CLI flags "
        "may not raise the gate past the contract (design draft sec 1.4 B3)")


def feasibility_verdict(experiment_status, workload_status, peak_gb, budget_gb,
                        oom_repeats, resident_lower_bound_gb):
    """sec 2.2's three-state feasibility rule, transcribed literally:

      - `feasible_l4_contract`: experiment_status=="ok" and workload_status
        =="completed" and peak_gb <= budget_gb.
      - `infeasible_l4_contract`: experiment_status=="ok" and *any* of:
          (a) workload_status=="completed" but peak_gb > budget_gb;
          (b) workload_status=="oom" and reproduced twice under the same
              contract (oom_repeats >= 2);
          (c) analytic lower bound resident_lower_bound_gb > 21.7 GiB.
      - `invalid_measurement`: experiment_status != "ok" -- **and** (sec
        2.2's own "不存在'永遠未定'" -- no 4th "undetermined" state exists)
        any experiment_status=="ok" combination that satisfies none of the
        above (an as-yet-unreproduced OOM, or a crash): sec 2.2's table
        does not name a 4th literal for this, so it is folded into
        `invalid_measurement` too -- the honest reading of "not yet a
        *reproduced* terminal call" is "not yet a valid measurement of the
        thing sec 2.2's other two states are about", which is exactly what
        `invalid_measurement`'s own "至多重試兩次" retry clause is for.

    Exactly one of the three literals is always returned -- six ways to
    reach one of them (this function's "六分支": feasible; infeasible (a);
    infeasible (b); infeasible (c); invalid via experiment_status != "ok";
    invalid via the ok-but-unreproduced-or-crashed fallback), one test per
    branch (design draft sec 7.1 T2b's acceptance: "verdict 表六分支全覆蓋")."""
    if experiment_status != "ok":
        return "invalid_measurement"
    if workload_status == "completed" and peak_gb is not None and peak_gb <= budget_gb:
        return "feasible_l4_contract"
    cond_a = (workload_status == "completed" and peak_gb is not None
             and peak_gb > budget_gb)
    cond_b = (workload_status == "oom" and oom_repeats is not None and oom_repeats >= 2)
    cond_c = resident_lower_bound_gb is not None and resident_lower_bound_gb > L4_DEVICE_GIB
    if cond_a or cond_b or cond_c:
        return "infeasible_l4_contract"
    return "invalid_measurement"


def _non_finite_fields(obj, prefix=""):
    """Recursively finds every NaN/Inf float anywhere in `obj` (sec 7.0
    rule 3: "所有數值欄位非 NaN/Inf") -- descends into dicts and lists
    (a `phases`/`buffers`/`totals` record is nested), returning the list
    of dotted/indexed paths so a failure is actionable, not just a bool."""
    bad = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(prefix or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(_non_finite_fields(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(_non_finite_fields(v, f"{prefix}[{i}]"))
    return bad


def _rule6_nonempty(record, schema):
    """sec 7.0 rule 6, schema-dependent ("對 lifetime/spike 記錄映射為
    n_interleaved_iters >= 3 或對應語意")."""
    if schema is LIFETIME_SCHEMA_FIELDS:
        return (record.get("n_callbacks_with_active") or 0) >= 3
    if schema is SPIKE_SCHEMA_FIELDS:
        return (record.get("n_interleaved_iters") or 0) >= 3
    # Default / PROFILE_SCHEMA_FIELDS semantics, literal sec 7.0 rule 6:
    # len(iter_ms) >= 0.9*n_iter_expected and n_nets > 0 and t_total > 0.
    # PROFILE_SCHEMA_FIELDS itself carries no n_iter_expected field --
    # callers that have one (e.g. run_placement_io's gp_iteration_budget)
    # may put it in the record under that name; its absence (None/0) is
    # treated as "no expectation to check against", not a failure.
    iter_ms = record.get("iter_ms") or []
    n_expected = record.get("n_iter_expected") or record.get("gp_iteration_budget")
    ok_iters = not n_expected or len(iter_ms) >= 0.9 * n_expected
    n_nets = record.get("n_nets") or 0
    t_total = record.get("t_total") or 0
    return ok_iters and n_nets > 0 and t_total > 0


def check(record, schema=PROFILE_SCHEMA_FIELDS):
    """sec 7.0's six RESULT GATE rules (minus rule 5, uniqueness, which is
    a cross-record/table check -- see `check_table_uniqueness`). Returns
    `(ok: bool, reasons: list[str])` rather than raising: a RESULT GATE
    failure is a fact about the *artifact* a caller needs to report (T11's
    linter, or this module's own pytest -- "對故意損壞的樣本必須全部攔下"
    is about *detecting* every deliberately-broken sample, which needs the
    specific reasons, not just a pass/fail bit)."""
    reasons = []

    # 1. status (two-layer: experiment_status/workload_status).
    if record.get("experiment_status") != "ok":
        reasons.append(f"experiment_status != 'ok' (got {record.get('experiment_status')!r})")
    ws = record.get("workload_status")
    if ws not in _VALID_WORKLOAD_STATUS:
        reasons.append(f"workload_status invalid: {ws!r} (must be one of "
                       f"{[s for s in _VALID_WORKLOAD_STATUS if s is not None]})")

    # 2. schema.
    missing = [f for f in schema if f not in record]
    if missing:
        reasons.append(f"missing schema fields: {missing}")
    bk = record.get("benchmark_kind")
    if bk not in _VALID_BENCHMARK_KIND:
        reasons.append(f"benchmark_kind invalid: {bk!r} (must be 'real' or 'synthetic')")

    # 3. finite.
    non_finite = _non_finite_fields(record)
    if non_finite:
        reasons.append(f"non-finite numeric fields: {non_finite}")

    # 4. provenance -- unconditional (sec 7.0 rule 4 applies to every
    # RESULT GATE'd artifact regardless of which schema flavor it is;
    # PROFILE_SCHEMA_FIELDS itself doesn't happen to list these field names
    # -- provenance is assembled separately, e.g. by run_placement's
    # _t8a_provenance -- so this must not be gated on "is the field in
    # `schema`", or it would silently no-op for that schema).
    missing_prov = [f for f in _PROVENANCE_FIELDS if record.get(f) in (None, "", {})]
    if missing_prov:
        reasons.append(f"missing/empty provenance fields: {missing_prov}")

    # 5. uniqueness: see check_table_uniqueness (cross-record, not here).

    # 6. non-empty.
    if not _rule6_nonempty(record, schema):
        reasons.append("rule 6 (non-empty) failed")

    return (not reasons, reasons)


def check_table_uniqueness(run_ids):
    """sec 7.0 rule 5: no two cells in the same results table may cite the
    same `run_id`. Returns `(ok: bool, duplicate_run_ids: list[str])`."""
    seen = set()
    dupes = set()
    for rid in run_ids:
        if rid in seen:
            dupes.add(rid)
        seen.add(rid)
    return (not dupes, sorted(dupes))
