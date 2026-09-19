"""Select legal IO improvements under an explicit wirelength budget.

Budgets are fractions (0.05 means +5%). HPWL, evaluator tree length and
routed wirelength are distinct measurements; missing evidence fails closed.
"""
import math


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def assess(baseline, candidate, *, hpwl_budget=0.05, routed_budget=None):
    for budget in (hpwl_budget, routed_budget):
        if budget is not None and (not _finite_nonnegative(budget)):
            raise ValueError("wirelength budgets must be finite nonnegative fractions")
    reasons = []
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if (result.get("workload_status") != "completed"
                or result.get("legalization_status") != "success"
                or result.get("num_unplaced_cells") != 0
                or result.get("stop_overflow_reached") is not True):
            reasons.append(label + "_not_legal_and_converged")
        for key in ("hpwl", "io_count", "ft_count"):
            if not _finite_nonnegative(result.get(key)):
                reasons.append(label + "_invalid_" + key)
    # Same config, partition and RNG regime are required for a paired comparison.
    for key in ("config", "k", "rtype", "seed", "dp_seed", "det", "benchmark_kind"):
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            reasons.append("unmatched_" + key)
    delta = {}
    for key, budget in (("hpwl", hpwl_budget), ("routed_wirelength", routed_budget)):
        if key == "routed_wirelength" and budget is None:
            continue
        b, c = baseline.get(key), candidate.get(key)
        if not _finite_nonnegative(b) or not _finite_nonnegative(c):
            reasons.append("missing_or_invalid_" + key)
            continue
        delta[key + "_delta_pct"] = 100 * (c / b - 1) if b else (0 if c == 0 else None)
        if c > b * (1 + budget):
            reasons.append(key + "_budget_exceeded")
    b, c = baseline.get("io_count"), candidate.get("io_count")
    if _finite_nonnegative(b) and _finite_nonnegative(c):
        delta["io_delta_pct"] = 100 * (c / b - 1) if b else None
        if c >= b:
            reasons.append("io_not_reduced")
    return {"accepted": not reasons, "reasons": reasons, **delta,
            "hpwl_budget": hpwl_budget, "routed_budget": routed_budget,
            "routing_verified": routed_budget is not None and not any(
                "routed_wirelength" in reason for reason in reasons)}


def select(baseline, candidates, *, hpwl_budget=0.05, routed_budget=None):
    """Return candidate index, or None to retain the matched baseline."""
    checks = [assess(baseline, c, hpwl_budget=hpwl_budget,
                     routed_budget=routed_budget) for c in candidates]
    eligible = [i for i, check in enumerate(checks) if check["accepted"]]
    chosen = min(eligible, key=lambda i: (candidates[i]["io_count"],
                 candidates[i]["hpwl"], candidates[i]["ft_count"], i)) if eligible else None
    return {"selected_index": chosen, "improved": chosen is not None,
            "checks": checks, "hpwl_budget": hpwl_budget,
            "routed_budget": routed_budget}
