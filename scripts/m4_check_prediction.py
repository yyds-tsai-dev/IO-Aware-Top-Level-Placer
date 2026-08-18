#!/usr/bin/env python3
"""M4 T14 prediction checker (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 6.3 E5
/ T14 row): compares an actual H100 27.7M run (a schema-v3 profile JSON
from `ioplace.drivers.run_placement`) against the frozen `scripts/
m4_forecast.py` output (`results/m4/forecast/h100_prediction.json`),
item by item -- wall-time, GPU peak, host RSS.

This is a **hypothesis test, not an interval refit** (sec 6.3: "T14 首跑
的判定:檢定假設,不是檢定區間"). Falling outside a band is not itself an
error in this script; it is a finding that must be disclosed, verbatim,
per the design draft's own T14 row:

    "任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間"

(`MANDATORY_REFIT_DISCLOSURE_SENTENCE` below, copied character-for-
character so it can never silently drift from the source doc -- same
convention as `scripts/m4_report_lint.py`'s `MANDATORY_DISCLOSURE_SENTENCE`.)

Three top-level items, two of which have real [low, high] bands to test
against (wall-time, GPU peak); host RSS has no band (the design draft's
own sec 6.3 E5(iii) "否則只給解析估計" branch is what `m4_forecast.py`
freezes when the host-RSS model is unfit) and is reported as
`not_applicable` rather than silently graded against an interval no one
froze.

**Per-phase `s_p^obs` protocol (sec 6.3 T14 row / `h100_prediction.json`'s
own `t14_check_protocol` field, both verbatim):** the wall-time item above
only judges the *total*; this module additionally reads the frozen
prediction's `wall_time_forecast.phases.*` (each phase's already-scaled
`t_l4_extrapolated_27m_s` and declared `[s_p_lo, s_p_hi]` band -- the
count-ratio scaling itself is `m4_forecast.py`'s stage A and is never
recomputed here, only read back) and the actual run's T8a-schema
`phases.{read,gp,lg,eval}.t_s`, and computes, per phase:

    s_p_obs = t_l4_extrapolated_27m_s / T_H100,p

**Mainline-hypothesis verdict (sec 6.3 T14 row, verbatim):** "主線假設被
判為成立僅當每個 phase 的 s_p^obs 與宣告值同序(相對誤差 <= 50%)且總時間
落在 [total_fast_s, total_slow_s] 之間" -- `same_order` per phase is: the
observed `s_p_obs` inside its declared `[s_p_lo, s_p_hi]` band counts as
zero relative error; outside the band, relative error is measured against
the nearer edge, and must still be <= `PHASE_RELATIVE_ERROR_THRESHOLD`
(0.5). Memory is judged separately, directly against the analytic
`[lower_bound_gb, upper_bound_gb]` bound (already the existing `gpu_peak`
item -- sec 6.3's own text keeps memory as its own sentence, not folded
into the wall-time mainline hypothesis). **No new prediction interval is
produced by a first H100 run** (sec 6.3(ii)) -- this module only tests the
already-frozen bands, never widens or refits them.

CLI:
    m4_check_prediction.py ACTUAL.json --prediction h100_prediction.json [--out OUT.json]

Exit 0 if every item with a real band is "in" AND the mainline hypothesis
is not explicitly disconfirmed; exit 1 if any is "out" (the mandatory
disclosure sentence is also printed to stderr in that case, so a caller
that only checks the exit code still sees it in logs).
"""
import argparse
import json
import sys
from pathlib import Path

# design draft T14 row, verbatim.
MANDATORY_REFIT_DISCLOSURE_SENTENCE = "任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間"

# design draft sec 6.3 T14 row, verbatim: "相對誤差 <= 50%".
PHASE_RELATIVE_ERROR_THRESHOLD = 0.5

# design draft sec 6.3 E5(ii), verbatim: "首跑不產生 prediction interval";
# a calibrated interval needs >= 2 H100 observations (future work).
FIRST_RUN_NO_INTERVAL_NOTE = (
    "design draft sec 6.3 E5: 首跑不產生 prediction interval;要有校準區間,"
    "至少需要第二次以上的 H100 觀測(記入 future work)"
)


def _extract_wall_time_s(actual):
    """Schema-v3 profile JSONs (see `ioplace/drivers/run_placement.py`,
    `results/m4/t8b/cluster__k16__grid__flat.json`) do not all carry a
    literal `t_total` field -- fall back through what they do carry, in
    order of preference: an explicit `t_total`, the sum of `phases[*].t_s`,
    then the sum of the four named `t_read`/`t_gp`/`t_lg`/`t_eval` fields."""
    if actual.get("t_total"):
        return float(actual["t_total"])
    phases = actual.get("phases")
    if phases:
        return float(sum(p.get("t_s", 0.0) for p in phases.values()))
    keys = ("t_read", "t_gp", "t_lg", "t_eval")
    if all(k in actual for k in keys):
        return float(sum(actual[k] for k in keys))
    raise KeyError("m4_check_prediction: could not extract a wall-time total from the "
                    "actual run JSON (need t_total, phases[*].t_s, or all of "
                    "t_read/t_gp/t_lg/t_eval)")


def _judge_band(actual_value, low, high):
    ok = low <= actual_value <= high
    return {
        "actual": actual_value,
        "band_low": low,
        "band_high": high,
        "in_band": ok,
        "verdict": "in" if ok else "out",
    }


def check_wall_time(actual, prediction):
    wtf = prediction["wall_time_forecast"]
    actual_s = _extract_wall_time_s(actual)
    result = _judge_band(actual_s, wtf["total_fast_s"], wtf["total_slow_s"])
    result["unit"] = "seconds"
    return result


def check_gpu_peak(actual, prediction):
    gmf = prediction["gpu_memory_forecast"]
    if actual.get("device_used_gb") is None:
        raise KeyError("m4_check_prediction: actual run JSON missing device_used_gb")
    actual_gb = float(actual["device_used_gb"])
    result = _judge_band(actual_gb, gmf["lower_bound_gb"], gmf["upper_bound_gb"])
    result["unit"] = "GB"
    result["caveat"] = (
        "actual compares whole-device device_used_gb (includes CUDA context / "
        "allocator fragmentation / any co-resident process) against a component-level "
        "resident-tensor analytic bound; this unit-of-comparison gap is disclosed here, "
        "not papered over"
    )
    return result


def check_host_rss(actual, prediction):
    hrf = prediction["host_rss_forecast"]
    if actual.get("host_peak_rss_gb") is None:
        raise KeyError("m4_check_prediction: actual run JSON missing host_peak_rss_gb")
    actual_gb = float(actual["host_peak_rss_gb"])
    interval = hrf.get("interval")
    if interval:
        result = _judge_band(actual_gb, interval["low"], interval["high"])
    else:
        result = {
            "actual": actual_gb,
            "point_estimate_gb": hrf.get("point_estimate_gb"),
            "in_band": None,
            "verdict": "not_applicable",
            "reason": "host RSS model unfit -- no validated interval was frozen "
                      "(design draft sec 6.3 E5(iii)'s 否則 branch); point estimate "
                      "reported for context only, not graded",
        }
    result["unit"] = "GB"
    return result


def _phase_relative_error(s_p_obs, s_p_lo, s_p_hi):
    """0.0 if `s_p_obs` is inside the declared `[s_p_lo, s_p_hi]` band;
    otherwise the relative error to the nearer edge (sec 6.3 T14 row's
    "同序(相對誤差 <= 50%)" -- in-band always counts as same-order, outside
    the band is only same-order if within 50% relative error of the edge
    it overshot/undershot)."""
    if s_p_lo <= s_p_obs <= s_p_hi:
        return 0.0
    edge = s_p_lo if s_p_obs < s_p_lo else s_p_hi
    if edge == 0:
        return float("inf")
    return abs(s_p_obs - edge) / edge


def check_phase_s_p(name, prediction_phase, actual_t_s):
    """One phase's `s_p_obs = t_l4_extrapolated_27m_s / T_H100,p` against
    its declared `[s_p_lo, s_p_hi]` band (`prediction_phase` is
    `h100_prediction.json`'s `wall_time_forecast.phases.<name>` entry --
    the scaled L4 time and the band are read back verbatim, never
    recomputed here)."""
    t_extrap = prediction_phase["t_l4_extrapolated_27m_s"]
    s_lo = prediction_phase["s_p_lo"]
    s_hi = prediction_phase["s_p_hi"]
    if actual_t_s is None or actual_t_s <= 0:
        raise ValueError(
            f"m4_check_prediction: phase {name!r} actual t_s must be > 0, got {actual_t_s!r}")
    s_p_obs = t_extrap / actual_t_s
    relative_error = _phase_relative_error(s_p_obs, s_lo, s_hi)
    same_order = relative_error <= PHASE_RELATIVE_ERROR_THRESHOLD
    return {
        "phase": name,
        "t_l4_extrapolated_27m_s": t_extrap,
        "s_p_lo": s_lo,
        "s_p_hi": s_hi,
        "t_h100_actual_s": actual_t_s,
        "s_p_obs": s_p_obs,
        "relative_error": relative_error,
        "same_order": same_order,
        "verdict": "in" if same_order else "out",
    }


def check_all_phases(actual, prediction):
    """Runs `check_phase_s_p` for every phase in the frozen prediction's
    `wall_time_forecast.phases`, reading the actual run's T8a-schema
    `phases.{read,gp,lg,eval}.t_s`. Degrades to `"available": False`
    (never raises) when the actual run JSON carries no `phases` block at
    all -- older/minimal actual JSONs (e.g. the total-only fixtures this
    module's own tests used before this per-phase protocol existed) are
    still gradeable on the total-wall-time/GPU-peak/host-RSS items above,
    just not on the per-phase mainline hypothesis."""
    pred_phases = prediction.get("wall_time_forecast", {}).get("phases")
    actual_phases = actual.get("phases")
    if not pred_phases:
        return {
            "available": False,
            "reason": "frozen prediction's wall_time_forecast has no 'phases' block -- "
                      "per-phase s_p^obs check skipped",
            "phases": {},
        }
    if not actual_phases:
        return {
            "available": False,
            "reason": "actual run JSON has no 'phases' block (T8a schema "
                      "phases.{read,gp,lg,eval}.t_s) -- per-phase s_p^obs check skipped",
            "phases": {},
        }
    out = {}
    for name, pred_phase in pred_phases.items():
        actual_phase = actual_phases.get(name)
        if not actual_phase or actual_phase.get("t_s") is None:
            out[name] = {
                "phase": name,
                "available": False,
                "reason": f"actual phases.{name}.t_s missing",
            }
            continue
        out[name] = check_phase_s_p(name, pred_phase, float(actual_phase["t_s"]))
    return {"available": True, "phases": out}


def build_mainline_hypothesis_verdict(phase_check, wall_time_check):
    """Design draft sec 6.3 T14 row, verbatim: 主線假設被判為成立僅當每個
    phase 的 s_p^obs 與宣告值同序(相對誤差 <= 50%)且總時間落在
    [total_fast_s, total_slow_s] 之間. `confirmed` is `True`/`False` when
    fully evaluable, `None` when some phase's actual `t_s` is missing (an
    indeterminate verdict is not silently treated as a pass)."""
    total_in_band = wall_time_check["in_band"]

    if not phase_check.get("available"):
        return {
            "confirmed": None,
            "phases_same_order": None,
            "total_time_in_band": total_in_band,
            "reason": phase_check.get("reason", "per-phase data unavailable"),
        }

    phases = phase_check["phases"]
    missing = sorted(n for n, p in phases.items() if "same_order" not in p)
    same_order_flags = {n: p["same_order"] for n, p in phases.items() if "same_order" in p}
    all_same_order = all(same_order_flags.values()) if same_order_flags else None

    if missing:
        return {
            "confirmed": None,
            "phases_same_order": all_same_order,
            "total_time_in_band": total_in_band,
            "reason": f"phase(s) missing actual t_s, cannot fully evaluate: {missing}",
        }

    if all_same_order and total_in_band:
        return {
            "confirmed": True,
            "phases_same_order": True,
            "total_time_in_band": True,
            "reason": "every phase's s_p_obs is within 50% relative error of its declared "
                      "band, and total wall-time falls in [total_fast_s, total_slow_s]",
        }

    reasons = []
    out_of_order = sorted(n for n, ok in same_order_flags.items() if not ok)
    if out_of_order:
        reasons.append(f"phase(s) out of declared s_p order (>50% relative error): "
                        f"{out_of_order}")
    if not total_in_band:
        reasons.append("total wall-time outside [total_fast_s, total_slow_s]")
    return {
        "confirmed": False,
        "phases_same_order": all_same_order,
        "total_time_in_band": total_in_band,
        "reason": "; ".join(reasons),
    }


def check_prediction(actual, prediction):
    items = {
        "wall_time": check_wall_time(actual, prediction),
        "gpu_peak": check_gpu_peak(actual, prediction),
        "host_rss": check_host_rss(actual, prediction),
    }
    phase_check = check_all_phases(actual, prediction)
    mainline = build_mainline_hypothesis_verdict(phase_check, items["wall_time"])

    # sec 6.3 T14 row's mainline hypothesis is wall-time-only (memory is
    # judged separately via items["gpu_peak"] above) -- an explicit
    # `confirmed is False` (not `None`, which means "not evaluable from
    # this actual JSON") is itself an out-of-band finding.
    any_out = any(v.get("verdict") == "out" for v in items.values()) or mainline["confirmed"] is False
    result = {
        "items": items,
        "phase_checks": phase_check,
        "mainline_hypothesis": mainline,
        "any_out_of_band": any_out,
        "no_new_prediction_interval_note": FIRST_RUN_NO_INTERVAL_NOTE,
    }
    if any_out:
        result["mandatory_disclosure"] = MANDATORY_REFIT_DISCLOSURE_SENTENCE
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("actual", help="path to the actual H100 run's schema-v3 profile JSON")
    ap.add_argument("--prediction", required=True,
                     help="path to the frozen h100_prediction.json")
    ap.add_argument("--out", default=None,
                     help="optional path to also write the judgement JSON to")
    args = ap.parse_args(argv)

    with open(args.actual) as f:
        actual = json.load(f)
    with open(args.prediction) as f:
        prediction = json.load(f)

    result = check_prediction(actual, prediction)
    text = json.dumps(result, indent=1)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text + "\n")

    print(text)

    if result["any_out_of_band"]:
        print(MANDATORY_REFIT_DISCLOSURE_SENTENCE, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
