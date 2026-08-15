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

Three items, two of which have real [low, high] bands to test against
(wall-time, GPU peak); host RSS has no band (the design draft's own sec
6.3 E5(iii) "否則只給解析估計" branch is what `m4_forecast.py` freezes
when the host-RSS model is unfit) and is reported as `not_applicable`
rather than silently graded against an interval no one froze.

CLI:
    m4_check_prediction.py ACTUAL.json --prediction h100_prediction.json [--out OUT.json]

Exit 0 if every item with a real band is "in"; exit 1 if any is "out"
(the mandatory disclosure sentence is also printed to stderr in that
case, so a caller that only checks the exit code still sees it in logs).
"""
import argparse
import json
import sys
from pathlib import Path

# design draft T14 row, verbatim.
MANDATORY_REFIT_DISCLOSURE_SENTENCE = "任一落外 ⇒ 在報告發表重擬合模型與歸因,不得事後放寬區間"


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


def check_prediction(actual, prediction):
    items = {
        "wall_time": check_wall_time(actual, prediction),
        "gpu_peak": check_gpu_peak(actual, prediction),
        "host_rss": check_host_rss(actual, prediction),
    }
    any_out = any(v.get("verdict") == "out" for v in items.values())
    result = {"items": items, "any_out_of_band": any_out}
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
