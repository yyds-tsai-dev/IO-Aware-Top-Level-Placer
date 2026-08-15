"""M4 T6b/T7b count freeze (2026-08-15 T6 holdout adjudication doc
`docs/results/2026-08-15-m4-t6-holdout-adjudication.md` sec 6.1: "T6b:
照常,用 N2 臂"; design draft `docs/superpowers/specs/2026-08-13-m4-scale-up-
design-draft.md` sec 2.1/sec 8/Task table's T6b/T7b rows).

The design draft's sec 2.1 benchmark ladder carried `g`/`gp` (glue nets/
pins) as *placeholders* -- "所有下游記憶體/runtime 帳在 T6b(<=12.3M)/
T7b(27.7M) count freeze 之前都是低估" -- because at the time it was written,
no glue had actually been generated yet. `build_t6_arrays.py` has since
produced real 1x2/2x2/3x3 N2 arrays with real manifests; this module reads
those manifests' `base`/`glue` sections (never re-derives a count from
first principles) and freezes the *actual* `g`/`gp` values plus the int32/
int64 headroom checks sec 2.1's Codex #4 finding calls for, replacing every
`g`/`gp` placeholder downstream is supposed to cite from here on (T7b's
`citation_declaration` field, sec 6.1: "T9 與 E5 的所有數字一律引用本檔").

Two outputs, one per design-draft task:
  `results/m4/scaling/count_freeze_small.json`  (T6b, <=12.3M: 1x2/2x2)
  `results/m4/scaling/count_freeze_30m.json`     (T7b, 27.7M: 3x3)

Both carry the adjudication doc sec 5.4 provenance tags (T6's holdout FAIL
does not block count freeze -- manifest counts are file facts, sec 6.1:
"manifest 計數是檔案事實,`matches_manifest` 為 true ... 不受 verdict 影響"
-- but every downstream consumer must still see the synthetic/uncertified
labels).

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.count_freeze
"""
import argparse
import json
import os

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DEFAULT_ARRAYS_ROOT = os.path.join(REPO, "results/m4/bench/arrays")
DEFAULT_OUT_SMALL = os.path.join(REPO, "results/m4/scaling/count_freeze_small.json")
DEFAULT_OUT_30M = os.path.join(REPO, "results/m4/scaling/count_freeze_30m.json")

ADJUDICATION_DOC = "docs/results/2026-08-15-m4-t6-holdout-adjudication.md"
DESIGN_DRAFT = "docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md"

INT32_MAX = 2 ** 31 - 1          # design draft sec 2.1's Codex #4 headroom line
INT64_MAX = 2 ** 63 - 1
COMPOSITE_KEY_MULTIPLIER = 64    # composite key = net_id * 64 + rid
HEADROOM_WARNING_THRESHOLD = 0.20

# sec 5.4's provenance tags, verbatim -- every T6b/T7b output carries these.
PROVENANCE_TAGS = {
    "benchmark_kind": "synthetic",
    "glue_provenance": "recipe_A_prime_uncertified",
    "generator_verified": False,
    "t6_verdict": "FAIL",
    "t6_fail_mechanism": ["H1_band_estimator_mismatch", "H6_1x1_leg_zero_glue_baseline"],
    "normalization": "n2",
    "n2_rule": "sinkhorn",
    "quality_claims_prohibited": True,
}


def _load_manifest(arrays_root, shape, norm):
    path = os.path.join(arrays_root, f"{shape}_{norm}", f"{shape}_{norm}.manifest.json")
    with open(path) as f:
        return json.load(f)


def _headroom_frac(used, capacity):
    """(capacity - used) / used -- matches design draft sec 2.1's own
    worked example ("3×3 base 已用掉 31,535,928,只剩 2,018,503(6.40%)給
    glue"): 2,018,503 / 31,535,928 = 6.40%, i.e. headroom is reported
    relative to the count already committed (how much further,
    proportionally, the count could still grow), not relative to raw
    capacity. `None` if `used <= 0` (undefined)."""
    if used is None or used <= 0:
        return None
    return (capacity - used) / used


def _int_headroom(name, count, capacity, capacity_label):
    headroom = _headroom_frac(count, capacity)
    return {
        "field": name, "count": count, "capacity": capacity, "capacity_label": capacity_label,
        "fits": count <= capacity,
        "headroom_frac": headroom,
        "headroom_pct": None if headroom is None else headroom * 100.0,
        "warning_headroom_lt_20pct": bool(headroom is not None and headroom < HEADROOM_WARNING_THRESHOLD),
    }


def int_headroom_report(n_nets, n_pins, composite_key_multiplier=COMPOSITE_KEY_MULTIPLIER):
    """design draft sec 2.1's Codex #4 int32 headroom warning line,
    generalized into a reusable {nets, pins, composite_key} x {int32,
    int64} report. `warning_headroom_lt_20pct=True` on any entry means the
    "任何 headroom < 20% 的 int32 決策必須回退 int64" rule (T6b task row)
    is triggered for that field."""
    composite_key_capacity_int32 = (2 ** 31 - 1) // composite_key_multiplier
    composite_key_capacity_int64 = (2 ** 63 - 1) // composite_key_multiplier
    return {
        "formula": "design draft sec 2.1 Codex #4: composite key = net_id*64+rid; "
                   "n_nets <= floor((2**31-1)/64) = 33,554,431 for a signed int32 composite "
                   "key; headroom_frac = (capacity - count) / count (relative to the count "
                   "already committed, matching sec 2.1's own worked 6.40% example)",
        "nets": {
            "int32": _int_headroom("n_nets", n_nets, INT32_MAX, "2**31-1"),
            "int64": _int_headroom("n_nets", n_nets, INT64_MAX, "2**63-1"),
        },
        "pins": {
            "int32": _int_headroom("n_pins", n_pins, INT32_MAX, "2**31-1"),
            "int64": _int_headroom("n_pins", n_pins, INT64_MAX, "2**63-1"),
        },
        "composite_key_net_times_64_plus_rid": {
            "int32": _int_headroom("n_nets (composite key capacity)", n_nets,
                                    composite_key_capacity_int32, "floor((2**31-1)/64)"),
            "int64": _int_headroom("n_nets (composite key capacity)", n_nets,
                                    composite_key_capacity_int64, "floor((2**63-1)/64)"),
        },
    }


# ---------------------------------------------------------------------------
# T6b: count_freeze_small.json (1x2 / 2x2)
# ---------------------------------------------------------------------------

def build_count_freeze_small(arrays_root=DEFAULT_ARRAYS_ROOT):
    rows = {}
    rejected_alternative = {}
    for shape, scale_label in (("1x2", "6.2M"), ("2x2", "12.3M")):
        m_n2 = _load_manifest(arrays_root, shape, "n2")
        m_n1 = _load_manifest(arrays_root, shape, "n1")
        base_nets, base_pins = m_n2["base"]["n_nets"], m_n2["base"]["n_pins"]
        g_n2, gp_n2 = m_n2["glue"]["n_nets"], m_n2["glue"]["n_pins"]
        g_n1, gp_n1 = m_n1["glue"]["n_nets"], m_n1["glue"]["n_pins"]
        total_nets, total_pins = base_nets + g_n2, base_pins + gp_n2

        rows[shape] = {
            "scale_label": scale_label,
            "base_n_nets": base_nets, "base_n_pins": base_pins,
            "g_n2": g_n2, "gp_n2": gp_n2,
            "total_n_nets": total_nets, "total_n_pins": total_pins,
            "glue_share_nets": g_n2 / total_nets, "glue_share_pins": gp_n2 / total_pins,
            "int_headroom": int_headroom_report(total_nets, total_pins),
            "source_manifest": os.path.join(arrays_root, f"{shape}_n2", f"{shape}_n2.manifest.json"),
        }
        rejected_alternative[shape] = {
            "normalization": "n1", "reason": "N1 rejected by T6 adjudication (selected_normalization=n2); "
                                              "listed for scale-of-choice comparison only, sec 6.1: "
                                              "\"N1 的計數 ... 以 rejected_alternative 欄位併列,供讀者檢核"
                                              "正規化選擇的量級後果\"",
            "g_n1": g_n1, "gp_n1": gp_n1,
            "total_n_nets_n1": base_nets + g_n1, "total_n_pins_n1": base_pins + gp_n1,
        }

    return {
        "task": "T6b", "scope": "<=12.3M (1x2, 2x2)",
        "provenance": dict(PROVENANCE_TAGS,
                            basis_doc=f"{ADJUDICATION_DOC} sec 6.1", spec_doc=f"{DESIGN_DRAFT} sec 2.1/sec 8"),
        "rows": rows,
        "rejected_alternative": rejected_alternative,
        "host_rss_three_coefficient_model": {
            "status": "pending_t0b",
            "reason": "design draft sec 1.1's three-coefficient PlaceDB.read host RSS model "
                      "(fitting set: adaptec1/bigblue4/group-BS/1x2/net-drop variants, 2x2 held "
                      "out) has not been fit/reproduced in this task's scope (a T0b product); "
                      "this file's rows carry actual net/pin counts only, not host RSS "
                      "predictions -- 2x2 remains holdout-only, never enters fitting, per T6b's "
                      "task row",
        },
        "m4_l12_note": "design draft M4-L12 (glue treated as 0 in sec 2.1's benchmark ladder) is "
                       "resolved by this file's actual g/gp values (sec 6.1: \"修正量 < 0.2%\")",
    }


# ---------------------------------------------------------------------------
# T7b: count_freeze_30m.json (3x3)
# ---------------------------------------------------------------------------

def build_count_freeze_30m(arrays_root=DEFAULT_ARRAYS_ROOT):
    m = _load_manifest(arrays_root, "3x3", "n2")
    base_nets, base_pins = m["base"]["n_nets"], m["base"]["n_pins"]
    g9, gp9 = m["glue"]["n_nets"], m["glue"]["n_pins"]
    total_nets, total_pins = base_nets + g9, base_pins + gp9

    expected_pair_counts = m.get("t6", {}).get("expected_pair_counts")
    expected_g9_total = sum(expected_pair_counts.values()) if expected_pair_counts else None

    return {
        "task": "T7b", "scope": "27.7M (3x3)",
        "provenance": dict(PROVENANCE_TAGS,
                            basis_doc=f"{ADJUDICATION_DOC} sec 5.3/sec 6.1", spec_doc=f"{DESIGN_DRAFT} sec 2.1/sec 8"),
        "base_n_nodes": m["base"].get("n_nodes"),
        "base_n_nets": base_nets, "base_n_pins": base_pins,
        "g9": g9, "gp9": gp9,
        "g9_expected_pre_measurement": expected_g9_total,
        "g9_expected_pre_measurement_note": "adjudication doc sec 5.3: \"期望 glue = 59,228.74 "
                                             "nets -> 118,457 pins\" -- superseded here by the "
                                             "actual manifest count (sec 6.1's table row: "
                                             "\"59,229(期望,T7b 回填實數)\")",
        "total_n_nets": total_nets, "total_n_pins": total_pins,
        "glue_share_nets": g9 / total_nets, "glue_share_pins": gp9 / total_pins,
        "int_headroom": int_headroom_report(total_nets, total_pins),
        "citation_declaration": "T9 與 E5 的所有數字一律引用本檔 (count_freeze_30m.json)",
        "source_manifest": os.path.join(arrays_root, "3x3_n2", "3x3_n2.manifest.json"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", default=DEFAULT_ARRAYS_ROOT)
    ap.add_argument("--out-small", default=DEFAULT_OUT_SMALL)
    ap.add_argument("--out-30m", default=DEFAULT_OUT_30M)
    args = ap.parse_args(argv)

    small = build_count_freeze_small(arrays_root=args.arrays_root)
    os.makedirs(os.path.dirname(args.out_small), exist_ok=True)
    with open(args.out_small, "w") as f:
        json.dump(small, f, indent=1, sort_keys=True)
    print(f"[count_freeze] wrote {args.out_small}")

    m30 = build_count_freeze_30m(arrays_root=args.arrays_root)
    os.makedirs(os.path.dirname(args.out_30m), exist_ok=True)
    with open(args.out_30m, "w") as f:
        json.dump(m30, f, indent=1, sort_keys=True)
    print(f"[count_freeze] wrote {args.out_30m}")

    return small, m30


if __name__ == "__main__":
    main()
