"""Stage 2 S2 real acceptance script (`docs/superpowers/specs/2026-08-13-
stage2-innovus-calibration-plan.md` sec 10 S2 row): the "獨立的驗收腳本" this
task's instructions ask for, since the odb-vs-text-parser cross-check and
the Sigma(route_wl)-vs-OpenROAD-report comparison both need a real OpenROAD
process (`import odb`) and therefore can't run under `$DP/.venv312`'s
pytest (see `tests/test_route_eval_s2.py`'s module docstring).

Runs two checks against one routed DEF:

  1. Sigma(route_wl) (this pipeline's `dump_segments.dump_segments()`,
     summed over KIND_WIRE rows) vs two OpenROAD-native oracles that never
     touch this repo's decode loop: `report_wire_length`'s own total, and
     `sum(net.getWire().getLength() for net in block.getNets())` (odb's own
     wire-length accumulator, `dbWire::getLength()`). The *gate* is the
     reconciled comparison: `dbWire::getLength()` additionally counts each
     RECT patch's `(long_side - short_side)` (spec sec 7.4's "RECT patch
     忽略" applies to route_wl/crossing, not to odb's own accumulator), so
     `recon = Sigma(route_wl) + Sigma_RECT(long_side - short_side)` is what's
     compared against the native total; relative error must be < 1%
     (sec 10 S2 row's acceptance number). The raw, un-reconciled
     `Sigma(route_wl)` vs native delta is still reported as
     `definitional_delta_pct` -- informational only, since it isn't
     comparing the same quantity.
  2. odb-decoded segments vs `ioplace.route_eval.def_text_parser`-decoded
     segments, per net, over all routed nets by default (`--sample 0`; pass
     a positive `--sample N` to check only a random N-net subsample) --
     sec 10 S2 row's "odb vs 文字解析逐 net 相同". Segments are compared as
     endpoint-order-independent sets (a WIRE row's two endpoints can come
     out in either order from the two decoders without that being a real
     disagreement).

Usage:
    openroad -python ioplace/route_eval/or_scripts/verify_routed_def.py \\
        --lef tech.lef --lef cells.lef ... --def routed.def \\
        [--sample 0] [--seed 0] [--report-out wl_report.rpt]
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import openroad
import odb  # noqa: F401 (import-time check: fail fast if not under `openroad -python`)

# dump_segments.py is a sibling script (not an importable package member
# under OpenROAD's Python -- see its own docstring), so pull it in by path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_segments  # noqa: E402

# def_text_parser.py *is* a regular ioplace.route_eval package module (no
# torch anywhere on its import chain), reachable once the repo root is on
# sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from ioplace.route_eval import def_text_parser as dtp  # noqa: E402


def _kind_name(k):
    return dump_segments.KIND_NAMES[k]


def _canon_row(kind, layer, x0, y0, x1, y1, via_name):
    """Endpoint order shouldn't matter for equality (odb and the text
    parser may walk a given wire polyline in either direction)."""
    a, b = (x0, y0), (x1, y1)
    if b < a:
        a, b = b, a
    return (kind, layer, a[0], a[1], b[0], b[1], via_name)


def odb_rows_by_net(rows, net_names, layer_names, via_names):
    by_net = {}
    for net_id, kind, layer, x0, y0, x1, y1, width, via_id in rows:
        layer_name = layer_names[layer] if layer >= 0 else ""
        via_name = via_names[via_id] if via_id >= 0 else ""
        by_net.setdefault(net_names[net_id], []).append(
            _canon_row(_kind_name(kind), layer_name, x0, y0, x1, y1, via_name))
    return by_net


def text_rows_for_net(parsed_net):
    return [_canon_row(seg.kind, seg.layer, seg.x0, seg.y0, seg.x1, seg.y1, seg.via_name)
            for seg in parsed_net.segments]


def _extract_nets_region(text):
    """Slice `text` down to the SPECIALNETS/NETS tail so the pure-Python
    tokenizer in def_text_parser doesn't have to walk the (often much
    bigger) COMPONENTS/PINS/VIAS/ROWS preamble of a real routed DEF."""
    starts = [i for i in (text.find("\nSPECIALNETS"), text.find("\nNETS"))
              if i != -1]
    start = min(starts) if starts else 0
    end_marker = "END NETS"
    end = text.rfind(end_marker)
    end = end + len(end_marker) if end != -1 else len(text)
    return text[start:end]


def report_wire_length_total(design, out_path):
    """`report_wire_length -net [get_nets *]` (this build needs an explicit
    -net list, see GRT-0238) writes a per-net table with *two* rows per
    routed net -- one "grt:" (global-route estimate) and one "drt:" (actual
    detailed-route length), both in fractional microns:

        tool net total_wl #pins
        grt: x_out_0_0 45 3
        drt: x_out_0_0 45.06 3
        ...

    Only "drt:" rows are real wire geometry (what `dump_segments.py` also
    reads, from the same post-detailed-route db); "grt:" rows are a coarse
    global-router estimate and must not be mixed into the sum. Returns the
    total in *microns* (float) -- caller converts using
    `meta["units_distance_microns"]`, same as everywhere else in this repo
    (spec sec 7.3's coordinate-mapping convention).
    """
    design.evalTclString(f"report_wire_length -net [get_nets *] -file {{{out_path}}}")
    with open(out_path) as f:
        report_text = f.read()
    total_um = 0.0
    n_rows = 0
    for line in report_text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "drt:":
            continue
        try:
            total_um += float(parts[-2])
        except ValueError:
            continue
        n_rows += 1
    if n_rows == 0:
        return None, report_text
    return total_um, report_text


def odb_native_wire_length_dbu(block):
    total = 0
    for net in block.getNets():
        if net.isSpecial():
            continue
        wire = net.getWire()
        if wire is not None:
            total += wire.getLength()
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lef", action="append", required=True)
    ap.add_argument("--def", dest="def_path", required=True)
    ap.add_argument("--sample", type=int, default=0,
                     help="check all routed nets (default, 0); pass a "
                          "positive N to check only a random N-net "
                          "subsample instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-out", default="s2_verify_wire_length.rpt")
    ap.add_argument("--json-out", default=None,
                     help="write check1/check2 summary numbers as JSON to this path "
                          "(sec 10 S2 row's verify_s2.json)")
    args = ap.parse_args(argv)

    print(f"[verify] reading {len(args.lef)} LEF(s) + {args.def_path}")
    t0 = time.time()
    rows, net_names, net_has_wire, layer_names, via_names, meta = dump_segments.dump_segments(
        args.lef, args.def_path, design_name="s2_verify")
    print(f"[verify] dump_segments: {len(rows)} rows, "
          f"{meta['routed_net_count']}/{meta['net_count']} nets routed, "
          f"{time.time() - t0:.1f}s")

    # --- check 1: total wirelength vs OpenROAD-native oracles -------------
    ours_dbu = sum(x1 - x0 if x0 <= x1 else x0 - x1
                   for _, kind, _, x0, y0, x1, y1, *_ in rows if kind == dump_segments.KIND_WIRE) \
        + sum(y1 - y0 if y0 <= y1 else y0 - y1
              for _, kind, _, x0, y0, x1, y1, *_ in rows if kind == dump_segments.KIND_WIRE)
    units = meta["units_distance_microns"]
    ours_um = ours_dbu / units
    print(f"[verify] Sigma(route_wl) ours = {ours_dbu} DBU = {ours_um:.3f} um")

    # Re-open a fresh Tech/Design for the Tcl-command oracle so this script
    # doesn't depend on dump_segments() leaving any particular db state
    # behind.
    tech = openroad.Tech()
    for lef in args.lef:
        tech.readLef(lef)
    design = openroad.Design(tech)
    design.readDef(args.def_path)
    block = design.getBlock()

    native_dbu = odb_native_wire_length_dbu(block)
    native_um = native_dbu / units
    print(f"[verify] dbWire.getLength() total = {native_dbu} DBU = {native_um:.3f} um")

    report_total_um, report_text = report_wire_length_total(design, args.report_out)
    print(f"[verify] report_wire_length (drt: rows only) total = {report_total_um} um "
          f"(raw report at {args.report_out})")

    def _pct_err(a, b):
        return abs(a - b) / b * 100.0 if b else float("nan")

    # Reconciliation: `dbWire::getLength()` (the `native_dbu` oracle) counts
    # each RECT patch as a rectangle's long side, not (like route_wl) as a
    # zero-length excluded row -- spec sec 7.4's "RECT patch 忽略" is a
    # route_wl/crossing convention, not odb's own. The gap between
    # `ours_dbu` (route_wl, RECT excluded) and `native_dbu` is therefore
    # explained by exactly `Sigma_RECT(long_side - short_side)` per patch;
    # add that back in before comparing against the native oracle.
    rect_term_dbu = sum(max(abs(x1 - x0), abs(y1 - y0)) - min(abs(x1 - x0), abs(y1 - y0))
                        for _, kind, _, x0, y0, x1, y1, *_ in rows
                        if kind == dump_segments.KIND_RECT)
    recon_dbu = ours_dbu + rect_term_dbu
    recon_um = recon_dbu / units
    print(f"[verify] RECT reconciliation term Sigma(long-short) = {rect_term_dbu} DBU")
    print(f"[verify] reconciled = Sigma(route_wl) + RECT term = {recon_dbu} DBU = {recon_um:.3f} um")

    definitional_delta_pct = _pct_err(ours_dbu, native_dbu)
    print(f"[verify] definitional delta, Sigma(route_wl) vs dbWire.getLength() "
          f"(RECT-exclusion convention, informational only): "
          f"{definitional_delta_pct:.4f}%")
    err_recon = _pct_err(recon_dbu, native_dbu)
    print(f"[verify] GATE: reconciled vs dbWire.getLength(): {err_recon:.4f}% "
          f"({'PASS' if err_recon < 1.0 else 'FAIL'} < 1%)")
    err_report = None
    if report_total_um is not None:
        err_report = _pct_err(ours_um, report_total_um)
        print(f"[verify] Sigma(route_wl) vs report_wire_length (informational only): "
              f"{err_report:.4f}%")
    else:
        print("[verify] report_wire_length total: could not parse report file "
              "(see raw report above); relying on dbWire.getLength() oracle only")

    # --- check 2: odb vs text-parser sample cross-check --------------------
    with open(args.def_path) as f:
        def_text = f.read()
    nets_region = _extract_nets_region(def_text)
    t1 = time.time()
    try:
        parsed = dtp.parse_def_routed(nets_region)
    except dtp.DefParseError as exc:
        # def_text_parser is intentionally strict (sec 7.1: it's a sampling
        # cross-check, not the primary parser) -- a construct it doesn't
        # know about anywhere in the NETS/SPECIALNETS region aborts the
        # whole parse rather than silently mis-decoding. Report *what*
        # broke instead of a bare traceback, since that's the actionable
        # signal for extending the grammar or narrowing the sample.
        print(f"[verify] def_text_parser FAILED to parse the NETS/SPECIALNETS "
              f"region: {exc}")
        print("[verify] OVERALL: FAIL (text parser could not run; "
              "Sigma(route_wl) oracle checks above still stand on their own)")
        return 1
    print(f"[verify] def_text_parser: {len(parsed.nets)} nets, "
          f"{len(parsed.special_net_names)} special nets skipped, "
          f"{time.time() - t1:.1f}s")

    odb_by_net = odb_rows_by_net(rows, net_names, layer_names, via_names)
    routed_names = [name for name, has in zip(net_names, net_has_wire) if has]
    rng = random.Random(args.seed)
    # --sample 0 (default) means "check all routed nets"; a positive N
    # checks only a random N-net subsample.
    sample_names = routed_names if args.sample <= 0 or len(routed_names) <= args.sample else \
        rng.sample(routed_names, args.sample)

    n_checked = n_match = n_mismatch = n_missing_from_text = 0
    n_mismatch_rect_only = n_mismatch_other = 0
    mismatches = []
    other_mismatches = []
    for name in sample_names:
        if name not in parsed.nets:
            n_missing_from_text += 1
            continue
        odb_set = set(odb_by_net.get(name, []))
        text_set = set(text_rows_for_net(parsed.nets[name]))
        n_checked += 1
        if odb_set == text_set:
            n_match += 1
            continue
        n_mismatch += 1
        diff = (odb_set - text_set) | (text_set - odb_set)
        # Both decoders now resolve RECT/POINT_EXT for real (dump_segments.py's
        # `_OP_RECT`/`_OP_POINT_EXT` branches), so this bucket is expected to
        # be empty -- kept as a separate count rather than folded into
        # n_mismatch_other so a regression that reopens the old gap is
        # immediately visible as a nonzero n_mismatch_rect_only instead of
        # blending into "other".
        if all(row[0] == "RECT" for row in diff):
            n_mismatch_rect_only += 1
        else:
            n_mismatch_other += 1
            if len(other_mismatches) < 10:
                other_mismatches.append((name, odb_set - text_set, text_set - odb_set))
        if len(mismatches) < 10:
            mismatches.append((name, odb_set - text_set, text_set - odb_set))

    print(f"[verify] checked {len(sample_names)} routed nets; "
          f"{n_missing_from_text} not found in text-parser output "
          f"(NETS region slice mismatch?); of {n_checked} compared: "
          f"{n_match} exact match, {n_mismatch} mismatch "
          f"({n_mismatch_rect_only} RECT-only, {n_mismatch_other} other)")
    print("[verify] first 10 mismatches (any kind):")
    for name, only_odb, only_text in mismatches:
        print(f"  MISMATCH net={name!r}: only_in_odb={sorted(only_odb)[:3]} "
              f"only_in_text={sorted(only_text)[:3]}")
    print("[verify] first 10 *non-RECT-only* mismatches:")
    for name, only_odb, only_text in other_mismatches:
        print(f"  MISMATCH(non-RECT) net={name!r}: only_in_odb={sorted(only_odb)[:3]} "
              f"only_in_text={sorted(only_text)[:3]}")

    # JUNCTION carries no geometry of its own and isn't decoded/validated by
    # anything in this pipeline (dump_segments.py's `_OP_JUNCTION` branch
    # only tallies it, unchanged by the POINT_EXT/RECT binding workaround --
    # this corpus has 0 occurrences, so it's untested territory). Flag it
    # loudly rather than silently passing if a future DEF exercises it.
    n_junction = meta["opcode_counts"].get("JUNCTION", 0)
    junction_unvalidated = n_junction > 0
    if junction_unvalidated:
        print(f"[verify] WARNING: {n_junction} JUNCTION opcode(s) seen -- "
              f"this decode path is untested (0 occurrences in the corpus "
              f"this pipeline was validated against); treat this run's "
              f"result as unvalidated for any net touching a JUNCTION")

    ok = err_recon < 1.0 and n_mismatch == 0 and n_missing_from_text == 0
    print(f"[verify] OVERALL: {'PASS' if ok else 'FAIL'}")

    if args.json_out:
        summary = {
            "generated_at": datetime.datetime.now().isoformat(),
            "def_path": args.def_path,
            "lef_paths": list(args.lef),
            "junction_unvalidated": junction_unvalidated,
            "check1_wire_length": {
                "ours_dbu": ours_dbu,
                "ours_um": ours_um,
                "dbwire_native_dbu": native_dbu,
                "dbwire_native_um": native_um,
                "rect_term_dbu": rect_term_dbu,
                "recon_dbu": recon_dbu,
                "recon_um": recon_um,
                "err_vs_dbwire_native_recon_pct": err_recon,
                "definitional_delta_pct": definitional_delta_pct,
                "report_wire_length_drt_total_um": report_total_um,
                "err_vs_report_wire_length_pct": err_report,
                "pass_lt_1pct": bool(err_recon < 1.0),
            },
            "check2_odb_vs_text_parser": {
                "sample_requested": args.sample,
                "seed": args.seed,
                "routed_net_count": len(routed_names),
                "n_sampled": len(sample_names),
                "n_checked": n_checked,
                "n_match": n_match,
                "n_mismatch": n_mismatch,
                "n_mismatch_rect_only": n_mismatch_rect_only,
                "n_mismatch_other": n_mismatch_other,
                "n_missing_from_text": n_missing_from_text,
                "pass_n_mismatch_eq_0": n_mismatch == 0 and n_missing_from_text == 0,
            },
            "overall_pass": ok,
        }
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[verify] wrote summary JSON to {args.json_out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
