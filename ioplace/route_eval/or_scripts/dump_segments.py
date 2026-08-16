"""Stage 2 S2 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
plan.md` sec 7.1 / sec 10 S2 row): dump every routed net's wire geometry out
of a routed DEF into a flat `.npz` + JSON sidecar, using OpenROAD's own
`odb.dbWireDecoder` (sec 7.1's stated rationale for not hand-rolling a DEF
wire parser: the decoder already resolves `*`-continuation, `VIA`/`TECH_VIA`,
`RECT` patches and `VWIRE` markers -- see sec 3.4/P7).

**Known limitation (found running this against a real TritonRoute-written
DEF, not just synthetic fixtures):** `POINT_EXT` cannot be decoded via this
OpenROAD build's (v2.0-17598) Python bindings -- `decoder.getPoint()` always
resolves to the 2-int `getPoint(int&,int&)` C++ overload regardless of how
it's called from Python, which hard-asserts (`_opcode == POINT`, SIGABRT,
*not* a catchable Python exception) whenever the real opcode is POINT_EXT.
`decode_net_wire()` below detects this opcode and skips it safely (drops at
most the one wire segment touching that extended endpoint, never calls the
crashing accessor) -- see that function's `_OP_POINT_EXT` branch for the
full account of what was tried and why, and this task's final report for
the measured wirelength impact on a real design.

This script runs standalone under OpenROAD's embedded Python
(`openroad -python dump_segments.py ...`), NOT the DREAMPlace venv -- it
imports only `openroad`/`odb`/`numpy`/stdlib, never `torch` or anything
under `ioplace/` that pulls torch in (repo-wide rule: DREAMPlace/torch code
must stay out of the OpenROAD-side scripts, see the CLAUDE.md worktree
instructions for this task). `ioplace/route_eval/segments.py` is the
`.venv312`-side loader for the files this script writes; keep the KIND_*
constants and row layout below in sync with that module's copy.

Row schema (one row per WIRE/VIA/RECT/VWIRE/SHORT opcode emitted by the
decoder; JUNCTION/RULE/ITERM/BTERM carry no geometry of their own and are
only tallied in the JSON sidecar's `counts`):

    net_id     int32   index into `net_names`
    kind       uint8   KIND_WIRE=0 KIND_VIA=1 KIND_RECT=2 KIND_VWIRE=3 KIND_SHORT=4
    layer      int16   index into `layer_names`; the *active path layer* at
                        the time this opcode was seen (for VIA this is the
                        "from" layer, i.e. the layer of the PATH segment the
                        via hangs off of, not the via's own top/bottom pair --
                        see `via_names`/the via object itself for that)
    x0,y0,x1,y1  int64  DBU. WIRE: the two endpoints of one polyline edge
                        (each consecutive POINT/POINT_EXT pair decodes to one
                        row). VIA/VWIRE/SHORT: x0==x1,y0==y1, the single
                        point. RECT: the patch's own absolute bbox (decoder's
                        deltas resolved against the current point).
    width      int64   WIRE only: `layer.getWidth()` (nominal LEF width, DBU).
                        NDR (non-default rule) overrides are not applied --
                        informational field only, not used by any crossing
                        math (sec 7.4 doesn't need it). 0 for other kinds.
    via_id     int32   VIA only: index into `via_names`. -1 otherwise.

`route_cross_raw`/`route_wl` (sec 7 / S3) are computed from KIND_WIRE rows
only: RECT patches are ignored (sec 7.4: "RECT patch 忽略") and VWIRE rows
are excluded (sec 7.4: "VWIRE 一律跳過") -- this script still records both,
tagged, so nothing is silently dropped and S3 (or a debugging session) can
audit what was excluded and why instead of taking it on faith.

Usage:
    openroad -python ioplace/route_eval/or_scripts/dump_segments.py \\
        --lef tech.lef --lef cells.lef ... \\
        --def routed.def \\
        --out segments.npz [--json-out segments.json] [--design-name NAME]

Only `block.getNets()` (regular signal nets) is walked; SPECIALNETS
(power/ground) are a separate odb accessor (`block.getSNets()`) and are
never touched here -- sec 7.2's "SPECIALNETS ... 不算落差" is therefore true
by construction, not by a filter that could have a bug in it.
"""
import argparse
import datetime
import json
import sys
import time

import numpy as np

import openroad
import odb

# --- row `kind` values; keep in sync with ioplace/route_eval/segments.py ---
KIND_WIRE = 0
KIND_VIA = 1
KIND_RECT = 2
KIND_VWIRE = 3
KIND_SHORT = 4
KIND_NAMES = {KIND_WIRE: "WIRE", KIND_VIA: "VIA", KIND_RECT: "RECT",
              KIND_VWIRE: "VWIRE", KIND_SHORT: "SHORT"}

# odb.dbWireDecoder opcode values (odb.dbWireDecoder.PATH etc. give the same
# ints; spelled out here so this file has no odb-object-attribute lookups
# beyond the decoder/layer/via calls it actually needs per row).
_OP_PATH, _OP_JUNCTION, _OP_SHORT, _OP_VWIRE = 0, 1, 2, 3
_OP_POINT, _OP_POINT_EXT, _OP_VIA, _OP_TECH_VIA = 4, 5, 6, 7
_OP_RECT, _OP_ITERM, _OP_BTERM, _OP_RULE, _OP_END_DECODE = 8, 9, 10, 11, 12


def _intern(name, table, index):
    idx = index.get(name)
    if idx is None:
        idx = len(table)
        table.append(name)
        index[name] = idx
    return idx


def decode_net_wire(decoder, wire, net_id, layer_table, layer_index,
                     via_table, via_index, layer_width_cache, rows, counts):
    """Replay one net's `dbWire` and append one row per WIRE/VIA/RECT/VWIRE/
    SHORT opcode to `rows`. Mutates `counts` (a dict of opcode-name -> int)
    in place for the JSON sidecar.
    """
    decoder.begin(wire)
    cur_layer = -1
    cur_pt = None  # (x, y) of the last POINT/POINT_EXT since the last PATH/JUNCTION
    while True:
        op = decoder.next()
        if op == _OP_END_DECODE:
            break
        elif op == _OP_PATH:
            layer_obj = decoder.getLayer()
            cur_layer = _intern(layer_obj.getName(), layer_table, layer_index)
            if cur_layer not in layer_width_cache:
                layer_width_cache[cur_layer] = int(layer_obj.getWidth())
            cur_pt = None
        elif op == _OP_JUNCTION:
            # Branch point referencing an earlier point in the tree. We don't
            # need the id/value to compute geometry -- every subsequent PATH
            # still lists its own explicit POINT sequence -- but reset
            # `cur_pt` defensively so a POINT immediately after a JUNCTION
            # (no intervening PATH) can never be paired with a stale point
            # from a different branch (see module docstring's row schema
            # note; this is the one conservative choice in this decoder that
            # could, in principle, undercount one edge per junction if a
            # branch's first two points both follow the JUNCTION op with no
            # PATH between them -- not observed in any trace inspected while
            # writing this, but flagged here for anyone debugging a
            # wirelength mismatch).
            decoder.getJunctionId()
            decoder.getJunctionValue()
            cur_pt = None
            counts["JUNCTION"] = counts.get("JUNCTION", 0) + 1
        elif op == _OP_POINT:
            pt = decoder.getPoint()
            x, y = int(pt[0]), int(pt[1])
            if cur_pt is not None and (x, y) != cur_pt:
                x0, y0 = cur_pt
                width = layer_width_cache.get(cur_layer, 0)
                rows.append((net_id, KIND_WIRE, cur_layer, x0, y0, x, y, width, -1))
                counts["WIRE"] = counts.get("WIRE", 0) + 1
            cur_pt = (x, y)
        elif op == _OP_POINT_EXT:
            # `dbWireDecoder.getPoint()` is unsafe here: this build's SWIG
            # binding only ever reaches the 2-int `getPoint(int&,int&)`
            # overload (asserting `_opcode == POINT`) no matter how it's
            # called from Python -- 0 args always resolves to that overload
            # and hard-aborts the whole process (SIGABRT, not a catchable
            # Python exception) when the real opcode is POINT_EXT; passing
            # any number/type of extra args is rejected outright as "wrong
            # number or type of arguments". Confirmed empirically against a
            # real TritonRoute-written routed DEF (results/stage2/rehearsal/
            # mgc_fft_1) while building this pipeline -- there is no
            # reachable Python call that returns POINT_EXT's (x, y, ext).
            # `dbWire.getCoord(jid)` is *not* a substitute either: it's
            # paired with JUNCTION's getJunctionValue() (a data-array
            # index), not a general point-index accessor, and asserts on
            # any other input.
            #
            # Safe fallback: don't extract this point's coordinates at all,
            # and reset `cur_pt` (like a PATH boundary) so the *next* real
            # POINT starts a fresh segment instead of silently bridging
            # across the unresolved one with a phantom edge. This costs at
            # most the one short wire segment touching the extended
            # endpoint (POINT_EXT is a wire-end extension/end-cap in
            # practice, typically the first or last point of a PATH, so the
            # segment lost is usually a few hundred DBU at most) -- see
            # or_scripts/verify_routed_def.py's Sigma(route_wl)-vs-
            # report_wire_length check for how much this actually costs on
            # real designs, and this file's own module docstring/the task's
            # final report for the measured number.
            cur_pt = None
            counts["POINT_EXT_SKIPPED"] = counts.get("POINT_EXT_SKIPPED", 0) + 1
        elif op in (_OP_VIA, _OP_TECH_VIA):
            via_obj = decoder.getVia() if op == _OP_VIA else decoder.getTechVia()
            via_idx = _intern(via_obj.getName(), via_table, via_index)
            x, y = cur_pt if cur_pt is not None else (0, 0)
            rows.append((net_id, KIND_VIA, cur_layer, x, y, x, y, 0, via_idx))
            counts["VIA"] = counts.get("VIA", 0) + 1
        elif op == _OP_RECT:
            x0, y0 = cur_pt if cur_pt is not None else (0, 0)
            try:
                try:
                    dx1, dy1, dx2, dy2 = decoder.getRect()
                except TypeError:
                    dx1, dy1, dx2, dy2 = decoder.getRect(0, 0, 0, 0)
                rows.append((net_id, KIND_RECT, cur_layer,
                             x0 + int(dx1), y0 + int(dy1),
                             x0 + int(dx2), y0 + int(dy2), 0, -1))
                counts["RECT"] = counts.get("RECT", 0) + 1
            except Exception:
                # RECT patches are ignored by every crossing/wirelength
                # computation anyway (spec sec 7.4); don't let a decode
                # surprise here kill the whole net's wire.
                counts["RECT_DECODE_ERROR"] = counts.get("RECT_DECODE_ERROR", 0) + 1
        elif op == _OP_VWIRE:
            x, y = cur_pt if cur_pt is not None else (0, 0)
            rows.append((net_id, KIND_VWIRE, cur_layer, x, y, x, y, 0, -1))
            counts["VWIRE"] = counts.get("VWIRE", 0) + 1
        elif op == _OP_SHORT:
            x, y = cur_pt if cur_pt is not None else (0, 0)
            rows.append((net_id, KIND_SHORT, cur_layer, x, y, x, y, 0, -1))
            counts["SHORT"] = counts.get("SHORT", 0) + 1
        elif op == _OP_RULE:
            decoder.getRule()
            counts["RULE"] = counts.get("RULE", 0) + 1
        elif op in (_OP_ITERM, _OP_BTERM):
            name = "ITERM" if op == _OP_ITERM else "BTERM"
            counts[name] = counts.get(name, 0) + 1
        else:
            counts["OTHER"] = counts.get("OTHER", 0) + 1


def dump_segments(lef_paths, def_path, design_name=None):
    """Read `lef_paths` (in the given order -- caller is responsible for
    putting tech.lef first, per sec 3.2's ISPD2025 lib-glob pitfall) and
    `def_path`, walk every regular signal net's wire, and return
    `(rows, net_names, net_has_wire, layer_names, via_names, meta)`.
    """
    tech = openroad.Tech()
    for lef in lef_paths:
        tech.readLef(lef)
    design = openroad.Design(tech)
    design.readDef(def_path)
    block = design.getBlock()

    layer_table, layer_index = [], {}
    via_table, via_index = [], {}
    layer_width_cache = {}
    rows = []
    counts = {}

    net_names = []
    net_has_wire = []
    decoder = odb.dbWireDecoder()
    t0 = time.time()
    for net_id, net in enumerate(block.getNets()):
        if net.isSpecial():
            # Regular NETS only (sec 7.2): should never trigger given
            # getNets() vs getSNets() are already disjoint accessors, but
            # kept as a belt-and-suspenders assert-by-skip.
            continue
        net_names.append(net.getName())
        wire = net.getWire()
        has_wire = wire is not None
        net_has_wire.append(has_wire)
        if has_wire:
            decode_net_wire(decoder, wire, len(net_names) - 1, layer_table,
                             layer_index, via_table, via_index,
                             layer_width_cache, rows, counts)
    decode_s = time.time() - t0

    meta = {
        "design_name": design_name or block.getName(),
        "lef_paths": list(lef_paths),
        "def_path": def_path,
        "units_distance_microns": block.getDbUnitsPerMicron(),
        "openroad_version": openroad.openroad_version(),
        "generated_at": datetime.datetime.now().isoformat(),
        "net_count": len(net_names),
        "routed_net_count": int(sum(net_has_wire)),
        "unrouted_net_count": int(len(net_names) - sum(net_has_wire)),
        "segment_row_count": len(rows),
        "layer_count": len(layer_table),
        "via_type_count": len(via_table),
        "opcode_counts": counts,
        "decode_seconds": decode_s,
    }
    return rows, net_names, net_has_wire, layer_table, via_table, meta


def _str_array(names):
    """`np.array([])` on an empty list of strings infers dtype=float64,
    which is a nuisance for consumers that assume a unicode dtype (e.g. to
    concatenate/compare with another `<U...` array) -- pin the dtype so an
    empty via_names/layer_names array round-trips as text either way.
    """
    if not names:
        return np.array([], dtype="<U1")
    return np.array(names)


def write_outputs(out_path, rows, net_names, net_has_wire, layer_names,
                   via_names, meta, json_out_path=None):
    if rows:
        arr = np.asarray(rows, dtype=np.int64)
        seg_net_id = arr[:, 0].astype(np.int32)
        seg_kind = arr[:, 1].astype(np.uint8)
        seg_layer = arr[:, 2].astype(np.int16)
        seg_x0, seg_y0 = arr[:, 3], arr[:, 4]
        seg_x1, seg_y1 = arr[:, 5], arr[:, 6]
        seg_width = arr[:, 7]
        seg_via_id = arr[:, 8].astype(np.int32)
    else:
        seg_net_id = np.empty(0, dtype=np.int32)
        seg_kind = np.empty(0, dtype=np.uint8)
        seg_layer = np.empty(0, dtype=np.int16)
        seg_x0 = seg_y0 = seg_x1 = seg_y1 = seg_width = np.empty(0, dtype=np.int64)
        seg_via_id = np.empty(0, dtype=np.int32)

    np.savez_compressed(
        out_path,
        seg_net_id=seg_net_id, seg_kind=seg_kind, seg_layer=seg_layer,
        seg_x0=seg_x0, seg_y0=seg_y0, seg_x1=seg_x1, seg_y1=seg_y1,
        seg_width=seg_width, seg_via_id=seg_via_id,
        net_names=_str_array(net_names), net_has_wire=np.array(net_has_wire, dtype=bool),
        layer_names=_str_array(layer_names), via_names=_str_array(via_names),
    )

    json_out_path = json_out_path or (str(out_path).rsplit(".", 1)[0] + ".json")
    with open(json_out_path, "w") as f:
        json.dump(meta, f, indent=1)
    return json_out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lef", action="append", required=True,
                     help="LEF file, repeatable; given order is read order "
                          "(put tech.lef first).")
    ap.add_argument("--def", dest="def_path", required=True)
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--json-out", default=None,
                     help="output JSON sidecar path (default: <out> with "
                          ".json extension)")
    ap.add_argument("--design-name", default=None)
    args = ap.parse_args(argv)

    rows, net_names, net_has_wire, layer_names, via_names, meta = dump_segments(
        args.lef, args.def_path, design_name=args.design_name)
    json_out = write_outputs(args.out, rows, net_names, net_has_wire,
                              layer_names, via_names, meta,
                              json_out_path=args.json_out)
    print(f"wrote {args.out} ({len(rows)} segment rows, "
          f"{meta['routed_net_count']}/{meta['net_count']} nets routed) "
          f"and {json_out}")


if __name__ == "__main__":
    main(sys.argv[1:])
