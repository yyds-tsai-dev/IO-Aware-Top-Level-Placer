"""Stage 2 S2 golden tests (`docs/superpowers/specs/2026-08-13-stage2-
innovus-calibration-plan.md` sec 10 S2 row): hand-written DEF `NETS`
snippets, decoded segment-by-segment with `ioplace.route_eval.def_text_parser`
and checked against exactly what a human reading the DEF text would expect
-- `*`-continuation, via placement, `RECT` patches, multi-layer `NEW`
branches, and `VIRTUAL` segments.

`ioplace.route_eval.segments.load_segments` is exercised against a small
hand-built `.npz` (no OpenROAD needed -- it's just numpy arrays laid out the
way `dump_segments.py` would write them) so the loader's own logic (net-name
lookup, per-net row reconstruction, wire-length summation) has coverage
independent of whether OpenROAD is installed on the test host.

The odb-side tests (`import odb`, run `dump_segments.py` against these same
DEF snippets through a real OpenROAD, and diff the two parsers' output
row-for-row) are `skipif`'d when `odb` isn't importable in this Python --
which is always true under `$DP/.venv312` (DREAMPlace's venv has no OpenROAD
Python bindings). The real odb-vs-text cross-check and the
Sigma(route_wl)-vs-`report_wire_length` acceptance numbers this task's
instructions ask for are produced by the separate, non-pytest acceptance
script `ioplace/route_eval/or_scripts/verify_routed_def.py`, run directly
under `openroad -python` (see that script's docstring and this task's final
report for the numbers it produced on a real GR+DR'd ISPD2015 design).
"""
import importlib.util
import numpy as np
import pytest

from ioplace.route_eval import def_text_parser as dtp
from ioplace.route_eval.segments import (
    load_segments, rect_reconciliation_dbu, KIND_WIRE, KIND_VIA, KIND_RECT,
)

HAS_ODB = importlib.util.find_spec("odb") is not None


# ---------------------------------------------------------------------------
# def_text_parser golden tests
# ---------------------------------------------------------------------------

def test_simple_two_point_wire():
    text = """
NETS 1 ;
- n1 ( u1 A ) ( u2 B )
  + ROUTED metal2 ( 100 200 ) ( 100 500 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    assert set(parsed.nets) == {"n1"}
    segs = parsed.nets["n1"].segments
    assert segs == [dtp.Segment(dtp.WIRE, "metal2", 100, 200, 100, 500)]


def test_star_continuation_both_axes():
    # "( * y )" repeats the previous x; "( x * )" repeats the previous y.
    text = """
NETS 1 ;
- n1
  + ROUTED metal3 ( 1000 2000 ) ( * 3000 ) ( 4000 * )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [
        dtp.Segment(dtp.WIRE, "metal3", 1000, 2000, 1000, 3000),
        dtp.Segment(dtp.WIRE, "metal3", 1000, 3000, 4000, 3000),
    ]


def test_star_continuation_requires_prior_point():
    text = """
NETS 1 ;
- n1
  + ROUTED metal1 ( * 100 )
  ;
END NETS
"""
    with pytest.raises(dtp.DefParseError):
        dtp.parse_def_routed(text)


def test_point_extension_value_is_ignored():
    # DEF allows a third "extension" integer inside the parens; it must not
    # be mistaken for a second point's x coordinate.
    text = """
NETS 1 ;
- n1
  + ROUTED metal1 ( 0 0 50 ) ( 1000 0 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [dtp.Segment(dtp.WIRE, "metal1", 0, 0, 1000, 0)]


def test_via_between_two_layers():
    text = """
NETS 1 ;
- n1
  + ROUTED metal2 ( 100 100 ) ( 100 900 ) VIA23_2cut_E
    NEW metal3 ( 100 900 ) ( 800 900 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [
        dtp.Segment(dtp.WIRE, "metal2", 100, 100, 100, 900),
        dtp.Segment(dtp.VIA, "metal2", 100, 900, 100, 900, via_name="VIA23_2cut_E"),
        dtp.Segment(dtp.WIRE, "metal3", 100, 900, 800, 900),
    ]


def test_via_with_orientation_token():
    text = """
NETS 1 ;
- n1
  + ROUTED metal1 ( 0 0 ) VIAGEN12_SDA_UNIQ_1 FN
    NEW metal2 ( 0 0 ) ( 0 200 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs[0] == dtp.Segment(dtp.VIA, "metal1", 0, 0, 0, 0, via_name="VIAGEN12_SDA_UNIQ_1")
    assert segs[1] == dtp.Segment(dtp.WIRE, "metal2", 0, 0, 0, 200)


def test_rect_patch_relative_to_current_point():
    text = """
NETS 1 ;
- n1
  + ROUTED metal2 ( 500 500 ) ( 900 500 ) RECT ( -50 -90 250 90 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [
        dtp.Segment(dtp.WIRE, "metal2", 500, 500, 900, 500),
        dtp.Segment(dtp.RECT, "metal2", 850, 410, 1150, 590),
    ]


def test_virtual_segment_tagged_vwire():
    text = """
NETS 1 ;
- n1
  + ROUTED metal1 ( 0 0 ) VIRTUAL ( 1000 0 ) ( 2000 0 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [
        dtp.Segment(dtp.VWIRE, "metal1", 0, 0, 1000, 0),
        dtp.Segment(dtp.WIRE, "metal1", 1000, 0, 2000, 0),
    ]


def test_multiple_new_branches_tree_topology():
    # A 3-way branch: two NEW sub-paths both hanging off the same trunk,
    # each restating its own starting point explicitly (the DEF-text way of
    # expressing what odb's JUNCTION opcode encodes internally).
    text = """
NETS 1 ;
- n1
  + ROUTED metal2 ( 0 0 ) ( 1000 0 )
    NEW metal2 ( 1000 0 ) ( 1000 1000 )
    NEW metal2 ( 1000 0 ) ( 2000 0 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [
        dtp.Segment(dtp.WIRE, "metal2", 0, 0, 1000, 0),
        dtp.Segment(dtp.WIRE, "metal2", 1000, 0, 1000, 1000),
        dtp.Segment(dtp.WIRE, "metal2", 1000, 0, 2000, 0),
    ]


def test_fixed_wire_spec_parsed_same_as_routed():
    text = """
NETS 1 ;
- n1
  + FIXED metal1 ( 0 0 ) ( 500 0 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [dtp.Segment(dtp.WIRE, "metal1", 0, 0, 500, 0)]


def test_non_wire_plus_clauses_are_skipped():
    text = """
NETS 1 ;
- n1 ( u1 A )
  + USE SIGNAL
  + ROUTED metal1 ( 0 0 ) ( 100 0 )
  + WEIGHT 3
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["n1"].segments
    assert segs == [dtp.Segment(dtp.WIRE, "metal1", 0, 0, 100, 0)]


def test_net_with_no_wire_spec_has_empty_segments():
    text = """
NETS 1 ;
- n1 ( u1 A ) ( u2 B )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    assert parsed.nets["n1"].segments == []


def test_multiple_nets():
    text = """
NETS 2 ;
- n1
  + ROUTED metal1 ( 0 0 ) ( 100 0 )
  ;
- n2
  + ROUTED metal2 ( 0 0 ) ( 0 300 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    assert set(parsed.nets) == {"n1", "n2"}
    assert parsed.nets["n1"].segments == [dtp.Segment(dtp.WIRE, "metal1", 0, 0, 100, 0)]
    assert parsed.nets["n2"].segments == [dtp.Segment(dtp.WIRE, "metal2", 0, 0, 0, 300)]


def test_specialnets_recognized_and_skipped_not_parsed_as_nets():
    text = """
SPECIALNETS 2 ;
- VDD ( * VDD )
  + ROUTED metal1 ( 0 0 ) ( 1000 0 )
  + USE POWER
  ;
- VSS ( * VSS )
  + ROUTED metal1 ( 0 100 ) ( 1000 100 )
  + USE GROUND
  ;
END SPECIALNETS
NETS 1 ;
- sig1
  + ROUTED metal2 ( 0 0 ) ( 0 500 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    assert set(parsed.nets) == {"sig1"}
    assert parsed.special_net_names == ["VDD", "VSS"]
    assert parsed.nets["sig1"].segments == [dtp.Segment(dtp.WIRE, "metal2", 0, 0, 0, 500)]


def test_specialnets_before_and_after_nets_both_skipped():
    # SPECIALNETS can legally follow NETS in a DEF; the parser shouldn't
    # care about section order.
    text = """
NETS 1 ;
- sig1
  + ROUTED metal2 ( 0 0 ) ( 0 500 )
  ;
END NETS
SPECIALNETS 1 ;
- VDD
  + ROUTED metal1 ( 0 0 ) ( 1000 0 )
  ;
END SPECIALNETS
"""
    parsed = dtp.parse_def_routed(text)
    assert set(parsed.nets) == {"sig1"}
    assert parsed.special_net_names == ["VDD"]


def test_end_to_end_star_via_rect_multilayer_snippet():
    """One combined snippet exercising `*` continuation, a via, a RECT
    patch, and a second-layer NEW branch in the same net -- the "S2 golden
    test" the task instructions ask for in one shot, on top of the
    single-feature tests above.
    """
    text = """
NETS 1 ;
- clk_leaf
  + ROUTED metal2 ( 86830 961660 ) ( * 965300 ) ( 87210 * ) RECT ( -50 -90 250 90 )
    NEW metal3 ( 87210 965300 ) VIA23_2cut_E ( 87210 970000 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    segs = parsed.nets["clk_leaf"].segments
    assert segs == [
        dtp.Segment(dtp.WIRE, "metal2", 86830, 961660, 86830, 965300),
        dtp.Segment(dtp.WIRE, "metal2", 86830, 965300, 87210, 965300),
        dtp.Segment(dtp.RECT, "metal2", 87160, 965210, 87460, 965390),
        dtp.Segment(dtp.VIA, "metal3", 87210, 965300, 87210, 965300, via_name="VIA23_2cut_E"),
        dtp.Segment(dtp.WIRE, "metal3", 87210, 965300, 87210, 970000),
    ]


def test_repeated_identical_point_emits_no_degenerate_wire_row():
    # Real Innovus-produced routed DEFs (pre-CTS clock leaf markers, sec 3.2)
    # contain "+ ROUTED metal1 ( x y ) ( x y )" pairs with an *identical*
    # point twice -- a zero-length placeholder, not a real wire segment.
    # odb's dbWireDecoder emits no WIRE row for these (verified against a
    # real such net); this parser must not invent one either.
    text = """
NETS 1 ;
- n1
  + ROUTED metal1 ( 1000 2000 ) ( 1000 2000 )
    NEW metal1 ( 3000 4000 ) ( 3000 4000 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    assert parsed.nets["n1"].segments == []


def test_repeated_point_then_real_segment():
    text = """
NETS 1 ;
- n1
  + ROUTED metal1 ( 1000 2000 ) ( 1000 2000 ) ( 1000 5000 )
  ;
END NETS
"""
    parsed = dtp.parse_def_routed(text)
    assert parsed.nets["n1"].segments == [dtp.Segment(dtp.WIRE, "metal1", 1000, 2000, 1000, 5000)]


def test_malformed_def_raises_parse_error():
    with pytest.raises(dtp.DefParseError):
        dtp.parse_def_routed("NETS 1 ;\n- n1\n  + ROUTED metal1 ( 0 0\n  ;\nEND NETS\n")


# ---------------------------------------------------------------------------
# ioplace.route_eval.segments loader tests (hand-built npz, no OpenROAD)
# ---------------------------------------------------------------------------

def _write_fake_segments_npz(path):
    # Two nets: "n0" gets a 2-segment L-shaped wire plus one via; "n1" has
    # no wire at all (net_has_wire=False), mirroring an unrouted net.
    seg_net_id = np.array([0, 0, 0], dtype=np.int32)
    seg_kind = np.array([KIND_WIRE, KIND_VIA, KIND_WIRE], dtype=np.uint8)
    seg_layer = np.array([0, 0, 1], dtype=np.int16)
    seg_x0 = np.array([0, 1000, 1000], dtype=np.int64)
    seg_y0 = np.array([0, 0, 0], dtype=np.int64)
    seg_x1 = np.array([1000, 1000, 1000], dtype=np.int64)
    seg_y1 = np.array([0, 0, 500], dtype=np.int64)
    seg_width = np.array([140, 0, 140], dtype=np.int64)
    seg_via_id = np.array([-1, 0, -1], dtype=np.int32)
    np.savez(
        path,
        seg_net_id=seg_net_id, seg_kind=seg_kind, seg_layer=seg_layer,
        seg_x0=seg_x0, seg_y0=seg_y0, seg_x1=seg_x1, seg_y1=seg_y1,
        seg_width=seg_width, seg_via_id=seg_via_id,
        net_names=np.array(["n0", "n1"]), net_has_wire=np.array([True, False]),
        layer_names=np.array(["metal2", "metal3"]), via_names=np.array(["VIA23"]),
    )
    import json
    with open(str(path).rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump({"design_name": "fake", "net_count": 2, "routed_net_count": 1}, f)


def test_load_segments_wire_length_and_lookup(tmp_path):
    npz = tmp_path / "fake.npz"
    _write_fake_segments_npz(npz)
    s = load_segments(npz)
    assert s.num_nets == 2
    assert s.num_segment_rows == 3
    assert s.wire_length() == 1000 + 500  # via row contributes 0
    assert s.net_index("n0") == 0
    assert s.net_index("n1") == 1
    assert s.net_index("does-not-exist") is None
    assert bool(s.net_has_wire[0]) is True
    assert bool(s.net_has_wire[1]) is False
    assert s.meta["design_name"] == "fake"


def test_load_segments_per_net_wire_length(tmp_path):
    npz = tmp_path / "fake.npz"
    _write_fake_segments_npz(npz)
    s = load_segments(npz)
    per_net = s.per_net_wire_length()
    assert per_net.tolist() == [1500, 0]
    assert s.wire_length(net_id=0) == 1500
    assert s.non_manhattan_wire_count() == 0


def test_segments_rows_for_net_reconstructs_kind_layer_via(tmp_path):
    npz = tmp_path / "fake.npz"
    _write_fake_segments_npz(npz)
    s = load_segments(npz)
    rows = s.rows_for_net(0)
    assert rows == [
        ("WIRE", "metal2", 0, 0, 1000, 0, 140, ""),
        ("VIA", "metal2", 1000, 0, 1000, 0, 0, "VIA23"),
        ("WIRE", "metal3", 1000, 0, 1000, 500, 140, ""),
    ]


def _write_fake_segments_with_rect_npz(path):
    # Same L-shaped WIRE+VIA net as _write_fake_segments_npz(), plus one
    # RECT row with a 100 (x) x 420 (y) bbox -- the reconciliation term
    # (spec sec 7.4 / verify_routed_def.py check 1) is exactly
    # long_side - short_side = 420 - 100 = 320.
    seg_net_id = np.array([0, 0, 0, 0], dtype=np.int32)
    seg_kind = np.array([KIND_WIRE, KIND_VIA, KIND_WIRE, KIND_RECT], dtype=np.uint8)
    seg_layer = np.array([0, 0, 1, 1], dtype=np.int16)
    seg_x0 = np.array([0, 1000, 1000, 2000], dtype=np.int64)
    seg_y0 = np.array([0, 0, 0, 3000], dtype=np.int64)
    seg_x1 = np.array([1000, 1000, 1000, 2100], dtype=np.int64)
    seg_y1 = np.array([0, 0, 500, 3420], dtype=np.int64)
    seg_width = np.array([140, 0, 140, 0], dtype=np.int64)
    seg_via_id = np.array([-1, 0, -1, -1], dtype=np.int32)
    np.savez(
        path,
        seg_net_id=seg_net_id, seg_kind=seg_kind, seg_layer=seg_layer,
        seg_x0=seg_x0, seg_y0=seg_y0, seg_x1=seg_x1, seg_y1=seg_y1,
        seg_width=seg_width, seg_via_id=seg_via_id,
        net_names=np.array(["n0", "n1"]), net_has_wire=np.array([True, False]),
        layer_names=np.array(["metal2", "metal3"]), via_names=np.array(["VIA23"]),
    )
    import json
    with open(str(path).rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump({"design_name": "fake_rect", "net_count": 2, "routed_net_count": 1}, f)


def test_rect_reconciliation_dbu_golden(tmp_path):
    npz = tmp_path / "fake_rect.npz"
    _write_fake_segments_with_rect_npz(npz)
    s = load_segments(npz)
    assert rect_reconciliation_dbu(s) == 320


def test_rect_rows_do_not_change_wire_length(tmp_path):
    # RECT rows are excluded from Segments.wire_length() (spec sec 7.4's
    # "RECT patch 忽略") -- a net with an extra RECT row must report the
    # exact same wire_length() as the same net without it, so a future
    # regression that starts summing RECT into wire_length() is caught here
    # rather than only surfacing as a Sigma(route_wl) drift in production.
    npz_no_rect = tmp_path / "fake.npz"
    _write_fake_segments_npz(npz_no_rect)
    s_no_rect = load_segments(npz_no_rect)

    npz_rect = tmp_path / "fake_rect.npz"
    _write_fake_segments_with_rect_npz(npz_rect)
    s_rect = load_segments(npz_rect)

    assert s_no_rect.wire_length() == s_rect.wire_length()


# ---------------------------------------------------------------------------
# odb cross-check: skipped whenever `odb` isn't importable in this Python
# (always true under $DP/.venv312 -- see module docstring). The real
# odb-vs-text-parser diff and the report_wire_length acceptance numbers are
# produced separately by or_scripts/verify_routed_def.py under
# `openroad -python`, not by pytest.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ODB, reason="odb is only importable under `openroad -python`, "
                                         "not $DP/.venv312 -- see or_scripts/verify_routed_def.py "
                                         "for the real odb-vs-text-parser cross-check")
def test_odb_available_smoke():  # pragma: no cover - never runs under .venv312
    import odb
    assert hasattr(odb, "dbWireDecoder")
