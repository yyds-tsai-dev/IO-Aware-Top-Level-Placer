import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ioplace.route_eval.detailed_openroad as detailed
from ioplace.route_eval.or_scripts.run_detailed_route import route_commands, count_drc
from ioplace.route_eval.or_scripts.run_detailed_route import _routing_status, _snapshot
from ioplace.route_eval.or_scripts.checked_tcl import checked_eval
import ioplace.route_eval.or_scripts.dump_online_route as online_script


def _export(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    for name, text in {
        "out.def": "VERSION 5.8 ;\nDESIGN tiny ;\nEND DESIGN\n",
        "coord.json": '{"scale_factor":1,"shift_factor":[0,0]}',
        "regions.json": "{}",
        "netmap.json": "{}",
    }.items():
        (export / name).write_text(text)
    lef = tmp_path / "cells.lef"
    lef.write_text("VERSION 5.8 ;\nEND LIBRARY\n")
    binary = tmp_path / "openroad"
    binary.write_text("binary")
    return export, lef, binary


def test_route_commands_use_full_default_detailed_route_and_emit_evidence(tmp_path):
    settings = {
        "threads": 8, "clear_script": "/clear.tcl", "cleared": "/cleared.txt",
        "routing_input": "/routing.def", "congestion": "/congestion.rpt",
        "guide": "/route.guide", "drc": "/drc.rpt", "routed_def": "/routed.def",
    }
    commands = route_commands(settings)
    assert commands == [
        "check_placement -verbose",
        "source {/clear.tcl}",
        "ioplace_clear_signal_routing $block {/cleared.txt}",
        "write_def {/routing.def}",
        "set_thread_count 8",
        "global_route -allow_congestion -congestion_report_file {/congestion.rpt} -guide_file {/route.guide}",
        "detailed_route -output_drc {/drc.rpt} -verbose 1",
        "write_def {/routed.def}",
    ]
    assert count_drc("violation type: Short\n  bbox = x\nviolation type: Spacing\n") == 2
    assert count_drc("No DRC violations found.\n") == 0


def test_adapter_records_hashes_decoded_io_native_wl_and_identity(monkeypatch, tmp_path):
    export, lef, binary = _export(tmp_path)
    out = tmp_path / "dr"

    def fake_run(command, stdout, stderr):
        settings = json.loads(Path(command[-1]).read_text())
        Path(settings["routed_def"]).write_text("routed")
        Path(settings["segments"]).write_bytes(b"segments")
        Path(settings["segments_json"]).write_text("{}")
        Path(settings["drc"]).write_text("violation type: Short\n")
        Path(settings["identity_before"]).write_text("I\tu0\t1\t2\tR0\nN\tn0\n")
        Path(settings["identity_after"]).write_text("I\tu0\t1\t2\tR0\nN\tn0\n")
        Path(settings["metrics"]).write_text(json.dumps({
            "drc_count": 1, "native_wirelength_dbu": 123,
            "decoded_wirelength_dbu": 120, "rect_reconciliation_dbu": 3,
            "component_count": 1, "net_count": 1,
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(detailed.subprocess, "run", fake_run)
    monkeypatch.setattr(detailed, "evaluate_route_from_files", lambda *a, **k:
                        SimpleNamespace(total_route_cross_raw=7, total_route_wl=125,
                                        route_cross_raw=__import__("numpy").array([7])))
    result = detailed.run_detailed_route(export, [lef], out, binary, threads=3)
    assert result["returncode"] == 0
    assert result["drc_count"] == 1
    assert result["actual_io"] == 7
    assert result["native_wirelength_dbu"] == 123
    assert result["actual_wirelength_dbu"] == 125
    assert result["identity_unchanged"] is True
    assert result["inputs"][str((export / "out.def").resolve())]
    assert {Path(path).name for path in result["inputs"]} >= {
        "detailed_openroad.py", "run_detailed_route.py", "dump_segments.py",
        "route_crossings.py", "segments.py", "evaluator_ref.py",
        "stage2_fix_def_vias.py", "clear_signal_routing.tcl", "checked_tcl.py",
    }
    assert result["outputs"]["routed.def"]
    assert json.loads((out / "settings.json").read_text())["threads"] == 3


def test_adapter_rejects_changed_placement_identity_and_existing_output(monkeypatch, tmp_path):
    export, lef, binary = _export(tmp_path)
    out = tmp_path / "dr"

    def fake_run(command, stdout, stderr):
        settings = json.loads(Path(command[-1]).read_text())
        for key in ("routed_def", "segments", "segments_json", "drc"):
            Path(settings[key]).write_text("x")
        Path(settings["identity_before"]).write_text("I\tu0\t1\t2\tR0\n")
        Path(settings["identity_after"]).write_text("I\tu0\t3\t2\tR0\n")
        Path(settings["metrics"]).write_text(json.dumps({"drc_count": 0}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(detailed.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="identity or coordinates"):
        detailed.run_detailed_route(export, [lef], out, binary)
    with pytest.raises(FileExistsError):
        detailed.run_detailed_route(export, [lef], out, binary)


def test_duplicate_def_vias_are_removed_without_modifying_lef_or_input(tmp_path):
    lef = tmp_path / "tech.lef"
    lef.write_text("VIA V12\nEND V12\n")
    source = tmp_path / "placed.def"
    source.write_text("VIAS 2 ;\n- V12\n  + RECT M1 ( 0 0 ) ( 1 1 ) ;\n"
                      "- LOCAL\n  + RECT M2 ( 0 0 ) ( 1 1 ) ;\nEND VIAS\n")
    cleaned = tmp_path / "fixed.def"
    selected, meta = detailed._prepare_def_vias(source, [lef], cleaned)
    assert selected == cleaned
    assert meta == {"applied": True, "dropped_names": ["V12"],
                    "original_count": 2, "kept_count": 1}
    assert "VIAS 1 ;" in cleaned.read_text()
    assert "- LOCAL" in cleaned.read_text() and "- V12" not in cleaned.read_text()
    assert "VIA V12" in lef.read_text() and "VIAS 2 ;" in source.read_text()


def test_unwired_nontrivial_nets_are_distinguished_from_trivial_nets():
    class Net:
        def __init__(self, name, iterms, bterms, wire=None):
            self.name, self.iterms, self.bterms, self.wire = name, iterms, bterms, wire
        def isSpecial(self): return False
        def getName(self): return self.name
        def getITerms(self): return [None] * self.iterms
        def getBTerms(self): return [None] * self.bterms
        def getWire(self): return self.wire
    block = SimpleNamespace(getNets=lambda: [
        Net("routed", 2, 0, object()), Net("floating", 1, 0),
        Net("broken", 2, 1), Net("empty", 0, 0)])
    status = _routing_status(block)
    assert status == {
        "unwired_net_count": 3, "unwired_trivial_net_count": 2,
        "unrouted_nontrivial_net_count": 1,
        "unrouted_nontrivial_net_names": ["broken"],
    }


def test_snapshot_covers_status_connectivity_and_bpin_geometry(tmp_path):
    class Box:
        def xMin(self): return 1
        def yMin(self): return 2
        def xMax(self): return 3
        def yMax(self): return 4
        def getTechLayer(self): return SimpleNamespace(getName=lambda: "M2")
    inst = SimpleNamespace(
        getOrigin=lambda: (10, 20), getName=lambda: "u0",
        getMaster=lambda: SimpleNamespace(getName=lambda: "INV"),
        getOrient=lambda: "R0", getPlacementStatus=lambda: "LOCKED")
    iterm = SimpleNamespace(getInst=lambda: inst,
                            getMTerm=lambda: SimpleNamespace(getName=lambda: "A"))
    bpin = SimpleNamespace(getPlacementStatus=lambda: "FIRM",
                           getBoxes=lambda: [Box()])
    bterm = SimpleNamespace(getName=lambda: "IN", getBPins=lambda: [bpin])
    net = SimpleNamespace(isSpecial=lambda: False, getName=lambda: "n0",
                          getITerms=lambda: [iterm], getBTerms=lambda: [bterm])
    block = SimpleNamespace(getInsts=lambda: [inst], getNets=lambda: [net],
                            getBTerms=lambda: [bterm])
    path = tmp_path / "identity.tsv"
    _snapshot(block, path)
    assert path.read_text().splitlines() == [
        "I\tu0\tINV\t10\t20\tR0\tLOCKED",
        "N\tn0\tB:IN,I:u0/A",
        "B\tIN\t0\tFIRM\tM2:1,2,3,4",
    ]


def test_checked_tcl_wraps_failure_with_nonzero_process_exit():
    calls = []
    design = SimpleNamespace(evalTclString=calls.append)
    checked_eval(design, "check_placement -verbose", label="placement")
    assert len(calls) == 1
    assert "catch {check_placement -verbose}" in calls[0]
    assert "exit 1" in calls[0]
    assert "IOPLACE_TCL_ERROR placement" in calls[0]


def test_host_rejects_logged_openroad_error_even_with_zero_return(monkeypatch, tmp_path):
    export, lef, binary = _export(tmp_path)
    out = tmp_path / "dr"
    def fake_run(command, stdout, stderr):
        stdout.write("[ERROR DPL-0033] detailed placement checks failed.\n")
        stdout.flush()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(detailed.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="logged an error"):
        detailed.run_detailed_route(export, [lef], out, binary)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["tool_returncode"] == 0
    assert receipt["returncode"] == 1
    assert receipt["logged_errors"] == [
        "[ERROR DPL-0033] detailed placement checks failed."]


def test_online_route_executes_every_tcl_command_through_checked_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr(online_script, "checked_eval",
                        lambda design, command, label: calls.append((command, label)))
    settings = {"clear_script": "/clear.tcl", "cleared": "/cleared.txt",
                "segments": "/segments.txt"}
    online_script._run_commands(object(), settings)
    assert [label for _, label in calls] == [f"online-route-{i}" for i in range(6)]
    assert [command for command, _ in calls][3:] == [
        "check_placement -verbose",
        "global_route -congestion_iterations 50 -verbose",
        "grt::write_segments {/segments.txt}",
    ]
