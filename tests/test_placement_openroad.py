import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ioplace.route_eval.placement_openroad as placement
from ioplace.route_eval.or_scripts.legalize_placement import repair_commands, _replace_components


def _inputs(tmp_path):
    export = tmp_path / "export"; export.mkdir()
    (export / "out.def").write_text("VERSION 5.8 ;\nDESIGN tiny ;\nEND DESIGN\n")
    lef = tmp_path / "cells.lef"; lef.write_text("VERSION 5.8 ;\nEND LIBRARY\n")
    binary = tmp_path / "openroad"; binary.write_text("binary")
    return export, lef, binary


def test_repair_commands_use_common_full_design_bound_and_checked_postcondition():
    assert repair_commands(731, 12) == [
        "set_thread_count 12",
        "detailed_placement -max_displacement {731 731}",
        "check_placement -verbose",
    ]


def test_adapter_returns_requested_movable_bbox_coordinates_and_receipt(monkeypatch, tmp_path):
    export, lef, binary = _inputs(tmp_path)
    out = tmp_path / "legal"
    def fake_run(command, stdout, stderr):
        settings = json.loads(Path(command[-1]).read_text())
        Path(settings["preplacement_def"]).write_text("before")
        Path(settings["openroad_legalized_def"]).write_text("openroad after")
        Path(settings["legalized_def"]).write_text("after")
        Path(settings["identity_before"]).write_text("immutable")
        Path(settings["identity_after"]).write_text("immutable")
        np.savez(settings["coordinates"], names=np.array(["u1", "u0"]),
                 x_dbu=np.array([30, 10]), y_dbu=np.array([40, 20]),
                 orientations=np.array(["MY", "R0"]))
        Path(settings["report"]).write_text(json.dumps({
            "before_check_code": 1, "displacement_bound_microns": 100,
            "changed_location_count": 2, "changed_orientation_count": 1,
            "average_displacement_dbu": 3., "maximum_displacement_dbu": 5.,
            "average_displacement_microns": .003,
            "maximum_displacement_microns": .005,
        }))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(placement.subprocess, "run", fake_run)
    result = placement.legalize_export(export, [lef], out, binary, ["u1", "u0"], threads=4)
    assert result["x_dbu"].tolist() == [30, 10]
    assert result["y_dbu"].tolist() == [40, 20]
    assert result["orientations"].tolist() == ["MY", "R0"]
    assert result["legal_def_path"] == str((out / "legalized.def").resolve())
    assert result["receipt"]["before_check_code"] == 1
    assert result["receipt"]["immutable_identity_unchanged"] is True
    assert Path(result["receipt_path"]).is_file()


def test_adapter_rejects_reordered_or_missing_names_and_logged_errors(monkeypatch, tmp_path):
    export, lef, binary = _inputs(tmp_path)
    def fake_run(command, stdout, stderr):
        settings = json.loads(Path(command[-1]).read_text())
        Path(settings["preplacement_def"]).write_text("before")
        Path(settings["openroad_legalized_def"]).write_text("openroad after")
        Path(settings["legalized_def"]).write_text("after")
        Path(settings["identity_before"]).write_text("same")
        Path(settings["identity_after"]).write_text("same")
        np.savez(settings["coordinates"], names=np.array(["wrong"]),
                 x_dbu=np.array([1]), y_dbu=np.array([2]),
                 orientations=np.array(["R0"]))
        Path(settings["report"]).write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(placement.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="movable name order"):
        placement.legalize_export(export, [lef], tmp_path / "bad", binary, ["u0"])

    def error_run(command, stdout, stderr):
        stdout.write("[ERROR DPL-0033] failed\n"); stdout.flush()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(placement.subprocess, "run", error_run)
    with pytest.raises(RuntimeError, match="failed"):
        placement.legalize_export(export, [lef], tmp_path / "error", binary, ["u0"])


def test_expected_precheck_error_is_not_a_fatal_log_error(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("IOPLACE_EXPECTED_PRECHECK_BEGIN\n[ERROR DPL-0033] before\n"
                   "IOPLACE_EXPECTED_PRECHECK_END code=1\n[ERROR DPL-9999] after\n")
    assert placement._logged_errors(log) == ["[ERROR DPL-9999] after"]


def test_persisted_def_replaces_only_components_and_preserves_original_nets(tmp_path):
    original = tmp_path / "original.def"
    raw = tmp_path / "openroad.def"
    final = tmp_path / "final.def"
    original.write_text("HEAD\nCOMPONENTS 1 ;\n- u0 INV + PLACED ( 0 0 ) N ;\n"
                        "END COMPONENTS\nNETS 1 ;\n- VDD\n( PIN VDD )\n;\nEND NETS\nTAIL\n")
    raw.write_text("OTHERHEAD\nCOMPONENTS 1 ;\n- u0 INV + PLACED ( 10 20 ) FS ;\n"
                   "END COMPONENTS\nNETS 0 ;\nEND NETS\nOTHERTAIL\n")
    _replace_components(original, raw, final)
    assert final.read_text() == (
        "HEAD\nCOMPONENTS 1 ;\n- u0 INV + PLACED ( 10 20 ) FS ;\n"
        "END COMPONENTS\nNETS 1 ;\n- VDD\n( PIN VDD )\n;\nEND NETS\nTAIL\n")


def test_identity_comparison_is_order_independent_but_content_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("M\tu0\tINV\nN\tn0\tI:u0/A\n")
    b.write_text("N\tn0\tI:u0/A\nM\tu0\tINV\n")
    assert placement._identity_digest(a) == placement._identity_digest(b)
    b.write_text("N\tn0\tI:u0/Z\nM\tu0\tINV\n")
    assert placement._identity_digest(a) != placement._identity_digest(b)
