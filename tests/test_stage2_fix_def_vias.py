"""Stage 2 S8 route prerequisite (`docs/superpowers/specs/2026-08-13-
stage2-innovus-calibration-plan.md` sec 10 S8 row): unit tests for
`scripts/stage2_fix_def_vias.py` -- strips a DEF's `VIAS` section of any
block whose name is also defined in the given LEF(s), matching the
LEF-vs-DEF via duplication observed on ISPD2015 (see that script's
docstring and `results/stage2/rehearsal/mgc_fft_1/
after_legalized.ntup.fix.def.diff`, VIAS 12 -> 6). These are hand-crafted
small fixtures, not the real multi-hundred-KB ISPD2015 DEF (that
cross-check was done manually once against the real `mgc_fft_1` DEF while
building the script, reproducing the existing rehearsal diff byte-for-byte;
this file exercises the algorithm's edge cases instead: multi-drop, order
preservation, no-op, malformed input).
"""
import os
import subprocess
import sys

import pytest

from scripts.stage2_fix_def_vias import fix_def_vias, lef_via_names, main

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LEF = """\
VERSION 5.7 ;
VIA VIA_A DEFAULT
   LAYER metal1 ;
      RECT -50 -50 50 50 ;
END VIA_A
VIA VIA_B DEFAULT
   LAYER metal3 ;
      RECT -20 -20 20 20 ;
END VIA_B
END LIBRARY
"""

# VIAS section declares 3 vias in this order: VIA_A (dup w/ LEF), VIA_C
# (not in LEF), VIA_B (dup w/ LEF) -- exercises both "drop a non-adjacent
# pair" and "preserve order/content of the surviving block".
_DEF_WITH_DUPES = """\
VERSION 5.7 ;
DESIGN test ;
UNITS DISTANCE MICRONS 2000 ;
DIEAREA ( 0 0 ) ( 1000 1000 ) ;
COMPONENTS 1 ;
   - inst1 CELL1 + PLACED ( 0 0 ) N ;
END COMPONENTS
TRACKS Y 0 DO 10 STEP 100 LAYER metal1 ;
VIAS 3 ;
   - VIA_A
      + RECT metal1 ( -50 -50 ) ( 50 50 ) ;
   - VIA_C
      + RECT metal2 ( -10 -10 ) ( 10 10 ) ;
   - VIA_B
      + RECT metal3 ( -20 -20 ) ( 20 20 ) ;
END VIAS
NETS 1 ;
   - net1 ( inst1 A ) + ROUTED metal1 ( 0 0 ) ( 100 100 ) ;
END NETS
END DESIGN
"""

_DEF_NO_DUPES = _DEF_WITH_DUPES.replace(
    "VIAS 3 ;\n"
    "   - VIA_A\n"
    "      + RECT metal1 ( -50 -50 ) ( 50 50 ) ;\n"
    "   - VIA_C\n"
    "      + RECT metal2 ( -10 -10 ) ( 10 10 ) ;\n"
    "   - VIA_B\n"
    "      + RECT metal3 ( -20 -20 ) ( 20 20 ) ;\n"
    "END VIAS\n",
    "VIAS 1 ;\n"
    "   - VIA_C\n"
    "      + RECT metal2 ( -10 -10 ) ( 10 10 ) ;\n"
    "END VIAS\n",
)

_DEF_NO_VIAS_SECTION = """\
VERSION 5.7 ;
DESIGN test ;
COMPONENTS 0 ;
END COMPONENTS
NETS 0 ;
END NETS
END DESIGN
"""


@pytest.fixture
def lef_path(tmp_path):
    p = tmp_path / "tech.lef"
    p.write_text(_LEF)
    return str(p)


@pytest.fixture
def def_with_dupes_path(tmp_path):
    p = tmp_path / "in.def"
    p.write_text(_DEF_WITH_DUPES)
    return str(p)


@pytest.fixture
def def_no_dupes_path(tmp_path):
    p = tmp_path / "in_nodupes.def"
    p.write_text(_DEF_NO_DUPES)
    return str(p)


# ---------------------------------------------------------------------------
# lef_via_names
# ---------------------------------------------------------------------------

def test_lef_via_names(lef_path):
    assert lef_via_names([lef_path]) == {"VIA_A", "VIA_B"}


def test_lef_via_names_multiple_files(tmp_path, lef_path):
    other = tmp_path / "cells.lef"
    other.write_text("VERSION 5.7 ;\nEND LIBRARY\n")  # no VIA blocks
    assert lef_via_names([lef_path, str(other)]) == {"VIA_A", "VIA_B"}


# ---------------------------------------------------------------------------
# fix_def_vias -- the algorithm
# ---------------------------------------------------------------------------

def test_drops_lef_duplicate_vias_only(def_with_dupes_path, lef_path):
    original, cleaned, dropped, kept_count, declared_count = fix_def_vias(
        def_with_dupes_path, [lef_path])
    assert declared_count == 3
    assert dropped == ["VIA_A", "VIA_B"]  # order = order encountered in DEF
    assert kept_count == 1


def test_cleaned_output_keeps_only_non_lef_via_and_updates_count(
        def_with_dupes_path, lef_path):
    _, cleaned, _, _, _ = fix_def_vias(def_with_dupes_path, [lef_path])
    text = "".join(cleaned)
    assert "VIAS 1 ;" in text
    assert "VIA_A" not in text
    assert "VIA_B" not in text
    assert "VIA_C" in text
    # everything outside VIAS..END VIAS passes through untouched
    assert "COMPONENTS 1 ;" in text
    assert "- inst1 CELL1 + PLACED ( 0 0 ) N ;" in text
    assert "NETS 1 ;" in text
    assert text == _DEF_NO_DUPES


def test_noop_when_no_names_overlap(tmp_path):
    lef = tmp_path / "tech.lef"
    lef.write_text("VERSION 5.7 ;\nVIA UNRELATED DEFAULT\n"
                    "   LAYER metal1 ;\n      RECT 0 0 1 1 ;\n"
                    "END UNRELATED\nEND LIBRARY\n")
    d = tmp_path / "in.def"
    d.write_text(_DEF_WITH_DUPES)
    original, cleaned, dropped, kept_count, declared_count = fix_def_vias(
        str(d), [str(lef)])
    assert dropped == []
    assert kept_count == declared_count == 3
    assert cleaned == original


def test_raises_on_missing_vias_section(tmp_path, lef_path):
    d = tmp_path / "novias.def"
    d.write_text(_DEF_NO_VIAS_SECTION)
    with pytest.raises(ValueError, match="no 'VIAS <n> ;' header found"):
        fix_def_vias(str(d), [lef_path])


def test_raises_on_unterminated_vias_section(tmp_path, lef_path):
    d = tmp_path / "unterminated.def"
    d.write_text("VIAS 1 ;\n   - VIA_A\n      + RECT metal1 ( 0 0 ) ( 1 1 ) ;\n")
    with pytest.raises(ValueError, match="no matching 'END VIAS'"):
        fix_def_vias(str(d), [lef_path])


# ---------------------------------------------------------------------------
# CLI (main()) -- writes --out and a unified diff to --diff
# ---------------------------------------------------------------------------

def test_cli_writes_out_and_diff(tmp_path, def_with_dupes_path, lef_path):
    out = tmp_path / "out.def"
    diff = tmp_path / "out.def.diff"
    rc = main(["--lef", lef_path, "--def", def_with_dupes_path,
               "--out", str(out), "--diff", str(diff)])
    assert rc == 0
    assert out.read_text() == _DEF_NO_DUPES
    diff_text = diff.read_text()
    assert "-VIAS 3 ;" in diff_text
    assert "+VIAS 1 ;" in diff_text
    assert "-   - VIA_A" in diff_text
    assert "-   - VIA_B" in diff_text
    assert "   - VIA_C" in diff_text  # kept, unchanged (no +/- prefix on context line)


def test_cli_default_diff_path(tmp_path, def_with_dupes_path, lef_path):
    out = tmp_path / "out.def"
    rc = main(["--lef", lef_path, "--def", def_with_dupes_path, "--out", str(out)])
    assert rc == 0
    assert (tmp_path / "out.def.diff").exists()


def test_cli_noop_diff_has_explanatory_comment(tmp_path, def_no_dupes_path, lef_path):
    # def_no_dupes_path's only via (VIA_C) is not in lef_path's LEF -> no-op.
    out = tmp_path / "out.def"
    diff = tmp_path / "out.def.diff"
    rc = main(["--lef", lef_path, "--def", def_no_dupes_path,
               "--out", str(out), "--diff", str(diff)])
    assert rc == 0
    assert out.read_text() == _DEF_NO_DUPES
    assert "no changes" in diff.read_text()


def test_cli_subprocess_smoke(tmp_path, def_with_dupes_path, lef_path):
    """Exercise the script the way stage2_s8_route.sh will invoke it: as a
    plain `python3` subprocess (no torch/DP needed -- pure text)."""
    out = tmp_path / "out.def"
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "src", "scripts", "stage2_fix_def_vias.py"),
         "--lef", lef_path, "--def", def_with_dupes_path, "--out", str(out)],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    assert "VIAS 3 -> 1" in result.stdout
    assert out.read_text() == _DEF_NO_DUPES
