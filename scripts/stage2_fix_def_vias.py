"""Stage 2 S8 route prerequisite (`docs/superpowers/specs/2026-08-13-
stage2-innovus-calibration-plan.md` sec 10 S8 row): strip a routed/legalized
DEF's `VIAS` section of any VIA block whose name is *also* defined in the
design's own LEF (tech.lef in practice).

Why this is needed (found by `scripts/stage2_dedupe_lef.py` while chasing
TritonRoute's `[ERROR DRT-0338] Duplicated via definition` on ISPD2015 --
see that script's docstring, "Known limitation" paragraph): the duplicate
that actually blocks routing on e.g. `mgc_fft_1` is *not* a within-LEF
duplicate (`tech.lef` defines each of its VIA names exactly once). It is a
LEF-vs-DEF duplicate: 6 of `tech.lef`'s VIA names (`VIA23_2cut_{E,N,S,W}`,
`VIA34_2cut_{N,W}`) are *also* redefined -- with byte-identical geometry,
once LEF microns and DEF DBU are reconciled -- in the design's own
`floorplan.def` / `after_legalized...def`'s `VIAS` section. DREAMPlace's DEF
writer is a passthrough (`ops/place_io/src/DefWriter.cpp:12-83`, see spec
sec 3.3): whatever `VIAS` section the *input* DEF has is copied verbatim
into `--emit-def`'s `out.def`, so this duplication survives straight through
placement into the file OpenROAD is asked to route. Stripping the redundant
DEF-side blocks (keeping the LEF's own definitions, which OpenROAD/TritonRoute
already read via `read_lef`) removes the duplication without touching the
LEF at all. This has already been hand-verified once, non-programmatically,
against `mgc_fft_1`: see `results/stage2/rehearsal/mgc_fft_1/
after_legalized.ntup.fix.def.diff` (VIAS count 12 -> 6, the same 6 names
this script's algorithm identifies). This script is that fix, made
reusable/automatic and diff-preserving (same convention as
`stage2_dedupe_lef.py`: original files are never modified in place, output
goes to `--out`, and a unified diff of what changed is always written next
to it).

Scope: this script only ever reads the given LEF file(s) (name-scan only,
never rewrites them) and reads+rewrites the *one* given DEF's `VIAS`
section. It does not touch `COMPONENTS`/`NETS`/`GROUPS`/`REGIONS`/anything
else in the DEF -- those pass through completely unchanged, line for line.
It does not deduplicate VIA names *within* the DEF's own `VIAS` section
(that would be a different bug class -- see `stage2_dedupe_lef.py` for the
LEF-internal analogue) or within the LEF (ditto) -- only DEF-name-is-also-
a-LEF-name duplicates are in scope, because that is the specific case
observed to break TritonRoute on this benchmark family.

Usage:
    python3 scripts/stage2_fix_def_vias.py \\
        --lef .../tech.lef [--lef .../cells.lef ...] \\
        --def IN.def --out OUT.def [--diff OUT.def.diff]

Algorithm:
    1. Scan the given LEF file(s) for top-level `VIA <name> ...` opening
       lines (same regex family as `stage2_dedupe_lef.py`'s
       `_VIA_START_RE`) and collect the set of VIA names they define.
    2. Locate the DEF's `VIAS <count> ;` ... `END VIAS` span (regex-only,
       single pass, no general DEF grammar needed for this narrow task).
    3. Within that span, split into per-VIA blocks: each block starts at a
       line matching `^\\s*-\\s+(\\S+)\\s*$` (the `- <name>` declaration line)
       and runs up to (but not including) the next such line or `END VIAS`.
    4. Drop any block whose name is in the LEF name set; keep the rest,
       in original order. Rewrite `VIAS <count> ;` with the new kept count.
    5. Everything outside the `VIAS ... END VIAS` span passes through
       byte-for-byte unchanged.
"""
import argparse
import difflib
import os
import re
import sys

_LEF_VIA_START_RE = re.compile(r'^VIA\s+(\S+)\s+\S+\s*$')
_VIAS_HEADER_RE = re.compile(r'^VIAS\s+(\d+)\s*;\s*$')
_DEF_VIA_NAME_RE = re.compile(r'^\s*-\s+(\S+)\s*$')


def lef_via_names(lef_paths):
    """Return the set of VIA names defined (top-level `VIA <name> ...`
    opening lines) across all given LEF files."""
    names = set()
    for path in lef_paths:
        with open(path) as f:
            for line in f:
                m = _LEF_VIA_START_RE.match(line.rstrip('\n'))
                if m:
                    names.add(m.group(1))
    return names


def _split_def_via_blocks(span_lines):
    """`span_lines` is the DEF text strictly between the `VIAS <n> ;` header
    and the `END VIAS` trailer (both exclusive). Returns a list of
    (name, block_lines) -- one entry per `- <name> ... ;` via declaration,
    in original order. Any content before the first `- <name>` line (should
    not normally occur, but pass-through-safe) is attached to the first
    block, or dropped as a leading no-op block with name=None if there are
    no via blocks at all."""
    blocks = []
    cur_name = None
    cur_lines = []
    started = False
    for line in span_lines:
        m = _DEF_VIA_NAME_RE.match(line.rstrip('\n'))
        if m:
            if started:
                blocks.append((cur_name, cur_lines))
            cur_name = m.group(1)
            cur_lines = [line]
            started = True
        else:
            cur_lines.append(line)
    if started:
        blocks.append((cur_name, cur_lines))
    return blocks


def fix_def_vias(def_path, lef_paths):
    """Returns (original_lines, cleaned_lines, dropped_names, kept_count,
    original_count). Raises ValueError if the DEF has no `VIAS ... END
    VIAS` section (nothing to fix -- caller should treat that as a no-op,
    not an error, at the CLI level; raised here so callers/tests can tell
    'no VIAS section' apart from 'VIAS section with 0 dropped')."""
    with open(def_path) as f:
        original_lines = f.readlines()

    via_start = None
    declared_count = None
    for i, line in enumerate(original_lines):
        m = _VIAS_HEADER_RE.match(line.rstrip('\n'))
        if m:
            via_start = i
            declared_count = int(m.group(1))
            break
    if via_start is None:
        raise ValueError(f"{def_path}: no 'VIAS <n> ;' header found")

    via_end = None
    for i in range(via_start + 1, len(original_lines)):
        if original_lines[i].rstrip('\n') == 'END VIAS':
            via_end = i
            break
    if via_end is None:
        raise ValueError(f"{def_path}: 'VIAS' header at line {via_start + 1} "
                          f"has no matching 'END VIAS'")

    names_in_lef = lef_via_names(lef_paths)
    span = original_lines[via_start + 1:via_end]
    blocks = _split_def_via_blocks(span)

    kept_blocks = []
    dropped = []
    for name, block_lines in blocks:
        if name is not None and name in names_in_lef:
            dropped.append(name)
            continue
        kept_blocks.append(block_lines)

    kept_lines = [line for block in kept_blocks for line in block]
    new_header = f"VIAS {len(kept_blocks)} ;\n"

    cleaned_lines = (original_lines[:via_start] + [new_header] + kept_lines
                      + original_lines[via_end:])

    return original_lines, cleaned_lines, dropped, len(kept_blocks), declared_count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lef", action="append", required=True,
                     help="LEF file, repeatable; only scanned for VIA names, "
                          "never modified (put tech.lef first, matching "
                          "stage2_dedupe_lef.py's convention, though order "
                          "does not affect this script's output).")
    ap.add_argument("--def", dest="def_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diff", default=None,
                     help="unified diff output path (default: <out>.diff)")
    args = ap.parse_args(argv)

    diff_path = args.diff or (args.out + ".diff")

    original_lines, cleaned_lines, dropped, kept_count, declared_count = \
        fix_def_vias(args.def_path, args.lef)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(cleaned_lines)

    diff_lines = list(difflib.unified_diff(
        original_lines, cleaned_lines, fromfile=args.def_path, tofile=args.out))
    with open(diff_path, "w") as f:
        if diff_lines:
            f.writelines(diff_lines)
        else:
            f.write(f"# no changes: {args.def_path}'s VIAS section had no "
                     f"blocks whose name is also defined in the given LEF(s)\n")

    if dropped:
        print(f"[fix_def_vias] {args.def_path}: VIAS {declared_count} -> "
              f"{kept_count} (dropped {len(dropped)} LEF-duplicate VIA "
              f"block(s): {dropped})")
    else:
        print(f"[fix_def_vias] {args.def_path}: VIAS {declared_count} "
              f"unchanged (no-op; 0 blocks matched a LEF VIA name; wrote "
              f"unchanged copy to {args.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
