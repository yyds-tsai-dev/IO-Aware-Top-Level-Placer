"""Stage 2 S4 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
plan.md` sec 10 S4 row / sec 11 R4): deterministic VIA dedupe for a set of
LEF files, keeping the first definition of any duplicated VIA name and
dropping later occurrences.

Scope: this script only ever reads and rewrites the LEF files it is given
(`--lef`, repeatable, in read order -- put tech.lef first). It never touches
any DEF. Benchmark originals are never modified in place: outputs go to
`--out-dir`, one file per input LEF (same basename) plus a unified diff per
file that actually changed.

Usage:
    python3 scripts/stage2_dedupe_lef.py \\
        --lef .../tech.lef --lef .../cells.lef \\
        --out-dir results/stage2/rehearsal/<design>/lef_fixed/

Algorithm: scan each LEF's top-level `VIA <name> ...  END <name>` blocks in
file order (files themselves processed in the order given on the command
line, so "first definition" spans across files, not just within one). The
first time a VIA name is seen, its block is kept; every later occurrence
(same file or a later file) is dropped from its file's output. Everything
that isn't a VIA block (macros, layers, rules, ...) passes through
unchanged.

**Known limitation, found running this against ISPD2015 mgc_fft_1 (sec 11
R4's DRT-0338 case) while building this script:** TritonRoute's "duplicated
via definition" error is not always a *within-LEF* duplicate. For mgc_fft_1
specifically, `tech.lef` defines each of its 58 VIA names exactly once (no
LEF-internal duplication at all -- `cells.lef` has zero VIA blocks) -- so
this script is a documented no-op on that design/LEF pair. The actual
duplicate there is LEF-vs-DEF: 6 of tech.lef's VIA names (VIA23_2cut_{E,N,
S,W}, VIA34_2cut_{N,W}) are *also* redefined, with byte-identical geometry
(down to the RECT coordinates, once LEF microns and DEF DBU are reconciled),
in the design's own legalized DEF's `VIAS` section. That is a different bug
class from what this script fixes, and fixing it means editing the DEF (or
suppressing the DEF's redundant VIAS entries), not the LEF -- out of this
script's stated scope. See this task's final report for the full finding
and which artifact was used to unblock routing instead.
"""
import argparse
import difflib
import os
import re
import sys

_VIA_START_RE = re.compile(r'^VIA\s+(\S+)\s+\S+\s*$')


def _split_via_blocks(lines):
    """Return a list of (name_or_None, block_lines) covering the whole file
    in order. name_or_None is the VIA name for a `VIA <name> ... END <name>`
    block, or None for any other line (pass-through, kept verbatim, one
    "block" per non-VIA line so ordering/whitespace is preserved exactly).
    """
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        m = _VIA_START_RE.match(lines[i].rstrip('\n'))
        if m:
            name = m.group(1)
            end_marker = f"END {name}"
            j = i
            block = [lines[i]]
            j += 1
            while j < n and lines[j].rstrip('\n') != end_marker:
                block.append(lines[j])
                j += 1
            if j < n:
                block.append(lines[j])  # the END line itself
                j += 1
            blocks.append((name, block))
            i = j
        else:
            blocks.append((None, [lines[i]]))
            i += 1
    return blocks


def dedupe_lefs(lef_paths):
    """Returns dict: lef_path -> (original_lines, cleaned_lines, dropped_names)."""
    seen = set()
    result = {}
    for path in lef_paths:
        with open(path) as f:
            original_lines = f.readlines()
        blocks = _split_via_blocks(original_lines)
        cleaned_lines = []
        dropped = []
        for name, block in blocks:
            if name is None:
                cleaned_lines.extend(block)
                continue
            if name in seen:
                dropped.append(name)
                continue
            seen.add(name)
            cleaned_lines.extend(block)
        result[path] = (original_lines, cleaned_lines, dropped)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lef", action="append", required=True,
                     help="LEF file, repeatable; given order is read order "
                          "(put tech.lef first) and determines which "
                          "occurrence of a duplicated VIA name is 'first'.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    result = dedupe_lefs(args.lef)

    total_dropped = 0
    for path, (original_lines, cleaned_lines, dropped) in result.items():
        base = os.path.basename(path)
        out_path = os.path.join(args.out_dir, base)
        with open(out_path, "w") as f:
            f.writelines(cleaned_lines)

        diff_path = out_path + ".diff"
        diff_lines = list(difflib.unified_diff(
            original_lines, cleaned_lines, fromfile=path, tofile=out_path))
        with open(diff_path, "w") as f:
            if diff_lines:
                f.writelines(diff_lines)
            else:
                f.write(f"# no changes: {path} had no duplicate VIA "
                        f"definitions relative to files processed before it\n")

        total_dropped += len(dropped)
        if dropped:
            print(f"[dedupe] {path}: dropped {len(dropped)} duplicate VIA "
                  f"block(s): {dropped}")
        else:
            print(f"[dedupe] {path}: 0 duplicate VIA blocks found "
                  f"(no-op; wrote unchanged copy to {out_path})")

    print(f"[dedupe] TOTAL duplicate VIA blocks removed across "
          f"{len(args.lef)} file(s): {total_dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
