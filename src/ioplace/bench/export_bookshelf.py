"""M4 T4 `export_bookshelf.py`: a `PlaceDB.read` -> `write(BOOKSHELFALL)`
wrapper CLI (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 3.1
evidence 1): DREAMPlace's own
`PlaceDB.write(params, path, place_io.SolutionFileFormat.BOOKSHELFALL)`
(`$DP/dreamplace/PlaceDB.py:1010-1033` -> `BookShelfWriter::writeAll`,
`$DP/dreamplace/ops/place_io/src/BookshelfWriter.cpp:22-51,100-138`) can
losslessly export ANY loaded PlaceDB -- LEF/DEF-origin or already
Bookshelf-origin -- to a complete `.nodes/.nets/.wts/.pl/.scl/.shapes/
.route/.aux` Bookshelf design. This is the one-time conversion
`tile_bookshelf.py`'s streaming tiler is defined to read as input (sec
3.2's "(a) export_bookshelf.py <- one-shot: PlaceDB.read(mempool_group) ->
write(BOOKSHELFALL)").

Same call sequence as `ioplace/diagnostics/probes_m4/probe_bookshelf_export.py`
(T0's probe, which already exercised this on `adaptec1`), reworked here into
a reusable `ioplace/bench/` module + CLI rather than a one-off diagnostic
script. This task's scope note (sec 3.2/T4): only verified here on small
cases (`adaptec1` re-export or `mempool_tile_wrap`) -- `mempool_group`'s
production export is a later experiment task (feeding T3a/T6/T7), not this
one.

**M4 T6 bugfix (2026-08-15, discovered exporting `mempool_group`):** DEF-
origin instance/net names are `\\`-escaped hierarchical paths (`gen_tiles
\\[0\\].i_tile/...` -- see `hierarchy_gate.py`'s docstring on `DIVIDERCHAR
"/"`/`BUSBITCHARS "[]"`), and `PlaceDB.write(BOOKSHELFALL)` copies those
names verbatim into the Bookshelf files. But DREAMPlace's own Bookshelf
reader (`thirdparty/Limbo/limbo/parsers/bookshelf/bison/
BookshelfScanner.ll`) tokenizes identifiers with `[A-Za-z][A-Za-z0-9_,.\$\-
\[\]\/]*` -- no `\` in that character class -- so a name containing a
literal backslash is *not* a valid Bookshelf STRING token and the file
fails to re-parse (confirmed: `mempool_group.nodes` written straight from
`placedb.write` errors `syntax error, unexpected invalid token, expecting
integer` at the first escaped name's backslash). This was invisible on
every case exercised before `mempool_group` (`adaptec1`, `mempool_tile_wrap`)
because their names are already flat/unescaped. Since the grammar accepts
literal `[`, `]`, `/` unescaped, stripping every `\` byte from the written
`.nodes`/`.nets`/`.pl`/`.wts`/`.scl`/`.shapes`/`.route` bodies is a lossless
fix (DEF's escaping exists only to disambiguate `[`/`]` from Bookshelf's own
bus-bit syntax, which this codebase's names never otherwise use) -- `export`
below post-processes every file `placedb.write` produced, so every
downstream reader (`tile_bookshelf.py`'s streaming copy, DREAMPlace's own
`PlaceDB.read` for V1, `rent.py`'s mtkahypar/geometric Rent measurement)
sees escape-free names.

**Second M4 T6 bugfix (same discovery pass):** `mempool_group.def` elaborates
a handful of nets (4 out of 3,503,992 for `mempool_group` -- synthesis-
artifact `FE_RN_*`/`FE_DBTN*_data_o_*`/`FE_OCPN*_data_o_*` nets whose only
declared pins were pruned during LEF/DEF import, e.g. tie-cell nets)
down to **zero pins**. `PlaceDB.write` faithfully emits their `NetDegree : 0
<name>` record with no pin lines following, which is DREAMPlace's own
Bookshelf reader's *other* parse failure on this design (independent of the
backslash-escaping bug above): confirmed error `unexpected NetDegree,
expecting end of line or string` at the very next record's keyword, i.e. the
grammar does not accept a 0-pin `NetDegree` record with nothing after it.
A net with 0 pins carries no hypergraph information (it cannot appear in any
cut, degree histogram, or Rent-relevant terminal count) -- `export` below
also drops these records from `.nets` (adjusting the `NumNets`/`NumPins`
headers accordingly) rather than attempting to special-case them past a
parser that was never going to accept them.

**Third M4 T6 bugfix (same discovery pass):** `PlaceDB.write(BOOKSHELFALL)`'s
own `.shapes` output -- `shapes 1.0\\n\\nNumNonRectangularNodes : 0\\n`, with
nothing after it -- is *itself* not valid input to DREAMPlace's own reader
(`unexpected end of file, expecting end of line or string`; reproduced even
on a plain re-export of the ISPD2005 `adaptec1` benchmark T0's probe
"already exercised", which never actually re-read its output -- see that
probe's docstring/`run()`, write-only). `tile_bookshelf.py` already excludes
`.shapes`/`.route` from its own `CORE_SUFFIXES` (its docstring: neither is
read by any downstream path this bench feeds), so the fix here is the same
exclusion one level up: `export` rewrites the `.aux` file's file list to
reference only `.nodes/.nets/.wts/.pl/.scl` -- the 5 files DREAMPlace's
reader, `tile_bookshelf.py`, and `verify_bench.py` all actually use -- so
nothing downstream ever asks DREAMPlace's reader to parse `.shapes`/`.route`
again. The two files are still written to disk (unused, but not deleted).
"""
import argparse
import os
import shutil
import time

from ioplace.drivers.run_placement import _load_dreamplace

# The 5 Bookshelf file kinds actually read downstream (matches
# `tile_bookshelf.CORE_SUFFIXES`) -- `.shapes`/`.route` are written by
# `write(BOOKSHELFALL)` but excluded from `.aux` (see module docstring's
# third bugfix) and left alone otherwise (no name/net-degree content to fix).
_BOOKSHELF_BODY_SUFFIXES = ("nodes", "wts", "pl", "scl")
_AUX_SUFFIXES = ("nodes", "nets", "wts", "pl", "scl")


def _rewrite_aux(aux_path, design_name):
    """Rewrites `aux_path`'s `RowBasedPlacement : <files...>` line to
    reference only `_AUX_SUFFIXES` (module docstring's third bugfix)."""
    files = " ".join(f"{design_name}.{suf}" for suf in _AUX_SUFFIXES)
    with open(aux_path, "w") as f:
        f.write(f"RowBasedPlacement : {files}\n")


def _strip_backslash_escapes(path):
    """Rewrites `path` in place with every literal `\\` byte removed (see
    module docstring's M4 T6 bugfix note). Streams line-by-line so this is
    safe on multi-GB `.nets`/`.pl` files."""
    if not os.path.exists(path):
        return
    tmp_path = path + ".tmp"
    with open(path, "r") as fin, open(tmp_path, "w") as fout:
        for line in fin:
            if "\\" in line:
                line = line.replace("\\", "")
            fout.write(line)
    os.replace(tmp_path, path)


def _fix_nets_file(path):
    """`.nets`-specific pass: strips `\\` escapes (as `_strip_backslash_
    escapes`) *and* drops any `NetDegree : 0 <name>` record (module
    docstring's second bugfix). Two streaming passes over `.nets` (the
    largest output file, multi-GB): (1) body -> a side temp file, tracking
    the true post-drop `NumNets`; (2) corrected header + `shutil.
    copyfileobj` of the body temp file into the final path -- the body is
    never held in memory (unlike a naive read-all-lines/rewrite-header
    fixup). Returns the number of zero-degree records dropped."""
    if not os.path.exists(path):
        return 0
    body_path = path + ".body.tmp"
    n_nets_declared = n_dropped = 0
    header_lines = []
    with open(path, "r") as fin, open(body_path, "w") as fbody:
        in_header = True
        skipping = False
        for line in fin:
            if "\\" in line:
                line = line.replace("\\", "")
            if in_header:
                s = line.strip()
                if s.startswith("NumNets"):
                    n_nets_declared = int(s.split(":")[1])
                elif s.startswith("NumPins"):
                    in_header = False
                header_lines.append(line)
                continue
            s = line.strip()
            if s.startswith("NetDegree"):
                deg = int(s.split(":", 1)[1].split()[0])
                skipping = (deg == 0)
                if skipping:
                    n_dropped += 1
                    continue
                fbody.write(line)
                continue
            if skipping:
                continue
            fbody.write(line)

    out_path = path + ".tmp"
    with open(out_path, "w") as fout:
        for line in header_lines:
            if line.strip().startswith("NumNets"):
                fout.write(f"NumNets : {n_nets_declared - n_dropped}\n")
            else:
                fout.write(line)
        with open(body_path, "r") as fbody:
            shutil.copyfileobj(fbody, fout)
    os.remove(body_path)
    os.replace(out_path, path)
    return n_dropped


def export(config_json, out_prefix, *, random_seed=1000, deterministic=True):
    """Read `config_json` via DREAMPlace's `PlaceDB` and write a complete
    Bookshelf design at `out_prefix` (creating its parent directory).
    Returns a summary dict (node/net/pin counts, write time)."""
    params, placedb = _load_dreamplace(config_json)
    params.random_seed = random_seed
    params.deterministic_flag = 1 if deterministic else 0
    placedb.initialize(params)

    # Only importable after _load_dreamplace() -> setup_dreamplace() has
    # pushed $DP/install onto sys.path (see probe_bookshelf_export.py).
    import dreamplace.ops.place_io.place_io as place_io_mod

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    placedb.write(params, out_prefix, place_io_mod.SolutionFileFormat.BOOKSHELFALL)
    write_s = time.time() - t0

    t1 = time.time()
    for suf in _BOOKSHELF_BODY_SUFFIXES:
        _strip_backslash_escapes(f"{out_prefix}.{suf}")
    n_zero_degree_dropped = _fix_nets_file(f"{out_prefix}.nets")
    _rewrite_aux(f"{out_prefix}.aux", os.path.basename(out_prefix))
    unescape_s = time.time() - t1

    return {
        "config": config_json,
        "out_prefix": out_prefix,
        "write_s": write_s,
        "unescape_s": unescape_s,
        "n_physical": int(placedb.num_physical_nodes),
        "n_movable": int(placedb.num_movable_nodes),
        "n_nets": int(placedb.num_nets) - n_zero_degree_dropped,
        "n_pins": int(len(placedb.pin2node_map)),
        "n_zero_degree_nets_dropped": n_zero_degree_dropped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config_json", help="DREAMPlace config JSON (Bookshelf or LEF/DEF origin)")
    ap.add_argument("out_prefix", help="output Bookshelf prefix, e.g. results/.../mempool_group")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-deterministic", action="store_true")
    args = ap.parse_args()
    result = export(args.config_json, args.out_prefix, random_seed=args.seed,
                     deterministic=not args.no_deterministic)
    print(result)


if __name__ == "__main__":
    main()
