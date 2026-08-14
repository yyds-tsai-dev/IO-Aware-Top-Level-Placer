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
"""
import argparse
import os
import time

from ioplace.drivers.run_placement import _load_dreamplace


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

    return {
        "config": config_json,
        "out_prefix": out_prefix,
        "write_s": write_s,
        "n_physical": int(placedb.num_physical_nodes),
        "n_movable": int(placedb.num_movable_nodes),
        "n_nets": int(placedb.num_nets),
        "n_pins": int(len(placedb.pin2node_map)),
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
