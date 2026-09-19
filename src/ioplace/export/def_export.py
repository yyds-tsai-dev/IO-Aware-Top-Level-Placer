"""Stage 2 S1 -- DEF export (design spec sec 5.1, R1: geometric filtering,
no region info written into the DEF itself).

`export_def()` writes four files per run into `out_dir`:

    out.def       -- DREAMPlace's own DEF writer (PlaceDB.write, PlaceDB.py:
                      1010-1033 -> place_io.PlaceIOFunction.write ->
                      DefWriter.cpp): a line-for-line passthrough of the
                      *input* DEF (params.def_input via rawdb's defInput),
                      with only the COMPONENTS block replaced by
                      (node_x, node_y) -- via PlaceIOFunction.apply(), which
                      also bumps status to PLACED, see the inline comment
                      below for why that matters -- and ROW rewritten from
                      row data. instance/net sets are therefore identical to
                      the input DEF by construction -- the writer never
                      adds/drops anything else.
    regions.json  -- RegionSet.to_json (ioplace/regions.py:45): die +
                      lattice + the K region rects used for this run. Kept
                      out of the DEF (spec sec 5: R1 decision) -- crossing
                      extraction filters routed wire geometry against this
                      sidecar after the fact instead.
    netmap.json   -- {str(net_index): net_name}, built from
                      placedb.net_names -- the net_index<->net_name bijection
                      a routed-DEF parser (S2/S3) keys back into evaluator
                      net indices with (spec sec 7.2).
    coord.json    -- {shift_factor, scale_factor, def_units_per_micron,
                      xl, yl, xh, yh}: the DEF(DBU)<->internal coordinate
                      mapping (spec sec 7.3), read back explicitly rather
                      than hardcoded (ISPD2015 configs leave scale_factor
                      unset in the input JSON -- PlaceDB.initialize()
                      resolves it to 1/site_width, never 1.0).

DREAMPlace source under $DREAMPLACE_ROOT is never modified; this only wraps
the existing read-only C++ writer and reads plain data attributes off
`placedb`/`params`.
"""
import json
import os

import numpy as np


def export_def(placedb, params, node_x, node_y, out_dir, region_set):
    """Write out.def/regions.json/netmap.json/coord.json into `out_dir`.

    `placedb`/`params` must already be past `placedb.initialize(params)` --
    that call is what resolves `params.scale_factor` from its unset (0.0)
    config value to a concrete one (PlaceDB.py:764-767), and DEF export
    without it would silently divide by zero or (if guarded) use a wrong
    unscale factor.

    `node_x`/`node_y` must be in placedb's internal (shifted+scaled)
    coordinate system -- the same system `run_placement.extract_final_positions`
    returns positions in, and the same one `placedb.node_x`/`node_y` already
    live in. Only entries `[:placedb.num_movable_nodes]` are read by the
    underlying writer (fixed/terminal cells keep their as-read DEF position
    regardless of what's passed here), but the full physical-node-length
    array is accepted for convenience -- callers do not need to slice.

    `region_set` is the `ioplace.regions.RegionSet` describing this run's
    K-region geometry (e.g. `run_placement.get_regions_for(die, k, rtype,
    seed)`) -- not implied by placedb/params alone, so it is threaded
    through explicitly rather than reconstructed here.

    Returns a dict of the four output paths.
    """
    # Absolute before any chdir below -- out_dir may be a caller-relative
    # path, and DefWriter::write (see the chdir block further down) needs
    # the process cwd pinned to $DREAMPLACE_ROOT/install while it runs.
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    shift_factor = params.shift_factor
    scale_factor = float(params.scale_factor)
    if scale_factor == 0.0:
        raise ValueError(
            "params.scale_factor is 0.0 -- export_def must be called after "
            "placedb.initialize(params), which resolves it from the "
            "config's unset value (PlaceDB.py:764-767); calling before "
            "initialize() would divide by zero unscaling coordinates")

    # Mirrors PlaceDB.unscale_pl (PlaceDB.py:135-147) exactly, applied to the
    # caller-supplied node_x/node_y instead of placedb's own self.node_x/
    # self.node_y -- export_def must not assume the caller already wrote
    # node_x/node_y back into placedb.
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    unscale_factor = 1.0 / scale_factor
    if shift_factor[0] == 0 and shift_factor[1] == 0 and unscale_factor == 1.0:
        def_x, def_y = node_x, node_y
    else:
        def_x = node_x * unscale_factor + shift_factor[0]
        def_y = node_y * unscale_factor + shift_factor[1]

    # DREAMPlace's own DEF writer -- same call PlaceDB.write() makes
    # (PlaceDB.py:1033), just with our own def_x/def_y instead of
    # placedb.node_x/node_y. DefWriter.cpp is a passthrough of
    # userParam().defInput (params.def_input): only COMPONENTS/ROW are
    # rewritten, everything else (NETS/TRACKS/VIAS/PINS/...) copied verbatim
    # -- so instance/net identity is preserved by construction.
    #
    # DefWriter::write() opens userParam().defInput (params.def_input) as a
    # plain relative path (Params.py loads it verbatim from the config
    # JSON, e.g. "benchmarks/ispd2015/mgc_fft_1/floorplan.def"), the same
    # way params.load()/placedb.read() need cwd == $DREAMPLACE_ROOT/install
    # to resolve it (run_placement._load_dreamplace's os.chdir dance). By
    # the time GP+LG has run and export_def() is called, that chdir has
    # long since been undone -- so it must be redone here, only for the
    # duration of this call, since DREAMPlace source can't be changed to
    # take an absolute defInput instead.
    from ioplace.dreamplace_env import setup_dreamplace
    root = setup_dreamplace()
    import dreamplace.ops.place_io.place_io as place_io
    out_def = os.path.join(out_dir, "out.def")
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))
    try:
        # writeComp() (DefWriter.cpp:118-140) always emits "+ <status> ( x y )
        # <orient> ;" using node.status()/node.orient() straight off the
        # rawdb -- it never bumps status itself. Any cell whose *input* DEF
        # left it UNPLACED (mgc_fft_1's floorplan.def: all 32,281 movable
        # cells, since it's a pre-placement DEF) would otherwise still read
        # UNPLACED here even though we're handing it real coordinates,
        # producing a malformed "+ UNPLACED ( x y ) UNKNOWN ;" (found via a
        # DREAMPlace-reread round-trip: this exact construct made
        # DREAMPlace's own reader silently drop those node names, then abort
        # on the first Verilog net referencing one). place_io.PlaceIOFunction
        # .apply() is DREAMPlace's own existing fix for this -- it is what
        # NonLinearPlace.__call__ already calls at the end of a real GP+LG
        # run (see run_placement.extract_final_positions's docstring: "the
        # unscale inside apply() only feeds the separate rawdb/C++ mirror
        # used for file export") -- it sets status=PLACED and squares up
        # orient against the row, on the *rawdb* mirror only (does not touch
        # placedb.node_x/node_y). Calling it again here is idempotent for a
        # real run_io() caller (apply() already ran once) and is what makes
        # export_def() correct standalone too, without relying on every
        # caller having gone through a full NonLinearPlace run first.
        place_io.PlaceIOFunction.apply(placedb.rawdb, def_x, def_y)
        ok = place_io.PlaceIOFunction.write(
            placedb.rawdb, out_def, place_io.SolutionFileFormat.DEF, def_x, def_y)
    finally:
        os.chdir(cwd)
    if not ok:
        raise RuntimeError(f"DREAMPlace DEF writer failed for {out_def}")

    regions_json = os.path.join(out_dir, "regions.json")
    region_set.to_json(regions_json)

    net_names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in placedb.net_names]
    netmap_json = os.path.join(out_dir, "netmap.json")
    with open(netmap_json, "w") as f:
        json.dump({str(i): name for i, name in enumerate(net_names)}, f, indent=1)

    coord_json = os.path.join(out_dir, "coord.json")
    coord = {
        "shift_factor": [float(shift_factor[0]), float(shift_factor[1])],
        "scale_factor": scale_factor,
        "def_units_per_micron": int(placedb.rawdb.defUnit()),
        "xl": float(placedb.xl), "yl": float(placedb.yl),
        "xh": float(placedb.xh), "yh": float(placedb.yh),
    }
    with open(coord_json, "w") as f:
        json.dump(coord, f, indent=1)

    return {"out_def": out_def, "regions_json": regions_json,
            "netmap_json": netmap_json, "coord_json": coord_json}
