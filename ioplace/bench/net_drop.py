"""M4 T0b net-drop variant generator (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.1 T0b,
Codex D4): the T0b bins/target_density sweep on `mempool_group` cannot move
`N_pins` independently of `N_total` -- the netlist is fixed, so pin count is
a constant across every point in that sweep, and the GP-memory design matrix
is unidentifiable in the pin-count coefficient without a design point that
varies it. This module produces that design point: given a Bookshelf design,
drop a fixed fraction of its *nets* (uniformly at random, fixed seed) while
keeping every node -- `N_pins` (and `N_total`... well, `N_total` is nodes +
fillers, unaffected) moves independently of node count and bin count.

Output convention matches `tile_bookshelf.tile`'s (the other T4/T0b-family
Bookshelf-to-Bookshelf transform in this package): a full standalone 5-file
design is written at `dst_prefix` (`.nodes`/`.wts`/`.pl`/`.scl` copied
verbatim -- the node/row/weight tables are untouched by a net-only edit --
`.nets` rewritten with the dropped net blocks removed and `NumNets`/
`NumPins` headers corrected, `.aux` rewritten to reference the new prefix's
own files), plus a `<dst_prefix>.manifest.json` recording the source design,
seed, requested fraction, and the actual before/after net and pin counts.

CLI (design draft sec 7.1 T0b deliverable 1's fixed interface):
    PYTHONPATH=. $PY -m ioplace.bench.net_drop \\
        --source results/m4/bench/mempool_group_export/mempool_group \\
        --drop 0.25 --seed 0 --out-dir results/m4/bench/netdrop
`dst_prefix` is derived from `--out-dir` + the source basename + the
(2-decimal-formatted) drop fraction + seed via `derive_dst_prefix` below --
not a caller-supplied path -- so every net-drop variant this tool ever
produces lands at a name downstream tools (e.g. M4 T0b's
`run_t0b_matrix.py`) can reconstruct from `(out_dir, source_prefix,
drop_fraction, seed)` alone, without reading a directory listing.
"""
import argparse
import json
import os
import shutil

import numpy as np

from ioplace.bench import tile_bookshelf as tb


def _stream_drop_nets(nets_path, body_out_path, dropped_indices):
    """Single streaming pass over `nets_path`'s net blocks: writes every
    block whose 0-based position in file order is *not* in
    `dropped_indices` to `body_out_path` (header lines excluded -- the
    caller writes a fresh header once the final kept counts are known,
    same two-pass shape as `export_bookshelf._fix_nets_file`). Returns
    (n_nets_kept, n_pins_kept)."""
    n_nets_kept = n_pins_kept = 0
    idx = -1
    skipping = False
    with open(nets_path) as fin, open(body_out_path, "w") as fbody:
        started = False
        for line in fin:
            s = line.strip()
            if not started:
                if s.startswith("NumPins"):
                    started = True
                continue
            if not s:
                continue
            if s.startswith("NetDegree"):
                idx += 1
                skipping = idx in dropped_indices
                if skipping:
                    continue
                fbody.write(line)
                n_nets_kept += 1
                continue
            if skipping:
                continue
            fbody.write(line)
            n_pins_kept += 1
    return n_nets_kept, n_pins_kept


def drop_nets(src_prefix, dst_prefix, drop_fraction, seed=0):
    """Write a full standalone Bookshelf design at `dst_prefix`: same nodes/
    rows/weights as `src_prefix`, but with `round(drop_fraction *
    n_nets_source)` nets removed uniformly at random (`numpy.random.
    default_rng(seed).choice(n_nets_source, size=n_drop, replace=False)` --
    same RNG convention as `glue_gen.py`'s samplers). Returns the manifest
    dict (also written to `<dst_prefix>.manifest.json`)."""
    if not (0.0 <= drop_fraction <= 1.0):
        raise ValueError(f"drop_fraction={drop_fraction} must be in [0, 1]")

    paths = tb.read_aux(src_prefix + ".aux")
    n_nets_src, n_pins_src = tb._nets_header(paths["nets"])

    n_drop = round(drop_fraction * n_nets_src)
    rng = np.random.default_rng(seed)
    dropped_indices = set(int(i) for i in rng.choice(n_nets_src, size=n_drop, replace=False))

    out_dir = os.path.dirname(dst_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    src_sha256 = {suf: tb.sha256_file(p) for suf, p in paths.items()}

    # .nets: filtered body (two-pass: body to a temp file while counting,
    # then a fresh file with the corrected header prepended).
    body_tmp = dst_prefix + ".nets.body.tmp"
    n_nets_kept, n_pins_kept = _stream_drop_nets(paths["nets"], body_tmp, dropped_indices)
    with open(dst_prefix + ".nets", "w") as fout:
        fout.write("UCLA nets 1.0\n\n")
        fout.write(f"NumNets : {n_nets_kept}\n")
        fout.write(f"NumPins : {n_pins_kept}\n\n")
        with open(body_tmp) as fbody:
            shutil.copyfileobj(fbody, fout)
    os.remove(body_tmp)

    # .nodes/.wts/.pl/.scl: node set (and everything about it) is untouched
    # by dropping nets -- copied verbatim.
    for suf in ("nodes", "wts", "pl", "scl"):
        shutil.copy(paths[suf], dst_prefix + "." + suf)

    # .aux
    base = os.path.basename(dst_prefix)
    with open(dst_prefix + ".aux", "w") as fout:
        fout.write(f"RowBasedPlacement : {' '.join(base + '.' + s for s in tb.CORE_SUFFIXES)}\n")

    out_sha256 = {suf: tb.sha256_file(dst_prefix + "." + suf) for suf in tb.CORE_SUFFIXES}

    manifest = {
        "source_prefix": os.path.abspath(src_prefix),
        "source_sha256": src_sha256,
        "seed": seed,
        "drop_fraction": drop_fraction,
        "n_nets_source": n_nets_src,
        "n_nets_dropped": n_nets_src - n_nets_kept,
        "n_nets_kept": n_nets_kept,
        "n_pins_source": n_pins_src,
        "n_pins_dropped": n_pins_src - n_pins_kept,
        "n_pins_kept": n_pins_kept,
        "output_sha256": out_sha256,
    }
    with open(dst_prefix + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


def derive_dst_prefix(out_dir, source_prefix, drop_fraction, seed):
    """`<out_dir>/<source_basename>__drop<XX.XX>__seed<seed>` -- the one
    naming rule every net-drop variant's output prefix is derived by,
    shared between this module's CLI and any caller (e.g. M4 T0b's
    `run_t0b_matrix.py`) that needs to point a DREAMPlace config at a
    variant without re-running `drop_nets`. `drop_fraction` is formatted to
    2 decimals (`0.25` -> `0.25`, `0.5` -> `0.50`) so 25%/50% sort and read
    consistently regardless of how the caller wrote the float."""
    base = os.path.basename(os.path.normpath(source_prefix))
    return os.path.join(out_dir, f"{base}__drop{drop_fraction:.2f}__seed{seed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, dest="source",
                     help="source Bookshelf prefix (must have a .aux)")
    ap.add_argument("--drop", required=True, type=float, dest="drop_fraction",
                     help="fraction of nets to drop, e.g. 0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, dest="out_dir",
                     help="output directory; the output prefix within it is "
                          "derived by derive_dst_prefix()")
    args = ap.parse_args()
    dst_prefix = derive_dst_prefix(args.out_dir, args.source, args.drop_fraction, args.seed)
    manifest = drop_nets(args.source, dst_prefix, args.drop_fraction, seed=args.seed)
    print(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
