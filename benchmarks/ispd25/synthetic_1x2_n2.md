# `synthetic_1x2_n2.json` -- derivation notes

M4 T8 synthetic-arm config for the tiler's 1x2 array (`mempool_group` x2,
Bookshelf, n2 glue normalization -- N1 vetoed by the 2026-08-15 T6 holdout
adjudication, see `docs/results/2026-08-15-m4-t6-holdout-adjudication.md`
sec 3). Bookshelf input:
`results/m4/bench/arrays/1x2_n2/1x2_n2.aux` (6.18M nodes / 6.16M movable
cells per the design draft's "6.2M" ladder rung,
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 2.1).

## Format template

Structural fields (`aux_input`, `scale_factor`, `gift_init_flag`) follow
`$DP/install/test/ispd2005/bigblue4.json` (the repo's only other Bookshelf-
input DREAMPlace config) since `mempool_group.json`/`mempool_cluster.json`/
`mempool_tile_wrap.json` are all LEF/DEF and don't carry those fields.
Numeric GP-recipe fields (`target_density`, `density_weight`, `gamma`,
`random_seed`, `deterministic_flag`, `ignore_net_degree`, `enable_fillers`,
`gp_noise_ratio`, `stop_overflow`, `dtype`, `random_center_init_flag`,
`sort_nets_by_degree`) are copied verbatim from `mempool_group.json`, since
this array is a *precise tiling* of `mempool_group` and the goal is to
reproduce its GP recipe unchanged, not bigblue4's (which targets a smaller,
unrelated ISPD2005 design and uses `target_density=1.0`). JSON layout
(1-space indent, `"key": value`) matches this repo's own
`benchmarks/ispd25/mempool_group.json` rather than bigblue4.json's
inconsistent DREAMPlace-install spacing.

`detailed_place_flag=0` (matching `mempool_group.json`, not bigblue4.json's
`1`): the M4 measurement protocol runs "GP+LG only, DP off" throughout
(design draft sec 1.1 table header, "DREAMPlace GP+LG(det=1、DP 關...)"),
so keeping detailed placement off is required for T8's numbers to be
comparable to the rest of the ladder, not a stylistic pick.

`num_threads=16` (matching group, not bigblue4's `8`) and `sol_file_format`
is dropped entirely: it is dead in this codebase (`Placer.py:79` calls
`placedb.write(params, gp_out_file)` with no `sol_file_format` arg, so the
key mempool_group.json carries is never read) and doesn't apply to a
Bookshelf-input config the way it did for group's DEF output anyway.

## `aux_input`: absolute path, verified safe

Written as an absolute path
(`/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/m4/bench/arrays/1x2_n2/1x2_n2.aux`),
per the repo's own T3-corpus-config convention (`ioplace/diagnostics/
probes_m4/probe_corpus_stats.py`'s docstring: "Configs hold absolute paths
... unlike DREAMPlace's own test configs which are relative to $DP/install").

Verified (code read, not executed) that this is safe regardless of the
loader's cwd: the Bookshelf reader
(`$DP/thirdparty/Limbo/limbo/parsers/bookshelf/bison/BookshelfDriver.cc:343-402`)
resolves each file referenced inside the `.aux` line
(`RowBasedPlacement : 1x2_n2.nodes 1x2_n2.nets 1x2_n2.wts 1x2_n2.pl
1x2_n2.scl`) as `auxPath + "/" + filename`, where `auxPath =
get_file_path(auxFile)` is derived from the `aux_input` string itself
(`:362`) -- i.e. from the directory portion of whatever path we hand it, not
from the process's current working directory. This is a different code path
from `ioplace/netlist.py`'s `load_netlist()`, whose docstring warns that
*its own* `aux_input` handling is relative to `$DREAMPLACE_ROOT/install`
(that helper does an explicit `os.chdir()` and is unrelated to
`Params.load()`/`PlaceDB.read()`, which is what these DREAMPlace config
JSONs feed). An absolute `aux_input` sidesteps both code paths' cwd
assumptions cleanly.

## `target_density = 0.714`

Copied unchanged from `mempool_group.json`. Tiling by direct replication (2
non-overlapping copies of the identical `mempool_group` tile side by side,
plus a small number of inter-tile glue nets that touch existing pins only --
see `1x2_n2.manifest.json`'s `glue` block, no new nodes) does not change the
movable-area / free-area ratio per unit area, so the utilization that
`target_density` targets is unchanged from the single-tile source and no
recomputation is needed.

## Bin sizing: `num_bins_x=4096, num_bins_y=2048`

`mempool_group.json` uses `num_bins_x=num_bins_y=2048` over the standalone
group's die. `num_bins_x`/`num_bins_y` are independent, DREAMPlace-supported
fields (`$DP/dreamplace/params.json` lines 26-33: separate "number of bins
in horizontal/vertical direction" entries, also accepted per-stage inside
`global_place_stages`), so bin *count* can differ per axis without any
fallback compromise.

The 1x2 array's manifest (`results/m4/bench/arrays/1x2_n2/1x2_n2.manifest.json`)
records `"C": 2, "R": 1` -- 2 tile-columns, 1 tile-row, i.e. the array is
built by placing 2 exact copies of the group tile side by side along x, with
the y extent unchanged from a single group tile. Because the width scales by
exactly the tiler's own column count (2x) and the height by exactly its row
count (1x, unchanged), preserving the source's physical bin size (`die_size /
num_bins`) on both axes reduces to scaling `num_bins` by the same integer
factors: `num_bins_x = 2 * 2048 = 4096`, `num_bins_y = 1 * 2048 = 2048`. This
holds exactly regardless of the die's absolute DBU/micron values (no
site-width arithmetic needed) -- it only relies on the array being an exact
integer C x R replication, which the manifest's `C`/`R` fields confirm.

Cross-check against the task's own order-of-magnitude figures ("group
2048²/17754寬"): the group tile's placeable-area width in NanGate45 site
units is `tile_width / site_width_dbu = 6746520 / 380 = 17754.0` exactly
(from `1x2_n2.manifest.json`'s `tile_width`; `mempool_group.def`'s `UNITS
DISTANCE MICRONS 2000` and a 0.19 um NanGate45 site pitch give
`site_width_dbu = 380`), consistent with the instruction's "17754寬" figure
and the "35597/17743-level" doubled figures being coarse rounding of the
same quantity -- not used directly in the `num_bins_x/y` derivation above,
which only needs the exact C/R integers.

## Other fields unchanged from `mempool_group.json`

`stop_overflow=0.07`, `random_seed=1000`, `deterministic_flag=1`,
`legalize_flag=1`, `gpu=1`, `enable_fillers=1` -- all per the task
instruction, matching the source recipe. `global_place_stages[0].iteration`
kept at `1000` (group's value) pending the scheduler's overflow-probe-driven
update across the whole ladder; not decided here.

## Not done in this pass

No `PlaceDB.read()` validation was run against this config (host RAM is
committed to a concurrent cluster probe). The validation script is
`ioplace/diagnostics/probes_m4/probe_synth_config_read.py`, meant to be run
later:

```
PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_synth_config_read \
    benchmarks/ispd25/synthetic_1x2_n2.json 1x2_n2
```
