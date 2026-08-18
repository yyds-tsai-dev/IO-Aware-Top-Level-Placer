# `synthetic_2x2_n2.json` -- derivation notes

M4 T8 synthetic-arm config for the tiler's 2x2 array (`mempool_group` x4,
Bookshelf, n2 glue normalization -- N1 vetoed by the 2026-08-15 T6 holdout
adjudication, see `docs/results/2026-08-15-m4-t6-holdout-adjudication.md`
sec 3; this is the "12.3M" ladder rung and the synthetic counterpart to the
real `mempool_cluster` holdout target,
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 2.1).
Bookshelf input: `results/m4/bench/arrays/2x2_n2/2x2_n2.aux` (12.36M nodes /
12.31M movable cells).

Same derivation as `synthetic_1x2_n2.md` (read that file for the full
reasoning on format template, `aux_input` absolute-path safety,
`target_density`, and field provenance); this file only records what
differs for the 2x2 case.

## Bin sizing: `num_bins_x=4096, num_bins_y=4096`

`2x2_n2.manifest.json` records `"C": 2, "R": 2` -- 2 tile-columns, 2
tile-rows, i.e. both axes scale by exactly 2x relative to a single group
tile (unlike the 1x2 array, which only scales x). Preserving the source's
physical bin size on both axes therefore scales `num_bins` by the same
factor on both axes: `num_bins_x = num_bins_y = 2 * 2048 = 4096`. This is
square, so it does not exercise the non-square `num_bins_x != num_bins_y`
path that the 1x2 config does, but it's derived by the identical rule (`2 *
group's num_bins` per axis, per that axis's tile-replication count) --
not chosen as a fallback because non-square bins are unsupported. DREAMPlace
does support non-square `num_bins_x`/`num_bins_y` (`$DP/dreamplace/params.
json` lines 26-33); it just happens that 2x2's C=R=2 makes the square case
the correct answer here too.

## Other fields

Identical convention to `synthetic_1x2_n2.json`/`.md`: `target_density=
0.714`, `stop_overflow=0.07`, `random_seed=1000`, `deterministic_flag=1`,
`legalize_flag=1`, `detailed_place_flag=0`, `gpu=1`, `enable_fillers=1`,
`num_threads=16`, `scale_factor=1.0`, all copied from `mempool_group.json`
(or, for the Bookshelf-only structural fields, from bigblue4.json's
convention). `global_place_stages[0].iteration` kept at `1000` pending the
scheduler's overflow-probe-driven update.

## Not done in this pass

No `PlaceDB.read()` validation was run (host RAM committed to a concurrent
cluster probe). Run later via:

```
PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_synth_config_read \
    benchmarks/ispd25/synthetic_2x2_n2.json 2x2_n2
```
