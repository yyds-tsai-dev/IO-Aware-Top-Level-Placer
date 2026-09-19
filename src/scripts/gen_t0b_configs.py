"""M4 T0b factorial-probe config generator (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.1 T0b,
design point 1): 9 DREAMPlace configs on `mempool_group`, independently
varying `num_bins_x`/`num_bins_y` in {1024, 2048, 4096} and `target_density`
in {0.70, 0.835, 0.90} (a 3x3 factorial), so the GP-memory/runtime model's
`n_bins` and `N_total` (filler count, driven by `target_density`)
coefficients can be identified. Every other field is copied verbatim from
`benchmarks/ispd25/mempool_group.json` (iteration 2000, det=1, etc.) --
bins/density are the only things this sweep is meant to move.

`num_bins_x`/`num_bins_y` are set in *two* places, both required by
DREAMPlace: the top-level fields (read once at `PlaceDB.initialize`) and
`global_place_stages[0]`'s own copy (the actual GP stage's bin resolution).
Leaving either one at the base config's 2048 would silently make bins not
actually vary the run.

`legalize_flag: 0` (T0b design point 1's own requirement: "9 runs, GP-only,
det=1" -- sec 7.1 T0b row). `ioplace/drivers/run_placement.py`'s driver
always forces `detailed_place_flag = 0` in `_load_dreamplace` regardless of
what a config says (DP is already unconditionally off), but it reads
`legalize_flag` straight from the config and honors it as-is -- there is no
CLI override -- so GP-only has to be requested here, not at run time.

Deterministic and idempotent: re-running overwrites the same 9 files with
byte-identical content (no timestamps, no randomness -- `json.dump` on a
plain dict built from a fixed loop order).
"""
import copy
import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.path.basename(_REPO_ROOT) == "src":
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
_BASE_CONFIG = os.path.join(_REPO_ROOT, "benchmarks", "ispd25", "mempool_group.json")
_OUT_DIR = os.path.join(_REPO_ROOT, "benchmarks", "ispd25", "t0b")

# (num_bins, [(target_density value, filename-string label), ...]) -- the
# label preserves each value's own decimal precision (0.70/0.90 vs 0.835)
# rather than a fixed format spec that would either truncate 0.835 or pad
# 0.70/0.90 with a spurious trailing digit.
BINS = (1024, 2048, 4096)
TARGET_DENSITIES = (("0.70", 0.70), ("0.835", 0.835), ("0.90", 0.90))


def build_config(base_config, num_bins, target_density):
    """Returns a deep copy of `base_config` with `num_bins_x`/`num_bins_y`
    (top-level and `global_place_stages[0]`), `target_density`, and
    `legalize_flag` (forced 0, GP-only) overridden; every other field
    untouched."""
    cfg = copy.deepcopy(base_config)
    cfg["num_bins_x"] = num_bins
    cfg["num_bins_y"] = num_bins
    cfg["global_place_stages"][0]["num_bins_x"] = num_bins
    cfg["global_place_stages"][0]["num_bins_y"] = num_bins
    cfg["target_density"] = target_density
    cfg["legalize_flag"] = 0
    return cfg


def generate(base_config_path=_BASE_CONFIG, out_dir=_OUT_DIR):
    """Writes all 9 `bins x target_density` configs to `out_dir`. Returns
    the list of output paths written, in generation order."""
    with open(base_config_path) as f:
        base_config = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for num_bins in BINS:
        for td_label, td_value in TARGET_DENSITIES:
            cfg = build_config(base_config, num_bins, td_value)
            out_path = os.path.join(out_dir, f"group__bins{num_bins}__td{td_label}.json")
            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=4)
                f.write("\n")
            written.append(out_path)
    return written


def main():
    written = generate()
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
