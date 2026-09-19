"""M4 T0b factorial-probe config generator, net-drop variant (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.1 T0b,
design point 2, Codex D4): 2 DREAMPlace configs pointing at the net-drop
Bookshelf designs produced by `ioplace/bench/net_drop.py`
(`results/m4/bench/net_drop/group_drop{25,50}/mempool_group.aux`) -- these
hold `N_total`/`n_bins` fixed while moving `N_pins` independently of them
(dropping nets keeps every node), the design point design point 1's
bins/target_density sweep cannot produce on its own since it never touches
the netlist.

**Bookshelf-origin field set (repo convention, no invented fields):**
`aux_input` (this is what distinguishes a Bookshelf design from a LEF/DEF
one -- no `lef_input`/`def_input`) plus `scale_factor`/`gift_init_flag`,
both of which the two existing Bookshelf-origin configs in this repo
(`benchmarks/ispd2005_bigblue4_m4.json`, `benchmarks/ispd25/
synthetic_1x2_n2.json`) carry and the LEF/DEF `mempool_group.json` omits;
`sol_file_format` (DEF-only, `mempool_group.json`'s own field) is likewise
omitted here, matching both Bookshelf references. Every other field is
copied verbatim from `mempool_group.json`: `num_bins_x`/`num_bins_y` fixed
at the base's own 2048 (both top-level and `global_place_stages[0]` --
same two-place rule as `gen_t0b_configs.py`) since this design point's
whole point is to hold bins/N_total fixed while pins move, and
`target_density` left at the base's actual 0.714 (not the 9-config sweep's
0.835 -- this is a *different* design point, not a member of that sweep).
`legalize_flag: 0` for the same "9 runs, GP-only, det=1" reason as
`gen_t0b_configs.py` (T0b's net-drop variants are GP-only too -- sec 7.1's
"det=1" scope note names the bins/td sweep specifically, but there is no
GP-memory/runtime-modeling reason for the net-drop points to run LG,
and driver's `run_placement.py` only honors `legalize_flag` from the
config, never a CLI override -- see `gen_t0b_configs.py`'s docstring).

Deterministic and idempotent, same shape as `gen_t0b_configs.py`.
"""
import copy
import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.path.basename(_REPO_ROOT) == "src":
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
_BASE_CONFIG = os.path.join(_REPO_ROOT, "benchmarks", "ispd25", "mempool_group.json")
_OUT_DIR = os.path.join(_REPO_ROOT, "benchmarks", "ispd25", "t0b")
_NET_DROP_DIR = os.path.join(_REPO_ROOT, "results", "m4", "bench", "net_drop")

# (label, net_drop subdirectory -> "<subdir>/mempool_group.aux")
NET_DROP_VARIANTS = (
    ("netdrop25", "group_drop25"),
    ("netdrop50", "group_drop50"),
)

# Bookshelf-origin extras present in both `benchmarks/ispd2005_bigblue4_m4.json`
# and `benchmarks/ispd25/synthetic_1x2_n2.json` but absent from the LEF/DEF
# `mempool_group.json` base -- added, not overridden, so `build_config`
# below can otherwise reuse `gen_t0b_configs.build_config`'s shape.
_BOOKSHELF_EXTRAS = {"scale_factor": 1.0, "gift_init_flag": 0}


def build_config(base_config, aux_input):
    """Returns a deep copy of `base_config` rewired to a Bookshelf origin:
    `lef_input`/`def_input`/`sol_file_format` dropped, `aux_input` set,
    `scale_factor`/`gift_init_flag` added, `legalize_flag` forced 0
    (GP-only). `num_bins_x`/`num_bins_y`/`target_density` are left at the
    base's own values (2048/2048/0.714) -- this design point varies
    `N_pins` via the source `.aux`, not bins/density."""
    cfg = copy.deepcopy(base_config)
    for key in ("lef_input", "def_input", "sol_file_format"):
        cfg.pop(key, None)
    cfg["aux_input"] = aux_input
    cfg.update(_BOOKSHELF_EXTRAS)
    cfg["legalize_flag"] = 0
    return cfg


def generate(base_config_path=_BASE_CONFIG, net_drop_dir=_NET_DROP_DIR, out_dir=_OUT_DIR):
    """Writes both net-drop configs to `out_dir`. Returns the list of
    output paths written, in generation order."""
    with open(base_config_path) as f:
        base_config = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for label, subdir in NET_DROP_VARIANTS:
        aux_input = os.path.join(net_drop_dir, subdir, "mempool_group.aux")
        cfg = build_config(base_config, aux_input)
        out_path = os.path.join(out_dir, f"group__{label}.json")
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
