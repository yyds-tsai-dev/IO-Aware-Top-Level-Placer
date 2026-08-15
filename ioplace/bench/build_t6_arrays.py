"""M4 T6 array builder (adjudication doc `docs/results/2026-08-14-m4-t6-
adjudication.md` sec 4 "T6 本體"): builds the four Bookshelf arrays T6's
holdout gates need -- {1x2, 2x2} x {N1, N2} -- from the `mempool_group`
Bookshelf export (`export_bookshelf.py`) and the calibrated statistics in
`results/m4/bench/cluster_stats.json`.

**T6B (2026-08-15 holdout adjudication `docs/results/2026-08-15-m4-t6-
holdout-adjudication.md` sec 4/8-3):** this module also builds the 3x3
array T6B's B-0..B-6 checks run against (`SHAPES["3x3"]`, N2 only -- N1 is
`not_run` there per sec 3), and, via `--no-glue`/`build_shape_no_glue`, the
zero-glue control arrays B-1's Rent-invariance DiD needs.

Parameters (adjudication sec 4): `alpha=0.1023` (dimensionless, unaffected
by tile-size rescaling); generation uses `lambda_0_tile` (not the raw
`lambda_0=4155.2`) as the kernel's `lambda_0` -- `cluster_stats.json`'s
`lambda_0_tile` is exactly the Rent-consistent rescaling of the real
(smaller) cluster-group calibration to the actual (larger) `mempool_group`
Bookshelf tile size this builder replicates (sec 2d's `lambda_0 x
(3.0780/2.8246)^0.633 = 4387`; `lambda_0` itself, uncorrected for this
scale mismatch, would systematically undercount glue nets for a tile this
size).

N1 vs N2 (sec 2d): N1 is `glue_gen.sample_glue_net_count` unmodified (per-
pair `lambda_0*phi(d,alpha)`, independent of array size). N2 is
`glue_gen.n2_pair_counts_sinkhorn` (fixed per-tile terminal budget
`B=3*lambda_0`, redistributed by `phi(d)` via symmetric Sinkhorn scaling --
2026-08-15 sec 5.2/8-2: generalizes `n2_pair_counts`, which only supported
`1xC`/`Rx1`/`2x2`, to arbitrary shapes including 3x3's non-uniform corner/
edge/center neighbor multisets; bit-for-bit identical to the old closed
form on every shape this builder already used it on).

**Known simplification (documented, not hidden):** every generated glue net
is a degree-2 bipartite tile-pair net (`glue_gen.sample_glue_nets`'s only
implemented mode); the real cluster's cross-group nets are 96.2% degree-2
but 3.8% touch 3-4 groups (`cluster_stats.json`'s `ngroups_hist`) -- this
builder does not reproduce that tail. Candidate interface-cell endpoints are
sampled *uniformly* from each tile's own movable-cell names (not biased
toward the tile edge facing the neighbor, and not drawn from a real
iface_dist calibration -- no per-cell "distance to real cross-group
boundary" data was extracted from the cluster for this task). Both
simplifications mean the *fit* metrics (glue degree histogram, pinshare,
iface_dist -- adjudication sec 4 "只報,不判定") will not match the real
distributions by construction here, unlike the ideal design draft sec 3.2
mechanism; this is disclosed in each build's manifest and in `t6_summary.
json`, not silently glossed over. It does not touch any *holdout* gate
(H1/H2/H3/H6) definition, which is about the array's bulk net-degree/cut/
Rent structure, dominated by the (exactly-replicated) base netlist -- only
H4 (interface-cell distance) is directly about this choice; see
`t6_summary.json`'s H4 entry for how that gate is scoped down accordingly.

**Efficiency note:** `tile_bookshelf.tile`'s output does not depend on
`seed` (a pure deterministic replication -- `seed` is recorded in its
manifest but never consumed by the streaming copy itself), so this builder
tiles each shape exactly *once* and reuses that base for both N1 and N2
(each gets its own on-disk copy before glue nets are appended in place,
since `append_glue_nets` rewrites `.nets`). The "≥5 seeds reproduction
trajectory" requirement is satisfied without 5x re-tiling a multi-GB base:
only the *glue* step is seed-dependent, so extra seeds re-run just the
(cheap) Poisson pair-count draw + candidate-node sampling and record a
sha256 of that deterministic recipe (JSON-serialized sampled pair counts +
net endpoint list) -- proving the generation is seed-reproducible -- rather
than materializing 5 full copies of the array to hash. Only the primary
seed's array is written to disk, per this task's instruction.

Usage:
    PYTHONPATH=. $PY -m ioplace.bench.build_t6_arrays \\
        --source results/m4/bench/mempool_group_export/mempool_group \\
        --cluster-stats results/m4/bench/cluster_stats.json \\
        --out-dir results/m4/bench/arrays --seed 0
"""
import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np

from ioplace.bench import glue_gen, tile_bookshelf as tb

SHAPES = {"1x2": (1, 2), "2x2": (2, 2), "3x3": (3, 3)}
NORMALIZATIONS = ("n1", "n2")


def _movable_node_names(nodes_path):
    """Names of every non-terminal node in a Bookshelf `.nodes` file (the
    candidate pool for glue-net interface-cell endpoints)."""
    names = []
    with open(nodes_path) as f:
        started = False
        for line in f:
            s = line.strip()
            if not started:
                if s.startswith("NumTerminals"):
                    started = True
                continue
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 4 and parts[3] in ("terminal", "terminal_NI"):
                continue
            names.append(parts[0])
    return names


def _expected_pair_counts(norm, lambda_0_tile, alpha, R, C):
    """Returns (expected_pair_counts, sinkhorn_diagnostics_or_None).

    **2026-08-15 T6 holdout adjudication sec 5.2/8-3:** the n2 branch now
    calls `glue_gen.n2_pair_counts_sinkhorn` (the general-shape solver)
    instead of `glue_gen.n2_pair_counts` (which raises `NotImplementedError`
    on 3x3's non-uniform corner/edge/center neighbor multisets). On the
    1x2/2x2 shapes this builder already produced, the two are bit-for-bit
    equivalent (`test_bench_glue_gen.py`'s
    `test_n2_sinkhorn_matches_closed_form_bit_for_bit_on_uniform_shapes`) --
    those manifests' expected-pair-count values are unchanged."""
    if norm == "n1":
        dists = glue_gen.tile_pair_distances(R, C)
        return {pair: lambda_0_tile * glue_gen.phi(d, alpha) for pair, d in dists.items()}, None
    if norm == "n2":
        return glue_gen.n2_pair_counts_sinkhorn(lambda_0_tile, alpha, R, C, budget_pairs=3.0)
    raise ValueError(f"unknown normalization {norm!r}")


def _copy_base(base_prefix, dst_prefix):
    """Copies the base tiling's 5 core Bookshelf files to `dst_prefix`.

    **Bugfix (2026-08-15, caught by `rent.load_bookshelf_netlist` failing
    on the very first built array):** `.aux` must *not* be copied verbatim
    -- its `RowBasedPlacement : <design>.nodes <design>.nets ...` line
    names the *base* prefix's own basename (e.g. "1x2"), which is wrong
    once copied under a different basename ("1x2_n1"); every downstream
    reader that resolves file paths via `.aux` (`tile_bookshelf.read_aux`,
    used by `rent.load_bookshelf_netlist`, `verify_bench.check_v0_
    structural`, ...) would silently look for the base's files at the new
    location and fail with `FileNotFoundError`. Regenerated instead, same
    format as `tile_bookshelf.tile`'s own `.aux` writer."""
    os.makedirs(os.path.dirname(dst_prefix), exist_ok=True)
    for suf in tb.CORE_SUFFIXES:
        shutil.copyfile(f"{base_prefix}.{suf}", f"{dst_prefix}.{suf}")
    base = os.path.basename(dst_prefix)
    with open(dst_prefix + ".aux", "w") as f:
        f.write(f"RowBasedPlacement : {' '.join(base + '.' + s for s in tb.CORE_SUFFIXES)}\n")
    with open(f"{base_prefix}.manifest.json") as f:
        manifest = json.load(f)
    with open(f"{dst_prefix}.manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)


def _sample_glue_recipe(seed, expected, candidate_names):
    """The seed-dependent part only: Poisson pair counts + concrete glue
    net endpoint list. Returns (counts_per_pair, glue_nets)."""
    rng = np.random.default_rng(seed)
    counts_per_pair = glue_gen.sample_glue_net_count_from_expected(rng, expected, mode="poisson")
    tiles = sorted({t for pair in expected for t in pair})
    candidate_nodes = {tile: candidate_names for tile in tiles}
    glue_nets = glue_gen.sample_glue_nets(rng, counts_per_pair, candidate_nodes, deg_hist=(2,))
    return counts_per_pair, glue_nets


def _recipe_sha256(counts_per_pair, glue_nets):
    payload = json.dumps({
        "counts_per_pair": sorted((f"{a}|{b}", v) for (a, b), v in counts_per_pair.items()),
        "glue_nets": [net["pins"] for net in glue_nets],
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_shape(source_prefix, lambda_0_tile, alpha, shape_name, R, C, out_dir, seed,
                 extra_seeds, candidate_names):
    t0 = time.time()
    base_prefix = os.path.join(out_dir, "_base", shape_name, shape_name)
    os.makedirs(os.path.dirname(base_prefix), exist_ok=True)
    tb.tile(source_prefix, base_prefix, R, C, seed)
    t_tile = time.time() - t0

    result = {}
    for norm in NORMALIZATIONS:
        expected, sinkhorn_diag = _expected_pair_counts(norm, lambda_0_tile, alpha, R, C)
        dst_prefix = os.path.join(out_dir, f"{shape_name}_{norm}", f"{shape_name}_{norm}")
        _copy_base(base_prefix, dst_prefix)

        counts_per_pair, glue_nets = _sample_glue_recipe(seed, expected, candidate_names)
        t1 = time.time()
        n_glue_nets, n_glue_pins = glue_gen.append_glue_nets(
            dst_prefix, glue_nets, lambda_0_tile, alpha, kernel="power_law", seed=seed)
        t_glue = time.time() - t1

        seed_trajectory = {seed: _recipe_sha256(counts_per_pair, glue_nets)}
        for extra_seed in extra_seeds:
            cpp, gn = _sample_glue_recipe(extra_seed, expected, candidate_names)
            seed_trajectory[extra_seed] = _recipe_sha256(cpp, gn)

        with open(dst_prefix + ".manifest.json") as f:
            manifest = json.load(f)
        manifest["t6"] = {
            "shape": shape_name, "normalization": norm, "primary_seed": seed,
            "lambda_0_tile": lambda_0_tile, "alpha": alpha,
            "expected_pair_counts": {f"{a}|{b}": v for (a, b), v in expected.items()},
            "sampled_pair_counts": {f"{a}|{b}": v for (a, b), v in counts_per_pair.items()},
            "n_glue_nets": n_glue_nets, "n_glue_pins": n_glue_pins,
            "t_tile_s": t_tile, "t_glue_s": t_glue,
            "seed_reproduction_trajectory_sha256": seed_trajectory,
            "seed_trajectory_note": "sha256 of the deterministic (sampled pair counts + glue "
                                     "net endpoint list) recipe per seed, not of a re-"
                                     "materialized Bookshelf file -- see module docstring's "
                                     "efficiency note; only primary_seed's array is on disk",
            "known_simplification": "degree-2-only glue nets; uniform (not boundary-biased or "
                                     "iface_dist-calibrated) candidate interface-cell sampling "
                                     "-- see module docstring",
        }
        if sinkhorn_diag is not None:
            manifest["t6"]["n2_rule"] = "sinkhorn"
            manifest["t6"]["n2_iters"] = sinkhorn_diag["n2_iters"]
            manifest["t6"]["n2_max_rel_dev"] = sinkhorn_diag["n2_max_rel_dev"]
        output_sha256 = {suf: tb.sha256_file(f"{dst_prefix}.{suf}") for suf in tb.CORE_SUFFIXES}
        manifest["output_sha256"] = output_sha256
        with open(dst_prefix + ".manifest.json", "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        result[norm] = manifest

    return result


def build_shape_no_glue(source_prefix, shape_name, R, C, out_dir, seed):
    """B-1's zero-glue control array (2026-08-15 T6 holdout adjudication
    sec 4.1/4.3/8-3): the base tiling only, glue step skipped entirely.
    Isolates a combinatorial artifact of an odd shape like 3x3 (9 tiles
    can't be halved into an integer tile count, so recursive bisection
    systematically inflates low-level terminal counts near the fit window
    independent of glue) from the *glue* contribution to Rent's p, via a
    difference-in-differences against the glued array
    (`verify_bench.check_b1_rent_invariance_did`). One array per shape
    (not per normalization -- there is no glue to normalize)."""
    t0 = time.time()
    dst_prefix = os.path.join(out_dir, f"{shape_name}_noglue", f"{shape_name}_noglue")
    manifest = tb.tile(source_prefix, dst_prefix, R, C, seed)
    manifest["glue"] = None
    manifest["no_glue"] = True
    manifest["t6"] = {"shape": shape_name, "normalization": None, "primary_seed": seed,
                       "no_glue": True, "t_tile_s": time.time() - t0}
    with open(dst_prefix + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--cluster-stats", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="primary seed; its full output is kept")
    ap.add_argument("--extra-seeds", type=int, nargs="*", default=[1, 2, 3, 4],
                     help="additional seeds for the reproduction trajectory (recipe sha256 "
                          "recorded; see module docstring's efficiency note)")
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES))
    ap.add_argument("--no-glue", action="store_true",
                     help="build the zero-glue control array(s) instead of the normal N1/N2 "
                          "arrays (adjudication sec 4.1/4.3/8-3's B-1 DiD baseline): base tiling "
                          "only, glue step skipped, manifest records glue=null/no_glue=true. "
                          "Mutually exclusive with the normal build -- run this as a separate "
                          "invocation, one array per shape (no N1/N2 split).")
    args = ap.parse_args(argv)

    if args.no_glue:
        results = {}
        for shape_name in args.shapes:
            R, C = SHAPES[shape_name]
            results[shape_name] = build_shape_no_glue(args.source, shape_name, R, C,
                                                        args.out_dir, args.seed)
            print(f"[build_t6_arrays] {shape_name}-noglue done", flush=True)
        out_path = os.path.join(args.out_dir, "build_summary_noglue.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
        print(f"[build_t6_arrays] wrote {out_path}")
        return results

    with open(args.cluster_stats) as f:
        cs = json.load(f)
    lambda_0_tile = cs["lambda_0_tile"]["value"]
    alpha = cs["mom"]["alpha"]

    candidate_names = _movable_node_names(args.source + ".nodes")

    results = {}
    for shape_name in args.shapes:
        R, C = SHAPES[shape_name]
        results[shape_name] = build_shape(args.source, lambda_0_tile, alpha, shape_name, R, C,
                                           args.out_dir, args.seed, args.extra_seeds,
                                           candidate_names)
        print(f"[build_t6_arrays] {shape_name} done", flush=True)

    out_path = os.path.join(args.out_dir, "build_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f"[build_t6_arrays] wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
