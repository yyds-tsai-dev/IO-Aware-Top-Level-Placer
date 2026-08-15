"""M4 T8 synthetic-config read validation probe (NOT run yet -- host RAM is
committed to a concurrent cluster probe as of the config-authoring pass that
wrote this script; run later once that probe finishes).

Purpose: verify `benchmarks/ispd25/synthetic_1x2_n2.json` /
`synthetic_2x2_n2.json` (the T8 synthetic-arm DREAMPlace configs for the
tiler's 1x2/2x2 n2 arrays) actually read through DREAMPlace's
`Params.load()` + `PlaceDB.read()` with the node/net/pin counts the tiler's
own manifest promised, by comparing against `<case>.manifest.json` sitting
next to the `.aux` file (`results/m4/bench/arrays/<case>/<case>.manifest.
json`). Per that manifest's schema (see `synthetic_1x2_n2.md`'s note on the
`.nodes` file's `NumNodes`/`NumTerminals` matching `manifest["base"]`
exactly): glue only adds nets/pins between *existing* pins, it adds no new
nodes -- so the expected counts are `n_nodes`/`n_terminals` = `base` as-is,
`n_nets`/`n_pins` = `base` + `glue`.

Style follows `probe_corpus_stats.py` (same `PlaceDB.read()`+`initialize()`
pattern, same chdir bracket rationale: `aux_input` in these configs is
already absolute -- see `synthetic_1x2_n2.md` -- so the chdir is a no-op for
path resolution here, but kept for parity with the sibling probes and in
case a future config in this family uses relative paths).

Usage:
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_synth_config_read \\
        <config.json> <case_name>

Example (not run by this pass):
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m4.probe_synth_config_read \\
        benchmarks/ispd25/synthetic_1x2_n2.json 1x2_n2
"""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

from ioplace.dreamplace_env import setup_dreamplace

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _manifest_path(aux_input):
    """`<dir>/<case>.aux` -> `<dir>/<case>.manifest.json` (the tiler's own
    naming, see `results/m4/bench/arrays/1x2_n2/1x2_n2.manifest.json`)."""
    base, _ = os.path.splitext(aux_input)
    return base + ".manifest.json"


def expected_counts(manifest):
    """`base` nets/pins + `glue` nets/pins; nodes/terminals get no glue
    additions (glue only adds nets between existing pins -- see module
    docstring)."""
    base = manifest["base"]
    glue = manifest.get("glue", {})
    return dict(
        n_nodes=base["n_nodes"],
        n_terminals=base["n_terminals"],
        n_nets=base["n_nets"] + glue.get("n_nets", 0),
        n_pins=base["n_pins"] + glue.get("n_pins", 0),
    )


def run(config_json, case) -> dict:
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    # Same chdir bracket as run_placement.py's `_load_dreamplace` / probe_corpus_stats.py's
    # `run()`: harmless here since aux_input is absolute (see synthetic_1x2_n2.md), kept for
    # parity in case a relative-path config in this family is added later.
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))
    try:
        params.load(config_json)
        aux_input = params.aux_input

        t0 = time.time()
        db = PlaceDB.PlaceDB()
        db.read(params)
        t_read = time.time() - t0
        rss_after_read_gb = _rss_gb()

        t1 = time.time()
        db.initialize(params)
        t_init = time.time() - t1
        rss_after_init_gb = _rss_gb()
    finally:
        os.chdir(cwd)

    manifest_path = _manifest_path(aux_input)
    with open(manifest_path) as f:
        manifest = json.load(f)
    expected = expected_counts(manifest)

    actual = dict(
        n_nodes=int(db.num_physical_nodes),
        n_terminals=int(db.num_terminals),
        n_nets=int(db.num_nets),
        n_pins=int(len(db.pin2node_map)),
    )
    diff = {k: actual[k] - expected[k] for k in expected}
    match = all(v == 0 for v in diff.values())

    return dict(
        case=case,
        config=config_json,
        manifest=manifest_path,
        read_s=t_read, init_s=t_init,
        peak_rss_gb=rss_after_init_gb,
        rss_after_read_gb=rss_after_read_gb, rss_after_init_gb=rss_after_init_gb,
        expected=expected, actual=actual, diff=diff, match=match,
        num_movable=int(db.num_movable_nodes),
        num_filler=int(getattr(db, "num_filler_nodes", 0)),
        sha256={
            os.path.abspath(config_json): _sha256(config_json),
            os.path.abspath(aux_input): _sha256(aux_input),
            os.path.abspath(manifest_path): _sha256(manifest_path),
        },
        env=dict(dp_commit=_git_head(DP), ioplace_commit=_git_head(REPO)),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <dreamplace_config.json> <case_name>")
    cfg = os.path.abspath(sys.argv[1])
    case = sys.argv[2]
    result = run(cfg, case)
    out_path = os.path.join(REPO, "results", "m4", "corpus", f"{case}_read_check.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    verdict = "MATCH" if result["match"] else "MISMATCH"
    print(f"[probe_synth_config_read] wrote {out_path}: {verdict} "
          f"expected={result['expected']} actual={result['actual']} "
          f"read_s={result['read_s']:.1f} peak_rss_gb={result['peak_rss_gb']:.2f}")
