"""M4 T8 synthetic-arm config schema tests (design draft `docs/superpowers/
specs/2026-08-13-m4-scale-up-design-draft.md` sec 2.1/7.1 T8 row): pure
JSON/field-existence checks for `benchmarks/ispd25/synthetic_1x2_n2.json` /
`synthetic_2x2_n2.json`, the DREAMPlace configs for the tiler's 1x2/2x2 n2
arrays (N1 vetoed by the 2026-08-15 T6 holdout adjudication -- only n2
configs exist).

Deliberately does NOT run `Params.load()`/`PlaceDB.read()` -- see
`ioplace/diagnostics/probes_m4/probe_synth_config_read.py` for that (not run
yet, host RAM committed to a concurrent cluster probe as of this pass). This
module only needs the stdlib, no DREAMPlace venv."""
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO, "benchmarks", "ispd25")
ARRAYS_DIR = os.path.join(REPO, "results", "m4", "bench", "arrays")

# case -> (config filename, expected num_bins_x, expected num_bins_y)
# Bins: mempool_group.json uses 2048x2048; each axis scales by the tiler's own
# C (columns)/R (rows) replication factor for that array (1x2_n2.manifest.json:
# C=2,R=1; 2x2_n2.manifest.json: C=2,R=2) -- see synthetic_1x2_n2.md/synthetic_2x2_n2.md.
CASES = {
    "1x2_n2": ("synthetic_1x2_n2.json", 4096, 2048),
    "2x2_n2": ("synthetic_2x2_n2.json", 4096, 4096),
}

# Fields copied verbatim from benchmarks/ispd25/mempool_group.json (the tiling
# source's own GP recipe) -- see the per-config .md notes for why.
EXPECTED_GROUP_RECIPE = dict(
    gpu=1,
    target_density=0.714,
    density_weight=8e-05,
    gamma=4.0,
    random_seed=1000,
    deterministic_flag=1,
    ignore_net_degree=100,
    enable_fillers=1,
    gp_noise_ratio=0.025,
    global_place_flag=1,
    legalize_flag=1,
    detailed_place_flag=0,
    stop_overflow=0.07,
    dtype="float32",
    plot_flag=0,
    random_center_init_flag=1,
    sort_nets_by_degree=0,
    num_threads=16,
)

REQUIRED_KEYS = set(EXPECTED_GROUP_RECIPE) | {
    "aux_input", "num_bins_x", "num_bins_y", "global_place_stages",
    "scale_factor", "gift_init_flag",
}


def _load(case):
    fname, _, _ = CASES[case]
    path = os.path.join(BENCH_DIR, fname)
    assert os.path.exists(path), f"missing config: {path}"
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("case", CASES)
def test_config_is_valid_json_object(case):
    data = _load(case)
    assert isinstance(data, dict)


@pytest.mark.parametrize("case", CASES)
def test_config_has_all_required_keys(case):
    data = _load(case)
    missing = REQUIRED_KEYS - set(data)
    assert not missing, f"{case}: config missing keys {missing}"


@pytest.mark.parametrize("case", CASES)
def test_config_has_no_lef_def_fields(case):
    """These are Bookshelf-input configs (aux_input) -- lef_input/def_input
    are the LEF/DEF-style fields used by mempool_group.json and must not be
    present here (would silently be ignored by DREAMPlace but signals a
    copy-paste mistake)."""
    data = _load(case)
    assert "lef_input" not in data
    assert "def_input" not in data


@pytest.mark.parametrize("case", CASES)
def test_config_group_recipe_fields_match_source(case):
    data = _load(case)
    for key, expected in EXPECTED_GROUP_RECIPE.items():
        assert data[key] == expected, (
            f"{case}: {key}={data[key]!r}, expected {expected!r} "
            f"(copied from mempool_group.json)"
        )


@pytest.mark.parametrize("case", CASES)
def test_config_bins_match_tiler_replication_factor(case):
    _, expected_x, expected_y = CASES[case]
    data = _load(case)
    assert data["num_bins_x"] == expected_x
    assert data["num_bins_y"] == expected_y
    stages = data["global_place_stages"]
    assert isinstance(stages, list) and len(stages) == 1
    stage = stages[0]
    assert stage["num_bins_x"] == expected_x
    assert stage["num_bins_y"] == expected_y
    assert stage["iteration"] == 1000


@pytest.mark.parametrize("case", CASES)
def test_config_bins_match_manifest_c_r_replication(case):
    """Cross-check the hardcoded expected bins in CASES against the tiler's
    own manifest C (columns)/R (rows) fields, so this test fails if a future
    re-tile changes the array shape without the config being updated."""
    data = _load(case)
    manifest_path = os.path.join(ARRAYS_DIR, case, f"{case}.manifest.json")
    if not os.path.exists(manifest_path):
        pytest.skip(f"manifest not present: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert data["num_bins_x"] == manifest["C"] * 2048
    assert data["num_bins_y"] == manifest["R"] * 2048


@pytest.mark.parametrize("case", CASES)
def test_aux_input_is_absolute_and_exists(case):
    data = _load(case)
    aux_input = data["aux_input"]
    assert os.path.isabs(aux_input), f"{case}: aux_input must be absolute, got {aux_input!r}"
    assert os.path.exists(aux_input), f"{case}: aux_input does not exist: {aux_input}"


@pytest.mark.parametrize("case", CASES)
def test_aux_input_points_at_expected_case_file(case):
    data = _load(case)
    expected_path = os.path.join(ARRAYS_DIR, case, f"{case}.aux")
    assert os.path.abspath(data["aux_input"]) == os.path.abspath(expected_path)


@pytest.mark.parametrize("case", CASES)
def test_aux_referenced_sibling_files_exist(case):
    """Static filesystem check only (no Bookshelf parse): every file the
    .aux line lists must exist next to it."""
    data = _load(case)
    aux_input = data["aux_input"]
    aux_dir = os.path.dirname(aux_input)
    with open(aux_input) as f:
        aux_line = f.read()
    # Bookshelf .aux format: "RowBasedPlacement : <nodes> <nets> <wts> <pl> <scl>"
    _, _, rhs = aux_line.partition(":")
    referenced = rhs.split()
    assert referenced, f"{case}: could not parse referenced files out of {aux_input}"
    for fname in referenced:
        fpath = os.path.join(aux_dir, fname)
        assert os.path.exists(fpath), f"{case}: .aux references missing sibling file {fpath}"


@pytest.mark.parametrize("case", CASES)
def test_derivation_note_exists(case):
    fname, _, _ = CASES[case]
    note_path = os.path.join(BENCH_DIR, os.path.splitext(fname)[0] + ".md")
    assert os.path.exists(note_path), f"{case}: missing derivation note {note_path}"
