"""M3 T0-b (design draft `2026-08-13-m3-differentiable-ft-design-draft.md`
sec 8): schema validation for the 8 reissued-hermetic probes plus T11's
`probe_maze_sample.json` -- all share the unified `env` provenance schema
(`command`, `python_executable`, `python_version`, `hostname`,
`utc_timestamp`, `repo_commit`, `dp_commit`, `input_sha256`, `exactness`).

`probe_p0b.json`/`probe_p0b.checkpoint.jsonl` (T0-c) are a separate task with
an already-established, deliberately different schema (no `exactness` key --
see `probe_p0b.py`'s own `_env_metadata`) -- out of scope here, not an
oversight (see `test_probe_p0b_json_is_out_of_scope` below).
"""
import datetime
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES_DIR = os.path.join(REPO, "results/m3/probes")

# T0-b's 8 reissued probes + T11's new probe.
SEALED_PROBES = [
    "probe_m3_rg.json",
    "probe_m3_bb.json",
    "probe_m3_surrogate.json",
    "probe_m3_util.json",
    "probe_beta_tau.json",
    "probe_free_area_util.json",
    "probe_l_convention.json",
    "probe_ft_surrogate_soft.json",
    "probe_maze_sample.json",
]

REQUIRED_ENV_KEYS = (
    "command", "python_executable", "python_version", "hostname",
    "utc_timestamp", "repo_commit", "dp_commit", "input_sha256", "exactness",
)


def _load(name):
    path = os.path.join(PROBES_DIR, name)
    assert os.path.exists(path), f"missing probe output: {path}"
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("name", SEALED_PROBES)
def test_probe_json_is_valid_and_has_env_block(name):
    data = _load(name)
    assert isinstance(data, dict), f"{name}: top level must be an object (found {type(data)})"
    assert "env" in data, f"{name}: no top-level 'env' block"
    assert isinstance(data["env"], dict), f"{name}: 'env' must be an object"


@pytest.mark.parametrize("name", SEALED_PROBES)
def test_probe_json_env_has_all_unified_schema_keys(name):
    env = _load(name)["env"]
    missing = [k for k in REQUIRED_ENV_KEYS if k not in env]
    assert not missing, f"{name}: env missing required keys {missing}"


@pytest.mark.parametrize("name", SEALED_PROBES)
def test_probe_json_env_values_are_populated(name):
    env = _load(name)["env"]
    for k in REQUIRED_ENV_KEYS:
        v = env[k]
        assert v is not None, f"{name}: env[{k!r}] is None"
        if isinstance(v, str):
            assert v.strip() != "", f"{name}: env[{k!r}] is an empty/blank string"
    assert isinstance(env["input_sha256"], dict) and len(env["input_sha256"]) > 0, \
        f"{name}: env['input_sha256'] must be a non-empty object"
    # repo_commit/dp_commit: full 40-char git SHA
    assert len(env["repo_commit"]) == 40, f"{name}: repo_commit doesn't look like a git SHA"
    assert len(env["dp_commit"]) == 40, f"{name}: dp_commit doesn't look like a git SHA"
    # utc_timestamp: parseable ISO-8601
    datetime.datetime.fromisoformat(env["utc_timestamp"])


@pytest.mark.parametrize("name", SEALED_PROBES)
def test_probe_json_input_paths_are_repo_relative(name):
    """design draft sec 8 T0-b / F8: 'v2 宣稱已 hermetic 是錯的(路徑硬編碼...)' --
    every hashed input path must be repo-relative, never an absolute /tmp or
    worktree path."""
    env = _load(name)["env"]
    for p in env["input_sha256"]:
        assert not os.path.isabs(p), f"{name}: input_sha256 key {p!r} is an absolute path"
        assert "/tmp" not in p, f"{name}: input_sha256 key {p!r} touches /tmp"
        assert "worktrees" not in p, f"{name}: input_sha256 key {p!r} touches a worktree path"


@pytest.mark.parametrize("name", SEALED_PROBES)
def test_probe_json_command_has_no_hardcoded_absolute_repo_path(name):
    """The `command`/`argv` fields must not embed this machine's absolute repo
    path (that would defeat repo-relative hermeticity for anyone diffing
    provenance across machines)."""
    env = _load(name)["env"]
    assert REPO not in env["command"], f"{name}: command embeds an absolute repo path"


def test_probe_p0b_json_is_out_of_scope():
    """Sanity check that probe_p0b.json (T0-c, a separate task) intentionally
    uses a different schema (no 'exactness' key) -- confirms its omission
    from SEALED_PROBES above is deliberate, not an oversight."""
    path = os.path.join(PROBES_DIR, "probe_p0b.json")
    if not os.path.exists(path):
        pytest.skip("probe_p0b.json not produced yet")
    with open(path) as f:
        data = json.load(f)
    assert "env" in data
    for k in ("command", "python_executable", "python_version", "hostname",
             "utc_timestamp", "repo_commit", "dp_commit", "input_sha256"):
        assert k in data["env"], f"probe_p0b.json env unexpectedly missing {k!r}"
