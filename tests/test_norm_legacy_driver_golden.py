"""Driver-level golden for `--norm-policy legacy` (adversarial review I5).

`tests/test_norm_legacy_adapter.py` locks the *adapter* (normalizer vs. direct
`publish_atomic`, bit-for-bit) and `tests/test_norm.py` locks `schedules.py`'s
maths, but nothing in the suite recorded what `run_io(..., norm_policy=
"legacy")` actually produced before the P-H range -- so a numeric drift
introduced by a future edit to `cb()`'s legacy branch (P-B edits exactly that
branch) would go unnoticed.

The fixture `tests/fixtures/legacy_driver_golden_simple.json` was produced by
running this same configuration through `run_io` in a `git worktree` at
**43854b2**, the commit immediately before the P-H range, on GPU 3 with
`deterministic_flag=1` and `dp_seed=1000`. It is therefore a parity proof
against the pre-P-H driver, not merely a regression lock against HEAD.

If this test fails, do not regenerate the fixture: the `--norm-policy legacy`
bit-exactness guarantee is void and the regression has to be found. Regenerate
only when the *fixture's own* preconditions changed (a different DREAMPlace
build or a different `simple.json`), which the `config_sha256` check below
catches for the config half.
"""
import hashlib
import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement_io import run_io

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "legacy_driver_golden_simple.json"

#: The deterministic flags the fixture was recorded with -- the same ones
#: `tests/test_norm_driver.py`'s legacy driver tests use, plus a fixed
#: `dp_seed`/`deterministic` pair.
RUN_KWARGS = dict(rho_max=.4, every=5, of_on=2., of_full=1.,
                  callback_order="atomic", f_ft_max=.25,
                  ft_ramp_mode="constant", no_diag=True, dp_seed=1000,
                  deterministic=1)


def _golden_config(tmp_path):
    """The exact config the fixture was generated from. `config_sha256` in the
    fixture locks this byte-for-byte, so a changed `simple.json` fails loudly
    instead of silently comparing two different placements."""
    cfg = json.load(open(SIMPLE))
    cfg.update(num_threads=4, plot_flag=0, num_bins_x=16, num_bins_y=16,
               global_place_stages=[dict(num_bins_x=16, num_bins_y=16,
                                         iteration=40, learning_rate=.01,
                                         wirelength="weighted_average",
                                         optimizer="nesterov")])
    text = json.dumps(cfg)
    path = tmp_path / "simple_norm.json"
    path.write_text(text)
    return str(path), hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.slow
def test_legacy_driver_trajectory_matches_the_pre_ph_golden(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["provenance"]["revision"] == "43854b2"
    for name, value in RUN_KWARGS.items():
        assert fixture["provenance"]["run_io_kwargs"][name] == value, (
            "the fixture was recorded with a different %r" % (name,))

    config, sha = _golden_config(tmp_path)
    assert sha == fixture["config_sha256"], (
        "the golden config changed (simple.json at %s?) -- this test would be "
        "comparing two different placements" % (SIMPLE,))

    result = run_io(config, 4, "grid", 0, str(tmp_path / "legacy_golden.json"),
                    **RUN_KWARGS)
    assert result["norm_policy"] == "legacy"

    keys = tuple(fixture["keys"])
    got = [dict((key, event[key]) for key in keys)
           for event in result["trajectory"]]
    want = fixture["trajectory"]
    assert [row["iteration"] for row in got] == [row["iteration"] for row in want]
    for row, expected in zip(got, want):
        for key in keys:
            if isinstance(expected[key], bool) or isinstance(expected[key], int):
                assert row[key] == expected[key], (key, row["iteration"])
            else:
                assert row[key] == pytest.approx(expected[key], rel=1e-12), (
                    "%s at iteration %d: %r != %r (pre-P-H)"
                    % (key, row["iteration"], row[key], expected[key]))
    assert any(row["lambda_io"] > 0.0 for row in got), "lambda never activated"
