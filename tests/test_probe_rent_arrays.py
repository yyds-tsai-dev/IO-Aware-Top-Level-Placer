"""M4 T6/T6B (2026-08-15 T6 holdout adjudication doc `docs/results/2026-08-
15-m4-t6-holdout-adjudication.md` sec 8 item 1): tests for `ioplace/
diagnostics/probes_m4/probe_rent_arrays.py`.

No test here actually runs mtkahypar or loads a real (multi-GB) Bookshelf/
DEF netlist -- `rent.measure_rent`/`rent.load_bookshelf_netlist`/
`rent.load_def_netlist` are monkeypatched with small fakes throughout, in
the same spirit as `tests/test_probes_stage2_schema.py`'s "don't scan the
real multi-GB file in a test" instruction. Coverage is split the same way
that file's docstring describes:
  - unit tests for the probe's building blocks (`resolve_target`,
    `_bookshelf_input_paths`) against tiny `tmp_path` fixtures;
  - an end-to-end `main()` CLI/schema test with `rent.measure_rent`/
    `rent.load_bookshelf_netlist` monkeypatched to fast fakes, checking the
    written JSON's schema and the `.net_degrees.npy` sidecar;
  - a frozen-parameters test locking `BACKEND`/`THREADS`/`B_LO`/`B_HI`/
    `N_BOOTSTRAP`/`MAX_NET_DEGREE` to the adjudication doc sec 4.1 values
    and confirming none of them is exposed as a CLI flag (sec 4.1: "所有
    項目一律 ... Rent 一律 backend="mtkahypar" ... 兩邊同套" -- a future
    run must not be able to silently pick a different ruler).
"""
import json
import os

import numpy as np
import pytest

from ioplace.bench import rent
from ioplace.diagnostics.probes_m4 import probe_rent_arrays as pra

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Frozen Rent parameters (adjudication doc sec 4.1) -- not a CLI knob.
# ---------------------------------------------------------------------------

def test_rent_params_match_adjudication_doc_sec_4_1():
    assert pra.BACKEND == "mtkahypar"
    assert pra.THREADS == 16
    assert pra.B_LO == 1e3
    assert pra.B_HI == 1e6
    assert pra.N_BOOTSTRAP == 200
    assert pra.MAX_NET_DEGREE == 100


@pytest.mark.parametrize("flag", ["--backend", "--threads", "--b-lo", "--b-hi",
                                   "--n-bootstrap", "--max-net-degree"])
def test_cli_does_not_expose_rent_params(flag):
    with pytest.raises(SystemExit):
        pra.main(["--target", "cluster", flag, "1"])


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------

def test_resolve_target_cluster():
    kind, path = pra.resolve_target("cluster", None)
    assert kind == "def"
    assert path == pra.CLUSTER_DEF


@pytest.mark.parametrize("alias", ["group", "source", "1x1_source"])
def test_resolve_target_source_aliases(alias):
    kind, path = pra.resolve_target(alias, None)
    assert kind == "bookshelf"
    assert path == pra.SOURCE_PREFIX


@pytest.mark.parametrize("target", ["1x2_n1", "1x2_n2", "2x2_n1", "2x2_n2", "3x3_n2", "3x3_n2_noglue"])
def test_resolve_target_generic_array_convention(target):
    """T6B (adjudication doc sec 4.1) needs new array names (`3x3_n2`,
    `*-noglue` variants) that don't exist as arrays yet -- this probe must
    resolve them via `build_t6_arrays.py`'s existing dir==prefix
    convention without needing a code change per new shape/variant."""
    kind, path = pra.resolve_target(target, None)
    assert kind == "bookshelf"
    assert path == os.path.join(REPO, "results", "m4", "bench", "arrays", target, target)


def test_resolve_target_prefix_override_wins_over_target():
    kind, path = pra.resolve_target("cluster", "/some/explicit/prefix")
    assert kind == "bookshelf"
    assert path == "/some/explicit/prefix"


def test_resolve_target_prefix_only():
    kind, path = pra.resolve_target(None, "/some/explicit/prefix")
    assert kind == "bookshelf"
    assert path == "/some/explicit/prefix"


# ---------------------------------------------------------------------------
# _bookshelf_input_paths
# ---------------------------------------------------------------------------

def test_bookshelf_input_paths_only_lists_existing_files(tmp_path):
    prefix = str(tmp_path / "case")
    (tmp_path / "case.nodes").write_text("x")
    (tmp_path / "case.nets").write_text("x")
    # .pl deliberately missing
    paths = pra._bookshelf_input_paths(prefix)
    assert paths == [prefix + ".nodes", prefix + ".nets"]


def test_bookshelf_input_paths_all_missing(tmp_path):
    prefix = str(tmp_path / "nope")
    assert pra._bookshelf_input_paths(prefix) == []


# ---------------------------------------------------------------------------
# run() / main() end to end, with rent.measure_rent and the netlist loaders
# monkeypatched to small fakes.
# ---------------------------------------------------------------------------

class _FakeNetlist:
    def __init__(self, num_physical, num_nets, net_degrees):
        self.num_physical = num_physical
        self.num_nets = num_nets
        self.net_degrees = np.asarray(net_degrees)


def _fake_rent_result():
    levels = [
        rent.RentLevel(level=0, n_blocks=1, block_sizes=[1000], terminals=[0]),
        rent.RentLevel(level=1, n_blocks=2, block_sizes=[500, 500], terminals=[10, 12]),
    ]
    return rent.RentResult(p=0.5, p_ci_lo=0.45, p_ci_hi=0.55, log_t=1.0,
                            backend="mtkahypar", levels_used=[1], levels=levels, n_bootstrap=5)


def test_run_bookshelf_target_calls_measure_rent_with_frozen_params(tmp_path, monkeypatch):
    prefix = str(tmp_path / "arrays" / "2x2_n2" / "2x2_n2")
    os.makedirs(os.path.dirname(prefix))
    for ext in (".nodes", ".nets", ".pl"):
        with open(prefix + ext, "w") as f:
            f.write("x")

    captured = {}

    def fake_load_bookshelf_netlist(p):
        captured["load_prefix"] = p
        return _FakeNetlist(100, 50, [1, 2, 5, 5])

    def fake_measure_rent(nl, **kwargs):
        captured["measure_rent_kwargs"] = kwargs
        captured["nl"] = nl
        return _fake_rent_result()

    monkeypatch.setattr(pra.rent, "load_bookshelf_netlist", fake_load_bookshelf_netlist)
    monkeypatch.setattr(pra.rent, "measure_rent", fake_measure_rent)

    result, net_degrees = pra.run(prefix=prefix, seed=0)

    assert captured["load_prefix"] == prefix
    # frozen ruler, sec 4.1 -- every call must use exactly these.
    assert captured["measure_rent_kwargs"] == dict(
        b_lo=pra.B_LO, b_hi=pra.B_HI, seed=0, backend=pra.BACKEND,
        threads=pra.THREADS, n_bootstrap=pra.N_BOOTSTRAP, max_net_degree=pra.MAX_NET_DEGREE)

    assert result["p"] == 0.5
    assert result["p_ci_lo"] == 0.45
    assert result["p_ci_hi"] == 0.55
    assert result["backend"] == "mtkahypar"
    assert result["levels_used"] == [1]
    assert result["n_bootstrap"] == 5
    assert result["levels"] == [
        {"level": 0, "n_blocks": 1, "avg_block_size": 1000.0, "avg_terminals": 0.0},
        {"level": 1, "n_blocks": 2, "avg_block_size": 500.0, "avg_terminals": 11.0},
    ]
    assert result["net_degrees_max"] == 5
    assert result["num_nets"] == 50
    assert result["num_physical"] == 100
    assert result["rent_params"] == dict(backend=pra.BACKEND, threads=pra.THREADS, b_lo=pra.B_LO,
                                          b_hi=pra.B_HI, n_bootstrap=pra.N_BOOTSTRAP,
                                          max_net_degree=pra.MAX_NET_DEGREE, seed=0)
    assert "meta" not in result  # bookshelf loader has no meta, unlike the DEF loader
    assert set(result["input_paths"]) == {prefix + ".nodes", prefix + ".nets", prefix + ".pl"}
    assert list(net_degrees) == [1, 2, 5, 5]


def test_run_def_target_includes_meta(tmp_path, monkeypatch):
    def_path = str(tmp_path / "mempool_cluster.def")
    with open(def_path, "w") as f:
        f.write("x")

    def fake_load_def_netlist(p):
        return _FakeNetlist(10, 5, [1, 1]), {"n_physical": 10, "n_nets": 5, "n_pins": 20}

    monkeypatch.setattr(pra, "CLUSTER_DEF", def_path)
    monkeypatch.setattr(pra.rent, "load_def_netlist", fake_load_def_netlist)
    monkeypatch.setattr(pra.rent, "measure_rent", lambda nl, **kw: _fake_rent_result())

    result, _ = pra.run(target="cluster", seed=0)
    assert result["kind"] == "def"
    assert result["meta"] == {"n_physical": 10, "n_nets": 5, "n_pins": 20}
    assert result["input_paths"] == [def_path]


def test_main_requires_target_or_prefix():
    with pytest.raises(SystemExit):
        pra.main([])


def test_main_writes_json_and_degrees_npy(tmp_path, monkeypatch):
    prefix = str(tmp_path / "arrays" / "2x2_n2" / "2x2_n2")
    os.makedirs(os.path.dirname(prefix))
    for ext in (".nodes", ".nets", ".pl"):
        with open(prefix + ext, "w") as f:
            f.write("x")

    monkeypatch.setattr(pra.rent, "load_bookshelf_netlist", lambda p: _FakeNetlist(100, 50, [1, 2, 5, 5]))
    monkeypatch.setattr(pra.rent, "measure_rent", lambda nl, **kw: _fake_rent_result())

    out_path = str(tmp_path / "out" / "probe_rent_arrays__2x2_n2.json")
    pra.main(["--prefix", prefix, "--out", out_path])

    assert os.path.exists(out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert data["p"] == 0.5
    assert data["backend"] == "mtkahypar"
    assert "env" in data
    for k in ("command", "python_executable", "python_version", "hostname",
              "utc_timestamp", "repo_commit", "input_sha256"):
        assert k in data["env"], f"env missing {k!r}"
    assert set(data["env"]["input_sha256"]) == {prefix + ".nodes", prefix + ".nets", prefix + ".pl"}
    assert "input_paths" not in data  # consumed into env.input_sha256, not duplicated

    deg_path = str(tmp_path / "out" / "probe_rent_arrays__2x2_n2.net_degrees.npy")
    assert os.path.exists(deg_path)
    assert list(np.load(deg_path)) == [1, 2, 5, 5]


def test_main_no_save_degrees_skips_npy(tmp_path, monkeypatch):
    prefix = str(tmp_path / "case")
    for ext in (".nodes", ".nets", ".pl"):
        with open(prefix + ext, "w") as f:
            f.write("x")
    monkeypatch.setattr(pra.rent, "load_bookshelf_netlist", lambda p: _FakeNetlist(10, 5, [1, 1]))
    monkeypatch.setattr(pra.rent, "measure_rent", lambda nl, **kw: _fake_rent_result())

    out_path = str(tmp_path / "probe_rent_arrays__case.json")
    pra.main(["--prefix", prefix, "--out", out_path, "--no-save-degrees"])

    assert os.path.exists(out_path)
    assert not os.path.exists(str(tmp_path / "probe_rent_arrays__case.net_degrees.npy"))


def test_main_default_out_path_uses_target_name(tmp_path, monkeypatch):
    prefix = str(tmp_path / "arrays" / "2x2_n2" / "2x2_n2")
    os.makedirs(os.path.dirname(prefix))
    for ext in (".nodes", ".nets", ".pl"):
        with open(prefix + ext, "w") as f:
            f.write("x")
    monkeypatch.setattr(pra, "ALIASES", {})
    monkeypatch.setattr(pra, "REPO", str(tmp_path))
    # tmp_path isn't a git checkout -- only the default-out-path resolution
    # (which reads the patched REPO) is under test here, not provenance.
    monkeypatch.setattr(pra, "_git_head", lambda path: "0" * 40)
    monkeypatch.setattr(pra.rent, "load_bookshelf_netlist", lambda p: _FakeNetlist(10, 5, [1, 1]))
    monkeypatch.setattr(pra.rent, "measure_rent", lambda nl, **kw: _fake_rent_result())

    pra.main(["--target", "2x2_n2"])

    expected = tmp_path / "results" / "m4" / "probes" / "probe_rent_arrays__2x2_n2.json"
    assert expected.exists()
