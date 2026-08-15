"""Stage 2 S0 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
plan.md` sec 10 S0 row / sec 3 前置條件盤點): tests for `ioplace/diagnostics/
probes_stage2/probe_env.py`.

Two kinds of coverage, mirroring `tests/test_probes_m3_schema.py`'s sealed-
probe pattern but adapted to S0's own instruction ("不要在測試裡真的掃 9.7GB
檔案"):
  - schema tests against the real, already-produced `results/stage2/
    env.json` -- these only `json.load` a file already on disk, so they
    never scan anything themselves.
  - unit tests for the probe's building blocks (`_file_info`,
    `_dir_manifest`, `_tcp_probe`, `_run`, `probe_benchmarks`) on
    `tmp_path`/`monkeypatch` fixtures, so the >1GB sha256-skip gate and the
    benchmark-inventory walk are exercised against small fake files/trees
    instead of the real multi-GB `mempool_cluster`/`mempool_group` DEFs.
"""
import datetime
import hashlib
import json
import os

from ioplace.diagnostics.probes_stage2 import probe_env as pe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_JSON = os.path.join(REPO, "results", "stage2", "env.json")


# ---------------------------------------------------------------------------
# Schema tests against the real, already-produced results/stage2/env.json.
# ---------------------------------------------------------------------------

def _load_env_json():
    assert os.path.exists(ENV_JSON), f"missing probe output: {ENV_JSON} (run probe_env.py first)"
    with open(ENV_JSON) as f:
        return json.load(f)


def test_env_json_top_level_keys():
    d = _load_env_json()
    assert isinstance(d, dict)
    for k in ("provenance", "innovus", "openroad", "benchmarks"):
        assert k in d, f"env.json missing top-level key {k!r}"
        assert isinstance(d[k], dict)


def test_env_json_provenance_schema():
    prov = _load_env_json()["provenance"]
    required = ("hostname", "utc_timestamp", "repo_commit", "dp_commit",
                "command", "python_executable", "python_version")
    for k in required:
        assert k in prov, f"provenance missing {k!r}"
        assert prov[k] is not None
    assert len(prov["repo_commit"]) == 40, "repo_commit doesn't look like a git SHA"
    assert len(prov["dp_commit"]) == 40, "dp_commit doesn't look like a git SHA"
    datetime.datetime.fromisoformat(prov["utc_timestamp"])


def test_env_json_innovus_schema_and_license_not_faked():
    inv = _load_env_json()["innovus"]
    for k in ("cshrc_path", "cshrc_exists", "binary_symlink_path",
              "binary_symlink_exists", "resolved_binary_path", "version_dir",
              "version_probe", "ksh_interpreter_available", "ksh_diagnostic",
              "license"):
        assert k in inv, f"innovus missing {k!r}"
    lic = inv["license"]
    for k in ("lm_license_file_pattern", "port", "servers", "any_reachable"):
        assert k in lic, f"innovus.license missing {k!r}"
    assert isinstance(lic["servers"], list) and len(lic["servers"]) == 3, \
        "spec sec 3.1: exactly 3 license servers (lshc/lstc/lstn)"
    for s in lic["servers"]:
        for k in ("host", "port", "dns_resolved", "resolved_ip", "tcp_connect_ok", "error"):
            assert k in s, f"license server entry missing {k!r}"
        assert isinstance(s["tcp_connect_ok"], bool)
    # sec 10 S0 red line: "license 不通如實記為 false" -- any_reachable must
    # be genuinely derived from the per-server results, never hardcoded.
    assert lic["any_reachable"] == any(s["tcp_connect_ok"] for s in lic["servers"])


def test_env_json_openroad_schema():
    orr = _load_env_json()["openroad"]
    for k in ("binary_path", "binary_exists", "version_probe", "version_string",
              "features", "commands_available", "odb_import_ok", "odb_probe"):
        assert k in orr, f"openroad missing {k!r}"
    assert isinstance(orr["binary_exists"], bool)
    for cmd in ("read_lef", "read_def", "global_route", "detailed_route",
                "write_def", "set_thread_count"):
        assert cmd in orr["commands_available"], f"commands_available missing {cmd!r}"
        assert isinstance(orr["commands_available"][cmd], bool)


def test_env_json_benchmarks_schema():
    bm = _load_env_json()["benchmarks"]
    for k in ("sha256_size_limit_bytes", "ispd2005", "ispd2015", "ispd2025_nangate45"):
        assert k in bm, f"benchmarks missing {k!r}"
    assert bm["sha256_size_limit_bytes"] == 1_000_000_000
    assert set(bm["ispd2005"]["designs"]) == set(pe.ISPD2005_DESIGNS)
    assert set(bm["ispd2015"]["designs"]) == set(pe.ISPD2015_DESIGNS)
    assert bm["ispd2015"]["config_root_exists"] is True
    for variant in pe.ISPD25_VARIANTS:
        assert set(bm["ispd2025_nangate45"]["variants"][variant]["designs"]) == set(pe.ISPD25_DESIGNS)
    nangate45 = bm["ispd2025_nangate45"]["nangate45_common"]
    assert nangate45["lef_count"] == 15, "spec sec 3.2b: NanGate45/lef has 15 files"
    assert nangate45["lib_count"] == 15, "spec sec 3.2b: NanGate45/lib has 15 files"


def test_env_json_large_files_are_size_skipped_not_hashed():
    """The property S0 exists to guarantee: mempool_cluster's multi-GB
    def/net must never carry a real sha256, only a recorded size."""
    bm = _load_env_json()["benchmarks"]
    for variant in pe.ISPD25_VARIANTS:
        cluster = bm["ispd2025_nangate45"]["variants"][variant]["designs"]["mempool_cluster"]["files"]
        for fname in ("mempool_cluster.def", "mempool_cluster.net"):
            info = cluster[fname]
            assert info["size_bytes"] > pe.SHA256_SIZE_LIMIT_BYTES
            assert info["sha256"] == "skipped_size"


# ---------------------------------------------------------------------------
# Unit tests for the probe's building blocks, all on tmp_path fixtures --
# never touch the real multi-GB benchmark trees.
# ---------------------------------------------------------------------------

def test_file_info_missing_file(tmp_path):
    missing = str(tmp_path / "does_not_exist.def")
    info = pe._file_info(missing)
    assert info == dict(path=missing, exists=False, size_bytes=None, sha256=None)


def test_file_info_small_file_hashes(tmp_path):
    p = tmp_path / "small.txt"
    p.write_bytes(b"hello stage2")
    info = pe._file_info(str(p))
    assert info["exists"] is True
    assert info["size_bytes"] == len(b"hello stage2")
    assert info["sha256"] == hashlib.sha256(b"hello stage2").hexdigest()


def test_file_info_skips_sha256_over_size_limit(tmp_path, monkeypatch):
    """Exercises the >1GB skip path without writing a real 1GB file --
    monkeypatch the limit down instead, since the mechanism under test is
    the size comparison, not any particular byte count."""
    monkeypatch.setattr(pe, "SHA256_SIZE_LIMIT_BYTES", 4)
    p = tmp_path / "over_limit.txt"
    p.write_bytes(b"12345")  # 5 bytes > monkeypatched 4-byte limit
    info = pe._file_info(str(p))
    assert info["exists"] is True
    assert info["size_bytes"] == 5
    assert info["sha256"] == "skipped_size"


def test_dir_manifest_missing_dir(tmp_path):
    missing = str(tmp_path / "nope")
    m = pe._dir_manifest(missing)
    assert m == dict(dir=missing, exists=False, files={})


def test_dir_manifest_lists_files_not_subdirs(tmp_path):
    (tmp_path / "a.def").write_bytes(b"x")
    (tmp_path / "b.lef").write_bytes(b"yy")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.def").write_bytes(b"zzz")
    m = pe._dir_manifest(str(tmp_path))
    assert m["exists"] is True
    assert set(m["files"]) == {"a.def", "b.lef"}
    assert m["files"]["a.def"]["size_bytes"] == 1
    assert m["files"]["b.lef"]["size_bytes"] == 2


def test_tcp_probe_dns_failure(monkeypatch):
    def _raise(host):
        raise OSError("nope")
    monkeypatch.setattr(pe.socket, "gethostbyname", _raise)
    r = pe._tcp_probe("no-such-host", 5280)
    assert r["dns_resolved"] is False
    assert r["tcp_connect_ok"] is False
    assert r["resolved_ip"] is None


def test_tcp_probe_connect_failure(monkeypatch):
    """Mirrors sec 12 P1's real result (DNS OK, TCP connect fails) without
    depending on the live network state of this host."""
    monkeypatch.setattr(pe.socket, "gethostbyname", lambda host: "127.0.0.1")

    class _FakeSocket:
        def __init__(self, *a, **k): pass
        def settimeout(self, t): pass
        def connect(self, addr): raise OSError(113, "No route to host")
        def close(self): pass

    monkeypatch.setattr(pe.socket, "socket", lambda *a, **k: _FakeSocket())
    r = pe._tcp_probe("fake-host", 5280)
    assert r["dns_resolved"] is True
    assert r["resolved_ip"] == "127.0.0.1"
    assert r["tcp_connect_ok"] is False


def test_tcp_probe_connect_success(monkeypatch):
    monkeypatch.setattr(pe.socket, "gethostbyname", lambda host: "127.0.0.1")

    class _FakeSocket:
        def __init__(self, *a, **k): pass
        def settimeout(self, t): pass
        def connect(self, addr): pass
        def close(self): pass

    monkeypatch.setattr(pe.socket, "socket", lambda *a, **k: _FakeSocket())
    r = pe._tcp_probe("fake-host", 5280)
    assert r["tcp_connect_ok"] is True
    assert r["error"] is None


def test_run_missing_binary_returns_error_not_raise():
    r = pe._run(["/no/such/binary", "-version"], timeout=5)
    assert r["returncode"] is None
    assert r["error"] is not None


def test_run_normal_command():
    r = pe._run(["echo", "hi"], timeout=5)
    assert r["returncode"] == 0
    assert r["stdout"].strip() == "hi"
    assert r["error"] is None


def test_run_timeout():
    r = pe._run(["sleep", "5"], timeout=0.2)
    assert r["timed_out"] is True


# ---------------------------------------------------------------------------
# probe_benchmarks() end-to-end shape, on a fake fixture tree (no real
# benchmark I/O -- per S0's "don't scan the real 9.7GB file" instruction).
# ---------------------------------------------------------------------------

def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def test_probe_benchmarks_shape_on_fake_tree(tmp_path, monkeypatch):
    dp_root = tmp_path / "dp"
    bench25_root = tmp_path / "ispd25"

    # ISPD2005: fabricate only one design; the rest stay missing on purpose
    # to exercise the exists=False path too.
    _write(str(dp_root / "benchmarks" / "ispd2005" / "adaptec1" / "adaptec1.aux"), b"aux")

    # ISPD2015: one design + its DREAMPlace config.
    _write(str(dp_root / "benchmarks" / "ispd2015" / "mgc_fft_1" / "tech.lef"), b"lef")
    _write(str(dp_root / "install" / "test" / "ispd2015" / "lefdef" / "mgc_fft_1.json"), b"{}")

    # ISPD2025: NanGate45 common (1 lef + 1 lib, not the real 15) + one
    # design per variant, with a def file that exceeds the (monkeypatched,
    # tiny) sha256 limit so the size-skip path is exercised without a real
    # multi-GB file.
    _write(str(bench25_root / "NanGate45" / "lef" / "NangateOpenCellLibrary.tech.lef"), b"techlef")
    _write(str(bench25_root / "NanGate45" / "lib" / "NangateOpenCellLibrary_typical.lib"), b"lib")
    for variant in pe.ISPD25_VARIANTS:
        _write(str(bench25_root / variant / "mempool_cluster" / "mempool_cluster.def"), b"x" * 10)

    monkeypatch.setattr(pe, "DP", str(dp_root))
    monkeypatch.setattr(pe, "BENCH25_ROOT", str(bench25_root))
    monkeypatch.setattr(pe, "SHA256_SIZE_LIMIT_BYTES", 5)

    bm = pe.probe_benchmarks()

    assert bm["ispd2005"]["designs"]["adaptec1"]["exists"] is True
    assert bm["ispd2005"]["designs"]["adaptec2"]["exists"] is False
    assert bm["ispd2015"]["designs"]["mgc_fft_1"]["dreamplace_config_json"]["exists"] is True
    assert bm["ispd2015"]["designs"]["mgc_superblue12"]["dreamplace_config_json"]["exists"] is False
    assert bm["ispd2025_nangate45"]["nangate45_common"]["lef_count"] == 1
    assert bm["ispd2025_nangate45"]["nangate45_common"]["lib_count"] == 1
    for variant in pe.ISPD25_VARIANTS:
        files = bm["ispd2025_nangate45"]["variants"][variant]["designs"]["mempool_cluster"]["files"]
        cluster_def = files["mempool_cluster.def"]
        assert cluster_def["size_bytes"] == 10
        assert cluster_def["sha256"] == "skipped_size"  # 10 > monkeypatched 5-byte limit
        # a design with no fabricated files at all still gets a (empty) manifest.
        assert bm["ispd2025_nangate45"]["variants"][variant]["designs"]["ariane"]["exists"] is False
