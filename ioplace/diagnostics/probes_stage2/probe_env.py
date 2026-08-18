"""Stage 2 S0 environment probe (`docs/superpowers/specs/2026-08-13-stage2-
innovus-calibration-plan.md` sec 3 前置條件盤點 / sec 10 S0 row / sec 12 證據
附錄): re-runs every check in sec 3/sec 12 on this host and writes the
result to `results/stage2/env.json`, so the plan's "實測" claims are
reproducible from the repo instead of living only in `/tmp/or_probe/`
(sec 12's own "待遷入 repo" note).

Covers, one-to-one with sec 3:
  - sec 3.1 EDA tools: Innovus binary/cshrc existence, `innovus -version`
    (via `tcsh -c "source <cshrc>; innovus -version"`, matching I2's own
    invocation), LM_LICENSE_FILE TCP reachability for all three license
    hosts (`lshc`/`lstc`/`lstn`, port 5280 -- P1), OpenROAD version/feature
    string and the six Tcl commands F-OR's flow skeleton (sec 6.1) needs,
    plus the `odb` Python import (P7).
  - sec 3.2 benchmarks: existence + key-file size/sha256 for ISPD2005
    bookshelf (`$DP/benchmarks/ispd2005`), ISPD2015 LEF/DEF
    (`$DP/benchmarks/ispd2015` + its 20 DREAMPlace configs under
    `$DP/install/test/ispd2015/lefdef`), and ISPD2025 NanGate45
    (`~/benchmarks/ispd25/extracted/ISPD2025_benchmarks`, `visible`/`blind`
    x 6 designs + the shared NanGate45 LEF/LIB library).

sha256 is only computed for files <= `SHA256_SIZE_LIMIT_BYTES` (1e9 bytes,
sec 10 S0's own "≤1GB" cutoff) -- `mempool_cluster.def`/`.net` (9.7GB/8.2GB
in both `visible/` and `blind/`) are the only files this repo has that trip
it; they get `sha256: "skipped_size"` with `size_bytes` still recorded, so
the probe finishes in about a minute instead of hashing ~18GB.

This module is intentionally torch-free (plain stdlib: `socket`,
`subprocess`, `hashlib`, `os`) so it runs under the system `python3`, not
just the DREAMPlace venv -- it never touches `PlaceDB`/DREAMPlace.

Usage:
    python3 -m ioplace.diagnostics.probes_stage2.probe_env
        (also runnable under the DREAMPlace venv; it just doesn't need it)
"""
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
BENCH25_ROOT = "/nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks"

INNOVUS_CSHRC = "/nashome/CAD/cadence/CIC/innovus.cshrc"
INNOVUS_BIN_SYMLINK = "/usr/cad/cadence/INNOVUS/cur/tools/bin/innovus"
LICENSE_SERVERS = ("lshc", "lstc", "lstn")
LICENSE_PORT = 5280

OPENROAD_BIN = "/usr/bin/openroad"
OPENROAD_COMMANDS = (
    "read_lef", "read_def", "global_route", "detailed_route",
    "write_def", "set_thread_count",
)

# sec 10 S0: "sha256 只對 <=1GB 的檔案算" -- decimal GB, matching the plan's
# own "9.7GB"/"2.3GB" file-size prose (sec 3.2b).
SHA256_SIZE_LIMIT_BYTES = 1_000_000_000

ISPD2005_DESIGNS = (
    "adaptec1", "adaptec2", "adaptec3", "adaptec4",
    "bigblue1", "bigblue2", "bigblue3", "bigblue4",
)
ISPD2015_DESIGNS = (
    "mgc_des_perf_1", "mgc_des_perf_a", "mgc_des_perf_b",
    "mgc_edit_dist_a",
    "mgc_fft_1", "mgc_fft_2", "mgc_fft_a", "mgc_fft_b",
    "mgc_matrix_mult_1", "mgc_matrix_mult_2", "mgc_matrix_mult_a",
    "mgc_matrix_mult_b", "mgc_matrix_mult_c",
    "mgc_pci_bridge32_a", "mgc_pci_bridge32_b",
    "mgc_superblue11_a", "mgc_superblue12", "mgc_superblue14",
    "mgc_superblue16_a", "mgc_superblue19",
)
ISPD25_DESIGNS = (
    "ariane", "bsg_chip", "mempool_cluster", "mempool_group",
    "mempool_tile_wrap", "NV_NVDLA_partition_c",
)
ISPD25_VARIANTS = ("visible", "blind")


def _git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.STDOUT,
        ).decode().strip()
    except Exception as e:
        return f"error: {e!r}"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _file_info(path):
    """Existence + size + (size-gated) sha256 for one file. Never raises --
    a missing benchmark file is a fact to record (`exists: false`), not a
    crash."""
    exists = os.path.isfile(path)
    if not exists:
        return dict(path=path, exists=False, size_bytes=None, sha256=None)
    size_bytes = os.path.getsize(path)
    if size_bytes > SHA256_SIZE_LIMIT_BYTES:
        sha = "skipped_size"
    else:
        sha = _sha256(path)
    return dict(path=path, exists=True, size_bytes=size_bytes, sha256=sha)


def _dir_manifest(dir_path):
    """Non-recursive per-file manifest of every regular file directly under
    `dir_path` (sorted by name) -- used instead of a hardcoded per-design
    filename list so an unexpected extra/missing file (e.g. sec 3.2b's
    `blind/mempool_tile_wrap` shipping an uncompressed `.v` the other
    designs don't have) is recorded as-is rather than silently dropped."""
    if not os.path.isdir(dir_path):
        return dict(dir=dir_path, exists=False, files={})
    names = sorted(
        n for n in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, n))
    )
    return dict(
        dir=dir_path, exists=True,
        files={n: _file_info(os.path.join(dir_path, n)) for n in names},
    )


def _run(cmd, timeout=30, **kwargs):
    """Generic subprocess runner: never raises for a normal exec failure --
    returns a dict with `returncode`/`stdout`/`stderr` (or `error`) so a
    failed probe (missing binary, license error, timeout) is data, not an
    exception that aborts the whole probe run."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, **kwargs,
        )
        return dict(
            command=cmd if isinstance(cmd, str) else " ".join(cmd),
            returncode=r.returncode, stdout=r.stdout, stderr=r.stderr,
            timed_out=False, error=None,
        )
    except subprocess.TimeoutExpired as e:
        return dict(
            command=cmd if isinstance(cmd, str) else " ".join(cmd),
            returncode=None, stdout=e.stdout or "", stderr=e.stderr or "",
            timed_out=True, error=None,
        )
    except OSError as e:
        # e.g. FileNotFoundError for a missing binary, or (this host's own
        # finding, sec 12 addendum) a script whose #! interpreter is
        # missing -- execve(2) reports that as ENOENT too.
        return dict(
            command=cmd if isinstance(cmd, str) else " ".join(cmd),
            returncode=None, stdout="", stderr="", timed_out=False,
            error=repr(e),
        )


def _tcp_probe(host, port, timeout=3.0):
    try:
        resolved_ip = socket.gethostbyname(host)
    except OSError as e:
        return dict(
            host=host, port=port, dns_resolved=False, resolved_ip=None,
            tcp_connect_ok=False, error=repr(e),
        )
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((resolved_ip, port))
        return dict(
            host=host, port=port, dns_resolved=True, resolved_ip=resolved_ip,
            tcp_connect_ok=True, error=None,
        )
    except OSError as e:
        return dict(
            host=host, port=port, dns_resolved=True, resolved_ip=resolved_ip,
            tcp_connect_ok=False, error=repr(e),
        )
    finally:
        s.close()


def probe_innovus():
    """sec 3.1 Innovus rows + I1/I2 + P1/P2."""
    cshrc_exists = os.path.isfile(INNOVUS_CSHRC)
    symlink_exists = os.path.islink(INNOVUS_BIN_SYMLINK) or os.path.exists(INNOVUS_BIN_SYMLINK)
    resolved_binary_path = os.path.realpath(INNOVUS_BIN_SYMLINK) if symlink_exists else None
    version_dir = None
    if os.path.islink("/usr/cad/cadence/INNOVUS/cur"):
        version_dir = os.readlink("/usr/cad/cadence/INNOVUS/cur")

    # I2: the documented invocation (`source <cshrc>; innovus -version`).
    version_probe = None
    if cshrc_exists:
        version_probe = _run(
            ["tcsh", "-c", f"source {INNOVUS_CSHRC} >& /dev/null; innovus -version"],
            timeout=30,
        )

    # Addendum to sec 12 (this host, not previously recorded there): check
    # whether the #! interpreter of the wrapped binary itself is present --
    # explains a `tcsh`-reported "innovus: Command not found." that is
    # *not* the license failure (I1) but a missing `/bin/ksh` on this host.
    ksh_available = bool(_run(["sh", "-c", "command -v ksh"], timeout=10)["stdout"].strip())
    ksh_diagnostic = None
    if resolved_binary_path and os.path.isfile(resolved_binary_path):
        ksh_diagnostic = _run([resolved_binary_path, "-version"], timeout=15)

    servers = [_tcp_probe(h, LICENSE_PORT) for h in LICENSE_SERVERS]

    return dict(
        cshrc_path=INNOVUS_CSHRC, cshrc_exists=cshrc_exists,
        binary_symlink_path=INNOVUS_BIN_SYMLINK, binary_symlink_exists=symlink_exists,
        resolved_binary_path=resolved_binary_path, version_dir=version_dir,
        version_probe=version_probe,
        ksh_interpreter_available=ksh_available, ksh_diagnostic=ksh_diagnostic,
        license=dict(
            lm_license_file_pattern=":".join(f"{LICENSE_PORT}@{h}" for h in LICENSE_SERVERS),
            port=LICENSE_PORT, servers=servers,
            any_reachable=any(s["tcp_connect_ok"] for s in servers),
        ),
    )


def _is_executable_file(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def probe_openroad():
    """sec 3.1 OpenROAD rows + P3/P7."""
    binary_exists = _is_executable_file(OPENROAD_BIN)
    version_probe = _run([OPENROAD_BIN, "-version"], timeout=20)
    # Feature flags (+Charts +GPU +GUI +Python) are only printed on the
    # normal splash banner, not `-version`'s bare version string (P3) --
    # get both from one `exit`-only run.
    splash_probe = _run([OPENROAD_BIN], timeout=20, input="exit\n")
    features = {}
    for line in splash_probe["stdout"].splitlines():
        if line.startswith("Features included"):
            for tok in line.split(":")[1].split():
                if tok.startswith("+"):
                    features[tok[1:]] = True
                elif tok.startswith("-"):
                    features[tok[1:]] = False

    tcl = "; ".join(f"puts [info commands {c}]" for c in OPENROAD_COMMANDS) + "; exit"
    cmds_probe = _run([OPENROAD_BIN, "-no_splash"], timeout=20, input=tcl + "\n")
    cmds_out_lines = [ln.strip() for ln in cmds_probe["stdout"].splitlines()]
    commands_available = {c: (c in cmds_out_lines) for c in OPENROAD_COMMANDS}

    odb_probe = _run(
        [OPENROAD_BIN, "-python", "-c",
         "import odb; print('ODB_IMPORT_OK'); "
         "print(sorted(a for a in dir(odb.dbWireDecoder) if not a.startswith('_')))"],
        timeout=20,
    )
    odb_import_ok = "ODB_IMPORT_OK" in odb_probe["stdout"]

    return dict(
        binary_path=OPENROAD_BIN, binary_exists=bool(binary_exists),
        version_probe=version_probe,
        version_string=version_probe["stdout"].strip() or None,
        features=features,
        commands_available=commands_available,
        odb_import_ok=odb_import_ok, odb_probe=odb_probe,
    )


def probe_benchmarks():
    """sec 3.2 benchmark inventory."""
    ispd2005_root = os.path.join(DP, "benchmarks", "ispd2005")
    ispd2005 = dict(
        root=ispd2005_root, root_exists=os.path.isdir(ispd2005_root),
        note="bookshelf only, not used for calibration (spec sec 4.2 4.2c)",
        designs={d: _dir_manifest(os.path.join(ispd2005_root, d)) for d in ISPD2005_DESIGNS},
    )

    ispd2015_root = os.path.join(DP, "benchmarks", "ispd2015")
    ispd2015_config_root = os.path.join(DP, "install", "test", "ispd2015", "lefdef")
    ispd2015_designs = {}
    for d in ISPD2015_DESIGNS:
        manifest = _dir_manifest(os.path.join(ispd2015_root, d))
        cfg_path = os.path.join(ispd2015_config_root, f"{d}.json")
        manifest["dreamplace_config_json"] = dict(
            path=cfg_path, exists=os.path.isfile(cfg_path),
        )
        ispd2015_designs[d] = manifest
    ispd2015 = dict(
        root=ispd2015_root, root_exists=os.path.isdir(ispd2015_root),
        config_root=ispd2015_config_root, config_root_exists=os.path.isdir(ispd2015_config_root),
        designs=ispd2015_designs,
    )

    nangate45_root = os.path.join(BENCH25_ROOT, "NanGate45")
    lef_manifest = _dir_manifest(os.path.join(nangate45_root, "lef"))
    lib_manifest = _dir_manifest(os.path.join(nangate45_root, "lib"))
    nangate45_common = dict(
        root=nangate45_root, root_exists=os.path.isdir(nangate45_root),
        lef=lef_manifest, lef_count=len(lef_manifest["files"]),
        lib=lib_manifest, lib_count=len(lib_manifest["files"]),
    )

    variants = {}
    for variant in ISPD25_VARIANTS:
        variant_root = os.path.join(BENCH25_ROOT, variant)
        variants[variant] = dict(
            root=variant_root, root_exists=os.path.isdir(variant_root),
            designs={
                d: _dir_manifest(os.path.join(variant_root, d)) for d in ISPD25_DESIGNS
            },
        )

    ispd2025_nangate45 = dict(
        root=BENCH25_ROOT, root_exists=os.path.isdir(BENCH25_ROOT),
        nangate45_common=nangate45_common, variants=variants,
    )

    return dict(
        sha256_size_limit_bytes=SHA256_SIZE_LIMIT_BYTES,
        ispd2005=ispd2005, ispd2015=ispd2015, ispd2025_nangate45=ispd2025_nangate45,
    )


def provenance():
    return dict(
        hostname=socket.gethostname(),
        utc_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repo_commit=_git_head(REPO), dp_commit=_git_head(DP),
        command=" ".join(sys.argv),
        python_executable=sys.executable, python_version=platform.python_version(),
    )


def run():
    return dict(
        provenance=provenance(),
        innovus=probe_innovus(),
        openroad=probe_openroad(),
        benchmarks=probe_benchmarks(),
    )


def main():
    result = run()
    out_path = os.path.join(REPO, "results", "stage2", "env.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)

    inv, orr = result["innovus"], result["openroad"]
    print(f"[probe_env] wrote {out_path}")
    print(f"  innovus: license any_reachable={inv['license']['any_reachable']} "
          f"ksh_available={inv['ksh_interpreter_available']}")
    print(f"  openroad: version={orr['version_string']!r} features={orr['features']} "
          f"odb_import_ok={orr['odb_import_ok']}")
    return result


if __name__ == "__main__":
    main()
