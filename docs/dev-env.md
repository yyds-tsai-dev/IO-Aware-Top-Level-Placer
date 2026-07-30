# Development Environment

Built for Task 1 (rebuild DREAMPlace against Python 3.12 + repo skeleton). This is the
authoritative reference for how `$DP` (DREAMPlace) is built and run; every later task depends on
it.

## Summary

| | |
|---|---|
| Python | **3.12.13** (via `uv`), venv at `$DP/.venv312` |
| torch | **2.8.0+cu128** (`torch.cuda.is_available()==True`, device: NVIDIA L4) |
| numpy | **1.26.4** (pinned `<2`; DREAMPlace's `PlaceDB.py` uses `np.string_`, removed in numpy 2.x) |
| pytest | 9.1.1 |
| DREAMPlace build | CUDA-enabled (`CUDA_FOUND: "TRUE"`), fresh `cpython-312` ABI `.so`s |
| `DREAMPLACE_ROOT` | defaults to `/nashome/NVL4/vdalab/yyds-dev/DREAMPlace` |

`DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`. `$PY` throughout this doc (and every later task)
means the absolute path `$DP/.venv312/bin/python`.

## Why a from-scratch rebuild was necessary

The pre-existing `$DP/.venv` (Python 3.9) had its `bin/python` as a **broken symlink** to
`/usr/bin/python3.9`, which does not exist on this host. `.venv/lib/.../site-packages` is
otherwise intact; that tree (and the untouched original `install/`, preserved at
`$DP/install.cp39.bak/`) is a fallback asset only, not used by this environment. Rebuilding
against a fresh interpreter is safe because Python-version binding only lives in the compiled
`*.so` extensions (pybind11/torch glue) — the C++ parsers and CUDA kernels themselves are
Python-version-independent.

## Host toolchain gaps found, and how they were resolved (no root available on this host)

Beyond the Python-level risks the plan anticipated (`numpy<2` for `np.string_`, possible
`distutils` issues on 3.12 — neither actually materialized), this host turned out to be missing
two **system-level, non-Python** build dependencies entirely:

- **No CUDA devel toolkit**: no `nvcc` anywhere on the root filesystem, no `/usr/local/cuda`. Only
  the NVIDIA driver (550.120) and pip-installable CUDA *runtime* shared libraries were present.
  `find_package(CUDA 9.0)` in `cmake/TorchExtension.cmake` is non-`REQUIRED`, so this would have
  **silently** produced a CPU-only build (`TORCH_ENABLE_CUDA=0`) rather than erroring.
- **No Boost development headers or Boost.Graph library**: `/usr/include/boost` did not exist;
  only 5 unrelated Boost *runtime* `.so`s were installed system-wide (no headers, no
  `libboost-graph` at all). `find_package(Boost 1.55.0 REQUIRED)`
  (`$DP/CMakeLists.txt:60`) and, more specifically, `find_package(Boost 1.55.0 REQUIRED COMPONENTS
  graph regex)` (`$DP/thirdparty/Limbo/CMakeLists.txt:123`, needed by Limbo's Bookshelf/LEF/DEF
  parsers — i.e. needed even for the tiny `simple` smoke case) is `REQUIRED` and fails configure
  outright without this.

Root/sudo is not available on this host (`sudo -n true` / `sudo -n -l` both require a password).
The NVIDIA driver's CUDA 12.x **minor-version compatibility** policy means a toolkit-built-with-
12.8 binary runs fine against the 550.120 driver (confirmed empirically before doing any of this:
`torch==2.8.0+cu128` already reported `torch.cuda.is_available()==True` on this driver). Both gaps
were resolved **without root**, self-contained under `$DP/deps/` — outside git, outside the venv,
not part of any tracked repo, safe to `rm -rf` and redo if ever needed:

### Boost 1.74.0 (headers + Boost.Graph + Boost.Regex) — `$DP/deps/boost/`

Ubuntu 22.04 (jammy, the host's apt suite) ships Boost 1.74.0. Rather than building a newer Boost
from source (heavier, and would mix a newer header version against no matching compiled `.so` —
an avoidable ABI risk), the exact matching-version `.deb`s were fetched with `apt-get download`
(works without root — fetches to cwd only, does not install) and unpacked with `dpkg-deb -x`
(extracts a `.deb`'s file tree into an arbitrary directory; no root, no dependency
resolution/enforcement, since we never register these with dpkg):

```bash
mkdir -p $DP/deps/boost_debs && cd $DP/deps/boost_debs
apt-get download libboost1.74-dev libboost-graph1.74-dev libboost-graph1.74.0 \
                  libboost-regex1.74-dev libboost-regex1.74.0
mkdir -p $DP/deps/boost
for f in *.deb; do dpkg-deb -x "$f" $DP/deps/boost; done
```

Result: `$DP/deps/boost/usr/include/boost/*` — the **full** 1.74.0 header set (Ubuntu ships all
Boost headers, including `graph/`, `regex/`, `serialization/`, etc., in the single
`libboost1.74-dev` package; only *compiled* libraries are split per-component) — plus
`$DP/deps/boost/usr/lib/x86_64-linux-gnu/{libboost_graph,libboost_regex}.so{,.1.74.0,.a}`.
`libboost_regex.so` needs ICU at runtime (`libicui18n.so.70` etc.); already present as a system
runtime lib on this host (`ldd` resolved it cleanly), so `libicu-dev` headers were not fetched —
nothing in this build includes ICU headers directly, only Boost's own public regex API.
`$DP/deps/boost_debs/*.deb` (the 5 source packages, ~11MB) are kept for provenance.

cmake flags used to point `FindBoost` (legacy MODULE mode — `$DP/CMakeLists.txt` sets
`Boost_NO_BOOST_CMAKE TRUE` before `find_package(Boost)`, so the CONFIG-mode `cmake/` files that
ship inside the same `.deb`s are never used) at this tree instead of the headerless system Boost:

```
-DBOOST_ROOT=$DP/deps/boost/usr
-DBOOST_INCLUDEDIR=$DP/deps/boost/usr/include
-DBOOST_LIBRARYDIR=$DP/deps/boost/usr/lib/x86_64-linux-gnu
-DBoost_NO_SYSTEM_PATHS=ON
```

### CUDA 12.8 toolkit (nvcc + headers + libcudart) — `$DP/deps/cuda-12.8/`

Installed from NVIDIA's official Linux x86_64 **runfile**, toolkit-only, custom prefix, silent —
this mode does not touch the driver, install no kernel modules, and needs no root:

```bash
cd $DP/deps
curl -LO https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
chmod +x cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=$DP/deps/cuda-12.8 --no-man-page
rm cuda_12.8.1_570.124.06_linux.run   # ~5.4GB, deleted after install; installed toolkit is ~8.7GB
```

`--toolkit --toolkitpath=<path>` installs only compiler/headers/runtime libs (no driver, no kernel
modules) to the given prefix. The install log's `[WARNING]: Unable to write to /var/log` lines are
expected/harmless (no root — only uninstall-manifest bookkeeping under `/var/log` is skipped; the
actual toolkit files under `--toolkitpath` are unaffected). Verified: `nvcc --version` →
`Cuda compilation tools, release 12.8, V12.8.93`.

Env vars needed **before** running cmake configure (and every rebuild thereafter):

```bash
export PATH=$DP/deps/cuda-12.8/bin:$PATH
export CUDACXX=$DP/deps/cuda-12.8/bin/nvcc
export CUDA_HOME=$DP/deps/cuda-12.8
```

cmake flag: `-DCUDA_TOOLKIT_ROOT_DIR=$DP/deps/cuda-12.8` (the project uses the legacy
`find_package(CUDA 9.0)` in `cmake/TorchExtension.cmake`, not modern `enable_language(CUDA)`).

## Compatibility fix required: `_GLIBCXX_USE_CXX11_ABI` mismatch (build flag, not a source patch)

The first full build + install **succeeded** (cmake exit 0, zero compile errors) but failed at
**runtime**, on `import PlaceDB` in the Step 3 smoke run:

```
ImportError: .../place_io_cpp.cpython-312-x86_64-linux-gnu.so: undefined symbol:
_ZN3c106detail14torchCheckFailEPKcS2_jRKSs
```

This demangles to `c10::detail::torchCheckFail(char const*, char const*, unsigned int,
std::basic_string<...> const&)` using the **old** pre-C++11 libstdc++ ABI mangling for
`std::string` (`Ss`). `$DP/CMakeLists.txt` defaults `CMAKE_CXX_ABI` (i.e.
`_GLIBCXX_USE_CXX11_ABI`) to **0** (old ABI) unless overridden. torch 2.8.0's official PyPI wheel
is built with the **new** ABI: `python -c "import torch; print(torch.compiled_with_cxx11_abi())"`
→ `True`. This is a documented, anticipated gotcha in `$DP/CLAUDE.md` ("`_GLIBCXX_USE_CXX11_ABI`
mismatch with PyTorch causes undefined-symbol link errors at import time — check
`CMAKE_CXX_ABI`.") — it just hadn't been hit yet on this checkout.

**Fix**: add `-DCMAKE_CXX_ABI=1` to the cmake configure command, then a full rebuild + reinstall.
This is a **build parameter**, not a change to any DREAMPlace source file, so **no `io-aware`
branch or commit in `$DP` was needed** — `$DP` remains on `master`, git-state untouched by Task 1.
After this fix, the Step 3 smoke run passed cleanly (see below).

## Full build procedure (final, verified)

```bash
DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace
PY=$DP/.venv312/bin/python
export PATH=$DP/deps/cuda-12.8/bin:$PATH
export CUDACXX=$DP/deps/cuda-12.8/bin/nvcc
export CUDA_HOME=$DP/deps/cuda-12.8

cd $DP && cmake -B build312 \
  -DPython_EXECUTABLE=$PY \
  -DCMAKE_INSTALL_PREFIX=$DP/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_ABI=1 \
  -DBOOST_ROOT=$DP/deps/boost/usr \
  -DBOOST_INCLUDEDIR=$DP/deps/boost/usr/include \
  -DBOOST_LIBRARYDIR=$DP/deps/boost/usr/lib/x86_64-linux-gnu \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=$DP/deps/cuda-12.8

cmake --build build312 -j $(nproc)
cmake --build build312 --target install
```

Confirmed in the configure log: `CMAKE_CXX_ABI: _GLIBCXX_USE_CXX11_ABI=1`, `Found CUDA:
$DP/deps/cuda-12.8 (found suitable version "12.8", minimum required is "9.0")`,
`TORCH_ENABLE_CUDA=1`, `Found Boost: $DP/deps/boost/usr/include (found suitable version
"1.74.0"...)` for both the top-level headers-only check and Limbo's `COMPONENTS graph regex`
check, `CUDA_ARCH_FLAGS` includes `-gencode arch=compute_89,code=sm_89` (the dev GPU, NVIDIA L4 /
Ada). Zero `CMake Error`s in either configure; ~58 benign `CMake Warning`s about RPATH /
`libgomp.so.1` ("Cannot generate a safe runtime search path" — happens whenever a bundled torch
`libgomp` sits alongside the system one; does not block configure or the build). Post-install,
`$DP/install/dreamplace/configure.py` records `"CUDA_FOUND" : "TRUE"`. Old `cpython-38`-tagged
`.so`s from the pre-existing install (see below — an older build than the `.venv`'s own Python
3.9, evidently from a stale `/opt/conda` python 3.8 environment referenced in the leftover
`$DP/build/CMakeCache.txt`) are left in place alongside the new `cpython-312` ones; harmless dead
weight, not selected by a 3.12 interpreter's import machinery.

Timings observed on this host (64 cores):
- First attempt (before the ABI fix): configure ~a few sec, build 2m57s (02:53:55–02:56:52),
  install 2m24s (02:57:34–02:59:58) — cmake-successful but produced a runtime-broken `.so` (see
  above).
- Final attempt (`-DCMAKE_CXX_ABI=1`, clean `rm -rf build312` first): build 2m12s
  (03:03:51–03:06:03), install 1m42s (03:06:14–03:07:56).
- Both are well under the brief's 10–40 minute estimate, thanks to the 64-core host.
- Boost `.deb` fetch: ~2s (11.4MB). CUDA runfile download: well under a minute (5.38GB on this
  network); toolkit install itself: under 2 minutes.

## Smoke run (Step 3)

```bash
cd $DP/install && $PY dreamplace/Placer.py test/simple.json
```

Passed after the ABI fix: exit 0, `results/simple/simple.gp.pl` written (valid `UCLA pl 1.0`
placement, 10 nodes). DREAMPlace's own reported `non-linear placement takes 61.24 seconds`;
total wall time ~100s including Python/torch/CUDA-context/import startup — notably longer than the
brief's "數秒" estimate, most likely first-run CUDA module-loading/JIT overhead across the ~30
separate compiled extension `.so`s (each importing torch and initializing its own CUDA module) on
a freshly built, never-before-run install; not a functional problem (no errors, correct output).
Only benign warnings appeared: Python 3.12 `SyntaxWarning: invalid escape sequence '\s'` from
LaTeX-style backslashes in `dreamplace/ops/dct/discrete_spectral_transform.py`'s **docstrings**
(cosmetic only, Python being stricter about this since 3.12; does not affect execution — left
unpatched as it is not a functional issue and patching docstrings is out of Task 1's scope), plus
DREAMPlace's normal informational warnings for a DEF/Verilog-less Bookshelf-only tiny benchmark.

## `DREAMPLACE_ROOT` and import path

Default: `/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`. Everything runs from
`$DREAMPLACE_ROOT/install` (the source tree has no compiled `.so`s). Two `sys.path` entries are
needed: `$DREAMPLACE_ROOT/install` and `$DREAMPLACE_ROOT/install/dreamplace` (DREAMPlace mixes
`import dreamplace.ops.*` and bare `import Params` / `import PlaceDB` style imports).

## DREAMPlace source-level compatibility patches

**None.** No DREAMPlace source file or git state was touched — no `io-aware` branch was needed.
Both blockers hit in Task 1 (missing Boost dev headers, missing CUDA devel toolkit) were host
*toolchain* gaps, not DREAMPlace/Python/torch/numpy incompatibilities; the one real compatibility
issue found (the C++11 ABI mismatch) is fixed by a **cmake configure flag**
(`-DCMAKE_CXX_ABI=1`), not a source edit. `$DP` remains on branch `master`, matching
`origin/master`, with only its pre-existing untracked `CLAUDE.md` plus the new (untracked, not
part of any repo) `deps/`, `build312/`, `install.cp39.bak/` directories.

## Artifacts under `$DP` (all outside git; nothing here is committed anywhere)

- `$DP/.venv312/` — the Python 3.12 venv (`$PY`).
- `$DP/deps/boost/` — extracted Boost 1.74.0 headers + compiled graph/regex libs.
- `$DP/deps/boost_debs/` — the 5 source `.deb`s, kept for provenance.
- `$DP/deps/cuda-12.8/` — CUDA 12.8.93 toolkit (nvcc, headers, libcudart, etc.), ~8.7GB.
- `$DP/build312/` — cmake build tree (`configure2.log`, `build2.log`, `install2.log` are the final
  successful ones; `configure.log`/`build.log`/`install.log` are the pre-ABI-fix attempt, kept for
  the record).
- `$DP/install.cp39.bak/` — byte-identical backup of the pre-existing `install/`, taken before
  rebuilding (verified via `diff -rq` immediately after copying), in case of fallback.
- `$DP/install/` — rebuilt in place; now contains both the old `cpython-38` `.so`s and the new
  `cpython-312` ones (the latter is what a 3.12 interpreter picks up automatically).
