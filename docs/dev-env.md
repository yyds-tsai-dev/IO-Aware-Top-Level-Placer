# Development Environment — H100 / ldaphome

This is the current environment reference for both Claude and Codex agents,
updated on 2026-09-06. The previous NVIDIA L4 build record is archived in
[dev-env-nvl4.md](dev-env-nvl4.md). Old experiment reports retain their original
paths and hardware provenance.

## Run project commands

From the repository root:

```bash
source src/scripts/env.sh
"$IOPLACE_PYTHON" -m pytest -m "not slow"
```

The script derives defaults from the repository location:

| Variable | Current default |
| --- | --- |
| `IOPLACE_REPO` | `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer` |
| `DREAMPLACE_ROOT` | `/ldaphome/yyds-tsai-dev/DREAMPlace` |
| `IOPLACE_PYTHON` | `/ldaphome/yyds-tsai-dev/DREAMPlace/.venv312/bin/python` |
| `CUDA_HOME` | `/ldaphome/yyds-tsai-dev/DREAMPlace/deps/cuda-12.8` |

Set `DREAMPLACE_ROOT` or `IOPLACE_PYTHON` before sourcing to override them.
The script adds `src/` to `PYTHONPATH` and local CUDA/Bison/Flex tools
to `PATH`; use `"$IOPLACE_PYTHON"` explicitly for Python. The code's
`ioplace.dreamplace_env.DEFAULT_ROOT` also resolves to the sibling DREAMPlace
directory, while explicit `setup_dreamplace(root=...)` and environment
overrides remain supported.

Some historical tests and diagnostic scripts still contain NVL4 fallbacks or
absolute benchmark paths. Exporting `DREAMPLACE_ROOT` covers environment-aware
callers; inspect old standalone experiment scripts before rerunning them.
External benchmark corpora and old experiment data are separate from the
DREAMPlace installation.

The host is shared. Check `nvidia-smi` and select an available GPU with
`CUDA_VISIBLE_DEVICES`; the script preserves the caller's selection.
`-m "not slow"` excludes placement integration tests but still includes GPU
unit tests. Report missing benchmark data or unavailable GPUs explicitly.

### Normalisation module (P-H) flags

`ioplace.drivers.run_placement --mode io` normalises every extra objective term
through `ioplace.norm.TermNormalizer` (v2 design section 4):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--norm-policy` | `legacy` | `legacy` reproduces the retired λ_IO EMA + κ_FT force share exactly; `grandplan` is `λ_t = wt_t·‖∇WL‖_p/‖∇T_t‖_p`; `adaptive` targets a per-term force share |
| `--norm-p` | `1` | Gradient norm order (L1 default, L2 switch) |
| `--norm-ramp-period` | `100` | Policy `grandplan`: iterations between `+0.05` steps of `wt`, from `0.05` |
| `--norm-wt-max` | `1.0` | Policy `grandplan`: upper bound on `wt` |
| `--norm-probe-every` | `50` | Iterations between probes; must be a positive multiple of `--every` |
| `--norm-target-share` | unset | Policy `adaptive`: `io=0.3,ft=0.1`. Unnamed terms fall back to `io=0.3` and `ft=--f-ft-max`; names must match registered terms (`io`, and `ft` when `--f-ft-max > 0`), values must be finite and in `[0, 1]`, duplicates are rejected |
| `--norm-trace` | unset | Trace path; defaults to `<out>.norm_trace.jsonl` for non-legacy policies |

Non-legacy policies require `--callback-order atomic`. They also reject
`--rho-max 0` and `--rho-margin > 0`: both only reach the coefficients through
`ScheduleState`, which these policies bypass, so `--rho-max 0` (a reweight-only
run under `legacy`) would silently turn the IO penalty on at full normalizer
strength and `--rho-margin` would be dropped while still being echoed into the
result JSON. Use `--norm-policy legacy` for either.

Under `grandplan`, `ft`'s weight ceiling is `--f-ft-max x --norm-wt-max`, so the
FT force share mirrors the retired `f_ft_max` instead of converging to IO's
ceiling; `ft` is also registered as a *dependent* term (`requires="io"`), so its
coefficient is published as 0 whenever λ_IO is 0 -- `FtTerm` can only express
the FT force as `λ_IO·κ`, and one transient IO-gradient dip used to abort the
whole run. The recovered `κ = λ_FT/λ_IO` is clamped at 100, mirroring
`ScheduleState.kappa_max`.

Coefficient timing on the non-legacy arms: `transaction()` commits an *un-ramped*
λ every `--every` iterations, and the objective term multiplies it by
`activation_ramp(iteration, it_activate, n_ramp)` on every GP iteration, so λ
drifts continuously through the activation window exactly as the retired
`ρ·ramp·ratio_ema` did (no `obj_version` bump for that drift -- it is a known
monotone function of the iteration counter, not a new measurement). `it_activate`
comes from `ScheduleState`'s per-iteration activation, not from the first
`--every`-gated callback, so both arms ramp off the same instant. The trace
records both: `terms.<t>.lam` (committed) and `terms.<t>.lam_applied` (ramped);
the result trajectory adds `lambda_io_applied`/`lambda_ft_applied`.
There is no relative deadness threshold on the non-legacy arm: every probe
measures `ratio_inst = ‖∇WL‖/‖∇T‖` and updates the EMA, and λ is
`min(policy(ratio_ema), the term's share of the Lipschitz cap)` — the retired
path's own `min(base, cap)` shape. Only a gradient of exactly 0 gives λ = 0 and
skips the EMA. (The earlier `eps_rel·‖∇WL‖` test was removed after it was
measured to track WL's growth rather than the term: on `mempool_group` it
misclassified a flat IO gradient as dead for the last quarter of GP.) `norm_trace.jsonl` gets
one row per coefficient transaction (>= one per probe, since a transaction may
reuse the previous probe's measurements), with the per-term gradient norm,
instantaneous and EMA ratio, weight, coefficient, realised force share,
`cmax` (the pre-clip λ-weighted mean curvature over the active terms --
controller ruling 2026-09-19; there is no separate κ field to log),
`cap_binding`, `cancellation_ratio` and the objective/refresh versions. The
result JSON's own `trajectory` entries differ by policy too: only
`--norm-policy legacy` runs carry the eight `ops/ft_callback.publish_atomic`
keys `grad_l1_merged`, `f_ft`, `kappa_ft`, `kappa_clamped`, `Cmax`,
`f_effective`, `applied_ft_force_l1` and `home_version` -- `grandplan`/`adaptive`
runs get the normalizer's own `grad_l1_wl`/`grad_l1_io`/`grad_l1_ft`/`ratio_inst`/
`ratio_ema`/`lambda_io`/`obj_version` fields instead, with none of the eight
legacy-only keys present.

## Installed toolchain

| Component | Version / configuration |
| --- | --- |
| GPU | H100 NVL, compute capability 9.0; four devices visible |
| NVIDIA driver | 580.173.02 |
| Python | 3.12.13, dedicated `$DREAMPLACE_ROOT/.venv312` |
| PyTorch | 2.8.0+cu128, `_GLIBCXX_USE_CXX11_ABI=1` |
| NumPy | 1.26.4, retaining `np.string_` compatibility |
| Shapely | 2.1.2, with the project's empty-polygon patch |
| pytest / Mt-KaHyPar | 9.1.1 / 1.6.2 |
| CUDA toolkit | 12.8.1, `nvcc` 12.8.93, under `$DREAMPLACE_ROOT/deps` |
| Compiler / CMake | GCC 11.4.0 / system CMake 3.22 |
| Boost | System 1.74 headers, graph and regex libraries |
| Bison / Flex | 3.8.2 / 2.6.4, under `$DREAMPLACE_ROOT/deps/tools` |

The CUDA version displayed by `nvidia-smi` is the driver's capability; the
compiler and PyTorch used here are CUDA 12.8. Python dependencies are recorded
in `$DREAMPLACE_ROOT/logs/requirements-h100.txt`. Read the Mt-KaHyPar version
with `importlib.metadata.version("mtkahypar")`; its module version may say `dev`.

## Source and project patches

[Upstream DREAMPlace](https://github.com/limbo018/DREAMPlace) was cloned with
recursive submodules at commit `6627f3327e6cc17db7782c0b90073a498531ca3c`.
These repository patches are applied in order before installing Python modules:

1. `src/ioplace/dp_patch/iteration-callback.patch`
2. `src/ioplace/dp_patch/m2-extra-obj-terms.patch`
3. `src/ioplace/dp_patch/shapely2-compat.patch`

They provide the iteration callback, per-instance IO objectives and exposed
optimizer/model, and Shapely 2 empty-polygon handling. Apply them with
`git -C "$DREAMPLACE_ROOT" apply` on a fresh checkout. On an already patched
installation, `git apply --reverse --check` verifies their presence; do not
apply them twice.

## Build and reinstall

The venv uses upstream `requirements.txt` plus pinned `torch==2.8.0`,
`numpy==1.26.4`, `pytest==9.1.1`, and `mtkahypar==1.6.2`, with `pyyaml` and
`pandas` for project tooling. It was created with `uv venv --python 3.12` and
populated with `uv pip install --python "$IOPLACE_PYTHON"`.

CUDA was installed using NVIDIA's `cuda_12.8.1_570.124.06_linux.run`, with
`--silent --toolkit --toolkitpath="$DREAMPLACE_ROOT/deps/cuda-12.8" --no-man-page`.
Only the toolkit was requested; the host driver was retained. Bison, Flex,
libfl-dev and libfl2 were downloaded with `apt-get download` and extracted
with `dpkg-deb -x` into `deps/tools`. `src/scripts/env.sh` sets the local
`BISON_PKGDATADIR`. The build uses system Boost/Cairo/zlib development libraries.

```bash
source src/scripts/env.sh
cmake -S "$DREAMPLACE_ROOT" -B "$DREAMPLACE_ROOT/build312" \
  -DPython_EXECUTABLE="$IOPLACE_PYTHON" \
  -DCMAKE_INSTALL_PREFIX="$DREAMPLACE_ROOT/install" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_ABI=1 \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=9.0 \
  -DFLEX_INCLUDE_DIR="$DREAMPLACE_ROOT/deps/tools/usr/include" \
  -DFL_LIBRARY="$DREAMPLACE_ROOT/deps/tools/usr/lib/x86_64-linux-gnu/libfl.so"
cmake --build "$DREAMPLACE_ROOT/build312" --parallel 16
cmake --install "$DREAMPLACE_ROOT/build312"
```

This upstream CMake also adds `sm_120` for CUDA 12.8; the generated build
includes `sm_90` for H100. Upstream downloads its HeteroSTA release into the
build directory during configuration.

Runtime imports use `$DREAMPLACE_ROOT/install` and its `dreamplace` child;
`setup_dreamplace()` adds both. Relative benchmark configs run with `install`
as the working directory. The bundled 10-node `simple` input was copied from
upstream's `unittest/ops/place_io_unittest/simple` into `benchmarks/simple`.
Upstream's CMake does not install `test/simple.json`; copy it to
`install/test/simple.json` when setting up a fresh installation.

The [ISPD2005 archive used by upstream](https://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz)
was extracted into `$DREAMPLACE_ROOT/benchmarks/ispd2005`. The runtime data
links are `install/benchmarks/simple -> ../../benchmarks/simple` and
`install/benchmarks/ispd2005 -> ../../benchmarks/ispd2005`. Other benchmark
corpora have not been downloaded as part of this setup.

## Validation and logs

Verified on 2026-09-06 with `CUDA_VISIBLE_DEVICES=3` for bounded GPU checks:

| Check | Result | Log under `$DREAMPLACE_ROOT/logs/` |
| --- | --- | --- |
| CUDA compilation and installation | Both commands exited 0 | `build-h100.log`, `install-h100.log` |
| GPU evaluator + Mt-KaHyPar | 28 passed, 4 slow tests deselected | `pytest-gpu-h100.log` |
| Objective hooks, netlist and driver unit tests | 14 passed, 6 slow tests deselected | `pytest-native-h100.log` |
| Native loading of simple and adaptec1 configs | 2 passed | `pytest-inputs-h100.log` |
| H100 simple placement + legalization | Exit 0, 1,000 GP iterations, 10-node placement output | `smoke-simple-h100.log` |

The 44 tests above are targeted environment checks, not a full-suite run.
The native-import run emitted 10 upstream Python docstring `SyntaxWarning`s;
these did not fail the tests. The smoke config is `logs/smoke-simple-h100.json`
(upstream `simple.json` with four CPU threads, plotting disabled, and a log
output directory); its result is `logs/smoke-results/simple/simple.gp.pl`.

Exact pytest selections, after sourcing `src/scripts/env.sh`:

```bash
CUDA_VISIBLE_DEVICES=3 "$IOPLACE_PYTHON" -m pytest -q \
  tests/test_hgr.py tests/test_evaluator_gpu.py -m "not slow"
CUDA_VISIBLE_DEVICES=3 "$IOPLACE_PYTHON" -m pytest -q \
  tests/test_dp_hook.py tests/test_netlist.py tests/test_driver.py -m "not slow"
CUDA_VISIBLE_DEVICES=3 "$IOPLACE_PYTHON" -m pytest -q \
  tests/test_netlist.py::test_load_netlist_simple \
  tests/test_driver.py::test_load_dreamplace_forces_dp_off
```

Also verified: both Codex TOML files parse with their required fields;
`src/scripts/env.sh` passes `bash -n`, derives the H100 paths, and preserves
explicit Python/root/GPU overrides. Logs for dependency setup are
`cuda-install.log`, `configure-h100.log`, and `requirements-h100.txt`.

## Repository submodules (2026-09-14)

DREAMPlace and the actual OpenROAD ISPD25 fork are pinned under `third_party/`.
Initialize with `git submodule update --init --recursive`, then run
`bash src/scripts/apply_dependency_patches.sh` before building. The complete runtime
patches, exact commits and SHA-256 values are recorded in
[`third_party/versions.json`](../third_party/versions.json); setup and patch
semantics are in [`third_party/README.md`](../third_party/README.md).

`src/scripts/env.sh` and `setup_dreamplace()` prefer the submodule when it has an
installed `dreamplace/PlaceDB.py`; otherwise they retain the sibling installation.
Explicit `DREAMPLACE_ROOT` still takes priority. `DREAMPLACE_SOURCE` and
`OPENROAD_SOURCE` identify pinned source directories independently of the current
runtime builds. No existing external installation was moved or overwritten.
