# IO-Aware Top-Level Placer

Implementation source lives under `src/`:

- `src/ioplace/`: placement, partitioning, evaluators, native helpers, and OpenROAD scripts.
- `src/scripts/`: benchmark runners, environment setup, and maintenance commands.
- `tests/`: tests; `benchmarks/`: input configurations; `results/`: experiment outputs and historical source snapshots.
- `third_party/`: pinned DREAMPlace and OpenROAD submodules and runtime patches.

Use `src/ioplace/` and `src/scripts/` for source paths. The former root-level directories and compatibility symlinks have been removed.

From the repository root:

```bash
source src/scripts/env.sh
source src/scripts/openroad_env.sh
uv pip install --python "$IOPLACE_PYTHON" --no-deps --no-build-isolation -e .
"$IOPLACE_PYTHON" -m ioplace.drivers.run_placement --help
"$IOPLACE_PYTHON" src/scripts/run_io_tradeoff.py --help
"$IOPLACE_PYTHON" -m pytest -q -m 'not slow and not gpu'
```

`env.sh` adds `src/` to `PYTHONPATH`. Editable installation also makes `ioplace` importable from other working directories. Configure the native runtime and benchmark data separately; see [development setup](docs/dev-env.md) and [dependency setup](third_party/README.md).
