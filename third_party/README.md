# Pinned placement and routing sources

| Submodule | Upstream | Pinned commit |
| --- | --- | --- |
| DREAMPlace | limbo018/DREAMPlace | `6627f3327e6cc17db7782c0b90073a498531ca3c` |
| OpenROAD | liangrj2014/OpenROAD_ISPD25 | `1a29dadda2c06e2cb2f5b09b8f7425778dd88b47` |

```bash
git submodule update --init --recursive
bash src/scripts/apply_dependency_patches.sh
```

The gitlinks pin upstream commits available from their public remotes. The
versioned `patches/*-runtime.patch` files capture the complete tracked source
diffs of the installations used on 2026-09-14. Apply these before building;
patched submodules correctly show as modified in `git status`. The script is
idempotent and refuses conflicting local edits.

DREAMPlace includes the IO callback/objective and Shapely 2 compatibility
patches, equivalent to the three patches in `src/ioplace/dp_patch`. Apply either
the aggregate patch here or those three, never both. OpenROAD includes GRT
restoration to upstream base `c90cd95a8723efb4de67cd8c6efc69f715fc9052`, the readline
portability guard, and the uint64 SWIG typemap. The unpatched contest fork is
an exporter and cannot reproduce normal global routing.

See [the environment guide](../docs/dev-env.md) for DREAMPlace build settings,
and [input/tool provenance](../docs/handover/2026-09-06-input-inventory.md)
for existing OpenROAD installations. Source pins do not contain compiled
binaries, Python environments, toolchains, or benchmark corpora. Existing
external builds remain usable through `DREAMPLACE_ROOT` and `OPENROAD_BIN`;
`source src/scripts/env.sh` prefers a built submodule install and otherwise uses
the existing sibling DREAMPlace. `OPENROAD_SOURCE` identifies the pinned source
independently from the external runtime installation.
