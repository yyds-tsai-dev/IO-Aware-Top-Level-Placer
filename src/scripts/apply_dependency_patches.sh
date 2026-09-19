#!/usr/bin/env bash
# Run after git submodule update --init --recursive. Safe to repeat.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)"
if [[ "${repo##*/}" == "src" ]]; then
    repo="$(dirname -- "$repo")"
fi
for entry in 'DREAMPlace dreamplace-runtime.patch' 'OpenROAD openroad-runtime.patch'; do
    read -r name patch <<< "$entry"
    target="$repo/third_party/$name"
    patch_path="$repo/third_party/patches/$patch"
    if git -C "$target" apply --reverse --check "$patch_path" 2>/dev/null; then
        printf '%s already applied\n' "$name"
    else
        git -C "$target" apply --check "$patch_path"
        git -C "$target" apply "$patch_path"
        printf '%s patches applied\n' "$name"
    fi
done
