# Source after scripts/env.sh to use the locally built OpenROAD fork.
# This file only sets user-space paths; it does not alter the system install.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source scripts/openroad_env.sh' >&2
    exit 1
fi

export OPENROAD_ROOT="${OPENROAD_ROOT:-/ldaphome/yyds-tsai-dev/tools/openroad}"
export OPENROAD_PREFIX="${OPENROAD_PREFIX:-$OPENROAD_ROOT/prefix-upstream-grt-uint64}"
export OPENROAD_BIN="${OPENROAD_BIN:-$OPENROAD_PREFIX/bin/openroad}"

_ioplace_openroad_libs=(
    "$OPENROAD_ROOT/deps/ortools/lib64"
    "$OPENROAD_ROOT/deps/ortools/lib"
    "$OPENROAD_ROOT/deps/boost-prefix/lib"
    "$OPENROAD_ROOT/debs/spdlog-root/usr/lib/x86_64-linux-gnu"
    "$OPENROAD_ROOT/debs/fmt-root/usr/lib/x86_64-linux-gnu"
)
_ioplace_openroad_ld=""
for _ioplace_openroad_lib in "${_ioplace_openroad_libs[@]}"; do
    if [[ -z "$_ioplace_openroad_ld" ]]; then
        _ioplace_openroad_ld="$_ioplace_openroad_lib"
    else
        _ioplace_openroad_ld="$_ioplace_openroad_ld:$_ioplace_openroad_lib"
    fi
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _ioplace_openroad_ld="$_ioplace_openroad_ld:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$_ioplace_openroad_ld"
unset _ioplace_openroad_libs _ioplace_openroad_lib _ioplace_openroad_ld
