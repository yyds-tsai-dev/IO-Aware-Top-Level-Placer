# Source this file from Bash: source /path/to/IO-Aware-Top-Level-Placer/scripts/env.sh
# Keep host paths in one place; callers can override the installation or Python.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source scripts/env.sh' >&2
    exit 1
fi

export IOPLACE_REPO="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)"
if [[ "${IOPLACE_REPO##*/}" == "src" ]]; then
    IOPLACE_REPO="$(dirname -- "$IOPLACE_REPO")"
fi
if [[ -f "$IOPLACE_REPO/third_party/DREAMPlace/install/dreamplace/PlaceDB.py" ]]; then
    _ioplace_dp_default="$IOPLACE_REPO/third_party/DREAMPlace"
else
    _ioplace_dp_default="$(dirname -- "$IOPLACE_REPO")/DREAMPlace"
fi
export DREAMPLACE_SOURCE="${DREAMPLACE_SOURCE:-$IOPLACE_REPO/third_party/DREAMPlace}"
export DREAMPLACE_ROOT="${DREAMPLACE_ROOT:-$_ioplace_dp_default}"
unset _ioplace_dp_default
export IOPLACE_PYTHON="${IOPLACE_PYTHON:-$DREAMPLACE_ROOT/.venv312/bin/python}"
export PYTHONPATH="$IOPLACE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "$DREAMPLACE_ROOT/deps/cuda-12.8/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_HOME:-$DREAMPLACE_ROOT/deps/cuda-12.8}"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -x "$DREAMPLACE_ROOT/deps/tools/usr/bin/bison" ]]; then
    export BISON_PKGDATADIR="$DREAMPLACE_ROOT/deps/tools/usr/share/bison"
    export PATH="$DREAMPLACE_ROOT/deps/tools/usr/bin:$PATH"
fi

# CUDA_VISIBLE_DEVICES is intentionally supplied by the caller on shared hosts.

# mt-kahypar 1.6.2 segfaults in parallel coarsening (see partition/mtkahypar_runtime.py);
# default to the serial mitigation unless the caller chose a thread count.
export IOPLACE_MTKAHYPAR_THREADS="${IOPLACE_MTKAHYPAR_THREADS:-1}"
