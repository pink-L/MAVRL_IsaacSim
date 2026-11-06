#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PROJECT_ROOT/.runtime/cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.runtime/config"
export XDG_DATA_HOME="$PROJECT_ROOT/.runtime/data"
export CUDA_CACHE_PATH="$PROJECT_ROOT/.runtime/cuda"
export MPLCONFIGDIR="$PROJECT_ROOT/.runtime/matplotlib"
export PIP_CACHE_DIR="$PROJECT_ROOT/.runtime/pip"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH"
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/python" "$@"
