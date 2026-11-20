#!/usr/bin/env bash
# Runs once after the container is created (as the `vscode` user).
set -euo pipefail

# Named volumes are created root-owned
sudo chown vscode: .venv .tox ~/.claude ~/.cache ~/.local/share/nvim

# Editable dev install, as in CLAUDE.md, plus the GPU extras of the
# `py314-cupy` tox env. Re-running this script is safe.
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.14 .venv
fi
uv pip install --python .venv/bin/python \
    -e ".[gpu,gui]" -r test_requirements.txt "numba>=0.63" \
    cupy-cuda13x "cuda-toolkit[all]<14"

# .git is shared with the host: don't overwrite a hook installed there
if [ ! -e "$(git rev-parse --git-path hooks/pre-commit)" ]; then
    pre-commit install
fi

if command -v nvidia-smi >/dev/null; then
    nvidia-smi -L
else
    echo "No GPU visible in the container; cupy tests will skip." >&2
fi
