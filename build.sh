#!/usr/bin/env bash
set -euo pipefail
unset CONDA_PREFIX
export VIRTUAL_ENV=/Users/andrewpullin/personal/executorch-work/u55-pipeline/.venv
export PATH="/Users/andrewpullin/.cargo/bin:$VIRTUAL_ENV/bin:/Users/andrewpullin/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
maturin develop --release
