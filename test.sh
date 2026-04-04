#!/usr/bin/env bash
set -euo pipefail
unset CONDA_PREFIX
export VIRTUAL_ENV=/Users/andrewpullin/personal/executorch-work/u55-pipeline/.venv
export PATH="/Users/andrewpullin/.cargo/bin:$VIRTUAL_ENV/bin:/Users/andrewpullin/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "=== Test 1: Basic import ==="
python -c "import retab; print('Import OK:', dir(retab))"

echo ""
echo "=== Test 2: Ingestion test ==="
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
python tests/test_ingest.py

echo ""
echo "=== Test 3: E2E test ==="
python tests/test_e2e.py
