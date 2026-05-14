#!/usr/bin/env bash
# ============================================================================
# smoke_test.sh — End-to-end smoke test
#
# Runs a demo experiment (2 iterations) to verify the environment is working.
#
# Usage:
#   bash scripts/dev/smoke_test.sh [task_name]
#
# Examples:
#   bash scripts/dev/smoke_test.sh                    # default: mnist_classification
#   bash scripts/dev/smoke_test.sh assist2009_kt      # specific task
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${1:-mnist_classification}"
GPUS="${FARBENCH_GPUS:-0}"
CUDA="${FARBENCH_CUDA:-cu118}"
export FARBENCH_GPUS="$GPUS"
export FARBENCH_CUDA="$CUDA"

echo "==> Smoke test: ${TASK}"
echo "==> GPUs: ${FARBENCH_GPUS}, CUDA: ${FARBENCH_CUDA}"
echo ""

echo "[1/2] Preparing task and running demo..."
bash scripts/run.sh \
    --task "$TASK" \
    --mode demo \
    --agent-id smoke_test \
    --gpus "$FARBENCH_GPUS" \
    --cuda "$FARBENCH_CUDA"

echo ""
echo "[2/2] Done."
echo "==> Smoke test PASSED"
