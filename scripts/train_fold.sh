#!/usr/bin/env bash
# Train one fold of the four-fold recipe on a CUDA GPU:
#     DATA_ROOT=... scripts/train_fold.sh K            K = 0, 1, 2 or 3
# Quick check of the same code path on CPU with real data (16 training plays, 2 short epochs):
#     DATA_ROOT=... scripts/train_fold.sh K --smoke
# Outputs: OUT_ROOT/train/foldK (or OUT_ROOT/smoke/foldK): per-epoch checkpoints, history.csv, run.json.
set -euo pipefail
FOLD=${1:?usage: scripts/train_fold.sh FOLD [--smoke]}
MODE=${2:-}
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export NVIDIA_TF32_OVERRIDE=0          # full float32 matrix products, as in the recorded runs
PYTHON=${PYTHON:-python}
if [[ "$MODE" == "--smoke" ]]; then
    "$PYTHON" -m src.train --fold "$FOLD" --smoke --device cpu --out "${OUT_ROOT:-outputs}/smoke/fold$FOLD"
else
    "$PYTHON" -m src.train --fold "$FOLD" --device cuda
fi
