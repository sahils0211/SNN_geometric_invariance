#!/usr/bin/env bash
# Chained driver: run representation-analysis plots for the GPU-0/T=50
# checkpoints first; only if that succeeds, run again for the GPU-1/T=30
# checkpoints into a separate output dir. Errors are NOT swallowed --
# `set -e` means this script stops (and reports failure) the moment either
# run_all_representation_plots.sh invocation fails.
set -e
cd /home/sahilsingh/snn
source .venv/bin/activate

echo "=================================================================="
echo " [1/2] GPU-0 / T=50 checkpoints (models_small/) -> plots_small/repr/"
echo "=================================================================="
./run_all_representation_plots.sh

echo ""
echo "=================================================================="
echo " [1/2] DONE without errors. Starting [2/2] GPU-1 / T=30 checkpoints"
echo "=================================================================="
MODELS_DIR=models_small_t30 T=30 OUT_DIR=plots_small_t30/repr ./run_all_representation_plots.sh

echo ""
echo "=== BOTH REPR ANALYSIS RUNS COMPLETE ==="
