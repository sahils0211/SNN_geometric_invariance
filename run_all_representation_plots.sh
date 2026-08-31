#!/usr/bin/env bash
# run_all_representation_plots.sh
#
# Generates every notation-based plot (sec:notation of the findings report)
# for the reduced-capacity (5-layer, 4-channel, 2,420-param) model, across
# all 5 spiking configs:
#
#   1. Rotation sensitivity   R_i = Var_theta(a_i(R_theta x))   [Eq. Ri]
#   2. Direct equivariance error(theta) = ||f(R_theta x)-f(x)||/||f(x)|| [Eq. eqerr]
#   3. Representation similarity: cosine / CKA / SVCCA, layer-wise      [sec:simmetrics]
#   4. Accumulated spike counts a_i(x), histograms + example heatmaps   [Eq. accumcount]
#   5. Truncated-T' accuracy (T' in 1,5,10,20,50, no retraining)        [sec:truncT]
#
# Requires: the model checkpoints already saved by train_spiking_small.py's
# training runs (models_small/<mode>_seed<seed>_small.pt), e.g. from GPU-0's
# T=50/10-epoch sweep. Run from the ~/snn directory, with representations_small.py,
# plot_rotation_sensitivity.py, plot_equivariance_error.py,
# plot_representation_similarity.py, plot_spike_accumulation.py, and
# plot_truncated_T.py all present alongside gconv_small.py / p4cnn_small.py /
# spiking_p4cnn_small.py.
#
# Usage:
#   chmod +x run_all_representation_plots.sh
#   ./run_all_representation_plots.sh                 # defaults: seed 0, models_small/
#   MODELS_DIR=models_small_t30 SEED=0 ./run_all_representation_plots.sh   # T=30 checkpoints
#
# Run this INSIDE a tmux window (see earlier setup) if it might take a while --
# the similarity/rotation-sensitivity metrics do 24 rotated forward passes per
# config, and forward_with_activations does a full T-step spiking simulation
# each time, so this is slower than a single accuracy eval.

set -e

SEED="${SEED:-0}"
N_PER_CLASS="${N_PER_CLASS:-4}"
T="${T:-50}"
MODELS_DIR="${MODELS_DIR:-models_small}"
OUT_DIR="${OUT_DIR:-plots_small/repr}"
MODES=(none alternating full hybrid_early hybrid_late)

echo "=================================================================="
echo " Notation-based representation analysis plots (reduced model)"
echo " seed=$SEED  T=$T  n_per_class=$N_PER_CLASS"
echo " models_dir=$MODELS_DIR  out_dir=$OUT_DIR"
echo "=================================================================="

mkdir -p "$OUT_DIR"

echo ""
echo "--- [1/5] Rotation sensitivity (R_i) ---"
python3 plot_rotation_sensitivity.py --modes "${MODES[@]}" --seed "$SEED" \
  --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

echo ""
echo "--- [2/5] Direct equivariance error ---"
python3 plot_equivariance_error.py --modes "${MODES[@]}" --seed "$SEED" \
  --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

echo ""
echo "--- [3/5] Representation similarity (cosine / CKA / SVCCA) ---"
python3 plot_representation_similarity.py --modes "${MODES[@]}" --seed "$SEED" \
  --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

echo ""
echo "--- [4/5] Accumulated spike counts a_i(x) ---"
python3 plot_spike_accumulation.py --modes "${MODES[@]}" --seed "$SEED" \
  --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

echo ""
echo "--- [5/5] Truncated-T' accuracy ---"
python3 plot_truncated_T.py --modes "${MODES[@]}" --seed "$SEED" \
  --n_per_class 6 --T_full "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

echo ""
echo "All representation-analysis plots done. See: $OUT_DIR"
ls -la "$OUT_DIR"
