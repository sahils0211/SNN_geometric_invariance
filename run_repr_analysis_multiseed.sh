#!/usr/bin/env bash
# run_repr_analysis_multiseed.sh
#
# Closes the multi-seed gap: runs the full representation-analysis suite
# (rotation sensitivity incl. --temporal, equivariance error, representation
# similarity, spike accumulation, angle-averaged AND angle-resolved
# truncated-T' accuracy) for every seed in SEEDS, against one checkpoint set
# (MODELS_DIR/T), writing each seed's plots to its own OUT_ROOT/repr_seed<seed>/
# subdirectory -- matching the original findings report's per-seed figure
# convention. Seed 0 already had a partial run (missing --temporal and the
# angle-resolved truncated-T plot); this script re-runs seed 0 too so its
# directory ends up with the complete set alongside seeds 1 and 2.
set -e

SEEDS="${SEEDS:-0 1 2}"
N_PER_CLASS="${N_PER_CLASS:-4}"
T="${T:-50}"
MODELS_DIR="${MODELS_DIR:-models_small}"
OUT_ROOT="${OUT_ROOT:-plots_small}"
MODES=(none alternating full hybrid_early hybrid_late)

for seed in $SEEDS; do
  OUT_DIR="$OUT_ROOT/repr_seed${seed}"
  mkdir -p "$OUT_DIR"
  echo "=================================================================="
  echo " seed=$seed  T=$T  models_dir=$MODELS_DIR  out_dir=$OUT_DIR"
  echo "=================================================================="

  echo "--- [1/6] Rotation sensitivity (R_i), incl. --temporal (R_i(t)) ---"
  python3 plot_rotation_sensitivity.py --modes "${MODES[@]}" --seed "$seed" \
    --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR" --temporal

  echo ""
  echo "--- [2/6] Direct equivariance error ---"
  python3 plot_equivariance_error.py --modes "${MODES[@]}" --seed "$seed" \
    --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

  echo ""
  echo "--- [3/6] Representation similarity (cosine / CKA / SVCCA) ---"
  python3 plot_representation_similarity.py --modes "${MODES[@]}" --seed "$seed" \
    --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

  echo ""
  echo "--- [4/6] Accumulated spike counts a_i(x) ---"
  python3 plot_spike_accumulation.py --modes "${MODES[@]}" --seed "$seed" \
    --n_per_class "$N_PER_CLASS" --T "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

  echo ""
  echo "--- [5/6] Truncated-T' accuracy (angle-averaged) ---"
  python3 plot_truncated_T.py --modes "${MODES[@]}" --seed "$seed" \
    --n_per_class 6 --T_full "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

  echo ""
  echo "--- [6/6] Truncated-T' accuracy (angle-resolved, matches original report style) ---"
  python3 plot_truncated_T_by_angle.py --modes "${MODES[@]}" --seed "$seed" \
    --n_per_class 6 --T_full "$T" --models_dir "$MODELS_DIR" --out_dir "$OUT_DIR"

  echo ""
  echo "=== seed=$seed done. See: $OUT_DIR ==="
done

echo ""
echo "=== ALL SEEDS COMPLETE for T=$T / $MODELS_DIR ==="
