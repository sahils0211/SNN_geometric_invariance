#!/usr/bin/env bash
# Runs all 5 modes x 3 seeds for the reduced-capacity (2,420-param) ablation,
# sequentially on GPU 0, using the same protocol as train_spiking.py
# (epochs=20, T=50, batch_size=64, lr=1e-3 -- all script defaults).
set -e
cd /home/sahilsingh/snn
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export OUT_SUFFIX=_t30

for mode in none alternating full hybrid_early hybrid_late; do
  for seed in 0 1 2; do
    echo "=== mode=${mode} seed=${seed} ==="
    python train_spiking_small.py --mode "${mode}" --seed "${seed}" --epochs 5 --T 30
  done
done

echo "=== all 15 runs done, generating plots ==="
for mode in none alternating full hybrid_early hybrid_late; do
  python train_spiking_small.py --aggregate --mode "${mode}"
done
python train_spiking_small.py --compare --modes none alternating full hybrid_early hybrid_late

echo "=== SWEEP COMPLETE ==="
^X


