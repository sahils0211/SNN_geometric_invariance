"""
Run the full Section 4/5 representation analysis (5.1, 5.3, 5.4, 5.7, 5.8)
across ALL THREE SEEDS for all four configurations (analog, full, alternating,
hybrid_late).

This extends the earlier seed-0-only analysis to seeds 1 and 2 as well, so
that the cross-configuration patterns already observed (layer-6/7 collapse,
R_i shape differences, equivariance error ordering) can be checked for
stability across random initialization, not just trusted from a single run.

Output naming convention (deliberately explicit about config AND seed, so
nothing gets overwritten or confused later):
    plots/{config}_seed{N}/{config}_seed{N}_5.1_layerwise_similarity.png
    plots/{config}_seed{N}/{config}_seed{N}_5.3_rotation_sensitivity.png
    plots/{config}_seed{N}/{config}_seed{N}_5.4_rotation_sensitivity_through_time.png
    plots/{config}_seed{N}/{config}_seed{N}_5.7_temporal_evolution.png
    plots/{config}_seed{N}/{config}_seed{N}_5.8_equivariance_error.png

where {config} is one of: analog, full, alternating, hybrid_late
and {N} is 0, 1, or 2.

Analog configs skip 5.4 and 5.7 (no time axis), matching the seed-0 run.
"""

import torch
from representations import load_spiking_model, load_analog_model, run_full_analysis
from train_spiking import load_mnist_csv, split_train_test, DATA_PATH, FOUR_CLASS_DIGITS

images, labels = load_mnist_csv(DATA_PATH, digits=FOUR_CLASS_DIGITS)
(_, _), (test_x, test_y) = split_train_test(images, labels, seed=42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

x = torch.from_numpy(test_x[:150]).unsqueeze(1).to(device)

SEEDS = [0, 1, 2]

# Base config definitions -- checkpoint path pattern per seed, filled in below.
CONFIG_TEMPLATES = [
    {"config": "analog", "kind": "analog",
     "path_fmt": "models/n4_seed{seed}.pt"},
    {"config": "full", "kind": "spiking", "mode": "full",
     "path_fmt": "models/spike_n4_full_T50_seed{seed}.pt"},
    {"config": "alternating", "kind": "spiking", "mode": "alternating",
     "path_fmt": "models/spike_n4_alternating_T50_seed{seed}.pt"},
    {"config": "hybrid_late", "kind": "spiking", "mode": "hybrid_late",
     "path_fmt": "models/spike_n4_hybrid_late_T50_seed{seed}.pt"},
]

# Build the full (config, seed) job list, skipping seed 0 by default since
# that was already run in the earlier pass -- set SKIP_SEED0 = False if you
# want to regenerate seed 0's plots too (e.g. to confirm reproducibility of
# the naming/script itself), otherwise this only fills in the new seeds.
SKIP_SEED0 = True

jobs = []
for tmpl in CONFIG_TEMPLATES:
    for seed in SEEDS:
        if SKIP_SEED0 and seed == 0:
            continue
        job = dict(tmpl)
        job["seed"] = seed
        job["path"] = tmpl["path_fmt"].format(seed=seed)
        jobs.append(job)

print(f"Planned jobs: {len(jobs)}")
for j in jobs:
    print(f"  {j['config']}_seed{j['seed']}  <-  {j['path']}")

for job in jobs:
    label = f"{job['config']}_seed{job['seed']}"
    print(f"\n{'='*70}\nProcessing: {label}\n{'='*70}")
    try:
        if job["kind"] == "analog":
            model = load_analog_model(job["path"], n_classes=4, channels=10, device=device)
            run_full_analysis(
                model, x, test_x, test_y,
                plot_dir=f"plots/{label}", label=label, is_spiking=False,
            )
        else:
            model = load_spiking_model(job["path"], n_classes=4, channels=10, T=50,
                                        mode=job["mode"], device=device)
            run_full_analysis(
                model, x, test_x, test_y,
                plot_dir=f"plots/{label}", label=label, is_spiking=True,
            )
    except FileNotFoundError:
        print(f"  [skipped] checkpoint not found: {job['path']}")
        continue

print("\nAll (config, seed) jobs processed.")