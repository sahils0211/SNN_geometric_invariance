"""
plot_representation_similarity.py

Plots layer-wise representation similarity between h_ell(x) and
h_ell(R_theta x) using the three measures defined in sec:simmetrics:
cosine similarity, CKA (batch-side Gram formulation), and SVCCA.

High similarity at layer ell indicates that layer has learned a
rotation-invariant representation; the central question (sec:notation,
"why this matters") is whether similarity rises as ell increases.

Produces:
  plots_small/repr/similarity_by_layer_<mode>.png
      -- for one config: cosine / CKA / SVCCA vs layer, 3 lines on one axis
  plots_small/repr/similarity_comparison_<metric>.png
      -- for each metric (cosine, cka, svcca) separately: one line per
         config, all 5 configs overlaid, vs layer -- directly comparable
         to the accuracy-comparison and rotsens-comparison plots.

Usage:
    python3 plot_representation_similarity.py --modes none alternating full hybrid_early hybrid_late \
        --seed 0 --n_per_class 4 --T 50 --models_dir models_small --out_dir plots_small/repr
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, layerwise_similarity, MODES, ANGLES,
)

COLORS = {
    "none": "tab:blue", "alternating": "tab:orange", "full": "tab:green",
    "hybrid_early": "tab:red", "hybrid_late": "tab:purple",
}
METRIC_LABELS = {"cosine": "Cosine similarity", "cka": "CKA", "svcca": "SVCCA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=4,
                     help="NOTE: sec:simmetrics requires batch >=10x neuron count for "
                          "reliable SVCCA; with the reduced model's small per-layer neuron "
                          "counts, n_per_class=4 (16 images x 4 classes = 16 total... use "
                          "more classes/images if warned) is usually already sufficient -- "
                          "the script will print a warning per layer if the ratio is low.")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/repr")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=args.seed)
    N = len(imgs)
    print(f"Loaded {N} probe images")

    all_sim = {}  # mode -> layer -> {metric: val}
    for mode in args.modes:
        print(f"=== mode={mode} ===")
        model = load_model(mode, args.seed, models_dir=args.models_dir)
        sim = layerwise_similarity(model, imgs, angles=ANGLES, T=args.T, seed=args.seed)
        all_sim[mode] = sim

        layers = sorted(sim.keys())
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in ("cosine", "cka", "svcca"):
            vals = [sim[l][metric] for l in layers]
            ax.plot(layers, vals, marker="o", label=METRIC_LABELS[metric])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Similarity (h_ell(x) vs h_ell(R_theta x), averaged over theta)")
        ax.set_title(f"P4CNNSmall [{mode}] representation similarity by layer\n"
                      f"(n={N} probe images, params=2420)")
        ax.legend()
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"similarity_by_layer_{mode}.png"), dpi=150)
        plt.close(fig)
        print(f"  saved similarity_by_layer_{mode}.png")

    # per-metric cross-config comparison
    for metric in ("cosine", "cka", "svcca"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for mode in args.modes:
            layers = sorted(all_sim[mode].keys())
            vals = [all_sim[mode][l][metric] for l in layers]
            ax.plot(layers, vals, marker="o", label=mode, color=COLORS.get(mode))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Layer")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"P4CNNSmall: {METRIC_LABELS[metric]} by layer, all configs\n"
                      f"(reduced capacity: 5 layers, 4 channels, 2,420 params)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"similarity_comparison_{metric}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved similarity_comparison_{metric}.png")


if __name__ == "__main__":
    main()
