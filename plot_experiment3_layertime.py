"""
plot_experiment3_layertime.py

Experiment 3 (Additional-Experiments-Spiked.pdf, sec 6.3): "Where and When
Does Rotation Sensitivity Remain?" -- layer/time representation distance
D_{ell,t}(theta) for theta in {45, 90} degrees, using EXISTING checkpoints
only (Analog/Alternating/Full Spiking p4, same 3 configs as Experiment 2).

Produces, per mode per seed:
  <out_dir>/exp3_heatmap_<mode>_seed<seed>.png   -- layer x timestep heatmap, theta=45
  <out_dir>/exp3_lines_<mode>_seed<seed>.png     -- D_l,t(45) vs t, per layer, + theta=90 overlay for L5

Usage:
    python3 plot_experiment3_layertime.py --modes none alternating full --seeds 0 1 2 \
        --T 50 --models_dir models_small --out_dir plots_small/experiment3
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from representations_small import load_probe_images, load_model, layer_time_distance

MODES_DEFAULT = ["none", "alternating", "full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/experiment3")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=0)
    print(f"Loaded {len(imgs)} probe images")

    for mode in args.modes:
        for seed in args.seeds:
            model = load_model(mode, seed, models_dir=args.models_dir)
            d45 = layer_time_distance(model, imgs, theta=45, T=args.T, seed=seed)
            d90 = layer_time_distance(model, imgs, theta=90, T=args.T, seed=seed)
            layers = sorted(d45.keys())

            # --- heatmap: layer (y) x timestep (x), D_l,t(45) ---
            # Log color scale: D_l,t can span several orders of magnitude when a
            # deep spiking layer has near-zero reference-norm at a sparsely-firing
            # timestep (division by ~eps in the denominator). Even in log scale, one
            # extreme layer (typically the deepest) can still dominate the color
            # range and wash out the others, so we show two panels: all layers
            # (full dynamic range) and layers 1..L-1 only (own, finer color scale).
            mat = np.stack([d45[l] for l in layers], axis=0)  # (n_layers, T)
            mat_shallow = mat[:-1]  # all but the deepest layer
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
            im1 = ax1.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest",
                              extent=[1, args.T, layers[-1] + 0.5, layers[0] - 0.5],
                              norm=LogNorm(vmin=max(mat.min(), 1e-3), vmax=mat.max()))
            ax1.set_yticks(layers)
            ax1.set_xlabel("Timestep t")
            ax1.set_ylabel("Layer")
            ax1.set_title(f"All layers (log scale)")
            fig.colorbar(im1, ax=ax1, label=r"$D_{\ell,t}(45^\circ)$")

            im2 = ax2.imshow(mat_shallow, aspect="auto", cmap="magma", interpolation="nearest",
                              extent=[1, args.T, layers[-2] + 0.5, layers[0] - 0.5],
                              norm=LogNorm(vmin=max(mat_shallow.min(), 1e-3), vmax=mat_shallow.max()))
            ax2.set_yticks(layers[:-1])
            ax2.set_xlabel("Timestep t")
            ax2.set_ylabel("Layer")
            ax2.set_title(f"Layers 1--{layers[-2]} only (own log color scale)")
            fig.colorbar(im2, ax=ax2, label=r"$D_{\ell,t}(45^\circ)$")

            fig.suptitle(f"P4CNNSmall [{mode}] seed={seed} $D_{{\\ell,t}}(45^\\circ)$ heatmap "
                          f"(T={args.T}, params=2420)")
            fig.tight_layout()
            out_h = os.path.join(args.out_dir, f"exp3_heatmap_{mode}_seed{seed}.png")
            fig.savefig(out_h, dpi=150)
            plt.close(fig)

            # --- line plot: D_l,t(45) vs t for each layer, dashed = D_l,t(90) for comparison ---
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = plt.get_cmap("viridis")
            for i, l in enumerate(layers):
                c = cmap(i / max(len(layers) - 1, 1))
                ax.plot(range(1, args.T + 1), d45[l], color=c, label=f"L{l} ($45^\\circ$)")
                ax.plot(range(1, args.T + 1), d90[l], color=c, linestyle="--", alpha=0.6,
                         label=f"L{l} ($90^\\circ$)" if l == layers[-1] else None)
            ax.set_yscale("log")
            ax.set_xlabel("Timestep t")
            ax.set_ylabel(r"$D_{\ell,t}(\theta)$ (log scale)")
            ax.set_title(f"P4CNNSmall [{mode}] seed={seed}: $D_{{\\ell,t}}(45^\\circ)$ (solid) vs.\n"
                          f"$D_{{\\ell,t}}(90^\\circ)$ (dashed, L5 only shown for clarity), T={args.T}")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            out_l = os.path.join(args.out_dir, f"exp3_lines_{mode}_seed{seed}.png")
            fig.savefig(out_l, dpi=150)
            plt.close(fig)

            print(f"  saved {out_h} and {out_l}")


if __name__ == "__main__":
    main()
