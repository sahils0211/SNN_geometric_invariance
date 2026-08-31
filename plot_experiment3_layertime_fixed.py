"""
plot_experiment3_layertime_fixed.py

Fix 1 (audit follow-up): re-runs Experiment 3's layer/time representation
distance D_{ell,t}(theta) using the layer-relative-epsilon + narrow-layer
masking fix in representations_small.layer_time_distance (see that
function's docstring), instead of the old fixed eps=1e-8.

Does NOT overwrite the original plots_small*/experiment3/ outputs -- writes
to a separate *_fixed output dir so old vs new can be compared side by side.
Also writes a per-(mode,seed,layer) diagnostics CSV reporting the per-layer
eps actually used and how many (timestep, image) cells were excluded as
floor-dominated, so the scale of the original problem is visible.

Usage:
    python3 plot_experiment3_layertime_fixed.py --modes none alternating full --seeds 0 1 2 \
        --T 50 --models_dir models_small --out_dir plots_small/experiment3_fixed
"""
import os
import csv
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
    ap.add_argument("--out_dir", default="plots_small/experiment3_fixed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=0)
    print(f"Loaded {len(imgs)} probe images")

    diag_rows = []

    for mode in args.modes:
        for seed in args.seeds:
            model = load_model(mode, seed, models_dir=args.models_dir)
            d45, diag45 = layer_time_distance(model, imgs, theta=45, T=args.T, seed=seed,
                                               return_diagnostics=True)
            d90, diag90 = layer_time_distance(model, imgs, theta=90, T=args.T, seed=seed,
                                               return_diagnostics=True)
            layers = sorted(d45.keys())

            for l in layers:
                diag_rows.append({
                    "mode": mode, "seed": seed, "T": args.T, "layer": l,
                    "eps_used": diag45[l]["eps"],
                    "n_excluded_theta45": diag45[l]["n_excluded"],
                    "n_excluded_theta90": diag90[l]["n_excluded"],
                    "n_total_cells": diag45[l]["n_total"],
                    "frac_excluded_theta45": diag45[l]["n_excluded"] / diag45[l]["n_total"],
                    "mean_D45": float(np.mean(d45[l])),
                    "median_D45": float(np.median(d45[l])),
                    "max_D45": float(np.max(d45[l])),
                })
                if diag45[l]["n_excluded"] > 0:
                    print(f"  [{mode} seed{seed} T{args.T}] L{l}: eps={diag45[l]['eps']:.4g}, "
                          f"excluded {diag45[l]['n_excluded']}/{diag45[l]['n_total']} cells "
                          f"({100*diag45[l]['n_excluded']/diag45[l]['n_total']:.1f}%) as floor-dominated (theta=45)")

            # --- heatmap: layer (y) x timestep (x), D_l,t(45) ---
            mat = np.stack([d45[l] for l in layers], axis=0)  # (n_layers, T)
            mat_shallow = mat[:-1]  # all but the deepest layer
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
            im1 = ax1.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest",
                              extent=[1, args.T, layers[-1] + 0.5, layers[0] - 0.5],
                              norm=LogNorm(vmin=max(mat.min(), 1e-3), vmax=max(mat.max(), 1e-2)))
            ax1.set_yticks(layers)
            ax1.set_xlabel("Timestep t")
            ax1.set_ylabel("Layer")
            ax1.set_title("All layers (log scale)")
            fig.colorbar(im1, ax=ax1, label=r"$D_{\ell,t}(45^\circ)$")

            im2 = ax2.imshow(mat_shallow, aspect="auto", cmap="magma", interpolation="nearest",
                              extent=[1, args.T, layers[-2] + 0.5, layers[0] - 0.5],
                              norm=LogNorm(vmin=max(mat_shallow.min(), 1e-3), vmax=max(mat_shallow.max(), 1e-2)))
            ax2.set_yticks(layers[:-1])
            ax2.set_xlabel("Timestep t")
            ax2.set_ylabel("Layer")
            ax2.set_title(f"Layers 1--{layers[-2]} only (own log color scale)")
            fig.colorbar(im2, ax=ax2, label=r"$D_{\ell,t}(45^\circ)$")

            fig.suptitle(f"P4CNNSmall [{mode}] seed={seed} $D_{{\\ell,t}}(45^\\circ)$ heatmap "
                          f"(T={args.T}, params=2420) -- FIXED: layer-relative eps + floor masking")
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
                          f"$D_{{\\ell,t}}(90^\\circ)$ (dashed, L5 only shown for clarity), T={args.T} -- FIXED")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            out_l = os.path.join(args.out_dir, f"exp3_lines_{mode}_seed{seed}.png")
            fig.savefig(out_l, dpi=150)
            plt.close(fig)

            print(f"  saved {out_h} and {out_l}")

    csv_path = os.path.join(args.out_dir, "exp3_eps_diagnostics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)
    print(f"Saved diagnostics CSV: {csv_path}")


if __name__ == "__main__":
    main()
