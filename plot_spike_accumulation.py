"""
plot_spike_accumulation.py

Plots a_i(x) = sum_{t=1}^{T} S_i(t)  (Eq. accumcount, sec:layerrep) for the
reduced (2,420-param) model -- the accumulated spike-count representation
that h_ell(x) is built from throughout this work.

Produces:
  plots_small/repr/spike_accum_hist_<mode>.png
      -- histogram of a_i(x) values across all neurons x all probe images,
         one subplot per layer, for a single config -- shows the overall
         "how much do neurons fire" distribution per layer.
  plots_small/repr/spike_accum_heatmap_<mode>_<layer>.png
      -- heatmap of a_i(x) for one example probe image, one per orientation
         channel (4 rows) x output channel (columns) -- a direct visual of
         which orientation channels are active for spiking layers only
         (skipped for analog layers, which have no orientation-selective
         spike pattern in the same sense -- their activation heatmap is
         included too, just without a "spike count" interpretation).
  plots_small/repr/firing_rate_by_layer_comparison.png
      -- mean firing rate (= mean a_i(x) / T for spiking layers) per layer,
         one line per config -- only layers in that config's spiking_layers
         set are plotted, so this shows exactly which layers are spiking
         and roughly how active they are, across configs.

Usage:
    python3 plot_spike_accumulation.py --modes none alternating full hybrid_early hybrid_late \
        --seed 0 --n_per_class 4 --T 50 --models_dir models_small --out_dir plots_small/repr
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, forward_with_activations, MODES,
)

COLORS = {
    "none": "tab:blue", "alternating": "tab:orange", "full": "tab:green",
    "hybrid_early": "tab:red", "hybrid_late": "tab:purple",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=4)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/repr")
    ap.add_argument("--heatmap_layer", type=int, default=3,
                     help="which layer (1-5) to render the example heatmap for")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=args.seed)
    print(f"Loaded {len(imgs)} probe images")

    firing_rate_by_mode = {}  # mode -> layer -> mean firing rate

    for mode in args.modes:
        print(f"=== mode={mode} ===")
        model = load_model(mode, args.seed, models_dir=args.models_dir)
        logits, h, h_t = forward_with_activations(model, imgs, T=args.T, seed=args.seed)

        # --- histogram of a_i(x), one subplot per layer ---
        layers = sorted(h.keys())
        fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), sharey=False)
        if len(layers) == 1:
            axes = [axes]
        for ax, layer_idx in zip(axes, layers):
            vals = h[layer_idx].flatten()
            is_spiking = layer_idx in model.spiking_layers
            ax.hist(vals, bins=30, color=COLORS.get(mode), alpha=0.8)
            unit = "spike count" if is_spiking else "activation"
            ax.set_title(f"L{layer_idx} ({'spiking' if is_spiking else 'analog'})")
            ax.set_xlabel(f"$a_i(x)$ [{unit}]")
        axes[0].set_ylabel("count (neurons x images)")
        fig.suptitle(f"P4CNNSmall [{mode}] accumulated activation distribution per layer "
                      f"(T={args.T}, params=2420)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"spike_accum_hist_{mode}.png"), dpi=150)
        plt.close(fig)
        print(f"  saved spike_accum_hist_{mode}.png")

        # --- example heatmap for one probe image, one chosen layer ---
        li = args.heatmap_layer
        if li in h_t:
            # h_t[li]: (T, N, n_i) flattened; recover (T,N,C,4,H,W) shape via re-forward
            with_shapes = True
            import torch
            x0 = imgs[0:1]
            _, h_full, _ = forward_with_activations(model, x0, T=args.T, seed=args.seed)
            # re-run with shapes preserved for the heatmap (small extra forward, single image)
            xt = torch.from_numpy(x0).float().unsqueeze(1).to(next(model.parameters()).device)
            with torch.no_grad():
                _, acts = model(xt, T=args.T, record_activations=True)
            a = acts[li].sum(dim=0)[0]  # (C, 4, H, W), summed over T for image 0
            C, R, H, W = a.shape
            fig, axes = plt.subplots(R, C, figsize=(2.2 * C, 2.2 * R), squeeze=False)
            vmax = a.max().item() + 1e-8
            for r in range(R):
                for c in range(C):
                    im = axes[r][c].imshow(a[c, r].cpu().numpy(), cmap="viridis", vmin=0, vmax=vmax)
                    axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
                    if r == 0:
                        axes[r][c].set_title(f"ch{c}", fontsize=9)
                    if c == 0:
                        axes[r][c].set_ylabel(f"{r*90}°", fontsize=9)
            is_spiking = li in model.spiking_layers
            fig.suptitle(f"P4CNNSmall [{mode}] L{li} ({'spiking' if is_spiking else 'analog'}) "
                          f"a_i(x) heatmap, rows=orientation, cols=channel (example image)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"spike_accum_heatmap_{mode}_L{li}.png"), dpi=150)
            plt.close(fig)
            print(f"  saved spike_accum_heatmap_{mode}_L{li}.png")

        # --- firing rate per (spiking) layer for the comparison plot ---
        firing_rate_by_mode[mode] = {}
        for layer_idx in model.spiking_layers:
            firing_rate_by_mode[mode][layer_idx] = float(h[layer_idx].mean() / args.T)

    # --- firing-rate comparison across configs (spiking layers only) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode in args.modes:
        fr = firing_rate_by_mode[mode]
        if not fr:
            continue
        layers = sorted(fr.keys())
        vals = [fr[l] for l in layers]
        ax.plot(layers, vals, marker="o", label=f"{mode} (layers {sorted(model.spiking_layers) if mode==args.modes[-1] else ''})".split(" (layers")[0],
                 color=COLORS.get(mode))
        ax.scatter(layers, vals, color=COLORS.get(mode))
    ax.set_xlabel("Layer (only layers that spike in that config are shown)")
    ax.set_ylabel("Mean firing rate ($\\bar{a_i}(x)/T$)")
    ax.set_title("P4CNNSmall: mean firing rate of spiking layers, all configs\n"
                  "(reduced capacity: 5 layers, 4 channels, 2,420 params)")
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "firing_rate_by_layer_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved firing_rate_by_layer_comparison.png to {args.out_dir}")


if __name__ == "__main__":
    main()
