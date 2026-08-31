"""
plot_training_curves.py

Reproduces the original findings report's Figure 2 layout for the reduced-
capacity model: a top panel with the rotation-angle accuracy sweep (mean +/-
std over seeds), plus two bottom panels with per-seed training-loss and
per-seed upright (0deg) test-accuracy curves across epochs. Reads directly
from the JSON files train_spiking_small.py already saves per run --
no model loading / no GPU needed.

Usage:
    python3 plot_training_curves.py --mode none --seeds 0 1 2 \
        --results_dir results_small --out_dir plots_small
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANGLES = list(range(0, 360, 15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--results_dir", default="results_small")
    ap.add_argument("--out_dir", default="plots_small")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = []
    for seed in args.seeds:
        path = os.path.join(args.results_dir, f"{args.mode}_seed{seed}_small.json")
        if not os.path.exists(path):
            print(f"  missing {path}, skipping seed {seed}")
            continue
        with open(path) as f:
            runs.append((seed, json.load(f)))

    if not runs:
        print(f"no results found for mode={args.mode} in {args.results_dir}")
        return

    fig = plt.figure(figsize=(9, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])

    # --- top panel: rotation-angle sweep, mean +/- std over seeds ---
    ax_top = fig.add_subplot(gs[0, :])
    accs = np.array([[r["rotation_sweep"][str(a)] for a in ANGLES] for _, r in runs]) * 100
    mean_acc, std_acc = accs.mean(axis=0), accs.std(axis=0)
    ax_top.errorbar(ANGLES, mean_acc, yerr=std_acc, marker="o", capsize=3, color="tab:blue")
    ax_top.axhline(25.0, linestyle=":", color="gray", label="chance (25.0%)")
    ax_top.set_xlabel("Test rotation angle (degrees)")
    ax_top.set_ylabel("Test accuracy (%)")
    param_count = runs[0][1].get("param_count", "?")
    T = runs[0][1].get("T", "?")
    ax_top.set_title(f"P4CNNSmall [{args.mode}] rotation-angle sweep, mean +/- std over "
                      f"{len(runs)} seeds ({[s for s, _ in runs]})  (T={T}, params={param_count})")
    ax_top.legend()

    # --- bottom-left: per-seed training loss ---
    ax_loss = fig.add_subplot(gs[1, 0])
    for seed, r in runs:
        ax_loss.plot(range(1, len(r["train_loss_curve"]) + 1), r["train_loss_curve"],
                      label=f"seed {seed}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(fontsize=8)

    # --- bottom-right: per-seed upright (0deg) test accuracy ---
    ax_acc = fig.add_subplot(gs[1, 1])
    for seed, r in runs:
        curve = np.array(r["upright_acc_curve"]) * 100
        ax_acc.plot(range(1, len(curve) + 1), curve, label=f"seed {seed}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test acc (0deg) [%]")
    ax_acc.set_title("Upright test accuracy")
    ax_acc.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"small_{args.mode}_rotation_curve_training_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
