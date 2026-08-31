"""
plot_experiment2_summary.py

Experiment 2 (Additional-Experiments-Spiked.pdf, sec 6.2): "Does Spike
Placement Matter?" -- compares Analog p4 (mode='none'), Full Spiking p4
(mode='full'), and Alternating p4 (mode='alternating') at fixed p4
architecture, using EXISTING checkpoints only (no retraining).

Produces:
  <out_dir>/exp2_accuracy_by_angle.png   -- accuracy(theta) mean+-std, 3 lines
  <out_dir>/exp2_rotation_spread.png     -- boxplot/strip of D_s per config
  <out_dir>/exp2_summary.csv             -- Upright/Mean/Spread/E90/FiringRate table

Usage:
    python3 plot_experiment2_summary.py --seeds 0 1 2 --T 50 \
        --models_dir models_small --out_dir plots_small/experiment2
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, rotation_spread_and_e90, mean_firing_rate, ANGLES,
)

MODES = ["none", "alternating", "full"]
LABELS = {"none": "Analog $p4$", "alternating": "Alternating $p4$", "full": "Full Spiking $p4$"}
COLORS = {"none": "tab:blue", "alternating": "tab:orange", "full": "tab:green"}


def per_angle_accuracy(model, imgs, labels, T, seed):
    from representations_small import forward_with_activations
    accs = {}
    for theta in ANGLES:
        from representations_small import rotate_batch
        rot = rotate_batch(imgs, theta)
        logits, _, _ = forward_with_activations(model, rot, T=T, seed=seed)
        pred = logits.argmax(axis=1)
        accs[theta] = float((pred == labels).mean())
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/experiment2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=0)
    print(f"Loaded {len(imgs)} probe images (shared across seeds for consistency)")

    rows = []
    acc_by_mode_seed = {mode: {} for mode in MODES}

    for mode in MODES:
        for seed in args.seeds:
            model = load_model(mode, seed, models_dir=args.models_dir)
            accs = per_angle_accuracy(model, imgs, labels, args.T, seed)
            acc_by_mode_seed[mode][seed] = accs
            D_s, E90, mean_acc, upright_acc = rotation_spread_and_e90(
                model, imgs, labels, angles=ANGLES, T=args.T, seed=seed)
            fr = mean_firing_rate(model, imgs, T=args.T, seed=seed)
            rows.append({
                "mode": mode, "seed": seed,
                "upright_acc": upright_acc, "mean_acc": mean_acc,
                "rotation_spread": D_s, "E90": E90,
                "firing_rate": fr,
            })
            print(f"  {mode} seed={seed}: upright={upright_acc:.4f} mean={mean_acc:.4f} "
                  f"D_s={D_s:.4f} E90={E90:.4f} firing_rate={fr}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "exp2_summary_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # aggregate mean +/- std over seeds
    agg = df.groupby("mode").agg(["mean", "std"])
    agg_path = os.path.join(args.out_dir, "exp2_summary_agg.csv")
    agg.to_csv(agg_path)
    print(f"Saved {agg_path}")
    print(agg)

    # --- Plot 1: accuracy(theta) mean +/- std, 3 lines ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in MODES:
        arr = np.array([[acc_by_mode_seed[mode][s][a] for a in ANGLES] for s in args.seeds]) * 100
        mean_a, std_a = arr.mean(axis=0), arr.std(axis=0)
        ax.errorbar(ANGLES, mean_a, yerr=std_a, marker="o", capsize=3,
                     label=f"{LABELS[mode]} (n={len(args.seeds)} seeds)", color=COLORS[mode])
    ax.axhline(25.0, linestyle=":", color="gray", label="chance (25.0%)")
    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Experiment 2: accuracy(theta), Analog / Alternating / Full Spiking $p4$\n"
                  f"(T={args.T}, reduced capacity, 2420 params)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "exp2_accuracy_by_angle.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved {out1}")

    # --- Plot 2: rotation spread D_s distribution per config ---
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["mode"] == mode]["rotation_spread"].values * 100 for mode in MODES]
    bp = ax.boxplot(data, labels=[LABELS[m] for m in MODES], showfliers=False, widths=0.5)
    for i, mode in enumerate(MODES):
        y = df[df["mode"] == mode]["rotation_spread"].values * 100
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, color=COLORS[mode], zorder=3, s=40, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(r"Rotation spread $D_s$ = max$_\theta$ Acc $-$ min$_\theta$ Acc (%)")
    ax.set_title(f"Experiment 2: rotation-spread distribution across {len(args.seeds)} seeds\n"
                  f"(T={args.T}, reduced capacity)")
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "exp2_rotation_spread.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
