"""
plot_truncated_T.py

Plots Truncated-T accuracy (sec:truncT, Experiment 5.7): a single model
trained once at T=50 is evaluated by truncating its own recorded spike
train to the first T' in {1,5,10,20,50} timesteps, without retraining --
measuring how quickly rotation-robust accuracy structure becomes
available as spikes accumulate.

Produces:
  plots_small/repr/truncated_T_<mode>.png       -- accuracy vs T' for one config
  plots_small/repr/truncated_T_comparison.png   -- all configs overlaid

Usage:
    python3 plot_truncated_T.py --modes none alternating full hybrid_early hybrid_late \
        --seed 0 --n_per_class 6 --models_dir models_small --out_dir plots_small/repr
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, truncated_T_accuracy, MODES, ANGLES,
)

COLORS = {
    "none": "tab:blue", "alternating": "tab:orange", "full": "tab:green",
    "hybrid_early": "tab:red", "hybrid_late": "tab:purple",
}
T_PRIMES = (1, 5, 10, 20, 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=6)
    ap.add_argument("--T_full", type=int, default=50)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/repr")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=args.seed)
    print(f"Loaded {len(imgs)} probe images")

    all_curves = {}
    for mode in args.modes:
        print(f"=== mode={mode} ===")
        model = load_model(mode, args.seed, models_dir=args.models_dir)
        acc = truncated_T_accuracy(model, imgs, labels, angles=ANGLES,
                                    T_full=args.T_full, T_primes=T_PRIMES, seed=args.seed)
        all_curves[mode] = acc
        print(f"  {acc}")

        tps = sorted(acc.keys())
        vals = [acc[tp] * 100 for tp in tps]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tps, vals, marker="o", color=COLORS.get(mode))
        ax.set_xscale("log")
        ax.set_xticks(tps)
        ax.set_xticklabels([str(tp) for tp in tps])
        ax.set_ylim(0, 100)
        ax.axhline(25.0, color="gray", linestyle=":", label="chance (25.0%)")
        ax.set_xlabel("T' (truncated timesteps used for decoding)")
        ax.set_ylabel("Test accuracy (%), averaged over all 24 rotation angles")
        ax.set_title(f"P4CNNSmall [{mode}] truncated-T' accuracy\n"
                      f"(model trained once at T={args.T_full}, no retraining, params=2420)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"truncated_T_{mode}.png"), dpi=150)
        plt.close(fig)
        print(f"  saved truncated_T_{mode}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for mode in args.modes:
        tps = sorted(all_curves[mode].keys())
        vals = [all_curves[mode][tp] * 100 for tp in tps]
        ax.plot(tps, vals, marker="o", label=mode, color=COLORS.get(mode))
    ax.set_xscale("log")
    ax.set_xticks(T_PRIMES)
    ax.set_xticklabels([str(tp) for tp in T_PRIMES])
    ax.set_ylim(0, 100)
    ax.axhline(25.0, color="gray", linestyle=":", label="chance (25.0%)")
    ax.set_xlabel("T' (truncated timesteps used for decoding)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("P4CNNSmall: truncated-T' accuracy, all configs\n"
                  "(reduced capacity: 5 layers, 4 channels, 2,420 params)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "truncated_T_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved truncated_T_comparison.png to {args.out_dir}")


if __name__ == "__main__":
    main()
