"""
plot_truncated_T_by_angle.py

Angle-resolved truncated-T' accuracy (sec:truncT, Experiment 5.7), matching
the ORIGINAL findings report's Figure 42-50 style exactly: accuracy vs.
ROTATION ANGLE, one line per T' in {1,5,10,20,50}, showing how quickly the
familiar W-shaped rotation-accuracy curve "switches on" as more spiking
timesteps become available. This is a different (finer-grained) view than
plot_truncated_T.py's angle-AVERAGED accuracy-vs-T' curve.

Produces, per mode per seed:
  plots_small*/repr_seed<seed>/truncated_T_by_angle_<mode>.png

Usage:
    python3 plot_truncated_T_by_angle.py --modes none alternating full hybrid_early hybrid_late \
        --seed 0 --n_per_class 6 --models_dir models_small --out_dir plots_small/repr_seed0
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, truncated_T_accuracy_by_angle, MODES, ANGLES,
)

T_PRIMES = (1, 5, 10, 20, 50)
T_COLORS = {1: "tab:purple", 5: "tab:blue", 10: "tab:cyan", 20: "tab:green", 50: "gold"}


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

    for mode in args.modes:
        print(f"=== mode={mode} ===")
        model = load_model(mode, args.seed, models_dir=args.models_dir)
        acc_by_tp_angle = truncated_T_accuracy_by_angle(
            model, imgs, labels, angles=ANGLES, T_full=args.T_full,
            T_primes=T_PRIMES, seed=args.seed)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for tp in T_PRIMES:
            thetas = sorted(acc_by_tp_angle[tp].keys())
            vals = [acc_by_tp_angle[tp][t] * 100 for t in thetas]
            ax.plot(thetas, vals, marker="o", markersize=3, label=f"T'={tp}",
                     color=T_COLORS.get(tp))
        ax.axhline(25.0, color="gray", linestyle=":", alpha=0.6)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Test rotation angle (degrees)")
        ax.set_ylabel("Accuracy (%), decoded from first T' timesteps")
        ax.set_title(f"P4CNNSmall [{mode}] seed={args.seed} temporal evolution of invariance\n"
                      f"(model trained once at T={args.T_full}, no retraining, params=2420)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"truncated_T_by_angle_{mode}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
