"""
plot_equivariance_error.py

Plots Equivariance error(theta) = ||f(R_theta x) - f(x)|| / ||f(x)||
(Eq. eqerr, sec:eqerr) for the reduced (2,420-param) model, across configs.

Expected pattern (per sec:eqerr): error should be exactly (near) 0 at
theta in {0,90,180,270} (guaranteed by the p4-equivariant architecture)
and rise in between, since only those four angles are elements of the
p4 group the convolution kernels are built from.

Produces:
  plots_small/repr/equivariance_error_<mode>.png   -- one curve per config
  plots_small/repr/equivariance_error_comparison.png -- all configs overlaid

Usage:
    python3 plot_equivariance_error.py --modes none alternating full hybrid_early hybrid_late \
        --seed 0 --n_per_class 4 --T 50 --models_dir models_small --out_dir plots_small/repr
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, equivariance_error_curve, MODES, ANGLES,
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=args.seed)
    print(f"Loaded {len(imgs)} probe images")

    all_curves = {}
    for mode in args.modes:
        print(f"=== mode={mode} ===")
        model = load_model(mode, args.seed, models_dir=args.models_dir)
        errs = equivariance_error_curve(model, imgs, angles=ANGLES, T=args.T, seed=args.seed)
        all_curves[mode] = errs

        thetas = sorted(errs.keys())
        vals = [errs[t] for t in thetas]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thetas, vals, marker="o", color=COLORS.get(mode))
        for group_angle in (0, 90, 180, 270):
            ax.axvline(group_angle, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Rotation angle theta (degrees)")
        ax.set_ylabel(r"Equivariance error $(\theta)$ = $\|f(R_\theta x)-f(x)\|/\|f(x)\|$")
        ax.set_title(f"P4CNNSmall [{mode}] direct equivariance error vs. theta\n"
                      f"(dotted lines = p4 group angles 0/90/180/270, params=2420)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"equivariance_error_{mode}.png"), dpi=150)
        plt.close(fig)
        print(f"  saved equivariance_error_{mode}.png "
              f"(mean={np.mean(vals):.4f}, max={np.max(vals):.4f}, "
              f"at-group-angles={[errs[a] for a in (0,90,180,270)]})")

    # comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in args.modes:
        thetas = sorted(all_curves[mode].keys())
        vals = [all_curves[mode][t] for t in thetas]
        ax.plot(thetas, vals, marker="o", label=mode, color=COLORS.get(mode))
    for group_angle in (0, 90, 180, 270):
        ax.axvline(group_angle, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Rotation angle theta (degrees)")
    ax.set_ylabel(r"Equivariance error $(\theta)$")
    ax.set_title("P4CNNSmall: direct equivariance error, all configs\n"
                  "(reduced capacity: 5 layers, 4 channels, 2,420 params)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "equivariance_error_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved equivariance_error_comparison.png to {args.out_dir}")


if __name__ == "__main__":
    main()
