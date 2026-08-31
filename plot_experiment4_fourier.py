"""
plot_experiment4_fourier.py

Experiment 4 (Additional-Experiments-Spiked.pdf, sec 6.6): "Fourier Analysis
of the Rotational Response" -- fine-resolution (1-degree, 0..359) sweep of
the classification margin M(theta) and accuracy A(theta), Fourier-decomposed
to test the p4 architecture's predicted E_forbidden ~= 0 (only k=0,4,8,12,...
harmonics allowed) versus the seed-dependent, learned E_noninv. Uses EXISTING
checkpoints only (Analog/Alternating/Full Spiking p4, same 3 configs as
Experiments 2-3); no retraining.

Raw per-seed M(theta)/A(theta) arrays are cached to .npz (expensive: a 360-
angle sweep takes 1-2 minutes per checkpoint) so plotting/re-analysis never
needs to re-run the sweep.

Produces:
  <out_dir>/cache/<mode>_seed<seed>.npz         -- raw M, A arrays (cache)
  <out_dir>/exp4_rotation_response_<mode>.png   -- Plot 1: M(theta), all seeds + mean
  <out_dir>/exp4_fourier_spectrum_<mode>.png    -- Plot 2: |a_k+ib_k| vs k
  <out_dir>/exp4_seed_variability_modes.png     -- Plot 3: boxplot of |coef| at k=4,8,12, all modes
  <out_dir>/exp4_enoninv_eforbidden.png         -- Plot 4: E_noninv, E_forbidden per seed, all modes
  <out_dir>/exp4_summary.csv

Usage:
    python3 plot_experiment4_fourier.py --modes none alternating full --seeds 0 1 2 \
        --T 50 --models_dir models_small --out_dir plots_small/experiment4
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from representations_small import (
    load_probe_images, load_model, rotation_response_sweep, fourier_decompose,
)

MODES_DEFAULT = ["none", "alternating", "full"]
LABELS = {"none": "Analog $p4$", "alternating": "Alternating $p4$", "full": "Full Spiking $p4$"}
COLORS = {"none": "tab:blue", "alternating": "tab:orange", "full": "tab:green"}
K_MAX = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/experiment4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=0)
    print(f"Loaded {len(imgs)} probe images")

    results = {}  # mode -> seed -> dict(angles, M, A, fd)
    rows = []
    for mode in args.modes:
        results[mode] = {}
        for seed in args.seeds:
            cache_path = os.path.join(cache_dir, f"{mode}_seed{seed}.npz")
            if os.path.exists(cache_path):
                npz = np.load(cache_path)
                angles, M, A = npz["angles"], npz["M"], npz["A"]
                print(f"  {mode} seed={seed}: loaded from cache")
            else:
                model = load_model(mode, seed, models_dir=args.models_dir)
                print(f"  {mode} seed={seed}: running 360-angle sweep (T={args.T})...")
                angles, M, A = rotation_response_sweep(model, imgs, labels, T=args.T, seed=seed)
                np.savez(cache_path, angles=angles, M=M, A=A)
                print(f"    saved cache {cache_path}")

            fd = fourier_decompose(M, angles, K_max=K_MAX)
            results[mode][seed] = dict(angles=angles, M=M, A=A, fd=fd)
            rows.append({
                "mode": mode, "seed": seed,
                "a0": fd["a0"], "E_noninv": fd["E_noninv"], "E_forbidden": fd["E_forbidden"],
                "coef_k4": float(np.hypot(fd["a"][4], fd["b"][4])),
                "coef_k8": float(np.hypot(fd["a"][8], fd["b"][8])),
                "coef_k12": float(np.hypot(fd["a"][12], fd["b"][12])),
            })
            print(f"    a0={fd['a0']:.4f} E_noninv={fd['E_noninv']:.4f} "
                  f"E_forbidden={fd['E_forbidden']:.6f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "exp4_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # --- Plot 1 (per mode): M(theta), all seeds + mean ---
    for mode in args.modes:
        fig, ax = plt.subplots(figsize=(9, 5))
        all_M = []
        for seed in args.seeds:
            r = results[mode][seed]
            ax.plot(r["angles"], r["M"], alpha=0.4, linewidth=1, label=f"seed {seed}")
            all_M.append(r["M"])
        mean_M = np.mean(all_M, axis=0)
        ax.plot(results[mode][args.seeds[0]]["angles"], mean_M, color="black", linewidth=2,
                 label="mean")
        for ga in (0, 90, 180, 270):
            ax.axvline(ga, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Rotation angle theta (degrees)")
        ax.set_ylabel("Classification margin M(theta)")
        ax.set_title(f"Experiment 4: rotation response, [{mode}] (T={args.T})\n"
                      f"dotted lines = p4 group angles 0/90/180/270")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out1 = os.path.join(args.out_dir, f"exp4_rotation_response_{mode}.png")
        fig.savefig(out1, dpi=150)
        plt.close(fig)
        print(f"Saved {out1}")

    # --- Plot 2 (per mode): Fourier spectrum magnitude vs k ---
    for mode in args.modes:
        fig, ax = plt.subplots(figsize=(9, 5))
        ks = np.arange(1, K_MAX + 1)
        for seed in args.seeds:
            fd = results[mode][seed]["fd"]
            mag = np.hypot(fd["a"][1:K_MAX + 1], fd["b"][1:K_MAX + 1])
            colors = ["tab:red" if k % 4 == 0 else "tab:gray" for k in ks]
            ax.scatter(ks + (seed - 1) * 0.15, mag, c=colors, s=30, alpha=0.8)
        ax.set_xlabel("Fourier frequency k")
        ax.set_ylabel(r"$\sqrt{a_k^2+b_k^2}$")
        ax.set_xticks(ks)
        ax.set_title(f"Experiment 4: Fourier spectrum, [{mode}] (T={args.T})\n"
                      "red = allowed by p4 (k is a multiple of 4), gray = forbidden")
        fig.tight_layout()
        out2 = os.path.join(args.out_dir, f"exp4_fourier_spectrum_{mode}.png")
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        print(f"Saved {out2}")

    # --- Plot 3: seed variability of modes k=4,8,12, all modes overlaid ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)
    for ax, k in zip(axes, (4, 8, 12)):
        data = [df[df["mode"] == mode][f"coef_k{k}"].values for mode in args.modes]
        ax.boxplot(data, labels=[LABELS[m] for m in args.modes], showfliers=False)
        for i, mode in enumerate(args.modes):
            y = df[df["mode"] == mode][f"coef_k{k}"].values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, color=COLORS[mode], zorder=3, s=30, edgecolor="black", linewidth=0.4)
        ax.set_title(f"$k={k}$")
        ax.set_ylabel(r"$\sqrt{a_k^2+b_k^2}$")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle(f"Experiment 4: seed variability of allowed Fourier modes (T={args.T})")
    fig.tight_layout()
    out3 = os.path.join(args.out_dir, "exp4_seed_variability_modes.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved {out3}")

    # --- Plot 4: E_noninv and E_forbidden per seed, all modes ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, title in zip(axes, ("E_noninv", "E_forbidden"),
                               ("Non-invariant Fourier energy", "Forbidden (non-$C_4$) Fourier energy")):
        data = [df[df["mode"] == mode][col].values for mode in args.modes]
        ax.boxplot(data, labels=[LABELS[m] for m in args.modes], showfliers=False)
        for i, mode in enumerate(args.modes):
            y = df[df["mode"] == mode][col].values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, color=COLORS[mode], zorder=3, s=30, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle(f"Experiment 4: $E_{{noninv}}$ (learned) vs. $E_{{forbidden}}$ (architectural), T={args.T}")
    fig.tight_layout()
    out4 = os.path.join(args.out_dir, "exp4_enoninv_eforbidden.png")
    fig.savefig(out4, dpi=150)
    plt.close(fig)
    print(f"Saved {out4}")


if __name__ == "__main__":
    main()
