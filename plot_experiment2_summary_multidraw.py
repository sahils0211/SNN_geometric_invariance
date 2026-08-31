"""
plot_experiment2_summary_multidraw.py

Fix 2 (audit follow-up): recomputes Experiment 2's E90 using
representations_small.equivariance_error_multidraw (N=8 independent
Bernoulli-encoding draws per (seed, config), each with a genuinely
different encoding seed = seed*1000 + draw_index -- not the same
torch.manual_seed value reused), instead of the single-draw E90 in
exp2_summary_raw.csv / exp2_summary_agg.csv.

D_s / mean_acc / upright_acc / firing_rate are NOT the target of this fix
(the audit's Part 1 finding was specifically about the theta=90
equivariance-error value) and are recomputed with the original single-seed
rotation_spread_and_e90 / mean_firing_rate exactly as before, for context
alongside the new E90.

Does NOT overwrite exp2_summary_raw.csv / exp2_summary_agg.csv -- writes
experiment2_summary_multidraw_raw.csv (one row per seed, with E90_mean,
E90_std, and the 8 raw per-draw values) and
experiment2_summary_multidraw_agg.csv (mean/std over the 3 model seeds of
E90_mean -- i.e. combining model-seed variance and encoding-draw variance)
into the same out_dir as the original Experiment 2 outputs.

Usage:
    python3 plot_experiment2_summary_multidraw.py --seeds 0 1 2 --T 50 \
        --models_dir models_small --out_dir plots_small/experiment2 --n_draws 8
"""
import os
import argparse
import numpy as np
import pandas as pd

from representations_small import (
    load_probe_images, load_model, rotation_spread_and_e90, mean_firing_rate,
    equivariance_error_multidraw, ANGLES,
)

MODES = ["none", "alternating", "full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--n_draws", type=int, default=8)
    ap.add_argument("--models_dir", default="models_small")
    ap.add_argument("--out_dir", default="plots_small/experiment2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_probe_images(n_per_class=args.n_per_class, seed=0)
    print(f"Loaded {len(imgs)} probe images (shared across seeds for consistency)")

    rows = []
    for mode in MODES:
        for seed in args.seeds:
            model = load_model(mode, seed, models_dir=args.models_dir)

            # D_s / mean_acc / upright_acc / firing_rate: unchanged, single-seed,
            # not the target of this fix (context columns only).
            D_s, _E90_old, mean_acc, upright_acc = rotation_spread_and_e90(
                model, imgs, labels, angles=ANGLES, T=args.T, seed=seed)
            fr = mean_firing_rate(model, imgs, T=args.T, seed=seed)

            # E90: multi-draw, genuinely different encoding seed per draw.
            e90_mean, e90_std, e90_draws = equivariance_error_multidraw(
                model, imgs, theta=90, T=args.T, seed=seed, n_draws=args.n_draws)

            row = {
                "mode": mode, "seed": seed,
                "upright_acc": upright_acc, "mean_acc": mean_acc,
                "rotation_spread": D_s, "firing_rate": fr,
                "E90_singledraw_old": _E90_old,
                "E90_multidraw_mean": e90_mean, "E90_multidraw_std": e90_std,
                "n_draws": args.n_draws,
            }
            for k, v in enumerate(e90_draws):
                row[f"E90_draw{k}"] = v
            rows.append(row)
            print(f"  {mode} seed={seed}: E90 single-draw(old)={_E90_old:.4f}  "
                  f"E90 multi-draw mean={e90_mean:.4f} std={e90_std:.4f} "
                  f"(n_draws={args.n_draws}, per-draw range [{min(e90_draws):.4f}, {max(e90_draws):.4f}])")

    df = pd.DataFrame(rows)
    raw_path = os.path.join(args.out_dir, "experiment2_summary_multidraw_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # aggregate over the 3 model seeds -- this combines model-seed variance
    # AND encoding-draw variance (via E90_multidraw_mean's own std column,
    # each of which already has encoding-draw noise baked into its mean).
    agg = df.groupby("mode").agg(
        E90_mean_over_seeds=("E90_multidraw_mean", "mean"),
        E90_std_over_seeds=("E90_multidraw_mean", "std"),
        E90_mean_encoding_std=("E90_multidraw_std", "mean"),
        upright_acc_mean=("upright_acc", "mean"),
        mean_acc_mean=("mean_acc", "mean"),
        rotation_spread_mean=("rotation_spread", "mean"),
    )
    agg_path = os.path.join(args.out_dir, "experiment2_summary_multidraw_agg.csv")
    agg.to_csv(agg_path)
    print(f"Saved {agg_path}")
    print(agg)


if __name__ == "__main__":
    main()
