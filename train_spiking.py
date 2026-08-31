"""
Train + rotation-angle test for the SPIKING p4-equivariant P4CNN
(Fully Spiking or Hybrid Alternating configs).

Same experimental design as train.py (upright-only training, fixed-angle
rotation sweep at test time), but for SpikingP4CNN instead of the analog
P4CNN. Key differences from train.py:

  - Extra --T and --mode arguments (mode: "full" or "alternating")
  - Prints per-epoch wall-clock time explicitly, since this is the number
    you need to decide whether T=50 vs T=100, and Full vs Alternating, are
    affordable within your remaining timeline -- don't guess from my
    estimates, read these numbers directly off your first real epoch.
  - Everything else (data loading, upright-only train/test split, rotation
    sweep, JSON output format, aggregation/plotting) is identical to
    train.py, so results from both scripts can be aggregated together by
    n_classes if you want spiking and analog curves on the same plot later.

RECOMMENDED FIRST RUN (pilot, not a real result):
    python train_spiking.py --seed 0 --n-classes 4 --mode alternating \
        --T 50 --epochs 1 --out results/spike_pilot.json

This runs ONLY 1 epoch so you get a real per-epoch timing number in a
few minutes, before committing to the full 20-epoch x 3-seed x 5-config
grid. Multiply the printed epoch time by 20 (or however many epochs you
choose) to get a real per-run estimate, then by however many (config,
seed) combinations you plan to run.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import rotate as scipy_rotate

from spiking_p4cnn import SpikingP4CNN

DATA_PATH = Path.home() / "Downloads" / "digit-recognizer" / "train.csv"
FOUR_CLASS_DIGITS = [3, 4, 8, 9]

TEST_ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
               195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]


# --------------------------------------------------------------------------
# Data loading (identical to train.py)
# --------------------------------------------------------------------------

def load_mnist_csv(path, digits=None):
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    labels = raw[:, 0].astype(np.int64)
    images = raw[:, 1:].astype(np.float32) / 255.0
    images = images.reshape(-1, 28, 28)

    if digits is not None:
        mask = np.isin(labels, digits)
        images, labels = images[mask], labels[mask]
        remap = {d: i for i, d in enumerate(sorted(digits))}
        labels = np.array([remap[l] for l in labels], dtype=np.int64)

    return images, labels


def split_train_test(images, labels, seed=42, test_frac=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (images[train_idx], labels[train_idx]), (images[test_idx], labels[test_idx])


def rotate_batch(images, angle):
    if angle == 0:
        return images.copy()
    rotated = scipy_rotate(images, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
    return rotated.astype(np.float32)


def make_loader(images, labels, batch_size, shuffle):
    x = torch.from_numpy(images).unsqueeze(1)
    y = torch.from_numpy(labels)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n_correct, n_total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        n_correct += (out.argmax(1) == yb).sum().item()
        n_total += xb.size(0)
    return n_correct / n_total


def evaluate_rotation_sweep(model, test_images, test_labels, batch_size, device, angles=TEST_ANGLES):
    results = {}
    for angle in angles:
        rotated = rotate_batch(test_images, angle)
        loader = make_loader(rotated, test_labels, batch_size, shuffle=False)
        acc = evaluate(model, loader, device)
        results[angle] = acc
    return results


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

def train_one_run(seed, n_classes, mode, T, epochs, batch_size, lr, device,
                   base_channels=10, run_rotation_sweep=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    digits = FOUR_CLASS_DIGITS if n_classes == 4 else None
    images, labels = load_mnist_csv(DATA_PATH, digits=digits)
    (train_x, train_y), (test_x, test_y) = split_train_test(images, labels, seed=42)

    train_loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    upright_test_loader = make_loader(test_x, test_y, batch_size=256, shuffle=False)

    model = SpikingP4CNN(in_channels=1, n_classes=n_classes, channels=base_channels,
                          T=T, mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    history = []
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n_correct += (out.argmax(1) == yb).sum().item()
            n_total += xb.size(0)
        train_acc = n_correct / n_total
        train_loss = total_loss / n_total

        test_acc = evaluate(model, upright_test_loader, device)
        epoch_time = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # epoch_time is the single most useful number in this whole script --
        # multiply by remaining epochs to know if this run is worth finishing.
        print(f"[seed={seed} mode={mode} T={T}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_acc(0deg)={test_acc:.4f} time={epoch_time:.2f}s "
              f"(est. remaining: {epoch_time*(epochs-epoch-1):.1f}s)", flush=True)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "test_acc_0deg": test_acc, "epoch_time_s": epoch_time,
        })

    if best_state is not None:
        model.load_state_dict(best_state)

    rotation_results = {}
    if run_rotation_sweep:
        print(f"[seed={seed} mode={mode} T={T}] running rotation sweep "
              f"({len(TEST_ANGLES)} angles)...", flush=True)
        t0 = time.time()
        rotation_results = evaluate_rotation_sweep(model, test_x, test_y, batch_size=256,
                                                    device=device, angles=TEST_ANGLES)
        print(f"[seed={seed}] rotation sweep done in {time.time()-t0:.1f}s", flush=True)
        for angle, acc in rotation_results.items():
            print(f"[seed={seed}]   angle={angle:3d}deg  acc={acc:.4f}", flush=True)

    result = {
        "seed": seed, "n_classes": n_classes, "mode": mode, "T": T,
        "best_test_acc_0deg": best_acc, "history": history,
        "rotation_sweep": rotation_results,
    }
    return result, best_state


# --------------------------------------------------------------------------
# Aggregation + plotting across seeds (mirrors train.py, but keyed by
# n_classes AND mode/T, since spiking has more axes of variation)
# --------------------------------------------------------------------------

def _load_spiking_results(results_dir, n_classes, mode, T=None):
    """Find all results/spike_n{n}_mode{mode}[_T{T}]_seed*.json files."""
    results_dir = Path(results_dir)
    if T is not None:
        pattern = f"spike_n{n_classes}_{mode}_T{T}_seed*.json"
    else:
        pattern = f"spike_n{n_classes}_{mode}_T*_seed*.json"
    return sorted(results_dir.glob(pattern))


def aggregate_and_plot(results_dir, n_classes, mode, T, plot_out):
    """Same style as train.py's aggregate_and_plot: rotation curve (mean +-
    std across seeds) and training curves, for ONE spiking config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = _load_spiking_results(results_dir, n_classes, mode, T)
    if not files:
        print(f"No result files found matching spike_n{n_classes}_{mode}_T{T}_seed*.json in {results_dir}")
        return

    all_sweeps, seeds_used = [], []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        sweep = {int(k): v for k, v in data["rotation_sweep"].items()}
        all_sweeps.append(sweep)
        seeds_used.append(data["seed"])

    angles = sorted(all_sweeps[0].keys())
    mean_acc = np.array([np.mean([s[a] for s in all_sweeps]) for a in angles])
    std_acc = np.array([np.std([s[a] for s in all_sweeps]) for a in angles])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(angles, mean_acc * 100, yerr=std_acc * 100, marker="o",
                capsize=3, linewidth=1.5, color="#dc2626", ecolor="#fca5a5")
    for group_angle in (0, 90, 180, 270):
        ax.axvline(group_angle, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    chance = 100.0 / n_classes
    ax.axhline(chance, color="gray", linestyle=":", alpha=0.6, linewidth=1, label=f"chance ({chance:.1f}%)")

    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Spiking P4CNN ({mode}, T={T}) rotation-angle sweep, {n_classes}-class "
                 f"(mean \u00b1 std over {len(files)} seeds: {seeds_used})")
    ax.set_xticks(angles[::2])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved plot to {plot_out}")

    print("\nangle  mean_acc  std_acc")
    for a, m, s in zip(angles, mean_acc, std_acc):
        print(f"{a:5d}  {m*100:7.2f}%  {s*100:6.2f}%")

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        epochs_ = [h["epoch"] for h in data["history"]]
        losses = [h["train_loss"] for h in data["history"]]
        accs = [h["test_acc_0deg"] for h in data["history"]]
        axes[0].plot(epochs_, losses, alpha=0.7, label=f"seed {data['seed']}")
        axes[1].plot(epochs_, accs, alpha=0.7, label=f"seed {data['seed']}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss"); axes[0].set_title("Training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test acc (0deg)"); axes[1].set_title("Upright test accuracy")
    for ax_ in axes:
        ax_.legend(fontsize=8); ax_.grid(alpha=0.2)
    fig2.tight_layout()
    curves_out = plot_out.parent / f"{plot_out.stem}_training_curves.png"
    fig2.savefig(curves_out, dpi=150)
    print(f"Saved training curves to {curves_out}")


def compare_configs_plot(results_dir, n_classes, configs, plot_out):
    """Overlay rotation curves from MULTIPLE configs on one plot -- e.g.
    analog vs spiking-alternating vs spiking-full, or different T values.
    This is the plot you want for "did spiking actually change anything
    relative to analog" questions.

    configs: list of dicts, each either
        {"label": "Analog",     "kind": "analog", "results_dir": "results"}
        {"label": "Alternating T=50", "kind": "spiking", "mode": "alternating", "T": 50}
        {"label": "Full T=50",        "kind": "spiking", "mode": "full", "T": 50}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir_path = Path(results_dir)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#9333ea", "#0891b2"]

    any_plotted = False
    for i, cfg in enumerate(configs):
        if cfg["kind"] == "analog":
            files = sorted(results_dir_path.glob(f"n{n_classes}_seed*.json"))
        else:
            files = _load_spiking_results(results_dir, n_classes, cfg["mode"], cfg.get("T"))

        if not files:
            print(f"  (skipping '{cfg['label']}' -- no matching result files found)")
            continue

        all_sweeps, seeds_used = [], []
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            sweep = {int(k): v for k, v in data["rotation_sweep"].items()}
            all_sweeps.append(sweep)
            seeds_used.append(data["seed"])

        angles = sorted(all_sweeps[0].keys())
        mean_acc = np.array([np.mean([s[a] for s in all_sweeps]) for a in angles])
        std_acc = np.array([np.std([s[a] for s in all_sweeps]) for a in angles])

        color = colors[i % len(colors)]
        ax.errorbar(angles, mean_acc * 100, yerr=std_acc * 100, marker="o",
                    capsize=3, linewidth=1.5, color=color,
                    label=f"{cfg['label']} (n={len(files)} seeds)")
        any_plotted = True

    if not any_plotted:
        print("Nothing to plot -- no matching result files for any requested config.")
        return

    for group_angle in (0, 90, 180, 270):
        ax.axvline(group_angle, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    chance = 100.0 / n_classes
    ax.axhline(chance, color="gray", linestyle=":", alpha=0.5, linewidth=1, label=f"chance ({chance:.1f}%)")

    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Config comparison, {n_classes}-class rotation-angle sweep")
    ax.set_xticks(sorted(set(a for cfg in configs for a in TEST_ANGLES))[::2])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved comparison plot to {plot_out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-classes", type=int, choices=[4, 10], default=4)
    p.add_argument("--mode", type=str,
                    choices=["full", "alternating", "hybrid_early", "hybrid_mid", "hybrid_late"],
                    default=None)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=10)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-model", type=str, default=None)
    p.add_argument("--skip-rotation-sweep", action="store_true",
                    help="Skip the 24-angle eval sweep -- useful for a pure timing pilot.")

    p.add_argument("--aggregate", action="store_true",
                    help="Skip training; aggregate existing results for ONE mode/T and plot.")
    p.add_argument("--aggregate-after", action="store_true",
                    help="After this training run, also re-aggregate results for this mode/T and refresh the plot.")
    p.add_argument("--compare", action="store_true",
                    help="Skip training; overlay analog + spiking config(s) on one comparison plot. "
                         "Configs to compare are the built-in default set below unless edited.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--plot-out", type=str, default=None)
    args = p.parse_args()

    if args.compare:
        if args.plot_out is None:
            args.plot_out = f"plots/n{args.n_classes}_comparison.png"
        # Edit this list to match whichever configs you actually have results for.
        configs = [
            {"label": "Analog", "kind": "analog"},
            {"label": "Spiking Alternating T=50", "kind": "spiking", "mode": "alternating", "T": 50},
            {"label": "Spiking Full T=50", "kind": "spiking", "mode": "full", "T": 50},
            {"label": "Spiking Hybrid Late T=50", "kind": "spiking", "mode": "hybrid_late", "T": 50},
        ]
        compare_configs_plot(args.results_dir, args.n_classes, configs, args.plot_out)
        return

    if args.aggregate:
        if args.mode is None:
            p.error("--mode is required with --aggregate")
        if args.plot_out is None:
            args.plot_out = f"plots/spike_n{args.n_classes}_{args.mode}_T{args.T}_rotation_curve.png"
        aggregate_and_plot(args.results_dir, args.n_classes, args.mode, args.T, args.plot_out)
        return

    if args.seed is None or args.out is None or args.mode is None:
        p.error("--seed, --mode, and --out are required unless using --aggregate or --compare")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seed={args.seed}] using device={device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}) "
          f"mode={args.mode} T={args.T}", flush=True)

    result, best_state = train_one_run(
        seed=args.seed, n_classes=args.n_classes, mode=args.mode, T=args.T,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
        base_channels=args.base_channels, run_rotation_sweep=not args.skip_rotation_sweep,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[seed={args.seed}] wrote results to {out_path}", flush=True)

    if args.save_model and best_state is not None:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)
        print(f"[seed={args.seed}] wrote best model weights to {model_path}", flush=True)

    if args.aggregate_after:
        if args.plot_out is None:
            args.plot_out = f"plots/spike_n{args.n_classes}_{args.mode}_T{args.T}_rotation_curve.png"
        aggregate_and_plot(args.results_dir, args.n_classes, args.mode, args.T, args.plot_out)


if __name__ == "__main__":
    main()