"""
Train + rotation-angle test for the analog p4-equivariant P4CNN
(Cohen & Welling rotated-MNIST architecture).

Matches the experimental design from the internship report: train ONLY on
upright (0-degree) digits, then evaluate test accuracy at a sweep of fixed
rotation angles. This isolates architectural rotation invariance from
data-augmentation effects -- the network never sees a rotated digit during
training, so any accuracy at non-zero angles reflects what the p4 group
structure gives you "for free", not memorization of rotated examples.

Usage (single run):
    python train.py --seed 0 --n-classes 4 --epochs 20 \
        --out results/n4_seed0.json --save-model models/n4_seed0.pt

After running all your seeds, aggregate + plot with:
    python train.py --aggregate --results-dir results --n-classes 4 \
        --plot-out plots/n4_rotation_curve.png

Or do both (train this seed, then re-aggregate everything currently in
results/ and refresh the plot) with --aggregate-after:
    python train.py --seed 1 --n-classes 4 --epochs 20 \
        --out results/n4_seed1.json --save-model models/n4_seed1.pt \
        --aggregate-after --plot-out plots/n4_rotation_curve.png
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

from p4cnn import P4CNN

DATA_PATH = Path.home() / "Downloads" / "digit-recognizer" / "train.csv"
FOUR_CLASS_DIGITS = [3, 4, 8, 9]

# Rotation angles to test at, degrees. 0/90/180/270 are p4 "group angles"
# (exact symmetries of the architecture); the rest probe generalization
# to rotations the group does NOT exactly cover.
TEST_ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
               195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_mnist_csv(path, digits=None):
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    labels = raw[:, 0].astype(np.int64)
    images = raw[:, 1:].astype(np.float32) / 255.0
    images = images.reshape(-1, 28, 28)  # keep 2D for easy rotation later

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
    """Rotate a batch of (N, 28, 28) images by `angle` degrees, each
    independently re-cropped/padded back to 28x28 (reshape=False keeps
    output size fixed). angle=0 returns the array unchanged (no-op, exact)."""
    if angle == 0:
        return images.copy()
    rotated = scipy_rotate(images, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
    return rotated.astype(np.float32)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def make_loader(images, labels, batch_size, shuffle):
    x = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 28, 28)
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
    """Evaluate test accuracy at each rotation angle. test_images stays
    upright in storage; we rotate a fresh copy per angle."""
    results = {}
    for angle in angles:
        rotated = rotate_batch(test_images, angle)
        loader = make_loader(rotated, test_labels, batch_size, shuffle=False)
        acc = evaluate(model, loader, device)
        results[angle] = acc
    return results


def train_one_run(seed, n_classes, epochs, batch_size, lr, device, base_channels=10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    digits = FOUR_CLASS_DIGITS if n_classes == 4 else None
    images, labels = load_mnist_csv(DATA_PATH, digits=digits)
    (train_x, train_y), (test_x, test_y) = split_train_test(images, labels, seed=42)

    train_loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    upright_test_loader = make_loader(test_x, test_y, batch_size=256, shuffle=False)

    model = P4CNN(in_channels=1, n_classes=n_classes, channels=base_channels).to(device)
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

        print(f"[seed={seed}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_acc(0deg)={test_acc:.4f} time={epoch_time:.2f}s", flush=True)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "test_acc_0deg": test_acc, "epoch_time_s": epoch_time,
        })

    model.load_state_dict(best_state)
    print(f"[seed={seed}] running rotation sweep ({len(TEST_ANGLES)} angles)...", flush=True)
    t0 = time.time()
    rotation_results = evaluate_rotation_sweep(model, test_x, test_y, batch_size=256,
                                                device=device, angles=TEST_ANGLES)
    print(f"[seed={seed}] rotation sweep done in {time.time()-t0:.1f}s", flush=True)
    for angle, acc in rotation_results.items():
        print(f"[seed={seed}]   angle={angle:3d}deg  acc={acc:.4f}", flush=True)

    result = {
        "seed": seed, "n_classes": n_classes, "best_test_acc_0deg": best_acc,
        "history": history, "rotation_sweep": rotation_results,
    }
    return result, best_state


# --------------------------------------------------------------------------
# Aggregation + plotting across seeds
# --------------------------------------------------------------------------

def aggregate_and_plot(results_dir, n_classes, plot_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f"n{n_classes}_seed*.json"))
    if not files:
        print(f"No result files found matching n{n_classes}_seed*.json in {results_dir}")
        return

    all_sweeps = []
    seeds_used = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        sweep = {int(k): v for k, v in data["rotation_sweep"].items()}
        all_sweeps.append(sweep)
        seeds_used.append(data["seed"])

    angles = sorted(all_sweeps[0].keys())
    mean_acc, std_acc = [], []
    for angle in angles:
        vals = [sweep[angle] for sweep in all_sweeps]
        mean_acc.append(np.mean(vals))
        std_acc.append(np.std(vals))
    mean_acc = np.array(mean_acc)
    std_acc = np.array(std_acc)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(angles, mean_acc * 100, yerr=std_acc * 100, marker="o",
                capsize=3, linewidth=1.5, color="#2563eb", ecolor="#93c5fd")
    for group_angle in (0, 90, 180, 270):
        ax.axvline(group_angle, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    chance = 100.0 / n_classes
    ax.axhline(chance, color="red", linestyle=":", alpha=0.6, linewidth=1, label=f"chance ({chance:.1f}%)")

    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Analog P4CNN rotation-angle sweep, {n_classes}-class "
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-classes", type=int, choices=[4, 10], default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=10)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-model", type=str, default=None)

    p.add_argument("--aggregate", action="store_true",
                    help="Skip training; just aggregate existing results/*.json and plot.")
    p.add_argument("--aggregate-after", action="store_true",
                    help="After this training run, also re-aggregate all results and refresh the plot.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--plot-out", type=str, default=None)

    args = p.parse_args()

    if args.aggregate:
        if args.plot_out is None:
            args.plot_out = f"plots/n{args.n_classes}_rotation_curve.png"
        aggregate_and_plot(args.results_dir, args.n_classes, args.plot_out)
        return

    if args.seed is None or args.out is None:
        p.error("--seed and --out are required unless using --aggregate")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seed={args.seed}] using device={device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})", flush=True)

    result, best_state = train_one_run(
        seed=args.seed, n_classes=args.n_classes, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=device,
        base_channels=args.base_channels,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[seed={args.seed}] wrote results to {out_path}", flush=True)

    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)
        print(f"[seed={args.seed}] wrote best model weights to {model_path}", flush=True)

    if args.aggregate_after:
        if args.plot_out is None:
            args.plot_out = f"plots/n{args.n_classes}_rotation_curve.png"
        aggregate_and_plot(args.results_dir, args.n_classes, args.plot_out)


if __name__ == "__main__":
    main()