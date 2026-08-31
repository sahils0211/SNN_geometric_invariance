"""
train_spiking_small.py

Training/evaluation script for the reduced-capacity (5-layer, 4-channel,
2,420-param) SpikingP4CNNSmall, mirroring train_spiking.py's conventions
from the full-size (7-layer, 10-channel, 24,744-param) codebase so results
are directly comparable in format (same JSON schema, same CSV loading,
same rotation sweep, same seed convention) and differ ONLY in model
capacity and depth/width, not in protocol.

Usage (matches the full-size script's CLI):

    python train_spiking_small.py --mode full --seed 0
    python train_spiking_small.py --mode none --seed 0          # analog control
    python train_spiking_small.py --mode alternating --seed 0
    python train_spiking_small.py --mode hybrid_early --seed 0
    python train_spiking_small.py --mode hybrid_late --seed 0

    # after all 5 configs x 3 seeds are done:
    python train_spiking_small.py --aggregate --mode full
    python train_spiking_small.py --compare --modes none alternating full hybrid_early hybrid_late

Data: expects the same MNIST CSV used throughout this project,
~/Downloads/digit-recognizer/train.csv (Kaggle "Digit Recognizer" format:
first column "label", remaining 784 columns are pixel0..pixel783).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import rotate as scipy_rotate

from spiking_p4cnn_small import SpikingP4CNNSmall, count_params, MODES


def fmt_time(seconds):
    """Human-readable duration, e.g. 92 -> '1m32s', 4000 -> '1h6m'."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressTracker:
    """Tracks elapsed/remaining time across a known total number of "units"
    of work (here: batches across all epochs, plus sweep angles), printing
    a live-updating single-line progress bar with a rolling-average ETA.
    Call .tick() once per unit of work; call .close() when done with that
    phase so the line doesn't get overwritten by the next phase's prints.
    """

    def __init__(self, total_units, label, bar_width=28):
        self.total = max(total_units, 1)
        self.label = label
        self.bar_width = bar_width
        self.done = 0
        self.start = time.time()
        self._last_print = 0.0

    def tick(self, n=1, suffix=""):
        self.done += n
        now = time.time()
        # throttle printing to ~5x/sec so it doesn't spam the terminal/log
        if now - self._last_print < 0.2 and self.done < self.total:
            return
        self._last_print = now

        elapsed = now - self.start
        frac = self.done / self.total
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")

        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        eta_str = fmt_time(remaining) if remaining != float("inf") else "?"
        line = (f"\r[{self.label}] |{bar}| {self.done}/{self.total} "
                f"({frac*100:5.1f}%)  elapsed {fmt_time(elapsed)}  ETA {eta_str}  {suffix}")
        sys.stdout.write(line[:200].ljust(200))
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

DIGITS = [3, 4, 8, 9]
NUM_CLASSES = 4
ANGLES = list(range(0, 360, 15))  # 24 angles, 0..345
DATA_PATH = os.path.expanduser("~/Downloads/digit-recognizer/train.csv")
import os as _os
_suffix = _os.environ.get("OUT_SUFFIX", "")
RESULTS_DIR = "results_small" + _suffix
MODELS_DIR = "models_small" + _suffix
PLOTS_DIR = "plots_small" + _suffix


def load_data(seed):
    raw = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    labels_all = raw[:, 0].astype(int)
    pixels_all = raw[:, 1:].astype(np.float32) / 255.0

    mask = np.isin(labels_all, DIGITS)
    labels = labels_all[mask]
    pixels = pixels_all[mask]
    label_map = {d: i for i, d in enumerate(DIGITS)}
    labels = np.array([label_map[l] for l in labels])

    rng = np.random.default_rng(42)
    n = len(labels)
    perm = rng.permutation(n)
    pixels, labels = pixels[perm], labels[perm]

    n_test = int(0.15 * n)
    test_pixels, test_labels = pixels[:n_test], labels[:n_test]
    train_pixels, train_labels = pixels[n_test:], labels[n_test:]

    train_images = train_pixels.reshape(-1, 28, 28)
    test_images = test_pixels.reshape(-1, 28, 28)
    return train_images, train_labels, test_images, test_labels


def to_tensor_batch(images_np):
    return torch.from_numpy(images_np).float().unsqueeze(1)  # (B, 1, 28, 28)


def rotate_batch(images_np, angle):
    if angle == 0:
        return images_np
    out = np.empty_like(images_np)
    for i in range(images_np.shape[0]):
        out[i] = scipy_rotate(images_np[i], angle, reshape=False, order=1, mode="constant", cval=0.0)
    return out


def evaluate(model, images_np, labels_np, device, T=50, batch_size=256):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels_np), batch_size):
            xb = to_tensor_batch(images_np[i:i + batch_size]).to(device)
            yb = torch.from_numpy(labels_np[i:i + batch_size]).long().to(device)
            out = model(xb, T=T)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return correct / len(labels_np)


def train_one(mode, seed, epochs=20, batch_size=64, lr=1e-3, T=50, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_start = time.time()
    train_images, train_labels, test_images, test_labels = load_data(seed)

    model = SpikingP4CNNSmall(num_classes=NUM_CLASSES, mode=mode).to(device)
    print(f"[{mode} seed={seed}] device={device}  param count = {count_params(model)}")
    print(f"[{mode} seed={seed}] train={len(train_labels)} test={len(test_labels)} "
          f"epochs={epochs} batch_size={batch_size} T={T}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()

    n = len(train_labels)
    n_batches_per_epoch = (n + batch_size - 1) // batch_size
    total_train_batches = n_batches_per_epoch * epochs
    best_acc0 = -1.0
    best_state = None
    train_loss_curve = []
    upright_acc_curve = []

    # single progress bar spanning ALL epochs (not reset each epoch), so the
    # ETA gets more accurate as training goes rather than restarting cold
    # every epoch. Suffix shows current epoch/loss so you don't lose that info.
    prog = ProgressTracker(total_train_batches, label=f"{mode} seed{seed} train")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = to_tensor_batch(train_images[idx]).to(device)
            yb = torch.from_numpy(train_labels[idx]).long().to(device)

            opt.zero_grad()
            out = model(xb, T=T)
            loss = lossfn(out, yb)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
            prog.tick(1, suffix=f"epoch {epoch}/{epochs}  batch_loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        acc0 = evaluate(model, test_images, test_labels, device, T=T)
        train_loss_curve.append(avg_loss)
        upright_acc_curve.append(acc0)
        # newline before the epoch summary so it doesn't collide with the bar
        print(f"\n[{mode} seed={seed}] epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"0deg_acc={acc0:.4f}  elapsed={fmt_time(time.time()-run_start)}")

        if acc0 > best_acc0:
            best_acc0 = acc0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prog.close()

    model.load_state_dict(best_state)

    # full rotation sweep with the best checkpoint -- separate progress bar,
    # since its per-unit cost (a full test-set eval) differs a lot from a
    # single training batch and would otherwise throw off the ETA above.
    sweep = {}
    sweep_prog = ProgressTracker(len(ANGLES), label=f"{mode} seed{seed} sweep")
    for angle in ANGLES:
        rotated = rotate_batch(test_images, angle)
        acc = evaluate(model, rotated, test_labels, device, T=T)
        sweep[angle] = acc
        sweep_prog.tick(1, suffix=f"angle={angle:3d}deg acc={acc:.4f}")
    sweep_prog.close()
    print(f"[{mode} seed={seed}] rotation sweep done. total run time: "
          f"{fmt_time(time.time()-run_start)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    result = {
        "mode": mode,
        "seed": seed,
        "T": T,
        "epochs": epochs,
        "param_count": count_params(model),
        "architecture": "P4CNNSmall (5 layers, 4 channels)",
        "best_upright_acc": best_acc0,
        "train_loss_curve": train_loss_curve,
        "upright_acc_curve": upright_acc_curve,
        "rotation_sweep": sweep,
    }
    result_path = os.path.join(RESULTS_DIR, f"{mode}_seed{seed}_small.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{mode} seed={seed}] results saved to {result_path}")

    model_path = os.path.join(MODELS_DIR, f"{mode}_seed{seed}_small.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[{mode} seed={seed}] model saved to {model_path}")

    return result


def aggregate(mode, seeds=(0, 1, 2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    all_sweeps = []
    for seed in seeds:
        path = os.path.join(RESULTS_DIR, f"{mode}_seed{seed}_small.json")
        if not os.path.exists(path):
            print(f"  missing {path}, skipping seed {seed}")
            continue
        with open(path) as f:
            r = json.load(f)
        all_sweeps.append(r)

    if not all_sweeps:
        print(f"no results found for mode={mode}")
        return

    angles = ANGLES
    accs = np.array([[r["rotation_sweep"][str(a)] for a in angles] for r in all_sweeps])
    mean_acc = accs.mean(axis=0) * 100
    std_acc = accs.std(axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(angles, mean_acc, yerr=std_acc, marker="o", capsize=3)
    ax.axhline(25.0, linestyle=":", color="gray", label="chance (25.0%)")
    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"P4CNNSmall [{mode}] rotation-angle sweep "
                 f"(mean +/- std over {len(all_sweeps)} seeds, params={all_sweeps[0]['param_count']})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"small_{mode}_rotation_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def compare(modes, seeds=(0, 1, 2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    angles = ANGLES

    for mode in modes:
        sweeps = []
        for seed in seeds:
            path = os.path.join(RESULTS_DIR, f"{mode}_seed{seed}_small.json")
            if os.path.exists(path):
                with open(path) as f:
                    sweeps.append(json.load(f))
        if not sweeps:
            print(f"  no results for mode={mode}, skipping")
            continue
        accs = np.array([[r["rotation_sweep"][str(a)] for a in angles] for r in sweeps])
        mean_acc = accs.mean(axis=0) * 100
        std_acc = accs.std(axis=0) * 100
        ax.errorbar(angles, mean_acc, yerr=std_acc, marker="o", capsize=3,
                     label=f"{mode} (n={len(sweeps)} seeds)")

    ax.axhline(25.0, linestyle=":", color="gray", label="chance (25.0%)")
    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("P4CNNSmall config comparison, 4-class rotation-angle sweep\n"
                 "(reduced capacity: 5 layers, 4 channels, 2,420 params)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "small_n5_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=list(MODES.keys()))
    parser.add_argument("--modes", type=str, nargs="+", default=list(MODES.keys()),
                         help="used only with --compare")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare(args.modes)
    elif args.aggregate:
        aggregate(args.mode)
    else:
        train_one(args.mode, args.seed, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, T=args.T)
