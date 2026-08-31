import torch
from representations import load_spiking_model, load_analog_model, run_full_analysis
from train_spiking import load_mnist_csv, split_train_test, DATA_PATH, FOUR_CLASS_DIGITS

images, labels = load_mnist_csv(DATA_PATH, digits=FOUR_CLASS_DIGITS)
(_, _), (test_x, test_y) = split_train_test(images, labels, seed=42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

x = torch.from_numpy(test_x[:150]).unsqueeze(1).to(device)

configs = [
    {"label": "analog_seed0", "kind": "analog", "path": "models/n4_seed0.pt"},
    {"label": "full_seed0", "kind": "spiking", "mode": "full", "path": "models/spike_n4_full_T50_seed0.pt"},
    {"label": "alternating_seed0", "kind": "spiking", "mode": "alternating", "path": "models/spike_n4_alternating_T50_seed0.pt"},
    {"label": "hybrid_late_seed0", "kind": "spiking", "mode": "hybrid_late", "path": "models/spike_n4_hybrid_late_T50_seed0.pt"},
]

for cfg in configs:
    print(f"\n{'='*70}\nProcessing: {cfg['label']}\n{'='*70}")
    try:
        if cfg["kind"] == "analog":
            model = load_analog_model(cfg["path"], n_classes=4, channels=10, device=device)
            run_full_analysis(
                model, x, test_x, test_y,
                plot_dir=f"plots/{cfg['label']}", label=cfg["label"], is_spiking=False,
            )
        else:
            model = load_spiking_model(cfg["path"], n_classes=4, channels=10, T=50, mode=cfg["mode"], device=device)
            run_full_analysis(
                model, x, test_x, test_y,
                plot_dir=f"plots/{cfg['label']}", label=cfg["label"], is_spiking=True,
            )
    except FileNotFoundError:
        print(f"  [skipped] checkpoint not found: {cfg['path']}")
        continue

print("\nAll configurations processed.")
