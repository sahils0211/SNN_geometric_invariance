import torch
from representations import (
    load_analog_model, plot_layerwise_similarity,
    rotation_sweep_representations, rotation_sensitivity,
    plot_rotation_sensitivity_distribution,
)
from train_spiking import load_mnist_csv, split_train_test, DATA_PATH, FOUR_CLASS_DIGITS

images, labels = load_mnist_csv(DATA_PATH, digits=FOUR_CLASS_DIGITS)
(_, _), (test_x, test_y) = split_train_test(images, labels, seed=42)
x = torch.from_numpy(test_x[:16]).unsqueeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_analog_model("models/n4_seed0.pt", n_classes=4, channels=10, device=device)
x = x.to(device)

plot_layerwise_similarity(model, x, angles=[15,30,45,60,75,90],
                           metric="cka", plot_out="plots/analog_layerwise_cka.png")

all_reps = rotation_sweep_representations(model, x, angles=list(range(0,360,15)))
R = rotation_sensitivity(all_reps)
plot_rotation_sensitivity_distribution(R, plot_out="plots/analog_rotation_sensitivity.png")

print("Done.")
