"""
Section 4 infrastructure: layer representations, neuron responses,
representation similarity, and rotation sensitivity, as defined in
KV sir's research plan ("Spike Dynamics for Learning Invariant
Representations").

This module takes an ALREADY-TRAINED SpikingP4CNN (or the analog P4CNN)
checkpoint and extracts the quantities Section 4 defines:

  4.1  h_l(x)        -- layer representation (accumulated spike count
                         per neuron, vector over the whole layer)
  4.2  a_i(x)         -- individual neuron response (same accumulated
                         count, viewed per-neuron rather than per-layer)
  4.3  similarity(h_l(x), h_l(R_theta x))  -- cosine / CKA / SVCCA
  4.4  R_i = Var_theta(a_i(R_theta x))      -- rotation sensitivity
  4.5  h_l(t,x)       -- temporal representation (instantaneous, not
                         accumulated -- available directly from
                         SpikingP4CNN's record_activations output before
                         any summing over T)

Works on both:
  - SpikingP4CNN checkpoints (uses the T-timestep activations directly)
  - The analog P4CNN (T=1 conceptually -- wrapped so the same downstream
    similarity/sensitivity code works on both without special-casing)

USAGE (typical Experiment 5.1 / 5.3 workflow):

    from representations import (
        load_spiking_model, load_analog_model,
        get_layer_representations, rotation_sweep_representations,
        cosine_similarity, linear_cka, rotation_sensitivity,
        plot_layerwise_similarity, plot_rotation_sensitivity_distribution,
    )

    model = load_spiking_model("models/spike_n4_full_T50_seed0.pt",
                                n_classes=4, channels=10, T=50, mode="full")
    x = ...  # a batch of upright test images, (B, 1, 28, 28) in [0,1]

    # 4.1/4.2: representations at 0 degrees vs 90 degrees
    reps_0   = get_layer_representations(model, x, angle=0)
    reps_90  = get_layer_representations(model, x, angle=90)

    # 4.3: similarity per layer
    for layer in range(7):
        sim = cosine_similarity(reps_0[layer], reps_90[layer])
        print(f"layer {layer+1}: cosine similarity = {sim:.4f}")

    # 4.4: rotation sensitivity per neuron, then plot distribution per layer
    all_reps = rotation_sweep_representations(model, x, angles=range(0,360,15))
    R = rotation_sensitivity(all_reps)  # dict: layer -> (n_neurons,) tensor
    plot_rotation_sensitivity_distribution(R, plot_out="plots/rotation_sensitivity.png")
"""

import torch
import numpy as np
from pathlib import Path

from spiking_p4cnn import SpikingP4CNN
from p4cnn import P4CNN


# --------------------------------------------------------------------------
# Model loading helpers
# --------------------------------------------------------------------------

def load_spiking_model(checkpoint_path, n_classes, channels, T, mode, device="cpu"):
    model = SpikingP4CNN(in_channels=1, n_classes=n_classes, channels=channels, T=T, mode=mode)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_analog_model(checkpoint_path, n_classes, channels, device="cpu"):
    model = P4CNN(in_channels=1, n_classes=n_classes, channels=channels)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------
# 4.1 / 4.2 / 4.5: extracting representations
# --------------------------------------------------------------------------

def _rotate_tensor_batch(x, angle):
    """Rotate a (B, 1, H, W) tensor batch by `angle` degrees using the same
    scipy-based rotation as train.py/train_spiking.py, so representation
    analysis uses EXACTLY the same rotation operator as the accuracy
    rotation sweep (important for consistency between Experiment 5's
    accuracy numbers and this section's representation numbers)."""
    from scipy.ndimage import rotate as scipy_rotate
    if angle == 0:
        return x.clone()
    x_np = x.squeeze(1).cpu().numpy()  # (B, H, W)
    rotated = scipy_rotate(x_np, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
    return torch.from_numpy(rotated.astype(np.float32)).unsqueeze(1).to(x.device)


@torch.no_grad()
def get_layer_representations(model, x, angle=0):
    """Returns h_l(x) for every layer l, for a SpikingP4CNN, at rotation
    angle `angle` (degrees). h_l(x) is the ACCUMULATED spike count per
    neuron (summed over T) -- Section 4.1's definition.

    Returns: dict {layer_idx (0-6): tensor of shape (B, n_neurons_l)},
    flattened over channel/orientation/spatial dims so each layer's
    representation is a flat per-sample vector, ready for cosine/CKA/SVCCA.
    """
    x_rot = _rotate_tensor_batch(x, angle)
    model.eval()
    _, activations = model(x_rot, record_activations=True)
    # activations[l]: (T, B, C, H, W) -- sum over T for accumulated spike count
    reps = {}
    for layer_idx, tensor in activations.items():
        h = tensor.sum(dim=0)              # (B, C, H, W) -- accumulated over time
        B = h.shape[0]
        reps[layer_idx] = h.reshape(B, -1)  # (B, n_neurons_l) flat per-sample vector
    return reps


@torch.no_grad()
def get_temporal_representations(model, x, angle=0):
    """Returns h_l(t,x) for every layer l and timestep t -- Section 4.5.
    Same as get_layer_representations but WITHOUT summing over T, so you
    can study how representations evolve timestep by timestep.

    Returns: dict {layer_idx: tensor of shape (T, B, n_neurons_l)}.
    """
    x_rot = _rotate_tensor_batch(x, angle)
    model.eval()
    _, activations = model(x_rot, record_activations=True)
    reps = {}
    for layer_idx, tensor in activations.items():
        T, B = tensor.shape[0], tensor.shape[1]
        reps[layer_idx] = tensor.reshape(T, B, -1)  # (T, B, n_neurons_l)
    return reps


def rotation_sweep_representations(model, x, angles):
    """Run get_layer_representations at every angle in `angles`.
    Returns: dict {angle: {layer_idx: (B, n_neurons_l) tensor}}.
    This is the core data structure both 4.3 (similarity vs angle) and
    4.4 (rotation sensitivity per neuron) are computed from.
    """
    return {angle: get_layer_representations(model, x, angle=angle) for angle in angles}


# --------------------------------------------------------------------------
# 4.3: representation similarity
# --------------------------------------------------------------------------

def cosine_similarity(h_a, h_b):
    """Mean cosine similarity between h_a and h_b across the batch.
    h_a, h_b: (B, n_neurons) tensors (e.g. from get_layer_representations
    at two different angles, same layer).
    Returns a scalar float.
    """
    h_a = h_a.float()
    h_b = h_b.float()
    num = (h_a * h_b).sum(dim=1)
    denom = h_a.norm(dim=1) * h_b.norm(dim=1) + 1e-8
    return (num / denom).mean().item()


def linear_cka(h_a, h_b):
    """Linear Centered Kernel Alignment between two representation
    matrices. h_a, h_b: (B, n_neurons_a), (B, n_neurons_b) -- n_neurons
    can differ between the two (CKA doesn't require matching dimensions,
    only matching B).

    IMPORTANT: uses the Gram-matrix (batch x batch) formulation rather
    than the neuron x neuron formulation. Early conv layers here have
    tens of thousands of neurons (e.g. 27,040 for layer 1), so a naive
    X^T X computation would be a ~27000x27000 matrix (~3GB+ per matrix)
    and can OOM-kill the process. The Gram-matrix form below is
    mathematically equivalent for linear CKA but only ever computes
    B x B matrices (B = batch size, typically small), which is the
    standard way to make CKA tractable when neurons >> batch size.
    """
    X = h_a.float()
    Y = h_b.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.t()  # (B, B) Gram matrix, NOT (n_neurons, n_neurons)
    L = Y @ Y.t()  # (B, B)

    hsic_kl = (K * L).sum()
    hsic_kk = (K * K).sum()
    hsic_ll = (L * L).sum()

    denom = torch.sqrt(hsic_kk * hsic_ll) + 1e-8
    return (hsic_kl / denom).item()


def svcca_similarity(h_a, h_b, n_components=10, min_ratio=10):
    """SVCCA similarity: SVD-truncate both representations to
    `n_components` (or fewer, if the representation is smaller), then run
    CCA (canonical correlation analysis) between the truncated bases, and
    return the mean canonical correlation. Requires scikit-learn.

    h_a, h_b: (B, n_neurons_a), (B, n_neurons_b).

    IMPORTANT -- SVCCA's sample-size requirement is stricter than it may
    look. An empirical check (see accompanying analysis) showed that even
    at a batch-size-to-n_components ratio of 10x, CCA finds a "spurious"
    mean canonical correlation of roughly 0.2-0.4 on data that is
    actually INDEPENDENT -- i.e. the metric is meaningfully biased
    upward from small samples well beyond the naive B > n_components
    threshold. This function defaults to requiring B >= min_ratio *
    n_components (default min_ratio=10) and RAISES rather than silently
    returning a number if that is not satisfied, since a silently
    returned but statistically meaningless "high similarity" number is
    much more dangerous than a clear error -- it looks like a real
    finding.

    Even when this minimum is satisfied, treat SVCCA values as having a
    substantial positive floor (roughly 0.2-0.4 at this ratio) rather
    than 0 for "no relationship" -- compare configurations/angles to EACH
    OTHER under matched batch size, not to an absolute zero baseline.
    """
    from sklearn.cross_decomposition import CCA

    X = h_a.float().cpu().numpy()
    Y = h_b.float().cpu().numpy()
    B = X.shape[0]

    max_safe_components = max(1, B // min_ratio)
    safe_n_components = min(n_components, max_safe_components)

    if B < min_ratio * 2:  # need at least 2 components' worth of margin
        raise ValueError(
            f"svcca_similarity: batch size {B} is too small for a reliable SVCCA "
            f"estimate at min_ratio={min_ratio} (need B >= {min_ratio*2} for even "
            f"n_components=2). Use a larger evaluation batch (recommend B >= 64 "
            f"for n_components=10 at the default ratio), or explicitly pass a "
            f"smaller min_ratio if you understand and accept the resulting bias."
        )

    def svd_truncate(M, k):
        U, S, _ = np.linalg.svd(M - M.mean(axis=0, keepdims=True), full_matrices=False)
        k = min(k, U.shape[1])
        return U[:, :k] * S[:k]

    X_r = svd_truncate(X, safe_n_components)
    Y_r = svd_truncate(Y, safe_n_components)

    n_comp = min(X_r.shape[1], Y_r.shape[1])
    if n_comp < 1:
        return float("nan")

    cca = CCA(n_components=n_comp)
    cca.fit(X_r, Y_r)
    X_c, Y_c = cca.transform(X_r, Y_r)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_comp)]
    return float(np.nanmean(correlations))


def layerwise_similarity(model, x, angle, metric="cosine"):
    """Compute similarity between h_l(x) and h_l(R_theta x) for every
    layer l, at a single rotation angle. This is Experiment 5.1 -- plot
    the returned list against layer depth to see where invariance
    emerges.

    metric: "cosine", "cka", or "svcca"
    Returns: list of 7 similarity values, one per layer (layer 1 first).
    """
    metric_fn = {"cosine": cosine_similarity, "cka": linear_cka, "svcca": svcca_similarity}[metric]

    reps_0 = get_layer_representations(model, x, angle=0)
    reps_theta = get_layer_representations(model, x, angle=angle)

    return [metric_fn(reps_0[l], reps_theta[l]) for l in range(7)]


def plot_layerwise_similarity(model, x, angles, metric="cosine", plot_out="plots/layerwise_similarity.png"):
    """Experiment 5.1: similarity vs layer depth, one line per test angle.
    Saves a plot and returns the underlying data as a dict {angle: [sim_per_layer]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for angle in angles:
        if angle == 0:
            continue  # comparing 0 to itself is trivially 1.0, not informative
        results[angle] = layerwise_similarity(model, x, angle, metric=metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(1, 8))
    for angle, sims in results.items():
        ax.plot(layers, sims, marker="o", label=f"{angle}\u00b0", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"{metric.upper()} similarity to upright (0\u00b0)")
    ax.set_title(f"Layer-wise emergence of rotation invariance ({metric})")
    ax.set_xticks(layers)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved layer-wise similarity plot to {plot_out}")

    return results


def plot_layerwise_similarity_all_metrics(model, x, angles, plot_out="plots/layerwise_similarity_all_metrics.png", svcca_min_batch=100):
    """Experiment 5.1, extended: cosine, CKA, and SVCCA side by side as
    three subplots, each showing similarity vs layer depth for every
    tested angle. This is the single most direct implementation of
    Section 5.1 as specified (all three named metrics), rather than one
    metric at a time.

    SVCCA needs a much larger batch than cosine/CKA to be statistically
    meaningful (see svcca_similarity's docstring) -- if x's batch size is
    below svcca_min_batch, the SVCCA panel is skipped with a printed
    warning rather than silently plotting a biased or degenerate result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = x.shape[0]
    compute_svcca = B >= svcca_min_batch
    if not compute_svcca:
        print(f"  [note] batch size {B} < {svcca_min_batch}: skipping SVCCA panel "
              f"(cosine and CKA are still computed -- see svcca_similarity's docstring "
              f"for why small-batch SVCCA is unreliable, not just noisy)")

    metrics = ["cosine", "cka"] + (["svcca"] if compute_svcca else [])
    results = {m: {} for m in metrics}
    layers = list(range(1, 8))

    for angle in angles:
        if angle == 0:
            continue
        reps_0 = get_layer_representations(model, x, angle=0)
        reps_theta = get_layer_representations(model, x, angle=angle)
        for m in metrics:
            metric_fn = {"cosine": cosine_similarity, "cka": linear_cka, "svcca": svcca_similarity}[m]
            results[m][angle] = [metric_fn(reps_0[l], reps_theta[l]) for l in range(7)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.3 * len(metrics), 5), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        for angle, sims in results[m].items():
            ax.plot(layers, sims, marker="o", markersize=4, label=f"{angle}\u00b0", alpha=0.75)
        ax.set_xlabel("Layer")
        ax.set_title(m.upper())
        ax.set_xticks(layers)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Similarity to upright (0\u00b0)")
    axes[-1].legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(1.02, 0))
    fig.suptitle("Layer-wise emergence of rotation invariance -- similarity metrics")
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150, bbox_inches="tight")
    print(f"Saved combined layer-wise similarity plot ({', '.join(metrics)}) to {plot_out}")
    return results


# --------------------------------------------------------------------------
# 4.4: rotation sensitivity of individual neurons
# --------------------------------------------------------------------------

def rotation_sensitivity(all_reps):
    """Section 4.4: R_i = Var_theta(a_i(R_theta x)), computed per neuron,
    per layer, averaged over the batch.

    all_reps: output of rotation_sweep_representations -- dict
              {angle: {layer_idx: (B, n_neurons_l) tensor}}

    Returns: dict {layer_idx: (n_neurons_l,) tensor of R_i values},
    where each R_i is the variance (over the angles swept) of that
    neuron's mean-over-batch response.
    """
    angles = sorted(all_reps.keys())
    layers = sorted(all_reps[angles[0]].keys())

    R = {}
    for layer_idx in layers:
        # stack: (n_angles, B, n_neurons) -> mean over batch -> (n_angles, n_neurons)
        stacked = torch.stack([all_reps[a][layer_idx] for a in angles], dim=0)
        per_angle_mean = stacked.mean(dim=1)  # (n_angles, n_neurons)
        R[layer_idx] = per_angle_mean.var(dim=0)  # (n_neurons,)
    return R


def plot_rotation_sensitivity_distribution(R, plot_out="plots/rotation_sensitivity.png"):
    """Experiment 5.3: distribution of R_i across layers, as a boxplot/
    violin-style comparison. Low R_i = rotation-invariant neuron, high
    R_i = rotation-selective neuron."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(R.keys())
    data = [R[l].cpu().numpy() for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, positions=[l + 1 for l in layers], showfliers=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rotation sensitivity $R_i$ (variance across angles)")
    ax.set_title("Distribution of per-neuron rotation sensitivity by layer")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved rotation sensitivity distribution plot to {plot_out}")


# --------------------------------------------------------------------------
# 4.5 / 5.4: rotation sensitivity THROUGH TIME (temporal, spiking only)
# --------------------------------------------------------------------------

@torch.no_grad()
def rotation_sweep_temporal_representations(model, x, angles):
    """Same as rotation_sweep_representations, but keeps the full
    per-timestep axis (does not sum over T). Only meaningful for
    SpikingP4CNN -- calling this on the analog P4CNN will work (T=1) but
    is trivial, since there is no time axis to study.

    Returns: dict {angle: {layer_idx: (T, B, n_neurons_l) tensor}}.
    """
    return {angle: get_temporal_representations(model, x, angle=angle) for angle in angles}


def rotation_sensitivity_through_time(model, x, angles):
    """Section 4.5 / Experiment 5.4: a_i(t, theta) -- rotation sensitivity
    computed SEPARATELY at every timestep, rather than on the accumulated
    (summed-over-T) representation. Answers "does temporal integration
    gradually remove rotation dependence" by showing whether R_i shrinks
    as t increases within a single trained model's forward pass.

    Unlike an earlier version of this function, this one takes the model
    and input directly and processes ONE angle at a time, immediately
    reducing each angle's (T, B, n_neurons) activation to a running
    sum/sum-of-squares per layer before moving to the next angle, rather
    than materializing every angle's full temporal recording
    simultaneously. The naive all-angles-at-once approach was found,
    empirically, to exceed available memory even at a modest batch size
    once every layer and every timestep is held live at once for every
    angle in the sweep -- 24 angles x 20+ timesteps x 7 layers x a
    realistic batch size is tens of gigabytes if not reduced
    incrementally.

    Returns: dict {layer_idx: (T, n_neurons) tensor}, i.e. R_i(t) for
    every neuron at every timestep, in that layer, computed via a
    streaming (Welford-style, but simplified since we don't need the
    running estimate mid-stream) mean/variance across angles.
    """
    angles = sorted(angles)
    n_angles = len(angles)

    sum_per_layer = {}
    sumsq_per_layer = {}

    for angle in angles:
        temporal_reps = get_temporal_representations(model, x, angle=angle)
        for layer_idx, tensor in temporal_reps.items():
            # tensor: (T, B, n_neurons) -- reduce over batch immediately
            per_angle_mean = tensor.mean(dim=1)  # (T, n_neurons)
            if layer_idx not in sum_per_layer:
                sum_per_layer[layer_idx] = per_angle_mean.clone()
                sumsq_per_layer[layer_idx] = per_angle_mean.clone() ** 2
            else:
                sum_per_layer[layer_idx] += per_angle_mean
                sumsq_per_layer[layer_idx] += per_angle_mean ** 2
        del temporal_reps

    R_t = {}
    for layer_idx in sum_per_layer:
        mean = sum_per_layer[layer_idx] / n_angles
        mean_sq = sumsq_per_layer[layer_idx] / n_angles
        # population variance = E[X^2] - (E[X])^2, matches torch.var(unbiased=False)
        R_t[layer_idx] = (mean_sq - mean ** 2).clamp(min=0)
    return R_t


def plot_rotation_sensitivity_through_time(R_t, layers_to_plot=None, plot_out="plots/rotation_sensitivity_through_time.png"):
    """Experiment 5.4: median R_i(t) vs t, one line per layer. A downward
    trend within a layer's line means temporal integration is reducing
    that layer's rotation sensitivity as more spikes accumulate -- the
    "uniquely spiking phenomenon" the research plan highlights, since it
    has no analogue in a single-pass analog network.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = layers_to_plot if layers_to_plot is not None else sorted(R_t.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for layer_idx in layers:
        Rt = R_t[layer_idx]  # (T, n_neurons)
        median_per_t = Rt.median(dim=1).values.cpu().numpy()
        T = len(median_per_t)
        ax.plot(range(1, T + 1), median_per_t, marker="o", markersize=3,
                 label=f"layer {layer_idx+1}", alpha=0.8)

    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Median rotation sensitivity $R_i(t)$")
    ax.set_title("Rotation sensitivity through time (does temporal integration reduce it?)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved rotation-sensitivity-through-time plot to {plot_out}")


# --------------------------------------------------------------------------
# 5.7: temporal evolution of invariance -- INFERENCE-TIME T sweep
# --------------------------------------------------------------------------
#
# NOTE ON RETRAINING: this evaluates a model that was TRAINED at a fixed T
# (e.g. T=50) by truncating its own recorded spike train to the first t
# timesteps, for t in a swept list, and re-decoding the classifier output
# from only those t timesteps. This does NOT require retraining and
# answers "does invariance emerge progressively as spikes accumulate
# WITHIN this trained model's dynamics." It is a different (cheaper, and
# arguably more directly relevant) question than "does a model trained
# from scratch at each T converge to a different final answer," which
# WOULD require retraining.

@torch.no_grad()
def evaluate_accuracy_at_truncated_T(model, images, labels, angles, t_values, device="cpu", batch_size=64):
    """Re-decode classification accuracy using only the first t timesteps
    of the model's own T-timestep spike train, for each t in t_values, at
    each rotation angle in angles. Requires SpikingP4CNN (has a time axis
    to truncate); does not apply to the analog P4CNN.

    Returns: dict {t: {angle: accuracy}}.
    """
    from torch.utils.data import DataLoader, TensorDataset

    results = {t: {} for t in t_values}

    for angle in angles:
        rotated = _rotate_tensor_batch(torch.from_numpy(images).unsqueeze(1), angle).numpy()
        ds = TensorDataset(torch.from_numpy(rotated), torch.from_numpy(labels))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        correct_per_t = {t: 0 for t in t_values}
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            for lif in model.lifs:
                if hasattr(lif, "reset_mem"):
                    lif.reset_mem()

            B = xb.shape[0]
            from spiking_p4cnn import poisson_encode
            spike_train = poisson_encode(xb, model.T)

            mem_state = [None] * 7
            out_accum_per_t = {t: 0.0 for t in t_values}
            out_accum = 0.0

            for step in range(model.T):
                xt = spike_train[step]
                h = model.bn1(model.conv1(xt)); h = model._apply_activation(0, h, mem_state)
                h = model.bn2(model.conv2(h)); h = model._apply_activation(1, h, mem_state)
                h = model.pool(h)
                h = model.bn3(model.conv3(h)); h = model._apply_activation(2, h, mem_state)
                h = model.bn4(model.conv4(h)); h = model._apply_activation(3, h, mem_state)
                h = model.bn5(model.conv5(h)); h = model._apply_activation(4, h, mem_state)
                h = model.bn6(model.conv6(h)); h = model._apply_activation(5, h, mem_state)
                h = model.bn7(model.conv7(h)); h = model._apply_activation(6, h, mem_state)
                h = model.group_pool(h).flatten(1)
                out_t = model.fc(h)
                out_accum = out_accum + out_t

                current_t = step + 1
                if current_t in t_values:
                    out_accum_per_t[current_t] = out_accum.clone() / current_t

            for t in t_values:
                pred = out_accum_per_t[t].argmax(1)
                correct_per_t[t] += (pred == yb).sum().item()
            total += B

        for t in t_values:
            results[t][angle] = correct_per_t[t] / total

    return results


def plot_temporal_evolution_of_invariance(t_results, plot_out="plots/temporal_evolution_invariance.png"):
    """Experiment 5.7: accuracy-vs-angle curve, one line per T value, to
    see whether the network's rotation invariance (curve flatness /
    height) improves as more timesteps are allowed to accumulate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    t_values = sorted(t_results.keys())
    cmap = plt.cm.viridis
    for i, t in enumerate(t_values):
        angles = sorted(t_results[t].keys())
        accs = [t_results[t][a] * 100 for a in angles]
        ax.plot(angles, accs, marker="o", markersize=3, color=cmap(i / max(1, len(t_values) - 1)),
                 label=f"T={t}", alpha=0.85)

    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Accuracy (%), decoded from first T timesteps")
    ax.set_title("Temporal evolution of invariance: accuracy vs angle at increasing T")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved temporal-evolution-of-invariance plot to {plot_out}")


# --------------------------------------------------------------------------
# 5.8: direct equivariance error
# --------------------------------------------------------------------------

@torch.no_grad()
def equivariance_error(model, x, angle):
    """Section 5.8: ||f(Rx) - rho(R)f(x)||, measured directly on the
    model's FINAL output (post-classifier, pre-argmax logits/currents).

    Since the classifier output is meant to be INVARIANT (not merely
    equivariant) under rotation -- the group-pooling layer's whole
    purpose is to remove orientation dependence before the linear head --
    the correct comparison here is f(Rx) vs f(x) directly (rho(R) = identity
    on the invariant output), NOT a rotated-and-shifted comparison as
    would be needed for an equivariant (not invariant) intermediate
    representation. This matches how accuracy is actually evaluated
    elsewhere in this codebase (compare model output at angle theta
    directly against model output at 0 degrees).

    Returns: mean per-sample L2 norm ||f(Rx) - f(x)|| across the batch,
    as a plain float, plus a normalized fraction obtained by dividing by
    the mean norm of f(x) itself. The normalized ratio is generally the
    more comparable-across-models number, since raw output magnitude can
    differ between configurations.
    """
    model.eval()
    x_rot = _rotate_tensor_batch(x, angle)

    out_0 = model(x)
    out_theta = model(x_rot)
    # SpikingP4CNN's forward returns just the output tensor by default
    # (record_activations=False), so out_0/out_theta are already the
    # final logits here for both P4CNN and SpikingP4CNN.

    diff_norm = (out_theta - out_0).norm(dim=1)         # (B,)
    base_norm = out_0.norm(dim=1) + 1e-8                # (B,)

    return {
        "mean_abs_error": diff_norm.mean().item(),
        "mean_relative_error": (diff_norm / base_norm).mean().item(),
    }


def equivariance_error_sweep(model, x, angles):
    """Run equivariance_error at every angle in `angles`.
    Returns: dict {angle: {"mean_abs_error":..., "mean_relative_error":...}}.
    """
    return {angle: equivariance_error(model, x, angle) for angle in angles}


def plot_equivariance_error(err_by_config, plot_out="plots/equivariance_error.png"):
    """Plot mean relative equivariance error vs angle, one line per
    configuration, for direct cross-configuration comparison. This is
    the plot Experiment 5.8 is built around, and is arguably the single
    cheapest, most theory-connected plot in the whole Section 5 list --
    no retraining, no sweep infrastructure beyond a forward pass at each
    angle, and it measures exactly the quantity Section 9's theoretical
    program (equivariance of spike dynamics, equivariance of spike-count
    representations) is trying to explain.

    err_by_config: dict {config_label: {angle: {"mean_relative_error":...}}}
                   i.e. one entry per config, each itself the output of
                   equivariance_error_sweep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, err_by_angle in err_by_config.items():
        angles = sorted(err_by_angle.keys())
        errs = [err_by_angle[a]["mean_relative_error"] for a in angles]
        ax.plot(angles, errs, marker="o", markersize=4, label=label, alpha=0.85)

    for group_angle in (0, 90, 180, 270):
        ax.axvline(group_angle, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Test rotation angle (degrees)")
    ax.set_ylabel("Mean relative equivariance error  $\\|f(Rx)-f(x)\\| / \\|f(x)\\|$")
    ax.set_title("Direct equivariance error vs rotation angle")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved equivariance error plot to {plot_out}")


# --------------------------------------------------------------------------
# Orchestration: run EVERYTHING (no-retraining items) in one call
# --------------------------------------------------------------------------

def run_full_analysis(model, x, images_np, labels_np, angles=None, t_values=None,
                        plot_dir="plots", label="model", is_spiking=True):
    """Runs every no-retraining analysis in this module on a single
    trained model, in one call, and saves every associated plot.

    model:      a loaded SpikingP4CNN or P4CNN (eval mode)
    x:          (B, 1, H, W) tensor batch, upright images in [0,1], used
                for the representation-based analyses (4.1-4.5, 5.1, 5.3,
                5.4, 5.8's per-batch version)
    images_np:  full numpy test set (N, H, W) in [0,1], used for the
                accuracy-based T-sweep (5.7), which needs a proper-sized
                evaluation set, not just one small batch
    labels_np:  matching (N,) integer labels
    angles:     angles to sweep; defaults to every 15 degrees, 0-345
    t_values:   T values for the 5.7 sweep; defaults to [1,5,10,20,50]
                capped at model.T if the model is spiking
    plot_dir:   directory to save all plots into (prefixed with `label`)
    is_spiking: set False for the analog P4CNN, which skips 5.4, 5.7, and
                the temporal parts of 4.5 (no time axis to study)
    """
    from pathlib import Path
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if angles is None:
        angles = list(range(0, 360, 15))
    nonzero_angles = [a for a in angles if a != 0]

    print(f"=== Running full representation analysis for '{label}' ===")

    print("[5.1] layer-wise similarity (cosine, CKA, SVCCA)...")
    plot_layerwise_similarity_all_metrics(
        model, x, angles, plot_out=str(plot_dir / f"{label}_5.1_layerwise_similarity.png")
    )

    print("[5.3] rotation sensitivity distribution...")
    all_reps = rotation_sweep_representations(model, x, angles)
    R = rotation_sensitivity(all_reps)
    plot_rotation_sensitivity_distribution(
        R, plot_out=str(plot_dir / f"{label}_5.3_rotation_sensitivity.png")
    )

    print("[5.8] equivariance error sweep...")
    err_sweep = equivariance_error_sweep(model, x, angles)
    err_by_config = {label: err_sweep}
    plot_equivariance_error(
        err_by_config, plot_out=str(plot_dir / f"{label}_5.8_equivariance_error.png")
    )

    if is_spiking:
        print("[5.4] rotation sensitivity through time...")
        all_temporal = nonzero_angles + [0]
        R_t = rotation_sensitivity_through_time(model, x, all_temporal)
        plot_rotation_sensitivity_through_time(
            R_t, plot_out=str(plot_dir / f"{label}_5.4_rotation_sensitivity_through_time.png")
        )

        print("[5.7] temporal evolution of invariance (T sweep)...")
        if t_values is None:
            t_values = sorted(set([t for t in [1, 5, 10, 20, 50] if t <= model.T] + [model.T]))
        device = next(model.parameters()).device
        t_results = evaluate_accuracy_at_truncated_T(
            model, images_np, labels_np, angles, t_values, device=device
        )
        plot_temporal_evolution_of_invariance(
            t_results, plot_out=str(plot_dir / f"{label}_5.7_temporal_evolution.png")
        )
    else:
        print("[5.4, 5.7] skipped -- analog model has no time axis to study.")

    print(f"=== Done. All plots written to {plot_dir}/{label}_*.png ===")


if __name__ == "__main__":
    # confirms the machinery runs end-to-end. Replace with a real
    # checkpoint path to analyze actual trained results.
    torch.manual_seed(0)
    model = SpikingP4CNN(in_channels=1, n_classes=4, channels=10, T=10, mode="alternating")
    model.eval()

    x = torch.rand(8, 1, 28, 28)

    print("Testing get_layer_representations...")
    reps = get_layer_representations(model, x, angle=0)
    for l, r in reps.items():
        print(f"  layer {l+1}: {tuple(r.shape)}")

    print("\nTesting layerwise_similarity (cosine)...")
    sims = layerwise_similarity(model, x, angle=45, metric="cosine")
    print(f"  cosine similarity per layer (0 vs 45 deg): {[f'{s:.3f}' for s in sims]}")

    print("\nTesting layerwise_similarity (CKA)...")
    sims_cka = layerwise_similarity(model, x, angle=45, metric="cka")
    print(f"  CKA similarity per layer (0 vs 45 deg): {[f'{s:.3f}' for s in sims_cka]}")

    print("\nTesting rotation_sensitivity...")
    all_reps = rotation_sweep_representations(model, x, angles=[0, 45, 90, 135])
    R = rotation_sensitivity(all_reps)
    for l, r in R.items():
        print(f"  layer {l+1}: R_i shape {tuple(r.shape)}, mean={r.mean().item():.4f}")

    print("\nAll Section 4 infrastructure smoke tests passed.")