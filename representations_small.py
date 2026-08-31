"""
representations_small.py

Shared utilities for the notation-based representation analysis of the
REDUCED-CAPACITY (5-layer, 4-channel, 2,420-param) SpikingP4CNNSmall model.

Every quantity here matches a symbol defined in sec_notation.tex of the
findings report exactly:

    a_i(x)              -- accumulated spike count of neuron i (Eq. accumcount)
    h_ell(x)             -- layer-ell representation = vector of a_i(x) over
                             all neurons i in that layer
    R_i = Var_theta(a_i(R_theta x))   -- rotation sensitivity of neuron i (Eq. Ri)
    cosine similarity, CKA, SVCCA     -- representation similarity measures
                                         (sec:simmetrics)
    h_ell(t,x)            -- instantaneous (per-timestep) representation
    R_i(t)               -- rotation sensitivity at timestep t (temporal)
    Equivariance error(theta) = ||f(R_theta x) - f(x)|| / ||f(x)||   (Eq. eqerr)
    Truncated-T accuracy  -- decode from first T' timesteps of a T=50-trained
                             model, no retraining

This file only defines functions; the plot_*.py scripts import from here.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spiking_p4cnn_small import SpikingP4CNNSmall, MODES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANGLES = list(range(0, 360, 15))  # 24 angles, matches ANGLES in train_spiking_small.py
DIGITS = [3, 4, 8, 9]


# ---------------------------------------------------------------------------
# Data loading (mirrors train_spiking_small.py's DIGITS / rotation convention)
# ---------------------------------------------------------------------------

def load_probe_images(n_per_class=4, seed=0):
    """Loads a small, fixed probe set of upright images (n_per_class per
    digit class) from the same MNIST CSV used for training, for use as the
    x in all of the R_theta x comparisons below. A small, fixed probe set
    is standard for this kind of analysis (see sec:finding1/2 setup) --
    we are studying the geometry of the learned representation, not
    re-estimating test accuracy.
    """
    import pandas as pd
    rng = np.random.RandomState(seed)
    data_path = os.path.expanduser("~/Downloads/digit-recognizer/train.csv")
    df = pd.read_csv(data_path)
    imgs, labels = [], []
    for d in DIGITS:
        sub = df[df["label"] == d]
        idx = rng.choice(len(sub), size=n_per_class, replace=False)
        for i in idx:
            row = sub.iloc[i]
            img = row.drop("label").values.astype(np.float32).reshape(28, 28) / 255.0
            imgs.append(img)
            labels.append(DIGITS.index(d))
    imgs = np.stack(imgs)  # (N, 28, 28)
    labels = np.array(labels)
    return imgs, labels


def rotate_batch(imgs_np, theta_deg):
    """Rotates a batch of (N, 28, 28) numpy images by theta_deg using
    bilinear interpolation, matching the rotation convention used for
    training/eval in train_spiking_small.py."""
    x = torch.from_numpy(imgs_np).float().unsqueeze(1)  # (N, 1, 28, 28)
    theta = torch.tensor(np.deg2rad(theta_deg))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    N = x.shape[0]
    aff = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]]).unsqueeze(0).repeat(N, 1, 1).float()
    grid = F.affine_grid(aff, x.shape, align_corners=False)
    x_rot = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    return x_rot.squeeze(1).numpy()


def load_model(mode, seed, models_dir="models_small"):
    ckpt_path = os.path.join(models_dir, f"{mode}_seed{seed}_small.pt")
    model = SpikingP4CNNSmall(num_classes=len(DIGITS), mode=mode).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Forward pass returning accumulated + per-timestep activations for all layers
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_with_activations(model, x_np, T=50, seed=None):
    """x_np: (N, 28, 28) numpy in [0,1]. Returns:
       logits: (N, num_classes)
       h: dict layer_idx -> (N, n_i) accumulated spike-count / activation
          vector per neuron (flattened over channel, orientation, H, W) --
          this is exactly h_ell(x) from sec:layerrep.
       h_t: dict layer_idx -> (T, N, n_i) instantaneous representation
          h_ell(t,x) from sec:temporal.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.from_numpy(x_np).float().unsqueeze(1).to(DEVICE)  # (N,1,28,28)
    logits, acts = model(x, T=T, record_activations=True)
    h, h_t = {}, {}
    for layer_idx, a in acts.items():
        # a: (T, N, C, 4, H, W)
        Tt, N = a.shape[0], a.shape[1]
        flat = a.reshape(Tt, N, -1)          # (T, N, n_i)
        h_t[layer_idx] = flat.cpu().numpy()
        h[layer_idx] = flat.sum(dim=0).cpu().numpy()  # accumulate over T -> a_i(x)
    return logits.cpu().numpy(), h, h_t


# ---------------------------------------------------------------------------
# Rotation sensitivity: R_i = Var_theta(a_i(R_theta x))   (Eq. Ri)
# ---------------------------------------------------------------------------

def rotation_sensitivity(model, imgs_np, T=50, angles=ANGLES, per_timestep=False, seed=0):
    """Returns dict layer_idx -> R_i array of shape (n_i,), R_i averaged
    over the probe images (one Var_theta per image, then meaned across
    images, matching sec:finding2's per-neuron reporting convention).
    If per_timestep=True, additionally returns dict layer_idx -> (T, n_i)
    array of R_i(t) (sec:temporal), median over probe images.
    """
    per_theta_h = {}   # layer -> list over theta of (N, n_i)
    per_theta_ht = {}  # layer -> list over theta of (T, N, n_i)
    for theta in angles:
        rot = rotate_batch(imgs_np, theta)
        _, h, h_t = forward_with_activations(model, rot, T=T, seed=seed)
        for layer_idx, v in h.items():
            per_theta_h.setdefault(layer_idx, []).append(v)
        if per_timestep:
            for layer_idx, v in h_t.items():
                per_theta_ht.setdefault(layer_idx, []).append(v)

    R = {}
    for layer_idx, lst in per_theta_h.items():
        arr = np.stack(lst, axis=0)              # (n_theta, N, n_i)
        var_per_image = arr.var(axis=0)           # (N, n_i), Var_theta per image per neuron
        R[layer_idx] = var_per_image.mean(axis=0)  # (n_i,), averaged over probe images

    R_t = None
    if per_timestep:
        R_t = {}
        for layer_idx, lst in per_theta_ht.items():
            arr = np.stack(lst, axis=0)               # (n_theta, T, N, n_i)
            var_per_image = arr.var(axis=0)             # (T, N, n_i)
            R_t[layer_idx] = np.median(var_per_image, axis=1)  # (T, n_i), median over images

    return R, R_t


# ---------------------------------------------------------------------------
# Representation similarity: cosine, CKA, SVCCA  (sec:simmetrics)
# ---------------------------------------------------------------------------

def cosine_similarity_batch(h_x, h_rx):
    """h_x, h_rx: (N, n_i). Returns mean cosine similarity over the N
    image/rotated-image pairs."""
    num = (h_x * h_rx).sum(axis=1)
    denom = np.linalg.norm(h_x, axis=1) * np.linalg.norm(h_rx, axis=1) + 1e-8
    return float(np.mean(num / denom))


def linear_cka(h_x, h_rx):
    """Batch-side Gram matrix CKA, per the implementation note in
    sec:simmetrics: for layer 1 the neuron-space Gram matrix is
    infeasible, so we use the mathematically-equivalent batch-side
    Gram formulation K = X X^T (N x N) instead of neuron-space (n_i x n_i).
    """
    def gram_centered(h):
        h = h - h.mean(axis=0, keepdims=True)
        K = h @ h.T
        N = K.shape[0]
        unit = np.ones((N, N)) / N
        Kc = K - unit @ K - K @ unit + unit @ K @ unit
        return Kc

    Kx = gram_centered(h_x)
    Ky = gram_centered(h_rx)
    hsic = np.sum(Kx * Ky)
    norm_x = np.sqrt(np.sum(Kx * Kx))
    norm_y = np.sqrt(np.sum(Ky * Ky))
    return float(hsic / (norm_x * norm_y + 1e-8))


def svcca(h_x, h_rx, n_components=None):
    """SVCCA: PCA-reduce both representations, then CCA-correlate.
    Caution (per sec:simmetrics): enforce >=10x batch/neuron ratio where
    possible; report values are NOT on an absolute 0-1 invariance scale
    -- compare relatively across layers/configs only.
    """
    N = h_x.shape[0]
    if n_components is None:
        n_components = max(1, min(N - 1, h_x.shape[1], h_rx.shape[1]))

    def pca_reduce(h, k):
        h = h - h.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        k = min(k, Vt.shape[0])
        return h @ Vt[:k].T  # (N, k)

    hx_r = pca_reduce(h_x, n_components)
    hy_r = pca_reduce(h_rx, n_components)

    # CCA via QR + SVD (standard SVCCA implementation)
    def qr_ortho(a):
        q, _ = np.linalg.qr(a)
        return q

    qx = qr_ortho(hx_r)
    qy = qr_ortho(hy_r)
    k = min(qx.shape[1], qy.shape[1])
    M = qx[:, :k].T @ qy[:, :k]
    svals = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(svals))


def layerwise_similarity(model, imgs_np, angles=ANGLES, T=50, seed=0, metrics=("cosine", "cka", "svcca")):
    """For each layer, for each nonzero rotation angle theta (theta=0 is
    trivially similarity=1 and excluded), computes similarity between
    h_ell(x) and h_ell(R_theta x), then averages over theta and over the
    probe-image batch (batch-level metrics like CKA/SVCCA already
    aggregate over the batch; we average their single scalar over theta).
    Returns dict layer_idx -> {"cosine": v, "cka": v, "svcca": v}.
    """
    _, h0, _ = forward_with_activations(model, imgs_np, T=T, seed=seed)
    per_layer_metric_theta = {layer_idx: {m: [] for m in metrics} for layer_idx in h0}

    for theta in angles:
        if theta == 0:
            continue
        rot = rotate_batch(imgs_np, theta)
        _, h_rot, _ = forward_with_activations(model, rot, T=T, seed=seed)
        for layer_idx in h0:
            hx, hrx = h0[layer_idx], h_rot[layer_idx]
            if "cosine" in metrics:
                per_layer_metric_theta[layer_idx]["cosine"].append(cosine_similarity_batch(hx, hrx))
            if "cka" in metrics:
                per_layer_metric_theta[layer_idx]["cka"].append(linear_cka(hx, hrx))
            if "svcca" in metrics:
                per_layer_metric_theta[layer_idx]["svcca"].append(svcca(hx, hrx))

    result = {}
    for layer_idx, md in per_layer_metric_theta.items():
        result[layer_idx] = {m: float(np.mean(vals)) for m, vals in md.items()}
    return result


# ---------------------------------------------------------------------------
# Direct equivariance error(theta) = ||f(R_theta x) - f(x)|| / ||f(x)||  (Eq eqerr)
# ---------------------------------------------------------------------------

@torch.no_grad()
def equivariance_error_multidraw(model, imgs_np, theta, T=50, seed=0, n_draws=8):
    """Multi-draw equivariance error at a single angle theta (audit Fix 2).

    equivariance_error_curve / rotation_spread_and_e90's E90 both call
    forward_with_activations(..., seed=seed) for BOTH x and R_theta x with
    the SAME seed value. torch.manual_seed(seed) resets the global RNG to
    the same state for both calls, but since rotation moves image content
    between tensor positions, the same raw random stream gets thresholded
    against different per-pixel probabilities in each call -- it does not
    reproduce a matched spike encoding under rotation, only reproducibility
    run-to-run. The result is that a single-draw equivariance error at a
    group angle is dominated by Monte-Carlo spike-sampling variance, not a
    stable property of the trained model (see audit: re-drawing the
    encoding seed alone, no rotation or config change, reproduces the
    reported group-angle magnitudes and ordering).

    This function draws the Bernoulli spike encoding n_draws independent
    times for the (x, R_theta x) comparison -- each draw uses a genuinely
    different encoding seed (seed*1000 + draw_index), never the same
    torch.manual_seed value reused -- and returns the mean and std of the
    resulting per-draw error, so the Monte-Carlo floor is quantified rather
    than trusted as a single number. Uses the same softmax-probability
    formula as equivariance_error_curve for direct comparability with
    existing single-draw results.

    Returns (mean, std, list_of_per_draw_values).
    """
    rot = rotate_batch(imgs_np, theta)
    draws = []
    for k in range(n_draws):
        draw_seed = seed * 1000 + k
        logits0, _, _ = forward_with_activations(model, imgs_np, T=T, seed=draw_seed)
        probs0 = torch.softmax(torch.from_numpy(logits0), dim=1).numpy()
        logits_r, _, _ = forward_with_activations(model, rot, T=T, seed=draw_seed)
        probs_r = torch.softmax(torch.from_numpy(logits_r), dim=1).numpy()
        num = np.linalg.norm(probs_r - probs0, axis=1)
        denom = np.linalg.norm(probs0, axis=1) + 1e-8
        draws.append(float(np.mean(num / denom)))
    draws_arr = np.array(draws)
    return float(draws_arr.mean()), float(draws_arr.std()), draws


@torch.no_grad()
def equivariance_error_curve(model, imgs_np, angles=ANGLES, T=50, seed=0):
    """Returns dict theta -> mean relative error over the probe batch."""
    logits0, _, _ = forward_with_activations(model, imgs_np, T=T, seed=seed)
    probs0 = torch.softmax(torch.from_numpy(logits0), dim=1).numpy()

    errs = {}
    for theta in angles:
        rot = rotate_batch(imgs_np, theta)
        logits_r, _, _ = forward_with_activations(model, rot, T=T, seed=seed)
        probs_r = torch.softmax(torch.from_numpy(logits_r), dim=1).numpy()
        num = np.linalg.norm(probs_r - probs0, axis=1)
        denom = np.linalg.norm(probs0, axis=1) + 1e-8
        errs[theta] = float(np.mean(num / denom))
    return errs


# ---------------------------------------------------------------------------
# Truncated-T accuracy: decode from first T' timesteps, no retraining
# ---------------------------------------------------------------------------

@torch.no_grad()
def truncated_T_accuracy(model, imgs_np, labels_np, angles=ANGLES,
                          T_full=50, T_primes=(1, 5, 10, 20, 50), seed=0):
    """Trains once at T_full (model is already trained); at eval time,
    truncates the recorded per-timestep classifier contribution to the
    first T' steps and re-sums, WITHOUT re-running the network -- exactly
    Experiment 5.7 / sec:truncT. Returns dict T' -> mean accuracy over
    (probe images x all rotation angles).
    """
    if seed is not None:
        torch.manual_seed(seed)
    acc_by_Tprime = {tp: [] for tp in T_primes}

    for theta in angles:
        rot = rotate_batch(imgs_np, theta)
        x = torch.from_numpy(rot).float().unsqueeze(1).to(DEVICE)
        y = torch.from_numpy(labels_np).long()

        model._reset_mem()
        per_t_logits = []
        for t in range(T_full):
            spk_in = torch.bernoulli(x.clamp(0, 1))
            h = model._layer(1, model.conv1, model.bn1, spk_in)
            h = model._layer(2, model.conv2, model.bn2, h, pool=model.pool1)
            h = model._layer(3, model.conv3, model.bn3, h, pool=model.pool2)
            h = model._layer(4, model.conv4, model.bn4, h)
            h = model._layer(5, model.conv5, model.bn5, h)
            h = model.group_pool(h).flatten(1)
            per_t_logits.append(model.classifier(h))  # contribution at this t

        per_t_logits = torch.stack(per_t_logits, dim=0)  # (T_full, N, num_classes)
        for tp in T_primes:
            tp_clamped = min(tp, T_full)
            logits_trunc = per_t_logits[:tp_clamped].sum(dim=0)
            pred = logits_trunc.argmax(dim=1).cpu()
            acc = (pred == y).float().mean().item()
            acc_by_Tprime[tp].append(acc)

    return {tp: float(np.mean(v)) for tp, v in acc_by_Tprime.items()}


@torch.no_grad()
def truncated_T_accuracy_by_angle(model, imgs_np, labels_np, angles=ANGLES,
                                   T_full=50, T_primes=(1, 5, 10, 20, 50), seed=0):
    """Same recipe as truncated_T_accuracy(), but WITHOUT collapsing over
    rotation angle -- returns dict T' -> dict theta -> accuracy at that single
    angle, matching the findings report's Figure 42-50 style (accuracy vs.
    angle, one line per T', showing the W-shape switch on as T' grows),
    rather than truncated_T_accuracy()'s single angle-averaged curve.
    """
    if seed is not None:
        torch.manual_seed(seed)
    acc_by_Tprime_angle = {tp: {} for tp in T_primes}

    for theta in angles:
        rot = rotate_batch(imgs_np, theta)
        x = torch.from_numpy(rot).float().unsqueeze(1).to(DEVICE)
        y = torch.from_numpy(labels_np).long()

        model._reset_mem()
        per_t_logits = []
        for t in range(T_full):
            spk_in = torch.bernoulli(x.clamp(0, 1))
            h = model._layer(1, model.conv1, model.bn1, spk_in)
            h = model._layer(2, model.conv2, model.bn2, h, pool=model.pool1)
            h = model._layer(3, model.conv3, model.bn3, h, pool=model.pool2)
            h = model._layer(4, model.conv4, model.bn4, h)
            h = model._layer(5, model.conv5, model.bn5, h)
            h = model.group_pool(h).flatten(1)
            per_t_logits.append(model.classifier(h))

        per_t_logits = torch.stack(per_t_logits, dim=0)  # (T_full, N, num_classes)
        for tp in T_primes:
            tp_clamped = min(tp, T_full)
            logits_trunc = per_t_logits[:tp_clamped].sum(dim=0)
            pred = logits_trunc.argmax(dim=1).cpu()
            acc = (pred == y).float().mean().item()
            acc_by_Tprime_angle[tp][theta] = acc

    return acc_by_Tprime_angle


# ---------------------------------------------------------------------------
# Experiment 2 (Additional-Experiments-Spiked.pdf, sec 6.1-6.2): rotation
# spread D_s = max_theta A(theta) - min_theta A(theta), and E_90 = the
# direct-equivariance-error metric evaluated at exactly theta=90 degrees.
# Both are computed directly from a checkpoint + probe set here (rather than
# only from the already-saved rotation_sweep JSON) so that every seed has a
# value regardless of which stage of the original repr-analysis pipeline
# happened to complete for it.
# ---------------------------------------------------------------------------

@torch.no_grad()
def rotation_spread_and_e90(model, imgs_np, labels_np, angles=ANGLES, T=50, seed=0):
    """Returns (D_s, E90, mean_acc, upright_acc) for one trained model on one
    probe set. D_s = max_theta A(theta) - min_theta A(theta) (Eq. in sec 6.1).
    E90 = ||f(R_90 x) - f(x)|| / (||f(x)|| + eps), i.e. equivariance_error_curve
    evaluated at theta=90 specifically (rho(r) is trivial here since the
    output is a class-probability vector, not a group-indexed feature map --
    same convention as Eq. eqerr elsewhere in this file).
    """
    accs = {}
    logits0, _, _ = forward_with_activations(model, imgs_np, T=T, seed=seed)
    probs0 = torch.softmax(torch.from_numpy(logits0), dim=1).numpy()
    y = labels_np
    for theta in angles:
        rot = rotate_batch(imgs_np, theta)
        logits_r, _, _ = forward_with_activations(model, rot, T=T, seed=seed)
        pred = logits_r.argmax(axis=1)
        accs[theta] = float((pred == y).mean())
    mean_acc = float(np.mean(list(accs.values())))
    upright_acc = accs.get(0, None)
    D_s = max(accs.values()) - min(accs.values())

    rot90 = rotate_batch(imgs_np, 90)
    logits90, _, _ = forward_with_activations(model, rot90, T=T, seed=seed)
    probs90 = torch.softmax(torch.from_numpy(logits90), dim=1).numpy()
    num = np.linalg.norm(probs90 - probs0, axis=1)
    denom = np.linalg.norm(probs0, axis=1) + 1e-8
    E90 = float(np.mean(num / denom))
    return D_s, E90, mean_acc, upright_acc


def mean_firing_rate(model, imgs_np, T=50, seed=0):
    """Mean firing rate (mean accumulated spike count / T) averaged over all
    spiking layers of this model -- the optional Firing Rate column of
    Experiment 2's summary table. Returns None if the model has no spiking
    layers (mode='none')."""
    if not model.spiking_layers:
        return None
    _, h, _ = forward_with_activations(model, imgs_np, T=T, seed=seed)
    rates = [float(h[layer_idx].mean() / T) for layer_idx in sorted(model.spiking_layers)]
    return float(np.mean(rates))


# ---------------------------------------------------------------------------
# Experiment 3 (Additional-Experiments-Spiked.pdf, sec 6.3): layer/time
# representation distance
#   D_{ell,t}(theta) = E_x[ ||h_{ell,t}(R_theta x) - h_{ell,t}(x)||_2
#                            / (||h_{ell,t}(x)||_2 + eps) ]
# -- distinct from R_i(t) (per-neuron variance over 24 angles): this is a
# whole-layer, single-angle, normalized distance, matching the experiment doc
# exactly.
# ---------------------------------------------------------------------------

def _layer_reference_eps(a, floor_frac=0.01, min_eps=1e-8):
    """Layer-relative epsilon floor for a normalization denominator (audit
    fix, see representations_small.py Fix 1 notes below layer_time_distance).
    A single global eps=1e-8 is ~8 orders of magnitude too small at a narrow
    layer (e.g. layer 5, 16 units), where the true reference norm can be
    legitimately 0 for a nontrivial fraction of (timestep, image) cells --
    dividing by that floor rather than a real signal is what produces the
    ~1e7 blowups. eps = max(min_eps, floor_frac * median of the NONZERO
    reference norms at that layer), computed once from the reference
    (unrotated) batch `a` and reused for every timestep/image at that layer.
    """
    norms = np.linalg.norm(a, axis=2)  # (T, N)
    nonzero = norms[norms > 0]
    median_nonzero = float(np.median(nonzero)) if nonzero.size > 0 else min_eps
    return max(min_eps, floor_frac * median_nonzero)


@torch.no_grad()
def layer_time_distance(model, imgs_np, theta, T=50, seed=0,
                         floor_frac=0.01, min_eps=1e-8, exclude_thresh=1e-4,
                         return_diagnostics=False):
    """Returns dict layer_idx -> (T,) array of D_{ell,t}(theta), one value per
    timestep, averaged over the probe-image batch.

    Epsilon handling (audit fix): the denominator uses a per-layer
    eps = max(min_eps, floor_frac * median(nonzero reference norms at that
    layer)) -- see _layer_reference_eps -- computed once per layer from the
    reference (unrotated) batch, instead of a single global 1e-8.

    On top of the eps fix, as a cheap safety net, any (timestep, image) cell
    whose RAW reference norm (i.e. before eps is added) is below
    `exclude_thresh` is excluded from the per-timestep mean entirely -- the
    ratio there is dominated by floor noise, not signal, regardless of which
    eps is used, so it should not be trusted at all rather than merely
    downweighted. If every image at a given timestep is excluded, that
    timestep falls back to the unmasked mean (so the returned (T,) array
    never silently contains NaN).

    If return_diagnostics=True, also returns a second dict
    layer_idx -> {"eps": float, "n_excluded": int, "n_total": int} so the
    scale of the floor-collapse problem is visible per layer/config.
    """
    _, _, h_t0 = forward_with_activations(model, imgs_np, T=T, seed=seed)
    rot = rotate_batch(imgs_np, theta)
    _, _, h_t_rot = forward_with_activations(model, rot, T=T, seed=seed)

    result = {}
    diagnostics = {}
    for layer_idx in h_t0:
        a = h_t0[layer_idx]      # (T, N, n_i)
        b = h_t_rot[layer_idx]   # (T, N, n_i)
        num = np.linalg.norm(b - a, axis=2)           # (T, N)
        raw_denom = np.linalg.norm(a, axis=2)          # (T, N), BEFORE eps
        eps_layer = _layer_reference_eps(a, floor_frac=floor_frac, min_eps=min_eps)
        denom = raw_denom + eps_layer                  # (T, N)
        ratio = num / denom                             # (T, N)

        excluded = raw_denom < exclude_thresh           # (T, N)
        masked_ratio = np.where(excluded, np.nan, ratio)
        with np.errstate(invalid="ignore"):
            d = np.nanmean(masked_ratio, axis=1)        # (T,)
        all_excluded = np.isnan(d)
        if all_excluded.any():
            d[all_excluded] = ratio.mean(axis=1)[all_excluded]
        result[layer_idx] = d

        diagnostics[layer_idx] = {
            "eps": eps_layer,
            "n_excluded": int(excluded.sum()),
            "n_total": int(excluded.size),
        }

    if return_diagnostics:
        return result, diagnostics
    return result


# ---------------------------------------------------------------------------
# Experiment 4 (Additional-Experiments-Spiked.pdf, sec 6.6): fine-resolution
# (1-degree) rotation response + Fourier decomposition.
# ---------------------------------------------------------------------------

@torch.no_grad()
def rotation_response_sweep(model, imgs_np, labels_np, T=50, seed=0, angles_deg=None):
    """Evaluates margin M(theta) = z_y(R_theta x) - max_{c!=y} z_c(R_theta x)
    (mean over probe images) and accuracy A(theta), at fine angular
    resolution (default: every 1 degree, 0..359). Returns
    (angles_deg array, M array, A array). Uses RAW accumulated-spike-count
    logits for the margin (not softmax probabilities), matching the
    experiment doc's z_c(R_theta x) definition exactly.
    """
    if angles_deg is None:
        angles_deg = np.arange(360)
    if seed is not None:
        torch.manual_seed(seed)
    y = labels_np
    M = np.zeros(len(angles_deg))
    A = np.zeros(len(angles_deg))
    for i, theta in enumerate(angles_deg):
        rot = rotate_batch(imgs_np, float(theta))
        logits, _, _ = forward_with_activations(model, rot, T=T, seed=None)  # seed already set once, don't reset per angle
        N, C = logits.shape
        true_logit = logits[np.arange(N), y]
        masked = logits.copy()
        masked[np.arange(N), y] = -np.inf
        other_max = masked.max(axis=1)
        margin = true_logit - other_max
        M[i] = float(margin.mean())
        pred = logits.argmax(axis=1)
        A[i] = float((pred == y).mean())
    return angles_deg, M, A


def fourier_decompose(M, angles_deg, K_max=20):
    """Decomposes M(theta) (sampled at angles_deg, degrees, one full period)
    into a Fourier series M(theta) = a0 + sum_k [a_k cos(k theta) + b_k sin(k theta)]
    (theta in radians). Returns dict with a0, a (1..K_max), b (1..K_max),
    E_noninv, E_forbidden (Eqs. in sec 6.6.4).
    """
    N = len(angles_deg)
    theta_rad = np.deg2rad(angles_deg.astype(np.float64))
    a0 = float(M.mean())
    a = np.zeros(K_max + 1)
    b = np.zeros(K_max + 1)
    for k in range(1, K_max + 1):
        a[k] = (2.0 / N) * np.sum(M * np.cos(k * theta_rad))
        b[k] = (2.0 / N) * np.sum(M * np.sin(k * theta_rad))

    energy_k = a[1:] ** 2 + b[1:] ** 2  # index 0 here is k=1
    total_energy = a0 ** 2 + energy_k.sum()
    E_noninv = float(energy_k.sum() / total_energy) if total_energy > 0 else 0.0

    forbidden_mask = np.array([k % 4 != 0 for k in range(1, K_max + 1)])
    E_forbidden = float(energy_k[forbidden_mask].sum() / total_energy) if total_energy > 0 else 0.0

    return {
        "a0": a0, "a": a, "b": b,
        "E_noninv": E_noninv, "E_forbidden": E_forbidden,
        "K_max": K_max,
    }


# ---------------------------------------------------------------------------
# Small self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("representations_small.py loaded OK.")
    print("ANGLES:", ANGLES)
    print("Layers analyzed: 1..5 (matches SpikingP4CNNSmall L1..L5)")
