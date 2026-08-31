"""
Spiking variants of the P4CNN architecture (Cohen & Welling rotated-MNIST
structure), built on the equivariance-corrected primitives in gconv.py
(P4ConvZ2, P4ConvP4, P4BatchNorm2d, P4GroupPool).

Two configs, matching the report's naming convention:

  SpikingP4CNN(mode="full")
      Every layer's activation is replaced by a LIF neuron (snntorch.Leaky).
      Input is Poisson rate-encoded into a spike train once, of length T.
      Output is decoded as the mean spike count of the final classifier
      layer's output across all T timesteps.

  SpikingP4CNN(mode="alternating")
      Alternates analog (ReLU) and spiking (LIF) layers through the depth
      of the network: layer 1 analog, layer 2 spiking, layer 3 analog,
      layer 4 spiking, ... matching "Hybrid Alternating" from the report,
      which was found to be the most robust rotation-invariant config
      (flat curve across 0-180 degrees) -- analog layers inside the time
      loop are hypothesized to act as regularizers on the spiking dynamics.

Both share the SAME underlying P4ConvZ2/P4ConvP4/P4BatchNorm2d layer
definitions and channel counts as the analog-only P4CNN in p4cnn.py, so
comparisons between configs isolate the effect of spiking vs analog
units specifically, not confounds from different architectures.

IMPORTANT ON EFFICIENCY: the conv layers themselves are only computed
ONCE per timestep (not redundantly recomputed), and the same conv/BN
weight objects are reused across all T timesteps -- only the LIF neuron
state (membrane potential) evolves across time. This is the standard,
efficient way to implement a surrogate-gradient SNN and is much faster
than the naive "rerun the whole network T times independently" proxy
used earlier just to produce a rough timing estimate.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from gconv import P4ConvZ2, P4ConvP4, P4BatchNorm2d, P4GroupPool


def poisson_encode(x, T):
    """Rate-encode a static image batch into a spike train of length T.

    x: (B, C, H, W) in [0, 1] (already normalized pixel intensities).
    Returns: (T, B, C, H, W) binary spike tensor -- at each timestep,
    each pixel independently "fires" with probability equal to its
    intensity. This is the standard Poisson/rate encoding used in the
    internship report; using this (rather than direct/constant-current
    encoding) is important -- the report found direct encoding at T=30
    caused fixed-rate LIF oscillation that collapsed the temporal
    dimension, hurting performance.
    """
    x = x.clamp(0, 1)
    spikes = torch.rand(T, *x.shape, device=x.device) < x.unsqueeze(0)
    return spikes.float()


class SpikingP4CNN(nn.Module):
    """
    mode="full":         every layer spiking (LIF after every conv+BN)
    mode="alternating":  alternates analog (ReLU) / spiking (LIF) by depth,
                          starting with analog at layer 1
    mode="hybrid_early":  layers 1-2 spiking, layers 3-7 analog
    mode="hybrid_mid":    layers 3-5 spiking, layers 1-2 and 6-7 analog
    mode="hybrid_late":   layers 6-7 spiking, layers 1-5 analog
                          (matches the report's most dramatic prior result:
                          collapse to chance at non-group-element angles)
    """

    MODES = ("full", "alternating", "hybrid_early", "hybrid_mid", "hybrid_late")

    def __init__(self, in_channels=1, n_classes=10, channels=10, T=50,
                 beta=0.9, threshold=1.0, mode="full"):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        self.mode = mode
        self.T = T
        C = channels

        spike_grad = surrogate.fast_sigmoid(slope=25)  # standard surrogate gradient for LIF backprop

        # 7 conv+BN layers, same structure/channel counts as the analog P4CNN
        self.conv1 = P4ConvZ2(in_channels, C, kernel_size=3, stride=1, padding=0)
        self.bn1 = P4BatchNorm2d(C)
        self.conv2 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn2 = P4BatchNorm2d(C)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn3 = P4BatchNorm2d(C)
        self.conv4 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn4 = P4BatchNorm2d(C)
        self.conv5 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn5 = P4BatchNorm2d(C)
        self.conv6 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn6 = P4BatchNorm2d(C)
        self.conv7 = P4ConvP4(C, C, kernel_size=4, stride=1, padding=0)
        self.bn7 = P4BatchNorm2d(C)

        self.group_pool = P4GroupPool(channels=C, mode="max")
        self.fc = nn.Linear(C, n_classes)

        # Per-layer activation: LIF (spiking) or ReLU (analog), decided by mode.
        # Indices 0-6 correspond to conv layers 1-7.
        if mode == "full":
            self.layer_is_spiking = [True] * 7
        elif mode == "alternating":
            # layers 1,3,5,7 analog; layers 2,4,6 spiking
            self.layer_is_spiking = [False, True, False, True, False, True, False]
        elif mode == "hybrid_early":
            # layers 1-2 spiking, layers 3-7 analog
            self.layer_is_spiking = [True, True, False, False, False, False, False]
        elif mode == "hybrid_mid":
            # layers 3-5 spiking, layers 1-2 and 6-7 analog
            self.layer_is_spiking = [False, False, True, True, True, False, False]
        else:  # hybrid_late
            # layers 6-7 spiking, layers 1-5 analog
            self.layer_is_spiking = [False, False, False, False, False, True, True]

        self.lifs = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=False)
            if spiking else nn.Identity()
            for spiking in self.layer_is_spiking
        ])
        self.relu = nn.ReLU(inplace=True)

    def _apply_activation(self, layer_idx, x, mem_state):
        """Apply either LIF (returning new spike + membrane state) or ReLU
        (stateless, mem_state passed through unchanged) for a given layer."""
        if self.layer_is_spiking[layer_idx]:
            lif = self.lifs[layer_idx]
            if mem_state[layer_idx] is None:
                spk, mem = lif(x)
            else:
                spk, mem = lif(x, mem_state[layer_idx])
            mem_state[layer_idx] = mem
            return spk
        else:
            return self.relu(x)

    def forward(self, x, record_activations=False):
        """x: (B, in_channels, H, W) static image, pixel values in [0,1].

        Encodes to a Poisson spike train of length self.T, runs the whole
        network once per timestep (conv/BN weights shared and only computed
        once per timestep -- not redundantly repeated), and averages the
        final layer's spike output across time as the decoded prediction.

        record_activations: if True, also returns a dict of per-layer,
        per-timestep activations -- this is the h_l(t,x) / a_i(x) machinery
        from Section 4 of the research plan (layer representations, neuron
        responses, temporal representations). Off by default so normal
        training/eval is unaffected; only turn on for analysis passes on
        already-trained checkpoints (it costs extra memory to store T
        timesteps x 7 layers of activations, and detaching them from the
        graph -- see below -- means it must NOT be used mid-training).
        """
        for lif in self.lifs:
            if isinstance(lif, snn.Leaky):
                lif.reset_mem()

        B = x.shape[0]
        spike_train = poisson_encode(x, self.T)  # (T, B, C, H, W)

        mem_state = [None] * 7
        out_accum = 0.0

        # activations[layer_idx] -> list of length T, each entry the
        # (B, C, 4, H, W)-shaped post-activation tensor at that timestep
        # (spikes for LIF layers, ReLU output for analog layers -- both are
        # valid choices of a_i(t,x) per Section 4.2's discussion, and using
        # the same post-activation quantity for both keeps analog and
        # spiking layers directly comparable in the same representation).
        activations = {i: [] for i in range(7)} if record_activations else None

        for t in range(self.T):
            xt = spike_train[t]

            h = self.bn1(self.conv1(xt))
            h = self._apply_activation(0, h, mem_state)
            if record_activations:
                activations[0].append(h.detach())

            h = self.bn2(self.conv2(h))
            h = self._apply_activation(1, h, mem_state)
            if record_activations:
                activations[1].append(h.detach())

            h = self.pool(h)

            h = self.bn3(self.conv3(h))
            h = self._apply_activation(2, h, mem_state)
            if record_activations:
                activations[2].append(h.detach())

            h = self.bn4(self.conv4(h))
            h = self._apply_activation(3, h, mem_state)
            if record_activations:
                activations[3].append(h.detach())

            h = self.bn5(self.conv5(h))
            h = self._apply_activation(4, h, mem_state)
            if record_activations:
                activations[4].append(h.detach())

            h = self.bn6(self.conv6(h))
            h = self._apply_activation(5, h, mem_state)
            if record_activations:
                activations[5].append(h.detach())

            h = self.bn7(self.conv7(h))
            h = self._apply_activation(6, h, mem_state)
            if record_activations:
                activations[6].append(h.detach())

            h = self.group_pool(h)       # (B, C, 1, 1)
            h = h.flatten(1)             # (B, C)
            out_t = self.fc(h)           # (B, n_classes) -- logits/current at this timestep

            out_accum = out_accum + out_t

        out = out_accum / self.T

        if record_activations:
            # stack each layer's per-timestep list into a single tensor:
            # (T, B, C_out, 4, H, W) -- this IS h_l(t,x) from Section 4.5.
            # Summing over dim 0 (T) gives accumulated spike count = h_l(x)
            # from Section 4.1. Both are trivial to derive from this one
            # returned structure, so record_activations=True unlocks 4.1,
            # 4.2, and 4.5 simultaneously.
            stacked = {i: torch.stack(activations[i], dim=0) for i in activations}
            return out, stacked

        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for mode in ("full", "alternating"):
        m = SpikingP4CNN(in_channels=1, n_classes=4, channels=10, T=10, mode=mode)
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.rand(2, 1, 28, 28)  # pixel intensities in [0,1] for Poisson encoding
        y = m(x)
        print(f"mode={mode}: params={n_params}, output shape={tuple(y.shape)}")