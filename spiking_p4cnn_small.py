"""
spiking_p4cnn_small.py

Reduced-capacity (5-layer, 4-channel, ~2.4k-param) spiking counterpart of
p4cnn_small.py -- SAME reduced backbone, with a configurable per-layer
choice of spiking LIF (snntorch.Leaky, fast-sigmoid surrogate gradient) vs.
analog ReLU, exactly mirroring the mode system used for the full-size
(7-layer, 10-channel, 24,744-param) SpikingP4CNN in the existing findings
report.

Five configs, matching the ones requested for this ablation:

    mode="none"          -- all 5 layers analog (ReLU)  [= P4CNNSmall itself]
    mode="alternating"   -- spiking at layers 2, 4        (odd layers analog)
    mode="full"          -- all 5 layers spiking
    mode="hybrid_early"  -- spiking at layers 1, 2         (near input)
    mode="hybrid_late"   -- spiking at layer 5 only        (near output)

Layer numbering matches p4cnn_small.py's L1..L5 (L1 = P4ConvZ2 lifting
layer nearest input, L5 = final P4ConvP4 layer nearest the classifier).

All five configs share IDENTICAL architecture and parameter count
(2,420 params) -- only which layers use LIF vs. ReLU differs, exactly as
in the full-size 24,744-param comparison, so the ONLY variable across
configs is spike placement, and the only variable relative to the
existing findings report is capacity (depth 7->5, width 10->4,
params 24,744->2,420, ~10.2x fewer parameters).
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from gconv_small import P4ConvZ2, P4ConvP4, P4BatchNorm2d, P4GroupPool

NUM_CHANNELS = 4

MODES = {
    "none":          set(),              # Analog control (no spiking layers)
    "alternating":   {2, 4},
    "full":          {1, 2, 3, 4, 5},
    "hybrid_early":  {1, 2},
    "hybrid_late":   {5},
}


class SpikingP4CNNSmall(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, channels=NUM_CHANNELS,
                 mode="full", beta=0.9, threshold=1.0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        self.mode = mode
        self.spiking_layers = MODES[mode]
        spike_grad = surrogate.fast_sigmoid()

        self.conv1 = P4ConvZ2(in_channels, channels, kernel_size=3)
        self.bn1 = P4BatchNorm2d(channels)

        self.conv2 = P4ConvP4(channels, channels, kernel_size=3)
        self.bn2 = P4BatchNorm2d(channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = P4ConvP4(channels, channels, kernel_size=3)
        self.bn3 = P4BatchNorm2d(channels)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4 = P4ConvP4(channels, channels, kernel_size=3)
        self.bn4 = P4BatchNorm2d(channels)

        self.conv5 = P4ConvP4(channels, channels, kernel_size=3)
        self.bn5 = P4BatchNorm2d(channels)

        self.group_pool = P4GroupPool()
        self.classifier = nn.Linear(channels, num_classes)
        self.relu = nn.ReLU(inplace=True)

        # one LIF neuron module per potentially-spiking layer; unused ones
        # (layers not in self.spiking_layers for this mode) are simply
        # never called in forward(), so param count stays identical across
        # modes for the conv/bn/classifier weights that matter for the
        # "identical parameter count" comparison in the findings report.
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=False)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=False)
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=False)

    def _reset_mem(self):
        # snntorch.Leaky silently persists membrane state across separate
        # forward() calls even with init_hidden=False; without this reset,
        # the 2nd training batch crashes with "backward through graph a
        # second time". Same bug/fix as documented for the full-size model.
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.lif4.init_leaky()
        self.mem5 = self.lif5.init_leaky()

    def _layer(self, idx, conv, bn, h, pool=None):
        h = bn(conv(h))
        if idx in self.spiking_layers:
            lif = getattr(self, f"lif{idx}")
            mem = getattr(self, f"mem{idx}")
            spk, mem = lif(h, mem)
            setattr(self, f"mem{idx}", mem)
            h = spk
        else:
            h = self.relu(h)
        if pool is not None:
            h = pool(h)
        return h

    def forward(self, x, T=50, record_activations=False):
        """x: (B, 1, 28, 28) static image, Poisson rate-coded into T spike
        frames internally. Returns accumulated-spike-count logits
        (summed over T, matching a_i(x) = sum_t S_i(t) from the notation
        section of the findings report), or (logits, activations) if
        record_activations=True, where activations[layer_idx] has shape
        (T, B, C, 4, H, W).
        """
        self._reset_mem()
        acts = {i: [] for i in range(1, 6)} if record_activations else None
        out_accum = 0.0

        for t in range(T):
            # Poisson rate coding: probability of a spike at this pixel/timestep
            # is proportional to pixel intensity in [0, 1].
            spk_in = torch.bernoulli(x.clamp(0, 1))

            h = self._layer(1, self.conv1, self.bn1, spk_in)
            if record_activations:
                acts[1].append(h)
            h = self._layer(2, self.conv2, self.bn2, h, pool=self.pool1)
            if record_activations:
                acts[2].append(h)
            h = self._layer(3, self.conv3, self.bn3, h, pool=self.pool2)
            if record_activations:
                acts[3].append(h)
            h = self._layer(4, self.conv4, self.bn4, h)
            if record_activations:
                acts[4].append(h)
            h = self._layer(5, self.conv5, self.bn5, h)
            if record_activations:
                acts[5].append(h)

            h = self.group_pool(h).flatten(1)
            out_accum = out_accum + self.classifier(h)

        if record_activations:
            acts = {k: torch.stack(v, dim=0) for k, v in acts.items()}
            return out_accum, acts
        return out_accum


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    for mode in MODES:
        m = SpikingP4CNNSmall(num_classes=4, mode=mode)
        x = torch.rand(2, 1, 28, 28)
        y = m(x, T=5)
        print(f"{mode:14s} output {tuple(y.shape)}  params {count_params(m)}")
