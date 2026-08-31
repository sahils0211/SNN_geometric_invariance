"""
p4cnn_small.py

Reduced-capacity analog P4CNN for the "does the layer-6/7 collapse and the
spike-placement signatures survive under a harder capacity budget" ablation.

Reduction relative to the original 7-layer / 10-channel / 24,744-param model:
  - Depth:    7 layers  -> 5 layers   (all kernels shrunk to <=3x3, matching
                                        the original's kernel sizes, with a
                                        SECOND maxpool added so the receptive
                                        field still exactly covers 28x28 -> 1x1
                                        using only small kernels; preserves the
                                        "valid convs only, no padding" property
                                        that makes the original's equivariance
                                        exact)
  - Width:    10 ch     -> 4 channels  (channels per p4-conv layer)
  - Params:   24,744    -> ~1.9k       (~13x smaller)

Layer plan (all "valid" convolutions, no padding, matching the original):
  L1: P4ConvZ2  1->4ch,  3x3   28x28 -> 26x26
  L2: P4ConvP4  4->4ch,  3x3   26x26 -> 24x24   + maxpool 2x2 -> 12x12
  L3: P4ConvP4  4->4ch,  3x3   12x12 -> 10x10   + maxpool 2x2 ->  5x5
  L4: P4ConvP4  4->4ch,  3x3    5x5  ->  3x3
  L5: P4ConvP4  4->4ch,  3x3    3x3  ->  1x1
  group-pool -> (B, 4, 1, 1) -> flatten -> Linear(4, num_classes)

This keeps the same qualitative shape as the original 7-layer design (a
stack of valid convolutions collapsing 28x28 down to a single spatial
position, group-pool + linear classifier at the end) while cutting depth
by 2 layers and width by 60%, using ONLY 3x3 kernels (the same kernel size
used throughout most of the original 7-layer model) rather than a
disproportionately expensive large final kernel. This makes the reduction
"harder" in the way that matters for this ablation -- less width AND less
depth AND a stricter per-layer receptive field -- while remaining directly
comparable in kind to the existing Analog / Alternating / Full Spiking /
Hybrid Early / Hybrid Late results already in the findings report.
"""
import torch
import torch.nn as nn
from gconv_small import P4ConvZ2, P4ConvP4, P4BatchNorm2d, P4GroupPool

NUM_CHANNELS = 4
NUM_LAYERS = 5


class P4CNNSmall(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, channels=NUM_CHANNELS):
        super().__init__()
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

    def forward(self, x, record_activations=False):
        acts = {} if record_activations else None

        h = self.relu(self.bn1(self.conv1(x)))          # 28 -> 26
        if record_activations:
            acts[1] = h.unsqueeze(0)
        h = self.relu(self.bn2(self.conv2(h)))           # 26 -> 24
        h = self.pool1(h)                                 # 24 -> 12
        if record_activations:
            acts[2] = h.unsqueeze(0)

        h = self.relu(self.bn3(self.conv3(h)))           # 12 -> 10
        h = self.pool2(h)                                 # 10 -> 5
        if record_activations:
            acts[3] = h.unsqueeze(0)

        h = self.relu(self.bn4(self.conv4(h)))           # 5 -> 3
        if record_activations:
            acts[4] = h.unsqueeze(0)

        h = self.relu(self.bn5(self.conv5(h)))           # 3 -> 1
        if record_activations:
            acts[5] = h.unsqueeze(0)

        h = self.group_pool(h)              # (B, C, 1, 1)
        h = h.flatten(1)                    # (B, C)
        out = self.classifier(h)

        if record_activations:
            return out, acts
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = P4CNNSmall(num_classes=4)
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    print("output shape:", y.shape)
    print("param count:", count_params(m))
