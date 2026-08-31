"""
P4CNN -- faithful reproduction of the rotated-MNIST architecture from
Cohen & Welling, "Group Equivariant Convolutional Networks" (ICML 2016),
Section 8.1.

From the paper:
    "We performed model selection using the validation set, yielding a
    CNN architecture (Z2CNN) with 7 layers of 3x3 convolutions (4x4 in
    the final layer), 20 channels in each layer, relu activation
    functions, batch normalization, dropout, and max-pooling after
    layer 2 ... Next, we replaced each convolution by a p4-convolution,
    divided the number of filters by sqrt(4) = 2 (so as to keep the
    number of parameters approximately fixed), and added max-pooling
    over rotations after the last convolution layer. This architecture
    (P4CNN) was found to perform better without dropout, so we removed
    it."

So P4CNN is:
    7 conv layers: six 3x3 (VALID/no padding) + one 4x4 (VALID) final layer
    10 channels per layer (20 halved by sqrt(4))
    BatchNorm + ReLU after every conv layer
    2x2 max-pool (stride 2) after layer 2 only
    NO dropout
    G-pooling (max over the 4 p4 orientations) after the final conv layer
    Linear classifier on top

On 28x28 MNIST input with valid convolutions this traces to exactly
1x1 spatial output before the final classifier -- see the shape trace
in the accompanying conversation; the 4x4 final kernel is sized
specifically to collapse the last 4x4 feature map to 1x1.

Uses the from-scratch, numerically-verified P4ConvZ2/P4ConvP4/P4GroupPool
primitives in gconv.py (pure PyTorch, rot90-based, no compiled CUDA/C++
extensions required).

Param count at C=10, in_channels=1 (grayscale), n_classes=4:  ~25.2k
Param count at C=10, in_channels=1 (grayscale), n_classes=10: ~25.2k
(The paper's own P4CNN, on 1-channel 28x28 rotated-MNIST with 10 classes,
is in this same few-tens-of-thousands-of-parameters regime -- NOT the
1.3M-parameter figure, which is from the separate CIFAR-10 All-CNN-p4
comparison in Table 2 of the paper, a 9-layer, 48->96-channel RGB
architecture. See p4allcnn.py in this same directory for that one.)
"""

import torch
import torch.nn as nn

from gconv import P4ConvZ2, P4ConvP4, P4GroupPool, P4BatchNorm2d


class P4CNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, channels=10):
        super().__init__()
        C = channels

        # Layer 1: Z2 -> p4, 3x3, valid (no padding)
        self.conv1 = P4ConvZ2(in_channels, C, kernel_size=3, stride=1, padding=0)
        self.bn1 = P4BatchNorm2d(C)

        # Layer 2: p4 -> p4, 3x3, valid
        self.conv2 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn2 = P4BatchNorm2d(C)

        # max-pool after layer 2 only, per the paper
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layers 3-6: p4 -> p4, 3x3, valid
        self.conv3 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn3 = P4BatchNorm2d(C)

        self.conv4 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn4 = P4BatchNorm2d(C)

        self.conv5 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn5 = P4BatchNorm2d(C)

        self.conv6 = P4ConvP4(C, C, kernel_size=3, stride=1, padding=0)
        self.bn6 = P4BatchNorm2d(C)

        # Layer 7: p4 -> p4, 4x4, valid -- collapses remaining 4x4 spatial map to 1x1
        self.conv7 = P4ConvP4(C, C, kernel_size=4, stride=1, padding=0)
        self.bn7 = P4BatchNorm2d(C)

        # G-pooling: max over the 4 orientation channels -> rotation-invariant features
        self.group_pool = P4GroupPool(channels=C, mode="max")

        self.fc = nn.Linear(C, n_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, record_activations=False):
        """record_activations: if True, matches SpikingP4CNN's interface
        so the same representations.py analysis code works on both model
        types unmodified. Returns (output, activations) where activations
        is shaped (1, B, C, H, W) per layer -- a size-1 "time" axis -- so
        downstream code that sums/reshapes over a T dimension works
        identically whether T=1 (analog) or T=50 (spiking). This is the
        natural analog interpretation: a single forward pass IS the
        "whole" representation, there's no time axis to begin with.
        """
        activations = {} if record_activations else None

        x = self.act(self.bn1(self.conv1(x)))
        if record_activations:
            activations[0] = x.detach().unsqueeze(0)  # (1, B, C, H, W)

        x = self.act(self.bn2(self.conv2(x)))
        if record_activations:
            activations[1] = x.detach().unsqueeze(0)

        x = self.pool(x)

        x = self.act(self.bn3(self.conv3(x)))
        if record_activations:
            activations[2] = x.detach().unsqueeze(0)

        x = self.act(self.bn4(self.conv4(x)))
        if record_activations:
            activations[3] = x.detach().unsqueeze(0)

        x = self.act(self.bn5(self.conv5(x)))
        if record_activations:
            activations[4] = x.detach().unsqueeze(0)

        x = self.act(self.bn6(self.conv6(x)))
        if record_activations:
            activations[5] = x.detach().unsqueeze(0)

        x = self.act(self.bn7(self.conv7(x)))
        if record_activations:
            activations[6] = x.detach().unsqueeze(0)

        x = self.group_pool(x)          # (B, C, 1, 1) at 28x28 input
        x = x.flatten(1)                # (B, C)
        out = self.fc(x)

        if record_activations:
            return out, activations
        return out


if __name__ == "__main__":
    for n_classes in (4, 10):
        m = P4CNN(in_channels=1, n_classes=n_classes, channels=10)
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 1, 28, 28)
        y = m(x)
        print(f"n_classes={n_classes}: params={n_params}, output shape={tuple(y.shape)}")