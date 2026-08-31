"""
Pure-PyTorch p4 group-equivariant convolution layers.

No custom CUDA/C++ extensions required — rotated filter copies are built
with torch.rot90() directly from a single learnable master weight tensor
inside forward(), so gradients flow correctly (this sidesteps the
register_buffer bug noted in the internship report, where rotated kernels
built once at init time and stored as buffers were detached from autograd).

Two layer types:
  - P4ConvZ2:  Z^2 -> p4.   Input has no orientation channel (plain image).
               Produces 4 rotated copies of each output channel, one per
               orientation in {0,90,180,270}.
  - P4ConvP4:  p4 -> p4.    Input already has 4 orientation channels per
               "logical" input channel (e.g. output of a P4ConvZ2 or
               another P4ConvP4). Correctly cyclically shifts the
               orientation axis of the filter as well as rotating spatially,
               which is required for correct p4 equivariance in layers
               beyond the first.

Both layers store weights as (out_channels, in_channels, [in_rot,] k, k)
and expand to a standard (out_channels*4, in_channels*[4], k, k) conv
weight at forward time via rot90 + roll, then call F.conv2d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class P4ConvZ2(nn.Module):
    """First layer: plain image (Z^2) -> p4 feature map (4 orientations)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # master_w: (out_channels, in_channels, k, k) -- this IS the learnable parameter.
        self.master_w = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.master_w, mode="fan_in", nonlinearity="relu")

        if bias:
            # one bias per (out_channel, orientation) — orientation-dependent bias
            # would break equivariance, so we share bias across orientations:
            # one bias per output channel, broadcast to all 4 rotated copies.
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def _build_filter_bank(self):
        # Build 4 rotated copies of master_w, stacked on a new orientation axis.
        # rot90(w, k, dims=(-2,-1)) rotates the spatial kernel by k*90 degrees.
        rotated = [torch.rot90(self.master_w, k, dims=(-2, -1)) for k in range(4)]
        # stack -> (4, out_channels, in_channels, k, k)
        stacked = torch.stack(rotated, dim=0)
        # rearrange -> (out_channels*4, in_channels, k, k) for a standard conv2d weight
        w = stacked.permute(1, 0, 2, 3, 4).reshape(
            self.out_channels * 4, self.in_channels, self.kernel_size, self.kernel_size
        )
        return w

    def forward(self, x):
        # x: (B, in_channels, H, W)
        w = self._build_filter_bank()  # (out_channels*4, in_channels, k, k)
        bias = None
        if self.bias is not None:
            bias = self.bias.repeat_interleave(4)  # (out_channels*4,)
        out = F.conv2d(x, w, bias=bias, stride=self.stride, padding=self.padding)
        # out: (B, out_channels*4, H', W') -- interpreted as (B, out_channels, 4, H', W')
        return out


class P4ConvP4(nn.Module):
    """Subsequent layers: p4 feature map -> p4 feature map.

    Input x has shape (B, in_channels*4, H, W), i.e. 4 orientation channels
    per logical input channel, ordered as [c0_r0, c0_r90, c0_r180, c0_r270,
    c1_r0, ...]. To stay p4-equivariant, rotating the output by 90 degrees
    must correspond to *both* spatially rotating the filter AND cyclically
    shifting which input-orientation-channel aligns with which filter slot.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # master_w: (out_channels, in_channels, 4, k, k) -- 4 = input orientation axis
        self.master_w = nn.Parameter(
            torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.master_w, mode="fan_in", nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def _build_filter_bank(self):
        # For each output rotation k in {0,1,2,3}:
        #   - spatially rotate the (k,k) filter by k*90
        #   - cyclically roll the input-orientation axis by k
        # This is the standard p4->p4 filter transformation (Cohen & Welling 2016).
        banks = []
        for k in range(4):
            w_rot = torch.rot90(self.master_w, k, dims=(-2, -1))  # rotate spatially
            w_rot = torch.roll(w_rot, shifts=k, dims=2)           # cyclic shift orientation axis
            banks.append(w_rot)
        # stack -> (4, out_channels, in_channels, 4, k, k)
        stacked = torch.stack(banks, dim=0)
        # merge in_channels and orientation axis -> (out_channels*4, in_channels*4, k, k)
        w = stacked.permute(1, 0, 2, 3, 4, 5).reshape(
            self.out_channels * 4, self.in_channels * 4, self.kernel_size, self.kernel_size
        )
        return w

    def forward(self, x):
        # x: (B, in_channels*4, H, W)
        w = self._build_filter_bank()
        bias = None
        if self.bias is not None:
            bias = self.bias.repeat_interleave(4)
        out = F.conv2d(x, w, bias=bias, stride=self.stride, padding=self.padding)
        return out


class P4BatchNorm2d(nn.Module):
    """BatchNorm for p4 feature maps that shares statistics ACROSS the 4
    orientation channels of each logical channel, instead of treating each
    orientation as an independent feature.

    This is required for equivariance: a standard nn.BatchNorm2d(C*4)
    learns a separate running mean/var per orientation slot. Since training
    data in this project is upright-only, the "0-degree slot" and the
    "180-degree slot" end up calibrated on very different statistics purely
    because of which orientation channel they happen to occupy -- this
    breaks exact invariance at test time even though the conv layers
    themselves are equivariant (verified separately in gconv.py's own
    equivariance tests). Sharing stats across the orientation axis fixes
    this: all 4 orientation channels of a given logical channel are
    normalized identically, so rotating the input just permutes which
    orientation channel holds which value -- it can't change the
    normalization applied to that value.

    Input/output shape: (B, channels*4, H, W), same convention as
    P4ConvZ2/P4ConvP4 output (orientation-fastest layout: channel c's 4
    orientations are contiguous at indices [4c, 4c+1, 4c+2, 4c+3]).
    """

    def __init__(self, channels, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        if affine:
            # one scale/shift per LOGICAL channel, shared across its 4 orientations
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        B, C4, H, W = x.shape
        x = x.view(B, self.channels, 4, H, W)

        if self.training:
            # mean/var over batch, the 4-orientation axis, and spatial dims --
            # this is what makes it orientation-shared rather than per-slot.
            dims = (0, 2, 3, 4)
            batch_mean = x.mean(dim=dims)
            batch_var = x.var(dim=dims, unbiased=False)

            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                # unbiased variance for the running estimate, matching nn.BatchNorm2d convention
                n = B * 4 * H * W
                unbiased_var = batch_var * n / max(n - 1, 1)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var)
                self.num_batches_tracked += 1

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        mean = mean.view(1, self.channels, 1, 1, 1)
        var = var.view(1, self.channels, 1, 1, 1)
        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            w = self.weight.view(1, self.channels, 1, 1, 1)
            b = self.bias.view(1, self.channels, 1, 1, 1)
            x = x * w + b

        return x.view(B, C4, H, W)


class P4GroupPool(nn.Module):
    """Collapse the 4 orientation channels per logical channel via max (or mean).

    Input:  (B, channels*4, H, W)
    Output: (B, channels, H, W)  -- invariant to which of the 4 rotations
            the input pattern arrived in, since pooling over the group is
            a standard way to go from equivariant to invariant features.
    """

    def __init__(self, channels, mode="max"):
        super().__init__()
        self.channels = channels
        assert mode in ("max", "mean")
        self.mode = mode

    def forward(self, x):
        B, C4, H, W = x.shape
        x = x.view(B, self.channels, 4, H, W)
        if self.mode == "max":
            x, _ = x.max(dim=2)
        else:
            x = x.mean(dim=2)
        return x