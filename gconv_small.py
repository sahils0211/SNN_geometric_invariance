"""
gconv_small.py

p4-equivariant convolution / batchnorm / pooling primitives, unchanged in
mathematical construction from the full-size codebase's gconv.py — reused
here verbatim (just re-declared in a standalone file so this reduced-capacity
experiment is self-contained and drop-in-runnable on gpuserver without
depending on the original module).

If you already have gconv.py on gpuserver with P4ConvZ2 / P4ConvP4 /
P4BatchNorm2d / P4GroupPool verified to ~1e-6 equivariance, you can just
`from gconv import P4ConvZ2, P4ConvP4, P4BatchNorm2d, P4GroupPool` instead
of this file and delete this one. This copy exists only so everything for
the reduced-parameter experiment is in one place.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class P4ConvZ2(nn.Module):
    """Z^2 -> p4 lifting convolution. Input: (B, C_in, H, W).
    Output: (B, C_out, 4, H', W') -- one feature map per orientation 0/90/180/270.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.master_w = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.master_w, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        outs = []
        for r in range(4):
            w_r = torch.rot90(self.master_w, k=r, dims=(2, 3))
            outs.append(F.conv2d(x, w_r, bias=self.bias))
        # (B, C_out, 4, H', W')
        return torch.stack(outs, dim=2)


class P4ConvP4(nn.Module):
    """p4 -> p4 convolution. Input/Output: (B, C, 4, H, W).
    Rotates the spatial kernel AND cyclically shifts the orientation axis
    of the input channels, so the composition remains a valid p4-group
    convolution (Cohen & Welling, 2016).
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.master_w = nn.Parameter(
            torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.master_w, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # x: (B, C_in, 4, H, W)
        B = x.shape[0]
        outs = []
        for r in range(4):
            # rotate spatial kernel
            w_r = torch.rot90(self.master_w, k=r, dims=(3, 4))
            # cyclically shift orientation axis of the kernel to match
            w_r = torch.roll(w_r, shifts=r, dims=2)
            # flatten orientation into channel dim for a plain conv2d
            w_r_flat = w_r.reshape(self.out_channels, self.in_channels * 4,
                                    self.kernel_size, self.kernel_size)
            x_flat = x.reshape(B, self.in_channels * 4, x.shape[-2], x.shape[-1])
            outs.append(F.conv2d(x_flat, w_r_flat, bias=self.bias))
        return torch.stack(outs, dim=2)  # (B, C_out, 4, H', W')


class P4BatchNorm2d(nn.Module):
    """BatchNorm for p4 feature maps that SHARES statistics and affine
    parameters across the 4 orientation channels. Plain nn.BatchNorm2d
    applied per-orientation breaks equivariance once trained upright-only
    (verified in the full-size codebase: equivariance error rises to
    0.1-0.2 after training with standard BatchNorm; this version keeps
    equivariance at ~1e-6 after training).
    """
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x):
        # x: (B, C, 4, H, W) -- reduce over batch, orientation(4), H, W jointly
        B, C, R, H, W = x.shape
        if self.training:
            mean = x.mean(dim=(0, 2, 3, 4))
            var = x.var(dim=(0, 2, 3, 4), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean = self.running_mean
            var = self.running_var
        mean = mean.view(1, C, 1, 1, 1)
        var = var.view(1, C, 1, 1, 1)
        w = self.weight.view(1, C, 1, 1, 1)
        b = self.bias.view(1, C, 1, 1, 1)
        return (x - mean) / torch.sqrt(var + self.eps) * w + b


class P4GroupPool(nn.Module):
    """Max-pool over the 4 orientation channels -> invariant-at-group-angles
    feature map. Input: (B, C, 4, H, W). Output: (B, C, H, W)."""
    def forward(self, x):
        return x.max(dim=2).values
