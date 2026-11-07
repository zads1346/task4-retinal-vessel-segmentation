"""A randomly initialized U-Net for binary retinal vessel segmentation.

Inputs are RGB tensors in [0, 1], in NCHW order. The network returns one
logit per pixel; apply sigmoid only for metrics or prediction. No model or
pretrained weights are downloaded by this module.
"""

from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two spatial convolutions with batch-size-independent normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = gcd(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Four-downsampling U-Net with exact skip-size interpolation.

    ``base_channels=24`` gives a compact baseline for an 8 GB GPU. Odd
    image sizes are supported without resizing the final prediction.
    """

    def __init__(self, base_channels: int = 24):
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 2:
            raise ValueError("base_channels must be an integer of at least 2")
        self.base_channels = base_channels
        widths = [base_channels * 2**i for i in range(5)]
        self.encoders = nn.ModuleList(
            [DoubleConv(3, widths[0])]
            + [DoubleConv(widths[i - 1], widths[i]) for i in range(1, 5)]
        )
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.decoders = nn.ModuleList(
            [DoubleConv(widths[i + 1] + widths[i], widths[i]) for i in range(3, -1, -1)]
        )
        self.output = nn.Conv2d(widths[0], 1, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected non-empty N x 3 x H x W RGB input")
        # Padding permits even tiny synthetic patches used by integrity checks.
        # Normal train/test images never need this branch.
        height, width = x.shape[-2:]
        if min(height, width) < 16:
            x = F.pad(x, (0, max(0, 16 - width), 0, max(0, 16 - height)))
        skips = []
        for level, encoder in enumerate(self.encoders):
            if level:
                x = self.pool(x)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.output(x)[..., :height, :width]


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal-weight BCE and per-image soft Dice, accumulated in float32.

    ``target`` must be N x 1 x H x W with vessels represented by 1. Dice
    uses only the vessel class, rather than averaging it with background.
    """
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits and target must have identical N x 1 x H x W shapes")
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    axes = (1, 2, 3)
    intersection = (probabilities * target).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + target.sum(dim=axes)
    soft_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 0.5 * bce + 0.5 * (1.0 - soft_dice.mean())
