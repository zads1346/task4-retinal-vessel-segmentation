"""Conservative thin-vessel loss with an exact v2 baseline switch.

Defaults: 0.5 normalized weighted BCE + 0.47 soft Dice loss + 0.03 soft
ground-truth-skeleton miss loss. Thin-region positives receive BCE weight 1.5;
all other positive and negative pixels retain weight 1. This auxiliary term
uses the supplied GT skeleton only and is not a predicted-skeleton clDice loss.
"""

import math

import torch
from torch.nn import functional as F

from model import dice_bce_loss


def thin_aware_loss(
    logits,
    target,
    thin_region,
    skeleton,
    thin_weight=0.5,
    skeleton_weight=0.03,
):
    """Return one float32 scalar; empty skeletons contribute zero miss loss.

    Weight normalization is per image before averaging, so images with more
    designated thin pixels do not automatically dominate the batch. Setting
    both extra weights to zero delegates directly to the original v2 loss.
    """
    if not math.isfinite(thin_weight) or thin_weight < 0:
        raise ValueError("thin_weight must be finite and non-negative")
    if not math.isfinite(skeleton_weight) or not 0 <= skeleton_weight <= 0.5:
        raise ValueError("skeleton_weight must be finite and in [0,0.5]")
    if logits.ndim != 4 or logits.shape[1] != 1 or min(logits.shape) < 1:
        raise ValueError("Expected non-empty N x 1 x H x W logits")
    for name, mask in (("target", target), ("thin_region", thin_region), ("skeleton", skeleton)):
        if mask.shape != logits.shape:
            raise ValueError(f"{name} shape must match N x 1 x H x W logits")
        if mask.device != logits.device:
            raise ValueError(f"{name} must be on the same device as logits")
        if not bool((torch.isfinite(mask) & (mask >= 0) & (mask <= 1)).all()):
            raise ValueError(f"{name} must contain finite mask values in [0,1]")
    if bool(((thin_region > target) | (skeleton > target)).any()):
        raise ValueError("Thin-region and skeleton targets must be subsets of the vessel target")
    if thin_weight == 0 and skeleton_weight == 0:
        return dice_bce_loss(logits, target)

    logits = logits.float()
    target = target.float()
    thin_region = thin_region.float()
    skeleton = skeleton.float()
    axes = (1, 2, 3)
    pixel_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = 1.0 + float(thin_weight) * thin_region
    weighted_bce = ((weights * pixel_bce).sum(dim=axes) / weights.sum(dim=axes)).mean()

    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=axes)
    denominator = probability.sum(dim=axes) + target.sum(dim=axes)
    dice_loss = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    skeleton_size = skeleton.sum(dim=axes)
    skeleton_hit = (probability * skeleton).sum(dim=axes) / skeleton_size.clamp_min(1.0)
    skeleton_miss = torch.where(skeleton_size > 0, 1.0 - skeleton_hit, torch.zeros_like(skeleton_hit)).mean()
    return (0.5 * weighted_bce
            + (0.5 - float(skeleton_weight)) * dice_loss
            + float(skeleton_weight) * skeleton_miss)
