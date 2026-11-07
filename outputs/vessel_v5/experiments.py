"""Isolated v5 treatments. No learned preprocessing or external images."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from thin_loss import thin_aware_loss


def transform_cases(cases, experiment):
    if experiment not in ('A', 'B'):
        raise ValueError(experiment)
    if experiment == 'B':
        return cases
    result = {}
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for key, case in cases.items():
        rgb = case['image'].copy()
        green = np.rint(rgb[..., 1] * 255).astype(np.uint8)
        enhanced = clahe.apply(green).astype(np.float32) / 255.0
        rgb[..., 1] = 0.5 * rgb[..., 1] + 0.5 * enhanced
        rgb[case['image'].max(axis=-1) <= 0.05] = case['image'][case['image'].max(axis=-1) <= 0.05]
        result[key] = {**case, 'image': rgb}
    return result


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x, iterations=10):
    opened = F.max_pool2d(soft_erode(x), 3, 1, 1)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = F.max_pool2d(soft_erode(x), 3, 1, 1)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice(probability, target):
    axes = (1, 2, 3)
    pred_skeleton = soft_skeleton(probability)
    with torch.no_grad():
        target_skeleton = soft_skeleton(target)
    smooth = 1e-6
    precision = ((pred_skeleton * target).sum(axes) + smooth) / (pred_skeleton.sum(axes) + smooth)
    sensitivity = ((target_skeleton * probability).sum(axes) + smooth) / (target_skeleton.sum(axes) + smooth)
    return (1 - (2 * precision * sensitivity + smooth) / (precision + sensitivity + smooth)).mean()


def experiment_loss(logits, target, thin_region, skeleton, experiment, thin_weight=0.5, skeleton_weight=0.03):
    if experiment == 'A':
        return thin_aware_loss(logits, target, thin_region, skeleton, thin_weight, skeleton_weight)
    if experiment != 'B':
        raise ValueError(experiment)
    logits, target = logits.float(), target.float()
    axes = (1, 2, 3)
    weights = 1 + thin_weight * thin_region
    bce = ((F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weights).sum(axes) / weights.sum(axes)).mean()
    probability = logits.sigmoid()
    dice = (1 - (2 * (probability * target).sum(axes) + 1e-6) /
            (probability.sum(axes) + target.sum(axes) + 1e-6)).mean()
    return 0.5 * bce + (0.5 - skeleton_weight) * dice + skeleton_weight * soft_cldice(probability, target)
