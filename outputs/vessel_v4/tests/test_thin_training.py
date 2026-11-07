"""CPU checks for training-only thin targets, geometry, and loss behavior."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import dice_bce_loss
from thin_data import ThinPatchDataset
from thin_loss import thin_aware_loss

torch.set_num_threads(2)


def _synthetic_case():
    # An asymmetric one-pixel branch and a broad rectangular vessel component
    # expose both unsynchronized rotations and accidental thick-edge targeting.
    mask = np.zeros((48, 48), dtype=np.float32)
    mask[7:37, 8] = 1
    mask[9, 8:24] = 1
    mask[27:40, 28:41] = 1
    skeleton = skeletonize(mask.astype(bool))
    thin_skeleton = skeleton & (distance_transform_edt(mask) <= 2)
    thin_region = binary_dilation(thin_skeleton, structure=np.ones((3, 3)), iterations=1) & (mask > 0)
    image = np.stack((mask, thin_region.astype(np.float32), skeleton.astype(np.float32)), axis=-1)
    return {"image": image, "mask": mask}


def test_only_selected_labeled_training_ids_are_processed():
    accessed = []

    class GuardedCases(dict):
        def __getitem__(self, key):
            accessed.append(key)
            if key == "held_out":
                raise AssertionError("Validation case accessed during training-target construction")
            return super().__getitem__(key)

    cases = GuardedCases(train=_synthetic_case(), held_out={"unreadable": object()})
    dataset = ThinPatchDataset(cases, ["train"], patch_size=48, length=8)
    assert accessed == ["train"]
    assert set(dataset.cases) == {"train"}
    case = dataset.cases["train"]
    assert np.all(case["thin_region"] <= case["mask"])
    assert np.all(case["skeleton"] <= case["mask"])
    # The central part of the 13-pixel-wide vessel must not become thin merely
    # because its boundary has a small distance to background.
    assert case["thin_region"][30:37, 31:38].sum() == 0
    assert case["thin_region"][10:30, 8].sum() > 0
    with pytest.raises(ValueError, match="supplied label"):
        ThinPatchDataset({"test": {"image": cases["train"]["image"]}}, ["test"])


def test_determinism_and_all_targets_follow_rotation_flip_and_crop():
    original = _synthetic_case()
    untouched_image = original["image"].copy()
    untouched_mask = original["mask"].copy()
    for patch_size in (48, 24):
        dataset = ThinPatchDataset({"train": original}, ["train"], patch_size=patch_size,
                                   length=40, seed=2026, augment=True)
        for index in range(len(dataset)):
            output = dataset[index]
            repeated = dataset[index]
            assert len(output) == 4
            assert all(torch.equal(a, b) for a, b in zip(output, repeated))
            image, target, thin, skeleton = output
            assert image.shape == (3, patch_size, patch_size)
            assert all(value.shape == (1, patch_size, patch_size) for value in output[1:])
            assert all(value.dtype == torch.float32 for value in output)
            assert all(set(value.unique().tolist()) <= {0.0, 1.0} for value in output[1:])
            assert torch.all(thin <= target) and torch.all(skeleton <= target)
            # Synthetic RGB channels encode the three source masks. The v2
            # tone transforms preserve their two-level ordering, allowing an
            # independent geometric synchronization check after augmentation.
            for channel, expected in zip(image, (target[0], thin[0], skeleton[0])):
                if bool(expected.any()) and not bool(expected.all()):
                    midpoint = (channel.min() + channel.max()) / 2
                    assert torch.equal(channel > midpoint, expected.bool())
    np.testing.assert_array_equal(original["image"], untouched_image)
    np.testing.assert_array_equal(original["mask"], untouched_mask)


def test_empty_mask_pools_fallback_and_loss_backward_are_finite():
    case = {"image": np.zeros((2, 3, 3), np.float32), "mask": np.zeros((2, 3), np.float32)}
    dataset = ThinPatchDataset({"empty": case}, ["empty"], patch_size=16, length=12)
    for index in range(len(dataset)):
        _, target, thin, skeleton = dataset[index]
        assert float(target.sum() + thin.sum() + skeleton.sum()) == 0
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = thin_aware_loss(logits, target, target, target)
    probability = logits.sigmoid()
    dice = (1 - 1e-6 / (probability.sum((1, 2, 3)) + 1e-6)).mean()
    expected = 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.47 * dice
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad > 0)  # Background false positives are penalized.


def test_thin_positive_false_negative_receives_stronger_gradient():
    logits = torch.full((1, 1, 4, 4), -2.0, requires_grad=True)
    target = torch.zeros_like(logits)
    target[..., 1, 1] = 1
    target[..., 1, 2] = 1
    thin = torch.zeros_like(logits)
    thin[..., 1, 1] = 1
    loss = thin_aware_loss(logits, target, thin, torch.zeros_like(logits),
                           thin_weight=0.5, skeleton_weight=0)
    loss.backward()
    assert logits.grad[..., 1, 1] < logits.grad[..., 1, 2] < 0
    assert logits.grad[..., 0, 0] > 0


def test_zero_extra_weights_exactly_match_v2_loss_and_gradients():
    torch.manual_seed(31)
    logits = torch.randn(2, 1, 13, 17, requires_grad=True)
    target = (torch.rand_like(logits) > 0.8).float()
    baseline = dice_bce_loss(logits, target)
    candidate = thin_aware_loss(logits, target, target, target, thin_weight=0, skeleton_weight=0)
    assert torch.equal(candidate, baseline)
    baseline_grad = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
    candidate_grad = torch.autograd.grad(candidate, logits)[0]
    assert torch.equal(candidate_grad, baseline_grad)


def test_bad_shapes_ranges_subsets_and_weights_are_rejected():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros_like(logits)
    with pytest.raises(ValueError, match="shape"):
        thin_aware_loss(logits, target, target[..., :2], target)
    invalid = target.clone()
    invalid[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        thin_aware_loss(logits, target, invalid, target)
    invalid[..., 0, 0] = 1
    with pytest.raises(ValueError, match="subsets"):
        thin_aware_loss(logits, target, invalid, target)
    for settings in ({"thin_weight": -1}, {"skeleton_weight": 0.6}):
        with pytest.raises(ValueError):
            thin_aware_loss(logits, target, target, target, **settings)


def test_half_precision_logits_accumulate_loss_in_float32_on_cpu():
    logits = torch.tensor([[[[-12.0, 12.0], [0.0, -1.0]]]], dtype=torch.float16, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float16)
    loss = thin_aware_loss(logits, target, target, target)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
