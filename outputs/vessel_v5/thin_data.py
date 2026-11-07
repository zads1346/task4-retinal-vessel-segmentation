"""Training-only thin-vessel targets and deterministic v2-style patch augmentation.

Only explicitly selected training cases are processed. A thin-vessel proxy is
defined on the supplied label's skeleton by Euclidean distance-to-background
radius <= 2 pixels. Its one-pixel (3x3) dilation is intersected with the original
vessel label. No validation label, test prediction, or external data informs
these targets. The original v2 data.py remains unchanged.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize
import torch

from data import PatchDataset


class ThinPatchDataset(PatchDataset):
    """Return RGB, vessel label, thin-region label, and full vessel skeleton.

    Image selection remains uniform. Patch locations use 40% uniform, 20%
    thin-skeleton centers, 15% all-vessel centers, and 25% hard-background
    centers. An empty requested pool falls back to a uniform crop. Every
    output is float32; target tensors have shape 1 x patch_size x patch_size.
    """

    def __init__(
        self,
        cases,
        ids,
        patch_size=256,
        length=100000,
        seed=2026,
        augment=True,
    ):
        super().__init__(cases, ids, patch_size, length, seed, augment)
        # Parent construction only copied references/padded cases in self.ids;
        # do not iterate the original cases mapping, which also contains val.
        for case_id in self.ids:
            case = self.cases[case_id]
            vessel = case["mask"] > 0.5
            skeleton = skeletonize(vessel)
            radius = distance_transform_edt(vessel)
            thin_skeleton = skeleton & (radius <= 2.0)
            thin_region = binary_dilation(
                thin_skeleton, structure=np.ones((3, 3), dtype=bool), iterations=1
            ) & vessel
            case["skeleton"] = skeleton.astype(np.float32)
            case["thin_region"] = thin_region.astype(np.float32)
            case["thin_centers"] = np.argwhere(thin_skeleton)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        case_id = self.ids[int(rng.integers(len(self.ids)))]
        case = self.cases[case_id]
        size = self.patch_size
        max_y, max_x = case["mask"].shape[0] - size, case["mask"].shape[1] - size
        mode = rng.random()
        if mode < 0.40:
            centers = None
        elif mode < 0.60:
            centers = case["thin_centers"]
        elif mode < 0.75:
            centers = case["vessel_centers"]
        else:
            centers = case["hard_centers"]
        if centers is not None and len(centers):
            center_y, center_x = centers[int(rng.integers(len(centers)))]
            y = int(np.clip(int(center_y) - size // 2, 0, max_y))
            x = int(np.clip(int(center_x) - size // 2, 0, max_x))
        else:
            y = int(rng.integers(max_y + 1))
            x = int(rng.integers(max_x + 1))
        crop = np.s_[y : y + size, x : x + size]
        image = case["image"][crop].copy()
        mask = case["mask"][crop].copy()
        thin_region = case["thin_region"][crop].copy()
        skeleton = case["skeleton"][crop].copy()
        if self.augment:
            # Same geometric and photometric distribution/order as v2. Only
            # geometric operations apply to all three aligned target maps.
            turns = int(rng.integers(4))
            image = np.rot90(image, turns)
            mask = np.rot90(mask, turns)
            thin_region = np.rot90(thin_region, turns)
            skeleton = np.rot90(skeleton, turns)
            if rng.random() < 0.5:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
                thin_region = thin_region[:, ::-1]
                skeleton = skeleton[:, ::-1]
            if rng.random() < 0.5:
                image = image * np.float32(rng.uniform(0.88, 1.12))
            if rng.random() < 0.5:
                mean = image.mean(axis=(0, 1), keepdims=True)
                image = (image - mean) * np.float32(rng.uniform(0.85, 1.15)) + mean
            image = np.clip(image, 0, 1)
            if rng.random() < 0.3:
                image = image ** np.float32(rng.uniform(0.85, 1.15))
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32))
        targets = [torch.from_numpy(np.ascontiguousarray(value[None], dtype=np.float32))
                   for value in (mask, thin_region, skeleton)]
        return image, *targets
