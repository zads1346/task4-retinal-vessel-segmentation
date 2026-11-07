"""Local-only data loading, whole-image folds, and deterministic patch sampling.

The supplied files have .jpg names but TIFF/GIF contents. Pillow detects
their actual formats. Actual training labels use white vessels (255), so
all internal targets use vessel=1 and background=0. Test images are never
used by the training patch dataset.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp"}


def _id_key(value: str):
    return (0, int(value)) if value.isdecimal() else (1, value)


def _image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    found = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in _EXTENSIONS:
            if path.stem in found:
                raise ValueError(f"Duplicate image ID {path.stem!r} in {directory}")
            found[path.stem] = path
    if not found:
        raise ValueError(f"No images found in {directory}")
    return found


def load_cases(data_root, split: str = "train") -> dict[str, dict]:
    """Load an extracted ``segmentation`` directory into memory.

    ``data_root`` may also be its parent. Train and test are selected
    explicitly; images from the test split do not receive invented masks.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_root)
    if not (root / split).is_dir() and (root / "segmentation" / split).is_dir():
        root = root / "segmentation"
    images = _image_files(root / split / "image")
    labels = _image_files(root / split / "label") if split == "train" else {}
    if split == "train" and set(images) != set(labels):
        raise ValueError(
            f"Image/label ID mismatch: missing labels={sorted(set(images) - set(labels))}; "
            f"missing images={sorted(set(labels) - set(images))}"
        )
    cases = {}
    for case_id in sorted(images, key=_id_key):
        with Image.open(images[case_id]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        case = {"image": rgb}
        if split == "train":
            with Image.open(labels[case_id]) as label:
                mask = (np.asarray(label.convert("L")) > 127).astype(np.float32)
            if rgb.shape[:2] != mask.shape:
                raise ValueError(f"Image/label dimensions differ for ID {case_id}")
            case["mask"] = mask
        cases[case_id] = case
    return cases


def make_folds(ids, n_splits: int = 5, seed: int = 2026) -> list[dict]:
    """Deterministic, disjoint folds of original images, before patch extraction."""
    ordered = sorted([str(value) for value in ids], key=_id_key)
    if len(ordered) != len(set(ordered)):
        raise ValueError("IDs must be unique")
    if not 2 <= n_splits <= len(ordered):
        raise ValueError("n_splits must be between 2 and the number of images")
    shuffled = np.asarray(ordered, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    folds = []
    for validation in np.array_split(shuffled, n_splits):
        val_ids = sorted(validation.tolist(), key=_id_key)
        val_set = set(val_ids)
        train_ids = [case_id for case_id in ordered if case_id not in val_set]
        folds.append({"train_ids": train_ids, "val_ids": val_ids})
    return folds


class PatchDataset(Dataset):
    """Sample images uniformly and patches using a 50/25/25 location mixture.

    Half of patch locations are uniform; one quarter center on vessels;
    one quarter center on bright background or field-of-view boundaries.
    Missing foreground/hard-background pools fall back to uniform crops.

    The RNG depends only on ``seed`` and index, making samples invariant
    to DataLoader worker count. Use a distinct seed for each fold, and a
    sequential, non-repeating index range for a fixed-step training run.
    """

    def __init__(
        self,
        cases: dict[str, dict],
        ids,
        patch_size: int = 256,
        length: int = 100000,
        seed: int = 2026,
        augment: bool = True,
    ):
        self.ids = [str(case_id) for case_id in ids]
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("Training IDs must be non-empty and unique")
        if not isinstance(patch_size, int) or patch_size < 1 or length < 1 or seed < 0:
            raise ValueError("patch_size/length must be positive and seed non-negative")
        self.patch_size = patch_size
        self.length = int(length)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.cases = {}
        for case_id in self.ids:
            case = cases[case_id]
            if "mask" not in case:
                raise ValueError(f"Training requires a supplied label for ID {case_id}")
            image = np.asarray(case["image"], dtype=np.float32)
            mask = np.asarray(case["mask"], dtype=np.float32)
            if image.ndim != 3 or image.shape[-1] != 3 or image.shape[:2] != mask.shape:
                raise ValueError(f"Invalid RGB/mask shapes for ID {case_id}")
            if min(mask.shape) < 1 or not np.isfinite(image).all() or not np.isfinite(mask).all():
                raise ValueError(f"Empty or non-finite image/mask for ID {case_id}")
            if image.min() < 0 or image.max() > 1 or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Expected RGB in [0,1] and binary mask for ID {case_id}")
            pad_h = max(0, patch_size - image.shape[0])
            pad_w = max(0, patch_size - image.shape[1])
            if pad_h or pad_w:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
            vessel = mask > 0.5
            field = image.max(axis=-1) > 0.05
            gray = image.mean(axis=-1)
            height, width = mask.shape
            margin = min(8, max(1, min(height, width) // 8))
            padded = np.pad(field, margin, constant_values=False)
            eroded = field.copy()
            dilated = field.copy()
            for dy, dx in [(-margin, 0), (margin, 0), (0, -margin), (0, margin)]:
                shifted = padded[margin + dy : margin + dy + height,
                                 margin + dx : margin + dx + width]
                eroded &= shifted
                dilated |= shifted
            edge = dilated & ~eroded
            bright = np.zeros_like(field)
            if field.any():
                bright = field & (gray >= np.quantile(gray[field], 0.80))
            hard_background = (~vessel) & (edge | bright)
            self.cases[case_id] = {
                "image": image,
                "mask": mask,
                "vessel_centers": np.argwhere(vessel),
                "hard_centers": np.argwhere(hard_background),
            }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        case_id = self.ids[int(rng.integers(len(self.ids)))]
        case = self.cases[case_id]
        image, mask = case["image"], case["mask"]
        size = self.patch_size
        max_y, max_x = mask.shape[0] - size, mask.shape[1] - size
        mode = rng.random()
        centers = case["vessel_centers"] if mode < 0.75 else case["hard_centers"]
        if mode >= 0.5 and len(centers):
            center_y, center_x = centers[int(rng.integers(len(centers)))]
            y = int(np.clip(int(center_y) - size // 2, 0, max_y))
            x = int(np.clip(int(center_x) - size // 2, 0, max_x))
        else:
            y = int(rng.integers(max_y + 1))
            x = int(rng.integers(max_x + 1))
        image = image[y : y + size, x : x + size].copy()
        mask = mask[y : y + size, x : x + size].copy()
        if self.augment:
            turns = int(rng.integers(4))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if rng.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                image = image * np.float32(rng.uniform(0.88, 1.12))
            if rng.random() < 0.5:
                mean = image.mean(axis=(0, 1), keepdims=True)
                image = (image - mean) * np.float32(rng.uniform(0.85, 1.15)) + mean
            image = np.clip(image, 0, 1)
            if rng.random() < 0.3:
                image = image ** np.float32(rng.uniform(0.85, 1.15))
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        mask = np.ascontiguousarray(mask[None], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask)
