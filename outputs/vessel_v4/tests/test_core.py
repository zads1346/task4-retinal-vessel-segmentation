"""Integrity checks for spatial inference, image splitting, and training inputs."""

import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import PatchDataset, load_cases, make_folds
from inference import binary_metrics, predict_image
from model import UNet, dice_bce_loss


class ConstantLogit(nn.Module):
    def forward(self, image):
        return torch.full_like(image[:, :1], float(np.log(4.0)))


class PointwiseLogit(nn.Module):
    def forward(self, image):
        return 3.0 * image[:, :1] - 1.7


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.device = torch.device("cpu")

    def test_overlap_covers_odd_and_smaller_than_window_images(self):
        rng = np.random.default_rng(3)
        for shape in [(37, 53), (7, 9), (16, 16), (31, 16)]:
            image = rng.random((*shape, 3), dtype=np.float32)
            for tta in [False, True]:
                with self.subTest(shape=shape, tta=tta):
                    probability = predict_image(
                        ConstantLogit(), image, self.device,
                        patch_size=16, stride=7, batch_size=3, tta=tta,
                    )
                    self.assertEqual(probability.shape, shape)
                    self.assertTrue(np.isfinite(probability).all())
                    np.testing.assert_allclose(probability, 0.8, atol=2e-6, rtol=0)

    def test_spatial_pointwise_predictions_survive_tta_inverse_and_stitching(self):
        # An asymmetric spatial pattern exposes missing inverse flips or seams.
        rng = np.random.default_rng(11)
        image = rng.random((29, 43, 3), dtype=np.float32)
        expected = 1.0 / (1.0 + np.exp(-(3.0 * image[..., 0] - 1.7)))
        for tta in [False, True]:
            probability = predict_image(
                PointwiseLogit(), image, self.device,
                patch_size=16, stride=9, batch_size=4, tta=tta,
            )
            np.testing.assert_allclose(probability, expected, atol=2e-6, rtol=0)

    def test_whole_image_folds_and_deterministic_label_preserving_patches(self):
        ids = [str(i) for i in range(1, 21)]
        folds = make_folds(ids, seed=2026)
        self.assertEqual(folds, make_folds(reversed(ids), seed=2026))
        held_out = []
        for fold in folds:
            self.assertEqual(len(fold["train_ids"]), 16)
            self.assertEqual(len(fold["val_ids"]), 4)
            self.assertFalse(set(fold["train_ids"]) & set(fold["val_ids"]))
            held_out.extend(fold["val_ids"])
        self.assertCountEqual(held_out, ids)

        mask = np.zeros((37, 41), dtype=np.float32)
        mask[3:32:4, 6:29] = 1
        image = np.repeat(mask[..., None], 3, axis=-1)
        cases = {"train": {"image": image, "mask": mask},
                 "unlabeled_test": {"image": image}}
        dataset = PatchDataset(cases, ["train"], patch_size=16, length=20, augment=False)
        for index in range(len(dataset)):
            x, y = dataset[index]
            self.assertTrue(torch.equal(x, dataset[index][0]))
            self.assertTrue(torch.equal(y, dataset[index][1]))
            self.assertTrue(torch.equal(x[0], y[0]))
        augmented = PatchDataset(cases, ["train"], patch_size=16, length=8, augment=True)
        for index in range(len(augmented)):
            x, y = augmented[index]
            self.assertTrue(torch.equal(x, augmented[index][0]))
            self.assertTrue(torch.equal(y, augmented[index][1]))
            self.assertTrue(set(y.unique().tolist()) <= {0.0, 1.0})
            self.assertGreaterEqual(float(x.min()), 0.0)
            self.assertLessEqual(float(x.max()), 1.0)
        with self.assertRaisesRegex(ValueError, "supplied label"):
            PatchDataset(cases, ["unlabeled_test"])

    def test_actual_file_formats_white_vessel_polarity_and_ignored_test_labels(self):
        # Temporary fixtures exercise Pillow format detection, rather than
        # relying on a mocked loader or downloading any data.
        with tempfile.TemporaryDirectory(prefix="vessel_core_") as directory:
            root = Path(directory) / "segmentation"
            for subdirectory in ["train/image", "train/label", "test/image", "test/label"]:
                (root / subdirectory).mkdir(parents=True)
            mask = np.zeros((19, 23), dtype=np.uint8)
            mask[2:16, 8] = 255
            rgb = np.zeros((19, 23, 3), dtype=np.uint8)
            rgb[..., 0] = mask
            for case_id in ["1", "10", "2"]:
                Image.fromarray(rgb).save(root / "train/image" / f"{case_id}.jpg", format="TIFF")
                Image.fromarray(mask).save(root / "train/label" / f"{case_id}.jpg", format="GIF")
                Image.fromarray(rgb).save(root / "test/image" / f"{case_id}.jpg", format="TIFF")
                # Reading this would fail: test loading must ignore labels.
                (root / "test/label" / f"{case_id}.jpg").write_bytes(b"not an image")
            train = load_cases(directory, "train")
            test = load_cases(directory, "test")
            self.assertEqual(list(train), ["1", "2", "10"])
            self.assertEqual(list(test), ["1", "2", "10"])
            np.testing.assert_array_equal(train["1"]["mask"], mask > 127)
            self.assertEqual(train["1"]["image"].dtype, np.float32)
            self.assertTrue(all("mask" not in case for case in test.values()))

    def test_empty_small_training_patch_and_finite_backward(self):
        cases = {"x": {"image": np.zeros((1, 2, 3), np.float32),
                       "mask": np.zeros((1, 2), np.float32)}}
        image, mask = PatchDataset(cases, ["x"], patch_size=16, length=2)[1]
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(float(mask.sum()), 0.0)
        torch.manual_seed(2)
        model = UNet(base_channels=4)
        for height, width in [(33, 47), (7, 9)]:
            x = torch.rand(1, 3, height, width)
            target = torch.zeros(1, 1, height, width)
            target[..., 1:3, 1:3] = 1
            logits = model(x)
            self.assertEqual(logits.shape, target.shape)
            loss = dice_bce_loss(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            model.zero_grad(set_to_none=True)

    def test_vessel_metrics_do_not_include_background_dice(self):
        target = np.zeros((100, 100), bool)
        target[20:30, 40:50] = True
        report = binary_metrics(np.zeros_like(target), target)
        self.assertEqual(report["dice"], 0.0)
        self.assertEqual(report["fn"], 100)
        self.assertEqual(binary_metrics(target, target)["dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
