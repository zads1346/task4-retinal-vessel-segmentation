"""Render actual v4 held-out and test predictions after final outputs exist.

CPU rendering only: no checkpoint is loaded and no model is instantiated or
executed. This writes three PNG previews, never changes prediction arrays,
CSV files, labels, or saved metadata, and never reads test ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

from data import load_cases
from inference import binary_metrics, save_preview


IDS = [str(index) for index in range(1, 21)]
TEST_IDS = ["1", "6", "11", "16"]
GREEN = (20, 230, 100)
RED = (255, 55, 60)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, label):
    if not np.isfinite(float(right)) or abs(float(left) - float(right)) > 1e-9:
        raise ValueError(f"Inconsistent {label}: {left} != {right}")


def checked_probabilities(path, cases):
    with np.load(path, allow_pickle=False) as source:
        if len(source.files) != 20 or set(source.files) != set(IDS):
            raise ValueError(f"Expected exactly 20 probability IDs: {path}")
        probabilities = {}
        for image_id in IDS:
            values = np.asarray(source[image_id])
            if (values.shape != cases[image_id]["image"].shape[:2]
                    or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
                    or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
                raise ValueError(f"Invalid native probability for {image_id}: {path}")
            probabilities[image_id] = values
    return probabilities


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def thin_overlay(image, target, probability, threshold, expected_row):
    """Classify thin GT skeleton pixels before display-only point expansion."""
    skeleton = skeletonize(target, method="zhang")
    thin = skeleton & (distance_transform_edt(target) <= 2.0)
    prediction = probability >= threshold
    hits = thin & prediction
    misses = thin & ~prediction
    count = int(thin.sum())
    if count == 0:
        raise ValueError("Selected preview has no thin GT skeleton pixels")
    recall = int(hits.sum()) / count
    close(recall, expected_row["thin_skeleton_recall"], "thin skeleton recall")
    for key, actual in (("thin_skeleton_gt_pixels", count), ("thin_skeleton_hits", int(hits.sum())),
                        ("thin_skeleton_misses", int(misses.sum()))):
        if key in expected_row and int(expected_row[key]) != actual:
            raise ValueError(f"Comparison count differs from actual label/probability: {key}")
    background = np.uint8(np.clip(image * 255.0 * 0.38, 0, 255))
    # Enlarge the symbols for visibility, never the target or prediction.
    # The exact central pixels are reapplied after overlapping display marks.
    cross = np.array([[False, True, False], [True, True, True], [False, True, False]])
    background[binary_dilation(hits, structure=cross, iterations=1)] = GREEN
    background[binary_dilation(misses, structure=cross, iterations=1)] = RED
    background[hits] = GREEN
    background[misses] = RED
    return background, {"gt_pixels": count, "hits": int(hits.sum()), "misses": int(misses.sum()), "recall": recall}


def render_thin_sheet(ids, train_cases, probabilities, rows, threshold, mode):
    # Keep source images and error overlays at native scale for zoom inspection.
    cell_w = max(train_cases[image_id]["image"].shape[1] for image_id in ids)
    cell_h = max(train_cases[image_id]["image"].shape[0] for image_id in ids)
    banner, header, gutter = 102, 52, 8
    width = 4 * cell_w + 5 * gutter
    height = banner + len(ids) * (header + cell_h + gutter) + gutter
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, body_font = font(22), font(16)
    draw.text((gutter, 8), f"v4 selected OOF: {mode}, threshold {threshold:.2f} | Lowest four thin-skeleton recalls", fill="black", font=title_font)
    draw.rectangle((gutter, 44, gutter + 16, 60), fill=GREEN)
    draw.text((gutter + 24, 42), "Green = strict native thin-GT-skeleton hit", fill="black", font=body_font)
    draw.rectangle((570, 44, 586, 60), fill=RED)
    draw.text((594, 42), "Red = strict native thin-GT-skeleton miss", fill="black", font=body_font)
    draw.text((gutter, 72), "Symbols expanded by a 1-native-pixel cross for DISPLAY ONLY; overlap may occur. Counts/recall use exact pixels before expansion.", fill="black", font=body_font)
    summaries = {}
    for index, image_id in enumerate(ids):
        case = train_cases[image_id]
        target = case["mask"] > 0.5
        probability = probabilities[image_id]
        overlay, counts = thin_overlay(case["image"], target, probability, threshold, rows[image_id])
        summaries[image_id] = counts
        pictures = [
            ("RGB (native)", np.uint8(np.clip(case["image"] * 255, 0, 255))),
            ("GT vessels (native)", target.astype(np.uint8) * 255),
            (f"Selected binary p >= {threshold:.2f}", (probability >= threshold).astype(np.uint8) * 255),
            ("Thin GT skeleton coverage", overlay),
        ]
        y = banner + index * (header + cell_h + gutter)
        for column, (title, array) in enumerate(pictures):
            x = gutter + column * (cell_w + gutter)
            draw.text((x + 2, y + 3), f"ID {image_id} | {title}", fill="black", font=body_font)
            if column == 3:
                draw.text((x + 2, y + 25), f"Recall {counts['recall']:.4f}; hits {counts['hits']}; misses {counts['misses']}", fill="black", font=body_font)
            elif column == 0:
                draw.text((x + 2, y + 25), "GT skeleton & EDT <= 2 px; not anatomical diameter", fill="black", font=body_font)
            tile = Image.fromarray(array).convert("RGB")
            sheet.paste(tile, (x, y + header))
    return sheet, summaries


def render_test_sheet(cases, probabilities, threshold, mode):
    cell_w, cell_h, header, banner = 280, 290, 28, 48
    sheet = Image.new("RGB", (cell_w * 3, banner + (cell_h + header) * len(TEST_IDS)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f"Test previews: {mode}, t={threshold:.2f} | NO ground truth available", fill="black", font=font(16))
    draw.text((4, 26), "Display thumbnails only; native predictions and saved arrays remain unchanged.", fill="black", font=font(13))
    for row, image_id in enumerate(TEST_IDS):
        probability = probabilities[image_id]
        pictures = [
            ("Test RGB", np.uint8(np.clip(cases[image_id]["image"] * 255, 0, 255)), Image.Resampling.BILINEAR),
            ("Ensemble probability", np.uint8(np.clip(probability * 255, 0, 255)), Image.Resampling.BILINEAR),
            (f"Native prediction t={threshold:.2f}", (probability >= threshold).astype(np.uint8) * 255, Image.Resampling.NEAREST),
        ]
        for column, (title, array, resampling) in enumerate(pictures):
            y = banner + row * (cell_h + header)
            draw.text((column * cell_w + 4, y + 5), f"{image_id}: {title}", fill="black", font=font(13))
            tile = ImageOps.contain(Image.fromarray(array).convert("RGB"), (cell_w, cell_h), method=resampling)
            sheet.paste(tile, (column * cell_w, y + header))
    return sheet


def preview(data, root):
    data, root = Path(data).expanduser().resolve(), Path(root).expanduser().resolve()
    selection_path = root / "oof/selection.json"
    selection = read_json(selection_path)
    report = read_json(root / "oof/oof_report.json")
    comparison = read_json(root / "comparison.json")
    prediction = read_json(root / "submission/prediction_run.json")
    submission = read_json(root / "submission/submission_metadata.json")
    if selection["version"] != "v4.0.0" or report["selection"] != selection:
        raise ValueError("Expected consistent final v4 OOF selection")
    if Path(report["data_root"]).resolve() != data:
        raise ValueError("Preview data path differs from the OOF data path")
    mode, threshold = selection["selected_variant"], float(selection["threshold"])
    if mode not in ("none", "flip4", "d4") or selection["tta_mode"] != mode:
        raise ValueError("Unknown selected TTA mode")
    if comparison["selection"]["selected_variant"] != mode:
        raise ValueError("Comparison belongs to another selected mode")
    close(comparison["selection"]["threshold"], threshold, "comparison threshold")
    if prediction["version"] != "v4.0.0" or prediction["selected_variant"] != mode or prediction["tta_mode"] != mode:
        raise ValueError("Test prediction used a different selection")
    close(prediction["threshold"], threshold, "test prediction threshold")
    close(submission["threshold"], threshold, "submission threshold")
    if prediction["selection_sha256"] != sha(selection_path):
        raise ValueError("Test predictions refer to another selection hash")
    if submission["polarity"] != "vessel-white" or submission["resize_mode"] != "official":
        raise ValueError("Expected the native white-vessel export pipeline")
    if submission["roundtrip_verified"] is not True or submission["official_script_verified"] is not True:
        raise ValueError("Final submission verification has not completed")
    selected = report["variants"][mode]
    probability_path = Path(selected["oof_probabilities_path"]).resolve()
    if not probability_path.is_relative_to(root) or sha(probability_path) != selected["oof_probabilities_sha256"]:
        raise ValueError("Selected OOF probability path/hash mismatch")
    close(selected["native_tuned"]["threshold"], threshold, "OOF threshold")
    diagnostic = comparison["v4_candidates"][mode]["native_tuned"]
    close(diagnostic["threshold"], threshold, "diagnostic threshold")
    dice_rows = selected["native_tuned"]["per_image"]
    thin_rows = diagnostic["per_image"]
    for rows in (dice_rows, thin_rows):
        if len(rows) != 20 or {row["id"] for row in rows} != set(IDS):
            raise ValueError("Expected complete 20-image metric records")
    worst_ids = [row["id"] for row in sorted(dice_rows, key=lambda row: (row["dice"], int(row["id"])))[:4]]
    if any(row["thin_skeleton_recall"] is None or not np.isfinite(row["thin_skeleton_recall"]) for row in thin_rows):
        raise ValueError("Missing/non-finite thin-skeleton recalls")
    thin_ids = [row["id"] for row in sorted(thin_rows, key=lambda row: (row["thin_skeleton_recall"], int(row["id"])))[:4]]
    train_cases, test_cases = load_cases(data, "train"), load_cases(data, "test")
    if set(train_cases) != set(IDS) or set(test_cases) != set(IDS):
        raise ValueError("Expected all 20 train and test image IDs")
    probabilities = checked_probabilities(probability_path, train_cases)
    test_path = root / "submission/test_probabilities.npz"
    test_probabilities = checked_probabilities(test_path, test_cases)
    for row in dice_rows:
        image_id = row["id"]
        measured = binary_metrics(probabilities[image_id] >= threshold, train_cases[image_id]["mask"] > 0.5)
        close(measured["dice"], row["dice"], "actual OOF Dice")
    # Check that the four displayed test masks reproduce the already saved PNGs.
    # This reads predictions only; no test mask/ground-truth directory is opened.
    for image_id in TEST_IDS:
        with Image.open(root / f"submission/image/{image_id}.png") as image:
            saved = np.asarray(image.convert("L"))
        expected = (test_probabilities[image_id] >= threshold).astype(np.uint8) * 255
        if not np.array_equal(saved, expected):
            raise ValueError(f"Saved test probability/mask disagreement for ID {image_id}")
    thin_sheet, counts = render_thin_sheet(thin_ids, train_cases, probabilities,
                                          {row["id"]: row for row in thin_rows}, threshold, mode)
    test_sheet = render_test_sheet(test_cases, test_probabilities, threshold, mode)
    # All required final inputs and scores are checked before any preview is saved.
    worst_path = root / "oof/validation_worst4_preview.png"
    thin_path = root / "oof/thin_error_preview.png"
    test_preview_path = root / "submission/test_preview.png"
    save_preview({image_id: probabilities[image_id] for image_id in worst_ids}, train_cases, worst_path, threshold)
    thin_sheet.save(thin_path, format="PNG")
    test_sheet.save(test_preview_path, format="PNG")
    result = {"version": "v4.0.0", "selected_variant": mode, "threshold": threshold,
              "worst_dice_ids": worst_ids, "lowest_thin_skeleton_recall_ids": thin_ids,
              "thin_skeleton_counts": counts, "test_ids": TEST_IDS,
              "overlay_display_expansion": "one-pixel cross expansion for display only; exact native hits/misses computed first",
              "oof_probabilities_sha256": sha(probability_path), "test_probabilities_sha256": sha(test_path),
              "outputs": [str(path) for path in (worst_path, thin_path, test_preview_path)],
              "gpu_models_executed": False, "test_ground_truth_used": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here.parents[1] / "work/data")
    parser.add_argument("--root", type=Path, default=here)
    args = parser.parse_args()
    preview(args.data, args.root)


if __name__ == "__main__":
    main()
