"""Lossless masks and rigorously checked RLE submissions for 2026task-4.

The supplied labels use white vessels, while the competition description says
black vessels. The hidden ground-truth polarity and resize pipeline are unknown.
Defaults follow the supplied labels and converter; they are not confirmation of
the hidden evaluator. Keep native vessel probabilities to permit re-exporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

SUBMISSION_SHAPE = (512, 512)
_EXPORTER = "vessel_baseline.submission.v1"
_METADATA_NAME = "submission_metadata.json"


def official_mask(png_or_array: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    """Reproduce the official converter: L, bilinear resize to 512, then > 0.

    ToTensor divides uint8 values by 255, so its subsequent > 0 test is exactly
    equivalent to the integer comparison here. This does not invert polarity.
    """
    if isinstance(png_or_array, (str, Path)):
        with Image.open(png_or_array) as source:
            gray = source.convert("L")
    elif isinstance(png_or_array, Image.Image):
        gray = png_or_array.convert("L")
    else:
        gray = Image.fromarray(np.asarray(png_or_array)).convert("L")
    resized = gray.resize((512, 512), Image.Resampling.BILINEAR)
    return np.asarray(resized) > 0


def encode_rle(mask: np.ndarray) -> str:
    """Encode a 2-D binary mask in row-major order with 1-based starts."""
    array = np.asarray(mask)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("mask must be a nonempty 2-D array")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
        raise ValueError("mask must have a numeric or boolean dtype")
    if not np.isfinite(array).all():
        raise ValueError("mask must contain only finite values")
    binary = array.astype(bool).ravel(order="C")
    padded = np.concatenate(([False], binary, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[::2] + 1
    lengths = changes[1::2] - changes[::2]
    pairs = np.column_stack((starts, lengths)).ravel()
    return " ".join(str(int(value)) for value in pairs)


def decode_rle(s: str, shape: tuple[int, int] = SUBMISSION_SHAPE) -> np.ndarray:
    """Decode RLE, rejecting invalid tokens, overlap, disorder, and bad bounds."""
    if (
        len(shape) != 2
        or any(isinstance(value, (bool, np.bool_)) for value in shape)
        or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in shape)
    ):
        raise ValueError("shape must contain two positive integers")
    if not isinstance(s, str):
        raise TypeError("RLE must be a string; use an empty string for an empty mask")
    tokens = s.split()
    if len(tokens) % 2:
        raise ValueError("RLE must contain start/length pairs")
    if any(re.fullmatch(r"[0-9]+", token) is None for token in tokens):
        raise ValueError("RLE tokens must be positive decimal integers")
    total = math.prod(int(value) for value in shape)
    flat = np.zeros(total, dtype=bool)
    previous_end = 0
    for start_text, length_text in zip(tokens[::2], tokens[1::2]):
        start, length = int(start_text), int(length_text)
        if start <= 0 or length <= 0:
            raise ValueError("RLE starts and lengths must be positive")
        end = start + length - 1
        if start <= previous_end:
            raise ValueError("RLE runs must be sorted and must not overlap")
        if end > total:
            raise ValueError(f"RLE run ends beyond the {total} pixels in shape {shape}")
        flat[start - 1 : end] = True
        previous_end = end
    return flat.reshape(shape, order="C")


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError(f"{csv_path}: columns must be exactly Id,Predicted")
        rows = list(reader)
    if len(rows) != 20:
        raise ValueError(f"{csv_path}: expected 20 rows, found {len(rows)}")
    ids = []
    for row in rows:
        if None in row or row.get("Id") is None or row.get("Predicted") is None:
            raise ValueError(f"{csv_path}: malformed CSV row")
        image_id = row["Id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError(f"{csv_path}: unsafe or empty image ID {image_id!r}")
        ids.append(image_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{csv_path}: duplicate image IDs")
    return rows


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Id", "Predicted"])
        writer.writeheader()
        writer.writerows(rows)


def _validate_csv(
    csv_path: Path, ids: list[str], expected_masks: Mapping[str, np.ndarray]
) -> list[dict[str, str]]:
    rows = _read_rows(csv_path)
    if {row["Id"] for row in rows} != set(ids):
        raise ValueError("submission IDs do not exactly match sample_submission.csv")
    by_id = {row["Id"]: row for row in rows}
    for image_id in ids:
        decoded = decode_rle(by_id[image_id]["Predicted"])
        if not np.array_equal(decoded, expected_masks[image_id]):
            raise ValueError(f"RLE round-trip mismatch for image ID {image_id}")
    return [by_id[image_id] for image_id in ids]


def export_submission(
    probabilities: dict[str, np.ndarray],
    output_dir: Path,
    sample_csv: Path,
    threshold: float,
    polarity: str = "vessel-white",
    resize_mode: str = "official",
    official_script: str | Path | None = None,
) -> dict[str, Any]:
    """Export 20 native vessel-probability maps and validate every CSV row.

    ``official`` thresholds native probabilities before saving a native-size PNG;
    the official L/bilinear/>0 conversion then determines the encoded pixels.
    ``probability`` bilinearly resizes float probabilities to 512 before applying
    the threshold. Both modes use >= threshold and save only 0/255 PNG pixels.

    If supplied, the original script runs in output_dir after our own generated
    CSV has been written. Its output must agree pixel-for-pixel, by ID. The final
    CSV always follows sample order. Existing submissions are overwritten only
    if accompanied by this exporter's ownership metadata. Unexpected files or
    subdirectories in image/ are rejected rather than removed.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    if polarity not in {"vessel-white", "vessel-black"}:
        raise ValueError("polarity must be vessel-white or vessel-black")
    if resize_mode not in {"official", "probability"}:
        raise ValueError("resize_mode must be official or probability")
    sample_csv = Path(sample_csv).resolve()
    sample_rows = _read_rows(sample_csv)
    ids = [row["Id"] for row in sample_rows]
    if set(probabilities) != set(ids):
        missing = sorted(set(ids) - set(probabilities))
        extra = sorted(set(probabilities) - set(ids), key=str)
        raise ValueError(f"probability IDs mismatch; missing={missing}, extra={extra}")
    checked_probabilities = {}
    for image_id in ids:
        array = np.asarray(probabilities[image_id])
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{image_id}: probabilities must be a nonempty HxW array")
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError(f"{image_id}: probabilities must be real numeric values")
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError(f"{image_id}: probabilities must be finite and in [0, 1]")
        checked_probabilities[image_id] = array.astype(np.float32)
    script_path = None
    if official_script is not None:
        script_path = Path(official_script).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(script_path)

    output_dir = Path(output_dir).resolve()
    image_dir = output_dir / "image"
    csv_path = output_dir / "submission.csv"
    metadata_path = output_dir / _METADATA_NAME
    if csv_path == sample_csv:
        raise ValueError("output submission.csv must not overwrite the sample CSV")
    if image_dir.exists():
        if not image_dir.is_dir() or image_dir.resolve().parent != output_dir:
            raise ValueError("image/ must be a direct directory inside output_dir")
        expected_names = {f"{image_id}.png" for image_id in ids}
        unexpected = [
            path.name
            for path in image_dir.iterdir()
            if not path.is_file() or path.is_symlink() or path.name not in expected_names
        ]
        if unexpected:
            raise ValueError(f"image/ contains unexpected stale files: {sorted(unexpected)}")
    if csv_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise FileExistsError("refusing to overwrite a submission without exporter metadata") from error
        if previous.get("exporter") != _EXPORTER:
            raise FileExistsError("refusing to overwrite a submission owned by another exporter")

    image_dir.mkdir(parents=True, exist_ok=True)
    masks: dict[str, np.ndarray] = {}
    rows: list[dict[str, str]] = []
    image_metadata: dict[str, dict[str, Any]] = {}
    for image_id in ids:
        native = checked_probabilities[image_id]
        if resize_mode == "probability":
            probability_image = Image.fromarray(native)
            current = np.asarray(probability_image.resize((512, 512), Image.Resampling.BILINEAR))
        else:
            current = native
        vessels = current >= threshold
        white = vessels if polarity == "vessel-white" else ~vessels
        pixels = white.astype(np.uint8) * 255
        png_path = image_dir / f"{image_id}.png"
        Image.fromarray(pixels).save(png_path, format="PNG")
        masks[image_id] = official_mask(png_path)
        rle = encode_rle(masks[image_id])
        if not np.array_equal(decode_rle(rle), masks[image_id]):
            raise RuntimeError(f"internal RLE round-trip failure for {image_id}")
        rows.append({"Id": image_id, "Predicted": rle})
        image_metadata[image_id] = {
            "native_shape_hw": list(native.shape),
            "saved_shape_hw": list(pixels.shape),
            "native_vessel_ratio": float((native >= threshold).mean()),
            "saved_vessel_ratio": float(vessels.mean()),
            "encoded_positive_ratio": float(masks[image_id].mean()),
            "rle_runs": len(rle.split()) // 2,
        }

    metadata: dict[str, Any] = {
        "exporter": _EXPORTER,
        "threshold": float(threshold),
        "threshold_comparison": ">=",
        "polarity": polarity,
        "resize_mode": resize_mode,
        "hidden_ground_truth": "unknown: polarity and resize/binarization pipeline need confirmation",
        "encoded_positive_class": "white pixels after PIL L/bilinear resize to 512 and >0",
        "rle_order": "C (row-major), 1-based starts",
        "rle_shape_hw": [512, 512],
        "sample_csv": str(sample_csv),
        "sample_ids": ids,
        "submission_csv": str(csv_path),
        "image_dir": str(image_dir),
        "official_script": str(script_path) if script_path is not None else None,
        "official_script_verified": False,
        "roundtrip_verified": False,
        "images": image_metadata,
    }
    # This marker also distinguishes our generated CSV from unrelated files on reruns.
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows(csv_path, rows)
    if script_path is not None:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"official converter exited with code {result.returncode}:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
            )
        rows = _validate_csv(csv_path, ids, masks)
        metadata["official_script_verified"] = True
    else:
        rows = _validate_csv(csv_path, ids, masks)
    _write_rows(csv_path, rows)
    _validate_csv(csv_path, ids, masks)
    metadata["roundtrip_verified"] = True
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission structure and every RLE run.")
    parser.add_argument("submission_csv", type=Path)
    parser.add_argument("--sample-csv", type=Path, required=True)
    args = parser.parse_args()
    sample = _read_rows(args.sample_csv)
    rows = _read_rows(args.submission_csv)
    if {row["Id"] for row in rows} != {row["Id"] for row in sample}:
        raise ValueError("submission IDs do not match sample IDs")
    for row in rows:
        decode_rle(row["Predicted"])
    print("Validated 20 IDs and all 512x512 RLE bounds. Pixel agreement requires export_submission.")


if __name__ == "__main__":
    _main()
