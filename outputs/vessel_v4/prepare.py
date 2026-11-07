"""Validate and unpack the supplied competition archive without running its code.

Example:
    python prepare.py --zip 2026task-4.zip --destination data --report audit.json

Dependencies: Python 3.10+, NumPy, Pillow. Original files remain byte-for-byte
unchanged. The audit measures file/pixel duplication, not patient independence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image


IDS = tuple(str(i) for i in range(1, 21))
GROUPS = ("train/image", "train/label", "test/image")
EXPECTED_FILES = frozenset(
    [f"{group}/{image_id}.jpg" for group in GROUPS for image_id in IDS]
    + ["sample_submission.csv", "segmentation_to_csv.py"]
)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    """Reject symlink/junction parents before creating or replacing a file."""
    for candidate in (path, *path.parents):
        is_junction = getattr(candidate, "is_junction", lambda: False)
        if candidate.is_symlink() or is_junction():
            raise ValueError(f"Symlink/junction paths are not accepted: {candidate}")


def _safe_target(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    _reject_symlink_components(path)
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Destination escapes data directory: {relative}")
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".prepare-", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_expected_archive(zip_path: Path) -> dict[str, bytes]:
    allowed_directories = {"segmentation"}
    for relative in EXPECTED_FILES:
        for parent in PurePosixPath(f"segmentation/{relative}").parents:
            if str(parent) != ".":
                allowed_directories.add(str(parent))

    records: dict[str, zipfile.ZipInfo] = {}
    seen: set[str] = set()
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename
            # Never normalize an untrusted path into an accepted path.
            if (
                name.startswith("/")
                or "\\" in name
                or ":" in name
                or "\x00" in name
                or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
            ):
                raise ValueError(f"Unsafe archive member: {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate archive member: {name!r}")
            seen.add(name)
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"Archive symlink is not accepted: {name!r}")
            if info.is_dir():
                if name.rstrip("/") not in allowed_directories:
                    raise ValueError(f"Unexpected archive directory: {name!r}")
                continue
            if not name.startswith("segmentation/"):
                raise ValueError(f"Unexpected archive member: {name!r}")
            relative = name[len("segmentation/") :]
            if relative not in EXPECTED_FILES:
                raise ValueError(f"Unexpected data filename: {name!r}")
            if info.flag_bits & 1:
                raise ValueError(f"Encrypted archive member is not accepted: {name!r}")
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError(f"Archive member exceeds size limit: {name!r}")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Archive exceeds total uncompressed size limit")
            records[relative] = info
        missing = sorted(EXPECTED_FILES.difference(records))
        if missing:
            raise ValueError(f"Missing expected files: {missing}")
        # ZipFile.read verifies CRC; no extract/extractall and no archive code runs.
        return {relative: archive.read(records[relative]) for relative in sorted(records)}


def _image_record(relative: str, payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        image_format = image.format
        mode = image.mode
        width, height = image.size
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Multi-frame image is not accepted: {relative}")
        array = np.asarray(image).copy()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        pixel_header = f"RGB:{width}:{height}:".encode("ascii")
        decoded_hash = _sha256(pixel_header + rgb.tobytes(order="C"))
    record: dict[str, Any] = {
        "relative_path": relative,
        "id": PurePosixPath(relative).stem,
        "byte_size": len(payload),
        "sha256": _sha256(payload),
        "decoded_rgb_sha256": decoded_hash,
        "width": width,
        "height": height,
        "mode": mode,
        "actual_format": image_format,
        "filename_extension": PurePosixPath(relative).suffix,
    }
    if "/label/" in relative:
        if array.ndim != 2:
            raise ValueError(f"Expected a single-channel label: {relative}, mode={mode}")
        values, counts = np.unique(array, return_counts=True)
        record["unique_values"] = [int(value) for value in values]
        record["value_counts"] = {str(int(value)): int(count) for value, count in zip(values, counts)}
        if not set(record["unique_values"]).issubset({0, 255}):
            raise ValueError(f"Expected binary 0/255 label: {relative}, values={values.tolist()}")
        mask = array > 0
        record["positive_pixels"] = int(mask.sum())
        record["positive_ratio"] = float(mask.mean())
        label_image = Image.fromarray(array.astype(np.uint8))
        bilinear = np.asarray(label_image.resize((512, 512), Image.Resampling.BILINEAR)) > 0
        nearest = np.asarray(label_image.resize((512, 512), Image.Resampling.NEAREST)) > 0
        bilinear_count = int(bilinear.sum())
        nearest_count = int(nearest.sum())
        record["official_512_conversion"] = {
            "operation": "Pillow grayscale uint8 -> resize 512x512 BILINEAR -> greater than zero",
            "bilinear_positive_pixels": bilinear_count,
            "bilinear_positive_ratio": float(bilinear.mean()),
            "nearest_positive_pixels": nearest_count,
            "nearest_positive_ratio": float(nearest.mean()),
            "net_added_positive_pixels_vs_nearest": bilinear_count - nearest_count,
            "positive_area_increase_fraction_vs_nearest": (
                (bilinear_count - nearest_count) / nearest_count if nearest_count else None
            ),
            "added_pixels_vs_nearest": int(np.logical_and(bilinear, ~nearest).sum()),
            "removed_pixels_vs_nearest": int(np.logical_and(nearest, ~bilinear).sum()),
        }
    elif mode != "RGB":
        raise ValueError(f"Expected RGB image: {relative}, mode={mode}")
    return record, array


def _duplicates(records: list[dict[str, Any]], key: str) -> list[list[str]]:
    buckets: dict[str, list[str]] = {}
    for record in records:
        buckets.setdefault(record[key], []).append(record["relative_path"])
    return [sorted(paths) for paths in buckets.values() if len(paths) > 1]


def prepare(zip_path: str | Path, destination: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Validate the entire package, extract exact originals, and write a JSON audit.

    Existing identical files make reruns safe. Existing files with different
    bytes are rejected rather than overwritten. Returns the JSON report object.
    """
    archive_path = Path(zip_path).expanduser().resolve(strict=True)
    destination_input = Path(destination).expanduser().absolute()
    report_input = Path(report_path).expanduser().absolute()
    _reject_symlink_components(destination_input)
    _reject_symlink_components(report_input)
    root = destination_input.resolve()
    report_file = report_input.resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Destination is not a directory: {root}")
    if report_file == archive_path:
        raise ValueError("Report path must not overwrite the input archive")

    payloads = _read_expected_archive(archive_path)
    targets = {relative: _safe_target(root, relative) for relative in payloads}
    if report_file in targets.values():
        raise ValueError("Report path collides with an extracted original file")
    for relative, target in targets.items():
        if target.exists() and (not target.is_file() or target.read_bytes() != payloads[relative]):
            raise ValueError(f"Existing destination differs from archive: {target}")

    sample_text = payloads["sample_submission.csv"].decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(sample_text, newline=""))
    if reader.fieldnames != ["Id", "Predicted"]:
        raise ValueError(f"Unexpected sample submission columns: {reader.fieldnames}")
    rows = list(reader)
    sample_ids = [row["Id"] for row in rows]
    if len(sample_ids) != 20 or len(set(sample_ids)) != 20 or set(sample_ids) != set(IDS):
        raise ValueError(f"Sample IDs must be the unique integers 1..20: {sample_ids}")

    records: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        for image_id in IDS:
            relative = f"{group}/{image_id}.jpg"
            record, _ = _image_record(relative, payloads[relative])
            records.append(record)
            by_path[relative] = record
    for image_id in IDS:
        image_record = by_path[f"train/image/{image_id}.jpg"]
        label_record = by_path[f"train/label/{image_id}.jpg"]
        if (image_record["width"], image_record["height"]) != (label_record["width"], label_record["height"]):
            raise ValueError(f"Image/label size mismatch for training ID {image_id}")

    image_records = [record for record in records if "/image/" in record["relative_path"]]
    raw_duplicates = _duplicates(image_records, "sha256")
    pixel_duplicates = _duplicates(image_records, "decoded_rgb_sha256")
    cross_split = [
        paths for paths in pixel_duplicates
        if any(path.startswith("train/") for path in paths)
        and any(path.startswith("test/") for path in paths)
    ]
    if cross_split:
        raise ValueError(f"Exact decoded image duplication between train and test: {cross_split}")

    labels = [record for record in records if "/label/" in record["relative_path"]]
    original_script = payloads["segmentation_to_csv.py"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_archive": str(archive_path),
        "source_archive_sha256": _sha256(archive_path.read_bytes()),
        "destination": str(root),
        "validation_passed": True,
        "counts": {"train_images": 20, "train_labels": 20, "test_images": 20},
        "sample_submission": {
            "columns": reader.fieldnames,
            "ids_in_file_order": sample_ids,
            "rows": len(rows),
            "blank_prediction_rows": sum(not (row.get("Predicted") or "").strip() for row in rows),
            "sha256": _sha256(payloads["sample_submission.csv"]),
        },
        "original_converter": {
            "relative_path": "segmentation_to_csv.py",
            "sha256": _sha256(original_script),
            "byte_size": len(original_script),
            "executed": False,
            "preserved_byte_for_byte": True,
            "observed_contract_for_supplied_version": {
                "resize_width_height": [512, 512],
                "resize_method": "Pillow BILINEAR on grayscale uint8",
                "encoded_foreground": "grayscale value > 0 after resize",
                "rle_flatten_order": "C (row-major)",
                "rle_start_index": 1,
                "id_rule": "filename before the first dot",
            },
        },
        "label_summary": {
            "training_target": "label > 0 (observed white vessel annotations)",
            "positive_ratio_mean": float(np.mean([record["positive_ratio"] for record in labels])),
            "positive_ratio_min": min(record["positive_ratio"] for record in labels),
            "positive_ratio_max": max(record["positive_ratio"] for record in labels),
            "official_bilinear_512_positive_ratio_mean": float(np.mean([
                record["official_512_conversion"]["bilinear_positive_ratio"] for record in labels
            ])),
            "nearest_512_positive_ratio_mean": float(np.mean([
                record["official_512_conversion"]["nearest_positive_ratio"] for record in labels
            ])),
        },
        "duplicate_audit": {
            "image_file_sha256_duplicate_groups": raw_duplicates,
            "decoded_rgb_sha256_duplicate_groups": pixel_duplicates,
            "exact_train_test_duplicates": cross_split,
            "patient_independence_verified": False,
            "scope": "Exact encoded-file and decoded-RGB comparisons only; no patient identities or external image matching inferred.",
        },
        "constraints": {
            "training_performed": False,
            "external_data_used": False,
            "external_weights_used": False,
            "training_data_policy": "Use only the 20 supplied train images and their labels for training/validation; supplied test images are reserved for inference.",
            "split_manifest_created": False,
            "competition_rules_not_supplied": True,
            "polarity_conflict": "Task prose says vessel=0/background=255; supplied labels show white vessel=255/background=0 and the supplied converter encodes values >0. Do not silently invert training labels or treat the prose as confirmation of evaluator polarity.",
            "evaluation_geometry_warning": "The supplied converter resizes to 512x512 with BILINEAR then uses >0. This can enlarge foreground relative to nearest-neighbor binary resizing; compare the conversion area measurements and validate using the exact intended target convention.",
            "score_claim": "No training, Dice measurement, submission, or hidden-evaluator verification is performed by this preparation script.",
        },
        "files": records,
    }

    # Every validation above completes before the first original file is written.
    for relative, target in targets.items():
        if not target.exists():
            _atomic_write(target, payloads[relative])
        if target.read_bytes() != payloads[relative]:
            raise RuntimeError(f"Extracted file failed byte-for-byte verification: {target}")
    _atomic_write(report_file, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = prepare(args.zip_path, args.destination, args.report)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps({
        "status": "ok",
        "destination": report["destination"],
        "report": str(args.report.resolve()),
        "counts": report["counts"],
        "mean_label_positive_ratio": report["label_summary"]["positive_ratio_mean"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
