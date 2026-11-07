"""Compare saved validation probabilities on the same held-out training images.

This utility never loads test images/labels or checkpoints, and writes only the
requested JSON output. Scores remain biased by repeated use of one holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from data import load_cases
from inference import summarize, threshold_search


FIXED_THRESHOLDS = {"fixed_0_50": 0.50, "fixed_0_34": 0.34}
COMMON_GRID = np.round(np.arange(0.20, 0.801, 0.02), 2).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdecimal() else (1, value)


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": report["threshold"],
        "mean_dice": report["mean_dice"],
        "std_dice": report["std_dice"],
        "micro_dice": report["micro_dice"],
        "official_white_vessel_proxy_mean_dice": report["official_white_vessel_proxy_mean_dice"],
        "per_image": [
            {key: row[key] for key in ("id", "dice", "precision", "recall")}
            for row in report["per_image"]
        ],
    }


def _read_variant(
    run_dir: Path,
    cases: dict[str, dict],
    *,
    expected_tta: bool,
    fallback_version: str,
) -> tuple[dict[str, Any], list[str], set[Path]]:
    run_dir = run_dir.expanduser().resolve(strict=True)
    report_path = run_dir / "validation_report.json"
    probability_path = run_dir / "validation_probabilities.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_ids = report.get("validation_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"Missing validation_ids list in {report_path}")
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in raw_ids):
        raise ValueError(f"Invalid validation ID type in {report_path}")
    ids = [str(value) for value in raw_ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate validation IDs in {report_path}")
    ids = sorted(ids, key=_id_key)
    if not set(ids).issubset(cases):
        raise ValueError(f"Validation IDs not present in supplied training data: {report_path}")
    if bool(report.get("tta", False)) != expected_tta:
        raise ValueError(f"Report TTA flag does not match its comparison role: {report_path}")
    if "at_0_5" in report:
        rows = report["at_0_5"].get("per_image", [])
        row_ids = [str(row["id"]) for row in rows]
        if len(row_ids) != len(ids) or set(row_ids) != set(ids):
            raise ValueError(f"Reported metric image IDs differ from validation_ids: {report_path}")

    probabilities: dict[str, np.ndarray] = {}
    with np.load(probability_path, allow_pickle=False) as archive:
        if len(archive.files) != len(ids) or set(archive.files) != set(ids):
            raise ValueError(f"NPZ IDs do not exactly match the report: {probability_path}")
        for image_id in ids:
            probability = np.asarray(archive[image_id])
            if (
                not np.issubdtype(probability.dtype, np.number)
                or np.iscomplexobj(probability)
                or probability.shape != cases[image_id]["mask"].shape
                or not np.isfinite(probability).all()
                or probability.min() < 0
                or probability.max() > 1
            ):
                raise ValueError(f"Invalid native probability map for ID {image_id}: {probability_path}")
            probabilities[image_id] = probability

    measured = {
        name: _compact_metrics(summarize(probabilities, cases, threshold))
        for name, threshold in FIXED_THRESHOLDS.items()
    }
    if "at_0_5" in report:
        prior_score = float(report["at_0_5"]["mean_dice"])
        actual_score = measured["fixed_0_50"]["mean_dice"]
        if not np.isfinite(prior_score) or abs(prior_score - actual_score) > 1e-7:
            raise ValueError(
                f"Saved report and recomputed Dice disagree for {run_dir}: "
                f"reported={prior_score}, recomputed={actual_score}. Check dataset/probability provenance."
            )
    tuning = threshold_search(probabilities, cases)
    actual_grid = [row["threshold"] for row in tuning["sweep"]]
    if actual_grid != COMMON_GRID:
        raise ValueError("inference.threshold_search no longer uses the declared common threshold grid")
    native_threshold = float(tuning["native_best_threshold"])
    measured["native_tuned"] = _compact_metrics(summarize(probabilities, cases, native_threshold))
    config_path = run_dir / "config.json"
    inputs = {report_path, probability_path}
    config_hash = None
    if config_path.is_file():
        inputs.add(config_path)
        config_hash = _sha256(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_val_ids = [str(value) for value in config.get("val_ids", [])]
        if len(config_val_ids) != len(ids) or set(config_val_ids) != set(ids):
            raise ValueError(f"Run config validation IDs differ from report: {config_path}")
        if set(str(value) for value in config.get("train_ids", [])) & set(ids):
            raise ValueError(f"Training/validation IDs overlap in {config_path}")
    result = {
        "run_dir": str(run_dir),
        "version": report.get("version", fallback_version),
        "tta": expected_tta,
        "weight_kind": report.get("weight_kind", "raw"),
        "checkpoint_step": report.get("best_step", report.get("checkpoint_step")),
        "checkpoint_path_from_report": report.get("checkpoint"),
        "report_sha256": _sha256(report_path),
        "probabilities_sha256": _sha256(probability_path),
        "config_sha256": config_hash,
        "native_best_threshold": native_threshold,
        "official_proxy_best_threshold": float(tuning["official_proxy_best_threshold"]),
        **measured,
    }
    return result, ids, inputs


def _compare_pair(baseline: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_name in (*FIXED_THRESHOLDS, "native_tuned"):
        left, right = baseline[metric_name], new[metric_name]
        left_rows = {row["id"]: row for row in left["per_image"]}
        right_rows = {row["id"]: row for row in right["per_image"]}
        result[metric_name] = {
            "v1_threshold": left["threshold"],
            "v2_threshold": right["threshold"],
            "v1_mean_dice": left["mean_dice"],
            "v2_mean_dice": right["mean_dice"],
            "delta_mean_dice": right["mean_dice"] - left["mean_dice"],
            "delta_micro_dice": right["micro_dice"] - left["micro_dice"],
            "delta_official_white_vessel_proxy_mean_dice": (
                right["official_white_vessel_proxy_mean_dice"]
                - left["official_white_vessel_proxy_mean_dice"]
            ),
            "per_image": [
                {
                    "id": image_id,
                    "v1_dice": left_rows[image_id]["dice"],
                    "v2_dice": right_rows[image_id]["dice"],
                    "delta_dice": right_rows[image_id]["dice"] - left_rows[image_id]["dice"],
                }
                for image_id in sorted(left_rows, key=_id_key)
            ],
        }
    return result


def compare_versions(
    v1_dir: str | Path,
    v2_dir: str | Path,
    data: str | Path,
    output: str | Path,
    v1_tta_dir: str | Path | None = None,
    v2_tta_dir: str | Path | None = None,
) -> dict[str, Any]:
    cases = load_cases(Path(data), "train")  # Test images/hidden labels are not accessed.
    variants: dict[str, dict[str, Any]] = {}
    input_paths: set[Path] = set()
    validation_ids: list[str] | None = None
    specifications = [("v1_no_tta", v1_dir, False, "v1.0.0"), ("v2_no_tta", v2_dir, False, "v2.0.0")]
    if v1_tta_dir is not None:
        specifications.append(("v1_tta", v1_tta_dir, True, "v1.0.0"))
    if v2_tta_dir is not None:
        specifications.append(("v2_tta", v2_tta_dir, True, "v2.0.0"))
    for name, run_dir, tta, fallback_version in specifications:
        variant, ids, paths = _read_variant(Path(run_dir), cases, expected_tta=tta, fallback_version=fallback_version)
        if validation_ids is None:
            validation_ids = ids
        elif ids != validation_ids:
            raise ValueError(f"Holdout IDs differ between comparison variants: {name}: {ids} != {validation_ids}")
        variants[name] = variant
        input_paths.update(paths)

    for family in ("v1", "v2"):
        plain = variants[f"{family}_no_tta"]
        tta = variants.get(f"{family}_tta")
        if tta is None:
            continue
        for key in ("checkpoint_step", "weight_kind", "version"):
            if plain[key] != tta[key]:
                raise ValueError(f"{family} TTA and non-TTA reports use different {key}")
        reported_checkpoint = tta["checkpoint_path_from_report"]
        expected_checkpoint = (Path(plain["run_dir"]) / "best.pt").resolve()
        if not reported_checkpoint or Path(reported_checkpoint).resolve() != expected_checkpoint:
            raise ValueError(f"{family} TTA report must reference the same run's best.pt: {expected_checkpoint}")

    comparisons = {"no_tta": _compare_pair(variants["v1_no_tta"], variants["v2_no_tta"])}
    if "v1_tta" in variants and "v2_tta" in variants:
        comparisons["tta"] = _compare_pair(variants["v1_tta"], variants["v2_tta"])
    candidates = [name for name in ("v2_no_tta", "v2_tta") if name in variants]
    selected_name = max(
        candidates,
        key=lambda name: (variants[name]["fixed_0_50"]["mean_dice"], not variants[name]["tta"]),
    )
    selected = variants[selected_name]
    corresponding_v1 = "v1_tta" if selected["tta"] else "v1_no_tta"
    matched_delta = None
    if corresponding_v1 in variants:
        matched_delta = selected["fixed_0_50"]["mean_dice"] - variants[corresponding_v1]["fixed_0_50"]["mean_dice"]
    result = {
        "schema_version": 1,
        "validation_ids": validation_ids,
        "data_root": str(Path(data).resolve()),
        "common_threshold_grid": COMMON_GRID,
        "variants": variants,
        "like_for_like": comparisons,
        "selection": {
            "candidate_family": "v2 only; this does not automatically promote v2 over v1",
            "criterion": "Highest native mean Dice at fixed threshold 0.50; exact ties prefer no TTA",
            "selected_variant": selected_name,
            "tta": selected["tta"],
            "selected_native_threshold": selected["native_best_threshold"],
            "fixed_0_50_mean_dice": selected["fixed_0_50"]["mean_dice"],
            "native_tuned_mean_dice": selected["native_tuned"]["mean_dice"],
            "same_inference_v1_fixed_0_50_delta": matched_delta,
            "same_inference_v1_comparison_available": corresponding_v1 in variants,
        },
        "caveats": [
            "The same holdout was used repeatedly for model/checkpoint/variant/threshold selection. Scores are selection-biased and are not independent test estimates or five-fold OOF results.",
            "Fixed threshold 0.34 is included as a common baseline reference; if originally tuned on these images, it is not an independent prespecified threshold.",
            "native_tuned compares each variant at its own threshold from the same common grid; fixed-threshold comparisons are reported separately.",
            "An across-version TTA comparison is included only when both TTA reports are supplied. TTA versus no-TTA scores are not labeled a matched across-version comparison.",
            "The official-white-vessel 512 conversion metric is a proxy. Hidden ground-truth polarity and preprocessing remain unknown; no Kaggle score is claimed.",
            "Only supplied training images/labels and saved held-out predictions are read. No test ground truth, external data, pretrained weights, or checkpoint execution is used.",
        ],
    }
    output_path = Path(output).expanduser().resolve()
    if output_path in input_paths:
        raise ValueError("Output path must not overwrite an input artifact")
    if output_path.suffix.lower() != ".json":
        raise ValueError("Comparison output must have a .json suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1-tta-dir", type=Path)
    parser.add_argument("--v2-tta-dir", type=Path)
    args = parser.parse_args()
    result = compare_versions(**vars(args))
    print(json.dumps({"output": str(args.output.resolve()), "validation_ids": result["validation_ids"], "selection": result["selection"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
