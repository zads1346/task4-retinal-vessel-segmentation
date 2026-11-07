"""Aggregate strictly held-out five-fold vessel probabilities across model families.

Each image contributes exactly one held-out prediction per encoder family.
The five checkpoints are NOT averaged on training images. The equal-family
ensemble averages those held-out predictions only. Test-time inference may use
all five folds per selected family, as explicitly described in selection.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from data import load_cases
from inference import summarize, threshold_search


_AGGREGATOR = "vessel_v3.aggregate_oof.v1"
_IDS = [str(index) for index in range(1, 21)]
_GRID = np.round(np.arange(0.20, 0.801, 0.02), 2).tolist()
_SCOPE = (
    "Five-fold out-of-fold predictions from the supplied 20 images only: each "
    "image is predicted by one checkpoint per family whose training split "
    "excluded that image. Fold checkpoints, model/TTA variants, and thresholds "
    "were selected using these validation labels, so the reported CV scores "
    "remain selection-optimistic; they are not independent test or Kaggle scores."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _normalize_ids(value: Any, label: str, count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label}: expected a list of {count} image IDs")
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in value):
        raise ValueError(f"{label}: IDs must be strings or integers")
    ids = [str(item) for item in value]
    if len(set(ids)) != count or not set(ids).issubset(_IDS):
        raise ValueError(f"{label}: duplicate or unexpected image IDs")
    return sorted(ids, key=int)


def _probabilities(path: Path, val_ids: list[str], cases: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != set(val_ids):
            raise ValueError(
                f"{path}: must contain exactly its four held-out IDs; "
                "in-training predictions must not enter OOF"
            )
        result = {}
        for image_id in val_ids:
            array = np.asarray(archive[image_id])
            if (
                not np.issubdtype(array.dtype, np.number)
                or np.iscomplexobj(array)
                or array.shape != cases[image_id]["mask"].shape
                or not np.isfinite(array).all()
                or array.min() < 0
                or array.max() > 1
            ):
                raise ValueError(f"Invalid native probability array {image_id}: {path}")
            result[image_id] = array.astype(np.float32)
    return result


def _bootstrap_interval(values: list[float], seed: int = 2026) -> dict[str, Any]:
    """A descriptive image-resampling interval, not a model-selection correction."""
    values_array = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(seed).integers(0, len(values), size=(5000, len(values)))
    means = values_array[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "lower": float(low),
        "upper": float(high),
        "confidence_level": 0.95,
        "resamples": 5000,
        "seed": seed,
        "method": "Percentile bootstrap of 20 per-image Dice values",
        "limitation": (
            "Descriptive conditional interval: does not account for overlapping "
            "fold training sets, checkpoint/variant/threshold selection, or new-data shifts."
        ),
    }


def _metric_detail(report: dict[str, Any], partition: list[list[str]]) -> dict[str, Any]:
    result = dict(report)
    by_id = {row["id"]: row for row in report["per_image"]}
    fold_rows = []
    for fold, ids in enumerate(partition):
        fold_rows.append({
            "fold": fold,
            "validation_ids": ids,
            "mean_dice": float(np.mean([by_id[image_id]["dice"] for image_id in ids])),
        })
    result["per_fold"] = fold_rows
    result["std_of_fold_means_ddof_1"] = float(np.std([row["mean_dice"] for row in fold_rows], ddof=1))
    result["descriptive_image_bootstrap_95_interval"] = _bootstrap_interval(
        [by_id[image_id]["dice"] for image_id in _IDS]
    )
    return result


def aggregate(runs, data, output) -> dict[str, Any]:
    """Validate complete families, write OOF maps/reports, and return the report.

    ``runs`` is a sequence of family directories containing fold0 through fold4.
    Each fold needs best.pt, config.json, validation_report.json, and
    validation_probabilities.npz. A complete second pass is enabled only when
    validation_tta_probabilities.npz exists for every fold of every family.

    Configs must declare encoder_name, pretrained_weights=true, and
    external_fine_tuning_data=false. All families must use identical folds.
    Checkpoints are hashed but are not deserialized or executed by this utility.
    """
    if isinstance(runs, (str, Path)):
        raise TypeError("runs must be a sequence of model-family directories")
    run_roots = [Path(path).expanduser().resolve(strict=True) for path in runs]
    if not run_roots or len(set(run_roots)) != len(run_roots):
        raise ValueError("At least one distinct model-family directory is required")
    data_root = Path(data).expanduser().resolve(strict=True)
    if not (data_root / "train").is_dir() and (data_root / "segmentation" / "train").is_dir():
        data_root = data_root / "segmentation"
    cases = load_cases(data_root, "train")
    if set(cases) != set(_IDS):
        raise ValueError("Expected exactly the supplied training image IDs 1 through 20")
    output_dir = Path(output).expanduser().resolve()
    report_path = output_dir / "oof_report.json"
    selection_path = output_dir / "selection.json"

    family_infos = []
    used_names = set()
    partition = None
    input_stats: dict[Path, tuple[int, int]] = {}

    def artifact(path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required artifact: {path}")
        stat = path.stat()
        input_stats[path] = (stat.st_size, stat.st_mtime_ns)
        return {"path": str(path), "bytes": stat.st_size, "sha256": _sha256(path)}

    # Validate every fold and partition before writing any output.
    for family_root in run_roots:
        if not family_root.is_dir():
            raise NotADirectoryError(family_root)
        observed_folds = {
            path.name for path in family_root.iterdir()
            if path.is_dir() and re.fullmatch(r"fold[0-9]+", path.name)
        }
        if observed_folds != {f"fold{fold}" for fold in range(5)}:
            raise ValueError(f"{family_root}: requires exactly fold0 through fold4")
        folds = []
        family_name = None
        seen_ids = set()
        for fold in range(5):
            fold_root = family_root / f"fold{fold}"
            artifacts = {name: artifact(fold_root / name) for name in (
                "config.json", "best.pt", "validation_report.json", "validation_probabilities.npz"
            )}
            config = _read_json(fold_root / "config.json")
            report = _read_json(fold_root / "validation_report.json")
            name = config.get("encoder_name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Invalid or missing encoder_name in {fold_root}")
            if name == "equal_ensemble":
                raise ValueError("encoder_name equal_ensemble is reserved")
            if family_name is None:
                family_name = name
            if name != family_name:
                raise ValueError(f"Encoder family changes between folds in {family_root}")
            if config.get("pretrained_weights") is not True:
                raise ValueError(f"{fold_root}: pretrained_weights must explicitly be true")
            if config.get("external_fine_tuning_data") is not False:
                raise ValueError(f"{fold_root}: external_fine_tuning_data must explicitly be false")
            if config.get("fold", fold) != fold:
                raise ValueError(f"{fold_root}: config fold index does not match directory")
            val_ids = _normalize_ids(config.get("val_ids"), str(fold_root) + " val_ids", 4)
            train_ids = _normalize_ids(config.get("train_ids"), str(fold_root) + " train_ids", 16)
            reported_ids = _normalize_ids(report.get("validation_ids"), str(fold_root) + " report IDs", 4)
            if train_ids != sorted(set(_IDS) - set(val_ids), key=int) or set(train_ids) & set(val_ids):
                raise ValueError(f"Training/validation leakage or incorrect complement in {fold_root}")
            if reported_ids != val_ids or seen_ids & set(val_ids):
                raise ValueError(f"Repeated held-out IDs or report/config mismatch in {fold_root}")
            seen_ids.update(val_ids)
            plain = _probabilities(fold_root / "validation_probabilities.npz", val_ids, cases)
            measured = summarize(plain, cases, 0.5)
            if "at_0_5" in report:
                recorded = float(report["at_0_5"]["mean_dice"])
                if not np.isfinite(recorded) or abs(recorded - measured["mean_dice"]) > 1e-7:
                    raise ValueError(f"Saved report and normal OOF probabilities disagree: {fold_root}")
            tta_path = fold_root / "validation_tta_probabilities.npz"
            folds.append({
                "fold": fold,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "artifacts": artifacts,
                "initialization": config.get("initialization"),
                "pretrained_weights_source": config.get("pretrained_weights_source", config.get("weights_source")),
                "pretrained_dataset": config.get("pretrained_dataset"),
                "pretrained_sha256": config.get("pretrained_sha256"),
                "pretrained_weights_path": config.get("weights"),
                "checkpoint_step_from_report": report.get("best_step", report.get("checkpoint_step")),
                "checkpoint_weight_kind_from_report": report.get("weight_kind"),
                "tta_fixed_0_5_mean_from_report": report.get("tta_at_0_5", {}).get("mean_dice"),
                "plain": plain,
                "tta_path": tta_path,
            })
        if seen_ids != set(_IDS):
            raise ValueError(f"{family_root}: held-out folds must cover all 20 IDs exactly once")
        current_partition = [item["val_ids"] for item in folds]
        if partition is None:
            partition = current_partition
        if current_partition != partition:
            raise ValueError(f"Fold partition differs across model families: {family_root}")
        if family_name in used_names:
            raise ValueError(f"Duplicate encoder family name: {family_name}")
        used_names.add(family_name)
        family_infos.append({"name": family_name, "root": str(family_root), "folds": folds})

    family_infos.sort(key=lambda item: item["name"])
    missing_tta = [str(fold["tta_path"]) for family in family_infos for fold in family["folds"] if not fold["tta_path"].is_file()]
    tta_complete = not missing_tta
    family_oof: dict[tuple[str, bool], dict[str, np.ndarray]] = {}
    for family in family_infos:
        plain = {}
        augmented = {}
        for fold in family["folds"]:
            # Dict update is safe only after the strict disjointness checks above.
            plain.update(fold.pop("plain"))
            if tta_complete:
                fold["artifacts"]["validation_tta_probabilities.npz"] = artifact(fold["tta_path"])
                fold_augmented = _probabilities(fold["tta_path"], fold["val_ids"], cases)
                recorded_tta = fold["tta_fixed_0_5_mean_from_report"]
                if recorded_tta is not None:
                    measured_tta = summarize(fold_augmented, cases, 0.5)["mean_dice"]
                    if not np.isfinite(float(recorded_tta)) or abs(float(recorded_tta) - measured_tta) > 1e-7:
                        raise ValueError(f"Saved TTA report and probabilities disagree: {fold['tta_path']}")
                augmented.update(fold_augmented)
            del fold["tta_path"]
        family_oof[(family["name"], False)] = {image_id: plain[image_id] for image_id in _IDS}
        if tta_complete:
            family_oof[(family["name"], True)] = {image_id: augmented[image_id] for image_id in _IDS}

    names = [item["name"] for item in family_infos]
    candidate_inputs = []
    for tta in ([False, True] if tta_complete else [False]):
        suffix = "tta" if tta else "no_tta"
        for name in names:
            candidate_inputs.append((f"{name}_{suffix}", tta, {name: 1.0}, family_oof[(name, tta)]))
        # The only multi-family blend is a predefined equal-weight average.
        ensemble = {}
        for image_id in _IDS:
            accumulator = np.zeros_like(family_oof[(names[0], tta)][image_id], dtype=np.float32)
            for name in names:
                accumulator += family_oof[(name, tta)][image_id] / len(names)
            ensemble[image_id] = accumulator
        candidate_inputs.append((f"equal_ensemble_{suffix}", tta, {name: 1.0 / len(names) for name in names}, ensemble))

    output_files = [report_path, selection_path] + [output_dir / f"oof_{item[0]}.npz" for item in candidate_inputs]
    if any(path in input_stats for path in output_files):
        raise ValueError("OOF output would overwrite an input artifact")
    if any(path.exists() for path in output_files):
        if not report_path.is_file() or _read_json(report_path).get("aggregator") != _AGGREGATOR:
            raise FileExistsError("Refusing to overwrite OOF output without matching aggregator metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {}
    for name, tta, weights, probabilities in candidate_inputs:
        fixed = _metric_detail(summarize(probabilities, cases, 0.5), partition)
        tuning = threshold_search(probabilities, cases)
        if [row["threshold"] for row in tuning["sweep"]] != _GRID:
            raise ValueError("threshold_search does not match the declared 0.20:0.02:0.80 grid")
        tuning["caveat"] = _SCOPE
        threshold = float(tuning["native_best_threshold"])
        tuned = _metric_detail(summarize(probabilities, cases, threshold), partition)
        probability_path = output_dir / f"oof_{name}.npz"
        np.savez_compressed(probability_path, **probabilities)
        variants[name] = {
            "tta": tta,
            "family_weights": weights,
            "family_weights_selected_by_optimization": False,
            "oof_probabilities_path": str(probability_path),
            "oof_probabilities_sha256": _sha256(probability_path),
            "fixed_0_50": fixed,
            "native_tuned": tuned,
            "threshold_tuning": tuning,
            "prediction_provenance": {
                image_id: {family: {"fold": next(i for i, ids in enumerate(partition) if image_id in ids), "held_out": True} for family in weights}
                for image_id in _IDS
            },
        }

    # Choose architecture/TTA using fixed 0.5, then tune only the selected map's
    # operating threshold. Exact ties prefer no TTA, fewer families, then name.
    selected_name = sorted(variants, key=lambda name: (
        -variants[name]["fixed_0_50"]["mean_dice"],
        variants[name]["tta"],
        len(variants[name]["family_weights"]),
        name,
    ))[0]
    selected = variants[selected_name]
    selected_models = []
    for family in family_infos:
        if family["name"] not in selected["family_weights"]:
            continue
        family_weight = selected["family_weights"][family["name"]]
        for fold in family["folds"]:
            selected_models.append({
                "family": family["name"],
                "fold": fold["fold"],
                "checkpoint": fold["artifacts"]["best.pt"]["path"],
                "checkpoint_sha256": fold["artifacts"]["best.pt"]["sha256"],
                "test_prediction_weight": family_weight / 5,
                "train_ids": fold["train_ids"],
                "val_ids": fold["val_ids"],
            })
    selection = {
        "schema_version": 1,
        "aggregator": _AGGREGATOR,
        "version": "v3.0.0",
        "selected_variant": selected_name,
        "criterion": "Highest native 20-image OOF mean Dice at fixed threshold 0.50; ties prefer no TTA, fewer families, then name",
        "threshold": selected["threshold_tuning"]["native_best_threshold"],
        "tta": selected["tta"],
        "family_weights": selected["family_weights"],
        "family_weights_selected_by_optimization": False,
        "models": selected_models,
        "test_inference_rule": "For each selected family average all five fold probabilities, then combine families using these fixed family_weights; apply TTA consistently and threshold once after the final average",
        "oof_rule": "For each image use only its one held-out fold per family, then combine families; never average other folds' in-training predictions into OOF",
        "fixed_0_5_mean_dice": selected["fixed_0_50"]["mean_dice"],
        "native_tuned_mean_dice": selected["native_tuned"]["mean_dice"],
        "validation_ids": _IDS,
        "report_path": str(report_path),
        "limitations": _SCOPE,
    }
    for path, original_stat in input_stats.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != original_stat:
            raise RuntimeError(f"Input changed while OOF aggregation was running: {path}")
    result = {
        "schema_version": 1,
        "aggregator": _AGGREGATOR,
        "version": "v3.0.0",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "data_root": str(data_root),
        "ids": _IDS,
        "fold_partition": partition,
        "families": family_infos,
        "pretrained_weights": True,
        "external_fine_tuning_data": False,
        "data_policy_evidence": "Supplied train split only; every fold config explicitly declares pretrained weights and no external fine-tuning data. This utility verifies saved metadata and probabilities, not process-level training I/O.",
        "test_images_loaded": False,
        "checkpoint_models_executed": False,
        "tta_complete": tta_complete,
        "missing_tta_artifacts": missing_tta,
        "tta_policy": "TTA candidates are compared only when all five folds of every family have complete TTA probability files",
        "threshold_grid": _GRID,
        "variants": variants,
        "selection": selection,
        "report_path": str(report_path),
        "selection_path": str(selection_path),
        "scope": _SCOPE,
        "hidden_score_caveat": "Official white-vessel conversion Dice is a conditional proxy; hidden ground-truth polarity and resize/binarization remain unconfirmed.",
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Encoder-family directories containing fold0..fold4")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.runs, args.data, args.output)
    print(json.dumps({
        "report": report["report_path"],
        "selection": report["selection_path"],
        "selected_variant": report["selection"]["selected_variant"],
        "fixed_0_5_mean_dice": report["selection"]["fixed_0_5_mean_dice"],
        "threshold": report["selection"]["threshold"],
        "native_tuned_mean_dice": report["selection"]["native_tuned_mean_dice"],
    }, indent=2))


if __name__ == "__main__":
    main()
