"""Strict five-fold OOF for the v4 scratch U-Net: none, flip4, and d4.

One held-out model supplies each image's prediction. The five-fold test average
is a different operation and must never be used to compute training-image OOF.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from data import load_cases
from inference import summarize, threshold_search


IDS = [str(i) for i in range(1, 21)]
MODES = {
    "none": ("validation_probabilities.npz", "at_0_5"),
    "flip4": ("validation_tta_probabilities.npz", "tta_at_0_5"),
    "d4": ("validation_d4_probabilities.npz", "d4_at_0_5"),
}
GRID = np.round(np.arange(0.20, 0.801, 0.02), 2).tolist()
OWNER = "vessel_v4.aggregate_oof.v1"
SCOPE = (
    "Five-fold OOF over the supplied 20 images; each image is predicted only by "
    "the fold whose training split excluded it. Checkpoint, TTA protocol and "
    "threshold selection reuse validation labels, so these are selection-optimistic "
    "CV scores, not an independent test estimate or a Kaggle score."
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def checked_ids(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"Expected {count} IDs")
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in value):
        raise ValueError("Invalid image ID type")
    result = [str(item) for item in value]
    if len(set(result)) != count or not set(result).issubset(IDS):
        raise ValueError("Duplicate or unknown image IDs")
    return sorted(result, key=int)


def read_probabilities(path, ids, cases):
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(ids) or set(saved.files) != set(ids):
            raise ValueError(f"{path}: expected only this fold's held-out IDs; in-training predictions are forbidden")
        values = {image_id: saved[image_id] for image_id in ids}
    if any(not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)
           or value.shape != cases[image_id]["mask"].shape
           or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1
           for image_id, value in values.items()):
        raise ValueError(f"Invalid native probability map: {path}")
    return {image_id: value.astype(np.float32) for image_id, value in values.items()}


def detail(metrics, partition):
    result = dict(metrics)
    by_id = {row["id"]: row["dice"] for row in metrics["per_image"]}
    result["per_fold"] = [{"fold": fold, "validation_ids": ids,
                           "mean_dice": float(np.mean([by_id[i] for i in ids]))}
                          for fold, ids in enumerate(partition)]
    result["std_of_fold_means_ddof_1"] = float(np.std([r["mean_dice"] for r in result["per_fold"]], ddof=1))
    return result


def aggregate(runs, data, output):
    """Return the OOF report and write oof_report.json, selection.json, 3 NPZs.

    runs is the directory containing fold0..fold4. All three probability files
    and reports must exist for all folds. Checkpoints are read one at a time on
    CPU for provenance; no model is constructed or executed.
    """
    import torch
    runs, data, output = Path(runs).resolve(strict=True), Path(data).resolve(strict=True), Path(output).resolve()
    code = Path(__file__).resolve().parent
    if not (data / "train").is_dir() and (data / "segmentation/train").is_dir():
        data = data / "segmentation"
    cases = load_cases(data, "train")
    if set(cases) != set(IDS):
        raise ValueError("Expected exactly training IDs 1 through 20")
    fold_names = {p.name for p in runs.iterdir() if p.is_dir() and re.fullmatch(r"fold[0-9]+", p.name)}
    if fold_names != {f"fold{i}" for i in range(5)}:
        raise ValueError("Expected exactly fold0 through fold4")
    oof = {mode: {} for mode in MODES}
    records, partition, seen, input_stamps = [], [], set(), {}

    def artifact(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty artifact: {path}")
        stat = path.stat()
        input_stamps[path] = (stat.st_size, stat.st_mtime_ns)
        return {"path": str(path), "bytes": stat.st_size, "sha256": sha(path)}

    for fold in range(5):
        run = runs / f"fold{fold}"
        required = ["config.json", "best.pt", "validation_report.json"] + [entry[0] for entry in MODES.values()]
        artifacts = {name: artifact(run / name) for name in required}
        config, report = read_json(run / "config.json"), read_json(run / "validation_report.json")
        ids, train = checked_ids(config.get("val_ids"), 4), checked_ids(config.get("train_ids"), 16)
        if train != sorted(set(IDS) - set(ids), key=int) or set(ids) & seen:
            raise ValueError(f"Fold leakage, incorrect complement, or duplicate held-out IDs: {run}")
        if checked_ids(report.get("validation_ids"), 4) != ids:
            raise ValueError(f"Report IDs differ from config: {run}")
        if not (config.get("version") == report.get("version") == "v4.0.0"
                and config.get("encoder_name") == "unet_v4" and config.get("fold") == fold
                and config.get("base_channels") == 24 and config.get("steps") == report.get("completed_steps") == 6000
                and config.get("pretrained_weights") is False and config.get("external_training_data") is False
                and config.get("external_fine_tuning_data") is False
                and config.get("initialization") == "random_from_scratch"
                and abs(float(config.get("thin_weight", -1)) - 0.5) < 1e-12
                and abs(float(config.get("skeleton_weight", -1)) - 0.03) < 1e-12):
            raise ValueError(f"Unexpected v4 training configuration: {run}")
        if config.get("resume") and Path(config["resume"]).resolve().parent != run:
            raise ValueError(f"Cross-run initialization/resume is forbidden: {run}")
        source_hashes = config.get("source_sha256", {})
        expected_sources = {"train.py", "model.py", "data.py", "thin_data.py", "thin_loss.py", "ema.py", "inference.py", "submission.py"}
        if set(source_hashes) != expected_sources or any(sha(code / name) != digest for name, digest in source_hashes.items()):
            raise ValueError(f"Captured source hashes do not match current v4 code: {run}")
        state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        try:
            keys = ("version", "encoder_name", "fold", "base_channels", "steps", "seed", "train_ids", "val_ids", "pretrained_weights", "external_training_data", "thin_weight", "skeleton_weight", "initial_weights_sha256", "source_sha256")
            if any(state["config"][key] != config[key] for key in keys):
                raise ValueError(f"Checkpoint provenance differs from config: {run}")
            if not (state["version"] == "v4.0.0" and state["step"] == report["best_step"]
                    and state["weight_kind"] == report["weight_kind"] == state["best_kind"]
                    and state["weight_kind"] in ("raw", "ema")
                    and abs(state["best_score"] - report["at_0_5"]["mean_dice"]) < 1e-12):
                raise ValueError(f"Checkpoint does not match selected validation report: {run}")
        finally:
            del state
            gc.collect()
        seen.update(ids)
        partition.append(ids)
        for mode, (filename, metric_key) in MODES.items():
            values = read_probabilities(run / filename, ids, cases)
            measured = summarize(values, cases, 0.5)
            if abs(float(report[metric_key]["mean_dice"]) - measured["mean_dice"]) > 1e-12:
                raise ValueError(f"Saved {mode} metric does not match probabilities: {run}")
            oof[mode].update(values)
        records.append({"fold": fold, "train_ids": train, "val_ids": ids, "artifacts": artifacts,
                        "initial_weights_sha256": config["initial_weights_sha256"],
                        "source_sha256": source_hashes, "checkpoint_step": report["best_step"],
                        "checkpoint_weight_kind": report["weight_kind"]})
    if seen != set(IDS):
        raise ValueError("Five folds do not cover each image exactly once")
    for mode in MODES:
        oof[mode] = {i: oof[mode][i] for i in IDS}
    report_path, selection_path = output / "oof_report.json", output / "selection.json"
    paths = [report_path, selection_path] + [output / f"oof_{mode}.npz" for mode in MODES]
    if any(path in input_stamps for path in paths):
        raise ValueError("Output would overwrite input")
    if any(path.exists() for path in paths) and (not report_path.is_file() or read_json(report_path).get("aggregator") != OWNER):
        raise FileExistsError("Refusing to overwrite outputs not owned by this aggregator")
    variants = {}
    for mode in MODES:
        fixed = detail(summarize(oof[mode], cases, 0.5), partition)
        tuning = threshold_search(oof[mode], cases)
        if [row["threshold"] for row in tuning["sweep"]] != GRID:
            raise ValueError("Unexpected threshold grid")
        tuning["caveat"] = SCOPE
        tuned = detail(summarize(oof[mode], cases, tuning["native_best_threshold"]), partition)
        variants[mode] = {"tta_mode": mode, "tta": mode != "none", "family_weights": {"unet_v4": 1.0},
                          "fixed_0_50": fixed, "native_tuned": tuned, "threshold_tuning": tuning,
                          "oof_probabilities_path": str(output / f"oof_{mode}.npz"),
                          "prediction_provenance": {i: {"unet_v4": {"fold": next(f for f, ids in enumerate(partition) if i in ids), "held_out": True}} for i in IDS}}
    winner = max(MODES, key=lambda mode: (variants[mode]["fixed_0_50"]["mean_dice"], -list(MODES).index(mode)))
    selected = variants[winner]
    models = [{"family": "unet_v4", "fold": record["fold"],
               "checkpoint": record["artifacts"]["best.pt"]["path"],
               "checkpoint_sha256": record["artifacts"]["best.pt"]["sha256"],
               "test_prediction_weight": 0.2, "train_ids": record["train_ids"], "val_ids": record["val_ids"]} for record in records]
    selection = {"schema_version": 1, "aggregator": OWNER, "version": "v4.0.0",
                 "selected_variant": winner, "tta_mode": winner, "tta": winner != "none",
                 "criterion": "Highest native OOF mean Dice at fixed 0.50; exact ties prefer none, flip4, then d4",
                 "threshold": selected["threshold_tuning"]["native_best_threshold"],
                 "family_weights": {"unet_v4": 1.0}, "family_weights_selected_by_optimization": False,
                 "models": models, "fixed_0_5_mean_dice": selected["fixed_0_50"]["mean_dice"],
                 "native_tuned_mean_dice": selected["native_tuned"]["mean_dice"],
                 "validation_ids": IDS, "report_path": str(report_path),
                 "test_inference_rule": "Average all five fold probability maps with weight 0.2 each using selected tta_mode; threshold once after averaging",
                 "oof_rule": "Each image uses only its own held-out fold, never predictions from the other four models trained on that image",
                 "limitations": SCOPE}
    for path, stamp in input_stamps.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != stamp:
            raise RuntimeError(f"Input changed during aggregation: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        path = output / f"oof_{mode}.npz"
        np.savez_compressed(path, **oof[mode])
        variants[mode]["oof_probabilities_sha256"] = sha(path)
    result = {"schema_version": 1, "aggregator": OWNER, "version": "v4.0.0",
              "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "data_root": str(data), "runs_root": str(runs), "ids": IDS, "fold_partition": partition,
              "family": "unet_v4", "folds": records, "variants": variants, "selection": selection,
              "threshold_grid": GRID, "pretrained_weights": False, "external_training_data": False,
              "test_images_loaded": False, "model_forward_executed": False,
              "report_path": str(report_path), "selection_path": str(selection_path), "scope": SCOPE,
              "hidden_score_caveat": "Official white-vessel conversion Dice is a conditional proxy; hidden polarity and preprocessing remain unconfirmed."}
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.runs, args.data, args.output)
    print(json.dumps({"report": result["report_path"], "selection": result["selection_path"],
                      "tta_mode": result["selection"]["tta_mode"], "threshold": result["selection"]["threshold"],
                      "fixed_0_5_mean_dice": result["selection"]["fixed_0_5_mean_dice"],
                      "native_tuned_mean_dice": result["selection"]["native_tuned_mean_dice"]}, indent=2))


if __name__ == "__main__":
    main()
