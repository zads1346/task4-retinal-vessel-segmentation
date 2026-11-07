"""CPU-only independent integrity audit of the completed v3 five-fold release.

Run only after all ten folds, OOF aggregation, and test prediction finish. This
script does not import training, inference, aggregation, or submission modules.
It reads one local checkpoint at a time, never executes a model, and writes only
the requested audit JSON. Checkpoints must be trusted artifacts from this run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

# This changes only this audit process, and ensures no accidental GPU use.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image


IDS = [str(index) for index in range(1, 21)]
FAMILIES = ("resnet18", "resnet34")
GRID = np.round(np.arange(0.20, 0.801, 0.02), 2).tolist()
DATA_SHA256 = "6f6e36243092f4cf55c1be4ae7c5227e3c3686501e25487ada7ef03fa9b7634a"
OFFICIAL_WEIGHT_SHA256 = {
    "resnet18": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "resnet34": "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
}


class AuditStopped(RuntimeError):
    pass


class Audit:
    def __init__(self):
        self.checks = []
        self.hashes = {}
        self.input_stats = {}
        self.details = {}

    def check(self, name, condition, details=None, *, required=False):
        row = {"name": name, "passed": bool(condition)}
        if details is not None:
            row["details"] = details
        self.checks.append(row)
        if required and not condition:
            raise AuditStopped(name)
        return bool(condition)

    def sha(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if path in self.input_stats:
            self.check("input_stable_before_reuse:" + str(path), self.input_stats[path] == stamp, required=True)
            return self.hashes[str(path)]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.input_stats[path] = stamp
        self.hashes[str(path)] = digest.hexdigest()
        return digest.hexdigest()

    def json(self, path):
        self.sha(path)
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return value

    def probabilities(self, path, ids, shapes):
        self.sha(path)
        with np.load(path, allow_pickle=False) as archive:
            self.check("probability_ids:" + str(path), len(archive.files) == len(ids) and set(archive.files) == set(ids), required=True)
            result = {image_id: archive[image_id] for image_id in ids}
        valid = all(
            np.issubdtype(value.dtype, np.number)
            and not np.iscomplexobj(value)
            and value.shape == shapes[image_id]
            and np.isfinite(value).all()
            and value.min() >= 0
            and value.max() <= 1
            for image_id, value in result.items()
        )
        self.check("probability_shapes_and_range:" + str(path), valid, required=True)
        return result


def _native_metrics(probabilities, labels, threshold):
    rows = []
    for image_id in sorted(probabilities, key=int):
        prediction, target = probabilities[image_id] >= threshold, labels[image_id]
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        rows.append({
            "id": image_id, "tp": tp, "fp": fp, "fn": fn,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        })
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    return {
        "threshold": float(threshold),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "std_dice": float(np.std([row["dice"] for row in rows])),
        "micro_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "per_image": rows,
    }


def _metric_matches(measured, reported):
    if any(abs(measured[key] - float(reported[key])) > 1e-12 for key in ("mean_dice", "std_dice", "micro_dice")):
        return False
    if "per_image" in reported:
        original = {str(row["id"]): row for row in reported["per_image"]}
        return set(original) == {row["id"] for row in measured["per_image"]} and all(
            abs(row["dice"] - float(original[row["id"]]["dice"])) <= 1e-12
            for row in measured["per_image"]
        )
    return True


def _decode_rle(text):
    tokens = text.split()
    if len(tokens) % 2 or any(re.fullmatch(r"[0-9]+", token) is None for token in tokens):
        raise ValueError("RLE must contain positive integer start/length pairs")
    flat = np.zeros(512 * 512, dtype=bool)
    last_end = 0
    for index in range(0, len(tokens), 2):
        start, length = int(tokens[index]), int(tokens[index + 1])
        end = start + length - 1
        if start <= last_end or start <= 0 or length <= 0 or end > flat.size:
            raise ValueError("RLE is out of bounds, unsorted, or overlapping")
        flat[start - 1:end] = True
        last_end = end
    return flat.reshape(512, 512)


def _snapshot_previous(audit, path):
    snapshot = audit.json(path)
    summaries = []
    for version in snapshot["versions"]:
        root = Path(version["root"])
        expected = {row["path"]: row for row in version["files"]}
        current = []
        excluded = set(snapshot["excluded_directory_names"])
        for file in root.rglob("*"):
            if file.is_file() and not (set(file.relative_to(root).parts) & excluded) and file.suffix not in snapshot["excluded_suffixes"]:
                current.append({"path": file.relative_to(root).as_posix(), "bytes": file.stat().st_size, "sha256": audit.sha(file)})
        current.sort(key=lambda row: row["path"])
        by_path = {row["path"]: row for row in current}
        changed = [name for name in expected if name not in by_path or by_path[name] != expected[name]]
        added = sorted(set(by_path) - set(expected))
        canonical = "\n".join(row["path"] + "\0" + row["sha256"] for row in current).encode("utf-8")
        root_hash = hashlib.sha256(canonical).hexdigest()
        archive_hash = audit.sha(version["source_archive"]["path"])
        audit.check(version["version"] + ":previous_files_preserved", not changed and not added, {"changed_or_missing": changed, "added": added})
        audit.check(version["version"] + ":previous_root_digest", root_hash == version["root_sha256"])
        audit.check(version["version"] + ":previous_archive_digest", archive_hash == version["source_archive"]["sha256"])
        summaries.append({"version": version["version"], "file_count": len(current), "root_sha256": root_hash, "source_zip_sha256": archive_hash})
    audit.details["previous_versions"] = summaries


def _data_provenance(audit, root, data, archive_override):
    manifest = audit.json(root / "data_audit.json")
    archive_path = Path(archive_override or manifest["source_archive"])
    audit.check("original_data_archive_sha256", audit.sha(archive_path) == DATA_SHA256, required=True)
    matches = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename).relative_to("segmentation")
            local = data / relative
            matches.append(local.is_file() and audit.sha(local) == hashlib.sha256(archive.read(entry.filename)).hexdigest())
    audit.check("all_62_supplied_files_match_original_archive", len(matches) == 62 and all(matches), required=True)
    labels, shapes, test_shapes, test_input_hashes = {}, {}, {}, {}
    expected_names = {f"{image_id}.jpg" for image_id in IDS}
    for directory in (data / "train/image", data / "train/label", data / "test/image"):
        audit.check("exact_20_data_files:" + str(directory), {p.name for p in directory.iterdir()} == expected_names, required=True)
    for image_id in IDS:
        with Image.open(data / "train/label" / f"{image_id}.jpg") as image:
            labels[image_id] = np.asarray(image.convert("L")) > 127
            shapes[image_id] = labels[image_id].shape
        with Image.open(data / "test/image" / f"{image_id}.jpg") as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
            test_shapes[image_id] = rgb.shape[:2]
            test_input_hashes[image_id] = hashlib.sha256(rgb.tobytes()).hexdigest()
    audit.details["data"] = {"archive_sha256": DATA_SHA256, "archive_files_verified": len(matches), "train_images": 20, "train_labels": 20, "test_images": 20}
    return labels, shapes, test_shapes, test_input_hashes


def _folds_and_checkpoints(audit, root, report, data, labels, shapes):
    import torch
    torch.set_num_threads(1)
    sources = audit.json(root / "pretrained_sources.json")["sources"]
    audit.check("exact_two_official_encoder_families", set(sources) == set(FAMILIES), required=True)
    for family in FAMILIES:
        audit.check(family + ":official_pretrained_digest", sources[family]["sha256"] == OFFICIAL_WEIGHT_SHA256[family] and audit.sha(sources[family]["path"]) == OFFICIAL_WEIGHT_SHA256[family], required=True)
    audit.check("oof_report_has_both_families", {f["name"] for f in report["families"]} == set(FAMILIES) and len(report["families"]) == 2, required=True)
    report_families = {item["name"]: item for item in report["families"]}
    family_oof, fold_records, partition, model_lookup = {}, [], None, {}
    for family in FAMILIES:
        family_root = root / "runs" / family
        audit.check(family + ":exact_five_fold_directories", {p.name for p in family_root.iterdir() if p.is_dir() and re.fullmatch(r"fold[0-9]+", p.name)} == {f"fold{i}" for i in range(5)}, required=True)
        coverage, this_partition = [], []
        family_oof[(family, False)] = {}
        family_oof[(family, True)] = {}
        for fold in range(5):
            run = family_root / f"fold{fold}"
            prefix = f"{family}/fold{fold}"
            config = audit.json(run / "config.json")
            validation = audit.json(run / "validation_report.json")
            val_ids, train_ids = config["val_ids"], config["train_ids"]
            audit.check(prefix + ":16_4_disjoint_complement", len(train_ids) == len(set(train_ids)) == 16 and len(val_ids) == len(set(val_ids)) == 4 and not set(train_ids) & set(val_ids) and set(train_ids) | set(val_ids) == set(IDS), required=True)
            audit.check(prefix + ":configuration_identity", config["version"] == "v3.0.0" and config["encoder_name"] == family and config["fold"] == fold and Path(config["data"]).resolve() == data, required=True)
            audit.check(prefix + ":supplied_fine_tuning_only", config["pretrained_weights"] is True and config["external_fine_tuning_data"] is False and config["pretrained_dataset"] == "ImageNet-1K" and config["pretrained_sha256"] == OFFICIAL_WEIGHT_SHA256[family], required=True)
            audit.check(prefix + ":training_complete", validation["completed_steps"] == config["steps"] and validation["validation_ids"] == val_ids, required=True)
            current_source_checks = {name: audit.sha(root / name) == digest for name, digest in config["source_sha256"].items()}
            audit.check(prefix + ":captured_sources_unchanged", all(current_source_checks.values()), current_source_checks)
            coverage.extend(val_ids)
            this_partition.append(sorted(val_ids, key=int))
            checkpoint_path = run / "best.pt"
            checkpoint_hash = audit.sha(checkpoint_path)
            # Only one checkpoint is resident at a time. No model is constructed.
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            try:
                ck_config = state["config"]
                essential_keys = ("version", "encoder_name", "fold", "train_ids", "val_ids", "pretrained_weights", "external_fine_tuning_data", "pretrained_sha256", "steps", "seed", "patch_size", "batch_size", "lr", "encoder_lr", "weight_decay", "ema_decay")
                audit.check(prefix + ":checkpoint_config_matches", all(ck_config[key] == config[key] for key in essential_keys), required=True)
                checkpoint_source_checks = {name: audit.sha(root / name) == digest for name, digest in ck_config["source_sha256"].items()}
                audit.check(prefix + ":checkpoint_captured_sources_unchanged", all(checkpoint_source_checks.values()), checkpoint_source_checks)
                audit.check(prefix + ":checkpoint_metadata_matches_report", state["version"] == "v3.0.0" and state["step"] == validation["best_step"] and state["weight_kind"] == validation["weight_kind"] == state["best_kind"] and state["weight_kind"] in ("raw", "ema"), required=True)
                audit.check(prefix + ":checkpoint_score_matches_report", abs(state["best_score"] - validation["at_0_5"]["mean_dice"]) <= 1e-12)
                audit.check(prefix + ":checkpoint_tensors_cpu_finite", all(t.device.type == "cpu" and (not torch.is_floating_point(t) or torch.isfinite(t).all().item()) for t in state["model"].values()))
                tensor_digest = hashlib.sha256()
                for name, tensor in state["model"].items():
                    tensor_digest.update(name.encode())
                    tensor_digest.update(tensor.detach().contiguous().numpy().tobytes())
                logs = [json.loads(line) for line in (run / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                audit.sha(run / "validation.jsonl")
                winner = max(logs, key=lambda item: item["mean_dice"])
                audit.check(prefix + ":checkpoint_matches_logged_best", winner["step"] == state["step"] and winner["weight_kind"] == state["weight_kind"] and abs(winner["mean_dice"] - state["best_score"]) <= 1e-12)
                model_record = {"family": family, "fold": fold, "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash, "state_dict_sha256": tensor_digest.hexdigest(), "step": state["step"], "weight_kind": state["weight_kind"], "train_ids": train_ids, "val_ids": val_ids}
            finally:
                del state
                gc.collect()
            fold_records.append(model_record)
            model_lookup[(family, fold)] = model_record
            declared_fold = next(item for item in report_families[family]["folds"] if item["fold"] == fold)
            audit.check(prefix + ":oof_input_checkpoint_hash", declared_fold["artifacts"]["best.pt"]["sha256"] == checkpoint_hash)
            audit.check(prefix + ":oof_declared_split", declared_fold["val_ids"] == sorted(val_ids, key=int) and declared_fold["train_ids"] == sorted(train_ids, key=int))
            for tta, filename, metric_key in ((False, "validation_probabilities.npz", "at_0_5"), (True, "validation_tta_probabilities.npz", "tta_at_0_5")):
                probabilities = audit.probabilities(run / filename, val_ids, shapes)
                audit.check(prefix + ":no_duplicate_oof_image:" + str(tta), not set(probabilities) & set(family_oof[(family, tta)]), required=True)
                family_oof[(family, tta)].update(probabilities)
                measured = _native_metrics(probabilities, labels, 0.5)
                audit.check(prefix + ":independent_fold_metric:" + str(tta), _metric_matches(measured, validation[metric_key]))
                audit.check(prefix + ":oof_input_probability_hash:" + str(tta), declared_fold["artifacts"][filename]["sha256"] == audit.sha(run / filename))
        audit.check(family + ":each_image_held_out_exactly_once", len(coverage) == 20 and set(coverage) == set(IDS), required=True)
        if partition is None:
            partition = this_partition
        audit.check(family + ":same_cross_family_fold_partition", this_partition == partition, required=True)
    audit.check("oof_report_partition_matches", report["fold_partition"] == partition, required=True)
    audit.details["checkpoints"] = fold_records
    return family_oof, partition, model_lookup


def _oof_and_selection(audit, report, selection, family_oof, partition, labels, shapes, model_lookup):
    audit.check("both_normal_and_tta_oof_complete", report["tta_complete"] is True and not report["missing_tta_artifacts"], required=True)
    expected_variants = {f"{name}_{suffix}" for name in (*FAMILIES, "equal_ensemble") for suffix in ("no_tta", "tta")}
    audit.check("exact_six_oof_candidates", set(report["variants"]) == expected_variants, required=True)
    audit.check("fixed_common_threshold_grid", report["threshold_grid"] == GRID, required=True)
    independently_measured = {}
    for name, variant in report["variants"].items():
        family_weights = variant["family_weights"]
        expected_weights = {family: 0.5 for family in FAMILIES} if name.startswith("equal_ensemble_") else {name.split("_")[0]: 1.0}
        audit.check(name + ":preset_family_weights", family_weights == expected_weights and variant["family_weights_selected_by_optimization"] is False, required=True)
        tta = name.endswith("_tta") and not name.endswith("_no_tta")
        audit.check(name + ":tta_identity", variant["tta"] is tta, required=True)
        reconstructed = {}
        for image_id in IDS:
            if len(family_weights) == 1:
                family = next(iter(family_weights))
                reconstructed[image_id] = family_oof[(family, tta)][image_id]
            else:
                value = np.zeros(shapes[image_id], dtype=np.float32)
                for family in sorted(family_weights):
                    value += family_oof[(family, tta)][image_id] * family_weights[family]
                reconstructed[image_id] = value
            held_out_fold = next(index for index, ids in enumerate(partition) if image_id in ids)
            provenance = variant["prediction_provenance"][image_id]
            audit.check(name + ":held_out_route:" + image_id, set(provenance) == set(family_weights) and all(item["held_out"] is True and item["fold"] == held_out_fold for item in provenance.values()))
        saved_path = Path(variant["oof_probabilities_path"])
        saved = audit.probabilities(saved_path, IDS, shapes)
        audit.check(name + ":oof_hash", audit.sha(saved_path) == variant["oof_probabilities_sha256"])
        audit.check(name + ":exact_held_out_probability_reconstruction", all(np.array_equal(saved[image_id], reconstructed[image_id]) for image_id in IDS), required=True)
        fixed = _native_metrics(saved, labels, 0.5)
        sweep = [_native_metrics(saved, labels, threshold) for threshold in GRID]
        tuned = max(sweep, key=lambda row: (row["mean_dice"], -abs(row["threshold"] - 0.5)))
        audit.check(name + ":fixed_native_metrics", _metric_matches(fixed, variant["fixed_0_50"]))
        audit.check(name + ":tuned_native_metrics", _metric_matches(tuned, variant["native_tuned"]) and tuned["threshold"] == variant["threshold_tuning"]["native_best_threshold"])
        independently_measured[name] = {"tta": tta, "family_weights": family_weights, "fixed_0_50": fixed, "native_tuned": tuned}
    winner_name = sorted(independently_measured, key=lambda name: (-independently_measured[name]["fixed_0_50"]["mean_dice"], independently_measured[name]["tta"], len(independently_measured[name]["family_weights"]), name))[0]
    winner = independently_measured[winner_name]
    audit.check("independent_variant_selection", selection["selected_variant"] == winner_name and selection["family_weights"] == winner["family_weights"] and selection["tta"] == winner["tta"], required=True)
    audit.check("independent_threshold_selection", selection["threshold"] == winner["native_tuned"]["threshold"] and abs(selection["fixed_0_5_mean_dice"] - winner["fixed_0_50"]["mean_dice"]) <= 1e-12 and abs(selection["native_tuned_mean_dice"] - winner["native_tuned"]["mean_dice"]) <= 1e-12)
    audit.check("selection_matches_oof_report", selection == report["selection"])
    model_keys = [(item["family"], item["fold"]) for item in selection["models"]]
    expected_keys = {(family, fold) for family in winner["family_weights"] for fold in range(5)}
    audit.check("selected_test_models_all_five_per_family", len(model_keys) == len(set(model_keys)) and set(model_keys) == expected_keys, required=True)
    for item in selection["models"]:
        record = model_lookup[(item["family"], item["fold"])]
        audit.check(f"selected_model:{item['family']}/fold{item['fold']}", item["checkpoint_sha256"] == record["checkpoint_sha256"] and Path(item["checkpoint"]).resolve() == Path(record["checkpoint"]).resolve() and item["train_ids"] == record["train_ids"] and item["val_ids"] == record["val_ids"] and abs(item["test_prediction_weight"] - winner["family_weights"][item["family"]] / 5) <= 1e-12, required=True)
    audit.check("test_prediction_weights_sum_to_one", abs(sum(item["test_prediction_weight"] for item in selection["models"]) - 1) <= 1e-12)
    audit.details["selected_oof_recomputed"] = {"selected_variant": winner_name, **winner}
    audit.details["all_candidate_native_scores"] = {name: {"fixed_0_50": value["fixed_0_50"]["mean_dice"], "native_tuned": value["native_tuned"]["mean_dice"], "threshold": value["native_tuned"]["threshold"]} for name, value in independently_measured.items()}


def _test_submission(audit, root, data, oof, submission, selection, test_shapes, input_hashes):
    prediction = audit.json(submission / "prediction_run.json")
    metadata = audit.json(submission / "submission_metadata.json")
    audit.check("test_selection_provenance", prediction["version"] == "v3.0.0" and prediction["selection_sha256"] == audit.sha(oof / "selection.json") and Path(prediction["selection_file"]).resolve() == (oof / "selection.json").resolve() and prediction["selected_variant"] == selection["selected_variant"] and prediction["tta"] == selection["tta"] and prediction["threshold"] == selection["threshold"], required=True)
    audit.check("prediction_model_record_count", len(prediction["models"]) == len(selection["models"]), required=True)
    total = {image_id: np.zeros(test_shapes[image_id], dtype=np.float32) for image_id in IDS}
    cache_records = []
    for item, produced in zip(selection["models"], prediction["models"]):
        tag = f"{item['family']}_fold{item['fold']}"
        audit.check(tag + ":test_record_matches_selection", all(produced[key] == value for key, value in item.items()), required=True)
        probability_path = Path(produced["probabilities"]).resolve()
        expected_path = (submission / "model_probabilities" / f"{tag}.npz").resolve()
        audit.check(tag + ":cache_path", probability_path == expected_path, required=True)
        cache = audit.json(probability_path.with_suffix(".json"))
        actual_sha = audit.sha(probability_path)
        audit.check(tag + ":cache_digest", actual_sha == produced["probabilities_sha256"] == cache["probabilities_sha256"])
        audit.check(tag + ":cache_input_and_model_provenance", cache["checkpoint_sha256"] == item["checkpoint_sha256"] and cache["tta"] == selection["tta"] and cache["ids"] == IDS and cache["test_input_sha256"] == input_hashes and cache["inference_code_sha256"] == audit.sha(root / "inference.py") and cache["model_code_sha256"] == audit.sha(root / "pretrained_model.py"))
        values = audit.probabilities(probability_path, IDS, test_shapes)
        # Use the predictor's documented model order and float32 arithmetic.
        for image_id in IDS:
            total[image_id] += values[image_id] * item["test_prediction_weight"]
        cache_records.append({"model": tag, "weight": item["test_prediction_weight"], "sha256": actual_sha})
        del values
    combined = audit.probabilities(submission / "test_probabilities.npz", IDS, test_shapes)
    exact = all(np.array_equal(total[image_id], combined[image_id]) for image_id in IDS)
    maximum_error = max(float(np.max(np.abs(total[image_id] - combined[image_id]))) for image_id in IDS)
    audit.check("test_weighted_cache_ensemble_exact", exact, {"max_absolute_error": maximum_error}, required=True)
    audit.check("submission_export_configuration", metadata["polarity"] == "vessel-white" and metadata["resize_mode"] == "official" and metadata["threshold"] == selection["threshold"] and metadata["threshold_comparison"] == ">=")
    audit.check("producer_executed_original_converter", metadata["official_script_verified"] is True and metadata["roundtrip_verified"] is True and Path(metadata["official_script"]).resolve() == (data / "segmentation_to_csv.py").resolve())
    audit.sha(data / "segmentation_to_csv.py")
    with (data / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as handle:
        samples = list(csv.DictReader(handle))
    sample_ids = [row["Id"] for row in samples]
    audit.sha(submission / "submission.csv")
    with (submission / "submission.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    audit.check("csv_columns_ids_order_and_count", columns == ["Id", "Predicted"] and len(rows) == 20 and [row["Id"] for row in rows] == sample_ids == IDS, required=True)
    audit.check("exact_20_prediction_png_files", {path.name for path in (submission / "image").iterdir()} == {f"{image_id}.png" for image_id in IDS}, required=True)
    rows_result = []
    for row in rows:
        image_id = row["Id"]
        path = submission / "image" / f"{image_id}.png"
        audit.sha(path)
        with Image.open(path) as image:
            image_format, image_mode = image.format, image.mode
            pixels = np.asarray(image)
        expected_pixels = (combined[image_id] >= selection["threshold"]).astype(np.uint8) * 255
        audit.check("png_matches_selected_probability:" + image_id, image_format == "PNG" and image_mode == "L" and pixels.shape == test_shapes[image_id] and set(np.unique(pixels)) <= {0, 255} and np.array_equal(pixels, expected_pixels), required=True)
        independently_transformed = np.asarray(Image.fromarray(pixels).convert("L").resize((512, 512), Image.Resampling.BILINEAR)) > 0
        decoded = _decode_rle(row["Predicted"])
        audit.check("csv_rle_exact_official_transform:" + image_id, np.array_equal(decoded, independently_transformed), required=True)
        rows_result.append({"id": image_id, "encoded_positive_ratio": float(decoded.mean())})
    # If release copies have already been created, verify them too.
    named_csv = root / "submission_v3.0.0.csv"
    if named_csv.exists():
        audit.check("named_release_csv_exact", audit.sha(named_csv) == audit.sha(submission / "submission.csv"))
    audit.details["test_submission"] = {"selected_models": cache_records, "weighted_ensemble_max_absolute_error": maximum_error, "csv_sha256": audit.sha(submission / "submission.csv"), "rows": rows_result, "original_converter_evidence": "Producer metadata records successful execution. This audit separately reproduced the PIL grayscale/bilinear/>0 transform and RLE pixels, without rerunning the converter."}


def run_audit(root, data, oof=None, submission=None, archive=None):
    root, data = Path(root).resolve(), Path(data).resolve()
    if not (data / "train").is_dir() and (data / "segmentation/train").is_dir():
        data = data / "segmentation"
    oof = Path(oof).resolve() if oof is not None else root / "oof"
    submission = Path(submission).resolve() if submission is not None else root / "submission"
    audit = Audit()
    try:
        mandatory = [root / "previous_versions_snapshot.json", root / "data_audit.json", root / "pretrained_sources.json", oof / "oof_report.json", oof / "selection.json", submission / "prediction_run.json", submission / "submission_metadata.json", submission / "submission.csv", submission / "test_probabilities.npz"]
        for family in FAMILIES:
            for fold in range(5):
                run = root / "runs" / family / f"fold{fold}"
                mandatory.extend(run / name for name in ("config.json", "best.pt", "validation_report.json", "validation.jsonl", "validation_probabilities.npz", "validation_tta_probabilities.npz"))
        missing = [str(path) for path in mandatory if not path.is_file()]
        audit.check("all_final_artifacts_exist", not missing, {"missing": missing}, required=True)
        _snapshot_previous(audit, root / "previous_versions_snapshot.json")
        labels, shapes, test_shapes, input_hashes = _data_provenance(audit, root, data, archive)
        report = audit.json(oof / "oof_report.json")
        selection = audit.json(oof / "selection.json")
        family_oof, partition, models = _folds_and_checkpoints(audit, root, report, data, labels, shapes)
        _oof_and_selection(audit, report, selection, family_oof, partition, labels, shapes, models)
        del family_oof
        gc.collect()
        _test_submission(audit, root, data, oof, submission, selection, test_shapes, input_hashes)
        changed = [str(path) for path, stamp in audit.input_stats.items() if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != stamp]
        audit.check("all_inputs_stable_during_audit", not changed, {"changed": changed})
    except AuditStopped:
        pass
    except Exception as error:
        audit.check("audit_execution_error", False, {"type": type(error).__name__, "message": str(error)})
    failures = [row for row in audit.checks if not row["passed"]]
    return {
        "schema_version": 1,
        "version": "v3.0.0",
        "audit_time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "failed" if failures else "passed",
        "check_count": len(audit.checks),
        "failure_count": len(failures),
        "failures": failures,
        "cpu_only": True,
        "models_executed": False,
        "checkpoint_loading": "One trusted local best.pt at a time, map_location=cpu; immediately released after inspection",
        "root": str(root),
        "checks": audit.checks,
        "details": audit.details,
        "artifact_sha256": audit.hashes,
        "limitations": [
            "Held-out routing, recorded splits, weights, and saved probability arithmetic are verified; model inference and training were not rerun.",
            "Checkpoint provenance relies on checkpoint metadata, captured source hashes, and matching validation logs; historical live weights at every training step were not independently retained.",
            "The only fine-tuning data claim is supported by source/config declarations and byte equality to supplied data, not process-level file/network tracing. Official ImageNet-1K initialization is external pretraining and is explicitly recorded.",
            "OOF checkpoint, model/TTA variant, and threshold selection used these labels, so this is not an independent test estimate or a Kaggle score.",
            "Hidden ground-truth polarity and resize/binarization remain unconfirmed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof", type=Path, help="Defaults to ROOT/oof")
    parser.add_argument("--submission", type=Path, help="Defaults to ROOT/submission")
    parser.add_argument("--archive", type=Path, help="Original competition ZIP; defaults to the data-audit source path")
    parser.add_argument("--output", type=Path, help="Defaults to ROOT/independent_audit.json")
    args = parser.parse_args()
    output = args.output.resolve() if args.output else args.root.resolve() / "independent_audit.json"
    if output.suffix.lower() != ".json" or output.name in {"selection.json", "oof_report.json", "previous_versions_snapshot.json", "version_manifest.json", "data_audit.json", "config.json", "submission_metadata.json", "prediction_run.json"}:
        raise ValueError("Choose a separate JSON audit output path")
    root = args.root.resolve()
    if output != root and root not in output.parents:
        raise ValueError("Audit output must stay inside the v3 root")
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old.get("audit_script") != "audit_final.py":
            raise FileExistsError("Refusing to overwrite a JSON file not owned by this audit script")
    result = run_audit(root, args.data, args.oof, args.submission, args.archive)
    result["audit_script"] = "audit_final.py"
    result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": result["check_count"], "failure_count": result["failure_count"], "failures": result["failures"], "audit_file": str(output)}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
