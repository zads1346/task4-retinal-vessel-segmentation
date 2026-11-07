"""Independent CPU-only final audit of v4 scratch five-fold thin-vessel training.
No model forward/inference/training is run. A frozen v2 U-Net is initialized on
CPU only to verify the random initial-state digest for each fold. Checkpoints
are loaded sequentially and released. Only the audit JSON is written.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
IDS = [str(i) for i in range(1,21)]
GRID = np.round(np.arange(0.20,0.801,0.02),2).tolist()
DATA_SHA256 = "6f6e36243092f4cf55c1be4ae7c5227e3c3686501e25487ada7ef03fa9b7634a"
MODES = {"none":("validation_probabilities.npz","at_0_5"), "flip4":("validation_tta_probabilities.npz","tta_at_0_5"), "d4":("validation_d4_probabilities.npz","d4_at_0_5")}

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



def _state_digest(state):
    result = hashlib.sha256()
    for name, tensor in state.items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _folds(audit, root, data, report, labels, shapes):
    import torch
    torch.set_num_threads(1)
    snapshot = audit.json(root / "previous_versions_snapshot.json")
    v2 = next(v for v in snapshot["versions"] if v["version"] == "v2.0.0")
    frozen_model = Path(v2["root"]) / "model.py"
    audit.check("v4_model_identical_to_frozen_v2", audit.sha(root / "model.py") == audit.sha(frozen_model), required=True)
    spec = importlib.util.spec_from_file_location("audit_frozen_v2_model", frozen_model)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit.check("exact_five_fold_directories", {p.name for p in (root/"runs").iterdir() if p.is_dir() and re.fullmatch(r"fold[0-9]+",p.name)} == {f"fold{i}" for i in range(5)}, required=True)
    routes, seen, records = [], set(), []
    oof = {mode:{} for mode in MODES}
    audit.check("oof_declares_single_scratch_family", report["version"] == "v4.0.0" and report["family"] == "unet_v4" and report["pretrained_weights"] is False and report["external_training_data"] is False, required=True)
    for fold in range(5):
        run = root/"runs"/f"fold{fold}"
        config = audit.json(run/"config.json")
        validation = audit.json(run/"validation_report.json")
        prefix = f"fold{fold}"
        train, valid = config["train_ids"], config["val_ids"]
        audit.check(prefix+":16_4_disjoint_split", len(train)==len(set(train))==16 and len(valid)==len(set(valid))==4 and not(set(train)&set(valid)) and set(train)|set(valid)==set(IDS), required=True)
        audit.check(prefix+":held_out_once", not(seen&set(valid)), required=True)
        seen.update(valid);routes.append(sorted(valid,key=int))
        audit.check(prefix+":v4_configuration", config["version"]==validation["version"]=="v4.0.0" and config["encoder_name"]==config["model_family"]=="unet_v4" and config["fold"]==fold and config["base_channels"]==24 and config["architecture"]=="v2_UNet_base24_unchanged" and Path(config["data"]).resolve()==data, required=True)
        audit.check(prefix+":scratch_supplied_data_only", config["pretrained_weights"] is False and config["external_training_data"] is False and config["external_fine_tuning_data"] is False and config["initialization"]=="random_from_scratch", required=True)
        audit.check(prefix+":thin_loss_weights", abs(config["thin_weight"]-0.5)<1e-12 and abs(config["skeleton_weight"]-0.03)<1e-12, required=True)
        audit.check(prefix+":completed_6000_steps", config["steps"]==validation["completed_steps"]==6000 and validation["validation_ids"]==valid, required=True)
        audit.check(prefix+":resume_stays_in_same_fold", not config.get("resume") or Path(config["resume"]).resolve().parent==run, required=True)
        sources=config["source_sha256"]
        expected_sources={"train.py","model.py","data.py","thin_data.py","thin_loss.py","ema.py","inference.py","submission.py"}
        audit.check(prefix+":source_file_set",set(sources)==expected_sources,required=True)
        source_matches={name:audit.sha(root/name)==digest for name,digest in sources.items()}
        audit.check(prefix+":captured_sources_unchanged",all(source_matches.values()),source_matches)
        torch.manual_seed(config["seed"]+fold)
        fresh=module.UNet(24).cpu()
        initial_digest=_state_digest(fresh.state_dict())
        del fresh;gc.collect()
        audit.check(prefix+":independent_scratch_initial_digest",initial_digest==config["initial_weights_sha256"],required=True)
        checkpoint_hash=audit.sha(run/"best.pt")
        state=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
        try:
            keys=("version","encoder_name","fold","base_channels","train_ids","val_ids","pretrained_weights","external_training_data","external_fine_tuning_data","initialization","initial_weights_sha256","thin_weight","skeleton_weight","source_sha256","seed","steps","patch_size","batch_size","lr","weight_decay","ema_decay")
            audit.check(prefix+":checkpoint_config_provenance",all(state["config"][k]==config[k] for k in keys),required=True)
            audit.check(prefix+":checkpoint_selected_kind_step",state["version"]=="v4.0.0" and state["step"]==validation["best_step"] and state["weight_kind"]==state["best_kind"]==validation["weight_kind"] and state["weight_kind"] in ("raw","ema"),required=True)
            audit.check(prefix+":best_score",abs(state["best_score"]-validation["at_0_5"]["mean_dice"])<1e-12)
            audit.check(prefix+":cpu_finite_tensors",all(t.device.type=="cpu" and (not torch.is_floating_point(t) or torch.isfinite(t).all().item()) for t in state["model"].values()))
            audit.sha(run/"validation.jsonl")
            logs=[json.loads(line) for line in (run/"validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            winner=max(logs,key=lambda row:row["mean_dice"])
            audit.check(prefix+":logged_winner",winner["step"]==state["step"] and winner["weight_kind"]==state["weight_kind"] and abs(winner["mean_dice"]-state["best_score"])<1e-12)
            record={"family":"unet_v4","fold":fold,"checkpoint":str(run/"best.pt"),"checkpoint_sha256":checkpoint_hash,"state_dict_sha256":_state_digest(state["model"]),"initial_weights_sha256_recomputed":initial_digest,"step":state["step"],"weight_kind":state["weight_kind"],"train_ids":train,"val_ids":valid}
        finally:
            del state;gc.collect()
        declared=next(item for item in report["folds"] if item["fold"]==fold)
        audit.check(prefix+":oof_checkpoint_digest",declared["artifacts"]["best.pt"]["sha256"]==checkpoint_hash)
        audit.check(prefix+":oof_recorded_split",declared["val_ids"]==sorted(valid,key=int) and declared["train_ids"]==sorted(train,key=int))
        for mode,(filename,metric_key) in MODES.items():
            path=run/filename
            values=audit.probabilities(path,valid,shapes)
            audit.check(prefix+":"+mode+":oof_input_digest",audit.sha(path)==declared["artifacts"][filename]["sha256"])
            audit.check(prefix+":"+mode+":fixed_native_metric",_metric_matches(_native_metrics(values,labels,0.5),validation[metric_key]))
            oof[mode].update(values)
        records.append(record)
    audit.check("all_twenty_held_out_exactly_once",seen==set(IDS),required=True)
    audit.check("oof_partition_matches",report["fold_partition"]==routes,required=True)
    audit.details["checkpoints"]=records
    return oof,routes,records


def _select(audit, report, selection, oof, routes, records, labels, shapes):
    audit.check("exact_three_oof_modes",set(report["variants"])==set(MODES),required=True)
    audit.check("threshold_grid",report["threshold_grid"]==GRID,required=True)
    measured={}
    for mode in MODES:
        variant=report["variants"][mode]
        path=Path(variant["oof_probabilities_path"])
        saved=audit.probabilities(path,IDS,shapes)
        audit.check(mode+":held_out_npz_exact",all(np.array_equal(saved[i],oof[mode][i]) for i in IDS),required=True)
        audit.check(mode+":oof_hash",audit.sha(path)==variant["oof_probabilities_sha256"])
        audit.check(mode+":tta_identity",variant["tta_mode"]==mode and variant["tta"] is (mode!="none") and variant["family_weights"]=={"unet_v4":1.0},required=True)
        for image_id in IDS:
            fold=next(f for f,ids in enumerate(routes) if image_id in ids)
            declared=variant["prediction_provenance"][image_id]
            audit.check(mode+":held_out_route:"+image_id,set(declared)=={"unet_v4"} and declared["unet_v4"]=={"fold":fold,"held_out":True})
        fixed=_native_metrics(saved,labels,0.5)
        sweep=[_native_metrics(saved,labels,t) for t in GRID]
        tuned=max(sweep,key=lambda row:(row["mean_dice"],-abs(row["threshold"]-0.5)))
        audit.check(mode+":fixed_native_metrics",_metric_matches(fixed,variant["fixed_0_50"]))
        audit.check(mode+":tuned_metrics_and_threshold",_metric_matches(tuned,variant["native_tuned"]) and tuned["threshold"]==variant["threshold_tuning"]["native_best_threshold"])
        measured[mode]={"fixed_0_50":fixed,"native_tuned":tuned}
    winner=max(MODES,key=lambda mode:(measured[mode]["fixed_0_50"]["mean_dice"],-list(MODES).index(mode)))
    audit.check("selected_mode_recomputed",selection["version"]=="v4.0.0" and selection["selected_variant"]==selection["tta_mode"]==winner and selection["tta"] is (winner!="none"),required=True)
    audit.check("selected_metrics_and_threshold",selection["threshold"]==measured[winner]["native_tuned"]["threshold"] and abs(selection["fixed_0_5_mean_dice"]-measured[winner]["fixed_0_50"]["mean_dice"])<1e-12 and abs(selection["native_tuned_mean_dice"]-measured[winner]["native_tuned"]["mean_dice"])<1e-12)
    audit.check("selection_matches_report",selection==report["selection"])
    models=selection["models"]
    audit.check("five_models_and_single_family",len(models)==5 and sorted(m["fold"] for m in models)==list(range(5)) and selection["family_weights"]=={"unet_v4":1.0},required=True)
    for model in models:
        record=records[model["fold"]]
        audit.check("selected_model_fold"+str(model["fold"]),model["family"]=="unet_v4" and model["checkpoint_sha256"]==record["checkpoint_sha256"] and Path(model["checkpoint"]).resolve()==Path(record["checkpoint"]).resolve() and model["train_ids"]==record["train_ids"] and model["val_ids"]==record["val_ids"] and abs(model["test_prediction_weight"]-0.2)<1e-12,required=True)
    audit.check("test_weights_sum_to_one",abs(sum(m["test_prediction_weight"] for m in models)-1)<1e-12)
    audit.details["selected_oof_recomputed"]={"tta_mode":winner,**measured[winner]}
    audit.details["all_candidate_native_scores"]={mode:{"fixed_0_50":m["fixed_0_50"]["mean_dice"],"native_tuned":m["native_tuned"]["mean_dice"],"threshold":m["native_tuned"]["threshold"]} for mode,m in measured.items()}



def _test_submission(audit, root, data, oof, submission, selection, test_shapes, input_hashes):
    prediction = audit.json(submission / "prediction_run.json")
    metadata = audit.json(submission / "submission_metadata.json")
    audit.check("prediction_tta_mode", prediction["tta_mode"] == selection["tta_mode"], required=True)
    audit.check("test_selection_provenance", prediction["version"] == "v4.0.0" and prediction["selection_sha256"] == audit.sha(oof / "selection.json") and Path(prediction["selection_file"]).resolve() == (oof / "selection.json").resolve() and prediction["selected_variant"] == selection["selected_variant"] and prediction["tta"] == selection["tta"] and prediction["threshold"] == selection["threshold"], required=True)
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
        audit.check(tag + ":cache_tta_mode", cache["tta_mode"] == selection["tta_mode"], required=True)
        actual_sha = audit.sha(probability_path)
        audit.check(tag + ":cache_digest", actual_sha == produced["probabilities_sha256"] == cache["probabilities_sha256"])
        audit.check(tag + ":cache_input_and_model_provenance", cache["checkpoint_sha256"] == item["checkpoint_sha256"] and cache["tta"] == selection["tta"] and cache["ids"] == IDS and cache["test_input_sha256"] == input_hashes and cache["inference_code_sha256"] == audit.sha(root / "inference.py") and cache["model_code_sha256"] == audit.sha(root / "model.py"))
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
    named_csv = root / "submission_v4.0.0.csv"
    if named_csv.exists():
        audit.check("named_release_csv_exact", audit.sha(named_csv) == audit.sha(submission / "submission.csv"))
    audit.details["test_submission"] = {"selected_models": cache_records, "weighted_ensemble_max_absolute_error": maximum_error, "csv_sha256": audit.sha(submission / "submission.csv"), "rows": rows_result, "original_converter_evidence": "Producer metadata records successful execution. This audit separately reproduced the PIL grayscale/bilinear/>0 transform and RLE pixels, without rerunning the converter."}



def run_audit(root,data,oof=None,submission=None,archive=None):
    root,data=Path(root).resolve(),Path(data).resolve()
    if not(data/"train").is_dir() and (data/"segmentation/train").is_dir():data=data/"segmentation"
    oof=Path(oof).resolve() if oof else root/"oof"
    submission=Path(submission).resolve() if submission else root/"submission"
    audit=Audit()
    try:
        required=[root/"previous_versions_snapshot.json",root/"data_audit.json",oof/"oof_report.json",oof/"selection.json",submission/"prediction_run.json",submission/"submission_metadata.json",submission/"submission.csv",submission/"test_probabilities.npz"]
        for fold in range(5):
            run=root/"runs"/f"fold{fold}"
            required.extend(run/name for name in ["config.json","best.pt","validation_report.json","validation.jsonl"]+[p[0] for p in MODES.values()])
        missing=[str(p) for p in required if not p.is_file()]
        audit.check("all_final_artifacts_exist",not missing,{"missing":missing},required=True)
        _snapshot_previous(audit,root/"previous_versions_snapshot.json")
        labels,shapes,test_shapes,input_hashes=_data_provenance(audit,root,data,archive)
        report=audit.json(oof/"oof_report.json");selection=audit.json(oof/"selection.json")
        values,partition,records=_folds(audit,root,data,report,labels,shapes)
        _select(audit,report,selection,values,partition,records,labels,shapes)
        del values;gc.collect()
        _test_submission(audit,root,data,oof,submission,selection,test_shapes,input_hashes)
        changed=[str(p) for p,stamp in audit.input_stats.items() if not p.is_file() or (p.stat().st_size,p.stat().st_mtime_ns)!=stamp]
        audit.check("all_inputs_stable_during_audit",not changed,{"changed":changed})
    except AuditStopped:pass
    except Exception as error:audit.check("audit_execution_error",False,{"type":type(error).__name__,"message":str(error)})
    failed=[r for r in audit.checks if not r["passed"]]
    return {"schema_version":1,"version":"v4.0.0","audit_time":datetime.now(timezone(timedelta(hours=8))).isoformat(),"status":"failed" if failed else "passed","check_count":len(audit.checks),"failure_count":len(failed),"failures":failed,"cpu_only":True,"models_executed":False,"scratch_initialization_reconstructed_on_cpu":True,"checkpoint_loading":"One local best.pt at a time on CPU; no model forward, inference or training","root":str(root),"checks":audit.checks,"details":audit.details,"artifact_sha256":audit.hashes,"limitations":["Held-out routing and saved probability arithmetic are verified; neural-network inference/training were not rerun.","Scratch digests are reconstructed from frozen v2 model code. Checkpoint provenance also relies on captured source hashes, configs and training logs; no process-level I/O tracing was performed.","OOF checkpoint/TTA/threshold selection reused validation labels; this is not independent test or Kaggle performance.","Hidden scoring polarity and resize/binarization remain unconfirmed."]}



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
        raise ValueError("Audit output must stay inside the v4 root")
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
