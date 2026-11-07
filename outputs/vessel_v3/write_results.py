"""Write a source-checked Chinese result summary after all v3 outputs exist.

Uses only the Python standard library. It does not load models, probability
arrays, images, or test labels, and never starts GPU work. Default usage:
    python write_results.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


IDS = [str(index) for index in range(1, 21)]
FAMILIES = ("resnet18", "resnet34")
VARIANTS = tuple(
    f"{family}_{suffix}"
    for suffix in ("no_tta", "tta")
    for family in (*FAMILIES, "equal_ensemble")
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def _close(left: Any, right: Any, label: str) -> None:
    if not math.isclose(_number(left, label), _number(right, label), rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"Inconsistent {label}: {left!r} != {right!r}")


def _ids(values: Any, label: str, count: int | None = None) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"Expected ID list: {label}")
    result = [str(value) for value in values]
    if len(result) != len(set(result)) or not set(result).issubset(IDS):
        raise ValueError(f"Unexpected or repeated IDs: {label}")
    if count is not None and len(result) != count:
        raise ValueError(f"Expected {count} IDs: {label}")
    return sorted(result, key=int)


def _metric(report: dict[str, Any], expected_ids: list[str], label: str) -> dict[str, dict]:
    rows = report["per_image"]
    if _ids([row["id"] for row in rows], label) != expected_ids:
        raise ValueError(f"Metric ID coverage mismatch: {label}")
    by_id = {str(row["id"]): row for row in rows}
    scores = []
    for image_id in expected_ids:
        row = by_id[image_id]
        score = _number(row["dice"], label)
        if not 0 <= score <= 1:
            raise ValueError(f"Dice outside [0,1]: {label}")
        tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
        if min(tp, fp, fn) < 0:
            raise ValueError(f"Negative pixel count: {label}")
        denominator = 2 * tp + fp + fn
        _close(score, 2 * tp / denominator if denominator else 1.0, label + " vessel Dice")
        scores.append(score)
    _close(report["mean_dice"], sum(scores) / len(scores), label + " image mean")
    return by_id


def _label(name: str) -> str:
    if name.startswith("equal_ensemble"):
        return "R18/R34 等权"
    return "ResNet18" if name.startswith("resnet18") else "ResNet34"


def _optional_v2_comparison(v2_dir: Path, oof: dict[str, Any]) -> list[str]:
    """Only compare native, non-TTA, fixed-0.5 metrics on identical fold-0 IDs."""
    try:
        report = _read(v2_dir / "runs/fold0/validation_report.json")
        config = _read(v2_dir / "runs/fold0/config.json")
        if report.get("version") != "v2.0.0" or bool(report.get("tta", False)):
            return []
        if Path(config["data"]).resolve() != Path(oof["data_root"]).resolve():
            return []
        target = str(config.get("target", ""))
        if "vessel=1" not in target or "> 127" not in target:
            return []
        image_ids = _ids(report["validation_ids"], "v2 fold0", 4)
        if image_ids != _ids(oof["fold_partition"][0], "v3 fold0", 4):
            return []
        _close(report["at_0_5"]["threshold"], 0.5, "v2 comparison threshold")
        _metric(report["at_0_5"], image_ids, "v2 non-TTA native")
        baseline = float(report["at_0_5"]["mean_dice"])
        lines = [
            "## 与 v2 的同图描述性比较", "",
            "下面只比较两版共同的 fold 0 图像（ID " + "、".join(image_ids)
            + "），均使用原生尺寸、非 TTA、固定阈值 0.50 的血管逐图平均 Dice。"
            "v2 数值来自其非 TTA 训练报告，不是 v2 后续选定 TTA 的成绩。", "",
            "| 方案 | 同 4 图 Dice | 相对 v2 的差值 |",
            "|---|---:|---:|",
            f"| v2 非 TTA | {baseline:.6f} | — |",
        ]
        for name in ("resnet18_no_tta", "resnet34_no_tta", "equal_ensemble_no_tta"):
            metric = oof["variants"][name]["fixed_0_50"]
            _close(metric["threshold"], 0.5, "v3 comparison threshold")
            rows = _metric(metric, IDS, "v3 comparison native")
            score = sum(float(rows[image_id]["dice"]) for image_id in image_ids) / len(image_ids)
            lines.append(f"| v3 {_label(name)}，非 TTA | {score:.6f} | {score - baseline:+.6f} |")
        lines += ["", "这些图已经多次用于配置与检查点选择；差值仅描述该子集，不能证明泛化提升。"
                  "没有把 v2 的 4 图均值与 v3 的 20 图 OOF 均值直接相减。", ""]
        return lines
    except (OSError, ValueError, KeyError, TypeError):
        # An optional comparison is omitted when exact compatibility is unclear.
        return []


def write_results(root: str | Path, output: str | Path | None = None, v2_dir: str | Path | None = None) -> Path:
    root = Path(root).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve() if output is not None else root / "RESULTS.md"
    if output_path.suffix.lower() != ".md":
        raise ValueError("Output must be a Markdown file")
    progress_path = root / "runs/cv_progress.json"
    report_path = root / "oof/oof_report.json"
    selection_path = root / "oof/selection.json"
    prediction_path = root / "submission/prediction_run.json"
    metadata_path = root / "submission/submission_metadata.json"
    csv_path = root / "submission/submission.csv"
    progress = _read(progress_path)
    if progress.get("status") != "completed":
        raise ValueError("CV driver is not completed; do not create a final results document yet")
    if progress.get("version") != "v3.0.0" or set(progress["families"]) != set(FAMILIES) or progress["folds"] != 5:
        raise ValueError("Unexpected v3 cross-validation design")
    completed = progress["completed"]
    expected_runs = {(family, fold) for family in FAMILIES for fold in range(5)}
    if len(completed) != 10 or {(item["encoder"], item["fold"]) for item in completed} != expected_runs:
        raise ValueError("Progress does not contain exactly ten distinct completed models")
    completed_by_run = {(item["encoder"], item["fold"]): item for item in completed}

    fold_reports = {}
    for family, fold in sorted(expected_runs):
        path = root / "runs" / family / f"fold{fold}" / "validation_report.json"
        report = _read(path)
        if report.get("version") != "v3.0.0" or report["fold"] != fold:
            raise ValueError(f"Unexpected fold/version: {path}")
        if report["completed_steps"] != progress["steps_per_fold"]:
            raise ValueError(f"Completed step count differs from CV driver: {path}")
        if not 0 < report["best_step"] <= report["completed_steps"]:
            raise ValueError(f"Invalid best step: {path}")
        ids = _ids(report["validation_ids"], str(path), 4)
        _close(report["at_0_5"]["threshold"], 0.5, "fold threshold")
        _close(report["tta_at_0_5"]["threshold"], 0.5, "fold TTA threshold")
        _metric(report["at_0_5"], ids, str(path) + " native")
        _metric(report["tta_at_0_5"], ids, str(path) + " TTA native")
        _close(report["at_0_5"]["mean_dice"], completed_by_run[(family, fold)]["native_dice_at_0_5"], "driver/fold Dice")
        fold_reports[(family, fold)] = report
    partitions = {}
    for family in FAMILIES:
        partitions[family] = [_ids(fold_reports[(family, fold)]["validation_ids"], family, 4) for fold in range(5)]
        counts = Counter(image_id for part in partitions[family] for image_id in part)
        if counts != Counter({image_id: 1 for image_id in IDS}):
            raise ValueError(f"Incomplete or repeated held-out image coverage: {family}")
    if partitions["resnet18"] != partitions["resnet34"]:
        raise ValueError("Encoder families use different fold partitions")

    oof, selection = _read(report_path), _read(selection_path)
    prediction, metadata = _read(prediction_path), _read(metadata_path)
    if oof.get("version") != "v3.0.0" or selection.get("version") != "v3.0.0" or prediction.get("version") != "v3.0.0":
        raise ValueError("Final artifact version mismatch")
    if _ids(oof["ids"], "OOF IDs", 20) != IDS or set(oof["variants"]) != set(VARIANTS) or oof["tta_complete"] is not True:
        raise ValueError("Final OOF report must contain all 20 images and all six candidates")
    if oof["pretrained_weights"] is not True or oof["external_fine_tuning_data"] is not False:
        raise ValueError("Unexpected pretraining/fine-tuning data policy")
    if [_ids(part, "OOF partition", 4) for part in oof["fold_partition"]] != partitions["resnet18"]:
        raise ValueError("OOF partition differs from the completed fold reports")
    for name, variant in oof["variants"].items():
        _close(variant["fixed_0_50"]["threshold"], 0.5, name + " threshold")
        _metric(variant["fixed_0_50"], IDS, name + " fixed")
        _metric(variant["native_tuned"], IDS, name + " tuned")
        _close(variant["native_tuned"]["threshold"], variant["threshold_tuning"]["native_best_threshold"], name + " tuned threshold")
    if oof["selection"] != selection:
        raise ValueError("Standalone selection differs from the OOF report")
    name = selection["selected_variant"]
    selected = oof["variants"][name]
    _close(selection["threshold"], selected["native_tuned"]["threshold"], "selected threshold")
    if selected["tta"] != selection["tta"] or selected["family_weights"] != selection["family_weights"]:
        raise ValueError("Selected variant attributes differ")
    if prediction["selected_variant"] != name or prediction["tta"] != selection["tta"]:
        raise ValueError("Test prediction did not use the selected variant")
    _close(prediction["threshold"], selection["threshold"], "test threshold")
    _close(metadata["threshold"], selection["threshold"], "submission threshold")
    if prediction["selection_sha256"] != _sha(selection_path):
        raise ValueError("Prediction selection hash differs from the saved selection")
    if prediction.get("uploaded_to_kaggle") is not False:
        raise ValueError("Expected explicit no-upload provenance")
    if prediction["submission_checks"] != metadata:
        raise ValueError("Submission metadata differs from the prediction report")
    if metadata["official_script_verified"] is not True or metadata["roundtrip_verified"] is not True:
        raise ValueError("Official conversion or RLE round-trip is not verified")
    if metadata["polarity"] != "vessel-white" or metadata["resize_mode"] != "official":
        raise ValueError("Unexpected submission polarity/resize pipeline")
    if _ids(metadata["sample_ids"], "submission IDs", 20) != IDS or set(metadata["images"]) != set(IDS):
        raise ValueError("Submission metadata lacks exact 20-image coverage")
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise FileNotFoundError("Final submission.csv is missing or empty")

    models = selection["models"]
    expected_model_count = 5 * len(selection["family_weights"])
    if len(models) != expected_model_count or expected_model_count not in (5, 10):
        raise ValueError("Selection must use all five folds of one or both families")
    _close(sum(item["test_prediction_weight"] for item in models), 1, "test weight sum")
    prediction_models = {(item["family"], item["fold"]): item for item in prediction["models"]}
    if len(prediction_models) != len(prediction["models"]) or set(prediction_models) != {(item["family"], item["fold"]) for item in models}:
        raise ValueError("Actual test model set differs from the selection")
    for family, family_weight in selection["family_weights"].items():
        members = [item for item in models if item["family"] == family]
        if sorted(item["fold"] for item in members) != list(range(5)):
            raise ValueError("A selected family is missing a fold")
        for item in members:
            _close(item["test_prediction_weight"], family_weight / 5, "fold weight")
            actual = prediction_models[(family, item["fold"])]
            if actual["checkpoint_sha256"] != item["checkpoint_sha256"] or actual["checkpoint"] != item["checkpoint"]:
                raise ValueError("Actual test checkpoint differs from selection")
            _close(actual["test_prediction_weight"], item["test_prediction_weight"], "actual test weight")

    elapsed = _number(progress["elapsed_seconds"], "CV elapsed time")
    if elapsed < 0:
        raise ValueError("Negative runtime")
    selected_rows = _metric(selected["native_tuned"], IDS, "selected per-image")
    fixed_rows = _metric(selected["fixed_0_50"], IDS, "selected fixed per-image")
    held_out_fold = {image_id: fold for fold, ids in enumerate(partitions["resnet18"]) for image_id in ids}
    tta_label = "启用 4 视角翻转 TTA" if selection["tta"] else "不启用 TTA"
    tuned_score = selected["native_tuned"]["mean_dice"]
    fixed_score = selected["fixed_0_50"]["mean_dice"]
    lines = [
        "# 眼底血管分割 v3.0.0 实际结果", "",
        f"已完成 **10/10 个模型、两个编码器族各 5 折**，每模型 {progress['steps_per_fold']:,} 步，"
        f"共 {sum(report['completed_steps'] for report in fold_reports.values()):,} 个计划训练步骤。"
        f"20 张图均具有每模型族一次且仅一次的折外预测。", "",
        f"最终选定 **{name}**，{tta_label}，统一原生阈值 **{selection['threshold']:.2f}**。"
        f"其 20 图 OOF 原生平均 Dice：固定阈值 0.50 为 **{fixed_score:.6f}**，"
        f"调参后为 **{tuned_score:.6f}**。测试集使用 **{len(models)} 个模型**的概率加权平均。", "",
        "这些是经过检查点、候选与阈值选择后的交叉验证结果，具有选择偏差；"
        "不是独立测试成绩、不是嵌套交叉验证估计，也不是 Kaggle 分数。", "",
        "## 六个候选的完整 OOF 比较", "",
        "所有候选覆盖相同 20 张图。先以固定阈值 0.50 的平均 Dice 选择模型/TTA，"
        "再从 0.20–0.80、步长 0.02 的共同网格中选择统一原生阈值。双族融合权重预先固定为各 0.5。", "",
        "| 候选 | TTA | 固定 0.50 Dice | 原生最优阈值 | 调参后 Dice |",
        "|---|---|---:|---:|---:|",
    ]
    for variant_name in VARIANTS:
        variant = oof["variants"][variant_name]
        display = _label(variant_name) + ("（选定）" if variant_name == name else "")
        lines.append(f"| {display} | {'是' if variant['tta'] else '否'} | {variant['fixed_0_50']['mean_dice']:.6f} | "
                     f"{variant['native_tuned']['threshold']:.2f} | {variant['native_tuned']['mean_dice']:.6f} |")
    lines += ["", "## 选定方案的逐折与逐图结果", "",
              "下面的每折调参后分数均使用同一个全局阈值，不为每折单独挑选阈值。", "",
              "| 折 | 验证 ID | 固定 0.50 Dice | 全局阈值下 Dice |",
              "|---:|---|---:|---:|"]
    for fold, ids in enumerate(partitions["resnet18"]):
        fixed_mean = sum(float(fixed_rows[image_id]["dice"]) for image_id in ids) / 4
        tuned_mean = sum(float(selected_rows[image_id]["dice"]) for image_id in ids) / 4
        lines.append(f"| {fold} | {', '.join(ids)} | {fixed_mean:.6f} | {tuned_mean:.6f} |")
    lines += ["", "| 图像 ID | 持出折 | 每个选定模型族的 OOF 次数 | 全局阈值 Dice | Precision | Recall |",
              "|---:|---:|---:|---:|---:|---:|"]
    for image_id in IDS:
        row = selected_rows[image_id]
        lines.append(f"| {image_id} | {held_out_fold[image_id]} | 1 | {row['dice']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} |")
    lines += ["", f"选定阈值下 micro Dice 为 **{selected['native_tuned']['micro_dice']:.6f}**。"
              "OOF 每图只使用未训练过它的对应折；双族融合也只平均各自对应持出折的概率。"
              "测试阶段才使用选定模型族的全部 5 折。", "",
              "## 最终测试集成与提交", "",
              f"推理配置：`{name}`；{tta_label}；阈值 `{selection['threshold']:.2f}`。"
              "先加权平均概率，再阈值化一次。", "",
              "| 模型族 | 折 | 最佳步骤 | 权重类型 | 测试概率权重 |",
              "|---|---:|---:|---|---:|"]
    for model in sorted(models, key=lambda item: (item["family"], item["fold"])):
        report = fold_reports[(model["family"], model["fold"])]
        lines.append(f"| {model['family']} | {model['fold']} | {report['best_step']} | {report['weight_kind']} | {model['test_prediction_weight']:.3f} |")
    lines += ["", f"总权重为 **{sum(item['test_prediction_weight'] for item in models):.6f}**。"
              "最终 CSV 为 `submission/submission.csv`，20 个图像 ID；原始官方脚本验证和 RLE 像素往返验证均通过。"
              "没有上传 Kaggle。", "", f"CSV SHA-256：`{_sha(csv_path)}`。", "",
              "导出采用白色血管、原生二值图经官方 512×512 双线性缩放后 `>0` 编码。"
              "任务文字要求黑色血管与实际标签/脚本仍有语义冲突，隐藏标签极性和预处理未确认。"
              f"选定方案在该白色血管转换假设下的代理平均 Dice 为 "
              f"**{selected['native_tuned']['official_white_vessel_proxy_mean_dice']:.6f}**；"
              "它不能替代已确认的隐藏评分。", "",
              "## 完成统计、数据与边界", "",
              f"CV 驱动记录的总耗时为 **{elapsed:.1f} 秒（{elapsed / 60:.1f} 分钟）**，"
              f"开始时间记录为 `{progress.get('started_at', '未记录')}`。"
              "该时间来自本次驱动调用，覆盖其训练/验证与 OOF 聚合，不含后续测试集预测及归档；"
              "若经历中断或复用了已有折，不能把它解释为所有历史训练的累计用时。", "",
              "本次眼底分割监督微调只使用提供的 20 张图像及标签，每折 16 张训练、4 张验证。"
              "编码器初始化来自公开的 ImageNet-1K 预训练权重；这不等于模型从未使用过外部预训练数据。"
              "没有新增图像数据集参与本次监督微调。", "",
              "v1/v2 保留在原目录。不能直接用本版 20 图 OOF 均值与旧版 4 图验证均值宣称提升。"
              "OOF 中仍含检查点、模型/TTA 和阈值选择；报告的图像重采样区间也是条件性的描述区间，"
              "未消除选择偏差、折间训练集重叠或未来数据分布变化。", ""]
    lines += ["各模型的实际完成记录如下，两个 Dice 列均使用固定阈值 0.50：", "",
              "| 模型族 | 折 | 完成步数 | 最佳步骤 | raw/EMA | 非 TTA Dice | TTA Dice |",
              "|---|---:|---:|---:|---|---:|---:|"]
    for (family, fold), report in sorted(fold_reports.items()):
        lines.append(f"| {family} | {fold} | {report['completed_steps']} | {report['best_step']} | "
                     f"{report['weight_kind']} | {report['at_0_5']['mean_dice']:.6f} | {report['tta_at_0_5']['mean_dice']:.6f} |")
    lines.append("")
    baseline_dir = Path(v2_dir).expanduser().resolve() if v2_dir is not None else root.parent / "vessel_v2"
    lines += _optional_v2_comparison(baseline_dir, oof)
    lines += ["## 来源与复核", "",
              "本文件仅从已完成的 JSON 报告汇总数值并检查一致性；不重新训练、推理或读取测试真值。"
              "结果以以下机器可读产物为准：", "",
              "| 来源 | SHA-256 |", "|---|---|"]
    for path in (progress_path, report_path, selection_path, prediction_path, metadata_path):
        lines.append(f"| `{path.relative_to(root).as_posix()}` | `{_sha(path)}` |")
    lines += ["", "各折原始验证结果位于 `runs/<encoder>/fold<n>/validation_report.json`；"
              "预训练来源见 `pretrained_sources.json`，版本和最终归档信息见 `version_manifest.json`。", "",
              "生成时间（Asia/Shanghai）：" + datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds") + "。", ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--v2-dir", type=Path, help="Optional old version root for a strictly matched fold-0 description")
    args = parser.parse_args()
    path = write_results(args.root, args.output, args.v2_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
