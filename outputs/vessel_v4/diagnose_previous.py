"""CPU-only native-pixel diagnosis of frozen v2/v3 vessel predictions.

Writes previous_comparison.json and previous_comparison.md in this directory
by default. It never loads checkpoints, invokes torch, trains models, installs
packages, or reads test images/labels. Dependencies: NumPy, Pillow, SciPy,
scikit-image, all already present in the task environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image
import scipy
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
import skimage
from skimage.morphology import skeletonize


IDS = [str(index) for index in range(1, 21)]
PROXIES = ("edt_le2", "skeleton", "thin_skeleton", "thick_skeleton")
METRICS = ("dice", "precision", "recall", *(name + "_recall" for name in PROXIES))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ids(values, expected, label):
    ids = [str(value) for value in values]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"ID mismatch: {label}: {ids}")
    return sorted(ids, key=int)


def load_cases(data):
    cases = {}
    for image_id in IDS:
        image_path = data / f"train/image/{image_id}.jpg"
        label_path = data / f"train/label/{image_id}.jpg"
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(label_path) as source:
            label = np.asarray(source.convert("L"))
        if image.shape[:2] != label.shape or not set(np.unique(label)).issubset({0, 255}):
            raise ValueError(f"Unexpected image/label shape or values: {image_id}")
        target = label > 127
        distance = distance_transform_edt(target)
        skeleton = skeletonize(target, method="zhang")
        approximate_fov = image.max(axis=-1) > 0.05
        luminance = image @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        masks = {
            "edt_le2": target & (distance <= 2.0),
            "skeleton": skeleton,
            "thin_skeleton": skeleton & (distance <= 2.0),
            "thick_skeleton": skeleton & (distance > 2.0),
        }
        profile = {
            "id": image_id,
            "shape_hw": list(target.shape),
            "gt_vessel_pixels": int(target.sum()),
            "gt_vessel_density_native": float(target.mean()),
            "approximate_fov_pixels": int(approximate_fov.sum()),
            "gt_vessel_density_in_approximate_fov": float((target & approximate_fov).sum() / approximate_fov.sum()),
            "mean_luminance_in_approximate_fov": float(luminance[approximate_fov].mean()),
            "gt_skeleton_pixels": int(skeleton.sum()),
            "gt_thin_skeleton_pixels": int(masks["thin_skeleton"].sum()),
            "gt_thin_skeleton_fraction": float(masks["thin_skeleton"].sum() / skeleton.sum()),
            "image_sha256": sha(image_path),
            "label_sha256": sha(label_path),
        }
        cases[image_id] = {"target": target, "proxy_masks": masks, "profile": profile}
    return cases


def load_probabilities(path, ids, cases):
    with np.load(path, allow_pickle=False) as archive:
        validate_ids(archive.files, ids, str(path))
        probabilities = {}
        for image_id in ids:
            values = np.asarray(archive[image_id])
            if (values.shape != cases[image_id]["target"].shape or not np.issubdtype(values.dtype, np.number)
                    or np.iscomplexobj(values) or not np.isfinite(values).all()
                    or values.min() < 0 or values.max() > 1):
                raise ValueError(f"Invalid native probability: {path}: {image_id}")
            probabilities[image_id] = values
    return probabilities


def metrics(probabilities, cases, threshold, ids):
    rows = []
    for image_id in ids:
        target = cases[image_id]["target"]
        prediction = probabilities[image_id] >= threshold
        tp = int((prediction & target).sum())
        fp = int((prediction & ~target).sum())
        fn = int((~prediction & target).sum())
        row = {"id": image_id, "tp": tp, "fp": fp, "fn": fn,
               "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
               "precision": tp / (tp + fp) if tp + fp else 0.0,
               "recall": tp / (tp + fn) if tp + fn else 1.0,
               "predicted_positive_pixels": int(prediction.sum()),
               "predicted_positive_fraction": float(prediction.mean())}
        for name, mask in cases[image_id]["proxy_masks"].items():
            count = int(mask.sum())
            hits = int((prediction & mask).sum())
            row[name + "_gt_pixels"] = count
            row[name + "_hits"] = hits
            row[name + "_misses"] = count - hits
            row[name + "_recall"] = hits / count if count else None
        rows.append(row)
    return {"threshold": float(threshold), **summarize_rows(rows), "per_image": rows}


def summarize_rows(rows):
    means = {}
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        means[metric] = float(np.mean(values)) if values else None
    tp, fp, fn = [sum(row[key] for row in rows) for key in ("tp", "fp", "fn")]
    pooled = {"tp": tp, "fp": fp, "fn": fn,
              "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
              "precision": tp / (tp + fp) if tp + fp else 0.0,
              "recall": tp / (tp + fn) if tp + fn else 1.0}
    for name in PROXIES:
        count = sum(row[name + "_gt_pixels"] for row in rows)
        hits = sum(row[name + "_hits"] for row in rows)
        pooled[name + "_gt_pixels"] = count
        pooled[name + "_hits"] = hits
        pooled[name + "_misses"] = count - hits
        pooled[name + "_recall"] = hits / count if count else None
    return {"n_images": len(rows), "macro_mean": means, "pooled_counts_and_metrics": pooled}


def delta(v2, v3):
    left = {row["id"]: row for row in v2["per_image"]}
    right = {row["id"]: row for row in v3["per_image"]}
    if set(left) != set(right):
        raise ValueError("Unmatched comparison IDs")
    return {
        "direction": "v3 minus v2; Dice/precision/recall differences use units on [0,1]",
        "macro_mean": {key: v3["macro_mean"][key] - v2["macro_mean"][key] for key in METRICS},
        "per_image": [
            {"id": image_id, **{key: right[image_id][key] - left[image_id][key] for key in METRICS}}
            for image_id in sorted(left, key=int)
        ],
    }


def quartiles(profile_key, rows, cases):
    ordered = sorted(rows, key=lambda row: (cases[row["id"]]["profile"][profile_key], int(row["id"])))
    result = []
    for index in range(4):
        group = ordered[index * 5:(index + 1) * 5]
        values = [cases[row["id"]]["profile"][profile_key] for row in group]
        result.append({"rank_quartile": index + 1, "ids": [row["id"] for row in group],
                       "profile_min": min(values), "profile_max": max(values),
                       "profile_mean": float(np.mean(values)), **summarize_rows(group)})
    return {"profile": profile_key, "rule": "Sort feature ascending, tie-break numeric ID, four groups of five; descriptive only", "groups": result}


def render_markdown(report):
    comparison = report["matched_fold0"]
    lines = ["# v2 / v3 血管分割误差诊断", "",
             "比较只使用提供的训练标签与已保存的原生尺寸概率，没有运行 GPU、重新训练或读取测试真值。"
             "两版公平对齐的子集为 fold 0：ID " + "、".join(comparison["ids"]) + "；两者都使用已选定的 TTA 预测。", "",
             "## 先读指标定义", "",
             "所有重合判断都在原生 584×565 像素网格上进行，阈值规则为 `p >= t`，不做 512 缩放、膨胀或距离容差。"
             "表中指标均先逐图计算再取平均，JSON 同时保存合并像素计数与 micro/pooled 指标。", "",
             "- **EDT≤2 像素召回**：真实血管像素中，到最近背景像素的欧氏距离不超过 2 像素者被预测覆盖的比例。"
             "它包含粗血管的边界带，不能单独叫作纯细血管召回。",
             "- **全骨架召回**：对真实标签使用 Zhang 骨架化后，骨架像素被预测覆盖的比例。",
             "- **薄骨架召回**：仅在真实骨架且 EDT≤2 的点上计算覆盖率，是更针对细小标注分支的代理。"
             "EDT>2 的骨架另报粗骨架召回。",
             "- 这些是二值标注的数字几何代理，受像素化、标签厚度和骨架化影响；没有物理标定，"
             "**不是解剖血管直径，也不是临床测量**。严格骨架覆盖还会受一像素位置偏移影响。", "",
             "## 同 4 张图的实际比较", "",
             "| 设置 | 版本 | 阈值 | Dice | Precision | Recall | EDT≤2 召回 | 全骨架召回 | 薄骨架召回 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for regime, title in (("fixed_0_50", "固定阈值"), ("selected_thresholds", "各自已选阈值")):
        for version in ("v2", "v3"):
            result = comparison[regime][version]
            m = result["macro_mean"]
            lines.append(f"| {title} | {version} | {result['threshold']:.2f} | {m['dice']:.6f} | {m['precision']:.6f} | "
                         f"{m['recall']:.6f} | {m['edt_le2_recall']:.6f} | {m['skeleton_recall']:.6f} | {m['thin_skeleton_recall']:.6f} |")
    fixed = comparison["fixed_0_50"]["delta_v3_minus_v2"]["macro_mean"]
    tuned = comparison["selected_thresholds"]["delta_v3_minus_v2"]["macro_mean"]
    lines += ["", f"固定 0.50 时，v3−v2 的 Dice 差为 **{fixed['dice']:+.6f}**，薄骨架召回差为 "
              f"**{100 * fixed['thin_skeleton_recall']:+.2f} 个百分点**；各自已选阈值下分别为 "
              f"**{tuned['dice']:+.6f}** 与 **{100 * tuned['thin_skeleton_recall']:+.2f} 个百分点**。", "",
              "阈值来源不同：v2 在这 4 张图上选择阈值，v3 在全部 20 张 OOF 图上选择阈值。"
              "本诊断没有为 v3 重新搜索这 4 张图的最优阈值；固定 0.50 一行用于控制阈值这一差异。"
              "两版的模型/检查点也都经历验证选择，不能把它们当成独立测试结果。", "",
              "在各自已选阈值下，逐图差异如下（均为 v3−v2）：", "",
              "| ID | ΔDice | ΔPrecision | ΔRecall | Δ薄骨架召回 |",
              "|---:|---:|---:|---:|---:|"]
    for row in comparison["selected_thresholds"]["delta_v3_minus_v2"]["per_image"]:
        lines.append(f"| {row['id']} | {row['dice']:+.6f} | {row['precision']:+.6f} | {row['recall']:+.6f} | {row['thin_skeleton_recall']:+.6f} |")
    lines += ["", "## 测量支持的结论与尚未证明的原因", ""]
    lines.extend("- " + observation for observation in report["measured_observations"])
    lines += ["", "可能原因只能作为后续消融假设：两版同时改变了网络结构、预训练初始化、归一化方式、"
              "学习率、训练时长和模型融合，现有比较不能把差距归因于某一个因素。"
              "尤其不能据此断言“预训练无效”或“下采样必然导致这些漏检”。"
              "v4 应在固定划分与指标下逐项检验细节保留、训练充分程度和损失/采样修改，"
              "同时监控误检与整体 Dice，避免用更粗的预测换取单一召回指标上升。", "",
              "## v3 全部 20 图的错误分布", "",
              "以下使用 v3 选定方案 `equal_ensemble_tta` 与既定阈值 "
              f"{report['v3_full_oof']['selected']['threshold']:.2f}，仅用于内部诊断。"
              "每张图的概率来自对应持出折的两模型平均，没有混入其余训练过该图的模型。", "",
              "| ID | GT 血管占比 | 视野内亮度代理 | Dice | Precision | Recall | 薄骨架召回 | 漏掉薄骨架点 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["v3_full_oof"]["selected"]["per_image"]:
        p = report["image_profiles"][row["id"]]
        lines.append(f"| {row['id']} | {100*p['gt_vessel_density_native']:.2f}% | {p['mean_luminance_in_approximate_fov']:.3f} | "
                     f"{row['dice']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['thin_skeleton_recall']:.6f} | {row['thin_skeleton_misses']} |")
    lines += ["", "血管占比以整张原生标签为分母。亮度采用 RGB 加权灰度均值，近似视野取 `max(R,G,B)>0.05`；"
              "这不是相机曝光或病理测量。下面按属性排序分成 4 组，每组 5 张，属于小样本描述，不做因果或显著性结论。", "",
              "| 分组属性 | 从低到高组 | ID | Dice | Recall | 薄骨架召回 |",
              "|---|---:|---|---:|---:|---:|"]
    for key, title in (("density", "GT 血管占比"), ("brightness", "亮度代理")):
        for group in report["v3_full_oof"]["rank_quartiles"][key]["groups"]:
            m = group["macro_mean"]
            lines.append(f"| {title} | {group['rank_quartile']} | {','.join(group['ids'])} | {m['dice']:.6f} | {m['recall']:.6f} | {m['thin_skeleton_recall']:.6f} |")
    worst = report["v3_full_oof"]["lowest_thin_skeleton_recall_ids"]
    lines += ["", "薄骨架召回最低的 5 张图：**" + "、".join(worst) + "**。完整逐图计数、宏平均、"
              "合并像素指标与属性相关系数保存在 `previous_comparison.json`。这些排序用于定位检查对象，"
              "不代表新增的独立验证结论。", "",
              "## 可复现与范围", "",
              "运行命令：`..\\..\\work\\.venv-gpu\\Scripts\\python.exe diagnose_previous.py`（在本目录执行）。"
              "脚本只读取旧版本与提供的训练集，只向指定的新输出目录写 JSON/Markdown。"
              "没有改动 v1/v2/v3、下载图像数据或模型权重，也没有 GPU 推理。", "",
              "骨架实现固定为 `skimage.morphology.skeletonize(..., method='zhang')`；欧氏距离采用 "
              "`scipy.ndimage.distance_transform_edt`，单位为原生像素。环境版本与输入 SHA-256 均记录在 JSON。", "",
              "本报告所有结论均限定于可见训练标签与既有折外预测；未评价隐藏测试标签、官方 512 转换指标或 Kaggle 分数。", ""]
    return "\n".join(lines)


def diagnose(data, v2_root, v3_root, output_dir):
    data, v2_root, v3_root, output_dir = [Path(path).expanduser().resolve() for path in (data, v2_root, v3_root, output_dir)]
    if any(output_dir.is_relative_to(old_root) for old_root in (v2_root, v3_root)):
        raise ValueError("Diagnosis output must not modify a frozen old version")
    v2_selection_path, v3_selection_path = v2_root / "selection.json", v3_root / "oof/selection.json"
    v2_selection, v3_selection = read_json(v2_selection_path), read_json(v3_selection_path)
    v2_report_path = v2_root / "runs/fold0/tta_validation/validation_report.json"
    v2_report = read_json(v2_report_path)
    v3_report_path = v3_root / "oof/oof_report.json"
    v3_report = read_json(v3_report_path)
    if not (v2_selection["tta"] and v2_report["tta"] and v3_selection["tta"]):
        raise ValueError("Expected TTA predictions for both versions")
    if v3_selection["selected_variant"] != "equal_ensemble_tta":
        raise ValueError("Expected the frozen v3 equal_ensemble_tta selection")
    fold0_ids = validate_ids(v2_report["validation_ids"], ["1", "10", "15", "19"], "v2 fold0")
    validate_ids(v2_selection["validation_ids"], fold0_ids, "v2 selected fold")
    validate_ids(v3_report["fold_partition"][0], fold0_ids, "v3 fold0")
    if Path(v2_report["checkpoint"]).resolve() != Path(v2_selection["checkpoint"]).resolve():
        raise ValueError("v2 TTA report belongs to another checkpoint")
    if Path(v3_report["data_root"]).resolve() != data:
        raise ValueError("v3 report data root differs from the supplied data")
    v3_variant = v3_report["variants"]["equal_ensemble_tta"]
    for image_id in IDS:
        provenance = v3_variant["prediction_provenance"][image_id]
        if set(provenance) != {"resnet18", "resnet34"}:
            raise ValueError("Unexpected v3 OOF family provenance")
        for source in provenance.values():
            if source["held_out"] is not True or image_id not in v3_report["fold_partition"][source["fold"]]:
                raise ValueError("v3 OOF record is not a held-out prediction")
    v2_npz = v2_root / "runs/fold0/tta_validation/validation_probabilities.npz"
    v3_npz = v3_root / "oof/oof_equal_ensemble_tta.npz"
    if sha(v3_npz) != v3_variant["oof_probabilities_sha256"]:
        raise ValueError("v3 probability hash differs from its OOF report")
    cases = load_cases(data)
    v2_probabilities = load_probabilities(v2_npz, fold0_ids, cases)
    v3_probabilities = load_probabilities(v3_npz, IDS, cases)
    fixed_v2 = metrics(v2_probabilities, cases, 0.5, fold0_ids)
    fixed_v3 = metrics(v3_probabilities, cases, 0.5, fold0_ids)
    selected_v2 = metrics(v2_probabilities, cases, float(v2_selection["threshold"]), fold0_ids)
    selected_v3 = metrics(v3_probabilities, cases, float(v3_selection["threshold"]), fold0_ids)
    full_fixed = metrics(v3_probabilities, cases, 0.5, IDS)
    full_selected = metrics(v3_probabilities, cases, float(v3_selection["threshold"]), IDS)
    checks = [
        (fixed_v2["macro_mean"]["dice"], v2_report["at_0_5"]["mean_dice"]),
        (selected_v2["macro_mean"]["dice"], v2_selection["native_tuned_mean_dice"]),
        (full_fixed["macro_mean"]["dice"], v3_variant["fixed_0_50"]["mean_dice"]),
        (full_selected["macro_mean"]["dice"], v3_variant["native_tuned"]["mean_dice"]),
    ]
    if any(abs(measured - saved) > 1e-9 for measured, saved in checks):
        raise ValueError("Recomputed native Dice disagrees with a frozen source report")

    fixed_delta, selected_delta = delta(fixed_v2, fixed_v3), delta(selected_v2, selected_v3)
    observations = [
        f"在共同4图、TTA、固定0.50下，v3相对v2的Dice差为{fixed_delta['macro_mean']['dice']:+.6f}，"
        f"Precision差为{fixed_delta['macro_mean']['precision']:+.6f}，Recall差为{fixed_delta['macro_mean']['recall']:+.6f}。",
        f"相同阈值下，v3薄骨架召回相对v2变化{100*fixed_delta['macro_mean']['thin_skeleton_recall']:+.2f}个百分点；"
        f"各自已选阈值下仍变化{100*selected_delta['macro_mean']['thin_skeleton_recall']:+.2f}个百分点。",
    ]
    if selected_delta["macro_mean"]["recall"] >= 0 and selected_delta["macro_mean"]["precision"] < 0 and selected_delta["macro_mean"]["dice"] < 0:
        observations.append("降低到v3已选阈值后，其总体召回已略高于v2，但Precision和Dice仍更低；"
                            "本次差距不能仅用过高的二值阈值解释，同时存在误检与局部细结构覆盖不足的问题。")
    if selected_delta["macro_mean"]["edt_le2_recall"] > 0 and selected_delta["macro_mean"]["thin_skeleton_recall"] < 0:
        observations.append("各自已选阈值下，v3的EDT≤2像素召回略高，但薄骨架召回较低。"
                            "这两个区域并不等价，证明不能把粗血管边界带混同为细血管中心线。")
    v2_config_path = v2_root / "runs/fold0/config.json"
    v3_config_paths = [v3_root / f"runs/{family}/fold0/config.json" for family in ("resnet18", "resnet34")]
    fields = ("steps", "lr", "encoder_lr", "encoder_freeze_steps", "patch_size", "batch_size", "pretrained_weights", "initialization", "encoder_name")
    configs = {"v2": read_json(v2_config_path), **{family: read_json(path) for family, path in zip(("v3_resnet18", "v3_resnet34"), v3_config_paths)}}
    for config in configs.values():
        if Path(config["data"]).resolve() != data or set(config["val_ids"]) != set(fold0_ids) or set(config["train_ids"]) & set(fold0_ids):
            raise ValueError("Configuration data path or held-out partition mismatch")
    rows = full_selected["per_image"]
    correlations = []
    for feature in ("gt_vessel_density_native", "mean_luminance_in_approximate_fov", "gt_thin_skeleton_fraction"):
        for metric in ("dice", "thin_skeleton_recall"):
            rho = float(spearmanr([cases[row["id"]]["profile"][feature] for row in rows], [row[metric] for row in rows]).statistic)
            correlations.append({"feature": feature, "metric": metric, "spearman_rho": rho if np.isfinite(rho) else None,
                                 "scope": "20-image descriptive association only; no causal or significance claim"})
    source_paths = [v2_selection_path, v3_selection_path, v2_report_path, v3_report_path, v2_npz, v3_npz, v2_config_path, *v3_config_paths]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__},
        "input_sources": [{"path": str(path), "sha256": sha(path)} for path in source_paths],
        "data_root": str(data),
        "definitions": {
            "foreground": "provided label >127; vessel=1",
            "prediction": "native probability >= threshold, no resizing/dilation/tolerance",
            "edt": "scipy.ndimage.distance_transform_edt(GT), distance from GT pixels to nearest zero background pixel; unit=native pixels",
            "edt_le2": "GT & EDT<=2; includes thick-vessel boundaries, NOT a pure thin-vessel class",
            "skeleton": "skimage.morphology.skeletonize(GT, method='zhang')",
            "thin_skeleton": "GT skeleton & EDT<=2; digital thin-centerline proxy, not anatomical diameter",
            "thick_skeleton": "GT skeleton & EDT>2",
            "recall": "count(prediction & proxy) / count(proxy), exact pixel overlap, zero spatial tolerance",
            "brightness": "mean(0.299R+0.587G+0.114B) inside approximate FOV max(R,G,B)>0.05, RGB in[0,1]",
            "averaging": "macro_mean is equal image weighting; pooled_counts_and_metrics uses pooled pixel denominators",
        },
        "matched_fold0": {
            "ids": fold0_ids, "both_tta": True,
            "v3_oof_rule": "equal average of the two held-out family models for each image, not all ten models",
            "fixed_0_50": {"v2": fixed_v2, "v3": fixed_v3, "delta_v3_minus_v2": fixed_delta},
            "selected_thresholds": {"v2": selected_v2, "v3": selected_v3, "delta_v3_minus_v2": selected_delta},
            "threshold_selection_scope": {"v2": "selected on four fold0 images", "v3": "selected on all twenty OOF images; not retuned for this four-image comparison"},
        },
        "measured_observations": observations,
        "recorded_configuration_contrasts": {name: {key: config.get(key) for key in fields} for name, config in configs.items()},
        "possible_causes_not_proven": ["Architecture/detail preservation", "Different optimization budget and learning-rate schedules", "Different initialization/normalization", "Changes in precision/recall tradeoffs and family blending"],
        "v3_full_oof": {
            "ids": IDS, "selected_variant": "equal_ensemble_tta", "fixed_0_50": full_fixed, "selected": full_selected,
            "rank_quartiles": {"density": quartiles("gt_vessel_density_native", rows, cases), "brightness": quartiles("mean_luminance_in_approximate_fov", rows, cases)},
            "descriptive_correlations": correlations,
            "lowest_thin_skeleton_recall_ids": [row["id"] for row in sorted(rows, key=lambda row: (row["thin_skeleton_recall"], int(row["id"])))[:5]],
        },
        "image_profiles": {image_id: cases[image_id]["profile"] for image_id in IDS},
        "limitations": [
            "Only supplied training labels and saved held-out predictions are used; no test labels or GPU inference.",
            "Repeated model/checkpoint/threshold selection means these are not independent validation or Kaggle estimates.",
            "Two pipelines changed several factors simultaneously; no causal attribution without controlled ablation.",
            "EDT and skeleton metrics depend on raster annotations and pixel alignment; they are not anatomical vessel diameters.",
            "The native four-image comparison is distinct from the twenty-image descriptive OOF breakdown; do not directly subtract unequal cohorts.",
            "Brightness/density groups and correlations contain five or twenty images only and are descriptive, not causal or confirmatory.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previous_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "previous_comparison.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main():
    here = Path(__file__).resolve().parent
    workspace = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=workspace / "work/data")
    parser.add_argument("--v2-root", type=Path, default=here.parent / "vessel_v2")
    parser.add_argument("--v3-root", type=Path, default=here.parent / "vessel_v3")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    result = diagnose(args.data, args.v2_root, args.v3_root, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "matched_fixed_0_5_delta": result["matched_fold0"]["fixed_0_50"]["delta_v3_minus_v2"]["macro_mean"],
                      "matched_selected_delta": result["matched_fold0"]["selected_thresholds"]["delta_v3_minus_v2"]["macro_mean"],
                      "v3_lowest_thin_skeleton_recall_ids": result["v3_full_oof"]["lowest_thin_skeleton_recall_ids"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
