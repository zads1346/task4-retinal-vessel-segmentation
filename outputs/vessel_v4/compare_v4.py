"""Compare completed v4 OOF outputs with frozen v2/v3 on CPU only.

Reuses the exact EDT/skeleton definitions in diagnose_previous.py. This module
does not import torch, execute models, read test labels, or tune new thresholds.
Run only after v4 oof/oof_report.json and oof/selection.json have been finalized.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from diagnose_previous import IDS, METRICS, load_cases, load_probabilities, metrics, read_json, sha, validate_ids


MODES = ("none", "flip4", "d4")


def close(measured, recorded, label):
    if not np.isfinite(float(recorded)) or abs(float(measured) - float(recorded)) > 1e-9:
        raise ValueError(f"Metric/threshold disagreement: {label}: {measured} != {recorded}")


def pair(old, new, old_label, new_label, protocol):
    left = {row["id"]: row for row in old["per_image"]}
    right = {row["id"]: row for row in new["per_image"]}
    if set(left) != set(right):
        raise ValueError("Pair must compare exactly the same image IDs")
    return {
        "baseline_label": old_label, "v4_label": new_label, "protocol": protocol,
        "ids": sorted(left, key=int), "baseline": old, "v4": new,
        "delta_v4_minus_baseline": {
            "macro_mean": {key: new["macro_mean"][key] - old["macro_mean"][key] for key in METRICS},
            "per_image": [{"id": image_id, **{key: right[image_id][key] - left[image_id][key] for key in METRICS}}
                          for image_id in sorted(left, key=int)],
        },
    }


def read_oof(root, data, expected_version, cases, required_variants):
    report_path, selection_path = root / "oof/oof_report.json", root / "oof/selection.json"
    report, selection = read_json(report_path), read_json(selection_path)
    if report["version"] != expected_version or selection["version"] != expected_version:
        raise ValueError(f"Wrong version in {root}")
    if report["selection"] != selection or Path(report["data_root"]).resolve() != data:
        raise ValueError(f"Selection/data provenance mismatch in {root}")
    validate_ids(report["ids"], IDS, "OOF IDs")
    partition = report["fold_partition"]
    if len(partition) != 5 or any(len(part) != 4 for part in partition):
        raise ValueError("Expected five four-image folds")
    validate_ids([image_id for part in partition for image_id in part], IDS, "OOF fold coverage")
    if not set(required_variants).issubset(report["variants"]):
        raise ValueError(f"Missing requested OOF candidates in {root}")
    maps, source_paths = {}, [report_path, selection_path]
    for name in required_variants:
        variant = report["variants"][name]
        path = Path(variant["oof_probabilities_path"]).resolve()
        if not path.is_relative_to(root) or sha(path) != variant["oof_probabilities_sha256"]:
            raise ValueError(f"Invalid OOF probability path/hash: {name}")
        values = load_probabilities(path, IDS, cases)
        provenance = variant["prediction_provenance"]
        validate_ids(list(provenance), IDS, name + " provenance")
        for image_id in IDS:
            expected_families = {"unet_v4"} if expected_version == "v4.0.0" else {"resnet18", "resnet34"}
            if set(provenance[image_id]) != expected_families:
                raise ValueError("Unexpected held-out family provenance")
            for source in provenance[image_id].values():
                fold = source["fold"]
                if not isinstance(fold, int) or not 0 <= fold < 5 or source["held_out"] is not True or image_id not in partition[fold]:
                    raise ValueError("OOF contains non-held-out image provenance")
        fixed = metrics(values, cases, 0.5, IDS)
        native_threshold = float(variant["threshold_tuning"]["native_best_threshold"])
        tuned = metrics(values, cases, native_threshold, IDS)
        close(fixed["macro_mean"]["dice"], variant["fixed_0_50"]["mean_dice"], name + " fixed Dice")
        close(tuned["macro_mean"]["dice"], variant["native_tuned"]["mean_dice"], name + " tuned Dice")
        close(native_threshold, variant["native_tuned"]["threshold"], name + " threshold")
        maps[name] = {"probabilities": values, "fixed_0_50": fixed, "native_tuned": tuned}
        source_paths.append(path)
    selected_name = selection["selected_variant"]
    if selected_name not in maps:
        raise ValueError("Selected variant is not one of the requested comparisons")
    close(selection["threshold"], maps[selected_name]["native_tuned"]["threshold"], "selected global threshold")
    return report, selection, maps, source_paths


def table_row(label, result):
    m = result["macro_mean"]
    return (f"| {label} | {result['threshold']:.2f} | {m['dice']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | "
            f"{m['edt_le2_recall']:.6f} | {m['skeleton_recall']:.6f} | {m['thin_skeleton_recall']:.6f} |")


def add_pair_table(lines, comparison):
    lines += ["| 方案 | 阈值 | Dice | Precision | Recall | EDT≤2 召回 | 全骨架召回 | 薄骨架召回 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|",
              table_row(comparison["baseline_label"], comparison["baseline"]),
              table_row(comparison["v4_label"], comparison["v4"])]
    d = comparison["delta_v4_minus_baseline"]["macro_mean"]
    lines += ["", f"v4−对照：ΔDice **{d['dice']:+.6f}**，ΔPrecision **{d['precision']:+.6f}**，"
              f"ΔRecall **{d['recall']:+.6f}**，薄骨架召回变化 **{100*d['thin_skeleton_recall']:+.2f} 个百分点**。", ""]


def markdown(report):
    selected = report["selection"]
    lines = ["# v4 与旧版本的原生像素比较", "",
             f"v4 已保存的选择为 **{selected['selected_variant']}**，统一阈值 **{selected['threshold']:.2f}**。"
             "本报告仅重算已有概率的指标，没有重新训练、运行 GPU、搜索新阈值或读取测试真值。", "",
             "所有表格使用逐图指标的算术平均。JSON 同时保存逐图 TP/FP/FN、代理区域命中/漏检计数及 pooled 指标。"
             "原生尺寸为 H584×W565，预测规则 `p>=t`，不做缩放、膨胀或位置容差。", "",
             "## 与 v2：相同 4 图、相同 flip4 协议", "",
             "只使用 ID 1、10、15、19，v4 概率直接核对 `runs/fold0/validation_tta_probabilities.npz` "
             "与完整 OOF 中这 4 张图完全一致。下面固定阈值 0.50，控制 TTA 与阈值协议。", ""]
    add_pair_table(lines, report["matched_v2_fold0"]["flip4_fixed_0_50"])
    lines += ["两版实际选择的流程单独列出：v2 是 flip4 与其 4 图所选阈值；v4 使用其 20 图 OOF "
              f"选择的 **{selected['selected_variant']}** 与全局阈值。若 v4 为 none 或 d4，以下就不是相同 TTA 协议的比较。", ""]
    add_pair_table(lines, report["matched_v2_fold0"]["selected_pipelines_own_thresholds"])
    lines += ["没有重新给 v4 在这 4 张图上挑阈值。JSON 另保存 flip4 下套用两版全局阈值的数值；"
              "若 v4 的阈值来自其他 TTA 候选，该项不视为 flip4 独立校准成绩。", "",
              "## 与 v3：相同 20 图、相同 flip4 协议", "",
              "v3 对照为 `equal_ensemble_tta` 的完整 OOF；每张图只平均其持出折的 R18/R34。"
              "v4 每张图只使用自己的持出折。下面两者均采用 flip4 和固定阈值 0.50，"
              "便于比较整体重合、误检与细结构覆盖。", ""]
    add_pair_table(lines, report["full20_vs_v3"]["flip4_fixed_0_50"])
    lines += ["## 与 v3：各自选定的最终验证流程", "",
              f"v3 保持 flip4，v4 采用 **{selected['selected_variant']}**。"
              "本节先固定阈值 0.50，再比较各自已选阈值；如果 TTA 不同，这是完整方案比较，"
              "不是严格保持推理协议相同的结构或损失消融。", "", "固定阈值 0.50：", ""]
    add_pair_table(lines, report["full20_vs_v3"]["selected_pipelines_fixed_0_50"])
    lines += ["各自已选阈值：", ""]
    add_pair_table(lines, report["full20_vs_v3"]["selected_pipelines_own_thresholds"])
    lines += ["## v4 候选与逐图变化", "",
              "| v4 模式 | 固定 0.50 Dice | 各候选原生阈值 | 调参后 Dice | 调参后薄骨架召回 |",
              "|---|---:|---:|---:|---:|"]
    for name in MODES:
        result = report["v4_candidates"][name]
        lines.append(f"| {name}{'（选定）' if name == selected['selected_variant'] else ''} | "
                     f"{result['fixed_0_50']['macro_mean']['dice']:.6f} | {result['native_tuned']['threshold']:.2f} | "
                     f"{result['native_tuned']['macro_mean']['dice']:.6f} | {result['native_tuned']['macro_mean']['thin_skeleton_recall']:.6f} |")
    lines += ["", "以下逐图变化对应两版各自选定流程和阈值，数值为 v4−v3；不同 TTA 的限制同上。", "",
              "| ID | 持出折 | ΔDice | ΔPrecision | ΔRecall | Δ全骨架召回 | Δ薄骨架召回 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["full20_vs_v3"]["selected_pipelines_own_thresholds"]["delta_v4_minus_baseline"]["per_image"]:
        fold = next(index for index, ids in enumerate(report["fold_partition"]) if row["id"] in ids)
        lines.append(f"| {row['id']} | {fold} | {row['dice']:+.6f} | {row['precision']:+.6f} | {row['recall']:+.6f} | "
                     f"{row['skeleton_recall']:+.6f} | {row['thin_skeleton_recall']:+.6f} |")
    lines += ["", "## 定义与解释边界", "",
              "- EDT 使用真实标签到背景的欧氏像素距离。`GT & EDT<=2` 含粗血管边界，属于边界/窄区域代理，并非纯细血管。",
              "- 全骨架使用 Zhang 骨架化；薄骨架为 `skeleton(GT) & EDT<=2`，严格检查预测覆盖。"
              "这些是数字标注代理，不是解剖直径；没有物理标定，也不容忍一个像素的平移。",
              "- v4 训练时的薄区是薄骨架膨胀 1 像素再与 GT 相交；诊断指标仍沿用原始薄骨架与 EDT 代理，"
              "没有为了本次训练损失改变评价定义。",
              "- v4 保持 v2 网络但同时改变损失、采样及 TTA 候选；没有分别完成这些因素的消融，也没有完整 5 折 v2 对照。"
              "同 4 图与同 20 图比较不能拼成一个无偏因果结论。",
              "- 所有版本经历检查点、模型/推理模式或阈值选择；本报告不是嵌套验证、独立测试或 Kaggle 分数。"
              "不能把旧 v2 的 4 图均值直接与 v4 的 20 图均值相减。",
              "- 召回提高若伴随 Precision 或整体 Dice 下降，需要共同报告；单一薄骨架指标提高不等于分割任务整体更好。", "",
              "来源、哈希、逐图计数、所有比较的协议标记及校验结果保存在 `comparison.json`。"
              "原始 v3 诊断见 `previous_comparison.md`。", "",
              "生成时间（Asia/Shanghai）：" + report["created_at"] + "。", ""]
    return "\n".join(lines)


def compare(data, v2_root, v3_root, v4_root, output_dir):
    data, v2_root, v3_root, v4_root, output_dir = [Path(path).expanduser().resolve() for path in (data, v2_root, v3_root, v4_root, output_dir)]
    if any(output_dir.is_relative_to(root) for root in (v2_root, v3_root)):
        raise ValueError("Comparison outputs must not modify frozen v2/v3")
    cases = load_cases(data)
    v4_report, v4_selection, v4, sources = read_oof(v4_root, data, "v4.0.0", cases, MODES)
    v3_report, v3_selection, v3, v3_sources = read_oof(v3_root, data, "v3.0.0", cases, ("equal_ensemble_tta",))
    sources += v3_sources
    if set(v4_report["variants"]) != set(MODES) or v4_selection["selected_variant"] not in MODES:
        raise ValueError("Expected the three frozen v4 inference candidates")
    if v4_selection["tta_mode"] != v4_selection["selected_variant"] or v4_selection["tta"] != (v4_selection["tta_mode"] != "none"):
        raise ValueError("Inconsistent selected v4 TTA metadata")
    if v4_report.get("pretrained_weights") is not False or v4_report.get("external_training_data") is not False:
        raise ValueError("v4 must explicitly declare no pretrained weights/external training data")
    if v4_report["fold_partition"] != v3_report["fold_partition"]:
        raise ValueError("v4/v3 fold partitions differ")
    if v3_selection["selected_variant"] != "equal_ensemble_tta" or v3_selection["tta"] is not True:
        raise ValueError("Expected frozen v3 equal-ensemble flip4 selection")
    v2_selection_path, v2_report_path = v2_root / "selection.json", v2_root / "runs/fold0/tta_validation/validation_report.json"
    v2_selection, v2_report = read_json(v2_selection_path), read_json(v2_report_path)
    if v2_selection["tta"] is not True or v2_report["tta"] is not True:
        raise ValueError("v2 comparison requires its selected flip4 predictions")
    ids = validate_ids(v2_report["validation_ids"], ["1", "10", "15", "19"], "v2 fold0")
    validate_ids(v2_selection["validation_ids"], ids, "v2 selected IDs")
    if Path(v2_selection["checkpoint"]).resolve() != Path(v2_report["checkpoint"]).resolve():
        raise ValueError("v2 probability report is not from the selected checkpoint")
    validate_ids(v4_report["fold_partition"][0], ids, "v4 fold0")
    v2_path = v2_root / "runs/fold0/tta_validation/validation_probabilities.npz"
    v4_fold0_path = v4_root / "runs/fold0/validation_tta_probabilities.npz"
    v4_fold0_report_path = v4_root / "runs/fold0/validation_report.json"
    v4_fold0_report = read_json(v4_fold0_report_path)
    if v4_fold0_report["version"] != "v4.0.0" or v4_fold0_report["fold"] != 0:
        raise ValueError("Unexpected v4 fold0 report provenance")
    validate_ids(v4_fold0_report["validation_ids"], ids, "v4 fold0 report")
    p2, p4_flip0 = load_probabilities(v2_path, ids, cases), load_probabilities(v4_fold0_path, ids, cases)
    for image_id in ids:
        if not np.array_equal(p4_flip0[image_id], v4["flip4"]["probabilities"][image_id]):
            raise ValueError("v4 fold0 flip4 probability differs from its OOF record")
    sources += [v2_selection_path, v2_report_path, v2_path, v4_fold0_path, v4_fold0_report_path]
    selected_mode = v4_selection["selected_variant"]
    selected4 = v4[selected_mode]
    p3 = v3["equal_ensemble_tta"]["probabilities"]
    p4_selected = selected4["probabilities"]
    t2, t3, t4 = [float(item["threshold"]) for item in (v2_selection, v3_selection, v4_selection)]
    fixed2, tuned2 = metrics(p2, cases, 0.5, ids), metrics(p2, cases, t2, ids)
    fixed4_flip0 = metrics(p4_flip0, cases, 0.5, ids)
    close(fixed2["macro_mean"]["dice"], v2_report["at_0_5"]["mean_dice"], "v2 fold0 fixed")
    close(tuned2["macro_mean"]["dice"], v2_selection["native_tuned_mean_dice"], "v2 selected")
    close(fixed4_flip0["macro_mean"]["dice"], v4_fold0_report["tta_at_0_5"]["mean_dice"], "v4 fold0 flip4")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "data_root": str(data), "fold_partition": v4_report["fold_partition"],
        "selection": {key: v4_selection[key] for key in ("version", "selected_variant", "threshold", "tta_mode")},
        "metric_implementation": {"module": str(Path(__file__).with_name("diagnose_previous.py")), "sha256": sha(Path(__file__).with_name("diagnose_previous.py")),
                                  "definition": "Identical native EDT<=2, Zhang skeleton, thin skeleton EDT<=2, exact-pixel coverage; not anatomical diameter"},
        "input_sources": [{"path": str(path), "sha256": sha(path)} for path in sources],
        "matched_v2_fold0": {
            "flip4_fixed_0_50": pair(fixed2, fixed4_flip0, "v2 flip4", "v4 flip4", "Same four images, same flip4 TTA, fixed 0.50; different trained models"),
            "flip4_at_version_selected_thresholds": pair(tuned2, metrics(p4_flip0, cases, t4, ids), "v2 flip4", "v4 flip4",
                "Same flip4 protocol, each version's global selected threshold; v4 threshold may have been calibrated for a different selected TTA mode"),
            "selected_pipelines_own_thresholds": pair(tuned2, metrics(p4_selected, cases, t4, ids), "v2 selected flip4", f"v4 selected {selected_mode}",
                "Same four images but selected TTA may differ; v2 threshold selected on four images, v4 threshold selected on twenty OOF images"),
        },
        "full20_vs_v3": {
            "flip4_fixed_0_50": pair(v3["equal_ensemble_tta"]["fixed_0_50"], v4["flip4"]["fixed_0_50"], "v3 equal ensemble flip4", "v4 flip4",
                                     "Same twenty images/folds, same flip4 TTA, fixed 0.50; architecture/training differ"),
            "selected_pipelines_fixed_0_50": pair(v3["equal_ensemble_tta"]["fixed_0_50"], selected4["fixed_0_50"], "v3 selected flip4", f"v4 selected {selected_mode}",
                                                  "Same twenty images, fixed 0.50; selected TTA modes differ unless v4 selects flip4"),
            "selected_pipelines_own_thresholds": pair(metrics(p3, cases, t3, IDS), metrics(p4_selected, cases, t4, IDS), "v3 selected flip4", f"v4 selected {selected_mode}",
                                                      "Same twenty images, each pipeline's selected threshold; selected TTA modes may differ; not a strict ablation"),
        },
        "v4_candidates": {name: {"fixed_0_50": entry["fixed_0_50"], "native_tuned": entry["native_tuned"]} for name, entry in v4.items()},
        "same_selected_tta_protocol_as_v3": selected_mode == "flip4",
        "new_threshold_search_performed": False, "gpu_used": False, "test_data_used": False,
        "limitations": ["Repeated checkpoint/TTA/threshold selection is not nested or independent validation.",
                        "No complete five-fold v2 control exists; four-image and twenty-image comparisons are separate cohorts.",
                        "V4 changes loss, sampling and TTA options together, so no single-factor causal claim is supported.",
                        "EDT<=2 GT pixels include thick-vessel boundaries; thin skeleton is a distinct digital proxy, not physical diameter.",
                        "The official 512 conversion and hidden Kaggle labels are not evaluated here."],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "RESULT_COMPARISON.md").write_text(markdown(report), encoding="utf-8")
    return report


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here.parents[1] / "work/data")
    parser.add_argument("--v2-root", type=Path, default=here.parent / "vessel_v2")
    parser.add_argument("--v3-root", type=Path, default=here.parent / "vessel_v3")
    parser.add_argument("--v4-root", type=Path, default=here)
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    result = compare(args.data, args.v2_root, args.v3_root, args.v4_root, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "selection": result["selection"],
                      "v4_minus_v3_flip4_fixed_0_5": result["full20_vs_v3"]["flip4_fixed_0_50"]["delta_v4_minus_baseline"]["macro_mean"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
