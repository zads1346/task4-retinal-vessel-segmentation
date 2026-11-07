"""Publish local versioned v4 artifacts only after completed CV and final audit."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def artifact(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'path':str(path.resolve()), 'bytes':path.stat().st_size, 'sha256':digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-passed', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if root.name != 'vessel_v4' or root.parent.name != 'outputs':
        raise ValueError('Run from outputs/vessel_v4')
    progress = read(root/'runs/cv_progress.json')
    selection = read(root/'oof/selection.json')
    oof = read(root/'oof/oof_report.json')
    prediction = read(root/'submission/prediction_run.json')
    comparison = read(root/'comparison.json')
    audit = read(root/'independent_audit.json')
    if progress['status'] != 'completed' or len(progress['completed']) != 5:
        raise ValueError('Complete all five folds and OOF first')
    if {item['fold'] for item in progress['completed']} != set(range(5)):
        raise ValueError('Expected five distinct fold indices')
    if any(item['family'] != 'unet_v4' or Path(item['run']).resolve() != root/'runs'/f"fold{item['fold']}"
           for item in progress['completed']):
        raise ValueError('Noncanonical fold family/path')
    if audit['version'] != 'v4.0.0' or Path(audit['root']).resolve() != root or audit['status'] != 'passed' or audit['failure_count']:
        raise ValueError('Independent audit must pass for this release')
    critical = [root/name for name in ('previous_versions_snapshot.json', 'oof/selection.json',
                'oof/oof_report.json', 'submission/submission.csv', 'submission/prediction_run.json',
                'submission/test_probabilities.npz')]
    critical += [root/'runs'/f'fold{fold}'/'best.pt' for fold in range(5)]
    if not all(str(path.resolve()) in audit['artifact_sha256'] for path in critical):
        raise ValueError('Audit omitted critical artifacts')
    for name, expected in audit['artifact_sha256'].items():
        if artifact(Path(name))['sha256'] != expected:
            raise ValueError('Audited artifact changed: ' + name)
    for source in comparison['input_sources']:
        if artifact(Path(source['path']))['sha256'] != source['sha256']:
            raise ValueError('Comparison input changed: ' + source['path'])
    metric_source = comparison['metric_implementation']
    if (Path(metric_source['module']).resolve() != root/'diagnose_previous.py'
        or artifact(root/'diagnose_previous.py')['sha256'] != metric_source['sha256']):
        raise ValueError('Thin-vessel metric implementation changed after comparison')
    if prediction['version'] != 'v4.0.0' or prediction['selection_sha256'] != artifact(root/'oof/selection.json')['sha256']:
        raise ValueError('Prediction selection/version differs')
    if (comparison['selection']['version'] != 'v4.0.0'
        or comparison['selection']['selected_variant'] != selection['selected_variant']
        or comparison['selection']['threshold'] != selection['threshold']):
        raise ValueError('Comparison belongs to a different selection')
    if args.pytest_passed <= 0 or not (root/'RESULT_COMPARISON.md').is_file():
        raise ValueError('Observed passing tests and comparison report required')
    checks = prediction['submission_checks']
    if not checks['official_script_verified'] or not checks['roundtrip_verified'] or prediction['uploaded_to_kaggle']:
        raise ValueError('Submission verification or no-upload provenance differs')

    shutil.copy2(root/'submission/submission.csv', root/'submission_v4.0.0.csv')
    rows = []
    cumulative_model_seconds = 0.0
    for item in progress['completed']:
        run = Path(item['run'])
        config, report = read(run/'config.json'), read(run/'validation_report.json')
        cumulative_model_seconds += report['total_seconds']
        rows.append({'family':'unet_v4', 'fold':item['fold'], 'checkpoint':artifact(run/'best.pt'),
                     'resume_checkpoint':artifact(run/'last.pt'), 'config':artifact(run/'config.json'),
                     'best_step':report['best_step'], 'weight_kind':report['weight_kind'],
                     'completed_steps':report['completed_steps'], 'train_ids':config['train_ids'],
                     'val_ids':config['val_ids'], 'initial_weights_sha256':config['initial_weights_sha256'],
                     'pretrained_weights':False, 'external_training_data':False})
    write(root/'model_manifest.json', {'version':'v4.0.0', 'models':rows, 'selection':selection,
          'initialization':'Independent random initialization per fold; never warm-started from v2/v3 checkpoints'})

    selected = oof['variants'][selection['selected_variant']]
    same4 = comparison['matched_v2_fold0']['flip4_fixed_0_50']
    same20 = comparison['full20_vs_v3']['flip4_fixed_0_50']
    lines = ['# v4.0.0 实际完成结果', '',
             f"已完成 5/5 折，每折 6,000 步，共 30,000 个计划步骤。各折训练与验证累计 {cumulative_model_seconds/60:.1f} 分钟，不含失败尝试和重启间隔。", '',
             f"选定 **{selection['tta_mode']}**，五模型各 0.2 集成，统一阈值 **{selection['threshold']:.2f}**。完整 20 图 OOF 原生 Dice：固定 0.5 为 **{selection['fixed_0_5_mean_dice']:.6f}**；所选阈值下为 **{selection['native_tuned_mean_dice']:.6f}**。", '',
             '这些是经检查点、TTA、阈值选择的本地 OOF 指标，不是独立测试成绩或 Kaggle 得分。', '',
             '| OOF 候选 | 固定 0.5 Dice | 所选阈值 | 调参后 Dice |', '|---|---:|---:|---:|']
    for mode, variant in oof['variants'].items():
        lines.append(f"| {mode} | {variant['fixed_0_50']['mean_dice']:.6f} | {variant['native_tuned']['threshold']:.2f} | {variant['native_tuned']['mean_dice']:.6f} |")
    lines += ['', '| 折 | 验证 ID | 全局所选阈值 Dice |', '|---:|---|---:|']
    for row in selected['native_tuned']['per_fold']:
        lines.append(f"| {row['fold']} | {', '.join(row['validation_ids'])} | {row['mean_dice']:.6f} |")
    lines += ['', '同协议对照均使用 flip4 和固定阈值 0.5：', '',
              f"- 同 4 图对 v2：Dice {same4['baseline']['macro_mean']['dice']:.6f} → {same4['v4']['macro_mean']['dice']:.6f}。",
              f"- 同 20 图对 v3：Dice {same20['baseline']['macro_mean']['dice']:.6f} → {same20['v4']['macro_mean']['dice']:.6f}。", '',
              '细血管骨架召回、精度和逐图差值见 [完整比较](RESULT_COMPARISON.md)；v3 原因分析见 [前版诊断](previous_comparison.md)。没有完整五折 v2 对照或单项消融，因此不能把全部差值归因于某一损失项。', '',
              f"通过 {args.pytest_passed} 项代码测试与 {audit['check_count']} 项独立核验。v1/v2/v3 原文件哈希保持一致。分割监督只用提供的 20 张训练图及标签，无外部数据或预训练权重。", '',
              '[提交 CSV](submission_v4.0.0.csv) · [五模型清单](model_manifest.json) · [验证预览](oof/validation_worst4_preview.png) · [细血管命中/漏检](oof/thin_error_preview.png) · [测试预览](submission/test_preview.png)', '',
              'CSV 已按原官方转换器逐像素核对，未上传 Kaggle。任务文字与实际标签的极性冲突仍未确认；沿用实际白色血管和官方非零前景编码约定。', '']
    (root/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

    archive_path = root.parent/'vessel_v4.0.0_source.zip'
    sources = [p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.ps1','.md','.json','.txt'} and p.name != 'version_manifest.json']
    sources += [root/'VERSION', root/'oof/oof_report.json', root/'oof/selection.json', root/'runs/cv_progress.json']
    sources += list((root/'tests').glob('*.py'))
    for item in progress['completed']:
        sources += [Path(item['run'])/name for name in ('config.json','validation_report.json','training.jsonl','validation.jsonl')]
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(sources)):
            archive.write(path, 'vessel_v4/' + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Source ZIP CRC verification failed')
    manifest_path = root/'version_manifest.json'
    manifest = read(manifest_path)
    manifest.update(status='completed', completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    cv_cumulative_model_seconds=cumulative_model_seconds,
                    latest_driver_elapsed_seconds=progress['elapsed_seconds'], selected_variant=selection['selected_variant'],
                    selected_model_count=5, threshold=selection['threshold'], tta_mode=selection['tta_mode'],
                    native_oof_dice_at_0_5=selection['fixed_0_5_mean_dice'],
                    native_oof_dice_at_selected_threshold=selection['native_tuned_mean_dice'],
                    matched_comparison_deltas={'vs_v2_same4_flip4_0_5':same4['delta_v4_minus_baseline']['macro_mean'],
                                              'vs_v3_same20_flip4_0_5':same20['delta_v4_minus_baseline']['macro_mean']},
                    verification={'pytest_passed':args.pytest_passed, 'independent_audit_checks':audit['check_count'],
                                  'independent_audit_failures':0, 'previous_versions_unchanged':True,
                                  'official_converter_and_rle_verified':True},
                    source_archive=artifact(archive_path),
                    artifacts={name:artifact(root/name) for name in ('submission_v4.0.0.csv','model_manifest.json',
                        'RESULTS.md','RESULT_COMPARISON.md','comparison.json','previous_comparison.md','previous_comparison.json',
                        'README.md','oof/selection.json','oof/oof_report.json','submission/prediction_run.json',
                        'submission/test_probabilities.npz','independent_audit.json','environment_v4.txt',
                        'oof/validation_worst4_preview.png','oof/thin_error_preview.png','submission/test_preview.png')})
    write(manifest_path, manifest)
    registry = root.parent/'versions'
    write(registry/'v4.0.0.json', {'version':'v4.0.0','status':'completed','directory':str(root),
          'manifest':str(manifest_path),'manifest_sha256':artifact(manifest_path)['sha256'],
          'submission':artifact(root/'submission_v4.0.0.csv'),'model_manifest':artifact(root/'model_manifest.json'),
          'native_20_image_oof_dice':selection['native_tuned_mean_dice'],'parents':['v1.0.0','v2.0.0','v3.0.0']})
    index = read(registry/'index.json')
    index.update(latest_completed='v4.0.0', in_progress=None,
                 caveat='v3/v4 have full20image OOF. v1/v2 have reused4image holdout only. See v4comparison for matched protocols; none is an independent test score.')
    index['versions']['v4.0.0'] = 'v4.0.0.json'
    write(registry/'index.json', index)
    print(json.dumps({'version':'v4.0.0','status':'completed','csv':manifest['artifacts']['submission_v4.0.0.csv'],
                      'source_archive':manifest['source_archive']}, indent=2))


if __name__ == '__main__':
    main()
