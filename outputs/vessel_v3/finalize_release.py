"""Create the v3 release registry only after real CV, prediction and audit pass."""
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
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-passed', type=int, required=True,
                        help='Number of tests actually observed passing in this release session')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if root.name != 'vessel_v3' or root.parent.name != 'outputs':
        raise ValueError('Release must run from outputs/vessel_v3')
    progress = read(root / 'runs/cv_progress.json')
    selection = read(root / 'oof/selection.json')
    oof = read(root / 'oof/oof_report.json')
    prediction = read(root / 'submission/prediction_run.json')
    audit = read(root / 'independent_audit.json')
    if progress['status'] != 'completed' or len(progress['completed']) != 10:
        raise ValueError('All ten fold runs and OOF aggregation must complete')
    expected_runs = {(family, fold) for family in ('resnet18', 'resnet34') for fold in range(5)}
    if {(item['encoder'], item['fold']) for item in progress['completed']} != expected_runs:
        raise ValueError('Completed runs must be the ten unique predefined models')
    if any(Path(item['run']).resolve() != root / 'runs' / item['encoder'] / f"fold{item['fold']}"
           for item in progress['completed']):
        raise ValueError('Completed run path differs from its canonical fold directory')
    if audit['status'] != 'passed' or audit['failure_count'] != 0:
        raise ValueError('Independent final audit must pass')
    if audit.get('version') != 'v3.0.0' or Path(audit['root']).resolve() != root or audit['check_count'] <= 0:
        raise ValueError('Independent audit belongs to a different release')
    audited_hashes = audit['artifact_sha256']
    required_audited = [root / name for name in ('previous_versions_snapshot.json',
                        'oof/selection.json', 'oof/oof_report.json', 'submission/submission.csv',
                        'submission/prediction_run.json', 'submission/test_probabilities.npz')]
    required_audited += [Path(item['run']) / 'best.pt' for item in progress['completed']]
    if not all(str(path.resolve()) in audited_hashes for path in required_audited):
        raise ValueError('Independent audit omitted a critical release artifact')
    for name, expected_sha in audited_hashes.items():
        if artifact(Path(name))['sha256'] != expected_sha:
            raise ValueError('Artifact changed after independent audit: ' + name)
    if prediction['selection_sha256'] != artifact(root / 'oof/selection.json')['sha256']:
        raise ValueError('Final predictions use a different selection')
    if not (root / 'RESULTS.md').is_file() or args.pytest_passed <= 0:
        raise ValueError('Final results and observed passing tests are required')
    metadata = prediction['submission_checks']
    if not metadata['official_script_verified'] or not metadata['roundtrip_verified']:
        raise ValueError('Official conversion and round-trip must pass')

    shutil.copy2(root / 'submission/submission.csv', root / 'submission_v3.0.0.csv')
    model_rows = []
    for item in progress['completed']:
        run = Path(item['run'])
        config, report = read(run / 'config.json'), read(run / 'validation_report.json')
        model_rows.append({
            'family': item['encoder'], 'fold': item['fold'],
            'checkpoint': artifact(run / 'best.pt'),
            'resume_checkpoint': artifact(run / 'last.pt'),
            'config': artifact(run / 'config.json'),
            'best_step': report['best_step'], 'weight_kind': report['weight_kind'],
            'completed_steps': report['completed_steps'],
            'train_ids': config['train_ids'], 'val_ids': config['val_ids'],
            'pretrained_sha256': config['pretrained_sha256'],
        })
    write(root / 'model_manifest.json', {'version': 'v3.0.0', 'all_trained_models': model_rows,
          'selected_test_models': selection['models'], 'inference_selection': selection,
          'note': 'All ten best and resume checkpoints are retained in their versioned run directories.'})

    # A descriptive comparison on the same four held-out images and threshold.
    v2_run = root.parent / 'vessel_v2/runs/fold0'
    v2 = read(v2_run / ('tta_validation/validation_report.json' if selection['tta'] else 'validation_report.json'))
    variant = oof['variants'][selection['selected_variant']]
    fold0_ids = set(oof['fold_partition'][0])
    if fold0_ids != set(v2['validation_ids']) or bool(v2.get('tta', False)) != selection['tta']:
        raise ValueError('Cannot compare different fold-0 validation images')
    v3_score = sum(row['dice'] for row in variant['fixed_0_50']['per_image'] if row['id'] in fold0_ids) / 4
    v2_score = v2['at_0_5']['mean_dice']
    comparison = {'validation_ids': sorted(fold0_ids, key=int), 'threshold': 0.5,
                  'tta': selection['tta'], 'v2_native_mean_dice': v2_score,
                  'v3_selected_oof_native_mean_dice_on_same_images': v3_score,
                  'difference_v3_minus_v2': v3_score - v2_score,
                  'scope': 'Descriptive reused four-image subset; does not establish generalization improvement or a Kaggle score.'}
    write(root / 'matched_v2_comparison.json', comparison)

    archive_path = root.parent / 'vessel_v3.0.0_source.zip'
    source_files = [p for p in root.iterdir() if p.is_file() and p.suffix in
                    {'.py', '.ps1', '.txt', '.md', '.json'} and p.name != 'version_manifest.json']
    source_files += [root / 'VERSION']
    source_files += [p for p in (root / 'tests').rglob('*.py')]
    source_files += [root / 'oof/selection.json', root / 'oof/oof_report.json', root / 'runs/cv_progress.json']
    for item in progress['completed']:
        run = Path(item['run'])
        source_files += [run / name for name in ('config.json', 'validation_report.json', 'training.jsonl', 'validation.jsonl')]
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(source_files)):
            archive.write(path, 'vessel_v3/' + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Source archive CRC verification failed')

    manifest_path = root / 'version_manifest.json'
    manifest = read(manifest_path)
    manifest.update(status='completed', completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    cv_elapsed_seconds=progress['elapsed_seconds'], selected_variant=selection['selected_variant'],
                    selected_model_count=len(selection['models']), threshold=selection['threshold'],
                    tta=selection['tta'], family_weights=selection['family_weights'],
                    native_oof_dice_at_0_5=selection['fixed_0_5_mean_dice'],
                    native_oof_dice_at_selected_threshold=selection['native_tuned_mean_dice'],
                    matched_v2_comparison=comparison,
                    verification={'pytest_passed': args.pytest_passed, 'independent_audit_checks': audit['check_count'],
                                  'independent_audit_failures': 0, 'previous_versions_unchanged': True,
                                  'official_converter_and_rle_verified': True},
                    artifacts={name: artifact(root / name) for name in
                        ('submission_v3.0.0.csv', 'model_manifest.json', 'RESULTS.md', 'README.md',
                         'oof/selection.json', 'oof/oof_report.json', 'submission/prediction_run.json',
                         'submission/test_probabilities.npz', 'independent_audit.json',
                         'matched_v2_comparison.json', 'environment_v3.txt', 'VISUAL_REVIEW.md',
                         'oof/validation_worst4_preview.png', 'submission/test_preview.png')},
                    source_archive=artifact(archive_path))
    write(manifest_path, manifest)
    registry = root.parent / 'versions'
    registry.mkdir(exist_ok=True)
    write(registry / 'v3.0.0.json', {
        'version': 'v3.0.0', 'status': 'completed', 'directory': str(root),
        'manifest': str(manifest_path), 'manifest_sha256': artifact(manifest_path)['sha256'],
        'submission': artifact(root / 'submission_v3.0.0.csv'),
        'model_manifest': artifact(root / 'model_manifest.json'),
        'native_20_image_oof_dice': selection['native_tuned_mean_dice'], 'parents': ['v1.0.0', 'v2.0.0']})
    index = read(registry / 'index.json')
    index.update(latest_completed='v3.0.0', in_progress=None,
                 caveat='v3 uses full 20-image OOF; v1/v2 used a reused four-image holdout. See matched_v2_comparison.json for a descriptive same-image comparison. Hidden scoring convention remains unconfirmed.')
    index.pop('recommended_for_this_holdout', None)
    index['versions']['v3.0.0'] = 'v3.0.0.json'
    write(registry / 'index.json', index)
    print(json.dumps({'version': 'v3.0.0', 'status': 'completed', 'manifest': str(manifest_path),
                      'csv': manifest['artifacts']['submission_v3.0.0.csv'], 'source_archive': manifest['source_archive']}, indent=2))


if __name__ == '__main__':
    main()
