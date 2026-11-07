"""Run the predefined v4 U-Net five-fold experiment sequentially on one GPU."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
# Process-local limits: avoid large host-side BLAS thread pools on this 16 GB machine.
# Set before importing NumPy indirectly through Torch; no system/user settings change.
for key, value in {'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'4',
                   'MKL_NUM_THREADS':'4', 'NUMEXPR_NUM_THREADS':'4'}.items():
    os.environ[key] = value
import torch


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.steps != 6000 or args.batch_size != 8:
        raise ValueError('The frozen v4 release uses 6000 steps and batch 8; use train.py in a separate directory for smoke experiments')
    args.data, args.output = args.data.resolve(), args.output.resolve()
    code = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    progress = {'version':'v4.0.0', 'status':'running', 'model_family':'unet_v4', 'folds':5,
                'started_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'steps_per_fold':args.steps, 'completed':[]}
    started = time.perf_counter()
    for fold in range(5):
        run = args.output / f'fold{fold}'
        run.mkdir(parents=True, exist_ok=True)
        progress.update(current_fold=fold, current_run=str(run))
        write(args.output/'cv_progress.json', progress)
        report_path = run/'validation_report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            config = json.loads((run/'config.json').read_text())
            if (report['completed_steps'] != args.steps or config['fold'] != fold
                or config['version'] != 'v4.0.0' or config['batch_size'] != args.batch_size
                or Path(config['data']).resolve() != args.data
                or config['thin_weight'] != 0.5 or config['skeleton_weight'] != 0.03):
                raise ValueError('Existing completed run differs from the predefined v4 recipe')
            print(f'Reusing completed fold {fold}', flush=True)
        else:
            command = [sys.executable, str(code/'train.py'), '--data', str(args.data), '--output', str(run),
                       '--fold', str(fold), '--steps', str(args.steps), '--batch-size', str(args.batch_size),
                       '--version', 'v4.0.0', '--log-every', '100', '--eval-every', '500']
            if (run/'last.pt').exists():
                command += ['--resume', str(run/'last.pt')]
                last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
                if last['step'] == args.steps:
                    command += ['--finalize-only']
                del last
            print(f'Starting v4 fold {fold}/4, {args.steps} steps', flush=True)
            with (run/'console.log').open('a', encoding='utf-8') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                progress.update(status='failed', failed_run=str(run), return_code=result.returncode)
                write(args.output/'cv_progress.json', progress)
                raise RuntimeError(f'Training failed: {run / "console.log"}')
            report = json.loads(report_path.read_text())
        progress['completed'].append({'family':'unet_v4', 'fold':fold, 'run':str(run),
            'best_step':report['best_step'], 'weight_kind':report['weight_kind'],
            'native_dice_at_0_5':report['at_0_5']['mean_dice']})
        write(args.output/'cv_progress.json', progress)
        print(json.dumps({'event':'fold_complete', **progress['completed'][-1]}), flush=True)
    progress.update(status='training_complete', elapsed_seconds=time.perf_counter()-started)
    write(args.output/'cv_progress.json', progress)
    result = subprocess.run([sys.executable, str(code/'aggregate_oof.py'), '--runs', str(args.output),
                             '--data', str(args.data), '--output', str(code/'oof')])
    progress.update(status='completed' if result.returncode == 0 else 'aggregation_failed',
                    elapsed_seconds=time.perf_counter()-started)
    write(args.output/'cv_progress.json', progress)
    if result.returncode:
        raise RuntimeError('OOF aggregation failed; inspect driver output')
    print('All five v4 models and OOF aggregation completed.', flush=True)


if __name__ == '__main__':
    main()
