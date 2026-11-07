"""Train every predefined fold of both pretrained families, then assemble OOF."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys
import time
import torch


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    a.data=a.data.resolve();a.output=a.output.resolve()
    sources=json.loads(a.weights_manifest.read_text())['sources']
    code=Path(__file__).resolve().parent
    a.output.mkdir(parents=True,exist_ok=True)
    progress={'version':'v3.0.0','status':'running','started_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'families':['resnet18','resnet34'],'folds':5,'steps_per_fold':a.steps,'completed':[]}
    started=time.perf_counter()
    # One GPU training job at a time. No model sees its validation images in training.
    for encoder in progress['families']:
        for fold in range(5):
            run=a.output/encoder/f'fold{fold}';run.mkdir(parents=True,exist_ok=True)
            report_path=run/'validation_report.json'
            progress.update(current_encoder=encoder,current_fold=fold,current_run=str(run))
            write(a.output/'cv_progress.json',progress)
            if report_path.exists():
                report=json.loads(report_path.read_text());config=json.loads((run/'config.json').read_text())
                if (report.get('completed_steps')!=a.steps or config['encoder_name']!=encoder or config['fold']!=fold
                    or config['batch_size']!=a.batch_size or Path(config['data']).resolve()!=a.data
                    or config['pretrained_sha256']!=sources[encoder]['sha256']):
                    raise ValueError(f'Existing completed run differs from requested CV: {run}')
                print(f'Reusing completed {encoder} fold {fold}',flush=True)
            else:
                command=[sys.executable,str(code/'train.py'),'--data',str(a.data),'--output',str(run),
                         '--encoder-name',encoder,'--weights',sources[encoder]['path'],'--fold',str(fold),
                         '--steps',str(a.steps),'--batch-size',str(a.batch_size),'--log-every','100',
                         '--eval-every','500','--version','v3.0.0']
                if (run/'last.pt').exists():
                    command+=['--resume',str(run/'last.pt')]
                    last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
                    if last['step']==a.steps:command+=['--finalize-only']
                    del last
                print(f'Starting {encoder} fold {fold}/4, {a.steps} steps',flush=True)
                with (run/'console.log').open('a',encoding='utf-8') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    progress.update(status='failed',failed_run=str(run),return_code=result.returncode)
                    write(a.output/'cv_progress.json',progress)
                    raise RuntimeError(f'Training failed; inspect {run / "console.log"}')
                report=json.loads(report_path.read_text())
            item={'encoder':encoder,'fold':fold,'run':str(run),'best_step':report['best_step'],
                  'weight_kind':report['weight_kind'],'native_dice_at_0_5':report['at_0_5']['mean_dice']}
            progress['completed'].append(item)
            write(a.output/'cv_progress.json',progress)
            print(json.dumps({'event':'fold_complete',**item}),flush=True)
    progress.update(status='training_complete',elapsed_seconds=time.perf_counter()-started)
    write(a.output/'cv_progress.json',progress)
    command=[sys.executable,str(code/'aggregate_oof.py'),'--runs',str(a.output/'resnet18'),str(a.output/'resnet34'),
             '--data',str(a.data),'--output',str(code/'oof')]
    subprocess.run(command,check=True)
    progress.update(status='completed',elapsed_seconds=time.perf_counter()-started)
    write(a.output/'cv_progress.json',progress)
    print('All ten models and full OOF aggregation completed.',flush=True)


if __name__=='__main__':main()
