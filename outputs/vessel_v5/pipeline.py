"""Serial GPU experiment runner with fixed screening and honest OOF blending."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from data import load_cases, make_folds
from experiments import transform_cases
from inference import summarize, predict_image, write_json, save_preview
from model import UNet
from submission import export_submission

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DATA = PROJECT / 'work/data'
V4 = ROOT.parent / 'vessel_v4'
VERSION = 'v5.0.0'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def status(stage, **extra):
    record = {'stage':stage, 'time':time.strftime('%Y-%m-%d %H:%M:%S'), **extra}
    write_json(ROOT/'status.json', record)
    print(json.dumps(record), flush=True)


def snapshot_previous():
    marker = ROOT/'previous_versions_snapshot.json'
    if marker.exists():
        return read(marker)
    paths = []
    for name in ('vessel_baseline','vessel_v2','vessel_v3','vessel_v4'):
        paths.extend(p for p in (ROOT.parent/name).rglob('*') if p.is_file())
    paths.extend(p for p in (ROOT.parent/'versions').glob('*.json') if p.stem != VERSION)
    value = {str(p):sha(p) for p in paths}
    write_json(marker, value)
    return value


def run_fold(experiment, fold):
    directory = ROOT/'experiments'/experiment/f'fold{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    report = directory/'validation_report.json'
    if report.exists() and read(report)['completed_steps'] == 6000:
        return
    status('training', experiment=experiment, fold=fold, steps=6000)
    command = [sys.executable, '-u', str(ROOT/'train.py'), '--data', str(DATA),
               '--output', str(directory), '--experiment', experiment, '--fold', str(fold)]
    if (directory/'last.pt').exists():
        state = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
        command += ['--resume',str(directory/'last.pt')]
        if state['step']==6000:
            command += ['--finalize-only']
        del state
    with (directory/'console.log').open('a', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not report.exists():
        raise RuntimeError('Training did not finalize')


def collect(directory, folds, cases, version, experiment=None):
    result, records = {}, []
    partition = make_folds(cases)
    for fold in folds:
        folder = directory/f'fold{fold}'
        config = read(folder/'config.json')
        assert config['version']==version
        assert config['train_ids']==partition[fold]['train_ids']
        assert config['val_ids']==partition[fold]['val_ids']
        assert not set(config['train_ids']) & set(config['val_ids'])
        assert config['external_training_data'] is False and config['pretrained_weights'] is False
        assert config['initialization']=='random_from_scratch'
        if experiment:
            assert config['experiment']==experiment
        checkpoint = folder/'best.pt'
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['config']==config
        del state
        probabilities_path = folder/'validation_d4_probabilities.npz'
        with np.load(probabilities_path, allow_pickle=False) as saved:
            assert set(saved.files)==set(config['val_ids'])
            for key in saved.files:
                values = saved[key]
                assert key not in result and values.shape==cases[key]['mask'].shape
                assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
                result[key] = values
        records.append({'fold':fold,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                        'config':config,'oof_probabilities':str(probabilities_path),
                        'oof_sha256':sha(probabilities_path)})
    return result, records


def diagnostics(probabilities,cases,threshold):
    report=summarize(probabilities,cases,threshold,include_proxy=False)
    recalls=[]
    for key,p in probabilities.items():
        target=cases[key]['mask']>0.5
        thin=skeletonize(target)&(distance_transform_edt(target)<=2)
        recalls.append(float(((p>=threshold)&thin).sum()/max(1,thin.sum())))
    report['thin_skeleton_recall']=float(np.mean(recalls))
    report['mean_precision']=float(np.mean([r['precision'] for r in report['per_image']]))
    report['mean_recall']=float(np.mean([r['recall'] for r in report['per_image']]))
    return report


def blend(a,b,weight):
    assert set(a)==set(b)
    return {key:np.clip(weight*a[key]+(1-weight)*b[key],0,1) for key in a}


def predict(records,cases,experiment):
    cache=ROOT/'test_model_probabilities'
    cache.mkdir(exist_ok=True)
    transformed=transform_cases(cases,experiment) if experiment else cases
    total={key:np.zeros(case['image'].shape[:2],np.float32) for key,case in cases.items()}
    for record in records:
        name=f'{experiment or "v4"}_fold{record["fold"]}'
        status('test_prediction',model=name)
        path=cache/f'{name}.npz'
        metadata=cache/f'{name}.json'
        key={'checkpoint_sha256':sha(record['checkpoint']),'tta':'d4',
             'experiment':experiment,'experiments_sha256':sha(ROOT/'experiments.py'),
             'inference_sha256':sha(ROOT/'inference.py'),
             'images':{k:hashlib.sha256(v['image'].tobytes()).hexdigest() for k,v in transformed.items()}}
        reusable=path.exists() and metadata.exists()
        if reusable:
            saved=read(metadata)
            reusable=saved.get('key')==key and saved.get('sha256')==sha(path)
        if reusable:
            with np.load(path,allow_pickle=False) as saved:
                probabilities={k:saved[k] for k in cases}
        else:
            assert key['checkpoint_sha256']==record['checkpoint_sha256']
            state=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
            model=UNet(state['config']['base_channels']).to('cuda')
            model.load_state_dict(state['model'])
            probabilities={k:predict_image(model,v['image'],torch.device('cuda'),256,128,8,tta='d4')
                           for k,v in transformed.items()}
            np.savez_compressed(path,**probabilities)
            write_json(metadata,{'key':key,'sha256':sha(path)})
            del state,model
            torch.cuda.empty_cache()
        for k,p in probabilities.items():
            assert p.shape==total[k].shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            total[k]+=p/5
    return total


def main():
    torch.set_num_threads(4)
    assert ROOT.drive.upper()=='D:'
    previous=snapshot_previous()
    cases=load_cases(DATA,'train')
    assert len(cases)==20
    input_snapshot={str(p):sha(p) for p in (DATA/'train').rglob('*') if p.is_file()}
    write_json(ROOT/'training_data_snapshot.json',input_snapshot)
    # Equal complete budgets; these folds can be reused for the selected family.
    for experiment in ('A','B'):
        for fold in (0,1):
            run_fold(experiment,fold)
    screening={}
    for experiment in ('A','B'):
        p,_=collect(ROOT/'experiments'/experiment,[0,1],cases,VERSION,experiment)
        screening[experiment]=diagnostics(p,cases,0.5)
    baseline,_=collect(V4/'runs',[0,1],cases,'v4.0.0')
    screening['v4']=diagnostics(baseline,cases,0.5)
    winner=max(('A','B'),key=lambda k:(screening[k]['mean_dice'],screening[k]['thin_skeleton_recall']))
    write_json(ROOT/'screening.json',{'folds':[0,1],'selection_rule':'D4 fixed 0.50 mean Dice; tie thin recall',
                                     'selected_experiment':winner,'results':screening})
    status('screening_complete',selected_experiment=winner,
           dice={k:v['mean_dice'] for k,v in screening.items()})
    for fold in (2,3,4):
        run_fold(winner,fold)
    status('oof_selection',selected_experiment=winner)
    candidate,models=collect(ROOT/'experiments'/winner,range(5),cases,VERSION,winner)
    baseline,v4_models=collect(V4/'runs',range(5),cases,'v4.0.0')
    oof=ROOT/'oof';oof.mkdir(exist_ok=True)
    np.savez_compressed(oof/'candidate_probabilities.npz',**candidate)
    # Limit the blend search; select weights at fixed threshold, then one common threshold.
    weights=[0,0.25,0.5,0.75,1]
    weight_table=[{'candidate_weight':w,'mean_dice':summarize(blend(candidate,baseline,w),cases,0.5,False)['mean_dice']} for w in weights]
    chosen=max(weight_table,key=lambda r:(r['mean_dice'],-abs(r['candidate_weight']-0.5)))
    weight=chosen['candidate_weight']
    combined=blend(candidate,baseline,weight)
    threshold_table=[{'threshold':float(t),'mean_dice':summarize(combined,cases,float(t),False)['mean_dice']}
                     for t in np.round(np.arange(0.40,0.601,0.02),2)]
    threshold=max(threshold_table,key=lambda r:(r['mean_dice'],-abs(r['threshold']-0.5)))['threshold']
    # Leave-one-fold-out calibration assesses weight/threshold tuning sensitivity.
    calibration=[]
    for fold,split in enumerate(make_folds(cases)):
        train_ids=split['train_ids'];val_ids=split['val_ids']
        options=[]
        for w in weights:
            p=blend(candidate,baseline,w)
            for t in np.round(np.arange(0.40,0.601,0.02),2):
                score=summarize({k:p[k] for k in train_ids},cases,float(t),False)['mean_dice']
                options.append((score,-abs(float(t)-0.5),-abs(w-0.5),w,float(t)))
        *_,w,t=max(options)
        p=blend(candidate,baseline,w)
        calibration.append({'fold':fold,'candidate_weight':w,'threshold':t,
                            'heldout':diagnostics({k:p[k] for k in val_ids},cases,t)})
    report={'version':VERSION,'selected_experiment':winner,'tta':'d4',
            'candidate_weight':weight,'v4_weight':1-weight,'threshold':threshold,
            'weight_table_fixed_0_5':weight_table,'threshold_table':threshold_table,
            'v4_fixed_0_5':diagnostics(baseline,cases,0.5),
            'candidate_fixed_0_5':diagnostics(candidate,cases,0.5),
            'ensemble_fixed_0_5':diagnostics(combined,cases,0.5),
            'ensemble_selected':diagnostics(combined,cases,threshold),
            'v4_original_selected':diagnostics(baseline,cases,0.46),
            'cross_fitted_calibration':calibration,
            'caveat':'OOF model, checkpoint and experiment selection reuse these 20 labels; calibration cross-fitting does not undo architecture selection or create an independent test score.',
            'models':models,'v4_models':v4_models}
    write_json(oof/'selection.json',report)
    np.savez_compressed(oof/'ensemble_probabilities.npz',**combined)
    save_preview({k:combined[k] for k in ('1','7','12','19')},cases,oof/'preview.png',threshold)
    test=load_cases(DATA,'test')
    assert len(test)==20
    candidate_test=predict(models,test,winner)
    baseline_test=predict(v4_models,test,None)
    final=blend(candidate_test,baseline_test,weight)
    submission=ROOT/'submission';submission.mkdir(exist_ok=True)
    np.savez_compressed(submission/'test_probabilities.npz',**final)
    metadata=export_submission(final,submission,DATA/'sample_submission.csv',threshold,
                               polarity='vessel-white',resize_mode='official',official_script=DATA/'segmentation_to_csv.py')
    shutil.copy2(submission/'submission.csv',ROOT/'submission_v5.0.0.csv')
    # Always provide the explicitly requested equal blend as a separately labeled alternative.
    equal=ROOT/'submission_equal_blend'
    export_submission(blend(candidate_test,baseline_test,0.5),equal,DATA/'sample_submission.csv',threshold,
                      polarity='vessel-white',resize_mode='official',official_script=DATA/'segmentation_to_csv.py')
    status('final_audit')
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    with (ROOT/'submission_v5.0.0.csv').open(newline='') as f:
        reader=csv.DictReader(f);check(reader.fieldnames==['Id','Predicted'],'CSV header');rows=list(reader)
    with (DATA/'sample_submission.csv').open(newline='') as f:
        expected=[r['Id'] for r in csv.DictReader(f)]
    check([r['Id'] for r in rows]==expected,'20 IDs and sample order')
    for row in rows:
        values=list(map(int,row['Predicted'].split()))
        check(len(values)%2==0,f'{row["Id"]}: RLE pairs')
        flat=np.zeros(512*512,np.uint8);previous_end=0
        for start,length in zip(values[::2],values[1::2]):
            check(start>=1 and length>0 and start>previous_end and start+length-1<=flat.size,f'{row["Id"]}: run {start}')
            flat[start-1:start-1+length]=1;previous_end=start+length-1
        with Image.open(submission/'image'/f'{row["Id"]}.png') as im:
            original=np.asarray(im)
            check(set(np.unique(original))<={0,255},f'{row["Id"]}: binary PNG')
            encoded=np.asarray(im.convert('L').resize((512,512),Image.Resampling.BILINEAR))>0
        check(np.array_equal(flat.reshape(512,512),encoded),f'{row["Id"]}: independent decode equals official resize')
        check(np.array_equal(original>0,final[row['Id']]>=threshold),f'{row["Id"]}: threshold match')
    for path,digest in previous.items():
        check(sha(path)==digest,f'preserved: {path}')
    for path,digest in input_snapshot.items():
        check(sha(path)==digest,f'input preserved: {path}')
    check(metadata['official_script_verified'] and metadata['roundtrip_verified'],'official converter and roundtrip')
    write_json(ROOT/'audit.json',{'passed':True,'checks':len(checks),'previous_files_verified':len(previous),
                                 'submission_sha256':sha(ROOT/'submission_v5.0.0.csv'),
                                 'training_only_supplied_data':True,'details':checks})
    results=(f'# v5.0.0 results\n\nSelected experiment: {winner}. All new files on D; all prior versions preserved.\n\n'
             f'A screening Dice: {screening["A"]["mean_dice"]:.6f}; B: {screening["B"]["mean_dice"]:.6f}.\n\n'
             f'Full OOF candidate Dice at 0.5: {report["candidate_fixed_0_5"]["mean_dice"]:.6f}. '
             f'v4: {report["v4_fixed_0_5"]["mean_dice"]:.6f}.\n\n'
             f'Final candidate/v4 weights: {weight}/{1-weight}; threshold {threshold}; '
             f'tuned OOF Dice {report["ensemble_selected"]["mean_dice"]:.6f}.\n\n'
             'submission_v5.0.0.csv is the OOF-selected primary file. submission_equal_blend/submission.csv '
             'is the 50:50 alternative at the primary threshold, not separately optimized.\n\n'
             'These are selection-biased validation scores, not Kaggle scores. Thin skeleton is a digital-label proxy. '
             'E/IterNet informs attention to continuity only; no iterative architecture is claimed or added to A/B.\n\n'
             f'Independent audit passed {len(checks)} checks. No upload performed.\n')
    (ROOT/'RESULTS.md').write_text(results,encoding='utf-8')
    archive=ROOT.parent/'vessel_v5.0.0_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in ('.py','.ps1','.md','.txt'):
                z.write(p,p.name)
    manifest={'version':VERSION,'status':'completed','directory':str(ROOT),
              'submission':{'path':str(ROOT/'submission_v5.0.0.csv'),'sha256':sha(ROOT/'submission_v5.0.0.csv')},
              'source_archive':{'path':str(archive),'sha256':sha(archive)},
              'selected_experiment':winner,'oof_dice':report['ensemble_selected']['mean_dice'],
              'parents':['v4.0.0'],'all_new_files_on_D':True,'external_training_data':False}
    write_json(ROOT/'version_manifest.json',manifest)
    write_json(ROOT.parent/'versions'/f'{VERSION}.json',manifest)
    status('completed',**manifest)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
