"""Post-run audit independently reconstructing OOF and test blending."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import load_cases, make_folds
from inference import summarize


def main():
    root=Path(__file__).resolve().parent
    project=root.parent.parent
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    selection=read(root/'oof/selection.json')
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    cases=load_cases(project/'work/data','train')
    folds=make_folds(cases)
    oof_families=[]
    for family,records in [('new',selection['models']),('v4',selection['v4_models'])]:
        check(len(records)==5,f'{family} 5 folds')
        probabilities={}
        for record in records:
            fold=record['fold']
            config=record['config']
            check(config['train_ids']==folds[fold]['train_ids'],f'{family}/{fold} training IDs')
            check(config['val_ids']==folds[fold]['val_ids'],f'{family}/{fold} validation IDs')
            check(config['steps']==6000,f'{family}/{fold} training budget')
            check(not config['external_training_data'] and not config['pretrained_weights'],f'{family}/{fold} no external data')
            checkpoint=Path(record['checkpoint'])
            check(sha(checkpoint)==record['checkpoint_sha256'],f'{family}/{fold} checkpoint digest')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            check(state['config']==config,f'{family}/{fold} embedded config')
            del state
            code_root=root if family=='new' else root.parent/'vessel_v4'
            for filename,digest in config['source_sha256'].items():
                check(sha(code_root/filename)==digest,f'{family}/{fold} source {filename}')
            folder=checkpoint.parent
            report=read(folder/'validation_report.json')
            check(report['completed_steps']==6000,f'{family}/{fold} completed budget')
            check(sha(record['oof_probabilities'])==record['oof_sha256'],f'{family}/{fold} OOF digest')
            with np.load(record['oof_probabilities'],allow_pickle=False) as data:
                check(set(data.files)==set(config['val_ids']),f'{family}/{fold} OOF heldout-only')
                for key in data.files:
                    check(key not in probabilities,f'{family}/{key} unique heldout image')
                    probabilities[key]=data[key]
        check(set(probabilities)==set(cases),f'{family} OOF coverage')
        oof_families.append(probabilities)
    weight=selection['candidate_weight']
    combined={k:np.clip(weight*oof_families[0][k]+(1-weight)*oof_families[1][k],0,1) for k in cases}
    with np.load(root/'oof/ensemble_probabilities.npz',allow_pickle=False) as saved:
        for key,p in combined.items():
            check(np.array_equal(p,saved[key]),f'{key} OOF weighted probability')
    metric=summarize(combined,cases,selection['threshold'],False)
    check(abs(metric['mean_dice']-selection['ensemble_selected']['mean_dice'])<1e-12,'recomputed OOF Dice')
    test_families=[]
    for name in (selection['selected_experiment'],'v4'):
        total={}
        for fold in range(5):
            path=root/'test_model_probabilities'/f'{name}_fold{fold}.npz'
            metadata=read(path.with_suffix('.json'))
            check(sha(path)==metadata['sha256'],f'{name}/{fold} test cache digest')
            with np.load(path,allow_pickle=False) as saved:
                check(set(saved.files)==set(cases),f'{name}/{fold} 20 test IDs')
                for key in saved.files:
                    p=saved[key]
                    check(np.isfinite(p).all() and p.min()>=0 and p.max()<=1,f'{name}/{fold}/{key} finite probabilities')
                    if key not in total:total[key]=np.zeros_like(p)
                    total[key]+=p/5
        test_families.append(total)
    with np.load(root/'submission/test_probabilities.npz',allow_pickle=False) as saved:
        for key in cases:
            expected=np.clip(weight*test_families[0][key]+(1-weight)*test_families[1][key],0,1)
            check(np.array_equal(expected,saved[key]),f'{key} final test weighting')
    manifest=read(root/'version_manifest.json')
    check(sha(root/'submission_v5.0.0.csv')==manifest['submission']['sha256'],'primary CSV digest')
    for path,digest in read(root/'previous_versions_snapshot.json').items():
        check(sha(path)==digest,f'previous preserved: {path}')
    result={'passed':True,'checks':len(checks),'details':checks}
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks)}))


if __name__=='__main__':main()
