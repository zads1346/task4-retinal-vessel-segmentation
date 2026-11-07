"""Average all selected fold models on test images, using the OOF-selected rule."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import load_cases
from pretrained_model import PretrainedUNet
from inference import predict_image, write_json
from submission import export_submission


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Use the configured CUDA Python')
    torch.set_num_threads(4)
    selection=json.loads(a.selection.read_text())
    models=selection['models']
    if selection['version']!='v3.0.0' or not models:raise ValueError('Unexpected selection')
    if not isinstance(selection['tta'],bool) or not 0<float(selection['threshold'])<1:raise ValueError('Invalid inference configuration')
    ids={str(i) for i in range(1,21)}
    if set(m['family'] for m in models)!=set(selection['family_weights']):raise ValueError('Unknown or missing model family')
    if any(not np.isfinite(m['test_prediction_weight']) or m['test_prediction_weight']<=0 for m in models):
        raise ValueError('Model weights must be positive finite numbers')
    if abs(sum(m['test_prediction_weight'] for m in models)-1)>1e-8:raise ValueError('Weights do not sum to one')
    partition=None
    for family,weight in selection['family_weights'].items():
        members=[m for m in models if m['family']==family]
        if sorted(m['fold'] for m in members)!=list(range(5)):raise ValueError('Each family must have all five folds')
        if any(abs(m['test_prediction_weight']-weight/5)>1e-8 for m in members):raise ValueError('Unequal fold weights')
        for m in members:
            train,valid=m['train_ids'],m['val_ids']
            if len(train)!=16 or len(set(train))!=16 or len(valid)!=4 or len(set(valid))!=4:
                raise ValueError('Invalid 16/4 fold membership')
            if set(train)&set(valid) or set(train)|set(valid)!=ids:raise ValueError('Fold training/validation overlap or incomplete data')
        held_out=[i for m in members for i in m['val_ids']]
        if len(held_out)!=20 or set(held_out)!=ids:raise ValueError('Validation folds must partition all 20 images')
        current=[sorted(m['val_ids']) for m in sorted(members,key=lambda m:m['fold'])]
        if partition is not None and partition!=current:raise ValueError('Families use inconsistent fold assignments')
        partition=current
    cases=load_cases(a.data,'test')
    input_hashes={i:hashlib.sha256(case['image'].tobytes()).hexdigest() for i,case in cases.items()}
    a.output.mkdir(parents=True,exist_ok=True)
    cache=a.output/'model_probabilities';cache.mkdir(exist_ok=True)
    total={i:np.zeros(case['image'].shape[:2],np.float32) for i,case in cases.items()}
    model_records=[]
    for item in models:
        checkpoint=Path(item['checkpoint'])
        if sha(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Checkpoint changed since OOF selection')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=state['config']
        if config['version']!='v3.0.0' or config['encoder_name']!=item['family'] or config['fold']!=item['fold']:
            raise ValueError('Checkpoint family/version/fold differs from OOF selection')
        if config['val_ids']!=item['val_ids'] or config['train_ids']!=item['train_ids']:
            raise ValueError('Checkpoint split differs from OOF selection')
        if config['pretrained_weights'] is not True or config['external_fine_tuning_data'] is not False:
            raise ValueError('Unexpected fine-tuning provenance')
        name=f"{item['family']}_fold{item['fold']}"
        probability_file=cache/f'{name}.npz';metadata_file=cache/f'{name}.json'
        cache_key={'checkpoint_sha256':item['checkpoint_sha256'],'tta':selection['tta'],'ids':list(cases),
                   'test_input_sha256':input_hashes,'inference_code_sha256':sha(Path(__file__).with_name('inference.py')),
                   'model_code_sha256':sha(Path(__file__).with_name('pretrained_model.py')),
                   'batch_size':a.batch_size}
        cached=json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        reusable=(probability_file.exists() and all(cached.get(k)==v for k,v in cache_key.items())
                  and cached.get('probabilities_sha256')==sha(probability_file))
        if reusable:
            with np.load(probability_file,allow_pickle=False) as saved:
                probabilities={i:saved[i] for i in cases}
            print(f'Reused verified cache {name}',flush=True)
        else:
            model=PretrainedUNet(config['encoder_name'],weights_path=None).to('cuda')
            model.load_state_dict(state['model'])
            probabilities={}
            for i,case in cases.items():
                probabilities[i]=predict_image(model,case['image'],torch.device('cuda'),
                                  config['patch_size'],config['patch_size']//2,a.batch_size,tta=selection['tta'])
            np.savez_compressed(probability_file,**probabilities)
            write_json(metadata_file,{**cache_key,'probabilities_sha256':sha(probability_file)})
            del model
            torch.cuda.empty_cache()
            print(f'Predicted {name}, 20 test images',flush=True)
        for i,probability in probabilities.items():
            if probability.shape!=total[i].shape or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                raise ValueError('Invalid cached test probability')
            total[i]+=probability*item['test_prediction_weight']
        model_records.append({**item,'probabilities':str(probability_file.resolve()),'probabilities_sha256':sha(probability_file)})
        del state
    np.savez_compressed(a.output/'test_probabilities.npz',**total)
    metadata=export_submission(total,a.output,a.data/'sample_submission.csv',selection['threshold'],
                               polarity='vessel-white',resize_mode='official',
                               official_script=a.data/'segmentation_to_csv.py')
    write_json(a.output/'prediction_run.json',{'version':'v3.0.0','selection_file':str(a.selection.resolve()),
               'selection_sha256':sha(a.selection),'selected_variant':selection['selected_variant'],
               'models':model_records,'tta':selection['tta'],'threshold':selection['threshold'],
               'supervised_fine_tuning_data':'Only the provided 20 image/label pairs',
               'pretrained_initialization':'Official ImageNet-1K encoder weights',
               'submission_checks':metadata,'uploaded_to_kaggle':False})
    print('Verified ensemble CSV: '+str((a.output/'submission.csv').resolve()),flush=True)


if __name__=='__main__':main()
