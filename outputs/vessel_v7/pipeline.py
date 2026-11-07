"""v7: five-fold adapter-strength comparison followed by two short LR runs."""
import hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from data import load_cases,make_folds
from model import UNet
from inference import predict_image,summarize,save_preview,write_json
from lora import inject,load_adapter,merged
from scaling import scaled_state
from common import ROOT,PROJECT,DATA,read,sha,status,diagnostics,select,predict_all,audit_csv
from submission import export_submission

V4=ROOT.parent/'vessel_v4'
V6=ROOT.parent/'vessel_v6'
BETAS=(0.,.25,.5,.75,1.)
RATES=(('lr_1e-5',1e-5),('lr_3e-5',3e-5))
TOLERANCE=1e-5


def capture_inputs():
    path=ROOT/'previous_versions_snapshot.json'
    if path.exists():previous=read(path)
    else:
        paths=[]
        for name in ('vessel_baseline','vessel_v2','vessel_v3','vessel_v4','vessel_v5','vessel_v6'):
            paths.extend(p for p in (ROOT.parent/name).rglob('*') if p.is_file())
        paths.extend(p for p in (ROOT.parent/'versions').glob('*.json') if p.stem!='v7.0.0')
        previous={str(p):sha(p) for p in paths};write_json(path,previous)
    input_path=ROOT/'input_snapshot.json'
    inputs={str(p):sha(p) for p in DATA.rglob('*') if p.is_file()}
    if input_path.exists():assert inputs==read(input_path),'Input data changed'
    else:write_json(input_path,inputs)
    return previous,inputs


def load_probabilities(path,ids,cases):
    with np.load(path,allow_pickle=False) as f:
        assert set(f.files)==set(ids)
        result={i:f[i] for i in ids}
    for i,p in result.items():
        assert p.shape==cases[i]['mask'].shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    return result


def evaluate(checkpoint,ids,cases):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=UNet(24).cuda();model.load_state_dict(saved['model'])
    result={i:predict_image(model,cases[i]['image'],torch.device('cuda'),256,128,8,tta='d4') for i in ids}
    del saved,model;torch.cuda.empty_cache()
    return result


def add_candidate(name,records,probabilities,cases,extra):
    folder=ROOT/'oof';folder.mkdir(exist_ok=True)
    path=folder/f'{name}.npz'
    np.savez_compressed(path,**probabilities)
    return {'name':name,'models':records,'oof_path':str(path),'oof_sha256':sha(path),
            'fixed_0_5':diagnostics(probabilities,cases,.5),**extra}


def scaling_stage(cases,folds):
    v6_reference={r['fold']:r for r in read(V6/'oof/selection.json')['models']}
    candidates=[]
    for beta in BETAS:
        name=f'scale_{round(beta*100):03d}';records=[];probabilities={}
        for fold,split in enumerate(folds):
            status('scaling_oof',beta=beta,fold=fold)
            base_path=V4/'runs'/f'fold{fold}'/'best.pt'
            adapted_path=V6/'runs'/f'fold{fold}'/'best_merged.pt'
            base=torch.load(base_path,map_location='cpu',weights_only=False)
            adapted=torch.load(adapted_path,map_location='cpu',weights_only=False)
            assert sha(adapted_path)==v6_reference[fold]['sha256']
            assert sha(base_path)==adapted['config']['base_sha256']
            for key in ('train_ids','val_ids'):
                assert base['config'][key]==split[key] and adapted['config'][key]==split[key]
            assert not base['config']['external_training_data'] and not adapted['config']['external_training_data']
            state=scaled_state(base['model'],adapted['model'],beta)
            folder=ROOT/'scaling'/name/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
            if beta==0:
                checkpoint=base_path
                p=load_probabilities(base_path.parent/'validation_d4_probabilities.npz',split['val_ids'],cases)
            elif beta==1:
                checkpoint=adapted_path
                p=load_probabilities(adapted_path.parent/'validation_d4_probabilities.npz',split['val_ids'],cases)
            else:
                checkpoint=folder/'model.pt'
                config={'version':'v7.0.0','method':'parameter_delta_scaling','beta':beta,'fold':fold,**split,
                        'base_sha256':sha(base_path),'v6_merged_sha256':sha(adapted_path)}
                if checkpoint.exists():
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    assert saved['config']==config
                    for k,v in state.items():assert torch.equal(v,saved['model'][k])
                    del saved
                else:torch.save({'model':state,'config':config},checkpoint)
                cache=folder/'validation_d4_probabilities.npz';meta=folder/'cache.json'
                key={'checkpoint_sha256':sha(checkpoint),'inference_sha256':sha(ROOT/'inference.py'),
                     'ids':split['val_ids'],'images':{i:hashlib.sha256(cases[i]['image'].tobytes()).hexdigest() for i in split['val_ids']}}
                if cache.exists() and meta.exists() and read(meta).get('key')==key and read(meta).get('sha256')==sha(cache):
                    p=load_probabilities(cache,split['val_ids'],cases)
                else:
                    p=evaluate(checkpoint,split['val_ids'],cases)
                    np.savez_compressed(cache,**p);write_json(meta,{'key':key,'sha256':sha(cache)})
            probabilities.update(p)
            records.append({'family':name,'fold':fold,'checkpoint':str(checkpoint),'sha256':sha(checkpoint),
                            'base_checkpoint':str(base_path),'base_sha256':sha(base_path),'v6_checkpoint':str(adapted_path),
                            'v6_sha256':sha(adapted_path),'beta':beta,**split})
            del state,base,adapted
        candidates.append(add_candidate(name,records,probabilities,cases,{'method':'parameter_scaling','beta':beta}))
    best=max(c['fixed_0_5']['mean_dice'] for c in candidates)
    chosen=next(c for c in candidates if c['fixed_0_5']['mean_dice']>=best-TOLERANCE)
    write_json(ROOT/'scaling_comparison.json',{'selected':chosen['name'],'tie_tolerance':TOLERANCE,'candidates':candidates})
    status('scaling_complete',selected=chosen['name'],scores={c['name']:c['fixed_0_5']['mean_dice'] for c in candidates})
    return candidates


def train_short(name,lr,fold):
    folder=ROOT/'runs'/name/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'validation_report.json').exists():return
    status('training',experiment=name,lr=lr,fold=fold,total_batches=300,run_dir=str(folder))
    cmd=[sys.executable,'-u',str(ROOT/'train_lora.py'),'--fold',str(fold),'--lr',str(lr),'--output',str(folder)]
    if (folder/'last_adapter.pt').exists():cmd+=['--resume']
    with (folder/'console.log').open('a',encoding='utf-8') as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def collect_short(name,lr,cases,folds):
    records=[];probabilities={};steps=[];audits=[]
    for fold,split in enumerate(folds):
        folder=ROOT/'runs'/name/f'fold{fold}'
        cfg=read(folder/'config.json');report=read(folder/'validation_report.json')
        assert cfg['lr']==lr and cfg['version']=='v7.0.0' and cfg['steps']==300
        assert report['completed_steps']==300 and report['base_frozen_exact'] and report['adapter_merge_verified']
        assert cfg['selection_tta']=='d4' and cfg['selection_includes_step0']
        for k in ('train_ids','val_ids'):assert cfg[k]==split[k]
        for f,h in cfg['source_sha256'].items():assert sha(ROOT/f)==h
        assert sha(cfg['base_checkpoint'])==cfg['base_sha256']
        curve=[json.loads(line) for line in (folder/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
        assert set(r['step'] for r in curve)==set(range(0,301,25))
        expected_score,expected_step=curve[0]['mean_dice'],0
        for row in curve[1:]:
            if row['mean_dice']>expected_score+TOLERANCE:expected_score,expected_step=row['mean_dice'],row['step']
        best=torch.load(folder/'best_adapter.pt',map_location='cpu',weights_only=False)
        saved=torch.load(folder/'best_merged.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'last_adapter.pt',map_location='cpu',weights_only=False)
        assert last['step']==300 and 'optimizer' in last
        assert best['step']==expected_step==report['best_step'] and abs(best['score']-expected_score)<1e-12
        assert saved['config']==cfg==best['config']
        base=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=False)
        model=UNet(24);model.load_state_dict(base['model']);inject(model)
        load_adapter(model,best['adapter'])
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            rebuilt=merged(model.cuda()).state_dict()
            for k,v in saved['model'].items():assert torch.equal(v,rebuilt[k].cpu()),k
        finally:torch.backends.cuda.matmul.allow_tf32=old_tf32
        p=load_probabilities(folder/'validation_d4_probabilities.npz',split['val_ids'],cases)
        recomputed=summarize(p,cases,.5,False)['mean_dice']
        assert abs(recomputed-report['d4_at_0_5']['mean_dice'])<1e-12
        if expected_step==0:
            for k,v in saved['model'].items():assert torch.equal(v,base['model'][k])
            # Canonical v4 probabilities prevent a numerically re-evaluated
            # zero adapter being mislabeled as a fine-tuning improvement.
            p=load_probabilities(Path(cfg['base_checkpoint']).parent/'validation_d4_probabilities.npz',split['val_ids'],cases)
        probabilities.update(p);steps.append(expected_step)
        audits.append({'fold':fold,'selected_step':expected_step,'completed_steps':300,
                       'selected_from_d4_with_step0':True,'merged_gpu_tf32_bitwise_equal':True,'backbone_frozen_exact':True})
        records.append({'family':name,'fold':fold,'checkpoint':str(folder/'best_merged.pt'),'sha256':sha(folder/'best_merged.pt'),
                        'adapter':str(folder/'best_adapter.pt'),'adapter_sha256':sha(folder/'best_adapter.pt'),
                        'base_checkpoint':cfg['base_checkpoint'],'base_sha256':cfg['base_sha256'],'step':expected_step,**split})
        del best,saved,last,base,model,rebuilt;torch.cuda.empty_cache()
    write_json(ROOT/'runs'/name/'selection_audit.json',audits)
    return add_candidate(name,records,probabilities,cases,{'method':'short_lora','lr':lr,'best_steps':steps,'base_only_folds':steps.count(0)})


def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and ROOT.drive.upper()=='D:'
    previous,inputs=capture_inputs()
    plan={'version':'v7.0.0','betas':list(BETAS),'rates':[r for _,r in RATES],'folds':5,'steps':300,'eval_every':25,
          'checkpoint_tta':'d4','checkpoint_threshold':.5,'include_step0':True,'min_delta':TOLERANCE,
          'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')}}
    if (ROOT/'plan.json').exists():assert read(ROOT/'plan.json')==plan,'Protocol or source changed'
    else:write_json(ROOT/'plan.json',plan)
    cases=load_cases(DATA,'train');assert len(cases)==20
    folds=make_folds(cases)
    candidates=scaling_stage(cases,folds)
    for name,lr in RATES:
        for fold in range(5):train_short(name,lr,fold)
        candidates.append(collect_short(name,lr,cases,folds))
        write_json(ROOT/f'{name}_comparison.json',candidates[-1])
        status('short_run_complete',experiment=name,dice=candidates[-1]['fixed_0_5']['mean_dice'],best_steps=candidates[-1]['best_steps'])
    status('final_oof_selection')
    best=max(c['fixed_0_5']['mean_dice'] for c in candidates)
    winner=next(c for c in candidates if c['fixed_0_5']['mean_dice']>=best-TOLERANCE)
    selected=load_probabilities(winner['oof_path'],cases,cases)
    threshold,threshold_table=select(selected,cases)
    selection={'version':'v7.0.0','winner':winner['name'],'threshold':threshold,'tta':'d4','tie_tolerance':TOLERANCE,
               'candidates':candidates,'threshold_table':threshold_table,'selected':diagnostics(selected,cases,threshold),
               'models':winner['models'],'warning':'Checkpoint, method and threshold selection reuse the 20 labels. These are selection-biased OOF scores, not an independent test or Kaggle score.'}
    write_json(ROOT/'oof/selection.json',selection)
    save_preview({i:selected[i] for i in ('1','7','12','19')},cases,ROOT/'oof/preview.png',threshold)
    test=load_cases(DATA,'test');assert len(test)==20
    final=predict_all(test,winner['models'])
    folder=ROOT/'submission';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'test_probabilities.npz',**final)
    meta=export_submission(final,folder,DATA/'sample_submission.csv',threshold,polarity='vessel-white',resize_mode='official',official_script=DATA/'segmentation_to_csv.py')
    assert meta['official_script_verified'] and meta['roundtrip_verified'];audit_csv(folder,final,threshold)
    shutil.copy2(folder/'submission.csv',ROOT/'submission_v7.0.0.csv')
    status('final_audit')
    for p,h in previous.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    for f,h in plan['source_sha256'].items():assert sha(ROOT/f)==h,f
    for c in candidates:
        assert sha(c['oof_path'])==c['oof_sha256']
        for record in c['models']:assert sha(record['checkpoint'])==record['sha256']
    audit={'passed':True,'previous_files_unchanged':len(previous),'input_files_unchanged':len(inputs),
           'scaling_candidates':5,'short_training_runs':10,'all_five_fold_partitions_verified':True,
           'short_runs_backbones_frozen_and_merges_bitwise_verified':True,'checkpoint_selection_replayed':True,
           'official_script_verified':True,'independent_RLE_pixel_roundtrip_images':20,'all_sources_unchanged':True}
    write_json(ROOT/'audit.json',audit)
    table='\n'.join(f'| {c["name"]} | {c["fixed_0_5"]["mean_dice"]:.8f} | {c["fixed_0_5"]["mean_precision"]:.6f} | {c["fixed_0_5"]["mean_recall"]:.6f} | {c["fixed_0_5"]["thin_skeleton_recall"]:.6f} |' for c in candidates)
    steps='\n'.join(f'- {c["name"]}: best steps {c["best_steps"]}; {c["base_only_folds"]}/5 folds retained step0.' for c in candidates if c['method']=='short_lora')
    report=(f'# v7.0.0 results\n\nOnly the provided 20 labeled training images. No external data or external pretrained weights. All prior versions preserved.\n\n'
            'Stage1: scale the saved v6 parameter update relative to its corresponding v4 fold: W(beta)=W_v4+beta*(W_v6-W_v4). '
            'Beta grid0/0.25/0.5/0.75/1; this is parameter scaling, not probability blending. Endpoints reuse verified existing held-out probabilities.\n\n'
            'Stage2: each rate1e-5 and3e-5 runs all5 folds from the matching v4 checkpoint with newly initialized rank4/alpha4 decoder adapters; '
            '54,720 trainable parameters, 4,416,673 frozen. Train300 batches, batch8, native256 patches, AdamW weight decay1e-4, warmup30, cosine decay, AMP, gradient clip1. '
            'Unchanged v4 RGB, sampling, augmentation and loss. Sample seed12026+fold; split seed2026. No extra EMA.\n\n'
            'Evaluate D4 at step0 and every25 steps at threshold0.5; include step0 in selection; require improvement exceeding1e-5. '
            'Select the best complete OOF candidate at fixed0.5; scores within1e-5 prefer the earlier/simpler candidate. Then select a global threshold from0.40:0.02:0.60.\n\n'
            '| Candidate | Fixed0.5 Dice | Precision | Recall | Thin skeleton recall |\n|---|---:|---:|---:|---:|\n'+table+
            '\n\nSelected checkpoint steps:\n'+steps+
            f'\n\nWinner: {winner["name"]}; threshold {threshold}; selected OOF Dice {selection["selected"]["mean_dice"]:.8f}. '
            'Final submission_v7.0.0.csv averages the selected family\'s5 folds with D4 on20 test images.\n\n'
            'Selecting checkpoints on each validation fold guarantees a favorable in-sample comparison when step0 is available; it does not establish independent generalization. '
            'All scores reuse labels for checkpoint/method/threshold choice, and are not Kaggle scores. Step0-retained folds are not counted as adaptation improvements.\n\n'
            f'All10 training runs completed. Prior {len(previous)} files unchanged. Source hashes, fold isolation, frozen backbones, matching-CUDA merged weights, '
            'official conversion and independent20-image RLE decoding verified. White-vessel encoding follows supplied labels and original converter; conflicting prose remains unconfirmed by hidden scoring. '
            'No Kaggle upload. All new files on D.\n')
    (ROOT/'RESULTS.md').write_text(report,encoding='utf-8')
    manifest={'version':'v7.0.0','status':'completed','directory':str(ROOT),'parents':['v4.0.0','v6.0.0'],
              'winner':winner['name'],'threshold':threshold,'fixed_0_5_dice':winner['fixed_0_5']['mean_dice'],
              'selected_oof_dice':selection['selected']['mean_dice'],
              'submission':{'path':str(ROOT/'submission_v7.0.0.csv'),'sha256':sha(ROOT/'submission_v7.0.0.csv')},'audit':audit}
    write_json(ROOT/'version_manifest.json',manifest);write_json(ROOT.parent/'versions/v7.0.0.json',manifest)
    status('completed',**manifest)


if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
