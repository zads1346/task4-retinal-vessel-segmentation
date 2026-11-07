"""Five paired-fold LoRA runs, OOF selection, test ensemble and release audit."""
import csv,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from data import load_cases,make_folds
from inference import predict_image,summarize,write_json,save_preview
from model import UNet
from lora import inject,load_adapter,merged
from submission import export_submission

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
DATA=PROJECT/'work/data'
V4=PROJECT/'outputs/vessel_v4'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def status(stage,**extra):
    row={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**extra}
    write_json(ROOT/'status.json',row);print(json.dumps(row),flush=True)


def snapshot():
    path=ROOT/'previous_versions_snapshot.json'
    if path.exists():return read(path)
    paths=[]
    for name in ('vessel_baseline','vessel_v2','vessel_v3','vessel_v4','vessel_v5'):
        paths.extend(p for p in (ROOT.parent/name).rglob('*') if p.is_file())
    result={str(p):sha(p) for p in paths}
    write_json(path,result)
    return result


def run_fold(fold):
    out=ROOT/'runs'/f'fold{fold}';out.mkdir(parents=True,exist_ok=True)
    if (out/'validation_report.json').exists():return
    status('training',fold=fold,total_batches=1500)
    cmd=[sys.executable,'-u',str(ROOT/'train_lora.py'),'--fold',str(fold),'--output',str(out)]
    if (out/'last_adapter.pt').exists():cmd+=['--resume']
    with (out/'console.log').open('a',encoding='utf-8') as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def diagnostics(p,cases,t):
    result=summarize(p,cases,t,False)
    thin=[]
    for k,v in p.items():
        y=cases[k]['mask']>0.5;s=skeletonize(y)&(distance_transform_edt(y)<=2)
        thin.append(float(((v>=t)&s).sum()/max(1,s.sum())))
    result['thin_skeleton_recall']=float(np.mean(thin))
    result['mean_precision']=float(np.mean([r['precision'] for r in result['per_image']]))
    result['mean_recall']=float(np.mean([r['recall'] for r in result['per_image']]))
    return result


def blend(new,base,w):return {k:np.clip(w*new[k]+(1-w)*base[k],0,1) for k in new}


def select(p,cases):
    table=[{'threshold':float(t),'mean_dice':summarize(p,cases,float(t),False)['mean_dice']}
           for t in np.round(np.arange(.4,.601,.02),2)]
    return max(table,key=lambda r:(r['mean_dice'],-abs(r['threshold']-.5)))['threshold'],table


def predict_all(cases,records):
    cache=ROOT/'test_model_probabilities';cache.mkdir(exist_ok=True)
    total={i:np.zeros(v['image'].shape[:2],np.float32) for i,v in cases.items()}
    for record in records:
        name=record['family']+f'_fold{record["fold"]}'
        status('test_prediction',model=name)
        checkpoint=Path(record['checkpoint']);assert sha(checkpoint)==record['sha256']
        path=cache/(name+'.npz');meta=path.with_suffix('.json')
        key={'checkpoint_sha256':record['sha256'],'tta':'d4','inference_sha256':sha(ROOT/'inference.py'),
             'images':{i:hashlib.sha256(v['image'].tobytes()).hexdigest() for i,v in cases.items()}}
        reuse=path.exists() and meta.exists() and read(meta).get('key')==key and read(meta).get('sha256')==sha(path)
        if reuse:
            with np.load(path,allow_pickle=False) as f:probabilities={i:f[i] for i in cases}
        else:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=UNet(24).cuda();model.load_state_dict(state['model'])
            probabilities={i:predict_image(model,v['image'],torch.device('cuda'),256,128,8,tta='d4') for i,v in cases.items()}
            np.savez_compressed(path,**probabilities);write_json(meta,{'key':key,'sha256':sha(path)})
            del model,state;torch.cuda.empty_cache()
        for i,p in probabilities.items():
            assert p.shape==total[i].shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            total[i]+=p/5
    return total


def audit_csv(folder,p,threshold):
    with (folder/'submission.csv').open(newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['Id','Predicted'];rows=list(reader)
    with (DATA/'sample_submission.csv').open(newline='') as f:ids=[r['Id'] for r in csv.DictReader(f)]
    assert [r['Id'] for r in rows]==ids and len(rows)==20
    for r in rows:
        values=list(map(int,r['Predicted'].split()));assert len(values)%2==0
        flat=np.zeros(512*512,bool);end=0
        for start,length in zip(values[::2],values[1::2]):
            assert start>=1 and length>0 and start>end and start+length-1<=len(flat)
            flat[start-1:start-1+length]=True;end=start+length-1
        with Image.open(folder/'image'/f'{r["Id"]}.png') as im:
            assert set(np.unique(im))<={0,255}
            assert np.array_equal(np.asarray(im)>0,p[r['Id']]>=threshold)
            expected=np.asarray(im.convert('L').resize((512,512),Image.Resampling.BILINEAR))>0
        assert np.array_equal(flat.reshape(512,512),expected)


def main():
    torch.set_num_threads(4)
    previous=snapshot()
    input_hashes={str(p):sha(p) for p in (DATA/'train').rglob('*') if p.is_file()}
    write_json(ROOT/'training_data_snapshot.json',input_hashes)
    for fold in range(5):run_fold(fold)
    status('oof_selection')
    cases=load_cases(DATA,'train');folds=make_folds(cases)
    candidate={};baseline={};records=[];base_records=[];fold_reports=[]
    for fold,split in enumerate(folds):
        folder=ROOT/'runs'/f'fold{fold}';cfg=read(folder/'config.json');report=read(folder/'validation_report.json')
        assert report['completed_steps']==1500 and report['base_frozen_exact'] and report['adapter_merge_verified']
        assert cfg['train_ids']==split['train_ids'] and cfg['val_ids']==split['val_ids']
        assert not set(cfg['train_ids'])&set(cfg['val_ids'])
        assert sha(cfg['base_checkpoint'])==cfg['base_sha256']
        for n,h in cfg['source_sha256'].items():assert sha(ROOT/n)==h
        # Reconstruct merged weights independently from the frozen base and compact adapter.
        state=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=False)
        adapted=UNet(24);adapted.load_state_dict(state['model']);inject(adapted)
        adapter=torch.load(folder/'best_adapter.pt',map_location='cpu',weights_only=False)
        assert adapter['config']==cfg
        load_adapter(adapted,adapter['adapter'])
        check=merged(adapted).state_dict()
        saved=torch.load(folder/'best_merged.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg
        cpu_max_difference=max(float((v-check[name]).abs().max()) for name,v in saved['model'].items())
        # Training explicitly enabled TF32 for BA on CUDA. Reconstruct with the
        # same arithmetic, then require bitwise equality, rather than silently
        # weakening the original audit to tolerate a different CPU BLAS result.
        previous_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            matched=merged(adapted.cuda()).state_dict()
            for name,v in saved['model'].items():
                torch.testing.assert_close(v,matched[name].cpu(),rtol=0,atol=0)
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous_tf32
        write_json(folder/'merge_reconstruction_audit.json',{
            'matched_cuda_tf32_bitwise_equal':True,'cpu_full_precision_max_abs_difference':cpu_max_difference,
            'explanation':'Training BA used CUDA TF32; the independent matching-arithmetic reconstruction is exact. CPU full-precision difference is recorded, not hidden.'})
        del adapted,state,adapter,saved,check,matched
        torch.cuda.empty_cache()
        for target,path in ((candidate,folder/'validation_d4_probabilities.npz'),(baseline,V4/'runs'/f'fold{fold}'/'validation_d4_probabilities.npz')):
            with np.load(path,allow_pickle=False) as f:
                assert set(f.files)==set(split['val_ids'])
                for i in f.files:
                    assert i not in target and f[i].shape==cases[i]['mask'].shape
                    assert np.isfinite(f[i]).all() and f[i].min()>=0 and f[i].max()<=1
                    target[i]=f[i]
        records.append({'family':'lora','fold':fold,'checkpoint':str(folder/'best_merged.pt'),'sha256':sha(folder/'best_merged.pt'),
                        'adapter_sha256':sha(folder/'best_adapter.pt'),'base_sha256':cfg['base_sha256'],**split})
        base_records.append({'family':'v4','fold':fold,'checkpoint':cfg['base_checkpoint'],'sha256':cfg['base_sha256'],**split})
        fold_reports.append({'fold':fold,'best_step':report['best_step'],'before':diagnostics({i:baseline[i] for i in split['val_ids']},cases,.5),
                             'after':diagnostics({i:candidate[i] for i in split['val_ids']},cases,.5)})
    weights=[{'lora_weight':w,'mean_dice':summarize(blend(candidate,baseline,w),cases,.5,False)['mean_dice']} for w in (0,.25,.5,.75,1)]
    # Prefer less adaptation on exact ties; retain the pure adapted result regardless.
    weight=max(weights,key=lambda r:(r['mean_dice'],-r['lora_weight']))['lora_weight']
    combined=blend(candidate,baseline,weight)
    threshold,table=select(combined,cases)
    lora_threshold,lora_table=select(candidate,cases)
    oof=ROOT/'oof';oof.mkdir(exist_ok=True)
    np.savez_compressed(oof/'lora_probabilities.npz',**candidate)
    np.savez_compressed(oof/'selected_probabilities.npz',**combined)
    result={'version':'v6.0.0','lora_weight':weight,'v4_weight':1-weight,'threshold':threshold,'lora_only_threshold':lora_threshold,
            'weight_table':weights,'threshold_table':table,'lora_threshold_table':lora_table,
            'baseline_fixed_0_5':diagnostics(baseline,cases,.5),'lora_fixed_0_5':diagnostics(candidate,cases,.5),
            'selected':diagnostics(combined,cases,threshold),'baseline_original_threshold':diagnostics(baseline,cases,.46),
            'folds':fold_reports,'models':records,'base_models':base_records,
            'caveat':'v4 and LoRA checkpoints were selected on the same validation folds. OOF is fold-isolated but selection-biased; not an independent test or Kaggle score.'}
    write_json(oof/'selection.json',result)
    save_preview({i:combined[i] for i in ('1','7','12','19')},cases,oof/'preview.png',threshold)
    test=load_cases(DATA,'test')
    new_test=predict_all(test,records);base_test=predict_all(test,base_records)
    final=blend(new_test,base_test,weight)
    for name,p,t in (('submission',final,threshold),('submission_lora',new_test,lora_threshold)):
        folder=ROOT/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'test_probabilities.npz',**p)
        meta=export_submission(p,folder,DATA/'sample_submission.csv',t,polarity='vessel-white',resize_mode='official',official_script=DATA/'segmentation_to_csv.py')
        assert meta['official_script_verified'] and meta['roundtrip_verified']
        audit_csv(folder,p,t)
    shutil.copy2(ROOT/'submission/submission.csv',ROOT/'submission_v6.0.0.csv')
    shutil.copy2(ROOT/'submission_lora/submission.csv',ROOT/'submission_v6.0.0_lora.csv')
    status('final_audit')
    for path,h in previous.items():assert sha(path)==h,path
    for path,h in input_hashes.items():assert sha(path)==h,path
    audit={'passed':True,'previous_files_unchanged':len(previous),'training_data_files_unchanged':len(input_hashes),
           'all_folds_isolated':True,'base_frozen_exact_all_folds':True,'all_merged_checkpoints_reconstructed_from_adapters':True,
           'both_submissions_20_image_independent_RLE_roundtrip':True,'external_training_data':False}
    write_json(ROOT/'audit.json',audit)
    report=(f'# v6.0.0：基于 v4 的卷积 LoRA 微调\n\n'
            '只使用提供的20张训练图。每折加载对应v4检查点，只用原16张训练图微调，4张验证图保持持出。保留所有旧版本。\n\n'
            '冻结原U-Net24全部4,416,673个参数，仅训练解码器8个卷积的rank4/alpha4低秩矩阵，共54,720参数。'
            '没有使用Transformer专用LoRA接口；对卷积权重按[out,in*k*k]展开应用BA增量。'
            'B零初始化；A采用Kaiming初始化；推理时合并回普通卷积。\n\n'
            '每折1500步，batch8，原生256裁块，AdamW学习率1e-4、权重衰减1e-4，100步预热后余弦衰减，AMP，梯度裁剪1。'
            '保留v4原RGB、细区采样和损失。每150步按无TTA、固定0.5的验证Dice选择微调检查点；不额外使用EMA。'
            '新裁块随机流seed=12026+fold，原图划分seed=2026。每batch记录日志。\n\n'
            f'相同D4/0.5下：v4 Dice {result["baseline_fixed_0_5"]["mean_dice"]:.6f}，纯LoRA Dice {result["lora_fixed_0_5"]["mean_dice"]:.6f}。\n\n'
            f'OOF选择LoRA权重{weight}、v4权重{1-weight}，阈值{threshold}，最终选择后Dice {result["selected"]["mean_dice"]:.6f}。'
            '选择允许权重0，若LoRA未改善则保留v4；纯LoRA提交仍另存，不能把回退称为微调提升。\n\n'
            'submission_v6.0.0.csv 为验证选定文件；submission_v6.0.0_lora.csv 为纯LoRA五折文件。'
            '每张OOF图只用其持出折；测试集才平均五折。阈值网格0.40至0.60、间隔0.02，融合网格0/0.25/0.5/0.75/1。\n\n'
            '这些是经过多次选择的本地验证结果，不是独立测试或Kaggle分数。沿用提供标签/官方脚本的白色血管编码，题面极性冲突未获隐藏评分确认。'
            '官方转换、两套提交的20图独立RLE逐像素核验、主干冻结与适配器合并核验均通过。\n\n'
            f'旧版本{len(previous)}个文件SHA256核验不变。所有新文件均在D盘，未上传Kaggle。\n\n'
            '方法参考：https://arxiv.org/abs/2106.09685 。原论文针对大模型；本项目将低秩更新形式用于卷积解码器，并不预设涨分。\n')
    (ROOT/'RESULTS_CN.md').write_text(report,encoding='utf-8')
    pack=ROOT.parent/'releases/vessel_v6.0.0_submission_light.zip'
    with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('训练说明.txt',report.encode('utf-8-sig'));z.write(ROOT/'submission_v6.0.0.csv','submission.csv')
    with zipfile.ZipFile(pack) as z:
        assert hashlib.sha256(z.read('submission.csv')).hexdigest()==sha(ROOT/'submission_v6.0.0.csv')
        assert z.testzip() is None
    manifest={'version':'v6.0.0','status':'completed','parent':'v4.0.0','directory':str(ROOT),
              'submission':{'path':str(ROOT/'submission_v6.0.0.csv'),'sha256':sha(ROOT/'submission_v6.0.0.csv')},
              'pure_lora_submission':{'path':str(ROOT/'submission_v6.0.0_lora.csv'),'sha256':sha(ROOT/'submission_v6.0.0_lora.csv')},
              'lora_weight':weight,'threshold':threshold,'oof_dice':result['selected']['mean_dice'],
              'light_package':{'path':str(pack),'sha256':sha(pack)},'audit':audit}
    write_json(ROOT/'version_manifest.json',manifest)
    write_json(ROOT.parent/'versions/v6.0.0.json',manifest)
    status('completed',**manifest)


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e));raise
