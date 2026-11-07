"""Fine-tune one v4 held-out fold using decoder-only rank4 Conv-LoRA."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from data import load_cases,make_folds
from thin_data import ThinPatchDataset
from thin_loss import thin_aware_loss
from inference import predict_image,summarize,write_json
from model import UNet
from lora import inject,adapter_state,load_adapter,merged,frozen_state

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--lr',type=float,choices=[1e-5,3e-5],required=True)
    a=p.parse_args()
    if not 0<=a.fold<5 or a.steps<1:raise ValueError('Invalid fold or steps')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(12026+a.fold);np.random.seed(12026+a.fold);torch.manual_seed(12026+a.fold)
    device=torch.device('cuda')
    cases=load_cases(PROJECT/'work/data','train')
    split=make_folds(cases,seed=2026)[a.fold]
    base_path=PROJECT/'outputs/vessel_v4/runs'/f'fold{a.fold}'/'best.pt'
    state=torch.load(base_path,map_location='cpu',weights_only=False)
    for key in ('train_ids','val_ids'):
        if state['config'][key]!=split[key]:raise ValueError('Base checkpoint fold contamination')
    if state['config']['version']!='v4.0.0' or state['config']['external_training_data']:
        raise ValueError('Wrong parent checkpoint')
    model=UNet(24);model.load_state_dict(state['model']);model.to(device)
    targets=inject(model,rank=4,alpha=4)
    frozen=frozen_state(model)
    trainable=[p for p in model.parameters() if p.requires_grad]
    config={'version':'v7.0.0','parent_version':'v4.0.0','fold':a.fold,**split,'steps':a.steps,
            'batch_size':8,'patch_size':256,'rank':4,'alpha':4,'lr':a.lr,'weight_decay':1e-4,
            'warmup':30,'eval_every':a.eval_every,'sample_seed':12026+a.fold,'split_seed':2026,
            'checkpoint_min_delta':1e-5,'selection_tta':'d4','selection_includes_step0':True,
            'base_checkpoint':str(base_path),'base_sha256':sha(base_path),'targets':targets,
            'trainable_parameters':sum(p.numel() for p in trainable),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'external_training_data':False,'external_pretrained_weights':False,'input':'original RGB',
            'loss':'unchanged v4 thin-aware loss','sampling':'unchanged v4 mixture with fresh deterministic crop stream',
            'source_sha256':{n:sha(ROOT/n) for n in ('train_lora.py','lora.py','model.py','data.py','thin_data.py','thin_loss.py','inference.py','submission.py')}}
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'config.json').exists() and not a.resume:raise FileExistsError('Use a new output or --resume')
    optimizer=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    start=0;best=-1.;best_step=0;prior_elapsed=0.
    if a.resume:
        last=torch.load(a.output/'last_adapter.pt',map_location='cpu',weights_only=False)
        if last['config']!=config:raise ValueError('Resume configuration changed')
        load_adapter(model,last['adapter']);optimizer.load_state_dict(last['optimizer']);scaler.load_state_dict(last['scaler'])
        start,best,best_step,prior_elapsed=last['step'],last['best_score'],last['best_step'],last['elapsed_seconds']
    write_json(a.output/'config.json',config)
    def evaluate(tta='d4'):
        return {i:predict_image(model,cases[i]['image'],device,256,128,8,tta=tta) for i in split['val_ids']}
    if start==0:
        baseline=evaluate()
        baseline_report=summarize(baseline,cases,0.5,False)
        write_json(a.output/'baseline_report.json',baseline_report)
        original=json.loads((base_path.parent/'validation_report.json').read_text())['d4_at_0_5']['mean_dice']
        if abs(baseline_report['mean_dice']-original)>1e-4:raise ValueError('Zero-adapter prediction differs from v4 baseline')
        best=baseline_report['mean_dice'];best_step=0
        torch.save({'adapter':adapter_state(model),'config':config,'step':0,'score':best},a.output/'best_adapter.pt')
        np.savez_compressed(a.output/'baseline_d4_probabilities.npz',**baseline)
        with (a.output/'validation.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'step':0,'best_step':0,**baseline_report})+'\n')
    dataset=ThinPatchDataset(cases,split['train_ids'],256,length=a.steps*8,seed=12026+a.fold)
    dataset=Subset(dataset,range(start*8,a.steps*8))
    loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
    begun=time.perf_counter()
    print(json.dumps({'event':'start',**config}),flush=True)
    for step,(image,target,thin,skeleton) in enumerate(loader,start=start+1):
        model.train()
        warmup=min(30,max(1,a.steps//10))
        lr=a.lr*(step/warmup if step<=warmup else 0.05+0.95*0.5*(1+math.cos(math.pi*(step-warmup)/max(1,a.steps-warmup))))
        for group in optimizer.param_groups:group['lr']=lr
        image,target,thin,skeleton=[x.to(device,non_blocking=True) for x in (image,target,thin,skeleton)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=thin_aware_loss(model(image),target,thin,skeleton)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable,1.0)
        scaler.step(optimizer);scaler.update()
        elapsed=prior_elapsed+time.perf_counter()-begun
        row={'event':'train','fold':a.fold,'batch':step,'total_batches':a.steps,'loss':float(loss.detach()),'lr':lr,'elapsed_seconds':round(elapsed,2)}
        print(json.dumps(row),flush=True)
        with (a.output/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        write_json(a.output/'live.json',row)
        if step%a.eval_every==0 or step==a.steps:
            probabilities=evaluate()
            report=summarize(probabilities,cases,0.5,False)
            torch.save({'adapter':adapter_state(model),'config':config,'step':step,'score':report['mean_dice']},a.output/f'adapter_step{step:04d}.pt')
            if report['mean_dice']>best+1e-5:
                best,best_step=report['mean_dice'],step
                torch.save({'adapter':adapter_state(model),'config':config,'step':step,'score':best},a.output/'best_adapter.pt')
            with (a.output/'validation.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'step':step,'best_step':best_step,**report})+'\n')
            torch.save({'adapter':adapter_state(model),'config':config,'step':step,'best_score':best,'best_step':best_step,
                        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'elapsed_seconds':prior_elapsed+time.perf_counter()-begun},a.output/'last_adapter.pt')
    for name,value in frozen_state(model).items():
        if not torch.equal(value,frozen[name]):raise AssertionError('Backbone changed: '+name)
    best_state=torch.load(a.output/'best_adapter.pt',map_location='cpu',weights_only=False)
    load_adapter(model,best_state['adapter'])
    plain=merged(model)
    model.eval();plain.eval()
    probe=torch.from_numpy(cases[split['val_ids'][0]]['image'][:256,:256].transpose(2,0,1).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        torch.testing.assert_close(plain(probe),model(probe),rtol=1e-5,atol=1e-5)
    torch.save({'model':{k:v.cpu() for k,v in plain.state_dict().items()},'config':config,'step':best_step,'version':'v7.0.0'},a.output/'best_merged.pt')
    probabilities=evaluate('d4')
    np.savez_compressed(a.output/'validation_d4_probabilities.npz',**probabilities)
    result={'fold':a.fold,'completed_steps':a.steps,'best_step':best_step,'d4_at_0_5':summarize(probabilities,cases,0.5,False),
            'base_frozen_exact':True,'adapter_merge_verified':True,'base_sha256':sha(base_path),
            'trainable_parameters':config['trainable_parameters'],'frozen_parameters':config['frozen_parameters'],
            'baseline_d4_at_0_5':json.loads((a.output/'baseline_report.json').read_text()),
            'selection_includes_step0':True,'selection_tta':'d4',
            'elapsed_seconds':prior_elapsed+time.perf_counter()-begun}
    write_json(a.output/'validation_report.json',result)
    print(json.dumps({'event':'complete',**result}),flush=True)


if __name__=='__main__':main()
