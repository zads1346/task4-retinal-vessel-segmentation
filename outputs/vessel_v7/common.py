"""Inference/diagnostic helpers inherited from v6; no training entry point."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from inference import predict_image,summarize,write_json
from model import UNet
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
DATA=PROJECT/'work/data'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def status(stage,**extra):
    row={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**extra}
    write_json(ROOT/'status.json',row);print(json.dumps(row),flush=True)


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

