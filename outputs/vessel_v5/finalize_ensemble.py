"""Preserve unconstrained selection and deliver the requested genuine v4 blend."""
import csv
import json
import shutil
import zipfile
import numpy as np
from PIL import Image
from pipeline import ROOT,DATA,VERSION,read,sha,blend,diagnostics
from data import load_cases
from inference import summarize,write_json
from submission import export_submission


def main():
    selection=read(ROOT/'oof/selection.json')
    assert read(ROOT/'status.json')['stage']=='completed'
    cases=load_cases(DATA,'train')
    with np.load(ROOT/'oof/candidate_probabilities.npz',allow_pickle=False) as saved:
        candidate={k:saved[k] for k in saved.files}
    baseline={}
    for record in selection['v4_models']:
        with np.load(record['oof_probabilities'],allow_pickle=False) as saved:
            baseline.update({k:saved[k] for k in saved.files})
    interior=[r for r in selection['weight_table_fixed_0_5'] if 0<r['candidate_weight']<1]
    weight=max(interior,key=lambda r:r['mean_dice'])['candidate_weight']
    oof=blend(candidate,baseline,weight)
    thresholds=[{'threshold':float(t),'mean_dice':summarize(oof,cases,float(t),False)['mean_dice']}
                for t in np.round(np.arange(0.4,0.601,0.02),2)]
    threshold=max(thresholds,key=lambda r:(r['mean_dice'],-abs(r['threshold']-0.5)))['threshold']
    totals=[]
    for name in (selection['selected_experiment'],'v4'):
        total={}
        for fold in range(5):
            path=ROOT/'test_model_probabilities'/f'{name}_fold{fold}.npz'
            assert sha(path)==read(path.with_suffix('.json'))['sha256']
            with np.load(path,allow_pickle=False) as saved:
                for key in saved.files:
                    if key not in total:total[key]=np.zeros_like(saved[key])
                    total[key]+=saved[key]/5
        totals.append(total)
    final=blend(totals[0],totals[1],weight)
    folder=ROOT/'submission_v4_blend';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'test_probabilities.npz',**final)
    metadata=export_submission(final,folder,DATA/'sample_submission.csv',threshold,
                               polarity='vessel-white',resize_mode='official',official_script=DATA/'segmentation_to_csv.py')
    destination=ROOT/'submission_v5.0.0_ensemble.csv'
    shutil.copy2(folder/'submission.csv',destination)
    # Separate row decoder, independent of submission.py.
    with destination.open(newline='') as f:rows=list(csv.DictReader(f))
    with (DATA/'sample_submission.csv').open(newline='') as f:expected=list(csv.DictReader(f))
    assert [r['Id'] for r in rows]==[r['Id'] for r in expected]
    for row in rows:
        values=list(map(int,row['Predicted'].split()));assert len(values)%2==0
        flat=np.zeros(512*512,np.uint8);end=0
        for start,length in zip(values[::2],values[1::2]):
            assert start>=1 and length>0 and start>end and start+length-1<=flat.size
            flat[start-1:start-1+length]=1;end=start+length-1
        with Image.open(folder/'image'/f'{row["Id"]}.png') as im:
            assert np.array_equal(np.asarray(im)>0,final[row['Id']]>=threshold)
            expected_mask=np.asarray(im.convert('L').resize((512,512),Image.Resampling.BILINEAR))>0
        assert np.array_equal(flat.reshape(512,512),expected_mask)
    report={'version':VERSION,'candidate_weight':weight,'v4_weight':1-weight,'threshold':threshold,
            'selection_rule':'Best strictly interior weight from the predeclared grid at fixed0.5; then same global threshold grid. Explicit user request for v4 integration.',
            'unconstrained_selection_preserved':'oof/selection.json and submission_v5.0.0.csv',
            'at_0_5':diagnostics(oof,cases,0.5),'at_selected_threshold':diagnostics(oof,cases,threshold),
            'threshold_table':thresholds,'official_converter_verified':metadata['official_script_verified'],
            'independent_RLE_roundtrip_20_images':True,'submission':{'path':str(destination),'sha256':sha(destination)},
            'warning':'Tuned OOF validation, not an independent test score; no Kaggle upload.'}
    write_json(ROOT/'oof/requested_ensemble_selection.json',report)
    manifest=read(ROOT/'version_manifest.json')
    manifest['requested_v4_ensemble_submission']=report['submission']
    manifest['recommended_submission']=report['submission']['path']
    manifest['requested_v4_ensemble_oof_dice']=report['at_selected_threshold']['mean_dice']
    with (ROOT/'RESULTS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## Requested v4 integration\n\nUse submission_v5.0.0_ensemble.csv for the requested genuine blend: '
                f'A {weight:.0%} + v4 {1-weight:.0%}, threshold {threshold}. '
                f'OOF Dice {report["at_selected_threshold"]["mean_dice"]:.8f}. '
                'Unconstrained selection chose A alone by a negligible fixed-threshold margin and is retained in submission_v5.0.0.csv. '
                'Both outputs passed official conversion and independent pixelwise RLE decoding.\n')
    archive=ROOT.parent/'vessel_v5.0.0_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in ('.py','.ps1','.md','.txt'):z.write(p,p.name)
    manifest['source_archive']['sha256']=sha(archive)
    write_json(ROOT/'version_manifest.json',manifest)
    write_json(ROOT.parent/'versions'/f'{VERSION}.json',manifest)
    status=read(ROOT/'status.json')
    status.update(manifest)
    write_json(ROOT/'status.json',status)
    print(json.dumps({k:v for k,v in report.items() if k not in ('threshold_table','at_0_5','at_selected_threshold')}))
    print(json.dumps({'mean_dice':report['at_selected_threshold']['mean_dice'],
                      'thin_recall':report['at_selected_threshold']['thin_skeleton_recall']}))


if __name__=='__main__':main()
