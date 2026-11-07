"""Fetch only explicitly authorized official ImageNet classification weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import torch
import torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, weights in [('resnet18', ResNet18_Weights.IMAGENET1K_V1),
                          ('resnet34', ResNet34_Weights.IMAGENET1K_V1)]:
        filename = Path(urlparse(weights.url).path).name
        prefix = filename.rsplit('-', 1)[1].split('.')[0]
        target = destination/filename
        if not target.exists():
            torch.hub.download_url_to_file(weights.url, str(target), hash_prefix=prefix, progress=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if not digest.startswith(prefix):
            raise ValueError(f'Official weight hash prefix mismatch: {target}')
        sources[name] = {'path':str(target.resolve()), 'url':weights.url,
                         'enum':f'{name}.IMAGENET1K_V1', 'sha256':digest,
                         'expected_sha256_prefix':prefix, 'bytes':target.stat().st_size,
                         'pretraining_dataset':'ImageNet-1K',
                         'normalization_mean':[0.485,0.456,0.406],
                         'normalization_std':[0.229,0.224,0.225]}
    report = {'torchvision_version':torchvision.__version__, 'sources':sources,
              'fine_tuning_data_policy':'Only the supplied 20 retinal image/label pairs; no additional image dataset downloaded.',
              'authorization':'User explicitly authorized public pretrained models for v3.',
              'documentation':['https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html',
                               'https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet34.html']}
    (destination/'weights_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    report = fetch(a.destination)
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
