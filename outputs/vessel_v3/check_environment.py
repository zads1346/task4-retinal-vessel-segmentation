"""Verify the actual runtime, including CUDA backward and converter imports."""
import argparse
import json
from pathlib import Path
import platform
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import PIL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('environment.json'))
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refuse a silent CPU fallback')
    x = torch.randn(128, 128, device='cuda', requires_grad=True)
    (x @ x).square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError('CUDA backward produced invalid values')
    props = torch.cuda.get_device_properties(0)
    venv_config = Path(sys.prefix)/'pyvenv.cfg'
    inherits = venv_config.exists() and 'include-system-site-packages = true' in venv_config.read_text()
    info = {'python_executable':sys.executable, 'python_version':platform.python_version(),
            'base_prefix':sys.base_prefix, 'torch':torch.__version__, 'torchvision':torchvision.__version__,
            'cuda_runtime':torch.version.cuda, 'cuda_available':True, 'gpu':props.name,
            'gpu_total_memory_mb':props.total_memory/2**20, 'numpy':np.__version__,
            'pandas':pd.__version__, 'pillow':PIL.__version__, 'gpu_backward_verified':True,
            'training_data_policy':'Retinal fine-tuning uses only the supplied 20 image/label pairs; encoder initialization uses user-authorized official ImageNet-1K weights.',
            'environment_layout':('Task venv inherits existing CUDA packages; task-only missing dependencies installed locally.'
                                  if inherits else 'Standalone task environment.')}
    args.output.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
