"""Re-evaluate only a checkpoint's held-out images; optional calibrated TTA."""
import argparse
from pathlib import Path
import numpy as np
import torch
from data import load_cases
from model import UNet
from inference import predict_image, summarize, threshold_search, save_preview, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Use the configured CUDA Python')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    if set(config['train_ids']) & set(config['val_ids']):
        raise ValueError('Checkpoint contains overlapping training and validation IDs')
    cases = load_cases(args.data, 'train')
    model = UNet(config['base_channels']).to(device)
    model.load_state_dict(checkpoint['model'])
    probabilities = {i:predict_image(model, cases[i]['image'], device, config['patch_size'],
                     config['patch_size']//2, args.batch_size, tta=args.tta) for i in config['val_ids']}
    args.output.mkdir(parents=True, exist_ok=True)
    tuning = threshold_search(probabilities, cases)
    report = {'version':checkpoint.get('version',config.get('version','v1.0.0')),
              'weight_kind':checkpoint.get('weight_kind','raw'),
              'checkpoint':str(args.checkpoint.resolve()), 'checkpoint_step':checkpoint['step'],
              'tta':args.tta, 'validation_ids':config['val_ids'], 'at_0_5':summarize(probabilities, cases),
              'threshold_tuning':tuning,
              'at_native_selected_threshold':summarize(probabilities, cases, tuning['native_best_threshold']),
              'scope':'Only checkpoint held-out images. Tuned holdout result, not independent test performance.'}
    np.savez_compressed(args.output/'validation_probabilities.npz', **probabilities)
    write_json(args.output/'validation_report.json', report)
    save_preview(probabilities, cases, args.output/'validation_preview.png', tuning['native_best_threshold'])
    print(report['at_native_selected_threshold'])


if __name__ == '__main__':
    main()
