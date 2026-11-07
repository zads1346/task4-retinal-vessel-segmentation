"""Predict held-out test images and verify a CSV against the official converter."""
import argparse
from pathlib import Path
import numpy as np
import torch
from data import load_cases
from model import UNet
from inference import predict_image, write_json
from submission import export_submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--polarity', choices=['vessel-white', 'vessel-black'], default='vessel-white')
    p.add_argument('--resize-mode', choices=['official', 'probability'], default='official')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if not (args.data/'test').is_dir() and (args.data/'segmentation'/'test').is_dir():
        args.data = args.data/'segmentation'
    if not 0 < args.threshold < 1 or args.batch_size < 1:
        raise ValueError('Threshold must be in (0,1), batch size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use configured Python')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    cases = load_cases(args.data, 'test')
    probabilities = {}
    model_metadata = []
    for path in args.checkpoint:
        state = torch.load(path, map_location=device, weights_only=False)
        config = state['config']
        model = UNet(config['base_channels']).to(device)
        model.load_state_dict(state['model'])
        model_metadata.append({'checkpoint':str(path.resolve()), 'step':state['step'],
                               'version':state.get('version',config.get('version','v1.0.0')),
                               'weight_kind':state.get('weight_kind','raw'),
                               'train_ids':config['train_ids'], 'validation_ids':config['val_ids']})
        for case_id, case in cases.items():
            probability = predict_image(model, case['image'], device, config['patch_size'],
                                        config['patch_size']//2, args.batch_size, tta=args.tta)
            if case_id not in probabilities:
                probabilities[case_id] = probability / len(args.checkpoint)
            else:
                probabilities[case_id] += probability / len(args.checkpoint)
            print(f'Predicted model={path.name} image={case_id}', flush=True)
        del model
    np.savez_compressed(args.output/'test_probabilities.npz', **probabilities)
    report = export_submission(probabilities, args.output, args.data/'sample_submission.csv',
                               args.threshold, args.polarity, args.resize_mode,
                               official_script=args.data/'segmentation_to_csv.py')
    write_json(args.output/'prediction_run.json', {'models':model_metadata, 'tta':args.tta,
               'threshold':args.threshold, 'polarity':args.polarity, 'resize_mode':args.resize_mode,
               'external_training_data':False, 'pretrained_weights':False, 'submission_checks':report})
    print(f'CSV verified: {args.output.resolve() / "submission.csv"}', flush=True)


if __name__ == '__main__':
    main()
