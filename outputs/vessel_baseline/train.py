"""Train one whole-image fold using only the supplied training images."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data import load_cases, make_folds, PatchDataset
from model import UNet, dice_bce_loss
from inference import predict_image, summarize, threshold_search, save_preview, write_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--patch-size', type=int, default=256)
    p.add_argument('--base-channels', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--resume', type=Path)
    p.add_argument('--no-amp', action='store_true')
    return p.parse_args()


def main():
    args = arguments()
    if not 0 <= args.fold < 5 or min(args.steps, args.batch_size, args.eval_every, args.log_every) < 1:
        raise ValueError('Invalid fold or positive numeric argument')
    if args.patch_size < 32 or args.patch_size % 16:
        raise ValueError('Patch size must be >=32 and divisible by 16')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required: use the configured GPU Python, not system Python')
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError('Resume must use the original run directory as --output (preserves historical best.pt)')
    if (args.output/'last.pt').exists() and args.resume is None:
        raise FileExistsError('Run exists: choose a new --output or explicitly --resume last.pt')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    random.seed(args.seed+args.fold)
    np.random.seed(args.seed+args.fold)
    torch.manual_seed(args.seed+args.fold)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed+args.fold)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    cases = load_cases(args.data, 'train')  # Test images are never opened here.
    folds = make_folds(cases, seed=args.seed)
    split = folds[args.fold]
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(split)
    config.update(external_training_data=False, pretrained_weights=False,
                  target='vessel=1 (supplied label > 127)', torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start_step, best_score, best_step = 0, -1., 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ('train_ids', 'val_ids'):
            if checkpoint['config'][key] != config[key]:
                raise ValueError('Resume split mismatch')
        if 'optimizer' not in checkpoint or not (args.output/'best.pt').is_file():
            raise ValueError('Resume requires last.pt and the corresponding best.pt in the run directory')
        for key in ('base_channels', 'patch_size', 'batch_size', 'seed', 'lr', 'weight_decay', 'no_amp'):
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_step, best_score, best_step = checkpoint['step'], checkpoint['best_score'], checkpoint['best_step']
    if args.steps <= start_step:
        raise ValueError('--steps must exceed completed steps when resuming')
    write_json(args.output/'folds.json', folds)
    write_json(args.output/'config.json', config)
    dataset = PatchDataset(cases, split['train_ids'], args.patch_size,
                           length=args.steps*args.batch_size, seed=args.seed+args.fold)
    dataset = Subset(dataset, range(start_step*args.batch_size, args.steps*args.batch_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type=='cuda', drop_last=True)
    started = time.perf_counter()
    losses = []
    print(json.dumps({'event':'start', 'parameters':sum(p.numel() for p in model.parameters()),
                      'device':str(device), **split, 'steps':args.steps}), flush=True)
    for step, (image, target) in enumerate(loader, start=start_step+1):
        model.train()
        warmup = min(100, max(1, args.steps//20))
        if step <= warmup:
            lr = args.lr * step/warmup
        else:
            phase = (step-warmup)/max(1, args.steps-warmup)
            lr = args.lr * (0.01 + 0.99*0.5*(1+math.cos(math.pi*phase)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
            loss = dice_bce_loss(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == start_step+1:
            record = {'event':'train', 'step':step, 'loss':float(np.mean(losses)), 'lr':lr,
                      'elapsed_seconds':round(time.perf_counter()-started, 2)}
            print(json.dumps(record), flush=True)
            with (args.output/'training.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            losses.clear()
        if step % args.eval_every == 0 or step == args.steps:
            probabilities = {i: predict_image(model, cases[i]['image'], device, args.patch_size,
                             args.patch_size//2, args.batch_size, amp=amp) for i in split['val_ids']}
            report = summarize(probabilities, cases, 0.5)
            improved = report['mean_dice'] > best_score
            if improved:
                best_score, best_step = report['mean_dice'], step
            state = {'model':model.state_dict(), 'config':config, 'step':step,
                     'best_score':best_score, 'best_step':best_step}
            if improved:
                torch.save(state, args.output/'best.pt')
            state.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
            torch.save(state, args.output/'last.pt')
            record = {'event':'validation', 'step':step, 'best_step':best_step,
                      'elapsed_seconds':round(time.perf_counter()-started, 2), **report}
            print(json.dumps(record), flush=True)
            with (args.output/'validation.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
    checkpoint = torch.load(args.output/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    probabilities = {i: predict_image(model, cases[i]['image'], device, args.patch_size,
                     args.patch_size//2, args.batch_size, amp=amp) for i in split['val_ids']}
    np.savez_compressed(args.output/'validation_probabilities.npz', **probabilities)
    tuning = threshold_search(probabilities, cases)
    report = {'best_step':best_step, 'completed_steps':args.steps, 'fold':args.fold,
              'validation_ids':split['val_ids'], 'at_0_5':summarize(probabilities, cases, 0.5),
              'threshold_tuning':tuning,
              'at_native_selected_threshold':summarize(probabilities, cases, tuning['native_best_threshold']),
              'total_seconds':time.perf_counter()-started,
              'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
              'scope':'Single 16/4 holdout fold; not five-fold OOF and not a Kaggle score.'}
    write_json(args.output/'validation_report.json', report)
    save_preview(probabilities, cases, args.output/'validation_preview.png', tuning['native_best_threshold'])
    print(json.dumps({'event':'complete', 'best_step':best_step, 'best_dice_at_0_5':best_score,
                      'selected_threshold':tuning['native_best_threshold'], 'output':str(args.output.resolve())}), flush=True)


if __name__ == '__main__':
    main()
