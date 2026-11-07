"""Overlap inference and honest vessel-only validation metrics."""
from contextlib import nullcontext
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch


def positions(length, patch, stride):
    if stride < 1 or stride > patch:
        raise ValueError('stride must be in [1, patch_size]')
    return sorted(set(list(range(0, max(0, length-patch)+1, stride)) + [max(0, length-patch)]))


@torch.inference_mode()
def predict_image(model, image, device, patch_size=256, stride=128, batch_size=8, tta=False, amp=True):
    """Native HxW probability with none, legacy flip4, or 8-view dihedral TTA."""
    mode = ('flip4' if tta else 'none') if isinstance(tta, bool) else tta
    if mode not in {'none', 'flip4', 'd4'}:
        raise ValueError('tta must be bool or none/flip4/d4')
    model.eval()
    height, width = image.shape[:2]
    ph, pw = max(height, patch_size), max(width, patch_size)
    padded = np.pad(image, ((0, ph-height), (0, pw-width), (0, 0)), mode='constant')
    coords = [(y, x) for y in positions(ph, patch_size, stride)
              for x in positions(pw, patch_size, stride)]
    axis = np.maximum(np.hanning(patch_size), 0.05).astype(np.float32)
    weight = np.outer(axis, axis)
    total = np.zeros((ph, pw), dtype=np.float32)
    denominator = np.zeros_like(total)
    variants = ([(0, axes) for axes in [(), (2,), (3,), (2, 3)]] if mode == 'flip4'
                else [(turns, axes) for turns in range(4) for axes in [(), (3,)]] if mode == 'd4'
                else [(0, ())])
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start+batch_size]
        array = np.stack([padded[y:y+patch_size, x:x+patch_size].transpose(2, 0, 1) for y, x in chunk])
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        prediction = None
        for turns, axes in variants:
            augmented = torch.rot90(tensor, turns, (2, 3)) if turns else tensor
            augmented = torch.flip(augmented, axes) if axes else augmented
            context = torch.autocast('cuda', dtype=torch.float16) if amp and device.type == 'cuda' else nullcontext()
            with context:
                probability = model(augmented).float().sigmoid()
            if axes:
                probability = torch.flip(probability, axes)
            if turns:
                probability = torch.rot90(probability, -turns, (2, 3))
            prediction = probability if prediction is None else prediction + probability
        result = (prediction / len(variants))[:, 0].cpu().numpy()
        for (y, x), patch in zip(chunk, result):
            total[y:y+patch_size, x:x+patch_size] += patch * weight
            denominator[y:y+patch_size, x:x+patch_size] += weight
    if not np.all(denominator > 0):
        raise RuntimeError('Sliding window left uncovered pixels')
    return total[:height, :width] / denominator[:height, :width]


def binary_metrics(prediction, target):
    p, y = np.asarray(prediction, bool), np.asarray(target, bool)
    if p.shape != y.shape:
        raise ValueError('Metric shape mismatch')
    tp = int(np.count_nonzero(p & y))
    fp = int(np.count_nonzero(p & ~y))
    fn = int(np.count_nonzero(~p & y))
    return dict(dice=(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
                precision=tp/(tp+fp) if tp+fp else (1.0 if not y.any() else 0.0),
                recall=tp/(tp+fn) if tp+fn else 1.0,
                tp=tp, fp=fp, fn=fn, positive_fraction=float(p.mean()))


def summarize(probabilities, cases, threshold=0.5, include_proxy=True):
    # Imported lazily so inference itself does not need the CSV module.
    if include_proxy:
        from submission import official_mask
    rows = []
    for case_id, probability in probabilities.items():
        target = cases[case_id]['mask'] > 0.5
        prediction = probability >= threshold
        row = {'id': case_id, **binary_metrics(prediction, target)}
        if include_proxy:
            proxy = binary_metrics(official_mask(prediction.astype(np.uint8)*255),
                                   official_mask(target.astype(np.uint8)*255))
            row['official_white_vessel_proxy_dice'] = proxy['dice']
        rows.append(row)
    tp, fp, fn = [sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')]
    result = {'threshold': float(threshold), 'mean_dice': float(np.mean([r['dice'] for r in rows])),
              'std_dice': float(np.std([r['dice'] for r in rows])),
              'micro_dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
              'per_image': rows}
    if include_proxy:
        result['official_white_vessel_proxy_mean_dice'] = float(np.mean([r['official_white_vessel_proxy_dice'] for r in rows]))
    return result


def threshold_search(probabilities, cases):
    table = []
    for threshold in np.round(np.arange(0.20, 0.801, 0.02), 2):
        report = summarize(probabilities, cases, threshold)
        table.append({k: v for k, v in report.items() if k != 'per_image'})
    native = max(table, key=lambda row: (row['mean_dice'], -abs(row['threshold']-0.5)))
    proxy = max(table, key=lambda row: (row['official_white_vessel_proxy_mean_dice'], -abs(row['threshold']-0.5)))
    return {'native_best_threshold': native['threshold'],
            'official_proxy_best_threshold': proxy['threshold'], 'sweep': table,
            'caveat': 'Thresholds were selected on this validation fold; these are tuned validation scores, not independent test estimates. Hidden scoring polarity/preprocessing is unconfirmed.'}


def save_preview(probabilities, cases, path, threshold=0.5):
    """Four columns: RGB, supplied label, probability, thresholded prediction."""
    cell_w, cell_h, header = 280, 290, 24
    sheet = Image.new('RGB', (cell_w*4, (cell_h+header)*len(probabilities)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (case_id, probability) in enumerate(probabilities.items()):
        case = cases[case_id]
        pictures = [('RGB', np.uint8(np.clip(case['image']*255, 0, 255))),
                    ('Ground truth', np.uint8(case['mask']*255)),
                    ('Vessel probability', np.uint8(np.clip(probability*255, 0, 255))),
                    (f'Prediction t={threshold:.2f}', np.uint8(probability >= threshold)*255)]
        for col, (title, array) in enumerate(pictures):
            tile = ImageOps.contain(Image.fromarray(array).convert('RGB'), (cell_w, cell_h))
            y = row*(cell_h+header)
            draw.text((col*cell_w+3, y+4), f'{case_id}: {title}', fill='black')
            sheet.paste(tile, (col*cell_w, y+header))
    sheet.save(path)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
