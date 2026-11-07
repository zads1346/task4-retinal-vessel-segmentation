"""Render actual held-out and test probabilities for visual inspection, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from data import load_cases
from inference import save_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = json.loads((root / 'oof/oof_report.json').read_text())
    selected = report['variants'][report['selection']['selected_variant']]
    threshold = float(report['selection']['threshold'])
    probability_path = Path(selected['oof_probabilities_path'])
    if hashlib.sha256(probability_path.read_bytes()).hexdigest() != selected['oof_probabilities_sha256']:
        raise ValueError('OOF probabilities changed after selection')
    worst_ids = [row['id'] for row in sorted(selected['native_tuned']['per_image'], key=lambda row: row['dice'])[:4]]
    cases = load_cases(args.data.resolve(), 'train')
    with np.load(probability_path, allow_pickle=False) as archive:
        worst = {image_id: archive[image_id] for image_id in worst_ids}
    save_preview(worst, cases, root / 'oof/validation_worst4_preview.png', threshold)

    cases = load_cases(args.data.resolve(), 'test')
    test_ids = ['1', '6', '11', '16']
    cell_w, cell_h, header = 280, 290, 24
    sheet = Image.new('RGB', (cell_w * 3, (cell_h + header) * len(test_ids)), 'white')
    draw = ImageDraw.Draw(sheet)
    with np.load(root / 'submission/test_probabilities.npz', allow_pickle=False) as archive:
        for row, image_id in enumerate(test_ids):
            probability = archive[image_id]
            arrays = [('Test RGB', np.uint8(np.clip(cases[image_id]['image'] * 255, 0, 255))),
                      ('Ensemble probability', np.uint8(np.clip(probability * 255, 0, 255))),
                      (f'Prediction t={threshold:.2f}', np.uint8(probability >= threshold) * 255)]
            for col, (title, array) in enumerate(arrays):
                tile = ImageOps.contain(Image.fromarray(array).convert('RGB'), (cell_w, cell_h))
                y = row * (cell_h + header)
                draw.text((col * cell_w + 3, y + 4), f'{image_id}: {title}', fill='black')
                sheet.paste(tile, (col * cell_w, y + header))
    sheet.save(root / 'submission/test_preview.png')
    print(json.dumps({'worst_held_out_ids': worst_ids, 'test_preview_ids': test_ids, 'threshold': threshold}))


if __name__ == '__main__':
    main()
