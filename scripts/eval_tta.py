"""TTA (test-time augmentation) evaluation: average recognition probabilities
over shifted/scaled crop variants before decoding, then apply post-processing.

Stabilizes borderline predictions (the punctuation/digit flips).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from IndicPhotoOCR.ocr import OCR
from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
from postprocess import MarathiPostprocessor
from score_deterministic import levenshtein


def make_variants(img: Image.Image):
    """Original + 2 shifts + 2 scales (no flips/rotations for Devanagari)."""
    w, h = img.size
    variants = [img]
    # shifts
    for dx in (-2, 2):
        canvas = Image.new('RGB', (w, h), (255, 255, 255))
        canvas.paste(img, (dx, 0))
        variants.append(canvas)
    # scale 0.9 (pad)
    sw, sh = max(1, int(w * 0.9)), max(1, int(h * 0.9))
    small = img.resize((sw, sh), Image.BICUBIC)
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    canvas.paste(small, ((w - sw) // 2, (h - sh) // 2))
    variants.append(canvas)
    # scale 1.1 (center crop)
    sw, sh = int(w * 1.1), int(h * 1.1)
    big = img.resize((sw, sh), Image.BICUBIC)
    variants.append(big.crop(((sw - w) // 2, (sh - h) // 2,
                              (sw - w) // 2 + w, (sh - h) // 2 + h)))
    return variants


def tta_recognise(model, device, img: Image.Image, transform):
    """Average output probabilities over crop variants, decode once."""
    tensors = [transform(v) for v in make_variants(img.convert('RGB'))]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = logits.softmax(-1)
        avg_probs = probs.mean(dim=0, keepdim=True)
        preds, _ = model.tokenizer.decode(avg_probs)
        text = model.charset_adapter(preds[0])
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/tmp/test_marathi2.png')
    ap.add_argument('--gt-dump', default='/tmp/gt_lines/eval_dump.json')
    ap.add_argument('--ckpt', default='data/marathi_stage5.ckpt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--models', nargs='+', default=[
        'stage5=data/marathi_stage5.ckpt',
        'stage3=data/marathi_stage3_final.ckpt',
    ])
    args = ap.parse_args()

    # GT + boxes from the last eval dump
    data = json.load(open(args.gt_dump))
    gt_at_box = {int(k): v for k, v in data['gt_at_box'].items()}
    preds_base = data['preds']
    covered = sorted(gt_at_box)

    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)
    image = cv2.imread(args.image)
    detections = ocr.detect(args.image)

    print(f'{"model":<10}{"exact":>6}{"exact%":>9}{"characc%":>10}')
    all_tta = {}
    for spec in args.models:
        slug, ckpt = spec.split('=', 1)
        if not os.path.exists(ckpt):
            print(f'  skipping {slug}')
            continue
        model = load_from_checkpoint(ckpt).eval().to(args.device)
        hp = model.hparams
        transform = T.Compose([
            T.Resize(hp.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(0.5, 0.5)])

        texts = {}
        for i, bbox in enumerate(detections):
            if i not in gt_at_box:
                continue
            # crop directly with cv2 (same geometry as crop_bbox, no temp file)
            pts = np.array(bbox, np.int32)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            crop = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            texts[i] = tta_recognise(model, args.device, crop, transform).strip()
            if (i + 1) % 100 == 0:
                print(f'    {i + 1}/{len(detections)}')
        all_tta[slug] = texts

        ex = sum(1 for b in covered if texts[b] == gt_at_box[b])
        tot_ed = sum(levenshtein(gt_at_box[b], texts[b]) for b in covered)
        tot_len = sum(max(len(gt_at_box[b]), 1) for b in covered)
        print(f'{slug:<10}{ex:>6}{100 * ex / len(covered):>9.2f}'
              f'{100 * (1 - tot_ed / tot_len):>10.2f}')

        # what TTA changed vs the eval-dump predictions (if same slug present)
        if slug in preds_base:
            flips = [(b, preds_base[slug][b], texts[b], gt_at_box[b])
                     for b in covered if preds_base[slug][b] != texts[b]]
            better = sum(1 for _, o, n, g in flips if n == g)
            worse = sum(1 for _, o, n, g in flips if o == g)
            print(f'  TTA flips: {len(flips)} (better={better}, worse={worse}, '
                  f'same-wrong={len(flips) - better - worse})')

    # dump TTA texts + failures for the LAST model
    last = args.models[-1].split('=')[0]
    if last in all_tta:
        texts = all_tta[last]
        with open('/tmp/gt_lines/tta_dump.json', 'w', encoding='utf-8') as f:
            json.dump({'tta': texts, 'gt': gt_at_box}, f, ensure_ascii=False, indent=1)
        fails = [(b, gt_at_box[b], texts[b]) for b in covered
                 if texts[b] != gt_at_box[b]]
        print(f'\n{last.upper()} + TTA REMAINING FAILURES: {len(fails)}/{len(covered)}')
        for b, g, p in fails:
            print(f'  #{b:>3} GT={g!r}')
            print(f'        TTA={p!r}')


if __name__ == '__main__':
    main()
