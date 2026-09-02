"""Evaluate Gujarati checkpoints on the held-out test word crops (110 words).

Usage:
  python scripts/eval_gujarati.py [--models slug=ckpt ...] [--tta]
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
from eval_tta import tta_recognise
from score_deterministic import levenshtein

SPECIAL = set('!-./:;(),')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-jsonl', default='data/gujarati_test/labels.jsonl')
    ap.add_argument('--test-dir', default='data/gujarati_test')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--models', nargs='+', default=[
        'original=IndicPhotoOCR/recognition/models/gujarati.ckpt',
        'stage1=data/gujarati_stage1.ckpt',
    ])
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.test_jsonl, encoding='utf-8')]
    print(f'test crops: {len(samples)}   TTA: {args.tta}')

    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)

    print(f'{"model":<12}{"exact":>6}{"exact%":>8}{"characc%":>10}{"specials":>10}')
    for spec in args.models:
        slug, ckpt = spec.split('=', 1)
        if not os.path.exists(ckpt):
            print(f'  skipping {slug}: not found')
            continue
        model = load_from_checkpoint(ckpt).eval().to(args.device)
        hp = model.hparams
        transform = T.Compose([
            T.Resize(hp.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(0.5, 0.5)])

        correct = sp_total = sp_correct = 0
        tot_ed = tot_len = 0
        fails = []
        for s in samples:
            img = Image.open(os.path.join(args.test_dir, s['image_filename'])).convert('RGB')
            if args.tta:
                text = tta_recognise(model, args.device, img, transform).strip()
            else:
                with torch.no_grad():
                    logits = model(transform(img).unsqueeze(0).to(args.device))
                    probs = logits.softmax(-1)
                    preds, _ = model.tokenizer.decode(probs)
                    text = model.charset_adapter(preds[0]).strip()
            gt = s['expected_text'].strip()
            if any(c in SPECIAL for c in gt):
                sp_total += 1
                if text == gt:
                    sp_correct += 1
            if text == gt:
                correct += 1
            else:
                fails.append((gt, text))
            tot_ed += levenshtein(gt, text)
            tot_len += max(len(gt), 1)

        print(f'{slug:<12}{correct:>6}{100 * correct / len(samples):>8.2f}'
              f'{100 * (1 - tot_ed / tot_len):>10.2f}'
              f'{sp_correct:>4}/{sp_total:<5}')
        if fails:
            print('  failures:')
            for gt, t in fails[:15]:
                print(f'    GT={gt!r:<28} pred={t!r}')


if __name__ == '__main__':
    main()
