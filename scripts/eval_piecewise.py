"""Piecewise recognition for wide code crops.

The 32x128 input transform crushes 40+ char codes below the readability
floor (~3px/char). Piecewise inference splits wide crops at projection
gaps, pads each piece to >=150px width (avoiding the transform's
horizontal stretch), recognizes pieces, and concatenates.

Rule (fixed from synthetic + training-crop analysis, NOT tuned on test):
  crop width >= 450px  ->  piecewise;  else single-view.

Usage:
  python scripts/eval_piecewise.py [--models slug=ckpt ...] [--width-thresh 450]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
from IndicPhotoOCR.recognition.generate_dataset import _vertical_projection, _find_word_gaps
from score_deterministic import levenshtein

SPECIAL = set('!-./:;(),')


def pad_piece(img, min_w=150):
    w, h = img.size
    if w >= min_w:
        return img
    canvas = Image.new('RGB', (min_w, h), (255, 255, 255))
    canvas.paste(img, ((min_w - w) // 2, 0))
    return canvas


def _find_low_ink_gaps(arr, min_gap_width=3, ink_frac=0.06):
    """Gaps where column ink is below ink_frac of the crop height.

    Real document code crops have no zero-ink columns (the Devanagari
    shirorekha and anti-aliasing bridge everything), but inter-glyph-group
    boundaries show up as very low ink."""
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    ink = (gray < 200).astype(np.int32)
    proj = ink.sum(axis=0)
    thresh = max(1, int(arr.shape[0] * ink_frac))
    is_gap = proj <= thresh
    gaps = []
    in_gap = False
    start = 0
    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap, start = True, i
        elif not g and in_gap:
            in_gap = False
            if i - start >= min_gap_width:
                gaps.append((start, i))
    if in_gap and len(is_gap) - start >= min_gap_width:
        gaps.append((start, len(is_gap)))
    return gaps


def piecewise_recognise(model, img, transform, min_piece_w=300):
    arr = np.array(img.convert('L'))
    gaps = _find_low_ink_gaps(arr, min_gap_width=3, ink_frac=0.06)
    frags, prev = [], 0
    for gs, ge in gaps:
        if gs - prev >= 4:
            frags.append((prev, gs))
        prev = ge
    if arr.shape[1] - prev >= 4:
        frags.append((prev, arr.shape[1]))
    if len(frags) <= 1:
        return None
    # merge adjacent fragments into word-sized pieces (>= min_piece_w)
    pieces = [list(frags[0])]
    for x1, x2 in frags[1:]:
        if pieces[-1][1] - pieces[-1][0] < min_piece_w:
            pieces[-1][1] = x2
        else:
            pieces.append([x1, x2])
    texts = []
    for x1, x2 in pieces:
        piece = pad_piece(img.crop((x1, 0, x2, img.size[1])))
        with torch.no_grad():
            logits = model(transform(piece).unsqueeze(0))
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            texts.append(model.charset_adapter(preds[0]).strip())
    return ''.join(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-jsonl', default='data/marathi_v2_test/labels.jsonl')
    ap.add_argument('--test-dir', default='data/marathi_v2_test')
    ap.add_argument('--models', nargs='+',
                    default=['v41=data/marathi_v2_stage5_syn.ckpt'])
    ap.add_argument('--width-thresh', type=int, default=450)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.test_jsonl, encoding='utf-8')]
    print(f'test crops: {len(samples)}   piecewise threshold: {args.width_thresh}px')

    for spec in args.models:
        slug, ckpt = spec.split('=', 1)
        if not os.path.exists(ckpt):
            print(f'skipping {slug}: not found')
            continue
        m = load_from_checkpoint(ckpt).eval().to(args.device)
        tr = T.Compose([T.Resize(m.hparams.img_size, T.InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(0.5, 0.5)])

        def single(img):
            with torch.no_grad():
                logits = m(tr(img).unsqueeze(0).to(args.device))
                probs = logits.softmax(-1)
                preds, _ = m.tokenizer.decode(probs)
                return m.charset_adapter(preds[0]).strip()

        correct = correct_single = pw_used = pw_better = pw_worse = 0
        sp_total = sp_correct = 0
        tot_ed = tot_len = 0
        for s in samples:
            img = Image.open(os.path.join(args.test_dir, s['image_filename'])).convert('RGB')
            gt = s['expected_text'].strip()
            sv = single(img)
            final = sv
            # piecewise only when the single-view prediction implies the crop
            # is below the readability floor (<5 px/char at model input)
            if len(sv) >= 4 and 128.0 / len(sv) < 5.0 and img.size[0] >= args.width_thresh:
                pw = piecewise_recognise(m, img, tr)
                if pw is not None:
                    pw_used += 1
                    final = pw
            if any(c in SPECIAL for c in gt):
                sp_total += 1
                sp_correct += (final == gt)
            correct += (final == gt)
            correct_single += (sv == gt)
            tot_ed += levenshtein(gt, final)
            tot_len += max(len(gt), 1)

        print(f'\n{slug}:')
        print(f'  single-view: {correct_single}/{len(samples)} '
              f'({100 * correct_single / len(samples):.2f}%)')
        print(f'  + piecewise: {correct}/{len(samples)} '
              f'({100 * correct / len(samples):.2f}%)   '
              f'characc {100 * (1 - tot_ed / tot_len):.2f}%')
        print(f'  piecewise applied: {pw_used} crops   specials: {sp_correct}/{sp_total}')
        print(f'  net vs single: {correct - correct_single:+d} words')


if __name__ == '__main__':
    main()
