"""Generalization test: run all Marathi model stages + TTA on fresh documents.

No GT needed - scores what we can measure without it:
  - special-char emission rate (original strips specials)
  - format-valid words (dates / gov codes / URLs, via score_deterministic regexes)
  - mean confidence
  - majority-vote agreement (how many models agree with the modal prediction)

Outputs a side-by-side per-word table (like the earlier comparison files) and
an annotated production image per doc.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from IndicPhotoOCR.ocr import OCR
from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
from eval_tta import tta_recognise
from score_deterministic import (levenshtein, has_special, DATE_PAT, CODE_PAT,
                                 URL_PAT, has_digit)

MODELS = [
    ('original', 'IndicPhotoOCR/recognition/models/marathi.ckpt'),
    ('stage1', 'data/marathi_stage1.ckpt'),
    ('stage2', 'data/marathi_finetuned.ckpt'),
    ('stage3', 'data/marathi_stage3_final.ckpt'),
    ('stage4', 'data/marathi_stage4.ckpt'),
    ('stage5', 'data/marathi_stage5.ckpt'),
]
NUM_FONT = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
DEV_FONT = '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--outdir', default=os.path.expanduser('~/Desktop/marathi_newdocs_comparison'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)

    loaded = {}
    for slug, ckpt in MODELS:
        if os.path.exists(ckpt):
            m = load_from_checkpoint(ckpt).eval().to(args.device)
            tr = T.Compose([T.Resize(m.hparams.img_size, T.InterpolationMode.BICUBIC),
                            T.ToTensor(), T.Normalize(0.5, 0.5)])
            loaded[slug] = (m, tr)
        else:
            print(f'skipping {slug}: {ckpt} not found')

    def single(slug, img):
        m, tr = loaded[slug]
        with torch.no_grad():
            logits = m(tr(img.convert('RGB')).unsqueeze(0).to(args.device))
            probs = logits.softmax(-1)
            preds, _ = m.tokenizer.decode(probs)
            conf = float(probs.max(-1).values.mean())
            return m.charset_adapter(preds[0]).strip(), conf

    def tta(slug, img):
        m, tr = loaded[slug]
        return tta_recognise(m, args.device, img, tr).strip()

    for doc_idx, image_path in enumerate(args.images, 1):
        print(f'\n===== DOC {doc_idx}: {image_path} =====')
        image = cv2.imread(image_path)
        detections = ocr.detect(image_path)
        n = len(detections)
        print(f'  {n} boxes')

        def crop_for(bbox):
            pts = np.array(bbox, np.int32)
            x1, y1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
            x2 = min(image.shape[1], pts[:, 0].max())
            y2 = min(image.shape[0], pts[:, 1].max())
            return Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

        cols = {slug: [] for slug in loaded}
        cols['stage5+TTA'] = []
        confs = {slug: [] for slug in loaded}
        for i, bbox in enumerate(detections):
            img = crop_for(bbox)
            for slug in loaded:
                t, c = single(slug, img)
                cols[slug].append(t)
                confs[slug].append(c)
            cols['stage5+TTA'].append(tta('stage5', img))
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{n}')

        slugs = list(cols.keys())

        # ------- metrics (GT-free) -------
        print(f'\n  {"model":<12}{"special%":>9}{"dates":>7}{"codes":>7}{"urls":>6}'
              f'{"meanconf":>10}{"avglen":>7}')
        metrics_lines = []
        for slug in slugs:
            texts = cols[slug]
            sp = sum(1 for t in texts if has_special(t))
            dates = sum(1 for t in texts if DATE_PAT.match(t))
            codes = sum(1 for t in texts if CODE_PAT.match(t) and has_digit(t))
            urls = sum(1 for t in texts if URL_PAT.match(t))
            mc = (sum(confs[slug]) / len(confs[slug])) if confs.get(slug) else float('nan')
            al = sum(len(t) for t in texts) / max(len(texts), 1)
            row = (f'  {slug:<12}{100 * sp / n:>8.1f}%{dates:>7}{codes:>7}{urls:>6}'
                   f'{mc:>10.3f}{al:>7.1f}')
            print(row)
            metrics_lines.append(row)

        # majority-vote agreement per word
        agree = []
        for i in range(n):
            votes = Counter(cols[s][i] for s in slugs)
            top = votes.most_common(1)[0]
            agree.append(top[1])
        print(f'  majority-vote: modal prediction backed by >=5/7 models on '
              f'{sum(1 for a in agree if a >= 5)}/{n} words')

        # ------- side-by-side table -------
        out_txt = os.path.join(args.outdir, f'comparison_doc{doc_idx}.txt')
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(f'FRESH DOC {doc_idx} (never used in training/tuning): {image_path}\n')
            f.write(f'Boxes: {n}   Models: {", ".join(slugs)}\n')
            f.write('=' * 150 + '\n')
            f.write('word# | ' + ' | '.join(f'{s:<38}' for s in slugs) + '\n')
            f.write('-' * 150 + '\n')
            for i in range(n):
                cells = [f'{cols[s][i]}'[:36].ljust(38) for s in slugs]
                f.write(f'{i:>5} | ' + ' | '.join(cells) + '\n')
            f.write('=' * 150 + '\n\nMETRICS (no GT - emission/format/confidence)\n')
            f.writelines(metrics_lines)
            f.write(f'\nmajority-vote: >=5/7 models agree on '
                    f'{sum(1 for a in agree if a >= 5)}/{n} words\n')
            f.write('majority prediction per word (consensus):\n')
            for i in range(n):
                votes = Counter(cols[s][i] for s in slugs)
                top, cnt = votes.most_common(1)[0]
                f.write(f'  {i:>4} [{cnt}/7] {top!r}\n')
        print(f'  saved {out_txt}')

        # ------- annotated production image -------
        scale = max(2.0, 1600 / image.shape[1])
        base = Image.open(image_path).convert('RGB')
        w, h = base.size
        im = base.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(NUM_FONT, max(18, int(24 * scale / 2)))
        for i, bbox in enumerate(detections):
            pts = np.array(bbox, np.int32)
            x1 = int(pts[:, 0].min() * scale); y1 = int(pts[:, 1].min() * scale)
            x2 = int(pts[:, 0].max() * scale); y2 = int(pts[:, 1].max() * scale)
            t = cols['stage5+TTA'][i]
            if not t:
                color = (255, 0, 0)
            elif has_special(t):
                color = (255, 140, 0)
            elif any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in t):
                color = (0, 120, 255)
            else:
                color = (0, 180, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = str(i)
            tb = font.getbbox(label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 6)
            draw.rectangle([x1, ty, x1 + tw + 5, ty + th + 4], fill=(255, 255, 0))
            draw.text((x1 + 2, ty), label, font=font, fill=(0, 0, 0))
        out_png = os.path.join(args.outdir, f'annotated_stage5_tta_doc{doc_idx}.png')
        im.save(out_png)
        print(f'  saved {out_png}')

    print('\nDone.')


if __name__ == '__main__':
    main()
