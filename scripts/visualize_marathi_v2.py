"""Marathi v2 test visualization: per-line check images + tables + failure analysis.

For each held-out test line: line image on top, then per-word rows:
crop | GT | ORIGINAL | STAGE5 | FROM_STAGE5, color-coded green/red by match.

Outputs (~/Desktop/marathi_v2_visual/):
  annotated_line_XX.png   per-line check images
  scores_table.txt        every word: line, word#, GT, 3 models, verdicts
  failures_by_category.txt  from_stage5 failures grouped by failure type
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
from score_deterministic import levenshtein

DEV_FONT = '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc'
NUM_FONT = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'

MODELS = [
    ('ORIGINAL', 'IndicPhotoOCR/recognition/models/marathi.ckpt'),
    ('STAGE5', 'data/marathi_stage5.ckpt'),
    ('FROM_S5', 'data/marathi_v2_from_stage5.ckpt'),
]


def categorize(gt):
    """Failure category for analysis."""
    if len(gt) > 18 and '/' in gt:
        return 'long_code'
    if len(gt) <= 2 and any(c in '!-./:;,' for c in gt):
        return 'standalone_punct'
    if all(c in '०१२३४५६७८९0123456789.,-' for c in gt) and any(c.isdigit() or c in '०१२३४५६७८९' for c in gt):
        return 'digits'
    if any(c in '!-./:;(),"\'' for c in gt):
        return 'punctuated_word'
    return 'plain_word'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines-dir', default='data/marathi_v2_test_lines')
    ap.add_argument('--test-dir', default='data/marathi_v2_test')
    ap.add_argument('--outdir', default=os.path.expanduser('~/Desktop/marathi_v2_visual'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    lines = [json.loads(l) for l in open(f'{args.lines_dir}/labels.jsonl', encoding='utf-8')]
    lines.sort(key=lambda l: l['image_filename'])
    crops = [json.loads(l) for l in open(f'{args.test_dir}/labels.jsonl', encoding='utf-8')]
    by_line = defaultdict(list)
    for c in crops:
        ln = int(c['image_filename'].split('/')[-1].split('_')[0])
        by_line[ln].append(c)
    for ln in by_line:
        by_line[ln].sort(key=lambda c: c['image_filename'])

    def load(ckpt):
        m = load_from_checkpoint(ckpt).eval().to(args.device)
        tr = T.Compose([T.Resize(m.hparams.img_size, T.InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(0.5, 0.5)])
        return m, tr

    loaded = [(name, *load(ckpt)) for name, ckpt in MODELS if os.path.exists(ckpt)]

    def single(model, tr, img):
        with torch.no_grad():
            logits = model(tr(img.convert('RGB')).unsqueeze(0).to(args.device))
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            return model.charset_adapter(preds[0]).strip()

    # Recognize everything once
    all_rows = []  # (line_no, word_idx, gt, [preds per model])
    for ln in sorted(by_line):
        for wi, c in enumerate(by_line[ln]):
            img = Image.open(os.path.join(args.test_dir, c['image_filename']))
            gt = c['expected_text'].strip()
            preds = [single(m, tr, img) for _, m, tr in loaded]
            all_rows.append((ln, wi, gt, preds))
        print(f'  line {ln}: {len(by_line[ln])} words')

    # ---- per-line images ----
    f_dev = ImageFont.truetype(DEV_FONT, 24)
    f_num = ImageFont.truetype(NUM_FONT, 20)
    f_hdr = ImageFont.truetype(NUM_FONT, 26)
    THUMB_H = 40
    ROW_H = 54

    for ln in sorted(by_line):
        line_row = lines[ln]
        line_img = Image.open(os.path.join(args.lines_dir, line_row['image_filename'])).convert('RGB')
        lw, lh = line_img.size
        disp_w = 1900
        scale = disp_w / lw
        disp_line = line_img.resize((disp_w, max(1, int(lh * scale))), Image.LANCZOS)

        rows = [r for r in all_rows if r[0] == ln]
        W = 1920
        H = 70 + disp_line.size[1] + 30 + (len(rows) + 1) * ROW_H + 40
        canvas = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)

        n_ok = sum(1 for r in rows if r[3][-1] == r[2])
        d.text((30, 16), f'line {ln}: {line_row["image_filename"]}   '
                f'FROM_S5 {n_ok}/{len(rows)} correct', font=f_hdr, fill=(0, 0, 0))
        canvas.paste(disp_line, (10, 62))

        y0 = 62 + disp_line.size[1] + 26
        d.text((34, y0 + 10), 'w#', font=f_num, fill=(120, 120, 120))
        d.text((110, y0 + 10), 'crop', font=f_num, fill=(120, 120, 120))
        d.text((470, y0 + 10), 'GT', font=f_num, fill=(120, 120, 120))
        xcol = 900
        for name, _, _ in loaded:
            d.text((xcol, y0 + 10), name, font=f_num, fill=(120, 120, 120))
            xcol += 330
        d.line([(20, y0 + 40), (W - 20, y0 + 40)], fill=(200, 200, 200), width=2)

        for ri, (lno, wi, gt, preds) in enumerate(rows):
            y = y0 + ROW_H * (ri + 1)
            if ri % 2 == 0:
                d.rectangle([20, y, W - 20, y + ROW_H - 6], fill=(246, 248, 250))
            d.text((34, y + 12), str(wi), font=f_num, fill=(80, 80, 80))
            c = by_line[ln][wi]
            cimg = Image.open(os.path.join(args.test_dir, c['image_filename'])).convert('RGB')
            cw, ch = cimg.size
            nw = max(1, int(cw * THUMB_H / ch))
            cimg = cimg.resize((min(nw, 340), THUMB_H), Image.LANCZICUBIC if False else Image.LANCZOS)
            canvas.paste(cimg, (110, y + 6))
            d.rectangle([110, y + 6, 110 + min(nw, 340), y + 6 + THUMB_H],
                        outline=(180, 180, 180), width=1)
            # GT (truncate long codes for display width)
            gt_disp = gt if len(gt) <= 30 else gt[:28] + '..'
            d.text((470, y + 8), gt_disp, font=f_dev, fill=(0, 0, 0))
            xcol = 900
            for pi, (name, _, _) in enumerate(loaded):
                p = preds[pi]
                p_disp = p if len(p) <= 26 else p[:24] + '..'
                d.text((xcol, y + 8), p_disp or '(empty)', font=f_dev,
                       fill=(0, 130, 0) if p == gt else (200, 0, 0))
                xcol += 330

        out = os.path.join(args.outdir, f'annotated_line_{ln:02d}.png')
        canvas.save(out)
    print('per-line images saved')

    # ---- scores table ----
    out_t = os.path.join(args.outdir, 'scores_table.txt')
    names = [n for n, _, _ in loaded]
    with open(out_t, 'w', encoding='utf-8') as f:
        f.write('MARATHI V2 TEST — per-word scores (CPU)\n')
        f.write(f'Test: 90 held-out lines, {len(all_rows)} words\n')
        for pi, name in enumerate(names):
            ex = sum(1 for r in all_rows if r[3][pi] == r[2])
            ed = sum(levenshtein(r[2], r[3][pi]) for r in all_rows)
            tl = sum(max(len(r[2]), 1) for r in all_rows)
            f.write(f'  {name:<10} {ex}/{len(all_rows)} = {100 * ex / len(all_rows):.2f}%   '
                    f'characc {100 * (1 - ed / tl):.2f}%\n')
        f.write('=' * 130 + '\n')
        f.write(f'{"line":>4} {"w#":>3}  {"GT":<44} ' +
                ' '.join(f'{n:<28}' for n in names) + ' verdicts\n')
        f.write('-' * 130 + '\n')
        for ln, wi, gt, preds in all_rows:
            cells = ' '.join((p[:26] if p else '(empty)').ljust(28) for p in preds)
            vs = ' '.join('ok' if p == gt else 'X' for p in preds)
            f.write(f'{ln:>4} {wi:>3}  {gt[:42]:<44} {cells} {vs}\n')
    print(f'saved {out_t}')

    # ---- failures by category ----
    out_f = os.path.join(args.outdir, 'failures_by_category.txt')
    cats = defaultdict(list)
    for ln, wi, gt, preds in all_rows:
        if preds[-1] != gt:
            cats[categorize(gt)].append((ln, wi, gt, preds[-1]))
    with open(out_f, 'w', encoding='utf-8') as f:
        total_fail = sum(len(v) for v in cats.values())
        f.write(f'FROM_STAGE5 FAILURES BY CATEGORY (total {total_fail}/{len(all_rows)})\n')
        f.write('=' * 100 + '\n')
        for cat in ['long_code', 'standalone_punct', 'digits', 'punctuated_word', 'plain_word']:
            items = cats.get(cat, [])
            # category size in full test
            size = sum(1 for r in all_rows if categorize(r[2]) == cat)
            f.write(f'\n{cat}: {len(items)}/{size} failed\n')
            f.write('-' * 100 + '\n')
            for ln, wi, gt, p in items[:60]:
                f.write(f'  line {ln:>2} w{wi}: GT={gt!r}\n')
                f.write(f'           pred={p!r}\n')
            if len(items) > 60:
                f.write(f'  ... and {len(items) - 60} more\n')
    print(f'saved {out_f}')
    print('\nCategory summary (from_stage5):')
    for cat in ['long_code', 'standalone_punct', 'digits', 'punctuated_word', 'plain_word']:
        items = cats.get(cat, [])
        size = sum(1 for r in all_rows if categorize(r[2]) == cat)
        if size:
            print(f'  {cat:<18} {len(items):>3}/{size:<4} failed ({100 * len(items) / size:.0f}%)')


if __name__ == '__main__':
    main()
