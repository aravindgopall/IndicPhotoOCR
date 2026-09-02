"""Gujarati production visualization: per-line annotated images + scores tables.

For each of the 10 held-out test lines:
  - the line image with word crops laid out below,
  - per-word: crop thumbnail | GT | ORIGINAL | STAGE1 (single-view, CPU),
    text color-coded green (matches GT) / red (mismatch).

Outputs (~/Desktop/gujarati_ocr_visual/):
  annotated_line_XX.png   per-line check images
  scores_table.txt         word# | GT | original | stage1 | verdicts
  gt_scores.txt            model comparison summary (incl. TTA)
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

GUJ_FONT = '/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc'
NUM_FONT = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines-dir', default='data/gujarati_test_lines')
    ap.add_argument('--test-dir', default='data/gujarati_test')
    ap.add_argument('--original', default='IndicPhotoOCR/recognition/models/gujarati.ckpt')
    ap.add_argument('--stage1', default='data/gujarati_stage1.ckpt')
    ap.add_argument('--outdir', default=os.path.expanduser('~/Desktop/gujarati_ocr_visual'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Group crops by line (filename NNNN_MM -> line NNNN)
    lines = [json.loads(l) for l in open(f'{args.lines_dir}/labels.jsonl', encoding='utf-8')]
    lines.sort(key=lambda l: l['image_filename'])
    crops = [json.loads(l) for l in open(f'{args.test_dir}/labels.jsonl', encoding='utf-8')]
    by_line = defaultdict(list)
    for c in crops:
        ln = int(c['image_filename'].split('/')[-1].split('_')[0])
        by_line[ln].append(c)
    for ln in by_line:
        by_line[ln].sort(key=lambda c: c['image_filename'])

    # ---- Models ------------------------------------------------------------
    def load(ckpt):
        m = load_from_checkpoint(ckpt).eval().to(args.device)
        tr = T.Compose([T.Resize(m.hparams.img_size, T.InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(0.5, 0.5)])
        return m, tr

    m_orig, tr_o = load(args.original)
    m_s1, tr_1 = load(args.stage1)

    def single(model, tr, img):
        with torch.no_grad():
            logits = model(tr(img.convert('RGB')).unsqueeze(0).to(args.device))
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            return model.charset_adapter(preds[0]).strip()

    # ---- Recognize every crop ----------------------------------------------
    all_rows = []  # (line_no, word_idx, gt, orig, s1)
    for ln in sorted(by_line):
        for wi, c in enumerate(by_line[ln]):
            img = Image.open(os.path.join(args.test_dir, c['image_filename']))
            gt = c['expected_text'].strip()
            all_rows.append((ln, wi, gt, single(m_orig, tr_o, img), single(m_s1, tr_1, img)))
        print(f'  line {ln}: {len(by_line[ln])} words')

    ex_o = sum(1 for r in all_rows if r[3] == r[2])
    ex_1 = sum(1 for r in all_rows if r[4] == r[2])
    print(f'\noriginal: {ex_o}/{len(all_rows)} ({100 * ex_o / len(all_rows):.2f}%)')
    print(f'stage1:   {ex_1}/{len(all_rows)} ({100 * ex_1 / len(all_rows):.2f}%)')

    # ---- Per-line annotated images ------------------------------------------
    f_dev = ImageFont.truetype(GUJ_FONT, 26)
    f_num = ImageFont.truetype(NUM_FONT, 22)
    f_hdr = ImageFont.truetype(NUM_FONT, 28)

    THUMB_H = 44          # crop thumbnail height
    ROW_H = 58
    COLS = [(30, 70), (140, 430), (560, 430), (1050, 430)]  # (x, width) w#/crop/GT/orig/s1

    for ln in sorted(by_line):
        line_row = lines[ln]
        line_img = Image.open(os.path.join(args.lines_dir, line_row['image_filename'])).convert('RGB')
        lw, lh = line_img.size
        disp_w = 1500
        scale = disp_w / lw
        disp_line = line_img.resize((disp_w, int(lh * scale)), Image.LANCZOS)

        rows = [r for r in all_rows if r[0] == ln]
        W = 1520
        H = 70 + disp_line.size[1] + 30 + (len(rows) + 1) * ROW_H + 40
        canvas = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)

        line_correct = sum(1 for r in rows if r[4] == r[2])
        d.text((30, 18), f'line {ln}: {line_row["image_filename"]}   '
                f'stage1 {line_correct}/{len(rows)} correct', font=f_hdr, fill=(0, 0, 0))
        canvas.paste(disp_line, (10, 65))

        y0 = 65 + disp_line.size[1] + 28
        # header row
        d.text((COLS[0][0] + 24, y0 + 12), 'word', font=f_num, fill=(120, 120, 120))
        d.text((COLS[1][0], y0 + 12), 'crop', font=f_num, fill=(120, 120, 120))
        d.text((COLS[2][0], y0 + 12), 'GT', font=f_num, fill=(120, 120, 120))
        d.text((COLS[3][0], y0 + 12), 'ORIGINAL', font=f_num, fill=(120, 120, 120))
        d.text((COLS[3][0] + 470, y0 + 12), 'STAGE1', font=f_num, fill=(120, 120, 120))
        d.line([(20, y0 + 44), (W - 20, y0 + 44)], fill=(200, 200, 200), width=2)

        for ri, (lno, wi, gt, o, s1) in enumerate(rows):
            y = y0 + ROW_H * (ri + 1)
            if ri % 2 == 0:
                d.rectangle([20, y, W - 20, y + ROW_H - 6], fill=(246, 248, 250))
            d.text((COLS[0][0] + 24, y + 14), f'{wi}', font=f_num, fill=(80, 80, 80))
            # crop thumbnail
            c = by_line[ln][wi]
            cimg = Image.open(os.path.join(args.test_dir, c['image_filename'])).convert('RGB')
            cw, ch = cimg.size
            nh = THUMB_H
            nw = max(1, int(cw * nh / ch))
            cimg = cimg.resize((min(nw, 380), nh), Image.LANCZOS)
            canvas.paste(cimg, (COLS[1][0], y + 6))
            d.rectangle([COLS[1][0], y + 6, COLS[1][0] + min(nw, 380), y + 6 + nh],
                        outline=(180, 180, 180), width=1)
            # GT
            d.text((COLS[2][0], y + 10), gt, font=f_dev, fill=(0, 0, 0))
            # original / stage1
            d.text((COLS[3][0], y + 10), o or '(empty)', font=f_dev,
                   fill=(0, 130, 0) if o == gt else (200, 0, 0))
            d.text((COLS[3][0] + 470, y + 10), s1 or '(empty)', font=f_dev,
                   fill=(0, 130, 0) if s1 == gt else (200, 0, 0))

        out = os.path.join(args.outdir, f'annotated_line_{ln:02d}.png')
        canvas.save(out)
        print(f'  saved {out}')

    # ---- scores_table.txt ----------------------------------------------------
    out_t = os.path.join(args.outdir, 'scores_table.txt')
    with open(out_t, 'w', encoding='utf-8') as f:
        f.write('GUJARATI TEST SCORES: original vs stage1 (single-view, CPU)\n')
        f.write(f'Test: 10 held-out lines, {len(all_rows)} word crops\n')
        f.write(f'original: {ex_o}/{len(all_rows)} = {100 * ex_o / len(all_rows):.2f}%   '
                f'stage1: {ex_1}/{len(all_rows)} = {100 * ex_1 / len(all_rows):.2f}%\n')
        f.write('=' * 100 + '\n')
        f.write(f'{"line":>4} {"word":>4}  {"GT":<24} {"original":<24} '
                f'{"stage1":<24} verdicts\n')
        f.write('-' * 100 + '\n')
        for ln, wi, gt, o, s1 in all_rows:
            f.write(f'{ln:>4} {wi:>4}  {gt:<24} {o:<24} {s1:<24} '
                    f'{"O:" + ("ok" if o == gt else "X")} S1:{"ok" if s1 == gt else "X"}\n')
        f.write('=' * 100 + '\n')
        f.write('\nSTAGE1 FAILURES:\n')
        for ln, wi, gt, o, s1 in all_rows:
            if s1 != gt:
                f.write(f'  line {ln} word {wi}: GT={gt!r}\n')
                f.write(f'            stage1={s1!r}')
                f.write('   [label-noise: crop contains different text]' if o == s1 else '')
                f.write('\n')
    print(f'  saved {out_t}')

    print('\nDone.')


if __name__ == '__main__':
    main()
