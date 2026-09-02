"""Production-model visualization: stage5 + TTA on the Marathi test screenshot.

Produces (in ~/Desktop/marathi_ocr_visual/):
  - annotated_stage5_tta.png   : screenshot with green (correct) / red (wrong)
                                 boxes + word# tags (user's GT numbering)
  - gt_check_production.png    : rendered table word# | GT | stage5+TTA | ok
  - scores_table_production.txt: text table + failure summary

Everything runs on CPU (deterministic). Box->word# mapping and per-box GT come
from the fingerprint-aligned dumps; alignment is re-verified against fresh
original-model predictions before drawing.
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
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from IndicPhotoOCR.ocr import OCR
from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
from eval_tta import tta_recognise
from score_deterministic import levenshtein

DEV_FONT = '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc'
NUM_FONT = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'


def load_dev_font(size):
    return ImageFont.truetype(DEV_FONT, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/tmp/test_marathi2.png')
    ap.add_argument('--stage5', default='data/marathi_stage5.ckpt')
    ap.add_argument('--original', default='IndicPhotoOCR/recognition/models/marathi.ckpt')
    ap.add_argument('--eval-dump', default='/tmp/gt_lines/eval_dump.json')
    ap.add_argument('--tta-dump', default='/tmp/gt_lines/tta_dump.json')
    ap.add_argument('--wordno-map', default='/tmp/gt_lines/box_to_wordno.json')
    ap.add_argument('--outdir', default=os.path.expanduser('~/Desktop/marathi_ocr_visual'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load dumps -------------------------------------------------------
    eval_dump = json.load(open(args.eval_dump))
    tta_dump = json.load(open(args.tta_dump))
    wordno_map = {int(k): v for k, v in json.load(open(args.wordno_map)).items()}
    gt_at_box = {int(k): v for k, v in eval_dump['gt_at_box'].items()}
    dump_original = eval_dump['preds']['original']
    dump_tta = {int(k): v for k, v in tta_dump['tta'].items()}

    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)
    image = cv2.imread(args.image)
    print('Detecting (CPU)...')
    detections = ocr.detect(args.image)
    n = len(detections)
    print(f'  {n} boxes (dump had {len(dump_original)})')

    def crop_for(bbox):
        pts = np.array(bbox, np.int32)
        x1, y1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
        x2 = min(image.shape[1], pts[:, 0].max())
        y2 = min(image.shape[0], pts[:, 1].max())
        return Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

    # ---- Fresh predictions ------------------------------------------------
    def load_model(ckpt):
        m = load_from_checkpoint(ckpt).eval().to(args.device)
        tr = T.Compose([T.Resize(m.hparams.img_size, T.InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(0.5, 0.5)])
        return m, tr

    model_orig, tr_orig = load_model(args.original)
    model_s5, tr_s5 = load_model(args.stage5)

    def single(model, tr, img):
        with torch.no_grad():
            logits = model(tr(img).unsqueeze(0).to(args.device))
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            return model.charset_adapter(preds[0]).strip()

    fresh_original, fresh_tta = [], []
    for i, bbox in enumerate(detections):
        img = crop_for(bbox)
        fresh_original.append(single(model_orig, tr_orig, img))
        fresh_tta.append(tta_recognise(model_s5, args.device, img, tr_s5).strip())
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{n}')

    # ---- Alignment verification ------------------------------------------
    if n == len(dump_original):
        same = sum(1 for i in range(n) if fresh_original[i] == dump_original[i])
        print(f'alignment check: fresh original == dump original on {same}/{n}')
        same_t = sum(1 for i in range(min(n, len(dump_tta)))
                     if i in dump_tta and fresh_tta[i] == dump_tta[i])
        print(f'alignment check: fresh stage5+TTA == dump TTA on {same_t}/{len(dump_tta)}')
    else:
        print('WARNING: box count differs from dump - GT assignment may be off!')

    # ---- Scores -----------------------------------------------------------
    covered = sorted(gt_at_box)
    ex_orig = sum(1 for b in covered if fresh_original[b] == gt_at_box[b])
    ex_tta = sum(1 for b in covered if fresh_tta[b] == gt_at_box[b])
    ed = sum(levenshtein(gt_at_box[b], fresh_tta[b]) for b in covered)
    ln = sum(max(len(gt_at_box[b]), 1) for b in covered)
    print(f'\noriginal:      {ex_orig}/{len(covered)} ({100 * ex_orig / len(covered):.2f}%)')
    print(f'stage5 + TTA:  {ex_tta}/{len(covered)} ({100 * ex_tta / len(covered):.2f}%)'
          f'  characc {100 * (1 - ed / ln):.2f}%')

    # ---- 1. Annotated image ------------------------------------------------
    print('\nDrawing annotated image...')
    scale = 2.0
    base = Image.open(args.image).convert('RGB')
    w, h = base.size
    img = base.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font_num = ImageFont.truetype(NUM_FONT, 24)

    n_green = n_red = n_gray = 0
    for i, bbox in enumerate(detections):
        pts = np.array(bbox, np.int32)
        x1 = int(pts[:, 0].min() * scale); y1 = int(pts[:, 1].min() * scale)
        x2 = int(pts[:, 0].max() * scale); y2 = int(pts[:, 1].max() * scale)
        if i in gt_at_box:
            ok = fresh_tta[i] == gt_at_box[i]
            color = (0, 200, 0) if ok else (255, 0, 0)
            n_green += ok; n_red += (not ok)
        else:
            color = (160, 160, 160)
            n_gray += 1
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        wn = wordno_map.get(i)
        if wn is not None:
            label = str(wn)
            tb = font_num.getbbox(label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 6)
            draw.rectangle([x1, ty, x1 + tw + 5, ty + th + 4], fill=(255, 255, 0))
            draw.text((x1 + 2, ty), label, font=font_num, fill=(0, 0, 0))

    # Legend strip at the bottom
    leg_h = 120
    canvas = Image.new('RGB', (img.size[0], img.size[1] + leg_h), (245, 245, 245))
    canvas.paste(img, (0, 0))
    d2 = ImageDraw.Draw(canvas)
    lf = ImageFont.truetype(NUM_FONT, 30)
    d2.text((30, img.size[1] + 10),
            f'stage5 + TTA (production)   green = correct ({n_green})   '
            f'red = wrong ({n_red})   gray = no GT ({n_gray})   '
            f'exact {ex_tta}/{len(covered)} = {100 * ex_tta / len(covered):.1f}%',
            font=lf, fill=(0, 0, 0))
    out1 = os.path.join(args.outdir, 'annotated_stage5_tta.png')
    canvas.save(out1)
    print(f'  saved {out1}')

    # ---- 2. Scores table (text) --------------------------------------------
    rows = []
    for b in covered:
        wn = wordno_map.get(b, -1)
        rows.append((wn, b, gt_at_box[b], fresh_original[b], fresh_tta[b]))
    rows.sort()

    out2 = os.path.join(args.outdir, 'scores_table_production.txt')
    with open(out2, 'w', encoding='utf-8') as f:
        f.write('PRODUCTION SCORES: stage5 + TTA vs GT (CPU, deterministic)\n')
        f.write(f'Image: {args.image}   GT words: {len(covered)}\n')
        f.write(f'original: {ex_orig}/{len(covered)} = {100 * ex_orig / len(covered):.2f}%   '
                f'stage5+TTA: {ex_tta}/{len(covered)} = {100 * ex_tta / len(covered):.2f}%   '
                f'characc: {100 * (1 - ed / ln):.2f}%\n')
        f.write('=' * 110 + '\n')
        f.write(f'{"word#":>5} {"box":>4}  {"GT":<28} {"original":<28} '
                f'{"stage5+TTA":<28} verdict\n')
        f.write('-' * 110 + '\n')
        for wn, b, g, o, t in rows:
            v = 'OK' if t == g else 'FAIL'
            f.write(f'{wn:>5} {b:>4}  {g:<28} {o:<28} {t:<28} {v}\n')
        f.write('=' * 110 + '\n')
        f.write(f'\nSTAGE5+TTA FAILURES ({n_red}):\n')
        for wn, b, g, o, t in rows:
            if t != g:
                f.write(f'  word#{wn} (box {b}): GT={g!r}\n')
                f.write(f'               TTA={t!r}\n')
                if o == g:
                    f.write(f'               (original had this right!)\n')
    print(f'  saved {out2}')

    # ---- 3. Rendered GT check table (PNG) ----------------------------------
    print('Rendering GT check table image...')
    f_dev = load_dev_font(30)
    f_num = ImageFont.truetype(NUM_FONT, 26)

    row_h, col_w = 52, 1000
    half = (len(rows) + 1) // 2
    table_h = (half + 2) * row_h + 40
    table = Image.new('RGB', (col_w * 2 + 30, table_h), (255, 255, 255))
    td = ImageDraw.Draw(table)

    def draw_cell(x, y, txt, color=(0, 0, 0), font=None, dev=False):
        td.text((x, y), txt, font=(font or f_num), fill=color)

    # header
    for cx, title in [(0, 'PRODUCTION CHECK: stage5 + TTA vs GT'),
                      (col_w + 30, '(continued)')]:
        td.text((cx + 20, 15), title, font=f_num, fill=(60, 60, 60))
    hdr_y = 60
    for cx in (0, col_w + 30):
        draw_cell(cx + 20, hdr_y, 'word#', (100, 100, 100))
        draw_cell(cx + 110, hdr_y, 'GT', (100, 100, 100))
        draw_cell(cx + 500, hdr_y, 'stage5+TTA', (100, 100, 100))
        draw_cell(cx + 890, hdr_y, 'ok', (100, 100, 100))

    for r_i, (wn, b, g, o, t) in enumerate(rows):
        col = 0 if r_i < half else 1
        row = r_i if col == 0 else r_i - half
        cx = 0 if col == 0 else col_w + 30
        y = hdr_y + row_h * (row + 1)
        if row % 2 == 0:
            td.rectangle([cx, y, cx + col_w, y + row_h - 4], fill=(246, 246, 250))
        ok = (t == g)
        draw_cell(cx + 20, y + 8, str(wn))
        draw_cell(cx + 110, y + 8, g, (0, 0, 0), f_dev)
        draw_cell(cx + 500, y + 8, t, (0, 130, 0) if ok else (200, 0, 0), f_dev)
        draw_cell(cx + 890, y + 8, 'YES' if ok else 'NO',
                  (0, 130, 0) if ok else (200, 0, 0))

    out3 = os.path.join(args.outdir, 'gt_check_production.png')
    table.save(out3)
    print(f'  saved {out3}')

    print(f'\nDone. exact: {ex_tta}/{len(covered)} ({100 * ex_tta / len(covered):.2f}%)')


if __name__ == '__main__':
    main()
