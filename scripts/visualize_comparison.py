"""Generate annotated comparison images: boxes + word# tags (matching SIDE-BY-SIDE table).

Word numbers replicate detect_para ordering (line grouping by vertical overlap,
then x-sort within line) so they match the word# column in the comparison txt.
Boxes are color-coded by recognition output:
  green  = pure Devanagari
  orange = contains special characters (-,./:()% etc.)
  blue   = contains Latin letters
  red    = empty prediction
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from IndicPhotoOCR.ocr import OCR

SPECIAL = set('()-,./:;%|—""?!&')


def order_like_detect_para(detections, langs):
    """Replicate detect_para() with the batch path's language-grouped insertion order.

    The batch path builds recognized_texts grouped by identified language
    (first-seen language order, detection order within each language), and
    detect_para processes items in that insertion order. Returns a list where
    element[i] = det_idx of word# i.
    """
    # Language grouping (first-seen order), detection order within language
    langs_to_ids = {}
    for i, lang in enumerate(langs):
        langs_to_ids.setdefault(lang, []).append(i)
    insertion_order = [i for ids in langs_to_ids.values() for i in ids]

    bboxes = []
    for i in insertion_order:
        pts = np.array(detections[i], np.int32)
        bboxes.append([int(pts[:, 0].min()), int(pts[:, 1].min()),
                       int(pts[:, 0].max()), int(pts[:, 1].max())])

    def overlap(b1, b2):
        ov = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        h = min(b1[3] - b1[1], b2[3] - b2[1])
        return ov / h if h > 0 else 0

    items = list(zip(insertion_order, bboxes))
    lines = []
    while items:
        cur = items.pop(0)
        cur_line = [cur]
        remaining = []
        for it in items:
            if overlap(cur[1], it[1]) > 0.4:
                cur_line.append(it)
            else:
                remaining.append(it)
        items = remaining
        lines.append(sorted(cur_line, key=lambda x: x[1][0]))

    return [e[0] for line in lines for e in line]


def box_color(text):
    t = (text or '').strip()
    if not t:
        return (255, 0, 0)
    if any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in t):
        return (0, 120, 255)
    if any(c in SPECIAL for c in t):
        return (255, 140, 0)
    return (0, 180, 0)


def load_num_font(size):
    for p in ('/System/Library/Fonts/Supplemental/Arial Bold.ttf',
              '/System/Library/Fonts/Helvetica.ttc',
              '/System/Library/Fonts/Supplemental/Arial.ttf'):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default(size)


def conf_bg(conf):
    """Tag background color by confidence."""
    if conf is None:
        return (200, 200, 200)
    if conf >= 0.75:
        return (144, 238, 144)   # light green
    if conf >= 0.5:
        return (255, 255, 0)     # yellow
    return (255, 160, 160)       # light red


def annotate(image_path, detections, word_nos, texts, save_path, scale=2.0,
             num_size=26, color_code=True, confs=None):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = load_num_font(num_size)

    for det_idx, word_no in word_nos.items():
        pts = np.array(detections[det_idx], np.int32)
        x_min = int(pts[:, 0].min() * scale)
        y_min = int(pts[:, 1].min() * scale)
        x_max = int(pts[:, 0].max() * scale)
        y_max = int(pts[:, 1].max() * scale)

        color = box_color(texts[det_idx]) if color_code else (0, 180, 0)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

        if confs is not None:
            c = confs[det_idx]
            label = f'{word_no}|{c:.2f}' if c is not None else f'{word_no}|?'
            bg = conf_bg(c)
        else:
            label = str(word_no)
            bg = (255, 255, 0)
        tb = font.getbbox(label)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y_min - th - 8)
        draw.rectangle([x_min, ty, x_min + tw + 6, ty + th + 5], fill=bg)
        draw.text((x_min + 3, ty + 1), label, font=font, fill=(0, 0, 0))

    img.save(save_path)
    return img


def composite(images_with_titles, save_path):
    imgs = [im for _, im in images_with_titles]
    h = max(im.size[1] for im in imgs)
    bar_h = 70
    total_w = sum(im.size[0] for im in imgs) + 20 * (len(imgs) + 1)
    canvas = Image.new('RGB', (total_w, h + bar_h + 20), (30, 30, 30))
    title_font = load_num_font(40)
    draw = ImageDraw.Draw(canvas)

    x = 20
    for title, im in images_with_titles:
        draw.text((x + 10, 18), title, font=title_font, fill=(255, 255, 255))
        canvas.paste(im, (x, bar_h))
        x += im.size[0] + 20

    # Legend
    leg_font = load_num_font(30)
    lx = 30
    for color, label in [((0, 180, 0), 'Devanagari'),
                         ((255, 140, 0), 'special chars'),
                         ((0, 120, 255), 'Latin'),
                         ((255, 0, 0), 'empty')]:
        draw.rectangle([lx, h + bar_h + 30, lx + 30, h + bar_h + 60], fill=color)
        draw.text((lx + 38, h + bar_h + 32), label, font=leg_font, fill=(255, 255, 255))
        lx += 38 + 260
    lx2 = 30
    for color, label in [((144, 238, 144), 'conf >= 0.75'),
                         ((255, 255, 0), 'conf 0.50-0.74'),
                         ((255, 160, 160), 'conf < 0.50')]:
        draw.rectangle([lx2, h + bar_h + 75, lx2 + 30, h + bar_h + 105], fill=color)
        draw.text((lx2 + 38, h + bar_h + 77), label, font=leg_font, fill=(255, 255, 255))
        lx2 += 38 + 260
    canvas.save(save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/tmp/test_marathi2.png')
    ap.add_argument('--original', default='IndicPhotoOCR/recognition/models/marathi.ckpt')
    ap.add_argument('--stage1', default='data/marathi_stage1.ckpt')
    ap.add_argument('--stage2', default='data/marathi_finetuned.ckpt')
    ap.add_argument('--outdir', default=os.path.expanduser('~/Desktop/marathi_ocr_visual'))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ocr = OCR(device='mps', identifier_lang='auto', verbose=False)
    image = cv2.imread(args.image)

    print('Running detection...')
    detections = ocr.detect(args.image)
    print(f'  {len(detections)} boxes detected')

    # Identify script once (shared across models)
    print('Identifying scripts...')
    langs = []
    for i, bbox in enumerate(detections):
        cp = ocr.crop_bbox(image, bbox)
        try:
            langs.append(ocr.identifier.identify(cp, 'auto', ocr._pipeline_device))
        except Exception:
            langs.append(None)
        finally:
            if os.path.exists(cp):
                os.remove(cp)
        if (i + 1) % 50 == 0:
            print(f'    {i + 1}/{len(detections)}')

    # word# ordering matching the SIDE-BY-SIDE table
    order = order_like_detect_para(detections, langs)
    word_nos = {det_idx: word_no for word_no, det_idx in enumerate(order)}
    print(f'  word# 0..{len(order) - 1} assigned (detect_para order)')

    models = [
        ('original', 'ORIGINAL', args.original),
        ('stage1', 'STAGE 1 (real data)', args.stage1),
        ('stage2', 'STAGE 2 (real+synthetic)', args.stage2),
    ]

    annotated = []
    all_model_data = {}
    for slug, name, ckpt in models:
        if not os.path.exists(ckpt):
            print(f'  skipping {name}: {ckpt} not found')
            continue
        print(f'Recognizing with {name}...')
        texts = []
        confs = []
        for i, bbox in enumerate(detections):
            cropped_path = ocr.crop_bbox(image, bbox)
            try:
                result = ocr.recognise(cropped_path, langs[i], checkpoint=ckpt,
                                       return_confidence=True)
                if isinstance(result, tuple) and len(result) == 2:
                    text, conf = result
                else:
                    text, conf = result, None
                texts.append(text if isinstance(text, str) else str(text))
                confs.append(float(conf) if conf is not None else None)
            except Exception:
                texts.append('')
                confs.append(None)
            finally:
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)
            if (i + 1) % 50 == 0:
                print(f'    {i + 1}/{len(detections)}')

        all_model_data[slug] = (name, texts, confs)
        out_path = os.path.join(args.outdir, f'annotated_{slug}.png')
        im = annotate(args.image, detections, word_nos, texts, out_path, confs=confs)
        annotated.append((name, im))
        print(f'  saved {out_path}')

    # Plain reference (no color coding) for box locations
    ref_path = os.path.join(args.outdir, 'annotated_reference.png')
    annotate(args.image, detections, word_nos,
             [''] * len(detections), ref_path, color_code=False)
    print(f'  saved {ref_path} (plain reference, green boxes)')

    if annotated:
        comp_path = os.path.join(args.outdir, 'comparison_side_by_side.png')
        composite(annotated, comp_path)
        print(f'\nSide-by-side composite: {comp_path}')

    # Scores table: word# | text [conf] for each model
    if all_model_data:
        scores_path = os.path.join(args.outdir, 'scores_table.txt')
        slugs = [s for s, _, _ in models if s in all_model_data]
        with open(scores_path, 'w', encoding='utf-8') as f:
            f.write('Word-level scores (word# matches SIDE-BY-SIDE table and image tags)\n')
            f.write(f'Image: {args.image}   Words: {len(order)}\n')
            f.write('=' * 130 + '\n')
            header = 'word# | ' + ' | '.join(
                f'{all_model_data[s][0]:<40}' for s in slugs)
            f.write(header + '\n')
            f.write('-' * 130 + '\n')
            for word_no, det_idx in enumerate(order):
                cells = []
                for s in slugs:
                    _, texts, confs = all_model_data[s]
                    t = texts[det_idx].strip()
                    c = confs[det_idx]
                    cs = f'{c:.2f}' if c is not None else '  ? '
                    cells.append(f'{t} [{cs}]'.ljust(40)[:40])
                f.write(f'{word_no:5d} | ' + ' | '.join(cells) + '\n')
            # Summary
            f.write('=' * 130 + '\n\nCONFIDENCE SUMMARY\n')
            for s in slugs:
                name, texts, confs = all_model_data[s]
                valid = [c for c in confs if c is not None]
                if valid:
                    mean_c = sum(valid) / len(valid)
                    hi = sum(1 for c in valid if c >= 0.75)
                    mid = sum(1 for c in valid if 0.5 <= c < 0.75)
                    lo = sum(1 for c in valid if c < 0.5)
                    f.write(f'  {name}:\n')
                    f.write(f'    mean confidence: {mean_c:.4f}\n')
                    f.write(f'    conf >= 0.75: {hi}   0.50-0.74: {mid}   < 0.50: {lo}'
                            f'   (of {len(valid)})\n')
        print(f'Scores table: {scores_path}')


if __name__ == '__main__':
    main()
