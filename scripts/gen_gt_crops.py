"""Generate line crops with word# tags for ground-truth transcription, and persist model predictions."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from IndicPhotoOCR.ocr import OCR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_comparison import order_like_detect_para, load_num_font


def bbox_xyxy(bbox):
    pts = np.array(bbox, np.int32)
    return [int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max())]


def cluster_lines(detections):
    """Cluster boxes into visual lines by y-overlap; returns list of det_idx lists."""
    boxes = [bbox_xyxy(d) for d in detections]
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    lines = []
    for i in order:
        placed = False
        for line in lines:
            ly1 = min(boxes[j][1] for j in line)
            ly2 = max(boxes[j][3] for j in line)
            ov = max(0, min(ly2, boxes[i][3]) - max(ly1, boxes[i][1]))
            h = min(ly2 - ly1, boxes[i][3] - boxes[i][1])
            if h > 0 and ov / h > 0.4:
                line.append(i)
                placed = True
                break
        if not placed:
            lines.append([i])
    for line in lines:
        line.sort(key=lambda i: boxes[i][0])
    return lines


def make_line_crop(image_pil, detections, line_idxs, word_nos, scale=2.0,
                   num_size=22, pad=6):
    """Crop a line region at `scale` with boxes + word# tags drawn."""
    boxes = [bbox_xyxy(detections[i]) for i in line_idxs]
    x1 = max(0, min(b[0] for b in boxes) - pad)
    y1 = max(0, min(b[1] for b in boxes) - pad)
    x2 = min(image_pil.size[0], max(b[2] for b in boxes) + pad)
    y2 = min(image_pil.size[1], max(b[3] for b in boxes) + pad)

    crop = image_pil.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(crop)
    font = load_num_font(num_size)

    for i, b in zip(line_idxs, boxes):
        bx1 = int((b[0] - x1) * scale)
        by1 = int((b[1] - y1) * scale)
        bx2 = int((b[2] - x1) * scale)
        by2 = int((b[3] - y1) * scale)
        draw.rectangle([bx1, by1, bx2, by2], outline=(0, 180, 0), width=2)
        label = str(word_nos[i])
        tb = font.getbbox(label)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, by1 - th - 6)
        draw.rectangle([bx1, ty, bx1 + tw + 4, ty + th + 4], fill=(255, 255, 0))
        draw.text((bx1 + 2, ty + 1), label, font=font, fill=(0, 0, 0))
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/tmp/test_marathi2.png')
    ap.add_argument('--original', default='IndicPhotoOCR/recognition/models/marathi.ckpt')
    ap.add_argument('--stage1', default='data/marathi_stage1.ckpt')
    ap.add_argument('--stage2', default='data/marathi_finetuned.ckpt')
    ap.add_argument('--outdir', default='/tmp/gt_lines')
    ap.add_argument('--lines-per-batch', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ocr = OCR(device='mps', identifier_lang='auto', verbose=False)
    image = cv2.imread(args.image)
    image_pil = Image.open(args.image).convert('RGB')

    print('Detecting...')
    detections = ocr.detect(args.image)
    print(f'  {len(detections)} boxes')

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

    order = order_like_detect_para(detections, langs)
    word_nos = {det_idx: word_no for word_no, det_idx in enumerate(order)}

    lines = cluster_lines(detections)
    print(f'  {len(lines)} visual lines')

    # Save individual line crops + mapping
    mapping = []
    for li, line_idxs in enumerate(lines):
        crop = make_line_crop(image_pil, detections, line_idxs, word_nos)
        p = os.path.join(args.outdir, f'line_{li:02d}.png')
        crop.save(p)
        mapping.append({
            'line': li,
            'word_nos': [word_nos[i] for i in line_idxs],
            'det_idxs': line_idxs,
        })
    with open(os.path.join(args.outdir, 'mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=1)

    # Stack into batches
    n_per = args.lines_per_batch
    batches = [lines[i:i + n_per] for i in range(0, len(lines), n_per)]
    for bi, batch in enumerate(batches):
        crops = []
        for li, line_idxs in enumerate(batch):
            g_li = bi * n_per + li
            crops.append(Image.open(os.path.join(args.outdir, f'line_{g_li:02d}.png')))
        w = max(c.size[0] for c in crops)
        gap = 26
        label_font = load_num_font(26)
        total_h = sum(c.size[1] for c in crops) + gap * (len(crops) + 1) + 40
        canvas = Image.new('RGB', (w + 40, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        y = gap
        for g_li, c in zip(range(bi * n_per, bi * n_per + len(crops)), crops):
            draw.text((10, y), f'L{g_li}', font=label_font, fill=(200, 0, 0))
            canvas.paste(c, (36, y))
            y += c.size[1] + gap
        bp = os.path.join(args.outdir, f'batch_{bi}.png')
        canvas.save(bp)
        print(f'  saved {bp} ({len(batch)} lines)')

    # Persist predictions for all models
    models = [('original', args.original), ('stage1', args.stage1), ('stage2', args.stage2)]
    preds = {}
    for slug, ckpt in models:
        if not os.path.exists(ckpt):
            continue
        print(f'Recognizing with {slug}...')
        texts, confs = [], []
        for i, bbox in enumerate(detections):
            cp = ocr.crop_bbox(image, bbox)
            try:
                r = ocr.recognise(cp, langs[i], checkpoint=ckpt, return_confidence=True)
                if isinstance(r, tuple) and len(r) == 2:
                    t, c = r
                else:
                    t, c = r, None
                texts.append(t if isinstance(t, str) else str(t))
                confs.append(float(c) if c is not None else None)
            except Exception:
                texts.append('')
                confs.append(None)
            finally:
                if os.path.exists(cp):
                    os.remove(cp)
        preds[slug] = {'texts': texts, 'confs': confs}

    with open(os.path.join(args.outdir, 'predictions.json'), 'w', encoding='utf-8') as f:
        json.dump({'word_nos': word_nos, 'order': order,
                   'preds': preds}, f, ensure_ascii=False, indent=1)
    print(f"Predictions saved for: {list(preds)}")
    print('Next: transcribe batch images, then write ground_truth.jsonl')


if __name__ == '__main__':
    main()
