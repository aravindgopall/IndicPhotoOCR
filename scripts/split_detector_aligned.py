"""Production-aligned split: use the ACTUAL detector's boxes for word crops.

The zero-ink projection splitter merges whole codes into single crops (up to
46 chars) that production never produces - the detector emits fragment-level
boxes (median 6 chars, p90 13). This script:

  1. runs the production detector on each line image
  2. chunks long GT tokens (> max_token_len) at separator boundaries into
     <= chunk_len pieces
  3. aligns text chunks to detected boxes proportionally (cumulative char
     count vs cumulative box width)
  4. saves fragment crops + labels

Usage:
  python scripts/split_detector_aligned.py --lines-dir data/marathi_v2_test_lines \
      --outdir data/marathi_v2_test_det [--device cpu]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from IndicPhotoOCR.ocr import OCR

SEPS = '/.,-'


def chunk_token(token, max_len=15):
    """Split a long token at separator boundaries into <= max_len chunks."""
    if len(token) <= 18:
        return [token]
    chunks = []
    cur = ''
    for ch in token:
        cur += ch
        if ch in SEPS and len(cur) >= max_len - 6:
            chunks.append(cur)
            cur = ''
    if cur:
        if chunks and len(cur) < 3:
            chunks[-1] += cur
        else:
            chunks.append(cur)
    # merge any still-too-long tail pieces at remaining separators
    out = []
    for c in chunks:
        while len(c) > max_len + 8:
            # split at the last separator before max_len
            best = -1
            for i, ch in enumerate(c[:max_len]):
                if ch in SEPS:
                    best = i + 1
            if best <= 0:
                break
            out.append(c[:best])
            c = c[best:]
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines-dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-label-length', type=int, default=30)
    ap.add_argument('--manifest', default=None,
                    help='optional path for per-line manifest jsonl')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lang_dir = os.path.join(args.outdir, 'marathi')
    os.makedirs(lang_dir, exist_ok=True)

    rows = [json.loads(l) for l in
            open(f'{args.lines_dir}/labels.jsonl', encoding='utf-8')]
    rows.sort(key=lambda r: r['image_filename'])

    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)

    n_crops = n_lines = n_matched = n_aligned = n_skipped = 0
    manifest = [] if args.manifest else None
    out_path = os.path.join(args.outdir, 'labels.jsonl')
    with open(out_path, 'w', encoding='utf-8') as out:
        for line_idx, row in enumerate(rows):
            img_path = os.path.join(args.lines_dir, row['image_filename'])
            image = cv2.imread(img_path)
            if image is None:
                continue
            try:
                detections = ocr.detect(img_path)
            except Exception:
                n_skipped += 1
                continue
            if not detections:
                n_skipped += 1
                continue

            boxes = []
            for d in detections:
                pts = np.array(d, np.int32)
                x1, y1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
                x2 = min(image.shape[1], pts[:, 0].max())
                y2 = min(image.shape[0], pts[:, 1].max())
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
            boxes.sort(key=lambda b: b[0])
            if not boxes:
                n_skipped += 1
                continue

            chunks = []
            for tok in row['expected_text'].split():
                chunks.extend(chunk_token(tok))
            chunks = [c for c in chunks if len(c) <= args.max_label_length]
            if not chunks:
                n_skipped += 1
                continue

            n_lines += 1
            if len(boxes) == len(chunks):
                pairs = list(zip(boxes, chunks))
                n_matched += 1
                if manifest is not None:
                    manifest.append({'line_idx': line_idx, 'kind': '1:1',
                                     'image': row['image_filename']})
            else:
                # midpoint snapping: position each chunk proportionally by
                # cumulative char count, then assign it to the box whose
                # x-range contains its midpoint (local errors stay local)
                line_w = boxes[-1][2] - boxes[0][0]
                base_x = boxes[0][0]
                lens = [max(1, len(c)) for c in chunks]
                total_c = sum(lens)
                assign = [[] for _ in boxes]  # chunk indices per box
                acc = 0
                for ci, c in enumerate(chunks):
                    mid_frac = (acc + lens[ci] / 2) / total_c
                    mid_x = base_x + mid_frac * line_w
                    # find box containing mid_x
                    best_bi = 0
                    best_dist = None
                    for bi, b in enumerate(boxes):
                        if b[0] <= mid_x <= b[2]:
                            best_bi = bi
                            best_dist = 0
                            break
                        dist = min(abs(mid_x - b[0]), abs(mid_x - b[2]))
                        if best_dist is None or dist < best_dist:
                            best_dist, best_bi = dist, bi
                    assign[best_bi].append(ci)
                    acc += lens[ci]
                pairs = []
                for bi, cidx in enumerate(boxes and assign):
                    if not cidx:
                        continue
                    label = ' '.join(chunks[ci] for ci in cidx)
                    pairs.append((bi, label))
                pairs = [(boxes[bi], label) for bi, label in pairs]
                n_aligned += 1
                if manifest is not None:
                    manifest.append({'line_idx': line_idx, 'kind': 'proportional',
                                     'image': row['image_filename']})

            for w_idx, (b, label) in enumerate(pairs):
                x1, y1, x2, y2 = b
                crop = image[y1:y2, x1:x2]
                name = f'{line_idx:04d}_{w_idx:02d}.jpg'
                p = os.path.join(lang_dir, name)
                cv2.imwrite(p, crop)
                out.write(json.dumps({
                    'image_filename': f'marathi/{name}',
                    'expected_text': label, 'language': 'marathi'},
                    ensure_ascii=False) + '\n')
                n_crops += 1

    print(f'lines processed: {n_lines} (1:1 matched: {n_matched}, '
          f'proportional: {n_aligned}, skipped: {n_skipped})')
    print(f'word crops: {n_crops}')
    print(f'labels: {out_path}')
    if args.manifest:
        with open(args.manifest, 'w', encoding='utf-8') as f:
            for rec in manifest:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'manifest: {args.manifest}')


if __name__ == '__main__':
    main()
