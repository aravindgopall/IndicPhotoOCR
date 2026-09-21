"""End-to-end doc OCR with the SFT Chitrapathak-2 VLM.

Two reading paths, both compared when GT is given:
  1. lines : detector (CPU, deterministic) -> group boxes into lines by
             y-overlap -> hull crop -> VLM reads each line
  2. page  : single VLM read of the whole page (the model's native
             distribution; its own implicit line segmentation)

Scoring vs GT (word#-ordered list): Needleman-Wunsch alignment of the GT
word sequence to the flattened prediction word sequence, then per-word
exact + characc on aligned pairs (same protocol as the historic
scores_table_production.txt: original 60.20%, stage5+TTA 79.08%).
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image


def group_boxes_into_lines(boxes, y_overlap_frac=0.4):
    """boxes: list of (x1, y1, x2, y2). Returns list of hull rects in
    top-to-bottom reading order."""
    rects = [list(b) for b in boxes]
    rects.sort(key=lambda r: (r[1] + r[3]) / 2)
    lines = []
    for r in rects:
        placed = False
        for ln in lines:
            y1, y2 = max(ln[1], r[1]), min(ln[3], r[3])
            h_min = min(ln[3] - ln[1], r[3] - r[1])
            if h_min > 0 and (y2 - y1) / h_min >= y_overlap_frac:
                ln[0], ln[1], ln[2], ln[3] = min(ln[0], r[0]), min(ln[1], r[1]), \
                    max(ln[2], r[2]), max(ln[3], r[3])
                placed = True
                break
        if not placed:
            lines.append(list(r))
    lines.sort(key=lambda r: (r[1] + r[3]) / 2)
    return lines


def nw_align(gt_words, pred_words):
    """Needleman-Wunsch (gap=1, sub=lev>0) — returns list of (gt, pred)."""
    n, m = len(gt_words), len(pred_words)

    def cost(a, b):
        d = lev(a, b)
        return d / max(len(a), len(b)) if a and b else 1.0

    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + cost(gt_words[i - 1], pred_words[j - 1]),
                          D[i - 1, j] + 1, D[i, j - 1] + 1)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + cost(gt_words[i - 1], pred_words[j - 1]):
            pairs.append((gt_words[i - 1], pred_words[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            pairs.append((gt_words[i - 1], ''))
            i -= 1
        else:
            pairs.append(('', pred_words[j - 1]))
            j -= 1
    return list(reversed(pairs))


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(gt_words, pred_words, label):
    pairs = [p for p in nw_align(gt_words, pred_words) if p[0]]
    exact = sum(1 for g, p in pairs if g == p)
    cacc = sum(100 * (1 - lev(g, p) / max(len(g), 1)) for g, p in pairs)
    print(f'{label:<14} exact {exact}/{len(pairs)} ({100 * exact / len(pairs):.2f}%)   '
          f'characc {cacc / len(pairs):.2f}')
    return exact, len(pairs), cacc / len(pairs), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model', default='data/chitrapathak2')
    ap.add_argument('--adapter', default='data/cp2_sft_lora')
    ap.add_argument('--gt', default=None, help='jsonl with word_no/text fields')
    ap.add_argument('--max-pixels', type=int, default=600000)
    ap.add_argument('--min-pixels', type=int, default=56 * 56,
                    help='raise to force upscaling of small-text lines (e.g. 200000)')
    ap.add_argument('--max-new-tokens', type=int, default=640)
    ap.add_argument('--page-max-new-tokens', type=int, default=2048)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from IndicPhotoOCR.ocr import OCR
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    # ---- detect + line group (CPU deterministic) ----
    print('detecting ...', flush=True)
    ocr = OCR(device='cpu', identifier_lang='auto', verbose=False)
    dets = ocr.detect(args.image)
    boxes = []
    for d in dets:
        pts = np.array(d)
        boxes.append((int(pts[:, 0].min()), int(pts[:, 1].min()),
                      int(pts[:, 0].max()), int(pts[:, 1].max())))
    lines = group_boxes_into_lines(boxes)
    print(f'boxes: {len(boxes)} -> lines: {len(lines)}', flush=True)

    # ---- VLM ----
    print('loading VLM ...', flush=True)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa' if device != 'cpu' else 'eager')
    if args.adapter and os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels,
                                              min_pixels=args.min_pixels)
    print(f'loaded (device={device}).', flush=True)

    def ocr_img(img, max_new=None):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': 'Perform OCR on this image and transcribe all visible text exactly as it appears.'},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new or args.max_new_tokens,
                                 do_sample=False, use_cache=True)
        gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    page = Image.open(args.image).convert('RGB')
    W, H = page.size

    # ---- path 1: line crops ----
    t0 = time.time()
    line_texts = []
    for k, (x1, y1, x2, y2) in enumerate(lines):
        pad = 3
        crop = page.crop((max(0, x1 - pad), max(0, y1 - pad),
                          min(W, x2 + pad), min(H, y2 + pad)))
        txt = ocr_img(crop)
        line_texts.append(txt)
        if (k + 1) % 10 == 0:
            print(f'  {k + 1}/{len(lines)} lines ({(time.time() - t0) / (k + 1):.1f}s/line)', flush=True)
    line_words = ' '.join(line_texts).split()

    # ---- path 2: full page ----
    page_text = ocr_img(page, max_new=args.page_max_new_tokens)
    page_words = page_text.split()

    print(f'\n=== {os.path.basename(args.image)} ===')
    print(f'lines read: {len(line_texts)}   page words: {len(page_words)}   line words: {len(line_words)}')
    print('\n--- PAGE READ (first 400 chars) ---')
    print(page_text[:400])

    results = {'image': args.image, 'boxes': len(boxes), 'lines': lines,
               'line_texts': line_texts, 'page_text': page_text}
    if args.gt:
        gt_words = [json.loads(l)['text'] for l in open(args.gt, encoding='utf-8')]
        print(f'\nGT words: {len(gt_words)}')
        print('historic reference: original 60.20% / stage5+TTA 79.08% / characc 93.77')
        r1 = score(gt_words, line_words, 'VLM lines')
        r2 = score(gt_words, page_words, 'VLM page')
        results['scores'] = {'lines_exact': r1[0], 'lines_n': r1[1], 'lines_characc': r1[2],
                             'page_exact': r2[0], 'page_n': r2[1], 'page_characc': r2[2],
                             'pairs_lines': r1[3]}

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=1))
        print(f'\nsaved: {args.out}')


if __name__ == '__main__':
    main()
