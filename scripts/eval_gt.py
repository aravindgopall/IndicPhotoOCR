"""Ground-truth evaluation robust to identification non-determinism.

Strategy:
  1. Detection is deterministic (verified) -> boxes are stable across runs.
  2. Order boxes into canonical reading order (lines top-to-bottom, words
     left-to-right) using POSITION only -- no language identification.
  3. Recognize every box with a reference model (stage1) -> canonical sequence.
  4. DP-align (Needleman-Wunsch) the GT word sequence (table order) to the
     canonical prediction sequence -> gt_at_box[det_idx].
  5. Recognize every box with each model and score against gt_at_box.

Usage:
  python scripts/eval_gt.py [--models original=... stage1=... stage2=... stage3=...]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from IndicPhotoOCR.ocr import OCR
from score_deterministic import levenshtein, has_special, has_latin, has_digit

SPECIAL = set('()-,./:;%|—""?!&')


def bbox_xyxy(bbox):
    pts = np.array(bbox, np.int32)
    return [int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max())]


def canonical_order(detections):
    """det_idxs in reading order: lines top-to-bottom, words left-to-right."""
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
    lines.sort(key=lambda line: min(boxes[i][1] for i in line))
    return [i for line in lines for i in line]


def nw_align(a_list, b_list):
    """Needleman-Wunsch. Returns list of (a_idx or None, b_idx or None)."""
    n, m = len(a_list), len(b_list)
    GAP = 1.0

    def sub_cost(a, b):
        return levenshtein(a, b) / max(len(a), len(b), 1)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP
    for i in range(1, n + 1):
        ai = a_list[i - 1]
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost(ai, b_list[j - 1]),
                dp[i - 1][j] + GAP,
                dp[i][j - 1] + GAP,
            )
    # traceback
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + sub_cost(a_list[i - 1], b_list[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + GAP:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def fingerprint_assign(table_sbs, fresh_preds, canon):
    """Assign table word#s to boxes via 3-model text fingerprint matching.

    table_sbs: {word_no: {model: text}} from the comparison-file side-by-side.
    fresh_preds: {model: [text per det_idx]}.
    Returns {det_idx: word_no}, plus diagnostics.
    """
    from scipy.optimize import linear_sum_assignment

    models = [m for m in table_sbs[next(iter(table_sbs))] if m in fresh_preds]
    word_nos = sorted(table_sbs)
    n_words, n_boxes = len(word_nos), len(fresh_preds[models[0]])

    cost = np.zeros((n_words, n_boxes))
    for wi, wn in enumerate(word_nos):
        fp = table_sbs[wn]
        for b in range(n_boxes):
            c = 0.0
            for m in models:
                a, p = fp[m], fresh_preds[m][b]
                c += levenshtein(a, p) / max(len(a), len(p), 1)
            cost[wi, b] = c

    row_ind, col_ind = linear_sum_assignment(cost)
    assign = {int(col): int(word_nos[ri]) for ri, col in zip(row_ind, col_ind)}

    # Diagnostics: exact 3-tuple fingerprint matches
    exact = 0
    for b, wn in assign.items():
        if all(table_sbs[wn][m] == fresh_preds[m][b] for m in models):
            exact += 1
    return assign, exact, len(assign), models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/tmp/test_marathi2.png')
    ap.add_argument('--gt', default='/tmp/gt_lines/ground_truth.jsonl')
    ap.add_argument('--output', default=os.path.expanduser(
        '~/Desktop/marathi_ocr_visual/gt_scores.txt'))
    ap.add_argument('--models', nargs='+', default=[
        'original=IndicPhotoOCR/recognition/models/marathi.ckpt',
        'stage1=data/marathi_stage1.ckpt',
        'stage2=data/marathi_finetuned.ckpt',
        'stage3=data/marathi_stage3.ckpt',
    ], help='name=checkpoint pairs')
    ap.add_argument('--dump', default='/tmp/gt_lines/eval_dump.json')
    ap.add_argument('--device', default='cpu',
                   help='cpu (default, deterministic) or mps (faster but flaky)')
    args = ap.parse_args()

    # Load GT in word# order (table order)
    gt = {}
    for line in open(args.gt, encoding='utf-8'):
        line = line.strip()
        if line:
            r = json.loads(line)
            gt[r['word_no']] = r['text'].strip()
    gt_seq = [gt[wn] for wn in sorted(gt)]
    print(f'GT words: {len(gt_seq)}')

    # Load the comparison-file side-by-side fingerprints (table run)
    sbs_path = os.path.expanduser('~/Desktop/marathi_ocr_comparison_2026-09-01.txt')
    table_sbs = {}
    for line in open(sbs_path, encoding='utf-8'):
        if re.match(r'^\s*\d+ \|', line):
            p = [x.strip() for x in line.split('|')]
            table_sbs[int(p[0])] = {'original': p[1], 'stage1': p[2], 'stage2': p[3]}
    print(f'table fingerprints: {len(table_sbs)} word#s')

    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)
    image = cv2.imread(args.image)
    print('Detecting...')
    detections = ocr.detect(args.image)
    print(f'  {len(detections)} boxes')
    canon = canonical_order(detections)

    # Recognize every box with each model
    model_preds = {}
    model_slugs = []
    for spec in args.models:
        slug, ckpt = spec.split('=', 1)
        if not os.path.exists(ckpt):
            print(f'  skipping {slug}: {ckpt} not found')
            continue
        model_slugs.append(slug)
        print(f'Recognizing with {slug}...')
        texts = []
        for i, bbox in enumerate(detections):
            cp = ocr.crop_bbox(image, bbox)
            try:
                t = ocr.recognise(cp, None, checkpoint=ckpt)
                texts.append((t or '').strip() if isinstance(t, str) else str(t).strip())
            except Exception:
                texts.append('')
            finally:
                if os.path.exists(cp):
                    os.remove(cp)
            if (i + 1) % 100 == 0:
                print(f'    {i + 1}/{len(detections)}')
        model_preds[slug] = texts

    if 'stage1' not in model_preds or 'original' not in model_preds or 'stage2' not in model_preds:
        print('ERROR: original, stage1, stage2 models required for fingerprint alignment')
        return

    # Assign GT word#s to boxes via 3-model fingerprint matching
    assign, exact, n_assign, fp_models = fingerprint_assign(table_sbs, model_preds, canon)
    print(f'Fingerprint assignment: {n_assign} word#s assigned, {exact} '
          f'({100 * exact / max(n_assign, 1):.1f}%) exact 3-tuple matches '
          f'(alignment sanity, models={fp_models})')
    gt_at_box = {b: gt[wn] for b, wn in assign.items() if wn in gt}
    aligned_exact = sum(1 for b in gt_at_box if model_preds['stage1'][b] == gt_at_box[b])
    aligned_total = len(gt_at_box)
    print(f'GT mapped to {aligned_total} boxes; stage1 exactly matches GT on '
          f'{aligned_exact} ({100 * aligned_exact / max(aligned_total, 1):.1f}%)')

    # Score all models
    covered = sorted(gt_at_box)
    lines = []
    lines.append('GROUND-TRUTH SCORES (per-box, alignment-robust)')
    lines.append(f'GT words mapped: {len(covered)}   Boxes: {len(detections)}')
    lines.append(f'Alignment sanity: stage1 matches GT exactly on '
                 f'{aligned_exact}/{aligned_total} aligned words')
    lines.append('=' * 80)
    lines.append(f'{"model":<12}{"exact":>6}{"exact%":>9}{"characc%":>10}{"CER":>8}')
    results = {}
    for slug in model_slugs:
        preds = model_preds[slug]
        ex = sum(1 for b in covered if preds[b] == gt_at_box[b])
        tot_ed = sum(levenshtein(gt_at_box[b], preds[b]) for b in covered)
        tot_len = sum(max(len(gt_at_box[b]), 1) for b in covered)
        results[slug] = (ex, 100 * ex / len(covered),
                         100 * (1 - tot_ed / tot_len), tot_ed / tot_len)
        lines.append(f'{slug:<12}{ex:>6}{100 * ex / len(covered):>9.2f}'
                     f'{100 * (1 - tot_ed / tot_len):>10.2f}{tot_ed / tot_len:>8.4f}')

    # Category breakdown
    lines.append('')
    lines.append('CATEGORY BREAKDOWN (exact match %):')
    cats = [('special chars', lambda g: has_special(g)),
            ('digits', lambda g: has_digit(g)),
            ('Latin', lambda g: has_latin(g)),
            ('plain Devanagari', lambda g: not has_special(g) and not has_latin(g))]
    for cname, fn in cats:
        subset = [b for b in covered if fn(gt_at_box[b])]
        if not subset:
            continue
        row = f'  {cname} (n={len(subset)}):'.ljust(32)
        for slug in model_slugs:
            ok = sum(1 for b in subset if model_preds[slug][b] == gt_at_box[b])
            row += f'{slug}={100 * ok / len(subset):.1f}%  '
        lines.append(row)

    # Improvement vs stage2 for every model
    if 'stage2' in results and len(model_slugs) > 1:
        lines.append('')
        lines.append('IMPROVEMENT vs each prior model (exact match):')
        for slug in model_slugs:
            if slug not in results:
                continue
            for base in model_slugs:
                if base == slug or base not in results:
                    continue
                d = results[slug][1] - results[base][1]
                lines.append(f'  {slug} vs {base}: {d:+.2f} pp '
                             f'({results[slug][0]} vs {results[base][0]} words)')

    # Failures for the last model (newest)
    last = model_slugs[-1]
    lines.append('')
    lines.append(f'{last.upper()} FAILURES (exact-match misses):')
    fails = [(b, gt_at_box[b], model_preds[last][b]) for b in covered
             if model_preds[last][b] != gt_at_box[b]]
    lines.append(f'total: {len(fails)}/{len(covered)}')
    for b, g, p in sorted(fails, key=lambda x: x[0]):
        lines.append(f'  box#{b} GT={g!r}')
        lines.append(f'         {last}={p!r}')

    report = '\n'.join(lines)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(args.dump, 'w', encoding='utf-8') as f:
        json.dump({'gt_at_box': gt_at_box,
                   'preds': {s: model_preds[s] for s in model_slugs},
                   'canon': canon}, f, ensure_ascii=False, indent=1)
    print(report)
    print(f'\nWritten to: {args.output}')


if __name__ == '__main__':
    main()
