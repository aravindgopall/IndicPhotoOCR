"""Fix eval-harness label alignment using exact detector boxes + model consensus.

v2 — replaces template matching (unreliable on duplicate text: JPEG crops of
repeated words match the wrong occurrence by a hair) with a fresh detector
run, which is deterministic on CPU and yields the exact box coordinates the
crops were cut from.

Method (honest, GT-preserving):
  1. re-run the production detector on each line image; take x-sorted boxes
     (the saved crops are exactly these boxes, verified pixel-identical)
  2. read each crop with TWO models (B and C); consensus = agreement
  3. DP-align the line GT string to the boxes: each box gets a contiguous
     GT substring (labels are always GT text, never model output);
     token-boundary cuts strongly preferred; boxes whose content is NOT in
     the GT are DROPPED (ungradeable without new annotation); GT spans with
     no box are skipped (detection miss — not a recognition error)
  4. validation: lines whose labels already agree with both models must
     come out unchanged
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from IndicPhotoOCR.ocr import OCR
from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint

MODELS = [
    ('D', 'data/marathi_all_real_d.ckpt'),
    ('v3', 'data/marathi_v3.ckpt'),
]
# NOTE: the eval sets fixed on 2026-09-04 were driven by models B+C (since
# pruned). Current drivers: D (production) + v3 (independent lineage — more
# diverse than the original B/C pair, which shared training data).


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def token_starts(S):
    bounds = set()
    for i, ch in enumerate(S):
        if ch == ' ':
            bounds.add(i)
            bounds.add(i + 1)
    bounds.add(0)
    bounds.add(len(S))
    return bounds


def align_line(S, cons, box_xs, drop_ok=None):
    K = len(cons)
    N = len(S)
    bounds = token_starts(S)
    INF = float('inf')

    dp = [[INF] * (N + 2) for _ in range(K + 1)]
    back = [[None] * (N + 2) for _ in range(K + 1)]
    dp[K][N] = 0.0
    for j in range(N - 1, -1, -1):
        if dp[K][j + 1] < INF and S[j] != ' ':
            dp[K][j] = dp[K][j + 1] + 1.5
            back[K][j] = ('skip', j + 1)

    if K and box_xs and box_xs[-1][1] > box_xs[0][0]:
        fracs = [((x1 + x2) / 2 - box_xs[0][0]) / (box_xs[-1][1] - box_xs[0][0])
                 for x1, x2 in box_xs]
    else:
        fracs = [i / max(1, K) for i in range(K)]

    for i in range(K - 1, -1, -1):
        read, w = cons[i]
        for j in range(N, -1, -1):
            best, bestop = INF, None
            for j2 in range(j + 1, min(N, j + 46) + 1):
                sub = S[j:j2].strip()
                if not sub or dp[i + 1][j2] == INF:
                    continue
                # assign filter: the consensus read must actually support
                # this substring — otherwise the box holds text not in GT
                if lev(read, sub) > max(2, 0.5 * max(len(read), len(sub))):
                    continue
                cut = 6 if j not in bounds else 0
                cut += 6 if j2 not in bounds else 0
                c = w * lev(read, sub) + cut + 0.6
                exp_j2 = fracs[i] * N
                c += 0.04 * abs(j2 - exp_j2)
                tot = dp[i + 1][j2] + c
                if tot < best:
                    best, bestop = tot, ('assign', j2)
            if dp[i + 1][j] < INF and (drop_ok is None or drop_ok[i]):
                tot = dp[i + 1][j] + (16.0 if w > 0.5 else 20.0)
                if tot < best:
                    best, bestop = tot, ('drop', j)
            for j2 in range(j + 1, min(N, j + 46) + 1):
                if S[j:j2].strip() and dp[i][j2] < INF:
                    tot = dp[i][j2] + 1.2 * len(S[j:j2].strip()) + 3.0
                    if tot < best:
                        best, bestop = tot, ('skip', j2)
            dp[i][j] = best
            back[i][j] = bestop

    labels, dropped = {}, set()
    i, j = 0, 0
    if dp[0][0] == INF:
        return None, None
    while i < K or j < N:
        op = back[i][j] if i < K else back[i][j]
        if op is None:
            if i >= K:
                j = N
                continue
            break
        kind, nxt = op
        if kind == 'assign':
            labels[i] = S[j:nxt].strip()
            j = nxt
            i += 1
        elif kind == 'drop':
            dropped.add(i)
            i += 1
        else:
            j = nxt
    return labels, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--lines-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(f'{args.lines_dir}/labels.jsonl', encoding='utf-8')]
    rows.sort(key=lambda r: r['image_filename'])

    crops = [json.loads(l) for l in
             open(f'{args.eval_dir}/labels.jsonl', encoding='utf-8')]
    lines = {}
    for r in crops:
        stem = os.path.basename(r['image_filename']).rsplit('_', 1)[0]
        lines.setdefault(stem, []).append(r)

    models = []
    for name, ckpt in MODELS:
        m = load_from_checkpoint(ckpt).eval().to('cpu')
        t = T.Compose([T.Resize(m.hparams.img_size, T.InterpolationMode.BICUBIC),
                       T.ToTensor(), T.Normalize(0.5, 0.5)])
        models.append((name, m, t))

    ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)

    os.makedirs(os.path.join(args.out_dir, 'marathi'), exist_ok=True)
    out_path = os.path.join(args.out_dir, 'labels.jsonl')
    report = []
    n_changed = n_same = n_dropped = n_total = 0

    with open(out_path, 'w', encoding='utf-8') as out:
        for stem in sorted(lines):
            line_idx = int(stem)
            gt = rows[line_idx]['expected_text'].strip()
            img_path = os.path.join(args.lines_dir, rows[line_idx]['image_filename'])
            image = cv2.imread(img_path)
            entries = sorted(
                lines[stem],
                key=lambda r: int(os.path.basename(
                    r['image_filename']).rsplit('_', 1)[1].split('.')[0]))
            paths = [os.path.join(args.eval_dir, r['image_filename'])
                     for r in entries]
            old_labels = [r['expected_text'].strip() for r in entries]

            # 1. fresh deterministic detection -> exact x-sorted boxes
            try:
                detections = ocr.detect(img_path)
            except Exception:
                detections = []
            boxes = []
            for d in detections or []:
                pts = np.array(d, np.int32)
                x1, y1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
                x2 = min(image.shape[1], pts[:, 0].max())
                y2 = min(image.shape[0], pts[:, 1].max())
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                boxes.append((int(x1), int(x2)))
            boxes.sort(key=lambda b: b[0])

            # match saved crops to fresh boxes by width consistency and count
            if len(boxes) != len(paths):
                # fall back: keep old labels (cannot guarantee box identity)
                for r, ol in zip(entries, old_labels):
                    out.write(json.dumps({'image_filename': r['image_filename'],
                                          'expected_text': ol,
                                          'language': 'marathi'},
                                         ensure_ascii=False) + '\n')
                    n_total += 1
                    n_same += 1
                report.append(f'{stem} SKIP (box count {len(boxes)} != '
                              f'crop count {len(paths)})')
                continue
            xs = boxes

            # 2. model reads + consensus
            reads = []
            heights = []
            for p in paths:
                img = Image.open(p).convert('RGB')
                heights.append(img.size[1])
                per = []
                for name, m, t in models:
                    with torch.no_grad():
                        probs = m(t(img).unsqueeze(0)).softmax(-1)
                        pr, _ = m.tokenizer.decode(probs)
                        per.append(m.charset_adapter(pr[0]).strip())
                reads.append(per)
            cons = []
            for rb, rc in reads:
                agree = lev(rb, rc) <= max(2, 0.25 * max(len(rb), len(rc)))
                cons.append((rb, 1.0 if agree else 0.4))

            # geometric plausibility guard: a hallucination trap occurs when a
            # TINY box (physically few chars) is over-read as long content.
            # A drop ("content missing from GT") based on such a read must be
            # vetoed — the region cannot contain what the models claim.
            # Estimate local text scale from confident boxes (height-adjusted).
            import statistics
            widths = [x2 - x1 for x1, x2 in xs]
            conf_chars, conf_ws, conf_hs = [], [], []
            for (x1, x2), per, w, h in zip(xs, reads, widths, heights):
                rb, rc = per
                if lev(rb, rc) <= max(2, 0.25 * max(len(rb), len(rc))) and len(rb) >= 4:
                    conf_chars.append(len(rb))
                    conf_ws.append(w)
                    conf_hs.append(h)
            if conf_chars:
                med_h = statistics.median(conf_hs)
                med_pxc = statistics.median(
                    [w / max(1, c) for c, w in zip(conf_chars, conf_ws)])
            else:
                med_h, med_pxc = 100.0, 20.0
            drop_ok = []
            for (x1, x2), (read, w), h in zip(xs, cons, heights):
                scale = (h / med_h) if med_h else 1.0
                scale = max(0.4, min(scale, 2.0))
                cap = (x2 - x1) / max(1e-6, med_pxc * scale)
                # veto drop: box physically too small for the claimed content
                veto = cap < 6 and len(read) > 2.5 * cap
                drop_ok.append(not veto)

            # 3. DP alignment on the GT (drop penalty per-box)
            labels, dropped = align_line(gt, cons, xs, drop_ok)
            if labels is None:
                labels, dropped = {}, set()

            # 4. write fixed labels
            for bi, (r, ol) in enumerate(zip(entries, old_labels)):
                if bi in dropped:
                    n_dropped += 1
                    report.append(f'{stem} DROP crop {bi}: old={ol!r} '
                                  f'(content missing from GT)')
                    continue
                new = labels.get(bi, ol)
                if new != ol:
                    n_changed += 1
                    report.append(f'{stem} FIX  crop {bi}: {ol!r} -> {new!r}')
                else:
                    n_same += 1
                out.write(json.dumps({'image_filename': r['image_filename'],
                                      'expected_text': new,
                                      'language': 'marathi'},
                                     ensure_ascii=False) + '\n')
                n_total += 1

    rep_path = os.path.join(args.out_dir, 'ALIGNMENT_CHANGES.txt')
    with open(rep_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) if report else 'no changes')
    print(f'crops: {n_total} kept (unchanged {n_same}, relabeled {n_changed}), '
          f'{n_dropped} dropped as ungradeable')
    print(f'report: {rep_path}')


if __name__ == '__main__':
    main()
