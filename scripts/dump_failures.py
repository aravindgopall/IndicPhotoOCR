"""Dump every Run-B failure on the held-out eval sets, auto-tagged by error
kind, for the team's data-collection planning. Reporting only — no synthetic
design happens here."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms as T

from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint

CKPT = 'data/marathi_all_real_d.ckpt'
EVALS = [
    ('clean_production_305', 'data/marathi_v2_test_det_clean/labels.jsonl', 'data/marathi_v2_test_det_clean'),
    ('short_holdout_75', 'data/marathi_v3_short_test_det/labels.jsonl', 'data/marathi_v3_short_test_det'),
]
OUT = os.path.expanduser('~/Desktop/marathi_failure_report')


def lev(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev, dp[j] = dp[j], cur
    return dp[-1]


PUNCT = set('.,:;()/-!?"|')


def classify(gt, pred):
    if pred == gt:
        return None
    gtp = ''.join(c for c in gt if c not in PUNCT)
    prp = ''.join(c for c in pred if c not in PUNCT)
    if gtp == prp:
        if pred.rstrip('.,:;)') == gt.rstrip('.,:;)'):
            return 'trailing-punct lost/gained'
        if gt.startswith(pred):
            return 'leading/trailing chars dropped'
        if pred.startswith(gt):
            return 'extra chars added'
        return 'punct type or position wrong'
    d = lev(gt, pred)
    digit_gt = sum(c in '०१२३४५६७८९' for c in gt)
    if d <= 2:
        if digit_gt >= 2:
            return 'near-miss (<=2 chars), digit-heavy'
        return 'near-miss (<=2 chars)'
    if d >= max(3, len(gt) * 0.5):
        return 'substantial misread'
    if digit_gt >= len(gt) * 0.4:
        return 'partially read digits/code'
    return 'several chars wrong'


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_from_checkpoint(CKPT).eval().to('cpu')
    tr = T.Compose([T.Resize(model.hparams.img_size, T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(0.5, 0.5)])
    report = []
    for eval_name, jsonl, d in EVALS:
        rows = [json.loads(l) for l in open(jsonl, encoding='utf-8')]
        n_ok, fails = 0, []
        for r in rows:
            p = os.path.join(d, r['image_filename'])
            if not os.path.exists(p):
                continue
            img = Image.open(p).convert('RGB')
            with torch.no_grad():
                logits = model(tr(img).unsqueeze(0))
                probs = logits.softmax(-1)
                preds, _ = model.tokenizer.decode(probs)
                pred = model.charset_adapter(preds[0]).strip()
            gt = r['expected_text'].strip()
            if pred == gt:
                n_ok += 1
            else:
                fails.append((os.path.basename(p), gt, pred, classify(gt, pred), img.size))
        n = n_ok + len(fails)
        report.append((eval_name, n, n_ok, fails))

        # per-eval section
        print(f'\n{"=" * 70}\n{eval_name}: {n_ok}/{n} exact ({100 * n_ok / n:.1f}%), {len(fails)} failures\n{"=" * 70}')
        from collections import Counter
        cats = Counter(f[3] for f in fails)
        print('error kinds:')
        for c, k in cats.most_common():
            print(f'  {k:>3}  {c}')
        print('\nall failures (image | GT | prediction | width×height):')
        for fn, gt, pred, cat, size in fails:
            print(f'  {fn}  [{size[0]}x{size[1]}] {cat}')
            print(f'     GT  = {gt!r}')
            print(f'     pred= {pred!r}')
    with open(os.path.join(OUT, 'ALL_FAILURES.txt'), 'w', encoding='utf-8') as f:
        for eval_name, n, n_ok, fails in report:
            f.write(f'{"=" * 70}\n{eval_name}: {n_ok}/{n} exact, {len(fails)} failures\n{"=" * 70}\n')
            for fn, gt, pred, cat, size in fails:
                f.write(f'{fn}  [{size[0]}x{size[1]}] {cat}\n  GT  = {gt!r}\n  pred= {pred!r}\n')
    print(f'\nsaved: {OUT}/ALL_FAILURES.txt')


if __name__ == '__main__':
    main()
