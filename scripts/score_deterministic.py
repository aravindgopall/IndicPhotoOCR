"""Deterministic scoring of OCR outputs.

Without ground truth:
  - Format-validity: valid date/code/URL patterns, special-char retention
  - Consensus: 2-of-3 model agreement as pseudo-reference

With ground truth (--gt ground_truth.jsonl, format: {"word_no": N, "text": "..."}):
  - Exact match accuracy
  - Character accuracy (1 - Levenshtein/len)
  - Per-category breakdown (special chars, digits, Latin)
"""
import argparse
import json
import os
import re
import sys

SPECIAL = set('()-,./:;%|—""?!&')
DEV_DIGITS = '०१२३४५६७८९'

DATE_PAT = re.compile(r'^[०-९0-9]{1,2}[.][०-९0-9]{1,2}[.][०-९0-9]{4}$')
CODE_PAT = re.compile(r'^\S+-\S*/\S+$')  # prefix-YYYY/pr.kr.NNN/... shape
URL_PAT = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9./-]*\.(org|in|com|gov|net)\S*$|^[a-zA-Z0-9]+/gov')


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def has_special(t):
    return any(c in SPECIAL for c in t)


def has_latin(t):
    return any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in t)


def has_digit(t):
    return any(c in DEV_DIGITS or c in '0123456789' for c in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='/tmp/gt_lines/predictions.json')
    ap.add_argument('--gt', default=None,
                    help='ground_truth.jsonl with {"word_no": N, "text": "..."}')
    ap.add_argument('--output', default=os.path.expanduser(
        '~/Desktop/marathi_ocr_visual/deterministic_scores.txt'))
    args = ap.parse_args()

    with open(args.preds, encoding='utf-8') as f:
        data = json.load(f)
    word_nos = {int(k): v for k, v in data['word_nos'].items()}
    order = data['order']
    models = list(data['preds'].keys())
    # texts_by_model[slug][word_no]
    texts = {}
    for slug in models:
        arr = data['preds'][slug]['texts']
        texts[slug] = {word_nos[i]: (arr[i] or '').strip() for i in range(len(arr))}

    n_words = len(order)
    lines = []
    lines.append('DETERMINISTIC OCR SCORES')
    lines.append(f'Words: {n_words}   Models: {models}')
    lines.append('=' * 90)

    # ------------------------------------------------------------------
    # A. Format validity (no GT needed)
    # ------------------------------------------------------------------
    lines.append('')
    lines.append('A. FORMAT-VALIDITY SCORES (objective, no ground truth needed)')
    lines.append('-' * 90)
    header = f'{"metric":<38}' + ''.join(f'{m:>16}' for m in models)
    lines.append(header)
    rows = []
    for slug in models:
        t = texts[slug]
        dates = sum(1 for w in t.values() if DATE_PAT.match(w))
        codes = sum(1 for w in t.values() if CODE_PAT.match(w) and has_digit(w))
        urls = sum(1 for w in t.values() if URL_PAT.match(w))
        sp_words = sum(1 for w in t.values() if has_special(w))
        sp_chars = sum(1 for w in t.values() for c in w if c in SPECIAL)
        latin = sum(1 for w in t.values() if has_latin(w))
        rows.append((dates, codes, urls, sp_words, sp_chars, latin))
    for name, idx in [('valid dates (dd.mm.yyyy)', 0), ('valid gov-code shapes', 1),
                      ('valid URL/Latin tokens', 2), ('words w/ special chars', 3),
                      ('total special chars', 4), ('words w/ Latin letters', 5)]:
        lines.append(f'{name:<38}' + ''.join(f'{r[idx]:>16}' for r in rows))

    # ------------------------------------------------------------------
    # B. Consensus (2-of-3)
    # ------------------------------------------------------------------
    lines.append('')
    lines.append('B. CONSENSUS SCORES (2-of-3 models agree => pseudo-reference)')
    lines.append('-' * 90)
    consensus = {}
    all_agree = 0
    for wn in order:
        vals = [texts[s][wn] for s in models]
        if len(set(vals)) == 1:
            consensus[wn] = vals[0]
            all_agree += 1
        else:
            for v in set(vals):
                if vals.count(v) >= 2:
                    consensus[wn] = v
                    break
            else:
                consensus[wn] = None
    lines.append(f'  all 3 models agree: {all_agree}/{n_words}')
    lines.append(f'  2-of-3 agreement available: {sum(1 for v in consensus.values() if v is not None)}/{n_words}')
    lines.append('')
    lines.append(f'{"model":<20}{"agree w/ consensus":>22}{"mismatch":>12}')
    for slug in models:
        t = texts[slug]
        avail = [(wn, consensus[wn]) for wn in order if consensus[wn] is not None]
        match = sum(1 for wn, c in avail if t[wn] == c)
        lines.append(f'{slug:<20}{match:>14}/{len(avail):<7}{len(avail) - match:>12}')

    # ------------------------------------------------------------------
    # C. Ground-truth scores
    # ------------------------------------------------------------------
    gt = None
    if args.gt and os.path.exists(args.gt):
        gt = {}
        with open(args.gt, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gt[int(row['word_no'])] = row['text'].strip()
        lines.append('')
        lines.append('C. GROUND-TRUTH SCORES')
        lines.append('-' * 90)
        lines.append(f'  GT words loaded: {len(gt)} (of {n_words} detected)')
        covered = [wn for wn in order if wn in gt]
        lines.append(f'  scored on covered words: {len(covered)}')

        def score(slug):
            t = texts[slug]
            exact = sum(1 for wn in covered if t[wn] == gt[wn])
            char_accs = []
            total_ed = total_len = 0
            for wn in covered:
                g, p = gt[wn], t[wn]
                ed = levenshtein(g, p)
                total_ed += ed
                total_len += max(len(g), 1)
                char_accs.append(1 - ed / max(len(g), 1))
            return (exact, 100 * exact / max(len(covered), 1),
                    100 * (1 - total_ed / max(total_len, 1)),
                    100 * sum(char_accs) / max(len(char_accs), 1))

        lines.append('')
        lines.append(f'{"model":<20}{"exact":>8}{"exact %":>10}{"char acc % (corpus)":>22}{"char acc % (mean)":>20}')
        results = {}
        for slug in models:
            ex, exp, ca_c, ca_m = score(slug)
            results[slug] = (ex, exp, ca_c, ca_m)
            lines.append(f'{slug:<20}{ex:>8}{exp:>10.2f}{ca_c:>22.2f}{ca_m:>20.2f}')

        # Category breakdown
        lines.append('')
        lines.append('  Category breakdown (exact match % on covered words):')
        cats = [('with special chars', lambda g: has_special(g)),
                ('with digits', lambda g: has_digit(g)),
                ('with Latin letters', lambda g: has_latin(g)),
                ('plain Devanagari', lambda g: not has_special(g) and not has_latin(g))]
        for cname, pred_fn in cats:
            subset = [wn for wn in covered if pred_fn(gt[wn])]
            if not subset:
                continue
            row = f'    {cname} (n={len(subset)}):'.ljust(40)
            for slug in models:
                m = sum(1 for wn in subset if texts[slug][wn] == gt[wn])
                row += f'  {slug}={100 * m / len(subset):.1f}%'
            lines.append(row)

        # Worst errors sample
        lines.append('')
        lines.append('  Worst character errors (stage2 vs GT, top 15):')
        errs = sorted(covered, key=lambda wn: -levenshtein(gt[wn], texts['stage2'][wn]))[:15]
        for wn in errs:
            lines.append(f'    word#{wn}: GT={gt[wn]!r}  stage2={texts["stage2"][wn]!r}')
    else:
        lines.append('')
        lines.append('C. GROUND-TRUTH SCORES: not run')
        lines.append('   Provide --gt ground_truth.jsonl ({"word_no": N, "text": "..."}) to enable.')

    report = '\n'.join(lines)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report)
    print(f'\nWritten to: {args.output}')


if __name__ == '__main__':
    main()
