"""Marathi synthetic v7 — punctuation edge-case round (2026-09-07).

Targets the two biggest genuine failure categories of Run D (fixed evals,
43% of remaining failure mass), patterns only, random values:
  1. leading_punct_pair (28%) — ':-' prefix variants: ':-X)' vs '-X' vs 'X';
     paren wrappers '(X),' vs '(X)' vs 'X),' (models drop/mix leading
     punctuation pairs at box start)
  2. fragment_start_contrast (24%) — code-fragment prefix retention:
     'क्र.१३७/कार्या-' vs '१३७/कार्या-'; 'याचीका' no-src. Models curtail
     the FIRST fragment at box start
  3. trailing_punct (20%) — '.', ',', ':' endings at boosted share
  4. dates_year (8%) — 2016/2026-style confusion share from v6
  5. plain (20%) — real-word retention

Values random/year-varying; vocabulary: training sources only.
Geometry: tight fill + min-width 150, same as v6.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

from IndicPhotoOCR.recognition.synthetic_data import (
    render_text, augment_image, _font_supports_text,
)

FONTS = [
    '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc',
    '/System/Library/Fonts/Kohinoor.ttc',
]
DEV_DIGITS = '०१२३४५६७८९'
DEPT_WORDS = ['सारसं', 'ससं', 'विसं', 'प्रशा', 'सेवा', 'भाषा', 'नियं', 'अधी', 'महा',
              'कायदा', 'वित्त', 'गृह', 'वैद्यकीय', 'शिक्षण', 'कृषी', 'ऊर्जा', 'नगर',
              'ग्राम', 'जल', 'वन', 'आस्था', 'अर्थ', 'कार्या', 'प्रतिनि', 'पूरक', 'सनिवे']
PREFIXES = ['क्र.', 'जा.क्र.', 'प्र.क्र.', 'रुसा/', 'प्रशा-', 'आस्था-', 'संकीर्ण-']


def _pick_font(text):
    fonts = list(FONTS)
    random.shuffle(fonts)
    for fp in fonts:
        if _font_supports_text(fp, text):
            return fp
    return FONTS[0]


def _d(n):
    return ''.join(random.choice(DEV_DIGITS) for _ in range(n))


def _year():
    y = random.randint(2008, 2027)
    return ''.join(DEV_DIGITS[int(c)] for c in str(y))


def load_vocab():
    words = set()
    for line in open('data/marathi_v3_short_train_lines/labels.jsonl', encoding='utf-8'):
        for t in json.loads(line)['expected_text'].split():
            t = t.strip('.,:;()"-\u200c\u200d')
            if 2 <= len(t) <= 12 and re.match(r'^[\u0900-\u097F]+$', t):
                words.add(t)
    for line in open('data/marathi_words/labels.jsonl', encoding='utf-8'):
        t = json.loads(line)['expected_text'].strip()
        if 2 <= len(t) <= 12 and re.match(r'^[\u0900-\u097F]+$', t):
            words.add(t)
    return sorted(words)


# ---- categories ---------------------------------------------------------------

def gen_leading_punct_pair(vocab):
    """':-X' variants + paren wrappers — the leading-char drop cluster."""
    w = random.choice(vocab)
    r = random.random()
    if r < 0.18:
        return f':-{w}'
    if r < 0.32:
        return f'-{w}'
    if r < 0.44:
        return f':-{_d(2)})'
    if r < 0.56:
        return f'({w}),'
    if r < 0.70:
        return f'({w})'
    if r < 0.82:
        return f'{w}),'
    if r < 0.92:
        return f'{w}:-'
    return f'({_d(2)})'


def gen_fragment_start_contrast(vocab):
    """Prefix retention: 'क्र.१३७/कार्या-' vs '१३७/कार्या-' etc."""
    pre = random.choice(PREFIXES)
    tail = f'''{_d(random.randint(2, 4))}/{random.choice(DEPT_WORDS)}-'''
    r = random.random()
    if r < 0.52:
        s = f'{pre}{tail}'
    else:
        s = tail
    if random.random() < 0.3:
        s = f'{s}{_d(2)}'
    return s


def gen_trailing_punct(vocab):
    w = random.choice(vocab)
    r = random.random()
    if r < 0.45:
        return w
    if r < 0.67:
        return w + '.'
    if r < 0.90:
        return w + ','
    return w + ':'


def gen_dates_year(vocab):
    d, m = random.randint(1, 31), random.randint(1, 12)
    y = _year()
    dv = f'{d:02d}'
    mv = f'{m:02d}'
    dv = ''.join(DEV_DIGITS[int(c)] for c in dv)
    mv = ''.join(DEV_DIGITS[int(c)] for c in mv)
    r = random.random()
    if r < 0.5:
        return f'दि.{dv}.{mv}.{y}.'
    return f'{dv}/{mv}/{y}'


def gen_plain(vocab):
    return random.choice(vocab)


CATEGORIES = [
    ('leading_punct_pair', gen_leading_punct_pair, 0.28, True),
    ('fragment_start_contrast', gen_fragment_start_contrast, 0.24, True),
    ('trailing_punct', gen_trailing_punct, 0.20, True),
    ('dates_year', gen_dates_year, 0.08, False),
    ('plain', gen_plain, 0.20, False),
]


def render_matched(text, font_path):
    font_size = random.randint(80, 125)
    img = render_text(text, font_path, font_size=font_size)
    w, h = img.size
    target_fill = random.uniform(0.80, 0.98)
    target_h = max(h + 4, int(h / target_fill))
    if target_h > h:
        canvas = Image.new('RGB', (w, target_h), (255, 255, 255))
        canvas.paste(img, (0, (target_h - h) // 2))
        img = canvas
        h = target_h
    if img.size[0] < 150:
        canvas = Image.new('RGB', (150, h), (255, 255, 255))
        canvas.paste(img, ((150 - img.size[0]) // 2, 0))
        return canvas
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-dir', required=True)
    ap.add_argument('-n', '--num-texts', type=int, default=3500)
    ap.add_argument('-a', '--num-augment', type=int, default=1)
    ap.add_argument('--seed', type=int, default=77)
    args = ap.parse_args()

    random.seed(args.seed)
    vocab = load_vocab()
    print(f'vocab: {len(vocab)} words (training sources only)')

    lang_subdir = os.path.join(args.output_dir, 'marathi')
    os.makedirs(lang_subdir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, 'labels.jsonl')

    seen, texts, by_cat = set(), [], {}
    attempts = 0
    while len(texts) < args.num_texts and attempts < args.num_texts * 40:
        attempts += 1
        r = random.random()
        acc = 0.0
        for cat, fn, w, dedup in CATEGORIES:
            acc += w
            if r <= acc:
                t = fn(vocab)
                if t and (not dedup or t not in seen):
                    seen.add(t)
                    texts.append((t, cat))
                    by_cat[cat] = by_cat.get(cat, 0) + 1
                break

    total = 0
    with open(jsonl_path, 'w', encoding='utf-8') as out:
        for i, (text, cat) in enumerate(texts):
            font_path = _pick_font(text)
            img = render_matched(text, font_path)
            name = f'v7_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            out.write(json.dumps({'image_filename': f'marathi/{name}',
                                  'expected_text': text, 'category': cat},
                                 ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = augment_image(render_matched(text, font_path),
                                    seed=args.seed + i * 100 + a)
                aname = f'v7_{i:05d}_aug{a}.png'
                aug.save(os.path.join(lang_subdir, aname))
                out.write(json.dumps({'image_filename': f'marathi/{aname}',
                                      'expected_text': text, 'category': cat},
                                     ensure_ascii=False) + '\n')
                total += 1

    print(f'total_samples: {total}  unique_texts: {len(texts)}')
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f'  {c}: {n}')


if __name__ == '__main__':
    main()
