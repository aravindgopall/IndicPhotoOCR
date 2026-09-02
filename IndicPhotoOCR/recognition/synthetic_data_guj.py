"""Gujarati synthetic data generator — with all Marathi lessons applied.

Lessons baked in from the Marathi effort:
  1. Tofu-verified font selection (exact .notdef bitmap comparison)
  2. RAQM shaping (real conjuncts like the documents)
  3. bare-dominant punctuation contrast (74% bare — prevents over-prediction)
  4. Digit-script consistency: ENGLISH digits only (this dataset's convention)
  5. Standalone punctuation drills ('-', '/', ':') — worst baseline failures
  6. No zero-width characters in labels
  7. Category balance: ~60% bare overall
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from IndicPhotoOCR.recognition.synthetic_data import (
    render_text, augment_image, _font_supports_text,
)

GUJ_FONTS = [
    '/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc',
    '/System/Library/Fonts/KohinoorGujarati.ttc',
    '/System/Library/Fonts/Supplemental/GujaratiMT.ttc',
]

EN_DIGITS = '0123456789'
LATIN_TOKENS = ['OPD', 'DOB', 'ID', 'PDF', 'OTP', 'SMS', 'IR', 'PAN', 'Aadhaar', 'CEO', 'PI']


def _pick_font_guj(text: str) -> str:
    fonts = list(GUJ_FONTS)
    random.shuffle(fonts)
    for fp in fonts:
        if _font_supports_text(fp, text):
            return fp
    return GUJ_FONTS[0]


def load_vocab():
    with open('data/gujarati_vocab.json', encoding='utf-8') as f:
        v = json.load(f)
    plain = [w for w in v['plain'] if len(w) > 1]
    return plain


def _ed(n):
    return ''.join(random.choice(EN_DIGITS) for _ in range(n))


def _date():
    return f'{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2015, 2027)}'


def gen_punct_contrast(vocab):
    w = random.choice(vocab)
    r = random.random()
    if r < 0.78:
        return w
    if r < 0.88:
        return w + '.'
    if r < 0.94:
        return w + ':'
    if r < 0.97:
        return w + ' -'
    return w + ' :-'


def gen_standalone_special(vocab=None):
    return random.choice(['-', '/', ':', '-', '/', '.'])


def gen_times(vocab=None):
    h1, m1 = random.randint(8, 17), random.choice([0, 15, 30, 45])
    h2 = random.randint(h1 + 1, 19)
    m2 = random.choice([0, 15, 30, 45])
    return random.choice([
        f'{h1}:{m1:02d}-{h2}:{m2:02d}',
        f'{h1}:{m1:02d}',
        f'સવારે {h1}:{m1:02d}',
    ])


def gen_dates(vocab=None):
    d = _date()
    return random.choice([d, d + '.', f'તારીખ: {d}', d.replace('/', '.')])


def gen_year_slash(vocab=None):
    y = random.randint(2020, 2027)
    return f'{y}/{str(y + 1)[-2:]}'


def gen_slash_pair(vocab):
    return f'{random.choice(vocab)} / {random.choice(vocab)}'


def gen_hyphen_phrase(vocab):
    return f'{random.choice(vocab)} - {random.choice(vocab)}'


def gen_plain(vocab):
    return random.choice(vocab)


def gen_latin(vocab=None):
    return random.choice(LATIN_TOKENS)


def gen_colon_phrase(vocab):
    return f'{random.choice(vocab)}: {random.choice(vocab)}'


def gen_multiword(vocab):
    return ' '.join(random.sample(vocab, random.randint(2, 3)))


CATEGORIES = [
    ('punct_contrast', gen_punct_contrast, 0.32, True),
    ('standalone_special', gen_standalone_special, 0.05, False),
    ('times', gen_times, 0.05, True),
    ('dates', gen_dates, 0.07, True),
    ('year_slash', gen_year_slash, 0.02, False),
    ('slash_pair', gen_slash_pair, 0.04, True),
    ('hyphen_phrase', gen_hyphen_phrase, 0.03, True),
    ('plain', gen_plain, 0.30, False),
    ('latin', gen_latin, 0.05, False),
    ('colon_phrase', gen_colon_phrase, 0.03, True),
    ('multiword', gen_multiword, 0.04, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-dir', required=True)
    ap.add_argument('-n', '--num-texts', type=int, default=4000)
    ap.add_argument('-a', '--num-augment', type=int, default=2)
    ap.add_argument('--seed', type=int, default=99)
    args = ap.parse_args()

    random.seed(args.seed)
    vocab = load_vocab()
    lang_subdir = os.path.join(args.output_dir, 'gujarati')
    os.makedirs(lang_subdir, exist_ok=True)

    seen = set()
    texts = []
    by_cat = {}
    attempts = 0
    while len(texts) < args.num_texts and attempts < args.num_texts * 30:
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

    jsonl_path = os.path.join(args.output_dir, 'labels.jsonl')
    total = 0
    with open(jsonl_path, 'w', encoding='utf-8') as out:
        for i, (text, cat) in enumerate(texts):
            font_path = _pick_font_guj(text)
            font_size = random.randint(30, 48)
            img = render_text(text, font_path, font_size=font_size)
            name = f'guj_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            out.write(json.dumps({'image_filename': f'gujarati/{name}',
                                  'expected_text': text,
                                  'category': cat}, ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = render_text(text, font_path, font_size=random.randint(30, 48))
                aug = augment_image(aug, seed=args.seed + i * 200 + a)
                aname = f'guj_{i:05d}_aug{a}.png'
                aug.save(os.path.join(lang_subdir, aname))
                out.write(json.dumps({'image_filename': f'gujarati/{aname}',
                                      'expected_text': text,
                                      'category': cat}, ensure_ascii=False) + '\n')
                total += 1
            if (i + 1) % 700 == 0:
                print(f'  {i + 1}/{len(texts)} texts ({total} samples)')

    print(f'output_dir: {args.output_dir}')
    print(f'total_samples: {total}')
    print(f'unique_texts: {len(texts)}')
    print('category_counts:')
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f'  {c}: {n}')


if __name__ == '__main__':
    main()
