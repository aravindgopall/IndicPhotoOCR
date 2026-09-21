"""Marathi synthetic v6 — failure-pattern-driven, systematic values only.

Categories target the GENUINE failure patterns from the Run-B audit
(pattern knowledge only — every value is randomly generated; vocabulary
comes from training sources only, never eval):
  1. amount_indian   — rupee amounts, Indian comma grouping, /- suffix,
                       रु./रू. prefixes, .०० decimals
  2. dates            — दि.डी.एम.यू., डी/एम/यू, सन ranges, trailing-dot variants
  3. punct_fragment   — word + dangling / ( ) , -
  4. dotted_chain     — dotted abbreviations joined by slashes
  5. code_with_spaces — kept (was effective)
  6. list_markers     — १. २. style, rendered at NATIVE small width to
                        teach the stretched-tiny-crop geometry
  7. paren_contrast / punct_contrast / plain — retained, smaller shares

Geometry: tight fill 0.80-0.98 (detector), min-width 150px — except
list_markers which render at native width (matching real <100px crops
that get anamorphically stretched by the 32x128 transform).
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
CONS = 'कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह'
MATRAS = ['', 'ा', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो']
CONJ = ['ग्र', 'स्थ', 'क्र', 'त्र', 'द्व', 'श्र', 'प्र', 'ब्र']
DEPT_WORDS = ['सारसं', 'ससं', 'विसं', 'प्रशा', 'सेवा', 'भाषा', 'नियं', 'अधी', 'महा',
              'कायदा', 'वित्त', 'गृह', 'वैद्यकीय', 'शिक्षण', 'कृषी', 'ऊर्जा', 'नगर',
              'ग्राम', 'जल', 'वन', 'आस्था', 'अर्थ', 'कार्या', 'प्रतिनि', 'पूरक', 'सनिवे']


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


def dotted_abbrev():
    n = random.randint(2, 4)
    units = []
    for _ in range(n):
        if random.random() < 0.15:
            units.append(random.choice(CONJ) + random.choice(['', 'ा', 'ि']))
        else:
            units.append(random.choice(CONS) + random.choice(MATRAS))
    units = [u for u in units if u] or ['क']
    return '.'.join(units) + '.'


def _indian_groups():
    """Random digit string in Indian comma grouping (last 3, then 2s)."""
    total = random.choice([4, 5, 6, 7, 8, 9, 10, 11])
    digits = _d(total)
    groups = [digits[-3:]]
    rest = digits[:-3]
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.append(rest)
    groups.reverse()
    return ','.join(groups)


# ---- categories ---------------------------------------------------------------

def gen_amount_indian(vocab):
    r = random.random()
    amt = _indian_groups()
    if r < 0.28:
        return amt + '/-'
    if r < 0.46:
        return f'({random.choice(["रू.", "रु."])}{amt}/-'
    if r < 0.60:
        return f'{random.choice(["रू.", "रु."])}{amt}/-'
    if r < 0.74:
        return f'{random.choice(["रू.", "रु."])}{amt}.००'
    if r < 0.86:
        return f'{amt}/- ({random.choice(["रू.", "रु."])}'
    if r < 0.94:
        return f'{amt} {random.choice(vocab)}'
    return amt


def gen_dates(vocab):
    d, m = random.randint(1, 31), random.randint(1, 12)
    y = _year()
    dd = f'{d:02d}' if random.random() < 0.7 else f'{d:d}'
    mm = f'{m:02d}' if random.random() < 0.7 else f'{m:d}'
    dv = ''.join(DEV_DIGITS[int(c)] for c in dd)
    mv = ''.join(DEV_DIGITS[int(c)] for c in mm)
    r = random.random()
    if r < 0.30:
        return f'दि.{dv}.{mv}.{y}.'
    if r < 0.45:
        return f'दि.{dv}.{mv}.{y}'
    if r < 0.62:
        return f'{dv}/{mv}/{y}'
    if r < 0.72:
        return f'{dv}.{mv}.{y}'
    if r < 0.84:
        return f'({random.choice(["सन", "मु.दि."])} {y}-{_d(2)} व {_year()}-{_d(2)})'
    return f'सन {y}-{_d(2)}'


def gen_punct_fragment(vocab):
    w = random.choice(vocab)
    r = random.random()
    if r < 0.16:
        return f'/ {w}'
    if r < 0.30:
        return f'{w}/'
    if r < 0.44:
        return f'{w} ('
    if r < 0.56:
        return f'{w})'
    if r < 0.66:
        return f'({w})'
    if r < 0.78:
        return f'{w},'
    if r < 0.88:
        return f'{w} -'
    return f'{w}:'


def gen_dotted_chain(vocab):
    n = random.randint(2, 3)
    parts = []
    for _ in range(n):
        r = random.random()
        if r < 0.45:
            parts.append(dotted_abbrev())
        elif r < 0.75:
            w = random.choice(DEPT_WORDS)
            step = max(2, len(w) // random.randint(2, 3))
            parts.append('.'.join(w[i:i + step] for i in range(0, len(w), step)) + '.')
        else:
            parts.append(random.choice(vocab))
    sep = []
    for _ in range(n - 1):
        s = random.choice(['/', '/', '/', '/ ', '/', '/'])
        sep.append(s)
    out = parts[0]
    for s, p in zip(sep, parts[1:]):
        out += s + p
    if random.random() < 0.35:
        out += ','
    return out


def gen_code_with_spaces(vocab):
    dept = random.choice(DEPT_WORDS)
    sp1 = ' ' if random.random() < 0.5 else ''
    sp2 = ' ' if random.random() < 0.5 else ''
    s = f'{dept}-{_year()}/{sp1}प्र.क्र.{_d(2)}/{sp2}{random.choice(DEPT_WORDS)}-{_d(1)}'
    if random.random() < 0.4:
        s += random.choice([',', ', दि.', ',दि.'])
    if random.random() < 0.2:
        s = f'क्र.{s}'
    return s


def gen_list_markers(vocab):
    n = random.randint(1, 50)
    dv = ''.join(DEV_DIGITS[int(c)] for c in str(n))
    r = random.random()
    if r < 0.6:
        return f'{dv}.'
    if r < 0.8:
        return f'{dv})'
    return f'{dv}.'


def gen_paren_contrast(vocab):
    w = random.choice(vocab)
    r = random.random()
    if r < 0.30:
        return f'({w})'
    if r < 0.50:
        return f'({w}),'
    if r < 0.64:
        return f'{w}),'
    if r < 0.78:
        return f'({w}'
    return f'{w})'


def gen_punct_contrast(vocab):
    w = random.choice(vocab)
    r = random.random()
    if r < 0.55:
        return w
    if r < 0.72:
        return w + '.'
    if r < 0.86:
        return w + ','
    return w + ':'


def gen_plain(vocab):
    return random.choice(vocab)


CATEGORIES = [
    ('amount_indian', gen_amount_indian, 0.20, True),
    ('dates', gen_dates, 0.13, False),
    ('punct_fragment', gen_punct_fragment, 0.12, True),
    ('dotted_chain', gen_dotted_chain, 0.12, False),
    ('code_with_spaces', gen_code_with_spaces, 0.10, True),
    ('list_markers', gen_list_markers, 0.07, True),
    ('paren_contrast', gen_paren_contrast, 0.10, True),
    ('punct_contrast', gen_punct_contrast, 0.06, True),
    ('plain', gen_plain, 0.10, False),
]
# categories rendered at NATIVE small width (no 150px padding) to teach
# the stretched-tiny-crop geometry seen on real <100px crops
NATIVE_WIDTH_CATS = {'list_markers'}


def render_matched(text, font_path, native=False):
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
    if native:
        return img
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
    ap.add_argument('--seed', type=int, default=66)
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
            img = render_matched(text, font_path, native=cat in NATIVE_WIDTH_CATS)
            name = f'v6_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            out.write(json.dumps({'image_filename': f'marathi/{name}',
                                  'expected_text': text, 'category': cat},
                                 ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = augment_image(render_matched(text, font_path, native=cat in NATIVE_WIDTH_CATS),
                                    seed=args.seed + i * 100 + a)
                aname = f'v6_{i:05d}_aug{a}.png'
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
