"""Marathi synthetic v5 — basic, diagnosis-driven (training data only).

Diagnosed from Run A's IN-SAMPLE errors on real training crops (never eval):
  1. parenthesis endings: (w), vs (w) vs w), confusion cluster
  2. trailing punctuation drops + leading comma hallucination
  3. slash-adjacent spacing (', दि.' vs ',दि.')
  4. dotted abbreviations (सा.आ.वि.) fragile at edges
  5. amounts with parens: १५,००,०००/- (रू.

Design sources: training-line vocabulary + marathi_words + systematic
generation + prior-training dept words. Geometry: tight fill (detector
boxes), min-width 150px. Basic level: ~3.5k texts.
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


def _dev(n):
    return ''.join(random.choice(DEV_DIGITS) for _ in range(n))


def _dev_year():
    y = random.randint(2008, 2027)
    return ''.join(DEV_DIGITS[int(d)] for d in str(y))


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


def dotted_abbrev(n=None):
    n = n or random.randint(2, 4)
    units = []
    for _ in range(n):
        if random.random() < 0.15:
            units.append(random.choice(CONJ) + random.choice(['', 'ा', 'ि']))
        else:
            units.append(random.choice(CONS) + random.choice(MATRAS))
    units = [u for u in units if u] or ['क']
    return '.'.join(units) + '.'


# ---- categories (diagnosis-driven) -------------------------------------------

def gen_paren_contrast(vocab):
    """THE confusion cluster: (w) / (w), / w), / w, / w."""
    w = random.choice(vocab)
    r = random.random()
    if r < 0.30:
        return f'({w})'
    if r < 0.50:
        return f'({w}),'
    if r < 0.62:
        return f'{w}),'
    if r < 0.74:
        return f'({w}'
    if r < 0.86:
        return f'{w})'
    return w


def gen_punct_contrast(vocab):
    """Trailing punctuation discrimination; bare-dominant."""
    w = random.choice(vocab)
    r = random.random()
    if r < 0.55:
        return w
    if r < 0.70:
        return w + '.'
    if r < 0.82:
        return w + ','
    if r < 0.90:
        return w + ':'
    if r < 0.95:
        return w + ' -'
    return w + '.' * 0 + '.,' if False else w + ' :-'


def gen_dotted_abbrev(vocab):
    r = random.random()
    if r < 0.5:
        return dotted_abbrev()
    if r < 0.75:
        # dept word with dots: सा.आ.वि. style from real dept fragments
        w = random.choice(DEPT_WORDS)
        step = max(2, len(w) // random.randint(2, 3))
        parts = [w[i:i + step] for i in range(0, len(w), step)]
        return '.'.join(parts) + '.'
    return random.choice(DEPT_WORDS)


def gen_code_with_spaces(vocab):
    """Codes with slash-adjacent space variants (train distribution)."""
    dept = random.choice(DEPT_WORDS)
    sp1 = ' ' if random.random() < 0.5 else ''
    sp2 = ' ' if random.random() < 0.5 else ''
    s = f'{dept}-{_dev_year()}/{sp1}प्र.क्र.{_dev(2)}/{sp2}{random.choice(DEPT_WORDS)}-{_dev(1)}'
    if random.random() < 0.4:
        s += random.choice([',', ', दि.', ',दि.'])
        if s.endswith('दि.'):
            pass
    if random.random() < 0.2:
        s = f'क्र.{s}'
    return s


def gen_amount_paren(vocab):
    """Amounts with parens/slashes: १५,००,०००/- (रू."""
    r = random.random()
    if r < 0.35:
        return f'{_dev(2)},{_dev(2)},{_dev(3)}/-'
    if r < 0.55:
        return f'({random.choice(["रू.", "रु."])}{_dev(3)}'
    if r < 0.75:
        return f'{_dev(2)},{_dev(2)},{_dev(3)}/- ({random.choice(["रू.", "रु."])}'
    return f'रु.{_dev(2)}.{_dev(2)}'


def gen_digit_drill(vocab):
    r = random.random()
    if r < 0.3:
        return f'{_dev(2)},{_dev(2)},{_dev(3)}'
    if r < 0.5:
        return f'{_dev(5)}-{_dev(5)}'
    if r < 0.7:
        d, m = random.randint(1, 28), random.randint(1, 12)
        return f'दि.{d:02d}.{m:02d}.{_dev_year()}'
    if r < 0.85:
        return f'({random.choice(["सन", "मु.दि."])} {_dev_year()}-{_dev(2)} व {_dev_year()}-{_dev(2)})'
    return _dev(random.randint(2, 6))


def gen_plain(vocab):
    return random.choice(vocab)


CATEGORIES = [
    ('paren_contrast', gen_paren_contrast, 0.22, True),
    ('punct_contrast', gen_punct_contrast, 0.20, True),
    ('dotted_abbrev', gen_dotted_abbrev, 0.13, False),
    ('code_with_spaces', gen_code_with_spaces, 0.12, True),
    ('amount_paren', gen_amount_paren, 0.10, True),
    ('digit_drill', gen_digit_drill, 0.10, True),
    ('plain', gen_plain, 0.13, False),
]


def render_matched(text, font_path):
    """Tight vertical fill (detector geometry) + min width 150px."""
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
    ap.add_argument('--seed', type=int, default=55)
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
            name = f'v5_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            out.write(json.dumps({'image_filename': f'marathi/{name}',
                                  'expected_text': text, 'category': cat},
                                 ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = augment_image(render_matched(text, font_path),
                                    seed=args.seed + i * 100 + a)
                aname = f'v5_{i:05d}_aug{a}.png'
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
