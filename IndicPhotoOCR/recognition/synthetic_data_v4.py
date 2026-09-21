"""Marathi synthetic v4 — complex-reference-code genre.

Design sources (NO held-out data):
  - 10 training lines from the 100 Val Dataset (grammar, vocab, issue mix)
  - old real training data (marathi_words vocabulary)
  - systematic generation (dept abbreviations, digit drills)

Lessons applied:
  - split composite patterns: code and code+date are SEPARATE samples
  - bare-dominant contrast within punctuated categories
  - Devanagari digits only (this genre's convention)
  - vertical-fill matching real crops (~0.55): text pasted on taller canvas
  - tofu-verified fonts + RAQM
  - no zero-width chars
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
MATRAS = ['', 'ा', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो']  # realistic abbrev vowels
CONJ = ['ग्र', 'स्थ', 'क्र', 'त्र', 'द्व', 'श्र', 'प्र', 'ब्र', 'द्र', 'म्र']

# Realistic dept word fragments — prior training data (synthetic v2 vocabulary)
DEPT_WORDS = ['सारसं', 'ससं', 'विसं', 'प्रशा', 'सेवा', 'भाषा', 'नियं', 'अधी', 'महा',
              'कायदा', 'वित्त', 'गृह', 'वैद्यकीय', 'शिक्षण', 'कृषी', 'ऊर्जा', 'नगर',
              'ग्राम', 'जल', 'वन', 'आस्था', 'अर्थ', 'कार्या', 'प्रतिनि', 'पूरक', 'सनिवे',
              'जनरल', 'प्रशासन', 'निर्माण', 'विकास', 'नियंत्रण', 'पर्यवेक्षण', 'अनुदान']


def _pick_font(text):
    fonts = list(FONTS)
    random.shuffle(fonts)
    for fp in fonts:
        if _font_supports_text(fp, text):
            return fp
    return FONTS[0]


def _dev(n):
    return ''.join(random.choice(DEV_DIGITS) for _ in range(n))


def _dev_num(lo, hi):
    n = random.randint(lo, hi)
    return _dev(n).lstrip('०') or '०'


def _dev_year():
    y = random.randint(2008, 2027)
    return ''.join(DEV_DIGITS[int(d)] for d in str(y))


def _dev_date():
    style = random.random()
    d, m = random.randint(1, 28), random.randint(1, 12)
    dd = f'{DEV_DIGITS[d // 10]}{DEV_DIGITS[d % 10]}'
    mm = f'{DEV_DIGITS[m // 10]}{DEV_DIGITS[m % 10]}'
    if style < 0.6:
        return f'{dd}.{mm}.{_dev_year()}'          # ०९.०४.२०२६
    return f'{dd}/{mm}/{_dev_year()}'               # १५/०२/२०२३


def dept_abbrev(n_units=None, dotted=None):
    """Department abbreviation: realistic word-style or systematic syllables."""
    r = random.random()
    if r < 0.55:
        # real-style word fragment (prior training vocab)
        return random.choice(DEPT_WORDS)
    n = n_units or random.randint(2, 4)
    units = []
    for _ in range(n):
        if random.random() < 0.15:
            units.append(random.choice(CONJ) + random.choice(['', 'ा', 'ि', 'ी']))
        else:
            units.append(random.choice(CONS) + random.choice(MATRAS))
    units = [u for u in units if u]
    if not units:
        units = ['क']
    if dotted if dotted is not None else random.random() < 0.3:
        return '.'.join(units)
    return ''.join(units)


# ---- vocab from TRAINING sources only ---------------------------------------

def load_vocab():
    train_words = set()
    for line in open('data/marathi_v2_train_lines/labels.jsonl', encoding='utf-8'):
        row = json.loads(line)
        for t in row['expected_text'].split():
            t = t.strip('.,:;()"-')
            if 2 <= len(t) <= 15 and re.match(r'^[\u0900-\u097F]+$', t):
                train_words.add(t)
    mw_words = set()
    for line in open('data/marathi_words/labels.jsonl', encoding='utf-8'):
        row = json.loads(line)
        t = row['expected_text'].strip()
        if 2 <= len(t) <= 14 and re.match(r'^[\u0900-\u097F]+$', t):
            mw_words.add(t)
    return sorted(train_words), sorted(mw_words)


MONTHS = ['जानेवारी', 'फेब्रुवारी', 'मार्च', 'एप्रिल', 'मे', 'जून', 'जुलै',
          'ऑगस्ट', 'सप्टेंबर', 'ऑक्टोबर', 'नोव्हेंबर', 'डिसेंबर']


# ---- category generators -----------------------------------------------------

def gen_code_full(vocab):
    """Full code + date (training-line grammar)."""
    r = random.random()
    if r < 0.35:
        prefix = f'{dept_abbrev(dotted=True)}क्र.'
    elif r < 0.55:
        prefix = 'क्र.'
    elif r < 0.7:
        prefix = ''
    else:
        prefix = dept_abbrev()
    body = (f'{prefix}{dept_abbrev()}-{_dev_year()}/प्र.क्र.{_dev_num(1, 3)}/'
            f'{dept_abbrev(random.randint(1, 3))}-{_dev_num(1, 2)}')
    tail = random.choice([', दि.', ', दिनांक ', ',दि.', ', '])
    return body + tail + _dev_date()


def gen_code_no_date(vocab):
    r = random.random()
    if r < 0.3:
        prefix = f'{dept_abbrev(dotted=True)}क्र.'
    elif r < 0.5:
        prefix = 'क्र.'
    else:
        prefix = ''
    body = (f'{prefix}{dept_abbrev()}-{_dev_year()}/प्र.क्र.{_dev_num(1, 3)}/'
            f'{dept_abbrev(random.randint(1, 3))}-{_dev_num(1, 2)}')
    if random.random() < 0.5:
        body += random.choice([',', '.'])
    return body


def gen_code_medium(vocab):
    """10-18 char codes: YYYY/प्र.क्र.NN/dept-N, etc."""
    style = random.random()
    if style < 0.4:
        s = f'{_dev_year()}/प्र.क्र.{_dev_num(1, 3)}/{dept_abbrev(random.randint(1, 2))}-{_dev_num(1, 2)}'
    elif style < 0.6:
        s = f'प्र.क्र.{_dev_num(2, 3)}/{dept_abbrev(2)}-{_dev_num(1, 2)}'
    elif style < 0.8:
        s = f'{_dev_num(4, 5)}/प्र.क्र.{_dev_num(2, 3)}/{_dev_num(1, 2)}-{random.choice("अबक")}'
    else:
        s = f'क्र.{dept_abbrev()}-{_dev_num(3, 5)}/{_dev_num(2, 3)}'
    if random.random() < 0.45:
        s += random.choice([',', '.'])
    return s


def gen_dept_drill(vocab):
    return dept_abbrev()


def gen_digit_drill(vocab):
    """Amounts, comma-grouped, pay scales, dates — training-line formats."""
    r = random.random()
    if r < 0.2:   # रु.६४०.००
        return f'रु.{_dev_num(2, 4)}.{_dev(2)}'
    if r < 0.4:   # Indian comma grouping: १४,७९,३६६
        return f'{_dev_num(2, 3)},{_dev(2)},{_dev(3)}'
    if r < 0.55:  # pay scale range: ३५४००-११२४००
        return f'{_dev(5)}-{_dev(random.randint(5, 6))}'
    if r < 0.7:   # dates
        return random.choice([f'दि.{_dev_date()}', f'दिनांक {_dev_date()}',
                              f'दिनांक {_dev_num(1, 2)} {random.choice(MONTHS)}, {_dev_year()}'])
    if r < 0.85:  # city-pin style: मुंबई-३२.
        return f'{random.choice(["मुंबई", "पुणे", "नागपूर", "औरंगाबाद", "नाशिक"])}-{_dev_num(1, 2)}'
    return _dev_num(2, 6)  # plain digit runs


def gen_punct_contrast(vocab):
    w = random.choice(vocab)
    r = random.random()
    if r < 0.60:
        return w
    if r < 0.78:
        return w + random.choice(['.', ','])
    if r < 0.88:
        return w + ':'
    if r < 0.94:
        return w + ' -'
    return w + ' :-'


def gen_plain(vocab):
    return random.choice(vocab)


def gen_conjunct_drill(vocab):
    """Words with conjuncts (from training + marathi_words vocab)."""
    conj = [w for w in vocab if any(c in w for c in 'द्धत्रक्रद्वश्रष्ट्व्य')]
    return random.choice(conj) if conj else random.choice(vocab)


def gen_standalone_punct(vocab):
    return random.choice(['-', '/', ':', '.', ',', '-'])


def gen_paren(vocab):
    inner = ' '.join(random.sample(vocab, random.randint(1, 3)))
    return f'({inner})'


def gen_separator_drill(vocab):
    """Isolated separator glyphs — trained at padded (in-distribution) geometry
    so piecewise inference can read them."""
    return random.choice(['.', '-', '/', ':', ',', '.', '-'])


def gen_digit_piece(vocab):
    """Isolated 1-4 digit runs — the piecewise code pieces."""
    return _dev_num(1, 4)


CATEGORIES = [
    ('code_medium', gen_code_medium, 0.12, True),
    ('dept_drill', gen_dept_drill, 0.10, False),
    ('digit_drill', gen_digit_drill, 0.10, True),
    ('digit_piece', gen_digit_piece, 0.06, False),
    ('separator_drill', gen_separator_drill, 0.06, False),
    ('punct_contrast', gen_punct_contrast, 0.18, True),
    ('plain', gen_plain, 0.20, False),
    ('conjunct_drill', gen_conjunct_drill, 0.08, True),
    ('standalone_punct', gen_standalone_punct, 0.04, False),
    ('paren', gen_paren, 0.06, True),
]


def render_matched(text, font_path):
    """Render with vertical fill ~0.4-0.7, px/char ~30-57, and min width
    150px (avoids the transform's horizontal stretch on narrow images —
    the bug that made standalone glyphs out-of-distribution)."""
    font_size = random.randint(80, 125)
    img = render_text(text, font_path, font_size=font_size)
    w, h = img.size
    # vertical fill padding - match DETECTOR-box geometry (tight: 0.80-0.98)
    target_fill = random.uniform(0.80, 0.98)
    target_h = max(h + 4, int(h / target_fill))
    if target_h > h:
        canvas = Image.new('RGB', (w, target_h), (255, 255, 255))
        canvas.paste(img, (0, (target_h - h) // 2))
        img = canvas
        h = target_h
    # min-width padding (in-distribution for the 32x128 transform)
    min_w = 150
    if img.size[0] < min_w:
        canvas = Image.new('RGB', (min_w, h), (255, 255, 255))
        canvas.paste(img, ((min_w - img.size[0]) // 2, 0))
        return canvas
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-dir', required=True)
    ap.add_argument('-n', '--num-texts', type=int, default=4500)
    ap.add_argument('-a', '--num-augment', type=int, default=1)
    ap.add_argument('--seed', type=int, default=77)
    args = ap.parse_args()

    random.seed(args.seed)
    train_vocab, mw_vocab = load_vocab()
    vocab = train_vocab + mw_vocab
    print(f'vocab: {len(train_vocab)} train-line + {len(mw_vocab)} marathi_words')

    out_dir = args.output_dir
    lang_subdir = os.path.join(out_dir, 'marathi')
    os.makedirs(lang_subdir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, 'labels.jsonl')

    seen = set()
    texts = []
    by_cat = {}
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
            name = f'v4_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            out.write(json.dumps({'image_filename': f'marathi/{name}',
                                  'expected_text': text, 'category': cat},
                                 ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = render_matched(text, font_path)
                aug = augment_image(aug, seed=args.seed + i * 100 + a)
                aname = f'v4_{i:05d}_aug{a}.png'
                aug.save(os.path.join(lang_subdir, aname))
                out.write(json.dumps({'image_filename': f'marathi/{aname}',
                                      'expected_text': text, 'category': cat},
                                     ensure_ascii=False) + '\n')
                total += 1
            if (i + 1) % 1000 == 0:
                print(f'  {i + 1}/{len(texts)} texts ({total} samples)')

    print(f'output_dir: {out_dir}')
    print(f'total_samples: {total}')
    print(f'unique_texts: {len(texts)}')
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f'  {c}: {n}')


if __name__ == '__main__':
    main()
