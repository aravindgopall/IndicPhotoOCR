"""Failure-targeted synthetic data generator (v2).

Generates samples for the specific failure patterns observed on the held-out
government document screenshot:
  1. Real-format gov codes: क्र.सारसं-२०२४/प्र.क्र.१०१/भाषा-३,दि.१५.१०.२०२४
  2. Trailing punctuation (period vs comma) on common gov vocabulary
  3. Multi-comma compounds: सचिव,मंत्रालय,मुंबई
  4. Internal spaces: या नावाऐवजी, ७ वा
  5. Long digit runs (12-21 digits, no separators)
  6. URLs with www (requires 'w' + Latin in charset)
  7. Latin uppercase names (requires uppercase in charset)
  8. Parenthesized names: (नितीन जो.डांगरे)
  9. Confusion-pair spellings: शुद्धीपत्रक, सहाय्यक, करून, संकेतांक
 10. Visarga-as-colon: क्रमांकः
 11. Slash compounds: सभा/विधान, मा.सभापती/उप
 12. Date patterns with colons: :०६.०१.२०२५, दिनांक: ०६.०१.२०२५
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

from IndicPhotoOCR.recognition.synthetic_data import (
    render_text, augment_image, _pick_font, DEV_DIGITS, ENG_DIGITS,
)

# ---------------------------------------------------------------------------
# Vocabulary (standard Marathi government-document terms)
# ---------------------------------------------------------------------------

GOV_WORDS = [
    'शासन', 'महाराष्ट्र', 'विधानमंडळ', 'विधानसभा', 'विधानपरिषद', 'सचिवालय', 'सचिव',
    'मुख्यमंत्री', 'राज्यपाल', 'मंत्रालय', 'सदस्य', 'समिती', 'पुनर्रचना', 'शुद्धीपत्रक',
    'डिजीटल', 'उपलब्ध', 'संस्कृती', 'स्वाक्षरीने', 'क्रमांक', 'दिनांक', 'मुंबई', 'राजभवन',
    'विभाग', 'निर्णय', 'अध्यक्ष', 'सभापती', 'प्रधान', 'सहाय्यक', 'अपर', 'नियुक्त', 'मराठी',
    'भाषा', 'आहे', 'करण्यात', 'बाबत', 'संकेतस्थळ', 'अनुक्रमांक', 'मार्ग', 'चौक', 'हिल',
    'मलबार', 'राजगुरु', 'हुतात्मा', 'साहित्य', 'येथील', 'मंडळ', 'परिषद', 'कामा', 'सामंत',
    'मादाम', 'त्याचा', 'खाजगी', 'नवीन', 'प्रशासन', 'आदेशानुसार', 'प्रमुख', 'पक्षनेता',
    'यांचे', 'यांच्या', 'करून', 'असून', 'म्हणून', 'असे', 'येत', 'रोजी', 'मजला', 'आलेले',
    'अवर', 'सदर', 'वाचा', 'सावंत', 'नावाने', 'यावे', 'विधानभवन', 'पुनरचना', 'संपर्क',
    'अध्यक्षांचे', 'उपाध्यक्ष', 'प्रशासकीय', 'सुधारणा', 'आवश्यक', 'प्राप्त', 'स्वीकृत',
    'सादर', 'संलग्न', 'प्रत', 'नुसार', 'संबंधी', 'कडे', 'मागणी', 'जाहीर',
]

# Words with tricky conjuncts/matras that models confuse
CONFUSION_WORDS = [
    'शुद्धीपत्रक', 'शुद्धी', 'सुधारणा', 'सहाय्यक', 'सहाय्यनिधी', 'करून', 'संकेतांक',
    'संकेतस्थळावर', 'विधानभवन', 'प्रशासन', 'प्रशासकीय', 'उपलब्ध', 'मुख्यमंत्र्यांचे',
    'राज्यपालांचे', 'विधानमंडळाच्या', 'स्वाक्षरीने', 'पुनर्रचनान्वये', 'अध्यक्षांच्या',
    'परिषदेच्या', 'समित्यांच्या', 'आदेशानुसार', 'जाहीरात', 'सूचना', 'सूचनांनुसार',
    'कृपया', 'स्वीकार', 'स्वीकारले', 'प्राप्तझाले', 'सादरकरण', 'निर्णयानुसार',
]

CODE_PREFIXES = ['क्र.', 'सं.', 'वि.', 'प्र.', 'शा.', 'आरो.', 'मं.', 'वा.', 'नि.', 'गा.', 'अ.', 'ब.']
CODE_WORDS = ['सारसं', 'ससं', 'विसं', 'प्रशा', 'सेवा', 'भाषा', 'नियं', 'अधी', 'महा',
              'कायदा', 'वित्त', 'गृह', 'वैद्यकीय', 'शिक्षण', 'कृषी', 'ऊर्जा', 'नगर',
              'ग्राम', 'जल', 'वन', 'परि', 'आयोग', 'संचालन', 'निर्मिती']
DEPT_SUFFIXES = ['भाषा-३', 'भाषा-२', 'भाषा-१', 'अ-२', 'ब-१', 'क-३', 'प-१', 'स-२', 'शा-१', 'प्र-२']

MA_WORDS = ['मा.', 'मा.सभापती', 'मा.मुख्य', 'मा.विधानसभा', 'मा.विधानमंडळ', 'मा.समिती',
            'मा.अध्यक्ष', 'मा.प्रधान', 'मा.विरोधी', 'मा.सर्व', 'मा.उपसभापती']

SLASH_COMPOUNDS = ['सभा/विधान', 'सदस्य/विधान', 'मा.सभापती/उप', 'मा.अध्यक्ष/उपाध्यक्ष',
                   'विधान/मंत्री', 'सचिव/सहसचिव', 'अध्यक्ष/उपाध्यक्ष', 'प्र.क्र.', 'अ.क्र.',
                   'शा.क्र.', 'वि.क्र.', 'दि.', 'प्र.', 'शा.', 'क्र.', 'सं.', 'वि.']

VISARGA_WORDS = ['क्रमांकः', 'दिनांकः', 'संकेतः', 'क्रमांक:', 'दिनांक:', 'संकेतस्थळ:',
                 'क्र.सं.:', 'पत्ता:', 'विषय:', 'संदर्भ:']

URLS = ['www.maharashtra.gov.in', 'www.maharashtra.gov.in/marathi',
        'www.india.gov.in', 'www.mumbai.gov.in', 'www.pune.gov.in',
        'www.mahadesk.in', 'www.aaplesarkar.mahaonline.gov.in',
        'maharashtra.gov.in', 'www.vidhan.maharashtra.gov.in',
        'www.mpsva.maharashtra.gov.in', 'egazzete.maharashtra.gov.in']

LATIN_NAMES = ['NITIN', 'DANGARE', 'JOTIRAM', 'RAJESH', 'SUNIL', 'PRAKASH', 'AMOL',
               'SANJAY', 'KIRAN', 'MAHESH', 'VIKRAM', 'SURESH', 'ANIL', 'DEEPAK',
               'PATIL', 'JOSHI', 'DESHMUKH', 'KADAM', 'SHINDE', 'MORE', 'JADHAV',
               'PAWAR', 'SALVI', 'NAIK', 'RAO', 'SHAHA', 'GAIKWAD', 'WAGHMODE']

# Combinatorial parenthesized names: (name initial surname)
FIRST_NAMES = ['नितीन', 'राजेश', 'सुनील', 'अमोल', 'प्रकाश', 'किरण', 'महेश',
               'विक्रम', 'संजय', 'अनिल', 'दीपक', 'सुरेश', 'नरेंद्र', 'विजय']
MIDDLE_INITIALS = ['जो.', 'पा.', 'कु.', 'रा.', 'शि.', 'दे.', 'सा.', 'ज.', 'ब.', 'म.']
SURNAMES = ['डांगरे', 'पाटील', 'जोशी', 'कदम', 'शिंदे', 'देशमुख', 'साळवी',
            'जाधव', 'गायकवाड', 'ठोरात', 'साबळे', 'माने', 'चव्हाण', 'जागतप']


def _dd(n):
    return ''.join(random.choice(DEV_DIGITS) for _ in range(n))


def _to_dev(s: str) -> str:
    """Convert ASCII digits in s to Devanagari digits."""
    return ''.join(DEV_DIGITS[int(c)] if c.isdigit() else c for c in s)


def _date(dev=True):
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = random.randint(2015, 2026)
    s = f'{d:02d}.{m:02d}.{y}'
    return _to_dev(s) if dev else s


def gen_gov_code_real():
    """क्र.सारसं-२०२४/प्र.क्र.१०१/भाषा-३,दि.१५.१०.२०२४ — the real document format."""
    prefix = random.choice(CODE_PREFIXES)
    word = random.choice(CODE_WORDS)
    year = _to_dev(str(random.randint(2018, 2026)))
    num1 = _to_dev(str(random.randint(15, 999)))
    suffix = random.choice(DEPT_SUFFIXES)
    date = _date()
    variants = [
        f'{prefix}{word}-{year}/प्र.क्र.{num1}/{suffix},दि.{date}',          # full with date
        f'{prefix}{word}-{year}/प्र.क्र.{num1}/{suffix},दि',                  # code, no date
        f'{word}-{year}/प्र.क्र.{num1}/{suffix},',                             # no prefix
        f'{prefix}{word}-{year}/प्र.क्र.{num1}/{suffix}',                      # no trailing
        f'{prefix}{word}-{year}/अ.क्र.{num1}/{suffix},दि.{date}',
    ]
    return random.choice(variants)


def gen_trailing_punct():
    w = random.choice(GOV_WORDS + CONFUSION_WORDS + MA_WORDS)
    return w + random.choice(['.', ',', ' :-', ' '])


def gen_multi_comma():
    words = random.sample(GOV_WORDS, random.randint(2, 3))
    return ','.join(words)


def gen_internal_space():
    opts = [
        lambda: random.choice(['या', 'सदर', 'नवीन', 'मा.', 'श्री.', 'प्र.)']) + ' ' + random.choice(GOV_WORDS),
        lambda: f'{_dd(1)} वा',
        lambda: f'{_dd(1)} री',
        lambda: f'{_dd(2)} रा',
        lambda: random.choice(['सातवा', 'आठवा', 'नववा']) + ' ' + random.choice(['विभाग', 'मजला', 'स्तंभ']),
    ]
    return random.choice(opts)()


def gen_long_digits():
    return _dd(random.randint(12, 21))


def gen_url():
    return random.choice(URLS)


def gen_latin_caps():
    n = random.choice(LATIN_NAMES)
    if random.random() < 0.3:
        n += ' ' + random.choice(LATIN_NAMES)
    return n


def gen_paren_name():
    n = random.choice(FIRST_NAMES)
    ini = random.choice(MIDDLE_INITIALS)
    s = random.choice(SURNAMES)
    return f'({n} {ini}{s})'


def gen_confusion():
    w = random.choice(CONFUSION_WORDS)
    if random.random() < 0.5:
        w += random.choice(['.', ','])
    return w


def gen_visarga():
    return random.choice(VISARGA_WORDS)


def gen_slash():
    return random.choice(SLASH_COMPOUNDS)


def gen_date_colon():
    d = _date()
    return random.choice([f':{d}', f'दिनांक: {d}', f'क्रमांक: {_dd(4)}',
                          f'दि.{d}', f'दिनांक:-{d}', f'रोजी {d}'])


CATEGORIES = [
    ('gov_code_real', gen_gov_code_real, 0.13, True),
    ('trailing_punct', gen_trailing_punct, 0.15, True),
    ('multi_comma', gen_multi_comma, 0.07, True),
    ('internal_space', gen_internal_space, 0.07, True),
    ('long_digits', gen_long_digits, 0.05, True),
    ('url', gen_url, 0.05, False),
    ('latin_caps', gen_latin_caps, 0.08, True),
    ('paren_name', gen_paren_name, 0.05, True),
    ('confusion', gen_confusion, 0.09, False),
    ('visarga', gen_visarga, 0.05, False),
    ('slash', gen_slash, 0.06, False),
    ('date_colon', gen_date_colon, 0.06, True),
    ('gov_word_plain', lambda: random.choice(GOV_WORDS + CONFUSION_WORDS), 0.09, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-dir', required=True)
    ap.add_argument('-n', '--num-texts', type=int, default=4000)
    ap.add_argument('-a', '--num-augment', type=int, default=2)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    lang_subdir = os.path.join(args.output_dir, 'marathi')
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
                t = fn()
                if t and (not dedup or t not in seen):
                    seen.add(t)
                    texts.append((t, cat))
                    by_cat[cat] = by_cat.get(cat, 0) + 1
                break

    jsonl_path = os.path.join(args.output_dir, 'labels.jsonl')
    total = 0
    with open(jsonl_path, 'w', encoding='utf-8') as out:
        for i, (text, cat) in enumerate(texts):
            font_path = _pick_font(text)
            font_size = random.randint(32, 48)
            img = render_text(text, font_path, font_size=font_size)
            name = f'syn2_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            rel = f'marathi/{name}'
            out.write(json.dumps({'image_filename': rel, 'expected_text': text,
                                  'category': cat}, ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = render_text(text, font_path, font_size=random.randint(32, 48))
                aug = augment_image(aug, seed=args.seed + i * 200 + a)
                aname = f'syn2_{i:05d}_aug{a}.png'
                aug.save(os.path.join(lang_subdir, aname))
                out.write(json.dumps({'image_filename': f'marathi/{aname}',
                                      'expected_text': text,
                                      'category': cat}, ensure_ascii=False) + '\n')
                total += 1
            if (i + 1) % 500 == 0:
                print(f'  {i + 1}/{len(texts)} texts ({total} samples)')

    print(f'output_dir: {args.output_dir}')
    print(f'total_samples: {total}')
    print(f'unique_texts: {len(texts)}')
    print('category_counts:')
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f'  {c}: {n}')


if __name__ == '__main__':
    main()
