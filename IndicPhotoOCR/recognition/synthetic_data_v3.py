"""Corrective synthetic data generator (v3) — targets stage3's remaining failures.

Failure analysis (62 misses on the held-out document):
  1. Punctuation over-prediction ~20: adds , . : to bare words; swaps , <-> .
     -> punct_contrast: same word rendered bare vs with each punctuation mark,
        with BARE dominating (55%) to unlearn over-prediction.
  2. Digit value/script errors ~15: १५->9५ (script mix), ०१->०५, ५.->५५
     -> digit_drills: Devanagari-only digits, numbered lists, dates, long runs.
  3. Conjunct/matras ~8: शुद्धी->शुध्दी, संकेतांक->संकेताक, करून->करुन
     -> conjunct drills with correct spellings only (no wrong variants).
  4. Prefix confusion: क्र.->मा./ब.  -> prefix_codes with full gov codes.
  5. Names: शेखर->दोखर  -> name drills.
  6. Spaces: या नावाऐवजी->या.नावारेवजी  -> internal-space drills.
  7. Visarga vs colon: क्रमांकः vs क्रमांक:  -> both forms drilled.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

from IndicPhotoOCR.recognition.synthetic_data import (
    render_text, augment_image, _pick_font, DEV_DIGITS,
)

# ---------------------------------------------------------------------------
# Vocabulary
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
    'सादर', 'संलग्न', 'प्रत', 'नुसार', 'संबंधी', 'कडे', 'मागणी', 'जाहीर', 'विषय',
    'संदर्भ', 'पत्ता', 'कार्यालय', 'स्थळ', 'मिळाला', 'पाठविला', 'जारी', 'मुहूर्त',
]

# Conjunct/matra confusion families — CORRECT spellings only
CONJUNCT_WORDS = [
    # द्ध family (the शुद्धी -> शुध्दी confusion)
    'शुद्ध', 'शुद्धी', 'शुद्धीपत्रक', 'शुद्धता', 'शुद्धलेखन', 'बुद्ध', 'बुद्धी',
    'बुद्धिमत्ता', 'सिद्ध', 'सिद्धी', 'युद्ध', 'युद्धार्थ', 'विद्ध', 'प्रसिद्ध',
    'प्रसिद्धी', 'संप्रदाय', 'विस्मृत',
    # ां (anunasika) family: संकेतांक -> संकेताक
    'संकेतांक', 'संकेतस्थळावर', 'विद्यार्थी', 'विद्यांक', 'संख्यांक', 'अंक',
    'प्रांत', 'स्थानांतर', 'विभागांचे', 'मंडळांच्या', 'कार्यांत', 'विषयांचे',
    # ू vs ु family: करून -> करुन
    'करून', 'करूण', 'सूचना', 'मूल्य', 'सूर्य', 'पूर्ण', 'प्रारूप', 'वापरून',
    'घेऊन', 'देऊन', 'होऊन', 'जाऊन', 'पाठवून', 'सादर करून',
    # य्य family: सहाय्यक -> सहायक
    'सहाय्य', 'सहाय्यक', 'सहाय्यनिधी', 'अधिसूचना', 'स्वयं', 'प्रार्थना',
    'अभ्यास', 'व्यवस्थापन', 'व्यसन', 'अनुदान',
    # स्भा / स्भ family: विधानसभा -> विधानभा
    'विधानसभा', 'विधानसभेच्या', 'लोकसभा', 'राज्यसभा', 'सभागृह', 'सभापती',
    'सभासद', 'सभा',
    # misc matras
    'उपलब्ध', 'संकेत', 'विधानभवन', 'प्रशासन', 'स्वाक्षरी',
]

PREFIXES = ['क्र.', 'सं.', 'वि.', 'प्र.', 'शा.', 'अ.', 'ब.', 'मा.', 'नि.', 'गा.']
CODE_WORDS = ['सारसं', 'ससं', 'विसं', 'प्रशा', 'सेवा', 'भाषा', 'नियं', 'अधी', 'महा',
              'कायदा', 'वित्त', 'गृह', 'शिक्षण', 'कृषी', 'नगर', 'जल', 'वन', 'आयोग']
DEPT_SUFFIXES = ['भाषा-३', 'भाषा-२', 'भाषा-१', 'अ-२', 'ब-१', 'क-३', 'प-१', 'स-२']

NAMES_DEV = ['शेखर', 'रोखर', 'डांगरे', 'नितीन', 'जो.डांगरे', 'राजेश', 'सुनील', 'अमोल',
             'प्रकाश', 'किरण', 'महेश', 'विक्रम', 'पाटील', 'जोशी', 'कदम', 'शिंदे',
             'देशमुख', 'साळवी', 'जाधव', 'ठोरात', 'श्री.शेखर', 'श्री.रोखर']
NAMES_LATIN = ['NITIN', 'DANGARE', 'JOTIRAM', 'RAJESH', 'SUNIL', 'PRAKASH', 'AMOL',
               'PATIL', 'JOSHI', 'KADAM', 'SHINDE', 'DESHMUKH']
URLS = ['www.maharashtra.gov.in', 'maharashtra.gov.in', 'www.vidhan.maharashtra.gov.in',
        'www.india.gov.in', 'www.mumbailyellow.in']

SPACE_WORDS = ['या नावाऐवजी', 'नवीन प्रशासन', 'सदर शुद्धीपत्रक', 'मा. मुख्य',
               'श्री. शेखर', 'अपर मुख्य', 'वरिष्ठ सचिव', 'उप सचिव']

VISARGA_PAIRS = ['क्रमांकः', 'क्रमांक:', 'दिनांकः', 'दिनांक:', 'संकेतस्थळ:',
                 'संदर्भ:', 'विषय:', 'पत्ता:', 'स्थळ:', 'कार्यालय:']


def _dd(n):
    return ''.join(random.choice(DEV_DIGITS) for _ in range(n))


def _to_dev(s):
    return ''.join(DEV_DIGITS[int(c)] if c.isdigit() else c for c in s)


def _date():
    return _to_dev(f'{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(2015, 2026)}')


def _year():
    return _to_dev(str(random.randint(2018, 2026)))


# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

def gen_punct_contrast():
    """Same word across punctuation states — BARE dominates to fix over-prediction."""
    w = random.choice(GOV_WORDS + CONJUNCT_WORDS)
    r = random.random()
    if r < 0.74:
        return w                      # bare
    if r < 0.85:
        return w + '.'
    if r < 0.94:
        return w + ','
    if r < 0.97:
        return w + ':'
    return w + ' :-'


def gen_numbered_list():
    """Digit + period (५.) — attacks ५.->५५ and ११.->9१ confusions."""
    n = random.randint(1, 25)
    d = _to_dev(str(n))
    return random.choice([f'{d}.', f'{d}. ', f'({d})', f'{d})'])


def gen_digit_drills():
    r = random.random()
    if r < 0.25:
        return _date()                                   # dd.mm.yyyy
    if r < 0.45:
        return _dd(random.randint(12, 21))               # long run
    if r < 0.60:
        return _dd(6)                                    # pincode-like
    if r < 0.75:
        # confusable pairs: १/९ ०/५ ७/८ mixed with others
        confusables = '१९०५७८'
        return ''.join(random.choice(confusables + DEV_DIGITS) for _ in range(random.randint(3, 8)))
    if r < 0.90:
        return f'{_dd(1)}.{_dd(1)}'                      # short dotted
    return f'{_dd(2)}/{_dd(2)}'                          # slashed


def gen_conjunct():
    w = random.choice(CONJUNCT_WORDS)
    if random.random() < 0.15:
        w += random.choice(['.', ','])
    return w


def gen_prefix_codes():
    prefix = random.choice(PREFIXES)
    word = random.choice(CODE_WORDS)
    year = _year()
    num = _to_dev(str(random.randint(15, 999)))
    suffix = random.choice(DEPT_SUFFIXES)
    r = random.random()
    if r < 0.20:
        return prefix                                    # bare prefix drill
    if r < 0.40:
        return f'{prefix}{word}-{year}/प्र.क्र.{num}/{suffix},दि.{_date()}'
    if r < 0.60:
        return f'{word}-{year}/प्र.क्र.{num}/{suffix},'
    if r < 0.80:
        return f'{prefix}{word}-{year}/प्र.क्र.{num}/{suffix}'
    return f'{prefix}क्र.{num}'


def gen_plain():
    return random.choice(GOV_WORDS)


def gen_names():
    r = random.random()
    if r < 0.35:
        return random.choice(NAMES_LATIN)
    if r < 0.55:
        return 'www.' + random.choice(['maharashtra', 'india', 'mumbai']) + '.gov.in'
    if r < 0.80:
        return random.choice(NAMES_DEV)
    n = random.choice(['नितीन', 'राजेश', 'सुनील', 'अमोल', 'प्रकाश'])
    ini = random.choice(['जो.', 'पा.', 'कु.', 'रा.', 'शि.', 'दे.', 'सा.'])
    s = random.choice(['डांगरे', 'पाटील', 'जोशी', 'कदम', 'शिंदे', 'देशमुख'])
    return f'({n} {ini}{s})' if random.random() < 0.6 else f'{n} {ini}{s}'


def gen_spaces():
    w = random.choice(SPACE_WORDS)
    if random.random() < 0.3:
        return f'{_dd(1)} ' + random.choice(['वा', 'रा', 'री'])
    return w


def gen_multi_comma():
    return ','.join(random.sample(GOV_WORDS, random.randint(2, 3)))


def gen_visarga():
    return random.choice(VISARGA_PAIRS)


CATEGORIES = [
    ('punct_contrast', gen_punct_contrast, 0.24, False),
    ('numbered_list', gen_numbered_list, 0.08, False),
    ('digit_drills', gen_digit_drills, 0.17, True),
    ('conjunct', gen_conjunct, 0.15, False),
    ('prefix_codes', gen_prefix_codes, 0.12, True),
    ('plain', gen_plain, 0.12, False),
    ('names', gen_names, 0.06, True),
    ('spaces', gen_spaces, 0.04, False),
    ('multi_comma', gen_multi_comma, 0.03, True),
    ('visarga', gen_visarga, 0.02, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-dir', required=True)
    ap.add_argument('-n', '--num-texts', type=int, default=4200)
    ap.add_argument('-a', '--num-augment', type=int, default=2)
    ap.add_argument('--seed', type=int, default=777)
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
            font_size = random.randint(30, 48)
            img = render_text(text, font_path, font_size=font_size)
            name = f'syn3_{i:05d}.png'
            img.save(os.path.join(lang_subdir, name))
            out.write(json.dumps({'image_filename': f'marathi/{name}',
                                  'expected_text': text,
                                  'category': cat}, ensure_ascii=False) + '\n')
            total += 1
            for a in range(args.num_augment):
                aug = render_text(text, font_path, font_size=random.randint(30, 48))
                aug = augment_image(aug, seed=args.seed + i * 200 + a)
                aname = f'syn3_{i:05d}_aug{a}.png'
                aug.save(os.path.join(lang_subdir, aname))
                out.write(json.dumps({'image_filename': f'marathi/{aname}',
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
