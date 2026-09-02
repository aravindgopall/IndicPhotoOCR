"""Generate synthetic word-level training crops for Marathi OCR.

Renders Devanagari text to images using system fonts, focusing on the patterns
that the recognition model struggles with: dates, government codes, document
numbering with parentheses, digit sequences, and mixed-script text.

IMPORTANT: This generates DIFFERENT text from any test image — it creates
similar *patterns* (dates, codes, etc.) but with different actual content,
ensuring proper train/test separation.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ---------------------------------------------------------------------------
# Fonts that render Devanagari correctly on macOS
# ---------------------------------------------------------------------------

# All 5 Devanagari fonts — without libraqm, each codepoint is rendered
# independently, and most fonts handle common Devanagari characters.
# MuktaMahee is the most reliable (handles all chars); others may produce
# tofu for rare conjuncts.  We check glyph availability per-font before use.
ALL_FONTS = [
    '/System/Library/Fonts/MuktaMahee.ttc',
    '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc',
    '/System/Library/Fonts/Supplemental/DevanagariMT.ttc',
    '/System/Library/Fonts/Supplemental/ITFDevanagari.ttc',
    '/System/Library/Fonts/Kohinoor.ttc',
]
FALLBACK_FONT = '/System/Library/Fonts/MuktaMahee.ttc'

# Cache of which fonts support which characters (verified against .notdef)
_font_char_cache: dict = {}
_font_notdef_cache: dict = {}


def _render_char_pattern(font, ch):
    """Render a single char; return dark-pixel boolean array."""
    img = Image.new('L', (120, 120), 255)
    ImageDraw.Draw(img).text((10, 10), ch, font=font, fill=0)
    return np.array(img) < 128


def _get_notdef(font, font_path):
    """Cached .notdef glyph pattern for a font (via unassigned codepoint)."""
    if font_path not in _font_notdef_cache:
        _font_notdef_cache[font_path] = _render_char_pattern(font, '\u0378')
    return _font_notdef_cache[font_path]


def _font_supports_text(font_path: str, text: str) -> bool:
    """Check if a font has real glyphs for ALL characters in text (no tofu).

    A character is tofu when its rendered bitmap is exactly the font's
    .notdef glyph (hollow box) or blank. Plain dark-pixel counting is NOT
    enough -- a tofu box has plenty of dark pixels.
    """
    if font_path not in _font_char_cache:
        _font_char_cache[font_path] = {}
    cache = _font_char_cache[font_path]
    font = None
    notdef = None
    for ch in text:
        if ch == ' ' or ch in cache:
            if ch != ' ' and not cache[ch]:
                return False
            continue
        if font is None:
            font = ImageFont.truetype(font_path, 48)
            notdef = _get_notdef(font, font_path)
        pat = _render_char_pattern(font, ch)
        nd_dark = int(notdef.sum())
        dark = int(pat.sum())
        if dark == 0:
            supported = False          # blank
        elif nd_dark > 0 and dark == nd_dark and int((pat & notdef).sum()) == dark:
            supported = False          # exact .notdef match -> tofu
        else:
            supported = True
        cache[ch] = supported
        if not supported:
            return False
    return True


def _pick_font(text: str) -> str:
    """Pick a random font that supports all characters in text."""
    random.shuffle(ALL_FONTS)
    for font_path in ALL_FONTS:
        if _font_supports_text(font_path, text):
            return font_path
    return FALLBACK_FONT

# Devanagari digits
DEV_DIGITS = '०१२३४५६७८९'
ENG_DIGITS = '0123456789'


def _rand_digits(n: int, devanagari: bool = True) -> str:
    table = DEV_DIGITS if devanagari else ENG_DIGITS
    return ''.join(random.choice(table) for _ in range(n))


def _rand_year(devanagari: bool = True) -> str:
    year = random.randint(1950, 2026)
    s = str(year)
    if devanagari:
        return ''.join(DEV_DIGITS[int(c)] for c in s)
    return s


# ---------------------------------------------------------------------------
# Pattern generators — produce (text, category) tuples
# ---------------------------------------------------------------------------

# Common Marathi government words for realistic context
DEPARTMENTS = [
    'वित्त', 'शिक्षण', 'आरोग्य', 'सार्वजनिक', 'गृह', 'कृषी', 'उद्योग',
    'ऊर्जा', 'पर्यावरण', 'समाजकल्याण', 'महिला', 'विधी', 'संसदीय',
    'तंत्रशिक्षण', 'पाणीपुरवठा', 'वन', 'परिवहन', 'भूमीअभिलेख', 'महसूल',
    'सहकार', 'वस्तुपुरवठा', 'खरेदी', 'कामगार', 'नियोजन', 'विज्ञान',
]
DEPT_SUFFIX = ['विभाग', 'मंत्रालय', 'सचिवालय', 'कार्यालय']
CODE_PREFIXES = ['ससं', 'शिसु', 'आरोग्य', 'वित्त', 'शिक्ष', 'कृषि', 'वन', 'उद्योग']
CODE_PARTS = ['प्र.क्र.', 'क्र.', 'प्रक्र.', 'अ.क्र.', 'वि.क्र.']
SUBJECTS = ['भाषा', 'वित्त', 'शिक्षा', 'आरोग्य', 'वन', 'कृषी', 'ऊर्जा', 'नियोजन']
MONTHS_DEV = ['०१', '०२', '०३', '०४', '०५', '०६', '०७', '०८', '०९', '१०', '११', '१२']
MONTHS_ENG = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

LEGAL_WORDS = [
    'अधिनियम', 'नियम', 'धारा', 'उपधारा', 'कलम', 'परिशिष्ट', 'अनुसूची',
    'संशोधन', 'अधिसूचना', 'परिपत्रक', 'शासन', 'निर्णय', 'आदेश', 'विधेयक',
    'महाराष्ट्र', 'शासन', 'राजपत्र', 'असाधारण', 'क्रमांक', 'प्राधिकृत',
    'प्रकाशन', 'विधी', 'न्याय', 'विभाग', 'मंत्रालय', 'सचिव',
]
SECTION_LETTERS = ['अ', 'ब', 'क', 'ड', 'इ', 'फ', 'ग', 'ह', 'ज', 'जे']


def gen_date() -> str:
    """Generate a date in various formats: DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY"""
    dev = random.random() < 0.7  # 70% Devanagari digits
    dd = random.choice(MONTHS_DEV if dev else MONTHS_ENG)
    # Use day 01-31
    day = random.randint(1, 28)
    if dev:
        dd_str = DEV_DIGITS[day // 10] + DEV_DIGITS[day % 10]
    else:
        dd_str = f'{day:02d}'
    yy = _rand_year(dev)
    sep = random.choice(['.', '-', '/'])
    return f'{dd_str}{sep}{dd}{sep}{yy}'


def gen_gov_code() -> str:
    """Generate a government reference code like ससं-२०२४/प्र.क्र.१०१/भाषा-३"""
    dev = random.random() < 0.7
    prefix = random.choice(CODE_PREFIXES)
    year = _rand_year(dev)
    part = random.choice(CODE_PARTS)
    num = random.randint(1, 999)
    subj = random.choice(SUBJECTS)
    sub_num = random.randint(1, 20)
    if dev:
        num_s = ''.join(DEV_DIGITS[int(c)] for c in str(num))
        sub_num_s = ''.join(DEV_DIGITS[int(c)] for c in str(sub_num))
    else:
        num_s = str(num)
        sub_num_s = str(sub_num)
    # Random format variations
    fmt = random.randint(0, 3)
    if fmt == 0:
        return f'{prefix}-{year}/{part}{num_s}/{subj}-{sub_num_s}'
    elif fmt == 1:
        return f'{prefix}-{year}/{part}.{num_s}/{subj}-{sub_num_s}'
    elif fmt == 2:
        return f'{prefix}-{year}/{part}{num_s}/{subj}{sub_num_s}'
    else:
        return f'{prefix}-{year}/{part}{num_s}'


def gen_doc_number() -> str:
    """Generate document numbering: (१), (२), (अ), etc."""
    dev = random.random() < 0.7
    if dev and random.random() < 0.7:
        n = random.randint(1, 30)
        n_s = ''.join(DEV_DIGITS[int(c)] for c in str(n))
        return f'({n_s})'
    else:
        return f'({random.choice(SECTION_LETTERS)})'


def gen_digit_with_special() -> str:
    """Generate digits mixed with special chars: 10., 26,, 1932, ४०६,"""
    dev = random.random() < 0.7
    n = random.randint(1, 9999)
    if dev:
        num_s = ''.join(DEV_DIGITS[int(c)] for c in str(n))
    else:
        num_s = str(n)
    suffix = random.choice(['.', ',', '-', '/', ':', ';', ')', '(', '%'])
    prefix = random.choice(['', '(', '.', '-', '/'])
    if random.random() < 0.5:
        return f'{prefix}{num_s}{suffix}'
    else:
        return f'{num_s}{suffix}'


def gen_long_digits() -> str:
    """Generate long digit sequences (10-20 digits)"""
    dev = random.random() < 0.5
    n = random.randint(10, 20)
    return _rand_digits(n, dev)


def gen_mixed_word() -> str:
    """Generate a Marathi word + digit/special combo: धारा४५, नियम-१२,"""
    word = random.choice(LEGAL_WORDS)
    dev = random.random() < 0.7
    n = random.randint(1, 200)
    if dev:
        num_s = ''.join(DEV_DIGITS[int(c)] for c in str(n))
    else:
        num_s = str(n)
    sep = random.choice(['', '-', '.', '/', ':'])
    suffix = random.choice(['', '.', ',', '-', ':'])
    return f'{word}{sep}{num_s}{suffix}'


def gen_pure_devanagari() -> str:
    """Generate a single Marathi word"""
    return random.choice(LEGAL_WORDS + DEPARTMENTS + DEPT_SUFFIX + SUBJECTS)


def gen_dept_ref() -> str:
    """Generate department reference: वित्त विभाग, शिक्षण मंत्रालय"""
    return f'{random.choice(DEPARTMENTS)} {random.choice(DEPT_SUFFIX)}'


def gen_section_ref() -> str:
    """Generate section reference: धारा ४५, नियम-१२, कलम ३"""
    word = random.choice(['धारा', 'नियम', 'कलम', 'उपधारा'])
    dev = random.random() < 0.7
    n = random.randint(1, 100)
    if dev:
        num_s = ''.join(DEV_DIGITS[int(c)] for c in str(n))
    else:
        num_s = str(n)
    sep = random.choice([' ', '-', '.'])
    return f'{word}{sep}{num_s}'


def gen_url() -> str:
    """Generate a website URL: maharashtra.gov.in"""
    names = ['maharashtra', 'maharashtra.gov', 'mumbai', 'pune', 'nagpur', 'govt', 'india']
    tlds = ['.gov.in', '.nic.in', '.org', '.com', '.in']
    return f'{random.choice(names)}{random.choice(tlds)}'


# Pattern registry with weights (higher = more samples)
PATTERNS = [
    (gen_date, 15),
    (gen_gov_code, 15),
    (gen_doc_number, 10),
    (gen_digit_with_special, 15),
    (gen_long_digits, 5),
    (gen_mixed_word, 15),
    (gen_pure_devanagari, 10),
    (gen_dept_ref, 5),
    (gen_section_ref, 5),
    (gen_url, 5),
]


# ---------------------------------------------------------------------------
# Image renderer
# ---------------------------------------------------------------------------

def render_text(
    text: str,
    font_path: str,
    font_size: int = 32,
    padding: int = 4,
    bg_color: int = 255,
    text_color: int = 0,
) -> Image.Image:
    """Render text to a PIL image with padding."""
    font = ImageFont.truetype(font_path, font_size)
    # Measure text
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # Create image
    w = text_w + padding * 2
    h = text_h + padding * 2 + 4  # extra for descenders
    img = Image.new('RGB', (max(w, 1), max(h, 1)), (bg_color, bg_color, bg_color))
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=(text_color, text_color, text_color))
    return img


def render_text_variant(
    text: str,
    font_path: str,
    seed: int = 0,
) -> Image.Image:
    """Render text with random size, background, and text color variation.

    Each call produces a visually different image (different font size,
    background shade, text shade, padding) — not just an augmented copy.
    """
    rng = random.Random(seed)
    font_size = rng.randint(24, 42)
    # Vary background: white, light gray, light cream
    bg_val = rng.randint(225, 255)
    # Vary text color: black, dark gray, dark brown
    text_val = rng.randint(0, 50)
    padding = rng.randint(3, 8)
    return render_text(text, font_path, font_size=font_size, padding=padding,
                       bg_color=bg_val, text_color=text_val)


def augment_image(img: Image.Image, seed: int = 0) -> Image.Image:
    """Apply augmentation to a rendered image. Always applies at least 2 transforms."""
    rng = random.Random(seed)
    out = img.copy()
    transforms_applied = 0

    # Rotation (always applied, varied angle)
    angle = rng.uniform(-3, 3)
    bg_color = (rng.randint(220, 255),) * 3
    out = out.rotate(angle, expand=True, fillcolor=bg_color)
    transforms_applied += 1

    # Brightness variation (always applied)
    factor = rng.uniform(0.75, 1.2)
    out = Image.eval(out, lambda v: min(255, max(0, int(v * factor))))
    transforms_applied += 1

    # Blur (50% chance, stronger)
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
        transforms_applied += 1

    # Gaussian noise (60% chance, stronger)
    if rng.random() < 0.6:
        arr = np.array(out, dtype=np.float32)
        noise = np.random.normal(0, rng.uniform(5, 20), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        transforms_applied += 1

    # Background tint (40% chance)
    if rng.random() < 0.4:
        arr = np.array(out)
        mask = arr > 200  # near-white pixels
        tint = rng.randint(-30, -5)
        arr[mask] = np.clip(arr[mask].astype(int) + tint, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        transforms_applied += 1

    # Horizontal compression/stretch (30% chance — simulates different aspect ratios)
    if rng.random() < 0.3:
        w, h = out.size
        scale = rng.uniform(0.85, 1.15)
        new_w = max(1, int(w * scale))
        out = out.resize((new_w, h), Image.BILINEAR)
        transforms_applied += 1

    return out


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: str,
    num_samples: int = 5000,
    num_augment: int = 2,
    seed: int = 42,
) -> dict:
    """Generate a synthetic word-crop dataset.

    Args:
        output_dir: Output directory for crops + labels.jsonl.
        num_samples: Number of unique text samples to generate.
        num_augment: Augmented variants per sample.
        seed: RNG seed.

    Returns:
        Summary dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    lang_subdir = os.path.join(output_dir, 'marathi')
    os.makedirs(lang_subdir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, 'labels.jsonl')

    # Build weighted pattern list
    pattern_pool = []
    for gen_fn, weight in PATTERNS:
        pattern_pool.extend([gen_fn] * weight)

    total = 0
    category_counts = {}
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            gen_fn = random.choice(pattern_pool)
            text = gen_fn()
            cat = gen_fn.__name__.replace('gen_', '')
            category_counts[cat] = category_counts.get(cat, 0) + 1

            # Pick a font that supports all chars in this text (avoids tofu)
            font_path = _pick_font(text)
            font_size = random.randint(32, 48)

            # Render original
            img = render_text(text, font_path, font_size=font_size)

            # Save original
            crop_name = f'syn_{i:05d}.png'
            crop_path = os.path.join(lang_subdir, crop_name)
            img.save(crop_path)
            rel = os.path.relpath(crop_path, output_dir)
            f.write(json.dumps({
                'image_filename': rel, 'expected_text': text, 'language': 'marathi',
            }, ensure_ascii=False) + '\n')
            total += 1

            # Save augmented variants — each is an INDEPENDENT render + augmentation,
            # not just a mild perturbation of the same image.
            for a in range(num_augment):
                aug_img = render_text(text, font_path, font_size=random.randint(32, 48))
                aug_img = augment_image(aug_img, seed=seed + i * 200 + a)
                aug_name = f'syn_{i:05d}_aug{a}.png'
                aug_path = os.path.join(lang_subdir, aug_name)
                aug_img.save(aug_path)
                rel_a = os.path.relpath(aug_path, output_dir)
                f.write(json.dumps({
                    'image_filename': rel_a, 'expected_text': text, 'language': 'marathi',
                }, ensure_ascii=False) + '\n')
                total += 1

    return {
        'output_dir': output_dir,
        'jsonl_path': jsonl_path,
        'total_samples': total,
        'unique_texts': num_samples,
        'category_counts': category_counts,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic Marathi OCR training data.')
    parser.add_argument('--output-dir', '-o', required=True)
    parser.add_argument('--num-samples', '-n', type=int, default=5000)
    parser.add_argument('--augment', '-a', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    result = generate_dataset(args.output_dir, args.num_samples, args.augment, args.seed)
    print(f'Synthetic dataset generated:')
    for k, v in result.items():
        if k == 'category_counts':
            print(f'  {k}:')
            for cat, count in sorted(v.items()):
                print(f'    {cat}: {count}')
        else:
            print(f'  {k}: {v}')
