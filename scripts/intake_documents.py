"""Document intake: PDFs / page images / labeled lines -> analysis report.

Day-1 tool for a new batch of production documents. Handles the three input
shapes we have actually seen:

  1. Folder of PDFs            -> pages (pdftoppm) -> line images (detection)
  2. Folder of page images     -> line images (detection)
  3. Labeled line dataset      -> word crops + charset analysis
     (jsonl: image_filename + expected_text, like the Gujarati dataset)

Outputs an analysis report: counts, resolution, script mix, charset
requirements (vs existing checkpoints), digit convention, punctuation
frequency, vocabulary. Optionally pseudo-labels lines with the current best
models (clearly marked as pseudo-GT, never for reported numbers).

Usage:
  python scripts/intake_documents.py --input-dir ~/Downloads/new_docs \
      --outdir data/intake_new [--language auto|marathi|gujarati] \
      [--dpi 300] [--pseudo-label] [--device cpu]
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# Script ranges
RANGES = {
    'devanagari': (0x0900, 0x097F),
    'gujarati': (0x0A80, 0x0AFF),
}
DEV_DIGITS = set('०१२३४५६७८९')
GUJ_DIGITS = set('૦૧૨૩૪૫૬૭૮૯')
ASCII_DIGITS = set('0123456789')
PUNCT = set('!()-./:;,"\'|—?%&+=')

CKPTS = {
    'marathi': 'data/marathi_stage5.ckpt',
    'gujarati': 'data/gujarati_stage1.ckpt',
}


def script_of(ch):
    o = ord(ch)
    for name, (lo, hi) in RANGES.items():
        if lo <= o <= hi:
            return name
    return None


def pdfs_to_pages(input_dir, pages_dir, dpi):
    n = 0
    for f in sorted(os.listdir(input_dir)):
        if not f.lower().endswith('.pdf'):
            continue
        stem = os.path.splitext(f)[0][:40].replace(' ', '_')
        out = os.path.join(pages_dir, f'page_{stem}')
        subprocess.run(['pdftoppm', '-png', '-r', str(dpi),
                        os.path.join(input_dir, f), out], check=True)
        n += 1
    return n


def lines_from_pages(pages_dir, lines_dir, language, device):
    """Detect words on each page, group into line strips, save line images."""
    from IndicPhotoOCR.ocr import OCR
    ocr = OCR(device=device, identifier_lang='auto', verbose=False)
    line_no = 0
    meta = []
    for f in sorted(os.listdir(pages_dir)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(pages_dir, f)
        image = cv2.imread(path)
        if image is None:
            continue
        detections = ocr.detect(path)
        # group word boxes into lines by vertical overlap
        boxes = []
        for det in detections:
            pts = np.array(det, np.int32)
            boxes.append([int(pts[:, 0].min()), int(pts[:, 1].min()),
                          int(pts[:, 0].max()), int(pts[:, 1].max())])
        boxes.sort(key=lambda b: (b[1], b[0]))
        lines = []
        for b in boxes:
            placed = False
            for ln in lines:
                ov = max(0, min(ln[3], b[3]) - max(ln[1], b[1]))
                h = min(ln[3] - ln[1], b[3] - b[1])
                if h > 0 and ov / h > 0.4:
                    ln[0] = min(ln[0], b[0]); ln[1] = min(ln[1], b[1])
                    ln[2] = max(ln[2], b[2]); ln[3] = max(ln[3], b[3])
                    placed = True
                    break
            if not placed:
                lines.append(list(b))
        for ln in sorted(lines, key=lambda l: (l[1], l[0])):
            x1, y1, x2, y2 = [max(0, v) for v in ln]
            x2 = min(x2, image.shape[1]); y2 = min(y2, image.shape[0])
            if x2 - x1 < 40 or y2 - y1 < 8:
                continue
            crop = image[y1:y2, x1:x2]
            name = f'line_{line_no:04d}.png'
            cv2.imwrite(os.path.join(lines_dir, name), crop)
            meta.append({'image_filename': name, 'page': f,
                         'language': language})
            line_no += 1
    return line_no, meta


def analyze_chars(texts):
    chars = Counter()
    for t in texts:
        chars.update(t)
    scripts = Counter()
    for c, n in chars.items():
        s = script_of(c)
        if s:
            scripts[s] += n
    dev_d = sum(chars[d] for d in DEV_DIGITS)
    guj_d = sum(chars[d] for d in GUJ_DIGITS)
    asc_d = sum(chars[d] for d in ASCII_DIGITS)
    punct = {c: chars[c] for c in PUNCT if chars[c] > 0}
    return chars, scripts, dev_d, guj_d, asc_d, punct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--language', default='auto')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--pseudo-label', action='store_true',
                    help='recognize lines with current best models (pseudo-GT)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pages_dir = os.path.join(args.outdir, 'pages')
    lines_dir = os.path.join(args.outdir, 'lines')
    words_dir = os.path.join(args.outdir, 'words')
    report = []

    def say(s):
        print(s)
        report.append(s)

    # ---- 1. detect input shape ------------------------------------------
    files = sorted(os.listdir(args.input_dir))
    pdfs = [f for f in files if f.lower().endswith('.pdf')]
    imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    jsonls = [f for f in files if f.endswith('.jsonl')]

    say(f'INPUT: {args.input_dir}')
    say(f'  {len(pdfs)} PDFs, {len(imgs)} images, {len(jsonls)} jsonl files')

    labeled_lines = []
    if jsonls:
        # shape 3: labeled line dataset
        jl = os.path.join(args.input_dir, jsonls[0])
        labeled_lines = [json.loads(l) for l in open(jl, encoding='utf-8') if l.strip()]
        say(f'  labeled lines: {len(labeled_lines)} from {jsonls[0]}')
        import shutil
        os.makedirs(lines_dir, exist_ok=True)
        for row in labeled_lines:
            src = os.path.join(args.input_dir, os.path.basename(row['image_filename']))
            if os.path.exists(src):
                shutil.copy(src, os.path.join(lines_dir, os.path.basename(row['image_filename'])))
    elif pdfs or imgs:
        # shapes 1+2: PDFs / page images -> pages -> lines
        os.makedirs(pages_dir, exist_ok=True)
        os.makedirs(lines_dir, exist_ok=True)
        if pdfs:
            say(f'  converting {len(pdfs)} PDFs at {args.dpi} DPI...')
            pdfs_to_pages(args.input_dir, pages_dir, args.dpi)
        else:
            import shutil
            for f in imgs:
                shutil.copy(os.path.join(args.input_dir, f), pages_dir)
        say('  detecting lines...')
        n_lines, meta = lines_from_pages(pages_dir, lines_dir, args.language, args.device)
        say(f'  line images: {n_lines}')
        with open(os.path.join(args.outdir, 'lines_meta.jsonl'), 'w', encoding='utf-8') as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

    # ---- 2. resolution stats ---------------------------------------------
    res = []
    for f in sorted(os.listdir(lines_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            im = cv2.imread(os.path.join(lines_dir, f))
            if im is not None:
                res.append((im.shape[1], im.shape[0]))
    if res:
        ws = [r[0] for r in res]; hs = [r[1] for r in res]
        say(f'\nRESOLUTION: {len(res)} line images')
        say(f'  width  min/median/max: {min(ws)}/{sorted(ws)[len(ws)//2]}/{max(ws)}')
        say(f'  height min/median/max: {min(hs)}/{sorted(hs)[len(hs)//2]}/{max(hs)}')
        low = sum(1 for w, h in res if h < 24)
        say(f'  low-res lines (height < 24px): {low} ({100 * low / len(res):.0f}%)'
            + ('  <-- WARNING: recognition will struggle' if low > len(res) * 0.2 else ''))

    # ---- 3. labeled path: split to words + charset analysis ---------------
    if labeled_lines:
        jl_path = os.path.join(args.outdir, 'lines_labels.jsonl')
        with open(jl_path, 'w', encoding='utf-8') as f:
            for row in labeled_lines:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        say('\nSplitting labeled lines to word crops...')
        from IndicPhotoOCR.recognition.generate_dataset import split_lines_to_dataset
        split_lines_to_dataset(jl_path, lines_dir, words_dir,
                               language=args.language if args.language != 'auto' else 'marathi',
                               augment=0)
        word_rows = [json.loads(l) for l in
                     open(os.path.join(words_dir, 'labels.jsonl'), encoding='utf-8') if l.strip()]
        texts = [r['expected_text'] for r in word_rows]
        say(f'  word crops: {len(word_rows)}')

        chars, scripts, dev_d, guj_d, asc_d, punct = analyze_chars(texts)
        say(f'\nCHARSET: {len(chars)} unique chars')
        say(f'  script mix (chars): ' +
            ' '.join(f'{k}={v}' for k, v in scripts.most_common()))
        say(f'  digit convention: devanagari={dev_d} gujarati={guj_d} ascii={asc_d}')
        conv = 'devanagari' if dev_d > asc_d else ('ascii' if asc_d > dev_d else 'mixed')
        say(f'  -> synthetic data must use {conv} digits')
        say(f'  punctuation: {dict(sorted(punct.items(), key=lambda x: -x[1]))}')
        sp_words = sum(1 for t in texts if any(c in PUNCT for c in t))
        say(f'  words with special chars: {sp_words}/{len(texts)} '
            f'({100 * sp_words / max(len(texts), 1):.0f}%)')

        # vs existing checkpoints
        from IndicPhotoOCR.recognition.charset_extension import inspect_checkpoint
        for lang, ckpt in [('marathi', CKPTS['marathi']), ('gujarati', CKPTS['gujarati'])]:
            if os.path.exists(ckpt):
                cs = set(inspect_checkpoint(ckpt)['charset_train'])
                missing = sorted(c for c in chars if c not in cs and c != ' ')
                say(f'  missing vs {lang} best ckpt ({len(missing)}): {"".join(missing)!r}')

        vocab = Counter(' '.join(t.split()) for t in texts)
        say(f'  unique words: {len(vocab)}, top-10: '
            + ', '.join(f'{w}({n})' for w, n in vocab.most_common(10)))

    # ---- 4. optional pseudo-labeling --------------------------------------
    if args.pseudo_label and os.listdir(lines_dir):
        say('\nPseudo-labeling lines with current best models (NOT for reported numbers)...')
        from IndicPhotoOCR.ocr import OCR
        ocr = OCR(device=args.device, identifier_lang='auto', verbose=False)
        out_rows = []
        for lang, ckpt in CKPTS.items():
            if not os.path.exists(ckpt):
                continue
            for f in sorted(os.listdir(lines_dir)):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                try:
                    t = ocr.recognise(os.path.join(lines_dir, f), None, checkpoint=ckpt)
                except Exception:
                    t = ''
                out_rows.append({'image_filename': f, 'pseudo_text': (t or '').strip(),
                                 'model': lang})
            say(f'  {lang}: labeled {len(out_rows)} lines')
        with open(os.path.join(args.outdir, 'pseudo_labels.jsonl'), 'w', encoding='utf-8') as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        say('  saved pseudo_labels.jsonl (pseudo-GT: use for triage only)')

    out = os.path.join(args.outdir, 'analysis_report.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print(f'\nReport: {out}')


if __name__ == '__main__':
    main()
