"""Compare original vs fine-tuned Marathi recognition on a screenshot."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IndicPhotoOCR.ocr import OCR


def run_ocr(image_path: str, checkpoint=None):
    ocr = OCR(device='mps', identifier_lang='auto', verbose=False,
              recognition_checkpoint=checkpoint)
    t0 = time.time()
    results = ocr.ocr(image_path, batch_size=1)
    elapsed = time.time() - t0
    return results, elapsed


def flatten_lines(results):
    """Flatten detect_para output to a list of {'txt', 'confidence'} dicts."""
    words = []
    if results is None:
        return words
    for line in results:
        if isinstance(line, dict):
            words.append(line)
        elif isinstance(line, (list, tuple)):
            for w in line:
                if isinstance(w, dict):
                    words.append(w)
                elif isinstance(w, str):
                    words.append({'txt': w, 'confidence': None})
                elif isinstance(w, (list, tuple)) and len(w) >= 1:
                    words.append({'txt': w[0], 'confidence': w[2] if len(w) > 2 else None})
        elif isinstance(line, str):
            words.append({'txt': line, 'confidence': None})
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/tmp/test_marathi2.png')
    ap.add_argument('--original', default='IndicPhotoOCR/recognition/models/marathi.ckpt')
    ap.add_argument('--stage1', default='data/marathi_stage1.ckpt')
    ap.add_argument('--finetuned', default='data/marathi_finetuned.ckpt')
    ap.add_argument('--output', default=os.path.expanduser(
        '~/Desktop/marathi_ocr_comparison_2026-09-01.txt'))
    args = ap.parse_args()

    models = [
        ('ORIGINAL (downloaded)', args.original),
        ('STAGE 1 (real data only)', args.stage1),
        ('STAGE 2 (real + synthetic)', args.finetuned),
    ]

    all_results = {}
    for name, ckpt in models:
        if not os.path.exists(ckpt):
            print(f'  skipping {name}: {ckpt} not found')
            continue
        print(f'Running OCR with {name}...')
        results, elapsed = run_ocr(args.image, ckpt)
        all_results[name] = (results, elapsed)
        print(f'  done in {elapsed:.1f}s')

    with open(args.output, 'w', encoding='utf-8') as f:
        from PIL import Image
        img = Image.open(args.image)
        f.write('Marathi OCR Comparison\n')
        f.write(f'Image: {args.image} ({img.size[0]}x{img.size[1]})\n')
        f.write(f'Date: {time.strftime("%Y-%m-%d %H:%M")}\n')
        f.write('=' * 100 + '\n\n')

        for name, (results, elapsed) in all_results.items():
            words = flatten_lines(results)
            f.write(f'--- {name} --- ({len(words)} words, {elapsed:.1f}s)\n')
            for w in words:
                txt = str(w.get('txt', '')).strip()
                conf = w.get('confidence', 0)
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.0
                f.write(f'  [{conf:.2f}] {txt}\n')
            f.write('\n')

        f.write('=' * 100 + '\n')
        f.write('SIDE-BY-SIDE (word# | original | stage1 | stage2)\n')
        f.write('=' * 100 + '\n')
        flattened = {nm: flatten_lines(all_results[nm][0]) for nm in all_results}
        names = list(all_results.keys())
        n = max((len(flattened[nm]) for nm in names), default=0)
        for i in range(n):
            row = []
            for nm in names:
                ws = flattened[nm]
                if i < len(ws):
                    row.append(str(ws[i].get('txt', '')).strip())
                else:
                    row.append('(none)')
            f.write(f'{i:3d} | ' + ' | '.join(row) + '\n')

    print(f'\nComparison written to: {args.output}')


if __name__ == '__main__':
    main()
