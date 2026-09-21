"""Zero-shot eval of a VLM OCR model (Chitrapathak-2 / Qwen2.5-VL arch) on the
fixed Marathi eval sets, using the model card's exact prompt and greedy decoding.

Compares against production PARSeq model D under the same protocol:
word exact-match (stripped), characc, specials subset.
Also reports a digit-normalized variant (Devanagari <-> ASCII digits) as a
diagnostic for script-choice failures.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

SPECIAL = set('!-./:;(),')
DEV = '०१२३४५६७८९'
ASCII = '0123456789'


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_digits(s, to_dev=False):
    if to_dev:
        return s.translate(str.maketrans(ASCII, DEV))
    return s.translate(str.maketrans(DEV, ASCII))


def characc(gt, pred):
    if not gt:
        return 100.0
    return 100.0 * (1 - levenshtein(gt, pred) / len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-jsonl', required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--model-path', default='data/chitrapathak2')
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--out', default=None, help='save per-crop predictions jsonl')
    ap.add_argument('--limit', type=int, default=0, help='eval first N only (smoke test)')
    ap.add_argument('--max-pixels', type=int, default=600000,
                    help='processor image cap in pixels (bounds visual tokens/MPS memory)')
    ap.add_argument('--min-pixels', type=int, default=56 * 56,
                    help='raise to force upscaling of small-text lines')
    ap.add_argument('--prompt', default='Perform OCR on this image and transcribe all visible text exactly as it appears.')
    ap.add_argument('--adapter', default=None, help='optional LoRA adapter dir to load on top of the base model')
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f'loading {args.model_path} ...', flush=True)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.bfloat16 if device in ('cuda', 'mps') else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation='sdpa' if device != 'cpu' else 'eager')
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f'adapter loaded: {args.adapter}', flush=True)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path,
                                            max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    print(f'loaded (device={device}).', flush=True)

    def ocr(img):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': args.prompt},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, use_cache=True)
        gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    rows = [json.loads(l) for l in open(args.test_jsonl, encoding='utf-8')]
    if args.limit:
        rows = rows[:args.limit]
    n = ok = dok = sn = sok = 0
    cacc = dcacc = 0.0
    preds = []
    import time
    t0 = time.time()
    for k, r in enumerate(rows):
        p = os.path.join(args.test_dir, r['image_filename'])
        img = Image.open(p).convert('RGB')
        pred = ocr(img)
        gt = r['expected_text'].strip()
        n += 1
        ok += pred.strip() == gt
        dok += norm_digits(pred.strip()) == norm_digits(gt)
        cacc += characc(gt, pred.strip())
        dcacc += characc(norm_digits(gt), norm_digits(pred.strip()))
        if SPECIAL & set(gt):
            sn += 1
            sok += pred.strip() == gt
        preds.append({'image_filename': r['image_filename'], 'gt': gt, 'pred': pred})
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {k + 1}/{len(rows)}  exact={ok}/{n}  ({el / (k + 1):.1f}s/img)', flush=True)

    print(f'\ntest crops: {n}   ({(time.time() - t0) / n:.1f}s/img)')
    print(f'raw        exact {ok}/{n} ({100 * ok / n:.2f}%)   characc {cacc / n:.2f}')
    print(f'digit-norm exact {dok}/{n} ({100 * dok / n:.2f}%)   characc {dcacc / n:.2f}')
    print(f'specials:  {sok}/{sn}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        print(f'predictions saved: {args.out}')


if __name__ == '__main__':
    main()
