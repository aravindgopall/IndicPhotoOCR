"""LoRA SFT of Chitrapathak-2 (Qwen2.5-VL arch) on our Marathi OCR data.

Design decisions (matching the zero-shot eval exactly):
  - SAME prompt as eval: "Perform OCR on this image and transcribe all visible
    text exactly as it appears." with the model card's system message.
  - SAME image preprocessing: processor max_pixels=300k, min_pixels=56*56.
  - Loss ONLY on the assistant response tokens (prompt masked with -100).
  - LoRA on language-model attention+MLP projections only (vision tower
    untouched); module list discovered at runtime -> version-proof.
  - bf16, gradient checkpointing, cosine LR, warmup.

Data: data/cp2_sft/train.jsonl — {"image": <repo-relative path>, "text": GT}.
Held-out evals are NEVER in this file.

Run (GPU box):
  python scripts/sft_chitrapathak.py --model data/chitrapathak2 \
      --data data/cp2_sft/train.jsonl --out out/cp2_sft_lora \
      --epochs 2 --batch-size 4 --grad-accum 4 --device cuda
Merge after training:
  python scripts/sft_chitrapathak.py --merge-only --model data/chitrapathak2 \
      --adapter out/cp2_sft_lora --out out/cp2_sft_merged
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SYSTEM = 'You are a helpful assistant.'
PROMPT = 'Perform OCR on this image and transcribe all visible text exactly as it appears.'
PROJ_SUFFIX = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def find_lora_targets(model):
    """All Linear proj modules in the LANGUAGE model (skip vision tower)."""
    names = []
    for name, mod in model.named_modules():
        if 'visual' in name or 'vision' in name:
            continue
        if name.endswith(PROJ_SUFFIX):
            names.append(name)
    return names


class OcrDataset(Dataset):
    def __init__(self, rows, root):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.root, r['image'])).convert('RGB')
        return {'image': img, 'text': r['text']}


def identity_collate(batch):
    return batch


def build_batch(batch, processor, device):
    """Tokenize prompt-only and full (prompt+assistant) for every sample.

    processor.apply_chat_template(tokenize=False) yields the text; the
    processor then expands the image placeholder consistently, so
    prompt_ids is a prefix of full_ids (asserted on the first batch).
    """
    prompts, fulls, images = [], [], []
    for ex in batch:
        msgs = [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': ex['image']},
                {'type': 'text', 'text': PROMPT},
            ]},
        ]
        prompts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        full_msgs = msgs + [{'role': 'assistant', 'content': [{'type': 'text', 'text': ex['text']}]}]
        fulls.append(processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False))
        images.append(ex['image'])
    p = processor(text=prompts, images=images, padding=True, return_tensors='pt')
    f = processor(text=fulls, images=images, padding=True, return_tensors='pt')
    input_ids = f.input_ids
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    assert processor.tokenizer.padding_side == 'right', \
        f"expected right padding, got {processor.tokenizer.padding_side}"
    # mask everything up to each sample's true prompt length (right padding)
    for i in range(len(batch)):
        plen = int((p.input_ids[i] != processor.tokenizer.pad_token_id).sum())
        labels[i, :plen] = -100
    pv = f.pixel_values.to(device=device, dtype=torch.bfloat16) if hasattr(f, 'pixel_values') else None
    return {'input_ids': input_ids.to(device),
            'attention_mask': f.attention_mask.to(device),
            'pixel_values': pv,
            'image_grid_thw': f.image_grid_thw.to(device) if hasattr(f, 'image_grid_thw') else None,
            'labels': labels.to(device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='data/chitrapathak2')
    ap.add_argument('--data', default='data/cp2_sft/train.jsonl')
    ap.add_argument('--root', default='.', help='root dir for relative image paths')
    ap.add_argument('--out', default='out/cp2_sft_lora')
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--lora-alpha', type=int, default=64)
    ap.add_argument('--warmup-pct', type=float, default=0.03)
    ap.add_argument('--max-pixels', type=int, default=300000)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke test: stop after N optimizer steps')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--merge-only', action='store_true',
                    help='merge --adapter into --model and save to --out, no training')
    ap.add_argument('--adapter', default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText

    if args.merge_only:
        from peft import PeftModel
        assert args.adapter, '--adapter required with --merge-only'
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels,
                                      min_pixels=56 * 56).save_pretrained(args.out)
        print(f'merged model saved: {args.out}')
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'loading {args.model} ...', flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels,
                                              min_pixels=56 * 56)
    processor.tokenizer.padding_side = 'right'
    device = args.device

    from peft import LoraConfig, get_peft_model
    targets = find_lora_targets(model)
    print(f'LoRA targets: {len(targets)} modules '
          f'(e.g. {targets[0] if targets else "NONE!"})', flush=True)
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias='none', task_type='CAUSAL_LM', target_modules=targets)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.to(device)

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    except Exception as e:
        print(f'(gradient checkpointing enable failed: {e}; continuing)', flush=True)
    model.enable_input_require_grads()

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8')]
    print(f'train rows: {len(rows)}', flush=True)
    ds = OcrDataset(rows, args.root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=identity_collate, drop_last=True)

    steps_per_epoch = len(dl) // args.grad_accum
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(8, int(total_steps * args.warmup_pct))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    os.makedirs(args.out, exist_ok=True)
    step = 0
    t0 = time.time()
    losses = []
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for bi, batch in enumerate(dl):
            model.train()
            inputs = build_batch(batch, processor, device)
            out = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        image_grid_thw=inputs['image_grid_thw'],
                        labels=inputs['labels'])
            loss = out.loss / args.grad_accum
            loss.backward()
            losses.append(out.loss.item())
            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g['lr'] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    el = time.time() - t0
                    print(f'epoch {epoch} step {step}/{total_steps} '
                          f'loss {sum(losses[-40:]) / len(losses[-40:]):.4f} '
                          f'({el / step:.1f}s/step, eta {(total_steps - step) * el / step / 3600:.2f}h)',
                          flush=True)
                if step >= total_steps or (args.max_steps and step >= args.max_steps):
                    done = True
                    break

    model.save_pretrained(args.out)
    print(f'\nLoRA adapter saved: {args.out}')
    print(f'total optimizer steps: {step}, final loss {sum(losses[-40:]) / len(losses[-40:]):.4f}')


if __name__ == '__main__':
    main()
