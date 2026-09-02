#!/usr/bin/env python3
"""Thin CLI wrapper around IndicPhotoOCR.recognition.finetune.

Example
-------
# 1. (optional) extend charset to cover punctuation/digits in the data
python extend_charset.py -c marathi.ckpt --data-jsonl "Marathi OCR/validation.jsonl" -o marathi_ext.ckpt

# 2. fine-tune on the Marathi data (extends charset automatically here too)
python finetune_recognition.py \
    -c marathi_ext.ckpt \
    --image-dir "Marathi OCR/images" \
    --labels "Marathi OCR/validation.jsonl" \
    -o marathi_finetuned.ckpt \
    --epochs 20 --lr 7e-4 --batch-size 8 --extend-charset
"""
import sys

from IndicPhotoOCR.recognition.finetune import main

if __name__ == "__main__":
    sys.exit(main())
