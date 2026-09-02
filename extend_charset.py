#!/usr/bin/env python3
"""Thin CLI wrapper around IndicPhotoOCR.recognition.charset_extension.

Examples
--------
Inspect a checkpoint's charset::

    python extend_charset.py --checkpoint marathi.ckpt --inspect

Add punctuation/digits to a checkpoint::

    python extend_charset.py -c marathi.ckpt --extra-chars "()/.,*-:;\"" -o marathi_ext.ckpt

Auto-discover chars from a JSONL dataset::

    python extend_charset.py -c marathi.ckpt --data-jsonl data/validation.jsonl -o marathi_ext.ckpt
"""
import sys

from IndicPhotoOCR.recognition.charset_extension import main

if __name__ == "__main__":
    sys.exit(main())
