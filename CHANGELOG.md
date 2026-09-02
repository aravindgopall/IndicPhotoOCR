# Changelog

## [Unreleased]
### Added
- **Local checkpoint support for inference**: ``OCR(recognition_checkpoint=...)``
  and ``PARseqrecogniser.recognise(checkpoint=...)`` now accept a path to a
  local ``.ckpt`` file (e.g. a fine-tuned model), loading it directly instead of
  downloading the default.  Supports a string (one checkpoint for all languages)
  or a ``{language: path}`` dict for per-language overrides.
- **Dataset generation** (``IndicPhotoOCR.recognition.generate_dataset``): turn
  BSTD scene images or line-level Marathi OCR data into word-level training
  crops + JSONL.  Three modes: ``convert`` (BSTD recognition JSON → JSONL),
  ``crop`` (BSTD scenes → word crops via polygon annotations), and ``split``
  (line images → word crops via vertical-projection segmentation with
  proportional alignment for mismatched lines).  Includes augmentation
  (rotation, brightness, blur, noise) and ``--append`` for combining multiple
  data sources.
- **Charset extension** for PARSeq recognition checkpoints
  (`IndicPhotoOCR.recognition.charset_extension`): grows the model vocabulary so
  it can emit characters (e.g. punctuation `()/.,*-`, digits, rare matras) that
  were absent from training. Learned weights for existing characters are
  preserved exactly; only fresh token rows are added. CLI: `extend_charset.py`.
- **`max_label_length` extension**: grows `pos_queries` so the model can decode
  longer sequences (needed for long Indic government text).
- **Fine-tuning** on JSONL image datasets (`IndicPhotoOCR.recognition.finetune`):
  a `JsonlDataset` that preserves Indic unicode (no NFKD/ASCII stripping), a
  PyTorch-Lightning training loop, optional automatic charset +
  `max_label_length` extension before training, and best-checkpoint saving.
  CLI: `finetune_recognition.py`.
- Tests for charset computation, checkpoint extension, weight preservation, and
  `max_label_length` extension.

## [1.3.1] - 2025-17-25
### Added
- Seperate function for script identification
- Added device opts for recognition

## [1.3.0] - 2025-17-25
### Added
- ViT models for script detection
- Word order based on horizontal and vertical lines

## [1.2.0] - 2024-11-24
### Added
- Textbpn++ detection model added

## [1.1.0] - 2024-11-24
### Added
- Updated package naming convention

## [1.0.3] - 2024-11-06
### Added
- Python package requirements sorted with setup.py

## [1.0.2] - 2024-11-01
### Added
- Added language support for 10 additional models in the recognition module.

## [1.0.1] - 2024-10-28
### Added
- Added support for detecting polygonal bounding boxes in `visualize_detection`.
- Introduced the `show` argument in `visualize_detection` to control image display.

