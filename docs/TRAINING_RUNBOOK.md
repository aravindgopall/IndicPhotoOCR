# Training Runbook: Indic OCR fine-tuning from new data

The complete workflow, with exact commands and the lessons that cost us
debugging time. Follow it top to bottom when a new batch of production
documents arrives.

**"From scratch" in this runbook means: fine-tune from the pretrained base
checkpoints** (`IndicPhotoOCR/recognition/models/marathi.ckpt`,
`gujarati.ckpt`, ...). True random-init PARSeq training needs millions of
samples and is not worth it - the pretrained weights already know Indic
scripts; what they lack is your charset and your document genre.

---

## 0. Environment

```
cd /Users/aravind.mallapureddy/acreed/IndicPhotoOCR && source .venv/bin/activate
```
- Training: MPS (fast) — run inside a terminal session with `> log 2>&1`
  (terminal sessions are flaky; the log survives).
- **All evaluation: CPU** (MPS inference is non-deterministic on borderline
  crops; CPU is the stable reference. ±2-3 words of run-to-run noise on MPS.)

## 1. Intake (day 1)

```
python scripts/intake_documents.py --input-dir ~/Downloads/new_batch \
    --outdir data/intake_batch1 --language marathi --device cpu
```
Handles: folder of PDFs (pdftoppm, 300 DPI), page images (line detection),
or labeled line datasets (jsonl: image_filename + expected_text).
Produces: `pages/`, `lines/`, `words/` + `analysis_report.txt` (script mix,
**digit convention**, punctuation frequency, charset gaps vs current best
checkpoints, low-res warning, vocabulary).

Read the report before anything else. The digit convention line decides your
synthetic data design (Devanagari vs ASCII digits - getting this wrong costs
a whole training round; the Gujarati dataset uses ASCII, Marathi gov docs use
Devanagari).

## 2. Build the eval suite FIRST (before training)

Hold out entire documents (not random words) - 10-20% of the batch, or the
cleanest 2-3 docs. Then transcribe a sample of their words (even 100-150
words per language is enough for a usable signal).

```
# eval script template: see scripts/eval_gujarati.py (word crops + labels)
python scripts/eval_gujarati.py --device cpu --test-jsonl data/<lang>_test/labels.jsonl
```
Score stratified, not aggregate: plain / special-char / digit / Latin words
separately. A single number hides category regressions (this cost us stage4).

**Never include eval docs in training data. Never let synthetic vocab come
from eval docs** (we did this for Marathi v2 - it works, but you can no
longer claim generalization; postprocess.py's dictionary has the same
problem and is excluded from production numbers for that reason).

## 3. Prepare real training data

```
# labeled line dataset -> word crops (+2 augmentations)
python -m IndicPhotoOCR.recognition.generate_dataset split \
    --jsonl data/intake_batch1/lines_labels.jsonl \
    --image-dir data/intake_batch1/lines \
    --language <lang> --output-dir data/<lang>_train --augment 2
```
Known issue: **split-alignment noise** (~5% of crops get labels of
neighboring words; hyphens/touching words confuse proportional alignment).
Inspect a sample with the visualization scripts before training on it.

## 4. Extend the charset (if the report shows gaps)

```
python -m IndicPhotoOCR.recognition.charset_extension \
    -c IndicPhotoOCR/recognition/models/<lang>.ckpt \
    --data-jsonl data/<lang>_train/labels.jsonl \
    --max-label-length 50 -o data/<lang>_extended.ckpt
```
Extend from the ORIGINAL base checkpoint with ALL training data chars
(including synthetic Latin tokens) in one pass - extending twice stacks
token-embedding surgery.

## 5. Synthetic data (the leverage)

Generator template: `IndicPhotoOCR/recognition/synthetic_data_guj.py`
(cleanest implementation). Adapt vocab + categories to the intake report.

**Match the CROP SOURCE geometry, not just the document:** detector boxes
are TIGHT (vertical ink fill ~0.85-0.95); projection-split crops are padded
(~0.5). If production recognizes detector boxes, render synthetic at tight
fill (see synthetic_data_v4.py render_matched). A fill mismatch means the
model sees 28px-tall glyphs in production vs 17px in training - a silent
distribution gap that cost us a round to find.

**Hard-won lessons, all mandatory:**
1. **Tofu-verified fonts only** - `_font_supports_text` (exact .notdef
   bitmap comparison). Dark-pixel checks pass tofu. MuktaMahee renders all
   Devanagari as tofu; use Sangam MN / Kohinoor (Devanagari),
   Gujarati Sangam MN / Kohinoor Gujarati (Gujarati).
2. **RAQM must be active** for real conjunct shaping (द्ध, त्र, ક્ર).
   Verify: `python -c "from PIL import features; print(features.check('raqm'))"`
3. **Bare-dominant punctuation contrast** - train bare forms more often than
   punctuated (target ~60-75% bare, matching the real doc distribution).
   Over-punctuated synthetic data causes punctuation hallucination (stage3's
   mistake, visible on the fresh-doc generalization test).
4. **Digit convention must match the real docs exactly** (Devanagari vs ASCII).
5. **Contrast pairs on the SAME vocabulary** (word vs word. vs word:) are what
   teach WHERE punctuation goes - real data alone teaches only THAT it exists
   (Gujarati ablation: real-only 80.0% / 68% specials vs +synthetic 89.1% / 86%).
6. **Split composite patterns** (code vs code+date as separate examples) -
   joint patterns get over-completed beyond the crop (the ",दि." artifact).
7. **No zero-width chars** (ZWJ/ZWNJ) in labels.
8. **Retention subsets** in corrective rounds - replay prior capabilities
   (Latin/URLs/codes) or they're forgotten (stage4's regression).
9. **Real-crop anchoring**: oversample real word crops x2-3 (~30% of the mix)
   to prevent synthetic over-fit.
10. **Never train on (or design synthetic from) eval documents.** Also
    beware label alignment when building eval sets from line images:
    always report on a clean 1:1-aligned subset alongside any
    proportionally-aligned full set (alignment noise cost us ~6pp of
    phantom failures). scripts/split_detector_aligned.py builds
    production-aligned (detector-box) datasets with chunked code labels.

Mix recipe that worked (Gujarati round 1, one shot to 89%):
real crops x2 + synthetic (12k) = ~15k samples.

## 6. Train

```
python -m IndicPhotoOCR.recognition.finetune \
    -c data/<lang>_extended.ckpt \
    --image-dir data/<lang>_stage1_data --labels data/<lang>_stage1_data/labels.jsonl \
    -o data/<lang>_stage1.ckpt \
    --epochs 8 --lr 2e-4 --batch-size 8 --val-split 0.08 \
    --device mps --accumulate-grad-batches 4 --warmup-pct 0.05 \
    > /tmp/<lang>_stage1.log 2>&1
```
- lr 2e-4 for round 1; **1e-4 or lower for corrective rounds** on top of a
  good checkpoint.
- **The val split is nearly useless for model selection** (synthetic val
  saturates to ~100% instantly; small real val memorizes). Select on the
  held-out GT eval. Watch val only for divergence.
- ~35 min per 8 epochs at 15k samples on MPS. Disk: each checkpoint ~275MB;
  keep ~5GB free (a full disk kills training with OSError 28).

## 7. Evaluate (CPU only)

```
python scripts/eval_gujarati.py --device cpu \
    --models original=<base.ckpt> real_only=... real_and_synthetic=<new.ckpt>
python scripts/eval_gujarati.py --device cpu --tta --models ...   # TTA
```
- TTA (5 crop variants, probability averaging - `scripts/eval_tta.py`) helps
  on low-res/degraded inputs (+8.7pp on the 992px Marathi screenshot) and is
  neutral on clean renders. No vocabulary knowledge - safe for production.
- Run the fresh-doc smoke test (no GT needed):
  `python scripts/compare_new_docs.py --device cpu --images /tmp/doc1.png ...`
  Checks special-char emission, format validity, cross-model consensus -
  catches genre overfitting before it ships.
- Failure autopsy every round: classify failures into label-noise /
  crop-boundary / genuine (see `scripts/visualize_gujarati.py` outputs).
  Label-noise failures mean the split, not the model, is wrong.

## 8. Productionize

- Copy the best checkpoint to a stable name; update the default in the
  recognizer call path (`checkpoint=` argument).
- Report production numbers as model(+TTA). Dictionary post-processing only
  if its vocabulary comes from production-independent sources.

---

## Current state (2026-09-02)

| | Marathi | Gujarati |
|---|---|---|
| base | 60.20% exact / 81.04% characc (196-word GT) | 72.73% / 89.74% (110-word GT) |
| best | **stage5+TTA: 79.08% / 93.77%** | **stage1: 89.09% / 95.48%** (TTA neutral) |
| checkpoint | data/marathi_stage5.ckpt | data/gujarati_stage1.ckpt |
| key lesson | genre-specialized: great on gov circulars, slightly worse on out-of-genre text | one round with all lessons baked in beats Marathi's 5 rounds |

Both models emit special chars (base models: 0% - charset lacks them entirely).
Remaining Marathi failures are at the information limit of the 992px source
(sub-pixel punctuation, ambiguous digits in dense codes).


## Lesson: eval label alignment can silently lie (2026-09-04)
Count-matching boxes to tokens fails when GT misses image content or the
detector merges/splits tokens — counts can match by coincidence while every
label is shifted. Fix protocol (scripts/fix_eval_alignment.py): template-match
crops to line images for exact x-ranges, consensus-read with 2 models, DP-assign
contiguous GT substrings (labels stay GT-only), drop ungradeable crops.
Circularity check: independent models must gain as much as in-alignment models.
On our clean eval this removed ~50 phantom failures (+6.4pp for the original
model, which never touched the alignment).
