"""Generate recognition training datasets from BSTD scene images and annotations.

The Bharat Scene Text Dataset (BSTD) ships full scene images plus polygon
annotations. This module turns those into the word-level crop + JSONL format
that :mod:`IndicPhotoOCR.recognition.finetune` consumes.

Three modes
-----------
1. **convert** -- convert an existing BSTD recognition JSON (``{image: {path,
   language, text}}``) into a JSONL file. Uses the already-cropped word images
   in ``recognition/train/<lang>/``.  Quickest way to start fine-tuning.

2. **crop** -- crop word images directly from BSTD *scene* images using the
   polygon annotations in ``BSTD_v17.57.json`` / ``BSTD_release_v1.json``.
   Generates fresh crops (with configurable padding) for every polygon of the
   requested language, including ones that were filtered out of the pre-packaged
   recognition set.

3. **split** -- split *line-level* images (like the Marathi OCR data where each
   image is a single line of text) into word crops using vertical-projection
   segmentation.  Each crop is matched to its expected text token.  This is the
   primary tool for turning sentence-level data into word-level training crops.

Augmentation (``--augment N``) generates *N* extra variants per crop with random
padding, rotation, brightness, and noise -- a cheap way to multiply the dataset
and make the model robust to different detection boundaries.

Output
------
A directory of crop images + a ``labels.jsonl`` file::

    {"image_filename": "marathi/G_image_10178_5.jpg", "expected_text": "विमानतळ", "language": "marathi"}
    ...

Usage
-----
::

    # 1. Quick start: convert existing BSTD recognition crops to JSONL
    python generate_recognition_dataset.py convert \\
        --bstd-recognition-json ~/Downloads/recognition/train_recognition_data.json \\
        --bstd-recognition-root ~/Downloads/recognition/train \\
        --language marathi --output-dir data/marathi_bstd

    # 2. Crop fresh word images from BSTD scenes (all 5k+ marathi polygons)
    python generate_recognition_dataset.py crop \\
        --bstd-json ~/Downloads/detection/BSTD_v17.57.json \\
        --scene-root ~/Downloads/detection \\
        --language marathi --output-dir data/marathi_crops

    # 3. Same, but also generate 2-word crops + 2x augmentation
    python generate_recognition_dataset.py crop \\
        --bstd-json ~/Downloads/detection/BSTD_v17.57.json \\
        --scene-root ~/Downloads/detection \\
        --language marathi --output-dir data/marathi_aug \\
        --two-word --augment 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


__all__ = [
    "polygon_bbox",
    "crop_polygon",
    "find_adjacent_pairs",
    "merge_bboxes",
    "augment_crop",
    "convert_recognition_json",
    "crop_from_scenes",
]


# ---------------------------------------------------------------------------
# Geometry helpers (no PIL/cv2 -- easily unit-testable)
# ---------------------------------------------------------------------------

def polygon_bbox(coords: List[List[int]]) -> Tuple[int, int, int, int]:
    """Return ``(x_min, y_min, x_max, y_max)`` for a polygon's coordinate list."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def merge_bboxes(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Return the union of two axis-aligned bounding boxes."""
    return min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def find_adjacent_pairs(
    polygons: List[dict],
    max_y_gap: float = 15.0,
    max_x_gap_ratio: float = 2.0,
) -> List[Tuple[int, int]]:
    """Find indices of horizontally-adjacent word polygons on the same line.

    Two polygons are "adjacent" if:
    * Their vertical centers are within ``max_y_gap`` pixels (same line).
    * The horizontal gap between them is positive but no more than
      ``max_x_gap_ratio * min_word_width`` (not too far apart).

    Returns a list of ``(i, j)`` index pairs with ``i`` left of ``j``.
    """
    n = len(polygons)
    if n < 2:
        return []
    bboxes = [polygon_bbox(p["coordinates"]) for p in polygons]
    centers = [_bbox_center(b) for b in bboxes]
    widths = [b[2] - b[0] for b in bboxes]

    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            # Same line?
            if abs(centers[i][1] - centers[j][1]) > max_y_gap:
                continue
            # Determine left/right
            if centers[i][0] <= centers[j][0]:
                left, right = i, j
            else:
                left, right = j, i
            gap = bboxes[right][0] - bboxes[left][2]
            if gap < 0:
                continue  # overlapping
            min_w = max(1, min(widths[left], widths[right]))
            if gap > max_x_gap_ratio * min_w:
                continue
            pairs.append((left, right))
    return pairs


# ---------------------------------------------------------------------------
# Cropping (needs PIL + numpy)
# ---------------------------------------------------------------------------

def crop_polygon(
    image: np.ndarray,
    coords: List[List[int]],
    padding: int = 4,
) -> Optional[np.ndarray]:
    """Crop a polygon region from an image, applying symmetric padding.

    Uses a binary mask so only the text pixels are kept (background outside the
    polygon becomes black), then extracts the bounding rect with padding.
    Returns None for empty/degenerate crops.
    """
    h_img, w_img = image.shape[:2]
    pts = np.array(coords, dtype=np.int32)
    # Clamp to image bounds
    pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    import cv2
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)

    x_min, y_min, x_max, y_max = polygon_bbox(coords)
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w_img, x_max + padding)
    y_max = min(h_img, y_max + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    crop = masked[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return crop


def augment_crop(
    img: Image.Image,
    max_rotation: float = 3.0,
    max_brightness: float = 0.2,
    max_noise_std: float = 10.0,
    blur_prob: float = 0.3,
    seed: Optional[int] = None,
) -> Image.Image:
    """Apply random augmentations to a word-crop PIL image.

    Augmentations are kept mild to preserve readability (this is fine-tuning,
    not pretraining).  Each transform is applied independently with 50% prob.
    """
    rng = random.Random(seed) if seed is not None else random
    out = img

    # Rotation
    if rng.random() < 0.5:
        angle = rng.uniform(-max_rotation, max_rotation)
        out = out.rotate(angle, expand=True, fillcolor=(0, 0, 0))

    # Brightness
    if rng.random() < 0.5:
        factor = 1.0 + rng.uniform(-max_brightness, max_brightness)
        out = Image.eval(out, lambda v: min(255, max(0, int(v * factor))))

    # Slight blur (simulate motion / low-res)
    if rng.random() < blur_prob:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # Gaussian noise
    if rng.random() < 0.5 and max_noise_std > 0:
        arr = np.array(out, dtype=np.float32)
        noise = np.random.normal(0, max_noise_std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    return out


# ---------------------------------------------------------------------------
# Mode 1: convert existing BSTD recognition JSON to JSONL
# ---------------------------------------------------------------------------

def convert_recognition_json(
    json_path: str,
    recognition_root: str,
    language: str,
    output_dir: str,
    min_text_len: int = 1,
) -> Dict:
    """Convert a BSTD recognition JSON to crop-dir + JSONL for fine-tuning.

    The BSTD format is ``{filename: {path, language, text}}`` where ``path`` is
    like ``Recognition/train/marathi/G_image_10178_5.jpg``.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    lang_subdir = os.path.join(output_dir, language)
    os.makedirs(lang_subdir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "labels.jsonl")
    count = 0
    skipped = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for filename, info in data.items():
            if info.get("language") != language:
                continue
            text = info.get("text", "").strip()
            if len(text) < min_text_len:
                skipped += 1
                continue
            # Resolve source path: try the stored path, then just the filename
            rel_path = info.get("path", "")
            src_candidates = [
                os.path.join(recognition_root, language, filename),
                os.path.join(recognition_root, rel_path),
                os.path.join(recognition_root, filename),
            ]
            src = next((c for c in src_candidates if os.path.exists(c)), None)
            if src is None:
                skipped += 1
                continue
            # Copy / hardlink the crop into our output dir
            dst = os.path.join(lang_subdir, filename)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)  # fast, same-filesystem hardlink
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
            rel_dst = os.path.relpath(dst, output_dir)
            f.write(json.dumps({
                "image_filename": rel_dst,
                "expected_text": text,
                "language": language,
            }, ensure_ascii=False) + "\n")
            count += 1

    return {"output_dir": output_dir, "jsonl_path": jsonl_path, "count": count, "skipped": skipped}


# ---------------------------------------------------------------------------
# Mode 2: crop word images from BSTD scene images + annotations
# ---------------------------------------------------------------------------

def _resolve_scene_path(scene_root: str, folder: str, image_name: str) -> str:
    """Resolve the full path to a BSTD scene image."""
    # image_name is like "G/image_10178.jpg"
    candidates = [
        os.path.join(scene_root, image_name),
        os.path.join(scene_root, folder, os.path.basename(image_name)),
        os.path.join(scene_root, os.path.basename(image_name)),
    ]
    return next((c for c in candidates if os.path.exists(c)), candidates[0])


def _normalize_lang(lang: str) -> str:
    """Normalize messy BSTD language labels."""
    fixes = {
        "telegu": "telugu", "gujrati": "gujarati", "gujarti": "gujarati",
        "gujrati": "gujarati", "udru": "urdu", "teluguN": "telugu",
    }
    lang = (lang or "").strip().lower()
    return fixes.get(lang, lang)


def crop_from_scenes(
    bstd_json_path: str,
    scene_root: str,
    language: str,
    output_dir: str,
    padding: int = 4,
    two_word: bool = False,
    augment: int = 0,
    max_per_image: int = 0,
    seed: int = 42,
) -> Dict:
    """Crop word images from BSTD scenes and write crops + JSONL.

    Args:
        bstd_json_path: Path to ``BSTD_v17.57.json`` or ``BSTD_release_v1.json``.
        scene_root: Directory containing the A-L scene image folders.
        language: Language to crop (e.g. ``"marathi"``).
        output_dir: Where to write crops + ``labels.jsonl``.
        padding: Pixels of padding around each polygon.
        two_word: Also generate 2-word crops from adjacent polygons.
        augment: Number of augmented variants per crop (0 = none).
        max_per_image: If > 0, cap the number of crops per scene image.
        seed: RNG seed for augmentation.

    Returns:
        Summary dict with counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(bstd_json_path, "r", encoding="utf-8") as f:
        bstd = json.load(f)

    import cv2  # local import; crop_polygon needs it

    lang_subdir = os.path.join(output_dir, language)
    twoword_subdir = os.path.join(output_dir, f"{language}_2word")
    os.makedirs(lang_subdir, exist_ok=True)
    if two_word:
        os.makedirs(twoword_subdir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "labels.jsonl")
    count = 0
    twoword_count = 0
    aug_count = 0
    skipped = 0
    images_processed = 0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for scene_key, scene_data in bstd.items():
            folder = scene_data.get("folderName", "")
            image_name = scene_data.get("image_name", f"{scene_key}.jpg")
            scene_path = _resolve_scene_path(scene_root, folder, image_name)
            if not os.path.exists(scene_path):
                continue

            image = cv2.imread(scene_path)
            if image is None:
                continue
            images_processed += 1

            anns = scene_data.get("annotations", {})
            # Collect polygons of the target language
            lang_polys: List[dict] = []
            for pname, p in anns.items():
                if _normalize_lang(p.get("script_language")) != language:
                    continue
                coords = p.get("coordinates")
                text = (p.get("text") or "").strip()
                if not coords or not text:
                    continue
                lang_polys.append({"name": pname, "coordinates": coords, "text": text})

            if max_per_image > 0 and len(lang_polys) > max_per_image:
                lang_polys = lang_polys[:max_per_image]

            # --- single-word crops ---
            for idx, poly in enumerate(lang_polys):
                crop = crop_polygon(image, poly["coordinates"], padding=padding)
                if crop is None:
                    skipped += 1
                    continue
                crop_name = f"{scene_key}_{idx}.jpg"
                crop_path = os.path.join(lang_subdir, crop_name)
                cv2.imwrite(crop_path, crop)
                rel = os.path.relpath(crop_path, output_dir)
                jsonl_f.write(json.dumps({
                    "image_filename": rel, "expected_text": poly["text"], "language": language,
                }, ensure_ascii=False) + "\n")
                count += 1

                # --- augmented variants ---
                for a in range(augment):
                    pil = Image.open(crop_path).convert("RGB")
                    aug = augment_crop(pil, seed=seed + count * 100 + a)
                    aug_name = f"{scene_key}_{idx}_aug{a}.jpg"
                    aug_path = os.path.join(lang_subdir, aug_name)
                    aug.save(aug_path)
                    rel = os.path.relpath(aug_path, output_dir)
                    jsonl_f.write(json.dumps({
                        "image_filename": rel, "expected_text": poly["text"], "language": language,
                    }, ensure_ascii=False) + "\n")
                    aug_count += 1

            # --- two-word crops ---
            if two_word and len(lang_polys) >= 2:
                pairs = find_adjacent_pairs(lang_polys)
                for pi, pj in pairs:
                    b1 = polygon_bbox(lang_polys[pi]["coordinates"])
                    b2 = polygon_bbox(lang_polys[pj]["coordinates"])
                    merged = merge_bboxes(b1, b2)
                    # Crop the merged region directly from the original image (no mask)
                    x1 = max(0, merged[0] - padding)
                    y1 = max(0, merged[1] - padding)
                    x2 = min(image.shape[1], merged[2] + padding)
                    y2 = min(image.shape[0], merged[3] + padding)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    tw_crop = image[y1:y2, x1:x2]
                    if tw_crop.size == 0:
                        continue
                    text = lang_polys[pi]["text"] + " " + lang_polys[pj]["text"]
                    tw_name = f"{scene_key}_tw_{pi}_{pj}.jpg"
                    tw_path = os.path.join(twoword_subdir, tw_name)
                    cv2.imwrite(tw_path, tw_crop)
                    rel = os.path.relpath(tw_path, output_dir)
                    jsonl_f.write(json.dumps({
                        "image_filename": rel, "expected_text": text, "language": language,
                    }, ensure_ascii=False) + "\n")
                    twoword_count += 1

    return {
        "output_dir": output_dir,
        "jsonl_path": jsonl_path,
        "single_word_crops": count,
        "two_word_crops": twoword_count,
        "augmented_crops": aug_count,
        "skipped": skipped,
        "images_processed": images_processed,
        "total_samples": count + twoword_count + aug_count,
    }


# ---------------------------------------------------------------------------
# Mode 3: split line-level images into word crops (vertical projection)
# ---------------------------------------------------------------------------

def _vertical_projection(image: np.ndarray, threshold: int = 200) -> np.ndarray:
    """Return a 1-D array of dark-pixel counts per column (vertical projection).

    Pixels darker than ``threshold`` (0=black, 255=white) are counted as "ink".
    """
    gray = image.mean(axis=2) if image.ndim == 3 else image
    ink = (gray < threshold).astype(np.int32)
    return ink.sum(axis=0)


def _find_word_gaps(projection: np.ndarray, min_gap_width: int = 3) -> List[Tuple[int, int]]:
    """Find ``(start, end)`` column ranges where the projection is zero (gaps).

    Only gaps at least ``min_gap_width`` columns wide are returned (thin gaps
    inside a character conjunct are ignored).
    """
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for i, val in enumerate(projection):
        if val == 0 and not in_gap:
            in_gap = True
            gap_start = i
        elif val > 0 and in_gap:
            in_gap = False
            if i - gap_start >= min_gap_width:
                gaps.append((gap_start, i))
    if in_gap and len(projection) - gap_start >= min_gap_width:
        gaps.append((gap_start, len(projection)))
    return gaps


def _proportional_align(
    crops: List[np.ndarray], tokens: List[str]
) -> List[Tuple[np.ndarray, str]]:
    """Align word crops to text tokens using proportional width matching.

    When the crop count equals the token count, it's a direct 1:1 match.
    When they differ (common with hyphens splitting one token, or adjacent
    words touching), crops and tokens are merged proportionally:

    * ``crops > tokens``: merge adjacent crops whose centers fall in the same
      token's proportional range (e.g. ``चार-अ`` split at the hyphen).
    * ``crops < tokens``: merge adjacent tokens whose centers fall in the same
      crop's proportional range.

    This is a heuristic -- Devanagari glyphs vary in width -- but recovers most
    mismatched lines that would otherwise be discarded.
    """
    n_crops = len(crops)
    n_tokens = len(tokens)
    if n_crops == 0 or n_tokens == 0:
        return []
    if n_crops == n_tokens:
        return list(zip(crops, tokens))

    token_lens = [max(1, len(t)) for t in tokens]
    total_len = sum(token_lens)
    crop_widths = [max(1, c.shape[1]) for c in crops]
    total_width = sum(crop_widths)

    if n_crops > n_tokens:
        # Over-split: merge adjacent crops to match token count.
        result: List[Tuple[np.ndarray, str]] = []
        crop_idx = 0
        for t_idx, token in enumerate(tokens):
            expected_end = sum(token_lens[: t_idx + 1]) / total_len
            merged: List[np.ndarray] = []
            while crop_idx < n_crops:
                center = (sum(crop_widths[: crop_idx]) + crop_widths[crop_idx] / 2) / total_width
                if center < expected_end or t_idx == n_tokens - 1:
                    merged.append(crops[crop_idx])
                    crop_idx += 1
                else:
                    break
            if merged:
                crop_out = merged[0] if len(merged) == 1 else np.hstack(merged)
                result.append((crop_out, token))
        return result
    else:
        # Under-split: merge adjacent tokens to match crop count.
        result = []
        token_idx = 0
        for c_idx, crop in enumerate(crops):
            expected_end = sum(crop_widths[: c_idx + 1]) / total_width
            merged_tokens: List[str] = []
            while token_idx < n_tokens:
                center = (sum(token_lens[: token_idx]) + token_lens[token_idx] / 2) / total_len
                if center < expected_end or c_idx == n_crops - 1:
                    merged_tokens.append(tokens[token_idx])
                    token_idx += 1
                else:
                    break
            if merged_tokens:
                result.append((crop, " ".join(merged_tokens)))
        return result


def split_line_to_words(
    image: np.ndarray,
    threshold: int = 200,
    min_gap_width: int = 3,
    min_word_width: int = 5,
) -> List[np.ndarray]:
    """Segment a single-line text image into word crops.

    Uses the classic vertical-projection method: columns with no dark pixels
    are word boundaries.  This works well for clean line images (like the
    Marathi OCR data where each image is already a single line of text).

    Returns a list of word crop arrays (left to right).
    """
    projection = _vertical_projection(image, threshold)
    gaps = _find_word_gaps(projection, min_gap_width)

    words: List[np.ndarray] = []
    prev_end = 0
    h = image.shape[0]
    for gap_start, gap_end in gaps:
        if gap_start - prev_end >= min_word_width:
            words.append(image[:, prev_end:gap_start])
        prev_end = gap_end
    # Last word (after the final gap)
    if image.shape[1] - prev_end >= min_word_width:
        words.append(image[:, prev_end:])
    return words


def split_lines_to_dataset(
    jsonl_path: str,
    image_dir: str,
    output_dir: str,
    language: str = "marathi",
    augment: int = 0,
    max_label_length: int = 25,
    threshold: int = 200,
    min_gap_width: int = 3,
    append: bool = False,
    prefix: str = "",
    seed: int = 42,
    max_split_ratio: float = 3.0,
) -> Dict:
    """Split line-level images into word crops and build a JSONL dataset.

    For each line image, the expected_text is split by whitespace into tokens
    and the image is segmented into word crops via vertical projection.  Crops
    are matched to tokens left-to-right.  When the crop count matches the token
    count, every crop gets its exact label; when they mismatch (hyphens,
    touching words), proportional alignment merges crops/tokens to recover as
    many labeled samples as possible.

    Args:
        jsonl_path: JSONL with ``image_filename`` + ``expected_text`` rows.
        image_dir: Directory containing the line images.
        output_dir: Where to write word crops + ``labels.jsonl``.
        language: Language label for the output.
        augment: Number of augmented variants per word crop.
        max_label_length: Drop tokens longer than this.
        append: If True, append to an existing dataset (JSONL opened in append
            mode, crops prefixed with ``prefix`` to avoid name collisions).
        prefix: Filename prefix for crops when appending (e.g. ``"v2_"``).

    Returns:
        Summary dict with counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    import cv2

    os.makedirs(output_dir, exist_ok=True)
    lang_subdir = os.path.join(output_dir, language)
    os.makedirs(lang_subdir, exist_ok=True)
    out_jsonl = os.path.join(output_dir, "labels.jsonl")

    total_crops = 0
    matched = 0
    aligned = 0
    skipped_mismatch = 0
    skipped_long = 0
    lines_processed = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    with open(out_jsonl, "a" if append else "w", encoding="utf-8") as out_f:
        for line_idx, row in enumerate(lines):
            img_rel = row.get("image_filename", "")
            expected = row.get("expected_text", "")
            # Resolve the image path
            name = img_rel.lstrip("/")
            img_path = os.path.join(image_dir, os.path.basename(name))
            if not os.path.exists(img_path):
                img_path = name if os.path.isabs(name) else os.path.join(image_dir, name)
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue
            lines_processed += 1

            tokens = expected.split()
            crops = split_line_to_words(image, threshold=threshold, min_gap_width=min_gap_width)

            if not crops or not tokens:
                skipped_mismatch += 1
                continue

            # Perfect match or proportional alignment
            if len(crops) == len(tokens):
                pairs = list(zip(crops, tokens))
                matched += 1
            else:
                # Skip if the ratio is too extreme (likely a multi-line image or
                # bad expected_text).
                ratio = max(len(crops), len(tokens)) / max(1, min(len(crops), len(tokens)))
                if ratio > max_split_ratio:
                    skipped_mismatch += 1
                    continue
                pairs = _proportional_align(crops, tokens)
                aligned += 1

            for w_idx, (crop, token) in enumerate(pairs):
                if len(token) > max_label_length:
                    skipped_long += 1
                    continue
                crop_name = f"{prefix}{line_idx:04d}_{w_idx:02d}.jpg"
                crop_path = os.path.join(lang_subdir, crop_name)
                cv2.imwrite(crop_path, crop)
                rel = os.path.relpath(crop_path, output_dir)
                out_f.write(json.dumps({
                    "image_filename": rel, "expected_text": token, "language": language,
                }, ensure_ascii=False) + "\n")
                total_crops += 1
                # Augmentation
                for a in range(augment):
                    pil = Image.open(crop_path).convert("RGB")
                    aug = augment_crop(pil, seed=seed + total_crops * 100 + a)
                    aug_name = f"{prefix}{line_idx:04d}_{w_idx:02d}_aug{a}.jpg"
                    aug_path = os.path.join(lang_subdir, aug_name)
                    aug.save(aug_path)
                    rel_a = os.path.relpath(aug_path, output_dir)
                    out_f.write(json.dumps({
                        "image_filename": rel_a, "expected_text": token, "language": language,
                    }, ensure_ascii=False) + "\n")
                    total_crops += 1

    return {
        "output_dir": output_dir,
        "jsonl_path": out_jsonl,
        "lines_processed": lines_processed,
        "perfect_match_lines": matched,
        "aligned_lines": aligned,
        "skipped_mismatch": skipped_mismatch,
        "word_crops": total_crops,
        "skipped_long": skipped_long,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate recognition training datasets from BSTD data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert existing BSTD recognition JSON to JSONL")
    p_conv.add_argument("--bstd-recognition-json", required=True, help="Path to train_recognition_data.json")
    p_conv.add_argument("--bstd-recognition-root", required=True, help="Root of recognition/train/ crops")
    p_conv.add_argument("--language", required=True, help="Language to extract (e.g. marathi)")
    p_conv.add_argument("--output-dir", required=True, help="Output directory for crops + labels.jsonl")

    # --- crop ---
    p_crop = sub.add_parser("crop", help="Crop word images from BSTD scene images")
    p_crop.add_argument("--bstd-json", required=True, help="Path to BSTD_v17.57.json or BSTD_release_v1.json")
    p_crop.add_argument("--scene-root", required=True, help="Directory with A-L scene image folders")
    p_crop.add_argument("--language", required=True)
    p_crop.add_argument("--output-dir", required=True)
    p_crop.add_argument("--padding", type=int, default=4, help="Pixels of padding around polygons")
    p_crop.add_argument("--two-word", action="store_true", help="Also generate 2-word crops from adjacent polygons")
    p_crop.add_argument("--augment", type=int, default=0, help="Number of augmented variants per crop")
    p_crop.add_argument("--max-per-image", type=int, default=0, help="Cap crops per scene image (0 = no cap)")
    p_crop.add_argument("--seed", type=int, default=42)

    # --- split ---
    p_split = sub.add_parser("split", help="Split line-level images into word crops")
    p_split.add_argument("--jsonl", required=True, help="JSONL with image_filename + expected_text (line-level)")
    p_split.add_argument("--image-dir", required=True, help="Directory containing the line images")
    p_split.add_argument("--language", default="marathi")
    p_split.add_argument("--output-dir", required=True)
    p_split.add_argument("--augment", type=int, default=0, help="Augmented variants per word crop")
    p_split.add_argument("--max-label-length", type=int, default=25, help="Drop tokens longer than this")
    p_split.add_argument("--threshold", type=int, default=200, help="Binarization threshold (0-255)")
    p_split.add_argument("--min-gap-width", type=int, default=3, help="Min gap width to split words")
    p_split.add_argument("--max-split-ratio", type=float, default=3.0,
                         help="Max crop/token count ratio before a line is skipped "
                              "(raise for code-heavy lines that fragment at slashes)")
    p_split.add_argument("--append", action="store_true", help="Append to existing dataset")
    p_split.add_argument("--prefix", default="", help="Filename prefix for crops (use with --append)")
    p_split.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.mode == "convert":
        result = convert_recognition_json(
            args.bstd_recognition_json, args.bstd_recognition_root,
            args.language, args.output_dir,
        )
        print(f"Conversion complete: {result['count']} samples, {result['skipped']} skipped")
        print(f"  JSONL: {result['jsonl_path']}")
    elif args.mode == "crop":
        result = crop_from_scenes(
            args.bstd_json, args.scene_root, args.language, args.output_dir,
            padding=args.padding, two_word=args.two_word, augment=args.augment,
            max_per_image=args.max_per_image, seed=args.seed,
        )
        print(f"Cropping complete:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif args.mode == "split":
        result = split_lines_to_dataset(
            args.jsonl, args.image_dir, args.output_dir, language=args.language,
            augment=args.augment, max_label_length=args.max_label_length,
            threshold=args.threshold, min_gap_width=args.min_gap_width,
            append=args.append, prefix=args.prefix, seed=args.seed,
            max_split_ratio=args.max_split_ratio,
        )
        print(f"Line splitting complete:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
