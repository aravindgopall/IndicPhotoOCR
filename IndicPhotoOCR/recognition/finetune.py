"""Fine-tune a PARSeq recognition checkpoint on a JSONL image dataset.

This is designed for the Marathi (and other Indic) data format used by the
project's evaluation sets::

    {"image_filename": "/images/0001.png", "expected_text": "महाराष्ट्र शासन", "issue_type": "..."}

The standard strhub ``LmdbDataset`` performs NFKD->ASCII unicode normalisation
which **destroys Devanagari/Indic characters**, so a small dedicated
``JsonlDataset`` is provided here that loads images directly from a folder and
keeps the unicode labels intact.

Typical workflow
----------------
1. (Optional) extend the checkpoint's charset first so the model can output any
   new characters present in the data::

       python -m IndicPhotoOCR.recognition.charset_extension \\
           --checkpoint marathi.ckpt --data-jsonl data/validation.jsonl \\
           --output marathi_extended.ckpt

2. Fine-tune::

       python -m IndicPhotoOCR.recognition.finetune \\
           --checkpoint marathi_extended.ckpt \\
           --image-dir data/images --labels data/validation.jsonl \\
           --output marathi_finetuned.ckpt --epochs 20 --lr 1e-4

The output is a normal PyTorch-Lightning checkpoint loadable by
``IndicPhotoOCR.recognition.parseq_recogniser.PARseqrecogniser``.

Note
----
The fine-tuned checkpoint is *drop-in*: ``charset_train`` is preserved in the
hyper-parameters, so no inference code changes are needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from torchvision import transforms as T

from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
from IndicPhotoOCR.recognition.charset_extension import (
    compute_extended_charset,
    inspect_checkpoint,
    extend_checkpoint_charset,
)


__all__ = ["JsonlDataset", "collate_fn", "collect_dataset_chars", "finetune"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JsonlDataset(Dataset):
    """Dataset over a JSONL file of ``{"image_filename", "expected_text"}`` rows.

    Unlike ``LmdbDataset``, labels are kept as-is (no NFKD/ASCII normalisation)
    so Indic unicode is preserved. Labels are filtered through the model's
    charset via ``CharsetAdapter`` and samples that end up empty (no supported
    characters) or are longer than ``max_label_length`` are dropped.
    """

    def __init__(
        self,
        labels_file: str,
        image_dir: str,
        charset: str,
        max_label_length: int,
        transform: Optional[Callable] = None,
        remove_whitespace: bool = True,
        min_image_dim: int = 0,
    ) -> None:
        super().__init__()
        self.labels_file = labels_file
        self.image_dir = image_dir
        self.transform = transform
        self.max_label_length = max_label_length
        self.min_image_dim = min_image_dim
        self.remove_whitespace = remove_whitespace

        # Local import to avoid a hard dependency at module import time.
        from IndicPhotoOCR.utils.strhub.data.utils import CharsetAdapter

        self._charset_adapter = CharsetAdapter(charset)
        self.samples: List[Tuple[str, str]] = self._load_samples()

    def _resolve_image(self, image_filename: str) -> str:
        """Resolve a JSONL image path against ``image_dir``.

        Accepts absolute paths, repo-relative paths (``/images/0001.png``), and
        bare filenames.
        """
        name = image_filename.strip()
        # Strip a leading slash so "/images/0001.png" -> "images/0001.png"
        rel = name.lstrip("/")
        candidates = [
            os.path.join(self.image_dir, os.path.basename(name)),
            os.path.join(self.image_dir, rel),
            name,  # absolute or already-correct path
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]  # let the open() raise a clear error later

    def _load_samples(self) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        dropped_empty = 0
        dropped_long = 0
        with open(self.labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_filename = row.get("image_filename") or row.get("image") or row.get("filename")
                text = row.get("expected_text") or row.get("text") or row.get("label") or ""
                if image_filename is None:
                    continue
                if self.remove_whitespace:
                    text = "".join(text.split())
                text = self._charset_adapter(text)
                if not text:
                    dropped_empty += 1
                    continue
                if len(text) > self.max_label_length:
                    dropped_long += 1
                    continue
                samples.append((self._resolve_image(image_filename), text))
        if dropped_empty or dropped_long:
            print(
                f"[JsonlDataset] loaded {len(samples)} samples "
                f"(dropped {dropped_empty} empty, {dropped_long} too-long)"
            )
        else:
            print(f"[JsonlDataset] loaded {len(samples)} samples from {self.labels_file}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        img = Image.open(image_path).convert("RGB")
        if self.min_image_dim > 0:
            w, h = img.size
            if w < self.min_image_dim or h < self.min_image_dim:
                # Return a neighbour instead of crashing a whole training run.
                return self.__getitem__((index + 1) % len(self))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    """Stack image tensors, keep labels as a list of strings (PARSeq expects this)."""
    images, labels = zip(*batch)
    return torch.stack(images, 0), list(labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_dataset_chars(labels_file: str) -> str:
    """Return the deduplicated union of all characters in ``expected_text``."""
    chars: list[str] = []
    seen: set[str] = set()
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("expected_text", "")
            except json.JSONDecodeError:
                continue
            for c in text:
                if c not in seen:
                    seen.add(c)
                    chars.append(c)
    return "".join(chars)


def get_transform(img_size: Sequence[int], augment: bool = False) -> T.Compose:
    transforms: list = []
    if augment:
        from IndicPhotoOCR.utils.strhub.data.augment import rand_augment_transform

        transforms.append(rand_augment_transform())
    transforms.extend([
        T.Resize(tuple(img_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    return T.Compose(transforms)


def _split_samples(dataset: JsonlDataset, val_fraction: float) -> Tuple[JsonlDataset, JsonlDataset]:
    """Split a dataset's samples into train/val (shares config, no re-read).

    Caller guarantees ``0 < val_fraction < 1`` and ``len(dataset) > 2``.
    """
    n = len(dataset)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    train_ds = _ShallowCopy(dataset, dataset.samples[:n_train])
    val_ds = _ShallowCopy(dataset, dataset.samples[n_train:])
    return train_ds, val_ds


class _ShallowCopy(JsonlDataset):
    """A JsonlDataset that reuses another's config but owns its own sample list."""

    def __init__(self, parent: "JsonlDataset", samples: List[Tuple[str, str]]):
        self.labels_file = parent.labels_file
        self.image_dir = parent.image_dir
        self.transform = parent.transform
        self.max_label_length = parent.max_label_length
        self.min_image_dim = parent.min_image_dim
        self.remove_whitespace = parent.remove_whitespace
        self._charset_adapter = parent._charset_adapter
        self.samples = samples


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def finetune(
    checkpoint: str,
    image_dir: str,
    labels: str,
    output: str,
    val_labels: Optional[str] = None,
    val_split: float = 0.1,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 8,
    num_workers: int = 0,
    max_label_length: Optional[int] = None,
    warmup_pct: float = 0.1,
    weight_decay: float = 0.0,
    accumulate_grad_batches: int = 1,
    augment: bool = False,
    extend_charset: bool = False,
    extra_chars: str = "",
    device: Optional[str] = None,
    seed: int = 42,
    limit_train_batches: Optional[float] = None,
) -> Dict:
    """Fine-tune a PARSeq checkpoint on a JSONL dataset.

    Args:
        checkpoint: Source ``.ckpt`` (will be loaded with ``load_from_checkpoint``).
        image_dir: Directory containing the images referenced by the JSONL.
        labels: JSONL file with ``image_filename`` + ``expected_text`` rows.
        output: Where to write the fine-tuned checkpoint.
        val_labels: Optional separate JSONL for validation. If omitted, a
            ``val_split`` fraction of ``labels`` is held out.
        val_split: Fraction of ``labels`` to hold out for validation (ignored
            when ``val_labels`` is given or when 0).
        epochs: Number of training epochs.
        lr: Peak learning rate (OneCycleLR is used).
        batch_size: Mini-batch size.
        num_workers: DataLoader workers.
        max_label_length: If larger than the checkpoint's ``max_label_length``,
            the model's ``pos_queries`` are grown so longer sequences can be
            decoded (needed for long Indic government text). If smaller, labels
            are filtered more aggressively. Defaults to the checkpoint's value.
        extend_charset: If True, grow the checkpoint's charset to cover every
            character in ``labels`` (plus ``extra_chars``) before training, so
            the model can actually emit them. The extension is written to a
            temporary checkpoint and used as the starting point.
        extra_chars: Extra characters to add when ``extend_charset`` is True.
        device: ``'cpu'``, ``'cuda'``, ``'cuda:0'``, ... Defaults to CUDA if available.
        seed: RNG seed.

    Returns:
        A dict summarising the run (paths, charset, sample counts, best score).
    """
    pl.seed_everything(seed)

    # ---- optional charset / max_label_length extension -------------------
    # We may need to grow the checkpoint's charset (to emit new chars) and/or
    # its max_label_length (to decode long Indic lines). Both are handled in a
    # single pass by extend_checkpoint_charset.
    ckpt_info = inspect_checkpoint(checkpoint)
    ckpt_max_label_length = ckpt_info.get("max_label_length")

    combined_chars = extra_chars
    charset_grows = False
    if extend_charset:
        data_chars = collect_dataset_chars(labels)
        combined_chars = extra_chars + data_chars
        new_charset = compute_extended_charset(ckpt_info["charset_train"], combined_chars)
        charset_grows = len(new_charset) > len(ckpt_info["charset_train"])
        if not charset_grows:
            print("[finetune] dataset chars already covered by checkpoint charset; no charset extension needed")

    need_max_ext = (
        max_label_length is not None
        and ckpt_max_label_length is not None
        and max_label_length > ckpt_max_label_length
    )

    if charset_grows or need_max_ext:
        ext_path = os.path.splitext(output)[0] + "_extended_src.ckpt"
        ext_info = extend_checkpoint_charset(
            checkpoint,
            combined_chars if charset_grows else "",
            output_path=ext_path,
            init="mean",
            seed=seed,
            max_label_length=max_label_length if need_max_ext else None,
        )
        print(f"[finetune] extended source checkpoint -> {ext_path}")
        print(f"           +{len(ext_info['added_chars'])} chars, "
              f"max_label_length={ext_info.get('max_label_length')}")
        checkpoint = ext_path

    # ---- load model -------------------------------------------------------
    model = load_from_checkpoint(checkpoint).train()
    hp = model.hparams
    charset_train = hp.charset_train
    # Filter labels at the model's actual capacity (allow clamping DOWN only).
    effective_max_label_length = min(max_label_length or hp.max_label_length, hp.max_label_length)

    # Warn about characters the model still cannot represent.
    data_chars = collect_dataset_chars(labels)
    missing = "".join(c for c in data_chars if c not in charset_train)
    if missing:
        print(
            f"[finetune] WARNING: {len(missing)} dataset chars are NOT in the model charset "
            f"and will be stripped from labels: {missing!r}"
        )
        print("           Re-run with --extend-charset to grow the vocabulary first.")

    img_size = tuple(hp.img_size)
    train_transform = get_transform(img_size, augment=augment)
    val_transform = get_transform(img_size, augment=False)

    # ---- datasets ---------------------------------------------------------
    full_train = JsonlDataset(
        labels, image_dir, charset_train, effective_max_label_length,
        transform=train_transform, remove_whitespace=True,
    )
    if val_labels:
        train_ds = full_train
        val_ds = JsonlDataset(
            val_labels, image_dir, charset_train, effective_max_label_length,
            transform=val_transform, remove_whitespace=True,
        )
    elif 0 < val_split < 1 and len(full_train) > 2:
        train_ds, val_ds = _split_samples(full_train, val_split)
        val_ds.transform = val_transform
    else:
        train_ds = full_train
        val_ds = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, collate_fn=collate_fn,
        )

    # ---- trainer ----------------------------------------------------------
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    # Map device string to Lightning accelerator.
    if "cuda" in str(device):
        accelerator = "gpu"
    elif "mps" in str(device):
        accelerator = "mps"
    else:
        accelerator = "cpu"
    devices = 1
    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    if val_loader is not None:
        trainer_kwargs["callbacks"] = [
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.dirname(os.path.abspath(output)) or ".",
                filename=os.path.splitext(os.path.basename(output))[0],
                monitor="val_accuracy",
                mode="max",
                save_top_k=1,
                save_last=False,
            )
        ]
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches

    trainer = pl.Trainer(**trainer_kwargs)

    # configure_optimizers() reads these instance attributes (set in BaseSystem.__init__),
    # not self.hparams, so override them directly.
    model.lr = lr
    model.warmup_pct = warmup_pct
    model.weight_decay = weight_decay
    # Note: PARSeq scales lr by batch_size/256 (see BaseSystem.configure_optimizers),
    # so the *effective* lr = lr * batch_size * accumulate_grad_batches / 256.

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ---- save -------------------------------------------------------------
    # Prefer the best checkpoint if validation was used, else the final state.
    best_path = None
    if val_loader is not None and trainer.checkpoint_callback is not None:
        best_path = trainer.checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        # The callback may already have written straight to `output`.
        try:
            same = os.path.samefile(best_path, output)
        except FileNotFoundError:
            same = False
        if not same:
            import shutil
            shutil.copyfile(best_path, output)
    else:
        trainer.save_checkpoint(output)

    return {
        "output": output,
        "charset_train": charset_train,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds) if val_ds is not None else 0,
        "epochs": epochs,
        "best_val_path": best_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a PARSeq recognition checkpoint on a JSONL image dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Source .ckpt to fine-tune")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--labels", required=True, help="JSONL with image_filename + expected_text")
    parser.add_argument("--val-labels", default=None, help="Optional separate validation JSONL")
    parser.add_argument("--output", "-o", required=True, help="Output .ckpt path")
    parser.add_argument("--val-split", type=float, default=0.1, help="Holdout fraction if no --val-labels")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-label-length", type=int, default=None)
    parser.add_argument("--warmup-pct", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--augment", action="store_true", help="Apply strhub RandAugment during training")
    parser.add_argument(
        "--extend-charset", action="store_true",
        help="Grow the checkpoint charset to cover dataset chars before training",
    )
    parser.add_argument("--extra-chars", default="", help="Extra chars to add with --extend-charset")
    parser.add_argument("--device", default=None, help="'cpu' / 'cuda' / 'cuda:0' (auto if omitted)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=float, default=None, help="Cap batches/epoch (debug/smoke)")
    args = parser.parse_args(argv)

    result = finetune(
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        labels=args.labels,
        output=args.output,
        val_labels=args.val_labels,
        val_split=args.val_split,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_label_length=args.max_label_length,
        warmup_pct=args.warmup_pct,
        weight_decay=args.weight_decay,
        accumulate_grad_batches=args.accumulate_grad_batches,
        augment=args.augment,
        extend_charset=args.extend_charset,
        extra_chars=args.extra_chars,
        device=args.device,
        seed=args.seed,
        limit_train_batches=args.limit_train_batches,
    )
    print("Fine-tuning complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
