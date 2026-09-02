"""Extend the charset of a PARSeq recognition checkpoint.

The Indic PARSeq recognisers are trained on a fixed character set. When a
scene image contains characters that were never seen during training
(e.g. punctuation ``/ ( ) * .``, digits, or rare conjuncts/matras), the model
has no output slot for them and the ``CharsetAdapter`` silently strips them
from the prediction.

This module grows the vocabulary of an existing checkpoint so that the new
characters become *predictable*. The learned weights for every existing
character are preserved exactly; only fresh rows are added for the new
characters (initialised from the mean of the existing rows, or randomly).

How token indices are laid out
------------------------------
``Tokenizer`` orders tokens as ``(EOS,) + tuple(charset) + (BOS, PAD)``, so for
a charset of length ``N``:

* ``text_embed``               -> ``N + 3`` rows  (0 = EOS, 1..N = charset, N+1 = BOS, N+2 = PAD)
* ``head`` (Linear, no bias on BOS/PAD) -> ``N + 1`` outputs (0 = EOS, 1..N = charset)

When the charset grows from ``N_old`` to ``N_new`` (old chars kept as a prefix
so their indices are unchanged), the special tokens BOS/PAD simply shift to the
end of the embedding table. ``head`` only grows by the number of new chars.

The resulting checkpoint is a standard PyTorch-Lightning checkpoint that
``load_from_checkpoint`` (and therefore ``IndicPhotoOCR.recognition``) loads
transparently -- no inference code changes are required.

Usage
-----
::

    python -m IndicPhotoOCR.recognition.charset_extension \\
        --checkpoint marathi.ckpt \\
        --extra-chars "()/.,*-:;\"" \\
        --output marathi_extended.ckpt

    # or discover the extra chars automatically from a JSONL dataset
    python -m IndicPhotoOCR.recognition.charset_extension \\
        --checkpoint marathi.ckpt \\
        --data-jsonl "/path/to/validation.jsonl" \\
        --output marathi_extended.ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch


__all__ = [
    "compute_extended_charset",
    "extract_charset_from_checkpoint",
    "extend_checkpoint_charset",
    "inspect_checkpoint",
]


# ---------------------------------------------------------------------------
# Pure-string helpers (no torch required -- easily unit-testable)
# ---------------------------------------------------------------------------

def compute_extended_charset(old_charset: str, extra_chars: str) -> str:
    """Return ``old_charset`` with any new characters from ``extra_chars`` appended.

    The original order is preserved and duplicates (case-sensitive, character
    by character) are dropped. Keeping the old charset as a strict prefix is
    essential: it guarantees existing token indices are unchanged, so every
    learned weight continues to line up with the same character.
    """
    if not old_charset:
        return extra_chars
    seen = set(old_charset)
    new_chars = [c for c in extra_chars if c not in seen and not seen.add(c)]
    return old_charset + "".join(new_chars)


# ---------------------------------------------------------------------------
# Checkpoint inspection
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # weights_only=False is required for PyTorch-Lightning checkpoints, which
    # store arbitrary Python objects (e.g. hyper_parameters dicts, optimizer
    # states).  These checkpoints come from the project's own release assets.
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def extract_charset_from_checkpoint(checkpoint_path: str) -> Dict[str, str]:
    """Return the ``charset_train`` / ``charset_test`` stored in a checkpoint."""
    ckpt = _load_checkpoint(checkpoint_path)
    hp = ckpt.get("hyper_parameters", {})
    charset_train = hp.get("charset_train", "")
    charset_test = hp.get("charset_test", charset_train)
    return {"charset_train": charset_train, "charset_test": charset_test}


def inspect_checkpoint(checkpoint_path: str) -> Dict:
    """Return a human-readable summary of a checkpoint's charset + dimensions."""
    ckpt = _load_checkpoint(checkpoint_path)
    hp = dict(ckpt.get("hyper_parameters", {}))
    sd = ckpt.get("state_dict", {})
    info = {
        "checkpoint": checkpoint_path,
        "charset_train": hp.get("charset_train", ""),
        "charset_test": hp.get("charset_test", hp.get("charset_train", "")),
        "charset_train_len": len(hp.get("charset_train", "")),
        "max_label_length": hp.get("max_label_length"),
        "img_size": hp.get("img_size"),
        "embed_dim": hp.get("embed_dim"),
    }
    # Cross-check against the actual weight shapes so we never silently lie.
    for key in ("model.text_embed.embedding.weight", "model.head.weight", "model.head.bias"):
        if key in sd:
            info.setdefault("weight_shapes", {})[key] = tuple(sd[key].shape)
    return info


# ---------------------------------------------------------------------------
# State-dict remapping (the core of the extension)
# ---------------------------------------------------------------------------

def _init_rows(reference: torch.Tensor, num_new: int, init: str, generator: torch.Generator) -> torch.Tensor:
    """Create ``num_new`` rows shaped like ``reference``'s feature dimension.

    ``reference`` is the existing weight block (rows x features). We only use
    its feature shape and per-row statistics, never its row count.
    """
    feat_dim = reference.shape[1] if reference.dim() == 2 else 1
    if init == "random":
        new_rows = torch.empty(num_new, *reference.shape[1:], dtype=reference.dtype)
        torch.nn.init.trunc_normal_(new_rows, std=0.02, generator=generator)
        return new_rows
    # default: mean of existing rows -- a safe, neutral starting point for a
    # freshly added token (keeps initial logits/embeddings near the average).
    mean_row = reference.mean(dim=0, keepdim=True)
    return mean_row.expand(num_new, *reference.shape[1:]).contiguous().clone()


def _expand_pos_queries(weight: torch.Tensor, old_max: int, new_max: int, init: str, gen: torch.Generator) -> torch.Tensor:
    """pos_queries: (1, old_max+1, D) -> (1, new_max+1, D).

    Existing position embeddings keep their indices; new positions are
    appended (mean or random init of existing rows). NB: the expanding
    dimension is dim 1, not dim 0, so this does NOT reuse the 2-D ``_init_rows``.
    """
    old_len = old_max + 1
    new_len = new_max + 1
    if new_len == old_len:
        return weight
    batch, _, feat = weight.shape
    new_w = torch.empty(batch, new_len, feat, dtype=weight.dtype)
    new_w[:, :old_len] = weight
    num_new = new_len - old_len
    if init == "random":
        block = torch.empty(batch, num_new, feat, dtype=weight.dtype)
        torch.nn.init.trunc_normal_(block, std=0.02, generator=gen)
    else:
        # mean over existing positions -> (B, 1, D), broadcast to new positions
        block = weight.mean(dim=1, keepdim=True).expand(batch, num_new, feat).contiguous().clone()
    new_w[:, old_len:new_len] = block
    return new_w


def _expand_head_weight(weight: torch.Tensor, n_old: int, n_new: int, init: str, gen: torch.Generator) -> torch.Tensor:
    """head.weight: (N_old+1, D) -> (N_new+1, D). Rows 0..N_old preserved (EOS + old charset)."""
    old_rows = n_old + 1
    new_rows = n_new + 1
    if new_rows == old_rows:
        return weight
    new_w = torch.empty(new_rows, *weight.shape[1:], dtype=weight.dtype)
    new_w[:old_rows] = weight
    new_w[old_rows:new_rows] = _init_rows(weight, new_rows - old_rows, init, gen)
    return new_w


def _expand_head_bias(bias: torch.Tensor, n_old: int, n_new: int, init: str, gen: torch.Generator) -> torch.Tensor:
    """head.bias: (N_old+1,) -> (N_new+1,). Mirrors head.weight row mapping."""
    old_rows = n_old + 1
    new_rows = n_new + 1
    if new_rows == old_rows:
        return bias
    new_b = torch.empty(new_rows, dtype=bias.dtype)
    new_b[:old_rows] = bias
    ref = bias.unsqueeze(1)  # treat as (rows, 1) so _init_rows works generically
    new_b[old_rows:new_rows] = _init_rows(ref, new_rows - old_rows, init, gen).squeeze(1)
    return new_b


def _expand_text_embed(weight: torch.Tensor, n_old: int, n_new: int, init: str, gen: torch.Generator) -> torch.Tensor:
    """text_embed.embedding.weight: (N_old+3, D) -> (N_new+3, D).

    Layout: 0=EOS, 1..N=charset, N+1=BOS, N+2=PAD.
    EOS + old charset keep their indices; new chars occupy N_old+1..N_new;
    BOS/PAD move from N_old+1/N_old+2 to N_new+1/N_new+2.
    """
    old_total = n_old + 3
    new_total = n_new + 3
    if new_total == old_total:
        return weight
    new_w = torch.empty(new_total, *weight.shape[1:], dtype=weight.dtype)
    # 1. EOS + old charset (indices unchanged)
    new_w[: n_old + 1] = weight[: n_old + 1]
    # 2. new charset characters
    new_w[n_old + 1 : n_new + 1] = _init_rows(weight, n_new - n_old, init, gen)
    # 3. relocate BOS and PAD to the new tail
    new_w[n_new + 1] = weight[n_old + 1]  # BOS
    new_w[n_new + 2] = weight[n_old + 2]  # PAD
    return new_w


def _remap_state_dict(
    state_dict: Dict[str, torch.Tensor],
    n_old: int,
    n_new: int,
    init: str,
    gen: torch.Generator,
    old_max_label_length: Optional[int] = None,
    new_max_label_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict with head + text_embed (and optionally pos_queries) grown."""
    new_sd: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if n_new != n_old and key.endswith("text_embed.embedding.weight"):
            new_sd[key] = _expand_text_embed(tensor, n_old, n_new, init, gen)
        elif n_new != n_old and key.endswith("head.weight"):
            new_sd[key] = _expand_head_weight(tensor, n_old, n_new, init, gen)
        elif n_new != n_old and key.endswith("head.bias"):
            new_sd[key] = _expand_head_bias(tensor, n_old, n_new, init, gen)
        elif (
            old_max_label_length is not None
            and new_max_label_length is not None
            and new_max_label_length > old_max_label_length
            and key.endswith("pos_queries")
        ):
            new_sd[key] = _expand_pos_queries(tensor, old_max_label_length, new_max_label_length, init, gen)
        else:
            new_sd[key] = tensor
    return new_sd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extend_checkpoint_charset(
    checkpoint_path: str,
    extra_chars: str,
    output_path: Optional[str] = None,
    init: str = "mean",
    seed: int = 42,
    also_extend_charset_test: bool = True,
    max_label_length: Optional[int] = None,
) -> Dict:
    """Extend a PARSeq checkpoint's charset by ``extra_chars``.

    Args:
        checkpoint_path: Path to the source ``.ckpt``.
        extra_chars: Characters to add. Duplicates / chars already present are ignored.
        output_path: Where to write the extended checkpoint. Defaults to
            ``<checkpoint>_extended.ckpt`` next to the source.
        init: ``"mean"`` (default) initialises new rows from the mean of existing
            rows; ``"random"`` uses truncated-normal initialisation.
        seed: RNG seed for reproducible ``random`` init.
        also_extend_charset_test: Also grow ``charset_test`` (used by the
            ``CharsetAdapter`` post-filter). Almost always wanted.
        max_label_length: If given and larger than the checkpoint's current
            ``max_label_length``, also grow ``pos_queries`` so the model can
            decode longer sequences. Useful for Indic government text.

    Returns:
        A dict with ``output_path``, ``old_charset``, ``new_charset``,
        ``added_chars``, ``old_num_tokens``, ``new_num_tokens`` and the (possibly
        changed) ``max_label_length``.
    """
    if init not in ("mean", "random"):
        raise ValueError(f"init must be 'mean' or 'random', got {init!r}")
    if not extra_chars and not max_label_length:
        raise ValueError("Provide extra_chars and/or a larger max_label_length; nothing to extend")

    ckpt = _load_checkpoint(checkpoint_path)
    hp = dict(ckpt.get("hyper_parameters", {}))
    old_charset_train = hp.get("charset_train", "")
    if not old_charset_train:
        raise ValueError("Checkpoint has no 'charset_train' hyperparameter; cannot extend.")

    old_charset_test = hp.get("charset_test", old_charset_train)

    new_charset_train = compute_extended_charset(old_charset_train, extra_chars)
    new_charset_test = compute_extended_charset(old_charset_test, extra_chars) if also_extend_charset_test else old_charset_test

    n_old, n_new = len(old_charset_train), len(new_charset_train)
    added = new_charset_train[n_old:]
    if not added and not (max_label_length and hp.get("max_label_length") and max_label_length > hp["max_label_length"]):
        raise ValueError(
            "All provided characters are already in the checkpoint charset, "
            "and max_label_length was not increased; nothing to extend."
        )

    old_max_label_length = hp.get("max_label_length")
    effective_new_max = max_label_length if (max_label_length and old_max_label_length and max_label_length > old_max_label_length) else old_max_label_length

    gen = torch.Generator().manual_seed(seed)
    ckpt["state_dict"] = _remap_state_dict(
        ckpt.get("state_dict", {}), n_old, n_new, init, gen,
        old_max_label_length=old_max_label_length, new_max_label_length=effective_new_max,
    )
    new_hp = {
        **hp,
        "charset_train": new_charset_train,
        "charset_test": new_charset_test,
    }
    if effective_new_max is not None:
        new_hp["max_label_length"] = effective_new_max
    ckpt["hyper_parameters"] = new_hp
    # Reset training bookkeeping so the extended checkpoint looks like a fresh one.
    ckpt["epoch"] = 0
    ckpt["global_step"] = 0
    for stale in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
        ckpt.pop(stale, None)

    if output_path is None:
        root, ext = os.path.splitext(checkpoint_path)
        output_path = f"{root}_extended{ext or '.ckpt'}"
    torch.save(ckpt, output_path)

    return {
        "output_path": output_path,
        "old_charset": old_charset_train,
        "new_charset": new_charset_train,
        "added_chars": added,
        "old_num_tokens": n_old + 3,
        "new_num_tokens": n_new + 3,
        "max_label_length": effective_new_max,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _chars_from_jsonl(jsonl_path: str) -> str:
    chars: list[str] = []
    seen: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
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


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extend the charset of a PARSeq recognition checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to source .ckpt")
    parser.add_argument(
        "--extra-chars",
        default="",
        help="Characters to add (e.g. '()/.,*-:;'). Use --data-jsonl as an alternative.",
    )
    parser.add_argument(
        "--data-jsonl",
        default=None,
        help="JSONL dataset file; its expected_text characters are auto-added.",
    )
    parser.add_argument("--output", "-o", default=None, help="Output checkpoint path")
    parser.add_argument("--init", choices=("mean", "random"), default="mean", help="Init for new token rows")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for 'random' init")
    parser.add_argument(
        "--max-label-length", type=int, default=None,
        help="Grow pos_queries to decode longer sequences (e.g. 100 for long Indic text).",
    )
    parser.add_argument("--inspect", action="store_true", help="Just print checkpoint info and exit")
    args = parser.parse_args(argv)

    if args.inspect:
        info = inspect_checkpoint(args.checkpoint)
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0

    extra = args.extra_chars
    if args.data_jsonl:
        discovered = _chars_from_jsonl(args.data_jsonl)
        print(f"Discovered {len(discovered)} unique chars in {args.data_jsonl}")
        extra = extra + discovered

    if not extra and not args.max_label_length:
        parser.error("Provide --extra-chars and/or --data-jsonl and/or --max-label-length.")

    result = extend_checkpoint_charset(
        args.checkpoint, extra, output_path=args.output, init=args.init, seed=args.seed,
        max_label_length=args.max_label_length,
    )
    print(f"Extended checkpoint written to: {result['output_path']}")
    print(f"  old charset length : {len(result['old_charset'])}")
    print(f"  new charset length : {len(result['new_charset'])}")
    print(f"  added chars        : {result['added_chars']!r}  ({len(result['added_chars'])} new)")
    print(f"  num tokens         : {result['old_num_tokens']} -> {result['new_num_tokens']}")
    if result.get("max_label_length") is not None:
        print(f"  max_label_length   : {result['max_label_length']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
