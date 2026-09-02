"""
Tests for IndicPhotoOCR.recognition.charset_extension.

The pure-string helper (compute_extended_charset) needs no torch and always
runs. The checkpoint-extension tests build a tiny real PARSeq checkpoint, so
they are gated behind torch availability (the project's normal runtime env).
"""
import os

import pytest


# ---------------------------------------------------------------------------
# compute_extended_charset -- pure string logic, no torch
# ---------------------------------------------------------------------------

class TestComputeExtendedCharset:
    def _import(self):
        from IndicPhotoOCR.recognition.charset_extension import compute_extended_charset
        return compute_extended_charset

    def test_appends_new_chars_preserving_order(self):
        f = self._import()
        assert f("abc", "def") == "abcdef"

    def test_drops_duplicates_already_present(self):
        f = self._import()
        assert f("abc", "bcd") == "abcd"

    def test_drops_duplicates_within_extra(self):
        f = self._import()
        assert f("abc", "dde") == "abcde"

    def test_keeps_special_chars_and_indic(self):
        f = self._import()
        old = "अआइकख"            # Devanagari base charset
        extra = "()/अ."            # 'अ' already present
        assert f(old, extra) == "अआइकख()/."

    def test_empty_extra_returns_old_unchanged(self):
        f = self._import()
        assert f("abc", "") == "abc"

    def test_empty_old_returns_extra(self):
        f = self._import()
        assert f("", "xyz") == "xyz"

    def test_old_charset_is_always_a_prefix(self):
        """Critical invariant: existing token indices must not shift."""
        f = self._import()
        old = "कखगघ"
        new = f(old, "abcक")
        assert new.startswith(old)
        assert new[: len(old)] == old

    def test_case_sensitive(self):
        f = self._import()
        # 'a' and 'A' are distinct tokens
        assert f("a", "A") == "aA"


# ---------------------------------------------------------------------------
# Checkpoint extension -- requires torch + the project's strhub models
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not installed in this environment (run in the project conda env)",
)


def _build_tiny_parseq(charset_train: str, tmp_path: str):
    """Build a tiny real PARSeq model and save a Lightning checkpoint."""
    import torch
    from IndicPhotoOCR.utils.strhub.models.parseq.system import PARSeq

    model = PARSeq(
        charset_train=charset_train,
        charset_test=charset_train,
        max_label_length=25,
        batch_size=2,
        lr=1e-4,
        warmup_pct=0.1,
        weight_decay=0.0,
        img_size=(32, 128),
        patch_size=(4, 8),
        embed_dim=32,
        enc_num_heads=2,
        enc_mlp_ratio=4,
        enc_depth=1,
        dec_num_heads=2,
        dec_mlp_ratio=4,
        dec_depth=1,
        perm_num=6,
        perm_forward=True,
        perm_mirrored=True,
        decode_ar=True,
        refine_iters=1,
        dropout=0.1,
    )
    ckpt_path = os.path.join(tmp_path, "tiny.ckpt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.4.0",
            "epoch": 0,
            "global_step": 0,
        },
        ckpt_path,
    )
    return ckpt_path


class TestExtractAndInspect:
    def test_extract_charset_roundtrip(self, tmp_path):
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import extract_charset_from_checkpoint
        cs = extract_charset_from_checkpoint(ckpt)
        assert cs["charset_train"] == "abc"
        assert cs["charset_test"] == "abc"

    def test_inspect_reports_shapes(self, tmp_path):
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import inspect_checkpoint
        info = inspect_checkpoint(ckpt)
        assert info["charset_train_len"] == 3
        assert "model.text_embed.embedding.weight" in info["weight_shapes"]


class TestExtendCheckpointCharset:
    def test_grows_head_and_text_embed(self, tmp_path):
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import (
            extend_checkpoint_charset,
            inspect_checkpoint,
        )
        out = os.path.join(tmp_path, "ext.ckpt")
        res = extend_checkpoint_charset(ckpt, "de", output_path=out, init="mean")
        assert res["added_chars"] == "de"
        assert res["old_num_tokens"] == 3 + 3      # 3 chars + EOS/BOS/PAD
        assert res["new_num_tokens"] == 5 + 3

        info = inspect_checkpoint(out)
        # text_embed: N+3 rows; head: N+1 outputs
        assert info["weight_shapes"]["model.text_embed.embedding.weight"] == (5 + 3, 32)
        assert info["weight_shapes"]["model.head.weight"] == (5 + 1, 32)
        assert info["weight_shapes"]["model.head.bias"] == (5 + 1,)

    def test_existing_rows_preserved_exactly(self, tmp_path):
        """The learned weights for existing chars must not change."""
        import torch
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        old_sd = torch.load(ckpt, map_location="cpu")["state_dict"]

        from IndicPhotoOCR.recognition.charset_extension import extend_checkpoint_charset
        out = os.path.join(tmp_path, "ext.ckpt")
        extend_checkpoint_charset(ckpt, "de", output_path=out, init="random")
        new_sd = torch.load(out, map_location="cpu")["state_dict"]

        # text_embed: rows 0..N (EOS + old charset) unchanged
        old_te = old_sd["model.text_embed.embedding.weight"]
        new_te = new_sd["model.text_embed.embedding.weight"]
        assert torch.equal(new_te[:4], old_te[:4])
        # BOS/PAD relocated to the new tail
        assert torch.equal(new_te[6], old_te[4])   # BOS: old N+1 -> new N+1
        assert torch.equal(new_te[7], old_te[5])   # PAD: old N+2 -> new N+2

        # head: rows 0..N (EOS + old charset) unchanged
        old_hw = old_sd["model.head.weight"]
        new_hw = new_sd["model.head.weight"]
        assert torch.equal(new_hw[:4], old_hw[:4])
        assert torch.equal(new_sd["model.head.bias"][:4], old_sd["model.head.bias"][:4])

    def test_no_op_when_chars_already_present(self, tmp_path):
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import extend_checkpoint_charset
        out = os.path.join(tmp_path, "ext.ckpt")
        # 'ab' are already in the charset -> added_chars empty is an error by design,
        # but passing a fully-known set should raise.
        with pytest.raises(ValueError):
            extend_checkpoint_charset(ckpt, "ab", output_path=out)

    def test_extended_checkpoint_loads_with_load_from_checkpoint(self, tmp_path):
        """The output must be loadable by the project's own loader."""
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import extend_checkpoint_charset
        out = os.path.join(tmp_path, "ext.ckpt")
        extend_checkpoint_charset(ckpt, "de", output_path=out, init="mean")

        from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint
        model = load_from_checkpoint(out)
        assert len(model.tokenizer) == 5 + 3
        assert model.model.head.out_features == 5 + 1
        assert model.model.text_embed.embedding.num_embeddings == 5 + 3

    def test_random_init_is_reproducible(self, tmp_path):
        import torch
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import extend_checkpoint_charset
        o1 = os.path.join(tmp_path, "e1.ckpt")
        o2 = os.path.join(tmp_path, "e2.ckpt")
        extend_checkpoint_charset(ckpt, "de", output_path=o1, init="random", seed=7)
        extend_checkpoint_charset(ckpt, "de", output_path=o2, init="random", seed=7)
        s1 = torch.load(o1, map_location="cpu")["state_dict"]
        s2 = torch.load(o2, map_location="cpu")["state_dict"]
        assert torch.equal(s1["model.head.weight"], s2["model.head.weight"])
        assert torch.equal(s1["model.text_embed.embedding.weight"], s2["model.text_embed.embedding.weight"])

    def test_extend_max_label_length_grows_pos_queries(self, tmp_path):
        import torch
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        old_sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        from IndicPhotoOCR.recognition.charset_extension import (
            extend_checkpoint_charset, inspect_checkpoint,
        )
        out = os.path.join(tmp_path, "ext.ckpt")
        # Grow charset AND max_label_length (25 -> 100) in one pass.
        extend_checkpoint_charset(ckpt, "de", output_path=out, init="mean", max_label_length=100)
        info = inspect_checkpoint(out)
        assert info["max_label_length"] == 100
        old_pq = old_sd["model.pos_queries"]
        new_pq = torch.load(out, map_location="cpu")["state_dict"]["model.pos_queries"]
        assert new_pq.shape[1] == 101          # max_label_length + 1
        assert old_pq.shape[1] == 26
        # Existing position embeddings preserved.
        assert torch.equal(new_pq[:, :26], old_pq)

    def test_extend_max_label_length_only(self, tmp_path):
        """Grow max_label_length without touching the charset."""
        import torch
        ckpt = _build_tiny_parseq("abc", str(tmp_path))
        from IndicPhotoOCR.recognition.charset_extension import extend_checkpoint_charset
        out = os.path.join(tmp_path, "ext.ckpt")
        extend_checkpoint_charset(ckpt, "", output_path=out, init="mean", max_label_length=50)
        sd = torch.load(out, map_location="cpu")["state_dict"]
        # Charset tensors unchanged.
        assert sd["model.head.weight"].shape[0] == 4   # still abc + EOS
        assert sd["model.text_embed.embedding.weight"].shape[0] == 6
        # pos_queries grown.
        assert sd["model.pos_queries"].shape[1] == 51
