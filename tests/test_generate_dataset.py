"""
Tests for IndicPhotoOCR.recognition.generate_dataset.

Geometry and alignment helpers are pure-Python/numpy and always run.
The crop/split integration tests need cv2 + PIL.
"""
import json
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# polygon_bbox + merge_bboxes -- pure geometry
# ---------------------------------------------------------------------------

class TestPolygonBBox:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import polygon_bbox
        return polygon_bbox

    def test_simple_rectangle(self):
        f = self._import()
        coords = [[10, 20], [80, 20], [80, 40], [10, 40]]
        assert f(coords) == (10, 20, 80, 40)

    def test_irregular_polygon(self):
        f = self._import()
        coords = [[5, 1], [10, 3], [8, 9], [2, 7]]
        assert f(coords) == (2, 1, 10, 9)


class TestMergeBboxes:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import merge_bboxes
        return merge_bboxes

    def test_merge_two(self):
        f = self._import()
        assert f((10, 20, 50, 40), (30, 10, 80, 60)) == (10, 10, 80, 60)

    def test_merge_disjoint(self):
        f = self._import()
        assert f((0, 0, 10, 10), (50, 50, 60, 60)) == (0, 0, 60, 60)


# ---------------------------------------------------------------------------
# find_adjacent_pairs -- geometry, no PIL/cv2
# ---------------------------------------------------------------------------

class TestFindAdjacentPairs:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import find_adjacent_pairs
        return find_adjacent_pairs

    def _make_poly(self, x1, y1, x2, y2):
        return {"coordinates": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "text": "x"}

    def test_finds_adjacent_on_same_line(self):
        f = self._import()
        polys = [self._make_poly(0, 10, 50, 30), self._make_poly(60, 10, 100, 30)]
        pairs = f(polys)
        assert (0, 1) in pairs

    def test_skips_different_lines(self):
        f = self._import()
        polys = [self._make_poly(0, 10, 50, 30), self._make_poly(60, 100, 100, 120)]
        assert f(polys) == []

    def test_skips_far_apart(self):
        f = self._import()
        polys = [self._make_poly(0, 10, 10, 30), self._make_poly(500, 10, 510, 30)]
        assert f(polys) == []

    def test_empty(self):
        f = self._import()
        assert f([]) == []
        assert f([self._make_poly(0, 0, 10, 10)]) == []


# ---------------------------------------------------------------------------
# _normalize_lang
# ---------------------------------------------------------------------------

class TestNormalizeLang:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import _normalize_lang
        return _normalize_lang

    def test_fixes_typos(self):
        f = self._import()
        assert f("telegu") == "telugu"
        assert f("gujrati") == "gujarati"
        assert f("gujarti") == "gujarati"
        assert f("udru") == "urdu"

    def test_preserves_correct(self):
        f = self._import()
        assert f("marathi") == "marathi"
        assert f("Hindi") == "hindi"

    def test_empty(self):
        f = self._import()
        assert f("") == ""
        assert f(None) == ""


# ---------------------------------------------------------------------------
# _vertical_projection + _find_word_gaps -- numpy only
# ---------------------------------------------------------------------------

def _make_line_image(words, gap_width=10, word_width=30, height=20):
    """Build a synthetic line image: white bg, black rectangles = words."""
    img = np.full((height, 0), 255, dtype=np.uint8)
    for i, _ in enumerate(words):
        word = np.zeros((height, word_width), dtype=np.uint8)
        if i > 0:
            gap = np.full((height, gap_width), 255, dtype=np.uint8)
            img = np.hstack([img, gap])
        img = np.hstack([img, word])
    return np.stack([img, img, img], axis=2)  # RGB


class TestVerticalProjection:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import _vertical_projection
        return _vertical_projection

    def test_projection_shape(self):
        f = self._import()
        img = _make_line_image(["a", "b", "c"])
        proj = f(img)
        assert proj.shape[0] == img.shape[1]

    def test_gaps_are_zero(self):
        f = self._import()
        img = _make_line_image(["a", "b"], gap_width=10, word_width=30)
        proj = f(img)
        # Gap is at columns 30..39
        assert all(v == 0 for v in proj[30:40])


class TestFindWordGaps:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import _find_word_gaps
        return _find_word_gaps

    def test_finds_gap_between_two_words(self):
        f = self._import()
        proj = np.array([10, 10, 10, 0, 0, 0, 10, 10, 10])
        gaps = f(proj, min_gap_width=2)
        assert len(gaps) == 1
        assert gaps[0] == (3, 6)

    def test_ignores_thin_gaps(self):
        f = self._import()
        proj = np.array([10, 10, 0, 10, 10])
        gaps = f(proj, min_gap_width=3)
        assert gaps == []


class TestSplitLineToWords:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import split_line_to_words
        return split_line_to_words

    def test_splits_three_words(self):
        f = self._import()
        img = _make_line_image(["a", "b", "c"], gap_width=10, word_width=30)
        words = f(img)
        assert len(words) == 3
        # Each word crop should be 30px wide
        for w in words:
            assert w.shape[1] == 30

    def test_single_word(self):
        f = self._import()
        img = _make_line_image(["only"], gap_width=10, word_width=30)
        words = f(img)
        assert len(words) == 1


# ---------------------------------------------------------------------------
# _proportional_align -- numpy only
# ---------------------------------------------------------------------------

class TestProportionalAlign:
    def _import(self):
        from IndicPhotoOCR.recognition.generate_dataset import _proportional_align
        return _proportional_align

    def test_perfect_match(self):
        f = self._import()
        crops = [np.zeros((10, 20), dtype=np.uint8) for _ in range(3)]
        tokens = ["अ", "ब", "क"]
        pairs = f(crops, tokens)
        assert len(pairs) == 3
        assert [t for _, t in pairs] == ["अ", "ब", "क"]

    def test_oversplit_merges_crops(self):
        """4 crops but 3 tokens -> one pair of crops is merged."""
        f = self._import()
        crops = [np.zeros((10, 20), dtype=np.uint8) for _ in range(4)]
        tokens = ["aaa", "bbb", "ccc"]
        pairs = f(crops, tokens)
        assert len(pairs) == 3
        # The first pair should have a merged (wider) crop
        assert pairs[0][0].shape[1] >= 20  # at least one crop wide
        assert pairs[0][1] == "aaa"

    def test_undersplit_merges_tokens(self):
        """2 crops but 3 tokens -> one crop gets two tokens."""
        f = self._import()
        crops = [np.zeros((10, 30), dtype=np.uint8) for _ in range(2)]
        tokens = ["aaa", "bbb", "ccc"]
        pairs = f(crops, tokens)
        assert len(pairs) == 2
        # One of the labels should contain a space (merged tokens)
        labels = [t for _, t in pairs]
        assert any(" " in l for l in labels)

    def test_empty(self):
        f = self._import()
        assert f([], ["a"]) == []
        assert f([np.zeros((10, 10))], []) == []


# ---------------------------------------------------------------------------
# Integration: split_lines_to_dataset (needs cv2)
# ---------------------------------------------------------------------------

def _cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cv2_available(), reason="cv2 not installed")
class TestSplitLinesToDataset:
    def test_generates_jsonl_and_crops(self, tmp_path):
        import cv2
        from IndicPhotoOCR.recognition.generate_dataset import split_lines_to_dataset

        # Build a synthetic line image with 3 words
        img = _make_line_image(["a", "b", "c"], gap_width=15, word_width=40, height=30)
        img_path = str(tmp_path / "line.png")
        cv2.imwrite(img_path, img)

        jsonl_path = str(tmp_path / "input.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"image_filename": "line.png", "expected_text": "word1 word2 word3"}) + "\n")

        result = split_lines_to_dataset(jsonl_path, str(tmp_path), str(tmp_path / "out"))
        assert result["lines_processed"] == 1
        assert result["word_crops"] >= 3
        assert os.path.exists(result["jsonl_path"])

    def test_append_mode(self, tmp_path):
        import cv2
        from IndicPhotoOCR.recognition.generate_dataset import split_lines_to_dataset

        img = _make_line_image(["a", "b"], gap_width=15, word_width=40, height=30)
        img_path = str(tmp_path / "line.png")
        cv2.imwrite(img_path, img)
        jsonl_path = str(tmp_path / "input.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"image_filename": "line.png", "expected_text": "w1 w2"}) + "\n")

        out_dir = str(tmp_path / "out")
        r1 = split_lines_to_dataset(jsonl_path, str(tmp_path), out_dir)
        r2 = split_lines_to_dataset(jsonl_path, str(tmp_path), out_dir, append=True, prefix="v2_")
        # Second run should append, not overwrite
        with open(r2["jsonl_path"]) as f:
            lines = f.readlines()
        assert len(lines) == r1["word_crops"] + r2["word_crops"]
