"""Unit tests for eole/inputters/ – Step 6: data pipeline utilities."""

import os
import tempfile
import unittest

from eole.inputters.text_corpus import exfile_open, ParallelCorpus, ParallelCorpusIterator
from eole.inputters.text_utils import text_sort_key, clean_example, parse_align_idx


# ===========================================================================
# exfile_open context manager
# ===========================================================================


class TestExfileOpen(unittest.TestCase):

    def test_none_filename_yields_none_forever(self):
        """exfile_open(None) should yield None indefinitely."""
        with exfile_open(None) as f:
            values = [next(f) for _ in range(5)]
        self.assertEqual(values, [None, None, None, None, None])

    def test_real_file_yields_lines(self):
        """exfile_open with a real file should yield its lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write("line1\nline2\nline3\n")
            tmp_path = tmp.name
        try:
            with exfile_open(tmp_path, "r", encoding="utf-8") as f:
                lines = list(f)
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0].strip(), "line1")
            self.assertEqual(lines[2].strip(), "line3")
        finally:
            os.unlink(tmp_path)

    def test_empty_file_yields_no_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp_path = tmp.name
        try:
            with exfile_open(tmp_path, "r", encoding="utf-8") as f:
                lines = list(f)
            self.assertEqual(lines, [])
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# text_sort_key
# ===========================================================================


class TestTextSortKey(unittest.TestCase):

    def _make_ex(self, src_ids, tgt_ids=None):
        ex = {"src": {"src_ids": src_ids}}
        if tgt_ids is not None:
            ex["tgt"] = {"tgt_ids": tgt_ids}
        return ex

    def test_src_only_returns_int(self):
        key = text_sort_key(self._make_ex([1, 2, 3]))
        self.assertEqual(key, 3)

    def test_src_tgt_returns_tuple(self):
        key = text_sort_key(self._make_ex([1, 2], [3, 4, 5]))
        self.assertEqual(key, (2, 3))

    def test_longer_src_sorts_later(self):
        k1 = text_sort_key(self._make_ex([1]))
        k2 = text_sort_key(self._make_ex([1, 2, 3]))
        self.assertLess(k1, k2)


# ===========================================================================
# clean_example
# ===========================================================================


class TestCleanExample(unittest.TestCase):

    def test_string_src_split_into_list(self):
        ex = {"src": "hello world", "tgt": None}
        result = clean_example(ex)
        self.assertEqual(result["src"]["src"], ["hello", "world"])

    def test_list_src_unchanged(self):
        ex = {"src": ["hello", "world"], "tgt": None}
        result = clean_example(ex)
        self.assertEqual(result["src"]["src"], ["hello", "world"])

    def test_string_tgt_split(self):
        ex = {"src": "a b", "tgt": "c d e"}
        result = clean_example(ex)
        self.assertEqual(result["tgt"]["tgt"], ["c", "d", "e"])

    def test_none_tgt_stays_none(self):
        ex = {"src": "hello", "tgt": None}
        result = clean_example(ex)
        self.assertIsNone(result["tgt"])

    def test_align_list_joined(self):
        ex = {"src": "a", "tgt": "b", "align": ["0-0", "1-1"]}
        result = clean_example(ex)
        self.assertEqual(result["align"], "0-0 1-1")

    def test_sco_default_added(self):
        ex = {"src": "a"}
        result = clean_example(ex)
        self.assertEqual(result["sco"], 1)

    def test_sco_preserved_if_present(self):
        ex = {"src": "a", "sco": 0.75}
        result = clean_example(ex)
        self.assertEqual(result["sco"], 0.75)


# ===========================================================================
# parse_align_idx
# ===========================================================================


class TestParseAlignIdx(unittest.TestCase):

    def test_simple_pair(self):
        """parse_align_idx should parse 'src-tgt' format and return [src, tgt] lists."""
        from eole.inputters.text_utils import parse_align_idx
        result = parse_align_idx("0-0 1-2")
        # Returns list of [src_idx, tgt_idx] lists
        self.assertEqual(result, [[0, 0], [1, 2]])

    def test_single_pair(self):
        from eole.inputters.text_utils import parse_align_idx
        result = parse_align_idx("2-3")
        self.assertEqual(result, [[2, 3]])

    def test_empty_string_raises(self):
        """Empty string raises ValueError because it can't be split."""
        from eole.inputters.text_utils import parse_align_idx
        with self.assertRaises(ValueError):
            parse_align_idx("")


# ===========================================================================
# ParallelCorpus
# ===========================================================================


class TestParallelCorpus(unittest.TestCase):

    def _create_corpus_files(self, src_lines, tgt_lines):
        src_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".src", delete=False, encoding="utf-8")
        tgt_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".tgt", delete=False, encoding="utf-8")
        src_tmp.write("\n".join(src_lines) + "\n")
        tgt_tmp.write("\n".join(tgt_lines) + "\n")
        src_tmp.close()
        tgt_tmp.close()
        return src_tmp.name, tgt_tmp.name

    def tearDown(self):
        # Clean up any leftover temp files
        pass

    def test_load_basic(self):
        """ParallelCorpus.load should yield one example per line."""
        src_path, tgt_path = self._create_corpus_files(
            ["Hello world", "Foo bar"], ["Bonjour monde", "Foo baz"]
        )
        try:
            corpus = ParallelCorpus("test", src_path, tgt_path)
            examples = list(corpus.load())
            self.assertEqual(len(examples), 2)
            self.assertIn("src", examples[0])
            self.assertIn("tgt", examples[0])
        finally:
            os.unlink(src_path)
            os.unlink(tgt_path)

    def test_load_content(self):
        """Examples should contain correct src/tgt text."""
        src_path, tgt_path = self._create_corpus_files(
            ["Hello"], ["Bonjour"]
        )
        try:
            corpus = ParallelCorpus("test", src_path, tgt_path)
            examples = list(corpus.load())
            self.assertEqual(examples[0]["src"], "Hello")
            self.assertEqual(examples[0]["tgt"], "Bonjour")
        finally:
            os.unlink(src_path)
            os.unlink(tgt_path)

    def test_str_representation(self):
        corpus = ParallelCorpus("mycorpus", "/src/path", "/tgt/path")
        s = str(corpus)
        self.assertIn("mycorpus", s)

    def test_load_stride(self):
        """With stride=2, the (i // stride) % stride == offset logic applies."""
        src_lines = ["a", "b", "c", "d"]
        src_path, tgt_path = self._create_corpus_files(src_lines, src_lines)
        try:
            corpus = ParallelCorpus("test", src_path, tgt_path)
            # stride=2, offset=0: keeps i where (i//2) % 2 == 0, i.e. i in {0,1,4,5,...}
            examples = list(corpus.load(offset=0, stride=2))
            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["src"], "a")
            self.assertEqual(examples[1]["src"], "b")
        finally:
            os.unlink(src_path)
            os.unlink(tgt_path)

    def test_src_only_corpus(self):
        """ParallelCorpus with tgt=None should load src only."""
        src_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".src", delete=False, encoding="utf-8")
        src_tmp.write("Hello\nWorld\n")
        src_tmp.close()
        try:
            corpus = ParallelCorpus("test", src_tmp.name, None)
            examples = list(corpus.load())
            self.assertEqual(len(examples), 2)
            self.assertIsNone(examples[0]["tgt"])
        finally:
            os.unlink(src_tmp.name)


if __name__ == "__main__":
    unittest.main()
