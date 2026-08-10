"""A line budget does not bound a file that has one line.

read_file truncated at 500 lines. A minified bundle, a base64 blob or a
one-line JSON dump is a single line of arbitrary size, so the cap never fired
and the whole file went into the context window. The ranged path was worse: it
called readlines() — the entire file into memory — and returned the selected
span uncapped.

append_to_file wrote a newline before every append unless the content already
started with one, without looking at what the file ended with. Nearly every
file ends in a newline, so each call added a blank line; a large file produced
through repeated salvage continuations got one at every seam.
"""

import pytest

from tools import file_ops


@pytest.fixture(autouse=True)
def in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestReadIsBounded:
    def test_one_enormous_line_does_not_flood_the_context(self, tmp_path):
        _write(tmp_path, "bundle.min.js", "var a=1;" * 40_000)      # ~320 KB, 1 line
        out = file_ops.read_file("bundle.min.js")
        assert len(out) < file_ops.MAX_READ_CHARS + 500
        assert "Stopped at" in out

    def test_it_says_what_to_do_instead(self, tmp_path):
        _write(tmp_path, "blob.json", '{"data":"' + "A" * 200_000 + '"}')
        out = file_ops.read_file("blob.json")
        assert "grep_search" in out and "start_line" in out

    def test_a_ranged_read_is_bounded_too(self, tmp_path):
        _write(tmp_path, "big.txt", "\n".join("x" * 5_000 for _ in range(100)))
        out = file_ops.read_file("big.txt", start_line=1, end_line=100)
        assert len(out) < file_ops.MAX_READ_CHARS + 500


class TestOrdinaryReadsAreUnchanged:
    """Line numbering is covered in test_read_line_numbers.py; here only the
    bounds matter, so the gutter is stripped before asserting on content."""

    @staticmethod
    def _plain(out):
        from tools._helpers import _strip_read_line_numbers
        return _strip_read_line_numbers(out)

    def test_a_small_file_comes_back_whole(self, tmp_path):
        _write(tmp_path, "a.py", "def f():\n    return 1\n")
        assert self._plain(file_ops.read_file("a.py")) == "def f():\n    return 1\n"

    def test_the_line_cap_still_reports_the_true_total(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.get_model_size_category", lambda m: "tiny")
        _write(tmp_path, "long.py", "".join(f"line {i}\n" for i in range(1, 801)))
        out = file_ops.read_file("long.py")
        assert "800 total" in out
        assert "\tline 400\n" in out and "\tline 401\n" not in out

    def test_a_range_reports_its_span(self, tmp_path):
        _write(tmp_path, "m.txt", "".join(f"{i}\n" for i in range(1, 51)))
        out = file_ops.read_file("m.txt", start_line=10, end_line=12)
        assert out.startswith("[Lines 10-12 of 50]")
        assert out.endswith("   10\t10\n   11\t11\n   12\t12\n")

    def test_a_range_past_the_end_is_clamped(self, tmp_path):
        _write(tmp_path, "s.txt", "a\nb\n")
        out = file_ops.read_file("s.txt", start_line=1, end_line=99)
        assert "of 2]" in out and out.endswith("    1\ta\n    2\tb\n")


class TestAppendDoesNotAccumulateBlankLines:
    def test_repeated_appends_leave_no_gaps(self, tmp_path):
        """The salvage path: one file written as several continuations."""
        file_ops.write_file("out.txt", "line 1\n")
        file_ops.append_to_file("out.txt", "line 2\n")
        file_ops.append_to_file("out.txt", "line 3\n")
        assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "line 1\nline 2\nline 3\n"

    def test_a_file_without_a_final_newline_still_gets_separated(self, tmp_path):
        _write(tmp_path, "raw.txt", "no newline here")
        file_ops.append_to_file("raw.txt", "next\n")
        assert (tmp_path / "raw.txt").read_text(encoding="utf-8") == "no newline here\nnext\n"

    def test_appending_to_a_new_file_adds_no_leading_blank(self, tmp_path):
        file_ops.append_to_file("fresh.txt", "first\n")
        assert (tmp_path / "fresh.txt").read_text(encoding="utf-8") == "first\n"

    def test_content_that_starts_with_a_newline_is_left_alone(self, tmp_path):
        _write(tmp_path, "n.txt", "a\n")
        file_ops.append_to_file("n.txt", "\nb\n")
        assert (tmp_path / "n.txt").read_text(encoding="utf-8") == "a\n\nb\n"
