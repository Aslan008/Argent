"""A model cannot count lines, so the reader has to number them.

read_file returned raw text. Every time the model then said "line 45 of
memory_manager.py" it was estimating, and it showed: a code report produced
through Argent named memory_manager.py:45 for something on line 62,
session.py:108 for line 101, trimmer.py:145 for line 183. The offsets ran in
both directions, which is what a guess looks like — the file contents were
described correctly, only the coordinates were invented.

grep_search already returns line numbers and its citations were right. This
closes the same gap for read_file.

The predictable cost is that the model pastes the gutter back as an edit
target, so replace_in_file strips it.
"""

import pytest

from tools import file_ops
from tools._helpers import _strip_read_line_numbers


@pytest.fixture(autouse=True)
def local_only(monkeypatch, tmp_path):
    monkeypatch.setattr("tools._helpers.get_obsidian_vault", lambda: None)
    monkeypatch.chdir(tmp_path)


class TestTheGutter:
    def test_every_line_carries_its_number(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nx = 1\n", encoding="utf-8")
        out = file_ops.read_file("a.py")
        assert out == "    1\timport os\n    2\tx = 1\n"

    def test_a_range_numbers_from_the_real_position(self, tmp_path):
        (tmp_path / "m.txt").write_text("".join(f"v{i}\n" for i in range(1, 21)),
                                        encoding="utf-8")
        out = file_ops.read_file("m.txt", start_line=12, end_line=14)
        assert "   12\tv12\n   13\tv13\n   14\tv14\n" in out
        assert out.startswith("[Lines 12-14 of 20]")

    def test_the_number_is_the_file_line_not_the_output_line(self, tmp_path):
        """The whole point: what the model reads off is what start_line takes."""
        (tmp_path / "big.py").write_text("".join(f"# {i}\n" for i in range(1, 101)),
                                         encoding="utf-8")
        out = file_ops.read_file("big.py", start_line=90)
        assert "   90\t# 90\n" in out and "  100\t# 100\n" in out

    def test_a_file_without_a_trailing_newline_still_ends_cleanly(self, tmp_path):
        (tmp_path / "n.txt").write_text("only", encoding="utf-8")
        assert file_ops.read_file("n.txt") == "    1\tonly\n"


class TestTheGutterDoesNotReachTheFile:
    def test_an_edit_quoting_numbered_lines_still_matches(self, tmp_path):
        (tmp_path / "s.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        result = file_ops.replace_in_file(
            "s.py",
            "    1\tdef f():\n    2\t    return 1",     # copied from read_file
            "def f():\n    return 2",
        )
        assert "Success" in result
        assert (tmp_path / "s.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"

    def test_a_tab_separated_table_is_left_alone(self):
        """"1\\tAlice / 2\\tBob" is data with an id column. It is numbered and
        consecutive; only the missing right-alignment tells it apart, and
        stripping it would silently mangle the file."""
        table = "1\tAlice\n2\tBob\n"
        assert _strip_read_line_numbers(table) == table

    def test_non_consecutive_numbers_are_left_alone(self):
        excerpt = "   10\ta\n   40\tb\n"
        assert _strip_read_line_numbers(excerpt) == excerpt

    def test_a_single_line_is_never_treated_as_a_gutter(self):
        assert _strip_read_line_numbers("42\tvalue") == "42\tvalue"

    def test_a_mixed_block_is_left_alone(self):
        mixed = "    1\timport os\nplain line\n"
        assert _strip_read_line_numbers(mixed) == mixed

    def test_blank_source_lines_come_back_blank(self):
        """read_file numbers a blank line too, so the gutter block is
        unbroken — this is the shape it actually produces."""
        assert _strip_read_line_numbers("    1\ta\n    2\t\n    3\tc\n") == "a\n\nc\n"

    def test_an_excerpt_with_a_gap_is_left_alone(self):
        assert _strip_read_line_numbers("    1\ta\n\n    3\tc\n") == "    1\ta\n\n    3\tc\n"


class TestBudgetFollowsTheModel:
    def test_a_capable_model_reads_a_long_file_in_one_call(self, tmp_path, monkeypatch):
        """agent.py is 1529 lines. At the old 500-line cap that was four reads,
        four round trips and four chances to lose the thread."""
        monkeypatch.setattr("config.get_model_size_category", lambda m: "cloud")
        (tmp_path / "long.py").write_text("".join(f"# {i}\n" for i in range(1, 1401)),
                                          encoding="utf-8")
        out = file_ops.read_file("long.py")
        assert "exceeds" not in out
        assert " 1400\t# 1400\n" in out

    def test_a_tiny_model_is_still_protected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.get_model_size_category", lambda m: "tiny")
        (tmp_path / "long.py").write_text("".join(f"# {i}\n" for i in range(1, 1401)),
                                          encoding="utf-8")
        out = file_ops.read_file("long.py")
        assert "exceeds 400 lines (1400 total)" in out
        assert "  401\t" not in out
