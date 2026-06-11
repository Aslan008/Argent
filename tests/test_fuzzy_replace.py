from unittest.mock import MagicMock

import pytest

import tools.file_ops as file_ops
from tools._helpers import _shift_indent


@pytest.fixture(autouse=True)
def quiet_side_effects(monkeypatch):
    monkeypatch.setattr(file_ops, "memory", MagicMock())
    monkeypatch.setattr(file_ops, "snapshot", MagicMock())
    monkeypatch.setattr(file_ops, "_print_diff", MagicMock())


FILE_BODY = (
    "def greet(name):\n"
    "    if name:\n"
    "        print(f'Hello {name}')\n"
    "    return name\n"
    "\n"
    "def farewell(name):\n"
    "    print('Bye')\n"
)


def make_file(tmp_path, body=FILE_BODY, name="sample.txt"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


class TestExactReplaceStillWorks:
    def test_exact_match(self, tmp_path):
        p = make_file(tmp_path)
        res = file_ops.replace_in_file(str(p), "    print('Bye')", "    print('Goodbye')")
        assert res.startswith("Successfully")
        assert "fuzzy" not in res
        assert "Goodbye" in p.read_text(encoding="utf-8")

    def test_ambiguous_exact_match_rejected(self, tmp_path):
        p = make_file(tmp_path, "x = 1\ny = 2\nx = 1\n")
        res = file_ops.replace_in_file(str(p), "x = 1", "x = 3")
        assert res.startswith("Error") and "2 times" in res


class TestFuzzyAutoApply:
    def test_wrong_indentation_is_corrected_and_applied(self, tmp_path):
        p = make_file(tmp_path)
        # Model lost all leading whitespace in both target and replacement.
        res = file_ops.replace_in_file(
            str(p),
            "if name:\nprint(f'Hello {name}')",
            "if name:\nprint(f'Hi {name}')",
        )
        assert res.startswith("Successfully")
        assert "fuzzy match" in res
        text = p.read_text(encoding="utf-8")
        assert "    if name:" in text
        assert "print(f'Hi {name}')" in text
        assert "Hello" not in text

    def test_relative_depth_preserved(self, tmp_path):
        p = make_file(tmp_path)
        res = file_ops.replace_in_file(
            str(p),
            "if name:\n    print(f'Hello {name}')",
            "if name:\n    print(f'Hey {name}')",
        )
        assert res.startswith("Successfully")
        text = p.read_text(encoding="utf-8")
        # First line re-based to 4 spaces, nested line keeps +4 relative depth.
        assert "    if name:\n        print(f'Hey {name}')" in text

    def test_trailing_newline_preserved(self, tmp_path):
        p = make_file(tmp_path)
        file_ops.replace_in_file(str(p), "def farewell(name):\nprint('Bye')", "def farewell(name):\nprint('Ciao')")
        assert p.read_text(encoding="utf-8").endswith("\n")

    def test_ambiguous_fuzzy_match_rejected(self, tmp_path):
        p = make_file(tmp_path, "    x = 1\n\n    x = 1\n")
        res = file_ops.replace_in_file(str(p), "x = 1", "x = 2")
        assert res.startswith("Error") and "ambiguous" in res

    def test_no_match_returns_hint_error(self, tmp_path):
        p = make_file(tmp_path)
        res = file_ops.replace_in_file(str(p), "this text does not exist", "anything")
        assert res.startswith("Error") and "not found" in res


class TestShiftIndent:
    def test_noop_when_equal(self):
        assert _shift_indent("  a\n  b", "  ", "  ") == "  a\n  b"

    def test_rebase_deeper(self):
        assert _shift_indent("a\n  b", "", "    ") == "    a\n      b"

    def test_rebase_shallower(self):
        assert _shift_indent("        a\n            b", "        ", "    ") == "    a\n        b"

    def test_blank_lines_stay_empty(self):
        assert _shift_indent("a\n\nb", "", "  ") == "  a\n\n  b"
