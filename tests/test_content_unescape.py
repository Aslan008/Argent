"""Conditional unescape: never corrupt a literal \\n that lives inside real code."""

from tools._helpers import _maybe_unescape_content
from tools.file_ops import write_file


def test_real_newlines_preserve_literal_backslash_n():
    # Multi-line source with a literal \n inside a string must stay byte-for-byte.
    src = 'using UnityEngine;\npublic class L {\n    void S(){ Debug.Log("a\\nb"); }\n}'
    assert _maybe_unescape_content(src) == src


def test_single_line_escaped_payload_is_decoded():
    # No real line breaks + literal markers = leaked escaping -> decode it.
    assert _maybe_unescape_content("line1\\nline2\\tend") == "line1\nline2\tend"


def test_double_backslash_collapsed_only_within_escaped_payload():
    # Collapsing \\ happens only alongside \n/\t decoding (a leaked payload)...
    assert _maybe_unescape_content("a\\nC:\\\\x") == "a\nC:\\x"
    # ...a lone double backslash (a real Windows path) is left untouched.
    assert _maybe_unescape_content("C:\\\\path\\\\x") == "C:\\\\path\\\\x"


def test_empty_and_plain_untouched():
    assert _maybe_unescape_content("") == ""
    assert _maybe_unescape_content("just text") == "just text"


def test_write_file_keeps_literal_newline_in_string(tmp_path):
    f = tmp_path / "L.cs"
    body = 'public class L {\n    void S(){ System.Console.Write("a\\nb"); }\n}'
    write_file(str(f), body)
    on_disk = f.read_text(encoding="utf-8")
    assert '"a\\nb"' in on_disk                    # the literal \n survived
    assert on_disk.count("\n") == body.count("\n")  # only the real newlines exist
