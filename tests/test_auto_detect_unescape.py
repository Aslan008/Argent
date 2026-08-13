"""Auto-detect: _maybe_unescape_content must not corrupt literal \\n inside
string literals (e.g. print("a\\nb")), but must still decode leaked JSON
escaping in plain text (e.g. "line1\\nline2")."""

from tools._helpers import _maybe_unescape_content


class TestAutoDetectQuotes:
    def test_literal_backslash_n_in_double_quoted_string_preserved(self):
        # The \n inside the string literal must stay as literal backslash+n.
        src = 'print("hello\\nworld")'
        assert _maybe_unescape_content(src) == src

    def test_literal_backslash_n_in_single_quoted_string_preserved(self):
        src = "print('hello\\nworld')"
        assert _maybe_unescape_content(src) == src

    def test_literal_backslash_n_in_fstring_preserved(self):
        src = 'parts.append(f"text\\n{var}")'
        assert _maybe_unescape_content(src) == src

    def test_plain_text_with_backslash_n_still_decoded(self):
        # No quotes -> auto-detect sees all-even quotes -> unescape is safe.
        assert _maybe_unescape_content("line1\\nline2") == "line1\nline2"

    def test_plain_text_with_backslash_n_and_tab_still_decoded(self):
        assert _maybe_unescape_content("a\\nb\\tc") == "a\nb\tc"

    def test_code_with_balanced_quotes_per_line_still_decoded(self):
        # Each line after unescape would have even quotes -> safe to unescape.
        src = 'x = "hello"\\ny = "world"'
        assert _maybe_unescape_content(src) == 'x = "hello"\ny = "world"'

    def test_mixed_code_and_string_literal_preserved(self):
        # The \n inside the string breaks the quote balance -> don't unescape.
        src = 'x = "a\\nb" + "c"'
        assert _maybe_unescape_content(src) == src

    def test_windows_path_with_backslash_n_in_string_preserved(self):
        src = 'path = "C:\\\\folder\\\\file.txt"'
        assert _maybe_unescape_content(src) == src