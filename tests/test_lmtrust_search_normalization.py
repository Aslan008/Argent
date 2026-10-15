import pytest
from hypothesis import given, strategies as st
from tools.search_ops import _normalize_grep_pattern, grep_search, _python_grep_search
from src.agent.parser import normalize_tool_params, _PARAM_ALIASES


# ===========================================================================
# L0: Smoke — Basic execution sanity
# ===========================================================================
class TestSmoke:
    def test_normalize_basic_string(self):
        assert _normalize_grep_pattern("sample") == "sample"

    def test_grep_search_smoke(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello world')\n", encoding="utf-8")
        out = grep_search(str(tmp_path), "hello")
        assert "Found" in out

    def test_parser_smoke(self):
        parsed = {"name": "grep_search", "arguments": {"query": "test"}}
        res = normalize_tool_params(parsed)
        assert res["arguments"]["pattern"] == "test"


# ===========================================================================
# L1: Contract (D7 API Contract)
# ===========================================================================
class TestContract:
    def test_normalize_returns_str_for_none(self):
        res = _normalize_grep_pattern(None)
        assert isinstance(res, str)
        assert res == ""

    def test_normalize_returns_str_for_non_str(self):
        res = _normalize_grep_pattern(12345)
        assert isinstance(res, str)
        assert res == "12345"

    def test_grep_search_always_returns_str(self, tmp_path):
        res = grep_search(str(tmp_path), pattern="anything")
        assert isinstance(res, str)

    def test_parser_returns_same_dict_identity(self):
        data = {"name": "grep_search", "arguments": {"query": "x"}}
        res = normalize_tool_params(data)
        assert res is data


# ===========================================================================
# L2: Boundary (8 Edge Conditions: empty, single-char, escapes, bounds)
# ===========================================================================
class TestBoundary:
    # 1. Empty & whitespace
    def test_empty_string(self):
        assert _normalize_grep_pattern("") == ""
        assert _normalize_grep_pattern("   ") == ""

    # 2. Single quote character
    def test_single_quote_char(self):
        assert _normalize_grep_pattern('"') == '"'
        assert _normalize_grep_pattern("'") == "'"

    # 3. Empty pair of quotes
    def test_empty_pair_quotes(self):
        assert _normalize_grep_pattern('""') == ""
        assert _normalize_grep_pattern("''") == ""

    # 4. Trailing backslash
    def test_trailing_backslash(self):
        assert _normalize_grep_pattern(r"path\to\\") == r"path\to\\"

    # 5. Escaped closing quote — should NOT be stripped as a delimiter
    def test_escaped_closing_quote_not_stripped(self):
        # r'"foo\"' ends with an escaped quote, so the outer quotes are not a pair
        assert _normalize_grep_pattern(r'"foo\"') == r'"foo\"'
        assert _normalize_grep_pattern("'foo\\'") == "'foo\\'"

    # 6. Standalone \| represents literal pipe, NOT alternation
    def test_standalone_bre_pipe_preserved(self):
        assert _normalize_grep_pattern(r"\|") == r"\|"
        assert _normalize_grep_pattern(r" \| ") == r" \| "

    # 7. Escaped backslash before open bracket r"foo\\[bar\|baz]"
    def test_escaped_backslash_before_bracket(self):
        # \\ is a literal backslash, so [ is an unescaped bracket
        # Inside [bar\|baz], \| should NOT be converted to alternation |
        pattern = r"foo\\[bar\|baz]"
        assert _normalize_grep_pattern(pattern) == r"foo\\[bar\|baz]"

    # 8. Unicode, Cyrillic, and emojis
    def test_unicode_and_cyrillic(self):
        assert _normalize_grep_pattern('"Функция|Класс"') == "Функция|Класс"
        assert _normalize_grep_pattern(r'"Поиск\|Замена"') == "Поиск|Замена"
        assert _normalize_grep_pattern('"🚀|⚡"') == "🚀|⚡"


# ===========================================================================
# L3: Properties (Hypothesis — Invariants)
# ===========================================================================
class TestProperties:
    @given(st.text(max_size=200))
    def test_idempotence(self, text):
        """Invariant: Normalization must be idempotent for all text."""
        first = _normalize_grep_pattern(text)
        second = _normalize_grep_pattern(first)
        assert first == second

    @given(st.text(max_size=200))
    def test_always_returns_str(self, text):
        """Invariant: _normalize_grep_pattern never raises and always returns str."""
        result = _normalize_grep_pattern(text)
        assert isinstance(result, str)


# ===========================================================================
# L4: Adversarial (Breaking assumptions, invalid types, conflicts)
# ===========================================================================
class TestAdversarial:
    def test_non_string_types_to_normalize(self):
        assert _normalize_grep_pattern(42) == "42"
        assert _normalize_grep_pattern(True) == "True"
        assert _normalize_grep_pattern(3.14) == "3.14"

    def test_parser_non_dict_parsed(self):
        assert normalize_tool_params(None) is None
        assert normalize_tool_params("not_a_dict") == "not_a_dict"
        assert normalize_tool_params([1, 2, 3]) == [1, 2, 3]

    def test_parser_non_dict_arguments(self):
        assert normalize_tool_params({"name": "grep_search", "arguments": None}) == {
            "name": "grep_search", "arguments": None
        }
        assert normalize_tool_params({"name": "grep_search", "arguments": "raw_str"}) == {
            "name": "grep_search", "arguments": "raw_str"
        }

    def test_parser_canonical_precedence_over_alias(self):
        # When both pattern and query are provided, pattern (canonical) takes priority!
        parsed = {
            "name": "grep_search",
            "arguments": {
                "pattern": "CANONICAL_PATTERN",
                "query": "ALIAS_QUERY",
            }
        }
        res = normalize_tool_params(parsed)
        assert res["arguments"]["pattern"] == "CANONICAL_PATTERN"

    def test_parser_empty_canonical_filled_by_alias(self):
        # When pattern is empty and query is provided, query fills pattern
        parsed = {
            "name": "grep_search",
            "arguments": {
                "pattern": "",
                "query": "FILL_ME",
            }
        }
        res = normalize_tool_params(parsed)
        assert res["arguments"]["pattern"] == "FILL_ME"

    def test_bre_pipe_parity_with_quad_backslashes(self):
        # \\\\| = 4 backslashes (2 literal backslashes) + |
        # \\\\\| = 4 backslashes + \| -> 2 literal backslashes + alternation |
        p = r"foo\\\\\|bar"
        assert _normalize_grep_pattern(p) == r"foo\\\\|bar"


# ===========================================================================
# L4b: Fallback Paths (ripgrep fallback to Python grep)
# ===========================================================================
class TestFallback:
    def test_python_fallback_handles_query_and_normalization(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("def find_me():\n    pass\n", encoding="utf-8")

        # Query wrapped in quotes
        res = _python_grep_search(directory=str(tmp_path), query='"find_me"')
        assert "Found" in res
        assert "find_me" in res

    def test_python_fallback_handles_bre_pipes(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("item_apple = 1\nitem_banana = 2\n", encoding="utf-8")

        res = _python_grep_search(directory=str(tmp_path), query=r"item_apple\|item_banana")
        assert "Found" in res
        assert "item_apple" in res
        assert "item_banana" in res


# ===========================================================================
# L9: Negative Space (What should NOT happen)
# ===========================================================================
class TestNegativeSpace:
    def test_does_not_strip_multiple_quoted_tokens(self):
        # '"foo" "bar"' must not become 'foo" "bar'
        assert _normalize_grep_pattern('"foo" "bar"') == '"foo" "bar"'

    def test_does_not_convert_pipe_inside_character_classes(self):
        # [a\|b] must not become [a|b]
        assert _normalize_grep_pattern(r"[a\|b]") == r"[a\|b]"
        assert _normalize_grep_pattern(r"prefix_[a-z\|0-9]_suffix") == r"prefix_[a-z\|0-9]_suffix"

    def test_does_not_mutate_non_aliased_tools(self):
        parsed = {"name": "custom_plugin_tool", "arguments": {"query": "leave_me"}}
        res = normalize_tool_params(parsed)
        assert res["arguments"]["query"] == "leave_me"
