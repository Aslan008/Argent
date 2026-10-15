import pytest
from pathlib import Path
from tools.search_ops import _normalize_grep_pattern, grep_search, _python_grep_search
from src.agent.parser import normalize_tool_params, _PARAM_ALIASES


class TestNormalizeGrepPattern:
    """Test suite for _normalize_grep_pattern: quote stripping and BRE pipe fix."""

    def test_strip_outer_double_quotes(self):
        assert _normalize_grep_pattern('"Jev|jev"') == "Jev|jev"
        assert _normalize_grep_pattern('"hello world"') == "hello world"

    def test_strip_outer_single_quotes(self):
        assert _normalize_grep_pattern("'class ArgentAgent'") == "class ArgentAgent"
        assert _normalize_grep_pattern("'foo'") == "foo"

    def test_preserve_multiple_quoted_segments(self):
        # Multiple quoted strings in one pattern should not be stripped into invalid syntax
        assert _normalize_grep_pattern('"foo" "bar"') == '"foo" "bar"'

    def test_strip_outer_quotes_with_escaped_quotes_inside(self):
        assert _normalize_grep_pattern(r'"foo \"bar\" baz"') == r'foo \"bar\" baz'

    def test_bre_pipe_conversion(self):
        # Model sends \| intending alternation (like in unix grep BRE)
        assert _normalize_grep_pattern(r"Jev\|jev") == "Jev|jev"
        assert _normalize_grep_pattern(r"foo\|bar\|baz") == "foo|bar|baz"
        assert _normalize_grep_pattern(r"def (get_model\|set_model)") == "def (get_model|set_model)"

    def test_bre_pipe_preserves_character_class(self):
        # Inside [\|], the pipe is inside character class and should remain as is
        assert _normalize_grep_pattern(r"[\|]") == r"[\|]"

    def test_bre_pipe_preserves_escaped_backslash(self):
        # \\| represents a literal backslash followed by alternation
        assert _normalize_grep_pattern(r"foo\\|bar") == r"foo\\|bar"

    def test_combined_quotes_and_bre_pipe(self):
        # Model wraps in quotes AND uses \|
        assert _normalize_grep_pattern(r'"Jev\|jev"') == "Jev|jev"
        assert _normalize_grep_pattern(r"'foo\|bar'") == "foo|bar"

    def test_empty_and_whitespace(self):
        assert _normalize_grep_pattern("") == ""
        assert _normalize_grep_pattern(None) == ""
        assert _normalize_grep_pattern("   ") == ""
        assert _normalize_grep_pattern('"') == '"'
        assert _normalize_grep_pattern("''") == ""
        assert _normalize_grep_pattern('""') == ""


class TestGrepSearchQueryAlias:
    """Test suite for query parameter support in grep_search and _python_grep_search."""

    def test_grep_search_with_query_param(self, tmp_path):
        test_file = tmp_path / "sample.py"
        test_file.write_text("class MyAgent:\n    pass\n", encoding="utf-8")

        result = grep_search(directory=str(tmp_path), query="MyAgent")
        assert "Found" in result
        assert "MyAgent" in result

    def test_python_grep_search_with_query_param(self, tmp_path):
        test_file = tmp_path / "sample.py"
        test_file.write_text("class MyAgent:\n    pass\n", encoding="utf-8")

        result = _python_grep_search(directory=str(tmp_path), query="MyAgent")
        assert "Found" in result
        assert "MyAgent" in result

    def test_grep_search_with_bre_pipe_query(self, tmp_path):
        test_file = tmp_path / "sample.py"
        test_file.write_text("alpha = 1\nbeta = 2\n", encoding="utf-8")

        result = grep_search(directory=str(tmp_path), query=r"alpha\|beta")
        assert "Found" in result
        assert "alpha = 1" in result
        assert "beta = 2" in result

    def test_grep_search_with_quoted_query(self, tmp_path):
        test_file = tmp_path / "sample.py"
        test_file.write_text("alpha = 1\n", encoding="utf-8")

        result = grep_search(directory=str(tmp_path), query='"alpha"')
        assert "Found" in result
        assert "alpha = 1" in result


class TestParserGrepAliases:
    """Test suite for parser normalization of grep_search aliases."""

    def test_normalize_query_to_pattern(self):
        parsed = {
            "name": "grep_search",
            "arguments": {
                "directory": "src",
                "query": "Jev|jev"
            }
        }
        res = normalize_tool_params(parsed)
        assert res["arguments"]["pattern"] == "Jev|jev"
        assert "query" not in res["arguments"]

    def test_normalize_search_path_to_directory(self):
        parsed = {
            "name": "grep_search",
            "arguments": {
                "search_path": "tools",
                "pattern": "def "
            }
        }
        res = normalize_tool_params(parsed)
        assert res["arguments"]["directory"] == "tools"
        assert "search_path" not in res["arguments"]

    def test_normalize_query_and_search_path_together(self):
        parsed = {
            "name": "grep_search",
            "arguments": {
                "search_path": "tools",
                "query": "grep_search"
            }
        }
        res = normalize_tool_params(parsed)
        assert res["arguments"]["directory"] == "tools"
        assert res["arguments"]["pattern"] == "grep_search"
