"""Deep mutation-killing tests for src.agent.parser.

These tests target individual branches and edge cases so that small
mutations (off-by-one, flipped conditions, swapped operators) are
detected.  We import the module under test directly and use
``monkeypatch`` (via pytest) to stub ``get_available_tools`` where
needed.
"""
import json
import sys
import os

import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.parser import (
    extract_balanced_json,
    decode_json_escapes,
    parse_fenced_write,
    repair_json_strings,
    fix_common_json_errors,
    try_parse_json_tool,
    normalize_tool_params,
    try_recover_malformed_tool,
)


# ---------------------------------------------------------------------------
# extract_balanced_json
# ---------------------------------------------------------------------------
class TestExtractBalancedJson:
    def test_simple_object(self):
        assert extract_balanced_json("{}", 0) == "{}"

    def test_simple_object_with_content(self):
        text = '{"a": 1}'
        assert extract_balanced_json(text, 0) == text

    def test_nested_objects(self):
        text = '{"outer": {"inner": {}}}'
        assert extract_balanced_json(text, 0) == text

    def test_string_with_escaped_quote(self):
        # The escaped quote must not toggle in_string, so the closing
        # brace after it is correctly detected.
        text = '{"key": "val\\"ue"}'
        assert extract_balanced_json(text, 0) == text

    def test_no_closing_brace_returns_none(self):
        assert extract_balanced_json('{"a": 1', 0) is None

    def test_empty_string_returns_none(self):
        assert extract_balanced_json("", 0) is None

    def test_start_pos_beyond_length_returns_none(self):
        assert extract_balanced_json("{}", 5) is None

    def test_start_pos_at_non_brace_returns_none(self):
        assert extract_balanced_json("abc{}", 0) is None

    def test_braces_inside_strings_not_counted(self):
        # The literal { and } inside the string value must not affect depth.
        text = '{"k": "a{b}c"}'
        assert extract_balanced_json(text, 0) == text

    def test_start_pos_at_second_object(self):
        text = 'prefix {"a": 1} suffix'
        assert extract_balanced_json(text, 7) == '{"a": 1}'

    def test_escaped_backslash_before_quote(self):
        # \\\"  -> escaped backslash then a real quote that closes the string
        text = r'{"k": "a\\"}'
        assert extract_balanced_json(text, 0) == text

    def test_multiple_nested_levels(self):
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        assert extract_balanced_json(text, 0) == text

    def test_brace_in_string_then_unbalanced(self):
        # An opening brace inside a string should NOT increase depth,
        # so the single real closing brace balances the object.
        text = '{"k": "{"}'
        assert extract_balanced_json(text, 0) == text


# ---------------------------------------------------------------------------
# decode_json_escapes
# ---------------------------------------------------------------------------
class TestDecodeJsonEscapes:
    def test_newline(self):
        assert decode_json_escapes(r"a\nb") == "a\nb"

    def test_tab(self):
        assert decode_json_escapes(r"a\tb") == "a\tb"

    def test_carriage_return(self):
        assert decode_json_escapes(r"a\rb") == "a\rb"

    def test_unicode_uppercase_A(self):
        assert decode_json_escapes(r"\u0041") == "A"

    def test_unicode_B(self):
        assert decode_json_escapes(r"\u0042") == "B"

    def test_surrogate_pair_emoji(self):
        # 😀 is U+1F600, encoded as surrogate pair \ud83d\ude00
        assert decode_json_escapes(r"\ud83d\ude00") == "😀"

    def test_truncated_unicode_less_than_four_hex(self):
        # \u00 (only 2 hex digits) — i+6 > n so the \u branch is skipped;
        # the 'u' is kept literally (unknown escape) then '00' passes through.
        result = decode_json_escapes(r"\u00")
        assert result == "u00"

    def test_truncated_unicode_three_hex(self):
        # \u004 (only 3 hex digits) — i+6 > n so \u branch skipped;
        # 'u' kept literally then '004' passes through.
        result = decode_json_escapes(r"\u004")
        assert result == "u004"

    def test_unknown_escape_keeps_char(self):
        # \x is not a valid JSON escape; the 'x' is kept literally
        assert decode_json_escapes(r"\x") == "x"

    def test_escaped_quote(self):
        assert decode_json_escapes(r'\"') == '"'

    def test_escaped_backslash(self):
        assert decode_json_escapes(r'\\') == '\\'

    def test_escaped_slash(self):
        assert decode_json_escapes(r'\/') == '/'

    def test_escaped_backspace(self):
        assert decode_json_escapes(r'\b') == '\b'

    def test_escaped_formfeed(self):
        assert decode_json_escapes(r'\f') == '\f'

    def test_empty_string(self):
        assert decode_json_escapes("") == ""

    def test_no_escapes_passthrough(self):
        assert decode_json_escapes("hello world") == "hello world"

    def test_multiple_escapes(self):
        assert decode_json_escapes(r"\n\t\\") == "\n\t\\"

    def test_backslash_at_end_of_string(self):
        # A lone trailing backslash: i+1 >= n so it is kept literally
        assert decode_json_escapes("\\") == "\\"

    def test_unicode_with_invalid_hex(self):
        # \uGGGG — int(..., 16) raises ValueError; 'u' kept literally
        # then 'GGGG' passes through untouched.
        assert decode_json_escapes(r"\uGGGG") == "uGGGG"

    def test_surrogate_pair_with_invalid_low(self):
        # High surrogate followed by \u but invalid low surrogate
        # Should fall back to chr(high) then process the rest
        result = decode_json_escapes(r"\ud83d\u0041")
        # First: chr(0xd83d) is a lone surrogate -> repair_surrogates replaces
        # Then \u0041 -> 'A'
        assert result.endswith("A")


# ---------------------------------------------------------------------------
# parse_fenced_write
# ---------------------------------------------------------------------------
class TestParseFencedWrite:
    def test_valid_block_with_path(self):
        content = "```write_file hello.py\nprint('hi')\n```"
        result = parse_fenced_write(content)
        assert result is not None
        assert result["parsed"]["name"] == "write_file"
        assert result["parsed"]["arguments"]["file_path"] == "hello.py"
        assert result["parsed"]["arguments"]["content"] == "print('hi')"

    def test_no_path_returns_none(self):
        # No path after write_file
        content = "```write_file\nprint('hi')\n```"
        assert parse_fenced_write(content) is None

    def test_path_with_double_quotes(self):
        content = '```write_file "my file.py"\nprint(1)\n```'
        result = parse_fenced_write(content)
        assert result is not None
        assert result["parsed"]["arguments"]["file_path"] == "my file.py"

    def test_path_with_single_quotes(self):
        content = "```write_file 'my file.py'\nprint(1)\n```"
        result = parse_fenced_write(content)
        assert result is not None
        assert result["parsed"]["arguments"]["file_path"] == "my file.py"

    def test_content_with_backticks_inside(self):
        # Inner triple backticks should be part of the greedy body
        content = "```write_file test.md\n```\ninner\n```\n```"
        result = parse_fenced_write(content)
        assert result is not None
        assert "inner" in result["parsed"]["arguments"]["content"]

    def test_no_closing_fence_returns_none(self):
        content = "```write_file hello.py\nprint('hi')\n"
        assert parse_fenced_write(content) is None

    def test_no_write_file_marker_returns_none(self):
        content = "```python\nprint('hi')\n```"
        assert parse_fenced_write(content) is None

    def test_empty_content(self):
        assert parse_fenced_write("") is None

    def test_none_content(self):
        assert parse_fenced_write(None) is None  # type: ignore[arg-type]

    def test_path_cannot_contain_backticks(self):
        # The regex path group [^\n`]+? excludes backticks, so a path
        # wrapped in backticks cannot match at all.
        content = "```write_file `hello.py`\nprint(1)\n```"
        assert parse_fenced_write(content) is None

    def test_match_str_is_full_block(self):
        content = "```write_file hello.py\nprint('hi')\n```"
        result = parse_fenced_write(content)
        assert result["match_str"] == content


# ---------------------------------------------------------------------------
# repair_json_strings
# ---------------------------------------------------------------------------
class TestRepairJsonStrings:
    def test_newline_inside_string(self):
        assert repair_json_strings('{"a": "line1\nline2"}') == '{"a": "line1\\nline2"}'

    def test_tab_inside_string(self):
        assert repair_json_strings('{"a": "x\ty"}') == '{"a": "x\\ty"}'

    def test_carriage_return_inside_string(self):
        assert repair_json_strings('{"a": "x\ry"}') == '{"a": "x\\ry"}'

    def test_escaped_quote_preserved(self):
        # \" is already escaped; the backslash sets escape=True so the
        # quote does NOT toggle in_string and is kept as-is.
        result = repair_json_strings(r'{"a": "x\"y"}')
        assert result == r'{"a": "x\"y"}'

    def test_backslash_before_quote(self):
        # \\\" -> escaped backslash then a real quote that closes the string
        result = repair_json_strings(r'{"a": "x\\"}')
        assert result == r'{"a": "x\\"}'

    def test_no_strings_passthrough(self):
        # No string context: characters pass through unchanged
        assert repair_json_strings('{"a": 1}') == '{"a": 1}'

    def test_newline_outside_string_not_escaped(self):
        # A literal newline outside a string value should pass through
        result = repair_json_strings('{\n"a": 1\n}')
        assert result == '{\n"a": 1\n}'

    def test_multiple_control_chars_in_string(self):
        result = repair_json_strings('{"a": "x\n\ty"}')
        assert result == r'{"a": "x\n\ty"}'


# ---------------------------------------------------------------------------
# fix_common_json_errors
# ---------------------------------------------------------------------------
class TestFixCommonJsonErrors:
    def test_trailing_comma_before_brace(self):
        assert fix_common_json_errors('{"a": 1,}') == '{"a": 1}'

    def test_trailing_comma_before_bracket(self):
        assert fix_common_json_errors('[1, 2,]') == '[1, 2]'

    def test_trailing_comma_with_whitespace(self):
        assert fix_common_json_errors('{"a": 1 ,  }') == '{"a": 1  }' or \
               fix_common_json_errors('{"a": 1 ,  }') == '{"a": 1 }'

    def test_missing_closing_braces(self):
        assert fix_common_json_errors('{"a": {"b": 1') == '{"a": {"b": 1}}'

    def test_missing_closing_brackets(self):
        assert fix_common_json_errors('[1, [2, 3') == '[1, [2, 3]]'

    def test_balanced_json_passthrough(self):
        assert fix_common_json_errors('{"a": 1}') == '{"a": 1}'

    def test_nested_with_missing_closers(self):
        # {"a": {"b": [1, 2  -> missing 1 ] and 2 }
        result = fix_common_json_errors('{"a": {"b": [1, 2')
        assert result.endswith(']}}')
        assert result.count('}') == 2
        assert result.count(']') == 1

    def test_strips_whitespace(self):
        assert fix_common_json_errors('  {"a": 1}  ') == '{"a": 1}'

    def test_both_trailing_comma_and_missing_brace(self):
        result = fix_common_json_errors('{"a": 1, "b": [1, 2,')
        # trailing comma before ] removed, then ] added
        assert ']' in result
        assert '}' in result


# ---------------------------------------------------------------------------
# try_parse_json_tool
# ---------------------------------------------------------------------------
class TestTryParseJsonTool:
    def test_name_arguments_format(self):
        result = try_parse_json_tool('{"name": "read_file", "arguments": {"file_path": "a.txt"}}')
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        assert result["parsed"]["arguments"]["file_path"] == "a.txt"

    def test_function_arguments_format(self):
        result = try_parse_json_tool('{"function": "read_file", "arguments": {"file_path": "a.txt"}}')
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        assert result["parsed"]["arguments"]["file_path"] == "a.txt"

    def test_shorthand_format(self, monkeypatch):
        monkeypatch.setattr("src.agent.parser.get_available_tools",
                            lambda: {"read_file": {}})
        result = try_parse_json_tool('{"read_file": {"file_path": "a.txt"}}')
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        assert result["parsed"]["arguments"]["file_path"] == "a.txt"

    def test_non_dict_returns_none(self):
        assert try_parse_json_tool('[1, 2, 3]') is None

    def test_non_dict_string_returns_none(self):
        assert try_parse_json_tool('"hello"') is None

    def test_no_name_or_function_returns_none(self, monkeypatch):
        monkeypatch.setattr("src.agent.parser.get_available_tools",
                            lambda: {})
        assert try_parse_json_tool('{"arguments": {}}') is None

    def test_no_arguments_returns_none(self):
        assert try_parse_json_tool('{"name": "read_file"}') is None

    def test_name_not_string_returns_none(self):
        assert try_parse_json_tool('{"name": 123, "arguments": {}}') is None

    def test_invalid_json_returns_none(self):
        assert try_parse_json_tool('not json at all') is None

    def test_name_with_empty_arguments(self):
        result = try_parse_json_tool('{"name": "read_file", "arguments": {}}')
        assert result is not None
        assert result["parsed"]["arguments"] == {}

    def test_function_canonicalized_to_name(self):
        result = try_parse_json_tool('{"function": "write_file", "arguments": {"file_path": "x.py", "content": "hi"}}')
        assert result["parsed"]["name"] == "write_file"

    def test_shorthand_not_in_tools_returns_none(self, monkeypatch):
        monkeypatch.setattr("src.agent.parser.get_available_tools",
                            lambda: {"read_file": {}})
        assert try_parse_json_tool('{"nonexistent_tool": {"a": 1}}') is None

    def test_shorthand_value_not_dict_returns_none(self, monkeypatch):
        monkeypatch.setattr("src.agent.parser.get_available_tools",
                            lambda: {"read_file": {}})
        assert try_parse_json_tool('{"read_file": "not a dict"}') is None


# ---------------------------------------------------------------------------
# normalize_tool_params
# ---------------------------------------------------------------------------
class TestNormalizeToolParams:
    def test_write_file_filename_alias(self):
        parsed = {"name": "write_file", "arguments": {"filename": "a.py"}}
        result = normalize_tool_params(parsed)
        assert result["arguments"]["file_path"] == "a.py"
        assert "filename" not in result["arguments"]

    def test_write_file_path_alias(self):
        parsed = {"name": "write_file", "arguments": {"path": "a.py"}}
        result = normalize_tool_params(parsed)
        assert result["arguments"]["file_path"] == "a.py"

    def test_write_file_file_alias(self):
        parsed = {"name": "write_file", "arguments": {"file": "a.py"}}
        result = normalize_tool_params(parsed)
        assert result["arguments"]["file_path"] == "a.py"

    def test_write_file_filepath_alias(self):
        parsed = {"name": "write_file", "arguments": {"filepath": "a.py"}}
        result = normalize_tool_params(parsed)
        assert result["arguments"]["file_path"] == "a.py"

    def test_unknown_tool_passthrough(self):
        parsed = {"name": "unknown_tool", "arguments": {"xyz": 1}}
        result = normalize_tool_params(parsed)
        assert result["arguments"] == {"xyz": 1}

    def test_no_aliases_for_tool(self):
        # A tool with no alias mapping should pass arguments through unchanged
        parsed = {"name": "some_random_tool", "arguments": {"a": 1, "b": 2}}
        result = normalize_tool_params(parsed)
        assert result["arguments"] == {"a": 1, "b": 2}

    def test_mixed_alias_and_canonical(self):
        parsed = {"name": "write_file", "arguments": {"filename": "a.py", "content": "hi"}}
        result = normalize_tool_params(parsed)
        assert result["arguments"]["file_path"] == "a.py"
        assert result["arguments"]["content"] == "hi"

    def test_create_directory_path_alias(self):
        parsed = {"name": "create_directory", "arguments": {"path": "/tmp/x"}}
        result = normalize_tool_params(parsed)
        assert result["arguments"]["dir_path"] == "/tmp/x"

    def test_empty_name(self):
        parsed = {"name": "", "arguments": {"a": 1}}
        result = normalize_tool_params(parsed)
        assert result["arguments"] == {"a": 1}


# ---------------------------------------------------------------------------
# try_recover_malformed_tool
# ---------------------------------------------------------------------------
class TestTryRecoverMalformedTool:
    def test_write_file_with_file_path_and_content(self):
        raw = '{"name": "write_file", "arguments": {"file_path": "z.txt", "content": "hello\\nworld"}}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result is not None
        assert result["parsed"]["name"] == "write_file"
        assert result["parsed"]["arguments"]["file_path"] == "z.txt"
        assert "hello" in result["parsed"]["arguments"]["content"]

    def test_non_content_tool_returns_none(self):
        # read_file is not in CONTENT_TOOLS
        raw = '{"name": "read_file", "arguments": {"file_path": "z.txt"}}'
        assert try_recover_malformed_tool(raw, "read_file") is None

    def test_no_file_path_match_returns_none(self):
        raw = '{"content": "hello"}'
        assert try_recover_malformed_tool(raw, "write_file") is None

    def test_no_content_match_returns_none(self):
        raw = '{"file_path": "z.txt"}'
        assert try_recover_malformed_tool(raw, "write_file") is None

    def test_replace_in_file_recovery(self):
        raw = '{"name": "replace_in_file", "arguments": {"file_path": "z.txt", "content": "data"}}'
        result = try_recover_malformed_tool(raw, "replace_in_file")
        assert result is not None
        assert result["parsed"]["arguments"]["file_path"] == "z.txt"

    def test_replace_python_function_recovery(self):
        raw = '{"name": "replace_python_function", "arguments": {"file_path": "z.py", "content": "def f(): pass"}}'
        result = try_recover_malformed_tool(raw, "replace_python_function")
        assert result is not None

    def test_filename_alias_in_recovery(self):
        raw = '{"name": "write_file", "arguments": {"filename": "z.txt", "content": "hello"}}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result is not None
        assert result["parsed"]["arguments"]["file_path"] == "z.txt"

    def test_content_with_trailing_braces_stripped(self):
        raw = '{"file_path": "z.txt", "content": "hello"}}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result is not None
        content = result["parsed"]["arguments"]["content"]
        assert "}" not in content
        assert "hello" in content

    def test_match_str_is_full_content(self):
        raw = '{"file_path": "z.txt", "content": "hello"}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result["match_str"] == raw

    def test_filepath_alias_in_recovery(self):
        raw = '{"filepath": "z.txt", "content": "hello"}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result is not None
        assert result["parsed"]["arguments"]["file_path"] == "z.txt"

    def test_path_alias_in_recovery(self):
        raw = '{"path": "z.txt", "content": "hello"}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result is not None
        assert result["parsed"]["arguments"]["file_path"] == "z.txt"

    def test_content_with_newline_escape_decoded(self):
        raw = '{"file_path": "z.txt", "content": "line1\\nline2"}'
        result = try_recover_malformed_tool(raw, "write_file")
        assert result is not None
        assert "\n" in result["parsed"]["arguments"]["content"]