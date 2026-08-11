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
    parse_raw_tool_call,
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
# ---------------------------------------------------------------------------
# Surrogate-pair edge cases for decode_json_escapes
# ---------------------------------------------------------------------------
class TestDecodeJsonEscapesSurrogatePairs:
    """Tests targeting the surrogate-pair combination logic (lines 145-149).

    The mutations ADD_TO_SUB on ``s[i + 6:i + 8]`` and ``s[i + 8:i + 12]``
    and SUB_TO_ADD on ``(cp - 0xD800)`` all silently disable the in-line
    surrogate-pair combination.  ``repair_surrogates`` at the end of the
    function recombines adjacent lone surrogates, so these mutations are
    near-equivalent — but we still assert exact output strings so that any
    subtle difference (e.g. non-adjacent surrogates, wrong codepoint) is
    caught.
    """

    def test_surrogate_pair_d83d_de00_exact(self):
        # U+1F600 GRINNING FACE — the canonical surrogate pair test.
        assert decode_json_escapes(r"\ud83d\ude00") == "\U0001F600"

    def test_surrogate_pair_d83d_dc4d_exact(self):
        # U+1F44D THUMBS UP SIGN
        assert decode_json_escapes(r"\ud83d\udc4d") == "\U0001F44D"

    def test_surrogate_pair_d83c_df89_exact(self):
        # U+1F389 PARTY POPPER — different high surrogate base (0xD83C)
        assert decode_json_escapes(r"\ud83c\udf89") == "\U0001F389"

    def test_surrogate_pair_with_surrounding_text(self):
        # Text before and after the surrogate pair must be preserved exactly.
        assert decode_json_escapes(r"hello\ud83d\ude00world") == "hello\U0001F600world"

    def test_two_consecutive_surrogate_pairs(self):
        # Two emoji back-to-back — both must be decoded correctly.
        assert decode_json_escapes(r"\ud83d\ude00\ud83d\udc4d") == "\U0001F600\U0001F44D"

    def test_surrogate_pair_then_regular_unicode(self):
        # A surrogate pair followed by a regular \u escape.
        assert decode_json_escapes(r"\ud83d\ude00\u0041") == "\U0001F600A"

    def test_regular_unicode_then_surrogate_pair(self):
        # A regular \u escape followed by a surrogate pair.
        assert decode_json_escapes(r"\u0041\ud83d\ude00") == "A\U0001F600"

    def test_surrogate_pair_in_longer_string_at_offset(self):
        # The surrogate pair starts at a non-zero offset (i > 0), which
        # changes the mutated slice indices ``s[i-6:i-8]`` / ``s[i-8:i-12]``.
        prefix = "x" * 20
        assert decode_json_escapes(prefix + r"\ud83d\ude00") == prefix + "\U0001F600"


# ---------------------------------------------------------------------------
# parse_raw_tool_call — mutations in the markdown / brace extraction paths
# ---------------------------------------------------------------------------
class TestParseRawToolCallMarkdown:
    """Tests for ``parse_raw_tool_call`` that kill ADD_TO_SUB and SUB_TO_ADD
    mutations in the markdown code-block extraction and bare-brace fallback
    paths (lines 222-237)."""

    def test_markdown_block_with_backticks_inside_json_match_str(self):
        """Kill mutation 5 (inner_start + len → inner_start - len) and
        mutation 7 (closing + 3 → closing - 3).

        The JSON value contains a literal ````` ```` triple-backtick inside
        a string.  When ``end_pos`` is computed correctly (``inner_start +
        len(json_candidate)``), ``clean.find('```', end_pos)`` skips past
        the inner backticks and finds the *closing* fence.  When the
        mutation flips ``+`` to ``-``, ``end_pos`` becomes negative, Python
        adjusts it to a position *before* the inner backticks, and
        ``find`` returns the position of the inner ````` ```` — producing a
        truncated ``match_str``.

        Similarly, ``closing + 3`` → ``closing - 3`` truncates the
        ``match_str`` by 6 characters, dropping the closing fence.
        """
        content = '```json\n{"name":"read_file","arguments":{"content":"```"}}\n```'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        assert result["parsed"]["arguments"]["content"] == "```"
        # The match_str must be the ENTIRE input — including the closing ``` fence.
        assert result["match_str"] == content

    def test_markdown_block_without_closing_fence_match_str(self):
        """Kill mutation 6 (``closing != -1`` → ``closing != 1``).

        When there is no closing ````` ```` fence, ``closing = -1``.
        Original: ``if closing != -1`` → False → else branch →
        ``match_str = clean[md_match.start():end_pos]``.
        Mutated:  ``if closing != 1``  → True  → if branch →
        ``match_str = clean[md_match.start():closing + 3]`` =
        ``clean[md_match.start():2]`` — a 2-char string ``"``"``.

        Asserting the exact ``match_str`` distinguishes the two.
        """
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        assert result["parsed"]["arguments"]["file_path"] == "a.txt"
        # With no closing fence, match_str is the prefix + JSON (no trailing ```).
        assert result["match_str"] == content

    def test_markdown_block_simple_match_str(self):
        """A simple markdown JSON block — match_str must include the closing fence."""
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}\n```'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["match_str"] == content

    def test_markdown_block_no_json_prefix_match_str(self):
        """Markdown block without 'json' label — match_str must still be exact."""
        content = '```\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}\n```'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["match_str"] == content


class TestParseRawToolCallBraceFallback:
    """Tests for the bare-brace fallback path (line 237: ``brace_pos != -1``)."""

    def test_brace_at_position_one(self, monkeypatch):
        """Kill mutation 9 (``brace_pos != -1`` → ``brace_pos != 1``).

        When the first ``{`` is at position 1, ``brace_pos = 1``.
        Original: ``if 1 != -1`` → True  → extracts JSON from pos 1 → returns result.
        Mutated:  ``if 1 != 1``  → False → skips to LAST RESORT → returns None
        (because no tool name matches in the last-resort regex path when
        ``get_available_tools`` is empty).

        Asserting ``result is not None`` and the exact ``match_str``
        distinguishes the two.
        """
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = 'A{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        assert result["parsed"]["arguments"]["file_path"] == "a.txt"
        assert result["match_str"] == '{"name": "read_file", "arguments": {"file_path": "a.txt"}}'

    def test_brace_at_position_zero(self, monkeypatch):
        """When ``{`` is at position 0, ``brace_pos = 0`` — both original
        (``0 != -1`` → True) and mutated (``0 != 1`` → True) enter the if
        block, so this test alone does NOT kill the mutation.  It is here
        for coverage of the normal path."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = '{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["parsed"]["name"] == "read_file"

    def test_no_brace_returns_none(self, monkeypatch):
        """When there is no ``{`` at all, ``brace_pos = -1``.
        Original: ``if -1 != -1`` → False → skips.
        Mutated:  ``if -1 != 1``  → True  → calls ``extract_balanced_json(clean, -1)``
        which checks ``text[-1] != '{'`` → True → returns None.
        Both paths then fall through to LAST RESORT, which also returns None
        when ``get_available_tools`` is empty.  So this test does NOT kill
        the mutation but documents the no-brace behaviour."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = "no json here at all"
        result = parse_raw_tool_call(content)
        assert result is None
# ─── Surrogate pair handling (kills ADD_TO_SUB/SUB_TO_ADD in decode_json_escapes) ──

class TestSurrogatePairExact:
    """Test that decode_json_escapes correctly combines UTF-16 surrogate pairs.

    Kills: ADD_TO_SUB @ pos 4335,4341 (i+6:i+8 → i-6:i-8),
           ADD_TO_SUB @ pos 4410,4416 (i+8:i+12 → i-8:i-12),
           SUB_TO_ADD @ pos 4528 (cp-0xD800 → cp+0xD800).
    """

    def test_emoji_surrogate_pair(self):
        """\\uD83D\\uDE00 should decode to U+1F600 (😀)."""
        result = decode_json_escapes('hello \\uD83D\\uDE00 world')
        assert result == 'hello \U0001F600 world'

    def test_emoji_exact_codepoint(self):
        """The combined codepoint must be exactly 0x1F600."""
        result = decode_json_escapes('\\uD83D\\uDE00')
        assert len(result) == 1
        assert ord(result) == 0x1F600

    def test_emoji_not_lone_high_surrogate(self):
        """Mutant (cp+0xD800) would overflow chr() → ValueError → lone surrogate.
        Original produces a single emoji character, not a lone high surrogate."""
        result = decode_json_escapes('\\uD83D\\uDE00')
        assert ord(result[0]) >= 0x10000  # supplementary plane, not BMP surrogate

    def test_multiple_surrogate_pairs(self):
        """Two emoji in sequence: \\uD83D\\uDE00\\uD83D\\uDC4D (😀👍)."""
        result = decode_json_escapes('\\uD83D\\uDE00\\uD83D\\uDC4D')
        assert result == '\U0001F600\U0001F44D'

    def test_surrogate_pair_preserves_surrounding_text(self):
        """Text before and after surrogate pair is preserved exactly."""
        result = decode_json_escapes('A\\uD83D\\uDE00B')
        assert result == 'A\U0001F600B'
        assert len(result) == 3  # A + emoji + B

    def test_high_surrogate_without_low(self):
        """A high surrogate not followed by \\u low surrogate → repair_surrogates
        replaces the lone surrogate with U+FFFD."""
        result = decode_json_escapes('\\uD83D alone')
        # Lone surrogates are replaced by repair_surrogates → U+FFFD
        assert result[0] == '\ufffd'

    def test_surrogate_pair_with_escapes_between(self):
        """\\uD83D\\n\\uDE00 — the \\n breaks the pair, no combination."""
        result = decode_json_escapes('\\uD83D\\n\\uDE00')
        # Should decode \\n to newline, not combine surrogates
        assert '\\u' not in result
        assert '\n' in result


# ─── Markdown code fence extraction (kills ADD_TO_SUB/SUB_TO_ADD in parse_raw_tool_call) ──

class TestMarkdownFenceExact:
    """Test exact match_str from markdown code fence extraction.

    Kills: ADD_TO_SUB @ pos 7567 (inner_start+len → inner_start-len),
           SUB_TO_ADD @ pos 7664 (closing != -1 → closing != 1),
           ADD_TO_SUB @ pos 7727 (closing+3 → closing-3).
    """

    def test_match_str_includes_closing_fence(self, monkeypatch):
        """match_str should include the closing ``` (closing+3)."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}\n```'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["match_str"].endswith('```')

    def test_match_str_exact_full_block(self, monkeypatch):
        """match_str should be the exact full markdown block."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}\n```'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["match_str"] == content

    def test_match_str_without_closing_fence(self, monkeypatch):
        """When there's no closing ```, match_str should end at the JSON, not include closing+3."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
        result = parse_raw_tool_call(content)
        assert result is not None
        # No closing ``` → match_str should NOT end with ```
        assert not result["match_str"].endswith('```')

    def test_match_str_closing_plus_3_exact(self, monkeypatch):
        """Verify closing+3 captures exactly the closing fence, not more."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}\n```\nextra text'
        result = parse_raw_tool_call(content)
        assert result is not None
        # match_str should end at ``` not include 'extra text'
        assert result["match_str"].endswith('```')
        assert 'extra text' not in result["match_str"]

    def test_no_closing_fence_does_not_crash(self, monkeypatch):
        """When closing ``` is not found (closing=-1), should use end_pos path.
        Mutant (closing != 1) would enter the block with closing=-1 and
        try clean[md_match.start():-1+3] = clean[...:2], wrong match_str."""
        monkeypatch.setattr("src.agent.parser.get_available_tools", lambda: set())
        content = '```json\n{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
        result = parse_raw_tool_call(content)
        assert result is not None
        assert result["parsed"]["name"] == "read_file"
        # match_str should contain the JSON, not a truncated 2-char string
        assert len(result["match_str"]) > 10