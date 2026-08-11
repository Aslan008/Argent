"""Blind-spot tests for tool_recovery.py.

Covers edge cases for _accepts_args, fuzzy_match_tool, recover_json_arguments,
and recover_tool_call that are easy to break during refactoring.
"""

import pytest

from tool_recovery import (
    _accepts_args,
    fuzzy_match_tool,
    recover_json_arguments,
    recover_tool_call,
)


# ---------------------------------------------------------------------------
# recover_json_arguments
# ---------------------------------------------------------------------------

class TestRecoverJsonEmpty:
    """L2×D1 — empty/None input returns empty dict."""

    def test_empty_string(self):
        assert recover_json_arguments('') == {}

    def test_none(self):
        # Code: 'if not raw_args: return {}'
        assert recover_json_arguments(None) == {}


class TestRecoverJsonAlreadyDict:
    """L1×D7 — dict input passes through unchanged."""

    def test_dict_passthrough(self):
        # Code: 'if isinstance(raw_args, dict): return raw_args'
        assert recover_json_arguments({'a': 1}) == {'a': 1}


class TestRecoverJsonValidJson:
    """L1×D7 — valid JSON string parses normally."""

    def test_valid_json(self):
        assert recover_json_arguments('{"key": "value"}') == {'key': 'value'}


class TestRecoverJsonSingleQuotes:
    """L4×D3 — single quotes are replaced with double quotes."""

    def test_single_quotes(self):
        result = recover_json_arguments("{'key': 'value'}")
        assert result == {'key': 'value'}


class TestRecoverJsonTrailingComma:
    """L4×D3 — trailing commas are stripped before parsing."""

    def test_trailing_comma(self):
        result = recover_json_arguments('{"a": 1, "b": 2,}')
        assert result == {'a': 1, 'b': 2}


class TestRecoverJsonEmbeddedInText:
    """L4×D3 — JSON embedded in prose is extracted via first { / last }."""

    def test_embedded(self):
        result = recover_json_arguments('Here is the tool call: {"file": "a.py"} done')
        assert result == {'file': 'a.py'}


class TestRecoverJsonRegexExtraction:
    """L4×D3 — non-JSON key:value pairs recovered via regex."""

    def test_regex_extraction(self):
        result = recover_json_arguments('file_path: "a.py" content: "hello"')
        # Regex extracts key:value pairs when JSON parsing fails entirely.
        assert result is not None
        assert 'file_path' in result


class TestRecoverJsonUnrecoverable:
    """L4×D4 — garbage with no recoverable structure returns None."""

    def test_unrecoverable(self):
        result = recover_json_arguments('totally broken no json here')
        assert result is None


# ---------------------------------------------------------------------------
# fuzzy_match_tool
# ---------------------------------------------------------------------------

class TestFuzzyExactMatch:
    """L1×D7 — exact name returns immediately."""

    def test_exact_match(self):
        assert fuzzy_match_tool('read_file', {'read_file': lambda **k: None}) == 'read_file'


class TestFuzzyNormalizedMatch:
    """L1×D7 — case/separator differences resolve to the real name."""

    def test_normalized_match(self):
        result = fuzzy_match_tool('Read-File', {'read_file': lambda **k: None})
        assert result == 'read_file'


class TestFuzzyNoMatch:
    """L1×D7 — unrelated name with no close neighbour returns None."""

    def test_no_match(self):
        assert fuzzy_match_tool('xyzzy', {'read_file': lambda **k: None}) is None


class TestFuzzyArgsRejection:
    """L4×D7 — signature-incompatible fuzzy candidates are skipped."""

    def test_args_rejection(self):
        def writer(file_path, content):
            pass

        def reader(file_path):
            pass

        available_tools = {'write_file': writer, 'read_file': reader}
        # 'write_fil' is close to both, but only 'write_file' accepts 'content'.
        result = fuzzy_match_tool('write_fil', available_tools, args={'content': 'x'})
        assert result == 'write_file'


# ---------------------------------------------------------------------------
# _accepts_args
# ---------------------------------------------------------------------------

class TestAcceptsKwargs:
    """L1×D7 — **kwargs accepts anything (VAR_KEYWORD check)."""

    def test_accepts_kwargs(self):
        def func(**kwargs):
            pass
        assert _accepts_args(func, {'anything': 1}) is True


class TestAcceptsNoArgs:
    """L1×D7 — no args means no signature check needed."""

    def test_no_args(self):
        def func():
            pass
        # Code: 'if not args: return True'
        assert _accepts_args(func, None) is True


# ---------------------------------------------------------------------------
# recover_tool_call
# ---------------------------------------------------------------------------

class TestRecoverToolCallDictArgs:
    """L1×D7 — dict arguments pass through unchanged."""

    def test_dict_args(self):
        raw = {'function': {'name': 'read_file', 'arguments': {'file_path': 'a.py'}}}
        result = recover_tool_call(raw, {'read_file': lambda **k: None})
        assert result is not None
        assert result['function']['arguments'] == {'file_path': 'a.py'}


class TestRecoverToolCallUnrecoverable:
    """L4×D4 — unknown name with no fuzzy match returns None."""

    def test_unrecoverable(self):
        raw = {'function': {'name': 'xyzzy', 'arguments': '{}'}}
        result = recover_tool_call(raw, {'read_file': lambda **k: None})
        assert result is None


class TestRecoverToolCallRawToolDirect:
    """L1×D7 — raw_tool without 'function' key uses itself as func."""

    def test_raw_tool_direct(self):
        raw = {'name': 'read_file', 'arguments': {}}
        result = recover_tool_call(raw, {'read_file': lambda **k: None})
        # Code: 'func = raw_tool.get('function', raw_tool)'
        assert result is not None
        assert result['function']['name'] == 'read_file'


class TestRecoverToolCallNonStrNonDictArgs:
    """L4×D7 — non-str/non-dict args fall through to empty dict."""

    def test_non_str_non_dict_args(self):
        raw = {'function': {'name': 'read_file', 'arguments': 12345}}
        result = recover_tool_call(raw, {'read_file': lambda **k: None})
        # Code: 'else: recovered_args = {}'
        assert result is not None
        assert result['function']['arguments'] == {}