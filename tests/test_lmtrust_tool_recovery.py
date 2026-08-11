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


class TestAcceptsSignatureError:
    """L4×D7 — when inspect.signature raises TypeError/ValueError, _accepts_args
    returns True (don't guess). Kills TRUE_TO_FALSE mutation on the exception handler."""

    def test_typeerror_returns_true(self, monkeypatch):
        import inspect as _inspect
        def boom(func):
            raise TypeError("no signature")
        monkeypatch.setattr(_inspect, "signature", boom)
        def func(a, b):
            pass
        # Code: 'except (TypeError, ValueError): return True'
        assert _accepts_args(func, {'x': 1}) is True

    def test_valueerror_returns_true(self, monkeypatch):
        import inspect as _inspect
        def boom(func):
            raise ValueError("no signature")
        monkeypatch.setattr(_inspect, "signature", boom)
        def func(a, b):
            pass
        assert _accepts_args(func, {'x': 1}) is True


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


# ---------------------------------------------------------------------------
# Fragment extraction (third try block) — kills mutations on the
# `start != -1 and end != -1 and end > start` condition and `text[start:end+1]` slice.
# Key insight: the regex fallback only captures string values `"([^"]*)"` —
# it silently drops numbers, booleans, and null. So fragment extraction with
# non-string values produces different results than the regex fallback.
# ---------------------------------------------------------------------------

class TestFragmentNumericValue:
    """Fragment extraction must capture numeric values that regex fallback misses.
    Kills: SUB_TO_ADD @ start!=-1 → start!=1 (when { is at position 1)."""

    def test_brace_at_position_1_with_number(self):
        # '{' at index 1 → start=1. Mutated `start != 1` is False → skips block.
        # Regex fallback misses numeric 'count' → returns {'name': 'test'} not full dict.
        result = recover_json_arguments('x{"count": 42, "name": "test"}')
        assert result == {'count': 42, 'name': 'test'}

    def test_brace_at_position_0_with_number(self):
        # Control: '{' at index 0 → start=0. Both original and mutated enter block.
        result = recover_json_arguments('{"count": 42, "name": "test"} extra')
        assert result == {'count': 42, 'name': 'test'}


class TestFragmentEmptyJson:
    """Empty JSON object extracted from surrounding text.
    Kills: SUB_TO_ADD @ end!=-1 → end!=1 (when } is at position 1)."""

    def test_empty_json_with_trailing_text(self):
        # '{}' at positions 0,1 → start=0, end=1. Mutated `end != 1` is False.
        # Fragment: json.loads('{}') → {}. Regex: no kv pairs → None.
        result = recover_json_arguments('{} extra text')
        assert result == {}


class TestFragmentBooleanValue:
    """Fragment extraction must capture boolean values that regex misses.
    Kills: GT_TO_LTE @ end>start → end<=start, NE_TO_EQ @ end!=-1 → end==-1."""

    def test_fragment_with_boolean(self):
        # Regex can't capture `true` (not in quotes) → would return {'name': 'test'} only.
        result = recover_json_arguments('prefix {"flag": true, "name": "test"} suffix')
        assert result == {'flag': True, 'name': 'test'}

    def test_fragment_with_null(self):
        # Regex can't capture `null` → would return {'name': 'test'} only.
        result = recover_json_arguments('prefix {"data": null, "name": "test"} suffix')
        assert result == {'data': None, 'name': 'test'}


class TestFragmentSliceEndPlusOne:
    """The slice text[start:end+1] must include the closing brace.
    Kills: ADD_TO_SUB @ end+1 → end-1, ONE_TO_ZERO @ end+1 → end+0."""

    def test_slice_includes_closing_brace_with_number(self):
        # Without +1: text[1:13] = '{"count": 42' → json.loads fails → regex → None.
        # With +1: text[1:14] = '{"count": 42}' → json.loads → {'count': 42}.
        result = recover_json_arguments('x{"count": 42}')
        assert result == {'count': 42}

    def test_slice_includes_closing_brace_with_boolean(self):
        result = recover_json_arguments('x{"flag": true}')
        assert result == {'flag': True}


class TestFuzzyRejectionLogging:
    """Verify that rejected fuzzy matches log the actual args (not empty).
    Kills: OR_TO_AND @ sorted(args or ()) → sorted(args and ())."""

    def test_rejection_log_shows_args(self, caplog):
        def writer(file_path, content):
            pass

        def reader(file_path):
            pass

        available_tools = {'write_file': writer, 'read_file': reader}
        # 'read_fil' is closest to 'read_file' (0.94), which doesn't accept 'content'.
        # read_file is rejected → log should show actual args.
        # write_file accepts 'content' → returned as fallback.
        import logging
        with caplog.at_level(logging.INFO, logger='tool_recovery'):
            result = fuzzy_match_tool('read_fil', available_tools, args={'content': 'x'})
        # Verify the rejection happened (read_file was skipped)
        rejection_logs = [r for r in caplog.records if 'Rejected' in r.getMessage()]
        assert rejection_logs, "Expected at least one rejection log"
        # The log message should contain the actual arg names, not an empty tuple.
        # Mutation: sorted(args or ()) → sorted(args and ()) → sorted(()) → '[]'
        assert 'content' in rejection_logs[0].getMessage()