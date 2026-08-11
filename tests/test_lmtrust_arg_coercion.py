"""Deep mutation-killing tests for arg_coercion edge cases.

Targets blind spots not covered by test_arg_coercion.py: case-insensitive
bool strings, multilingual values, float-to-bool, scientific notation,
empty/whitespace arrays, non-string wrapping, _json key detection, and
coerce_arguments/coerce_and_log integration paths.

Each test asserts exact values AND the changed flag to kill mutations that
only affect one of the two.
"""

import json
import logging

import pytest

from src.agent import arg_coercion
from src.agent.arg_coercion import (
    _to_array,
    _to_bool,
    _to_number,
    _to_object,
    _to_string,
    coerce_and_log,
    coerce_arguments,
)


def _make_schema(monkeypatch, tool_name, types_dict):
    """Patch _schema_types so *tool_name* appears to declare *types_dict*."""

    def fake(name):
        return types_dict if name == tool_name else {}

    monkeypatch.setattr(arg_coercion, "_schema_types", fake)


# ---------------------------------------------------------------------------
# _to_bool
# ---------------------------------------------------------------------------

class TestToBoolDeep:
    """Edge cases for _to_bool: case-insensitive strings, multilingual
    values, numeric-to-bool, and passthrough for unrecognised types."""

    def test_yes_uppercase(self):
        assert _to_bool("YES") == (True, True)

    def test_yes_mixed_case(self):
        assert _to_bool("Yes") == (True, True)

    def test_yes_with_whitespace(self):
        assert _to_bool("  yes  ") == (True, True)

    def test_on_string_to_true(self):
        assert _to_bool("on") == (True, True)

    def test_off_string_to_false(self):
        assert _to_bool("off") == (False, True)

    def test_y_string_to_true(self):
        assert _to_bool("y") == (True, True)

    def test_n_string_to_false(self):
        assert _to_bool("n") == (False, True)

    def test_russian_da_to_true(self):
        assert _to_bool("да") == (True, True)

    def test_russian_net_to_false(self):
        assert _to_bool("нет") == (False, True)

    def test_float_1_to_bool(self):
        result, changed = _to_bool(1.0)
        assert result is True
        assert changed is True

    def test_float_0_to_bool(self):
        result, changed = _to_bool(0.0)
        assert result is False
        assert changed is True

    def test_int_1_to_bool(self):
        result, changed = _to_bool(1)
        assert result is True
        assert changed is True

    def test_int_0_to_bool(self):
        result, changed = _to_bool(0)
        assert result is False
        assert changed is True

    def test_empty_string_passthrough(self):
        assert _to_bool("") == ("", False)

    def test_whitespace_only_string_passthrough(self):
        assert _to_bool("   ") == ("   ", False)

    def test_bool_true_unchanged(self):
        assert _to_bool(True) == (True, False)

    def test_bool_false_unchanged(self):
        assert _to_bool(False) == (False, False)

    def test_none_passthrough(self):
        assert _to_bool(None) == (None, False)

    def test_list_passthrough(self):
        assert _to_bool([1, 2]) == ([1, 2], False)

    def test_dict_passthrough(self):
        assert _to_bool({"a": 1}) == ({"a": 1}, False)


# ---------------------------------------------------------------------------
# _to_number
# ---------------------------------------------------------------------------

class TestToNumberDeep:
    """Edge cases for _to_number: float strings for int fields, scientific
    notation, bool passthrough, float-to-int, and type preservation."""

    def test_string_5_point_0_for_int(self):
        result, changed = _to_number("5.0", True)
        assert result == 5
        assert isinstance(result, int)
        assert changed is True

    def test_string_5_point_5_for_int_passthrough(self):
        assert _to_number("5.5", True) == ("5.5", False)

    def test_string_negative_zero_for_int(self):
        result, changed = _to_number("-0", True)
        assert result == 0
        assert isinstance(result, int)
        assert changed is True

    def test_string_0_for_number(self):
        result, changed = _to_number("0", False)
        assert result == 0.0
        assert isinstance(result, float)
        assert changed is True

    def test_string_5_for_number_becomes_float(self):
        result, changed = _to_number("5", False)
        assert result == 5.0
        assert isinstance(result, float)
        assert changed is True

    def test_scientific_notation_for_int(self):
        result, changed = _to_number("1e5", True)
        assert result == 100000
        assert isinstance(result, int)
        assert changed is True

    def test_scientific_notation_for_number(self):
        result, changed = _to_number("1e5", False)
        assert result == 100000.0
        assert isinstance(result, float)
        assert changed is True

    def test_bool_true_passthrough_for_int(self):
        result, changed = _to_number(True, True)
        assert result is True
        assert changed is False

    def test_bool_false_passthrough_for_int(self):
        result, changed = _to_number(False, True)
        assert result is False
        assert changed is False

    def test_bool_true_passthrough_for_number(self):
        result, changed = _to_number(True, False)
        assert result is True
        assert changed is False

    def test_float_to_int_when_is_integer(self):
        result, changed = _to_number(5.0, True)
        assert result == 5
        assert isinstance(result, int)
        assert changed is True

    def test_float_5_point_5_passthrough_for_int(self):
        result, changed = _to_number(5.5, True)
        assert result == 5.5
        assert changed is False

    def test_int_passthrough_for_number(self):
        result, changed = _to_number(5, False)
        assert result == 5
        assert isinstance(result, int)
        assert changed is False

    def test_float_passthrough_for_number(self):
        result, changed = _to_number(5.5, False)
        assert result == 5.5
        assert changed is False

    def test_int_passthrough_for_int(self):
        result, changed = _to_number(5, True)
        assert result == 5
        assert changed is False

    def test_string_with_whitespace_for_int(self):
        result, changed = _to_number("  42  ", True)
        assert result == 42
        assert changed is True

    def test_string_nonsense_passthrough(self):
        assert _to_number("abc", True) == ("abc", False)

    def test_string_negative_for_number(self):
        result, changed = _to_number("-3.14", False)
        assert result == -3.14
        assert changed is True

    def test_none_passthrough(self):
        assert _to_number(None, True) == (None, False)


# ---------------------------------------------------------------------------
# _to_array
# ---------------------------------------------------------------------------

class TestToArrayDeep:
    """Edge cases for _to_array: empty JSON, numeric JSON arrays,
    whitespace-only strings, non-string wrapping, and tuple conversion."""

    def test_empty_json_array(self):
        assert _to_array("[]") == ([], True)

    def test_json_array_with_numbers(self):
        assert _to_array("[1, 2]") == ([1, 2], True)

    def test_json_array_with_whitespace(self):
        assert _to_array("  [1, 2]  ") == ([1, 2], True)

    def test_empty_lines_passthrough(self):
        result, changed = _to_array("\n\n")
        assert result == "\n\n"
        assert changed is False

    def test_whitespace_only_passthrough(self):
        result, changed = _to_array("   ")
        assert result == "   "
        assert changed is False

    def test_int_to_single_item_list(self):
        assert _to_array(42) == ([42], True)

    def test_float_to_single_item_list(self):
        assert _to_array(3.14) == ([3.14], True)

    def test_bool_to_single_item_list(self):
        result, changed = _to_array(True)
        assert result == [True]
        assert changed is True

    def test_dict_to_single_item_list(self):
        assert _to_array({"a": 1}) == ([{"a": 1}], True)

    def test_tuple_to_list(self):
        result, changed = _to_array((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)
        assert changed is True

    def test_empty_tuple_to_list(self):
        result, changed = _to_array(())
        assert result == []
        assert changed is True

    def test_single_string_to_list(self):
        assert _to_array("single") == (["single"], True)

    def test_list_unchanged_identity(self):
        val = [1, 2]
        result, changed = _to_array(val)
        assert result is val
        assert changed is False

    def test_none_passthrough(self):
        assert _to_array(None) == (None, False)

    def test_set_passthrough(self):
        val = {1, 2}
        result, changed = _to_array(val)
        assert result is val
        assert changed is False


# ---------------------------------------------------------------------------
# _to_object
# ---------------------------------------------------------------------------

class TestToObjectDeep:
    """Edge cases for _to_object: valid/invalid JSON, non-string
    passthrough, dict identity, and JSON array string passthrough."""

    def test_valid_json_string_to_dict(self):
        result, changed = _to_object('{"a": 1}')
        assert result == {"a": 1}
        assert changed is True

    def test_valid_json_string_with_nested(self):
        result, changed = _to_object('{"a": {"b": 2}}')
        assert result == {"a": {"b": 2}}
        assert changed is True

    def test_invalid_json_passthrough(self):
        assert _to_object("not json") == ("not json", False)

    def test_non_string_passthrough_int(self):
        assert _to_object(42) == (42, False)

    def test_dict_unchanged_identity(self):
        val = {"a": 1}
        result, changed = _to_object(val)
        assert result is val
        assert changed is False

    def test_json_array_string_passthrough(self):
        assert _to_object("[1, 2]") == ("[1, 2]", False)

    def test_none_passthrough(self):
        assert _to_object(None) == (None, False)

    def test_list_passthrough(self):
        assert _to_object([1, 2]) == ([1, 2], False)

    def test_json_null_string_passthrough(self):
        assert _to_object("null") == ("null", False)

    def test_json_number_string_passthrough(self):
        assert _to_object("42") == ("42", False)


# ---------------------------------------------------------------------------
# _to_string
# ---------------------------------------------------------------------------

class TestToStringDeep:
    """Edge cases for _to_string: bool serialization, float formatting,
    _json key detection, ensure_ascii, and passthrough for non-serialisable."""

    def test_true_to_string(self):
        assert _to_string(True) == ("true", True)

    def test_false_to_string(self):
        assert _to_string(False) == ("false", True)

    def test_float_to_string(self):
        assert _to_string(3.14) == ("3.14", True)

    def test_int_to_string(self):
        assert _to_string(42) == ("42", True)

    def test_string_unchanged(self):
        assert _to_string("hello") == ("hello", False)

    def test_dict_with_json_key(self):
        result, changed = _to_string({"a": 1}, "data_json")
        assert result == '{"a": 1}'
        assert changed is True

    def test_list_with_json_key(self):
        result, changed = _to_string([1, 2], "items_json")
        assert result == "[1, 2]"
        assert changed is True

    def test_dict_without_json_key_passthrough(self):
        assert _to_string({"a": 1}, "content") == ({"a": 1}, False)

    def test_list_without_json_key_passthrough(self):
        assert _to_string([1, 2], "content") == ([1, 2], False)

    def test_none_passthrough(self):
        assert _to_string(None) == (None, False)

    def test_dict_with_unicode_json_key_ensure_ascii_false(self):
        """ensure_ascii=False keeps café readable, not \\u00e9."""
        result, changed = _to_string({"name": "café"}, "data_json")
        assert "café" in result
        assert "\\u" not in result
        assert changed is True

    def test_empty_key_does_not_trigger_json(self):
        """Default key='' must not end with '_json'."""
        assert _to_string({"a": 1}) == ({"a": 1}, False)

    def test_bool_not_treated_as_int(self):
        """bool is checked before int/float — True must become 'true' not '1'."""
        assert _to_string(True) == ("true", True)
        assert _to_string(False) == ("false", True)


# ---------------------------------------------------------------------------
# coerce_arguments
# ---------------------------------------------------------------------------

class TestCoerceArgumentsDeep:
    """Integration tests for coerce_arguments: unknown types, multiple
    conversions, non-dict input, and custom schema types via monkeypatch."""

    def test_unknown_type_passthrough(self, monkeypatch):
        _make_schema(monkeypatch, "custom_tool", {"data": "custom_type"})
        result, changed = coerce_arguments("custom_tool", {"data": "value"})
        assert result == {"data": "value"}
        assert changed == []

    def test_multiple_conversions_in_one_call(self, monkeypatch):
        _make_schema(monkeypatch, "multi_tool", {
            "flag": "boolean",
            "count": "integer",
            "items": "array",
        })
        result, changed = coerce_arguments("multi_tool", {
            "flag": "true",
            "count": "5",
            "items": "single",
        })
        assert result == {"flag": True, "count": 5, "items": ["single"]}
        assert len(changed) == 3

    def test_changed_entry_format(self, monkeypatch):
        _make_schema(monkeypatch, "bool_tool", {"flag": "boolean"})
        _, changed = coerce_arguments("bool_tool", {"flag": "true"})
        assert changed == ["flag='true'->True"]

    def test_changed_entries_for_multiple_conversions(self, monkeypatch):
        _make_schema(monkeypatch, "multi_tool", {
            "flag": "boolean",
            "count": "integer",
        })
        _, changed = coerce_arguments("multi_tool", {"flag": "true", "count": "5"})
        assert changed == ["flag='true'->True", "count='5'->5"]

    def test_non_dict_arguments_passthrough(self):
        result, changed = coerce_arguments("write_file", ["not", "a", "dict"])
        assert result == ["not", "a", "dict"]
        assert changed == []

    def test_no_schema_passthrough_identity(self):
        args = {"key": "value"}
        result, changed = coerce_arguments("nonexistent_tool", args)
        assert result is args
        assert changed == []

    def test_empty_dict_passthrough(self):
        result, changed = coerce_arguments("write_file", {})
        assert result == {}
        assert changed == []

    def test_number_type_conversion(self, monkeypatch):
        _make_schema(monkeypatch, "num_tool", {"value": "number"})
        result, changed = coerce_arguments("num_tool", {"value": "3.14"})
        assert result == {"value": 3.14}
        assert isinstance(result["value"], float)
        assert len(changed) == 1

    def test_object_type_conversion(self, monkeypatch):
        _make_schema(monkeypatch, "obj_tool", {"data": "object"})
        result, changed = coerce_arguments("obj_tool", {"data": '{"a": 1}'})
        assert result == {"data": {"a": 1}}
        assert len(changed) == 1

    def test_undeclared_parameter_passthrough(self, monkeypatch):
        _make_schema(monkeypatch, "some_tool", {"known": "boolean"})
        result, changed = coerce_arguments("some_tool", {
            "known": "true",
            "unknown": "value",
        })
        assert result == {"known": True, "unknown": "value"}
        assert len(changed) == 1

    def test_string_converter_receives_key_for_json(self, monkeypatch):
        """The string converter must get the key so _json detection works."""
        _make_schema(monkeypatch, "json_tool", {"payload_json": "string"})
        result, changed = coerce_arguments("json_tool", {"payload_json": [1, 2]})
        assert result == {"payload_json": "[1, 2]"}
        assert len(changed) == 1

    def test_none_arguments_passthrough(self):
        result, changed = coerce_arguments("write_file", None)
        assert result is None
        assert changed == []


# ---------------------------------------------------------------------------
# coerce_and_log
# ---------------------------------------------------------------------------

class TestCoerceAndLogDeep:
    """Integration tests for coerce_and_log: logging behavior, return
    values, and silence when nothing changes."""

    def test_logging_when_changes_occur(self, caplog):
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            result = coerce_and_log("write_file", {
                "file_path": "x", "content": "y", "overwrite": "false",
            })
        assert result["overwrite"] is False
        assert "coerced 1 argument(s) for write_file" in caplog.text
        assert "overwrite='false'->False" in caplog.text

    def test_no_logging_when_no_changes(self, caplog):
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            result = coerce_and_log("write_file", {
                "file_path": "x", "content": "y", "overwrite": False,
            })
        assert result == {"file_path": "x", "content": "y", "overwrite": False}
        assert "coerced" not in caplog.text

    def test_multiple_changes_logged_with_count(self, caplog, monkeypatch):
        _make_schema(monkeypatch, "multi_tool", {
            "flag": "boolean",
            "count": "integer",
        })
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            coerce_and_log("multi_tool", {"flag": "true", "count": "5"})
        assert "coerced 2 argument(s) for multi_tool" in caplog.text

    def test_return_value_is_coerced_dict(self, monkeypatch):
        _make_schema(monkeypatch, "obj_tool", {"data": "object"})
        result = coerce_and_log("obj_tool", {"data": '{"a": 1}'})
        assert result == {"data": {"a": 1}}

    def test_no_schema_silent(self, caplog):
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            coerce_and_log("nonexistent_tool", {"key": "value"})
        assert "coerced" not in caplog.text

    def test_empty_dict_silent(self, caplog):
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            coerce_and_log("write_file", {})
        assert "coerced" not in caplog.text

    def test_non_dict_silent(self, caplog):
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            coerce_and_log("write_file", None)
        assert "coerced" not in caplog.text
class TestNewlineArrayDidFlag:
    """Kill TRUE_TO_FALSE @ pos 3785: ``return lines, True`` → ``return lines, False``.

    When a newline-separated string is coerced to an array, the ``did`` flag
    must be ``True`` so that ``coerce_arguments`` logs the change.
    """

    def test_newline_string_did_is_true(self):
        """Direct _to_array call: did must be True for newline split."""
        val, did = _to_array("alpha\nbeta\ngamma")
        assert val == ["alpha", "beta", "gamma"]
        assert did is True

    def test_newline_string_changed_logged(self, monkeypatch):
        """Through coerce_arguments: changed list must include the coercion."""
        _make_schema(monkeypatch, "test_tool", {"items": "array"})
        coerced, changed = coerce_arguments("test_tool", {"items": "a\nb"})
        assert coerced["items"] == ["a", "b"]
        assert len(changed) == 1
        assert "items" in changed[0]